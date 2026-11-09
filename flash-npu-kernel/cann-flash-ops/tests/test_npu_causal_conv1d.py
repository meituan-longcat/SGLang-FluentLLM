import pytest
import torch
import torch_npu

import flash_ops  # noqa: F401

pytestmark = pytest.mark.skipif(
    not torch.npu.is_available(), reason="NPU device not found"
)

# Packed QKV channels for Lite 3B/Large at TP1/2/4/8, deduplicated.
LITE_CHANNELS = sorted(
    {12288 // tp for tp in (1, 2, 4, 8)} | {24576 // tp for tp in (1, 2, 4, 8)}
)


def _reference(
    x, weight, state, starts, read_indices, write_indices, bias, initial_state=None
):
    output = torch.zeros_like(x)
    source_state = state.clone()
    updates = []
    channels = x.shape[-1]
    width = weight.shape[0]
    modes = (
        [True] * read_indices.numel()
        if initial_state is None
        else initial_state.tolist()
    )

    for request, (start, end) in enumerate(
        zip(starts[:-1].tolist(), starts[1:].tolist())
    ):
        read = int(read_indices[request])
        write = int(write_indices[request])
        if (
            start == end
            or not 0 <= read < state.size(0)
            or not 0 <= write < state.size(0)
        ):
            continue
        history = (
            source_state[read].transpose(0, 1).clone()
            if modes[request]
            else x.new_zeros((channels, width - 1))
        )
        values = []
        for token in x[start:end]:
            window = torch.cat((history, token.unsqueeze(-1)), dim=-1)
            value = (window.float() * weight.transpose(0, 1).float()).sum(dim=-1)
            if bias is not None:
                value = value + bias.float()
            values.append(torch.nn.functional.silu(value).to(x.dtype))
            history = window[:, 1:]
        output[start:end] = torch.stack(values)
        updates.append((write, history.transpose(0, 1)))

    for write, update in updates:
        state[write].copy_(update)
    return output


def _make_inputs(channels, batch=3, seed=0, dtype=torch.bfloat16):
    torch.manual_seed(20260904 + channels + seed)
    x = torch.randn(batch, channels, dtype=dtype, device="npu") * 0.05
    weight = torch.randn(4, channels, dtype=dtype, device="npu") * 0.03
    bias = torch.randn(channels, dtype=dtype, device="npu") * 0.02
    state = torch.randn(5, 3, channels, dtype=dtype, device="npu") * 0.02
    return x, weight, bias, state


def _assert_close(actual, expected):
    torch.testing.assert_close(
        actual.float().cpu(), expected.float().cpu(), rtol=0.03, atol=1e-4
    )


def _strided_pool(initial):
    slots, history, channels = initial.shape
    # Nonzero storage offset and unrelated data between consecutive state pages.
    stride = history * channels + 128 * channels + 32
    backing = torch.full(
        (slots * stride + 64,), 7, dtype=initial.dtype, device=initial.device
    )
    state = backing.as_strided(initial.shape, (stride, channels, 1), 32)
    state.copy_(initial)
    return backing, state


@pytest.mark.parametrize("channels", LITE_CHANNELS)
@pytest.mark.parametrize("decode", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_npu_causal_conv1d_strided_state(channels, decode, dtype):
    lengths = [1] * 5 if decode else [2, 1025, 5, 3, 2]
    x, weight, bias, initial = _make_inputs(channels, batch=sum(lengths), dtype=dtype)
    starts = torch.tensor([0] + lengths, dtype=torch.int32).cumsum(0).int()
    index_dtype = torch.int32 if dtype == torch.bfloat16 else torch.int64
    # Invalid read, invalid write, and out-of-range write independently skip work.
    reads = torch.tensor([0, 1, -1, 0, 0], dtype=index_dtype, device="npu")
    writes = torch.tensor([2, 3, 4, -1, 5], dtype=index_dtype, device="npu")
    modes = torch.tensor(
        [True, False, True, True, True], dtype=torch.bool, device="npu"
    )
    backing, state = _strided_pool(initial)
    before = backing.cpu()
    expected_state = initial.cpu()
    expected = _reference(
        x.cpu(),
        weight.cpu(),
        expected_state,
        starts,
        reads.cpu(),
        writes.cpu(),
        bias.cpu(),
        None if decode else modes.cpu(),
    )
    expected_backing = before.clone()
    expected_backing.as_strided(
        state.shape, state.stride(), state.storage_offset()
    ).copy_(expected_state)
    kwargs = (
        {} if decode else {"query_start_loc": starts.npu(), "initial_state_mode": modes}
    )
    actual = torch.ops.custom.npu_causal_conv1d(
        x,
        weight,
        state,
        bias=bias,
        cache_indices=reads,
        write_indices=writes,
        run_mode=int(decode),
        **kwargs
    )
    torch.npu.synchronize()
    _assert_close(actual, expected)
    assert torch.equal(
        backing.cpu(), expected_backing
    ), "state mutation or adjacent arena bytes differ"


@pytest.mark.parametrize("decode", [False, True])
def test_npu_causal_conv1d_padding_native_graph_replay(decode):
    x, weight, bias, initial = _make_inputs(1536)
    backing, state = _strided_pool(initial)
    reads = torch.tensor([0, 1, -1], dtype=torch.int32, device="npu")
    writes = torch.tensor([2, 3, 4], dtype=torch.int32, device="npu")
    starts = torch.arange(4, dtype=torch.int32)
    kwargs = (
        {}
        if decode
        else {
            "query_start_loc": starts.npu(),
            "initial_state_mode": torch.ones(3, dtype=torch.bool, device="npu"),
        }
    )

    def call():
        return torch.ops.custom.npu_causal_conv1d(
            x,
            weight,
            state,
            bias=bias,
            cache_indices=reads,
            write_indices=writes,
            run_mode=int(decode),
            **kwargs
        )

    call()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        output = call()
    for read_pages, write_pages in (
        ([0, 1, -1], [2, 3, 4]),
        ([0, -1, 1], [-1, 3, 4]),
    ):
        x.add_(0.01)
        state.copy_(initial)
        reads.copy_(torch.tensor(read_pages, dtype=reads.dtype, device="npu"))
        writes.copy_(torch.tensor(write_pages, dtype=writes.dtype, device="npu"))
        expected_backing = backing.cpu()
        expected_state = expected_backing.as_strided(
            state.shape, state.stride(), state.storage_offset()
        )
        expected = _reference(
            x.cpu(),
            weight.cpu(),
            expected_state,
            starts,
            reads.cpu(),
            writes.cpu(),
            bias.cpu(),
        )
        graph.replay()
        torch.npu.synchronize()
        _assert_close(output, expected)
        assert torch.equal(backing.cpu(), expected_backing)


def test_npu_causal_conv1d_rejects_inner_stride():
    x, weight, bias, initial = _make_inputs(1536)
    state = torch.empty(5, 3, 3072, dtype=initial.dtype, device="npu")[:, :, ::2]
    with pytest.raises(RuntimeError, match="requires strides"):
        torch.ops.custom.npu_causal_conv1d(x, weight, state, run_mode=1)


@pytest.mark.parametrize("channels", LITE_CHANNELS)
def test_npu_causal_conv1d_decode_lite_channels(channels):
    x, weight, bias, initial = _make_inputs(channels)
    read_indices = torch.tensor([0, 1, -1], dtype=torch.int32, device="npu")
    write_indices = torch.tensor([2, 3, -1], dtype=torch.int32, device="npu")
    starts = torch.arange(4, dtype=torch.int32)

    expected_state = initial.cpu().clone()
    expected = _reference(
        x.cpu(),
        weight.cpu(),
        expected_state,
        starts,
        read_indices.cpu(),
        write_indices.cpu(),
        bias.cpu(),
    )
    actual_state = initial.clone()
    actual = torch.ops.custom.npu_causal_conv1d(
        x,
        weight,
        actual_state,
        bias=bias,
        cache_indices=read_indices,
        write_indices=write_indices,
        run_mode=1,
    )
    torch.npu.synchronize()

    _assert_close(actual, expected)
    assert torch.equal(actual[2].cpu(), torch.zeros_like(actual[2].cpu()))
    assert torch.equal(actual_state.cpu(), expected_state)


def test_npu_causal_conv1d_prefill():
    channels = 1536
    x, weight, bias, initial = _make_inputs(channels, batch=7)
    starts = torch.tensor([0, 3, 7], dtype=torch.int32, device="npu")
    read_indices = torch.tensor([0, 1], dtype=torch.int32, device="npu")
    initial_state = torch.tensor([True, False], dtype=torch.bool, device="npu")

    expected_state = initial.cpu().clone()
    expected = _reference(
        x.cpu(),
        weight.cpu(),
        expected_state,
        starts.cpu(),
        read_indices.cpu(),
        read_indices.cpu(),
        bias.cpu(),
        initial_state.cpu(),
    )
    actual_state = initial.clone()
    actual = torch.ops.custom.npu_causal_conv1d(
        x,
        weight,
        actual_state,
        bias=bias,
        query_start_loc=starts,
        cache_indices=read_indices,
        initial_state_mode=initial_state,
        run_mode=0,
    )
    torch.npu.synchronize()

    _assert_close(actual, expected)
    assert torch.equal(actual_state.cpu(), expected_state)


@pytest.mark.parametrize("strided", [False, True])
def test_npu_causal_conv1d_decode_torchair_replay(strided):
    import torchair

    def call(x, weight, state, read_indices, write_indices, bias):
        return torch.ops.custom.npu_causal_conv1d(
            x,
            weight,
            state,
            bias=bias,
            cache_indices=read_indices,
            write_indices=write_indices,
            run_mode=1,
        )

    backend = torchair.get_npu_backend(compiler_config=torchair.CompilerConfig())
    compiled = torch.compile(call, backend=backend, fullgraph=True, dynamic=False)

    for replay, (reads, writes) in enumerate(
        (([0, 1, -1], [2, 3, -1]), ([1, -1, 0], [3, -1, 2]))
    ):
        x, weight, bias, initial = _make_inputs(1536, seed=replay)
        read_indices = torch.tensor(reads, dtype=torch.int64, device="npu")
        write_indices = torch.tensor(writes, dtype=torch.int64, device="npu")
        expected_state = initial.cpu().clone()
        expected = _reference(
            x.cpu(),
            weight.cpu(),
            expected_state,
            torch.arange(4, dtype=torch.int32),
            read_indices.cpu(),
            write_indices.cpu(),
            bias.cpu(),
        )
        if strided:
            backing, actual_state = _strided_pool(initial)
            expected_backing = backing.cpu()
            expected_backing.as_strided(
                actual_state.shape, actual_state.stride(), actual_state.storage_offset()
            ).copy_(expected_state)
        else:
            actual_state = initial.clone()
        if strided:
            before = backing.cpu()
            with pytest.raises(
                Exception, match="strided state requires eager or native NPUGraph"
            ):
                compiled(x, weight, actual_state, read_indices, write_indices, bias)
            assert torch.equal(backing.cpu(), before)
            return
        actual = compiled(x, weight, actual_state, read_indices, write_indices, bias)
        torch.npu.synchronize()

        _assert_close(actual, expected)
        assert torch.equal(actual_state.cpu(), expected_state)
        if strided:
            assert torch.equal(backing.cpu(), expected_backing)
