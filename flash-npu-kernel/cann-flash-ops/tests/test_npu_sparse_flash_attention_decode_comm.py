import pytest
import torch

import flash_ops  # noqa: F401


def make_inputs(valid_tokens, total_query=64):
    torch.manual_seed(20260830 + valid_tokens)
    heads = 64
    block_size = 128
    sparse_count = 2048
    blocks = sparse_count // block_size

    query = torch.randn(
        total_query, heads, 512, dtype=torch.bfloat16, device="npu"
    )
    query_rope = torch.randn(
        total_query, heads, 64, dtype=torch.bfloat16, device="npu"
    )
    key = torch.randn(
        blocks, block_size, 1, 512,
        dtype=torch.bfloat16, device="npu"
    )
    key_rope = torch.randn(
        blocks, block_size, 1, 64,
        dtype=torch.bfloat16, device="npu"
    )
    indices = torch.full(
        (total_query, 1, sparse_count), -1, dtype=torch.int32
    )
    selected = torch.arange(
        valid_tokens - 1, -1, -1, dtype=torch.int32
    )
    indices[:, 0, :valid_tokens] = selected

    return {
        "query": query,
        "key": key,
        "value": key,
        "sparse_indices": indices.npu(),
        "scale_value": 1.0,
        "valid_chunks": torch.full(
            (total_query,), (valid_tokens + block_size - 1) // block_size,
            dtype=torch.int32, device="npu"
        ),
        "block_table": torch.arange(
            blocks, dtype=torch.int32, device="npu"
        ).reshape(1, blocks),
        "actual_seq_lengths_query": torch.tensor(
            [total_query], dtype=torch.int32, device="npu"
        ),
        "actual_seq_lengths_kv": torch.tensor(
            [sparse_count], dtype=torch.int32, device="npu"
        ),
        "query_rope": query_rope,
        "key_rope": key_rope,
        "sparse_block_size": 1,
        "layout_kv": "PA_BSND",
        "sparse_mode": 0,
        "attention_mode": 2,
        "return_softmax_lse": True,
    }


def run(inputs, layout):
    return torch.ops.custom.npu_sparse_flash_attention_decode(
        layout_query=layout, **inputs
    )


def make_packed_inputs(inputs):
    packed = dict(inputs)
    query = torch.cat((inputs["query"], inputs["query_rope"]), dim=-1)
    packed["query"] = (
        query.reshape(query.shape[0], 8, 8, 576)
        .permute(1, 0, 2, 3)
        .contiguous()
    )
    packed["query_rope"] = None
    return packed


def make_int8_cache_inputs(inputs, packed_row_bytes=656):
    """Build the exact MLA packed-C8 row and its BF16 dequantized oracle."""
    assert packed_row_bytes in (656, 672)
    key = inputs["key"].cpu()
    key_rope = inputs["key_rope"].cpu().contiguous()
    key_tiles = key.float().reshape(*key.shape[:-1], 4, 128)
    scales = key_tiles.abs().amax(dim=-1, keepdim=True) / 127.0
    scales = torch.where(scales == 0, torch.ones_like(scales), scales)
    quant = torch.round(key_tiles / scales).clamp(-127, 127).to(torch.int8)
    dequant = (quant.float() * scales).reshape_as(key).to(torch.bfloat16)

    packed = torch.empty(
        *key.shape[:-1], packed_row_bytes, dtype=torch.int8
    )
    packed[..., :512] = quant.reshape_as(key)
    packed[..., 512:640] = key_rope.view(torch.int8)
    packed[..., 640:656] = scales.squeeze(-1).contiguous().view(torch.int8)
    if packed_row_bytes == 672:
        packed[..., 656:672] = 0

    int8_inputs = dict(inputs)
    int8_inputs["key"] = packed.npu()
    int8_inputs["value"] = int8_inputs["key"]
    int8_inputs["key_rope"] = None

    oracle_inputs = dict(inputs)
    oracle_inputs["key"] = dequant.npu()
    oracle_inputs["value"] = oracle_inputs["key"]
    return int8_inputs, oracle_inputs


@pytest.mark.parametrize("valid_tokens", (256, 2048))
@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_comm_layout_matches_tnd(valid_tokens):
    inputs = make_inputs(valid_tokens)
    old_out, old_max, old_sum = run(inputs, "TND")
    comm_out, comm_max, comm_sum = run(inputs, "TND_KVP_COMM")
    torch.npu.synchronize()

    assert comm_out.shape == (64, 64, 512)
    assert comm_max.shape == (1, 64, 64)
    assert comm_sum.shape == (1, 64, 64)
    torch.testing.assert_close(
        comm_out.cpu(), old_out.transpose(0, 1).contiguous().cpu(),
        rtol=0, atol=0
    )
    torch.testing.assert_close(
        comm_max.cpu(), old_max.cpu(),
        rtol=0, atol=0
    )
    torch.testing.assert_close(
        comm_sum.cpu(), old_sum.cpu(),
        rtol=0, atol=0
    )


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_comm_layout_keeps_non_square_lse_native():
    # T=64 and G=64 have the same shape after a transpose.  Use T=32 so the
    # eager/Meta allocation contract is checked in addition to tensor values.
    inputs = make_inputs(256, total_query=32)
    old_out, old_max, old_sum = run(inputs, "TND")
    comm_out, comm_max, comm_sum = run(inputs, "TND_KVP_COMM")
    torch.npu.synchronize()

    assert comm_out.shape == (64, 32, 512)
    assert comm_max.shape == old_max.shape == (1, 32, 64)
    assert comm_sum.shape == old_sum.shape == (1, 32, 64)
    torch.testing.assert_close(
        comm_out.cpu(), old_out.transpose(0, 1).contiguous().cpu(),
        rtol=0, atol=0
    )
    torch.testing.assert_close(comm_max.cpu(), old_max.cpu(), rtol=0, atol=0)
    torch.testing.assert_close(comm_sum.cpu(), old_sum.cpu(), rtol=0, atol=0)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_runtime_scheduler_supports_more_than_512_queries():
    full_scan_inputs = make_inputs(0, total_query=513)
    bounded_inputs = dict(full_scan_inputs)
    bounded_inputs["sparse_prep_mode"] = 1

    full_scan = run(full_scan_inputs, "TND")
    bounded = run(bounded_inputs, "TND")
    torch.npu.synchronize()

    for actual, expected in zip(bounded, full_scan):
        torch.testing.assert_close(
            actual.cpu(), expected.cpu(), rtol=0, atol=0, equal_nan=True
        )
    assert torch.count_nonzero(bounded[0]) == 0


@pytest.mark.parametrize("valid_tokens", (256, 2048))
@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_packed_comm_layout_matches_tnd(valid_tokens):
    inputs = make_inputs(valid_tokens)
    packed_inputs = make_packed_inputs(inputs)
    old_out, old_max, old_sum = run(inputs, "TND")
    comm_out, comm_max, comm_sum = run(
        packed_inputs, "RANK_MAJOR_TND_PACKED_KVP_COMM"
    )
    torch.npu.synchronize()

    torch.testing.assert_close(
        comm_out.cpu(), old_out.transpose(0, 1).contiguous().cpu(),
        rtol=0, atol=0
    )
    torch.testing.assert_close(
        comm_max.cpu(), old_max.cpu(),
        rtol=0, atol=0
    )
    torch.testing.assert_close(
        comm_sum.cpu(), old_sum.cpu(),
        rtol=0, atol=0
    )


@pytest.mark.parametrize("valid_tokens", (256, 2048))
@pytest.mark.parametrize("packed_row_bytes", (656, 672))
@pytest.mark.parametrize(
    "layout", ("TND_KVP_COMM", "RANK_MAJOR_TND_PACKED_KVP_COMM")
)
@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_int8_packed_cache_matches_exact_bf16_dequant(
    valid_tokens, packed_row_bytes, layout
):
    inputs = make_inputs(valid_tokens)
    int8_inputs, oracle_inputs = make_int8_cache_inputs(
        inputs, packed_row_bytes
    )
    if layout.startswith("RANK_MAJOR"):
        int8_inputs = make_packed_inputs(int8_inputs)
        oracle_inputs = make_packed_inputs(oracle_inputs)

    expected = run(oracle_inputs, layout)
    actual = run(int8_inputs, layout)
    torch.npu.synchronize()

    for actual_tensor, expected_tensor in zip(actual, expected):
        torch.testing.assert_close(
            actual_tensor.cpu(), expected_tensor.cpu(), rtol=0, atol=0
        )


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_packed_comm_layout_torchair_replay_is_stable():
    import torchair

    class Model(torch.nn.Module):
        def forward(
            self, query, key, sparse_indices, valid_chunks, block_table,
            actual_query, actual_kv, key_rope
        ):
            return torch.ops.custom.npu_sparse_flash_attention_decode(
                query, key, key, sparse_indices, 1.0,
                valid_chunks=valid_chunks,
                block_table=block_table,
                actual_seq_lengths_query=actual_query,
                actual_seq_lengths_kv=actual_kv,
                query_rope=None,
                key_rope=key_rope,
                sparse_block_size=1,
                layout_query="RANK_MAJOR_TND_PACKED_KVP_COMM",
                layout_kv="PA_BSND",
                sparse_mode=0,
                attention_mode=2,
                return_softmax_lse=True,
            )

    inputs = make_packed_inputs(make_inputs(256))
    args = (
        inputs["query"], inputs["key"], inputs["sparse_indices"],
        inputs["valid_chunks"], inputs["block_table"],
        inputs["actual_seq_lengths_query"],
        inputs["actual_seq_lengths_kv"], inputs["key_rope"],
    )
    expected = run(inputs, "RANK_MAJOR_TND_PACKED_KVP_COMM")
    backend = torchair.get_npu_backend(
        compiler_config=torchair.CompilerConfig()
    )
    model = torch.compile(Model().npu(), backend=backend, fullgraph=True)

    for _ in range(3):
        actual = model(*args)
        torch.npu.synchronize()
        for actual_tensor, expected_tensor in zip(actual, expected):
            torch.testing.assert_close(
                actual_tensor.cpu(), expected_tensor.cpu(), rtol=0, atol=0
            )
