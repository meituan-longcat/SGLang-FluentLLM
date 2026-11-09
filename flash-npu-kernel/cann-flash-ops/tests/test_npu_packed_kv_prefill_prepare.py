import pytest
import torch


torch_npu = pytest.importorskip("torch_npu")
try:
    import custom_ops  # noqa: F401
except ImportError:
    pytest.importorskip("flash_ops")


LATENT_WIDTH = 512
ROPE_WIDTH = 64
INPUT_WIDTH = LATENT_WIDTH + ROPE_WIDTH
PACKED_PAYLOAD_BYTES = 656
PACKED_ALIGNED_ROW_BYTES = 672


def _make_inputs(rows: int, owners: int):
    torch.manual_seed(20260907 + rows + owners)
    latent = torch.randn(
        rows, INPUT_WIDTH, dtype=torch.bfloat16, device="npu"
    )
    gamma = torch.randn(
        LATENT_WIDTH, dtype=torch.bfloat16, device="npu"
    )
    cos = torch.randn(rows, ROPE_WIDTH, dtype=torch.bfloat16, device="npu")
    sin = torch.randn(rows, ROPE_WIDTH, dtype=torch.bfloat16, device="npu")
    owner_rows = torch.arange(
        owners, dtype=torch.int64, device="npu"
    ).remainder(rows)
    return latent, gamma, cos, sin, owner_rows


def _run(inputs, packed_row_bytes=None):
    kwargs = {"epsilon": 1e-6}
    if packed_row_bytes is not None:
        kwargs["packed_row_bytes"] = packed_row_bytes
    outputs = torch.ops.custom.npu_packed_kv_prefill_prepare(
        *inputs, **kwargs
    )
    torch.npu.synchronize()
    return tuple(output.cpu() for output in outputs)


@pytest.mark.parametrize(
    "rows,owners", [(1, 1), (7, 0), (7, 4), (49, 49)]
)
def test_packed_kv_prefill_prepare_656_672_are_payload_exact(rows, owners):
    inputs = _make_inputs(rows, owners)
    key_656, rope_656, packed_656 = _run(inputs, PACKED_PAYLOAD_BYTES)
    key_672, rope_672, packed_672 = _run(
        inputs, PACKED_ALIGNED_ROW_BYTES
    )

    assert key_656.shape == key_672.shape == (rows, 1, LATENT_WIDTH)
    assert rope_656.shape == rope_672.shape == (rows, 1, ROPE_WIDTH)
    assert packed_656.shape == (owners, 1, PACKED_PAYLOAD_BYTES)
    assert packed_672.shape == (owners, 1, PACKED_ALIGNED_ROW_BYTES)
    torch.testing.assert_close(key_672, key_656, rtol=0, atol=0)
    torch.testing.assert_close(rope_672, rope_656, rtol=0, atol=0)
    torch.testing.assert_close(
        packed_672[..., :PACKED_PAYLOAD_BYTES], packed_656,
        rtol=0, atol=0,
    )
    assert torch.count_nonzero(
        packed_672[..., PACKED_PAYLOAD_BYTES:]
    ).item() == 0


def test_packed_kv_prefill_prepare_default_remains_656():
    outputs = _run(_make_inputs(3, 2))
    assert outputs[2].shape == (2, 1, PACKED_PAYLOAD_BYTES)


@pytest.mark.parametrize("packed_row_bytes", [0, 655, 657, 671, 673])
def test_packed_kv_prefill_prepare_rejects_invalid_row_bytes(
    packed_row_bytes,
):
    inputs = _make_inputs(1, 1)
    with pytest.raises(RuntimeError, match="must be 656 or 672"):
        torch.ops.custom.npu_packed_kv_prefill_prepare(
            *inputs, epsilon=1e-6, packed_row_bytes=packed_row_bytes
        )


@pytest.mark.parametrize(
    "packed_row_bytes", [PACKED_PAYLOAD_BYTES, PACKED_ALIGNED_ROW_BYTES]
)
def test_packed_kv_prefill_prepare_torchair_graph(packed_row_bytes):
    torchair = pytest.importorskip("torchair")

    class Model(torch.nn.Module):
        def __init__(self, row_bytes):
            super().__init__()
            self.row_bytes = row_bytes

        def forward(self, latent, gamma, cos, sin, owner_rows):
            return torch.ops.custom.npu_packed_kv_prefill_prepare(
                latent,
                gamma,
                cos,
                sin,
                owner_rows,
                epsilon=1e-6,
                packed_row_bytes=self.row_bytes,
            )

    inputs = _make_inputs(7, 4)
    expected = _run(inputs, packed_row_bytes)
    backend = torchair.get_npu_backend(
        compiler_config=torchair.CompilerConfig()
    )
    model = torch.compile(
        Model(packed_row_bytes).npu(), backend=backend, fullgraph=True
    )
    actual = model(*inputs)
    torch.npu.synchronize()
    actual = tuple(output.cpu() for output in actual)

    for actual_output, expected_output in zip(actual, expected):
        torch.testing.assert_close(
            actual_output, expected_output, rtol=0, atol=0
        )
