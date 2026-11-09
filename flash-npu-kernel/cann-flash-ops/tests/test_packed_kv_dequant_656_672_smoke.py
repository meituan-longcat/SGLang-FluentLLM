#!/usr/bin/env python3
"""NPU smoke for the dual-stride PackedKvDequant ABI."""

import pytest
import torch
import torch_npu  # noqa: F401 - registers device/operator implementations

# flash_ops links against libraries loaded by torch/torch_npu.
import flash_ops  # noqa: F401  # isort: skip


def make_rows(rows: int, row_bytes: int):
    quant = torch.randint(-127, 128, (rows, 512), dtype=torch.int8)
    rope = torch.randn((rows, 64), dtype=torch.bfloat16)
    scale = torch.rand((rows, 4), dtype=torch.float32) * 0.25 + 0.001
    payload = torch.cat((quant, rope.view(torch.int8), scale.view(torch.int8)), dim=-1)
    assert payload.shape == (rows, 656)
    packed = torch.zeros((rows, row_bytes), dtype=torch.int8)
    packed[:, :656] = payload
    if row_bytes > 656:
        # Padding is not part of the ABI payload.  Keep it deliberately dirty
        # so the test catches kernels that accidentally consume the aligned
        # tail rather than merely using it as the next-row stride.
        packed[:, 656:] = torch.randint(-128, 128, (rows, row_bytes - 656), dtype=torch.int8)
    reference_key = (quant.reshape(rows, 4, 128).float() * scale.unsqueeze(-1)).to(torch.bfloat16).reshape(rows, 1, 512)
    return packed.reshape(rows, 1, row_bytes), reference_key, rope.reshape(rows, 1, 64)


@pytest.mark.parametrize("rows", [1, 49, 4096])
def test_packed_kv_dequant_dual_stride(rows: int) -> None:
    results = {}
    for row_bytes in (656, 672):
        packed, reference_key, reference_rope = make_rows(rows, row_bytes)
        key, rope = torch.ops.custom.npu_packed_kv_dequant(packed.npu())
        torch.npu.synchronize()
        key_cpu = key.cpu()
        rope_cpu = rope.cpu()
        torch.testing.assert_close(key_cpu, reference_key, rtol=0, atol=0)
        torch.testing.assert_close(rope_cpu, reference_rope, rtol=0, atol=0)
        results[row_bytes] = (packed[:, :, :656].clone(), key_cpu, rope_cpu)

    # Re-run the exact 656-byte payload with a 672-byte physical stride. This
    # catches a kernel that accepts the shape but still advances rows by 656.
    payload, key_656, rope_656 = results[656]
    aligned = torch.zeros((rows, 1, 672), dtype=torch.int8)
    aligned[:, :, :656] = payload
    aligned[:, :, 656:] = torch.randint(-128, 128, (rows, 1, 16), dtype=torch.int8)
    key_672, rope_672 = torch.ops.custom.npu_packed_kv_dequant(aligned.npu())
    torch.npu.synchronize()
    torch.testing.assert_close(key_672.cpu(), key_656, rtol=0, atol=0)
    torch.testing.assert_close(rope_672.cpu(), rope_656, rtol=0, atol=0)
    print(f"PASS rows={rows}: 656/672 are bit-exact")


@pytest.mark.parametrize("row_bytes", [656, 672])
@pytest.mark.parametrize("rows", [1, 49, 4096])
def test_packed_kv_dequant_out(rows: int, row_bytes: int) -> None:
    packed, reference_key, reference_rope = make_rows(rows, row_bytes)
    packed = packed.npu()
    key = torch.empty_like(reference_key, device="npu")
    rope = torch.empty_like(reference_rope, device="npu")
    pointers = (key.data_ptr(), rope.data_ptr())
    for _ in range(3):
        key.fill_(17)
        rope.fill_(23)
        assert torch.ops.custom.npu_packed_kv_dequant_out(packed, key, rope) is None
        torch.testing.assert_close(key.cpu(), reference_key, rtol=0, atol=0)
        torch.testing.assert_close(rope.cpu(), reference_rope, rtol=0, atol=0)
        assert (key.data_ptr(), rope.data_ptr()) == pointers


@pytest.mark.parametrize("row_bytes", [656, 672])
def test_packed_kv_dequant_out_rejects_invalid_outputs(row_bytes: int) -> None:
    rows = 17
    packed, _, _ = make_rows(rows, row_bytes)
    packed = packed.npu()
    key = torch.empty((rows, 1, 512), dtype=torch.bfloat16, device="npu")
    rope = torch.empty((rows, 1, 64), dtype=torch.bfloat16, device="npu")
    noncontiguous = torch.empty((rows, 1, 128), dtype=torch.bfloat16, device="npu")[:, :, ::2]
    overlapping = key.view(-1)[: rows * 64].view(rows, 1, 64)
    for args in (
        (packed, key.float(), rope),
        (packed, key[:1], rope),
        (packed, key, rope[:1]),
        (packed, key, noncontiguous),
        (packed, key, overlapping),
        (packed.cpu(), key, rope),
    ):
        with pytest.raises(RuntimeError):
            torch.ops.custom.npu_packed_kv_dequant_out(*args)


if __name__ == "__main__":
    torch.npu.set_device(0)
    for row_count in (1, 49, 4096):
        test_packed_kv_dequant_dual_stride(row_count)
        for row_bytes in (656, 672):
            test_packed_kv_dequant_out(row_count, row_bytes)
    for row_bytes in (656, 672):
        test_packed_kv_dequant_out_rejects_invalid_outputs(row_bytes)
