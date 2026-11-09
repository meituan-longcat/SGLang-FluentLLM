import pytest
import torch
import torch_npu

import flash_ops  # noqa: F401


@pytest.mark.parametrize(
    "dtype,width,index_dtype",
    (
        (torch.float16, 128, torch.int32),
        (torch.float16, 128, torch.int64),
        (torch.bfloat16, 128, torch.int32),
        (torch.bfloat16, 128, torch.int64),
        (torch.int8, 640, torch.int32),
        (torch.int8, 640, torch.int64),
        (torch.int8, 656, torch.int32),
        (torch.int8, 656, torch.int64),
        (torch.int8, 672, torch.int32),
        (torch.int8, 672, torch.int64),
    ),
)
def test_kv_cache_copy_rows_functional_and_inplace(
    dtype, width, index_dtype
):
    cache_cpu = torch.arange(512 * width, dtype=torch.float32).reshape(512, width).to(dtype)
    src_cpu = torch.tensor([0, 1, 2, 64, 65, 66], dtype=index_dtype)
    dst_cpu = torch.tensor([1, 2, 0, 65, 66, 64], dtype=index_dtype)
    expected = cache_cpu.clone()
    expected[dst_cpu] = cache_cpu[src_cpu]

    functional = torch.ops.custom.npu_kv_cache_copy_rows(
        cache_cpu.npu(), src_cpu.npu(), dst_cpu.npu())
    torch.npu.synchronize()
    assert torch.equal(functional.cpu(), expected)

    inplace = cache_cpu.npu()
    returned = torch.ops.custom.npu_kv_cache_copy_rows_(inplace, src_cpu.npu(), dst_cpu.npu())
    torch.npu.synchronize()
    assert returned.data_ptr() == inplace.data_ptr()
    assert torch.equal(inplace.cpu(), expected)


@pytest.mark.parametrize("index_dtype", (torch.int32, torch.int64))
def test_kv_cache_copy_rows_skips_out_of_range_indices(index_dtype):
    cache_cpu = torch.arange(32 * 656, dtype=torch.int32).reshape(32, 656).to(torch.int8)
    src_cpu = torch.tensor([2, -1, 32, 5], dtype=index_dtype)
    dst_cpu = torch.tensor([10, 11, 12, 32], dtype=index_dtype)
    expected = cache_cpu.clone()
    expected[10] = cache_cpu[2]

    actual = cache_cpu.npu()
    torch.ops.custom.npu_kv_cache_copy_rows_(
        actual, src_cpu.npu(), dst_cpu.npu()
    )
    torch.npu.synchronize()
    assert torch.equal(actual.cpu(), expected)


def test_kv_cache_copy_rows_rejects_unaligned_bf16_rows():
    cache = torch.zeros(32, 127, dtype=torch.bfloat16, device="npu")
    indices = torch.tensor([0], dtype=torch.int64, device="npu")
    with pytest.raises(RuntimeError, match="only for int8"):
        torch.ops.custom.npu_kv_cache_copy_rows_(cache, indices, indices)
