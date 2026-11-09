import pytest
import torch
import torch_npu

import flash_ops  # noqa: F401


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
@pytest.mark.parametrize("start", (0, 3072, 7168))
def test_npu_router_input_slice_cast_matches_slice_and_cast(dtype, start):
    torch.manual_seed(20260826)
    length = 1024
    x = torch.randn(64, 8192, dtype=dtype, device="npu")

    expected = x[:, start : start + length].contiguous().float()
    actual = torch.ops.custom.npu_router_input_slice_cast(x, start, length)
    torch.npu.synchronize()

    assert torch.equal(actual.cpu(), expected.cpu())
