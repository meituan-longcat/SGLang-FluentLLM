#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch_npu
import flash_npu_kernel
import pytest

if not hasattr(torch.ops.flash, "npu_rearrange_accept_index"):
    pytest.skip(
        "flash.npu_rearrange_accept_index not registered for current NPU_ARCH; skipping module",
        allow_module_level=True,
    )


def test_rearrange_accept_index_interface_exist():
    assert hasattr(torch.ops.flash, "npu_rearrange_accept_index"), \
        "The 'npu_rearrange_accept_index' operator is not registered in 'torch.ops.flash'."


def _ref_rearrange_accept_index(accept_index, accept_length, bs, output_torch):
    range_tensor = torch.arange(accept_index.size(1),
                                device=accept_index.device).expand(bs, -1)
    expanded_length = torch.unsqueeze(accept_length, dim=1)
    mask = range_tensor < expanded_length
    valid_elements = accept_index[mask]
    output_torch[:valid_elements.size(0)] = valid_elements
    return output_torch


BS_VALUES = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    12, 14, 16, 20, 24, 28, 32,
    40, 48, 56, 64, 80, 96, 112, 128,
    144, 160, 176, 192, 208, 224, 240, 256, 384, 512,
]
POOL_SIZES = [16, 64, 256, 1024, 2048, 4096, 8192, 16384, 65535, 131070]
DTYPES = [torch.int64, torch.int32]


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("pool_size", POOL_SIZES)
@pytest.mark.parametrize("bs", BS_VALUES)
def test_rearrange_accept_index(bs, pool_size, dtype):
    seed = (bs * 139 + pool_size * 13 + hash(str(dtype))) % (2**31)
    torch.manual_seed(seed)

    accept_index = torch.arange(0, bs * pool_size, dtype=dtype,
                                device='cpu').reshape(bs, pool_size)
    accept_length = torch.randint(1, pool_size if pool_size > 1 else 2,
                                  (bs,), dtype=dtype)
    total_tokens = int(torch.sum(accept_length).item())

    output_cpu = torch.zeros((total_tokens,), dtype=dtype, device='cpu')
    output_npu = torch.zeros((total_tokens,), dtype=dtype).npu()

    expected = _ref_rearrange_accept_index(
        accept_index.clone(), accept_length.clone(), bs, output_cpu,
    )

    torch.ops.flash.npu_rearrange_accept_index(
        accept_index.npu(), accept_length.npu(), bs, output_npu,
    )
    torch.npu.synchronize()
    actual = output_npu.cpu()

    assert torch.equal(actual, expected), (
        f"Mismatch for bs={bs}, pool_size={pool_size}, dtype={dtype}: "
        f"diff_count={(actual != expected).sum().item()}"
    )
