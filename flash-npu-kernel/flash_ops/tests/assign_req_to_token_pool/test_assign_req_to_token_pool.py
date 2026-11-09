#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch_npu
import numpy as np
import flash_npu_kernel
import pytest

if not hasattr(torch.ops.flash, "npu_assign_req_to_token_pool"):
    pytest.skip(
        "flash.npu_assign_req_to_token_pool not registered for current NPU_ARCH; skipping module",
        allow_module_level=True,
    )


def test_npu_assign_req_to_token_pool_interface_exist():
    assert hasattr(torch.ops.flash, "npu_assign_req_to_token_pool"), \
        "The 'npu_assign_req_to_token_pool' operator is not registered in 'torch.ops.flash'."


def _ref_assign_req_to_token_pool(extend_lens, out_cache_loc, req_to_token,
                                  alloced_lens, req_pool_indices):
    pt_offsets = torch.zeros_like(extend_lens)
    torch.cumsum(extend_lens[:-1], dim=0, out=pt_offsets[1:])
    current_allocs = alloced_lens[req_pool_indices]
    row_indices = torch.repeat_interleave(req_pool_indices, extend_lens)
    col_offsets = torch.arange(extend_lens.sum(),
                               dtype=extend_lens.dtype,
                               device=extend_lens.device) - \
                  torch.repeat_interleave(pt_offsets, extend_lens)
    col_indices = torch.repeat_interleave(current_allocs, extend_lens) + col_offsets
    req_to_token.index_put_((row_indices, col_indices),
                            out_cache_loc[:extend_lens.sum()])
    return req_to_token


BS_VALUES = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    12, 14, 16, 20, 24, 28, 32,
    40, 48, 56, 64, 80, 96, 112, 128,
    144, 160, 176, 192, 208, 224, 240, 256, 384, 512,
]
ALLOC_SIZES = [16, 64, 256, 512, 1024, 2048, 4096, 8192, 32768, 131070]
DTYPES = [torch.int64, torch.int32]


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("alloc_size", ALLOC_SIZES)
@pytest.mark.parametrize("bs", BS_VALUES)
def test_assign_req_to_token_pool(bs, alloc_size, dtype):
    seed = (bs * 131 + alloc_size * 7 + hash(str(dtype))) % (2**31)
    np.random.seed(seed)
    torch.manual_seed(seed)

    pool_len = alloc_size * 3
    pool_size = int(np.random.randint(bs, bs * 2 if bs > 1 else bs + 1))

    req_to_token = torch.zeros((pool_size, pool_len), dtype=dtype)
    extend_lens = torch.randint(1, alloc_size, (bs,), dtype=dtype) \
        if alloc_size > 1 else torch.ones((bs,), dtype=dtype)
    out_cache_loc = torch.arange(1, extend_lens.sum().item() + 1, dtype=dtype)
    req_pool_indices = torch.randperm(bs, dtype=dtype)
    alloced_lens = torch.randint(alloc_size, 2 * alloc_size, (bs,), dtype=dtype)

    req_to_token_npu = req_to_token.clone().npu()

    expected = _ref_assign_req_to_token_pool(
        extend_lens.clone(), out_cache_loc.clone(), req_to_token.clone(),
        alloced_lens.clone(), req_pool_indices.clone(),
    )

    torch.ops.flash.npu_assign_req_to_token_pool(
        req_pool_indices.npu(), extend_lens.npu(),
        alloced_lens.npu(), out_cache_loc.npu(),
        req_to_token_npu, bs,
    )
    torch.npu.synchronize()
    actual = req_to_token_npu.cpu()

    assert torch.equal(actual, expected), (
        f"Mismatch for bs={bs}, alloc_size={alloc_size}, dtype={dtype}: "
        f"diff_count={(actual != expected).sum().item()}"
    )
