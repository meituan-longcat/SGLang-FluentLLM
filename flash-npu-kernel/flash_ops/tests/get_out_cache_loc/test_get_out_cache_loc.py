#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch_npu
import numpy as np
import flash_npu_kernel
import pytest

if not hasattr(torch.ops.flash, "npu_get_out_cache_loc"):
    pytest.skip(
        "flash.npu_get_out_cache_loc not registered for current NPU_ARCH; skipping module",
        allow_module_level=True,
    )


def test_get_out_cache_loc_interface_exist():
    """Test that the operator is registered in torch.ops."""
    assert hasattr(torch.ops.flash, "npu_get_out_cache_loc"), \
        "The 'npu_get_out_cache_loc' operator is not registered in 'torch.ops.flash'."


def _ref_get_out_cache_loc(out_cache_loc, req_pool_indices, new_compute_lens,
                           cache_lens, req_to_token):
    """CPU reference implementation (matches the original test script)."""
    cumsum_offsets = torch.zeros_like(new_compute_lens)
    torch.cumsum(new_compute_lens[:-1], dim=0, out=cumsum_offsets[1:])
    cache_starts = cache_lens[req_pool_indices]
    row_indices = torch.repeat_interleave(req_pool_indices, new_compute_lens)
    total_new_compute = torch.sum(new_compute_lens)
    col_indices = (torch.repeat_interleave(cache_starts, new_compute_lens) +
                   torch.arange(total_new_compute,
                                dtype=new_compute_lens.dtype,
                                device=new_compute_lens.device) -
                   torch.repeat_interleave(cumsum_offsets, new_compute_lens))
    out_cache_loc[:col_indices.size(0)] = req_to_token[row_indices, col_indices]
    return out_cache_loc


# Mirror the original cann_m test grid: bs ∈ [1, 191], 13 pool_len values, 2 dtypes.
BS_VALUES = list(range(1, 192))
POOL_LENS = [16, 64, 128, 256, 1024, 2048, 4096, 6400, 8192, 16384, 32768, 65535, 131070]
DTYPES = [torch.int64, torch.int32]


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("pool_len", POOL_LENS)
@pytest.mark.parametrize("bs", BS_VALUES)
def test_get_out_cache_loc(bs, pool_len, dtype):
    """Compare NPU in-place op vs CPU reference across full bs/pool_len/dtype grid."""
    max_cache_len = pool_len // 4
    req_to_token = torch.arange(1, bs * pool_len + 1, dtype=dtype).reshape(bs, pool_len)
    torch.manual_seed(42)
    req_pool_indices = torch.randperm(bs, dtype=dtype)
    cache_lens = torch.randint(0, max_cache_len if max_cache_len > 0 else 1,
                               (bs,), dtype=dtype)
    upper = pool_len - max_cache_len
    new_compute_lens = torch.randint(1, upper if upper > 1 else 2,
                                     (bs,), dtype=dtype)

    total_tokens = int(new_compute_lens.sum().item())
    out_cache_loc = torch.zeros(total_tokens, dtype=dtype)
    out_cache_loc_npu = out_cache_loc.clone().npu()

    # CPU reference (mutates out_cache_loc in place).
    expected = _ref_get_out_cache_loc(
        out_cache_loc.clone(), req_pool_indices.clone(),
        new_compute_lens.clone(), cache_lens.clone(), req_to_token.clone(),
    )

    torch.ops.flash.npu_get_out_cache_loc(
        req_to_token.npu(), req_pool_indices.npu(),
        new_compute_lens.npu(), cache_lens.npu(),
        out_cache_loc_npu, bs,
    )
    torch.npu.synchronize()
    actual = out_cache_loc_npu.cpu()

    assert torch.equal(actual, expected), (
        f"Mismatch for bs={bs}, pool_len={pool_len}, dtype={dtype}: "
        f"diff_count={(actual != expected).sum().item()}"
    )
