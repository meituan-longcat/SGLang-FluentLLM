#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch_npu
import flash_npu_kernel
import pytest

if not hasattr(torch.ops.flash, "npu_update_oe_token_table"):
    pytest.skip(
        "flash.npu_update_oe_token_table not registered for current NPU_ARCH; skipping module",
        allow_module_level=True,
    )


def test_npu_update_oe_token_table_interface_exist():
    assert hasattr(torch.ops.flash, "npu_update_oe_token_table"), \
        "The 'npu_update_oe_token_table' operator is not registered in 'torch.ops.flash'."


def _ref_update_oe_token_table(tokens, row_indices, column_starts,
                               oe_req_lens, ignore_token, oe_token_table):
    if tokens.numel() != oe_req_lens.sum():
        raise ValueError(
            f"Token total {tokens.numel()} != sum(req_lens) {oe_req_lens.sum()}")
    device = oe_token_table.device
    rows_to_write = row_indices.repeat_interleave(oe_req_lens)
    col_starts_expanded = column_starts.repeat_interleave(oe_req_lens)
    offsets = torch.cat([
        torch.arange(length, device=device) for length in oe_req_lens
    ])
    cols_to_write = col_starts_expanded + offsets
    oe_token_table[rows_to_write, cols_to_write] = tokens
    mask = torch.isin(oe_token_table, ignore_token)
    oe_token_table[mask] = -1
    return oe_token_table


BS_VALUES = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    11, 12, 13, 14, 15, 16, 18, 20, 22, 24,
    26, 28, 30, 32, 36, 40, 44, 48, 52, 56,
    60, 64, 72, 80, 88, 96, 112, 128, 160, 192,
]
MAX_LENS = [20, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65535, 102400]


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("max_len", MAX_LENS)
@pytest.mark.parametrize("bs", BS_VALUES)
def test_update_oe_token_table(bs, max_len):
    seed = (bs * 149 + max_len * 17) % (2**31)
    torch.manual_seed(seed)

    max_reqs = int(bs + torch.randint(0, 100, (1,), dtype=torch.int32).item())

    oe_token_table = torch.zeros(max_reqs * max_len,
                                 dtype=torch.int32).reshape(max_reqs, max_len)
    row_indices = torch.randperm(bs, dtype=torch.int64)
    oe_req_lens = torch.randint(1, max_len, (bs,), dtype=torch.int32)
    start_max = max_len - oe_req_lens - 1
    column_starts = (torch.rand_like(start_max, dtype=torch.float32)
                     * (start_max + 1)).to(dtype=torch.int32)

    total_tokens = oe_req_lens.sum().item()
    tokens = torch.arange(1, total_tokens + 1, dtype=torch.int32)

    num_ignore_select = int(torch.randint(3, 10, (1,)).item())
    ignore_indices = torch.randperm(len(tokens))[:num_ignore_select]
    ignore_token = tokens[ignore_indices].contiguous()

    oe_token_table_npu = oe_token_table.clone().npu()

    expected = _ref_update_oe_token_table(
        tokens.clone(), row_indices.clone(), column_starts.clone(),
        oe_req_lens.clone(), ignore_token.clone(), oe_token_table.clone())

    torch.ops.flash.npu_update_oe_token_table(
        tokens.npu(), oe_req_lens.npu(), row_indices.npu(),
        column_starts.npu(), ignore_token.npu(),
        bs, max_len, oe_token_table_npu)
    torch.npu.synchronize()
    actual = oe_token_table_npu.cpu()

    assert torch.equal(actual, expected), (
        f"Mismatch for bs={bs}, max_len={max_len}: "
        f"diff_count={(actual != expected).sum().item()}"
    )
