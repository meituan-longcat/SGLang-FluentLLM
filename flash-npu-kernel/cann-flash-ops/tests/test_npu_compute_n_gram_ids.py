# This program is free software, you can redistribute it and/or modify it.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Precision tests for flash.npu_compute_n_gram_ids.

Production constraints reflected here (from actual caller code):
  - over_embedding_n (oe_n)       <= 7
  - over_embedding_k (oe_k)       <= 7
  - batch_size                    <  256
  - oe_token_table.shape[0]       =  256   (fixed request pool rows)
  - oe_token_table.shape[1]       <= 1M    (max_context_len / max_token_num)
  - sum(oe_req_lens) == input_len <= 65536
  - exclusive_oe_embeder_size_sums is built as:
        for i in range(n_grams):
            m = over_embedding_m + 2*i + 1
            exclusive[i+1] = exclusive[i] + m
"""

import numpy as np
import torch
import torch_npu
import flash_ops
import pytest

if not hasattr(torch.ops.custom, "npu_compute_n_gram_ids"):
    pytest.skip(
        "custom.npu_compute_n_gram_ids not registered for current NPU_ARCH; skipping module",
        allow_module_level=True,
    )


# Production pool row count — every test uses [POOL_ROWS, max_context_len].
POOL_ROWS = 256
# Production input_len cap — sum(req_lens) stays below this.
INPUT_LEN_CAP = 65536
# EOS token id sentinel. The kernel breaks the n-gram accumulation when it hits
# the eos token (token == eosTokenId && j > 0). The CPU reference below does not
# model EOS, so pass -1 to make the kernel's EOS branch a no-op: tokens equal to
# -1 are already caught by the ignore-marker check (token < 0) before the EOS
# check, and valid tokens are >= 0, so token == -1 never holds there.
EOS_TOKEN_ID = -1


def test_npu_compute_n_gram_ids_interface_exist():
    """Test that the operator is registered in torch.ops."""
    assert hasattr(torch.ops.custom, "npu_compute_n_gram_ids"), \
        "The 'npu_compute_n_gram_ids' operator is not registered in 'torch.ops.custom'."


# ---------------------------------------------------------------------------
# CPU reference — vectorized over `i` for fixed (req_id, n, k).
#
# For each (req_id, n, k):
#   alive[i]      : True while the j-loop hasn't broken for that i
#   The j loop breaks when either:
#     1. current_token_table_index - j < req_token_table_index  (position out
#        of req window, maps to col0 + i - j < 0);
#     2. the read token at that position is < 0 (ignore marker).
#   Contributions (token * weight) % oe_mod are accumulated only while alive.
# ---------------------------------------------------------------------------

def _ref_compute_n_gram_ids(
        oe_weights, oe_mods, exclusive_oe_embeder_size_sums,
        tokens, exclusive_req_len_sums, oe_token_table,
        row_indices, column_starts,
        batch_size, oe_n, oe_k, max_context_len):
    """CPU reference mirroring op_kernel/compute_n_gram_ids.h semantics."""
    assert oe_n >= 2 and oe_k > 0 and batch_size > 0
    ngram_cols = (oe_n - 1) * oe_k
    token_num = int(tokens.shape[0])
    out = np.zeros((token_num, ngram_cols), dtype=np.int32)

    ow = oe_weights.reshape(oe_n - 1, oe_k, oe_n).astype(np.int64)
    om = oe_mods.reshape(oe_n - 1, oe_k).astype(np.int64)
    embed = exclusive_oe_embeder_size_sums.reshape(-1).astype(np.int64)
    reqlen = exclusive_req_len_sums.reshape(-1).astype(np.int64)
    rowidx = row_indices.reshape(-1).astype(np.int64)
    colstart = column_starts.reshape(-1).astype(np.int64)
    tok_table = oe_token_table.astype(np.int64)  # [pool_rows, max_context_len]

    for req_id in range(batch_size):
        start = 0 if req_id == 0 else int(reqlen[req_id - 1])
        end = int(reqlen[req_id])
        req_len = end - start
        if req_len <= 0:
            continue
        row = int(rowidx[req_id])
        col0 = int(colstart[req_id])
        row_tokens = tok_table[row]  # [max_context_len]

        i_arr = np.arange(req_len, dtype=np.int64)

        for n in range(oe_n - 1):
            jmax = n + 2
            for k in range(oe_k):
                oe_mod = int(om[n, k])
                ngram_ids = np.zeros(req_len, dtype=np.int64)
                alive = np.ones(req_len, dtype=bool)

                for j in range(jmax):
                    pos = col0 + i_arr - j
                    valid = pos >= 0
                    tok = np.zeros(req_len, dtype=np.int64)
                    if valid.any():
                        tok[valid] = row_tokens[pos[valid]]
                    # Kernel order: position-check THEN token-check; once
                    # either fails, break the j-loop for that i.
                    alive = alive & valid & (tok >= 0)
                    if not alive.any():
                        break
                    weight = int(ow[n, k, j])
                    ngram_ids = ngram_ids + ((tok * weight) % oe_mod) * alive

                ngram_ids = ngram_ids % oe_mod + int(embed[n * oe_k + k])
                out[start:end, n * oe_k + k] = ngram_ids.astype(np.int32)

    return out


# ---------------------------------------------------------------------------
# Input generator — mirrors the production shapes:
#   oe_weights  : [oe_n-1, oe_k, oe_n]        int32
#   oe_mods     : [oe_n-1, oe_k]              int32
#   exclusive_oe_embeder_size_sums : [n_grams+1]  int32, production formula
#   tokens      : [token_num = sum(req_lens)] int32 (values unread by kernel)
#   exclusive_req_len_sums : [batch_size]     int32 (inclusive prefix sums)
#   oe_token_table : [POOL_ROWS, max_context_len]  int32
#   row_indices : [batch_size]                int64 in [0, POOL_ROWS)
#   column_starts : [batch_size]              int32, colstart+req_len<max_ctx
# ---------------------------------------------------------------------------

def _make_case(batch_size, oe_n, oe_k, max_context_len, seed=0,
               over_embedding_m=1, ignore_density=0.05):
    rng = np.random.default_rng(seed)
    assert 1 <= batch_size < POOL_ROWS
    ngram_cols = (oe_n - 1) * oe_k

    # Budget per req so total tokens <= INPUT_LEN_CAP.
    budget = INPUT_LEN_CAP // batch_size
    upper = min(budget, max(2, max_context_len - 1))
    req_lens = rng.integers(1, max(2, upper), size=batch_size,
                            dtype=np.int64)
    req_lens = np.clip(req_lens, 1, max_context_len - 1).astype(np.int32)
    exclusive_req_len_sums = np.cumsum(req_lens).astype(np.int32)
    token_num = int(exclusive_req_len_sums[-1])

    # Pool shape matches production: [256, max_context_len].
    oe_token_table = np.full((POOL_ROWS, max_context_len), -1, dtype=np.int32)
    row_indices = rng.permutation(POOL_ROWS)[:batch_size].astype(np.int64)

    column_starts = np.zeros(batch_size, dtype=np.int32)
    for b in range(batch_size):
        slack = max_context_len - int(req_lens[b])
        column_starts[b] = rng.integers(0, max(1, slack))

    # Populate each req's contiguous token window in its pool row.
    for b in range(batch_size):
        r = int(row_indices[b])
        c0 = int(column_starts[b])
        n_tok = int(req_lens[b])
        vals = rng.integers(0, 1 << 20, size=n_tok).astype(np.int32)
        if n_tok >= 3 and ignore_density > 0:
            n_ignore = max(1, int(n_tok * ignore_density))
            ignore_pos = rng.choice(n_tok, size=min(n_ignore, n_tok // 2),
                                    replace=False)
            vals[ignore_pos] = -1
        oe_token_table[r, c0:c0 + n_tok] = vals

    # tokens: kernel does not read it — shape is what matters for this op.
    tokens = np.arange(token_num, dtype=np.int32)

    # oe_weights / oe_mods — production dtypes + shapes.
    oe_weights = rng.integers(1, 100, size=(oe_n - 1, oe_k, oe_n),
                              dtype=np.int64).astype(np.int32)
    oe_mods = rng.integers(32, 2048, size=(oe_n - 1, oe_k),
                           dtype=np.int64).astype(np.int32)

    # Production exclusive_oe_embeder_size_sums formula:
    #   for i in range(n_grams): m = over_embedding_m + 2i + 1; sums[i+1] += m
    increments = np.array(
        [over_embedding_m + 2 * i + 1 for i in range(ngram_cols)],
        dtype=np.int32)
    exclusive_oe_embeder_size_sums = np.concatenate([
        np.zeros(1, dtype=np.int32), np.cumsum(increments).astype(np.int32)])

    return dict(
        oe_weights=oe_weights,
        oe_mods=oe_mods,
        exclusive_oe_embeder_size_sums=exclusive_oe_embeder_size_sums,
        tokens=tokens,
        exclusive_req_len_sums=exclusive_req_len_sums,
        oe_token_table=oe_token_table,
        row_indices=row_indices,
        column_starts=column_starts,
    )


# ---------------------------------------------------------------------------
# Test case grids
# ---------------------------------------------------------------------------

# (1) Edge cases / small-grid sanity — exercise bs=1 and small (oe_n, oe_k).
SMALL_CASES = [
    # (batch_size, oe_n, oe_k, max_context_len, seed)
    (1, 2, 1, 16, 1),
    (1, 3, 2, 32, 2),
    (2, 2, 4, 32, 3),
    (2, 3, 4, 64, 4),
    (4, 3, 2, 64, 5),
    (4, 4, 4, 128, 6),
    (8, 2, 8, 128, 7),      # test oe_k up to boundary-adjacent value
    (8, 3, 8, 256, 8),
    (16, 3, 4, 256, 9),
    (16, 4, 2, 512, 10),
]

# (2) Full (oe_n, oe_k) grid over the production range [2..7] × [1..7], at a
# representative batch / context size, to catch dtype / stride edge cases.
_OEN_RANGE = [2, 3, 4, 5, 6, 7]
_OEK_RANGE = [1, 2, 3, 4, 5, 6, 7]
GRID_CASES = [(32, n, k, 512, 100 + n * 10 + k)
              for n in _OEN_RANGE for k in _OEK_RANGE]

# (3) Large / production-like cases: stress core scheduling, ignore-token
# fan-out, 256-pool layout, and long max_context_len (up to 1M).
# total_task = batch_size * (oe_n-1) * oe_k; must exceed core count on AIV
# to exercise the per-core task split.
LARGE_CASES = [
    # (batch_size, oe_n, oe_k, max_context_len, seed)
    (64, 5, 5, 1024, 201),           # total_task = 64*4*5 = 1280
    (128, 4, 4, 1024, 202),          # total_task = 128*3*4 = 1536
    (200, 3, 3, 2048, 203),          # total_task = 200*2*3 = 1200
    (255, 7, 7, 512, 204),           # total_task = 255*6*7 = 10710 (max task)
    (255, 2, 1, 4096, 205),          # minimal work per task, max reqs
    (32, 7, 7, 8192, 206),           # larger context, full oe_n/oe_k
    (64, 6, 6, 16384, 207),          # 16K context
]

# (4) Very-large-context case — one big run at max_context_len = 1M, with a
# small batch to keep memory tractable. Skipped automatically if NPU memory
# cannot hold a [256, 1M, int32] = 1 GiB table.
HUGE_CASES = [
    (4, 3, 3, 1 << 20, 300),         # max_context_len = 1M (1GB pool)
    (8, 5, 5, 1 << 19, 301),         # max_context_len = 512K (512MB pool)
]


def _run_case(batch_size, oe_n, oe_k, max_context_len, seed,
              over_embedding_m=1):
    case = _make_case(batch_size, oe_n, oe_k, max_context_len, seed=seed,
                      over_embedding_m=over_embedding_m)

    expected = _ref_compute_n_gram_ids(
        case["oe_weights"], case["oe_mods"],
        case["exclusive_oe_embeder_size_sums"],
        case["tokens"], case["exclusive_req_len_sums"],
        case["oe_token_table"], case["row_indices"], case["column_starts"],
        batch_size, oe_n, oe_k, max_context_len)

    def to_npu(arr):
        return torch.from_numpy(np.ascontiguousarray(arr)).npu()

    out_npu = torch.ops.custom.npu_compute_n_gram_ids(
        to_npu(case["oe_weights"]),
        to_npu(case["oe_mods"]),
        to_npu(case["exclusive_oe_embeder_size_sums"]),
        to_npu(case["tokens"]),
        to_npu(case["exclusive_req_len_sums"]),
        to_npu(case["oe_token_table"]),
        to_npu(case["row_indices"]),
        to_npu(case["column_starts"]),
        batch_size, oe_n, oe_k, max_context_len, EOS_TOKEN_ID,
    )
    torch.npu.synchronize()
    actual = out_npu.cpu().numpy()

    assert actual.shape == expected.shape, \
        f"shape mismatch: actual={actual.shape} expected={expected.shape}"
    if not np.array_equal(actual, expected):
        diffs = np.argwhere(actual != expected)[:5]
        raise AssertionError(
            f"value mismatch for bs={batch_size}, oe_n={oe_n}, oe_k={oe_k}, "
            f"max_context_len={max_context_len}: "
            f"diff_count={(actual != expected).sum()}  "
            f"first_diffs={diffs.tolist()}  "
            f"actual[{tuple(diffs[0])}]={int(actual[tuple(diffs[0])])} "
            f"expected[{tuple(diffs[0])}]={int(expected[tuple(diffs[0])])}"
        )


# ---------------------------------------------------------------------------
# Parametrized tests
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("batch_size,oe_n,oe_k,max_context_len,seed", SMALL_CASES)
def test_small_cases(batch_size, oe_n, oe_k, max_context_len, seed):
    """Small edge cases: bs=1, minimal oe_n/oe_k."""
    _run_case(batch_size, oe_n, oe_k, max_context_len, seed)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("batch_size,oe_n,oe_k,max_context_len,seed", GRID_CASES)
def test_oen_oek_grid(batch_size, oe_n, oe_k, max_context_len, seed):
    """Full (oe_n, oe_k) production grid over [2..7] × [1..7]."""
    _run_case(batch_size, oe_n, oe_k, max_context_len, seed)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("batch_size,oe_n,oe_k,max_context_len,seed", LARGE_CASES)
def test_large_cases(batch_size, oe_n, oe_k, max_context_len, seed):
    """Production-scale: batch_size up to 255, total_task well above core count."""
    _run_case(batch_size, oe_n, oe_k, max_context_len, seed)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("batch_size,oe_n,oe_k,max_context_len,seed", HUGE_CASES)
def test_huge_context_cases(batch_size, oe_n, oe_k, max_context_len, seed):
    """Very long max_context_len — up to 1M (1 GiB pool)."""
    _run_case(batch_size, oe_n, oe_k, max_context_len, seed)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_no_ignore_tokens():
    """Sanity: disable ignore-token markers; kernel should accumulate all j's."""
    case = _make_case(16, 5, 3, 512, seed=400, ignore_density=0.0)
    expected = _ref_compute_n_gram_ids(
        case["oe_weights"], case["oe_mods"],
        case["exclusive_oe_embeder_size_sums"],
        case["tokens"], case["exclusive_req_len_sums"],
        case["oe_token_table"], case["row_indices"], case["column_starts"],
        16, 5, 3, 512)

    def to_npu(arr):
        return torch.from_numpy(np.ascontiguousarray(arr)).npu()

    out_npu = torch.ops.custom.npu_compute_n_gram_ids(
        to_npu(case["oe_weights"]),
        to_npu(case["oe_mods"]),
        to_npu(case["exclusive_oe_embeder_size_sums"]),
        to_npu(case["tokens"]),
        to_npu(case["exclusive_req_len_sums"]),
        to_npu(case["oe_token_table"]),
        to_npu(case["row_indices"]),
        to_npu(case["column_starts"]),
        16, 5, 3, 512, EOS_TOKEN_ID,
    )
    torch.npu.synchronize()
    actual = out_npu.cpu().numpy()
    assert np.array_equal(actual, expected)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_column_start_zero():
    """Sanity: column_starts all zero — every req starts at row head."""
    case = _make_case(8, 4, 4, 256, seed=500)
    case["column_starts"] = np.zeros_like(case["column_starts"])
    # Repopulate rows with col0 forced to 0 for self-consistency.
    for b in range(8):
        r = int(case["row_indices"][b])
        case["oe_token_table"][r, :] = -1
    rng = np.random.default_rng(500)
    req_lens = np.diff(np.concatenate([[0], case["exclusive_req_len_sums"]])).astype(int)
    for b in range(8):
        r = int(case["row_indices"][b])
        n_tok = int(req_lens[b])
        vals = rng.integers(0, 1 << 20, size=n_tok).astype(np.int32)
        if n_tok >= 3:
            vals[rng.choice(n_tok, size=max(1, n_tok // 10), replace=False)] = -1
        case["oe_token_table"][r, 0:n_tok] = vals

    expected = _ref_compute_n_gram_ids(
        case["oe_weights"], case["oe_mods"],
        case["exclusive_oe_embeder_size_sums"],
        case["tokens"], case["exclusive_req_len_sums"],
        case["oe_token_table"], case["row_indices"], case["column_starts"],
        8, 4, 4, 256)

    def to_npu(arr):
        return torch.from_numpy(np.ascontiguousarray(arr)).npu()

    out_npu = torch.ops.custom.npu_compute_n_gram_ids(
        to_npu(case["oe_weights"]),
        to_npu(case["oe_mods"]),
        to_npu(case["exclusive_oe_embeder_size_sums"]),
        to_npu(case["tokens"]),
        to_npu(case["exclusive_req_len_sums"]),
        to_npu(case["oe_token_table"]),
        to_npu(case["row_indices"]),
        to_npu(case["column_starts"]),
        8, 4, 4, 256, EOS_TOKEN_ID,
    )
    torch.npu.synchronize()
    actual = out_npu.cpu().numpy()
    assert np.array_equal(actual, expected)


# ---------------------------------------------------------------------------
# torchair 入图测试
# ---------------------------------------------------------------------------

import torchair
from torchair import patch_for_hcom
patch_for_hcom()


class ComputeNGramIdsModel(torch.nn.Module):
    def __init__(self, batch_size, oe_n, oe_k, max_context_len):
        super().__init__()
        self.batch_size = batch_size
        self.oe_n = oe_n
        self.oe_k = oe_k
        self.max_context_len = max_context_len

    def forward(self, oe_weights, oe_mods, exclusive_oe_embeder_size_sums,
                tokens, exclusive_req_len_sums, oe_token_table,
                row_indices, column_starts):
        return torch.ops.custom.npu_compute_n_gram_ids(
            oe_weights, oe_mods, exclusive_oe_embeder_size_sums,
            tokens, exclusive_req_len_sums, oe_token_table,
            row_indices, column_starts,
            self.batch_size, self.oe_n, self.oe_k, self.max_context_len, EOS_TOKEN_ID)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_compute_n_gram_ids_torchair():
    """torchair 入图测试：通过 torch.compile + torchair backend 执行算子并比对结果。"""
    batch_size, oe_n, oe_k, max_context_len, seed = 8, 3, 4, 256, 600
    case = _make_case(batch_size, oe_n, oe_k, max_context_len, seed=seed)

    expected = _ref_compute_n_gram_ids(
        case["oe_weights"], case["oe_mods"],
        case["exclusive_oe_embeder_size_sums"],
        case["tokens"], case["exclusive_req_len_sums"],
        case["oe_token_table"], case["row_indices"], case["column_starts"],
        batch_size, oe_n, oe_k, max_context_len)

    def to_npu(arr):
        return torch.from_numpy(np.ascontiguousarray(arr)).npu()

    model = ComputeNGramIdsModel(batch_size, oe_n, oe_k, max_context_len).npu()
    config = torchair.CompilerConfig()
    npu_backend = torchair.get_npu_backend(compiler_config=config)
    opt_model = torch.compile(model, backend=npu_backend)

    out_npu = opt_model(
        to_npu(case["oe_weights"]),
        to_npu(case["oe_mods"]),
        to_npu(case["exclusive_oe_embeder_size_sums"]),
        to_npu(case["tokens"]),
        to_npu(case["exclusive_req_len_sums"]),
        to_npu(case["oe_token_table"]),
        to_npu(case["row_indices"]),
        to_npu(case["column_starts"]),
    )
    torch.npu.synchronize()
    actual = out_npu.cpu().numpy()

    assert actual.shape == expected.shape, \
        f"shape mismatch: actual={actual.shape} expected={expected.shape}"
    assert np.array_equal(actual, expected), \
        f"value mismatch: diff_count={(actual != expected).sum()}"


# ---------------------------------------------------------------------------
# 入口：支持 `python3 test_npu_compute_n_gram_ids.py` 直接执行。
# 本文件用例基于 pytest 参数化编写，无原生 __main__ 入口；此处通过
# pytest.main 驱动自身，使 `python3` 与 `pytest` 两种方式均可运行，并透传退出码。
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))

