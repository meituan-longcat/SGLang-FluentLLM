#!/usr/bin/python3
"""MlpDflashInferAttn（DFlash PageAttention 推理）上板精度用例。

用例自 FIA 的 tests/pytest/test_fia_dflash_paged.py（sparse_mode=15）改写，
通过 flash_npu_kernel / flash_ops 完成扩展注册，调用
torch.ops.custom.mlp_dflash_infer_attn：
  - seq 长度使用 int64 NPU tensor，并作为 fullgraph 的真实运行时输入；
  - sparse_mode/inner_precise 不再需要（本算子语义固定 DFlash paged）；
  - K/V cache 为 [numBlocks, page_size, Nkv*D]，物理页随机打散。

语义：request 的 context_end = kv_len - q_len（anchor），可见逻辑 KV =
    [0, min(next_tokens, context_end)) ∪ [max(0, context_end - pre_tokens), kv_len)
    pre_tokens=INT32_MAX 全 context；next_tokens=0 关闭 sink；draft 块双向。

运行：pytest test_mlp_dflash_infer_attn.py（或 python3 直接跑，见文件末尾 main）。
"""

import math
import os

try:
    import pytest
except ModuleNotFoundError:
    # The file also supports ``python3 test_mlp_dflash_infer_attn.py`` on
    # minimal board environments where pytest is not installed.
    class _PytestMarkFallback:
        @staticmethod
        def parametrize(*_args, **_kwargs):
            return lambda function: function

    class _PytestFallback:
        mark = _PytestMarkFallback()

    pytest = _PytestFallback()
import torch
import torch_npu

import flash_npu_kernel  # noqa: F401  # Registers torch.ops.flash operators.
import flash_ops  # noqa: F401  # Registers torch.ops.custom operators.

DEVICE = "npu:0"
DTYPE = torch.bfloat16
INT32_MAX = 2147483647


def _visible_indices(q_len, kv_len, pre_tokens, prefix_sink):
    context_end = kv_len - q_len
    tail_start = 0 if pre_tokens == INT32_MAX else max(0, context_end - pre_tokens)
    sink_end = min(prefix_sink, context_end)
    if sink_end == 0:
        return list(range(tail_start, kv_len))
    if tail_start <= sink_end:
        return list(range(kv_len))
    return list(range(sink_end)) + list(range(tail_start, kv_len))


def _logical_cache(cache, block_table, kv_len, page_size, kv_heads, head_dim):
    page_count = math.ceil(kv_len / page_size)
    pages = cache[block_table[0, :page_count].long()]
    return pages.reshape(-1, kv_heads, head_dim)[:kv_len]


def _call_op(query, key_cache, value_cache, actual_q, actual_kv, block_table,
             q_heads, kv_heads, scale, pre_tokens, prefix_sink, page_size,
             softmax_lse_flag=False):
    return torch.ops.custom.mlp_dflash_infer_attn(
        query.to(DEVICE),
        key_cache.to(DEVICE),
        value_cache.to(DEVICE),
        actual_seq_lengths=actual_q.to(DEVICE),
        actual_seq_lengths_kv=actual_kv.to(DEVICE),
        block_table=block_table.to(DEVICE),
        num_heads=q_heads,
        scale=scale,
        pre_tokens=pre_tokens,
        next_tokens=prefix_sink,
        input_layout="TND",
        num_key_value_heads=kv_heads,
        block_size=page_size,
        softmax_lse_flag=softmax_lse_flag,
    )


@pytest.mark.parametrize(
    "pre_tokens,prefix_sink",
    [
        (INT32_MAX, 0),   # FA + DFlash
        (4096, 0),        # SWA + DFlash
        (4096, 8),        # 最小正 sink
        (4096, 16),       # SWA + prefix sink 16 + DFlash（默认配置）
        (4096, 24),       # 非默认对齐 sink
    ],
)
def test_dflash_infer_paged_precision(pre_tokens, prefix_sink):
    torch_npu.npu.set_device(0)
    torch.manual_seed(20260809)

    q_len = 16
    # context_end 与 tail_start 都落在 128-token 页内（默认 kv_len=8192）。
    # DFLASH_TEST_KV_LEN 可放大到长 cache 场景，避免常规用例物化大 golden。
    kv_len = int(os.getenv("DFLASH_TEST_KV_LEN", "8192"))
    page_size = 128
    q_heads = 8
    kv_heads = 2
    head_dim = 128
    logical_page_count = math.ceil(kv_len / page_size)
    physical_page_count = logical_page_count + 7

    query = torch.randn(q_len, q_heads, head_dim, dtype=DTYPE)
    key_cache = torch.randn(
        physical_page_count, page_size, kv_heads * head_dim, dtype=DTYPE
    )
    value_cache = torch.randn_like(key_cache)
    page_ids = torch.randperm(physical_page_count, dtype=torch.int32)
    block_table = -torch.ones(1, physical_page_count, dtype=torch.int32)
    block_table[0, :logical_page_count] = page_ids[:logical_page_count]

    logical_key = _logical_cache(
        key_cache, block_table, kv_len, page_size, kv_heads, head_dim
    )
    logical_value = _logical_cache(
        value_cache, block_table, kv_len, page_size, kv_heads, head_dim
    )
    visible = torch.tensor(
        _visible_indices(q_len, kv_len, pre_tokens, prefix_sink), dtype=torch.long
    )
    key = logical_key[visible].repeat_interleave(q_heads // kv_heads, dim=1).float()
    value = logical_value[visible].repeat_interleave(q_heads // kv_heads, dim=1).float()
    scale = 1.0 / math.sqrt(head_dim)
    score = torch.einsum("qhd,khd->qhk", query.float(), key) * scale
    golden = torch.einsum("qhk,khd->qhd", torch.softmax(score, dim=-1), value)

    actual_q = torch.tensor([q_len], dtype=torch.int64)
    actual_kv = torch.tensor([kv_len], dtype=torch.int64)
    output, _ = _call_op(query, key_cache, value_cache, actual_q, actual_kv,
                         block_table, q_heads, kv_heads, scale, pre_tokens, prefix_sink,
                         page_size)
    torch.npu.synchronize()

    torch.testing.assert_close(
        output.cpu().float(), golden, rtol=5e-3, atol=5e-2
    )


@pytest.mark.parametrize("prefix_sink", [0, 8, 16, 24])
def test_dflash_infer_paged_variable_length_batch(prefix_sink):
    torch_npu.npu.set_device(0)
    torch.manual_seed(20260809)

    q_lens = [7, 16]
    kv_lens = [4123, 8207]
    page_size = 128
    q_heads = 8
    kv_heads = 2
    head_dim = 128
    page_counts = [math.ceil(length / page_size) for length in kv_lens]
    max_page_count = max(page_counts)
    physical_page_count = sum(page_counts) + 7

    query = torch.randn(sum(q_lens), q_heads, head_dim, dtype=DTYPE)
    key_cache = torch.randn(
        physical_page_count, page_size, kv_heads * head_dim, dtype=DTYPE
    )
    value_cache = torch.randn_like(key_cache)
    page_ids = torch.randperm(physical_page_count, dtype=torch.int32)
    block_table = -torch.ones(len(q_lens), max_page_count, dtype=torch.int32)

    goldens = []
    page_offset = 0
    q_offset = 0
    scale = 1.0 / math.sqrt(head_dim)
    for batch_idx, (q_len, kv_len, page_count) in enumerate(
        zip(q_lens, kv_lens, page_counts)
    ):
        block_table[batch_idx, :page_count] = page_ids[
            page_offset : page_offset + page_count
        ]
        page_offset += page_count
        logical_key = _logical_cache(
            key_cache,
            block_table[batch_idx : batch_idx + 1],
            kv_len,
            page_size,
            kv_heads,
            head_dim,
        )
        logical_value = _logical_cache(
            value_cache,
            block_table[batch_idx : batch_idx + 1],
            kv_len,
            page_size,
            kv_heads,
            head_dim,
        )
        visible = torch.tensor(
            _visible_indices(q_len, kv_len, 4096, prefix_sink), dtype=torch.long
        )
        key = logical_key[visible].repeat_interleave(q_heads // kv_heads, dim=1).float()
        value = logical_value[visible].repeat_interleave(q_heads // kv_heads, dim=1).float()
        query_slice = query[q_offset : q_offset + q_len].float()
        score = torch.einsum("qhd,khd->qhk", query_slice, key) * scale
        goldens.append(
            torch.einsum("qhk,khd->qhd", torch.softmax(score, dim=-1), value)
        )
        q_offset += q_len

    actual_q = torch.tensor([q_lens[0], sum(q_lens)], dtype=torch.int64)  # TND 前缀和
    actual_kv = torch.tensor(kv_lens, dtype=torch.int64)
    output, softmax_lse = _call_op(
        query, key_cache, value_cache, actual_q, actual_kv, block_table,
        q_heads, kv_heads, scale, 4096, prefix_sink, page_size,
    )
    torch.npu.synchronize()

    assert softmax_lse.shape == (0,)
    torch.testing.assert_close(
        output.cpu().float(), torch.cat(goldens), rtol=5e-3, atol=5e-2
    )


def test_dflash_infer_paged_lse_precision():
    """Check the optional LSE output contract and its numerical accuracy."""
    torch_npu.npu.set_device(0)
    torch.manual_seed(20260810)

    q_len = 9
    kv_len = 1051
    page_size = 128
    q_heads = 8
    kv_heads = 2
    head_dim = 128
    pre_tokens = 512
    prefix_sink = 16
    logical_page_count = math.ceil(kv_len / page_size)
    physical_page_count = logical_page_count + 3

    query = torch.randn(q_len, q_heads, head_dim, dtype=DTYPE)
    key_cache = torch.randn(
        physical_page_count, page_size, kv_heads * head_dim, dtype=DTYPE
    )
    value_cache = torch.randn_like(key_cache)
    page_ids = torch.randperm(physical_page_count, dtype=torch.int32)
    block_table = -torch.ones(1, physical_page_count, dtype=torch.int32)
    block_table[0, :logical_page_count] = page_ids[:logical_page_count]

    logical_key = _logical_cache(
        key_cache, block_table, kv_len, page_size, kv_heads, head_dim
    )
    logical_value = _logical_cache(
        value_cache, block_table, kv_len, page_size, kv_heads, head_dim
    )
    visible = torch.tensor(
        _visible_indices(q_len, kv_len, pre_tokens, prefix_sink), dtype=torch.long
    )
    key = logical_key[visible].repeat_interleave(q_heads // kv_heads, dim=1).float()
    value = logical_value[visible].repeat_interleave(q_heads // kv_heads, dim=1).float()
    scale = 1.0 / math.sqrt(head_dim)
    score = torch.einsum("qhd,khd->qhk", query.float(), key) * scale
    golden = torch.einsum("qhk,khd->qhd", torch.softmax(score, dim=-1), value)
    golden_lse = torch.logsumexp(score, dim=-1).unsqueeze(-1)

    actual_q = torch.tensor([q_len], dtype=torch.int64)
    actual_kv = torch.tensor([kv_len], dtype=torch.int64)
    output, softmax_lse = _call_op(
        query, key_cache, value_cache, actual_q, actual_kv, block_table,
        q_heads, kv_heads, scale, pre_tokens, prefix_sink, page_size,
        softmax_lse_flag=True,
    )
    torch.npu.synchronize()

    assert softmax_lse.shape == (q_len, q_heads, 1)
    assert softmax_lse.dtype == torch.float32
    torch.testing.assert_close(output.cpu().float(), golden, rtol=5e-3, atol=5e-2)
    torch.testing.assert_close(
        softmax_lse.cpu(), golden_lse, rtol=5e-3, atol=5e-2
    )


@pytest.mark.parametrize("softmax_lse_flag", [False, True])
def test_dflash_infer_paged_graph_precision(softmax_lse_flag):
    """Check fullgraph execution with and without the optional LSE output."""
    import torchair

    torch_npu.npu.set_device(0)
    torch.manual_seed(20260817)

    q_len = 9
    kv_len = 1051
    page_size = 128
    q_heads = 8
    kv_heads = 2
    head_dim = 128
    pre_tokens = 512
    prefix_sink = 16
    scale = 1.0 / math.sqrt(head_dim)
    logical_page_count = math.ceil(kv_len / page_size)
    physical_page_count = logical_page_count + 3

    query = torch.randn(q_len, q_heads, head_dim, dtype=DTYPE)
    key_cache = torch.randn(
        physical_page_count, page_size, kv_heads * head_dim, dtype=DTYPE
    )
    value_cache = torch.randn_like(key_cache)
    page_ids = torch.randperm(physical_page_count, dtype=torch.int32)
    block_table = -torch.ones(1, physical_page_count, dtype=torch.int32)
    block_table[0, :logical_page_count] = page_ids[:logical_page_count]

    logical_key = _logical_cache(
        key_cache, block_table, kv_len, page_size, kv_heads, head_dim
    )
    logical_value = _logical_cache(
        value_cache, block_table, kv_len, page_size, kv_heads, head_dim
    )
    visible = torch.tensor(
        _visible_indices(q_len, kv_len, pre_tokens, prefix_sink), dtype=torch.long
    )
    key = logical_key[visible].repeat_interleave(q_heads // kv_heads, dim=1).float()
    value = logical_value[visible].repeat_interleave(q_heads // kv_heads, dim=1).float()
    score = torch.einsum("qhd,khd->qhk", query.float(), key) * scale
    golden = torch.einsum("qhk,khd->qhd", torch.softmax(score, dim=-1), value)
    golden_lse = torch.logsumexp(score, dim=-1).unsqueeze(-1)

    def graph_op(
        query_npu,
        key_cache_npu,
        value_cache_npu,
        actual_q_npu,
        actual_kv_npu,
        block_table_npu,
    ):
        return torch.ops.custom.mlp_dflash_infer_attn(
            query_npu,
            key_cache_npu,
            value_cache_npu,
            actual_seq_lengths=actual_q_npu,
            actual_seq_lengths_kv=actual_kv_npu,
            block_table=block_table_npu,
            num_heads=q_heads,
            scale=scale,
            pre_tokens=pre_tokens,
            next_tokens=prefix_sink,
            input_layout="TND",
            num_key_value_heads=kv_heads,
            block_size=page_size,
            softmax_lse_flag=softmax_lse_flag,
        )

    compiler_config = torchair.CompilerConfig()
    # Exercise the production path that sinks value-dependent tiling to AICPU.
    compiler_config.experimental_config.tiling_schedule_optimize = True
    compiled = torch.compile(
        graph_op,
        backend=torchair.get_npu_backend(compiler_config=compiler_config),
        fullgraph=True,
        dynamic=False,
    )
    actual_q_npu = torch.tensor([q_len], dtype=torch.int64, device=DEVICE)
    actual_kv_npu = torch.tensor([kv_len], dtype=torch.int64, device=DEVICE)
    output, softmax_lse = compiled(
        query.to(DEVICE),
        key_cache.to(DEVICE),
        value_cache.to(DEVICE),
        actual_q_npu,
        actual_kv_npu,
        block_table.to(DEVICE),
    )
    torch_npu.npu.synchronize()

    assert output.shape == (q_len, q_heads, head_dim)
    assert output.dtype == DTYPE
    torch.testing.assert_close(output.cpu().float(), golden, rtol=5e-3, atol=5e-2)
    if softmax_lse_flag:
        assert softmax_lse.shape == (q_len, q_heads, 1)
        assert softmax_lse.dtype == torch.float32
        torch.testing.assert_close(
            softmax_lse.cpu(), golden_lse, rtol=5e-3, atol=5e-2
        )
    else:
        assert softmax_lse.shape == (0,)


def test_dflash_infer_paged_stability():
    """The same A2 input must produce bitwise-identical output repeatedly."""
    torch_npu.npu.set_device(0)
    torch.manual_seed(20260811)

    q_len = 16
    kv_len = 4097
    page_size = 128
    q_heads = 8
    kv_heads = 2
    head_dim = 128
    logical_page_count = math.ceil(kv_len / page_size)
    physical_page_count = logical_page_count + 3

    query = torch.randn(q_len, q_heads, head_dim, dtype=DTYPE)
    key_cache = torch.randn(
        physical_page_count, page_size, kv_heads * head_dim, dtype=DTYPE
    )
    value_cache = torch.randn_like(key_cache)
    page_ids = torch.randperm(physical_page_count, dtype=torch.int32)
    block_table = -torch.ones(1, physical_page_count, dtype=torch.int32)
    block_table[0, :logical_page_count] = page_ids[:logical_page_count]
    actual_q = torch.tensor([q_len], dtype=torch.int64)
    actual_kv = torch.tensor([kv_len], dtype=torch.int64)
    scale = 1.0 / math.sqrt(head_dim)

    def run_once():
        output, softmax_lse = _call_op(
            query, key_cache, value_cache, actual_q, actual_kv, block_table,
            q_heads, kv_heads, scale, 4096, 16, page_size,
            softmax_lse_flag=True,
        )
        torch.npu.synchronize()
        return output.cpu(), softmax_lse.cpu()

    expected_output, expected_lse = run_once()
    for iteration in range(20):
        output, softmax_lse = run_once()
        assert torch.equal(output, expected_output), f"output changed at iteration {iteration}"
        assert torch.equal(softmax_lse, expected_lse), f"LSE changed at iteration {iteration}"


if __name__ == "__main__":
    # 无 pytest 时也可直接跑：逐个执行全部参数组合
    for pre, sink in [(INT32_MAX, 0), (4096, 0), (4096, 8), (4096, 16), (4096, 24)]:
        test_dflash_infer_paged_precision(pre, sink)
        print(f"precision pre={pre} sink={sink} PASS")
    for sink in [0, 8, 16, 24]:
        test_dflash_infer_paged_variable_length_batch(sink)
        print(f"varlen-batch sink={sink} PASS")
    test_dflash_infer_paged_lse_precision()
    print("lse precision PASS")
    for softmax_lse_flag in [False, True]:
        test_dflash_infer_paged_graph_precision(softmax_lse_flag)
        print(f"fullgraph softmax_lse_flag={softmax_lse_flag} PASS")
    test_dflash_infer_paged_stability()
    print("stability 20 iterations PASS")
    print("ALL PASS")
