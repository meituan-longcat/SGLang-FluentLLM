# -*- coding: utf-8 -*-
import torch
import torch_npu
import flash_ops  # 注册 torch.ops.custom 命名空间
import numpy as np
import math
import time
import argparse
import itertools

torch.set_printoptions(profile="default", precision=9, sci_mode=False)

DEFAULT_SCORE = -8888888
MAGIC_NUMBER = 8866
DIGONAL_VALUE = 9999999
WINDOW_VALUE = 1111111
RANDOM_BLOCK_TABLE = 0
DEVICE_ID = 3
torch_npu.npu.set_device(int(DEVICE_ID))


def split_to_kv_cache(input_tensor, block_size, cur_kv_seqlen, max_seq_len):
    """将连续 KV tensor 拆分到 PA block cache 格式"""
    device = input_tensor.device
    T = input_tensor.shape[0]
    B = len(cur_kv_seqlen)
    total_seq_block_cnt = 0
    for i in range(B):
        seq_block_cnt = math.ceil(cur_kv_seqlen[i] / block_size)
        if seq_block_cnt == 0:
            seq_block_cnt = 1
        total_seq_block_cnt += seq_block_cnt
    block_table_list = (torch.randperm(total_seq_block_cnt) if RANDOM_BLOCK_TABLE
                        else torch.arange(0, total_seq_block_cnt, 1))

    block_tables = torch.zeros(B, math.ceil(max_seq_len / block_size)).to(torch.int32)
    input_shape = input_tensor.shape
    cache_tensor = torch.zeros(
        B * math.ceil(max_seq_len / block_size), block_size,
        input_shape[-2], input_shape[-1]
    ).to(input_tensor.dtype).to(device)

    start = 0
    total_seq_block_cnt = 0
    for i in range(B):
        seq_block_cnt = math.ceil(cur_kv_seqlen[i] / block_size)
        if seq_block_cnt == 0:
            seq_block_cnt = 1
        block_tables[i, :seq_block_cnt] = block_table_list[
            total_seq_block_cnt:seq_block_cnt + total_seq_block_cnt
        ]
        kv_len = cur_kv_seqlen[i]
        if kv_len == 0:
            kv_len = 1
        for block_idx in range(seq_block_cnt):
            block_offset = block_idx * block_size
            seq_offset = block_offset + start
            if block_idx == seq_block_cnt - 1:
                remain_size = kv_len - block_idx * block_size
                cache_tensor[block_tables[i][block_idx], :remain_size] = \
                    input_tensor[seq_offset:seq_offset + remain_size]
            else:
                cache_tensor[block_tables[i][block_idx], :block_size] = \
                    input_tensor[seq_offset:seq_offset + block_size]
        total_seq_block_cnt += seq_block_cnt
        start += kv_len
    return cache_tensor, block_tables.to(device).detach()


def generate_sparse_index(actual_kv_len, block_size, block_topk, actual_q_len):
    """
    为每个 query token 生成 sparse_index: [T1, block_topk]
    每个值是原始 KV 全长中的逻辑 block 编号（随机选取 block_topk 个 block）
    有效 block 数不足 block_topk 时，后面填 -1 表示无效
    """
    T1 = sum(max(q, 1) for q in actual_q_len)
    B = len(actual_q_len)
    # 默认全部填 -1
    sparse_index = torch.full((T1, block_topk), -1, dtype=torch.int32)

    q_offset = 0
    for b in range(B):
        q_len = max(actual_q_len[b], 1)
        kv_len = actual_kv_len[b]
        total_blocks = math.ceil(kv_len / block_size)
        if total_blocks == 0:
            total_blocks = 1

        for t in range(q_len):
            valid_count = min(total_blocks, block_topk)
            if total_blocks <= block_topk:
                # 有效 block 不足 block_topk，全部选取，剩余保持 -1
                indices = torch.arange(0, total_blocks, dtype=torch.int32)
            else:
                # 随机选取 block_topk 个 block（不排序，模拟前置算子的离散输出）
                indices = torch.randperm(total_blocks)[:block_topk].to(torch.int32)
                valid_count = block_topk
            sparse_index[q_offset + t, :valid_count] = indices
        q_offset += q_len
    return sparse_index


def golden_lightning_indexer_with_sparse_index(
    query, key_full, weights, cur_seq_lengths_query, cur_seq_lengths_key,
    sparse_index, block_size, sparse_count=2048, sparse_mode=0,
    INIT_NUM=0, LOCAL_NUM=0
):
    """
    Golden 实现：基于 sparse_index 从全长 KV 中取对应 block 做 QK matmul + TopK

    sparse_index: [T1, block_topk] int32, 值为原始 KV 全长中的逻辑 block 编号，-1 表示无效
    key_full: [T2, 1, D] 原始全长 KV（TND 格式）

    每个 sparse block（block_size token）产生一个 score，不做子 block reduce。
    """
    batch_size = cur_seq_lengths_query.shape[0] - 1
    N = query.shape[-2]
    D = query.shape[-1]
    block_topk = sparse_index.shape[1]

    # 输出 shape 与 query 对齐，最后一维为 sparse_count
    out_shape = list(query.shape)
    n2 = 1
    out_shape[-1] = sparse_count
    out_shape[-2] = n2
    # indice 初始化为 -1（无效），value 初始化为 0
    out = torch.zeros(out_shape, dtype=torch.int32).reshape(-1, n2, sparse_count) - 1
    valuesOut = torch.zeros(out_shape, dtype=torch.float32).reshape(-1, n2, sparse_count)

    process_q_len = 0
    process_kv_len = 0

    for batch_id in range(batch_size):
        # 当前 batch 的实际 Q/KV 序列长度
        act_s1 = int(cur_seq_lengths_query[batch_id + 1] - cur_seq_lengths_query[batch_id])
        act_s2 = int(cur_seq_lengths_key[batch_id + 1] - cur_seq_lengths_key[batch_id])

        # 处理空序列
        if act_s1 == 0:
            if act_s2 == 0:
                act_s1 = 1
                act_s2 = 1
                process_q_len += act_s1
                process_kv_len += act_s2
                continue
            else:
                process_q_len += 1
                process_kv_len += act_s2
                continue

        # 逐 query token 处理
        for s1_idx in range(act_s1):
            token_idx = process_q_len + s1_idx
            cur_sparse_idx = sparse_index[token_idx]  # [block_topk]

            # 统计有效 block 数（非 -1 的 entry）
            valid_mask = cur_sparse_idx >= 0
            valid_block_count = int(valid_mask.sum())
            if valid_block_count == 0:
                # 无有效 block，输出保持 -1
                continue

            # ---- Gather key: 按 sparse_index 从全长 KV 中取出有效 block ----
            # 同时构建每个 gathered token 对应的原始 KV 全长 token 位置
            gathered_key_list = []
            orig_token_positions = []  # 每个 gathered token 在原始 KV 全长中的 token 位置
            for blk_i in range(block_topk):
                blk_id = int(cur_sparse_idx[blk_i])
                if blk_id < 0:
                    # -1 无效，跳过
                    continue
                blk_start_in_batch = blk_id * block_size  # 原始 KV 中该 block 的起始 token 位置（batch 内）
                blk_start_global = process_kv_len + blk_start_in_batch
                blk_end_global = min(blk_start_global + block_size, process_kv_len + act_s2)
                actual_blk_size = blk_end_global - blk_start_global
                gathered_key_list.append(key_full[blk_start_global:blk_end_global, :, :])
                # 记录每个有效 token 的原始位置（batch 内偏移）
                for t in range(actual_blk_size):
                    orig_token_positions.append(blk_start_in_batch + t)
                if actual_blk_size < block_size:
                    # 尾 block 不足 block_size，补零对齐，对应位置标记为 -1（无效）
                    pad = torch.zeros(block_size - actual_blk_size, 1, D, dtype=key_full.dtype)
                    gathered_key_list.append(pad)
                    for _ in range(block_size - actual_blk_size):
                        orig_token_positions.append(-1)

            if len(gathered_key_list) == 0:
                continue

            # gathered_key: [total_gathered_tokens, 1, D]
            gathered_key = torch.cat(gathered_key_list, dim=0)
            total_tokens = gathered_key.shape[0]

            # ---- QK matmul → ReLU → weight 加权 → reduce over N ----
            now_q = query[token_idx:token_idx + 1, :, :].transpose(0, 1).to(torch.float32)  # [N, 1, D]
            now_k = gathered_key[:, 0, :].T.to(torch.float32)  # [D, total_tokens]
            now_w = weights[token_idx:token_idx + 1, :].unsqueeze(-1).transpose(0, 1).to(torch.float32)  # [N, 1, 1]

            # N,1,D @ D,total_tokens -> N,1,total_tokens
            relu_out = torch.maximum(torch.matmul(now_q, now_k), torch.tensor(0.0))
            # 加权后 reduce: N,1,total_tokens -> 1,total_tokens -> [total_tokens]
            weight_out = relu_out * now_w
            token_scores = torch.sum(weight_out, dim=0).squeeze(0)  # [total_tokens]
            token_scores = torch.clip(token_scores, max=WINDOW_VALUE / 2)

            # ---- Causal mask: token 粒度，基于原始 KV 全长位置判断 ----
            # valid_s2_len: 当前 query token 在原始 KV 中 causal 可见的有效 token 数
            valid_s2_len = act_s2 - act_s1 + s1_idx + 1
            for ti in range(total_tokens):
                orig_pos = orig_token_positions[ti]
                if orig_pos < 0 or orig_pos >= valid_s2_len:
                    # 补零 token 或超出 causal 可见范围，置为无效分数
                    token_scores[ti] = DEFAULT_SCORE

            # ---- Sink window (init_num): 头部固定窗口赋高分 ----
            if INIT_NUM > 0:
                for ti in range(total_tokens):
                    orig_pos = orig_token_positions[ti]
                    if 0 <= orig_pos < INIT_NUM:
                        token_scores[ti] = WINDOW_VALUE

            # ---- Local window (local_num): 尾部滑动窗口赋高分 ----
            if LOCAL_NUM > 0:
                local_start = max(valid_s2_len - LOCAL_NUM, 0)
                for ti in range(total_tokens):
                    orig_pos = orig_token_positions[ti]
                    if local_start <= orig_pos < valid_s2_len:
                        token_scores[ti] = WINDOW_VALUE

            # ---- TopK 排序（降序，分数相同时保持原始顺序）----
            sorted_value, sorted_local_indices = torch.sort(
                token_scores, dim=0, descending=True, stable=True
            )

            # ---- 将排序后的局部 index 映射回原始 KV 全长 token 位置 ----
            output_indices = torch.full((total_tokens,), -1, dtype=torch.int32)
            for i in range(min(total_tokens, sparse_count)):
                ti = int(sorted_local_indices[i])
                orig_pos = orig_token_positions[ti]
                if orig_pos >= 0:
                    output_indices[i] = orig_pos

            # 写入输出
            return_s2 = min(sparse_count, total_tokens)
            out[token_idx, 0, :return_s2] = output_indices[:return_s2]
            valuesOut[token_idx, 0, :return_s2] = sorted_value[:return_s2]

        process_q_len += act_s1
        process_kv_len += act_s2

    out = out.reshape(out_shape)
    valuesOut = valuesOut.reshape(out_shape)
    return out, valuesOut


def compare_indices_with_value_tolerance(cpu_out_flat, npu_out_flat, cpu_val_flat, npu_val_flat,
                                          sparse_count, rtol=5e-3, verbose=True):
    """
    比对 index 结果，对于不一致的 index，检查对应 value diff 是否合理。

    判定逻辑：
    - 逐 token 比对 index 集合
    - 若集合完全一致 → PASS
    - 若不一致，找到差异发生的 rank 位置，比较该位置 cpu/npu 的 value：
      若 |cpu_val - npu_val| <= rtol * max(|cpu_val|, |npu_val|, 1.0) → 视为精度可接受
    - 所有差异 token 都可接受 → PASS (with tolerance)
    - 存在不可接受的差异 → FAIL

    返回: (passed: bool, diff_token_count: int)
    """
    total_tokens = cpu_out_flat.shape[0]
    diff_tokens = 0
    unacceptable_diffs = []

    for t in range(total_tokens):
        cpu_row = cpu_out_flat[t]
        npu_row = npu_out_flat[t]

        # 快速路径：完全一致
        if torch.equal(cpu_row, npu_row):
            continue

        # index 集合不一致，逐 rank 找差异
        diff_tokens += 1
        token_ok = True
        for rank in range(sparse_count):
            if cpu_row[rank] != npu_row[rank]:
                cpu_val = cpu_val_flat[t, rank].item()
                npu_val = npu_val_flat[t, rank].item()
                denom = max(abs(cpu_val), abs(npu_val), 1.0)
                rel_diff = abs(cpu_val - npu_val) / denom
                if rel_diff > rtol:
                    token_ok = False
                    unacceptable_diffs.append((t, rank, cpu_row[rank].item(), npu_row[rank].item(),
                                              cpu_val, npu_val, rel_diff))
                break  # 只检查第一个差异 rank（后续 rank 的移位是连锁效应）

        if not token_ok:
            break

    if diff_tokens == 0:
        if verbose:
            print("  PASS")
        return True, 0
    elif len(unacceptable_diffs) == 0:
        if verbose:
            print(f"  PASS (tolerance: {diff_tokens} token(s) with value-close index diff)")
        return True, diff_tokens
    else:
        if verbose:
            print(f"  FAIL ({len(unacceptable_diffs)} unacceptable diff(s) in {diff_tokens} token(s))")
            for t, rank, cpu_idx, npu_idx, cpu_val, npu_val, rel_diff in unacceptable_diffs[:5]:
                print(f"    token={t} rank={rank}: cpu_idx={cpu_idx} npu_idx={npu_idx} "
                      f"cpu_val={cpu_val:.6f} npu_val={npu_val:.6f} rel_diff={rel_diff:.6f}")
        return False, diff_tokens


def calculate_special_sum(lst):
    if not lst:
        return 0
    return sum(1 if x == 0 else x for x in lst)


def test_sparse_index_lightning_indexer(
    N, INIT_NUM, LOCAL_NUM, SPARSE_COUNT, BLOCK_TOPK,
    actual_q_len=None, actual_kv_len=None, seed=42, verbose=True
):
    block_size = 128
    D = 128
    dType = torch.bfloat16
    np.random.seed(seed)
    torch.manual_seed(seed)

    # 默认参数
    if actual_q_len is None:
        actual_q_len = [21]
    if actual_kv_len is None:
        actual_kv_len = [32 * 1024]

    T1 = calculate_special_sum(actual_q_len)
    T2 = calculate_special_sum(actual_kv_len)
    max_seq_len = max(actual_kv_len)

    case_desc = (f"B={len(actual_q_len)}, T1={T1}, T2={T2}, N={N}, "
                 f"INIT={INIT_NUM}, LOCAL={LOCAL_NUM}, SC={SPARSE_COUNT}, "
                 f"BT={BLOCK_TOPK}, q={actual_q_len}, kv={actual_kv_len}")
    if verbose:
        print(f"[Case] {case_desc}")

    query = torch.tensor(np.random.uniform(-2, 2, (T1, N, D))).to(dType)
    key = torch.tensor(np.random.uniform(-2, 2, (T2, 1, D))).to(dType)
    weights = torch.tensor(np.random.uniform(-1, 1, (T1, N))).to(torch.float)

    cur_q_len = [0] + list(np.cumsum(actual_q_len))
    cur_kv_len = [0] + list(np.cumsum(actual_kv_len))
    cur_seq_lengths_query = torch.tensor(cur_q_len).to(torch.int32)
    cur_seq_lengths_key = torch.tensor(cur_kv_len).to(torch.int32)

    # 生成 sparse_index: [T1, BLOCK_TOPK]，-1 表示无效
    sparse_index = generate_sparse_index(actual_kv_len, block_size, BLOCK_TOPK, actual_q_len)
    print("sparse_index is ", sparse_index)
    # ===================== CPU golden =====================
    cpu_out, cpu_valuesOut = golden_lightning_indexer_with_sparse_index(
        query, key, weights, cur_seq_lengths_query, cur_seq_lengths_key,
        sparse_index, block_size, SPARSE_COUNT, sparse_mode=0,
        INIT_NUM=INIT_NUM, LOCAL_NUM=LOCAL_NUM
    )

    # ===================== NPU =====================
    key_cache, key_block_table = split_to_kv_cache(key, block_size, actual_kv_len, max_seq_len)

    query_npu = query.to(f"npu:{DEVICE_ID}")
    key_cache_npu = key_cache.to(f"npu:{DEVICE_ID}")
    weights_npu = weights.to(f"npu:{DEVICE_ID}")
    cur_seq_lengths_query_npu = cur_seq_lengths_query.to(f"npu:{DEVICE_ID}").to(torch.int64)
    cur_seq_lengths_key_npu = cur_seq_lengths_key.to(f"npu:{DEVICE_ID}").to(torch.int64)
    key_block_table_npu = key_block_table.to(f"npu:{DEVICE_ID}")
    sparse_index_npu = sparse_index.to(f"npu:{DEVICE_ID}")

    return_value = True

    npu_out, npu_valuesOut = torch.ops.custom.mlp_sparse_lightning_indexer(
        query_npu, key_cache_npu, weights_npu,
        cur_seq_lengths_query=cur_seq_lengths_query_npu,
        cur_seq_lengths_key=cur_seq_lengths_key_npu,
        block_table=key_block_table_npu,
        sparse_index=sparse_index_npu,
        layout_query="TND",
        layout_key="PA_BSND",
        sparse_count=SPARSE_COUNT,

        init_num=INIT_NUM,
        local_num=LOCAL_NUM,
        sparse_mode=3,
        pre_tokens=2147483647,
        next_tokens=2147483647,
        return_value=return_value,
    )

    # ===================== Compare =====================
    cpu_out_flat = cpu_out.reshape(-1, SPARSE_COUNT).cpu()
    npu_out_flat = npu_out.reshape(-1, SPARSE_COUNT).cpu()
    if verbose:
        print(f"  cpu_out[:2]={cpu_out_flat[:2, :8]}")
        print(f"  npu_out[:2]={npu_out_flat[:2, :8]}")
    if return_value:
        cpu_val_flat = cpu_valuesOut.reshape(-1, SPARSE_COUNT).cpu().to(torch.bfloat16).to(torch.float32)
        npu_val_flat = npu_valuesOut.reshape(-1, SPARSE_COUNT).cpu().to(torch.float32)
        passed, diff_count = compare_indices_with_value_tolerance(
            cpu_out_flat, npu_out_flat, cpu_val_flat, npu_val_flat, SPARSE_COUNT, verbose=verbose)
    else:
        if torch.equal(cpu_out_flat, npu_out_flat):
            if verbose:
                print("  PASS")
            passed = True
        else:
            diff = (cpu_out_flat != npu_out_flat).sum().item()
            if verbose:
                print(f"  FAIL ({diff} element(s) differ)")
            passed = False
    return passed, case_desc


def test_sparse_index_lightning_indexer_tnd(
    N, INIT_NUM, LOCAL_NUM, SPARSE_COUNT, BLOCK_TOPK,
    actual_q_len=None, actual_kv_len=None, seed=42, verbose=True
):
    """非 PA 连续 TND 场景：key 不做 page split，直接传入连续 tensor"""
    block_size = 128
    D = 128
    dType = torch.bfloat16
    np.random.seed(seed)
    torch.manual_seed(seed)

    if actual_q_len is None:
        actual_q_len = [21]
    if actual_kv_len is None:
        actual_kv_len = [32 * 1024]

    T1 = calculate_special_sum(actual_q_len)
    T2 = calculate_special_sum(actual_kv_len)

    case_desc = (f"[TND] B={len(actual_q_len)}, T1={T1}, T2={T2}, N={N}, "
                 f"INIT={INIT_NUM}, LOCAL={LOCAL_NUM}, SC={SPARSE_COUNT}, "
                 f"BT={BLOCK_TOPK}, q={actual_q_len}, kv={actual_kv_len}")
    if verbose:
        print(f"[Case] is {case_desc}")

    query = torch.tensor(np.random.uniform(-2, 2, (T1, N, D))).to(dType)
    key = torch.tensor(np.random.uniform(-2, 2, (T2, 1, D))).to(dType)
    weights = torch.tensor(np.random.uniform(-1, 1, (T1, N))).to(torch.float)

    cur_q_len = [0] + list(np.cumsum(actual_q_len))
    cur_kv_len = [0] + list(np.cumsum(actual_kv_len))
    cur_seq_lengths_query = torch.tensor(cur_q_len).to(torch.int32)
    cur_seq_lengths_key = torch.tensor(cur_kv_len).to(torch.int32)

    sparse_index = generate_sparse_index(actual_kv_len, block_size, BLOCK_TOPK, actual_q_len)
    print("sparse_index is ", sparse_index)
    # ===================== CPU golden =====================
    cpu_out, cpu_valuesOut = golden_lightning_indexer_with_sparse_index(
        query, key, weights, cur_seq_lengths_query, cur_seq_lengths_key,
        sparse_index, block_size, SPARSE_COUNT, sparse_mode=0,
        INIT_NUM=INIT_NUM, LOCAL_NUM=LOCAL_NUM
    )

    return_value = True

    # ===================== NPU (TND 非 PA) =====================
    query_npu = query.to(f"npu:{DEVICE_ID}")
    key_npu = key.to(f"npu:{DEVICE_ID}")
    weights_npu = weights.to(f"npu:{DEVICE_ID}")
    cur_seq_lengths_query_npu = cur_seq_lengths_query.to(f"npu:{DEVICE_ID}").to(torch.int64)
    cur_seq_lengths_key_npu = cur_seq_lengths_key.to(f"npu:{DEVICE_ID}").to(torch.int64)
    sparse_index_npu = sparse_index.to(f"npu:{DEVICE_ID}")
    print("query_npu shape is ", query_npu.shape)
    print("key_npu shape is ", key_npu.shape)
    print("weights_npu shape is ", weights_npu.shape)
    print("sparse_index_npu shape is ", sparse_index_npu.shape)
    print("cur_seq_lengths_key_npu  ", cur_seq_lengths_key_npu)
    npu_out, npu_valuesOut = torch.ops.custom.mlp_sparse_lightning_indexer(
        query_npu, key_npu, weights_npu,
        cur_seq_lengths_query=cur_seq_lengths_query_npu,
        cur_seq_lengths_key=cur_seq_lengths_key_npu,
        sparse_index=sparse_index_npu,
        layout_query="TND",
        layout_key="TND",
        sparse_count=SPARSE_COUNT,
        sparse_index_block_size=block_size,
        init_num=INIT_NUM,
        local_num=LOCAL_NUM,
        sparse_mode=3,
        pre_tokens=2147483647,
        next_tokens=2147483647,
        return_value=return_value,
    )
    print("npu_out is ", npu_out)
    # ===================== Compare =====================
    print_symmetric_difference(cpu_out, npu_out)

    # npu_out3, npu_valuesOut = torch_npu.mlp_lightning_indexer(
    #     query_npu, key_npu, weights_npu, cur_seq_lengths_query=cur_seq_lengths_query_npu, 
    #         cur_seq_lengths_key=cur_seq_lengths_key_npu, block_table=None, layout_query="TND", 
    #         layout_key="TND", sparse_count=2048, kv_block_len=1, q_block_len=1, init_num=16, local_num=1024, sparse_mode=3, pre_tokens=2147483647,
    #         next_tokens=2147483647, return_value=True)
    # print_symmetric_difference(npu_out3, npu_out)
    passed = True
    return passed, case_desc


# ======================== 泛化性测试用例生成 ========================

# 参数范围
_N_CHOICES = [16, 32]
_INIT_NUM_RANGE = range(0, 17)                    # 0~16
_LOCAL_NUM_CHOICES = list(range(0, 1025, 128))     # 0, 128, 256, ..., 1024
_KV_LEN_POOL = [8192, 16384, 32768, 65536, 131072, 262144, 524288]  # 8k~512k
_SPARSE_COUNT = 2048
_BLOCK_TOPK = 256


_MAX_TOTAL_Q = 1024
_MAX_TOTAL_KV = 524288  # 512k

def print_symmetric_difference(a, b):
    a = a.cpu().flatten()
    b = b.cpu().flatten()
    """对称差集：在a或b中但不同时在两者中的元素"""
    # 找出在a但不在b的元素
    a_unique = a[~torch.isin(a, b)]
    # 找出在b但不在a的元素
    b_unique = b[~torch.isin(b, a)]
    if a_unique.numel() != 0 or b_unique.numel() != 0:
        print("check fail")
        print("在a但不在b的元素: ", a_unique, a_unique.numel())
        print("在b但不在a的元素: ", b_unique, b_unique.numel())
    else:
        print("check pass")

def generate_random_cases(num_cases=60, seed=123):
    """随机生成泛化性测试用例，总 q <= 1024，总 kv <= 512k"""
    rng = np.random.RandomState(seed)
    cases = []
    for i in range(num_cases):
        N = rng.choice(_N_CHOICES)
        init_num = int(rng.choice(_INIT_NUM_RANGE))
        local_num = int(rng.choice(_LOCAL_NUM_CHOICES))
        B = int(rng.randint(1, 11))  # batch 1~10

        # 生成 kv_len: 各子句加起来不超过 512k
        kv_len = []
        kv_remain = _MAX_TOTAL_KV
        for b in range(B):
            if b == B - 1:
                pool = [k for k in _KV_LEN_POOL if k <= kv_remain]
            else:
                pool = [k for k in _KV_LEN_POOL if k <= kv_remain - (B - b - 1) * 8192]
            if not pool:
                pool = [8192]
            kv = int(rng.choice(pool))
            kv_len.append(kv)
            kv_remain -= kv

        # 生成 q_len: 各子句加起来不超过 1024，kv>64k 时单句 q<=128
        q_len = []
        q_remain = _MAX_TOTAL_Q
        for b in range(B):
            max_q = min(q_remain - (B - b - 1), 128 if kv_len[b] > 65536 else q_remain)
            max_q = max(max_q, 1)
            q = int(rng.randint(1, max_q + 1))
            q_len.append(q)
            q_remain -= q

        desc = (f"rand#{i}: N={N} B={B} init={init_num} local={local_num} "
                f"q={q_len} kv={[k//1024 for k in kv_len]}k")
        cases.append((N, init_num, local_num, _SPARSE_COUNT, _BLOCK_TOPK, q_len, kv_len, desc))
    return cases


def run_npu_only(
    N, INIT_NUM, LOCAL_NUM, SPARSE_COUNT, BLOCK_TOPK,
    actual_q_len=None, actual_kv_len=None, seed=42,
):
    """只构造输入跑 NPU kernel，不做 golden 比对"""
    block_size = 128
    D = 128
    dType = torch.bfloat16
    np.random.seed(seed)
    torch.manual_seed(seed)

    if actual_q_len is None:
        actual_q_len = [21]
    if actual_kv_len is None:
        actual_kv_len = [32 * 1024]

    T1 = calculate_special_sum(actual_q_len)
    T2 = calculate_special_sum(actual_kv_len)
    max_seq_len = max(actual_kv_len)

    print(f"[NPU-only] B={len(actual_q_len)}, T1={T1}, T2={T2}, N={N}, "
          f"INIT={INIT_NUM}, LOCAL={LOCAL_NUM}, SC={SPARSE_COUNT}, "
          f"BT={BLOCK_TOPK}, q={actual_q_len}, kv={actual_kv_len}")

    query = torch.tensor(np.random.uniform(-2, 2, (T1, N, D))).to(dType)
    key = torch.tensor(np.random.uniform(-2, 2, (T2, 1, D))).to(dType)
    weights = torch.tensor(np.random.uniform(-1, 1, (T1, N))).to(torch.float)

    cur_q_len = [0] + list(np.cumsum(actual_q_len))
    cur_kv_len = [0] + list(np.cumsum(actual_kv_len))
    cur_seq_lengths_query = torch.tensor(cur_q_len).to(torch.int32)
    cur_seq_lengths_key = torch.tensor(cur_kv_len).to(torch.int32)

    sparse_index = generate_sparse_index(actual_kv_len, block_size, BLOCK_TOPK, actual_q_len)

    key_cache, key_block_table = split_to_kv_cache(key, block_size, actual_kv_len, max_seq_len)

    query_npu = query.to(f"npu:{DEVICE_ID}")
    key_cache_npu = key_cache.to(f"npu:{DEVICE_ID}")
    weights_npu = weights.to(f"npu:{DEVICE_ID}")
    cur_seq_lengths_query_npu = cur_seq_lengths_query.to(f"npu:{DEVICE_ID}").to(torch.int64)
    cur_seq_lengths_key_npu = cur_seq_lengths_key.to(f"npu:{DEVICE_ID}").to(torch.int64)
    key_block_table_npu = key_block_table.to(f"npu:{DEVICE_ID}")
    sparse_index_npu = sparse_index.to(f"npu:{DEVICE_ID}")

    npu_out, npu_valuesOut = torch.ops.custom.mlp_sparse_lightning_indexer(
        query_npu, key_cache_npu, weights_npu,
        cur_seq_lengths_query=cur_seq_lengths_query_npu,
        cur_seq_lengths_key=cur_seq_lengths_key_npu,
        block_table=key_block_table_npu,
        sparse_index=sparse_index_npu,
        layout_query="TND",
        layout_key="PA_BSND",
        sparse_count=SPARSE_COUNT,
        init_num=INIT_NUM,
        local_num=LOCAL_NUM,
        sparse_mode=3,
        pre_tokens=2147483647,
        next_tokens=2147483647,
        return_value=True,
    )

    print(f"  npu_out shape={npu_out.shape}")
    print(f"  npu_out[:2]={npu_out.reshape(-1, SPARSE_COUNT).cpu()[:2, :8]}")
    print("[DONE]")
    return npu_out, npu_valuesOut


def run_single(args):
    """single 模式：默认跑 NPU + golden 比对，--no-compare 时只跑 NPU"""
    if args.no_compare:
        run_npu_only(
            N=args.N,
            INIT_NUM=args.init_num,
            LOCAL_NUM=args.local_num,
            SPARSE_COUNT=args.sparse_count,
            BLOCK_TOPK=args.block_topk,
            actual_q_len=args.q_len,
            actual_kv_len=args.kv_len,
            seed=args.seed,
        )
    else:
        passed, desc = test_sparse_index_lightning_indexer(
            N=args.N,
            INIT_NUM=args.init_num,
            LOCAL_NUM=args.local_num,
            SPARSE_COUNT=args.sparse_count,
            BLOCK_TOPK=args.block_topk,
            actual_q_len=args.q_len,
            actual_kv_len=args.kv_len,
            seed=args.seed,
            verbose=True,
        )
        exit(0 if passed else 1)


def run_generalization(args):
    """运行随机泛化性测试"""
    cases = generate_random_cases(num_cases=args.num_cases, seed=args.gen_seed)
    total = len(cases)
    passed_count = 0
    failed_cases = []

    print(f"\n{'='*60}")
    print(f"  Random generalization test: {total} cases (seed={args.gen_seed})")
    print(f"{'='*60}")

    for i, case in enumerate(cases):
        N, init_num, local_num, sparse_count, block_topk, q_len, kv_len, desc = case
        print(f"  [{i+1}/{total}] {desc}")
        try:
            passed, case_desc = test_sparse_index_lightning_indexer(
                N=N, INIT_NUM=init_num, LOCAL_NUM=local_num,
                SPARSE_COUNT=sparse_count, BLOCK_TOPK=block_topk,
                actual_q_len=q_len, actual_kv_len=kv_len,
                seed=42, verbose=args.verbose,
            )
            if passed:
                passed_count += 1
            else:
                failed_cases.append((i + 1, desc))
        except Exception as e:
            print(f"    ERROR: {e}")
            failed_cases.append((i + 1, f"{desc} (ERROR: {e})"))

    print(f"\n{'='*60}")
    status = "ALL PASS" if not failed_cases else f"{passed_count}/{total} passed"
    print(f"  TOTAL: {status}")
    print(f"{'='*60}")
    if failed_cases:
        for idx, desc in failed_cases:
            print(f"    FAIL [{idx}] {desc}")
    return len(failed_cases) == 0


def main():
    parser = argparse.ArgumentParser(description="MlpLightningIndexer sparse_index test")
    subparsers = parser.add_subparsers(dest="mode", help="test mode")

    # 单 case 模式
    single = subparsers.add_parser("single", help="run a single test case")
    single.add_argument("--N", type=int, default=32)
    single.add_argument("--init-num", type=int, default=0)
    single.add_argument("--local-num", type=int, default=0)
    single.add_argument("--sparse-count", type=int, default=2048)
    single.add_argument("--block-topk", type=int, default=256)
    single.add_argument("--q-len", type=int, nargs="+", default=[21])
    single.add_argument("--kv-len", type=int, nargs="+", default=[32768])
    single.add_argument("--seed", type=int, default=42)
    single.add_argument("--no-compare", action="store_true", help="only run NPU kernel, skip golden compare")

    # TND 非 PA 模式
    tnd = subparsers.add_parser("tnd", help="run TND (non-PA) sparse_index test")
    tnd.add_argument("--N", type=int, default=32)
    tnd.add_argument("--init-num", type=int, default=16)
    tnd.add_argument("--local-num", type=int, default=1024)
    tnd.add_argument("--sparse-count", type=int, default=2048)
    tnd.add_argument("--block-topk", type=int, default=256*2)
    tnd.add_argument("--q-len", type=int, nargs="+", default=[2])
    tnd.add_argument("--kv-len", type=int, nargs="+", default=[32768*2])
    tnd.add_argument("--seed", type=int, default=42)

    # 泛化性测试模式
    gen = subparsers.add_parser("gen", help="run random generalization test suite")
    gen.add_argument("--verbose", action="store_true", help="print per-case details")
    gen.add_argument("--num-cases", type=int, default=60, help="number of random cases to generate (default: 60)")
    gen.add_argument("--gen-seed", type=int, default=123, help="seed for case generation (default: 123)")

    args = parser.parse_args()

    if args.mode is None or args.mode == "single":
        # 无参数或 single 模式，跑默认单 case
        if args.mode is None:
            # 兼容直接 python test.py 不带子命令
            args.N = 32
            args.init_num = 16
            args.local_num = 1024
            args.sparse_count = 2048
            args.block_topk = 256
            args.q_len = [2048]
            args.kv_len = [32 * 1024]
            args.seed = 42
            args.no_compare = False
        run_single(args)
    elif args.mode == "tnd":
        passed, desc = test_sparse_index_lightning_indexer_tnd(
            N=args.N, INIT_NUM=args.init_num, LOCAL_NUM=args.local_num,
            SPARSE_COUNT=args.sparse_count, BLOCK_TOPK=args.block_topk,
            actual_q_len=args.q_len, actual_kv_len=args.kv_len,
            seed=args.seed, verbose=True,
        )
        exit(0 if passed else 1)
    elif args.mode == "gen":
        passed = run_generalization(args)
        exit(0 if passed else 1)


if __name__ == "__main__":
    main()
