# This program is free software, you can redistribute it and/or modify it.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import torch
import torch_npu
import flash_ops  # 注册 torch.ops.custom 命名空间
# import torchair
# import custom_ops
import numpy as np
import torch.nn as nn
import time
import math
# torch.set_printoptions(profile="default", threshold=float("inf"), precision=9, sci_mode=False)
torch.set_printoptions(profile="default", precision=9, sci_mode=False)
# torch.set_printoptions(threshold=1000000, sci_mode=False) 
# Q_BLOCK_LEN=2
# INIT_NUM=0
# LOCAL_NUM=0
DEFAULT_SCORE=-8888888
MAGIC_NUMBER=8866
DIGONAL_VALUE=9999999
WINDOW_VALUE=1111111
RANDOM_BLOCK_TABLE=0

def split_to_kv_cache(input_tensor, block_size, cur_kv_seqlen, max_seq_len, end_batch):
    device = input_tensor.device
    # T = input_tensor.shape[0]
    B = len(cur_kv_seqlen)
    total_seq_block_cnt = 0
    for i in range(B):
        seq_block_cnt = math.ceil(cur_kv_seqlen[i] / block_size)
        if seq_block_cnt == 0:
            seq_block_cnt = 1
        total_seq_block_cnt += seq_block_cnt
    block_table_list = torch.randperm(total_seq_block_cnt) if RANDOM_BLOCK_TABLE else torch.arange(0, total_seq_block_cnt, 1)
    B2 = B + end_batch
    block_tables = torch.zeros(B2, math.ceil(max_seq_len / block_size)).to(torch.int32)
    input_shape = input_tensor.shape
    cache_tensor = torch.zeros(B2 * math.ceil(max_seq_len / block_size), block_size, input_shape[-2], input_shape[-1]).to(input_tensor.dtype).to(device)
    
    start = 0
    spilt_tensors_list = []
    total_seq_block_cnt = 0
    for i in range(B):
        seq_block_cnt = math.ceil(cur_kv_seqlen[i] / block_size)
        if seq_block_cnt == 0:
            seq_block_cnt = 1
        block_tables[i, :seq_block_cnt] = block_table_list[total_seq_block_cnt:seq_block_cnt + total_seq_block_cnt]
        kv_len = cur_kv_seqlen[i]
        if kv_len == 0:
            kv_len = 1
        for block_idx in range(seq_block_cnt):
            block_offset = block_idx * block_size
            seq_offset = block_offset + start
            if block_idx == seq_block_cnt-1:
                remain_size = kv_len - block_idx * block_size
                cache_tensor[block_tables[i][block_idx], :remain_size] = input_tensor[seq_offset : seq_offset + remain_size]
            else:
                cache_tensor[block_tables[i][block_idx], :block_size] = input_tensor[seq_offset : seq_offset + block_size]
        
        total_seq_block_cnt += seq_block_cnt
        tensor = input_tensor[start : start+kv_len]
        split_tensor = torch.split(tensor, block_size, dim=0)
        remaining_size = kv_len % block_size
        if remaining_size != 0:
            last_split = torch.narrow(tensor, 0, tensor.shape[0] - remaining_size, remaining_size)
            split_tensor = split_tensor[:-1] + (last_split,)
        start += kv_len
        spilt_tensors_list += split_tensor
    return cache_tensor, block_tables.to(device).detach()

def check_symmetric_difference(a, b):
    a = a.cpu().flatten()
    b = b.cpu().flatten()
    a_unique = a[~torch.isin(a, b)]
    b_unique = b[~torch.isin(b, a)]
    if a_unique.numel() != 0 or b_unique.numel() != 0:
        print("在a但不在b的元素: ", a_unique, a_unique.numel())
        print("在b但不在a的元素: ", b_unique, b_unique.numel())
        return False
    return True

def print_diff(name, tensor_npu, tensor_gpu):
    max_diff = torch.max(torch.abs(tensor_npu.float() - tensor_gpu.float()))
    mean_diff = torch.mean(torch.abs(tensor_npu.float() - tensor_gpu.float()))
    print(f'{name} max diff: %.9f' % max_diff)
    print(f'{name} mean diff: %.9f' % mean_diff)
    return max_diff, mean_diff

DEVICE_ID = 12
torch_npu.npu.set_device(int(DEVICE_ID))

def _lightning_indexer(query_pos, key_max, weights, cur_seq_lengths_query, cur_seq_lengths_key,
                       layout_query="TND", sparse_count=2048, sparse_mode=3, pre_tokens=2147483647, next_tokens=2147483647, 
                       return_value=False, BLOCK_LEN=1, INIT_NUM=0, LOCAL_NUM=0, Q_BLOCK_LEN=1, origin_seq_k = 1):
    if BLOCK_LEN == 1:
        seq_q_len_for_one_mask = 1
    else:
        seq_q_len_for_one_mask = 128 // BLOCK_LEN
    batch_size = cur_seq_lengths_query.shape[0] - 1
    out_shape = list(query_pos.shape)
    n2 = 1
    N = query_pos.shape[-2]
    D = query_pos.shape[-1]
    out_shape[-1] = sparse_count
    out_shape[-2] = n2
    # indice初始化为全-1
    out = torch.zeros(out_shape, dtype=torch.int32, device=query_pos.device).reshape(-1, n2, sparse_count) - 1
    # value初始化为全0
    valuesOut = torch.zeros(out_shape, dtype=torch.float32, device=query_pos.device).reshape(-1, n2, sparse_count)
    act_s1 = 0
    act_s2 = 0
    process_q_len = 0
    process_kv_len = 0
    for batch_id in range(batch_size):
        act_s1 = cur_seq_lengths_query[batch_id + 1] - cur_seq_lengths_query[batch_id]
        act_s2 = cur_seq_lengths_key[batch_id + 1] - cur_seq_lengths_key[batch_id]
        print("act_s2 ", act_s2, " seq_q_len_for_one_mask ", seq_q_len_for_one_mask, " origin_seq_k ", origin_seq_k)
        if act_s1 < 0:
            break
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
        now_q = query_pos.reshape(-1, N, D)[process_q_len:process_q_len+act_s1, :, :].transpose(0, 1).to(torch.float32)
        now_k = key_max.reshape(-1, D)[process_kv_len:process_kv_len+act_s2, :].transpose(0, 1).to(torch.float32)
        now_weights = weights.reshape(-1, N, 1)[process_q_len:process_q_len+act_s1, :, :].transpose(0, 1).to(torch.float32)
        process_q_len += act_s1
        process_kv_len += act_s2
        # N,s1,D @ D,s2 -> N,s1,s2
        # relu_out = torch.maximum(torch.matmul(now_q, now_k), torch.tensor(0))
        # relu_out2 = torch.maximum(torch.matmul(now_q2, now_k2), torch.tensor(0))
        mm_res1 = torch.matmul(now_q, now_k)
        relu_out = torch.maximum(mm_res1, torch.tensor(0))
        print("here relu_out shape is ", relu_out.shape)
        print("now_weights shape is ", now_weights.shape)
        weight_out = relu_out * now_weights
        reduce_out = torch.sum(weight_out, dim=0)
        # sparse场景下三角置为-inf
        # Q 第 s1 行能看到的最后一个 KV token 是 origin_seq_k - distToEnd，
        # 转成 token 粒度有效长度：(origin_seq_k - distToEnd) // seq_q_len_for_one_mask * seq_q_len_for_one_mask
        # 转成 block 粒度有效块数：(origin_seq_k - distToEnd) // seq_q_len_for_one_mask
        if sparse_mode == 3:
            atten_mask_u = torch.zeros([act_s1, act_s2], dtype=torch.uint8)
            for s1 in range(act_s1):
                distToEnd = act_s1 -1 - s1
                valid_len = (origin_seq_k - distToEnd + seq_q_len_for_one_mask -1) // seq_q_len_for_one_mask #* seq_q_len_for_one_mask
                valid_len = min(valid_len, act_s2)
                if valid_len < act_s2:
                    atten_mask_u[s1, valid_len:] = 1
            reduce_out = reduce_out.masked_fill(atten_mask_u.to(torch.bool), DEFAULT_SCORE)
            # reduce_out.diagonal(offset=act_s2-act_s1)[:] = DIGONAL_VALUE  # 直接赋值给对角线的视图
        print('第一次mask之后 reduce_out', reduce_out)
        act_s2_align = (act_s2 + BLOCK_LEN - 1) // BLOCK_LEN * BLOCK_LEN
        reduce_out_tmp = torch.zeros([act_s1, act_s2_align], dtype=torch.float)
        reduce_out_tmp[:, :act_s2] = reduce_out
        reduce_out_tmp[:, act_s2:] = DEFAULT_SCORE

        reduce_out_tmp = reduce_out_tmp.view(act_s1, act_s2_align // BLOCK_LEN, BLOCK_LEN)
        reduce_out_tmp = torch.sum(reduce_out_tmp, dim=-1, keepdim=False) / BLOCK_LEN
        reduce_out_tmp = torch.clip(reduce_out_tmp, max=WINDOW_VALUE/2) # logits做max clip，防止冲撞WINDOW_VALUE和DIGONAL_VALUE
        block_init_num = INIT_NUM // BLOCK_LEN
        block_local_num = LOCAL_NUM // BLOCK_LEN
        reduce_out_tmp[:, :block_init_num] = WINDOW_VALUE
        for s1 in range(act_s1):
            if sparse_mode == 3:
                distToEnd = act_s1 -1 - s1
                valid_s2_block = (origin_seq_k - distToEnd + seq_q_len_for_one_mask * BLOCK_LEN -1) // (seq_q_len_for_one_mask * BLOCK_LEN)
                valid_s2_block = min(valid_s2_block, act_s2_align // BLOCK_LEN) #后一个是640//8=80
                valid_s2_len = valid_s2_block * BLOCK_LEN  # 仅用于 local window 起始计算
            else:
                valid_s2_len = act_s2
                valid_s2_block = (valid_s2_len + BLOCK_LEN - 1) // BLOCK_LEN
            start_idx = (valid_s2_block - block_local_num) if (valid_s2_block - block_local_num) > 0  else 0
            reduce_out_tmp[s1, start_idx : valid_s2_block] = WINDOW_VALUE # 指定窗口区域赋予第二大分数
            # reduce_out_tmp[s1, valid_s2_block - 1] = DIGONAL_VALUE # 对角线上元素赋予最大分数
            if sparse_mode == 3:
                reduce_out_tmp[s1, valid_s2_block:] = DEFAULT_SCORE #对角线右侧直接赋予最小分数
        print('init local 必选之后 reduce_out_tmp shape', reduce_out_tmp.shape)

        if sparse_mode == 3:
            atten_mask_s2_scale_u = torch.zeros([act_s1, act_s2_align // BLOCK_LEN], dtype=torch.uint8)
            for s1 in range(act_s1):
                distToEnd = act_s1 -1 - s1
                valid_block = (origin_seq_k - distToEnd + seq_q_len_for_one_mask * BLOCK_LEN -1) // (seq_q_len_for_one_mask * BLOCK_LEN)
                valid_block = min(valid_block, act_s2_align // BLOCK_LEN)
                if valid_block < act_s2_align // BLOCK_LEN:
                    atten_mask_s2_scale_u[s1, valid_block:] = 1
        else:
            atten_mask_s2_scale_u = torch.zeros([act_s1, act_s2_align // BLOCK_LEN], dtype=torch.uint8)

        reduce_out = reduce_out_tmp.contiguous()

        sorted_value, sorted_indices = torch.sort(reduce_out, dim=1, descending=True, stable=True)
        sorted_indices = sorted_indices.masked_fill(atten_mask_s2_scale_u.to(torch.bool), -1)

        return_s2 = min(sparse_count, act_s2_align // BLOCK_LEN)
        out[process_q_len - act_s1:process_q_len, 0, :return_s2] = sorted_indices.to(torch.int32)[:, :return_s2]
        if return_value:
            valuesOut[process_q_len - act_s1:process_q_len, 0, :return_s2] = sorted_value[:, :return_s2]

    out = out.reshape(out_shape)
    valuesOut = valuesOut.reshape(out_shape)
    return out, valuesOut

def gen_seq_len(B, T1):
    random_numbers = np.random.rand(B)
    normalized_numbers = random_numbers / random_numbers.sum()
    q_len_tmp = [int(T1 * normalized_numbers[i]) for i in range(B)]
    q_len = [1 if q_len_tmp[i] == 0 else q_len_tmp[i] for i in range(B)]
    kv_len = [q_len[i] + np.random.randint(1000, 8000) for i in range(B)]
    # kv_len = [q_len[i] + 2622 for i in range(B)]
    # kv_len = [q_len[i]*4 for i in range(B)]
    return q_len, kv_len


def calculate_special_sum(lst):
    """
    计算特殊列表的和：
    - 0元素长度为1（即每个0贡献1到总和）
    - 非零元素为它本身的值
    
    参数:
        lst: 输入列表，如 [0, 0, 1, 10]
    
    返回:
        计算后的总和，如 [0, 0, 1, 10] 返回 13 (0->1 + 0->1 + 1->1 + 10->10)
    
    示例:
        >>> calculate_special_sum([0, 0, 1, 10])
        13
        >>> calculate_special_sum([1, 2, 3])
        6
        >>> calculate_special_sum([0, 0, 0])
        3
    """
    if not lst:  # 处理空列表
        return 0
    
    total = 0
    for element in lst:
        if element == 0:
            total += 1  # 0元素长度为1
        else:
            total += element  # 非零元素为它本身的值
    
    return total

def test_tnd_lightning_indexer_eager(B, origin_seq_k, N, BLOCK_LEN, INIT_NUM, LOCAL_NUM, Q_BLOCK_LEN, SPARSE_COUNT):
    assert INIT_NUM % BLOCK_LEN == 0, "INIT_NUM invalid"
    assert LOCAL_NUM % BLOCK_LEN == 0, "LOCAL_NUM invalid"
    # actual_q_len, actual_kv_len = gen_seq_len(B, T1)
    pooling_len = 128 // BLOCK_LEN
    T2 = (origin_seq_k + pooling_len - 1) // pooling_len # 上取整
    actual_q_len = [2048]
    actual_kv_len = [T2]
    T1 = calculate_special_sum(actual_q_len)#np.sum(actual_q_len) #重新更新T1
    T2 = calculate_special_sum(actual_kv_len)#np.sum(actual_kv_len) #重新更新T2
    print('B, T1, T2, N, BLOCK_LEN, INIT_NUM, LOCAL_NUM, Q_BLOCK_LEN, SPARSE_COUNT: ', B, T1, T2, N, BLOCK_LEN, INIT_NUM, LOCAL_NUM, Q_BLOCK_LEN, SPARSE_COUNT)
    print('actual_q_len', actual_q_len)
    print('actual_kv_len', actual_kv_len)
    D = 128
    block_size = 128
    max_seq_len = max(actual_kv_len)
    layout_query = 'TND'
    layout_key = 'TND'
    dType = torch.bfloat16
    np.random.seed(3)

    query_pos = torch.tensor(np.random.uniform(-2, 2, (T1, N, D))).to(dType)
    key_max = torch.tensor(np.random.uniform(-2, 2, (T2, 1, D))).to(dType) # T 1 D
    weights = torch.tensor(np.random.uniform(-1, 1, (T1, N))).to(torch.float)
    # TND格式下，actual_seq_lengths_query为前缀和表示
    end_batch = 0
    cur_q_len = [0] + list(np.cumsum(actual_q_len)) + [-1] * end_batch
    cur_kv_len = [0] + list(np.cumsum(actual_kv_len)) + [-1] * end_batch
    cur_seq_lengths_query = torch.tensor(cur_q_len).to(torch.int32)
    cur_seq_lengths_key = torch.tensor(cur_kv_len).to(torch.int32)
    org_kv_len = [0] + [origin_seq_k]
    org_seq_lengths_key = torch.tensor(org_kv_len).to(torch.int32)
    print('cu_q_len', cur_q_len)
    print('cu_kv_len', cur_kv_len)
    print('org_seq_lengths_key', org_seq_lengths_key)

    sparse_count = SPARSE_COUNT 
    # sparse_count = 32
    sparse_mode = 3
    pre_tokens = 2147483647
    next_tokens = 2147483647 # TODO: 设成0还是？
    return_value = True
    cpu_out, cpu_valuesOut = _lightning_indexer(query_pos, key_max, weights, cur_seq_lengths_query, cur_seq_lengths_key,
                                layout_query, sparse_count, sparse_mode, pre_tokens, next_tokens, return_value, BLOCK_LEN, INIT_NUM, LOCAL_NUM, Q_BLOCK_LEN,
                                origin_seq_k)

    torch_npu.npu.set_device(int(DEVICE_ID))
    query_pos = query_pos.to("npu:%s" % DEVICE_ID)
    key_max = key_max.to("npu:%s" % DEVICE_ID)
    weights = weights.to("npu:%s" % DEVICE_ID)
    cur_seq_lengths_query = cur_seq_lengths_query.to("npu:%s" % DEVICE_ID)
    cur_seq_lengths_key = cur_seq_lengths_key.to("npu:%s" % DEVICE_ID)
    org_seq_lengths_key = org_seq_lengths_key.to("npu:%s" % DEVICE_ID)

    # start run custom ops
    if layout_key == "TND":
        print("start to run mlp")
        npu_out, npu_valuesOut = torch.ops.custom.mlp_compressed_lightning_indexer(
            query_pos, query_pos, key_max, key_max, weights, cur_seq_lengths_query=cur_seq_lengths_query, 
                cur_seq_lengths_key=org_seq_lengths_key, block_table=None, layout_query=layout_query, 
                layout_key=layout_key, sparse_count=sparse_count, kv_block_len=BLOCK_LEN, q_block_len=Q_BLOCK_LEN, init_num=INIT_NUM, local_num=LOCAL_NUM, sparse_mode=sparse_mode, pre_tokens=pre_tokens,
                next_tokens=next_tokens, return_value=return_value)
    else:
        key_cache, key_block_table = split_to_kv_cache(key, block_size, actual_kv_len, max_seq_len, end_batch)
        npu_out, npu_valuesOut = torch_npu.mlp_lightning_indexer(
            query, key_cache, weights, cur_seq_lengths_query=cur_seq_lengths_query, 
                cur_seq_lengths_key=cur_seq_lengths_key, block_table=key_block_table, 
                layout_query=layout_query, 
                layout_key=layout_key, 
                sparse_count=sparse_count, kv_block_len=BLOCK_LEN, q_block_len=Q_BLOCK_LEN, init_num=INIT_NUM, local_num=LOCAL_NUM, sparse_mode=sparse_mode, return_value=return_value)
    
    # compare result
    cpu_out = cpu_out.reshape(-1, sparse_count).cpu()
    npu_out = npu_out.reshape(-1, sparse_count).cpu()
    print('cpu_out', cpu_out)
    print('npu_out', npu_out)
    if return_value:
        cpu_valuesOut = cpu_valuesOut.reshape(-1, sparse_count).cpu()
        npu_valuesOut = npu_valuesOut.reshape(-1, sparse_count).cpu()
        print('cpu_valuesOut', cpu_valuesOut)
        print('npu_valuesOut', npu_valuesOut)
        real_mask = (npu_valuesOut.to(torch.float) == float('-inf'))
        npu_valuesOut = npu_valuesOut.to(torch.float).masked_fill(real_mask.to(torch.bool), MAGIC_NUMBER)

        gt_mask = (cpu_valuesOut == DEFAULT_SCORE)
        cpu_valuesOut = cpu_valuesOut.masked_fill(gt_mask.to(torch.bool), MAGIC_NUMBER)


    for i in range(T1):
        assert check_symmetric_difference(npu_out[i], cpu_out[i]), f"symmetric difference check failed at row {i}"

    return

    # 测性能
    for i in range(10):
        print('warmup i', i)
        npu_out, npu_valuesOut = torch_npu.mlp_lightning_indexer(
            query, key, weights, cur_seq_lengths_query=cur_seq_lengths_query, 
                cur_seq_lengths_key=cur_seq_lengths_key, block_table=None, layout_query=layout_query, 
                layout_key=layout_key, sparse_count=sparse_count, kv_block_len=BLOCK_LEN, q_block_len=Q_BLOCK_LEN, init_num=INIT_NUM, local_num=LOCAL_NUM, sparse_mode=sparse_mode, pre_tokens=pre_tokens,
                next_tokens=next_tokens, return_value=return_value)

    torch.npu.synchronize()
    runtimes = 50
    start = time.time()
    for i in range(runtimes):
        print('speed i', i)
        npu_out, npu_valuesOut = torch_npu.mlp_lightning_indexer(
            query, key, weights, cur_seq_lengths_query=cur_seq_lengths_query, 
                cur_seq_lengths_key=cur_seq_lengths_key, block_table=None, layout_query=layout_query, 
                layout_key=layout_key, sparse_count=sparse_count, kv_block_len=BLOCK_LEN, q_block_len=Q_BLOCK_LEN, init_num=INIT_NUM, local_num=LOCAL_NUM, sparse_mode=sparse_mode, pre_tokens=pre_tokens,
                next_tokens=next_tokens, return_value=return_value)

    torch.npu.synchronize()
    end = time.time()
    avg_time = (end - start) * 1000 / runtimes
    print('mlp_lightning_indexer avg time(ms): ', avg_time)

if __name__ == "__main__":


    test_tnd_lightning_indexer_eager(1, 32768+888, 32, 8, 8, 72, 1, 256)
