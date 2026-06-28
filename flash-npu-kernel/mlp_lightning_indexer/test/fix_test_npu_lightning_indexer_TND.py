# This program is free software, you can redistribute it and/or modify it.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import torch
import torch_npu
# import torchair
# import custom_ops
import numpy as np
import torch.nn as nn
import time

# torch.set_printoptions(profile="default", threshold=float("inf"), precision=9, sci_mode=False)
torch.set_printoptions(profile="default", precision=9, sci_mode=False)
# Q_BLOCK_LEN=2
# INIT_NUM=0
# LOCAL_NUM=0
DEFAULT_SCORE=-8888888
MAGIC_NUMBER=8866
DIGONAL_VALUE=9999999
WINDOW_VALUE=1111111

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

def print_diff(name, tensor_npu, tensor_gpu):
    max_diff = torch.max(torch.abs(tensor_npu.float() - tensor_gpu.float()))
    mean_diff = torch.mean(torch.abs(tensor_npu.float() - tensor_gpu.float()))
    print(f'{name} max diff: %.9f' % max_diff)
    print(f'{name} mean diff: %.9f' % mean_diff)
    return max_diff, mean_diff

DEVICE_ID = 11
torch_npu.npu.set_device(int(DEVICE_ID))

def _lightning_indexer(query, key, weights, actual_seq_lengths_query, actual_seq_lengths_key,
                       layout_query="TND", sparse_count=2048, sparse_mode=3, pre_tokens=2147483647, next_tokens=2147483647, 
                       return_value=False, BLOCK_LEN=1, INIT_NUM=0, LOCAL_NUM=0, Q_BLOCK_LEN=1):
    batch_size = actual_seq_lengths_query.shape[0]
    out_shape = list(query.shape)
    n2 = 1
    N = query.shape[-2]
    D = query.shape[-1]
    out_shape[-1] = sparse_count
    out_shape[-2] = n2
    # indice初始化为全-1
    out = torch.zeros(out_shape, dtype=torch.int32, device=query.device).reshape(-1, n2, sparse_count) - 1
    # value初始化为全0
    valuesOut = torch.zeros(out_shape, dtype=torch.float32, device=query.device).reshape(-1, n2, sparse_count)
    act_s1 = 0
    act_s2 = 0
    process_q_len = 0
    process_kv_len = 0
    for batch_id in range(batch_size):
        act_s1 = actual_seq_lengths_query[batch_id] - process_q_len
        act_s2 = actual_seq_lengths_key[batch_id] - process_kv_len
        now_q = query.reshape(-1, N, D)[process_q_len:process_q_len+act_s1, :, :].transpose(0, 1).to(torch.float32)
        now_k = key.reshape(-1, D)[process_kv_len:process_kv_len+act_s2, :].transpose(0, 1).to(torch.float32)
        now_weights = weights.reshape(-1, N, 1)[process_q_len:process_q_len+act_s1, :, :].transpose(0, 1).to(torch.float32)
        process_q_len += act_s1
        process_kv_len += act_s2
        # N,s1,D @ D,s2 -> N,s1,s2
        relu_out = torch.maximum(torch.matmul(now_q, now_k), torch.tensor(0))
        weight_out = relu_out * now_weights
        # N,s1,s2 -> s1,s2
        reduce_out = torch.sum(weight_out, dim=0)
        # sparse场景下三角置为-inf
        tmp_s1 = reduce_out.shape[0]
        tmp_s2 = reduce_out.shape[1]
        # print('tmp_s1, tmp_s2', tmp_s1, tmp_s2)

        atten_mask_u = torch.triu(torch.ones([act_s1, act_s2], dtype=torch.uint8), diagonal=(act_s2 - act_s1) + 1)
        reduce_out = reduce_out.masked_fill(atten_mask_u.to(torch.bool), DEFAULT_SCORE)
        # reduce_out.diagonal(offset=act_s2-act_s1)[:] = DIGONAL_VALUE  # 直接赋值给对角线的视图
        
        act_s2_align = (act_s2 + BLOCK_LEN - 1) // BLOCK_LEN * BLOCK_LEN
        reduce_out_tmp = torch.zeros([act_s1, act_s2_align], dtype=torch.float)
        reduce_out_tmp[:, :act_s2] = reduce_out
        reduce_out_tmp[:, act_s2:] = DEFAULT_SCORE
        # print('reduce_out_tmp', reduce_out_tmp)

        reduce_out_tmp = reduce_out_tmp.view(act_s1, act_s2_align // BLOCK_LEN, BLOCK_LEN)
        reduce_out_tmp = torch.sum(reduce_out_tmp, dim=-1, keepdim=False) / BLOCK_LEN
        reduce_out_tmp = torch.clip(reduce_out_tmp, max=WINDOW_VALUE/2) # logits做max clip，防止冲撞WINDOW_VALUE和DIGONAL_VALUE
        block_init_num = INIT_NUM // BLOCK_LEN
        block_local_num = LOCAL_NUM // BLOCK_LEN
        reduce_out_tmp[:, :block_init_num] = WINDOW_VALUE
        for s1 in range(act_s1):
            valid_s2_len = act_s2 - act_s1 + s1 + 1
            valid_s2_block = (valid_s2_len + BLOCK_LEN - 1) // BLOCK_LEN
            start_idx = (valid_s2_block - block_local_num) if (valid_s2_block - block_local_num) > 0  else 0
            reduce_out_tmp[s1, start_idx : valid_s2_block] = WINDOW_VALUE # 指定窗口区域赋予第二大分数
            reduce_out_tmp[s1, valid_s2_block - 1] = DIGONAL_VALUE # 对角线上元素赋予最大分数
            reduce_out_tmp[s1, valid_s2_block:] = DEFAULT_SCORE #对角线右侧直接赋予最小分数
        # print('reduce_out_tmp')
        # print(reduce_out_tmp)

        atten_mask_s2_scale_u = torch.triu(torch.ones([act_s1, act_s2_align], dtype=torch.uint8), diagonal=(act_s2 - act_s1) + 1)
        atten_mask_s2_scale_u = atten_mask_s2_scale_u.view(act_s1, act_s2_align // BLOCK_LEN, BLOCK_LEN)
        atten_mask_s2_scale_u = torch.sum(atten_mask_s2_scale_u, dim=-1, keepdim=False)
        atten_mask_s2_scale_u = (atten_mask_s2_scale_u == BLOCK_LEN).to(torch.uint8)
        # print('atten_mask_s2_scale_u')
        # print(atten_mask_s2_scale_u)
        # exit(0)

        reduce_out = reduce_out_tmp.contiguous()
        # print('reduce_out scale', reduce_out)

        sorted_value, sorted_indices = torch.sort(reduce_out, dim=1, descending=True, stable=True)
        sorted_indices = sorted_indices.masked_fill(atten_mask_s2_scale_u.to(torch.bool), -1)
        # print('sorted_indices', sorted_indices.shape)
        # print(sorted_indices)
        if Q_BLOCK_LEN > 1:
            for s1 in range(act_s1):
                if s1 % Q_BLOCK_LEN == 0:
                    # 对角线block当前在队首位置，我们把它放回到对角线上
                    valid_s2_len = act_s2 - act_s1 + s1 + 1
                    valid_s2_block = (valid_s2_len + BLOCK_LEN - 1) // BLOCK_LEN
                    valid_s2_block = valid_s2_block if valid_s2_block < sparse_count else sparse_count
                    tmp = sorted_indices[s1, 0].clone()
                    sorted_indices[s1, 0] = sorted_indices[s1, valid_s2_block - 1]
                    sorted_indices[s1, valid_s2_block - 1] = tmp
                    tmp = sorted_value[s1, 0].clone()
                    sorted_value[s1, 0] = sorted_value[s1, valid_s2_block - 1]
                    sorted_value[s1, valid_s2_block - 1] = tmp

                    for q in range(1, Q_BLOCK_LEN):
                        if s1+q >= act_s1:
                            break
                        valid_s2_len = act_s2 - act_s1 + s1 + 1 + q
                        valid_s2_block = (valid_s2_len + BLOCK_LEN - 1) // BLOCK_LEN
                        if valid_s2_block < sparse_count:
                            sorted_indices[s1+q, :] = sorted_indices[s1+q-1, :].clone()
                            sorted_indices[s1+q, valid_s2_block - 1] = valid_s2_block - 1
                        else:
                            sorted_indices[s1+q, :] = sorted_indices[s1+q-1, :].clone()
                            for i in range(q+1):
                                if i == q:
                                    tmp = sorted_indices[s1+q, sparse_count-1-i].clone()
                                sorted_indices[s1+q, sparse_count-1-i] = valid_s2_block-1-i
                                if i == q:
                                    sorted_indices[s1+q, 0] = tmp                                
                else:
                    continue
        else:
            for s1 in range(act_s1):
                valid_s2_len = act_s2 - act_s1 + s1 + 1
                valid_s2_block = (valid_s2_len + BLOCK_LEN - 1) // BLOCK_LEN
                valid_s2_block = valid_s2_block if valid_s2_block < sparse_count else sparse_count
                tmp = sorted_indices[s1, 0].clone()
                sorted_indices[s1, 0] = sorted_indices[s1, valid_s2_block - 1]
                sorted_indices[s1, valid_s2_block - 1] = tmp
                tmp = sorted_value[s1, 0].clone()
                sorted_value[s1, 0] = sorted_value[s1, valid_s2_block - 1]
                sorted_value[s1, valid_s2_block - 1] = tmp

        # print('sorted_value', sorted_value)
        # print('sorted_indices', sorted_indices)
        return_s2 = min(sparse_count, act_s2_align // BLOCK_LEN)
        out[process_q_len - act_s1:process_q_len, 0, :return_s2] = sorted_indices.to(torch.int32)[:, :return_s2]
        if return_value:
            valuesOut[process_q_len - act_s1:process_q_len, 0, :return_s2] = sorted_value[:, :return_s2]
        # print('out', out)
        # print('valuesOut', valuesOut)

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

def test_tnd_lightning_indexer_eager():
    path = '/workspace/dsa_debug/dump_content_layer17_rank191_klloss65231468.00.pt'
    tmp = torch.load(path)
    print(f"Loaded object type: {type(tmp)}")
    print(f"Is dict? {isinstance(tmp, dict)}")
    
    if isinstance(tmp, dict):
        print("\nDictionary keys:")
        for key in tmp.keys():
            value = tmp[key]
            print(f"  - '{key}': {type(value).__name__}", end="")
            if torch.is_tensor(value):
                print(f", shape: {value.shape}, dtype: {value.dtype}")
            elif isinstance(value, np.ndarray):
                print(f", shape: {value.shape}, dtype: {value.dtype}")
            else:
                print(f", value: {str(value)[:50]}...")
        
        # 查看具体张量信息
        print("\nTensor details:")
        for key, value in tmp.items():
            if torch.is_tensor(value):
                print(f"{key}: shape={value.shape}, dtype={value.dtype}, device={value.device}")
    else:
        # 如果不是字典，直接查看
        print(f"Value: {tmp}")
        if hasattr(tmp, 'shape'):
            print(f"Shape: {tmp.shape}")

    query = tmp['li_index_q']
    key = tmp['ag_index_k']
    weights = tmp['li_chunk_w']
    actual_seq_lengths_query = tmp['li_cu_seqlens_q']
    actual_seq_lengths_key = tmp['actual_seq_lengths_key']
    
    B = actual_seq_lengths_query.shape[0]
    T1 = query.shape[0]
    T2 = key.shape[0]
    N = 8
    SPARSE_COUNT = 2560
    BLOCK_LEN = 1
    INIT_NUM = 0
    LOCAL_NUM = 0
    Q_BLOCK_LEN = 1
    print('B, T1, T2, N, SPARSE_COUNT, actual_seq_lengths_query, actual_seq_lengths_key: ', B, T1, T2, N, SPARSE_COUNT, actual_seq_lengths_query, actual_seq_lengths_key)
    
    D = 128
    layout_query = 'TND'
    layout_key = 'TND'
    dType = torch.bfloat16
    np.random.seed(3)

    # query = torch.tensor(np.random.uniform(-2, 2, (T1, N, D))).to(dType)
    # key = torch.tensor(np.random.uniform(-2, 2, (T2, 1, D))).to(dType) # T 1 D
    # weights = torch.tensor(np.random.uniform(-1, 1, (T1, N))).to(dType)
    # # TND格式下，actual_seq_lengths_query为前缀和表示
    # actual_q_len = np.cumsum(actual_q_len)
    # actual_kv_len = np.cumsum(actual_kv_len)
    # actual_seq_lengths_query = torch.tensor(actual_q_len).to(torch.int32)
    # actual_seq_lengths_key = torch.tensor(actual_kv_len).to(torch.int32)
    # print('cu_q_len', actual_q_len)
    # print('cu_kv_len', actual_kv_len)

    sparse_count = SPARSE_COUNT 
    # sparse_count = 32
    sparse_mode = 3
    pre_tokens = 2147483647
    next_tokens = 2147483647 # TODO: 设成0还是？
    return_value = False
    cpu_out, cpu_valuesOut = _lightning_indexer(query, key, weights, actual_seq_lengths_query, actual_seq_lengths_key,
                                layout_query, sparse_count, sparse_mode, pre_tokens, next_tokens, return_value, BLOCK_LEN, INIT_NUM, LOCAL_NUM, Q_BLOCK_LEN)

    torch_npu.npu.set_device(int(DEVICE_ID))
    query = query.to("npu:%s" % DEVICE_ID)
    key = key.to("npu:%s" % DEVICE_ID)
    weights = weights.to("npu:%s" % DEVICE_ID)
    actual_seq_lengths_query = actual_seq_lengths_query.to("npu:%s" % DEVICE_ID)
    actual_seq_lengths_key = actual_seq_lengths_key.to("npu:%s" % DEVICE_ID)

    # start run custom ops
    npu_out, npu_valuesOut = torch_npu.mlp_lightning_indexer(
        query, key, weights, actual_seq_lengths_query=actual_seq_lengths_query, 
            actual_seq_lengths_key=actual_seq_lengths_key, block_table=None, layout_query=layout_query, 
            layout_key=layout_key, sparse_count=sparse_count, kv_block_len=BLOCK_LEN, q_block_len=Q_BLOCK_LEN, init_num=INIT_NUM, local_num=LOCAL_NUM, sparse_mode=sparse_mode, pre_tokens=pre_tokens,
            next_tokens=next_tokens, return_value=return_value)
    
    # compare result
    cpu_out = cpu_out.reshape(-1, sparse_count).cpu()
    npu_out = npu_out.reshape(-1, sparse_count).cpu()
    cpu_valuesOut = cpu_valuesOut.reshape(-1, sparse_count).cpu()
    npu_valuesOut = npu_valuesOut.reshape(-1, sparse_count).cpu()
    print('cpu_out', cpu_out)
    print('npu_out', npu_out)
    # print('cpu_valuesOut', cpu_valuesOut)
    # print('npu_valuesOut', npu_valuesOut)
    real_mask = (npu_valuesOut.to(torch.float) == float('-inf'))
    npu_valuesOut = npu_valuesOut.to(torch.float).masked_fill(real_mask.to(torch.bool), MAGIC_NUMBER)
    # print('npu_valuesOut', npu_valuesOut)

    gt_mask = (cpu_valuesOut == DEFAULT_SCORE)
    cpu_valuesOut = cpu_valuesOut.masked_fill(gt_mask.to(torch.bool), MAGIC_NUMBER)
    # print('cpu_valuesOut', cpu_valuesOut)

    # T1 = npu_out.shape[0]
    # for i in range(T1):
    #     for j in range(sparse_count):
    #         if npu_out[i][j] != cpu_out[i][j]:
    #             print(f"npu_out[{i},{j}] vs cpu_out[{i},{j}]:", npu_out[i][j], cpu_out[i][j])
    #             # print(f"npu_valuesOut[{i},{j}] vs cpu_valuesOut[{i},{j}]:", npu_valuesOut[i][j], cpu_valuesOut[i][j])
    #             print(f'======================== check indice fail ========================')

    # T1 = npu_valuesOut.shape[0]
    # for i in range(T1):
    #     for j in range(sparse_count):
    #         if npu_valuesOut[i][j] != cpu_valuesOut[i][j]:
    #             print(f"npu_valuesOut[{i},{j}] vs cpu_valuesOut[{i},{j}]:", npu_valuesOut[i][j], cpu_valuesOut[i][j])
    #             print(f'======================== check value fail ========================')
    
    # max_diff,_ = print_diff('out indice', npu_out.sum(), cpu_out.sum())
    # if (max_diff > 0):
        # print('watchout indice')
    print_symmetric_difference(npu_out, cpu_out)
    # _, mean_diff = print_diff('out value', npu_valuesOut, cpu_valuesOut)
    # if (mean_diff > 0.1):
    #     print('watchout value')

    return

    # 测性能
    for i in range(10):
        print('warmup i', i)
        npu_out, npu_valuesOut = torch_npu.mlp_lightning_indexer(
            query, key, weights, actual_seq_lengths_query=actual_seq_lengths_query, 
                actual_seq_lengths_key=actual_seq_lengths_key, block_table=None, layout_query=layout_query, 
                layout_key=layout_key, sparse_count=sparse_count, kv_block_len=BLOCK_LEN, q_block_len=Q_BLOCK_LEN, init_num=INIT_NUM, local_num=LOCAL_NUM, sparse_mode=sparse_mode, pre_tokens=pre_tokens,
                next_tokens=next_tokens, return_value=return_value)

    torch.npu.synchronize()
    runtimes = 50
    start = time.time()
    for i in range(runtimes):
        print('speed i', i)
        npu_out, npu_valuesOut = torch_npu.mlp_lightning_indexer(
            query, key, weights, actual_seq_lengths_query=actual_seq_lengths_query, 
                actual_seq_lengths_key=actual_seq_lengths_key, block_table=None, layout_query=layout_query, 
                layout_key=layout_key, sparse_count=sparse_count, kv_block_len=BLOCK_LEN, q_block_len=Q_BLOCK_LEN, init_num=INIT_NUM, local_num=LOCAL_NUM, sparse_mode=sparse_mode, pre_tokens=pre_tokens,
                next_tokens=next_tokens, return_value=return_value)

    torch.npu.synchronize()
    end = time.time()
    avg_time = (end - start) * 1000 / runtimes
    print('mlp_lightning_indexer avg time(ms): ', avg_time)

if __name__ == "__main__":
    test_tnd_lightning_indexer_eager()

