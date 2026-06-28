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

DEVICE_ID = 0
torch_npu.npu.set_device(int(DEVICE_ID))


def _lightning_indexer(query, key, weights, actual_seq_lengths_query, actual_seq_lengths_key,
                       layout_query="TND", sparse_count=2048, sparse_mode=3, pre_tokens=2147483647, next_tokens=2147483647, return_value=False, dType=torch.float16):
    batch_size = actual_seq_lengths_query.shape[0]
    out_shape = list(query.shape)
    n2 = 1
    N = query.shape[-2]
    D = query.shape[-1]
    out_shape[-1] = sparse_count
    out_shape[-2] = n2
    # 初始化为全-1
    out = torch.zeros(out_shape, dtype=torch.int32, device=query.device).reshape(-1, n2, sparse_count) - 1
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
        print('tmp_s1, tmp_s2', tmp_s1, tmp_s2)

        atten_mask_u = torch.triu(torch.ones([act_s1, act_s2], dtype=torch.uint8), diagonal=(act_s2 - act_s1) + 1)
        reduce_out = reduce_out.masked_fill(atten_mask_u.to(torch.bool), float('-inf'))
        print('reduce_out', reduce_out)

        sorted_value, sorted_indices = torch.sort(reduce_out, dim=1, descending=True)
        sorted_indices = sorted_indices.masked_fill(atten_mask_u.to(torch.bool), -1)

        return_s2 = min(sparse_count, tmp_s2)
        out[process_q_len - act_s1:process_q_len, 0, :return_s2] = sorted_indices.to(torch.int32)[:, :return_s2]
        if return_value:
            valuesOut[process_q_len - act_s1:process_q_len, 0, :return_s2] = sorted_value.to(dType)[:, :return_s2]
        print('out', out)
        print('valuesOut', valuesOut)

    out = out.reshape(out_shape)
    valuesOut = valuesOut.reshape(out_shape)
    return out, valuesOut

def gen_seq_len(B, T1):
    random_numbers = np.random.rand(B)
    normalized_numbers = random_numbers / random_numbers.sum()
    q_len_tmp = [int(T1 * normalized_numbers[i]) for i in range(B)]
    q_len = [1 if q_len_tmp[i] == 0 else q_len_tmp[i] for i in range(B)]
    kv_len = [q_len[i] + np.random.randint(100, 2000) for i in range(B)]
    kv_len = [q_len[i] for i in range(B)]
    return q_len, kv_len

def test_tnd_lightning_indexer_eager(B, T1, N):
    actual_q_len, actual_kv_len = gen_seq_len(B, T1)
    T1 = np.sum(actual_q_len) #重新更新T1
    T2 = np.sum(actual_kv_len) #重新更新T2
    print('B, T1, T2, N: ', B, T1, T2, N)
    print('actual_q_len', actual_q_len)
    print('actual_kv_len', actual_kv_len)
    D = 128
    layout_query = 'TND'
    layout_key = 'TND'
    dType = torch.float16
    np.random.seed(3)

    query = torch.tensor(np.random.uniform(-10, 10, (T1, N, D))).to(dType)
    key = torch.tensor(np.random.uniform(-10, 10, (T2, 1, D))).to(dType) # T 1 D 
    weights = torch.tensor(np.random.uniform(-1, 1, (T1, N))).to(dType)
    # TND格式下，actual_seq_lengths_query为前缀和表示
    actual_q_len = np.cumsum(actual_q_len)
    actual_kv_len = np.cumsum(actual_kv_len)
    actual_seq_lengths_query = torch.tensor(actual_q_len).to(torch.int32)
    actual_seq_lengths_key = torch.tensor(actual_kv_len).to(torch.int32)
    print('cu_q_len', actual_q_len)
    print('cu_kv_len', actual_kv_len)

    sparse_count = 10
    sparse_mode = 3
    pre_tokens = 2147483647
    next_tokens = 2147483647 # TODO: 设成0还是？
    return_value = True
    cpu_out, cpu_valuesOut = _lightning_indexer(query, key, weights, actual_seq_lengths_query, actual_seq_lengths_key,
                                layout_query, sparse_count, sparse_mode, pre_tokens, next_tokens, return_value)

    torch_npu.npu.set_device(int(DEVICE_ID))
    query = query.to("npu:%s" % DEVICE_ID)
    key = key.to("npu:%s" % DEVICE_ID)
    weights = weights.to("npu:%s" % DEVICE_ID)
    actual_seq_lengths_query = actual_seq_lengths_query.to("npu:%s" % DEVICE_ID)
    actual_seq_lengths_key = actual_seq_lengths_key.to("npu:%s" % DEVICE_ID)

    # start run custom ops
    npu_out, npu_valuesOut = torch_npu.npu_lightning_indexer(
        query, key, weights, actual_seq_lengths_query=actual_seq_lengths_query, 
            actual_seq_lengths_key=actual_seq_lengths_key, block_table=None, layout_query=layout_query, 
            layout_key=layout_key, sparse_count=sparse_count, sparse_mode=sparse_mode, pre_tokens=pre_tokens,
            next_tokens=next_tokens, return_value=return_value)
    
    # compare result
    cpu_out = cpu_out.reshape(-1, sparse_count).cpu()
    npu_out = npu_out.reshape(-1, sparse_count).cpu()
    cpu_valuesOut = cpu_valuesOut.reshape(-1, sparse_count).cpu()
    npu_valuesOut = npu_valuesOut.reshape(-1, sparse_count).cpu()
    print('npu_out', npu_out)
    print('npu_valuesOut', npu_valuesOut)
    T1 = npu_out.shape[0]
    for i in range(T1):
        for j in range(sparse_count):
            if npu_out[i][j] != cpu_out[i][j]:
                print("T1 K npu_out cpu_out = ", i, j, npu_out[i][j], cpu_out[i][j])
                print(f'======================== check indice fail ========================')

    T1 = npu_valuesOut.shape[0]
    for i in range(T1):
        for j in range(sparse_count):
            if npu_valuesOut[i][j] != cpu_valuesOut[i][j]:
                print("T1 K npu_valuesOut cpu_valuesOut = ", i, j, npu_valuesOut[i][j], cpu_valuesOut[i][j])
                print(f'======================== check value fail ========================')

if __name__ == "__main__":
    test_tnd_lightning_indexer_eager(1, 30, 32)
