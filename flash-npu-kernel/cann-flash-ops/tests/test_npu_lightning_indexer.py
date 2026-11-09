# This program is free software, you can redistribute it and/or modify it.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import torch
import torch_npu
import flash_ops
import numpy as np
import torch.nn as nn

from torch_npu.testing.testcase import TestCase, run_tests

DEVICE_ID = 0
torch_npu.npu.set_device(int(DEVICE_ID))


def _get_data_from_pa_cache(key, block_table, act_s2):
    block_num, block_size, n2, d = key.shape
    if n2 != 1:
        raise ValueError("n2 only support 1")
    need_blcok_num = (act_s2 + block_size - 1) // block_size
    act_s2_align = need_blcok_num * block_size
    out = torch.zeros((act_s2_align, d), dtype=key.dtype, device=key.device)
    for i in range(need_blcok_num):
        out[i*block_size:(i+1)*block_size, :] = key[block_table[i], ...].reshape(block_size, d)

    return out[:act_s2, :]


def _lightning_indexer(query, key, weights, actual_seq_lengths_query, actual_seq_lengths_key, block_table,
                       layout_query="BSND", layout_key="PA_BSND", sparse_count=2048, sparse_mode=3, pre_tokens=9223372036854775807,
                       next_tokens=9223372036854775807, return_value=False, dType=torch.float16):
    batch_size = query.shape[0]
    if layout_query == "TND":
        batch_size = actual_seq_lengths_query.shape[0]
    out_shape = list(query.shape)
    n2 = key.shape[-2]
    d = query.shape[-1]
    n1 = query.shape[-2]
    out_shape[-1] = sparse_count
    out_shape[-2] = n2
    out = torch.zeros(out_shape, dtype=torch.int32, device=query.device).reshape(-1, n2, sparse_count) - 1

    valuesOut = torch.zeros(out_shape, dtype=torch.float32, device=query.device).reshape(-1, n2, sparse_count)
    act_s1 = 0
    act_s2 = 0
    process_q_len = 0
    for batch_id in range(batch_size):
        if actual_seq_lengths_query is None:
            act_s1 = query.shape[1]
        else:
            if layout_query == "TND":
                act_s1 = actual_seq_lengths_query[batch_id] - process_q_len
            else:
                act_s1 = actual_seq_lengths_query[batch_id]

        now_q = query.reshape(-1, n1, d)[process_q_len:process_q_len+act_s1, :, :].transpose(0, 1).to(torch.float32)
        now_weights = weights.reshape(-1, n1, 1)[process_q_len:process_q_len+act_s1, :, :] \
                    .transpose(0, 1).to(torch.float32)
        process_q_len += act_s1
        if layout_key == 'PA_BSND':
            act_s2 = actual_seq_lengths_key[batch_id]
            now_block_table = block_table[batch_id, :]
            now_k = _get_data_from_pa_cache(key, now_block_table, act_s2).transpose(0, 1).to(torch.float32)
        elif layout_key == 'BSND':
            now_k = key[batch_id, :, :, :].reshape(-1, d).transpose(0, 1).to(torch.float32)
        elif layout_key == 'TND':
            lastKeyPrefix = 0 if batch_id == 0 else actual_seq_lengths_key[batch_id - 1]
            curKeyPrefix = actual_seq_lengths_key[batch_id]
            now_k = key[lastKeyPrefix: curKeyPrefix, :, :].reshape(-1, d).transpose(0, 1).to(torch.float32)

        relu_out = torch.maximum(torch.matmul(now_q, now_k), torch.tensor(0))
        weight_out = relu_out * now_weights
        reduce_out = torch.sum(weight_out, dim=0)

        tmp_s1 = reduce_out.shape[0]
        tmp_s2 = reduce_out.shape[1]

        if sparse_mode == 3:
            for i in range(tmp_s1):
                reduce_out[-1-i, tmp_s2-i:] = float('-inf')

        sorted_value, sorted_indices = torch.sort(reduce_out, dim=1, descending=True)
        if sparse_mode == 3:
            for i in range(tmp_s1):
                thr = tmp_s2 - tmp_s1 + i + 1
                sorted_indices[i, thr: ] = -1

        return_s2 = min(sparse_count, tmp_s2)

        out[process_q_len - act_s1:process_q_len, 0, :return_s2] = sorted_indices.to(torch.int32)[:, :return_s2]

        if return_value:
            valuesOut[process_q_len - act_s1:process_q_len, 0, :return_s2] = sorted_value.to(dType)[:, :return_s2]

    out = out.reshape(out_shape)
    valuesOut = valuesOut.reshape(out_shape)
    return out, valuesOut


def _lightning_indexer_tnd_simple(query, key, weights, actual_seq_lengths_query, actual_seq_lengths_key,
                                  layout_query="TND", sparse_count=2048, sparse_mode=3, pre_tokens=9223372036854775807,
                                  next_tokens=9223372036854775807, return_value=False, dType=torch.float16):
    batch_size = actual_seq_lengths_query.shape[0]
    out_shape = list(query.shape)
    n2 = 1
    N = query.shape[-2]
    D = query.shape[-1]
    out_shape[-1] = sparse_count
    out_shape[-2] = n2
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
        relu_out = torch.maximum(torch.matmul(now_q, now_k), torch.tensor(0))
        weight_out = relu_out * now_weights
        reduce_out = torch.sum(weight_out, dim=0)
        tmp_s1 = reduce_out.shape[0]
        tmp_s2 = reduce_out.shape[1]

        atten_mask_u = torch.triu(torch.ones([act_s1, act_s2], dtype=torch.uint8), diagonal=(act_s2 - act_s1) + 1)
        reduce_out = reduce_out.masked_fill(atten_mask_u.to(torch.bool), float('-inf'))

        sorted_value, sorted_indices = torch.sort(reduce_out, dim=1, descending=True)
        sorted_indices = sorted_indices.masked_fill(atten_mask_u.to(torch.bool), -1)

        return_s2 = min(sparse_count, tmp_s2)
        out[process_q_len - act_s1:process_q_len, 0, :return_s2] = sorted_indices.to(torch.int32)[:, :return_s2]
        if return_value:
            valuesOut[process_q_len - act_s1:process_q_len, 0, :return_s2] = sorted_value.to(dType)[:, :return_s2]

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


class TestCustomLightningIndexer(TestCase):
    def test_bsnd_lightning_indexer_eager(self):
        b = 3
        s1 = 2
        s2 = 8192
        n1 = 32
        n2 = 1
        d = 128
        block_size = 256
        layout_query = 'BSND'
        layout_key = 'BSND'
        dType = torch.float16
        np.random.seed(0)
        query = torch.tensor(np.random.uniform(-10, 10, (b, s1, n1, d))).to(dType)
        if layout_key == 'PA_BSND':
            key = torch.tensor(np.random.uniform(-10, 10, (b*(s2//block_size), block_size, n2, d))).to(dType)
            block_table = torch.tensor([range(b*s2//block_size)], dtype=torch.int32).reshape(b, -1)
        else:
            key = torch.tensor(np.random.uniform(-10, 10, (b, s2, n2, d))).to(dType)
            block_table = None

        weights = torch.tensor(np.random.uniform(-1, 1, (b, s1, n1))).to(dType)
        actual_seq_lengths_query = torch.tensor(np.random.uniform(s1, s1, (b))).to(torch.int32)
        actual_seq_lengths_key = torch.tensor(np.random.uniform(s2, s2, (b))).to(torch.int32)

        sparse_count = 2048
        sparse_mode = 3
        pre_tokens = 9223372036854775807
        next_tokens = 9223372036854775807
        return_value = True

        cpu_out, cpu_valuesOut = _lightning_indexer(query, key, weights, actual_seq_lengths_query, actual_seq_lengths_key, block_table,
                                    layout_query, layout_key, sparse_count, sparse_mode, pre_tokens, next_tokens, return_value, dType)

        torch_npu.npu.set_device(int(DEVICE_ID))
        query = query.to("npu:%s" % DEVICE_ID)
        key = key.to("npu:%s" % DEVICE_ID)
        weights = weights.to("npu:%s" % DEVICE_ID)
        actual_seq_lengths_query = actual_seq_lengths_query.to("npu:%s" % DEVICE_ID)
        actual_seq_lengths_key = actual_seq_lengths_key.to("npu:%s" % DEVICE_ID)
        if layout_key == 'PA_BSND':
            block_table = block_table.to("npu:%s" % DEVICE_ID)

        print(f'======================== PTA eager BEGIN ========================')
        npu_out, npu_valuesOut = torch.ops.custom.npu_lightning_indexer(
            query, key, weights, actual_seq_lengths_query=actual_seq_lengths_query,
                actual_seq_lengths_key=actual_seq_lengths_key, block_table=block_table, layout_query=layout_query,
                layout_key=layout_key, sparse_count=sparse_count, sparse_mode=sparse_mode, pre_tokens=pre_tokens,
                next_tokens=next_tokens, return_value=return_value)

        cpu_out = cpu_out.reshape(-1, sparse_count).cpu()
        npu_out = npu_out.reshape(-1, sparse_count).cpu()
        cpu_valuesOut = cpu_valuesOut.reshape(-1, sparse_count).cpu()
        npu_valuesOut = npu_valuesOut.reshape(-1, sparse_count).cpu()

        t = npu_out.shape[0]
        for i in range(t):
            for j in range(sparse_count):
                if npu_out[i][j] != cpu_out[i][j]:
                    print("t K npu_out cpu_out = ", i, j, npu_out[i][j], cpu_out[i][j])

        t = npu_valuesOut.shape[0]
        for i in range(t):
            for j in range(sparse_count):
                if npu_valuesOut[i][j] != cpu_valuesOut[i][j]:
                    print("t K npu_valuesOut cpu_valuesOut = ", i, j, npu_valuesOut[i][j], cpu_valuesOut[i][j])

        print(f'======================== PTA eager FINISH ========================')


    def test_tnd_lightning_indexer_eager(self):
        b = 3
        t1 = 7
        t2 = 8192
        s2 = 8192
        n1 = 8

        n2 = 1
        d = 128
        block_size = 256
        layout_query = 'TND'
        layout_key = 'TND'
        dType = torch.float16
        np.random.seed(3)
        query = torch.tensor(np.random.uniform(-10, 10, (t1, n1, d))).to(dType)
        if layout_key == 'PA_BSND':
            key = torch.tensor(np.random.uniform(-10, 10, (b*(s2//block_size), block_size, n2, d))).to(dType)
            actual_seq_lengths_key = torch.tensor(np.random.uniform(s2, s2, (b))).to(torch.int32)
            block_table = torch.tensor([range(b*s2//block_size)], dtype=torch.int32).reshape(b, -1)
        else:
            key = torch.tensor(np.random.uniform(-10, 10, (t2, n2, d))).to(dType)
            actual_seq_lengths_key = torch.tensor([2048, 4096, 8192]).to(torch.int32)
            block_table = None

        weights = torch.tensor(np.random.uniform(-1, 1, (t1, n1))).to(dType)
        actual_seq_lengths_query = torch.tensor([1, 4, 7]).to(torch.int32)
        sparse_count = 2048
        sparse_mode = 3
        pre_tokens = 9223372036854775807
        next_tokens = 9223372036854775807
        return_value = True
        cpu_out, cpu_valuesOut = _lightning_indexer(query, key, weights, actual_seq_lengths_query, actual_seq_lengths_key, block_table,
                                    layout_query, layout_key, sparse_count, sparse_mode, pre_tokens, next_tokens, return_value)

        torch_npu.npu.set_device(int(DEVICE_ID))
        query = query.to("npu:%s" % DEVICE_ID)
        key = key.to("npu:%s" % DEVICE_ID)
        weights = weights.to("npu:%s" % DEVICE_ID)
        actual_seq_lengths_query = actual_seq_lengths_query.to("npu:%s" % DEVICE_ID)
        actual_seq_lengths_key = actual_seq_lengths_key.to("npu:%s" % DEVICE_ID)
        if layout_key == 'PA_BSND':
            block_table = block_table.to("npu:%s" % DEVICE_ID)

        print(f'======================== PTA eager BEGIN ========================')
        npu_out, npu_valuesOut = torch.ops.custom.npu_lightning_indexer(
            query, key, weights, actual_seq_lengths_query=actual_seq_lengths_query,
                actual_seq_lengths_key=actual_seq_lengths_key, block_table=block_table, layout_query=layout_query,
                layout_key=layout_key, sparse_count=sparse_count, sparse_mode=sparse_mode, pre_tokens=pre_tokens,
                next_tokens=next_tokens, return_value=return_value)

        cpu_out = cpu_out.reshape(-1, sparse_count).cpu()
        npu_out = npu_out.reshape(-1, sparse_count).cpu()
        cpu_valuesOut = cpu_valuesOut.reshape(-1, sparse_count).cpu()
        npu_valuesOut = npu_valuesOut.reshape(-1, sparse_count).cpu()

        t = npu_out.shape[0]
        for i in range(t):
            for j in range(sparse_count):
                if npu_out[i][j] != cpu_out[i][j]:
                    print("t K npu_out cpu_out = ", i, j, npu_out[i][j], cpu_out[i][j])

        t = npu_valuesOut.shape[0]
        for i in range(t):
            for j in range(sparse_count):
                if npu_valuesOut[i][j] != cpu_valuesOut[i][j]:
                    print("t K npu_valuesOut cpu_valuesOut = ", i, j, npu_valuesOut[i][j], cpu_valuesOut[i][j])
        print(f'======================== PTA eager FINISH ========================')


    def test_tnd_lightning_indexer_random_eager(self):
        B = 1
        T1 = 30
        N = 32
        actual_q_len, actual_kv_len = gen_seq_len(B, T1)
        T1 = np.sum(actual_q_len)
        T2 = np.sum(actual_kv_len)
        print('B, T1, T2, N: ', B, T1, T2, N)
        print('actual_q_len', actual_q_len)
        print('actual_kv_len', actual_kv_len)
        D = 128
        layout_query = 'TND'
        layout_key = 'TND'
        dType = torch.float16
        np.random.seed(3)

        query = torch.tensor(np.random.uniform(-10, 10, (T1, N, D))).to(dType)
        key = torch.tensor(np.random.uniform(-10, 10, (T2, 1, D))).to(dType)
        weights = torch.tensor(np.random.uniform(-1, 1, (T1, N))).to(dType)
        actual_q_len = np.cumsum(actual_q_len)
        actual_kv_len = np.cumsum(actual_kv_len)
        actual_seq_lengths_query = torch.tensor(actual_q_len).to(torch.int32)
        actual_seq_lengths_key = torch.tensor(actual_kv_len).to(torch.int32)

        sparse_count = 10
        sparse_mode = 3
        pre_tokens = 9223372036854775807
        next_tokens = 9223372036854775807
        return_value = True
        cpu_out, cpu_valuesOut = _lightning_indexer_tnd_simple(query, key, weights, actual_seq_lengths_query, actual_seq_lengths_key,
                                    layout_query, sparse_count, sparse_mode, pre_tokens, next_tokens, return_value)

        torch_npu.npu.set_device(int(DEVICE_ID))
        query = query.to("npu:%s" % DEVICE_ID)
        key = key.to("npu:%s" % DEVICE_ID)
        weights = weights.to("npu:%s" % DEVICE_ID)
        actual_seq_lengths_query = actual_seq_lengths_query.to("npu:%s" % DEVICE_ID)
        actual_seq_lengths_key = actual_seq_lengths_key.to("npu:%s" % DEVICE_ID)

        npu_out, npu_valuesOut = torch.ops.custom.npu_lightning_indexer(
            query, key, weights, actual_seq_lengths_query=actual_seq_lengths_query,
                actual_seq_lengths_key=actual_seq_lengths_key, block_table=None, layout_query=layout_query,
                layout_key=layout_key, sparse_count=sparse_count, sparse_mode=sparse_mode, pre_tokens=pre_tokens,
                next_tokens=next_tokens, return_value=return_value)

        cpu_out = cpu_out.reshape(-1, sparse_count).cpu()
        npu_out = npu_out.reshape(-1, sparse_count).cpu()
        cpu_valuesOut = cpu_valuesOut.reshape(-1, sparse_count).cpu()
        npu_valuesOut = npu_valuesOut.reshape(-1, sparse_count).cpu()
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


    def test_tnd_lightning_indexer_simple_eager(self):
        B = 1
        T1 = 30
        N = 32
        actual_q_len, actual_kv_len = gen_seq_len(B, T1)
        T1 = np.sum(actual_q_len)
        T2 = np.sum(actual_kv_len)
        print('B, T1, T2, N: ', B, T1, T2, N)
        D = 128
        layout_query = 'TND'
        layout_key = 'TND'
        dType = torch.float16
        np.random.seed(3)

        query = torch.tensor(np.random.uniform(-10, 10, (T1, N, D))).to(dType)
        key = torch.tensor(np.random.uniform(-10, 10, (T2, 1, D))).to(dType)
        weights = torch.tensor(np.random.uniform(-1, 1, (T1, N))).to(dType)
        actual_q_len = np.cumsum(actual_q_len)
        actual_kv_len = np.cumsum(actual_kv_len)
        actual_seq_lengths_query = torch.tensor(actual_q_len).to(torch.int32)
        actual_seq_lengths_key = torch.tensor(actual_kv_len).to(torch.int32)

        sparse_count = 10
        sparse_mode = 3
        pre_tokens = 9223372036854775807
        next_tokens = 9223372036854775807
        return_value = True
        cpu_out, cpu_valuesOut = _lightning_indexer_tnd_simple(query, key, weights, actual_seq_lengths_query, actual_seq_lengths_key,
                                    layout_query, sparse_count, sparse_mode, pre_tokens, next_tokens, return_value, dType)

        torch_npu.npu.set_device(int(DEVICE_ID))
        query = query.to("npu:%s" % DEVICE_ID)
        key = key.to("npu:%s" % DEVICE_ID)
        weights = weights.to("npu:%s" % DEVICE_ID)
        actual_seq_lengths_query = actual_seq_lengths_query.to("npu:%s" % DEVICE_ID)
        actual_seq_lengths_key = actual_seq_lengths_key.to("npu:%s" % DEVICE_ID)

        npu_out, npu_valuesOut = torch.ops.custom.npu_lightning_indexer(
            query, key, weights, actual_seq_lengths_query=actual_seq_lengths_query,
                actual_seq_lengths_key=actual_seq_lengths_key, block_table=None, layout_query=layout_query,
                layout_key=layout_key, sparse_count=sparse_count, sparse_mode=sparse_mode, pre_tokens=pre_tokens,
                next_tokens=next_tokens, return_value=return_value)

        cpu_out = cpu_out.reshape(-1, sparse_count).cpu()
        npu_out = npu_out.reshape(-1, sparse_count).cpu()
        cpu_valuesOut = cpu_valuesOut.reshape(-1, sparse_count).cpu()
        npu_valuesOut = npu_valuesOut.reshape(-1, sparse_count).cpu()
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
    run_tests()
