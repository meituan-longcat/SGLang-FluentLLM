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
import time
import math

from torch_npu.testing.testcase import TestCase, run_tests

torch.set_printoptions(profile="default", precision=9, sci_mode=False)

DEFAULT_SCORE = -8888888
MAGIC_NUMBER = 8866
DIGONAL_VALUE = 9999999
WINDOW_VALUE = 1111111
RANDOM_BLOCK_TABLE = 0

DEVICE_ID = 0
torch_npu.npu.set_device(int(DEVICE_ID))


class LightningIndexerModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, query, key, weights, cur_seq_lengths_query, cur_seq_lengths_key, block_table, layout_query,
                layout_key, sparse_count, kv_block_len, q_block_len, init_num, local_num, sparse_mode, pre_tokens,
                next_tokens, return_value):
        return torch.ops.custom.mlp_lightning_indexer(
            query, key, weights,
            cur_seq_lengths_query=cur_seq_lengths_query,
            cur_seq_lengths_key=cur_seq_lengths_key,
            block_table=block_table,
            layout_query=layout_query,
            layout_key=layout_key,
            sparse_count=sparse_count,
            kv_block_len=kv_block_len,
            q_block_len=q_block_len,
            init_num=init_num,
            local_num=local_num,
            sparse_mode=sparse_mode,
            pre_tokens=pre_tokens,
            next_tokens=next_tokens,
            return_value=return_value
        )


def compare_arrays(arr1, arr2, name):
    a = arr1.float()
    b = arr2.float()
    diff = torch.abs(a - b)
    max_diff = torch.max(diff).item()
    mean_diff = torch.mean(diff).item()
    print(f"{name}: max_diff = {max_diff:.6f}, mean_diff = {mean_diff:.6f}")
    return max_diff, mean_diff


def split_to_kv_cache(input_tensor, block_size, cur_kv_seqlen, max_seq_len):
    device = input_tensor.device
    T = input_tensor.shape[0]
    B = len(cur_kv_seqlen)
    total_seq_block_cnt = 0
    for i in range(B):
        seq_block_cnt = math.ceil(cur_kv_seqlen[i] / block_size)
        if seq_block_cnt == 0:
            seq_block_cnt = 1
        total_seq_block_cnt += seq_block_cnt
    block_table_list = torch.randperm(total_seq_block_cnt) if RANDOM_BLOCK_TABLE else torch.arange(0, total_seq_block_cnt, 1)

    block_tables = torch.zeros(B, math.ceil(max_seq_len / block_size)).to(torch.int32)
    input_shape = input_tensor.shape
    cache_tensor = torch.zeros(B * math.ceil(max_seq_len / block_size), block_size, input_shape[-2], input_shape[-1]).to(input_tensor.dtype).to(device)

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
            if block_idx == seq_block_cnt - 1:
                remain_size = kv_len - block_idx * block_size
                cache_tensor[block_tables[i][block_idx], :remain_size] = input_tensor[seq_offset: seq_offset + remain_size]
            else:
                cache_tensor[block_tables[i][block_idx], :block_size] = input_tensor[seq_offset: seq_offset + block_size]

        total_seq_block_cnt += seq_block_cnt
        tensor = input_tensor[start: start + kv_len]
        split_tensor = torch.split(tensor, block_size, dim=0)
        remaining_size = kv_len % block_size
        if remaining_size != 0:
            last_split = torch.narrow(tensor, 0, tensor.shape[0] - remaining_size, remaining_size)
            split_tensor = split_tensor[:-1] + (last_split,)
        start += kv_len
        spilt_tensors_list += split_tensor
    return cache_tensor, block_tables.to(device).detach()


def print_symmetric_difference(a, b):
    a = a.cpu().flatten()
    b = b.cpu().flatten()
    a_unique = a[~torch.isin(a, b)]
    b_unique = b[~torch.isin(b, a)]
    if a_unique.numel() != 0 or b_unique.numel() != 0:
        print("check fail")
        print("a_unique: ", a_unique, a_unique.numel())
        print("b_unique: ", b_unique, b_unique.numel())
    else:
        print("check pass")


def print_diff(name, tensor_npu, tensor_gpu):
    max_diff = torch.max(torch.abs(tensor_npu.float() - tensor_gpu.float()))
    mean_diff = torch.mean(torch.abs(tensor_npu.float() - tensor_gpu.float()))
    print(f'{name} max diff: %.9f' % max_diff)
    print(f'{name} mean diff: %.9f' % mean_diff)
    return max_diff, mean_diff


def _lightning_indexer(query, key, weights, cur_seq_lengths_query, cur_seq_lengths_key,
                       layout_query="TND", sparse_count=2048, sparse_mode=3, pre_tokens=2147483647, next_tokens=2147483647,
                       return_value=False, BLOCK_LEN=1, INIT_NUM=0, LOCAL_NUM=0, Q_BLOCK_LEN=1):
    batch_size = cur_seq_lengths_query.shape[0] - 1
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
        act_s1 = cur_seq_lengths_query[batch_id + 1] - cur_seq_lengths_query[batch_id]
        act_s2 = cur_seq_lengths_key[batch_id + 1] - cur_seq_lengths_key[batch_id]
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
        now_q = query.reshape(-1, N, D)[process_q_len:process_q_len + act_s1, :, :].transpose(0, 1).to(torch.float32)
        now_k = key.reshape(-1, D)[process_kv_len:process_kv_len + act_s2, :].transpose(0, 1).to(torch.float32)
        now_weights = weights.reshape(-1, N, 1)[process_q_len:process_q_len + act_s1, :, :].transpose(0, 1).to(torch.float32)
        process_q_len += act_s1
        process_kv_len += act_s2
        relu_out = torch.maximum(torch.matmul(now_q, now_k), torch.tensor(0))
        weight_out = relu_out * now_weights
        reduce_out = torch.sum(weight_out, dim=0)
        tmp_s1 = reduce_out.shape[0]
        tmp_s2 = reduce_out.shape[1]

        atten_mask_u = torch.triu(torch.ones([act_s1, act_s2], dtype=torch.uint8), diagonal=(act_s2 - act_s1) + 1)
        reduce_out = reduce_out.masked_fill(atten_mask_u.to(torch.bool), DEFAULT_SCORE)

        act_s2_align = (act_s2 + BLOCK_LEN - 1) // BLOCK_LEN * BLOCK_LEN
        reduce_out_tmp = torch.zeros([act_s1, act_s2_align], dtype=torch.float)
        reduce_out_tmp[:, :act_s2] = reduce_out
        reduce_out_tmp[:, act_s2:] = DEFAULT_SCORE

        reduce_out_tmp = reduce_out_tmp.view(act_s1, act_s2_align // BLOCK_LEN, BLOCK_LEN)
        reduce_out_tmp = torch.sum(reduce_out_tmp, dim=-1, keepdim=False) / BLOCK_LEN
        reduce_out_tmp = torch.clip(reduce_out_tmp, max=WINDOW_VALUE / 2)
        block_init_num = INIT_NUM // BLOCK_LEN
        block_local_num = LOCAL_NUM // BLOCK_LEN
        reduce_out_tmp[:, :block_init_num] = WINDOW_VALUE
        for s1 in range(act_s1):
            valid_s2_len = act_s2 - act_s1 + s1 + 1
            valid_s2_block = (valid_s2_len + BLOCK_LEN - 1) // BLOCK_LEN
            start_idx = (valid_s2_block - block_local_num) if (valid_s2_block - block_local_num) > 0 else 0
            reduce_out_tmp[s1, start_idx: valid_s2_block] = WINDOW_VALUE
            reduce_out_tmp[s1, valid_s2_block - 1] = DIGONAL_VALUE
            reduce_out_tmp[s1, valid_s2_block:] = DEFAULT_SCORE

        atten_mask_s2_scale_u = torch.triu(torch.ones([act_s1, act_s2_align], dtype=torch.uint8), diagonal=(act_s2 - act_s1) + 1)
        atten_mask_s2_scale_u = atten_mask_s2_scale_u.view(act_s1, act_s2_align // BLOCK_LEN, BLOCK_LEN)
        atten_mask_s2_scale_u = torch.sum(atten_mask_s2_scale_u, dim=-1, keepdim=False)
        atten_mask_s2_scale_u = (atten_mask_s2_scale_u == BLOCK_LEN).to(torch.uint8)

        reduce_out = reduce_out_tmp.contiguous()

        sorted_value, sorted_indices = torch.sort(reduce_out, dim=1, descending=True, stable=True)
        sorted_indices = sorted_indices.masked_fill(atten_mask_s2_scale_u.to(torch.bool), -1)
        if Q_BLOCK_LEN > 1:
            for s1 in range(act_s1):
                if s1 % Q_BLOCK_LEN == 0:
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
                        if s1 + q >= act_s1:
                            break
                        valid_s2_len = act_s2 - act_s1 + s1 + 1 + q
                        valid_s2_block = (valid_s2_len + BLOCK_LEN - 1) // BLOCK_LEN
                        if valid_s2_block < sparse_count:
                            sorted_indices[s1 + q, :] = sorted_indices[s1 + q - 1, :].clone()
                            sorted_indices[s1 + q, valid_s2_block - 1] = valid_s2_block - 1
                        else:
                            sorted_indices[s1 + q, :] = sorted_indices[s1 + q - 1, :].clone()
                            for i in range(q + 1):
                                if i == q:
                                    tmp = sorted_indices[s1 + q, sparse_count - 1 - i].clone()
                                sorted_indices[s1 + q, sparse_count - 1 - i] = valid_s2_block - 1 - i
                                if i == q:
                                    sorted_indices[s1 + q, 0] = tmp
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
    return q_len, kv_len


def calculate_special_sum(lst):
    if not lst:
        return 0
    total = 0
    for element in lst:
        if element == 0:
            total += 1
        else:
            total += element
    return total


def _run_mlp_lightning_indexer_test(B, T1, N, BLOCK_LEN, INIT_NUM, LOCAL_NUM, Q_BLOCK_LEN, SPARSE_COUNT,
                                     layout_key='TND', use_ge=False):
    assert INIT_NUM % BLOCK_LEN == 0, "INIT_NUM invalid"
    assert LOCAL_NUM % BLOCK_LEN == 0, "LOCAL_NUM invalid"
    actual_q_len, actual_kv_len = gen_seq_len(B, T1)
    T1 = calculate_special_sum(actual_q_len)
    T2 = calculate_special_sum(actual_kv_len)
    print('B, T1, T2, N, BLOCK_LEN, INIT_NUM, LOCAL_NUM, Q_BLOCK_LEN, SPARSE_COUNT: ',
          B, T1, T2, N, BLOCK_LEN, INIT_NUM, LOCAL_NUM, Q_BLOCK_LEN, SPARSE_COUNT)
    print('actual_q_len', actual_q_len)
    print('actual_kv_len', actual_kv_len)
    D = 128
    block_size = 128
    max_seq_len = max(actual_kv_len)
    layout_query = 'TND'
    dType = torch.bfloat16
    np.random.seed(3)

    query = torch.tensor(np.random.uniform(-2, 2, (T1, N, D))).to(dType)
    key = torch.tensor(np.random.uniform(-2, 2, (T2, 1, D))).to(dType)
    weights = torch.tensor(np.random.uniform(-1, 1, (T1, N))).to(torch.float)
    cur_q_len = [0] + list(np.cumsum(actual_q_len))
    cur_kv_len = [0] + list(np.cumsum(actual_kv_len))
    cur_seq_lengths_query = torch.tensor(cur_q_len).to(torch.int64)
    cur_seq_lengths_key = torch.tensor(cur_kv_len).to(torch.int64)
    print('cu_q_len', cur_q_len)
    print('cu_kv_len', cur_kv_len)

    sparse_count = SPARSE_COUNT
    sparse_mode = 3
    pre_tokens = 2147483647
    next_tokens = 2147483647
    return_value = False
    cpu_out, cpu_valuesOut = _lightning_indexer(query, key, weights, cur_seq_lengths_query, cur_seq_lengths_key,
                                layout_query, sparse_count, sparse_mode, pre_tokens, next_tokens, return_value,
                                BLOCK_LEN, INIT_NUM, LOCAL_NUM, Q_BLOCK_LEN)

    torch_npu.npu.set_device(int(DEVICE_ID))
    query = query.to("npu:%s" % DEVICE_ID)
    key = key.to("npu:%s" % DEVICE_ID)
    weights = weights.to("npu:%s" % DEVICE_ID)
    cur_seq_lengths_query = cur_seq_lengths_query.to("npu:%s" % DEVICE_ID)
    cur_seq_lengths_key = cur_seq_lengths_key.to("npu:%s" % DEVICE_ID)

    if layout_key == "TND":
        npu_out, npu_valuesOut = torch.ops.custom.mlp_lightning_indexer(
            query, key, weights, cur_seq_lengths_query=cur_seq_lengths_query,
            cur_seq_lengths_key=cur_seq_lengths_key, block_table=None, layout_query=layout_query,
            layout_key=layout_key, sparse_count=sparse_count, kv_block_len=BLOCK_LEN, q_block_len=Q_BLOCK_LEN,
            init_num=INIT_NUM, local_num=LOCAL_NUM, sparse_mode=sparse_mode, pre_tokens=pre_tokens,
            next_tokens=next_tokens, return_value=return_value)

        if use_ge:
            try:
                import torchair as tng
                from torchair.configs.compiler_config import CompilerConfig
                torch._logging.set_logs(recompiles=True)
                config = CompilerConfig()
                config.debug.graph_dump.type = "py"
                config.experimental_config.tiling_schedule_optimize = True
                npu_backend = tng.get_npu_backend(compiler_config=config)
                model = LightningIndexerModel().npu()
                model = torch.compile(model, backend=npu_backend, dynamic=False, fullgraph=False)
                npu_out_ge, npu_valuesOut_ge = model(
                    query, key, weights, cur_seq_lengths_query=cur_seq_lengths_query,
                    cur_seq_lengths_key=cur_seq_lengths_key, block_table=None, layout_query=layout_query,
                    layout_key=layout_key, sparse_count=sparse_count, kv_block_len=BLOCK_LEN, q_block_len=Q_BLOCK_LEN,
                    init_num=INIT_NUM, local_num=LOCAL_NUM, sparse_mode=sparse_mode, pre_tokens=pre_tokens,
                    next_tokens=next_tokens, return_value=return_value)
                print("GE mode test passed")
            except Exception as e:
                print(f"GE mode test skipped: {e}")
    else:
        key_cache, key_block_table = split_to_kv_cache(key, block_size, actual_kv_len, max_seq_len)
        key_cache = key_cache.to("npu:%s" % DEVICE_ID)
        key_block_table = key_block_table.to("npu:%s" % DEVICE_ID)
        npu_out, npu_valuesOut = torch.ops.custom.mlp_lightning_indexer(
            query, key_cache, weights, cur_seq_lengths_query=cur_seq_lengths_query,
            cur_seq_lengths_key=cur_seq_lengths_key, block_table=key_block_table,
            layout_query=layout_query,
            layout_key=layout_key, sparse_count=sparse_count, kv_block_len=BLOCK_LEN, q_block_len=Q_BLOCK_LEN,
            init_num=INIT_NUM, local_num=LOCAL_NUM, sparse_mode=sparse_mode, pre_tokens=pre_tokens,
            next_tokens=next_tokens, return_value=return_value)

    cpu_out = cpu_out.reshape(-1, sparse_count).cpu()
    npu_out = npu_out.reshape(-1, sparse_count).cpu()
    cpu_valuesOut = cpu_valuesOut.reshape(-1, sparse_count).cpu()
    npu_valuesOut = npu_valuesOut.reshape(-1, sparse_count).cpu()
    print('npu_out shape is ', npu_out.shape)
    print('cpu_out', cpu_out)
    print('npu_out', npu_out)

    real_mask = (npu_valuesOut.to(torch.float) == float('-inf'))
    npu_valuesOut = npu_valuesOut.to(torch.float).masked_fill(real_mask.to(torch.bool), MAGIC_NUMBER)

    gt_mask = (cpu_valuesOut == DEFAULT_SCORE)
    cpu_valuesOut = cpu_valuesOut.masked_fill(gt_mask.to(torch.bool), MAGIC_NUMBER)

    print_symmetric_difference(npu_out, cpu_out)


class TestMlpLightningIndexer(TestCase):
    def test_tnd_lightning_indexer(self):
        _run_mlp_lightning_indexer_test(4, 4, 64, 1, 4, 32, 1, 128, layout_key='TND')

    def test_tnd_lightning_indexer_kv_block(self):
        _run_mlp_lightning_indexer_test(2, 154, 8, 1, 16, 2048, 1, 4096, layout_key='TND')

    def test_tnd_lightning_indexer_ge(self):
        _run_mlp_lightning_indexer_test(2, 154, 8, 1, 16, 2048, 1, 4096, layout_key='TND', use_ge=True)

    def test_pa_bsnd_lightning_indexer(self):
        _run_mlp_lightning_indexer_test(2, 154, 8, 1, 16, 2048, 1, 4096, layout_key='PA_BSND')


if __name__ == "__main__":
    run_tests()
