#!/usr/bin/python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import sys
import numpy as np

def adapter_capacity(sorted_row_idx, sorted_expert_idx, capacity):
    count = 0
    last = sorted_expert_idx[0]
    for i, val in enumerate(sorted_expert_idx):
        if last != val:
            count = 1
            last = val
        else:
            count += 1
            if count > capacity:
                sorted_expert_idx[i] = -1
                sorted_row_idx[i] = -1

def gen_golden(x, expert_idx, scale, offset, active_num, expert_capacity,
                expert_num, drop_pad_mode, expert_tokens_num_type, expert_tokens_num_flag,
                quant_mode, active_expert_range, row_idx_type):
    if drop_pad_mode == 1:
        if expert_num <= 0:
            print("expert num can not be 0")
            return
    expert_start = active_expert_range[0] if drop_pad_mode == 0 else 0
    expert_end = active_expert_range[1] if drop_pad_mode == 0 else expert_num
    num_rows = x.shape[0]
    h = x.shape[1]
    k = expert_idx.shape[-1]
    expert_idx_in = expert_idx.copy().reshape(-1)
    actual_expert_total_num = np.sum((expert_idx_in >= expert_start) & (expert_idx_in < expert_end))

    expert_idx_in[(expert_idx_in < expert_start)] = np.int32(np.iinfo(np.int32).max)
    sorted_expert_indices = np.argsort(expert_idx_in, axis=-1, kind="stable")
    sorted_expert_idx = expert_idx_in[sorted_expert_indices]
    if row_idx_type == 1:
        expanded_row_idx = sorted_expert_indices[:actual_expert_total_num]
    else:
        expanded_row_idx = np.ones(num_rows * k).astype(np.int32) * -1
        tmp_indices = np.arange(actual_expert_total_num)
        expanded_row_idx[sorted_expert_indices[:actual_expert_total_num]] = tmp_indices

    if not expert_tokens_num_flag:
        expert_tokens_count = None
    else:
        if drop_pad_mode == 0:
            if expert_tokens_num_type == 1:
                expert_tokens_count = np.bincount(
                    sorted_expert_idx[:actual_expert_total_num] - expert_start)
                expert_tokens_count = np.concatenate(
                    [expert_tokens_count, np.zeros((expert_end - expert_start) - len(expert_tokens_count)).astype(np.int64)])
            elif expert_tokens_num_type == 0:
                expert_tokens_count = np.bincount(
                    sorted_expert_idx[:actual_expert_total_num] - expert_start)
                expert_tokens_count = np.concatenate(
                    [expert_tokens_count, np.zeros((expert_end - expert_start) - len(expert_tokens_count)).astype(np.int64)])
                expert_tokens_count = np.cumsum(expert_tokens_count)
            elif expert_tokens_num_type == 2:
                expert_id, counts = np.unique(sorted_expert_idx[:actual_expert_total_num], return_counts=True)
                expert_tokens_count = np.column_stack((expert_id, counts))
                if expert_tokens_count.shape[0] < expert_num:
                    expert_tokens_count = np.concatenate((expert_tokens_count, [[0,0],]), axis=0)
        else:
            expert_tokens_count = np.bincount(
                    sorted_expert_idx[:actual_expert_total_num] - expert_start)
            expert_tokens_count = np.concatenate(
                [expert_tokens_count, np.zeros((expert_end - expert_start) - len(expert_tokens_count)).astype(np.int64)])
        expert_tokens_count = expert_tokens_count.astype(np.int64)

    if drop_pad_mode == 0:
        if active_num == 0:
            active_num = actual_expert_total_num
        else:
            active_num = min(active_num, actual_expert_total_num)
        expanded_scale = None
        expanded_x = x[sorted_expert_indices[:active_num] // k, :]
        if scale is not None and quant_mode == -1:
            expanded_scale = scale[sorted_expert_indices[:active_num] // k]
    else:
        adapter_capacity(sorted_expert_indices, sorted_expert_idx, expert_capacity)

        sort_row_tmp = np.full((expert_num * expert_capacity), -1, dtype=int)
        offset_tmp = 0
        lastExpertId = 0
        for i, val in enumerate(sorted_expert_indices):
            if val != -1:
                if lastExpertId != sorted_expert_idx[i]:
                    offset_tmp = 0
                    lastExpertId = sorted_expert_idx[i]
                sort_row_tmp[sorted_expert_idx[i] * expert_capacity + offset_tmp] = sorted_expert_indices[i]
                offset_tmp = offset_tmp + 1

        expanded_row_idx = np.full(sorted_expert_indices.shape, -1)
        for i, val in enumerate(sort_row_tmp):
            if val != -1:
                expanded_row_idx[val] = i

        expanded_x_mask = np.full((expert_num * expert_capacity, h), 1, dtype=int)
        expanded_x = np.full((expert_num * expert_capacity, h), 0, dtype=x.dtype)
        for i, val in enumerate(sort_row_tmp):
            if val != -1:
                expanded_x[i] = x[val // k]
                expanded_x_mask[i] = np.full((h,), 0, dtype=int)

    if quant_mode == -1:
        expanded_x = expanded_x
        expanded_row_idx = expanded_row_idx
        if scale is not None and drop_pad_mode == 1:
            expanded_scale = np.full((expert_num * expert_capacity,), 0, dtype=scale.dtype)
            for i, val in enumerate(sort_row_tmp):
                if val != -1:
                    expanded_scale[i] = scale[val // k]
        if scale is None:
            expanded_scale = None

    if quant_mode == 0:
        expanded_scale = None
        expanded_x_fp16 = expanded_x.astype(np.float16)
        scale_val = scale.astype(np.float16)
        offset_val = offset.astype(np.float16)
        scale_rst = expanded_x_fp16 * scale_val[0]
        add_offset = scale_rst + offset_val[0]
        round_data = np.rint(add_offset)
        round_data = np.clip(round_data, -128, 127)
        expanded_x = round_data.astype(np.int8)

    if quant_mode == 1:
        x_final = expanded_x.astype(np.float32)
        if scale is None:
            x_abs = np.abs(x_final)
            x_max = np.max(x_abs, axis=-1, keepdims=True)
            expanded_scale = x_max / 127
            expanded_x = x_final / expanded_scale
            expanded_x = np.round(expanded_x).astype(np.int8)
        else:
            if scale.shape[0] == 1:
                x_final = x_final * scale
            else:
                if drop_pad_mode == 0:
                    x_final = x_final * scale[sorted_expert_idx[:active_num] - expert_start]

                else:
                    for i, val in enumerate(sort_row_tmp):
                        if val != -1:
                            x_final[i] = x_final[i] * scale[i // expert_capacity]
            x_abs = np.abs(x_final)
            x_max = np.max(x_abs, axis=-1, keepdims=True)
            expanded_scale = x_max / 127
            expanded_x = x_final / expanded_scale
            expanded_x = np.round(expanded_x).astype(np.int8)
        if x.dtype == np.int8:
            expanded_scale == None
    if drop_pad_mode == 1:
        expanded_x = np.ma.array(expanded_x, mask=expanded_x_mask).filled(0)
        expanded_x = expanded_x.reshape(expert_num, expert_capacity, h)

    return expanded_x, expanded_row_idx, expert_tokens_count, expanded_scale



def gen_golden_data_simple(N, H, k, E, dtype, quant_mode, active_num, expert_capacity,
                            expert_num, drop_pad, expert_token_num_type,
                            expert_tokens_num_flag, active_range, row_idx_type):
    N = int(N)
    H = int(H)
    k = int(k)
    E = int(E)
    expert_num = int(expert_num)
    quant_mode = int(quant_mode)
    active_num  = int(active_num)
    expert_capacity = int(expert_capacity)
    drop_pad = int(drop_pad)
    expert_token_num_type = int(expert_token_num_type)
    expert_tokens_num_flag = int(expert_tokens_num_flag)
    active_range = eval(active_range)
    row_idx_type = int(row_idx_type)

    input_x = np.random.uniform(-2, 2, [N, H]).astype(dtype)
    input_expertIdx = np.random.randint(0, E, size=(N, k)).astype("int32")
    scale = np.random.uniform(-2, 2, [N]).astype(np.float32)
    offset = None
    if quant_mode == 0:
        offset = np.random.uniform(-2, 2, [1]).astype(np.float32)
        scale = np.random.uniform(-2, 2, [1]).astype(np.float32)


    input_x.tofile("./input_x.bin")
    input_expertIdx.tofile("./input_expertIdx.bin")
    scale.tofile("./scale.bin")
    if quant_mode == 0:
        offset.tofile("./offset.bin")
    expanded_x, expanded_row_idx, expert_tokens_count, expanded_scale = gen_golden(input_x, input_expertIdx, scale,
                                                                                    offset, active_num, expert_capacity,
                                                                                    expert_num, drop_pad, expert_token_num_type, expert_tokens_num_flag,
                                                                                    quant_mode, active_range, row_idx_type)
    if quant_mode == -1:
        expanded_x.astype(dtype).tofile("./expanded_x.bin")
    else:
        expanded_x.astype(np.int8).tofile("./expanded_x.bin")
    expanded_row_idx.astype(np.int32).tofile("./expanded_row_idx.bin")
    if expert_tokens_num_flag:
        expert_tokens_count.tofile("./expert_tokens_count.bin")
    if expanded_scale is not None:
        expanded_scale.tofile("./expanded_scale.bin")

if __name__ == "__main__":
    gen_golden_data_simple(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5],
                            sys.argv[6], sys.argv[7], sys.argv[8], sys.argv[9], sys.argv[10], sys.argv[11],
                            sys.argv[12], sys.argv[13], sys.argv[14])