#!/usr/bin/python3
# -*- coding:utf-8 -*-
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
import glob
import os
import tensorflow as tf

bf16 = tf.bfloat16.as_numpy_dtype

curr_dir = os.path.dirname(os.path.realpath(__file__))


def compare_data(expanded_x_golden_file, expanded_row_idx_golden_file, expert_tokens_count_golden_file, expanded_scale_golden_file,
                expanded_x_file, expanded_row_idx_file, expert_tokens_count_file, expanded_scale_file, d_type):
    if d_type == "float16":
        precision = 1/1000
    else:
        precision = 1/10000
    expanded_x_type_dict = {
        "int8": np.int8,
        "float16": np.float16,
        "bfloat16": bf16,
        "float32": np.float32
    }
    expanded_x_type = expanded_x_type_dict.get(d_type)
    expanded_row_idx_type = np.int32
    expert_tokens_count_type = np.int64
    expanded_scale_type = np.float32


    expanded_x_golden = np.fromfile(expanded_x_golden_file, expanded_x_type)
    expanded_row_idx_golden = np.fromfile(expanded_row_idx_golden_file, expanded_row_idx_type)
    expert_tokens_count_golden = np.fromfile(expert_tokens_count_golden_file, expert_tokens_count_type) if expert_tokens_count_golden_file else None
    expanded_scale_golden = np.fromfile(expanded_scale_golden_file, expanded_scale_type) if expanded_scale_golden_file else None

    expanded_x = np.fromfile(expanded_x_file, expanded_x_type)[:len(expanded_x_golden)]
    expanded_row_idx = np.fromfile(expanded_row_idx_file, expanded_row_idx_type)[:len(expanded_row_idx_golden)]
    expert_tokens_count = np.fromfile(expert_tokens_count_file, expert_tokens_count_type)[:len(expert_tokens_count_golden)] if expert_tokens_count_file else None
    expanded_scale = np.fromfile(expanded_scale_file, expanded_scale_type)[:len(expanded_scale_golden)] if expanded_scale_file else None


    diff_res_expanded_x = np.isclose(expanded_x, expanded_x_golden, precision, 0, True)
    diff_idx_expanded_x = np.where(diff_res_expanded_x != True)[0]
    diff_ratio_expanded_x = len(diff_idx_expanded_x) / expanded_x_golden.size

    diff_res_expanded_row_idx = np.isclose(expanded_row_idx, expanded_row_idx_golden, precision, 0, True)
    diff_idx_expanded_row_idx = np.where(diff_res_expanded_row_idx != True)[0]
    diff_ratio_expanded_row_idx = len(diff_idx_expanded_row_idx) / expanded_row_idx_golden.size

    if expert_tokens_count is not None:
        diff_res_expert_tokens_count = np.isclose(expert_tokens_count, expert_tokens_count_golden, precision, 0, True)
        diff_idx_expert_tokens_count = np.where(diff_res_expert_tokens_count != True)[0]
        diff_ratio_expert_tokens_count = len(diff_idx_expert_tokens_count) / expert_tokens_count.size
    else:
        diff_idx_expert_tokens_count = np.array([])
        diff_ratio_expert_tokens_count = 0

    if expanded_scale is not None:
        diff_res_expanded_scale = np.isclose(expanded_scale, expanded_scale_golden, precision, 0, True)
        diff_idx_expanded_scale = np.where(diff_res_expanded_scale != True)[0]
        diff_ratio_expanded_scale = len(diff_idx_expanded_scale) / expanded_scale_golden.size
    else:
        diff_idx_expanded_scale = np.array([])
        diff_ratio_expanded_scale = 0

    if len(diff_idx_expanded_x) + len(diff_idx_expanded_row_idx) + len(diff_idx_expert_tokens_count) + len(diff_idx_expanded_scale) == 0 or \
         (diff_ratio_expanded_x < precision and diff_ratio_expanded_row_idx < precision and diff_ratio_expert_tokens_count < precision and diff_ratio_expanded_scale < precision):
        print("PASSED!")

    else:
        print("FAILED!")
        print(f"expanded_x Diff ratio: {diff_ratio_expanded_x}")
        for idx in diff_idx_expanded_x[:5]:
            print(f"index: {idx}, expanded_x output: {expanded_x[idx]}, expanded_x golden: {expanded_x_golden[idx]}")
        print(f"expanded_row_idx Diff ratio: {diff_ratio_expanded_row_idx}")
        for idx in diff_idx_expanded_row_idx[:5]:
            print(f"index: {idx}, expanded_row_idx output: {expanded_row_idx[idx]}, expanded_row_idx golden: {expanded_row_idx_golden[idx]}")
        print(f"expert_tokens_count Diff ratio: {diff_ratio_expert_tokens_count}")
        for idx in diff_idx_expert_tokens_count[:5]:
            print(f"index: {idx}, expert_tokens_count output: {expert_tokens_count[idx]}, expert_tokens_count golden: {expert_tokens_count_golden[idx]}")
        print(f"expanded_scale Diff ratio: {diff_ratio_expanded_scale}")
        for idx in diff_idx_expanded_scale[:5]:
            print(f"index: {idx}, expanded_scale output: {expanded_scale[idx]}, expanded_scale golden: {expanded_scale_golden[idx]}")
    return


def process(d_type, is_output_expert_tokens_count = 1, is_output_expanded_scale = 1):
    expanded_x_golden_file = os.path.join(curr_dir, "expanded_x.bin") # to do
    expanded_row_idx_golden_file = os.path.join(curr_dir, "expanded_row_idx.bin")
    expert_tokens_count_golden_file = os.path.join(curr_dir, "expert_tokens_count.bin") if int(is_output_expert_tokens_count) else None
    expanded_scale_golden_file = os.path.join(curr_dir, "expanded_scale.bin") if int(is_output_expanded_scale) else None
    expanded_x_file = os.path.join(curr_dir, "output_expanded_x.bin")
    expanded_row_idx_file = os.path.join(curr_dir, "output_expanded_row_idx.bin")
    expert_tokens_count_file = os.path.join(curr_dir, "output_expert_tokens_count.bin") if int(is_output_expert_tokens_count) else None
    expanded_scale_file = os.path.join(curr_dir, "output_expanded_scale.bin") if int(is_output_expanded_scale) else None
    # result = compare_data(golden_file, output_file, d_type)
    result = compare_data(expanded_x_golden_file, expanded_row_idx_golden_file, expert_tokens_count_golden_file, expanded_scale_golden_file,
                expanded_x_file, expanded_row_idx_file, expert_tokens_count_file, expanded_scale_file, d_type)

if __name__ == '__main__':
    process(sys.argv[1], sys.argv[2], sys.argv[3])
