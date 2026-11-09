# This program is free software, you can redistribute it and/or modify it.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import os
import shutil
import numpy as np
import torch
import torch_npu
import torchair._contrib.custom_torch_ops
import torchair as tng
# import custom_ops
import flash_ops
from torch_npu.testing.testcase import TestCase, run_tests
from torchair.configs.compiler_config import CompilerConfig


config = CompilerConfig()
config.experimental_config.keep_inference_input_mutations = True
npu_backend = tng.get_npu_backend(compiler_config=config)

DEVICE_ID = 0
torch_npu.npu.set_device(int(DEVICE_ID))

# hf32的设置
option = dict()
option["ACL_PRECISION_MODE"] = "must_keep_origin_dtype"
torch.npu.set_option(option)
np.set_printoptions(threshold=np.inf)


def remove(path):
    if os.path.exists(path):
        shutil.rmtree(path)


def sort_with_negative_ones_last(arr):
    """
    对数组的最后一维进行排序，-1排在最后, 支持多维
    """
    # 创建掩码数组，-1的位置为True
    negative_mask = (arr == -1)
    # 将-1替换为无穷大，这样排序时会排在最后
    sort_arr = np.where(negative_mask, np.inf, arr)
    # 获取排序索引（沿最后一个轴）
    sorted_indices = np.argsort(sort_arr, axis=-1)
    # 使用高级索引重新排列数组
    # 创建一个索引数组来匹配3维结构
    indices = np.indices(arr.shape)
    indices[-1] = sorted_indices
    return arr[tuple(indices)]


def do_golden_all_host(batch_size, seq_length, head_num, selection_top_k, select_k_rope, select_kv_cache,
                       select_kv_block_table, select_kv_block_status, select_topk_indices,
                       full_k_ropes, full_kv_caches, full_kv_block_tables, full_kv_actual_seqs, full_q_actual_seqs,
                       selection_topk_block_sizes, lay_out):
    s_block_num, s_block_sizes, _ = select_k_rope.shape
    s_block_tokens = s_block_sizes
    f_block_num, f_block_sizes, _ = full_k_ropes.shape
    f_block_tokens = f_block_sizes

    if lay_out == 'TND':
        select_topk_indices = select_topk_indices.reshape(
            batch_size, seq_length, head_num, selection_top_k)
        select_kv_block_status = select_kv_block_status.reshape(
            batch_size, seq_length, head_num, selection_top_k + 1)
    batch_size, seq_num, head_dim, topk_num = select_topk_indices.shape
    s_maxblocknum = select_kv_block_table.shape[-1]

    select_kv_block_table = select_kv_block_table.reshape(
        batch_size, seq_num, head_dim, -1)
    selection_kv_actual_seq_out = np.zeros(
        [batch_size, seq_num, head_dim], dtype=np.int32)

    for _bs in range(batch_size):
        f_q_actual_sq = full_q_actual_seqs[_bs]
        if f_q_actual_sq > seq_num:
            assert(False)
        for _s in range(seq_num):
            f_kv_actual_sq = full_kv_actual_seqs[_bs]
            if _s < f_q_actual_sq:
                mtp_gep = f_q_actual_sq - 1 - _s
                f_kv_actual_sq = f_kv_actual_sq - mtp_gep
            if f_kv_actual_sq <= 0:
                continue
            max_selection_num = (
                f_kv_actual_sq + selection_topk_block_sizes - 1) // selection_topk_block_sizes
            last_block_size = f_kv_actual_sq - \
                (max_selection_num - 1) * selection_topk_block_sizes
            for _hd in range(head_dim):
                actual_seq = 0
                valid_topk_num = 0
                for topk_block_idx in range(topk_num):
                    topk_block_segment_id = select_topk_indices[_bs][_s][_hd][topk_block_idx]
                    if topk_block_segment_id < 0:
                        break
                    if topk_block_segment_id > max_selection_num - 1:
                        continue
                    select_kv_block_status[_bs][_s][_hd][valid_topk_num] = topk_block_segment_id

                    s_tokens_offset = valid_topk_num * selection_topk_block_sizes
                    s_block_offset = s_tokens_offset // s_block_tokens
                    s_block_offset_batch = select_kv_block_table[_bs][_s][_hd][s_block_offset]
                    s_block_tokens_offset = s_tokens_offset - \
                        (s_block_offset * s_block_tokens)

                    f_tokens_offset = topk_block_segment_id * selection_topk_block_sizes
                    block_table_id = f_tokens_offset // f_block_tokens
                    f_block_offset = full_kv_block_tables[_bs][block_table_id]
                    f_block_tokens_offset = f_tokens_offset - \
                        (block_table_id * f_block_tokens)

                    # 计算时间tokens数
                    cu_block_size = selection_topk_block_sizes if topk_block_segment_id != max_selection_num - \
                        1 else last_block_size
                    end_index = s_block_tokens_offset + cu_block_size
                    select_k_rope[s_block_offset_batch][s_block_tokens_offset: end_index] = \
                        full_k_ropes[f_block_offset][f_block_tokens_offset: f_block_tokens_offset + cu_block_size]
                    select_kv_cache[s_block_offset_batch][s_block_tokens_offset:end_index] = \
                        full_kv_caches[f_block_offset][f_block_tokens_offset: f_block_tokens_offset + cu_block_size]

                    actual_seq += cu_block_size
                    valid_topk_num = valid_topk_num + 1

                selection_kv_actual_seq_out[_bs][_s][_hd] = actual_seq

                select_kv_block_status[_bs][_s][_hd][valid_topk_num:topk_num] = -1
                select_kv_block_status[_bs][_s][_hd][topk_num] = actual_seq

    return select_k_rope, select_kv_cache, select_kv_block_table.reshape(batch_size * seq_num * head_dim, -1), \
           select_kv_block_status, selection_kv_actual_seq_out.reshape(batch_size * seq_num * head_dim)


def gather_info_gen(_bs, _s, _hd, max_selection_num, select_kv_block_status,
                    select_topk_indices, select_topk_block_size):
    batch_size, seq_num, head_dim, topk_num = select_topk_indices.shape

    valid_topk_num = 0
    max_topk_id = -1
    max_topk_writed_idx = -1
    max_hit_same_seq_idx = -1
    insert_status_same_seq = np.ones(topk_num, dtype=np.int32) * -1
    hit_from_src_seq = np.ones(topk_num, dtype=np.int32) * -1

    for topk_block_idx in range(topk_num):
        topk_block_segment_id = select_topk_indices[_bs][_s][_hd][topk_block_idx]
        if topk_block_segment_id < 0:
            break
        if topk_block_segment_id > max_selection_num - 1:
            continue
        if topk_block_segment_id >= max_topk_id:  # 表明上一个max_topk_id是假的max, max_topk_writed_idx赋值为-1
            max_topk_writed_idx = -1

        valid_topk_num = valid_topk_num + 1
        # topk indices是无序的
        maybe_max_topk_id = False if max_topk_id > topk_block_segment_id else True
        max_topk_id = max_topk_id if max_topk_id > topk_block_segment_id else topk_block_segment_id
        for s_idx in range(0, seq_num):
            topk_actual_seqlen_old = select_kv_block_status[_bs][s_idx][_hd][topk_num]
            if topk_actual_seqlen_old <= 0:
                continue
            topk_status_old = select_kv_block_status[_bs][s_idx][_hd][0:topk_num]
            hit_indices_idx = np.where(
                topk_status_old == topk_block_segment_id)[0]
            if hit_indices_idx.size > 0:  # 命中
                is_invalid_block = False
                if topk_actual_seqlen_old % select_topk_block_size != 0:
                    tail_block_idx = topk_actual_seqlen_old // select_topk_block_size  # + 1
                    # 不完整的尾块
                    is_invalid_block = hit_indices_idx[0] == tail_block_idx

                hit_value = (topk_actual_seqlen_old + select_topk_block_size - 1) // select_topk_block_size - 1
                if hit_indices_idx[0] > hit_value:
                    is_invalid_block = True
                if not is_invalid_block:
                    if s_idx == _s:  # 同seq命中必更新
                        insert_status_same_seq[hit_indices_idx[0]
                                               ] = topk_block_idx
                        hit_from_src_seq[topk_block_idx] = -10
                        if hit_indices_idx[0] > max_hit_same_seq_idx:
                            max_hit_same_seq_idx = hit_indices_idx[0]
                        if maybe_max_topk_id:
                            max_topk_writed_idx = hit_indices_idx[0]
                    else:
                        # 非同seq命中,如果已经命中了其他的,就不更新,防止把同seq命中的给刷成非同seq命中
                        if hit_from_src_seq[topk_block_idx] == -1:
                            hit_from_src_seq[topk_block_idx] = s_idx * \
                                topk_num + hit_indices_idx[0]
                    if s_idx >= _s:
                        break

    empty_pos_cnt = max_hit_same_seq_idx + 1 - valid_topk_num
    if empty_pos_cnt > 0:  # valid_topk_num为0时,max_hit_same_seq_idx=-1,不满足该条件,也不会认为有空位
        tmp_cnt = 0
        for ei in range(max_hit_same_seq_idx, -1, -1):
            if insert_status_same_seq[ei] >= 0:
                hit_from_src_seq[insert_status_same_seq[ei]
                                 ] = _s * topk_num + ei
                insert_status_same_seq[ei] = -1
            tmp_cnt = tmp_cnt + 1
            if tmp_cnt >= empty_pos_cnt:
                break

    return valid_topk_num, max_topk_id, max_topk_writed_idx, insert_status_same_seq, hit_from_src_seq


def do_golden_gen(batch_size, seq_length, head_num, select_topk, select_k_rope, select_kv_cache,
                  select_kv_block_table, select_kv_block_status, select_topk_indices,
                  full_k_ropes, full_kv_caches, full_kv_block_tables, full_kv_actual_seqs, full_q_actual_seqs,
                  selection_topk_block_sizes, lay_out):
    s_block_num, s_block_sizes, _ = select_k_rope.shape
    s_block_tokens = s_block_sizes
    f_block_num, f_block_sizes, _ = full_k_ropes.shape
    f_block_tokens = f_block_sizes

    if lay_out == 'TND':
        select_topk_indices = select_topk_indices.reshape(
            batch_size, seq_length, head_num, select_topk)
        select_kv_block_status = select_kv_block_status.reshape(
            batch_size, seq_length, head_num, select_topk + 1)

    batch_size, seq_num, head_dim, topk_num = select_topk_indices.shape
    s_maxblocknum = select_kv_block_table.shape[-1]

    select_kv_block_table = select_kv_block_table.reshape(
        batch_size, seq_num, head_dim, -1)
    selection_kv_actual_seq_out = np.zeros(
        [batch_size, seq_num, head_dim], dtype=np.int32)

    gather_total_num = 0
    gather_from_host_num = 0
    gather_from_gm_num = 0
    gather_same_seq_hit_num = 0
    gather_swap_num = 0

    for _bs in range(batch_size):
        f_q_actual_sq = full_q_actual_seqs[_bs]
        if f_q_actual_sq > seq_num:
            assert(False)
        hit_in_old_same_seq_idx = np.ones(
            (seq_num, head_dim, topk_num), dtype=np.int32) * -1
        select_cache_origin = np.ones(
            (seq_num, head_dim, topk_num), dtype=np.int32) * -1
        for _s in range(seq_num):
            f_kv_actual_sq = full_kv_actual_seqs[_bs]
            if _s < f_q_actual_sq:
                mtp_gep = f_q_actual_sq - 1 - _s
                f_kv_actual_sq = f_kv_actual_sq - mtp_gep
            if f_kv_actual_sq <= 0:
                continue
            max_selection_num = (
                f_kv_actual_sq + selection_topk_block_sizes - 1) // selection_topk_block_sizes
            last_block_size = f_kv_actual_sq - \
                (max_selection_num - 1) * selection_topk_block_sizes
            for _hd in range(head_dim):

                valid_topk_num, max_topk_id, max_topk_writed_idx, insert_status_same_seq, hit_from_src_seq = \
                    gather_info_gen(_bs, _s, _hd, max_selection_num, select_kv_block_status,
                                    select_topk_indices, selection_topk_block_sizes)

                total_insert_idx = 0
                actual_seq = 0
                valid_topk_num = 0
                for topk_block_idx in range(topk_num):
                    topk_block_segment_id = select_topk_indices[_bs][_s][_hd][topk_block_idx]
                    if topk_block_segment_id < 0:
                        break
                    if topk_block_segment_id > max_selection_num - 1:
                        continue

                    gather_total_num = gather_total_num + 1

                    # 计算时间tokens数
                    cu_block_size = selection_topk_block_sizes if topk_block_segment_id != max_selection_num - \
                        1 else last_block_size
                    actual_seq += cu_block_size
                    valid_topk_num += 1

                    if hit_from_src_seq[topk_block_idx] == -10:
                        gather_same_seq_hit_num = gather_same_seq_hit_num + 1
                        continue
                    # 本seq未命中,都需要进行拷贝,先找出可以插入的空位
                    insert_idx = 0

                    mask = insert_status_same_seq[total_insert_idx:topk_num] < 0
                    valid_indices = np.where(mask)[0]
                    if len(valid_indices) == 0:
                        assert(False)
                    insert_idx = total_insert_idx + valid_indices[0]

                    total_insert_idx = insert_idx + 1
                    s_tokens_offset = insert_idx * selection_topk_block_sizes
                    s_block_offset = s_tokens_offset // s_block_tokens
                    s_block_offset_batch = select_kv_block_table[_bs][_s][_hd][s_block_offset]
                    s_block_tokens_offset = s_tokens_offset - \
                        (s_block_offset * s_block_tokens)

                    if hit_from_src_seq[topk_block_idx] == -1:
                        gather_from_host_num = gather_from_host_num + 1
                        f_tokens_offset = topk_block_segment_id * selection_topk_block_sizes
                        block_table_id = f_tokens_offset // f_block_tokens
                        f_block_offset = full_kv_block_tables[_bs][block_table_id]
                        f_block_tokens_offset = f_tokens_offset - \
                            (block_table_id * f_block_tokens)
                        
                        end_index = s_block_tokens_offset + cu_block_size
                        select_k_rope[s_block_offset_batch][s_block_tokens_offset: end_index] = \
                            full_k_ropes[f_block_offset][f_block_tokens_offset:
                                                        f_block_tokens_offset + cu_block_size]
                        select_kv_cache[s_block_offset_batch][s_block_tokens_offset: end_index] = \
                            full_kv_caches[f_block_offset][f_block_tokens_offset:
                                                          f_block_tokens_offset + cu_block_size]
                    else:
                        gather_from_gm_num = gather_from_gm_num + 1
                        gm_src_seq = hit_from_src_seq[topk_block_idx] // topk_num
                        gm_src_idx = hit_from_src_seq[topk_block_idx] - \
                            gm_src_seq * topk_num
                        gm_src_tokens_offset = gm_src_idx * selection_topk_block_sizes
                        gm_src_block_offset = gm_src_tokens_offset // s_block_tokens
                        gm_src_block_offset_batch = select_kv_block_table[
                            _bs][_s][_hd][gm_src_block_offset]
                        gm_src_block_tokens_offset = gm_src_tokens_offset - \
                            (gm_src_block_offset * s_block_tokens)

                        end_index = s_block_tokens_offset + cu_block_size
                        select_k_rope[s_block_offset_batch][s_block_tokens_offset: end_index] = \
                            select_k_rope[gm_src_block_offset_batch][gm_src_block_tokens_offset:
                                                                     gm_src_block_tokens_offset + cu_block_size]
                        select_kv_cache[s_block_offset_batch][s_block_tokens_offset: end_index] = \
                            select_kv_cache[gm_src_block_offset_batch][gm_src_block_tokens_offset:
                                                                       gm_src_block_tokens_offset + cu_block_size]

                    # 更新 selection_kv_block_status
                    select_kv_block_status[_bs][_s][_hd][insert_idx] = topk_block_segment_id
                    if topk_block_segment_id == max_topk_id:
                        max_topk_writed_idx = insert_idx

                for neg_idx in range(valid_topk_num, topk_num):
                    select_kv_block_status[_bs][_s][_hd][neg_idx] = -1
                for postive_idx in range(valid_topk_num):
                    if select_kv_block_status[_bs][_s][_hd][postive_idx] < 0:
                        assert(False)
                select_kv_block_status[_bs][_s][_hd][topk_num] = actual_seq
                selection_kv_actual_seq_out[_bs][_s][_hd] = actual_seq
                if max_topk_writed_idx != valid_topk_num - 1:
                    gather_swap_num = gather_swap_num + 1
                    # 1. 交换kv cache数据
                    last_pos_tokens_offset = (
                        valid_topk_num - 1) * selection_topk_block_sizes
                    last_pos_block_offset = last_pos_tokens_offset // s_block_tokens
                    last_pos_block_offset_batch = select_kv_block_table[
                        _bs][_s][_hd][last_pos_block_offset]
                    last_pos_block_tokens_offset = last_pos_tokens_offset - \
                        (last_pos_block_offset * s_block_tokens)
                    end_index = last_pos_block_tokens_offset + selection_topk_block_sizes
                    lasttmp = select_k_rope[last_pos_block_offset_batch][last_pos_block_tokens_offset: end_index].copy()
                    lastkvcachetmp = select_kv_cache[last_pos_block_offset_batch][
                        last_pos_block_tokens_offset: last_pos_block_tokens_offset + selection_topk_block_sizes].copy()

                    max_pos_tokens_offset = max_topk_writed_idx * selection_topk_block_sizes
                    max_pos_block_offset = max_pos_tokens_offset // s_block_tokens
                    max_pos_block_offset_batch = select_kv_block_table[
                        _bs][_s][_hd][max_pos_block_offset]
                    max_pos_block_tokens_offset = max_pos_tokens_offset - \
                        (max_pos_block_offset * s_block_tokens)
                    cu_block_size = selection_topk_block_sizes if max_topk_id != max_selection_num - \
                        1 else last_block_size
                    maxtmp = select_k_rope[max_pos_block_offset_batch][
                        max_pos_block_tokens_offset: max_pos_block_tokens_offset + cu_block_size].copy()
                    maxkvcachetmp = select_kv_cache[max_pos_block_offset_batch][
                        max_pos_block_tokens_offset: max_pos_block_tokens_offset + cu_block_size].copy()

                    end_index = max_pos_block_tokens_offset + selection_topk_block_sizes
                    select_end_index = last_pos_block_tokens_offset + cu_block_size
                    select_k_rope[last_pos_block_offset_batch][last_pos_block_tokens_offset: select_end_index] = maxtmp
                    select_k_rope[max_pos_block_offset_batch][max_pos_block_tokens_offset: end_index] = lasttmp
                    select_kv_cache[last_pos_block_offset_batch][last_pos_block_tokens_offset:
                                                                 select_end_index] = maxkvcachetmp
                    select_kv_cache[max_pos_block_offset_batch][max_pos_block_tokens_offset: end_index] = lastkvcachetmp

                    # 2. 交换 selection_kv_block_status
                    last_status_topk_id = select_kv_block_status[_bs][_s][_hd][valid_topk_num - 1]
                    select_kv_block_status[_bs][_s][_hd][valid_topk_num -
                                                         1] = select_kv_block_status[_bs][_s][_hd][max_topk_writed_idx]
                    select_kv_block_status[_bs][_s][_hd][max_topk_writed_idx] = last_status_topk_id

    return select_k_rope, select_kv_cache, select_kv_block_table.reshape(batch_size * seq_num * head_dim, -1), \
           select_kv_block_status, selection_kv_actual_seq_out.reshape(batch_size * seq_num * head_dim)


# 1. 比较两个golden的输出 out_host:有序的topk的输出  out:无序的topk的输出
def compare_out_of_order(batch_size, seq_length, select_topk, full_q_actual_seqs, full_kv_actual_seqs,
                         selection_topk_block_sizes, select_topk_indices, s_max_block_nums, s_block_sizes,
                         out_hosts, output, lay_out, if_quant=False):
    select_kv_block_status = output[3]
    if lay_out == 'TND':
        select_topk_indices = select_topk_indices.reshape(
            batch_size, seq_length, 1, select_topk)
        select_kv_block_status = select_kv_block_status.reshape(
            batch_size, seq_length, 1, select_topk + 1)

    for _bs in range(batch_size):
        f_q_actual_sq = full_q_actual_seqs[_bs]
        for _s in range(seq_length):
            f_kv_actual_sq = full_kv_actual_seqs[_bs]
            if _s < f_q_actual_sq:
                mtp_gep = f_q_actual_sq - 1 - _s
                f_kv_actual_sq = f_kv_actual_sq - mtp_gep
            if f_kv_actual_sq <= 0:
                continue
            max_selection_num = (
                f_kv_actual_sq + selection_topk_block_sizes - 1) // selection_topk_block_sizes
            last_block_size = f_kv_actual_sq - \
                (max_selection_num - 1) * selection_topk_block_sizes

            valid_topk_num = 0
            for topk_block_idx in range(select_topk):
                topk_block_segment_id = select_topk_indices[_bs][_s][0][topk_block_idx]
                if topk_block_segment_id < 0:
                    break
                if topk_block_segment_id > max_selection_num - 1:
                    continue

                cu_block_size = selection_topk_block_sizes if topk_block_segment_id != max_selection_num - \
                    1 else last_block_size

                s_bn_batch = (_bs * seq_length + _s) * s_max_block_nums
                t_offset = valid_topk_num * selection_topk_block_sizes
                s_bn = t_offset // s_block_sizes
                s_bs = t_offset % s_block_sizes
                out_host_k_rope = out_hosts[0][s_bn + s_bn_batch][s_bs:s_bs + cu_block_size]
                out_host_kv_cache = out_hosts[1][s_bn + s_bn_batch][s_bs:s_bs + cu_block_size]

                valid_topk_num += 1
                # 从out中找到同一个topk_id对应的k_rope
                out_block_status = select_kv_block_status[_bs][_s][0][0:select_topk]
                insert_idx = -1
                mask = (out_block_status[:select_topk]
                        == topk_block_segment_id)
                valid_indices = np.where(mask)[0]
                if len(valid_indices) == 0:
                    assert(False)
                insert_idx = valid_indices[0]

                obs_t_offset = insert_idx * selection_topk_block_sizes
                obs_s_bn = obs_t_offset // s_block_sizes
                obs_s_bs = obs_t_offset % s_block_sizes
                if not if_quant:
                    out_k_rope = output[0][obs_s_bn + s_bn_batch][obs_s_bs:obs_s_bs + cu_block_size]
                else:
                    out_k_rope = True
                out_kv_cache = output[1][obs_s_bn + s_bn_batch][obs_s_bs:obs_s_bs + cu_block_size]

                is_equal_k_rope = (out_k_rope == out_host_k_rope).all() if not if_quant else True
                is_equal_kv_cache = (out_kv_cache == out_host_kv_cache).all()
                if not is_equal_k_rope or not is_equal_kv_cache:
                    assert(is_equal_k_rope)
                    assert(is_equal_kv_cache)


def random_selection_topk(all_topk_nums, batch_size, seq_length, head_num, select_topk):
    result = np.zeros((batch_size, seq_length, head_num,
                      select_topk), dtype=np.int32)
    for batch in range(batch_size):
        for seq in range(seq_length):
            for head in range(head_num):
                # 为每个位置独立选择selection_topk个不同的数字
                result[batch, seq, head] = np.random.choice(all_topk_nums, size=select_topk, replace=False)
    return result


# NPU Models
class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, selection_k_rope, selection_kv_cache, selection_kv_block_table,
                selection_kv_block_status, selection_topk_indices, full_k_rope,
                full_kv_cache, full_kv_block_table, full_kv_actual_seq, full_q_actual_seq,
                selection_topk_block_size):
        y = torch.ops.custom.npu_gather_selection_kv_cache(
            selection_k_rope, selection_kv_cache,
            selection_kv_block_table, selection_kv_block_status,
            selection_topk_indices, full_k_rope, full_kv_cache,
            full_kv_block_table, full_kv_actual_seq,
            full_q_actual_seq, selection_topk_block_size=selection_topk_block_size)
        return y


def do_npu(select_k_rope_npu, select_kv_cache_npu, select_kv_block_table_npu, select_kv_block_status_npu,
           select_topk_indices_npu, full_k_ropes_npu, full_kv_caches_npu, full_kv_block_tables_npu,
           full_kv_actual_seqs_npu, full_q_actual_seqs_npu, select_topk_block_size, api_impl_mode="dynamo"):
    if api_impl_mode == "eager":
        y = torch.ops.custom.npu_gather_selection_kv_cache(
            selection_k_rope=select_k_rope_npu, selection_kv_cache=select_kv_cache_npu,
            selection_kv_block_table=select_kv_block_table_npu, selection_kv_block_status=select_kv_block_status_npu,
            selection_topk_indices=select_topk_indices_npu, full_k_rope=full_k_ropes_npu,
            full_kv_cache=full_kv_caches_npu, full_kv_block_table=full_kv_block_tables_npu,
            full_kv_actual_seq=full_kv_actual_seqs_npu, full_q_actual_seq=full_q_actual_seqs_npu,
            selection_topk_block_size=select_topk_block_size)
    elif api_impl_mode == "dynamo":
        model = Net().npu()
        model = torch.compile(model, backend=npu_backend, dynamic=True)
        y = model(selection_k_rope=select_k_rope_npu, selection_kv_cache=select_kv_cache_npu,
                  selection_kv_block_table=select_kv_block_table_npu,
                  selection_kv_block_status=select_kv_block_status_npu,
                  selection_topk_indices=select_topk_indices_npu,
                  full_k_rope=full_k_ropes_npu, full_kv_cache=full_kv_caches_npu,
                  full_kv_block_table=full_kv_block_tables_npu, full_kv_actual_seq=full_kv_actual_seqs_npu,
                  full_q_actual_seq=full_q_actual_seqs_npu, selection_topk_block_size=select_topk_block_size)
    return select_k_rope_npu.cpu().numpy(), select_kv_cache_npu.cpu().numpy(), \
        select_kv_block_table_npu.cpu().numpy(), select_kv_block_status_npu.cpu().numpy(), y.cpu().numpy()


class TestCustomGatherSelectionKvCache(TestCase):
    def test_gather_selection_kv_cache_eager(self):
        k_rope = 64
        kvcahce = 512
        selection_topk_block_size = 1
        batchsize = 4
        seq_len = 4
        headnum = 1
        selection_topk = int(2048*0.17)
        print(selection_topk)
        max_seq_len = 65536
        s_block_size = 128
        f_block_size = 128
        lay_out = 'BSND'  # 'TND'
        reuse_input = False
        is_offload = True

        selection_max_seq_len = selection_topk * selection_topk_block_size
        all_topk_num = np.arange(0, (max_seq_len + selection_topk_block_size -
                                1) // selection_topk_block_size, dtype=np.int32)
        
        # 每个batch 最大blocknum
        s_max_block_num = (selection_max_seq_len + s_block_size - 1) // s_block_size
        f_max_block_num = (max_seq_len + f_block_size - 1) // f_block_size
        selection_k_rope = np.random.uniform(
            size=[s_max_block_num * batchsize * seq_len * headnum, s_block_size, k_rope]).astype(np.float16)
        selection_kv_cache = np.random.uniform(
            size=[s_max_block_num * batchsize * seq_len * headnum, s_block_size, kvcahce]).astype(np.float16)
        selection_kv_block_table = np.arange(0, batchsize * seq_len * headnum * s_max_block_num).reshape(
            batchsize * seq_len * headnum, s_max_block_num).astype(np.int32)
        selection_kv_block_status = np.ones(
            (batchsize, seq_len, headnum, selection_topk + 1), dtype=np.int32) * -1
        selection_topk_indices = random_selection_topk(
            all_topk_num, batchsize, seq_len, headnum, selection_topk)

        full_k_rope = np.random.uniform(
            size=[f_max_block_num * batchsize, f_block_size, k_rope]).astype(np.float16)
        full_kv_cache = np.random.uniform(
            size=[f_max_block_num * batchsize, f_block_size, kvcahce]).astype(np.float16)

        full_kv_block_table = np.random.uniform(low=0, high=f_max_block_num, size=[
                                                batchsize, f_max_block_num]).astype(np.int32)

        full_kv_actual_seq = np.random.uniform(
            low=max_seq_len, high=max_seq_len, size=[batchsize]).astype(np.int32)
        full_q_actual_seq = np.random.uniform(
            low=0, high=0, size=[batchsize]).astype(np.int32) + seq_len
        
        # 1. 产生reuse可复用的输入
        out_for_in = do_golden_all_host(batchsize, seq_len, headnum, selection_topk, selection_k_rope,
                                        selection_kv_cache, selection_kv_block_table, selection_kv_block_status,
                                        selection_topk_indices, full_k_rope, full_kv_cache, full_kv_block_table,
                                        full_kv_actual_seq, full_q_actual_seq, selection_topk_block_size, lay_out)
        selection_k_rope = out_for_in[0]
        selection_kv_cache = out_for_in[1]
        selection_kv_block_table = out_for_in[2]
        selection_kv_block_status = out_for_in[3]
        
        # 2.2 控制topk复用率
        for b in range(batchsize):
            curvv = selection_topk_indices[b].copy()
            useable = list(set(all_topk_num) - set(curvv.reshape(-1)))
            for s in range(seq_len):
                if s > 0:
                    useable = list(
                        set(useable) - set(selection_topk_indices[b][s - 1].copy().reshape(-1)))
                for h in range(headnum):
                    not_dup_num = selection_topk  # 2048
                    not_dup_value = np.random.choice(
                        useable, size=not_dup_num, replace=False)
                    not_dup_idx = np.random.choice(
                        np.arange(selection_topk), size=not_dup_num, replace=False)
                    selection_topk_indices[b][s][h][not_dup_idx] = not_dup_value

        if lay_out == 'TND':
            selection_kv_block_status = selection_kv_block_status.reshape(
                batchsize * seq_len, headnum, selection_topk + 1)
            selection_topk_indices = selection_topk_indices.reshape(
                batchsize * seq_len, headnum, selection_topk)

        selection_k_rope_npu = torch.from_numpy(selection_k_rope.copy()).npu()
        selection_kv_cache_npu = torch.from_numpy(selection_kv_cache.copy()).npu()
        selection_kv_block_table_npu = torch.from_numpy(
            selection_kv_block_table.copy()).npu()
        selection_kv_block_status_npu = torch.from_numpy(
            selection_kv_block_status.copy()).npu()
        selection_topk_indices_npu = torch.from_numpy(
            selection_topk_indices.copy()).npu()
        full_k_rope_npu = torch.from_numpy(full_k_rope.copy()).npu()
        full_kv_cache_npu = torch.from_numpy(full_kv_cache.copy()).npu()
        full_kv_block_table_npu = torch.from_numpy(full_kv_block_table.copy()).npu()
        full_kv_actual_seq_npu = torch.from_numpy(full_kv_actual_seq.copy()).npu()
        full_q_actual_seq_npu = torch.from_numpy(full_q_actual_seq.copy()).npu()

        selection_k_rope_h = selection_k_rope.copy()
        selection_kv_cache_h = selection_kv_cache.copy()
        selection_kv_block_table_h = selection_kv_block_table.copy()
        selection_kv_block_status_h = selection_kv_block_status.copy()
        selection_topk_indices_h = selection_topk_indices.copy()
        out_host = do_golden_all_host(batchsize, seq_len, headnum, selection_topk, selection_k_rope_h,
                                      selection_kv_cache_h, selection_kv_block_table_h,
                                      selection_kv_block_status_h, selection_topk_indices_h, full_k_rope,
                                      full_kv_cache, full_kv_block_table, full_kv_actual_seq,
                                      full_q_actual_seq, selection_topk_block_size, lay_out)

        if is_offload:
            full_k_rope_npu = torch_npu.empty_with_swapped_memory(
                full_k_rope_npu.shape, dtype=full_k_rope_npu.dtype, device=torch.device("npu:0"))
            full_k_rope_npu.fill_(0)
            full_k_rope_npu.add_(torch.from_numpy(full_k_rope.copy()).npu())

            full_kv_cache_npu = torch_npu.empty_with_swapped_memory(
                full_kv_cache_npu.shape, dtype=full_k_rope_npu.dtype, device=torch.device("npu:0"))
            full_kv_cache_npu.fill_(0)
            full_kv_cache_npu.add_(torch.from_numpy(full_kv_cache.copy()).npu())
        #####################
        model = Net().npu()
        model = torch.compile(model, backend=npu_backend, dynamic=True)
        import time
        for i in range(10):
            y = model(selection_k_rope_npu, selection_kv_cache_npu, selection_kv_block_table_npu,
                            selection_kv_block_status_npu, selection_topk_indices_npu, full_k_rope_npu,
                            full_kv_cache_npu, full_kv_block_table_npu, full_kv_actual_seq_npu,
                            full_q_actual_seq_npu, selection_topk_block_size)
        torch.npu.synchronize()
        start = time.time()
        # with torch_npu.profiler.profile(activities=[torch_npu.profiler.ProfilerActivity.CPU,torch_npu.profiler.ProfilerActivity.NPU],record_shapes=True, 
        # on_trace_ready=torch_npu.profiler.tensorboard_trace_handler("./result_li"), 
        # schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=10, repeat=1, skip_first=10), ) as prof:
        for i in range(100):
            y = model(selection_k_rope_npu, selection_kv_cache_npu, selection_kv_block_table_npu,
                        selection_kv_block_status_npu, selection_topk_indices_npu, full_k_rope_npu,
                        full_kv_cache_npu, full_kv_block_table_npu, full_kv_actual_seq_npu,
                        full_q_actual_seq_npu, selection_topk_block_size)
                # prof.step()
        torch.npu.synchronize()
        end = time.time()
        print(f'graph use time:{(end-start)/100}')
        print(f'======================== PTA eager BEGIN ========================')
        out = do_golden_gen(batchsize, seq_len, headnum, selection_topk, selection_k_rope, selection_kv_cache,
                            selection_kv_block_table, selection_kv_block_status, selection_topk_indices,
                            full_k_rope, full_kv_cache, full_kv_block_table, full_kv_actual_seq, full_q_actual_seq,
                            selection_topk_block_size, lay_out)

        outnpu = do_npu(selection_k_rope_npu, selection_kv_cache_npu, selection_kv_block_table_npu,
                        selection_kv_block_status_npu, selection_topk_indices_npu, full_k_rope_npu,
                        full_kv_cache_npu, full_kv_block_table_npu, full_kv_actual_seq_npu,
                        full_q_actual_seq_npu, selection_topk_block_size, api_impl_mode="eager")
        print(f'======================== PTA eager FINISH ========================')
        
        if lay_out == 'TND':
            selection_topk_indices = selection_topk_indices.reshape(
                batchsize * seq_len, headnum, selection_topk)

        compare_out_of_order(batchsize, seq_len, selection_topk, full_q_actual_seq, full_kv_actual_seq,
                            selection_topk_block_size, selection_topk_indices, s_max_block_num,
                            s_block_size, out_host, out, lay_out)

        kv_blk_t_equal = (out[2] == out_host[2].reshape(out[2].shape)).all()
        kv_blk_s_equal = (sort_with_negative_ones_last(
            out[3]) == sort_with_negative_ones_last(out_host[3])).all()
        kv_blk_seq_equ = (out[4] == out_host[4].reshape(out[4].shape)).all()
        assert(kv_blk_t_equal)
        assert(kv_blk_s_equal)
        assert(kv_blk_seq_equ)

        # 2. 比较npu和golden的输出
        if selection_topk <= 32:
            assert((out[0] == outnpu[0].reshape(out[0].shape)).all())
            assert((out[1] == outnpu[1].reshape(out[1].shape)).all())
            assert((out[2] == outnpu[2].reshape(out[2].shape)).all())
            assert((out[3] == outnpu[3].reshape(out[3].shape)).all())
            assert((out[4] == outnpu[4].reshape(out[4].shape)).all())
        else:
            compare_out_of_order(batchsize, seq_len, selection_topk, full_q_actual_seq, full_kv_actual_seq,
                                selection_topk_block_size, selection_topk_indices, s_max_block_num, s_block_size,
                                out_host, outnpu, lay_out)
            kv_blk_t_equal = (outnpu[2] == out_host[2].reshape(outnpu[2].shape)).all()
            if lay_out == 'TND':
                kv_blk_s_equal = (sort_with_negative_ones_last(
                    outnpu[3]) == sort_with_negative_ones_last(out_host[3]).reshape(
                        batchsize * seq_len, headnum, selection_topk + 1)).all()
            else:
                kv_blk_s_equal = (sort_with_negative_ones_last(
                    outnpu[3]) == sort_with_negative_ones_last(out_host[3])).all()
            kv_blk_seq_equ = (outnpu[4] == out_host[4].reshape(outnpu[4].shape)).all()
            assert(kv_blk_t_equal)
            assert(kv_blk_s_equal)
            assert(kv_blk_seq_equ)
            
    def test_gather_selection_kv_cache_graph(self):
        k_rope = 64
        kvcahce = 512
        selection_topk_block_size = 1
        batchsize = 16
        seq_len = 4
        headnum = 1
        selection_topk = int(2048*0.34)
        max_seq_len = 65536
        s_block_size = 128
        f_block_size = 128
        lay_out = 'BSND'  # 'TND'
        reuse_input = False
        is_offload = True

        selection_max_seq_len = selection_topk * selection_topk_block_size
        all_topk_num = np.arange(0, (max_seq_len + selection_topk_block_size -
                                1) // selection_topk_block_size, dtype=np.int32)
        
        # 每个batch 最大blocknum
        s_max_block_num = (selection_max_seq_len + s_block_size - 1) // s_block_size
        f_max_block_num = (max_seq_len + f_block_size - 1) // f_block_size
        selection_k_rope = np.random.uniform(
            size=[s_max_block_num * batchsize * seq_len * headnum, s_block_size, k_rope]).astype(np.float16)
        selection_kv_cache = np.random.uniform(
            size=[s_max_block_num * batchsize * seq_len * headnum, s_block_size, kvcahce]).astype(np.float16)
        selection_kv_block_table = np.arange(0, batchsize * seq_len * headnum * s_max_block_num).reshape(
            batchsize * seq_len * headnum, s_max_block_num).astype(np.int32)
        selection_kv_block_status = np.ones(
            (batchsize, seq_len, headnum, selection_topk + 1), dtype=np.int32) * -1
        selection_topk_indices = random_selection_topk(
            all_topk_num, batchsize, seq_len, headnum, selection_topk)

        full_k_rope = np.random.uniform(
            size=[f_max_block_num * batchsize, f_block_size, k_rope]).astype(np.float16)
        full_kv_cache = np.random.uniform(
            size=[f_max_block_num * batchsize, f_block_size, kvcahce]).astype(np.float16)

        full_kv_block_table = np.random.uniform(low=0, high=f_max_block_num, size=[
                                                batchsize, f_max_block_num]).astype(np.int32)

        full_kv_actual_seq = np.random.uniform(
            low=max_seq_len, high=max_seq_len, size=[batchsize]).astype(np.int32)
        full_q_actual_seq = np.random.uniform(
            low=0, high=0, size=[batchsize]).astype(np.int32) + seq_len
        
        # 1. 产生reuse可复用的输入
        out_for_in = do_golden_all_host(batchsize, seq_len, headnum, selection_topk, selection_k_rope,
                                        selection_kv_cache, selection_kv_block_table,
                                        selection_kv_block_status, selection_topk_indices, full_k_rope,
                                        full_kv_cache, full_kv_block_table, full_kv_actual_seq,
                                        full_q_actual_seq, selection_topk_block_size, lay_out)
        selection_k_rope = out_for_in[0]
        selection_kv_cache = out_for_in[1]
        selection_kv_block_table = out_for_in[2]
        selection_kv_block_status = out_for_in[3]
        
        # 2.2 控制topk复用率
        for b in range(batchsize):
            curvv = selection_topk_indices[b].copy()
            useable = list(set(all_topk_num) - set(curvv.reshape(-1)))
            for s in range(seq_len):
                if s > 0:
                    useable = list(
                        set(useable) - set(selection_topk_indices[b][s - 1].copy().reshape(-1)))
                for h in range(headnum):
                    not_dup_num = selection_topk  # 2048
                    not_dup_value = np.random.choice(
                        useable, size=not_dup_num, replace=False)
                    not_dup_idx = np.random.choice(
                        np.arange(selection_topk), size=not_dup_num, replace=False)
                    selection_topk_indices[b][s][h][not_dup_idx] = not_dup_value

        if lay_out == 'TND':
            selection_kv_block_status = selection_kv_block_status.reshape(
                batchsize * seq_len, headnum, selection_topk + 1)
            selection_topk_indices = selection_topk_indices.reshape(
                batchsize * seq_len, headnum, selection_topk)

        selection_k_rope_npu = torch.from_numpy(selection_k_rope.copy()).npu()
        selection_kv_cache_npu = torch.from_numpy(selection_kv_cache.copy()).npu()
        selection_kv_block_table_npu = torch.from_numpy(
            selection_kv_block_table.copy()).npu()
        selection_kv_block_status_npu = torch.from_numpy(
            selection_kv_block_status.copy()).npu()
        selection_topk_indices_npu = torch.from_numpy(
            selection_topk_indices.copy()).npu()
        full_k_rope_npu = torch.from_numpy(full_k_rope.copy()).npu()
        full_kv_cache_npu = torch.from_numpy(full_kv_cache.copy()).npu()
        full_kv_block_table_npu = torch.from_numpy(full_kv_block_table.copy()).npu()
        full_kv_actual_seq_npu = torch.from_numpy(full_kv_actual_seq.copy()).npu()
        full_q_actual_seq_npu = torch.from_numpy(full_q_actual_seq.copy()).npu()

        selection_k_rope_h = selection_k_rope.copy()
        selection_kv_cache_h = selection_kv_cache.copy()
        selection_kv_block_table_h = selection_kv_block_table.copy()
        selection_kv_block_status_h = selection_kv_block_status.copy()
        selection_topk_indices_h = selection_topk_indices.copy()
        out_host = do_golden_all_host(batchsize, seq_len, headnum, selection_topk, selection_k_rope_h,
                                      selection_kv_cache_h, selection_kv_block_table_h,
                                      selection_kv_block_status_h, selection_topk_indices_h, full_k_rope,
                                      full_kv_cache, full_kv_block_table, full_kv_actual_seq,
                                      full_q_actual_seq, selection_topk_block_size, lay_out)

        if is_offload:
            full_k_rope_npu = torch_npu.empty_with_swapped_memory(
                full_k_rope_npu.shape, dtype=full_k_rope_npu.dtype, device=torch.device("npu:0"))
            full_k_rope_npu.fill_(0)
            full_k_rope_npu.add_(torch.from_numpy(full_k_rope.copy()).npu())

            full_kv_cache_npu = torch_npu.empty_with_swapped_memory(
                full_kv_cache_npu.shape, dtype=full_k_rope_npu.dtype, device=torch.device("npu:0"))
            full_kv_cache_npu.fill_(0)
            full_kv_cache_npu.add_(torch.from_numpy(full_kv_cache.copy()).npu())

        print(f'======================== PTA graph BEGIN ========================')
        out = do_golden_gen(batchsize, seq_len, headnum, selection_topk, selection_k_rope, selection_kv_cache,
                            selection_kv_block_table, selection_kv_block_status, selection_topk_indices,
                            full_k_rope, full_kv_cache, full_kv_block_table, full_kv_actual_seq, full_q_actual_seq,
                            selection_topk_block_size, lay_out)


        ##########test
        model = Net().npu()
        model = torch.compile(model, backend=npu_backend, dynamic=True)
        import time
        for i in range(10):
            y = model(selection_k_rope_npu, selection_kv_cache_npu, selection_kv_block_table_npu,
                            selection_kv_block_status_npu, selection_topk_indices_npu, full_k_rope_npu,
                            full_kv_cache_npu, full_kv_block_table_npu, full_kv_actual_seq_npu,
                            full_q_actual_seq_npu, selection_topk_block_size)
        torch.npu.synchronize()
        start = time.time()
        with torch_npu.profiler.profile(activities=[torch_npu.profiler.ProfilerActivity.CPU,torch_npu.profiler.ProfilerActivity.NPU],record_shapes=True, 
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(f"./result_offload_{batchsize}_{selection_topk}"), 
        schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=10, repeat=1, skip_first=10), ) as prof:
            for i in range(20):
                y = model(selection_k_rope_npu, selection_kv_cache_npu, selection_kv_block_table_npu,
                            selection_kv_block_status_npu, selection_topk_indices_npu, full_k_rope_npu,
                            full_kv_cache_npu, full_kv_block_table_npu, full_kv_actual_seq_npu,
                            full_q_actual_seq_npu, selection_topk_block_size)
                prof.step()
        torch.npu.synchronize()
        end = time.time()
        outnpu = do_npu(selection_k_rope_npu, selection_kv_cache_npu, selection_kv_block_table_npu,
                        selection_kv_block_status_npu, selection_topk_indices_npu, full_k_rope_npu,
                        full_kv_cache_npu, full_kv_block_table_npu, full_kv_actual_seq_npu,
                        full_q_actual_seq_npu, selection_topk_block_size, api_impl_mode="dynamo")
        print(f'======================== PTA graph FINISH ========================')
        if lay_out == 'TND':
            selection_topk_indices = selection_topk_indices.reshape(
                batchsize * seq_len, headnum, selection_topk)

        compare_out_of_order(batchsize, seq_len, selection_topk, full_q_actual_seq, full_kv_actual_seq,
                            selection_topk_block_size, selection_topk_indices, s_max_block_num,
                            s_block_size, out_host, out, lay_out)

        kv_blk_t_equal = (out[2] == out_host[2].reshape(out[2].shape)).all()
        kv_blk_s_equal = (sort_with_negative_ones_last(
            out[3]) == sort_with_negative_ones_last(out_host[3])).all()
        kv_blk_seq_equ = (out[4] == out_host[4].reshape(out[4].shape)).all()
        assert(kv_blk_t_equal)
        assert(kv_blk_s_equal)
        assert(kv_blk_seq_equ)

        # 2. 比较npu和golden的输出
        if selection_topk <= 32:
            assert((out[0] == outnpu[0].reshape(out[0].shape)).all())
            assert((out[1] == outnpu[1].reshape(out[1].shape)).all())
            assert((out[2] == outnpu[2].reshape(out[2].shape)).all())
            assert((out[3] == outnpu[3].reshape(out[3].shape)).all())
            assert((out[4] == outnpu[4].reshape(out[4].shape)).all())
        else:
            compare_out_of_order(batchsize, seq_len, selection_topk, full_q_actual_seq, full_kv_actual_seq,
                                selection_topk_block_size, selection_topk_indices, s_max_block_num, s_block_size,
                                out_host, outnpu, lay_out)
            kv_blk_t_equal = (outnpu[2] == out_host[2].reshape(outnpu[2].shape)).all()
            if lay_out == 'TND':
                kv_blk_s_equal = (sort_with_negative_ones_last(
                    outnpu[3]) == sort_with_negative_ones_last(out_host[3]).reshape(
                        batchsize * seq_len, headnum, selection_topk + 1)).all()
            else:
                kv_blk_s_equal = (sort_with_negative_ones_last(
                    outnpu[3]) == sort_with_negative_ones_last(out_host[3])).all()
            kv_blk_seq_equ = (outnpu[4] == out_host[4].reshape(outnpu[4].shape)).all()
            assert(kv_blk_t_equal)
            assert(kv_blk_s_equal)
            assert(kv_blk_seq_equ)

    def test_gather_selection_kv_cache_quant_graph(self):
        k_rope = 656
        kvcahce = 512
        selection_topk_block_size = 1
        batchsize = 2
        seq_len = 2
        headnum = 1
        selection_topk = 2048
        max_seq_len = 1024 * 16
        s_block_size = 128
        f_block_size = 128
        lay_out = 'BSND'  # 'TND'
        reuse_input = False
        is_offload = False

        selection_max_seq_len = selection_topk * selection_topk_block_size
        all_topk_num = np.arange(0, (max_seq_len + selection_topk_block_size -
                                1) // selection_topk_block_size, dtype=np.int32)
        
        # 每个batch 最大blocknum
        s_max_block_num = (selection_max_seq_len + s_block_size - 1) // s_block_size
        f_max_block_num = (max_seq_len + f_block_size - 1) // f_block_size
        selection_k_rope = np.random.uniform(low=0, high=127,
            size=[s_max_block_num * batchsize * seq_len * headnum, s_block_size, k_rope]).astype(np.int8)
        selection_kv_cache = np.random.uniform(low=0, high=127,
            size=[s_max_block_num * batchsize * seq_len * headnum, s_block_size, kvcahce]).astype(np.int8)
        selection_kv_block_table = np.arange(0, batchsize * seq_len * headnum * s_max_block_num).reshape(
            batchsize * seq_len * headnum, s_max_block_num).astype(np.int32)
        selection_kv_block_status = np.ones(
            (batchsize, seq_len, headnum, selection_topk + 1), dtype=np.int32) * -1
        selection_topk_indices = random_selection_topk(
            all_topk_num, batchsize, seq_len, headnum, selection_topk)

        full_k_rope = np.random.uniform(low=0, high=127,
            size=[f_max_block_num * batchsize, f_block_size, k_rope]).astype(np.int8)
        full_kv_cache = np.random.uniform(low=0, high=127,
            size=[f_max_block_num * batchsize, f_block_size, kvcahce]).astype(np.int8)

        full_kv_block_table = np.random.uniform(low=0, high=f_max_block_num, size=[
                                                batchsize, f_max_block_num]).astype(np.int32)

        full_kv_actual_seq = np.random.uniform(
            low=max_seq_len, high=max_seq_len, size=[batchsize]).astype(np.int32)
        full_q_actual_seq = np.random.uniform(
            low=0, high=0, size=[batchsize]).astype(np.int32) + seq_len
        
        # 1. 产生reuse可复用的输入
        out_for_in = do_golden_all_host(batchsize, seq_len, headnum, selection_topk, selection_k_rope,
                                        selection_kv_cache, selection_kv_block_table,
                                        selection_kv_block_status, selection_topk_indices, full_k_rope,
                                        full_kv_cache, full_kv_block_table, full_kv_actual_seq,
                                        full_q_actual_seq, selection_topk_block_size, lay_out)
        selection_k_rope = out_for_in[0]
        selection_kv_cache = out_for_in[1]
        selection_kv_block_table = out_for_in[2]
        selection_kv_block_status = out_for_in[3]
        
        # 2.2 控制topk复用率
        for b in range(batchsize):
            curvv = selection_topk_indices[b].copy()
            useable = list(set(all_topk_num) - set(curvv.reshape(-1)))
            for s in range(seq_len):
                if s > 0:
                    useable = list(
                        set(useable) - set(selection_topk_indices[b][s - 1].copy().reshape(-1)))
                for h in range(headnum):
                    not_dup_num = 2048  # 2048
                    not_dup_value = np.random.choice(
                        useable, size=not_dup_num, replace=False)
                    not_dup_idx = np.random.choice(
                        np.arange(selection_topk), size=not_dup_num, replace=False)
                    selection_topk_indices[b][s][h][not_dup_idx] = not_dup_value

        if lay_out == 'TND':
            selection_kv_block_status = selection_kv_block_status.reshape(
                batchsize * seq_len, headnum, selection_topk + 1)
            selection_topk_indices = selection_topk_indices.reshape(
                batchsize * seq_len, headnum, selection_topk)

        selection_k_rope_npu = torch.from_numpy(selection_k_rope.copy()).npu()
        selection_kv_cache_npu = torch.from_numpy(selection_kv_cache.copy()).npu()
        selection_kv_block_table_npu = torch.from_numpy(
            selection_kv_block_table.copy()).npu()
        selection_kv_block_status_npu = torch.from_numpy(
            selection_kv_block_status.copy()).npu()
        selection_topk_indices_npu = torch.from_numpy(
            selection_topk_indices.copy()).npu()
        full_k_rope_npu = torch.from_numpy(full_k_rope.copy()).npu()
        full_kv_cache_npu = torch.from_numpy(full_kv_cache.copy()).npu()
        full_kv_block_table_npu = torch.from_numpy(full_kv_block_table.copy()).npu()
        full_kv_actual_seq_npu = torch.from_numpy(full_kv_actual_seq.copy()).npu()
        full_q_actual_seq_npu = torch.from_numpy(full_q_actual_seq.copy()).npu()

        selection_k_rope_h = selection_k_rope.copy()
        selection_kv_cache_h = selection_kv_cache.copy()
        selection_kv_block_table_h = selection_kv_block_table.copy()
        selection_kv_block_status_h = selection_kv_block_status.copy()
        selection_topk_indices_h = selection_topk_indices.copy()
        out_host = do_golden_all_host(batchsize, seq_len, headnum, selection_topk, selection_k_rope_h,
                                      selection_kv_cache_h, selection_kv_block_table_h,
                                      selection_kv_block_status_h, selection_topk_indices_h, full_k_rope,
                                      full_kv_cache, full_kv_block_table, full_kv_actual_seq,
                                      full_q_actual_seq, selection_topk_block_size, lay_out)

        if is_offload:
            full_k_rope_npu = torch_npu.empty_with_swapped_memory(
                full_k_rope_npu.shape, dtype=full_k_rope_npu.dtype, device=torch.device("npu:0"))
            full_k_rope_npu.fill_(0)
            full_k_rope_npu.add_(torch.from_numpy(full_k_rope.copy()).npu())

            full_kv_cache_npu = torch_npu.empty_with_swapped_memory(
                full_kv_cache_npu.shape, dtype=full_k_rope_npu.dtype, device=torch.device("npu:0"))
            full_kv_cache_npu.fill_(0)
            full_kv_cache_npu.add_(torch.from_numpy(full_kv_cache.copy()).npu())

        print(f'======================== PTA graph BEGIN ========================')
        out = do_golden_gen(batchsize, seq_len, headnum, selection_topk, selection_k_rope, selection_kv_cache,
                            selection_kv_block_table, selection_kv_block_status, selection_topk_indices,
                            full_k_rope, full_kv_cache, full_kv_block_table, full_kv_actual_seq, full_q_actual_seq,
                            selection_topk_block_size, lay_out)
        selection_k_rope_npu = torch.tensor([], dtype=torch.int8).npu()
        full_k_rope_npu = torch.tensor([], dtype=torch.int8).npu()
        outnpu = do_npu(selection_k_rope_npu, selection_kv_cache_npu, selection_kv_block_table_npu,
                        selection_kv_block_status_npu, selection_topk_indices_npu, full_k_rope_npu,
                        full_kv_cache_npu, full_kv_block_table_npu, full_kv_actual_seq_npu,
                        full_q_actual_seq_npu, selection_topk_block_size, api_impl_mode="dynamo")
        print(f'======================== PTA graph FINISH ========================')
        if lay_out == 'TND':
            selection_topk_indices = selection_topk_indices.reshape(
                batchsize * seq_len, headnum, selection_topk)

        compare_out_of_order(batchsize, seq_len, selection_topk, full_q_actual_seq, full_kv_actual_seq,
                            selection_topk_block_size, selection_topk_indices, s_max_block_num,
                            s_block_size, out_host, out, lay_out)

        kv_blk_t_equal = (out[2] == out_host[2].reshape(out[2].shape)).all()
        kv_blk_s_equal = (sort_with_negative_ones_last(
            out[3]) == sort_with_negative_ones_last(out_host[3])).all()
        kv_blk_seq_equ = (out[4] == out_host[4].reshape(out[4].shape)).all()
        assert(kv_blk_t_equal)
        assert(kv_blk_s_equal)
        assert(kv_blk_seq_equ)

        # 2. 比较npu和golden的输出
        if selection_topk <= 32:
            assert((out[0] == outnpu[0].reshape(out[0].shape)).all())
            assert((out[1] == outnpu[1].reshape(out[1].shape)).all())
            assert((out[2] == outnpu[2].reshape(out[2].shape)).all())
            assert((out[3] == outnpu[3].reshape(out[3].shape)).all())
            assert((out[4] == outnpu[4].reshape(out[4].shape)).all())
        else:
            compare_out_of_order(batchsize, seq_len, selection_topk, full_q_actual_seq, full_kv_actual_seq,
                                selection_topk_block_size, selection_topk_indices, s_max_block_num, s_block_size,
                                out_host, outnpu, lay_out, if_quant=True)
            kv_blk_t_equal = (outnpu[2] == out_host[2].reshape(outnpu[2].shape)).all()
            if lay_out == 'TND':
                kv_blk_s_equal = (sort_with_negative_ones_last(
                    outnpu[3]) == sort_with_negative_ones_last(out_host[3]).reshape(
                        batchsize * seq_len, headnum, selection_topk + 1)).all()
            else:
                kv_blk_s_equal = (sort_with_negative_ones_last(
                    outnpu[3]) == sort_with_negative_ones_last(out_host[3])).all()
            kv_blk_seq_equ = (outnpu[4] == out_host[4].reshape(outnpu[4].shape)).all()
            assert(kv_blk_t_equal)
            assert(kv_blk_s_equal)
            assert(kv_blk_seq_equ)
            
    def test_gather_selection_kv_cache_tnd_graph(self):
        k_rope = 64
        kvcahce = 512
        selection_topk_block_size = 1
        batchsize = 2
        seq_len = 2
        headnum = 1
        selection_topk = 2048
        max_seq_len = 1024 * 16
        s_block_size = 128
        f_block_size = 128
        lay_out = 'TND'  # 'TND'
        reuse_input = False
        is_offload = False

        selection_max_seq_len = selection_topk * selection_topk_block_size
        all_topk_num = np.arange(0, (max_seq_len + selection_topk_block_size -
                                1) // selection_topk_block_size, dtype=np.int32)
        
        # 每个batch 最大blocknum
        s_max_block_num = (selection_max_seq_len + s_block_size - 1) // s_block_size
        f_max_block_num = (max_seq_len + f_block_size - 1) // f_block_size
        selection_k_rope = np.random.uniform(
            size=[s_max_block_num * batchsize * seq_len * headnum, s_block_size, k_rope]).astype(np.float16)
        selection_kv_cache = np.random.uniform(
            size=[s_max_block_num * batchsize * seq_len * headnum, s_block_size, kvcahce]).astype(np.float16)
        selection_kv_block_table = np.arange(0, batchsize * seq_len * headnum * s_max_block_num).reshape(
            batchsize * seq_len * headnum, s_max_block_num).astype(np.int32)
        selection_kv_block_status = np.ones(
            (batchsize, seq_len, headnum, selection_topk + 1), dtype=np.int32) * -1
        selection_topk_indices = random_selection_topk(
            all_topk_num, batchsize, seq_len, headnum, selection_topk)

        full_k_rope = np.random.uniform(
            size=[f_max_block_num * batchsize, f_block_size, k_rope]).astype(np.float16)
        full_kv_cache = np.random.uniform(
            size=[f_max_block_num * batchsize, f_block_size, kvcahce]).astype(np.float16)

        full_kv_block_table = np.random.uniform(low=0, high=f_max_block_num, size=[
                                                batchsize, f_max_block_num]).astype(np.int32)

        full_kv_actual_seq = np.random.uniform(
            low=max_seq_len, high=max_seq_len, size=[batchsize]).astype(np.int32)
        full_q_actual_seq = np.random.uniform(
            low=0, high=0, size=[batchsize]).astype(np.int32) + seq_len
        
        # 1. 产生reuse可复用的输入
        out_for_in = do_golden_all_host(batchsize, seq_len, headnum, selection_topk, selection_k_rope,
                                        selection_kv_cache, selection_kv_block_table,
                                        selection_kv_block_status, selection_topk_indices, full_k_rope,
                                        full_kv_cache, full_kv_block_table, full_kv_actual_seq,
                                        full_q_actual_seq, selection_topk_block_size, lay_out)
        selection_k_rope = out_for_in[0]
        selection_kv_cache = out_for_in[1]
        selection_kv_block_table = out_for_in[2]
        selection_kv_block_status = out_for_in[3]
        
        # 2.2 控制topk复用率
        for b in range(batchsize):
            curvv = selection_topk_indices[b].copy()
            useable = list(set(all_topk_num) - set(curvv.reshape(-1)))
            for s in range(seq_len):
                if s > 0:
                    useable = list(
                        set(useable) - set(selection_topk_indices[b][s - 1].copy().reshape(-1)))
                for h in range(headnum):
                    not_dup_num = 2048  # 2048
                    not_dup_value = np.random.choice(
                        useable, size=not_dup_num, replace=False)
                    not_dup_idx = np.random.choice(
                        np.arange(selection_topk), size=not_dup_num, replace=False)
                    selection_topk_indices[b][s][h][not_dup_idx] = not_dup_value

        if lay_out == 'TND':
            selection_kv_block_status = selection_kv_block_status.reshape(
                batchsize * seq_len, headnum, selection_topk + 1)
            selection_topk_indices = selection_topk_indices.reshape(
                batchsize * seq_len, headnum, selection_topk)

        selection_k_rope_npu = torch.from_numpy(selection_k_rope.copy()).npu()
        selection_kv_cache_npu = torch.from_numpy(selection_kv_cache.copy()).npu()
        selection_kv_block_table_npu = torch.from_numpy(
            selection_kv_block_table.copy()).npu()
        selection_kv_block_status_npu = torch.from_numpy(
            selection_kv_block_status.copy()).npu()
        selection_topk_indices_npu = torch.from_numpy(
            selection_topk_indices.copy()).npu()
        full_k_rope_npu = torch.from_numpy(full_k_rope.copy()).npu()
        full_kv_cache_npu = torch.from_numpy(full_kv_cache.copy()).npu()
        full_kv_block_table_npu = torch.from_numpy(full_kv_block_table.copy()).npu()
        full_kv_actual_seq_npu = torch.from_numpy(full_kv_actual_seq.copy()).npu()
        full_q_actual_seq_npu = torch.from_numpy(full_q_actual_seq.copy()).npu()

        selection_k_rope_h = selection_k_rope.copy()
        selection_kv_cache_h = selection_kv_cache.copy()
        selection_kv_block_table_h = selection_kv_block_table.copy()
        selection_kv_block_status_h = selection_kv_block_status.copy()
        selection_topk_indices_h = selection_topk_indices.copy()
        out_host = do_golden_all_host(batchsize, seq_len, headnum, selection_topk, selection_k_rope_h,
                                      selection_kv_cache_h, selection_kv_block_table_h,
                                      selection_kv_block_status_h, selection_topk_indices_h, full_k_rope,
                                      full_kv_cache, full_kv_block_table, full_kv_actual_seq,
                                      full_q_actual_seq, selection_topk_block_size, lay_out)

        if is_offload:
            full_k_rope_npu = torch_npu.empty_with_swapped_memory(
                full_k_rope_npu.shape, dtype=full_k_rope_npu.dtype, device=torch.device("npu:0"))
            full_k_rope_npu.fill_(0)
            full_k_rope_npu.add_(torch.from_numpy(full_k_rope.copy()).npu())

            full_kv_cache_npu = torch_npu.empty_with_swapped_memory(
                full_kv_cache_npu.shape, dtype=full_k_rope_npu.dtype, device=torch.device("npu:0"))
            full_kv_cache_npu.fill_(0)
            full_kv_cache_npu.add_(torch.from_numpy(full_kv_cache.copy()).npu())

        print(f'======================== PTA graph BEGIN ========================')
        out = do_golden_gen(batchsize, seq_len, headnum, selection_topk, selection_k_rope, selection_kv_cache,
                            selection_kv_block_table, selection_kv_block_status, selection_topk_indices,
                            full_k_rope, full_kv_cache, full_kv_block_table, full_kv_actual_seq, full_q_actual_seq,
                            selection_topk_block_size, lay_out)

        outnpu = do_npu(selection_k_rope_npu, selection_kv_cache_npu, selection_kv_block_table_npu,
                        selection_kv_block_status_npu, selection_topk_indices_npu, full_k_rope_npu,
                        full_kv_cache_npu, full_kv_block_table_npu, full_kv_actual_seq_npu,
                        full_q_actual_seq_npu, selection_topk_block_size, api_impl_mode="dynamo")
        print(f'======================== PTA graph FINISH ========================')
        if lay_out == 'TND':
            selection_topk_indices = selection_topk_indices.reshape(
                batchsize * seq_len, headnum, selection_topk)

        compare_out_of_order(batchsize, seq_len, selection_topk, full_q_actual_seq, full_kv_actual_seq,
                            selection_topk_block_size, selection_topk_indices, s_max_block_num,
                            s_block_size, out_host, out, lay_out)

        kv_blk_t_equal = (out[2] == out_host[2].reshape(out[2].shape)).all()
        kv_blk_s_equal = (sort_with_negative_ones_last(
            out[3]) == sort_with_negative_ones_last(out_host[3])).all()
        kv_blk_seq_equ = (out[4] == out_host[4].reshape(out[4].shape)).all()
        assert(kv_blk_t_equal)
        assert(kv_blk_s_equal)
        assert(kv_blk_seq_equ)

        # 2. 比较npu和golden的输出
        if selection_topk <= 32:
            assert((out[0] == outnpu[0].reshape(out[0].shape)).all())
            assert((out[1] == outnpu[1].reshape(out[1].shape)).all())
            assert((out[2] == outnpu[2].reshape(out[2].shape)).all())
            assert((out[3] == outnpu[3].reshape(out[3].shape)).all())
            assert((out[4] == outnpu[4].reshape(out[4].shape)).all())
        else:
            compare_out_of_order(batchsize, seq_len, selection_topk, full_q_actual_seq, full_kv_actual_seq,
                                selection_topk_block_size, selection_topk_indices, s_max_block_num, s_block_size,
                                out_host, outnpu, lay_out)
            kv_blk_t_equal = (outnpu[2] == out_host[2].reshape(outnpu[2].shape)).all()
            if lay_out == 'TND':
                kv_blk_s_equal = (sort_with_negative_ones_last(
                    outnpu[3]) == sort_with_negative_ones_last(out_host[3]).reshape(
                        batchsize * seq_len, headnum, selection_topk + 1)).all()
            else:
                kv_blk_s_equal = (sort_with_negative_ones_last(
                    outnpu[3]) == sort_with_negative_ones_last(out_host[3])).all()
            kv_blk_seq_equ = (outnpu[4] == out_host[4].reshape(outnpu[4].shape)).all()
            assert(kv_blk_t_equal)
            assert(kv_blk_s_equal)
            assert(kv_blk_seq_equ)

if __name__ == "__main__":
    run_tests()
