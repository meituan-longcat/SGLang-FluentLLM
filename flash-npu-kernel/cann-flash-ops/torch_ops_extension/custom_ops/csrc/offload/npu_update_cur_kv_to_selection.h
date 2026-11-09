#pragma once
/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <torch/library.h>
#include "../ops_common.h"

namespace flash_offload {
using namespace at_npu::native;

constexpr int SIZE = 8;

// step1, 为NPU设备实现前向接口
// 原地语义：selection_k_nope / selection_k_rope 由输入就地写回（Tensor(a!)/Tensor(b!)），
// 接口只返回 kernel_done[1] int32 图返回值标记（kernel 不修改其值）。
inline at::Tensor npu_update_cur_kv_to_selection_npu(
    const at::Tensor& cur_k_nope,
    const at::Tensor& cur_k_rope,
    const at::Tensor& cur_kv_write_slots,
    const at::Tensor& selection_k_nope,
    const at::Tensor& selection_k_rope,
    const at::Tensor& selection_kv_block_table)
{
    // 构造 kernel_done[1] int32 输出（设备与输入一致；内容未定义，不参与精度比对）
    // 注：kernel 不写 kernel_done（仅 GE 图返回值标记），故必须显式初始化为 0，
    //     保证 torch 侧返回值恒 0（与 op_host 契约/ST 断言一致）；at::empty 会返回未初始化垃圾值。
    at::Tensor kernel_done = at::zeros({1}, cur_k_nope.options().dtype(at::kInt));
    // aclnn 参数序：6 输入 → 1 输出 kernel_done（与 op_host def 的 Input/Output 顺序严格一致；
    // selection_k_nope / selection_k_rope 为 inplace 输出，aclnn 接口不暴露为输出参数）
    EXEC_NPU_CMD_V1(aclnnUpdateCurKvToSelection, cur_k_nope, cur_k_rope, cur_kv_write_slots,
                    selection_k_nope, selection_k_rope, selection_kv_block_table,
                    kernel_done);
    return kernel_done;
}

// step2, 为META设备实现前向接口
inline at::Tensor npu_update_cur_kv_to_selection_meta(
    const at::Tensor& cur_k_nope,
    const at::Tensor& cur_k_rope,
    const at::Tensor& cur_kv_write_slots,
    const at::Tensor& selection_k_nope,
    const at::Tensor& selection_k_rope,
    const at::Tensor& selection_kv_block_table)
{
    // meta 阶段不执行 kernel，kernel_done 恒 0（与 NPU 实现 at::zeros 一致）
    at::Tensor kernel_done = at::zeros({1}, cur_k_nope.options().dtype(at::kInt));
    return kernel_done;
}

// step4, 实现函数化前向接口（供 torch.compile functionalization 成图使用）
// functional 语义：clone selection 输入后执行相同写入，返回 kernel_done 与更新后的 selection 副本，
// 由 Functionalize impl 在 Python 侧 copy_ 回写原 selection。
inline std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_update_cur_kv_to_selection_functional_npu(
    const at::Tensor& cur_k_nope,
    const at::Tensor& cur_k_rope,
    const at::Tensor& cur_kv_write_slots,
    const at::Tensor& selection_k_nope,
    const at::Tensor& selection_k_rope,
    const at::Tensor& selection_kv_block_table)
{
    at::Tensor kernel_done = at::zeros({1}, cur_k_nope.options().dtype(at::kInt));
    // 保证接口对外为 functional：克隆 selection，kernel 就地在 clone 上执行写入
    at::Tensor selection_k_nope_out = selection_k_nope.clone();
    at::Tensor selection_k_rope_out = selection_k_rope.clone();
    EXEC_NPU_CMD_V1(aclnnUpdateCurKvToSelection, cur_k_nope, cur_k_rope, cur_kv_write_slots,
                    selection_k_nope_out, selection_k_rope_out, selection_kv_block_table,
                    kernel_done);
    return std::tie(kernel_done, selection_k_nope_out, selection_k_rope_out);
}

// step5, 为META设备实现函数化前向接口
inline std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_update_cur_kv_to_selection_functional_meta(
    const at::Tensor& cur_k_nope,
    const at::Tensor& cur_k_rope,
    const at::Tensor& cur_kv_write_slots,
    const at::Tensor& selection_k_nope,
    const at::Tensor& selection_k_rope,
    const at::Tensor& selection_kv_block_table)
{
    at::Tensor kernel_done = at::zeros({1}, cur_k_nope.options().dtype(at::kInt));
    at::Tensor selection_k_nope_out = at::empty_like(selection_k_nope);
    at::Tensor selection_k_rope_out = at::empty_like(selection_k_rope);
    return std::tie(kernel_done, selection_k_nope_out, selection_k_rope_out);
}


}
