/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file update_cur_kv_to_selection.cpp
 * \brief UpdateCurKvToSelection kernel 入口（单模板 TilingKey=1，T=bfloat16_t）
 *        参数顺序与 def Input/Output 注册顺序、aclnn 接口参数顺序严格一致；
 *        kernel_done 入参但不写（仅防 GE 判为无返回值原地算子）。
 */

#include "kernel_operator.h"
#include "update_cur_kv_to_selection_impl.h"

using namespace AscendC;
using namespace UpdateCurKvToSelectionNs;

extern "C" __global__ __aicore__ void update_cur_kv_to_selection(
    GM_ADDR cur_k_nope, GM_ADDR cur_k_rope, GM_ADDR cur_kv_write_slots,
    GM_ADDR selection_k_nope, GM_ADDR selection_k_rope, GM_ADDR selection_kv_block_table,
    GM_ADDR selection_k_nope_out, GM_ADDR selection_k_rope_out, GM_ADDR kernel_done,
    GM_ADDR workspace, GM_ADDR tiling)
{
    if (g_coreType == AIC) {
        return;
    }

    TPipe pipe;
    GET_TILING_DATA(tilingData, tiling);

    // 入参与同名输出（selection_k_nope/_out 等）在 inplace 场景指向同一 GM 缓冲，
    // 计算统一走输入地址；_out 形参仅用于对齐编译框架的输入+输出参数展开。
    (void)selection_k_nope_out;
    (void)selection_k_rope_out;

    if (TILING_KEY_IS(1)) {
        KernelUpdateCurKvToSelection<bfloat16_t> op(&pipe);
        op.Init(cur_k_nope, cur_k_rope, cur_kv_write_slots,
                selection_k_nope, selection_k_rope, selection_kv_block_table,
                kernel_done, &tilingData);
        op.Process();
    }
}
