/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file grouped_matmul_weight_quant_a16w4_msd_basic_block_config.h
 * \brief
 */
#ifndef GROUPED_MATMUL_WEIGHT_QUANT_A16W4_MSD_BASIC_BLOCK_CONFIG_H
#define GROUPED_MATMUL_WEIGHT_QUANT_A16W4_MSD_BASIC_BLOCK_CONFIG_H

#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#include "lib/matmul_intf.h"

namespace GROUPED_MATMUL::A16W4Msd {
static constexpr uint64_t WORKSPACE_CACHE_WEIGHT_S8_SIZE = 512 * 1024;
static constexpr uint64_t WORKSPACE_CACHE_C_F32_SIZE = 32 * 320;
static constexpr uint64_t CV_LOOP_BUF_NUM = 2;
static constexpr uint64_t BASE_KUB_SIZE = 1024;
static constexpr uint64_t K_L0_BASE_SIZE = 32;

using CUBE_FIXP_DTYPE = float;

struct A16W4MsdConstParam {
    uint64_t kaL1Size;
    uint64_t kbL1Size;

    uint64_t kSize;
    uint64_t nSize;
    uint64_t nL1BaseSize;
};

struct A16W4MsdBasicBlockOffsetParam {
    uint64_t mSize;
    uint64_t mOffset;
    uint64_t mL1Size;
    uint64_t nOffset;
    uint64_t nL1Size;
    uint64_t kOffset;
    uint64_t kUbSize;
    GM_ADDR yGmAddr;
    GM_ADDR antiquantScaleGm;
    GM_ADDR aMaxGmAddr;
};
}  // namespace GROUPED_MATMUL::A16W4Msd
#endif  // GROUPED_MATMUL_WEIGHT_QUANT_BASIC_BLOCK_CONFIG_H