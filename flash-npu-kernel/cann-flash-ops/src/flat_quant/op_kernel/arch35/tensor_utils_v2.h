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
 * \file tensor_utils.h
 * \brief
 */
#ifndef TENSOR_UTILS_V2_H
#define TENSOR_UTILS_V2_H

#pragma once
#include "kernel_operator.h"
#include "lib/matmul_intf.h"

using namespace AscendC;

namespace FlatQuantNS {
constexpr int32_t M_SPLIT_COUNT = 2;
constexpr int32_t GROUP_SIZE = 32;
constexpr int32_t DOUBLE_GROUP_SIZE = 64;
constexpr int32_t VEC_N_LEN = 64;
constexpr int32_t DOUBLE_VEC_N_LEN = 128;
constexpr int32_t ALIGNLENGTH = 64;
constexpr int32_t PAD_NUM = 8;
constexpr int32_t MN_SIZE = 32 * 1024;
constexpr int32_t OUT_SIZE = 16 * 1024;
constexpr int32_t EMAX_SIZE = 1024; // DATA_COUNT /2
constexpr int32_t SCALE_SIZE = 512;
constexpr int32_t MXFP_DIVISOR_SIZE = 64;
constexpr int32_t MXFP_MULTI_BASE_SIZE = 2;
constexpr uint16_t MAX_EXP_FOR_BF16 = 0x7f80;
constexpr uint16_t MAX_EXP_FOR_FP8 = 0x00ff;
constexpr uint16_t BF16_EXP_BIAS = 0x7f00;
constexpr int16_t SHR_NUM_FOR_BF16 = 7;
constexpr uint16_t NAN_CUSTOMIZATION = 0x7f81;
constexpr uint16_t SPECIAL_EXP_THRESHOLD = 0x0040;
constexpr uint16_t FP4_E2M1_MAX_EXP = 0x0100;

// Constants for cuBLAS FP4 scaling
constexpr float ZERO_FLOAT = 0.0f;
constexpr float SIX_FLOAT = 6.0f;
constexpr float SEVEN_FLOAT = 7.0f;
constexpr float TWELVE_FLOAT = 12.0f;
constexpr uint32_t STORE_UNALIGN_STRIDE_BYTES = 8;
constexpr uint32_t SCALE_STORE_STRIDE = 32;
constexpr int16_t SHR_NUM_FOR_FP32 = 23;
constexpr uint32_t NUMBER_ZERO = 0x00000000;
constexpr uint32_t NUMBER_TWO_FIVE_FOUR = 0x000000fe;
constexpr uint16_t ADD_VALUE_FOR_BF16_MAN1 = 0x003f;
constexpr uint16_t ADD_VALUE_FOR_BF16_MAN2 = 0x001f;
constexpr uint32_t vfLen16 = AscendC::GetVecLen() / sizeof(uint16_t);
constexpr uint32_t vfLen32 = AscendC::GetVecLen() / sizeof(uint32_t);
constexpr uint16_t ABS_MASK_FOR_16BIT = 0x7fff;
constexpr uint32_t MAN_MASK_FLOAT = 0x007fffff;
constexpr uint32_t MAX_EXP_FOR_FP32 = 0x7f800000;
constexpr uint32_t FP32_EXP_BIAS_CUBLAS = 0x00007f00;
constexpr uint32_t NAN_CUSTOMIZATION_PACK = 0x00007f81;
constexpr uint32_t MAX_EXP_FOR_FP8_IN_FP32 = 0x000000ff;
} // namespace FlatQuantNS

#endif // TENSOR_UTILS_V2_H