/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file dispatch_policy.h
 * \brief
 */
#ifndef CMCT_INCLUDE_PROLOGUE_DISPATCH_POLICY_H
#define CMCT_INCLUDE_PROLOGUE_DISPATCH_POLICY_H

#include "../utils/constant.h"

namespace Cmct::Prologue {
using Cmct::Gemm::KB_ELEM;

struct BCastScsc{};

template <
    typename ArchTag_, typename HighBitType, typename ScaleType, bool InnerK, bool WeightNz, uint64_t AivNum,
    bool HasOffset, uint64_t UbInBufNum, uint64_t InnerSize, uint64_t VfN, uint64_t VfK, uint64_t UbOutBufNum,
    uint64_t UbInSize, uint64_t UbOutSize, uint64_t ScaleSize, uint64_t OffsetSize,
    uint64_t antiQuantScaleAfterCastSize = 0>
struct BAntiquantScmc {
    using ArchTag = ArchTag_;
    // innerK
    //   true:  nd nk, nz kn
    //   false: nd kn
    static constexpr bool INNER_K = InnerK;
    static constexpr bool WEIGHT_NZ = WeightNz;
    static constexpr bool HAS_OFFSET = HasOffset;
    static constexpr uint64_t AIV_NUM = AivNum;
    // VecAntiQuantConfig
    static constexpr uint64_t UB_MTE2_BUF_NUM = UbInBufNum;
    static constexpr uint64_t UB_MTE2_INNER_SIZE = InnerSize;
    // VfConfig
    static constexpr uint64_t VF_N_STANDARD_LEN = VfN;
    static constexpr uint64_t VF_K_STANDARD_LEN = VfK;
    // UbBufferInfo
    static constexpr uint64_t UB_WEIGHT_OUTPUT_HIGH_BIT_BUFFER_NUM = UbOutBufNum;
    // unit: element
    static constexpr uint64_t WEIGHT_INPUT_LOW_BIT_UB_TOTAL_SIZE = UbInSize * 1024; // always int8
    static constexpr uint64_t HIGH_BIT_DATA_UB_TOTAL_SIZE = UbOutSize * KB_ELEM<HighBitType>;
    static constexpr uint64_t ANTIQUANT_SCALE_UB_TOTAL_SIZE = ScaleSize * KB_ELEM<ScaleType>;
    static constexpr uint64_t ANTIQUANT_OFFSET_UB_TOTAL_SIZE = OffsetSize * KB_ELEM<ScaleType>;
    static constexpr uint64_t ANTIQUANT_SCALE_AFTER_CAST_UB_TOTAL_SIZE =
        antiQuantScaleAfterCastSize * KB_ELEM<HighBitType>;
    static constexpr uint64_t WEIGHT_INPUT_LOW_BIT_UB_SINGLE_BUFFER_SIZE =
        WEIGHT_INPUT_LOW_BIT_UB_TOTAL_SIZE / UB_MTE2_BUF_NUM;
    static constexpr uint64_t ANTIQUANT_SCALE_UB_SINGLE_BUFFER_SIZE = ANTIQUANT_SCALE_UB_TOTAL_SIZE / UB_MTE2_BUF_NUM;
    static constexpr uint64_t ANTIQUANT_SCALE_AFTER_CAST_UB_SINGLE_BUFFER_SIZE =
        ANTIQUANT_SCALE_AFTER_CAST_UB_TOTAL_SIZE / UB_MTE2_BUF_NUM;
    static constexpr uint64_t ANTIQUANT_OFFSET_UB_SINGLE_BUFFER_SIZE = ANTIQUANT_OFFSET_UB_TOTAL_SIZE / UB_MTE2_BUF_NUM;
    static constexpr uint64_t HIGH_BIT_DATA_UB_SINGLE_BUFFER_SIZE =
        HIGH_BIT_DATA_UB_TOTAL_SIZE / UB_WEIGHT_OUTPUT_HIGH_BIT_BUFFER_NUM;
};
}  // namespace Cmct::Prologue

#endif