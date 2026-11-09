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
 * \file gather_elements_v2_tiling.h
 * \brief
 */

#ifndef GATHER_ELEMENTS_V2_TILING_H
#define GATHER_ELEMENTS_V2_TILING_H
#include "register/tilingdata_base.h"

namespace optiling {

constexpr int64_t TILING_ARRAY_LEN_EIGHT = 8;

BEGIN_TILING_DATA_DEF(GatherElementsV2TilingParam)
TILING_DATA_FIELD_DEF(uint64_t, xPreDim)
TILING_DATA_FIELD_DEF(uint64_t, xGatherDim)
TILING_DATA_FIELD_DEF(uint64_t, xPostDim)
TILING_DATA_FIELD_DEF(uint64_t, idxPreDim)
TILING_DATA_FIELD_DEF(uint64_t, idxGatherDim)
TILING_DATA_FIELD_DEF(uint64_t, idxPostDim)

TILING_DATA_FIELD_DEF(uint64_t, coreGroupNum)
TILING_DATA_FIELD_DEF(uint64_t, formerGroupNum) // 大组数
TILING_DATA_FIELD_DEF(uint64_t, tailGroupNum)   // 小组

TILING_DATA_FIELD_DEF(uint64_t, formerGroupPreDim)
TILING_DATA_FIELD_DEF(uint64_t, tailGroupPreDim)
TILING_DATA_FIELD_DEF(uint64_t, formerGroupCoreNum) // 大组中大核数
TILING_DATA_FIELD_DEF(uint64_t, tailGroupCoreNum)   // 大组中小核数

TILING_DATA_FIELD_DEF(uint64_t, formerGroupFormerNum)
TILING_DATA_FIELD_DEF(uint64_t, formerGroupTailNum)
TILING_DATA_FIELD_DEF(uint64_t, formerGroupFormerPostDim)
TILING_DATA_FIELD_DEF(uint64_t, formerGroupTailPostDim)

TILING_DATA_FIELD_DEF(uint64_t, tailGroupFormerNum)
TILING_DATA_FIELD_DEF(uint64_t, tailGroupTailNum)
TILING_DATA_FIELD_DEF(uint64_t, tailGroupFormerPostDim)
TILING_DATA_FIELD_DEF(uint64_t, tailGroupTailPostDim)
END_TILING_DATA_DEF

REGISTER_TILING_DATA_CLASS(GatherElementsV2TilingParamOp, GatherElementsV2TilingParam)

// transpose
BEGIN_TILING_DATA_DEF(GatherElementsV2TransTiling)
TILING_DATA_FIELD_DEF(uint64_t, carryNumAlign)
TILING_DATA_FIELD_DEF(uint64_t, xCarryNumAlign)
TILING_DATA_FIELD_DEF(uint64_t, idxCarryNumAlign)

TILING_DATA_FIELD_DEF(uint64_t, inBufferSize)
TILING_DATA_FIELD_DEF(uint64_t, outBufferSize)
TILING_DATA_FIELD_DEF(uint64_t, transGatherDimSlice)
TILING_DATA_FIELD_DEF(uint64_t, idxGatherDimSlice)

TILING_DATA_FIELD_DEF(uint64_t, workspacePerBlock)
END_TILING_DATA_DEF

REGISTER_TILING_DATA_CLASS(GatherElementsV2TransTilingOp, GatherElementsV2TransTiling)

// scalar
BEGIN_TILING_DATA_DEF(GatherElementsV2ScalarTiling)
TILING_DATA_FIELD_DEF(uint64_t, formerGroupFormerData)
TILING_DATA_FIELD_DEF(uint64_t, formerGroupTailData)
TILING_DATA_FIELD_DEF(uint64_t, tailGroupFormerData)
TILING_DATA_FIELD_DEF(uint64_t, tailGroupTailData)
TILING_DATA_FIELD_DEF(uint64_t, maxIdxDataAlign)
END_TILING_DATA_DEF

BEGIN_TILING_DATA_DEF(GatherElementsV2LastDimTilingParam)
TILING_DATA_FIELD_DEF_ARR(int64_t, TILING_ARRAY_LEN_EIGHT, xShape)
TILING_DATA_FIELD_DEF_ARR(int64_t, TILING_ARRAY_LEN_EIGHT, indexShape)
TILING_DATA_FIELD_DEF_ARR(int64_t, TILING_ARRAY_LEN_EIGHT, xStrideArray)
TILING_DATA_FIELD_DEF_ARR(int64_t, TILING_ARRAY_LEN_EIGHT, indexStrideArray)

TILING_DATA_FIELD_DEF(int64_t, dimNum)
TILING_DATA_FIELD_DEF(int64_t, specialDataMove)
TILING_DATA_FIELD_DEF(int64_t, xSliceNum)
TILING_DATA_FIELD_DEF(int64_t, indexSliceNum)
TILING_DATA_FIELD_DEF(int64_t, reservedXSize)
TILING_DATA_FIELD_DEF(int64_t, reservedIndexSize)

TILING_DATA_FIELD_DEF(int64_t, indexAxisSizeEqualOne)
TILING_DATA_FIELD_DEF(int64_t, scalarMode)
TILING_DATA_FIELD_DEF(int64_t, formerCoreRowNum)
TILING_DATA_FIELD_DEF(int64_t, formerCoreNum)
TILING_DATA_FIELD_DEF(int64_t, eachCalculationLines)

TILING_DATA_FIELD_DEF(int64_t, xBufferSize)
TILING_DATA_FIELD_DEF(int64_t, indexBufferSize)
TILING_DATA_FIELD_DEF(int64_t, yBufferSize)
TILING_DATA_FIELD_DEF(int64_t, maskBufferSize)
TILING_DATA_FIELD_DEF(int64_t, scalarModeLength)

TILING_DATA_FIELD_DEF(int64_t, dataMoveUBStride)
END_TILING_DATA_DEF

REGISTER_TILING_DATA_CLASS(GatherElementsV2LastDimTilingParamOp, GatherElementsV2LastDimTilingParam)

REGISTER_TILING_DATA_CLASS(GatherElementsV2ScalarTilingOp, GatherElementsV2ScalarTiling)

BEGIN_TILING_DATA_DEF(GatherElementsV2TilingData)
TILING_DATA_FIELD_DEF_STRUCT(GatherElementsV2TilingParam, params)
TILING_DATA_FIELD_DEF_STRUCT(GatherElementsV2TransTiling, transTiling)
TILING_DATA_FIELD_DEF_STRUCT(GatherElementsV2ScalarTiling, scalarTiling)
TILING_DATA_FIELD_DEF_STRUCT(GatherElementsV2LastDimTilingParam, lastDimTiling)
END_TILING_DATA_DEF

REGISTER_TILING_DATA_CLASS(GatherElementsV2, GatherElementsV2TilingData)
REGISTER_TILING_DATA_CLASS(GatherElementsV2TilingDataOp, GatherElementsV2TilingData)

struct GatherElementsV2CompileInfo {
    uint64_t totalCoreNum = 0;
    uint64_t ubSizePlatForm = 0;
    uint64_t sysWorkspaceSize = 0;
};
} // namespace optiling
#endif // GATHER_ELEMENTS_V2_TILING_H