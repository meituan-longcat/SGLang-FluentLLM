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
 * \file grouped_no_quant_matmul_tiling.h
 * \brief
 */
#ifndef GROUPED_NO_QUANT_MATMUL_TILING_H
#define GROUPED_NO_QUANT_MATMUL_TILING_H

#include "../grouped_matmul_tiling.h"
#include "../../../op_kernel/arch35/grouped_matmul_tiling_data_apt.h"
#include "log/log.h"
#include "log/error_code.h"
#include "register/op_impl_registry.h"

namespace optiling {
constexpr uint64_t TILING_KEY_DELIMITER = 9UL;
constexpr uint64_t BASIC_BLOCK_SIZE_16 = 16UL;
constexpr uint64_t STEP_K_DEFAULT = 4UL;
constexpr uint64_t DEPTH_DEFAULT = 8UL;
constexpr uint64_t DB_SIZE = 2UL;
constexpr uint32_t NZ_DIM_NUM = 4U;
constexpr uint32_t MIN_DIM = 2U;
constexpr uint64_t BASE_M_DEFAULT = 256UL;
constexpr uint64_t BASE_N_DEFAULT = 256UL;
constexpr uint64_t BASE_K_DEFAULT = 64UL;
constexpr uint64_t NUM_TWO = 2UL;
constexpr uint64_t BIAS_TABLE_NUM = 256UL;
constexpr uint64_t DATA_SIZE_FP32 = 4UL;
constexpr uint32_t INDEX_X = 0U;
constexpr uint32_t INDEX_WEIGHT = 1U;
constexpr uint32_t INDEX_BIAS = 2U;
constexpr uint32_t MAX_TENSOR = 1024U;
constexpr uint32_t INDEX_GROUPLIST = 7U;
constexpr uint32_t DIM_ONE = 1U;
constexpr uint32_t DIM_TWO = 2U;
constexpr uint32_t DIM_THREE = 3U;
constexpr uint32_t DIM_FOUR = 4U;
constexpr uint32_t FP32_DTYPE_SIZE = 4U;
constexpr uint64_t ATTR_IDX_SPLIT_ITEM = 0UL;
constexpr uint64_t ATTR_IDX_TRANS_W = 2UL;
constexpr uint64_t ATTR_IDX_TRANS_X = 3UL;
constexpr uint64_t ATTR_IDX_GROUPTYPE = 4UL;
constexpr uint64_t ATTR_IDX_GROUP_LIST_TYPE = 5UL;
constexpr int32_t NO_SPLIT = -1;
constexpr int32_t SPLIT_M = 0;
constexpr int32_t SPLIT_K = 2;
constexpr uint64_t BLOCK_SIZE = 32UL;
constexpr uint64_t ALIGN_DOWN_16 = 15UL;
constexpr uint64_t ALIGN_128 = 128UL;
constexpr uint64_t PARTA_L1_SIZE = 256UL * 1024UL;

enum class GMMTrans : std::uint8_t {
    NO_TRANS = 0,
    A_TRANS = 1,
    B_TRANS = 2,
    AB_TRANS = 3
};

enum class Model : std::uint8_t {
    BASIC = 0
};

class TilingKeyBuilder {
public:
    // 对应0-1位 平台大类，平台小类
    uint8_t model = 0;

    // 对应20位，转置场景
    uint8_t gmmTrans = 0;

public:
    uint64_t GenTilingKey();
};

class GroupedNoQuantMatmulTiling {
public:
    bool SetTiling(gert::TilingContext *context);

protected:
    bool Init(const gert::TilingContext* context);
    bool CalBaseMMTiling(const gert::TilingContext* context, const GMMCompileInfo* compileInfoPtr);
    void FormulateBasicBlock(const GMMCompileInfo* compileInfoPtr, uint32_t remainCoreNum);
    void CalAswtL1Tiling(const GMMCompileInfo* compileInfoPtr);
    bool CalL1Tiling(const gert::TilingContext* context, const GMMCompileInfo* compileInfoPtr);
    void CalcTailBasicBlock(const GMMCompileInfo* compileInfoPtr);
    bool SetCustomParam(gert::TilingContext *context);
    bool GetAttrs(const gert::TilingContext* context);
    bool CalMatMulTiling(const gert::TilingContext* context, const GMMCompileInfo* compileInfoPtr);
    void SetGMMTiling();
    void SetMatMulTiling();
    void SetTilingKey(gert::TilingContext *context);
    bool GMMGetTensorShapeSplitM(const gert::TilingContext* context, const gert::Shape xShape, const gert::Shape wShape);
    bool GMMGetTensorShapeSplitK(const gert::TilingContext* context, const gert::Shape xShape, const gert::Shape wShape);
    bool SplitMSingleXSingleWeightSingleY(const gert::Shape xShape, const gert::Shape wShape);
    bool SplitMSingleXSeparatedWeight(const gert::TilingContext* context, const gert::Shape xShape);
    bool SeparatedXSeparatedWeight(const gert::TilingContext* context);
    bool SeparatedXSingleWeight(const gert::TilingContext* context, const gert::Shape wShape);
    bool SplitKSingleXSingleWeightSingleY(const gert::TilingContext* context, const gert::Shape xShape, const gert::Shape wShape);
    void PrintTilingResult(const gert::TilingContext *context);
    void SetDisableL2Cache(const gert::TilingContext *context, const GMMCompileInfo *compileInfoPtr);

private:

    bool transposeX_ = false;
    bool transposeWeight_ = false;
    bool isSingleX_ = true;
    bool isSingleWeight_ = true;
    bool isSingleY_ = true;
    bool hasBias_ = false;
    bool weightNzFlag_ = false;
    bool weightNoL2Cache_ = false;
    bool kZero = false;

    uint64_t m_ = 0;
    uint64_t k_ = 0;
    uint64_t n_ = 0;
    uint64_t nSizeOri_ = 0;
    int32_t groupType_ = 0;
    int64_t splitItem_ = 0;
    uint32_t groupNum_ = 0;
    uint32_t groupListType_ = 0;
    uint32_t groupSize_ = 0;
    uint32_t xKDim_ = 0;
    uint32_t weightNDim_ = 0;
    uint32_t xDimNum_ = 0;
    int64_t nzFactor_ = 1;  // for weight nz format
    uint64_t baseM_ = BASE_M_DEFAULT;
    uint64_t baseN_ = BASE_N_DEFAULT;
    uint64_t baseK_ = BASE_K_DEFAULT;
    uint32_t usedCoreNum_ = 0;
    uint64_t stepKa_ = STEP_K_DEFAULT;
    uint64_t stepKb_ = STEP_K_DEFAULT;
    uint64_t depthA1_ = DEPTH_DEFAULT;
    uint64_t depthB1_ = DEPTH_DEFAULT;
    uint64_t mTailCnt_ = 1UL;
    uint64_t nTailCnt_ = 1UL;

    ge::DataType xDType_ = ge::DT_UNDEFINED;
    ge::DataType weightDtype_ = ge::DT_UNDEFINED;

    TilingKeyBuilder tilingKeyBuilder_;
    GroupedMatmulTilingData::GMMNoQuantTilingData tilingData_;
};
}  // namespace optiling

#endif  // GROUPED_WEIGHT_QUANT_BATCH_MATMUL_TILING_H