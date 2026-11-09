/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
* \file mla_prolog_tiling.cpp
* \file mla_prolog_tiling.cpp
* \brief
*/

#include "../op_kernel/mla_prolog_v3_policy.h"

#include <numeric>
#include <functional>
#include <algorithm>
#include <cmath>
#include <limits>
#include <graph/utils/type_utils.h>
#include "error/ops_error.h"
#include "register/op_def_registry.h"
#include "mla_prolog_v3_tiling_check.h"

using namespace ge;
using namespace AscendC;
namespace optiling {
#if MLA_PROLOG_V3_SFA1
namespace fluent_mla_prolog_v3 {
#endif

const std::unordered_map<ge::DataType, uint32_t> DTYPE_TO_SIZE {
    {ge::DT_BF16, 2},
    {ge::DT_FLOAT16, 2},
    {ge::DT_INT8, 1},
    {ge::DT_INT32, 4}};

const std::unordered_map<ge::DataType, matmul_tiling::DataType> GE_TO_MM_DTYPE {
    {ge::DT_FLOAT16, matmul_tiling::DataType::DT_FLOAT16},
    {ge::DT_BF16, matmul_tiling::DataType::DT_BF16},
    {ge::DT_INT8, matmul_tiling::DataType::DT_INT8},
    {ge::DT_INT4, matmul_tiling::DataType::DT_INT4},
    {ge::DT_FLOAT, matmul_tiling::DataType::DT_FLOAT}};

template <typename T>
inline auto CeilDiv(T a, T b) -> T
{
    if (b == 0) {
        return b;
    }
    return (a + b - 1) / b;
}

template <typename T>
inline auto Align(T num, T rnd) -> T
{
    return (((rnd) == 0) ? 0 : (((num) + (rnd) - 1) / (rnd) * (rnd)));
}

ge::graphStatus MlaPrologV3Tiling::GetNpuInfo()
{
    OPS_ERR_IF(context_->platformInfo == nullptr,
        OPS_REPORT_VECTOR_INNER_ERR(context_->opName, "GetPlatformInfo is nullptr."), return ge::GRAPH_FAILED);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->platformInfo);
    libapiSize_ = ascendcPlatform.GetLibApiWorkSpaceSize();

    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize_);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, l1Size_);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, l0cSize_);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, l0bSize_);

    aivNum_ = ascendcPlatform.GetCoreNumAiv();
    aicNum_ = ascendcPlatform.GetCoreNumAic();

    OPS_ERR_IF(aicNum_ == 0 || aivNum_ == 0,
        OPS_REPORT_VECTOR_INNER_ERR(context_->opName, "num of core obtained is 0."), return GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

QUANT_MODE MlaPrologV3Tiling::GetQuantizationMode() const
{
#if !MLA_PROLOG_V3_SFA1
    if (*(context_->weightQuantMode) == static_cast<int>(WEIGHT_QUANT_MODE::NO_QUANT) &&
        *(context_->kvQuantMode) == static_cast<int>(KV_QUANT_MODE::PER_TILE)) {
        return QUANT_MODE::NO_QUANT_KV_QUANT_PER_TILE;
    }
#endif
    if (*(context_->weightQuantMode) == static_cast<int>(WEIGHT_QUANT_MODE::PARTIAL_QUANT)) {
        if (*(context_->kvQuantMode) == static_cast<int>(KV_QUANT_MODE::NO_QUANT)) {
            return QUANT_MODE::PARTIAL_QUANT_KV_NO_QUANT;
        } else if (*(context_->kvQuantMode) == static_cast<int>(KV_QUANT_MODE::PER_CHANNEL)) {
            return QUANT_MODE::PARTIAL_QUANT_KV_QUANT_PER_CHANNEL;
        } else if (*(context_->kvQuantMode) == static_cast<int>(KV_QUANT_MODE::PER_TILE)) {
            return QUANT_MODE::PARTIAL_QUANT_KV_QUANT_PER_TILE;
        }
    }
    if (*(context_->weightQuantMode) == static_cast<int>(WEIGHT_QUANT_MODE::FULL_QUANT)) {
        if (*(context_->kvQuantMode) == static_cast<int>(KV_QUANT_MODE::NO_QUANT)) {
            return QUANT_MODE::FULL_QUANT_KV_NO_QUANT;
        } else if (*(context_->kvQuantMode) == static_cast<int>(KV_QUANT_MODE::PER_TENSOR)) {
            return QUANT_MODE::FULL_QUANT_KV_QUANT_PER_TENSOR;
        } else if (*(context_->kvQuantMode) == static_cast<int>(KV_QUANT_MODE::PER_TILE)) {
            return QUANT_MODE::FULL_QUANT_KV_QUANT_PER_TILE;
        }
    }
    return QUANT_MODE::NO_QUANT;
}

ge::graphStatus MlaPrologV3Tiling::SetShapeInfo()
{
    if (context_->tokenX.shape->GetStorageShape().GetDimNum() == MLA_PROLOG_V3_DIM_NUM_3) {
        baseShapeInfo_.bSize = context_->tokenX.shape->GetStorageShape().GetDim(MLA_PROLOG_V3_DIM_INDEX_0);
        baseShapeInfo_.s1Size = context_->tokenX.shape->GetStorageShape().GetDim(MLA_PROLOG_V3_DIM_INDEX_1);
        baseShapeInfo_.heSize = context_->tokenX.shape->GetStorageShape().GetDim(MLA_PROLOG_V3_DIM_INDEX_2);
        baseShapeInfo_.tSize = baseShapeInfo_.bSize * baseShapeInfo_.s1Size;
    } else {
        baseShapeInfo_.tSize = context_->tokenX.shape->GetStorageShape().GetDim(MLA_PROLOG_V3_DIM_INDEX_0);
        baseShapeInfo_.heSize = context_->tokenX.shape->GetStorageShape().GetDim(MLA_PROLOG_V3_DIM_INDEX_1);
    }
    if (context_->weightDq.shape->GetStorageShape().GetDimNum() == MLA_PROLOG_V3_DIM_NUM_2) {
        baseShapeInfo_.hcqSize = context_->weightDq.shape->GetStorageShape().GetDim(MLA_PROLOG_V3_DIM_INDEX_1);
    } else {
        uint32_t weightDqAxisSize_ = 32U / ge::GetSizeByDataType(context_->weightDq.desc->GetDataType());
        // weightDq: [He, Hcq] -> [Hcq/16, He/16, 16, 16] || [Hcq/32, He/16, 16, 32]
        baseShapeInfo_.hcqSize =
            weightDqAxisSize_ * context_->weightDq.shape->GetStorageShape().GetDim(MLA_PROLOG_V3_DIM_INDEX_0);
    }
    baseShapeInfo_.nSize = context_->weightUk.shape->GetStorageShape().GetDim(MLA_PROLOG_V3_DIM_INDEX_0);
    baseShapeInfo_.drSize =
        context_->ropeCos.shape->GetStorageShape().GetDim(context_->ropeCos.shape->GetStorageShape().GetDimNum() - 1);
    baseShapeInfo_.dSize = context_->weightUk.shape->GetStorageShape().GetDim(MLA_PROLOG_V3_DIM_INDEX_1);
    baseShapeInfo_.headSizeQc = baseShapeInfo_.dSize * baseShapeInfo_.nSize;
    baseShapeInfo_.headSizeQr = baseShapeInfo_.drSize * baseShapeInfo_.nSize;
    baseShapeInfo_.headSizeUqQr = baseShapeInfo_.headSizeQc + baseShapeInfo_.headSizeQr;
    if (context_->kvCache.shape->GetStorageShape().GetDimNum() == MLA_PROLOG_V3_DIM_NUM_3) {
        baseShapeInfo_.blockNum = context_->kvCache.shape->GetStorageShape().GetDim(MLA_PROLOG_V3_DIM_INDEX_0); // kvT
        baseShapeInfo_.nkvSize = context_->kvCache.shape->GetStorageShape().GetDim(MLA_PROLOG_V3_DIM_INDEX_1);
        baseShapeInfo_.dtileSize = context_->kvCache.shape->GetStorageShape().GetDim(MLA_PROLOG_V3_DIM_INDEX_2);
    } else {
        baseShapeInfo_.blockNum = context_->kvCache.shape->GetStorageShape().GetDim(MLA_PROLOG_V3_DIM_INDEX_0);
        baseShapeInfo_.blockSize = context_->kvCache.shape->GetStorageShape().GetDim(MLA_PROLOG_V3_DIM_INDEX_1);
        baseShapeInfo_.nkvSize = context_->kvCache.shape->GetStorageShape().GetDim(MLA_PROLOG_V3_DIM_INDEX_2);
        baseShapeInfo_.dtileSize = context_->kvCache.shape->GetStorageShape().GetDim(MLA_PROLOG_V3_DIM_INDEX_3);
    }
    if (context_->weightDkvKr.shape->GetStorageShape().GetDimNum() == MLA_PROLOG_V3_DIM_NUM_4) {
        baseShapeInfo_.hckvSize =
            context_->weightDkvKr.shape->GetStorageShape().GetDim(MLA_PROLOG_V3_DIM_INDEX_0) *
                context_->weightDkvKr.shape->GetStorageShape().GetDim(MLA_PROLOG_V3_DIM_INDEX_3) -
                    baseShapeInfo_.drSize;
    } else {
        baseShapeInfo_.hckvSize =
            context_->weightDkvKr.shape->GetStorageShape().GetDim(MLA_PROLOG_V3_DIM_INDEX_1) - baseShapeInfo_.drSize;
    }
    baseShapeInfo_.s2Size = baseShapeInfo_.nkvSize;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MlaPrologV3Tiling::SetScenarioInfo()
{
    scenarioInfo_.isV1Flag_ = (strcmp(context_->opType, V1_OP_NAME) == 0);
    scenarioInfo_.batchSeqFusedFlag_ = context_->tokenX.shape->GetStorageShape().GetDimNum() == MLA_PROLOG_V3_DIM_NUM_2;
    scenarioInfo_.quantMode_ = GetQuantizationMode();
    if (std::strncmp(context_->cacheMode, CACHE_MODE_BSND, CACHE_MODE_BSND_LEN) == 0) {
        scenarioInfo_.cacheMode_ = CACHE_MODE::BSND;
    } else if (std::strncmp(context_->cacheMode, CACHE_MODE_TND, CACHE_MODE_TND_LEN) == 0) {
        scenarioInfo_.cacheMode_ = CACHE_MODE::TND;
    } else if (std::strncmp(context_->cacheMode, CACHE_MODE_PA_BSND, CACHE_MODE_PA_BSND_LEN) == 0) {
        scenarioInfo_.cacheMode_ = CACHE_MODE::PA_BSND;
    } else {
        scenarioInfo_.cacheMode_ = CACHE_MODE::PA_NZ;
    }

    scenarioInfo_.actualSeqMode_ = ACTUAL_SEQ_MODE::DISABLED;

    if ((scenarioInfo_.batchSeqFusedFlag_ && baseShapeInfo_.tSize == 0U) ||
        (!scenarioInfo_.batchSeqFusedFlag_ && (baseShapeInfo_.bSize * baseShapeInfo_.s1Size == 0U))) {
        scenarioInfo_.emptyTensorMode_ = EMPTY_TENSOR_MODE::EMPTY_QUERY;
    } else if (baseShapeInfo_.blockNum == 0U) {
        scenarioInfo_.emptyTensorMode_ = EMPTY_TENSOR_MODE::EMPTY_CACHE;
    } else {
        scenarioInfo_.emptyTensorMode_ = EMPTY_TENSOR_MODE::NON_EMPTY;
    }

    // 当前不支持切M模板，全部路由到切N模板
    scenarioInfo_.splitMFlag_ = false;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MlaPrologV3Tiling::SetAttrInfo()
{
    reciprocalCq_ = 1.0f / (baseShapeInfo_.hcqSize);
    epsilonCq_ = *context_->rmsNormEspilonCq;
    reciprocalCkv_ = 1.0f / (baseShapeInfo_.hckvSize);
    epsilonCkv_ = *context_->rmsNormEspilonCkv;
    queryNormFlag_ = *context_->queryNormFlag;

#if MLA_PROLOG_V3_SFA1
    outputCurKNopeRope_ = (context_->outputCurKNopeRopeFlag == nullptr) ? true : *context_->outputCurKNopeRopeFlag;

#endif
    weightQuantMode_ = *context_->weightQuantMode;
    kvQuantMode_ = static_cast<KV_QUANT_MODE>(*context_->kvQuantMode);
    queryQuantMode_ = *context_->queryQuantMode;
    ckvkrRepoMode_ = *context_->ckvkrRepoMode;
    quantScaleRepoMode_ = *context_->quantScaleRepoMode;
    tileSize_ = *context_->tileSize;
    qcQrScale_ = *context_->qcQrScale;
    kcScale_ = *context_->kcScale;
#if !MLA_PROLOG_V3_SFA1
    enableRope_ = *context_->enableRope;
#endif

    return ge::GRAPH_SUCCESS;
}

bool MlaPrologV3Tiling::GetMatmulType(ge::DataType getype, matmul_tiling::DataType *mmType)
{
    auto mmdt = GE_TO_MM_DTYPE.find(getype);
    if (mmdt != GE_TO_MM_DTYPE.end()) {
        *mmType = mmdt->second;
        return true;
    }
    return false;
}

uint32_t MlaPrologV3Tiling::CalcSingleCoreN(uint32_t n, uint32_t coreNum, uint32_t alignNum) const
{
    return CeilDiv(n, alignNum * coreNum) * alignNum;
}

// mm1.m = stepBatchSize            // 32
// mm1.n = singlecoreHeadSizeCq     // 64
// mm1.k = headSizeX                // 7168
// mm1.baseM = stepBatchSize        // 32
// mm1.baseN = singlecoreHeadSizeCq // 64
// mm1.baseK = 256
ge::graphStatus MlaPrologV3Tiling::FillMatmul1Tiling()
{
    auto dataType = context_->weightDq.desc->GetDataType();
    singlecoreHeadSizeCq_ =
        CalcSingleCoreN(baseShapeInfo_.hcqSize, aicNum_, BLOCK_SIZE / DTYPE_TO_SIZE.at(dataType));
    mm1BlockNum_ = CeilDiv(baseShapeInfo_.hcqSize, singlecoreHeadSizeCq_);
    return ge::GRAPH_SUCCESS;
}

// singlecoreHeadSizeCkvKr =  HeadSizeCkvDr / mm2CoreNum // 576 / 9 == 64
// mm2.m = stepBatchSize
// mm2.n = singlecoreHeadSizeCkvKr
// mm2.k = headSizeX // size of He
// mm2.baseN = n
// mm2.baseK = 256
ge::graphStatus MlaPrologV3Tiling::FillMatmul2Tiling()
{
    if (scenarioInfo_.emptyTensorMode_ == EMPTY_TENSOR_MODE::EMPTY_CACHE) {
        return ge::GRAPH_SUCCESS;
    }
    if (scenarioInfo_.splitMFlag_) {
        singlecoreHeadSizeCkvKr_ = baseShapeInfo_.hckvSize + baseShapeInfo_.drSize;
        mm2BlockNum_ = aicNum_;
    } else if (aicNum_ >= 9U) { // 9是经验值
        uint32_t baseN = 64U;
        mm2BlockNum_ = (baseShapeInfo_.hckvSize + baseShapeInfo_.drSize) / baseN;
        singlecoreHeadSizeCkvKr_ = baseN;
    } else {
        auto dataType = context_->weightDkvKr.desc->GetDataType();
        singlecoreHeadSizeCkvKr_ = CalcSingleCoreN(baseShapeInfo_.hckvSize + baseShapeInfo_.drSize, aicNum_,
                                                   BLOCK_SIZE / DTYPE_TO_SIZE.at(dataType));
        mm2BlockNum_ = CeilDiv(baseShapeInfo_.hckvSize + baseShapeInfo_.drSize, singlecoreHeadSizeCkvKr_);
    }
    return ge::GRAPH_SUCCESS;
}

// singlecoreHeadSizeQcQr = headNum * (dimHeadSizeQc + dimHeadRope) / mm3CoreNum  = 32 * (128 + 64) / 24
// mm3.m = stepBatchSize
// mm3.n = singlecoreHeadSizeQcQr   // 256
// mm3.k = headSizeCq // size of Hcq   1536
// mm3.baseN = 64  //
// mm3.baseK = 256 //
ge::graphStatus MlaPrologV3Tiling::FillMatmul3Tiling()
{
    auto dataType = context_->weightUqQr.desc->GetDataType();
    auto oriM = baseShapeInfo_.nSize * (baseShapeInfo_.dSize + baseShapeInfo_.drSize);
    if (enableGroupComputeOpt_) {
        // 算力分组场景下G=8，dimHeadSizeQc跨8核切，dimHeadSizeQr跨4核切；matmulQc和matmulQr的singleN都取128
        singlecoreHeadSizeQcQr_ =
            CalcSingleCoreN(baseShapeInfo_.nSize * baseShapeInfo_.dSize,
                GROUP_COMPUTE_CUBE_NUM_PER_GROUP, baseShapeInfo_.dSize);
    } else if (enableDequantOpt_) {
        // dequant流水掩盖场景，dimHeadSizeQc + dimHeadRope不跨核
        singlecoreHeadSizeQcQr_ = CalcSingleCoreN(oriM, aicNum_, baseShapeInfo_.dSize + baseShapeInfo_.drSize);
    } else {
        // headnum * (dimHeadSizeQc + dimHeadRope) 合轴切
        singlecoreHeadSizeQcQr_ = CalcSingleCoreN(oriM, aicNum_, BLOCK_SIZE / DTYPE_TO_SIZE.at(dataType));
    }
    mm3BlockNum_ = CeilDiv(oriM, singlecoreHeadSizeQcQr_);

    if (scenarioInfo_.splitMFlag_) {
        singlecoreHeadSizeQcQr_ = oriM;
        mm3BlockNum_ = aicNum_;
    }
    return ge::GRAPH_SUCCESS;
}

// mm4.m = stepBatchSize
// mm4.n = headSizeCkv  // 512
// mm4.k = dimHeadSizeQc // size of Qc  128
// mm4.baseN = 128 //
// mm4.baseK = 128 //
// mm4.Kstride = dimHeadSizeQc + dimHeadRope
ge::graphStatus MlaPrologV3Tiling::FillMatmul4Tiling()
{
    if (scenarioInfo_.splitMFlag_) {
        singlecoreNumHeadSize_ = baseShapeInfo_.nSize;
        mm4BlockNum_ = aicNum_;
    } else {
        singlecoreNumHeadSize_ = CeilDiv(baseShapeInfo_.nSize, aicNum_);
        mm4BlockNum_ = CeilDiv(baseShapeInfo_.nSize, singlecoreNumHeadSize_);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MlaPrologV3Tiling::ProcessBaseInputs()
{
    stepBatchSize_ = std::min(128U, baseShapeInfo_.tSize);
    if (scenarioInfo_.splitMFlag_ && (stepBatchSize_ > 0) && (aicNum_ > 0) &&
        (baseShapeInfo_.tSize > 0)) {
        uint32_t mBaseSize = stepBatchSize_;
        mSubSize_ = (baseShapeInfo_.tSize + aicNum_ - 1) / aicNum_;
        // idx为[0, mSubCoreNum_]的核分到mSubSize_, 其余核分到mSubSize_ - 1
        mSubCoreNum_ = baseShapeInfo_.tSize - (mSubSize_ - 1) * aicNum_;
    }
    if (baseShapeInfo_.dSize == HIGH_THROUGHPUT__D_SIZE) {
        stepNumHeadDequant_ = std::min(64U, baseShapeInfo_.nSize);
    } else {
        stepNumHeadDequant_ = std::min(16U, baseShapeInfo_.nSize);
    }
    vectorBlockNum_ = std::min(stepBatchSize_, aivNum_);

    // 算力分组开关，仅当半量化场景，BS=1，G=8，可用核数大于等于16时进入分支
    if ((scenarioInfo_.quantMode_ == QUANT_MODE::PARTIAL_QUANT_KV_NO_QUANT ||
         scenarioInfo_.quantMode_ == QUANT_MODE::PARTIAL_QUANT_KV_QUANT_PER_CHANNEL) &&
        baseShapeInfo_.tSize == GROUP_COMPUTE_T_SIZE &&
        baseShapeInfo_.nkvSize == GROUP_COMPUTE_NKV_SIZE &&
        aivNum_ >= GROUP_COMPUTE_MIN_AIV_NUM &&
        aicNum_ >= GROUP_COMPUTE_MIN_AIC_NUM) {
        enableGroupComputeOpt_ = true;
        aivNum_ = 32U;
        aicNum_ = 16U;
    } else if (context_->weightUqQr.desc->GetDataType() == ge::DT_INT8 &&
               baseShapeInfo_.nSize >= GROUP_COMPUTE_N_SIZE) {
        // N大于等于8时通过切N处理MM3，MM4之后的操作例如Rope，DynamicQuant等会有性能收益
        enableDequantOpt_ = true;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MlaPrologV3Tiling::FillTiling()
{
    baseParams_->batchSize = baseShapeInfo_.bSize;
    baseParams_->stepBatchSize = stepBatchSize_;
    baseParams_->stepNumHeadDequant = stepNumHeadDequant_;
    baseParams_->mSubSize = mSubSize_;
    baseParams_->mSubCoreNum = mSubCoreNum_;
    baseParams_->tokenSize = baseShapeInfo_.tSize;
    baseParams_->seq1Size = baseShapeInfo_.s1Size;
    baseParams_->seq2Size = baseShapeInfo_.s2Size;
    baseParams_->headSizeX = baseShapeInfo_.heSize;
    baseParams_->headSizeCq = baseShapeInfo_.hcqSize;
    baseParams_->headSizeCkv = baseShapeInfo_.hckvSize;
    baseParams_->dtileSize = baseShapeInfo_.dtileSize;
    baseParams_->headSizeQc = baseShapeInfo_.headSizeQc;
    baseParams_->headSizeQr = baseShapeInfo_.headSizeQr;
    baseParams_->headSizeKr = baseShapeInfo_.drSize;
    baseParams_->numHeadSize = baseShapeInfo_.nSize;
    baseParams_->numHeadKvSize = baseShapeInfo_.nkvSize;
    baseParams_->dimHeadSizeQc = baseShapeInfo_.dSize;
    baseParams_->dimHeadRope = baseShapeInfo_.drSize;
    baseParams_->blockNum = baseShapeInfo_.blockNum;
    baseParams_->blockSize = baseShapeInfo_.blockSize;
    baseParams_->mm1BlockNum = mm1BlockNum_;
    baseParams_->mm2BlockNum = mm2BlockNum_;
    baseParams_->mm3BlockNum = mm3BlockNum_;
    baseParams_->mm4BlockNum = mm4BlockNum_;
    baseParams_->mm1SingleCoreN = singlecoreHeadSizeCq_;
    baseParams_->mm2SingleCoreN = singlecoreHeadSizeCkvKr_;
    baseParams_->mm3SingleCoreN = singlecoreHeadSizeQcQr_;
    baseParams_->mm4SingleCoreBatch = singlecoreNumHeadSize_;
    baseParams_->vectorBlockNum = vectorBlockNum_;
    baseParams_->reciprocalCq = reciprocalCq_;
    baseParams_->epsilonCq = epsilonCq_;
    baseParams_->reciprocalCkv = reciprocalCkv_;
    baseParams_->epsilonCkv = epsilonCkv_;
    baseParams_->queryNormFlag = queryNormFlag_;
#if MLA_PROLOG_V3_SFA1
    baseParams_->outputCurKNopeRopeFlag = outputCurKNopeRope_;    baseParams_->kvQuantMode = static_cast<uint32_t>(kvQuantMode_);
#else
    baseParams_->kvQuantMode = static_cast<uint32_t>(kvQuantMode_);
#endif
    baseParams_->ckvkrRepoMode = ckvkrRepoMode_;
    baseParams_->quantScaleRepoMode = quantScaleRepoMode_;
    baseParams_->tileSize = tileSize_;
#if MLA_PROLOG_V3_SFA1

    float qcQrScale = qcQrScale_;
    float kcScale = kcScale_;
    baseParams_->qcQrScale = qcQrScale;
    baseParams_->kcScale = kcScale;
    if (std::abs(qcQrScale - 1.0f) >= std::numeric_limits<float>::epsilon()) {
        baseParams_->isQcQrScaleEnable = 1;
    } else {
        baseParams_->isQcQrScaleEnable = 0;
    }
    if (std::abs(kcScale - 1.0f) >= std::numeric_limits<float>::epsilon()) {
        baseParams_->isKcScaleEnable = 1;
    } else {
        baseParams_->isKcScaleEnable = 0;
    }

#else
    baseParams_->qcQrScale = qcQrScale_;
    baseParams_->kcScale = kcScale_;
    baseParams_->enableRope = enableRope_;
    baseParams_->isQcQrScaleEnable = static_cast<uint16_t>(std::abs(qcQrScale_ - 1.0f) >= std::numeric_limits<float>::epsilon());
    baseParams_->isKcScaleEnable = static_cast<uint16_t>(std::abs(kcScale_ - 1.0f) >= std::numeric_limits<float>::epsilon());
#endif
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MlaPrologV3Tiling::CalcWorkSpace()
{
    workspaceSize_ = libapiSize_;
    if (scenarioInfo_.quantMode_ == QUANT_MODE::FULL_QUANT_KV_NO_QUANT ||
        scenarioInfo_.quantMode_ == QUANT_MODE::FULL_QUANT_KV_QUANT_PER_TENSOR ||
        scenarioInfo_.quantMode_ == QUANT_MODE::FULL_QUANT_KV_QUANT_PER_TILE) {
        workspaceSize_ += static_cast<size_t>(stepBatchSize_) * static_cast<size_t>(baseShapeInfo_.hcqSize) *
                          static_cast<size_t>(NUM_BYTES_INT32);
        workspaceSize_ += static_cast<size_t>(stepBatchSize_) * static_cast<size_t>(baseShapeInfo_.hcqSize) *
                          static_cast<size_t>(NUM_BYTES_BF16);
        workspaceSize_ += static_cast<size_t>(stepBatchSize_) *
                          static_cast<size_t>(baseShapeInfo_.hckvSize + baseShapeInfo_.drSize) *
                          static_cast<size_t>(NUM_BYTES_INT32);
        if (scenarioInfo_.quantMode_ == QUANT_MODE::FULL_QUANT_KV_QUANT_PER_TENSOR) {
            // 全量化场景mmQnRes输出到workspace, B, S1, N, Hckv, BF16
            workspaceSize_ += static_cast<size_t>(stepBatchSize_) * static_cast<size_t>(baseShapeInfo_.nSize) *
                              static_cast<size_t>(baseShapeInfo_.hckvSize) * static_cast<size_t>(NUM_BYTES_BF16);
        }
    } else {
        workspaceSize_ += static_cast<size_t>(stepBatchSize_) * static_cast<size_t>(baseShapeInfo_.hcqSize) *
                          static_cast<size_t>(NUM_BYTES_BF16) * static_cast<size_t>(2);  // 2: double
        workspaceSize_ += static_cast<size_t>(stepBatchSize_) *
                          static_cast<size_t>(baseShapeInfo_.hckvSize + baseShapeInfo_.drSize) *
                          static_cast<size_t>(NUM_BYTES_BF16);
    }
    workspaceSize_ += static_cast<size_t>(stepBatchSize_) *
        static_cast<size_t>(baseShapeInfo_.headSizeQc + baseShapeInfo_.headSizeQr) *
        static_cast<size_t>(NUM_BYTES_INT32);
    workspaceSize_ += static_cast<size_t>(stepBatchSize_) * static_cast<size_t>(baseShapeInfo_.nSize) *
        static_cast<size_t>(baseShapeInfo_.dSize) * static_cast<size_t>(NUM_BYTES_BF16);

    if (enableGroupComputeOpt_ || enableDequantOpt_) {
        workspaceSize_ += static_cast<size_t>(stepBatchSize_) * static_cast<size_t>(BLOCK_SIZE);
    }
    workspaceSize_ += 1024 * 1024 * 1024;
    if (context_->workSpaces) {
        context_->workSpaces[0] = workspaceSize_;
    }
    OPS_LOG_I(context_->opName, "Tiling info: workspaceSize_ = %zu", workspaceSize_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MlaPrologV3Tiling::GenTilingKey() const
{
    uint32_t typeValue = 0;
    uint32_t quantType = 0;
    if (scenarioInfo_.quantMode_ == QUANT_MODE::NO_QUANT) {
        typeValue = 1U;
    } else {
        typeValue = 2U;
        // kvCache量化场景，对应tiling key为1(半量化:0 + kv量化:1)或3(全量化:2 + kv量化:1)
        // 全量化场景，对应tiling key为2+0(全量化:2)或2+1（全量化:2+ kv量化:1）
        // 非量化和半量化场景，对应tiling key为0
        quantType = static_cast<uint32_t>(scenarioInfo_.quantMode_);
    }

    if (scenarioInfo_.emptyTensorMode_ == EMPTY_TENSOR_MODE::EMPTY_QUERY) {
        context_->tilingKey = GET_TPL_TILING_KEY(
            0,
            0,
            0,
            false,
            false,
            static_cast<uint8_t>(scenarioInfo_.emptyTensorMode_),
            0,
#if MLA_PROLOG_V3_SFA1
            0
#else
            0,
            ASCENDC_TPL_MIX_AIC_1_2
#endif
        );
    } else {
        uint8_t cacheMode = scenarioInfo_.cacheMode_ == CACHE_MODE::TND ?
            0 : static_cast<uint8_t>(scenarioInfo_.cacheMode_);
        context_->tilingKey = GET_TPL_TILING_KEY(
            static_cast<uint8_t>(cacheMode),
            typeValue,
            quantType,
            enableDequantOpt_,
            enableGroupComputeOpt_,
            static_cast<uint8_t>(scenarioInfo_.emptyTensorMode_),
            static_cast<uint8_t>(scenarioInfo_.actualSeqMode_),
#if MLA_PROLOG_V3_SFA1
            static_cast<uint32_t>(scenarioInfo_.splitMFlag_)
#else
            static_cast<uint32_t>(scenarioInfo_.splitMFlag_),
            ASCENDC_TPL_MIX_AIC_1_2
#endif
        );
    }
    OPS_LOG_I(context_->opName, "MlaPrologV3 tilingKey:%lu", context_->tilingKey);

    return ge::GRAPH_SUCCESS;
}

#if MLA_PROLOG_V3_SFA1
ge::graphStatus MlaPrologV3Tiling::RunBigKernelTiling(MlaPrologV3Context &context, FluentMlaPrologV3TilingData* tilingData)
#else
ge::graphStatus MlaPrologV3Tiling::RunBigKernelTiling(MlaPrologV3Context &context, MlaPrologV3TilingData* tilingData)
#endif
{
    this->context_ = &context;
    this->baseParams_ = &tilingData->baseParams;
    MlaPrologV3TilingCheck tilingCheck_ {*context_, baseShapeInfo_, scenarioInfo_};

    OPS_LOG_I("Run big kernel");

    using StatusFunction = std::function<ge::graphStatus()>;
    std::vector<StatusFunction> requiredTilingFuncs {
        std::bind(&MlaPrologV3Tiling::GetNpuInfo, this),
        std::bind(&MlaPrologV3TilingCheck::CheckSingleRequiredParam, &tilingCheck_),
        std::bind(&MlaPrologV3TilingCheck::CheckCacheMode, &tilingCheck_),
        std::bind(&MlaPrologV3Tiling::SetShapeInfo, this),
        std::bind(&MlaPrologV3Tiling::SetAttrInfo, this),
        std::bind(&MlaPrologV3Tiling::SetScenarioInfo, this),
        std::bind(&MlaPrologV3TilingCheck::CheckScenarParam, &tilingCheck_),
        std::bind(&MlaPrologV3TilingCheck::CheckPANZPerTile, &tilingCheck_),
        std::bind(&MlaPrologV3TilingCheck::CheckDims, &tilingCheck_),
        std::bind(&MlaPrologV3TilingCheck::CheckParamByScenario, &tilingCheck_),
        std::bind(&MlaPrologV3Tiling::ProcessBaseInputs, this),
    };
    for (const auto &func: requiredTilingFuncs) {
        if (func() != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }

    if (scenarioInfo_.emptyTensorMode_ == EMPTY_TENSOR_MODE::EMPTY_QUERY) {
        FillTiling();
        if (context_->workSpaces) {
            context_->workSpaces[0] = libapiSize_;
        }
        GenTilingKey();
        context_->blockDim = 1U;
        return ge::GRAPH_SUCCESS;
    }

    std::vector<StatusFunction> optionalTilingFuncs {
        std::bind(&MlaPrologV3Tiling::FillMatmul1Tiling, this),
        std::bind(&MlaPrologV3Tiling::FillMatmul2Tiling, this),
        std::bind(&MlaPrologV3Tiling::FillMatmul3Tiling, this),
        std::bind(&MlaPrologV3Tiling::FillMatmul4Tiling, this),
        std::bind(&MlaPrologV3Tiling::FillTiling, this),
        std::bind(&MlaPrologV3Tiling::CalcWorkSpace, this),
        std::bind(&MlaPrologV3Tiling::GenTilingKey, this)
    };
    for (const auto &func : optionalTilingFuncs) {
        if (func() != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }

    context_->blockDim = aicNum_;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MlaPrologV3Tiling::ConvertContext(gert::TilingContext &context, MlaPrologV3Context &mlaPrologV3Context)
{
    if (context.GetNodeName() == nullptr) {
        OPS_LOG_E(V1_OP_NAME, "opName got from TilingContext is nullptr");
        return ge::GRAPH_FAILED;
    }

    OPS_LOG_I("Getting Context");

    mlaPrologV3Context.opName = context.GetNodeName();
    mlaPrologV3Context.opType = context.GetNodeType();
    mlaPrologV3Context.platformInfo = context.GetPlatformInfo();
    ConvertRequiredParams(context, mlaPrologV3Context);
    ConvertOptionalParams(context, mlaPrologV3Context);

    auto attrs = context.GetAttrs();
    OPS_ERR_IF(attrs == nullptr, OPS_LOG_E(context.GetNodeName(), "attrs got from ge is nullptr"),
               return ge::GRAPH_FAILED);
    mlaPrologV3Context.rmsNormEspilonCq = attrs->
        GetAttrPointer<float>(RMS_NORM_EPSILON_CQ_ATTR_INDEX);
    mlaPrologV3Context.rmsNormEspilonCkv = attrs->GetAttrPointer<float>(RMS_NORM_EPSILON_CKV_ATTR_INDEX);
    mlaPrologV3Context.cacheMode = attrs->GetStr(CACHE_MODE_ATTR_INDEX);
    mlaPrologV3Context.queryNormFlag = attrs->GetAttrPointer<bool>(QUERY_NORM_ATTR_INDEX);
#if MLA_PROLOG_V3_SFA1
    mlaPrologV3Context.outputCurKNopeRopeFlag = attrs->GetAttrPointer<bool>(OUTPUT_CUR_K_NOPE_ROPE_ATTR_INDEX);
#endif

    mlaPrologV3Context.weightQuantMode = attrs->GetAttrPointer<int>(WEIGHT_QUANT_MODE_INDEX);
    mlaPrologV3Context.kvQuantMode = attrs->GetAttrPointer<int>(KV_QUANT_MODE_INDEX);
    mlaPrologV3Context.queryQuantMode = attrs->GetAttrPointer<int>(QUERY_QUANT_MODE_INDEX);
    mlaPrologV3Context.ckvkrRepoMode = attrs->GetAttrPointer<int>(CKVKR_REPO_MODE_INDEX);
    mlaPrologV3Context.quantScaleRepoMode = attrs->GetAttrPointer<int>(QUANT_SCALE_REPO_MODE_INDEX);
    mlaPrologV3Context.tileSize = attrs->GetAttrPointer<int>(TILE_SIZE_INDEX);
    mlaPrologV3Context.qcQrScale = attrs->GetAttrPointer<float>(QCQR_SCALE_INDEX);
    mlaPrologV3Context.kcScale = attrs->GetAttrPointer<float>(KC_SCALE_INDEX);
#if !MLA_PROLOG_V3_SFA1
    mlaPrologV3Context.enableRope = attrs->GetAttrPointer<bool>(ENABLE_ROPE_INDEX);
#endif

    int32_t weightQuantMode = *mlaPrologV3Context.weightQuantMode;
    int32_t kvQuantMode = *mlaPrologV3Context.kvQuantMode;
    int32_t queryQuantMode = *mlaPrologV3Context.queryQuantMode;
    int32_t ckvkrRepoMode = *mlaPrologV3Context.ckvkrRepoMode;
    int32_t quantScaleRepoMode = *mlaPrologV3Context.quantScaleRepoMode;
    int32_t tileSize = *mlaPrologV3Context.tileSize;

    float qcQrScale = *mlaPrologV3Context.qcQrScale;
    float kcScale = *mlaPrologV3Context.kcScale;

    OPS_ERR_IF(context.GetWorkspaceSizes(1) == nullptr,
               OPS_REPORT_VECTOR_INNER_ERR(context.GetNodeName(), "workSpaceSize got from ge is nullptr"),
               return ge::GRAPH_FAILED);
    mlaPrologV3Context.workSpaces = context.GetWorkspaceSizes(1);
    return ge::GRAPH_SUCCESS;
}

void MlaPrologV3Tiling::ConvertRequiredParams(gert::TilingContext &context, MlaPrologV3Context &mlaPrologV3Context)
{
    mlaPrologV3Context.tokenX.desc = context.GetRequiredInputDesc(TOKEN_X_INPUT_INDEX);
    mlaPrologV3Context.tokenX.shape = context.GetRequiredInputShape(TOKEN_X_INPUT_INDEX);
    mlaPrologV3Context.weightDq.desc = context.GetRequiredInputDesc(WEIGHT_DQ_INPUT_INDEX);
    mlaPrologV3Context.weightDq.shape = context.GetRequiredInputShape(WEIGHT_DQ_INPUT_INDEX);
    mlaPrologV3Context.weightUqQr.desc = context.GetRequiredInputDesc(WEIGHT_UQ_QR_INPUT_INDEX);
    mlaPrologV3Context.weightUqQr.shape = context.GetRequiredInputShape(WEIGHT_UQ_QR_INPUT_INDEX);
    mlaPrologV3Context.weightUk.desc = context.GetRequiredInputDesc(WEIGHT_UK_INPUT_INDEX);
    mlaPrologV3Context.weightUk.shape = context.GetRequiredInputShape(WEIGHT_UK_INPUT_INDEX);
    mlaPrologV3Context.weightDkvKr.desc = context.GetRequiredInputDesc(WEIGHT_DKV_KR_INPUT_INDEX);
    mlaPrologV3Context.weightDkvKr.shape = context.GetRequiredInputShape(WEIGHT_DKV_KR_INPUT_INDEX);
    mlaPrologV3Context.rmsnormGammaCq.desc = context.GetRequiredInputDesc(RMSNORM_GAMMA_CQ_INPUT_INDEX);
    mlaPrologV3Context.rmsnormGammaCq.shape = context.GetRequiredInputShape(RMSNORM_GAMMA_CQ_INPUT_INDEX);
    mlaPrologV3Context.rmsnormGammaCkv.desc = context.GetRequiredInputDesc(RMS_NORM_GAMMA_CKV_INPUT_INDEX);
    mlaPrologV3Context.rmsnormGammaCkv.shape = context.GetRequiredInputShape(RMS_NORM_GAMMA_CKV_INPUT_INDEX);
    mlaPrologV3Context.ropeSin.desc = context.GetRequiredInputDesc(ROPE_SIN_INPUT_INDEX);
    mlaPrologV3Context.ropeSin.shape = context.GetRequiredInputShape(ROPE_SIN_INPUT_INDEX);
    mlaPrologV3Context.ropeCos.desc = context.GetRequiredInputDesc(ROPE_COS_INPUT_INDEX);
    mlaPrologV3Context.ropeCos.shape = context.GetRequiredInputShape(ROPE_COS_INPUT_INDEX);
    mlaPrologV3Context.kvCache.desc = context.GetRequiredInputDesc(KV_CACHE_INPUT_INDEX);
    mlaPrologV3Context.kvCache.shape = context.GetRequiredInputShape(KV_CACHE_INPUT_INDEX);
    mlaPrologV3Context.krCache.desc = context.GetRequiredInputDesc(KR_CACHE_INPUT_INDEX);
    mlaPrologV3Context.krCache.shape = context.GetRequiredInputShape(KR_CACHE_INPUT_INDEX);

    mlaPrologV3Context.query.desc = context.GetOutputDesc(QUERY_OUTPUT_INDEX);
    mlaPrologV3Context.query.shape = context.GetOutputShape(QUERY_OUTPUT_INDEX);
    mlaPrologV3Context.queryRope.desc = context.GetOutputDesc(QUERY_ROPE_OUTPUT_INDEX);
    mlaPrologV3Context.queryRope.shape = context.GetOutputShape(QUERY_ROPE_OUTPUT_INDEX);
    mlaPrologV3Context.kvCacheOut.desc = context.GetOutputDesc(KV_CACHE_OUT_OUTPUT_INDEX);
    mlaPrologV3Context.kvCacheOut.shape = context.GetOutputShape(KV_CACHE_OUT_OUTPUT_INDEX);
    mlaPrologV3Context.krCacheOut.desc = context.GetOutputDesc(KR_CACHE_OUT_OUTPUT_INDEX);
    mlaPrologV3Context.krCacheOut.shape = context.GetOutputShape(KR_CACHE_OUT_OUTPUT_INDEX);
#if MLA_PROLOG_V3_SFA1
    mlaPrologV3Context.curKNope.desc = context.GetOutputDesc(CUR_K_NOPE_OUTPUT_INDEX);
    mlaPrologV3Context.curKNope.shape = context.GetOutputShape(CUR_K_NOPE_OUTPUT_INDEX);
    mlaPrologV3Context.curKRope.desc = context.GetOutputDesc(CUR_K_ROPE_OUTPUT_INDEX);
    mlaPrologV3Context.curKRope.shape = context.GetOutputShape(CUR_K_ROPE_OUTPUT_INDEX);
#endif
}

void MlaPrologV3Tiling::ConvertOptionalParams(gert::TilingContext &context, MlaPrologV3Context &mlaPrologV3Context)
{
    mlaPrologV3Context.cacheIndex.desc = context.GetOptionalInputDesc(CACHE_INDEX_INPUT_INDEX);
    mlaPrologV3Context.cacheIndex.shape = context.GetOptionalInputShape(CACHE_INDEX_INPUT_INDEX);
    mlaPrologV3Context.dequantScaleX.desc = context.GetOptionalInputDesc(DEQUANT_SCALE_X_INDEX);
    mlaPrologV3Context.dequantScaleX.shape = context.GetOptionalInputShape(DEQUANT_SCALE_X_INDEX);
    mlaPrologV3Context.dequantScaleWDq.desc = context.GetOptionalInputDesc(DEQUANT_SCALE_W_DQ_INDEX);
    mlaPrologV3Context.dequantScaleWDq.shape = context.GetOptionalInputShape(DEQUANT_SCALE_W_DQ_INDEX);
    mlaPrologV3Context.dequantScaleWUqQr.desc = context.GetOptionalInputDesc(DEQUANT_SCALE_W_UQ_QR_INDEX);
    mlaPrologV3Context.dequantScaleWUqQr.shape = context.GetOptionalInputShape(DEQUANT_SCALE_W_UQ_QR_INDEX);
    mlaPrologV3Context.dequantScaleWDkvKr.desc = context.GetOptionalInputDesc(DEQUANT_SCALE_W_DKV_KR_INDEX);
    mlaPrologV3Context.dequantScaleWDkvKr.shape = context.GetOptionalInputShape(DEQUANT_SCALE_W_DKV_KR_INDEX);
    mlaPrologV3Context.quantScaleCkv.desc = context.GetOptionalInputDesc(QUANT_SCALE_CKV_INDEX);
    mlaPrologV3Context.quantScaleCkv.shape = context.GetOptionalInputShape(QUANT_SCALE_CKV_INDEX);
    mlaPrologV3Context.quantScaleCkr.desc = context.GetOptionalInputDesc(QUANT_SCALE_CKR_INDEX);
    mlaPrologV3Context.quantScaleCkr.shape = context.GetOptionalInputShape(QUANT_SCALE_CKR_INDEX);
    mlaPrologV3Context.smoothScalesCq.desc = context.GetOptionalInputDesc(SMOOTH_SCALES_CQ_INDEX);
    mlaPrologV3Context.smoothScalesCq.shape = context.GetOptionalInputShape(SMOOTH_SCALES_CQ_INDEX);
    mlaPrologV3Context.actualSeqLen.desc = context.GetOptionalInputDesc(ACTUAL_SEQ_LEN_INDEX);
    mlaPrologV3Context.actualSeqLen.shape = context.GetOptionalInputShape(ACTUAL_SEQ_LEN_INDEX);
    mlaPrologV3Context.kNopeClipAlpha.desc = context.GetOptionalInputDesc(K_NOPE_CLIP_ALPHA_INDEX);
    mlaPrologV3Context.kNopeClipAlpha.shape = context.GetOptionalInputShape(K_NOPE_CLIP_ALPHA_INDEX);
    // only v1 does not support dequantScaleQNope
    if (strcmp(mlaPrologV3Context.opType, V1_OP_NAME) == 0) {
        mlaPrologV3Context.dequantScaleQNope.desc = nullptr;
        mlaPrologV3Context.dequantScaleQNope.shape = nullptr;
    } else {
        mlaPrologV3Context.dequantScaleQNope.desc = context.GetOutputDesc(DEQUANT_SCALE_Q_NOPE_OUTPUT_INDEX);
        mlaPrologV3Context.dequantScaleQNope.shape = context.GetOutputShape(DEQUANT_SCALE_Q_NOPE_OUTPUT_INDEX);
    }
    if (strcmp(mlaPrologV3Context.opType, V3_OP_NAME) == 0) {
        mlaPrologV3Context.queryNorm.desc = context.GetOutputDesc(QUERY_NORM_OUTPUT_INDEX);
        mlaPrologV3Context.queryNorm.shape = context.GetOutputShape(QUERY_NORM_OUTPUT_INDEX);
        mlaPrologV3Context.dequantScaleQNorm.desc = context.GetOutputDesc(DEQUANT_SCALE_Q_NORM_OUTPUT_INDEX);
        mlaPrologV3Context.dequantScaleQNorm.shape = context.GetOutputShape(DEQUANT_SCALE_Q_NORM_OUTPUT_INDEX);
    } else {
        mlaPrologV3Context.queryNorm.desc = nullptr;
        mlaPrologV3Context.queryNorm.shape = nullptr;
        mlaPrologV3Context.dequantScaleQNorm.desc = nullptr;
        mlaPrologV3Context.dequantScaleQNorm.shape = nullptr;
    }
}

#if MLA_PROLOG_V3_SFA1
MLA_EXTERN_C ge::graphStatus TilingFluentMlaPrologV3(gert::TilingContext *context)
#else
MLA_EXTERN_C ge::graphStatus TilingMlaPrologV3(gert::TilingContext *context)
#endif
{
    OPS_ERR_IF(context == nullptr, OPS_REPORT_VECTOR_INNER_ERR(V1_OP_NAME, "Context is nullptr."),
               return ge::GRAPH_FAILED);

    OPS_LOG_I("Getting Tiling");

    MlaPrologV3Context mlaPrologV3Context{};
    if (MlaPrologV3Tiling::ConvertContext(*context, mlaPrologV3Context) != ge::GRAPH_SUCCESS) {
        OPS_LOG_E(context->GetNodeName(), "Error occurred while converting tilingContext to MlaPrologV3 context");
        return ge::GRAPH_FAILED;
    }

    MlaPrologV3Tiling mlaPrologV3Tiling;
#if MLA_PROLOG_V3_SFA1
    FluentMlaPrologV3TilingData* tilingData = context->GetTilingData<FluentMlaPrologV3TilingData>();
#else
    MlaPrologV3TilingData* tilingData = context->GetTilingData<MlaPrologV3TilingData>();
#endif
    OPS_ERR_IF(tilingData == nullptr,
            OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "TilingData is nullptr."),
            return ge::GRAPH_FAILED);
    if (mlaPrologV3Tiling.RunBigKernelTiling(mlaPrologV3Context, tilingData) == ge::SUCCESS) {
        context->SetTilingKey(mlaPrologV3Context.tilingKey);
        context->SetBlockDim(mlaPrologV3Context.blockDim);
        return ge::GRAPH_SUCCESS;
    }
    return ge::GRAPH_FAILED;
}
ge::graphStatus TilingPrepareForMlaPrologV3(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

#if MLA_PROLOG_V3_SFA1
IMPL_OP_OPTILING(FluentMlaPrologV3)
    .Tiling(TilingFluentMlaPrologV3)
#else
IMPL_OP_OPTILING(MlaPrologV3)
    .Tiling(TilingMlaPrologV3)
#endif
    .TilingParse<MlaPrologV3CompileInfo>(TilingPrepareForMlaPrologV3);
} // namespace implementation
#if MLA_PROLOG_V3_SFA1
} // namespace optiling
#endif
