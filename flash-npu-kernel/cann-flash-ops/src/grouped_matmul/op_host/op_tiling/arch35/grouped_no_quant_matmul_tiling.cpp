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
 * \file grouped_no_quant_matmul_tiling.cpp
 * \brief
 */
#include "grouped_no_quant_matmul_tiling.h"
#include "../../../op_kernel/arch35/non_quant/grouped_matmul_tiling_key.h"

enum class GmmTrans {
    NoTrans = 0,
    ATrans = 1,
    BTrans = 2,
    ABTrans = 3
};
namespace optiling {
bool GroupedNoQuantMatmulTiling::SetTiling(gert::TilingContext *context)
{
    auto compileInfoPtr = context->GetCompileInfo<GMMCompileInfo>();
    OP_CHECK_IF(compileInfoPtr == nullptr, OP_LOGE(context->GetNodeName(), "compileInfoPtr is nullptr."), return false);
    usedCoreNum_ = compileInfoPtr->aicNum;
    OP_CHECK_IF(!Init(context), OP_LOGE(context->GetNodeName(), "Init failed"), return false);
    OP_CHECK_IF(!CalMatMulTiling(context, compileInfoPtr),
                OP_LOGE(context->GetNodeName(), "Unable to calculate matmul-tiling"), return false);
    SetGMMTiling();
    SetMatMulTiling();
    SetTilingKey(context);
    OP_CHECK_IF(!SetCustomParam(context), OP_LOGE(context->GetNodeName(), "Unable to set custom param"), return false);
    PrintTilingResult(context);
    return true;
}

bool GroupedNoQuantMatmulTiling::CalBaseMMTiling(const gert::TilingContext *context,
                                                 const GMMCompileInfo *compileInfoPtr)
{
    // according to the double buffer enabled L0B, compute baseK
    baseK_ = (compileInfoPtr->l0BSize / DB_SIZE) / (baseN_ * GetSizeByDataType(xDType_));
    baseK_ = baseK_ & ~ALIGN_DOWN_16; // 16 bytes down-align
    OP_CHECK_IF(baseK_ == 0, OP_LOGE(context->GetNodeName(), "baseK_ cannot be 0."), return false);
    // according to the double buffer enabled L0A/L0C, compute baseM(cube)
    uint32_t maxBaseM = static_cast<uint32_t>(compileInfoPtr->l0CSize / (baseN_ * FP32_DTYPE_SIZE));
    baseM_ = std::min<uint32_t>((compileInfoPtr->l0ASize / DB_SIZE) / (baseK_ * GetSizeByDataType(xDType_)), maxBaseM);
    if (baseM_ > BASE_M_DEFAULT) {
        baseM_ = BASE_M_DEFAULT;
    }
    OP_CHECK_IF(baseM_ == 0, OP_LOGE(context->GetNodeName(), "baseM_ cannot be 0."), return false);
    return CalL1Tiling(context, compileInfoPtr);
}

void GroupedNoQuantMatmulTiling::FormulateBasicBlock(const GMMCompileInfo *compileInfoPtr, uint32_t remainCoreNum)
{
    uint64_t mCnt = GroupedMatmul::CeilDiv(m_, baseM_);
    uint64_t nCnt = GroupedMatmul::CeilDiv(n_, baseN_);
    if (mCnt * nCnt >= remainCoreNum) {
        return;
    }
    if (mCnt <= nCnt) {
        baseM_ = GroupedMatmul::CeilAlign(GroupedMatmul::CeilDiv(m_, mCnt), BASIC_BLOCK_SIZE_16);
        mCnt = GroupedMatmul::CeilDiv(m_, baseM_);
        nCnt = remainCoreNum / mCnt;
        baseN_ = GroupedMatmul::CeilAlign(GroupedMatmul::CeilDiv(n_, nCnt), BASIC_BLOCK_SIZE_16);
    } else {
        baseN_ = GroupedMatmul::CeilAlign(GroupedMatmul::CeilDiv(n_, nCnt), BASIC_BLOCK_SIZE_16);
        nCnt = GroupedMatmul::CeilDiv(n_, baseN_);
        mCnt = remainCoreNum / nCnt;
        baseM_ = GroupedMatmul::CeilAlign(GroupedMatmul::CeilDiv(m_, mCnt), BASIC_BLOCK_SIZE_16);
    }

    while (baseN_ >= baseM_ * NUM_TWO && nCnt < remainCoreNum / NUM_TWO) {
        nCnt = nCnt * NUM_TWO;
        mCnt = remainCoreNum / nCnt;
        baseM_ = GroupedMatmul::CeilAlign(GroupedMatmul::CeilDiv(m_, mCnt), BASIC_BLOCK_SIZE_16);
        baseN_ = GroupedMatmul::CeilAlign(GroupedMatmul::CeilDiv(n_, nCnt), BASIC_BLOCK_SIZE_16);
        mCnt = GroupedMatmul::CeilDiv(m_, baseM_);
        nCnt = GroupedMatmul::CeilDiv(n_, baseN_);
    }

    while (baseM_ >= baseN_ * NUM_TWO && mCnt < remainCoreNum / NUM_TWO) {
        mCnt = mCnt * NUM_TWO;
        nCnt = remainCoreNum / mCnt;
        baseM_ = GroupedMatmul::CeilAlign(GroupedMatmul::CeilDiv(m_, mCnt), BASIC_BLOCK_SIZE_16);
        baseN_ = GroupedMatmul::CeilAlign(GroupedMatmul::CeilDiv(n_, nCnt), BASIC_BLOCK_SIZE_16);
        mCnt = GroupedMatmul::CeilDiv(m_, baseM_);
        nCnt = GroupedMatmul::CeilDiv(n_, baseN_);
    }
    mCnt = GroupedMatmul::CeilDiv(m_, baseM_);
    nCnt = GroupedMatmul::CeilDiv(n_, baseN_);
    uint64_t kValueAlign = GroupedMatmul::CeilAlign(k_, BASIC_BLOCK_SIZE_16);
    uint64_t kValueMax =
        GroupedMatmul::FloorAlign(static_cast<uint64_t>(compileInfoPtr->l0ASize / DB_SIZE / GetSizeByDataType(xDType_) /
                                                        std::max(baseM_, baseN_)),
                                  BASIC_BLOCK_SIZE_16);
    baseK_ = std::min(kValueAlign, kValueMax);
    usedCoreNum_ = std::min(static_cast<uint32_t>(mCnt * nCnt * groupNum_), compileInfoPtr->aicNum);
}

void GroupedNoQuantMatmulTiling::CalAswtL1Tiling(const GMMCompileInfo *compileInfoPtr)
{
    uint64_t totalL1Size = compileInfoPtr->l1Size + 256UL; // 256B为预留给rpc使用，单算子不涉及
    uint64_t reserveBTSize = hasBias_ ? BIAS_TABLE_NUM * DATA_SIZE_FP32 : 0UL;
    depthA1_ = totalL1Size / NUM_TWO / baseM_ / baseK_ / GetSizeByDataType(xDType_);      // 2: half of l1
    depthB1_ = totalL1Size / NUM_TWO / baseN_ / baseK_ / GetSizeByDataType(weightDtype_); // 2: half of l1

    uint64_t depthASize = depthA1_ * baseM_ * baseK_ * GetSizeByDataType(xDType_);
    uint64_t depthBSize = depthB1_ * baseN_ * baseK_ * GetSizeByDataType(weightDtype_);
    if (depthASize + depthBSize > totalL1Size - reserveBTSize) {
        if (baseM_ <= baseN_) {
            depthA1_ = std::max(depthA1_ / NUM_TWO, 1UL); // 2: adjust deptch for l1 buffer
        } else {
            depthB1_ = std::max(depthB1_ / NUM_TWO, 1UL); // 2: adjust deptch for l1 buffer
        }
    }
    stepKa_ = std::max(depthA1_ / DB_SIZE, 1UL);
    stepKb_ = std::max(depthB1_ / DB_SIZE, 1UL);
    if (stepKa_ >= stepKb_) {
        stepKa_ = stepKa_ / stepKb_ * stepKb_;
    } else {
        stepKb_ = stepKb_ / stepKa_ * stepKa_;
    }
    depthA1_ = stepKa_ * DB_SIZE; // depth % (stepKa * stepM) == 0
    depthB1_ = stepKb_ * DB_SIZE; // depth % (stepKb * stepN) == 0
    return;
}

bool GroupedNoQuantMatmulTiling::CalL1Tiling(const gert::TilingContext *context, const GMMCompileInfo *compileInfoPtr)
{
    uint64_t reserveBTSize = hasBias_ ? BIAS_TABLE_NUM * DATA_SIZE_FP32 : 0UL;
    uint64_t totalL1Size = hasBias_ ? compileInfoPtr->l1Size - reserveBTSize : compileInfoPtr->l1Size;
    uint64_t l1ASize = baseM_ > baseN_ ? PARTA_L1_SIZE : totalL1Size - PARTA_L1_SIZE;
    uint64_t l1BSize = totalL1Size - l1ASize;
    stepKa_ = l1ASize / NUM_TWO / baseM_ / baseK_ / GetSizeByDataType(xDType_);      // 2: half of l1
    stepKb_ = l1BSize / NUM_TWO / baseN_ / baseK_ / GetSizeByDataType(weightDtype_); // 2: half of l1

    OP_CHECK_IF(stepKa_ == 0 || stepKb_ == 0, OP_LOGE(context->GetNodeName(), "stepka or stepkb cannot be 0"),
                return false);
    if (stepKa_ >= stepKb_) {
        stepKa_ = stepKa_ / stepKb_ * stepKb_;
    } else {
        stepKb_ = stepKb_ / stepKa_ * stepKa_;
    }
    depthA1_ = stepKa_ * DB_SIZE;
    depthB1_ = stepKb_ * DB_SIZE;
    return true;
}

void GroupedNoQuantMatmulTiling::CalcTailBasicBlock(const GMMCompileInfo *compileInfoPtr)
{
    uint64_t mCnt = GroupedMatmul::CeilDiv(m_, baseM_);
    uint64_t nCnt = GroupedMatmul::CeilDiv(n_, baseN_);
    uint64_t mnCnt = mCnt * nCnt;
    uint64_t tailCnt = mnCnt <= compileInfoPtr->aicNum ? 0UL : mnCnt % compileInfoPtr->aicNum;

    if (tailCnt != 0UL) {
        while ((mTailCnt_ + 1UL) * nTailCnt_ * tailCnt <= compileInfoPtr->aicNum) {
            mTailCnt_ += 1UL;
            if (mTailCnt_ * (nTailCnt_ + 1UL) * tailCnt <= compileInfoPtr->aicNum) {
                nTailCnt_ += 1UL;
            }
        }
    }
}

/** @brief only support split M :s-s-s and s-m-s
 */
void GroupedNoQuantMatmulTiling::SetDisableL2Cache(const gert::TilingContext *context,
                                                   const GMMCompileInfo *compileInfoPtr)
{
    uint64_t innerB = transposeWeight_ ? k_ : n_;
    uint64_t dataTypeSize = GetSizeByDataType(xDType_);
    bool flag = transposeWeight_ ? (baseK_ * stepKb_ * dataTypeSize % ALIGN_128 == 0) :
                                   (baseN_ * dataTypeSize % ALIGN_128 == 0);
    uint64_t totalSize = m_ * k_ * dataTypeSize +
                         groupNum_ * k_ * n_ * dataTypeSize +
                         m_ * n_ * dataTypeSize;
    OP_LOGD(context->GetNodeName(), "Input and Output TotalSize: %lu, l2Size: %lu", totalSize, compileInfoPtr->l2Size);
    if (totalSize < compileInfoPtr->l2Size) {
        weightNoL2Cache_ = false;
        return;
    }
    weightNoL2Cache_ = ((innerB * dataTypeSize % ALIGN_128 == 0) && flag || weightNzFlag_) && weightNoL2Cache_;
}

bool GroupedNoQuantMatmulTiling::Init(const gert::TilingContext *context)
{
    OP_CHECK_IF(!GetAttrs(context), OP_LOGE(context->GetNodeName(), "Unable to calculate matmul-tiling"), return false);

    auto xTensor = context->GetDynamicInputTensor(INDEX_X, 0);
    OP_CHECK_IF(xTensor == nullptr, OP_LOGE(context->GetNodeName(), "xTensor is nullptr."), return false);
    gert::Shape xShape = xTensor->GetStorageShape();
    xDimNum_ = static_cast<uint32_t>(xShape.GetDimNum());

    auto wTensor = context->GetDynamicInputTensor(INDEX_WEIGHT, 0);
    OP_CHECK_IF(wTensor == nullptr, OP_LOGE(context->GetNodeName(), "wTensor is nullptr."), return false);
    gert::Shape wShape = wTensor->GetOriginShape();
    uint32_t wDimNum = static_cast<uint32_t>(wShape.GetDimNum());

    OP_CHECK_IF(wDimNum < MIN_DIM || xDimNum_ < MIN_DIM,
                OP_LOGE(context->GetNodeName(), "The dimension of x or weight should be at least 2"), return false);
    xKDim_ = transposeX_ ? 0U : xDimNum_ - 1U;
    weightNDim_ = transposeWeight_ ? wDimNum - DIM_TWO : wDimNum - DIM_ONE;

    auto biasPtr = context->GetDynamicInputTensor(INDEX_BIAS, 0); // 0: obtain the first tensor of the tensorList
    hasBias_ = !(biasPtr == nullptr || biasPtr->GetStorageShape().GetShapeSize() == 0);

    isSingleWeight_ = (context->GetDynamicInputTensor(INDEX_WEIGHT, 1) == nullptr);
    isSingleX_ = (context->GetDynamicInputTensor(INDEX_X, 1) == nullptr);
    isSingleY_ = (splitItem_ == 2 || splitItem_ == 3); // 2&3: output tensor is single

    if (weightNzFlag_) {
        uint64_t c0 = BLOCK_SIZE / std::max(1, GetSizeByDataType(weightDtype_));
        if (wDimNum > NZ_DIM_NUM) {
            weightNDim_ = transposeWeight_ ? wDimNum - DIM_THREE : wDimNum - DIM_FOUR;
            nzFactor_ = transposeWeight_ ? BASIC_BLOCK_SIZE_16 : static_cast<int64_t>(c0);
        }
    }

    if (groupType_ == SPLIT_K) {
        return GMMGetTensorShapeSplitK(context, xShape, wShape);
    }
    if (groupType_ == SPLIT_M) {
        return GMMGetTensorShapeSplitM(context, xShape, wShape);
    }
    if (groupType_ == NO_SPLIT) {              // not split any axis
        if (isSingleWeight_ && wDimNum > 2U) { // 2: dim of splited weight tensor
            return SeparatedXSingleWeight(context, wShape);
        }
        return SeparatedXSeparatedWeight(context);
    }
    OP_LOGE(context->GetNodeName(),
            "GMM_tiling: not support groupType_=%d, isSingleWeight_=%d, isSingleX_=%d, isSingleY_=%d", groupType_,
            isSingleWeight_, isSingleX_, isSingleY_);
    return false;
}

bool GroupedNoQuantMatmulTiling::GetAttrs(const gert::TilingContext *context)
{
    auto attr = context->GetAttrs();
    OP_CHECK_IF(attr == nullptr, OP_LOGE(context->GetNodeName(), "attr is nullptr."),
                return false); // check attr is not null
    const bool *transposeWeightPtr = attr->GetAttrPointer<bool>(ATTR_IDX_TRANS_W);
    const bool *transposeXPtr = attr->GetAttrPointer<bool>(ATTR_IDX_TRANS_X);
    const int32_t *groupTypePtr = attr->GetAttrPointer<int32_t>(ATTR_IDX_GROUPTYPE);
    const int64_t *splitItemPtr = attr->GetAttrPointer<int64_t>(ATTR_IDX_SPLIT_ITEM);
    const uint32_t *groupListTypePtr = attr->GetAttrPointer<uint32_t>(ATTR_IDX_GROUP_LIST_TYPE);
    transposeWeight_ = transposeWeightPtr != nullptr ? *transposeWeightPtr : false;
    transposeX_ = transposeXPtr != nullptr ? *transposeXPtr : false;
    groupType_ = groupTypePtr != nullptr ? *groupTypePtr : NO_SPLIT;
    splitItem_ = splitItemPtr != nullptr ? *splitItemPtr : 0;
    groupListType_ = groupListTypePtr != nullptr ? *groupListTypePtr : 0;

    auto xDesc = context->GetDynamicInputDesc(INDEX_X, 0);
    OP_CHECK_IF(xDesc == nullptr, OP_LOGE(context->GetNodeName(), "xDesc is nullptr."), return false);
    xDType_ = xDesc->GetDataType();

    auto w0Desc = context->GetDynamicInputDesc(INDEX_WEIGHT, 0);
    OP_CHECK_IF(w0Desc == nullptr, OP_LOGE(context->GetNodeName(), "w0Desc is nullptr."), return false);
    weightDtype_ = w0Desc->GetDataType();
    auto wFormat0 = static_cast<ge::Format>(ge::GetPrimaryFormat(w0Desc->GetStorageFormat()));
    weightNzFlag_ = wFormat0 == ge::FORMAT_FRACTAL_NZ;
    return true;
}

bool GroupedNoQuantMatmulTiling::CalMatMulTiling(const gert::TilingContext *context,
                                                 const GMMCompileInfo *compileInfoPtr)
{
    if (groupNum_ < 1U || groupNum_ > MAX_TENSOR) {
        OP_LOGE(context->GetNodeName(), "GMM no quant group num cannot less than 1 or larger than 1024.");
        return false;
    }
    OP_CHECK_IF(kZero, OP_LOGE(context->GetNodeName(), "GMM no quant case does not support input K is 0."),
                return false);
    OP_CHECK_IF(!CalBaseMMTiling(context, compileInfoPtr),
                OP_LOGE(context->GetNodeName(), "Unable to calculate BaseMMTiling."), return false);
    if (groupNum_ == 1U) {
        FormulateBasicBlock(compileInfoPtr, usedCoreNum_);
        CalcTailBasicBlock(compileInfoPtr);
        CalAswtL1Tiling(compileInfoPtr);
        SetDisableL2Cache(context, compileInfoPtr);
        return true;
    }
    if (groupType_ == SPLIT_M || groupType_ == NO_SPLIT) {
        SetDisableL2Cache(context, compileInfoPtr);
        return true;
    } else if (groupType_ == SPLIT_K) {
        uint32_t remainCoreNum = std::max(1U, compileInfoPtr->aicNum / groupNum_);
        FormulateBasicBlock(compileInfoPtr, remainCoreNum);
        return true;
    }
    return false;
}

void GroupedNoQuantMatmulTiling::SetGMMTiling()
{
    tilingData_.gmmNoQuantParam.groupNum = groupNum_;
    tilingData_.gmmNoQuantParam.hasBias = static_cast<uint32_t>(hasBias_);
    tilingData_.gmmNoQuantParam.groupType = groupType_;
    tilingData_.gmmNoQuantParam.groupListType = groupListType_;
    tilingData_.gmmNoQuantParam.singleWeight = static_cast<uint32_t>(isSingleWeight_);
    tilingData_.gmmNoQuantParam.singleX = static_cast<uint32_t>(isSingleX_);
    tilingData_.gmmNoQuantParam.singleY = static_cast<uint32_t>(isSingleY_);
    tilingData_.gmmNoQuantParam.coreNum = usedCoreNum_;
    tilingData_.gmmNoQuantParam.mTailCnt = static_cast<uint32_t>(mTailCnt_);
    tilingData_.gmmNoQuantParam.nTailCnt = static_cast<uint32_t>(nTailCnt_);
    tilingData_.gmmNoQuantParam.weightNoL2Cache = weightNoL2Cache_;
}

void GroupedNoQuantMatmulTiling::SetMatMulTiling()
{
    tilingData_.mmTilingData.isBias = static_cast<int32_t>(hasBias_);
    tilingData_.mmTilingData.M = m_;
    tilingData_.mmTilingData.N = n_;
    tilingData_.mmTilingData.Ka = k_;
    tilingData_.mmTilingData.Kb = k_;
    tilingData_.mmTilingData.singleCoreM = m_;
    tilingData_.mmTilingData.singleCoreN = baseN_;
    tilingData_.mmTilingData.singleCoreK = k_;
    tilingData_.mmTilingData.dbL0A = DB_SIZE;
    tilingData_.mmTilingData.dbL0B = DB_SIZE;
    tilingData_.mmTilingData.dbL0C = 1;
    tilingData_.mmTilingData.baseM = baseM_;
    tilingData_.mmTilingData.baseN = baseN_;
    tilingData_.mmTilingData.baseK = baseK_;
    tilingData_.mmTilingData.stepKa = stepKa_;
    tilingData_.mmTilingData.stepKb = stepKb_;
    tilingData_.mmTilingData.depthA1 = depthA1_;
    tilingData_.mmTilingData.depthB1 = depthB1_;
    tilingData_.mmTilingData.stepM = 1;
    tilingData_.mmTilingData.stepN = 1;
    tilingData_.mmTilingData.usedCoreNum = usedCoreNum_;
}

bool GroupedNoQuantMatmulTiling::SetCustomParam(gert::TilingContext *context)
{
    size_t *workspaces = context->GetWorkspaceSizes(1); // get second variable
    OP_CHECK_IF(workspaces == nullptr, OP_LOGE(context->GetNodeName(), "workspaces is nullptr."),
                return false); // check workspaces is not null
    workspaces[0] = 32U;       // 32: default workspace size

    context->SetBlockDim(usedCoreNum_);
    OP_CHECK_IF(context->GetRawTilingData() == nullptr, OP_LOGE(context->GetNodeName(), "RawTilingData is nullptr."),
                return false);
    errno_t ret = memcpy_s(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity(),
                           reinterpret_cast<void *>(&tilingData_), sizeof(tilingData_));
    if (ret != EOK) {
        OP_LOGE(context->GetNodeName(), "memcpy_s failed, ret = %d", ret);
        return false;
    }
    context->GetRawTilingData()->SetDataSize(sizeof(tilingData_));
    return true;
}

void GroupedNoQuantMatmulTiling::SetTilingKey(gert::TilingContext *context)
{
    tilingKeyBuilder_.gmmTrans = static_cast<uint8_t>(transposeX_) | (static_cast<uint8_t>(transposeWeight_) << 1);
    context->SetTilingKey(tilingKeyBuilder_.GenTilingKey());
}

bool GroupedNoQuantMatmulTiling::GMMGetTensorShapeSplitM(const gert::TilingContext *context, const gert::Shape xShape,
                                                         const gert::Shape wShape)
{
    if (isSingleX_ && isSingleWeight_ && isSingleY_) { // split M, s-s-s
        return SplitMSingleXSingleWeightSingleY(xShape, wShape);
    }
    if (!isSingleX_ && !isSingleWeight_ && isSingleY_) { // split M, m-m-s
        return SeparatedXSeparatedWeight(context);
    }
    if (isSingleX_ && !isSingleWeight_ && !isSingleY_) { // splitM, s-m-m
        return SplitMSingleXSeparatedWeight(context, xShape);
    }
    if (isSingleX_ && !isSingleWeight_ && isSingleY_) { // split M, s-m-s
        return SplitMSingleXSeparatedWeight(context, xShape);
    }
    if (!isSingleX_ && !isSingleWeight_ && !isSingleY_) { // split M, m-m-m
        return SeparatedXSeparatedWeight(context);
    }
    if (!isSingleX_ && isSingleWeight_) { // split M, m-s-m/m-s-s
        return SeparatedXSingleWeight(context, wShape);
    }
    OP_LOGE(context->GetNodeName(),
            "GMM_tiling: not support groupType_=%d, isSingleWeight_=%d, isSingleX_=%d, isSingleY_=%d", groupType_,
            isSingleWeight_, isSingleX_, isSingleY_);
    return false;
}

bool GroupedNoQuantMatmulTiling::GMMGetTensorShapeSplitK(const gert::TilingContext *context, const gert::Shape xShape,
                                                         const gert::Shape wShape)
{
    if (isSingleX_ && isSingleWeight_ && isSingleY_) { // splitK, s-s-s
        return SplitKSingleXSingleWeightSingleY(context, xShape, wShape);
    }
    OP_LOGE(context->GetNodeName(),
            "GMM_tiling: not support groupType_=%d, isSingleWeight_=%d, isSingleX_=%d, isSingleY_=%d", groupType_,
            isSingleWeight_, isSingleX_, isSingleY_);
    return false;
}

/** @brief split M：single-single-single(s-s-s)
 */
bool GroupedNoQuantMatmulTiling::SplitMSingleXSingleWeightSingleY(const gert::Shape xShape, const gert::Shape wShape)
{
    groupNum_ = static_cast<int32_t>(wShape.GetDim(0));
    int64_t m = xShape.GetDim(0);
    int64_t k = xShape.GetDim(xKDim_);
    int64_t n = wShape.GetDim(weightNDim_) * nzFactor_;
    m_ = static_cast<uint64_t>(m);
    k_ = static_cast<uint64_t>(k);
    n_ = static_cast<uint64_t>(n);
    kZero = kZero || (k == 0);
    weightNoL2Cache_ = true;
    return true;
}

/** @brief split M：single-multi-single(s-m-s)/single-multi-multi(s-m-m), share the same function.
 */
bool GroupedNoQuantMatmulTiling::SplitMSingleXSeparatedWeight(const gert::TilingContext *context,
                                                              const gert::Shape xShape)
{
    int64_t m = xShape.GetDim(0);
    int64_t k = xShape.GetDim(xKDim_);
    for (uint32_t i = 0; i < MAX_TENSOR; i++) {
        auto wTensor = context->GetDynamicInputTensor(INDEX_WEIGHT, i);
        if (wTensor == nullptr) {
            break;
        } // when x has multi tensors, xTensor is allowed to be empty
        auto wShape = wTensor->GetOriginShape();

        groupNum_ += 1U;
        int64_t n = wShape.GetDim(weightNDim_) * nzFactor_;
        n_ = std::max(n_, static_cast<uint64_t>(n));
    }
    m_ = static_cast<uint64_t>(m);
    k_ = static_cast<uint64_t>(k);
    kZero = kZero || (k == 0);
    weightNoL2Cache_ = isSingleY_;
    return true;
}

/** @brief split M：multi-multi-single(m-m-s); no split: multi-multi-multi(m-m-m), share the same function
 */
bool GroupedNoQuantMatmulTiling::SeparatedXSeparatedWeight(const gert::TilingContext *context)
{
    for (uint32_t i = 0; i < MAX_TENSOR; i++) {
        auto wTensor = context->GetDynamicInputTensor(INDEX_WEIGHT, i);
        auto xTensor = context->GetDynamicInputTensor(INDEX_X, i);
        if (wTensor == nullptr || xTensor == nullptr) {
            break;
        }
        auto wShape = wTensor->GetOriginShape();
        auto xShape = xTensor->GetStorageShape();
        groupNum_ += 1U;
        int64_t m = xShape.GetDim(0);
        int64_t k = xShape.GetDim(xKDim_);
        kZero = kZero || (k == 0);
        int64_t n = wShape.GetDim(weightNDim_) * nzFactor_;
        m_ = std::max(m_, static_cast<uint64_t>(m));
        k_ = std::max(k_, static_cast<uint64_t>(k));
        n_ = std::max(n_, static_cast<uint64_t>(n));
    }
    groupType_ = NO_SPLIT;
    return true;
}

/** @brief split M : multi-single-multi(m-s-m), share the same function
 */
bool GroupedNoQuantMatmulTiling::SeparatedXSingleWeight(const gert::TilingContext *context, const gert::Shape wShape)
{
    int64_t n = wShape.GetDim(weightNDim_) * nzFactor_;
    for (uint32_t i = 0; i < MAX_TENSOR; i++) {
        auto xTensor = context->GetDynamicInputTensor(INDEX_X, i);
        if (xTensor == nullptr) {
            break;
        } // when x has multi tensors, xTensor is allowed to be empty
        auto xShape = xTensor->GetStorageShape();
        groupNum_ += 1U;
        int64_t m = xShape.GetDim(0);
        int64_t k = xShape.GetDim(xKDim_);
        kZero = kZero || (k == 0);
        m_ = std::max(m_, static_cast<uint64_t>(m));
        k_ = std::max(k_, static_cast<uint64_t>(k));
    }
    n_ = static_cast<uint64_t>(n);
    groupType_ = NO_SPLIT;
    return true;
}

/** @brief split K single-single-single
 */
bool GroupedNoQuantMatmulTiling::SplitKSingleXSingleWeightSingleY(const gert::TilingContext *context,
                                                                  const gert::Shape xShape, const gert::Shape wShape)
{
    int64_t m = xShape.GetDim(1);
    int64_t k = xShape.GetDim(xKDim_);
    int64_t n = wShape.GetDim(weightNDim_) * nzFactor_;

    auto groupListTensor = context->GetDynamicInputTensor(INDEX_GROUPLIST, 0);
    if (groupListTensor == nullptr) {
        OP_LOGE(context->GetNodeName(), "groupListTensor is nullptr");
        return false;
    }
    gert::Shape groupListShape = groupListTensor->GetStorageShape();
    groupNum_ = static_cast<int32_t>(groupListShape.GetDim(0)); // 0: the first dim of groupList is groupNum
    m_ = static_cast<uint64_t>(m);
    n_ = static_cast<uint64_t>(n);
    k_ = static_cast<uint64_t>(k);
    kZero = kZero || (k == 0);
    return true;
}

void GroupedNoQuantMatmulTiling::PrintTilingResult(const gert::TilingContext *context)
{
    OP_LOGI(context->GetNodeName(),
            "GMM Tiling result: groupNum: %u, singleX: %u, singleWeight: %u, singleY: %u,"
            "groupType: %d, groupListType: %u, hasBias: %u, mTailCnt: %u, nTailCnt: %u, weightNoL2Cache: %u",
            tilingData_.gmmNoQuantParam.groupNum, tilingData_.gmmNoQuantParam.singleX,
            tilingData_.gmmNoQuantParam.singleWeight, tilingData_.gmmNoQuantParam.singleY,
            tilingData_.gmmNoQuantParam.groupType, tilingData_.gmmNoQuantParam.groupListType,
            tilingData_.gmmNoQuantParam.hasBias, tilingData_.gmmNoQuantParam.mTailCnt,
            tilingData_.gmmNoQuantParam.nTailCnt, tilingData_.gmmNoQuantParam.weightNoL2Cache);

    OP_LOGI(context->GetNodeName(),
            "GMM MatMul Tiling result: usedCoreNum: %d, baseM: %d, baseN: %d, baseK: %d, stepKa: %d,"
            "stepKb: %d, depthA1: %d, depthB1: %d",
            tilingData_.mmTilingData.usedCoreNum, tilingData_.mmTilingData.baseM, tilingData_.mmTilingData.baseN,
            tilingData_.mmTilingData.baseK, tilingData_.mmTilingData.stepKa, tilingData_.mmTilingData.stepKb,
            tilingData_.mmTilingData.depthA1, tilingData_.mmTilingData.depthB1);
}

uint64_t TilingKeyBuilder::GenTilingKey()
{
    uint64_t transInfo = static_cast<uint64_t>(this->gmmTrans);
    bool atrans_ = (transInfo == static_cast<uint64_t>(GmmTrans::ATrans)) ||
                   (transInfo == static_cast<uint64_t>(GmmTrans::ABTrans));
    bool btrans_ = (transInfo == static_cast<uint64_t>(GmmTrans::BTrans)) ||
                   (transInfo == static_cast<uint64_t>(GmmTrans::ABTrans));
    return GET_TPL_TILING_KEY(static_cast<uint64_t>(btrans_), static_cast<uint64_t>(atrans_));
}
} // namespace optiling
