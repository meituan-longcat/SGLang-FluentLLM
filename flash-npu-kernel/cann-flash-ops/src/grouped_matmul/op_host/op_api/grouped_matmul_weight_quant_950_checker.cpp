/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "grouped_matmul_weight_quant_950_checker.h"

using namespace gmm;

namespace {
static const std::unordered_set<DataType> FP8_SUPPORT_SET = {ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2, ge::DT_HIFLOAT8};
static constexpr int64_t X_Y_SEPARATED = 0L;  // x,y no split
static constexpr int64_t Y_SEPARATED = 1L;    // x split
static constexpr int64_t X_SEPARATED = 2L;    // y split
static constexpr int64_t NO_SEPARATED = 3L;   // x,y split
static constexpr int64_t MAX_GROUP_LIST_SIZE_ARRAY = 128L;
}  // namespace

bool AclnnGroupedMatmulWeightQuantDAV3510Checker::IsA16MxFp4NZ() const
{
    return (xDtype_ == ge::DT_FLOAT16 || xDtype_ == ge::DT_BF16) && weightDtype_ == ge::DT_FLOAT4_E2M1;
}

bool AclnnGroupedMatmulWeightQuantDAV3510Checker::IsMxA8W4NZ() const
{
    return xDtype_ == ge::DT_FLOAT8_E4M3FN && weightDtype_ == ge::DT_FLOAT4_E2M1;
}

bool AclnnGroupedMatmulWeightQuantDAV3510Checker::IsA16W8ND() const
{
    return (xDtype_ == ge::DT_FLOAT16 || xDtype_ == ge::DT_BF16) && weightDtype_ == ge::DT_INT8;
}

bool AclnnGroupedMatmulWeightQuantDAV3510Checker::IsA16F8ND() const
{
    return (xDtype_ == ge::DT_FLOAT16 || xDtype_ == ge::DT_BF16) &&
           FP8_SUPPORT_SET.find(weightDtype_) != FP8_SUPPORT_SET.end();
}

bool AclnnGroupedMatmulWeightQuantDAV3510Checker::IsS8S4NZ() const
{
    return xDtype_ == ge::DT_INT8 && weightDtype_ == ge::DT_INT4;
}

bool AclnnGroupedMatmulWeightQuantDAV3510Checker::IsA16W4() const
{
    return (xDtype_ == ge::DT_FLOAT16 || xDtype_ == ge::DT_BF16) && weightDtype_ == ge::DT_INT4;
}

aclnnStatus AclnnGroupedMatmulWeightQuantDAV3510Checker::CheckTensorNotNull(size_t idx) const
{
    CHECK_RET(CheckTensorNotNullPtr(gmmParams_.x, idx, "x") == ACLNN_SUCCESS, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(CheckTensorNotNullPtr(gmmParams_.weight, idx, "weight") == ACLNN_SUCCESS, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(CheckTensorNotNullPtr(gmmParams_.y, idx, "y") == ACLNN_SUCCESS, ACLNN_ERR_PARAM_NULLPTR);
    if (gmmParams_.antiquantScaleOptional != nullptr) {
        CHECK_RET(CheckTensorNotNullPtr(gmmParams_.antiquantScaleOptional, idx, "antiquantScale") == ACLNN_SUCCESS,
                  ACLNN_ERR_PARAM_NULLPTR);
    }

    if (gmmParams_.antiquantOffsetOptional != nullptr) {
        CHECK_RET(CheckTensorNotNullPtr(gmmParams_.antiquantOffsetOptional, idx, "antiquantOffset") == ACLNN_SUCCESS,
                  ACLNN_ERR_PARAM_NULLPTR);
    }

    if (gmmParams_.biasOptional != nullptr) {
        CHECK_RET(CheckTensorNotNullPtr(gmmParams_.biasOptional, idx, "bias") == ACLNN_SUCCESS,
                  ACLNN_ERR_PARAM_NULLPTR);
    }

    return ACLNN_SUCCESS;
}

aclnnStatus AclnnGroupedMatmulWeightQuantDAV3510Checker::CheckTensorNotNullPtr(const aclTensorList *tensorList,
                                                                               size_t idx,
                                                                               const std::string &tensorType) const
{
    const aclTensor *tensor = (*tensorList)[idx];
    CHECK_COND(tensor != nullptr, ACLNN_ERR_PARAM_NULLPTR, "%s[%lu] is null, which is not supported.",
               tensorType.c_str(), idx);
    return ACLNN_SUCCESS;
}

aclnnStatus AclnnGroupedMatmulWeightQuantDAV3510Checker::CheckTensorDtype(const aclTensorList *tensorList,
                                                                          const DataType &tensorDtype, size_t idx,
                                                                          const std::string &tensorType) const
{
    const aclTensor *tensor = (*tensorList)[idx];
    CHECK_COND(
        tensor->GetDataType() == tensorDtype, ACLNN_ERR_PARAM_INVALID,
        "The dtype of each tensor in %s tensor list must be consistent. %s[%lu]'s dtype [%s] is different from the "
        "expected dtype [%s]. ",
        tensorType.c_str(), tensorType.c_str(), idx, op::ToString(tensor->GetDataType()).GetString(),
        op::ToString(tensorDtype).GetString());
    return ACLNN_SUCCESS;
}

aclnnStatus AclnnGroupedMatmulWeightQuantDAV3510Checker::CheckTensorShape(const aclTensorList *tensorList, size_t idx,
                                                                          const std::string &tensorType) const
{
    // 校验bias、antiquantScale和antiquantOffset的dim和shape
    auto tensorShape = (*tensorList)[idx]->GetViewShape();
    auto wShape = (*gmmParams_.weight)[idx]->GetViewShape();

    size_t tensorDimNum = tensorShape.GetDimNum();
    size_t expectedDimNum = gmmParams_.groupType == SPLIT_M ? 2 : 1;  // 单单单场景默认维度为2，多多多场景默认维度为1

    if ((IsA16MxFp4NZ() || IsS8S4NZ()) && tensorType.find("antiquant") != std::string::npos) {
        expectedDimNum = 3; // Mx / PerGroup量化，仅支持antiquantSacle/antiquantOffset维度为3
    } else if (IsMxA8W4NZ()) {
        if (tensorType.find("antiquant") != std::string::npos) {
            expectedDimNum = 4; // MxA8W4场景，antiquantScale维度为4
        } else if (tensorType.find("token") != std::string::npos) {
            expectedDimNum = 3; // MxA8W4场景，perTokenScale维度为3
        }
    }

    CHECK_COND(tensorDimNum == expectedDimNum, ACLNN_ERR_PARAM_INVALID, "%s Dim must be [%zu], but now is [%zu].",
               tensorType.c_str(), expectedDimNum, tensorDimNum);

    if (gmmParams_.groupType == SPLIT_M) {
        // Check the first dimension, batch size must match the group size.
        uint64_t groupNum = wShape.GetDim(0);
        uint64_t batchSize = tensorShape.GetDim(0);
        CHECK_COND(batchSize == groupNum, ACLNN_ERR_PARAM_INVALID,
                   "%s batch size[%llu] should be equal with groupList length[%llu].", tensorType.c_str(), batchSize,
                   groupNum);
    }

    // Check tensor’s Ndim must match weight’s Ndim.
    uint64_t weightNDimIdx = wShape.GetDimNum() - 1;
    int64_t weightNDimValue = wShape.GetDim(weightNDimIdx);
    int64_t tensorNDimValue;
    if (IsMxA8W4NZ() && tensorType.find("antiquant") != std::string::npos) { // viewShape,所以是-2
        tensorNDimValue = tensorShape.GetDim(tensorDimNum - 2);
    } else {
        tensorNDimValue = tensorShape.GetDim(tensorDimNum - 1);
    }
    CHECK_COND(tensorNDimValue == weightNDimValue, ACLNN_ERR_PARAM_INVALID,
               "NDim[%ld] of %s should be equal to NDim[%ld] of weight.", tensorNDimValue, tensorType.c_str(),
               weightNDimValue);

    return ACLNN_SUCCESS;
}

aclnnStatus AclnnGroupedMatmulWeightQuantDAV3510Checker::CheckWeightInnerAxisEven(size_t idx) const
{
    if (weightDtype_ == ge::DT_INT4) {
        auto wShape = (*gmmParams_.weight)[idx]->GetViewShape();
        bool isTrans = IsTransposeLastTwoDims((*gmmParams_.weight)[idx]);
        // -2：weight的倒数第二跟轴;-1：表示weight的倒数第一跟轴
        uint64_t weightLastDimIdx = isTrans ? wShape.GetDimNum() - 2 : wShape.GetDimNum() - 1;
        // 2：对内轴的维度是否位奇数
        size_t evenDivider = 2;
        CHECK_COND(wShape.GetDim(weightLastDimIdx) % evenDivider == 0, ACLNN_ERR_PARAM_INVALID,
                   "Last dimension of each weight tensor must be even.");
    }
    return ACLNN_SUCCESS;
}

aclnnStatus AclnnGroupedMatmulWeightQuantDAV3510Checker::CheckAntiQuantParams() const
{
    // 单单单场景antiquantScale为[(g, n)]或[(g, k/gs, n)]，g不为0所以一定不为nullptr
    // 多多多场景antiQuantScale可能为[(0)]，此时会被aclnn_grouped_matmul.cpp中的CheckOptionalTensorListEmpty置为nullptr
    auto w0Shape = (*gmmParams_.weight)[0]->GetViewShape();
    int64_t w0NDim = w0Shape.GetDim(w0Shape.GetDimNum() - 1);
    bool antiquantScaleNullFlag =
        gmmParams_.groupType == NO_SPLIT && gmmParams_.weight->Size() == 1 && w0NDim == 0;
    if (!antiquantScaleNullFlag) {
        CHECK_COND(gmmParams_.antiquantScaleOptional != nullptr, ACLNN_ERR_PARAM_NULLPTR,
                   "AntiquantScale must not be nullptr in antiquant, but now is nullptr.");
    }

    if (IsA16F8ND() || IsA16MxFp4NZ() || IsMxA8W4NZ() || IsS8S4NZ()) {
        CHECK_COND(gmmParams_.antiquantOffsetOptional == nullptr, ACLNN_ERR_PARAM_INVALID,
                   "In weight quant case, antiquantOffsetOptional is not supported when weightDtype is fp8/fp4 or "
                   "xDtype-weightDtype is int8-int4.");
    }

    return ACLNN_SUCCESS;
}

aclnnStatus AclnnGroupedMatmulWeightQuantDAV3510Checker::CheckQuantParams() const
{
    if (!IsS8S4NZ()) {
        CHECK_COND(gmmParams_.scaleOptional == nullptr, ACLNN_ERR_PARAM_INVALID,
                   "In weight quant case, scale must be null when xDtype-weightDtype is not int8-int4.");
    } else {
        CHECK_COND(gmmParams_.scaleOptional != nullptr, ACLNN_ERR_PARAM_INVALID,
                   "In weight quant case, scale must not be null when xDtype-weightDtype is int8-int4.");
    }

    CHECK_COND(gmmParams_.offsetOptional == nullptr, ACLNN_ERR_PARAM_INVALID,
               "In WeightQuant case, offset must be null.");

    if (!IsMxA8W4NZ() && !IsS8S4NZ()) {
        CHECK_COND(gmmParams_.perTokenScaleOptional == nullptr, ACLNN_ERR_PARAM_INVALID,
                   "In WeightQuant case, perTokenScale must be null when xDtype-weightDtype is not "
                   "float8_e4m3fn-float4_e2m1 or int8-int4.");
    } else {
        CHECK_COND(gmmParams_.perTokenScaleOptional != nullptr, ACLNN_ERR_PARAM_INVALID,
                   "In WeightQuant case, perTokenScale must not be null when xDtype-weightDtype is "
                   "float8_e4m3fn-float4_e2m1 or int8-int4.");
    }
    return ACLNN_SUCCESS;
}

aclnnStatus AclnnGroupedMatmulWeightQuantDAV3510Checker::CheckDimNumAndFormat(size_t idx) const
{
    // check format
    CHECK_COND(!op::IsPrivateFormat((*gmmParams_.x)[idx]->GetStorageFormat()), ACLNN_ERR_PARAM_INVALID,
               "The format of x is invalid. It should only be ND.");
    CHECK_COND(!op::IsPrivateFormat((*gmmParams_.y)[idx]->GetStorageFormat()), ACLNN_ERR_PARAM_INVALID,
               "The format of y is invalid. It should only be ND.");

    if (IsA16W8ND() || IsA16F8ND() || IsA16W4()) {
        CHECK_COND(!op::IsPrivateFormat((*gmmParams_.weight)[idx]->GetStorageFormat()), ACLNN_ERR_PARAM_INVALID,
                   "The format of weight is invalid. It should only be ND for GMM when xDtype-weightDtype is "
                   "bf16/fp16-int8, bf16/fp16-float8_e4m3fn/float8_e5m2/hif8 or bf16/fp16-int4. ");
    } else {
        CHECK_COND(op::IsPrivateFormat((*gmmParams_.weight)[idx]->GetStorageFormat()), ACLNN_ERR_PARAM_INVALID,
                   "The format of weight is invalid. It should only be NZ for GMM when xDtype-weightDtype is "
                   "bf16/fp16-float4_e2m1/float4_e1m2, float8-float4_e2m1 or int8-int4. ");
    }

    // check dimNum
    size_t xDimNum = (*gmmParams_.x)[idx]->GetViewShape().GetDimNum();
    size_t weightDimNum = (*gmmParams_.weight)[idx]->GetViewShape().GetDimNum();
    size_t yDimNum = (*gmmParams_.y)[idx]->GetViewShape().GetDimNum();

    if (gmmParams_.groupType == NO_SPLIT) {
        CHECK_COND(xDimNum <= MAX_FM_DIM && xDimNum >= MIN_FM_DIM, ACLNN_ERR_PARAM_INVALID,
                   "x[%lu] dimNum is [%lu], but only support %s-%s.", idx, xDimNum, MIN_FM_DIM, MAX_FM_DIM);
        CHECK_COND(weightDimNum == MIN_FM_DIM, ACLNN_ERR_PARAM_INVALID,
                   "weight[%lu] dimNum is %lu, but only support 2 when weight separated.", idx, weightDimNum);
    } else {
        CHECK_COND(xDimNum == MIN_FM_DIM, ACLNN_ERR_PARAM_INVALID,
                   "x[%lu] dimNum should be 2 in this case, but now is [%lu].", idx, xDimNum);
        CHECK_COND(weightDimNum == SPLIT_M_SINGLE_WEIGHT_DIM, ACLNN_ERR_PARAM_INVALID,
                   "weight[%lu] dimNum should be 3 in this case, but now is [%lu].", idx, weightDimNum);
    }

    CHECK_COND(xDimNum == yDimNum, ACLNN_ERR_PARAM_INVALID, "y[%lu] dimNum %lu should be equal to x[%lu] DimNum %lu.",
               idx, yDimNum, idx, xDimNum);

    return ACLNN_SUCCESS;
}

aclnnStatus AclnnGroupedMatmulWeightQuantDAV3510Checker::CheckTransposeStatus() const
{
    CHECK_COND(!gmmParams_.transposeX, ACLNN_ERR_PARAM_INVALID, "In weight quant case, x must not be transposed.");

    if (gmmParams_.groupType == NO_SPLIT) {
        CHECK_COND(!(gmmParams_.apiVersion == gmm::GMMApiVersion::V1 && gmmParams_.transposeWeight),
                   ACLNN_ERR_PARAM_INVALID,
                   "For aclnnGroupedMatmul V1, when x, weight and y are all separated, weight can not be transposed.");
    }

    if (IsA16F8ND() || IsMxA8W4NZ()) {
        CHECK_COND(gmmParams_.transposeWeight, ACLNN_ERR_PARAM_INVALID,
                   "In weight quant case fp16/bf16-int8, fp16/bf16-fp8/hif8 and fp8_e4m3-fp4_e2m1, weight must be "
                   "transposed.");
    } else if (IsA16MxFp4NZ() || IsS8S4NZ()) {
        CHECK_COND(!gmmParams_.transposeWeight, ACLNN_ERR_PARAM_INVALID,
                   "In weight quant case fp16/bf16-fp4_e2m1 and int8-int4, weight must be not transposed.");
    }
    return ACLNN_SUCCESS;
}

aclnnStatus AclnnGroupedMatmulWeightQuantDAV3510Checker::CheckDimValue(size_t idx) const
{
    // 校验x, weight, y的各轴匹配
    size_t xDimNum = (*gmmParams_.x)[idx]->GetViewShape().GetDimNum();
    // 验证到倒数第二维，x和y除最后一维其他必须相等
    for (size_t dimIdx = 0; dimIdx < xDimNum - 1; dimIdx++) {
        size_t xDimValue = (*gmmParams_.x)[idx]->GetViewShape().GetDim(dimIdx);
        size_t yDimValue = (*gmmParams_.y)[idx]->GetViewShape().GetDim(dimIdx);
        CHECK_COND(xDimValue == yDimValue, ACLNN_ERR_PARAM_INVALID,
                   "y[%zu] dim %zu value %zu should equal to x[%zu] dim %zu value %zu.", idx, dimIdx, xDimValue, idx,
                   dimIdx, yDimValue);
    }

    size_t xKDim = (*gmmParams_.x)[idx]->GetViewShape().GetDim(xDimNum - 1);
    auto weightNIdx = (*gmmParams_.weight)[idx]->GetViewShape().GetDimNum() - 1;
    auto weightKIdx = (*gmmParams_.weight)[idx]->GetViewShape().GetDimNum() - 2;
    size_t weightKDim = (*gmmParams_.weight)[idx]->GetViewShape().GetDim(weightKIdx);
    size_t weightNDim = (*gmmParams_.weight)[idx]->GetViewShape().GetDim(weightNIdx);

    CHECK_COND(xKDim == weightKDim, ACLNN_ERR_PARAM_INVALID,
               "x[%zu] dim k value %zu should equal to weight[%zu] dim k value %zu.", idx, xKDim, idx, weightKDim);
    // check y[n] = weight[n]
    size_t yNDim = (*gmmParams_.y)[idx]->GetViewShape().GetDim(xDimNum - 1);
    CHECK_COND(yNDim == weightNDim, ACLNN_ERR_PARAM_INVALID,
               "y[%zu] dim n value %zu should equal to weight[%zu] dim n value %zu.", idx, yNDim, idx, weightNDim);

    if (IsA16MxFp4NZ() || IsMxA8W4NZ() || IsS8S4NZ()) {
        CHECK_COND((weightNDim % N_K_ALIGN_VALUE_WEIGHT_QUANT_4BIT == 0) &&
                       (weightKDim % N_K_ALIGN_VALUE_WEIGHT_QUANT_4BIT == 0),
                   ACLNN_ERR_PARAM_INVALID,
                   "The value of dim n, k should be an integer multiple of [%lld], but actual n is [%zu], k is [%zu].",
                   N_K_ALIGN_VALUE_WEIGHT_QUANT_4BIT, weightNDim, weightKDim);
    }

    return ACLNN_SUCCESS;
}

aclnnStatus AclnnGroupedMatmulWeightQuantDAV3510Checker::CheckV1GroupList(size_t idx) const
{
    // 多多多 V1接口校验groupList
    if (gmmParams_.groupType != NO_SPLIT || gmmParams_.apiVersion != GMMApiVersion::V1) {
        return ACLNN_SUCCESS;
    }

    size_t xDimNum = (*gmmParams_.x)[idx]->GetViewShape().GetDimNum();
    if (xDimNum > MIN_FM_DIM) {
        CHECK_COND(gmmParams_.groupListOptional == nullptr, ACLNN_ERR_PARAM_INVALID,
                   "groupListOptional should be nullptr when x, y both separated and dim num larger than 2.");
    }

    if (xDimNum == MIN_FM_DIM && gmmParams_.groupListOptional != nullptr) {
        int64_t xMDimValue = (*gmmParams_.x)[idx]->GetViewShape().GetDim(0);
        int64_t preGroupList = idx == 0 ? 0 : (*gmmParams_.groupListOptional)[idx - 1];
        int64_t mValueGroupList = (*gmmParams_.groupListOptional)[idx] - preGroupList;
        std::string errorMessage = idx == 0 ? "groupListOptional[0]"
                                            : "groupListOptional[" + std::to_string(idx) + "] - groupListOptional[" +
                                                  std::to_string(idx - 1) + "]";
        CHECK_COND(xMDimValue == mValueGroupList, ACLNN_ERR_PARAM_INVALID,
                   "x[%lu] dim 0 value %ld should be equal to %s %ld.", idx, xMDimValue, errorMessage.c_str(),
                   mValueGroupList);
    }

    return ACLNN_SUCCESS;
}

aclnnStatus AclnnGroupedMatmulWeightQuantDAV3510Checker::CheckYDtype() const
{
    if (IsMxA8W4NZ() || IsS8S4NZ()) {
        CHECK_COND(yDtype_ == DataType::DT_BF16 || yDtype_ == DataType::DT_FLOAT16, ACLNN_ERR_PARAM_INVALID,
                   "When xDtype-weightDtype is float8_e4m3fn-float4_e2m1 or int8-int4, y dtype can only be float16 or "
                   "bfloat16 but the actual y dtype is [%s]",
                   op::ToString(yDtype_).GetString());
    } else {
        CHECK_COND(
            yDtype_ == xDtype_, ACLNN_ERR_PARAM_INVALID,
            "In weight quant case, y dtype should be equal to x dtype but the actual y dtype is [%s], x dtype is [%s].",
            op::ToString(yDtype_).GetString(), op::ToString(xDtype_).GetString());
    }

    return ACLNN_SUCCESS;
}

aclnnStatus AclnnGroupedMatmulWeightQuantDAV3510Checker::CheckBiasDtype()
{
    if (gmmParams_.biasOptional != nullptr) {
        biasDtype_ = (*gmmParams_.biasOptional)[0]->GetDataType();
        if (xDtype_ == DataType::DT_BF16) {
            CHECK_COND(
                biasDtype_ == DataType::DT_BF16 || biasDtype_ == DataType::DT_FLOAT, ACLNN_ERR_PARAM_INVALID,
                "When x dtype is bfloat16, the bias dtype should be bfloat16 or float32, but the actual dtype is [%s].",
                op::ToString(biasDtype_).GetString());
        } else if (xDtype_ == DataType::DT_FLOAT16) {
            CHECK_COND(biasDtype_ == DataType::DT_FLOAT16, ACLNN_ERR_PARAM_INVALID,
                       "When x dtype is float16, the bias dtype should be float16, but the actual dtype is [%s].",
                       op::ToString(biasDtype_).GetString());
        } else if (IsMxA8W4NZ()) {
            CHECK_COND(biasDtype_ == yDtype_, ACLNN_ERR_PARAM_INVALID,
                       "When xDtype-weightDtype is fp8_e4m3fn-fp4_e2m1, the bias dtype must be equal to y dtype, but the "
                       "actual bias dtype is [%s], y dtype is [%s].",
                       op::ToString(biasDtype_).GetString(), op::ToString(yDtype_).GetString());
        } else if (IsS8S4NZ()) {
            CHECK_COND(
                biasDtype_ == DataType::DT_FLOAT, ACLNN_ERR_PARAM_INVALID,
                "When xDtype-weightDtype is int8-int4, the bias dtype should be float32, but the actual dtype is [%s].",
                op::ToString(biasDtype_).GetString());
        }
    }

    return ACLNN_SUCCESS;
}

aclnnStatus AclnnGroupedMatmulWeightQuantDAV3510Checker::CheckAntiQuantDtype(size_t idx) const
{
    if (gmmParams_.antiquantScaleOptional != nullptr) {
        if (IsA16W8ND() || IsA16F8ND() || IsA16W4()) {
            CHECK_RET(CheckTensorDtype(gmmParams_.antiquantScaleOptional, xDtype_, idx, "antiquantScale") ==
                          ACLNN_SUCCESS,
                      ACLNN_ERR_PARAM_INVALID);
        } else if (IsA16MxFp4NZ() || IsMxA8W4NZ()) {
            CHECK_RET(CheckTensorDtype(gmmParams_.antiquantScaleOptional, ge::DT_FLOAT8_E8M0, idx, "antiquantScale") ==
                          ACLNN_SUCCESS,
                      ACLNN_ERR_PARAM_INVALID);
        } else {
            // S8S4的antiquantScale类型为FP16
            CHECK_RET(CheckTensorDtype(gmmParams_.antiquantScaleOptional, DataType::DT_FLOAT16, idx,
                                       "antiquantScale") == ACLNN_SUCCESS,
                      ACLNN_ERR_PARAM_INVALID);
        }
    }

    if (gmmParams_.antiquantOffsetOptional != nullptr) {
        // 当前有antiquantOffset的数据流，antiquantOffset的数据类型均和x一致
        CHECK_RET(
            CheckTensorDtype(gmmParams_.antiquantOffsetOptional, xDtype_, idx, "antiquantOffset") == ACLNN_SUCCESS,
            ACLNN_ERR_PARAM_INVALID);
    }
    return ACLNN_SUCCESS;
}

aclnnStatus AclnnGroupedMatmulWeightQuantDAV3510Checker::CheckAntiQuantShape(size_t idx) const
{
    if (gmmParams_.antiquantScaleOptional != nullptr) {
        CHECK_RET(CheckTensorShape(gmmParams_.antiquantScaleOptional, idx, "antiquantScale") == ACLNN_SUCCESS,
                  ACLNN_ERR_PARAM_INVALID);
    }

    if (gmmParams_.antiquantOffsetOptional != nullptr) {
        CHECK_RET(CheckTensorShape(gmmParams_.antiquantOffsetOptional, idx, "antiquantOffset") == ACLNN_SUCCESS,
                  ACLNN_ERR_PARAM_INVALID);
    }

    return ACLNN_SUCCESS;
}

aclnnStatus AclnnGroupedMatmulWeightQuantDAV3510Checker::CheckQuantDtype() const
{
    // check pertokenScaleDtype for MxA8W4
    if (IsMxA8W4NZ()) {
        auto pertokenScaleDtype = (*gmmParams_.perTokenScaleOptional)[0]->GetDataType();
        CHECK_COND(pertokenScaleDtype == ge::DT_FLOAT8_E8M0, ACLNN_ERR_PARAM_INVALID,
                   "PertokenScale dtype must be float8_e8m0 when xDtype-weightDtype is float8_e4m3fn-float4_e2m1.");
    }

    if (IsS8S4NZ()) {
        auto pertokenScaleDtype = (*gmmParams_.perTokenScaleOptional)[0]->GetDataType();
        CHECK_COND(pertokenScaleDtype == ge::DT_FLOAT, ACLNN_ERR_PARAM_INVALID,
                   "PertokenScale dtype must be DT_FLOAT when xDtype-weightDtype is int8-int4.");
        auto scaleDtype = (*gmmParams_.scaleOptional)[0]->GetDataType();
        CHECK_COND(scaleDtype == ge::DT_FLOAT, ACLNN_ERR_PARAM_INVALID,
                   "Scale dtype must be DT_FLOAT when xDtype-weightDtype is int8-int4.");
    }
    return ACLNN_SUCCESS;
}

aclnnStatus AclnnGroupedMatmulWeightQuantDAV3510Checker::CheckScaleAndPerTokenScaleShape() const
{
    if (IsMxA8W4NZ()) {
        // check pertokenscale shape for MxA8W4
        auto perTokenScaleShape = (*gmmParams_.perTokenScaleOptional)[0]->GetViewShape();
        auto perTokenScaleShapeDimNum = perTokenScaleShape.GetDimNum();
        // MxA8W4NZ仅支持perTokenScale维度为3
        size_t perTokenScaleSupportDimNum = 3;
        CHECK_COND(perTokenScaleShapeDimNum == perTokenScaleSupportDimNum, ACLNN_ERR_PARAM_INVALID,
                   "The dim of pertokenscale must be 3!");
        auto xShape = (*gmmParams_.x)[0]->GetViewShape();
        auto perTokenScaleShapeMDim = perTokenScaleShape.GetDim(0);
        auto perTokenScaleShapeKDim = perTokenScaleShape.GetDim(1);
        auto xShapeKDim = xShape.GetDim(1);
        auto xShapeMDim = xShape.GetDim(0);
        CHECK_COND(xShapeMDim == perTokenScaleShapeMDim, ACLNN_ERR_PARAM_INVALID,
                   "The first dim of pertokenscale must be equal to the first dim of x!");
        // 64含义：pertokenscale的shape应为(m,k/64,2)
        CHECK_COND(xShapeKDim == perTokenScaleShapeKDim * 64, ACLNN_ERR_PARAM_INVALID,
                   "The second dim of x must be 64 times the second dim of pertokenscale!");
    } else if (IsS8S4NZ()) {
        auto perTokenScaleShape = (*gmmParams_.perTokenScaleOptional)[0]->GetViewShape();
        auto perTokenScaleShapeDimNum = perTokenScaleShape.GetDimNum();
        CHECK_COND(perTokenScaleShapeDimNum == 1, ACLNN_ERR_PARAM_INVALID,
                   "The dim of pertokenscale must be 1!");  // 仅支持perTokenScale维度为1
        auto xShape = (*gmmParams_.x)[0]->GetViewShape();
        auto weightShape = (*gmmParams_.weight)[0]->GetViewShape();
        auto xShapeMDim = xShape.GetDim(0);
        auto perTokenScaleShapeMDim = perTokenScaleShape.GetDim(0);
        CHECK_COND(xShapeMDim == perTokenScaleShapeMDim, ACLNN_ERR_PARAM_INVALID,
                   "The first dim of pertokenscale must be equal to the first dim of x!");
        auto scaleShape = (*gmmParams_.scaleOptional)[0]->GetViewShape();
        // S8S4中scale的shape维度仅支持2
        CHECK_COND(scaleShape.GetDimNum() == 2, ACLNN_ERR_PARAM_INVALID, "The dim of scaleShape must be 2!");
        CHECK_COND(scaleShape.GetDim(0) == weightShape.GetDim(0), ACLNN_ERR_PARAM_INVALID,
                   "The dim0 of scaleShape must be g!");
        // scale的index为1的维度需要为n，即weight的index为2时的数值
        CHECK_COND(scaleShape.GetDim(1) == weightShape.GetDim(2), ACLNN_ERR_PARAM_INVALID,
                   "The dim1 of scaleShape must be n!");
    }
    return ACLNN_SUCCESS;
}

aclnnStatus AclnnGroupedMatmulWeightQuantDAV3510Checker::CheckUnsupportedApi() const
{
    if (IsA16W8ND()) {
        if (gmmParams_.groupType == NO_SPLIT) {
            CHECK_COND(gmmParams_.apiVersion != GMMApiVersion::WeightNz, ACLNN_ERR_PARAM_INVALID,
                       "When xDtype-weightDtype is fp16/bf16-int8, only aclnnGroupedMatmul V1/V2/V3/V4/V5 support "
                       "multi-multi-multi scenario.");
        } else {
            CHECK_COND(gmmParams_.apiVersion != GMMApiVersion::WeightNz && gmmParams_.apiVersion != GMMApiVersion::V1,
                       ACLNN_ERR_PARAM_INVALID,
                       "When xDtype-weightDtype is fp16/bf16-int8, only aclnnGroupedMatmul V2/V3/V4/V5 support "
                       "single-single-single scenario.");
        }
    } else if (IsA16F8ND() || IsA16W4()) {
        CHECK_COND(gmmParams_.apiVersion == GMMApiVersion::V5 || gmmParams_.apiVersion == GMMApiVersion::V4,
                   ACLNN_ERR_PARAM_INVALID,
                   "Only AclnnGroupedMatmulV4/V5 support fp16/bf16-fp8/hif8/int4 for xDtype-weightDtype.");
    } else if (IsA16MxFp4NZ() || IsMxA8W4NZ() || IsS8S4NZ()) {
        CHECK_COND(gmmParams_.apiVersion == GMMApiVersion::WeightNz, ACLNN_ERR_PARAM_INVALID,
                   "Only AclnnGroupedMatmulNz supports fp16/bf16-fp4_e2m1, fp8_e4m3fn-fp4_e2m1 and int8-int4 for "
                   "xDtype-weightDtype.");
    } else {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Weight quant case with x dtype [%s] and weight dtype [%s] is not supported.",
                op::ToString(xDtype_).GetString(), op::ToString(weightDtype_).GetString());
        return ACLNN_ERR_PARAM_INVALID;
    }

    return ACLNN_SUCCESS;
}

aclnnStatus AclnnGroupedMatmulWeightQuantDAV3510Checker::CheckGroupSize(size_t idx) const
{
    if (!(IsA16MxFp4NZ() || IsMxA8W4NZ() || IsS8S4NZ())) {
        return ACLNN_SUCCESS;
    }

    auto antiquantScaleShape = (*gmmParams_.antiquantScaleOptional)[idx]->GetViewShape();
    auto weightShape = (*gmmParams_.weight)[idx]->GetViewShape();
    auto antiquantScaleDimNum = antiquantScaleShape.GetDimNum();
    int64_t groupSize = 0;

    // 2含义: (g,k,n)的k轴索引
    int64_t kSize = weightShape.GetDim(weightShape.GetDimNum() - 2);
    // 2含义: (g,k/groupSize,n)的k轴索引
    int64_t groupNum = IsMxA8W4NZ() ? antiquantScaleShape.GetDim(antiquantScaleDimNum - 3) * 2:
                                      antiquantScaleShape.GetDim(antiquantScaleDimNum - 2);
    CHECK_COND(groupNum > 0, ACLNN_ERR_PARAM_INVALID,
               "GroupNum must be greater than 0, but the actual groupNum is [%ld].", groupNum);
    CHECK_COND(kSize % groupNum == 0, ACLNN_ERR_PARAM_INVALID,
               "kSize must be a multiple of groupNum, but the actual kSize is [%ld], groupNum is [%ld].", kSize,
               groupNum);
    groupSize = kSize / groupNum;
    if (IsMxA8W4NZ()) {
        // 2：MxA8W4NZ的antiquantScaleViewShape: (g, k / groupSIze / 2, n, 2)
        groupSize = kSize / groupNum;
    }
    // 当前伪量化仅支持groupsize为32整数倍
    if (IsS8S4NZ()) {
        // 伪量化S8S4场景支持groupsize为128/192/256/512
        CHECK_COND(groupSize == 128 || groupSize == 256 || groupSize == 512 || groupSize == 192,
                   ACLNN_ERR_PARAM_INVALID, "GroupSize must be 128/192/256/512, but the actual groupSize is [%ld].",
                   groupSize);
    } else {
        // 当前伪量化非S8S4仅支持groupsize为32
        CHECK_COND(groupSize == 32, ACLNN_ERR_PARAM_INVALID, "GroupSize must be 32, but the actual groupSize is [%ld].",
                   groupSize);
    }
    return ACLNN_SUCCESS;
}

aclnnStatus AclnnGroupedMatmulWeightQuantDAV3510Checker::CheckGroupTypeScenario() const
{
    std::string errorMessage;

    // groupType校验
    // V1接口没有groupType字段，是在aclnnGroupedMatmulGetWorkspaceSize赋值的，只会出现0/-1，不会出现2
    if (IsA16W8ND() || IsA16W4()) {
        CHECK_COND(gmmParams_.groupType == NO_SPLIT || gmmParams_.groupType == SPLIT_M, ACLNN_ERR_PARAM_INVALID,
                   "When xDtype-weightDtype is fp16/bf16-int8 or fp16/bf16-int4, GMM only support groupType 0 (split "
                   "M) or groupType -1 (no split), but the actual groupType is [%ld].",
                   gmmParams_.groupType);
    } else {
        errorMessage =
            gmmParams_.apiVersion == gmm::GMMApiVersion::V1
                ? "M-split scenario, but the actual scenario is no-split (multi-multi-multi)"
                : "groupType 0 (split M), but the actual groupType is [" + std::to_string(gmmParams_.groupType) + "]";
        CHECK_COND(gmmParams_.groupType == SPLIT_M, ACLNN_ERR_PARAM_INVALID,
                   "Weight quant cases with x dtype [%s] and weight dtype [%s] only support %s.",
                   op::ToString(xDtype_).GetString(), op::ToString(weightDtype_).GetString(), errorMessage.c_str());
    }

    // 多多多/单单单校验
    // groupType为-1仅对应多多多；groupType为0会出现单单单/单多单/单多多/多多单，仅支持单单单
    if (gmmParams_.groupType == NO_SPLIT) {
        CHECK_COND(gmmParams_.x->Size() == gmmParams_.weight->Size() && gmmParams_.x->Size() == gmmParams_.y->Size(),
                   ACLNN_ERR_PARAM_INVALID,
                   "In multi-multi-multi scenario, the sizes of x, weight and y should be all the same, but the "
                   "actual sizes are [%zu], [%zu] and [%zu].",
                   gmmParams_.x->Size(), gmmParams_.weight->Size(), gmmParams_.y->Size());
    } else {
        errorMessage = gmmParams_.apiVersion == gmm::GMMApiVersion::V1 ? "When splited axis is M"
                                                                       : "When groupType is 0 (split M)";
        CHECK_COND(gmmParams_.x->Size() == 1 && gmmParams_.weight->Size() == 1 && gmmParams_.y->Size() == 1,
                   ACLNN_ERR_PARAM_INVALID,
                   "%s, the sizes of x, weight and y should all be 1, but the actual sizes are [%zu], [%zu] and [%zu].",
                   errorMessage.c_str(), gmmParams_.x->Size(), gmmParams_.weight->Size(), gmmParams_.y->Size());
    }

    return ACLNN_SUCCESS;
}

aclnnStatus AclnnGroupedMatmulWeightQuantDAV3510Checker::CheckGroupListAndSplitItem() const
{
    if (gmmParams_.groupType == NO_SPLIT) {
        if (gmmParams_.apiVersion == GMMApiVersion::V2) {
            CHECK_COND(gmmParams_.groupListOptional == nullptr, ACLNN_ERR_PARAM_INVALID,
                       "GroupListOptional should be nullptr when groupType is -1.");
        } else if (gmmParams_.apiVersion != GMMApiVersion::V1) {
            CHECK_COND(gmmParams_.groupTensorOptional == nullptr, ACLNN_ERR_PARAM_INVALID,
                       "GroupListOptional(groupTensorOptional) should be nullptr when groupType is -1.");
        }
        CHECK_COND(gmmParams_.splitItem == X_Y_SEPARATED || gmmParams_.splitItem == Y_SEPARATED,
                   ACLNN_ERR_PARAM_INVALID,
                   "When y is separated, splitItem should be 0/1, but current splitItem is %ld.", gmmParams_.splitItem);
    } else {
        if (gmmParams_.apiVersion == gmm::GMMApiVersion::V1 || gmmParams_.apiVersion == gmm::GMMApiVersion::V2) {
            CHECK_COND(gmmParams_.groupListOptional != nullptr, ACLNN_ERR_PARAM_INVALID,
                       "GroupListOptional should not be nullptr when splited axis is M.");  // V1 没有groupType参数
        } else {
            CHECK_COND(gmmParams_.groupTensorOptional != nullptr, ACLNN_ERR_PARAM_INVALID,
                       "GroupListOptional should not be nullptr when groupType is 0.");
        }

        CHECK_COND(gmmParams_.splitItem == X_SEPARATED || gmmParams_.splitItem == NO_SEPARATED, ACLNN_ERR_PARAM_INVALID,
                   "When y is not separated, splitItem should be 2/3, but current splitItem is %ld.",
                   gmmParams_.splitItem);
    }
    return ACLNN_SUCCESS;
}

aclnnStatus AclnnGroupedMatmulWeightQuantDAV3510Checker::CheckTensorListSize() const
{
    if (gmmParams_.antiquantScaleOptional != nullptr) {
        CHECK_COND(gmmParams_.antiquantScaleOptional->Size() == gmmParams_.weight->Size(), ACLNN_ERR_PARAM_INVALID,
                   "AntiquantScaleOptional size should be equal to weight size, actual sizes are [%zu], [%zu]",
                   gmmParams_.antiquantScaleOptional->Size(), gmmParams_.weight->Size());
    }

    if (gmmParams_.antiquantOffsetOptional != nullptr) {
        CHECK_COND(gmmParams_.antiquantOffsetOptional->Size() == gmmParams_.weight->Size(), ACLNN_ERR_PARAM_INVALID,
                   "AntiquantOffsetOptional size should be equal to weight size, actual sizes are [%zu], [%zu]",
                   gmmParams_.antiquantOffsetOptional->Size(), gmmParams_.weight->Size());
    }

    if (gmmParams_.biasOptional != nullptr) {
        CHECK_COND(gmmParams_.biasOptional->Size() == gmmParams_.weight->Size(), ACLNN_ERR_PARAM_INVALID,
                   "BiasOptional size should be equal to weight size, actual sizes are [%zu], [%zu]",
                   gmmParams_.biasOptional->Size(), gmmParams_.weight->Size());
    }

    if (gmmParams_.perTokenScaleOptional != nullptr) {
        CHECK_COND(gmmParams_.perTokenScaleOptional->Size() == gmmParams_.x->Size(), ACLNN_ERR_PARAM_INVALID,
                   "PerTokenScaleOptional size should be equal to x size, actual sizes are [%zu], [%zu]",
                   gmmParams_.perTokenScaleOptional->Size(), gmmParams_.x->Size());
    }

    if (gmmParams_.scaleOptional != nullptr) {
        CHECK_COND(gmmParams_.scaleOptional->Size() == gmmParams_.weight->Size(), ACLNN_ERR_PARAM_INVALID,
                   "scaleOptional size should be equal to weight size, actual sizes are [%zu], [%zu]",
                   gmmParams_.scaleOptional->Size(), gmmParams_.weight->Size());
    }

    return ACLNN_SUCCESS;
}

aclnnStatus AclnnGroupedMatmulWeightQuantDAV3510Checker::CheckGroupedMatmulWeightQuantDAV3510()
{
    xDtype_ = gmmParams_.xDtype;
    weightDtype_ = (*gmmParams_.weight)[0]->GetDataType();
    yDtype_ = (*gmmParams_.y)[0]->GetDataType();

    CHECK_COND(CheckGroupTypeScenario() == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID, "CheckGroupTypeScenario failed.");
    CHECK_COND(CheckUnsupportedApi() == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID, "CheckUnsupportedApi failed.");
    CHECK_COND(CheckGroupListAndSplitItem() == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
               "CheckGroupListAndSplitItem failed.");

    // CheckAntiQuantParams和CheckQuantParams校验各种量化参数在各数据流的支持情况，后续对量化参数的通用校验不再区分数据流
    CHECK_COND(CheckAntiQuantParams() == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID, "CheckAntiQuantParams failed.");
    CHECK_COND(CheckQuantParams() == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID, "CheckQuantParams failed!");
    CHECK_COND(CheckTensorListSize() == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID, "CheckTensorListSize failed.");
    CHECK_RET(CheckYDtype() == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckQuantDtype() == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckBiasDtype() == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);

    CHECK_RET(CheckTransposeStatus() == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckScaleAndPerTokenScaleShape() == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);

    // 当前仅支持单单单/多多多场景，各tensorList的size相同，使用x的Size循环
    for (size_t i = 0; i < gmmParams_.x->Size(); i++) {
        CHECK_RET(CheckTensorNotNull(i) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_NULLPTR);

        CHECK_RET(CheckTensorDtype(gmmParams_.x, xDtype_, i, "x") == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
        CHECK_RET(CheckTensorDtype(gmmParams_.weight, weightDtype_, i, "weight") == ACLNN_SUCCESS,
                  ACLNN_ERR_PARAM_INVALID);
        CHECK_RET(CheckTensorDtype(gmmParams_.y, yDtype_, i, "y") == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);

        CHECK_RET(CheckDimNumAndFormat(i) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
        CHECK_COND(IsTransposeLastTwoDims((*gmmParams_.weight)[i]) == gmmParams_.transposeWeight,
                   ACLNN_ERR_PARAM_INVALID, "The transpose state must be the same for each tensor in weight.");
        CHECK_RET(CheckDimValue(i) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
        CHECK_RET(CheckWeightInnerAxisEven(i) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);

        CHECK_RET(CheckV1GroupList(i) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);

        if (gmmParams_.biasOptional != nullptr) {
            CHECK_RET(CheckTensorDtype(gmmParams_.biasOptional, biasDtype_, i, "bias") == ACLNN_SUCCESS,
                      ACLNN_ERR_PARAM_INVALID);
            CHECK_RET(CheckTensorShape(gmmParams_.biasOptional, i, "bias") == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
        }

        CHECK_RET(CheckAntiQuantDtype(i) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
        CHECK_RET(CheckAntiQuantShape(i) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);

        CHECK_COND(CheckGroupSize(i) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID, "CheckGroupSize failed");
    }
    return ACLNN_SUCCESS;
}
