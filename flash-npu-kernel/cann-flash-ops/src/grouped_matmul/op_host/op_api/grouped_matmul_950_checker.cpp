/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "grouped_matmul_950_checker.h"

using namespace gmm;

namespace gmm {
    template class AclnnGroupedMatmulDAV3510Checker<aclTensorList>;
    template class AclnnGroupedMatmulDAV3510Checker<aclTensor>;
}

namespace {
const aclTensor *GetInputTensor(const aclTensorList *input, size_t index = 0)
{
    if (index >= input->Size()) {
        return nullptr;
    }
    return (*input)[index];
}

const aclTensor *GetInputTensor(const aclTensor *input, size_t index = 0)
{
    (void)index;
    return input;
}

size_t GetInputTensorSize(const aclTensorList *input)
{
    return input->Size();
}

size_t GetInputTensorSize(const aclTensor *input)
{
    (void)input;
    return 1;
}
} // namespace

template <typename T>
void AclnnGroupedMatmulDAV3510Checker<T>::SetInputName(const std::string &xName, const std::string &weightName,
                                                       const std::string &perTokenScaleName,
                                                       const std::string &scaleName, const std::string &groupTensorName)
{
    this->xName_ = xName;
    this->weightName_ = weightName;
    this->perTokenScaleName_ = perTokenScaleName;
    this->scaleName_ = scaleName;
    this->groupTensorName_ = groupTensorName;
}

template <typename T>
bool AclnnGroupedMatmulDAV3510Checker<T>::LastTwoDimValueIsOne(const aclTensor *tensor) const
{
    // 检查tensor维度是否小于2
    if (tensor->GetViewShape().GetDimNum() < 2) {
        return false;
    }
    auto dim1 = tensor->GetViewShape().GetDimNum() - 1;
    auto dim2 = tensor->GetViewShape().GetDimNum() - LAST_TWO_DIM_INDEX;
    if (tensor->GetViewShape().GetDim(dim1) == 1 && tensor->GetViewShape().GetDim(dim2) == 1) {
        return true;
    }
    return false;
}

template <typename T>
bool AclnnGroupedMatmulDAV3510Checker<T>::CheckTensorListSizeForEachInput() const
{
    if (gmmParams_.scaleOptional == nullptr && GetInputTensor(gmmParams_.y)->GetDataType() == DataType::DT_INT32) {
        return true;
    }
    if (GetInputTensorSize(gmmParams_.scaleOptional) != GetInputTensorSize(gmmParams_.x)) {
        return false;
    }
    if (gmmParams_.perTokenScaleOptional != nullptr &&
        GetInputTensorSize(gmmParams_.perTokenScaleOptional) != GetInputTensorSize(gmmParams_.x)) {
        return false;
    }
    return true;
}

template <typename T>
aclnnStatus AclnnGroupedMatmulDAV3510Checker<T>::CheckGeneralQuantShape() const
{
    if (!CheckTensorListSizeForEachInput()) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    auto groupNum = gmmParams_.groupTensorOptional->GetViewShape().GetDim(0);
    for (size_t i = 0; i < GetInputTensorSize(gmmParams_.x); i++) {
        auto weightNIndex = GetInputTensor(gmmParams_.weight, i)->GetViewShape().GetDimNum() - 1;
        if (GetInputTensor(gmmParams_.x)->GetViewShape().GetDim(0) == 0 ||
            GetInputTensor(gmmParams_.weight, i)->GetViewShape().GetDim(weightNIndex) == 0) {
            return ACLNN_SUCCESS;
        }
        auto weightKIndex = GetInputTensor(gmmParams_.weight, i)->GetViewShape().GetDimNum() - LAST_TWO_DIM_INDEX;
        CHECK_COND(GetInputTensor(gmmParams_.x, i)->GetViewShape().GetDim(1) > 0, ACLNN_ERR_PARAM_INVALID,
                "When the M or N value is not 0, the K value[%ld] in %s should be positive.",
                GetInputTensor(gmmParams_.x, i)->GetViewShape().GetDim(1), xName_.c_str());
        CHECK_COND(GetInputTensor(gmmParams_.weight, i)->GetViewShape().GetDim(weightKIndex) > 0,
                ACLNN_ERR_PARAM_INVALID, "When the M or N value is not 0, The K value[%ld] in %s should be positive.",
                GetInputTensor(gmmParams_.weight, i)->GetViewShape().GetDim(weightKIndex), weightName_.c_str());
        if (gmmParams_.groupType == SPLIT_K) {
            CHECK_COND(GetInputTensor(gmmParams_.y, i)->GetViewShape().GetDim(0) == groupNum, ACLNN_ERR_PARAM_INVALID,
                       "When groupType is 2 (split K), the first dim of %s[%ld] should be equal to that of \
%s[%ld].",
                       yName_.c_str(), GetInputTensor(gmmParams_.y, i)->GetViewShape().GetDim(0),
                       groupTensorName_.c_str(), groupNum);
        }
    }
    return ACLNN_SUCCESS;
}

template <typename T>
aclnnStatus AclnnGroupedMatmulDAV3510Checker<T>::CheckQuantCasesFormat() const
{
    for (size_t i = 0; i < GetInputTensorSize(gmmParams_.x); i++) {
        CHECK_COND(!op::IsPrivateFormat(GetInputTensor(gmmParams_.x, i)->GetStorageFormat()), ACLNN_ERR_PARAM_INVALID,
                   "The format of %s[%zu] %s is invalid. It should only be ND.", xName_.c_str(), i,
                   op::ToString(GetInputTensor(gmmParams_.x, i)->GetStorageFormat()).GetString());
        CHECK_COND(!op::IsPrivateFormat(GetInputTensor(gmmParams_.weight, i)->GetStorageFormat()) ||
                       GetInputTensor(gmmParams_.weight, i)->GetStorageFormat() == op::Format::FORMAT_FRACTAL_NZ,
                   ACLNN_ERR_PARAM_INVALID, "The format of %s[%zu] %s is invalid. It should be ND or FRACTAL_NZ.",
                   weightName_.c_str(), i,
                   op::ToString(GetInputTensor(gmmParams_.weight, i)->GetStorageFormat()).GetString());
        CHECK_COND(!op::IsPrivateFormat(GetInputTensor(gmmParams_.y, i)->GetStorageFormat()), ACLNN_ERR_PARAM_INVALID,
                   "The format of %s[%zu] %s is invalid. It should only be ND.", yName_.c_str(), i,
                   op::ToString(GetInputTensor(gmmParams_.y, i)->GetStorageFormat()).GetString());
    }
    return ACLNN_SUCCESS;
}

template <typename T>
aclnnStatus AclnnGroupedMatmulDAV3510Checker<T>::CheckWeightStorageShape(int64_t kDimValue, int64_t nDimValue) const
{
    auto weightStorage = GetInputTensor(gmmParams_.weight)->GetStorageShape();
    auto weightStorageShapeDim = weightStorage.GetDimNum();
    CHECK_COND(weightStorageShapeDim == QUANT_WEIGHTNZ_STORAGE_DIM, ACLNN_ERR_PARAM_INVALID,
               "When format of weight is FRACTAL_NZ, the storage dim num of %s should be 5, but actual dim num is %lu",
               weightName_.c_str(), weightStorageShapeDim);

    auto weightStorageLastFourthDim = weightStorage.GetDim(weightStorageShapeDim - LAST_FOURTH_DIM_INDEX);
    auto weightStorageLastThirdDim = weightStorage.GetDim(weightStorageShapeDim - LAST_THIRD_DIM_INDEX);
    auto weightStorageLastSecondDim = weightStorage.GetDim(weightStorageShapeDim - LAST_SECOND_DIM_INDEX);
    auto weightStorageLastDim = weightStorage.GetDim(weightStorageShapeDim - LAST_FIRST_DIM_INDEX);
    CHECK_COND(weightStorageLastDim == CUBE_BLOCK_SIZE_32, ACLNN_ERR_PARAM_INVALID,
               "When format of weight is FRACTAL_NZ, the storage shape last dim of %s should be 32, but actual last \
dim is %ld",
               weightName_.c_str(), weightStorageLastDim);
    CHECK_COND(weightStorageLastSecondDim == CUBE_BLOCK_SIZE_16, ACLNN_ERR_PARAM_INVALID,
               "When format of weight is FRACTAL_NZ, the storage shape last second dim of %s should be 16, but actual \
last second dim is %ld",
               weightName_.c_str(), weightStorageLastSecondDim);
    if (gmmParams_.transposeWeight) {
        CHECK_COND(weightStorageLastFourthDim == (kDimValue + CUBE_BLOCK_SIZE_32 - 1) / CUBE_BLOCK_SIZE_32,
                   ACLNN_ERR_PARAM_INVALID,
                   "When format of weight is FRACTAL_NZ and transposition is true, the storage shape second dim of %s \
should be ceil(k/32), but actual second dim is %ld",
                   weightName_.c_str(), weightStorageLastFourthDim);
        CHECK_COND(weightStorageLastThirdDim == (nDimValue + CUBE_BLOCK_SIZE_16 - 1) / CUBE_BLOCK_SIZE_16,
                   ACLNN_ERR_PARAM_INVALID,
                   "When format of weight is FRACTAL_NZ and transposition is true, the storage shape third dim of %s \
should be ceil(n/16), but actual third dim is %ld",
                   weightName_.c_str(), weightStorageLastThirdDim);
    } else {
        CHECK_COND(weightStorageLastFourthDim == (nDimValue + CUBE_BLOCK_SIZE_32 - 1) / CUBE_BLOCK_SIZE_32,
                   ACLNN_ERR_PARAM_INVALID,
                   "When format of weight is FRACTAL_NZ and transposition is false, the storage shape second dim of %s \
should be ceil(n/32), but actual second dim is %ld",
                   weightName_.c_str(), weightStorageLastFourthDim);
        CHECK_COND(weightStorageLastThirdDim == (kDimValue + CUBE_BLOCK_SIZE_16 - 1) / CUBE_BLOCK_SIZE_16,
                   ACLNN_ERR_PARAM_INVALID,
                   "When format of weight is FRACTAL_NZ and transposition is false, the storage shape third dim of %s \
should be ceil(k/16), but actual third dim is %ld",
                   weightName_.c_str(), weightStorageLastThirdDim);
    }
    return ACLNN_SUCCESS;
}

template <typename T>
aclnnStatus AclnnGroupedMatmulDAV3510Checker<T>::CheckWeightNzSpecialParams() const
{
    CHECK_COND(gmmParams_.apiVersion == gmm::GMMApiVersion::WeightNz, ACLNN_ERR_PARAM_INVALID,
               "WeightNz feature is only supported in aclnnGroupedMatmulWeightNz");

    auto wDtype = GetInputTensor(gmmParams_.weight)->GetDataType();
    CHECK_COND(
        gmmParams_.xDtype == DataType::DT_INT8 && wDtype == DataType::DT_INT8, ACLNN_ERR_PARAM_INVALID,
        "When format of weight is FRACTAL_NZ, the x dtype and weight dtype should be int8, but x dtype is %s, weight \
dtype is %s",
        op::ToString(gmmParams_.xDtype).GetString(), op::ToString(wDtype).GetString());
    auto yDtype = GetInputTensor(gmmParams_.y)->GetDataType();
    CHECK_COND(yDtype != DataType::DT_INT8, ACLNN_ERR_PARAM_INVALID,
               "When format of weight is FRACTAL_NZ, the y dtype should not be int8.");

    auto weightViewShapeDim = GetInputTensor(gmmParams_.weight)->GetViewShape().GetDimNum();
    auto kDimValue =
        GetInputTensor(gmmParams_.weight)->GetViewShape().GetDim(weightViewShapeDim - LAST_SECOND_DIM_INDEX);
    auto nDimValue =
        GetInputTensor(gmmParams_.weight)->GetViewShape().GetDim(weightViewShapeDim - LAST_FIRST_DIM_INDEX);
    CHECK_COND(kDimValue != 1L && nDimValue != 1L, ACLNN_ERR_PARAM_INVALID,
               "When format of weight is FRACTAL_NZ, neither of the last two dimensions of %s can be 1, but actual \
k is %ld, n is %ld",
               weightName_.c_str(), kDimValue, nDimValue);
    return CheckWeightStorageShape(kDimValue, nDimValue);
}

template <typename T>
aclnnStatus AclnnGroupedMatmulDAV3510Checker<T>::CheckGroupedMatmulMxDtype() const
{
    if (gmmParams_.biasOptional != nullptr) {
       DataType biasDtype = GetInputTensor(gmmParams_.biasOptional)->GetDataType();
       CHECK_COND(biasDtype == DataType::DT_FLOAT, ACLNN_ERR_PARAM_INVALID,
                  "The %s dtype in mx should be FLOAT32, but actual dtype is %s.", biasName_.c_str(),
                  op::ToString(biasDtype).GetString());
    }
    DataType perTokenDtype = GetInputTensor(gmmParams_.perTokenScaleOptional)->GetDataType();
    CHECK_COND(perTokenDtype == DataType::DT_FLOAT8_E8M0, ACLNN_ERR_PARAM_INVALID,
               "The %s dtype in mx should be FLOAT8_E8M0, but actual dtype is %s.", perTokenScaleName_.c_str(),
               op::ToString(perTokenDtype).GetString());

    DataType yDtype = GetInputTensor(gmmParams_.y)->GetDataType();
    CHECK_COND(yDtype == DataType::DT_FLOAT16 || yDtype == DataType::DT_BF16 || yDtype == DataType::DT_FLOAT,
               ACLNN_ERR_PARAM_INVALID,
               "The output dtype should be float16, bfloat16 or float32 in mx quant case, but now %s dtype is %s",
               yName_.c_str(), op::ToString(yDtype).GetString());
    return ACLNN_SUCCESS;
}

template <typename T>
aclnnStatus AclnnGroupedMatmulDAV3510Checker<T>::CheckGroupedMatmulPerGroupDim() const
{
    for (size_t i = 0; i < GetInputTensorSize(gmmParams_.x); i++) {
        auto xDimNumber = GetInputTensor(gmmParams_.x, i)->GetViewShape().GetDimNum();
        auto weightDimNumber = GetInputTensor(gmmParams_.weight, i)->GetViewShape().GetDimNum();
        auto scaleDimNumber = GetInputTensor(gmmParams_.scaleOptional, i)->GetViewShape().GetDimNum();
        auto perTokenDimNumber = GetInputTensor(gmmParams_.perTokenScaleOptional, i)->GetViewShape().GetDimNum();
        CHECK_COND(xDimNumber == MIN_FM_DIM, ACLNN_ERR_PARAM_INVALID,
                   "The dim num of %s should be 2, but actual dim num is %zu.", xName_.c_str(), xDimNumber);
        if (gmmParams_.groupType == SPLIT_M) {
            CHECK_COND(weightDimNumber == SPLIT_M_SINGLE_WEIGHT_DIM, ACLNN_ERR_PARAM_INVALID,
                       "The dim num of %s should be 3 when groupType is 0 (split M), but actual dim num is %zu.",
                       weightName_.c_str(), weightDimNumber);
        } else if (gmmParams_.groupType == SPLIT_K) {
            CHECK_COND(weightDimNumber == SPLIT_K_SINGLE_WEIGHT_DIM, ACLNN_ERR_PARAM_INVALID,
                       "The dim num of %s should be 2 when groupType is 2 (split K), but actual dim num is %zu.",
                       weightName_.c_str(), weightDimNumber);
        }
        if (gmmParams_.groupType == SPLIT_M) {
            DataType scaleDtype = GetInputTensor(gmmParams_.scaleOptional, i)->GetDataType();
            if (scaleDtype == DataType::DT_FLOAT8_E8M0) {
                CHECK_COND(scaleDimNumber == MX_SPLIT_M_SCALE_DIM, ACLNN_ERR_PARAM_INVALID,
                           "When split m, the dim num of %s[%zu] should be equal 4 in mx quant mode.",
                           scaleName_.c_str(), scaleDimNumber);
                CHECK_COND(perTokenDimNumber == MX_SPLIT_M_PER_TOKEN_SCALE_DIM, ACLNN_ERR_PARAM_INVALID,
                           "When split m, the dim num of %s[%zu] should be 3 in mx quant mode.",
                           perTokenScaleName_.c_str(), perTokenDimNumber);
            } else {
                CHECK_COND(scaleDimNumber == weightDimNumber, ACLNN_ERR_PARAM_INVALID,
                           "When split m, the dim num of %s[%zu] should be equal to that of %s[%zu] in G-B \
quant mode.",
                           scaleName_.c_str(), scaleDimNumber, weightName_.c_str(), weightDimNumber);
                CHECK_COND(perTokenDimNumber == xDimNumber, ACLNN_ERR_PARAM_INVALID,
                           "When split m, the dim num of %s[%zu] should be equal to that of %s[%zu] in G-B \
quant mode.",
                           perTokenScaleName_.c_str(), perTokenDimNumber, xName_.c_str(), xDimNumber);
            }
        }
    }
    return ACLNN_SUCCESS;
}

template <typename T>
aclnnStatus AclnnGroupedMatmulDAV3510Checker<T>::CheckMxBiasInputShape(const TensorDimInfo &dimInfo, size_t index) const
{
    auto weightNIndex = GetInputTensor(gmmParams_.weight, index)->GetViewShape().GetDimNum() - 1;
    size_t biasDimNum = dimInfo.biasDimNum;
    int64_t groupNum = dimInfo.groupNum;
    if (biasDimNum != 0) {
        CHECK_COND(biasDimNum == MX_BIAS_DIM, ACLNN_ERR_PARAM_INVALID,
                   "In mx quant mode, the %s dim num should be 2, but actual is [%zu].", biasName_.c_str(), biasDimNum);
    }
    auto weightNDimValue = GetInputTensor(gmmParams_.weight, index)->GetViewShape().GetDim(weightNIndex);
    if (gmmParams_.biasOptional != nullptr) {
        auto biasGDimValue = GetInputTensor(gmmParams_.biasOptional, index)->GetViewShape().GetDim(0);
        auto biasNDimValue = GetInputTensor(gmmParams_.biasOptional, index)->GetViewShape().GetDim(1);
        CHECK_COND(biasGDimValue == groupNum, ACLNN_ERR_PARAM_INVALID,
               "The group dim of %s[%ld] and group number[%ld] should be equal.", biasName_.c_str(), biasGDimValue,
               groupNum);
        CHECK_COND(biasNDimValue == weightNDimValue, ACLNN_ERR_PARAM_INVALID,
               "The n dim of %s[%ld] and n dim of %s[%ld] should be equal.", biasName_.c_str(), biasNDimValue,
               weightName_.c_str(), weightNDimValue);
    }
    return ACLNN_SUCCESS;
}

template <typename T>
aclnnStatus AclnnGroupedMatmulDAV3510Checker<T>::CheckMxTypeMCaseInputShape(const TensorDimInfo &dimInfo,
                                                                            size_t index) const
{
    auto weightNIndex = GetInputTensor(gmmParams_.weight, index)->GetViewShape().GetDimNum() - 1;
    size_t scaleDimNum = dimInfo.scaleDimNum;
    size_t pertokenScaleDimNum = dimInfo.pertokenScaleDimNum;
    int64_t groupNum = dimInfo.groupNum;
    auto xMDimValue = GetInputTensor(gmmParams_.x, index)->GetViewShape().GetDim(0);
    auto xKDimValue = GetInputTensor(gmmParams_.x, index)->GetViewShape().GetDim(1);
    auto pertokenMDimValue = GetInputTensor(gmmParams_.perTokenScaleOptional, index)->GetViewShape().GetDim(0);
    auto pertokenScaleKDimValue = GetInputTensor(gmmParams_.perTokenScaleOptional, index)->GetViewShape().GetDim(1);
    auto pertokenScaleLastDimValue =
        GetInputTensor(gmmParams_.perTokenScaleOptional, index)->GetViewShape().GetDim(pertokenScaleDimNum - 1);
    auto weightNDimValue = GetInputTensor(gmmParams_.weight, index)->GetViewShape().GetDim(weightNIndex);
    auto scaleNDimValue = GetInputTensor(gmmParams_.scaleOptional, index)->GetViewShape().GetDim(weightNIndex);
    auto scaleGDimValue = GetInputTensor(gmmParams_.scaleOptional, index)->GetViewShape().GetDim(0);
    auto scaleKDimValue = GetInputTensor(gmmParams_.scaleOptional, index)->GetViewShape().GetDim(1);
    auto scaleLastDimValue = GetInputTensor(gmmParams_.scaleOptional, index)->GetViewShape().GetDim(scaleDimNum - 1);
    CHECK_COND(CheckMxBiasInputShape(dimInfo, index) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
               "CheckMxBiasInputShape failed.");
    CHECK_COND(xMDimValue == pertokenMDimValue, ACLNN_ERR_PARAM_INVALID,
               "The m dim of %s[%ld] and m dim of %s[%ld] should be equal.", xName_.c_str(), xMDimValue,
               perTokenScaleName_.c_str(), pertokenMDimValue);
    CHECK_COND(weightNDimValue == scaleNDimValue, ACLNN_ERR_PARAM_INVALID,
               "The n dim of %s[%ld] and n dim of %s[%ld] should be equal.", weightName_.c_str(), weightNDimValue,
               scaleName_.c_str(), scaleNDimValue);
    CHECK_COND(scaleGDimValue == groupNum, ACLNN_ERR_PARAM_INVALID,
               "The group dim of %s[%ld] and group number[%ld] should be equal.", scaleName_.c_str(), scaleGDimValue,
               groupNum);
    auto inferedScaleKDimValue = (xKDimValue + MXFP_DIVISOR_SIZE - 1) / MXFP_DIVISOR_SIZE;
    CHECK_COND(scaleKDimValue == inferedScaleKDimValue, ACLNN_ERR_PARAM_INVALID,
               "The k dim of %s[%ld] should be equal to ceil(k/64), which is [%ld].", scaleName_.c_str(),
               scaleKDimValue, inferedScaleKDimValue);
    CHECK_COND(pertokenScaleKDimValue == inferedScaleKDimValue, ACLNN_ERR_PARAM_INVALID,
               "The k dim of %s[%ld] should be equal to ceil(k/64), which is [%ld].", perTokenScaleName_.c_str(),
               pertokenScaleKDimValue, inferedScaleKDimValue);
    CHECK_COND(pertokenScaleLastDimValue == MXFP_MULTI_BASE_SIZE, ACLNN_ERR_PARAM_INVALID,
               "The last dim of %s should be equal to 2, but the actual is [%ld].", perTokenScaleName_.c_str(),
               pertokenScaleLastDimValue);
    CHECK_COND(scaleLastDimValue == MXFP_MULTI_BASE_SIZE, ACLNN_ERR_PARAM_INVALID,
               "The last dim of %s should be equal to 2, but the actual is [%ld].", scaleName_.c_str(), scaleLastDimValue);
    return ACLNN_SUCCESS;
}

template <typename T>
aclnnStatus AclnnGroupedMatmulDAV3510Checker<T>::CheckMxFp8TypeKCaseInputShape(const TensorDimInfo &dimInfo,
                                                                               size_t index) const
{
    size_t xDimNum = dimInfo.xDimNum;
    size_t weightDimNum = dimInfo.weightDimNum;
    size_t scaleDimNum = dimInfo.scaleDimNum;
    size_t pertokenScaleDimNum = dimInfo.pertokenScaleDimNum;
    int64_t groupNum = dimInfo.groupNum;
    // split k, x is (m,k), weight is (k,n), scale is (k//64+g, n, 2), pertoken is (m, k//64+g, 2)
    CHECK_COND(xDimNum == MX_SPLIT_K_SINGLE_X_DIM, ACLNN_ERR_PARAM_INVALID,
               "The %s dim num should be 2 when split k, but actual is [%zu].", xName_.c_str(), xDimNum);
    CHECK_COND(weightDimNum == MX_SPLIT_K_SINGLE_WEIGHT_DIM, ACLNN_ERR_PARAM_INVALID,
               "The %s dim num should be 2 when split k, but actual is [%zu].", weightName_.c_str(), weightDimNum);
    CHECK_COND(scaleDimNum == MX_SPLIT_K_SCALE_DIM, ACLNN_ERR_PARAM_INVALID,
               "The %s dim num should be 3 when split k, but actual is [%zu].", scaleName_.c_str(), scaleDimNum);
    CHECK_COND(pertokenScaleDimNum == MX_SPLIT_K_PER_TOKEN_SCALE_DIM, ACLNN_ERR_PARAM_INVALID,
               "The %s dim num should be 3 when split k, but actual is [%zu].", perTokenScaleName_.c_str(),
               pertokenScaleDimNum);
    auto xMDimValue = GetInputTensor(gmmParams_.x, index)->GetViewShape().GetDim(0);
    auto xKDimValue = GetInputTensor(gmmParams_.x, index)->GetViewShape().GetDim(1);
    auto weightNDimValue = GetInputTensor(gmmParams_.weight, index)->GetViewShape().GetDim(1);
    auto pertokenMDimValue = GetInputTensor(gmmParams_.perTokenScaleOptional, index)->GetViewShape().GetDim(0);
    auto pertokenKDimValue = GetInputTensor(gmmParams_.perTokenScaleOptional, index)->GetViewShape().GetDim(1);
    auto pertokenLastDimValue = GetInputTensor(gmmParams_.perTokenScaleOptional, index)
                                    ->GetViewShape()
                                    .GetDim(MX_SPLIT_K_PER_TOKEN_SCALE_DIM - 1);
    auto scaleKDimValue = GetInputTensor(gmmParams_.scaleOptional, index)->GetViewShape().GetDim(0);
    auto scaleNDimValue = GetInputTensor(gmmParams_.scaleOptional, index)->GetViewShape().GetDim(1);
    auto scaleLastDimValue =
        GetInputTensor(gmmParams_.scaleOptional, index)->GetViewShape().GetDim(MX_SPLIT_K_SCALE_DIM - 1);
    CHECK_COND(xMDimValue == pertokenMDimValue, ACLNN_ERR_PARAM_INVALID,
               "The m dim of %s[%ld] and m dim of %s[%ld] should be equal.", xName_.c_str(), xMDimValue,
               perTokenScaleName_.c_str(), pertokenMDimValue);
    CHECK_COND(weightNDimValue == scaleNDimValue, ACLNN_ERR_PARAM_INVALID,
               "The n dim of %s[%ld] and n dim of %s[%ld] should be equal.", weightName_.c_str(), weightNDimValue,
               scaleName_.c_str(), scaleNDimValue);
    CHECK_COND(pertokenKDimValue == (xKDimValue / MXFP_DIVISOR_SIZE + groupNum), ACLNN_ERR_PARAM_INVALID,
               "The k dim of %s[%ld] should be equal to the k dim of %s[%ld] divided by 64, plus the groupSize[%ld].",
                perTokenScaleName_.c_str(), pertokenKDimValue, xName_.c_str(), xKDimValue, groupNum);
    CHECK_COND(scaleKDimValue == (xKDimValue / MXFP_DIVISOR_SIZE + groupNum), ACLNN_ERR_PARAM_INVALID,
               "The k dim of %s[%ld] should be equal to the k dim of %s[%ld] divided by 64, plus the groupSize[%ld].",
                scaleName_.c_str(), scaleKDimValue, xName_.c_str(), xKDimValue, groupNum);
    CHECK_COND(scaleLastDimValue == 2, ACLNN_ERR_PARAM_INVALID, // last dim should be 2 in mx typek quant mode
               "The last dim of %s[%ld] should be 2 when split k in mx quant mode.", scaleName_.c_str(),
               scaleLastDimValue);
    CHECK_COND(pertokenLastDimValue == 2, ACLNN_ERR_PARAM_INVALID, // last dim should be 2 in mx typek quant mode
               "The last dim of %s[%ld] should be 2 when split k in mx quant mode.", perTokenScaleName_.c_str(),
               pertokenLastDimValue);
    return ACLNN_SUCCESS;
}

template <typename T>
aclnnStatus AclnnGroupedMatmulDAV3510Checker<T>::CheckGroupedMatmulMxShape() const
{
    for (size_t i = 0; i < GetInputTensorSize(gmmParams_.x); i++) {
        auto xDimNum = GetInputTensor(gmmParams_.x, i)->GetViewShape().GetDimNum();
        auto weightDimNum = GetInputTensor(gmmParams_.weight, i)->GetViewShape().GetDimNum();
        auto scaleDimNum = GetInputTensor(gmmParams_.scaleOptional, i)->GetViewShape().GetDimNum();
        auto pertokenScaleDimNum = GetInputTensor(gmmParams_.perTokenScaleOptional, i)->GetViewShape().GetDimNum();
        auto groupNum = gmmParams_.groupTensorOptional->GetViewShape().GetDim(0);
        size_t biasDimNum = 0;
        if (gmmParams_.biasOptional != nullptr) {
            biasDimNum = GetInputTensor(gmmParams_.biasOptional, i)->GetViewShape().GetDimNum();
        }
        const TensorDimInfo dimInfo = {xDimNum, weightDimNum, scaleDimNum, pertokenScaleDimNum, groupNum, biasDimNum};
        if (gmmParams_.groupType == SPLIT_M) {
            CHECK_COND(CheckMxTypeMCaseInputShape(dimInfo, i) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
                       "CheckMxTypeMCaseInputShape failed.");
        } else {
            CHECK_COND(CheckMxFp8TypeKCaseInputShape(dimInfo, i) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
                       "CheckMxFp8TypeKCaseInputShape failed.");
        }
    }
    return ACLNN_SUCCESS;
}

template <typename T>
bool AclnnGroupedMatmulDAV3510Checker<T>::IsSpecialMXCase(const T *tensorList) const
{
    // 已校验mx场景scale的shape大于或等于3维度，不存在越界取值问题
    // mx特殊场景 (m,k,2) -> shape(1,1,2), stride(2,2,1); (k,m,2) -> shape(1,1,2), stride(2,2,1), 无法通过stride识别转置
    for (size_t i = 0; i < GetInputTensorSize(tensorList); i++) {
        auto tensorDimNum = GetInputTensor(tensorList, i)->GetViewShape().GetDimNum();
        auto secondLastDimValue =
            GetInputTensor(tensorList, i)->GetViewShape().GetDim(tensorDimNum - LAST_SECOND_DIM_INDEX);
        auto thirdLastDimValue =
            GetInputTensor(tensorList, i)->GetViewShape().GetDim(tensorDimNum - LAST_THIRD_DIM_INDEX);
        if (secondLastDimValue == 1 && thirdLastDimValue == 1) {
            return true;
        }
    }
    return false;
}

template <typename T>
aclnnStatus AclnnGroupedMatmulDAV3510Checker<T>::CheckGroupedMatmulMxScaleTranspose() const
{
    bool transposeScale = IsTransposeForMxShape(GetInputTensor(gmmParams_.scaleOptional));
    bool transposePerTokenScale = IsTransposeForMxShape(GetInputTensor(gmmParams_.perTokenScaleOptional));
    if (!IsSpecialMXCase(gmmParams_.scaleOptional)) {
        CHECK_COND(transposeScale == gmmParams_.transposeWeight, ACLNN_ERR_PARAM_INVALID,
                   "The transposition of %s/%s should be equal, but actual transpositions are %s/%s in mx case.",
                   scaleName_.c_str(), weightName_.c_str(), transposeScale ? "true" : "false",
                   gmmParams_.transposeWeight ? "true" : "false");
    }
    if (!IsSpecialMXCase(gmmParams_.perTokenScaleOptional)) {
        CHECK_COND(transposePerTokenScale == gmmParams_.transposeX, ACLNN_ERR_PARAM_INVALID,
                   "The transposition of %s/%s should be equal, but actual transpositions are %s/%s in mx case.",
                   perTokenScaleName_.c_str(), xName_.c_str(), transposePerTokenScale ? "true" : "false",
                   gmmParams_.transposeX ? "true" : "false");
    }
    return ACLNN_SUCCESS;
}

template <typename T>
aclnnStatus AclnnGroupedMatmulDAV3510Checker<T>::CheckGroupedMatmulMxfp8() const
{
    CHECK_COND(gmmParams_.biasOptional == nullptr, ACLNN_ERR_PARAM_INVALID, "mxfp8 does not support bias.");
    CHECK_COND(gmmParams_.perTokenScaleOptional != nullptr, ACLNN_ERR_PARAM_INVALID,
               "%s should not be nullptr in mx case.", perTokenScaleName_.c_str());
    CHECK_COND(CheckGroupedMatmulMxDtype() == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
               "CheckGroupedMatmulMxDtype failed");
    if (gmmParams_.groupType == SPLIT_M) {
        CHECK_COND(!gmmParams_.transposeX, ACLNN_ERR_PARAM_INVALID,
                   "When groupType is 0 (split m), the transposition of X only support false, but actual \
tranposition is %s in mx case.",
                   gmmParams_.transposeX ? "true" : "false");
    } else if (gmmParams_.groupType == SPLIT_K) {
        CHECK_COND(!gmmParams_.transposeWeight && gmmParams_.transposeX, ACLNN_ERR_PARAM_INVALID,
                   "When groupType is 2 (split K), the transposition of %s/%s only support true/false, but actual \
transpositions are %s/%s in mx case.",
                   xName_.c_str(), weightName_.c_str(), gmmParams_.transposeX ? "true" : "false",
                   gmmParams_.transposeWeight ? "true" : "false");
    }
    CHECK_COND(CheckGroupedMatmulMxScaleTranspose() == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
               "CheckGroupedMatmulMxScaleTranspose failed");
    CHECK_COND(CheckGroupedMatmulPerGroupDim() == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
               "CheckGroupedMatmulPerGroupDim failed");
    CHECK_COND(CheckGroupedMatmulMxShape() == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
               "CheckGroupedMatmulMxShape failed");
    return ACLNN_SUCCESS;
}

template <typename T>
aclnnStatus AclnnGroupedMatmulDAV3510Checker<T>::CheckGroupedMatmulMxfp4() const
{
    CHECK_COND(gmmParams_.perTokenScaleOptional != nullptr, ACLNN_ERR_PARAM_INVALID,
               "%s should not be nullptr in mx case.", perTokenScaleName_.c_str());
    CHECK_COND(CheckGroupedMatmulMxDtype() == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
               "CheckGroupedMatmulMxDtype failed");
    CHECK_COND(!gmmParams_.transposeX, ACLNN_ERR_PARAM_INVALID,
               "When groupType is 0 (split m), the transposition of %s only support false, but actual \
tranposition is %s in mxfp4.",
               xName_.c_str(), gmmParams_.transposeX ? "true" : "false");
    CHECK_COND(CheckGroupedMatmulMxScaleTranspose() == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
               "CheckGroupedMatmulMxScaleTranspose failed");
    CHECK_COND(CheckGroupedMatmulPerGroupDim() == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
               "CheckGroupedMatmulPerGroupDim failed");
    CHECK_COND(CheckGroupedMatmulMxShape() == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
               "CheckGroupedMatmulMxShape failed");
    CHECK_COND(CheckGroupedMatmulFp4MxDimValue() == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
               "CheckGroupedMatmulFp4MxDimValue failed");
    return ACLNN_SUCCESS;
}

template <typename T>
aclnnStatus AclnnGroupedMatmulDAV3510Checker<T>::CheckGroupedMatmulFp4MxDimValue() const
{
    for (size_t i = 0; i < GetInputTensorSize(gmmParams_.x); i++) {
        auto weightNIndex = GetInputTensor(gmmParams_.weight, i)->GetViewShape().GetDimNum() - 1;
        auto xKDimValue = GetInputTensor(gmmParams_.x, i)->GetViewShape().GetDim(1);
        auto weightNDimValue = GetInputTensor(gmmParams_.weight, i)->GetViewShape().GetDim(weightNIndex);
        auto weightKIndex = GetInputTensor(gmmParams_.weight, i)->GetViewShape().GetDimNum() - LAST_TWO_DIM_INDEX;
        auto weightKDimValue = GetInputTensor(gmmParams_.weight, i)->GetViewShape().GetDim(weightKIndex);
        //2：检查N是否为偶数
        auto weightNDimModValue = weightNDimValue % 2;
        //2：检查K是否为偶数
        auto xKDimModValue = xKDimValue % 2;
        if (!gmmParams_.transposeWeight) {
            CHECK_COND(weightNDimModValue == 0, ACLNN_ERR_PARAM_INVALID,
                   "When the weight is not transposed, the dim N value of %s should be even, but actual dim \
value is %lu",
                   weightName_.c_str(), weightNDimValue);
        }
        CHECK_COND(xKDimModValue == 0, ACLNN_ERR_PARAM_INVALID,
                   "When the dtypes of x and weight inputs are fp4 , the dim K value of %s should be even, but actual dim \
value is %lu",
                   xName_.c_str(), xKDimValue);
        // 2: mxfp4场景下不支持k轴为2
        CHECK_COND(xKDimValue != 2 && weightKDimValue != 2, ACLNN_ERR_PARAM_INVALID,
                   "When the dtypes of x and weight inputs are fp4, the dim K value should not be 2, but actual dim K \
value of %s is %lu and dim K value of %s is %lu",
                   xName_.c_str(), xKDimValue, weightName_.c_str(), weightKDimValue);
    }
    return ACLNN_SUCCESS;
}

template <typename T>
aclnnStatus AclnnGroupedMatmulDAV3510Checker<T>::CheckNonMxQuantTransposeStatus() const
{
    if (gmmParams_.groupType == SPLIT_M) {
        CHECK_COND(!gmmParams_.transposeX, ACLNN_ERR_PARAM_INVALID,
                   "In non-pergroup quantification mode, when groupType is 0 (split M), %s must not be transposed.",
                   xName_.c_str());
    } else if (gmmParams_.groupType == SPLIT_K) {
        CHECK_COND(gmmParams_.transposeX && !gmmParams_.transposeWeight, ACLNN_ERR_PARAM_INVALID,
                   "In non-pergroup quantification mode, when groupType is 2 (split K), the transposition of %s and \
%s must be true/false, but actual transpositions are %s/%s.",
                   xName_.c_str(), weightName_.c_str(), gmmParams_.transposeX ? "true" : "false",
                   gmmParams_.transposeWeight ? "true" : "false");
    }
    return ACLNN_SUCCESS;
}

template <typename T>
aclnnStatus AclnnGroupedMatmulDAV3510Checker<T>::CheckNonPerGroupQuantDim() const
{
    for (size_t i = 0; i < GetInputTensorSize(gmmParams_.x); ++i) {
        auto scaleDimNumber = GetInputTensor(gmmParams_.scaleOptional, i)->GetViewShape().GetDimNum();
        // 2: (E, N)
        CHECK_COND(scaleDimNumber == 1 || scaleDimNumber == 2, ACLNN_ERR_PARAM_INVALID,
                   "In non-pergroup quantification mode, the dim num of %s should be 1 or 2, but actual dim \
num is %lu",
                   scaleName_.c_str(), scaleDimNumber);

        if (gmmParams_.perTokenScaleOptional != nullptr) {
            auto perTokenDimNumber = GetInputTensor(gmmParams_.perTokenScaleOptional, i)->GetViewShape().GetDimNum();
            CHECK_COND(perTokenDimNumber == 1 || perTokenDimNumber == 2, ACLNN_ERR_PARAM_INVALID, // 2: (E, M)
                       "In non-pergroup quantification mode, the dim num of %s should be 1 or 2, but \
actual dim num is %lu.",
                       perTokenScaleName_.c_str(), perTokenDimNumber);
        }
    }
    return ACLNN_SUCCESS;
}

template <typename T>
aclnnStatus AclnnGroupedMatmulDAV3510Checker<T>::CheckNonPerGroupQuantPertokenShape() const
{
    auto perTokenDimNumber = GetInputTensor(gmmParams_.perTokenScaleOptional)->GetViewShape().GetDimNum();
    auto perTokenFirstDim = GetInputTensor(gmmParams_.perTokenScaleOptional)->GetViewShape().GetDim(0);
    auto xMDim = GetInputTensor(gmmParams_.x)->GetViewShape().GetDim(0);
    auto groupNum = gmmParams_.groupTensorOptional->GetViewShape().GetDim(0);
    if (gmmParams_.groupType == SPLIT_M) {
        if (perTokenDimNumber == 1) {
            CHECK_COND(perTokenFirstDim == xMDim || perTokenFirstDim == groupNum, ACLNN_ERR_PARAM_INVALID,
                       "In non-pergroup quantification mode, when groupType is 0 (split M) and %s \
has 1 dim, the first dim of %s[%ld] should be equal to that of %s[%ld] or that of \
%s[%ld].",
                       perTokenScaleName_.c_str(), perTokenScaleName_.c_str(), perTokenFirstDim, xName_.c_str(), xMDim,
                       groupTensorName_.c_str(), groupNum);
        } else {
            CHECK_COND(perTokenFirstDim == groupNum, ACLNN_ERR_PARAM_INVALID,
                       "In non-pergroup quantification mode, when groupType is 0 (split M) and %s \
has 2 dims, the first dim of %s[%ld] should be equal to that of %s[%ld].",
                       perTokenScaleName_.c_str(), perTokenScaleName_.c_str(), perTokenFirstDim,
                       groupTensorName_.c_str(), groupNum);
            CHECK_COND(GetInputTensor(gmmParams_.perTokenScaleOptional)->GetViewShape().GetDim(1) == 1,
                       ACLNN_ERR_PARAM_INVALID,
                       "In non-pergroup quantification mode, when groupType is 0 (split M) and %s \
has 2 dims, the second dim of %s[%ld] should be equal to 1.",
                       perTokenScaleName_.c_str(), perTokenScaleName_.c_str(),
                       GetInputTensor(gmmParams_.perTokenScaleOptional)->GetViewShape().GetDim(1));
        }
    } else if (gmmParams_.groupType == SPLIT_K) {
        CHECK_COND(perTokenFirstDim == groupNum, ACLNN_ERR_PARAM_INVALID,
                   "In non-pergroup quantification mode, when groupType is 2 (split K), the first dim of \
%s[%ld] should be equal to that of %s[%ld].",
                   perTokenScaleName_.c_str(), perTokenFirstDim, groupTensorName_.c_str(), groupNum);
        if (perTokenDimNumber > 1) {
            auto perTokenSecondDim = GetInputTensor(gmmParams_.perTokenScaleOptional)->GetViewShape().GetDim(1);
            CHECK_COND(perTokenSecondDim == xMDim || perTokenSecondDim == 1, ACLNN_ERR_PARAM_INVALID,
                       "In non-pergroup quantification mode, when groupType is 2 (split K) and %s \
has 2 dims, the second dim of %s[%ld] should be equal to that of %s[%ld] or 1.",
                       perTokenScaleName_.c_str(), perTokenScaleName_.c_str(), perTokenSecondDim, xName_.c_str(),
                       xMDim);
        }
    }
    return ACLNN_SUCCESS;
}

template <typename T>
aclnnStatus AclnnGroupedMatmulDAV3510Checker<T>::CheckNonPerGroupQuantShape() const
{
    auto groupNum = gmmParams_.groupTensorOptional->GetViewShape().GetDim(0);
    for (size_t i = 0; i < GetInputTensorSize(gmmParams_.x); ++i) {
        auto scaleDimNumber = GetInputTensor(gmmParams_.scaleOptional, i)->GetViewShape().GetDimNum();
        auto scaleNDim = GetInputTensor(gmmParams_.scaleOptional, i)->GetViewShape().GetDim(scaleDimNumber - 1);
        auto weightNIndex = GetInputTensor(gmmParams_.weight, i)->GetViewShape().GetDimNum() - 1;
        auto weightNDim = GetInputTensor(gmmParams_.weight, i)->GetViewShape().GetDim(weightNIndex);
        CHECK_COND(GetInputTensor(gmmParams_.scaleOptional, i)->GetViewShape().GetDim(0) == groupNum,
                   ACLNN_ERR_PARAM_INVALID,
                   "In non-pergroup quantification mode, the first dim of %s[%ld] should be equal to that of \
%s[%ld].",
                   scaleName_.c_str(), GetInputTensor(gmmParams_.scaleOptional, i)->GetViewShape().GetDim(0),
                   groupTensorName_.c_str(), groupNum);
        if (scaleDimNumber > 1) {
            CHECK_COND(scaleNDim == 1 || scaleNDim == weightNDim, ACLNN_ERR_PARAM_INVALID,
                       "In non-pergroup quantification mode, the N dim of %s[%ld] should be 1 or equal to \
that of %s[%ld].",
                       scaleName_.c_str(), scaleNDim, weightName_.c_str(), weightNDim);
            DataType yDtype = GetInputTensor(gmmParams_.y)->GetDataType();
            if (yDtype == DataType::DT_INT8) {
                CHECK_COND(scaleNDim == weightNDim, ACLNN_ERR_PARAM_INVALID,
                           "When the output dtype is int8, the N dim of %s[%ld] should be equal to that of %s[%ld].",
                           scaleName_.c_str(), scaleNDim, weightName_.c_str(), weightNDim);
            }
        }

        if (gmmParams_.perTokenScaleOptional != nullptr) {
            CHECK_RET(CheckNonPerGroupQuantPertokenShape() == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
        }
    }
    return ACLNN_SUCCESS;
}

template <typename T>
aclnnStatus AclnnGroupedMatmulDAV3510Checker<T>::CheckInt8QuantDtype() const
{
    static const std::vector<DataType> legalOutputDtypes = {DataType::DT_INT8, DataType::DT_INT32, DataType::DT_BF16,
                                                            DataType::DT_FLOAT16};
    DataType yDtype = GetInputTensor(gmmParams_.y)->GetDataType();
    CHECK_COND(std::find(legalOutputDtypes.begin(), legalOutputDtypes.end(), yDtype) != legalOutputDtypes.end(),
               ACLNN_ERR_PARAM_INVALID,
               "Expect y dtype to be int8, int32, float16 or bfloat16 in int8 quant case, but actual dtype is %s",
               op::ToString(yDtype).GetString());
    if (gmmParams_.biasOptional != nullptr) {
        DataType biasDtype = (*gmmParams_.biasOptional)[0]->GetDataType();
        if (yDtype == DataType::DT_BF16) {
            CHECK_COND(
                biasDtype == DataType::DT_INT32 || biasDtype == DataType::DT_BF16 || biasDtype == DataType::DT_FLOAT,
                ACLNN_ERR_PARAM_INVALID,
                "When y dtype is bfloat16, the bias dtype should be int32, bfloat16 or float32, but actual dtype is %s",
                op::ToString(biasDtype).GetString());
        } else if (yDtype == DataType::DT_FLOAT16) {
            CHECK_COND(
                biasDtype == DataType::DT_INT32 || biasDtype == DataType::DT_FLOAT16 || biasDtype == DataType::DT_FLOAT,
                ACLNN_ERR_PARAM_INVALID,
                "When y dtype is float16, the bias dtype should be int32, float16 or float32, but actual dtype is %s",
                op::ToString(biasDtype).GetString());
        } else if (yDtype == DataType::DT_INT8 || yDtype == DataType::DT_INT32) {
            CHECK_COND(biasDtype == DataType::DT_INT32, ACLNN_ERR_PARAM_INVALID,
                       "When y dtype is int8 or int32, the bias dtype should be int32, but actual dtype is %s",
                       op::ToString(biasDtype).GetString());
        }
    }
    if (yDtype == DataType::DT_INT32 && gmmParams_.scaleOptional == nullptr) {
        return ACLNN_SUCCESS;
    }
    DataType scaleDtype = GetInputTensor(gmmParams_.scaleOptional)->GetDataType();
    if (yDtype == DataType::DT_BF16) {
        CHECK_COND(scaleDtype == DataType::DT_BF16 || scaleDtype == DataType::DT_UINT64 ||
                       scaleDtype == DataType::DT_INT64 || scaleDtype == DataType::DT_FLOAT, ACLNN_ERR_PARAM_INVALID,
                   "When y dtype is bfloat16, the scale dtype should be bfloat16, uint64, int64 or float32, but actual \
dtype is %s", op::ToString(scaleDtype).GetString());
    } else if (yDtype == DataType::DT_FLOAT16) {
        CHECK_COND(scaleDtype == DataType::DT_UINT64 || scaleDtype == DataType::DT_INT64 ||
                       scaleDtype == DataType::DT_FLOAT, ACLNN_ERR_PARAM_INVALID,
                   "When y dtype is float16, the scale dtype should be uint64, int64 or float32, but actual dtype is %s",
                   op::ToString(scaleDtype).GetString());
    } else if (yDtype == DataType::DT_INT8) {
        CHECK_COND(scaleDtype == DataType::DT_UINT64 || scaleDtype == DataType::DT_INT64, ACLNN_ERR_PARAM_INVALID,
                   "When y dtype is int8, the scale dtype should be uint64 or int64, but actual dtype is %s",
                   op::ToString(scaleDtype).GetString());
    } else if (yDtype == DataType::DT_INT32) {
        CHECK_COND(scaleDtype == DataType::DT_UINT64 || scaleDtype == DataType::DT_INT64, ACLNN_ERR_PARAM_INVALID,
                   "When y dtype is int32 and scaleOptional is not nullptr, the scale dtype should be uint64 or int64, \
but actual dtype is %s",
                   op::ToString(scaleDtype).GetString());
    }
    if (gmmParams_.perTokenScaleOptional != nullptr) {
        CHECK_COND(yDtype != DataType::DT_INT8, ACLNN_ERR_PARAM_INVALID,
                   "When perTokenScaleOptional is not nullptr, y dtype should not be int8, but actual dtype is %s",
                   op::ToString(yDtype).GetString());
        DataType perTokenScaleDtype = GetInputTensor(gmmParams_.perTokenScaleOptional)->GetDataType();
        CHECK_COND(perTokenScaleDtype == DataType::DT_FLOAT, ACLNN_ERR_PARAM_INVALID,
                   "The perTokenScaleOptional dtype should be float32 in int8 quant case, but actual dtype is %s",
                   op::ToString(perTokenScaleDtype).GetString());
        if (yDtype == DataType::DT_BF16) {
            CHECK_COND(scaleDtype == DataType::DT_BF16 || scaleDtype == DataType::DT_FLOAT, ACLNN_ERR_PARAM_INVALID,
                       "When perTokenScaleOptional is not nullptr and y dtype is bfloat16, the scale dtype should be \
bfloat16 or float32, but actual dtype is %s", op::ToString(scaleDtype).GetString());
        } else if (yDtype == DataType::DT_FLOAT16) {
            CHECK_COND(scaleDtype == DataType::DT_FLOAT, ACLNN_ERR_PARAM_INVALID,
                       "When perTokenScaleOptional is not nullptr and y dtype is float16, the scale dtype should be \
float32, but actual dtype is %s", op::ToString(scaleDtype).GetString());
        }
    }
    return ACLNN_SUCCESS;
}

template <typename T>
aclnnStatus AclnnGroupedMatmulDAV3510Checker<T>::CheckInt8QuantParams() const
{
    CHECK_COND(gmmParams_.groupType == SPLIT_M, ACLNN_ERR_PARAM_INVALID,
               "In int8 quant case only supports groupType 0 (split M), but actual groupType is %ld",
               gmmParams_.groupType);
    CHECK_RET(CheckInt8QuantDtype() == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckNonMxQuantTransposeStatus() == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    if (GetInputTensor(gmmParams_.y)->GetDataType() == DataType::DT_INT32) {
        return ACLNN_SUCCESS;
    }
    CHECK_RET(CheckNonPerGroupQuantDim() == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckNonPerGroupQuantShape() == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);

    if (gmmParams_.perTokenScaleOptional != nullptr) {
        CHECK_COND(GetInputTensor(gmmParams_.perTokenScaleOptional)->GetViewShape().GetDimNum() == 1,
                   ACLNN_ERR_PARAM_INVALID,
                   "In int8 quant case, the dim num of perTokenScaleOptional should be 1, but actual dim num is %lu",
                   GetInputTensor(gmmParams_.perTokenScaleOptional)->GetViewShape().GetDimNum());
        auto perTokenMDim = GetInputTensor(gmmParams_.perTokenScaleOptional)->GetViewShape().GetDim(0);
        CHECK_COND(perTokenMDim == GetInputTensor(gmmParams_.x)->GetViewShape().GetDim(0), ACLNN_ERR_PARAM_INVALID,
                   "In int8 quant case, the M dim[%ld] of perTokenScaleOptional should be equal to that of x[%ld]",
                   perTokenMDim, GetInputTensor(gmmParams_.x)->GetViewShape().GetDim(0));
    }
    return ACLNN_SUCCESS;
}

template <typename T>
bool AclnnGroupedMatmulDAV3510Checker<T>::IsSpecialperTileScene(int64_t groupNum, int64_t weightNDim,
                                                                int64_t weightKDim, int64_t xMDim,
                                                                int64_t perTokenMDim) const
{
    return groupNum > 1L && gmmParams_.groupType == SPLIT_K && weightKDim < PERTILE_GROUP_SIZE &&
           weightNDim <= PERTILE_GROUP_SIZE && xMDim > 1L && xMDim == perTokenMDim;
}

template <typename T>
bool AclnnGroupedMatmulDAV3510Checker<T>::IsPerTileQuantMode() const
{
    bool isPerTileQuantMode = false;
    if (gmmParams_.perTokenScaleOptional == nullptr) {
        return isPerTileQuantMode;
    }
    auto groupNum = gmmParams_.groupTensorOptional->GetViewShape().GetDim(0);
    for (size_t i = 0; i < GetInputTensorSize(gmmParams_.weight); ++i) {
        auto scaleDimNumber = GetInputTensor(gmmParams_.scaleOptional, i)->GetViewShape().GetDimNum();
        auto weightDimNumber = GetInputTensor(gmmParams_.weight, i)->GetViewShape().GetDimNum();
        // 2 is the minimum dimension for scale in perTile case
        if (scaleDimNumber < 2 || weightDimNumber != scaleDimNumber) {
            return false;
        }
        auto xDimNumber = GetInputTensor(gmmParams_.x, i)->GetViewShape().GetDimNum();
        auto perTokenDimNumber = GetInputTensor(gmmParams_.perTokenScaleOptional, i)->GetViewShape().GetDimNum();
        // 2 is the minimum dimension for perTokenScaleOptional in perTile case
        if (perTokenDimNumber < 2 || xDimNumber != perTokenDimNumber) {
            return false;
        }
        bool transposePerTokenScale = IsTransposeLastTwoDims(GetInputTensor(gmmParams_.perTokenScaleOptional, i));
        auto scaleNDim = GetInputTensor(gmmParams_.scaleOptional, i)->GetViewShape().GetDim(scaleDimNumber - 1);
        auto scaleKDim =
            GetInputTensor(gmmParams_.scaleOptional, i)->GetViewShape().GetDim(scaleDimNumber - LAST_TWO_DIM_INDEX);
        auto weightNDim = GetInputTensor(gmmParams_.weight, i)->GetViewShape().GetDim(weightDimNumber - 1);
        auto weightKDim =
            GetInputTensor(gmmParams_.weight, i)->GetViewShape().GetDim(weightDimNumber - LAST_TWO_DIM_INDEX);
        auto xMDim = GetInputTensor(gmmParams_.x, i)->GetViewShape().GetDim(0);
        auto perTokenMDim = GetInputTensor(gmmParams_.perTokenScaleOptional, i)->GetViewShape().GetDim(0);
        // m轴分组通过校验weight与scale是否维度一致可区分开其他场景，这里仅通过判断k分组时x与perToken的转置是否一致区分其他场景
        if (!LastTwoDimValueIsOne(GetInputTensor(gmmParams_.perTokenScaleOptional, i)) &&
            transposePerTokenScale != gmmParams_.transposeX &&
            !IsSpecialperTileScene(groupNum, weightNDim, weightKDim, xMDim, perTokenMDim)) {
            return false;
        }
        isPerTileQuantMode = false;
        bool isKdimValid =
            ((gmmParams_.groupType == SPLIT_K && scaleKDim == (weightKDim / PERTILE_GROUP_SIZE) + groupNum) ||
             (gmmParams_.groupType == SPLIT_M &&
              scaleKDim == (weightKDim + PERTILE_GROUP_SIZE - 1) / PERTILE_GROUP_SIZE));
        if (scaleNDim == (weightNDim + PERTILE_GROUP_SIZE - 1) / PERTILE_GROUP_SIZE && isKdimValid) {
            isPerTileQuantMode = true;
        }
    }
    return isPerTileQuantMode;
}

template <typename T>
aclnnStatus AclnnGroupedMatmulDAV3510Checker<T>::CheckGroupedMatmulPerTileShape() const
{
    for (size_t i = 0; i < GetInputTensorSize(gmmParams_.x); i++) {
        auto xMIndex = 0;
        auto weightNIndex = GetInputTensor(gmmParams_.weight, i)->GetViewShape().GetDimNum() - 1;
        auto xKIndex = 1;
        auto weightKIndex = GetInputTensor(gmmParams_.weight, i)->GetViewShape().GetDimNum() - LAST_TWO_DIM_INDEX;
        auto weightKDim = GetInputTensor(gmmParams_.weight, i)->GetViewShape().GetDim(weightKIndex);
        auto xMDim = GetInputTensor(gmmParams_.x, i)->GetViewShape().GetDim(xMIndex);
        auto weightNDim = GetInputTensor(gmmParams_.weight, i)->GetViewShape().GetDim(weightNIndex);
        auto scaleKDim = GetInputTensor(gmmParams_.scaleOptional, i)->GetViewShape().GetDim(weightKIndex);
        auto perTokenKDim = GetInputTensor(gmmParams_.perTokenScaleOptional, i)->GetViewShape().GetDim(xKIndex);
        auto scaleNDim = GetInputTensor(gmmParams_.scaleOptional, i)->GetViewShape().GetDim(weightNIndex);
        auto perTokenMDim = GetInputTensor(gmmParams_.perTokenScaleOptional, i)->GetViewShape().GetDim(xMIndex);
        CHECK_COND(xMDim == perTokenMDim, ACLNN_ERR_PARAM_INVALID,
                   "When quantification mode is G-B quantification, the M value in x[%ld] and \
perTokenScale[%ld] should be consistent.",
                   xMDim, perTokenMDim);
        CHECK_COND(scaleNDim == (weightNDim + PERTILE_GROUP_SIZE - 1) / PERTILE_GROUP_SIZE, ACLNN_ERR_PARAM_INVALID,
                   "When quantification mode is G-B quantification, the N value in \
scaleOptional [%ld] must be equal to the N value in weight [%ld] divided by 128.",
                   scaleNDim, weightNDim);
        if (gmmParams_.groupType == SPLIT_M) {
            int64_t expectScaleKValue = (weightKDim + PERTILE_GROUP_SIZE - 1) / PERTILE_GROUP_SIZE;
            CHECK_COND(perTokenKDim == scaleKDim && scaleKDim == expectScaleKValue, ACLNN_ERR_PARAM_INVALID,
                       "When quantification mode is G-B quantification, and groupType is 0 (split M), the K dim of \
perTokenScaleOptional [%ld] should equal the K dim of scaleOptional [%ld], and its value should be equal to the K dim \
of weight [%ld] divided by 128, rounded up to the next integer.",
                       perTokenKDim, scaleKDim, weightKDim);
        } else {
            int64_t expectScaleKValue =
                (weightKDim / PERTILE_GROUP_SIZE) + gmmParams_.groupTensorOptional->GetViewShape().GetDim(0);
            CHECK_COND(perTokenKDim == scaleKDim && scaleKDim == expectScaleKValue, ACLNN_ERR_PARAM_INVALID,
                       "When quantification mode is G-B quantification, and groupType is 2 (split K), the K dim of \
perTokenScaleOptional [%ld] should equal the K dim of scaleOptional [%ld], its value must be equal to the K dim of \
weight [%ld] divided by 128, plus the groupSize [%ld].",
                       perTokenKDim, scaleKDim, weightKDim, gmmParams_.groupTensorOptional->GetViewShape().GetDim(0));
        }
    }
    return ACLNN_SUCCESS;
}

template <typename T>
aclnnStatus AclnnGroupedMatmulDAV3510Checker<T>::CheckGroupedMatmulPerTile() const
{
    CHECK_COND(CheckGroupedMatmulPerGroupDim() == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
               "CheckGroupedMatmulPerGroupDim failed");
    bool transposeScale = IsTransposeLastTwoDims(GetInputTensor(gmmParams_.scaleOptional));
    bool transposePerTokenScale = IsTransposeLastTwoDims(GetInputTensor(gmmParams_.perTokenScaleOptional));
    CHECK_COND(transposeScale == gmmParams_.transposeWeight ||
                   LastTwoDimValueIsOne(GetInputTensor(gmmParams_.scaleOptional)),
               ACLNN_ERR_PARAM_INVALID,
               "When quantification mode is G-B quantification, the transposition of scale/weight should be equal, \
but actual transpositions are %s/%s.",
               transposeScale ? "true" : "false", gmmParams_.transposeWeight ? "true" : "false");
    for (size_t i = 0; i < GetInputTensorSize(gmmParams_.weight); ++i) {
        auto weightDimNumber = GetInputTensor(gmmParams_.weight, i)->GetViewShape().GetDimNum();
        auto groupNum = gmmParams_.groupTensorOptional->GetViewShape().GetDim(0);
        auto weightKDim =
            GetInputTensor(gmmParams_.weight, i)->GetViewShape().GetDim(weightDimNumber - LAST_TWO_DIM_INDEX);
        auto weightNDim = GetInputTensor(gmmParams_.weight, i)->GetViewShape().GetDim(weightDimNumber - 1);
        auto perTokenMDim = GetInputTensor(gmmParams_.perTokenScaleOptional, i)->GetViewShape().GetDim(0);
        auto xMDim = GetInputTensor(gmmParams_.x, i)->GetViewShape().GetDim(0);
        CHECK_COND(
            transposePerTokenScale == gmmParams_.transposeX ||
                LastTwoDimValueIsOne(GetInputTensor(gmmParams_.perTokenScaleOptional)) ||
                IsSpecialperTileScene(groupNum, weightNDim, weightKDim, xMDim, perTokenMDim),
            ACLNN_ERR_PARAM_INVALID,
            "When quantification mode is G-B quantification, the transposition of perTokenScale/x should be equal, \
but actual transpositions are %s/%s.",
            transposePerTokenScale ? "true" : "false", gmmParams_.transposeX ? "true" : "false");
    }
    CHECK_COND(CheckGroupedMatmulPerTileShape() == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
               "CheckGroupedMatmulPerTileShape failed");
    return ACLNN_SUCCESS;
}

template <typename T>
aclnnStatus AclnnGroupedMatmulDAV3510Checker<T>::CheckFp8Hif8QuantParams() const
{
    CHECK_COND(gmmParams_.biasOptional == nullptr, ACLNN_ERR_PARAM_INVALID, "Float8/hifloat8 does not support bias.");
    CHECK_RET(CheckNonMxQuantTransposeStatus() == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);

    DataType scaleDtype = GetInputTensor(gmmParams_.scaleOptional)->GetDataType();
    if (scaleDtype == DataType::DT_UINT64 || scaleDtype == DataType::DT_INT64) {
        CHECK_COND(gmmParams_.groupType == SPLIT_M, ACLNN_ERR_PARAM_INVALID,
                   "In float8/hifloat8 quant case, when the %s dtype is uint64 or int64, only groupType 0 (split M) is supported.",
                   scaleName_.c_str());
    } else {
        CHECK_COND(gmmParams_.groupType != SPLIT_N, ACLNN_ERR_PARAM_INVALID,
                   "In float8/hifloat8 quant case, when the %s dtype is float32, only groupType 0 \
(split M) and  groupType 2 (split K) are supported.",
                   scaleName_.c_str());
    }

    if (gmmParams_.perTokenScaleOptional != nullptr) {
        DataType perTokenScaleDtype = GetInputTensor(gmmParams_.perTokenScaleOptional)->GetDataType();
        CHECK_COND(perTokenScaleDtype == DataType::DT_FLOAT, ACLNN_ERR_PARAM_INVALID,
                   "The %s dtype should be float32 in float8/hifloat8 quant case, but actual dtype is %s",
                   perTokenScaleName_.c_str(), op::ToString(perTokenScaleDtype).GetString());
        CHECK_COND(scaleDtype == DataType::DT_FLOAT, ACLNN_ERR_PARAM_INVALID,
                   "The %s dtype should be float32 when %s is not nullptr in \
float8/hifloat8 quant case, but actual dtype is %s",
                   scaleName_.c_str(), perTokenScaleName_.c_str(), op::ToString(scaleDtype).GetString());
    }
    DataType yDtype = GetInputTensor(gmmParams_.y)->GetDataType();
    CHECK_COND(yDtype == DataType::DT_FLOAT || yDtype == DataType::DT_BF16 || yDtype == DataType::DT_FLOAT16,
               ACLNN_ERR_PARAM_INVALID,
               "Expect y dtype to be float32, float16 or bfloat16 in float8/hifloat8 quant case, but actual dtype is %s",
               op::ToString(yDtype).GetString());
    if (IsPerTileQuantMode()) {
        CHECK_RET(CheckGroupedMatmulPerTile() == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
        return ACLNN_SUCCESS;
    }
    CHECK_RET(CheckNonPerGroupQuantDim() == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckNonPerGroupQuantShape() == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

template <typename T>
aclnnStatus AclnnGroupedMatmulDAV3510Checker<T>::CheckFp8Params(const DataType &scaleDtype) const
{
    if (scaleDtype == DataType::DT_FLOAT8_E8M0) {
        return CheckGroupedMatmulMxfp8();
    } else if (scaleDtype == DataType::DT_FLOAT || scaleDtype == DataType::DT_UINT64 ||
               scaleDtype == DataType::DT_INT64) {
        return CheckFp8Hif8QuantParams();
    } else {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Unsupported quant scale dtype %s with float8 case",
                op::ToString(scaleDtype).GetString());
        return ACLNN_ERR_PARAM_INVALID;
    }
}

template <typename T>
aclnnStatus AclnnGroupedMatmulDAV3510Checker<T>::CheckFp4Params(const DataType &scaleDtype) const
{
    CHECK_COND(gmmParams_.groupType == SPLIT_M, ACLNN_ERR_PARAM_INVALID,
               "In mxfp4 quant mode, mxfp4 case only supports groupType 0 (split M), but actual groupType is %ld",
               gmmParams_.groupType);
    if (scaleDtype == DataType::DT_FLOAT8_E8M0) {
        return CheckGroupedMatmulMxfp4();
    }
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Unsupported quant scale dtype %s with float4 case",
            op::ToString(scaleDtype).GetString());
    return ACLNN_ERR_PARAM_INVALID;
}

template <typename T>
aclnnStatus AclnnGroupedMatmulDAV3510Checker<T>::CheckInputAndOutputDtypeForV3Version() const
{
    DataType xDtype = gmmParams_.xDtype;
    DataType weightDtype = GetInputTensor(gmmParams_.weight)->GetDataType();
    DataType scaleDtype = GetInputTensor(gmmParams_.scaleOptional)->GetDataType();
    DataType yDtype = GetInputTensor(gmmParams_.y)->GetDataType();
    CHECK_COND(xDtype == DataType::DT_INT8 && weightDtype == DataType::DT_INT8, ACLNN_ERR_PARAM_INVALID,
               "In quant case, both x dtype and weight dtype should be int8, but the actual are %s and %s.",
               op::ToString(xDtype).GetString(), op::ToString(weightDtype).GetString());
    CHECK_COND(scaleDtype == DataType::DT_UINT64 || scaleDtype == DataType::DT_INT64, ACLNN_ERR_PARAM_INVALID,
               "In quant case, scale dtype should be uint64 or int64, but the actual is %s.",
               op::ToString(scaleDtype).GetString());
    CHECK_COND(yDtype == DataType::DT_INT8, ACLNN_ERR_PARAM_INVALID,
               "In quant case, y dtype should be int8, but the actual is %s.",
               op::ToString(yDtype).GetString());
    if (gmmParams_.biasOptional != nullptr) {
        DataType biasDtype = (*gmmParams_.biasOptional)[0]->GetDataType();
        CHECK_COND(biasDtype == DataType::DT_INT32, ACLNN_ERR_PARAM_INVALID,
                   "In quant case, bias dtype should be int32, but the actual is %s.",
                   op::ToString(biasDtype).GetString());
    }
    return ACLNN_SUCCESS;
}

template <typename T>
aclnnStatus AclnnGroupedMatmulDAV3510Checker<T>::CheckInputShapeForV3Version() const
{
    auto groupNum = gmmParams_.groupTensorOptional->GetViewShape().GetDim(0);
    auto scaleDimNumber = GetInputTensor(gmmParams_.scaleOptional)->GetViewShape().GetDimNum();
    auto scaleNDim = GetInputTensor(gmmParams_.scaleOptional)->GetViewShape().GetDim(scaleDimNumber - 1);
    auto weightNIndex = GetInputTensor(gmmParams_.weight)->GetViewShape().GetDimNum() - 1;
    auto weightNDim = GetInputTensor(gmmParams_.weight)->GetViewShape().GetDim(weightNIndex);
    CHECK_COND(GetInputTensor(gmmParams_.scaleOptional)->GetViewShape().GetDim(0) == groupNum,
               ACLNN_ERR_PARAM_INVALID,
               "In quant case, the first dim of %s[%ld] should be equal to that of %s[%ld].",
               scaleName_.c_str(), GetInputTensor(gmmParams_.scaleOptional)->GetViewShape().GetDim(0),
               groupTensorName_.c_str(), groupNum);
    CHECK_COND(scaleNDim == weightNDim, ACLNN_ERR_PARAM_INVALID,
               "In quant case, the N dim of %s[%ld] should be equal to that of %s[%ld].",
               scaleName_.c_str(), scaleNDim, weightName_.c_str(), weightNDim);
    if (gmmParams_.biasOptional != nullptr) {
        auto biasGDimValue = GetInputTensor(gmmParams_.biasOptional)->GetViewShape().GetDim(0);
        auto biasNDimValue = GetInputTensor(gmmParams_.biasOptional)->GetViewShape().GetDim(1);
        CHECK_COND(biasGDimValue == groupNum, ACLNN_ERR_PARAM_INVALID,
                   "The group dim of %s[%ld] and group number[%ld] should be equal.", biasName_.c_str(), biasGDimValue,
                   groupNum);
        CHECK_COND(biasNDimValue == weightNDim, ACLNN_ERR_PARAM_INVALID,
                   "The n dim of %s[%ld] and n dim of %s[%ld] should be equal.", biasName_.c_str(), biasNDimValue,
                   weightName_.c_str(), weightNDim);
    }
    return ACLNN_SUCCESS;
}

template <typename T>
aclnnStatus AclnnGroupedMatmulDAV3510Checker<T>::CheckInputParamsForV3Version() const
{
    CHECK_COND(CheckInputAndOutputDtypeForV3Version() == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
               "CheckInputAndOutputDtypeForV3Version failed.");
    CHECK_COND(CheckInputShapeForV3Version() == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
               "CheckInputShapeForV3Version failed.");
    return ACLNN_SUCCESS;
}

template <typename T>
aclnnStatus AclnnGroupedMatmulDAV3510Checker<T>::CheckGroupedMatmulDAV3510() const
{
    DataType xDtype = gmmParams_.xDtype;
    DataType weightDtype = GetInputTensor(gmmParams_.weight)->GetDataType();
    DataType yDtype = GetInputTensor(gmmParams_.y)->GetDataType();
    if (yDtype != DataType::DT_INT32) {
        CHECK_COND(gmmParams_.scaleOptional != nullptr, ACLNN_ERR_PARAM_INVALID,
                   "In quant case, scaleOptional should not be nullptr when the output dtype is not int32.");
    }
    CHECK_COND(gmmParams_.groupTensorOptional != nullptr, ACLNN_ERR_PARAM_INVALID,
               "In quant case, groupListOptional should not be nullptr.");
    CHECK_COND(gmmParams_.offsetOptional == nullptr, ACLNN_ERR_PARAM_INVALID, "Quant case does not support offset.");
    CHECK_COND(gmmParams_.groupType != SPLIT_N, ACLNN_ERR_PARAM_INVALID,
               "Quant case does not support groupType 1 (split N).");
    CHECK_COND(GetInputTensorSize(gmmParams_.x) == 1 && GetInputTensorSize(gmmParams_.weight) == 1 &&
                   GetInputTensorSize(gmmParams_.y) == 1,
               ACLNN_ERR_PARAM_INVALID,
               "In quant case, the size of x, weight and y should all be 1, but actual sizes are %zu, %zu and %zu.",
               GetInputTensorSize(gmmParams_.x), GetInputTensorSize(gmmParams_.weight),
               GetInputTensorSize(gmmParams_.y));
    CHECK_RET(CheckQuantCasesFormat() == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    if (GetInputTensor(gmmParams_.weight)->GetStorageFormat() == op::Format::FORMAT_FRACTAL_NZ){
        CHECK_RET(CheckWeightNzSpecialParams() == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    }
    CHECK_RET(CheckGeneralQuantShape() == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);

    if (gmmParams_.apiVersion == gmm::GMMApiVersion::V3) {
        CHECK_COND(CheckInputParamsForV3Version() == ACLNN_SUCCESS,
        ACLNN_ERR_PARAM_INVALID, "CheckInputParamsForV3Version failed.");
    }
    DataType scaleDtype = DataType::DT_UINT64;
    if (gmmParams_.scaleOptional != nullptr) {
        scaleDtype = GetInputTensor(gmmParams_.scaleOptional)->GetDataType();
    }
    if (xDtype == DataType::DT_INT8 && weightDtype == DataType::DT_INT8) {
        return CheckInt8QuantParams();
    } else if (xDtype == DataType::DT_HIFLOAT8 && weightDtype == DataType::DT_HIFLOAT8) {
        CHECK_COND(
            scaleDtype == DataType::DT_UINT64 || scaleDtype == DataType::DT_FLOAT || scaleDtype == DataType::DT_INT64,
            ACLNN_ERR_PARAM_INVALID,
            "In hifloat8 quant case, scale dtype should be uint64, int64 or float32, but actual dtype is %s",
            op::ToString(scaleDtype).GetString());
        return CheckFp8Hif8QuantParams();
    } else if ((xDtype == DataType::DT_FLOAT8_E4M3FN || xDtype == DataType::DT_FLOAT8_E5M2) &&
                (weightDtype == DataType::DT_FLOAT8_E4M3FN || weightDtype == DataType::DT_FLOAT8_E5M2)) {
        return CheckFp8Params(scaleDtype);
    } else if (xDtype == DataType::DT_FLOAT4_E2M1 && weightDtype == DataType::DT_FLOAT4_E2M1) {
        return CheckFp4Params(scaleDtype);
    } else {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Quant case with x dtype %s and weight dtype %s is not supported.",
                op::ToString(xDtype).GetString(), op::ToString(weightDtype).GetString());
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}