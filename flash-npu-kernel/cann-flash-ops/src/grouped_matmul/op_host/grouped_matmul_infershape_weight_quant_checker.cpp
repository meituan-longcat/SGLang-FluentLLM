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
 * \file grouped_matmul_infershape_weight_quant_checker.cpp
 * \brief
 */
#include "grouped_matmul_infershape_weight_quant_checker.h"
#include "grouped_matmul_infershape_common_util.h"

namespace ops {
static const std::map<ge::DataType, std::unordered_set<ge::DataType>> BIAS_TYPE_SUPPORT_MAP = {
    {ge::DT_FLOAT16, {ge::DT_FLOAT16}},
    {ge::DT_BF16, {ge::DT_BF16, ge::DT_FLOAT}},
    {ge::DT_INT8, {ge::DT_FLOAT}},
    {ge::DT_FLOAT8_E4M3FN, {ge::DT_BF16, ge::DT_FLOAT16}}};
static const std::unordered_set<ge::DataType> FP8_SUPPORT_SET = {ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2,
                                                                 ge::DT_HIFLOAT8};
const int64_t UNKNOWN_SHAPE_VALUE = -1;
const int64_t SHAPE_UNKNOWN_DIM_NUM = -2;
const size_t ANTIQUANT_PARAM_DIM_NUM_PER_GROUP_SINGLE = 3;
const size_t ANTIQUANT_PARAM_DIM_NUM_MX = 4;
const size_t OPTIONAL_PARAM_DIM_NUM_DEFAULT_SINGLE = 2;
const int64_t B4_NUMS_IN_B32 = 8;
const int64_t MX_GROUP_SIZE = 32;

static bool inline IsNonEmpty(const gert::Shape *shape)
{
    return (shape != nullptr && !(shape->GetDimNum() == 1 && shape->GetDim(0) == 0));
}

bool GroupedMatmulWeightQuantChecker::IsA16MxFp4NZ(const ge::DataType xDtype, const ge::DataType weightDtype) const
{
    return (xDtype == ge::DT_FLOAT16 || xDtype == ge::DT_BF16) &&
           (weightDtype == ge::DT_FLOAT4_E2M1 || weightDtype == ge::DT_FLOAT);
}

bool GroupedMatmulWeightQuantChecker::IsMxA8W4NZ(const ge::DataType xDtype, const ge::DataType weightDtype) const
{
    return xDtype == ge::DT_FLOAT8_E4M3FN && (weightDtype == ge::DT_FLOAT4_E2M1 || weightDtype == ge::DT_FLOAT);
}

bool GroupedMatmulWeightQuantChecker::IsS8S4NZ(const ge::DataType xDtype, const ge::DataType weightDtype) const
{
    return xDtype == ge::DT_INT8 && (weightDtype == ge::DT_INT4 || weightDtype == ge::DT_INT32);
}

bool GroupedMatmulWeightQuantChecker::IsA16W8(const ge::DataType xDtype, const ge::DataType weightDtype) const
{
    return (xDtype == ge::DT_FLOAT16 || xDtype == ge::DT_BF16) && weightDtype == ge::DT_INT8;
}

bool GroupedMatmulWeightQuantChecker::IsA16F8(const ge::DataType xDtype, const ge::DataType weightDtype) const
{
    return (xDtype == ge::DT_FLOAT16 || xDtype == ge::DT_BF16) &&
           FP8_SUPPORT_SET.find(weightDtype) != FP8_SUPPORT_SET.end();
}

bool GroupedMatmulWeightQuantChecker::IsA16W4(const ge::DataType xDtype, const ge::DataType weightDtype) const
{
    return (xDtype == ge::DT_FLOAT16 || xDtype == ge::DT_BF16) &&
           (weightDtype == ge::DT_INT4 || weightDtype == ge::DT_INT32);
}

ge::graphStatus GroupedMatmulWeightQuantChecker::GetXandWeightDtype(const gert::InferShapeContext *context)
{
    auto xDesc = context->GetDynamicInputDesc(GMM_INDEX_IN_X, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xDesc);
    xDtype_ = xDesc->GetDataType();

    auto weightDesc = context->GetDynamicInputDesc(GMM_INDEX_IN_WEIGHT, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightDesc);
    weightDtype_ = weightDesc->GetDataType();

    OP_LOGD(context->GetNodeName(), "Weight quant case xDtype is [%s], weightDtype is [%s]. ",
            ge::TypeUtils::DataTypeToAscendString(xDtype_).GetString(),
            ge::TypeUtils::DataTypeToAscendString(weightDtype_).GetString());

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedMatmulWeightQuantChecker::GetXAndWeightDimValue(const gert::InferShapeContext *context,
                                                                       const GMMAttrs &gmmAttrs)
{
    OP_CHECK_IF(GetXandWeightDtype(context) != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "GetXandWeightDtype failed."), return ge::GRAPH_FAILED);

    auto xShape = context->GetDynamicInputShape(GMM_INDEX_IN_X, 0);
    auto weightShape = context->GetDynamicInputShape(GMM_INDEX_IN_WEIGHT, 0);
    OP_CHECK_IF(!(IsNonEmpty(xShape) && IsNonEmpty(weightShape)),
                OP_LOGE(context->GetNodeName(), "The 1st tensor of tensor list x and weight cannot be empty."),
                return ge::GRAPH_FAILED);

    xdimNum_ = xShape->GetDimNum();
    weightdimNum_ = weightShape->GetDimNum();
    OP_CHECK_IF(xdimNum_ < GMM_MIN_FM_DIM,
                OP_LOGE(context->GetNodeName(),
                        "The dim num of x's 1st tensor should be greater than 1, but it is [%zu].", xdimNum_),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(weightdimNum_ < GMM_MIN_WEIGHT_DIM,
                OP_LOGE(context->GetNodeName(),
                        "The dim num of weight's 1st tensor should be greater than 1, but it is [%zu].", weightdimNum_),
                return ge::GRAPH_FAILED);

    if (gmmAttrs.groupType == GMM_NO_SPLIT) {
        return ge::GRAPH_SUCCESS;
    }

    xMDim_ = gmmAttrs.transposeX ? xShape->GetDim(xdimNum_ - 1) : xShape->GetDim(xdimNum_ - PENULTIMATE_DIM);
    xKDim_ = gmmAttrs.transposeX ? xShape->GetDim(xdimNum_ - PENULTIMATE_DIM) : xShape->GetDim(xdimNum_ - 1);
    weightKDim_ = gmmAttrs.transposeWeight ? weightShape->GetDim(weightdimNum_ - 1)
                                           : weightShape->GetDim(weightdimNum_ - PENULTIMATE_DIM);
    weightNDim_ = gmmAttrs.transposeWeight ? weightShape->GetDim(weightdimNum_ - PENULTIMATE_DIM)
                                           : weightShape->GetDim(weightdimNum_ - 1);

    // 1个float32/int32表示8个float4_e2m1/int4，推导shape时，尾轴扩大8倍
    if ((weightDtype_ == ge::DT_FLOAT || weightDtype_ == ge::DT_INT32) && weightKDim_ > 0 && xKDim_ > 0) {
        bool transWeightB32 = false;
        if (!gmmAttrs.transposeWeight) {
            transWeightB32 = weightKDim_ * B4_NUMS_IN_B32 == xKDim_;
        }
        if (gmmAttrs.transposeWeight || transWeightB32) {
            weightKDim_ = weightKDim_ * B4_NUMS_IN_B32;
        } else {
            weightNDim_ = weightNDim_ * B4_NUMS_IN_B32;
        }
    }

    OP_LOGD(context->GetNodeName(), "xMDim = [%lld], xKDim = [%lld], weightKDim = [%lld], weightNDim = [%lld]. ",
            xMDim_, xKDim_, weightKDim_, weightNDim_);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedMatmulWeightQuantChecker::CheckShapeForXAndWeight(const gert::InferShapeContext *context) const
{
    OP_CHECK_IF(xKDim_ != weightKDim_,
                OP_LOGE(context->GetNodeName(),
                        "The k dim of x should be equal to the k dim of weight, "
                        "but x's k is [%ld] and weight's k is [%ld].",
                        xKDim_, weightKDim_),
                return ge::GRAPH_FAILED);
    // 动态图走到此处时不允许出现非确定值
    OP_CHECK_IF(weightNDim_ < 0,
                OP_LOGE(context->GetNodeName(),
                        "The n dim value should not be negative, but the actual value is [%ld].", weightNDim_),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(weightKDim_ < 0,
                OP_LOGE(context->GetNodeName(),
                        "The k dim value should not be negative, but the actual value is [%ld].", weightKDim_),
                return ge::GRAPH_FAILED);
    if (weightDtype_ == ge::DT_FLOAT4_E2M1 || weightDtype_ == ge::DT_FLOAT || IsS8S4NZ(xDtype_, weightDtype_)) {
        // A16MxF4/MxA8W4/S8S4校验32B对齐
        OP_CHECK_IF(
            !((weightNDim_ % GMM_N_K_ALIGN_VALUE_WEIGHT_QUANT_4BIT == 0) &&
              (weightKDim_ % GMM_N_K_ALIGN_VALUE_WEIGHT_QUANT_4BIT == 0)),
            OP_LOGE(context->GetNodeName(),
                    "The value of dim n, k should be an integer multiple of [%ld], but actual n is [%ld], k is [%ld].",
                    GMM_N_K_ALIGN_VALUE_WEIGHT_QUANT_4BIT, weightNDim_, weightKDim_),
            return ge::GRAPH_FAILED);
    }
    auto weightShape = context->GetDynamicInputShape(GMM_INDEX_IN_WEIGHT, 0);
    OP_CHECK_IF(xdimNum_ != GMM_MIN_FM_DIM || weightdimNum_ != GMM_SPLIT_M_SINGLE_WEIGHT_DIM,
                OP_LOGE(context->GetNodeName(),
                        "When split m, x dim num should be 2, weight dim num should be 3, "
                        "but the actual x dim num is [%zu], actual weight dim num is [%zu].",
                        xdimNum_, weightdimNum_),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(weightShape->GetDim(0) != groupNum_,
                OP_LOGE(context->GetNodeName(),
                        "When split m, 1st dim value of weight should be g, "
                        "which is [%ld], but the actual value is [%ld].",
                        groupNum_, weightShape->GetDim(0)),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedMatmulWeightQuantChecker::CheckTensorDimEqualOne(const gert::InferShapeContext *context,
                                                                        const gert::Shape *shape,
                                                                        const std::string paramName,
                                                                        const size_t index) const
{
    OP_CHECK_IF(
        shape == nullptr,
        OP_LOGE(context->GetNodeName(), "%s Shape[%zu] is null, which is not supported.", paramName.c_str(), index),
        return ge::GRAPH_FAILED);
    size_t dimNum = shape->GetDimNum();
    OP_CHECK_IF(
        dimNum != 1,
        OP_LOGE(context->GetNodeName(), "%s[%zu] dimNum is %zu, but only support 1.", paramName.c_str(), index, dimNum),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedMatmulWeightQuantChecker::CheckDimNumNoSplit(const gert::InferShapeContext *context,
                                                                    const GMMInputParamsInfo &paramsInputInfo) const
{
    const size_t &tensorListLength = paramsInputInfo.numX;
    auto wShape = context->GetDynamicInputShape(GMM_INDEX_IN_WEIGHT, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context, wShape);
    // check dimension
    for (size_t i = 0; i < tensorListLength; ++i) {
        auto xShape = context->GetDynamicInputShape(GMM_INDEX_IN_X, i);
        OP_CHECK_IF(xShape == nullptr, OP_LOGE(context->GetNodeName(), "x[%zu] is null, which is not supported.", i),
                    return ge::GRAPH_FAILED);
        wShape = context->GetDynamicInputShape(GMM_INDEX_IN_WEIGHT, i);
        OP_CHECK_NULL_WITH_CONTEXT(context, wShape);
        size_t weightDimNum = wShape->GetDimNum();
        OP_CHECK_IF(weightDimNum != GMM_SEPARATED_WEIGHT_DIM,
                    OP_LOGE(context->GetNodeName(),
                            "weight[%zu] dimNum is %zu, but only support 2 when weight separated.", i, weightDimNum),
                    return ge::GRAPH_FAILED);
        // 校验 x 中每个tensor的维度必须在[2,6]之间
        size_t xDimNum = xShape->GetDimNum();
        OP_CHECK_IF(xDimNum > GMM_MAX_FM_DIM || xDimNum < GMM_MIN_FM_DIM,
                    OP_LOGE(context->GetNodeName(), "x[%zu] dimNum is %zu, but only support 2-6.", i, xDimNum),
                    return ge::GRAPH_FAILED);
        // 检测 bias antiquantScale antiquantOffset 的每个tensor的dim都需要为1
        if (hasBias_) {
            auto biasShape = context->GetDynamicInputShape(GMM_INDEX_IN_BIAS, i);
            OP_CHECK_IF(CheckTensorDimEqualOne(context, biasShape, "bias", i) != ge::GRAPH_SUCCESS,
                        OP_LOGE(context->GetNodeName(), "CheckTensorDimEqualOne is failed."), return ge::GRAPH_FAILED);
        }
        if (hasAntiquantOffset_) {
            auto antiquantOffsetShape = context->GetDynamicInputShape(GMM_INDEX_IN_ANTIQUANT_OFFSET, i);
            OP_CHECK_IF(
                CheckTensorDimEqualOne(context, antiquantOffsetShape, "antiquantOffset", i) != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "CheckTensorDimEqualOne is failed."), return ge::GRAPH_FAILED);
        }
        auto antiquantScaleShape = context->GetDynamicInputShape(GMM_INDEX_IN_ANTIQUANT_SCALE, i);
        OP_CHECK_IF(CheckTensorDimEqualOne(context, antiquantScaleShape, "antiquantScale", i) != ge::GRAPH_SUCCESS,
                    OP_LOGE(context->GetNodeName(), "CheckTensorDimEqualOne is failed."), return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedMatmulWeightQuantChecker::CheckXWeightYGroupSizeMultiScenario(
    const gert::InferShapeContext *context, const GMMInputParamsInfo &paramsInputInfo) const
{
    const size_t &xSize = paramsInputInfo.numX;
    const size_t &weightSize = paramsInputInfo.numWeight;
    size_t numY = context->GetComputeNodeOutputNum();
    // check group size
    OP_CHECK_IF(xSize != numY,
                OP_LOGE(context->GetNodeName(), "When y is separated, size of x %zu should equal to size of y %zu.",
                        xSize, numY),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(xSize != weightSize,
                OP_LOGE(context->GetNodeName(),
                        "When weight is separated, size of w %zu should equal to size of x %zu.", weightSize, xSize),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedMatmulWeightQuantChecker::CheckTensorNDimMultiScenario(const gert::InferShapeContext *context,
                                                                              const GMMInputParamsInfo &paramsInputInfo,
                                                                              const size_t wNDimIdx,
                                                                              const int64_t weightNDimValue,
                                                                              const size_t index) const
{
    // 检验weight的n轴和antiquantScale的n轴一致
    auto antiquantScaleShape = context->GetDynamicInputShape(GMM_INDEX_IN_ANTIQUANT_SCALE, index);
    OP_CHECK_NULL_WITH_CONTEXT(context, antiquantScaleShape);
    int64_t antiquantScaleNDim = antiquantScaleShape->GetDim(0);
    OP_CHECK_IF(antiquantScaleNDim != weightNDimValue,
                OP_LOGE(context->GetNodeName(),
                        "weight[%zu] dim %zu value %ld should equal to antiquantScale[%zu] dim 0 value %ld.", index,
                        wNDimIdx, weightNDimValue, index, antiquantScaleNDim),
                return ge::GRAPH_FAILED);

    if (hasBias_) {
        // 检验weigh的n轴和bias的n轴一致
        auto biasShape = context->GetDynamicInputShape(GMM_INDEX_IN_BIAS, index);
        OP_CHECK_NULL_WITH_CONTEXT(context, biasShape);
        int64_t biasNDim = biasShape->GetDim(0);
        OP_CHECK_IF(
            biasNDim != weightNDimValue,
            OP_LOGE(context->GetNodeName(), "weight[%zu] dim %zu value %ld should equal to bias[%zu] dim 0 value %ld.",
                    index, wNDimIdx, weightNDimValue, index, biasNDim),
            return ge::GRAPH_FAILED);
    }
    if (hasAntiquantOffset_) {
        // 检验weight的n轴和antiquantOffset的n轴一致
        auto antiquantOffsetShape = context->GetDynamicInputShape(GMM_INDEX_IN_ANTIQUANT_OFFSET, index);
        OP_CHECK_NULL_WITH_CONTEXT(context, antiquantOffsetShape);
        int64_t antiquantOffsetNDim = antiquantOffsetShape->GetDim(0);
        OP_CHECK_IF(antiquantOffsetNDim != weightNDimValue,
                    OP_LOGE(context->GetNodeName(),
                            "weight[%zu] dim %zu value %ld should equal to antiquantOffset[%zu] dim 0 value %ld.",
                            index, wNDimIdx, weightNDimValue, index, antiquantOffsetNDim),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedMatmulWeightQuantChecker::CheckCaseMultiScenario(const gert::InferShapeContext *context,
                                                                        const GMMAttrs &gmmAttrs,
                                                                        const GMMInputParamsInfo &paramsInputInfo) const
{
    const size_t &xSize = paramsInputInfo.numX;
    // check group size
    OP_CHECK_IF(CheckXWeightYGroupSizeMultiScenario(context, paramsInputInfo) != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "The size of X, Y and weight are not equal."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckTensorListSizeMultiScenario(context, paramsInputInfo) != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "CheckTensorListSizeMultiScenario failed."), return ge::GRAPH_FAILED);
    // check dimension
    OP_CHECK_IF(CheckDimNumNoSplit(context, paramsInputInfo) != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "Dim num of tensor in tensor lists or grouplist is invalid."),
                return ge::GRAPH_FAILED);

    auto xShape = context->GetDynamicInputShape(GMM_INDEX_IN_X, 0);
    auto wShape = context->GetDynamicInputShape(GMM_INDEX_IN_WEIGHT, 0);
    size_t wKDimIdx = gmmAttrs.transposeWeight ? 1UL : 0UL;
    size_t wNDimIdx = gmmAttrs.transposeWeight ? 0UL : 1UL;

    int64_t weightKDimValue = wShape->GetDim(wKDimIdx);
    int64_t weightNDimValue = wShape->GetDim(wNDimIdx);

    for (size_t i = 0; i < xSize; i++) {
        xShape = context->GetDynamicInputShape(GMM_INDEX_IN_X, i);
        size_t xDimNum = xShape->GetDimNum();
        int64_t xKDimValue = xShape->GetDim(xDimNum - 1);  // x always is not transposed

        wShape = context->GetDynamicInputShape(GMM_INDEX_IN_WEIGHT, i);
        weightKDimValue = wShape->GetDim(wKDimIdx);
        weightNDimValue = wShape->GetDim(wNDimIdx);
        // 校验M轴和batch轴大于等于0
        for (size_t j = 0; j < xDimNum - 1; j++) {
            int64_t xNDimValue = xShape->GetDim(j);
            OP_CHECK_IF(xNDimValue < 0,
                        OP_LOGE(context->GetNodeName(), "x[%zu] dim %zu value %ld should be more than or equal to 0.",
                                i, j, xNDimValue),
                        return ge::GRAPH_FAILED);
        }
        // 校验K轴和N轴非不确定值
        OP_CHECK_IF(
            xKDimValue < 0,
            OP_LOGE(context->GetNodeName(), "x[%zu] dim %zu value %ld should not be negative.", i, xDimNum - 1, xKDimValue),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF(weightNDimValue < 0,
                    OP_LOGE(context->GetNodeName(), "w[%zu] dim %zu value %ld should not be negative.", i, wNDimIdx,
                            weightNDimValue),
                    return ge::GRAPH_FAILED);
        // 校验X和weight矩阵的K轴
        OP_CHECK_IF(
            xKDimValue != weightKDimValue,
            OP_LOGE(context->GetNodeName(), "x[%zu] dim %zu value %ld should equal to weight[%zu] dim 0 value %ld.", i,
                    xDimNum - 1, xKDimValue, i, weightKDimValue),
            return ge::GRAPH_FAILED);
        // 校验 antiquantScale bias antiquantOffset 每个tensor的中n轴的大小
        OP_CHECK_IF(CheckTensorNDimMultiScenario(context, paramsInputInfo, wNDimIdx, weightNDimValue, i),
                    OP_LOGE(context->GetNodeName(), "CheckTensorNDimMultiScenario is failed."),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedMatmulWeightQuantChecker::CheckScenarioValid(const gert::InferShapeContext *context,
                                                                    const GMMAttrs &gmmAttrs) const
{
    auto xSecondTensorShape = context->GetDynamicInputShape(GMM_INDEX_IN_X, 1);
    auto weightSecondTensorShape = context->GetDynamicInputShape(GMM_INDEX_IN_WEIGHT, 1);
    // 检验groupType
    OP_CHECK_IF((gmmAttrs.groupType != GMM_SPLIT_M) && (gmmAttrs.groupType != GMM_NO_SPLIT),
                OP_LOGE(context->GetNodeName(), "Invalid groupType, which can only be 0 or -1, but it is [%ld].",
                        gmmAttrs.groupType),
                return ge::GRAPH_FAILED);

    if (gmmAttrs.groupType == GMM_SPLIT_M) {  // single/single/single Scenario
        OP_CHECK_IF(IsNonEmpty(xSecondTensorShape),
                    OP_LOGE(context->GetNodeName(), "The second tensor of tensor list x is not empty."),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(IsNonEmpty(weightSecondTensorShape),
                    OP_LOGE(context->GetNodeName(), "The second tensor of tensor list weight is not empty."),
                    return ge::GRAPH_FAILED);

        // check split item value valid
        OP_CHECK_IF(gmmAttrs.splitItem != GMM_X_SEPARATED && gmmAttrs.splitItem != GMM_NO_SEPARATED,
                    OP_LOGE(context->GetNodeName(),
                            "Invalid splitItem, which can only be one of 2 or 3, but it is [%ld].", gmmAttrs.splitItem),
                    return ge::GRAPH_FAILED);
    } else {  // multi/multi/multi Scenario
        // check split item value valid
        OP_CHECK_IF(!IsA16W8(xDtype_, weightDtype_),
                    OP_LOGE(context->GetNodeName(),
                            "Multi/Multi/Multi Scenario is supported only when xDtype-weightDtype is fp16/bf16-int8."),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(gmmAttrs.splitItem != GMM_X_Y_SEPARATED && gmmAttrs.splitItem != GMM_Y_SEPARATED,
                    OP_LOGE(context->GetNodeName(),
                            "Invalid splitItem, which can only be one of 0 or 1, but it is [%ld].", gmmAttrs.splitItem),
                    return ge::GRAPH_FAILED);
    }
    // check activation is null
    OP_CHECK_IF(gmmAttrs.activeType != static_cast<int64_t>(GMMActType::GMM_ACT_TYPE_NONE),
                OP_LOGE(context->GetNodeName(), "Activation function is not supported in weight quant mode now."),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedMatmulWeightQuantChecker::CheckShapeForGrouplist(const gert::InferShapeContext *context,
                                                                        const gert::Shape *groupListShape) const
{
    OP_CHECK_IF(groupListShape->GetDimNum() != 1,
                OP_LOGE(context->GetNodeName(),
                        "In single-single-single scenario, "
                        "the groupList only support 1 dim num for now, but the actual dim num is [%zu].",
                        groupListShape->GetDimNum()),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(groupListShape->GetDim(0) <= 0,
                OP_LOGE(context->GetNodeName(),
                        "The groupList 1st dim value should be greater than 0, but the actual value is [%ld].",
                        groupListShape->GetDim(0)),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(groupListShape->GetDim(0) > GMM_MAX_GROUP_LIST_SIZE_TENSOR,
                OP_LOGE(context->GetNodeName(),
                        "Only support [%ld] groups at MAX for now, but the actual group number is [%ld].",
                        GMM_MAX_GROUP_LIST_SIZE_TENSOR, groupListShape->GetDim(0)),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedMatmulWeightQuantChecker::CheckPertokenScaleForA8W4(const gert::InferShapeContext *context) const
{
    auto tensorShape = context->GetDynamicInputShape(GMM_INDEX_IN_PERTOKEN_SCALE, 0);
    size_t tensorDimNum = tensorShape->GetDimNum();
    // S8S4的PerTokenScale维度为1，shape为(m)
    // MxA8W4的PerTokenScale维度为3，shape为(m, k/64, 2)
    size_t tensorDimNumExp = IsS8S4NZ(xDtype_, weightDtype_) ? 1 : 3;
    OP_CHECK_IF(tensorDimNum != tensorDimNumExp,
                OP_LOGE(context->GetNodeName(), "PertokenScale dim should be [%zu], but the actual dim num is [%zu].",
                        tensorDimNumExp, tensorDimNum),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(tensorShape->GetDim(0) != xMDim_,
                OP_LOGE(context->GetNodeName(),
                        "The shape of PertokenScale should be (m), which is (%ld), but the actual shape is (%ld).",
                        xMDim_, tensorShape->GetDim(0)),
                return ge::GRAPH_FAILED);
    // 2: MxA8W4 PertokenScaleShape:(m, k / MX_GROUPSIZE / 2, 2)
    if (IsMxA8W4NZ(xDtype_, weightDtype_)) {
        OP_CHECK_IF(
            tensorShape->GetDim(1) != weightKDim_ / MX_GROUP_SIZE / 2,
            OP_LOGE(
                context->GetNodeName(),
                "PerTokenScale shape should be (m, k/%ld, 2) when xDtype-weightDtype is fp8_e4m3-fp4_e2m1, which is "
                "(%ld, %ld, 2), but the actual shape is (%ld, %ld, %ld).",
                MX_GROUP_SIZE * 2, xMDim_, weightKDim_ / MX_GROUP_SIZE / 2, tensorShape->GetDim(0),
                tensorShape->GetDim(1), tensorShape->GetDim(2)),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedMatmulWeightQuantChecker::CheckShapeForTensorList(const gert::InferShapeContext *context,
                                                                         size_t gmm_index,
                                                                         const std::string &tensorType,
                                                                         const GMMAttrs &gmmAttrs) const
{
    // 校验单单单场景antiquantScale/antiquantOffset/bias/scale的shape
    // A16MxF4/S8S4校验antiquant params的Shape为(g, k/groupSize, n)/(g, n, k/groupsize)，维度数为3
    // MxA8W4校验antiquant params的Shape为（g, n, k / groupSize / 2, 2）
    // 其他场景及参数校验shape为(g, n)
    auto tensorShape = context->GetDynamicInputShape(gmm_index, 0);
    if (IsNonEmpty(tensorShape)) {
        size_t tensorDimNum = tensorShape->GetDimNum();
        size_t expectedDimNum = OPTIONAL_PARAM_DIM_NUM_DEFAULT_SINGLE;
        if ((gmm_index == GMM_INDEX_IN_ANTIQUANT_SCALE || gmm_index == GMM_INDEX_IN_ANTIQUANT_OFFSET) &&
            (IsA16MxFp4NZ(xDtype_, weightDtype_) || IsS8S4NZ(xDtype_, weightDtype_))) {
            expectedDimNum = ANTIQUANT_PARAM_DIM_NUM_PER_GROUP_SINGLE;
        } else if ((gmm_index == GMM_INDEX_IN_ANTIQUANT_SCALE) && IsMxA8W4NZ(xDtype_, weightDtype_)) {
            expectedDimNum = ANTIQUANT_PARAM_DIM_NUM_MX;
        }
        // check dim num
        OP_CHECK_IF(tensorDimNum != expectedDimNum,
                    OP_LOGE(context->GetNodeName(), "%s dim num should be [%zu], but the actual dim num is [%zu].",
                            tensorType.c_str(), expectedDimNum, tensorDimNum),
                    return ge::GRAPH_FAILED);

        // check shape
        OP_CHECK_IF(tensorShape->GetDim(0) != groupNum_,
                    OP_LOGE(context->GetNodeName(),
                            "The first dim of %s should be g, which is [%ld], but the actual value is [%ld].",
                            tensorType.c_str(), groupNum_, tensorShape->GetDim(0)),
                    return ge::GRAPH_FAILED);

        size_t tensorNDimIdx = tensorDimNum - 1;
        if (expectedDimNum == ANTIQUANT_PARAM_DIM_NUM_PER_GROUP_SINGLE && gmmAttrs.transposeWeight) {
            // per_group量化weight转置时antiquant params同步转置，shape为(g, n, k/groupsize)，n轴的索引为-2
            tensorNDimIdx = tensorDimNum - 2;
        } else if (expectedDimNum == ANTIQUANT_PARAM_DIM_NUM_MX) {
            tensorNDimIdx =
                gmmAttrs.transposeWeight ? tensorDimNum - 3 : tensorDimNum - 2; // 静态图做两次inferShape，对应两个分支
        }

        OP_CHECK_IF(tensorShape->GetDim(tensorNDimIdx) != weightNDim_,
                    OP_LOGE(context->GetNodeName(),
                            "The n dim of %s should be equal to the weightNDim[%ld], but the actual value is [%ld].",
                            tensorType.c_str(), weightNDim_, tensorShape->GetDim(tensorNDimIdx)),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedMatmulWeightQuantChecker::CheckShapeForWeightQuantParam(const gert::InferShapeContext *context,
                                                                               const GMMAttrs &gmmAttrs) const
{
    auto scaleShape = context->GetDynamicInputShape(GMM_INDEX_IN_SCALE, 0);
    auto offsetShape = context->GetDynamicInputShape(GMM_INDEX_IN_OFFSET, 0);
    auto perTokenScaleShape = context->GetDynamicInputShape(GMM_INDEX_IN_PERTOKEN_SCALE, 0);
    auto antiquantScaleShape = context->GetDynamicInputShape(GMM_INDEX_IN_ANTIQUANT_SCALE, 0);
    auto antiquantOffsetShape = context->GetDynamicInputShape(GMM_INDEX_IN_ANTIQUANT_OFFSET, 0);
    OP_CHECK_IF(IsNonEmpty(antiquantOffsetShape) && (FP8_SUPPORT_SET.find(weightDtype_) != FP8_SUPPORT_SET.end() ||
                                                     weightDtype_ == ge::DT_FLOAT4_E2M1 ||
                                                     weightDtype_ == ge::DT_FLOAT || IsS8S4NZ(xDtype_, weightDtype_)),
                OP_LOGE(context->GetNodeName(),
                        "In weight quant case, only support antiquantOffset is none when weightDtype is fp8/hif8/fp4 "
                        "or xDtype-weightDtype is int8-int4."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(IsNonEmpty(scaleShape) && !IsS8S4NZ(xDtype_, weightDtype_),
                OP_LOGE(context->GetNodeName(), "In weight quant case, scale must be empty."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(!IsNonEmpty(scaleShape) && IsS8S4NZ(xDtype_, weightDtype_),
                OP_LOGE(context->GetNodeName(), "When xDtype-weightDtype is int8-int4, scale must not be empty."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(IsNonEmpty(offsetShape), OP_LOGE(context->GetNodeName(), "In weight quant case, offset must be empty."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        IsNonEmpty(perTokenScaleShape) && (!IsMxA8W4NZ(xDtype_, weightDtype_) && !IsS8S4NZ(xDtype_, weightDtype_)),
        OP_LOGE(context->GetNodeName(),
                "If xDtype-weightDtype is not fp8_e4m3-fp4_e2m1 or int8-int4, pertokenscale must be empty."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        !IsNonEmpty(perTokenScaleShape) && (IsMxA8W4NZ(xDtype_, weightDtype_) || IsS8S4NZ(xDtype_, weightDtype_)),
        OP_LOGE(context->GetNodeName(),
                "When xDtype-weightDtype is fp8_e4m3-fp4_e2m1 or int8-int4, pertokenscale must be not empty."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(!IsNonEmpty(antiquantScaleShape),
                OP_LOGE(context->GetNodeName(), "In weight quant case, antiquantScale must be not empty."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        CheckShapeForTensorList(context, GMM_INDEX_IN_ANTIQUANT_SCALE, "antiquantScale", gmmAttrs) != ge::GRAPH_SUCCESS,
        OP_LOGE(context->GetNodeName(), "CheckShapeForAntiquantScale failed."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckShapeForTensorList(context, GMM_INDEX_IN_ANTIQUANT_OFFSET, "antiquantOffset", gmmAttrs) !=
                    ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "CheckShapeForAntiquantOffset failed."), return ge::GRAPH_FAILED);
    if (IsMxA8W4NZ(xDtype_, weightDtype_) || IsS8S4NZ(xDtype_, weightDtype_)) {
        OP_CHECK_IF(CheckPertokenScaleForA8W4(context) != ge::GRAPH_SUCCESS,
                    OP_LOGE(context->GetNodeName(), "CheckPertokenScale failed."), return ge::GRAPH_FAILED);
    }
    if (IsS8S4NZ(xDtype_, weightDtype_)) {
        OP_CHECK_IF(CheckShapeForTensorList(context, GMM_INDEX_IN_SCALE, "scale", gmmAttrs) != ge::GRAPH_SUCCESS,
                    OP_LOGE(context->GetNodeName(), "CheckShapeForscale failed."), return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedMatmulWeightQuantChecker::CheckGroupSize(const gert::InferShapeContext *context,
                                                                const GMMAttrs &gmmAttrs) const
{
    if (!(IsA16MxFp4NZ(xDtype_, weightDtype_) || IsMxA8W4NZ(xDtype_, weightDtype_) ||
          IsS8S4NZ(xDtype_, weightDtype_))) {
        return ge::GRAPH_SUCCESS;
    }

    int64_t groupSize = 0;

    auto antiquantScaleShape = context->GetDynamicInputShape(GMM_INDEX_IN_ANTIQUANT_SCALE, 0);
    auto antiquantScaleDimNum = antiquantScaleShape->GetDimNum();
    // 2含义: (g, k/groupSize, n)的k轴索引，此处groupNum是K轴上量化分组的groupNum，与groupNum_含义不同
    int64_t groupNum;
    if (IsMxA8W4NZ(xDtype_, weightDtype_)) {
        // antiquantScaleShape: (g,n,k/64,2) (g, k/64, n, 2) 静态图两次infershape，两个分支
        groupNum = gmmAttrs.transposeWeight ? antiquantScaleShape->GetDim(antiquantScaleDimNum - 2) * 2 :
                                              antiquantScaleShape->GetDim(antiquantScaleDimNum - 3) * 2;
    } else {
        groupNum = gmmAttrs.transposeWeight ? antiquantScaleShape->GetDim(antiquantScaleDimNum - 1) :
                                              antiquantScaleShape->GetDim(antiquantScaleDimNum - 2);
    }
    OP_CHECK_IF(groupNum <= 0, OP_LOGE(context->GetNodeName(), "GroupNum must be greater than 0."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(weightKDim_ % groupNum != 0,
                OP_LOGE(context->GetNodeName(), "GroupNum must be multiple of the k axis of weight."),
                return ge::GRAPH_FAILED);
    groupSize = weightKDim_ / groupNum;

    if (IsS8S4NZ(xDtype_, weightDtype_)) {
        // 伪量化S8S4场景支持groupsize为128/192/256/512
        OP_CHECK_IF(groupSize != 128 && groupSize != 256 && groupSize != 512 && groupSize != 192,
                    OP_LOGE(context->GetNodeName(),
                            "groupSize must be 128/192/256/512, but current groupSize is (%ld).", groupSize),
                    return ge::GRAPH_FAILED);
    } else {
        // Mx量化的groupSize为32
        OP_CHECK_IF(groupSize != MX_GROUP_SIZE,
                    OP_LOGE(context->GetNodeName(), "groupSize must be [%ld], but current groupSize is (%ld).",
                            MX_GROUP_SIZE, groupSize),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedMatmulWeightQuantChecker::GetNumOfInputs(const gert::InferShapeContext *context,
                                                                GMMInputParamsInfo &paramsInputInfo) const
{
    ge::graphStatus res = ge::GRAPH_SUCCESS;
    const gert::Shape *shape = nullptr;
    struct ParamInfoTmp {
        int index;
        size_t &count;
        const char *name;
    };
    ParamInfoTmp params[] = {{GMM_INDEX_IN_X, paramsInputInfo.numX, "numX"},
                             {GMM_INDEX_IN_WEIGHT, paramsInputInfo.numWeight, "numWeight"},
                             {GMM_INDEX_IN_BIAS, paramsInputInfo.numBias, "numBias"},
                             {GMM_INDEX_IN_ANTIQUANT_SCALE, paramsInputInfo.numAntiquantScale, "numAntiquantScale"},
                             {GMM_INDEX_IN_ANTIQUANT_OFFSET, paramsInputInfo.numAntiquantOffset, "numAntiquantOffset"}};

    for (auto &param : params) {
        param.count = 0;
        for (int i = 0; i < GMM_MAX_GROUP_LIST_SIZE_ARRAY; i++) {
            shape = context->GetDynamicInputShape(param.index, param.count);
            if (shape == nullptr) {
                break;
            }
            ++param.count;
        }
        OP_CHECK_IF(param.count >= GMM_MAX_GROUP_LIST_SIZE_ARRAY,
                    OP_LOGE(context->GetNodeName(),
                            "In multi/multi/multi Scenario, each tensorlist's length cannot exceed 128"),
                    return ge::GRAPH_FAILED);
        OP_LOGI(context->GetNodeName(), "%s = %zu", param.name, param.count);
    }
    return res;
}

ge::graphStatus GroupedMatmulWeightQuantChecker::CheckTensorListSizeMultiScenario(
    const gert::InferShapeContext *context, const GMMInputParamsInfo &paramsInputInfo) const
{
    // 检测 bias antiquantScale antiquantOffset的 tensorListsize 需要等于 weightSize
    if (hasBias_) {
        OP_CHECK_IF(
            paramsInputInfo.numBias != paramsInputInfo.numWeight,
            OP_LOGE(context->GetNodeName(), "Bias size should be equal to weight size, actual size are [%zu] and [%zu]",
                    paramsInputInfo.numBias, paramsInputInfo.numWeight),
            return ge::GRAPH_FAILED);
    }

    if (hasAntiquantOffset_) {
        OP_CHECK_IF(paramsInputInfo.numAntiquantOffset != paramsInputInfo.numWeight,
                    OP_LOGE(context->GetNodeName(),
                            "AntiquantOffset size should be equal to weight size, actual size are [%zu] and [%zu]",
                            paramsInputInfo.numAntiquantOffset, paramsInputInfo.numWeight),
                    return ge::GRAPH_FAILED);
    }

    OP_CHECK_IF(paramsInputInfo.numAntiquantScale != paramsInputInfo.numWeight,
                OP_LOGE(context->GetNodeName(),
                        "AntiquantScale size should be equal to weight size, actual size are [%zu] and [%zu]",
                        paramsInputInfo.numAntiquantScale, paramsInputInfo.numWeight),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedMatmulWeightQuantChecker::CheckShapeValid(const gert::InferShapeContext *context,
                                                                 const GMMAttrs &gmmAttrs)
{
    if (gmmAttrs.groupType == GMM_NO_SPLIT) {
        // 多多多场景校验
        GMMInputParamsInfo paramsInputInfo{0, 0, 0, 0, 0, 0, 0};
        OP_CHECK_IF(GetNumOfInputs(context, paramsInputInfo) != ge::GRAPH_SUCCESS,
                    OP_LOGE(context->GetNodeName(), "GetNumOfInputs failed."), return ge::GRAPH_FAILED);
        hasBias_ = paramsInputInfo.numBias > 0;
        if (paramsInputInfo.numBias == 1) {
            auto biasShape = context->GetDynamicInputShape(GMM_INDEX_IN_BIAS, 0);
            hasBias_ = biasShape->GetShapeSize() != 0;
        }
        hasAntiquantOffset_ = paramsInputInfo.numAntiquantOffset > 0;
        if (paramsInputInfo.numAntiquantOffset == 1) {
            auto antiQuantOffsetShape = context->GetDynamicInputShape(GMM_INDEX_IN_ANTIQUANT_OFFSET, 0);
            hasAntiquantOffset_ = antiQuantOffsetShape->GetShapeSize() != 0;
        }
        OP_CHECK_IF(CheckCaseMultiScenario(context, gmmAttrs, paramsInputInfo) != ge::GRAPH_SUCCESS,
                    OP_LOGE(context->GetNodeName(), "CheckCaseMultiScenario failed."), return ge::GRAPH_FAILED);
    } else {
        // 单单单场景校验
        auto groupListShape = context->GetOptionalInputShape(GMM_INDEX_IN_GROUP_LIST);
        OP_CHECK_NULL_WITH_CONTEXT(context, groupListShape);
        OP_CHECK_IF(CheckShapeForGrouplist(context, groupListShape) != ge::GRAPH_SUCCESS,
                    OP_LOGE(context->GetNodeName(), "CheckShapeForGrouplist failed."), return ge::GRAPH_FAILED);
        groupNum_ = groupListShape->GetDim(0);
        OP_CHECK_IF(CheckShapeForXAndWeight(context) != ge::GRAPH_SUCCESS,
                    OP_LOGE(context->GetNodeName(), "CheckShapeForXAndWeight failed."), return ge::GRAPH_FAILED);
        OP_CHECK_IF(CheckShapeForTensorList(context, GMM_INDEX_IN_BIAS, "bias", gmmAttrs) != ge::GRAPH_SUCCESS,
                    OP_LOGE(context->GetNodeName(), "CheckShapeForBias failed."), return ge::GRAPH_FAILED);
        OP_CHECK_IF(CheckShapeForWeightQuantParam(context, gmmAttrs) != ge::GRAPH_SUCCESS,
                    OP_LOGE(context->GetNodeName(), "CheckShapeForWeightQuantParam failed."), return ge::GRAPH_FAILED);
        OP_CHECK_IF(CheckGroupSize(context, gmmAttrs) != ge::GRAPH_SUCCESS,
                    OP_LOGE(context->GetNodeName(), "CheckGroupSize failed."), return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

bool IsUnknownShape(const gert::Shape *shape)
{
    if (IsNonEmpty(shape)) {
        size_t size = shape->GetDimNum();
        for (size_t i = 0; i < size; i++) {
            if (shape->GetDim(i) == UNKNOWN_SHAPE_VALUE || shape->GetDim(i) == SHAPE_UNKNOWN_DIM_NUM) {
                return true;
            }
        }
        return false;
    }
    return false;
}

bool CheckUnknownShape(const gert::InferShapeContext *context)
{
    bool hasUnknownShape = false;
    auto xShape = context->GetDynamicInputShape(GMM_INDEX_IN_X, 0);
    auto weightShape = context->GetDynamicInputShape(GMM_INDEX_IN_WEIGHT, 0);
    auto antiquantScaleShape = context->GetDynamicInputShape(GMM_INDEX_IN_ANTIQUANT_SCALE, 0);
    auto antiquantOffsetShape = context->GetDynamicInputShape(GMM_INDEX_IN_ANTIQUANT_OFFSET, 0);
    auto biasShape = context->GetDynamicInputShape(GMM_INDEX_IN_BIAS, 0);
    auto groupListShape = context->GetOptionalInputShape(GMM_INDEX_IN_GROUP_LIST);
    hasUnknownShape = IsUnknownShape(xShape) || IsUnknownShape(weightShape) || IsUnknownShape(antiquantScaleShape) ||
                      IsUnknownShape(groupListShape);
    if (!hasUnknownShape && IsNonEmpty(antiquantOffsetShape)) {
        hasUnknownShape |= IsUnknownShape(antiquantOffsetShape);
    }
    if (!hasUnknownShape && IsNonEmpty(biasShape)) {
        hasUnknownShape |= IsUnknownShape(biasShape);
    }
    return hasUnknownShape;
}

ge::graphStatus GroupedMatmulWeightQuantChecker::CheckShape(const gert::InferShapeContext *context,
                                                            const GroupedMatmulCommonUtil &commonUtil)
{
    if (CheckUnknownShape(context)) {
        return ge::GRAPH_SUCCESS;
    }

    OP_CHECK_IF(CheckScenarioValid(context, commonUtil.attrsInfo) != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "CheckScenarioValid failed."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckShapeValid(context, commonUtil.attrsInfo) != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "CheckShapeValid failed."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedMatmulWeightQuantChecker::UpdateShapeY(gert::InferShapeContext *context, size_t idxY,
                                                              std::vector<int64_t> &yDims) const
{
    gert::Shape *yShape = context->GetOutputShape(idxY);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);
    yShape->SetDimNum(yDims.size());
    for (size_t dim = 0; dim < yDims.size(); ++dim) {
        yShape->SetDim(dim, yDims[dim]);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedMatmulWeightQuantChecker::UpdateShapeYMultiDim(gert::InferShapeContext *context, size_t idxY,
                                                                      const gert::Shape *xShape,
                                                                      const gert::Shape *weightShape) const
{
    gert::Shape *yShape = context->GetOutputShape(idxY);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);
    *yShape = *xShape;
    size_t dimY = yShape->GetDimNum();
    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const bool *transposeWPtr = attrs->GetAttrPointer<bool>(GMM_INDEX_ATTR_TRANSPOSE_W);
    const bool *transposeXPtr = attrs->GetAttrPointer<bool>(GMM_INDEX_ATTR_TRANSPOSE_X);

    OP_CHECK_NULL_WITH_CONTEXT(context, weightShape);
    if (transposeWPtr != nullptr && *transposeWPtr) {
        yShape->SetDim(dimY - 1, weightShape->GetDim(weightShape->GetDimNum() - 2));  // -2: transpose weight
    } else {
        yShape->SetDim(dimY - 1, weightShape->GetDim(weightShape->GetDimNum() - 1));
    }
    if (transposeXPtr != nullptr && *transposeXPtr) {
        yShape->SetDim(dimY - 2, xShape->GetDim(xShape->GetDimNum() - 1));  // -2: last two dim of Y
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedMatmulWeightQuantChecker::InferOutShape(gert::InferShapeContext *context,
                                                               const GMMAttrs &gmmAttrs) const
{
    if (gmmAttrs.groupType == GMM_NO_SPLIT) {
        size_t idx = 0;
        size_t idw = 0;
        const gert::Shape *w0Shape = context->GetDynamicInputShape(GMM_INDEX_IN_WEIGHT, 0);
        OP_CHECK_NULL_WITH_CONTEXT(context, w0Shape);
        for (int i = 0; i < GMM_MAX_GROUP_LIST_SIZE_ARRAY; i++) {
            const gert::Shape *xShape = context->GetDynamicInputShape(GMM_INDEX_IN_X, idx);
            if (xShape == nullptr) {
                break;
            }
            ++idx;
            const gert::Shape *wShape = context->GetDynamicInputShape(GMM_INDEX_IN_WEIGHT, idw);
            if (wShape) {
                ++idw;
            } else {
                wShape = w0Shape;
            }
            OP_CHECK_IF(UpdateShapeYMultiDim(context, GMM_INDEX_OUT_Y + idx - 1, xShape, wShape) != ge::GRAPH_SUCCESS,
                        OP_LOGE(context->GetNodeName(), "Failed to update shape of y."), return ge::GRAPH_FAILED);
        }
    } else {
        std::vector<int64_t> yDims = {xMDim_, weightNDim_};
        OP_CHECK_IF(UpdateShapeY(context, GMM_INDEX_OUT_Y, yDims) != ge::GRAPH_SUCCESS,
                    OP_LOGE(context->GetNodeName(), "Failed to update y shape."), return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedMatmulWeightQuantChecker::CheckScaleDtypeForS8S4(const gert::InferDataTypeContext *context) const
{
    auto perTokenScaleDtype = context->GetDynamicInputDataType(GMM_INDEX_IN_PERTOKEN_SCALE, 0);
    auto scaleDtype = context->GetDynamicInputDataType(GMM_INDEX_IN_SCALE, 0);
    auto antiquantScaleDtype = context->GetDynamicInputDataType(GMM_INDEX_IN_ANTIQUANT_SCALE, 0);
    OP_CHECK_IF(scaleDtype != ge::DT_FLOAT,
                OP_LOGE(context->GetNodeName(), "scaleDtype datatype [%s] does not match float32.",
                        ge::TypeUtils::DataTypeToAscendString(scaleDtype).GetString()),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(antiquantScaleDtype != ge::DT_FLOAT16,
                OP_LOGE(context->GetNodeName(), "antiquantScaleDtype datatype [%s] does not match float16.",
                        ge::TypeUtils::DataTypeToAscendString(antiquantScaleDtype).GetString()),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(perTokenScaleDtype != ge::DT_FLOAT,
                OP_LOGE(context->GetNodeName(), "perTokenScaleDtype datatype [%s] does not match float32.",
                        ge::TypeUtils::DataTypeToAscendString(perTokenScaleDtype).GetString()),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedMatmulWeightQuantChecker::CheckBiasDtype(const gert::InferDataTypeContext *context) const
{
    auto xDtype = context->GetDynamicInputDataType(GMM_INDEX_IN_X, 0);
    auto biasDtype = context->GetDynamicInputDataType(GMM_INDEX_IN_BIAS, 0);
    OP_CHECK_IF(BIAS_TYPE_SUPPORT_MAP.find(xDtype) == BIAS_TYPE_SUPPORT_MAP.end(),
                OP_LOGE(context->GetNodeName(), "Cannot find bias dtype match with xDtype [%s].",
                        ge::TypeUtils::DataTypeToAscendString(xDtype).GetString()),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(BIAS_TYPE_SUPPORT_MAP.at(xDtype).find(biasDtype) == BIAS_TYPE_SUPPORT_MAP.at(xDtype).end(),
                OP_LOGE(context->GetNodeName(), "Data type [%s] is not supported for bias, when xDtype is [%s].",
                        ge::TypeUtils::DataTypeToAscendString(biasDtype).GetString(),
                        ge::TypeUtils::DataTypeToAscendString(xDtype).GetString()),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedMatmulWeightQuantChecker::CheckTensorListDataType(const gert::InferDataTypeContext *context,
                                                                         uint32_t index, const ge::DataType dtype) const
{
    size_t inIdx = 0;
    for (int i = 0; i < GMM_MAX_GROUP_LIST_SIZE_ARRAY; i++) {
        auto iDtype = context->GetDynamicInputDataType(index, inIdx);
        if (iDtype == ge::DT_UNDEFINED) {
            break;
        }
        OP_CHECK_IF(iDtype != dtype,
                    OP_LOGE(context->GetNodeName(), "data type of tensors in a tensorList should all be the same!"),
                    return ge::GRAPH_FAILED);
        ++inIdx;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedMatmulWeightQuantChecker::CheckMatmulDataType(
    const gert::InferDataTypeContext *context, const ge::DataType xDtype, const ge::DataType weightDtype,
    const ge::DataType biasDtype, const ge::DataType antiquantScaleDtype, const ge::DataType antiquantOffsetDtype) const
{
    // 单单单/多多多常规参数通用校验，PerTokenScale和scale当前仅支持单单单，暂不在此处校验
    OP_CHECK_IF(CheckTensorListDataType(context, GMM_INDEX_IN_X, xDtype) != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "x dtype does not match with required dtype[%s].",
                        ge::TypeUtils::DataTypeToAscendString(xDtype).GetString()),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckTensorListDataType(context, GMM_INDEX_IN_WEIGHT, weightDtype) != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "weight dtype does not match with required dtype[%s].",
                        ge::TypeUtils::DataTypeToAscendString(weightDtype).GetString()),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckTensorListDataType(context, GMM_INDEX_IN_BIAS, biasDtype) != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "bias dtype does not match with required dtype[%s].",
                        ge::TypeUtils::DataTypeToAscendString(biasDtype).GetString()),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        CheckTensorListDataType(context, GMM_INDEX_IN_ANTIQUANT_SCALE, antiquantScaleDtype) != ge::GRAPH_SUCCESS,
        OP_LOGE(context->GetNodeName(), "antiquantScaleDtype dtype does not match with required dtype[%s].",
                ge::TypeUtils::DataTypeToAscendString(antiquantScaleDtype).GetString()),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        CheckTensorListDataType(context, GMM_INDEX_IN_ANTIQUANT_OFFSET, antiquantOffsetDtype) != ge::GRAPH_SUCCESS,
        OP_LOGE(context->GetNodeName(), "antiquantOffsetDtype dtype does not match with required dtype[%s].",
                ge::TypeUtils::DataTypeToAscendString(antiquantOffsetDtype).GetString()),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedMatmulWeightQuantChecker::CheckDtype(const gert::InferDataTypeContext *context) const
{
    auto xDtype = context->GetDynamicInputDataType(GMM_INDEX_IN_X, 0);
    auto weightDtype = context->GetDynamicInputDataType(GMM_INDEX_IN_WEIGHT, 0);
    auto biasDtype = context->GetDynamicInputDataType(GMM_INDEX_IN_BIAS, 0);
    auto antiquantScaleDtype = context->GetDynamicInputDataType(GMM_INDEX_IN_ANTIQUANT_SCALE, 0);
    auto antiquantOffsetDtype = context->GetDynamicInputDataType(GMM_INDEX_IN_ANTIQUANT_OFFSET, 0);
    auto perTokenScaleDtype = context->GetDynamicInputDataType(GMM_INDEX_IN_PERTOKEN_SCALE, 0);

    // 必选参数校验，同时校验数据流是否支持
    if (IsA16W8(xDtype, weightDtype) || IsA16F8(xDtype, weightDtype) || IsA16W4(xDtype, weightDtype)) {
        OP_CHECK_IF(antiquantScaleDtype != xDtype,
                    OP_LOGE(context->GetNodeName(), "AntiquantScale datatype [%s] does not match xDtype [%s].",
                            ge::TypeUtils::DataTypeToAscendString(antiquantScaleDtype).GetString(),
                            ge::TypeUtils::DataTypeToAscendString(xDtype).GetString()),
                    return ge::GRAPH_FAILED);
    } else if (IsA16MxFp4NZ(xDtype, weightDtype) || IsMxA8W4NZ(xDtype, weightDtype)) {
        OP_CHECK_IF(antiquantScaleDtype != ge::DT_FLOAT8_E8M0,
                    OP_LOGE(context->GetNodeName(),
                            "Only support float8_e8m0 for antiquantScaleDataType when xDtype-weightDtype is "
                            "fp16/bf16-fp4_e2m1 or fp8_e4m3-fp4_e2m1, but got [%s].",
                            ge::TypeUtils::DataTypeToAscendString(antiquantScaleDtype).GetString()),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(perTokenScaleDtype != ge::DT_FLOAT8_E8M0 && IsMxA8W4NZ(xDtype, weightDtype),
                    OP_LOGE(context->GetNodeName(),
                            "Only support float8_e8m0 for perTokenScaleDtype when xDtype-weightDtype is "
                            "fp8_e4m3-fp4_e2m1, but got [%s].",
                            ge::TypeUtils::DataTypeToAscendString(perTokenScaleDtype).GetString()),
                    return ge::GRAPH_FAILED);
    } else if (IsS8S4NZ(xDtype, weightDtype)) {
        OP_CHECK_IF(CheckScaleDtypeForS8S4(context) != ge::GRAPH_SUCCESS,
                    OP_LOGE(context->GetNodeName(), "CheckScaleDtype failed."), return ge::GRAPH_FAILED);
    } else {
        OP_LOGE(context->GetNodeName(), "Weight quant case does not support xDtype [%s] and weightDtype [%s].",
                ge::TypeUtils::DataTypeToAscendString(xDtype).GetString(),
                ge::TypeUtils::DataTypeToAscendString(weightDtype).GetString());
        return ge::GRAPH_FAILED;
    }

    if (IsA16W8(xDtype, weightDtype) || IsA16W4(xDtype, weightDtype)) {
        OP_CHECK_IF(antiquantOffsetDtype != xDtype,
                    OP_LOGE(context->GetNodeName(), "AntiquantOffset datatype [%s] does not match xDtype [%s].",
                            ge::TypeUtils::DataTypeToAscendString(antiquantOffsetDtype).GetString(),
                            ge::TypeUtils::DataTypeToAscendString(xDtype).GetString()),
                    return ge::GRAPH_FAILED);
    }

    OP_CHECK_IF(CheckBiasDtype(context) != ge::GRAPH_SUCCESS, OP_LOGE(context->GetNodeName(), "CheckBiasDtype failed."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckMatmulDataType(context, xDtype, weightDtype, biasDtype, antiquantScaleDtype,
                                    antiquantOffsetDtype) != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "CheckMatmulDataType is failed!"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedMatmulWeightQuantChecker::InferOutDtype(gert::InferDataTypeContext *context) const
{
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t *outputDtype = attrs->GetInt(GMM_INDEX_ATTR_OUTPUT_DTYPE);
    ge::DataType yDtype = context->GetDynamicInputDataType(GMM_INDEX_IN_X, 0);
    auto it = GMM_OUTPUT_DTYPE_MAP.find(*outputDtype);
    OP_CHECK_IF(it == GMM_OUTPUT_DTYPE_MAP.end(),
                OP_LOGE(context->GetNodeName(),
                        "The output dtype should be bfloat16 or float16 , but the actual output dtype is [%s].",
                        ge::TypeUtils::DataTypeToAscendString(yDtype).GetString()),
                return ge::GRAPH_FAILED);
    yDtype = it->second;
    context->SetOutputDataType(GMM_INDEX_OUT_Y, yDtype);
    return ge::GRAPH_SUCCESS;
}

}  // namespace ops