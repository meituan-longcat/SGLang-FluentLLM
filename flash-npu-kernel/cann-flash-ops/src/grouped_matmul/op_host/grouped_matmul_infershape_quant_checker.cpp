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
 * \file grouped_matmul_infershape_quant_checker.cpp
 * \brief
 */
#include "grouped_matmul_infershape_common_util.h"
#include "grouped_matmul_infershape_quant_checker.h"

namespace ops {

static const std::unordered_set<ge::DataType> X_TYPE_SUPPORT_SET = {ge::DT_INT8, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2,
                                                                    ge::DT_HIFLOAT8, ge::DT_FLOAT4_E2M1};
static const std::unordered_set<ge::DataType> WEIGHT_TYPE_SUPPORT_SET = {ge::DT_INT8, ge::DT_FLOAT8_E4M3FN,
                                                                    ge::DT_FLOAT8_E5M2, ge::DT_HIFLOAT8, ge::DT_FLOAT4_E2M1};
static const std::unordered_set<ge::DataType> BIAS_TYPE_SUPPORT_SET = {ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT,
                                                                   ge::DT_INT32};
static const std::unordered_set<ge::DataType> SCALE_TYPE_SUPPORT_SET = {ge::DT_UINT64, ge::DT_INT64, ge::DT_FLOAT,
                                                                    ge::DT_BF16, ge::DT_FLOAT8_E8M0};
static const std::unordered_set<ge::DataType> PERTOEKN_SCALE_TYPE_SUPPORT_SET = {ge::DT_FLOAT, ge::DT_FLOAT8_E8M0};

static bool inline IsNonEmpty(const gert::Shape *shape)
{
    return (shape != nullptr && !(shape->GetDimNum() == 1 && shape->GetDim(0) == 0));
}

bool GroupedMatmulQuantChecker::LogicXOR(bool cond1, bool cond2) const
{
    uint64_t result = static_cast<uint64_t>(cond1) ^ static_cast<uint64_t>(cond2);
    return static_cast<bool>(result);
}

ge::graphStatus GroupedMatmulQuantChecker::GetXAndWeightDimValue(const gert::InferShapeContext *context,
                                                                 const GMMAttrs &gmmAttrs)
{
    auto xShape = context->GetDynamicInputShape(GMM_INDEX_IN_X, 0);
    auto weightShape = context->GetDynamicInputShape(GMM_INDEX_IN_WEIGHT, 0);
    OP_CHECK_IF(!(IsNonEmpty(xShape) && IsNonEmpty(weightShape)),
                OP_LOGE(context->GetNodeName(), "The 1st tensor of tensor list x and weight cannot be empty."),
                return ge::GRAPH_FAILED);
    xdimNum_ = xShape->GetDimNum();
    weightdimNum_ = weightShape->GetDimNum();
    OP_CHECK_IF(xdimNum_ < GMM_MIN_FM_DIM,
                OP_LOGE(context->GetNodeName(), "The dim num of x's 1st tensor should be greater than 1, but it is \
[%zu].",
                        xdimNum_),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(weightdimNum_ < GMM_MIN_WEIGHT_DIM,
                OP_LOGE(context->GetNodeName(), "The dim num of weight's 1st tensor should be greater than 1, \
but it is [%zu].",
                        weightdimNum_),
                return ge::GRAPH_FAILED);
    xMDim_ = gmmAttrs.transposeX ? xShape->GetDim(xdimNum_ - 1) : xShape->GetDim(xdimNum_ - PENULTIMATE_DIM);
    xKDim_ = gmmAttrs.transposeX ? xShape->GetDim(xdimNum_ - PENULTIMATE_DIM) : xShape->GetDim(xdimNum_ - 1);
    weightKDim_ = gmmAttrs.transposeWeight ? weightShape->GetDim(weightdimNum_ - 1) :
                                             weightShape->GetDim(weightdimNum_ - PENULTIMATE_DIM);
    weightNDim_ = gmmAttrs.transposeWeight ? weightShape->GetDim(weightdimNum_ - PENULTIMATE_DIM) :
                                             weightShape->GetDim(weightdimNum_ - 1);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedMatmulQuantChecker::CheckDtypeValid(const gert::InferDataTypeContext *context) const
{
    auto xDtype = context->GetDynamicInputDataType(GMM_INDEX_IN_X, 0);
    auto weightDtype = context->GetDynamicInputDataType(GMM_INDEX_IN_WEIGHT, 0);
    // mandory param dtype check
    OP_CHECK_IF(X_TYPE_SUPPORT_SET.find(xDtype) == X_TYPE_SUPPORT_SET.end(),
                OP_LOGE(context->GetNodeName(),
                        "Data type [%s] is not supported for x's 1st tensor; supported types are: INT8, "
                        "FLOAT8_E4M3FN, FLOAT8_E5M2, HIFLOAT8, and FLOAT4_E2M1.",
                        ge::TypeUtils::DataTypeToAscendString(xDtype).GetString()),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(WEIGHT_TYPE_SUPPORT_SET.find(weightDtype) == WEIGHT_TYPE_SUPPORT_SET.end(),
                OP_LOGE(context->GetNodeName(),
                        "Data type [%s] is not supported for weight; supported types are: INT8, FLOAT8_E4M3FN, "
                        "FLOAT8_E5M2, HIFLOAT8, and FLOAT4_E2M1.",
                        ge::TypeUtils::DataTypeToAscendString(weightDtype).GetString()),
                return ge::GRAPH_FAILED);
    if (xDtype == ge::DataType::DT_INT8 || weightDtype == ge::DataType::DT_INT8) {
        OP_CHECK_IF(
            !(xDtype == ge::DataType::DT_INT8 && weightDtype == ge::DataType::DT_INT8),
            OP_LOGE(
                context->GetNodeName(),
                "When data type of x is int8, data type of weight should also be int8, vice versa. But data type of \
x is [%s] and the data type of weight is [%s].",
                ge::TypeUtils::DataTypeToAscendString(xDtype).GetString(),
                ge::TypeUtils::DataTypeToAscendString(weightDtype).GetString()),
            return ge::GRAPH_FAILED);
    }
    OP_CHECK_IF(
        LogicXOR((xDtype == ge::DataType::DT_HIFLOAT8), (weightDtype == ge::DataType::DT_HIFLOAT8)),
        OP_LOGE(context->GetNodeName(),
                "When one input dtype is HIFLOAT8, then the other input dtype must be HIFLOAT8, vice versa, actual \
x is %s, weight is %s.",
                ge::TypeUtils::DataTypeToAscendString(xDtype).GetString(),
                ge::TypeUtils::DataTypeToAscendString(weightDtype).GetString()),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        LogicXOR((xDtype == ge::DataType::DT_FLOAT8_E4M3FN || xDtype == ge::DataType::DT_FLOAT8_E5M2),
                 (weightDtype == ge::DataType::DT_FLOAT8_E4M3FN || weightDtype == ge::DataType::DT_FLOAT8_E5M2)),
        OP_LOGE(
            context->GetNodeName(),
            "When x input dtype is FLOAT8, then the weight input dtype must be FLOAT8, vice versa, actual x is %s, \
weight is %s.", ge::TypeUtils::DataTypeToAscendString(xDtype).GetString(),
                ge::TypeUtils::DataTypeToAscendString(weightDtype).GetString()),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(LogicXOR((xDtype == ge::DataType::DT_FLOAT4_E2M1), (weightDtype == ge::DataType::DT_FLOAT4_E2M1)),
        OP_LOGE(context->GetNodeName(),
            "When x input dtype is FLOAT4, then the weight input dtype must be FLOAT4, vice versa, actual x is %s, \
weight is %s.", ge::TypeUtils::DataTypeToAscendString(xDtype).GetString(),
                ge::TypeUtils::DataTypeToAscendString(weightDtype).GetString()),
                return ge::GRAPH_FAILED);
    auto ScaleDtype = context->GetDynamicInputDataType(GMM_INDEX_IN_SCALE, 0);
    OP_CHECK_IF(SCALE_TYPE_SUPPORT_SET.find(ScaleDtype) == SCALE_TYPE_SUPPORT_SET.end(),
                OP_LOGE(context->GetNodeName(),
                        "Data type [%s] is not supported for scale; supported types are: UINT64, INT64, "
                        "FLOAT, BF16, and FLOAT8_E8M0.",
                        ge::TypeUtils::DataTypeToAscendString(ScaleDtype).GetString()),
                return ge::GRAPH_FAILED);

    auto biasDtype = context->GetDynamicInputDataType(GMM_INDEX_IN_BIAS, 0);
    OP_CHECK_IF(
        BIAS_TYPE_SUPPORT_SET.find(biasDtype) == BIAS_TYPE_SUPPORT_SET.end(),
        OP_LOGE(context->GetNodeName(),
                "Data type [%s] is not supported for bias; supported types are: FLOAT16, BF16, FLOAT, and INT32.",
                ge::TypeUtils::DataTypeToAscendString(biasDtype).GetString()),
        return ge::GRAPH_FAILED);

    auto pertokenScaleDtype = context->GetDynamicInputDataType(GMM_INDEX_IN_PERTOKEN_SCALE, 0);
    if (pertokenScaleDtype != ge::DT_UNDEFINED) {
        OP_CHECK_IF(
            PERTOEKN_SCALE_TYPE_SUPPORT_SET.find(pertokenScaleDtype) == PERTOEKN_SCALE_TYPE_SUPPORT_SET.end(),
            OP_LOGE(context->GetNodeName(),
                    "Data type [%s] is not supported for pertokenScale; supported types are: FLOAT, and FLOAT8_E8M0.",
                    ge::TypeUtils::DataTypeToAscendString(pertokenScaleDtype).GetString()),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedMatmulQuantChecker::CheckNotZeroValueForNoneSplitAxis(const gert::InferShapeContext *context,
                                                                             const GMMAttrs &gmmAttrs) const
{
    auto weightShape = context->GetDynamicInputShape(GMM_INDEX_IN_WEIGHT, 0);
    if (gmmAttrs.groupType == GMM_SPLIT_M) {
        auto eDimValue = weightShape->GetDim(0);
        OP_CHECK_IF(xKDim_ == 0 || eDimValue == 0 || weightNDim_ == 0,
                    OP_LOGE(context->GetNodeName(),
                            "Non split axis' dim value cannot be zero, but kDimValue is [%ld], eDimValue is [%ld], \
nDimValue is [%ld].",
                            xKDim_, eDimValue, weightNDim_),
                    return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF(xMDim_ == 0 || weightNDim_ == 0,
                    OP_LOGE(context->GetNodeName(),
                            "Non split axis' dim value cannot be zero but mDimValue is [%ld], nDimValue is [%ld].",
                            xMDim_, weightNDim_),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedMatmulQuantChecker::CheckScenarioValidForShape(const gert::InferShapeContext *context,
                                                                      const GMMAttrs &gmmAttrs) const
{
    // only support for single/single/single Scenario
    auto xSecondTensorShape = context->GetDynamicInputShape(GMM_INDEX_IN_X, 1);
    auto weightSecondTensorShape = context->GetDynamicInputShape(GMM_INDEX_IN_WEIGHT, 1);
    OP_CHECK_IF(IsNonEmpty(xSecondTensorShape),
                OP_LOGE(context->GetNodeName(), "Only support single/single/single scenario for now, but the second \
tensor of tensor list x is not empty."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(IsNonEmpty(weightSecondTensorShape),
                OP_LOGE(context->GetNodeName(), "Only support single/single/single scenario for now, but the second \
tensor of tensor list weight is not empty."),
                return ge::GRAPH_FAILED);
    // check split item value valid
    OP_CHECK_IF(gmmAttrs.splitItem != GMM_X_SEPARATED && gmmAttrs.splitItem != GMM_NO_SEPARATED,
                OP_LOGE(context->GetNodeName(), "Invalid splitItem, which can only be one of 2/3, but it is [%ld].",
                        gmmAttrs.splitItem),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(gmmAttrs.groupType != GMM_SPLIT_M && gmmAttrs.groupType != GMM_SPLIT_K,
                OP_LOGE(context->GetNodeName(), "Invalid groupType, which can only be one of 0/2, but it is [%ld].",
                        gmmAttrs.groupType),
                return ge::GRAPH_FAILED);
    // non split axis cannot be empty tensor
    OP_CHECK_IF(CheckNotZeroValueForNoneSplitAxis(context, gmmAttrs) != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "CheckNotZeroValueForNoneSplitAxis Failed."), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedMatmulQuantChecker::CheckFormatValid(const gert::InferShapeContext *context) const
{
    const auto xDesc = context->GetDynamicInputDesc(GMM_INDEX_IN_X, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xDesc);
    const auto xFormat = xDesc->GetOriginFormat();
    OP_CHECK_IF(xFormat != ge::FORMAT_ND && xFormat != ge::FORMAT_NCL && xFormat != ge::FORMAT_NCHW,
                OP_LOGE(context->GetNodeName(), "Format of x only supports ND, NCL or NCHW for now, but it is [%s].",
                        ge::TypeUtils::FormatToAscendString(xFormat).GetString()),
                return ge::GRAPH_FAILED);
    const auto weightDesc = context->GetDynamicInputDesc(GMM_INDEX_IN_WEIGHT, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightDesc);
    const auto weightFormat = weightDesc->GetOriginFormat();
    OP_CHECK_IF(weightFormat != ge::FORMAT_ND && weightFormat != ge::FORMAT_NCL && weightFormat != ge::FORMAT_NCHW,
                OP_LOGE(context->GetNodeName(),
                        "Format of weight only supports ND, NCL or NCHW for now, but it is [%s].",
                        ge::TypeUtils::FormatToAscendString(weightFormat).GetString()),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedMatmulQuantChecker::CheckShapeForBias(const gert::InferShapeContext *context) const
{
    auto biasShape = context->GetDynamicInputShape(GMM_INDEX_IN_BIAS, 0);
    if (IsNonEmpty(biasShape)) {
        size_t biasDimNum = biasShape->GetDimNum();
        OP_CHECK_IF(biasDimNum != 2,
                    OP_LOGE(context->GetNodeName(),
                            "When bias is not null, its dim should be 2, but the actual is [%zu].", biasDimNum),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(
            biasShape->GetDim(0) != groupNum_ || biasShape->GetDim(1) != static_cast<int64_t>(weightNDim_),
            OP_LOGE(context->GetNodeName(),
                    "The shape of bias should be (g, n), which is ([%ld], [%ld]), but the actual is ([%ld], [%ld]).",
                    groupNum_, weightNDim_, biasShape->GetDim(0), biasShape->GetDim(1)),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

bool GroupedMatmulQuantChecker::IsDoubleScaleScenario(const gert::Shape *scaleShape,
                                                      const gert::Shape *perTokenScaleShape) const
{ // pertoken dim num support 2 or 1 for now
    if (perTokenScaleShape->GetDimNum() != 2 && perTokenScaleShape->GetDimNum() != 1) {
        return false;
    }
    bool IsScaleMatchedDoubleScale = false;
    bool IsPertokenScaleMatchedDoubleScale = false;
    if (scaleShape->GetDimNum() == 2) { // dim num 2 with shape (e, 1)
        IsScaleMatchedDoubleScale = (scaleShape->GetDim(0) == groupNum_) && (scaleShape->GetDim(1) == 1);
    } else {
        IsScaleMatchedDoubleScale = scaleShape->GetDim(0) == groupNum_;
    }

    if (perTokenScaleShape->GetDimNum() == 2) { // dim num 2 with shape (e, 1)
        IsPertokenScaleMatchedDoubleScale =
            (perTokenScaleShape->GetDim(0) == groupNum_) && (perTokenScaleShape->GetDim(1) == 1);
    } else {
        IsPertokenScaleMatchedDoubleScale = perTokenScaleShape->GetDim(0) == groupNum_;
    }
    return IsScaleMatchedDoubleScale && IsPertokenScaleMatchedDoubleScale;
}

bool GroupedMatmulQuantChecker::IsPerTileQuantMode(const gert::InferShapeContext *context,
                                                   const GMMAttrs &gmmAttrs) const
{
    auto perTokenScaleShape = context->GetDynamicInputShape(GMM_INDEX_IN_PERTOKEN_SCALE, 0);
    bool isPerTileQuantMode = false;
    if (!IsNonEmpty(perTokenScaleShape)) {
        return isPerTileQuantMode;
    }
    auto perTokenScaleDimNum = perTokenScaleShape->GetDimNum();
    // 2 is the minimum dimension for perTokenScale in perTile case
    if (perTokenScaleDimNum < 2 || perTokenScaleDimNum != xdimNum_) {
        return false;
    }
    auto perTokenMDim = gmmAttrs.transposeX ? perTokenScaleShape->GetDim(xdimNum_ - 1) :
                                              perTokenScaleShape->GetDim(xdimNum_ - PENULTIMATE_DIM);
    if (perTokenMDim != xMDim_) {
        return false;
    }
    auto scaleShape = context->GetDynamicInputShape(GMM_INDEX_IN_SCALE, 0);
    auto scaleDimNum = scaleShape->GetDimNum();
    // 2 is the minimum dimension for scale in perTile case
    if (scaleDimNum < 2 || scaleDimNum != weightdimNum_) {
        return false;
    }
    auto scaleKDim = gmmAttrs.transposeWeight ? scaleShape->GetDim(scaleDimNum - 1) :
                                                scaleShape->GetDim(scaleDimNum - PENULTIMATE_DIM);
    auto scaleNDim = gmmAttrs.transposeWeight ? scaleShape->GetDim(scaleDimNum - PENULTIMATE_DIM) :
                                                scaleShape->GetDim(scaleDimNum - 1);
    bool isKdimValid =
        ((gmmAttrs.groupType == GMM_SPLIT_K && scaleKDim == (weightKDim_ / PERTILE_GROUP_SIZE) + groupNum_) ||
         (gmmAttrs.groupType == GMM_SPLIT_M &&
          scaleKDim == (weightKDim_ + PERTILE_GROUP_SIZE - 1) / PERTILE_GROUP_SIZE));
    if (scaleNDim == (weightNDim_ + PERTILE_GROUP_SIZE - 1) / PERTILE_GROUP_SIZE && isKdimValid) {
        isPerTileQuantMode = true;
    }
    return isPerTileQuantMode;
}

ge::graphStatus GroupedMatmulQuantChecker::CheckGroupedMatmulPerGroupDim(const gert::InferShapeContext *context,
                                                                         const GMMAttrs &gmmAttrs) const
{
    auto scaleDimNum = context->GetDynamicInputShape(GMM_INDEX_IN_SCALE, 0)->GetDimNum();
    auto perTokenDimNum = context->GetDynamicInputShape(GMM_INDEX_IN_PERTOKEN_SCALE, 0)->GetDimNum();
    OP_CHECK_IF(xdimNum_ != GMM_MIN_FM_DIM,
                OP_LOGE(context->GetNodeName(), "The dim num of x should be 2, but actual dim num is %zu.", xdimNum_),
                return ge::GRAPH_FAILED);
    if (gmmAttrs.groupType == GMM_SPLIT_M) {
        OP_CHECK_IF(weightdimNum_ != GMM_SPLIT_M_SINGLE_WEIGHT_DIM,
                    OP_LOGE(context->GetNodeName(),
                            "The dim num of weight should be 3 when groupType is 0 (split M), but \
actual dim num is %zu.",
                            weightdimNum_),
                    return ge::GRAPH_FAILED);
    } else if (gmmAttrs.groupType == GMM_SPLIT_K) {
        OP_CHECK_IF(weightdimNum_ != GMM_SPLIT_K_SINGLE_WEIGHT_DIM,
                    OP_LOGE(context->GetNodeName(),
                            "The dim num of weight should be 2 when groupType is 2 (split K), but \
actual dim num is %zu.",
                            weightdimNum_),
                    return ge::GRAPH_FAILED);
    }
    OP_CHECK_IF(scaleDimNum != weightdimNum_,
                OP_LOGE(context->GetNodeName(), "The dim num of scale[%zu] should be equal to that of weight[%zu] when \
groupType is %ld.",
                        scaleDimNum, weightdimNum_, gmmAttrs.groupType),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(perTokenDimNum != xdimNum_,
                OP_LOGE(context->GetNodeName(),
                        "The dim num of per_token_scale[%zu] should be equal to that of x[%zu].", perTokenDimNum,
                        xdimNum_),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedMatmulQuantChecker::CheckGroupedMatmulPerTileShape(const gert::InferShapeContext *context,
                                                                          const GMMAttrs &gmmAttrs) const
{
    auto scaleShape = context->GetDynamicInputShape(GMM_INDEX_IN_SCALE, 0);
    auto perTokenShape = context->GetDynamicInputShape(GMM_INDEX_IN_PERTOKEN_SCALE, 0);
    auto perTokenMDim =
        gmmAttrs.transposeX ? perTokenShape->GetDim(xdimNum_ - 1) : perTokenShape->GetDim(xdimNum_ - PENULTIMATE_DIM);
    auto perTokenKDim =
        gmmAttrs.transposeX ? perTokenShape->GetDim(xdimNum_ - PENULTIMATE_DIM) : perTokenShape->GetDim(xdimNum_ - 1);
    auto scaleKDim = gmmAttrs.transposeWeight ? scaleShape->GetDim(weightdimNum_ - 1) :
                                                scaleShape->GetDim(weightdimNum_ - PENULTIMATE_DIM);
    auto scaleNDim = gmmAttrs.transposeWeight ? scaleShape->GetDim(weightdimNum_ - PENULTIMATE_DIM) :
                                                scaleShape->GetDim(weightdimNum_ - 1);
    OP_CHECK_IF(perTokenMDim != xMDim_,
                OP_LOGE(context->GetNodeName(),
                        "When quantification mode is G-B quantification, the M value in x[%ld] and \
per_token_scale[%ld] should be consistent.",
                        xMDim_, perTokenMDim),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(scaleNDim != (weightNDim_ + PERTILE_GROUP_SIZE - 1) / PERTILE_GROUP_SIZE,
                OP_LOGE(context->GetNodeName(),
                        "When quantification mode is G-B quantification, the N value in scale [%ld] \
must be equal to the N value in weight [%ld] divided by 128.",
                        scaleNDim, weightNDim_),
                return ge::GRAPH_FAILED);
    if (gmmAttrs.groupType == GMM_SPLIT_M) {
        int64_t expectScaleKValue = (weightKDim_ + PERTILE_GROUP_SIZE - 1) / PERTILE_GROUP_SIZE;
        OP_CHECK_IF(perTokenKDim != scaleKDim || scaleKDim != expectScaleKValue,
                    OP_LOGE(context->GetNodeName(),
                            "When quantification mode is G-B quantification, and groupType is 0 (split \
M), the K dim of per_token_scale [%ld] should equal the K dim of scalel [%ld], and its value should be equal to the K \
dim of weight [%ld] divided by 128, rounded up to the next integer.",
                            perTokenKDim, scaleNDim, weightNDim_),
                    return ge::GRAPH_FAILED);
    } else {
        int64_t expectScaleKValue = (weightKDim_ / PERTILE_GROUP_SIZE) + groupNum_;
        OP_CHECK_IF(perTokenKDim != scaleKDim || scaleKDim != expectScaleKValue,
                    OP_LOGE(context->GetNodeName(),
                            "When quantification mode is G-B quantification, and groupType is 2 (split \
K), the K dim of per_token_scale [%ld] should equal the K dim of scale [%ld], its value must be equal to the K dim of \
weight [%ld] divided by 128, plus the groupSize [%ld].",
                            perTokenKDim, scaleKDim, weightKDim_, groupNum_),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedMatmulQuantChecker::CheckShapeForPerGroupQuantParam(const gert::InferShapeContext *context,
                                                                           const GMMAttrs &gmmAttrs) const
{
    if (IsPerTileQuantMode(context, gmmAttrs)) {
        OP_CHECK_IF(CheckGroupedMatmulPerGroupDim(context, gmmAttrs) != ge::GRAPH_SUCCESS ||
                        CheckGroupedMatmulPerTileShape(context, gmmAttrs) != ge::GRAPH_SUCCESS,
                    OP_LOGE(context->GetNodeName(), "CheckShapeForPerGroupQuantParam failed"), return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedMatmulQuantChecker::CheckPertokenShapeInNormalQuantMode(
    const gert::InferShapeContext *context, const GMMAttrs &gmmAttrs, const gert::Shape *perTokenScaleShape) const
{
    OP_CHECK_IF(perTokenScaleShape->GetDimNum() != 1 && perTokenScaleShape->GetDimNum() != 2,
                OP_LOGE(context->GetNodeName(),
                        "In T-C or K-C quant mode, the dim num of perTokenScale should be 1 or 2, \
but the actual is [%zu].",
                        perTokenScaleShape->GetDimNum()),
                return ge::GRAPH_FAILED);
    if (gmmAttrs.groupType == GMM_SPLIT_M) {
        if (perTokenScaleShape->GetDimNum() == 1) {
            OP_CHECK_IF(perTokenScaleShape->GetDim(0) != xMDim_ && perTokenScaleShape->GetDim(0) != groupNum_,
                        OP_LOGE(context->GetNodeName(),
                                "When perTokenScale dim num is 1 in split m scenario, the expected shape of \
perTokenScale is (m,) or (g,), which is (%ld,) or (%ld,), but the actual shape is (%ld,).",
                                xMDim_, groupNum_, perTokenScaleShape->GetDim(0)),
                        return ge::GRAPH_FAILED);
        } else {
            OP_CHECK_IF(perTokenScaleShape->GetDim(0) != groupNum_ || perTokenScaleShape->GetDim(1) != 1,
                        OP_LOGE(context->GetNodeName(),
                                "When perTokenScale dim num is 2 in split m scenario, the expected shape of \
perTokenScale is (g,1), which is (%ld,1), but the actual shape is (%ld, %ld).",
                                groupNum_, perTokenScaleShape->GetDim(0), perTokenScaleShape->GetDim(1)),
                        return ge::GRAPH_FAILED);
        }
    } else {
        // split k
        if (perTokenScaleShape->GetDimNum() == 1) {
            OP_CHECK_IF(perTokenScaleShape->GetDim(0) != groupNum_,
                        OP_LOGE(context->GetNodeName(),
                                "When perTokenScale dim num is 1 in split k scenario, the expected shape of \
perTokenScale is (g,), which is (%ld,), but the actual shape is (%ld,).",
                                groupNum_, perTokenScaleShape->GetDim(0)),
                        return ge::GRAPH_FAILED);
        } else {
            OP_CHECK_IF(
                perTokenScaleShape->GetDim(0) != groupNum_ ||
                    (perTokenScaleShape->GetDim(1) != xMDim_ && perTokenScaleShape->GetDim(1) != 1),
                OP_LOGE(
                    context->GetNodeName(),
                    "When perTokenScale dim num is 2 in split k scenario, the expected shape of perTokenScale is (g,m) \
or (g,1), which is (%ld,%ld) or (%ld,1), but the actual shape is (%ld,%ld).",
                    groupNum_, xMDim_, groupNum_, perTokenScaleShape->GetDim(0), perTokenScaleShape->GetDim(1)),
                return ge::GRAPH_FAILED);
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedMatmulQuantChecker::CheckShapeForQuantParam(const gert::InferShapeContext *context,
                                                                   const GMMAttrs &gmmAttrs) const
{
    if (gmmAttrs.outputDtype == GMM_OUT_DTYPE_INT32) {
        return ge::GRAPH_SUCCESS;
    }
    auto scaleShape = context->GetDynamicInputShape(GMM_INDEX_IN_SCALE, 0);
    auto perTokenScaleShape = context->GetDynamicInputShape(GMM_INDEX_IN_PERTOKEN_SCALE, 0);
    if (IsPerTileQuantMode(context, gmmAttrs)) {
        return CheckShapeForPerGroupQuantParam(context, gmmAttrs);
    }
    // mx在tiling中校验异常场景
    if (scaleShape->GetDimNum() == MXFP_TYPEM_SCALE_DIM_NUM || scaleShape->GetDimNum() == MXFP_TYPEK_SCALE_DIM_NUM) {
        return ge::GRAPH_SUCCESS;
    }

    OP_CHECK_IF(scaleShape->GetDimNum() != 2 && scaleShape->GetDimNum() != 1,
                OP_LOGE(context->GetNodeName(), "The dim num of scale should be 1 or 2, but the actual is [%zu].",
                        scaleShape->GetDimNum()),
                return ge::GRAPH_FAILED);
    if (scaleShape->GetDimNum() == 1) {
        OP_CHECK_IF(scaleShape->GetDim(0) != groupNum_,
                    OP_LOGE(context->GetNodeName(),
                            "The 1st dim value of scale should be g[%ld], but the actual 1st dim value is [%ld].",
                            groupNum_, scaleShape->GetDim(0)),
                    return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF(
            scaleShape->GetDim(0) != groupNum_ || (scaleShape->GetDim(1) != 1 && scaleShape->GetDim(1) != weightNDim_),
            OP_LOGE(context->GetNodeName(),
                    "The 1st dim value of scale should be g[%ld] and 2nd dim value of scale should be 1 or n[%ld] \
, but the actual 1st dim value is [%ld], and the 2nd dim value is [%ld].",
                    groupNum_, weightNDim_, scaleShape->GetDim(0), scaleShape->GetDim(1)),
            return ge::GRAPH_FAILED);
    }
    if (IsNonEmpty(perTokenScaleShape)) {
        if (IsDoubleScaleScenario(scaleShape, perTokenScaleShape)) {
            return ge::GRAPH_SUCCESS;
        }
        if (CheckPertokenShapeInNormalQuantMode(context, gmmAttrs, perTokenScaleShape) == ge::GRAPH_FAILED) {
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedMatmulQuantChecker::GetGroupNumValue(const gert::InferShapeContext *context)
{
    auto groupListShape = context->GetOptionalInputShape(GMM_INDEX_IN_GROUP_LIST);
    OP_CHECK_NULL_WITH_CONTEXT(context, groupListShape);
    OP_CHECK_IF(groupListShape->GetDimNum() != 1,
                OP_LOGE(context->GetNodeName(),
                        "The groupList only support 1 dim num for now, but the actual is [%zu].",
                        groupListShape->GetDimNum()),
                return ge::GRAPH_FAILED);
    groupNum_ = groupListShape->GetDim(0);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedMatmulQuantChecker::CheckShapeForGrouplist(const gert::InferShapeContext *context) const
{
    OP_CHECK_IF(groupNum_ <= 0,
                OP_LOGE(context->GetNodeName(),
                        "The groupList 1st dim value should be greater than 0, but the actual is [%ld].", groupNum_),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(groupNum_ > 1024,
                OP_LOGE(context->GetNodeName(),
                        "Only support 1024 groups at MAX for now, but the actual group number is [%ld].", groupNum_),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedMatmulQuantChecker::CheckShapeValid(const gert::InferShapeContext *context,
                                                           const GMMAttrs &gmmAttrs) const
{
    auto weightShape = context->GetDynamicInputShape(GMM_INDEX_IN_WEIGHT, 0);
    OP_CHECK_IF(
        xKDim_ != weightKDim_,
        OP_LOGE(context->GetNodeName(),
                "The k dim of x should be equal to the k dim of weight, but x's k is [%ld] and weight's k is [%ld].",
                xKDim_, weightKDim_),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckShapeForGrouplist(context) != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "CheckShapeForGrouplist failed."), return ge::GRAPH_FAILED);
    if (gmmAttrs.groupType == GMM_SPLIT_M) {
        OP_CHECK_IF(xdimNum_ != 2 && weightdimNum_ != 3,
                    OP_LOGE(context->GetNodeName(),
                            "When split m, x dim num should be 2, weight dim num should be 3, y dim num should be 2 \
but the actual x dim num is [%zu], actual weight dim num is [%zu].",
                            xdimNum_, weightdimNum_),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(
            weightShape->GetDim(0) != groupNum_,
            OP_LOGE(context->GetNodeName(),
                    "When split m, 1st dim value of weight should be g, which is [%ld], but actual value is [%ld].",
                    groupNum_, weightShape->GetDim(0)),
            return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF(xdimNum_ != 2 && weightdimNum_ != 2,
                    OP_LOGE(context->GetNodeName(),
                            "When split k, x dim num should be 2, weight dim num should be 2, y dim num should be 3 \
but the actual x dim num is [%zu], actual weight dim num is [%zu].",
                            xdimNum_, weightdimNum_),
                    return ge::GRAPH_FAILED);
    }
    OP_CHECK_IF(CheckShapeForBias(context) != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "CheckShapeForBias failed."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(!IsNonEmpty(context->GetDynamicInputShape(GMM_INDEX_IN_SCALE, 0)),
                OP_LOGE(context->GetNodeName(), "The 1st tensor of scale cannot be empty."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckShapeForQuantParam(context, gmmAttrs) != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "CheckShapeForQuantParam failed."), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

bool GroupedMatmulQuantChecker::CheckDataContainsNegativeValue() const
{
    return xMDim_ < 0L || xKDim_ < 0L || weightNDim_ < 0L || weightKDim_ < 0L;
}

bool GroupedMatmulQuantChecker::CheckShapeContainsNegativeValue(const gert::Shape *shape) const
{
    if (shape == nullptr) {
        return false;
    }
    size_t dimNum = shape->GetDimNum();
    for (size_t i = 0; i < dimNum; i++) {
        if (shape->GetDim(i) < 0L) {
            return true;
        }
    }
    return false;
}

bool GroupedMatmulQuantChecker::CheckNonDataInputContainsNegativeValue(const gert::InferShapeContext *context) const
{
    if (groupNum_ < 0L) {
        return true;
    }
    auto biasShape = context->GetDynamicInputShape(GMM_INDEX_IN_BIAS, 0);
    if (IsNonEmpty(biasShape) && CheckShapeContainsNegativeValue(biasShape)) {
        return true;
    }
    auto scaleShape = context->GetDynamicInputShape(GMM_INDEX_IN_SCALE, 0);
    if (IsNonEmpty(scaleShape) && CheckShapeContainsNegativeValue(scaleShape)) {
        return true;
    }
    auto pertokenScaleShape = context->GetDynamicInputShape(GMM_INDEX_IN_PERTOKEN_SCALE, 0);
    if (IsNonEmpty(pertokenScaleShape) && CheckShapeContainsNegativeValue(pertokenScaleShape)) {
        return true;
    }
    return false;
}

ge::graphStatus GroupedMatmulQuantChecker::CheckShape(const gert::InferShapeContext *context,
                                                      const GroupedMatmulCommonUtil &commonUtil) const
{
    if (CheckDataContainsNegativeValue() || CheckNonDataInputContainsNegativeValue(context)) {
        return ge::GRAPH_SUCCESS;
    }
    OP_CHECK_IF(CheckScenarioValidForShape(context, commonUtil.attrsInfo) != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "CheckScenarioValidForShape failed."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckShapeValid(context, commonUtil.attrsInfo) != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "CheckShapeValid failed."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckFormatValid(context) != ge::GRAPH_SUCCESS, OP_LOGE(context->GetNodeName(), "CheckFormatValid failed."),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedMatmulQuantChecker::UpdateShapeY(gert::InferShapeContext *context, size_t idxY,
                                                        std::vector<int64_t> &yDims)
{
    gert::Shape *yShape = context->GetOutputShape(idxY);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);
    yShape->SetDimNum(yDims.size());
    for (size_t dim = 0; dim < yDims.size(); ++dim) {
        yShape->SetDim(dim, yDims[dim]);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedMatmulQuantChecker::InferOutShape(gert::InferShapeContext *context, const GMMAttrs &gmmAttrs)
{
    std::vector<int64_t> yDims = {xMDim_, weightNDim_};
    if (gmmAttrs.groupType == GMM_SPLIT_K) {
        yDims.insert(yDims.begin(), groupNum_);
    }
    OP_CHECK_IF(UpdateShapeY(context, GMM_INDEX_OUT_Y, yDims) != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "Failed to update y shape."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedMatmulQuantChecker::CheckScenarioValidForDtype(const gert::InferDataTypeContext *context,
                                                                      const GMMAttrs &gmmAttrs) const
{
    auto xDtype = context->GetDynamicInputDataType(GMM_INDEX_IN_X, 0);
    auto scaleDtype = context->GetDynamicInputDataType(GMM_INDEX_IN_SCALE, 0);
    auto perTokenScaleDtype = context->GetDynamicInputDataType(GMM_INDEX_IN_PERTOKEN_SCALE, 0);

    if (gmmAttrs.groupType == GMM_SPLIT_K) {
        OP_CHECK_IF(xDtype == ge::DT_INT8 || scaleDtype == ge::DT_INT64 || scaleDtype == ge::DT_UINT64,
                    OP_LOGE(context->GetNodeName(), "When split k, x with int8 dtype or scale with int64/uint64 dtype \
is not supported, but the acutal x dtype is [%s] and actual scale dtype is [%s].",
                            ge::TypeUtils::DataTypeToAscendString(xDtype).GetString(),
                            ge::TypeUtils::DataTypeToAscendString(scaleDtype).GetString()),
                    return ge::GRAPH_FAILED);
    }
    if ((xDtype == ge::DT_HIFLOAT8 || xDtype == ge::DT_FLOAT8_E5M2 ||
         xDtype == ge::DT_FLOAT8_E4M3FN) && perTokenScaleDtype != ge::DT_UNDEFINED) {
        OP_CHECK_IF(
            scaleDtype != perTokenScaleDtype || (scaleDtype != ge::DataType::DT_FLOAT && scaleDtype != ge::DataType::DT_FLOAT8_E8M0),
            OP_LOGE(context->GetNodeName(), "When data type of x is float8/hifloat8, data type of scale [%s] should be \
equal to per_token_scale's dtype [%s], and be float32/float8_e8m0.",
                    ge::TypeUtils::DataTypeToAscendString(scaleDtype).GetString(),
                    ge::TypeUtils::DataTypeToAscendString(perTokenScaleDtype).GetString()), return ge::GRAPH_FAILED);
    } else if (xDtype == ge::DT_FLOAT4_E2M1) {
        OP_CHECK_IF(
            scaleDtype != ge::DataType::DT_FLOAT8_E8M0,
            OP_LOGE(context->GetNodeName(), "When data type of x is float4, data type of scale [%s] should be \
equal to float8_e8m0.",
                    ge::TypeUtils::DataTypeToAscendString(scaleDtype).GetString()), return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedMatmulQuantChecker::CheckDtype(const gert::InferDataTypeContext *context,
                                                      const GroupedMatmulCommonUtil &commonUtil) const
{
    OP_CHECK_IF(CheckDtypeValid(context) != ge::GRAPH_SUCCESS, OP_LOGE(context->GetNodeName(), "CheckDtypeValid failed."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckScenarioValidForDtype(context, commonUtil.attrsInfo) != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "CheckScenarioValidForDtype failed."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedMatmulQuantChecker::InferOutDtype(gert::InferDataTypeContext *context)
{
    auto attrs = context->GetAttrs();
    const int64_t *outputDtype = attrs->GetInt(GMM_INDEX_ATTR_OUTPUT_DTYPE);
    ge::DataType yDtype = context->GetDynamicInputDataType(GMM_INDEX_IN_X, 0);
    auto it = GMM_OUTPUT_DTYPE_MAP.find(*outputDtype);
    OP_CHECK_IF(it == GMM_OUTPUT_DTYPE_MAP.end(),
                OP_LOGE(context->GetNodeName(), "The output dtype should be int8, bfloat16, int32, float16 or float32 \
, but the actual output dtype is [%s].",
                        ge::TypeUtils::DataTypeToAscendString(yDtype).GetString()),
                return ge::GRAPH_FAILED);
    yDtype = it->second;
    context->SetOutputDataType(GMM_INDEX_OUT_Y, yDtype);
    return ge::GRAPH_SUCCESS;
}
} // namespace ops