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
 * \file mc2_common_infershape.cpp
 * \brief
 */

#include "mc2_common_infershape.h"
#include "mc2_log.h"

using namespace ge;
namespace ops {
// infershape 公共函数
ge::graphStatus CommonParamCheck(
    const gert::InferShapeContext* context, const size_t isTransAIndex, const size_t isTransBIndex, CommParas& commParas)
{
    commParas.x1MatrixShape = context->GetInputShape(0);
    OPS_CHECK_NULL_WITH_CONTEXT(context, commParas.x1MatrixShape);
    commParas.x2MatrixShape = context->GetInputShape(1);
    OPS_CHECK_NULL_WITH_CONTEXT(context, commParas.x2MatrixShape);
    if (commParas.x1MatrixShape->GetDimNum() != SUPPORT_DIM_SIZE ||
        commParas.x2MatrixShape->GetDimNum() != SUPPORT_DIM_SIZE) {
        OP_LOGE(context->GetNodeName(), "Input x1 and Input x2 must be the same with 2 dims.");
        return ge::GRAPH_FAILED;
    }
    auto attrs = context->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const bool* isTransA = attrs->GetAttrPointer<bool>(isTransAIndex);
    const bool* isTransB = attrs->GetAttrPointer<bool>(isTransBIndex);
    const int64_t* rankSizeAttr = attrs->GetAttrPointer<int64_t>(RANK_SIZE);

    const char* groupStr = attrs->GetAttrPointer<char>(GROUP);
    OP_LOGE_IF(groupStr == nullptr, GRAPH_FAILED, context->GetNodeName(), "Get group failed.");
    commParas.rankSize = -1;
    uint32_t rankNum = 0;
    if (*rankSizeAttr <= 0) {
        if ((Mc2Hcom::MC2HcomTopology::CommGetInstSizeByGroup(groupStr, &rankNum)) != HCCL_SUCCESS || rankNum == 0) {
            OP_LOGE(
                context->GetNodeName(), "Get rank size failed, group [%s], rankSize [%u]", groupStr, rankNum);
            return ge::GRAPH_FAILED;
        } else {
            commParas.rankSize = rankNum;
        }
        commParas.rankSize = static_cast<int64_t>(rankNum);
    } else {
        commParas.rankSize = *rankSizeAttr;
    }

    commParas.dimM = !(*isTransA) ? commParas.x1MatrixShape->GetDim(0) : commParas.x1MatrixShape->GetDim(1);
    commParas.dimKX1 = !(*isTransA) ? commParas.x1MatrixShape->GetDim(1) : commParas.x1MatrixShape->GetDim(0);
    commParas.dimKX2 = !(*isTransB) ? commParas.x2MatrixShape->GetDim(0) : commParas.x2MatrixShape->GetDim(1);
    commParas.dimN = !(*isTransB) ? commParas.x2MatrixShape->GetDim(1) : commParas.x2MatrixShape->GetDim(0);

    OP_LOGI(
        context->GetNodeName(),
        "group = %s isTransA %d isTransB %d x1.M = [%ld] x1.K = [%ld]"
        " x2.K = [%ld] x2.N = [%ld] rankSize = [%ld].",
        groupStr, (*isTransA), (*isTransB), commParas.x1MatrixShape->GetDim(0), commParas.x1MatrixShape->GetDim(1),
        commParas.x2MatrixShape->GetDim(0), commParas.x2MatrixShape->GetDim(1), commParas.rankSize);
    if (commParas.dimKX1 != commParas.dimKX2) {
        OP_LOGE(
            context->GetNodeName(), "Input x1/x2 dim k must be same, but given x1.k %ld, x2.k %ld.", commParas.dimKX1,
            commParas.dimKX2);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AllGatherMatmulInferYShape(gert::InferShapeContext* context, CommParas& commParas)
{
    OP_LOGE_IF(
        CommonParamCheck(context, AG_IS_TRANS_A, AG_IS_TRANS_B, commParas) != GRAPH_SUCCESS, GRAPH_FAILED,
        context->GetNodeName(), "CommonParamCheck excute failed.");
    // 动态shape入图时 m轴-1时，不再进行(dimM * rankSize)的处理
    if (commParas.dimM == -1) {
        commParas.rankSize = 1;
    }
    // 不支持k = 0
    if (commParas.dimKX1 == 0) {
        commParas.dimM = commParas.dimN = 0;
        OP_LOGE(context->GetNodeName(), "X1/X2 are empty tensors with zero dimK.");
        return ge::GRAPH_FAILED;
    }
    gert::Shape* yShape = context->GetOutputShape(0);
    OPS_CHECK_NULL_WITH_CONTEXT(context, yShape);
    yShape->SetDimNum(SUPPORT_DIM_SIZE);
    yShape->SetDim(0, commParas.dimM * commParas.rankSize);
    yShape->SetDim(1, commParas.dimN);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AllGatherMatmulInferGatherOutShape(gert::InferShapeContext* context, const CommParas& commParas,
                                                   const size_t gatherIndex)
{
    if (context->GetAttrs() == nullptr) {
        OP_LOGE(context->GetNodeName(), "get attrs failed.");
        return ge::GRAPH_FAILED;
    }
    const bool* isGatherOut = context->GetAttrs()->GetAttrPointer<bool>(gatherIndex);
    OPS_CHECK_NULL_WITH_CONTEXT(context, isGatherOut);
    gert::Shape* gatherOutShape = context->GetOutputShape(1);
    OPS_CHECK_NULL_WITH_CONTEXT(context, gatherOutShape);
    if (*isGatherOut) {
        gatherOutShape->SetDimNum(SUPPORT_DIM_SIZE);
        gatherOutShape->SetDim(0, commParas.dimM * commParas.rankSize);
        gatherOutShape->SetDim(1, commParas.dimKX1);
    } else {
        gatherOutShape->SetDimNum(1);
        gatherOutShape->SetDim(0, 0);
    }
    return GRAPH_SUCCESS;
}

ge::graphStatus AllGatherMatmulCommonInferShape(gert::InferShapeContext* context, const size_t gatherIndex)
{
    CommParas commParas;
    OP_LOGE_IF(
        AllGatherMatmulInferYShape(context, commParas) != GRAPH_SUCCESS, GRAPH_FAILED,
        context->GetNodeName(), "InferShapeAllGatherMatmul inferYshape excute failed.");
    
    OP_LOGE_IF(
        AllGatherMatmulInferGatherOutShape(context, commParas, gatherIndex) != GRAPH_SUCCESS, GRAPH_FAILED,
        context->GetNodeName(), "InferShapeAllGatherMatmul inferYshape excute failed.");

    return GRAPH_SUCCESS;
}

ge::graphStatus InferMatmulReduceScatterCommon(gert::InferShapeContext* context)
{
    CommParas commParas;
    OP_LOGE_IF(
        CommonParamCheck(context, RS_IS_TRANS_A, RS_IS_TRANS_B, commParas) != GRAPH_SUCCESS, GRAPH_FAILED,
        context->GetNodeName(), "CommonParamCheck excute failed.");
    // 动态shape入图时 m轴-1时，不再进行(dimM / rankSize)的处理
    if (commParas.dimM == -1) {
        commParas.rankSize = 1;
    }
    if (commParas.dimKX1 == 0) {
        commParas.dimM = commParas.dimN = 0;
        OP_LOGE(context->GetNodeName(), "X1/X2 are empty tensors with zero dimK");
        return ge::GRAPH_FAILED;
    }
    gert::Shape* yShape = context->GetOutputShape(0);
    OPS_CHECK_NULL_WITH_CONTEXT(context, yShape);
    yShape->SetDimNum(SUPPORT_DIM_SIZE);
    yShape->SetDim(0, commParas.dimM / commParas.rankSize);
    yShape->SetDim(1, commParas.dimN);
    return GRAPH_SUCCESS;
}
} // namespace ops