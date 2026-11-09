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
 * \file moe_distribute_dispatch_infer_v2.cpp
 * \brief
 */
#include "register/op_impl_registry.h"
#include "mc2_log.h"
#include "platform/platform_info.h"
using namespace ge;
namespace ops {
static constexpr size_t DIM_ONE = 1UL;
static constexpr size_t DIM_TWO = 2UL;
static constexpr int64_t NEG_ONE = -1;
static constexpr int64_t RANK_NUM_PER_NODE = 8;

static constexpr size_t COMBINE_INPUT_EXPAND_X_INDEX = 0;
static constexpr size_t COMBINE_INPUT_EXPERT_IDS_INDEX = 1;
static constexpr size_t COMBINE_OUTPUT_X_INDEX = 0;

static ge::graphStatus InferShapeMoeDistributeCombineV2(gert::InferShapeContext *context)
{
    if (context == nullptr){
        return ge::GRAPH_FAILED;
    }
    OP_LOGD(context->GetNodeName(), "Begin to do InferShapeMoeDistributeCombineV2.");
    // 获取输入shape
    const gert::Shape *expandXShape = context->GetInputShape(COMBINE_INPUT_EXPAND_X_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, expandXShape);
    const gert::Shape *expertIdsShape = context->GetInputShape(COMBINE_INPUT_EXPERT_IDS_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, expertIdsShape);
    gert::Shape *xShape = context->GetOutputShape(COMBINE_OUTPUT_X_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, xShape);

    int64_t bs = ((expertIdsShape->GetDimNum() == 1U) ? NEG_ONE : expertIdsShape->GetDim(0));
    int64_t h = ((expandXShape->GetDimNum() == 1U) ? NEG_ONE : expandXShape->GetDim(1));

    xShape->SetDimNum(DIM_TWO);
    xShape->SetDim(0U, bs);
    xShape->SetDim(1U, h);

    OP_LOGD(context->GetNodeName(), "x shape shape is :%s after infershape.",
        Ops::Base::ToString(*xShape).c_str());
    OP_LOGD(context->GetNodeName(), "End to do InferShapeMoeDistributeCombineV2.");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeMoeDistributeCombineV2(gert::InferDataTypeContext *context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferDataTypeMoeDistributeCombineV2.");
    auto xDtype = context->GetInputDataType(COMBINE_INPUT_EXPAND_X_INDEX);
    context->SetOutputDataType(COMBINE_OUTPUT_X_INDEX, xDtype);
    OP_LOGD(context->GetNodeName(), "End to do InferDataTypeMoeDistributeCombineV2.");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(MoeDistributeCombineV2)
    .InferShape(InferShapeMoeDistributeCombineV2)
    .InferDataType(InferDataTypeMoeDistributeCombineV2);
}  // namespace ops