/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file gather_elements_v2_infershape.cpp
 * \brief
 */
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "util/shape_util.h"

using namespace ge;
using namespace Ops::Base;
namespace ops {
const size_t IDX_X = 0;
const size_t IDX_INDEX = 1;
const size_t IDX_Y = 0;

// output dtype is the same as x
static graphStatus InferDataType4GatherElementsV2(gert::InferDataTypeContext* context)
{
    auto x_dtype = context->GetInputDataType(IDX_X);
    context->SetOutputDataType(IDX_Y, x_dtype);

    return GRAPH_SUCCESS;
}

// output shape is the same as index (gather semantics: every index element yields one output element).
// unknown rank (-2) and unknown dim (-1) are both propagated by copying the index shape.
static ge::graphStatus InferShape4GatherElementsV2(gert::InferShapeContext* context)
{
    const gert::Shape* index_shape = context->GetInputShape(IDX_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, index_shape);
    gert::Shape* y_shape = context->GetOutputShape(IDX_Y);
    OP_CHECK_NULL_WITH_CONTEXT(context, y_shape);

    // dynamic rank (-2): keep the output rank unknown
    if (IsUnknownRank(*index_shape)) {
        OP_LOGD(context->GetNodeName(), "index shape is UnknownRank, set y shape to UnknownRank");
        *y_shape = *index_shape;
        return ge::GRAPH_SUCCESS;
    }

    // static rank (may carry unknown dims, i.e. -1): copy index shape verbatim
    *y_shape = *index_shape;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(GatherElementsV2)
    .InferShape(InferShape4GatherElementsV2)
    .InferDataType(InferDataType4GatherElementsV2);
} // namespace ops
