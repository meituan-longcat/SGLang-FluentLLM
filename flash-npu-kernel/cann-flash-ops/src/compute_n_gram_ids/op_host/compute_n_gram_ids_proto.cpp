/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file compute_n_gram_ids_proto.cpp
 * \brief
 */
#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>

#include "error/ops_error.h"

using namespace ge;
namespace ops {

graphStatus InferShape4ComputeNGramIds(gert::InferShapeContext* context)
{
    const gert::Shape* tokens_shape = context->GetInputShape(3);
    auto token_num = tokens_shape->GetDim(0);
    auto oe_n = *(context->GetAttrs()->GetAttrPointer<int64_t>(1));
    auto oe_k = *(context->GetAttrs()->GetAttrPointer<int64_t>(2));
    gert::Shape* outShape = context->GetOutputShape(0);
    *outShape = gert::Shape({token_num, (oe_n - 1) * oe_k});
    return GRAPH_SUCCESS;
}

graphStatus InferDtype4ComputeNGramIds(gert::InferDataTypeContext* context)
{
    const auto inputDataType = context->GetInputDataType(3);
    context->SetOutputDataType(0, inputDataType);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(ComputeNGramIds)
    .InferShape(InferShape4ComputeNGramIds)
    .InferDataType(InferDtype4ComputeNGramIds);
}  // namespace ops
