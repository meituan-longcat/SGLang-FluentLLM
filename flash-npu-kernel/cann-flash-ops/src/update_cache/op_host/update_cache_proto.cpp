/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file update_cache_proto.cpp
 * \brief
 */

#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "error/ops_error.h"

using namespace ge;
namespace ops {
// input indices
constexpr size_t INPUT_IDX_CACHE_K_ROPE = 3;
constexpr size_t INPUT_IDX_CACHE_KV_CACHE = 4;
constexpr size_t INPUT_IDX_CACHE_STATUS = 5;
constexpr size_t INPUT_IDX_CACHE_AGE = 6;
// output indices (in-place cache tensors)
constexpr size_t OUTPUT_IDX_CACHE_K_ROPE = 0;
constexpr size_t OUTPUT_IDX_CACHE_KV_CACHE = 1;
constexpr size_t OUTPUT_IDX_CACHE_STATUS = 2;
constexpr size_t OUTPUT_IDX_CACHE_AGE = 3;

static ge::graphStatus InferShape4UpdateCache(gert::InferShapeContext* context)
{
    OPS_LOG_I(context->GetNodeName(), "Begin to do InferShape4UpdateCache.");

    const gert::Shape* cacheKRope = context->GetInputShape(INPUT_IDX_CACHE_K_ROPE);
    OPS_LOG_E_IF_NULL(context, cacheKRope, return ge::GRAPH_FAILED);
    auto outKRope = context->GetOutputShape(OUTPUT_IDX_CACHE_K_ROPE);
    OPS_LOG_E_IF_NULL(context, outKRope, return ge::GRAPH_FAILED);
    *outKRope = *cacheKRope;

    const gert::Shape* cacheKv = context->GetInputShape(INPUT_IDX_CACHE_KV_CACHE);
    OPS_LOG_E_IF_NULL(context, cacheKv, return ge::GRAPH_FAILED);
    auto outKv = context->GetOutputShape(OUTPUT_IDX_CACHE_KV_CACHE);
    OPS_LOG_E_IF_NULL(context, outKv, return ge::GRAPH_FAILED);
    *outKv = *cacheKv;

    const gert::Shape* cacheStatus = context->GetInputShape(INPUT_IDX_CACHE_STATUS);
    OPS_LOG_E_IF_NULL(context, cacheStatus, return ge::GRAPH_FAILED);
    auto outStatus = context->GetOutputShape(OUTPUT_IDX_CACHE_STATUS);
    OPS_LOG_E_IF_NULL(context, outStatus, return ge::GRAPH_FAILED);
    *outStatus = *cacheStatus;

    const gert::Shape* cacheAge = context->GetInputShape(INPUT_IDX_CACHE_AGE);
    OPS_LOG_E_IF_NULL(context, cacheAge, return ge::GRAPH_FAILED);
    auto outAge = context->GetOutputShape(OUTPUT_IDX_CACHE_AGE);
    OPS_LOG_E_IF_NULL(context, outAge, return ge::GRAPH_FAILED);
    *outAge = *cacheAge;

    OPS_LOG_I(context->GetNodeName(), "End to do InferShape4UpdateCache");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDtype4UpdateCache(gert::InferDataTypeContext* context)
{
    OPS_LOG_I(context->GetNodeName(), "InferDtype4UpdateCache enter");
    context->SetOutputDataType(OUTPUT_IDX_CACHE_K_ROPE, context->GetInputDataType(INPUT_IDX_CACHE_K_ROPE));
    context->SetOutputDataType(OUTPUT_IDX_CACHE_KV_CACHE, context->GetInputDataType(INPUT_IDX_CACHE_KV_CACHE));
    context->SetOutputDataType(OUTPUT_IDX_CACHE_STATUS, ge::DT_INT32);
    context->SetOutputDataType(OUTPUT_IDX_CACHE_AGE, ge::DT_INT32);
    OPS_LOG_I(context->GetNodeName(), "InferDtype4UpdateCache end");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(UpdateCache)
    .InferShape(InferShape4UpdateCache)
    .InferDataType(InferDtype4UpdateCache);
}  // namespace ops
