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
 * \file gen_from_cache_indices_proto.cpp
 * \brief
 */

#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "error/ops_error.h"

using namespace ge;
namespace ops {
constexpr size_t INPUT_IDX_TOPK = 0;
constexpr size_t OUTPUT_IDX_FROM = 0;
constexpr size_t OUTPUT_IDX_ACTSEQ = 1;
constexpr size_t OUTPUT_IDX_WRITESLOT = 2;
constexpr size_t OUTPUT_IDX_UNISRC = 3;

static ge::graphStatus InferShape4GenFromCacheIndices(gert::InferShapeContext* context)
{
    OPS_LOG_I(context->GetNodeName(), "Begin to do InferShape4GenFromCacheIndices.");
    // selection_topk_indices: [B,S,TOPK]
    const gert::Shape* topkShape = context->GetInputShape(INPUT_IDX_TOPK);
    OPS_LOG_E_IF_NULL(context, topkShape, return ge::GRAPH_FAILED);

    // from_cache_indices == selection_topk_indices shape
    auto fromShape = context->GetOutputShape(OUTPUT_IDX_FROM);
    OPS_LOG_E_IF_NULL(context, fromShape, return ge::GRAPH_FAILED);
    *fromShape = *topkShape;

    // selection_kv_actual_seq: [B*S]  (headnum == 1)
    auto actShape = context->GetOutputShape(OUTPUT_IDX_ACTSEQ);
    OPS_LOG_E_IF_NULL(context, actShape, return ge::GRAPH_FAILED);
    int64_t bs = 1;
    size_t nd = topkShape->GetDimNum();
    if (nd >= 2) {
        const int64_t batch = topkShape->GetDim(0);
        const int64_t queries = topkShape->GetDim(1);
        // Preserve unknown dimensions during GE shape-range inference.
        // Multiplying -1 by a known extent creates an invalid dimension.
        bs = (batch < 0 || queries < 0) ? -1 : batch * queries;
    } else {
        bs = topkShape->GetDim(0);
    }
    actShape->SetDimNum(1);
    actShape->SetDim(0, bs);

    // write_slot == selection_topk_indices shape [B,S,TOPK]
    auto wsShape = context->GetOutputShape(OUTPUT_IDX_WRITESLOT);
    OPS_LOG_E_IF_NULL(context, wsShape, return ge::GRAPH_FAILED);
    *wsShape = *topkShape;

    // unified_slots_in_lru_or_cache == selection_topk_indices shape [B,S,TOPK]
    auto uniShape = context->GetOutputShape(OUTPUT_IDX_UNISRC);
    OPS_LOG_E_IF_NULL(context, uniShape, return ge::GRAPH_FAILED);
    *uniShape = *topkShape;

    OPS_LOG_I(context->GetNodeName(), "End to do InferShape4GenFromCacheIndices");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDtype4GenFromCacheIndices(gert::InferDataTypeContext* context)
{
    OPS_LOG_I(context->GetNodeName(), "InferDtype4GenFromCacheIndices enter");
    context->SetOutputDataType(OUTPUT_IDX_FROM, ge::DT_INT32);
    context->SetOutputDataType(OUTPUT_IDX_ACTSEQ, ge::DT_INT32);
    context->SetOutputDataType(OUTPUT_IDX_WRITESLOT, ge::DT_INT32);
    context->SetOutputDataType(OUTPUT_IDX_UNISRC, ge::DT_INT32);
    OPS_LOG_I(context->GetNodeName(), "InferDtype4GenFromCacheIndices end");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(GenFromCacheIndices)
    .InferShape(InferShape4GenFromCacheIndices)
    .InferDataType(InferDtype4GenFromCacheIndices);
}  // namespace ops
