/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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
 * \file gather_selection_kv_cache.cc
 * \brief
 */

#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "error/ops_error.h"

using namespace ge;
namespace ops {
constexpr size_t INPUT_IDX_SELECTION_K_ROPE = 0;
constexpr size_t INPUT_IDX_SELECTION_KV_CACHE = 1;
constexpr size_t INPUT_IDX_SELECTION_KV_BLOCK_TABLE = 2;
constexpr size_t INPUT_IDX_SELECTION_KV_BLOCK_STATUS = 3;
constexpr size_t OUTPUT_IDX_Y = 4;
const int32_t INDEX_INPUT_0 = 0;
const int32_t INDEX_INPUT_1 = 1;
const int32_t INDEX_INPUT_2 = 2;
const int32_t INDEX_INPUT_3 = 3;
const int32_t INDEX_INPUT_4 = 4;
const int32_t INDEX_OUTPUT_0 = 0;
const int32_t INDEX_OUTPUT_1 = 1;
const int32_t INDEX_OUTPUT_2 = 2;
const int32_t INDEX_OUTPUT_3 = 3;
const int32_t INDEX_OUTPUT_4 = 4;

static ge::graphStatus InferShape4GatherSelectionKvCache(gert::InferShapeContext* context)
{
    OPS_LOG_I(context->GetNodeName(), "Begin to do InferShape4GatherSelectionKvCache.");

    const gert::Shape* selectionKRopeShape = context->GetInputShape(INDEX_INPUT_0);
    OPS_LOG_E_IF_NULL(context, selectionKRopeShape, return ge::GRAPH_FAILED);
    auto selectionKRopeInplaceShape = context->GetOutputShape(INDEX_OUTPUT_0);
    *selectionKRopeInplaceShape = *selectionKRopeShape;

    const gert::Shape* selectionKvCacheShape = context->GetInputShape(INDEX_INPUT_1);
    OPS_LOG_E_IF_NULL(context, selectionKvCacheShape, return ge::GRAPH_FAILED);
    auto selectionKvCacheInplaceShape = context->GetOutputShape(INDEX_OUTPUT_1);
    *selectionKvCacheInplaceShape = *selectionKvCacheShape;

    const gert::Shape* selectionKvBlockTableShape = context->GetInputShape(INDEX_INPUT_2);
    OPS_LOG_E_IF_NULL(context, selectionKvBlockTableShape, return ge::GRAPH_FAILED);
    auto selectionKvBlockTableInplaceShape = context->GetOutputShape(INDEX_OUTPUT_2);
    *selectionKvBlockTableInplaceShape = *selectionKvBlockTableShape;

    const gert::Shape* selectionKvBlocStatusShape = context->GetInputShape(INDEX_INPUT_3);
    OPS_LOG_E_IF_NULL(context, selectionKvBlocStatusShape, return ge::GRAPH_FAILED);
    auto selectionKvBlocStatusInplaceShape = context->GetOutputShape(INDEX_OUTPUT_3);
    *selectionKvBlocStatusInplaceShape = *selectionKvBlocStatusShape;
    
    gert::Shape* selectionKvActualSeqShape = context->GetOutputShape(INDEX_OUTPUT_4);
    OPS_LOG_E_IF_NULL(context, selectionKvActualSeqShape, return ge::GRAPH_FAILED);
    *selectionKvActualSeqShape = *selectionKvBlockTableShape;
    int64_t selectionKvBlockTableDim = static_cast<int64_t>(selectionKvBlockTableShape->GetDimNum());
    // 设置selection_kv_actual_seq的shape
    selectionKvActualSeqShape->SetDimNum(selectionKvBlockTableDim - 1);

    OPS_LOG_I(context->GetNodeName(), "End to do InferShape4GatherSelectionKvCache");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDtype4GatherSelectionKvCache(gert::InferDataTypeContext* context)
{
    OPS_LOG_I(context->GetNodeName(), "InferDtype4GatherSelectionKvCache enter");

    const auto selection_k_rope_dtype = context->GetInputDataType(INPUT_IDX_SELECTION_K_ROPE);
    const auto selection_kv_cache_dtype = context->GetInputDataType(INPUT_IDX_SELECTION_KV_CACHE);
    const auto selection_kv_block_table_dtype = context->GetInputDataType(INPUT_IDX_SELECTION_KV_BLOCK_TABLE);
    const auto selection_kv_block_status_dtype = context->GetInputDataType(INPUT_IDX_SELECTION_KV_BLOCK_STATUS);
    context->SetOutputDataType(INDEX_OUTPUT_0, selection_k_rope_dtype);
    context->SetOutputDataType(INDEX_OUTPUT_1, selection_kv_cache_dtype);
    context->SetOutputDataType(INDEX_OUTPUT_2, selection_kv_block_table_dtype);
    context->SetOutputDataType(INDEX_OUTPUT_3, selection_kv_block_status_dtype);
    context->SetOutputDataType(INDEX_OUTPUT_4, ge::DT_INT32);

    OPS_LOG_I(context->GetNodeName(), "InferDtype4GatherSelectionKvCache end");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(GatherSelectionKvCache)
    .InferShape(InferShape4GatherSelectionKvCache)
    .InferDataType(InferDtype4GatherSelectionKvCache);
}  // namespace ops

