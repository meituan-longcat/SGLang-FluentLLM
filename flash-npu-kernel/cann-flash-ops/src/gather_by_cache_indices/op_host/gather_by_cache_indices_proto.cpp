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
 * \file gather_by_cache_indices_proto.cpp
 * \brief
 */

#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "error/ops_error.h"

using namespace ge;
namespace ops {
constexpr size_t INPUT_IDX_CACHE_K_ROPE = 0;
constexpr size_t INPUT_IDX_CACHE_KV_CACHE = 1;
constexpr size_t INPUT_IDX_FROM_CACHE = 2;
constexpr size_t OUTPUT_IDX_SEL_K_ROPE = 0;
constexpr size_t OUTPUT_IDX_SEL_KV_CACHE = 1;
constexpr size_t OUTPUT_IDX_CUR_KV_WRITE_SLOTS = 2;

// selection_*_out 为扁平 [totalUnits, dim]：totalUnits = prod(from_cache_indices shape)，
// dim 取自对应 cache 的最后一维（k_rope / kv）。
static ge::graphStatus InferShape4GatherByCacheIndices(gert::InferShapeContext* context)
{
    OPS_LOG_I(context->GetNodeName(), "Begin to do InferShape4GatherByCacheIndices.");

    const gert::Shape* cacheKRopeShape = context->GetInputShape(INPUT_IDX_CACHE_K_ROPE);
    OPS_LOG_E_IF_NULL(context, cacheKRopeShape, return ge::GRAPH_FAILED);
    const gert::Shape* cacheKvShape = context->GetInputShape(INPUT_IDX_CACHE_KV_CACHE);
    OPS_LOG_E_IF_NULL(context, cacheKvShape, return ge::GRAPH_FAILED);
    const gert::Shape* fromShape = context->GetInputShape(INPUT_IDX_FROM_CACHE);
    OPS_LOG_E_IF_NULL(context, fromShape, return ge::GRAPH_FAILED);

    int64_t totalUnits = 1;
    for (size_t i = 0; i < fromShape->GetDimNum(); ++i) {
        totalUnits *= fromShape->GetDim(i);
    }
    int64_t kRopeDim = cacheKRopeShape->GetDim(cacheKRopeShape->GetDimNum() - 1);
    int64_t kvDim = cacheKvShape->GetDim(cacheKvShape->GetDimNum() - 1);
    (void)totalUnits;

    // 分页输出：selection 为 [S_BLOCK_NUM, selKvBlockSize, dim]（S_BLOCK_NUM/块长由 attr 指定）。
    auto* attrs = context->GetAttrs();
    OPS_LOG_E_IF_NULL(context, attrs, return ge::GRAPH_FAILED);
    const int64_t* pNum = attrs->GetAttrPointer<int64_t>(1);   // selection_kv_block_num
    const int64_t* pSz  = attrs->GetAttrPointer<int64_t>(2);   // selection_kv_block_size
    int64_t selBlkNum = (pNum != nullptr) ? *pNum : 0;
    int64_t selBlkSz  = (pSz != nullptr && *pSz > 0) ? *pSz : 128;

    auto selKRopeShape = context->GetOutputShape(OUTPUT_IDX_SEL_K_ROPE);
    OPS_LOG_E_IF_NULL(context, selKRopeShape, return ge::GRAPH_FAILED);
    selKRopeShape->SetDimNum(3);
    selKRopeShape->SetDim(0, selBlkNum);
    selKRopeShape->SetDim(1, selBlkSz);
    selKRopeShape->SetDim(2, kRopeDim);

    auto selKvShape = context->GetOutputShape(OUTPUT_IDX_SEL_KV_CACHE);
    OPS_LOG_E_IF_NULL(context, selKvShape, return ge::GRAPH_FAILED);
    selKvShape->SetDimNum(3);
    selKvShape->SetDim(0, selBlkNum);
    selKvShape->SetDim(1, selBlkSz);
    selKvShape->SetDim(2, kvDim);

    // cur_kv_write_slots [B, S, S]（int32，初始化为 -1）：B、S 都取自实际输入
    // from_cache_indices [B,S,K]。不取 cache_k_rope dim0：该维可能是预分配的 max_batch_size，
    // 而实际参与计算的 batch 以 from_cache_indices 为准（tiling 的 unitsPerRealBatch 同此约定）。
    int64_t batch = fromShape->GetDim(0);
    int64_t seq = fromShape->GetDim(1);
    auto curWriteSlotsShape = context->GetOutputShape(OUTPUT_IDX_CUR_KV_WRITE_SLOTS);
    OPS_LOG_E_IF_NULL(context, curWriteSlotsShape, return ge::GRAPH_FAILED);
    curWriteSlotsShape->SetDimNum(3);
    curWriteSlotsShape->SetDim(0, batch);
    curWriteSlotsShape->SetDim(1, seq);
    curWriteSlotsShape->SetDim(2, seq);

    OPS_LOG_I(context->GetNodeName(), "End to do InferShape4GatherByCacheIndices");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDtype4GatherByCacheIndices(gert::InferDataTypeContext* context)
{
    OPS_LOG_I(context->GetNodeName(), "InferDtype4GatherByCacheIndices enter");
    const auto cacheKRopeDtype = context->GetInputDataType(INPUT_IDX_CACHE_K_ROPE);
    const auto cacheKvDtype = context->GetInputDataType(INPUT_IDX_CACHE_KV_CACHE);
    context->SetOutputDataType(OUTPUT_IDX_SEL_K_ROPE, cacheKRopeDtype);
    context->SetOutputDataType(OUTPUT_IDX_SEL_KV_CACHE, cacheKvDtype);
    context->SetOutputDataType(OUTPUT_IDX_CUR_KV_WRITE_SLOTS, ge::DT_INT32);
    OPS_LOG_I(context->GetNodeName(), "InferDtype4GatherByCacheIndices end");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(GatherByCacheIndices)
    .InferShape(InferShape4GatherByCacheIndices)
    .InferDataType(InferDtype4GatherByCacheIndices);
}  // namespace ops
