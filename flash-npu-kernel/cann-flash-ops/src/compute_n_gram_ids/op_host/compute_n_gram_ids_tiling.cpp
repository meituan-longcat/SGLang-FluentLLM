/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "compute_n_gram_ids_tiling.h"

using namespace ge;

namespace optiling {

namespace {
constexpr int64_t ATTR_BATCH_SIZE = 0;
constexpr int64_t ATTR_OE_N = 1;
constexpr int64_t ATTR_OE_K = 2;
constexpr int64_t ATTR_MAX_CTX = 3;
constexpr int64_t ATTR_EOS_TOKEN_ID = 4;
}  // namespace

ge::graphStatus ComputeNGramIdsTiling::ResolvePlatform()
{
    auto platInfo = context_->GetPlatformInfo();
    if (platInfo == nullptr) {
        auto* compileInfo = context_->GetCompileInfo<ComputeNGramIdsCompileInfo>();
        OPS_ERR_IF(compileInfo == nullptr, OPS_LOG_E(context_, "compile info is null"),
                   return GRAPH_FAILED);
        coreNum_ = compileInfo->coreNum;
        ubSize_ = compileInfo->ubSize;
    } else {
        auto ascendc = platform_ascendc::PlatformAscendC(platInfo);
        coreNum_ = ascendc.GetCoreNumAiv();
        ascendc.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize_);
        socVersion_ = ascendc.GetSocVersion();
    }
    return GRAPH_SUCCESS;
}

ge::graphStatus ComputeNGramIdsTiling::PopulateTiling()
{
    auto batch_size = *(context_->GetAttrs()->GetAttrPointer<int64_t>(0));
    auto oe_n = *(context_->GetAttrs()->GetAttrPointer<int64_t>(1));
    auto oe_k = *(context_->GetAttrs()->GetAttrPointer<int64_t>(2));
    auto maxContextLen = *(context_->GetAttrs()->GetAttrPointer<int64_t>(3));
    auto eosTokenId = *(context_->GetAttrs()->GetAttrPointer<int64_t>(4));
    // const gert::Shape* tokens_shape = context_->GetInputShape(3);
    // auto token_num = tokens_shape->GetDim(0);
    // TODO 仅通过batch_size划分对prefill不好，如果token_num长，使用别的切分方式
    uint32_t total_task = batch_size * (oe_n - 1) * oe_k;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    aivNum = std::min(aivNum, total_task);

    context_->SetBlockDim(aivNum);
    tilingData_.set_totalTask(total_task);
    tilingData_.set_batchSize(batch_size);
    tilingData_.set_coreNum(aivNum);
    tilingData_.set_oeN(oe_n);
    tilingData_.set_oeK(oe_k);
    tilingData_.set_maxContextLen(maxContextLen);
    tilingData_.set_eosTokenId(eosTokenId);
    return GRAPH_SUCCESS;
}

ge::graphStatus ComputeNGramIdsTiling::Finalize()
{
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->SetTilingKey(1);
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    size_t *currentWorkspace = context_->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return GRAPH_SUCCESS;
}

ge::graphStatus ComputeNGramIdsTiling::DoTiling()
{
    OPS_ERR_IF(ResolvePlatform() != GRAPH_SUCCESS,
               OPS_LOG_E(context_, "resolve platform failed"),
               return GRAPH_FAILED);
    OPS_ERR_IF(PopulateTiling() != GRAPH_SUCCESS,
               OPS_LOG_E(context_, "populate tiling failed"),
               return GRAPH_FAILED);
    OPS_ERR_IF(Finalize() != GRAPH_SUCCESS,
               OPS_LOG_E(context_, "finalize tiling failed"),
               return GRAPH_FAILED);
    context_->SetTilingKey(1);
    return GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepare(gert::TilingParseContext* ctx)
{
    (void)ctx;
    return GRAPH_SUCCESS;
}

static ge::graphStatus TilingEntry(gert::TilingContext* ctx)
{
    OPS_ERR_IF(ctx == nullptr,
               OPS_REPORT_VECTOR_INNER_ERR("ComputeNGramIds", "null tiling context"),
               return GRAPH_FAILED);
    ComputeNGramIdsTiling solver(ctx);
    return solver.DoTiling();
}

IMPL_OP_OPTILING(ComputeNGramIds)
    .Tiling(TilingEntry)
    .TilingParse<ComputeNGramIdsCompileInfo>(TilingPrepare);

}  // namespace optiling
