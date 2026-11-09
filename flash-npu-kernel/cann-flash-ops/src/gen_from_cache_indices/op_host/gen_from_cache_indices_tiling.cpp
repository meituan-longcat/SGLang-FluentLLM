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
 * \file gen_from_cache_indices_tiling.cpp
 * \brief
 */

#include "gen_from_cache_indices_tiling.h"

namespace optiling {

constexpr int32_t TOPK_IDX = 0;
constexpr int32_t CACHE_IDX = 1;
constexpr int32_t KV_ACTSEQ_IDX = 2;
constexpr int32_t Q_ACTSEQ_IDX = 3;
constexpr int32_t FULL_BT_IDX = 4;   // full_kv_block_table（OPTIONAL）
constexpr size_t DIM3 = 3;
constexpr size_t DIM2 = 2;
constexpr uint32_t ALIGN_UNIT = 64;  // Compare/Select 等 mask 类 API：count*sizeof(float) 须 256B 对齐 → topk/cache 须 64 倍数
constexpr int64_t DEFAULT_WORKSPACE_SIZE = 16 * 1024 * 1024;
// 当前 token 编码基址（须与 gen_from_cache_indices_vec.h 的 GFCI_CUR_TOKEN_BASE 一致）。
// hit 编码 = b*cache_+slot+1，最大值 = B*cache_；须 < BASE，否则 hit 值与 current(ws+BASE) 撞值。
constexpr int32_t GEN_CUR_TOKEN_BASE = 1 << 20;

ge::graphStatus GenFromCacheIndicesTiling::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    OPS_ERR_IF(platformInfo == nullptr, OPS_LOG_E(context_->GetNodeName(), "get platformInfo nullptr."),
        return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    coreNum_ = ascendcPlatform.GetCoreNumAiv();
    OPS_ERR_IF(coreNum_ <= 0, OPS_LOG_E(context_->GetNodeName(), "coreNum must be greater than 0."),
        return ge::GRAPH_FAILED);

    uint64_t ubSizePlatForm;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    ubSize_ = static_cast<int64_t>(ubSizePlatForm);
    OPS_ERR_IF(ubSize_ <= 0, OPS_LOG_E(context_->GetNodeName(), "ubSize must be greater than 0."),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GenFromCacheIndicesTiling::GetShapeAttrsInfo()
{
    OPS_ERR_IF(context_ == nullptr, OPS_LOG_E("GenFromCacheIndices", "context can not be nullptr."),
        return ge::GRAPH_FAILED);

    // selection_topk_indices: [B, S, TOPK]
    auto topkIn = context_->GetInputShape(TOPK_IDX);
    OPS_ERR_IF(topkIn == nullptr, OPS_LOG_E(context_->GetNodeName(), "get topkIn nullptr."),
        return ge::GRAPH_FAILED);
    gert::Shape topkShape = topkIn->GetStorageShape();
    OPS_ERR_IF(topkShape.GetDimNum() != DIM3,
        OPS_LOG_E(context_->GetNodeName(), "selection_topk_indices dim:%zu should be 3.", topkShape.GetDimNum()),
        return ge::GRAPH_FAILED);
    B_ = static_cast<uint32_t>(topkShape.GetDim(0));
    S_ = static_cast<uint32_t>(topkShape.GetDim(1));
    topk_ = static_cast<uint32_t>(topkShape.GetDim(DIM2));

    // cache_status: [batch, CACHE]
    auto cacheIn = context_->GetInputShape(CACHE_IDX);
    OPS_ERR_IF(cacheIn == nullptr, OPS_LOG_E(context_->GetNodeName(), "get cacheIn nullptr."),
        return ge::GRAPH_FAILED);
    gert::Shape cacheShape = cacheIn->GetStorageShape();
    OPS_ERR_IF(cacheShape.GetDimNum() != DIM2,
        OPS_LOG_E(context_->GetNodeName(), "cache_status dim:%zu should be 2.", cacheShape.GetDimNum()),
        return ge::GRAPH_FAILED);
    OPS_ERR_IF(static_cast<uint32_t>(cacheShape.GetDim(0)) != B_,
        OPS_LOG_E(context_->GetNodeName(), "cache_status batch:%ld != topk batch:%u.", cacheShape.GetDim(0), B_),
        return ge::GRAPH_FAILED);
    cache_ = static_cast<uint32_t>(cacheShape.GetDim(1));

    // full_kv_actual_seq / full_q_actual_seq: [batch]
    auto kvSeqIn = context_->GetInputShape(KV_ACTSEQ_IDX);
    OPS_ERR_IF(kvSeqIn == nullptr, OPS_LOG_E(context_->GetNodeName(), "get full_kv_actual_seq nullptr."),
        return ge::GRAPH_FAILED);
    OPS_ERR_IF(static_cast<uint32_t>(kvSeqIn->GetStorageShape().GetDim(0)) != B_,
        OPS_LOG_E(context_->GetNodeName(), "full_kv_actual_seq dim0 should be batch:%u.", B_),
        return ge::GRAPH_FAILED);
    auto qSeqIn = context_->GetInputShape(Q_ACTSEQ_IDX);
    OPS_ERR_IF(qSeqIn == nullptr, OPS_LOG_E(context_->GetNodeName(), "get full_q_actual_seq nullptr."),
        return ge::GRAPH_FAILED);
    OPS_ERR_IF(static_cast<uint32_t>(qSeqIn->GetStorageShape().GetDim(0)) != B_,
        OPS_LOG_E(context_->GetNodeName(), "full_q_actual_seq dim0 should be batch:%u.", B_),
        return ge::GRAPH_FAILED);

    // attr selection_topk_block_size
    auto attrs = context_->GetAttrs();
    OPS_ERR_IF(attrs == nullptr, OPS_LOG_E(context_->GetNodeName(), "get attrs nullptr."),
        return ge::GRAPH_FAILED);
    const int64_t* bsPtr = attrs->GetAttrPointer<int64_t>(0);
    blockSize_ = (bsPtr == nullptr || *bsPtr <= 0) ? 1U : static_cast<uint32_t>(*bsPtr);

    // attr enable_topk_indices_opt（Int 类型，原值直传不截断为 bool）：
    //   0=关闭；1=opt（topkIndices 有效值在前，write_slot 走 CreateVecIndex 捷径）；
    //   2=unifiedSrc（额外输出 unified_slots_in_lru_or_cache，需 full_kv_block_table 输入）。
    const int64_t* optPtr = attrs->GetAttrPointer<int64_t>(1);
    enableOpt_ = (optPtr != nullptr) ? static_cast<uint32_t>(*optPtr) : 0U;
    OPS_ERR_IF(enableOpt_ != 0U && blockSize_ != 1,
        OPS_LOG_E(context_->GetNodeName(),
            "enable_topk_indices_opt requires selection_topk_block_size == 1, got %u.", blockSize_),
        return ge::GRAPH_FAILED);

    // attr full_kv_block_size（opt==2 时用于 miss 页/poff 计算）
    const int64_t* fbsPtr = attrs->GetAttrPointer<int64_t>(2);
    fullKvBlockSize_ = (fbsPtr != nullptr && *fbsPtr > 0) ? static_cast<uint32_t>(*fbsPtr) : 128U;

    // full_kv_block_table [B, fullMaxBlockNum]（OPTIONAL，opt==2 时必需）
    if (enableOpt_ == 2U) {
        auto fullBtIn = context_->GetInputShape(FULL_BT_IDX);
        OPS_ERR_IF(fullBtIn == nullptr,
            OPS_LOG_E(context_->GetNodeName(), "enable_topk_indices_opt==2 requires full_kv_block_table input."),
            return ge::GRAPH_FAILED);
        gert::Shape fullBtShape = fullBtIn->GetStorageShape();
        OPS_ERR_IF(fullBtShape.GetDimNum() != DIM2,
            OPS_LOG_E(context_->GetNodeName(), "full_kv_block_table dim:%zu should be 2.", fullBtShape.GetDimNum()),
            return ge::GRAPH_FAILED);
        OPS_ERR_IF(static_cast<uint32_t>(fullBtShape.GetDim(0)) != B_,
            OPS_LOG_E(context_->GetNodeName(), "full_kv_block_table batch:%ld != topk batch:%u.",
                fullBtShape.GetDim(0), B_),
            return ge::GRAPH_FAILED);
        fullMaxBlockNum_ = static_cast<uint32_t>(fullBtShape.GetDim(1));
        OPS_ERR_IF(fullKvBlockSize_ == 0,
            OPS_LOG_E(context_->GetNodeName(), "full_kv_block_size must be positive."),
            return ge::GRAPH_FAILED);
    }

    OPS_ERR_IF(topk_ == 0 || topk_ > 2048 || topk_ % ALIGN_UNIT != 0,
        OPS_LOG_E(context_->GetNodeName(), "topk:%u must be a multiple of 64 in [64, 2048].", topk_),
        return ge::GRAPH_FAILED);
    OPS_ERR_IF(cache_ == 0 || cache_ > 4096,
        OPS_LOG_E(context_->GetNodeName(), "cache:%u must be in [1, 4096].", cache_),
        return ge::GRAPH_FAILED);
    if (enableOpt_) {
        OPS_ERR_IF((cache_ % ALIGN_UNIT != 0),
            OPS_LOG_E(context_->GetNodeName(), "cache:%u should be multiple of 64 (opt Compare 256B alignment).", cache_),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GenFromCacheIndicesTiling::DoOpTiling()
{
    // 一核处理一个 (b,s)（headnum==1）：grid = B*S，便于 actual_seq 单核累加、无跨核归约。
    uint32_t blockNum = B_ * S_;
    if (blockNum == 0) {
        blockNum = 1;
    }

    tilingData_.set_blockNum(blockNum);
    tilingData_.set_B(B_);
    tilingData_.set_S(S_);
    tilingData_.set_topk(topk_);
    tilingData_.set_cache(cache_);
    tilingData_.set_blockSize(blockSize_);
    tilingData_.set_enableTopkIndicesOpt(enableOpt_);
    tilingData_.set_fullKvBlockSize(fullKvBlockSize_);
    tilingData_.set_fullMaxBlockNum(fullMaxBlockNum_);
    tilingData_.set_reserved1(0);

    context_->SetBlockDim(blockNum);
    // current-token 编码基址安全断言（仅 opt==2 生效）：hit 最大值 B*cache_ 必须 < BASE，
    // 否则 hit 值与 current(ws+BASE) 撞值，gather 会把 hit 误判为 current。见 gen_from_cache_indices_vec.h。
    if (enableOpt_ == 2U) {
        OPS_ERR_IF(static_cast<uint64_t>(B_) * cache_ >= static_cast<uint64_t>(GEN_CUR_TOKEN_BASE),
            OPS_LOG_E(context_->GetNodeName(),
                "enable_topk_indices_opt==2 requires B*cache < %d (CUR_TOKEN_BASE), got B=%u cache=%u.",
                GEN_CUR_TOKEN_BASE, B_, cache_),
            return ge::GRAPH_FAILED);
        // 与 kernel 的 E_BT=2*4096+2048 一致：复用已用完的 cache/zero/key 区。
        // 1M 上下文加 padding 需要 8194 列；超过缓冲上限仍报错，避免静默钳位。
        OPS_ERR_IF(fullMaxBlockNum_ > 10240U,
            OPS_LOG_E(context_->GetNodeName(),
                "enable_topk_indices_opt==2 requires full_kv_block_table columns (fullMaxBlockNum) <= 10240, got %u.",
                fullMaxBlockNum_),
            return ge::GRAPH_FAILED);
    }
    OPS_LOG_I(context_->GetNodeName(),
        "GenFromCacheIndices tiling B=%u S=%u topk=%u cache=%u blockSize=%u enableOpt=%u "
        "fullKvBlockSize=%u fullMaxBlockNum=%u blockNum=%u",
        B_, S_, topk_, cache_, blockSize_, enableOpt_, fullKvBlockSize_, fullMaxBlockNum_, blockNum);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GenFromCacheIndicesTiling::PostTiling()
{
    context_->SetTilingKey(1);
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    OPS_ERR_IF(workspaces == nullptr, OPS_LOG_E(context_->GetNodeName(), "get workspaces nullptr."),
        return ge::GRAPH_FAILED);
    workspaces[0] = static_cast<size_t>(DEFAULT_WORKSPACE_SIZE);
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GenFromCacheIndicesTiling::RunTiling()
{
    ge::graphStatus ret = GetShapeAttrsInfo();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    ret = GetPlatformInfo();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    ret = DoOpTiling();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    return PostTiling();
}

ge::graphStatus Tiling4GenFromCacheIndices(gert::TilingContext* context)
{
    OPS_LOG_I(context->GetNodeName(), "TilingForGenFromCacheIndices running.");
    GenFromCacheIndicesTiling tiling(context);
    return tiling.RunTiling();
}

ge::graphStatus TilingPrepare4GenFromCacheIndices(gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(GenFromCacheIndices)
    .Tiling(Tiling4GenFromCacheIndices)
    .TilingParse<GenFromCacheIndicesCompileInfo>(TilingPrepare4GenFromCacheIndices);

} // namespace optiling
