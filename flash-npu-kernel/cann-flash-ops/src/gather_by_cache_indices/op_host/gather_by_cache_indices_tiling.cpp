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
 * \file gather_by_cache_indices_tiling.cpp
 * \brief
 */

#include "gather_by_cache_indices_tiling.h"

namespace optiling {

constexpr int32_t CACHE_K_ROPE_IDX = 0;
constexpr int32_t CACHE_KV_CACHE_IDX = 1;
constexpr int32_t FROM_CACHE_IDX = 2;
constexpr int32_t FULL_KV_CACHE_IDX = 5;
constexpr int32_t FULL_KV_BLOCK_TABLE_IDX = 6;
constexpr int32_t SEL_KV_BLOCK_TABLE_IDX = 7;
constexpr uint32_t UNITS_PER_BATCH = 32;   // 批量粒度 P（一次处理的 gather 单元数）
constexpr int64_t DEFAULT_WORKSPACE_SIZE = 16 * 1024 * 1024;
// 当前 token 编码基址（须与 gather_by_cache_indices_impl.h 的 GBCI_CUR_TOKEN_BASE 一致）。
// gen 编码 hit=b*cache_+slot+1，最大值 = batch*cacheRows；须 < BASE，否则与 current(ws+BASE) 撞值。
constexpr int32_t GBCI_CUR_TOKEN_BASE_HOST = 1 << 20;

template <typename U>
inline U GbciCeilDiv(U a, U b)
{
    return (a + b - 1) / b;
}

ge::graphStatus GatherByCacheIndicesTiling::GetPlatformInfo()
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

ge::graphStatus GatherByCacheIndicesTiling::GetShapeAttrsInfo()
{
    OPS_ERR_IF(context_ == nullptr, OPS_LOG_E("GatherByCacheIndices", "context can not be nullptr."),
        return ge::GRAPH_FAILED);

    // cache_k_rope: [batch, cacheRows, kRopeDim]
    auto cacheKRopeIn = context_->GetInputShape(CACHE_K_ROPE_IDX);
    OPS_ERR_IF(cacheKRopeIn == nullptr, OPS_LOG_E(context_->GetNodeName(), "get cacheKRopeIn nullptr."),
        return ge::GRAPH_FAILED);
    gert::Shape cacheKRopeShape = cacheKRopeIn->GetStorageShape();
    OPS_ERR_IF(cacheKRopeShape.GetDimNum() != 3,
        OPS_LOG_E(context_->GetNodeName(), "cache_k_rope dim:%zu should be 3.", cacheKRopeShape.GetDimNum()),
        return ge::GRAPH_FAILED);
    batch_ = static_cast<uint32_t>(cacheKRopeShape.GetDim(0));
    cacheRows_ = static_cast<uint32_t>(cacheKRopeShape.GetDim(1));
    kRopeDim_ = static_cast<uint32_t>(cacheKRopeShape.GetDim(2));
    // 注意：batch_ 随后会被 from_cache_indices dim0 覆盖——cache 池可能按 max_batch_size 预分配，
    // 实际参与计算的 batch 以 from_cache_indices 为准（unit→(b,s,k) 分解、unitsPerRealBatch 均按此）。

    // cache_kv_cache: [batch, cacheRows, kvDim]
    auto cacheKvIn = context_->GetInputShape(CACHE_KV_CACHE_IDX);
    OPS_ERR_IF(cacheKvIn == nullptr, OPS_LOG_E(context_->GetNodeName(), "get cacheKvIn nullptr."),
        return ge::GRAPH_FAILED);
    gert::Shape cacheKvShape = cacheKvIn->GetStorageShape();
    OPS_ERR_IF(cacheKvShape.GetDimNum() != 3,
        OPS_LOG_E(context_->GetNodeName(), "cache_kv_cache dim:%zu should be 3.", cacheKvShape.GetDimNum()),
        return ge::GRAPH_FAILED);
    kvDim_ = static_cast<uint32_t>(cacheKvShape.GetDim(2));

    // from_cache_indices: 分页版必须 3D [B, S, TOPK]（需单独拿到 S/TOPK 做 unit→(b,s,k) 分解）
    auto fromIn = context_->GetInputShape(FROM_CACHE_IDX);
    OPS_ERR_IF(fromIn == nullptr, OPS_LOG_E(context_->GetNodeName(), "get fromIn nullptr."),
        return ge::GRAPH_FAILED);
    gert::Shape fromShape = fromIn->GetStorageShape();
    OPS_ERR_IF(fromShape.GetDimNum() != 3,
        OPS_LOG_E(context_->GetNodeName(), "from_cache_indices dim:%zu should be 3 [B,S,TOPK].", fromShape.GetDimNum()),
        return ge::GRAPH_FAILED);
    seq_  = static_cast<uint32_t>(fromShape.GetDim(1));
    topk_ = static_cast<uint32_t>(fromShape.GetDim(2));
    totalUnits_ = static_cast<uint32_t>(fromShape.GetDim(0) * fromShape.GetDim(1) * fromShape.GetDim(2));
    batch_ = static_cast<uint32_t>(fromShape.GetDim(0));   // 实际 batch 以 from 为准，见上注释
    OPS_ERR_IF(batch_ == 0 || totalUnits_ == 0 || (totalUnits_ % batch_ != 0),
        OPS_LOG_E(context_->GetNodeName(), "totalUnits:%u must be positive multiple of batch:%u.", totalUnits_, batch_),
        return ge::GRAPH_FAILED);
    // 910b 向量位掩码算子（Compare/And/Or/Select）要求 count*sizeof(T)=256B 对齐且 LocalTensor 起始
    // 32B 对齐：topk 必须 128 整数倍（int32 count=128→512B 整块，位掩码 16B 亦对齐；不足 128 时
    // kernel padding 的 Duplicate 起址 (alignedNel_+topk_)*4 非 32B 对齐会触发 VEC 对齐异常）。
    OPS_ERR_IF(topk_ == 0 || (topk_ % 128U) != 0,
        OPS_LOG_E(context_->GetNodeName(), "topk:%u must be a positive multiple of 128 (910b vector bit-mask alignment).", topk_),
        return ge::GRAPH_FAILED);

    // full_kv_cache: 3D [F_BLOCK_NUM, fullKvBlockSize, kvDim]（miss 源，分页物理块）
    auto fullKvIn = context_->GetInputShape(FULL_KV_CACHE_IDX);
    OPS_ERR_IF(fullKvIn == nullptr, OPS_LOG_E(context_->GetNodeName(), "get fullKvIn nullptr."),
        return ge::GRAPH_FAILED);
    gert::Shape fullKvShape = fullKvIn->GetStorageShape();
    OPS_ERR_IF(fullKvShape.GetDimNum() != 3,
        OPS_LOG_E(context_->GetNodeName(), "full_kv_cache dim:%zu should be 3 [F_BLOCK,BS,dim].", fullKvShape.GetDimNum()),
        return ge::GRAPH_FAILED);
    fullKvBlockNum_  = static_cast<uint32_t>(fullKvShape.GetDim(0));
    fullKvBlockSize_ = static_cast<uint32_t>(fullKvShape.GetDim(1));

    // 两级页表列数
    auto fullBtIn = context_->GetInputShape(FULL_KV_BLOCK_TABLE_IDX);
    OPS_ERR_IF(fullBtIn == nullptr, OPS_LOG_E(context_->GetNodeName(), "get full_kv_block_table nullptr."),
        return ge::GRAPH_FAILED);
    fullMaxBlockNum_ = static_cast<uint32_t>(fullBtIn->GetStorageShape().GetDim(1));
    auto selBtIn = context_->GetInputShape(SEL_KV_BLOCK_TABLE_IDX);
    OPS_ERR_IF(selBtIn == nullptr, OPS_LOG_E(context_->GetNodeName(), "get selection_kv_block_table nullptr."),
        return ge::GRAPH_FAILED);
    selMaxBlockNum_ = static_cast<uint32_t>(selBtIn->GetStorageShape().GetDim(1));

    // attrs：selection_topk_block_size / selection_kv_block_num / selection_kv_block_size
    auto attrs = context_->GetAttrs();
    OPS_ERR_IF(attrs == nullptr, OPS_LOG_E(context_->GetNodeName(), "get attrs nullptr."),
        return ge::GRAPH_FAILED);
    const int64_t* pBlk = attrs->GetAttrPointer<int64_t>(0);
    const int64_t* pSel = attrs->GetAttrPointer<int64_t>(2);
    const int64_t* pEn = attrs->GetAttrPointer<int64_t>(3);
    selTopKBlockSize_ = (pBlk != nullptr && *pBlk > 0) ? static_cast<uint32_t>(*pBlk) : 1;
    selKvBlockSize_   = (pSel != nullptr && *pSel > 0) ? static_cast<uint32_t>(*pSel) : 128;
    enableSlotsContiguous_ = (pEn != nullptr) ? static_cast<uint32_t>(*pEn) : 0U;
    // 本用例约束：selTopKBlockSize==1（写出段循环靠"连续 k→连续 off"成立）
    OPS_ERR_IF(selTopKBlockSize_ != 1,
        OPS_LOG_E(context_->GetNodeName(), "paged path only supports selection_topk_block_size==1, got %u.",
            selTopKBlockSize_),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GatherByCacheIndicesTiling::DoOpTiling()
{
    // Round-robin requests, keeping every query of a request on one core.
    // This balances an active batch prefix without splitting a row's writer
    // or losing reuse of the first completed query within the request.
    uint32_t totalRows = totalUnits_ / topk_;
    uint32_t usedCore = static_cast<uint32_t>(coreNum_);
    if (usedCore > batch_) {
        usedCore = batch_;
    }
    if (usedCore == 0) {
        usedCore = 1;
    }
    uint32_t rowsPerCore = GbciCeilDiv<uint32_t>(totalRows, usedCore);
    OPS_LOG_I(context_->GetNodeName(),
        "GatherByCacheIndices tiling totalUnits=%u rows=%u batch=%u cacheRows=%u kRopeDim=%u kvDim=%u "
        "usedCore=%u rowsPerCore=%u",
        totalUnits_, totalRows, batch_, cacheRows_, kRopeDim_, kvDim_, usedCore, rowsPerCore);
    uint32_t unitsPerRealBatch = totalUnits_ / batch_;   // = S*Topk

    tilingData_.set_blockNum(usedCore);
    tilingData_.set_totalUnits(totalUnits_);
    tilingData_.set_numPerCore(rowsPerCore * topk_);
    tilingData_.set_tailUnits(0);
    tilingData_.set_kRopeDim(kRopeDim_);
    tilingData_.set_kvDim(kvDim_);
    tilingData_.set_unitsPerBatch(UNITS_PER_BATCH);
    tilingData_.set_cacheRows(cacheRows_);
    tilingData_.set_unitsPerRealBatch(unitsPerRealBatch);
    tilingData_.set_seq(seq_);
    tilingData_.set_topk(topk_);
    tilingData_.set_selTopKBlockSize(selTopKBlockSize_);
    tilingData_.set_fullKvBlockSize(fullKvBlockSize_);
    tilingData_.set_fullMaxBlockNum(fullMaxBlockNum_);
    tilingData_.set_fullKvBlockNum(fullKvBlockNum_);
    tilingData_.set_selKvBlockSize(selKvBlockSize_);
    tilingData_.set_selMaxBlockNum(selMaxBlockNum_);
    tilingData_.set_enableSlotsContiguous(enableSlotsContiguous_);
    tilingData_.set_reserved(0);

    context_->SetBlockDim(usedCore);
    // current-token 编码基址安全断言（仅 mode==2 生效）：gen 编码 hit=b*cache_+slot+1，
    // 最大值 = batch*cacheRows 须 < BASE，否则 hit 与 current(ws+BASE) 撞值、pass1 误判。
    // 见 gather_by_cache_indices_impl.h GBCI_COPY_UNIT。
    if (enableSlotsContiguous_ == 2U) {
        OPS_ERR_IF(static_cast<uint64_t>(batch_) * cacheRows_ >= static_cast<uint64_t>(GBCI_CUR_TOKEN_BASE_HOST),
            OPS_LOG_E(context_->GetNodeName(),
                "enable_slots_contiguous==2 requires batch*cacheRows < %d (CUR_TOKEN_BASE), got batch=%u cacheRows=%u.",
                GBCI_CUR_TOKEN_BASE_HOST, batch_, cacheRows_),
            return ge::GRAPH_FAILED);
    }
    OPS_LOG_I(context_->GetNodeName(),
        "GatherByCacheIndices tiling totalUnits=%u rows=%u batch=%u cacheRows=%u kRopeDim=%u kvDim=%u "
        "usedCore=%u rowsPerCore=%u",
        totalUnits_, totalRows, batch_, cacheRows_, kRopeDim_, kvDim_, usedCore, rowsPerCore);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GatherByCacheIndicesTiling::PostTiling()
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

ge::graphStatus GatherByCacheIndicesTiling::RunTiling()
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

ge::graphStatus Tiling4GatherByCacheIndices(gert::TilingContext* context)
{
    OPS_LOG_I(context->GetNodeName(), "TilingForGatherByCacheIndices running.");
    GatherByCacheIndicesTiling tiling(context);
    return tiling.RunTiling();
}

ge::graphStatus TilingPrepare4GatherByCacheIndices(gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(GatherByCacheIndices)
    .Tiling(Tiling4GatherByCacheIndices)
    .TilingParse<GatherByCacheIndicesCompileInfo>(TilingPrepare4GatherByCacheIndices);

} // namespace optiling
