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
 * \file update_cache_tiling.cpp
 * \brief
 */

#include "update_cache_tiling.h"

namespace optiling {

constexpr int32_t SEL_TOPK_IDX = 0;
constexpr int32_t SEL_KV_CACHE_IDX = 2;
constexpr int32_t CACHE_K_ROPE_IDX = 3;
constexpr int32_t CACHE_KV_CACHE_IDX = 4;
constexpr int32_t CACHE_STATUS_IDX = 5;
constexpr int32_t SEL_KV_BLOCK_TABLE_IDX = 7;
constexpr int32_t WRITE_SLOT_IDX = 8;
constexpr int32_t STEP_IDX = 9;
constexpr uint32_t ROWS_PER_BATCH = 4;   // 阶段 II 批量粒度 P

ge::graphStatus UpdateCacheTiling::GetPlatformInfo()
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
    OPS_ERR_IF(ubSize_ < 185 * 1024, OPS_LOG_E(context_->GetNodeName(), "UpdateCache requires at least 185 KiB UB."),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus UpdateCacheTiling::GetShapeAttrsInfo()
{
    OPS_ERR_IF(context_ == nullptr, OPS_LOG_E("UpdateCache", "context can not be nullptr."),
        return ge::GRAPH_FAILED);

    // selection_topk_indices: [B, S, TOPK]
    auto selTopkIn = context_->GetInputShape(SEL_TOPK_IDX);
    OPS_ERR_IF(selTopkIn == nullptr, OPS_LOG_E(context_->GetNodeName(), "get selTopkIn nullptr."),
        return ge::GRAPH_FAILED);
    gert::Shape selTopkShape = selTopkIn->GetStorageShape();
    OPS_ERR_IF(selTopkShape.GetDimNum() != 3,
        OPS_LOG_E(context_->GetNodeName(), "selection_topk_indices dim:%zu should be 3.", selTopkShape.GetDimNum()),
        return ge::GRAPH_FAILED);
    B_ = static_cast<uint32_t>(selTopkShape.GetDim(0));
    S_ = static_cast<uint32_t>(selTopkShape.GetDim(1));
    topk_ = static_cast<uint32_t>(selTopkShape.GetDim(2));

    // cache_k_rope: [B, CACHE, kRopeDim]
    auto cacheKRopeIn = context_->GetInputShape(CACHE_K_ROPE_IDX);
    OPS_ERR_IF(cacheKRopeIn == nullptr, OPS_LOG_E(context_->GetNodeName(), "get cacheKRopeIn nullptr."),
        return ge::GRAPH_FAILED);
    gert::Shape cacheKRopeShape = cacheKRopeIn->GetStorageShape();
    OPS_ERR_IF(cacheKRopeShape.GetDimNum() != 3,
        OPS_LOG_E(context_->GetNodeName(), "cache_k_rope dim:%zu should be 3.", cacheKRopeShape.GetDimNum()),
        return ge::GRAPH_FAILED);
    cache_ = static_cast<uint32_t>(cacheKRopeShape.GetDim(1));
    kRopeDim_ = static_cast<uint32_t>(cacheKRopeShape.GetDim(2));

    // cache_kv_cache: [B, CACHE, kvDim]
    auto cacheKvIn = context_->GetInputShape(CACHE_KV_CACHE_IDX);
    OPS_ERR_IF(cacheKvIn == nullptr, OPS_LOG_E(context_->GetNodeName(), "get cacheKvIn nullptr."),
        return ge::GRAPH_FAILED);
    gert::Shape cacheKvShape = cacheKvIn->GetStorageShape();
    OPS_ERR_IF(cacheKvShape.GetDimNum() != 3,
        OPS_LOG_E(context_->GetNodeName(), "cache_kv_cache dim:%zu should be 3.", cacheKvShape.GetDimNum()),
        return ge::GRAPH_FAILED);
    kvDim_ = static_cast<uint32_t>(cacheKvShape.GetDim(2));

    // Fixed UB partitions and SortSeg require power-of-two TopK and the split-KV cache shape.
    OPS_ERR_IF(topk_ < 64 || topk_ > 2048 || (topk_ & (topk_ - 1)) != 0 ||
               cache_ != 4096 || kvDim_ != 512 || kRopeDim_ != 64,
        OPS_LOG_E(context_->GetNodeName(), "UpdateCache requires TopK=64..2048 (power of two), cache=4096, KV=512, KR=64."),
        return ge::GRAPH_FAILED);

    // selection_kv_cache（NEW 源，分页）: [S_BLOCK_NUM, selKvBlockSize, kvDim] → 取块长
    auto selKvIn = context_->GetInputShape(SEL_KV_CACHE_IDX);
    OPS_ERR_IF(selKvIn == nullptr, OPS_LOG_E(context_->GetNodeName(), "get selKvIn nullptr."),
        return ge::GRAPH_FAILED);
    gert::Shape selKvShape = selKvIn->GetStorageShape();
    OPS_ERR_IF(selKvShape.GetDimNum() != 3,
        OPS_LOG_E(context_->GetNodeName(), "selection_kv_cache dim:%zu should be 3 [S_BLOCK,BS,dim].",
            selKvShape.GetDimNum()),
        return ge::GRAPH_FAILED);
    selKvBlockSize_ = static_cast<uint32_t>(selKvShape.GetDim(1));

    // selection_kv_block_table [B*S, selMaxBlockNum] → 取列数
    auto selBtIn = context_->GetInputShape(SEL_KV_BLOCK_TABLE_IDX);
    OPS_ERR_IF(selBtIn == nullptr, OPS_LOG_E(context_->GetNodeName(), "get selection_kv_block_table nullptr."),
        return ge::GRAPH_FAILED);
    selMaxBlockNum_ = static_cast<uint32_t>(selBtIn->GetStorageShape().GetDim(1));

    // dtype size (fp16/bf16 -> 2)
    auto cacheKRopeDesc = context_->GetInputDesc(CACHE_K_ROPE_IDX);
    OPS_ERR_IF(cacheKRopeDesc == nullptr, OPS_LOG_E(context_->GetNodeName(), "get cacheKRopeDesc nullptr."),
        return ge::GRAPH_FAILED);
    dtypeSize_ = ge::GetSizeByDataType(cacheKRopeDesc->GetDataType());
    OPS_ERR_IF(dtypeSize_ <= 0, OPS_LOG_E(context_->GetNodeName(), "invalid dtype size."),
        return ge::GRAPH_FAILED);

    // step 输入: [B] int32
    auto stepIn = context_->GetInputShape(STEP_IDX);
    OPS_ERR_IF(stepIn == nullptr, OPS_LOG_E(context_->GetNodeName(), "get step input nullptr."),
        return ge::GRAPH_FAILED);
    gert::Shape stepShape = stepIn->GetStorageShape();
    OPS_ERR_IF(stepShape.GetDimNum() != 1,
        OPS_LOG_E(context_->GetNodeName(), "step dim:%zu should be 1.", stepShape.GetDimNum()),
        return ge::GRAPH_FAILED);
    OPS_ERR_IF(static_cast<uint32_t>(stepShape.GetDim(0)) != B_,
        OPS_LOG_E(context_->GetNodeName(), "step shape[0]:%lld should equal B:%u.",
            stepShape.GetDim(0), B_),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus UpdateCacheTiling::DoOpTiling()
{
    // blockNum = min(可用核数, B)：kernel 以 for(b=blkIdx; b<B; b+=blockNum) 网格循环覆盖全部 batch。
    uint32_t blockNum = B_;
    if (blockNum == 0) {
        blockNum = 1;
    }
    tilingData_.set_blockNum(blockNum);
    tilingData_.set_B(B_);
    tilingData_.set_S(S_);
    tilingData_.set_topk(topk_);
    tilingData_.set_cache(cache_);
    tilingData_.set_kRopeDim(kRopeDim_);
    tilingData_.set_kvDim(kvDim_);
    tilingData_.set_rowsPerBatch(ROWS_PER_BATCH);
    tilingData_.set_selKvBlockSize(selKvBlockSize_);
    tilingData_.set_selMaxBlockNum(selMaxBlockNum_);
    tilingData_.set_reserved(0);

    context_->SetBlockDim(blockNum);
    OPS_LOG_I(context_->GetNodeName(),
        "UpdateCache tiling B=%u S=%u topk=%u cache=%u kRopeDim=%u kvDim=%u blockNum=%u "
        "(step is per-batch tensor)",
        B_, S_, topk_, cache_, kRopeDim_, kvDim_, blockNum);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus UpdateCacheTiling::PostTiling()
{
    context_->SetTilingKey(1);
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    OPS_ERR_IF(workspaces == nullptr, OPS_LOG_E(context_->GetNodeName(), "get workspaces nullptr."),
        return ge::GRAPH_FAILED);
    // 多 seq 折叠（方案B）：kernel 内 for(s) 只对 status/age + resolved 溯源码（均 [B,CACHE] int32）
    //   做 GM ping-pong，KV 只在最后 PhaseRowMoveFinal 搬一趟。每 region = [status|age|resolved] 三段
    //   int32，无 KV 段（相比方案A 省掉 ~37MB 的两份全池 KV）。
    // 淘汰 float scratch 紧跟两 region 之后：per-batch 3 段（statusOut/srcOutF/ageOut），段上界 E_UNION。
    constexpr size_t UC_E_UNION = 8192;   // 与 kernel 侧 E_UNION 一致（S*TOPK 上界）
    size_t statusSeg = static_cast<size_t>(B_) * cache_ * sizeof(int32_t);
    size_t regionBytes = 3 * statusSeg;   // status + age + resolved
    size_t wsBytes = static_cast<size_t>(B_) * 3 * UC_E_UNION * sizeof(float);
    // 保留 32MB 系统余量（ascendc 内部 + 对齐），这是 evict 路径正确性所必需。
    size_t stagedKvBytes = static_cast<size_t>(B_) * cache_ * (kvDim_ + kRopeDim_) * dtypeSize_;
    workspaces[0] = 2 * regionBytes + wsBytes + stagedKvBytes + 32 * 1024 * 1024;
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus UpdateCacheTiling::RunTiling()
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

ge::graphStatus Tiling4UpdateCache(gert::TilingContext* context)
{
    OPS_LOG_I(context->GetNodeName(), "TilingForUpdateCache running.");
    UpdateCacheTiling tiling(context);
    return tiling.RunTiling();
}

ge::graphStatus TilingPrepare4UpdateCache(gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(UpdateCache)
    .Tiling(Tiling4UpdateCache)
    .TilingParse<UpdateCacheCompileInfo>(TilingPrepare4UpdateCache);

} // namespace optiling
