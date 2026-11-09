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
 * \file update_cur_kv_to_selection_tiling.cpp
 * \brief UpdateCurKvToSelection Tiling 实现
 *        - GetShapeAttrsInfo：shape 契约校验（rank=3 / B 一致 / S-J 维相等 / dim 一致 / PAGE_SIZE=128）
 *        - DoOpTiling：多核切分（numPerCore + tailUnits）+ P 自适应（UB 预算驱动）
 *        - PostTiling：TilingKey=1、workspace=0
 */

#include "update_cur_kv_to_selection_tiling.h"

namespace optiling {

constexpr int32_t CUR_K_NOPE_IDX = 0;
constexpr int32_t CUR_K_ROPE_IDX = 1;
constexpr int32_t SLOTS_IDX = 2;
constexpr int32_t SEL_K_NOPE_IDX = 3;
constexpr int32_t SEL_K_ROPE_IDX = 4;
constexpr int32_t BLOCK_TABLE_IDX = 5;

constexpr uint32_t PAGE_SIZE = 128;                 // 物理页行数（编译期常量）
constexpr uint32_t DATA_ALIGN_32B = 32;             // UB 侧 32B 对齐粒度
constexpr uint32_t ELEMS_PER_32B_BF16 = 16;         // 16 bf16 元素 = 32B
constexpr double UB_BUDGET_RATIO = 0.95;            // UB 预算比例（预留系统区）
constexpr uint64_t IDX_RESERVE_BYTES = 4 * 1024;    // idx 批缓冲 + 余量
constexpr uint32_t PAGE_SIZE_CONSTRAINT = 128;      // selection.shape[1] 契约值

static inline uint32_t AlignUp32(uint32_t bytes)
{
    return (bytes + DATA_ALIGN_32B - 1) / DATA_ALIGN_32B * DATA_ALIGN_32B;
}

ge::graphStatus UpdateCurKvToSelectionTiling::GetPlatformInfo()
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

ge::graphStatus UpdateCurKvToSelectionTiling::GetShapeAttrsInfo()
{
    OPS_ERR_IF(context_ == nullptr, OPS_LOG_E("UpdateCurKvToSelection", "context can not be nullptr."),
        return ge::GRAPH_FAILED);

    // cur_k_nope: [B, S, kNopeDim]
    auto curNopeIn = context_->GetInputShape(CUR_K_NOPE_IDX);
    OPS_ERR_IF(curNopeIn == nullptr, OPS_LOG_E(context_->GetNodeName(), "get cur_k_nope shape nullptr."),
        return ge::GRAPH_FAILED);
    gert::Shape curNopeShape = curNopeIn->GetStorageShape();
    OPS_ERR_IF(curNopeShape.GetDimNum() != 3,
        OPS_LOG_E(context_->GetNodeName(), "cur_k_nope dim:%zu should be 3.", curNopeShape.GetDimNum()),
        return ge::GRAPH_FAILED);
    batch_ = static_cast<uint32_t>(curNopeShape.GetDim(0));
    seq_ = static_cast<uint32_t>(curNopeShape.GetDim(1));
    kNopeDim_ = static_cast<uint32_t>(curNopeShape.GetDim(2));

    // cur_k_rope: [B, S, kRopeDim]
    auto curRopeIn = context_->GetInputShape(CUR_K_ROPE_IDX);
    OPS_ERR_IF(curRopeIn == nullptr, OPS_LOG_E(context_->GetNodeName(), "get cur_k_rope shape nullptr."),
        return ge::GRAPH_FAILED);
    gert::Shape curRopeShape = curRopeIn->GetStorageShape();
    OPS_ERR_IF(curRopeShape.GetDimNum() != 3,
        OPS_LOG_E(context_->GetNodeName(), "cur_k_rope dim:%zu should be 3.", curRopeShape.GetDimNum()),
        return ge::GRAPH_FAILED);
    // B 一致：cur_k_rope batch 维必须与 cur_k_nope 相等
    OPS_ERR_IF(static_cast<uint32_t>(curRopeShape.GetDim(0)) != batch_,
        OPS_LOG_E(context_->GetNodeName(), "cur_k_rope batch:%ld != cur_k_nope batch:%u.",
            curRopeShape.GetDim(0), batch_),
        return ge::GRAPH_FAILED);
    kRopeDim_ = static_cast<uint32_t>(curRopeShape.GetDim(2));

    // cur_kv_write_slots: [B, S, S]（rank=3、B 一致、S-J 维相等）
    auto slotsIn = context_->GetInputShape(SLOTS_IDX);
    OPS_ERR_IF(slotsIn == nullptr, OPS_LOG_E(context_->GetNodeName(), "get cur_kv_write_slots shape nullptr."),
        return ge::GRAPH_FAILED);
    gert::Shape slotsShape = slotsIn->GetStorageShape();
    OPS_ERR_IF(slotsShape.GetDimNum() != 3,
        OPS_LOG_E(context_->GetNodeName(), "cur_kv_write_slots dim:%zu should be 3.", slotsShape.GetDimNum()),
        return ge::GRAPH_FAILED);
    OPS_ERR_IF(static_cast<uint32_t>(slotsShape.GetDim(0)) != batch_,
        OPS_LOG_E(context_->GetNodeName(), "cur_kv_write_slots batch:%ld != cur_k_nope batch:%u.",
            slotsShape.GetDim(0), batch_),
        return ge::GRAPH_FAILED);
    OPS_ERR_IF(static_cast<uint32_t>(slotsShape.GetDim(1)) != seq_ ||
        static_cast<uint32_t>(slotsShape.GetDim(2)) != seq_ ||
        static_cast<uint32_t>(curRopeShape.GetDim(1)) != seq_,
        OPS_LOG_E(context_->GetNodeName(),
            "S-J dim mismatch: slots[%ld,%ld] cur_k_rope[1]=%ld should equal seq:%u.",
            slotsShape.GetDim(1), slotsShape.GetDim(2), curRopeShape.GetDim(1), seq_),
        return ge::GRAPH_FAILED);

    // selection_k_nope: [TOTAL_PAGE_NUM, 128, kNopeDim]
    auto selNopeIn = context_->GetInputShape(SEL_K_NOPE_IDX);
    OPS_ERR_IF(selNopeIn == nullptr, OPS_LOG_E(context_->GetNodeName(), "get selection_k_nope shape nullptr."),
        return ge::GRAPH_FAILED);
    gert::Shape selNopeShape = selNopeIn->GetStorageShape();
    OPS_ERR_IF(selNopeShape.GetDimNum() != 3,
        OPS_LOG_E(context_->GetNodeName(), "selection_k_nope dim:%zu should be 3.", selNopeShape.GetDimNum()),
        return ge::GRAPH_FAILED);
    totalPageNum_ = static_cast<uint32_t>(selNopeShape.GetDim(0));
    OPS_ERR_IF(static_cast<uint32_t>(selNopeShape.GetDim(1)) != PAGE_SIZE_CONSTRAINT,
        OPS_LOG_E(context_->GetNodeName(), "selection_k_nope[1]:%ld should be 128.", selNopeShape.GetDim(1)),
        return ge::GRAPH_FAILED);
    OPS_ERR_IF(static_cast<uint32_t>(selNopeShape.GetDim(2)) != kNopeDim_,
        OPS_LOG_E(context_->GetNodeName(), "selection_k_nope[2]:%ld != cur_k_nope[2]:%u.",
            selNopeShape.GetDim(2), kNopeDim_),
        return ge::GRAPH_FAILED);

    // selection_k_rope: [TOTAL_PAGE_NUM, 128, kRopeDim]
    auto selRopeIn = context_->GetInputShape(SEL_K_ROPE_IDX);
    OPS_ERR_IF(selRopeIn == nullptr, OPS_LOG_E(context_->GetNodeName(), "get selection_k_rope shape nullptr."),
        return ge::GRAPH_FAILED);
    gert::Shape selRopeShape = selRopeIn->GetStorageShape();
    OPS_ERR_IF(selRopeShape.GetDimNum() != 3,
        OPS_LOG_E(context_->GetNodeName(), "selection_k_rope dim:%zu should be 3.", selRopeShape.GetDimNum()),
        return ge::GRAPH_FAILED);
    OPS_ERR_IF(static_cast<uint32_t>(selRopeShape.GetDim(0)) != totalPageNum_,
        OPS_LOG_E(context_->GetNodeName(), "selection_k_rope[0]:%ld != selection_k_nope[0]:%u.",
            selRopeShape.GetDim(0), totalPageNum_),
        return ge::GRAPH_FAILED);
    OPS_ERR_IF(static_cast<uint32_t>(selRopeShape.GetDim(1)) != PAGE_SIZE_CONSTRAINT,
        OPS_LOG_E(context_->GetNodeName(), "selection_k_rope[1]:%ld should be 128.", selRopeShape.GetDim(1)),
        return ge::GRAPH_FAILED);
    OPS_ERR_IF(static_cast<uint32_t>(selRopeShape.GetDim(2)) != kRopeDim_,
        OPS_LOG_E(context_->GetNodeName(), "selection_k_rope[2]:%ld != cur_k_rope[2]:%u.",
            selRopeShape.GetDim(2), kRopeDim_),
        return ge::GRAPH_FAILED);

    // selection_kv_block_table: [B, S, MAX_PAGE_PER_REQ]
    auto blockTableIn = context_->GetInputShape(BLOCK_TABLE_IDX);
    OPS_ERR_IF(blockTableIn == nullptr,
        OPS_LOG_E(context_->GetNodeName(), "get selection_kv_block_table shape nullptr."),
        return ge::GRAPH_FAILED);
    gert::Shape blockTableShape = blockTableIn->GetStorageShape();
    OPS_ERR_IF(blockTableShape.GetDimNum() != 3,
        OPS_LOG_E(context_->GetNodeName(), "selection_kv_block_table dim:%zu should be 3.",
            blockTableShape.GetDimNum()),
        return ge::GRAPH_FAILED);
    OPS_ERR_IF(static_cast<uint32_t>(blockTableShape.GetDim(0)) != batch_,
        OPS_LOG_E(context_->GetNodeName(), "selection_kv_block_table batch:%ld != cur_k_nope batch:%u.",
            blockTableShape.GetDim(0), batch_),
        return ge::GRAPH_FAILED);
    OPS_ERR_IF(static_cast<uint32_t>(blockTableShape.GetDim(1)) != seq_,
        OPS_LOG_E(context_->GetNodeName(), "selection_kv_block_table[1]:%ld != seq:%u.",
            blockTableShape.GetDim(1), seq_),
        return ge::GRAPH_FAILED);
    maxPagePerReq_ = static_cast<uint32_t>(blockTableShape.GetDim(2));

    // totalUnits = B*S*S，uint64 中间量防溢出；超 uint32 报错（现实受内存限制不可达）
    uint64_t totalUnits64 = static_cast<uint64_t>(batch_) * seq_ * seq_;
    OPS_ERR_IF(totalUnits64 > UINT32_MAX,
        OPS_LOG_E(context_->GetNodeName(), "totalUnits:%llu exceeds uint32 range.", totalUnits64),
        return ge::GRAPH_FAILED);
    totalUnits_ = static_cast<uint32_t>(totalUnits64);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus UpdateCurKvToSelectionTiling::DoOpTiling()
{
    uint32_t usedCore = static_cast<uint32_t>(coreNum_);
    if (usedCore > totalUnits_) {
        usedCore = totalUnits_;
    }
    if (usedCore == 0) {
        usedCore = 1;   // 空任务（totalUnits==0）：单核空转返回，kernel count_==0 直接 return
    }
    uint32_t numPerCore = totalUnits_ / usedCore;
    uint32_t tailUnits = totalUnits_ % usedCore;

    // ===== P 自适应（UB 预算驱动）=====
    uint32_t nopeRowPitchElems = (kNopeDim_ + ELEMS_PER_32B_BF16 - 1) / ELEMS_PER_32B_BF16 * ELEMS_PER_32B_BF16;
    uint32_t ropeRowPitchElems = (kRopeDim_ + ELEMS_PER_32B_BF16 - 1) / ELEMS_PER_32B_BF16 * ELEMS_PER_32B_BF16;
    uint32_t pitchNopeB = AlignUp32(kNopeDim_ * sizeof(uint16_t));
    uint32_t pitchRopeB = AlignUp32(kRopeDim_ * sizeof(uint16_t));

    uint64_t ubBudget = static_cast<uint64_t>(static_cast<double>(ubSize_) * UB_BUDGET_RATIO);
    uint64_t perUnitDouble = 2ULL * (pitchNopeB + pitchRopeB);   // 双缓冲下每单元开销
    uint32_t p = 1;
    uint32_t dbFlag = 0;    // 0=双缓冲
    if (ubBudget > IDX_RESERVE_BYTES && perUnitDouble > 0) {
        uint64_t pUb = (ubBudget - IDX_RESERVE_BYTES) / perUnitDouble;
        if (pUb >= 1) {
            p = static_cast<uint32_t>(pUb);
            if (p > seq_) {
                p = seq_;
            }
            if (p == 0) {
                p = 1;   // seq_==0（空任务），P 无意义，取 1
            }
        } else {
            // P_ub < 1：退化为单缓冲 P=1；数据区 + idx 区仍放不下 → GRAPH_FAILED
            uint64_t idxBytes = 8ULL * sizeof(int32_t);   // alignedNel = ceil(1/8)*8 = 8
            OPS_ERR_IF(static_cast<uint64_t>(pitchNopeB + pitchRopeB) + idxBytes > ubBudget,
                OPS_LOG_E(context_->GetNodeName(),
                    "UB capacity insufficient: single row pitch nope:%u rope:%u bytes > budget:%llu.",
                    pitchNopeB, pitchRopeB, ubBudget),
                return ge::GRAPH_FAILED);
            p = 1;
            dbFlag = 1;   // 单缓冲退化
        }
    } else {
        // 极端：UB_BUDGET 小于预留（理论不出现），按单缓冲 P=1 尝试
        dbFlag = 1;
    }

    tilingData_.set_blockNum(usedCore);
    tilingData_.set_totalUnits(totalUnits_);
    tilingData_.set_numPerCore(numPerCore);
    tilingData_.set_tailUnits(tailUnits);
    tilingData_.set_batch(batch_);
    tilingData_.set_seq(seq_);
    tilingData_.set_kNopeDim(kNopeDim_);
    tilingData_.set_kRopeDim(kRopeDim_);
    tilingData_.set_unitsPerBatch(p);
    tilingData_.set_maxPagePerReq(maxPagePerReq_);
    tilingData_.set_totalPageNum(totalPageNum_);
    tilingData_.set_nopeRowPitchElems(nopeRowPitchElems);
    tilingData_.set_ropeRowPitchElems(ropeRowPitchElems);
    tilingData_.set_rowAlignedNope((kNopeDim_ * sizeof(uint16_t) % DATA_ALIGN_32B == 0) ? 1U : 0U);
    tilingData_.set_rowAlignedRope((kRopeDim_ * sizeof(uint16_t) % DATA_ALIGN_32B == 0) ? 1U : 0U);
    tilingData_.set_reserved(dbFlag);

    context_->SetBlockDim(usedCore);
    OPS_LOG_I(context_->GetNodeName(),
        "UpdateCurKvToSelection tiling totalUnits=%u batch=%u seq=%u kNopeDim=%u kRopeDim=%u "
        "usedCore=%u P=%u dbFlag=%u rowAligned=%u/%u",
        totalUnits_, batch_, seq_, kNopeDim_, kRopeDim_, usedCore, p, dbFlag,
        tilingData_.get_rowAlignedNope(), tilingData_.get_rowAlignedRope());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus UpdateCurKvToSelectionTiling::PostTiling()
{
    context_->SetTilingKey(1);
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    OPS_ERR_IF(workspaces == nullptr, OPS_LOG_E(context_->GetNodeName(), "get workspaces nullptr."),
        return ge::GRAPH_FAILED);
    workspaces[0] = 0;   // 本算子无需 workspace
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus UpdateCurKvToSelectionTiling::RunTiling()
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

ge::graphStatus Tiling4UpdateCurKvToSelection(gert::TilingContext* context)
{
    OPS_LOG_I(context->GetNodeName(), "Tiling4UpdateCurKvToSelection running.");
    UpdateCurKvToSelectionTiling tiling(context);
    return tiling.RunTiling();
}

ge::graphStatus TilingPrepare4UpdateCurKvToSelection(gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(UpdateCurKvToSelection)
    .Tiling(Tiling4UpdateCurKvToSelection)
    .TilingParse<UpdateCurKvToSelectionCompileInfo>(TilingPrepare4UpdateCurKvToSelection);

} // namespace optiling
