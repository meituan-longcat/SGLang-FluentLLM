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
 * \file gather_selection_kv_cache_tiling.cpp
 * \brief
 */

#include "gather_selection_kv_cache_tiling.h"

namespace optiling {

constexpr int32_t SEL_K_ROPE_IDX = 0;
constexpr int32_t SEL_KV_CACHE_IDX = 1;
constexpr int32_t SEL_KV_BLOCK_TABLE_IDX = 2;
constexpr int32_t SEL_KV_BLOCK_STAT_IDX = 3;
constexpr int32_t SEL_TOPK_INDICES_IDX = 4;
constexpr int32_t FULL_K_ROPE_IDX = 5;
constexpr int32_t FULL_KV_CACHE_IDX = 6;
constexpr int32_t FULL_KV_BLOCK_TABLE_IDX = 7;
constexpr int32_t FULL_KV_ACTSEQ_IDX = 8;
constexpr int32_t FULL_Q_ACTSEQ_IDX = 9;

constexpr size_t CONST1 = 1;
constexpr size_t CONST2 = 2;
constexpr size_t CONST3 = 3;
constexpr size_t CONST4 = 4;

constexpr int32_t MAX_Q_SEQ_LEN = 8;
constexpr int32_t MAX_K_ROPE_DIM = 64;
// 512(int8) + 64(fp16) * 2 + 4(fp32) * 4
constexpr int32_t MAX_KV_CACHE_DIM = 656;
constexpr int32_t MAX_TOPK_NUM = 2048;
constexpr int32_t TOPK_SPLIT_NUM = 32;
constexpr int64_t DEFAULT_TOPK_BLOCK_SIZE = 64;
constexpr int64_t DEFAULT_WORKSPACE_SIZE = 32;

template <typename T>
static inline T CeilDiv(T num, T rnd)
{
    return (((rnd) == 0) ? 0 : (((num) + (rnd) - 1) / (rnd)));
}

template <typename T>
static inline T CeilAlign(T num, T rnd)
{
    return (((rnd) == 0) ? 0 : (((num) + (rnd) - 1) / (rnd)) * (rnd));
}

ge::graphStatus GatherSelectionKvCacheTiling::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    OPS_ERR_IF(platformInfo == nullptr, OPS_LOG_E(context_->GetNodeName(), "get platformInfo nullptr."),
        return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    coreNum_ = ascendcPlatform.GetCoreNumAiv();
    OPS_ERR_IF(
        coreNum_ <= 0, OPS_LOG_E(context_->GetNodeName(), "coreNum must be greater than 0."),
        return ge::GRAPH_FAILED);

    uint64_t ubSizePlatForm;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    ubSize_ = static_cast<int64_t>(ubSizePlatForm);
    OPS_ERR_IF(
        ubSize_ <= 0, OPS_LOG_E(context_->GetNodeName(), "ubSize must be greater than 0."),
        return ge::GRAPH_FAILED);

    ubBlockSize_ = 32; // 32: ub block size

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GatherSelectionKvCacheTiling::GetInputAttrs()
{
    auto attrs = context_->GetAttrs();
    OPS_ERR_IF(attrs == nullptr, OPS_LOG_E(context_->GetNodeName(), "get attrs nullptr."),
        return ge::GRAPH_FAILED);
    const int64_t* attrBlockSizePtr = attrs->GetAttrPointer<int64_t>(0);
    if (attrBlockSizePtr == nullptr || *attrBlockSizePtr <= 0) {
        selTopKBlockSize_ = DEFAULT_TOPK_BLOCK_SIZE;
    } else {
        selTopKBlockSize_ = *attrBlockSizePtr;
    }
    tilingData_.set_selTopKBlockSize(selTopKBlockSize_);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GatherSelectionKvCacheTiling::GetSelKvCacheShape()
{
    // selection_k_rope: [s_block_num, s_block_size, k_rope]
    auto selKRopeIn = context_->GetInputShape(SEL_K_ROPE_IDX);
    OPS_ERR_IF(selKRopeIn == nullptr, OPS_LOG_E(context_->GetNodeName(), "get selKRopeIn nullptr."),
        return ge::GRAPH_FAILED);
    gert::Shape selKRopeShape = selKRopeIn->GetStorageShape();
    size_t dimsNSelKRope = selKRopeShape.GetDimNum();
    int64_t ifQuant = 0;

    OPS_ERR_IF(
        (dimsNSelKRope != CONST3 && dimsNSelKRope != 1),
        OPS_LOG_E(context_->GetNodeName(), "selection_k_rope dim:%lu should be 3 or 1.", dimsNSelKRope),
        return ge::GRAPH_FAILED);

    // selection_kv_cache: [s_block_num, s_block_size, kv_cache]
    auto selKvCacheIn = context_->GetInputShape(SEL_KV_CACHE_IDX);
    OPS_ERR_IF(selKvCacheIn == nullptr, OPS_LOG_E(context_->GetNodeName(), "get selKvCacheIn nullptr."),
        return ge::GRAPH_FAILED);
    gert::Shape selKvCacheShape = selKvCacheIn->GetStorageShape();
    size_t dimsN = selKvCacheShape.GetDimNum();
    OPS_ERR_IF(
        (dimsN != CONST3),
        OPS_LOG_E(context_->GetNodeName(), "selection_kv_cache dim:%lu should be 3.", dimsN),
        return ge::GRAPH_FAILED);
    tilingData_.set_selKvBlockNum(selKvCacheShape.GetDim(0));
    tilingData_.set_selKvBlockSize(selKvCacheShape.GetDim(1));
    
    if (dimsNSelKRope == CONST3) {
        // selKvCacheShape selKRopeShape 前两维相同
        OPS_ERR_IF(
            (selKvCacheShape.GetDim(0) != selKRopeShape.GetDim(0) ||
            selKvCacheShape.GetDim(1) != selKRopeShape.GetDim(1)),
            OPS_LOG_E(context_->GetNodeName(),
                "sel_kv_cache sel_k_rope dim0 [%ld %ld] or dim1 [%ld %ld] should be equal.",
                selKvCacheShape.GetDim(0), selKRopeShape.GetDim(0), selKvCacheShape.GetDim(1), selKRopeShape.GetDim(1)),
            return ge::GRAPH_FAILED);
    } else {
        OPS_ERR_IF(
            (selKRopeShape.GetDim(0) != 0),
            OPS_LOG_E(context_->GetNodeName(),
                "selection_k_rope dim0:[%ld] should be 0 when the dim of selection_k_rope is 1.",
                selKRopeShape.GetDim(0)),
            return ge::GRAPH_FAILED);
        ifQuant = 1;
    }

    OPS_ERR_IF(
            (selKvCacheShape.GetDim(1) % selTopKBlockSize_ != 0),
            OPS_LOG_E(context_->GetNodeName(), "selection_kv_cache dim1:%ld should be multiple of %ld.",
                selKvCacheShape.GetDim(1), selTopKBlockSize_),
            return ge::GRAPH_FAILED);

    tilingData_.set_ifQuant(ifQuant);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GatherSelectionKvCacheTiling::GetSelBlockTable()
{
    // selection_kv_block_table: [batchsize*seq*headnum, s_maxblocknum]
    auto selKvBlkTIn = context_->GetInputShape(SEL_KV_BLOCK_TABLE_IDX);
    OPS_ERR_IF(selKvBlkTIn == nullptr, OPS_LOG_E(context_->GetNodeName(), "get selKvBlkTIn nullptr."),
        return ge::GRAPH_FAILED);
    gert::Shape selKvBlkTInShape = selKvBlkTIn->GetStorageShape();
    size_t dimsN = selKvBlkTInShape.GetDimNum();
    OPS_ERR_IF(
        (dimsN != CONST2),
        OPS_LOG_E(
            context_->GetNodeName(), "selection_kv_block_table dim:%lu should be 2.", dimsN),
        return ge::GRAPH_FAILED);
    // 校验dim0
    tilingData_.set_selMaxBlockNum(selKvBlkTInShape.GetDim(1));
    selKvBlockTableRow_ = selKvBlkTInShape.GetDim(0);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GatherSelectionKvCacheTiling::GetTopkIndices()
{
    // selection_kv_block_status: [batchsize, seq, headnum, topk+1]
    auto selKvBlkStIn = context_->GetInputShape(SEL_KV_BLOCK_STAT_IDX);
    OPS_ERR_IF(selKvBlkStIn == nullptr, OPS_LOG_E(context_->GetNodeName(), "get selKvBlkStIn nullptr."),
        return ge::GRAPH_FAILED);
    gert::Shape selKvBlkStShape = selKvBlkStIn->GetStorageShape();
    size_t dimsN = selKvBlkStShape.GetDimNum();
    OPS_ERR_IF(
        (dimsN != CONST4 && dimsN != CONST3),
        OPS_LOG_E(
            context_->GetNodeName(), "selection_kv_block_status dim:%lu should be 3 or 4.", dimsN),
        return ge::GRAPH_FAILED);

    // selection_topk_indices: [batchsize, seq, headnum, topk].  BSND or TND
    auto selTopKIn = context_->GetInputShape(SEL_TOPK_INDICES_IDX);
    OPS_ERR_IF(selTopKIn == nullptr, OPS_LOG_E(context_->GetNodeName(), "get selTopKIn nullptr."),
        return ge::GRAPH_FAILED);
    gert::Shape selTopKInShape = selTopKIn->GetStorageShape();
    dimsN = selTopKInShape.GetDimNum();
    OPS_ERR_IF(
        (dimsN != CONST4 && dimsN != CONST3),
        OPS_LOG_E(context_->GetNodeName(), "selection_topk_indices dim:%lu should be 3 or 4.",
            dimsN),
        return ge::GRAPH_FAILED);

    OPS_ERR_IF(
        (selKvBlkStShape.GetDimNum() != selTopKInShape.GetDimNum()),
        OPS_LOG_E(context_->GetNodeName(),
            "selection_kv_block_status dim:%lu not equal selection_topk_indices dime:%lu.",
            selKvBlkStShape.GetDimNum(), selTopKInShape.GetDimNum()),
        return ge::GRAPH_FAILED);

    if (dimsN == CONST4) {
        topKLayout_ = DataLayout::BSND;
        batchSize_ = selTopKInShape.GetDim(0);
        seq_ = selTopKInShape.GetDim(CONST1);
        headnum_ = selTopKInShape.GetDim(CONST2);
        topk_ = selTopKInShape.GetDim(CONST3);
        OPS_ERR_IF(
            (selKvBlkStShape.GetDim(0) != batchSize_ || selKvBlkStShape.GetDim(1) != seq_ ||
                selKvBlkStShape.GetDim(CONST2) != headnum_ || selKvBlkStShape.GetDim(CONST3) != topk_ + 1),
            OPS_LOG_E(context_->GetNodeName(),
                "selection_kv_block_status[%ld %ld %ld %ld] selection_topk_indices[%ld %ld %ld %ld] is not satisfied",
                selKvBlkStShape.GetDim(0), selKvBlkStShape.GetDim(1), selKvBlkStShape.GetDim(CONST2),
                selKvBlkStShape.GetDim(CONST3), batchSize_, seq_, headnum_, topk_),
            return ge::GRAPH_FAILED);
    } else {
        topKLayout_ = DataLayout::TND;
        t_ = selTopKInShape.GetDim(0);
        headnum_ = selTopKInShape.GetDim(CONST1);
        topk_ = selTopKInShape.GetDim(CONST2);
        OPS_ERR_IF(
            (selKvBlkStShape.GetDim(0) != t_ || selKvBlkStShape.GetDim(1) != headnum_ ||
                selKvBlkStShape.GetDim(CONST2) != topk_ + 1),
            OPS_LOG_E(context_->GetNodeName(),
                "selection_kv_block_status[%ld %ld %ld] selection_topk_indices[%ld %ld %ld] is not satisfied",
                selKvBlkStShape.GetDim(0), selKvBlkStShape.GetDim(1), selKvBlkStShape.GetDim(CONST2),
                t_, headnum_, topk_),
            return ge::GRAPH_FAILED);
    }

    OPS_ERR_IF(
            headnum_ != 1,
            OPS_LOG_E(context_->GetNodeName(), "headnum:%ld should be 1.",
                headnum_),
            return ge::GRAPH_FAILED);

    OPS_ERR_IF(
            (topk_ > TOPK_SPLIT_NUM && selTopKBlockSize_ != 1),
            OPS_LOG_E(context_->GetNodeName(),
                "topk:%ld is more than %ld, only support block size 1 but got :%d",
                topk_, TOPK_SPLIT_NUM, selTopKBlockSize_),
            return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GatherSelectionKvCacheTiling::CheckSelInfo()
{
    OPS_ERR_IF(
        (topk_ > MAX_TOPK_NUM),
        OPS_LOG_E(context_->GetNodeName(), "selection_topk_indices topk_:%ld should <= %d.",
            topk_, MAX_TOPK_NUM),
        return ge::GRAPH_FAILED);

    OPS_ERR_IF(
        (headnum_ != 1),
        OPS_LOG_E(context_->GetNodeName(), "selection_topk_indices headnum_:%ld only surpport 1.",
            headnum_),
        return ge::GRAPH_FAILED);

    // selection_kv_block_table: [batchsize*seq*headnum, s_maxblocknum]
    int64_t BSH = (topKLayout_ == DataLayout::TND) ? t_ * headnum_ : batchSize_ * seq_ * headnum_;
    OPS_ERR_IF(
        (selKvBlockTableRow_ < BSH),
        OPS_LOG_E(context_->GetNodeName(), "selection_kv_block_table dim0:%lu need >= BSH:%ld.",
            selKvBlockTableRow_, BSH),
        return ge::GRAPH_FAILED);

    // s_block_num >= batchSize_ * seq_ * headnum_ * s_maxblocknum
    OPS_ERR_IF(
        (tilingData_.get_selKvBlockNum() < BSH * tilingData_.get_selMaxBlockNum()),
        OPS_LOG_E(
            context_->GetNodeName(),
            "selKvBlockNum:%ld should >= BSH(%ld)*selMaxBlockNum(%ld)",
            tilingData_.get_selKvBlockNum(), BSH, tilingData_.get_selMaxBlockNum()),
        return ge::GRAPH_FAILED);

    // s_maxblocknum >= ceil(topk*64, s_block_size)
    int64_t minBlockNum = (topk_ * selTopKBlockSize_ + tilingData_.get_selKvBlockSize() - 1) /
                          tilingData_.get_selKvBlockSize();
    OPS_ERR_IF(
        (tilingData_.get_selMaxBlockNum() < minBlockNum),
        OPS_LOG_E(
            context_->GetNodeName(),
            "s_maxblocknum:%ld is too small, topk_:%ld selTopKBlockSize_:%ld s_block_size:%ld",
            tilingData_.get_selMaxBlockNum(), topk_, selTopKBlockSize_, tilingData_.get_selKvBlockSize()),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GatherSelectionKvCacheTiling::GetFullKvCacheShape()
{
    // [f_block_num, block_size, k_rope]
    auto fulKRopeIn = context_->GetInputShape(FULL_K_ROPE_IDX);
    OPS_ERR_IF(fulKRopeIn == nullptr, OPS_LOG_E(context_->GetNodeName(), "get fulKRopeIn nullptr."),
        return ge::GRAPH_FAILED);
    gert::Shape fulKRopeShape = fulKRopeIn->GetStorageShape();
    size_t dimsNFullKRope = fulKRopeShape.GetDimNum();
    OPS_ERR_IF(
        (dimsNFullKRope != CONST3 && dimsNFullKRope != 1),
        OPS_LOG_E(context_->GetNodeName(), "full_k_rope dim:%lu should be 3 or 1.", dimsNFullKRope),
        return ge::GRAPH_FAILED);

    // [f_block_num, block_size, kv_cache]
    auto fulKvCacheIn = context_->GetInputShape(FULL_KV_CACHE_IDX);
    OPS_ERR_IF(fulKvCacheIn == nullptr, OPS_LOG_E(context_->GetNodeName(), "get fulKvCacheIn nullptr."),
        return ge::GRAPH_FAILED);
    gert::Shape fulKvCacheInShape = fulKvCacheIn->GetStorageShape();
    size_t dimsN = fulKvCacheInShape.GetDimNum();

    OPS_ERR_IF(
        (dimsN != CONST3),
        OPS_LOG_E(context_->GetNodeName(), "full_kv_cache dim:%lu should be 3.", dimsN),
        return ge::GRAPH_FAILED);

    tilingData_.set_kvCacheDim(fulKvCacheInShape.GetDim(CONST2));
    // fulKvCacheInShape fulKRopeShape 前两维相同
    if (dimsNFullKRope == CONST3) {
        OPS_ERR_IF(
            (fulKvCacheInShape.GetDim(0) != fulKRopeShape.GetDim(0) ||
            fulKvCacheInShape.GetDim(1) != fulKRopeShape.GetDim(1)),
            OPS_LOG_E(context_->GetNodeName(),
                "ful_kv_cache ful_k_rope dim0 [%ld %ld] or dim1 [%ld %ld] should be equal.",
                fulKvCacheInShape.GetDim(0),
                fulKRopeShape.GetDim(0), fulKvCacheInShape.GetDim(1), fulKRopeShape.GetDim(1)),
            return ge::GRAPH_FAILED);
    }
    
    OPS_ERR_IF(
            (fulKvCacheInShape.GetDim(1) % selTopKBlockSize_ != 0),
            OPS_LOG_E(context_->GetNodeName(), "full_kv_cache dim1:%ld should be multiple of %ld.",
                fulKvCacheInShape.GetDim(1), selTopKBlockSize_),
            return ge::GRAPH_FAILED);

    OPS_ERR_IF(
        (tilingData_.get_kRopeDim() > MAX_K_ROPE_DIM || tilingData_.get_kvCacheDim() > MAX_KV_CACHE_DIM),
        OPS_LOG_E(context_->GetNodeName(),
            "kRopeDim:%ld should <= %d and kvCacheDim:%ld should <= %d.",
            tilingData_.get_kRopeDim(), MAX_K_ROPE_DIM, tilingData_.get_kvCacheDim(), MAX_KV_CACHE_DIM),
        return ge::GRAPH_FAILED);
    
    if (dimsNFullKRope == CONST3) {
        tilingData_.set_fullKvBlockNum(fulKRopeShape.GetDim(0));
        tilingData_.set_fullKvBlockSize(fulKRopeShape.GetDim(1));
        tilingData_.set_kRopeDim(fulKRopeShape.GetDim(CONST2));
    } else {
        tilingData_.set_fullKvBlockNum(fulKvCacheInShape.GetDim(0));
        tilingData_.set_fullKvBlockSize(fulKvCacheInShape.GetDim(1));
        tilingData_.set_kRopeDim(0);

        OPS_ERR_IF(
            (fulKRopeShape.GetDim(0) != 0),
            OPS_LOG_E(context_->GetNodeName(),
                "full_k_rope dim0: [%ld] should be 0 when the dim of full_k_rope is 1.", fulKRopeShape.GetDim(0)),
            return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GatherSelectionKvCacheTiling::GetFullKvBlkTable()
{
    // [batchsize, f_maxblocknum]
    auto fulKvBlkTIn = context_->GetInputShape(FULL_KV_BLOCK_TABLE_IDX);
    OPS_ERR_IF(fulKvBlkTIn == nullptr, OPS_LOG_E(context_->GetNodeName(), "get fulKvBlkTIn nullptr."),
        return ge::GRAPH_FAILED);
    gert::Shape fulKvBlkTInShape = fulKvBlkTIn->GetStorageShape();
    size_t dimsN = fulKvBlkTInShape.GetDimNum();
    OPS_ERR_IF(
        (dimsN != CONST2),
        OPS_LOG_E(context_->GetNodeName(), "full_kv_block_table dim:%lu should be 2.", dimsN),
        return ge::GRAPH_FAILED);
    OPS_ERR_IF(
        (topKLayout_ == DataLayout::BSND && fulKvBlkTInShape.GetDim(0) != batchSize_),
        OPS_LOG_E(context_->GetNodeName(),
            "full_kv_block_table dim0:%ld should be batchSize:%ld.", fulKvBlkTInShape.GetDim(0), batchSize_),
        return ge::GRAPH_FAILED);

    tilingData_.set_fullMaxBlockNum(fulKvBlkTInShape.GetDim(1));
    if (topKLayout_ == DataLayout::TND) {
        batchSize_ = fulKvBlkTInShape.GetDim(0);
        OPS_ERR_IF(
            (t_ % batchSize_ != 0),
            OPS_LOG_E(context_->GetNodeName(),
                "TND format t_:%ld must be multiple of batchSize_:%ld", t_, batchSize_),
            return ge::GRAPH_FAILED);
        OPS_ERR_IF(
            (batchSize_ == 0),
            OPS_LOG_E(context_->GetNodeName(),
                "batchSize_:%ld must not be 0", batchSize_),
            return ge::GRAPH_FAILED);
            
        seq_ = t_ / batchSize_;
    }

    OPS_ERR_IF(
        (seq_ >= MAX_Q_SEQ_LEN), OPS_LOG_E(context_->GetNodeName(),
            "seq_:%ld should be less than %d.", seq_, MAX_Q_SEQ_LEN),
        return ge::GRAPH_FAILED);

    tilingData_.set_rawSeq(seq_);
    tilingData_.set_headnum(headnum_);
    tilingData_.set_topk(topk_);
    tilingData_.set_layOut(static_cast<int64_t>(topKLayout_));

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GatherSelectionKvCacheTiling::GetSeqLenIn()
{
    // full_kv_actual_seq: [batchsize]
    auto fulKvSeqIn = context_->GetInputShape(FULL_KV_ACTSEQ_IDX);
    OPS_ERR_IF(fulKvSeqIn == nullptr, OPS_LOG_E(context_->GetNodeName(), "get fulKvSeqIn nullptr."),
        return ge::GRAPH_FAILED);
    gert::Shape fulKvSeqInShape = fulKvSeqIn->GetStorageShape();
    size_t dimsN = fulKvSeqInShape.GetDimNum();
    OPS_ERR_IF(
        (dimsN != 1),
        OPS_LOG_E(context_->GetNodeName(), "full_kv_actual_seq dim:%lu should be 1.", dimsN),
        return ge::GRAPH_FAILED);

    // full_q_actual_seq: [batchsize]
    auto fulQSeqIn = context_->GetInputShape(FULL_Q_ACTSEQ_IDX);
    OPS_ERR_IF(fulQSeqIn == nullptr, OPS_LOG_E(context_->GetNodeName(), "get fulQSeqIn nullptr."),
        return ge::GRAPH_FAILED);
    gert::Shape fulQSeqInShape = fulQSeqIn->GetStorageShape();
    dimsN = fulQSeqInShape.GetDimNum();
    OPS_ERR_IF(
        (dimsN != 1),
        OPS_LOG_E(context_->GetNodeName(), "full_q_actual_seq dim:%lu should be 1.", dimsN),
        return ge::GRAPH_FAILED);

    OPS_ERR_IF(
        (fulKvSeqInShape.GetDim(0) != batchSize_ || fulQSeqInShape.GetDim(0) != batchSize_),
        OPS_LOG_E(context_->GetNodeName(),
            "full_kv_actual_seq dim0:%ld or full_q_actual_seq dim0:%ld should be equal batchSize:%ld.",
            fulKvSeqInShape.GetDim(0), fulQSeqInShape.GetDim(0), batchSize_),
        return ge::GRAPH_FAILED);
    
    batchSize_ = batchSize_ * seq_;
    seq_ = 1;
    tilingData_.set_batchsize(batchSize_);
    tilingData_.set_seq(seq_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GatherSelectionKvCacheTiling::GetShapeAttrsInfo()
{
    OPS_ERR_IF(
        context_ == nullptr, OPS_LOG_E("GatherSelectionKvCache", "context can not be nullptr."),
        return ge::GRAPH_FAILED);

    if (GetInputAttrs() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    if (GetSelKvCacheShape() != ge::GRAPH_SUCCESS || GetSelBlockTable() != ge::GRAPH_SUCCESS ||
        GetTopkIndices() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    if (CheckSelInfo() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    if (GetFullKvCacheShape() != ge::GRAPH_SUCCESS || GetFullKvBlkTable() != ge::GRAPH_SUCCESS ||
        GetSeqLenIn() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    // dtype校验
    if (GetInputDtypeInfo() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GatherSelectionKvCacheTiling::GetInputDtypeInfo()
{
    auto selKRopeDesc = context_->GetInputDesc(SEL_K_ROPE_IDX);
    OPS_ERR_IF(selKRopeDesc == nullptr, OPS_LOG_E(context_->GetNodeName(), "get selKRopeDesc nullptr."),
        return ge::GRAPH_FAILED);
    selKRopeDtype_ = selKRopeDesc->GetDataType();
    OPS_ERR_IF(
        (selKRopeDtype_ != ge::DT_FLOAT16 && selKRopeDtype_ != ge::DT_BF16 && selKRopeDtype_ != ge::DT_INT8),
        OPS_LOG_E(context_->GetNodeName(), "selKRopeDtype_ is not supported."),
        return ge::GRAPH_FAILED);

    auto selKvCacheDesc = context_->GetInputDesc(SEL_KV_CACHE_IDX);
    OPS_ERR_IF(selKvCacheDesc == nullptr, OPS_LOG_E(context_->GetNodeName(), "get selKvCacheDesc nullptr."),
        return ge::GRAPH_FAILED);
    ge::DataType selKvCacheDtype = selKvCacheDesc->GetDataType();

    auto selKvBTDesc = context_->GetInputDesc(SEL_KV_BLOCK_TABLE_IDX);
    OPS_ERR_IF(selKvBTDesc == nullptr, OPS_LOG_E(context_->GetNodeName(), "get selKvBTDesc nullptr."),
        return ge::GRAPH_FAILED);
    ge::DataType selKvBTDtype = selKvBTDesc->GetDataType();

    auto selKvBSDesc = context_->GetInputDesc(SEL_KV_BLOCK_STAT_IDX);
    OPS_ERR_IF(selKvBSDesc == nullptr, OPS_LOG_E(context_->GetNodeName(), "get selKvBSDesc nullptr."),
        return ge::GRAPH_FAILED);
    ge::DataType selKvBSDtype = selKvBSDesc->GetDataType();

    auto selTopkInDesc = context_->GetInputDesc(SEL_TOPK_INDICES_IDX);
    OPS_ERR_IF(selTopkInDesc == nullptr, OPS_LOG_E(context_->GetNodeName(), "get selTopkInDesc nullptr."),
        return ge::GRAPH_FAILED);
    ge::DataType selTopkInDtype = selTopkInDesc->GetDataType();

    auto fKRopeDesc = context_->GetInputDesc(FULL_K_ROPE_IDX);
    OPS_ERR_IF(fKRopeDesc == nullptr, OPS_LOG_E(context_->GetNodeName(), "get fKRopeDesc nullptr."),
        return ge::GRAPH_FAILED);
    ge::DataType fKRopeDtype = fKRopeDesc->GetDataType();

    auto fKvCacheDesc = context_->GetInputDesc(FULL_KV_CACHE_IDX);
    OPS_ERR_IF(fKvCacheDesc == nullptr, OPS_LOG_E(context_->GetNodeName(), "get fKvCacheDesc nullptr."),
        return ge::GRAPH_FAILED);
    ge::DataType fKvCacheDtype = fKvCacheDesc->GetDataType();

    auto fKvBTDesc = context_->GetInputDesc(FULL_KV_BLOCK_TABLE_IDX);
    OPS_ERR_IF(fKvBTDesc == nullptr, OPS_LOG_E(context_->GetNodeName(), "get fKvBTDesc nullptr."),
        return ge::GRAPH_FAILED);
    ge::DataType fKvBTDtype = fKvBTDesc->GetDataType();

    auto fKvActSeqDesc = context_->GetInputDesc(FULL_KV_ACTSEQ_IDX);
    OPS_ERR_IF(fKvActSeqDesc == nullptr, OPS_LOG_E(context_->GetNodeName(), "get fKvActSeqDesc nullptr."),
        return ge::GRAPH_FAILED);
    ge::DataType fKvActSeqDtype = fKvActSeqDesc->GetDataType();

    auto fQActSeqDesc = context_->GetInputDesc(FULL_Q_ACTSEQ_IDX);
    OPS_ERR_IF(fQActSeqDesc == nullptr, OPS_LOG_E(context_->GetNodeName(), "get fQActSeqDesc nullptr."),
        return ge::GRAPH_FAILED);
    ge::DataType fQActSeqDtype = fQActSeqDesc->GetDataType();

    OPS_ERR_IF(
        (selKvCacheDtype != selKRopeDtype_ || fKRopeDtype != selKRopeDtype_ || fKvCacheDtype != selKRopeDtype_),
        OPS_LOG_E(context_->GetNodeName(), "kv cache dtype is not supported."),
        return ge::GRAPH_FAILED);

    OPS_ERR_IF(
        (selKvBTDtype != ge::DT_INT32 || selKvBSDtype != ge::DT_INT32 || selTopkInDtype != ge::DT_INT32 ||
            fKvBTDtype != ge::DT_INT32 || fKvActSeqDtype != ge::DT_INT32 || fQActSeqDtype != ge::DT_INT32),
        OPS_LOG_E(context_->GetNodeName(), "kv cache idx dtype should be int32."),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GatherSelectionKvCacheTiling::DoOpTiling()
{
    if (batchSize_ == 0) {
        tilingData_.set_usedCoreNum(0);
        context_->SetBlockDim(1);
        return ge::GRAPH_SUCCESS;
    }

    int64_t bsCoreFactor = CeilDiv(batchSize_, static_cast<int64_t>(coreNum_));
    int64_t bsCoreNum = CeilDiv(batchSize_, bsCoreFactor);
    tilingData_.set_usedCoreNum(bsCoreNum);

    bsCoreFactor = CeilDiv(batchSize_, bsCoreNum);
    int64_t tailCoreBsFactor = batchSize_ - (bsCoreNum - 1) * bsCoreFactor;

    tilingData_.set_mainCoreBsLoopNum(bsCoreFactor);
    tilingData_.set_tailCoreBsLoopNum(tailCoreBsFactor);

    tilingKey_ = 0;
    if (topk_ <= TOPK_SPLIT_NUM) {
        tilingKey_ = 1; // 1:reuse
    } else {
        tilingKey_ = 2; // 2:reuse vec
    }

    int64_t kRopeUbV =
        tilingData_.get_selTopKBlockSize() * tilingData_.get_kRopeDim() * ge::GetSizeByDataType(selKRopeDtype_);
    int64_t kvCacheUbV =
        tilingData_.get_selTopKBlockSize() * tilingData_.get_kvCacheDim() * ge::GetSizeByDataType(selKRopeDtype_);
    int64_t kRopeUbSize = CeilAlign(kRopeUbV, ubBlockSize_);
    int64_t kvCacheUbSize = CeilAlign(kvCacheUbV, ubBlockSize_);

    int64_t SH = seq_ * headnum_;
    int64_t selTopKUb = SH * topk_ * sizeof(int32_t);
    selTopKUb = CeilAlign(selTopKUb, ubBlockSize_);
    int64_t topkStatUb = SH * (topk_ + 1) * sizeof(int32_t);
    topkStatUb = CeilAlign(topkStatUb, ubBlockSize_);
    int64_t selKvBlockTabUbSize = SH * tilingData_.get_selMaxBlockNum() * sizeof(int32_t);
    selKvBlockTabUbSize = CeilAlign(selKvBlockTabUbSize, ubBlockSize_);
    int64_t selKvSeqLenUbSize = CeilAlign(static_cast<int64_t>(SH * sizeof(int32_t)), ubBlockSize_);
    OPS_ERR_IF(
        (kRopeUbSize + kvCacheUbSize + selTopKUb + topkStatUb + selKvBlockTabUbSize + selKvSeqLenUbSize > ubSize_),
        OPS_LOG_E(context_->GetNodeName(),
            "ub size not enough, kRopeDim:%ld kvCacheDim:%ld BlockSize:%ld s:%ld h:%ld topk:%ld selMaxBlockNum:%ld.",
            tilingData_.get_kRopeDim(), tilingData_.get_kvCacheDim(), tilingData_.get_selTopKBlockSize(),
            seq_, headnum_, topk_, tilingData_.get_selMaxBlockNum()),
        return ge::GRAPH_FAILED);

    if ((kRopeUbSize + kvCacheUbSize) * CONST2 <=
            (ubSize_ - selTopKUb - topkStatUb - selKvBlockTabUbSize - selKvSeqLenUbSize)) {
        tilingData_.set_buffNum(CONST2);
    } else {
        OPS_LOG_E(context_->GetNodeName(),
                  "kRopeDim:%ld and kvCacheDim:%ld too big, not support",
                  tilingData_.get_kRopeDim(), tilingData_.get_kvCacheDim());
        return ge::GRAPH_FAILED;
    }

    tilingData_.set_kRopeUbSize(kRopeUbSize);
    tilingData_.set_kvCacheUbSize(kvCacheUbSize);

    context_->SetBlockDim(bsCoreNum);

    return ge::GRAPH_SUCCESS;
}

uint64_t GatherSelectionKvCacheTiling::GetTilingKey() const
{
    return tilingKey_;
}

void GatherSelectionKvCacheTiling::PrintTilingDatas()
{
    OPS_LOG_I(
        context_->GetNodeName(),
        "tilingData is coreNum:%ld ubSize:%ld usedCoreNum:%ld mainCoreBsLoopNum:%ld tailCoreBsLoopNum:%ld, \
          selTopKBlockSize:%ld, fullKvBlockNum:%ld, fullKvBlockSize:%ld, kRopeDim:%ld, \
          kvCacheDim:%ld, selKvBlockNum:%ld, selKvBlockSize:%ld, fullMaxBlockNum:%ld, selMaxBlockNum:%ld, \
          batchsize:%ld, seq:%ld, headnum:%ld, topk:%ld, kRopeUbSize:%ld, kvCacheUbSize:%ld tilingKey_:%lu, \
          layOut:%ld",
        coreNum_, ubSize_, tilingData_.get_usedCoreNum(), tilingData_.get_mainCoreBsLoopNum(),
        tilingData_.get_tailCoreBsLoopNum(), tilingData_.get_selTopKBlockSize(), tilingData_.get_fullKvBlockNum(),
        tilingData_.get_fullKvBlockSize(), tilingData_.get_kRopeDim(), tilingData_.get_kvCacheDim(),
        tilingData_.get_selKvBlockNum(), tilingData_.get_selKvBlockSize(), tilingData_.get_fullMaxBlockNum(),
        tilingData_.get_selMaxBlockNum(), tilingData_.get_batchsize(), tilingData_.get_seq(), tilingData_.get_headnum(),
        tilingData_.get_topk(), tilingData_.get_kRopeUbSize(), tilingData_.get_kvCacheUbSize(), tilingKey_,
        tilingData_.get_layOut());
}

ge::graphStatus GatherSelectionKvCacheTiling::PostTiling()
{
    PrintTilingDatas();
    context_->SetTilingKey(GetTilingKey());
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    OPS_ERR_IF(workspaces == nullptr, OPS_LOG_E(context_->GetNodeName(), "get workspaces nullptr."),
        return ge::GRAPH_FAILED);
    workspaces[0] = static_cast<size_t>(DEFAULT_WORKSPACE_SIZE);
    OPS_ERR_IF(context_->GetRawTilingData() == nullptr, OPS_LOG_E(context_->GetNodeName(), "get tilingdata nullptr."),
        return ge::GRAPH_FAILED);
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GatherSelectionKvCacheTiling::RunTiling()
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

ge::graphStatus Tiling4GatherSelectionKvCache(gert::TilingContext* context)
{
    OPS_LOG_I(context->GetNodeName(), "TilingForGatherSelectionKvCache running.");
    GatherSelectionKvCacheTiling tiling(context);
    return tiling.RunTiling();
}

ge::graphStatus TilingPrepare4GatherSelectionKvCache(gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(GatherSelectionKvCache)
    .Tiling(Tiling4GatherSelectionKvCache)
    .TilingParse<GatherSelectionKvCacheCompileInfo>(TilingPrepare4GatherSelectionKvCache);

} // namespace optiling