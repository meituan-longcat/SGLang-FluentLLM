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
 * \file gather_selection_kv_cache_split_bs_reuse_vec.h
 * \brief
 */
#ifndef GATHER_SELECTION_KV_CACHE_SPLIT_BS_REUSE_VEC_H
#define GATHER_SELECTION_KV_CACHE_SPLIT_BS_REUSE_VEC_H

#include "kernel_operator.h"

namespace GatherSelectionKvCacheNs {
using namespace AscendC;

constexpr int32_t COMPARE_SCALAR_NUM = 256 / sizeof(int32_t);
constexpr int32_t ONE_REPEAT_SORT_NUM = 32;
constexpr int32_t COMPARENUM_SIZE = 16;
constexpr int32_t TOPK_NUMS = 8;
constexpr int32_t SORT_OFFSET = 4;
constexpr int32_t SORTED_STAT_TOPK_IDX_OFFSET = 3;
constexpr int32_t SORTED_STAT_TOPK_OFFSET = 2;

template <typename T>
class GatherSelectionKvCacheSplitBsReuseVec {
public:
    __aicore__ inline GatherSelectionKvCacheSplitBsReuseVec(TPipe* pipe, const GatherSelectionKvCacheTilingData* tiling)
        : pipe_(pipe), tiling_(tiling)
    {}

    __aicore__ inline void Init(
        GM_ADDR selection_k_rope, GM_ADDR selection_kv_cache, GM_ADDR selection_kv_block_table,
        GM_ADDR selection_kv_block_status, GM_ADDR selection_topk_indices, GM_ADDR full_k_rope, GM_ADDR full_kv_cache,
        GM_ADDR full_kv_block_table, GM_ADDR full_kv_actual_seq, GM_ADDR full_q_actual_seq,
        GM_ADDR selection_kv_actual_seq)
    {
        blkIdx_ = GetBlockIdx();
        if (blkIdx_ >= tiling_->usedCoreNum) {
            return;
        }
        
        rawSeq_ = tiling_->rawSeq;
        int64_t SH = rawSeq_ * tiling_->headnum;

        // 512*2=1KB 64*2=128B -> db: 2*(1k+128B)=2304B=3KB
        kRopeUbOffset_ = tiling_->kvCacheUbSize / sizeof(T);
        pipe_->InitBuffer(kvCacheQue_, tiling_->buffNum, tiling_->kvCacheUbSize + tiling_->kRopeUbSize);

        topkAlign_ = CeilAlign(static_cast<int64_t>(tiling_->topk),
                               static_cast<int64_t>(BLOCK_BYTES / sizeof(int32_t)));
        topkSortAlign_ = CeilAlign(static_cast<int32_t>(tiling_->topk), ONE_REPEAT_SORT_NUM);
        topkOneAlign_ = CeilAlign(static_cast<int64_t>(tiling_->topk + 1),
                                  static_cast<int64_t>(BLOCK_BYTES / sizeof(int32_t)));
        topkOneSortAlign_ = topkSortAlign_ > topkOneAlign_ ? topkSortAlign_ : topkOneAlign_;
        selTopKIdxUb_ = SH * topkSortAlign_ * sizeof(int32_t);
        pipe_->InitBuffer(selTopKIdxQue_, 1, selTopKIdxUb_); // SH*2048*4=S*8K

        selKvBlockTableUb_ = CeilAlign(
            static_cast<int64_t>(SH * tiling_->selMaxBlockNum * sizeof(int32_t)),
            static_cast<int64_t>(BLOCK_BYTES)); // S*(2048/128)*4=S*64B
        selKvActSeqUb_ = CeilAlign(
            static_cast<int64_t>(SH * sizeof(int32_t)), static_cast<int64_t>(BLOCK_BYTES)); // S*4B
        selBlockStatUb_ = SH * topkOneSortAlign_ * sizeof(int32_t); // S*2056*4=S*8.1KB
        pipe_->InitBuffer(workBuf_, selKvBlockTableUb_ + selKvActSeqUb_ + selBlockStatUb_ +
                          topkSortAlign_ * sizeof(int32_t) * TOPK_NUMS);

        bsLoopNum_ = (blkIdx_ == tiling_->usedCoreNum - 1) ? tiling_->tailCoreBsLoopNum : tiling_->mainCoreBsLoopNum;

        int64_t BsiSH = tiling_->mainCoreBsLoopNum * SH;
        // [s_block_num, block_size, k_rope] // 64
        selKRopeGm_.SetGlobalBuffer((__gm__ T*)selection_k_rope);
        // [s_block_num, block_size, kv_cache] // 512
        selKvCacheGm_.SetGlobalBuffer((__gm__ T*)selection_kv_cache);
        // [batchsize*seq*headnum, s_maxblocknum]  初始全-1，表示全空
        selKvBlockTableGm_.SetGlobalBuffer((__gm__ int32_t*)selection_kv_block_table);
        // [batchsize, seq, headnum, topk+1] 初始全-1，表示全空
        selKvBlockStatusGm_.SetGlobalBuffer((__gm__ int32_t*)selection_kv_block_status);
        // [batchsize, seq, headnum, topk] token粒度=groupid * selection_topk_block_size
        selTopKIndicesGm_.SetGlobalBuffer((__gm__ int32_t*)selection_topk_indices);
        // [f_block_num, block_size, k_rope]
        fullKRopeGm_.SetGlobalBuffer((__gm__ T*)full_k_rope);
        // [f_block_num, block_size, kv_cache]
        fullKvCacheGm_.SetGlobalBuffer((__gm__ T*)full_kv_cache);
        // [batchsize, f_maxblocknum]
        fullKvBlockTableGm_.SetGlobalBuffer((__gm__ int32_t*)full_kv_block_table);
        // [batchsize] 每个batch实际的seq长度
        fullKvActualSeqGm_.SetGlobalBuffer((__gm__ int32_t*)full_kv_actual_seq);
        // [batchsize] 每个batch实际的seq长度   用来算MTP场景不同T下面的kv_actual_seq
        fullQActualSeqGm_.SetGlobalBuffer((__gm__ int32_t*)full_q_actual_seq);
        // [batchsize*seq*headnum]
        selKvActualSeqGm_.SetGlobalBuffer((__gm__ int32_t*)selection_kv_actual_seq);
    }

    __aicore__ inline void Process()
    {
        if (blkIdx_ >= tiling_->usedCoreNum) {
            return;
        }

        LocalTensor<int32_t> selKvBlockTableLocal = workBuf_.Get<int32_t>();
        LocalTensor<int32_t> selKvActSeqLocal = selKvBlockTableLocal[selKvBlockTableUb_ / sizeof(int32_t)];
        LocalTensor<int32_t> selBlockStatLocal = selKvActSeqLocal[selKvActSeqUb_ / sizeof(int32_t)];
        topkIndicesLocal_ = selBlockStatLocal[selBlockStatUb_ / sizeof(int32_t)];
        insertStatusSameSeqLocal_ = topkIndicesLocal_[topkSortAlign_];
        hitFromSrcSeqLocal_ = insertStatusSameSeqLocal_[topkSortAlign_];
        sortBuf_ = hitFromSrcSeqLocal_[topkSortAlign_];

        ArithProgression<int32_t>(topkIndicesLocal_, 0, 1, topkSortAlign_);
        PipeBarrier<PIPE_V>();

        int64_t curFullKvSeqLen = 0;
        int64_t curFullQSeqLen = 0;

        for (int64_t bsIdx = 0; bsIdx < bsLoopNum_; bsIdx++) {
            int64_t curBatchSize = (blkIdx_ * tiling_->mainCoreBsLoopNum + bsIdx) / rawSeq_;
            int64_t curSeq = (blkIdx_ * tiling_->mainCoreBsLoopNum + bsIdx) % rawSeq_;
            int64_t offset = (rawSeq_ - 1) - (blkIdx_ * tiling_->mainCoreBsLoopNum + bsIdx) % rawSeq_;
            curFullKvSeqLen = fullKvActualSeqGm_.GetValue(curBatchSize);
            if (curFullKvSeqLen <= 0) {
                continue;
            }

            curFullQSeqLen = tiling_->seq;
            ASSERT_MSG(curFullQSeqLen <= tiling_->seq, "curFullQSeqLen:%ld cannot be greater than seq:%ld",
                curFullQSeqLen, tiling_->seq);

            CopyInTopKIndices(bsIdx);
            LocalTensor<int32_t> selTopKIdxLocal = selTopKIdxQue_.DeQue<int32_t>();

            SetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);
            CopyInBlockStatus(bsIdx, selBlockStatLocal);

            CopyInSelKvBlockTable(bsIdx, selKvBlockTableLocal);

            SetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);

            int64_t seqIdx = curSeq;
            int64_t curFullKvSeqModify = curFullKvSeqLen - (offset);
            if (curFullKvSeqModify > 0) {
                for (int64_t hnIdx = 0; hnIdx < tiling_->headnum; hnIdx++) {
                    LocalTensor<int32_t> tmpSelTopkLocal = selTopKIdxLocal[hnIdx * topkSortAlign_];
                    ProcessGatherTopK(bsIdx, seqIdx, hnIdx, curFullKvSeqModify, selBlockStatLocal, tmpSelTopkLocal,
                        selKvBlockTableLocal, selKvActSeqLocal, curFullQSeqLen);
                }
            }

            selTopKIdxQue_.FreeTensor(selTopKIdxLocal);

            SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
            CopyOutSelKvBlockTableAndSeqLen(bsIdx, selKvActSeqLocal, selBlockStatLocal);
        }
    }

private:
    template <HardEvent event>
    __aicore__ inline void SetWaitFlag(HardEvent evt)
    {
        event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(evt));
        SetFlag<event>(eventId);
        WaitFlag<event>(eventId);
    }

    __aicore__ inline int64_t GetActualQSeqLen(int64_t bsIdx)
    {
        if (tiling_->layOut == static_cast<int64_t>(LAYOUT::BSND)) {
            return tiling_->seq;
        } else if (bsIdx > 0) {
            return fullQActualSeqGm_.GetValue(bsIdx) - fullQActualSeqGm_.GetValue(bsIdx - 1);
        } else {
            return fullQActualSeqGm_.GetValue(bsIdx);
        }
    }

    __aicore__ inline void CopyInTopKIndices(int64_t bsIdx)
    {
        LocalTensor<int32_t> selTopKIdxLocal = selTopKIdxQue_.AllocTensor<int32_t>();
        int64_t SH = rawSeq_ * tiling_->headnum;
        uint8_t padCnt = topkAlign_ - tiling_->topk;
        uint32_t dstStride = (topkSortAlign_ - topkAlign_) / (BLOCK_BYTES / sizeof(int32_t));
        int64_t curBatchSize = (blkIdx_ * tiling_->mainCoreBsLoopNum + bsIdx) / rawSeq_;
        int64_t curSeq = (blkIdx_ * tiling_->mainCoreBsLoopNum + bsIdx) % rawSeq_;

        DataCopyPadExtParams<int32_t> dataCopyPadParams{true, 0, padCnt, -1};
        DataCopyExtParams dataCopyParams{
            static_cast<uint16_t>(tiling_->headnum),
            static_cast<uint32_t>(tiling_->topk * sizeof(int32_t)), 0, dstStride, 0};
        DataCopyPad(selTopKIdxLocal,
            selTopKIndicesGm_[curBatchSize * SH * tiling_->topk + curSeq * tiling_->headnum * tiling_->topk],
            dataCopyParams, dataCopyPadParams);

        selTopKIdxQue_.EnQue(selTopKIdxLocal);
    }

    __aicore__ inline void CopyInBlockStatus(int64_t bsIdx, LocalTensor<int32_t>& selBlockStatLocal)
    {
        uint8_t padCnt = topkOneAlign_ - (tiling_->topk + 1);
        DataCopyPadExtParams<int32_t> dataCopyPadParams{true, 0, padCnt, -1};
        uint32_t dstStride = (topkOneSortAlign_ - topkOneAlign_) / (BLOCK_BYTES / sizeof(int32_t));
        int64_t SH = rawSeq_ * tiling_->headnum;
        int64_t curBatchSize = (blkIdx_ * tiling_->mainCoreBsLoopNum + bsIdx) / rawSeq_;
        int64_t curSeq = (blkIdx_ * tiling_->mainCoreBsLoopNum + bsIdx) % rawSeq_;
        int64_t batchOffset = curBatchSize * SH * (tiling_->topk + 1);

        DataCopyExtParams dataCopyParamsSt{
            static_cast<uint16_t>(tiling_->seq * tiling_->headnum),
            static_cast<uint32_t>((tiling_->topk + 1) * sizeof(int32_t)), 0, dstStride, 0};
        DataCopyPad(selBlockStatLocal,
            selKvBlockStatusGm_[batchOffset + curSeq * tiling_->headnum * (tiling_->topk + 1)],
            dataCopyParamsSt,
            dataCopyPadParams);
    }

    __aicore__ inline void CopyInSelKvBlockTable(int64_t bsIdx, LocalTensor<int32_t>& selKvBlockTableLocal)
    {
        int64_t curBatchSize = (blkIdx_ * tiling_->mainCoreBsLoopNum + bsIdx) / rawSeq_;
        int64_t curSeq = (blkIdx_ * tiling_->mainCoreBsLoopNum + bsIdx) % rawSeq_;
        int64_t SH = rawSeq_ * tiling_->headnum;
        int64_t batchOffset = curBatchSize * SH * tiling_->selMaxBlockNum;

        DataCopyPadExtParams<int32_t> dataCopyPadParams{false, 0, 0, 0};
        DataCopyExtParams dataCopyPar{
            static_cast<uint16_t>(1),
            static_cast<uint32_t>(tiling_->seq * tiling_->headnum * tiling_->selMaxBlockNum * sizeof(int32_t)), 0, 0,
            0};
        DataCopyPad(selKvBlockTableLocal,
            selKvBlockTableGm_[batchOffset + curSeq * tiling_->headnum * tiling_->selMaxBlockNum],
            dataCopyPar, dataCopyPadParams);
    }

    __aicore__ inline void SetInvalidBlockStatus(int64_t seqIdx, int64_t hnIdx, int32_t validTopkNum,
        int32_t selActualSeqLen, LocalTensor<int32_t>& selBlockStatLocal)
    {
        LocalTensor<int32_t> tmpBlockStatLocal = selBlockStatLocal[hnIdx * topkOneSortAlign_];
        int64_t duplicateNum = validTopkNum % ONE_REPEAT_SORT_NUM;
        if (duplicateNum > 0) {
            int duplicateIndex = validTopkNum - duplicateNum;
            uint64_t mask0 = UINT64_MAX;
            mask0 = mask0 << duplicateNum;
            mask0 = mask0 & (UINT64_MAX >> ONE_REPEAT_SORT_NUM);
            uint64_t mask[2] = {mask0, 0}; // 2 two mask
            AscendC::Duplicate(tmpBlockStatLocal[duplicateIndex], -1, mask, 1, 1, 8); // 8 means repeat strides
            PipeBarrier<PIPE_V>();
        }

        int32_t dupBegin = CeilAlign(validTopkNum, ONE_REPEAT_SORT_NUM);
        if (topkOneSortAlign_ > dupBegin) {
            AscendC::Duplicate(tmpBlockStatLocal[dupBegin], -1, topkOneSortAlign_ - dupBegin);
            PipeBarrier<PIPE_V>();
        }

        SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);
        tmpBlockStatLocal.SetValue(tiling_->topk, selActualSeqLen);
    }

    __aicore__ inline void PostHandleTopk(int64_t seqIdx, int64_t hnIdx, int32_t maxTopkWritedIdx, int32_t maxValidIdx,
        int32_t maxTopkId, int32_t maxSelectionId, int64_t lastGatherBlockSize, int64_t selBlkTableOffsetLocal,
        LocalTensor<int32_t>& selBlockStatLocal, LocalTensor<int32_t>& selKvBlockTableLocal)
    {
        if (tiling_->selTopKBlockSize <= 1) {
            return;
        }

        if (maxTopkWritedIdx != maxValidIdx) {
            // 1. 交换kv cache数据
            int32_t maxTopkIdBlockSize = maxTopkId == maxSelectionId ? lastGatherBlockSize : tiling_->selTopKBlockSize;
            SwapMaxTopkLast(selBlkTableOffsetLocal, maxTopkWritedIdx,
                            maxValidIdx, maxTopkIdBlockSize, selKvBlockTableLocal);

            // 2. 交换 selection_kv_block_status
            int32_t maxIdx = hnIdx * topkOneSortAlign_ + maxTopkWritedIdx;
            int32_t lastIdx = hnIdx * topkOneSortAlign_ + maxValidIdx;
            int32_t lastTopkId = selBlockStatLocal.GetValue(lastIdx);
            selBlockStatLocal.SetValue(lastIdx, maxTopkId);
            selBlockStatLocal.SetValue(maxIdx, lastTopkId);
        }
    }

    __aicore__ inline void SortTopk(LocalTensor<int32_t>& srcTopkLocal, LocalTensor<uint32_t>& idxLocal,
        LocalTensor<float>& tempTensor, LocalTensor<float>& sortedLocal,
        LocalTensor<int32_t>& sortedTopKLocal, LocalTensor<uint32_t>& sortedTopKIdx, int32_t validNum)
    {
        LocalTensor<float> topkFloatLocal = srcTopkLocal.template ReinterpretCast<float>();
        AscendC::Cast(topkFloatLocal, srcTopkLocal, RoundMode::CAST_ROUND, validNum);
        PipeBarrier<PIPE_V>();

        int64_t duplicateNum = validNum % ONE_REPEAT_SORT_NUM;
        if (duplicateNum > 0) {
            int duplicateIndex = validNum - duplicateNum;
            uint64_t mask0 = UINT64_MAX;
            mask0 = mask0 << duplicateNum;
            mask0 = mask0 & (UINT64_MAX >> ONE_REPEAT_SORT_NUM);
            uint64_t mask[2] = {mask0, 0}; // 2 two mask
            AscendC::Duplicate(topkFloatLocal[duplicateIndex], -1.0f, mask, 1, 1, 8); // 8 means repeat strides
            PipeBarrier<PIPE_V>();
        }

        int64_t sortAlignNum = CeilAlign(validNum, ONE_REPEAT_SORT_NUM);

        LocalTensor<float> concatLocal = topkFloatLocal;
        AscendC::Concat(concatLocal, topkFloatLocal, tempTensor, sortAlignNum / ONE_REPEAT_SORT_NUM);
        PipeBarrier<PIPE_V>();

        AscendC::Sort<float, true>(sortedLocal, concatLocal, idxLocal, tempTensor, sortAlignNum / ONE_REPEAT_SORT_NUM);
        PipeBarrier<PIPE_V>();

        LocalTensor<float> sortedTopKFloat = sortedTopKLocal.template ReinterpretCast<float>();
        AscendC::Extract(sortedTopKFloat, sortedTopKIdx, sortedLocal, sortAlignNum / ONE_REPEAT_SORT_NUM);
        PipeBarrier<PIPE_V>();

        AscendC::Cast(sortedTopKLocal, sortedTopKFloat, RoundMode::CAST_ROUND, sortAlignNum);
        PipeBarrier<PIPE_V>();

        AscendC::Cast(srcTopkLocal, topkFloatLocal, RoundMode::CAST_ROUND, sortAlignNum);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void FindTopkHit(LocalTensor<int32_t>& sortedTopKLocal, LocalTensor<uint32_t>& sortedTopKIdxLocal,
        LocalTensor<int32_t>& sortedStatTopKLocal, LocalTensor<uint32_t>& sortedStatTopKIdxLocal,
        LocalTensor<int32_t>& insertStatusSameSeqLocal, LocalTensor<int32_t>& hitFromSrcSeqLocal,
        bool sameSeq, int32_t stSeqIdx, int32_t stHnIdx, int32_t maxValidCacheId, int32_t& maxHitSameSeqIdx)
    {
        int32_t curIdx = 0;
        int32_t statIdx = 0;
        int32_t curTop = sortedTopKLocal.GetValue(curIdx);
        int32_t statTop = sortedStatTopKLocal.GetValue(statIdx);
        while (curIdx < tiling_->topk && statIdx < tiling_->topk) {
            if (curTop < 0 || statTop < 0) {
                break;
            }

            if (curTop > maxValidCacheId) {
                curIdx++;
                curTop = sortedTopKLocal.GetValue(curIdx);
                continue;
            }

            if (curTop == statTop) {
                // 命中
                int32_t curRealIdx = sortedTopKIdxLocal.GetValue(curIdx);
                int32_t statRealIdx = sortedStatTopKIdxLocal.GetValue(statIdx);
                if (sameSeq) {
                    // 同seq命中必更新
                    insertStatusSameSeqLocal.SetValue(statRealIdx, curRealIdx);
                    hitFromSrcSeqLocal.SetValue(curRealIdx, CUR_SEG_HIT_FLAG);
                    maxHitSameSeqIdx = (statRealIdx > maxHitSameSeqIdx) ? statRealIdx : maxHitSameSeqIdx;
                } else {
                    if (hitFromSrcSeqLocal.GetValue(curRealIdx) == -1) {
                        int32_t statHitIdx = stSeqIdx * tiling_->headnum * tiling_->topk +
                            stHnIdx * tiling_->topk + statRealIdx;
                        hitFromSrcSeqLocal.SetValue(curRealIdx, statHitIdx);
                    }
                }

                curIdx++;
                statIdx++;
                curTop = sortedTopKLocal.GetValue(curIdx);
                statTop = sortedStatTopKLocal.GetValue(statIdx);
            } else if (curTop > statTop) {
                curIdx++;
                curTop = sortedTopKLocal.GetValue(curIdx);
            } else {
                statIdx++;
                statTop = sortedStatTopKLocal.GetValue(statIdx);
            }
        }
    }

    __aicore__ inline void GatherValidTopk(LocalTensor<int32_t>& srcTopkLocal, LocalTensor<int32_t>& tmp1Local,
        LocalTensor<uint32_t>& tmp2Local, LocalTensor<uint32_t>& tmp3Local,
        int32_t maxSelectionId, int32_t& validTopkNum)
    {
        int64_t compareNum = CeilAlign(topkAlign_, static_cast<int64_t>(COMPARE_SCALAR_NUM));
        LocalTensor<float> topkFloatLocal = tmp1Local.template ReinterpretCast<float>();
        AscendC::Cast(topkFloatLocal, srcTopkLocal, RoundMode::CAST_ROUND, topkAlign_);
        PipeBarrier<PIPE_V>();

        LocalTensor<uint8_t> maskTensor = tmp2Local.template ReinterpretCast<uint8_t>();
        AscendC::CompareScalar(maskTensor, topkFloatLocal, -1.0f, CMPMODE::GT, compareNum);
        PipeBarrier<PIPE_V>();

        LocalTensor<uint8_t> maskTensor1 = tmp3Local.template ReinterpretCast<uint8_t>();
        AscendC::CompareScalar(maskTensor1, topkFloatLocal,
                               static_cast<float>(maxSelectionId), CMPMODE::LE, compareNum);
        PipeBarrier<PIPE_V>();

        LocalTensor<uint16_t> maskTensorU16 = maskTensor.template ReinterpretCast<uint16_t>();
        LocalTensor<uint16_t> maskTensor1U16 = maskTensor1.template ReinterpretCast<uint16_t>();
        AscendC::And(maskTensorU16, maskTensorU16, maskTensor1U16, compareNum / COMPARENUM_SIZE);
        PipeBarrier<PIPE_V>();

        uint64_t rsvdCnt = 0;
        GatherMaskParams gatherMaskParams;
        gatherMaskParams.repeatTimes = 1;
        gatherMaskParams.src0BlockStride = 1;
        gatherMaskParams.src0RepeatStride = 8; // 8 blocks
        gatherMaskParams.src1RepeatStride = 0;

        LocalTensor<float> dstFloatLocal = srcTopkLocal.template ReinterpretCast<float>();
        LocalTensor<uint32_t> maskTensorU32 = maskTensor.template ReinterpretCast<uint32_t>();
        AscendC::GatherMask(dstFloatLocal, topkFloatLocal, maskTensorU32, true, topkAlign_, gatherMaskParams, rsvdCnt);
        PipeBarrier<PIPE_V>();

        AscendC::Cast(srcTopkLocal, dstFloatLocal, RoundMode::CAST_ROUND, topkAlign_);
        PipeBarrier<PIPE_V>();

        validTopkNum = static_cast<int32_t>(rsvdCnt);
    }

    __aicore__ inline void GatherInfoGen(int64_t seqIdx, int64_t hnIdx, int32_t maxSelectionId, int32_t maxValidCacheId,
        LocalTensor<int32_t>& selBlockStatLocal, LocalTensor<int32_t>& selTopKIdxLocal,
        int32_t& validTopkNum, int32_t& maxTopkId, int32_t& maxTopkWritedIdx, int32_t& maxHitSameSeqIdx)
    {
        LocalTensor<uint32_t> idxLocal = topkIndicesLocal_.template ReinterpretCast<uint32_t>();

        LocalTensor<int32_t> sortedTopKLocal = sortBuf_;
        LocalTensor<uint32_t> sortedTopKIdxLocal = sortBuf_[topkSortAlign_].template ReinterpretCast<uint32_t>();
        LocalTensor<int32_t> sortedStatTopKLocal = sortBuf_[topkSortAlign_ * SORTED_STAT_TOPK_OFFSET];
        LocalTensor<uint32_t> sortedStatTopKIdxLocal = 
            sortBuf_[topkSortAlign_ * SORTED_STAT_TOPK_IDX_OFFSET].template ReinterpretCast<uint32_t>();

        // 会改变 selTopKIdxLocal 中的值，有效数据都在前面
        // 筛选出有效个数，一是为了减少无效for循环，二是为了统计有效个数判断是否有空位
        GatherValidTopk(selTopKIdxLocal, sortedTopKLocal, sortedTopKIdxLocal, sortedStatTopKIdxLocal,
            maxSelectionId, validTopkNum);

        // 会将 selTopKIdxLocal 中 validTopkNum 后的值都置成-1
        LocalTensor<float> tempTensor = sortedTopKLocal.template ReinterpretCast<float>(); // 复用结果的topk和topkIdx
        LocalTensor<float> sortedLocal = sortBuf_[topkSortAlign_ * 4].template ReinterpretCast<float>();
        SortTopk(selTopKIdxLocal, idxLocal, tempTensor, sortedLocal, sortedTopKLocal, sortedTopKIdxLocal, validTopkNum);

        int64_t sOld = seqIdx;
        for (int64_t hOld = 0; hOld < tiling_->headnum; hOld++) {
            LocalTensor<int32_t> tmpBlockStatus = selBlockStatLocal[hOld * topkOneSortAlign_];
            tempTensor = sortedStatTopKLocal.template ReinterpretCast<float>(); // 复用结果的topk和topkIdx
            sortedLocal = sortBuf_[topkSortAlign_ * SORT_OFFSET].template ReinterpretCast<float>();
            SortTopk(tmpBlockStatus, idxLocal, tempTensor, sortedLocal, sortedStatTopKLocal, sortedStatTopKIdxLocal,
                tiling_->topk);

            SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);
            bool sameSeq = sOld == seqIdx && hOld == hnIdx;
            FindTopkHit(sortedTopKLocal, sortedTopKIdxLocal, sortedStatTopKLocal, sortedStatTopKIdxLocal,
                insertStatusSameSeqLocal_, hitFromSrcSeqLocal_, sameSeq, sOld, hOld, maxValidCacheId, maxHitSameSeqIdx);
        }

        int32_t emptyPosCnt = maxHitSameSeqIdx + 1 - validTopkNum;
        if (emptyPosCnt > 0) {
            // 中间有空位
            int32_t tmpCnt = 0;
            for (int32_t ei = maxHitSameSeqIdx; ei >= 0; ei--) {
                int32_t curIdx = insertStatusSameSeqLocal_.GetValue(ei);
                if (curIdx >= 0) {
                    hitFromSrcSeqLocal_.SetValue(curIdx, seqIdx * tiling_->headnum * tiling_->topk +
                        hnIdx * tiling_->topk + ei);
                    insertStatusSameSeqLocal_.SetValue(ei, -1);
                }
                tmpCnt = tmpCnt + 1;
                if (tmpCnt >= emptyPosCnt) {
                    break;
                }
            }
        }
    }

    __aicore__ inline void ProcessGatherTopK(int64_t bsIdx, int64_t seqIdx, int64_t hnIdx, int64_t curFullKvSeqModify,
        LocalTensor<int32_t>& selBlockStatLocal, LocalTensor<int32_t>& selTopKIdxLocal,
        LocalTensor<int32_t>& selKvBlockTableLocal, LocalTensor<int32_t>& selKvActSeqLocal, int64_t curFullQSeqLen)
    {
        int32_t selActualSeqLen = 0;
        int64_t selBlkTableOffsetLocal = hnIdx * tiling_->selMaxBlockNum; // ub localTensor中的第几行
        int32_t maxSelectionId = CeilDiv(curFullKvSeqModify, tiling_->selTopKBlockSize) - 1;
        int32_t curSeq = (blkIdx_ * tiling_->mainCoreBsLoopNum + bsIdx) % rawSeq_;
        int32_t maxValidCacheId = (maxSelectionId - curSeq - 1) > 0 ? (maxSelectionId - curSeq - 1) : 0;
        int64_t lastGatherBlockSize = curFullKvSeqModify - maxSelectionId * tiling_->selTopKBlockSize;

        int32_t validTopkNum = 0;
        int32_t maxTopkId = -1;
        int32_t maxTopkWritedIdx = -1; // 本seq命中,第一次遍历才会有值
        int32_t maxHitSameSeqIdx = -1; // 本seq中命中,在laststatus中的最大位置
        int32_t totalInsertIdx = 0;

        AscendC::Duplicate(insertStatusSameSeqLocal_, -1, topkSortAlign_);
        AscendC::Duplicate(hitFromSrcSeqLocal_, -1, topkSortAlign_);
        PipeBarrier<PIPE_V>();
        GatherInfoGen(seqIdx, hnIdx, maxSelectionId, maxValidCacheId, selBlockStatLocal, selTopKIdxLocal,
            validTopkNum, maxTopkId, maxTopkWritedIdx, maxHitSameSeqIdx);

        for (int64_t topKIdx = 0; topKIdx < validTopkNum; topKIdx++) {
            int32_t topKId = selTopKIdxLocal.GetValue(topKIdx);
            if (topKId < 0) {
                break;
            }
            // 是否尾块
            int64_t gatherBlockSize = topKId == maxSelectionId ? lastGatherBlockSize : tiling_->selTopKBlockSize;
            selActualSeqLen += gatherBlockSize;

            if (hitFromSrcSeqLocal_.GetValue(topKIdx) == CUR_SEG_HIT_FLAG) {
                continue;
            }

            // 先找到当前可以插入的空位
            int32_t insertIdx = -1;
            for (int32_t insIdx = totalInsertIdx; insIdx < tiling_->topk; insIdx++) {
                if (insertStatusSameSeqLocal_.GetValue(insIdx) < 0) {
                    // 小于0,表示当前位置可以插入
                    insertIdx = insIdx;
                    break;
                }
            }
            ASSERT_MSG(insertIdx >= 0, "!!!!!!!!!!! Find insertIdx failed !!!!!!!!!!!");

            totalInsertIdx = insertIdx + 1;
            int64_t selKvBlkTableIdx = (insertIdx * tiling_->selTopKBlockSize) / tiling_->selKvBlockSize;
            int64_t selKvBlkSizeOffset = (insertIdx * tiling_->selTopKBlockSize) % tiling_->selKvBlockSize;
            int32_t selKvBlockNumIdx = selKvBlockTableLocal.GetValue(selBlkTableOffsetLocal + selKvBlkTableIdx);
            int64_t selKRopeAddr = selKvBlockNumIdx * tiling_->selKvBlockSize * tiling_->kRopeDim +
                                   selKvBlkSizeOffset * tiling_->kRopeDim;
            int64_t selKvCacheAddr = selKvBlockNumIdx * tiling_->selKvBlockSize * tiling_->kvCacheDim +
                                     selKvBlkSizeOffset * tiling_->kvCacheDim;

            if (hitFromSrcSeqLocal_.GetValue(topKIdx) == -1) {
                CopyFromFullKv(bsIdx, topKId, gatherBlockSize, selKRopeAddr, selKvCacheAddr);
            } else {
                CopyFromSelKv(bsIdx, hitFromSrcSeqLocal_.GetValue(topKIdx), selKvBlockTableLocal, gatherBlockSize,
                    selKRopeAddr, selKvCacheAddr);
            }

            // 更新 selection_kv_block_status
            int32_t statIdx = hnIdx * topkOneSortAlign_ + insertIdx;
            selBlockStatLocal.SetValue(statIdx, topKId);
            if (topKId == maxTopkId) {
                maxTopkWritedIdx = insertIdx;
            }
        }

        // 清理无效的block_status位置及设置seqlen
        SetInvalidBlockStatus(seqIdx, hnIdx, validTopkNum, selActualSeqLen, selBlockStatLocal);

        // 后处理,将max topk id调整至最后
        PostHandleTopk(seqIdx, hnIdx, maxTopkWritedIdx, validTopkNum - 1, maxTopkId, maxSelectionId,
            lastGatherBlockSize, selBlkTableOffsetLocal, selBlockStatLocal, selKvBlockTableLocal);

        SetSelectionKvActualSeqLen(seqIdx, hnIdx, selActualSeqLen, selKvActSeqLocal);
    }

    __aicore__ inline void CopyFromFullKv(int64_t bsIdx, int64_t topKId, int64_t gatherBlockSize,
        int64_t selKRopeAddr, int64_t selKvCacheAddr)
    {
        // 从topk中获取索引，再根据full_kv_block_table中的地址，计算full kv cache的偏移地址
        int64_t kvBlockTableIdx = (topKId * tiling_->selTopKBlockSize) / tiling_->fullKvBlockSize;
        int64_t kvBlockSizeOffset = (topKId * tiling_->selTopKBlockSize) % tiling_->fullKvBlockSize;
        int64_t curBatchSize = (blkIdx_ * tiling_->mainCoreBsLoopNum + bsIdx) / rawSeq_;
        int32_t kvBlockNumIdx = fullKvBlockTableGm_.GetValue(curBatchSize * tiling_->fullMaxBlockNum + kvBlockTableIdx);
        ASSERT_MSG(kvBlockNumIdx >= 0, "kvBlockTableIdx:%ld should be greater than 0", kvBlockNumIdx);

        int64_t fullKRopeAddr =
            kvBlockNumIdx * tiling_->fullKvBlockSize * tiling_->kRopeDim + kvBlockSizeOffset * tiling_->kRopeDim;
        int64_t fullKvCacheAddr =
            kvBlockNumIdx * tiling_->fullKvBlockSize * tiling_->kvCacheDim + kvBlockSizeOffset * tiling_->kvCacheDim;

        LocalTensor<T> inTensor = kvCacheQue_.AllocTensor<T>();
        DataCopyPadExtParams<T> dataCopyPadParams{false, 0, 0, 0};
        DataCopyExtParams dataCopyParams{
            static_cast<uint16_t>(1), static_cast<uint32_t>(gatherBlockSize * tiling_->kvCacheDim * sizeof(T)), 0, 0,
            0};

        DataCopyPad(inTensor, fullKvCacheGm_[fullKvCacheAddr], dataCopyParams, dataCopyPadParams);
        
        DataCopyExtParams dataCopyParams1{
            static_cast<uint16_t>(1), static_cast<uint32_t>(gatherBlockSize * tiling_->kRopeDim * sizeof(T)), 0, 0, 0};
        if (tiling_->ifQuant != 1) {
            DataCopyPad(inTensor[kRopeUbOffset_], fullKRopeGm_[fullKRopeAddr], dataCopyParams1, dataCopyPadParams);
        }

        kvCacheQue_.EnQue(inTensor);
        inTensor = kvCacheQue_.DeQue<T>();
        
        DataCopyPad(selKvCacheGm_[selKvCacheAddr], inTensor, dataCopyParams);
        if (tiling_->ifQuant != 1) {
            DataCopyPad(selKRopeGm_[selKRopeAddr], inTensor[kRopeUbOffset_], dataCopyParams1);
        }
        
        kvCacheQue_.FreeTensor(inTensor);
    }

    __aicore__ inline void CopyFromSelKv(int64_t bsIdx, int64_t hitFromValue,
                                         LocalTensor<int32_t>& selKvBlockTableLocal,
                                         int64_t gatherBlockSize, int64_t selKRopeAddr, int64_t selKvCacheAddr)
    {
        int32_t gmSrcSeq = hitFromValue / tiling_->topk;
        int32_t gmSrcHdn = 0;
        int32_t gmSrcIdx = hitFromValue - gmSrcSeq * tiling_->topk;

        int64_t selBlkTableIdx = gmSrcSeq * tiling_->headnum * tiling_->selMaxBlockNum +
                                    gmSrcHdn * tiling_->selMaxBlockNum;
        int64_t selKvBlkTableIdx = (gmSrcIdx * tiling_->selTopKBlockSize) / tiling_->selKvBlockSize;
        int64_t selKvBlkSizeOffset = (gmSrcIdx * tiling_->selTopKBlockSize) % tiling_->selKvBlockSize;
        int32_t selKvBlockNumIdx = selKvBlockTableLocal.GetValue(selBlkTableIdx + selKvBlkTableIdx);
        int64_t srcKRopeAddr = selKvBlockNumIdx * tiling_->selKvBlockSize * tiling_->kRopeDim +
                                selKvBlkSizeOffset * tiling_->kRopeDim;
        int64_t srcKvCacheAddr = selKvBlockNumIdx * tiling_->selKvBlockSize * tiling_->kvCacheDim +
                                    selKvBlkSizeOffset * tiling_->kvCacheDim;

        LocalTensor<T> inTensor = kvCacheQue_.AllocTensor<T>();
        DataCopyPadExtParams<T> dataCopyPadParams{false, 0, 0, 0};
        DataCopyExtParams dataCopyParams{
            static_cast<uint16_t>(1), static_cast<uint32_t>(gatherBlockSize * tiling_->kvCacheDim * sizeof(T)), 0, 0,
            0};

        DataCopyPad(inTensor, selKvCacheGm_[srcKvCacheAddr], dataCopyParams, dataCopyPadParams);
        
        DataCopyExtParams dataCopyParams1{
            static_cast<uint16_t>(1), static_cast<uint32_t>(gatherBlockSize * tiling_->kRopeDim * sizeof(T)), 0, 0, 0};

        if (tiling_->ifQuant != 1) {
            DataCopyPad(inTensor[kRopeUbOffset_], selKRopeGm_[srcKRopeAddr], dataCopyParams1, dataCopyPadParams);
        }
        
        kvCacheQue_.EnQue(inTensor);
        inTensor = kvCacheQue_.DeQue<T>();
        DataCopyPad(selKvCacheGm_[selKvCacheAddr], inTensor, dataCopyParams);

        if (tiling_->ifQuant != 1) {
            DataCopyPad(selKRopeGm_[selKRopeAddr], inTensor[kRopeUbOffset_], dataCopyParams1);
        }
        kvCacheQue_.FreeTensor(inTensor);
    }

    __aicore__ inline void SwapMaxTopkLast(int64_t selBlkTableOffsetLocal, int32_t maxTopkWritedIdx,
        int32_t lastTopKIdx, int32_t maxTopkIdBlockSize, LocalTensor<int32_t>& selKvBlockTableLocal)
    {
        int64_t selKvBlkTableIdx = (maxTopkWritedIdx * tiling_->selTopKBlockSize) / tiling_->selKvBlockSize;
        int64_t selKvBlkSizeOffset = (maxTopkWritedIdx * tiling_->selTopKBlockSize) % tiling_->selKvBlockSize;
        int32_t selKvBlockNumIdx = selKvBlockTableLocal.GetValue(selBlkTableOffsetLocal + selKvBlkTableIdx);
        int64_t maxKRopeAddr = selKvBlockNumIdx * tiling_->selKvBlockSize * tiling_->kRopeDim +
                                selKvBlkSizeOffset * tiling_->kRopeDim;
        int64_t maxKvCacheAddr = selKvBlockNumIdx * tiling_->selKvBlockSize * tiling_->kvCacheDim +
                                    selKvBlkSizeOffset * tiling_->kvCacheDim;

        selKvBlkTableIdx = (lastTopKIdx * tiling_->selTopKBlockSize) / tiling_->selKvBlockSize;
        selKvBlkSizeOffset = (lastTopKIdx * tiling_->selTopKBlockSize) % tiling_->selKvBlockSize;
        selKvBlockNumIdx = selKvBlockTableLocal.GetValue(selBlkTableOffsetLocal + selKvBlkTableIdx);
        int64_t lastKRopeAddr = selKvBlockNumIdx * tiling_->selKvBlockSize * tiling_->kRopeDim +
                                selKvBlkSizeOffset * tiling_->kRopeDim;
        int64_t lastKvCacheAddr = selKvBlockNumIdx * tiling_->selKvBlockSize * tiling_->kvCacheDim +
                                    selKvBlkSizeOffset * tiling_->kvCacheDim;

        LocalTensor<T> maxTensor = kvCacheQue_.AllocTensor<T>();
        LocalTensor<T> lastTensor = kvCacheQue_.AllocTensor<T>();
        DataCopyPadExtParams<T> dataCopyPadParams{false, 0, 0, 0};

        DataCopyExtParams maxKropeParam{
            static_cast<uint16_t>(1), static_cast<uint32_t>(maxTopkIdBlockSize * tiling_->kRopeDim * sizeof(T)), 0, 0,
            0};
        DataCopyExtParams maxKvCacheParam{
            static_cast<uint16_t>(1), static_cast<uint32_t>(maxTopkIdBlockSize * tiling_->kvCacheDim * sizeof(T)), 0, 0,
            0};
        DataCopyExtParams lastKropeParam{
            static_cast<uint16_t>(1), static_cast<uint32_t>(tiling_->selTopKBlockSize * tiling_->kRopeDim * sizeof(T)),
            0, 0, 0};
        DataCopyExtParams lastKvCacheParam{
            static_cast<uint16_t>(1),
            static_cast<uint32_t>(tiling_->selTopKBlockSize * tiling_->kvCacheDim * sizeof(T)),
            0, 0, 0};
        
        DataCopyPad(maxTensor, selKvCacheGm_[maxKvCacheAddr], maxKvCacheParam, dataCopyPadParams);
        DataCopyPad(lastTensor, selKvCacheGm_[lastKvCacheAddr], lastKvCacheParam, dataCopyPadParams);

        if (tiling_->ifQuant != 1) {
            DataCopyPad(maxTensor[kRopeUbOffset_], selKRopeGm_[maxKRopeAddr], maxKropeParam, dataCopyPadParams);
            DataCopyPad(lastTensor[kRopeUbOffset_], selKRopeGm_[lastKRopeAddr], lastKropeParam, dataCopyPadParams);
        }

        SetWaitFlag<HardEvent::MTE2_MTE3>(HardEvent::MTE2_MTE3);

        DataCopyPad(selKvCacheGm_[maxKvCacheAddr], lastTensor, lastKvCacheParam);
        DataCopyPad(selKvCacheGm_[lastKvCacheAddr], maxTensor, maxKvCacheParam);

        if (tiling_->ifQuant != 1) {
            DataCopyPad(selKRopeGm_[maxKRopeAddr], lastTensor[kRopeUbOffset_], lastKropeParam);
            DataCopyPad(selKRopeGm_[lastKRopeAddr], maxTensor[kRopeUbOffset_], maxKropeParam);
        }

        kvCacheQue_.FreeTensor(lastTensor);
        kvCacheQue_.FreeTensor(maxTensor);
    }

    __aicore__ inline void SetSelectionKvActualSeqLen(
        int64_t seqIdx, int64_t hnIdx, int32_t selActualSeqLen, LocalTensor<int32_t>& selKvActSeqLocal)
    {
        int64_t seqLenIdx = hnIdx;
        selKvActSeqLocal.SetValue(seqLenIdx, selActualSeqLen);
    }

    __aicore__ inline void CopyOutSelKvBlockTableAndSeqLen(int64_t bsIdx, LocalTensor<int32_t>& selKvActSeqLocal,
        LocalTensor<int32_t>& selBlockStatLocal)
    {
        int64_t curBatchSize = (blkIdx_ * tiling_->mainCoreBsLoopNum + bsIdx) / rawSeq_;
        int64_t curSeq = (blkIdx_ * tiling_->mainCoreBsLoopNum + bsIdx) % rawSeq_;

        DataCopyExtParams dataCopyParSeqLen{
            static_cast<uint16_t>(1), static_cast<uint32_t>(tiling_->seq * tiling_->headnum * sizeof(int32_t)), 0, 0,
            0};
        DataCopyPad(selKvActualSeqGm_[(curBatchSize * rawSeq_) * tiling_->headnum + curSeq * tiling_->headnum],
                    selKvActSeqLocal, dataCopyParSeqLen);

        int32_t statSH = rawSeq_ * tiling_->headnum;
        int64_t SH = rawSeq_ * tiling_->headnum;
        int64_t BsiSH = tiling_->mainCoreBsLoopNum * SH;
        uint32_t srcStride = (topkOneSortAlign_ - topkOneAlign_) / (BLOCK_BYTES / sizeof(int32_t));
        int64_t batchOffset = curBatchSize * SH * (tiling_->topk + 1);

        DataCopyExtParams dataCopyParBlkStat{
            static_cast<uint16_t>(tiling_->headnum),
            static_cast<uint32_t>((tiling_->topk + 1) * sizeof(int32_t)), srcStride, 0, 0};
        DataCopyPad(selKvBlockStatusGm_[batchOffset + curSeq * tiling_->headnum * (tiling_->topk + 1)],
                    selBlockStatLocal, dataCopyParBlkStat);
    }

    template <typename U>
    __aicore__ inline U CeilAlign(U a, U b)
    {
        return (a + b - 1) / b * b;
    }

    template <typename U>
    __aicore__ inline U CeilDiv(U a, U b)
    {
        if (b == 0) {
            return a;
        }
        return (a + b - 1) / b;
    }

private:
    TPipe* pipe_;
    const GatherSelectionKvCacheTilingData* tiling_;

    int32_t blkIdx_ = -1;
    int64_t bsLoopNum_ = 0;
    int64_t topkAlign_ = 0; // align to block
    int64_t topkSortAlign_ = 0; // align to 32 num
    int64_t topkOneAlign_ = 0; // k+1 align
    int64_t topkOneSortAlign_ = 0;
    int64_t rawSeq_ = 0;

    int32_t kRopeUbOffset_ = 0;
    int32_t selTopKIdxUb_ = 0;
    int32_t selBlockStatUb_ = 0;
    int32_t selKvBlockTableUb_ = 0;
    int32_t selKvActSeqUb_ = 0;

    constexpr static int32_t BLOCK_BYTES = 32;

    GlobalTensor<T> selKRopeGm_;
    GlobalTensor<T> selKvCacheGm_;
    GlobalTensor<int32_t> selKvBlockTableGm_;
    GlobalTensor<int32_t> selKvBlockStatusGm_;
    GlobalTensor<int32_t> selTopKIndicesGm_;
    GlobalTensor<T> fullKRopeGm_;
    GlobalTensor<T> fullKvCacheGm_;
    GlobalTensor<int32_t> fullKvBlockTableGm_;
    GlobalTensor<int32_t> fullKvActualSeqGm_;
    GlobalTensor<int32_t> fullQActualSeqGm_;
    GlobalTensor<int32_t> selKvActualSeqGm_;

    LocalTensor<int32_t> topkIndicesLocal_;
    LocalTensor<int32_t> insertStatusSameSeqLocal_;
    LocalTensor<int32_t> hitFromSrcSeqLocal_;
    LocalTensor<int32_t> sortBuf_;

    TQue<QuePosition::VECIN, 1> selTopKIdxQue_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> kvCacheQue_;
    TBuf<QuePosition::VECCALC> workBuf_;

    enum class LAYOUT : uint32_t {
        BSND = 0,
        TND = 1
    };
};

} // namespace GatherSelectionKvCacheNs
#endif // GATHER_SELECTION_KV_CACHE_SPLIT_BS_REUSE_VEC_H