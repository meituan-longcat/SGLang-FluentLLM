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
 * \file gather_selection_kv_cache_split_bs_reuse.h
 * \brief
 */
#ifndef GATHER_SELECTION_KV_CACHE_SPLIT_BS_REUSE_H
#define GATHER_SELECTION_KV_CACHE_SPLIT_BS_REUSE_H

#include "kernel_operator.h"

namespace GatherSelectionKvCacheNs {
using namespace AscendC;

constexpr int32_t MAX_TOP_K_NUM = 32;
constexpr int32_t CUR_SEG_HIT_FLAG = -10000;

template <typename T>
class GatherSelectionKvCacheSplitBsReuse {
public:
    __aicore__ inline GatherSelectionKvCacheSplitBsReuse(TPipe* pipe, const GatherSelectionKvCacheTilingData* tiling)
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

        int64_t SH = tiling_->seq * tiling_->headnum;

        // 64*512*2=64KB 64*64*2=8KB
        kRopeUbOffset_ = tiling_->kvCacheUbSize / sizeof(T);
        pipe_->InitBuffer(kvCacheQue_, tiling_->buffNum, tiling_->kvCacheUbSize + tiling_->kRopeUbSize);

        selTopKIdxUb_ = CeilAlign(
            static_cast<int64_t>(SH * tiling_->topk * sizeof(int32_t)),
            static_cast<int64_t>(BLOCK_BYTES));
        pipe_->InitBuffer(selTopKIdxQue_, 1, selTopKIdxUb_);

        selKvBlockTableUb_ = CeilAlign(
            static_cast<int64_t>(SH * tiling_->selMaxBlockNum * sizeof(int32_t)),
            static_cast<int64_t>(BLOCK_BYTES));
        selKvActSeqUb_ = CeilAlign(
            static_cast<int64_t>(SH * sizeof(int32_t)), static_cast<int64_t>(BLOCK_BYTES));
        selBlockStatUb_ = CeilAlign(
            static_cast<int64_t>(SH * (tiling_->topk + 1) * sizeof(int32_t)),
            static_cast<int64_t>(BLOCK_BYTES));
        pipe_->InitBuffer(workBuf_, selKvBlockTableUb_ + selKvActSeqUb_ + selBlockStatUb_);

        bsLoopNum_ = (blkIdx_ == tiling_->usedCoreNum - 1) ? tiling_->tailCoreBsLoopNum : tiling_->mainCoreBsLoopNum;

        int64_t BsiSH = tiling_->mainCoreBsLoopNum * SH;
        // [s_block_num, block_size, k_rope] // 64
        selKRopeGm_.SetGlobalBuffer((__gm__ T*)selection_k_rope);
        // [s_block_num, block_size, kv_cache] // 512
        selKvCacheGm_.SetGlobalBuffer((__gm__ T*)selection_kv_cache);
        // [batchsize*seq*headnum, s_maxblocknum]  初始全-1，表示全空
        selKvBlockTableGm_.SetGlobalBuffer(
            (__gm__ int32_t*)selection_kv_block_table + blkIdx_ * BsiSH * tiling_->selMaxBlockNum);
        // [batchsize, seq, headnum, topk+1] 初始全-1，表示全空
        selKvBlockStatusGm_.SetGlobalBuffer(
            (__gm__ int32_t*)selection_kv_block_status + blkIdx_ * BsiSH * (tiling_->topk + 1));
        // [batchsize, seq, headnum, topk] token粒度=groupid * selection_topk_block_size
        selTopKIndicesGm_.SetGlobalBuffer((__gm__ int32_t*)selection_topk_indices + blkIdx_ * BsiSH * tiling_->topk);
        // [f_block_num, block_size, k_rope]
        fullKRopeGm_.SetGlobalBuffer((__gm__ T*)full_k_rope);
        // [f_block_num, block_size, kv_cache]
        fullKvCacheGm_.SetGlobalBuffer((__gm__ T*)full_kv_cache);
        // [batchsize, f_maxblocknum]
        fullKvBlockTableGm_.SetGlobalBuffer(
            (__gm__ int32_t*)full_kv_block_table + blkIdx_ * tiling_->mainCoreBsLoopNum * tiling_->fullMaxBlockNum);
        // [batchsize] 每个batch实际的seq长度
        fullKvActualSeqGm_.SetGlobalBuffer((__gm__ int32_t*)full_kv_actual_seq + blkIdx_ * tiling_->mainCoreBsLoopNum);
        // [batchsize] 每个batch实际的seq长度   用来算MTP场景不同T下面的kv_actual_seq
        fullQActualSeqGm_.SetGlobalBuffer((__gm__ int32_t*)full_q_actual_seq + blkIdx_ * tiling_->mainCoreBsLoopNum);
        // [batchsize*seq*headnum]
        selKvActualSeqGm_.SetGlobalBuffer((__gm__ int32_t*)selection_kv_actual_seq + blkIdx_ * BsiSH);
    }

    __aicore__ inline void Process()
    {
        if (blkIdx_ >= tiling_->usedCoreNum) {
            return;
        }

        LocalTensor<int32_t> selKvBlockTableLocal = workBuf_.Get<int32_t>();
        LocalTensor<int32_t> selKvActSeqLocal = selKvBlockTableLocal[selKvBlockTableUb_ / sizeof(int32_t)];
        LocalTensor<int32_t> selBlockStatLocal = selKvActSeqLocal[selKvActSeqUb_ / sizeof(int32_t)];

        int64_t curFullKvSeqLen = 0;
        int64_t curFullQSeqLen = 0;

        for (int64_t bsIdx = 0; bsIdx < bsLoopNum_; bsIdx++) {
            curFullKvSeqLen = fullKvActualSeqGm_.GetValue(bsIdx);
            if (curFullKvSeqLen <= 0) {
                continue;
            }
            curFullQSeqLen = fullQActualSeqGm_.GetValue(bsIdx);
            ASSERT_MSG(curFullQSeqLen <= tiling_->seq, "curFullQSeqLen:%ld cannot be greater than seq:%ld",
                curFullQSeqLen, tiling_->seq);

            CopyInTopKIndices(bsIdx);
            LocalTensor<int32_t> selTopKIdxLocal = selTopKIdxQue_.DeQue<int32_t>();

            SetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);
            CopyInBlockStatus(bsIdx, selBlockStatLocal);

            CopyInSelKvBlockTable(bsIdx, selKvBlockTableLocal);

            SetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);

            for (int64_t seqIdx = 0; seqIdx < tiling_->seq; seqIdx++) {
                int64_t curFullKvSeqModify = seqIdx < curFullQSeqLen ? (curFullKvSeqLen - (curFullQSeqLen - 1 - seqIdx))
                                                                     : curFullKvSeqLen;
                if (curFullKvSeqModify <= 0) {
                    continue;
                }

                for (int64_t hnIdx = 0; hnIdx < tiling_->headnum; hnIdx++) {
                    ProcessGatherTopK(bsIdx, seqIdx, hnIdx, curFullKvSeqModify, selBlockStatLocal, selTopKIdxLocal,
                        selKvBlockTableLocal, selKvActSeqLocal);
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

    __aicore__ inline void CopyInTopKIndices(int64_t bsIdx)
    {
        LocalTensor<int32_t> selTopKIdxLocal = selTopKIdxQue_.AllocTensor<int32_t>();
        int32_t SHT = tiling_->seq * tiling_->headnum * tiling_->topk;
        DataCopyPadExtParams<int32_t> dataCopyPadParams{false, 0, 0, 0};
        DataCopyExtParams dataCopyParams{
            static_cast<uint16_t>(1), static_cast<uint32_t>(SHT * sizeof(int32_t)), 0, 0, 0};
        DataCopyPad(selTopKIdxLocal, selTopKIndicesGm_[bsIdx * SHT], dataCopyParams, dataCopyPadParams);

        selTopKIdxQue_.EnQue(selTopKIdxLocal);
    }

    __aicore__ inline void CopyInBlockStatus(int64_t bsIdx, LocalTensor<int32_t>& selBlockStatLocal)
    {
        DataCopyPadExtParams<int32_t> dataCopyPadParams{false, 0, 0, 0};
        int32_t statSHT = tiling_->seq * tiling_->headnum * (tiling_->topk + 1);
        DataCopyExtParams dataCopyParamsSt{
            static_cast<uint16_t>(1), static_cast<uint32_t>(statSHT * sizeof(int32_t)), 0, 0, 0};
        DataCopyPad(selBlockStatLocal, selKvBlockStatusGm_[bsIdx * statSHT], dataCopyParamsSt, dataCopyPadParams);
    }

    __aicore__ inline void CopyInSelKvBlockTable(int64_t bsIdx, LocalTensor<int32_t>& selKvBlockTableLocal)
    {
        DataCopyPadExtParams<int32_t> dataCopyPadParams{false, 0, 0, 0};
        DataCopyExtParams dataCopyPar{
            static_cast<uint16_t>(1),
            static_cast<uint32_t>(tiling_->seq * tiling_->headnum * tiling_->selMaxBlockNum * sizeof(int32_t)), 0, 0,
            0};
        DataCopyPad(selKvBlockTableLocal,
                    selKvBlockTableGm_[bsIdx * tiling_->seq * tiling_->headnum * tiling_->selMaxBlockNum],
                    dataCopyPar, dataCopyPadParams);
    }

    __aicore__ inline void SetInvalidBlockStatus(int64_t seqIdx, int64_t hnIdx, int32_t validTopkNum,
        int32_t selActualSeqLen, LocalTensor<int32_t>& selBlockStatLocal)
    {
        for (int32_t negIdx = validTopkNum; negIdx < tiling_->topk; negIdx++) {
            int32_t statIdx = seqIdx * tiling_->headnum * (tiling_->topk + 1) +
                                hnIdx * (tiling_->topk + 1) + negIdx;
            selBlockStatLocal.SetValue(statIdx, -1);
        }

        int32_t actSeqBlkStIdx = seqIdx * tiling_->headnum * (tiling_->topk + 1) +
            hnIdx * (tiling_->topk + 1) + tiling_->topk;
        selBlockStatLocal.SetValue(actSeqBlkStIdx, selActualSeqLen);
    }

    __aicore__ inline void PostHandleTopk(int64_t seqIdx, int64_t hnIdx, int32_t maxTopkWritedIdx, int32_t maxValidIdx,
        int32_t maxTopkId, int32_t maxSelectionId, int64_t lastGatherBlockSize, int64_t selBlkTableOffsetLocal,
        LocalTensor<int32_t>& selBlockStatLocal, LocalTensor<int32_t>& selKvBlockTableLocal)
    {
        if (maxTopkWritedIdx != maxValidIdx) {
            // 1. 交换kv cache数据
            int32_t maxTopkIdBlockSize = maxTopkId == maxSelectionId ? lastGatherBlockSize : tiling_->selTopKBlockSize;
            SwapMaxTopkLast(selBlkTableOffsetLocal, maxTopkWritedIdx,
                            maxValidIdx, maxTopkIdBlockSize, selKvBlockTableLocal);

            // 2. 交换 selection_kv_block_status
            int32_t maxIdx = seqIdx * tiling_->headnum * (tiling_->topk + 1) +
                                hnIdx * (tiling_->topk + 1) + maxTopkWritedIdx;
            int32_t lastIdx = seqIdx * tiling_->headnum * (tiling_->topk + 1) +
                                hnIdx * (tiling_->topk + 1) + maxValidIdx;
            int32_t lastTopkId = selBlockStatLocal.GetValue(lastIdx);
            selBlockStatLocal.SetValue(lastIdx, maxTopkId);
            selBlockStatLocal.SetValue(maxIdx, lastTopkId);
        }
    }

    __aicore__ inline void ProcessGatherTopK(int64_t bsIdx, int64_t seqIdx, int64_t hnIdx, int64_t curFullKvSeqModify,
        LocalTensor<int32_t>& selBlockStatLocal, LocalTensor<int32_t>& selTopKIdxLocal,
        LocalTensor<int32_t>& selKvBlockTableLocal, LocalTensor<int32_t>& selKvActSeqLocal)
    {
        int32_t selActualSeqLen = 0;
        int64_t selBlkTableOffsetLocal = seqIdx * tiling_->headnum * tiling_->selMaxBlockNum +
                                            hnIdx * tiling_->selMaxBlockNum; // ub localTensor中的第几行
        int32_t maxSelectionId = CeilDiv(curFullKvSeqModify, tiling_->selTopKBlockSize) - 1;
        int64_t lastGatherBlockSize = curFullKvSeqModify - maxSelectionId * tiling_->selTopKBlockSize;
        int32_t insertStatusSameSeq[MAX_TOP_K_NUM] = {-1, -1, -1, -1, -1, -1, -1, -1,
                                                      -1, -1, -1, -1, -1, -1, -1, -1,
                                                      -1, -1, -1, -1, -1, -1, -1, -1,
                                                      -1, -1, -1, -1, -1, -1, -1, -1};
        int32_t hitFromSrcSeq[MAX_TOP_K_NUM] = {-1, -1, -1, -1, -1, -1, -1, -1,
                                                -1, -1, -1, -1, -1, -1, -1, -1,
                                                -1, -1, -1, -1, -1, -1, -1, -1,
                                                -1, -1, -1, -1, -1, -1, -1, -1};
        int32_t validTopkNum = 0;
        int32_t maxTopkId = -1;
        int32_t maxTopkWritedIdx = -1; // 本seq命中,第一次遍历才会有值
        int32_t maxHitSameSeqIdx = -1; // 本seq中命中,在laststatus中的最大位置
        int32_t totalInsertIdx = 0;

        gatherInfoGen(seqIdx, hnIdx, maxSelectionId, selBlockStatLocal, selTopKIdxLocal,
            validTopkNum, maxTopkId, maxTopkWritedIdx, maxHitSameSeqIdx, insertStatusSameSeq, hitFromSrcSeq);

        for (int64_t topKIdx = 0; topKIdx < tiling_->topk; topKIdx++) {
            int32_t topKId = selTopKIdxLocal.GetValue(
                seqIdx * tiling_->headnum * tiling_->topk + hnIdx * tiling_->topk + topKIdx);
            if (topKId < 0) {
                break;
            }
            if (topKId > maxSelectionId) {
                continue;
            }

            // 是否尾块
            int64_t gatherBlockSize = topKId == maxSelectionId ? lastGatherBlockSize : tiling_->selTopKBlockSize;
            selActualSeqLen += gatherBlockSize;

            if (hitFromSrcSeq[topKIdx] == CUR_SEG_HIT_FLAG) {
                continue;
            }

            // 先找到当前可以插入的空位
            int32_t insertIdx = -1;
            for (int32_t insIdx = totalInsertIdx; insIdx < tiling_->topk; insIdx++) {
                if (insertStatusSameSeq[insIdx] < 0) {
                    insertIdx = insIdx;
                    break;
                }
            }
            ASSERT_MSG(insertIdx >= 0, "Find insertIdx failed!");

            totalInsertIdx = insertIdx + 1;

            int64_t selKvBlkTableIdx = (insertIdx * tiling_->selTopKBlockSize) / tiling_->selKvBlockSize;
            int64_t selKvBlkSizeOffset = (insertIdx * tiling_->selTopKBlockSize) % tiling_->selKvBlockSize;
            int32_t selKvBlockNumIdx = selKvBlockTableLocal.GetValue(selBlkTableOffsetLocal + selKvBlkTableIdx);
            int64_t selKRopeAddr = selKvBlockNumIdx * tiling_->selKvBlockSize * tiling_->kRopeDim +
                                   selKvBlkSizeOffset * tiling_->kRopeDim;
            int64_t selKvCacheAddr = selKvBlockNumIdx * tiling_->selKvBlockSize * tiling_->kvCacheDim +
                                     selKvBlkSizeOffset * tiling_->kvCacheDim;

            if (hitFromSrcSeq[topKIdx] == -1) {
                CopyFromFullKv(bsIdx, topKId, gatherBlockSize, selKRopeAddr, selKvCacheAddr);
            } else {
                CopyFromSelKv(bsIdx, hitFromSrcSeq[topKIdx], selKvBlockTableLocal, gatherBlockSize,
                    selKRopeAddr, selKvCacheAddr);
            }

            // 更新 selection_kv_block_status
            int32_t statIdx = seqIdx * tiling_->headnum * (tiling_->topk + 1) +
                                hnIdx * (tiling_->topk + 1) + insertIdx;
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

    __aicore__ inline void gatherInfoGen(int64_t seqIdx, int64_t hnIdx, int32_t maxSelectionId,
        LocalTensor<int32_t>& selBlockStatLocal, LocalTensor<int32_t>& selTopKIdxLocal,
        int32_t& validTopkNum, int32_t& maxTopkId, int32_t& maxTopkWritedIdx, int32_t& maxHitSameSeqIdx,
        int32_t *insertStatusSameSeq, int32_t *hitFromSrcSeq)
    {
        for (int64_t topKIdx = 0; topKIdx < tiling_->topk; topKIdx++) {
            int32_t topKId = selTopKIdxLocal.GetValue(
                seqIdx * tiling_->headnum * tiling_->topk + hnIdx * tiling_->topk + topKIdx);
            if (topKId < 0) {
                break;
            }
            if (topKId > maxSelectionId) {
                continue;
            }

            if (topKId >= maxTopkId) {
                maxTopkWritedIdx = -1;
            }

            validTopkNum += 1;
            bool maybeMaxTopkId = maxTopkId > topKId ? false : true;
            maxTopkId = maxTopkId > topKId ? maxTopkId : topKId;

            gatherInfoFromBlockStatus(topKIdx, topKId, maybeMaxTopkId, seqIdx, selBlockStatLocal, maxTopkWritedIdx,
                maxHitSameSeqIdx, insertStatusSameSeq, hitFromSrcSeq);
        }

        int32_t emptyPosCnt = maxHitSameSeqIdx + 1 - validTopkNum;
        if (emptyPosCnt > 0) {
            // 中间有空位
            int32_t tmpCnt = 0;
            for (int32_t ei = maxHitSameSeqIdx; ei >= 0; ei--) {
                if (insertStatusSameSeq[ei] >= 0) {
                    hitFromSrcSeq[insertStatusSameSeq[ei]] = seqIdx * tiling_->headnum * tiling_->topk +
                                                             hnIdx * tiling_->topk + ei;
                    insertStatusSameSeq[ei] = -1;
                }
                tmpCnt = tmpCnt + 1;
                if (tmpCnt >= emptyPosCnt) {
                    break;
                }
            }
        }
    }

    __aicore__ inline void gatherInfoFromBlockStatus(int64_t topKIdx, int32_t topKId, bool maybeMaxTopkId,
        int64_t seqIdx, LocalTensor<int32_t>& selBlockStatLocal, int32_t& maxTopkWritedIdx, int32_t& maxHitSameSeqIdx,
        int32_t *insertStatusSameSeq, int32_t *hitFromSrcSeq)
    {
        for (int64_t sOld = 0; sOld < tiling_->seq; sOld++) {
            bool isHit = false;
            bool isHitSameSeq = false;
            for (int64_t hOld = 0; hOld < tiling_->headnum; hOld++) {
                int64_t actualSeqLenOld = selBlockStatLocal.GetValue(sOld * tiling_->headnum * (tiling_->topk + 1) +
                    hOld * (tiling_->topk + 1) + tiling_->topk);
                if (actualSeqLenOld <= 0) {
                    continue;
                }
                for (int64_t kOld = 0; kOld < tiling_->topk; kOld++) {
                    int32_t topKIdHit = selBlockStatLocal.GetValue(
                        sOld * tiling_->headnum * (tiling_->topk + 1) + hOld * (tiling_->topk + 1) + kOld);
                    if (topKIdHit != topKId) {
                        continue;
                    }
                    // 命中
                    bool isInvalidBlock = false;
                    if (actualSeqLenOld % tiling_->selTopKBlockSize != 0) {
                        int32_t tailBlockIdx = actualSeqLenOld / tiling_->selTopKBlockSize;
                        isInvalidBlock = kOld == tailBlockIdx;
                    }
                    if (kOld > CeilDiv(actualSeqLenOld, tiling_->selTopKBlockSize) - 1) {
                        isInvalidBlock = true;
                    }

                    if (!isInvalidBlock) {
                        isHit = true;
                        if (seqIdx == sOld) {
                            // 同seq命中必更新
                            isHitSameSeq = true;
                            insertStatusSameSeq[kOld] = topKIdx;
                            hitFromSrcSeq[topKIdx] = CUR_SEG_HIT_FLAG;
                            maxHitSameSeqIdx = (kOld > maxHitSameSeqIdx) ? kOld : maxHitSameSeqIdx;
                            if (maybeMaxTopkId) {
                                maxTopkWritedIdx = kOld;
                            }
                        } else {
                            // 非同seq命中,如果已经命中了其他的,就不更新,防止把同seq命中的给刷成非同seq命中
                            if (hitFromSrcSeq[topKIdx] == -1) {
                                hitFromSrcSeq[topKIdx] = sOld * tiling_->headnum * tiling_->topk +
                                                            hOld * tiling_->topk + kOld;
                            }
                        }
                    }
                    break;
                }
                if (isHit) {
                    break;
                }
            }
            if (isHit && sOld >= seqIdx) {
                // 同seq命中或下面的seq有命中,不再往下找
                break;
            }
        }
    }

    __aicore__ inline void CopyFromFullKv(int64_t bsIdx, int64_t topKId, int64_t gatherBlockSize,
        int64_t selKRopeAddr, int64_t selKvCacheAddr)
    {
        // 从topk中获取索引，再根据full_kv_block_table中的地址，计算full kv cache的偏移地址
        int64_t kvBlockTableIdx = (topKId * tiling_->selTopKBlockSize) / tiling_->fullKvBlockSize;
        int64_t kvBlockSizeOffset = (topKId * tiling_->selTopKBlockSize) % tiling_->fullKvBlockSize;
        int32_t kvBlockNumIdx = fullKvBlockTableGm_.GetValue(bsIdx * tiling_->fullMaxBlockNum + kvBlockTableIdx);
        ASSERT_MSG(kvBlockNumIdx >= 0, "kvBlockTableIdx:%ld should be greater than 0", kvBlockNumIdx);
        ASSERT_MSG(
            kvBlockNumIdx < tiling_->fullKvBlockNum, "kvBlockNumIdx:%ld should be less than fullKvBlockNum:%ld",
            kvBlockNumIdx, tiling_->fullKvBlockNum);
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
        DataCopyPad(inTensor[kRopeUbOffset_], fullKRopeGm_[fullKRopeAddr], dataCopyParams1, dataCopyPadParams);

        kvCacheQue_.EnQue(inTensor);
        inTensor = kvCacheQue_.DeQue<T>();

        DataCopyPad(selKvCacheGm_[selKvCacheAddr], inTensor, dataCopyParams);
        DataCopyPad(selKRopeGm_[selKRopeAddr], inTensor[kRopeUbOffset_], dataCopyParams1);
        kvCacheQue_.FreeTensor(inTensor);
    }

    __aicore__ inline void CopyFromSelKv(int64_t bsIdx, int64_t hitFromValue,
        LocalTensor<int32_t>& selKvBlockTableLocal, int64_t gatherBlockSize,
        int64_t selKRopeAddr, int64_t selKvCacheAddr)
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
        DataCopyPad(inTensor[kRopeUbOffset_], selKRopeGm_[srcKRopeAddr], dataCopyParams1, dataCopyPadParams);

        kvCacheQue_.EnQue(inTensor);
        inTensor = kvCacheQue_.DeQue<T>();

        DataCopyPad(selKvCacheGm_[selKvCacheAddr], inTensor, dataCopyParams);
        DataCopyPad(selKRopeGm_[selKRopeAddr], inTensor[kRopeUbOffset_], dataCopyParams1);
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
        DataCopyPad(maxTensor[kRopeUbOffset_], selKRopeGm_[maxKRopeAddr], maxKropeParam, dataCopyPadParams);
        DataCopyPad(lastTensor, selKvCacheGm_[lastKvCacheAddr], lastKvCacheParam, dataCopyPadParams);
        DataCopyPad(lastTensor[kRopeUbOffset_], selKRopeGm_[lastKRopeAddr], lastKropeParam, dataCopyPadParams);

        SetWaitFlag<HardEvent::MTE2_MTE3>(HardEvent::MTE2_MTE3);

        DataCopyPad(selKvCacheGm_[maxKvCacheAddr], lastTensor, lastKvCacheParam);
        DataCopyPad(selKRopeGm_[maxKRopeAddr], lastTensor[kRopeUbOffset_], lastKropeParam);
        DataCopyPad(selKvCacheGm_[lastKvCacheAddr], maxTensor, maxKvCacheParam);
        DataCopyPad(selKRopeGm_[lastKRopeAddr], maxTensor[kRopeUbOffset_], maxKropeParam);

        kvCacheQue_.FreeTensor(lastTensor);
        kvCacheQue_.FreeTensor(maxTensor);
    }

    __aicore__ inline void SetSelectionKvActualSeqLen(
        int64_t seqIdx, int64_t hnIdx, int32_t selActualSeqLen, LocalTensor<int32_t>& selKvActSeqLocal)
    {
        int64_t seqLenIdx = seqIdx * tiling_->headnum + hnIdx;
        selKvActSeqLocal.SetValue(seqLenIdx, selActualSeqLen);
    }

    __aicore__ inline void CopyOutSelKvBlockTableAndSeqLen(int64_t bsIdx, LocalTensor<int32_t>& selKvActSeqLocal,
        LocalTensor<int32_t>& selBlockStatLocal)
    {
        DataCopyExtParams dataCopyParSeqLen{
            static_cast<uint16_t>(1), static_cast<uint32_t>(tiling_->seq * tiling_->headnum * sizeof(int32_t)), 0, 0,
            0};
        DataCopyPad(selKvActualSeqGm_[bsIdx * tiling_->seq * tiling_->headnum], selKvActSeqLocal, dataCopyParSeqLen);

        int32_t statSHT = tiling_->seq * tiling_->headnum * (tiling_->topk + 1);
        DataCopyExtParams dataCopyParBlkStat{
            static_cast<uint16_t>(1), static_cast<uint32_t>(statSHT * sizeof(int32_t)), 0, 0, 0};
        DataCopyPad(selKvBlockStatusGm_[bsIdx * statSHT], selBlockStatLocal, dataCopyParBlkStat);
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

    int32_t kRopeUbOffset_ = 0;
    int32_t selTopKIdxUb_;
    int32_t selBlockStatUb_;
    int32_t selKvBlockTableUb_;
    int32_t selKvActSeqUb_;

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

    TQue<QuePosition::VECIN, 1> selTopKIdxQue_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> kvCacheQue_;
    TBuf<QuePosition::VECCALC> workBuf_;
};

} // namespace GatherSelectionKvCacheNs
#endif // GATHER_SELECTION_KV_CACHE_SPLIT_BS_REUSE_H