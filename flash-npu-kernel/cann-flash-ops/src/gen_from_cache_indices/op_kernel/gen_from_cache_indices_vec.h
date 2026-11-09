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
 * \file gen_from_cache_indices_vec.h
 * \brief 向量化二分查找（lower_bound）+ MTP 因果筛选 + actual_seq。
 *
 * 每核以 grid-stride 处理 (b,s) 行（headnum==1，tiling 限 grid≤2，流限制 2 核）：
 *   1. 标量算 curKvModify = full_kv_actual_seq[b] - offset，offset = (full_q_actual_seq[b]-1) - s；
 *      maxSelectionId = CeilDiv(curKvModify, block_size) - 1；lastGatherBlockSize = 尾块实长。
 *      整行无效（s>=q / curKvModify<=0 / kv<=0）→ from = 全 -2，actual_seq = 0。
 *   2. 有效行：cache_status[b,:] 升序 sanitize，对每个 topk id 向量化二分（lower_bound）→ 命中 slot / 未命中 -1。
 *   3. 因果掩码：id<0 或 id>maxSelectionId → 覆盖为 -2。
 *   4. actual_seq = Σ 有效 topk 位置的 gatherBlockSize（尾块 id==maxSelectionId 用 lastGatherBlockSize；
 *      **按位置计数、不去重**——与解耦流水中 gather 每 topk 位置产一行的语义一致）。
 */
#ifndef GEN_FROM_CACHE_INDICES_VEC_H_
#define GEN_FROM_CACHE_INDICES_VEC_H_

#include "kernel_operator.h"

namespace GenFromCacheIndicesNs {
using namespace AscendC;

constexpr uint32_t GFCI_TOPK = 2048;
constexpr uint32_t GFCI_CACHE = 4096;
// 当前 token 编码基址（opt==2 unifiedSrc）：current → ws + BASE。
// hit 编码 = b*cache_+slot+1，最大值 = B*cache_（推理 batch≪256 → ≪ 2^20），
// 故 us≥BASE ⟺ current，gather pass1 据值大小识别、无需遍历 topk 区间。
// host 侧 PostTiling 断言 B*cache_ < BASE 防静默撞值（见 gen_from_cache_indices_tiling.cpp）。
constexpr int32_t GFCI_CUR_TOKEN_BASE = 1 << 20;

class KernelGenFromCacheIndices {
public:
    __aicore__ inline KernelGenFromCacheIndices(TPipe* pipe)
    {
        pipe_ = pipe;
    }

    __aicore__ inline void Init(GM_ADDR topkIndices, GM_ADDR cacheStatus, GM_ADDR kvActualSeq,
                                GM_ADDR qActualSeq, GM_ADDR fullBtTable, GM_ADDR out,
                                GM_ADDR selKvActualSeq, GM_ADDR writeSlot, GM_ADDR uniSlots,
                                const GenFromCacheIndicesTilingData* tiling)
    {
        S_ = tiling->S;
        B_ = tiling->B;
        topk_ = tiling->topk;
        cache_ = tiling->cache;
        blockSize_ = tiling->blockSize;
        enableOpt_ = tiling->enableTopkIndicesOpt;   // 原值直传（0/1/2）
        fullKvBlockSize_ = tiling->fullKvBlockSize;
        fullMaxBlockNum_ = tiling->fullMaxBlockNum;

        // grid 上限 = tiling->blockNum（= min(2, B*S)，流限制 2 核，见 tiling 注释）；
        // 每核以 grid-stride 遍历全部 B*S 行，b_/s_ 在 Process 行循环内逐行重算。
        blkIdx_ = GetBlockIdx();
        grid_ = tiling->blockNum;
        active_ = (blkIdx_ < grid_);

        topkGm_.SetGlobalBuffer((__gm__ int32_t*)topkIndices);
        cacheGm_.SetGlobalBuffer((__gm__ int32_t*)cacheStatus);
        kvActSeqGm_.SetGlobalBuffer((__gm__ int32_t*)kvActualSeq);
        qActSeqGm_.SetGlobalBuffer((__gm__ int32_t*)qActualSeq);
        outGm_.SetGlobalBuffer((__gm__ int32_t*)out);
        selKvActSeqGm_.SetGlobalBuffer((__gm__ int32_t*)selKvActualSeq);
        writeSlotGm_.SetGlobalBuffer((__gm__ int32_t*)writeSlot);
        fullBtGm_.SetGlobalBuffer((__gm__ int32_t*)fullBtTable);
        uniSlotsGm_.SetGlobalBuffer((__gm__ int32_t*)uniSlots);

        pipe_->InitBuffer(inTopkQue_, 1, topk_ * sizeof(int32_t));
        pipe_->InitBuffer(inCacheQue_, 1, cache_ * sizeof(int32_t));
        pipe_->InitBuffer(workBuf_, WORK_ELEMS * sizeof(int32_t));
        // opt==2：full_kv_block_table 行缓存复用 OFF_CACHE_SAN+OFF_ZERO+OFF_KEY，
        // E_BT=10240 int32；这三段在 ComputeUnifiedSlots 入口均已用完，不增加 UB。
        // 靠时序隔离：btL load→Gather→死，之后 unifiedSrc/page/poff 才写该区，互不踩踏（见 ComputeUnifiedSlots）。
        if (enableOpt_ == 2) {
            curBtBatch_ = 0xFFFFFFFFU;
        }
    }

    __aicore__ inline void Process()
    {
        if (!active_) {
            return;
        }
        // grid-stride 行循环：每核串行处理多行（原一核一行改为按 2 核网格切分）。
        // 行内状态（verifiedKvLen_/maxSelId_/lastGBS_/b_/s_/t_）逐行重算，各队列
        // （inTopkQue_/inCacheQue_）每行行内收支平衡，行间无继承状态、无跨行别名。
        uint32_t totalRows = B_ * S_;
        for (uint32_t row = blkIdx_; row < totalRows; row += grid_) {
            t_ = row;              // 展平 (b,s) 序号 = b*S + s
            b_ = row / S_;
            s_ = row % S_;

            // ---- 标量：因果可见 KV 长度 + maxSelectionId ----
            int32_t kvLen = kvActSeqGm_.GetValue(b_);
            kvLen_ = kvLen;       // 供 ComputeUnifiedSlots 用（避免 override 中途标量 GM 读）
            int32_t qLen = qActSeqGm_.GetValue(b_);
            // MTP：verifiedKvLen_ = kvLen - qLen = curStart（当前 token 区间下界，亦 LoadAndSanitize 失效阈值）。
            // qLen==1 时 = kvLen-1，失效当前 token——decode 必须失效当前 token，否则 update_cache
            // 判 HIT 保留旧池 KV 而不刷新 → 精度坏（原 +1 写法 qLen==1 时退化为不失效，正是该 bug）。
            verifiedKvLen_ = kvLen - qLen;
            // offset = (q-1) - s；s>=q 为 padding 位（offset<0），整行无效。
            int32_t offset = (qLen - 1) - static_cast<int32_t>(s_);
            int32_t curKvModify = kvLen - offset;
            bool rowValid = (kvLen > 0) && (offset >= 0) && (curKvModify > 0);

            if (!rowValid) {
                WriteInvalidRow();
                continue;
            }

            int32_t bs = static_cast<int32_t>(blockSize_);
            maxSelId_ = (curKvModify + bs - 1) / bs - 1;          // CeilDiv - 1
            lastGBS_ = curKvModify - maxSelId_ * bs;              // 尾块实长 ∈ [1, bs]

            LoadAndSanitizeCache(b_);
            ProcessRow();
        }
    }

private:
    static constexpr uint32_t E_TOPK = GFCI_TOPK;
    static constexpr uint32_t E_CACHE = GFCI_CACHE;
    static constexpr float BIG = 1073741824.0f;      // 2^30
    static constexpr uint32_t BSEARCH_ITERS = 13;

    static constexpr uint32_t OFF_CACHE_SAN = 0;
    static constexpr uint32_t OFF_ZERO = OFF_CACHE_SAN + E_CACHE;
    static constexpr uint32_t OFF_KEY = OFF_ZERO + E_CACHE;
    // opt==2 btL 区：复用 OFF_CACHE_SAN+OFF_ZERO+OFF_KEY（10240 int32=40KB）。
    // OFF_KEY 的二分、actual_seq 和 write_slot 临时值均在 ComputeUnifiedSlots 前用完；
    // 行缓存止于 OFF_LO_A，保留从该处开始的 write_slot 等输出。
    // 与 unifiedSrc（OFF_UNI/UNIPAGE/UNIPOFF）时序隔离：btL load→Gather→死，之后 unifiedSrc 才写。
    static constexpr uint32_t OFF_BT = OFF_CACHE_SAN;
    static constexpr uint32_t E_BT = E_CACHE * 2 + E_TOPK;   // 10240
    static constexpr uint32_t OFF_LO_A = OFF_KEY + E_TOPK;
    static constexpr uint32_t OFF_LO_B = OFF_LO_A + E_TOPK;
    static constexpr uint32_t OFF_HI_A = OFF_LO_B + E_TOPK;
    static constexpr uint32_t OFF_HI_B = OFF_HI_A + E_TOPK;
    static constexpr uint32_t OFF_MIDF = OFF_HI_B + E_TOPK;
    static constexpr uint32_t OFF_MIDI = OFF_MIDF + E_TOPK;
    static constexpr uint32_t OFF_MIDP1 = OFF_MIDI + E_TOPK;
    static constexpr uint32_t OFF_GATH = OFF_MIDP1 + E_TOPK;
    static constexpr uint32_t OFF_OUTF = OFF_GATH + E_TOPK;
    static constexpr uint32_t OFF_KEYF = OFF_OUTF + E_TOPK;    // topk id 原值（float），因果比较用
    static constexpr uint32_t OFF_CONTRIB = OFF_KEYF + E_TOPK; // 每 lane actual_seq 贡献
    static constexpr uint32_t OFF_WORK = OFF_CONTRIB + E_TOPK; // ReduceSum work
    static constexpr uint32_t OFF_MASK = OFF_WORK + E_TOPK;
    static constexpr uint32_t OFF_MASK2 = OFF_MASK + E_TOPK;
    // unifiedSrc（opt==2）复用 OFF_CACHE_SAN+OFF_ZERO 区域（LoadAndSanitizeCache 完成后空闲）。
    // OFF_CACHE_SAN(4096) + OFF_ZERO(4096) = 8192 int32 = 32KB，足够放 3×E_TOPK(6144) = 24KB。
    static constexpr uint32_t OFF_UNI = OFF_CACHE_SAN;          // 复用 cache sanitize 区
    static constexpr uint32_t OFF_UNIPAGE = OFF_UNI + E_TOPK;   // unifiedSrc: page 暂存
    static constexpr uint32_t OFF_UNIPOFF = OFF_UNIPAGE + E_TOPK; // unifiedSrc: poff 暂存
    static constexpr uint32_t WORK_ELEMS = OFF_MASK2 + E_TOPK;

    template <HardEvent event>
    __aicore__ inline void SyncEvent()
    {
        event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(event));
        SetFlag<event>(eventId);
        WaitFlag<event>(eventId);
    }

    __aicore__ inline void WriteInvalidRow()
    {
        LocalTensor<int32_t> buf = workBuf_.Get<int32_t>();
        LocalTensor<int32_t> outI = buf[OFF_OUTF];
        Duplicate(outI, (int32_t)-2, topk_);
        PipeBarrier<PIPE_V>();
        SyncEvent<HardEvent::V_MTE3>();
        DataCopyExtParams outParams{1, static_cast<uint32_t>(topk_ * sizeof(int32_t)), 0, 0, 0};
        DataCopyPad(outGm_[t_ * topk_], outI, outParams);
        // MTE3 读完 -2 后才允许 V 复用同一缓冲，否则输出会被后续 -1/0 覆盖。
        SyncEvent<HardEvent::MTE3_V>();
        // write_slot 整行 = -1（整行被掩，无有效槽）。复用 outI（值同为 -1，但语义不同，单独写清晰）。
        Duplicate(outI, (int32_t)-1, topk_);
        PipeBarrier<PIPE_V>();
        SyncEvent<HardEvent::V_MTE3>();
        DataCopyPad(writeSlotGm_[t_ * topk_], outI, outParams);
        SyncEvent<HardEvent::MTE3_V>();
        // unified_slots_in_lru_or_cache 整行 = 0（全 skip）
        Duplicate(outI, (int32_t)0, topk_);
        PipeBarrier<PIPE_V>();
        SyncEvent<HardEvent::V_MTE3>();
        DataCopyPad(uniSlotsGm_[t_ * topk_], outI, outParams);
        // actual_seq = 0
        LocalTensor<int32_t> actI = buf[OFF_CONTRIB];
        Duplicate(actI, (int32_t)0, 8);
        PipeBarrier<PIPE_V>();
        SyncEvent<HardEvent::V_MTE3>();
        DataCopyExtParams actParams{1, static_cast<uint32_t>(sizeof(int32_t)), 0, 0, 0};
        DataCopyPad(selKvActSeqGm_[t_], actI, actParams);
        // 跨行：等 MTE3 读完 UB 再进入下一行（workBuf_ 被下一行向量复用）
        SyncEvent<HardEvent::MTE3_V>();
    }

    __aicore__ inline void LoadAndSanitizeCache(uint32_t b)
    {
        LocalTensor<int32_t> cacheLocal = inCacheQue_.AllocTensor<int32_t>();
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(cache_ * sizeof(int32_t)), 0, 0, 0};
        DataCopyPadExtParams<int32_t> padParams{false, 0, 0, 0};
        DataCopyPad(cacheLocal, cacheGm_[b * cache_], copyParams, padParams);
        inCacheQue_.EnQue(cacheLocal);
        LocalTensor<int32_t> cacheIn = inCacheQue_.DeQue<int32_t>();

        LocalTensor<int32_t> buf = workBuf_.Get<int32_t>();
        LocalTensor<float> cacheSanF = buf[OFF_CACHE_SAN].ReinterpretCast<float>();
        LocalTensor<float> zeroF = buf[OFF_ZERO].ReinterpretCast<float>();
        LocalTensor<uint8_t> maskU8 = buf[OFF_MASK].ReinterpretCast<uint8_t>();

        Cast(cacheSanF, cacheIn, RoundMode::CAST_ROUND, cache_);
        PipeBarrier<PIPE_V>();

        // 失效 id >= verifiedKvLen_(=curStart) 的条目：覆盖当前 token 区间 [curStart,curEnd) 全部
        // draft（含上一步已 commit 的 token）及上步被拒草稿，置 -1 供 update_cache 判 NEW 刷新池 KV；
        // -1 随后 sanitize 置 BIG，二分查找不可命中。
        LocalTensor<float> verifiedF = buf[OFF_ZERO].ReinterpretCast<float>();
        Duplicate(verifiedF, static_cast<float>(verifiedKvLen_), cache_);
        PipeBarrier<PIPE_V>();
        Compare(maskU8, cacheSanF, verifiedF, CMPMODE::LT, cache_);
        PipeBarrier<PIPE_V>();
        Select(cacheSanF, maskU8, cacheSanF, (float)-1.0, SELMODE::VSEL_TENSOR_SCALAR_MODE, cache_);
        PipeBarrier<PIPE_V>();

        // s_ == 0：失效结果原路写回输入 cacheStatus（每 batch 一次，供后续 forward 复用）。
        if (s_ == 0) {
            Cast(cacheIn, cacheSanF, RoundMode::CAST_ROUND, cache_);
            PipeBarrier<PIPE_V>();
            SyncEvent<HardEvent::V_MTE3>();
            DataCopyPad(cacheGm_[b * cache_], cacheIn, copyParams);
        }

        Duplicate(zeroF, (float)0.0, cache_);
        PipeBarrier<PIPE_V>();
        Compare(maskU8, cacheSanF, zeroF, CMPMODE::GE, cache_);
        PipeBarrier<PIPE_V>();
        Select(cacheSanF, maskU8, cacheSanF, BIG, SELMODE::VSEL_TENSOR_SCALAR_MODE, cache_);
        PipeBarrier<PIPE_V>();

        inCacheQue_.FreeTensor(cacheLocal);
    }

    // unifiedSrc 计算（opt==2）：
    //   >0 且 <BASE=hit(b*cache_+slot+1)；
    //   ≥BASE=当前token(ws+BASE)；<0=miss(-(pb*bs+poff+1))；0=skip
    //   当前 token 与 hit 都编码为正值，但 hit 最大值=B*cache_<BASE，故 gather 按值大小
    //   区分（us≥BASE→current），无需遍历 topk 区间。
    //   page/poff 向量化算，pb 标量 GM GetValue 查页表（避免 UB 溢出）。
    __aicore__ inline void ComputeUnifiedSlots(LocalTensor<int32_t>& buf,
                                               LocalTensor<int32_t>& outI,
                                               LocalTensor<float>& keyOrigF,
                                               LocalTensor<uint8_t>& maskLo,
                                               LocalTensor<uint8_t>& maskU8)
    {
        // 刷新 batch（标量 GM 读 seq range，与 ComputeUnifiedSlots 无直接依赖）

        // ---- hit 编码：uniL = b_*cache_ + slot + 1（slot = outI，hit 时 >=0）----
        LocalTensor<int32_t> uniL = buf[OFF_UNI];
        // outI 已有 from_cache_indices 值（hit=slot, miss=-1, skip=-2）
        // hit mask: outI >= 0
        // 先保存 topk 原值到 OFF_HI_B（二分已结束，HI_B 空闲；keyOrigF 后面要被覆盖为 outF）
        LocalTensor<int32_t> topkI = buf[OFF_HI_B];
        Cast(topkI, keyOrigF, RoundMode::CAST_ROUND, topk_);
        PipeBarrier<PIPE_V>();

        LocalTensor<float> outF = keyOrigF;   // keyOrigF 原值已保存到 topkI，可复用
        Cast(outF, outI, RoundMode::CAST_ROUND, topk_);
        PipeBarrier<PIPE_V>();
        LocalTensor<float> zeroF = buf[OFF_MIDP1].ReinterpretCast<float>();
        Duplicate(zeroF, (float)0.0, topk_);
        PipeBarrier<PIPE_V>();
        LocalTensor<uint8_t> hitMask = maskU8;   // 复用 maskU8（opt 路径已覆写，但此处重新计算）
        Compare(hitMask, outF, zeroF, CMPMODE::GE, topk_);   // outI >= 0 = hit
        PipeBarrier<PIPE_V>();
        // hit Adds 延后到 btL Gather 之后：uniL=OFF_UNI=OFF_BT，若先写会被 btL load 踩踏。

        // ---- btL 每行重读到 OFF_BT（复用已用完的 cache/zero/key，≤E_BT=10240 页）----
        // page/poff 用 OFF_MIDP1/OFF_MIDI/OFF_MIDF（避开 OFF_BT），时序：btL load→Gather→死，hit Adds 再覆写 OFF_BT。
        uint32_t btLoadCount = fullMaxBlockNum_ < E_BT ? fullMaxBlockNum_ : E_BT;
        LocalTensor<int32_t> btL = buf[OFF_BT];
        SyncEvent<HardEvent::V_MTE2>();   // 等上轮 V 侧 Gather 读完 OFF_BT 才允许覆写
        DataCopyExtParams btParams{1, static_cast<uint32_t>(btLoadCount * sizeof(int32_t)), 0, 0, 0};
        DataCopyPadExtParams<int32_t> btPad{false, 0, 0, 0};
        DataCopyPad(btL, fullBtGm_[static_cast<uint64_t>(b_) * fullMaxBlockNum_], btParams, btPad);
        SyncEvent<HardEvent::MTE2_V>();   // 搬完才让本轮 V 侧 Gather 读

        // ---- page/poff（OFF_MIDP1/OFF_MIDI/OFF_MIDF，避开 OFF_BT）----
        LocalTensor<float> pageF = buf[OFF_MIDP1].ReinterpretCast<float>();   // zeroF 已死
        LocalTensor<float> bsF = buf[OFF_MIDI].ReinterpretCast<float>();
        Duplicate(bsF, static_cast<float>(static_cast<int32_t>(fullKvBlockSize_)), topk_);
        PipeBarrier<PIPE_V>();
        LocalTensor<float> topkF2 = outF;  // outF 已死
        Cast(topkF2, topkI, RoundMode::CAST_ROUND, topk_);
        PipeBarrier<PIPE_V>();
        Div(pageF, topkF2, bsF, topk_);
        PipeBarrier<PIPE_V>();
        LocalTensor<int32_t> pageI = buf[OFF_MIDP1];
        Cast(pageI, pageF, RoundMode::CAST_FLOOR, topk_);
        PipeBarrier<PIPE_V>();
        Maxs(pageI, pageI, 0, topk_);   // 防 topk<0 → 负 page
        PipeBarrier<PIPE_V>();
        Mins(pageI, pageI, static_cast<int32_t>(btLoadCount - 1), topk_);  // 防 page ≥ btLoadCount 越界 Gather
        PipeBarrier<PIPE_V>();
        // poff = topk - page * fullKvBlockSize
        LocalTensor<int32_t> poffI = buf[OFF_MIDF];
        Muls(poffI, pageI, static_cast<int32_t>(fullKvBlockSize_), topk_);
        PipeBarrier<PIPE_V>();
        Muls(poffI, poffI, -1, topk_);
        PipeBarrier<PIPE_V>();
        Add(poffI, topkI, poffI, topk_);
        PipeBarrier<PIPE_V>();
        // page*4 字节偏移 → Gather pb from btL
        Muls(pageI, pageI, static_cast<int32_t>(sizeof(int32_t)), topk_);
        PipeBarrier<PIPE_V>();
        // 保存 topk 副本到 OFF_MIDI（bsF 已死；OFF_HI_B 将被 pbI 复用），供 current-token 覆盖用真实 topk
        LocalTensor<int32_t> topkSave = buf[OFF_MIDI];
        Adds(topkSave, topkI, (int32_t)0, topk_);
        PipeBarrier<PIPE_V>();
        // topkI 已不再需要，复用 OFF_HI_B 作 pbI
        LocalTensor<int32_t> pbI = buf[OFF_HI_B];
        Gather(pbI, btL, pageI.ReinterpretCast<uint32_t>(), (uint32_t)0, topk_);
        PipeBarrier<PIPE_V>();
        // ---- hit Adds（btL 已 Gather 完、死，覆写 OFF_UNI=OFF_BT）----
        Adds(uniL, outI, static_cast<int32_t>(b_ * cache_ + 1), topk_);
        PipeBarrier<PIPE_V>();
        // fullOffset = pb * fullKvBlockSize + poff
        Muls(pbI, pbI, static_cast<int32_t>(fullKvBlockSize_), topk_);
        PipeBarrier<PIPE_V>();
        Add(pbI, poffI, pbI, topk_);
        PipeBarrier<PIPE_V>();
        // miss 值 = -(fullOffset + 1)，存 OFF_MIDF（poffI 已死）
        LocalTensor<int32_t> missVal = buf[OFF_MIDF];
        Adds(missVal, pbI, 1, topk_);
        PipeBarrier<PIPE_V>();
        Muls(missVal, missVal, -1, topk_);
        PipeBarrier<PIPE_V>();
        // miss mask: outI == -1 AND validCausal
        Cast(outF, outI, RoundMode::CAST_ROUND, topk_);
        PipeBarrier<PIPE_V>();
        Duplicate(zeroF, (float)-1.0, topk_);
        PipeBarrier<PIPE_V>();
        LocalTensor<uint8_t> missEqMask = buf[OFF_HI_A].ReinterpretCast<uint8_t>();
        Compare(missEqMask, outF, zeroF, CMPMODE::EQ, topk_);
        PipeBarrier<PIPE_V>();
        uint32_t m16 = (topk_ + 15) / 16;
        And(missEqMask.ReinterpretCast<int16_t>(), missEqMask.ReinterpretCast<int16_t>(),
            maskLo.ReinterpretCast<int16_t>(), static_cast<int32_t>(m16));
        PipeBarrier<PIPE_V>();
        // uniL = miss ? missVal : uniL
        LocalTensor<float> missValF = buf[OFF_GATH].ReinterpretCast<float>();
        Cast(missValF, missVal, RoundMode::CAST_ROUND, topk_);
        PipeBarrier<PIPE_V>();
        LocalTensor<float> uniF = buf[OFF_UNI].ReinterpretCast<float>();
        Cast(uniF, uniL, RoundMode::CAST_ROUND, topk_);
        PipeBarrier<PIPE_V>();
        // uniL = miss ? missVal : uniL
        Select(uniF, missEqMask, missValF, uniF, SELMODE::VSEL_TENSOR_TENSOR_MODE, topk_);
        PipeBarrier<PIPE_V>();
        // skip → 0：非 hit 且非 miss → 0
        LocalTensor<uint8_t> hitOrMiss = hitMask;
        Or(hitOrMiss.ReinterpretCast<int16_t>(), hitOrMiss.ReinterpretCast<int16_t>(),
           missEqMask.ReinterpretCast<int16_t>(), static_cast<int32_t>(m16));
        PipeBarrier<PIPE_V>();
        Select(uniF, hitOrMiss, uniF, (float)0.0, SELMODE::VSEL_TENSOR_SCALAR_MODE, topk_);
        PipeBarrier<PIPE_V>();
        // ---- 当前 token 覆盖（向量化，消除 2048 标量扫描）----
        // 编码：current → ws + GFCI_CUR_TOKEN_BASE（≥BASE，与 hit 的 b*cache_+slot+1 区分；
        //   hit 最大值 = bs*cache_ < BASE，故 us≥BASE ⟺ current）。gather pass1 据值识别，
        //   无需遍历 topk 区间。curEnd=kvLen。
        //   curStart = verifiedKvLen_ = kvLen - qLen：与 mode=1 gather 的 [curStart_,curEnd_)
        //   一致，覆盖全部 qLen 个 draft（含上一步已 commit 的 token，curS=0）。
        //   历史坑：curStart 曾比 kvLen-qLen 大 1（旧 verifiedKvLen_=kvLen-qLen+1 直接作 curStart），
        //   slotBuf[0] 未登记 → 精度坏；现已修正为 curStart = verifiedKvLen_ = kvLen-qLen。
        //   inRange 用真实 topk（topkSave 副本），不依赖 full_kv_block_table 恒等映射。
        int32_t curStart = verifiedKvLen_;
        int32_t curEnd = kvLen_;   // Process 已读 kvLen，避免 override 中途标量 GM 读
        LocalTensor<int32_t> wsI = buf[OFF_LO_A];   // write_slot（二分已结束，LO_A 存 wsI）
        // topk→float（910b int32 Compare 仅 EQ，GE/LT 须 float 域）
        LocalTensor<float> topkF3 = buf[OFF_GATH].ReinterpretCast<float>();   // missValF 已死
        Cast(topkF3, topkSave, RoundMode::CAST_ROUND, topk_);
        PipeBarrier<PIPE_V>();
        // inRange = (topk >= curStart) & (topk < curEnd)
        LocalTensor<float> curStartF = buf[OFF_MIDP1].ReinterpretCast<float>();
        Duplicate(curStartF, static_cast<float>(curStart), topk_);
        PipeBarrier<PIPE_V>();
        LocalTensor<uint8_t> geM = buf[OFF_MASK].ReinterpretCast<uint8_t>();
        Compare(geM, topkF3, curStartF, CMPMODE::GE, topk_);
        PipeBarrier<PIPE_V>();
        LocalTensor<float> curEndF = buf[OFF_MIDI].ReinterpretCast<float>();   // topkSave 已转存 topkF3，OFF_MIDI 可复用
        Duplicate(curEndF, static_cast<float>(curEnd), topk_);
        PipeBarrier<PIPE_V>();
        LocalTensor<uint8_t> ltM = buf[OFF_MASK2].ReinterpretCast<uint8_t>();
        Compare(ltM, topkF3, curEndF, CMPMODE::LT, topk_);
        PipeBarrier<PIPE_V>();
        LocalTensor<int16_t> inRangeM = geM.ReinterpretCast<int16_t>();
        And(inRangeM, inRangeM, ltM.ReinterpretCast<int16_t>(), static_cast<int32_t>(m16));
        PipeBarrier<PIPE_V>();
        // hasSlot = (wsI >= 0)（对应原 if(ws>=0) 守卫）
        LocalTensor<float> wsF = buf[OFF_KEYF].ReinterpretCast<float>();   // outF 已死
        Cast(wsF, wsI, RoundMode::CAST_ROUND, topk_);
        PipeBarrier<PIPE_V>();
        LocalTensor<float> zeroF3 = buf[OFF_MIDF].ReinterpretCast<float>();
        Duplicate(zeroF3, (float)0.0, topk_);
        PipeBarrier<PIPE_V>();
        LocalTensor<uint8_t> hasM = buf[OFF_HI_A].ReinterpretCast<uint8_t>();   // missEqMask 已死
        Compare(hasM, wsF, zeroF3, CMPMODE::GE, topk_);
        PipeBarrier<PIPE_V>();
        // curMask = inRange & hasSlot
        And(inRangeM, inRangeM, hasM.ReinterpretCast<int16_t>(), static_cast<int32_t>(m16));
        PipeBarrier<PIPE_V>();
        // curVal = ws + BASE
        LocalTensor<int32_t> curValI = buf[OFF_UNIPAGE];   // pageI 已死
        Adds(curValI, wsI, static_cast<int32_t>(GFCI_CUR_TOKEN_BASE), topk_);
        PipeBarrier<PIPE_V>();
        LocalTensor<float> curValF = buf[OFF_UNIPOFF].ReinterpretCast<float>();   // poffI/missVal 已死
        Cast(curValF, curValI, RoundMode::CAST_ROUND, topk_);
        PipeBarrier<PIPE_V>();
        // uniF = curMask ? curValF : uniF
        Select(uniF, inRangeM.ReinterpretCast<uint8_t>(), curValF, uniF,
               SELMODE::VSEL_TENSOR_TENSOR_MODE, topk_);
        PipeBarrier<PIPE_V>();

        Cast(uniL, uniF, RoundMode::CAST_ROUND, topk_);
        PipeBarrier<PIPE_V>();
        // 向量化写入由 ProcessRow 的 V_MTE3 覆盖（原 S_MTE3 仅标量路径需要）
    }

    __aicore__ inline void ProcessRow()
    {
        LocalTensor<int32_t> topkLocal = inTopkQue_.AllocTensor<int32_t>();
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(topk_ * sizeof(int32_t)), 0, 0, 0};
        DataCopyPadExtParams<int32_t> padParams{false, 0, 0, 0};
        DataCopyPad(topkLocal, topkGm_[t_ * topk_], copyParams, padParams);
        inTopkQue_.EnQue(topkLocal);
        LocalTensor<int32_t> topkIn = inTopkQue_.DeQue<int32_t>();

        LocalTensor<int32_t> buf = workBuf_.Get<int32_t>();
        LocalTensor<float> cacheSanF = buf[OFF_CACHE_SAN].ReinterpretCast<float>();
        LocalTensor<float> keyF = buf[OFF_KEY].ReinterpretCast<float>();
        LocalTensor<float> keyOrigF = buf[OFF_KEYF].ReinterpretCast<float>();
        LocalTensor<float> loA = buf[OFF_LO_A].ReinterpretCast<float>();
        LocalTensor<float> loB = buf[OFF_LO_B].ReinterpretCast<float>();
        LocalTensor<float> hiA = buf[OFF_HI_A].ReinterpretCast<float>();
        LocalTensor<float> hiB = buf[OFF_HI_B].ReinterpretCast<float>();
        LocalTensor<float> midF = buf[OFF_MIDF].ReinterpretCast<float>();
        LocalTensor<int32_t> midI = buf[OFF_MIDI];
        LocalTensor<uint32_t> midU32 = buf[OFF_MIDI].ReinterpretCast<uint32_t>();
        LocalTensor<float> midP1F = buf[OFF_MIDP1].ReinterpretCast<float>();
        LocalTensor<float> gathF = buf[OFF_GATH].ReinterpretCast<float>();
        LocalTensor<float> outF = buf[OFF_OUTF].ReinterpretCast<float>();
        LocalTensor<int32_t> outI = buf[OFF_OUTF];
        LocalTensor<uint8_t> maskU8 = buf[OFF_MASK].ReinterpretCast<uint8_t>();
        LocalTensor<uint8_t> maskLo = buf[OFF_MASK2].ReinterpretCast<uint8_t>();

        // topk id → float key（原值留存 keyOrigF 供因果比较）
        Cast(keyF, topkIn, RoundMode::CAST_ROUND, topk_);
        PipeBarrier<PIPE_V>();
        inTopkQue_.FreeTensor(topkLocal);
        Adds(keyOrigF, keyF, (float)0.0, static_cast<int32_t>(topk_));
        PipeBarrier<PIPE_V>();

        Duplicate(loA, (float)0.0, topk_);
        Duplicate(hiA, static_cast<float>(static_cast<int32_t>(cache_)), topk_);
        PipeBarrier<PIPE_V>();

        LocalTensor<float> loCur = loA, loNext = loB;
        LocalTensor<float> hiCur = hiA, hiNext = hiB;
        for (uint32_t it = 0; it < BSEARCH_ITERS; ++it) {
            Add(midF, loCur, hiCur, topk_);
            PipeBarrier<PIPE_V>();
            Muls(midF, midF, (float)0.5, static_cast<int32_t>(topk_));
            PipeBarrier<PIPE_V>();
            Cast(midI, midF, RoundMode::CAST_FLOOR, topk_);
            PipeBarrier<PIPE_V>();
            Cast(midF, midI, RoundMode::CAST_ROUND, topk_);
            PipeBarrier<PIPE_V>();
            Mins(midI, midI, static_cast<int32_t>(cache_ - 1), static_cast<int32_t>(topk_));
            PipeBarrier<PIPE_V>();
            Muls(midI, midI, (int32_t)sizeof(float), static_cast<int32_t>(topk_));
            PipeBarrier<PIPE_V>();
            Gather(gathF, cacheSanF, midU32, (uint32_t)0, topk_);
            PipeBarrier<PIPE_V>();
            Compare(maskU8, gathF, keyF, CMPMODE::LT, topk_);
            PipeBarrier<PIPE_V>();
            Adds(midP1F, midF, (float)1.0, static_cast<int32_t>(topk_));
            PipeBarrier<PIPE_V>();
            Select(loNext, maskU8, midP1F, loCur, SELMODE::VSEL_TENSOR_TENSOR_MODE, topk_);
            Select(hiNext, maskU8, hiCur, midF, SELMODE::VSEL_TENSOR_TENSOR_MODE, topk_);
            PipeBarrier<PIPE_V>();
            LocalTensor<float> tmp;
            tmp = loCur; loCur = loNext; loNext = tmp;
            tmp = hiCur; hiCur = hiNext; hiNext = tmp;
        }

        // 命中判定：cacheSan[min(lo,CACHE-1)] == key ?
        Cast(midI, loCur, RoundMode::CAST_FLOOR, topk_);
        PipeBarrier<PIPE_V>();
        Mins(midI, midI, static_cast<int32_t>(cache_ - 1), static_cast<int32_t>(topk_));
        PipeBarrier<PIPE_V>();
        Muls(midI, midI, (int32_t)sizeof(float), static_cast<int32_t>(topk_));
        PipeBarrier<PIPE_V>();
        Gather(gathF, cacheSanF, midU32, (uint32_t)0, topk_);
        PipeBarrier<PIPE_V>();
        Compare(maskU8, gathF, keyF, CMPMODE::EQ, topk_);
        PipeBarrier<PIPE_V>();
        // out = 命中 ? lo(=slot) : -1
        Select(outF, maskU8, loCur, (float)-1.0, SELMODE::VSEL_TENSOR_SCALAR_MODE, topk_);
        PipeBarrier<PIPE_V>();

        // ---- 因果掩码：validCausal = (id >= 0) && (id <= maxSelId)；无效 → out=-2 ----
        LocalTensor<float> zeroF2 = buf[OFF_ZERO].ReinterpretCast<float>();
        Duplicate(zeroF2, (float)0.0, topk_);
        PipeBarrier<PIPE_V>();
        Compare(maskLo, keyOrigF, zeroF2, CMPMODE::GE, topk_);        // id >= 0
        PipeBarrier<PIPE_V>();
        LocalTensor<float> maxSelF = buf[OFF_MIDP1].ReinterpretCast<float>();
        Duplicate(maxSelF, static_cast<float>(maxSelId_), topk_);
        PipeBarrier<PIPE_V>();
        Compare(maskU8, keyOrigF, maxSelF, CMPMODE::LE, topk_);       // id <= maxSelId
        PipeBarrier<PIPE_V>();
        // validCausal = maskLo AND maskU8（int16 位与）
        uint32_t m16 = (topk_ + 15) / 16;
        And(maskLo.ReinterpretCast<int16_t>(), maskLo.ReinterpretCast<int16_t>(),
            maskU8.ReinterpretCast<int16_t>(), static_cast<int32_t>(m16));
        PipeBarrier<PIPE_V>();
        // out = validCausal ? out : -2
        Select(outF, maskLo, outF, (float)-2.0, SELMODE::VSEL_TENSOR_SCALAR_MODE, topk_);
        PipeBarrier<PIPE_V>();
        Cast(outI, outF, RoundMode::CAST_ROUND, topk_);
        PipeBarrier<PIPE_V>();

        // ---- actual_seq：Σ 有效位置 gatherBlockSize（尾块 id==maxSelId 用 lastGBS）----
        // contrib = validCausal ? blockSize : 0；再对 id==maxSelId 的有效位置补差 (lastGBS - blockSize)
        LocalTensor<float> contrib = buf[OFF_CONTRIB].ReinterpretCast<float>();
        LocalTensor<float> bsT = buf[OFF_HI_A].ReinterpretCast<float>();  // 复用（二分已结束）
        Duplicate(bsT, static_cast<float>(static_cast<int32_t>(blockSize_)), topk_);
        PipeBarrier<PIPE_V>();
        // Select: true→tensor(bsT), false→scalar(0)
        Select(contrib, maskLo, bsT, (float)0.0, SELMODE::VSEL_TENSOR_SCALAR_MODE, topk_);
        PipeBarrier<PIPE_V>();
        if (lastGBS_ != static_cast<int32_t>(blockSize_)) {
            // 对有效且 id==maxSelId 的位置，用 (lastGBS - blockSize) 修正
            LocalTensor<uint8_t> maskEq = buf[OFF_MASK].ReinterpretCast<uint8_t>();
            Duplicate(maxSelF, static_cast<float>(maxSelId_), topk_);
            PipeBarrier<PIPE_V>();
            Compare(maskEq, keyOrigF, maxSelF, CMPMODE::EQ, topk_);
            PipeBarrier<PIPE_V>();
            // eqValid = maskEq AND validCausal
            And(maskEq.ReinterpretCast<int16_t>(), maskEq.ReinterpretCast<int16_t>(),
                maskLo.ReinterpretCast<int16_t>(), static_cast<int32_t>(m16));
            PipeBarrier<PIPE_V>();
            LocalTensor<float> deltaT = buf[OFF_LO_A].ReinterpretCast<float>();  // 复用
            float diff = static_cast<float>(lastGBS_ - static_cast<int32_t>(blockSize_));
            Duplicate(deltaT, diff, topk_);
            PipeBarrier<PIPE_V>();
            LocalTensor<float> delta = buf[OFF_KEY].ReinterpretCast<float>();
            Select(delta, maskEq, deltaT, (float)0.0, SELMODE::VSEL_TENSOR_SCALAR_MODE, topk_);
            PipeBarrier<PIPE_V>();
            Add(contrib, contrib, delta, topk_);
            PipeBarrier<PIPE_V>();
        }
        LocalTensor<float> work = buf[OFF_WORK].ReinterpretCast<float>();
        LocalTensor<float> sumF = buf[OFF_GATH].ReinterpretCast<float>();  // 复用
        ReduceSum<float>(sumF, contrib, work, topk_);
        PipeBarrier<PIPE_V>();
        LocalTensor<int32_t> actI = buf[OFF_CONTRIB];  // 原地转 int
        Cast(actI, sumF, RoundMode::CAST_ROUND, 1);
        PipeBarrier<PIPE_V>();

        // ---- write_slot：有效项前缀和-1，无效项=-1（压缩前移落位，方案模式1）----
        LocalTensor<int32_t> wsI = buf[OFF_LO_A];       // write_slot 输出缓冲（二分已结束，LO_A 空闲）
        if (enableOpt_) {
            // opt：topkIndices 已保证有效值在前、-1 填充在后（前序算子保证)。
            // block_size==1（tiling 已校验）→ actI = selection_kv_actual_seq 即有效 kv 长度 L：
            //   slot[k] = (index[k] < L) ? index[k] : -1
            // CreateVecIndex 生成 topk 大小索引，Compare 得 <L 掩码，Select 掩码内取索引、掩码外 -1。
            SyncEvent<HardEvent::V_S>();
            int32_t actLen = actI.GetValue(0);
            LocalTensor<float> idxF = buf[OFF_KEY].ReinterpretCast<float>();      // keyF 已死
            LocalTensor<float> actLenF = buf[OFF_HI_B].ReinterpretCast<float>();  // onesF 已死
            LocalTensor<float> wsF = buf[OFF_GATH].ReinterpretCast<float>();      // sumF 已死
            CreateVecIndex(idxF, (float)0.0, topk_);
            PipeBarrier<PIPE_V>();
            Duplicate(actLenF, static_cast<float>(actLen), topk_);
            PipeBarrier<PIPE_V>();
            Compare(maskU8, idxF, actLenF, CMPMODE::LT, topk_);
            PipeBarrier<PIPE_V>();
            // Mode 2 writes selection rows at their original TopK positions.
            // KVP owner and causal masks may leave holes before later valid
            // entries; a prefix-count mask would discard those valid entries.
            if (enableOpt_ == 2) {
                Select(wsF, maskLo, idxF, (float)-1.0, SELMODE::VSEL_TENSOR_SCALAR_MODE, topk_);
            } else {
                Select(wsF, maskU8, idxF, (float)-1.0, SELMODE::VSEL_TENSOR_SCALAR_MODE, topk_);
            }
            PipeBarrier<PIPE_V>();
            Cast(wsI, wsF, RoundMode::CAST_ROUND, topk_);
            PipeBarrier<PIPE_V>();
        } else {
            // 原逻辑：valid[k] = validCausal（与 out=-2 掩码同源）。先把 validCausal 落成 int32 到 vBuf，
            // 再标量前缀和：slot = (Σ_{j<=k} valid) - 1 若 valid，否则 -1。
            // 标量基线（topk 次 GetValue/SetValue）先保证正确性，profile 后可换向量 scan。
            LocalTensor<float> validF = buf[OFF_KEY].ReinterpretCast<float>();   // keyF 已死
            LocalTensor<float> onesF = buf[OFF_HI_B].ReinterpretCast<float>();   // 二分已结束，HI_B 空闲
            Duplicate(onesF, (float)1.0, topk_);
            PipeBarrier<PIPE_V>();
            // Select TENSOR_SCALAR：true→tensor(onesF), false→scalar(0)（两操作数不能都是标量）
            Select(validF, maskLo, onesF, (float)0.0, SELMODE::VSEL_TENSOR_SCALAR_MODE, topk_);
            PipeBarrier<PIPE_V>();
            LocalTensor<int32_t> vI = buf[OFF_KEY];         // 原地转 int（与 validF 同址）
            Cast(vI, validF, RoundMode::CAST_ROUND, topk_);
            PipeBarrier<PIPE_V>();
            SyncEvent<HardEvent::V_S>();
            int32_t running = 0;
            for (uint32_t k = 0; k < topk_; ++k) {
                if (vI.GetValue(k) != 0) {
                    wsI.SetValue(k, running);
                    ++running;
                } else {
                    wsI.SetValue(k, (int32_t)-1);
                }
            }
            SyncEvent<HardEvent::S_MTE3>();
        }

        // ---- unified_slots_in_lru_or_cache（仅 enableOpt_==2）----
        //   >0 = hit：b*cache_ + slot + 1（slot = outI[k]，from_cache_indices 的命中值）
        //   <0 = miss：-(pb*fullKvBlockSize + poff + 1)（page=topk//bs, poff=topk%bs, pb=bt[b,page]）
        //   0  = skip（outI==-2 或 !validCausal）
        // 向量化实现：outI(keyOrigF 已死) 复用。与 gather PrecomputeRowVectors ③ 同款 Div/Gather。
        LocalTensor<int32_t> uniI = buf[OFF_UNI];
        if (enableOpt_ == 2) {
            ComputeUnifiedSlots(buf, outI, keyOrigF, maskLo, maskU8);
        } else {
            Duplicate(uniI, (int32_t)0, topk_);
        }

        // ---- 写出 ----
        SyncEvent<HardEvent::V_MTE3>();
        DataCopyExtParams outParams{1, static_cast<uint32_t>(topk_ * sizeof(int32_t)), 0, 0, 0};
        DataCopyPad(outGm_[t_ * topk_], outI, outParams);
        DataCopyPad(writeSlotGm_[t_ * topk_], wsI, outParams);
        DataCopyPad(uniSlotsGm_[t_ * topk_], uniI, outParams);
        DataCopyExtParams actParams{1, static_cast<uint32_t>(sizeof(int32_t)), 0, 0, 0};
        DataCopyPad(selKvActSeqGm_[t_], actI, actParams);
        // 跨行：等 MTE3 读完 UB 再进入下一行（workBuf_ 被下一行向量复用）
        SyncEvent<HardEvent::MTE3_V>();
    }

private:
    TPipe* pipe_;

    GlobalTensor<int32_t> topkGm_;
    GlobalTensor<int32_t> cacheGm_;
    GlobalTensor<int32_t> kvActSeqGm_;
    GlobalTensor<int32_t> qActSeqGm_;
    GlobalTensor<int32_t> outGm_;
    GlobalTensor<int32_t> selKvActSeqGm_;
    GlobalTensor<int32_t> writeSlotGm_;
    GlobalTensor<int32_t> fullBtGm_;
    GlobalTensor<int32_t> uniSlotsGm_;

    TQue<TPosition::VECIN, 1> inTopkQue_;
    TQue<TPosition::VECIN, 1> inCacheQue_;
    TBuf<TPosition::VECCALC> workBuf_;

    uint32_t S_ = 0;
    uint32_t B_ = 0;
    uint32_t topk_ = 0;
    uint32_t cache_ = 0;
    uint32_t blockSize_ = 1;
    uint32_t grid_ = 0;
    uint32_t blkIdx_ = 0;
    uint32_t t_ = 0;
    uint32_t b_ = 0;
    uint32_t s_ = 0;
    int32_t maxSelId_ = 0;
    int32_t lastGBS_ = 1;
    int32_t verifiedKvLen_ = 0;
    int32_t kvLen_ = 0;           // 当前行 kvLen（Process 标量读一次，供 ComputeUnifiedSlots 用，避免 override 中途标量 GM 读）
    uint32_t enableOpt_ = 0;      // 原值直传（0/1/2）
    uint32_t fullKvBlockSize_ = 128;
    uint32_t fullMaxBlockNum_ = 0;
    uint32_t curBtBatch_ = 0xFFFFFFFFU;
    bool active_ = false;
};

} // namespace GenFromCacheIndicesNs

#endif // GEN_FROM_CACHE_INDICES_VEC_H_
