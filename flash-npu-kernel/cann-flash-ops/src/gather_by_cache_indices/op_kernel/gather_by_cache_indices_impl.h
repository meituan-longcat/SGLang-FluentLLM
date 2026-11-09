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
 * \file gather_by_cache_indices_impl.h
 * \brief 条件 scatter-gather：当前 MTP token → 槽位登记；命中→device 共享缓存池；未命中(-1)→full cache。
 *
 * 对每个 unit（topk_indices[b, query_s, k]）的判定顺序（必须先于 hit/miss，因为当前 token 与
 * miss 的 from_cache_indices 都可能为 -1）：
 *   ① topk_index == -1                      → 不处理；
 *   ② cur_start <= topk_index < cur_end     → 当前 token：current_s = topk_index - cur_start，
 *                                              算 target slot 并写 cur_kv_write_slots[b, query_s, current_s]，
 *                                              不读 cache pool、不读 full KV、不写 selection KV；
 *   ③ from_cache_index >= 0                 → 原 cache pool hit 逻辑；
 *   ④ else（== -1）                          → 原 full KV miss 逻辑。
 * 其中 cur_start = actual_seq_lengths_kv[b] - actual_seq_qlen[b]；cur_end = actual_seq_lengths_kv[b]。
 * 被掩单元（from==-2 且不在 ② 区间）保持原逻辑：不读不写。
 *
 * target slot（当前 token）：由 gen_from_cache_indices 分配的 write_slot（前缀压缩槽，即 selection KV
 * 压缩布局的逻辑槽位，与 hit/miss 的写目标同体系）定位：lp = ws、logical_page = ws / PAGE_SIZE、
 * page_offset = ws % PAGE_SIZE、
 * physical_page = selection_kv_block_table[b, query_s, logical_page]、
 * target_slot = physical_page * PAGE_SIZE + page_offset（PAGE_SIZE = selection_kv_block_size）。
 * 不能用 token ID 分页：selection KV 是压缩槽布局（update_cache 解码端用 ws[b,q,k] 寻址，
 * KEEP IN SYNC: update_cache_impl.h PhaseRowMoveFinal），与实际 token 位置无关。
 * 注意 write_slot 是「selection KV 槽位」，不是「写入持久 cache pool 的位置」。
 *
 * cur_kv_write_slots 完全由 kernel 初始化：行对齐切核（每行只属一个核，见 tiling）后，
 * 每行先整行置 -1（UB Duplicate），当前 token 登记到行缓冲，行结束 MTE3 DataCopy 整行落 GM。
 * 不用 GlobalTensor.SetValue：S 管标量 GM 写在核退出时随机丢失（实测），MTE3 整行写可靠。
 * 因此 PTA 侧输出用 at::empty（无需预填 -1）。
 */
#ifndef GATHER_BY_CACHE_INDICES_IMPL_H_
#define GATHER_BY_CACHE_INDICES_IMPL_H_

#include "kernel_operator.h"

namespace GatherByCacheIndicesNs {

constexpr uint32_t GBCI_DOUBLE_BUFFER = 2;
// 当前 token 编码基址（须与 gen_from_cache_indices_vec.h 的 GFCI_CUR_TOKEN_BASE 一致）。
// unifiedSrc ≥ BASE ⟺ current token；pass1 据值大小识别，无需遍历 topk 区间（消除 pre-pass 2048 扫描）。
constexpr int32_t GBCI_CUR_TOKEN_BASE = 1 << 20;

template <typename U>
__aicore__ inline U GbciCeilDiv(U a, U b)
{
    return (a + b - 1) / b;
}

// 与 gen_from_cache_indices_vec.h 同款标量事件同步（GetTPipePtr/PIPE_V 为全局符号）
template <AscendC::HardEvent event>
__aicore__ inline void GbciSyncEvent()
{
    auto eventId = GetTPipePtr()->FetchEventID(event);
    AscendC::SetFlag<event>(eventId);
    AscendC::WaitFlag<event>(eventId);
}

template <typename T, bool Packed = false>
class KernelGatherByCacheIndices {
public:
    __aicore__ inline KernelGatherByCacheIndices(AscendC::TPipe* pipe)
    {
        pipe_ = pipe;
    }

    template <typename TilingData>
    __aicore__ inline void Init(GM_ADDR cache_k_rope, GM_ADDR cache_kv_cache,
                                GM_ADDR from_cache_indices, GM_ADDR selection_topk_indices,
                                GM_ADDR full_k_rope, GM_ADDR full_kv_cache,
                                GM_ADDR full_kv_block_table, GM_ADDR selection_kv_block_table,
                                GM_ADDR write_slot, GM_ADDR actual_seq_lengths_kv, GM_ADDR actual_seq_qlen,
                                GM_ADDR selection_k_rope_out, GM_ADDR selection_kv_cache_out,
                                GM_ADDR cur_kv_write_slots,
                                const TilingData* tiling)
    {
        blkIdx_ = AscendC::GetBlockIdx();
        kRopeDim_ = tiling->kRopeDim;
        kvDim_ = tiling->kvDim;
        P_ = tiling->unitsPerBatch;
        cacheRows_ = tiling->cacheRows;
        unitsPerRealBatch_ = tiling->unitsPerRealBatch;
        seq_ = tiling->seq;
        topk_ = tiling->topk;
        selTopKBlockSize_ = tiling->selTopKBlockSize;
        fullKvBlockSize_ = tiling->fullKvBlockSize;
        fullMaxBlockNum_ = tiling->fullMaxBlockNum;
        fullKvBlockNum_ = tiling->fullKvBlockNum;
        selKvBlockSize_ = tiling->selKvBlockSize;
        selMaxBlockNum_ = tiling->selMaxBlockNum;
        enableSlotsContiguous_ = tiling->enableSlotsContiguous;

        if (blkIdx_ >= tiling->blockNum) {
            valid_ = false;
            return;
        }
        valid_ = true;

        blockNum_ = tiling->blockNum;
        batchCount_ = tiling->totalUnits / unitsPerRealBatch_;

        cacheKRopeGm_.SetGlobalBuffer((__gm__ T*)cache_k_rope);
        cacheKvGm_.SetGlobalBuffer((__gm__ T*)cache_kv_cache);
        fromIdxGm_.SetGlobalBuffer((__gm__ int32_t*)from_cache_indices);
        topkIdxGm_.SetGlobalBuffer((__gm__ int32_t*)selection_topk_indices);
        fullKRopeGm_.SetGlobalBuffer((__gm__ T*)full_k_rope);
        fullKvGm_.SetGlobalBuffer((__gm__ T*)full_kv_cache);
        fullKvBlockTableGm_.SetGlobalBuffer((__gm__ int32_t*)full_kv_block_table);
        selKvBlockTableGm_.SetGlobalBuffer((__gm__ int32_t*)selection_kv_block_table);
        writeSlotGm_.SetGlobalBuffer((__gm__ int32_t*)write_slot);
        kvActSeqGm_.SetGlobalBuffer((__gm__ int32_t*)actual_seq_lengths_kv);
        qActSeqGm_.SetGlobalBuffer((__gm__ int32_t*)actual_seq_qlen);
        selKRopeOutGm_.SetGlobalBuffer((__gm__ T*)selection_k_rope_out);
        selKvOutGm_.SetGlobalBuffer((__gm__ T*)selection_kv_cache_out);
        curWriteSlotsGm_.SetGlobalBuffer((__gm__ int32_t*)cur_kv_write_slots);

        kRopeUbOffset_ = P_ * kvDim_;
        uint32_t bindBytes = (P_ * kvDim_ + P_ * kRopeDim_) * sizeof(T);
        pipe_->InitBuffer(dataQue_, GBCI_DOUBLE_BUFFER, bindBytes);

        // idxQue 按单行容量分配（CHUNK_ROWS == 1：每次只载入当前行的三组索引）。
        // 不能按 count_（每核总 workload）分配——流限制 2 核后 count_≈totalUnits/2，三组 int32 索引
        // 整段载入会击穿 192KB UB（如 B=16,S=1,TOPK=2048 → 3*ceil(16384/8)*8*4B ≈ 192KB，仅索引即超限）。
        // 910b 向量算子要求 count*sizeof(T)=256B（整 256bit 块）且 LocalTensor 起始 32B 对齐：
        //   int32/float(4B) → count 为 64 整数倍；位掩码 And/Or(int16,2B) → count 为 128 整数倍。
        // tiling 限制 topk 为 128 整数倍（int32 count=128→512B=2 块，位掩码 16B 亦整块对齐）。
        alignedNel_ = GbciCeilDiv<uint32_t>(topk_, 128U) * 128U;
        if (alignedNel_ == 0) {
            alignedNel_ = 128U;
        }
        // 三段：[0:]=from_idx, [alignedNel_:]=topk, [2*alignedNel_:]=write_slot（单行）
        pipe_->InitBuffer(idxQue_, 1, 3 * alignedNel_ * sizeof(int32_t));
        // cur_kv_write_slots 每行一个 UB 缓冲（S 个 int32，32B 对齐），行结束 DataCopy 落 GM。
        // 行对齐切核后每行单写者，-1 初始化与槽位登记都在本核内完成，无跨核竞态。
        curSlotBufElems_ = GbciCeilDiv<uint32_t>(seq_, 8U) * 8U;
        pipe_->InitBuffer(curSlotBuf_, curSlotBufElems_ * sizeof(int32_t));
        // ---- 行级向量预计算缓冲（循环2外每行一次算齐 mask / lru / full 偏移）----
        // rowVecBytes=alignedNel_*4 恒为 256 整数倍（alignedNel_ 为 64 倍数）→ int32/float 算子整块。
        uint32_t rowVecBytes = alignedNel_ * sizeof(int32_t);
        // 位掩码缓冲：1bit/元素 packed，And/Or 按 int16(2B) 整 256B 块处理 → 需 m16=int16 数，
        // m16*2 为 256 整数倍。m16=ceil(alignedNel_/2048)*128（128 int16=256B=2048 元素的位掩码）。
        maskBytes_ = GbciCeilDiv<uint32_t>(alignedNel_, 2048U) * 256U;   // ≥256，And/Or 整块访问
        maskM16_ = maskBytes_ / 2U;                                       // int16 元素数
        pipe_->InitBuffer(maskBuf_, enableSlotsContiguous_ == 2 ? 32U : rowVecBytes);       // topkIdxMask（V 装配，int32）
        pipe_->InitBuffer(lruIdxBuf_, enableSlotsContiguous_ == 2 ? 32U : rowVecBytes);     // hit 的 lru cache kv 元素偏移
        pipe_->InitBuffer(fullIdxBuf_, enableSlotsContiguous_ == 2 ? 32U : rowVecBytes);    // miss 的 full cache kv 元素偏移
        pipe_->InitBuffer(pageBuf_, enableSlotsContiguous_ == 2 ? 4096U * sizeof(int32_t) : rowVecBytes);       // page 暂存（Gather 索引）
        pipe_->InitBuffer(poffBuf_, enableSlotsContiguous_ == 2 ? 4096U * sizeof(int32_t) : rowVecBytes);       // poff 暂存（块内元素偏移）
        pipe_->InitBuffer(pbBuf_, enableSlotsContiguous_ == 2 ? 32U : rowVecBytes);         // pb 暂存（Gather 结果 / float 中转）
        pipe_->InitBuffer(cmpU8A_, maskBytes_);         // Compares 位掩码 A（1bit/元素，整 256B 块）
        pipe_->InitBuffer(cmpU8B_, maskBytes_);         // Compares 位掩码 B
        pipe_->InitBuffer(cmpU8C_, maskBytes_);         // Compares 位掩码 C
        pipe_->InitBuffer(btTableBuf_, fullMaxBlockNum_ * sizeof(int32_t));  // 当前 real batch 页表行缓存
        curBatch_ = 0xFFFFFFFFU;   // seq range 与页表行共用同一 batch 缓存
        curStart_ = 0;
        curEnd_ = 0;
    }

    __aicore__ inline void Process()
    {
        if (!valid_) {
            return;
        }
        for (uint32_t b = blkIdx_; b < batchCount_; b += blockNum_) {
            ProcessRequest(b);
        }
    }

private:
    __aicore__ inline void ProcessRequest(uint32_t b)
    {
        startUnit_ = b * unitsPerRealBatch_;
        AscendC::LocalTensor<int32_t> idxLocal;
        AscendC::DataCopyPadExtParams<T> dataPad{false, 0, 0, 0};

        // 逐行处理（行对齐切核：每行只属于本核）：
        //   行开始 → curSlotBuf 置 -1；pass1 读 / pass2 写；行结束 → DataCopy 整行落 GM。
        //   每行的三组索引（from_idx/topk/write_slot）在行首按行粒度载入（CHUNK_ROWS=1），
        //   行尾 FreeTensor 归还，下轮 AllocTensor 等待归还后复用——行内算法与一次性整段载入等价。
        uint32_t rowCount = seq_;
        for (uint32_t ri = 0; ri < rowCount; ++ri) {
            uint32_t rowBase = ri * topk_;
            uint32_t rowUnits = topk_;

            AscendC::LocalTensor<int32_t> slotBuf = curSlotBuf_.Get<int32_t>();
            GbciSyncEvent<AscendC::HardEvent::MTE3_V>();
            AscendC::Duplicate(slotBuf, (int32_t)-1, seq_);
            RefreshBatchState(startUnit_ + rowBase);
            // Match Gen's per-query causal validity; an empty owner must not
            // suppress a valid current-only row. Publish invalid current slots
            // even when no selection payload or indices need to be read.
            int32_t offset = curQlen_ - 1 - static_cast<int32_t>(ri);
            if (curEnd_ <= 0 || offset < 0 || curEnd_ - offset <= 0) {
                FlushCurSlots(slotBuf, startUnit_ / topk_ + ri);
                continue;
            }

            // ---- 载入当前行三组索引 ----
            AscendC::DataCopyExtParams idxExt{1, rowUnits * static_cast<uint32_t>(sizeof(int32_t)), 0, 0, 0};
            AscendC::DataCopyPadExtParams<int32_t> idxPad{false, 0, 0, 0};
            idxLocal = idxQue_.AllocTensor<int32_t>();
            AscendC::DataCopyPad(idxLocal, fromIdxGm_[startUnit_ + rowBase], idxExt, idxPad);
            AscendC::DataCopyPad(idxLocal[alignedNel_], topkIdxGm_[startUnit_ + rowBase], idxExt, idxPad);
            AscendC::DataCopyPad(idxLocal[2 * alignedNel_], writeSlotGm_[startUnit_ + rowBase], idxExt, idxPad);
            idxQue_.EnQue(idxLocal);
            idxLocal = idxQue_.DeQue<int32_t>();

            if (enableSlotsContiguous_ == 2 && curBatch_ != reuseBatch_) {
                reuseBatch_ = curBatch_;
                reuseReady_ = false;
                reuseEntries_ = 0;
            }


            // ---- 行级向量预计算（循环2外，每行一次）：mask / lru_cache_idx / full_cache_idx ----
            // enableSlotsContiguous_==2：unifiedSrc 模式，host 侧预计算统一源索引（编码在 write_slot 中），
            //   kernel 跳过 V-pipe 预计算 → 无 UB L2 write-back → 无 MTE2 污染。
            //   pass1 每 unit 仅 1 GetValue（从 idxLocal[2*alignedNel_] 读 unifiedSrc），原需 2（maskL+lruL/fullL）。
            // selKvBlockTableGm_ 值为 range(0, ROW_NUM*selMaxBlockNum_)（page id 连续），故
            // (row,blk,off) 扁平偏移 = rowIdx*selMaxBlockNum_*selKvBlockSize_ + ws（blk*bs+off=ws）。
            // 写 selection 与 cur_kv_write_slots 都用此线性偏移，无需查页表/拆 blk/off/段合并。
            int64_t rowBaseOff = static_cast<int64_t>(startUnit_ / topk_ + ri) *
                                 selMaxBlockNum_ * selKvBlockSize_;
            uint32_t rb = curBatch_;
            if (enableSlotsContiguous_ != 2) {
                PrecomputeRowVectors(idxLocal, rb);
            } else {
                // mode=2：跳过 V-pipe 预计算。当前 token 登记 slotBuf 折叠进 pass1：
                // gen 的 unifiedSrc 编码 current → ws+BASE（≥BASE），pass1 的 GBCI_COPY_UNIT
                // 据 us≥BASE 识别并登记 slotBuf、跳过数据拷贝，无需 pre-pass 遍历 topk_。
                rowUnits = UnifiedRowExtent(idxLocal);
            }
            // 预计算（V 写 mask/lru/full，含行首 slotBuf 的 Duplicate）完成后，
            // pass1/pass2 的标量 GetValue 与 RegisterCurWriteSlot 的 SetValue 才能读/写
            // （910b 上 GetValue/SetValue 走标量 pipe，与 V pipe 无自动排序，需 V_S 事件，gen 同款）。
            // unifiedSrc 模式：V 无预计算，但 idxLocal 由 idxQue DeQue(MTE2→V) 加载，仍需 V_S 过渡到 S pipe。
            GbciSyncEvent<AscendC::HardEvent::V_S>();
            AscendC::LocalTensor<int32_t> maskL = maskBuf_.Get<int32_t>();
            AscendC::LocalTensor<int32_t> lruL = lruIdxBuf_.Get<int32_t>();
            AscendC::LocalTensor<int32_t> fullL = fullIdxBuf_.Get<int32_t>();

            for (uint32_t base = 0; base < rowUnits; base += P_) {
                uint32_t curBatch = (base + P_ <= rowUnits) ? P_ : (rowUnits - base);

                AscendC::LocalTensor<T> t = dataQue_.AllocTensor<T>();

                AscendC::DataCopyExtParams kvExt1{1, kvDim_ * static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
                AscendC::DataCopyExtParams kRopeExt1{1, kRopeDim_ * static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
                // pass1（读 + 当前 token 登记 slotBuf）：
                //   enableSlotsContiguous_==2 (unifiedSrc): 4 路循环展开
                //     先连续读 4 个 GetValue（scalar 流水线不 stall），再发射 4 组 DataCopyPad，
                //     让 GetValue[i+1] 与 DataCopyPad[i] 发射 overlap。
                //     P_=32 是 4 的倍数，curBatch 为 P_ 或尾部余数（仍 ≤ P_），
                //     逐元素展开 4 组恰好覆盖，无需尾循环。
                //   其他模式：原始逐元素循环（2 GetValue/unit）
                uint32_t copyCount = 0;
                if (enableSlotsContiguous_ == 2) {
                    // 4 路展开：先连续读 4 个 GetValue（scalar 流水化），再发射 4 组 DataCopyPad。
                    // P_=32 是 4 的倍数，curBatch ≤ P_，逐组展开恰好覆盖，无需尾循环。
                    #define GBCI_COPY_UNIT(bb, uv) \
                        do { \
                            if (uv >= GBCI_CUR_TOKEN_BASE) { \
                                int32_t tpc = idxLocal.GetValue(alignedNel_ + (base + (bb))); \
                                int32_t curS = tpc - curStart_; \
                                int32_t ws = (uv) - GBCI_CUR_TOKEN_BASE; \
                                slotBuf.SetValue(static_cast<uint32_t>(curS), \
                                                 static_cast<int32_t>(rowBaseOff + ws)); \
                            } else if (uv > 0) { \
                                int64_t row = static_cast<int64_t>(uv - 1); \
                                AscendC::DataCopyPad(t[(bb) * kvDim_], cacheKvGm_[row * kvDim_], kvExt1, dataPad); \
                                if constexpr (!Packed) { AscendC::DataCopyPad(t[kRopeUbOffset_ + (bb) * kRopeDim_], \
                                                     cacheKRopeGm_[row * kRopeDim_], kRopeExt1, dataPad); } \
                            } else if (uv < 0) { \
                                int32_t cached = FindCompletedSelection(uv); \
                                int64_t off = cached >= 0 ? cached : static_cast<int64_t>(-uv - 1); \
                                if (cached >= 0) { \
                                    AscendC::DataCopyPad(t[(bb) * kvDim_], selKvOutGm_[off * kvDim_], kvExt1, dataPad); \
                                    if constexpr (!Packed) { AscendC::DataCopyPad(t[kRopeUbOffset_ + (bb) * kRopeDim_], selKRopeOutGm_[off * kRopeDim_], kRopeExt1, dataPad); } \
                                } else { \
                                    AscendC::DataCopyPad(t[(bb) * kvDim_], fullKvGm_[off * kvDim_], kvExt1, dataPad); \
                                    if constexpr (!Packed) { AscendC::DataCopyPad(t[kRopeUbOffset_ + (bb) * kRopeDim_], fullKRopeGm_[off * kRopeDim_], kRopeExt1, dataPad); } \
                                } \
                            } \
                        } while (0)
                    for (uint32_t b = 0; b < curBatch; b += 4) {
                        int32_t us0 = idxLocal.GetValue(2 * alignedNel_ + base + b);
                        int32_t us1 = idxLocal.GetValue(2 * alignedNel_ + base + b + 1);
                        int32_t us2 = idxLocal.GetValue(2 * alignedNel_ + base + b + 2);
                        int32_t us3 = idxLocal.GetValue(2 * alignedNel_ + base + b + 3);
                        GBCI_COPY_UNIT(b,     us0);
                        GBCI_COPY_UNIT(b + 1, us1);
                        GBCI_COPY_UNIT(b + 2, us2);
                        GBCI_COPY_UNIT(b + 3, us3);
                        // A zero unified source is an unused selection position.
                        // Do not write an entire 32-record chunk when every source
                        // is zero. Current-token entries remain nonzero and their
                        // slot registration/patching is unchanged.
                        copyCount |= static_cast<uint32_t>(us0 | us1 | us2 | us3);
                    }
                    #undef GBCI_COPY_UNIT
                } else {
                    // 原始模式：2 GetValue/unit（maskL + lruL/fullL）— 4 路展开
                    // 先连续读 4 个 maskL，再按类型读 lruL/fullL 并发射 DataCopyPad
                    #define GBCI_COPY_ORIG(bb, mm, loc) \
                        do { \
                            if (mm == -1) { \
                                int64_t row = static_cast<int64_t>(lruL.GetValue(loc)); \
                                AscendC::DataCopyPad(t[(bb) * kvDim_], cacheKvGm_[row * kvDim_], kvExt1, dataPad); \
                                if constexpr (!Packed) { AscendC::DataCopyPad(t[kRopeUbOffset_ + (bb) * kRopeDim_], \
                                                     cacheKRopeGm_[row * kRopeDim_], kRopeExt1, dataPad); } \
                                ++copyCount; \
                            } else if (mm == -2) { \
                                int64_t off = static_cast<int64_t>(fullL.GetValue(loc)); \
                                AscendC::DataCopyPad(t[(bb) * kvDim_], fullKvGm_[off * kvDim_], kvExt1, dataPad); \
                                if constexpr (!Packed) { AscendC::DataCopyPad(t[kRopeUbOffset_ + (bb) * kRopeDim_], \
                                                     fullKRopeGm_[off * kRopeDim_], kRopeExt1, dataPad); } \
                                ++copyCount; \
                            } else if (mm >= 0 && enableSlotsContiguous_) { \
                                int32_t ws = idxLocal.GetValue(2 * alignedNel_ + (loc)); \
                                if (ws >= 0) { \
                                    int32_t curS = mm - curStart_; \
                                    slotBuf.SetValue(static_cast<uint32_t>(curS), \
                                                     static_cast<int32_t>(rowBaseOff + ws)); \
                                } \
                            } \
                        } while (0)
                    for (uint32_t b = 0; b < curBatch; b += 4) {
                        int32_t m0 = maskL.GetValue(base + b);
                        int32_t m1 = maskL.GetValue(base + b + 1);
                        int32_t m2 = maskL.GetValue(base + b + 2);
                        int32_t m3 = maskL.GetValue(base + b + 3);
                        GBCI_COPY_ORIG(b,     m0, base + b);
                        GBCI_COPY_ORIG(b + 1, m1, base + b + 1);
                        GBCI_COPY_ORIG(b + 2, m2, base + b + 2);
                        GBCI_COPY_ORIG(b + 3, m3, base + b + 3);
                        copyCount = 1;
                    }
                    #undef GBCI_COPY_ORIG
                }

                dataQue_.EnQue(t);
                t = dataQue_.DeQue<T>();

                // pass2（写 selection KV）
                if (enableSlotsContiguous_) {
                    // 连续模式：t 整块一次拷出。ws 连续（range(0,valid_kv)+尾部-1）+ page id 连续，
                    //   目标偏移 = (rowBaseOff+base)*dim 线性，无需逐单元查 ws/页表/段合并。
                    //   current/skip 位在 t 中为 UB 垃圾，写到其 selection 槽（不在 golden[3]，无害）。
                    if (copyCount > 0) {
                        int64_t akv = (rowBaseOff + base) * kvDim_;
                        int64_t akr = (rowBaseOff + base) * kRopeDim_;
                        AscendC::DataCopyExtParams kvE{1, curBatch * kvDim_ * static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
                        AscendC::DataCopyExtParams krE{1, curBatch * kRopeDim_ * static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
                        AscendC::DataCopyPad(selKvOutGm_[akv], t[0], kvE);
                        if constexpr (!Packed) { AscendC::DataCopyPad(selKRopeOutGm_[akr], t[kRopeUbOffset_], krE); }
                    }
                } else {
                    // 通用模式：页表查找 + 段合并 + RegisterCurWriteSlot（当前 token 登记）
                    uint32_t jj = 0;
                    while (jj < curBatch) {
                        int32_t m = maskL.GetValue(base + jj);
                        if (m == -3) { ++jj; continue; }
                        if (m >= 0) {
                            int32_t ws = idxLocal.GetValue(2 * alignedNel_ + base + jj);
                            if (ws >= 0) {
                                RegisterCurWriteSlot(slotBuf, rowBase + base + jj, m, ws);
                            }
                            ++jj;
                            continue;
                        }
                        int32_t ws = idxLocal.GetValue(2 * alignedNel_ + base + jj);
                        if (ws < 0) { ++jj; continue; }
                        // ===== KEEP IN SYNC: update_cache_impl.h PhaseRowMoveFinal =====
                        int64_t gu = static_cast<int64_t>(startUnit_ + rowBase + base + jj);
                        int64_t rb = gu / unitsPerRealBatch_;
                        int64_t rem = gu % unitsPerRealBatch_;
                        int64_t s = rem / topk_;
                        int64_t row = rb * seq_ + s;
                        int64_t lp = (int64_t)ws * selTopKBlockSize_;
                        int64_t blk = lp / selKvBlockSize_;
                        int64_t off = lp % selKvBlockSize_;
                        int32_t pb = selKvBlockTableGm_.GetValue(row * selMaxBlockNum_ + blk);
                        int64_t akv = (static_cast<int64_t>(pb) * selKvBlockSize_ + off) * kvDim_;
                        int64_t akr = (static_cast<int64_t>(pb) * selKvBlockSize_ + off) * kRopeDim_;
                        // ==============================================================
                        uint32_t seg = 1;
                        while (jj + seg < curBatch) {
                            int32_t mN = maskL.GetValue(base + jj + seg);
                            if (mN != -1 && mN != -2) { break; }
                            int32_t wsN = idxLocal.GetValue(2 * alignedNel_ + base + jj + seg);
                            if (wsN != ws + (int32_t)seg) { break; }
                            int64_t guN = static_cast<int64_t>(startUnit_ + rowBase + base + jj + seg);
                            int64_t sN = (guN % unitsPerRealBatch_) / topk_;
                            int64_t rowN = (guN / unitsPerRealBatch_) * seq_ + sN;
                            if (rowN != row) { break; }
                            if ((uint32_t)(off + seg) >= (uint32_t)selKvBlockSize_) { break; }
                            ++seg;
                        }
                        AscendC::DataCopyExtParams kvE{1, seg * kvDim_ * static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
                        AscendC::DataCopyExtParams krE{1, seg * kRopeDim_ * static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
                        AscendC::DataCopyPad(selKvOutGm_[akv], t[jj * kvDim_], kvE);
                        if constexpr (!Packed) { AscendC::DataCopyPad(selKRopeOutGm_[akr], t[kRopeUbOffset_ + jj * kRopeDim_], krE); }
                        jj += seg;
                    }
                }

                dataQue_.FreeTensor(t);
            }

            // ---- 行落盘：-1 初始化 + 全部槽位登记 → GM（MTE3 整行写，可靠落盘）----
            // SetValue 走标量 pipe，MTE3 读前需 S_MTE3（gen 同款）；V 侧（行首 Duplicate）需 V_MTE3。
            FlushCurSlots(slotBuf, startUnit_ / topk_ + ri);
            // Only fully published historical records from this core's first
            // query are reusable. Never cache current-token or padding records.
            if (enableSlotsContiguous_ == 2 && !reuseReady_ && topk_ <= 2048 &&
                ri + 1 < rowCount && (startUnit_ + (ri + 1) * topk_) / unitsPerRealBatch_ == curBatch_ &&
                rowBaseOff + rowUnits < 2147483647LL) {
                GbciSyncEvent<AscendC::HardEvent::MTE3_MTE2>();
                BuildCompletedSelection(idxLocal, static_cast<int32_t>(rowBaseOff), rowUnits);
            }
            idxQue_.FreeTensor(idxLocal);                  // 归还索引缓冲，下轮 AllocTensor 等待后复用
        }
    }

private:
    __aicore__ inline void FlushCurSlots(AscendC::LocalTensor<int32_t>& slots, uint32_t row)
    {
        GbciSyncEvent<AscendC::HardEvent::S_MTE3>();
        GbciSyncEvent<AscendC::HardEvent::V_MTE3>();
        AscendC::DataCopyExtParams cp{1, seq_ * static_cast<uint32_t>(sizeof(int32_t)), 0, 0, 0};
        AscendC::DataCopyPad(curWriteSlotsGm_[static_cast<uint64_t>(row) * seq_], slots, cp);
        GbciSyncEvent<AscendC::HardEvent::MTE3_V>();
    }

    // Mode 2 has no use for from_idx, so reuse that segment as vector scratch.
    // Find the last nonzero source mask word, retaining holes and original
    // selection positions. This is safe for unsorted IDs and causal zeros too.
    __aicore__ inline uint32_t UnifiedRowExtent(AscendC::LocalTensor<int32_t> indices)
    {
        auto values = indices.template ReinterpretCast<float>();
        auto mask = cmpU8A_.Get<uint8_t>();
        AscendC::Cast(values, indices[2 * alignedNel_], AscendC::RoundMode::CAST_ROUND, topk_);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::CompareScalar(mask, values, 0.0f, AscendC::CMPMODE::NE, topk_);
        GbciSyncEvent<AscendC::HardEvent::V_S>();
        auto words = mask.template ReinterpretCast<uint32_t>();
        for (uint32_t n = topk_ / 32; n > 0; --n) {
            if (words.GetValue(n - 1) != 0) { return n * 32; }
        }
        return 0;
    }

    // Mode 2 does not use its six vector-precompute arrays. Reuse two of those
    // allocations for a private 4096-entry hash, shrinking the other four.
    // This reduces rather than increases total UB at topk=2048. No new GM
    // workspace or inter-core synchronization is required.
    __aicore__ inline int32_t FindCompletedSelection(int32_t source)
    {
        if (!reuseReady_ || reuseEntries_ == 0) { return -1; }
        auto keys = pageBuf_.Get<int32_t>();
        auto vals = poffBuf_.Get<int32_t>();
        uint32_t slot = (static_cast<uint32_t>(source) * 2654435761U) & 4095U;
        for (uint32_t probe = 0; probe < 4096; ++probe) {
            int32_t key = keys.GetValue(slot);
            if (key == source) { return vals.GetValue(slot); }
            if (key == 0) { return -1; }
            slot = (slot + 1U) & 4095U;
        }
        return -1;
    }

    __aicore__ inline void BuildCompletedSelection(AscendC::LocalTensor<int32_t> idx, int32_t rowOffset,
                                                   uint32_t rowUnits)
    {
        auto keys = pageBuf_.Get<int32_t>();
        auto vals = poffBuf_.Get<int32_t>();
        GbciSyncEvent<AscendC::HardEvent::S_V>();
        AscendC::Duplicate(keys, int32_t(0), 4096);
        GbciSyncEvent<AscendC::HardEvent::V_S>();
        for (uint32_t i = 0; i < rowUnits; ++i) {
            int32_t source = idx.GetValue(2 * alignedNel_ + i);
            if (source >= 0) { continue; }
            uint32_t slot = (static_cast<uint32_t>(source) * 2654435761U) & 4095U;
            for (uint32_t probe = 0; probe < 4096; ++probe) {
                int32_t key = keys.GetValue(slot);
                if (key == source) { break; }
                if (key == 0) {
                    keys.SetValue(slot, source);
                    vals.SetValue(slot, rowOffset + static_cast<int32_t>(i));
                    ++reuseEntries_;
                    break;
                }
                slot = (slot + 1U) & 4095U;
            }
        }
        reuseReady_ = true;
    }
    uint32_t reuseBatch_ = 0xFFFFFFFFU;
    uint32_t reuseEntries_ = 0;
    bool reuseReady_ = false;
    // 每 batch 刷新一次的行级状态：
    //   ① seq range：curStart_/curEnd_（当前 token 区间，S 管标量 GM 读 kvActSeq/qActSeq）；
    //   ② full_kv_block_table 行缓存到 UB btTableBuf_（MTE2 DataCopy，V_MTE2/MTE2_V 同步）。
    // 两者都只随 real batch 变化，行对齐切核 + unitsPerRealBatch 为 topk_ 整数倍 → 整行同属
    // 一个 real batch，故每行只刷新一次。同一 batch 跨行复用 curBatch_ 缓存。
    // 同步顺序：先 V_MTE2（等上轮 V 侧 Gather 读完 btTableBuf_ 才允许覆写）→ DataCopy 页表 →
    // MTE2_V（搬完才让本轮 V 侧 Gather 读）。seq range 的 S 管标量 GM 读与 MTE2 无依赖，
    // 放在 DataCopy 前后均可，这里放在最前以尽早拿到 curStart_/curEnd_ 供预计算用。
    __aicore__ inline void RefreshBatchState(uint32_t globalUnit)
    {
        uint32_t rb = globalUnit / unitsPerRealBatch_;
        if (rb == curBatch_) {
            return;
        }
        curBatch_ = rb;
        // ① seq range（S 管标量 GM 读，与 MTE2 无事件依赖）
        int32_t kvLen = kvActSeqGm_.GetValue(rb);
        int32_t qLen = qActSeqGm_.GetValue(rb);
        curQlen_ = qLen;
        curEnd_ = kvLen;
        curStart_ = kvLen - qLen;
        // ② 页表行缓存到 UB（MTE2 DataCopy，需同步事件保护 btTableBuf_ 不被 V 侧覆写/读越）
        GbciSyncEvent<AscendC::HardEvent::V_MTE2>();
        AscendC::DataCopyExtParams btExt{1, fullMaxBlockNum_ * static_cast<uint32_t>(sizeof(int32_t)),
                                         0, 0, 0};
        AscendC::DataCopyPadExtParams<int32_t> btPad{false, 0, 0, 0};
        AscendC::DataCopyPad(btTableBuf_.Get<int32_t>(),
                             fullKvBlockTableGm_[static_cast<uint64_t>(rb) * fullMaxBlockNum_], btExt, btPad);
        GbciSyncEvent<AscendC::HardEvent::MTE2_V>();
    }

    // 行级向量预计算（循环2之前调用一次，每行 topk_ 个 unit 一次算齐）：
    //   lruIdxBuf_  = hit 的 lru cache 行号（元素单位，循环2内再 *dim）：rb*cacheRows_ + fromIdx
    //   fullIdxBuf_ = miss 的 full cache 块内元素偏移（循环2内再 *dim）：pb*fullKvBlockSize_ + poff，
    //                 pb = fullKvBlockTableGm_[rb, page] 经 UB 页表行（btTableBuf_，按 batch 缓存）Gather 得到。
    //   maskBuf_    = topkIdxMask：0=跳过（topk==-1 / 当前 token 区间 [curStart_,curEnd_) / fromIdx==-2）；
    //                 1=hit（fromIdx>=0）；2=miss（fromIdx==-1）
    //   lruIdxBuf_  = hit 的 lru cache 行号（元素单位，循环2内再 *dim）：rb*cacheRows_ + fromIdx
    //   fullIdxBuf_ = miss 的 full cache 块内元素偏移（循环2内再 *dim）：pb*fullKvBlockSize_ + poff，
    //                 pb = fullKvBlockTableGm_[rb, page] 经 UB 页表行（btTableBuf_，按 batch 缓存）Gather 得到。
    // 910b 位掩码算子约束：Compare/Compares→u8 bit-mask(1bit/元素)；And/Or→int16 位运算，count 为
    //   int16 元素数；Select→按 bit 读 sel。三者均要求 count*sizeof(T)=256B（整 256bit 块）且
    //   LocalTensor 起始 32B 对齐。故 alignedNel_ 对齐 64（int32/float count=64→256B），And/Or 用
    //   maskM16_(128 整数倍 int16 → 256B 块)。标量比较走 Duplicate(scalar→tensor)+Compare(tensor-tensor)
    //   （gen 同款，已验证）。And/Or count=maskM16_（非 ceil(n/16)）。
    // 910b 说明：除法走 float 域（Div 支持 half/float；Divs 仅 Atlas 350 支持，A2/A3 不可用）：
    //           Cast int32→float(CAST_ROUND) → Duplicate(bs) → Div → Cast float→int32(CAST_ROUND)。
    //           topk==-1 → lp=-selTopKBlockSize_ → page≈-0.03→ROUND→0，Maxs 兜底防页表越界。
    // 偏移按 int32 元素计，需保证 rb*cacheRows_ 与 pb*fullKvBlockSize_+poff < 2^31（当前量级远小于）。
    __aicore__ inline void PrecomputeRowVectors(AscendC::LocalTensor<int32_t>& idxLocal, uint32_t rb)
    {
        AscendC::LocalTensor<int32_t> topkRow = idxLocal[alignedNel_];
        AscendC::LocalTensor<int32_t> fromRow = idxLocal;
        AscendC::LocalTensor<uint8_t> cmpA = cmpU8A_.Get<uint8_t>();
        AscendC::LocalTensor<uint8_t> cmpB = cmpU8B_.Get<uint8_t>();
        AscendC::LocalTensor<uint8_t> cmpC = cmpU8C_.Get<uint8_t>();
        AscendC::LocalTensor<int32_t> maskL = maskBuf_.Get<int32_t>();
        AscendC::LocalTensor<int32_t> lruL = lruIdxBuf_.Get<int32_t>();
        AscendC::LocalTensor<int32_t> fullL = fullIdxBuf_.Get<int32_t>();
        AscendC::LocalTensor<int32_t> pageL = pageBuf_.Get<int32_t>();
        AscendC::LocalTensor<int32_t> pbL = pbBuf_.Get<int32_t>();
        AscendC::LocalTensor<int32_t> btL = btTableBuf_.Get<int32_t>();
        const uint32_t n = alignedNel_;

        // 对齐到 64 后 [topk_..n] 是 DataCopyPad 未填尾巴，位掩码算子会扫到 → 填安全值
        // （topk=-1、from=-2 → mask=0 跳过），避免 UB 垃圾被当有效元素算出随机 mask。
        if (n > topk_) {
            AscendC::Duplicate(topkRow[topk_], static_cast<int32_t>(-1), n - topk_);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Duplicate(fromRow[topk_], static_cast<int32_t>(-2), n - topk_);
            AscendC::PipeBarrier<PIPE_V>();
        }

        // ---- ① topkIdxMask ----
        // 新语义（pass1/pass2 据此分支）：
        //   -3 = 跳过（topk==-1 或 fromIdx==-2）
        //   topkIdx(>=0) = 当前 token（curStart_<=topk<curEnd_）——存原 topk 值，pass2 直接当 topkRow
        //   -1 = hit（fromIdx>=0）；-2 = miss（fromIdx==-1）
        // 优先级：topk==-1(-3) > inRange(topkIdx) > from==-2(-3) > from>=0(-1) > else(-2)。
        //   topk==-1 必不在 inRange（curStart>=0），故二者互斥；from==-2 命中 inRange 时取 inRange。
        // 910b：Compare int32 仅 EQ，须 Cast int32→float 后 Compare float（模式才生效）。
        AscendC::LocalTensor<int16_t> zI = cmpA.template ReinterpretCast<int16_t>();   // inRange
        AscendC::LocalTensor<int16_t> tI = cmpB.template ReinterpretCast<int16_t>();   // skipMask/d/eqNeg1
        AscendC::LocalTensor<int16_t> uI = cmpC.template ReinterpretCast<int16_t>();   // eqNeg2
        AscendC::LocalTensor<float> maskF = maskBuf_.Get<float>();       // 最终 mask（float→Cast int32）
        AscendC::LocalTensor<float> tmpF = pageL.template ReinterpretCast<float>();     // d?-1:-2
        AscendC::LocalTensor<float> overrideF = pbL.template ReinterpretCast<float>();  // 非 inRange 覆盖值
        AscendC::LocalTensor<float> scalarF = poffBuf_.Get<float>();
        AscendC::LocalTensor<float> topkRowF = lruIdxBuf_.Get<float>();   // ② 在后复用 lruL int32
        AscendC::LocalTensor<float> fromRowF = fullIdxBuf_.Get<float>();  // ③ 在后复用 fullL int32
        AscendC::Cast(topkRowF, topkRow, AscendC::RoundMode::CAST_ROUND, n);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(fromRowF, fromRow, AscendC::RoundMode::CAST_ROUND, n);
        AscendC::PipeBarrier<PIPE_V>();
        // inRange = (topk>=curStart_) & (topk<curEnd_) → cmpA
        AscendC::Duplicate(scalarF, static_cast<float>(curStart_), n);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Compare(cmpA, topkRowF, scalarF, AscendC::CMPMODE::GE, n);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Duplicate(scalarF, static_cast<float>(curEnd_), n);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Compare(cmpB, topkRowF, scalarF, AscendC::CMPMODE::LT, n);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::And(zI, zI, tI, maskM16_);                // cmpA = inRange
        AscendC::PipeBarrier<PIPE_V>();
        // d = fromIdx>=0 → cmpB；tmpF = d ? -1 : -2
        AscendC::Duplicate(scalarF, 0.0f, n);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Compare(cmpB, fromRowF, scalarF, AscendC::CMPMODE::GE, n);  // d
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Duplicate(scalarF, static_cast<float>(-1), n);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Select(tmpF, cmpB, scalarF, static_cast<float>(-2),
                        AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, n);       // d?-1:-2
        AscendC::PipeBarrier<PIPE_V>();
        // skipMask = (topk==-1) | (fromIdx==-2) → cmpB | cmpC
        AscendC::Duplicate(scalarF, static_cast<float>(-1), n);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Compare(cmpB, topkRowF, scalarF, AscendC::CMPMODE::EQ, n);  // eqNeg1 = topk==-1
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Duplicate(scalarF, static_cast<float>(-2), n);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Compare(cmpC, fromRowF, scalarF, AscendC::CMPMODE::EQ, n);  // eqNeg2 = fromIdx==-2
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Or(tI, tI, uI, maskM16_);                 // cmpB = skipMask = eqNeg1 | eqNeg2
        AscendC::PipeBarrier<PIPE_V>();
        // overrideF = skipMask ? -3 : tmpF（非 inRange 的值）
        AscendC::Duplicate(scalarF, static_cast<float>(-3), n);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Select(overrideF, tI, scalarF, tmpF, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, n);
        AscendC::PipeBarrier<PIPE_V>();
        // maskF = inRange ? topkRowF : overrideF
        AscendC::Select(maskF, zI, topkRowF, overrideF, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, n);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(maskL, maskF, AscendC::RoundMode::CAST_ROUND, n);
        AscendC::PipeBarrier<PIPE_V>();

        // ---- ② lru_cache_idx（hit）：cache 行号 = rb*cacheRows_ + fromIdx（fromIdx 即行内槽偏移，无需乘 kvDim_）----
        AscendC::Adds(lruL, fromRow, static_cast<int32_t>(rb * cacheRows_), n);
        AscendC::PipeBarrier<PIPE_V>();

        // ---- ③ full_cache_idx（miss）：lp → page/poff → Gather(pb) → pb*bs+poff ----
        // 除法走 float 域（910b：Divs 仅 Atlas 350 支持，A2/A3 不可用；Div 支持 half/float）：
        // int32→float(CAST_ROUND) → Duplicate(bs 张量) → Div → float→int32(CAST_ROUND)。
        // |lp/bs| < 2^23（topk≤2048 量级）时 float 误差远小于 0.5，取整不翻界；topk==-1 → -0.03→ROUND→0，Maxs 兜底。
        // poff 用独立缓冲（poffBuf_），page 不被覆盖，逐指令语义单一：
        //   pageL = page，poffL = lp - page*bs，索引 = page*4 → Gather(pb)，fullL = poff + pb*bs。
        AscendC::LocalTensor<float> pageF = pageL.template ReinterpretCast<float>();
        AscendC::LocalTensor<float> pbF = pbL.template ReinterpretCast<float>();
        AscendC::LocalTensor<int32_t> poffL = poffBuf_.Get<int32_t>();
        AscendC::Muls(fullL, topkRow, static_cast<int32_t>(selTopKBlockSize_), n);  // fullL = lp
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(pageF, fullL, AscendC::RoundMode::CAST_ROUND, n);            // pageF = (float)lp
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Duplicate(pbF, static_cast<float>(static_cast<int32_t>(fullKvBlockSize_)), n);  // bs 张量
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Div(pageF, pageF, pbF, n);                                        // pageF = (float)page
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(pageL, pageF, AscendC::RoundMode::CAST_FLOOR, n);           // pageL = floor(lp/bs)（整除取下界，ROUND 会把 0.99→1 翻页）
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Maxs(pageL, pageL, 0, n);              // 防 topk==-1 → 负 page 越界
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(poffL, pageL, static_cast<int32_t>(fullKvBlockSize_), n);    // poffL = page*bs
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(poffL, poffL, -1, n);             // poffL = -page*bs
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add(poffL, fullL, poffL, n);           // poffL = lp - page*bs = poff（块内元素偏移）
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(pageL, pageL, static_cast<int32_t>(sizeof(int32_t)), n);     // pageL = page*4 字节偏移
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Gather(pbL, btL, pageL.template ReinterpretCast<uint32_t>(), 0, n);  // pbL = pb
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(pbL, pbL, static_cast<int32_t>(fullKvBlockSize_), n);        // pbL = pb*bs
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add(fullL, poffL, pbL, n);             // fullL = poff + pb*bs
        AscendC::PipeBarrier<PIPE_V>();
    }

    // 通用模式（enableSlotsContiguous_==0）：查 selKvBlockTableGm_ 页表算物理槽，登记 cur_kv_write_slots。
    __aicore__ inline void RegisterCurWriteSlot(AscendC::LocalTensor<int32_t>& slotBuf,
                                                uint32_t unitLocal, int32_t topkRow, int32_t ws)
    {
        uint64_t gu = static_cast<uint64_t>(startUnit_) + unitLocal;
        uint64_t rb = gu / unitsPerRealBatch_;
        uint64_t rem = gu % unitsPerRealBatch_;
        uint64_t s = rem / topk_;
        int64_t row = static_cast<int64_t>(rb * seq_ + s);
        int64_t lp = static_cast<int64_t>(ws) * selTopKBlockSize_;
        int64_t blk = lp / selKvBlockSize_;
        int64_t off = lp % selKvBlockSize_;
        int32_t pb = selKvBlockTableGm_.GetValue(row * selMaxBlockNum_ + blk);
        int64_t targetSlot = static_cast<int64_t>(pb) * selKvBlockSize_ + off;
        int64_t currentS = static_cast<int64_t>(topkRow) - curStart_;
        slotBuf.SetValue(static_cast<uint32_t>(currentS), static_cast<int32_t>(targetSlot));
    }

    AscendC::TPipe* pipe_;

    AscendC::GlobalTensor<T> cacheKRopeGm_, cacheKvGm_;
    AscendC::GlobalTensor<T> fullKRopeGm_, fullKvGm_;
    AscendC::GlobalTensor<T> selKRopeOutGm_, selKvOutGm_;
    AscendC::GlobalTensor<int32_t> fromIdxGm_, topkIdxGm_;
    AscendC::GlobalTensor<int32_t> fullKvBlockTableGm_, selKvBlockTableGm_;
    AscendC::GlobalTensor<int32_t> writeSlotGm_;
    AscendC::GlobalTensor<int32_t> kvActSeqGm_, qActSeqGm_;
    AscendC::GlobalTensor<int32_t> curWriteSlotsGm_;

    AscendC::TQueBind<AscendC::TPosition::VECIN, AscendC::TPosition::VECOUT, GBCI_DOUBLE_BUFFER> dataQue_;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> idxQue_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> curSlotBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> maskBuf_, lruIdxBuf_, fullIdxBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> pageBuf_, poffBuf_, pbBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> cmpU8A_, cmpU8B_, cmpU8C_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> btTableBuf_;

    uint32_t blkIdx_;
    bool valid_;
    uint32_t startUnit_, blockNum_, batchCount_;
    uint32_t kRopeDim_, kvDim_, P_;
    uint32_t cacheRows_, unitsPerRealBatch_;
    uint32_t kRopeUbOffset_;
    uint32_t alignedNel_;
    uint32_t curSlotBufElems_;
    uint32_t maskBytes_;   // 位掩码缓冲字节数（≥256，256 整数倍）
    uint32_t maskM16_;     // 位掩码 And/Or 的 int16 元素数（maskBytes_/2，128 整数倍）
    // 分页
    uint32_t seq_, topk_, selTopKBlockSize_;
    uint32_t fullKvBlockSize_, fullMaxBlockNum_, fullKvBlockNum_;
    uint32_t selKvBlockSize_, selMaxBlockNum_;
    uint32_t enableSlotsContiguous_;   // 1=page id 连续，走线性偏移
    // 当前 MTP token 区间（每 batch 缓存）
    int32_t curStart_, curEnd_, curQlen_;
    // 当前 real batch：seq range（curStart_/curEnd_）与 fullKvBlockTable 行缓存共用，
    // 同一 batch 下两者都只需算/搬一次，跨行复用。
    uint32_t curBatch_;
};

} // namespace GatherByCacheIndicesNs

#endif // GATHER_BY_CACHE_INDICES_IMPL_H_
