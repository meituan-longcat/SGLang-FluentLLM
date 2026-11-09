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
 * \file update_cur_kv_to_selection_impl.h
 * \brief UpdateCurKvToSelection kernel 实现
 *        压缩单元遍历：idx 批载入 → 源行按行搬入 UB（padded pitch）→
 *        物理扁平 slot 定位 + 连续段合并 → DataCopyPad scatter 写（就地）。
 *        - 单元段 [startUnit, endUnit) 循环，chunk = P 不跨 (b,s) 行（源行连续）
 *        - cur_kv_write_slots[b,s,j] 已由上游 gather_by_cache_indices 完成页表翻译，
 *          值为物理扁平 slot = physical_page * PAGE_SIZE + page_offset，
 *          本算子直接定位，不再查询 selection_kv_block_table（二次翻译已移除）
 *        - kv_pos < 0 跳过；kv_pos >= totalPageNum*PAGE_SIZE 跳过（OOB 防御）
 *        - 连续段合并仅当行宽 32B 对齐（tight pitch）且 kvPos 连续、不跨 128 行边界
 *        - kernel_done 不写
 */
#ifndef UPDATE_CUR_KV_TO_SELECTION_IMPL_H_
#define UPDATE_CUR_KV_TO_SELECTION_IMPL_H_

#include "kernel_operator.h"

namespace UpdateCurKvToSelectionNs {

constexpr uint32_t PAGE_SIZE = 128;             // 物理页行数（编译期常量）
constexpr uint32_t DB = 2;                      // 数据双缓冲（单缓冲退化时运行时传 1）

template <typename U>
__aicore__ inline U UcktsCeilDiv(U a, U b)
{
    return (a + b - 1) / b;
}

template <typename T, bool Packed = false>
class KernelUpdateCurKvToSelection {
public:
    __aicore__ inline KernelUpdateCurKvToSelection(AscendC::TPipe* pipe)
    {
        pipe_ = pipe;
    }

    template <typename TilingData>
    __aicore__ inline void Init(GM_ADDR cur_k_nope, GM_ADDR cur_k_rope, GM_ADDR cur_kv_write_slots,
                                GM_ADDR selection_k_nope, GM_ADDR selection_k_rope,
                                GM_ADDR selection_kv_block_table, GM_ADDR kernel_done,
                                const TilingData* tiling)
    {
        blkIdx_ = AscendC::GetBlockIdx();
        batch_ = tiling->batch;
        seq_ = tiling->seq;
        kNopeDim_ = tiling->kNopeDim;
        kRopeDim_ = tiling->kRopeDim;
        P_ = tiling->unitsPerBatch;
        maxPagePerReq_ = tiling->maxPagePerReq;
        totalPageNum_ = tiling->totalPageNum;
        nopeRowPitchElems_ = tiling->nopeRowPitchElems;
        ropeRowPitchElems_ = tiling->ropeRowPitchElems;
        rowAlignedNope_ = (tiling->rowAlignedNope == 1);
        rowAlignedRope_ = (tiling->rowAlignedRope == 1);
        (void)kernel_done;   // 仅图返回值标记，kernel 不修改其值

        if (blkIdx_ >= tiling->blockNum) {
            valid_ = false;
            return;
        }
        valid_ = true;

        // 单元段切分（与参考算子同构）：numPerCore + tailUnits 均衡
        uint32_t numPerCore = tiling->numPerCore;
        uint32_t tailUnits = tiling->tailUnits;
        if (blkIdx_ < tailUnits) {
            count_ = numPerCore + 1;
            startUnit_ = blkIdx_ * (numPerCore + 1);
        } else {
            count_ = numPerCore;
            startUnit_ = tailUnits * (numPerCore + 1) + (blkIdx_ - tailUnits) * numPerCore;
        }
        endUnit_ = startUnit_ + count_;

        curNopeGm_.SetGlobalBuffer((__gm__ T*)cur_k_nope);
        curRopeGm_.SetGlobalBuffer((__gm__ T*)cur_k_rope);
        slotsGm_.SetGlobalBuffer((__gm__ int32_t*)cur_kv_write_slots);
        selNopeGm_.SetGlobalBuffer((__gm__ T*)selection_k_nope);
        selRopeGm_.SetGlobalBuffer((__gm__ T*)selection_k_rope);
        // ABI compatibility only: selection_kv_block_table 输入保留（接口/图兼容），
        // cur_kv_write_slots 已含物理扁平 slot（上游 gather 完成页表翻译），kernel 不再读取 block table。
        (void)selection_kv_block_table;
        blockTableGm_.SetGlobalBuffer((__gm__ int32_t*)selection_kv_block_table);

        // 数据缓冲：双缓冲（P_ub≥1）或单缓冲退化（P_ub<1，reserved==1）
        kRopeUbOffset_ = P_ * nopeRowPitchElems_;
        uint32_t db = (tiling->reserved == 1U) ? 1U : DB;
        uint32_t bindBytes = P_ * (nopeRowPitchElems_ + ropeRowPitchElems_) * sizeof(T);
        pipe_->InitBuffer(dataQue_, db, bindBytes);

        // idx 缓冲（单缓冲，仅 GetValue 标量读）：alignedNel 保证 32B 对齐
        alignedNel_ = UcktsCeilDiv<uint32_t>(P_, 8U) * 8U;
        if (alignedNel_ == 0) {
            alignedNel_ = 8U;
        }
        pipe_->InitBuffer(idxQue_, 1, alignedNel_ * sizeof(int32_t));
    }

    __aicore__ inline void Process()
    {
        if (!valid_ || count_ == 0) {
            return;   // 空任务：无任何写入（spec boundary case1）
        }

        AscendC::LocalTensor<int32_t> idxLocal = idxQue_.AllocTensor<int32_t>();

        for (uint32_t base = 0; base < count_;) {
            uint32_t u0 = startUnit_ + base;                 // 本 chunk 首全局单元
            uint32_t b = u0 / (seq_ * seq_);                 // chunk 不跨 (b,s) 行 ⇒ 单次分解
            uint32_t rem = u0 % (seq_ * seq_);
            uint32_t s = rem / seq_;
            uint32_t j0 = rem % seq_;
            (void)s;   // s 不再参与地址计算（物理扁平 slot 寻址），保留 (b,s,j) 语义分解
            uint32_t cur = P_;
            if (cur > seq_ - j0) {
                cur = seq_ - j0;
            }
            if (cur > count_ - base) {
                cur = count_ - base;
            }

            // (1) idx 批载入：slots[b, s, j0 .. j0+cur)（MTE2）
            AscendC::DataCopyExtParams idxExt{1, cur * static_cast<uint32_t>(sizeof(int32_t)), 0, 0, 0};
            AscendC::DataCopyPadExtParams<int32_t> idxPad{false, 0, 0, 0};
            AscendC::DataCopyPad(idxLocal, slotsGm_[u0], idxExt, idxPad);
            idxQue_.EnQue(idxLocal);
            idxLocal = idxQue_.DeQue<int32_t>();   // MTE2→V
            auto event = pipe_->FetchEventID(AscendC::HardEvent::V_S);
            AscendC::SetFlag<AscendC::HardEvent::V_S>(event);
            AscendC::WaitFlag<AscendC::HardEvent::V_S>(event);

            // (2) 源行批载入（nope + rope，逐行 blockCount=1，padded pitch 保证 UB 32B 对齐）
            AscendC::DataCopyPadExtParams<T> dataPad{false, 0, 0, 0};
            AscendC::LocalTensor<T> t = dataQue_.AllocTensor<T>();
            for (uint32_t r = 0; r < cur; ++r) {
                int64_t srcRow = (int64_t)b * seq_ + (int64_t)j0 + r;   // 源行由 j 直接定位，chunk 内连续
                AscendC::DataCopyExtParams nopeExt{1, kNopeDim_ * static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
                AscendC::DataCopyPad(t[r * nopeRowPitchElems_], curNopeGm_[srcRow * kNopeDim_], nopeExt, dataPad);
                AscendC::DataCopyExtParams ropeExt{1, kRopeDim_ * static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
                if constexpr (!Packed) { AscendC::DataCopyPad(t[kRopeUbOffset_ + r * ropeRowPitchElems_], curRopeGm_[srcRow * kRopeDim_],
                                     ropeExt, dataPad); }
            }
            dataQue_.EnQue(t);
            t = dataQue_.DeQue<T>();   // MTE2→MTE3 同步点

            // (3) scatter 写（压缩单元遍历 + 连续段合并）
            // cur_kv_write_slots[b,s,j] 已是物理扁平 slot（上游 gather 完成页表翻译）：
            //   physicalSlot = physical_page * PAGE_SIZE + page_offset，
            //   直接 GM 定位 dst = physicalSlot * dim，不再查询 selection_kv_block_table。
            const int64_t totalPhysicalSlots = static_cast<int64_t>(totalPageNum_) * PAGE_SIZE;
            uint32_t jj = 0;
            while (jj < cur) {
                int32_t kvPos = idxLocal.GetValue(jj);      // 物理扁平 slot（UB 标量读）
                if (kvPos < 0) {
                    ++jj;                                   // 无效槽：跳过，不产生写入
                    continue;
                }
                // OOB 防御：物理 slot 超界（totalPageNum_*PAGE_SIZE）跳过，避免 GM 越界写
                int64_t physicalSlot = static_cast<int64_t>(kvPos);
                if (physicalSlot >= totalPhysicalSlots) {
                    ++jj;
                    continue;
                }
                int64_t dstNope = physicalSlot * kNopeDim_;
                int64_t dstRope = physicalSlot * kRopeDim_;

                // 连续段合并：仅当行宽 32B 对齐（tight pitch，UB 源连续）且下一单元有效、
                // kvPos 连续（+seg）、不跨 128 行边界（保守限制，物理 slot 连续即目标连续）
                uint32_t seg = 1;
                if (rowAlignedNope_ && rowAlignedRope_) {
                    int32_t off = kvPos & (PAGE_SIZE - 1);
                    while (jj + seg < cur) {
                        int32_t nextPos = idxLocal.GetValue(jj + seg);
                        if (nextPos < 0 ||
                            static_cast<int64_t>(nextPos) >= totalPhysicalSlots ||
                            nextPos != kvPos + static_cast<int32_t>(seg)) {
                            break;   // 无效/OOB/不连续：断开合并
                        }
                        if (off + static_cast<int32_t>(seg) >= static_cast<int32_t>(PAGE_SIZE)) {
                            break;   // 跨 128 行边界：保守断开（重新定位，不影响正确性）
                        }
                        ++seg;
                    }
                }
                AscendC::DataCopyExtParams nopeE{1, seg * kNopeDim_ * static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
                AscendC::DataCopyPad(selNopeGm_[dstNope], t[jj * nopeRowPitchElems_], nopeE);   // MTE3，UB→GM
                AscendC::DataCopyExtParams ropeE{1, seg * kRopeDim_ * static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
                if constexpr (!Packed) { AscendC::DataCopyPad(selRopeGm_[dstRope], t[kRopeUbOffset_ + jj * ropeRowPitchElems_], ropeE); }
                jj += seg;
            }

            dataQue_.FreeTensor(t);
            base += cur;
        }

        idxQue_.FreeTensor(idxLocal);
    }

private:
    AscendC::TPipe* pipe_;

    AscendC::GlobalTensor<T> curNopeGm_, curRopeGm_;
    AscendC::GlobalTensor<T> selNopeGm_, selRopeGm_;
    AscendC::GlobalTensor<int32_t> slotsGm_;
    AscendC::GlobalTensor<int32_t> blockTableGm_;   // ABI compatibility only（不再参与寻址）

    AscendC::TQueBind<AscendC::TPosition::VECIN, AscendC::TPosition::VECOUT, DB> dataQue_;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> idxQue_;

    uint32_t blkIdx_;
    bool valid_;
    uint32_t startUnit_, endUnit_, count_;
    uint32_t batch_, seq_;
    uint32_t kNopeDim_, kRopeDim_, P_;
    uint32_t maxPagePerReq_;   // ABI compatibility only（不再参与寻址）
    uint32_t totalPageNum_;    // 物理页总数：OOB 防御（slot < totalPageNum_*PAGE_SIZE）
    uint32_t nopeRowPitchElems_, ropeRowPitchElems_;
    uint32_t kRopeUbOffset_;
    uint32_t alignedNel_;
    bool rowAlignedNope_, rowAlignedRope_;
};

} // namespace UpdateCurKvToSelectionNs

#endif // UPDATE_CUR_KV_TO_SELECTION_IMPL_H_
