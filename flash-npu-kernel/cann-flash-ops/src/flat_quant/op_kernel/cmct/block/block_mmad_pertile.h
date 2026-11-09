/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file block_mmad_pertile.h
 * \brief
 */
#ifndef MATMUL_BLOCK_BLOCK_MMAD_PERTILE_H
#define MATMUL_BLOCK_BLOCK_MMAD_PERTILE_H

#include "../policy/dispatch_policy.h"
#include "../tile/tile_copy.h"
#include "../utils/layout_utils.h"
#include "../utils/quant_batch_matmul_constant.h"
#include "../utils/tuple_utils.h"
#include "block_mmad_pertile_param.h"
#include "block_scheduler_utils.h"
#include "lib/matmul/matmul.h"
#include "lib/matmul/tiling.h"

namespace Cmct {
namespace Gemm {
namespace Block {

struct PertileMmParam {
    bool fixpipeSplitN = false;
    uint64_t fixpipeN;
    uint64_t fixpipeM;
    uint64_t srcStride;
    uint64_t ndNum;
};

#define QBMM_BLOCK_MMAD_PERTILE_CLASS_LOCAL_PARAMS                                                                 \
    template <                                                                                                     \
        class AType_, class LayoutA_, class BType_, class LayoutB_, class CType_, class LayoutC_, class BiasType_, \
        class LayoutBias_, class L1TileShape_, class L0TileShape_, class TileCopyParam_>

#define QBMM_BLOCK_MMAD_PERTILE_FUNC_LOCAL_PARAMS                                                               \
    MmadCAccOnUb<>, AType_, LayoutA_, BType_, LayoutB_, CType_, LayoutC_, BiasType_, LayoutBias_, L1TileShape_, \
        L0TileShape_, TileCopyParam_

using namespace Cmct::Gemm::QuantBatchMatmul;

template <class BlockMatmulPolicy_, class AType_, class LayoutA_, class BType_, class LayoutB_, class CType_,
          class LayoutC_, class BiasType_, class LayoutBias_, class L1TileShape_, class L0TileShape_,
          class TileCopyParam_ = void>
class BlockMmadPertile {
    static_assert(AscendC::Std::always_false_v<BlockMatmulPolicy_>, "Should not be here!");
};

QBMM_BLOCK_MMAD_PERTILE_CLASS_LOCAL_PARAMS
class BlockMmadPertile<QBMM_BLOCK_MMAD_PERTILE_FUNC_LOCAL_PARAMS> {
public:
    using AType = AType_;
    using BType = BType_;
    using CType = CType_;
    using BiasType = BiasType_;
    using L1TileShape = L1TileShape_;
    using L0TileShape = L0TileShape_;
    using LayoutA = LayoutA_;
    using LayoutB = LayoutB_;
    using LayoutC = LayoutC_;
    using TileCopyParam = TileCopyParam_;

    static constexpr bool transA = TagToTrans<LayoutA>::value;
    static constexpr bool transB = TagToTrans<LayoutB>::value;

    using TupleShape = AscendC::Shape<int64_t, int64_t, int64_t>;              // m,n,k
    using TupleTileShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>; // m,n,ka,kb
    // host side kernel arguments
    struct Arguments {
        GM_ADDR aGmAddr{nullptr};
        GM_ADDR bGmAddr{nullptr};
        GM_ADDR cGmAddr{nullptr};
        GM_ADDR biasGmAddr{nullptr};
        GM_ADDR groupListGmAddr{nullptr};
    };

    // params
    using Params = Arguments;

public:
    __aicore__ inline BlockMmadPertile();
    __aicore__ inline void Init(
        const TupleShape& l0Shape, const TupleTileShape& tileL12L0, AscendC::LocalTensor<CType>* ping,
        AscendC::LocalTensor<CType>* pong, uint8_t l1BufNum = 2);
    __aicore__ inline void operator()(const TupleShape& actualSingleShape, const AscendC::GlobalTensor<AType>& aGlobal,
                                      const AscendC::GlobalTensor<BType>& bGlobal);
    __aicore__ inline void UpdateParamsForNextProblem(const TupleShape& problemShape);
    __aicore__ inline void End();
    __aicore__ inline ~BlockMmadPertile();

private:
    __aicore__ inline void StepDifferentKL1(
        const AscendC::GlobalTensor<AType>& aGlobal, const AscendC::GlobalTensor<BType>& bGlobal);
    __aicore__ inline void StepSameKL1(
        const AscendC::GlobalTensor<AType>& aGlobal, const AscendC::GlobalTensor<BType>& bGlobal);
    __aicore__ inline void AicBaseMadProcess(
        uint64_t kInner, uint64_t kAL1Offset, bool isTailAL1, uint64_t kBL1Offset, bool isTailBL1);
    __aicore__ inline void CopyInA1Nd2Nz(const AscendC::GlobalTensor<AType>& aGlobal, uint64_t kOffset, bool isTailAL1);
    __aicore__ inline void CopyInB1Nd2Nz(const AscendC::GlobalTensor<BType>& bGlobal, uint64_t kOffset, bool isTailBL1);
    __aicore__ inline void CopyInA2(uint64_t mAL1Offset, uint64_t kAL1Offset, uint64_t kOffset, bool isTailAL1);
    __aicore__ inline void CopyInB2(uint64_t nBL1Offset, uint64_t kBL1Offset, uint64_t kOffset, bool isTailBL1);
    __aicore__ inline void MmadBase(uint64_t kOffset);
    __aicore__ inline void UpdatePertileMmParam();
    __aicore__ inline void WaitForVector(uint16_t crossPingPongID)
    {
        AscendC::CrossCoreWaitFlag<QBMM_AIC_SYNC_AIV_MODE, PIPE_FIX>(QBMM_AIC_SYNC_AIV_FLAG + crossPingPongID);
        AscendC::CrossCoreWaitFlag<QBMM_AIC_SYNC_AIV_MODE, PIPE_FIX>(
            QBMM_AIC_SYNC_AIV_FLAG + crossPingPongID + QBMM_FLAG_ID_MAX);
    }
    __aicore__ inline void NotifyVector(uint16_t crossPingPongID)
    {
        AscendC::CrossCoreSetFlag<QBMM_AIC_SYNC_AIV_MODE, PIPE_FIX>(QBMM_AIV_SYNC_AIC_FLAG + crossPingPongID);
        AscendC::CrossCoreSetFlag<QBMM_AIC_SYNC_AIV_MODE, PIPE_FIX>(QBMM_AIV_SYNC_AIC_FLAG + crossPingPongID +
                                                                   QBMM_FLAG_ID_MAX);
    }

public:
    AscendC::LocalTensor<CType>* mmResPing_;
    AscendC::LocalTensor<CType>* mmResPong_;

    uint64_t l1BufferAOffset_[4] = {0}; // default 4 buffer
    uint64_t l1BufferBOffset_[4] = {0}; // default 4 buffer
    AscendC::LocalTensor<AType> al1Local_{AscendC::TPosition::A1, 0, AscendC::TOTAL_L1_SIZE};
    AscendC::LocalTensor<BType> bL1Local_{AscendC::TPosition::B1, 0, AscendC::TOTAL_L1_SIZE};
    AscendC::LocalTensor<AType> aL0Ping_;
    AscendC::LocalTensor<AType> aL0Pong_;
    AscendC::LocalTensor<BType> bL0Ping_;
    AscendC::LocalTensor<BType> bL0Pong_;
    AscendC::LocalTensor<CType> cL0Ping_;
    AscendC::LocalTensor<CType> cL0Pong_;

private:
    TupleShape problemShape_;
    TupleShape actualSingleShape_;
    PertileMmParam mmParams_;
    MatMulCommonParam<transA, transB> matmulParam_;
    uint64_t baseCount_{0};
    uint32_t baseM_;
    uint32_t baseN_;
    uint32_t baseK_;
    uint32_t kaL1_;
    uint32_t kbL1_;
    uint32_t maxKL1_{0};
    uint32_t minKL1_{0};
    bool needAicWait_{false};
    bool orderAL1BL1_{false};
    bool isSameKL1_{false};
    uint8_t l1BufNum_;
    uint8_t aL1BufferID_{0};
    uint8_t bL1BufferID_{0};
    uint8_t l0PingPongID_{0};
    uint8_t crossPingPongID_{0};
};

QBMM_BLOCK_MMAD_PERTILE_CLASS_LOCAL_PARAMS
__aicore__ inline BlockMmadPertile<QBMM_BLOCK_MMAD_PERTILE_FUNC_LOCAL_PARAMS>::BlockMmadPertile()
{
    if ASCEND_IS_AIC {
        #pragma unroll
        for (uint8_t i = 0; i < MTE1_MTE2_EVENT_ID_NUM; ++i) {
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(i);
        }
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(0 + QBMM_CUBE_SYNC_MTE1_FLAG);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(1 + QBMM_CUBE_SYNC_MTE1_FLAG);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(0);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(1);
    }
}

QBMM_BLOCK_MMAD_PERTILE_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockMmadPertile<QBMM_BLOCK_MMAD_PERTILE_FUNC_LOCAL_PARAMS>::Init(
    const TupleShape& l0Shape, const TupleTileShape& tileL12L0, AscendC::LocalTensor<CType>* ping,
    AscendC::LocalTensor<CType>* pong, uint8_t l1BufNum)
{
    if ASCEND_IS_AIC {
        baseM_ = Get<MNK_M>(l0Shape);
        baseN_ = Get<MNK_N>(l0Shape);
        baseK_ = Get<MNK_K>(l0Shape);
        kaL1_ = Get<2>(tileL12L0); // 2: idx of kaL1 in tileshape
        kbL1_ = Get<3>(tileL12L0); // 3: idx of kbL1 in tileshape
        l1BufNum_ = l1BufNum;
        orderAL1BL1_ = kaL1_ >= kbL1_;
        isSameKL1_ = kaL1_ == kbL1_;
        maxKL1_ = orderAL1BL1_ ? kaL1_ : kbL1_;
        minKL1_ = orderAL1BL1_ ? kbL1_ : kaL1_;
        matmulParam_.Init(l0Shape, tileL12L0);
        mmResPing_ = ping;
        mmResPong_ = pong;

        uint32_t aL0OneBuffer = baseM_ * baseK_;
        uint32_t bL0OneBuffer = baseN_ * baseK_;
        uint32_t cL0OneBuffer = baseM_ * baseN_;
        uint64_t AL1OneBufferSize = baseM_ * kaL1_ * sizeof(AType);
        uint64_t BL1OneBufferSize = baseN_ * kbL1_ * sizeof(BType);
        // 2 buffer L1 space is : |A0|B0|.........256KB|A1|B1|.........
        // 4 buffer L1 space is : |A0|B0|A2|B2|...256KB|A1|B1|A3|B3|...
        l1BufferAOffset_[1] = AscendC::TOTAL_L1_SIZE >> 1;
        l1BufferBOffset_[0] = AL1OneBufferSize;
        l1BufferBOffset_[1] = l1BufferAOffset_[1] + AL1OneBufferSize;
        if (l1BufNum_ == FOUR_BUFFER) {
            l1BufferAOffset_[2] = l1BufferBOffset_[0] + BL1OneBufferSize;
            l1BufferAOffset_[3] = l1BufferBOffset_[1] + BL1OneBufferSize;
            l1BufferBOffset_[2] = l1BufferAOffset_[2] + AL1OneBufferSize;
            l1BufferBOffset_[3] = l1BufferAOffset_[3] + AL1OneBufferSize;
        }
        aL0Ping_ = AscendC::LocalTensor<AType>(AscendC::TPosition::A2, 0, aL0OneBuffer);
        aL0Pong_ = AscendC::LocalTensor<AType>(AscendC::TPosition::A2, aL0OneBuffer * sizeof(AType), aL0OneBuffer);
        bL0Ping_ = AscendC::LocalTensor<BType>(AscendC::TPosition::B2, 0, bL0OneBuffer);
        bL0Pong_ = AscendC::LocalTensor<BType>(AscendC::TPosition::B2, bL0OneBuffer * sizeof(BType), bL0OneBuffer);
        cL0Ping_ = AscendC::LocalTensor<CType>(AscendC::TPosition::CO1, 0, cL0OneBuffer);
        cL0Pong_ = AscendC::LocalTensor<CType>(AscendC::TPosition::CO1, cL0OneBuffer * sizeof(CType), cL0OneBuffer);
    }
}

QBMM_BLOCK_MMAD_PERTILE_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockMmadPertile<QBMM_BLOCK_MMAD_PERTILE_FUNC_LOCAL_PARAMS>::UpdateParamsForNextProblem(
    const TupleShape& problemShape)
{
    problemShape_ = problemShape;
    matmulParam_.UpdateProblemParams(problemShape_);
}

QBMM_BLOCK_MMAD_PERTILE_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockMmadPertile<QBMM_BLOCK_MMAD_PERTILE_FUNC_LOCAL_PARAMS>::UpdatePertileMmParam()
{
    mmParams_.fixpipeSplitN = Get<MNK_N>(actualSingleShape_) > PER_BLOCK_SIZE || Get<MNK_M>(actualSingleShape_) == 1;

    if constexpr (transA) {
        mmParams_.srcStride = Align(Get<MNK_M>(actualSingleShape_), static_cast<uint64_t>(AscendC::ONE_BLK_SIZE));
    } else {
        mmParams_.srcStride = Align(Get<MNK_M>(actualSingleShape_), static_cast<uint64_t>(AscendC::BLOCK_CUBE));
    }
    mmParams_.fixpipeM = mmParams_.fixpipeSplitN ?
                             Get<MNK_M>(actualSingleShape_) :
                             Align(Get<MNK_M>(actualSingleShape_), static_cast<uint64_t>(GetAicAivTaskRation()));
    if (mmParams_.fixpipeSplitN) {
        mmParams_.ndNum = Get<MNK_N>(actualSingleShape_) > UB_TWO_BANK_ELEMS_B32 ? 2 : 1; // 2: 2 ND
        int64_t alignedNBase =
            Get<MNK_N>(actualSingleShape_) > PER_BLOCK_SIZE ? PER_BLOCK_SIZE : AscendC::ONE_BLK_SIZE * mmParams_.ndNum;
        mmParams_.fixpipeN = Align(Get<MNK_N>(actualSingleShape_), static_cast<uint64_t>(alignedNBase));
    } else {
        mmParams_.ndNum = Get<MNK_N>(actualSingleShape_) > UB_SUB_BANK_ELEMS_B32 ? 2 : 1;
        mmParams_.fixpipeN =
            Align(Get<MNK_N>(actualSingleShape_), static_cast<uint64_t>(AscendC::BLOCK_CUBE) * mmParams_.ndNum);
    }
    mmParams_.fixpipeN /= mmParams_.ndNum;
}

QBMM_BLOCK_MMAD_PERTILE_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockMmadPertile<QBMM_BLOCK_MMAD_PERTILE_FUNC_LOCAL_PARAMS>::operator()(
    const TupleShape& actualSingleShape, const AscendC::GlobalTensor<AType>& aGlobal,
    const AscendC::GlobalTensor<BType>& bGlobal)
{
    actualSingleShape_ = actualSingleShape;
    matmulParam_.UpdateNextBlockParams(actualSingleShape_);
    UpdatePertileMmParam();
    if (isSameKL1_) {
        StepSameKL1(aGlobal, bGlobal);
    } else {
        StepDifferentKL1(aGlobal, bGlobal);
    }
}

QBMM_BLOCK_MMAD_PERTILE_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockMmadPertile<QBMM_BLOCK_MMAD_PERTILE_FUNC_LOCAL_PARAMS>::StepDifferentKL1(
    const AscendC::GlobalTensor<AType>& aGlobal, const AscendC::GlobalTensor<BType>& bGlobal)
{
    bool isTailAL1 = false;
    bool isTailBL1 = false;
    if (orderAL1BL1_) {
        for (uint64_t kOuter = 0; kOuter < Get<MNK_K>(problemShape_); kOuter += maxKL1_) {
            isTailAL1 = (kOuter + maxKL1_) >= Get<MNK_K>(problemShape_);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(aL1BufferID_);
            CopyInA1Nd2Nz(aGlobal, kOuter, isTailAL1);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(aL1BufferID_);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(aL1BufferID_);
            for (uint64_t kInner = kOuter;
                 kInner < AscendC::Std::min(kOuter + maxKL1_, static_cast<uint64_t>(Get<MNK_K>(problemShape_)));
                 kInner += minKL1_) {
                isTailBL1 = (kInner + minKL1_) >= Get<MNK_K>(problemShape_);
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(bL1BufferID_ + DOUBLE_BUFFER);
                CopyInB1Nd2Nz(bGlobal, kInner, isTailBL1);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(bL1BufferID_ + DOUBLE_BUFFER);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(bL1BufferID_ + DOUBLE_BUFFER);
                AicBaseMadProcess(kInner, kInner - kOuter, isTailAL1, 0UL, isTailBL1);
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(bL1BufferID_ + DOUBLE_BUFFER);
                bL1BufferID_ = bL1BufferID_ ^ 1;
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(aL1BufferID_);
            aL1BufferID_ = aL1BufferID_ ^ 1;
        }
    } else {
        for (uint64_t kOuter = 0; kOuter < Get<MNK_K>(problemShape_); kOuter += maxKL1_) {
            isTailBL1 = (kOuter + maxKL1_) >= Get<MNK_K>(problemShape_);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(bL1BufferID_ + DOUBLE_BUFFER);
            CopyInB1Nd2Nz(bGlobal, kOuter, isTailBL1);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(bL1BufferID_ + DOUBLE_BUFFER);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(bL1BufferID_ + DOUBLE_BUFFER);
            for (uint64_t kInner = kOuter;
                 kInner < AscendC::Std::min(kOuter + maxKL1_, static_cast<uint64_t>(Get<MNK_K>(problemShape_)));
                 kInner += minKL1_) {
                isTailAL1 = (kInner + minKL1_) >= Get<MNK_K>(problemShape_);
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(aL1BufferID_);
                CopyInA1Nd2Nz(aGlobal, kInner, isTailAL1);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(aL1BufferID_);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(aL1BufferID_);
                AicBaseMadProcess(kInner, 0UL, isTailAL1, kInner - kOuter, isTailBL1);
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(aL1BufferID_);
                aL1BufferID_ = aL1BufferID_ ^ 1;
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(bL1BufferID_ + DOUBLE_BUFFER);
            bL1BufferID_ = bL1BufferID_ ^ 1;
        }
    }
}

QBMM_BLOCK_MMAD_PERTILE_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockMmadPertile<QBMM_BLOCK_MMAD_PERTILE_FUNC_LOCAL_PARAMS>::StepSameKL1(
    const AscendC::GlobalTensor<AType>& aGlobal, const AscendC::GlobalTensor<BType>& bGlobal)
{
    bool isTailL1 = false;
    for (uint64_t kOuter = 0; kOuter < Get<MNK_K>(problemShape_); kOuter += kaL1_) {
        isTailL1 = (kOuter + kaL1_) >= Get<MNK_K>(problemShape_);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(aL1BufferID_);
        CopyInA1Nd2Nz(aGlobal, kOuter, isTailL1);
        CopyInB1Nd2Nz(bGlobal, kOuter, isTailL1);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(aL1BufferID_);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(aL1BufferID_);
        AicBaseMadProcess(kOuter, 0UL, isTailL1, 0UL, isTailL1);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(aL1BufferID_);
        aL1BufferID_ = (aL1BufferID_ + 1) & (l1BufNum_ - 1);
        bL1BufferID_ = aL1BufferID_;
    }
}

QBMM_BLOCK_MMAD_PERTILE_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockMmadPertile<QBMM_BLOCK_MMAD_PERTILE_FUNC_LOCAL_PARAMS>::AicBaseMadProcess(
    uint64_t kInner, uint64_t kAL1Offset, bool isTailAL1, uint64_t kBL1Offset, bool isTailBL1)
{
    for (uint64_t kb = kInner;
         kb < AscendC::Std::min(kInner + minKL1_, static_cast<uint64_t>(Get<MNK_K>(problemShape_))); kb += baseK_) {
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0PingPongID_ + QBMM_CUBE_SYNC_MTE1_FLAG);
        CopyInA2(0, kAL1Offset, kb, isTailAL1);
        CopyInB2(0, kBL1Offset, kb, isTailBL1);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0PingPongID_);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0PingPongID_);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(l0PingPongID_);
        MmadBase(kb);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0PingPongID_ + QBMM_CUBE_SYNC_MTE1_FLAG);
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(l0PingPongID_);
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(l0PingPongID_);
        AscendC::FixpipeParamsC310<AscendC::CO2Layout::ROW_MAJOR> fixpipeParams(
            mmParams_.fixpipeN, mmParams_.fixpipeM, mmParams_.srcStride, UB_TWO_BANK_ELEMS_B32); // dstStride is 128
        fixpipeParams.dualDstCtl = mmParams_.fixpipeSplitN ? 2 : 1; // 2 means splitting N with ratio 1:2
        // When nz2nd loop in copyout, srcndstride is unit of c0Size, dstndstride is unit of one element.
        fixpipeParams.params.ndNum = mmParams_.ndNum;
        fixpipeParams.params.srcNdStride = mmParams_.srcStride * (mmParams_.fixpipeN / AscendC::BLOCK_CUBE);
        fixpipeParams.params.dstNdStride = UB_TWO_BANK_ELEMS_B32 * PER_BLOCK_SIZE;
        if (needAicWait_) {
            WaitForVector(crossPingPongID_);
        }
        AscendC::Fixpipe<CType, CType, AscendC::Impl::CFG_ROW_MAJOR_UB>(
            crossPingPongID_ == 0 ? *mmResPing_ : *mmResPong_, l0PingPongID_ == 0 ? cL0Ping_ : cL0Pong_, fixpipeParams);
        NotifyVector(crossPingPongID_);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(l0PingPongID_);
        needAicWait_ = needAicWait_ || crossPingPongID_ == 1;
        crossPingPongID_ = (crossPingPongID_ + 1) & 1;
        l0PingPongID_ = l0PingPongID_ ^ 1;
        kAL1Offset = kAL1Offset + baseK_;
        kBL1Offset = kBL1Offset + baseK_;
        baseCount_++;
    }
}

QBMM_BLOCK_MMAD_PERTILE_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockMmadPertile<QBMM_BLOCK_MMAD_PERTILE_FUNC_LOCAL_PARAMS>::CopyInA2(
    uint64_t mAL1Offset, uint64_t kAL1Offset, uint64_t kOffset, bool isTailAL1)
{
    uint64_t offsetAL1 = matmulParam_.CalcAL1Offset(mAL1Offset, kAL1Offset, isTailAL1);
    AscendC::LoadData2DParamsV2 loadData2dParams;
    matmulParam_.LoadData2dParamsA(loadData2dParams, kOffset, isTailAL1);
    AscendC::LoadData(
        l0PingPongID_ == 0 ? aL0Ping_ : aL0Pong_, al1Local_[l1BufferAOffset_[aL1BufferID_] + offsetAL1],
        loadData2dParams);
}

QBMM_BLOCK_MMAD_PERTILE_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockMmadPertile<QBMM_BLOCK_MMAD_PERTILE_FUNC_LOCAL_PARAMS>::CopyInB2(
    uint64_t nBL1Offset, uint64_t kBL1Offset, uint64_t kOffset, bool isTailBL1)
{
    uint64_t offsetBL1 = matmulParam_.CalcBL1Offset(nBL1Offset, kBL1Offset, isTailBL1);
    AscendC::LoadData2DParamsV2 loadData2dParams;
    matmulParam_.LoadData2dParamsB(loadData2dParams, kOffset, isTailBL1);
    AscendC::LoadData(
        l0PingPongID_ == 0 ? bL0Ping_ : bL0Pong_, bL1Local_[l1BufferBOffset_[bL1BufferID_] + offsetBL1],
        loadData2dParams);
}

QBMM_BLOCK_MMAD_PERTILE_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockMmadPertile<QBMM_BLOCK_MMAD_PERTILE_FUNC_LOCAL_PARAMS>::MmadBase(uint64_t kOffset)
{
    uint32_t mmadK = AscendC::Std::min(static_cast<uint64_t>(baseK_), Get<MNK_K>(problemShape_) - kOffset);
    AscendC::MmadParams mmadParams;
    if constexpr (transA) {
        mmadParams.m = Align(Get<MNK_M>(actualSingleShape_), static_cast<uint64_t>(QBMM_DATA_BLOCK));
    } else {
        mmadParams.m = Get<MNK_M>(actualSingleShape_);
    }
    if constexpr (transB) {
        mmadParams.n = Get<MNK_N>(actualSingleShape_);
    } else {
        mmadParams.n = Align(Get<MNK_N>(actualSingleShape_), static_cast<uint64_t>(QBMM_DATA_BLOCK));
    }
    mmadParams.k = mmadK;
    mmadParams.disableGemv = true;
    AscendC::Mmad(l0PingPongID_ == 0 ? cL0Ping_ : cL0Pong_, l0PingPongID_ == 0 ? aL0Ping_ : aL0Pong_,
                  l0PingPongID_ == 0 ? bL0Ping_ : bL0Pong_, mmadParams);
}

QBMM_BLOCK_MMAD_PERTILE_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockMmadPertile<QBMM_BLOCK_MMAD_PERTILE_FUNC_LOCAL_PARAMS>::CopyInA1Nd2Nz(
    const AscendC::GlobalTensor<AType>& aGlobal, uint64_t kOffset, bool isTailAL1)
{
    uint64_t offset = matmulParam_.CalcAGMOffsetInnerLoop(0, kOffset);
    AscendC::Nd2NzParams nd2nzParam;
    matmulParam_.CalNd2NzParamA(nd2nzParam, isTailAL1);
    AscendC::DataCopy(al1Local_[l1BufferAOffset_[aL1BufferID_]], aGlobal[offset], nd2nzParam);
}

QBMM_BLOCK_MMAD_PERTILE_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockMmadPertile<QBMM_BLOCK_MMAD_PERTILE_FUNC_LOCAL_PARAMS>::CopyInB1Nd2Nz(
    const AscendC::GlobalTensor<BType>& bGlobal, uint64_t kOffset, bool isTailBL1)
{
    uint64_t offset = matmulParam_.CalcBGMOffsetInnerLoop(0, kOffset);
    AscendC::Nd2NzParams nd2nzParam;
    matmulParam_.CalNd2NzParamB(nd2nzParam, isTailBL1);
    AscendC::DataCopy(bL1Local_[l1BufferBOffset_[bL1BufferID_]], bGlobal[offset], nd2nzParam);
}

QBMM_BLOCK_MMAD_PERTILE_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockMmadPertile<QBMM_BLOCK_MMAD_PERTILE_FUNC_LOCAL_PARAMS>::End()
{
    if ASCEND_IS_AIC {
        if (baseCount_ > 1) {
            WaitForVector(0); // ping
            WaitForVector(1); // pong
        } else if (baseCount_ == 1) {
            WaitForVector(0);
        }
    }
}

QBMM_BLOCK_MMAD_PERTILE_CLASS_LOCAL_PARAMS
__aicore__ inline BlockMmadPertile<QBMM_BLOCK_MMAD_PERTILE_FUNC_LOCAL_PARAMS>::~BlockMmadPertile()
{
    if ASCEND_IS_AIC {
        #pragma unroll
        for(uint8_t i = 0; i < MTE1_MTE2_EVENT_ID_NUM; ++i) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(i);
        }
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(0 + QBMM_CUBE_SYNC_MTE1_FLAG);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(1 + QBMM_CUBE_SYNC_MTE1_FLAG);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(0);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(1);
    }
}
} // namespace Block
} // namespace Gemm
} // namespace Cmct
#endif
