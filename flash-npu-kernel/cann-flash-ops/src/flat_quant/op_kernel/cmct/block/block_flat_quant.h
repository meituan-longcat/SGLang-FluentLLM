/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file block_flat_quant.h
 * \brief
 */

#ifndef MATMUL_BLOCK_BLOCK_FLAT_QUANT_H
#define MATMUL_BLOCK_BLOCK_FLAT_QUANT_H
#include "./block_mmad.h"
#include "../utils/common_utils.h"
#include "../utils/layout_utils.h"
#include "../utils/tuple_utils.h"
#include "../policy/dispatch_policy.h"

namespace Cmct {
namespace Gemm {
namespace Block {
template <class DispatchPolicy_, class L1TileShape_, class L0TileShape_, class AType_, class BType_, class CType_,
          class BiasType_, class TileCopy_>
class BlockMmad<DispatchPolicy_, L1TileShape_, L0TileShape_, AType_, BType_, CType_, BiasType_, TileCopy_,
                AscendC::Std::enable_if_t<AscendC::Std::is_base_of_v<MatmulFlatQuant<>, DispatchPolicy_>>> {
public:
    using L0cType = float;
    using AType = AType_;
    using BType = BType_;
    using CType = CType_;
    using BiasType = BiasType_;
    using A_T = typename AType::T;
    using B_T = typename BType::T;
    using C_T = typename CType::T;
    using Bias_T = typename BiasType::T;
    using DispatchPolicy = DispatchPolicy_;
    using TupleShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;
    using TupleL1L0Shape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>;
    uint64_t m_{1};
    uint64_t n_{1};
    uint64_t k_{1};
    uint64_t mL1_{1};
    uint64_t nL1_{1};
    uint64_t kL1_{1};
    uint64_t baseM_{16};
    uint64_t baseN_{16};
    uint64_t baseK_{16};
    uint64_t kAlign_{1};
    uint64_t l0PingPong_{1};
    uint64_t l0cPingPong_{1};
    uint64_t subBlockId_{0};
    bool hasP2_{true};
    constexpr static uint64_t BUFFER_NUM = 2;
    constexpr static uint64_t HALF_L0_SIZE = L0A_SIZE / BUFFER_NUM / sizeof(A_T);
    constexpr static uint64_t HALF_L0C_SIZE = AscendC::TOTAL_L0C_SIZE / BUFFER_NUM / sizeof(L0cType);
    constexpr static int32_t C0_SIZE = AscendC::AuxGetC0Size<typename AType::T>();

    __aicore__ inline BlockMmad()
    {
        if ASCEND_IS_AIC {
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(ZERO_FLAG);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(FIRST_FLAG);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(SECOND_FLAG);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(THIRD_FLAG);
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(ZERO_FLAG);
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(FIRST_FLAG);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(SIXTH_FLAG);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(SEVENTH_FLAG);
        }
    }

    __aicore__ inline ~BlockMmad()
    {
        if ASCEND_IS_AIC {
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(ZERO_FLAG);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(FIRST_FLAG);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(SECOND_FLAG);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(THIRD_FLAG);
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(ZERO_FLAG);
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(FIRST_FLAG);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(SIXTH_FLAG);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(SEVENTH_FLAG);
        }
    }

public:
    template <uint64_t FULL_LOAD_MODE_ = B_FULL_LOAD_MODE>
    __aicore__ inline void Init(const TupleShape& shape, const TupleShape& tileL1, const TupleShape& tileL0, bool hasP2)
    {
        m_ = Get<MKN_M>(shape);
        n_ = Get<MKN_N>(shape);
        k_ = Get<MKN_K>(shape);
        mL1_ = Get<MKN_M>(tileL1);
        nL1_ = Get<MKN_N>(tileL1);
        kL1_ = Get<MKN_K>(tileL1);
        iterBatch_ = Get<MKN_B>(tileL1);
        baseM_ = Get<MKN_M>(tileL0);
        baseN_ = Get<MKN_N>(tileL0);
        baseK_ = Get<MKN_K>(tileL0);

        uint64_t nl1Align = Cmct::Gemm::Align(nL1_, AscendC::BLOCK_CUBE);
        uint64_t kl1Align = Cmct::Gemm::Align(nL1_, AscendC::BLOCK_CUBE);
        bl1OffsetP2_ = nl1Align * kl1Align;
        l0PingPong_ = 0;
        l0cPingPong_ = 0;
        subBlockId_ = 0;
        hasP2_ = hasP2;
    }

    __aicore__ inline void CopyInA1(const AscendC::GlobalTensor<A_T>& aGlobal,
                                    const AscendC::LocalTensor<A_T>& al1Local, uint64_t curML1, uint64_t curKL1,
                                    uint64_t srcDValue)
    {
        AscendC::Nd2NzParams nd2nzParams;
        nd2nzParams.ndNum = 1;
        uint64_t nDim = curML1;
        uint64_t dDim = curKL1;
        nd2nzParams.nValue = nDim;
        nd2nzParams.dValue = dDim;
        nd2nzParams.srcNdMatrixStride = 1;
        nd2nzParams.srcDValue = srcDValue;
        nd2nzParams.dstNzC0Stride = Cmct::Gemm::Align(nDim, AscendC::BLOCK_CUBE);
        nd2nzParams.dstNzNStride = 1;
        nd2nzParams.dstNzMatrixStride = nd2nzParams.dstNzC0Stride * Cmct::Gemm::Align(dDim, AscendC::BLOCK_CUBE);
        AscendC::DataCopy(al1Local, aGlobal, nd2nzParams);
    }

    __aicore__ inline void CopyInB1(const AscendC::GlobalTensor<B_T>& p1Global,
                                    const AscendC::LocalTensor<B_T>& bl1Local, uint64_t curNL1, uint64_t curKL1)
    {
        AscendC::Nd2NzParams nd2nzParams;
        nd2nzParams.ndNum = 1;
        uint64_t nDim = curKL1;
        uint64_t dDim = curNL1;
        nd2nzParams.nValue = nDim;
        nd2nzParams.dValue = dDim;
        nd2nzParams.srcNdMatrixStride = 1;
        nd2nzParams.srcDValue = n_;
        nd2nzParams.dstNzC0Stride = Cmct::Gemm::Align(nDim, AscendC::BLOCK_CUBE);
        nd2nzParams.dstNzNStride = 1;
        nd2nzParams.dstNzMatrixStride = 1;
        AscendC::DataCopy(bl1Local, p1Global, nd2nzParams);
    }

    __aicore__ inline void CopyInA2(const AscendC::LocalTensor<A_T>& a2Local, const AscendC::LocalTensor<A_T>& al1Local,
                                    uint64_t curML1, uint64_t mL0, uint64_t kL0)
    {
        // (M,K) use LoadData2D
        AscendC::LoadData2DParamsV2 loadDataParams;
        loadDataParams.mStartPosition = 0;
        loadDataParams.kStartPosition = 0;
        loadDataParams.mStep = Cmct::Gemm::CeilDiv(mL0, AscendC::BLOCK_CUBE);
        loadDataParams.kStep = Cmct::Gemm::CeilDiv(kL0, AscendC::BLOCK_CUBE);
        loadDataParams.srcStride = Cmct::Gemm::CeilDiv(curML1, AscendC::BLOCK_CUBE);
        loadDataParams.dstStride = loadDataParams.mStep;
        loadDataParams.ifTranspose = false;
        AscendC::LoadData<A_T>(a2Local, al1Local, loadDataParams);
    }

    __aicore__ inline void CopyInB2(const AscendC::LocalTensor<B_T>& b2Local, const AscendC::LocalTensor<B_T>& bl1Local,
                                    uint64_t curKL1, uint64_t nL0, uint64_t kL0, uint64_t batch)
    {
        // (K,N) use LoadData2D
        AscendC::LoadData2DParamsV2 loadDataParams;
        loadDataParams.mStartPosition = 0;
        loadDataParams.kStartPosition = 0;
        loadDataParams.mStep = Cmct::Gemm::CeilDiv(kL0, AscendC::BLOCK_CUBE);
        loadDataParams.kStep = Cmct::Gemm::CeilDiv(nL0, AscendC::BLOCK_CUBE);
        loadDataParams.srcStride = batch * Cmct::Gemm::CeilDiv(curKL1, AscendC::BLOCK_CUBE);
        loadDataParams.dstStride = loadDataParams.kStep;
        loadDataParams.ifTranspose = true;
        if constexpr (AscendC::IsSameType<B_T, bfloat16_t>::value) {
            AscendC::LoadData(b2Local, bl1Local, loadDataParams);
        } else {
            AscendC::LoadData<B_T>(b2Local, bl1Local, loadDataParams);
        }
    }

    __aicore__ inline void FixpipeToL1(const AscendC::LocalTensor<A_T>& dstLocal,
                                       AscendC::LocalTensor<L0cType>& c1Local, uint64_t baseM, uint64_t baseN)
    {
        AscendC::FixpipeParamsC310<AscendC::CO2Layout::NZ> fixpipeParams;
        fixpipeParams.nSize = Cmct::Gemm::Align(baseN, AscendC::BLOCK_CUBE);
        fixpipeParams.mSize = baseM;

        fixpipeParams.srcStride = Cmct::Gemm::Align(baseM, AscendC::BLOCK_CUBE); // 单位是CO_SIZE(16*sizeof(C_T))
        fixpipeParams.dstStride = Cmct::Gemm::Align(baseM, AscendC::BLOCK_CUBE) * AscendC::BLOCK_CUBE;

        if constexpr (AscendC::IsSameType<A_T, bfloat16_t>::value) {
            fixpipeParams.quantPre = QuantMode_t::F322BF16;
        } else if (AscendC::IsSameType<A_T, half>::value) {
            fixpipeParams.quantPre = QuantMode_t::F322F16;
        }
        fixpipeParams.dualDstCtl = 0;
        fixpipeParams.unitFlag = 0;
        AscendC::Fixpipe<A_T, L0cType, CFG_NZ>(dstLocal, c1Local, fixpipeParams);
    }

    __aicore__ inline void CopyOut(const AscendC::LocalTensor<A_T>& dstLocal, AscendC::LocalTensor<L0cType>& c1Local,
                                   uint64_t baseM, uint64_t baseN)
    {
        AscendC::FixpipeParamsC310<AscendC::CO2Layout::ROW_MAJOR> fixpipeParams;
        uint64_t c0 = AscendC::BLOCK_CUBE;
        fixpipeParams.nSize = Cmct::Gemm::Align(baseN, c0);
        fixpipeParams.mSize = baseM;
        fixpipeParams.srcStride = Cmct::Gemm::Align(baseM, AscendC::BLOCK_CUBE); // 单位是CO_SIZE(16*sizeof(C_T))
        fixpipeParams.dstStride = fixpipeParams.nSize;
        fixpipeParams.quantPre = QuantMode_t::F322BF16;
        fixpipeParams.dualDstCtl = 0;
        fixpipeParams.unitFlag = 0;
        fixpipeParams.subBlockId = (subBlockId_++) & 0x1;
        fixpipeParams.params.ndNum = iterBatch_;
        fixpipeParams.params.srcNdStride = baseM * Cmct::Gemm::Align(baseN, c0) / AscendC::BLOCK_CUBE;
        fixpipeParams.params.dstNdStride = baseM * Cmct::Gemm::Align(baseN, c0);
        AscendC::Fixpipe<A_T, L0cType, AscendC::Impl::CFG_ROW_MAJOR_UB>(dstLocal, c1Local, fixpipeParams);
    }

    template <typename T>
    __aicore__ inline void operator()(T cTensor, AscendC::GlobalTensor<A_T> aGlobal,
                                      AscendC::GlobalTensor<B_T> p1Global, AscendC::GlobalTensor<B_T> p2Global,
                                      TupleL1L0Shape tileShape, bool isFirstRound)
    {
        mL1_ = Get<MKN_M>(tileShape);
        nL1_ = Get<MKN_N>(tileShape);
        kL1_ = Get<MKN_K>(tileShape);
        aL1OneBuffer_ = Cmct::Gemm::Align(iterBatch_ * m_, AscendC::BLOCK_CUBE) *
                        Cmct::Gemm::Align(kL1_, AscendC::BLOCK_CUBE);
        uint64_t curML1 = Get<MNK_M>(tileShape);
        uint64_t curNL1 = Get<MNK_N>(tileShape);
        uint64_t curKL1 = Get<MNK_K>(tileShape);
        uint64_t curML0 = Get<MNK_M0>(tileShape);
        uint64_t curNL0 = Get<MNK_N0>(tileShape);
        iterBatch_ = Get<MNK_B>(tileShape);

        uint64_t l0cOffset = (l0cPingPong_ & 0x1) * HALF_L0C_SIZE;
        if (curML0 * curNL0 > HALF_L0C_SIZE) {
            l0cOffset = 0;
        }
        AscendC::LocalTensor<L0cType> c1Local = c1Local_[l0cOffset];
        uint64_t al1Offset = (l0cPingPong_ & 0x1) * aL1OneBuffer_;
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(l0cPingPong_ & 0x1);

        // first matmul:(k,m,n) * (n,n)
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l0cPingPong_ & 0x1);
        CopyInA1(aGlobal, l1Local_[al1Offset], curML1, curKL1, k_);
        if (hasP2_) {
            if (isFirstRound) {
                CopyInB1(p2Global, l1Local_[aL1OneBuffer_ * NUM_TWO], curNL1, curKL1);
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l0cPingPong_ & 0x1);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l0cPingPong_ & 0x1);

            uint64_t kL0Iter = Cmct::Gemm::CeilDiv(curKL1, baseK_);
            for (uint64_t iter1 = 0; iter1 < kL0Iter; ++iter1) {
                uint64_t curK0 = (iter1 + 1 == kL0Iter) ? (curKL1 - iter1 * baseK_) : baseK_;
                // 搬运到L0,开启double buffer
                uint64_t l0Offset = HALF_L0_SIZE * (l0PingPong_ & 0x1);
                uint64_t mte1Flag = ((l0PingPong_ & 0x1) + SIXTH_FLAG);
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(static_cast<uint16_t>(mte1Flag));
                // al1:(k1,m1,m0,k0)
                // bl1:(n1,k1,k0,n0)
                uint64_t bl1Offset = iter1 * baseK_ * AscendC::BLOCK_CUBE;
                CopyInA2(l0aLocal_[l0Offset], l1Local_[al1Offset + iter1 * baseK_ * curML1], curML1, curML0, curK0);
                CopyInB2(l0bLocal_[l0Offset], l1Local_[aL1OneBuffer_ * NUM_TWO + bl1Offset], curKL1, curNL0, curK0, 1);

                AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(static_cast<uint16_t>(mte1Flag));
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(static_cast<uint16_t>(mte1Flag));
                AscendC::MmadParams mmadParams;
                mmadParams.m = curML0;
                mmadParams.n = curNL0;
                mmadParams.k = curK0;
                mmadParams.disableGemv = true;
                mmadParams.unitFlag = 0;
                mmadParams.cmatrixInitVal = iter1 == 0;
                Mmad(mmadParams, l0cOffset, l0Offset);
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(static_cast<uint16_t>(mte1Flag));
                l0PingPong_++;
            }

            AscendC::SetFlag<AscendC::HardEvent::M_FIX>(l0cPingPong_ & 0x1);
            AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(l0cPingPong_ & 0x1);

            FixpipeToL1(l1Local_[al1Offset], c1Local, curML0, curNL0);
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(l0cPingPong_ & 0x1);
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(l0cPingPong_ & 0x1);
        }
        mL1_ = Get<MNK_M>(tileShape);
        curKL1 = mL1_ / iterBatch_;
        curML1 /= iterBatch_;
        curML0 /= iterBatch_;
        // second matmul :(m,m) * (k,m,n)
        if (isFirstRound) {
            CopyInA1(p1Global, l1Local_[aL1OneBuffer_ * NUM_TWO + bl1OffsetP2_], curML1, curKL1, m_);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l0cPingPong_ & 0x1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l0cPingPong_ & 0x1);

        AscendC::SetFlag<AscendC::HardEvent::FIX_MTE1>(l0cPingPong_ & 0x1);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_MTE1>(l0cPingPong_ & 0x1);

        uint64_t kL0Iter = Cmct::Gemm::CeilDiv(curKL1, baseK_);
        uint64_t l1BatchOffset = 0;
        uint64_t l0cBatchOffset = 0;
        for (uint64_t batchIdx = 0; batchIdx < iterBatch_; batchIdx++) {
            l0cBatchOffset = batchIdx * curML0 * curNL0;
            l1BatchOffset = batchIdx * curKL1 * AscendC::BLOCK_CUBE;
            for (uint64_t iter1 = 0; iter1 < kL0Iter; ++iter1) {
                uint64_t curK0 = (iter1 + 1 == kL0Iter) ? (curKL1 - iter1 * baseK_) : baseK_;
                // 搬运到L0,开启double buffer
                uint64_t l0Offset = HALF_L0_SIZE * (l0PingPong_ & 0x1);
                uint64_t mte1Flag = ((l0PingPong_ & 0x1) + SIXTH_FLAG);
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(static_cast<uint16_t>(mte1Flag));
                CopyInA2(l0aLocal_[l0Offset],
                         l1Local_[aL1OneBuffer_ * NUM_TWO + bl1OffsetP2_ + iter1 * baseK_ * curML1], curML1, curML0,
                         curK0);
                CopyInB2(l0bLocal_[l0Offset],
                         l1Local_[al1Offset + l1BatchOffset + iter1 * baseK_ * AscendC::BLOCK_CUBE], curKL1, curNL0,
                         curK0, iterBatch_);

                AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(static_cast<uint16_t>(mte1Flag));
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(static_cast<uint16_t>(mte1Flag));
                AscendC::MmadParams mmadParams;
                mmadParams.m = curML0;
                mmadParams.n = curNL0;
                mmadParams.k = curK0;
                mmadParams.disableGemv = true;
                mmadParams.unitFlag = 0;
                mmadParams.cmatrixInitVal = iter1 == 0;
                Mmad(mmadParams, l0cOffset + l0cBatchOffset, l0Offset);
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(static_cast<uint16_t>(mte1Flag));
                l0PingPong_++;
            }
        }

        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l0cPingPong_ & 0x1);
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(l0cPingPong_ & 0x1);
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(l0cPingPong_ & 0x1);
        // 数据搬出到UB
        CopyOut(cTensor, c1Local, curML0, curNL0);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(l0cPingPong_ & 0x1);
        if (curML0 * curNL0 <= HALF_L0C_SIZE) {
            l0cPingPong_++;
        }
    }

private:
    __aicore__ inline void Mmad(AscendC::MmadParams& mmadParams, uint64_t l0cOffset, uint64_t l0abOffset)
    {
        mmadParams.cmatrixSource = false;
        AscendC::Mmad(c1Local_[l0cOffset], l0aLocal_[l0abOffset], l0bLocal_[l0abOffset], mmadParams);
    }

private:
    constexpr static uint16_t MKN_M = 0;
    constexpr static uint16_t MKN_N = 1;
    constexpr static uint16_t MKN_K = 2;
    constexpr static uint16_t MKN_B = 3;
    constexpr static uint16_t ZERO_FLAG = 0;
    constexpr static uint16_t FIRST_FLAG = 1;
    constexpr static uint16_t SECOND_FLAG = 2;
    constexpr static uint16_t THIRD_FLAG = 3;
    constexpr static uint16_t FOURTH_FLAG = 4;
    constexpr static uint16_t FIFTH_FLAG = 5;
    constexpr static uint16_t SIXTH_FLAG = 6;
    constexpr static uint16_t SEVENTH_FLAG = 7;
    constexpr static uint16_t M_ALIGN = 16;
    constexpr static uint16_t TWO_ALIGN = 2;
    constexpr static uint16_t NUM_TWO = 2;
    uint64_t aL1OneBuffer_ = 0;
    uint64_t iterBatch_ = 0;
    uint64_t bl1OffsetP2_ = 0;
    AscendC::LocalTensor<A_T> l0aLocal_{AscendC::TPosition::A2, 0, L0A_SIZE};
    AscendC::LocalTensor<B_T> l0bLocal_{AscendC::TPosition::B2, 0, L0B_SIZE};
    AscendC::LocalTensor<L0cType> c1Local_{AscendC::TPosition::CO1, 0, AscendC::TOTAL_L0C_SIZE};
    AscendC::LocalTensor<A_T> l1Local_{AscendC::TPosition::A1, 0, AscendC::TOTAL_L1_SIZE};
};
} // namespace Block
} // namespace Gemm
} // namespace Cmct
#endif
