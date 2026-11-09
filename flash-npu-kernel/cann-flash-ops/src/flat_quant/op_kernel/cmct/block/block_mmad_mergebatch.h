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
 * \file block_mmad_mergebatch.h
 * \brief
 */

#ifndef MATMUL_BLOCK_BLOCK_MMAD_MERGEBATCH_H
#define MATMUL_BLOCK_BLOCK_MMAD_MERGEBATCH_H
#include "./block_mmad.h"
#include "../utils/layout_utils.h"
#include "../utils/tuple_utils.h"
#include "../policy/dispatch_policy.h"

namespace Cmct {
namespace Gemm {
namespace Block {
template <class L1TileShape_, class L0TileShape_, class AType_, class BType_, class CType_, class BiasType_,
          class TileCopy_>
class BlockMmad<MatmulMergeBatch<>, L1TileShape_, L0TileShape_, AType_, BType_, CType_, BiasType_, TileCopy_> {
public:
    using AType = AType_;
    using BType = BType_;
    using CType = CType_;
    using A_T = typename AType::T;
    using B_T = typename BType::T;
    using C_T = typename CType::T;
    using DispatchPolicy = MatmulMergeBatch<>;
    using TupleShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;

    __aicore__ inline BlockMmad()
    {
        if ASCEND_IS_AIC {
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(ZERO_FLAG);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(FIRST_FLAG);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(SECOND_FLAG);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(THIRD_FLAG);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(ZERO_FLAG);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(FIRST_FLAG);
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(ZERO_FLAG);
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(FIRST_FLAG);
        }
    }

    __aicore__ inline ~BlockMmad()
    {
        if ASCEND_IS_AIC {
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(ZERO_FLAG);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(FIRST_FLAG);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(SECOND_FLAG);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(THIRD_FLAG);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(ZERO_FLAG);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(FIRST_FLAG);
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(ZERO_FLAG);
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(FIRST_FLAG);
        }
    }

public:
    __aicore__ inline void Init(const TupleShape& shape, const TupleShape& iterBatchTuple,
                                const TupleShape& tileL1, const TupleShape& tileL0)
    {
        m_ = Get<DIMENSION_M>(shape);
        n_ = Get<DIMENSION_N>(shape);
        k_ = Get<DIMENSION_K>(shape);
        kL1_ = Get<DIMENSION_K>(tileL1);
        baseK_ = Get<DIMENSION_K>(tileL0);

        batchAL1_ = Get<ZERO_FLAG>(iterBatchTuple);
        batchBL1_ = Get<FIRST_FLAG>(iterBatchTuple);
        batchL0_ = Get<THIRD_FLAG>(iterBatchTuple);
        alignedKa_ =
            AType::isTrans ? CeilAlign(kL1_, AscendC::BLOCK_CUBE) : CeilAlign(kL1_, AscendC::AuxGetC0Size<A_T>());
        alignedKb_ =
            BType::isTrans ? CeilAlign(kL1_, AscendC::AuxGetC0Size<B_T>()) : CeilAlign(kL1_, AscendC::BLOCK_CUBE);
        alignedM_ = !AType::isTrans ? CeilAlign(batchAL1_ * m_, AscendC::BLOCK_CUBE)
                                    : CeilAlign(m_, AscendC::BLOCK_CUBE);
        // 无论什么dtype n轴按16对齐
        alignedN_ = CeilAlign(n_, AscendC::BLOCK_CUBE);

        al1Size_ = !AType::isTrans ? alignedKa_ * alignedM_ : batchAL1_ * alignedM_ * alignedKa_;
        bl1Size_ = batchBL1_ * alignedKb_ * alignedN_;
        l0EventID_ = 0;
        aL1EventID_ = 0;
        bL1EventID_ = 0;
    }

    __aicore__ inline void CopyInA1(const AscendC::GlobalTensor<A_T>& aGlobal,
                                    const AscendC::LocalTensor<A_T>& al1Local, uint64_t realKl1)
    {
        // (b, m, k) -> (k1, b*m/m0, m0, k0)
        AscendC::Nd2NzParams nd2NzParams;
        if constexpr(!AType::isTrans) {
            // mk场景
            nd2NzParams.ndNum = 1;
            nd2NzParams.nValue = curBatchAL1_ * m_;
            nd2NzParams.dValue = realKl1;
            nd2NzParams.srcNdMatrixStride = 0;
            nd2NzParams.srcDValue = k_;
            nd2NzParams.dstNzC0Stride = CeilAlign(nd2NzParams.nValue, AscendC::BLOCK_CUBE);
            nd2NzParams.dstNzNStride = 1;
            nd2NzParams.dstNzMatrixStride = 0;
        } else {
            // (b, k, m) -> (b, m1, k1, k0, m0)
            nd2NzParams.ndNum = curBatchAL1_;
            nd2NzParams.nValue = realKl1;
            nd2NzParams.dValue = m_;
            nd2NzParams.srcNdMatrixStride = m_ * k_;
            nd2NzParams.srcDValue = m_;
            nd2NzParams.dstNzC0Stride = CeilAlign(realKl1, AscendC::BLOCK_CUBE);
            nd2NzParams.dstNzNStride = 1;
            nd2NzParams.dstNzMatrixStride = nd2NzParams.dstNzC0Stride * CeilAlign(m_, AscendC::BLOCK_CUBE);
        }
        AscendC::DataCopy(al1Local, aGlobal, nd2NzParams);
    }

    __aicore__ inline void CopyInB1(const AscendC::GlobalTensor<B_T>& bGlobal,
                                    const AscendC::LocalTensor<B_T>& bl1Local,
                                    uint64_t batchBl1, uint64_t realKl1)
    {
        AscendC::Nd2NzParams nd2NzParams;
        // false (b, k, n) -> (b, n1, k1, k0, n0)
        if constexpr(!BType::isTrans) {
            nd2NzParams.ndNum = batchBl1;
            nd2NzParams.nValue = realKl1;
            nd2NzParams.dValue = n_;
            nd2NzParams.srcNdMatrixStride = k_ * n_;
            nd2NzParams.srcDValue = n_;
            nd2NzParams.dstNzC0Stride = CeilAlign(realKl1, AscendC::BLOCK_CUBE);
            nd2NzParams.dstNzNStride = 1;
            nd2NzParams.dstNzMatrixStride = nd2NzParams.dstNzC0Stride * alignedN_;
        } else {
            // true (b, n, k) -> (k1, b*n1, n0, k0)
            nd2NzParams.ndNum = batchBl1;
            nd2NzParams.nValue = n_;
            nd2NzParams.dValue = realKl1;
            nd2NzParams.srcNdMatrixStride = k_ * n_;
            nd2NzParams.srcDValue = k_;
            nd2NzParams.dstNzC0Stride = alignedN_ * batchBl1;
            nd2NzParams.dstNzNStride = 1;
            nd2NzParams.dstNzMatrixStride = alignedN_ * AscendC::AuxGetC0Size<B_T>();
        }
        AscendC::DataCopy(bl1Local, bGlobal, nd2NzParams);
    }

    __aicore__ inline void CopyInA2(const AscendC::LocalTensor<A_T>& a2Local,
                                    const AscendC::LocalTensor<A_T>& al1Local,
                                    uint64_t realBatchL0, uint64_t realKl1, uint64_t realK0)
    {
        AscendC::LoadData2DParamsV2 loadDataParams;
        if constexpr (!AType::isTrans) {
            // mk场景 (k1, b * m1, m0, k0) => (k1, b*m1, m0, k0)
            loadDataParams.mStartPosition = 0;
            loadDataParams.kStartPosition = 0;
            loadDataParams.mStep = CeilDiv(realBatchL0 * m_, AscendC::BLOCK_CUBE);
            loadDataParams.kStep = CeilDiv(realK0, AscendC::AuxGetC0Size<A_T>());
            loadDataParams.srcStride = CeilDiv(curBatchAL1_ * m_, AscendC::BLOCK_CUBE);
            loadDataParams.dstStride = loadDataParams.mStep;
            loadDataParams.ifTranspose = false;
        } else {
            // km场景 (b*m1, k1, k0, m0) => (k1, b*m1, m0, k0)
            loadDataParams.mStartPosition = 0;
            loadDataParams.kStartPosition = 0;
            loadDataParams.mStep = CeilDiv(realK0, AscendC::BLOCK_CUBE);
            loadDataParams.srcStride = CeilDiv(realKl1, AscendC::BLOCK_CUBE);
            loadDataParams.ifTranspose = true;
            if constexpr (AscendC::IsSameType<A_T, half>::value || AscendC::IsSameType<A_T, bfloat16_t>::value) {
                loadDataParams.kStep = realBatchL0 * Cmct::Gemm::CeilDiv(m_, AscendC::BLOCK_CUBE);
                loadDataParams.dstStride = loadDataParams.kStep;
            } else {
                // actually div 8 then align to 2
                loadDataParams.kStep = realBatchL0 * CeilAlign(
                    Cmct::Gemm::CeilDiv(m_, AscendC::AuxGetC0Size<A_T>()), NUM_TWO);
                loadDataParams.dstStride = loadDataParams.kStep >> 1;
            }
        }
        if constexpr (AscendC::IsSameType<A_T, bfloat16_t>::value) {
            AscendC::LoadData(a2Local, al1Local, loadDataParams);
        } else {
            AscendC::LoadData<A_T>(a2Local, al1Local, loadDataParams);
        }
    }

    __aicore__ inline void CopyInB2(const AscendC::LocalTensor<B_T>& b2Local,
                                    const AscendC::LocalTensor<B_T>& bl1Local,
                                    uint64_t realBatchL0, uint64_t realKl1, uint64_t realK0)
    {
        AscendC::LoadData2DParamsV2 loadDataParams;
        if constexpr (BType::isTrans) {
            // nk场景 (k1, b*n1, n0, k0) => (k1, b*n1, n0, k0)
            loadDataParams.mStartPosition = 0;
            loadDataParams.kStartPosition = 0;
            loadDataParams.mStep = realBatchL0 * CeilDiv(n_, AscendC::BLOCK_CUBE);
            loadDataParams.srcStride = realBatchL0 * CeilDiv(n_, AscendC::BLOCK_CUBE);
            loadDataParams.dstStride = loadDataParams.mStep;
            loadDataParams.ifTranspose = false;
            if constexpr (AscendC::IsSameType<B_T, half>::value || AscendC::IsSameType<B_T, bfloat16_t>::value) {
                loadDataParams.kStep = Cmct::Gemm::CeilDiv(realK0, AscendC::BLOCK_CUBE);
            } else {
                loadDataParams.kStep = Cmct::Gemm::CeilDiv(realK0, AscendC::AuxGetC0Size<A_T>());
            }
        } else {
            // kn场景 (b*n1, k1, k0, n0) => (k1, b*n1, n0, k0)
            loadDataParams.mStartPosition = 0;
            loadDataParams.kStartPosition = 0;
            loadDataParams.mStep = CeilDiv(realK0, AscendC::BLOCK_CUBE);
            loadDataParams.srcStride = CeilDiv(realKl1, AscendC::BLOCK_CUBE);
            loadDataParams.ifTranspose = true;
            if constexpr (AscendC::IsSameType<B_T, half>::value || AscendC::IsSameType<B_T, bfloat16_t>::value) {
                loadDataParams.kStep = realBatchL0 * Cmct::Gemm::CeilDiv(n_, AscendC::BLOCK_CUBE);
                loadDataParams.dstStride = loadDataParams.kStep;
            } else {
                // actually div 8 then align to 2
                loadDataParams.kStep = realBatchL0 * CeilAlign(
                    Cmct::Gemm::CeilDiv(n_, AscendC::AuxGetC0Size<A_T>()), NUM_TWO);
                loadDataParams.dstStride = loadDataParams.kStep >> 1;
            }
        }
        if constexpr (AscendC::IsSameType<B_T, bfloat16_t>::value) {
            AscendC::LoadData(b2Local, bl1Local, loadDataParams);
        } else {
            AscendC::LoadData<B_T>(b2Local, bl1Local, loadDataParams);
        }
    }

    __aicore__ inline void Fixp2Out(const AscendC::GlobalTensor<C_T>& cGlobal,
                                    const AscendC::LocalTensor<float>& l0cLocal)
    {
        // mk场景,合并bm
        AscendC::FixpipeParamsC310<AscendC::CO2Layout::ROW_MAJOR> fixpParams;
        fixpParams.mSize = m_;
        fixpParams.nSize = n_;
        fixpParams.srcStride = CeilAlign(curBatchAL1_ * m_, AscendC::BLOCK_CUBE);
        fixpParams.dstStride = n_;
        if constexpr (AscendC::IsSameType<C_T, half>::value) {
            fixpParams.quantPre = QuantMode_t::F322F16;
        } else if constexpr (AscendC::IsSameType<C_T, bfloat16_t>::value) {
            fixpParams.quantPre = QuantMode_t::F322BF16;
        }
        fixpParams.reluEn = false;
        fixpParams.params.ndNum = curBatchAL1_;
        fixpParams.params.srcNdStride = CeilAlign(curBatchAL1_ * m_, AscendC::BLOCK_CUBE) *
            CeilDiv(n_, AscendC::BLOCK_CUBE) + m_;
        if constexpr (AType::isTrans) {
            fixpParams.srcStride = curBatchAL1_ * CeilAlign(m_, AscendC::BLOCK_CUBE);
            fixpParams.params.srcNdStride = (curBatchAL1_ * CeilDiv(n_, AscendC::BLOCK_CUBE) + 1) *
                CeilAlign(m_, AscendC::BLOCK_CUBE);
        }
        fixpParams.params.dstNdStride = m_ * n_;
        fixpParams.dualDstCtl = 0;
        static constexpr AscendC::FixpipeConfig config = {AscendC::CO2Layout::ROW_MAJOR, true};
        AscendC::Fixpipe<C_T, float, config>(cGlobal, l0cLocal, fixpParams);
    }

    __aicore__ inline void ProcessL0(const AscendC::LocalTensor<B_T>& al1Local,
                                     const AscendC::LocalTensor<B_T>& bl1Local,
                                     uint64_t iterK1, uint64_t iterK0, uint64_t realKl1, uint64_t realK0,
                                     uint64_t realBatchL0)
    {
        // 搬运AL0
        AscendC::LocalTensor<A_T> a2Local = l0aLocal_[L0A_ELEMENTS / BUFFER_NUM * (l0EventID_ & 0x1)];
        // false (k1, (b*m)/m0, m0, k0)
        // true (b*m1, k1, k0, m0)
        uint64_t realAlignedM = CeilAlign(realBatchL0 * m_, AscendC::BLOCK_CUBE);
        uint64_t al1Offset = AType::isTrans
                                ? (iterK0 * baseK_ * AscendC::AuxGetC0Size<A_T>())
                                : (iterK0 * baseK_ * realAlignedM);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0EventID_ & 0x1);
        CopyInA2(a2Local, al1Local[al1Offset], realBatchL0, realKl1, realK0);
        // 搬运BL0
        AscendC::LocalTensor<B_T> b2Local = l0bLocal_[L0B_ELEMENTS / BUFFER_NUM * (l0EventID_ & 0x1)];
        // false (b, n1, k1, k0, n0)
        // true (k1, b*n1, n0, k0)
        uint64_t bl1Offset = BType::isTrans ? (iterK0 * realBatchL0 * baseK_ * alignedN_)
                                            : (iterK0 * baseK_ * AscendC::AuxGetC0Size<A_T>());
        if (iterK0 == 0) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>((bL1EventID_ & 0x1) + L1_EVENT_ID_OFFSET);
        }
        CopyInB2(b2Local, bl1Local[bl1Offset], realBatchL0, realKl1, realK0);

        // MMAD
        AscendC::MmadParams mmadParams;
        mmadParams.m = realAlignedM;
        if constexpr (AType::isTrans) {
            mmadParams.m = realBatchL0 * CeilAlign(m_, AscendC::BLOCK_CUBE);
        }
        mmadParams.n = realBatchL0 * CeilAlign(n_, AscendC::BLOCK_CUBE);
        mmadParams.k = realK0;
        mmadParams.disableGemv = true;
        mmadParams.cmatrixInitVal = iterK0 == 0 && iterK1 == 0;
        AscendC::LocalTensor<float> l0cLocal = c1Local_[l0cDBOffset_];
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0EventID_ & 0x1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0EventID_ & 0x1);
        AscendC::Mmad(l0cLocal, a2Local, b2Local, mmadParams);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0EventID_ & 0x1);
    }

    __aicore__ inline void operator()(AscendC::GlobalTensor<C_T> cGlobal,
                                      AscendC::GlobalTensor<A_T> aGlobal,
                                      AscendC::GlobalTensor<B_T> bGlobal,
                                      uint64_t curBatchAL1)
    {
        if (curBatchAL1 == 0) {
            return;
        }
        curBatchAL1_ = curBatchAL1;
        l0cDBOffset_ = (AscendC::TOTAL_L0C_SIZE / BUFFER_NUM / sizeof(float)) * (l0cEventID_ & 0x1);
        if ASCEND_IS_AIC {
            uint64_t kL1Num = CeilDiv(k_, kL1_);
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(l0cEventID_ & 0x1);
            for (uint64_t iterK1 = 0; iterK1 < kL1Num; ++iterK1) {
                uint64_t realKl1 = (iterK1 == kL1Num - 1) ? (k_ - iterK1 * kL1_) : kL1_;
                uint64_t realKAlign = CeilAlign(realKl1, AscendC::BLOCK_CUBE);
                // 搬运AL1
                AscendC::LocalTensor<A_T> al1Local = l1Local_[al1Size_ * (aL1EventID_ & 0x1)];
                // false:(b, m, k)  true: (b, k, m)
                uint64_t offsetAGlobal = AType::isTrans ? iterK1 * kL1_ * m_ : iterK1 * kL1_;
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(aL1EventID_ & 0x1);
                CopyInA1(aGlobal[offsetAGlobal], al1Local, realKl1);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(aL1EventID_ & 0x1);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(aL1EventID_ & 0x1);

                // 搬运BL1
                uint64_t realBatchBL1 = curBatchAL1_;
                AscendC::LocalTensor<B_T> bl1Local = l1Local_[al1Size_ * BUFFER_NUM +
                    bl1Size_ * (bL1EventID_ & 0x1)];
                // false: (b, k, n)  true: (b, n, k)
                uint64_t offsetBGlobal = BType::isTrans ? (iterK1 * kL1_) : (iterK1 * kL1_ * n_);
                CopyInB1(bGlobal[offsetBGlobal], bl1Local, realBatchBL1, realKl1);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>((bL1EventID_ & 0x1) + L1_EVENT_ID_OFFSET);

                uint64_t kL0Num = CeilDiv(realKl1, baseK_);
                uint64_t realBatchL0 = realBatchBL1;
                // L1和L0之间的k轴循环
                for (uint64_t iterK0 = 0; iterK0 < kL0Num; ++iterK0) {
                    uint64_t realK0 = (iterK0 == kL0Num - 1) ? (realKl1 - iterK0 * baseK_) : baseK_;
                    ProcessL0(al1Local, bl1Local, iterK1, iterK0, realKl1, realK0, realBatchL0);
                    l0EventID_++;
                }
                bL1EventID_++;
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(aL1EventID_ & 0x1);
                aL1EventID_++;
            }
            AscendC::SetFlag<AscendC::HardEvent::M_FIX>(l0cEventID_ & 0x1);
            AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(l0cEventID_ & 0x1);
            Fixp2Out(cGlobal, c1Local_[l0cDBOffset_]);
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(l0cEventID_ & 0x1);
            l0cEventID_++;
        }
    }

private:
    constexpr static uint16_t L1_EVENT_ID_OFFSET = 2;
    constexpr static uint16_t DIMENSION_M = 0;
    constexpr static uint16_t DIMENSION_N = 1;
    constexpr static uint16_t DIMENSION_K = 2;
    constexpr static uint16_t NUM_TWO = 2;
    constexpr static uint16_t ZERO_FLAG = 0;
    constexpr static uint16_t FIRST_FLAG = 1;
    constexpr static uint16_t SECOND_FLAG = 2;
    constexpr static uint16_t THIRD_FLAG = 3;
    constexpr static uint64_t BUFFER_NUM = 2;
    constexpr static uint64_t L0A_ELEMENTS = AscendC::TOTAL_L0A_SIZE / sizeof(A_T);
    constexpr static uint64_t L0B_ELEMENTS = AscendC::TOTAL_L0B_SIZE / sizeof(B_T);
    constexpr static uint64_t L0C_ELEMENTS = AscendC::TOTAL_L0C_SIZE / sizeof(C_T);
    constexpr static uint64_t BT_SIZE = 4 * 1024;
    uint64_t m_;
    uint64_t n_;
    uint64_t k_;
    uint64_t alignedM_{1};
    uint64_t alignedN_{1};
    uint64_t alignedKa_{1};
    uint64_t alignedKb_{1};
    uint64_t aL1EventID_{0};
    uint64_t bL1EventID_{0};
    uint64_t l0EventID_{0};
    uint64_t l0cEventID_{0};
    uint64_t al1Size_{0};
    uint64_t bl1Size_{0};
    uint64_t batchAL1_{1};
    uint64_t curBatchAL1_{1};
    uint64_t batchBL1_{1};
    uint64_t batchL0_{1};
    uint64_t kL1_{1};
    uint64_t baseK_{1};
    uint64_t l0cDBOffset_{0};

    AscendC::LocalTensor<A_T> l0aLocal_{AscendC::TPosition::A2, 0, AscendC::TOTAL_L0A_SIZE};
    AscendC::LocalTensor<B_T> l0bLocal_{AscendC::TPosition::B2, 0, AscendC::TOTAL_L0B_SIZE};
    AscendC::LocalTensor<float> c1Local_{AscendC::TPosition::CO1, 0, AscendC::TOTAL_L0C_SIZE};
    AscendC::LocalTensor<A_T> l1Local_{AscendC::TPosition::A1, 0, AscendC::TOTAL_L1_SIZE};
};
} // namespace Block
} // namespace Gemm
} // namespace Cmct
#endif