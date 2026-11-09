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
 * \file block_mmad_iterbatch.h
 * \brief
 */

#ifndef MATMUL_BLOCK_BLOCK_MMAD_ITERBATCH_H
#define MATMUL_BLOCK_BLOCK_MMAD_ITERBATCH_H
#include "./block_mmad.h"
#include "../utils/layout_utils.h"
#include "../utils/tuple_utils.h"
#include "../policy/dispatch_policy.h"

namespace Cmct {
namespace Gemm {
namespace Block {
template <
    class DispatchPolicy_, class L1TileShape_, class L0TileShape_, class AType_, class BType_, class CType_,
    class BiasType_, class TileCopy_>
class BlockMmad<
    DispatchPolicy_, L1TileShape_, L0TileShape_, AType_, BType_, CType_, BiasType_, TileCopy_,
    AscendC::Std::enable_if_t<
        AscendC::Std::is_base_of_v<MatmulIterBatch<>, DispatchPolicy_> ||
        AscendC::Std::is_base_of_v<MatmulIterBatch<MatMulL0C2Out::ND_FIXPIPE_1_2>, DispatchPolicy_>>> {
public:
    using AType = AType_;
    using BType = BType_;
    using CType = CType_;
    using BiasT = BiasType_;
    using A_T = typename AType::T;
    using B_T = typename BType::T;
    using C_T = typename CType::T;
    using Bias_T = typename BiasT::T;
    using DispatchPolicy = DispatchPolicy_;
    using TupleShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;
    uint64_t m_;
    uint64_t n_;
    uint64_t k_;
    bool isBias_{false};
    uint64_t alignedM_{1};
    uint64_t alignedN_{1};
    uint64_t alignedK_{1};
    constexpr static uint64_t BUFFER_NUM = 2;
    uint64_t abL1EventID_{0};
    uint64_t l0EventID_{0};
    uint64_t l0CEventID_{0};
    uint64_t biasEventID_{0};
    uint64_t l0AOffset_ = AscendC::TOTAL_L0A_SIZE / BUFFER_NUM / sizeof(A_T);
    uint64_t l0BOffset_ = AscendC::TOTAL_L0B_SIZE / BUFFER_NUM / sizeof(B_T);
    uint64_t l0COffset_ = AscendC::TOTAL_L0C_SIZE / BUFFER_NUM / sizeof(float);
    uint64_t innerBatch_{0};

    __aicore__ inline BlockMmad()
    {
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(ZERO_FLAG);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(FIRST_FLAG);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(SECOND_FLAG);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(THIRD_FLAG);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(ZERO_FLAG);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(FIRST_FLAG);
    }

    __aicore__ inline ~BlockMmad()
    {
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(ZERO_FLAG);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(FIRST_FLAG);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(SECOND_FLAG);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(THIRD_FLAG);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(ZERO_FLAG);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(FIRST_FLAG);
    }

public:
    __aicore__ inline void Init(const TupleShape& shape, uint64_t innerBatch, uint64_t mainIterBatchL1, bool isBias)
    {
        m_ = Get<DIMENSION_M>(shape);
        n_ = Get<DIMENSION_N>(shape);
        k_ = Get<DIMENSION_K>(shape);
        // when fp16 or (fp32 and m,k), m align to 16; when fp32 and k,m, m align to 8 * 2 for frac combine in loadtol0a
        alignedM_ = CeilAlign(m_, AscendC::BLOCK_CUBE);
        // when fp16 or (fp32 and n,k), m align to 16; when fp32 and k,n, n align to 8 * 2 for frac combine in loadtol0b
        alignedN_ = CeilAlign(n_, AscendC::BLOCK_CUBE);
        alignedK_ = CeilAlign(k_, AscendC::BLOCK_CUBE);
        isBias_ = isBias;
        l0EventID_ = 0;
        abL1EventID_ = 0;
        innerBatch_ = innerBatch;
        if (isBias_) {
            biasL1Offset_ = alignedN_ * sizeof(Bias_T) / sizeof(A_T) * BUFFER_NUM;
        }
        uint64_t aL1OneBuffer = alignedM_ * alignedK_ * mainIterBatchL1;
        bL1Init_ = biasL1Offset_ + aL1OneBuffer * BUFFER_NUM;
    }
    __aicore__ inline void CopyInA1(const AscendC::GlobalTensor<A_T>& aGlobal,
                                    const AscendC::LocalTensor<A_T>& al1Local, const uint64_t curIterBatchL1,
                                    const uint64_t mInGM, const uint64_t kaInGM, const uint64_t mInL1A,
                                    const uint64_t kaInL1A)
    {
        AscendC::Nd2NzParams nd2nzParams;
        nd2nzParams.ndNum = curIterBatchL1; // represent how many matrices load to l1, so it is iterbatchl1 num.
        nd2nzParams.nValue = AType::isTrans ? kaInGM : mInGM;          // source N size
        nd2nzParams.dValue = AType::isTrans ? mInGM : kaInGM;          // source D size
        nd2nzParams.srcNdMatrixStride = mInGM * kaInGM;                // source gap of one block
        nd2nzParams.srcDValue = AType::isTrans ? mInGM : kaInGM;       // source D gap of one block
        nd2nzParams.dstNzC0Stride = AType::isTrans ? kaInL1A : mInL1A; // dst gap of one fractal nz block
        nd2nzParams.dstNzNStride = 1;                                  // dst N gap of one block, unit of matrix
        nd2nzParams.dstNzMatrixStride = mInL1A * kaInL1A;
        AscendC::DataCopy(al1Local, aGlobal, nd2nzParams);
    }

    __aicore__ inline void CopyInB1(const AscendC::GlobalTensor<B_T>& bGlobal,
                                    const AscendC::LocalTensor<B_T>& bl1Local, const uint64_t curIterBatchL1,
                                    const uint64_t kbInGM, const uint64_t nInGM, const uint64_t kbInL1B,
                                    const uint64_t nInL1B)
    {
        AscendC::Nd2NzParams nd2nzParams;
        if (innerBatch_ > 0) {
            nd2nzParams.ndNum = curIterBatchL1;
            uint64_t nDim = BType::isTrans ? nInGM : kbInGM;
            uint64_t dDim = BType::isTrans ? kbInGM : nInGM;

            nd2nzParams.nValue = nDim;
            nd2nzParams.dValue = dDim;
            nd2nzParams.srcNdMatrixStride = dDim;
            nd2nzParams.srcDValue = BType::isTrans ? innerBatch_ * kbInGM : innerBatch_ * nInGM;
            nd2nzParams.dstNzC0Stride = BType::isTrans ? nInL1B : kbInL1B;
            nd2nzParams.dstNzNStride = 1;
            nd2nzParams.dstNzMatrixStride = nInL1B * kbInL1B;
        } else {
            nd2nzParams.ndNum = curIterBatchL1;
            uint64_t nDim = BType::isTrans ? nInGM : kbInGM;
            uint64_t dDim = BType::isTrans ? kbInGM : nInGM;

            nd2nzParams.nValue = BType::isTrans ? nInGM : kbInGM;
            nd2nzParams.dValue = BType::isTrans ? kbInGM : nInGM;
            nd2nzParams.srcNdMatrixStride = nInGM * kbInGM;
            nd2nzParams.srcDValue = BType::isTrans ? kbInGM : nInGM;
            nd2nzParams.dstNzC0Stride = BType::isTrans ? nInL1B : kbInL1B;
            nd2nzParams.dstNzNStride = 1;
            nd2nzParams.dstNzMatrixStride = nInL1B * kbInL1B;
        }

        AscendC::DataCopy(bl1Local, bGlobal, nd2nzParams);
    }

    __aicore__ inline void CopyInC1(const AscendC::GlobalTensor<Bias_T>& biasGlobal,
                                    const AscendC::LocalTensor<Bias_T>& cl1Local, uint64_t nInL1B, bool needBias)
    {
        if (!needBias) {
            return;
        }
        AscendC::DataCopyPadParams padParams;
        // 单位为Byte
        AscendC::DataCopyParams biasParam{1, static_cast<uint16_t>(nInL1B * sizeof(Bias_T)), 0, 0};
        AscendC::DataCopyPad(cl1Local, biasGlobal, biasParam, padParams);
    }

    __aicore__ inline void CopyInA2(const AscendC::LocalTensor<A_T>& a2Local, const AscendC::LocalTensor<A_T>& al1Local,
        uint64_t kaInL0A, uint64_t mInL0A, uint64_t kaInL1A, uint64_t mInL1A, uint64_t curIterBatchL0)
    {
        if constexpr (!AType::isTrans) {
            AscendC::LoadData2DParamsV2 loadDataParams;
            loadDataParams.mStartPosition = 0;
            loadDataParams.kStartPosition = 0;
            loadDataParams.mStep = CeilDiv(mInL0A, AscendC::BLOCK_CUBE);
            loadDataParams.kStep = CeilDiv(kaInL0A, AscendC::AuxGetC0Size<A_T>());
            loadDataParams.srcStride = CeilDiv(mInL1A, AscendC::BLOCK_CUBE);
            loadDataParams.dstStride = loadDataParams.mStep;
            loadDataParams.ifTranspose = false;
            for (uint64_t iterL0AIndex = 0; iterL0AIndex < curIterBatchL0; iterL0AIndex++) {
                if constexpr (AscendC::IsSameType<A_T, bfloat16_t>::value) {
                    AscendC::LoadData(a2Local[iterL0AIndex * mInL1A * kaInL1A],
                                      al1Local[iterL0AIndex * mInL1A * kaInL1A], loadDataParams);
                } else {
                    AscendC::LoadData<A_T>(a2Local[iterL0AIndex * mInL1A * kaInL1A],
                                           al1Local[iterL0AIndex * mInL1A * kaInL1A], loadDataParams);
                }
            }
        } else {
            AscendC::LoadData2DParamsV2 loadDataParams;
            loadDataParams.mStartPosition = 0;
            loadDataParams.kStartPosition = 0;
            loadDataParams.mStep = CeilDiv(kaInL0A, AscendC::BLOCK_CUBE);
            loadDataParams.kStep = CeilDiv(CeilAlign(mInL0A, AscendC::BLOCK_CUBE), AscendC::AuxGetC0Size<A_T>());
            loadDataParams.srcStride = CeilDiv(kaInL1A, AscendC::BLOCK_CUBE);
            loadDataParams.dstStride = CeilDiv(mInL0A, AscendC::BLOCK_CUBE);
            loadDataParams.ifTranspose = true;
            for (uint64_t iterL0AIndex = 0; iterL0AIndex < curIterBatchL0; iterL0AIndex++) {
                if constexpr (AscendC::IsSameType<A_T, bfloat16_t>::value) {
                    AscendC::LoadData(a2Local[iterL0AIndex * mInL1A * kaInL1A],
                                      al1Local[iterL0AIndex * mInL1A * kaInL1A], loadDataParams);
                } else {
                    AscendC::LoadData<A_T>(a2Local[iterL0AIndex * mInL1A * kaInL1A],
                                           al1Local[iterL0AIndex * mInL1A * kaInL1A], loadDataParams);
                }
            }
        }
    }

    __aicore__ inline void CopyInB2(const AscendC::LocalTensor<B_T>& b2Local, const AscendC::LocalTensor<B_T>& bl1Local,
        uint64_t kbInL0B, uint64_t nInL0B, uint64_t kbInL1B, uint64_t nInL1B, uint64_t curIterBatchL0)
    {
        if constexpr (BType::isTrans) {
            AscendC::LoadData2DParamsV2 loadDataParams;
            loadDataParams.mStartPosition = 0;
            loadDataParams.kStartPosition = 0;
            loadDataParams.mStep = CeilDiv(nInL0B, AscendC::BLOCK_CUBE);
            loadDataParams.kStep = CeilDiv(kbInL0B, AscendC::AuxGetC0Size<B_T>());
            loadDataParams.srcStride = CeilDiv(nInL1B, AscendC::BLOCK_CUBE);
            loadDataParams.dstStride = loadDataParams.mStep;
            loadDataParams.ifTranspose = false;
            for (uint64_t iterL0BIndex = 0; iterL0BIndex < curIterBatchL0; iterL0BIndex++) {
                if constexpr (AscendC::IsSameType<B_T, bfloat16_t>::value) {
                    AscendC::LoadData(b2Local[iterL0BIndex * kbInL1B * nInL1B],
                                      bl1Local[iterL0BIndex * kbInL1B * nInL1B], loadDataParams);
                } else {
                    AscendC::LoadData<B_T>(b2Local[iterL0BIndex * kbInL1B * nInL1B],
                                           bl1Local[iterL0BIndex * kbInL1B * nInL1B], loadDataParams);
                }
            }
        } else {
            AscendC::LoadData2DParamsV2 loadDataParams;
            loadDataParams.mStartPosition = 0;
            loadDataParams.kStartPosition = 0;
            loadDataParams.mStep = CeilDiv(kbInL0B, AscendC::BLOCK_CUBE);
            loadDataParams.kStep = CeilDiv(CeilAlign(nInL0B, AscendC::BLOCK_CUBE), AscendC::AuxGetC0Size<B_T>());
            loadDataParams.srcStride = CeilDiv(kbInL1B, AscendC::BLOCK_CUBE);
            loadDataParams.dstStride = CeilDiv(nInL0B, AscendC::BLOCK_CUBE);
            loadDataParams.ifTranspose = true;
            for (uint64_t iterL0BIndex = 0; iterL0BIndex < curIterBatchL0; iterL0BIndex++) {
                if constexpr (AscendC::IsSameType<B_T, bfloat16_t>::value) {
                    AscendC::LoadData(b2Local[iterL0BIndex * kbInL1B * nInL1B],
                                      bl1Local[iterL0BIndex * kbInL1B * nInL1B], loadDataParams);
                } else {
                    AscendC::LoadData<B_T>(b2Local[iterL0BIndex * kbInL1B * nInL1B],
                                           bl1Local[iterL0BIndex * kbInL1B * nInL1B], loadDataParams);
                }
            }
        }
    }

    __aicore__ inline void CopyInC2(const AscendC::LocalTensor<float>& biasBt,
                                    const AscendC::LocalTensor<Bias_T>& biasL1Local, uint64_t alignedNL0, bool needBias)
    {
        if (!needBias) {
            return;
        }
        // s32场景要对齐到2 因此是align(alignedNL0 / 8, 2)
        constexpr uint64_t btAlign = AscendC::BLOCK_CUBE / BIAS_C0;
        uint16_t bustLenth = Cmct::Gemm::Align(alignedNL0 / BIAS_C0, btAlign);
        AscendC::DataCopyParams biasParam{1, static_cast<uint16_t>(bustLenth), 0, 0};
        // 当dstlocal位于C2时，C2中至少为fp32*16
        AscendC::DataCopy(biasBt, biasL1Local, biasParam);
    }

    __aicore__ inline void Mmad(const AscendC::LocalTensor<A_T>& l0a, const AscendC::LocalTensor<B_T>& l0b,
                                const AscendC::LocalTensor<float>& l0c, const AscendC::LocalTensor<float>& biasBt,
                                const uint64_t mInGM, const uint64_t nInGM, const uint64_t kInGM, const uint64_t mInL0a,
                                const uint64_t kaInL0a, const uint64_t kbInL0b, const uint64_t nInL0b,
                                const uint64_t mInL0c, const uint64_t nInL0c, const uint64_t curIterBatchL0,
                                const bool cmatrixInitVal, bool needBias)
    {
        AscendC::MmadParams mmadParams;
        mmadParams.m = mInGM;
        mmadParams.n = nInGM;
        mmadParams.k = kInGM;
        mmadParams.unitFlag = 0; // each l0 only process one block, disable unit flag.
        mmadParams.cmatrixInitVal = cmatrixInitVal;
        mmadParams.disableGemv = true; // disable gemv when m equals 1, which is not capable.
        mmadParams.cmatrixSource = needBias;
        if (needBias) {
        for (uint64_t iterL0CIndex = 0; iterL0CIndex < curIterBatchL0; iterL0CIndex++) {
                AscendC::Mmad(l0c[iterL0CIndex * mInL0c * nInL0c], l0a[iterL0CIndex * mInL0a * kaInL0a],
                              l0b[iterL0CIndex * kbInL0b * nInL0b], biasBt, mmadParams);
            }
        } else {
            for (uint64_t iterL0CIndex = 0; iterL0CIndex < curIterBatchL0; iterL0CIndex++) {
                AscendC::Mmad(l0c[iterL0CIndex * mInL0c * nInL0c], l0a[iterL0CIndex * mInL0a * kaInL0a],
                          l0b[iterL0CIndex * kbInL0b * nInL0b], mmadParams);
            }
        }
    }

    __aicore__ inline void CopyOut(
        const AscendC::GlobalTensor<C_T>& cGlobal, const AscendC::LocalTensor<float>& l0c, const uint64_t mInGM,
        const uint64_t nInGM, const uint64_t curIterBatchL0)
    {
        AscendC::DataCopyCO12DstParams intriParams;
        intriParams.nSize = nInGM;
        intriParams.mSize = mInGM;
        intriParams.dstStride = n_;
        intriParams.srcStride = Align(mInGM, AscendC::BLOCK_CUBE);
        if constexpr (AscendC::IsSameType<C_T, bfloat16_t>::value) {
            intriParams.quantPre = QuantMode_t::F322BF16;
        } else if (AscendC::IsSameType<C_T, half>::value) {
            intriParams.quantPre = QuantMode_t::F322F16;
        }
        intriParams.nz2ndEn = true;
        intriParams.unitFlag = 0;

        // When nz2nd loop in copyout, src stride is unit of c0Size, dst stride is unit of one element.
        AscendC::SetFixpipeNz2ndFlag(curIterBatchL0, Align(mInGM, AscendC::BLOCK_CUBE) *
                                     Align(nInGM, AscendC::BLOCK_CUBE) / AscendC::BLOCK_CUBE, mInGM * nInGM);
        AscendC::DataCopy(cGlobal, l0c, intriParams);
    }

    __aicore__ inline void CopyOut(
        const AscendC::LocalTensor<C_T>& dstLocal, const AscendC::LocalTensor<float>& l0c, const uint64_t mInGM,
        const uint64_t nInGM, const uint64_t curIterBatchL0)
    {
        AscendC::FixpipeParamsC310<AscendC::CO2Layout::ROW_MAJOR> fixpipeParams;
        fixpipeParams.nSize = Align(nInGM, AscendC::BLOCK_CUBE);
        fixpipeParams.mSize = Align(mInGM, AscendC::BLOCK_CUBE);
        fixpipeParams.dstStride = Align(nInGM, AscendC::BLOCK_CUBE);
        fixpipeParams.srcStride = Align(mInGM, AscendC::BLOCK_CUBE);
        if constexpr (AscendC::IsSameType<C_T, bfloat16_t>::value) {
            fixpipeParams.quantPre = QuantMode_t::F322BF16;
        } else if (AscendC::IsSameType<C_T, half>::value) {
            fixpipeParams.quantPre = QuantMode_t::F322F16;
        }
        fixpipeParams.unitFlag = 0;
        fixpipeParams.params.ndNum = curIterBatchL0;
        fixpipeParams.params.srcNdStride =
            Align(mInGM, AscendC::BLOCK_CUBE) * Align(nInGM, AscendC::BLOCK_CUBE) / AscendC::BLOCK_CUBE;
        fixpipeParams.params.dstNdStride = mInGM * Align(nInGM, AscendC::BLOCK_CUBE);
        fixpipeParams.subBlockId = (l0CEventID_ & 0x1);
        AscendC::Fixpipe<C_T, float, AscendC::Impl::CFG_ROW_MAJOR_UB>(dstLocal, l0c, fixpipeParams);
    }

    template <typename T>
    __aicore__ inline void operator()(T cTensor,
                                      AscendC::GlobalTensor<A_T> aGlobal,
                                      AscendC::GlobalTensor<B_T> bGlobal,
                                      AscendC::GlobalTensor<Bias_T> biasGlobal,
                                      uint64_t blockNum,
                                      uint64_t curIterBatchL1,
                                      uint64_t nextIterBatchL1,
                                      uint64_t mainIterBatchL1,
                                      uint64_t mainIterBatchL0,
                                      uint64_t baseM,
                                      uint64_t baseN,
                                      uint64_t baseK,
                                      bool isPreLoadRound,
                                      bool isFinalRound)
    {
        AscendC::LocalTensor<Bias_T> biasL1Local = l1Local_.template ReinterpretCast<Bias_T>();
        AscendC::LocalTensor<A_T> al1Local = l1Local_[biasL1Offset_]; // start of l1
        AscendC::LocalTensor<B_T> bl1Local = l1Local_[bL1Init_];
        // mov align to L1 with pingpong
        if (isPreLoadRound) { // first round, copy first loop of data
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(abL1EventID_ & 0x1); // wait last loop mte1 to finish
            CopyInA1(aGlobal, al1Local[alignedM_ * alignedK_ * mainIterBatchL1 * (abL1EventID_ & 0x1)],
                     curIterBatchL1, m_, k_, alignedM_, alignedK_);
            CopyInC1(biasGlobal, biasL1Local[alignedN_ * (abL1EventID_ & 0x1)], alignedN_, isBias_);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(abL1EventID_ & 0x1); // set current loop mte1 to wait

            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>((abL1EventID_ & 0x1) + L1_EVENT_ID_OFFSET);
            CopyInB1(bGlobal, bl1Local[alignedN_ * alignedK_ * mainIterBatchL1 * (abL1EventID_ & 0x1)],
                     curIterBatchL1, k_, n_, alignedK_, alignedN_);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>((abL1EventID_ & 0x1) + L1_EVENT_ID_OFFSET);
        }
        if (!isFinalRound) { // before last round need to precopy next loop of data
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>((abL1EventID_ + 1) & 0x1); // wait last loop mte1 to finish
            CopyInA1(aGlobal[m_ * k_ * mainIterBatchL1 * blockNum], al1Local[alignedM_ * alignedK_ *
                     mainIterBatchL1 * ((abL1EventID_ + 1) & 0x1)], nextIterBatchL1, m_, k_, alignedM_, alignedK_);
            CopyInC1(biasGlobal, biasL1Local[alignedN_ * ((abL1EventID_ + 1) & 0x1)], alignedN_, isBias_);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>((abL1EventID_ + 1) & 0x1); // set current loop mte1 to wait

            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(((abL1EventID_ + 1) & 0x1) + L1_EVENT_ID_OFFSET);
            int64_t offsetB = k_ * n_ * mainIterBatchL1 * blockNum;
            if (innerBatch_ > 0) {
                if (BType::isTrans) {
                    offsetB = k_ * mainIterBatchL1 * blockNum;
                } else {
                    offsetB = n_ * mainIterBatchL1 * blockNum;
                }
            }
            CopyInB1(bGlobal[offsetB], bl1Local[alignedN_ * alignedK_ *
                     mainIterBatchL1 * ((abL1EventID_ + 1) & 0x1)], nextIterBatchL1, k_, n_, alignedK_, alignedN_);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(((abL1EventID_ + 1) & 0x1) + L1_EVENT_ID_OFFSET);
        }
        uint64_t mL0Cnt = CeilDiv(m_, baseM);
        uint64_t nL0Cnt = CeilDiv(n_, baseN);
        uint64_t kL0Cnt = CeilDiv(k_, baseK);

        // calculate how much loop needed between l1 and l0.
        uint64_t stepIterBatchL1L0 = CeilDiv(curIterBatchL1, mainIterBatchL0);
        for (uint64_t iter1 = 0; iter1 < stepIterBatchL1L0; ++iter1) {
            uint64_t curIterBatchL0 = (iter1 + 1 == stepIterBatchL1L0) ? // if tailloop of l1 and l0, cal tail iter num.
                                      (curIterBatchL1 - mainIterBatchL0 * iter1) : mainIterBatchL0;
            for (uint64_t iterNL0 = 0; iterNL0 < nL0Cnt; ++iterNL0) {
                uint64_t curNL0 = (iterNL0 == nL0Cnt - 1) ? (n_ - (nL0Cnt - 1) * baseN) : baseN;
                for (uint64_t iterML0 = 0; iterML0 < mL0Cnt; ++iterML0) {
                    uint64_t curML0 = (iterML0 == mL0Cnt - 1) ? (m_ - (mL0Cnt - 1) * baseM) : baseM;
                    for (uint64_t iterKL0 = 0; iterKL0 < kL0Cnt; ++iterKL0) {
                        uint64_t curKL0 = (iterKL0 == kL0Cnt - 1) ? (k_ - (kL0Cnt - 1) * baseK) : baseK;
                        if (iter1 == 0 && iterNL0 == 0 && iterML0 == 0 && iterKL0 == 0) {
                            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(abL1EventID_ & 0x1);
                        }
                        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0EventID_ & 0x1);
                        uint64_t alignedNL0 = CeilAlign(curNL0, AscendC::BLOCK_CUBE);
                        CopyInC2(biasBt_[baseN * (biasEventID_ & 0x1)],
                                 biasL1Local[alignedN_ * (abL1EventID_ & 0x1) + iterNL0 * baseN], alignedNL0,
                                 (isBias_ && iterML0 == 0 && iterKL0 == 0));
                        uint64_t offsetL1AOfCopyInA2 = alignedM_ * alignedK_ * mainIterBatchL1 * (abL1EventID_ & 0x1) +
                                                       iter1 * mainIterBatchL0 * alignedM_ * alignedK_ +
                                                       (AType::isTrans ? (iterML0 * alignedK_ * baseM +
                                                       iterKL0 * baseK * AscendC::AuxGetC0Size<A_T>()) :
                                                       (iterML0 * AscendC::AuxGetC0Size<A_T>() * baseM +
                                                       iterKL0 * alignedM_ * baseK));
                        CopyInA2(l0a_[l0AOffset_ * (l0EventID_ & 0x1)], al1Local[offsetL1AOfCopyInA2], curKL0, curML0,
                                 alignedK_, alignedM_, curIterBatchL0);
                        if ((iter1 == stepIterBatchL1L0 - 1) && (iterNL0 == nL0Cnt - 1) && (iterML0 == mL0Cnt - 1) &&
                             (iterKL0 == kL0Cnt - 1)) {
                            // after last loop, notice Mte2 to wait Mte1
                            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(abL1EventID_ & 0x1);
                        }

                        if (iter1 == 0 && iterNL0 == 0 && iterML0 == 0 && iterKL0 == 0) {
                            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>((abL1EventID_ & 0x1) + L1_EVENT_ID_OFFSET);
                        }
                        uint64_t offsetL1BOfCopyInB2 = alignedN_ * alignedK_ * mainIterBatchL1 * (abL1EventID_ & 0x1) +
                                                       iter1 * mainIterBatchL0 * alignedK_ * alignedN_ +
                                                       (BType::isTrans ?
                                                       (iterNL0 * AscendC::AuxGetC0Size<B_T>() * baseN +
                                                       iterKL0 * baseK * alignedN_) :
                                                       (iterNL0 * alignedK_ * baseN +
                                                       iterKL0 * baseK * AscendC::AuxGetC0Size<B_T>()));
                        CopyInB2(l0b_[l0BOffset_ * (l0EventID_ & 0x1)], bl1Local[offsetL1BOfCopyInB2], curKL0, curNL0,
                                 alignedK_, alignedN_, curIterBatchL0);
                        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0EventID_ & 0x1);
                        if ((iter1 == stepIterBatchL1L0 - 1) && (iterNL0 == nL0Cnt - 1) && (iterML0 == mL0Cnt - 1) &&
                            (iterKL0 == kL0Cnt - 1)) {
                            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>((abL1EventID_ & 0x1) + L1_EVENT_ID_OFFSET);
                        }

                        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0EventID_ & 0x1);
                        if (iterKL0 == 0) {
                            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(l0CEventID_ & 0x1);
                        }
                        bool cmatrixInitVal = (iterKL0 == 0 && !isBias_);
                        Mmad(l0a_[l0AOffset_ * (l0EventID_ & 0x1)], l0b_[l0BOffset_ * (l0EventID_ & 0x1)],
                             l0c_[l0COffset_ * (l0CEventID_ & 0x1)], biasBt_[baseN * (biasEventID_ & 0x1)], curML0,
                             curNL0, curKL0, alignedM_, alignedK_, alignedK_, alignedN_, alignedM_, alignedN_,
                             curIterBatchL0, cmatrixInitVal, (isBias_ && iterKL0 == 0));
                        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0EventID_ & 0x1);
                        l0EventID_++;
                    }
                    AscendC::SetFlag<AscendC::HardEvent::M_FIX>(l0CEventID_ & 0x1);

                    AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(l0CEventID_ & 0x1);
                    if constexpr (DispatchPolicy::enableSync == MatMulL0C2Out::ND_FIXPIPE_1_2) {
                        if (l0CEventID_ > 1) {
                            AscendC::CrossCoreWaitFlag<AIC_SYNC_AIV_MODE_4, PIPE_FIX>(
                                (l0CEventID_ & 0x1) * FLAG_ID_MAX + SYNC_OFFSET);
                        }
                        CopyOut(cTensor, l0c_[l0COffset_ * (l0CEventID_ & 0x1)], curML0, curNL0, curIterBatchL0);
                        AscendC::CrossCoreSetFlag<AIC_SYNC_AIV_MODE_4, PIPE_FIX>((l0CEventID_ & 0x1) * FLAG_ID_MAX);
                    } else {
                        uint64_t offsetCGMOfCopyOut =
                            iter1 * mainIterBatchL0 * m_ * n_ + iterML0 * baseM * n_ + iterNL0 * baseN;
                        CopyOut(cTensor[offsetCGMOfCopyOut], l0c_[l0COffset_ * (l0CEventID_ & 0x1)], curML0, curNL0,
                            curIterBatchL0);
                    }
                    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(l0CEventID_ & 0x1);
                    l0CEventID_++;
                }
                biasEventID_++;
            }
        }
        abL1EventID_++;
    }

private:
    constexpr static uint16_t L1_EVENT_ID_OFFSET = 2;
    constexpr static uint16_t DIMENSION_M = 0;
    constexpr static uint16_t DIMENSION_N = 1;
    constexpr static uint16_t DIMENSION_K = 2;
    constexpr static uint16_t ZERO_FLAG = 0;
    constexpr static uint16_t FIRST_FLAG = 1;
    constexpr static uint16_t SECOND_FLAG = 2;
    constexpr static uint16_t THIRD_FLAG = 3;
    constexpr static uint16_t FLAG_ID_MAX = 16;
    constexpr static uint16_t SYNC_OFFSET = 2;
    constexpr static int32_t BT_SIZE = 4096;
    constexpr static int32_t BIAS_C0 = AscendC::AuxGetC0Size<Bias_T>();
    uint64_t biasL1Offset_ = 0;
    uint64_t bL1Init_ = 0;
    AscendC::LocalTensor<A_T> l1Local_{AscendC::TPosition::A1, 0, AscendC::TOTAL_L1_SIZE};
    AscendC::LocalTensor<A_T> l0a_{AscendC::TPosition::A2, 0, AscendC::TOTAL_L0A_SIZE};
    AscendC::LocalTensor<B_T> l0b_{AscendC::TPosition::B2, 0, AscendC::TOTAL_L0B_SIZE};
    AscendC::LocalTensor<float> l0c_{AscendC::TPosition::CO1, 0, AscendC::TOTAL_L0C_SIZE};
    AscendC::LocalTensor<float> biasBt_{AscendC::TPosition::C2, 0, BT_SIZE};
};
} // namespace Block
} // namespace Gemm
} // namespace Cmct
#endif