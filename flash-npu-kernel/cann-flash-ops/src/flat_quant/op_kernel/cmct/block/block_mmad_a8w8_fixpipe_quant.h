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
 * \file block_mmad_a8w8_fixpipe_quant.h
 * \brief
 */

#ifndef BLOCK_MMAD_A8W8_FIXPIPE_QUANT_H
#define BLOCK_MMAD_A8W8_FIXPIPE_QUANT_H
#include "../policy/dispatch_policy.h"
#include "../utils/common_utils.h"
#include "../utils/layout_utils.h"
#include "../utils/quant_batch_matmul_constant.h"
#include "../utils/tuple_utils.h"

namespace {
constexpr uint16_t SCALE_BUFFER_NUM = 2;
constexpr uint16_t AB_L1_TWO_BUFFER = 2;

constexpr uint64_t L0_TRANS_ALIGN = 2UL;

constexpr uint16_t BIAS_BUFFER_FLAG_0 = 4;
constexpr uint16_t BIAS_BUFFER_FLAG_1 = 5;
constexpr uint16_t X2_SCALE_BUFFER_FLAG_0 = 0;
constexpr uint16_t X2_SCALE_BUFFER_FLAG_1 = 1;
}

namespace Cmct {
namespace Gemm {
namespace Block {
using namespace AscendC;
using namespace Cmct::Gemm::QuantBatchMatmul;

template <class DispatchPolicy_, class L1TileShape_, class L0TileShape_, class AType_, class LayoutA_, class BType_,
          class LayoutB_, class CType_, class LayoutC_, class BiasType_, class LayoutBias_, class scaleType_,
          class Enable = void>
class BlockMmadA8W8FixpipeQuant {
    static_assert(AscendC::Std::always_false_v<DispatchPolicy_>, "Should not be here!");
};

template <class DispatchPolicy_, class L1TileShape_, class L0TileShape_, class AType_, class LayoutA_, class BType_,
          class LayoutB_, class CType_, class LayoutC_, class BiasType_, class LayoutBias_, class ScaleType_>
class BlockMmadA8W8FixpipeQuant<DispatchPolicy_, L1TileShape_, L0TileShape_, AType_, LayoutA_, BType_, LayoutB_, CType_, LayoutC_,
                    BiasType_, LayoutBias_, ScaleType_,
                    AscendC::Std::enable_if_t<
                        AscendC::Std::is_base_of_v<MatmulWithScale<>, DispatchPolicy_> ||
                        AscendC::Std::is_base_of_v<MatmulWithScale<AscendC::Shape<_0, _0, _0, _0>, A_FULL_LOAD_MODE>,
                                                   DispatchPolicy_>>> {
public:
    using AType = AType_;
    using BType = BType_;
    using CType = CType_;
    using BiasType = BiasType_;
    using X2ScaleType = ScaleType_;
    using LayoutA = LayoutA_;
    using LayoutB = LayoutB_;
    using LayoutC = LayoutC_;
    using L0CType = typename GetMmDstType<AType>::Type;
    using L1TileShape = L1TileShape_;
    using L0TileShape = L0TileShape_;
    using DispatchPolicy = DispatchPolicy_;
    using TupleShape = AscendC::Shape<int64_t, int64_t, int64_t>;
    using BlockShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;
    uint64_t m_;
    uint64_t n_;
    uint64_t k_;
    uint64_t l1BufNum_{1};
    uint64_t kL1Iter_{0};
    uint64_t kAL1Iter_{0};
    uint64_t kBL1Iter_{0};
    uint64_t kL1_{1};
    uint64_t kAL1_{1};
    uint64_t kBL1_{1};
    uint64_t baseM_{16};
    uint64_t baseN_{16};
    uint64_t baseK_{16};
    bool isBias_{false};
    uint16_t aPingPongID_{0};
    uint16_t bPingPongID_{0};
    uint64_t abL1LoopCnt_{0};
    uint64_t scaleLoopCnt_{0};
    uint64_t biasLoopCnt_{0};
    uint64_t l0PingPong_{0};
    uint64_t l0cPingPong_{0};
    bool enableL0cPingPong_{false};
    bool l1BufferOffsetCalculated_{false};
    static constexpr bool transA = TagToTrans<LayoutA>::value;
    static constexpr bool transB = TagToTrans<LayoutB>::value;

    struct Params {
        GM_ADDR aGmAddr{nullptr};
        GM_ADDR bGmAddr{nullptr};
        GM_ADDR cGmAddr{nullptr};
        GM_ADDR biasGmAddr{nullptr};
        GM_ADDR x1ScaleGmAddr{nullptr};
        GM_ADDR x2ScaleGmAddr{nullptr};
    };

    __aicore__ inline BlockMmadA8W8FixpipeQuant()
    {
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(INPUT_BUFFER_FLAG_0);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(INPUT_BUFFER_FLAG_1);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(INPUT_BUFFER_FLAG_2);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(INPUT_BUFFER_FLAG_3);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(BIAS_BUFFER_FLAG_0);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(BIAS_BUFFER_FLAG_1);
        AscendC::SetFlag<AscendC::HardEvent::FIX_MTE2>(X2_SCALE_BUFFER_FLAG_0);
        AscendC::SetFlag<AscendC::HardEvent::FIX_MTE2>(X2_SCALE_BUFFER_FLAG_1);
        AscendC::SetMMLayoutTransform(true); // true means column first when fixpipe_l0c2out
    }

    __aicore__ inline ~BlockMmadA8W8FixpipeQuant()
    {
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(INPUT_BUFFER_FLAG_0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(INPUT_BUFFER_FLAG_1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(INPUT_BUFFER_FLAG_2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(INPUT_BUFFER_FLAG_3);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(BIAS_BUFFER_FLAG_0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(BIAS_BUFFER_FLAG_1);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_MTE2>(X2_SCALE_BUFFER_FLAG_0);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_MTE2>(X2_SCALE_BUFFER_FLAG_1);
        AscendC::SetMMLayoutTransform(false); // false means row first when fixpipe_l0c2out
    }

public:
    __aicore__ inline void Init(const TupleShape &problemShape, const BlockShape &l0TileShape, const uint64_t &kAL1,
                                const uint64_t &kBL1, const uint64_t &l1BufferNum,
                                QuantBatchMatmul::QuantMode quantMode, bool isBias, bool dbL0C)
    {
        m_ = Get<IDX_M_IDX>(problemShape);
        n_ = Get<IDX_N_IDX>(problemShape);
        k_ = Get<IDX_K_IDX>(problemShape);
        baseM_ = Get<IDX_M_IDX>(l0TileShape);
        baseN_ = Get<IDX_N_IDX>(l0TileShape);
        baseK_ = Get<IDX_K_IDX>(l0TileShape);
        isBias_ = isBias;
        l1BufNum_ = l1BufferNum;
        enableL0cPingPong_ = dbL0C;
        if (quantMode == QuantBatchMatmul::QuantMode::PERCHANNEL_MODE) {
            x2ScaleL1OneBuffer_ = baseN_ * sizeof(uint64_t);
        }
        if (isBias_) {
            biasL1OneBuffer_ = baseN_ * sizeof(BiasType);
        }
        if constexpr (DispatchPolicy::fullLoadMode != 0) {
            kBL1_ = kBL1;
            kAL1_ = kBL1_;
            kL1_ = kBL1;
            uint64_t mAlign = Cmct::Gemm::Align(baseM_, transA ? C0_SIZE : BLOCK_CUBE);
            uint64_t kAlign = Cmct::Gemm::Align(k_, transA ? BLOCK_CUBE : C0_SIZE);
            aL1OneBuffer_ = mAlign * kAlign;
            bL1OneBuffer_ = baseN_ * kL1_;
            kL1Iter_ = CeilDiv(k_, kL1_);
        } else {
            if (l1BufferNum == AB_L1_TWO_BUFFER) {
                kAL1_ = kAL1;
                kBL1_ = kBL1;
                aL1OneBuffer_ = baseM_ * kAL1_;
                bL1OneBuffer_ = baseN_ * kBL1_;
                kAL1Iter_ = CeilDiv(k_, kAL1_);
                kBL1Iter_ = CeilDiv(k_, kBL1_);
                if (kAL1 == kBL1) {
                    kL1_ = kAL1;
                    kL1Iter_ = CeilDiv(k_, kL1_);
                }
            } else {
                kL1_ = Cmct::Gemm::Min(kAL1, kBL1);
                aL1OneBuffer_ = baseM_ * kL1_;
                bL1OneBuffer_ = baseN_ * kL1_;
                kL1Iter_ = CeilDiv(k_, kL1_);
                kAL1_ = kL1_;
                kBL1_ = kL1_;
            }
        }
        GetL1BufferOffset();
    }

    template <typename T>
    __aicore__ inline void operator()(AscendC::GlobalTensor<AType> aGlobal, AscendC::GlobalTensor<BType> bGlobal,
                                      T scaleGlobal, AscendC::GlobalTensor<BiasType> biasGlobal,
                                      AscendC::GlobalTensor<CType> cGlobal, BlockShape singleShape)
    {
        uint64_t curML1 = Get<IDX_M_TILEIDX>(singleShape);
        uint64_t curNL1 = Get<IDX_N_TILEIDX>(singleShape);
        AscendC::MmadParams mmadParams;
        mmadParams.m = curML1;
        mmadParams.n = curNL1;
        mmadParams.disableGemv = true;
        uint64_t l0cOffset = (l0cPingPong_ & 1) * HALF_L0C_SIZE;
        uint16_t scaleL1BufId = scaleLoopCnt_ & 1;
        uint16_t biasBufId = biasLoopCnt_ & 1;
        if constexpr (AscendC::IsSameType<T, AscendC::GlobalTensor<uint64_t>>::value) {
            CopyX2ScaleInL1(scaleGlobal, scaleL1Local_[l1BufferX2ScaleOffset_[scaleL1BufId] / sizeof(uint64_t)], curNL1,
                            scaleL1BufId);
        } else if constexpr (AscendC::IsSameType<T, uint64_t>::value) {
            scalarScale_ = scaleGlobal;
        }
        if (isBias_) {
            CopyBiasInL1(biasGlobal, biasL1Local_[l1BufferBiasOffset_[biasBufId] / sizeof(BiasType)], curNL1,
                         biasBufId);
        }
        if (kAL1_ == kBL1_) {
            IterateABL1(aGlobal, bGlobal, cGlobal, mmadParams, curML1, curNL1, l0cOffset, biasBufId);
        } else if (kAL1_ > kBL1_) {
            IterateAL1BL1(aGlobal, bGlobal, cGlobal, mmadParams, curML1, curNL1, l0cOffset, biasBufId);
        } else {
            IterateBL1AL1(aGlobal, bGlobal, cGlobal, mmadParams, curML1, curNL1, l0cOffset, biasBufId);
        }
        AscendC::LocalTensor<L0CType> c1Local = c1Local_[l0cOffset];
        if constexpr (AscendC::IsSameType<T, uint64_t>::value || AscendC::IsSameType<CType, int32_t>::value) {
            CopyFixpipeScalar(cGlobal, c1Local, mmadParams, scalarScale_);
        } else {
            CopyFixpipeTensor(cGlobal, c1Local, mmadParams,
                              scaleL1Local_[l1BufferX2ScaleOffset_[scaleL1BufId] / sizeof(uint64_t)]);
        }
        if (enableL0cPingPong_) {
            l0cPingPong_++;
        }
        if constexpr (AscendC::IsSameType<T, AscendC::GlobalTensor<uint64_t>>::value) {
            AscendC::SetFlag<AscendC::HardEvent::FIX_MTE2>(scaleL1BufId);
            scaleLoopCnt_++;
        }
        if (isBias_) {
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(BIAS_BUFFER_FLAG_0 + biasBufId);
            biasLoopCnt_++;
        }
    }

private:
    __aicore__ inline void GetL1BufferOffset()
    {
        if constexpr (DispatchPolicy::fullLoadMode == 0) {
             for (uint16_t bufferId = 0; bufferId < l1BufNum_; bufferId++) {
                 // 2 buffer: L1 space is : A0|B0|Scale0|bias0|...|A1|B1|Scale1|bias1|...
                 // 4 buffer: L1 space is : A0A2|B0B2|Scale0|bias0|...|A1A3|B1B3|Scale1|bias1|...
                 uint64_t l1Offset = (AscendC::TOTAL_L1_SIZE >> 1) * (bufferId & 1);
                 l1BufferAOffset_[bufferId] = l1Offset + aL1OneBuffer_ * (bufferId >> 1);
                 l1BufferBOffset_[bufferId] =
                     l1Offset + aL1OneBuffer_ * (l1BufNum_ >> 1) + bL1OneBuffer_ * (bufferId >> 1);
             }
             for (uint16_t bufferId = 0; bufferId < SCALE_BUFFER_NUM; bufferId++) {
                 l1BufferX2ScaleOffset_[bufferId] = l1BufferBOffset_[bufferId] + bL1OneBuffer_ * (l1BufNum_ >> 1);
                 l1BufferBiasOffset_[bufferId] = l1BufferX2ScaleOffset_[bufferId] + x2ScaleL1OneBuffer_;
             }
         } else {
             uint64_t mAlign = Cmct::Gemm::Align(baseM_, transA ? C0_SIZE : BLOCK_CUBE);
             uint64_t kAlign = Cmct::Gemm::Align(k_, transA ? BLOCK_CUBE : C0_SIZE);
             aL1OneBuffer_ = mAlign * kAlign;
             // 2 buffer: L1 space is : B0|Scale0|bias0|A|...|B1|Scale1|bias1|
             // 4 buffer: L1 space is : B0B2|Scale0|bias0|A|...|B1B3|Scale1|bias1|...
             l1BufferAOffset_[0] = bL1OneBuffer_ * (l1BufNum_ >> 1) + x2ScaleL1OneBuffer_ + biasL1OneBuffer_;
             uint64_t b1Offset = l1BufferAOffset_[0] + aL1OneBuffer_ >= (AscendC::TOTAL_L1_SIZE >> 1)
                                 ? l1BufferAOffset_[0] + aL1OneBuffer_ : (AscendC::TOTAL_L1_SIZE >> 1);
             for (uint16_t bufferId = 0; bufferId < l1BufNum_; bufferId++) {
                 l1BufferBOffset_[bufferId] = b1Offset * (bufferId & 1) + bL1OneBuffer_ * (bufferId >> 1);
             }
             for (uint16_t bufferId = 0; bufferId < SCALE_BUFFER_NUM; bufferId++) {
                 l1BufferX2ScaleOffset_[bufferId] = l1BufferBOffset_[bufferId] + bL1OneBuffer_ * (l1BufNum_ >> 1);
                 l1BufferBiasOffset_[bufferId] = l1BufferX2ScaleOffset_[bufferId] + x2ScaleL1OneBuffer_;
             }
         }
    }

    __aicore__ inline void DataCopyA(AscendC::GlobalTensor<AType> aGlobal, uint64_t curML1, uint64_t curKL1,
                                     uint64_t offsetA, uint64_t offsetAL1)
    {
        AscendC::Nd2NzParams nd2nzParams;
        nd2nzParams.ndNum = 1;
        uint64_t nDim = transA ? curKL1 : curML1;
        uint64_t dDim = transA ? curML1 : curKL1;

        nd2nzParams.nValue = nDim;
        nd2nzParams.dValue = dDim;
        nd2nzParams.srcNdMatrixStride = 1;
        nd2nzParams.srcDValue = transA ? m_ : k_;
        nd2nzParams.dstNzC0Stride = Cmct::Gemm::CeilAlign(transA ? curKL1 : curML1, BLOCK_CUBE);
        nd2nzParams.dstNzNStride = 1;
        nd2nzParams.dstNzMatrixStride = 1;
        AscendC::DataCopy(aL1Local_[offsetAL1], aGlobal[offsetA], nd2nzParams);
    }

    __aicore__ inline void CopyAInL1(AscendC::GlobalTensor<AType> aGlobal, uint64_t curML1, uint64_t curKL1,
                                     uint64_t l1Iter, uint16_t l1BufId)
    {
        uint64_t offsetA = transA ? l1Iter * kAL1_ * m_ : l1Iter * kAL1_;
        if constexpr (DispatchPolicy::fullLoadMode == 0) {
            uint64_t offsetAL1 = l1BufferAOffset_[l1BufId];
            DataCopyA(aGlobal, curML1, curKL1, offsetA, offsetAL1);
        } else {
            if (abL1LoopCnt_ < kL1Iter_) {
                uint64_t offsetAL1 =
                    l1BufferAOffset_[0] + l1Iter * kL1_ * Cmct::Gemm::CeilAlign(curML1, transA ? C0_SIZE : BLOCK_CUBE);
                DataCopyA(aGlobal, curML1, curKL1, offsetA, offsetAL1);
            }
        }
    }

    __aicore__ inline void CopyBInL1(AscendC::GlobalTensor<BType> bGlobal, uint64_t curNL1, uint64_t curKL1,
                                     uint64_t l1Iter, uint16_t l1BufId)
    {
        uint64_t offsetB = transB ? l1Iter * kBL1_ : l1Iter * kBL1_ * n_;
        AscendC::Nd2NzParams nd2nzParams;
        nd2nzParams.ndNum = 1;
        uint64_t nDim = transB ? curNL1 : curKL1;
        uint64_t dDim = transB ? curKL1 : curNL1;

        nd2nzParams.nValue = nDim;
        nd2nzParams.dValue = dDim;
        nd2nzParams.srcNdMatrixStride = 1;
        nd2nzParams.srcDValue = transB ? k_ : n_;
        nd2nzParams.dstNzC0Stride = Cmct::Gemm::CeilAlign(transB ? curNL1 : curKL1, BLOCK_CUBE);
        nd2nzParams.dstNzNStride = 1;
        nd2nzParams.dstNzMatrixStride = 1;
        AscendC::DataCopy(bL1Local_[l1BufferBOffset_[l1BufId]], bGlobal[offsetB], nd2nzParams);
    }

    __aicore__ inline void CopyBiasInL1(const AscendC::GlobalTensor<BiasType> &biasGlobal,
                                        const AscendC::LocalTensor<BiasType> &biasL1Local, uint64_t curNL1,
                                        uint16_t biasBufId)
    {
        // Set a sync flag here because bias is using separate sync flags
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(BIAS_BUFFER_FLAG_0 + biasBufId);
        AscendC::DataCopyPadParams padParams;
        // 单位为Byte
        AscendC::DataCopyParams biasParam{1, static_cast<uint16_t>(curNL1 * sizeof(BiasType)), 0, 0};
        AscendC::DataCopyPad(biasL1Local, biasGlobal, biasParam, padParams);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(BIAS_BUFFER_FLAG_0 + biasBufId);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(BIAS_BUFFER_FLAG_0 + biasBufId);
    }

    __aicore__ inline void CopyX2ScaleInL1(const AscendC::GlobalTensor<uint64_t> &scaleGlobal,
                                           const AscendC::LocalTensor<uint64_t> &scaleL1Local, uint64_t curNL1,
                                           uint16_t scaleL1BufId)
    {
        AscendC::WaitFlag<AscendC::HardEvent::FIX_MTE2>(scaleL1BufId);
        AscendC::DataCopyPadParams padParams;
        AscendC::DataCopyParams scaleParam{1, static_cast<uint16_t>(curNL1 * sizeof(uint64_t)), 0, 0};
        AscendC::DataCopyPad(scaleL1Local, scaleGlobal, scaleParam, padParams);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_FIX>(scaleL1BufId);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_FIX>(scaleL1BufId);
    }

    __aicore__ inline void CopyInL0A(const AscendC::LocalTensor<AType> &l0aLocal,
                                     const AscendC::LocalTensor<AType> &al1Local, uint64_t curML1, uint64_t curKL1,
                                     uint64_t curKL0, uint64_t l0Iter)
    {
        AscendC::LoadData2DParamsV2 loadDataParams;
        uint64_t m1 = Cmct::Gemm::CeilDiv(curML1, AscendC::BLOCK_CUBE);
        if constexpr (!transA) {
            loadDataParams.mStartPosition = 0;
            loadDataParams.kStartPosition = Cmct::Gemm::CeilDiv(l0Iter * baseK_, C0_SIZE);
            loadDataParams.mStep = m1;
            loadDataParams.kStep = Cmct::Gemm::CeilDiv(curKL0, C0_SIZE);
            loadDataParams.srcStride = loadDataParams.mStep;
            loadDataParams.dstStride = loadDataParams.mStep;
            loadDataParams.ifTranspose = false;
        } else {
            loadDataParams.mStartPosition = Cmct::Gemm::CeilDiv(l0Iter * baseK_, AscendC::BLOCK_CUBE);
            loadDataParams.kStartPosition = 0;
            if ((m1 & 1) == 0) {
                loadDataParams.mStep = Cmct::Gemm::CeilAlign(
                    Cmct::Gemm::CeilDiv(curKL0, AscendC::BLOCK_CUBE), L0_TRANS_ALIGN);
            } else {
                loadDataParams.mStep = B8_MIN_STEP;
            }
            loadDataParams.kStep = Cmct::Gemm::CeilDiv(curML1, C0_SIZE);
            loadDataParams.srcStride = Cmct::Gemm::CeilDiv(curKL1, AscendC::BLOCK_CUBE);
            loadDataParams.dstStride = m1;
            loadDataParams.ifTranspose = true;
        }
        AscendC::LoadData(l0aLocal, al1Local, loadDataParams);
        if constexpr (transA) {
            if ((m1 & 1) != 0) {
                PipeBarrier<PIPE_MTE1>();
                uint64_t loadTimes =
                    Cmct::Gemm::CeilDiv(Cmct::Gemm::CeilDiv(curKL0, BLOCK_CUBE), B8_MIN_STEP);
                for (uint16_t i = 1; i < loadTimes; i++) {
                    loadDataParams.mStartPosition = B8_MIN_STEP * i + Cmct::Gemm::CeilDiv(l0Iter * baseK_, BLOCK_CUBE);
                    AscendC::LoadData(l0aLocal[i * m1 * BLOCK_CUBE * C0_SIZE], al1Local, loadDataParams);
                    PipeBarrier<PIPE_MTE1>();
                }
            }
        }
    }

    __aicore__ inline void CopyInL0B(const AscendC::LocalTensor<BType> &l0bLocal,
                                     const AscendC::LocalTensor<BType> &bl1Local, uint64_t curNL1, uint64_t curKL1,
                                     uint64_t curKL0, uint64_t l0Iter)
    {
        AscendC::LoadData2DParamsV2 loadDataParams;
        uint64_t n1 = Cmct::Gemm::CeilDiv(curNL1, AscendC::BLOCK_CUBE);
        if constexpr (transB) {
            loadDataParams.mStartPosition = 0;
            loadDataParams.kStartPosition = Cmct::Gemm::CeilDiv(l0Iter * baseK_, C0_SIZE);
            loadDataParams.mStep = n1;
            loadDataParams.kStep = Cmct::Gemm::CeilDiv(curKL0, C0_SIZE);
            loadDataParams.srcStride = loadDataParams.mStep;
            loadDataParams.dstStride = loadDataParams.mStep;
            loadDataParams.ifTranspose = false;
        } else {
            loadDataParams.mStartPosition = Cmct::Gemm::CeilDiv(l0Iter * baseK_, AscendC::BLOCK_CUBE);
            loadDataParams.kStartPosition = 0;
            if ((n1 & 1) == 0) {
                loadDataParams.mStep =
                    Cmct::Gemm::CeilAlign(Cmct::Gemm::CeilDiv(curKL0, AscendC::BLOCK_CUBE), L0_TRANS_ALIGN);
            } else {
                loadDataParams.mStep = B8_MIN_STEP;
            }
            loadDataParams.kStep = Cmct::Gemm::CeilDiv(curNL1, C0_SIZE);
            loadDataParams.srcStride = Cmct::Gemm::CeilDiv(curKL1, AscendC::BLOCK_CUBE);
            loadDataParams.dstStride = n1;
            loadDataParams.ifTranspose = true;
        }
        AscendC::LoadData(l0bLocal, bl1Local, loadDataParams);
        if constexpr (!transB) {
            if ((n1 & 1) != 0) {
                PipeBarrier<PIPE_MTE1>();
                uint64_t loadTimes =
                    Cmct::Gemm::CeilDiv(Cmct::Gemm::CeilDiv(curKL0, BLOCK_CUBE), B8_MIN_STEP);
                for (uint16_t i = 1; i < loadTimes; i++) {
                    loadDataParams.mStartPosition = B8_MIN_STEP * i + Cmct::Gemm::CeilDiv(l0Iter * baseK_, BLOCK_CUBE);
                    AscendC::LoadData(l0bLocal[i * n1 * AscendC::BLOCK_CUBE * C0_SIZE], bl1Local, loadDataParams);
                    PipeBarrier<PIPE_MTE1>();
                }
            }
        }
    }

    __aicore__ inline void CopyBiasInBT(const AscendC::LocalTensor<BiasType> &biasL1Local,
                                        const AscendC::LocalTensor<BiasType> &biasBt, uint64_t nl1Align, bool needBias)
    {
        if (!needBias) {
            return;
        }
        // s32场景要对齐到2 因此是align(nl1Align / 8, 2)
        uint64_t btAlign = AscendC::BLOCK_CUBE / BIAS_C0;
        uint16_t bustLenth = Cmct::Gemm::Align(nl1Align / BIAS_C0, btAlign);
        AscendC::DataCopyParams biasParam{1, static_cast<uint16_t>(bustLenth), 0, 0};
        // 当dstlocal位于C2时，C2中至少为fp32*16
        AscendC::DataCopy(biasBt, biasL1Local, biasParam);
    }

    __aicore__ inline void Mmad(AscendC::MmadParams &mmadParams, uint64_t l0cOffset, uint64_t l0abOffset,
                                uint64_t biasOffset, bool needBias)
    {
        mmadParams.cmatrixSource = needBias;
        if (needBias) {
            AscendC::Mmad(
                c1Local_[l0cOffset], l0aLocal_[l0abOffset], l0bLocal_[l0abOffset], biasBt_[biasOffset], mmadParams);
        } else {
            mmadParams.cmatrixSource = false;
            AscendC::Mmad(c1Local_[l0cOffset], l0aLocal_[l0abOffset], l0bLocal_[l0abOffset], mmadParams);
        }
    }

    __aicore__ inline void Iterate(MmadParams &mmadParams, uint64_t curML1, uint64_t curNL1, uint64_t curOuterKL1,
                                   uint64_t curInnerKL1, uint64_t l1Iter, uint16_t aL1BufId, uint16_t bL1BufId,
                                   uint16_t biasBufId, uint64_t l0cOffset, bool isL1LastRound)
    {
        uint64_t kL0Iter = Cmct::Gemm::CeilDiv(curInnerKL1, baseK_);
        uint64_t offsetAL1 = 0;
        if constexpr (DispatchPolicy::fullLoadMode == 0) {
            uint64_t kBL1Multiple = kAL1_ >= kBL1_ ? (l1Iter % (kAL1_ / kBL1_)) * kBL1_ *
                                                         (transA ? C0_SIZE : Cmct::Gemm::CeilAlign(curML1, BLOCK_CUBE))
                                                   : 0;
            offsetAL1 = l1BufferAOffset_[aL1BufId] + kBL1Multiple;
        } else {
            offsetAL1 =
                l1BufferAOffset_[0] + l1Iter * kL1_ * Cmct::Gemm::CeilAlign(curML1, transA ? C0_SIZE : BLOCK_CUBE);
        }
        uint64_t offsetBL1 = l1BufferBOffset_[bL1BufId];
        if constexpr (DispatchPolicy::fullLoadMode == 0) {
            uint64_t kAL1Multiple = kBL1_ >= kAL1_ ? (l1Iter % (kBL1_ / kAL1_)) * kAL1_ *
                                                         (transB ? Cmct::Gemm::CeilAlign(curNL1, BLOCK_CUBE) : C0_SIZE)
                                                   : 0;
            offsetBL1 += kAL1Multiple;
        }
        for (uint16_t iter1 = 0; iter1 < kL0Iter; ++iter1) {
            uint64_t curKL0 = iter1 == kL0Iter - 1 ? curInnerKL1 - iter1 * baseK_ : baseK_;
            // Load data to L0 and open DB
            uint64_t l0Offset = HALF_L0_SIZE * (l0PingPong_ & 0x1);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0PingPong_ & 0x1);
            CopyInL0A(l0aLocal_[l0Offset], aL1Local_[offsetAL1], curML1, kAL1_ >= kBL1_ ? curOuterKL1 : curInnerKL1,
                      curKL0, iter1);
            // copy bias to bt
            bool needBias = isBias_ && l1Iter == 0 && iter1 == 0;
            CopyBiasInBT(biasL1Local_[l1BufferBiasOffset_[biasBufId] / sizeof(BiasType)], biasBt_[baseN_ * biasBufId],
                         Cmct::Gemm::Align(mmadParams.n, AscendC::BLOCK_CUBE), needBias);
            CopyInL0B(l0bLocal_[l0Offset], bL1Local_[offsetBL1], curNL1, kBL1_ >= kAL1_ ? curOuterKL1 : curInnerKL1,
                      curKL0, iter1);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0PingPong_ & 0x1);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0PingPong_ & 0x1);
            mmadParams.k = curKL0;
            mmadParams.unitFlag = (isL1LastRound && iter1 + 1 == kL0Iter) ? FINAL_ACCUMULATION : NON_FINAL_ACCUMULATION;
            mmadParams.cmatrixInitVal = (l1Iter == 0 && iter1 == 0 && !isBias_);
            Mmad(mmadParams, l0cOffset, l0Offset, baseN_ * biasBufId, needBias);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0PingPong_ & 0x1);
            l0PingPong_++;
        }
    }

    __aicore__ inline void CopyFixpipeScalar(const AscendC::GlobalTensor<CType> &cGlobal,
                                             AscendC::LocalTensor<L0CType> &c1Local, MmadParams &mmadParams,
                                             uint64_t &scalarX2Scale)
    {
        AscendC::FixpipeParamsC310 intriParams;
        intriParams.nSize = mmadParams.n;
        intriParams.mSize = mmadParams.m;
        intriParams.dstStride = n_;
        intriParams.srcStride = Cmct::Gemm::Align(mmadParams.m, AscendC::BLOCK_CUBE);
        if constexpr (AscendC::IsSameType<L0CType, int32_t>::value) {
            if constexpr (AscendC::IsSameType<CType, int8_t>::value) {
                intriParams.quantPre = QuantMode_t::REQ8;
            } else if constexpr (AscendC::IsSameType<CType, half>::value) {
                intriParams.quantPre = QuantMode_t::DEQF16;
            } else if constexpr (AscendC::IsSameType<CType, bfloat16_t>::value) {
                intriParams.quantPre = QuantMode_t::QS322BF16_PRE;
            } else {
                intriParams.quantPre = QuantMode_t::NoQuant;
            }
        } else {
            if constexpr (AscendC::IsSameType<CType, float>::value) {
                intriParams.quantPre = QuantMode_t::QF322F32_PRE;
            } else if constexpr (AscendC::IsSameType<CType, half>::value) {
                intriParams.quantPre = QuantMode_t::QF322F16_PRE;
            } else if constexpr (AscendC::IsSameType<CType, bfloat16_t>::value) {
                intriParams.quantPre = QuantMode_t::QF322BF16_PRE;
            } else if constexpr (AscendC::IsSameType<CType, fp8_e4m3fn_t>::value) {
                intriParams.quantPre = QuantMode_t::QF322FP8_PRE;
            } else if constexpr (AscendC::IsSameType<CType, hifloat8_t>::value) {
                intriParams.quantPre = QuantMode_t::QF322HIF8_PRE;
            }
        }
        intriParams.deqScalar = scalarX2Scale;
        intriParams.unitFlag = FINAL_ACCUMULATION;
        intriParams.params = {1, 1, 1};
        AscendC::Fixpipe(cGlobal, c1Local, intriParams);
    }

    __aicore__ inline void CopyFixpipeTensor(const AscendC::GlobalTensor<CType> &cGlobal,
                                             AscendC::LocalTensor<L0CType> &c1Local, MmadParams &mmadParams,
                                             const AscendC::LocalTensor<uint64_t> &tensorX2Scale)
    {
        AscendC::FixpipeParamsC310 intriParams;
        intriParams.nSize = mmadParams.n;
        intriParams.mSize = mmadParams.m;
        intriParams.dstStride = n_;
        intriParams.srcStride = Cmct::Gemm::Align(mmadParams.m, AscendC::BLOCK_CUBE);
        if constexpr (AscendC::IsSameType<L0CType, int32_t>::value) {
            if constexpr (AscendC::IsSameType<CType, int8_t>::value) {
                intriParams.quantPre = QuantMode_t::VREQ8;
            } else if constexpr (AscendC::IsSameType<CType, half>::value) {
                intriParams.quantPre = QuantMode_t::VDEQF16;
            } else if constexpr (AscendC::IsSameType<CType, bfloat16_t>::value) {
                intriParams.quantPre = QuantMode_t::VQS322BF16_PRE;
            } else {
                intriParams.quantPre = QuantMode_t::NoQuant;
            }
        } else {
            if constexpr (AscendC::IsSameType<CType, float>::value) {
                intriParams.quantPre = QuantMode_t::VQF322F32_PRE;
            } else if constexpr (AscendC::IsSameType<CType, half>::value) {
                intriParams.quantPre = QuantMode_t::VQF322F16_PRE;
            } else if constexpr (AscendC::IsSameType<CType, bfloat16_t>::value) {
                intriParams.quantPre = QuantMode_t::VQF322BF16_PRE;
            } else if constexpr (AscendC::IsSameType<CType, fp8_e4m3fn_t>::value) {
                intriParams.quantPre = QuantMode_t::VQF322FP8_PRE;
            } else if constexpr (AscendC::IsSameType<CType, hifloat8_t>::value) {
                intriParams.quantPre = QuantMode_t::VQF322HIF8_PRE;
            }
        }
        intriParams.unitFlag = FINAL_ACCUMULATION;
        intriParams.params = {1, 1, 1};
        AscendC::Fixpipe(cGlobal, c1Local, tensorX2Scale, intriParams);
    }

    __aicore__ inline void IterateAL1BL1(AscendC::GlobalTensor<AType> aGlobal, AscendC::GlobalTensor<BType> bGlobal,
                                         AscendC::GlobalTensor<CType> cGlobal, MmadParams &mmadParams, uint64_t curML1,
                                         uint64_t curNL1, uint64_t l0cOffset, uint16_t biasBufId)
    {
        for (uint64_t kAIter = 0; kAIter < kAL1Iter_; ++kAIter) {
            uint64_t curKAL1 = kAIter == kAL1Iter_ - 1 ? k_ - kAIter * kAL1_ : kAL1_;
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(INPUT_BUFFER_FLAG_0 + aPingPongID_);
            CopyAInL1(aGlobal, curML1, curKAL1, kAIter, aPingPongID_);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(INPUT_BUFFER_FLAG_0 + aPingPongID_);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(INPUT_BUFFER_FLAG_0 + aPingPongID_);
            uint64_t kBL1Iter = Cmct::Gemm::CeilDiv(curKAL1, kBL1_);
            for (uint64_t kBIter = 0; kBIter < kBL1Iter; ++kBIter) {
                uint64_t curKBL1 =
                    kAIter == kAL1Iter_ - 1 && kBIter == kBL1Iter - 1 ? k_ - kAIter * kAL1_ - kBIter * kBL1_ : kBL1_;
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(INPUT_BUFFER_FLAG_2 + bPingPongID_);
                CopyBInL1(bGlobal, curNL1, curKBL1, abL1LoopCnt_, bPingPongID_);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(INPUT_BUFFER_FLAG_2 + bPingPongID_);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(INPUT_BUFFER_FLAG_2 + bPingPongID_);
                Iterate(mmadParams, curML1, curNL1, curKAL1, curKBL1, abL1LoopCnt_, aPingPongID_, bPingPongID_,
                        biasBufId, l0cOffset, kAIter == kAL1Iter_ - 1 && kBIter == kBL1Iter - 1);
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(INPUT_BUFFER_FLAG_2 + bPingPongID_);
                bPingPongID_ = bPingPongID_ ^ 1;
                abL1LoopCnt_++;
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(INPUT_BUFFER_FLAG_0 + aPingPongID_);
            aPingPongID_ = aPingPongID_ ^ 1;
        }
        abL1LoopCnt_ = 0;
    }

    __aicore__ inline void IterateBL1AL1(AscendC::GlobalTensor<AType> aGlobal, AscendC::GlobalTensor<BType> bGlobal,
                                         AscendC::GlobalTensor<CType> cGlobal, MmadParams &mmadParams, uint64_t curML1,
                                         uint64_t curNL1, uint64_t l0cOffset, uint16_t biasBufId)
    {
        for (uint64_t kBIter = 0; kBIter < kBL1Iter_; ++kBIter) {
            uint64_t curKBL1 = kBIter == kBL1Iter_ - 1 ? k_ - kBIter * kBL1_ : kBL1_;
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(INPUT_BUFFER_FLAG_0 + bPingPongID_);
            CopyBInL1(bGlobal, curNL1, curKBL1, kBIter, bPingPongID_);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(INPUT_BUFFER_FLAG_0 + bPingPongID_);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(INPUT_BUFFER_FLAG_0 + bPingPongID_);
            uint64_t kAL1Iter = Cmct::Gemm::CeilDiv(curKBL1, kAL1_);
            for (uint64_t kAIter = 0; kAIter < kAL1Iter; ++kAIter) {
                uint64_t curKAL1 =
                    kBIter == kBL1Iter_ - 1 && kAIter == kAL1Iter - 1 ? k_ - kBIter * kBL1_ - kAIter * kAL1_ : kAL1_;
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(INPUT_BUFFER_FLAG_2 + aPingPongID_);
                CopyAInL1(aGlobal, curML1, curKAL1, abL1LoopCnt_, aPingPongID_);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(INPUT_BUFFER_FLAG_2 + aPingPongID_);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(INPUT_BUFFER_FLAG_2 + aPingPongID_);
                Iterate(mmadParams, curML1, curNL1, curKBL1, curKAL1, abL1LoopCnt_, aPingPongID_, bPingPongID_,
                        biasBufId, l0cOffset, kBIter == kBL1Iter_ - 1 && kAIter == kAL1Iter - 1);
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(INPUT_BUFFER_FLAG_2 + aPingPongID_);
                aPingPongID_ = aPingPongID_ ^ 1;
                abL1LoopCnt_++;
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(INPUT_BUFFER_FLAG_0 + bPingPongID_);
            bPingPongID_ = bPingPongID_ ^ 1;
        }
        abL1LoopCnt_ = 0;
    }

    __aicore__ inline void IterateABL1(AscendC::GlobalTensor<AType> aGlobal, AscendC::GlobalTensor<BType> bGlobal,
                                       AscendC::GlobalTensor<CType> cGlobal, MmadParams &mmadParams, uint64_t curML1,
                                       uint64_t curNL1, uint64_t l0cOffset, uint16_t biasBufId)
    {
        for (uint64_t iter0 = 0; iter0 < kL1Iter_; ++iter0) {
            uint64_t curKL1 = iter0 == kL1Iter_ - 1 ? k_ - iter0 * kL1_ : kL1_;
            uint16_t l1BufId = abL1LoopCnt_ & (l1BufNum_ - 1);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BufId);
            CopyAInL1(aGlobal, curML1, curKL1, iter0, l1BufId);
            CopyBInL1(bGlobal, curNL1, curKL1, iter0, l1BufId);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BufId);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1BufId);
            Iterate(mmadParams, curML1, curNL1, curKL1, curKL1, iter0, l1BufId, l1BufId, biasBufId, l0cOffset,
                    iter0 == kL1Iter_ - 1);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BufId);
            abL1LoopCnt_++;
        }
    }

private:
    constexpr static uint64_t HALF_L0_SIZE = L0A_SIZE / DOUBLE_BUFFER_COUNT / sizeof(AType);
    constexpr static uint64_t HALF_L0C_SIZE = AscendC::TOTAL_L0C_SIZE / DOUBLE_BUFFER_COUNT / sizeof(L0CType);
    constexpr static int32_t C0_SIZE = AscendC::AuxGetC0Size<AType>();
    constexpr static int32_t BIAS_C0 = AscendC::AuxGetC0Size<BiasType>();
    constexpr static uint64_t BLOCK_CUBE = 16UL;

    uint64_t aL1OneBuffer_ = 0UL;
    uint64_t bL1OneBuffer_ = 0UL;
    uint64_t l1BufferAOffset_[4] = {0UL}; // default 4 buffer
    uint64_t l1BufferBOffset_[4] = {0UL}; // default 4 buffer
    uint64_t x2ScaleL1OneBuffer_ = 0UL;
    uint64_t biasL1OneBuffer_ = 0UL;
    uint64_t l1BufferX2ScaleOffset_[2] = {0UL}; // default 2 buffer
    uint64_t l1BufferBiasOffset_[2] = {0UL}; // default 2 buffer
    uint64_t scalarScale_ = 0UL;
    AscendC::LocalTensor<AType> l0aLocal_{AscendC::TPosition::A2, 0, L0A_SIZE};
    AscendC::LocalTensor<BType> l0bLocal_{AscendC::TPosition::B2, 0, L0B_SIZE};
    AscendC::LocalTensor<L0CType> c1Local_{AscendC::TPosition::CO1, 0, AscendC::TOTAL_L0C_SIZE};
    AscendC::LocalTensor<BiasType> biasBt_{AscendC::TPosition::C2, 0, BT_SIZE};
    AscendC::LocalTensor<AType> aL1Local_{AscendC::TPosition::A1, 0, AscendC::TOTAL_L1_SIZE};
    AscendC::LocalTensor<BType> bL1Local_{AscendC::TPosition::A1, 0, AscendC::TOTAL_L1_SIZE};
    AscendC::LocalTensor<BiasType> biasL1Local_{AscendC::TPosition::A1, 0, AscendC::TOTAL_L1_SIZE / sizeof(BiasType)};
    AscendC::LocalTensor<uint64_t> scaleL1Local_{AscendC::TPosition::A1, 0, AscendC::TOTAL_L1_SIZE / sizeof(uint64_t)};
};
}
}
}
#endif