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
 * \file block_prologue_b_cast_scsc.h
 * \brief
 */
#ifndef CMCT_INCLUDE_PROLOGUE_BLOCK_PROLOGUE_B_CAST_SCSC_H
#define CMCT_INCLUDE_PROLOGUE_BLOCK_PROLOGUE_B_CAST_SCSC_H

#include "../utils/common_utils.h"
#include "../utils/integral_constant.h"
#include "../utils/tuple_utils.h"
#include "block_prologue.h"
#include "dispatch_policy.h"

namespace Cmct::Prologue {
using AscendC::BLOCK_CUBE;
using AscendC::CrossCoreSetFlag;
using AscendC::CrossCoreWaitFlag;
using AscendC::GetSubBlockIdx;
using AscendC::HardEvent;
using AscendC::IsSameType;
using AscendC::MakeCoord;
using AscendC::MakeShape;
using AscendC::ONE_BLK_SIZE;
using AscendC::ONE_BLOCK_SIZE;
using AscendC::SetFlag;
using AscendC::SupportType;
using AscendC::VECTOR_REG_WIDTH;
using AscendC::WaitFlag;
using Cmct::CeilAlign;
using Cmct::CeilDiv;
using Cmct::Gemm::_;
using Cmct::Gemm::_16;
using Cmct::Gemm::_32;
using Cmct::Gemm::Get;
using Cmct::Gemm::GetTile;
using Cmct::Gemm::Min;
namespace MicroAPI = AscendC::MicroAPI;

template <class InType, class OutType, class BiasType, class TileShapeL1>
class BlockPrologue<BCastScsc, InType, OutType, BiasType, TileShapeL1> {
public:
    using ElementIn = typename InType::Element;
    using LayoutIn = typename InType::Layout;
    using ElementOut = typename OutType::Element;
    using LayoutOut = typename OutType::Layout;
    using ElementBias = typename BiasType::Element;
    using LayoutBias = typename BiasType::Layout;

    struct Arguments {
        GM_ADDR ptrB;
        GM_ADDR ptrBias;
        LayoutIn layoutB;
        LayoutBias layoutBias;
    };

    struct Params {
        GM_ADDR ptrB;
        GM_ADDR ptrBias;
        TileShapeL1 tileShapeL1;
        LayoutIn layoutB;
        LayoutBias layoutBias;
        int64_t l1BufNum;
        int32_t nUbSize;
        int32_t kUbSize;
        bool hasBias;
    };

    template <class TensorB, class TensorBias, class ActualBlockShape>
    __aicore__ inline void operator()(
        const TensorB& bGlobal, const TensorBias& biasGlobal, const ActualBlockShape& actualBlockShape,
        const Params& params)
    {
        nL1Len_ = Get<1>(actualBlockShape);
        ComputeBiasUbParams();
        uint64_t kTileCount = CeilDiv(kSize_, static_cast<uint64_t>(Get<2>(params.tileShapeL1)));
        for (uint64_t kLoopIdx = 0; kLoopIdx < kTileCount; kLoopIdx++) {
            calcBias_ = hasBias_ && kLoopIdx == 0 && nBiasUbLen_ > 0;
            kGmOffset_ = kLoopIdx * kL1Size_;
            kL1Len_ = Min(kSize_ - kGmOffset_, kL1Size_);
            TensorB tensorBlockB;
            if constexpr (weightNz) {
                auto tileShape = MakeShape(
                    MakeShape(_16{}, static_cast<uint64_t>(CeilDiv(static_cast<uint64_t>(nL1Len_), 16UL))),
                    MakeShape(_32{}, static_cast<uint64_t>(CeilDiv(static_cast<uint64_t>(kL1Len_), 32UL))));
                tensorBlockB =
                    GetTile(bGlobal, MakeCoord(MakeCoord(_, _), MakeCoord(_, CeilDiv(kGmOffset_, 32UL))), tileShape);
            } else {
                tensorBlockB = GetTile(
                    bGlobal, MakeCoord(0, kGmOffset_),
                    MakeShape(static_cast<uint64_t>(nL1Len_), static_cast<uint64_t>(kL1Len_)));
            }
            nUbLen_ = nL1Len_;
            kUbLen_ = kL1Len_;
            if constexpr (weightNz) { // weightNz 只有4buffer 1buffer，1buffer直接使用AIV0
                if (likely(l1BufNum_ == QUADRUPLE_BUFFER)) {
                    ComputeUbParamsByL1Size();
                    VectorProcess(tensorBlockB, biasGlobal);
                } else if (GetSubBlockIdx() == 0) {
                    VectorProcess(tensorBlockB, biasGlobal);
                }
            } else { // ND格式有4 buffer 2 buffer 1 buffer场景
                if (likely(l1BufNum_ == QUADRUPLE_BUFFER)) {
                    ComputeUbParamsByL1Size();
                    VectorProcess(tensorBlockB, biasGlobal);
                } else if (l1BufNum_ == DOUBLE_BUFFER) {
                    if (l1BufIdx_ == GetSubBlockIdx()) {
                        VectorProcess(tensorBlockB, biasGlobal);
                    }
                } else if (GetSubBlockIdx() == 0) {
                    VectorProcess(tensorBlockB, biasGlobal);
                }
            }
            l1BufIdx_ = (l1BufIdx_ + 1) % l1BufNum_;
        }
    }

    __aicore__ inline BlockPrologue(const Params& params)
    {
        hasBias_ = params.hasBias;
        l1BufNum_ = params.l1BufNum;
        nUbSize_ = params.nUbSize;
        kUbSize_ = params.kUbSize;
        if constexpr (weightNz) {
            nSize_ = Get<0>(Get<0>(params.layoutB.GetShape())) * Get<1>(Get<0>(params.layoutB.GetShape()));
            kSize_ = Get<0>(Get<1>(params.layoutB.GetShape())) * Get<1>(Get<1>(params.layoutB.GetShape()));
        } else {
            nSize_ = Get<0>(params.layoutB.GetShape());
            kSize_ = Get<1>(params.layoutB.GetShape());
        }
        nL1Size_ = Get<1>(params.tileShapeL1);
        kL1Size_ = Get<3>(params.tileShapeL1); // 3 in order to obtain k
        bL1Size_ = nL1Size_ * kL1Size_;
        aL1Size_ = Get<0>(params.tileShapeL1) * Get<2>(params.tileShapeL1); // 2 in order to obtain k
        if (likely(l1BufNum_ == QUADRUPLE_BUFFER)) {
            if constexpr (weightNz) {
                vecWeightInLen_ = (l1BufNum_ * nUbSize_ * kUbSize_) >> INT4_DTYPE_PARAM;
                vecWeightOutLen_ = l1BufNum_ * nUbSize_ * kUbSize_;
            } else {
                vecWeightInLen_ = (l1BufNum_ * (nUbSize_ * CeilAlign(kUbSize_, OFFSET_64))) >> INT4_DTYPE_PARAM;
                vecWeightOutLen_ = l1BufNum_ * (CeilAlign(nUbSize_, BLOCK_CUBE) + 1) *
                                   CeilAlign(kUbSize_, static_cast<int32_t>(ONE_BLK_SIZE));
            }
        } else {
            if constexpr (weightNz) {
                vecWeightInLen_ = (nUbSize_ * kUbSize_) >> INT4_DTYPE_PARAM;
                vecWeightOutLen_ = nUbSize_ * kUbSize_;
            } else {
                vecBufNum_ = Min(l1BufNum_, DOUBLE_BUFFER);
                vecWeightInLen_ = (vecBufNum_ * (nUbSize_ * CeilAlign(kUbSize_, OFFSET_64))) >> INT4_DTYPE_PARAM;
                vecWeightOutLen_ = vecBufNum_ * (CeilAlign(nUbSize_, BLOCK_CUBE) + 1) *
                                   CeilAlign(kUbSize_, static_cast<int32_t>(ONE_BLK_SIZE));
            }
        }
        weightOutUb_ = AscendC::LocalTensor<ElementOut>(AscendC::TPosition::VECCALC, 0, vecWeightOutLen_);
        uint64_t ubOffset = vecWeightOutLen_ * sizeof(ElementOut);
        weightInUb_ = AscendC::LocalTensor<ElementIn>(AscendC::TPosition::VECCALC, ubOffset, vecWeightInLen_);
        if (hasBias_) {
            vecBiasLen_ = l1BufNum_ == QUADRUPLE_BUFFER ?
                              l1BufNum_ * BIAS_SPLIT_N_L1_SIZE :
                              l1BufNum_ * CeilAlign(nL1Size_, static_cast<uint64_t>(VEC_MAX_ELEM_B16));
            ubOffset += vecWeightInLen_ * sizeof(ElementIn);
            biasOutUb_ = AscendC::LocalTensor<ElementBias>(AscendC::TPosition::VECCALC, ubOffset, vecBiasLen_);
            ubOffset += vecBiasLen_ * sizeof(ElementBias);
            biasInUb_ = AscendC::LocalTensor<ElementBias>(AscendC::TPosition::VECCALC, ubOffset, vecBiasLen_);
        }
        l1Local_ = AscendC::LocalTensor<ElementOut>(AscendC::TPosition::B1, 0, L1_BUFFER_SIZE);
    }

    __aicore__ inline ~BlockPrologue()
    {
        if (l1BufNum_ == QUADRUPLE_BUFFER) {
            int64_t buffNum = Min(idx_ + 1, static_cast<int64_t>(l1BufNum_));
            for (int64_t index = 0; index < buffNum; index++) {
                WaitFlag<HardEvent::V_MTE2>(index);
                WaitFlag<HardEvent::MTE3_V>(index);
            }
        } else {
            if (idx_ > 0) {
                for (uint8_t index = 0; index < vecBufNum_; index++) {
                    WaitFlag<HardEvent::V_MTE2>(index);
                    WaitFlag<HardEvent::MTE3_V>(index);
                }
            } else if (idx_ == 0) {
                WaitFlag<HardEvent::V_MTE2>(0);
                WaitFlag<HardEvent::MTE3_V>(0);
            }
        }
        if (l1BufNum_ == QUADRUPLE_BUFFER) {
#pragma unroll
            for (int8_t index = 0; index < QUADRUPLE_BUFFER; index++) {
                WaitForCube();
            }
            return;
        }
        if (l1BufNum_ == vecBufNum_) {
            if (GetSubBlockIdx() == 0) {
                WaitForCube();
            }
            return;
        }
        if (l1BufNum_ == DOUBLE_BUFFER) {
            for (int8_t i = 0; i < DOUBLE_BUFFER; i++) {
                WaitForCube();
            }
            return;
        }
    }

private:
    struct VfParamsNormal {
        uint16_t outExtend;
        uint16_t innerExtend;
        uint32_t dataBlockStride;
        uint32_t repeatStride;
        int32_t outDimOffset;
        uint32_t maskB8Tail0;
        uint32_t maskB8Tail1;
        __ubuf__ int8_t* weightInUbBaseAddr;
        __ubuf__ ElementOut* weightOutUbAddr;
        __ubuf__ ElementOut* weightOutUbAddr1;
    };

    struct VfParamsNz {
        uint16_t innerExtend;
        uint32_t innerDstExtend;
        uint32_t innerSrcExtend;
        uint32_t shiftLeftSize;
        uint32_t andMask;
        __ubuf__ int8_t* weightInUbBaseAddr;
        __ubuf__ ElementOut* weightOutUbAddr;
    };

    struct VfParamsBias {
        uint16_t biasLoopNum;
        __ubuf__ ElementBias* biasInUbBaseAddr;
        __ubuf__ ElementBias* biasOutUbAddr;
    };

    __aicore__ inline void WaitForCube()
    {
        CrossCoreWaitFlag<SYNC_MODE4, PIPE_MTE3>(SYNC_AIV_AIC_FLAG);
    }

    __aicore__ inline void NotifyCube()
    {
        CrossCoreSetFlag<SYNC_MODE4, PIPE_MTE3>(1);
    }

    template <class TensorB, class TensorBias>
    __aicore__ inline void VectorProcess(const TensorB& tensorBlockB, const TensorBias& tensorBias)
    {
        WaitForCube();
        if (likely(l1BufNum_ == QUADRUPLE_BUFFER)) {
            ProcessL1NK4Buffer(tensorBlockB, tensorBias);
        } else {
            ProcessL1NK(tensorBlockB, tensorBias);
        }
        NotifyCube();
    }

    __aicore__ inline void ComputeUbParamsByL1Size()
    {
        if constexpr (weightNz) {
            if (kL1Len_ > kUbSize_) {
                kUbLen_ = kUbSize_;
                if (GetSubBlockIdx() == 1) {
                    kL1Offset_ = kUbLen_;
                    kUbLen_ = kL1Len_ - kUbLen_;
                }
            } else {
                kL1Offset_ = 0;
            }
        } else {
            if (nL1Len_ > nUbSize_) {
                nUbLen_ = nUbSize_;
                if (GetSubBlockIdx() == 1) {
                    nL1Offset_ = nUbLen_;
                    nUbLen_ = nL1Len_ - nUbLen_;
                }
            } else {
                nL1Offset_ = 0;
            }
        }
    }

    __aicore__ inline void ComputeBiasUbParams()
    {
        if (!hasBias_) {
            return;
        }

        nBiasUbLen_ = nL1Len_;
        if (likely(l1BufNum_ == QUADRUPLE_BUFFER)) {
            nBiasUbLen_ = nL1Len_ < BIAS_SPLIT_N_L1_SIZE ? nL1Len_ : BIAS_SPLIT_N_L1_SIZE;
            if (GetSubBlockIdx() == 1) {
                nL1BiasOffset_ = nBiasUbLen_;
                nBiasUbLen_ = nL1Len_ - nBiasUbLen_;
            }
        }
    }

    template <class TensorB, class TensorBias>
    __aicore__ inline void ProcessL1NK4Buffer(const TensorB& bGlobal, const TensorBias& biasGlobal)
    {
        int64_t l1Offset =
            (l1BufIdx_ & 0x1) *
                Cmct::Gemm::Max(L1_BUFFER_HALF_SIZE / sizeof(ElementOut), DOUBLE_BUFFER * bL1Size_ + aL1Size_) +
            ((l1BufIdx_ & 0x2) > 1) * bL1Size_;
        l1Offset += nL1Offset_ * ONE_BLK_SIZE + kL1Offset_ * CeilAlign(nL1Len_, BLOCK_CUBE);
        ProcessL1(bGlobal, biasGlobal, l1Offset);
    }

    template <class TensorB, class TensorBias>
    __aicore__ inline void ProcessL1NK(const TensorB& bGlobal, const TensorBias& biasGlobal)
    {
        int32_t bUbNFactor = CeilDiv(nL1Len_, nUbSize_);
        int32_t bUbKFactor = CeilDiv(kL1Len_, kUbSize_);
        for (int32_t bUbNLoopIdx = 0; bUbNLoopIdx < bUbNFactor; bUbNLoopIdx++) {
            nL1Offset_ = bUbNLoopIdx * nUbSize_;
            nUbLen_ = Min(nL1Len_ - static_cast<int32_t>(nL1Offset_), nUbSize_);
            for (int32_t bUbKLoopIdx = 0; bUbKLoopIdx < bUbKFactor; bUbKLoopIdx++) {
                kL1Offset_ = bUbKLoopIdx * kUbSize_;
                kUbLen_ = Min(kL1Len_ - static_cast<int32_t>(kL1Offset_), kUbSize_);
                int64_t l1Offset = (l1BufIdx_ & 0x1) * L1_BUFFER_HALF_SIZE / sizeof(ElementOut) +
                                   CeilAlign(nL1Len_, BLOCK_CUBE) * kL1Offset_ + nL1Offset_ * ONE_BLK_SIZE;
                calcBias_ = calcBias_ && bUbNLoopIdx == 0 && bUbKLoopIdx == 0; // 不管weight算几轮，bias只在第一轮计算
                ProcessL1(bGlobal, biasGlobal, l1Offset);
            }
        }
    }

    template <class TensorB, class TensorBias>
    __aicore__ inline void ProcessL1(const TensorB& bGlobal, const TensorBias& biasGlobal, int64_t l1Offset)
    {
        idx_ += 1;
        ubBufIdx_ = idx_ % l1BufNum_;
        if (idx_ >= l1BufNum_) {
            WaitFlag<HardEvent::V_MTE2>(ubBufIdx_);
        }
        CopyInTensorBias(biasGlobal);
        CopyInTensorWeight(bGlobal);
        SetFlag<HardEvent::MTE2_V>(ubBufIdx_);
        if (idx_ >= l1BufNum_) {
            WaitFlag<HardEvent::MTE3_V>(ubBufIdx_);
        }
        WaitFlag<HardEvent::MTE2_V>(ubBufIdx_);
        AntiQuantCompute();
        SetFlag<HardEvent::V_MTE3>(ubBufIdx_);
        SetFlag<HardEvent::V_MTE2>(ubBufIdx_);
        WaitFlag<HardEvent::V_MTE3>(ubBufIdx_);
        if constexpr (weightNz) {
            CopyVecOut2L1(l1Offset, weightOutUb_[ubBufIdx_ * VEC_MAX_ELEM_B8]);
        } else {
            uint64_t weightOutUbOffset = ubBufIdx_ * (vecWeightOutLen_ / sizeof(ElementOut) / l1BufNum_);
            CopyVecOut2L1(l1Offset, weightOutUb_[weightOutUbOffset]);
        }
        CopyBiasVecOut2L1();
        SetFlag<HardEvent::MTE3_V>(ubBufIdx_);
    }

    template <class TensorB>
    __aicore__ inline void CopyInTensorWeight(const TensorB& bGlobal)
    {
        AscendC::DataCopyExtParams intriParams;
        intriParams.dstStride = 0;
        AscendC::DataCopyPadExtParams<ElementIn> padParams;
        if constexpr (weightNz) {
            intriParams.blockCount = kUbLen_ / C0_SIZE_B8;
            intriParams.blockLen = nUbLen_ * BLOCK_CUBE;
            int64_t nAlignSize = CeilAlign(nSize_, static_cast<uint64_t>(BLOCK_CUBE));
            intriParams.srcStride = (nAlignSize - nUbLen_) * BLOCK_CUBE;
        } else {
            intriParams.blockCount = nUbLen_;
            intriParams.blockLen = kUbLen_ >> INT4_DTYPE_PARAM;
            intriParams.srcStride = (kSize_ - kUbLen_) >> INT4_DTYPE_PARAM;
        }
        uint64_t weightInOffset = ubBufIdx_ * (vecWeightInLen_ << INT4_DTYPE_PARAM) / l1BufNum_;
        AscendC::GlobalTensor<ElementIn> srcTensor;
        srcTensor.SetGlobalBuffer(bGlobal.address_);
        if constexpr (weightNz) {
            DataCopyPad(
                weightInUb_[weightInOffset], srcTensor[kL1Offset_ * nSize_ + nL1Offset_ * C0_SIZE_B8], intriParams,
                padParams);
        } else {
            DataCopyPad(
                weightInUb_[weightInOffset], srcTensor[nL1Offset_ * kSize_ + kL1Offset_], intriParams, padParams);
        }
    }

    template <class TensorBias>
    __aicore__ inline void CopyInTensorBias(const TensorBias& tensorBias)
    {
        if (!calcBias_) {
            return;
        }
        AscendC::DataCopyParams dataCopyParams = {1, static_cast<uint16_t>(nBiasUbLen_ * sizeof(ElementBias)), 0, 0};
        AscendC::DataCopyPadParams dataCopyPadParams;
        AscendC::GlobalTensor<ElementBias> srcTensor;
        srcTensor.SetGlobalBuffer(tensorBias.address_);
        uint64_t biasInOffset = ubBufIdx_ * vecBiasLen_ / l1BufNum_;
        AscendC::DataCopyPad(biasInUb_[biasInOffset], srcTensor[nL1BiasOffset_], dataCopyParams, dataCopyPadParams);
    }

    __aicore__ inline void CopyVecOut2L1(int64_t l1Offset, const AscendC::LocalTensor<ElementOut>& ubLocal)
    {
        AscendC::DataCopyParams params;
        if constexpr (weightNz) {
            params.blockLen = BLOCK_NUM_REG;
            params.blockCount = nUbLen_ * kUbLen_ * sizeof(ElementOut) / VECTOR_REG_WIDTH;
            params.srcStride = (l1BufNum_ - 1) * BLOCK_NUM_REG;
            params.dstStride = 0;
            DataCopy(l1Local_[l1Offset], ubLocal, params);
        } else {
            params.blockLen = nUbLen_;
            params.blockCount = CeilDiv(kUbLen_, static_cast<int32_t>(GROUP_SIZE));
            params.srcStride = 1 + CeilAlign(nUbLen_, BLOCK_CUBE) - nUbLen_;
            params.dstStride = CeilAlign(nL1Len_, BLOCK_CUBE) - nUbLen_;
            DataCopy(l1Local_[l1Offset], ubLocal, params);
        }
    }

    __aicore__ inline void CopyBiasVecOut2L1()
    {
        if (!calcBias_) {
            return;
        }

        // l1Local_原始数据类型为B8, biasL1Offset以B计，跳过前面weight所占的L1空间
        uint64_t biasL1Offset =
            (l1BufIdx_ & 0x1) * L1_BUFFER_HALF_SIZE + CeilDiv(l1BufNum_, DOUBLE_BUFFER) * bL1Size_ +
            ((l1BufIdx_ & 0x2) > 1) * CeilAlign(nL1Size_, static_cast<uint64_t>(BLOCK_CUBE)) * sizeof(ElementBias);
        if (likely(l1BufNum_ == QUADRUPLE_BUFFER) && GetSubBlockIdx() == 1) {
            biasL1Offset += BIAS_SPLIT_N_L1_SIZE * sizeof(ElementBias);
        }
        AscendC::DataCopyParams params;
        params.blockLen = CeilAlign(nBiasUbLen_, BLOCK_CUBE) * sizeof(ElementBias) / ONE_BLK_SIZE;
        params.blockCount = 1;
        params.srcStride = 0;
        params.dstStride = 0;
        DataCopy(
            l1Local_[biasL1Offset].template ReinterpretCast<ElementBias>(),
            biasOutUb_[ubBufIdx_ * vecBiasLen_ / l1BufNum_], params);
    }

    __aicore__ inline void AntiQuantComputeNormal(const VfParamsBias& biasParams)
    {
        VfParamsNormal wParams;
        wParams.outExtend = static_cast<uint16_t>(nUbLen_);
        wParams.innerExtend = CeilDiv(CeilAlign(kUbLen_, UB_ALIGN_SIZE_FOR_4BITS), VECTOR_REG_WIDTH_FOR_4BITS);
        wParams.dataBlockStride = CeilAlign(nUbLen_, BLOCK_CUBE) + 1;
        wParams.repeatStride = wParams.dataBlockStride * BLOCK_CUBE;
        wParams.outDimOffset = ONE_BLOCK_SIZE - wParams.innerExtend * wParams.repeatStride * ONE_BLOCK_SIZE;
        wParams.maskB8Tail0 = Min(kUbLen_ % VECTOR_REG_WIDTH_FOR_4BITS, static_cast<int32_t>(VECTOR_REG_WIDTH)) +
                              kUbLen_ / VECTOR_REG_WIDTH_FOR_4BITS * VECTOR_REG_WIDTH;
        wParams.maskB8Tail1 =
            Cmct::Gemm::Max(kUbLen_ % VECTOR_REG_WIDTH_FOR_4BITS - static_cast<int32_t>(VECTOR_REG_WIDTH), 0) +
            kUbLen_ / VECTOR_REG_WIDTH_FOR_4BITS * VECTOR_REG_WIDTH;
        wParams.weightInUbBaseAddr = weightInUbBaseAddr_;
        wParams.weightOutUbAddr = weightOutUbAddr_;
        wParams.weightOutUbAddr1 = weightOutUbAddr1_;
        if (calcBias_) {
            RegCompute<true>(wParams, biasParams);
        } else {
            RegCompute<false>(wParams, biasParams);
        }
    }

    __simd_callee__ inline void RegBiasCompute(const VfParamsBias biasParams)
    {
        MicroAPI::RegTensor<ElementBias> biasVreg, factorVreg;
        MicroAPI::MaskReg maskAll = MicroAPI::CreateMask<ElementBias, MicroAPI::MaskPattern::ALL>();
        MicroAPI::Duplicate<ElementBias, MicroAPI::MaskMergeMode::ZEROING>(factorVreg, BIAS_REDUCE_FACTOR, maskAll);

        for (uint16_t nLoopIdx = 0; nLoopIdx < biasParams.biasLoopNum; ++nLoopIdx) {
            MicroAPI::AddrReg biasAreg = MicroAPI::CreateAddrReg<ElementBias>(nLoopIdx, VEC_MAX_ELEM_B16);
            MicroAPI::LoadAlign<ElementBias, MicroAPI::LoadDist::DIST_NORM>(
                biasVreg, biasParams.biasInUbBaseAddr, biasAreg);
            MicroAPI::Mul(biasVreg, biasVreg, factorVreg, maskAll);
            MicroAPI::StoreAlign<ElementBias, MicroAPI::StoreDist::DIST_NORM_B16>(
                biasParams.biasOutUbAddr, biasVreg, biasAreg, maskAll);
        }
    }

    template <bool calcBias>
    __simd_vf__ inline void RegCompute(const VfParamsNormal wParams, const VfParamsBias biasParams)
    {
        if constexpr (calcBias) {
            RegBiasCompute(biasParams);
        }
        __ubuf__ ElementOut* weightOutUbAddr = wParams.weightOutUbAddr;
        __ubuf__ ElementOut* weightOutUbAddr1 = wParams.weightOutUbAddr1;
        MicroAPI::RegTensor<uint8_t> wDIntlv0, wDIntlv1, wLoad0, sAnd0, sAnd1, wShr, wShl, s1, wOr0, wOr1, wdup1, wdup4;
        MicroAPI::RegTensor<int8_t> wdup0, wdup2, wdup3;
        MicroAPI::MaskReg preg = MicroAPI::CreateMask<uint8_t, MicroAPI::MaskPattern::ALL>();
        MicroAPI::Duplicate<int8_t, MicroAPI::MaskMergeMode::ZEROING>(wdup0, DUP_CONFIG_2, preg);
        MicroAPI::Duplicate<uint8_t, MicroAPI::MaskMergeMode::ZEROING>(wdup1, DUP_CONFIG_MODE_1C, preg);
        MicroAPI::Duplicate<int8_t, MicroAPI::MaskMergeMode::ZEROING>(wdup2, DUP_CONFIG_2, preg);
        MicroAPI::Duplicate<int8_t, MicroAPI::MaskMergeMode::ZEROING>(wdup3, DUP_CONFIG_4, preg);
        MicroAPI::Duplicate<uint8_t, MicroAPI::MaskMergeMode::ZEROING>(wdup4, DUP_FLAG_80, preg);
        // 一次处理一个N轴
        for (uint16_t outIdx = 0; outIdx < wParams.outExtend; ++outIdx) {
            uint32_t maskWeight0Tmp = wParams.maskB8Tail0;
            uint32_t maskWeight1Tmp = wParams.maskB8Tail1;
            for (uint16_t repeatIdx = 0; repeatIdx < wParams.innerExtend; ++repeatIdx) {
                MicroAPI::MaskReg MaskRegB8Tail0 = MicroAPI::UpdateMask<uint8_t>(maskWeight0Tmp);
                MicroAPI::MaskReg MaskRegB8Tail1 = MicroAPI::UpdateMask<uint8_t>(maskWeight1Tmp);
                MicroAPI::AddrReg aregWeightB8 =
                    MicroAPI::CreateAddrReg<uint8_t>(outIdx, kUbLen_ >> 1, repeatIdx, VEC_MAX_ELEM_B8);
                MicroAPI::LoadAlign(wLoad0, (__ubuf__ uint8_t*&)wParams.weightInUbBaseAddr, aregWeightB8);
                // 提取E/M
                MicroAPI::ShiftRight(wShr, wLoad0, wdup0, preg); // vr1
                MicroAPI::And(wShr, wShr, wdup1, preg);          // vr1
                MicroAPI::ShiftLeft(wShl, wLoad0, wdup2, preg);  // vr2
                MicroAPI::And(wShl, wShl, wdup1, preg);          // vr2
                // 提取S
                MicroAPI::ShiftLeft(s1, wLoad0, wdup3, preg); // vr3
                MicroAPI::And(sAnd0, s1, wdup4, preg);        // vr3
                MicroAPI::And(sAnd1, wLoad0, wdup4, preg);    // vr4
                // 合并S/E/M
                MicroAPI::Or(wOr0, wShr, sAnd1, preg); // odd
                MicroAPI::Or(wOr1, wShl, sAnd0, preg); // even
                MicroAPI::Interleave(wDIntlv0, wDIntlv1, wOr1, wOr0);
                MicroAPI::StoreAlign<
                    uint8_t, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    (__ubuf__ uint8_t*&)weightOutUbAddr, wDIntlv0, wParams.dataBlockStride, wParams.repeatStride,
                    MaskRegB8Tail0);
                MicroAPI::StoreAlign<
                    uint8_t, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    (__ubuf__ uint8_t*&)weightOutUbAddr1, wDIntlv1, wParams.dataBlockStride, wParams.repeatStride,
                    MaskRegB8Tail1);
            }
            weightOutUbAddr += wParams.outDimOffset;
            weightOutUbAddr1 += wParams.outDimOffset;
        }
    }

    __aicore__ inline void AntiQuantCompute()
    {
        uint64_t weightOutUbOffset;
        uint64_t weightInUbOffset;
        if constexpr (weightNz) {
            weightOutUbOffset = ubBufIdx_ * VEC_MAX_ELEM_B8;
        } else {
            weightOutUbOffset = ubBufIdx_ * (vecWeightOutLen_ / sizeof(ElementOut) / l1BufNum_);
        }
        weightInUbOffset = ubBufIdx_ * (vecWeightInLen_ << INT4_DTYPE_PARAM) / l1BufNum_;
        weightInUbBaseAddr_ = (__ubuf__ int8_t*)weightInUb_[weightInUbOffset].GetPhyAddr();
        weightOutUbAddr_ = (__ubuf__ ElementOut*)weightOutUb_[weightOutUbOffset].GetPhyAddr();

        VfParamsBias biasParams;
        if (calcBias_) {
            uint64_t biasUbOffset = ubBufIdx_ * vecBiasLen_ / l1BufNum_;
            biasParams.biasLoopNum = CeilDiv(nBiasUbLen_, VEC_MAX_ELEM_B16);
            biasParams.biasInUbBaseAddr = (__ubuf__ ElementBias*)biasInUb_[biasUbOffset].GetPhyAddr();
            biasParams.biasOutUbAddr = (__ubuf__ ElementBias*)biasOutUb_[biasUbOffset].GetPhyAddr();
        }

        if constexpr (!weightNz) {
            uint16_t blockStride = CeilAlign(nUbLen_, BLOCK_CUBE) + 1;
            weightOutUbAddr1_ = weightOutUbAddr_ + VEC_MAX_ELEM_B8 * blockStride;
            AntiQuantComputeNormal(biasParams);
        } else {
            AntiQuantComputeNKMxNz(biasParams);
        }
    }

    __aicore__ inline void AntiQuantComputeNKMxNz(const VfParamsBias& biasParams)
    {
        static_assert(
            SupportType<ElementIn, fp4x2_e2m1_t, fp4x2_e1m2_t>(), "only support fp4x2_e2m1_t and fp4x2_e1m2_t");
        VfParamsNz wParams;
        wParams.shiftLeftSize =
            IsSameType<ElementIn, fp4x2_e2m1_t>::value ? E2M1_SHIFT_LEFT_SIZE : E1M2_SHIFT_LEFT_SIZE;
        wParams.andMask = IsSameType<ElementIn, fp4x2_e2m1_t>::value ? E2M1_AND_MASK : E1M2_AND_MASK;
        wParams.innerExtend = CeilDiv(kUbLen_ * nUbLen_, static_cast<int32_t>(VECTOR_REG_WIDTH));
        wParams.innerDstExtend = VECTOR_REG_WIDTH * l1BufNum_;
        wParams.innerSrcExtend = VECTOR_REG_WIDTH >> 1;
        wParams.weightInUbBaseAddr = weightInUbBaseAddr_;
        wParams.weightOutUbAddr = weightOutUbAddr_;
        if (calcBias_) {
            RegComputeNkNz<true>(wParams, biasParams);
        } else {
            RegComputeNkNz<false>(wParams, biasParams);
        }
    }

    template <bool calcBias>
    __simd_vf__ inline void RegComputeNkNz(const VfParamsNz wParams, const VfParamsBias biasParams)
    {
        if constexpr (calcBias) {
            RegBiasCompute(biasParams);
        }
        MicroAPI::RegTensor<int8_t> wdup0, wdup1, wdup2, wLoad0, wShl, wShr0, wShr1, wSel0, sAnd0;
        MicroAPI::MaskReg preg = MicroAPI::CreateMask<uint8_t, MicroAPI::MaskPattern::ALL>();
        MicroAPI::MaskReg pregVsel = MicroAPI::CreateMask<uint16_t, MicroAPI::MaskPattern::ALL>();
        MicroAPI::Duplicate<int8_t, MicroAPI::MaskMergeMode::ZEROING>(wdup0, wParams.shiftLeftSize, preg);
        MicroAPI::Duplicate<int8_t, MicroAPI::MaskMergeMode::ZEROING>(wdup1, SHIFT_RIGHT_SIZE, preg);
        MicroAPI::Duplicate<int8_t, MicroAPI::MaskMergeMode::ZEROING>(wdup2, wParams.andMask, preg);
        for (uint16_t repeatIdx = 0; repeatIdx < wParams.innerExtend; ++repeatIdx) {
            MicroAPI::AddrReg aregWeightB8In = MicroAPI::CreateAddrReg<uint8_t>(repeatIdx, wParams.innerSrcExtend);
            MicroAPI::AddrReg aregWeightB8Out = MicroAPI::CreateAddrReg<uint8_t>(repeatIdx, wParams.innerDstExtend);
            MicroAPI::LoadAlign<uint8_t, MicroAPI::LoadDist::DIST_US_B8>(
                (MicroAPI::RegTensor<uint8_t>&)wLoad0, (__ubuf__ uint8_t*&)wParams.weightInUbBaseAddr, aregWeightB8In);
            MicroAPI::ShiftRight(wShr0, wLoad0, wdup0, preg);
            MicroAPI::ShiftLeft(wShl, wLoad0, wdup1, preg);
            MicroAPI::ShiftRight(wShr1, wShl, wdup0, preg);
            MicroAPI::Select(wSel0, wShr1, wShr0, pregVsel);
            MicroAPI::And(sAnd0, wSel0, wdup2, preg);
            MicroAPI::StoreAlign<uint8_t, MicroAPI::StoreDist::DIST_NORM_B8>(
                (__ubuf__ uint8_t*&)wParams.weightOutUbAddr, (MicroAPI::RegTensor<uint8_t>&)sAnd0, aregWeightB8Out,
                preg);
        }
    }

    static constexpr int64_t DOUBLE_BUFFER = 2;
    static constexpr int64_t QUADRUPLE_BUFFER = 4;
    static constexpr uint64_t SYNC_MODE4 = 4;
    static constexpr uint64_t L1_BUFFER_SIZE = 512 * 1024;
    static constexpr uint64_t L1_BUFFER_HALF_SIZE = 256 * 1024;
    static constexpr uint64_t INT4_DTYPE_PARAM = 1;
    static constexpr uint64_t BLOCK_NUM_REG = VECTOR_REG_WIDTH / ONE_BLK_SIZE;
    static constexpr uint64_t SYNC_AIV_AIC_FLAG = 2;
    static constexpr uint64_t SINGLE_BUFFER = 1;
    static constexpr uint64_t GROUP_SIZE = 32;
    static constexpr int32_t C0_SIZE_B8 = 32;
    static constexpr int32_t UB_ALIGN_SIZE_FOR_4BITS = 64;
    static constexpr uint32_t DUP_CONFIG_2 = 0x2;
    static constexpr uint32_t DUP_CONFIG_MODE_1C = 0x1C;
    static constexpr uint32_t DUP_CONFIG_4 = 0x4;
    static constexpr uint32_t DUP_FLAG_80 = 0x80;
    static constexpr uint32_t E1M2_SHIFT_LEFT_SIZE = 0x3;
    static constexpr uint32_t E1M2_AND_MASK = 0x8E;
    static constexpr uint32_t E2M1_SHIFT_LEFT_SIZE = 0x2;
    static constexpr uint32_t E2M1_AND_MASK = 0x9C;
    static constexpr uint32_t SHIFT_RIGHT_SIZE = 0x4;
    static constexpr int32_t VEC_MAX_ELEM_B8 = VECTOR_REG_WIDTH / sizeof(ElementOut);
    static constexpr int32_t VEC_MAX_ELEM_B16 = VECTOR_REG_WIDTH / sizeof(ElementBias);
    static constexpr int32_t VECTOR_REG_WIDTH_FOR_4BITS = 512;
    static constexpr int32_t OFFSET_64 = 64;
    static constexpr uint64_t BIAS_SPLIT_N_L1_SIZE = 256UL;
    static constexpr ElementBias BIAS_REDUCE_FACTOR = static_cast<ElementBias>(0.015625f);

    uint64_t nSize_;
    uint64_t kSize_;
    int32_t nUbSize_;
    int32_t kUbSize_;
    int32_t nUbLen_;
    int32_t kUbLen_;
    int32_t nBiasUbLen_ = 0;
    int64_t l1BufNum_;
    uint64_t nL1Size_;
    uint64_t kL1Size_;
    uint64_t kGmOffset_;
    int32_t nL1Len_;
    int32_t kL1Len_;
    uint64_t aL1Size_;
    uint64_t bL1Size_;
    uint64_t vecWeightOutLen_;
    uint64_t vecWeightInLen_;
    uint64_t vecBiasLen_;
    uint64_t ubBufIdx_;
    int64_t l1BufIdx_ = 0;
    int64_t idx_ = -1;
    uint64_t nL1Offset_ = 0;
    uint64_t kL1Offset_ = 0;
    uint64_t nL1BiasOffset_ = 0;
    uint8_t vecBufNum_ = SINGLE_BUFFER;
    uint8_t occupied_ = 0; // unused
    bool hasBias_;
    bool calcBias_ = false;
    __ubuf__ ElementOut* weightOutUbAddr_;
    __ubuf__ ElementOut* weightOutUbAddr1_;
    __ubuf__ int8_t* weightInUbBaseAddr_;
    AscendC::LocalTensor<ElementIn> weightInUb_;
    AscendC::LocalTensor<ElementOut> weightOutUb_;
    AscendC::LocalTensor<ElementBias> biasOutUb_;
    AscendC::LocalTensor<ElementBias> biasInUb_;
    AscendC::LocalTensor<ElementOut> l1Local_;
    static constexpr bool weightNz = Gemm::is_2d_nz_c0_32<decltype(LayoutIn{}.GetStride())>::value;
};
} // namespace Cmct::Prologue

#endif