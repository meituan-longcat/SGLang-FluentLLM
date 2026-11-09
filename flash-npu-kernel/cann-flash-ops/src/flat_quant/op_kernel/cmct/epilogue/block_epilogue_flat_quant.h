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
 * \file block_epilogue_flat_quant.h
 * \brief
 */

#pragma once
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "fusion/default_fusion_op.h"

struct FlatQuantShapeInfo {
    int64_t K;
    int64_t M;
    int64_t N;
    int64_t Mceil;
    int64_t Nceil;
};

namespace Cmct {
namespace Gemm {
namespace Block {
template <typename DataTypeIn_, typename DataTypeOut_, typename DataTypeScale_,
          typename FusionOp_ = DefaultFusion<DataTypeOut_, DataTypeIn_>>
class BlockEpilogueFlatQuant {
public:
    __aicore__ inline BlockEpilogueFlatQuant() {}

    struct Arguments {
        GM_ADDR outGmAddr{nullptr};
        GM_ADDR scaleGmAddr{nullptr};
    };

    struct Params {
        GM_ADDR outGmAddr{nullptr};
        GM_ADDR scaleGmAddr{nullptr};
    };

    using DataTypeIn = DataTypeIn_;
    using DataTypeOut = DataTypeOut_;
    using DataTypeScale = DataTypeScale_;
    using FusionOp = FusionOp_;

    // block shape
    using BlockShape = Shape<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>;
    using ProblemShape = Shape<int64_t, int64_t, int64_t, int64_t>;
    TPipe pipe_;

    // input ub tensor and output global tensor
    AscendC::LocalTensor<DataTypeIn> ubLocal_{AscendC::TPosition::VECIN, 0, AscendC::TOTAL_UB_SIZE};
    AscendC::GlobalTensor<int8_t> cGlobal_;
    AscendC::GlobalTensor<int8_t> scaleGlobal_;

    // attribute
    ProblemShape problemShape_;

    __aicore__ inline void Init(Params const& params, ProblemShape& problemShape, float dstTypeMax, float invDstTypeMax)
    {
        cGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(params.outGmAddr));
        scaleGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(params.scaleGmAddr));
        problemShape_ = problemShape;

        shape_.M = Get<MNK_M>(problemShape_);
        shape_.N = Get<MNK_N>(problemShape_);
        shape_.K = Get<MNK_B>(problemShape_);
        dstTypeMax_ = dstTypeMax;
        invDstTypeMax_ = invDstTypeMax;
        if (dstTypeMax_ == SIX_FLOAT) {
            addValueBit_ = ADD_VALUE_FOR_BF16_MAN1;
        } else if (dstTypeMax_ == SEVEN_FLOAT) {
            addValueBit_ = ADD_VALUE_FOR_BF16_MAN2;
        }

        shape_.Mceil = FlatQuantAlign(shape_.M, CEIL_SIZE);
        shape_.Nceil = VEC_N_LEN;
        alignM_ = FlatQuantCeilDiv(shape_.M * shape_.N, VEC_N_LEN);

        pipe_.InitBuffer(bufQueue_, AscendC::TOTAL_UB_SIZE);
        xTensor_ = bufQueue_.Get<bfloat16_t>();
        yTensor_ = xTensor_[MN_SIZE].template ReinterpretCast<int8_t>();
        eMaxTensor_ = yTensor_[OUT_SIZE].template ReinterpretCast<uint16_t>();
        deQuantScaleTensor_ = eMaxTensor_[EMAX_SIZE];
        scaleTensor_ = deQuantScaleTensor_[EMAX_SIZE].template ReinterpretCast<int8_t>();
        scaleBlockTensor_ = scaleTensor_[EMAX_SIZE];

        eventIdVToMte3_ = static_cast<event_t>(pipe_.FetchEventID(HardEvent::V_MTE3));
        eventIdMte3ToV_ = static_cast<event_t>(pipe_.FetchEventID(HardEvent::MTE3_V));
    }

    __aicore__ inline void Quant(uint64_t batchIdx, uint64_t iterIdx)
    {
        int64_t mnSize = shape_.M * shape_.N;
        uint64_t yOffset = batchIdx * static_cast<uint64_t>(mnSize);
        uint64_t scaleOffset = batchIdx * FlatQuantCeilDiv(mnSize, MXFP_DIVISOR_SIZE) * 2;
        uint32_t totalDataInUB = static_cast<uint32_t>(mnSize);
        uint64_t inputOffset = iterIdx * totalDataInUB;

        ComputeMxQuant(xTensor_, yTensor_, eMaxTensor_, scaleTensor_, deQuantScaleTensor_, totalDataInUB, inputOffset);
        ComputeTransLayout(scaleTensor_, scaleBlockTensor_, static_cast<uint16_t>(alignM_),
                           static_cast<uint16_t>(shape_.Nceil));
        AscendC::SetFlag<HardEvent::V_MTE3>(eventIdVToMte3_);
        AscendC::WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3_);

        CopyOutputFromUbToGm(yOffset, yTensor_);
        CopyScaleFromUbToGm(scaleOffset, scaleBlockTensor_);
        AscendC::SetFlag<HardEvent::MTE3_V>(eventIdMte3ToV_);
        AscendC::WaitFlag<HardEvent::MTE3_V>(eventIdMte3ToV_);
    }

    __aicore__ inline void CopyOutputFromUbToGm(uint64_t offset, AscendC::LocalTensor<int8_t>& src)
    {
        uint64_t alignedOffset = offset >> 1;
        copy_ubuf_to_gm_align_v2(cGlobal_[alignedOffset].GetPhyAddr(), (__ubuf__ void*)src.GetPhyAddr(), 0, 1,
                                 static_cast<uint32_t>((shape_.M * shape_.N * sizeof(int8_t)) >> 1), 0, 0,
                                 0); // (dstAddr,srcAddr,SID,blockCount,blockLen,l2Cache, dstGap,srcGap)
    }

    __aicore__ inline void CopyScaleFromUbToGm(uint64_t offset, AscendC::LocalTensor<int8_t>& src)
    {
        uint32_t blockCount = static_cast<uint32_t>(FlatQuantCeilDiv(alignM_ * shape_.Nceil, MXFP_DIVISOR_SIZE));
        copy_ubuf_to_gm_align_v2(scaleGlobal_[offset].GetPhyAddr(), (__ubuf__ void*)src.GetPhyAddr(), 0, blockCount,
                                 BLOCK_SCALE, 0, BLOCK_SCALE,
                                 32); // (dstAddr,srcAddr,SID,blockCount,blockLen,l2Cache, dstGap,srcGap)
    }

    __aicore__ inline void ComputeMxQuant(LocalTensor<bfloat16_t>& xTensor, LocalTensor<int8_t>& yTensor,
                                          LocalTensor<uint16_t>& eMaxTensor, LocalTensor<int8_t>& scaleTensor,
                                          LocalTensor<uint16_t>& deQuantScaleTensor, uint32_t totalDataInUB,
                                          uint64_t inputOffset)
    {
        uint32_t oneRepeatSize = AscendC::GetVecLen() / sizeof(DataTypeIn);
        uint16_t repeatCount = static_cast<uint16_t>(FlatQuantCeilDiv(totalDataInUB, oneRepeatSize * 2));
        uint16_t scaleNum = static_cast<uint16_t>(FlatQuantCeilDiv(totalDataInUB, GROUP_SIZE));
        uint16_t repeatScaleCount = static_cast<uint16_t>(FlatQuantCeilDiv(scaleNum, oneRepeatSize));
        uint16_t repeatScaleHalfCount = static_cast<uint16_t>(FlatQuantCeilDiv(scaleNum, oneRepeatSize / 2));

        __ubuf__ bfloat16_t* xAddr = (__ubuf__ bfloat16_t*)xTensor.GetPhyAddr() + inputOffset;
        __ubuf__ uint16_t* maxExpAddr = (__ubuf__ uint16_t*)eMaxTensor.GetPhyAddr();
        if (dstTypeMax_ >= SIX_FLOAT && dstTypeMax_ <= TWELVE_FLOAT) {
            AscendC::VF_CALL<ExpMaxVfcuBLAS>(maxExpAddr, xAddr, totalDataInUB, repeatCount, oneRepeatSize);
        } else {
            AscendC::VF_CALL<ExpMaxVf>(maxExpAddr, xAddr, totalDataInUB, repeatCount, oneRepeatSize);
        }

        __ubuf__ uint16_t* deScaleAddr = (__ubuf__ uint16_t*)deQuantScaleTensor.GetPhyAddr();
        __ubuf__ uint16_t* scaleAddr = (__ubuf__ uint16_t*)scaleTensor.GetPhyAddr();
        if (dstTypeMax_ == ZERO_FLOAT) {
            AscendC::VF_CALL<ScaleVf>(scaleAddr, deScaleAddr, maxExpAddr, scaleNum, repeatScaleCount);
        } else if (dstTypeMax_ == SIX_FLOAT || dstTypeMax_ == SEVEN_FLOAT) {
            AscendC::VF_CALL<ScaleVfDynamic>(scaleAddr, deScaleAddr, maxExpAddr, scaleNum, repeatScaleCount,
                                             addValueBit_);
        } else {
            AscendC::VF_CALL<ScaleVfcuBLAS>(scaleAddr, deScaleAddr, maxExpAddr, scaleNum, repeatScaleHalfCount,
                                            invDstTypeMax_);
        }

        __ubuf__ int8_t* yAddr = (__ubuf__ int8_t*)yTensor.GetPhyAddr();
        AscendC::VF_CALL<QuantVf>(yAddr, xAddr, deScaleAddr, totalDataInUB, repeatCount);
    }

    __aicore__ inline void ComputeTransLayout(LocalTensor<int8_t>& scaleTensor, LocalTensor<int8_t>& scaleBlockTensor,
                                              uint16_t M, uint16_t N)
    {
        uint16_t scaleBlockN = FlatQuantCeilDiv(N, MXFP_DIVISOR_SIZE) * 2;

        __ubuf__ int8_t* qscaleAddr = (__ubuf__ int8_t*)scaleTensor.GetPhyAddr();
        __ubuf__ int8_t* qscaleBlkAddr = (__ubuf__ int8_t*)scaleBlockTensor.GetPhyAddr();
        AscendC::VF_CALL<TransLayoutVf>(qscaleAddr, qscaleBlkAddr, M, scaleBlockN);
    }

    static __simd_vf__ inline void ExpMaxVf(__ubuf__ uint16_t* dstPtr, __ubuf__ bfloat16_t* srcPtr, uint32_t count,
                                            uint16_t repeatTimes, uint32_t oneRepeatSize)
    {
        AscendC::MicroAPI::RegTensor<bfloat16_t> vSrcReg0;
        AscendC::MicroAPI::RegTensor<bfloat16_t> vSrcReg1;
        AscendC::MicroAPI::RegTensor<uint16_t> vExpExtract0;
        AscendC::MicroAPI::RegTensor<uint16_t> vExpExtract1;
        AscendC::MicroAPI::RegTensor<uint16_t> vdMaxExp;

        AscendC::MicroAPI::RegTensor<uint16_t> expMaskBF16;
        AscendC::MicroAPI::Duplicate(expMaskBF16, MAX_EXP_FOR_BF16);

        AscendC::MicroAPI::MaskReg maskReg;
        AscendC::MicroAPI::UnalignReg u1;
        AscendC::MicroAPI::AddrReg aReg;

        for (uint16_t i = 0; i < repeatTimes; i++) {
            aReg = AscendC::MicroAPI::CreateAddrReg<uint32_t>(i, oneRepeatSize);
            maskReg = AscendC::MicroAPI::UpdateMask<bfloat16_t>(count);

            AscendC::MicroAPI::LoadAlign<bfloat16_t, AscendC::MicroAPI::LoadDist::DIST_DINTLV_B16>(vSrcReg0, vSrcReg1,
                                                                                                   srcPtr, aReg);
            AscendC::MicroAPI::And(vExpExtract0, (AscendC::MicroAPI::RegTensor<uint16_t>&)vSrcReg0, expMaskBF16,
                                   maskReg);
            AscendC::MicroAPI::And(vExpExtract1, (AscendC::MicroAPI::RegTensor<uint16_t>&)vSrcReg1, expMaskBF16,
                                   maskReg);
            AscendC::MicroAPI::Max(vdMaxExp, vExpExtract0, vExpExtract1, maskReg);
            AscendC::MicroAPI::ReduceDataBlock<AscendC::MicroAPI::ReduceType::MAX>(vdMaxExp, vdMaxExp, maskReg);
            AscendC::MicroAPI::StoreUnAlign<uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                dstPtr, vdMaxExp, u1, STORE_UNALIGN_STRIDE_BYTES);
        }
        AscendC::MicroAPI::StoreUnAlignPost(dstPtr, u1, 0);
    }

    static __simd_vf__ inline void ExpMaxVfcuBLAS(__ubuf__ uint16_t* dstPtr, __ubuf__ bfloat16_t* srcPtr,
                                                  uint32_t count, uint16_t repeatTimes, uint32_t oneRepeatSize)
    {
        AscendC::MicroAPI::RegTensor<bfloat16_t> vSrcReg0;
        AscendC::MicroAPI::RegTensor<bfloat16_t> vSrcReg1;
        AscendC::MicroAPI::RegTensor<uint16_t> vdMaxExp;

        AscendC::MicroAPI::RegTensor<uint16_t> absMask16Bit;
        AscendC::MicroAPI::Duplicate(absMask16Bit, ABS_MASK_FOR_16BIT);

        AscendC::MicroAPI::MaskReg maskReg;
        AscendC::MicroAPI::UnalignReg u1;
        AscendC::MicroAPI::AddrReg aReg;

        for (uint16_t i = 0; i < repeatTimes; i++) {
            aReg = AscendC::MicroAPI::CreateAddrReg<uint32_t>(i, oneRepeatSize);
            maskReg = AscendC::MicroAPI::UpdateMask<bfloat16_t>(count);

            AscendC::MicroAPI::LoadAlign<bfloat16_t, AscendC::MicroAPI::LoadDist::DIST_DINTLV_B16>(vSrcReg0, vSrcReg1,
                                                                                                   srcPtr, aReg);
            AscendC::MicroAPI::And((AscendC::MicroAPI::RegTensor<uint16_t>&)vSrcReg0,
                                   (AscendC::MicroAPI::RegTensor<uint16_t>&)vSrcReg0, absMask16Bit, maskReg);
            AscendC::MicroAPI::And((AscendC::MicroAPI::RegTensor<uint16_t>&)vSrcReg1,
                                   (AscendC::MicroAPI::RegTensor<uint16_t>&)vSrcReg1, absMask16Bit, maskReg);
            AscendC::MicroAPI::Max(vdMaxExp, (AscendC::MicroAPI::RegTensor<uint16_t>&)vSrcReg0,
                                   (AscendC::MicroAPI::RegTensor<uint16_t>&)vSrcReg1, maskReg);

            AscendC::MicroAPI::ReduceDataBlock<AscendC::MicroAPI::ReduceType::MAX>(vdMaxExp, vdMaxExp, maskReg);
            AscendC::MicroAPI::StoreUnAlign<uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                dstPtr, vdMaxExp, u1, STORE_UNALIGN_STRIDE_BYTES);
        }
        AscendC::MicroAPI::StoreUnAlignPost(dstPtr, u1, 0);
    }

    static __simd_vf__ inline void ScaleVf(__ubuf__ uint16_t* dstPtr, __ubuf__ uint16_t* dst2Ptr,
                                           __ubuf__ uint16_t* srcPtr, uint32_t scaleNum, uint16_t repeatTimes)
    {
        AscendC::MicroAPI::RegTensor<uint16_t> expMask, sharedExp, scaleValue, scaleBias, halfScale, fp8NanRegTensor;
        AscendC::MicroAPI::Duplicate(expMask, MAX_EXP_FOR_BF16);
        AscendC::MicroAPI::RegTensor<uint16_t> vdMaxExp;
        AscendC::MicroAPI::RegTensor<bfloat16_t> vdExp0, vdExp1;
        AscendC::MicroAPI::MaskReg cmpResult, zeroMask, cmpResultSub, preMaskScale;
        AscendC::MicroAPI::RegTensor<uint16_t> maxExpValue, zeroRegTensor, nanRegTensor, specialExpRegTensor;
        AscendC::MicroAPI::Duplicate(maxExpValue, FP4_E2M1_MAX_EXP);
        AscendC::MicroAPI::Duplicate(scaleBias, BF16_EXP_BIAS);
        AscendC::MicroAPI::Duplicate(fp8NanRegTensor, MAX_EXP_FOR_FP8);
        AscendC::MicroAPI::Duplicate(zeroRegTensor, 0);
        AscendC::MicroAPI::Duplicate(nanRegTensor, NAN_CUSTOMIZATION);

        AscendC::MicroAPI::MaskReg invalidDataMask, specialDataMask;
        AscendC::MicroAPI::Duplicate(specialExpRegTensor, SPECIAL_EXP_THRESHOLD);
        for (uint16_t i = 0; i < repeatTimes; i++) {
            preMaskScale = AscendC::MicroAPI::UpdateMask<uint16_t>(scaleNum);
            AscendC::MicroAPI::LoadAlign<uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(vdMaxExp, srcPtr,
                                                                                                     128);
            AscendC::MicroAPI::Compare<uint16_t, AscendC::CMPMODE::NE>(cmpResult, vdMaxExp, expMask, preMaskScale);
            AscendC::MicroAPI::Compare<uint16_t, AscendC::CMPMODE::NE>(zeroMask, vdMaxExp, zeroRegTensor, preMaskScale);
            AscendC::MicroAPI::Compare<uint16_t, AscendC::CMPMODE::LE>(invalidDataMask, vdMaxExp, maxExpValue,
                                                                       preMaskScale);
            AscendC::MicroAPI::Select<uint16_t>(vdMaxExp, maxExpValue, vdMaxExp, invalidDataMask);
            AscendC::MicroAPI::Sub(sharedExp, vdMaxExp, maxExpValue, preMaskScale);
            AscendC::MicroAPI::ShiftRights(scaleValue, sharedExp, SHR_NUM_FOR_BF16, preMaskScale);
            AscendC::MicroAPI::Select<uint16_t>(scaleValue, scaleValue, fp8NanRegTensor, cmpResult);
            AscendC::MicroAPI::Select<uint16_t>(scaleValue, scaleValue, zeroRegTensor, zeroMask);
            AscendC::MicroAPI::StoreAlign<uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                                          AscendC::MicroAPI::StoreDist::DIST_PACK_B16>(dstPtr, scaleValue, 64,
                                                                                       preMaskScale);
            AscendC::MicroAPI::Compare<uint16_t, AscendC::CMPMODE::EQ>(specialDataMask, sharedExp, scaleBias,
                                                                       preMaskScale);

            AscendC::MicroAPI::Sub(halfScale, scaleBias, sharedExp, preMaskScale);
            AscendC::MicroAPI::Select<uint16_t>(halfScale, halfScale, nanRegTensor, cmpResult);
            AscendC::MicroAPI::Select<uint16_t>(halfScale, halfScale, zeroRegTensor, zeroMask);
            AscendC::MicroAPI::Select<uint16_t>(halfScale, specialExpRegTensor, halfScale, specialDataMask);
            AscendC::MicroAPI::StoreAlign<uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                dst2Ptr, halfScale, 128, preMaskScale);
        }
    }

    static __simd_vf__ inline void ScaleVfDynamic(__ubuf__ uint16_t* dstPtr, __ubuf__ uint16_t* dst2Ptr,
                                                  __ubuf__ uint16_t* srcPtr, uint32_t scaleNum, uint16_t repeatTimes,
                                                  uint16_t addValueBit)
    {
        AscendC::MicroAPI::RegTensor<uint16_t> vdMaxExp;
        AscendC::MicroAPI::RegTensor<uint16_t> sharedExp;
        AscendC::MicroAPI::RegTensor<uint16_t> scaleValue;
        AscendC::MicroAPI::RegTensor<uint16_t> halfScale;
        AscendC::MicroAPI::RegTensor<uint16_t> vdMaxExpAdd;
        AscendC::MicroAPI::RegTensor<uint16_t> vdMaxExpOnly;

        AscendC::MicroAPI::RegTensor<uint16_t> expMask;
        AscendC::MicroAPI::Duplicate(expMask, MAX_EXP_FOR_BF16);
        AscendC::MicroAPI::RegTensor<uint16_t> addValue;
        AscendC::MicroAPI::Duplicate(addValue, addValueBit);
        AscendC::MicroAPI::RegTensor<uint16_t> maxExpValue;
        AscendC::MicroAPI::Duplicate(maxExpValue, FP4_E2M1_MAX_EXP);
        AscendC::MicroAPI::RegTensor<uint16_t> scaleBias;
        AscendC::MicroAPI::Duplicate(scaleBias, BF16_EXP_BIAS);
        AscendC::MicroAPI::RegTensor<uint16_t> fp8NanRegTensor;
        AscendC::MicroAPI::Duplicate(fp8NanRegTensor, MAX_EXP_FOR_FP8);
        AscendC::MicroAPI::RegTensor<uint16_t> zeroRegTensor;
        AscendC::MicroAPI::Duplicate(zeroRegTensor, 0);
        AscendC::MicroAPI::RegTensor<uint16_t> nanRegTensor;
        AscendC::MicroAPI::Duplicate(nanRegTensor, NAN_CUSTOMIZATION);
        AscendC::MicroAPI::RegTensor<uint16_t> specialExpRegTensor;
        AscendC::MicroAPI::Duplicate(specialExpRegTensor, SPECIAL_EXP_THRESHOLD);

        AscendC::MicroAPI::MaskReg cmpResult;
        AscendC::MicroAPI::MaskReg zeroMask;
        AscendC::MicroAPI::MaskReg invalidDataMask;
        AscendC::MicroAPI::MaskReg specialDataMask;
        AscendC::MicroAPI::MaskReg preMaskScale;

        for (uint16_t i = 0; i < repeatTimes; i++) {
            preMaskScale = AscendC::MicroAPI::UpdateMask<uint16_t>(scaleNum);
            AscendC::MicroAPI::LoadAlign<uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(vdMaxExp, srcPtr,
                                                                                                     vfLen16);
            AscendC::MicroAPI::And(vdMaxExpOnly, vdMaxExp, expMask, preMaskScale); // Extract exponent bits
            AscendC::MicroAPI::Compare<uint16_t, AscendC::CMPMODE::NE>(cmpResult, vdMaxExpOnly, expMask,
                                                                       preMaskScale); // Check INF/NAN
            AscendC::MicroAPI::Compare<uint16_t, AscendC::CMPMODE::NE>(zeroMask, vdMaxExpOnly, zeroRegTensor,
                                                                       preMaskScale);
            AscendC::MicroAPI::Compare<uint16_t, AscendC::CMPMODE::LT>(invalidDataMask, vdMaxExpOnly, maxExpValue,
                                                                       preMaskScale);

            AscendC::MicroAPI::Add(vdMaxExpAdd, vdMaxExp, addValue, preMaskScale); // Result after carry
            AscendC::MicroAPI::And(vdMaxExpAdd, vdMaxExpAdd, expMask,
                                   preMaskScale); // Extract exponent bits from carry result
            AscendC::MicroAPI::Select<uint16_t>(vdMaxExpAdd, maxExpValue, vdMaxExpAdd, invalidDataMask);
            AscendC::MicroAPI::Sub(sharedExp, vdMaxExpAdd, maxExpValue, preMaskScale);

            AscendC::MicroAPI::ShiftRights(scaleValue, sharedExp, SHR_NUM_FOR_BF16, preMaskScale);
            AscendC::MicroAPI::Select<uint16_t>(scaleValue, scaleValue, fp8NanRegTensor, cmpResult);
            AscendC::MicroAPI::Select<uint16_t>(scaleValue, scaleValue, zeroRegTensor, zeroMask);

            AscendC::MicroAPI::StoreAlign<uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                                          AscendC::MicroAPI::StoreDist::DIST_PACK_B16>(dstPtr, scaleValue, vfLen32,
                                                                                       preMaskScale);

            AscendC::MicroAPI::Compare<uint16_t, AscendC::CMPMODE::EQ>(specialDataMask, sharedExp, scaleBias,
                                                                       preMaskScale);
            AscendC::MicroAPI::Sub(halfScale, scaleBias, sharedExp, preMaskScale);
            AscendC::MicroAPI::Select<uint16_t>(halfScale, halfScale, nanRegTensor, cmpResult);
            AscendC::MicroAPI::Select<uint16_t>(halfScale, halfScale, zeroRegTensor, zeroMask);
            AscendC::MicroAPI::Select<uint16_t>(halfScale, specialExpRegTensor, halfScale, specialDataMask);

            AscendC::MicroAPI::StoreAlign<uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                                          AscendC::MicroAPI::StoreDist::DIST_NORM>(dst2Ptr, halfScale, vfLen16,
                                                                                   preMaskScale);
        }
    }

    static __simd_vf__ inline void ScaleVfcuBLAS(__ubuf__ uint16_t* dstPtr, __ubuf__ uint16_t* dst2Ptr,
                                                 __ubuf__ uint16_t* srcPtr, uint32_t scaleNum, uint16_t repeatTimes,
                                                 float invDstTypeMax)
    {
        AscendC::MicroAPI::RegTensor<uint16_t> max16;
        AscendC::MicroAPI::RegTensor<uint32_t> max32;
        AscendC::MicroAPI::RegTensor<uint32_t> exp32;
        AscendC::MicroAPI::RegTensor<uint32_t> man32;
        AscendC::MicroAPI::RegTensor<uint32_t> normalExp32;
        AscendC::MicroAPI::RegTensor<uint32_t> expAddOne32;
        AscendC::MicroAPI::RegTensor<uint32_t> extractExp;
        AscendC::MicroAPI::RegTensor<uint16_t> expOut;
        AscendC::MicroAPI::RegTensor<uint32_t> halfScale;
        AscendC::MicroAPI::RegTensor<uint16_t> recExpOut;

        AscendC::MicroAPI::RegTensor<uint32_t> manMaskFP32;
        AscendC::MicroAPI::Duplicate(manMaskFP32, MAN_MASK_FLOAT);
        AscendC::MicroAPI::RegTensor<uint32_t> expMask;
        AscendC::MicroAPI::Duplicate(expMask, MAX_EXP_FOR_FP32);
        AscendC::MicroAPI::RegTensor<uint32_t> zeroRegTensor32;
        AscendC::MicroAPI::Duplicate(zeroRegTensor32, 0);
        AscendC::MicroAPI::RegTensor<uint32_t> scaleBias;
        AscendC::MicroAPI::Duplicate(scaleBias, FP32_EXP_BIAS_CUBLAS);
        AscendC::MicroAPI::RegTensor<uint32_t> nanRegTensor;
        AscendC::MicroAPI::Duplicate(nanRegTensor, NAN_CUSTOMIZATION_PACK);
        AscendC::MicroAPI::RegTensor<uint32_t> fp4NanRegTensor;
        AscendC::MicroAPI::Duplicate(fp4NanRegTensor, MAX_EXP_FOR_FP8_IN_FP32);
        AscendC::MicroAPI::RegTensor<float> invMax;
        AscendC::MicroAPI::Duplicate(invMax, invDstTypeMax);

        AscendC::MicroAPI::MaskReg cmpResult;
        AscendC::MicroAPI::MaskReg zeroMask;
        AscendC::MicroAPI::MaskReg p0;
        AscendC::MicroAPI::MaskReg p1;
        AscendC::MicroAPI::MaskReg p2;
        AscendC::MicroAPI::MaskReg preMaskScale;
        uint32_t SixtyFour = 64;
        AscendC::MicroAPI::MaskReg dataMaskB16Half = AscendC::MicroAPI::UpdateMask<uint16_t>(SixtyFour);

        static constexpr AscendC::MicroAPI::CastTrait castTraitHalf2Float = {
            AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::UNKNOWN,
            AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};

        for (uint16_t i = 0; i < repeatTimes; i++) {
            preMaskScale = AscendC::MicroAPI::UpdateMask<uint32_t>(scaleNum);
            AscendC::MicroAPI::LoadAlign<uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                                         AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(max16, srcPtr, vfLen32);

            AscendC::MicroAPI::Cast<float, bfloat16_t, castTraitHalf2Float>(
                (AscendC::MicroAPI::RegTensor<float>&)max32, (AscendC::MicroAPI::RegTensor<bfloat16_t>&)max16,
                preMaskScale);
            AscendC::MicroAPI::Compare<uint32_t, AscendC::CMPMODE::LT>(cmpResult, max32, expMask, preMaskScale);
            AscendC::MicroAPI::Compare<uint32_t, AscendC::CMPMODE::NE>(zeroMask, max32, zeroRegTensor32, preMaskScale);

            AscendC::MicroAPI::Mul((AscendC::MicroAPI::RegTensor<float>&)max32,
                                   (AscendC::MicroAPI::RegTensor<float>&)max32, invMax, preMaskScale);
            AscendC::MicroAPI::ShiftRights(exp32, max32, SHR_NUM_FOR_FP32, preMaskScale);
            AscendC::MicroAPI::And(man32, max32, manMaskFP32, preMaskScale);

            AscendC::MicroAPI::CompareScalar<uint32_t, AscendC::CMPMODE::GT>(p0, exp32, NUMBER_ZERO, preMaskScale);
            AscendC::MicroAPI::CompareScalar<uint32_t, AscendC::CMPMODE::LT>(p1, exp32, NUMBER_TWO_FIVE_FOUR,
                                                                             preMaskScale);
            AscendC::MicroAPI::CompareScalar<uint32_t, AscendC::CMPMODE::GT>(p2, man32, NUMBER_ZERO, preMaskScale);
            AscendC::MicroAPI::MaskAnd(p0, p0, p1, preMaskScale);
            AscendC::MicroAPI::MaskAnd(p0, p0, p2, preMaskScale);

            AscendC::MicroAPI::Adds(expAddOne32, exp32, 1, preMaskScale);
            AscendC::MicroAPI::Select(extractExp, expAddOne32, exp32, p0);
            AscendC::MicroAPI::Select<uint32_t>(extractExp, extractExp, fp4NanRegTensor, cmpResult);
            AscendC::MicroAPI::Select<uint32_t>(extractExp, extractExp, zeroRegTensor32, zeroMask);
            AscendC::MicroAPI::Pack<uint16_t, uint32_t, AscendC::MicroAPI::HighLowPart::LOWEST>(expOut, extractExp);

            AscendC::MicroAPI::StoreAlign<uint16_t, AscendC::MicroAPI::StoreDist::DIST_PACK_B16>(
                dstPtr + i * SCALE_STORE_STRIDE, expOut, dataMaskB16Half);

            AscendC::MicroAPI::ShiftLefts(extractExp, extractExp, SHR_NUM_FOR_BF16, preMaskScale);
            AscendC::MicroAPI::Sub(halfScale, scaleBias, extractExp, preMaskScale);
            AscendC::MicroAPI::Select<uint32_t>(halfScale, halfScale, nanRegTensor, cmpResult);
            AscendC::MicroAPI::Select<uint32_t>(halfScale, halfScale, zeroRegTensor32, zeroMask);
            AscendC::MicroAPI::Pack<uint16_t, uint32_t, AscendC::MicroAPI::HighLowPart::LOWEST>(recExpOut, halfScale);

            AscendC::MicroAPI::StoreAlign<uint16_t>(dst2Ptr + i * vfLen32, recExpOut, dataMaskB16Half);
        }
    }

    static __simd_vf__ inline void QuantVf(__ubuf__ int8_t* dstPtr, __ubuf__ bfloat16_t* srcPtr,
                                           __ubuf__ uint16_t* src2Ptr, uint32_t oneRepeatSize, uint16_t repeatTimes)
    {
        AscendC::MicroAPI::MaskReg dataMask1;
        AscendC::MicroAPI::MaskReg dataMask2;
        AscendC::MicroAPI::RegTensor<uint16_t> halfScaleForMul;
        AscendC::MicroAPI::RegTensor<bfloat16_t> vdExp0;
        AscendC::MicroAPI::RegTensor<bfloat16_t> vdExp1;
        AscendC::MicroAPI::RegTensor<bfloat16_t> vdExp0Convert;
        AscendC::MicroAPI::RegTensor<bfloat16_t> vdExp1Convert;

        AscendC::MicroAPI::RegTensor<bfloat16_t> vdExp0BF16;
        AscendC::MicroAPI::RegTensor<bfloat16_t> vdExp1Bf16;

        AscendC::MicroAPI::RegTensor<fp4x2_e2m1_t> vdExp0FP4;
        AscendC::MicroAPI::RegTensor<fp4x2_e2m1_t> vdExp1FP4;

        AscendC::MicroAPI::RegTensor<bfloat16_t> vdBf16Exp0FP4;
        AscendC::MicroAPI::RegTensor<bfloat16_t> vdBf16Exp1FP4;

        AscendC::MicroAPI::AddrReg aReg;
        static constexpr AscendC::MicroAPI::CastTrait castTrait = {
            AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::UNKNOWN,
            AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};
        for (uint16_t i = 0; i < repeatTimes; i++) {
            aReg = AscendC::MicroAPI::CreateAddrReg<uint16_t>(i, oneRepeatSize);
            dataMask1 = AscendC::MicroAPI::UpdateMask<bfloat16_t>(oneRepeatSize);
            dataMask2 = AscendC::MicroAPI::UpdateMask<bfloat16_t>(oneRepeatSize);

            AscendC::MicroAPI::DataCopy<bfloat16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                                        AscendC::MicroAPI::LoadDist::DIST_DINTLV_B16>(
                vdExp0, vdExp1, srcPtr, 128 * 2); // copy two chunks from srcAddr to regbase
            AscendC::MicroAPI::DataCopy<uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                                        AscendC::MicroAPI::LoadDist::DIST_E2B_B16>(halfScaleForMul, src2Ptr, 8);
            AscendC::MicroAPI::Mul(vdExp0, vdExp0, (AscendC::MicroAPI::RegTensor<bfloat16_t>&)halfScaleForMul,
                                   dataMask1);
            AscendC::MicroAPI::Mul(vdExp1, vdExp1, (AscendC::MicroAPI::RegTensor<bfloat16_t>&)halfScaleForMul,
                                   dataMask1);
            AscendC::MicroAPI::Interleave(vdExp0, vdExp1, vdExp0, vdExp1);
            AscendC::MicroAPI::Cast<fp4x2_e2m1_t, bfloat16_t, castTrait>(vdExp0FP4, vdExp0, dataMask1);
            AscendC::MicroAPI::Cast<fp4x2_e2m1_t, bfloat16_t, castTrait>(vdExp1FP4, vdExp1, dataMask2);

            AscendC::MicroAPI::DataCopy<int8_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                                        AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(
                dstPtr, (AscendC::MicroAPI::RegTensor<int8_t>&)vdExp0FP4, 64, dataMask1);
            AscendC::MicroAPI::DataCopy<int8_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                                        AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(
                dstPtr, (AscendC::MicroAPI::RegTensor<int8_t>&)vdExp1FP4, 64, dataMask2);
        }
    }

    static __simd_vf__ inline void TransLayoutVf(__ubuf__ int8_t* scaleAddr, __ubuf__ int8_t* scaleBlkAddr,
                                                 uint16_t mSize, uint16_t scaleBlockN)
    {
        for (uint16_t mIdx = 0; mIdx < mSize; ++mIdx) {
            uint32_t eleNum = scaleBlockN;
            AscendC::MicroAPI::MaskReg maskScaleN = AscendC::MicroAPI::UpdateMask<int8_t>(eleNum);
            AscendC::MicroAPI::RegTensor<int8_t> vReg0;
            AscendC::MicroAPI::UnalignReg u0, u1;
            auto srcUb = scaleAddr + mIdx * scaleBlockN;
            AscendC::MicroAPI::DataCopyUnAlignPre(u0, srcUb);
            AscendC::MicroAPI::DataCopyUnAlign(vReg0, u0, srcUb);
            auto dstUb = scaleBlkAddr + mIdx * 32;
            AscendC::MicroAPI::DataCopy<int8_t, AscendC::MicroAPI::StoreDist::DIST_NORM_B8>(dstUb, vReg0, maskScaleN);
        }
    }

    __aicore__ inline auto GetTensor() { return ubLocal_; }
    __aicore__ inline void operator()(uint64_t startBatchIdx, uint64_t iterBatch)
    {
        for (uint64_t iter = 0; iter < iterBatch; ++iter) {
            Quant(startBatchIdx + iter, iter);
        }
    }
    template <typename T1, typename T2>
    __aicore__ inline auto FlatQuantCeilDiv(T1 a, T2 b) -> decltype(a + b)
    {
        if (b == 0) {
            return 0;
        }
        return (a + b - 1) / b;
    }
    template <typename T1, typename T2>
    __aicore__ inline auto FlatQuantAlign(T1 a, T2 b) -> decltype(a + b)
    {
        if (b == 0) {
            return 0;
        }
        return ((a + b - 1) / b) * b;
    }

    __host_aicore__ static Status CanImplement(Arguments const& args) { return Status::success; }

private:
    FlatQuantShapeInfo shape_;
    TBuf<QuePosition::VECCALC> bufQueue_;
    AscendC::LocalTensor<bfloat16_t> xTensor_;
    AscendC::LocalTensor<int8_t> yTensor_;
    AscendC::LocalTensor<uint16_t> eMaxTensor_;
    AscendC::LocalTensor<int8_t> scaleTensor_;
    AscendC::LocalTensor<uint16_t> deQuantScaleTensor_;
    AscendC::LocalTensor<int8_t> scaleBlockTensor_;

    event_t eventIdVToMte3_;
    event_t eventIdMte3ToV_;

    int64_t alignM_ = 0;
    float dstTypeMax_ = 0.0f;
    float invDstTypeMax_ = 0.0f;
    uint16_t addValueBit_ = 0;
    static constexpr int32_t CEIL_SIZE = 16;
    static constexpr int32_t GROUP_SIZE = 32;
    static constexpr int32_t VEC_N_LEN = 64;
    static constexpr int32_t MN_SIZE = 64 * 1024;
    static constexpr int32_t OUT_SIZE = 32 * 1024;
    static constexpr int32_t EMAX_SIZE = 2 * 1024;
    static constexpr int32_t MXFP_DIVISOR_SIZE = 64;
    static constexpr uint16_t MAX_EXP_FOR_BF16 = 0x7f80;
    static constexpr uint16_t MAX_EXP_FOR_FP8 = 0x00ff;
    static constexpr uint16_t BF16_EXP_BIAS = 0x7f00;
    static constexpr int16_t SHR_NUM_FOR_BF16 = 7;
    static constexpr uint16_t NAN_CUSTOMIZATION = 0x7f81;
    static constexpr uint16_t SPECIAL_EXP_THRESHOLD = 0x0040;
    static constexpr uint16_t FP4_E2M1_MAX_EXP = 0x0100;
    static constexpr uint16_t BLOCK_SCALE = 2;

    // Constants for cuBLAS FP4 scaling
    static constexpr float ZERO_FLOAT = 0.0f;
    static constexpr float SIX_FLOAT = 6.0f;
    static constexpr float SEVEN_FLOAT = 7.0f;
    static constexpr float TWELVE_FLOAT = 12.0f;
    static constexpr uint32_t STORE_UNALIGN_STRIDE_BYTES = 8;
    static constexpr uint32_t SCALE_STORE_STRIDE = 32;
    static constexpr int16_t SHR_NUM_FOR_FP32 = 23;
    static constexpr uint32_t NUMBER_ZERO = 0x00000000;
    static constexpr uint32_t NUMBER_TWO_FIVE_FOUR = 0x000000fe;
    static constexpr uint16_t ADD_VALUE_FOR_BF16_MAN1 = 0x003f;
    static constexpr uint16_t ADD_VALUE_FOR_BF16_MAN2 = 0x001f;
    static constexpr uint32_t vfLen16 = AscendC::GetVecLen() / sizeof(uint16_t);
    static constexpr uint32_t vfLen32 = AscendC::GetVecLen() / sizeof(uint32_t);
    static constexpr uint16_t ABS_MASK_FOR_16BIT = 0x7fff;
    static constexpr uint32_t MAN_MASK_FLOAT = 0x007fffff;
    static constexpr uint32_t MAX_EXP_FOR_FP32 = 0x7f800000;
    static constexpr uint32_t FP32_EXP_BIAS_CUBLAS = 0x00007f00;
    static constexpr uint32_t NAN_CUSTOMIZATION_PACK = 0x00007f81;
    static constexpr uint32_t MAX_EXP_FOR_FP8_IN_FP32 = 0x000000ff;
};
} // namespace Block
} // namespace Gemm
} // namespace Cmct
#endif // CMCT_INCLUDE_EPILOGUE_BLOCK_EPILOGUE_FLAT_QUANT_H
