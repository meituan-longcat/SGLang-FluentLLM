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
 * \file moe_distribute_dispatch_v2_quant.h
 * \brief
 */

#ifndef MOE_DISTRIBUTE_DISPATCH_V2_QUANT_H
#define MOE_DISTRIBUTE_DISPATCH_V2_QUANT_H

#include "moe_distribute_v2_constant.h"
#include "moe_distribute_v2_base.h"

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
#include "quantize_functions.h"
#endif

namespace Mc2Kernel {
using namespace AscendC;
using namespace MoeDistributeV2Base;

template <typename XType, typename ExpandXOutType, int32_t QuantMode, bool IsSmoothScaleExist, bool IsNeedAllgather>
class MoeDistributeDispatchV2Quant{
public:
    uint32_t axisH_{0};
    uint32_t hOutSizeAlign_{0};

    LocalTensor<float> floatLocalTemp_;
    LocalTensor<float> smoothScalesTensor_;
    GlobalTensor<uint8_t> dynamicScalesOutGMTensor_;

    TBuf<> smoothScalesBuf_;

    DataCopyParams scalesInParams_;
    DataCopyPadParams scalesPadParams_;
    DataCopyParams scaleOutParams_;

    __aicore__ inline MoeDistributeDispatchV2Quant() = default;

    __aicore__ inline void SetQuantInitParams(LocalTensor<float> floatLocalTemp, LocalTensor<float> smoothScalesTensor, 
                                              TBuf<> smoothScalesBuf, GlobalTensor<uint8_t> dynamicScalesOutGMTensor) 
    {
        floatLocalTemp_ = floatLocalTemp;
        smoothScalesBuf_ = smoothScalesBuf;
        smoothScalesTensor_ = smoothScalesTensor;
        dynamicScalesOutGMTensor_ = dynamicScalesOutGMTensor;

        scalesInParams_ = {1U, static_cast<uint16_t>(axisH_ * sizeof(float)), 0U, 0U};
        scalesPadParams_ = {true, 0, 0, 0};
    }

    __aicore__ inline void QuantInit(uint32_t &hAlignSize_, uint32_t &hOutSize_, uint32_t scaleInBytes_, 
                                     int32_t &tokenQuantAlign_, uint32_t &hScaleIdxSize_, uint32_t &scaleOutBytes, uint32_t axisH)
    {
        axisH_ = axisH;
        hOutSizeAlign_ = Ceil(hOutSize_, UB_ALIGN) * UB_ALIGN; // scale起始放置偏移
        hAlignSize_ = Ceil(axisH_ * sizeof(XType), UB_ALIGN) * UB_ALIGN; //用于搬入token数据xInQueue_大小申请
        if constexpr ((QuantMode == UNQUANT) && IsSmoothScaleExist) {
            hOutSizeAlign_ += scaleInBytes_;
            hAlignSize_ += scaleInBytes_;
            scaleOutBytes = scaleInBytes_; // 外部输入的scales大小
        } else if constexpr (QuantMode == PERTOKEN_DYNAMIC_QUANT) {
            hOutSizeAlign_ += sizeof(float); 
            scaleOutBytes = sizeof(float); // PERTOKEN量化一个token生成一个scale
        }
        #if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
        else if constexpr (QuantMode == MX_QUANT) {
            hOutSizeAlign_ = Align256(axisH_) * sizeof(ExpandXOutType);
            hAlignSize_ = Align128(axisH_) * sizeof(XType); // MX量化计算scale时每次搬入128个数据
            hOutSizeAlign_ += Align2(Ceil32(axisH_)); 
            scaleOutBytes = Align2(Ceil32(axisH_)) * sizeof(fp8_e8m0_t); // MX量化每32个值生成一个scale，且scale数量需为偶数
        } else if constexpr (QuantMode == PERGROUP_DYNAMIC_QUANT) {
            hOutSizeAlign_ = Align128(axisH_) * sizeof(ExpandXOutType);
            hAlignSize_ = Align128(axisH_) * sizeof(XType); // PERGROUP量化计算scale时每次搬入128个数据
            hOutSizeAlign_ += Ceil128(axisH_) * sizeof(float); 
            scaleOutBytes = Ceil128(axisH_) * sizeof(float); // PERGROUP量化每128个值生成一个scale
        }
        #endif
        uint32_t hScaleSizeAlign = Ceil(hOutSizeAlign_, UB_ALIGN) * UB_ALIGN; //保证后面填充三元组的起始地址对齐32
        tokenQuantAlign_ = hScaleSizeAlign / sizeof(int32_t);
        // 实际搬运大小，搬运Align32(token_align + scaleOutBytes) + 3*4B(三元组)
        hScaleIdxSize_ = hScaleSizeAlign + EXPAND_IDX_INFO * sizeof(int32_t);
    }

    __aicore__ inline void QuantProcess(LocalTensor<ExpandXOutType>& outLocal, LocalTensor<XType>& inLocal, uint32_t expertIndex,
                                        uint32_t scalesCount_, GlobalTensor<float> &scalesGMTensor_)
    {
        if constexpr (QuantMode == STATIC_QUANT) {
            QuantStatic(outLocal, inLocal, expertIndex, scalesCount_, scalesGMTensor_);
        } else if constexpr (QuantMode == PERTOKEN_DYNAMIC_QUANT) {
            QuantDynamicPerToken(outLocal, inLocal, expertIndex, scalesGMTensor_);
        } 
        #if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
        else if constexpr (QuantMode == MX_QUANT) {
            QuantDynamicMxFp8(outLocal, inLocal);
        } else if constexpr (QuantMode == PERGROUP_DYNAMIC_QUANT) {
            QuantDynamicPerGroup(outLocal, inLocal, expertIndex, scalesGMTensor_);
        }
        #endif
    }

    __aicore__ inline void QuantStatic(LocalTensor<ExpandXOutType>& outLocal, LocalTensor<XType>& inLocal, uint32_t expertIndex, 
                                       uint32_t scalesCount_, GlobalTensor<float> &scalesGMTensor_)
    {
        Cast(floatLocalTemp_, inLocal, RoundMode::CAST_NONE, axisH_);
        if constexpr (Std::IsSame<ExpandXOutType, int8_t>::value) {
            if (scalesCount_ == 1) { // 所有专家共享scales，维度为(1,)
                DataCacheCleanAndInvalid<float, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(scalesGMTensor_);
                float scaleVal = scalesGMTensor_(0);
                Muls(floatLocalTemp_, floatLocalTemp_, scaleVal, axisH_);
            } else if (scalesCount_ == axisH_) { // 所有专家共享scales，维度为(h,)
                DataCopyPad(smoothScalesTensor_, scalesGMTensor_, scalesInParams_, scalesPadParams_);
            } else { // 所有专家不共享scales
                DataCopyPad(smoothScalesTensor_, scalesGMTensor_[expertIndex * axisH_], scalesInParams_, scalesPadParams_);
            }
            if (scalesCount_ != 1) {
                SyncFunc<AscendC::HardEvent::MTE2_V>();
                Mul(floatLocalTemp_, floatLocalTemp_, smoothScalesTensor_, axisH_);
            }

            LocalTensor<int32_t> tokenI32LT = floatLocalTemp_.ReinterpretCast<int32_t>();
            Cast(tokenI32LT, floatLocalTemp_, RoundMode::CAST_RINT, axisH_);
            LocalTensor<half> tokenF16LT = floatLocalTemp_.ReinterpretCast<half>();
            SetDeqScale((half)1.000000e+00f);
            Cast(tokenF16LT, tokenI32LT, RoundMode::CAST_ROUND, axisH_);
            Cast(outLocal, tokenF16LT, RoundMode::CAST_TRUNC, axisH_);
        }
        #if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
        else if constexpr (Std::IsSame<ExpandXOutType, hifloat8_t>::value) {
            DataCacheCleanAndInvalid<float, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(scalesGMTensor_);
            float scaleVal = scalesGMTensor_(0);
            Muls(floatLocalTemp_, floatLocalTemp_, scaleVal, axisH_);
            Cast(outLocal, floatLocalTemp_, RoundMode::CAST_ROUND, axisH_);
        }
        #endif
    }

    __aicore__ inline void QuantDynamicPerToken(LocalTensor<ExpandXOutType>& outLocal, LocalTensor<XType>& inLocal, 
                                                uint32_t expertIndex, GlobalTensor<float> &scalesGMTensor_)
    {
        float dynamicScale = 0.0;
        float maxVal = INT8_MAX_VALUE; // 获取输出类型的最大值（AscendC未提供相关接口）
        #if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
        if constexpr (Std::IsSame<ExpandXOutType, fp8_e5m2_t>::value) {
            maxVal = FP8_E5M2_MAX_VALUE;
        } else if constexpr (Std::IsSame<ExpandXOutType, fp8_e4m3fn_t>::value) {
            maxVal = FP8_E4M3_MAX_VALUE;
        }
        #endif
        Cast(floatLocalTemp_, inLocal, RoundMode::CAST_NONE, axisH_);
        PipeBarrier<PIPE_V>();
        if constexpr (IsSmoothScaleExist) { // 平滑系数
            SyncFunc<AscendC::HardEvent::V_MTE2>(); // ub复用，循环同步
            DataCopyPad(smoothScalesTensor_, scalesGMTensor_[expertIndex * axisH_], scalesInParams_, scalesPadParams_);
            SyncFunc<AscendC::HardEvent::MTE2_V>();
            Mul(floatLocalTemp_, floatLocalTemp_, smoothScalesTensor_, axisH_);
            PipeBarrier<PIPE_V>();
        }
        LocalTensor<float> floatLocalAbsTemp = smoothScalesBuf_.Get<float>();
        Abs(floatLocalAbsTemp, floatLocalTemp_, axisH_);
        PipeBarrier<PIPE_V>();
        ReduceMaxInplace(floatLocalAbsTemp, axisH_); // 获取最大值
        SyncFunc<AscendC::HardEvent::V_S>();
        dynamicScale = maxVal / floatLocalAbsTemp.GetValue(0);
        Muls(floatLocalTemp_, floatLocalTemp_, dynamicScale, axisH_);
        PipeBarrier<PIPE_V>();
        if constexpr (Std::IsSame<ExpandXOutType, int8_t>::value) {
            LocalTensor<half> halfLocalTemp = floatLocalTemp_.ReinterpretCast<half>();
            LocalTensor<int32_t> int32LocalTemp = floatLocalTemp_.ReinterpretCast<int32_t>();
            Cast(int32LocalTemp, floatLocalTemp_, RoundMode::CAST_RINT, axisH_);
            PipeBarrier<PIPE_V>();
            SetDeqScale((half)1.000000e+00f);
            PipeBarrier<PIPE_V>();
            Cast(halfLocalTemp, int32LocalTemp, RoundMode::CAST_ROUND, axisH_);
            PipeBarrier<PIPE_V>();
            Cast(outLocal, halfLocalTemp, RoundMode::CAST_TRUNC, axisH_);
        } 
        #if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
        else if constexpr (Std::IsSame<ExpandXOutType, fp8_e4m3fn_t>::value || 
            Std::IsSame<ExpandXOutType, fp8_e5m2_t>::value){
            Cast(outLocal, floatLocalTemp_, RoundMode::CAST_RINT, axisH_);
        }
        #endif
        LocalTensor<float> tokenF32Tmp = outLocal.template ReinterpretCast<float>();
        tokenF32Tmp.SetValue((Ceil(axisH_, UB_ALIGN) * UB_ALIGN) / sizeof(float), float(1.0) / dynamicScale); // int8->float32
        SyncFunc<AscendC::HardEvent::S_MTE3>();
    }

    #if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
    __aicore__ inline void QuantDynamicPerGroup(LocalTensor<ExpandXOutType>& outLocal, LocalTensor<XType>& inLocal, 
                                                uint32_t expertIndex, GlobalTensor<float> &scalesGMTensor_)
    {
        if constexpr (Std::IsSame<ExpandXOutType, fp8_e4m3fn_t>::value ||
            Std::IsSame<ExpandXOutType, fp8_e5m2_t>::value) {
            if constexpr (IsSmoothScaleExist) { // 平滑系数
                DataCopyPad(smoothScalesTensor_, scalesGMTensor_[expertIndex * axisH_], scalesInParams_, scalesPadParams_);
                SyncFunc<AscendC::HardEvent::MTE2_V>();
            }

            __ubuf__ XType* srcAddr = (__ubuf__ XType*)inLocal.GetPhyAddr();
            __ubuf__ float* smoothLocalAddr = (__ubuf__ float*)smoothScalesTensor_.GetPhyAddr();
            __ubuf__ int8_t* outLocalAddr = (__ubuf__ int8_t*)outLocal.GetPhyAddr();
            __ubuf__ float* scaleOutLocalAddr = (__ubuf__ float*)outLocal[Align128<uint32_t>(axisH_)].GetPhyAddr();

            quant::ComputePerTileDynamic<XType, ExpandXOutType, AscendC::RoundMode::CAST_RINT, IsSmoothScaleExist>(srcAddr,
                smoothLocalAddr, scaleOutLocalAddr, outLocalAddr, axisH_);
        }
    }

    __aicore__ inline void QuantDynamicMxFp8(LocalTensor<ExpandXOutType>& outLocal, LocalTensor<XType>& inLocal)
    {
        if constexpr (Std::IsSame<ExpandXOutType, fp8_e4m3fn_t>::value ||
            Std::IsSame<ExpandXOutType, fp8_e5m2_t>::value) {
            uint32_t mxScaleNum = Align2(Ceil32(axisH_));
            __ubuf__ XType* srcAddr = (__ubuf__ XType*)inLocal.GetPhyAddr();
            __ubuf__ uint16_t* maxExpAddr = (__ubuf__ uint16_t*)floatLocalTemp_.GetPhyAddr();
            __ubuf__ uint16_t* halfScaleLocalAddr = (__ubuf__ uint16_t*)floatLocalTemp_[Align32(mxScaleNum)].GetPhyAddr();
            __ubuf__ int8_t* outLocalAddr = (__ubuf__ int8_t*)outLocal.GetPhyAddr();
            __ubuf__ uint16_t* mxScaleLocalAddr = (__ubuf__ uint16_t*)outLocal[Align256<uint32_t>(axisH_)].GetPhyAddr();

            quant::ComputeMaxExp(srcAddr, maxExpAddr, axisH_); // 计算最大Exp
            quant::ComputeScale<ExpandXOutType>(maxExpAddr, mxScaleLocalAddr, halfScaleLocalAddr, mxScaleNum); // 计算scales并填充
            quant::ComputeData<XType, ExpandXOutType, AscendC::RoundMode::CAST_TRUNC, AscendC::RoundMode::CAST_RINT>(
                srcAddr, halfScaleLocalAddr, outLocalAddr, axisH_); // 计算量化后的expandx并填充
        }
    }
    #endif

    __aicore__ inline void CopyScalesToOut(uint32_t currentTokenIndex, uint32_t scaleOutBytes,
                                           LocalTensor<ExpandXOutType> &quantTok, DataCopyExtParams &scaleOutParams)
    {
        if constexpr (((QuantMode > UNQUANT) && (QuantMode != STATIC_QUANT)) ||
                    ((QuantMode == UNQUANT) && IsSmoothScaleExist)) {
            auto scaleLT = quantTok[(Ceil(axisH_, UB_ALIGN) * UB_ALIGN)].template ReinterpretCast<uint8_t>();
            #if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
            if constexpr (QuantMode == MX_QUANT) {
                scaleLT = quantTok[Align256<uint32_t>(axisH_)].template ReinterpretCast<uint8_t>();
            } else if constexpr (QuantMode == PERGROUP_DYNAMIC_QUANT) {
                scaleLT = quantTok[Align128<uint32_t>(axisH_)].template ReinterpretCast<uint8_t>();
            }
            #endif
            DataCopyPad(dynamicScalesOutGMTensor_[currentTokenIndex * scaleOutBytes], scaleLT, scaleOutParams);
        }
    }

    __aicore__ inline void ReduceMaxInplace(const LocalTensor<float>& srcLocal, uint32_t count)
    {
        uint64_t repsFp32 = count >> 6;        // 6 is count / elemPerRefFp32
        uint64_t offsetsFp32 = repsFp32 << 6;  // 6 is repsFp32 * elemPerRefFp32
        uint64_t remsFp32 = count & 0x3f;      // 0x3f 63, count % elemPerRefFp32
        const uint64_t elemPerRefFp32 = 64UL;  // 256 bit / sizeof(float)
        if (likely(repsFp32 > 1)) {
            // 8 is rep stride
            Max(srcLocal, srcLocal[elemPerRefFp32], srcLocal, elemPerRefFp32, repsFp32 - 1, {1, 1, 1, 0, 8, 0});
            PipeBarrier<PIPE_V>();
        }
        if (unlikely(remsFp32 > 0) && unlikely(offsetsFp32 > 0)) {
            Max(srcLocal, srcLocal[offsetsFp32], srcLocal, remsFp32, 1, {1, 1, 1, 0, 8, 0});
            PipeBarrier<PIPE_V>();
        }
        uint32_t mask = (repsFp32 > 0) ? elemPerRefFp32 : count;
        // 8 is rep stride
        WholeReduceMax(srcLocal, srcLocal, mask, 1, 8, 1, 8);
    }
};
}
#endif // MOE_DISTRIBUTE_DISPATCH_V2_QUANT_H