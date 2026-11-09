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
 * \file moe_distribute_v2_quant.h
 * \brief
 */

#ifndef MOE_DISTRIBUTE_COMBINE_V2_QUANT_H
#define MOE_DISTRIBUTE_COMBINE_V2_QUANT_H

#if __has_include("../moe_distribute_dispatch_v2/check_winsize.h")
#include "../moe_distribute_dispatch_v2/moe_distribute_v2_constant.h"
#include "../moe_distribute_dispatch_v2/moe_distribute_v2_base.h"
#else
#include "../../moe_distribute_dispatch_v2/op_kernel/moe_distribute_v2_constant.h"
#include "../../moe_distribute_dispatch_v2/op_kernel/moe_distribute_v2_base.h"
#endif

namespace Mc2Kernel {
using namespace AscendC;
using namespace MoeDistributeV2Base;

template <typename ExpandXType, typename XType, typename ExpandIdxType, bool IsNeedReduceScatter, bool IsInt8Quant, bool HasAddRmsNorm>
class MoeDistributeCombineQuant{
public:
    float scaleValFloat_;
    uint32_t axisH_{0};
    uint32_t mask_{0};
    uint32_t repeatNum_{0};
    uint32_t hAlign32Size_{0};
    uint32_t quantScaleNum_{0};
    LocalTensor<int8_t> castLocalTensor_;
    LocalTensor<XType> scaleDivTensor_;

    __aicore__ inline MoeDistributeCombineQuant() = default;

    __aicore__ inline void SetQuantInitParams(uint32_t axisH) 
    {
        axisH_ = axisH;
    }

    __aicore__ inline void InitInt8Quant(uint32_t &scaleNum_, uint32_t &hExpandXAlign32Size_, 
                                         uint32_t &hFloatAlign256Size_, uint32_t &tokenScaleCnt_)
    {
        hAlign32Size_ = Ceil(axisH_, UB_ALIGN) * UB_ALIGN;
        scaleValFloat_ = static_cast<float>(1.0f / SCALE_PARAM);
        uint32_t scaleGranu = static_cast<uint32_t>(UB_ALIGN / sizeof(float)); // 计算每个block得到的reducemax结果数量
        quantScaleNum_ = (hExpandXAlign32Size_ / sizeof(ExpandXType)) / scaleGranu; // 得到有效scale的个数
        scaleNum_ = quantScaleNum_;
        repeatNum_ = static_cast<uint32_t>(hFloatAlign256Size_ / ALIGNED_LEN); // BlockReduceMax 与 Brcb的重复迭代次数，每次256b参与计算
        mask_ = static_cast<uint32_t>(ALIGNED_LEN / sizeof(float));
        tokenScaleCnt_ = hAlign32Size_ / sizeof(ExpandXType) + quantScaleNum_; // int8_align + scale有效个数
    }

    __aicore__ inline void Int8QuantProcess(LocalTensor<XType> &sendLocalTensor_, LocalTensor<float> &winTpSendCountFloatTensor_, LocalTensor<ExpandXType> &gmTpSendCountTensor_, 
                                            LocalTensor<half> &fp16CastTensor_, LocalTensor<float> &absFloatTensor_, LocalTensor<float> reduceMaxFloatTensor_, LocalTensor<float> scaleDupLocalTensor_)
    {
        SyncFunc<AscendC::HardEvent::MTE2_V>();
        castLocalTensor_ = sendLocalTensor_.template ReinterpretCast<int8_t>(); // 长度为int8H_Align + scaleNum
        scaleDivTensor_ = castLocalTensor_[hAlign32Size_].template ReinterpretCast<XType>(); // 偏移前面的int8

        Cast(winTpSendCountFloatTensor_, gmTpSendCountTensor_, RoundMode::CAST_NONE, axisH_);
        PipeBarrier<PIPE_V>();
        Abs(absFloatTensor_, winTpSendCountFloatTensor_, axisH_); // absFloatTensor_ align到256并写0，支持ReduceMax与Brcb
        PipeBarrier<PIPE_V>();
        BlockReduceMax(reduceMaxFloatTensor_, absFloatTensor_, repeatNum_, mask_, 1, 1, BLOCK_NUM); // 32->1 256->8
        PipeBarrier<PIPE_V>();
        Muls(reduceMaxFloatTensor_, reduceMaxFloatTensor_, scaleValFloat_, quantScaleNum_); // 有效个数
        PipeBarrier<PIPE_V>();
        Cast(scaleDivTensor_, reduceMaxFloatTensor_, RoundMode::CAST_RINT, quantScaleNum_); // 有效个数
        PipeBarrier<PIPE_V>();
        Brcb(scaleDupLocalTensor_, reduceMaxFloatTensor_, repeatNum_, {1, BLOCK_NUM}); // 一次256
        PipeBarrier<PIPE_V>();
        Div(winTpSendCountFloatTensor_, winTpSendCountFloatTensor_, scaleDupLocalTensor_, axisH_); // 有效个数
        PipeBarrier<PIPE_V>();
        Cast(fp16CastTensor_, winTpSendCountFloatTensor_, RoundMode::CAST_RINT, axisH_);
        PipeBarrier<PIPE_V>();
        Cast(castLocalTensor_, fp16CastTensor_, RoundMode::CAST_RINT, axisH_);
        SyncFunc<AscendC::HardEvent::V_MTE3>();
    }

    __aicore__ inline void Int8DequantProcess(LocalTensor<XType>& src, LocalTensor<float> scaleDivFloatTensor_, LocalTensor<half> &fp16CastTensor_, 
                                              LocalTensor<float> &absFloatTensor_, LocalTensor<float> scaleDupLocalTensor_)
    {
        SyncFunc<AscendC::HardEvent::MTE2_V>();
        castLocalTensor_ = src.template ReinterpretCast<int8_t>();
        scaleDivTensor_ = src[hAlign32Size_ / INT8_DIVIVE];

        SyncFunc<AscendC::HardEvent::S_V>();
        Cast(scaleDivFloatTensor_, scaleDivTensor_, RoundMode::CAST_NONE, quantScaleNum_);
        Cast(fp16CastTensor_, castLocalTensor_, RoundMode::CAST_NONE, axisH_);
        PipeBarrier<PIPE_V>();
        Cast(absFloatTensor_, fp16CastTensor_, RoundMode::CAST_NONE, axisH_);
        Brcb(scaleDupLocalTensor_, scaleDivFloatTensor_, repeatNum_, {1, BLOCK_NUM});
        PipeBarrier<PIPE_V>();
        Mul(absFloatTensor_, absFloatTensor_, scaleDupLocalTensor_, axisH_);
        PipeBarrier<PIPE_V>();
        Cast(src, absFloatTensor_, RoundMode::CAST_RINT, axisH_);
        PipeBarrier<PIPE_V>();
    }
};
}
#endif // MOE_DISTRIBUTE_V2_QUANT_H