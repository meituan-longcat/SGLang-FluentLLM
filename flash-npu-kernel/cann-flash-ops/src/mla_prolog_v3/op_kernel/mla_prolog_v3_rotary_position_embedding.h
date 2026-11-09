/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file mla_prolog_v3_rotary_position_embedding.h
 * \brief
 */

#ifndef MLA_PROLOG_V3_ROTARY_POSITION_EMBEDDING_H
#define MLA_PROLOG_V3_ROTARY_POSITION_EMBEDDING_H

#include "mla_prolog_v3_comm.h"
#include "mla_prolog_v3_vector_comm.h"

namespace MlaPrologV3 {


/**
 * @brief RotaryPosEmbPerTensor 对一个tensor进行RotartPosEmb，tensor的维度为[row * col]
          行与行之间sin/cos公用；
          C：ropeComputType, float
 * @param outputLocal 输出tensor
 * @param inputGm 输入tensor
 * @param cosLocal cos系数
 * @param sinLocal sin系数
 * @param shareTmpUb 临时buffer，需要大小为 cnt * 5 * sizeof(float)
 * @param ropeParams 描述待处理数据的排布，包括
          row 行数
          col 列数
          stride 一行的真实长度
 * @param channelDeqScaleGm 量化参数；该tensor的每个元素不同
 * @param scale 量化参数；该tensor共用
 */
template <typename T, typename C, typename O>
__aicore__ inline void RotaryPosEmbPerTensor(LocalTensor<O>& outputLocal,
                                    const GlobalTensor<T>& inputGm, const LocalTensor<C>& cosLocal,
                                    const LocalTensor<C>& sinLocal, LocalTensor<uint8_t>& shareTmpUb,
                                    Rectangle ropeParams,
                                    LocalTensor<float> channelDeqScaleLocal = LocalTensor<float>(),
                                    LocalTensor<float> scale = LocalTensor<float>()) {

    int64_t cnt = ropeParams.row * ropeParams.col;
    LocalTensor<T> kLocal = shareTmpUb.ReinterpretCast<T>();
    DataCopyExtParams copyParams{
        static_cast<uint16_t>(ropeParams.row),
        static_cast<uint32_t>(ropeParams.col * sizeof(T)),
        static_cast<uint32_t>((ropeParams.stride - ropeParams.col) * sizeof(T)),
        0, 0};
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    DataCopyPad(kLocal, inputGm, copyParams, padParams);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    int64_t baseOffset;
    if constexpr (std::is_same<T, int32_t>::value) {
        baseOffset = cnt;
    } else {
        // C是ropeComputType始终是float类型，如果T是bf16类型其偏移只用float一半即可
        baseOffset = cnt >> 1;
    }
    LocalTensor<C> kFp32Local = shareTmpUb.ReinterpretCast<C>()[baseOffset];
    LocalTensor<C> ropeShareUB = shareTmpUb.ReinterpretCast<C>()[baseOffset + cnt];
    // RotaryPosEmb内部需要 2*cnt
    LocalTensor<C> kFp32OutputLocal = shareTmpUb.ReinterpretCast<C>()[baseOffset + cnt * 3];
    uint64_t rsvdCnt = 0;

    if constexpr (std::is_same<T, int32_t>::value) { // 反量化
        Rectangle rectangleParams {
            (uint32_t)1,   // row
            (uint32_t)cnt, // col
            (uint32_t)cnt  // columnStride
        };
        Dequant(kFp32Local, kLocal, channelDeqScaleLocal, scale, rectangleParams);
        pipe_barrier(PIPE_V);
    } else {
        Cast(kFp32Local, kLocal, RoundMode::CAST_NONE, cnt);
        pipe_barrier(PIPE_V);
    }
    if constexpr (std::is_same<O, C>::value) {
        RotaryPosEmb(outputLocal, kFp32Local, cosLocal, sinLocal,
                    ropeShareUB.template ReinterpretCast<uint8_t>(),
                    ropeParams.row, ropeParams.col, 0);
        PipeBarrier<PIPE_V>();
    } else {
        RotaryPosEmb(kFp32OutputLocal, kFp32Local, cosLocal, sinLocal,
                    ropeShareUB.template ReinterpretCast<uint8_t>(),
                    ropeParams.row, ropeParams.col, 0);
        pipe_barrier(PIPE_V);
        Cast(outputLocal, kFp32OutputLocal, RoundMode::CAST_RINT, cnt);
        PipeBarrier<PIPE_V>();
    }
}

/**
 * @brief RotaryPosEmbPerHead 进行row行col列的RotaryPosEmb
          每行的量化系数，sin/cos均不同
          C：ropeComputType，float
 * @param outputLocal 输出tensor
 * @param inputGm 输入tensor
 * @param cosLocal cos系数
 * @param sinLocal sin系数
 * @param shareTmpUb 临时buffer
 * @param ropeParams 描述待处理数据的排布，包括
          row 行数
          col 列数
          stride 一行的真实长度
 * @param strideScale 一段的真实长度，描述channelDeqScaledGm数据排布
 * @param channelDeqScaleGm 量化参数：最终使用shape[1,col]
 * @param deQuantScale 量化参数；最终使用shape[row,8]
 */
template <typename T, typename C, typename O>
__aicore__ inline void RotaryPosEmbPerHead(LocalTensor<O>& outputLocal,
                                    const GlobalTensor<T>& inputGm,
                                    const LocalTensor<C>& cosLocal,
                                    const LocalTensor<C>& sinLocal,
                                    LocalTensor<uint8_t>& shareTmpUb,
                                    Rectangle ropeParams,
                                    int64_t strideScale,
                                    GlobalTensor<float> channelDeqScaleGm = GlobalTensor<float>(),
                                    LocalTensor<float> deQuantScale = LocalTensor<float>()) {
    // 在 BS = 1 场景可能存在有row为零的情况，提前返回减少运算
    if (ropeParams.row == 0) {
        return;
    }

    int64_t cnt = ropeParams.row * ropeParams.col;
    LocalTensor<T> kLocal = shareTmpUb.ReinterpretCast<T>();

    // blockCount blockLen srcStride dstStride rsc
    DataCopyExtParams copyParams{static_cast<uint16_t>(ropeParams.row),
                                static_cast<uint32_t>(ropeParams.col * sizeof(T)),
                                static_cast<uint32_t>((ropeParams.stride - ropeParams.col) * sizeof(T)), 0, 0};
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    DataCopyPad(kLocal, inputGm, copyParams, padParams);
    int64_t baseOffset;
    if constexpr (std::is_same<T, int32_t>::value) {
        baseOffset = cnt;
    } else {
        // C是ropeComputType始终是float类型，如果T是bf16类型其偏移只用float一半即可。
        baseOffset = cnt >> 1;
    }
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    LocalTensor<C> kFp32Local = shareTmpUb.ReinterpretCast<C>()[baseOffset];
    // scale参数可以和rope使用的空间复用
    LocalTensor<C> ropeShareUB = shareTmpUb.ReinterpretCast<C>()[baseOffset + cnt];
    LocalTensor<C> scaleLocal = ropeShareUB;
    // RotaryPosEmb内部需要 2*cnt
    LocalTensor<C> kFp32OutputLocal = shareTmpUb.ReinterpretCast<C>()[baseOffset + cnt * 3];
    uint64_t rsvdCnt = 0;
    if constexpr (std::is_same<T, int32_t>::value) { // 反量化
        DataCopyExtParams copyParams1{static_cast<uint16_t>(1), static_cast<uint32_t>(ropeParams.col * sizeof(C)),
                                      static_cast<uint32_t>((strideScale - ropeParams.col) * sizeof(C)), 0, 0};
        DataCopyPadExtParams<C> padParams1{false, 0, 0, 0};
        DataCopyPad(scaleLocal, channelDeqScaleGm, copyParams1, padParams1); // 复用内存
        SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);

        uint8_t blockNumPerRow = ropeParams.col / (ALIGN_BLOCK_SIZE / sizeof(C));
        // row  col stride
        Rectangle rectangleParams {(uint32_t)ropeParams.row,  (uint32_t)ropeParams.col, (uint32_t)ropeParams.col};
        Dequant(kFp32Local, kLocal, scaleLocal, deQuantScale, rectangleParams);
        pipe_barrier(PIPE_V);
    } else {
        Cast(kFp32Local, kLocal, RoundMode::CAST_NONE, cnt);
        pipe_barrier(PIPE_V);
    }
    LocalTensor<C> kFp32OutputLocalSinTmp = shareTmpUb.ReinterpretCast<C>()[baseOffset + cnt * 2];
    RotaryPosEmb(kFp32OutputLocal, kFp32Local, cosLocal, sinLocal,
                ropeShareUB.template ReinterpretCast<uint8_t>(),
                ropeParams.row, ropeParams.col, ropeParams.col);
    PipeBarrier<PIPE_V>();

    if constexpr (std::is_same<O, C>::value) {
        DataCopy(outputLocal, kFp32OutputLocal, cnt);
        PipeBarrier<PIPE_V>();
    } else {
        Cast(outputLocal, kFp32OutputLocal, RoundMode::CAST_RINT, cnt);
        PipeBarrier<PIPE_V>();
    }
}

template <typename T, typename O>
__aicore__ inline void RopePostQuantPerChannel(LocalTensor<O> &outputLocal,
                                            LocalTensor<T> &inputLocal, LocalTensor<float> &quantScaleLocal,
                                            LocalTensor<uint8_t> &shareTmpUb, int64_t cnt) {
    LocalTensor<float> inFp32;
    if constexpr (std::is_same<T, float>::value) {
        inFp32 = inputLocal;
    } else {
        inFp32 = shareTmpUb.ReinterpretCast<float>()[cnt];
        Cast(inFp32, inputLocal, RoundMode::CAST_NONE, cnt);
        pipe_barrier(PIPE_V);
    }
    Mul(inFp32, inFp32, quantScaleLocal, cnt);
    pipe_barrier(PIPE_V);
    CastFP32ToINT8(outputLocal, inFp32, shareTmpUb, cnt);
    pipe_barrier(PIPE_V);
}

}

#endif