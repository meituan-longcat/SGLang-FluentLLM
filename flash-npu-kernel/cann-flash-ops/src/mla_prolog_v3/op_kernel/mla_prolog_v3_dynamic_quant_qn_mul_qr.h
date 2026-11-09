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
 * \file mla_prolog_v3_dynamic_quant_qn_mul_qr.h
 * \brief
 */

#ifndef MLA_PROLOG_V3_DYNAMIC_QUANT_QN_MUL_QR_H
#define MLA_PROLOG_V3_DYNAMIC_QUANT_QN_MUL_QR_H

#include "mla_prolog_v3_comm.h"
#include "mla_prolog_v3_vector_comm.h"

namespace MlaPrologV3 {

template <typename T, typename C, typename O>
__aicore__ inline void DynamicQuantMultiRow(const GlobalTensor<O>& outputGm,
                                            const LocalTensor<C>& scaleOutputLocal,
                                            const GlobalTensor<T>& inputGm,
                                            const LocalTensor<C> outputLocal,
                                            const LocalTensor<T>& inputHalf,
                                            const LocalTensor<C>& inputLocal,
                                            const LocalTensor<C>& maxInt8Tensor,
                                            const LocalTensor<uint8_t>& shareTmpUb,
                                            uint64_t row, uint64_t col,
                                            uint64_t subRow, uint64_t queryOutStride,
                                            uint32_t DYNAMIC_QUANT_INPUT_READY,
                                            uint32_t DYNAMIC_QUANT_OUTPUT_READY) {
    constexpr uint64_t inputBlockAlign = (ALIGN_BLOCK_SIZE / sizeof(T));
    constexpr uint64_t outputBlockAlign = (ALIGN_BLOCK_SIZE / sizeof(O));
    // DataCopyParams  : count len srcStrideIn dstStrideIn
    DataCopyParams outParams {(uint16_t)subRow,
                              (uint16_t)(col / outputBlockAlign), 0,
                              (uint16_t)((queryOutStride - col) / outputBlockAlign)};
    DataCopyParams inputParams {(uint16_t)subRow, (uint16_t)(col / inputBlockAlign),
                                (uint16_t)((queryOutStride - col) / inputBlockAlign), 0};

    uint32_t computeSize = subRow * col;
    uint64_t loopCnt = CeilDivT(row, subRow);
    uint64_t lastSubRow = row - (loopCnt - 1) * subRow;
    uint64_t inputGmOffset = 0;
    uint64_t scaleOffset = 0;

    for (uint64_t loopIdx = 0; loopIdx < loopCnt; loopIdx++) {
        if ((loopIdx == loopCnt - 1) && (lastSubRow != subRow)) {
            subRow = lastSubRow;
            computeSize = subRow * col;
            outParams.blockCount = subRow;
            inputParams.blockCount = subRow;
        }

        WaitFlag<HardEvent::V_MTE2>(DYNAMIC_QUANT_INPUT_READY); // 计算是否已经完成可以搬运
        DataCopy(inputHalf, inputGm[inputGmOffset], inputParams);
        SetFlag<HardEvent::MTE2_V>(DYNAMIC_QUANT_INPUT_READY);
        WaitFlag<HardEvent::MTE2_V>(DYNAMIC_QUANT_INPUT_READY); // 搬运是否已经完成可以计算

        Cast(inputLocal, inputHalf, RoundMode::CAST_NONE, computeSize);
        SetFlag<HardEvent::V_MTE2>(DYNAMIC_QUANT_INPUT_READY);
        pipe_barrier(PIPE_V);
        DynamicQuant(inputLocal, scaleOutputLocal[scaleOffset], inputLocal, maxInt8Tensor, shareTmpUb, subRow, col);
        pipe_barrier(PIPE_V);
        // float -> half -> int
        WaitFlag<HardEvent::MTE3_V>(DYNAMIC_QUANT_OUTPUT_READY); // 搬运是否已经完成可以计算
        LocalTensor<half> tmpf16Ub = outputLocal.template ReinterpretCast<half>();
        Cast(tmpf16Ub, inputLocal, AscendC::RoundMode::CAST_ROUND, computeSize);
        pipe_barrier(PIPE_V);
        // Calc: out = round(x / scaleOut)
        LocalTensor<O> tmpMMResCastTensor = outputLocal.template ReinterpretCast<O>();
        Cast(tmpMMResCastTensor, tmpf16Ub, AscendC::RoundMode::CAST_ROUND, computeSize);
        SetFlag<HardEvent::V_MTE3>(DYNAMIC_QUANT_OUTPUT_READY);
        WaitFlag<HardEvent::V_MTE3>(DYNAMIC_QUANT_OUTPUT_READY); // 计算是否已经完成可以搬运

        DataCopy(outputGm[inputGmOffset], tmpMMResCastTensor,  outParams);
        SetFlag<HardEvent::MTE3_V>(DYNAMIC_QUANT_OUTPUT_READY);
        inputGmOffset += subRow * queryOutStride;
        scaleOffset += subRow;
    }
}

template <typename T, typename C>
__aicore__ inline void MulQr(const GlobalTensor<T>& outputGmRope, const GlobalTensor<T>& inputGmRope,
                             LocalTensor<T> outputLocalRope,
                             const LocalTensor<T>& qrInputLocal, const LocalTensor<C>& qrFp32Local,
                             const LocalTensor<C>& reciprocalLocal, const LocalTensor<C>& dequantScaleBrcbLocal,
                             uint64_t row, uint64_t colRope, uint64_t subRowRope, uint64_t qrOutputStrideRope,
                             float quantScaleCkvRope, uint32_t MUL_QR_INPUT_COPY_READY, uint32_t MUL_QR) {
    constexpr uint64_t inputBlockAlign = (ALIGN_BLOCK_SIZE / sizeof(T));
    constexpr uint64_t computeBlockAlign = (ALIGN_BLOCK_SIZE / sizeof(C));

    uint64_t loopCntRope = CeilDivT(row, subRowRope);
    uint64_t lastSubRowRope = row - (loopCntRope - 1) * subRowRope;
    uint32_t computeSizeRope = subRowRope * colRope;

    // DataCopyParams  : count len srcStrideIn dstStrideIn
    DataCopyParams inputParamsRope {(uint16_t)row, (uint16_t)(colRope / inputBlockAlign),
        (uint16_t)((qrOutputStrideRope - colRope) / inputBlockAlign), 0};
    DataCopyParams outputParamsRope {(uint16_t)(subRowRope), (uint16_t)(colRope / inputBlockAlign),
        0, (uint16_t)((qrOutputStrideRope - colRope) / inputBlockAlign)};

    uint64_t inputGmRopeOffset = 0;
    uint64_t dequantScaleOffset = 0;
    uint64_t inputLocalRopeOffset = 0;


    WaitFlag<HardEvent::V_MTE2>(MUL_QR_INPUT_COPY_READY);
    DataCopy(qrInputLocal, inputGmRope, inputParamsRope);
    for (uint64_t loopIdx = 0; loopIdx < loopCntRope; loopIdx++) {
        if (loopIdx == (loopCntRope - 1) && subRowRope != lastSubRowRope) {
            subRowRope = lastSubRowRope;
            computeSizeRope = subRowRope * colRope;
            outputParamsRope.blockCount = subRowRope;
        }
        SetFlag<HardEvent::MTE3_V>(MUL_QR);
        WaitFlag<HardEvent::MTE3_V>(MUL_QR);

        SetFlag<HardEvent::MTE2_V>(MUL_QR);
        WaitFlag<HardEvent::MTE2_V>(MUL_QR);

        Cast(qrFp32Local, qrInputLocal[inputLocalRopeOffset], RoundMode::CAST_NONE, computeSizeRope);
        pipe_barrier(PIPE_V);
        Duplicate(reciprocalLocal, quantScaleCkvRope, subRowRope * computeBlockAlign);
        pipe_barrier(PIPE_V);
        // cal: quantScaleCkv / dequantScaleQn
        Div(reciprocalLocal, reciprocalLocal,
            dequantScaleBrcbLocal[dequantScaleOffset],
            subRowRope * computeBlockAlign);
        pipe_barrier(PIPE_V);
        // cal: x * quantScaleCkv / dequantScaleQn
        RowMuls(qrFp32Local, qrFp32Local,
                reciprocalLocal, Rectangle{(uint32_t)subRowRope,
                (uint32_t)colRope, (uint32_t)colRope});
        pipe_barrier(PIPE_V);
        Cast(outputLocalRope, qrFp32Local, RoundMode::CAST_RINT, computeSizeRope);
        pipe_barrier(PIPE_V);

        SetFlag<HardEvent::V_MTE3>(MUL_QR);
        WaitFlag<HardEvent::V_MTE3>(MUL_QR);

        DataCopy(outputGmRope[inputGmRopeOffset], outputLocalRope, outputParamsRope);
        inputLocalRopeOffset += subRowRope * colRope;
        inputGmRopeOffset += subRowRope * qrOutputStrideRope;
        dequantScaleOffset += subRowRope * computeBlockAlign;
    }
    SetFlag<HardEvent::V_MTE2>(MUL_QR_INPUT_COPY_READY);
}
/**
 * @brief DynamicQuantQnWithMulQr 流程中dynamicquant和ropeMul融合流程
          UB，流水，中间产物统筹规划
 * @tparam T mmQnOutputType -> bf16
 * @tparam C dequantScaleQNopeType -> float
 * @tparam O queryOutputType -> int8
 * @param scaleOutputGm 动态量化参数输出GM位置
 * @param outputGm 动态量化结果输出GM位置
 * @param outputGmRope rope处理后的输出位置
 * @param inputGm dynamicQuant部分的输入
 * @param shareTmpUb 临时buffer空间
 * @param row
 * @param col
 * @param scaleOutStride 描述动态量化参数输出GM位置的偏移关系
 * @param queryOutStride 描述动态量化结果输出GM位置
 * @param inputGmRope
 * @param quantScaleCkvRope
 * @param colRope
 * @param qrOutputStrideRope 描述rope处理后的输出位置
 */
template <typename T, typename C, typename O>
__aicore__ inline void DynamicQuantQnWithMulQr(
    // Dynamic Quant With MulQr 输出
    const GlobalTensor<C>& scaleOutputGm, const GlobalTensor<O>& outputGm, const GlobalTensor<T>& outputGmRope,
    // Dynamic Quant 入参
    const GlobalTensor<T>& inputGm, LocalTensor<uint8_t>& shareTmpUb, uint64_t row, uint64_t col,
    uint64_t scaleOutStride, uint64_t queryOutStride,
    // Mul Qr 入参
    const GlobalTensor<T>& inputGmRope, float quantScaleCkvRope, uint64_t colRope, uint64_t qrOutputStrideRope) {
    // 常量
    // MUL_QR: 用于控制Mul_Qr的同步
    // DYNAMIC_QUANT_INPUT_READY: dynamicquant输入的计算/搬运 是否已经完成可以开始下一轮的 搬运/计算
    // MUL_QR_INPUT_COPY_READY: 是否可以开始下一轮的MUL_QR_INPUT_COPY
    // CALC_SCALE_FINISH: dynamicquant的scale是否计算完成可以开始搬运
    // DYNAMIC_QUANT_OUTPUT_READY: dynamicquant输出的计算/搬运 是否已经完成可以开始下一轮的 搬运/计算
    constexpr uint32_t MUL_QR = EVENT_ID1;
    constexpr uint32_t DYNAMIC_QUANT_INPUT_READY = EVENT_ID0;
    constexpr uint32_t MUL_QR_INPUT_COPY_READY = EVENT_ID3;
    constexpr uint32_t CALC_SCALE_FINISH = EVENT_ID0;
    constexpr uint32_t DYNAMIC_QUANT_OUTPUT_READY = EVENT_ID3;

    constexpr uint64_t computeBlockAlign = (ALIGN_BLOCK_SIZE / sizeof(C));
    constexpr uint64_t inputBlockAlign = (ALIGN_BLOCK_SIZE / sizeof(T));
    constexpr float maxInt8 = 127.0;
    // Dynamic Quant 局部变量
    uint64_t rowStepSize = 16; // 单次处理最大行数
    uint64_t subRow = row < rowStepSize ? row : rowStepSize;
    uint32_t computeSize = subRow * col;
    uint32_t brcbCnt = Align(subRow, computeBlockAlign);

    // Rope Post Process 局部变量
    uint64_t rowStepSizeRope = 64; // 单次处理最大行数
    uint64_t subRowRope = row < rowStepSizeRope ? row : rowStepSizeRope;
    uint32_t computeSizeRope = subRowRope * colRope;
    uint32_t loadRopeGmSize = row * colRope;

    // Dynamic Quant & Rope Post Process UB Buffer分配
    LocalTensor<T> inputHalf = shareTmpUb.ReinterpretCast<T>();
    LocalTensor<T> qrInputLocal = inputHalf[computeSize];
    LocalTensor<T> outputLocalRope = qrInputLocal[loadRopeGmSize + inputBlockAlign];

    LocalTensor<C> inputLocal = outputLocalRope[computeSizeRope + inputBlockAlign].template ReinterpretCast<C>();
    LocalTensor<C> outputLocal = inputLocal[computeSize];
    LocalTensor<C> maxInt8Tensor = outputLocal[computeSize];
    LocalTensor<C> dynamicQuantUb = maxInt8Tensor[brcbCnt];
    LocalTensor<C> scaleOutputLocal = dynamicQuantUb[computeSize + brcbCnt * computeBlockAlign];
    LocalTensor<C> scaleBrcb = scaleOutputLocal[Align(row, computeBlockAlign)];

    // Rope Post Process流程在Dynamic Quant流程结束，两者UB Buffer不会同时使用，故可以从起始位置开始重新计算，减少UB Buffer使用
    LocalTensor<C> qrFp32Local = inputLocal;
    LocalTensor<C> reciprocalLocal = qrFp32Local[computeSizeRope + computeBlockAlign];

    // Dynamic Quant
    Duplicate(maxInt8Tensor, static_cast<C>(maxInt8), brcbCnt);
    pipe_barrier(PIPE_V);

    DynamicQuantMultiRow(outputGm, scaleOutputLocal, inputGm, outputLocal, inputHalf, inputLocal, maxInt8Tensor,
                        dynamicQuantUb.template ReinterpretCast<uint8_t>(), row, col, subRow,
                        queryOutStride, DYNAMIC_QUANT_INPUT_READY, DYNAMIC_QUANT_OUTPUT_READY);

    Brcb(scaleBrcb, scaleOutputLocal, CeilDivT(row, computeBlockAlign), {1, computeBlockAlign});
    pipe_barrier(PIPE_V);

    // DataCopyParams  : count len srcStrideIn dstStrideIn
    DataCopyParams scaleOutCopyParams {(uint16_t)row, (uint16_t)sizeof(C), 0,
                                       (uint16_t)((scaleOutStride - 1) * sizeof(C))};

    SetFlag<HardEvent::V_MTE3>(CALC_SCALE_FINISH);
    WaitFlag<HardEvent::V_MTE3>(CALC_SCALE_FINISH);
    DataCopyPad(scaleOutputGm, scaleBrcb, scaleOutCopyParams);

    // qr rope 后的乘法
    MulQr(outputGmRope, inputGmRope,  outputLocalRope, qrInputLocal,
        qrFp32Local, reciprocalLocal, scaleBrcb, row, colRope,
        subRowRope, qrOutputStrideRope, quantScaleCkvRope, MUL_QR_INPUT_COPY_READY, MUL_QR);

    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
}

}

#endif