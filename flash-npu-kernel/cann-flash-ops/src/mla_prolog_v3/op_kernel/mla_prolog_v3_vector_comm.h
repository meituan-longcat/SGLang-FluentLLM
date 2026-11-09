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
  * \file mla_prolog_comm.h
  * \brief 存放各种vector的公共组件
  */

#include "mla_prolog_v3_policy.h"

#ifndef MLA_PROLOG_V3_VECTOR_COMM_H
#define MLA_PROLOG_V3_VECTOR_COMM_H

#include "mla_prolog_v3_comm.h"
namespace MlaPrologV3 {

struct Rectangle {
    uint32_t row;
    uint32_t col;
    uint32_t stride;
};

struct RmsNormParam {
    float reciprocal;
    float epsilon;
    uint32_t row;
    uint32_t col;
    float scale;
#if !MLA_PROLOG_V3_SFA1
    uint16_t isScaleEnable;
#endif
};

struct PerTileQuantParams {
    int32_t tileSize;
    int32_t tileNum;
    float alpha;
    uint32_t row;
    uint32_t col;
};

/**
 * @brief vec mul by row;  dstUb[i, j] = src0Ub[j] * src1Ub[i, j],
 * @param dstUb 输出tensor [row, columnStride]
 * @param src0Ub 输入tensor [1, columnStride]
 * @param src1Ub 输入tensor src1Ub:[row, col]
 * @param rectangleParams 描述待处理数据的排布，包括
          row 行数
          col 列数
          stride 一行的真实长度
 */
__aicore__ inline void VecMulMat(LocalTensor<float> dstUb, LocalTensor<float> src0Ub, LocalTensor<float> src1Ub,
                                 const Rectangle& rectangleParams)
{
    uint32_t mask = FP32_REPEAT_ELEMENT_NUM;
    uint32_t columnLoopCount = rectangleParams.col / mask;
    uint32_t remainCount = rectangleParams.col % mask;
    // 选择迭代较少的方式
    // dstRepStride为0~255, columnCount需要小于2048
    if (columnLoopCount < rectangleParams.row &&
        rectangleParams.stride < REPEAT_STRIDE_UP_BOUND * FP32_BLOCK_ELEMENT_NUM) {
        BinaryRepeatParams repeatParams;
        repeatParams.dstBlkStride = 1;
        repeatParams.src0BlkStride = 1;
        repeatParams.src1BlkStride = 1;
        repeatParams.dstRepStride = rectangleParams.stride / FP32_BLOCK_ELEMENT_NUM;
        repeatParams.src0RepStride = 0;
        repeatParams.src1RepStride = rectangleParams.stride / FP32_BLOCK_ELEMENT_NUM;
        uint32_t offset = 0;
        for (int i = 0; i < columnLoopCount; i++) {
            // 偏移 offset : i * mask
            Mul(dstUb[offset], src0Ub[offset], src1Ub[offset], mask, rectangleParams.row, repeatParams);
            offset += mask;
        }
        if (remainCount > 0) {
            // 偏移 offset : columnLoopCount * mask
            Mul(dstUb[offset], src0Ub[offset], src1Ub[offset], remainCount, rectangleParams.row, repeatParams);
        }
    } else {
        uint32_t offset = 0;
        for (int i = 0; i < rectangleParams.row; i++) {
            Mul(dstUb[offset], src0Ub, src1Ub[offset], rectangleParams.col);
            offset += rectangleParams.stride;
        }
    }
}

/**
 * @brief RowMuls muls by row, 每行的元素乘以相同的元素，该元素需要扩展到一个数据块；
 *        dstUb[i, (j * 8) : (j * 8 + 7)] = src0Ub[i, (j * 8) : (j * 8 + 7)] * src1Ub[i, 0 : 7]
 * @param dstUb 输出tensor [row, columnStride]
 * @param src0Ub 输入tensor [row, columnStride]
 * @param src1Ub 输入tensor [row, FP32_BLOCK_ELEMENT_NUM]
 * @param rectangleParams 描述待处理数据的排布，包括
          row 行数
          col 列数
          stride 一行的真实长度
 */
template <typename T>
__aicore__ inline void RowMuls(LocalTensor<T> dstUb, LocalTensor<T> src0Ub, LocalTensor<T> src1Ub,
                               const Rectangle& rectangleParams)
{
    uint32_t repeatElementNum = FP32_REPEAT_ELEMENT_NUM;
    uint32_t blockElementNum = FP32_BLOCK_ELEMENT_NUM;

    if constexpr (std::is_same<T, half>::value) {
        // 此限制由于每个repeat至多连续读取256B数据
        repeatElementNum = FP32_REPEAT_ELEMENT_NUM * 2; // 256/4 * 2 = 128
        blockElementNum = FP32_BLOCK_ELEMENT_NUM * 2; // 32/4 * 2 = 16
    }

    // 每次只能连续读取256B的数据进行计算，故每次只能处理256B/sizeof(dType)=
    // 列方向分dLoop次，每次处理8列数据
    uint32_t dLoop = rectangleParams.col / repeatElementNum;
    uint32_t dRemain = rectangleParams.col % repeatElementNum;
    // REPEAT_STRIDE_UP_BOUND=256，此限制由于src0RepStride数据类型为uint8至多256个datablock间距
    if (rectangleParams.stride < REPEAT_STRIDE_UP_BOUND * blockElementNum) {
        // dstBlkStrideIn src0BlkStrideIn  src1BlkStrideIn dstRepStrideIn src0RepStrideIn src1RepStrideIn
        BinaryRepeatParams repeatParams{1, 1, 0, (uint8_t)(rectangleParams.stride / blockElementNum),
                                        (uint8_t)(rectangleParams.stride / blockElementNum), 1};

        // 如果以列为repeat所处理的次数小于行处理次数，则以列方式处理。反之则以行进行repeat处理
        if (dLoop <= rectangleParams.row) {
            uint32_t offset = 0;
            for (uint32_t i = 0; i < dLoop; i++) {
                Mul(dstUb[offset], src0Ub[offset], src1Ub,
                repeatElementNum, rectangleParams.row, repeatParams);
                offset += repeatElementNum;
            }
        } else {
            repeatParams.src0RepStride = 8; // 列方向上两次repeat起始地址间隔dtypeMask=64个元素，即8个block
            repeatParams.src1RepStride = 0;
            repeatParams.dstRepStride = 8; // 列方向上两次repeat起始地址间隔dtypeMask=64个元素，即8个block
            for (uint32_t i = 0; i < rectangleParams.row; i++) {
                Mul(dstUb[i * rectangleParams.stride], src0Ub[i * rectangleParams.stride], src1Ub[i * blockElementNum],
                    repeatElementNum, dLoop, repeatParams);
            }
        }

        // 最后一次完成[row, dRemain] * [row, blockElementNum] 只计算有效部分
        if (dRemain > 0) {
            Mul(dstUb[dLoop * repeatElementNum], src0Ub[dLoop * repeatElementNum],
                src1Ub, dRemain, rectangleParams.row,
                repeatParams);
        }
    } else {
        // dstBlkStrideIn src0BlkStrideIn  src1BlkStrideIn dstRepStrideIn src0RepStrideIn src1RepStrideIn
        // 8 : 每个repeat为256B数据，正好8个datablock
        BinaryRepeatParams repeatParams{1, 1, 0, 8, 8, 0};

        // 每次计算一行，共计算dealRowCount行
        for (uint32_t i = 0; i < rectangleParams.row; i++) {
            // 计算一行中的dLoop个repeat，每个repeat计算256/block_size个data_block
            Mul(dstUb[i * rectangleParams.stride], src0Ub[i * rectangleParams.stride], src1Ub[i * blockElementNum],
                repeatElementNum, dLoop, repeatParams);
            // 计算一行中的尾块
            if (dRemain > 0) {
                Mul(dstUb[i * rectangleParams.stride + dLoop * repeatElementNum],
                    src0Ub[i * rectangleParams.stride + dLoop * repeatElementNum],
                    src1Ub[i * blockElementNum], dRemain, 1, repeatParams);
            }
        }
    }
}

/**
 * @brief RowMax max by row, 按行求最大值
 * @param dstUb 输出tensor [row, 1]
                dstUb[i] = max(srcUb[i, :])
 * @param srcUb 输入tensor [row, columnStride]
 * @param rectangleParams 描述待处理数据的排布，包括
          row 行数
          col 列数
          stride 一行的真实长度
 */
__aicore__ inline void RowMax(LocalTensor<float> &dstUb, LocalTensor<float> &srcUb, const Rectangle rectangleParams)
{
    uint32_t dtypeMask = FP32_REPEAT_ELEMENT_NUM;
    uint32_t blockCount = rectangleParams.col / dtypeMask;
    uint32_t remain = rectangleParams.col % dtypeMask;

    BinaryRepeatParams repeatParamsMax;
    repeatParamsMax.src0BlkStride = 1;
    repeatParamsMax.src1BlkStride = 1;
    repeatParamsMax.dstBlkStride = 1;
    repeatParamsMax.src0RepStride = rectangleParams.stride / FP32_BLOCK_ELEMENT_NUM;
    repeatParamsMax.src1RepStride = rectangleParams.stride / FP32_BLOCK_ELEMENT_NUM;
    repeatParamsMax.dstRepStride = rectangleParams.stride / FP32_BLOCK_ELEMENT_NUM;
    if (blockCount > 0 && remain > 0) {
        Max(srcUb, srcUb, srcUb[blockCount * dtypeMask], remain, rectangleParams.row, repeatParamsMax);
        pipe_barrier(PIPE_V);
    }

    for (uint32_t columnLoopCount = blockCount >> 1; columnLoopCount > 0; columnLoopCount = blockCount >> 1) {
        blockCount = (blockCount + 1) >> 1; // 2: 每次处理2个block
        for (uint32_t j = 0; j < columnLoopCount; j++) {
            Max(srcUb[j * dtypeMask], srcUb[j * dtypeMask], srcUb[(j + blockCount) * dtypeMask], dtypeMask,
                rectangleParams.row, repeatParamsMax);
        }
        pipe_barrier(PIPE_V);
    }

    WholeReduceMax(dstUb, srcUb,
                   (rectangleParams.col < dtypeMask) ? rectangleParams.col : dtypeMask,
                   rectangleParams.row, 1, 1,
                   rectangleParams.stride / FP32_BLOCK_ELEMENT_NUM, ReduceOrder::ORDER_ONLY_VALUE);
}

/**
 * @brief Dequant 对[row * col]的tensor进行反量化。需要输入一个行向量和列向量，共同构成反量化的矩阵。
          outputLocal [i,j] = inputLocal[i,j] * scaleLocal[j] * scale2Local [i]
 * @param outputLocal 输出tensor [row , col]
 * @param inputLocal 输入tensor [row , col]
 * @param scaleLocal [1,col] 量化系数；行向量
 * @param scale2Local [row,8] 量化系数；列向量，8为float扩充为32Bytes
 * @param rectangleParams 描述待处理数据的排布，包括
          row 行数
          col 列数
          stride 一行的真实长度
 */
__aicore__ inline void Dequant(const LocalTensor<float> &outputLocal,
                               const LocalTensor<int32_t> &inputLocal,
                               const LocalTensor<float> &scaleLocal,
                               const LocalTensor<float> &scale2Local,
                               const Rectangle& rectangleParams)
{
    uint64_t cnt = rectangleParams.col * rectangleParams.row;
    Cast(outputLocal, inputLocal, RoundMode::CAST_RINT, cnt);
    PipeBarrier<PIPE_V>();
    RowMuls(outputLocal, outputLocal, scale2Local, rectangleParams);
    PipeBarrier<PIPE_V>();
    VecMulMat(outputLocal, scaleLocal, outputLocal, rectangleParams);
}

/**
 * @brief CastFP32ToINT8 将float类型cast为int8，路径为float--------->int32--------->half---------->int8
                                           CAST_RINT     CAST_ROUND     CAST_TRUNC
 * @param outputLocal 输出tensor [cnt]
 * @param inputLocal 输入tensor [cnt]
 * @param shareTmpUb 临时buffer 内部需要的空间为 [cnt * 4] 4 : sizeof(int32)
 * @param cnt tensor长度
 */
__aicore__ inline void CastFP32ToINT8(const LocalTensor<int8_t> outLocal, const LocalTensor<float> &inputLocal,
                                      const LocalTensor<uint8_t> &shareTmpUb, uint64_t cnt)
{
    LocalTensor<int32_t> int32 = shareTmpUb.ReinterpretCast<int32_t>();
    LocalTensor<half> tmpHalf = shareTmpUb.ReinterpretCast<half>();
    Cast(int32, inputLocal, RoundMode::CAST_RINT, cnt);
    pipe_barrier(PIPE_V);
    SetDeqScale(static_cast<half>(1.0));
    pipe_barrier(PIPE_V);
    Cast(tmpHalf, int32, RoundMode::CAST_ROUND, cnt);
    pipe_barrier(PIPE_V);
    Cast(outLocal, tmpHalf, RoundMode::CAST_TRUNC, cnt);
}

// divs by row, 每行的元素除以相同的元素
// dstUb[i, (j * 8) : (j * 8 + 7)] = src0Ub[i, (j * 8) : (j * 8 + 7)] / src1Ub[i, 0 : 7]
// src0Ub:[dealRowCount, columnCount], src1Ub:[dealRowCount, FP32_BLOCK_ELEMENT_NUM] dstUb:[dealRowCount,
// columnCount]
__aicore__ inline void RowClips(LocalTensor<float>& dstUb, const LocalTensor<float>& src0Ub,
    const LocalTensor<float>& maxUb, const LocalTensor<float>& minUb,
    uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount)
{
    uint32_t dtypeMask = FP32_REPEAT_ELEMENT_NUM;
    uint32_t dLoop = actualColumnCount / dtypeMask;
    uint32_t dRemain = actualColumnCount % dtypeMask;

    BinaryRepeatParams repeatParams;
    repeatParams.src0BlkStride = 1;
    repeatParams.src1BlkStride = 0;
    repeatParams.dstBlkStride = 1;
    repeatParams.src0RepStride = columnCount / FP32_BLOCK_ELEMENT_NUM;
    repeatParams.src1RepStride = 1;
    repeatParams.dstRepStride = columnCount / FP32_BLOCK_ELEMENT_NUM;
    uint32_t offset = 0;

    for (uint32_t i = 0; i < dLoop; i++) {
        Min(dstUb[offset], src0Ub[offset], maxUb, dtypeMask, dealRowCount, repeatParams);
        PipeBarrier<PIPE_V>();
        Max(dstUb[offset], dstUb[offset], minUb, dtypeMask, dealRowCount, repeatParams);
        offset += dtypeMask;
    }

    if (dRemain > 0) {
        Min(dstUb[dLoop * dtypeMask], src0Ub[dLoop * dtypeMask], maxUb, dRemain, dealRowCount, repeatParams);
        PipeBarrier<PIPE_V>();
        Max(dstUb[dLoop * dtypeMask], dstUb[dLoop * dtypeMask], minUb, dRemain, dealRowCount, repeatParams);
    }
}

__aicore__ inline void PerTileClipWithAlpha(LocalTensor<float>& dstClipUb,
                                            LocalTensor<float>& aMax,
                                            LocalTensor<float>& aMaxBrcb,
                                            const LocalTensor<float>& srcUb,
                                            const LocalTensor<uint8_t>& shareTmpUb,
                                            const PerTileQuantParams& perTileQuantParams)
{
    constexpr uint32_t brcnNum = 8; // brcn一次性处理8个数据
    uint32_t srcSize = perTileQuantParams.row * perTileQuantParams.col;
    uint32_t maxAlphaSize = perTileQuantParams.row * perTileQuantParams.tileNum * FP32_BLOCK_ELEMENT_NUM;

    LocalTensor<float> absSrcUb = shareTmpUb.template ReinterpretCast<float>();
    LocalTensor<float> maxAlpha = absSrcUb[Align(srcSize, FP32_BLOCK_ELEMENT_NUM)];
    LocalTensor<float> minAlpha = maxAlpha[Align(maxAlphaSize, FP32_BLOCK_ELEMENT_NUM)];

    Abs(absSrcUb, srcUb, srcSize);
    PipeBarrier<PIPE_V>();

    Rectangle rectangleParams {
        static_cast<uint32_t>(perTileQuantParams.row * perTileQuantParams.tileNum),
        static_cast<uint32_t>(perTileQuantParams.tileSize),
        static_cast<uint32_t>(perTileQuantParams.tileSize) // columnStride
    };
    RowMax(aMax, absSrcUb, rectangleParams);
    PipeBarrier<PIPE_V>();
    Brcb(aMaxBrcb, aMax, (CeilDivT((perTileQuantParams.row * perTileQuantParams.tileNum), brcnNum)), {1, brcnNum});
    PipeBarrier<PIPE_V>();

    Muls(maxAlpha, aMaxBrcb, perTileQuantParams.alpha, maxAlphaSize);
    Muls(minAlpha, aMaxBrcb, -perTileQuantParams.alpha, maxAlphaSize);
    PipeBarrier<PIPE_V>();

    RowClips(dstClipUb, srcUb, maxAlpha, minAlpha, perTileQuantParams.row * perTileQuantParams.tileNum,
        perTileQuantParams.tileSize, perTileQuantParams.tileSize);
    PipeBarrier<PIPE_V>();
}

/**
 * @brief DynamicQuant 对row行进行dynamicquant, float ---> int8, 每一行出一个系数。
 * @param outputLocal 输出tensor [row , col]，支持和inputLocal是同一块空间
 * @param scale 输出每行的反量化系数 [row]
 * @param inputLocal 输入tensor [row , col]
 * @param rowMax [row] 一行的最大值
 * @param rowMaxBrcb [row, 8] 一行的最大值，brcb扩充为8的倍数
 * @param shareTmpUb 临时buffer 内部需要的空间为 [row * col * sizeof(float)]
 * @param row 待处理的行数
 * @param col 待处理的列数
 */
__aicore__ inline void DynamicQuant(const LocalTensor<float> &outputLocal,
                                    const LocalTensor<float> &scale,
                                    const LocalTensor<float> &inputLocal,
                                    const LocalTensor<float> &aMax,
                                    const LocalTensor<float> &rowMaxBrcb,
                                    const LocalTensor<uint8_t> &shareTmpUb,
                                    float alpha,
                                    uint64_t row,
                                    uint64_t col)
{
    constexpr float maxInt8 = 127.0f;
    int32_t aMaxSizeAlign = Align(static_cast<uint32_t>(row), FP32_BLOCK_ELEMENT_NUM);
    LocalTensor<float> aMaxAlpha = shareTmpUb.ReinterpretCast<float>();
    LocalTensor<float> dupTensor = aMaxAlpha[aMaxSizeAlign];
    LocalTensor<float> scaleReciprocal = dupTensor[aMaxSizeAlign];
    LocalTensor<float> scaleReciprocalBrcb = scaleReciprocal[aMaxSizeAlign];

    Duplicate(dupTensor, 1.0f, row);
    Muls(aMaxAlpha, aMax, alpha, row);
    PipeBarrier<PIPE_V>();
    Muls(scale, aMaxAlpha, 1.0f / maxInt8, row);
    PipeBarrier<PIPE_V>();
    Div(scaleReciprocal, dupTensor, scale, row);
    PipeBarrier<PIPE_V>();
    Brcb(scaleReciprocalBrcb, scaleReciprocal, CeilDivT(row, 8UL), {1, 8});
    PipeBarrier<PIPE_V>();
    RowMuls(outputLocal, inputLocal, scaleReciprocalBrcb, Rectangle{(uint32_t)row, (uint32_t)col, (uint32_t)col});
    PipeBarrier<PIPE_V>();
}

/**
 * @brief QuantPerChannel 同时对row行进行FP32到int8的per-channel量化操作。一行中的每一列用不同的量化参数。
          outLocal[i, j] = inputLocal[i, j] * quantScaleLocal[j]
 * @param outLocal 输出tensor [row , col]
 * @param inputLocal 输入tensor [row , col]
 * @param quantScaleLocal quant系数 [1 , col]
 * @param shareTmpUb 临时buffer 内部需要的空间为 [row * col * 4]，源自CastFP32ToINT8
 * @param rectangleParams 描述待处理数据的排布，包括
          row 行数
          col 列数
          stride 一行的真实长度
 */
__aicore__ inline void QuantPerChannel(const LocalTensor<int8_t> &outLocal,
                                       const LocalTensor<float> &inputLocal,
                                       const LocalTensor<float> &quantScaleLocal,
                                       const LocalTensor<uint8_t> &shareTmpUb,
                                       const Rectangle& rectangleParams)
{
    VecMulMat(inputLocal, quantScaleLocal, inputLocal, rectangleParams);
    PipeBarrier<PIPE_V>();
    CastFP32ToINT8(outLocal, inputLocal, shareTmpUb, rectangleParams.row * rectangleParams.col);
}

/**
 * @brief QuantPerTensor 同时对row行进行FP32到int8的per-tensor量化操作。一行内共用同一个量化系数。
          outLocal[i , j] = inputLocal[i , j] * quantScaleLocal[i]
 * @param outLocal 输出tensor [row , col]
 * @param inputLocal 输入tensor [row , col]
 * @param quantScaleLocal quant系数 [row , 8]; 8 : 32Bytes对齐
 * @param shareTmpUb 临时buffer 内部需要的空间为 [row * col * 4]，源自CastFP32ToINT8
 * @param rectangleParams 描述待处理数据的排布，包括
          row 行数
          col 列数
          stride 一行的真实长度
 */
__aicore__ inline void QuantPerTensor(const LocalTensor<int8_t> &outLocal,
                                      const LocalTensor<float> &inputLocal,
                                      const LocalTensor<float> &quantScaleLocal,
                                      const LocalTensor<uint8_t> &shareTmpUb,
                                      const Rectangle& rectangleParams)
{
    RowMuls(inputLocal, inputLocal, quantScaleLocal, rectangleParams);
    PipeBarrier<PIPE_V>();
    CastFP32ToINT8(outLocal, inputLocal, shareTmpUb, rectangleParams.row * rectangleParams.col);
}

__aicore__ inline void QuantPerTile(const LocalTensor<int8_t> &outLocal,
                                    const LocalTensor<float> &inputLocal,
                                    const LocalTensor<uint8_t> &shareTmpUb,
                                    const PerTileQuantParams& perTileQuantParams)
{
    LocalTensor<float> scale =
        outLocal[perTileQuantParams.row * perTileQuantParams.col].template ReinterpretCast<float>();

    LocalTensor<float> clipOut = inputLocal;
    uint32_t aMaxSizeAlign = Align(static_cast<uint32_t>(perTileQuantParams.row * perTileQuantParams.tileNum),
        FP32_BLOCK_ELEMENT_NUM);
    uint32_t aMaxBrcbSize = perTileQuantParams.row * perTileQuantParams.tileNum * FP32_BLOCK_ELEMENT_NUM;
    LocalTensor<float> aMax = shareTmpUb.template ReinterpretCast<float>();
    LocalTensor<float> aMaxBrcb = aMax[aMaxSizeAlign];
    LocalTensor<uint8_t> sharedBuf = aMaxBrcb[aMaxBrcbSize].template ReinterpretCast<uint8_t>();

    PerTileClipWithAlpha(clipOut, aMax, aMaxBrcb, inputLocal, sharedBuf, perTileQuantParams);

    LocalTensor<float> quantOut = clipOut;
    DynamicQuant(quantOut, scale, clipOut,
                 aMax, aMaxBrcb, sharedBuf,
                 perTileQuantParams.alpha,
                 perTileQuantParams.row * perTileQuantParams.tileNum,
                 perTileQuantParams.tileSize);
    PipeBarrier<PIPE_V>();
    CastFP32ToINT8(outLocal, quantOut, sharedBuf, perTileQuantParams.row * perTileQuantParams.col);
}

/**
 * @brief DynamicQuant 同时对row行进行dynamicquant, float ---> int8, 每一行出一个系数。
 * @param outLocal 输出tensor [row , col]，支持和inputLocal是同一块空间
 * @param inputLocal 输入tensor [row , col]
 * @param scale 输出每行的反量化系数 [1 , row]
 * @param maxInt8Tensor [1 , row] 元素均为int8的最大值127
 * @param shareTmpUb 临时buffer 内部需要的空间为 [(Align(row * col, 8) + Align(row , 8) * ALIGN_BLOCK_SIZE) * sizeof(float)]
 * @param row 待处理的行数
 * @param col 待处理的列数
 */
__aicore__ inline void DynamicQuant(const LocalTensor<float> &outputLocal,
                                    const LocalTensor<float> &scale,
                                    const LocalTensor<float> &inputLocal,
                                    const LocalTensor<float> &maxInt8Tensor,
                                    const LocalTensor<uint8_t> &shareTmpUb,
                                    uint64_t row, uint64_t col)
{
    constexpr uint64_t brcnNum = 8; // brcb一次处理8个数据
    uint64_t computeSize = row * col;
    LocalTensor<float> inputCopy = shareTmpUb.ReinterpretCast<float>();
    LocalTensor<float> rowMaxBrcb = inputCopy[Align(computeSize, (uint64_t)ALIGN_BLOCK_SIZE)];
    // abs(x)
    Abs(inputCopy, inputLocal, computeSize);
    pipe_barrier(PIPE_V);
    Rectangle rectangleParams {
        (uint32_t)row,
        (uint32_t)col,
        (uint32_t)col // columnStride
    };
    // rowMax(abs(x))
    RowMax(inputCopy, inputCopy, rectangleParams);
    pipe_barrier(PIPE_V);

    // scaleOut = rowMax(abs(x)) / 127
    Div(scale, inputCopy, maxInt8Tensor, row);
    pipe_barrier(PIPE_V);

    // 1 / scaleOut = 127 / rowMax(abs(x))
    Div(inputCopy, maxInt8Tensor, inputCopy, row);
    pipe_barrier(PIPE_V);

    Brcb(rowMaxBrcb, inputCopy, static_cast<uint8_t> (CeilDivT(row, brcnNum)), {1, brcnNum});
    pipe_barrier(PIPE_V);

    // x * 1 / scaleOut
    RowMuls(outputLocal, inputLocal, rowMaxBrcb, rectangleParams);
}

/**
 * @brief RmsNorm 对一行进行rmsnorm
 * @param outLocal 输出tensor [1 * cnt]，支持和inputLocal是同一块空间
 * @param inputLocal 输入tensor [1 * cnt]
 * @param gammaLocal 系数gamma [1 * cnt]
 * @param shareTmpUb 临时buffer 内部需要的空间为 [cnt * sizeof(float) + ALIGN_BLOCK_SIZE]
 * @param rmsNormParams rms所需系数，包括
          reciprocal rmsnorm系数reciprocal
          epsilon rmsnorm系数epsilon
          row 处理的行数；预留参数，当前仅支持单个batch的处理，row为1，对应S1
          col 列数，对应H
 */
template <typename GammaType>
__aicore__ inline void RmsNorm(const LocalTensor<float> &outLocal,
                               const LocalTensor<float> &inputLocal,
                               const LocalTensor<GammaType> &gammaLocal,
                               const LocalTensor<uint8_t> &shareTmpUb,
                               const RmsNormParam& rmsNormParams) {
    uint64_t cnt = rmsNormParams.row * rmsNormParams.col;
    LocalTensor<float> xSquareLocal = shareTmpUb.ReinterpretCast<float>();
    LocalTensor<float> xSumLocal = xSquareLocal[cnt];
    Mul(xSquareLocal, inputLocal, inputLocal, cnt);
    pipe_barrier(PIPE_V);

    // calcNum >> 6 : calcNum / 64(FP32_REPEAT_ELEMENT_NUM)
    uint64_t repeatTimesAdd = static_cast<uint64_t>(cnt) >> 6;
    BinaryRepeatParams addParams = {
        1, // dstBlkStrideIn
        1, // src0BlkStrideIn
        1, // src1BlkStrideIn
        0, // dstRepStrideIn
        8, // src0RepStrideIn
        0 // src1RepStrideIn
    };
    Add(xSquareLocal,
        xSquareLocal[FP32_REPEAT_ELEMENT_NUM],
        xSquareLocal,
        FP32_REPEAT_ELEMENT_NUM,
        repeatTimesAdd - 1,
        addParams);
    pipe_barrier(PIPE_V);
    WholeReduceSum(xSumLocal, xSquareLocal, FP32_REPEAT_ELEMENT_NUM, 1, 8, 1, 8);
    pipe_barrier(PIPE_V);

    // Calc: xSum = xSum * reciprocal
    Muls<float>(xSumLocal, xSumLocal, rmsNormParams.reciprocal, 1);
    pipe_barrier(PIPE_V);

    // Calc: xSum = xSum + epsilon
    Adds<float>(xSumLocal, xSumLocal, rmsNormParams.epsilon, 1);
    pipe_barrier(PIPE_V);

    // Calc: xSum = sqrt(xSum)
    Sqrt(xSumLocal, xSumLocal, 1);
    pipe_barrier(PIPE_V);

    // Calc: xSquare[1, 8] = brc(xSum[1,1])
    BrcbRepeatParams repeatParams = {
        1, // dstBlkStride
        1 // dstRepStride
    };
    Brcb(xSquareLocal, xSumLocal, 1, repeatParams);
    pipe_barrier(PIPE_V);

    // Calc: inputLocal = inputLocal / xSquareLocal
    uint64_t mask[2] = {UINT64_MAX, UINT64_MAX};
    BinaryRepeatParams divParams = {
        1, // dstBlkStrideIn
        1, // src0BlkStrideIn
        0, // src1BlkStrideIn
        8, // dstRepStrideIn
        8, // src0RepStrideIn
        0 // src1RepStrideIn
    };
    Div(inputLocal, inputLocal, xSquareLocal, mask, cnt / 64, divParams);

    pipe_barrier(PIPE_V);

    Cast(xSquareLocal, gammaLocal, RoundMode::CAST_NONE, cnt);

    pipe_barrier(PIPE_V);

    Mul(outLocal, inputLocal, xSquareLocal, cnt);
#if MLA_PROLOG_V3_SFA1
    AscendC::PipeBarrier<PIPE_V>();
#else

    if (unlikely(rmsNormParams.isScaleEnable)) {
        pipe_barrier(PIPE_V);
#endif
        Muls(outLocal, outLocal, rmsNormParams.scale, cnt);
#if !MLA_PROLOG_V3_SFA1
    }
#endif
}

/**
 * @brief RotaryPosEmb, 同时做row行的RotaryPosEmb，每一行的元素为col
 * @param outputLocal 输出tensor [row * col]，支持和inputLocal是同一块空间
 * @param inputLocal 输入tensor [row * col]
 * @param cosLocal cos系数tensor [(row - 1) * sinCosRepStride + col]
 * @param sinLocal sin系数tensor [(row - 1) * sinCosRepStride + col] - 1 应已在sin中
 * @param shareTmpUb 临时buffer 内部需要的空间为 [2 * row * col * sizeof(C)]
 * @param row 待处理的行数
 * @param col 待处理的列数  col <= 512 / sizeof(C)
 * @param sinCosRepStride 行与行之间sin/cos系数的偏移，单位为元素个数。
 */
template <typename C>
__aicore__ inline void RotaryPosEmb(const LocalTensor<C> &outputLocal,
                                    const LocalTensor<C> &inputLocal,
                                    const LocalTensor<C> &cosLocal,
                                    const LocalTensor<C> &sinLocal,
                                    const LocalTensor<uint8_t> &shareTmpUb,
                                    uint64_t row, uint64_t col,
                                    uint8_t sinCosRepStride) {
    uint64_t cnt = row * col;
    uint64_t rsvdCnt = 0;
    LocalTensor<C> reArrLocal = shareTmpUb.ReinterpretCast<C>();
    LocalTensor<C> outputLocalSinTmp = shareTmpUb.ReinterpretCast<C>()[cnt];
    GatherMaskParams gatherMaskParams = {
        1,   // repeatTimes
        1,   // src0BlockStride
        0,   // src0RepeatStride
        0    // src1RepeatStride
    };
    // 取奇数索引元素
    GatherMask(reArrLocal, inputLocal, 1, true,
               col * row, gatherMaskParams, rsvdCnt);
    // 取偶数索引元素
    GatherMask(reArrLocal[cnt >> 1], inputLocal, 2, true,
               col * row, gatherMaskParams, rsvdCnt);
    pipe_barrier(PIPE_V);
    uint8_t blockNumPerRow = col / (ALIGN_BLOCK_SIZE / sizeof(C));
    uint8_t blockNumPerRowHalf = blockNumPerRow >> 1;
    uint8_t blockNumSinCosRepStride = sinCosRepStride / (ALIGN_BLOCK_SIZE / sizeof(C));
    BinaryRepeatParams mulParams = {
        1, // dstBlkStrideIn
        1, // src0BlkStrideIn
        1, // src1BlkStrideIn
        blockNumPerRow, // dstRepStrideIn
        blockNumPerRowHalf, // src0RepStrideIn
        blockNumSinCosRepStride // src1RepStrideIn
    };
    Mul(outputLocal, reArrLocal, cosLocal, col >> 1, row, mulParams);
    Mul(outputLocal[col >> 1], reArrLocal[cnt >> 1], cosLocal[col >> 1],
                 col >> 1, row, mulParams);
    Mul(outputLocalSinTmp, reArrLocal[cnt >> 1], sinLocal,
                 col >> 1, row, mulParams);
    Mul(outputLocalSinTmp[col >> 1], reArrLocal, sinLocal[col >> 1],
                 col >> 1, row, mulParams);
    pipe_barrier(PIPE_V);
    Add(outputLocal, outputLocal, outputLocalSinTmp, cnt);
}

}
#endif // MLA_PROLOG_V3_VECTOR_COMM_H
