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
 * \file flat_quant_high.h
 * \brief
 */
#ifndef FLAT_QUANT_HIGH_H
#define FLAT_QUANT_HIGH_H

#include "tensor_utils.h"

namespace FlatQuantNS {
template <typename T>
class FlatQuantHigh {
public:
    aifunc FlatQuantHigh() {}
    aifunc void Init(GM_ADDR xmtx_, GM_ADDR p1mtx_, GM_ADDR p2mtx_, GM_ADDR groupList_, GM_ADDR out_, GM_ADDR qscale_,
                     GM_ADDR workspace_, const FlatQuantTilingData* tilingData)
    {
        shape.M = tilingData->M;
        shape.N = tilingData->N;
        shape.K = tilingData->K;
        clipRatio = tilingData->clipRatio;
        if (tilingData->groupNum > 0) {
            groupListGM.SetGlobalBuffer((__gm__ int64_t*)groupList_);
            shape.K = GetQuantK(groupListGM, tilingData->groupNum, tilingData->groupListType, tilingData->K);
            if (tilingData->clearPadding != 0) {
                zeroK = tilingData->K - shape.K;
            }
        }
        tiling();

        xGM.SetGlobalBuffer((__gm__ T*)xmtx_);
        p1GM.SetGlobalBuffer((__gm__ T*)p1mtx_);
        p2GM.SetGlobalBuffer((__gm__ T*)p2mtx_);
        outGM.SetGlobalBuffer((__gm__ int4b_t*)out_);
        qscaleGM.SetGlobalBuffer((__gm__ float*)qscale_);
        x1GM.SetGlobalBuffer((__gm__ T*)workspace_ +
                             useAivNum * K_DOUBLE_VEC * shape.Mceil * shape.N * sizeof(float) / sizeof(T));
        x2GM.SetGlobalBuffer((__gm__ float*)workspace_);

        pipe.InitBuffer(bufQueue, HIGH_UB_SIZE);
        xTensor = bufQueue.Get<float>();
        qscaleTensor = xTensor[DATA_COUNT];
        yTensor = qscaleTensor[SCALE_COUNT].template ReinterpretCast<half>();
        outTensor = yTensor.template ReinterpretCast<int4b_t>();
        absTensor = yTensor[DATA_COUNT];

        eventIdVToS = static_cast<event_t>(pipe.FetchEventID(HardEvent::V_S));
        eventIdVToMte2 = static_cast<event_t>(pipe.FetchEventID(HardEvent::V_MTE2));
        eventIdMte2ToV = static_cast<event_t>(pipe.FetchEventID(HardEvent::MTE2_V));
        eventIdVToMte3 = static_cast<event_t>(pipe.FetchEventID(HardEvent::V_MTE3));
        eventIdMte3ToV = static_cast<event_t>(pipe.FetchEventID(HardEvent::MTE3_V));
        eventIdMte3ToS = static_cast<event_t>(pipe.FetchEventID(HardEvent::MTE3_S));
    }

    aifunc void tiling()
    {
        aivNum = GetBlockNum() * DOUBLE;
        useAivNum = (shape.K + K_PER_VEC - 1) / K_PER_VEC;
        if (useAivNum > aivNum) {
            useAivNum = aivNum;
        }
        int k_per_core = ((shape.K + aivNum - 1) / aivNum + K_PER_VEC - 1) / (K_PER_VEC) * (K_PER_VEC);
        shape.K1 = k_per_core * GetBlockIdx(); // vector blk idx is 0~40
        shape.K2 = ((k_per_core + shape.K1) > shape.K) ? shape.K : (k_per_core + shape.K1);
        shape.Mceil = (shape.M + CEIL_SIZE - 1) / CEIL_SIZE * CEIL_SIZE;
        shape.Nceil = (shape.N + CEIL_SIZE - 1) / CEIL_SIZE * CEIL_SIZE;
        splitRow = DATA_COUNT / shape.Nceil / CEIL_SIZE * CEIL_SIZE;
        x1Offset = GetBlockIdx() * K_PER_VEC * shape.M * shape.N;
        x2Offset = GetBlockIdx() * K_DOUBLE_VEC * shape.Mceil * shape.N;

        if (zeroK > 0) {
            int64_t singleCoreZeroK = (zeroK + aivNum - 1) / aivNum;
            zeroStartK = GetBlockIdx() * singleCoreZeroK;
            zeroEndK = ((zeroStartK + singleCoreZeroK) > zeroK) ? zeroK : (zeroStartK + singleCoreZeroK);
        }
    }

    aifunc void Process()
    {
        Duplicate<float>(xTensor, (T)0, DATA_COUNT);
        Duplicate<half>(absTensor, (half)0, DATA_COUNT);
        Duplicate<float>(qscaleTensor, (float)0, SCALE_COUNT);
        PipeBarrier<PIPE_V>();

        int64_t scaleK = shape.K1;
        int64_t k = shape.K1;
        for (int64_t startK = shape.K1; startK < shape.K2; startK += K_PER_VEC) {
            int64_t endK = startK + K_PER_VEC > shape.K2 ? shape.K2 : startK + K_PER_VEC;
            ProcessHighK(startK, endK - startK);
            while (k < endK) {
                SplitQuant(k, scaleK);
                k++;
            }
            if (k == shape.K2 || k == scaleK + SCALE_COUNT) {
                CopyOutQuant(scaleK, k - scaleK);
                scaleK = k;
            }
        }
        ProcessZeroK();
    }

    aifunc void ProcessZeroK()
    {
        if (zeroStartK >= zeroEndK) {
            return;
        }
        SetFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
        WaitFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
        Duplicate<float>(xTensor, (float)0, DATA_COUNT);
        Duplicate<float>(qscaleTensor, (float)0, SCALE_COUNT);
        SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        int64_t zeroOffset = (shape.K + zeroStartK) * shape.M * shape.N;
        int64_t zeroCount = (zeroEndK - zeroStartK) * shape.M * shape.N;
        ClearGM(outGM[zeroOffset], xTensor.template ReinterpretCast<int4b_t>(), DATA_COUNT * sizeof(float) * DOUBLE,
                zeroCount);
        ClearGM(qscaleGM[shape.K + zeroStartK], qscaleTensor, SCALE_COUNT, zeroEndK - zeroStartK);
    }

    aifunc void CopyOutQuant(int64_t scaleK, int64_t scaleCount)
    {
        SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(scaleCount * sizeof(float)), 0, 0, 0};
        DataCopyPad(qscaleGM[scaleK], qscaleTensor, copyParams);
        SetFlag<HardEvent::MTE3_S>(eventIdMte3ToS);
        WaitFlag<HardEvent::MTE3_S>(eventIdMte3ToS);
    }

    aifunc void SplitQuant(int64_t k, int64_t scaleK)
    {
        float maxValue = 0.0f;
        ComputeMaxValue(k, maxValue);

        qscaleTensor.SetValue(k - scaleK, maxValue * clipRatio / NUM_FLOAT_SEVEN);
        float maxValueFloat = maxValue != 0 ? ((NUM_FLOAT_SEVEN / clipRatio) / maxValue) : NUM_FLOAT_SEVEN;
        for (int64_t rowIdx = 0; rowIdx < shape.M; rowIdx += splitRow) {
            int64_t rowNum = (shape.M - rowIdx < splitRow) ? shape.M - rowIdx : splitRow;
            int64_t rowNumCeil = (shape.Mceil - rowIdx < splitRow) ? shape.Mceil - rowIdx : splitRow;
            int64_t realCount = rowNum * shape.N;

            DataCopy(xTensor, x2GM[x2Offset + (k % K_DOUBLE_VEC) * shape.Mceil * shape.N + rowIdx * shape.N],
                     rowNumCeil * shape.N);
            SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
            WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
            Cast(absTensor, xTensor, RoundMode::CAST_RINT, realCount);
            PipeBarrier<PIPE_V>();
            Cast(xTensor, absTensor, RoundMode::CAST_NONE, realCount);
            PipeBarrier<PIPE_V>();
            Muls(xTensor, xTensor, maxValueFloat, realCount);
            PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
            WaitFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
            Cast(yTensor, xTensor, RoundMode::CAST_RINT, realCount);
            PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_MTE2>(eventIdVToMte2);
            WaitFlag<HardEvent::V_MTE2>(eventIdVToMte2);
            Cast(outTensor, yTensor, RoundMode::CAST_NONE, realCount);
            SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
            WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
            DataCopyExtParams copyParams{1, (uint32_t)(realCount) / DOUBLE, 0, 0, 0};
            DataCopyPad(outGM[k * shape.M * shape.N + rowIdx * shape.N], outTensor, copyParams);
        }
    }

    aifunc void ComputeMaxValue(int64_t k, float& maxValue)
    {
        for (int64_t rowIdx = 0; rowIdx < shape.Mceil; rowIdx += splitRow) {
            int64_t rowNum = (shape.M - rowIdx < splitRow) ? shape.M - rowIdx : splitRow;
            int64_t rowNumCeil = (shape.Mceil - rowIdx < splitRow) ? shape.Mceil - rowIdx : splitRow;
            DataCopy(xTensor, x2GM[x2Offset + (k % K_DOUBLE_VEC) * shape.Mceil * shape.N + rowIdx * shape.N],
                     rowNumCeil * shape.N);
            SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
            WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
            Cast(xTensor.template ReinterpretCast<half>(), xTensor, RoundMode::CAST_RINT, rowNum * shape.N);
            PipeBarrier<PIPE_V>();
            Abs(absTensor, xTensor.template ReinterpretCast<half>(), rowNum * shape.N);
            PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_MTE2>(eventIdVToMte2);
            WaitFlag<HardEvent::V_MTE2>(eventIdVToMte2);

            CalReduceMax(absTensor, rowNumCeil * shape.Nceil, eventIdVToS);
            float tmpMax = static_cast<float>(absTensor.GetValue(0));
            if (tmpMax != tmpMax || tmpMax > maxValue) {
                maxValue = tmpMax;
            }
        }
    }

    aifunc void ProcessHighK(int64_t k, int64_t batch)
    {
        int64_t offset1 = x1Offset + (k % K_PER_VEC) * shape.M * shape.N;
        int64_t offset2 = x2Offset + (k % K_DOUBLE_VEC) * shape.Mceil * shape.N;
        matmulR.SetSingleShape(batch * shape.M, shape.N, shape.N);
        matmulR.SetTensorA(xGM[k * shape.M * shape.N], false);
        matmulR.SetTensorB(p2GM, false);
        matmulR.IterateAll(x1GM[offset1], false);
        PipeBarrier<PIPE_ALL>();

        matmulL.SetTensorA(p1GM, false);
        for (int64_t i = 0; i < batch; i++) {
            matmulL.SetTensorB(x1GM[offset1], false);
            matmulL.IterateAll(x2GM[offset2], false);
            offset1 += shape.M * shape.N;
            offset2 += shape.Mceil * shape.N;
        }
    }

public:
    TPipe pipe;
    matmul::Matmul<matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>,
                   matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>,
                   matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>,
                   matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>, MDL_CFG>
        matmulR;

    matmul::Matmul<matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>,
                   matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>,
                   matmul::MatmulType<TPosition::GM, CubeFormat::ND, float>,
                   matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>, MDL_CFG>
        matmulL;

private:
    FlatQuantShapeInfo shape;
    GlobalTensor<T> xGM;
    GlobalTensor<T> p1GM;
    GlobalTensor<T> p2GM;
    GlobalTensor<int64_t> groupListGM;
    GlobalTensor<int4b_t> outGM;
    GlobalTensor<float> qscaleGM;
    GlobalTensor<T> x1GM;
    GlobalTensor<float> x2GM;

    TBuf<QuePosition::VECCALC> bufQueue;

    LocalTensor<float> xTensor;
    LocalTensor<float> qscaleTensor;
    LocalTensor<half> yTensor;
    LocalTensor<int4b_t> outTensor;
    LocalTensor<half> absTensor;

    event_t eventIdVToS;
    event_t eventIdVToMte2;
    event_t eventIdMte2ToV;
    event_t eventIdVToMte3;
    event_t eventIdMte3ToV;
    event_t eventIdMte3ToS;

    float clipRatio = 0.0f;
    int64_t splitRow = 0;
    int64_t aivNum = 0;
    int64_t useAivNum = 0;
    int64_t x1Offset = 0;
    int64_t x2Offset = 0;
    int64_t zeroK = 0;
    int64_t zeroStartK = 0;
    int64_t zeroEndK = 0;
};
} // namespace FlatQuantNS

#endif // FLAT_QUANT_HIGH_H
