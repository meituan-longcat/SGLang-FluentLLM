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
 * \file flat_quant_vec.h
 * \brief
 */
#ifndef FLAT_QUANT_VEC_H
#define FLAT_QUANT_VEC_H

#include "tensor_utils.h"

namespace FlatQuantNS {
template <typename T, uint8_t MM_MODE>
class FlatQuantVec {
public:
    aifunc FlatQuantVec() {}
    aifunc void Init(GM_ADDR p1mtx_, GM_ADDR groupList_, GM_ADDR out_, GM_ADDR qscale_, GM_ADDR workspace_,
                     const FlatQuantTilingData* tilingData)
    {
        shape.K = tilingData->K;
        shape.M = tilingData->M;
        shape.N = tilingData->N;
        clipRatio = tilingData->clipRatio;
        if (tilingData->groupNum > 0) {
            groupListGM.SetGlobalBuffer((__gm__ int64_t*)groupList_);
            shape.K = GetQuantK(groupListGM, tilingData->groupNum, tilingData->groupListType, tilingData->K);
            if (tilingData->clearPadding != 0) {
                zeroK = tilingData->K - shape.K;
            }
        }
        if (shape.K == 0) {
            TilingZeroK();
        } else {
            tiling();
        }

        p1GM.SetGlobalBuffer((__gm__ T*)p1mtx_);
        outGM.SetGlobalBuffer((__gm__ int4b_t*)out_);
        qscaleGM.SetGlobalBuffer((__gm__ float*)qscale_);
        outnzGM.SetGlobalBuffer((__gm__ half*)workspace_);
        doubleP1GM.SetGlobalBuffer(
            (__gm__ T*)(workspace_ + (shape.K + shape.K % 2) * shape.Mceil * shape.N * sizeof(T)));

        pipe.InitBuffer(bufQueue, UB_SIZE);
        xTensor = bufQueue.Get<T>();
        x2Tensor = xTensor[DATA_COUNT];
        floatTensor = x2Tensor[DATA_COUNT].template ReinterpretCast<float>();
        yTensor = floatTensor[DATA_COUNT].template ReinterpretCast<int4b_t>();
        y2Tensor = yTensor[DATA_COUNT];
        qscaleTensor = y2Tensor[DATA_COUNT].template ReinterpretCast<float>();
        absTensor = qscaleTensor[SCALE_COUNT].template ReinterpretCast<half>();

        eventIdVToS = static_cast<event_t>(pipe.FetchEventID(HardEvent::V_S));
        eventIdVToMte2 = static_cast<event_t>(pipe.FetchEventID(HardEvent::V_MTE2));
        eventIdMte2ToMte3 = static_cast<event_t>(pipe.FetchEventID(HardEvent::MTE2_MTE3));
        eventIdVToMte3 = static_cast<event_t>(pipe.FetchEventID(HardEvent::V_MTE3));
        eventIdMte3ToV = static_cast<event_t>(pipe.FetchEventID(HardEvent::MTE3_V));
        eventIdMte3ToS = static_cast<event_t>(pipe.FetchEventID(HardEvent::MTE3_S));
    }

    aifunc void tiling()
    {
        int allTimes = GetBlockNum() * 16; // 每个核处理16个批次, 控制同步计数器不会超过16
        shape.perK = (shape.K + allTimes - 1) / allTimes;                      // 每个批次处理多少个K
        shape.perK = (shape.perK + K_PER_VEC - 1) / (K_PER_VEC) * (K_PER_VEC); // 和4对齐

        int k_per_core = ((shape.K + GetBlockNum() - 1) / GetBlockNum() + shape.perK - 1) / shape.perK * shape.perK;
        shape.K1 = k_per_core * (GetBlockIdx() >> 1); // vector blk idx is 0~40
        shape.K2 = ((k_per_core + shape.K1) > shape.K) ? shape.K : (k_per_core + shape.K1);
        shape.Mceil = (shape.M + CEIL_SIZE - 1) / CEIL_SIZE * CEIL_SIZE;
        shape.Nceil = (shape.N + CEIL_SIZE - 1) / CEIL_SIZE * CEIL_SIZE;
        splitRow = DATA_COUNT / shape.Nceil / CEIL_SIZE * CEIL_SIZE;
        splitCount = (shape.Mceil + splitRow - 1) / splitRow;
        subBlockIdx = GetSubBlockIdx();

        if (zeroK > 0) {
            int64_t aivNum = GetBlockNum() * DOUBLE;
            int64_t singleCoreZeroK = (zeroK + aivNum - 1) / aivNum;
            zeroStartK = GetBlockIdx() * singleCoreZeroK;
            zeroEndK = ((zeroStartK + singleCoreZeroK) > zeroK) ? zeroK : (zeroStartK + singleCoreZeroK);
        }
    }

    aifunc void Process()
    {
        // validK == 0 is routed here on AIV only. Skip every AIC/AIV
        // handshake and only materialize the all-zero public outputs.
        if (shape.K == 0) {
            if (zeroStartK < zeroEndK) {
                ProcessZeroK();
            }
            return;
        }
        if constexpr (MM_MODE == MM_DOUBLE_MODE) {
            ProcessP1();
        }
        PipeBarrier<PIPE_ALL>();
        CrossCoreSetFlag<SYNC_MODE0, PIPE_MTE3>(VEC_SYNC_ID);
        CrossCoreWaitFlag(VEC_SYNC_ID);
        CrossCoreSetFlag<SYNC_MODE2, PIPE_MTE3>(VEC_CUBE_SYNC_ID);
        ClearQuant();
        CrossCoreSetFlag<SYNC_MODE1, PIPE_MTE3>(TWO_VEC_SYNC_ID);
        CrossCoreWaitFlag(TWO_VEC_SYNC_ID);

        in_empty.setall();
        out_empty.setall();
        int64_t scaleK = shape.K1;
        for (int64_t startK = shape.K1; startK < shape.K2; startK += shape.perK) {
            int64_t endK = startK + shape.perK > shape.K2 ? shape.K2 : startK + shape.perK;
            for (int64_t k = startK + subBlockIdx; k < endK; k += DOUBLE) {
                // 量化分给两个vec核，0核做偶数，1核做奇数
                Quant(k, scaleK);
            }
            if (endK == shape.K2 || endK + shape.perK > scaleK + SCALE_COUNT) {
                CopyOutQuant(scaleK, endK - scaleK);
                scaleK = endK;
            }
        }
        if (zeroStartK < zeroEndK) {
            ProcessZeroK();
        }
        in_empty.release();
        out_empty.release();
    }

    aifunc void TilingZeroK()
    {
        shape.Mceil = (shape.M + CEIL_SIZE - 1) / CEIL_SIZE * CEIL_SIZE;
        shape.Nceil = (shape.N + CEIL_SIZE - 1) / CEIL_SIZE * CEIL_SIZE;
        int64_t aivNum = GetBlockNum() * DOUBLE;
        int64_t singleCoreZeroK = (zeroK + aivNum - 1) / aivNum;
        zeroStartK = GetBlockIdx() * singleCoreZeroK;
        zeroEndK = ((zeroStartK + singleCoreZeroK) > zeroK) ? zeroK : (zeroStartK + singleCoreZeroK);
    }

    aifunc void ProcessZeroK()
    {
        SetFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
        WaitFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
        Duplicate<float>(xTensor.template ReinterpretCast<float>(), (float)0, DATA_COUNT);
        Duplicate<float>(qscaleTensor, (float)0, SCALE_COUNT);
        SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        int64_t zeroOffset = (shape.K + zeroStartK) * shape.M * shape.N;
        int64_t zeroCount = (zeroEndK - zeroStartK) * shape.M * shape.N;
        ClearGM(outGM[zeroOffset], xTensor.template ReinterpretCast<int4b_t>(), DATA_COUNT * sizeof(float) * DOUBLE,
                zeroCount);
        ClearGM(qscaleGM[shape.K + zeroStartK], qscaleTensor, SCALE_COUNT, zeroEndK - zeroStartK);
    }

    aifunc void ProcessP1()
    {
        int64_t startM = GetBlockIdx() * K_PER_VEC;
        if (startM >= shape.Mceil) {
            return;
        }
        Duplicate(xTensor, (T)0, K_PER_VEC * shape.Mceil);
        if (startM < shape.M) {
            SetFlag<HardEvent::V_MTE2>(eventIdVToMte2);
            WaitFlag<HardEvent::V_MTE2>(eventIdVToMte2);
            uint16_t prows = shape.M - startM < K_PER_VEC ? shape.M - startM : K_PER_VEC;
            DataCopyExtParams copyParams{prows, static_cast<uint32_t>(shape.M * sizeof(T)), 0, 0, 0};
            DataCopyPadExtParams<T> padParams{true, 0, static_cast<uint8_t>(shape.Mceil - shape.M), 0};
            DataCopyPad(xTensor, p1GM[startM * shape.M], copyParams, padParams);
            SetFlag<HardEvent::MTE2_MTE3>(eventIdMte2ToMte3);
            WaitFlag<HardEvent::MTE2_MTE3>(eventIdMte2ToMte3);
        } else {
            SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
            WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        }
        DataCopy(doubleP1GM[startM * shape.Mceil], xTensor, K_PER_VEC * shape.Mceil);
    }

    aifunc void ClearQuant()
    {
        // pre-set all buffers to zero
        Duplicate<T>(xTensor, (T)0, DATA_COUNT);
        Duplicate<T>(x2Tensor, (T)0, DATA_COUNT);
        Duplicate<half>(absTensor, (half)0, DATA_COUNT);
        Duplicate<float>(qscaleTensor, (float)0, SCALE_COUNT);

        SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        int64_t midK = (shape.K2 - shape.K1) / 2 + shape.K1;
        int64_t startK = GetSubBlockIdx() == 0 ? shape.K1 : midK;
        int64_t endK = GetSubBlockIdx() == 0 ? midK : shape.K2;
        for (int64_t k = startK; k < endK; k += SCALE_COUNT) {
            int64_t len = endK - k > SCALE_COUNT ? SCALE_COUNT : endK - k;
            DataCopyExtParams copyParams{1, static_cast<uint32_t>(len * sizeof(float)), 0, 0, 0};
            DataCopyPad(qscaleGM[k], qscaleTensor, copyParams);
        }
        SetFlag<HardEvent::MTE3_S>(eventIdMte3ToS);
        WaitFlag<HardEvent::MTE3_S>(eventIdMte3ToS);
    }

    aifunc void CopyOutQuant(int64_t scaleK, int64_t scaleCount)
    {
        SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        SetAtomicAdd<float>();
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(scaleCount * sizeof(float)), 0, 0, 0};
        DataCopyPad(qscaleGM[scaleK], qscaleTensor, copyParams);
        SetAtomicNone();
        SetFlag<HardEvent::MTE3_S>(eventIdMte3ToS);
        WaitFlag<HardEvent::MTE3_S>(eventIdMte3ToS);
    }

    aifunc void Quant(int64_t k, int64_t scaleK)
    {
        if (splitRow < shape.M) {
            SplitQuant(k, scaleK);
            return;
        }

        LocalTensor<half> inTensor = GetXTensor(k >> 1).template ReinterpretCast<half>();
        if (k == shape.K1 + subBlockIdx) {
            CrossCoreWaitFlag(CUBE_VEC_SYNC_ID);
            in_empty.wait();
            DataCopy(inTensor, outnzGM[k * shape.Mceil * shape.N], shape.Mceil * shape.N);
            in_ready.set();
        }
        int64_t nextK = k + DOUBLE;
        if (nextK < shape.K2) {
            if (nextK % shape.perK == subBlockIdx) {
                CrossCoreWaitFlag(CUBE_VEC_SYNC_ID);
            }
            in_empty.wait();
            DataCopy(GetXTensor(nextK >> 1).template ReinterpretCast<half>(), outnzGM[nextK * shape.Mceil * shape.N],
                     shape.Mceil * shape.N);
            in_ready.set();
        }

        // Quant  (abs -> max -> 7/max -> x*(7/max))
        out_empty.wait();
        in_ready.wait();

        int64_t realCount = shape.M * shape.N;
        Abs(absTensor, inTensor, realCount);
        PipeBarrier<PIPE_V>();
        CalReduceMax(absTensor, shape.Mceil * shape.Nceil, eventIdVToS);
        float maxValue = static_cast<float>(absTensor.GetValue(0));
        qscaleTensor.SetValue(k - scaleK, maxValue * clipRatio / NUM_FLOAT_SEVEN);

        float maxValueFloat = maxValue != 0 ? ((NUM_FLOAT_SEVEN / clipRatio) / maxValue) : NUM_FLOAT_SEVEN;
        Cast(floatTensor, inTensor, RoundMode::CAST_NONE, realCount);
        PipeBarrier<PIPE_V>();
        Muls(floatTensor, floatTensor, maxValueFloat, realCount);
        PipeBarrier<PIPE_V>();
        Cast(inTensor, floatTensor, RoundMode::CAST_RINT, realCount);
        PipeBarrier<PIPE_V>();
        Cast(GetYTensor(k >> 1), inTensor, RoundMode::CAST_NONE, realCount);

        out_ready.set();
        in_empty.set();

        out_ready.wait();
        DataCopyExtParams copyParams{1, (uint32_t)(realCount >> 1), 0, 0, 0};
        DataCopyPad(outGM[k * shape.M * shape.N], GetYTensor(k >> 1), copyParams);
        out_empty.set();
    }

    aifunc void SplitQuant(int64_t k, int64_t scaleK)
    {
        float maxValue = 0.0f;
        ComputeMaxValue(k, maxValue);

        qscaleTensor.SetValue(k - scaleK, maxValue * clipRatio / NUM_FLOAT_SEVEN);
        float maxValueFloat = maxValue != 0 ? ((NUM_FLOAT_SEVEN / clipRatio) / maxValue) : NUM_FLOAT_SEVEN;
        int64_t p = (k - shape.K1) * splitCount + splitCount;
        int64_t y = (k - shape.K1) / 2 * splitCount;
        for (int64_t rowIdx = 0; rowIdx < shape.M; rowIdx += splitRow) {
            int64_t rowNum = (shape.M - rowIdx < splitRow) ? shape.M - rowIdx : splitRow;
            int64_t nextRowIdx = (rowIdx + splitRow >= shape.M) ? 0 : rowIdx + splitRow;
            int64_t nextRowNum = (shape.Mceil - nextRowIdx < splitRow) ? shape.Mceil - nextRowIdx : splitRow;
            int64_t nextK = (nextRowIdx == 0) ? k + 2 : k;
            if (nextK < shape.K2) {
                if (nextK % shape.perK == subBlockIdx && nextRowIdx == 0) {
                    CrossCoreWaitFlag(CUBE_VEC_SYNC_ID);
                }
                in_empty.wait();
                DataCopy(GetXTensor(p + 1).template ReinterpretCast<half>(),
                         outnzGM[nextK * shape.Mceil * shape.N + nextRowIdx * shape.N], nextRowNum * shape.N);
                in_ready.set();
            }
            out_empty.wait();
            in_ready.wait();

            int64_t realCount = rowNum * shape.N;
            LocalTensor<half> inTensor = GetXTensor(p).template ReinterpretCast<half>();
            Cast(floatTensor, inTensor, RoundMode::CAST_NONE, realCount);
            PipeBarrier<PIPE_V>();
            Muls(floatTensor, floatTensor, maxValueFloat, realCount);
            PipeBarrier<PIPE_V>();
            Cast(inTensor, floatTensor, RoundMode::CAST_RINT, realCount);
            PipeBarrier<PIPE_V>();

            Cast(GetYTensor(y), inTensor, RoundMode::CAST_NONE, realCount);

            out_ready.set();
            in_empty.set();

            out_ready.wait();
            DataCopyExtParams copyParams{1, (uint32_t)(realCount >> 1), 0, 0, 0};
            DataCopyPad(outGM[k * shape.M * shape.N + rowIdx * shape.N], GetYTensor(y), copyParams);
            out_empty.set();
            p++;
            y++;
        }
    }

    aifunc void ComputeMaxValue(int64_t k, float& maxValue)
    {
        int64_t p = (k - shape.K1) * splitCount;
        for (int64_t rowIdx = 0; rowIdx < shape.Mceil; rowIdx += splitRow) {
            int64_t rowNum = (shape.M - rowIdx < splitRow) ? shape.M - rowIdx : splitRow;
            int64_t rowNumCeil = (shape.Mceil - rowIdx < splitRow) ? shape.Mceil - rowIdx : splitRow;
            int64_t nextRowIdx = (rowIdx + splitRow >= shape.M) ? 0 : rowIdx + splitRow;
            int64_t nextRowNum = (shape.Mceil - nextRowIdx < splitRow) ? shape.Mceil - nextRowIdx : splitRow;
            if (k == shape.K1 + subBlockIdx && rowIdx == 0) {
                CrossCoreWaitFlag(CUBE_VEC_SYNC_ID);
                in_empty.wait();
                DataCopy(GetXTensor(p).template ReinterpretCast<half>(), outnzGM[k * shape.Mceil * shape.N],
                         rowNumCeil * shape.N);
                in_ready.set();
            }
            in_empty.wait();
            DataCopy(GetXTensor(p + 1).template ReinterpretCast<half>(),
                     outnzGM[k * shape.Mceil * shape.N + nextRowIdx * shape.N], nextRowNum * shape.N);
            in_ready.set();

            // Quant  (abs -> max -> 7/max -> x*(7/max))
            in_ready.wait();
            Abs(absTensor, GetXTensor(p).template ReinterpretCast<half>(), rowNum * shape.N);
            PipeBarrier<PIPE_V>();
            in_empty.set();

            CalReduceMax(absTensor, rowNumCeil * shape.Nceil, eventIdVToS);
            float tmpMax = static_cast<float>(absTensor.GetValue(0));
            if (tmpMax != tmpMax || tmpMax > maxValue) {
                maxValue = tmpMax;
            }
            p++;
        }
    }

    __aicore__ inline LocalTensor<T> GetXTensor(int64_t k) { return ((k & 1) == 0) ? xTensor : x2Tensor; };

    __aicore__ inline LocalTensor<int4b_t> GetYTensor(int64_t k) { return ((k & 1) == 0) ? yTensor : y2Tensor; };

private:
    TPipe pipe;
    FlatQuantShapeInfo shape;
    GlobalTensor<T> p1GM;
    GlobalTensor<int64_t> groupListGM;
    GlobalTensor<int4b_t> outGM;
    GlobalTensor<float> qscaleGM;
    GlobalTensor<half> outnzGM;
    GlobalTensor<T> doubleP1GM;

    TBuf<QuePosition::VECCALC> bufQueue;
    LocalTensor<float> floatTensor;
    LocalTensor<T> xTensor;
    LocalTensor<T> x2Tensor;
    LocalTensor<int4b_t> yTensor;
    LocalTensor<int4b_t> y2Tensor;
    LocalTensor<float> qscaleTensor;
    LocalTensor<half> absTensor;

    event_t eventIdVToS;
    event_t eventIdVToMte2;
    event_t eventIdMte2ToMte3;
    event_t eventIdVToMte3;
    event_t eventIdMte3ToV;
    event_t eventIdMte3ToS;

    DEvent<PIPE_MTE2, PIPE_V> in_ready{EVENT_ID4, EVENT_ID5};
    DEvent<PIPE_V, PIPE_MTE2> in_empty{EVENT_ID4, EVENT_ID5};
    DEvent<PIPE_V, PIPE_MTE3> out_ready{EVENT_ID4, EVENT_ID5};
    DEvent<PIPE_MTE3, PIPE_V> out_empty{EVENT_ID4, EVENT_ID5};

    int64_t subBlockIdx = 0;
    int64_t splitRow = 0;
    int64_t splitCount = 0;
    int64_t zeroK = 0;
    int64_t zeroStartK = 0;
    int64_t zeroEndK = 0;
    float clipRatio = 0.0f;
};
} // namespace FlatQuantNS

#endif // FLAT_QUANT_VEC_H
