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
#ifndef FLAT_QUANT_VEC_ONE_H
#define FLAT_QUANT_VEC_ONE_H

#include <cmath>
#include "tensor_utils.h"

namespace FlatQuantNS {
template <typename T>
class FlatQuantVecOne {
public:
    aifunc FlatQuantVecOne() {}
    aifunc void Init(GM_ADDR p1mtx_, GM_ADDR groupList_, GM_ADDR out_, GM_ADDR qscale_, GM_ADDR workspace_,
                     const FlatQuantTilingData* tilingData)
    {
        shape.M = tilingData->M;
        shape.N = tilingData->N;
        shape.K = tilingData->K;
        clipRatio = tilingData->clipRatio;

        p1GM.SetGlobalBuffer((__gm__ T*)p1mtx_);
        outGM.SetGlobalBuffer((__gm__ int4b_t*)out_);
        qscaleGM.SetGlobalBuffer((__gm__ float*)qscale_);
        outnzGM.SetGlobalBuffer((__gm__ T*)workspace_);
        if (tilingData->groupNum > 0) {
            groupListGM.SetGlobalBuffer((__gm__ int64_t*)groupList_);
            shape.K = GetQuantK(groupListGM, tilingData->groupNum, tilingData->groupListType, tilingData->K);
            if (tilingData->clearPadding != 0) {
                zeroK = tilingData->K - shape.K;
            }
        }
        tiling();

        pipe.InitBuffer(bufQueue, ONE_UB_SIZE);
        xTensor = bufQueue.Get<float>();
        x2Tensor = xTensor[DATA_COUNT_ONE_HALF];
        yTensor = x2Tensor[DATA_COUNT_ONE_HALF];
        y2Tensor = yTensor[DATA_COUNT_ONE];
        qscaleTensor = y2Tensor[DATA_COUNT_ONE];

        eventIdVToMte3 = static_cast<event_t>(pipe.FetchEventID(HardEvent::V_MTE3));
        eventIdMte3ToV = static_cast<event_t>(pipe.FetchEventID(HardEvent::MTE3_V));
    }

    aifunc void tiling()
    {
        int allTimes = GetBlockNum() * BATCH_SIZE; // 每个核处理16个批次, 控制同步计数器不会超过16

        int64_t oriK = shape.K;
        shape.Nceil = (shape.N + FLOAT_BASE_SIZE - 1) / FLOAT_BASE_SIZE * FLOAT_BASE_SIZE;
        shape.M = (((shape.K + GetBlockNum() - 1) / GetBlockNum()) + CEIL_SIZE - 1) / CEIL_SIZE *
                  CEIL_SIZE; // reshape K、M
        if (shape.M > BASE_SIZE) {
            shape.M = BASE_SIZE;
        }
        tailK = shape.K % shape.M == 0 ? shape.M : shape.K % shape.M;
        perM = shape.M;
        perKM = DATA_COUNT_ONE / shape.Nceil;
        if (perKM > MAX_REPEAT_TIMES) {
            perKM = MAX_REPEAT_TIMES;
        }
        perKM = perKM / CEIL_SIZE * CEIL_SIZE;
        perTailKM = perM % perKM;
        loopKM = perM / perKM;
        tailLoopKM = tailK / perKM;
        tailPerTailKM = tailK % perKM;
        shape.K = (shape.K + shape.M - 1) / shape.M;
        shape.perK = (shape.K + allTimes - 1) / allTimes; // 每个批次处理多少个K
        shape.perK = (shape.perK + K_PER_VEC_ONE - 1) / (K_PER_VEC_ONE) * (K_PER_VEC_ONE); // 和4对齐

        int k_per_core = ((shape.K + GetBlockNum() - 1) / GetBlockNum() + shape.perK - 1) / shape.perK * shape.perK;

        shape.K1 = k_per_core * (GetBlockIdx() / DOUBLE); // vector blk idx is 0~40
        shape.K2 = ((k_per_core + shape.K1) > shape.K) ? shape.K : (k_per_core + shape.K1);
        isLastK = (k_per_core + shape.K1) >= shape.K;
        shape.K = oriK;
        shape.M = 1;
        shape.Mceil = 1;

        if (zeroK > 0) {
            int64_t aivNum = GetBlockNum() * DOUBLE;
            int64_t singleCoreZeroK = (zeroK + aivNum - 1) / aivNum;
            zeroStartK = GetBlockIdx() * singleCoreZeroK;
            zeroEndK = ((zeroStartK + singleCoreZeroK) > zeroK) ? zeroK : (zeroStartK + singleCoreZeroK);
        }
    }

    aifunc void Process()
    {
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(1 * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        DataCopyPad(xTensor.template ReinterpretCast<T>()[CEIL_SIZE], p1GM, copyParams, padParams);
        SetEvtFlag<HardEvent::MTE2_V>();
        Cast(xTensor, xTensor.template ReinterpretCast<T>()[CEIL_SIZE], RoundMode::CAST_NONE, NUM_EIGHT);
        SetEvtFlag<HardEvent::V_S>();
        p1ValueCast = xTensor.GetValue(0);
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
        int64_t subBlockIdx = GetSubBlockIdx();
        for (int64_t startK = shape.K1; startK < shape.K2; startK += shape.perK) {
            int64_t endK = startK + shape.perK > shape.K2 ? shape.K2 : startK + shape.perK;
            bool isLast = shape.K - (endK - 1) * perM < perM;
            for (int64_t k = startK; k < endK; k++) {
                // 量化分给两个vec核，0核做偶数，1核做奇数
                if ((k & 1) == subBlockIdx) {
                    if (isLast && shape.K - k * perM < perM) {
                        MultiQuantTail(k, scaleK, isLast);
                    } else {
                        MultiQuant(k, scaleK, isLast);
                    }
                }
            }
            if (isLast) {
                CopyOutQuant(scaleK * perM, (endK - scaleK - 1) * perM + tailK);
                break;
            } else if (endK == shape.K2 || (endK + shape.perK) * perM > scaleK * perM + SCALE_COUNT) {
                CopyOutQuant(scaleK * perM, (endK - scaleK) * perM);
                scaleK = endK;
            }
        }
        ProcessZeroK();
        in_empty.release();
        out_empty.release();
    }

    aifunc void ProcessZeroK()
    {
        if (zeroStartK >= zeroEndK) {
            return;
        }
        SetFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
        WaitFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
        Duplicate<float>(xTensor, (float)0, DATA_COUNT_ONE);
        Duplicate<float>(qscaleTensor, (float)0, SCALE_COUNT);
        SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        int64_t zeroOffset = (shape.K + zeroStartK) * shape.M * shape.N;
        int64_t zeroCount = (zeroEndK - zeroStartK) * shape.M * shape.N;
        ClearGM(outGM[zeroOffset], xTensor.template ReinterpretCast<int4b_t>(), DATA_COUNT_ONE * sizeof(float) * DOUBLE,
                zeroCount);
        ClearGM(qscaleGM[shape.K + zeroStartK], qscaleTensor, SCALE_COUNT, zeroEndK - zeroStartK);
    }

    aifunc void ClearQuant()
    {
        // pre-set all buffers to zero
        Duplicate<float>(xTensor, (float)0, DATA_COUNT_ONE);
        Duplicate<float>(x2Tensor, (float)0, DATA_COUNT_ONE);
        Duplicate<float>(qscaleTensor, (float)0, SCALE_COUNT);

        SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        int64_t endKPre = isLastK ? (shape.K2 - 1) * perM + tailK : shape.K2 * perM;
        int64_t midK = (endKPre - shape.K1 * perM) / 2 + shape.K1 * perM;
        int64_t startK = GetSubBlockIdx() == 0 ? shape.K1 * perM : midK;
        int64_t endK = GetSubBlockIdx() == 0 ? midK : endKPre;
        for (int64_t k = startK; k < endK; k += SCALE_COUNT) {
            int64_t len = endK - k > SCALE_COUNT ? SCALE_COUNT : endK - k;
            DataCopyExtParams copyParams{1, static_cast<uint32_t>(len * sizeof(float)), 0, 0, 0};
            DataCopyPad(qscaleGM[k], qscaleTensor, copyParams);
        }
        SetFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
        WaitFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
    }

    aifunc void CopyOutQuant(int64_t scaleK, int64_t scaleCount)
    {
        SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        SetAtomicAdd<float>();
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(scaleCount * sizeof(float)), 0, 0, 0};
        DataCopyPad(qscaleGM[scaleK], qscaleTensor, copyParams);
        SetAtomicNone();
        SetFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
        WaitFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
    }

    aifunc void MultiQuant(int64_t k, int64_t scaleK, bool isLast)
    {
        if (k == shape.K1 || k == shape.K1 + 1 || k == shape.K1 + shape.perK || k % shape.perK == 0 ||
            k % shape.perK == 1) {
            CrossCoreWaitFlag(CUBE_VEC_SYNC_ID);
        }
        for (int64_t i = 0; i < loopKM; i++) {
            Quant(k * perM + i * perKM, scaleK * perM, isLast, perKM);
        }
        Quant(k * perM + loopKM * perKM, scaleK * perM, isLast, perTailKM);
        return;
    }

    aifunc void MultiQuantTail(int64_t k, int64_t scaleK, bool isLast)
    {
        if (k == shape.K1 || k == shape.K1 + 1 || k == shape.K1 + shape.perK || k % shape.perK == 0 ||
            k % shape.perK == 1) {
            CrossCoreWaitFlag(CUBE_VEC_SYNC_ID);
        }
        for (int64_t i = 0; i < tailLoopKM; i++) {
            Quant(k * perM + i * perKM, scaleK * perM, isLast, perKM);
        }
        Quant(k * perM + tailLoopKM * perKM, scaleK * perM, isLast, tailPerTailKM);
        return;
    }

    aifunc void Quant(int64_t k, int64_t scaleK, bool isLast, uint32_t rowNum)
    {
        LocalTensor<float> inTensor = GetXTensor(count);
        LocalTensor<float> outTensorFloat = GetYTensor(count).template ReinterpretCast<float>();
        LocalTensor<int4b_t> outTensor = GetYTensor(count).template ReinterpretCast<int4b_t>();
        LocalTensor<half> outTensorHalf = GetYTensor(count).template ReinterpretCast<half>();
        int64_t fullCount = rowNum * shape.Nceil;
        uint8_t repeatStride = shape.Nceil >> (LOG2_16 - 1);
        in_empty.wait();
        DataCopyStruct dataCopyStruct{shape.N == shape.Nceil ? 1 : rowNum,
                                      shape.N == shape.Nceil ?
                                          static_cast<uint32_t>(rowNum * shape.Mceil * shape.N * sizeof(T)) :
                                          static_cast<uint32_t>(shape.Mceil * shape.N * sizeof(T)),
                                      0,
                                      static_cast<uint32_t>((shape.Nceil - shape.N) / CEIL_SIZE),
                                      shape.N != shape.Nceil,
                                      0,
                                      static_cast<uint8_t>((shape.Nceil - shape.N) % 16)};
        DataCopyInContiguous(inTensor.template ReinterpretCast<T>(), outnzGM[k * shape.Mceil * shape.N], dataCopyStruct,
                             0);
        in_ready.set();

        out_empty.wait();
        in_ready.wait();
        Cast(outTensorFloat, inTensor.template ReinterpretCast<T>(), RoundMode::CAST_NONE, fullCount);
        PipeBarrier<PIPE_V>();

        Muls(outTensorFloat, outTensorFloat, p1ValueCast, fullCount);
        PipeBarrier<PIPE_V>();

        // 这边来回转是为了对齐标杆
        Cast(inTensor.template ReinterpretCast<half>(), outTensorFloat, RoundMode::CAST_NONE, fullCount);
        PipeBarrier<PIPE_V>();
        Cast(outTensorFloat, inTensor.template ReinterpretCast<half>(), RoundMode::CAST_NONE, fullCount);
        PipeBarrier<PIPE_V>();

        Abs(inTensor.template ReinterpretCast<half>(), inTensor.template ReinterpretCast<half>(), fullCount);
        PipeBarrier<PIPE_V>();
        CalReduceMaxOne(inTensor.template ReinterpretCast<half>(), rowNum, shape.Nceil, shape.N);
        Cast(inTensor[DATA_COUNT_ONE_HALF_HALF], inTensor.template ReinterpretCast<half>(), RoundMode::CAST_NONE,
             rowNum);
        PipeBarrier<PIPE_V>();
        Muls(qscaleTensor[k - scaleK], inTensor[DATA_COUNT_ONE_HALF_HALF], clipRatio / NUM_FLOAT_SEVEN, rowNum);
        PipeBarrier<PIPE_V>();

        uint32_t brcbRepeat = (rowNum + NUM_EIGHT - 1) / NUM_EIGHT;
        Brcb(inTensor[DATA_COUNT_ONE_HALF_HALF], qscaleTensor[k - scaleK], brcbRepeat, {1, 8});
        PipeBarrier<PIPE_V>();
        int32_t repeatTimes = shape.Nceil >> (LOG2_128 - 1); // 除64
        BinaryRepeatParams repeatParams = {1, 1, 0, repeatStride, repeatStride, 1};
        for (int64_t i = 0; i < repeatTimes; i++) {
            Div(outTensorFloat[FLOAT_BASE_SIZE * i], outTensorFloat[FLOAT_BASE_SIZE * i],
                inTensor[DATA_COUNT_ONE_HALF_HALF], FLOAT_BASE_SIZE, rowNum, repeatParams);
            PipeBarrier<PIPE_V>();
        }
        Div(outTensorFloat[FLOAT_BASE_SIZE * repeatTimes], outTensorFloat[FLOAT_BASE_SIZE * repeatTimes],
            inTensor[DATA_COUNT_ONE_HALF_HALF], shape.Nceil % FLOAT_BASE_SIZE, rowNum, repeatParams);
        PipeBarrier<PIPE_V>();
        Cast(outTensorHalf, outTensorFloat, RoundMode::CAST_NONE, fullCount);
        PipeBarrier<PIPE_V>();
        Cast(outTensor, outTensorHalf, RoundMode::CAST_NONE, fullCount);

        out_ready.set();
        in_empty.set();

        out_ready.wait();
        DataCopyExtParams copyParams{
            static_cast<uint16_t>(shape.Nceil == shape.N ? 1 : rowNum),
            static_cast<uint16_t>(shape.Nceil == shape.N ? (uint32_t)fullCount / DOUBLE :
                                                           (uint32_t)(shape.M * shape.N) / DOUBLE),
            0, 0, 0};
        DataCopyPad(outGM[k * shape.M * shape.N], outTensor, copyParams);
        out_empty.set();
        count++;
    }

    __aicore__ inline LocalTensor<float> GetXTensor(int64_t k) { return ((k & 1) == 0) ? xTensor : x2Tensor; };

    __aicore__ inline LocalTensor<float> GetYTensor(int64_t k) { return ((k & 1) == 0) ? yTensor : y2Tensor; };

private:
    TPipe pipe;
    FlatQuantShapeInfo shape;
    GlobalTensor<T> p1GM;
    GlobalTensor<int64_t> groupListGM;
    GlobalTensor<int4b_t> outGM;
    GlobalTensor<float> qscaleGM;
    GlobalTensor<T> outnzGM;
    GlobalTensor<T> doubleP1GM;

    TBuf<QuePosition::VECCALC> bufQueue;
    LocalTensor<float> xTensor;
    LocalTensor<float> x2Tensor;
    LocalTensor<float> yTensor;
    LocalTensor<float> y2Tensor;
    LocalTensor<float> qscaleTensor;

    event_t eventIdVToMte3;
    event_t eventIdMte3ToV;

    DEvent<PIPE_MTE2, PIPE_V> in_ready{EVENT_ID4, EVENT_ID5};
    DEvent<PIPE_V, PIPE_MTE2> in_empty{EVENT_ID4, EVENT_ID5};
    DEvent<PIPE_V, PIPE_MTE3> out_ready{EVENT_ID4, EVENT_ID5};
    DEvent<PIPE_MTE3, PIPE_V> out_empty{EVENT_ID4, EVENT_ID5};

    int64_t zeroK = 0;
    int64_t zeroStartK = 0;
    int64_t zeroEndK = 0;
    float clipRatio = 0.0f;
    uint32_t tailK = 0;
    float p1ValueCast = 0.0;
    int64_t perM = 0;
    uint32_t perKM = 0;
    uint32_t perTailKM = 0;
    uint32_t loopKM = 0;
    uint32_t tailLoopKM = 0;
    uint32_t tailPerTailKM = 0;
    uint32_t count = 0;
    bool isLastK = false;
};
} // namespace FlatQuantNS

#endif // FLAT_QUANT_VEC_H
