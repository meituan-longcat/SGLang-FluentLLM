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
 * \file tensor_utils.h
 * \brief
 */
#ifndef TENSOR_UTILS_H
#define TENSOR_UTILS_H

#pragma once
#include "kernel_operator.h"
#include "lib/matmul_intf.h"

using namespace AscendC;

namespace FlatQuantNS {
constexpr MatmulConfig MDL_CFG = GetMDLConfig(false, false, 0, false, false, false, true);
constexpr uint8_t MM_BASE_MODE = 1;
constexpr uint8_t MM_DOUBLE_MODE = 2;
constexpr uint8_t MM_SPLIT_MODE = 3;
constexpr uint8_t MM_HIGH_MODE = 4;

constexpr uint8_t SYNC_MODE0 = 0;
constexpr uint8_t SYNC_MODE1 = 1;
constexpr uint8_t SYNC_MODE2 = 2;
constexpr uint8_t CUBE_VEC_SYNC_ID = 0;
constexpr uint8_t VEC_CUBE_SYNC_ID = 4;
constexpr uint8_t VEC_SYNC_ID = 5;
constexpr uint8_t TWO_VEC_SYNC_ID = 6;

constexpr int32_t DOUBLE = 2;
constexpr int32_t NUM_TWO = 2;
constexpr int32_t NUM_EIGHT = 8;
constexpr int32_t CEIL_SIZE = 16;
constexpr int32_t BATCH_SIZE = 16;
constexpr int32_t UB_SIZE = 192 * 1024;
constexpr int32_t HIGH_UB_SIZE = 148 * 1024;
constexpr int32_t ONE_UB_SIZE = 128 * 1024;
constexpr int32_t L1_SIZE = 512 * 1024;
constexpr int32_t DATA_COUNT = 16384;
constexpr int32_t DATA_COUNT_ONE = 10240;
constexpr int32_t DATA_COUNT_ONE_HALF = DATA_COUNT_ONE >> 1;
constexpr int32_t DATA_COUNT_ONE_HALF_HALF = DATA_COUNT_ONE >> 2;
constexpr int32_t CAST_COUNT = 8192;
constexpr int32_t SCALE_COUNT = 2048;
constexpr int32_t BASE_SIZE = 128;
constexpr int32_t FLOAT_BASE_SIZE = 64;
constexpr int32_t MAX_REPEAT_TIMES = 255;
constexpr float NUM_FLOAT_SEVEN = 7.0f;
constexpr int32_t K_PER_VEC = 4; // batch number. Each loop processes K_PER_VEC*M*N
constexpr int32_t K_PER_VEC_ONE = 4;
constexpr int32_t K_DOUBLE_VEC = DOUBLE * K_PER_VEC;
constexpr int32_t LOG2_128 = 7;
constexpr int32_t LOG2_16 = 4;

struct FlatQuantShapeInfo {
    int64_t K;
    int64_t M;
    int64_t N; // basic shape
    int64_t perK;
    int64_t K1;
    int64_t K2; // loop start and loop end
    int64_t Mceil;
    int64_t Nceil; // ceil shape
    int64_t fractalM;
    int64_t fractalN;
    int64_t calFractalM;
    int64_t calFractalN;
    int64_t calM;
    int64_t calN;
};

struct MatmulInfo {
    int64_t splitCount;
    int64_t splitCount2;
    int64_t splitCount1;
};

#define aifunc __aicore__ inline

/* ------------- Events ------------- */

template <pipe_t p1, pipe_t p2>
class DEvent {
public:
    aifunc DEvent(event_t e_id1, event_t e_id2)
    {
        id1 = e_id1;
        id2 = e_id2;
    }
    aifunc void wait()
    {
        if ((wait_cnt & 1) == 0) {
            sync.WaitFlag(id1);
        } else {
            sync.WaitFlag(id2);
        }
        wait_cnt++;
    }
    aifunc void set()
    {
        if ((set_cnt & 1) == 0) {
            sync.SetFlag(id1);
        } else {
            sync.SetFlag(id2);
        }
        set_cnt++;
    }

    aifunc void setall()
    {
        set();
        set();
    }
    aifunc void release()
    {
        for (int i = wait_cnt; i < set_cnt; ++i) {
            wait();
        }
    }

private:
    TQueSync<p1, p2> sync;
    event_t id1 = (event_t)0;
    event_t id2 = (event_t)1;
    int wait_cnt = 0;
    int set_cnt = 0;
};

template <typename CType, typename DType>
__aicore__ inline void CalMatrix(LocalTensor<CType> c, LocalTensor<DType> a, LocalTensor<DType> b, uint16_t m,
                                 uint16_t k, uint16_t n, uint8_t unitFlag, bool kDirectionAlign, bool cmatrixSource,
                                 bool cmatrixInitVal)
{
    MmadParams mmadParams;
    mmadParams.m = m;
    mmadParams.n = n;
    mmadParams.k = k;
    mmadParams.cmatrixInitVal = cmatrixInitVal;
    mmadParams.cmatrixSource = cmatrixSource;
    mmadParams.unitFlag = unitFlag;
    Mmad(c, a, b, mmadParams);
}

template <typename T>
__aicore__ inline void CopyGmToL1(LocalTensor<T> dst, GlobalTensor<T> src, uint32_t realN, uint32_t realD,
                                  uint32_t ceilD)
{
    // nd2zz
    uint32_t tailN = realN % CEIL_SIZE;
    if (tailN < realN) {
        DataCopy(dst, src,
                 Nd2NzParams(realN / CEIL_SIZE, CEIL_SIZE, realD, CEIL_SIZE * realD, realD, CEIL_SIZE, 1,
                             CEIL_SIZE * ceilD));
    }
    if (tailN != 0) {
        int offsetN = realN / CEIL_SIZE * CEIL_SIZE;
        DataCopy(dst[offsetN * ceilD], src[offsetN * realD], Nd2NzParams(1, tailN, realD, 0, realD, CEIL_SIZE, 1, 0));
    }
}

template <typename T>
__aicore__ inline void CopyXToL1(LocalTensor<T> dst, GlobalTensor<T> src, bool useSlowCopy, FlatQuantShapeInfo shape)
{
    if (useSlowCopy) {
        CopyGmToL1(dst, src, shape.M, shape.N, shape.Nceil);
    } else {
        DataCopy(dst, src,
                 Nd2NzParams(shape.Mceil / CEIL_SIZE, CEIL_SIZE, shape.N, CEIL_SIZE * shape.N, shape.N, CEIL_SIZE, 1,
                             CEIL_SIZE * shape.Nceil));
    }
}

template <typename T>
__aicore__ inline void CalReduceMax(LocalTensor<T> srcTensor, int32_t len, event_t eventIdVToS)
{
    int32_t repeatTimes = len >> LOG2_128;
    if (repeatTimes > 1) {
        BinaryRepeatParams repeatParams = {1, 1, 1, 0, DEFAULT_REPEAT_STRIDE, 0};
        Max(srcTensor, srcTensor[BASE_SIZE], srcTensor, BASE_SIZE, repeatTimes - 1, repeatParams);
        PipeBarrier<PIPE_V>();
    }
    WholeReduceMax(srcTensor, srcTensor, BASE_SIZE, 1, 1, 1, DEFAULT_REPEAT_STRIDE, ReduceOrder::ORDER_ONLY_VALUE);
    SetFlag<HardEvent::V_S>(eventIdVToS);
    WaitFlag<HardEvent::V_S>(eventIdVToS);
}

template <HardEvent evt>
__aicore__ inline void SetEvtFlag()
{
    event_t eventFlag = static_cast<event_t>(GetTPipePtr()->FetchEventID(evt));
    SetFlag<evt>(eventFlag);
    WaitFlag<evt>(eventFlag);
}

struct DataCopyStruct {
    uint32_t blockCount = 0;
    uint32_t blockLen = 0;
    uint32_t srcStride = 0;
    uint32_t dstStride = 0;
    bool isPad = false;
    uint8_t leftPad = 0;
    uint8_t rightPad = 0;
};

template <typename T>
__aicore__ inline void DataCopyInContiguous(LocalTensor<T> dstTensor, GlobalTensor<T> srcTensor,
                                            DataCopyStruct dataCopyStruct, uint32_t tileLength)
{
    DataCopyExtParams dataCopyParams;
    DataCopyPadExtParams<T> padParams{dataCopyStruct.isPad, dataCopyStruct.leftPad, dataCopyStruct.rightPad, 0};
    dataCopyParams.blockLen = dataCopyStruct.blockLen;
    dataCopyParams.blockCount = dataCopyStruct.blockCount;
    dataCopyParams.srcStride = dataCopyStruct.srcStride;
    dataCopyParams.dstStride = dataCopyStruct.dstStride;
    DataCopyPad(dstTensor[(sizeof(T) == 2) * tileLength], srcTensor, dataCopyParams, padParams);
}

__aicore__ inline void CalReduceMaxOne(LocalTensor<half> srcTensor, int32_t rowNum, int32_t colAlign, int32_t colSize)
{
    int32_t repeatTimes = (colSize - 1) >> LOG2_128; // 除128
    uint8_t repeatStride = colAlign >> LOG2_16;
    BinaryRepeatParams repeatParams = {1, 1, 1, repeatStride, repeatStride, repeatStride};
    for (int64_t i = 1; i < repeatTimes; i++) {
        Max(srcTensor, srcTensor[i * BASE_SIZE], srcTensor, BASE_SIZE, rowNum, repeatParams);
        PipeBarrier<PIPE_V>();
    }
    Max(srcTensor, srcTensor[repeatTimes * BASE_SIZE], srcTensor, colSize - repeatTimes * BASE_SIZE, rowNum,
        repeatParams);
    PipeBarrier<PIPE_V>();
    WholeReduceMax(srcTensor, srcTensor, colSize < BASE_SIZE ? colSize : BASE_SIZE, rowNum, 1, 1, repeatStride,
                   ReduceOrder::ORDER_ONLY_VALUE);
    PipeBarrier<PIPE_V>();
}

__aicore__ inline int64_t GetQuantK(GlobalTensor<int64_t> groupListGM, int64_t groupNum, int64_t groupListType,
                                    int64_t maxK)
{
    int64_t quantK = 0;
    if (groupListType == 0) {
        quantK = groupListGM.GetValue(groupNum - 1);
    } else if (groupListType == 1) {
        for (int64_t i = 0; i < groupNum; i++) {
            quantK += groupListGM.GetValue(i);
        }
    } else {
        groupNum *= 2;
        for (int64_t i = 1; i < groupNum; i += 2) {
            quantK += groupListGM.GetValue(i);
        }
    }
    // Preserve the physical K for output sizing while bounding the compute prefix.
    // A zero total means every token is padding; it must not fall back to full K.
    if (quantK <= 0) {
        return 0;
    }
    return quantK < maxK ? quantK : maxK;
}

template <typename T>
__aicore__ inline void ClearGM(GlobalTensor<T> dst, LocalTensor<T> src, int64_t srcLength, int64_t dstLength)
{
    for (int64_t offset = 0; offset < dstLength; offset += srcLength) {
        int64_t copyLength = (offset + srcLength > dstLength) ? (dstLength - offset) : srcLength;
        if constexpr (std::is_same<T, int4b_t>::value) {
            DataCopyExtParams copyParams{1, static_cast<uint32_t>(copyLength >> 1), 0, 0, 0};
            DataCopyPad(dst[offset], src, copyParams);
        } else {
            DataCopyExtParams copyParams{1, static_cast<uint32_t>(copyLength * sizeof(T)), 0, 0, 0};
            DataCopyPad(dst[offset], src, copyParams);
        }
    }
}
} // namespace FlatQuantNS

#endif // TENSOR_UTILS_H
