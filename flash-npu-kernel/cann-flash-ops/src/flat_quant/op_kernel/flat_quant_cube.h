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
 * \file flat_quant_cube.h
 * \brief
 */
#ifndef FLAT_QUANT_CUBE_H
#define FLAT_QUANT_CUBE_H

#include "tensor_utils.h"

namespace FlatQuantNS {
template <typename T, uint8_t MM_MODE>
class FlatQuantCube {
public:
    aifunc FlatQuantCube() {}
    aifunc void Init(GM_ADDR xmtx_, GM_ADDR p1mtx_, GM_ADDR p2mtx_, GM_ADDR groupList_, GM_ADDR workspace_,
                     const FlatQuantTilingData* tilingData)
    {
        shape.M = tilingData->M;
        shape.N = tilingData->N;
        shape.K = tilingData->K;
        groupListGM.SetGlobalBuffer((__gm__ int64_t*)groupList_);
        if (tilingData->groupNum > 0) {
            shape.K = GetQuantK(groupListGM, tilingData->groupNum, tilingData->groupListType, tilingData->K);
        }
        tiling();

        xGM.SetGlobalBuffer((__gm__ T*)xmtx_);
        p1GM.SetGlobalBuffer((__gm__ T*)p1mtx_);
        p2GM.SetGlobalBuffer((__gm__ T*)p2mtx_);
        outnzGM.SetGlobalBuffer((__gm__ half*)workspace_);
        doubleP1GM.SetGlobalBuffer(
            (__gm__ T*)(workspace_ + (shape.K + shape.K % 2) * shape.Mceil * shape.N * sizeof(T)));

        SetFixpipeNz2ndFlag(1, 1, shape.Nceil);
        pipe.InitBuffer(l1Buf, L1_SIZE);
        xTensor = l1Buf.Get<T>();
        x2Tensor = xTensor[shape.calM * shape.Nceil];
        yTensor = x2Tensor[shape.calM * shape.Nceil];
        y2Tensor = yTensor[shape.calM * shape.Nceil];
        p1Tensor = y2Tensor[shape.calM * shape.Nceil];
        p2Tensor = p1Tensor[shape.Mceil * shape.Mceil];

        pipe.InitBuffer(l0aBuf, DOUBLE * DATA_COUNT * sizeof(T));
        aTensor = l0aBuf.Get<T>();
        a2Tensor = aTensor[DATA_COUNT];

        pipe.InitBuffer(l0bBuf, DOUBLE * DATA_COUNT * sizeof(T));
        bTensor = l0bBuf.Get<T>();
        b2Tensor = bTensor[DATA_COUNT];

        pipe.InitBuffer(l0cBuf, DOUBLE * DATA_COUNT * sizeof(float));
        cTensor = l0cBuf.Get<float>();
        c2Tensor = cTensor[DATA_COUNT];
    }

    aifunc void tiling()
    {
        int allTimes = GetBlockNum() * 16; // 每个核处理16个批次, 控制同步计数器不会超过16
        shape.perK = (shape.K + allTimes - 1) / allTimes;                      // 每个批次处理多少个K
        shape.perK = (shape.perK + K_PER_VEC - 1) / (K_PER_VEC) * (K_PER_VEC); // 和4对齐

        int k_per_core = ((shape.K + GetBlockNum() - 1) / GetBlockNum() + shape.perK - 1) / shape.perK * shape.perK;
        shape.K1 = k_per_core * (GetBlockIdx()); // cube blk idx is 0~20
        shape.K2 = ((k_per_core + shape.K1) > shape.K) ? shape.K : (k_per_core + shape.K1);
        shape.fractalM = (shape.M + CEIL_SIZE - 1) / CEIL_SIZE;
        shape.fractalN = (shape.N + CEIL_SIZE - 1) / CEIL_SIZE;
        shape.Mceil = shape.fractalM * CEIL_SIZE;
        shape.Nceil = shape.fractalN * CEIL_SIZE;
        shape.calM = shape.Mceil;
        shape.calN = shape.Nceil;
        shape.calFractalM = shape.fractalM;
        shape.calFractalN = shape.fractalN;

        if constexpr (MM_MODE == MM_SPLIT_MODE) {
            int64_t splitM = (shape.Mceil + BASE_SIZE - 1) / BASE_SIZE;
            int64_t splitN = (shape.Nceil + BASE_SIZE - 1) / BASE_SIZE;
            matmulInfo.splitCount = splitM * splitN;
            matmulInfo.splitCount2 = matmulInfo.splitCount * splitN;
            matmulInfo.splitCount1 = matmulInfo.splitCount * splitM;
        }
        if constexpr (MM_MODE == MM_DOUBLE_MODE) {
            shape.calM = shape.Mceil << 1;
            shape.calN = shape.Nceil << 1;
            shape.calFractalM = shape.fractalM << 1;
            shape.calFractalN = shape.fractalN << 1;
        }
        if (shape.K * shape.M - shape.Mceil < 0) {
            invalidK = -1;
        } else {
            invalidK = (shape.K * shape.M - shape.Mceil) / shape.M;
        }
    }

    aifunc void Process()
    {
        l1empty.setall();
        l0empty.setall();
        outempty.setall();
        // set zero for L1
        int dataCount = shape.Mceil * shape.Mceil + shape.Nceil * shape.Nceil +
                        DOUBLE * DOUBLE * shape.calM * shape.Nceil;
        InitConstValue(xTensor, {1, static_cast<uint16_t>(dataCount / CEIL_SIZE), 0, (T)0});
        AscendC::PipeBarrier<PIPE_MTE2>();
        // Preload P1,P2
        CrossCoreWaitFlag(VEC_CUBE_SYNC_ID);
        if constexpr (MM_MODE == MM_DOUBLE_MODE) {
            CopyGmToL1(p1Tensor, doubleP1GM, shape.Mceil, shape.Mceil, shape.Mceil);
        } else {
            CopyGmToL1(p1Tensor, p1GM, shape.M, shape.M, shape.Mceil);
        }
        CopyGmToL1(p2Tensor, p2GM, shape.N, shape.N, shape.Nceil);

        for (int64_t startK = shape.K1; startK < shape.K2; startK += shape.perK) {
            int64_t endK = (startK + shape.perK > shape.K2) ? shape.K2 : (startK + shape.perK);
            if constexpr (MM_MODE == MM_SPLIT_MODE) {
                for (int64_t k = startK; k < endK; ++k) {
                    ProcessSplitK(k);
                }
            } else if constexpr (MM_MODE == MM_DOUBLE_MODE) {
                for (int64_t k = startK; k < endK; k += DOUBLE) {
                    ProcessDoubleK(k, k >> 1);
                }
            } else {
                for (int64_t k = startK; k < endK; ++k) {
                    ProcessK(k);
                }
            }
            CrossCoreSetFlag<SYNC_MODE2, PIPE_FIX>(CUBE_VEC_SYNC_ID);
        }

        l1empty.release();
        l0empty.release();
        outempty.release();
    }

    aifunc void ProcessSplitK(int64_t k)
    {
        if (k == shape.K1) {
            l1empty.wait();
            CopyXToL1(GetXTensor(k), xGM[k * shape.M * shape.N], k > invalidK, shape);
            l1ready.set();
        }
        int64_t nextK = k + 1;
        if (nextK < shape.K2) {
            l1empty.wait();
            CopyXToL1(GetXTensor(nextK), xGM[nextK * shape.M * shape.N], nextK > invalidK, shape);
            l1ready.set();
        }
        ProcessSplitXP2(k);
        ProcessSplitP1X(k);
    }

    aifunc void ProcessSplitXP2(int64_t k)
    {
        int64_t c = (k - shape.K1) * (matmulInfo.splitCount + matmulInfo.splitCount);
        int64_t p = (k - shape.K1) * (matmulInfo.splitCount2 + matmulInfo.splitCount1);
        l1ready.wait();
        for (int32_t mIdx = 0; mIdx < shape.Mceil; mIdx += BASE_SIZE) {
            int32_t numM = (shape.Mceil - mIdx < BASE_SIZE) ? shape.Mceil - mIdx : BASE_SIZE;
            for (int32_t nIdx = 0; nIdx < shape.Nceil; nIdx += BASE_SIZE) {
                int32_t numN = (shape.Nceil - nIdx < BASE_SIZE) ? shape.Nceil - nIdx : BASE_SIZE;
                outempty.wait();
                for (int32_t kIdx = 0; kIdx < shape.Nceil; kIdx += BASE_SIZE) {
                    int32_t numK = (shape.Nceil - kIdx < BASE_SIZE) ? shape.Nceil - kIdx : BASE_SIZE;
                    l0empty.wait();
                    LoadDataSplitXP2(numM, mIdx, numN, nIdx, numK, kIdx, p, k);
                    l0ready.set();
                    l0ready.wait();
                    CalMatrix(GetCTensor(c), GetATensor(p), GetBTensor(p), numM, numK, numN,
                              kIdx + numK == shape.Nceil ? UFMode3 : UFMode2, false, false, kIdx == 0);
                    PipeBarrier<PIPE_M>();
                    l0empty.set();
                    p++;
                }

                QuantMode_t quantMode = F322F16;
                if constexpr (std::is_same<T, bfloat16_t>::value) {
                    quantMode = F322BF16;
                }
                DataCopyCO12DstParams dataCopyParams(numN, numM, shape.Mceil, numM, quantMode, 0, false, false);
                dataCopyParams.unitFlag = UFMode3;
                DataCopy(GetYTensor(k)[nIdx * shape.Mceil + mIdx * CEIL_SIZE], GetCTensor(c), dataCopyParams);
                outempty.set();
                c++;
            }
        }
        l1empty.set();
        x2ready.set();
    }

    aifunc void LoadDataSplitXP2(int32_t numM, int32_t mIdx, int32_t numN, int32_t nIdx, int32_t numK, int32_t kIdx,
                                 int64_t p, int64_t k)
    {
#if defined(__CCE_AICORE__) && (__CCE_AICORE__ == 310)
        int32_t fractalNumM = numM / CEIL_SIZE;
        int32_t fractalNumN = numN / CEIL_SIZE;
        int32_t fractalNumK = numK / CEIL_SIZE;
        // Zz => Nz
        for (int32_t sIdx = 0; sIdx < fractalNumM; sIdx++) {
            LoadData(GetATensor(p)[sIdx * CEIL_SIZE * CEIL_SIZE],
                     GetXTensor(k)[mIdx * shape.Nceil + kIdx * CEIL_SIZE + sIdx * shape.Nceil * CEIL_SIZE],
                     LoadData2DParamsV2(0, 0, 1, fractalNumK, 1, fractalNumM, false, 0));
        }
        // Zz => Zn
        for (int32_t sIdx = 0; sIdx < fractalNumK; sIdx++) {
            LoadData(GetBTensor(p)[sIdx * numN * CEIL_SIZE],
                     p2Tensor[kIdx * shape.Nceil + nIdx * CEIL_SIZE + sIdx * shape.Nceil * CEIL_SIZE],
                     LoadData2DParamsV2(0, 0, 1, fractalNumN, 1, 1, true, 0));
        }
#else
        uint16_t baseIdx = mIdx * shape.fractalN / CEIL_SIZE + kIdx / CEIL_SIZE;
        // Zz => Zz
        for (int32_t sIdx = 0; sIdx < numM / CEIL_SIZE; sIdx++) {
            LoadData(GetATensor(p)[sIdx * numK * CEIL_SIZE], GetXTensor(k),
                     LoadData2DParams(baseIdx, numK / CEIL_SIZE, 1, 0, 0, false, 0));
            baseIdx += shape.fractalN;
        }
        baseIdx = kIdx * shape.fractalN / CEIL_SIZE + nIdx / CEIL_SIZE;
        // Zz => Zn
        for (int32_t sIdx = 0; sIdx < numK / CEIL_SIZE; sIdx++) {
            LoadData(GetBTensor(p)[sIdx * numN * CEIL_SIZE], p2Tensor,
                     LoadData2DParams(baseIdx, numN / CEIL_SIZE, 1, 0, 0, true, 0));
            baseIdx += shape.fractalN;
        }
#endif
    }

    aifunc void ProcessSplitP1X(int64_t k)
    {
        int64_t c = (k - shape.K1) * (matmulInfo.splitCount + matmulInfo.splitCount) + matmulInfo.splitCount;
        int64_t p = (k - shape.K1) * (matmulInfo.splitCount2 + matmulInfo.splitCount1) + matmulInfo.splitCount2;
        x2ready.wait();

        for (int32_t mIdx = 0; mIdx < shape.Mceil; mIdx += BASE_SIZE) {
            int32_t numM = (shape.Mceil - mIdx < BASE_SIZE) ? shape.Mceil - mIdx : BASE_SIZE;
            int32_t realM = (shape.M - mIdx < BASE_SIZE) ? shape.M - mIdx : BASE_SIZE;
            for (int32_t nIdx = 0; nIdx < shape.Nceil; nIdx += BASE_SIZE) {
                int32_t numN = (shape.Nceil - nIdx < BASE_SIZE) ? shape.Nceil - nIdx : BASE_SIZE;
                int32_t realN = (shape.N - nIdx < BASE_SIZE) ? shape.N - nIdx : BASE_SIZE;
                outempty.wait();
                for (int32_t kIdx = 0; kIdx < shape.Mceil; kIdx += BASE_SIZE) {
                    int32_t numK = (shape.Mceil - kIdx < BASE_SIZE) ? shape.Mceil - kIdx : BASE_SIZE;
                    int32_t realK = (shape.M - kIdx < BASE_SIZE) ? shape.M - kIdx : BASE_SIZE;
                    l0empty.wait();
                    LoadDataSplitP1X(numM, mIdx, numN, nIdx, numK, kIdx, p, k);
                    l0ready.set();
                    l0ready.wait();
                    CalMatrix(GetCTensor(c), GetATensor(p), GetBTensor(p), realM, realK, realN,
                              kIdx + numK == shape.Mceil ? UFMode3 : UFMode2, false, false, kIdx == 0);
                    PipeBarrier<PIPE_M>();
                    l0empty.set();
                    p++;
                }
                DataCopyCO12DstParams dataCopyParams(realN, numM, shape.N, numM, F322F16, 0, false,
                                                     true); // enable NZ2ND
                dataCopyParams.unitFlag = UFMode3;
                DataCopy(outnzGM[k * shape.Mceil * shape.N + mIdx * shape.N + nIdx], GetCTensor(c), dataCopyParams);
                outempty.set();
                c++;
            }
        }
    }

    aifunc void LoadDataSplitP1X(int32_t numM, int32_t mIdx, int32_t numN, int32_t nIdx, int32_t numK, int32_t kIdx,
                                 int64_t p, int64_t k)
    {
#if defined(__CCE_AICORE__) && (__CCE_AICORE__ == 310)
        int32_t fractalNumM = numM / CEIL_SIZE;
        int32_t fractalNumN = numN / CEIL_SIZE;
        int32_t fractalNumK = numK / CEIL_SIZE;
        // Zz => Nz
        for (int32_t sIdx = 0; sIdx < fractalNumM; sIdx++) {
            LoadData(GetATensor(p)[sIdx * CEIL_SIZE * CEIL_SIZE],
                     p1Tensor[mIdx * shape.Mceil + kIdx * CEIL_SIZE + sIdx * shape.Mceil * CEIL_SIZE],
                     LoadData2DParamsV2(0, 0, 1, fractalNumK, 1, fractalNumM, false, 0));
        }
        // Nz => Zn
        for (int32_t mblk = 0; mblk < fractalNumK; ++mblk) {
            LoadData(GetBTensor(p)[mblk * numN * CEIL_SIZE],
                     GetYTensor(k)[nIdx * shape.Mceil + kIdx * CEIL_SIZE + mblk * CEIL_SIZE * CEIL_SIZE],
                     LoadData2DParamsV2(0, 0, 1, fractalNumN, shape.fractalM, 1, true, 0));
        }
#else
        uint16_t baseIdx = mIdx * shape.fractalM / CEIL_SIZE + kIdx / CEIL_SIZE;
        // Zz => Zz
        for (int32_t sIdx = 0; sIdx < numM / CEIL_SIZE; sIdx++) {
            LoadData(GetATensor(p)[sIdx * numK * CEIL_SIZE], p1Tensor,
                     LoadData2DParams(baseIdx, numK / CEIL_SIZE, 1, 0, 0, false, 0));
            baseIdx += shape.fractalM;
        }
        // Nz => Zn
        for (int mblk = 0; mblk < numK / CEIL_SIZE; ++mblk) {
            LoadData(GetBTensor(p)[mblk * numN * CEIL_SIZE],
                     GetYTensor(k)[nIdx * shape.Mceil + kIdx * CEIL_SIZE + mblk * CEIL_SIZE * CEIL_SIZE],
                     LoadData2DParams(0, numN / CEIL_SIZE, shape.fractalM, 0, 0, true, 0));
        }
#endif
    }

    aifunc void ProcessK(int64_t k)
    {
        if (k == shape.K1) {
            l1empty.wait();
            CopyXToL1(GetXTensor(k), xGM[k * shape.M * shape.N], k > invalidK, shape);
            l1ready.set();
            ProcessXP2(k);
        }
        int64_t nextK = k + 1;
        if (nextK < shape.K2) {
            l1empty.wait();
            CopyXToL1(GetXTensor(nextK), xGM[nextK * shape.M * shape.N], nextK > invalidK, shape);
            l1ready.set();
            ProcessXP2(nextK);
        }
        ProcessP1X(k, k * shape.Mceil * shape.N);
    }

    aifunc void ProcessDoubleK(int64_t k, int64_t p)
    {
        int64_t nextK = k + 1;
        if (k == shape.K1) {
            l1empty.wait();
            CopyXToL1(GetXTensor(p), xGM[k * shape.M * shape.N], k > invalidK, shape);
            if (nextK < shape.K2) {
                CopyXToL1(GetXTensor(p)[shape.Mceil * shape.Nceil], xGM[nextK * shape.M * shape.N], nextK > invalidK,
                          shape);
            }
            l1ready.set();
            ProcessXP2(p);
        }
        nextK++;
        if (nextK < shape.K2) {
            l1empty.wait();
            CopyXToL1(GetXTensor(p + 1), xGM[nextK * shape.M * shape.N], nextK > invalidK, shape);
            nextK++;
            if (nextK < shape.K2) {
                CopyXToL1(GetXTensor(p + 1)[shape.Mceil * shape.Nceil], xGM[nextK * shape.M * shape.N],
                          nextK > invalidK, shape);
            }
            l1ready.set();
            ProcessXP2(p + 1);
        }
        ProcessP1X(p, k * shape.Mceil * shape.N);
    }

    aifunc void ProcessXP2(int64_t p)
    {
        l1ready.wait();
        l0empty.wait();
#if defined(__CCE_AICORE__) && (__CCE_AICORE__ == 310)
        // Zz => Nz
        for (int32_t mblk = 0; mblk < shape.calFractalM; ++mblk) {
            LoadData(GetATensor(p)[mblk * CEIL_SIZE * CEIL_SIZE], GetXTensor(p)[mblk * shape.Nceil * CEIL_SIZE],
                     LoadData2DParamsV2(0, 0, 1, shape.fractalN, 1, shape.calFractalM, false, 0));
        }
        // Zz => Zn
        for (int32_t nblk = 0; nblk < shape.fractalN; ++nblk) {
            LoadData(GetBTensor(p)[nblk * shape.Nceil * CEIL_SIZE], p2Tensor[nblk * shape.Nceil * CEIL_SIZE],
                     LoadData2DParamsV2(0, 0, 1, shape.fractalN, 1, 1, true, 0));
        }
#else
        // Zz => Zz
        LoadData(GetATensor(p), GetXTensor(p),
                 LoadData2DParams(0, shape.calFractalM * shape.fractalN, 1, 0, 0, false, 0));
        // Zz => Zn
        LoadData(GetBTensor(p), p2Tensor, LoadData2DParams(0, shape.fractalN * shape.fractalN, 1, 0, 0, true, 0));
#endif
        l0ready.set();
        l1empty.set();

        outempty.wait();
        l0ready.wait();
        CalMatrix(GetCTensor(p), GetATensor(p), GetBTensor(p), shape.calM, shape.Nceil, shape.Nceil, UFMode3, false,
                  false, true);

        QuantMode_t quantMode = F322F16;
        if constexpr (std::is_same<T, bfloat16_t>::value) {
            quantMode = F322BF16;
        }
        DataCopyCO12DstParams dataCopyParams(shape.calM, shape.Nceil, shape.Nceil, shape.Nceil, quantMode, 0, false,
                                             false);
        dataCopyParams.unitFlag = UFMode3;
        DataCopy(GetYTensor(p), GetCTensor(p), dataCopyParams);
        x2ready.set();
    }

    aifunc void ProcessP1X(int64_t p, int64_t outOffset)
    {
        x2ready.wait();
#if defined(__CCE_AICORE__) && (__CCE_AICORE__ == 310)
        // Zz => Nz
        for (int32_t mblk = 0; mblk < shape.fractalM; ++mblk) {
            LoadData(GetATensor(p)[mblk * CEIL_SIZE * CEIL_SIZE], p1Tensor[mblk * shape.Mceil * CEIL_SIZE],
                     LoadData2DParamsV2(0, 0, 1, shape.fractalM, 1, shape.fractalM, false, 0));
        }
        // Nz => Zn
        for (int32_t mblk = 0; mblk < shape.fractalM; ++mblk) {
            LoadData(GetBTensor(p)[mblk * CEIL_SIZE * shape.calN], GetYTensor(p)[mblk * CEIL_SIZE * CEIL_SIZE],
                     LoadData2DParamsV2(0, 0, 1, shape.calFractalN, shape.fractalM, 1, true, 0));
        }
#else
        // Zz => Zz
        LoadData(GetATensor(p), p1Tensor, LoadData2DParams(0, shape.fractalM * shape.fractalM, 1, 0, 0, false, 0));
        // Nz => Zn
        for (int mblk = 0; mblk < shape.fractalM; ++mblk) {
            LoadData(GetBTensor(p)[mblk * CEIL_SIZE * shape.calN], GetYTensor(p)[mblk * CEIL_SIZE * CEIL_SIZE],
                     LoadData2DParams(0, shape.calFractalN, shape.fractalM, 0, 0, true, 0));
        }
#endif
        l0ready.set();
        l0ready.wait();
        CalMatrix(GetCTensor(p), GetATensor(p), GetBTensor(p), shape.M, shape.M, shape.calN, UFMode3, false, false,
                  true);
        l0empty.set();

        DataCopyCO12DstParams dataCopyParams(shape.N, shape.calM, shape.N, shape.calM, F322F16, 0, false, true);
        dataCopyParams.unitFlag = UFMode3;
        DataCopy(outnzGM[outOffset], GetCTensor(p), dataCopyParams);
        outempty.set();
    }

    __aicore__ inline LocalTensor<T> GetXTensor(int64_t k) { return ((k & 1) == 0) ? xTensor : x2Tensor; };

    __aicore__ inline LocalTensor<T> GetYTensor(int64_t k) { return ((k & 1) == 0) ? yTensor : y2Tensor; };

    __aicore__ inline LocalTensor<T> GetATensor(int64_t k) { return ((k & 1) == 0) ? aTensor : a2Tensor; };

    __aicore__ inline LocalTensor<T> GetBTensor(int64_t k) { return ((k & 1) == 0) ? bTensor : b2Tensor; };

    __aicore__ inline LocalTensor<float> GetCTensor(int64_t k) { return ((k & 1) == 0) ? cTensor : c2Tensor; };

private:
    TPipe pipe;
    FlatQuantShapeInfo shape;
    MatmulInfo matmulInfo;
    GlobalTensor<T> xGM;
    GlobalTensor<T> p1GM;
    GlobalTensor<T> p2GM;
    GlobalTensor<int64_t> groupListGM;
    GlobalTensor<half> outnzGM;
    GlobalTensor<T> doubleP1GM;

    TBuf<TPosition::A1> l1Buf;
    TBuf<TPosition::A2> l0aBuf;
    TBuf<TPosition::B2> l0bBuf;
    TBuf<TPosition::CO1> l0cBuf;

    LocalTensor<T> xTensor;
    LocalTensor<T> x2Tensor;
    LocalTensor<T> yTensor;
    LocalTensor<T> y2Tensor;
    LocalTensor<T> p1Tensor;
    LocalTensor<T> p2Tensor;
    LocalTensor<T> aTensor;
    LocalTensor<T> a2Tensor;
    LocalTensor<T> bTensor;
    LocalTensor<T> b2Tensor;
    LocalTensor<float> cTensor;
    LocalTensor<float> c2Tensor;

    DEvent<PIPE_MTE2, PIPE_MTE1> l1ready{EVENT_ID4, EVENT_ID5};
    DEvent<PIPE_MTE1, PIPE_MTE2> l1empty{EVENT_ID4, EVENT_ID5};
    DEvent<PIPE_MTE1, PIPE_M> l0ready{EVENT_ID4, EVENT_ID5};
    DEvent<PIPE_M, PIPE_MTE1> l0empty{EVENT_ID4, EVENT_ID5};
    DEvent<PIPE_FIX, PIPE_MTE1> x2ready{EVENT_ID4, EVENT_ID5};
    DEvent<PIPE_FIX, PIPE_M> outempty{EVENT_ID4, EVENT_ID5};

    int64_t invalidK = 0; // 从invalidK开始，需要区分尾块/非尾块搬入x
};
} // namespace FlatQuantNS

#endif // FLAT_QUANT_CUBE_H
