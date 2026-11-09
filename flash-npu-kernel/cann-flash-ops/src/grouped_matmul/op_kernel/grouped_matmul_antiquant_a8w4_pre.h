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
 * \file grouped_matmul_antiquant_a8w4_pre.h
 * \brief
 */

#ifndef ASCENDC_GROUPED_MATMUL_ANTIQUANT_A8W4_PRE_H
#define ASCENDC_GROUPED_MATMUL_ANTIQUANT_A8W4_PRE_H

#include "kernel_operator.h"
#ifdef GMM_ANTI_QUANT_A8W4_MSD
namespace GROUPED_MATMUL{
using namespace AscendC;

#if ORIG_DTYPE_Y == DT_FLOAT16
    using DTYPE_SCALE_OUT  = float;
#else
    using DTYPE_SCALE_OUT = bfloat16_t;
#endif

template <CubeFormat wFormat>
class GMMA8W4FakeQuantPreProcess {
public:
    __aicore__ inline GMMA8W4FakeQuantPreProcess(){};
    __aicore__ inline void Init(GM_ADDR weight, GM_ADDR y, GM_ADDR groupList, GM_ADDR workspace, const GMMBaseParams& tilingData, TPipe *pipe);
    __aicore__ inline void Process();
private:
    __aicore__ inline void ScaleProcess();

    GlobalTensor<int4b_t> weightGm;
    GlobalTensor<int8_t> yGm;
    GlobalTensor<DTYPE_SCALE_OUT> scaleOutGm;
    GlobalTensor<int64_t> scaleGm;

    uint32_t totalGroup{0};
    uint32_t blockDim;
    uint32_t startNum;
    uint32_t groupNum;
    bool isPerchannel = false;

    const int TWO = 2;
    const int EIGHT = 8;
    const int DATA_BLOCK_SIZE_32 = 32;
    const uint32_t WITH_OFFSET = 0;
    const GMMBaseParams *tiling;
};

template <CubeFormat wFormat>
__aicore__ inline void GMMA8W4FakeQuantPreProcess<wFormat>::Init(GM_ADDR weight, GM_ADDR y, GM_ADDR scale, GM_ADDR workspace, const GMMBaseParams& tilingData, TPipe *pipe){
    this->tiling = &tilingData;
    weightGm.SetGlobalBuffer(GetTensorAddr<int4b_t>(0, weight));
    yGm.SetGlobalBuffer((__gm__ int8_t *)workspace);
    if (tiling->quantGroupNum == 1) {   //per channel
        this->isPerchannel = true;
        scaleGm.SetGlobalBuffer(GetTensorAddr<int64_t>(0, scale));
        scaleOutGm.SetGlobalBuffer((__gm__ DTYPE_SCALE_OUT *)((__gm__ int8_t *)workspace + tiling->groupNum * tiling->n * tiling->k));
    }
}
template <CubeFormat wFormat>
__aicore__ inline void GMMA8W4FakeQuantPreProcess<wFormat>::Process()
{
    const size_t baseSize = 24 * 2 * 1024;
    const size_t HALF_UB = 96 * 1024;
    const size_t BOTTOM_LOOP = 72 * 1024;
    const size_t BOTTOM_LOOP_NZ = 48 * 1024;
    AscendC::LocalTensor<int4b_t> ALocalI4(AscendC::TPosition::VECIN, BOTTOM_LOOP, baseSize);
    AscendC::LocalTensor<half> ALocalF16(AscendC::TPosition::VECIN, 0, baseSize);
    AscendC::LocalTensor<int8_t> ALocalI8(AscendC::TPosition::VECOUT, 0, baseSize);
    AscendC::LocalTensor<int8_t> ALocalI8NZ(AscendC::TPosition::VECOUT, BOTTOM_LOOP_NZ, baseSize);
#if ASCENDC_CPU_DEBUG

    AscendC::LocalTensor<int4b_t> BLocalI4(AscendC::TPosition::VECIN, 0, baseSize);
    AscendC::LocalTensor<half> BLocalF16(AscendC::TPosition::VECIN, 0, baseSize);
    AscendC::LocalTensor<int8_t> BLocalI8(AscendC::TPosition::VECOUT, 0, baseSize);
    AscendC::LocalTensor<int8_t> BLocalI8NZ(AscendC::TPosition::VECOUT, 0, baseSize);
#else
    AscendC::LocalTensor<int4b_t> BLocalI4(AscendC::TPosition::VECIN, HALF_UB + BOTTOM_LOOP, baseSize);
    AscendC::LocalTensor<half> BLocalF16(AscendC::TPosition::VECIN, HALF_UB, baseSize);
    AscendC::LocalTensor<int8_t> BLocalI8(AscendC::TPosition::VECOUT, HALF_UB, baseSize);
    AscendC::LocalTensor<int8_t> BLocalI8NZ(AscendC::TPosition::VECOUT, HALF_UB + BOTTOM_LOOP_NZ, baseSize);
#endif
    constexpr uint32_t K_PER_LOOP = 24 * 1024 * 2 / 64;     //one buffer: 24k int4

    startNum = GetBlockIdx();
    totalGroup = tiling->n / 64;    //限制为64对齐
    uint64_t lastGroupRest = 0;
    if (totalGroup * 64 != tiling->n) {
        lastGroupRest = tiling->n - totalGroup * 64;
        totalGroup++;
    }
    blockDim = GetBlockNum() * GetTaskRation();

    for(uint32_t eStart = 0; eStart < tiling->groupNum; eStart++) {
        const size_t eStartAddrGm = eStart * tiling->n * tiling->k;
        for(uint32_t nStart = startNum; nStart < totalGroup; nStart += blockDim) {
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);

            bool isLastNotAlignN = false;
            size_t nBlockSize = 64;
            if (unlikely(nStart == totalGroup - 1 && lastGroupRest > 0)) {        //nd：n非64对齐
                isLastNotAlignN = true;
                nBlockSize = lastGroupRest;
            }
            for(uint32_t kStart = 0; kStart < tiling->k; kStart += K_PER_LOOP * 2) {  //2: double buffer
                const size_t kLenThisTimeTotal = (kStart + K_PER_LOOP * 2 > tiling->k) ? (tiling->k - kStart) : K_PER_LOOP * 2;
                const uint16_t kLen = kLenThisTimeTotal / 2;        //2: per buffer
                const size_t handleBytePerBuffer = kLen * nBlockSize;
                const size_t startAddrGm = tiling->k * 64 * nStart + kStart * nBlockSize;
#if ASCENDC_CPU_DEBUG
#else
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
#endif
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
                DataCopy(ALocalI4, weightGm[eStartAddrGm + startAddrGm], handleBytePerBuffer);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);

                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
                Cast(ALocalF16, ALocalI4, AscendC::RoundMode::CAST_NONE, handleBytePerBuffer);
                PipeBarrier<PIPE_V>();
                Cast(ALocalI8, ALocalF16, AscendC::RoundMode::CAST_NONE, handleBytePerBuffer);

                if constexpr (wFormat == CubeFormat::NZ) {
                    //to NZ 1
                    PipeBarrier<PIPE_V>();
                    DataCopy(ALocalI8NZ, ALocalI8, {kLen, 1, 1, 0});
                    PipeBarrier<PIPE_V>();
                    //to NZ 2
                    DataCopy(ALocalI8NZ[kLen * 32], ALocalI8[32], {kLen, 1, 1, 0});
                    PipeBarrier<PIPE_V>();
                }
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
                if constexpr (wFormat == CubeFormat::ND) {
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
                    DataCopy(yGm[eStartAddrGm + startAddrGm], ALocalI8, handleBytePerBuffer);
                    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
                }
#if ASCENDC_CPU_DEBUG
#else
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
#endif
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
                DataCopy(BLocalI4, weightGm[eStartAddrGm + startAddrGm + handleBytePerBuffer], handleBytePerBuffer);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);

                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
                Cast(BLocalF16, BLocalI4, AscendC::RoundMode::CAST_NONE, handleBytePerBuffer);
                PipeBarrier<PIPE_V>();
                Cast(BLocalI8, BLocalF16, AscendC::RoundMode::CAST_NONE, handleBytePerBuffer);

                if constexpr (wFormat == CubeFormat::NZ) {
                    //to NZ 1
                    PipeBarrier<PIPE_V>();
                    DataCopy(BLocalI8NZ, BLocalI8, {kLen, 1, 1, 0});
                    PipeBarrier<PIPE_V>();
                    //to NZ 2
                    DataCopy(BLocalI8NZ[kLen * 32], BLocalI8[32], {kLen, 1, 1, 0});
                    PipeBarrier<PIPE_V>();
                }
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID1);

                if constexpr (wFormat == CubeFormat::NZ) {            //nz -> nz
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
                    DataCopy(yGm[eStartAddrGm + tiling->k * 64 * nStart + kStart * 32], ALocalI8NZ, kLen * 32);
                    DataCopy(yGm[eStartAddrGm + tiling->k * 64 * nStart + tiling->k * 32 + kStart * 32], ALocalI8NZ[kLen * 32], kLen * 32);
                    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID1);
                    DataCopy(yGm[eStartAddrGm + tiling->k * 64 * nStart + kLen * 32 + kStart * 32], BLocalI8NZ, kLen * 32);
                    DataCopy(yGm[eStartAddrGm + tiling->k * 64 * nStart + kLen * 32 + tiling->k * 32 + kStart * 32], BLocalI8NZ[kLen * 32], kLen * 32);
                    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
                } else {
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID1);
                    DataCopy(yGm[eStartAddrGm + startAddrGm + handleBytePerBuffer], BLocalI8, handleBytePerBuffer);
                    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
                }
            }
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
        }
    }
    if(isPerchannel) {
        ScaleProcess();
    }
}
template <CubeFormat wFormat>
__aicore__ inline void GMMA8W4FakeQuantPreProcess<wFormat>::ScaleProcess() {
    const size_t SCALE_SIZE = 192 * 1024 / sizeof(int64_t); //u64
#if ASCENDC_CPU_DEBUG
    AscendC::LocalTensor<int64_t> scaleU64(AscendC::TPosition::VECIN, 0, SCALE_SIZE - 32);
    AscendC::LocalTensor<float> scaleF32(AscendC::TPosition::VECIN, 0, SCALE_SIZE - 32);
    AscendC::LocalTensor<bfloat16_t> scaleBF16(AscendC::TPosition::VECIN, 0, SCALE_SIZE - 32);
#else
    AscendC::LocalTensor<int64_t> scaleU64(AscendC::TPosition::VECIN, 0, SCALE_SIZE);
    AscendC::LocalTensor<float> scaleF32(AscendC::TPosition::VECIN, 0, SCALE_SIZE);
    AscendC::LocalTensor<bfloat16_t> scaleBF16(AscendC::TPosition::VECIN, 0, SCALE_SIZE);
#endif
    const uint64_t scale_n = tiling->n * 2;
    const uint64_t each_core_u64 = (tiling->groupNum * tiling->n / blockDim) / 16 * 16;    // align to 8
    const uint64_t this_core_u64 = startNum == blockDim - 1 ? tiling->groupNum * tiling->n - each_core_u64 * (blockDim - 1) : each_core_u64;
    const uint64_t start_element_u64 = each_core_u64 * startNum;
    int loop_size_u64 = 0;
    PipeBarrier<PIPE_ALL>();
    for (int start_loc = 0; start_loc < this_core_u64; start_loc += SCALE_SIZE) {
        loop_size_u64 = SCALE_SIZE;
        if(start_loc + SCALE_SIZE > this_core_u64) loop_size_u64 = this_core_u64 - start_loc;
        int loop_size_f32 = loop_size_u64 * 2;
        DataCopy(scaleU64, scaleGm[start_element_u64 + start_loc], loop_size_u64);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        const size_t PR_SIZE = 64 * 255; //maximum element number in one instruction
        int pr_last = 0;
        for(int pr_start = 0; pr_start < loop_size_f32 ; pr_start += PR_SIZE) {
            size_t pr_this_time = PR_SIZE;
            if (pr_start + PR_SIZE > loop_size_f32) {
                pr_this_time = loop_size_f32 - pr_start;
                pr_last = pr_this_time - pr_this_time / 64 * 64;
            }
            int repeatCast = pr_this_time / 64;
            AscendC::PairReduceSum<float>(scaleF32[pr_start / 2], scaleF32[pr_start], repeatCast, 64, 1, 1, 8);
            PipeBarrier<PIPE_V>();
        }
        if (pr_last > 0) {
            AscendC::PairReduceSum<float>(scaleF32[(loop_size_f32 - pr_last)/ 2], scaleF32[loop_size_f32 - pr_last], 1, pr_last, 1, 1, 8);
        }
        if constexpr (sizeof(DTYPE_SCALE_OUT) == 2) { // to bf16
            PipeBarrier<PIPE_V>();
            Cast(scaleBF16, scaleF32, AscendC::RoundMode::CAST_FLOOR, loop_size_u64);
        }
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        #if ORIG_DTYPE_Y == DT_FLOAT16
            DataCopy(scaleOutGm[start_element_u64+start_loc],scaleF32 , loop_size_u64);
        #else
            DataCopy(scaleOutGm[start_element_u64+start_loc], scaleBF16, (loop_size_u64 + 16 - 1) / 16 * 16);
        #endif
        PipeBarrier<PIPE_ALL>();
    }
}

} // namespace
#endif
#endif