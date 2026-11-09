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
 * \file grouped_matmul_weight_quant_a16w4_msd_basic_block.h
 * \brief
 */
#ifndef GROUPED_MATMUL_WEIGHT_QUANT_A16W4_MSD_BASIC_BLOCK_H
#define GROUPED_MATMUL_WEIGHT_QUANT_A16W4_MSD_BASIC_BLOCK_H

#include "grouped_matmul_weight_quant_a16w4_msd_basic_block_config.h"
#include "grouped_matmul_weight_quant_a16w4_msd_cube_service.h"
#include "grouped_matmul_weight_quant_a16w4_msd_vector_service.h"
#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#include "lib/matmul_intf.h"
#include "tool.h"

namespace GROUPED_MATMUL::A16W4Msd {
#define GMM_WQ_A16W4_MSD_BASIC_BLOCK_TEMPLATE_PARAM template <typename xType, typename wType, typename biasType>

#define GMM_WQ_A16W4_MSD_BASIC_BLOCK_CLASS WeightQuantA16W4MsdBasicBlock<xType, wType, biasType>

GMM_WQ_A16W4_MSD_BASIC_BLOCK_TEMPLATE_PARAM
class WeightQuantA16W4MsdBasicBlock {
public:
    using mm1OutputType = float;
    __aicore__ inline WeightQuantA16W4MsdBasicBlock(){};
    __aicore__ inline void InitPreProcess(TPipe *tPipe);
    __aicore__ inline void PreProcess(uint64_t curMSize, uint64_t mPreOffset, uint64_t mIdx,
                                      const A16W4MsdConstParam &constParams, const GlobalTensor<xType> &xGlobal,
                                      const GlobalTensor<float> &aMaxWorkspaceGm,
                                      const GlobalTensor<int8_t> &aUnfoldGlobal);
    __aicore__ inline void InitMsd(__gm__ int8_t *weightS8WsAddr, __gm__ int8_t *aUnfoldS8WsAddr,
                                   __gm__ int8_t *cF32WsAddr, uint64_t antiQuantGroupSize, TPipe *tPipe);
    __aicore__ inline void UpdateGlobalAddr(__gm__ int8_t *aUnfoldS8WsAddr, __gm__ wType *weight,
                                            __gm__ xType *antiquantScale, __gm__ biasType *bias, __gm__ xType *y,
                                            __gm__ float *aMaxWs, const bool hasBias);
    __aicore__ inline void ComputeBasicBlock(const A16W4MsdConstParam &constParams,
                                             const A16W4MsdBasicBlockOffsetParam &curOffsetParam,
                                             const A16W4MsdBasicBlockOffsetParam &lastOffsetParam);
    __aicore__ inline void EndMsd(const A16W4MsdConstParam &constParams,
                                  const A16W4MsdBasicBlockOffsetParam &lastOffsetParam);

protected:
    GMMA16W4MsdVecService<xType, wType, biasType> vecCompute_;
    GMMA16W4MsdCubeService<int8_t, mm1OutputType> cubeCompute_;

    uint64_t cvLoopIdx_ = 0;
};

GMM_WQ_A16W4_MSD_BASIC_BLOCK_TEMPLATE_PARAM
__aicore__ inline void GMM_WQ_A16W4_MSD_BASIC_BLOCK_CLASS::InitPreProcess(TPipe *tPipe)
{
    if ASCEND_IS_AIV {
        vecCompute_.InitPreProcess(tPipe);
    }
}

GMM_WQ_A16W4_MSD_BASIC_BLOCK_TEMPLATE_PARAM
__aicore__ inline void GMM_WQ_A16W4_MSD_BASIC_BLOCK_CLASS::PreProcess(uint64_t curMSize, uint64_t mPreOffset,
                                                                      uint64_t mIdx,
                                                                      const A16W4MsdConstParam &constParams,
                                                                      const GlobalTensor<xType> &xGlobal,
                                                                      const GlobalTensor<float> &aMaxWorkspaceGm,
                                                                      const GlobalTensor<int8_t> &aUnfoldGlobal)
{
    if ASCEND_IS_AIV {
        vecCompute_.PreProcess(curMSize, mPreOffset, mIdx, constParams, xGlobal, aMaxWorkspaceGm, aUnfoldGlobal);
    }
}

GMM_WQ_A16W4_MSD_BASIC_BLOCK_TEMPLATE_PARAM
__aicore__ inline void GMM_WQ_A16W4_MSD_BASIC_BLOCK_CLASS::InitMsd(__gm__ int8_t *weightS8WsAddr,
                                                                   __gm__ int8_t *aUnfoldS8WsAddr,
                                                                   __gm__ int8_t *cF32WsAddr,
                                                                   uint64_t antiQuantGroupSize, TPipe *tPipe)
{
    if ASCEND_IS_AIC {
        cubeCompute_.InitMsd(weightS8WsAddr, aUnfoldS8WsAddr, (__gm__ mm1OutputType *)cF32WsAddr, tPipe);
    } else {
        vecCompute_.InitMsd(weightS8WsAddr, (__gm__ mm1OutputType *)cF32WsAddr, tPipe);
    }
    cvLoopIdx_ = 0;
}

GMM_WQ_A16W4_MSD_BASIC_BLOCK_TEMPLATE_PARAM
__aicore__ inline void GMM_WQ_A16W4_MSD_BASIC_BLOCK_CLASS::UpdateGlobalAddr(__gm__ int8_t *aUnfoldS8WsAddr,
                                                                            __gm__ wType *weight,
                                                                            __gm__ xType *antiquantScale,
                                                                            __gm__ biasType *bias, __gm__ xType *y,
                                                                            __gm__ float *aMaxWs, const bool hasBias)
{
    if ASCEND_IS_AIC {
        cubeCompute_.UpdateGlobalAddr(aUnfoldS8WsAddr);
    } else {
        vecCompute_.UpdateGlobalAddr(aMaxWs, antiquantScale, weight, y);
    }
}

GMM_WQ_A16W4_MSD_BASIC_BLOCK_TEMPLATE_PARAM
__aicore__ inline void GMM_WQ_A16W4_MSD_BASIC_BLOCK_CLASS::ComputeBasicBlock(
    const A16W4MsdConstParam &constParams, const A16W4MsdBasicBlockOffsetParam &curOffsetParam,
    const A16W4MsdBasicBlockOffsetParam &lastOffsetParam)
{
    if ASCEND_IS_AIV {
        vecCompute_.CastW4ToW8(cvLoopIdx_, constParams, curOffsetParam);
        CrossCoreSetFlag<SYNC_MODE2, PIPE_MTE3>(SYNC_AIV_AIC_FLAG);
        if (cvLoopIdx_ > 0) {
            CrossCoreWaitFlag(SYNC_AIC_AIV_FLAG);
            vecCompute_.PostProcess(cvLoopIdx_ - 1, constParams, lastOffsetParam);
        }
    } else {
        CrossCoreWaitFlag(SYNC_AIV_AIC_FLAG);
        cubeCompute_.ComputeA8W8MatDuce(cvLoopIdx_, constParams, curOffsetParam);
        CrossCoreSetFlag<SYNC_MODE2, PIPE_FIX>(SYNC_AIC_AIV_FLAG);
    }
    cvLoopIdx_++;
}

GMM_WQ_A16W4_MSD_BASIC_BLOCK_TEMPLATE_PARAM
__aicore__ inline void GMM_WQ_A16W4_MSD_BASIC_BLOCK_CLASS::EndMsd(const A16W4MsdConstParam &constParams,
                                                                  const A16W4MsdBasicBlockOffsetParam &lastOffsetParam)
{
    if ASCEND_IS_AIC {
        cubeCompute_.EndMsd();
    } else {
        if (cvLoopIdx_ > 0) {
            CrossCoreWaitFlag(SYNC_AIC_AIV_FLAG);
            vecCompute_.PostProcess(cvLoopIdx_ - 1, constParams, lastOffsetParam);
        }
        vecCompute_.EndMsd();
    }
}

}  // namespace GROUPED_MATMUL::A16W4Msd

#endif  // GROUPED_MATMUL_WEIGHT_QUANT_A16W4_MSD_BASIC_BLOCK_H
