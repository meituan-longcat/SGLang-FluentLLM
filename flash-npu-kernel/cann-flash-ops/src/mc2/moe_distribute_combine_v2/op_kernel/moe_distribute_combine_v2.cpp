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
 * \file moe_distribute_combine_v2.cpp
 * \brief
 */
#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "lib/matmul_intf.h"
#include "moe_distribute_combine_v2_tiling_key.h"

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
#include "arch35/moe_distribute_combine_arch35.h"
#else
#include "moe_distribute_combine_a2.h"
#include "moe_distribute_combine_a2_layered.h"
#include "moe_distribute_combine_a2_layered_aicpu.h"
#endif // __NPU_ARCH__ == 3510
#include "moe_distribute_combine_v2_tiling.h"
#include "moe_distribute_combine_v2.h"
#include "moe_distribute_combine_v2_layered.h"

#if !(defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510))
using namespace MoeDistributeCombineA2Impl;
#endif // __NPU_ARCH__ == 3510

using namespace MoeDistributeCombineV2Impl;
using namespace Mc2Kernel;
using namespace Mc2Tiling;
using namespace AscendC;

namespace {
template <CombineMC2TypeClass>
__aicore__ inline void ExecMoeDistributeCombineV2(GM_ADDR expandX, GM_ADDR expertIds,
                                                GM_ADDR assistInfoForCombine, GM_ADDR epSendCount, 
                                                GM_ADDR tpSendCount, GM_ADDR scales, GM_ADDR xActiveMask, 
                                                GM_ADDR sharedExpertX, GM_ADDR elasticInfo, GM_ADDR oriX, 
                                                GM_ADDR constExpertAlpha1, GM_ADDR constExpertAlpha2,
                                                GM_ADDR constExpertV, GM_ADDR XOut,
                                                GM_ADDR workspaceGM, GM_ADDR tilingGM, TPipe *pipePtr)
{
    GET_TILING_DATA_WITH_STRUCT(MoeDistributeCombineV2TilingData, tilingData, tilingGM);
    MoeDistributeCombineV2<CombineMC2TypeFunc> op;
    op.Init(nullptr, expandX, expertIds, assistInfoForCombine, epSendCount, tpSendCount, nullptr, nullptr,
            scales, xActiveMask, sharedExpertX, elasticInfo, oriX, constExpertAlpha1, 
            constExpertAlpha2, constExpertV, nullptr, nullptr, XOut, workspaceGM, pipePtr, &tilingData);
    op.Process();
}
}
template<bool HasTp, uint8_t QuantMode, uint8_t LayeredMode, uint8_t ArchTag>
__global__ __aicore__ void moe_distribute_combine_v2(GM_ADDR expandX, GM_ADDR expertIds, GM_ADDR assistInfoForCombine,
                                                    GM_ADDR epSendCount, GM_ADDR scales, GM_ADDR tpSendCount,
                                                    GM_ADDR xActiveMask, GM_ADDR activationScale, GM_ADDR weightScale,
                                                    GM_ADDR groupList, GM_ADDR expandScales, GM_ADDR sharedExpertX, 
                                                    GM_ADDR elasticInfo, GM_ADDR oriX, GM_ADDR constExpertAlpha1,
                                                    GM_ADDR constExpertAlpha2, GM_ADDR constExpertV,
                                                    GM_ADDR XOut, GM_ADDR workspaceGM,
                                                    GM_ADDR tilingGM)

{
    REGISTER_TILING_DEFAULT(MoeDistributeCombineV2TilingData);
#if !(defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510))
    REGISTER_TILING_FOR_TILINGKEY("ArchTag == TILINGKEY_TPL_A2", MoeDistributeCombineA2TilingData);
#endif
    TPipe pipe;

#if ((ORIG_DTYPE_EXPAND_X == DT_BF16) || (ORIG_DTYPE_EXPAND_X == DT_FLOAT16))
    if constexpr ((ArchTag == TILINGKEY_TPL_A2) && 
                (LayeredMode == TILINGKEY_TPL_MTE) && (QuantMode == TILINGKEY_NO_QUANT)) {
        GET_TILING_DATA_WITH_STRUCT(MoeDistributeCombineA2TilingData, tilingData, tilingGM);
        MoeDistributeCombineA2<DTYPE_EXPAND_X, int32_t> op;
        op.Init(expandX, expertIds, assistInfoForCombine, epSendCount, scales, xActiveMask,
            oriX, constExpertAlpha1, constExpertAlpha2, constExpertV, XOut, workspaceGM, &pipe, &tilingData);
        op.Process();
    } else if constexpr ((ArchTag == TILINGKEY_TPL_A2) && 
                        (QuantMode == TILINGKEY_NO_QUANT) && (LayeredMode == TILINGKEY_TPL_AICPU)) {
        GET_TILING_DATA_WITH_STRUCT(MoeDistributeCombineA2TilingData, tilingData, tilingGM);
        auto contextGM0 = AscendC::GetHcclContext<HCCL_GROUP_ID_0>();
        DataplaneMode dataplaneMode = GetDataplaneMode(contextGM0);
        if (dataplaneMode == DataplaneMode::AICPU) {
            MoeDistributeCombineA2<DTYPE_EXPAND_X, int32_t> op;
            op.Init(expandX, expertIds, assistInfoForCombine, epSendCount, scales, xActiveMask,
                oriX, constExpertAlpha1, constExpertAlpha2, constExpertV, XOut, workspaceGM, &pipe, &tilingData);
            op.Process();
        } else if (dataplaneMode == DataplaneMode::AIV) {
            MoeDistributeCombineA2<DTYPE_EXPAND_X, int32_t> op;
            op.Init(expandX, expertIds, assistInfoForCombine, epSendCount, scales, xActiveMask,
                oriX, constExpertAlpha1, constExpertAlpha2, constExpertV, XOut, workspaceGM, &pipe, &tilingData);
            op.Process();
        }
    } else if constexpr ((ArchTag == TILINGKEY_TPL_A2) && 
                        (QuantMode == TILINGKEY_INT8_QUANT) && (LayeredMode == TILINGKEY_TPL_AICPU)) {
        GET_TILING_DATA_WITH_STRUCT(MoeDistributeCombineA2TilingData, tilingData, tilingGM);

        auto contextGM0 = AscendC::GetHcclContext<HCCL_GROUP_ID_0>();
        DataplaneMode dataplaneMode = GetDataplaneMode(contextGM0);
        if (dataplaneMode == DataplaneMode::AIV) {
            MoeDistributeCombineA2<DTYPE_EXPAND_X, int32_t> op;
            op.Init(expandX, expertIds, assistInfoForCombine, epSendCount, scales, xActiveMask,
                oriX, constExpertAlpha1, constExpertAlpha2, constExpertV, XOut, workspaceGM, &pipe, &tilingData);
            op.Process();
        }
    }
    if constexpr ((ArchTag == TILINGKEY_TPL_A3) && (LayeredMode == TILINGKEY_TPL_MTE)) {
        ExecMoeDistributeCombineV2<DTYPE_EXPAND_X, DTYPE_X, int32_t, HasTp, QuantMode == TILINGKEY_INT8_QUANT, false>(
        expandX, expertIds, assistInfoForCombine, epSendCount, tpSendCount, scales, xActiveMask, sharedExpertX, 
        elasticInfo, oriX, constExpertAlpha1, constExpertAlpha2, constExpertV, XOut, workspaceGM, tilingGM, &pipe);
    } else if constexpr ((ArchTag == TILINGKEY_TPL_A3) && (LayeredMode == TILINGKEY_TPL_HIERARCHY)) {
        if constexpr (QuantMode == TILINGKEY_NO_QUANT) {
            GET_TILING_DATA_WITH_STRUCT(MoeDistributeCombineV2TilingData, tilingData, tilingGM);
            auto contextGM0 = AscendC::GetHcclContext<HCCL_GROUP_ID_0>();
            MoeDistributeCombineV2Layered<DTYPE_EXPAND_X, int32_t, DTYPE_EXPAND_X> op;
            op.Init(expandX, expertIds, assistInfoForCombine, epSendCount, expandScales, xActiveMask, XOut, workspaceGM, &pipe, &tilingData,
                    contextGM0);
            op.Process();
        } else if constexpr (QuantMode == TILINGKEY_INT8_QUANT) {
            GET_TILING_DATA_WITH_STRUCT(MoeDistributeCombineV2TilingData, tilingData, tilingGM);
            auto contextGM0 = AscendC::GetHcclContext<HCCL_GROUP_ID_0>();
            MoeDistributeCombineV2Layered<DTYPE_EXPAND_X, int32_t, int8_t> op;
            op.Init(expandX, expertIds, assistInfoForCombine, epSendCount, expandScales, xActiveMask, XOut, workspaceGM, &pipe, &tilingData,
                    contextGM0);
            op.Process();
        }
    }
#endif
}
