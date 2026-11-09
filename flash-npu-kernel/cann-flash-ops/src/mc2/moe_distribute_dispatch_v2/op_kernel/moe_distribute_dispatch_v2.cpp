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
 * \file moe_distribute_dispatch_v2.cpp
 * \brief
 */
#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "moe_distribute_dispatch_v2.h"
#include "moe_distribute_dispatch_v2_tiling.h"
#include "moe_distribute_dispatch_v2_full_mesh.h"
#include "moe_distribute_dispatch_v2_layered.h"
#include "moe_distribute_dispatch_v2_tiling_key.h"

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
#include "arch35/moe_distribute_dispatch_arch35.h"
#else
#include "moe_distribute_dispatch_a2.h"
#include "moe_distribute_dispatch_a2_layered.h"
#include "moe_distribute_dispatch_a2_layered_aicpu.h"
#endif // __NPU_ARCH__ == 3510

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
using namespace MoeDistributeDispatchA5Impl;
#else
using namespace MoeDistributeDispatchA2Impl;
#endif

using namespace MoeDistributeDispatchV2Impl;
using namespace Mc2Kernel;
using namespace MoeDistributeDispatchV2FullMeshImpl;
using namespace Mc2Tiling;
using namespace AscendC;

template<bool HasTp, uint8_t QuantMode, bool ScaleMode, uint8_t FullMesh, uint8_t CommMode, uint8_t ArchTag>
__global__ __aicore__ void moe_distribute_dispatch_v2(
    GM_ADDR x, GM_ADDR expertIds, GM_ADDR scales, GM_ADDR xActiveMask, GM_ADDR expertScales, 
    GM_ADDR elasticInfo, GM_ADDR expandXOut, GM_ADDR dynamicScalesOut,
    GM_ADDR assistInfoOut, GM_ADDR expertTokenNumsOut, GM_ADDR epSendCountsOut, GM_ADDR tpSendCountsOut, 
    GM_ADDR expandScalesOut, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
REGISTER_TILING_DEFAULT(MoeDistributeDispatchV2TilingData);
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
    GET_TILING_DATA_WITH_STRUCT(MoeDistributeDispatchV2TilingData, tilingData, tilingGM);
#else
    REGISTER_TILING_FOR_TILINGKEY("ArchTag == TILINGKEY_TPL_A2", MoeDistributeDispatchA2TilingData);
#endif
    TPipe pipe;
    
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
    int64_t oriOverflowMode = AscendC::GetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>();
#if ((ORIG_DTYPE_EXPAND_X == DT_BF16) || (ORIG_DTYPE_EXPAND_X == DT_FLOAT16))
    if constexpr (ArchTag == TILINGKEY_TPL_A5) {
        if constexpr (CommMode == TILINGKEY_TPL_CCU) {
            MoeDistributeDispatchA5<DTYPE_X, DTYPE_EXPAND_X, MoeDistributeDispatchV2Impl::UNQUANT, false, false> op;
            op.Init(x, expertIds, scales, xActiveMask, expandXOut, dynamicScalesOut, assistInfoOut, 
                    expertTokenNumsOut, epSendCountsOut, tpSendCountsOut, workspaceGM, &pipe, &tilingData);
            op.Process();
        } else if constexpr (CommMode == TILINGKEY_TPL_MTE) {
            if constexpr (FullMesh == TILINGKEY_ENABLE_FULLMESH) {
                MoeDistributeDispatchV2FullMesh<DTYPE_X, DTYPE_EXPAND_X, MoeDistributeDispatchV2Impl::UNQUANT, false, false> op;
                op.Init(nullptr, x, expertIds, scales, xActiveMask, elasticInfo, expandXOut, dynamicScalesOut, assistInfoOut,
                        expertTokenNumsOut, epSendCountsOut, tpSendCountsOut, workspaceGM, &pipe, &tilingData);
                op.Process();
            } else {
                MoeDistributeDispatchV2<DTYPE_X, DTYPE_EXPAND_X, MoeDistributeDispatchV2Impl::UNQUANT, false, false> op;
                op.Init(nullptr, x, expertIds, scales, xActiveMask, elasticInfo, expandXOut, dynamicScalesOut, assistInfoOut,
                        expertTokenNumsOut, epSendCountsOut, tpSendCountsOut, workspaceGM, &pipe, &tilingData);
                op.Process();
            }
        }  
    } 
#elif ((ORIG_DTYPE_X == DT_FLOAT8_E5M2) && (ORIG_DTYPE_EXPAND_X == DT_FLOAT8_E5M2)) ||   \
    ((ORIG_DTYPE_X == DT_FLOAT8_E4M3FN) && (ORIG_DTYPE_EXPAND_X == DT_FLOAT8_E4M3FN)) || \
    ((ORIG_DTYPE_X == DT_HIFLOAT8) && (ORIG_DTYPE_EXPAND_X == DT_HIFLOAT8))
    if constexpr (ArchTag == TILINGKEY_TPL_A5) {
        if constexpr (CommMode == TILINGKEY_TPL_CCU) {
            MoeDistributeDispatchA5<DTYPE_X, DTYPE_EXPAND_X, MoeDistributeDispatchV2Impl::UNQUANT, true, false> op;
            op.Init(x, expertIds, scales, xActiveMask, expandXOut, dynamicScalesOut, assistInfoOut, 
                    expertTokenNumsOut, epSendCountsOut, tpSendCountsOut, workspaceGM, &pipe, &tilingData);
            op.Process();
        } else if constexpr (CommMode == TILINGKEY_TPL_MTE) {
            MoeDistributeDispatchV2<DTYPE_X, DTYPE_EXPAND_X, MoeDistributeDispatchV2Impl::UNQUANT, true, false> op;
            op.Init(nullptr, x, expertIds, scales, xActiveMask, elasticInfo, expandXOut, dynamicScalesOut, assistInfoOut,
                    expertTokenNumsOut, epSendCountsOut, tpSendCountsOut, workspaceGM, &pipe, &tilingData);
            op.Process();
        }
    } 
#elif ((ORIG_DTYPE_EXPAND_X == DT_INT8) || (ORIG_DTYPE_EXPAND_X == DT_FLOAT8_E5M2) || \
       (ORIG_DTYPE_EXPAND_X == DT_FLOAT8_E4M3FN) || (ORIG_DTYPE_EXPAND_X == DT_HIFLOAT8))
    if constexpr (ArchTag == TILINGKEY_TPL_A5) {
        if constexpr (QuantMode != TILINGKEY_NO_QUANT) {
            if constexpr (CommMode == TILINGKEY_TPL_CCU) {
                MoeDistributeDispatchA5<DTYPE_X, DTYPE_EXPAND_X, QuantMode, ScaleMode, false> op;
                op.Init(x, expertIds, scales, xActiveMask, expandXOut, dynamicScalesOut, assistInfoOut, 
                        expertTokenNumsOut, epSendCountsOut, tpSendCountsOut, workspaceGM, &pipe, &tilingData);
                op.Process();
            } else if constexpr (CommMode == TILINGKEY_TPL_MTE) {
                if constexpr (FullMesh == TILINGKEY_ENABLE_FULLMESH) {
                    MoeDistributeDispatchV2FullMesh<DTYPE_X, DTYPE_EXPAND_X, QuantMode, ScaleMode, false> op;
                    op.Init(nullptr, x, expertIds, scales, xActiveMask, elasticInfo, expandXOut, dynamicScalesOut, assistInfoOut,
                            expertTokenNumsOut, epSendCountsOut, tpSendCountsOut, workspaceGM, &pipe, &tilingData);
                    op.Process();
                } else {
                    MoeDistributeDispatchV2<DTYPE_X, DTYPE_EXPAND_X, QuantMode, ScaleMode, false> op;
                    op.Init(nullptr, x, expertIds, scales, xActiveMask, elasticInfo, expandXOut, dynamicScalesOut, assistInfoOut,
                            expertTokenNumsOut, epSendCountsOut, tpSendCountsOut, workspaceGM, &pipe, &tilingData);
                    op.Process();
                }
            }
        } 
    }
    AscendC::SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(oriOverflowMode);
#endif
#else
#if ((ORIG_DTYPE_EXPAND_X == DT_BF16) || (ORIG_DTYPE_EXPAND_X == DT_FLOAT16))
    if constexpr ((ArchTag == TILINGKEY_TPL_A2) && (CommMode == TILINGKEY_TPL_MTE)) { // 不分层
        GET_TILING_DATA_WITH_STRUCT(MoeDistributeDispatchA2TilingData, tilingData, tilingGM);
        MoeDistributeDispatchA2<DTYPE_X, DTYPE_EXPAND_X, false, false, false, false> op;
        op.Init(x, expertIds, scales, expertScales, xActiveMask, expandXOut, dynamicScalesOut, assistInfoOut, expertTokenNumsOut, epSendCountsOut, expandScalesOut,
            workspaceGM, &pipe, tilingGM);
        op.Process();

    } else if constexpr ((ArchTag == TILINGKEY_TPL_A2) && (CommMode == TILINGKEY_TPL_AICPU)) { // 分层
        GET_TILING_DATA_WITH_STRUCT(MoeDistributeDispatchA2TilingData, tilingData, tilingGM);
        GM_ADDR contextGM0 = AscendC::GetHcclContext<HCCL_GROUP_ID_0>();
        DataplaneMode dataplaneMode = GetDataplaneMode(contextGM0);
        if (dataplaneMode == DataplaneMode::AICPU) {
            MoeDistributeDispatchA2<DTYPE_X, DTYPE_EXPAND_X, false, false, false, false> op;
            op.Init(x, expertIds, scales, expertScales, xActiveMask, expandXOut, dynamicScalesOut, assistInfoOut, expertTokenNumsOut, epSendCountsOut, expandScalesOut,
                workspaceGM, &pipe, tilingGM);
            op.Process();
        } else if (dataplaneMode == DataplaneMode::AIV) {
            MoeDistributeDispatchA2<DTYPE_X, DTYPE_EXPAND_X, false, false, false, false> op;
            op.Init(x, expertIds, scales, expertScales, xActiveMask, expandXOut, dynamicScalesOut, assistInfoOut, expertTokenNumsOut, epSendCountsOut, expandScalesOut,
                workspaceGM, &pipe, tilingGM);
            op.Process();
        }
    }
#elif (ORIG_DTYPE_EXPAND_X == DT_INT32)
    if constexpr ((ArchTag == TILINGKEY_TPL_A2) && (CommMode == TILINGKEY_TPL_AICPU) && (ScaleMode == true)) {
        GET_TILING_DATA_WITH_STRUCT(MoeDistributeDispatchA2TilingData, tilingData, tilingGM);
        GM_ADDR contextGM0 = AscendC::GetHcclContext<HCCL_GROUP_ID_0>();
        DataplaneMode dataplaneMode = GetDataplaneMode(contextGM0);
        if (dataplaneMode == DataplaneMode::AICPU || dataplaneMode == DataplaneMode::AIV) {
            MoeDistributeDispatchA2<DTYPE_X, DTYPE_EXPAND_X, false, false, false, true> op;
            op.Init(x, expertIds, scales, expertScales, xActiveMask, expandXOut, dynamicScalesOut, assistInfoOut,
                expertTokenNumsOut, epSendCountsOut, expandScalesOut, workspaceGM, &pipe, tilingGM);
            op.Process();
        }
    }
#elif (ORIG_DTYPE_EXPAND_X == DT_INT8)
    if constexpr ((ArchTag == TILINGKEY_TPL_A2) && (CommMode == TILINGKEY_TPL_MTE) && (ScaleMode == false)) {
        GET_TILING_DATA_WITH_STRUCT(MoeDistributeDispatchA2TilingData, tilingData, tilingGM);
        MoeDistributeDispatchA2<DTYPE_X, DTYPE_EXPAND_X, false, true, false, false> op;
        op.Init(x, expertIds, scales, expertScales, xActiveMask, expandXOut, dynamicScalesOut, assistInfoOut, expertTokenNumsOut, epSendCountsOut, expandScalesOut,
            workspaceGM, &pipe, tilingGM);
        op.Process();
    }
    if constexpr ((ArchTag == TILINGKEY_TPL_A2) && (CommMode == TILINGKEY_TPL_MTE) && (ScaleMode == true)) {
        GET_TILING_DATA_WITH_STRUCT(MoeDistributeDispatchA2TilingData, tilingData, tilingGM);
        MoeDistributeDispatchA2<DTYPE_X, DTYPE_EXPAND_X, false, true, true, false> op;
        op.Init(x, expertIds, scales, expertScales, xActiveMask, expandXOut, dynamicScalesOut, assistInfoOut, expertTokenNumsOut, epSendCountsOut, expandScalesOut,
            workspaceGM, &pipe, tilingGM);
        op.Process();
    }
    if constexpr ((ArchTag == TILINGKEY_TPL_A2) && (CommMode == TILINGKEY_TPL_AICPU) && (ScaleMode == false)) {
        GET_TILING_DATA_WITH_STRUCT(MoeDistributeDispatchA2TilingData, tilingData, tilingGM);
        GM_ADDR contextGM0 = AscendC::GetHcclContext<HCCL_GROUP_ID_0>();
        DataplaneMode dataplaneMode = GetDataplaneMode(contextGM0);
        if (dataplaneMode == DataplaneMode::AICPU) {
            MoeDistributeDispatchA2<DTYPE_X, DTYPE_EXPAND_X, false, true, false, false> op;
            op.Init(x, expertIds, scales, expertScales, xActiveMask, expandXOut, dynamicScalesOut, assistInfoOut, expertTokenNumsOut, epSendCountsOut, expandScalesOut,
                workspaceGM, &pipe, tilingGM);
            op.Process();
        } else if (dataplaneMode == DataplaneMode::AIV) {
            MoeDistributeDispatchA2<DTYPE_X, DTYPE_EXPAND_X, false, true, false, false> op;
            op.Init(x, expertIds, scales, expertScales, xActiveMask, expandXOut, dynamicScalesOut, assistInfoOut, expertTokenNumsOut, epSendCountsOut, expandScalesOut,
                workspaceGM, &pipe, tilingGM);
            op.Process();
        }
    }
    if constexpr ((ArchTag == TILINGKEY_TPL_A2) && (CommMode == TILINGKEY_TPL_AICPU) && (ScaleMode == true)) {
        GET_TILING_DATA_WITH_STRUCT(MoeDistributeDispatchA2TilingData, tilingData, tilingGM);
        GM_ADDR contextGM0 = AscendC::GetHcclContext<HCCL_GROUP_ID_0>();
        DataplaneMode dataplaneMode = GetDataplaneMode(contextGM0);
        if (dataplaneMode == DataplaneMode::AICPU) {
            MoeDistributeDispatchA2<DTYPE_X, DTYPE_EXPAND_X, false, true, true, false> op;
            op.Init(x, expertIds, scales, expertScales, xActiveMask, expandXOut, dynamicScalesOut, assistInfoOut, expertTokenNumsOut, epSendCountsOut, expandScalesOut,
                workspaceGM, &pipe, tilingGM);
            op.Process();
        } else if (dataplaneMode == DataplaneMode::AIV) {
            MoeDistributeDispatchA2<DTYPE_X, DTYPE_EXPAND_X, false, true, true, false> op;
            op.Init(x, expertIds, scales, expertScales, xActiveMask, expandXOut, dynamicScalesOut, assistInfoOut, expertTokenNumsOut, epSendCountsOut, expandScalesOut,
                workspaceGM, &pipe, tilingGM);
            op.Process();
        }
    }
#endif 
#endif 
#if ((ORIG_DTYPE_EXPAND_X == DT_BF16) || (ORIG_DTYPE_EXPAND_X == DT_FLOAT16))
    if constexpr (ArchTag == TILINGKEY_TPL_A3) {
        if constexpr (FullMesh == TILINGKEY_ENABLE_FULLMESH) {
            GET_TILING_DATA_WITH_STRUCT(MoeDistributeDispatchV2TilingData, tilingData, tilingGM);
            MoeDistributeDispatchV2FullMesh<DTYPE_X, DTYPE_EXPAND_X, MoeDistributeDispatchV2Impl::UNQUANT, false, false> op;
            op.Init(nullptr, x, expertIds, scales, xActiveMask, elasticInfo, expandXOut, dynamicScalesOut, assistInfoOut,
                    expertTokenNumsOut, epSendCountsOut, tpSendCountsOut, workspaceGM, &pipe, &tilingData);
            op.Process();
            return;
        } else if constexpr (FullMesh == TILINGKEY_NO_FULLMESH) {
            GET_TILING_DATA_WITH_STRUCT(MoeDistributeDispatchV2TilingData, tilingData, tilingGM);
            MoeDistributeDispatchV2<DTYPE_X, DTYPE_EXPAND_X, MoeDistributeDispatchV2Impl::UNQUANT, false, HasTp> op;
            op.Init(nullptr, x, expertIds, scales, xActiveMask, elasticInfo, expandXOut, dynamicScalesOut, assistInfoOut,
                    expertTokenNumsOut, epSendCountsOut, tpSendCountsOut, workspaceGM, &pipe, &tilingData);
            op.Process();
            return;
        } else if constexpr (FullMesh == TILINGKEY_ENABLE_HIERARCHY) {
            GET_TILING_DATA_WITH_STRUCT(MoeDistributeDispatchV2TilingData, tilingData, tilingGM);
            MoeDistributeDispatchV2Layered<DTYPE_X, DTYPE_EXPAND_X, false, false, false> op;
            op.Init(x, expertIds, scales, xActiveMask, expertScales, expandXOut, dynamicScalesOut, assistInfoOut, expertTokenNumsOut,
                epSendCountsOut, expandScalesOut, workspaceGM, &pipe, &tilingData);
            op.Process();
            return;
        }
    }
#elif (ORIG_DTYPE_EXPAND_X == DT_INT8)
    if constexpr (ArchTag == TILINGKEY_TPL_A3) {
        if constexpr (FullMesh == TILINGKEY_ENABLE_FULLMESH) {
            if constexpr (QuantMode == TILINGKEY_STATIC_QUANT) {
                GET_TILING_DATA_WITH_STRUCT(MoeDistributeDispatchV2TilingData, tilingData, tilingGM);
                MoeDistributeDispatchV2FullMesh<DTYPE_X, DTYPE_EXPAND_X, MoeDistributeDispatchV2Impl::STATIC_QUANT, false, false> op;
                op.Init(nullptr, x, expertIds, scales, xActiveMask, elasticInfo, expandXOut, dynamicScalesOut, assistInfoOut,
                        expertTokenNumsOut, epSendCountsOut, tpSendCountsOut, workspaceGM, &pipe, &tilingData);
                op.Process();
                return;
            } else if constexpr (QuantMode == TILINGKEY_PERTOKEN_QUANT) {
                GET_TILING_DATA_WITH_STRUCT(MoeDistributeDispatchV2TilingData, tilingData, tilingGM);
                MoeDistributeDispatchV2FullMesh<DTYPE_X, DTYPE_EXPAND_X, MoeDistributeDispatchV2Impl::PERTOKEN_DYNAMIC_QUANT, ScaleMode, false> op;
                op.Init(nullptr, x, expertIds, scales, xActiveMask, elasticInfo, expandXOut, dynamicScalesOut, assistInfoOut,
                        expertTokenNumsOut, epSendCountsOut, tpSendCountsOut, workspaceGM, &pipe, &tilingData);
                op.Process();
                return;
            }
        } else if constexpr (FullMesh == TILINGKEY_NO_FULLMESH) {
            if constexpr (QuantMode == TILINGKEY_STATIC_QUANT) {
                GET_TILING_DATA_WITH_STRUCT(MoeDistributeDispatchV2TilingData, tilingData, tilingGM);
                MoeDistributeDispatchV2<DTYPE_X, DTYPE_EXPAND_X, MoeDistributeDispatchV2Impl::STATIC_QUANT, false, HasTp> op;
                op.Init(nullptr, x, expertIds, scales, xActiveMask, elasticInfo, expandXOut, dynamicScalesOut, assistInfoOut,
                        expertTokenNumsOut, epSendCountsOut, tpSendCountsOut, workspaceGM, &pipe, &tilingData);
                op.Process();
                return;
            } else if constexpr (QuantMode == TILINGKEY_PERTOKEN_QUANT) {
                GET_TILING_DATA_WITH_STRUCT(MoeDistributeDispatchV2TilingData, tilingData, tilingGM);
                MoeDistributeDispatchV2<DTYPE_X, DTYPE_EXPAND_X, MoeDistributeDispatchV2Impl::PERTOKEN_DYNAMIC_QUANT, ScaleMode, HasTp> op;
                op.Init(nullptr, x, expertIds, scales, xActiveMask, elasticInfo, expandXOut, dynamicScalesOut, assistInfoOut,
                        expertTokenNumsOut, epSendCountsOut, tpSendCountsOut, workspaceGM, &pipe, &tilingData);
                op.Process();
                return;
            }
        } else if constexpr (FullMesh == TILINGKEY_ENABLE_HIERARCHY) {
            if constexpr (QuantMode == TILINGKEY_PERTOKEN_QUANT) {
                GET_TILING_DATA_WITH_STRUCT(MoeDistributeDispatchV2TilingData, tilingData, tilingGM);
                MoeDistributeDispatchV2Layered<DTYPE_X, DTYPE_EXPAND_X, false, true, false> op;
                op.Init(x, expertIds, scales, xActiveMask, expertScales, expandXOut, dynamicScalesOut, assistInfoOut, expertTokenNumsOut,
                    epSendCountsOut, expandScalesOut, workspaceGM, &pipe, &tilingData);
                op.Process();
                return;
            }
        }
    }
#endif
}
