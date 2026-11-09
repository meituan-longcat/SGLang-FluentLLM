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
 * \file flat_quant.cpp
 * \brief
 */

#include "flat_quant_vec.h"
#include "flat_quant_cube.h"
#include "flat_quant_vec_one.h"
#include "flat_quant_cube_one.h"
#include "flat_quant_high.h"

using namespace FlatQuantNS;

#define INVOKE_FLAT_QUANT_IMPL(Ttype)                                                    \
    do {                                                                                 \
        if ASCEND_IS_AIV {                                                               \
            FlatQuantVecOne<Ttype> op;                                                   \
            op.Init(kronecker_p1, group_list, out, quant_scale, workspace, &tilingData); \
            op.Process();                                                                \
        }                                                                                \
        if ASCEND_IS_AIC {                                                               \
            FlatQuantCubeOne<Ttype> op;                                                  \
            op.Init(x, kronecker_p2, group_list, workspace, &tilingData);                \
            op.Process();                                                                \
        }                                                                                \
    } while (0)

extern "C" __global__ __aicore__ void flat_quant(GM_ADDR x, GM_ADDR kronecker_p1, GM_ADDR kronecker_p2,
                                                 GM_ADDR group_list, GM_ADDR out, GM_ADDR quant_scale,
                                                 GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);

    // Physical K == 0 has empty outputs. This also protects graph-mode
    // launches that bypass the ACLNN empty-tensor fast path.
    if (tilingData.K == 0) {
        return;
    }

    // An all-padding group_list has no compute work.  Only materialize its
    // physical output storage when clear_padding was explicitly requested;
    // otherwise all cores can return without entering the sync protocol.
    if (tilingData.groupNum > 0) {
        GlobalTensor<int64_t> groupListGM;
        groupListGM.SetGlobalBuffer((__gm__ int64_t*)group_list);
        int64_t validK = GetQuantK(groupListGM, tilingData.groupNum, tilingData.groupListType, tilingData.K);
        if (validK == 0) {
            if (tilingData.clearPadding != 0) {
                if ASCEND_IS_AIV {
                    FlatQuantVec<DTYPE_X, MM_BASE_MODE> op;
                    op.Init(kronecker_p1, group_list, out, quant_scale, workspace, &tilingData);
                    op.Process();
                }
            }
            return;
        }
    }
    const FlatQuantTilingData* __restrict tiling_data = &tilingData;
    const TCubeTiling* __restrict mmTilingR = &(tiling_data->matmulTilingR);
    const TCubeTiling* __restrict mmTilingL = &(tiling_data->matmulTilingL);

    GM_ADDR userWS = GetUserWorkspace(workspace);

    if (TILING_KEY_IS(1)) {
        if ASCEND_IS_AIV {
            FlatQuantVec<DTYPE_X, MM_BASE_MODE> op;
            op.Init(kronecker_p1, group_list, out, quant_scale, workspace, &tilingData);
            op.Process();
        }
        if ASCEND_IS_AIC {
            FlatQuantCube<DTYPE_X, MM_BASE_MODE> op;
            op.Init(x, kronecker_p1, kronecker_p2, group_list, workspace, &tilingData);
            op.Process();
        }
    } else if (TILING_KEY_IS(2)) {
        if ASCEND_IS_AIV {
            FlatQuantVec<DTYPE_X, MM_DOUBLE_MODE> op;
            op.Init(kronecker_p1, group_list, out, quant_scale, workspace, &tilingData);
            op.Process();
        }
        if ASCEND_IS_AIC {
            FlatQuantCube<DTYPE_X, MM_DOUBLE_MODE> op;
            op.Init(x, kronecker_p1, kronecker_p2, group_list, workspace, &tilingData);
            op.Process();
        }
    } else if (TILING_KEY_IS(3)) {
        if ASCEND_IS_AIV {
            FlatQuantVec<DTYPE_X, MM_SPLIT_MODE> op;
            op.Init(kronecker_p1, group_list, out, quant_scale, workspace, &tilingData);
            op.Process();
        }
        if ASCEND_IS_AIC {
            FlatQuantCube<DTYPE_X, MM_SPLIT_MODE> op;
            op.Init(x, kronecker_p1, kronecker_p2, group_list, workspace, &tilingData);
            op.Process();
        }
    } else if (TILING_KEY_IS(4)) {
        FlatQuantHigh<DTYPE_X> op;
        REGIST_MATMUL_OBJ(&op.pipe, GetSysWorkSpacePtr(), op.matmulR, mmTilingR, op.matmulL, mmTilingL);
        op.Init(x, kronecker_p1, kronecker_p2, group_list, out, quant_scale, workspace, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(5)) {
        INVOKE_FLAT_QUANT_IMPL(DTYPE_X);
    }
}
