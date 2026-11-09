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
#include "arch35/flat_quant_high_v2.h"
#include "arch35/flat_quant_basic_cmct.h"
#include "flat_quant_vec.h"
#include "flat_quant_cube.h"
#include "flat_quant_vec_one.h"
#include "flat_quant_cube_one.h"
#include "flat_quant_high.h"

using namespace FlatQuantNS;
using namespace Cmct;
using namespace Cmct::Gemm;

extern "C" __global__ __aicore__ void flat_quant(GM_ADDR x, GM_ADDR kronecker_p1, GM_ADDR kronecker_p2,
                                                 GM_ADDR group_list, GM_ADDR out, GM_ADDR quant_scale,
                                                 GM_ADDR workspace, GM_ADDR tiling)
{
    AscendC::InitSocState();
    GET_TILING_DATA(tilingData, tiling);
    if (tilingData.K == 0) {
        return;
    }
    const FlatQuantTilingData* __restrict tiling_data = &tilingData;
    const TCubeTiling* __restrict mmTilingR = &(tiling_data->matmulTilingR);
    const TCubeTiling* __restrict mmTilingL = &(tiling_data->matmulTilingL);
    if constexpr (std::is_same<DTYPE_OUT, fp4x2_e2m1_t>::value) {
        if (TILING_KEY_IS(4)) {
            FlatQuantHighV2<DTYPE_X> op;
            REGIST_MATMUL_OBJ(&op.pipe, GetSysWorkSpacePtr(), op.matmulR, mmTilingR, op.matmulL, mmTilingL);
            op.Init(x, kronecker_p1, kronecker_p2, out, quant_scale, workspace, &tilingData);
            op.Process();
        } else if (TILING_KEY_IS(6)) {
            FlatQuantCmctKernel<DTYPE_X, DTYPE_OUT, DTYPE_QUANT_SCALE, layout::RowMajor>(
                x, kronecker_p1, kronecker_p2, out, quant_scale, workspace, tilingData);
        }
    } else {
        if (tilingData.groupNum > 0 && tilingData.clearPadding == 0) {
            GlobalTensor<int64_t> groupListGM;
            groupListGM.SetGlobalBuffer((__gm__ int64_t*)group_list);
            int64_t validK = GetQuantK(groupListGM, tilingData.groupNum, tilingData.groupListType, tilingData.K);
            if (validK == 0) {
                return;
            }
        }
        if (TILING_KEY_IS(4)) {
            FlatQuantHigh<DTYPE_X> op;
            REGIST_MATMUL_OBJ(&op.pipe, GetSysWorkSpacePtr(), op.matmulR, mmTilingR, op.matmulL, mmTilingL);
            op.Init(x, kronecker_p1, kronecker_p2, group_list, out, quant_scale, workspace, &tilingData);
            op.Process();
        }
    }
}
