/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file quant_batch_matmul_v3_apt.cpp
 * \brief
 */
#include "arch35/quant_batch_matmul_v3_tiling_data.h"
#include "arch35/qbmm_cube_on_the_fly.h"
#include "arch35/qbmm_cube_on_the_fly_al1_full_load.h"
#if (ORIG_DTYPE_SCALE == DT_FLOAT || ORIG_DTYPE_SCALE == DT_BF16)
#include "arch35/qbmm_mix_online_dynamic.h"
#include "arch35/qbmm_mix_online_dynamic_al1_full_load.h"
#include "arch35/qbmm_mix_perblock.h"
#endif
#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif

// if run with ttk without bias, can't get DTYPE_BIAS macro
#undef DTYPE_BIAS
#if defined(ORIG_DTYPE_X1) && defined(DT_INT8) && ORIG_DTYPE_X1 == DT_INT8
    // s8->s32
    #define DTYPE_BIAS int32_t
#else
    // fp8/hif8->fp32
    #define DTYPE_BIAS float
#endif

#if (defined(ORIG_DTYPE_X1) && defined(ORIG_DTYPE_SCALE))
#define SUPPORT_PERBLOCK             \
    (ORIG_DTYPE_SCALE == DT_FLOAT && \
     (ORIG_DTYPE_X1 == DT_HIFLOAT8 || ORIG_DTYPE_X1 == DT_FLOAT8_E4M3FN || ORIG_DTYPE_X1 == DT_FLOAT8_E5M2))
#else
#define SUPPORT_PERBLOCK false
#endif

#undef ORIG_DTYPE_PERTOKEN_SCALE
#undef DTYPE_PERTOKEN_SCALE
#define ORIG_DTYPE_PERTOKEN_SCALE DT_FLOAT
#define DTYPE_PERTOKEN_SCALE float

using namespace AscendC;
using namespace matmul;

#ifndef FORMAT_FRACTAL_NZ
#define FORMAT_FRACTAL_NZ
#endif

#if defined(FORMAT_X1) && FORMAT_X1 == FORMAT_FRACTAL_NZ
constexpr CubeFormat format_x1 = CubeFormat::NZ;
#else
constexpr CubeFormat format_x1 = CubeFormat::ND;
#endif

#if defined(FORMAT_X2) && FORMAT_X2 == FORMAT_FRACTAL_NZ
constexpr CubeFormat format_x2 = CubeFormat::NZ;
#else
constexpr CubeFormat format_x2 = CubeFormat::ND;
#endif

#if defined(FORMAT_Y) && FORMAT_Y == FORMAT_FRACTAL_NZ
constexpr CubeFormat format_y = CubeFormat::NZ;
#else
constexpr CubeFormat format_y = CubeFormat::ND;
#endif

#if defined(ORIG_DTYPE_X1) && defined(DT_INT8) && ORIG_DTYPE_X1 == DT_INT8
    #define DTYPE_LOC_LOCAL int32_t
#else
    #define DTYPE_LOC_LOCAL float
#endif

#define QUANT_BMMV3_IMPL_CLASS(formatX1, formatX2, formatY, transposeX1, transposeX2, templateClass, ...)             \
    do {                                                                                                              \
        templateClass<DTYPE_X1, DTYPE_X2, DTYPE_SCALE, DTYPE_BIAS, DTYPE_PERTOKEN_SCALE, DTYPE_Y, formatX1, formatX2, \
                      formatY, transposeX1, transposeX2, DTYPE_LOC_LOCAL, __VA_ARGS__>                                \
            op;                                                                                                       \
        op.Init(x1, x2, scale, offset, bias, pertokenScale, y, user1, &tilingData, &tPipe);                            \
        op.Process();                                                                                                 \
    } while (0)

extern "C" __global__ __aicore__ void quant_batch_matmul_v3(
    GM_ADDR x1, GM_ADDR x2, GM_ADDR scale, GM_ADDR offset, GM_ADDR bias, GM_ADDR pertokenScale, GM_ADDR y,
    GM_ADDR workSpace, GM_ADDR tiling)
{
    if (workSpace == nullptr) {
        return;
    }
    TPipe tPipe;
    GM_ADDR user1 = GetUserWorkspace(workSpace);
    if (user1 == nullptr) {
        return;
    }
    REGISTER_TILING_DEFAULT(DequantBmm::Mc2QuantBatchMatmulV3TilingDataParams);
    GET_TILING_DATA(tilingData, tiling);

#if (ORIG_DTYPE_SCALE == DT_FLOAT || ORIG_DTYPE_SCALE == DT_BF16)
    if (TILING_KEY_IS(2000)) {
        KERNEL_TASK_TYPE(2000, KERNEL_TYPE_MIX_AIC_1_2);
        QUANT_BMMV3_IMPL_CLASS(format_x1, format_x2, format_y, false, false,
                                Mc2QuantBatchMatmulV3::Mc2QuantBmmPertokenRegbaseKernel, Mc2QuantBatchMatmulV3::Mc2QuantBmmAswBlock,
                                MM_CFG_NO_PRELOAD_OPEN_UNIT_FLAG);
    } else if (TILING_KEY_IS(2001)) {
        KERNEL_TASK_TYPE(2001, KERNEL_TYPE_MIX_AIC_1_2);
        QUANT_BMMV3_IMPL_CLASS(format_x1, format_x2, format_y, false, true,
                                Mc2QuantBatchMatmulV3::Mc2QuantBmmPertokenRegbaseKernel, Mc2QuantBatchMatmulV3::Mc2QuantBmmAswBlock,
                                MM_CFG_NO_PRELOAD_OPEN_UNIT_FLAG);
    } else if (TILING_KEY_IS(2010)) {
        KERNEL_TASK_TYPE(2010, KERNEL_TYPE_MIX_AIC_1_2);
        QUANT_BMMV3_IMPL_CLASS(format_x1, format_x2, format_y, true, false,
                                Mc2QuantBatchMatmulV3::Mc2QuantBmmPertokenRegbaseKernel, Mc2QuantBatchMatmulV3::Mc2QuantBmmAswBlock,
                                MM_CFG_NO_PRELOAD_OPEN_UNIT_FLAG);

    } else if (TILING_KEY_IS(2011)) {
        KERNEL_TASK_TYPE(2011, KERNEL_TYPE_MIX_AIC_1_2);
        QUANT_BMMV3_IMPL_CLASS(format_x1, format_x2, format_y, true, true,
                                Mc2QuantBatchMatmulV3::Mc2QuantBmmPertokenRegbaseKernel, Mc2QuantBatchMatmulV3::Mc2QuantBmmAswBlock,
                                MM_CFG_NO_PRELOAD_OPEN_UNIT_FLAG);
    } else if (TILING_KEY_IS(3000)) {
        KERNEL_TASK_TYPE(3000, KERNEL_TYPE_MIX_AIC_1_2);
        QUANT_BMMV3_IMPL_CLASS(format_x1, format_x2, format_y, false, false,
                                Mc2QuantBatchMatmulV3::Mc2QuantBmmPertokenAL1FullLoad, Mc2QuantBatchMatmulV3::Mc2QuantBmmAswBlock,
                                MM_CFG_NO_PRELOAD_OPEN_UNIT_FLAG);
    } else if (TILING_KEY_IS(3001)) {
        KERNEL_TASK_TYPE(3001, KERNEL_TYPE_MIX_AIC_1_2);
        QUANT_BMMV3_IMPL_CLASS(format_x1, format_x2, format_y, false, true,
                                Mc2QuantBatchMatmulV3::Mc2QuantBmmPertokenAL1FullLoad, Mc2QuantBatchMatmulV3::Mc2QuantBmmAswBlock,
                                MM_CFG_NO_PRELOAD_OPEN_UNIT_FLAG);
    } else if (TILING_KEY_IS(3010)) {
        KERNEL_TASK_TYPE(3010, KERNEL_TYPE_MIX_AIC_1_2);
        QUANT_BMMV3_IMPL_CLASS(format_x1, format_x2, format_y, true, false,
                                Mc2QuantBatchMatmulV3::Mc2QuantBmmPertokenAL1FullLoad, Mc2QuantBatchMatmulV3::Mc2QuantBmmAswBlock,
                                MM_CFG_NO_PRELOAD_OPEN_UNIT_FLAG);
    } else if (TILING_KEY_IS(3011)) {
        KERNEL_TASK_TYPE(3011, KERNEL_TYPE_MIX_AIC_1_2);
        QUANT_BMMV3_IMPL_CLASS(format_x1, format_x2, format_y, true, true,
                                Mc2QuantBatchMatmulV3::Mc2QuantBmmPertokenAL1FullLoad, Mc2QuantBatchMatmulV3::Mc2QuantBmmAswBlock,
                                MM_CFG_NO_PRELOAD_OPEN_UNIT_FLAG);
    }
#endif
    if constexpr (DequantBmm::IsMxType<DTYPE_SCALE>()) {
        if (TILING_KEY_IS(0)) {
            KERNEL_TASK_TYPE(0, KERNEL_TYPE_AIC_ONLY);
            MatMulASWKernel<DTYPE_X1, DTYPE_X2, DTYPE_SCALE, DTYPE_BIAS, DTYPE_Y, format_x1, format_x2, format_y, false,
                            false>
                op;
            op.Init(x1, x2, bias, scale, pertokenScale, y, user1, &tilingData, &tPipe);
            op.Process();
        } else if (TILING_KEY_IS(1)) {
            KERNEL_TASK_TYPE(1, KERNEL_TYPE_AIC_ONLY);
            MatMulASWKernel<DTYPE_X1, DTYPE_X2, DTYPE_SCALE, DTYPE_BIAS, DTYPE_Y, format_x1, format_x2, format_y, false,
                            true>
                op;
            op.Init(x1, x2, bias, scale, pertokenScale, y, user1, &tilingData, &tPipe);
            op.Process();
        } else if (TILING_KEY_IS(10)) {
            KERNEL_TASK_TYPE(10, KERNEL_TYPE_AIC_ONLY);
            MatMulASWKernel<DTYPE_X1, DTYPE_X2, DTYPE_SCALE, DTYPE_BIAS, DTYPE_Y, format_x1, format_x2, format_y, true,
                            false>
                op;
            op.Init(x1, x2, bias, scale, pertokenScale, y, user1, &tilingData, &tPipe);
            op.Process();
        } else if (TILING_KEY_IS(11)) {
            KERNEL_TASK_TYPE(11, KERNEL_TYPE_AIC_ONLY);
            MatMulASWKernel<DTYPE_X1, DTYPE_X2, DTYPE_SCALE, DTYPE_BIAS, DTYPE_Y, format_x1, format_x2, format_y, true,
                            true>
                op;
            op.Init(x1, x2, bias, scale, pertokenScale, y, user1, &tilingData, &tPipe);
            op.Process();
        } else if (TILING_KEY_IS(1001)) {
            KERNEL_TASK_TYPE(1001, KERNEL_TYPE_AIC_ONLY);
            Mc2QuantBatchMatmulV3::MatmulAswKernelAL1FullLoad<DTYPE_X1, DTYPE_X2, DTYPE_SCALE, DTYPE_BIAS, DTYPE_Y,
                                                            format_x1, format_x2, format_y, false, true>
                op;
            op.Init(x1, x2, bias, scale, pertokenScale, y, user1, &tilingData, &tPipe);
            op.Process();
        }
    } else {
        if (TILING_KEY_IS(0)) {
            KERNEL_TASK_TYPE(0, KERNEL_TYPE_AIC_ONLY);
            MatMulASWKernel<DTYPE_X1, DTYPE_X2, DTYPE_SCALE, DTYPE_BIAS, DTYPE_Y, format_x1, format_x2, format_y, false,
                            false>
                op;
            op.Init(x1, x2, bias, scale, pertokenScale, y, user1, &tilingData, &tPipe);
            op.Process();
        } else if (TILING_KEY_IS(1)) {
            KERNEL_TASK_TYPE(1, KERNEL_TYPE_AIC_ONLY);
            MatMulASWKernel<DTYPE_X1, DTYPE_X2, DTYPE_SCALE, DTYPE_BIAS, DTYPE_Y, format_x1, format_x2, format_y, false,
                            true>
                op;
            op.Init(x1, x2, bias, scale, pertokenScale, y, user1, &tilingData, &tPipe);
            op.Process();
        } else if (TILING_KEY_IS(10)) {
            KERNEL_TASK_TYPE(10, KERNEL_TYPE_AIC_ONLY);
            MatMulASWKernel<DTYPE_X1, DTYPE_X2, DTYPE_SCALE, DTYPE_BIAS, DTYPE_Y, format_x1, format_x2, format_y, true,
                            false>
                op;
            op.Init(x1, x2, bias, scale, pertokenScale, y, user1, &tilingData, &tPipe);
            op.Process();
        } else if (TILING_KEY_IS(11)) {
            KERNEL_TASK_TYPE(11, KERNEL_TYPE_AIC_ONLY);
            MatMulASWKernel<DTYPE_X1, DTYPE_X2, DTYPE_SCALE, DTYPE_BIAS, DTYPE_Y, format_x1, format_x2, format_y, true,
                            true>
                op;
            op.Init(x1, x2, bias, scale, pertokenScale, y, user1, &tilingData, &tPipe);
            op.Process();
        } else if (TILING_KEY_IS(1000)) {
            KERNEL_TASK_TYPE(1000, KERNEL_TYPE_AIC_ONLY);
            Mc2QuantBatchMatmulV3::MatmulAswKernelAL1FullLoad<DTYPE_X1, DTYPE_X2, DTYPE_SCALE, DTYPE_BIAS, DTYPE_Y,
                                                            format_x1, format_x2, format_y, false, false>
                op;
            op.Init(x1, x2, bias, scale, pertokenScale, y, user1, &tilingData, &tPipe);
            op.Process();
        } else if (TILING_KEY_IS(1001)) {
            KERNEL_TASK_TYPE(1001, KERNEL_TYPE_AIC_ONLY);
            Mc2QuantBatchMatmulV3::MatmulAswKernelAL1FullLoad<DTYPE_X1, DTYPE_X2, DTYPE_SCALE, DTYPE_BIAS, DTYPE_Y,
                                                            format_x1, format_x2, format_y, false, true>
                op;
            op.Init(x1, x2, bias, scale, pertokenScale, y, user1, &tilingData, &tPipe);
            op.Process();
        } else if (TILING_KEY_IS(1010)) {
            KERNEL_TASK_TYPE(1010, KERNEL_TYPE_AIC_ONLY);
            Mc2QuantBatchMatmulV3::MatmulAswKernelAL1FullLoad<DTYPE_X1, DTYPE_X2, DTYPE_SCALE, DTYPE_BIAS, DTYPE_Y,
                                                            format_x1, format_x2, format_y, true, false>
                op;
            op.Init(x1, x2, bias, scale, pertokenScale, y, user1, &tilingData, &tPipe);
            op.Process();
        } else if (TILING_KEY_IS(1011)) {
            KERNEL_TASK_TYPE(1011, KERNEL_TYPE_AIC_ONLY);
            Mc2QuantBatchMatmulV3::MatmulAswKernelAL1FullLoad<DTYPE_X1, DTYPE_X2, DTYPE_SCALE, DTYPE_BIAS, DTYPE_Y,
                                                            format_x1, format_x2, format_y, true, true>
                op;
            op.Init(x1, x2, bias, scale, pertokenScale, y, user1, &tilingData, &tPipe);
            op.Process();
        }
    }

#if SUPPORT_PERBLOCK
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    if (TILING_KEY_IS(4000)) {
        Mc2QuantBatchMatmulV3::MatMulPerBlockASW<DTYPE_X1, DTYPE_X2, DTYPE_BIAS, DTYPE_Y, format_x1, format_x2, format_y,
                                                false, false> op;
        op.Init(x1, x2, bias, scale, pertokenScale, y, user1, &tilingData, &tPipe);
        op.Process();
    } else if (TILING_KEY_IS(4001)) {
        Mc2QuantBatchMatmulV3::MatMulPerBlockASW<DTYPE_X1, DTYPE_X2, DTYPE_BIAS, DTYPE_Y, format_x1, format_x2, format_y,
                                                false, true> op;
        op.Init(x1, x2, bias, scale, pertokenScale, y, user1, &tilingData, &tPipe);
        op.Process();
    } else if (TILING_KEY_IS(4010)) {
        Mc2QuantBatchMatmulV3::MatMulPerBlockASW<DTYPE_X1, DTYPE_X2, DTYPE_BIAS, DTYPE_Y, format_x1, format_x2, format_y,
                                                true, false> op;
        op.Init(x1, x2, bias, scale, pertokenScale, y, user1, &tilingData, &tPipe);
        op.Process();
    } else if (TILING_KEY_IS(4011)) {
        Mc2QuantBatchMatmulV3::MatMulPerBlockASW<DTYPE_X1, DTYPE_X2, DTYPE_BIAS, DTYPE_Y, format_x1, format_x2, format_y,
                                                true, true> op;
        op.Init(x1, x2, bias, scale, pertokenScale, y, user1, &tilingData, &tPipe);
        op.Process();
    }
#endif
}