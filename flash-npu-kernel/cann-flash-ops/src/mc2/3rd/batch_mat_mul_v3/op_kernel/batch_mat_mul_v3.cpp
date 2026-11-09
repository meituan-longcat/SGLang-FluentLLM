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
 * \file batch_mat_mul_v3.cpp
 * \brief
 */
#include "./batch_mat_mul_v3.h"
#include "batch_mat_mul_v3_tiling_key.h"

using namespace AscendC;
using namespace matmul;
#ifndef DTYPE_BIAS
#define DTYPE_BIAS half
#endif

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

#define BMMV3_IMPL_CLASS(templateClass, ...)                                                                          \
    do {                                                                                                              \
        using cType = MatmulType<AscendC::TPosition::GM, format_y, DTYPE_Y, false, LayoutMode::NORMAL>;               \
        using biasType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_BIAS, false, LayoutMode::NORMAL>;   \
        TPipe pipe;                                                                                                   \
        if (tilingData.matmulTiling.matmulRunInfo.transA == 0 && tilingData.matmulTiling.matmulRunInfo.transB == 0) { \
            using aType = MatmulType<AscendC::TPosition::GM, format_x1, DTYPE_X1, false, LayoutMode::NORMAL>;         \
            using bType = MatmulType<AscendC::TPosition::GM, format_x2, DTYPE_X2, false, LayoutMode::NORMAL>;         \
            templateClass<aType, bType, cType, biasType, __VA_ARGS__> op;                                             \
            op.Init(aGM, bGM, cGM, biasGM, offsetWGM, user, &tilingData, &pipe);                                      \
            op.Process();                                                                                             \
        } else if (tilingData.matmulTiling.matmulRunInfo.transA == 1 &&                                               \
            tilingData.matmulTiling.matmulRunInfo.transB == 0) {                                                      \
            using aType = MatmulType<AscendC::TPosition::GM, format_x1, DTYPE_X1, true, LayoutMode::NORMAL>;          \
            using bType = MatmulType<AscendC::TPosition::GM, format_x2, DTYPE_X2, false, LayoutMode::NORMAL>;         \
            templateClass<aType, bType, cType, biasType, __VA_ARGS__> op;                                             \
            op.Init(aGM, bGM, cGM, biasGM, offsetWGM, user, &tilingData, &pipe);                                      \
            op.Process();                                                                                             \
        } else if (tilingData.matmulTiling.matmulRunInfo.transA == 0 &&                                               \
            tilingData.matmulTiling.matmulRunInfo.transB == 1) {                                                      \
            using aType = MatmulType<AscendC::TPosition::GM, format_x1, DTYPE_X1, false, LayoutMode::NORMAL>;         \
            using bType = MatmulType<AscendC::TPosition::GM, format_x2, DTYPE_X2, true, LayoutMode::NORMAL>;          \
            templateClass<aType, bType, cType, biasType, __VA_ARGS__> op;                                             \
            op.Init(aGM, bGM, cGM, biasGM, offsetWGM, user, &tilingData, &pipe);                                      \
            op.Process();                                                                                             \
        } else {                                                                                                      \
            using aType = MatmulType<AscendC::TPosition::GM, format_x1, DTYPE_X1, true, LayoutMode::NORMAL>;          \
            using bType = MatmulType<AscendC::TPosition::GM, format_x2, DTYPE_X2, true, LayoutMode::NORMAL>;          \
            templateClass<aType, bType, cType, biasType, __VA_ARGS__> op;                                             \
            op.Init(aGM, bGM, cGM, biasGM, offsetWGM, user, &tilingData, &pipe);                                      \
            op.Process();                                                                                             \
        }                                                                                                             \
    } while (0)

#define BMMV3_IMPL_CLASS_TRANS(transA, transB, templateClass, ...)                                                    \
    do {                                                                                                              \
        GET_TILING_DATA(tilingData, tilingGM);                                                                        \
        using cType = MatmulType<AscendC::TPosition::GM, format_y, DTYPE_Y, false, LayoutMode::NORMAL>;               \
        using biasType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_BIAS, false, LayoutMode::NORMAL>;   \
        TPipe pipe;                                                                                                   \
        using aType = MatmulType<AscendC::TPosition::GM, format_x1, DTYPE_X1, transA, LayoutMode::NORMAL>;            \
        using bType = MatmulType<AscendC::TPosition::GM, format_x2, DTYPE_X2, transB, LayoutMode::NORMAL>;            \
        templateClass<aType, bType, cType, biasType, __VA_ARGS__> op;                                                 \
        op.Init(aGM, bGM, cGM, biasGM, offsetWGM, user, &tilingData, &pipe);                                          \
        op.Process();                                                                                                 \
    } while (0)

#define BMMV3_IMPL_CLASS_COMMON(templateClass, ...)                                                                   \
    do {                                                                                                              \
        using cType = MatmulType<AscendC::TPosition::GM, format_y, DTYPE_Y>;                                          \
        using biasType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_BIAS>;                              \
        TPipe pipe;                                                                                                   \
        if (tilingData.matmulTiling.matmulRunInfo.transA == 0 && tilingData.matmulTiling.matmulRunInfo.transB == 0) { \
            using aType = MatmulType<AscendC::TPosition::GM, format_x1, DTYPE_X1, false>;                             \
            using bType = MatmulType<AscendC::TPosition::GM, format_x2, DTYPE_X2, false>;                             \
            templateClass<aType, bType, cType, biasType, __VA_ARGS__> op;                                             \
            op.Init(aGM, bGM, cGM, biasGM, offsetWGM, user, &tilingData, &pipe);                                      \
            op.Process();                                                                                             \
        } else if (tilingData.matmulTiling.matmulRunInfo.transA == 1 &&                                               \
            tilingData.matmulTiling.matmulRunInfo.transB == 0) {                                                      \
            using aType = MatmulType<AscendC::TPosition::GM, format_x1, DTYPE_X1, true>;                              \
            using bType = MatmulType<AscendC::TPosition::GM, format_x2, DTYPE_X2, false>;                             \
            templateClass<aType, bType, cType, biasType, __VA_ARGS__> op;                                             \
            op.Init(aGM, bGM, cGM, biasGM, offsetWGM, user, &tilingData, &pipe);                                      \
            op.Process();                                                                                             \
        } else if (tilingData.matmulTiling.matmulRunInfo.transA == 0 &&                                               \
            tilingData.matmulTiling.matmulRunInfo.transB == 1) {                                                      \
            using aType = MatmulType<AscendC::TPosition::GM, format_x1, DTYPE_X1, false>;                             \
            using bType = MatmulType<AscendC::TPosition::GM, format_x2, DTYPE_X2, true>;                              \
            templateClass<aType, bType, cType, biasType, __VA_ARGS__> op;                                             \
            op.Init(aGM, bGM, cGM, biasGM, offsetWGM, user, &tilingData, &pipe);                                      \
            op.Process();                                                                                             \
        } else {                                                                                                      \
            using aType = MatmulType<AscendC::TPosition::GM, format_x1, DTYPE_X1, true>;                              \
            using bType = MatmulType<AscendC::TPosition::GM, format_x2, DTYPE_X2, true>;                              \
            templateClass<aType, bType, cType, biasType, __VA_ARGS__> op;                                             \
            op.Init(aGM, bGM, cGM, biasGM, offsetWGM, user, &tilingData, &pipe);                                      \
            op.Process();                                                                                             \
        }                                                                                                             \
    } while (0)

#define BMMV3_IMPL_CLASS_COMMON_TRNAS(transA, transB, templateClass, ...)                                             \
    do {                                                                                                              \
        GET_TILING_DATA(tilingData, tilingGM);                                                                        \
        using cType = MatmulType<AscendC::TPosition::GM, format_y, DTYPE_Y>;                                          \
        using biasType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_BIAS>;                              \
        TPipe pipe;                                                                                                   \
        using aType = MatmulType<AscendC::TPosition::GM, format_x1, DTYPE_X1, transA>;                                \
        using bType = MatmulType<AscendC::TPosition::GM, format_x2, DTYPE_X2, transB>;                                \
        templateClass<aType, bType, cType, biasType, __VA_ARGS__> op;                                                 \
        op.Init(aGM, bGM, cGM, biasGM, offsetWGM, user, &tilingData, &pipe);                                          \
        op.Process();                                                                                                 \
    } while (0)

template<int MULTIBATCHL1FULLLOAD, int MULTIBATCH, int LOADMODE, int ISMULTIBATCHOUT, int MIXND2NZ>
__global__ __aicore__ void batch_mat_mul_v3(
    GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM, GM_ADDR offsetWGM, GM_ADDR cGM, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    REGISTER_TILING_DEFAULT(BatchMatmulTilingData);
    __gm__ uint8_t *user = GetUserWorkspace(workspaceGM);
#if defined(__CCE_AICORE__) && __CCE_AICORE__ < 220
    GET_TILING_DATA(tilingData, tilingGM);
    if (MULTIBATCHL1FULLLOAD == BATCH_MAT_MUL_V3_MULTI_BATCH_L1_FULLLOAD_FALSE &&
        MULTIBATCH == BATCH_MAT_MUL_V3_MULTI_BATCH_FALSE && LOADMODE == BATCH_MAT_MUL_V3_BASE_FULLLOAD &&
        ISMULTIBATCHOUT == BATCH_MAT_MUL_V3_ISMULTIBATCHOUT_FALSE && MIXND2NZ == BATCH_MAT_MUL_V3_MIXND2NZ_FALSE) {
        BMMV3_IMPL_CLASS_COMMON(Mc2BatchMatMulCommonKernel, Mc2BatchMatMulCommonBaseBlock, MM_CFG_VEC_ND2NZ);
    }
#else
    GET_TILING_DATA(tilingData, tilingGM);
    if (MULTIBATCHL1FULLLOAD == BATCH_MAT_MUL_V3_MULTI_BATCH_L1_FULLLOAD_FALSE &&
        MULTIBATCH == BATCH_MAT_MUL_V3_MULTI_BATCH_FALSE && LOADMODE == BATCH_MAT_MUL_V3_BASE_FULLLOAD &&
        ISMULTIBATCHOUT == BATCH_MAT_MUL_V3_ISMULTIBATCHOUT_FALSE && MIXND2NZ == BATCH_MAT_MUL_V3_MIXND2NZ_TRUE) {
        BMMV3_IMPL_CLASS_COMMON(Mc2BatchMatMulUnalignedKernel, Mc2MatmulBaseBlock, MM_CFG_NO_PRELOAD);
    } else if (
        MULTIBATCHL1FULLLOAD == BATCH_MAT_MUL_V3_MULTI_BATCH_L1_FULLLOAD_FALSE &&
        MULTIBATCH == BATCH_MAT_MUL_V3_MULTI_BATCH_FALSE && LOADMODE == BATCH_MAT_MUL_V3_BASE_FULLLOAD &&
        ISMULTIBATCHOUT == BATCH_MAT_MUL_V3_ISMULTIBATCHOUT_FALSE && MIXND2NZ == BATCH_MAT_MUL_V3_MIXND2NZ_FALSE) {
        BMMV3_IMPL_CLASS_COMMON(Mc2BatchMatMulCommonKernel, Mc2BatchMatMulCommonBaseBlock, MM_CFG_NO_PRELOAD);
    } else if (
        MULTIBATCHL1FULLLOAD == BATCH_MAT_MUL_V3_MULTI_BATCH_L1_FULLLOAD_FALSE &&
        MULTIBATCH == BATCH_MAT_MUL_V3_MULTI_BATCH_FALSE && LOADMODE == BATCH_MAT_MUL_V3_AL1_FULLLOAD &&
        ISMULTIBATCHOUT == BATCH_MAT_MUL_V3_ISMULTIBATCHOUT_FALSE && MIXND2NZ == BATCH_MAT_MUL_V3_MIXND2NZ_FALSE) {
        BMMV3_IMPL_CLASS_COMMON(Mc2BatchMatMulCommonKernel, Mc2BatchMatMulCommonBaseBlock, MM_CFG_NO_PRELOAD,
                                MatmulCallBackFunc<nullptr, BmmCopyAL1, nullptr>);
    } else if (
        MULTIBATCHL1FULLLOAD == BATCH_MAT_MUL_V3_MULTI_BATCH_L1_FULLLOAD_FALSE &&
        MULTIBATCH == BATCH_MAT_MUL_V3_MULTI_BATCH_FALSE && LOADMODE == BATCH_MAT_MUL_V3_BL1_FULLLOAD &&
        ISMULTIBATCHOUT == BATCH_MAT_MUL_V3_ISMULTIBATCHOUT_FALSE && MIXND2NZ == BATCH_MAT_MUL_V3_MIXND2NZ_FALSE) {
        BMMV3_IMPL_CLASS_COMMON(Mc2BatchMatMulCommonKernel, Mc2BatchMatMulCommonBaseBlock, MM_CFG_NO_PRELOAD,
                                MatmulCallBackFunc<nullptr, nullptr, BmmCopyBL1>);
    } else if (
        MULTIBATCHL1FULLLOAD == BATCH_MAT_MUL_V3_MULTI_BATCH_L1_FULLLOAD_FALSE && // need to be set
        MULTIBATCH == BATCH_MAT_MUL_V3_MULTI_BATCH_TRUE && LOADMODE == BATCH_MAT_MUL_V3_BASE_FULLLOAD &&
        ISMULTIBATCHOUT == BATCH_MAT_MUL_V3_ISMULTIBATCHOUT_FALSE && MIXND2NZ == BATCH_MAT_MUL_V3_MIXND2NZ_FALSE) {
        BMMV3_IMPL_CLASS(Mc2BatchMatMulMultiBatchKernel, Mc2BatchMatMulMultiBatchBaseBlock);
    } else if (
        MULTIBATCHL1FULLLOAD == BATCH_MAT_MUL_V3_MULTI_BATCH_L1_FULLLOAD_TRUE &&
        MULTIBATCH == BATCH_MAT_MUL_V3_MULTI_BATCH_FALSE && LOADMODE == BATCH_MAT_MUL_V3_BASE_FULLLOAD &&
        ISMULTIBATCHOUT == BATCH_MAT_MUL_V3_ISMULTIBATCHOUT_FALSE && MIXND2NZ == BATCH_MAT_MUL_V3_MIXND2NZ_FALSE) {
        BMMV3_IMPL_CLASS(Mc2BatchMatMulMultiBatchFullLoadKernel, Mc2BatchMatMulMultiBatchFullLoadBlock);
    } else if (
        MULTIBATCHL1FULLLOAD == BATCH_MAT_MUL_V3_MULTI_BATCH_L1_FULLLOAD_FALSE &&
        MULTIBATCH == BATCH_MAT_MUL_V3_MULTI_BATCH_TRUE && LOADMODE == BATCH_MAT_MUL_V3_BASE_FULLLOAD &&
        ISMULTIBATCHOUT == BATCH_MAT_MUL_V3_ISMULTIBATCHOUT_FALSE && MIXND2NZ == BATCH_MAT_MUL_V3_MIXND2NZ_TRUE) {
        BMMV3_IMPL_CLASS(Mc2BatchMatMulUnalignedMultiBatchKernel, Mc2BatchMatMulUnalignedMultiBatchBaseBlock);
    } else if (
        MULTIBATCHL1FULLLOAD == BATCH_MAT_MUL_V3_MULTI_BATCH_L1_FULLLOAD_FALSE &&
        MULTIBATCH == BATCH_MAT_MUL_V3_MULTI_BATCH_TRUE && LOADMODE == BATCH_MAT_MUL_V3_BASE_FULLLOAD &&
        ISMULTIBATCHOUT == BATCH_MAT_MUL_V3_ISMULTIBATCHOUT_TRUE && MIXND2NZ == BATCH_MAT_MUL_V3_MIXND2NZ_FALSE) {
        BMMV3_IMPL_CLASS(Mc2BatchMatMulMultiBatchKernel, Mc2BatchMatMulMultiBatchBaseBlock, MM_CFG_MULTI_BATCH_OUT);
    } else if (
        MULTIBATCHL1FULLLOAD == BATCH_MAT_MUL_V3_MULTI_BATCH_L1_FULLLOAD_FALSE &&
        MULTIBATCH == BATCH_MAT_MUL_V3_MULTI_BATCH_TRUE && LOADMODE == BATCH_MAT_MUL_V3_BASE_FULLLOAD &&
        ISMULTIBATCHOUT == BATCH_MAT_MUL_V3_ISMULTIBATCHOUT_TRUE && MIXND2NZ == BATCH_MAT_MUL_V3_MIXND2NZ_TRUE) {
        BMMV3_IMPL_CLASS(Mc2BatchMatMulUnalignedMultiBatchKernel, Mc2BatchMatMulUnalignedMultiBatchBaseBlock, MM_CFG_MULTI_BATCH_OUT);
    }
#endif
}