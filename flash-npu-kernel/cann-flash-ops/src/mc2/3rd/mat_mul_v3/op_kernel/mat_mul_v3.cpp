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
 * \file mat_mul_v3.cpp
 * \brief
 */

#include "mat_mul_v3_common.h"
#include "mat_mul_base_kernel.h"
#include "mat_mul_deterministic_splitk_kernel.h"
#include "mat_mul_sc_splitk_kernel.h"
#include "mat_mul_sc_splitk_kernel_gm_to_l1.h"
#include "mat_mul_unaligned_base_kernel.h"
#include "mat_mul_cvp_base_kernel.h"
#include "mat_mul_unaligned_deterministic_splitk_kernel.h"
#include "mat_mul_unaligned_sc_splitk_kernel.h"
#include "mat_mul_unaligned_sc_splitk_kernel_gm_to_l1.h"
#include "mat_mul_optimized_fixpipe_algorithm.h"
#include "mat_mul_l1_full_load.h"
#include "mat_mul_v3_tiling_key.h"


#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
#include "mat_mul_multi_core_splitk_kernel.h"
#endif

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

#define MMV3_IMPL(templateFunc, cFormat, ...)                                                                                 \
    do {                                                                                                             \
        using cType = MatmulType<AscendC::TPosition::GM, cFormat, DTYPE_Y>;                                         \
        using biasType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_BIAS>;                             \
        if (tilingData.matmulRunInfo.transA == 0 && tilingData.matmulRunInfo.transB == 0) {                          \
            using aType = MatmulType<AscendC::TPosition::GM, format_x1, DTYPE_X1, false>;                            \
            using bType = MatmulType<AscendC::TPosition::GM, format_x2, DTYPE_X2, false>;                            \
            templateFunc<aType, bType, cType, biasType, __VA_ARGS__>(aGM, bGM, cGM, biasGM, tilingData, user);                    \
        } else if (tilingData.matmulRunInfo.transA == 1 && tilingData.matmulRunInfo.transB == 0) {                   \
            using aType = MatmulType<AscendC::TPosition::GM, format_x1, DTYPE_X1, true>;                             \
            using bType = MatmulType<AscendC::TPosition::GM, format_x2, DTYPE_X2, false>;                            \
            templateFunc<aType, bType, cType, biasType, __VA_ARGS__>(aGM, bGM, cGM, biasGM, tilingData, user);                    \
        } else if (tilingData.matmulRunInfo.transA == 0 && tilingData.matmulRunInfo.transB == 1) {                   \
            using aType = MatmulType<AscendC::TPosition::GM, format_x1, DTYPE_X1, false>;                            \
            using bType = MatmulType<AscendC::TPosition::GM, format_x2, DTYPE_X2, true>;                             \
            templateFunc<aType, bType, cType, biasType, __VA_ARGS__>(aGM, bGM, cGM, biasGM, tilingData, user);                    \
        } else {                                                                                                     \
            using aType = MatmulType<AscendC::TPosition::GM, format_x1, DTYPE_X1, true>;                             \
            using bType = MatmulType<AscendC::TPosition::GM, format_x2, DTYPE_X2, true>;                             \
            templateFunc<aType, bType, cType, biasType, __VA_ARGS__>(aGM, bGM, cGM, biasGM, tilingData, user);                    \
        }                                                                                                            \
    } while(0)

#define MMV3_IMPL_CLASS(templateClass, aFormat, ...)                                                                 \
    do {                                                                                                             \
        using cType = MatmulType<AscendC::TPosition::GM, format_y, DTYPE_Y>;                                         \
        using biasType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_BIAS>;                             \
        TPipe pipe;                                                                                                  \
        if (tilingData.matmulRunInfo.transA == 0 && tilingData.matmulRunInfo.transB == 0) {                          \
            using aType = MatmulType<AscendC::TPosition::GM, aFormat, DTYPE_X1, false>;                              \
            using bType = MatmulType<AscendC::TPosition::GM, format_x2, DTYPE_X2, false>;                            \
            templateClass<aType, bType, cType, biasType, __VA_ARGS__> op;                                            \
            op.Init(aGM, bGM, cGM, biasGM, offsetWGM, user, &tilingData, &pipe);                                     \
            op.Process();                                                                                            \
        } else if (tilingData.matmulRunInfo.transA == 1 && tilingData.matmulRunInfo.transB == 0) {                   \
            using aType = MatmulType<AscendC::TPosition::GM, aFormat, DTYPE_X1, true>;                               \
            using bType = MatmulType<AscendC::TPosition::GM, format_x2, DTYPE_X2, false>;                            \
            templateClass<aType, bType, cType, biasType, __VA_ARGS__> op;                                            \
            op.Init(aGM, bGM, cGM, biasGM, offsetWGM, user, &tilingData, &pipe);                                     \
            op.Process();                                                                                            \
        } else if (tilingData.matmulRunInfo.transA == 0 && tilingData.matmulRunInfo.transB == 1) {                   \
            using aType = MatmulType<AscendC::TPosition::GM, aFormat, DTYPE_X1, false>;                              \
            using bType = MatmulType<AscendC::TPosition::GM, format_x2, DTYPE_X2, true>;                             \
            templateClass<aType, bType, cType, biasType, __VA_ARGS__> op;                                            \
            op.Init(aGM, bGM, cGM, biasGM, offsetWGM, user, &tilingData, &pipe);                                     \
            op.Process();                                                                                            \
        } else {                                                                                                     \
            using aType = MatmulType<AscendC::TPosition::GM, aFormat, DTYPE_X1, true>;                               \
            using bType = MatmulType<AscendC::TPosition::GM, format_x2, DTYPE_X2, true>;                             \
            templateClass<aType, bType, cType, biasType, __VA_ARGS__> op;                                            \
            op.Init(aGM, bGM, cGM, biasGM, offsetWGM, user, &tilingData, &pipe);                                     \
            op.Process();                                                                                            \
        }                                                                                                            \
    } while (0)

// cFormat is for tempCGlobal Nz out, not cTensor out
#define MMV3_IMPL_C_CLASS(templateClass, aFormat, cFormat, ...)                                                      \
    do {                                                                                                             \
        using cType = MatmulType<AscendC::TPosition::GM, cFormat, DTYPE_Y>;                                          \
        using biasType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_BIAS>;                             \
        TPipe pipe;                                                                                                  \
        if (tilingData.matmulRunInfo.transA == 0 && tilingData.matmulRunInfo.transB == 0) {                          \
            using aType = MatmulType<AscendC::TPosition::GM, aFormat, DTYPE_X1, false>;                              \
            using bType = MatmulType<AscendC::TPosition::GM, format_x2, DTYPE_X2, false>;                            \
            templateClass<aType, bType, cType, biasType, __VA_ARGS__> op;                                            \
            op.Init(aGM, bGM, cGM, biasGM, offsetWGM, user, &tilingData, &pipe);                                     \
            op.Process();                                                                                            \
        } else if (tilingData.matmulRunInfo.transA == 1 && tilingData.matmulRunInfo.transB == 0) {                   \
            using aType = MatmulType<AscendC::TPosition::GM, aFormat, DTYPE_X1, true>;                               \
            using bType = MatmulType<AscendC::TPosition::GM, format_x2, DTYPE_X2, false>;                            \
            templateClass<aType, bType, cType, biasType, __VA_ARGS__> op;                                            \
            op.Init(aGM, bGM, cGM, biasGM, offsetWGM, user, &tilingData, &pipe);                                     \
            op.Process();                                                                                            \
        } else if (tilingData.matmulRunInfo.transA == 0 && tilingData.matmulRunInfo.transB == 1) {                   \
            using aType = MatmulType<AscendC::TPosition::GM, aFormat, DTYPE_X1, false>;                              \
            using bType = MatmulType<AscendC::TPosition::GM, format_x2, DTYPE_X2, true>;                             \
            templateClass<aType, bType, cType, biasType, __VA_ARGS__> op;                                            \
            op.Init(aGM, bGM, cGM, biasGM, offsetWGM, user, &tilingData, &pipe);                                     \
            op.Process();                                                                                            \
        } else {                                                                                                     \
            using aType = MatmulType<AscendC::TPosition::GM, aFormat, DTYPE_X1, true>;                               \
            using bType = MatmulType<AscendC::TPosition::GM, format_x2, DTYPE_X2, true>;                             \
            templateClass<aType, bType, cType, biasType, __VA_ARGS__> op;                                            \
            op.Init(aGM, bGM, cGM, biasGM, offsetWGM, user, &tilingData, &pipe);                                     \
            op.Process();                                                                                            \
        }                                                                                                            \
    } while (0)

#define MMV3_IMPL_CLASS_TRANS(tilingData, tilingGM, transA, transB, templateClass, ...)                              \
    do {                                                                                                             \
        uint64_t tilingdata_offset = (sizeof(Mc2MatMulV3TilingData) > TILINGDATA_OFFSET) ? 0 :                          \
            MMV3DivFloor(GetCurrentBlockIdx(), MMV3DivCeil(GetBlockNum(), TILINGDATA_SPLIT_NUM)) * TILINGDATA_OFFSET;\
        GET_TILING_DATA_WITH_STRUCT(Mc2MatMulV3TilingData, tilingData, tilingGM + tilingdata_offset);                   \
        using cType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_Y>;                                   \
        using biasType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_BIAS>;                             \
        TPipe pipe;                                                                                                  \
        using aType = MatmulType<AscendC::TPosition::GM, format_x1, DTYPE_X1, transA>;                               \
        using bType = MatmulType<AscendC::TPosition::GM, format_x2, DTYPE_X2, transB>;                               \
        templateClass<aType, bType, cType, biasType, __VA_ARGS__> op;                                                \
        op.Init(aGM, bGM, cGM, biasGM, offsetWGM, user, &tilingData, &pipe);                                         \
        op.Process();                                                                                                \
    } while (0)

template <int LOADMODE, int SPLITCOREMODE, int FIXOPTI, int MIXND2NZ>
__global__ __aicore__ void mat_mul_v3(
    GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM, GM_ADDR offsetWGM, GM_ADDR cGM, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    REGISTER_TILING_DEFAULT(Mc2MatmulV3TilingData);
    __gm__ uint8_t *user = GetUserWorkspace(workspaceGM);
    GET_TILING_DATA(tilingData, tilingGM);
#if defined(__CCE_AICORE__) && __CCE_AICORE__ < 220
    // 第一个模板使用mix类型的，使得整个算子的coreType在dyn场景都为mix，静态则根据选择的tilingkey决定coreType
    if (LOADMODE == MAT_MUL_V3_BASE_FULLLOAD && SPLITCOREMODE == MAT_MUL_V3_BASE_SPLIT_K &&
        FIXOPTI == MAT_MUL_V3_BASE_FIXOPTI && MIXND2NZ == MAT_MUL_V3_MIXND2NZ_TRUE) {
        MMV3_IMPL_CLASS(
            Mc2MatmulBaseUnAlignedKernel, format_x1, Mc2MatmulBaseBlock, MM_CFG_VEC_ND2NZ
        );
    } else if (
        LOADMODE == MAT_MUL_V3_BASE_FULLLOAD && SPLITCOREMODE == MAT_MUL_V3_BASE_SPLIT_K &&
        FIXOPTI == MAT_MUL_V3_BASE_FIXOPTI && MIXND2NZ == MAT_MUL_V3_MIXND2NZ_FALSE) {
        MMV3_IMPL_CLASS(
            Mc2MatmulBaseKernel, format_x1, Mc2MatmulBaseBlock, MM_CFG_VEC_ND2NZ
        );
    }
#elif defined(FORMAT_X2) && FORMAT_X2 == FORMAT_FRACTAL_NZ
    if (LOADMODE == MAT_MUL_V3_BASE_FULLLOAD && SPLITCOREMODE == MAT_MUL_V3_BASE_SPLIT_K &&
        FIXOPTI == MAT_MUL_V3_BASE_FIXOPTI && MIXND2NZ == MAT_MUL_V3_MIXND2NZ_TRUE) {
        MMV3_IMPL_CLASS(
            Mc2MatmulBaseUnAlignedKernel, format_x1, Mc2MatmulBaseBlock, MM_CFG_NO_PRELOAD
        );
    } else if (
        LOADMODE == MAT_MUL_V3_BASE_FULLLOAD && SPLITCOREMODE == MAT_MUL_V3_BASE_SPLIT_K &&
        FIXOPTI == MAT_MUL_V3_BASE_FIXOPTI && MIXND2NZ == MAT_MUL_V3_MIXND2NZ_FALSE) {
        MMV3_IMPL_CLASS(
            Mc2MatmulBaseKernel, format_x1, Mc2MatmulBaseBlock, MM_CFG_NO_PRELOAD
        );
    }
#else
    if (LOADMODE == MAT_MUL_V3_BASE_FULLLOAD && SPLITCOREMODE == MAT_MUL_V3_BASE_SPLIT_K &&
        FIXOPTI == MAT_MUL_V3_BASE_FIXOPTI && MIXND2NZ == MAT_MUL_V3_MIXND2NZ_TRUE) {
        MMV3_IMPL_CLASS(
            Mc2MatmulBaseUnAlignedKernel, format_x1, Mc2MatmulBaseBlock, MM_CFG_NO_PRELOAD
        );
    } else if (
        LOADMODE == MAT_MUL_V3_BASE_FULLLOAD && SPLITCOREMODE == MAT_MUL_V3_SINGLE_CORE_SPLIT_K &&
        FIXOPTI == MAT_MUL_V3_BASE_FIXOPTI && MIXND2NZ == MAT_MUL_V3_MIXND2NZ_FALSE) {
        MMV3_IMPL_CLASS(
            Mc2MatMulSingleCoreSplitKKernel, format_x1, Mc2MatmulSingleCoreSplitKBaseBlock, MM_CFG_PRELOAD_MK
        );
    } else if (
        LOADMODE == MAT_MUL_V3_BASE_FULLLOAD && SPLITCOREMODE == MAT_MUL_V3_SINGLE_CORE_NKM_SPLIT_K &&
        FIXOPTI == MAT_MUL_V3_BASE_FIXOPTI && MIXND2NZ == MAT_MUL_V3_MIXND2NZ_FALSE) {
        MMV3_IMPL_CLASS(
            Mc2MatMulSingleCoreSplitKKernel, format_x1, Mc2MatmulSingleCoreSplitKBaseBlock, MM_CFG_PRELOAD_NK, true
        );
    } else if (
        LOADMODE == MAT_MUL_V3_BASE_FULLLOAD && SPLITCOREMODE == MAT_MUL_V3_SINGLE_CORE_SPLIT_K_GM_TO_L1 &&
        FIXOPTI == MAT_MUL_V3_BASE_FIXOPTI && MIXND2NZ == MAT_MUL_V3_MIXND2NZ_FALSE) {
        MMV3_IMPL_CLASS(
            MatMulSingleCoreSplitKKernelGmToL1, format_x1, Mc2MatmulSingleCoreSplitKBaseBlock, MM_CFG_PRELOAD_MK
        );
    } else if (
        LOADMODE == MAT_MUL_V3_BASE_FULLLOAD && SPLITCOREMODE == MAT_MUL_V3_SINGLE_CORE_SPLIT_K &&
        FIXOPTI == MAT_MUL_V3_BASE_FIXOPTI && MIXND2NZ == MAT_MUL_V3_MIXND2NZ_TRUE) {
        MMV3_IMPL_CLASS(
            Mc2MatMulUnAlignedSingleCoreSplitKKernel, format_x1, Mc2MatmulSingleCoreSplitKBaseBlock, MM_CFG_PRELOAD_MK
        );
    } else if (
        LOADMODE == MAT_MUL_V3_BASE_FULLLOAD && SPLITCOREMODE == MAT_MUL_V3_SINGLE_CORE_SPLIT_K_GM_TO_L1 &&
        FIXOPTI == MAT_MUL_V3_BASE_FIXOPTI && MIXND2NZ == MAT_MUL_V3_MIXND2NZ_TRUE) {
        MMV3_IMPL_CLASS(
            Mc2MatMulUnAlignedSingleCoreSplitKKernelGmToL1, format_x1, Mc2MatmulSingleCoreSplitKBaseBlock, MM_CFG_PRELOAD_MK
        );
    } else if (
        LOADMODE == MAT_MUL_V3_BASE_FULLLOAD && SPLITCOREMODE == MAT_MUL_V3_DETERMINISTIC_SPLIT_K &&
        FIXOPTI == MAT_MUL_V3_BASE_FIXOPTI && MIXND2NZ == MAT_MUL_V3_MIXND2NZ_FALSE) {
        MMV3_IMPL(
            Mc2MatMulKernelDeterministicSplitK, format_x1, FIXPIPE_OPT_SELECT::BASE
        );
    } else if (
        LOADMODE == MAT_MUL_V3_BASE_FULLLOAD && SPLITCOREMODE == MAT_MUL_V3_DETERMINISTIC_SPLIT_K &&
        FIXOPTI == MAT_MUL_V3_BASE_FIXOPTI && MIXND2NZ == MAT_MUL_V3_MIXND2NZ_TRUE) {
        MMV3_IMPL(
            Mc2MatMulUnAlignedKernelDeterministicSplitK, format_x1, FIXPIPE_OPT_SELECT::BASE
        );
    } else if (
        LOADMODE == MAT_MUL_V3_BASE_FULLLOAD && SPLITCOREMODE == MAT_MUL_V3_MULTI_CORE_SPLIT_K &&
        FIXOPTI == MAT_MUL_V3_BASE_FIXOPTI && MIXND2NZ == MAT_MUL_V3_MIXND2NZ_FALSE) {
        MMV3_IMPL(
            Mc2MatMulMultiCoreSplitK, format_x1, FIXPIPE_OPT_SELECT::BASE
        );
    } else if (
        LOADMODE == MAT_MUL_V3_BASE_FULLLOAD && SPLITCOREMODE == MAT_MUL_V3_BASE_SPLIT_K &&
        FIXOPTI == MAT_MUL_V3_BASE_FIXOPTI && MIXND2NZ == MAT_MUL_V3_MIXND2NZ_FALSE) {
        MMV3_IMPL_CLASS(
            Mc2MatmulBaseKernel, format_x1, Mc2MatmulBaseBlock, MM_CFG_NO_PRELOAD
        );
    } else if (
        LOADMODE == MAT_MUL_V3_AL1_FULLLOAD && SPLITCOREMODE == MAT_MUL_V3_BASE_SPLIT_K &&
        FIXOPTI == MAT_MUL_V3_BASE_FIXOPTI && MIXND2NZ == MAT_MUL_V3_MIXND2NZ_FALSE) {
        MMV3_IMPL_CLASS(
            Mc2MatmulBaseKernelAL1FullLoad, format_x1, Mc2MatmulBaseBlock, MM_CFG_MDL
        );
    } else if (
        LOADMODE == MAT_MUL_V3_BL1_FULLLOAD && SPLITCOREMODE == MAT_MUL_V3_BASE_SPLIT_K &&
        FIXOPTI == MAT_MUL_V3_BASE_FIXOPTI && MIXND2NZ == MAT_MUL_V3_MIXND2NZ_FALSE) {
        MMV3_IMPL_CLASS(
            MatmulBaseKernelBL1FullLoad, format_x1, Mc2MatmulBaseBlock, MM_CFG_NO_PRELOAD
        );
    } else if (
        LOADMODE == MAT_MUL_V3_BL1_FULLLOAD && SPLITCOREMODE == MAT_MUL_V3_BASE_SPLIT_K &&
        FIXOPTI == MAT_MUL_V3_BASE_ENABLE_ALIGNOUT && MIXND2NZ == MAT_MUL_V3_MIXND2NZ_FALSE) {
        MMV3_IMPL_CLASS(
            Mc2MatmulBaseUnalignedNKernel, format_x1, Mc2MatmulBaseBlock, MM_CFG_NO_PRELOAD,
            MatmulCallBackFunc<nullptr, nullptr, CopyBL1>
        );
    } else if (
        LOADMODE == MAT_MUL_V3_BL1_FULLLOAD && SPLITCOREMODE == MAT_MUL_V3_BASE_SPLIT_K &&
        FIXOPTI == MAT_MUL_V3_BASE_FIXOPTI && MIXND2NZ == MAT_MUL_V3_MIXND2NZ_TRUE) {
        MMV3_IMPL_CLASS(
            Mc2MatmulBaseUnAlignedKernelBL1FullLoad, format_x1, Mc2MatmulBaseBlock, MM_CFG_NO_PRELOAD,
            MatmulCallBackFunc<nullptr, nullptr, CopyBL1>
        );
    } else if (
        LOADMODE == MAT_MUL_V3_BL1_FULLLOAD && SPLITCOREMODE == MAT_MUL_V3_BASE_SPLIT_K &&
        FIXOPTI == MAT_MUL_V3_BASE_FIXOPTI && MIXND2NZ == MAT_MUL_V3_MIXND2NZ_TRUE_PARALLEL) {
        MMV3_IMPL_CLASS(
            Mc2MatmulBaseKernel, format_x1, Mc2MatmulBaseBlock, MM_CFG_NO_PRELOAD
        );
    } else if (
        LOADMODE == MAT_MUL_V3_BL1_FULLLOAD && SPLITCOREMODE == MAT_MUL_V3_BASE_SPLIT_K &&
        FIXOPTI == MAT_MUL_V3_BASE_ENABLE_ALIGNOUT && MIXND2NZ == MAT_MUL_V3_MIXND2NZ_TRUE) {
        MMV3_IMPL_CLASS(
            Mc2MatmulBaseAToNZWithBL1FixpipeKernel, CubeFormat::NZ, Mc2MatmulBaseBlock, MM_CFG_NO_PRELOAD,
            MatmulCallBackFunc<nullptr, nullptr, CopyBL1>
        );
    } else if (
        LOADMODE == MAT_MUL_V3_BL1_FULLLOAD && SPLITCOREMODE == MAT_MUL_V3_BASE_SPLIT_K &&
        FIXOPTI == MAT_MUL_V3_VEC_NZ2ND_UNALIGNOUT && MIXND2NZ == MAT_MUL_V3_MIXND2NZ_FALSE) {
        MMV3_IMPL_C_CLASS(
            Mc2MatmulBaseUnalignedNKernel, format_x1, CubeFormat::NZ, Mc2MatmulBaseBlock, MM_CFG_NO_PRELOAD,
            MatmulCallBackFunc<nullptr, nullptr, CopyBL1>, FIXPIPE_OPT_SELECT::VEC_NZ2ND_UNALIGNOUT
        );
    } else if (
        LOADMODE == MAT_MUL_V3_BASE_FULLLOAD && SPLITCOREMODE == MAT_MUL_V3_DETERMINISTIC_SPLIT_K &&
        FIXOPTI == MAT_MUL_V3_VEC_NZ2ND_UNALIGNOUT && MIXND2NZ == MAT_MUL_V3_MIXND2NZ_FALSE) {
        MMV3_IMPL(
            Mc2MatMulKernelDeterministicSplitK, CubeFormat::NZ, FIXPIPE_OPT_SELECT::VEC_NZ2ND_UNALIGNOUT
        );
    } else if (
        LOADMODE == MAT_MUL_V3_BASE_FULLLOAD && SPLITCOREMODE == MAT_MUL_V3_DETERMINISTIC_SPLIT_K &&
        FIXOPTI == MAT_MUL_V3_VEC_NZ2ND_UNALIGNOUT && MIXND2NZ == MAT_MUL_V3_MIXND2NZ_TRUE) {
        MMV3_IMPL(
            Mc2MatMulUnAlignedKernelDeterministicSplitK, CubeFormat::NZ, FIXPIPE_OPT_SELECT::VEC_NZ2ND_UNALIGNOUT
        );
    }
#endif
}
