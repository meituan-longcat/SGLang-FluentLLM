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
#include "../../mat_mul_v3/op_kernel/arch35/mat_mul_tiling_data.h"
#include "arch35/batch_mat_mul_v3_asw_kernel_advanced.h"
#include "arch35/batch_mat_mul_v3_iterbatch_kernel_advanced.h"

#if !(defined(__NPU_ARCH__) && (__NPU_ARCH__ == 5102))
#include "arch35/batch_mat_mul_v3_asw_al1_full_load_kernel_advanced.h"
#include "arch35/batch_mat_mul_v3_asw_bl1_full_load_kernel_advanced.h"
#include "arch35/batch_mat_mul_v3_iterbatch_basicapi_act.h"
#include "../../mat_mul_v3/op_kernel/arch35/mat_mul_pingpong_basic_act.h"
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

extern "C" __global__ __aicore__ void batch_mat_mul_v3(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM, GM_ADDR offsetWGM,
    GM_ADDR cGM, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    __gm__ uint8_t *user = GetUserWorkspace(workspaceGM);

    REGISTER_TILING_DEFAULT(BatchMatMulV3TilingData);
    REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR >= 10001900009002090000UL", Mc2BatchMatMulV3BasicTilingData);
    REGISTER_TILING_FOR_TILINGKEY(
        "TILING_KEY_VAR >= 10001900009000190000UL && TILING_KEY_VAR < 10001900009002090000UL",
         BatchMatMulV3IterBatchBasicTilingData);
    REGISTER_TILING_FOR_TILINGKEY(
        "TILING_KEY_VAR >= 10001900009000090000UL && TILING_KEY_VAR < 10001900009000190000UL",
         Mc2BatchMatMulV3BasicTilingData);
    REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR < 10001900009000090000UL", BatchMatMulV3TilingData);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);

    if (TILING_KEY_IS(10000900009000090000UL)) {
        BMMV3_IMPL_CLASS_COMMON_TRNAS(false, false, BatchMatMulV3Advanced::Mc2BatchMatMulAswKernel,
            BatchMatMulV3Advanced::Mc2BatchMatMulAswBlock, MM_CFG_NO_PRELOAD);
    } else if (TILING_KEY_IS(10000900009000090001UL)) {
        BMMV3_IMPL_CLASS_COMMON_TRNAS(true, false, BatchMatMulV3Advanced::Mc2BatchMatMulAswKernel,
            BatchMatMulV3Advanced::Mc2BatchMatMulAswBlock, MM_CFG_NO_PRELOAD);
    } else if (TILING_KEY_IS(10000900009000090002UL)) {
        BMMV3_IMPL_CLASS_COMMON_TRNAS(false, true, BatchMatMulV3Advanced::Mc2BatchMatMulAswKernel,
            BatchMatMulV3Advanced::Mc2BatchMatMulAswBlock, MM_CFG_NO_PRELOAD);
    } else if (TILING_KEY_IS(10000900009000090003UL)) {
        BMMV3_IMPL_CLASS_COMMON_TRNAS(true, true, BatchMatMulV3Advanced::Mc2BatchMatMulAswKernel,
            BatchMatMulV3Advanced::Mc2BatchMatMulAswBlock, MM_CFG_NO_PRELOAD);
    } else if (TILING_KEY_IS(10000900009000490000UL)) {
        BMMV3_IMPL_CLASS_COMMON_TRNAS(false, false, BatchMatMulV3Advanced::Mc2BatchMatMulAswKernel,
            BatchMatMulV3Advanced::Mc2BatchMatMulDaswBlock, MM_CFG_NO_PRELOAD);
    } else if (TILING_KEY_IS(10000900009000490001UL)) {
        BMMV3_IMPL_CLASS_COMMON_TRNAS(true, false, BatchMatMulV3Advanced::Mc2BatchMatMulAswKernel,
            BatchMatMulV3Advanced::Mc2BatchMatMulDaswBlock, MM_CFG_NO_PRELOAD);
    } else if (TILING_KEY_IS(10000900009000490002UL)) {
        BMMV3_IMPL_CLASS_COMMON_TRNAS(false, true, BatchMatMulV3Advanced::Mc2BatchMatMulAswKernel,
            BatchMatMulV3Advanced::Mc2BatchMatMulDaswBlock, MM_CFG_NO_PRELOAD);
    } else if (TILING_KEY_IS(10000900009000490003UL)) {
        BMMV3_IMPL_CLASS_COMMON_TRNAS(true, true, BatchMatMulV3Advanced::Mc2BatchMatMulAswKernel,
            BatchMatMulV3Advanced::Mc2BatchMatMulDaswBlock, MM_CFG_NO_PRELOAD);
    } else if (TILING_KEY_IS(10000900009000190000UL)) {
        BMMV3_IMPL_CLASS_TRANS(false, false, BatchMatMulV3Advanced::Mc2BatchMatMulMultiBatchKernel,
            BatchMatMulV3Advanced::Mc2BatchMatMulMultiBatchBaseBlock, MM_CFG_MULTI_BATCH_OUT_BATCH_BIAS);
    } else if (TILING_KEY_IS(10000900009000190001UL)) {
        BMMV3_IMPL_CLASS_TRANS(true, false, BatchMatMulV3Advanced::Mc2BatchMatMulMultiBatchKernel,
            BatchMatMulV3Advanced::Mc2BatchMatMulMultiBatchBaseBlock, MM_CFG_MULTI_BATCH_OUT_BATCH_BIAS);
    } else if (TILING_KEY_IS(10000900009000190002UL)) {
        BMMV3_IMPL_CLASS_TRANS(false, true, BatchMatMulV3Advanced::Mc2BatchMatMulMultiBatchKernel,
            BatchMatMulV3Advanced::Mc2BatchMatMulMultiBatchBaseBlock, MM_CFG_MULTI_BATCH_OUT_BATCH_BIAS);
    } else if (TILING_KEY_IS(10000900009000190003UL)) {
        BMMV3_IMPL_CLASS_TRANS(true, true, BatchMatMulV3Advanced::Mc2BatchMatMulMultiBatchKernel,
            BatchMatMulV3Advanced::Mc2BatchMatMulMultiBatchBaseBlock, MM_CFG_MULTI_BATCH_OUT_BATCH_BIAS);
    } else if (TILING_KEY_IS(10000900009000390000UL)) {
        BMMV3_IMPL_CLASS_TRANS(false, false, BatchMatMulV3Advanced::Mc2BatchMatMulMultiBatchKernel,
            BatchMatMulV3Advanced::Mc2BatchMatMulMultiBatchBaseBlock, MM_CFG_MULTI_BATCH_OUT_SING_BIAS);
    } else if (TILING_KEY_IS(10000900009000390001UL)) {
        BMMV3_IMPL_CLASS_TRANS(true, false, BatchMatMulV3Advanced::Mc2BatchMatMulMultiBatchKernel,
            BatchMatMulV3Advanced::Mc2BatchMatMulMultiBatchBaseBlock, MM_CFG_MULTI_BATCH_OUT_SING_BIAS);
    } else if (TILING_KEY_IS(10000900009000390002UL)) {
        BMMV3_IMPL_CLASS_TRANS(false, true, BatchMatMulV3Advanced::Mc2BatchMatMulMultiBatchKernel,
            BatchMatMulV3Advanced::Mc2BatchMatMulMultiBatchBaseBlock, MM_CFG_MULTI_BATCH_OUT_SING_BIAS);
    } else if (TILING_KEY_IS(10000900009000390003UL)) {
        BMMV3_IMPL_CLASS_TRANS(true, true, BatchMatMulV3Advanced::Mc2BatchMatMulMultiBatchKernel,
            BatchMatMulV3Advanced::Mc2BatchMatMulMultiBatchBaseBlock, MM_CFG_MULTI_BATCH_OUT_SING_BIAS);
#if !(defined(__NPU_ARCH__) && (__NPU_ARCH__ == 5102))
    } else if (TILING_KEY_IS(10001900009000190000UL)) {
        GET_TILING_DATA_WITH_STRUCT(BatchMatMulV3IterBatchBasicTilingData, tilingData, tilingGM);
        Mc2BatchMatMulActIterBatchKernel<DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS, layout::RowMajor,
            layout::RowMajor, layout::RowMajor>(aGM, bGM, biasGM, cGM, workspaceGM, tilingData);
    } else if (TILING_KEY_IS(10001900009000190001UL)) {
        GET_TILING_DATA_WITH_STRUCT(BatchMatMulV3IterBatchBasicTilingData, tilingData, tilingGM);
        Mc2BatchMatMulActIterBatchKernel<DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS, layout::ColumnMajor,
            layout::RowMajor, layout::RowMajor>(aGM, bGM, biasGM, cGM, workspaceGM, tilingData);
    } else if (TILING_KEY_IS(10001900009000190002UL)) {
        GET_TILING_DATA_WITH_STRUCT(BatchMatMulV3IterBatchBasicTilingData, tilingData, tilingGM);
        Mc2BatchMatMulActIterBatchKernel<DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS, layout::RowMajor,
            layout::ColumnMajor, layout::RowMajor>(aGM, bGM, biasGM, cGM, workspaceGM, tilingData);
    } else if (TILING_KEY_IS(10001900009000190003UL)) {
        GET_TILING_DATA_WITH_STRUCT(BatchMatMulV3IterBatchBasicTilingData, tilingData, tilingGM);
        Mc2BatchMatMulActIterBatchKernel<DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS, layout::ColumnMajor,
            layout::ColumnMajor, layout::RowMajor>(aGM, bGM, biasGM, cGM, workspaceGM, tilingData);
    } else if (TILING_KEY_IS(10001900009000090000UL)) {
        GET_TILING_DATA_WITH_STRUCT(Mc2BatchMatMulV3BasicTilingData, tilingData, tilingGM);
        Mc2MatmulV3Advanced::MatMulActKernel<
            DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS, layout::RowMajor, layout::RowMajor, layout::RowMajor>(
            aGM, bGM, biasGM, cGM, workspaceGM, tilingData.matMulTilingData, tilingData.batchDimAll);
    } else if (TILING_KEY_IS(10001900009000090001UL)) {
        GET_TILING_DATA_WITH_STRUCT(Mc2BatchMatMulV3BasicTilingData, tilingData, tilingGM);
        Mc2MatmulV3Advanced::MatMulActKernel<
            DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS, layout::ColumnMajor, layout::RowMajor, layout::RowMajor>(
            aGM, bGM, biasGM, cGM, workspaceGM, tilingData.matMulTilingData, tilingData.batchDimAll);
    } else if (TILING_KEY_IS(10001900009000090002UL)) {
        GET_TILING_DATA_WITH_STRUCT(Mc2BatchMatMulV3BasicTilingData, tilingData, tilingGM);
        Mc2MatmulV3Advanced::MatMulActKernel<
            DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS, layout::RowMajor, layout::ColumnMajor, layout::RowMajor>(
            aGM, bGM, biasGM, cGM, workspaceGM, tilingData.matMulTilingData, tilingData.batchDimAll);
    } else if (TILING_KEY_IS(10001900009000090003UL)) {
        GET_TILING_DATA_WITH_STRUCT(Mc2BatchMatMulV3BasicTilingData, tilingData, tilingGM);
        Mc2MatmulV3Advanced::MatMulActKernel<
            DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS, layout::ColumnMajor, layout::ColumnMajor, layout::RowMajor>(
            aGM, bGM, biasGM, cGM, workspaceGM, tilingData.matMulTilingData, tilingData.batchDimAll);
    } else if (TILING_KEY_IS(10001900009002090000UL)) {
        GET_TILING_DATA_WITH_STRUCT(Mc2BatchMatMulV3BasicTilingData, tilingData, tilingGM);
        Mc2MatmulV3Advanced::MatMulActKernel<
            DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS, layout::RowMajor, layout::RowMajor, layout::RowMajor,
            B_FULL_LOAD_MODE>(aGM, bGM, biasGM, cGM, workspaceGM, tilingData.matMulTilingData, tilingData.batchDimAll);
    } else if (TILING_KEY_IS(10001900009002090001UL)) {
        GET_TILING_DATA_WITH_STRUCT(Mc2BatchMatMulV3BasicTilingData, tilingData, tilingGM);
        Mc2MatmulV3Advanced::MatMulActKernel<
            DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS, layout::ColumnMajor, layout::RowMajor, layout::RowMajor,
            B_FULL_LOAD_MODE>(aGM, bGM, biasGM, cGM, workspaceGM, tilingData.matMulTilingData, tilingData.batchDimAll);
    } else if (TILING_KEY_IS(10001900009002090002UL)) {
        GET_TILING_DATA_WITH_STRUCT(Mc2BatchMatMulV3BasicTilingData, tilingData, tilingGM);
        Mc2MatmulV3Advanced::MatMulActKernel<
            DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS, layout::RowMajor, layout::ColumnMajor, layout::RowMajor,
            B_FULL_LOAD_MODE>(aGM, bGM, biasGM, cGM, workspaceGM, tilingData.matMulTilingData, tilingData.batchDimAll);
    } else if (TILING_KEY_IS(10001900009002090003UL)) {
        GET_TILING_DATA_WITH_STRUCT(Mc2BatchMatMulV3BasicTilingData, tilingData, tilingGM);
        Mc2MatmulV3Advanced::MatMulActKernel<
            DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS, layout::ColumnMajor, layout::ColumnMajor, layout::RowMajor,
            B_FULL_LOAD_MODE>(aGM, bGM, biasGM, cGM, workspaceGM, tilingData.matMulTilingData, tilingData.batchDimAll);
    } else if (TILING_KEY_IS(10000900009001090000UL)) {
        BMMV3_IMPL_CLASS_COMMON_TRNAS(false, false, Mc2BatchMatMulV3Advanced::Mc2BatchMatMulAswAL1FullLoadKernel,
            Mc2BatchMatMulV3Advanced::Mc2BatchMatMulAswBlock, MM_CFG_NO_PRELOAD);
    } else if (TILING_KEY_IS(10000900009001090001UL)) {
        BMMV3_IMPL_CLASS_COMMON_TRNAS(true, false, Mc2BatchMatMulV3Advanced::Mc2BatchMatMulAswAL1FullLoadKernel,
            Mc2BatchMatMulV3Advanced::Mc2BatchMatMulAswBlock, MM_CFG_NO_PRELOAD);
    } else if (TILING_KEY_IS(10000900009001090002UL)) {
        BMMV3_IMPL_CLASS_COMMON_TRNAS(false, true, Mc2BatchMatMulV3Advanced::Mc2BatchMatMulAswAL1FullLoadKernel,
            Mc2BatchMatMulV3Advanced::Mc2BatchMatMulAswBlock, MM_CFG_NO_PRELOAD);
    } else if (TILING_KEY_IS(10000900009001090003UL)) {
        BMMV3_IMPL_CLASS_COMMON_TRNAS(true, true, Mc2BatchMatMulV3Advanced::Mc2BatchMatMulAswAL1FullLoadKernel,
            Mc2BatchMatMulV3Advanced::Mc2BatchMatMulAswBlock, MM_CFG_NO_PRELOAD);
    } else if (TILING_KEY_IS(10000900009001490000UL)) {
        BMMV3_IMPL_CLASS_COMMON_TRNAS(false, false, Mc2BatchMatMulV3Advanced::Mc2BatchMatMulAswAL1FullLoadKernel,
            Mc2BatchMatMulV3Advanced::Mc2BatchMatMulDaswBlock, MM_CFG_NO_PRELOAD);
    } else if (TILING_KEY_IS(10000900009001490001UL)) {
        BMMV3_IMPL_CLASS_COMMON_TRNAS(true, false, Mc2BatchMatMulV3Advanced::Mc2BatchMatMulAswAL1FullLoadKernel,
            Mc2BatchMatMulV3Advanced::Mc2BatchMatMulDaswBlock, MM_CFG_NO_PRELOAD);
    } else if (TILING_KEY_IS(10000900009001490002UL)) {
        BMMV3_IMPL_CLASS_COMMON_TRNAS(false, true, Mc2BatchMatMulV3Advanced::Mc2BatchMatMulAswAL1FullLoadKernel,
            Mc2BatchMatMulV3Advanced::Mc2BatchMatMulDaswBlock, MM_CFG_NO_PRELOAD);
    } else if (TILING_KEY_IS(10000900009001490003UL)) {
        BMMV3_IMPL_CLASS_COMMON_TRNAS(true, true, Mc2BatchMatMulV3Advanced::Mc2BatchMatMulAswAL1FullLoadKernel,
            Mc2BatchMatMulV3Advanced::Mc2BatchMatMulDaswBlock, MM_CFG_NO_PRELOAD);
    } else if (TILING_KEY_IS(10000900009002090000UL)) {
        BMMV3_IMPL_CLASS_COMMON_TRNAS(false, false, Mc2BatchMatMulV3Advanced::Mc2BatchMatMulAswBL1FullLoadKernel,
            Mc2BatchMatMulV3Advanced::Mc2BatchMatMulAswBlock, MM_CFG_NO_PRELOAD);
    } else if (TILING_KEY_IS(10000900009002090001UL)) {
        BMMV3_IMPL_CLASS_COMMON_TRNAS(true, false, Mc2BatchMatMulV3Advanced::Mc2BatchMatMulAswBL1FullLoadKernel,
            Mc2BatchMatMulV3Advanced::Mc2BatchMatMulAswBlock, MM_CFG_NO_PRELOAD);
    } else if (TILING_KEY_IS(10000900009002090002UL)) {
        BMMV3_IMPL_CLASS_COMMON_TRNAS(false, true, Mc2BatchMatMulV3Advanced::Mc2BatchMatMulAswBL1FullLoadKernel,
            Mc2BatchMatMulV3Advanced::Mc2BatchMatMulAswBlock, MM_CFG_NO_PRELOAD);
    } else if (TILING_KEY_IS(10000900009002090003UL)) {
        BMMV3_IMPL_CLASS_COMMON_TRNAS(true, true, Mc2BatchMatMulV3Advanced::Mc2BatchMatMulAswBL1FullLoadKernel,
            Mc2BatchMatMulV3Advanced::Mc2BatchMatMulAswBlock, MM_CFG_NO_PRELOAD);
    } else if (TILING_KEY_IS(10000900009002490000UL)) {
        BMMV3_IMPL_CLASS_COMMON_TRNAS(false, false, Mc2BatchMatMulV3Advanced::Mc2BatchMatMulAswBL1FullLoadKernel,
            Mc2BatchMatMulV3Advanced::Mc2BatchMatMulDaswBlock, MM_CFG_NO_PRELOAD);
    } else if (TILING_KEY_IS(10000900009002490001UL)) {
        BMMV3_IMPL_CLASS_COMMON_TRNAS(true, false, Mc2BatchMatMulV3Advanced::Mc2BatchMatMulAswBL1FullLoadKernel,
            Mc2BatchMatMulV3Advanced::Mc2BatchMatMulDaswBlock, MM_CFG_NO_PRELOAD);
    } else if (TILING_KEY_IS(10000900009002490002UL)) {
        BMMV3_IMPL_CLASS_COMMON_TRNAS(false, true, Mc2BatchMatMulV3Advanced::Mc2BatchMatMulAswBL1FullLoadKernel,
            Mc2BatchMatMulV3Advanced::Mc2BatchMatMulDaswBlock, MM_CFG_NO_PRELOAD);
    } else if (TILING_KEY_IS(10000900009002490003UL)) {
        BMMV3_IMPL_CLASS_COMMON_TRNAS(true, true, Mc2BatchMatMulV3Advanced::Mc2BatchMatMulAswBL1FullLoadKernel,
            Mc2BatchMatMulV3Advanced::Mc2BatchMatMulDaswBlock, MM_CFG_NO_PRELOAD);
#endif
    }
}