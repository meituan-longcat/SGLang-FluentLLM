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
 * \file mat_mul_v3_apt.cpp
 * \brief
 */

#include "mat_mul_v3_common.h"
#include "arch35/mat_mul_asw_kernel.h"
#if !(defined(__NPU_ARCH__) && (__NPU_ARCH__ == 5102))
#include "arch35/mat_mul_tiling_data.h"
#include "arch35/mat_mul_stream_k_kernel.h"
#include "arch35/mat_mul_full_load.h"
#include "arch35/mat_mul_fixpipe_opti.h"

#include "include/matmul/block/block_scheduler_policy.h"
#include "include/matmul/block/block_scheduler_utils.h"
#include "arch35/block_scheduler_aswt.h"
#include "arch35/block_scheduler_streamk.h"
#include "include/epilogue/block_epilogue_empty.h"
#include "include/matmul/block/block_mmad_builder.h"
#include "include/matmul/kernel/kernel_matmul_without_que.h"
#include "include/matmul/kernel/kernel_matmul_streamk.h"
#include "arch35/mat_mul_pingpong_basic_act.h"
#include "arch35/mat_mul_streamk_basic_act.h"
#include "arch35/mat_mul_fixpipe_opti_basic_act.h"

using namespace Act;
using namespace Act::Gemm;
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

#define MMV3_IMPL_CLASS_TRANS(tilingData, tilingGM, transA, transB, workspace, templateClass, ...)                              \
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
        op.Init(aGM, bGM, cGM, biasGM, offsetWGM, workspace, &tilingData, &pipe);                                         \
        op.Process();                                                                                                \
    } while (0)

extern "C" __global__ __aicore__ void mat_mul_v3(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM,
    GM_ADDR offsetWGM, GM_ADDR cGM, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    // Adaptive Sliding Window Kernel
    REGISTER_TILING_DEFAULT(MatMulV3TilingDataCopy);
    REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR >= 10001900009000090000UL", Mc2MatMulV3BasicTilingData);
    REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR < 10001900009000090000UL", MatMulV3TilingDataCopy);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);

    if (TILING_KEY_IS(10000900009000090000UL)) {
        MMV3_IMPL_CLASS_TRANS(tilingData, tilingGM, false, false, nullptr, Mc2MatmulV3Advanced::Mc2MatmulAswKernel, Mc2MatmulV3Advanced::Mc2MatmulAswBlock,
                              MM_CFG_NO_PRELOAD);
    } else if (TILING_KEY_IS(10000900009000090001UL)) {
        MMV3_IMPL_CLASS_TRANS(tilingData, tilingGM, true, false, nullptr, Mc2MatmulV3Advanced::Mc2MatmulAswKernel, Mc2MatmulV3Advanced::Mc2MatmulAswBlock,
                              MM_CFG_NO_PRELOAD);
    } else if (TILING_KEY_IS(10000900009000090002UL)) {
        MMV3_IMPL_CLASS_TRANS(tilingData, tilingGM, false, true, nullptr, Mc2MatmulV3Advanced::Mc2MatmulAswKernel, Mc2MatmulV3Advanced::Mc2MatmulAswBlock,
                              MM_CFG_NO_PRELOAD);
    } else if (TILING_KEY_IS(10000900009000090003UL)) {
        MMV3_IMPL_CLASS_TRANS(tilingData, tilingGM, true, true, nullptr, Mc2MatmulV3Advanced::Mc2MatmulAswKernel, Mc2MatmulV3Advanced::Mc2MatmulAswBlock,
                              MM_CFG_NO_PRELOAD);
    } else if (TILING_KEY_IS(10000900009000490000UL)) {
        MMV3_IMPL_CLASS_TRANS(tilingData, tilingGM, false, false, nullptr, Mc2MatmulV3Advanced::Mc2MatmulAswKernel, Mc2MatmulV3Advanced::MatmulDaswBlock,
                              MM_CFG_NO_PRELOAD);
    } else if (TILING_KEY_IS(10000900009000490001UL)) {
        MMV3_IMPL_CLASS_TRANS(tilingData, tilingGM, true, false, nullptr, Mc2MatmulV3Advanced::Mc2MatmulAswKernel, Mc2MatmulV3Advanced::MatmulDaswBlock,
                              MM_CFG_NO_PRELOAD);
    } else if (TILING_KEY_IS(10000900009000490002UL)) {
        MMV3_IMPL_CLASS_TRANS(tilingData, tilingGM, false, true, nullptr, Mc2MatmulV3Advanced::Mc2MatmulAswKernel, Mc2MatmulV3Advanced::MatmulDaswBlock,
                              MM_CFG_NO_PRELOAD);
    } else if (TILING_KEY_IS(10000900009000490003UL)) {
        MMV3_IMPL_CLASS_TRANS(tilingData, tilingGM, true, true, nullptr, Mc2MatmulV3Advanced::Mc2MatmulAswKernel, Mc2MatmulV3Advanced::MatmulDaswBlock,
                              MM_CFG_NO_PRELOAD);
#if !(defined(__NPU_ARCH__) && (__NPU_ARCH__ == 5102))
    } else if (TILING_KEY_IS(10001900009000090000UL)) {
        GET_TILING_DATA_WITH_STRUCT(Mc2MatMulV3BasicTilingData, tilingData, tilingGM);
        Mc2MatmulV3Advanced::MatMulActKernel<
            DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS, layout::RowMajor, layout::RowMajor, layout::RowMajor>(
            aGM, bGM, biasGM, cGM, nullptr, tilingData);
    } else if (TILING_KEY_IS(10001900009000090001UL)) {
        GET_TILING_DATA_WITH_STRUCT(Mc2MatMulV3BasicTilingData, tilingData, tilingGM);
        Mc2MatmulV3Advanced::MatMulActKernel<
            DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS, layout::ColumnMajor, layout::RowMajor, layout::RowMajor>(
            aGM, bGM, biasGM, cGM, nullptr, tilingData);
    } else if (TILING_KEY_IS(10001900009000090002UL)) {
        GET_TILING_DATA_WITH_STRUCT(Mc2MatMulV3BasicTilingData, tilingData, tilingGM);
        Mc2MatmulV3Advanced::MatMulActKernel<
            DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS, layout::RowMajor, layout::ColumnMajor, layout::RowMajor>(
            aGM, bGM, biasGM, cGM, nullptr, tilingData);
    } else if (TILING_KEY_IS(10001900009000090003UL)) {
        GET_TILING_DATA_WITH_STRUCT(Mc2MatMulV3BasicTilingData, tilingData, tilingGM);
        Mc2MatmulV3Advanced::MatMulActKernel<
            DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS, layout::ColumnMajor, layout::ColumnMajor, layout::RowMajor>(
            aGM, bGM, biasGM, cGM, nullptr, tilingData);
    } else if (TILING_KEY_IS(10001900009002090000UL)) {
        GET_TILING_DATA_WITH_STRUCT(Mc2MatMulV3BasicTilingData, tilingData, tilingGM);
        Mc2MatmulV3Advanced::MatMulActKernel<
            DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS, layout::RowMajor, layout::RowMajor, layout::RowMajor,
            B_FULL_LOAD_MODE>(aGM, bGM, biasGM, cGM, nullptr, tilingData);
    } else if (TILING_KEY_IS(10001900009002090001UL)) {
        GET_TILING_DATA_WITH_STRUCT(Mc2MatMulV3BasicTilingData, tilingData, tilingGM);
        Mc2MatmulV3Advanced::MatMulActKernel<
            DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS, layout::ColumnMajor, layout::RowMajor, layout::RowMajor,
            B_FULL_LOAD_MODE>(aGM, bGM, biasGM, cGM, nullptr, tilingData);
    } else if (TILING_KEY_IS(10001900009002090002UL)) {
        GET_TILING_DATA_WITH_STRUCT(Mc2MatMulV3BasicTilingData, tilingData, tilingGM);
        Mc2MatmulV3Advanced::MatMulActKernel<
            DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS, layout::RowMajor, layout::ColumnMajor, layout::RowMajor,
            B_FULL_LOAD_MODE>(aGM, bGM, biasGM, cGM, nullptr, tilingData);
    } else if (TILING_KEY_IS(10001900009002090003UL)) {
        GET_TILING_DATA_WITH_STRUCT(Mc2MatMulV3BasicTilingData, tilingData, tilingGM);
        Mc2MatmulV3Advanced::MatMulActKernel<
            DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS, layout::ColumnMajor, layout::ColumnMajor, layout::RowMajor,
            B_FULL_LOAD_MODE>(aGM, bGM, biasGM, cGM, nullptr, tilingData);
    } else if (TILING_KEY_IS(10001900009000290000UL)) {
        KERNEL_TASK_TYPE(10001900009000290000UL, KERNEL_TYPE_MIX_AIC_1_2);
        GET_TILING_DATA_WITH_STRUCT(Mc2MatMulV3BasicTilingData, tilingData, tilingGM);
        MatMulStreamKActKernel<DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS, layout::RowMajor, layout::RowMajor,
                               layout::RowMajor, MatMulL0C2Out::ON_THE_FLY>(
            aGM, bGM, biasGM, cGM, workspaceGM, tilingData);
    } else if (TILING_KEY_IS(10001900009000290001UL)) {
        KERNEL_TASK_TYPE(10001900009000290001UL, KERNEL_TYPE_MIX_AIC_1_2);
        GET_TILING_DATA_WITH_STRUCT(Mc2MatMulV3BasicTilingData, tilingData, tilingGM);
        MatMulStreamKActKernel<DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS, layout::ColumnMajor, layout::RowMajor,
                               layout::RowMajor, MatMulL0C2Out::ON_THE_FLY>(
            aGM, bGM, biasGM, cGM, workspaceGM, tilingData);
    } else if (TILING_KEY_IS(10001900009000290002UL)) {
        KERNEL_TASK_TYPE(10001900009000290002UL, KERNEL_TYPE_MIX_AIC_1_2);
        GET_TILING_DATA_WITH_STRUCT(Mc2MatMulV3BasicTilingData, tilingData, tilingGM);
        MatMulStreamKActKernel<DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS, layout::RowMajor, layout::ColumnMajor,
                               layout::RowMajor, MatMulL0C2Out::ON_THE_FLY>(
            aGM, bGM, biasGM, cGM, workspaceGM, tilingData);
    } else if (TILING_KEY_IS(10001900009000290003UL)) {
        KERNEL_TASK_TYPE(10001900009000290003UL, KERNEL_TYPE_MIX_AIC_1_2);
        GET_TILING_DATA_WITH_STRUCT(Mc2MatMulV3BasicTilingData, tilingData, tilingGM);
        MatMulStreamKActKernel<DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS, layout::ColumnMajor, layout::ColumnMajor,
                               layout::RowMajor, MatMulL0C2Out::ON_THE_FLY>(
            aGM, bGM, biasGM, cGM, workspaceGM, tilingData);
    } else if (TILING_KEY_IS(10001902009000290000UL)) {
        KERNEL_TASK_TYPE(10001902009000290000UL, KERNEL_TYPE_MIX_AIC_1_2);
        GET_TILING_DATA_WITH_STRUCT(Mc2MatMulV3BasicTilingData, tilingData, tilingGM);
        MatMulStreamKActKernel<DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS, layout::RowMajor, layout::RowMajor,
                               layout::RowMajor, MatMulL0C2Out::ND_FIXPIPE_1_2>(
            aGM, bGM, biasGM, cGM, workspaceGM, tilingData);
    } else if (TILING_KEY_IS(10001902009000290001UL)) {
        KERNEL_TASK_TYPE(10001902009000290001UL, KERNEL_TYPE_MIX_AIC_1_2);
        GET_TILING_DATA_WITH_STRUCT(Mc2MatMulV3BasicTilingData, tilingData, tilingGM);
        MatMulStreamKActKernel<DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS, layout::ColumnMajor, layout::RowMajor,
                               layout::RowMajor, MatMulL0C2Out::ND_FIXPIPE_1_2>(
            aGM, bGM, biasGM, cGM, workspaceGM, tilingData);
    } else if (TILING_KEY_IS(10001902009000290002UL)) {
        KERNEL_TASK_TYPE(10001902009000290002UL, KERNEL_TYPE_MIX_AIC_1_2);
        GET_TILING_DATA_WITH_STRUCT(Mc2MatMulV3BasicTilingData, tilingData, tilingGM);
        MatMulStreamKActKernel<DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS, layout::RowMajor, layout::ColumnMajor,
                               layout::RowMajor, MatMulL0C2Out::ND_FIXPIPE_1_2>(
            aGM, bGM, biasGM, cGM, workspaceGM, tilingData);
    } else if (TILING_KEY_IS(10001902009000290003UL)) {
        KERNEL_TASK_TYPE(10001902009000290003UL, KERNEL_TYPE_MIX_AIC_1_2);
        GET_TILING_DATA_WITH_STRUCT(Mc2MatMulV3BasicTilingData, tilingData, tilingGM);
        MatMulStreamKActKernel<DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS, layout::ColumnMajor, layout::ColumnMajor,
                               layout::RowMajor, MatMulL0C2Out::ND_FIXPIPE_1_2>(
            aGM, bGM, biasGM, cGM, workspaceGM, tilingData);
    } else if (TILING_KEY_IS(10000900009000290000UL)) {
        KERNEL_TASK_TYPE(10000900009000290000UL, KERNEL_TYPE_MIX_AIC_1_2);
        MMV3_IMPL_CLASS_TRANS(tilingData, tilingGM, false, false, workspaceGM, Mc2MatmulV3Advanced::Mc2MatmulStreamKKernel, Mc2MatmulV3Advanced::MatmulStreamKBlock,
                              MM_CFG_NO_PRELOAD, false);
    } else if (TILING_KEY_IS(10000900009000290001UL)) {
        KERNEL_TASK_TYPE(10000900009000290001UL, KERNEL_TYPE_MIX_AIC_1_2);
        MMV3_IMPL_CLASS_TRANS(tilingData, tilingGM, true, false, workspaceGM, Mc2MatmulV3Advanced::Mc2MatmulStreamKKernel, Mc2MatmulV3Advanced::MatmulStreamKBlock,
                              MM_CFG_NO_PRELOAD, false);
    } else if (TILING_KEY_IS(10000900009000290002UL)) {
        KERNEL_TASK_TYPE(10000900009000290002UL, KERNEL_TYPE_MIX_AIC_1_2);
        MMV3_IMPL_CLASS_TRANS(tilingData, tilingGM, false, true, workspaceGM, Mc2MatmulV3Advanced::Mc2MatmulStreamKKernel, Mc2MatmulV3Advanced::MatmulStreamKBlock,
                              MM_CFG_NO_PRELOAD, false);
    } else if (TILING_KEY_IS(10000900009000290003UL)) {
        KERNEL_TASK_TYPE(10000900009000290003UL, KERNEL_TYPE_MIX_AIC_1_2);
        MMV3_IMPL_CLASS_TRANS(tilingData, tilingGM, true, true, workspaceGM, Mc2MatmulV3Advanced::Mc2MatmulStreamKKernel, Mc2MatmulV3Advanced::MatmulStreamKBlock,
                              MM_CFG_NO_PRELOAD, false);
    } else if (TILING_KEY_IS(10000902009000290000UL)) {
        KERNEL_TASK_TYPE(10000902009000290000UL, KERNEL_TYPE_MIX_AIC_1_2);
        MMV3_IMPL_CLASS_TRANS(tilingData, tilingGM, false, false, workspaceGM, Mc2MatmulV3Advanced::Mc2MatmulStreamKKernel,
                                    Mc2MatmulV3Advanced::MatmulStreamKBlock, MM_CFG_NO_PRELOAD, true);
    } else if (TILING_KEY_IS(10000902009000290001UL)) {
        KERNEL_TASK_TYPE(10000902009000290001UL, KERNEL_TYPE_MIX_AIC_1_2);
        MMV3_IMPL_CLASS_TRANS(tilingData, tilingGM, true, false, workspaceGM, Mc2MatmulV3Advanced::Mc2MatmulStreamKKernel,
                                    Mc2MatmulV3Advanced::MatmulStreamKBlock, MM_CFG_NO_PRELOAD, true);
    } else if (TILING_KEY_IS(10000902009000290002UL)) {
        KERNEL_TASK_TYPE(10000902009000290002UL, KERNEL_TYPE_MIX_AIC_1_2);
        MMV3_IMPL_CLASS_TRANS(tilingData, tilingGM, false, true, workspaceGM, Mc2MatmulV3Advanced::Mc2MatmulStreamKKernel,
                                    Mc2MatmulV3Advanced::MatmulStreamKBlock, MM_CFG_NO_PRELOAD, true);
    } else if (TILING_KEY_IS(10000902009000290003UL)) {
        KERNEL_TASK_TYPE(10000902009000290003UL, KERNEL_TYPE_MIX_AIC_1_2);
        MMV3_IMPL_CLASS_TRANS(tilingData, tilingGM, true, true, workspaceGM, Mc2MatmulV3Advanced::Mc2MatmulStreamKKernel,
                                    Mc2MatmulV3Advanced::MatmulStreamKBlock, MM_CFG_NO_PRELOAD, true);
    } else if (TILING_KEY_IS(10000900009001090000UL)) {
        MMV3_IMPL_CLASS_TRANS(tilingData, tilingGM, false, false, nullptr, Mc2MatmulV3Advanced::MatmulAswKernelAL1FullLoad,
            Mc2MatmulV3Advanced::Mc2MatmulAswBlock, MM_CFG_NO_PRELOAD);
    } else if (TILING_KEY_IS(10000900009001090001UL)) {
        MMV3_IMPL_CLASS_TRANS(tilingData, tilingGM, true, false, nullptr, Mc2MatmulV3Advanced::MatmulAswKernelAL1FullLoad,
            Mc2MatmulV3Advanced::Mc2MatmulAswBlock, MM_CFG_NO_PRELOAD);
    } else if (TILING_KEY_IS(10000900009001090002UL)) {
        MMV3_IMPL_CLASS_TRANS(tilingData, tilingGM, false, true, nullptr, Mc2MatmulV3Advanced::MatmulAswKernelAL1FullLoad,
            Mc2MatmulV3Advanced::Mc2MatmulAswBlock, MM_CFG_NO_PRELOAD);
    } else if (TILING_KEY_IS(10000900009001090003UL)) {
        MMV3_IMPL_CLASS_TRANS(tilingData, tilingGM, true, true, nullptr, Mc2MatmulV3Advanced::MatmulAswKernelAL1FullLoad,
            Mc2MatmulV3Advanced::Mc2MatmulAswBlock, MM_CFG_NO_PRELOAD);
    } else if (TILING_KEY_IS(10000900009001490000UL)) {
        MMV3_IMPL_CLASS_TRANS(tilingData, tilingGM, false, false, nullptr, Mc2MatmulV3Advanced::MatmulAswKernelAL1FullLoad,
            Mc2MatmulV3Advanced::MatmulDaswBlock, MM_CFG_NO_PRELOAD);
    } else if (TILING_KEY_IS(10000900009001490001UL)) {
        MMV3_IMPL_CLASS_TRANS(tilingData, tilingGM, true, false, nullptr, Mc2MatmulV3Advanced::MatmulAswKernelAL1FullLoad,
            Mc2MatmulV3Advanced::MatmulDaswBlock, MM_CFG_NO_PRELOAD);
    } else if (TILING_KEY_IS(10000900009001490002UL)) {
        MMV3_IMPL_CLASS_TRANS(tilingData, tilingGM, false, true, nullptr, Mc2MatmulV3Advanced::MatmulAswKernelAL1FullLoad,
            Mc2MatmulV3Advanced::MatmulDaswBlock, MM_CFG_NO_PRELOAD);
    } else if (TILING_KEY_IS(10000900009001490003UL)) {
        MMV3_IMPL_CLASS_TRANS(tilingData, tilingGM, true, true, nullptr, Mc2MatmulV3Advanced::MatmulAswKernelAL1FullLoad,
            Mc2MatmulV3Advanced::MatmulDaswBlock, MM_CFG_NO_PRELOAD);
    } else if (TILING_KEY_IS(10000900009002090000UL)) {
        MMV3_IMPL_CLASS_TRANS(tilingData, tilingGM, false, false, nullptr, Mc2MatmulV3Advanced::Mc2MatmulAswKernelBL1FullLoad,
            Mc2MatmulV3Advanced::Mc2MatmulAswBlock, MM_CFG_NO_PRELOAD);
    } else if (TILING_KEY_IS(10000900009002090001UL)) {
        MMV3_IMPL_CLASS_TRANS(tilingData, tilingGM, true, false, nullptr, Mc2MatmulV3Advanced::Mc2MatmulAswKernelBL1FullLoad,
            Mc2MatmulV3Advanced::Mc2MatmulAswBlock, MM_CFG_NO_PRELOAD);
    } else if (TILING_KEY_IS(10000900009002090002UL)) {
        MMV3_IMPL_CLASS_TRANS(tilingData, tilingGM, false, true, nullptr, Mc2MatmulV3Advanced::Mc2MatmulAswKernelBL1FullLoad,
            Mc2MatmulV3Advanced::Mc2MatmulAswBlock, MM_CFG_NO_PRELOAD);
    } else if (TILING_KEY_IS(10000900009002090003UL)) {
        MMV3_IMPL_CLASS_TRANS(tilingData, tilingGM, true, true, nullptr, Mc2MatmulV3Advanced::Mc2MatmulAswKernelBL1FullLoad,
            Mc2MatmulV3Advanced::Mc2MatmulAswBlock, MM_CFG_NO_PRELOAD);
    } else if (TILING_KEY_IS(10000900009002490000UL)) {
        MMV3_IMPL_CLASS_TRANS(tilingData, tilingGM, false, false, nullptr, Mc2MatmulV3Advanced::Mc2MatmulAswKernelBL1FullLoad,
            Mc2MatmulV3Advanced::MatmulDaswBlock, MM_CFG_NO_PRELOAD);
    } else if (TILING_KEY_IS(10000900009002490001UL)) {
        MMV3_IMPL_CLASS_TRANS(tilingData, tilingGM, true, false, nullptr, Mc2MatmulV3Advanced::Mc2MatmulAswKernelBL1FullLoad,
            Mc2MatmulV3Advanced::MatmulDaswBlock, MM_CFG_NO_PRELOAD);
    } else if (TILING_KEY_IS(10000900009002490002UL)) {
        MMV3_IMPL_CLASS_TRANS(tilingData, tilingGM, false, true, nullptr, Mc2MatmulV3Advanced::Mc2MatmulAswKernelBL1FullLoad,
            Mc2MatmulV3Advanced::MatmulDaswBlock, MM_CFG_NO_PRELOAD);
    } else if (TILING_KEY_IS(10000900009002490003UL)) {
        MMV3_IMPL_CLASS_TRANS(tilingData, tilingGM, true, true, nullptr, Mc2MatmulV3Advanced::Mc2MatmulAswKernelBL1FullLoad,
            Mc2MatmulV3Advanced::MatmulDaswBlock, MM_CFG_NO_PRELOAD);
    } else if (TILING_KEY_IS(10000901009000090000UL)) {
        KERNEL_TASK_TYPE(10000901009000090000UL, KERNEL_TYPE_MIX_AIC_1_2);
        MMV3_IMPL_CLASS_TRANS(tilingData, tilingGM, false, false, workspaceGM, Mc2MatmulV3Advanced::Mc2MatmulFixpipeOptiKernel,
            Mc2MatmulV3Advanced::Mc2MatmulAswBlock, MM_CFG_NO_PRELOAD);
    } else if (TILING_KEY_IS(10000901009000090001UL)) {
        KERNEL_TASK_TYPE(10000901009000090001UL, KERNEL_TYPE_MIX_AIC_1_2);
        MMV3_IMPL_CLASS_TRANS(tilingData, tilingGM, true, false, workspaceGM, Mc2MatmulV3Advanced::Mc2MatmulFixpipeOptiKernel,
            Mc2MatmulV3Advanced::Mc2MatmulAswBlock, MM_CFG_NO_PRELOAD);
    } else if (TILING_KEY_IS(10000901009000090002UL)) {
        KERNEL_TASK_TYPE(10000901009000090002UL, KERNEL_TYPE_MIX_AIC_1_2);
        MMV3_IMPL_CLASS_TRANS(tilingData, tilingGM, false, true, workspaceGM, Mc2MatmulV3Advanced::Mc2MatmulFixpipeOptiKernel,
            Mc2MatmulV3Advanced::Mc2MatmulAswBlock, MM_CFG_NO_PRELOAD);
    } else if (TILING_KEY_IS(10000901009000090003UL)) {
        KERNEL_TASK_TYPE(10000901009000090003UL, KERNEL_TYPE_MIX_AIC_1_2);
        MMV3_IMPL_CLASS_TRANS(tilingData, tilingGM, true, true, workspaceGM, Mc2MatmulV3Advanced::Mc2MatmulFixpipeOptiKernel,
            Mc2MatmulV3Advanced::Mc2MatmulAswBlock, MM_CFG_NO_PRELOAD);
    } else if (TILING_KEY_IS(10001901009002090000UL)) {
        GET_TILING_DATA_WITH_STRUCT(Mc2MatMulV3BasicTilingData, tilingData, tilingGM);
        KERNEL_TASK_TYPE(10001901009002090000UL, KERNEL_TYPE_MIX_AIC_1_2);
        Mc2MatmulV3Advanced::MatMulFixpipeOptiActKernel<DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS, layout::RowMajor,
        layout::RowMajor, layout::RowMajor, B_FULL_LOAD_MODE>(aGM, bGM, biasGM, cGM, workspaceGM, tilingData);
    } else if (TILING_KEY_IS(10001901009002090001UL)) {
        GET_TILING_DATA_WITH_STRUCT(Mc2MatMulV3BasicTilingData, tilingData, tilingGM);
        KERNEL_TASK_TYPE(10001901009002090001UL, KERNEL_TYPE_MIX_AIC_1_2);
        Mc2MatmulV3Advanced::MatMulFixpipeOptiActKernel<DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS, layout::ColumnMajor,
        layout::RowMajor, layout::RowMajor, B_FULL_LOAD_MODE>(aGM, bGM, biasGM, cGM, workspaceGM, tilingData);
    } else if (TILING_KEY_IS(10001901009002090002UL)) {
        GET_TILING_DATA_WITH_STRUCT(Mc2MatMulV3BasicTilingData, tilingData, tilingGM);
        KERNEL_TASK_TYPE(10001901009002090002UL, KERNEL_TYPE_MIX_AIC_1_2);
        Mc2MatmulV3Advanced::MatMulFixpipeOptiActKernel<DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS, layout::RowMajor,
        layout::ColumnMajor, layout::RowMajor, B_FULL_LOAD_MODE>(aGM, bGM, biasGM, cGM, workspaceGM, tilingData);
    } else if (TILING_KEY_IS(10001901009002090003UL)) {
        GET_TILING_DATA_WITH_STRUCT(Mc2MatMulV3BasicTilingData, tilingData, tilingGM);
        KERNEL_TASK_TYPE(10001901009002090003UL, KERNEL_TYPE_MIX_AIC_1_2);
        Mc2MatmulV3Advanced::MatMulFixpipeOptiActKernel<DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS, layout::ColumnMajor,
        layout::ColumnMajor, layout::RowMajor, B_FULL_LOAD_MODE>(aGM, bGM, biasGM, cGM, workspaceGM, tilingData);
    } else if (TILING_KEY_IS(10001901009000090000UL)) {
        GET_TILING_DATA_WITH_STRUCT(Mc2MatMulV3BasicTilingData, tilingData, tilingGM);
        KERNEL_TASK_TYPE(10001901009000090000UL, KERNEL_TYPE_MIX_AIC_1_2);
        Mc2MatmulV3Advanced::MatMulFixpipeOptiActKernel<DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS, layout::RowMajor,
        layout::RowMajor, layout::RowMajor>(aGM, bGM, biasGM, cGM, workspaceGM, tilingData);
    } else if (TILING_KEY_IS(10001901009000090001UL)) {
        GET_TILING_DATA_WITH_STRUCT(Mc2MatMulV3BasicTilingData, tilingData, tilingGM);
        KERNEL_TASK_TYPE(10001901009000090001UL, KERNEL_TYPE_MIX_AIC_1_2);
        Mc2MatmulV3Advanced::MatMulFixpipeOptiActKernel<DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS, layout::ColumnMajor,
        layout::RowMajor, layout::RowMajor>(aGM, bGM, biasGM, cGM, workspaceGM, tilingData);
    } else if (TILING_KEY_IS(10001901009000090002UL)) {
        GET_TILING_DATA_WITH_STRUCT(Mc2MatMulV3BasicTilingData, tilingData, tilingGM);
        KERNEL_TASK_TYPE(10001901009000090002UL, KERNEL_TYPE_MIX_AIC_1_2);
        Mc2MatmulV3Advanced::MatMulFixpipeOptiActKernel<DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS, layout::RowMajor,
        layout::ColumnMajor, layout::RowMajor>(aGM, bGM, biasGM, cGM, workspaceGM, tilingData);
    } else if (TILING_KEY_IS(10001901009000090003UL)) {
        GET_TILING_DATA_WITH_STRUCT(Mc2MatMulV3BasicTilingData, tilingData, tilingGM);
        KERNEL_TASK_TYPE(10001901009000090003UL, KERNEL_TYPE_MIX_AIC_1_2);
        Mc2MatmulV3Advanced::MatMulFixpipeOptiActKernel<DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS, layout::ColumnMajor,
        layout::ColumnMajor, layout::RowMajor>(aGM, bGM, biasGM, cGM, workspaceGM, tilingData);
    } else if (TILING_KEY_IS(10001902009000090000UL)) {
        GET_TILING_DATA_WITH_STRUCT(Mc2MatMulV3BasicTilingData, tilingData, tilingGM);
        KERNEL_TASK_TYPE(10001902009000090000UL, KERNEL_TYPE_MIX_AIC_1_2);
        Mc2MatmulV3Advanced::MatMulFixpipeOptiActKernel<DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS, layout::RowMajor,
        layout::RowMajor, layout::RowMajor>(aGM, bGM, biasGM, cGM, workspaceGM, tilingData);
    } else if (TILING_KEY_IS(10001902009000090001UL)) {
        GET_TILING_DATA_WITH_STRUCT(Mc2MatMulV3BasicTilingData, tilingData, tilingGM);
        KERNEL_TASK_TYPE(10001902009000090001UL, KERNEL_TYPE_MIX_AIC_1_2);
        Mc2MatmulV3Advanced::MatMulFixpipeOptiActKernel<DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS, layout::ColumnMajor,
        layout::RowMajor, layout::RowMajor>(aGM, bGM, biasGM, cGM, workspaceGM, tilingData);
    } else if (TILING_KEY_IS(10001902009000090002UL)) {
        GET_TILING_DATA_WITH_STRUCT(Mc2MatMulV3BasicTilingData, tilingData, tilingGM);
        KERNEL_TASK_TYPE(10001902009000090002UL, KERNEL_TYPE_MIX_AIC_1_2);
        Mc2MatmulV3Advanced::MatMulFixpipeOptiActKernel<DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS, layout::RowMajor,
        layout::ColumnMajor, layout::RowMajor>(aGM, bGM, biasGM, cGM, workspaceGM, tilingData);
    } else if (TILING_KEY_IS(10001902009000090003UL)) {
        GET_TILING_DATA_WITH_STRUCT(Mc2MatMulV3BasicTilingData, tilingData, tilingGM);
        KERNEL_TASK_TYPE(10001902009000090003UL, KERNEL_TYPE_MIX_AIC_1_2);
        Mc2MatmulV3Advanced::MatMulFixpipeOptiActKernel<DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS, layout::ColumnMajor,
        layout::ColumnMajor, layout::RowMajor>(aGM, bGM, biasGM, cGM, workspaceGM, tilingData);
    } else if (TILING_KEY_IS(10000901009000490000UL)) {
        KERNEL_TASK_TYPE(10000901009000490000UL, KERNEL_TYPE_MIX_AIC_1_2);
        MMV3_IMPL_CLASS_TRANS(tilingData, tilingGM, false, false, workspaceGM, Mc2MatmulV3Advanced::Mc2MatmulFixpipeOptiKernel,
            Mc2MatmulV3Advanced::MatmulDaswBlock, MM_CFG_NO_PRELOAD);
    } else if (TILING_KEY_IS(10000901009000490001UL)) {
        KERNEL_TASK_TYPE(10000901009000490001UL, KERNEL_TYPE_MIX_AIC_1_2);
        MMV3_IMPL_CLASS_TRANS(tilingData, tilingGM, true, false, workspaceGM, Mc2MatmulV3Advanced::Mc2MatmulFixpipeOptiKernel,
            Mc2MatmulV3Advanced::MatmulDaswBlock, MM_CFG_NO_PRELOAD);
    } else if (TILING_KEY_IS(10000901009000490002UL)) {
        KERNEL_TASK_TYPE(10000901009000490002UL, KERNEL_TYPE_MIX_AIC_1_2);
        MMV3_IMPL_CLASS_TRANS(tilingData, tilingGM, false, true, workspaceGM, Mc2MatmulV3Advanced::Mc2MatmulFixpipeOptiKernel,
            Mc2MatmulV3Advanced::MatmulDaswBlock, MM_CFG_NO_PRELOAD);
    } else if (TILING_KEY_IS(10000901009000490003UL)) {
        KERNEL_TASK_TYPE(10000901009000490003UL, KERNEL_TYPE_MIX_AIC_1_2);
        MMV3_IMPL_CLASS_TRANS(tilingData, tilingGM, true, true, workspaceGM, Mc2MatmulV3Advanced::Mc2MatmulFixpipeOptiKernel,
            Mc2MatmulV3Advanced::MatmulDaswBlock, MM_CFG_NO_PRELOAD);
    } else if (TILING_KEY_IS(10000902009000090000UL)) {
        KERNEL_TASK_TYPE(10000902009000090000UL, KERNEL_TYPE_MIX_AIC_1_2);
        MMV3_IMPL_CLASS_TRANS(tilingData, tilingGM, false, false, workspaceGM, Mc2MatmulV3Advanced::Mc2MatmulFixpipeOptiDualDstKernel,
            Mc2MatmulV3Advanced::Mc2MatmulAswBlock, MM_CFG_NO_PRELOAD);
    } else if (TILING_KEY_IS(10000902009000090001UL)) {
        KERNEL_TASK_TYPE(10000902009000090001UL, KERNEL_TYPE_MIX_AIC_1_2);
        MMV3_IMPL_CLASS_TRANS(tilingData, tilingGM, true, false, workspaceGM, Mc2MatmulV3Advanced::Mc2MatmulFixpipeOptiDualDstKernel,
            Mc2MatmulV3Advanced::Mc2MatmulAswBlock, MM_CFG_NO_PRELOAD);
    } else if (TILING_KEY_IS(10000902009000090002UL)) {
        KERNEL_TASK_TYPE(10000902009000090002UL, KERNEL_TYPE_MIX_AIC_1_2);
        MMV3_IMPL_CLASS_TRANS(tilingData, tilingGM, false, true, workspaceGM, Mc2MatmulV3Advanced::Mc2MatmulFixpipeOptiDualDstKernel,
            Mc2MatmulV3Advanced::Mc2MatmulAswBlock, MM_CFG_NO_PRELOAD);
    } else if (TILING_KEY_IS(10000902009000090003UL)) {
        KERNEL_TASK_TYPE(10000902009000090003UL, KERNEL_TYPE_MIX_AIC_1_2);
        MMV3_IMPL_CLASS_TRANS(tilingData, tilingGM, true, true, workspaceGM, Mc2MatmulV3Advanced::Mc2MatmulFixpipeOptiDualDstKernel,
            Mc2MatmulV3Advanced::Mc2MatmulAswBlock, MM_CFG_NO_PRELOAD);
    } else if (TILING_KEY_IS(10001902009002090000UL)) { // Fixpipe B全载fp32场景切换act kernel
        GET_TILING_DATA_WITH_STRUCT(Mc2MatMulV3BasicTilingData, tilingData, tilingGM);
        KERNEL_TASK_TYPE(10001902009002090000UL, KERNEL_TYPE_MIX_AIC_1_2);
        Mc2MatmulV3Advanced::MatMulFixpipeOptiActKernel<DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS, layout::RowMajor,
        layout::RowMajor, layout::RowMajor, B_FULL_LOAD_MODE>(aGM, bGM, biasGM, cGM, workspaceGM, tilingData);
    } else if (TILING_KEY_IS(10001902009002090001UL)) {
        GET_TILING_DATA_WITH_STRUCT(Mc2MatMulV3BasicTilingData, tilingData, tilingGM);
        KERNEL_TASK_TYPE(10001902009002090001UL, KERNEL_TYPE_MIX_AIC_1_2);
        Mc2MatmulV3Advanced::MatMulFixpipeOptiActKernel<DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS, layout::ColumnMajor,
        layout::RowMajor, layout::RowMajor, B_FULL_LOAD_MODE>(aGM, bGM, biasGM, cGM, workspaceGM, tilingData);
    } else if (TILING_KEY_IS(10001902009002090002UL)) {
        GET_TILING_DATA_WITH_STRUCT(Mc2MatMulV3BasicTilingData, tilingData, tilingGM);
        KERNEL_TASK_TYPE(10001902009002090002UL, KERNEL_TYPE_MIX_AIC_1_2);
        Mc2MatmulV3Advanced::MatMulFixpipeOptiActKernel<DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS, layout::RowMajor,
        layout::ColumnMajor, layout::RowMajor, B_FULL_LOAD_MODE>(aGM, bGM, biasGM, cGM, workspaceGM, tilingData);
    } else if (TILING_KEY_IS(10001902009002090003UL)) {
        GET_TILING_DATA_WITH_STRUCT(Mc2MatMulV3BasicTilingData, tilingData, tilingGM);
        KERNEL_TASK_TYPE(10001902009002090003UL, KERNEL_TYPE_MIX_AIC_1_2);
        Mc2MatmulV3Advanced::MatMulFixpipeOptiActKernel<DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS, layout::ColumnMajor,
        layout::ColumnMajor, layout::RowMajor, B_FULL_LOAD_MODE>(aGM, bGM, biasGM, cGM, workspaceGM, tilingData);
    } else if (TILING_KEY_IS(10000902009000490000UL)) {
        KERNEL_TASK_TYPE(10000902009000490000UL, KERNEL_TYPE_MIX_AIC_1_2);
        MMV3_IMPL_CLASS_TRANS(tilingData, tilingGM, false, false, workspaceGM, Mc2MatmulV3Advanced::Mc2MatmulFixpipeOptiDualDstKernel,
            Mc2MatmulV3Advanced::MatmulDaswBlock, MM_CFG_NO_PRELOAD);
    } else if (TILING_KEY_IS(10000902009000490001UL)) {
        KERNEL_TASK_TYPE(10000902009000490001UL, KERNEL_TYPE_MIX_AIC_1_2);
        MMV3_IMPL_CLASS_TRANS(tilingData, tilingGM, true, false, workspaceGM, Mc2MatmulV3Advanced::Mc2MatmulFixpipeOptiDualDstKernel,
            Mc2MatmulV3Advanced::MatmulDaswBlock, MM_CFG_NO_PRELOAD);
    } else if (TILING_KEY_IS(10000902009000490002UL)) {
        KERNEL_TASK_TYPE(10000902009000490002UL, KERNEL_TYPE_MIX_AIC_1_2);
        MMV3_IMPL_CLASS_TRANS(tilingData, tilingGM, false, true, workspaceGM, Mc2MatmulV3Advanced::Mc2MatmulFixpipeOptiDualDstKernel,
            Mc2MatmulV3Advanced::MatmulDaswBlock, MM_CFG_NO_PRELOAD);
    } else if (TILING_KEY_IS(10000902009000490003UL)) {
        KERNEL_TASK_TYPE(10000902009000490003UL, KERNEL_TYPE_MIX_AIC_1_2);
        MMV3_IMPL_CLASS_TRANS(tilingData, tilingGM, true, true, workspaceGM, Mc2MatmulV3Advanced::Mc2MatmulFixpipeOptiDualDstKernel,
            Mc2MatmulV3Advanced::MatmulDaswBlock, MM_CFG_NO_PRELOAD);
    } else if (TILING_KEY_IS(10000900009003090000UL)) {
        MMV3_IMPL_CLASS_TRANS(tilingData, tilingGM, false, false, nullptr, Mc2MatmulV3Advanced::Mc2MatmulAswKernelABL1FullLoad,
            Mc2MatmulV3Advanced::Mc2MatmulAswBlock, MM_CFG_NO_PRELOAD);
    } else if (TILING_KEY_IS(10000900009003090001UL)) {
        MMV3_IMPL_CLASS_TRANS(tilingData, tilingGM, true, false, nullptr, Mc2MatmulV3Advanced::Mc2MatmulAswKernelABL1FullLoad,
            Mc2MatmulV3Advanced::Mc2MatmulAswBlock, MM_CFG_NO_PRELOAD);
    } else if (TILING_KEY_IS(10000900009003090002UL)) {
        MMV3_IMPL_CLASS_TRANS(tilingData, tilingGM, false, true, nullptr, Mc2MatmulV3Advanced::Mc2MatmulAswKernelABL1FullLoad,
            Mc2MatmulV3Advanced::Mc2MatmulAswBlock, MM_CFG_NO_PRELOAD);
    } else if (TILING_KEY_IS(10000900009003090003UL)) {
        MMV3_IMPL_CLASS_TRANS(tilingData, tilingGM, true, true, nullptr, Mc2MatmulV3Advanced::Mc2MatmulAswKernelABL1FullLoad,
            Mc2MatmulV3Advanced::Mc2MatmulAswBlock, MM_CFG_NO_PRELOAD);
#endif
    }
}
