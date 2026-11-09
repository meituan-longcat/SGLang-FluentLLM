/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_init_routing_v3_apt.cpp
 * \brief
 */
#include "arch35/moe_v3_mrgsort_out.h"
#include "arch35/moe_v3_mrgsort.h"
#include "arch35/moe_v3_sort_one_core.h"
#include "arch35/moe_v3_sort_multi_core.h"
#include "arch35/moe_v3_expert_tokens_count.h"
#include "arch35/moe_v3_row_idx_gather.h"
#include "arch35/moe_v3_gather_out.h"
#include "arch35/moe_v3_gather_dynamic_quant.h"
#include "arch35/moe_v3_gather_mxfp8_quant.h"
#include "arch35/moe_v3_gather_hif8_pertensor_quant.h"
#include "arch35/moe_v3_gather_hif8_pertoken_quant.h"
#include "arch35/moe_v3_gather_hif8_quant.h"

/*
 * 非量化
 */
#define MOE_INIT_ROUTING_V3_SORTONECORE_GATHER 1000000    // 单核排序、非量化、GATHER索引
#define MOE_INIT_ROUTING_V3_SORTONECORE_SCATTER 1001000   // 单核排序、非量化、SCATTER索引
#define MOE_INIT_ROUTING_V3_SORTMULTICORE_GATHER 1100000  // 多核排序、非量化、GATHER索引
#define MOE_INIT_ROUTING_V3_SORTMULTICORE_SCATTER 1101000 // 多核排序、非量化、SCATTER索引

/*
 * 动态量化
 */
#define MOE_INIT_ROUTING_V3_SORTONECORE_DYNAMICQUANT_GATHER 1020000    // 单核排序、动态量化、GATHER索引
#define MOE_INIT_ROUTING_V3_SORTONECORE_DYNAMICQUANT_SCATTER 1021000   // 单核排序、动态量化、SCATTER索引
#define MOE_INIT_ROUTING_V3_SORTMULTICORE_DYNAMICQUANT_GATHER 1120000  // 多核排序、动态量化、GATHER索引
#define MOE_INIT_ROUTING_V3_SORTMULTICORE_DYNAMICQUANT_SCATTER 1121000 // 多核排序、动态量化、SCATTER索引

/*
 * MXFP8量化
 */
#define MOE_INIT_ROUTING_V3_SORTONECORE_MXFP8QUANT_GATHER 1030000    // 单核排序、MXFP8量化、GATHER索引
#define MOE_INIT_ROUTING_V3_SORTONECORE_MXFP8QUANT_SCATTER 1031000   // 单核排序、MXFP8量化、SCATTER索引
#define MOE_INIT_ROUTING_V3_SORTMULTICORE_MXFP8QUANT_GATHER 1130000  // 多核排序、MXFP8量化、GATHER索引
#define MOE_INIT_ROUTING_V3_SORTMULTICORE_MXFP8QUANT_SCATTER 1131000 // 多核排序、MXFP8量化、SCATTER索引

/*
 * Hif8 直转
 */
#define MOE_INIT_ROUTING_V3_SORTONECORE_HIF8CAST_GATHER 1070000    // 单核排序、Hif8直转、GATHER索引
#define MOE_INIT_ROUTING_V3_SORTONECORE_HIF8CAST_SCATTER 1071000   // 单核排序、Hif8直转、SCATTER索引
#define MOE_INIT_ROUTING_V3_SORTMULTICORE_HIF8CAST_GATHER 1170000  // 多核排序、Hif8直转、GATHER索引
#define MOE_INIT_ROUTING_V3_SORTMULTICORE_HIF8CAST_SCATTER 1171000 // 多核排序、Hif8直转、SCATTER索引

/*
 * HIF8 PENTENSOR量化
 */
#define MOE_INIT_ROUTING_V3_SORTONECORE_HIF8_PERTENSOR_QUANT_GATHER 1080000    // 单核排序、HIF8 PENTENSOR量化、GATHER索引
#define MOE_INIT_ROUTING_V3_SORTONECORE_HIF8_PERTENSOR_QUANT_SCATTER 1081000   // 单核排序、HIF8 PENTENSOR量化、SCATTER索引
#define MOE_INIT_ROUTING_V3_SORTMULTICORE_HIF8_PERTENSOR_QUANT_GATHER 1180000  // 多核排序、HIF8 PENTENSOR量化、GATHER索引
#define MOE_INIT_ROUTING_V3_SORTMULTICORE_HIF8_PERTENSOR_QUANT_SCATTER 1181000 // 多核排序、HIF8 PENTENSOR量化、SCATTER索引

/*
 * HIF8 PENTEOKEN量化
 */
#define MOE_INIT_ROUTING_V3_SORTONECORE_HIF8_PERTOKEN_QUANT_GATHER 1090000    // 单核排序、HIF8 PENTEOKEN量化、GATHER索引
#define MOE_INIT_ROUTING_V3_SORTONECORE_HIF8_PERTOKEN_QUANT_SCATTER 1091000   // 单核排序、HIF8 PENTEOKEN量化、SCATTER索引
#define MOE_INIT_ROUTING_V3_SORTMULTICORE_HIF8_PERTOKEN_QUANT_GATHER 1190000  // 多核排序、HIF8 PENTEOKEN量化、GATHER索引
#define MOE_INIT_ROUTING_V3_SORTMULTICORE_HIF8_PERTOKEN_QUANT_SCATTER 1191000 // 多核排序、HIF8 PENTEOKEN量化、SCATTER索引

using namespace AscendC;
using namespace MoeInitRoutingV3;
extern "C" __global__ __aicore__ void moe_init_routing_v3(GM_ADDR x, GM_ADDR expertIdx, GM_ADDR scale, GM_ADDR offset,
                                                          GM_ADDR expandedX, GM_ADDR expandedRowIdx,
                                                          GM_ADDR expertTokensCountOrCumsum, GM_ADDR expandedScale,
                                                          GM_ADDR workspace, GM_ADDR tiling)
{
    if (g_coreType == AIC) {
        return;
    }

    if (workspace == nullptr) {
        return;
    }

    REGISTER_TILING_DEFAULT(MoeInitRoutingV3Arch35TilingData);
    GET_TILING_DATA_WITH_STRUCT(MoeInitRoutingV3Arch35TilingData, tilingData, tiling);

    GM_ADDR userWS = GetUserWorkspace(workspace);
    if (userWS == nullptr) {
        return;
    }

#if (__NPU_ARCH__ == 3510)
    int64_t oriOverflowMode = GetCtrlSpr<OVERFLOW_MODE_CTRL, OVERFLOW_MODE_CTRL>();
#endif

    auto t = &tilingData;

    // 1.排序阶段，计算SortedExpertIdx和SortedRowIdx。若rowIdxType=1(Scatter)，则SortedRowIdx直接写到输出expandedRowIdx。
    TPipe sortPipe;
    if (TILING_KEY_IS(MOE_INIT_ROUTING_V3_SORTONECORE_GATHER) ||
        TILING_KEY_IS(MOE_INIT_ROUTING_V3_SORTONECORE_SCATTER) ||
        TILING_KEY_IS(MOE_INIT_ROUTING_V3_SORTONECORE_DYNAMICQUANT_GATHER) ||
        TILING_KEY_IS(MOE_INIT_ROUTING_V3_SORTONECORE_DYNAMICQUANT_SCATTER) ||
        TILING_KEY_IS(MOE_INIT_ROUTING_V3_SORTONECORE_MXFP8QUANT_GATHER) ||
        TILING_KEY_IS(MOE_INIT_ROUTING_V3_SORTONECORE_MXFP8QUANT_SCATTER) ||
        TILING_KEY_IS(MOE_INIT_ROUTING_V3_SORTONECORE_HIF8CAST_GATHER) ||
        TILING_KEY_IS(MOE_INIT_ROUTING_V3_SORTONECORE_HIF8CAST_SCATTER) ||
        TILING_KEY_IS(MOE_INIT_ROUTING_V3_SORTONECORE_HIF8_PERTENSOR_QUANT_GATHER) ||
        TILING_KEY_IS(MOE_INIT_ROUTING_V3_SORTONECORE_HIF8_PERTENSOR_QUANT_SCATTER) ||
        TILING_KEY_IS(MOE_INIT_ROUTING_V3_SORTONECORE_HIF8_PERTOKEN_QUANT_GATHER) ||
        TILING_KEY_IS(MOE_INIT_ROUTING_V3_SORTONECORE_HIF8_PERTOKEN_QUANT_SCATTER)) {
        // 单核排序
        MoeSortOneCore op;
        op.Init(expertIdx, expandedRowIdx, userWS, t, &sortPipe);
        op.Process();
    } else if (TILING_KEY_IS(MOE_INIT_ROUTING_V3_SORTMULTICORE_GATHER) ||
               TILING_KEY_IS(MOE_INIT_ROUTING_V3_SORTMULTICORE_SCATTER) ||
               TILING_KEY_IS(MOE_INIT_ROUTING_V3_SORTMULTICORE_DYNAMICQUANT_GATHER) ||
               TILING_KEY_IS(MOE_INIT_ROUTING_V3_SORTMULTICORE_DYNAMICQUANT_SCATTER) ||
               TILING_KEY_IS(MOE_INIT_ROUTING_V3_SORTMULTICORE_MXFP8QUANT_GATHER) ||
               TILING_KEY_IS(MOE_INIT_ROUTING_V3_SORTMULTICORE_MXFP8QUANT_SCATTER) ||
               TILING_KEY_IS(MOE_INIT_ROUTING_V3_SORTMULTICORE_HIF8CAST_GATHER) ||
               TILING_KEY_IS(MOE_INIT_ROUTING_V3_SORTMULTICORE_HIF8CAST_SCATTER) ||
               TILING_KEY_IS(MOE_INIT_ROUTING_V3_SORTMULTICORE_HIF8_PERTENSOR_QUANT_GATHER) ||
               TILING_KEY_IS(MOE_INIT_ROUTING_V3_SORTMULTICORE_HIF8_PERTENSOR_QUANT_SCATTER) ||
               TILING_KEY_IS(MOE_INIT_ROUTING_V3_SORTMULTICORE_HIF8_PERTOKEN_QUANT_GATHER) ||
               TILING_KEY_IS(MOE_INIT_ROUTING_V3_SORTMULTICORE_HIF8_PERTOKEN_QUANT_SCATTER)) {
        // 多核排序
        MoeSortMultiCore op;
        op.Init(expertIdx, expandedRowIdx, userWS, t, &sortPipe);
        op.Process();
    }
    sortPipe.Destroy();

    // 2.TokensCount阶段，计算输出expertTokensCountOrCumsum
    TPipe histogramPipe;
    ExpertTokensCount countOp;
    countOp.Init(expandedRowIdx, expertTokensCountOrCumsum, userWS, t, &histogramPipe);
    countOp.Process();
    histogramPipe.Destroy();

    // 3.若rowIdxType=0(Gather)，映射计算输出expandedRowIdx；否则该输出在阶段1就被写出
    if (TILING_KEY_IS(MOE_INIT_ROUTING_V3_SORTONECORE_GATHER) ||
        TILING_KEY_IS(MOE_INIT_ROUTING_V3_SORTMULTICORE_GATHER) ||
        TILING_KEY_IS(MOE_INIT_ROUTING_V3_SORTONECORE_DYNAMICQUANT_GATHER) ||
        TILING_KEY_IS(MOE_INIT_ROUTING_V3_SORTMULTICORE_DYNAMICQUANT_GATHER) ||
        TILING_KEY_IS(MOE_INIT_ROUTING_V3_SORTONECORE_MXFP8QUANT_GATHER) ||
        TILING_KEY_IS(MOE_INIT_ROUTING_V3_SORTMULTICORE_MXFP8QUANT_GATHER) ||
        TILING_KEY_IS(MOE_INIT_ROUTING_V3_SORTONECORE_HIF8CAST_GATHER) ||
        TILING_KEY_IS(MOE_INIT_ROUTING_V3_SORTMULTICORE_HIF8CAST_GATHER) ||
        TILING_KEY_IS(MOE_INIT_ROUTING_V3_SORTONECORE_HIF8_PERTENSOR_QUANT_GATHER) ||
        TILING_KEY_IS(MOE_INIT_ROUTING_V3_SORTMULTICORE_HIF8_PERTENSOR_QUANT_GATHER) ||
        TILING_KEY_IS(MOE_INIT_ROUTING_V3_SORTONECORE_HIF8_PERTOKEN_QUANT_GATHER) ||
        TILING_KEY_IS(MOE_INIT_ROUTING_V3_SORTMULTICORE_HIF8_PERTOKEN_QUANT_GATHER)) {
        // GATHER索引
        TPipe rowIdxPipe;
        RowIdxGather rowIdxGatherOp;
        rowIdxGatherOp.Init(expandedRowIdx, userWS, t, &rowIdxPipe);
        rowIdxGatherOp.Process();
        rowIdxPipe.Destroy();
    }

    // 4.直接搬运或是搬运的过程中对x进行量化
    if (TILING_KEY_IS(MOE_INIT_ROUTING_V3_SORTONECORE_GATHER) ||
        TILING_KEY_IS(MOE_INIT_ROUTING_V3_SORTONECORE_SCATTER) ||
        TILING_KEY_IS(MOE_INIT_ROUTING_V3_SORTMULTICORE_GATHER) ||
        TILING_KEY_IS(MOE_INIT_ROUTING_V3_SORTMULTICORE_SCATTER)) {
        // 非量化
        TPipe gatherPipe;
        MoeGatherOut<DTYPE_X> gatherOp;
        gatherOp.Init(x, scale, userWS, expandedRowIdx, expandedX, expandedScale, t, &gatherPipe);
        gatherOp.Process();
        gatherPipe.Destroy();
    } else if (TILING_KEY_IS(MOE_INIT_ROUTING_V3_SORTONECORE_DYNAMICQUANT_GATHER) ||
               TILING_KEY_IS(MOE_INIT_ROUTING_V3_SORTONECORE_DYNAMICQUANT_SCATTER) ||
               TILING_KEY_IS(MOE_INIT_ROUTING_V3_SORTMULTICORE_DYNAMICQUANT_GATHER) ||
               TILING_KEY_IS(MOE_INIT_ROUTING_V3_SORTMULTICORE_DYNAMICQUANT_SCATTER)) {
        // 动态量化
        if constexpr (!IsSameType<DTYPE_X, int8_t>::value && !IsSameType<DTYPE_EXPANDED_X, hifloat8_t>::value) {
            TPipe gatherPipe;
            MoeGatherOutDynamicQuant<DTYPE_X> gatherDynamicQuantOp;
            gatherDynamicQuantOp.Init(x, scale, userWS, expandedRowIdx, expandedX, expandedScale, t, &gatherPipe);
            gatherDynamicQuantOp.Process();
            gatherPipe.Destroy();
        }
    } else if (TILING_KEY_IS(MOE_INIT_ROUTING_V3_SORTONECORE_MXFP8QUANT_GATHER) ||
               TILING_KEY_IS(MOE_INIT_ROUTING_V3_SORTONECORE_MXFP8QUANT_SCATTER) ||
               TILING_KEY_IS(MOE_INIT_ROUTING_V3_SORTMULTICORE_MXFP8QUANT_GATHER) ||
               TILING_KEY_IS(MOE_INIT_ROUTING_V3_SORTMULTICORE_MXFP8QUANT_SCATTER)) {
        // MXFP8量化。由于MXFP8模板用到的指令不支持DTYPE_X为int8_t等类型，因此需要constexpr-if来规避编译
        if constexpr ((IsSameType<DTYPE_X, bfloat16_t>::value || IsSameType<DTYPE_X, half>::value) &&
            (IsSameType<DTYPE_EXPANDED_X, fp8_e4m3fn_t>::value || IsSameType<DTYPE_EXPANDED_X, fp8_e5m2_t>::value)) {
            TPipe gatherPipe;
            MoeGatherOutMxfp8Quant<DTYPE_X, DTYPE_EXPANDED_X> gatherMxfp8QuantOp;
            gatherMxfp8QuantOp.Init(x, scale, userWS, expandedRowIdx, expandedX, expandedScale, t, &gatherPipe);
            gatherMxfp8QuantOp.Process();
            gatherPipe.Destroy();
        }
    } else if (TILING_KEY_IS(MOE_INIT_ROUTING_V3_SORTONECORE_HIF8CAST_GATHER) ||
        TILING_KEY_IS(MOE_INIT_ROUTING_V3_SORTONECORE_HIF8CAST_SCATTER) ||
        TILING_KEY_IS(MOE_INIT_ROUTING_V3_SORTMULTICORE_HIF8CAST_GATHER) ||
        TILING_KEY_IS(MOE_INIT_ROUTING_V3_SORTMULTICORE_HIF8CAST_SCATTER)) {
        if constexpr ((IsSameType<DTYPE_X, bfloat16_t>::value || IsSameType<DTYPE_X, half>::value) && IsSameType<DTYPE_EXPANDED_X, hifloat8_t>::value) {
            TPipe gatherPipe;
            MoeGatherOutHif8Quant<DTYPE_X> gatherHif8QuantOp;
            gatherHif8QuantOp.Init(x, userWS, expandedRowIdx, expandedX, t, &gatherPipe);
            gatherHif8QuantOp.Process();
            gatherPipe.Destroy();
        }
    } else if (TILING_KEY_IS(MOE_INIT_ROUTING_V3_SORTONECORE_HIF8_PERTENSOR_QUANT_GATHER) ||
               TILING_KEY_IS(MOE_INIT_ROUTING_V3_SORTONECORE_HIF8_PERTENSOR_QUANT_SCATTER) ||
               TILING_KEY_IS(MOE_INIT_ROUTING_V3_SORTMULTICORE_HIF8_PERTENSOR_QUANT_GATHER) ||
               TILING_KEY_IS(MOE_INIT_ROUTING_V3_SORTMULTICORE_HIF8_PERTENSOR_QUANT_SCATTER)) {
        // HIF8 PERTENSOR量化
        if constexpr ((IsSameType<DTYPE_X, bfloat16_t>::value || IsSameType<DTYPE_X, half>::value) && IsSameType<DTYPE_EXPANDED_X, hifloat8_t>::value) {
            TPipe gatherPipe;
            MoeGatherOutHif8PertensorQuant<DTYPE_X> gatherHif8PerTensorQuantOp;
            gatherHif8PerTensorQuantOp.Init(x, scale, userWS, expandedRowIdx, expandedX, t, &gatherPipe);
            gatherHif8PerTensorQuantOp.Process();
            gatherPipe.Destroy();
        }
    } else if (TILING_KEY_IS(MOE_INIT_ROUTING_V3_SORTONECORE_HIF8_PERTOKEN_QUANT_GATHER) ||
               TILING_KEY_IS(MOE_INIT_ROUTING_V3_SORTONECORE_HIF8_PERTOKEN_QUANT_SCATTER) ||
               TILING_KEY_IS(MOE_INIT_ROUTING_V3_SORTMULTICORE_HIF8_PERTOKEN_QUANT_GATHER) ||
               TILING_KEY_IS(MOE_INIT_ROUTING_V3_SORTMULTICORE_HIF8_PERTOKEN_QUANT_SCATTER)) {
        // HIF8 PERTOKENR量化
        if constexpr ((IsSameType<DTYPE_X, bfloat16_t>::value || IsSameType<DTYPE_X, half>::value) && IsSameType<DTYPE_EXPANDED_X, hifloat8_t>::value) {
            TPipe gatherPipe;
            MoeGatherOutHif8PertokenQuant<DTYPE_X> gatherHif8PerTokenQuantOp;
            gatherHif8PerTokenQuantOp.Init(x, userWS, expandedRowIdx, expandedX, expandedScale, t, &gatherPipe);
            gatherHif8PerTokenQuantOp.Process();
            gatherPipe.Destroy();
        }
    }

#if (__NPU_ARCH__ == 3510)
    SetCtrlSpr<OVERFLOW_MODE_CTRL, OVERFLOW_MODE_CTRL>(oriOverflowMode);
#endif
}
