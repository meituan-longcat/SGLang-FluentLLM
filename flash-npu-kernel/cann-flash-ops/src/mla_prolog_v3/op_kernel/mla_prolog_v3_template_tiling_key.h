/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file mla_prolog_template_tiling_key.h
 * \brief
 */

#ifndef MLA_PROLOG_TEMPLATE_TILING_KEY_H
#define MLA_PROLOG_TEMPLATE_TILING_KEY_H

#ifndef ORIG_DTYPE_TOKEN_X
#define ORIG_DTYPE_TOKEN_X (-1)
#endif

#ifndef ORIG_DTYPE_WEIGHT_UQ_QR
#define ORIG_DTYPE_WEIGHT_UQ_QR (-1)
#endif

#ifndef ORIG_DTYPE_KV_CACHE
#define ORIG_DTYPE_KV_CACHE (-1)
#endif

#ifndef ORIG_DTYPE_KR_CACHE
#define ORIG_DTYPE_KR_CACHE (-1)
#endif

#include "ascendc/host_api/tiling/template_argument.h"

#define ASCENDC_TPL_2_BW 2 // 每个参数占用2个bit位
#define ASCENDC_TPL_4_BW 4 // 每个参数占用4个bit位


// 可表示的tilingkey范围为64bit，注意不可超过限制
ASCENDC_TPL_ARGS_DECL(mla_prolog, // 算子唯一标识，与opType保持一致
                      // bit:0-3 CACHE_MODE：0-CACHE_MODE_BNSD(预留) 1-PA_BSND 2-PA_NZ 3-PA_BLK_BSND 4-PA_BLK_NZ
                      ASCENDC_TPL_UINT_DECL(CACHE_MODE, ASCENDC_TPL_4_BW, ASCENDC_TPL_UI_LIST, 0, 1, 2, 3, 4),
                      // bit:4-5 场景标识：0-FP16(预留) 1-BF16  2-量化场景
                      ASCENDC_TPL_UINT_DECL(SCENARIO, ASCENDC_TPL_2_BW, ASCENDC_TPL_UI_LIST, 0, 1, 2),
                      // bit:6-9 量化场景：0-MMQcQr量化 1-MMQcQr量化+KVcache量化 2-MMcqCkvKr量化+MMQcQr量化
                      // 3-MMCqCkvkr量化+MMQcQr量化+KVcache量化 4-MMQcQr量化+KVcache pertile量化
                      // 5-MMCqCkvkr量化+MMQcQr量化+KVcache pertile量化
                      // 9-非权重量化+KVcache pertile量化
                      ASCENDC_TPL_UINT_DECL(QUANT_MODE, ASCENDC_TPL_4_BW, ASCENDC_TPL_UI_LIST, 0, 1, 2, 3, 4, 5, 9),
                      // bit:10 反量化使能：0-关闭 1-开启
                      ASCENDC_TPL_BOOL_DECL(ENABLE_DEQUANT_OPTIONAL, 0, 1),
                      // bit:11 量化算力分组：0-关闭 1-开启
                      ASCENDC_TPL_BOOL_DECL(ENABLE_GROUP_COMPUTE_OPTIONAL, 0, 1),
                      // bit:12-13 空tensor场景：0-无空tensor 1-kv_cache/kr_cache为空 2-query为空且不更新cache
                      ASCENDC_TPL_UINT_DECL(EMPTY_TENSOR_MODE, ASCENDC_TPL_2_BW, ASCENDC_TPL_UI_LIST, 0, 1, 2),
                      // bit:15-16 actualSeqLen使能场景 0-关闭 1-使能actualSeqLen
                      ASCENDC_TPL_UINT_DECL(ACTUAL_SEQ_LEN_MODE, ASCENDC_TPL_2_BW, ASCENDC_TPL_UI_LIST, 0, 1),
                      // bit:16-17 切M模式 0-关闭(切N) 1-使能(切M)
                      ASCENDC_TPL_UINT_DECL(SPLIT_M_MODE, ASCENDC_TPL_2_BW, ASCENDC_TPL_UI_LIST, 0, 1),
                      // bit:18-25 cv分核模式 ASCENDC_TPL_MIX_AIC_1_1(6)-1:1 ASCENDC_TPL_MIX_AIC_1_2(7)-1:2
                      ASCENDC_TPL_KERNEL_TYPE_DECL(CV_MODE, ASCENDC_TPL_MIX_AIC_1_1, ASCENDC_TPL_MIX_AIC_1_2),
);

ASCENDC_TPL_SEL(
// -------------------------- 非量化场景 --------------------------
#if (((ORIG_DTYPE_TOKEN_X == -1) || (ORIG_DTYPE_WEIGHT_UQ_QR == -1)) ||                                              \
     ((ORIG_DTYPE_TOKEN_X == DT_BF16) && (ORIG_DTYPE_WEIGHT_UQ_QR == DT_BF16)))
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(CACHE_MODE, ASCENDC_TPL_UI_LIST, 0, 1, 2, 3, 4),
                         ASCENDC_TPL_UINT_SEL(SCENARIO, ASCENDC_TPL_UI_LIST, 1),
                         ASCENDC_TPL_UINT_SEL(QUANT_MODE, ASCENDC_TPL_UI_LIST, 0),
                         ASCENDC_TPL_BOOL_SEL(ENABLE_DEQUANT_OPTIONAL, 0),
                         ASCENDC_TPL_BOOL_SEL(ENABLE_GROUP_COMPUTE_OPTIONAL, 0),
                         ASCENDC_TPL_UINT_SEL(EMPTY_TENSOR_MODE, ASCENDC_TPL_UI_LIST, 0, 1),
                         ASCENDC_TPL_UINT_SEL(ACTUAL_SEQ_LEN_MODE, ASCENDC_TPL_UI_LIST, 0, 1),
                         ASCENDC_TPL_UINT_SEL(SPLIT_M_MODE, ASCENDC_TPL_UI_LIST, 0),
                         ASCENDC_TPL_SHARED_KERNEL_TYPE_SEL(CV_MODE, ASCENDC_TPL_MIX_AIC_1_1, ASCENDC_TPL_MIX_AIC_1_2),
                         ASCENDC_TPL_TILING_STRUCT_SEL(optiling::MlaPrologV3TilingData)),
#endif

// -------------------------- 半量化kv非量化 --------------------------
#if ((ORIG_DTYPE_TOKEN_X == -1) || (ORIG_DTYPE_WEIGHT_UQ_QR == -1) || (ORIG_DTYPE_KV_CACHE == -1)) ||                \
    ((ORIG_DTYPE_TOKEN_X == DT_BF16) && (ORIG_DTYPE_WEIGHT_UQ_QR == DT_INT8) && (ORIG_DTYPE_KV_CACHE == DT_BF16))
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(CACHE_MODE, ASCENDC_TPL_UI_LIST, 0, 1, 2, 3, 4),
                         ASCENDC_TPL_UINT_SEL(SCENARIO, ASCENDC_TPL_UI_LIST, 2),
                         ASCENDC_TPL_UINT_SEL(QUANT_MODE, ASCENDC_TPL_UI_LIST, 0),
                         ASCENDC_TPL_BOOL_SEL(ENABLE_DEQUANT_OPTIONAL, 0, 1),
                         ASCENDC_TPL_BOOL_SEL(ENABLE_GROUP_COMPUTE_OPTIONAL, 0, 1),
                         ASCENDC_TPL_UINT_SEL(EMPTY_TENSOR_MODE, ASCENDC_TPL_UI_LIST, 0, 1),
                         ASCENDC_TPL_UINT_SEL(ACTUAL_SEQ_LEN_MODE, ASCENDC_TPL_UI_LIST, 0, 1),
                         ASCENDC_TPL_UINT_SEL(SPLIT_M_MODE, ASCENDC_TPL_UI_LIST, 0),
                         ASCENDC_TPL_SHARED_KERNEL_TYPE_SEL(CV_MODE, ASCENDC_TPL_MIX_AIC_1_1, ASCENDC_TPL_MIX_AIC_1_2),
                         ASCENDC_TPL_TILING_STRUCT_SEL(optiling::MlaPrologV3TilingData)),
#endif

// -------------------------- 半量化kv量化 --------------------------
#if ((ORIG_DTYPE_TOKEN_X == -1) || (ORIG_DTYPE_WEIGHT_UQ_QR == -1) || (ORIG_DTYPE_KV_CACHE == -1)) ||                \
    ((ORIG_DTYPE_TOKEN_X == DT_BF16) && (ORIG_DTYPE_WEIGHT_UQ_QR == DT_INT8) && (ORIG_DTYPE_KV_CACHE == DT_INT8))
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(CACHE_MODE, ASCENDC_TPL_UI_LIST, 0, 1, 2, 3, 4),
                         ASCENDC_TPL_UINT_SEL(SCENARIO, ASCENDC_TPL_UI_LIST, 2),
                         ASCENDC_TPL_UINT_SEL(QUANT_MODE, ASCENDC_TPL_UI_LIST, 1),
                         ASCENDC_TPL_BOOL_SEL(ENABLE_DEQUANT_OPTIONAL, 0, 1),
                         ASCENDC_TPL_BOOL_SEL(ENABLE_GROUP_COMPUTE_OPTIONAL, 0, 1),
                         ASCENDC_TPL_UINT_SEL(EMPTY_TENSOR_MODE, ASCENDC_TPL_UI_LIST, 0, 1),
                         ASCENDC_TPL_UINT_SEL(ACTUAL_SEQ_LEN_MODE, ASCENDC_TPL_UI_LIST, 0, 1),
                         ASCENDC_TPL_UINT_SEL(SPLIT_M_MODE, ASCENDC_TPL_UI_LIST, 0),
                         ASCENDC_TPL_SHARED_KERNEL_TYPE_SEL(CV_MODE, ASCENDC_TPL_MIX_AIC_1_1, ASCENDC_TPL_MIX_AIC_1_2),
                         ASCENDC_TPL_TILING_STRUCT_SEL(optiling::MlaPrologV3TilingData)),

    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(CACHE_MODE, ASCENDC_TPL_UI_LIST, 0, 1),
                         ASCENDC_TPL_UINT_SEL(SCENARIO, ASCENDC_TPL_UI_LIST, 2),
                         ASCENDC_TPL_UINT_SEL(QUANT_MODE, ASCENDC_TPL_UI_LIST, 4),
                         ASCENDC_TPL_BOOL_SEL(ENABLE_DEQUANT_OPTIONAL, 0, 1),
                         ASCENDC_TPL_BOOL_SEL(ENABLE_GROUP_COMPUTE_OPTIONAL, 0),
                         ASCENDC_TPL_UINT_SEL(EMPTY_TENSOR_MODE, ASCENDC_TPL_UI_LIST, 0),
                         ASCENDC_TPL_UINT_SEL(ACTUAL_SEQ_LEN_MODE, ASCENDC_TPL_UI_LIST, 0),
                         ASCENDC_TPL_UINT_SEL(SPLIT_M_MODE, ASCENDC_TPL_UI_LIST, 0),
                         ASCENDC_TPL_SHARED_KERNEL_TYPE_SEL(CV_MODE, ASCENDC_TPL_MIX_AIC_1_1, ASCENDC_TPL_MIX_AIC_1_2),
                         ASCENDC_TPL_TILING_STRUCT_SEL(optiling::MlaPrologV3TilingData)),
#endif

// -------------------------- 全量化kv非量化 --------------------------
#if ((ORIG_DTYPE_TOKEN_X == -1) || (ORIG_DTYPE_WEIGHT_UQ_QR == -1) || (ORIG_DTYPE_KV_CACHE == -1)) ||                \
    ((ORIG_DTYPE_TOKEN_X == DT_INT8) && (ORIG_DTYPE_WEIGHT_UQ_QR == DT_INT8) && (ORIG_DTYPE_KV_CACHE == DT_BF16))
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(CACHE_MODE, ASCENDC_TPL_UI_LIST, 0, 1, 2, 3, 4),
                         ASCENDC_TPL_UINT_SEL(SCENARIO, ASCENDC_TPL_UI_LIST, 2),
                         ASCENDC_TPL_UINT_SEL(QUANT_MODE, ASCENDC_TPL_UI_LIST, 2),
                         ASCENDC_TPL_BOOL_SEL(ENABLE_DEQUANT_OPTIONAL, 0, 1),
                         ASCENDC_TPL_BOOL_SEL(ENABLE_GROUP_COMPUTE_OPTIONAL, 0),
                         ASCENDC_TPL_UINT_SEL(EMPTY_TENSOR_MODE, ASCENDC_TPL_UI_LIST, 0, 1),
                         ASCENDC_TPL_UINT_SEL(ACTUAL_SEQ_LEN_MODE, ASCENDC_TPL_UI_LIST, 0, 1),
                         ASCENDC_TPL_UINT_SEL(SPLIT_M_MODE, ASCENDC_TPL_UI_LIST, 0),
                         ASCENDC_TPL_SHARED_KERNEL_TYPE_SEL(CV_MODE, ASCENDC_TPL_MIX_AIC_1_1, ASCENDC_TPL_MIX_AIC_1_2),
                         ASCENDC_TPL_TILING_STRUCT_SEL(optiling::MlaPrologV3TilingData)),
#endif

// -------------------------- 全量化kv量化 --------------------------
#if ((ORIG_DTYPE_TOKEN_X == -1) || (ORIG_DTYPE_WEIGHT_UQ_QR == -1) || (ORIG_DTYPE_KV_CACHE == -1) ||                 \
     (ORIG_DTYPE_KR_CACHE == -1)) ||                                                                                 \
    ((ORIG_DTYPE_TOKEN_X == DT_INT8) && (ORIG_DTYPE_WEIGHT_UQ_QR == DT_INT8) && (ORIG_DTYPE_KV_CACHE == DT_INT8) &&  \
     (ORIG_DTYPE_KR_CACHE == DT_BF16))
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(CACHE_MODE, ASCENDC_TPL_UI_LIST, 0, 1, 2, 3, 4),
                         ASCENDC_TPL_UINT_SEL(SCENARIO, ASCENDC_TPL_UI_LIST, 2),
                         ASCENDC_TPL_UINT_SEL(QUANT_MODE, ASCENDC_TPL_UI_LIST, 3),
                         ASCENDC_TPL_BOOL_SEL(ENABLE_DEQUANT_OPTIONAL, 0, 1),
                         ASCENDC_TPL_BOOL_SEL(ENABLE_GROUP_COMPUTE_OPTIONAL, 0),
                         ASCENDC_TPL_UINT_SEL(EMPTY_TENSOR_MODE, ASCENDC_TPL_UI_LIST, 0, 1),
                         ASCENDC_TPL_UINT_SEL(ACTUAL_SEQ_LEN_MODE, ASCENDC_TPL_UI_LIST, 0, 1),
                         ASCENDC_TPL_UINT_SEL(SPLIT_M_MODE, ASCENDC_TPL_UI_LIST, 0),
                         ASCENDC_TPL_SHARED_KERNEL_TYPE_SEL(CV_MODE, ASCENDC_TPL_MIX_AIC_1_1, ASCENDC_TPL_MIX_AIC_1_2),
                         ASCENDC_TPL_TILING_STRUCT_SEL(optiling::MlaPrologV3TilingData)),

    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(CACHE_MODE, ASCENDC_TPL_UI_LIST, 0, 1),
                         ASCENDC_TPL_UINT_SEL(SCENARIO, ASCENDC_TPL_UI_LIST, 2),
                         ASCENDC_TPL_UINT_SEL(QUANT_MODE, ASCENDC_TPL_UI_LIST, 5),
                         ASCENDC_TPL_BOOL_SEL(ENABLE_DEQUANT_OPTIONAL, 0, 1),
                         ASCENDC_TPL_BOOL_SEL(ENABLE_GROUP_COMPUTE_OPTIONAL, 0),
                         ASCENDC_TPL_UINT_SEL(EMPTY_TENSOR_MODE, ASCENDC_TPL_UI_LIST, 0),
                         ASCENDC_TPL_UINT_SEL(ACTUAL_SEQ_LEN_MODE, ASCENDC_TPL_UI_LIST, 0),
                         ASCENDC_TPL_UINT_SEL(SPLIT_M_MODE, ASCENDC_TPL_UI_LIST, 0),
                         ASCENDC_TPL_SHARED_KERNEL_TYPE_SEL(CV_MODE, ASCENDC_TPL_MIX_AIC_1_1, ASCENDC_TPL_MIX_AIC_1_2),
                         ASCENDC_TPL_TILING_STRUCT_SEL(optiling::MlaPrologV3TilingData)),
#endif

// -------------------------- 非权重量化kv per-tile量化 --------------------------
#if ((ORIG_DTYPE_TOKEN_X == -1) || (ORIG_DTYPE_WEIGHT_UQ_QR == -1) || (ORIG_DTYPE_KV_CACHE == -1) ||                 \
     (ORIG_DTYPE_KR_CACHE == -1)) ||                                                                                 \
    ((ORIG_DTYPE_TOKEN_X == DT_BF16) && (ORIG_DTYPE_WEIGHT_UQ_QR == DT_BF16) && (ORIG_DTYPE_KV_CACHE == DT_INT8) &&  \
     (ORIG_DTYPE_KR_CACHE == DT_BF16))
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(CACHE_MODE, ASCENDC_TPL_UI_LIST, 1),
                         ASCENDC_TPL_UINT_SEL(SCENARIO, ASCENDC_TPL_UI_LIST, 2),
                         ASCENDC_TPL_UINT_SEL(QUANT_MODE, ASCENDC_TPL_UI_LIST, 9),
                         ASCENDC_TPL_BOOL_SEL(ENABLE_DEQUANT_OPTIONAL, 0),
                         ASCENDC_TPL_BOOL_SEL(ENABLE_GROUP_COMPUTE_OPTIONAL, 0),
                         ASCENDC_TPL_UINT_SEL(EMPTY_TENSOR_MODE, ASCENDC_TPL_UI_LIST, 0, 1),
                         ASCENDC_TPL_UINT_SEL(ACTUAL_SEQ_LEN_MODE, ASCENDC_TPL_UI_LIST, 0),
                         ASCENDC_TPL_UINT_SEL(SPLIT_M_MODE, ASCENDC_TPL_UI_LIST, 0),
                         ASCENDC_TPL_SHARED_KERNEL_TYPE_SEL(CV_MODE, ASCENDC_TPL_MIX_AIC_1_1, ASCENDC_TPL_MIX_AIC_1_2),
                         ASCENDC_TPL_TILING_STRUCT_SEL(optiling::MlaPrologV3TilingData)),

    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(CACHE_MODE, ASCENDC_TPL_UI_LIST, 0),
                         ASCENDC_TPL_UINT_SEL(SCENARIO, ASCENDC_TPL_UI_LIST, 2),
                         ASCENDC_TPL_UINT_SEL(QUANT_MODE, ASCENDC_TPL_UI_LIST, 9),
                         ASCENDC_TPL_BOOL_SEL(ENABLE_DEQUANT_OPTIONAL, 0),
                         ASCENDC_TPL_BOOL_SEL(ENABLE_GROUP_COMPUTE_OPTIONAL, 0),
                         ASCENDC_TPL_UINT_SEL(EMPTY_TENSOR_MODE, ASCENDC_TPL_UI_LIST, 0),
                         ASCENDC_TPL_UINT_SEL(ACTUAL_SEQ_LEN_MODE, ASCENDC_TPL_UI_LIST, 0),
                         ASCENDC_TPL_UINT_SEL(SPLIT_M_MODE, ASCENDC_TPL_UI_LIST, 0),
                         ASCENDC_TPL_SHARED_KERNEL_TYPE_SEL(CV_MODE, ASCENDC_TPL_MIX_AIC_1_1, ASCENDC_TPL_MIX_AIC_1_2),
                         ASCENDC_TPL_TILING_STRUCT_SEL(optiling::MlaPrologV3TilingData)),
#endif

    // -------------------------- 空tensor场景 --------------------------
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(CACHE_MODE, ASCENDC_TPL_UI_LIST, 0),
                         ASCENDC_TPL_UINT_SEL(SCENARIO, ASCENDC_TPL_UI_LIST, 0),
                         ASCENDC_TPL_UINT_SEL(QUANT_MODE, ASCENDC_TPL_UI_LIST, 0),
                         ASCENDC_TPL_BOOL_SEL(ENABLE_DEQUANT_OPTIONAL, 0),
                         ASCENDC_TPL_BOOL_SEL(ENABLE_GROUP_COMPUTE_OPTIONAL, 0),
                         ASCENDC_TPL_UINT_SEL(EMPTY_TENSOR_MODE, ASCENDC_TPL_UI_LIST, 2),
                         ASCENDC_TPL_UINT_SEL(ACTUAL_SEQ_LEN_MODE, ASCENDC_TPL_UI_LIST, 0),
                         ASCENDC_TPL_UINT_SEL(SPLIT_M_MODE, ASCENDC_TPL_UI_LIST, 0),
                         ASCENDC_TPL_SHARED_KERNEL_TYPE_SEL(CV_MODE, ASCENDC_TPL_MIX_AIC_1_1, ASCENDC_TPL_MIX_AIC_1_2),
                         ASCENDC_TPL_TILING_STRUCT_SEL(optiling::MlaPrologV3TilingData)),
);

#endif // MLA_PROLOG_TEMPLATE_TILING_KEY_H
