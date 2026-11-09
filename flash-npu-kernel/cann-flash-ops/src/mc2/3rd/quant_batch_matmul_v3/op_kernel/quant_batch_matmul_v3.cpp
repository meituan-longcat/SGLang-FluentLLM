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
 * \file quant_batch_matmul_v3.cpp
 * \brief
 */

#include "quant_batch_matmul_v3.h"
#include "quant_batch_matmul_v3_init_output.h"
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
#include "quant_batch_matmul_v3_cube_basic.h"
#if (ORIG_DTYPE_Y == DT_BF16 || ORIG_DTYPE_SCALE == DT_FLOAT)
#include "quant_batch_matmul_v3_bf16_basic.h"
#include "quant_batch_matmul_v3_bf16.h"
#include "quant_batch_matmul_v3_bf16_opt.h"
#include "quant_batch_matmul_v3_pertoken.h"
#include "quant_batch_matmul_v3_pertoken_basic.h"
#include "quant_batch_matmul_v3_pertoken_opt.h"
#endif
#endif
#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "quant_batch_matmul_v3_tiling_key.h"
#include "quant_batch_matmul_v3_tiling_data.h"

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

#define INVOKE_QUANT_BATCH_MATMUL_V3_CUBE_IMPL(transposeX1, transposeX2)                                          \
    do {                                                                                                          \
        const Mc2QuantBatchMatmulV3TilingData *qBmmV3TilingData = &tilingData;                                       \
        Mc2QuantBatchMatmulV3BaseKernel<DTYPE_X1, DTYPE_X2, DTYPE_SCALE, DTYPE_Y, FORMAT_X1, FORMAT_X2, transposeX1, \
                                     transposeX2, Mc2QuantBatchMatmulV3Update> op;                                   \
        op.Init(x1, x2, scale, bias, y, user1, qBmmV3TilingData, &tPipe);                                         \
        op.Process();                                                                                             \
    } while (0)

#define INVOKE_QUANT_BATCH_MATMUL_DEQUANT_BF16_IMPL(transposeX1, transposeX2)                             \
    do {                                                                                                  \
        const Mc2QuantBatchMatmulV3TilingData *qBmmV3TilingData = &tilingData;                               \
        const TCubeTiling *mmTiling = &(qBmmV3TilingData->matmulTiling);                                  \
        BmmDequantBf16<DTYPE_X1, DTYPE_X2, FORMAT_X1, FORMAT_X2, DTYPE_SCALE, DTYPE_Y, transposeX1, transposeX2> op; \
        REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), op.mm, mmTiling);                                 \
        op.Init(x1, x2, bias, scale, y, user1, qBmmV3TilingData, &tPipe);                                 \
        op.Process();                                                                                     \
        tPipe.Destroy();                                                                                  \
    } while (0)

#define INVOKE_QUANT_BATCH_MATMUL_DEQUANT_BF16_OPT_IMPL(transposeX1, transposeX2)                            \
    do {                                                                                                     \
        const Mc2QuantBatchMatmulV3TilingData *qBmmV3TilingData = &tilingData;                                  \
        BmmDequantBf16Opt<DTYPE_X1, DTYPE_X2, FORMAT_X1, FORMAT_X2, DTYPE_SCALE, DTYPE_Y, transposeX1, transposeX2> op; \
        op.Init(x1, x2, bias, scale, y, user1, qBmmV3TilingData, &tPipe);                                    \
        op.Process();                                                                                        \
        tPipe.Destroy();                                                                                     \
    } while (0)

#define INVOKE_QUANT_BATCH_MATMUL_DEQUANT_PERTOKEN_IMPL(transposeX1, transposeX2)                             \
    do {                                                                                                      \
        const Mc2QuantBatchMatmulV3TilingData *qBmmV3TilingData = &tilingData;                                   \
        const TCubeTiling *mmTiling = &(qBmmV3TilingData->matmulTiling);                                      \
        BmmDequantPertoken<DTYPE_X1, DTYPE_X2, FORMAT_X1, FORMAT_X2, DTYPE_SCALE, DTYPE_Y, transposeX1, transposeX2> op; \
        REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), op.mm, mmTiling);                                     \
        op.Init(x1, x2, bias, scale, pertokenScale, y, user1, qBmmV3TilingData, &tPipe);                      \
        op.Process();                                                                                         \
        tPipe.Destroy();                                                                                      \
    } while (0)

#define INVOKE_QUANT_BATCH_MATMUL_DEQUANT_PERTOKEN_OPT_IMPL(transposeX1, transposeX2)                            \
    do {                                                                                                         \
        const Mc2QuantBatchMatmulV3TilingData *qBmmV3TilingData = &tilingData;                                      \
        BmmDequantPertokenOpt<DTYPE_X1, DTYPE_X2, FORMAT_X1, FORMAT_X2, DTYPE_SCALE, DTYPE_Y, transposeX1, transposeX2> op; \
        op.Init(x1, x2, bias, scale, pertokenScale, y, user1, qBmmV3TilingData, &tPipe);                         \
        op.Process();                                                                                            \
        tPipe.Destroy();                                                                                         \
    } while (0)

#define INVOKE_QUANT_BATCH_MATMUL_DEQUANT_PERTOKEN_BASIC_IMPL(transposeX1, transposeX2)                      \
    do {                                                                                                     \
        const Mc2QuantBatchMatmulV3TilingData *qBmmV3TilingData = &tilingData;                                  \
        BmmDequantPertokenBasic<DTYPE_X1, DTYPE_X2, DTYPE_SCALE, DTYPE_Y, FORMAT_X1, FORMAT_X2, transposeX1, \
                                transposeX2, Mc2QuantBatchMatmulV3Update> op;                                   \
        op.Init(x1, x2, scale, bias, pertokenScale, y, user1, qBmmV3TilingData, &tPipe);                     \
        op.Process();                                                                                        \
        tPipe.Destroy();                                                                                     \
    } while (0)

#define INVOKE_QUANT_BATCH_MATMUL_DEQUANT_BASIC_BLOCK_IMPL(transposeX1, transposeX2)                                  \
    do {                                                                                                              \
        const Mc2QuantBatchMatmulV3TilingData *qBmmV3TilingData = &tilingData;                                           \
        BmmBasicDequantBf16<DTYPE_X1, DTYPE_X2, DTYPE_SCALE, DTYPE_Y, FORMAT_X1, FORMAT_X2, transposeX1, transposeX2, \
                            Mc2QuantBatchMatmulV3Update>  op;                                                            \
        op.Init(x1, x2, scale, bias, y, user1, qBmmV3TilingData, &tPipe);                                             \
        op.Process();                                                                                                 \
        tPipe.Destroy();                                                                                              \
    } while (0)

#define INVOKE_QUANT_BATCH_MATMUL_DEQUANT_SPLITK_IMPL(transposeX1, transposeX2)                         \
    do {                                                                                                \
        BmmDequantInitOutput<DTYPE_Y> clearOp;                                                          \
        clearOp.Init(y, user1, &tilingData, &tPipe);                                                    \
        clearOp.Process();                                                                              \
        tPipe.Destroy();                                                                                \
        TPipe tPipeOp;                                                                                  \
        BmmDequant<DTYPE_X1, DTYPE_X2, FORMAT_X1, FORMAT_X2, int32_t, uint64_t, DTYPE_Y, transposeX1, transposeX2, \
                   BMM_DEQUANT_PRELOAD_CFG> op;                                                         \
        op.Init(x1, x2, bias, scale, y, user1, &tilingData, &tPipeOp);                                  \
        op.Process(true);                                                                               \
    } while (0)

template <int TRANS, int KERNEL_TEMPLATE_TYPE, int PERTOKEN, int OPTIONATTR>
__global__ __aicore__ void quant_batch_matmul_v3(GM_ADDR x1, GM_ADDR x2, GM_ADDR scale, GM_ADDR offset,
                                                            GM_ADDR bias, GM_ADDR pertokenScale, GM_ADDR y,
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
    REGISTER_TILING_DEFAULT(Mc2QuantBatchMatmulV3TilingData);
    GET_TILING_DATA(tilingData, tiling);

// 6bit from hight to low: needClean, pertoken, opt, basic, transX1, transX2
#if (ORIG_DTYPE_Y == DT_FLOAT16 || ORIG_DTYPE_Y == DT_INT8 || ORIG_DTYPE_Y == DT_INT32)  // fp16, int8, int32
#if (ORIG_DTYPE_SCALE != DT_FLOAT || ORIG_DTYPE_Y == DT_INT32)
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 200
    if (TRANS == QUANT_BATCH_MATMUL_V3_B_TRANS && KERNEL_TEMPLATE_TYPE == QUANT_BATCH_MATMUl_V3_KERNEL_TEMPLATE_TYPE_TBE &&
        PERTOKEN == QUANT_BATCH_MATMUL_V3_NOT_PERTOKEN && OPTIONATTR == QUANT_BATCH_MATMUL_V3_OPTION_ATTR_NONE) {  // false true
        BmmDequant<DTYPE_X1, DTYPE_X2, FORMAT_X1, FORMAT_X2, int32_t, uint64_t, DTYPE_Y, false, true> op;
        op.Init(x1, x2, bias, scale, y, user1, &tilingData, &tPipe);
        op.Process();
    }
    if (TRANS == QUANT_BATCH_MATMUL_V3_B_TRANS && KERNEL_TEMPLATE_TYPE == QUANT_BATCH_MATMUl_V3_KERNEL_TEMPLATE_TYPE_TBE &&
        PERTOKEN == QUANT_BATCH_MATMUL_V3_NOT_PERTOKEN && OPTIONATTR == QUANT_BATCH_MATMUL_V3_NEED_ATOMICLEAN) {  // false true
        BmmDequantInitOutput<DTYPE_Y> clearOp;
        clearOp.Init(y, user1, &tilingData, &tPipe);
        clearOp.Process();
        tPipe.Destroy();

        TPipe tPipeOp;
        BmmDequant<DTYPE_X1, DTYPE_X2, FORMAT_X1, FORMAT_X2, int32_t, uint64_t, DTYPE_Y, false, true> op;
        op.Init(x1, x2, bias, scale, y, user1, &tilingData, &tPipeOp);
        op.Process();
    }
#endif
#endif
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
    if (TRANS == QUANT_BATCH_MATMUL_V3_B_TRANS && KERNEL_TEMPLATE_TYPE == QUANT_BATCH_MATMUl_V3_KERNEL_TEMPLATE_TYPE_TBE &&
        PERTOKEN == QUANT_BATCH_MATMUL_V3_NOT_PERTOKEN && OPTIONATTR == QUANT_BATCH_MATMUL_V3_OPTION_ATTR_NONE) {  // false true
        BmmDequant<DTYPE_X1, DTYPE_X2, FORMAT_X1, FORMAT_X2, int32_t, uint64_t, DTYPE_Y, false, true> op;
        op.Init(x1, x2, bias, scale, y, user1, &tilingData, &tPipe);
        op.Process();
    }
#if (ORIG_DTYPE_SCALE != DT_FLOAT || ORIG_DTYPE_Y == DT_INT32)
    if (TRANS == QUANT_BATCH_MATMUL_V3_NOT_TRANS && KERNEL_TEMPLATE_TYPE == QUANT_BATCH_MATMUl_V3_KERNEL_TEMPLATE_TYPE_TBE &&
        PERTOKEN == QUANT_BATCH_MATMUL_V3_NOT_PERTOKEN && OPTIONATTR == QUANT_BATCH_MATMUL_V3_OPTION_ATTR_NONE) {  // false false
        BmmDequant<DTYPE_X1, DTYPE_X2, FORMAT_X1, FORMAT_X2, int32_t, uint64_t, DTYPE_Y, false, false> op;
        op.Init(x1, x2, bias, scale, y, user1, &tilingData, &tPipe);
        op.Process();
    } else if (
        TRANS == QUANT_BATCH_MATMUL_V3_A_TRANS && KERNEL_TEMPLATE_TYPE == QUANT_BATCH_MATMUl_V3_KERNEL_TEMPLATE_TYPE_TBE &&
        PERTOKEN == QUANT_BATCH_MATMUL_V3_NOT_PERTOKEN && OPTIONATTR == QUANT_BATCH_MATMUL_V3_OPTION_ATTR_NONE) {  // true false
        BmmDequant<DTYPE_X1, DTYPE_X2, FORMAT_X1, FORMAT_X2, int32_t, uint64_t, DTYPE_Y, true, false> op;
        op.Init(x1, x2, bias, scale, y, user1, &tilingData, &tPipe);
        op.Process();
    } else if (
        TRANS == QUANT_BATCH_MATMUL_V3_ALL_TRANS && KERNEL_TEMPLATE_TYPE == QUANT_BATCH_MATMUl_V3_KERNEL_TEMPLATE_TYPE_TBE &&
        PERTOKEN == QUANT_BATCH_MATMUL_V3_NOT_PERTOKEN && OPTIONATTR == QUANT_BATCH_MATMUL_V3_OPTION_ATTR_NONE) {  // true true
        BmmDequant<DTYPE_X1, DTYPE_X2, FORMAT_X1, FORMAT_X2, int32_t, uint64_t, DTYPE_Y, true, true> op;
        op.Init(x1, x2, bias, scale, y, user1, &tilingData, &tPipe);
        op.Process();
#if ORIG_DTYPE_Y == DT_INT32
    } else if (
        TRANS == QUANT_BATCH_MATMUL_V3_NOT_TRANS && KERNEL_TEMPLATE_TYPE == QUANT_BATCH_MATMUl_V3_KERNEL_TEMPLATE_TYPE_TBE &&
        PERTOKEN == QUANT_BATCH_MATMUL_V3_NOT_PERTOKEN && OPTIONATTR == QUANT_BATCH_MATMUL_V3_NEED_ATOMICLEAN) {
        INVOKE_QUANT_BATCH_MATMUL_DEQUANT_SPLITK_IMPL(false, false);
    } else if (
        TRANS == QUANT_BATCH_MATMUL_V3_B_TRANS && KERNEL_TEMPLATE_TYPE == QUANT_BATCH_MATMUl_V3_KERNEL_TEMPLATE_TYPE_TBE &&
        PERTOKEN == QUANT_BATCH_MATMUL_V3_NOT_PERTOKEN && OPTIONATTR == QUANT_BATCH_MATMUL_V3_NEED_ATOMICLEAN) {
        INVOKE_QUANT_BATCH_MATMUL_DEQUANT_SPLITK_IMPL(false, true);
    } else if (
        TRANS == QUANT_BATCH_MATMUL_V3_A_TRANS && KERNEL_TEMPLATE_TYPE == QUANT_BATCH_MATMUl_V3_KERNEL_TEMPLATE_TYPE_TBE &&
        PERTOKEN == QUANT_BATCH_MATMUL_V3_NOT_PERTOKEN && OPTIONATTR == QUANT_BATCH_MATMUL_V3_NEED_ATOMICLEAN) {
        INVOKE_QUANT_BATCH_MATMUL_DEQUANT_SPLITK_IMPL(true, false);
    } else if (
        TRANS == QUANT_BATCH_MATMUL_V3_ALL_TRANS && KERNEL_TEMPLATE_TYPE == QUANT_BATCH_MATMUl_V3_KERNEL_TEMPLATE_TYPE_TBE &&
        PERTOKEN == QUANT_BATCH_MATMUL_V3_NOT_PERTOKEN && OPTIONATTR == QUANT_BATCH_MATMUL_V3_NEED_ATOMICLEAN) {
        INVOKE_QUANT_BATCH_MATMUL_DEQUANT_SPLITK_IMPL(true, true);
#endif
    } else {  // matmul with basic tiling
        if (TRANS == QUANT_BATCH_MATMUL_V3_NOT_TRANS && KERNEL_TEMPLATE_TYPE == QUANT_BATCH_MATMUl_V3_KERNEL_TEMPLATE_TYPE_BASIC &&
            PERTOKEN == QUANT_BATCH_MATMUL_V3_NOT_PERTOKEN && OPTIONATTR == QUANT_BATCH_MATMUL_V3_OPTION_ATTR_NONE) {
            INVOKE_QUANT_BATCH_MATMUL_V3_CUBE_IMPL(false, false);
        } else if (
            TRANS == QUANT_BATCH_MATMUL_V3_B_TRANS && KERNEL_TEMPLATE_TYPE == QUANT_BATCH_MATMUl_V3_KERNEL_TEMPLATE_TYPE_BASIC &&
            PERTOKEN == QUANT_BATCH_MATMUL_V3_NOT_PERTOKEN && OPTIONATTR == QUANT_BATCH_MATMUL_V3_OPTION_ATTR_NONE) {
            INVOKE_QUANT_BATCH_MATMUL_V3_CUBE_IMPL(false, true);
        } else if (
            TRANS == QUANT_BATCH_MATMUL_V3_A_TRANS && KERNEL_TEMPLATE_TYPE == QUANT_BATCH_MATMUl_V3_KERNEL_TEMPLATE_TYPE_BASIC &&
            PERTOKEN == QUANT_BATCH_MATMUL_V3_NOT_PERTOKEN && OPTIONATTR == QUANT_BATCH_MATMUL_V3_OPTION_ATTR_NONE) {
            INVOKE_QUANT_BATCH_MATMUL_V3_CUBE_IMPL(true, false);
        } else if (
            TRANS == QUANT_BATCH_MATMUL_V3_ALL_TRANS && KERNEL_TEMPLATE_TYPE == QUANT_BATCH_MATMUl_V3_KERNEL_TEMPLATE_TYPE_BASIC &&
            PERTOKEN == QUANT_BATCH_MATMUL_V3_NOT_PERTOKEN && OPTIONATTR == QUANT_BATCH_MATMUL_V3_OPTION_ATTR_NONE) {
            INVOKE_QUANT_BATCH_MATMUL_V3_CUBE_IMPL(true, true);
        }
    }
#else
    if (TRANS == QUANT_BATCH_MATMUL_V3_NOT_TRANS && KERNEL_TEMPLATE_TYPE == QUANT_BATCH_MATMUl_V3_KERNEL_TEMPLATE_TYPE_TBE &&
        PERTOKEN == QUANT_BATCH_MATMUL_V3_IS_PERTOKEN && OPTIONATTR == QUANT_BATCH_MATMUL_V3_OPTION_ATTR_NONE) {  // false false
        INVOKE_QUANT_BATCH_MATMUL_DEQUANT_PERTOKEN_IMPL(false, false);
    } else if (
        TRANS == QUANT_BATCH_MATMUL_V3_B_TRANS && KERNEL_TEMPLATE_TYPE == QUANT_BATCH_MATMUl_V3_KERNEL_TEMPLATE_TYPE_TBE &&
        PERTOKEN == QUANT_BATCH_MATMUL_V3_IS_PERTOKEN && OPTIONATTR == QUANT_BATCH_MATMUL_V3_OPTION_ATTR_NONE) {  // false true
        INVOKE_QUANT_BATCH_MATMUL_DEQUANT_PERTOKEN_IMPL(false, true);
    } else if (
        TRANS == QUANT_BATCH_MATMUL_V3_A_TRANS && KERNEL_TEMPLATE_TYPE == QUANT_BATCH_MATMUl_V3_KERNEL_TEMPLATE_TYPE_TBE &&
        PERTOKEN == QUANT_BATCH_MATMUL_V3_IS_PERTOKEN && OPTIONATTR == QUANT_BATCH_MATMUL_V3_OPTION_ATTR_NONE) {  // true false
        INVOKE_QUANT_BATCH_MATMUL_DEQUANT_PERTOKEN_IMPL(true, false);
    } else if (
        TRANS == QUANT_BATCH_MATMUL_V3_ALL_TRANS && KERNEL_TEMPLATE_TYPE == QUANT_BATCH_MATMUl_V3_KERNEL_TEMPLATE_TYPE_TBE &&
        PERTOKEN == QUANT_BATCH_MATMUL_V3_IS_PERTOKEN && OPTIONATTR == QUANT_BATCH_MATMUL_V3_OPTION_ATTR_NONE) {  // true true
        INVOKE_QUANT_BATCH_MATMUL_DEQUANT_PERTOKEN_IMPL(true, true);
    } else if (
        TRANS == QUANT_BATCH_MATMUL_V3_NOT_TRANS && KERNEL_TEMPLATE_TYPE == QUANT_BATCH_MATMUl_V3_KERNEL_TEMPLATE_TYPE_OPT &&
        PERTOKEN == QUANT_BATCH_MATMUL_V3_IS_PERTOKEN && OPTIONATTR == QUANT_BATCH_MATMUL_V3_OPTION_ATTR_NONE) {  // false false
        INVOKE_QUANT_BATCH_MATMUL_DEQUANT_PERTOKEN_OPT_IMPL(false, false);
    } else if (
        TRANS == QUANT_BATCH_MATMUL_V3_B_TRANS && KERNEL_TEMPLATE_TYPE == QUANT_BATCH_MATMUl_V3_KERNEL_TEMPLATE_TYPE_OPT &&
        PERTOKEN == QUANT_BATCH_MATMUL_V3_IS_PERTOKEN && OPTIONATTR == QUANT_BATCH_MATMUL_V3_OPTION_ATTR_NONE) {  // false true
        INVOKE_QUANT_BATCH_MATMUL_DEQUANT_PERTOKEN_OPT_IMPL(false, true);
    } else if (
        TRANS == QUANT_BATCH_MATMUL_V3_A_TRANS && KERNEL_TEMPLATE_TYPE == QUANT_BATCH_MATMUl_V3_KERNEL_TEMPLATE_TYPE_OPT &&
        PERTOKEN == QUANT_BATCH_MATMUL_V3_IS_PERTOKEN && OPTIONATTR == QUANT_BATCH_MATMUL_V3_OPTION_ATTR_NONE) {  // true false
        INVOKE_QUANT_BATCH_MATMUL_DEQUANT_PERTOKEN_OPT_IMPL(true, false);
    } else if (
        TRANS == QUANT_BATCH_MATMUL_V3_ALL_TRANS && KERNEL_TEMPLATE_TYPE == QUANT_BATCH_MATMUl_V3_KERNEL_TEMPLATE_TYPE_OPT &&
        PERTOKEN == QUANT_BATCH_MATMUL_V3_IS_PERTOKEN && OPTIONATTR == QUANT_BATCH_MATMUL_V3_OPTION_ATTR_NONE) {  // true true
        INVOKE_QUANT_BATCH_MATMUL_DEQUANT_PERTOKEN_OPT_IMPL(true, true);
    } else if (
        TRANS == QUANT_BATCH_MATMUL_V3_NOT_TRANS && KERNEL_TEMPLATE_TYPE == QUANT_BATCH_MATMUl_V3_KERNEL_TEMPLATE_TYPE_BASIC &&
        PERTOKEN == QUANT_BATCH_MATMUL_V3_IS_PERTOKEN && OPTIONATTR == QUANT_BATCH_MATMUL_V3_OPTION_ATTR_NONE) {  // false false
        INVOKE_QUANT_BATCH_MATMUL_DEQUANT_PERTOKEN_BASIC_IMPL(false, false);
    } else if (
        TRANS == QUANT_BATCH_MATMUL_V3_B_TRANS && KERNEL_TEMPLATE_TYPE == QUANT_BATCH_MATMUl_V3_KERNEL_TEMPLATE_TYPE_BASIC &&
        PERTOKEN == QUANT_BATCH_MATMUL_V3_IS_PERTOKEN && OPTIONATTR == QUANT_BATCH_MATMUL_V3_OPTION_ATTR_NONE) {  // false true
        INVOKE_QUANT_BATCH_MATMUL_DEQUANT_PERTOKEN_BASIC_IMPL(false, true);
    } else if (
        TRANS == QUANT_BATCH_MATMUL_V3_A_TRANS && KERNEL_TEMPLATE_TYPE == QUANT_BATCH_MATMUl_V3_KERNEL_TEMPLATE_TYPE_BASIC &&
        PERTOKEN == QUANT_BATCH_MATMUL_V3_IS_PERTOKEN && OPTIONATTR == QUANT_BATCH_MATMUL_V3_OPTION_ATTR_NONE) {  // true false
        INVOKE_QUANT_BATCH_MATMUL_DEQUANT_PERTOKEN_BASIC_IMPL(true, false);
    } else if (
        TRANS == QUANT_BATCH_MATMUL_V3_ALL_TRANS && KERNEL_TEMPLATE_TYPE == QUANT_BATCH_MATMUl_V3_KERNEL_TEMPLATE_TYPE_BASIC &&
        PERTOKEN == QUANT_BATCH_MATMUL_V3_IS_PERTOKEN && OPTIONATTR == QUANT_BATCH_MATMUL_V3_OPTION_ATTR_NONE) {  // true true
        INVOKE_QUANT_BATCH_MATMUL_DEQUANT_PERTOKEN_BASIC_IMPL(true, true);
    }

#endif
#endif
#else  //  bf16
    if (TRANS == QUANT_BATCH_MATMUL_V3_NOT_TRANS && KERNEL_TEMPLATE_TYPE == QUANT_BATCH_MATMUl_V3_KERNEL_TEMPLATE_TYPE_BASIC &&
        PERTOKEN == QUANT_BATCH_MATMUL_V3_IS_PERTOKEN && OPTIONATTR == QUANT_BATCH_MATMUL_V3_IS_AICAIV_1_2) {
        INVOKE_QUANT_BATCH_MATMUL_DEQUANT_PERTOKEN_BASIC_IMPL(false, false);
    } else {
        if (TRANS == QUANT_BATCH_MATMUL_V3_NOT_TRANS && KERNEL_TEMPLATE_TYPE == QUANT_BATCH_MATMUl_V3_KERNEL_TEMPLATE_TYPE_TBE &&
            PERTOKEN == QUANT_BATCH_MATMUL_V3_NOT_PERTOKEN && OPTIONATTR == QUANT_BATCH_MATMUL_V3_OPTION_ATTR_NONE) {  // false false
            INVOKE_QUANT_BATCH_MATMUL_DEQUANT_BF16_IMPL(false, false);
        } else if (
            TRANS == QUANT_BATCH_MATMUL_V3_B_TRANS && KERNEL_TEMPLATE_TYPE == QUANT_BATCH_MATMUl_V3_KERNEL_TEMPLATE_TYPE_TBE &&
            PERTOKEN == QUANT_BATCH_MATMUL_V3_NOT_PERTOKEN && OPTIONATTR == QUANT_BATCH_MATMUL_V3_OPTION_ATTR_NONE) {  // false true
            INVOKE_QUANT_BATCH_MATMUL_DEQUANT_BF16_IMPL(false, true);
        } else if (
            TRANS == QUANT_BATCH_MATMUL_V3_A_TRANS && KERNEL_TEMPLATE_TYPE == QUANT_BATCH_MATMUl_V3_KERNEL_TEMPLATE_TYPE_TBE &&
            PERTOKEN == QUANT_BATCH_MATMUL_V3_NOT_PERTOKEN && OPTIONATTR == QUANT_BATCH_MATMUL_V3_OPTION_ATTR_NONE) {  // true  false
            INVOKE_QUANT_BATCH_MATMUL_DEQUANT_BF16_IMPL(true, false);
        } else if (
            TRANS == QUANT_BATCH_MATMUL_V3_ALL_TRANS && KERNEL_TEMPLATE_TYPE == QUANT_BATCH_MATMUl_V3_KERNEL_TEMPLATE_TYPE_TBE &&
            PERTOKEN == QUANT_BATCH_MATMUL_V3_NOT_PERTOKEN && OPTIONATTR == QUANT_BATCH_MATMUL_V3_OPTION_ATTR_NONE) {  // true  true
            INVOKE_QUANT_BATCH_MATMUL_DEQUANT_BF16_IMPL(true, true);
        } else if (
            TRANS == QUANT_BATCH_MATMUL_V3_NOT_TRANS && KERNEL_TEMPLATE_TYPE == QUANT_BATCH_MATMUl_V3_KERNEL_TEMPLATE_TYPE_OPT &&
            PERTOKEN == QUANT_BATCH_MATMUL_V3_NOT_PERTOKEN && OPTIONATTR == QUANT_BATCH_MATMUL_V3_OPTION_ATTR_NONE) {  // false false
            INVOKE_QUANT_BATCH_MATMUL_DEQUANT_BF16_OPT_IMPL(false, false);
        } else if (
            TRANS == QUANT_BATCH_MATMUL_V3_B_TRANS && KERNEL_TEMPLATE_TYPE == QUANT_BATCH_MATMUl_V3_KERNEL_TEMPLATE_TYPE_OPT &&
            PERTOKEN == QUANT_BATCH_MATMUL_V3_NOT_PERTOKEN && OPTIONATTR == QUANT_BATCH_MATMUL_V3_OPTION_ATTR_NONE) {  // false true
            INVOKE_QUANT_BATCH_MATMUL_DEQUANT_BF16_OPT_IMPL(false, true);
        } else if (
            TRANS == QUANT_BATCH_MATMUL_V3_A_TRANS && KERNEL_TEMPLATE_TYPE == QUANT_BATCH_MATMUl_V3_KERNEL_TEMPLATE_TYPE_OPT &&
            PERTOKEN == QUANT_BATCH_MATMUL_V3_NOT_PERTOKEN && OPTIONATTR == QUANT_BATCH_MATMUL_V3_OPTION_ATTR_NONE) {  // true  false
            INVOKE_QUANT_BATCH_MATMUL_DEQUANT_BF16_OPT_IMPL(true, false);
        } else if (
            TRANS == QUANT_BATCH_MATMUL_V3_ALL_TRANS && KERNEL_TEMPLATE_TYPE == QUANT_BATCH_MATMUl_V3_KERNEL_TEMPLATE_TYPE_OPT &&
            PERTOKEN == QUANT_BATCH_MATMUL_V3_NOT_PERTOKEN && OPTIONATTR == QUANT_BATCH_MATMUL_V3_OPTION_ATTR_NONE) {  // true  true
            INVOKE_QUANT_BATCH_MATMUL_DEQUANT_BF16_OPT_IMPL(true, true);
        } else if (
            TRANS == QUANT_BATCH_MATMUL_V3_NOT_TRANS && KERNEL_TEMPLATE_TYPE == QUANT_BATCH_MATMUl_V3_KERNEL_TEMPLATE_TYPE_TBE &&
            PERTOKEN == QUANT_BATCH_MATMUL_V3_IS_PERTOKEN && OPTIONATTR == QUANT_BATCH_MATMUL_V3_OPTION_ATTR_NONE) {  // false false pertoken begin
            INVOKE_QUANT_BATCH_MATMUL_DEQUANT_PERTOKEN_IMPL(false, false);
        } else if (
            TRANS == QUANT_BATCH_MATMUL_V3_B_TRANS && KERNEL_TEMPLATE_TYPE == QUANT_BATCH_MATMUl_V3_KERNEL_TEMPLATE_TYPE_TBE &&
            PERTOKEN == QUANT_BATCH_MATMUL_V3_IS_PERTOKEN && OPTIONATTR == QUANT_BATCH_MATMUL_V3_OPTION_ATTR_NONE) {  // false true
            INVOKE_QUANT_BATCH_MATMUL_DEQUANT_PERTOKEN_IMPL(false, true);
        } else if (
            TRANS == QUANT_BATCH_MATMUL_V3_A_TRANS && KERNEL_TEMPLATE_TYPE == QUANT_BATCH_MATMUl_V3_KERNEL_TEMPLATE_TYPE_TBE &&
            PERTOKEN == QUANT_BATCH_MATMUL_V3_IS_PERTOKEN && OPTIONATTR == QUANT_BATCH_MATMUL_V3_OPTION_ATTR_NONE) {  // true  false
            INVOKE_QUANT_BATCH_MATMUL_DEQUANT_PERTOKEN_IMPL(true, false);
        } else if (
            TRANS == QUANT_BATCH_MATMUL_V3_ALL_TRANS && KERNEL_TEMPLATE_TYPE == QUANT_BATCH_MATMUl_V3_KERNEL_TEMPLATE_TYPE_TBE &&
            PERTOKEN == QUANT_BATCH_MATMUL_V3_IS_PERTOKEN && OPTIONATTR == QUANT_BATCH_MATMUL_V3_OPTION_ATTR_NONE) {  // true  true
            INVOKE_QUANT_BATCH_MATMUL_DEQUANT_PERTOKEN_IMPL(true, true);
        } else if (
            TRANS == QUANT_BATCH_MATMUL_V3_NOT_TRANS && KERNEL_TEMPLATE_TYPE == QUANT_BATCH_MATMUl_V3_KERNEL_TEMPLATE_TYPE_OPT &&
            PERTOKEN == QUANT_BATCH_MATMUL_V3_IS_PERTOKEN && OPTIONATTR == QUANT_BATCH_MATMUL_V3_OPTION_ATTR_NONE) {  // false false
            INVOKE_QUANT_BATCH_MATMUL_DEQUANT_PERTOKEN_OPT_IMPL(false, false);
        } else if (
            TRANS == QUANT_BATCH_MATMUL_V3_B_TRANS && KERNEL_TEMPLATE_TYPE == QUANT_BATCH_MATMUl_V3_KERNEL_TEMPLATE_TYPE_OPT &&
            PERTOKEN == QUANT_BATCH_MATMUL_V3_IS_PERTOKEN && OPTIONATTR == QUANT_BATCH_MATMUL_V3_OPTION_ATTR_NONE) {  // false true
            INVOKE_QUANT_BATCH_MATMUL_DEQUANT_PERTOKEN_OPT_IMPL(false, true);
        } else if (
            TRANS == QUANT_BATCH_MATMUL_V3_A_TRANS && KERNEL_TEMPLATE_TYPE == QUANT_BATCH_MATMUl_V3_KERNEL_TEMPLATE_TYPE_OPT &&
            PERTOKEN == QUANT_BATCH_MATMUL_V3_IS_PERTOKEN && OPTIONATTR == QUANT_BATCH_MATMUL_V3_OPTION_ATTR_NONE) {  // true  false
            INVOKE_QUANT_BATCH_MATMUL_DEQUANT_PERTOKEN_OPT_IMPL(true, false);
        } else if (
            TRANS == QUANT_BATCH_MATMUL_V3_ALL_TRANS && KERNEL_TEMPLATE_TYPE == QUANT_BATCH_MATMUl_V3_KERNEL_TEMPLATE_TYPE_OPT &&
            PERTOKEN == QUANT_BATCH_MATMUL_V3_IS_PERTOKEN && OPTIONATTR == QUANT_BATCH_MATMUL_V3_OPTION_ATTR_NONE) {  // true  true
            INVOKE_QUANT_BATCH_MATMUL_DEQUANT_PERTOKEN_OPT_IMPL(true, true);
        } else if (
            TRANS == QUANT_BATCH_MATMUL_V3_NOT_TRANS && KERNEL_TEMPLATE_TYPE == QUANT_BATCH_MATMUl_V3_KERNEL_TEMPLATE_TYPE_BASIC &&
            PERTOKEN == QUANT_BATCH_MATMUL_V3_IS_PERTOKEN && OPTIONATTR == QUANT_BATCH_MATMUL_V3_OPTION_ATTR_NONE) {
            INVOKE_QUANT_BATCH_MATMUL_DEQUANT_PERTOKEN_BASIC_IMPL(false, false);
        } else if (
            TRANS == QUANT_BATCH_MATMUL_V3_B_TRANS && KERNEL_TEMPLATE_TYPE == QUANT_BATCH_MATMUl_V3_KERNEL_TEMPLATE_TYPE_BASIC &&
            PERTOKEN == QUANT_BATCH_MATMUL_V3_IS_PERTOKEN && OPTIONATTR == QUANT_BATCH_MATMUL_V3_OPTION_ATTR_NONE) {
            INVOKE_QUANT_BATCH_MATMUL_DEQUANT_PERTOKEN_BASIC_IMPL(false, true);
        } else if (
            TRANS == QUANT_BATCH_MATMUL_V3_A_TRANS && KERNEL_TEMPLATE_TYPE == QUANT_BATCH_MATMUl_V3_KERNEL_TEMPLATE_TYPE_BASIC &&
            PERTOKEN == QUANT_BATCH_MATMUL_V3_IS_PERTOKEN && OPTIONATTR == QUANT_BATCH_MATMUL_V3_OPTION_ATTR_NONE) {
            INVOKE_QUANT_BATCH_MATMUL_DEQUANT_PERTOKEN_BASIC_IMPL(true, false);
        } else if (
            TRANS == QUANT_BATCH_MATMUL_V3_ALL_TRANS && KERNEL_TEMPLATE_TYPE == QUANT_BATCH_MATMUl_V3_KERNEL_TEMPLATE_TYPE_BASIC &&
            PERTOKEN == QUANT_BATCH_MATMUL_V3_IS_PERTOKEN && OPTIONATTR == QUANT_BATCH_MATMUL_V3_OPTION_ATTR_NONE) {
            INVOKE_QUANT_BATCH_MATMUL_DEQUANT_PERTOKEN_BASIC_IMPL(true, true);
        } else {  // matmul with basic tiling no pertoken
            if (TRANS == QUANT_BATCH_MATMUL_V3_NOT_TRANS && KERNEL_TEMPLATE_TYPE == QUANT_BATCH_MATMUl_V3_KERNEL_TEMPLATE_TYPE_BASIC &&
                PERTOKEN == QUANT_BATCH_MATMUL_V3_NOT_PERTOKEN && OPTIONATTR == QUANT_BATCH_MATMUL_V3_OPTION_ATTR_NONE) {
                INVOKE_QUANT_BATCH_MATMUL_DEQUANT_BASIC_BLOCK_IMPL(false, false);
            } else if (
                TRANS == QUANT_BATCH_MATMUL_V3_B_TRANS && KERNEL_TEMPLATE_TYPE == QUANT_BATCH_MATMUl_V3_KERNEL_TEMPLATE_TYPE_BASIC &&
                PERTOKEN == QUANT_BATCH_MATMUL_V3_NOT_PERTOKEN && OPTIONATTR == QUANT_BATCH_MATMUL_V3_OPTION_ATTR_NONE) {
                INVOKE_QUANT_BATCH_MATMUL_DEQUANT_BASIC_BLOCK_IMPL(false, true);
            } else if (
                TRANS == QUANT_BATCH_MATMUL_V3_A_TRANS && KERNEL_TEMPLATE_TYPE == QUANT_BATCH_MATMUl_V3_KERNEL_TEMPLATE_TYPE_BASIC &&
                PERTOKEN == QUANT_BATCH_MATMUL_V3_NOT_PERTOKEN && OPTIONATTR == QUANT_BATCH_MATMUL_V3_OPTION_ATTR_NONE) {
                INVOKE_QUANT_BATCH_MATMUL_DEQUANT_BASIC_BLOCK_IMPL(true, false);
            } else if (
                TRANS == QUANT_BATCH_MATMUL_V3_ALL_TRANS && KERNEL_TEMPLATE_TYPE == QUANT_BATCH_MATMUl_V3_KERNEL_TEMPLATE_TYPE_BASIC &&
                PERTOKEN == QUANT_BATCH_MATMUL_V3_NOT_PERTOKEN && OPTIONATTR == QUANT_BATCH_MATMUL_V3_OPTION_ATTR_NONE) {
                INVOKE_QUANT_BATCH_MATMUL_DEQUANT_BASIC_BLOCK_IMPL(true, true);
            }
        }
    }
#endif
}