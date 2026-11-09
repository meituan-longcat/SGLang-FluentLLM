/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file mla_prolog_v3.cpp
 * \brief
 */

#include "mla_prolog_v3_kernel_split_n.h"

using namespace MlaPrologV3;

template<uint8_t CacheMode, uint8_t Scenario, uint8_t QuantMode,
         bool EnableDequantOpt, bool EnableGroupComputeOpt, uint8_t EmptyTensorMode,
         uint8_t ActualSeqMode, uint8_t EnableSpitMOpt, uint8_t CvMode>
__global__ __aicore__ void mla_prolog_v3(
    __gm__ uint8_t *tokenX,
    __gm__ uint8_t *weightDq,
    __gm__ uint8_t *weightUqQr,
    __gm__ uint8_t *weightUk,
    __gm__ uint8_t *weightDkvKr,
    __gm__ uint8_t *rmsnormGammaCq,
    __gm__ uint8_t *rmsnormGammaCkv,
    __gm__ uint8_t *ropeSin,
    __gm__ uint8_t *ropeCos,
    __gm__ uint8_t *kvCache,
    __gm__ uint8_t *krCache,
    __gm__ uint8_t *cacheIndex,
    __gm__ uint8_t *dequantScaleX,
    __gm__ uint8_t *dequantScaleWDq,
    __gm__ uint8_t *dequantScaleWUqQr,
    __gm__ uint8_t *dequantScaleWDkvKr,
    __gm__ uint8_t *quantScaleCkv,
    __gm__ uint8_t *quantScaleCkr,
    __gm__ uint8_t *smoothScalesCq,
    __gm__ uint8_t *actualSeqLen,
    __gm__ uint8_t *kNopeClipAlpha,
    __gm__ uint8_t *queryOut,
    __gm__ uint8_t *queryRopeOut,
    __gm__ uint8_t *kvCacheOut,
    __gm__ uint8_t *krCacheOut,
    __gm__ uint8_t *dequantScaleQNopeOut,
    __gm__ uint8_t *queryNormOut,
    __gm__ uint8_t *dequantScaleQNormOut,
    __gm__ uint8_t *workspace,
    __gm__ uint8_t *tiling) {
    REGISTER_TILING_DEFAULT(optiling::MlaPrologV3TilingData);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

    constexpr auto emptyMode = static_cast<EMPTY_TENSOR_MODE>(EmptyTensorMode);
    if constexpr (emptyMode == EMPTY_TENSOR_MODE::EMPTY_QUERY) {
        return;
    }
    constexpr auto cacheMode = static_cast<CACHE_MODE>(CacheMode);
    constexpr auto actualseqMode = static_cast<ACTUAL_SEQ_MODE>(ActualSeqMode);
    constexpr uint32_t cvRatio = CvMode == ASCENDC_TPL_MIX_AIC_1_1 ? 1 : 2;

    GET_TILING_DATA_WITH_STRUCT(optiling::MlaPrologV3TilingData, tilingDataIn, tiling);
    const optiling::MlaPrologV3TilingData *__restrict tilingData = nullptr;
    const optiling::MlaPrologV3BaseParams *__restrict tilingDataBaseParams = &tilingDataIn.baseParams;

    TPipe pipe;
    if constexpr (static_cast<SCENARIO>(Scenario) == SCENARIO::NO_QUANT) {
        MlaPrologV3SplitN<MLAPType<bfloat16_t, bfloat16_t, bfloat16_t, cacheMode,
            EnableDequantOpt, EnableGroupComputeOpt, emptyMode, actualseqMode, false, cvRatio>> op(&pipe, tilingData, tilingDataBaseParams);
        op.Init(tokenX, weightDq, weightUqQr, weightUk, weightDkvKr, rmsnormGammaCq, rmsnormGammaCkv, ropeSin,         \
                ropeCos, kvCacheOut, krCacheOut, cacheIndex, dequantScaleX, dequantScaleWDq, dequantScaleWUqQr,        \
                dequantScaleWDkvKr, quantScaleCkv, quantScaleCkr, smoothScalesCq, actualSeqLen, kNopeClipAlpha,        \
                queryOut, queryRopeOut, dequantScaleQNopeOut, queryNormOut, dequantScaleQNormOut, workspace);
        op.Process();

    } else if constexpr (static_cast<SCENARIO>(Scenario) == SCENARIO::QUANT && static_cast<QUANT_MODE>(QuantMode) == QUANT_MODE::PARTIAL_QUANT_KV_NO_QUANT) {
        MlaPrologV3SplitN<MLAPType<bfloat16_t, int8_t, bfloat16_t, cacheMode,
            EnableDequantOpt, EnableGroupComputeOpt, emptyMode, actualseqMode, false, cvRatio>> op(&pipe, tilingData, tilingDataBaseParams);
        op.Init(tokenX, weightDq, weightUqQr, weightUk, weightDkvKr, rmsnormGammaCq, rmsnormGammaCkv, ropeSin,         \
                ropeCos, kvCacheOut, krCacheOut, cacheIndex, dequantScaleX, dequantScaleWDq, dequantScaleWUqQr,        \
                dequantScaleWDkvKr, quantScaleCkv, quantScaleCkr, smoothScalesCq, actualSeqLen, kNopeClipAlpha,        \
                queryOut, queryRopeOut, dequantScaleQNopeOut, queryNormOut, dequantScaleQNormOut, workspace);
        op.Process();

    } else if constexpr (static_cast<SCENARIO>(Scenario) == SCENARIO::QUANT && static_cast<QUANT_MODE>(QuantMode) == QUANT_MODE::PARTIAL_QUANT_KV_QUANT) {
        MlaPrologV3SplitN<MLAPType<bfloat16_t, int8_t, int8_t, cacheMode,
            EnableDequantOpt, EnableGroupComputeOpt, emptyMode, actualseqMode, false, cvRatio>> op(&pipe, tilingData, tilingDataBaseParams);
        op.Init(tokenX, weightDq, weightUqQr, weightUk, weightDkvKr, rmsnormGammaCq, rmsnormGammaCkv, ropeSin,         \
                ropeCos, kvCacheOut, krCacheOut, cacheIndex, dequantScaleX, dequantScaleWDq, dequantScaleWUqQr,        \
                dequantScaleWDkvKr, quantScaleCkv, quantScaleCkr, smoothScalesCq, actualSeqLen, kNopeClipAlpha,        \
                queryOut, queryRopeOut, dequantScaleQNopeOut, queryNormOut, dequantScaleQNormOut, workspace);
        op.Process();

    } else if constexpr (static_cast<SCENARIO>(Scenario) == SCENARIO::QUANT && static_cast<QUANT_MODE>(QuantMode) == QUANT_MODE::FULL_QUANT_KV_NO_QUANT) {
        MlaPrologV3SplitN<MLAPType<int8_t, int8_t, bfloat16_t, cacheMode,
            EnableDequantOpt, EnableGroupComputeOpt, emptyMode, actualseqMode, false, cvRatio>> op(&pipe, tilingData, tilingDataBaseParams);
        op.Init(tokenX, weightDq, weightUqQr, weightUk, weightDkvKr, rmsnormGammaCq, rmsnormGammaCkv, ropeSin,         \
                ropeCos, kvCacheOut, krCacheOut, cacheIndex, dequantScaleX, dequantScaleWDq, dequantScaleWUqQr,        \
                dequantScaleWDkvKr, quantScaleCkv, quantScaleCkr, smoothScalesCq, actualSeqLen, kNopeClipAlpha,        \
                queryOut, queryRopeOut, dequantScaleQNopeOut, queryNormOut, dequantScaleQNormOut, workspace);
        op.Process();

    } else if constexpr (static_cast<SCENARIO>(Scenario) == SCENARIO::QUANT && static_cast<QUANT_MODE>(QuantMode) == QUANT_MODE::FULL_QUANT_KV_QUANT) {
        if (EnableSpitMOpt == 1U) {
            MlaPrologV3SplitN<MLAPType<int8_t, int8_t, int8_t, cacheMode,
                EnableDequantOpt, EnableGroupComputeOpt, emptyMode, actualseqMode, false, cvRatio>> op(&pipe, tilingData, tilingDataBaseParams);
            op.Init(tokenX, weightDq, weightUqQr, weightUk, weightDkvKr, rmsnormGammaCq, rmsnormGammaCkv, ropeSin,         \
                    ropeCos, kvCacheOut, krCacheOut, cacheIndex, dequantScaleX, dequantScaleWDq, dequantScaleWUqQr,        \
                    dequantScaleWDkvKr, quantScaleCkv, quantScaleCkr, smoothScalesCq, actualSeqLen, kNopeClipAlpha,        \
                    queryOut, queryRopeOut, dequantScaleQNopeOut, queryNormOut, dequantScaleQNormOut, workspace);
            op.Process();
        } else {
            MlaPrologV3SplitN<MLAPType<int8_t, int8_t, int8_t, cacheMode,
                EnableDequantOpt, EnableGroupComputeOpt, emptyMode, actualseqMode, false, cvRatio>> op(&pipe, tilingData, tilingDataBaseParams);
            op.Init(tokenX, weightDq, weightUqQr, weightUk, weightDkvKr, rmsnormGammaCq, rmsnormGammaCkv, ropeSin,         \
                    ropeCos, kvCacheOut, krCacheOut, cacheIndex, dequantScaleX, dequantScaleWDq, dequantScaleWUqQr,        \
                    dequantScaleWDkvKr, quantScaleCkv, quantScaleCkr, smoothScalesCq, actualSeqLen, kNopeClipAlpha,        \
                    queryOut, queryRopeOut, dequantScaleQNopeOut, queryNormOut, dequantScaleQNormOut, workspace);
            op.Process();
        }
    } else if constexpr (static_cast<SCENARIO>(Scenario) == SCENARIO::QUANT && static_cast<QUANT_MODE>(QuantMode) == QUANT_MODE::PARTIAL_QUANT_KV_QUANT_PERTILE) {
        MlaPrologV3SplitN<MLAPType<bfloat16_t, int8_t, int8_t, cacheMode,
            EnableDequantOpt, EnableGroupComputeOpt, emptyMode, actualseqMode, true, cvRatio>> op(&pipe, tilingData, tilingDataBaseParams);
        op.Init(tokenX, weightDq, weightUqQr, weightUk, weightDkvKr, rmsnormGammaCq, rmsnormGammaCkv, ropeSin,         \
                ropeCos, kvCacheOut, krCacheOut, cacheIndex, dequantScaleX, dequantScaleWDq, dequantScaleWUqQr,        \
                dequantScaleWDkvKr, quantScaleCkv, quantScaleCkr, smoothScalesCq, actualSeqLen, kNopeClipAlpha,        \
                queryOut, queryRopeOut, dequantScaleQNopeOut, queryNormOut, dequantScaleQNormOut, workspace);
        op.Process();
    } else if constexpr (static_cast<SCENARIO>(Scenario) == SCENARIO::QUANT && static_cast<QUANT_MODE>(QuantMode) == QUANT_MODE::FULL_QUANT_KV_QUANT_PERTILE) {
        if (EnableSpitMOpt == 1U) {
            MlaPrologV3SplitN<MLAPType<int8_t, int8_t, int8_t, cacheMode,
                EnableDequantOpt, EnableGroupComputeOpt, emptyMode, actualseqMode, true, cvRatio>> op(&pipe, tilingData, tilingDataBaseParams);
            op.Init(tokenX, weightDq, weightUqQr, weightUk, weightDkvKr, rmsnormGammaCq, rmsnormGammaCkv, ropeSin,         \
                    ropeCos, kvCacheOut, krCacheOut, cacheIndex, dequantScaleX, dequantScaleWDq, dequantScaleWUqQr,        \
                    dequantScaleWDkvKr, quantScaleCkv, quantScaleCkr, smoothScalesCq, actualSeqLen, kNopeClipAlpha,        \
                    queryOut, queryRopeOut, dequantScaleQNopeOut, queryNormOut, dequantScaleQNormOut, workspace);
            op.Process();
        } else {
            MlaPrologV3SplitN<MLAPType<int8_t, int8_t, int8_t, cacheMode,
                EnableDequantOpt, EnableGroupComputeOpt, emptyMode, actualseqMode, true, cvRatio>> op(&pipe, tilingData, tilingDataBaseParams);
            op.Init(tokenX, weightDq, weightUqQr, weightUk, weightDkvKr, rmsnormGammaCq, rmsnormGammaCkv, ropeSin,         \
                    ropeCos, kvCacheOut, krCacheOut, cacheIndex, dequantScaleX, dequantScaleWDq, dequantScaleWUqQr,        \
                    dequantScaleWDkvKr, quantScaleCkv, quantScaleCkr, smoothScalesCq, actualSeqLen, kNopeClipAlpha,        \
                    queryOut, queryRopeOut, dequantScaleQNopeOut, queryNormOut, dequantScaleQNormOut, workspace);
            op.Process();
        }
    } else if constexpr (static_cast<SCENARIO>(Scenario) == SCENARIO::QUANT && static_cast<QUANT_MODE>(QuantMode) == QUANT_MODE::NO_QUANT_KV_QUANT_PERTILE) {
        MlaPrologV3SplitN<MLAPType<bfloat16_t, bfloat16_t, int8_t, cacheMode,
            EnableDequantOpt, EnableGroupComputeOpt, emptyMode, actualseqMode, true, cvRatio>> op(&pipe, tilingData, tilingDataBaseParams);
        op.Init(tokenX, weightDq, weightUqQr, weightUk, weightDkvKr, rmsnormGammaCq, rmsnormGammaCkv, ropeSin,         \
                ropeCos, kvCacheOut, krCacheOut, cacheIndex, dequantScaleX, dequantScaleWDq, dequantScaleWUqQr,        \
                dequantScaleWDkvKr, quantScaleCkv, quantScaleCkr, smoothScalesCq, actualSeqLen, kNopeClipAlpha,        \
                queryOut, queryRopeOut, dequantScaleQNopeOut, queryNormOut, dequantScaleQNormOut, workspace);
        op.Process();
    }
}
