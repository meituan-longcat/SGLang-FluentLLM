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
* \file grouped_matmul_tiling_key.h
* \brief
*/

#ifndef __OP_KERNEL_NO_QUANT_GMM_TILING_KEY_H__
#define __OP_KERNEL_NO_QUANT_GMM_TILING_KEY_H__

#include "ascendc/host_api/tiling/template_argument.h"

#define GMM_NO_TRANS 0
#define GMM_TRANS 1

// 模板参数
ASCENDC_TPL_ARGS_DECL(
    GroupedMatmul, // 算子OpType
    ASCENDC_TPL_UINT_DECL(NO_QUANT_B_TRANS, ASCENDC_TPL_2_BW, ASCENDC_TPL_UI_LIST, GMM_NO_TRANS, GMM_TRANS),
    ASCENDC_TPL_UINT_DECL(NO_QUANT_A_TRANS, ASCENDC_TPL_2_BW, ASCENDC_TPL_UI_LIST, GMM_NO_TRANS, GMM_TRANS)
    );

// 模板参数组合
// 用于调用GET_TPL_TILING_KEY获取TilingKey时，接口内部校验TilingKey是否合法
ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIC_ONLY),
        ASCENDC_TPL_UINT_SEL(NO_QUANT_B_TRANS, ASCENDC_TPL_UI_LIST, GMM_NO_TRANS),
        ASCENDC_TPL_UINT_SEL(NO_QUANT_A_TRANS, ASCENDC_TPL_UI_LIST, GMM_NO_TRANS)),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_MIX_AIC_1_1),
        ASCENDC_TPL_UINT_SEL(NO_QUANT_B_TRANS, ASCENDC_TPL_UI_LIST, GMM_NO_TRANS),
        ASCENDC_TPL_UINT_SEL(NO_QUANT_A_TRANS, ASCENDC_TPL_UI_LIST, GMM_TRANS)),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIC_ONLY),
        ASCENDC_TPL_UINT_SEL(NO_QUANT_B_TRANS, ASCENDC_TPL_UI_LIST, GMM_TRANS),
        ASCENDC_TPL_UINT_SEL(NO_QUANT_A_TRANS, ASCENDC_TPL_UI_LIST, GMM_NO_TRANS))
);

#endif