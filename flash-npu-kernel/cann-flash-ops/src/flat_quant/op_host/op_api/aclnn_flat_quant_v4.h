/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_FLAT_QUANT_V4_H_
#define OP_API_INC_FLAT_QUANT_V4_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief FlatQuant with optional materialization of rows outside group_list's
 * valid prefix.
 *
 * @param clearPadding When true, zero the out and quantScale rows in
 * [validK, x.dim(0)). When false, those rows are left unspecified. The
 * default behavior exposed by higher-level bindings is false.
 */
ACLNN_API aclnnStatus aclnnFlatQuantV4GetWorkspaceSize(const aclTensor* x, const aclTensor* kroneckerP1,
                                                       const aclTensor* kroneckerP2,
                                                       const aclTensor* groupListOptional, double clipRatio,
                                                       double dstTypeMax, int64_t groupListType, bool clearPadding,
                                                       aclTensor* out, aclTensor* quantScale,
                                                       uint64_t* workspaceSize, aclOpExecutor** executor);

ACLNN_API aclnnStatus aclnnFlatQuantV4(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                       aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_FLAT_QUANT_V4_H_
