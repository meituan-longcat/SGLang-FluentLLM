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
 * \file flat_quant.cpp
 * \brief
 */

#include "flat_quant.h"
#include "opdev/op_log.h"
#include "opdev/op_executor.h"
#include "opdev/make_op_executor.h"
#include "opdev/shape_utils.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "aclnn_kernels/common/op_error_check.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(FlatQuant);

static std::tuple<aclTensor*, aclTensor*> FlatQuantAICore(const aclTensor* x, const aclTensor* kroneckerP1,
                                                          const aclTensor* kroneckerP2, const aclTensor* groupList,
                                                          float clipRatio, int64_t dstDtype, float dstTypeMax,
                                                          int64_t groupListType, bool clearPadding, aclTensor* out,
                                                          aclTensor* quantScale, aclOpExecutor* executor)
{
    L0_DFX(FlatQuantAICore, x, kroneckerP1, kroneckerP2, clipRatio, dstDtype, dstTypeMax, groupListType,
           clearPadding, out, quantScale);
    auto retAicore = ADD_TO_LAUNCHER_LIST_AICORE(FlatQuant, OP_INPUT(x, kroneckerP1, kroneckerP2, groupList),
                                                 OP_OUTPUT(out, quantScale),
                                                 OP_ATTR(clipRatio, dstDtype, dstTypeMax, groupListType,
                                                         clearPadding));
    OP_CHECK_ADD_TO_LAUNCHER_LIST_AICORE(retAicore != ACLNN_SUCCESS, return std::tuple(nullptr, nullptr),
                                         "FlatQuant add to aicore launch list failed.");
    return std::tie(out, quantScale);
}

const std::tuple<aclTensor*, aclTensor*> FlatQuant(const aclTensor* x, const aclTensor* kroneckerP1,
                                                   const aclTensor* kroneckerP2, const aclTensor* groupList,
                                                   float clipRatio, int64_t dstDtype, float dstTypeMax,
                                                   int64_t groupListType, bool clearPadding, aclTensor* out,
                                                   aclTensor* quantScale, aclOpExecutor* executor)
{
    // kernel输出为INT4，out统一以INT4分配，后续在L2层按需打包成INT32
    out = executor->AllocTensor(x->GetViewShape(), DataType::DT_INT4, x->GetViewFormat());
    OP_CHECK_NULL(out, return std::tuple(nullptr, nullptr));
    int64_t K = x->GetViewShape().GetDim(0);
    dstDtype = DataType::DT_INT32;
    quantScale = executor->AllocTensor(op::Shape({K}), op::DataType::DT_FLOAT, op::Format::FORMAT_ND);
    OP_CHECK_NULL(quantScale, return std::tuple(nullptr, nullptr));
    return FlatQuantAICore(x, kroneckerP1, kroneckerP2, groupList, clipRatio, dstDtype, dstTypeMax, groupListType,
                           clearPadding, out, quantScale, executor);
}

} // namespace l0op
