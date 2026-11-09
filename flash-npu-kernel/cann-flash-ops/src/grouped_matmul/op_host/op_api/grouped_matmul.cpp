/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "grouped_matmul.h"
#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/shape_utils.h"
#include "opdev/make_op_executor.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(GroupedMatmul);

const aclTensorList *GroupedMatmul(const aclTensorList *x,
                                   const aclTensorList *weight,
                                   const aclTensorList *biasOptional,
                                   const aclTensorList *scaleOptional,
                                   const aclTensorList *offsetOptional,
                                   const aclTensorList *antiquantScaleOptional,
                                   const aclTensorList *antiquantOffsetOptional,
                                   const aclTensor *groupListOptional,
                                   const aclTensor *perTokenScaleOptional,
                                   int64_t splitItem,
                                   op::DataType yDtype,
                                   bool transposeWeight,
                                   bool transposeX,
                                   int64_t groupType,
                                   int64_t groupListType,
                                   int64_t actType,
                                   const aclIntArray *tuningConfig,
                                   size_t outLength,
                                   aclOpExecutor *executor) {
    L0_DFX(GroupedMatmul, x, weight, biasOptional, scaleOptional, offsetOptional, antiquantScaleOptional,
           antiquantOffsetOptional, groupListOptional, perTokenScaleOptional, splitItem, yDtype,
           transposeWeight, transposeX, groupType, groupListType, actType, tuningConfig, outLength);

    std::vector<const aclTensor*> tensorsVec;
    const aclTensor *x0 = x->Size() > 0 ? (*x)[0] : nullptr;
    if (x0 == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "(*x)[0] is nullptr.");
        return nullptr;
    }

    for (size_t i(0); i < outLength; ++i) {
        tensorsVec.emplace_back(executor->AllocTensor(yDtype, x0->GetStorageFormat(), x0->GetOriginalFormat()));
    }

    int64_t outputDtype = yDtype == DataType::DT_INT32 ? 2 : -1;
    auto out = executor->AllocTensorList(tensorsVec.data(), outLength);

    auto ret = INFER_SHAPE(GroupedMatmul,
                           OP_INPUT(x, weight, biasOptional, scaleOptional, offsetOptional, antiquantScaleOptional,
                                    antiquantOffsetOptional, groupListOptional, perTokenScaleOptional),
                           OP_OUTPUT(out),
                           OP_ATTR(splitItem, outputDtype, transposeWeight, transposeX, groupType, groupListType, actType, tuningConfig));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "InferShape failed.");
        return nullptr;
    }
    ret = ADD_TO_LAUNCHER_LIST_AICORE(GroupedMatmul,
                                      OP_INPUT(x, weight, biasOptional, scaleOptional, offsetOptional,
                                               antiquantScaleOptional, antiquantOffsetOptional, groupListOptional,
                                               perTokenScaleOptional),
                                      OP_OUTPUT(out),
                                      OP_ATTR(splitItem, outputDtype, transposeWeight, transposeX, groupType, groupListType,
                                              actType, tuningConfig));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "ADD_TO_LAUNCHER_LIST_AICORE failed.");
        return nullptr;
    }

    return out;
}

}  // namespace l0op
