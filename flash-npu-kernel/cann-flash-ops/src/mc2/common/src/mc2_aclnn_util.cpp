/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "mc2_aclnn_util.h"
#include "opdev/common_types.h"

namespace {
bool IsTransposeLastTwoDims(const aclTensor *tensor)
{
    const auto &shape = tensor->GetViewShape();
    if (shape.GetDimNum() != 2) {
        return false;
    }

    const auto &strides = tensor->GetViewStrides();
    return strides[0] == 1 && strides[1] == shape.GetDim(0) &&
           !(shape.GetDim(0) == 1 && shape.GetDim(1) == 1);
}
} // namespace

namespace MC2Aclnn {

static constexpr size_t MX_SCALE_MAX_DIM = 3;

bool CustomIsNeedScaleTrans(const aclTensor *mxScaleTensor)
{
    if (mxScaleTensor->GetViewShape().GetDimNum() < MX_SCALE_MAX_DIM) {
        return IsTransposeLastTwoDims(mxScaleTensor);
    }
    bool transposeFlag = false;
    int64_t dimNum = mxScaleTensor->GetViewShape().GetDimNum();
    int64_t lastDim = mxScaleTensor->GetViewShape().GetDim(dimNum - 1);     // 1:最后一个维度
    int64_t lastSecondDim = mxScaleTensor->GetViewShape().GetDim(dimNum - 2); // 2:倒数第二个维度
    int64_t lastThirdDim = mxScaleTensor->GetViewShape().GetDim(dimNum - 3);  // 3:倒数第三个维度
    if (mxScaleTensor->GetViewStrides()[dimNum - 3] == lastDim &&   // 3:倒数第三个维度
        mxScaleTensor->GetViewStrides()[dimNum - 2] == lastDim * lastThirdDim) {    // 2:倒数第二个维度
        transposeFlag = true;
        if (lastSecondDim == 1 && lastThirdDim == 1) {
            transposeFlag = false;
        }
    }
    return transposeFlag;
}
} // namespace MC2Aclnn
