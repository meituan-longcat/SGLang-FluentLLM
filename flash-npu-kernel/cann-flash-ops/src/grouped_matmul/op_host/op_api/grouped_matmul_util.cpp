/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "grouped_matmul_util.h"

using namespace gmm;

namespace gmm {
bool IsTransposeLastTwoDims(const aclTensor *tensor)
{
    auto shape = tensor->GetViewShape();
    int64_t dim1 = shape.GetDimNum() - 1;
    int64_t dim2 = shape.GetDimNum() - 2;
    auto strides = tensor->GetViewStrides();
    if (strides[dim2] == 1 && strides[dim1] == shape.GetDim(dim2)) {
        int64_t tmpNxD = shape.GetDim(dim1) * shape.GetDim(dim2);
        for (int64_t batchDim = shape.GetDimNum() - 3; batchDim >= 0; batchDim--) {
            if (strides[batchDim] != tmpNxD) {
                return false;
            }
            tmpNxD *= shape.GetDim(batchDim);
        }
        return true;
    }
    return false;
}

bool IsTransposeForMxShape(const aclTensor *tensor)
{
    auto shape = tensor->GetViewShape();
    if (shape.GetDimNum() < MX_SPLIT_K_PER_TOKEN_SCALE_DIM) {
        return false;
    }
    int64_t firstLastDim = shape.GetDimNum() - 1;
    int64_t secondLastDim = shape.GetDimNum() - LAST_SECOND_DIM_INDEX;
    int64_t thirdLastDim = shape.GetDimNum() - LAST_THIRD_DIM_INDEX;
    auto strides = tensor->GetViewStrides();
    if (strides[firstLastDim] == 1 && strides[thirdLastDim] == MXFP_MULTI_BASE_SIZE &&
        strides[secondLastDim] == shape.GetDim(thirdLastDim) * MXFP_MULTI_BASE_SIZE) {
        return true;
    }
    return false;
}

void CreateContiguousTensorListForPertoken(const aclTensorList *tensorList, std::vector<aclTensor *> &newTensorList,
                                           aclOpExecutor *executor)
{
    op::Shape shape;
    for (uint64_t idx = 0; idx < (*tensorList).Size(); idx++) {
        const aclTensor *inputTensor = (*tensorList)[idx];
        op::Shape viewShape = inputTensor->GetViewShape();
        shape.SetScalar();
        if (viewShape.GetDimNum() <
            MX_SPLIT_K_PER_TOKEN_SCALE_DIM) { // only pertoken in mx typek quant mode have to trans, which dim num is 3
            continue;
        }
        // swap first two dim
        shape.AppendDim(viewShape.GetDim(1));
        shape.AppendDim(viewShape.GetDim(0));
        shape.AppendDim(viewShape.GetDim(2)); // 2 is last dim contiguous in k axis in mx typek quant mode
        aclTensor *tensor =
            executor->CreateView(inputTensor, shape, inputTensor->GetViewOffset()); // use executor to create tensor
        tensor->SetStorageFormat(inputTensor->GetStorageFormat());
        newTensorList.emplace_back(tensor);
    }
}

void CreateContiguousTensorListForMXTypeMScale(const aclTensorList *tensorList, std::vector<aclTensor *> &newTensorList,
                                               aclOpExecutor *executor)
{
    op::Shape shape;
    for (uint64_t idx = 0; idx < (*tensorList).Size(); idx++) {
        const aclTensor *inputTensor = (*tensorList)[idx];
        op::Shape viewShape = inputTensor->GetViewShape();
        shape.SetScalar();
        if (viewShape.GetDimNum() < MX_SPLIT_M_SCALE_DIM) {
            continue;
        }
        shape.AppendDim(viewShape.GetDim(0));
        shape.AppendDim(viewShape.GetDim(viewShape.GetDimNum() - LAST_SECOND_DIM_INDEX));
        shape.AppendDim(viewShape.GetDim(viewShape.GetDimNum() - LAST_THIRD_DIM_INDEX));
        shape.AppendDim(viewShape.GetDim(viewShape.GetDimNum() - 1));
        aclTensor *tensor =
            executor->CreateView(inputTensor, shape, inputTensor->GetViewOffset()); // use executor to create tensor
        tensor->SetStorageFormat(inputTensor->GetStorageFormat());
        newTensorList.emplace_back(tensor);
    }
}

void CreateContiguousTensorList(const aclTensorList *tensorList, std::vector<aclTensor *> &newTensorList,
                                aclOpExecutor *executor)
{
    op::Shape shape;
    for (uint64_t idx = 0; idx < (*tensorList).Size(); idx++) {
        const aclTensor *inputTensor = (*tensorList)[idx];
        op::Shape viewShape = inputTensor->GetViewShape();
        uint32_t viewShapeDimsNum = viewShape.GetDimNum();
        shape.SetScalar();
        // 2: the second last dimension; in for-loops, it indicates dimensions before the second last remain unchanged.
        for (uint32_t i = 0; i < viewShapeDimsNum - 2; ++i) {
            shape.AppendDim(viewShape.GetDim(i));
        }
        // viewShapeDimsNum - 1, the dim value of the last dim. viewShapeDimsNum - 2, the dim value of the second last
        // dim.
        shape.AppendDim(viewShape.GetDim(viewShapeDimsNum - 1));
        shape.AppendDim(viewShape.GetDim(viewShapeDimsNum - 2)); // 2:the second last dim.
        aclTensor *tensor =
            executor->CreateView(inputTensor, shape, inputTensor->GetViewOffset()); // use executor to create tensor
        tensor->SetStorageFormat(inputTensor->GetStorageFormat());
        newTensorList.emplace_back(tensor);
    }
}

std::string dTypeToString(const ge::DataType &dtype) {
    if(DTYPE_STRING.count(dtype) != 0) {
        return DTYPE_STRING.at(dtype);
    } else {
        return std::string(op::ToString(dtype).GetString());
    }
}

} // namespace gmm