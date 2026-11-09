/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_GROUPED_MATMUL_WEIGHT_QUANT_950_CHECKER_H
#define OP_API_INC_GROUPED_MATMUL_WEIGHT_QUANT_950_CHECKER_H
#include "opdev/format_utils.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "grouped_matmul_util.h"

namespace gmm {
class AclnnGroupedMatmulWeightQuantDAV3510Checker {
public:
    explicit AclnnGroupedMatmulWeightQuantDAV3510Checker(const GroupedMatmulParams &gmmParams)
        : gmmParams_(gmmParams){};
    ~AclnnGroupedMatmulWeightQuantDAV3510Checker(){};
    aclnnStatus CheckGroupedMatmulWeightQuantDAV3510();

private:
    aclnnStatus CheckGroupTypeScenario() const;
    aclnnStatus CheckUnsupportedApi() const;
    aclnnStatus CheckGroupListAndSplitItem() const;
    aclnnStatus CheckAntiQuantParams() const;
    aclnnStatus CheckQuantParams() const;
    aclnnStatus CheckYDtype() const;
    aclnnStatus CheckQuantDtype() const;
    aclnnStatus CheckBiasDtype();

    aclnnStatus CheckTensorListSize() const;
    aclnnStatus CheckTensorNotNull(size_t idx) const;
    aclnnStatus CheckTensorNotNullPtr(const aclTensorList *tensorList, size_t idx, const std::string &tensorType) const;
    aclnnStatus CheckTensorDtype(const aclTensorList *tensorList, const DataType &tensorDtype, size_t idx,
                                 const std::string &tensorType) const;
    aclnnStatus CheckTensorShape(const aclTensorList *tensorList, size_t idx, const std::string &tensorType) const;

    aclnnStatus CheckWeightInnerAxisEven(size_t idx) const;
    aclnnStatus CheckDimNumAndFormat(size_t idx) const;
    aclnnStatus CheckTransposeStatus() const;
    aclnnStatus CheckDimValue(size_t idx) const;
    aclnnStatus CheckV1GroupList(size_t idx) const;

    aclnnStatus CheckAntiQuantDtype(size_t idx) const;
    aclnnStatus CheckAntiQuantShape(size_t idx) const;
    aclnnStatus CheckScaleAndPerTokenScaleShape() const;
    aclnnStatus CheckGroupSize(size_t idx) const;

    bool IsA16MxFp4NZ() const;
    bool IsMxA8W4NZ() const;
    bool IsA16W8ND() const;
    bool IsA16F8ND() const;
    bool IsS8S4NZ() const;
    bool IsA16W4() const;

private:
    GroupedMatmulParams gmmParams_;
    DataType xDtype_ = ge::DT_UNDEFINED;
    DataType weightDtype_ = ge::DT_UNDEFINED;
    DataType biasDtype_ = ge::DT_UNDEFINED;
    DataType yDtype_ = ge::DT_UNDEFINED;
};
} // namespace gmm
#endif