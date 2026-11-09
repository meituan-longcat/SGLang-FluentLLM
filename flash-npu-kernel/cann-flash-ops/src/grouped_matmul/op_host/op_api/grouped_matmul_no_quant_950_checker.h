/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_GROUPED_MATMUL_NO_QUANT_910_95_CHECKER_H
#define OP_API_INC_GROUPED_MATMUL_NO_QUANT_910_95_CHECKER_H
#include "opdev/format_utils.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "grouped_matmul_util.h"

namespace gmm {
class AclnnGroupedMatmulNoQuantDAV3510Checker {
public:
    explicit AclnnGroupedMatmulNoQuantDAV3510Checker(const GroupedMatmulParams &gmmParams) : gmmParams_(gmmParams){};
    ~AclnnGroupedMatmulNoQuantDAV3510Checker(){};
    aclnnStatus CheckGroupedMatmulFunctionParamsNoQuantDAV3510() const;
    aclnnStatus CheckGroupedMatmulGroupSizeNoQuantDAV3510();


private:
    aclnnStatus CheckEmptyTensor() const;
    aclnnStatus CheckTensorListLength(const aclTensorList *tensorList) const;
private:
    GroupedMatmulParams gmmParams_;
};
} // namespace gmm
#endif