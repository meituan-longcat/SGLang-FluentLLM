/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GROUPED_MATMUL_INFERSHAPE_QUANT_CHECKER_H_
#define GROUPED_MATMUL_INFERSHAPE_QUANT_CHECKER_H_

#include "register/op_impl_registry.h"
#include "log/log.h"
#include "platform/platform_info.h"
#include "graph/utils/type_utils.h"
#include "grouped_matmul_infershape_common_util.h"

namespace ops {

class GroupedMatmulQuantChecker {
public:
    GroupedMatmulQuantChecker(){};
    ~GroupedMatmulQuantChecker(){};
    ge::graphStatus GetXAndWeightDimValue(const gert::InferShapeContext *context, const GMMAttrs &gmmAttrs);
    ge::graphStatus CheckShape(const gert::InferShapeContext *context, const GroupedMatmulCommonUtil &commonUtil) const;
    ge::graphStatus CheckDtype(const gert::InferDataTypeContext *context,
                               const GroupedMatmulCommonUtil &commonUtil) const;
    ge::graphStatus InferOutShape(gert::InferShapeContext *context, const GMMAttrs &gmmAttrs);
    ge::graphStatus InferOutDtype(gert::InferDataTypeContext *context);
    ge::graphStatus GetGroupNumValue(const gert::InferShapeContext *context);
private:
    ge::graphStatus CheckFormatValid(const gert::InferShapeContext *context) const;
    ge::graphStatus CheckDtypeValid(const gert::InferDataTypeContext *context) const;
    ge::graphStatus CheckScenarioValidForShape(const gert::InferShapeContext *context, const GMMAttrs &gmmAttrs) const;
    ge::graphStatus CheckScenarioValidForDtype(const gert::InferDataTypeContext *context,
                                               const GMMAttrs &gmmAttrs) const;
    ge::graphStatus CheckShapeValid(const gert::InferShapeContext *context, const GMMAttrs &gmmAttrs) const;
    ge::graphStatus CheckShapeForBias(const gert::InferShapeContext *context) const;
    ge::graphStatus CheckShapeForQuantParam(const gert::InferShapeContext *context, const GMMAttrs &gmmAttrs) const;
    ge::graphStatus CheckNotZeroValueForNoneSplitAxis(const gert::InferShapeContext *context,
                                                      const GMMAttrs &gmmAttrs) const;
    ge::graphStatus CheckShapeForGrouplist(const gert::InferShapeContext *context) const;
    ge::graphStatus UpdateShapeY(gert::InferShapeContext *context, size_t idxY, std::vector<int64_t> &yDims);
    ge::graphStatus CheckGroupedMatmulPerGroupDim(const gert::InferShapeContext *context,
                                                  const GMMAttrs &gmmAttrs) const;
    ge::graphStatus CheckShapeForPerGroupQuantParam(const gert::InferShapeContext *context,
                                                    const GMMAttrs &gmmAttrs) const;
    ge::graphStatus CheckGroupedMatmulPerTileShape(const gert::InferShapeContext *context,
                                                   const GMMAttrs &gmmAttrs) const;
    ge::graphStatus CheckPertokenShapeInNormalQuantMode(const gert::InferShapeContext *context,
                                                        const GMMAttrs &gmmAttrs,
                                                        const gert::Shape *perTokenScaleShape) const;
    bool IsDoubleScaleScenario(const gert::Shape *scaleShape, const gert::Shape *perTokenScaleShape) const;
    bool IsPerTileQuantMode(const gert::InferShapeContext *context, const GMMAttrs &gmmAttrs) const;
    bool IsDoubleScaleScenario(const gert::Shape *scaleShape, const gert::Shape *perTokenScaleShape);
    bool IsMxQuantMode(const gert::InferShapeContext *context, const GMMAttrs &gmmAttrs) const;
    bool LogicXOR(bool cond1, bool cond2) const;
    bool CheckShapeContainsNegativeValue(const gert::Shape *shape) const;
    bool CheckDataContainsNegativeValue() const;
    bool CheckNonDataInputContainsNegativeValue(const gert::InferShapeContext *context) const;
private:
    int64_t groupNum_ = 0L;
    int64_t xKDim_ = 0L;
    int64_t xMDim_ = 0L;
    int64_t weightKDim_ = 0L;
    int64_t weightNDim_ = 0L;
    size_t xdimNum_ = 0;
    size_t weightdimNum_ = 0;
};
} // namespace ops

#endif // GROUPED_MATMUL_INFERSHAPE_DAVID_QUANT_CHECKER_H_