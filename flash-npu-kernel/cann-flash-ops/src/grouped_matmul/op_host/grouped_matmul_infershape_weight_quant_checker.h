/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GROUPED_MATMUL_INFERSHAPE_WEIGHT_QUANT_CHECKER_H_
#define GROUPED_MATMUL_INFERSHAPE_WEIGHT_QUANT_CHECKER_H_

#include "graph/utils/type_utils.h"
#include "grouped_matmul_infershape_common_util.h"
#include "log/log.h"
#include "platform/platform_info.h"
#include "register/op_impl_registry.h"

namespace ops {

class GroupedMatmulWeightQuantChecker {
public:
    GroupedMatmulWeightQuantChecker(){};
    ~GroupedMatmulWeightQuantChecker(){};
    ge::graphStatus GetXAndWeightDimValue(const gert::InferShapeContext *context, const GMMAttrs &gmmAttrs);
    ge::graphStatus CheckShape(const gert::InferShapeContext *context, const GroupedMatmulCommonUtil &commonUtil);
    ge::graphStatus InferOutShape(gert::InferShapeContext *context, const GMMAttrs &gmmAttrs) const;
    ge::graphStatus CheckScaleDtypeForS8S4(const gert::InferDataTypeContext *context) const;
    ge::graphStatus CheckBiasDtype(const gert::InferDataTypeContext *context) const;
    ge::graphStatus CheckDtype(const gert::InferDataTypeContext *context) const;
    ge::graphStatus InferOutDtype(gert::InferDataTypeContext *context) const;

private:
    ge::graphStatus GetXandWeightDtype(const gert::InferShapeContext *context);
    ge::graphStatus CheckTensorDimEqualOne(const gert::InferShapeContext *context, const gert::Shape *shape,
                                           const std::string paramName, const size_t index) const;
    ge::graphStatus UpdateShapeYMultiDim(gert::InferShapeContext *context, size_t idxY, const gert::Shape *xShape,
                                         const gert::Shape *weightShape) const;
    ge::graphStatus CheckMatmulDataType(const gert::InferDataTypeContext *context, const ge::DataType xDtype,
                                        const ge::DataType weightDtype, const ge::DataType biasDtype,
                                        const ge::DataType antiquantScaleDtype,
                                        const ge::DataType antiquantOffsetDtype) const;
    ge::graphStatus CheckTensorListDataType(const gert::InferDataTypeContext *context, uint32_t index,
                                            const ge::DataType dtype) const;
    ge::graphStatus CheckShapeForXAndWeight(const gert::InferShapeContext *context) const;
    ge::graphStatus CheckDimNumNoSplit(const gert::InferShapeContext *context,
                                       const GMMInputParamsInfo &paramsInputInfo) const;
    ge::graphStatus CheckXWeightYGroupSizeMultiScenario(const gert::InferShapeContext *context,
                                                        const GMMInputParamsInfo &paramsInputInfo) const;
    ge::graphStatus CheckTensorNDimMultiScenario(const gert::InferShapeContext *context,
                                                 const GMMInputParamsInfo &paramsInputInfo, const size_t wNDimIdx,
                                                 const int64_t weightNDimValue, const size_t index) const;
    ge::graphStatus CheckCaseMultiScenario(const gert::InferShapeContext *context, const GMMAttrs &gmmAttrs,
                                           const GMMInputParamsInfo &paramsInputInfo) const;
    ge::graphStatus CheckShapeForTensorList(const gert::InferShapeContext *context, size_t gmm_index,
                                            const std::string &tensorType, const GMMAttrs &gmmAttrs) const;
    ge::graphStatus CheckScenarioValid(const gert::InferShapeContext *context, const GMMAttrs &gmmAttrs) const;
    ge::graphStatus GetNumOfInputs(const gert::InferShapeContext *context, GMMInputParamsInfo &paramsInputInfo) const;
    ge::graphStatus CheckTensorListSizeMultiScenario(const gert::InferShapeContext *context,
                                                     const GMMInputParamsInfo &paramsInputInfo) const;
    ge::graphStatus CheckShapeValid(const gert::InferShapeContext *context, const GMMAttrs &gmmAttrs);

    ge::graphStatus CheckShapeForWeightQuantParam(const gert::InferShapeContext *context,
                                                  const GMMAttrs &gmmAttrs) const;
    ge::graphStatus CheckShapeForGrouplist(const gert::InferShapeContext *context,
                                           const gert::Shape *groupListShape) const;
    ge::graphStatus UpdateShapeY(gert::InferShapeContext *context, size_t idxY, std::vector<int64_t> &yDims) const;
    ge::graphStatus CheckGroupAntiS(const gert::Shape *tensorShape, const gert::InferShapeContext *context,
                                    const std::string &tensorType) const;
    ge::graphStatus CheckPertokenScaleForA8W4(const gert::InferShapeContext *context) const;
    ge::graphStatus CheckGroupSize(const gert::InferShapeContext *context, const GMMAttrs &gmmAttrs) const;
    bool IsA16MxFp4NZ(const ge::DataType xDtype, const ge::DataType weightDtype) const;
    bool IsMxA8W4NZ(const ge::DataType xDtype, const ge::DataType weightDtype) const;
    bool IsS8S4NZ(const ge::DataType xDtype, const ge::DataType weightDtype) const;
    bool IsA16W8(const ge::DataType xDtype, const ge::DataType weightDtype) const;
    bool IsA16F8(const ge::DataType xDtype, const ge::DataType weightDtype) const;
    bool IsA16W4(const ge::DataType xDtype, const ge::DataType weightDtype) const;

private:
    int64_t groupNum_; //当前含义为M分组数g
    int64_t xKDim_;
    int64_t xMDim_;
    int64_t weightKDim_;
    int64_t weightNDim_;
    size_t xdimNum_;
    size_t weightdimNum_;
    ge::DataType xDtype_ = ge::DT_UNDEFINED;
    ge::DataType weightDtype_ = ge::DT_UNDEFINED;
    bool hasBias_ = false;
    bool hasAntiquantOffset_ = false;
};

}  // namespace ops

#endif  // GROUPED_MATMUL_INFERSHAPE_DAVID_QUANT_CHECKER_H_