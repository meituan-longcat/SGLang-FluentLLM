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
 * \file grouped_matmul_infershape.cpp
 * \brief
 */
#include <algorithm>

#include "register/op_impl_registry.h"
#include "log/log.h"
#include "platform/platform_info.h"
#include "grouped_matmul_infershape_weight_quant_checker.h"
#include "grouped_matmul_infershape_quant_checker.h"
#include "grouped_matmul_infershape_common_util.h"

using namespace ge;
namespace ops {

static std::set<std::string> GmmDavidSupportSoc = {"Ascend950"};

enum class PlatformID : std::uint8_t {
    UNKNOWN,
    ASCEND310P,
    ASCEND910B,
    ASCEND950
};

struct GMMParamsInfo {
    size_t numX;
    size_t numWeight;
    size_t numY;
    int64_t lenGroupList;
    size_t groupNum;
    size_t numScale;
    size_t numOffset;
    size_t numAntiquantScale;
    size_t numAntiquantOffset;
    PlatformID platform;
};

struct GMMSetOutputParams {
    bool isSingleX;
    bool isSingleY;
    size_t xDimM;
    size_t weightDimN;
    int64_t lenGroupList;
    size_t numWeight;
    size_t numX;
};

static inline std::string ToString(const std::int64_t value) {
    return std::to_string(value);
}

static ge::graphStatus CheckSplitItem(int64_t splitItem) {
    if (splitItem == GMM_X_Y_SEPARATED || splitItem == GMM_NO_SEPARATED ||
        splitItem == GMM_X_SEPARATED || splitItem == GMM_Y_SEPARATED) {
        return GRAPH_SUCCESS;
    } else {
        return GRAPH_FAILED;
    }
}

static bool IsTensorListNullOrEmpty(const gert::InferShapeContext* context, size_t index) {
    auto shape = context->GetDynamicInputShape(index, 0);
    if (shape == nullptr) {
        return true;
    }
    if (shape->GetDimNum() == 0 || (shape->GetDimNum() == 1 && shape->GetDim(0) == 0)) {
        if (context->GetDynamicInputShape(index, 1) == nullptr) {
            return true;
        }
    }
    return false;
}

static ge::graphStatus CheckGroupType(const gert::InferShapeContext* context, int64_t groupType) {
    if (groupType == GMM_NO_SPLIT || groupType == GMM_SPLIT_M || groupType == GMM_SPLIT_K) {
        return GRAPH_SUCCESS;
    } else if (groupType == GMM_SPLIT_N) {
        OP_LOGE(context->GetNodeName(), "Splitting tensor along the N-axis is not supported yet.");
        return GRAPH_FAILED;
    } else {
        OP_LOGE(context->GetNodeName(), "GroupType can only be -1/0/2 now, but actually %ld is given.", groupType);
        return GRAPH_FAILED;
    }
}

static ge::graphStatus UpdateShapeYMultiDim(gert::InferShapeContext* context, size_t idxY, const gert::Shape* xShape,
                                            const gert::Shape* weightShape) {
    gert::Shape* yShape = context->GetOutputShape(idxY);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);
    *yShape = *xShape;
    size_t dimY = yShape->GetDimNum();
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const bool* transposeWPtr = attrs->GetAttrPointer<bool>(GMM_INDEX_ATTR_TRANSPOSE_W);
    const bool* transposeXPtr = attrs->GetAttrPointer<bool>(GMM_INDEX_ATTR_TRANSPOSE_X);

    OP_CHECK_NULL_WITH_CONTEXT(context, weightShape);
    if (transposeWPtr != nullptr && *transposeWPtr) {
        yShape->SetDim(dimY - 1, weightShape->GetDim(weightShape->GetDimNum() - 2));  // -2: transpose weight
    } else {
        yShape->SetDim(dimY - 1, weightShape->GetDim(weightShape->GetDimNum() - 1));
    }
    if (transposeXPtr != nullptr && *transposeXPtr) {
        yShape->SetDim(dimY - 2, xShape->GetDim(xShape->GetDimNum() - 1));  // -2: last two dim of Y
    }
    return GRAPH_SUCCESS;
}

static ge::graphStatus UpdateShapeY(gert::InferShapeContext* context, size_t idxY, std::vector<int64_t> yDims) {
    gert::Shape* yShape = context->GetOutputShape(idxY);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);
    yShape->SetDimNum(yDims.size());
    for (size_t dim = 0; dim < yDims.size(); ++dim) {
        yShape->SetDim(dim, yDims[dim]);
    }
    return GRAPH_SUCCESS;
}

static ge::graphStatus UpdateMultipleShapeY(gert::InferShapeContext* context, const gert::Tensor* groupListTensor,
                                            size_t weightDimN, bool isXTransposed, size_t xDimM) {
    auto groupListData = groupListTensor->GetData<int64_t>();
    OP_CHECK_IF(groupListData == nullptr,
              OP_LOGE(context->GetNodeName(), "Failed to obtain necessary data from groupListTensor."),
              return GRAPH_FAILED);
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t* groupListTypePtr = attrs->GetAttrPointer<int64_t>(GMM_INDEX_ATTR_GROUP_LIST_TYPE);
    OP_CHECK_NULL_WITH_CONTEXT(context, groupListTypePtr);
    const gert::Shape* x0Shape = context->GetDynamicInputShape(GMM_INDEX_IN_X, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context, x0Shape);
    const gert::Shape* weight0Shape = context->GetDynamicInputShape(GMM_INDEX_IN_WEIGHT, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context, weight0Shape);
    int64_t preOffset = 0;
    for (int idx = 0; idx < groupListTensor->GetShapeSize(); ++idx) {
        const gert::Shape* weightShape = context->GetDynamicInputShape(GMM_INDEX_IN_WEIGHT, idx);
        if (weightShape == nullptr) {
            weightShape = weight0Shape;
        }
        if (isXTransposed) {
            const gert::Shape* xShape = context->GetDynamicInputShape(GMM_INDEX_IN_X, idx);
            if (xShape == nullptr) {
                xShape = x0Shape;
            }
            std::vector<int64_t> yDims = {xShape->GetDim(xDimM), weightShape->GetDim(weightDimN)};
            OP_CHECK_IF(UpdateShapeY(context, GMM_INDEX_OUT_Y + idx, yDims) != GRAPH_SUCCESS, OP_LOGE(context->GetNodeName(),
                      "Failed to update shape of y."), return GRAPH_FAILED);
        } else {
            std::vector<int64_t> yDims;
            if (*groupListTypePtr == 0) {
                yDims = {groupListData[idx] - preOffset, weightShape->GetDim(weightDimN)};
            } else if (*groupListTypePtr == 1) {
                yDims = {groupListData[idx], weightShape->GetDim(weightDimN)};
            } else {
                OP_LOGE(context->GetNodeName(), "Invalid groupListType = %ld", *groupListTypePtr);
                return GRAPH_FAILED;
            }
            OP_CHECK_IF(UpdateShapeY(context, GMM_INDEX_OUT_Y + idx, yDims) != GRAPH_SUCCESS, OP_LOGE(context->GetNodeName(),
                      "Failed to update shape of y."), return GRAPH_FAILED);
            preOffset = groupListData[idx];
        }
    }

    return GRAPH_SUCCESS;
}

static ge::graphStatus MultiInMultiOutWithoutGroupList(gert::InferShapeContext* context) {
    size_t idx = 0;
    size_t idw = 0;
    const gert::Shape* w0Shape = context->GetDynamicInputShape(GMM_INDEX_IN_WEIGHT, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context, w0Shape);
    while (true) {
        const gert::Shape* xShape = context->GetDynamicInputShape(GMM_INDEX_IN_X, idx);
        if (xShape == nullptr) {
            break;
        }
        ++idx;
        const gert::Shape* wShape = context->GetDynamicInputShape(GMM_INDEX_IN_WEIGHT, idw);
        if (wShape) {
            ++idw;
        } else {
            wShape = w0Shape;
        }
        OP_CHECK_IF(UpdateShapeYMultiDim(context, GMM_INDEX_OUT_Y + idx - 1, xShape, wShape) != GRAPH_SUCCESS,
                  OP_LOGE(context->GetNodeName(), "Failed to update shape of y."), return GRAPH_FAILED);
    }
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t* groupTypePtr = attrs->GetAttrPointer<int64_t>(GMM_INDEX_ATTR_GROUP_TYPE);
    bool success = true;
    if (w0Shape->GetDimNum() == 2) {  // 2 two-dim weight tensor
        if (groupTypePtr != nullptr && *groupTypePtr == 2) {
            success = true;
        } else {
            success = idx == idw;
        }
    } else {
        success = static_cast<int64_t>(idx) == w0Shape->GetDim(0);
    }
    OP_CHECK_IF(!success,
              OP_LOGE(context->GetNodeName(),
                        "x tensorList's length[%zu] != weight tensor's first dim[%ld] and length[%zu]",
                        idx, w0Shape->GetDim(0), idw),
             return GRAPH_FAILED);
    return GRAPH_SUCCESS;
}

static ge::graphStatus MultiWeightMultiOutWithoutGroupList(gert::InferShapeContext* context) {
    size_t idx = 0;
    const gert::Shape* x0Shape = context->GetDynamicInputShape(GMM_INDEX_IN_X, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context, x0Shape);
    while (true) {
        const gert::Shape* wShape = context->GetDynamicInputShape(GMM_INDEX_IN_WEIGHT, idx);
        if (!wShape) {
            break;
        }
        ++idx;
        OP_CHECK_IF(UpdateShapeYMultiDim(context, GMM_INDEX_OUT_Y + idx - 1, x0Shape, wShape) != GRAPH_SUCCESS,
                  OP_LOGE(context->GetNodeName(), "Failed to update shape of y."), return GRAPH_FAILED);
    }

    return GRAPH_SUCCESS;
}

template <typename T>
static ge::graphStatus GetAttrsValue(T context, GMMAttrs &gmmAttrs)
{
    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);

    const int64_t *splitItemPtr = attrs->GetAttrPointer<int64_t>(GMM_INDEX_ATTR_SPLIT_ITEM);
    OP_CHECK_NULL_WITH_CONTEXT(context, splitItemPtr);
    gmmAttrs.splitItem = *splitItemPtr;
    OP_LOGI(context->GetNodeName(), "Attr splitItem = %ld", gmmAttrs.splitItem);

    const int64_t *dtypePtr = attrs->GetAttrPointer<int64_t>(GMM_INDEX_ATTR_OUTPUT_DTYPE);
    OP_CHECK_NULL_WITH_CONTEXT(context, dtypePtr);
    gmmAttrs.outputDtype = *dtypePtr;
    OP_LOGI(context->GetNodeName(), "Attr dtype = %ld", gmmAttrs.outputDtype);

    const auto tuningConfigPtr = attrs->GetAttrPointer<gert::ContinuousVector>(GMM_INDEX_ATTR_TUNING_CONFIG);
    gmmAttrs.tuningConfig = (tuningConfigPtr != nullptr && tuningConfigPtr->GetSize() > 0) ?
                            (reinterpret_cast<const int64_t *>(tuningConfigPtr->GetData()))[0] : 0;
    OP_LOGI(context->GetNodeName(), "Attr tuningConfig = %ld", gmmAttrs.tuningConfig);

    const int64_t *groupTypePtr = attrs->GetAttrPointer<int64_t>(GMM_INDEX_ATTR_GROUP_TYPE);
    OP_CHECK_NULL_WITH_CONTEXT(context, groupTypePtr);
    gmmAttrs.groupType = *groupTypePtr;
    OP_LOGI(context->GetNodeName(), "Attr groupType = %ld", gmmAttrs.groupType);

    const bool *transposeWPtr = attrs->GetAttrPointer<bool>(GMM_INDEX_ATTR_TRANSPOSE_W);
    OP_CHECK_NULL_WITH_CONTEXT(context, transposeWPtr);
    gmmAttrs.transposeWeight = *transposeWPtr;
    OP_LOGI(context->GetNodeName(), "Attr isWeightTransposed = %d", gmmAttrs.transposeWeight);

    const bool *transposeXPtr = attrs->GetAttrPointer<bool>(GMM_INDEX_ATTR_TRANSPOSE_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, transposeXPtr);
    gmmAttrs.transposeX = *transposeXPtr;
    OP_LOGI(context->GetNodeName(), "Attr isXTransposed = %d", gmmAttrs.transposeX);

    const int64_t *activeType = attrs->GetInt(GMM_INDEX_ATTR_ACT_TYPE);
    OP_CHECK_NULL_WITH_CONTEXT(context, activeType);
    gmmAttrs.activeType = *activeType;
    OP_LOGI(context->GetNodeName(), "Attr activeType = %ld", gmmAttrs.activeType);
    return GRAPH_SUCCESS;
}

static ge::graphStatus CheckAttrs(gert::InferShapeContext* context, GMMAttrs& gmmAttrs) {
    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    OP_CHECK_IF(CheckSplitItem(gmmAttrs.splitItem) != GRAPH_SUCCESS, OP_LOGE(context->GetNodeName(),
              "Invalid splitItem, which can only be one of 0/1/2/3."), return GRAPH_FAILED);
    OP_CHECK_IF(CheckGroupType(context, gmmAttrs.groupType) != GRAPH_SUCCESS,
              OP_LOGE(context->GetNodeName(), "Invalid groupType."), return GRAPH_FAILED);
    const int64_t* activeType = attrs->GetInt(GMM_INDEX_ATTR_ACT_TYPE);
    OP_CHECK_NULL_WITH_CONTEXT(context, activeType);
    OP_CHECK_IF(*activeType < 0 || *activeType >= static_cast<int64_t>(GMMActType::END_ACT_TYPE_ENUM),
              OP_LOGE(context->GetNodeName(), "activeType must be no less than 0 and smaller than 6"),
              return GRAPH_FAILED);
    OP_CHECK_IF(*activeType == static_cast<int64_t>(GMMActType::GMM_ACT_TYPE_GELU_ERR_FUNC),
              OP_LOGE(context->GetNodeName(), "Activation function not support GELU_ERR_FUNC now."),
              return GRAPH_FAILED);
    return GRAPH_SUCCESS;
}

static ge::graphStatus GetNumOfInputs(const gert::InferShapeContext* context, size_t& numX,
                                      size_t& numWeight, int64_t& lenGroupList) {
    ge::graphStatus res = GRAPH_SUCCESS;
    const gert::Shape* shape = nullptr;
    while (true) {
        shape = context->GetDynamicInputShape(GMM_INDEX_IN_X, numX);
        if (shape == nullptr) {  // last shape
            break;
        }
        for (size_t i = 0; i < shape->GetDimNum(); ++i) {
            if (shape->GetDim(i) < 0) {  // shape dim cannot be smaller than 0
                res = GRAPH_FAILED;
                break;
            }
        }
        ++numX;
    }
    OP_LOGI(context->GetNodeName(), "numX = %lu", numX);

    while (true) {
        shape = context->GetDynamicInputShape(GMM_INDEX_IN_WEIGHT, numWeight);
        if (shape == nullptr) {  // last shape
            break;
        }
        for (size_t i = 0; i < shape->GetDimNum(); ++i) {
            if (shape->GetDim(i) < 0) {  // shape dim cannot be smaller than 0
                res = GRAPH_FAILED;
                break;
            }
        }
        ++numWeight;
    }
    OP_LOGI(context->GetNodeName(), "numWeight = %lu", numWeight);

    const gert::Tensor* groupListTensor = context->GetOptionalInputTensor(GMM_INDEX_IN_GROUP_LIST);
    if (groupListTensor != nullptr) {
        lenGroupList = groupListTensor->GetStorageShape().GetDim(0);  // groupListType 2 shape is [e, 2]
        if (lenGroupList < 0) {  // lenGroupList cannot be smaller than 0
            res = GRAPH_FAILED;
        }
    }
    OP_LOGI(context->GetNodeName(), "lenGroupList = %ld", lenGroupList);

    return res;
}

static int64_t GetDim0(const gert::InferShapeContext* context, bool isXTransposed, size_t numX, size_t xDimM) {
    int64_t dim0 = 0;
    if (isXTransposed) {
        const gert::Shape* x0Shape = context->GetDynamicInputShape(GMM_INDEX_IN_X, 0);
        dim0 = (x0Shape == nullptr ? 0 : x0Shape->GetDim(xDimM));
    } else {
        for (size_t idx = 0; idx < numX; ++idx) {
            const gert::Shape* xShape = context->GetDynamicInputShape(GMM_INDEX_IN_X, idx);
            int64_t tmpDim0 = (xShape == nullptr ? 0 : xShape->GetDim(0));
            if(tmpDim0 >= 0) {
                dim0 += tmpDim0;
            } else {
                return tmpDim0;
            }
        }
    }

    return dim0;
}

static bool inline IsNonEmpty(const gert::Shape* shape) {
    return (shape != nullptr && !(shape->GetDimNum() == 1 && shape->GetDim(0) == 0));
}

static ge::graphStatus IsGmmAntiQuantEmpty(gert::InferShapeContext* context) {
    OP_CHECK_IF(!IsTensorListNullOrEmpty(context, GMM_INDEX_IN_ANTIQUANT_SCALE),
              OP_LOGE(context->GetNodeName(), "antiquantScale is not null or empty!"),
              return GRAPH_FAILED);
    OP_CHECK_IF(!IsTensorListNullOrEmpty(context, GMM_INDEX_IN_ANTIQUANT_OFFSET),
              OP_LOGE(context->GetNodeName(), "antiquantOffset is not null or empty!"),
              return GRAPH_FAILED);
    return GRAPH_SUCCESS;
}

static ge::graphStatus IsGmmQuantEmpty(gert::InferShapeContext* context) {
    OP_CHECK_IF(!IsTensorListNullOrEmpty(context, GMM_INDEX_IN_SCALE),
              OP_LOGE(context->GetNodeName(), "scale is not null or empty!"),
              return GRAPH_FAILED);
    OP_CHECK_IF(!IsTensorListNullOrEmpty(context, GMM_INDEX_IN_OFFSET),
              OP_LOGE(context->GetNodeName(), "offset is not null or empty!"),
              return GRAPH_FAILED);
    const gert::Shape* pertokenQuantScale0Shape = context->GetOptionalInputShape(GMM_INDEX_IN_PERTOKEN_SCALE);
    OP_CHECK_IF(IsNonEmpty(pertokenQuantScale0Shape),
              OP_LOGE(context->GetNodeName(), "pertokenQuant scale is not null or empty!"),
              return GRAPH_FAILED);
    return GRAPH_SUCCESS;
}

static ge::graphStatus CheckNonQuant(gert::InferShapeContext* context) {
    OP_CHECK_IF(IsGmmQuantEmpty(context) != GRAPH_SUCCESS,
              OP_LOGE(context->GetNodeName(), "Detected nonquant, but quant inputs is not empty!"),
              return GRAPH_FAILED);
    OP_CHECK_IF(IsGmmAntiQuantEmpty(context) != GRAPH_SUCCESS,
              OP_LOGE(context->GetNodeName(), "Detected nonquant, but antiquant inputs is not empty!"),
              return GRAPH_FAILED);
    return GRAPH_SUCCESS;
}

static ge::graphStatus GetGroupSize(const gert::InferShapeContext* context, GMMParamsInfo& paramsInfo) {
    size_t groupNum = 1;
    // all case allows 1024 in infershape, specific check in tiling
    size_t maxGroupNum = GMM_MAX_GROUP_LIST_SIZE_TENSOR;
    if (paramsInfo.numX > 1UL) {
        groupNum = paramsInfo.numX;
    } else if (paramsInfo.numWeight > 1UL) {
        groupNum = paramsInfo.numWeight;
    } else if (paramsInfo.numY > 1UL) {
        groupNum = paramsInfo.numY;
    } else if (paramsInfo.lenGroupList > 0) {
        groupNum = static_cast<size_t>(paramsInfo.lenGroupList);
    }
    OP_CHECK_IF(groupNum > maxGroupNum,
              OP_LOGE(context->GetNodeName(), "groupNum[%zu] is larger than %zu.",
                        groupNum, maxGroupNum),
              return GRAPH_FAILED);
    paramsInfo.groupNum = groupNum;
    return GRAPH_SUCCESS;
}

static graphStatus CheckDimNumAndPerGroupNum(const gert::InferShapeContext* context, bool isAntiquantInt4,
    const std::tuple<size_t, size_t, int64_t>& dimData, const gert::Shape* tensorShape, const std::string& tensorType) {
    size_t tensorDimNum = std::get<0>(dimData);
    size_t expectedDimNum = std::get<1>(dimData);  // 1: the sceond element
    int64_t weightKDimValue = std::get<2>(dimData);  // 2: the third element
    if (isAntiquantInt4) {
        if (tensorDimNum == expectedDimNum) {
            int64_t perGroupNum = tensorShape->GetDim(tensorDimNum - 2);  // 2: the last 2-th index
            OP_CHECK_IF(!(perGroupNum > 0 && weightKDimValue % perGroupNum == 0),
                      OP_LOGE(context->GetNodeName(), "perGroupNum must be larger than 0, and can evenly divided "
                                "by K[%ld] in A16W4-pergroup case, but now perGroupNum is %ld.", weightKDimValue, perGroupNum),
                      return GRAPH_FAILED);
        } else {
            OP_CHECK_IF(tensorDimNum != expectedDimNum - 1,
                      OP_LOGE(context->GetNodeName(), "%s Dim must be %zu for in perchannel case or "
                                "%zu for pergroup case in A16W4, but now is %zu.",
                                tensorType.c_str(), expectedDimNum - 1, expectedDimNum, tensorDimNum),
                      return GRAPH_FAILED);
        }
    } else {
        OP_CHECK_IF(tensorDimNum != expectedDimNum - 1,
                  OP_LOGE(context->GetNodeName(), "%s Dim must be %zu, but now is %zu.",
                            tensorType.c_str(), expectedDimNum - 1, tensorDimNum),
                  return GRAPH_FAILED);
    }
    return GRAPH_SUCCESS;
}

static ge::graphStatus CheckOptionalTensorList(gert::InferShapeContext* context, const std::string tensorType,
                                               const GMMParamsInfo& paramsInfo, const GMMAttrs& gmmAttrs, size_t nodeIdx) {
    // check bias，scale, antiquant scale or antiquant offset's size，tensor dimension and shape.
    const size_t& groupNum = paramsInfo.groupNum;
    size_t tensorSize = 0;
    while (context->GetDynamicInputShape(nodeIdx, tensorSize) != nullptr) {
        ++tensorSize;
    }
    uint64_t weightGroupedSize = static_cast<uint64_t>(paramsInfo.numWeight);
    const int64_t& groupType = gmmAttrs.groupType;
    auto shape = context->GetDynamicInputShape(GMM_INDEX_IN_WEIGHT, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context, shape);
    uint64_t weightNDimIdx = shape->GetDimNum() - (gmmAttrs.transposeWeight ? 2 : 1);
    auto tensor0Shape = context->GetDynamicInputShape(nodeIdx, 0);
    // tensorList size should equals with weight's size
    OP_CHECK_IF(tensorSize != weightGroupedSize, OP_LOGE(context->GetNodeName(),
              "%s size[%lu] must be equal with weight size[%lu].", tensorType.c_str(), tensorSize, weightGroupedSize), return GRAPH_FAILED);
    bool isSingleWeight = (weightGroupedSize == 1 && groupType != GMM_NO_SPLIT);
    auto w0Desc = context->GetDynamicInputDesc(GMM_INDEX_IN_WEIGHT, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context, w0Desc);
    bool isAntiquantInt4 = (w0Desc->GetDataType() == DT_INT4 && tensorType.find("antiquant") != std::string::npos);
    if (isSingleWeight) {  // In this case, nodeIdx should have only single tensor, its dim should be 2.
        OP_CHECK_IF(IsTensorListNullOrEmpty(context, nodeIdx), OP_LOGE(context->GetNodeName(),
                  "%s must not be nullptr or empty, but now is nullptr or empty.", tensorType.c_str()), return GRAPH_FAILED);
        size_t tensorDimNum = tensor0Shape->GetDimNum();
        int64_t k = shape->GetDim(shape->GetDimNum() - (gmmAttrs.transposeWeight ? 1 : 2));  // 2: axis index
        // 3: shape is (E,G,N),G is the perGroupNum
        OP_CHECK_IF(CheckDimNumAndPerGroupNum(context, isAntiquantInt4, {tensorDimNum, 3, k}, tensor0Shape, tensorType) != GRAPH_SUCCESS,
                  OP_LOGE(context->GetNodeName(), "CheckDimNumAndPerGroupNum failed."), return GRAPH_FAILED);
        OP_CHECK_IF(static_cast<size_t>(tensor0Shape->GetDim(0)) != groupNum, OP_LOGE(context->GetNodeName(), "%s batch size[%ld] should be "
                  "euqal with groupList length[%lu].", tensorType.c_str(), tensor0Shape->GetDim(0), groupNum), return GRAPH_FAILED);
        // tensor's N axis size should equal with weight's N axis.
        int64_t weightNDimValue = context->GetDynamicInputShape(GMM_INDEX_IN_WEIGHT, 0)->GetDim(weightNDimIdx);
        int64_t tensorNDimValue = tensor0Shape->GetDim(tensorDimNum - 1);
        OP_CHECK_IF(tensorNDimValue != weightNDimValue, OP_LOGE(context->GetNodeName(),
                  "NDim[%ld] of %s should be equal with NDim[%ld] of weight.", tensorNDimValue, tensorType.c_str(), weightNDimValue),
                  return GRAPH_FAILED);
    } else {
        for (uint64_t i = 0; i < groupNum; i++) {
            auto tensorShape = context->GetDynamicInputShape(nodeIdx, i);
            OP_CHECK_IF(tensorShape == nullptr, OP_LOGE(context->GetNodeName(),
                      "%s[%lu] must not be nullptr, but now is nullptr.", tensorType.c_str(), i), return GRAPH_FAILED);
            // check each of tensor's dim to be 1
            size_t tensorDimNum = tensorShape->GetDimNum();
            auto wShape = context->GetDynamicInputShape(GMM_INDEX_IN_WEIGHT, i);
            OP_CHECK_NULL_WITH_CONTEXT(context, wShape);
            int64_t k = wShape->GetDim(wShape->GetDimNum() - (gmmAttrs.transposeWeight ? 1 : 2));  // 2: axis index
            // 2: shape is (G,N), G is the perGroupNum
            OP_CHECK_IF(CheckDimNumAndPerGroupNum(context, isAntiquantInt4, {tensorDimNum, 2, k}, tensorShape, tensorType) != GRAPH_SUCCESS,
                      OP_LOGE(context->GetNodeName(), "CheckDimNumAndPerGroupNum failed."), return GRAPH_FAILED);
            int64_t weightNDimValue = wShape->GetDim(weightNDimIdx);
            int64_t tensorNDimValue = tensorShape->GetDim(tensorDimNum - 1);
            OP_CHECK_IF(tensorNDimValue != weightNDimValue, OP_LOGE(context->GetNodeName(), "NDim[%ld] of %s[%lu] should be equal with "
                      "NDim[%ld] of weight[%lu].", tensorNDimValue, tensorType.c_str(), i, weightNDimValue, i), return GRAPH_FAILED);
        }
    }
    return GRAPH_SUCCESS;
}

static ge::graphStatus CheckPerTokenScale(const gert::InferShapeContext* context, const GMMParamsInfo& paramsInfo) {
    // check pertoken scale's size, tensor dimension and shape
    const size_t& xGroupedSize = paramsInfo.numX;
    const size_t& weightGroupedSize = paramsInfo.numWeight;
    const size_t& yGroupedSize = paramsInfo.numY;
    uint64_t xMDimIdx = 0;
    // check pertoken scale's size to be equal with x's
    if ((xGroupedSize == 1UL) && (yGroupedSize == 1UL)) {
        auto perTokenScale0Shape = context->GetOptionalInputShape(GMM_INDEX_IN_PERTOKEN_SCALE);
        OP_CHECK_IF(perTokenScale0Shape == nullptr,
                  OP_LOGE(context->GetNodeName(), "perTokenScaleOptional must not be nullptr, but now is nullptr."),
                  return GRAPH_FAILED);
        // tensor dimension of pertoken_scale should be 1.
        size_t tensorDimNum = perTokenScale0Shape->GetDimNum();
        OP_CHECK_IF(tensorDimNum != 1,
                  OP_LOGE(context->GetNodeName(),
                            "perTokenScaleOptional dim num must be 1 when x is single tensor, but now is %zu.", tensorDimNum),
                  return GRAPH_FAILED);
        // check pertoken_scale's tensor shape size to be equal with M axis size of x.
        auto xShape = context->GetDynamicInputShape(GMM_INDEX_IN_X, 0);
        OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
        int64_t xMDimValue = xShape->GetDim(xMDimIdx);
        int64_t tensorMDimValue = perTokenScale0Shape->GetDim(tensorDimNum - 1);
        OP_CHECK_IF(tensorMDimValue != xMDimValue,
                  OP_LOGE(context->GetNodeName(),
                            "MDim[%ld] of perTokenScaleOptional should be equal with MDim[%ld] of x.",
                            tensorMDimValue, xMDimValue),
                  return GRAPH_FAILED);
    } else {
        OP_LOGE(context->GetNodeName(), "per-token quant case is only supported "
                  "when x, weight and y are all single tensor, but now x size is %zu, weight size is %zu, y size is %zu",
                  xGroupedSize, weightGroupedSize, yGroupedSize);
        return GRAPH_FAILED;
    }
    return GRAPH_SUCCESS;
}

static ge::graphStatus CheckGroupedMatmulQuant(gert::InferShapeContext* context, const GMMAttrs& gmmAttrs,
                                               const GMMParamsInfo& paramsInfo) {
    OP_CHECK_IF(paramsInfo.platform == PlatformID::ASCEND310P,
              OP_LOGE(context->GetNodeName(), "quant cases do not support on Ascend310P."),
              return GRAPH_FAILED);
    OP_CHECK_IF(gmmAttrs.groupType == GMM_SPLIT_K,
              OP_LOGE(context->GetNodeName(), "quant cases do not support splited axis is K."),
              return GRAPH_FAILED);
    OP_CHECK_IF(!IsTensorListNullOrEmpty(context, GMM_INDEX_IN_OFFSET),
              OP_LOGE(context->GetNodeName(), "offset must be nullptr in quant, but now is not nullptr."),
              return GRAPH_FAILED);
    if (gmmAttrs.outputDtype != GMM_OUT_DTYPE_INT32) {  // output dtype is int32, this scene does not need scale
        OP_CHECK_IF(IsTensorListNullOrEmpty(context, GMM_INDEX_IN_SCALE),
                OP_LOGE(context->GetNodeName(), "scale must not be nullptr in quant, but now is nullptr."),
                return GRAPH_FAILED);
        OP_CHECK_IF(CheckOptionalTensorList(context, "scale", paramsInfo, gmmAttrs, GMM_INDEX_IN_SCALE) != GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "Invalid scale."),
                return GRAPH_FAILED);
    }
    bool isPerTokenQuant = context->GetOptionalInputShape(GMM_INDEX_IN_PERTOKEN_SCALE) != nullptr;
    if (isPerTokenQuant) {
        OP_CHECK_IF(CheckPerTokenScale(context, paramsInfo) != GRAPH_SUCCESS,
                  OP_LOGE(context->GetNodeName(), "Check perTokenScale failed!"),
                  return GRAPH_FAILED);
    }
    OP_CHECK_IF(IsGmmAntiQuantEmpty(context) != GRAPH_SUCCESS,
              OP_LOGE(context->GetNodeName(), "Detected quant, but antiquant inputs is not empty!"),
              return GRAPH_FAILED);
    return GRAPH_SUCCESS;
}
static bool isA8W4AsymmetricQuant(const gert::InferShapeContext* context) {
    auto offsetShape = context->GetDynamicInputShape(GMM_INDEX_IN_OFFSET, 0);
    if (offsetShape == nullptr) {
        return false;
    }
    size_t offsetDimNum = offsetShape->GetDimNum();
    if (offsetDimNum != GMM_A8W4_OFFSET_DIM_NUM) {
        return false;
    }
    auto weightShape = context->GetDynamicInputShape(GMM_INDEX_IN_WEIGHT, 0);
    if (offsetShape->GetDim(0) == weightShape->GetDim(0) && offsetShape->GetDim(1) == 1
        && offsetShape->GetDim(GMM_A8W4_OFFSET_DIM_NUM - 1) == weightShape->GetDim(GMM_A8W4_OFFSET_DIM_NUM - 1)) {
        return true;
    }
    return false;
}
static ge::graphStatus CheckA8W4AsymQuantParams(gert::InferShapeContext* context, const GMMParamsInfo& paramsInfo) {
    OP_CHECK_IF(paramsInfo.platform == PlatformID::ASCEND310P,
              OP_LOGE(context->GetNodeName(), "quant cases do not support on Ascend310P."),
              return GRAPH_FAILED);
    auto xShape = context->GetDynamicInputShape(GMM_INDEX_IN_X, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    auto weightShape = context->GetDynamicInputShape(GMM_INDEX_IN_WEIGHT, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightShape);
    auto biasShape = context->GetDynamicInputShape(GMM_INDEX_IN_BIAS, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context, biasShape);
    auto scaleShape = context->GetDynamicInputShape(GMM_INDEX_IN_SCALE, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context, scaleShape);
    size_t biasDimNum = biasShape->GetDimNum();
    size_t scaleDimNum = scaleShape->GetDimNum();
    int64_t e = weightShape->GetDim(0);
    int64_t n = weightShape->GetDim(GMM_A8W4_OFFSET_DIM_NUM - 1);
    OP_CHECK_IF(IsGmmAntiQuantEmpty(context) != GRAPH_SUCCESS,
              OP_LOGE(context->GetNodeName(), "antiquant inputs is not empty!"),
              return GRAPH_FAILED);
    OP_CHECK_IF(biasDimNum != GMM_A8W4_BIAS_DIM_NUM || biasShape->GetDim(0) != e || biasShape->GetDim(1) != n,
              OP_LOGE(context->GetNodeName(), "bias shape is invalid, must be (e,n)."),
              return GRAPH_FAILED);
    auto isScaleInvalid = !(scaleDimNum == GMM_A8W4_OFFSET_DIM_NUM && scaleShape->GetDim(0) == e
            && scaleShape->GetDim(1) == 1 && scaleShape->GetDim(GMM_A8W4_OFFSET_DIM_NUM - 1) == n);
    OP_CHECK_IF(isScaleInvalid, OP_LOGE(context->GetNodeName(), "scale shape is invalid, must be (e,1,n)."),
            return GRAPH_FAILED);
    return GRAPH_SUCCESS;
}

static int64_t GetPergroupSize(const GMMAttrs& gmmAttrs, bool isSingleWeight,
                               const gert::Shape* wShape, const gert::Shape* shape) {
  int64_t pergroupSize = 0;
  size_t shapeDimNum = shape->GetDimNum();
  if (isSingleWeight) {  // antiquant param shape (E, N), (E, G, N)
    if (shapeDimNum > GMM_SEPARATED_WEIGHT_DIM) {
      int64_t k = gmmAttrs.transposeWeight ?  wShape->GetDim(2) : wShape->GetDim(1);  // 2: the k axis index
      pergroupSize = k / shape->GetDim(shapeDimNum - 2);  // 2: the last 2-th index
    }
  } else {  //  antiquant param shape (N), (G, N)
    if (shapeDimNum > 1UL) {
      int64_t k = gmmAttrs.transposeWeight ? wShape->GetDim(1): wShape->GetDim(0);
      pergroupSize = k / shape->GetDim(shapeDimNum - 2);  // 2: the last 2-th index
    }
  }
  return pergroupSize;
}

static ge::graphStatus CheckGroupedMatmulAntiQuantGroupSize(gert::InferShapeContext *context, const GMMAttrs &gmmAttrs,
                                                            const GMMParamsInfo &paramsInfo, bool hasAntiquantOffset)
{
    auto antiquantScale0Shape = context->GetDynamicInputShape(GMM_INDEX_IN_ANTIQUANT_SCALE, 0);
    auto dimNum = antiquantScale0Shape->GetDimNum();
    bool isSingleWeight = ((paramsInfo.numWeight == 1UL) && (gmmAttrs.groupType != GMM_NO_SPLIT));
    int64_t pergroupSize = GetPergroupSize(gmmAttrs, isSingleWeight,
                                           context->GetDynamicInputShape(GMM_INDEX_IN_WEIGHT, 0), antiquantScale0Shape);
    OP_CHECK_IF(gmmAttrs.transposeWeight && pergroupSize % 2 != 0,  // 2: a factor
                OP_LOGE(context->GetNodeName(),
                        "pergroupSize should be even when weight is transposed"
                        "in A16W4-pergroup case, but now is %ld",
                        pergroupSize),
                return GRAPH_FAILED);
    for (size_t i = 0;; ++i) {
        auto antiquantScaleShape = context->GetDynamicInputShape(GMM_INDEX_IN_ANTIQUANT_SCALE, i);
        if (antiquantScaleShape == nullptr) {
            break;
        }
        size_t antiquantScaleDimNum = antiquantScaleShape->GetDimNum();
        OP_CHECK_IF(antiquantScaleDimNum != dimNum,
                    OP_LOGE(context->GetNodeName(), "antiquantScale[%zu] dim num[%zu] is not equal with %zu", i,
                            antiquantScaleDimNum, dimNum),
                    return GRAPH_FAILED);
        auto wShape = context->GetDynamicInputShape(GMM_INDEX_IN_WEIGHT, i);
        int64_t pergroupSizeOfScale = GetPergroupSize(gmmAttrs, isSingleWeight, wShape, antiquantScaleShape);
        OP_CHECK_IF(pergroupSizeOfScale != pergroupSize,
                    OP_LOGE(context->GetNodeName(),
                            "antiquantScale[%zu]'s pergroup size[%ld] "
                            "is not the required value[%ld]",
                            i, pergroupSizeOfScale, pergroupSize),
                    return GRAPH_FAILED);
        if (hasAntiquantOffset) {
            auto antiquantOffsetShape = context->GetDynamicInputShape(GMM_INDEX_IN_ANTIQUANT_OFFSET, i);
            size_t antiquantOffsetDimNum = antiquantOffsetShape->GetDimNum();
            OP_CHECK_IF(antiquantOffsetDimNum != dimNum,
                        OP_LOGE(context->GetNodeName(), "antiquantOffset[%zu] dim num[%zu] is not equal with %zu", i,
                                antiquantOffsetDimNum, dimNum),
                        return GRAPH_FAILED);
            int64_t pergroupSizeOfOffset = GetPergroupSize(gmmAttrs, isSingleWeight, wShape, antiquantOffsetShape);
            OP_CHECK_IF(pergroupSizeOfOffset != pergroupSize,
                        OP_LOGE(context->GetNodeName(),
                                "antiquantOffset[%zu]'s pergroup size[%ld]"
                                "is not the required value[%ld]",
                                i, pergroupSizeOfOffset, pergroupSize),
                        return GRAPH_FAILED);
        }
    }
    return GRAPH_SUCCESS;
}

static ge::graphStatus CheckGroupedMatmulAntiQuantForShape(gert::InferShapeContext* context, const GMMAttrs& gmmAttrs, const GMMParamsInfo& paramsInfo) {
    OP_CHECK_IF(paramsInfo.platform == PlatformID::ASCEND310P, OP_LOGE(context->GetNodeName(),
              "antiquant cases do not support on Ascend310P."), return GRAPH_FAILED);
    OP_CHECK_IF(gmmAttrs.groupType == GMM_SPLIT_K, OP_LOGE(context->GetNodeName(), "antiquant cases do not support splited axis is K."),
              return GRAPH_FAILED);
    OP_CHECK_IF(IsTensorListNullOrEmpty(context, GMM_INDEX_IN_ANTIQUANT_SCALE),
              OP_LOGE(context->GetNodeName(), "antiquantScale must not be nullptr in antiquant, but now is nullptr or empty."),
              return GRAPH_FAILED);
    // check antiquantScale and antiquantOffset's tensor shape
    OP_CHECK_IF(CheckOptionalTensorList(context, "antiquantScale", paramsInfo, gmmAttrs, GMM_INDEX_IN_ANTIQUANT_SCALE) != GRAPH_SUCCESS,
              OP_LOGE(context->GetNodeName(), "Invalid antiquantScale"),
              return GRAPH_FAILED);
    auto w0Desc = context->GetDynamicInputDesc(GMM_INDEX_IN_WEIGHT, 0);
    bool hasAntiquantOffset = !IsTensorListNullOrEmpty(context, GMM_INDEX_IN_ANTIQUANT_OFFSET);
    OP_CHECK_IF(w0Desc->GetDataType() != DT_INT4 && !hasAntiquantOffset,
              OP_LOGE(context->GetNodeName(), "antiquantOffset must not be nullptr in antiquant, but now is nullptr or empty."),
              return GRAPH_FAILED);
    if (hasAntiquantOffset) {
        OP_CHECK_IF(CheckOptionalTensorList(context, "antiquantOffset", paramsInfo, gmmAttrs, GMM_INDEX_IN_ANTIQUANT_OFFSET) != GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "Invalid antiquantOffset"),
                return GRAPH_FAILED);
    }
    // check perGroupSize
    if (w0Desc->GetDataType() == DT_INT4) {
        OP_CHECK_IF(
            CheckGroupedMatmulAntiQuantGroupSize(context, gmmAttrs, paramsInfo, hasAntiquantOffset) != GRAPH_SUCCESS,
            OP_LOGE(context->GetNodeName(), "Invalid antiquant group size."), return GRAPH_FAILED);
    }
    OP_CHECK_IF(IsGmmQuantEmpty(context) != GRAPH_SUCCESS, OP_LOGE(context->GetNodeName(),
              "Detected antiquant, but quant inputs is not empty!"), return GRAPH_FAILED);
    return GRAPH_SUCCESS;
}

static ge::graphStatus CheckQuantParams(gert::InferShapeContext* context, const GMMAttrs& gmmAttrs, GMMParamsInfo& paramsInfo) {
    auto x0Desc = context->GetDynamicInputDesc(GMM_INDEX_IN_X, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context, x0Desc);
    DataType xDtype = x0Desc->GetDataType();
    auto w0Desc = context->GetDynamicInputDesc(GMM_INDEX_IN_WEIGHT, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context, w0Desc);
    DataType weightDtype = w0Desc->GetDataType();
    if (xDtype == DataType::DT_INT8 && weightDtype == DataType::DT_INT4) {
        if (!isA8W4AsymmetricQuant(context)) {
            return GRAPH_SUCCESS;
        }
        return CheckA8W4AsymQuantParams(context, paramsInfo);
    }
    if ((xDtype == DataType::DT_BF16 || xDtype == DataType::DT_FLOAT16 ||
        xDtype == DataType::DT_FLOAT) && xDtype == weightDtype) {
        // nonquant
        return CheckNonQuant(context);
    }
    if (xDtype == DataType::DT_INT8 && weightDtype == DataType::DT_INT8) {
        // quant
        return CheckGroupedMatmulQuant(context, gmmAttrs, paramsInfo);
    }
    if ((xDtype == DataType::DT_BF16 || xDtype == DataType::DT_FLOAT16) &&
        (weightDtype == DataType::DT_INT8 || weightDtype == DataType::DT_INT4)) {
        // antiquant
        return CheckGroupedMatmulAntiQuantForShape(context, gmmAttrs, paramsInfo);
    }
    return GRAPH_SUCCESS;
}
static ge::graphStatus CheckFunctionParamsForShape(gert::InferShapeContext* context, const GMMAttrs& gmmAttrs,
                                                   GMMParamsInfo& paramsInfo) {
    if (context == nullptr) {
        return GRAPH_FAILED;
    }
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optionalInfo;
    auto ret = fe::PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platformInfo, optionalInfo);
    if (ret != ge::GRAPH_SUCCESS) {
        paramsInfo.platform = PlatformID::UNKNOWN;
        OP_LOGW(context->GetNodeName(), "Cannot get platform info!");
        return GRAPH_SUCCESS;
    } else {
        paramsInfo.platform = (optionalInfo.soc_version.find("310P") != std::string::npos) ?
                                PlatformID::ASCEND310P : (optionalInfo.soc_version.find("950") != std::string::npos) ?
                                PlatformID::ASCEND950 : PlatformID::ASCEND910B;
    }
    OP_CHECK_IF(CheckQuantParams(context, gmmAttrs, paramsInfo) != GRAPH_SUCCESS,
                  OP_LOGE(context->GetNodeName(), "CheckQuantParams failed!"),
                  return GRAPH_FAILED);
    return GRAPH_SUCCESS;
}
static ge::graphStatus CheckDimNumAndGroupListNoSplitAndFormat(const gert::InferShapeContext* context,
    uint64_t tensorListLength, const size_t numWeight) {
    // when groupList is not empty, check its size equal with the length of x.
    auto groupTensorOptionalShape = context->GetOptionalInputShape(GMM_INDEX_IN_GROUP_LIST);
    if (groupTensorOptionalShape != nullptr) {
        OP_CHECK_IF(groupTensorOptionalShape->GetDim(0) != static_cast<int64_t>(tensorListLength),
                  OP_LOGE(context->GetNodeName(), "Size of groupList(tensor) %ld should be equal to size of x %lu.",
                            groupTensorOptionalShape->GetDim(0), tensorListLength),
                  return GRAPH_FAILED);
    }
    auto wShape = context->GetDynamicInputShape(GMM_INDEX_IN_WEIGHT, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context, wShape);
    // check dimension
    for (size_t i = 0; i < tensorListLength; ++i) {
        auto xShape = context->GetDynamicInputShape(GMM_INDEX_IN_X, i);
        OP_CHECK_IF(xShape == nullptr,
                  OP_LOGE(context->GetNodeName(), "x[%lu] is null, which is not supported.", i),
                  return GRAPH_FAILED);
        if (numWeight > 1) {
            wShape = context->GetDynamicInputShape(GMM_INDEX_IN_WEIGHT, i);
            OP_CHECK_NULL_WITH_CONTEXT(context, wShape);
            size_t weightDimNum = wShape->GetDimNum();
            OP_CHECK_IF(weightDimNum != GMM_SEPARATED_WEIGHT_DIM,
                      OP_LOGE(context->GetNodeName(),
                                "weight[%lu] dimNum is %lu , but only support 2 when weight separated.",
                                i, weightDimNum),
                      return GRAPH_FAILED);
        }
        size_t xDimNum = xShape->GetDimNum();
        OP_CHECK_IF(xDimNum > GMM_MAX_FM_DIM || xDimNum < GMM_MIN_FM_DIM,
                  OP_LOGE(context->GetNodeName(), "x[%lu] dimNum is %lu , but only support 2-6.", i, xDimNum),
                  return GRAPH_FAILED);
    }
    return GRAPH_SUCCESS;
}

static ge::graphStatus TensorType2NodeId(const std::vector<std::string>& tensorType, std::vector<int64_t>& nodeIdx) {
    if (nodeIdx.size() > tensorType.size()) {
        return GRAPH_FAILED;
    }
    for (size_t i(0); i < nodeIdx.size(); ++i) {
        if (tensorType[i] == "x") {
            nodeIdx[i] = GMM_INDEX_IN_X;
        } else if (tensorType[i] == "weight") {
            nodeIdx[i] = GMM_INDEX_IN_WEIGHT;
        } else if (tensorType[i] == "y") {
            nodeIdx[i] = GMM_INDEX_OUT_Y;
        } else {
            return GRAPH_FAILED;
        }
    }
    return GRAPH_SUCCESS;
}

static ge::graphStatus CheckDimNum(gert::InferShapeContext* context, uint64_t tensorListLength,
                                            const size_t expectedDimNum, const std::string tensorType) {
    int64_t nodeIdx = 0;
    if (tensorType == "x") {
        nodeIdx = static_cast<int64_t>(GMM_INDEX_IN_X);
    } else if (tensorType == "weight") {
        nodeIdx = static_cast<int64_t>(GMM_INDEX_IN_WEIGHT);
    } else if (tensorType == "y") {
        nodeIdx = static_cast<int64_t>(GMM_INDEX_OUT_Y);
    } else {
        return GRAPH_FAILED;
    }
    const gert::Shape* shape;
    for (size_t i = 0; i < tensorListLength; ++i) {
        if (tensorType == "y") {
            shape = context->GetOutputShape(nodeIdx + i);
        } else {
            shape = context->GetDynamicInputShape(nodeIdx, i);
        }
        OP_CHECK_IF(shape == nullptr,
                  OP_LOGE(context->GetNodeName(), "%s[%lu] is null, which is not supported.", tensorType.c_str(), i),
                  return GRAPH_FAILED);
        size_t dimNum = shape->GetDimNum();
        OP_CHECK_IF(dimNum != expectedDimNum,
                  OP_LOGE(context->GetNodeName(), "%s[%lu] dim num should be %lu in this case, but now is %lu.",
                         tensorType.c_str(), i, expectedDimNum, dimNum),
                  return GRAPH_FAILED);
    }
    return GRAPH_SUCCESS;
}

static ge::graphStatus CheckWeightShapeInnerAxisEven(const gert::InferShapeContext* context, const size_t weightSize,
                                                     const int64_t innerAxisDimId) {
    auto w0Desc = context->GetDynamicInputDesc(GMM_INDEX_IN_WEIGHT, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context, w0Desc);
    DataType wDtype = w0Desc->GetDataType();
    if (wDtype == DataType::DT_INT4) {
        for (size_t i = 0; i < weightSize; ++i) {
            auto wShape = context->GetDynamicInputShape(GMM_INDEX_IN_WEIGHT, i);
            OP_CHECK_NULL_WITH_CONTEXT(context, wShape);
            int64_t n = wShape->GetDim(innerAxisDimId);
            OP_CHECK_IF(n % 2 != 0,
                      OP_LOGE(context->GetNodeName(), "w[%zu] dim %ld value %ld should be even when weight is int4 dtype.",
                                i, innerAxisDimId, n),
                      return GRAPH_FAILED);
        }
    }
    return GRAPH_SUCCESS;
}

static ge::graphStatus IsxSizeEqualWithWeightKAxis(const gert::InferShapeContext* context,
    const GMMParamsInfo& paramsInfo, const gert::Shape* wShape, size_t& wKDimIdx, size_t& wNDimIdx) {
    if (paramsInfo.numWeight == 1 && wShape->GetDimNum() > 2) {  // 2: separated tensor's dim
        wKDimIdx += 1UL;
        wNDimIdx += 1UL;
        OP_CHECK_IF(paramsInfo.numX != static_cast<size_t>(wShape->GetDim(0)),
                  OP_LOGE(context->GetNodeName(), "When x and y are separated, and weight is not separated, size of x "
                            "%zu should equal to the first dim of weight tensor %ld.", paramsInfo.numX, wShape->GetDim(0)),
                  return GRAPH_FAILED);
    }
    return GRAPH_SUCCESS;
}

static ge::graphStatus CheckCaseNoSplit(gert::InferShapeContext* context, bool transposeWeight,
                                        const GMMParamsInfo& paramsInfo) {
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optionalInfo;
    auto ret = fe::PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platformInfo, optionalInfo);
    const size_t& xSize = paramsInfo.numX;
    const size_t& weightSize = paramsInfo.numWeight;
    // check group num
    OP_CHECK_IF(xSize != paramsInfo.numY, OP_LOGE(context->GetNodeName(),
              "When y is separated, size of x %lu should equal to size of y %lu.", xSize, paramsInfo.numY), return GRAPH_FAILED);
    OP_CHECK_IF(weightSize != 1 && xSize != weightSize, OP_LOGE(context->GetNodeName(), "When x and weight are separated, "
              "size of x %lu should equal to size of weight %lu.", xSize, weightSize), return GRAPH_FAILED);
    // check dimension
    OP_CHECK_IF(CheckDimNumAndGroupListNoSplitAndFormat(context, xSize, weightSize) != GRAPH_SUCCESS,
              OP_LOGE(context->GetNodeName(), "Dim num or format of tensor in tensor lists or grouplist is invalid."),
              return GRAPH_FAILED);
    // check shape
    auto wShape = context->GetDynamicInputShape(GMM_INDEX_IN_WEIGHT, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context, wShape);
    size_t wKDimIdx = transposeWeight ? 1UL : 0UL;
    size_t wNDimIdx = transposeWeight ? 0UL : 1UL;
    OP_CHECK_IF(IsxSizeEqualWithWeightKAxis(context, paramsInfo, wShape, wKDimIdx, wNDimIdx) != GRAPH_SUCCESS,
              OP_LOGE(context->GetNodeName(), "IsxSizeEqualWithWeightKAxis failed."), return GRAPH_FAILED);
    int64_t weightKDimValue = wShape->GetDim(wKDimIdx);
    int64_t weightNDimValue = wShape->GetDim(wNDimIdx);
    auto w0Desc = context->GetDynamicInputDesc(GMM_INDEX_IN_WEIGHT, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context, w0Desc);
    DataType wDtype = w0Desc->GetDataType();
    // 2: an even factor
    OP_CHECK_IF(wDtype == DataType::DT_INT4 && weightNDimValue % 2 != 0, OP_LOGE(context->GetNodeName(),
              "w[0] dim %lu value %ld should be even when weight is int4 dtype.", wNDimIdx, weightNDimValue),
              return GRAPH_FAILED);
    for (size_t i = 0; i < xSize; i++) {
        auto xShape = context->GetDynamicInputShape(GMM_INDEX_IN_X, i);
        size_t xDimNum = xShape->GetDimNum();
        // check inner axis of x, which should not be larger than 65535
        int64_t xKDimValue = xShape->GetDim(xDimNum - 1);  // x always is not transposed
        if (!(ret == GRAPH_SUCCESS && GmmDavidSupportSoc.count(platformInfo.str_info.short_soc_version) > 0)) {
            OP_CHECK_IF(xKDimValue > GMM_MAX_INNER_AXIS,
                  OP_LOGE(context->GetNodeName(), "x[%lu] dim %lu value %ld should less or equal to %ld.",
                            i, xDimNum - 1, xKDimValue, GMM_MAX_INNER_AXIS),
                  return GRAPH_FAILED);
        }
        if (weightSize > 1UL) {
            wShape = context->GetDynamicInputShape(GMM_INDEX_IN_WEIGHT, i);
            weightKDimValue = wShape->GetDim(wKDimIdx);
            weightNDimValue = wShape->GetDim(wNDimIdx);
            // 2: an even factor
            OP_CHECK_IF(i > 0 && wDtype == DataType::DT_INT4 && weightNDimValue % 2 != 0, OP_LOGE(context->GetNodeName(),
                      "w[%lu] dim %lu value %ld should be even when weight is int4 dtype.", i, wNDimIdx, weightNDimValue),
                      return GRAPH_FAILED);
        }
        OP_CHECK_IF(xKDimValue != weightKDimValue,
                  OP_LOGE(context->GetNodeName(), "x[%lu] dim %lu value %ld should equal to weight[%lu] dim 0 value %ld.",
                            i, xDimNum - 1, xKDimValue, i, weightKDimValue),
                  return GRAPH_FAILED);
        // if weight is not transposed, check N aisx; otherwise, check K axis, which can be skiped
        if (!(ret == GRAPH_SUCCESS && GmmDavidSupportSoc.count(platformInfo.str_info.short_soc_version) > 0)) {
            OP_CHECK_IF(!transposeWeight && weightNDimValue > GMM_MAX_INNER_AXIS,
                  OP_LOGE(context->GetNodeName(), "w[%zu] dim %zu value %ld should less or equal to %ld.",
                            i, wNDimIdx, weightNDimValue, GMM_MAX_INNER_AXIS),
                  return GRAPH_FAILED);
        }
    }
    return GRAPH_SUCCESS;
}

static ge::graphStatus CheckInnerAxisOfTensorList(const gert::InferShapeContext* context, size_t nodeId,
                                                  int64_t innerAxisDimId, size_t checkNum) {
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optionalInfo;
    auto ret = fe::PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platformInfo, optionalInfo);
    for (size_t i = 0; i < checkNum; i++) {
        auto shape = context->GetDynamicInputShape(nodeId, i);
        OP_CHECK_NULL_WITH_CONTEXT(context, shape);
        int64_t innerAxisValue = shape->GetDim(innerAxisDimId);
        if (!(ret == GRAPH_SUCCESS && GmmDavidSupportSoc.count(platformInfo.str_info.short_soc_version) > 0)) {
            OP_CHECK_IF(innerAxisValue > GMM_MAX_INNER_AXIS,
                  OP_LOGE(context->GetNodeName(), "Dim %ld value of %zu-th shape should less or equal to %ld, "
                            "but now is %ld.", innerAxisDimId, i, GMM_MAX_INNER_AXIS, innerAxisValue),
                  return GRAPH_FAILED);
        }
    }
    return GRAPH_SUCCESS;
}

static ge::graphStatus CheckShapeSameLengthTensorList(gert::InferShapeContext* context,
                                                      const std::vector<size_t>& dimIds, const int64_t innerAxisDimId,
                                                      const std::vector<std::string> tensorType, uint64_t groupNum) {
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optionalInfo;
    auto ret = fe::PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platformInfo, optionalInfo);
    std::vector<int64_t> nodeIdx = {0, 0};
    OP_CHECK_IF(TensorType2NodeId(tensorType, nodeIdx) != GRAPH_SUCCESS,
              OP_LOGE(context->GetNodeName(), "TensorType2NodeId failed."),
              return GRAPH_FAILED);
    // check two tensorlist's size to be the same, and tensors to have consistant dimension.
    const gert::Shape* shape;
    for (uint64_t i = 0; i < groupNum; i++) {
        shape = context->GetDynamicInputShape(nodeIdx[0], i);
        OP_CHECK_NULL_WITH_CONTEXT(context, shape);
        int64_t dimValue1 = shape->GetDim(dimIds[0]);
        // tensorType[2] indicates whether check tensorList0's inner axis(innerAxisDimId)
        if (tensorType[2] == "true" && innerAxisDimId > -1) {
            auto shape0 = context->GetDynamicInputShape(nodeIdx[0], i);
            OP_CHECK_NULL_WITH_CONTEXT(context, shape0);
            int64_t innerAxisValue = shape0->GetDim(innerAxisDimId);
            if (!(ret == GRAPH_SUCCESS && GmmDavidSupportSoc.count(platformInfo.str_info.short_soc_version) > 0) && innerAxisValue > GMM_MAX_INNER_AXIS) {
                OP_LOGW(context->GetNodeName(), "Dim %lu value of %s[%lu] should less or equal to %ld,"
                "but now is %ld.", dimIds[0], tensorType[0].c_str(), i, GMM_MAX_INNER_AXIS, innerAxisValue);
            }
        }
        if (tensorType[1] == "y") {
            shape = context->GetOutputShape(nodeIdx[1] + i);
        } else {
            shape = context->GetDynamicInputShape(nodeIdx[1], i);
        }
        OP_CHECK_NULL_WITH_CONTEXT(context, shape);
        int64_t dimValue2 = shape->GetDim(dimIds[1]);
        if(dimValue1 != dimValue2){
            OP_LOGW(context->GetNodeName(),
                "Dim %lu value of %s[%lu] should be equal with dim %lu value of %s[%lu],"
                "but now is %ld and %ld respectively.", dimIds[0], tensorType[0].c_str(),
                i, dimIds[1], tensorType[1].c_str(), i, dimValue1, dimValue2);
        }
    }
    return GRAPH_SUCCESS;
}

static ge::graphStatus CheckShapeDiffLengthTensorList(gert::InferShapeContext* context,
                                                      const std::vector<size_t>& dimIds,
                                                      const int64_t innerAxisdimId,
                                                      const std::vector<std::string> tensorType,
                                                      uint64_t groupNum) {
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optionalInfo;
    auto ret = fe::PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platformInfo, optionalInfo);
    std::vector<int64_t> nodeIdx = {0, 0};
    OP_CHECK_IF(TensorType2NodeId(tensorType, nodeIdx) != GRAPH_SUCCESS,
              OP_LOGE(context->GetNodeName(), "TensorType2NodeId failed."),
              return GRAPH_FAILED);
    // check each tensor's selected dimension size in a multi-tensor tensorlist's to equal with
    // the tensor selected dimension in single-tensor tensorlist.
    // the selected axis is not the split-axis.
    const gert::Shape* singleTensor0;
    if (tensorType[1] == "y") {
        singleTensor0 = context->GetOutputShape(nodeIdx[1]);
    } else {
        singleTensor0 = context->GetDynamicInputShape(nodeIdx[1], 0);
    }
    OP_CHECK_NULL_WITH_CONTEXT(context, singleTensor0);
    int64_t dimValueSingle = singleTensor0->GetDim(dimIds[1]);
    // tensorType[2] indicates whether check single tensorList's inner axis(innerAxisDimId)
    if (tensorType[2] == "true" && innerAxisdimId > -1) {
        int64_t dimValue = singleTensor0->GetDim(innerAxisdimId);
        if (!(ret == GRAPH_SUCCESS && GmmDavidSupportSoc.count(platformInfo.str_info.short_soc_version) > 0)) {
            OP_CHECK_IF(dimValue > GMM_MAX_INNER_AXIS,
                  OP_LOGE(context->GetNodeName(),
                            "Dim %ld value of %s[0] should less or equal to %ld, but now is %ld.",
                            innerAxisdimId, tensorType[1].c_str(), GMM_MAX_INNER_AXIS, dimValue),
                  return GRAPH_FAILED);
        }
    }
    const gert::Shape* longTensor;
    for (uint64_t i = 0; i < groupNum; i++) {
        if (tensorType[0] == "y") {
            longTensor = context->GetOutputShape(nodeIdx[0] + i);
        } else {
            longTensor = context->GetDynamicInputShape(nodeIdx[0], i);
        }
        OP_CHECK_NULL_WITH_CONTEXT(context, longTensor);
        int64_t dimValueLong = longTensor->GetDim(dimIds[0]);
        OP_CHECK_IF(dimValueLong != dimValueSingle,
                  OP_LOGE(context->GetNodeName(),
                            "Dim %lu value of %s[%lu] %ld should be equal with dim %lu value of %s[0] %ld.",
                            dimIds[0], tensorType[0].c_str(), i, dimValueLong,
                            dimIds[1], tensorType[1].c_str(), dimValueSingle),
                  return GRAPH_FAILED);
    }
    return GRAPH_SUCCESS;
}

static ge::graphStatus CheckGroupListCommonTensor(const gert::InferShapeContext* context,
                                                  const bool isRequiredGroupList, const int64_t groupNum) {
    auto groupTensorOptionalShape = context->GetOptionalInputShape(GMM_INDEX_IN_GROUP_LIST);
    bool isNull = groupTensorOptionalShape == nullptr;
    OP_CHECK_IF(isNull && isRequiredGroupList,
              OP_LOGE(context->GetNodeName(), "groupListOptional(tensor) is required in this case, but get nullptr."),
              return GRAPH_FAILED);
    if (isNull) {
        return GRAPH_SUCCESS;
    }
    int64_t groupListSize = groupTensorOptionalShape->GetDim(0);
    OP_CHECK_IF(groupListSize > GMM_MAX_GROUP_LIST_SIZE_TENSOR,
              OP_LOGE(context->GetNodeName(),
                        "When groupList type is tenosr, size of groupList %ld should be less than or equal to %ld.",
                        groupListSize, GMM_MAX_GROUP_LIST_SIZE_TENSOR),
              return GRAPH_FAILED);
    OP_CHECK_IF(!((groupListSize == groupNum && groupNum > 1) || groupNum == 1),
              OP_LOGE(context->GetNodeName(),
                        "When groupList is not null, size of groupList(tensor) %ld should be equal to groupNum %ld.",
                        groupListSize, groupNum),
              return GRAPH_FAILED);
    auto groupListDesc = context->GetOptionalInputDesc(GMM_INDEX_IN_GROUP_LIST);
    OP_CHECK_NULL_WITH_CONTEXT(context, groupListDesc);
    OP_CHECK_IF(groupListDesc->GetDataType() != DataType::DT_INT64,
              OP_LOGE(context->GetNodeName(), "Invalid dtype: Only int64 is supported for groupList, but now is %s.",
                        TypeUtils::DataTypeToAscendString(groupListDesc->GetDataType()).GetString()),
              return GRAPH_FAILED);
    return GRAPH_SUCCESS;
}

static ge::graphStatus SplitMSingleXSingleWeightSingleY(gert::InferShapeContext* context, bool transposeWeight,
                                                        const GMMParamsInfo& paramsInfo) {
    std::vector<std::string> tenorXAndWeight{"x", "weight", "true"};
    // check dimension
    OP_CHECK_IF(CheckDimNum(context, paramsInfo.numX, GMM_MIN_FM_DIM, "x") != GRAPH_SUCCESS,
              OP_LOGE(context->GetNodeName(), "Dim num or format of tensor in tensor list x is invalid."),
              return GRAPH_FAILED);
    OP_CHECK_IF(CheckDimNum(context, paramsInfo.numWeight, GMM_SPLIT_M_SINGLE_WEIGHT_DIM, "weight") != GRAPH_SUCCESS,
              OP_LOGE(context->GetNodeName(), "Dim num or format of tensor in tensor list weight is invalid."),
              return GRAPH_FAILED);
    // check shape, x(m,k), weight(b,k,n), y(m,n)
    int64_t innerAxisDimId = 1;  // x always is not transposed, check K axis
    size_t kAxisOfWeight = transposeWeight ? 2UL : 1UL;  // if weight is transposed, 2 is the k axis idx of the weight, otherwise is 1
    OP_CHECK_IF(CheckShapeSameLengthTensorList(context, {1, kAxisOfWeight}, innerAxisDimId, tenorXAndWeight, paramsInfo.numX) != GRAPH_SUCCESS,
              OP_LOGE(context->GetNodeName(), "k dim value of x and weight is not matched."),
              return GRAPH_FAILED);
    innerAxisDimId = !transposeWeight ? 2 : -1;  // If w is not transposed, check N(2) asix; otherwise, check k axis, which can be skiped
    OP_CHECK_IF(CheckInnerAxisOfTensorList(context, GMM_INDEX_IN_WEIGHT, innerAxisDimId, paramsInfo.numWeight) != GRAPH_SUCCESS,
              OP_LOGE(context->GetNodeName(), "inner axis size of weight is larger than %ld!", GMM_MAX_INNER_AXIS),
              return GRAPH_FAILED);
    OP_CHECK_IF(CheckWeightShapeInnerAxisEven(context, paramsInfo.numWeight, 2) != GRAPH_SUCCESS,
              OP_LOGE(context->GetNodeName(), "weight's N axis size should be even when it is int4 dtype."),
              return GRAPH_FAILED);
    // check groupList
    OP_CHECK_IF(CheckGroupListCommonTensor(context, true, context->GetDynamicInputShape(GMM_INDEX_IN_WEIGHT, 0)->GetDim(0)) != GRAPH_SUCCESS,
              OP_LOGE(context->GetNodeName(), "Invalid groupList."),
              return GRAPH_FAILED);
    return GRAPH_SUCCESS;
}

static ge::graphStatus SplitMSingleXSeparatedWeightSingleY(gert::InferShapeContext* context, bool transposeWeight,
                                                           const GMMParamsInfo& paramsInfo) {
    std::vector<std::string> tenorWeightAndX{"weight", "x", "true"};
    // check dimension
    OP_CHECK_IF(CheckDimNum(context, paramsInfo.numX, GMM_MIN_FM_DIM, "x") != GRAPH_SUCCESS,
              OP_LOGE(context->GetNodeName(), "Dim num or format of tensor in tensor list x is invalid."),
              return GRAPH_FAILED);
    OP_CHECK_IF(CheckDimNum(context, paramsInfo.numWeight, GMM_SEPARATED_WEIGHT_DIM, "weight") != GRAPH_SUCCESS,
              OP_LOGE(context->GetNodeName(), "Dim num or format of tensor in tensor list weight is invalid."),
              return GRAPH_FAILED);
    // check shape, x(m,k), weight(k,n), y(m,n)
    int64_t innerAxisDimId = 1;  // x always is not transposed, check K axis
    size_t kAxisOfWeight = transposeWeight ? 1UL : 0UL;
    OP_CHECK_IF(CheckShapeDiffLengthTensorList(context, {kAxisOfWeight, 1}, innerAxisDimId, tenorWeightAndX, paramsInfo.numWeight) != GRAPH_SUCCESS,
              OP_LOGE(context->GetNodeName(), "k dim value of x and weight is not matched."),
              return GRAPH_FAILED);
    innerAxisDimId = !transposeWeight ? 1 : -1;  // if w is not transposed, check N asix; otherwise, check k axis, which can be skiped
    OP_CHECK_IF(CheckInnerAxisOfTensorList(context, GMM_INDEX_IN_WEIGHT, innerAxisDimId, 1) != GRAPH_SUCCESS,
              OP_LOGE(context->GetNodeName(), "inner axis size of weight is larger than %ld!", GMM_MAX_INNER_AXIS),
              return GRAPH_FAILED);
    OP_CHECK_IF(CheckWeightShapeInnerAxisEven(context, paramsInfo.numWeight, 1) != GRAPH_SUCCESS,
              OP_LOGE(context->GetNodeName(), "weight's N axis size should be even when it is int4 dtype."),
              return GRAPH_FAILED);
    // check groupList
    OP_CHECK_IF(CheckGroupListCommonTensor(context, true, paramsInfo.numWeight) != GRAPH_SUCCESS,
              OP_LOGE(context->GetNodeName(), "Invalid groupList."),
              return GRAPH_FAILED);
    return GRAPH_SUCCESS;
}

static ge::graphStatus SplitMSeparatedXSeparatedWeightSingleY(gert::InferShapeContext* context,
                                                              bool transposeWeight, const GMMParamsInfo& paramsInfo) {
    const size_t& xSize = paramsInfo.numX;
    const size_t& weightSize = paramsInfo.numWeight;
    std::vector<std::string> tenorWeightAndX{"weight", "x", "true"};
    OP_CHECK_IF(xSize != weightSize,
              OP_LOGE(context->GetNodeName(),
                        "When x and weight are separated, size of x %lu should equal to size of weight %lu.",
                        xSize, weightSize),
              return GRAPH_FAILED);
    // check dimension
    OP_CHECK_IF(CheckDimNum(context, xSize, GMM_MIN_FM_DIM, "x") != GRAPH_SUCCESS,
              OP_LOGE(context->GetNodeName(), "Dim num or format of tensor in tensor list x is invalid."),
              return GRAPH_FAILED);
    OP_CHECK_IF(CheckDimNum(context, weightSize, GMM_SEPARATED_WEIGHT_DIM, "weight") != GRAPH_SUCCESS,
              OP_LOGE(context->GetNodeName(), "Dim num or format of tensor in tensor list weight is invalid."),
              return GRAPH_FAILED);
    // check shape, x(m,k), weight(k,n), y(m,n)
    int64_t innerAxisDimId = 1;  // originalShape's inner axis of weight
    size_t kAxisOfWeight = transposeWeight ? 1UL : 0UL;
    OP_CHECK_IF(CheckShapeSameLengthTensorList(context, {kAxisOfWeight, 1}, innerAxisDimId, tenorWeightAndX, weightSize) != GRAPH_SUCCESS,
              OP_LOGE(context->GetNodeName(), "k dim value of x and weight is not matched."),
              return GRAPH_FAILED);
    innerAxisDimId = !transposeWeight ? 1 : -1;  // if w is not transposed, N asix has been checked, need to check x's inner axis(K, when x is always not transposed)
    OP_CHECK_IF(CheckInnerAxisOfTensorList(context, GMM_INDEX_IN_X, innerAxisDimId, 1) != GRAPH_SUCCESS,
              OP_LOGE(context->GetNodeName(), "inner axis size of x is larger than %ld!", GMM_MAX_INNER_AXIS),
              return GRAPH_FAILED);
    OP_CHECK_IF(CheckWeightShapeInnerAxisEven(context, weightSize, 1) != GRAPH_SUCCESS,
              OP_LOGE(context->GetNodeName(), "weight's N axis size should be even when it is int4 dtype."),
              return GRAPH_FAILED);
    // check groupList
    OP_CHECK_IF(CheckGroupListCommonTensor(context, false, xSize) != GRAPH_SUCCESS,
              OP_LOGE(context->GetNodeName(), "Invalid groupList."),
              return GRAPH_FAILED);
    return GRAPH_SUCCESS;
}

static ge::graphStatus CheckCaseSplitM(gert::InferShapeContext* context, bool transposeWeight,
                                       const GMMParamsInfo& paramsInfo) {
    const size_t& xSize = paramsInfo.numX;
    const size_t& weightSize = paramsInfo.numWeight;
    const size_t& ySize = paramsInfo.numY;
    if ((xSize == 1UL) && (weightSize == 1UL) && (ySize == 1UL)) {
        OP_CHECK_IF(SplitMSingleXSingleWeightSingleY(context, transposeWeight, paramsInfo) != GRAPH_SUCCESS,
                  OP_LOGE(context->GetNodeName(), "Split m, single x, single weight, single y case failed."),
                  return GRAPH_FAILED);
        return GRAPH_SUCCESS;
    }
    if ((xSize == 1UL) && (weightSize > 1UL) && (ySize == 1UL)) {
        OP_CHECK_IF(weightSize != paramsInfo.groupNum, OP_LOGE(context->GetNodeName(),
                  "weight Size [%zu] does not equal with groupNum %zu", weightSize, paramsInfo.groupNum),
                  return GRAPH_FAILED);
        OP_CHECK_IF(SplitMSingleXSeparatedWeightSingleY(context, transposeWeight, paramsInfo) != GRAPH_SUCCESS,
                  OP_LOGE(context->GetNodeName(), "Split m, single x, separated weight, single y case failed."),
                  return GRAPH_FAILED);
        return GRAPH_SUCCESS;
    }
    if ((xSize == 1UL) && (weightSize > 1UL) && (ySize > 1UL)) {
        const gert::Tensor* groupListTensor = context->GetOptionalInputTensor(GMM_INDEX_IN_GROUP_LIST);
        OP_CHECK_IF(groupListTensor == nullptr || groupListTensor->GetData<int64_t>() == nullptr,
                  OP_LOGE(context->GetNodeName(), "Failed to obtain necessary data from groupListTensor. "
                            "When grouplist is an invalid tensor, split m, single x, separated weight, separated y cases do not support."),
                  return GRAPH_FAILED);
        return GRAPH_SUCCESS;  // skip the check
    }
    if ((xSize > 1UL) && (weightSize > 1UL) && (ySize == 1UL)) {
        OP_CHECK_IF(weightSize != paramsInfo.groupNum, OP_LOGE(context->GetNodeName(),
                  "weight Size [%zu] does not equal with groupNum %zu", weightSize, paramsInfo.groupNum),
                  return GRAPH_FAILED);
        OP_CHECK_IF(SplitMSeparatedXSeparatedWeightSingleY(context, transposeWeight, paramsInfo) != GRAPH_SUCCESS,
                  OP_LOGE(context->GetNodeName(), "Split m, separated x, separated weight, single y case failed."),
                  return GRAPH_FAILED);
        return GRAPH_SUCCESS;
    }
    OP_LOGE(context->GetNodeName(), "When groupType is 0, current case with x %zu, weight %zu, y %zu is not supported.",
              xSize, weightSize, ySize);
    return GRAPH_FAILED;
}

static ge::graphStatus CheckCaseSplitK(gert::InferShapeContext* context, bool transposeX, bool transposeWeight,
                                       const GMMParamsInfo& paramsInfo) {
    std::vector<std::string> tenorXAndWeight{"x", "weight", "true"};
    const size_t& xSize = paramsInfo.numX;
    const size_t& weightSize = paramsInfo.numWeight;
    const size_t& ySize = paramsInfo.numY;
    if (xSize == 1UL) {
        if (paramsInfo.platform == PlatformID::ASCEND950) {
            return GRAPH_SUCCESS;
        }
        OP_CHECK_IF(!transposeX,
              OP_LOGE(context->GetNodeName(),
                        "When groupType is 2 and x is not separated, tensor in x should be transposed."),
              return GRAPH_FAILED);
        // check dimension
        OP_CHECK_IF(CheckDimNum(context, xSize, GMM_MIN_FM_DIM, "x") != GRAPH_SUCCESS,
                  OP_LOGE(context->GetNodeName(), "Dim num or format of tensor in tensor list x is invalid."),
                  return GRAPH_FAILED);
        OP_CHECK_IF(CheckDimNum(context, weightSize, GMM_SPLIT_K_SINGLE_WEIGHT_DIM, "weight") != GRAPH_SUCCESS,
                  OP_LOGE(context->GetNodeName(), "Dim num or format of tensor in tensor list weight is invalid."),
                  return GRAPH_FAILED);
        // check shape, x(m,k), weight(k,n), y(b,m,n)
        int64_t innerAxisDimId = 1;  // x always is transposed, and the inner axis is always the last axis, M axis.
        size_t kAxisOfWeight = transposeWeight ? 1UL : 0UL;
        if((weightSize == 1UL) && (ySize == 1UL)) {
            OP_CHECK_IF(CheckShapeSameLengthTensorList(context, {0, kAxisOfWeight}, innerAxisDimId, tenorXAndWeight, xSize) != GRAPH_SUCCESS,
                    OP_LOGE(context->GetNodeName(), "k dim value of x and weight is not matched."),
                    return GRAPH_FAILED);
            innerAxisDimId = 1;  // w always is not transposed, and the inner axis is always the last axis, N axis.
            // check groupList
            OP_CHECK_IF(CheckGroupListCommonTensor(context, true, 1) != GRAPH_SUCCESS,
                    OP_LOGE(context->GetNodeName(), "Invalid groupList."),
                    return GRAPH_FAILED);
        }
        OP_CHECK_IF(CheckInnerAxisOfTensorList(context, GMM_INDEX_IN_WEIGHT, innerAxisDimId, weightSize) != GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "inner axis size of weight is larger than %ld!", GMM_MAX_INNER_AXIS),
                return GRAPH_FAILED);
        return GRAPH_SUCCESS;
    }
    OP_LOGE(context->GetNodeName(),
              "When groupType is 2, only support case with unseparated x, weight and y, "
              "but now x size is %lu, weight size is %lu, y size is %lu.", xSize, weightSize, ySize);
    return GRAPH_FAILED;
}

static ge::graphStatus CheckParamDifferentGroupType(gert::InferShapeContext* context, const GMMAttrs& gmmAttrs,
                                                    const GMMParamsInfo& paramsInfo) {
    OP_CHECK_IF(paramsInfo.platform == PlatformID::UNKNOWN, OP_LOGW(context->GetNodeName(), "Cannot get platform info!"), return GRAPH_SUCCESS);
    const int64_t& groupType = gmmAttrs.groupType;
    const bool& transposeX = gmmAttrs.transposeX;
    const bool& transposeWeight = gmmAttrs.transposeWeight;
    OP_CHECK_IF(transposeX && transposeWeight, OP_LOGE(context->GetNodeName(),
              "x and weight can not be transposed at the same time."), return GRAPH_FAILED);
    auto groupTensorOptionalShape = context->GetOptionalInputShape(GMM_INDEX_IN_GROUP_LIST);
    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t* groupListTypePtr = attrs->GetAttrPointer<int64_t>(GMM_INDEX_ATTR_GROUP_LIST_TYPE);
    OP_CHECK_NULL_WITH_CONTEXT(context, groupListTypePtr);
    size_t validGroupTensorDimNum = (*groupListTypePtr == 2L) ? 2UL: 1UL;  // 2: split M sparse, group list shape [e, 2]
    OP_CHECK_IF(groupTensorOptionalShape != nullptr && (groupTensorOptionalShape->GetDimNum() > validGroupTensorDimNum ||
              groupTensorOptionalShape->GetDim(0) < 1),
              OP_LOGE(context->GetNodeName(),
                        "When groupList is a tensor, its dim only supports 1 or 2(only when groupListType is 2) and "
                        "size of elements should be larger than 0, but now are %zu and %ld, respectively.",
                        groupTensorOptionalShape->GetDimNum(), groupTensorOptionalShape->GetDim(0)),
              return GRAPH_FAILED);
    OP_CHECK_IF(paramsInfo.platform == PlatformID::ASCEND310P && !(groupType == GMM_SPLIT_M && paramsInfo.numX == 1 &&
              paramsInfo.numWeight == 1 && paramsInfo.numY == 1),
              OP_LOGE(context->GetNodeName(),
                        "When on ASCEND310P, it only supports split m, single x, single weight, single y."),
              return GRAPH_FAILED);

    if (groupType == GMM_NO_SPLIT) {
        OP_CHECK_IF(transposeX, OP_LOGE(context->GetNodeName(),
                  "When x, weight and y are all separated, x can not be transposed."), return GRAPH_FAILED);
        OP_CHECK_IF(CheckCaseNoSplit(context, transposeWeight, paramsInfo) != GRAPH_SUCCESS,
                  OP_LOGE(context->GetNodeName(), "Invalid inputs!"), return GRAPH_FAILED);
    } else if (groupType == GMM_SPLIT_M) {
        OP_CHECK_IF(transposeX,
                  OP_LOGE(context->GetNodeName(), "When groupType is 0, x can not be transposed."),
                  return GRAPH_FAILED);
        OP_CHECK_IF(CheckCaseSplitM(context, transposeWeight, paramsInfo) != GRAPH_SUCCESS,
                  OP_LOGE(context->GetNodeName(), "Invalid inputs!"), return GRAPH_FAILED);
    } else if (groupType == GMM_SPLIT_K) {
        OP_CHECK_IF(!IsTensorListNullOrEmpty(context, GMM_INDEX_IN_BIAS),
                  OP_LOGE(context->GetNodeName(), "When groupType is 2, bias must be empty."), return GRAPH_FAILED);
        OP_CHECK_IF(CheckCaseSplitK(context, transposeX, transposeWeight, paramsInfo) != GRAPH_SUCCESS,
                  OP_LOGE(context->GetNodeName(), "Invalid inputs!"), return GRAPH_FAILED);
    }
    if (!IsTensorListNullOrEmpty(context, GMM_INDEX_IN_BIAS)) {
        OP_CHECK_IF(CheckOptionalTensorList(context, "bias", paramsInfo, gmmAttrs, GMM_INDEX_IN_BIAS) != GRAPH_SUCCESS,
                  OP_LOGE(context->GetNodeName(), "Invalid bias!"), return GRAPH_FAILED);
    }
    return GRAPH_SUCCESS;
}

static ge::graphStatus XNotSingleYSeparated(gert::InferShapeContext* context,
                                            size_t weightDimN, bool isXTransposed, size_t xDimM) {
    const gert::Tensor* groupListTensor = context->GetOptionalInputTensor(GMM_INDEX_IN_GROUP_LIST);
    if (groupListTensor != nullptr) {
        OP_CHECK_IF(UpdateMultipleShapeY(context, groupListTensor, weightDimN, isXTransposed, xDimM) != GRAPH_SUCCESS,
                  OP_LOGE(context->GetNodeName(), "Failed to update shape of y."), return GRAPH_FAILED);
    } else {
        OP_CHECK_IF(MultiInMultiOutWithoutGroupList(context)!= GRAPH_SUCCESS, OP_LOGE(context->GetNodeName(),
                  "Failed to process multi-in-multi-out case without GroupList."), return GRAPH_FAILED);
    }
    return GRAPH_SUCCESS;
}

static ge::graphStatus XSingleYSeparated(gert::InferShapeContext* context,
                                         size_t weightDimN, bool isXTransposed, size_t xDimM) {
    const gert::Tensor* groupListTensor = context->GetOptionalInputTensor(GMM_INDEX_IN_GROUP_LIST);
    OP_CHECK_IF(groupListTensor == nullptr,
              OP_LOGE(context->GetNodeName(), "GroupList is required when x is single tensor while y is not."),
              return GRAPH_FAILED);
    OP_CHECK_IF(UpdateMultipleShapeY(context, groupListTensor, weightDimN, isXTransposed, xDimM) != GRAPH_SUCCESS,
              OP_LOGE(context->GetNodeName(), "Failed to update shape of y."),
              return GRAPH_FAILED);
    return GRAPH_SUCCESS;
}

static ge::graphStatus GMMSetOutputShape(gert::InferShapeContext* context, GMMAttrs& gmmAttrs,
                                        const GMMSetOutputParams& outputParams, const gert::Shape* x0Shape,
                                         const gert::Shape* w0Shape) {
    bool isSingleX = outputParams.isSingleX;
    bool isSingleY = outputParams.isSingleY;
    size_t xDimM = outputParams.xDimM;
    size_t weightDimN = outputParams.weightDimN;
    size_t numX = outputParams.numX;
    size_t numWeight = outputParams.numWeight;
    int64_t lenGroupList = outputParams.lenGroupList;
    // X单 Y多
    if (isSingleX && !isSingleY) {
        if(gmmAttrs.groupType != GMM_SPLIT_K) {
            OP_CHECK_IF(XSingleYSeparated(context, weightDimN, gmmAttrs.transposeX, xDimM) != GRAPH_SUCCESS,
            OP_LOGE(context->GetNodeName(), "Failed to update shape of y."), return GRAPH_FAILED);
        } else {
            OP_CHECK_IF(MultiWeightMultiOutWithoutGroupList(context)!= GRAPH_SUCCESS, OP_LOGE(context->GetNodeName(),
            "Failed to process multi-in-multi-out case without GroupList."), return GRAPH_FAILED);
        }
    // X单 Y单
    } else if (isSingleX && isSingleY) {
        OP_CHECK_IF(gmmAttrs.groupType != GMM_SPLIT_M && gmmAttrs.groupType != GMM_SPLIT_K,
                  OP_LOGE(context->GetNodeName(),
                  "When x is single tensor, input tensors can only be split along M or K axis."), return GRAPH_FAILED);
        std::vector<int64_t> yDims = {x0Shape->GetDim(xDimM), w0Shape->GetDim(weightDimN)};
        if (gmmAttrs.groupType == GMM_SPLIT_K) {
            yDims.insert(yDims.begin(), numWeight == 1 ? lenGroupList : numWeight);
        }
        OP_CHECK_IF(UpdateShapeY(context, GMM_INDEX_OUT_Y, yDims) != GRAPH_SUCCESS,
                  OP_LOGE(context->GetNodeName(), "Failed to update y shape."), return GRAPH_FAILED);
    }
    // X多 Y多
    else if (!isSingleX && !isSingleY) {
        OP_CHECK_IF(XNotSingleYSeparated(context, weightDimN, gmmAttrs.transposeX, xDimM) != GRAPH_SUCCESS,
                  OP_LOGE(context->GetNodeName(), "Failed to update shape of y."), return GRAPH_FAILED);
    }
    // X多 Y单
    else if (!isSingleX && isSingleY) {
        std::vector<int64_t> yDims = {GetDim0(context, gmmAttrs.transposeX, numX, xDimM), w0Shape->GetDim(weightDimN)};
        OP_CHECK_IF(UpdateShapeY(context, GMM_INDEX_OUT_Y, yDims) != GRAPH_SUCCESS,
                  OP_LOGE(context->GetNodeName(), "Failed to update shape of y."), return GRAPH_FAILED);
    }

    return GRAPH_SUCCESS;
}

static graphStatus InferShape4DavidWeightQuantGMM(gert::InferShapeContext *context)
{
    GroupedMatmulWeightQuantChecker davidWeightQuantGMMChecker;
    GroupedMatmulCommonUtil utilForDavidWeightQuantGMM;
    OP_CHECK_IF(GetAttrsValue(context, utilForDavidWeightQuantGMM.attrsInfo) != GRAPH_SUCCESS,
              OP_LOGE(context->GetNodeName(), "GetAttrsValue failed"), return GRAPH_FAILED);
    OP_CHECK_IF(davidWeightQuantGMMChecker.GetXAndWeightDimValue(context, utilForDavidWeightQuantGMM.attrsInfo) !=
                  GRAPH_SUCCESS,
              OP_LOGE(context->GetNodeName(), "GetXAndWeightDimValue failed"), return GRAPH_FAILED);
    OP_CHECK_IF(davidWeightQuantGMMChecker.CheckShape(context, utilForDavidWeightQuantGMM) != GRAPH_SUCCESS,
              OP_LOGE(context->GetNodeName(), "CheckShape failed"), return GRAPH_FAILED);
    OP_CHECK_IF(davidWeightQuantGMMChecker.InferOutShape(context, utilForDavidWeightQuantGMM.attrsInfo) != GRAPH_SUCCESS,
              OP_LOGE(context->GetNodeName(), "InferOutShape failed"), return GRAPH_FAILED);
    return GRAPH_SUCCESS;
}

static graphStatus InferShape4DavidQuantGMM(gert::InferShapeContext* context) {
    GroupedMatmulQuantChecker davidQuantGMMChecker;
    GroupedMatmulCommonUtil utilForDavidQuantGMM;
    OP_CHECK_IF(GetAttrsValue(context, utilForDavidQuantGMM.attrsInfo) != GRAPH_SUCCESS,
              OP_LOGE(context->GetNodeName(), "GetAttrsValue failed"), return GRAPH_FAILED);
    OP_CHECK_IF(davidQuantGMMChecker.GetXAndWeightDimValue(context, utilForDavidQuantGMM.attrsInfo) != GRAPH_SUCCESS,
              OP_LOGE(context->GetNodeName(), "GetXAndWeightDimValue failed"), return GRAPH_FAILED);
    OP_CHECK_IF(davidQuantGMMChecker.GetGroupNumValue(context) != GRAPH_SUCCESS,
              OP_LOGE(context->GetNodeName(), "GetGroupNumValue failed"), return GRAPH_FAILED);
    OP_CHECK_IF(davidQuantGMMChecker.CheckShape(context, utilForDavidQuantGMM) != GRAPH_SUCCESS,
              OP_LOGE(context->GetNodeName(), "CheckShape failed"), return GRAPH_FAILED);
    OP_CHECK_IF(davidQuantGMMChecker.InferOutShape(context, utilForDavidQuantGMM.attrsInfo) != GRAPH_SUCCESS,
              OP_LOGE(context->GetNodeName(), "InferOutShape failed"), return GRAPH_FAILED);
    return GRAPH_SUCCESS;
}

template <typename T>
static graphStatus IsDavidWeightQuantGMMByShape(T context)
{
    auto xDesc = context->GetDynamicInputDesc(GMM_INDEX_IN_X, 0);
    auto weightDesc = context->GetDynamicInputDesc(GMM_INDEX_IN_WEIGHT, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightDesc);
    DataType xDtype = xDesc->GetDataType();
    DataType weightDtype = weightDesc->GetDataType();
    return GetSizeByDataType(xDtype) != GetSizeByDataType(weightDtype) ? GRAPH_SUCCESS : GRAPH_FAILED;
}

template<typename T>
static graphStatus IsDavidQuantGMMByShape(T context) {
    auto xDesc = context->GetDynamicInputDesc(GMM_INDEX_IN_X, 0);
    auto weightDesc = context->GetDynamicInputDesc(GMM_INDEX_IN_WEIGHT, 0);
    auto scaleDesc = context->GetDynamicInputDesc(GMM_INDEX_IN_SCALE, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, scaleDesc);
    DataType xDtype = xDesc->GetDataType();
    DataType weightDtype = weightDesc->GetDataType();
    if (xDtype == ge::DT_FLOAT4_E2M1 || xDtype == ge::DT_INT4) {
        return GRAPH_SUCCESS;
    }
    return (GetSizeByDataType(xDtype) == 1 && GetSizeByDataType(weightDtype) == 1) ? GRAPH_SUCCESS : GRAPH_FAILED;
}

static ge::graphStatus InferShape4GroupedMatmul(gert::InferShapeContext* context) {
    OP_CHECK_NULL_WITH_CONTEXT(context, context);
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optionalInfo;
    auto ret = fe::PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platformInfo, optionalInfo);
    if (ret == GRAPH_SUCCESS && GmmDavidSupportSoc.count(platformInfo.str_info.short_soc_version) > 0) {
        if (IsDavidQuantGMMByShape(context) == GRAPH_SUCCESS) {
            OP_CHECK_IF(InferShape4DavidQuantGMM(context) != GRAPH_SUCCESS,
                      OP_LOGE(context->GetNodeName(), "Check params failed"), return GRAPH_FAILED);
            return GRAPH_SUCCESS;
        } else if (IsDavidWeightQuantGMMByShape(context) == GRAPH_SUCCESS) {
            OP_CHECK_IF(InferShape4DavidWeightQuantGMM(context) != GRAPH_SUCCESS,
                      OP_LOGE(context->GetNodeName(), "Check params failed"), return GRAPH_FAILED);
            return GRAPH_SUCCESS;
        }
    }
    GMMAttrs gmmAttrs{GMM_X_Y_SEPARATED, 0, GMM_NO_SPLIT, false, false, 0, 0};
    OP_CHECK_IF(GetAttrsValue(context, gmmAttrs) != GRAPH_SUCCESS || CheckAttrs(context, gmmAttrs) != GRAPH_SUCCESS,
              OP_LOGE(context->GetNodeName(), "Failed to get attrs."), return GRAPH_FAILED);

    size_t numX = 0;  // init numX
    size_t numWeight = 0;  // init numWeight
    int64_t lenGroupList = 0;  // init lenGroupList
    size_t numY = context->GetComputeNodeOutputNum();
    if (GetNumOfInputs(context, numX, numWeight, lenGroupList) == GRAPH_SUCCESS) {  // check input shape value inside
        GMMParamsInfo paramsInfo{numX, numWeight, numY, lenGroupList, 0, 0, 0, 0, 0, PlatformID::UNKNOWN};
        OP_CHECK_IF(GetGroupSize(context, paramsInfo) != GRAPH_SUCCESS,
                  OP_LOGE(context->GetNodeName(), "check groupNum failed"), return GRAPH_FAILED);
        OP_CHECK_IF(CheckFunctionParamsForShape(context, gmmAttrs, paramsInfo) != GRAPH_SUCCESS,
                  OP_LOGE(context->GetNodeName(), "CheckFunctionParamsForShape failed."), return GRAPH_FAILED);
        OP_CHECK_IF(CheckParamDifferentGroupType(context, gmmAttrs, paramsInfo) != GRAPH_SUCCESS,
                  OP_LOGE(context->GetNodeName(), "CheckParamDifferentGroupType failed."), return GRAPH_FAILED);
    } else {
        OP_CHECK_IF(CheckDimNum(context, numX, GMM_MIN_FM_DIM, "x") != GRAPH_SUCCESS,  // check dim number of tensors
                  OP_LOGE(context->GetNodeName(), "Dim num of tensor in tensorList x is invalid."),
                  return GRAPH_FAILED);
    }

    const gert::Shape* x0Shape = context->GetDynamicInputShape(GMM_INDEX_IN_X, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context, x0Shape);
    size_t xDimNum = x0Shape->GetDimNum();
    const gert::Shape* w0Shape = context->GetDynamicInputShape(GMM_INDEX_IN_WEIGHT, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context, w0Shape);
    size_t weightDimNum = w0Shape->GetDimNum();
    bool isSingleX = (numX == 1UL) && (gmmAttrs.groupType != GMM_NO_SPLIT);
    bool isSingleY = (numY == 1UL) && (gmmAttrs.groupType != GMM_NO_SPLIT);
    size_t xDimM = gmmAttrs.transposeX ? xDimNum - 1UL : xDimNum - 2UL;
    size_t weightDimN = gmmAttrs.transposeWeight ? weightDimNum - 2UL : weightDimNum - 1UL;

    GMMSetOutputParams outputParams;
    outputParams.isSingleX = isSingleX;
    outputParams.isSingleY = isSingleY;
    outputParams.xDimM = xDimM;
    outputParams.numX = numX;
    outputParams.weightDimN = weightDimN;
    outputParams.lenGroupList = lenGroupList;
    outputParams.numWeight = numWeight;
    OP_CHECK_IF(GMMSetOutputShape(context, gmmAttrs, outputParams, x0Shape, w0Shape) != GRAPH_SUCCESS,
              OP_LOGE(context->GetNodeName(), "GMMSetOutputShape failed"), return GRAPH_FAILED);

    return GRAPH_SUCCESS;
}

// =========================================================================================
// =========================================================================================
static graphStatus CheckTensorListDataType(const gert::InferDataTypeContext* context, uint32_t index,
                                           const DataType dtype) {
    size_t inIdx = 0;
    while (true) {
        auto iDtype = context->GetDynamicInputDataType(index, inIdx);
        if (iDtype == DT_UNDEFINED) {
            break;
        }
        OP_CHECK_IF(iDtype != dtype,
                  OP_LOGE(context->GetNodeName(), "data type of tensors in a tensorList should all be the same!"),
                  return GRAPH_FAILED);
        ++inIdx;
    }
    return GRAPH_SUCCESS;
}

static graphStatus CheckMatmulDataType(gert::InferDataTypeContext* context, const DataType xDtype,
                                       const DataType weightDtype, const DataType biasDtype) {
    OP_CHECK_IF(CheckTensorListDataType(context, GMM_INDEX_IN_X, xDtype) != GRAPH_SUCCESS,
              OP_LOGE(context->GetNodeName(), "x dtype does not match with required dtype[%s].",
                        TypeUtils::DataTypeToAscendString(xDtype).GetString()),
              return GRAPH_FAILED);
    OP_CHECK_IF(CheckTensorListDataType(context, GMM_INDEX_IN_WEIGHT, weightDtype) != GRAPH_SUCCESS,
              OP_LOGE(context->GetNodeName(), "weight dtype does not match with required dtype[%s].",
                        TypeUtils::DataTypeToAscendString(weightDtype).GetString()),
              return GRAPH_FAILED);
    OP_CHECK_IF(CheckTensorListDataType(context, GMM_INDEX_IN_BIAS, biasDtype) != GRAPH_SUCCESS,
              OP_LOGE(context->GetNodeName(), "bias dtype does not match with required dtype[%s].",
                        TypeUtils::DataTypeToAscendString(biasDtype).GetString()),
              return GRAPH_FAILED);
    return GRAPH_SUCCESS;
}

static graphStatus CheckNonQuantMatmulParams(fe::PlatformInfo& platformInfo, gert::InferDataTypeContext* context,
                                             const DataType xDtype, const DataType weightDtype)
{
    DataType biasDtype = xDtype == DataType::DT_BF16 ? DataType::DT_FLOAT: xDtype;
    if (GmmDavidSupportSoc.count(platformInfo.str_info.short_soc_version) > 0) {
        biasDtype = context->GetDynamicInputDataType(GMM_INDEX_IN_BIAS, 0);
        if (biasDtype != DT_UNDEFINED) {
            OP_CHECK_IF(std::find(BIAS_DTYPE_SUPPORT_LIST.begin(), BIAS_DTYPE_SUPPORT_LIST.end(), biasDtype) == BIAS_DTYPE_SUPPORT_LIST.end(),
                      OP_LOGE(context->GetNodeName(),"non quant case bias only support dtype float16, bfloat16 and float32"),
                      return GRAPH_FAILED);
        }
    }
    OP_CHECK_IF(CheckMatmulDataType(context, xDtype, weightDtype, biasDtype) != GRAPH_SUCCESS,
              OP_LOGE(context->GetNodeName(), "case with x dtype %s and weight dtype %s is not supported!",
                        TypeUtils::DataTypeToAscendString(xDtype).GetString(), TypeUtils::DataTypeToAscendString(weightDtype).GetString()),
              return GRAPH_FAILED);
    return GRAPH_SUCCESS;
}

static graphStatus CheckFunctionQuantParams(gert::InferDataTypeContext* context) {
    OP_CHECK_IF(CheckTensorListDataType(context, GMM_INDEX_IN_X, DataType::DT_INT8) != GRAPH_SUCCESS,
              OP_LOGE(context->GetNodeName(), "x dtype does not match with required dtype[INT8]."),
              return GRAPH_FAILED);
    OP_CHECK_IF(CheckTensorListDataType(context, GMM_INDEX_IN_WEIGHT, DataType::DT_INT8) != GRAPH_SUCCESS,
              OP_LOGE(context->GetNodeName(), "weight dtype does not match with required dtype[INT8]."),
              return GRAPH_FAILED);
    OP_CHECK_IF((CheckTensorListDataType(context, GMM_INDEX_IN_BIAS, DataType::DT_INT32) != GRAPH_SUCCESS) &&
                  (CheckTensorListDataType(context, GMM_INDEX_IN_BIAS, DataType::DT_BF16) != GRAPH_SUCCESS),
              OP_LOGE(context->GetNodeName(), "bias dtype does not match with required dtype int32 or bfloat16."),
              return GRAPH_FAILED);
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t* outputDtype = attrs->GetInt(GMM_INDEX_ATTR_OUTPUT_DTYPE);
    if (*outputDtype == GMM_OUT_DTYPE_INT32) {  // output dtype is int32, this scene does not need scale
        return GRAPH_SUCCESS;
    }
    auto scale0Dtype = context->GetDynamicInputDataType(GMM_INDEX_IN_SCALE, 0);
    // Now we cannot make sure if is pertoken quant case, so scale/offset dtype check is remained to the InferShape stage.
    OP_CHECK_IF(CheckTensorListDataType(context, GMM_INDEX_IN_SCALE, scale0Dtype) != GRAPH_SUCCESS,
              OP_LOGE(context->GetNodeName(), "dtypes of scales in the tensorList should all be the same."),
              return GRAPH_FAILED);
    auto offset0Dtype = context->GetDynamicInputDataType(GMM_INDEX_IN_OFFSET, 0);
    OP_CHECK_IF(CheckTensorListDataType(context, GMM_INDEX_IN_OFFSET, offset0Dtype) != GRAPH_SUCCESS,
              OP_LOGE(context->GetNodeName(), "dtypes of offsets in the tensorList should all be the same."),
              return GRAPH_FAILED);
    return GRAPH_SUCCESS;
}

static graphStatus CheckGroupedMatmulAntiQuantForDtype(gert::InferDataTypeContext* context) {
    auto xDtype = context->GetDynamicInputDataType(GMM_INDEX_IN_X, 0);
    OP_CHECK_IF(CheckTensorListDataType(context, GMM_INDEX_IN_ANTIQUANT_SCALE, xDtype) != GRAPH_SUCCESS,
              OP_LOGE(context->GetNodeName(), "antiquantScale dtype does not match with x dtype[%s].", TypeUtils::DataTypeToAscendString(xDtype).GetString()),
              return GRAPH_FAILED);
    OP_CHECK_IF(CheckTensorListDataType(context, GMM_INDEX_IN_ANTIQUANT_OFFSET, xDtype) != GRAPH_SUCCESS,
              OP_LOGE(context->GetNodeName(), "antiquantOffset dtype does not match with x dtype[%s].", TypeUtils::DataTypeToAscendString(xDtype).GetString()),
              return GRAPH_FAILED);
    return GRAPH_SUCCESS;
}

static graphStatus CheckFunctionParamsForDtype(gert::InferDataTypeContext* context) {
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optionalInfo;
    graphStatus ret = fe::PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platformInfo, optionalInfo);
    PlatformID platform = PlatformID::UNKNOWN;
    if (ret != ge::GRAPH_SUCCESS) {
        OP_LOGW(context->GetNodeName(), "Cannot get platform info.");
        return GRAPH_SUCCESS;
    } else {
        platform = (optionalInfo.soc_version.find("310P") != std::string::npos) ?
                    PlatformID::ASCEND310P : (optionalInfo.soc_version.find("950") != std::string::npos) ?
                    PlatformID::ASCEND950 : PlatformID::ASCEND910B;
    }
    DataType xDtype = context->GetDynamicInputDataType(GMM_INDEX_IN_X, 0);
    DataType weightDtype = context->GetDynamicInputDataType(GMM_INDEX_IN_WEIGHT, 0);
    if (platform == PlatformID::ASCEND310P) {
        bool isAllInputFP16 = xDtype == DataType::DT_FLOAT16 && weightDtype == DataType::DT_FLOAT16;
        OP_CHECK_IF(!isAllInputFP16, OP_LOGE(context->GetNodeName(),
                  "Only float16 is supported on Ascend310P platforms."), return GRAPH_FAILED);
        auto biasDtype = context->GetOptionalInputDataType(GMM_INDEX_IN_BIAS);
        OP_CHECK_IF(biasDtype != ge::DT_UNDEFINED && biasDtype != DataType::DT_FLOAT16, OP_LOGE(context->GetNodeName(),
                  "only bias float16 is supported on Ascend310P platforms."), return GRAPH_FAILED);
    }
    if ((xDtype == DataType::DT_INT8 || xDtype == DataType::DT_INT4) &&
        weightDtype == DataType::DT_INT4) { return GRAPH_SUCCESS; }
    if ((xDtype == DataType::DT_BF16 || xDtype == DataType::DT_FLOAT16 || xDtype == DataType::DT_FLOAT) &&
        xDtype == weightDtype) {  // nonquant
        return CheckNonQuantMatmulParams(platformInfo, context, xDtype, weightDtype);
    }
    if (xDtype == DataType::DT_INT8 && weightDtype == DataType::DT_INT8) {
        // quant
        OP_CHECK_IF(CheckFunctionQuantParams(context) != GRAPH_SUCCESS,
                  OP_LOGE(context->GetNodeName(), "CheckFunctionQuantParams failed."),
                  return GRAPH_FAILED);
        return GRAPH_SUCCESS;
    }
    if ((xDtype == DataType::DT_BF16 || xDtype == DataType::DT_FLOAT16) &&
        (weightDtype == DataType::DT_INT8 || weightDtype == DataType::DT_INT4)) {
        // antiquant
        DataType biasDtype = xDtype == DataType::DT_BF16 ? DataType::DT_FLOAT: DataType::DT_FLOAT16;
        OP_CHECK_IF(CheckMatmulDataType(context, xDtype, weightDtype, biasDtype) != GRAPH_SUCCESS,
                  OP_LOGE(context->GetNodeName(), "case with x dtype %s and weight dtype %s is not supported!",
                            TypeUtils::DataTypeToAscendString(xDtype).GetString(), TypeUtils::DataTypeToAscendString(weightDtype).GetString()),
                  return GRAPH_FAILED);
        return CheckGroupedMatmulAntiQuantForDtype(context);
    }
    OP_LOGE(context->GetNodeName(), "GMM: there is no matching xDtype and weightDtype pattern. "
              "case with x dtype %s and weight dtype %s is not supported.",
              TypeUtils::DataTypeToAscendString(xDtype).GetString(), TypeUtils::DataTypeToAscendString(weightDtype).GetString());
    return GRAPH_FAILED;
}

static graphStatus CheckQuantParamsDtype(const gert::InferDataTypeContext* context, const int64_t outputDtype,
                                         const DataType yDtype) {
    size_t i = 0;
    auto scale0Dtype = context->GetDynamicInputDataType(GMM_INDEX_IN_SCALE, 0);
    OP_CHECK_IF(scale0Dtype == ge::DT_UNDEFINED, OP_LOGE(context->GetNodeName(), "scale is undefined!"),
              return GRAPH_FAILED);
    auto perTokenScale0Dtype = context->GetDynamicInputDataType(GMM_INDEX_IN_PERTOKEN_SCALE, 0);
    bool isPerTokenQuant = perTokenScale0Dtype != ge::DT_UNDEFINED;
    if (isPerTokenQuant) {
        bool isOutputBF16 = scale0Dtype == DataType::DT_BF16 && outputDtype == 1;
        bool isOutputFloat16 = scale0Dtype == DataType::DT_FLOAT && outputDtype == 0;
        OP_CHECK_IF(!isOutputBF16 && !isOutputFloat16,
                  OP_LOGE(context->GetNodeName(), "per-token quant case only supports scale data type bfloat16 with "
                            "output data type bfloat16, or scale with data type float32 when output is float16, but "
                            "now scale[%zu] has data type %s and output has data type %s!",
                            i, TypeUtils::DataTypeToAscendString(scale0Dtype).GetString(), TypeUtils::DataTypeToAscendString(yDtype).GetString()),
                  return GRAPH_FAILED);
    } else {
        bool isOutputInt8 = scale0Dtype == DataType::DT_UINT64 && outputDtype == -1;
        bool isOutputBF16 = scale0Dtype == DataType::DT_BF16 && outputDtype == 1;
        bool isOutputFP16 = scale0Dtype == DataType::DT_FLOAT && outputDtype == 0;
        OP_CHECK_IF(!isOutputInt8 && !isOutputBF16 && !isOutputFP16,
                  OP_LOGE(context->GetNodeName(), "per-channel quant case only supports scale with data type uint64 "
                            "when output is int8, or data type bfloat16 when output is bfloat16, or data type float32 "
                            "when output is float16, but scale[%zu] has data type %s and output has data type %s!",
                            i, TypeUtils::DataTypeToAscendString(scale0Dtype).GetString(), TypeUtils::DataTypeToAscendString(yDtype).GetString()),
                  return GRAPH_FAILED);
    }
    if (isPerTokenQuant) {
        OP_CHECK_IF(perTokenScale0Dtype != DataType::DT_FLOAT,
                  OP_LOGE(context->GetNodeName(), "pertoken quant case only support perTokenScale with dtype float32,"
                            "but perTokenScale[%zu] has data type %s!", i, TypeUtils::DataTypeToAscendString(perTokenScale0Dtype).GetString()),
                  return GRAPH_FAILED);
    }
    return GRAPH_SUCCESS;
}

static graphStatus InferDtype4DavidWeightQuantGMM(gert::InferDataTypeContext *context)
{
    GroupedMatmulWeightQuantChecker davidWeightQuantGMMChecker;
    OP_CHECK_IF(davidWeightQuantGMMChecker.CheckDtype(context) != GRAPH_SUCCESS,
              OP_LOGE(context->GetNodeName(), "CheckDtype failed"), return GRAPH_FAILED);
    OP_CHECK_IF(davidWeightQuantGMMChecker.InferOutDtype(context) != GRAPH_SUCCESS,
              OP_LOGE(context->GetNodeName(), "SetYDtype failed"), return GRAPH_FAILED);
    return GRAPH_SUCCESS;
}

static graphStatus InferDtype4DavidQuantGMM(gert::InferDataTypeContext* context) {
    GroupedMatmulQuantChecker davidQuantGMMChecker;
    GroupedMatmulCommonUtil utilForDavidQuantGMM;
    OP_CHECK_IF(GetAttrsValue(context, utilForDavidQuantGMM.attrsInfo) != GRAPH_SUCCESS,
              OP_LOGE(context->GetNodeName(), "GetAttrsValue failed"), return GRAPH_FAILED);
    OP_CHECK_IF(davidQuantGMMChecker.CheckDtype(context, utilForDavidQuantGMM) != GRAPH_SUCCESS,
              OP_LOGE(context->GetNodeName(), "CheckDtype failed"), return GRAPH_FAILED);
    OP_CHECK_IF(davidQuantGMMChecker.InferOutDtype(context) != GRAPH_SUCCESS,
              OP_LOGE(context->GetNodeName(), "SetYDtype failed"), return GRAPH_FAILED);
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType4GroupedMatmul(gert::InferDataTypeContext *context){
    OP_CHECK_NULL_WITH_CONTEXT(context, context);
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optionalInfo;
    auto ret = fe::PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platformInfo, optionalInfo);
    if (ret == GRAPH_SUCCESS && GmmDavidSupportSoc.count(platformInfo.str_info.short_soc_version) > 0) {
        if (IsDavidQuantGMMByShape(context) == GRAPH_SUCCESS) {
            OP_CHECK_IF(InferDtype4DavidQuantGMM(context) != GRAPH_SUCCESS,
                      OP_LOGE(context->GetNodeName(), "InferDtype4DavidQuantGMM failed"), return GRAPH_FAILED);
            return GRAPH_SUCCESS;
        } else if (IsDavidWeightQuantGMMByShape(context) == GRAPH_SUCCESS) {
            OP_CHECK_IF(InferDtype4DavidWeightQuantGMM(context) != GRAPH_SUCCESS,
                      OP_LOGE(context->GetNodeName(), "InferDtype4DavidWeightQuantGMM failed"), return GRAPH_FAILED);
            return GRAPH_SUCCESS;
        }
    }
    OP_CHECK_IF(CheckFunctionParamsForDtype(context) != GRAPH_SUCCESS,
              OP_LOGE(context->GetNodeName(), "CheckFunctionParamsForDtype failed!"), return GRAPH_FAILED);

    auto x0Dtype = context->GetDynamicInputDataType(GMM_INDEX_IN_X, 0);
    auto weight0Dtype = context->GetDynamicInputDataType(GMM_INDEX_IN_WEIGHT, 0);
    size_t numY = context->GetComputeNodeOutputNum();
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    bool isQuantCase = x0Dtype == ge::DT_INT8 && weight0Dtype == ge::DT_INT8;
    bool isA8W4 = (x0Dtype == ge::DT_INT8 || x0Dtype == ge::DT_INT4) && weight0Dtype == ge::DT_INT4;
    const int64_t* outputDtype = attrs->GetInt(GMM_INDEX_ATTR_OUTPUT_DTYPE);
    DataType yDtype = x0Dtype;
    if (isQuantCase && outputDtype != nullptr) {
        auto it = GMM_OUTPUT_DTYPE_MAP.find(*outputDtype);
        OP_CHECK_IF(it == GMM_OUTPUT_DTYPE_MAP.end(),
                  OP_LOGE(context->GetNodeName(),
                            "value of attr dtype only supports -1/0/1/2, but now is %ld.", *outputDtype),
                  return GRAPH_FAILED);
        yDtype = it->second;
        if (*outputDtype != GMM_OUT_DTYPE_INT32) {  // output dtype is int32, this scene does not need scale
            OP_CHECK_IF(CheckQuantParamsDtype(context, *outputDtype, yDtype) != GRAPH_SUCCESS,
                      OP_LOGE(context->GetNodeName(), "Check quant params data type failed!"), return GRAPH_FAILED);
        }
    }
    if (isA8W4 && outputDtype != nullptr) {
        auto it = GMM_OUTPUT_DTYPE_MAP.find(*outputDtype);
        OP_CHECK_IF(it == GMM_OUTPUT_DTYPE_MAP.end(),
                  OP_LOGE(context->GetNodeName(),
                            "value of attr dtype only supports -1/0/1/2, but now is %ld.", *outputDtype),
                  return GRAPH_FAILED);
        yDtype = it->second;
    }
    for (size_t k = 0; k < numY; k++) {context->SetOutputDataType(GMM_INDEX_OUT_Y + k, yDtype);}
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(GroupedMatmul)
    .InferShape(InferShape4GroupedMatmul)
    .InferDataType(InferDataType4GroupedMatmul);
}  // namespace ops
