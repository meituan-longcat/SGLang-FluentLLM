/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include "fallback/fallback_comm.h"
#include "fallback/fallback.h"
#include "log/log.h"

#ifdef __cplusplus
extern "C" {
#endif

namespace fallback {
using namespace ge;
using namespace gert;

constexpr size_t INDEX_GMM_INPUT_X = 0;
constexpr size_t INDEX_GMM_INPUT_WEIGHT = 1;
constexpr size_t INDEX_GMM_INPUT_BIAS = 2;
constexpr size_t INDEX_GMM_INPUT_SCALE = 3;
constexpr size_t INDEX_GMM_INPUT_OFFSET = 4;
constexpr size_t INDEX_GMM_INPUT_ANTIQUANT_SCALE = 5;
constexpr size_t INDEX_GMM_INPUT_ANTIQUANT_OFFSET = 6;
constexpr size_t INDEX_GMM_INPUT_GROUP_LIST = 7;
constexpr size_t INDEX_GMM_INPUT_PER_TOKEN_SCALE = 8;
constexpr size_t INDEX_GMM_OUTPUT_Y = 0;
constexpr size_t INDEX_GMM_ATTR_SPLIT_ITEM = 0;
constexpr size_t INDEX_GMM_ATTR_TRANSPOSE_WEIGHT = 2;
constexpr size_t INDEX_GMM_ATTR_TRANSPOSE_X = 3;
constexpr size_t INDEX_GMM_ATTR_GROUP_TYPE = 4;
constexpr size_t INDEX_GMM_ATTR_GROUP_LIST_TYPE = 5;
constexpr size_t INDEX_GMM_ATTR_ACT_TYPE = 6;
constexpr size_t INDEX_GMM_ATTR_TUNING_CONFIG = 7;
constexpr size_t MXFP_TPYEM_SCALE_DIM_NUM = 4;
constexpr size_t MXFP_TPYEK_PER_TOKEN_SCALE_DIM_NUM = 3;
constexpr size_t MXFP_MULTI_BASE_SIZE = 2;
constexpr size_t TYPEM_SCALE_K_DIM_INDEX = 1;
constexpr size_t TYPEM_SCALE_N_DIM_INDEX = 2;
constexpr size_t PENULTIMATE_DIM = 2;
constexpr int64_t GMM_SPLIT_K = 2;
constexpr int64_t GMM_SPLIT_M = 0;
constexpr int64_t PERTILE_GROUP_SIZE = 128;

static inline aclDataType ToAclDataTypeForGMM(ge::DataType dtype) {
  static const std::vector<DataType> GMM_CONVERT_TO_ACL_DataType_LIST = {
    ge::DataType::DT_FLOAT8_E4M3FN, ge::DataType::DT_FLOAT8_E5M2,
    ge::DataType::DT_HIFLOAT8, ge::DataType::DT_FLOAT8_E8M0,
    ge::DataType::DT_FLOAT4_E2M1};
  auto iter = std::find(GMM_CONVERT_TO_ACL_DataType_LIST.begin(), GMM_CONVERT_TO_ACL_DataType_LIST.end(), dtype);
  if (iter == GMM_CONVERT_TO_ACL_DataType_LIST.end()) {
    return aclDataType::ACL_DT_UNDEFINED;
  }
  return static_cast<aclDataType>(dtype);
}

static inline aclTensorList* ConvertType(aclTensorList* geTensorList) {
  return geTensorList;
}

static bool IsFP8FP4BitsDataType(ge::DataType dataType)
{
  return dataType == ge::DataType::DT_FLOAT8_E4M3FN || dataType == ge::DataType::DT_FLOAT8_E5M2 ||
         dataType == ge::DataType::DT_HIFLOAT8 || dataType == ge::DataType::DT_FLOAT8_E8M0 ||
         dataType == ge::DataType::DT_FLOAT4_E2M1;
}

static bool SetShapeAndStrideForMXFPTypeKForGMM(std::vector<int64_t> &viewShape, std::vector<int64_t> &strides) {
  OP_CHECK_IF(viewShape.size() < MXFP_TPYEK_PER_TOKEN_SCALE_DIM_NUM,
            OP_LOGE("GroupedMatmul aclnnfallback",
                      "When split k, the dim num should be greater than 2 in mx quant mode, but the actual is %zu.",
                      viewShape.size()),
            return false);
  std::swap(viewShape[0], viewShape[1]);
  strides[0] = MXFP_MULTI_BASE_SIZE;
  strides[1] = viewShape[0] * MXFP_MULTI_BASE_SIZE;
  strides[viewShape.size() - 1] = 1;
  return true;
}

static bool SetShapeAndStrideForMXFPTypeMForGMM(std::vector<int64_t> &viewShape, std::vector<int64_t> &strides) {
  OP_CHECK_IF(viewShape.size() < MXFP_TPYEM_SCALE_DIM_NUM,
            OP_LOGE("GroupedMatmul aclnnfallback",
                      "When split m, the dim num should be greater than 3 in mx quant mode, but the actual is %zu.",
                      viewShape.size()),
            return false);
  std::swap(viewShape[TYPEM_SCALE_K_DIM_INDEX], viewShape[TYPEM_SCALE_N_DIM_INDEX]);
  strides[TYPEM_SCALE_K_DIM_INDEX] = MXFP_MULTI_BASE_SIZE;
  strides[TYPEM_SCALE_N_DIM_INDEX] = viewShape[TYPEM_SCALE_K_DIM_INDEX] * MXFP_MULTI_BASE_SIZE;
  strides[viewShape.size() - 1] = 1;
  return true;
}

static bool SetShapeAndStrideForLastTwoDimForGMM(std::vector<int64_t> &viewShape, std::vector<int64_t> &strides)
{
  OP_CHECK_IF(viewShape.size() < 2,
            OP_LOGE("GroupedMatmul aclnnfallback", "The dim num should be greater than 2, but the actual is %zu.",
                      viewShape.size()),
            return false);
  // when tensor is transposed, last two dims in strides and viewShape should swap
  auto dimM = viewShape.size() - 2;
  auto dimN = viewShape.size() - 1;
  std::swap(strides[dimM], strides[dimN]);
  // swap viewShape
  std::swap(viewShape[dimM], viewShape[dimN]);
  return true;
}

static void InitShapeAndStrideForGMM(const gert::Shape &originShape, std::vector<int64_t> &viewShape,
                                     std::vector<int64_t> &strides)
{
  for (size_t i = 0; i < originShape.GetDimNum(); ++i) {
    viewShape.push_back(originShape.GetDim(i));
  }
  strides.resize(viewShape.size(), 1);
  // Compute the strides of contiguous tensor
  for (int64_t i = viewShape.size() - 2; i >= 0; i--) {
    strides[i] = viewShape[i + 1] * strides[i + 1];
  }
}

static inline aclTensor *GeTensor2AclTensor(const gert::Tensor *geTensor, bool enableTranspose, size_t index,
                                            bool enableNZ = false)
{
  OP_CHECK_IF(geTensor == nullptr, OP_LOGE("GroupedMatmul aclnnfallback", "geTensor nullptr"), return nullptr);
  auto storageShape = geTensor->GetStorageShape();
  if (storageShape.GetDimNum() <= 1) {
    return ConvertType(geTensor);
  }
  std::vector<int64_t> storageShapeVec;
  for (size_t i = 0; i < storageShape.GetDimNum(); ++i) {
    storageShapeVec.push_back(storageShape.GetDim(i));
  }

  static const auto aclCreateTensor = GET_OP_API_FUNC(aclCreateTensor);
  OP_CHECK_IF(aclCreateTensor == nullptr, OP_LOGE("aclnnfallback", "aclCreateTensor nullptr"), return nullptr);

  void* deviceAddr = const_cast<void*>(geTensor->GetAddr());
  // convert data type
  auto dataTypeGE = geTensor->GetDataType();
  aclDataType dataType = ACL_DT_UNDEFINED;
  if (IsFP8FP4BitsDataType(dataTypeGE)) {
    dataType = ToAclDataTypeForGMM(dataTypeGE);
  } else {
    dataType = ToAclDataType(dataTypeGE);
  }
  // convert view shape
  const gert::Shape &origin_shape = geTensor->GetOriginShape();
  std::vector<int64_t> viewShape;
  std::vector<int64_t> strides;
  InitShapeAndStrideForGMM(origin_shape, viewShape, strides);
  if (enableTranspose) {
    if (dataTypeGE == ge::DataType::DT_FLOAT8_E8M0) {
      if (index == INDEX_GMM_INPUT_PER_TOKEN_SCALE) {
        OP_CHECK_IF(!SetShapeAndStrideForMXFPTypeKForGMM(viewShape, strides),
                  OP_LOGE("GroupedMatmul aclnnfallback", "SetShapeAndStrideForMXFPTypeKForGMM failed"),
                  return nullptr);
      } else if (index == INDEX_GMM_INPUT_SCALE) {
        OP_CHECK_IF(!SetShapeAndStrideForMXFPTypeMForGMM(viewShape, strides),
                  OP_LOGE("GroupedMatmul aclnnfallback", "SetShapeAndStrideForMXFPTypeMForGMM failed"),
                  return nullptr);
      }
    } else {
      OP_CHECK_IF(!SetShapeAndStrideForLastTwoDimForGMM(viewShape, strides),
                OP_LOGE("GroupedMatmul aclnnfallback", "SetShapeAndStrideForLastTwoDimForGMM failed"),
                return nullptr);
    }
  }
  auto aclFormat = aclFormat::ACL_FORMAT_ND;
  if (enableNZ && GetPrimaryFormat(geTensor->GetStorageFormat()) == ge::Format::FORMAT_FRACTAL_NZ) {
    aclFormat = aclFormat::ACL_FORMAT_FRACTAL_NZ;
  }
  aclTensor* out = aclCreateTensor(viewShape.data(), viewShape.size(), dataType, strides.data(),
                                   0, aclFormat, storageShapeVec.data(), storageShapeVec.size(), deviceAddr);
  OP_CHECK_IF(out == nullptr, OP_LOGE("aclnnfallback", "out nullptr"), return nullptr);
  return out;
}

static graphStatus PrepareGeTensorVector(const OpExecuteContext *host_api_ctx,
                                         std::vector<const gert::Tensor *> &tensorVector, size_t index)
{
  size_t cnt = 0;
  while (true) {
    auto inputGe = host_api_ctx->GetDynamicInputTensor(index, cnt);
    if (inputGe == nullptr) {
      break;
    }
    tensorVector.push_back(inputGe);
    cnt++;
  }
  return GRAPH_SUCCESS;
}

static graphStatus PrepareAclTensorVector(const OpExecuteContext *host_api_ctx, std::vector<const aclTensor *> &tensorVector,
                                          size_t index, bool enableTranspose, bool enableNZ)
{
  size_t cnt = 0;
  while (true) {
    auto inputGe = host_api_ctx->GetDynamicInputTensor(index, cnt);
    if (inputGe == nullptr) {
      break;
    }
    auto inputAcl = GeTensor2AclTensor(inputGe, enableTranspose, index, enableNZ);
    tensorVector.push_back(inputAcl);
    cnt++;
  }
  return GRAPH_SUCCESS;
}

static graphStatus PrepareOutputTensorVector(const OpExecuteContext *host_api_ctx,
                                             std::vector<const gert::Tensor *> &tensorVector, size_t index,
                                             size_t numGeWeight, int32_t splitItem)
{
  size_t numGeY = 0;
  if (0 == splitItem || 1 == splitItem) { // Length of tensorListY equals that of tensorListWeight when split_item = 0 / 1
    numGeY = numGeWeight;
  }
  else if (2 == splitItem || 3 == splitItem) { // Length of tensorListY equals 1 when split_item = 2 / 3
    numGeY = 1;
  }
  else {
    OP_LOGE("aclnnfallback", "Invalid value of split_item: %d, which must be one of 0/1/2/3.", splitItem);
    return GRAPH_FAILED;
  }

  for (size_t k = 0; k < numGeY; k++) {
    auto outputGe = host_api_ctx->GetOutputTensor(index + k);
    if (outputGe == nullptr) {return GRAPH_FAILED;}
    tensorVector.push_back(outputGe);
  }
  return GRAPH_SUCCESS;
}

static bool IsPerTileQuantMode(const OpExecuteContext* host_api_ctx, bool transposeX, bool transposeWeight, int64_t groupType)
{
  auto perTokenScaleTensor = host_api_ctx->GetOptionalInputTensor(INDEX_GMM_INPUT_PER_TOKEN_SCALE);
  auto x = host_api_ctx->GetDynamicInputTensor(INDEX_GMM_INPUT_X, 0);
  auto weight = host_api_ctx->GetDynamicInputTensor(INDEX_GMM_INPUT_WEIGHT, 0);
  auto scale = host_api_ctx->GetDynamicInputTensor(INDEX_GMM_INPUT_SCALE, 0);
  auto groupListTensor = host_api_ctx->GetOptionalInputTensor(INDEX_GMM_INPUT_GROUP_LIST);
  if (x == nullptr || weight == nullptr || scale == nullptr ||
    perTokenScaleTensor == nullptr || groupListTensor==nullptr) {
    return false;
  }
  auto &perTokenScaleOriginShape = perTokenScaleTensor->GetOriginShape();
  auto perTokenScaleDimNum = perTokenScaleOriginShape.GetDimNum();

  auto &xOriginShape = x->GetOriginShape();
  auto xDimNum = xOriginShape.GetDimNum();
  auto perTokenMDim = transposeX ?
    perTokenScaleOriginShape.GetDim(xDimNum - 1) : perTokenScaleOriginShape.GetDim(xDimNum - PENULTIMATE_DIM);
  auto xMDim = transposeX ? xOriginShape.GetDim(xDimNum - 1) : xOriginShape.GetDim(xDimNum - PENULTIMATE_DIM);
  // 2 is the minimum dimension for perTokenScale in perTile case
  if ((perTokenScaleDimNum < 2 || perTokenScaleDimNum != xDimNum) || perTokenMDim != xMDim) {
    return false;
  }
  auto &scaleShape = scale->GetOriginShape();
  auto scaleDimNum = scaleShape.GetDimNum();
  auto &weightShape = weight->GetOriginShape();
  auto weightDimNum = weightShape.GetDimNum();
  // 2 is the minimum dimension for scale in perTile case
  if (scaleDimNum < 2 || scaleDimNum != weightDimNum) {
    return false;
  }
  auto scaleKDim = transposeWeight ?
    scaleShape.GetDim(scaleDimNum - 1) : scaleShape.GetDim(scaleDimNum - PENULTIMATE_DIM);
  auto scaleNDim = transposeWeight ?
    scaleShape.GetDim(scaleDimNum - PENULTIMATE_DIM) : scaleShape.GetDim(scaleDimNum - 1);
  auto weightKDim = transposeWeight ?
    weightShape.GetDim(weightDimNum - 1) : weightShape.GetDim(weightDimNum - PENULTIMATE_DIM);
  auto weightNDim = transposeWeight ?
    weightShape.GetDim(weightDimNum - PENULTIMATE_DIM) : weightShape.GetDim(weightDimNum - 1);
  auto groupNum = groupListTensor->GetOriginShape().GetDim(0);
  bool isKdimValid =
    ((groupType == GMM_SPLIT_K && scaleKDim == (weightKDim / PERTILE_GROUP_SIZE) + groupNum) ||
     (groupType == GMM_SPLIT_M && scaleKDim == (weightKDim + PERTILE_GROUP_SIZE - 1) / PERTILE_GROUP_SIZE));
  return (scaleNDim == (weightNDim + PERTILE_GROUP_SIZE - 1) / PERTILE_GROUP_SIZE && isKdimValid);
}

static graphStatus GroupedMatmulExecuteFunc(OpExecuteContext* host_api_ctx)
{
  OP_CHECK_IF(host_api_ctx == nullptr, OP_LOGE("aclnnfallback", "host_api_ctx is null"), return GRAPH_FAILED);

  auto attrs = host_api_ctx->GetAttrs();
  OP_CHECK_IF(attrs == nullptr, OP_LOGE("aclnnfallback", "attrs is null"), return GRAPH_FAILED);
  const int64_t* splitItemGe = attrs->GetAttrPointer<int64_t>(INDEX_GMM_ATTR_SPLIT_ITEM);
  OP_CHECK_IF(splitItemGe == nullptr, OP_LOGE("aclnnfallback", "splitItemGe is null"), return GRAPH_FAILED);
  const bool* isWeightTransposedPtr = attrs->GetAttrPointer<bool>(INDEX_GMM_ATTR_TRANSPOSE_WEIGHT);
  OP_CHECK_IF(isWeightTransposedPtr == nullptr, OP_LOGE("aclnnfallback", "isWeightTransposedPtr is null"),
            return GRAPH_FAILED);
  const bool* isXTransposed = attrs->GetAttrPointer<bool>(INDEX_GMM_ATTR_TRANSPOSE_X);
  OP_CHECK_IF(isXTransposed == nullptr, OP_LOGE("aclnnfallback", "isXTransposed is null"), return GRAPH_FAILED);
  const int64_t* groupTypeGe = attrs->GetAttrPointer<int64_t>(INDEX_GMM_ATTR_GROUP_TYPE);
  OP_CHECK_IF(groupTypeGe == nullptr, OP_LOGE("aclnnfallback", "groupTypeGe is null"), return GRAPH_FAILED);
  const int64_t* groupListTypeGe = attrs->GetAttrPointer<int64_t>(INDEX_GMM_ATTR_GROUP_LIST_TYPE);
  OP_CHECK_IF(groupListTypeGe == nullptr, OP_LOGE("aclnnfallback", "groupListTypeGe is null"), return GRAPH_FAILED);
  const int64_t* actTypeGe = attrs->GetAttrPointer<int64_t>(INDEX_GMM_ATTR_ACT_TYPE);
  OP_CHECK_IF(actTypeGe == nullptr, OP_LOGE("aclnnfallback", "actTypeGe is null"), return GRAPH_FAILED);
  const auto tuningConfigGe = attrs->GetListInt(INDEX_GMM_ATTR_TUNING_CONFIG);
  OP_CHECK_IF(tuningConfigGe == nullptr, OP_LOGE("aclnnfallback", "tuningConfigGe is null"), return GRAPH_FAILED);

  static const auto aclCreateTensorList = GET_OP_API_FUNC(aclCreateTensorList);
  OP_CHECK_IF(aclCreateTensorList == nullptr,
            OP_LOGE("aclnnfallback", "Get opapi func aclCreateTensorList failed"), return GRAPH_FAILED);

  std::vector<const aclTensor*> aclTensorVectorX;
  PrepareAclTensorVector(host_api_ctx, aclTensorVectorX, INDEX_GMM_INPUT_X, *isXTransposed, false);
  auto aclTensorListX = aclCreateTensorList(aclTensorVectorX.data(), aclTensorVectorX.size());

  std::vector<const aclTensor*> aclTensorVectorWeight;
  PrepareAclTensorVector(host_api_ctx, aclTensorVectorWeight, INDEX_GMM_INPUT_WEIGHT, *isWeightTransposedPtr, true);
  size_t numGeWeight = aclTensorVectorWeight.size();
  auto aclTensorListWeight = aclCreateTensorList(aclTensorVectorWeight.data(), aclTensorVectorWeight.size());

  std::vector<const gert::Tensor*> geTensorVectorBias;
  PrepareGeTensorVector(host_api_ctx, geTensorVectorBias, INDEX_GMM_INPUT_BIAS);
  bool isScaleTransposed = false;
  const bool* isScaleTransposedPtr = &isScaleTransposed;
  auto scaleTensor = host_api_ctx->GetDynamicInputTensor(INDEX_GMM_INPUT_SCALE, 0);
  bool isPerTile = IsPerTileQuantMode(host_api_ctx, *isXTransposed, *isWeightTransposedPtr, *groupTypeGe);
  if (scaleTensor != nullptr && (scaleTensor->GetDataType() == ge::DataType::DT_FLOAT8_E8M0 || isPerTile)) {
    isScaleTransposedPtr = isWeightTransposedPtr;
  }
  std::vector<const aclTensor*> aclTensorVectorScale;
  PrepareAclTensorVector(host_api_ctx, aclTensorVectorScale, INDEX_GMM_INPUT_SCALE, *isScaleTransposedPtr, false);
  auto aclTensorListScale = aclCreateTensorList(aclTensorVectorScale.data(), aclTensorVectorScale.size());

  std::vector<const gert::Tensor*> geTensorVectorOffset;
  PrepareGeTensorVector(host_api_ctx, geTensorVectorOffset, INDEX_GMM_INPUT_OFFSET);

  std::vector<const gert::Tensor*> geTensorVectorAntiquantScale;
  PrepareGeTensorVector(host_api_ctx, geTensorVectorAntiquantScale, INDEX_GMM_INPUT_ANTIQUANT_SCALE);

  std::vector<const gert::Tensor*> geTensorVectorAntiquantOffset;
  PrepareGeTensorVector(host_api_ctx, geTensorVectorAntiquantOffset, INDEX_GMM_INPUT_ANTIQUANT_OFFSET);

  auto groupListTensor = host_api_ctx->GetOptionalInputTensor(INDEX_GMM_INPUT_GROUP_LIST);

  auto perTokenScaleTensor = host_api_ctx->GetOptionalInputTensor(INDEX_GMM_INPUT_PER_TOKEN_SCALE);
  std::vector<const aclTensor*> geTensorVectorPerTokenScale;
  auto perTokenScale = ConvertType(perTokenScaleTensor);
  if (perTokenScale == nullptr) {
    std::vector<int64_t> shape{0};
    static const auto aclCreateTensor = GET_OP_API_FUNC(aclCreateTensor);
    OP_CHECK_IF(aclCreateTensor == nullptr, OP_LOGE("aclnnfallback", "aclCreateTensor nullptr"), return GRAPH_FAILED);
    perTokenScale = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT, shape.data(),
                                    0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), nullptr);
    OP_CHECK_IF(perTokenScale == nullptr, OP_LOGE("aclnnfallback", "perTokenScale nullptr"), return GRAPH_FAILED);
  }
  if (perTokenScaleTensor != nullptr && (perTokenScaleTensor->GetDataType() == ge::DataType::DT_FLOAT8_E8M0 ||
                                         isPerTile)) {
    PrepareAclTensorVector(host_api_ctx, geTensorVectorPerTokenScale, INDEX_GMM_INPUT_PER_TOKEN_SCALE,
                           *isXTransposed, false);
  } else {
    geTensorVectorPerTokenScale.push_back(perTokenScale);
  }
  auto aclTensorListPerTokenScale = aclCreateTensorList(geTensorVectorPerTokenScale.data(),
                                                        geTensorVectorPerTokenScale.size());

  std::vector<const gert::Tensor*> geTensorVectorY;
  PrepareOutputTensorVector(host_api_ctx, geTensorVectorY, INDEX_GMM_OUTPUT_Y, numGeWeight, *splitItemGe);

  aclTensorList* activationInputOptional = nullptr;
  aclTensorList* activationQuantScaleOptional = nullptr;
  aclTensorList* activationQuantOffsetOptional = nullptr;
  aclTensorList* actFeatureOutOptional = nullptr;
  aclTensorList* dynQuantScaleOutOptional = nullptr;
  std::vector<int64_t> tuningConfig;
  tuningConfig.reserve(1);
  tuningConfig.push_back(tuningConfigGe->GetData()[0]);

  // execute opapi
  auto api_ret_gmm = EXEC_OPAPI_CMD(aclnnGroupedMatmulV5, aclTensorListX, aclTensorListWeight, geTensorVectorBias,
                                aclTensorListScale, geTensorVectorOffset, geTensorVectorAntiquantScale,
                                geTensorVectorAntiquantOffset, aclTensorListPerTokenScale, groupListTensor,
                                activationInputOptional, activationQuantScaleOptional, activationQuantOffsetOptional,
                                *splitItemGe, *groupTypeGe, *groupListTypeGe, *actTypeGe, tuningConfig,
                                geTensorVectorY, actFeatureOutOptional, dynQuantScaleOutOptional);
  OP_CHECK_IF(api_ret_gmm != GRAPH_SUCCESS, OP_LOGE("aclnnfallback", "api_ret failed:%u", api_ret_gmm),
            return GRAPH_FAILED);
  return GRAPH_SUCCESS;
}

IMPL_OP(GroupedMatmul).OpExecuteFunc(GroupedMatmulExecuteFunc);

}  // namespace fallback

#ifdef __cplusplus
}
#endif
