/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_GROUPED_MATMUL_UTIL_H
#define OP_API_INC_GROUPED_MATMUL_UTIL_H
#include "opdev/common_types.h"
#include "opdev/op_executor.h"

namespace gmm {
using namespace op;
constexpr int64_t NO_SPLIT = -1L;
constexpr int64_t SPLIT_M = 0L;
constexpr int64_t SPLIT_K = 2L;
constexpr int64_t SPLIT_N = 1L;

constexpr int64_t GROUP_LIST_SPARSE_M = 2L;

constexpr size_t MAX_FM_DIM = 6UL;
constexpr size_t MIN_FM_DIM = 2UL;
constexpr size_t SPLIT_M_SINGLE_WEIGHT_DIM = 3UL;
constexpr size_t SPLIT_K_SINGLE_WEIGHT_DIM = 2UL;
// mx dim num
constexpr size_t MX_SPLIT_M_SINGLE_X_DIM = 2UL;
constexpr size_t MX_SPLIT_M_SINGLE_WEIGHT_DIM = 3UL;
constexpr size_t MX_SPLIT_M_SCALE_DIM = 4UL;
constexpr size_t MX_BIAS_DIM = 2UL;
constexpr size_t MX_SPLIT_M_PER_TOKEN_SCALE_DIM = 3UL;
constexpr size_t MX_SPLIT_K_SINGLE_X_DIM = 2UL;
constexpr size_t MX_SPLIT_K_SINGLE_WEIGHT_DIM = 2UL;
constexpr size_t MX_SPLIT_K_SCALE_DIM = 3UL;
constexpr size_t MX_SPLIT_K_PER_TOKEN_SCALE_DIM = 3UL;

constexpr size_t MX_TUPLE_X_DIM_INDEX = 0UL;
constexpr size_t MX_TUPLE_WEIGHT_DIM_INDEX = 1UL;
constexpr size_t MX_TUPLE_SCALE_DIM_INDEX = 2UL;
constexpr size_t MX_TUPLE_PERTOKENSCALE_DIM_INDEX = 3UL;
constexpr size_t MX_TUPLE_GROUP_NUMBER_DIM_INDEX = 4UL;

constexpr int64_t CUBE_BLOCK_SIZE_16 = 16L;
constexpr int64_t CUBE_BLOCK_SIZE_32 = 32L;
constexpr int64_t PERTILE_GROUP_SIZE = 128L;
constexpr int64_t MXFP_DIVISOR_SIZE = 64L;
constexpr int64_t MXFP_MULTI_BASE_SIZE = 2L;
constexpr int64_t EVEN_FACTOR = 2L;
constexpr int64_t LAST_TWO_DIM_INDEX = 2L;

constexpr int64_t N_K_MAX_VALUE_WEIGHT_QUANT = 65535L;
constexpr int64_t N_K_ALIGN_VALUE_WEIGHT_QUANT_4BIT = 64L;

constexpr size_t LAST_FIRST_DIM_INDEX = 1;
constexpr size_t LAST_SECOND_DIM_INDEX = 2;
constexpr size_t LAST_THIRD_DIM_INDEX = 3;
constexpr size_t LAST_FOURTH_DIM_INDEX = 4;
constexpr size_t QUANT_WEIGHTNZ_STORAGE_DIM = 5;

enum class GMMApiVersion : uint32_t {
    V1 = 1U,
    V2 = 2U,
    V3 = 3U,
    V4 = 4U,
    V5 = 5U,
    WeightNz = 6U
};

template <typename T>
struct GroupedMatmulParamsBase {
    const T *x = nullptr;
    const T *weight = nullptr;
    const aclTensorList *biasOptional = nullptr;
    const aclIntArray *groupListOptional = nullptr;
    const aclTensor *groupTensorOptional = nullptr;
    const T *scaleOptional = nullptr;
    const aclTensorList *offsetOptional = nullptr;
    const aclTensorList *antiquantScaleOptional = nullptr;
    const aclTensorList *antiquantOffsetOptional = nullptr;
    const T *perTokenScaleOptional = nullptr;
    const aclTensorList *activationInputOptional = nullptr;
    const aclTensorList *activationQuantScaleOptional = nullptr;
    const aclTensorList *activationQuantOffsetOptional = nullptr;
    int64_t splitItem = 0;
    int64_t groupListType = 0;
    int64_t activeType = 0;
    bool transposeWeight = false;
    bool transposeX = false;
    bool isSingleWeight = false;
    GMMApiVersion apiVersion = GMMApiVersion::V1;
    int64_t groupType = -1L;
    aclIntArray *tuningConfigOptional = nullptr;
    const T *y = nullptr;
    const aclTensorList *activationFeatureOutOptional = nullptr;
    const aclTensorList *dynQuantScaleOutOptional = nullptr;
    DataType xDtype = DataType::DT_FLOAT16;
};

using GroupedMatmulParams = GroupedMatmulParamsBase<aclTensorList>;

const std::map<DataType, aclDataType> BIAS_DTYPE {
    {DataType::DT_FLOAT16, aclDataType::ACL_FLOAT16},
    {DataType::DT_BF16, aclDataType::ACL_FLOAT},
    {DataType::DT_INT8, aclDataType::ACL_INT32},
    {DataType::DT_FLOAT, aclDataType::ACL_FLOAT},
    {DataType::DT_FLOAT8_E4M3FN, aclDataType::ACL_FLOAT},
    {DataType::DT_FLOAT8_E5M2, aclDataType::ACL_FLOAT},
    {DataType::DT_HIFLOAT8, aclDataType::ACL_FLOAT},
    {DataType::DT_INT4, aclDataType::ACL_FLOAT16},
    {DataType::DT_FLOAT4_E2M1, aclDataType::ACL_FLOAT}
};

const std::map<DataType, std::string> DTYPE_STRING{
    {DataType::DT_FLOAT16, "FLOAT16"},
    {DataType::DT_BF16, "BF16"},
    {DataType::DT_INT8, "INT8"},
    {DataType::DT_INT4, "INT4"},
    {DataType::DT_FLOAT, "FLOAT"},
    {DataType::DT_FLOAT8_E4M3FN, "FLOAT8_E4M3FN"},
    {DataType::DT_FLOAT8_E5M2, "FLOAT8_E5M2"},
    {DataType::DT_HIFLOAT8, "HIFLOAT8"}
};

const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST = {DataType::DT_FLOAT, DataType::DT_FLOAT16,
                                                                DataType::DT_BF16};

bool IsTransposeLastTwoDims(const aclTensor *tensor);
bool IsTransposeForMxShape(const aclTensor *tensor);
void CreateContiguousTensorListForPertoken(const aclTensorList *tensorList, std::vector<aclTensor *> &newTensorList,
                                           aclOpExecutor *executor);
void CreateContiguousTensorListForMXTypeMScale(const aclTensorList *tensorList, std::vector<aclTensor *> &newTensorList,
                                               aclOpExecutor *executor);
void CreateContiguousTensorList(const aclTensorList *tensorList, std::vector<aclTensor *> &newTensorList,
                                aclOpExecutor *executor);
std::string dTypeToString(const ge::DataType &dtype);
} // namespace gmm
#endif