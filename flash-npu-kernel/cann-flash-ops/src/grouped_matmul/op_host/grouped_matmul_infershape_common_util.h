/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GROUPED_MATMUL_INFERSHAPE_COMMON_H_
#define GROUPED_MATMUL_INFERSHAPE_COMMON_H_

#include "register/op_impl_registry.h"
#include "log/log.h"
#include "platform/platform_info.h"

namespace ops {

constexpr size_t GMM_INDEX_IN_X = 0UL;
constexpr size_t GMM_INDEX_IN_WEIGHT = 1UL;
constexpr size_t GMM_INDEX_IN_BIAS = 2UL;
constexpr size_t GMM_INDEX_IN_SCALE = 3UL;
constexpr size_t GMM_INDEX_IN_OFFSET = 4UL;
constexpr size_t GMM_INDEX_IN_ANTIQUANT_SCALE = 5UL;
constexpr size_t GMM_INDEX_IN_ANTIQUANT_OFFSET = 6UL;
constexpr size_t GMM_INDEX_IN_GROUP_LIST = 7UL;
constexpr size_t GMM_INDEX_IN_PERTOKEN_SCALE = 8UL;

constexpr size_t GMM_INDEX_OUT_Y = 0UL;
constexpr size_t GMM_INDEX_ATTR_SPLIT_ITEM = 0UL;
constexpr size_t GMM_INDEX_ATTR_OUTPUT_DTYPE = 1UL;
constexpr size_t GMM_INDEX_ATTR_TRANSPOSE_W = 2UL;
constexpr size_t GMM_INDEX_ATTR_TRANSPOSE_X = 3UL;
constexpr size_t GMM_INDEX_ATTR_GROUP_TYPE = 4UL;
constexpr size_t GMM_INDEX_ATTR_GROUP_LIST_TYPE = 5UL;
constexpr size_t GMM_INDEX_ATTR_ACT_TYPE = 6UL;
constexpr size_t GMM_INDEX_ATTR_TUNING_CONFIG = 7UL;

constexpr int64_t GMM_X_Y_SEPARATED = 0; // x,y have been separated
constexpr int64_t GMM_Y_SEPARATED = 1;   // y has been separated
constexpr int64_t GMM_X_SEPARATED = 2;   // x has been separated
constexpr int64_t GMM_NO_SEPARATED = 3;  // x,y have not been separated
constexpr int64_t GMM_NO_SPLIT = -1L;
constexpr int64_t GMM_SPLIT_M = 0L;
constexpr int64_t GMM_SPLIT_N = 1L;
constexpr int64_t GMM_SPLIT_K = 2L;

constexpr int64_t GMM_OUT_DTYPE_INT32 = 2L;

constexpr int64_t GMM_MAX_GROUP_LIST_SIZE_ARRAY = 128L;
constexpr int64_t GMM_MAX_GROUP_LIST_SIZE_TENSOR = 1024L;
constexpr int64_t GMM_MAX_INNER_AXIS = 65535L;
constexpr int64_t GMM_N_K_ALIGN_VALUE_WEIGHT_QUANT = 32L;
constexpr int64_t GMM_N_K_ALIGN_VALUE_WEIGHT_QUANT_4BIT = 64L;
constexpr size_t GMM_MAX_FM_DIM = 6UL;
constexpr size_t GMM_MIN_FM_DIM = 2UL;
constexpr size_t GMM_MIN_WEIGHT_DIM = 2UL;
constexpr size_t GMM_SEPARATED_WEIGHT_DIM = 2UL;
constexpr size_t GMM_SPLIT_M_SINGLE_WEIGHT_DIM = 3UL;
constexpr size_t GMM_SPLIT_K_SINGLE_WEIGHT_DIM = 2UL;
constexpr size_t PENULTIMATE_DIM = 2UL;
constexpr int64_t PERTILE_GROUP_SIZE = 128UL;
constexpr size_t GMM_A8W4_OFFSET_DIM_NUM = 3UL;
constexpr size_t GMM_A8W4_BIAS_DIM_NUM = 2UL;

constexpr int64_t MXFP_DIVISOR_SIZE = 64;
constexpr int64_t MXFP_MULTI_BASE_SIZE = 2;
constexpr int64_t MXFP_TYPEM_SCALE_DIM_NUM = 4;
constexpr int64_t MXFP_TYPEK_SCALE_DIM_NUM = 3;
struct GMMAttrs {
    int64_t splitItem;
    int64_t outputDtype;
    int64_t groupType;
    bool transposeX;
    bool transposeWeight;
    int64_t activeType;
    int64_t tuningConfig;
};

struct GMMInputParamsInfo {
    size_t numX;
    size_t numWeight;
    size_t numBias;
    size_t numScale;
    size_t numOffset;
    size_t numAntiquantScale;
    size_t numAntiquantOffset;
};

const std::map<int64_t, ge::DataType> GMM_OUTPUT_DTYPE_MAP = {{0, ge::DataType::DT_FLOAT16},
                                                              {1, ge::DataType::DT_BF16},
                                                              {2, ge::DataType::DT_INT32},
                                                              {3, ge::DataType::DT_FLOAT},
                                                              {-1, ge::DataType::DT_INT8}};

enum class GMMActType : int64_t {
    GMM_ACT_TYPE_NONE,
    GMM_ACT_TYPE_RELU,
    GMM_ACT_TYPE_GELU_TANH,
    GMM_ACT_TYPE_GELU_ERR_FUNC,
    GMM_ACT_TYPE_FAST_GELU,
    GMM_ACT_TYPE_SILU,
    END_ACT_TYPE_ENUM
};

const std::initializer_list<ge::DataType> BIAS_DTYPE_SUPPORT_LIST = {ge::DataType::DT_FLOAT, ge::DataType::DT_FLOAT16,
                                                                     ge::DataType::DT_BF16};

class GroupedMatmulCommonUtil {
public:
    GMMAttrs attrsInfo;
    GroupedMatmulCommonUtil(){};
    ~GroupedMatmulCommonUtil(){};

private:
};
} // namespace ops

#endif // GROUPED_MATMUL_INFERSHAPE_COMMON_H_