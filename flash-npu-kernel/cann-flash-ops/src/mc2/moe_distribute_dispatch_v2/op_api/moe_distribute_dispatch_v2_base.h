/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file moe_distribute_dispatch_v2_base.h
 * \brief
 */

#ifndef MOE_DISTRIBUTE_DISPATCH_V2_BASE_
#define MOE_DISTRIBUTE_DISPATCH_V2_BASE_

#include <string>

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"
#include "common/op_host/op_api/matmul_util.h"

#ifdef __cplusplus
extern "C" {
#endif

static constexpr int32_t DISPATCH_DYNAMIC_QUANT_MODE = 2;
enum NnopbaseHcclServerType {
    NNOPBASE_HCCL_SERVER_TYPE_AICPU = 0,
    NNOPBASE_HCCL_SERVER_TYPE_MTE,
    NNOPBASE_HCCL_SERVER_TYPE_CCU,
    NNOPBASE_HCCL_SERVER_TYPE_END
};

// check nullptr
ACLNN_API bool CustomDispatchCheckNotNull(const aclTensor* x, const aclTensor* expertIds, const char* groupEp,
                                    [[maybe_unused]] const char* groupTp, aclTensor* expandX, [[maybe_unused]] aclTensor* dynamicScales,
                                    aclTensor* assistInfoForCombine, aclTensor* expertTokensNums,
                                    aclTensor* epRecvCounts, aclTensor* tpRecvCounts);

// 入参教验
ACLNN_API aclnnStatus CustomDispatchCheckParams(const aclTensor* x, const aclTensor* expertIds,
                                          const char* groupEp, const char* groupTp, int64_t quantMode,
                                          aclTensor* expandX, aclTensor* dynamicScales, aclTensor* assistInfoForCombine,
                                          aclTensor* expertTokensNums, aclTensor* epRecvCounts, aclTensor* tpRecvCounts);

ACLNN_API aclnnStatus aclnnCustomMoeDistributeDispatchGetWorkspaceSizeBase(
    const aclTensor* x, const aclTensor* expertIds, const aclTensor* scalesOptional,
    const aclTensor* xActiveMaskOptional, const aclTensor* expertScalesOptional,
    const aclTensor* elasticInfoOptional, const char* groupEp,
    int64_t epWorldSize, int64_t epRankId, int64_t moeExpertNum, const char* groupTp, int64_t tpWorldSize,
    int64_t tpRankId, int64_t expertShardType, int64_t sharedExpertNum, int64_t sharedExpertRankNum,
    int64_t quantMode, int64_t globalBs, int64_t expertTokenNumsType, const char* commAlg,
    int64_t zeroExpertNum, int64_t copyExpertNum, int64_t constExpertNum, aclTensor* expandXOut,
    aclTensor* dynamicScalesOut, aclTensor* assistInfoForCombineOut, aclTensor* expertTokenNumsOut, aclTensor* epRecvCountsOut,
    aclTensor* tpRecvCountsOut, aclTensor* expandScalesOut,
    uint64_t* workspaceSize, aclOpExecutor** executor);

#ifdef __cplusplus
}
#endif
#endif //MOE_DISTRIBUTE_DISPATCH_V2_BASE_
