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
 * \file moe_distribute_dispatch_v2_base.cpp
 * \brief
 */

#include <algorithm>
#include "op_mc2.h"
#include "op_mc2_def.h"
#include "opdev/op_log.h"
#include "opdev/common_types.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "moe_distribute_dispatch_v2_base.h"

using namespace Ops::Transformer;
using namespace op;
#ifdef __cplusplus
extern "C" {
#endif
extern aclnnStatus aclnnCustomInnerMoeDistributeDispatchV2GetWorkspaceSize(
    const aclTensor* x, const aclTensor* expertIds, const aclTensor* scales,
    const aclTensor* xActiveMask, const aclTensor* expertScales,  const aclTensor* elasticInfo,
    const char* groupEp, int64_t epWorldSize,
    int64_t epRankId, int64_t moeExpertNum, const char* groupTp, int64_t tpWorldSize,
    int64_t tpRankId, int64_t expertShardType, int64_t sharedExpertNum, int64_t shareExpertRankNum,
    int64_t quantMode, int64_t globalBs, int64_t expertTokenNumsType, const char* commAlg,
    int64_t zeroExpertNum, int64_t copyExpertNum, int64_t constExpertNum, int64_t ydtype, aclTensor* expandX,
    aclTensor* dynamicScales, aclTensor* assist_info_for_combine, aclTensor* expertTokensNums, aclTensor* epRecvCounts,
    aclTensor* tpRecvCounts, aclTensor* expandScales,
    uint64_t* workspaceSize, aclOpExecutor** executor);

extern "C" void __attribute__((weak)) NnopbaseSetHcclServerType(void *executor, NnopbaseHcclServerType sType);

bool CustomDispatchCheckNotNull(const aclTensor* x, const aclTensor* expertIds, const char* groupEp,
                          [[maybe_unused]] const char* groupTp, aclTensor* expandX, [[maybe_unused]] aclTensor* dynamicScales,
                          aclTensor* assistInfoForCombine, aclTensor* expertTokensNums, aclTensor* epRecvCounts,
                          aclTensor* tpRecvCounts)
{
    OP_CHECK_NULL(x, return false);
    OP_CHECK_NULL(expertIds, return false);
    OP_CHECK_NULL(expandX, return false);
    OP_CHECK_NULL(assistInfoForCombine, return false);
    OP_CHECK_NULL(expertTokensNums, return false);
    OP_CHECK_NULL(tpRecvCounts, return false);
    OP_CHECK_NULL(epRecvCounts, return false);
    if ((groupEp == nullptr)||(strnlen(groupEp, HCCL_GROUP_NAME_MAX) == 0)) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "group groupEp name is Empty");
        return false;
    }
    return true;
}

aclnnStatus CustomDispatchCheckParams(const aclTensor* x, const aclTensor* expertIds, const char* groupEp, const char* groupTp,
                                int64_t quantMode, aclTensor* expandX, aclTensor* dynamicScales, aclTensor* assistInfoForCombine,
                                aclTensor* expertTokensNums, aclTensor* epRecvCounts, aclTensor* tpRecvCounts)
{
    CHECK_RET(CustomDispatchCheckNotNull(x, expertIds, groupEp, groupTp, expandX, dynamicScales, assistInfoForCombine,
        expertTokensNums, epRecvCounts, tpRecvCounts), ACLNN_ERR_PARAM_NULLPTR);

    if (quantMode == DISPATCH_DYNAMIC_QUANT_MODE) {
        OP_LOGD("quantMode = 2, dynamicScales can't be null");
        CHECK_RET(dynamicScales != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    }
    if (strnlen(groupEp, HCCL_GROUP_NAME_MAX) >= HCCL_GROUP_NAME_MAX) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Required groupEp name exceeds %zu", HCCL_GROUP_NAME_MAX);
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (strnlen(groupTp, HCCL_GROUP_NAME_MAX) >= HCCL_GROUP_NAME_MAX) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Required groupTp name exceeds %zu", HCCL_GROUP_NAME_MAX);
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnCustomMoeDistributeDispatchGetWorkspaceSizeBase(
    const aclTensor* x, const aclTensor* expertIds, const aclTensor* scalesOptional,
    const aclTensor* xActiveMaskOptional, const aclTensor* expertScalesOptional,
    const aclTensor* elasticInfoOptional, const char* groupEp,
    int64_t epWorldSize, int64_t epRankId, int64_t moeExpertNum, const char* groupTp, int64_t tpWorldSize,
    int64_t tpRankId, int64_t expertShardType, int64_t sharedExpertNum, int64_t sharedExpertRankNum,
    int64_t quantMode, int64_t globalBs, int64_t expertTokenNumsType, const char* commAlg,
    int64_t zeroExpertNum, int64_t copyExpertNum, int64_t constExpertNum, aclTensor* expandXOut,
    aclTensor* dynamicScalesOut, aclTensor* assistInfoForCombineOut, aclTensor* expertTokenNumsOut, aclTensor* epRecvCountsOut,
    aclTensor* tpRecvCountsOut, aclTensor* expandScalesOut,
    uint64_t* workspaceSize, aclOpExecutor** executor)
{
    const static bool is910B = GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B;
    const static bool is950 = GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510;
    auto retParam = CustomDispatchCheckParams(x, expertIds, groupEp, groupTp, quantMode, expandXOut, dynamicScalesOut,
                                         assistInfoForCombineOut, expertTokenNumsOut, epRecvCountsOut, tpRecvCountsOut);
    CHECK_RET(retParam == ACLNN_SUCCESS, retParam);

    const char* groupTpDispatchV2Temp = groupTp;
    if (is910B) {
        groupTpDispatchV2Temp = "";
    }

    int64_t ydtype = expandXOut->GetDataType();
    aclnnStatus getWorkspaceSizesRes = aclnnCustomInnerMoeDistributeDispatchV2GetWorkspaceSize(
        x, expertIds, scalesOptional, xActiveMaskOptional, expertScalesOptional,
        elasticInfoOptional, groupEp, epWorldSize, epRankId, moeExpertNum,
        groupTpDispatchV2Temp, tpWorldSize, tpRankId, expertShardType, sharedExpertNum,
        sharedExpertRankNum, quantMode, globalBs, expertTokenNumsType, commAlg, zeroExpertNum, copyExpertNum,
        constExpertNum, ydtype, expandXOut, dynamicScalesOut, assistInfoForCombineOut, expertTokenNumsOut,
        epRecvCountsOut, tpRecvCountsOut, expandScalesOut, workspaceSize, executor);

    if (NnopbaseSetHcclServerType) {
        if (is910B) {
            NnopbaseSetHcclServerType(*executor, NNOPBASE_HCCL_SERVER_TYPE_MTE);
        } else if (is950 && commAlg != nullptr && std::strcmp(commAlg, "ccu") == 0) {
            NnopbaseSetHcclServerType(*executor, NNOPBASE_HCCL_SERVER_TYPE_CCU);
        } else {
            NnopbaseSetHcclServerType(*executor, NNOPBASE_HCCL_SERVER_TYPE_MTE);
        }
    }
    return getWorkspaceSizesRes;
}
#ifdef __cplusplus
}
#endif
