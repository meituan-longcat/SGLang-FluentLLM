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
 * \file moe_distribute_combine_v2_gen_task.cpp
 * \brief
 */
#include <vector>
#include <set>
#include <string>

#ifdef BUILD_OPEN_PROJECT
#include "mc2_gen_task_ops_utils.h"
#include "mc2_moe_gen_task_ops_utils.h"
#include "mc2_gen_task_ops_utils_arch35.h"
#include "register/op_impl_registry.h"
#include "mc2_log.h"
#else
#include "ops_error.h"
#include "mc2_gen_task_moe.h"
#include "mc2_gen_task_utils.h"
#include "mc2_a5_gen_task_utils.h"
#include "register/op_ct_impl_registry.h"
#include "register/op_ext_gentask_registry.h"
#endif

namespace ops {
static const size_t ATTR_INDEX_COMM_ALG_DISTRIBUTE_COMBINE_V2 = 14;
#ifdef BUILD_OPEN_PROJECT
ge::Status MoeDistributeCombineV2CalcParamFunc(gert::ExeResGenerationContext *context)
{
    if ((Mc2GenTaskOpsUtils::IsTargetPlatformNpuArch(context->GetNodeName(), NPUARCH_A5)) &&
        (Mc2Arch35GenTaskOpsUtils::GetCommAlg(context, ATTR_INDEX_COMM_ALG_DISTRIBUTE_COMBINE_V2) != COMM_ALG_MTE)) {
        OPS_LOG_D(context->GetNodeName(), "Do A5 ccu calc param.");
        return Mc2GenTaskOpsUtils::CommonKFCMc2CalcParamFunc(context, "ccu server", "ccu_stream");
    }
    OPS_LOG_D(context->GetNodeName(), "Do general calc param.");
    return Mc2GenTaskOpsUtils::CommonKFCMc2CalcParamFunc(context, "aicpu kfc server", "kfc_stream");
}

ge::Status MoeDistributeCombineV2GenTaskFunc(const gert::ExeResGenerationContext *context,
                                             std::vector<std::vector<uint8_t>> &tasks)
{
    const char *nodeName = context->GetNodeName();
    if (Mc2GenTaskOpsUtils::IsTargetPlatformSocVersion(nodeName, PLATFORM_A2)) {
        OPS_LOG_D(nodeName, "Do A2 gen task.");
        return Mc2MoeGenTaskOpsUtils::Mc2MoeGenTaskCallback(context, tasks);
    }
    if (Mc2GenTaskOpsUtils::IsTargetPlatformNpuArch(nodeName, NPUARCH_A5)) {
        const std::string commAlg =
            Mc2Arch35GenTaskOpsUtils::GetCommAlg(context, ATTR_INDEX_COMM_ALG_DISTRIBUTE_COMBINE_V2);
        if (commAlg == COMM_ALG_MTE) {
            OPS_LOG_D(nodeName, "Do A5 mte gen task.");
            return Mc2MoeGenTaskOpsUtils::Mc2MoeGenTaskCallbackV2(context, tasks);
        }
        if (commAlg == COMM_ALG_CCU) {
            OPS_LOG_D(nodeName, "Do A5 CCU gen task.");
            return Mc2Arch35GenTaskOpsUtils::Mc2Arch35GenTaskCallBack(context, tasks);
        }
        OPS_LOG_E(nodeName, "Got unsupported commAlg %s.", commAlg.c_str());
        return ge::GRAPH_FAILED;
    }
    OPS_LOG_D(nodeName, "Do A3 gen task.");
    return Mc2MoeGenTaskOpsUtils::Mc2MoeGenTaskCallbackV2(context, tasks);
}

// new ver
IMPL_OP(MoeDistributeCombineV2)
    .CalcOpParam(MoeDistributeCombineV2CalcParamFunc)
    .GenerateTask(MoeDistributeCombineV2GenTaskFunc);
#else // mc2 gen task utils
ge::Status MoeDistributeCombineV2CalcParamFunc(gert::ExeResGenerationContext *context)
{
    if ((Mc2A5GenTaskUtils::IsTargetPlatformNpuArch(context->GetNodeName(), NPUARCH_A5)) &&
        (Mc2A5GenTaskUtils::GetCommAlg(context, ATTR_INDEX_COMM_ALG_DISTRIBUTE_COMBINE_V2) != COMM_ALG_MTE)) {
        OPS_LOG_D(context->GetNodeName(), "Do A5 ccu calc param.");
        return Mc2GenTaskUtils::CommonKFCMc2CalcParamFunc(context, "ccu server", "ccu_stream");
    }
    const ge::AscendString name = "aicpu kfc server";
    const ge::AscendString reuseKey = "kfc_stream";
    return Mc2GenTaskUtils::CommonKFCMc2CalcParamFunc(context, name, reuseKey);
}

ge::Status MoeDistributeCombineV2GenTaskFunc(const gert::ExeResGenerationContext *context,
                                             std::vector<std::vector<uint8_t>> &tasks)
{
    const char *nodeName = context->GetNodeName();
    if (Mc2A5GenTaskUtils::IsTargetPlatformSocVersion(context->GetNodeName(), PLATFORM_A2)) {
        return Mc2GenTaskUtils::CommonKFCMc2GenTask(context, tasks, Mc2GenTaskMoe::Mc2MoeGenTaskCallback);
    } else if (Mc2A5GenTaskUtils::IsTargetPlatformNpuArch(context->GetNodeName(), NPUARCH_A5)) {
        const std::string commAlg = Mc2A5GenTaskUtils::GetCommAlg(context, ATTR_INDEX_COMM_ALG_DISTRIBUTE_COMBINE_V2);
        if (commAlg == COMM_ALG_MTE) {
            OPS_LOG_D(context->GetNodeName(), "Do A5 mte gen task.");
            return Mc2GenTaskUtils::CommonKFCMc2GenTask(context, tasks, Mc2GenTaskMoe::Mc2MoeGenTaskCallbackV2);
        } else if (commAlg == COMM_ALG_CCU) {
            OPS_LOG_D(context->GetNodeName(), "Do A5 CCU gen task.");
            return Mc2GenTaskUtils::CommonKFCMc2GenTask(context, tasks, Mc2A5GenTaskUtils::Mc2GenTaskCallBack910A5);
        } else {
            OPS_LOG_E(context->GetNodeName(), "Got unsupported commAlg %s.", commAlg.c_str());
            return ge::GRAPH_FAILED;
        }
    }
    OPS_LOG_D(context->GetNodeName(), "Do A3 gen task.");
    return Mc2GenTaskUtils::CommonKFCMc2GenTask(context, tasks, Mc2GenTaskMoe::Mc2MoeGenTaskCallbackV2);
}

IMPL_OP_CT(MoeDistributeCombineV2)
    .CalcOpParam(MoeDistributeCombineV2CalcParamFunc)
    .GenerateTask(MoeDistributeCombineV2GenTaskFunc);
REGISTER_EXT_TASK_TYPE(MoeDistributeCombineV2, fe::ExtTaskType::kAicoreTask);
#endif
} // namespace ops
