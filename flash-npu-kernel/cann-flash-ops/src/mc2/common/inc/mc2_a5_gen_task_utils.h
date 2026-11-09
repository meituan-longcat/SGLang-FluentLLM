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
 * \file mc2_a5_gen_task_utils.h
 * \brief
 */

#ifndef OPS_TRANSFORMER_DEV_MC2_COMMON_INC_MC2_A5_GEN_TASK_UTILS_H
#define OPS_TRANSFORMER_DEV_MC2_COMMON_INC_MC2_A5_GEN_TASK_UTILS_H

#ifndef BUILD_OPEN_PROJECT

#include <set>
#include "runtime/rt_model.h"
#include "proto/task.pb.h"
#include "exe_graph/runtime/exe_res_generation_context.h"
#include "graph/utils/args_format_desc_utils.h"
#include "platform/platform_info.h"
#include "platform/platform_ascendc.h"
#include "platform/soc_spec.h"

namespace ops {
const std::set<std::string> PLATFORM_A2 = {"Ascend910B"};
const std::set<std::string> PLATFORM_A3 = {"Ascend910_93"};
const std::set<std::string> NPUARCH_A5 = {std::to_string(static_cast<uint32_t>(NpuArch::DAV_3510))};

const std::string COMM_ALG_FULLMESH_V1 = "fullmesh_v1";
const std::string COMM_ALG_FULLMESH_V2 = "fullmesh_v2";
const std::string COMM_ALG_MTE = "mte";
const std::string COMM_ALG_CCU = "ccu";
class Mc2A5GenTaskUtils {
public:
  static void DeleteTaskIdxByType(const gert::ExeResGenerationContext *context, 
                                  const std::vector<domi::TaskDef> &tasks, rtModelTaskType_t type);
  static ge::Status CreateCcuFusionTask(const gert::ExeResGenerationContext *context, domi::TaskDef &ccu_fusion_task,
                                        rtModelTaskType_t type, bool is_attached_stream);
  static ge::Status InsertContextForCcuFusion(const gert::ExeResGenerationContext *context, domi::TaskDef &task_def,
                                              std::vector<ge::ArgDesc> args, bool isAllKernel);
  static ge::Status Mc2GenTaskCallBack910A5(const gert::ExeResGenerationContext *context,
                                            std::vector<domi::TaskDef> &tasks);
  static ge::Status GetArgsFormat(const gert::ExeResGenerationContext *context, domi::TaskDef &aicoreTask,
                                  std::vector<ge::ArgDesc> &argDescs);
  static bool IsTargetPlatformSocVersion(const char *nodeName, const std::set<std::string> &targetPlatform);
  static bool IsTargetPlatformNpuArch(const char *nodeName, const std::set<std::string> &targetPlatform);
  static const std::string GetCommAlg(const gert::ExeResGenerationContext *context, const size_t commAlgIdx);
};
}
#endif

#endif // OPS_TRANSFORMER_DEV_MC2_COMMON_INC_MC2_A5_GEN_TASK_UTILS_H