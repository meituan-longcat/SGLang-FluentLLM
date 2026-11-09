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
 * \file mc2_a5_gen_task_utils.cpp
 * \brief
 */

#ifndef BUILD_OPEN_PROJECT
#include "mc2_a5_gen_task_utils.h"
#include "mc2_gen_task_utils.h"
#include "runtime/rt_model.h"
#include "matmul_allto_all_util.h"
#include "error/ops_error.h"
#include "error_util.h"
#include "proto/task.pb.h"
#include "framework/common/taskdown_common.h"
#include "exe_graph/runtime/exe_res_generation_context.h"
#include "graph/utils/args_format_desc_utils.h"
#include "register/hidden_inputs_func_registry.h"

namespace ops {

constexpr int64_t INVALID_INT_VAL = -1;
// 对已有结构的重复定义，只在本文件插入 aicpu desc 的时候使用
struct HcclCommParamDescTemp {
  uint64_t version : 4;
  uint64_t groupNum : 4;
  uint64_t hasFfts : 1;
  uint64_t tilingOff : 7;
  uint64_t isDyn : 48;
};

int64_t GetAttachStreamIdByContext(const gert::ExeResGenerationContext *context, size_t idx = 0) {
#ifndef ASCEND_OPSPROTO_UT
  const auto stream_infos = context->GetAttachedStreamInfos();
  if (idx >= stream_infos.size()) {
    OPS_LOG_E(context->GetNodeName(), "Invalid index %zu in streams count %zu.", idx, stream_infos.size());
    return INVALID_INT_VAL;
  }

  const int64_t stream_id = (stream_infos[0].is_valid ? stream_infos[0].stream_id : INVALID_INT_VAL);
#else
  const int64_t stream_id = 1;
#endif
  return stream_id;
}

const std::string MOE_DISTRIBUTE_DISPATCH_OP_TYPE = "MoeDistributeDispatch";
const std::string MOE_DISTRIBUTE_COMBINE_OP_TYPE = "MoeDistributeCombine";
const std::string MOE_DISTRIBUTE_DISPATCH_V2_OP_TYPE = "MoeDistributeDispatchV2";
const std::string MOE_DISTRIBUTE_COMBINE_V2_OP_TYPE = "MoeDistributeCombineV2";
const std::string ALL_TO_ALLV_GROUPED_MM_OP_TYPE = "AlltoAllvGroupedMatMul";
const std::string GROUPED_MM_ALL_TO_ALLV_OP_TYPE = "GroupedMatMulAlltoAllv";
const std::string ALL_GATHER_MM_V2_OP_TYPE = "AllGatherMatmulV2";
const std::string MM_REDUCE_SCATTER_V2_OP_TYPE = "MatmulReduceScatterV2";
const std::string MM_ALL_REDUCE_OP_TYPE = "MatmulAllReduce";
const std::string MM_ALLTO_ALL_OP_TYPE = "MatmulAlltoAll";
const std::string ALLTO_ALL_MM_OP_TYPE = "AlltoAllMatmul";
const std::string ATTR_NAME_GROUP = "group";
const std::string ATTR_NAME_GROUP_EP = "group_ep";
const int32_t MAX_GROUP_CNT = 16;

struct GroupInfo {
  int32_t groupCnt; // 算子通信域数量
  std::vector<std::string> groupAttrNames; // 算子通信域属性名
};
// 当前 GetCCuTaskInfo 接口暂不支持多通信域的接口，对于双通信域的算子，暂时不增加 TP 属性名
static const std::map<const std::string, const GroupInfo> GROUP_INFO_MAP_A5 {
  {ALL_GATHER_MM_V2_OP_TYPE,           {1, {ATTR_NAME_GROUP}}},
  {MM_REDUCE_SCATTER_V2_OP_TYPE,       {1, {ATTR_NAME_GROUP}}},
  {MM_ALL_REDUCE_OP_TYPE,              {1, {ATTR_NAME_GROUP}}},
  {MM_ALLTO_ALL_OP_TYPE,               {1, {ATTR_NAME_GROUP}}},
  {ALLTO_ALL_MM_OP_TYPE,               {1, {ATTR_NAME_GROUP}}},
  {ALL_TO_ALLV_GROUPED_MM_OP_TYPE,     {1, {ATTR_NAME_GROUP}}},
  {GROUPED_MM_ALL_TO_ALLV_OP_TYPE,     {1, {ATTR_NAME_GROUP}}},
  {MOE_DISTRIBUTE_DISPATCH_OP_TYPE,    {2, {ATTR_NAME_GROUP_EP}}},
  {MOE_DISTRIBUTE_COMBINE_OP_TYPE,     {2, {ATTR_NAME_GROUP_EP}}},
  {MOE_DISTRIBUTE_DISPATCH_V2_OP_TYPE, {2, {ATTR_NAME_GROUP_EP}}},
  {MOE_DISTRIBUTE_COMBINE_V2_OP_TYPE,  {2, {ATTR_NAME_GROUP_EP}}},
};

static bool GetGroupInfo(const gert::ExeResGenerationContext *context, GroupInfo &groupInfo) {
  const char *opType = context->GetNodeType();
  if (opType == nullptr) {
    OPS_LOG_E(context->GetNodeName(), "Op type is nullptr.");
    return false;
  }
  const std::string opTypeStr = opType;
  if (GROUP_INFO_MAP_A5.find(opTypeStr) == GROUP_INFO_MAP_A5.end()) {
    OPS_LOG_E(context->GetNodeName(), "Op type [%s] has not registe in group cnt map.", opType);
    return false;
  }
  groupInfo = GROUP_INFO_MAP_A5.at(opTypeStr);
  const int32_t cnt = groupInfo.groupCnt;
  if ((cnt <= 0) || (cnt > MAX_GROUP_CNT)) {
    OPS_LOG_E(context->GetNodeName(), "Group cnt [%d] is invalid, it should in [1, %d].", cnt, MAX_GROUP_CNT);
    return false;
  }

  OPS_LOG_D(context->GetNodeName(), "Op [%s] get group [%d] success.", opType, cnt);
  return true;
}

ge::Status Mc2A5GenTaskUtils::InsertContextForCcuFusion(const gert::ExeResGenerationContext *context,
                                                        domi::TaskDef &task_def, std::vector<ge::ArgDesc> args,
                                                        bool isAllKernel)
{
  auto fusion_task = task_def.mutable_fusion_task();
  GE_ASSERT_NOTNULL(fusion_task);
  uint32_t subTaskInfoIdx = 0;
  // fusion_task中添加两个task: ccu task和 aicore_task
  fusion_task->add_fusion_sub_task_info();
  fusion_task->add_fusion_sub_task_info();
  // sub task 排列顺序ccu + aicore
  OPS_LOG_I(context->GetNodeName(), "subTaskInfoIdx is %u before before", subTaskInfoIdx);
  auto ccu_task_def = fusion_task->mutable_fusion_sub_task_info(subTaskInfoIdx++);
  OPS_LOG_I(context->GetNodeName(), "subTaskInfoIdx is %u before after", subTaskInfoIdx);
  GE_ASSERT_NOTNULL(ccu_task_def);
  // ccu task 填充type， ccu task 其他信息由ge填充
  ccu_task_def->set_type(domi::FusionSubTaskInfo_FusionType::FusionSubTaskInfo_FusionType_CCU);
  auto ccu_task = ccu_task_def->mutable_task();
  GE_ASSERT_NOTNULL(ccu_task);
  auto ccu_task_group = ccu_task->mutable_ccu_task_group();
  GE_ASSERT_NOTNULL(ccu_task_group);
  ccu_task_group->add_ccu_task_info();

  OPS_LOG_I(context->GetNodeName(), "subTaskInfoIdx is %u before", subTaskInfoIdx);
  auto aicore_task_def = fusion_task->mutable_fusion_sub_task_info(subTaskInfoIdx++);
  OPS_LOG_I(context->GetNodeName(), "subTaskInfoIdx is %u after", subTaskInfoIdx);
  GE_ASSERT_NOTNULL(aicore_task_def);

  // task 填充信息都填充到aicore中
  aicore_task_def->set_type(domi::FusionSubTaskInfo_FusionType::FusionSubTaskInfo_FusionType_AICORE);
  auto aicore_task = aicore_task_def->mutable_task();
  GE_ASSERT_NOTNULL(aicore_task);
  auto aicore_fusion_task_info = aicore_task->mutable_aicore_fusion_task_info();
  GE_ASSERT_NOTNULL(aicore_fusion_task_info);
  // is_all_kernel为true表示优先进行二进制复用，如果没有匹配到则进行在线编译
  aicore_fusion_task_info->set_is_all_kernel(isAllKernel);
  OPS_LOG_I(context->GetNodeName(), "set is all kernel to %u.", isAllKernel);
  // 设置attribute中的numBlocks
  auto config = aicore_fusion_task_info->mutable_config();
  GE_ASSERT_NOTNULL(config);
  config->add_launch_attribute();
  auto launch_attribute = config->mutable_launch_attribute(0);
  GE_ASSERT_NOTNULL(launch_attribute);
  launch_attribute->set_id(domi::LaunchAttribute_LaunchAttributeId::LaunchAttribute_LaunchAttributeId_BLOCKDIM);
  auto value = launch_attribute->mutable_value();
  GE_ASSERT_NOTNULL(value);

  int64_t numBlocks = 1;
  if (!context->GetIntAttrVal("tvm_blockdim", numBlocks) || numBlocks <= 0) {
    OPS_LOG_I(context->GetNodeName(), "Can't get valid numBlocks, get numBlocks %ld, set numBlocks 1.", numBlocks);
    numBlocks = 1;
  }
  OPS_LOG_I(context->GetNodeName(), "get numBlocks %ld", numBlocks);
  value->set_block_dim(numBlocks);

  auto aicore_context = aicore_fusion_task_info->mutable_context();
  GE_ASSERT_NOTNULL(aicore_context);
  aicore_context->set_kernel_type(static_cast<uint32_t>(ge::ccKernelType::MIX_AICORE));

  OPS_LOG_I(context->GetNodeName(), "fusion task op index %zu", context->GetOpId());
  fusion_task->set_op_index(context->GetOpId());

  GroupInfo groupInfo;
  if (!GetGroupInfo(context, groupInfo)) {
    return ge::GRAPH_FAILED;
  }
  for (const std::string &groupAttrName : groupInfo.groupAttrNames) {
    ccu_task_group->add_group(groupAttrName);
    OPS_LOG_D(context->GetNodeName(), "Set group attr [%s] to ccu task info success.", groupAttrName.c_str());
  }
  ge::ArgsFormatDescUtils::InsertHiddenInputs(args, 0, ge::HiddenInputsType::HCOM, groupInfo.groupCnt);

  // desc 配置
  union {
    HcclCommParamDescTemp hcclCommParaDesc;
    uint64_t customValue;
  } desc;
  desc.hcclCommParaDesc.version = 1;
  desc.hcclCommParaDesc.groupNum = groupInfo.groupCnt;
  desc.hcclCommParaDesc.hasFfts = 0; // david不使用ffts
  desc.hcclCommParaDesc.tilingOff = args.size() - 1;
  desc.hcclCommParaDesc.isDyn = 0;
  // args 参数顺序: {hcom}{INPUT0}...{INPUTN}{OUTPUT0}...{OUTPUTN}{WORKSPACE}{TILING}{desc}
  OPS_LOG_I(context->GetNodeName(), "tilingOff is : %d", desc.hcclCommParaDesc.tilingOff);

  ge::ArgsFormatDescUtils::InsertCustomValue(args, -1, desc.customValue);
  fusion_task->set_args_format(ge::ArgsFormatDescUtils::ToString(args));
  int32_t args_num = args.size() - 1;
  OPS_LOG_I(context->GetNodeName(), "args num is : %d", args_num);
  fusion_task->set_kfc_args_format_offset(args_num);
  OPS_LOG_I(context->GetNodeName(), "args_format is : %s", ge::ArgsFormatDescUtils::ToString(args).c_str());
  return ge::GRAPH_SUCCESS;
}

ge::Status Mc2A5GenTaskUtils::CreateCcuFusionTask(const gert::ExeResGenerationContext *context,
                                                  domi::TaskDef &ccu_fusion_task, rtModelTaskType_t type,
                                                  bool is_attached_stream)
{
  GE_ASSERT_NOTNULL(context);
  int64_t stream_id;
  if (is_attached_stream) {
    stream_id = GetAttachStreamIdByContext(context);
  } else {
    stream_id = context->GetStreamId();
  }
  GE_ASSERT_TRUE(stream_id >= 0);
  ccu_fusion_task.set_id(context->GetOpId());
  ccu_fusion_task.set_notify_id(UINT32_MAX);
  ccu_fusion_task.set_type(type);
  ccu_fusion_task.set_stream_id(stream_id);
  // 算子临时规避整网执行方案：aic（1个） + ccu（4个） 一共占据 5 个 sqe
  ccu_fusion_task.set_sqe_num(5);
  OPS_LOG_I(context->GetNodeName(), "Create fusion task(type %u) for mc2 node successfully, %s stream id %ld.",
            static_cast<uint32_t>(type), (is_attached_stream ? "attached" : "main"), stream_id);
  return ge::GRAPH_SUCCESS;
}

ge::Status Mc2A5GenTaskUtils::Mc2GenTaskCallBack910A5(const gert::ExeResGenerationContext *context,
                                                      std::vector<domi::TaskDef> &tasks)
{
  // 获取aicore task 并删除
  int64_t aicore_idx = Mc2GenTaskUtils::GetTaskIdxByType(context, tasks, RT_MODEL_TASK_ALL_KERNEL);
  bool isAllKernel = true;
  if (aicore_idx < 0) {
    OPS_LOG_D(context->GetNodeName(), "start Mc2MoeDistributeGenTaskCallBackA5 RT_MODEL_TASK_KERNEL");
    aicore_idx = Mc2GenTaskUtils::GetTaskIdxByType(context, tasks, RT_MODEL_TASK_KERNEL);
    isAllKernel = false;
    OPS_LOG_D(context->GetNodeName(), "Set is all kernel to false.");
  }
  OP_CHECK(aicore_idx < 0, OPS_LOG_E(context->GetNodeName(), "Failed to get AICore task."), return ge::GRAPH_FAILED);
  OPS_LOG_I(context->GetNodeName(), "Start to generate task for MC2, task def size %lu, aicore index %ld.",
            tasks.size(), aicore_idx);
  // 获取 aicore 的 args_format
  std::vector<ge::ArgDesc> argDescs;
  if (Mc2A5GenTaskUtils::GetArgsFormat(context, tasks[static_cast<size_t>(aicore_idx)], argDescs) != ge::GRAPH_SUCCESS) {
    return ge::GRAPH_FAILED;
  }
  OPS_LOG_I(context->GetNodeName(), "before args_format is : %s", ge::ArgsFormatDescUtils::ToString(argDescs).c_str());

  auto iter = tasks.erase(tasks.begin() + aicore_idx);
  // 创建 fusion task
  domi::TaskDef fusion_task{};
  GE_ASSERT_SUCCESS(CreateCcuFusionTask(context, fusion_task, RT_MODEL_TASK_FUSION_KERNEL, false));
  tasks.insert(iter, fusion_task);
  OPS_LOG_D(context->GetNodeName(), "after CreateCcuFusionTask.");
  return InsertContextForCcuFusion(context, tasks[static_cast<size_t>(aicore_idx)], argDescs, isAllKernel);
}

ge::Status Mc2A5GenTaskUtils::GetArgsFormat(const gert::ExeResGenerationContext *context, domi::TaskDef &aicoreTask,
  std::vector<ge::ArgDesc> &argDescs)
{
  domi::KernelContext *kernel_context;
  if (aicoreTask.type() == RT_MODEL_TASK_KERNEL) {
    auto kernel_def = aicoreTask.mutable_kernel();
    GE_ASSERT_NOTNULL(kernel_def);
    kernel_context = kernel_def->mutable_context();
  } else if (aicoreTask.type() == RT_MODEL_TASK_ALL_KERNEL) {
    auto kernel_with_handle = aicoreTask.mutable_kernel_with_handle();
    GE_ASSERT_NOTNULL(kernel_with_handle);
    kernel_context = kernel_with_handle->mutable_context();
  } else {
    OPS_LOG_E(context->GetNodeName(), "Invalid task type [%u].", aicoreTask.type());
    return ge::GRAPH_FAILED;
  }
  const std::string argsFormat = kernel_context->args_format();
  OPS_ERR_IF(ge::ArgsFormatDescUtils::Parse(argsFormat, argDescs) != ge::GRAPH_SUCCESS || argDescs.empty(),
    OPS_LOG_E(context->GetNodeName(), "Failed to parse, argsFormat:[%s]", argsFormat.c_str()), return ge::GRAPH_FAILED);
  OPS_LOG_D(context->GetNodeName(), "end GetArgsFormat %s", argsFormat.c_str());
  return ge::GRAPH_SUCCESS;
}

bool Mc2A5GenTaskUtils::IsTargetPlatformSocVersion(const char *nodeName, const std::set<std::string> &targetPlatform)
{
    fe::PlatFormInfos platform_info;
    fe::OptionalInfos optional_info;
    if (fe::PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platform_info, optional_info) !=
        ge::GRAPH_SUCCESS) {
        OPS_LOG_E(nodeName, "Cannot get platform info in gentask!");
        return false;
    }
    std::string short_soc_version;
    if (!platform_info.GetPlatformRes("version", "Short_SoC_version", short_soc_version) || short_soc_version.empty()) {
        OPS_LOG_E(nodeName, "Cannot get short soc version in Mc2A5GenTaskUtils::IsTargetPlatformSocVersion!");
        return false;
    }
    OPS_LOG_D(nodeName, "Get soc version: %s", short_soc_version.c_str());
    return targetPlatform.count(short_soc_version) > 0;
}

bool Mc2A5GenTaskUtils::IsTargetPlatformNpuArch(const char *nodeName, const std::set<std::string> &targetPlatform)
{
    fe::PlatFormInfos platform_info;
    fe::OptionalInfos optional_info;
    if (fe::PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platform_info, optional_info) !=
        ge::GRAPH_SUCCESS) {
        OPS_LOG_E(nodeName, "Cannot get platform info in Mc2A5GenTaskUtils::IsTargetPlatformNpuArch!");
        return false;
    }
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(&platform_info);
    std::string socNpuArch = std::to_string(static_cast<uint32_t>(ascendcPlatform.GetCurNpuArch()));
    OPS_LOG_D(nodeName, "Current GenTask Platform (NpuArch) %s", socNpuArch.c_str());
    return targetPlatform.count(socNpuArch) > 0;
}

const std::string Mc2A5GenTaskUtils::GetCommAlg(const gert::ExeResGenerationContext *context, const size_t commAlgIdx)
{
    auto *attrs = context->GetAttrs();
    if (attrs == nullptr) {
        OPS_LOG_E(context->GetNodeName(), "Attrs pointer is null.");
        return "";
    }
    const char *commAlgPtr = attrs->GetStr(commAlgIdx);
    if (commAlgPtr == nullptr) {
        OPS_LOG_E(context->GetNodeName(), "Comm alg pointer is null.");
        return "";
    }
    const std::string commAlg = commAlgPtr;
    if (commAlg.empty()) {
        OPS_LOG_W(context->GetNodeName(), "Comm alg is empty, will use mte alg.");
        return "mte";
    }
    OPS_LOG_D(context->GetNodeName(), "Comm alg is %s.", commAlg.c_str());
    return commAlg;
}

} // namespace ops

#endif