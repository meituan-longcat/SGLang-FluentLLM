/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file op_mc2_def.h
 * \brief
 */
#ifndef OP_MC2_DEF_H_
#define OP_MC2_DEF_H_

#include "acl/acl.h"
#include "runtime/kernel.h"
#include "op_mc2.h"
#include "hccl/hccl_types.h"
#include "opdev/common_types.h"

namespace op {
using s64 = signed long long;
using u32 = unsigned int;
using u64 = unsigned long long;
const int MAX_RANK_NUM = 8;
const int NUM_ONE = 1;
const int NUM_TWO = 2;
const int NUM_THREE = 3;
const int NUM_FOUR = 4;
const int NUM_FIVE = 5;
const int NUM_SIX = 6;
const int NUM_SEVEN = 7;
const int NUM_EIGHT = 8;
const int NUM_TEN = 10;
const int COMMUNICATION_DATA_SIZE = 10; // 10*1024以上two shot
static const char* REDUCE_OP_SUM = "sum";
static constexpr size_t BIAS_SUPPORTED_DIMENSIONAL = 2U;
static constexpr size_t SUPPORTED_DIMENSIONAL = 3U;
static constexpr int64_t EXPERT_LOWER_LIMIT = 2L;
static constexpr int64_t EXPERT_UPPER_LIMIT = 512L;
static constexpr int64_t MATMUL_E_OVER_EP_LIMIT = 32L;
static constexpr int64_t MATMUL_K_LIMIT = 65535L;
static constexpr size_t HCCL_GROUP_NAME_MAX = 128U;

constexpr uint64_t WORLD_SIZE_PAIR = 2U;
constexpr uint64_t WORLD_SIZE_QUAD = 4U;
constexpr uint64_t WORLD_SIZE_OCTET = 8U;
constexpr uint64_t WORLD_SIZE_SEXTET = 16U;
constexpr uint64_t WORLD_SIZE_THIRTYTWO = 32U;

enum HcclOperatorType {
  HCCL_OP_TYPE_INVALID = 0,
  HCCL_OP_TYPE_BROADCAST = 1,
  HCCL_OP_TYPE_ALLREDUCE,
  HCCL_OP_TYPE_REDUCE,
  HCCL_OP_TYPE_SEND,
  HCCL_OP_TYPE_RECEIVE,
  HCCL_OP_TYPE_ALLGATHER,
  HCCL_OP_TYPE_REDUCE_SCATTER,
  HCCL_OP_TYPE_ALLTOALLV,
  HCCL_OP_TYPE_ALLTOALLVC,
  HCCL_OP_TYPE_GATHER,
  HCCL_OP_TYPE_MAX
};

struct HcomParam {
  void* context;
  void* aicpuStream;
  void* aicpuNotify;
};

typedef struct tagAicpuRpcServerModeLaunchOpAPI {
  HcclOperatorType comm_type;
  HcclReduceOp op_type;
  uint32_t rank_id;
  uint32_t rank_num;
  uint64_t windows[MAX_RANK_NUM];
  uint64_t no_ipc_notify_ids[MAX_RANK_NUM * 2];
  uint64_t ipc_notify_ids[MAX_RANK_NUM * 4];
  uint64_t stream_ids[MAX_RANK_NUM];
  uint64_t sq_ids[MAX_RANK_NUM];
  uint64_t workSpaceAddr;
  char kernel_name[32];
  char so_name[32];
} AicpuRpcServerModeLaunchOpAPI;

typedef struct tagHcclCombinOpParam {
  u32 rank_id{0};
  u32 rank_num{0};
  u64 bufferSize{0};
  u64 windows[8] = {0};
  u64 no_ipc_notify_ids[16] = {0};
  u64 ipc_notify_ids[32] = {0};
  s64 stream_ids[8] = {0};
  u64 sq_ids[8] = {0};
} HcclCombinOpParam;

typedef struct tagHcclAndAicpuResource {
  HcclCombinOpParam hccl_param = {0};
  void* aicpu_win_addr = nullptr;
} HcclAndAicpuResource;

static void ConstructAicpuArgs(rtAicpuArgsEx_t& argsInfo, ops::ApiParamDef& apiParam) {
  rtHostInputInfo_t* host_info = (rtHostInputInfo_t*)apiParam.hostInputInfo;
  host_info->addrOffset = NUM_FIVE * sizeof(void*);
  host_info->dataOffset = NUM_SIX * sizeof(void*);

  argsInfo.args = (void*)&apiParam;
  argsInfo.hostInputInfoPtr = host_info;
  argsInfo.kernelOffsetInfoPtr = nullptr;
  argsInfo.argsSize = sizeof(apiParam);
  argsInfo.hostInputInfoNum = 1;
  argsInfo.kernelOffsetInfoNum = 0;
  argsInfo.soNameAddrOffset = (uint16_t)((char*)&apiParam.soName - (char*)&apiParam);
  argsInfo.kernelNameAddrOffset = (uint16_t)((char*)&apiParam.kernelName - (char*)&apiParam);
  argsInfo.isNoNeedH2DCopy = false;
  return;
}

static rtAicpuArgsEx_t GetArgsInfo(const AicpuRpcServerModeLaunchOpAPI &apiParam) {
  rtAicpuArgsEx_t argsInfo;
  argsInfo.args = (void *)&apiParam;
  argsInfo.hostInputInfoPtr = NULL;
  argsInfo.kernelOffsetInfoPtr = NULL;
  argsInfo.argsSize = sizeof(apiParam);
  argsInfo.hostInputInfoNum = 0;
  argsInfo.kernelOffsetInfoNum = 0;
  argsInfo.soNameAddrOffset = (uint16_t)((char *)&apiParam.so_name - (char *)&apiParam);
  argsInfo.kernelNameAddrOffset = (uint16_t)((char *)&apiParam.kernel_name - (char *)&apiParam);
  argsInfo.isNoNeedH2DCopy = false;
  return argsInfo;
}

// all_to_all allgather/reducescatter bmm算子X type支持fp16和bf16
static const std::initializer_list<op::DataType> MOE_X_DTYPE_SUPPORT_LIST = {
  op::DataType::DT_FLOAT16, op::DataType::DT_BF16
};

// all_to_all allgather/reducescatter bmm算子bias type支持fp16和fp32
static const std::initializer_list<op::DataType> MOE_BIAS_DTYPE_SUPPORT_LIST = {
  op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT
};

}  // namespace op
#endif  // OP_MC2_DEF_H_