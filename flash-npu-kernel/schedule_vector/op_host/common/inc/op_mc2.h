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

 #ifndef OPS_COMMON_INC_OP_MC2_H
 #define OPS_COMMON_INC_OP_MC2_H
 
 #include <stddef.h>
 #include <stdint.h>
 #include <vector>
 
 namespace ops {
 struct ApiParamDef {
   uint64_t x1;
   uint64_t y;
   uint64_t gatherOut;
   uint64_t context;
   uint64_t workspace;
   uint64_t tilingDataPtr;
   uint8_t tilingData[2048];
   const char soName[32] = {"libccl_kernel.so"};
   const char kernelName[32] = {"RunAicpuRpcSrvLaunch"};
   const char opName[32] = {"HcclAicpuOp"};
   char hostInputInfo[16];
 };
 
 enum class MC2Type : uint32_t {
   K_MM_ALL_REDUCE,
   K_ALL_GATHER_MM,
   K_MM_REDUCE_SCATTER
 };
 
 enum class MC2InputIdx : size_t {
   K_X1,
   K_X2,
   K_BIAS,
   K_X3,
   K_SCALE,
   K_OFFSET,
   K_DEQUANT,
   K_PERTOKEN,
   K_COMMQUANTSCALE1,
   K_COMMQUANTSCALE2
 };
 
 enum class MC2OutputIdx : size_t {
   K_Y,
   K_GATHER_OUT
 };
 
 enum class MC2AddRmsNormInputIdx : size_t {
     K_X1,
     K_X2,
     K_BIAS,
     K_RESIDUAL,
     K_GAMMA,
     K_SCALE,
     K_OFFSET,
     K_DEQUANT,
     K_MAX
 };
 
 enum class MC2AddRmsNormOutputIdx : size_t {
     K_Y,
     K_NORM_OUT
 };
 
 enum class AllGatherMMAttrIdx : size_t {
   K_GROUP,
   K_TRANS_X1,
   K_TRANS_X2,
   K_GATHER_IDX,
   K_COMM_TURN,
   K_RANK_SIZE,
   K_IS_GATHER_OUT
 };
 
 enum class MmReduceScatterAttrIdx : size_t {
   K_GROUP,
   K_OP,
   K_TRANS_X1,
   K_TRANS_X2,
   K_COMM_TURN,
   K_RANK_SIZE
 };
 
 enum class MmAllReduceAttrIdx : size_t {
   K_GROUP,
   K_OP,
   K_TRANS_X1,
   K_TRANS_X2,
   K_COMM_TURN,
   K_ANTIQUANT_GROUP_SIZE,
   K_EPSILON
 };
 
 enum class AlltoAllAllGatherBatchMatMulActType : int64_t {
   ALLTOALL_ALLGATHER_BATCHMATMUL_ACT_TYPE_NONE = 0,
   ALLTOALL_ALLGATHER_BATCHMATMUL_ACT_TYPE_GELU = 1,
   ALLTOALL_ALLGATHER_BATCHMATMUL_ACT_TYPE_SILU = 2,
   ALLTOALL_ALLGATHER_BATCHMATMUL_ACT_TYPE_RELU = 3,
   ALLTOALL_ALLGATHER_BATCHMATMUL_ACT_TYPE_FASTGELU = 4,
   ALLTOALL_ALLGATHER_BATCHMATMUL_ACT_TYPE_GEGLU = 5,
   ALLTOALL_ALLGATHER_BATCHMATMUL_ACT_TYPE_SWIGLU = 6,
   ALLTOALL_ALLGATHER_BATCHMATMUL_ACT_TYPE_REGLU = 7
 };
 
 const std::vector<int64_t> ACT_TYPE_SUPPORT_VEC = {
   static_cast<int64_t>(AlltoAllAllGatherBatchMatMulActType::ALLTOALL_ALLGATHER_BATCHMATMUL_ACT_TYPE_NONE),
   static_cast<int64_t>(AlltoAllAllGatherBatchMatMulActType::ALLTOALL_ALLGATHER_BATCHMATMUL_ACT_TYPE_GELU),
   static_cast<int64_t>(AlltoAllAllGatherBatchMatMulActType::ALLTOALL_ALLGATHER_BATCHMATMUL_ACT_TYPE_SILU),
   static_cast<int64_t>(AlltoAllAllGatherBatchMatMulActType::ALLTOALL_ALLGATHER_BATCHMATMUL_ACT_TYPE_RELU),
   static_cast<int64_t>(AlltoAllAllGatherBatchMatMulActType::ALLTOALL_ALLGATHER_BATCHMATMUL_ACT_TYPE_FASTGELU)
 };
 
 enum class MC2MoeInputIdx : size_t {
   K_X,
   K_WEIGHT,
   K_BIAS,
 };
 
 enum class AlltoAllAllGatherBmmAttrIdx : size_t {
   K_GROUP_EP,
   K_GROUP_TP,
   K_EP_WORLD_SIZE,
   K_TP_WORLD_SIZE,
   K_X_SHARD_TYPE,
   K_ACT_TYPE,
   K_IS_TRANS_W,
   K_OUTPUT_Y2_FLAG,
   K_OUTPUT_Y3_FLAG
 };
 
 enum class BmmReduceScatterAlltoAllAttrIdx : size_t {
   K_GROUP_EP,
   K_GROUP_TP,
   K_EP_WORLD_SIZE,
   K_TP_WORLD_SIZE,
   K_Y_SHARD_TYPE,
   K_IS_TRANS_W
 };
 
 enum class AlltoAllAllGatherBmmOutIdx : size_t {
   K_Y1,
   K_Y2,
   K_Y3
 };
 
 enum class BmmReduceScatterAlltoAllOutIdx : size_t {
   K_Y
 };
 
 enum class AlltoAllvGroupedMatMulInputIdx : size_t {
   K_GMM_X,
   K_GMM_WEIGHT,
   K_SEND_COUNTS_TENSOR,
   K_RECV_COUNTS_TENSOR,
   K_MM_X,
   K_MM_WEIGHT
 };
 
 enum class AlltoAllvGroupedMatMulOutputIdx : size_t {
   K_GMM_Y,
   K_MM_Y,
   K_PERMUTE_OUT
 };
 
 enum class AlltoAllvGroupedMatMulAttrIdx : size_t {
   K_GROUP,
   K_EP_WORLD_SIZE,
   K_SEND_COUNTS,
   K_RECV_COUNTS,
   K_TRANS_GMM_WEIGHT,
   K_TRANS_MM_WEIGHT,
   K_PERMUTE_OUT_FLAG
 };
 
 enum class GroupedMatMulAlltoAllvInputIdx : size_t {
   K_GMM_X,
   K_GMM_WEIGHT,
   K_SEND_COUNTS_TENSOR,
   K_RECV_COUNTS_TENSOR,
   K_MM_X,
   K_MM_WEIGHT
 };
 
 enum class GroupedMatMulAlltoAllvOutputIdx : size_t {
   K_Y,
   K_MM_Y
 };
 
 enum class GroupedMatMulAlltoAllvAttrIdx : size_t {
   K_GROUP,
   K_EP_WORLD_SIZE,
   K_SEND_COUNTS,
   K_RECV_COUNTS,
   K_TRANS_GMM_WEIGHT,
   K_TRANS_MM_WEIGHT
 };
 }  // namespace ops
 
 #endif  // OPS_COMMON_INC_OP_MC2_H
 