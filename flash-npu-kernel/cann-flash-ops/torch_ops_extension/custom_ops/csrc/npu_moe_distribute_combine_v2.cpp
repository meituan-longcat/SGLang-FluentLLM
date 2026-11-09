/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

#include <string>
#include <torch/library.h>

#include "ops_common.h"

namespace custom {
namespace {

at::Tensor AllocateCombineOutput(const at::Tensor& expandX, const at::Tensor& expertIds, int64_t epWorldSize,
                                 int64_t epRankId, int64_t globalBs)
{
    TORCH_CHECK(expandX.dim() == 2 && expertIds.dim() == 2,
                "expand_x and expert_ids must be 2-D tensors");
    TORCH_CHECK(expandX.scalar_type() == at::kBFloat16 || expandX.scalar_type() == at::kHalf ||
                    expandX.scalar_type() == at::kInt,
                "expand_x dtype must be bfloat16, float16 or int32");
    TORCH_CHECK(expertIds.scalar_type() == at::kInt, "expert_ids dtype must be int32");
    TORCH_CHECK(epWorldSize > 0, "ep_world_size must be greater than 0");
    TORCH_CHECK(epRankId >= 0 && epRankId < epWorldSize, "ep_rank_id must be in [0, ep_world_size)");
    (void)globalBs;
    const at::ScalarType outputDtype =
        expandX.scalar_type() == at::kInt ? at::kBFloat16 : expandX.scalar_type();
    return at::empty({expertIds.size(0), expandX.size(1)}, expertIds.options().dtype(outputDtype));
}

}  // namespace

at::Tensor MoeDistributeCombineV2Npu(
    const at::Tensor& expandX, const at::Tensor& expertIds, const at::Tensor& assistInfoForCombine,
    const at::Tensor& epSendCounts, const at::Tensor& expertScales, c10::string_view groupEp,
    int64_t epWorldSize, int64_t epRankId, int64_t moeExpertNum,
    const c10::optional<at::Tensor>& tpSendCounts, const c10::optional<at::Tensor>& xActiveMask,
    const c10::optional<at::Tensor>& expandScales, const c10::optional<at::Tensor>& sharedExpertX,
    const c10::optional<at::Tensor>& elasticInfo, const c10::optional<at::Tensor>& oriX,
    const c10::optional<at::Tensor>& constExpertAlpha1, const c10::optional<at::Tensor>& constExpertAlpha2,
    const c10::optional<at::Tensor>& constExpertV, c10::string_view groupTp, int64_t tpWorldSize,
    int64_t tpRankId, int64_t expertShardType, int64_t sharedExpertNum, int64_t sharedExpertRankNum,
    int64_t globalBs, int64_t commQuantMode, c10::string_view commAlg, int64_t zeroExpertNum,
    int64_t copyExpertNum, int64_t constExpertNum)
{
    at::Tensor output = AllocateCombineOutput(expandX, expertIds, epWorldSize, epRankId, globalBs);
    const int64_t globalBsReal = globalBs == 0 ? expertIds.size(0) * epWorldSize : globalBs;
    const c10::optional<at::Tensor> reserved = c10::nullopt;
    constexpr int64_t outDtype = 0;
    constexpr int64_t groupListType = 0;
    std::string groupEpString(groupEp);
    std::string groupTpString(groupTp);
    std::string commAlgString(commAlg);
    char* groupEpPtr = groupEpString.data();
    char* groupTpPtr = groupTpString.data();
    char* commAlgPtr = commAlgString.data();

    EXEC_NPU_CMD_V1(aclnnMoeDistributeCombineV4, expandX, expertIds, assistInfoForCombine, epSendCounts,
                    expertScales, tpSendCounts, xActiveMask, reserved, reserved, reserved, expandScales,
                    sharedExpertX, elasticInfo, oriX, constExpertAlpha1, constExpertAlpha2, constExpertV, reserved,
                    groupEpPtr, epWorldSize, epRankId, moeExpertNum, groupTpPtr,
                    tpWorldSize, tpRankId, expertShardType, sharedExpertNum, sharedExpertRankNum, globalBsReal,
                    outDtype, commQuantMode, groupListType, commAlgPtr, zeroExpertNum, copyExpertNum,
                    constExpertNum, output);
    return output;
}

at::Tensor MoeDistributeCombineV2Meta(
    const at::Tensor& expandX, const at::Tensor& expertIds, const at::Tensor&, const at::Tensor&,
    const at::Tensor&, c10::string_view, int64_t epWorldSize, int64_t epRankId, int64_t,
    const c10::optional<at::Tensor>&, const c10::optional<at::Tensor>&,
    const c10::optional<at::Tensor>&, const c10::optional<at::Tensor>&,
    const c10::optional<at::Tensor>&, const c10::optional<at::Tensor>&,
    const c10::optional<at::Tensor>&, const c10::optional<at::Tensor>&,
    const c10::optional<at::Tensor>&, c10::string_view, int64_t, int64_t, int64_t, int64_t, int64_t,
    int64_t globalBs, int64_t, c10::string_view, int64_t, int64_t, int64_t)
{
    return AllocateCombineOutput(expandX, expertIds, epWorldSize, epRankId, globalBs);
}

}  // namespace custom

TORCH_LIBRARY_IMPL(custom, PrivateUse1, m) {
    m.impl("npu_moe_distribute_combine_v2", &custom::MoeDistributeCombineV2Npu);
}

TORCH_LIBRARY_IMPL(custom, Meta, m) {
    m.impl("npu_moe_distribute_combine_v2", &custom::MoeDistributeCombineV2Meta);
}
