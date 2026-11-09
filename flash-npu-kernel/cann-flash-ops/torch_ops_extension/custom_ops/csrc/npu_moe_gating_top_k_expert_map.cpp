/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

#include <tuple>
#include <torch/library.h>

#include "ops_common.h"

namespace custom {
namespace {

using OutputTuple = std::tuple<at::Tensor, at::Tensor, at::Tensor>;

void CheckInputs(const at::Tensor &x, const at::Tensor &expertMap,
                 const c10::optional<at::Tensor> &bias, int64_t k,
                 int64_t kGroup, int64_t groupCount, bool checkSizes)
{
    TORCH_CHECK(x.dim() == 2, "x must be a 2-D tensor.");
    TORCH_CHECK(x.scalar_type() == at::kFloat,
                "x dtype must be float32 for the online router path.");
    TORCH_CHECK(expertMap.dim() == 1, "expert_map must be a 1-D tensor.");
    TORCH_CHECK(expertMap.scalar_type() == at::kInt,
                "expert_map dtype must be int32.");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous.");
    TORCH_CHECK(expertMap.is_contiguous(), "expert_map must be contiguous.");
    TORCH_CHECK(x.device() == expertMap.device(),
                "x and expert_map must be on the same device.");
    TORCH_CHECK(k > 0, "k must be greater than 0.");
    TORCH_CHECK(groupCount == 1 && kGroup == 1,
                "this fused operator only supports group_count=1 and k_group=1.");
    if (checkSizes) {
        TORCH_CHECK(x.size(1) > 0, "x.size(1) must be greater than 0.");
        TORCH_CHECK(expertMap.numel() > 0, "expert_map must be non-empty.");
        TORCH_CHECK(k <= x.size(1), "k must not exceed x.size(1).");
    }
    if (bias.has_value()) {
        const at::Tensor &biasTensor = bias.value();
        TORCH_CHECK(biasTensor.dim() == 1, "bias must be a 1-D tensor.");
        TORCH_CHECK(biasTensor.scalar_type() == x.scalar_type(),
                    "bias dtype must match x dtype.");
        TORCH_CHECK(biasTensor.is_contiguous(), "bias must be contiguous.");
        TORCH_CHECK(biasTensor.device() == x.device(),
                    "bias and x must be on the same device.");
        if (checkSizes) {
            TORCH_CHECK(biasTensor.size(0) == x.size(1),
                        "bias size must equal x.size(1).");
        }
    }
}

OutputTuple MakeOutputs(const at::Tensor &x, int64_t k)
{
    auto y = at::empty({x.size(0), k}, x.options());
    auto expertIdx = at::empty({x.size(0), k}, x.options().dtype(at::kInt));
    auto out = at::empty(x.sizes(), x.options().dtype(at::kFloat));
    return std::make_tuple(y, expertIdx, out);
}

} // namespace

OutputTuple NpuMoeGatingTopKExpertMap(
    const at::Tensor &x, const at::Tensor &expertMap,
    const c10::optional<at::Tensor> &bias, int64_t k, int64_t kGroup,
    int64_t groupCount, int64_t groupSelectMode, int64_t renorm,
    int64_t normType, bool outFlag, double routedScalingFactor, double eps)
{
    CheckInputs(x, expertMap, bias, k, kGroup, groupCount, true);
    auto [y, expertIdx, out] = MakeOutputs(x, k);
    if (x.size(0) == 0) {
        return std::make_tuple(y, expertIdx, out);
    }
    EXEC_NPU_CMD_V1(aclnnMoeGatingTopKExpertMap, x, bias, expertMap, k,
                    kGroup, groupCount, groupSelectMode, renorm, normType,
                    outFlag, routedScalingFactor, eps, y, expertIdx, out);
    return std::make_tuple(y, expertIdx, out);
}

OutputTuple NpuMoeGatingTopKExpertMapMeta(
    const at::Tensor &x, const at::Tensor &expertMap,
    const c10::optional<at::Tensor> &bias, int64_t k, int64_t kGroup,
    int64_t groupCount, int64_t, int64_t, int64_t, bool, double, double)
{
    CheckInputs(x, expertMap, bias, k, kGroup, groupCount, false);
    return MakeOutputs(x, k);
}

} // namespace custom

TORCH_LIBRARY_IMPL(custom, PrivateUse1, m) {
    m.impl("npu_moe_gating_top_k_expert_map", &custom::NpuMoeGatingTopKExpertMap);
}

TORCH_LIBRARY_IMPL(custom, Meta, m) {
    m.impl("npu_moe_gating_top_k_expert_map", &custom::NpuMoeGatingTopKExpertMapMeta);
}
