/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <torch/library.h>

#include "ops_common.h"

namespace custom {
namespace {

void CheckCausalConv1dInputs(const at::Tensor& x, const at::Tensor& weight, const at::Tensor& convState,
    const c10::optional<at::Tensor>& bias, int64_t activationMode, int64_t runMode)
{
    TORCH_CHECK(x.dim() == 2 || x.dim() == 3, "npu_causal_conv1d: x must be rank 2 or 3.");
    TORCH_CHECK(weight.dim() == 2, "npu_causal_conv1d: weight must be [width, channels].");
    TORCH_CHECK(convState.dim() == 3,
        "npu_causal_conv1d: conv_state must be [slots, state_len, channels].");
    TORCH_CHECK(weight.size(0) >= 2 && weight.size(0) <= 4,
        "npu_causal_conv1d: width must be in [2, 4].");
    TORCH_CHECK(weight.size(1) == x.size(-1) && convState.size(2) == x.size(-1),
        "npu_causal_conv1d: channel dimensions must match.");
    TORCH_CHECK(convState.size(0) > 0 && convState.size(1) >= weight.size(0) - 1,
        "npu_causal_conv1d: conv_state must have at least width - 1 history rows.");
    TORCH_CHECK(convState.stride(2) == 1 && convState.stride(1) == convState.size(2) &&
        convState.stride(0) >= convState.size(1) * convState.size(2),
        "npu_causal_conv1d: conv_state requires strides [S, channels, 1], S >= state_len * channels.");
    TORCH_CHECK(x.size(-1) % 16 == 0,
        "npu_causal_conv1d: channels must be a multiple of 16.");
    TORCH_CHECK(x.scalar_type() == at::kHalf || x.scalar_type() == at::kBFloat16,
        "npu_causal_conv1d: x dtype must be float16 or bfloat16.");
    TORCH_CHECK(weight.scalar_type() == x.scalar_type() && convState.scalar_type() == x.scalar_type(),
        "npu_causal_conv1d: x, weight, and conv_state dtypes must match.");
    if (bias.has_value() && bias->defined()) {
        TORCH_CHECK(bias->dim() == 1 && bias->size(0) == x.size(-1),
            "npu_causal_conv1d: bias must be [channels].");
        TORCH_CHECK(bias->scalar_type() == x.scalar_type(),
            "npu_causal_conv1d: bias dtype must match x.");
    }
    TORCH_CHECK(activationMode == 0 || activationMode == 1,
        "npu_causal_conv1d: activation_mode must be 0 or 1.");
    TORCH_CHECK(runMode == 0 || runMode == 1,
        "npu_causal_conv1d: run_mode must be 0 or 1.");
}

at::Tensor RunCausalConv1d(const at::Tensor& x, const at::Tensor& weight, at::Tensor& convState,
    const c10::optional<at::Tensor>& bias, const c10::optional<at::Tensor>& queryStartLoc,
    const c10::optional<at::Tensor>& cacheIndices, const c10::optional<at::Tensor>& initialStateMode,
    const c10::optional<at::Tensor>& numAcceptedTokens, const c10::optional<at::Tensor>& writeIndices,
    int64_t activationMode, int64_t padSlotId, int64_t runMode)
{
    at::Tensor output = at::zeros_like(x);
    int64_t stateSlotStride = convState.stride(0);
    EXEC_NPU_CMD_V1(aclnnCausalConv1d, x, weight, bias, convState, queryStartLoc, cacheIndices, initialStateMode,
        numAcceptedTokens, writeIndices, activationMode, padSlotId, runMode, stateSlotStride, output);
    return output;
}

void CheckWriteIndices(const at::Tensor& x, const at::Tensor& weight, const at::Tensor& convState,
    const c10::optional<at::Tensor>& cacheIndices, const c10::optional<at::Tensor>& writeIndices,
    const c10::optional<at::Tensor>& initialStateMode, int64_t runMode)
{
    TORCH_CHECK(runMode == 0 || !initialStateMode.has_value() || !initialStateMode->defined(),
        "npu_causal_conv1d: initial_state_mode is only supported in Prefill; use write_indices for Decode.");
    if (!writeIndices.has_value() || !writeIndices->defined()) {
        return;
    }
    TORCH_CHECK(cacheIndices.has_value() && cacheIndices->defined(),
        "npu_causal_conv1d: write_indices requires cache_indices.");
    TORCH_CHECK(x.dim() == 2, "npu_causal_conv1d: independent R/W requires rank-2 x.");
    TORCH_CHECK(runMode == 0 || convState.size(1) == weight.size(0) - 1,
        "npu_causal_conv1d: Decode conv_state must be [slots, width - 1, channels].");
    TORCH_CHECK(cacheIndices->dim() == 1 && writeIndices->dim() == 1 &&
        cacheIndices->sizes() == writeIndices->sizes() && (runMode == 0 || cacheIndices->size(0) == x.size(0)),
        "npu_causal_conv1d: cache_indices and write_indices must be [batch].");
    TORCH_CHECK(cacheIndices->scalar_type() == writeIndices->scalar_type() &&
        (cacheIndices->scalar_type() == at::kInt || cacheIndices->scalar_type() == at::kLong),
        "npu_causal_conv1d: cache_indices and write_indices must use the same int32 or int64 dtype.");
}

}  // namespace

at::Tensor NpuCausalConv1d(const at::Tensor& x, const at::Tensor& weight, at::Tensor& convState,
    const c10::optional<at::Tensor>& bias, const c10::optional<at::Tensor>& queryStartLoc,
    const c10::optional<at::Tensor>& cacheIndices, const c10::optional<at::Tensor>& writeIndices,
    const c10::optional<at::Tensor>& initialStateMode, const c10::optional<at::Tensor>& numAcceptedTokens,
    int64_t activationMode, int64_t padSlotId, int64_t runMode)
{
    CheckCausalConv1dInputs(x, weight, convState, bias, activationMode, runMode);
    CheckWriteIndices(x, weight, convState, cacheIndices, writeIndices, initialStateMode, runMode);
    return RunCausalConv1d(x, weight, convState, bias, queryStartLoc, cacheIndices, initialStateMode,
        numAcceptedTokens, writeIndices, activationMode, padSlotId, runMode);
}

at::Tensor NpuCausalConv1dMeta(const at::Tensor& x, const at::Tensor& weight, at::Tensor& convState,
    const c10::optional<at::Tensor>& bias, const c10::optional<at::Tensor>&,
    const c10::optional<at::Tensor>& cacheIndices, const c10::optional<at::Tensor>& writeIndices,
    const c10::optional<at::Tensor>& initialStateMode, const c10::optional<at::Tensor>&,
    int64_t activationMode, int64_t, int64_t runMode)
{
    CheckCausalConv1dInputs(x, weight, convState, bias, activationMode, runMode);
    CheckWriteIndices(x, weight, convState, cacheIndices, writeIndices, initialStateMode, runMode);
    return at::empty_like(x);
}

}  // namespace custom

TORCH_LIBRARY_IMPL(custom, PrivateUse1, m) {
    m.impl("npu_causal_conv1d", &custom::NpuCausalConv1d);
}

TORCH_LIBRARY_IMPL(custom, Meta, m) {
    m.impl("npu_causal_conv1d", &custom::NpuCausalConv1dMeta);
}
