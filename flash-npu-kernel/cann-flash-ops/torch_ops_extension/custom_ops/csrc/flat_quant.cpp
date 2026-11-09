/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#include <cmath>
#include <stdexcept>
#include <tuple>
#include <torch/library.h>

#include "ops_common.h"

namespace custom {
namespace {

constexpr int64_t DST_DTYPE_INT4 = 16;
constexpr int64_t DST_DTYPE_FP4 = 296;
constexpr int64_t INT4_PER_INT32 = 8;
constexpr int64_t FP4_PER_UINT8 = 2;
constexpr int64_t MX_BLOCK_SIZE = 64;
constexpr int64_t MAX_GROUP_NUM = 1024;

struct FlatQuantOutputs {
    at::Tensor out;
    at::Tensor scale;
};

void CheckInputShape(const at::Tensor& x, const at::Tensor& p1, const at::Tensor& p2, int64_t dstDtype)
{
    TORCH_CHECK(x.dim() == 3, "x must be a 3-D tensor [K, N1, N2]");
    const int64_t k = x.size(0);
    const int64_t n1 = x.size(1);
    const int64_t n2 = x.size(2);
    TORCH_CHECK(k >= 0 && n1 > 0 && n2 > 0,
                "x dim0 must be non-negative and dim1/dim2 must be greater than 0");
    TORCH_CHECK(p1.dim() == 2 && p1.size(0) == n1 && p1.size(1) == n1,
                "kronecker_p1 must have shape [N1, N1]");
    TORCH_CHECK(p2.dim() == 2 && p2.size(0) == n2 && p2.size(1) == n2,
                "kronecker_p2 must have shape [N2, N2]");
    if (dstDtype == DST_DTYPE_INT4) {
        TORCH_CHECK(n2 % INT4_PER_INT32 == 0, "N2 must be divisible by 8 for INT4 output");
    } else {
        TORCH_CHECK(n2 % FP4_PER_UINT8 == 0, "N2 must be even for FP4 output");
    }
}

void CheckGroupList(const c10::optional<at::Tensor>& groupList, int64_t groupListType, const at::Device& device)
{
    TORCH_CHECK(groupListType >= 0 && groupListType <= 2, "group_list_type must be in range [0, 2]");
    if (!groupList.has_value() || !groupList->defined()) {
        return;
    }
    TORCH_CHECK(groupList->device() == device, "group_list must be on the same NPU as x");
    TORCH_CHECK(groupList->scalar_type() == at::kLong, "group_list dtype must be int64");
    if (groupListType == 2) {
        TORCH_CHECK(groupList->dim() == 2 && groupList->size(1) == 2,
                    "group_list must have shape [G, 2] when group_list_type is 2");
    } else {
        TORCH_CHECK(groupList->dim() == 1,
                    "group_list must be 1-D when group_list_type is 0 or 1");
    }
    TORCH_CHECK(groupList->size(0) <= MAX_GROUP_NUM, "group_list dim0 must not exceed 1024");
}

void CheckFlatQuantInputs(const at::Tensor& x, const at::Tensor& p1, const at::Tensor& p2, double clipRatio,
                          int64_t dstDtype, double dstTypeMax, const c10::optional<at::Tensor>& groupList,
                          int64_t groupListType, bool checkDevice)
{
    if (checkDevice) {
        TORCH_CHECK(torch_npu::utils::is_npu(x) && torch_npu::utils::is_npu(p1) && torch_npu::utils::is_npu(p2),
                    "all input tensors must be NPU tensors");
        TORCH_CHECK(x.device() == p1.device() && x.device() == p2.device(),
                    "all input tensors must be on the same NPU");
    }
    TORCH_CHECK(x.scalar_type() == at::kHalf || x.scalar_type() == at::kBFloat16,
                "x dtype must be float16 or bfloat16");
    TORCH_CHECK(p1.scalar_type() == x.scalar_type() && p2.scalar_type() == x.scalar_type(),
                "kronecker_p1 and kronecker_p2 must have the same dtype as x");
    TORCH_CHECK(clipRatio > 0.0 && clipRatio <= 1.0, "clip_ratio must be in (0, 1]");
    TORCH_CHECK(dstDtype == DST_DTYPE_INT4 || dstDtype == DST_DTYPE_FP4,
                "dst_dtype must be 16 (INT4) or 296 (FP4)");
    TORCH_CHECK(dstTypeMax == 0.0 || (dstTypeMax >= 6.0 && dstTypeMax <= 12.0),
                "dst_type_max must be 0 or in range [6, 12]");
    CheckInputShape(x, p1, p2, dstDtype);
    CheckGroupList(groupList, groupListType, x.device());
}

FlatQuantOutputs FlatQuantMetaImpl(const at::Tensor& x, int64_t dstDtype)
{
    const int64_t k = x.size(0);
    const int64_t n1 = x.size(1);
    const int64_t n2 = x.size(2);
    if (dstDtype == DST_DTYPE_INT4) {
        return {at::empty({k, n1, n2 / INT4_PER_INT32}, x.options().dtype(at::kInt)),
                at::empty({k}, x.options().dtype(at::kFloat))};
    }
    const int64_t flattened = n1 * n2;
    const int64_t scaleBlocks = (flattened + MX_BLOCK_SIZE - 1) / MX_BLOCK_SIZE;
    return {at::empty({k, flattened / FP4_PER_UINT8}, x.options().dtype(at::kByte)),
            at::empty({k, scaleBlocks, 2}, x.options().dtype(at::kByte))};
}

std::tuple<TensorWrapper, TensorWrapper> FlatQuantTensorWrapper(const FlatQuantOutputs& outputs)
{
    return {TensorWrapper{outputs.out, ACL_FLOAT4_E2M1}, TensorWrapper{outputs.scale, ACL_FLOAT8_E8M0}};
}

} // namespace

std::tuple<at::Tensor, at::Tensor> FlatQuantNpu(const at::Tensor& x, const at::Tensor& kroneckerP1,
                                                const at::Tensor& kroneckerP2, double clipRatio, int64_t dstDtype,
                                                double dstTypeMax, const c10::optional<at::Tensor>& groupList,
                                                int64_t groupListType, bool clearPadding)
{
    CheckFlatQuantInputs(x, kroneckerP1, kroneckerP2, clipRatio, dstDtype, dstTypeMax, groupList, groupListType, true);
    auto outputs = FlatQuantMetaImpl(x, dstDtype);
    if (dstDtype == DST_DTYPE_INT4) {
        EXEC_NPU_CMD_V1(aclnnFlatQuantV4, x, kroneckerP1, kroneckerP2, groupList, clipRatio, dstTypeMax,
                        groupListType, clearPadding, outputs.out, outputs.scale);
    } else {
        auto wrappers = FlatQuantTensorWrapper(outputs);
        EXEC_NPU_CMD_V1(aclnnFlatQuantV4, x, kroneckerP1, kroneckerP2, groupList, clipRatio, dstTypeMax,
                        groupListType, clearPadding, std::get<0>(wrappers), std::get<1>(wrappers));
    }
    return {outputs.out, outputs.scale};
}

std::tuple<at::Tensor, at::Tensor> FlatQuantMeta(const at::Tensor& x, const at::Tensor& kroneckerP1,
                                                 const at::Tensor& kroneckerP2, double clipRatio, int64_t dstDtype,
                                                 double dstTypeMax, const c10::optional<at::Tensor>& groupList,
                                                 int64_t groupListType, bool /* clearPadding */)
{
    CheckFlatQuantInputs(x, kroneckerP1, kroneckerP2, clipRatio, dstDtype, dstTypeMax, groupList, groupListType, false);
    auto outputs = FlatQuantMetaImpl(x, dstDtype);
    return {outputs.out, outputs.scale};
}

} // namespace custom

TORCH_LIBRARY_IMPL(custom, PrivateUse1, m) {
    m.impl("flat_quant", &custom::FlatQuantNpu);
}

TORCH_LIBRARY_IMPL(custom, Meta, m) {
    m.impl("flat_quant", &custom::FlatQuantMeta);
}
