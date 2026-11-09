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

constexpr int64_t MAX_COPY_ROWS = 256;
constexpr int64_t DATA_BLOCK_BYTES = 32;

void CheckKvCacheCopyRowsInputs(
    const at::Tensor& cache,
    const at::Tensor& srcIndices,
    const at::Tensor& dstIndices)
{
    TORCH_CHECK(cache.dim() == 2,
                "cache must be a 2-D [rows, width] tensor, but got ", cache.dim(), " dimensions.");
    TORCH_CHECK(cache.size(0) > 0 && cache.size(1) > 0, "cache dimensions must be positive.");
    TORCH_CHECK(cache.scalar_type() == at::kHalf || cache.scalar_type() == at::kBFloat16 ||
                    cache.scalar_type() == at::kChar,
                "cache dtype must be float16, bfloat16, or int8, but got ", cache.scalar_type());
    TORCH_CHECK(srcIndices.scalar_type() == at::kInt || srcIndices.scalar_type() == at::kLong,
                "src_indices dtype must be int32 or int64, but got ", srcIndices.scalar_type());
    TORCH_CHECK(srcIndices.scalar_type() == dstIndices.scalar_type(),
                "src_indices and dst_indices must have the same dtype.");
    TORCH_CHECK(srcIndices.numel() == dstIndices.numel(),
                "src_indices and dst_indices must have the same number of elements.");
    TORCH_CHECK(srcIndices.numel() >= 1 && srcIndices.numel() <= MAX_COPY_ROWS,
                "the number of copied rows must be in [1, 256], but got ", srcIndices.numel());
    TORCH_CHECK(srcIndices.is_contiguous() && dstIndices.is_contiguous(),
                "src_indices and dst_indices must be contiguous.");
    TORCH_CHECK(cache.device() == srcIndices.device() && cache.device() == dstIndices.device(),
                "cache and indices must be on the same device.");
    TORCH_CHECK(cache.is_contiguous(), "cache must be contiguous.");
    const auto rowBytes = cache.size(1) * cache.element_size();
    TORCH_CHECK(rowBytes % DATA_BLOCK_BYTES == 0 || cache.scalar_type() == at::kChar,
                "unaligned cache rows are supported only for int8, but got ", rowBytes, " bytes.");
}

void RunKvCacheCopyRows(
    at::Tensor& cache,
    const at::Tensor& srcIndices,
    const at::Tensor& dstIndices)
{
    CheckKvCacheCopyRowsInputs(cache, srcIndices, dstIndices);
    EXEC_NPU_CMD_V1(aclnnKvCacheCopyRows, cache, srcIndices, dstIndices);
}

}  // namespace

at::Tensor NpuKvCacheCopyRows(
    const at::Tensor& cache,
    const at::Tensor& srcIndices,
    const at::Tensor& dstIndices)
{
    auto output = cache.clone();
    RunKvCacheCopyRows(output, srcIndices, dstIndices);
    return output;
}

at::Tensor NpuKvCacheCopyRowsInplace(
    at::Tensor& cache,
    const at::Tensor& srcIndices,
    const at::Tensor& dstIndices)
{
    RunKvCacheCopyRows(cache, srcIndices, dstIndices);
    return cache;
}

at::Tensor NpuKvCacheCopyRowsMeta(
    const at::Tensor& cache,
    const at::Tensor& srcIndices,
    const at::Tensor& dstIndices)
{
    CheckKvCacheCopyRowsInputs(cache, srcIndices, dstIndices);
    return at::empty_like(cache);
}

at::Tensor NpuKvCacheCopyRowsInplaceMeta(
    at::Tensor& cache,
    const at::Tensor& srcIndices,
    const at::Tensor& dstIndices)
{
    CheckKvCacheCopyRowsInputs(cache, srcIndices, dstIndices);
    return cache;
}

}  // namespace custom

TORCH_LIBRARY_IMPL(custom, PrivateUse1, m) {
    m.impl("npu_kv_cache_copy_rows", &custom::NpuKvCacheCopyRows);
    m.impl("npu_kv_cache_copy_rows_", &custom::NpuKvCacheCopyRowsInplace);
}

TORCH_LIBRARY_IMPL(custom, Meta, m) {
    m.impl("npu_kv_cache_copy_rows", &custom::NpuKvCacheCopyRowsMeta);
    m.impl("npu_kv_cache_copy_rows_", &custom::NpuKvCacheCopyRowsInplaceMeta);
}
