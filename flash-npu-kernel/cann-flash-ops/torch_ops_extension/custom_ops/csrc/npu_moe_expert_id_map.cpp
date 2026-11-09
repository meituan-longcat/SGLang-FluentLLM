/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

#include <torch/library.h>

#include "ops_common.h"

namespace custom {
namespace {

void CheckMoeExpertIdMapInputs(
    const at::Tensor &logicalIds, const at::Tensor &expertMap,
    bool checkMapNonEmpty)
{
    TORCH_CHECK(logicalIds.scalar_type() == at::kInt,
                "logical_ids dtype must be int32, but got ",
                logicalIds.scalar_type(), ".");
    TORCH_CHECK(expertMap.scalar_type() == at::kInt,
                "expert_map dtype must be int32, but got ",
                expertMap.scalar_type(), ".");
    TORCH_CHECK(logicalIds.is_contiguous(),
                "logical_ids must be contiguous.");
    TORCH_CHECK(expertMap.is_contiguous(),
                "expert_map must be contiguous.");
    TORCH_CHECK(expertMap.dim() == 1,
                "expert_map must be a 1-D tensor.");
    if (checkMapNonEmpty) {
        TORCH_CHECK(expertMap.numel() > 0,
                    "expert_map must be non-empty.");
    }
    TORCH_CHECK(logicalIds.device() == expertMap.device(),
                "logical_ids and expert_map must be on the same device.");
}

} // namespace

at::Tensor NpuMoeExpertIdMap(
    const at::Tensor &logicalIds, const at::Tensor &expertMap)
{
    CheckMoeExpertIdMapInputs(logicalIds, expertMap, true);
    auto physicalIds = at::empty_like(logicalIds);
    EXEC_NPU_CMD_V1(
        aclnnMoeExpertIdMap, logicalIds, expertMap, physicalIds);
    return physicalIds;
}

at::Tensor NpuMoeExpertIdMapMeta(
    const at::Tensor &logicalIds, const at::Tensor &expertMap)
{
    // Meta tensors may carry symbolic sizes. Avoid numel(), which requires
    // concrete sizes; the eager path and host tiling still reject an empty map.
    CheckMoeExpertIdMapInputs(logicalIds, expertMap, false);
    return at::empty_like(logicalIds);
}

} // namespace custom

TORCH_LIBRARY_IMPL(custom, PrivateUse1, m) {
    m.impl("npu_moe_expert_id_map", &custom::NpuMoeExpertIdMap);
}

TORCH_LIBRARY_IMPL(custom, Meta, m) {
    m.impl("npu_moe_expert_id_map", &custom::NpuMoeExpertIdMapMeta);
}
