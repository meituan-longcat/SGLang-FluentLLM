#include <torch/library.h>

#include "ops_common.h"

namespace custom {
using SelectResult = std::tuple<at::Tensor, at::Tensor>;

namespace {
void CheckInputs(const at::Tensor &topkIndicesLocal,
                 const at::Tensor &topkPos, int64_t kvpRank)
{
    TORCH_CHECK(topkIndicesLocal.scalar_type() == at::kInt,
        "topk_indices_local dtype must be int32, but got ",
        topkIndicesLocal.scalar_type(), ".");
    TORCH_CHECK(topkPos.scalar_type() == at::kInt ||
        topkPos.scalar_type() == at::kLong,
        "topk_pos dtype must be int32 or int64, but got ",
        topkPos.scalar_type(), ".");
    TORCH_CHECK(topkIndicesLocal.dim() == 3 &&
        topkIndicesLocal.size(1) == 1,
        "topk_indices_local must have shape [T, 1, D].");
    TORCH_CHECK(topkPos.dim() == 2,
        "topk_pos must have shape [T, K].");
    TORCH_CHECK(topkIndicesLocal.size(0) == topkPos.size(0),
        "topk_indices_local and topk_pos must have the same T.");
    TORCH_CHECK(topkIndicesLocal.size(2) > 0 &&
        topkIndicesLocal.size(2) <= 2048 &&
        topkIndicesLocal.size(2) % 8 == 0,
        "D must be a positive multiple of 8 no greater than 2048.");
    TORCH_CHECK(topkPos.size(1) == 2048,
        "K must be 2048, but got ", topkPos.size(1), ".");
    TORCH_CHECK(kvpRank >= 0, "kvp_rank must be non-negative.");
}

SelectResult MakeOutputs(const at::Tensor &topkIndicesLocal,
                         const at::Tensor &topkPos)
{
    auto options = topkIndicesLocal.options().dtype(at::kInt);
    auto output = at::empty(
        {topkPos.size(0), 1, topkPos.size(1)}, options);
    auto validChunks = at::empty({topkPos.size(0)}, options);
    return {output, validChunks};
}
} // namespace

SelectResult NpuSelectLocalTopkIndices(
    const at::Tensor &topkIndicesLocal, const at::Tensor &topkPos,
    int64_t kvpRank)
{
    CheckInputs(topkIndicesLocal, topkPos, kvpRank);
    at::Tensor topkPosInt = topkPos.scalar_type() == at::kInt
        ? topkPos : topkPos.to(at::kInt);
    auto [output, validChunks] = MakeOutputs(topkIndicesLocal, topkPos);
    EXEC_NPU_CMD_V1(aclnnSelectLocalTopkIndices, topkIndicesLocal,
        topkPosInt, kvpRank, output, validChunks);
    return {output, validChunks};
}

SelectResult NpuSelectLocalTopkIndicesMeta(
    const at::Tensor &topkIndicesLocal, const at::Tensor &topkPos,
    int64_t kvpRank)
{
    CheckInputs(topkIndicesLocal, topkPos, kvpRank);
    return MakeOutputs(topkIndicesLocal, topkPos);
}
} // namespace custom

TORCH_LIBRARY_IMPL(custom, PrivateUse1, m) {
    m.impl("npu_select_local_topk_indices",
           &custom::NpuSelectLocalTopkIndices);
}
TORCH_LIBRARY_IMPL(custom, Meta, m) {
    m.impl("npu_select_local_topk_indices",
           &custom::NpuSelectLocalTopkIndicesMeta);
}
