#include <torch/library.h>

#include "ops_common.h"

namespace custom {
namespace {
constexpr int64_t HEADS = 64;
constexpr int64_t LATENT = 512;
constexpr int64_t ROPE = 64;
constexpr int64_t WIDTH = LATENT + ROPE;
constexpr int64_t PAGE = 128;
constexpr int64_t TOPK = 2048;

at::Tensor MakeOutput(const at::Tensor &query, bool headMajorOutput)
{
    return headMajorOutput
        ? at::empty({HEADS, query.size(0), LATENT}, query.options())
        : at::empty({query.size(0), HEADS, LATENT}, query.options());
}

void CheckInputs(const at::Tensor &query,
                 const at::Tensor &gatheredKvPages,
                 const at::Tensor &logicalSparseRows,
                 const at::Tensor &scanCount,
                 const at::Tensor &pageTable,
                 const at::Tensor &querySeq)
{
    TORCH_CHECK(query.is_contiguous() && gatheredKvPages.is_contiguous() &&
                logicalSparseRows.is_contiguous() && scanCount.is_contiguous() &&
                pageTable.is_contiguous() && querySeq.is_contiguous(),
        "all inputs must be contiguous.");
    TORCH_CHECK(query.device() == gatheredKvPages.device() &&
                query.device() == logicalSparseRows.device() &&
                query.device() == scanCount.device() &&
                query.device() == pageTable.device() &&
                query.device() == querySeq.device(),
        "all inputs must be on the same device.");
    TORCH_CHECK(query.scalar_type() == at::kBFloat16,
        "query must be BF16.");
    TORCH_CHECK(query.dim() == 3 && query.size(0) > 0 &&
                query.size(1) == HEADS && query.size(2) == WIDTH,
        "query must have shape [Q, 64, 576] with Q > 0.");
    TORCH_CHECK(gatheredKvPages.scalar_type() == at::kBFloat16,
        "gathered_kv_pages must be BF16.");
    TORCH_CHECK(gatheredKvPages.dim() == 3 &&
                gatheredKvPages.size(0) > 0 &&
                gatheredKvPages.size(1) == PAGE &&
                gatheredKvPages.size(2) == WIDTH,
        "gathered_kv_pages must have shape [P, 128, 576].");
    TORCH_CHECK(logicalSparseRows.scalar_type() == at::kInt,
        "logical_sparse_rows must be int32.");
    TORCH_CHECK(logicalSparseRows.dim() == 2 &&
                logicalSparseRows.size(0) == query.size(0) &&
                logicalSparseRows.size(1) == TOPK,
        "logical_sparse_rows must have shape [Q, 2048].");
    TORCH_CHECK(scanCount.scalar_type() == at::kInt &&
                scanCount.dim() == 1 && scanCount.size(0) == query.size(0),
        "scan_count must be int32 with shape [Q].");
    TORCH_CHECK(pageTable.scalar_type() == at::kInt &&
                pageTable.dim() == 2 && pageTable.size(0) > 0 &&
                pageTable.size(1) > 0 && pageTable.size(1) <= 10000,
        "page_table must be int32 [B, max_pages], max_pages <= 10000.");
    TORCH_CHECK(querySeq.scalar_type() == at::kInt &&
                querySeq.dim() == 1 && querySeq.size(0) == query.size(0),
        "query_seq must be int32 with shape [Q].");
}
} // namespace

at::Tensor NpuSparseMlaAttentionPrefill(
    const at::Tensor &query, const at::Tensor &gatheredKvPages,
    const at::Tensor &logicalSparseRows, const at::Tensor &scanCount,
    const at::Tensor &pageTable, const at::Tensor &querySeq,
    double scaleValue, bool headMajorOutput)
{
    CheckInputs(query, gatheredKvPages, logicalSparseRows, scanCount,
                pageTable, querySeq);
    auto out = MakeOutput(query, headMajorOutput);
    EXEC_NPU_CMD_V1(aclnnSparseMlaAttentionPrefill, query, gatheredKvPages,
                    logicalSparseRows, scanCount, pageTable, querySeq,
                    scaleValue, headMajorOutput, out);
    return out;
}

at::Tensor NpuSparseMlaAttentionPrefillMeta(
    const at::Tensor &query, const at::Tensor &gatheredKvPages,
    const at::Tensor &logicalSparseRows, const at::Tensor &scanCount,
    const at::Tensor &pageTable, const at::Tensor &querySeq,
    double, bool headMajorOutput)
{
    CheckInputs(query, gatheredKvPages, logicalSparseRows, scanCount,
                pageTable, querySeq);
    return MakeOutput(query, headMajorOutput);
}
} // namespace custom

TORCH_LIBRARY_IMPL(custom, PrivateUse1, m) {
    m.impl("npu_sparse_mla_attention_prefill",
           &custom::NpuSparseMlaAttentionPrefill);
}
TORCH_LIBRARY_IMPL(custom, Meta, m) {
    m.impl("npu_sparse_mla_attention_prefill",
           &custom::NpuSparseMlaAttentionPrefillMeta);
}
