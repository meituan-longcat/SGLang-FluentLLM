#include "aclnn_sparse_mla_attention_prefill.h"

namespace {
extern "C" aclnnStatus
aclnnCustomInnerSparseMlaAttentionPrefillGetWorkspaceSize(
    const aclTensor *query, const aclTensor *gatheredKvPages,
    const aclTensor *logicalSparseRows, const aclTensor *scanCount,
    const aclTensor *pageTable, const aclTensor *querySeq,
    double scaleValue, bool headMajorOutput, const aclTensor *attentionOut,
    uint64_t *workspaceSize, aclOpExecutor **executor);

extern "C" aclnnStatus aclnnCustomInnerSparseMlaAttentionPrefill(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
    const aclrtStream stream);
} // namespace

extern "C" aclnnStatus aclnnInnerSparseMlaAttentionPrefillGetWorkspaceSize(
    const aclTensor *query, const aclTensor *gatheredKvPages,
    const aclTensor *logicalSparseRows, const aclTensor *scanCount,
    const aclTensor *pageTable, const aclTensor *querySeq,
    double scaleValue, bool headMajorOutput, const aclTensor *attentionOut,
    uint64_t *workspaceSize, aclOpExecutor **executor)
{
    return aclnnCustomInnerSparseMlaAttentionPrefillGetWorkspaceSize(
        query, gatheredKvPages, logicalSparseRows, scanCount, pageTable,
        querySeq, scaleValue, headMajorOutput, attentionOut, workspaceSize,
        executor);
}

extern "C" aclnnStatus aclnnInnerSparseMlaAttentionPrefill(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
    const aclrtStream stream)
{
    return aclnnCustomInnerSparseMlaAttentionPrefill(
        workspace, workspaceSize, executor, stream);
}

extern "C" aclnnStatus aclnnSparseMlaAttentionPrefillGetWorkspaceSize(
    const aclTensor *query, const aclTensor *gatheredKvPages,
    const aclTensor *logicalSparseRows, const aclTensor *scanCount,
    const aclTensor *pageTable, const aclTensor *querySeq,
    double scaleValue, bool headMajorOutput, const aclTensor *attentionOut,
    uint64_t *workspaceSize, aclOpExecutor **executor)
{
    return aclnnInnerSparseMlaAttentionPrefillGetWorkspaceSize(
        query, gatheredKvPages, logicalSparseRows, scanCount, pageTable,
        querySeq, scaleValue, headMajorOutput, attentionOut, workspaceSize,
        executor);
}

extern "C" aclnnStatus aclnnSparseMlaAttentionPrefill(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
    const aclrtStream stream)
{
    return aclnnInnerSparseMlaAttentionPrefill(
        workspace, workspaceSize, executor, stream);
}
