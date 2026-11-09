#ifndef ACLNN_SPARSE_MLA_ATTENTION_PREFILL_H
#define ACLNN_SPARSE_MLA_ATTENTION_PREFILL_H

#include "aclnn/acl_meta.h"
#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

__attribute__((visibility("default"))) aclnnStatus
aclnnSparseMlaAttentionPrefillGetWorkspaceSize(
    const aclTensor *query, const aclTensor *gatheredKvPages,
    const aclTensor *logicalSparseRows, const aclTensor *scanCount,
    const aclTensor *pageTable, const aclTensor *querySeq,
    double scaleValue, bool headMajorOutput, const aclTensor *attentionOut,
    uint64_t *workspaceSize, aclOpExecutor **executor);

__attribute__((visibility("default"))) aclnnStatus
aclnnSparseMlaAttentionPrefill(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
    const aclrtStream stream);

#ifdef __cplusplus
}
#endif
#endif
