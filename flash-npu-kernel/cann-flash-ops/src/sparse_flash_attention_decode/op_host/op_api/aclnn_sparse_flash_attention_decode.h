#ifndef ACLNN_SPARSE_FLASH_ATTENTION_DECODE_H
#define ACLNN_SPARSE_FLASH_ATTENTION_DECODE_H

#include "aclnn/acl_meta.h"
#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

__attribute__((visibility("default"))) aclnnStatus aclnnSparseFlashAttentionDecodeGetWorkspaceSize(
    const aclTensor *query, const aclTensor *key, const aclTensor *value,
    const aclTensor *sparseIndices, const aclTensor *validChunks,
    const aclTensor *blockTableOptional,
    const aclTensor *actualSeqLengthsQueryOptional,
    const aclTensor *actualSeqLengthsKvOptional,
    const aclTensor *queryRopeOptional, const aclTensor *keyRopeOptional,
    double scaleValue, int64_t sparseBlockSizeOptional,
    char *layoutQueryOptional, char *layoutKvOptional, int64_t sparseMode,
    int64_t preTokens, int64_t nextTokens, int64_t attentionMode,
    bool returnSoftmaxLse, const aclTensor *attentionOut,
    const aclTensor *softmaxMax, const aclTensor *softmaxSum,
    uint64_t *workspaceSize, aclOpExecutor **executor);

__attribute__((visibility("default"))) aclnnStatus aclnnSparseFlashAttentionDecode(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
    const aclrtStream stream);

__attribute__((visibility("default"))) aclnnStatus aclnnSparseFlashAttentionDecodeV2GetWorkspaceSize(
    const aclTensor *query, const aclTensor *key, const aclTensor *value,
    const aclTensor *sparseIndices, const aclTensor *validChunks,
    const aclTensor *blockTableOptional,
    const aclTensor *actualSeqLengthsQueryOptional,
    const aclTensor *actualSeqLengthsKvOptional,
    const aclTensor *queryRopeOptional, const aclTensor *keyRopeOptional,
    double scaleValue, int64_t sparseBlockSizeOptional,
    char *layoutQueryOptional, char *layoutKvOptional, int64_t sparseMode,
    int64_t preTokens, int64_t nextTokens, int64_t attentionMode,
    bool returnSoftmaxLse, int64_t sparsePrepMode,
    const aclTensor *attentionOut, const aclTensor *softmaxMax,
    const aclTensor *softmaxSum, uint64_t *workspaceSize,
    aclOpExecutor **executor);

__attribute__((visibility("default"))) aclnnStatus aclnnSparseFlashAttentionDecodeV2(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
    const aclrtStream stream);

#ifdef __cplusplus
}
#endif
#endif
