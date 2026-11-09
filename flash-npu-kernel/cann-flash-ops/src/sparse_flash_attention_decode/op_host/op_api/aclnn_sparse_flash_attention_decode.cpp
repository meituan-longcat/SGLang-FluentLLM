#include "aclnn_sparse_flash_attention_decode.h"

namespace {
constexpr int64_t SPARSE_PREP_MODE_FULL_SCAN = 0;

extern "C" aclnnStatus aclnnCustomInnerSparseFlashAttentionDecodeGetWorkspaceSize(
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
    const aclTensor *attentionOut,
    const aclTensor *softmaxMax, const aclTensor *softmaxSum,
    uint64_t *workspaceSize, aclOpExecutor **executor);

extern "C" aclnnStatus aclnnCustomInnerSparseFlashAttentionDecode(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
    const aclrtStream stream);
}

// The open-project generator names inner ACLNN entry points
// "aclnnCustomInner*". Keep the deployed "aclnnInner*" ABI as forwarding
// symbols so the public wrapper is self-contained and the resulting opapi can
// be loaded without an older vendor library already present in the process.
extern "C" aclnnStatus aclnnInnerSparseFlashAttentionDecodeGetWorkspaceSize(
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
    uint64_t *workspaceSize, aclOpExecutor **executor)
{
    return aclnnCustomInnerSparseFlashAttentionDecodeGetWorkspaceSize(
        query, key, value, sparseIndices, validChunks, blockTableOptional,
        actualSeqLengthsQueryOptional, actualSeqLengthsKvOptional,
        queryRopeOptional, keyRopeOptional, scaleValue, sparseBlockSizeOptional,
        layoutQueryOptional, layoutKvOptional, sparseMode, preTokens, nextTokens,
        attentionMode, returnSoftmaxLse, SPARSE_PREP_MODE_FULL_SCAN,
        attentionOut, softmaxMax, softmaxSum,
        workspaceSize, executor);
}

extern "C" aclnnStatus aclnnInnerSparseFlashAttentionDecode(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
    const aclrtStream stream)
{
    return aclnnCustomInnerSparseFlashAttentionDecode(
        workspace, workspaceSize, executor, stream);
}

extern "C" aclnnStatus aclnnSparseFlashAttentionDecodeGetWorkspaceSize(
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
    uint64_t *workspaceSize, aclOpExecutor **executor)
{
    return aclnnInnerSparseFlashAttentionDecodeGetWorkspaceSize(
        query, key, value, sparseIndices, validChunks, blockTableOptional,
        actualSeqLengthsQueryOptional, actualSeqLengthsKvOptional,
        queryRopeOptional, keyRopeOptional, scaleValue,
        sparseBlockSizeOptional,
        layoutQueryOptional, layoutKvOptional, sparseMode, preTokens, nextTokens,
        attentionMode, returnSoftmaxLse, attentionOut, softmaxMax,
        softmaxSum, workspaceSize, executor);
}

extern "C" aclnnStatus aclnnSparseFlashAttentionDecode(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
    const aclrtStream stream)
{
    return aclnnInnerSparseFlashAttentionDecode(
        workspace, workspaceSize, executor, stream);
}

extern "C" aclnnStatus aclnnSparseFlashAttentionDecodeV2GetWorkspaceSize(
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
    aclOpExecutor **executor)
{
    return aclnnCustomInnerSparseFlashAttentionDecodeGetWorkspaceSize(
        query, key, value, sparseIndices, validChunks, blockTableOptional,
        actualSeqLengthsQueryOptional, actualSeqLengthsKvOptional,
        queryRopeOptional, keyRopeOptional, scaleValue,
        sparseBlockSizeOptional, layoutQueryOptional, layoutKvOptional,
        sparseMode, preTokens, nextTokens, attentionMode, returnSoftmaxLse,
        sparsePrepMode, attentionOut, softmaxMax, softmaxSum, workspaceSize,
        executor);
}

extern "C" aclnnStatus aclnnSparseFlashAttentionDecodeV2(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
    const aclrtStream stream)
{
    return aclnnCustomInnerSparseFlashAttentionDecode(
        workspace, workspaceSize, executor, stream);
}
