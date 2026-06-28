
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_GET_OUT_CACHE_LOC_H_
#define ACLNN_GET_OUT_CACHE_LOC_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnGetOutCacheLocGetWorkspaceSize
 * parameters :
 * reqToToken : required
 * reqPoolIndices : required
 * newComputeLens : required
 * cacheLens : required
 * bs : required
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnGetOutCacheLocGetWorkspaceSize(
    const aclTensor *reqToToken,
    const aclTensor *reqPoolIndices,
    const aclTensor *newComputeLens,
    const aclTensor *cacheLens,
    int64_t bs,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnGetOutCacheLoc
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnGetOutCacheLoc(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
