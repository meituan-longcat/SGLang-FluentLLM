
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_ASSIGN_REQ_TO_TOKEN_POOL_H_
#define ACLNN_ASSIGN_REQ_TO_TOKEN_POOL_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnAssignReqToTokenPoolGetWorkspaceSize
 * parameters :
 * reqPoolIndices : required
 * extendLens : required
 * allocedLens : required
 * outCacheLoc : required
 * bs : required
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnAssignReqToTokenPoolGetWorkspaceSize(
    const aclTensor *reqPoolIndices,
    const aclTensor *extendLens,
    const aclTensor *allocedLens,
    const aclTensor *outCacheLoc,
    int64_t bs,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnAssignReqToTokenPool
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnAssignReqToTokenPool(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
