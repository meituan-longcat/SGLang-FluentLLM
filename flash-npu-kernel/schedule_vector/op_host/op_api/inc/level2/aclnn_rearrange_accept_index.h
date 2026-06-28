
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_REARRANGE_ACCEPT_INDEX_H_
#define ACLNN_REARRANGE_ACCEPT_INDEX_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnRearrangeAcceptIndexGetWorkspaceSize
 * parameters :
 * acceptIndex : required
 * acceptLength : required
 * bs : required
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnRearrangeAcceptIndexGetWorkspaceSize(
    const aclTensor *acceptIndex,
    const aclTensor *acceptLength,
    int64_t bs,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnRearrangeAcceptIndex
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnRearrangeAcceptIndex(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
