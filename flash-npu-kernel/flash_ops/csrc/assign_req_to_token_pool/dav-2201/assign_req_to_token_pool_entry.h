/*!
 * \file assign_req_to_token_pool_entry.h
 * \brief Direct-invoke kernel entry — glue between the host dispatch and the
 *        MTP-style kernel class in assign_req_to_token_pool_kernel.h.
 *
 * Counterpart of MTP's extern "C" kernel in
 *   op_kernel/assign_req_to_token_pool/assign_req_to_token_pool.cpp
 * but:
 *   - TilingData is passed by value as a struct kernel param (no
 *     GET_TILING_DATA);
 *   - dtype is a template parameter (no DTYPE_OUT_CACHE_LOC macro).
 *
 * Isolating this entry here keeps assign_req_to_token_pool_kernel.h as close
 * as possible to the original MTP kernel header, making future re-syncs easy.
 */

#ifndef ASSIGN_REQ_TO_TOKEN_POOL_ENTRY_H
#define ASSIGN_REQ_TO_TOKEN_POOL_ENTRY_H

#include "kernel_operator.h"
#include "assign_req_to_token_pool_tilingdata.h"
#include "assign_req_to_token_pool_kernel.h"

namespace flash {
namespace AssignReqToTokenPool {

template <typename T>
__global__ __aicore__ __vector__ void assign_req_to_token_pool(
    GM_ADDR req_pool_indices, GM_ADDR extend_lens, GM_ADDR alloced_lens,
    GM_ADDR out_cache_loc, GM_ADDR req_to_token_pool,
    AssignReqToTokenPoolTilingData tilingData)
{
    AscendC::TPipe pipe;
    AssignReqToTokenPool<T> op(&pipe, &tilingData);
    op.Init(req_pool_indices, extend_lens, alloced_lens, out_cache_loc,
            req_to_token_pool);
    op.Process();
}

} // namespace AssignReqToTokenPool
} // namespace flash

#endif // ASSIGN_REQ_TO_TOKEN_POOL_ENTRY_H
