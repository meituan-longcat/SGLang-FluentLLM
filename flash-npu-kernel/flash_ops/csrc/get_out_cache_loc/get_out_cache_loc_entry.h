/*!
 * \file get_out_cache_loc_entry.h
 * \brief Direct-invoke kernel entry — glue between the host dispatch and the
 *        MTP-style kernel class in get_out_cache_loc_kernel.h.
 *
 * Counterpart of MTP's extern "C" kernel in
 *   op_kernel/get_out_cache_loc/get_out_cache_loc.cpp
 * but:
 *   - TilingData is passed by value as a struct kernel param (no
 *     GET_TILING_DATA);
 *   - dtype is a template parameter (no DTYPE_OUT_CACHE_LOC macro).
 */

#ifndef GET_OUT_CACHE_LOC_ENTRY_H
#define GET_OUT_CACHE_LOC_ENTRY_H

#include "kernel_operator.h"
#include "get_out_cache_loc_tilingdata.h"
#include "get_out_cache_loc_kernel.h"

namespace flash {
namespace GetOutCacheLoc {

template <typename T>
__global__ __aicore__ __vector__ void get_out_cache_loc(
    GM_ADDR req_to_token, GM_ADDR req_pool_indices, GM_ADDR new_compute_lens,
    GM_ADDR cache_lens, GM_ADDR out_cache_loc,
    GetOutCacheLocTilingData tilingData)
{
    AscendC::TPipe pipe;
    GetOutCacheLoc<T> op(&pipe, &tilingData);
    op.Init(req_to_token, req_pool_indices, new_compute_lens, cache_lens,
            out_cache_loc);
    op.Process();
}

} // namespace GetOutCacheLoc
} // namespace flash

#endif // GET_OUT_CACHE_LOC_ENTRY_H
