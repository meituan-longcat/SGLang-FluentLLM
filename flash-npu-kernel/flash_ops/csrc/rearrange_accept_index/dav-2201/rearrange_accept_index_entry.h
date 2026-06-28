/*!
 * \file rearrange_accept_index_entry.h
 * \brief Direct-invoke kernel entry — glue between the host dispatch and the
 *        MTP-style kernel class in rearrange_accept_index_kernel.h.
 *
 * Counterpart of MTP's extern "C" kernel in
 *   op_kernel/rearrange_accept_index/rearrange_accept_index.cpp
 * but:
 *   - TilingData is passed by value as a struct kernel param (no
 *     GET_TILING_DATA);
 *   - dtype is a template parameter (no DTYPE_* macro).
 */

#ifndef REARRANGE_ACCEPT_INDEX_ENTRY_H
#define REARRANGE_ACCEPT_INDEX_ENTRY_H

#include "kernel_operator.h"
#include "rearrange_accept_index_tilingdata.h"
#include "rearrange_accept_index_kernel.h"

namespace flash {
namespace RearrangeAcceptIndex {

template <typename T>
__global__ __aicore__ __vector__ void rearrange_accept_index(
    GM_ADDR accept_index, GM_ADDR accept_length, GM_ADDR output,
    RearrangeAcceptIndexTilingData tilingData)
{
    AscendC::TPipe pipe;
    RearrangeAcceptIndex<T> op(&pipe, &tilingData);
    op.Init(accept_index, accept_length, output);
    op.Process();
}

} // namespace RearrangeAcceptIndex
} // namespace flash

#endif // REARRANGE_ACCEPT_INDEX_ENTRY_H
