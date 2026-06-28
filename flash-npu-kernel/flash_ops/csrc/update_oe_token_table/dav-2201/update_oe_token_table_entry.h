/*!
 * \file update_oe_token_table_entry.h
 * \brief Direct-invoke kernel entry — glue between the host dispatch and the
 *        kernel class in update_oe_token_table_kernel.h.
 *
 * Counterpart of the original extern "C" kernel in
 *   op_kernel/update_oe_token_table.cpp
 * but:
 *   - TilingData is passed by value as a struct kernel param (no
 *     GET_TILING_DATA);
 *   - no TILING_KEY_IS dispatch (single code path).
 *
 * The kernel is not templated (int32 hard-coded for tokens/req_lens/
 * column_starts/ignore_tokens/output, int64 for row_indices), so this entry
 * is also non-templated.
 */

#ifndef UPDATE_OE_TOKEN_TABLE_ENTRY_H
#define UPDATE_OE_TOKEN_TABLE_ENTRY_H

#include "kernel_operator.h"
#include "update_oe_token_table_tilingdata.h"
#include "update_oe_token_table_kernel.h"

namespace flash {
namespace UpdateOeTokenTable {

__global__ __aicore__ __vector__ void update_oe_token_table(
    GM_ADDR tokens, GM_ADDR req_lens, GM_ADDR row_indices,
    GM_ADDR column_starts, GM_ADDR ignore_tokens, GM_ADDR oe_token_table,
    GM_ADDR workspace, UpdateOeTokenTableTilingData tilingData)
{
    AscendC::TPipe pipe;
    UpdateOeTokenTable op(&pipe, &tilingData);
    op.Init(tokens, req_lens, row_indices, column_starts, ignore_tokens,
            oe_token_table, workspace);
    op.Process();
}

} // namespace UpdateOeTokenTable
} // namespace flash

#endif // UPDATE_OE_TOKEN_TABLE_ENTRY_H
