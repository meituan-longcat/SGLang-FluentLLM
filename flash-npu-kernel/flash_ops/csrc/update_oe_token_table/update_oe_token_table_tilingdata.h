/*!
 * \file update_oe_token_table_tilingdata.h
 * \brief POD counterpart of the original UpdateOeTokenTableTilingData.
 *
 * Originally generated via BEGIN_TILING_DATA_DEF / TILING_DATA_FIELD_DEF in
 * op_host/update_oe_token_table.h. Field names/types/order are kept identical
 * so the migrated kernel class (tl_->...) is reused verbatim, and host-side
 * tiling logic can populate it field-by-field.
 */

#ifndef UPDATE_OE_TOKEN_TABLE_TILINGDATA_H
#define UPDATE_OE_TOKEN_TABLE_TILINGDATA_H

#include <cstdint>

namespace flash {
namespace UpdateOeTokenTable {

struct UpdateOeTokenTableTilingData {
    uint32_t usedCoreNum;
    uint32_t blockFactor;
    uint32_t tailBlockFactor;
    uint32_t ubFactor;
    uint32_t batchSize;
    uint32_t maxContextLen;
    uint32_t ignoreTokenNum;
};

} // namespace UpdateOeTokenTable
} // namespace flash

#endif // UPDATE_OE_TOKEN_TABLE_TILINGDATA_H
