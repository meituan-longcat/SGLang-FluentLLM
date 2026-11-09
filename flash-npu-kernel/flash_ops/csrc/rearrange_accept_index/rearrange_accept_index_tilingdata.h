/*!
 * \file rearrange_accept_index_tilingdata.h
 * \brief POD counterpart of MTP's RearrangeAcceptIndexTilingData
 *        (originally generated via BEGIN_TILING_DATA_DEF / TILING_DATA_FIELD_DEF
 *        in op_host/op_tiling/.../rearrange_accept_index_tiling.h). Field names
 *        and types kept identical so the migrated kernel class (tiling_->...)
 *        is reused verbatim.
 */

#ifndef REARRANGE_ACCEPT_INDEX_TILINGDATA_H
#define REARRANGE_ACCEPT_INDEX_TILINGDATA_H

#include <cstdint>

namespace flash {
namespace RearrangeAcceptIndex {

struct RearrangeAcceptIndexTilingData {
    uint32_t usedCoreNum;
    uint32_t blockFactor;
    uint32_t tailBlockFactor;
    uint32_t ubFactor;
    uint32_t batchSize;
    uint32_t poolLen;
};

} // namespace RearrangeAcceptIndex
} // namespace flash

#endif // REARRANGE_ACCEPT_INDEX_TILINGDATA_H
