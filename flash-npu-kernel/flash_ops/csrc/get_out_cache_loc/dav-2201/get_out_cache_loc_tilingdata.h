/*!
 * \file get_out_cache_loc_tilingdata.h
 * \brief POD counterpart of MTP's GetOutCacheLocTilingData
 *        (originally generated via BEGIN_TILING_DATA_DEF / TILING_DATA_FIELD_DEF
 *        in op_host/op_tiling/.../get_out_cache_loc_tiling.h). Field names and
 *        types kept identical so the migrated kernel class (tl_->...) is reused
 *        verbatim.
 */

#ifndef GET_OUT_CACHE_LOC_TILINGDATA_H
#define GET_OUT_CACHE_LOC_TILINGDATA_H

#include <cstdint>

namespace flash {
namespace GetOutCacheLoc {

struct GetOutCacheLocTilingData {
    uint32_t usedCoreNum;
    uint32_t blockFactor;
    uint32_t tailBlockFactor;
    uint32_t ubFactor;
    uint32_t batchSize;
    uint32_t poolLen;
};

} // namespace GetOutCacheLoc
} // namespace flash

#endif // GET_OUT_CACHE_LOC_TILINGDATA_H
