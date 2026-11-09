/*!
 * \file assign_req_to_token_pool_tilingdata.h
 * \brief POD counterpart of MTP's AssignReqToTokenPoolTilingData.
 *
 * Originally generated in MTP via BEGIN_TILING_DATA_DEF / TILING_DATA_FIELD_DEF
 * inside op_host/op_tiling/.../assign_req_to_token_pool_tiling.h. Field names
 * and types are kept identical so the migrated kernel class (tl_->...) is
 * reused verbatim, and host-side tiling logic can populate it field-by-field.
 *
 * This header is the only tiling-related include needed by the kernel side
 * (kernel.h / entry.h); the host tiling logic lives in
 * assign_req_to_token_pool_tiling.h.
 */

#ifndef ASSIGN_REQ_TO_TOKEN_POOL_TILINGDATA_H
#define ASSIGN_REQ_TO_TOKEN_POOL_TILINGDATA_H

#include <cstdint>

namespace flash {
namespace AssignReqToTokenPool {

struct AssignReqToTokenPoolTilingData {
    uint32_t usedCoreNum;
    uint32_t blockFactor;
    uint32_t tailBlockFactor;
    uint32_t ubFactor;
    uint32_t batchSize;
    uint32_t poolLen;
};

} // namespace AssignReqToTokenPool
} // namespace flash

#endif // ASSIGN_REQ_TO_TOKEN_POOL_TILINGDATA_H
