/*!
 * \file rearrange_accept_index_tiling.h
 * \brief Host-side tiling — mirrors MTP's BlockTiling() + UbTiling() logic in
 *        op_host/op_tiling/.../rearrange_accept_index_tiling.h.
 *
 * The TilingData POD struct itself lives in
 * rearrange_accept_index_tilingdata.h (shared with the kernel side).
 */

#ifndef REARRANGE_ACCEPT_INDEX_TILING_H
#define REARRANGE_ACCEPT_INDEX_TILING_H

#include <cstdint>
#include <torch/all.h>
#include "platform/platform_ascendc.h"

#include "rearrange_accept_index_tilingdata.h"

namespace flash {
namespace RearrangeAcceptIndex {

constexpr int64_t BLOCK_SIZE = 32;
constexpr int64_t DOUBLE_BUFFER = 2;

inline RearrangeAcceptIndexTilingData calc_tiling_data(
    int64_t batchSize, int64_t poolLen, int64_t elemSize)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    uint64_t ubSize;
    ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    int64_t coreNum = ascendcPlatform->GetCoreNumAiv();
    TORCH_CHECK(coreNum > 0, "coreNum must be positive.");
    TORCH_CHECK(ubSize > 0, "ubSize must be positive.");

    auto ceilDiv = [](int64_t a, int64_t b) {
        return (a + b - 1) / b;
    };
    auto ceilAlign = [](int64_t a, int64_t b) {
        return ((a + b - 1) / b) * b;
    };

    int64_t blockFactor = ceilDiv(batchSize, coreNum);
    int64_t usedCoreNum = ceilDiv(batchSize, blockFactor);
    blockFactor = ceilDiv(batchSize, usedCoreNum);
    int64_t tailBlockFactor =
        (batchSize % usedCoreNum == 0) ? blockFactor : (batchSize % blockFactor);

    // accept_length UB reservation: match the original op_tiling which uses
    // sizeof(int64_t) regardless of the actual dtype.
    int64_t acceptLenSize =
        ceilAlign(batchSize * static_cast<int64_t>(sizeof(int64_t)), BLOCK_SIZE);
    int64_t reserverUbSize = static_cast<int64_t>(ubSize) - acceptLenSize;
    TORCH_CHECK(reserverUbSize > 0,
        "batchSize too large to fit accept_length slot in UB.");
    int64_t ubFactor =
        ceilAlign(reserverUbSize / DOUBLE_BUFFER, BLOCK_SIZE) / elemSize;

    return RearrangeAcceptIndexTilingData{
        static_cast<uint32_t>(usedCoreNum),
        static_cast<uint32_t>(blockFactor),
        static_cast<uint32_t>(tailBlockFactor),
        static_cast<uint32_t>(ubFactor),
        static_cast<uint32_t>(batchSize),
        static_cast<uint32_t>(poolLen),
    };
}

} // namespace RearrangeAcceptIndex
} // namespace flash

#endif // REARRANGE_ACCEPT_INDEX_TILING_H
