/*!
 * \file update_oe_token_table_tiling.h
 * \brief Host-side tiling — mirrors op_host/update_oe_token_table.cpp
 *        TilingFunc(). Only framework glue is replaced (gert::TilingContext*
 *        -> PlatformAscendCManager, attrs/shapes passed by value). Math is
 *        preserved line-by-line.
 */

#ifndef UPDATE_OE_TOKEN_TABLE_TILING_H
#define UPDATE_OE_TOKEN_TABLE_TILING_H

#include <cstdint>
#include <type_traits>
#include <torch/all.h>
#include "platform/platform_ascendc.h"

#include "update_oe_token_table_tilingdata.h"

namespace flash {
namespace UpdateOeTokenTable {

constexpr int64_t BLOCK_SIZE = 32;
constexpr int64_t DOUBLE_BUFEER = 2;

template <typename T>
inline typename std::enable_if<std::is_integral<T>::value, T>::type CeilDiv(T x, T y)
{
    if (y != 0 && x != 0) {
        const T quotient = x / y;
        return (x % y != 0 && ((x ^ y) >= 0)) ? (quotient + 1) : quotient;
    }
    return x;
}

template <typename T>
inline typename std::enable_if<std::is_integral<T>::value, T>::type CeilAlign(T x, T align)
{
    return CeilDiv(x, align) * align;
}

template <typename T>
inline typename std::enable_if<std::is_integral<T>::value, T>::type FloorAlign(T x, T align)
{
    return align == 0 ? 0 : x / align * align;
}

inline UpdateOeTokenTableTilingData calc_tiling_data(
    int64_t batch_size, int64_t max_context_len, int64_t ignore_token_num)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    uint32_t aivNum = ascendcPlatform->GetCoreNumAiv();
    uint64_t ubSize = 0;
    ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    TORCH_CHECK(aivNum > 0, "aivNum must be positive.");
    TORCH_CHECK(ubSize > 0, "ubSize must be positive.");

    auto blockFactor = CeilDiv(static_cast<uint32_t>(batch_size), aivNum);
    auto usedCoreNum = CeilDiv(static_cast<uint32_t>(batch_size), blockFactor);
    blockFactor = CeilDiv(static_cast<uint32_t>(batch_size), usedCoreNum);
    auto tailBlockFactor = batch_size % usedCoreNum == 0
                               ? blockFactor
                               : batch_size % blockFactor;

    int64_t ignoreUbSize = CeilAlign(
        static_cast<int64_t>(ignore_token_num * sizeof(int32_t)), BLOCK_SIZE);
    int64_t reqLenUbSize = CeilAlign(
        static_cast<int64_t>(batch_size * sizeof(int32_t)), BLOCK_SIZE);
    int64_t reserveUbSize = static_cast<int64_t>(ubSize) - ignoreUbSize - reqLenUbSize;
    TORCH_CHECK(reserveUbSize > 0,
        "batch_size/ignore_token_num too large to fit static UB slots.");
    // 4: the compute path needs 4 tensors (tokenQue / oeTokenTableQue / mask / minus)
    int32_t ubFactor =
        FloorAlign(reserveUbSize / (DOUBLE_BUFEER * 4), BLOCK_SIZE) / sizeof(int32_t);

    UpdateOeTokenTableTilingData tl{};
    tl.usedCoreNum = static_cast<uint32_t>(usedCoreNum);
    tl.blockFactor = static_cast<uint32_t>(blockFactor);
    tl.tailBlockFactor = static_cast<uint32_t>(tailBlockFactor);
    tl.ubFactor = static_cast<uint32_t>(ubFactor);
    tl.batchSize = static_cast<uint32_t>(batch_size);
    tl.maxContextLen = static_cast<uint32_t>(max_context_len);
    tl.ignoreTokenNum = static_cast<uint32_t>(ignore_token_num);
    return tl;
}

} // namespace UpdateOeTokenTable
} // namespace flash

#endif // UPDATE_OE_TOKEN_TABLE_TILING_H
