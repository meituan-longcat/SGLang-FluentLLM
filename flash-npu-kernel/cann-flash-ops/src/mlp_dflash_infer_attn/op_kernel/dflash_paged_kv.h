#ifndef DFLASH_PAGED_KV_H
#define DFLASH_PAGED_KV_H

#include <cstdint>

namespace SplitFuse::DflashPaged {

#ifndef DFLASH_PAGED_DEVICE
#define DFLASH_PAGED_DEVICE __aicore__
#endif

constexpr uint32_t SPARSE_MODE = 15U;
constexpr uint32_t KV_TILE_SIZE = 512U;
constexpr int64_t TOKEN_UNLIMITED = 2147483647LL;

struct LogicalRange {
    uint32_t start = 0U;
    uint32_t end = 0U;
};

struct Plan {
    LogicalRange first;
    LogicalRange second;
    uint32_t firstTileCount = 0U;
    uint32_t secondTileCount = 0U;
    uint32_t totalTileCount = 0U;
};

struct Tile {
    uint32_t logicalStart = 0U;
    uint32_t tokenCount = 0U;
};

DFLASH_PAGED_DEVICE inline uint32_t Min(uint32_t lhs, uint32_t rhs)
{
    return lhs < rhs ? lhs : rhs;
}

DFLASH_PAGED_DEVICE inline uint32_t RangeTileCount(const LogicalRange &range)
{
    uint32_t length = range.end > range.start ? range.end - range.start : 0U;
    return (length + KV_TILE_SIZE - 1U) / KV_TILE_SIZE;
}

DFLASH_PAGED_DEVICE inline Plan BuildPlan(uint32_t qLength, uint32_t kvLength, int64_t preTokens, int64_t nextTokens)
{
    Plan plan;
    if (qLength == 0U || qLength > kvLength || preTokens < 0 || nextTokens < 0) {
        return plan;
    }

    uint32_t contextEnd = kvLength - qLength;
    uint32_t tailStart = 0U;
    if (preTokens < TOKEN_UNLIMITED) {
        uint32_t window = static_cast<uint32_t>(preTokens);
        tailStart = contextEnd > window ? contextEnd - window : 0U;
    }
    uint32_t sinkLength =
        nextTokens > static_cast<int64_t>(contextEnd) ? contextEnd : static_cast<uint32_t>(nextTokens);

    if (sinkLength == 0U) {
        plan.first = {tailStart, kvLength};
    } else if (tailStart <= sinkLength) {
        plan.first = {0U, kvLength};
    } else {
        plan.first = {0U, sinkLength};
        plan.second = {tailStart, kvLength};
    }

    plan.firstTileCount = RangeTileCount(plan.first);
    plan.secondTileCount = RangeTileCount(plan.second);
    plan.totalTileCount = plan.firstTileCount + plan.secondTileCount;
    return plan;
}

DFLASH_PAGED_DEVICE inline Tile GetTile(const Plan &plan, uint32_t tileOrdinal)
{
    LogicalRange range = plan.first;
    uint32_t localOrdinal = tileOrdinal;
    if (tileOrdinal >= plan.firstTileCount) {
        range = plan.second;
        localOrdinal = tileOrdinal - plan.firstTileCount;
    }
    uint32_t logicalStart = range.start + localOrdinal * KV_TILE_SIZE;
    uint32_t remaining = range.end > logicalStart ? range.end - logicalStart : 0U;
    return {logicalStart, Min(remaining, KV_TILE_SIZE)};
}

} // namespace SplitFuse::DflashPaged

#undef DFLASH_PAGED_DEVICE

#endif // DFLASH_PAGED_KV_H
