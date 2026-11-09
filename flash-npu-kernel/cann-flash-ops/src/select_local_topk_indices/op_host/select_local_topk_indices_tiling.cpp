#include "select_local_topk_indices_tiling.h"

#include <algorithm>
#include <cstdint>
#include <limits>

#include "tiling/platform/platform_ascendc.h"

namespace optiling {
ge::graphStatus TilingSelectLocalTopkIndices(gert::TilingContext *context)
{
    const auto &local = context->GetInputShape(0)->GetStorageShape();
    const auto &position = context->GetInputShape(1)->GetStorageShape();
    if (local.GetDimNum() != 3 || position.GetDimNum() != 2 ||
        local.GetDim(0) != position.GetDim(0) || local.GetDim(1) != 1) {
        return ge::GRAPH_FAILED;
    }

    const int64_t tokens = position.GetDim(0);
    const int64_t localCount = local.GetDim(2);
    const int64_t topkCount = position.GetDim(1);
    const int64_t kvpRank =
        *context->GetAttrs()->GetAttrPointer<int64_t>(0);
    // The fused DCP path has a fixed 2048-wide output. Keeping both UB rows
    // 32-byte aligned also lets the kernel use one vector/data-copy path.
    if (tokens <= 0 || localCount <= 0 || localCount > 2048 ||
        topkCount != 2048 || localCount % 8 != 0 || kvpRank < 0 ||
        kvpRank > std::numeric_limits<int32_t>::max() / localCount) {
        return ge::GRAPH_FAILED;
    }

    platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
    const uint32_t availableCores = platform.GetCoreNumAiv();
    const uint32_t usedCores = std::min<uint32_t>(
        availableCores, static_cast<uint32_t>(tokens));

    SelectLocalTopkIndicesTilingData data;
    data.set_totalTokens(static_cast<uint32_t>(tokens));
    data.set_localCount(static_cast<uint32_t>(localCount));
    data.set_topkCount(static_cast<uint32_t>(topkCount));
    data.set_usedCores(usedCores);
    data.set_rankStart(static_cast<int32_t>(kvpRank * localCount));

    context->SetBlockDim(usedCores);
    context->SetTilingKey(1);
    context->SetScheduleMode(1);
    context->GetWorkspaceSizes(1)[0] = platform.GetLibApiWorkSpaceSize();
    data.SaveToBuffer(context->GetRawTilingData()->GetData(),
                      context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(data.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PrepareSelectLocalTopkIndices(gert::TilingParseContext *)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(SelectLocalTopkIndices)
    .Tiling(TilingSelectLocalTopkIndices)
    .TilingParse<SelectLocalTopkIndicesCompileInfo>(
        PrepareSelectLocalTopkIndices);
} // namespace optiling
