#include "moe_expert_id_map_tiling.h"

#include <cstdint>
#include <limits>

#include "error/ops_error.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
namespace {
constexpr uint64_t RESERVED_UB_BYTES = 8192;
constexpr uint64_t ELEMENT_BYTES = sizeof(int32_t);
constexpr uint64_t ALIGN_ELEMENTS = 32 / ELEMENT_BYTES;

uint64_t AlignElements(uint64_t value)
{
    return (value + ALIGN_ELEMENTS - 1) / ALIGN_ELEMENTS * ALIGN_ELEMENTS;
}
} // namespace

static ge::graphStatus TilingMoeExpertIdMap(gert::TilingContext *context)
{
    if (context == nullptr) {
        OPS_LOG_E("MoeExpertIdMap", "tiling context is null.");
        return ge::GRAPH_FAILED;
    }
    const char *nodeName = context->GetNodeName();
    const auto *idsTensor = context->GetInputShape(0);
    const auto *mapTensor = context->GetInputShape(1);
    const auto *idsDesc = context->GetInputDesc(0);
    const auto *mapDesc = context->GetInputDesc(1);
    if (idsTensor == nullptr || mapTensor == nullptr ||
        idsDesc == nullptr || mapDesc == nullptr) {
        OPS_LOG_E(nodeName, "logical_ids or expert_map shape/descriptor is null.");
        return ge::GRAPH_FAILED;
    }

    const auto &idsShape = idsTensor->GetStorageShape();
    const auto &mapShape = mapTensor->GetStorageShape();
    const int64_t idCount = idsShape.GetShapeSize();
    const int64_t mapSize = mapShape.GetShapeSize();
    constexpr uint64_t U32_MAX = std::numeric_limits<uint32_t>::max();
    if (idCount < 0) {
        OPS_LOG_E(nodeName, "logical_ids shape must be known at tiling time.");
        return ge::GRAPH_FAILED;
    }
    if (mapShape.GetDimNum() != 1 || mapSize <= 0) {
        OPS_LOG_E(nodeName, "expert_map must be a non-empty 1-D tensor, but rank is %zu and size is %ld.",
                  mapShape.GetDimNum(), mapSize);
        return ge::GRAPH_FAILED;
    }
    if (static_cast<uint64_t>(idCount) > U32_MAX ||
        static_cast<uint64_t>(mapSize) > U32_MAX) {
        OPS_LOG_E(nodeName, "logical_ids count and expert_map size must fit uint32, but got %ld and %ld.",
                  idCount, mapSize);
        return ge::GRAPH_FAILED;
    }
    if (idsDesc->GetDataType() != ge::DT_INT32 ||
        mapDesc->GetDataType() != ge::DT_INT32) {
        OPS_LOG_E(nodeName, "logical_ids and expert_map must both use INT32.");
        return ge::GRAPH_FAILED;
    }

    const uint64_t alignedIdCount = AlignElements(
        idCount == 0 ? ALIGN_ELEMENTS : static_cast<uint64_t>(idCount));
    const uint64_t alignedMapSize = AlignElements(static_cast<uint64_t>(mapSize));

    auto *platformInfo = context->GetPlatformInfo();
    if (platformInfo == nullptr) {
        OPS_LOG_E(nodeName, "platform info is null.");
        return ge::GRAPH_FAILED;
    }
    platform_ascendc::PlatformAscendC platform(platformInfo);
    uint64_t ubBytes = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubBytes);
    const uint64_t requiredUb =
        (alignedMapSize + alignedIdCount * 2) * ELEMENT_BYTES;
    if (ubBytes <= RESERVED_UB_BYTES ||
        requiredUb > ubBytes - RESERVED_UB_BYTES ||
        alignedIdCount > U32_MAX || alignedMapSize > U32_MAX) {
        OPS_LOG_E(nodeName,
                  "input does not fit the single-core UB design: required=%lu bytes, available=%lu bytes, "
                  "alignedIdCount=%lu, alignedMapSize=%lu.",
                  requiredUb, ubBytes > RESERVED_UB_BYTES ? ubBytes - RESERVED_UB_BYTES : 0,
                  alignedIdCount, alignedMapSize);
        return ge::GRAPH_FAILED;
    }

    auto *workspaceSizes = context->GetWorkspaceSizes(1);
    auto *rawTilingData = context->GetRawTilingData();
    if (workspaceSizes == nullptr || rawTilingData == nullptr) {
        OPS_LOG_E(nodeName, "workspace sizes or raw tiling data is null.");
        return ge::GRAPH_FAILED;
    }

    MoeExpertIdMapTilingData data;
    data.set_idCount(static_cast<uint32_t>(idCount));
    data.set_mapSize(static_cast<uint32_t>(mapSize));
    data.set_alignedIdCount(static_cast<uint32_t>(alignedIdCount));
    data.set_alignedMapSize(static_cast<uint32_t>(alignedMapSize));
    if (data.GetDataSize() > rawTilingData->GetCapacity()) {
        OPS_LOG_E(nodeName, "tiling data size %zu exceeds capacity %zu.",
                  data.GetDataSize(), rawTilingData->GetCapacity());
        return ge::GRAPH_FAILED;
    }

    context->SetBlockDim(1);
    context->SetTilingKey(1);
    workspaceSizes[0] = platform.GetLibApiWorkSpaceSize();
    data.SaveToBuffer(rawTilingData->GetData(), rawTilingData->GetCapacity());
    rawTilingData->SetDataSize(data.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus PrepareMoeExpertIdMap(gert::TilingParseContext *)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(MoeExpertIdMap)
    .Tiling(TilingMoeExpertIdMap)
    .TilingParse<MoeExpertIdMapCompileInfo>(PrepareMoeExpertIdMap);

} // namespace optiling
