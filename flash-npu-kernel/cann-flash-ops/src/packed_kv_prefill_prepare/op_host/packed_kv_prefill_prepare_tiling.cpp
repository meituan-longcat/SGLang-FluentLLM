#include "packed_kv_prefill_prepare_tiling.h"

#include <algorithm>
#include <cstdint>
#include <limits>

#include "tiling/platform/platform_ascendc.h"

namespace optiling {
namespace {
constexpr int64_t PACKED_PAYLOAD_BYTES = 656;
constexpr int64_t PACKED_ALIGNED_ROW_BYTES = 672;

ge::graphStatus TilingPackedKvPrefillPrepare(gert::TilingContext *context)
{
    const auto *latentShape = context->GetInputShape(0);
    const auto *latentDesc = context->GetInputDesc(0);
    const auto *ownerShape = context->GetInputShape(4);
    const auto *ownerDesc = context->GetInputDesc(4);
    if (latentShape == nullptr || latentDesc == nullptr ||
        ownerShape == nullptr || ownerDesc == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const auto &latent = latentShape->GetStorageShape();
    const auto &owner = ownerShape->GetStorageShape();
    if (latent.GetDimNum() != 2 || latent.GetDim(1) != 576 ||
        latentDesc->GetDataType() != ge::DT_BF16 ||
        owner.GetDimNum() != 1 || ownerDesc->GetDataType() != ge::DT_INT64) {
        return ge::GRAPH_FAILED;
    }
    const int64_t rows = latent.GetDim(0);
    const int64_t owners = owner.GetDim(0);
    if (rows <= 0 || owners < 0 || rows > std::numeric_limits<uint32_t>::max() ||
        owners > std::numeric_limits<uint32_t>::max()) {
        return ge::GRAPH_FAILED;
    }

    auto *platformInfo = context->GetPlatformInfo();
    if (platformInfo == nullptr) {
        return ge::GRAPH_FAILED;
    }
    platform_ascendc::PlatformAscendC platform(platformInfo);
    const uint32_t available = platform.GetCoreNumAiv();
    if (available == 0) {
        return ge::GRAPH_FAILED;
    }
    const uint32_t work = static_cast<uint32_t>(std::max(rows, owners));
    const uint32_t cores = std::min(work, available);
    if (cores == 0) {
        return ge::GRAPH_FAILED;
    }

    auto *workspace = context->GetWorkspaceSizes(1);
    auto *raw = context->GetRawTilingData();
    if (workspace == nullptr || raw == nullptr) {
        return ge::GRAPH_FAILED;
    }
    PackedKvPrefillPrepareTilingData data;
    data.set_rows(static_cast<uint32_t>(rows));
    data.set_ownerRows(static_cast<uint32_t>(owners));
    data.set_rowsPerCore(static_cast<uint32_t>((rows + cores - 1) / cores));
    data.set_ownersPerCore(static_cast<uint32_t>((owners + cores - 1) / cores));
    const auto *attrs = context->GetAttrs();
    const float *epsilon = attrs == nullptr ? nullptr : attrs->GetAttrPointer<float>(0);
    const int64_t *packedRowBytes =
        attrs == nullptr ? nullptr : attrs->GetAttrPointer<int64_t>(1);
    if (epsilon == nullptr || packedRowBytes == nullptr ||
        (*packedRowBytes != PACKED_PAYLOAD_BYTES &&
         *packedRowBytes != PACKED_ALIGNED_ROW_BYTES)) {
        return ge::GRAPH_FAILED;
    }
    data.set_epsilon(*epsilon);
    data.set_packedRowBytes(static_cast<uint32_t>(*packedRowBytes));
    if (data.GetDataSize() > raw->GetCapacity()) {
        return ge::GRAPH_FAILED;
    }
    context->SetBlockDim(cores);
    context->SetTilingKey(1);
    workspace[0] = platform.GetLibApiWorkSpaceSize();
    data.SaveToBuffer(raw->GetData(), raw->GetCapacity());
    raw->SetDataSize(data.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PreparePackedKvPrefillPrepare(gert::TilingParseContext *)
{
    return ge::GRAPH_SUCCESS;
}
} // namespace

IMPL_OP_OPTILING(PackedKvPrefillPrepare)
    .Tiling(TilingPackedKvPrefillPrepare)
    .TilingParse<PackedKvPrefillPrepareCompileInfo>(PreparePackedKvPrefillPrepare);
} // namespace optiling
