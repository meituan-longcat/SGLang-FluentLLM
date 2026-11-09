#include "selected_kv_materialize_tiling.h"

#include <algorithm>
#include <cstdint>
#include <limits>

#include "tiling/platform/platform_ascendc.h"

namespace optiling {

namespace {
constexpr int64_t PACKED_PAYLOAD_BYTES = 656;
constexpr int64_t PACKED_ALIGNED_ROW_BYTES = 672;
} // namespace

static ge::graphStatus TilingSelectedKvMaterialize(gert::TilingContext *context) {
    const auto *inputTensor = context->GetInputShape(0);
    const auto *inputDesc = context->GetInputDesc(0);
    if (inputTensor == nullptr || inputDesc == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const auto &input = inputTensor->GetStorageShape();
    if (input.GetDimNum() != 3 || inputDesc->GetDataType() != ge::DT_INT8) {
        return ge::GRAPH_FAILED;
    }
    const int64_t packedRowBytes = input.GetDim(2);
    if (input.GetDim(1) != 1 ||
        (packedRowBytes != PACKED_PAYLOAD_BYTES && packedRowBytes != PACKED_ALIGNED_ROW_BYTES)) {
        return ge::GRAPH_FAILED;
    }
    const int64_t rows = input.GetDim(0);
    if (rows < 0 || static_cast<uint64_t>(rows) > std::numeric_limits<uint32_t>::max()) {
        return ge::GRAPH_FAILED;
    }

    auto *platformInfo = context->GetPlatformInfo();
    if (platformInfo == nullptr) {
        return ge::GRAPH_FAILED;
    }
    platform_ascendc::PlatformAscendC platform(platformInfo);
    const uint32_t availableCores = platform.GetCoreNumAiv();
    if (availableCores == 0) {
        return ge::GRAPH_FAILED;
    }
    const int64_t extendRows = context->GetInputShape(1)->GetStorageShape().GetDim(0);
    const int64_t total = ((rows + extendRows + 127) / 128) * 128;
    if (total == 0)
        return ge::GRAPH_FAILED;
    const uint32_t usedCores = static_cast<uint32_t>(std::min<int64_t>(total, availableCores));
    const uint32_t rowsPerCore = static_cast<uint32_t>((total + usedCores - 1) / usedCores);

    auto *workspaceSizes = context->GetWorkspaceSizes(1);
    auto *rawTilingData = context->GetRawTilingData();
    if (workspaceSizes == nullptr || rawTilingData == nullptr) {
        return ge::GRAPH_FAILED;
    }
    SelectedKvMaterializeTilingData data;
    data.set_rows(static_cast<uint32_t>(rows));
    data.set_extendRows(static_cast<uint32_t>(extendRows));
    data.set_usedCores(usedCores);
    data.set_rowsPerCore(rowsPerCore);
    data.set_packedRowBytes(static_cast<uint32_t>(packedRowBytes));
    if (data.GetDataSize() > rawTilingData->GetCapacity()) {
        return ge::GRAPH_FAILED;
    }
    context->SetBlockDim(usedCores);
    context->SetTilingKey(1);
    workspaceSizes[0] = platform.GetLibApiWorkSpaceSize();
    data.SaveToBuffer(rawTilingData->GetData(), rawTilingData->GetCapacity());
    rawTilingData->SetDataSize(data.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus PrepareSelectedKvMaterialize(gert::TilingParseContext *) { return ge::GRAPH_SUCCESS; }

IMPL_OP_OPTILING(SelectedKvMaterialize)
    .Tiling(TilingSelectedKvMaterialize)
    .TilingParse<SelectedKvMaterializeCompileInfo>(PrepareSelectedKvMaterialize);

} // namespace optiling
