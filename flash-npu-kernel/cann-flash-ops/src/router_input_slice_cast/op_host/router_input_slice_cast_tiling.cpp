#include "router_input_slice_cast_tiling.h"

#include <algorithm>
#include <cstdint>
#include <limits>

#include "graph/utils/type_utils.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
namespace {
constexpr uint64_t RESERVED_UB_BYTES = 8192;
constexpr uint64_t INPUT_ELEMENT_BYTES = 2;
constexpr uint64_t OUTPUT_ELEMENT_BYTES = 4;
constexpr uint32_t ALIGN_ELEMENTS = 16;
} // namespace

static ge::graphStatus TilingRouterInputSliceCast(gert::TilingContext *context)
{
    const auto *inputTensor = context->GetInputShape(0);
    const auto *inputDesc = context->GetInputDesc(0);
    const auto *attrs = context->GetAttrs();
    if (inputTensor == nullptr || inputDesc == nullptr || attrs == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const auto &input = inputTensor->GetStorageShape();
    if (input.GetDimNum() != 2) {
        return ge::GRAPH_FAILED;
    }
    const int64_t rows = input.GetDim(0);
    const int64_t inputCols = input.GetDim(1);
    const auto *startAttr = attrs->GetAttrPointer<int64_t>(0);
    const auto *lengthAttr = attrs->GetAttrPointer<int64_t>(1);
    if (startAttr == nullptr || lengthAttr == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const int64_t start = *startAttr;
    const int64_t length = *lengthAttr;
    constexpr uint64_t U32_MAX = std::numeric_limits<uint32_t>::max();
    if (rows <= 0 || static_cast<uint64_t>(rows) > U32_MAX ||
        inputCols <= 0 || static_cast<uint64_t>(inputCols) > U32_MAX ||
        start < 0 || length <= 0 || start > inputCols - length ||
        start % ALIGN_ELEMENTS != 0 || inputCols % ALIGN_ELEMENTS != 0 ||
        length % ALIGN_ELEMENTS != 0) {
        return ge::GRAPH_FAILED;
    }

    const auto dtype = inputDesc->GetDataType();
    if (dtype != ge::DT_FLOAT16 && dtype != ge::DT_BF16) {
        return ge::GRAPH_FAILED;
    }

    auto *platformInfo = context->GetPlatformInfo();
    if (platformInfo == nullptr) {
        return ge::GRAPH_FAILED;
    }
    platform_ascendc::PlatformAscendC platform(platformInfo);
    uint64_t ubBytes = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubBytes);
    const uint32_t availableCores = platform.GetCoreNumAiv();
    if (availableCores == 0) {
        return ge::GRAPH_FAILED;
    }
    // Keep every core's rows contiguous and equally sized.  For example,
    // rows=64 on 48 AIV cores uses 32 cores x 2 rows.  Besides avoiding an
    // imbalanced second wave, this removes division/modulo from the kernel.
    const uint32_t rowsPerCore = static_cast<uint32_t>(
        (static_cast<uint64_t>(rows) + availableCores - 1) / availableCores);
    const uint32_t usedCores = static_cast<uint32_t>(
        (static_cast<uint64_t>(rows) + rowsPerCore - 1) / rowsPerCore);
    if (usedCores == 0) {
        return ge::GRAPH_FAILED;
    }
    const uint64_t requiredUb = static_cast<uint64_t>(rowsPerCore) *
        static_cast<uint64_t>(length) *
        (INPUT_ELEMENT_BYTES + OUTPUT_ELEMENT_BYTES);
    if (ubBytes <= RESERVED_UB_BYTES ||
        requiredUb > ubBytes - RESERVED_UB_BYTES) {
        return ge::GRAPH_FAILED;
    }

    const uint64_t blockLength =
        static_cast<uint64_t>(length) * INPUT_ELEMENT_BYTES / 32;
    const uint64_t sourceStride =
        static_cast<uint64_t>(inputCols - length) * INPUT_ELEMENT_BYTES / 32;
    if (blockLength > std::numeric_limits<uint16_t>::max() ||
        sourceStride > std::numeric_limits<uint16_t>::max() ||
        rowsPerCore > std::numeric_limits<uint16_t>::max()) {
        return ge::GRAPH_FAILED;
    }

    auto *workspaceSizes = context->GetWorkspaceSizes(1);
    auto *rawTilingData = context->GetRawTilingData();
    if (workspaceSizes == nullptr || rawTilingData == nullptr) {
        return ge::GRAPH_FAILED;
    }

    RouterInputSliceCastTilingData data;
    data.set_rows(static_cast<uint32_t>(rows));
    data.set_inputCols(static_cast<uint32_t>(inputCols));
    data.set_sliceStart(static_cast<uint32_t>(start));
    data.set_sliceLength(static_cast<uint32_t>(length));
    data.set_usedCores(usedCores);
    data.set_rowsPerCore(rowsPerCore);
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

static ge::graphStatus PrepareRouterInputSliceCast(gert::TilingParseContext *)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(RouterInputSliceCast)
    .Tiling(TilingRouterInputSliceCast)
    .TilingParse<RouterInputSliceCastCompileInfo>(PrepareRouterInputSliceCast);

} // namespace optiling
