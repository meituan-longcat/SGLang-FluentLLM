#include "kv_cache_copy_rows_tiling.h"

#include <cstdint>
#include <limits>

#include "graph/utils/type_utils.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
namespace {
constexpr uint64_t BLOCK_BYTES = 32;
constexpr uint64_t RESERVED_UB_BYTES = 4096;
constexpr uint32_t MAX_COPY_ROWS = 256;

constexpr uint64_t AlignUp(uint64_t value, uint64_t alignment)
{
    return (value + alignment - 1) / alignment * alignment;
}
} // namespace

static ge::graphStatus TilingKvCacheCopyRows(gert::TilingContext* context)
{
    auto cacheTensor = context->GetInputShape(0);
    auto srcTensor = context->GetInputShape(1);
    auto dstTensor = context->GetInputShape(2);
    auto cacheDesc = context->GetInputDesc(0);
    auto srcDesc = context->GetInputDesc(1);
    auto dstDesc = context->GetInputDesc(2);
    if (cacheTensor == nullptr || srcTensor == nullptr || dstTensor == nullptr || cacheDesc == nullptr ||
        srcDesc == nullptr || dstDesc == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const auto& cache = cacheTensor->GetStorageShape();
    const auto& src = srcTensor->GetStorageShape();
    const auto& dst = dstTensor->GetStorageShape();
    if (cache.GetDimNum() != 2 || src.GetShapeSize() != dst.GetShapeSize()) {
        return ge::GRAPH_FAILED;
    }

    const int64_t totalRows = cache.GetDim(0);
    const int64_t rowWidth = cache.GetDim(1);
    const int64_t rowCount = src.GetShapeSize();
    constexpr uint64_t UINT32_MAX_VALUE = std::numeric_limits<uint32_t>::max();
    if (totalRows <= 0 || static_cast<uint64_t>(totalRows) > UINT32_MAX_VALUE || rowWidth <= 0 ||
        static_cast<uint64_t>(rowWidth) > UINT32_MAX_VALUE || rowCount <= 0 || rowCount > MAX_COPY_ROWS) {
        return ge::GRAPH_FAILED;
    }

    const auto cacheType = cacheDesc->GetDataType();
    const auto srcType = srcDesc->GetDataType();
    const auto dstType = dstDesc->GetDataType();
    if ((cacheType != ge::DT_BF16 && cacheType != ge::DT_FLOAT16 && cacheType != ge::DT_INT8) ||
        (srcType != ge::DT_INT32 && srcType != ge::DT_INT64) || dstType != srcType) {
        return ge::GRAPH_FAILED;
    }

    const int64_t dtypeSize = ge::GetSizeByDataType(cacheType);
    if (dtypeSize <= 0) {
        return ge::GRAPH_FAILED;
    }
    const uint64_t rowBytes = static_cast<uint64_t>(rowWidth) * static_cast<uint64_t>(dtypeSize);
    if (rowBytes > UINT32_MAX_VALUE || (rowBytes % BLOCK_BYTES != 0 && cacheType != ge::DT_INT8)) {
        return ge::GRAPH_FAILED;
    }
    const bool needsByteExactCopy = rowBytes % BLOCK_BYTES != 0;
    const uint64_t snapshotStrideBytes = needsByteExactCopy ? AlignUp(rowBytes, BLOCK_BYTES) : rowBytes;
    const uint64_t snapshotBytes = static_cast<uint64_t>(rowCount) * snapshotStrideBytes;
    if (snapshotBytes > UINT32_MAX_VALUE) {
        return ge::GRAPH_FAILED;
    }

    auto platformInfo = context->GetPlatformInfo();
    if (platformInfo == nullptr) {
        return ge::GRAPH_FAILED;
    }
    platform_ascendc::PlatformAscendC platform(platformInfo);
    uint64_t ubBytes = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubBytes);
    if (ubBytes <= RESERVED_UB_BYTES || snapshotBytes > ubBytes - RESERVED_UB_BYTES) {
        return ge::GRAPH_FAILED;
    }

    auto workspaceSizes = context->GetWorkspaceSizes(1);
    auto rawTilingData = context->GetRawTilingData();
    if (workspaceSizes == nullptr || rawTilingData == nullptr) {
        return ge::GRAPH_FAILED;
    }

    KvCacheCopyRowsTilingData data;
    data.set_rowCount(static_cast<uint32_t>(rowCount));
    data.set_rowWidth(static_cast<uint32_t>(rowWidth));
    data.set_totalRows(static_cast<uint32_t>(totalRows));
    data.set_snapshotBytes(static_cast<uint32_t>(snapshotBytes));
    if (data.GetDataSize() > rawTilingData->GetCapacity()) {
        return ge::GRAPH_FAILED;
    }

    context->SetBlockDim(1);
    context->SetTilingKey(needsByteExactCopy ? 2 : 1);
    workspaceSizes[0] = platform.GetLibApiWorkSpaceSize();
    data.SaveToBuffer(rawTilingData->GetData(), rawTilingData->GetCapacity());
    rawTilingData->SetDataSize(data.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus PrepareKvCacheCopyRows(gert::TilingParseContext*)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(KvCacheCopyRows)
    .Tiling(TilingKvCacheCopyRows)
    .TilingParse<KvCacheCopyRowsCompileInfo>(PrepareKvCacheCopyRows);

} // namespace optiling
