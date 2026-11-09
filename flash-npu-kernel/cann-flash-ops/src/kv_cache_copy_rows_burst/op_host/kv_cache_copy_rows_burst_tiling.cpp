#include "kv_cache_copy_rows_burst_tiling.h"
#include <cstdint>
#include <limits>
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
static ge::graphStatus TilingKvCacheCopyRowsBurst(gert::TilingContext* context)
{
    // Use the existing GMM TilingContext dynamic-input API, not the
    // InferShapeContext-only flattened input-count convenience interface.
    uint32_t layers = 0;
    while (layers <= 38 && context->GetDynamicInputShape(0, layers) != nullptr) { ++layers; }
    if (layers == 0 || layers > 38) { return ge::GRAPH_FAILED; }
    int64_t rows = 0;
    int64_t columns = 0;
    uint32_t rowBytes = 0;
    ge::DataType cacheType = ge::DT_UNDEFINED;
    for (uint32_t i = 0; i < layers; ++i) {
        auto shape = context->GetDynamicInputShape(0, i);
        auto desc = context->GetInputDesc(i);
        if (shape == nullptr || desc == nullptr) {
            return ge::GRAPH_FAILED;
        }
        const auto& storage = shape->GetStorageShape();
        if (storage.GetDimNum() != 2 || storage.GetDim(0) < 1 ||
            storage.GetDim(0) > std::numeric_limits<int32_t>::max()) {
            return ge::GRAPH_FAILED;
        }
        if (i == 0) {
            rows = storage.GetDim(0);columns = storage.GetDim(1);cacheType = desc->GetDataType();
            if (cacheType == ge::DT_INT8 && columns == 672) { rowBytes = 672; }
            else if (cacheType == ge::DT_BF16 && (columns == 512 || columns == 64)) {
                rowBytes = static_cast<uint32_t>(columns) * 2U;
            } else { return ge::GRAPH_FAILED; }
        }
        if (storage.GetDim(0) != rows || storage.GetDim(1) != columns || desc->GetDataType() != cacheType) {
            return ge::GRAPH_FAILED;
        }
    }
    auto src = context->GetInputShape(layers);
    auto dst = context->GetInputShape(layers + 1);
    auto srcDesc = context->GetInputDesc(layers);
    auto dstDesc = context->GetInputDesc(layers + 1);
    if (src == nullptr || dst == nullptr || srcDesc == nullptr || dstDesc == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const auto& ss = src->GetStorageShape();
    const auto& ds = dst->GetStorageShape();
    const auto dtype = srcDesc->GetDataType();
    if (ss.GetDimNum() != 1 || ds.GetDimNum() != 1 || ss.GetDim(0) != ds.GetDim(0) ||
        ss.GetDim(0) < 0 || ss.GetDim(0) > (cacheType == ge::DT_BF16 ? 128 : 256) || dstDesc->GetDataType() != dtype ||
        (dtype != ge::DT_INT32 && dtype != ge::DT_INT64)) { return ge::GRAPH_FAILED; }
    const uint32_t count = static_cast<uint32_t>(ss.GetDim(0));
    const uint32_t indexBytes = ((count * (dtype == ge::DT_INT64 ? 8U : 4U) + 31U) / 32U) * 32U;
    auto platformInfo = context->GetPlatformInfo();
    if (platformInfo == nullptr) { return ge::GRAPH_FAILED; }
    platform_ascendc::PlatformAscendC platform(platformInfo);
    uint64_t ubBytes = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubBytes);
    if (platform.GetCoreNumAiv() < 4 || uint64_t(count) * rowBytes + 2U * indexBytes + 4096U > ubBytes) {
        return ge::GRAPH_FAILED;
    }
    auto workspace = context->GetWorkspaceSizes(1);
    auto raw = context->GetRawTilingData();
    if (workspace == nullptr || raw == nullptr) { return ge::GRAPH_FAILED; }
    KvCacheCopyRowsBurstTilingData data;
    data.set_layerCount(layers);data.set_totalRows(static_cast<uint32_t>(rows));
    data.set_rowCount(count);data.set_rowBytes(rowBytes);data.set_indexBufferBytes(indexBytes);
    if (data.GetDataSize() > raw->GetCapacity()) { return ge::GRAPH_FAILED; }
    context->SetBlockDim(4);
    context->SetTilingKey(dtype == ge::DT_INT64 ? 2 : 1);
    workspace[0] = 0;
    data.SaveToBuffer(raw->GetData(), raw->GetCapacity());raw->SetDataSize(data.GetDataSize());
    return ge::GRAPH_SUCCESS;
}
static ge::graphStatus PrepareKvCacheCopyRowsBurst(gert::TilingParseContext*) { return ge::GRAPH_SUCCESS; }
IMPL_OP_OPTILING(KvCacheCopyRowsBurst).Tiling(TilingKvCacheCopyRowsBurst)
    .TilingParse<KvCacheCopyRowsBurstCompileInfo>(PrepareKvCacheCopyRowsBurst);
} // namespace optiling
