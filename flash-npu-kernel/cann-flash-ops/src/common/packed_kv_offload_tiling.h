#pragma once
// Host-only shape and resource helpers for byte-exact packed KV movement.
#include <algorithm>
#include <initializer_list>
#include "exe_graph/runtime/tiling_context.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "error/ops_error.h"

#define PACKED_REQUIRE(condition, message) \
    do { if (!(condition)) { OPS_LOG_E(context->GetNodeName(), message); return ge::GRAPH_FAILED; } } while (0)

namespace PackedKvOffload {
constexpr uint32_t ROW_BYTES = 672;
constexpr uint32_t PAGE = 128;
constexpr uint32_t HOT = 4096;
constexpr uint32_t TOPK = 2048;
inline const gert::Shape* Shape(gert::TilingContext* context, size_t index)
{
    const auto* value = context->GetInputShape(index);
    return value ? &value->GetStorageShape() : nullptr;
}
inline bool Is(const gert::Shape* shape, std::initializer_list<int64_t> dims)
{
    if (!shape || shape->GetDimNum() != dims.size()) return false;
    size_t i = 0;
    for (auto dim : dims) { if (dim >= 0 && shape->GetDim(i) != dim) return false; ++i; }
    return true;
}
inline uint32_t Cores(gert::TilingContext* context, uint32_t tasks)
{
    platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
    return std::max(1U, std::min(tasks, static_cast<uint32_t>(platform.GetCoreNumAiv())));
}
template<class T>
inline ge::graphStatus Finish(gert::TilingContext* context, T& data, uint32_t cores, size_t workspace)
{
    auto* raw = context->GetRawTilingData();
    auto* sizes = context->GetWorkspaceSizes(1);
    PACKED_REQUIRE(raw && sizes && raw->GetCapacity() >= data.GetDataSize(), "Missing packed KV tiling/workspace storage");
    context->SetBlockDim(cores);
    context->SetTilingKey(1);
    sizes[0] = workspace;
    data.SaveToBuffer(raw->GetData(), raw->GetCapacity());
    raw->SetDataSize(data.GetDataSize());
    return ge::GRAPH_SUCCESS;
}
}
