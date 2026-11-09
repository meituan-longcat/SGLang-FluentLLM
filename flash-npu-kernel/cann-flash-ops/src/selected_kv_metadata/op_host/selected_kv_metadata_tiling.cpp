#include "selected_kv_metadata_tiling.h"
#include "tiling/platform/platform_ascendc.h"
namespace optiling {
static ge::graphStatus Tiling(gert::TilingContext *c) {
    const auto &q = c->GetInputShape(0)->GetStorageShape();
    const auto &p = c->GetInputShape(1)->GetStorageShape();
    auto a = c->GetAttrs();
    uint32_t world = *a->GetInt(0), local = *a->GetInt(1), reserved = *a->GetInt(2), phase = *a->GetInt(3);
    platform_ascendc::PlatformAscendC platform(c->GetPlatformInfo());
    uint32_t cores = world * 6, span = ((local + 5) / 6 + 255) / 256 * 256;
    if (q.GetDimNum() != 2 || q.GetDim(1) != 2048 || q.GetDim(0) <= 0 || p.GetDimNum() != 2 || p.GetDim(1) > 16384 ||
        p.GetDim(1) <= 0 || world == 0 || (world & (world - 1)) || cores > platform.GetCoreNumAiv() || span > 524288 ||
        phase > 3)
        return ge::GRAPH_FAILED;
    // Include all explicit TBuf allocations and leave 8 KiB for alignment/API headroom.
    uint64_t ubBytes = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubBytes);
    uint64_t pageBytes = phase == 3 ? 4096 * 32 : ((p.GetDim(1) + 7) / 8) * 32;
    uint64_t required = 6 * 2048 * 4 + 128 + pageBytes + ((phase == 1) ? span / 8 : 32) + 2 * (2048 / 8) +
                        cores * 32 + (phase == 0 ? (2048 + world * 8 + 8) * 4 : 32);
    if (required + 8192 > ubBytes)
        return ge::GRAPH_FAILED;
    SelectedKvMetadataTilingData t;
    t.set_requests(p.GetDim(0));
    t.set_queries(q.GetDim(0));
    t.set_pages(p.GetDim(1));
    t.set_world(world);
    t.set_localRows(local);
    t.set_reserved(reserved);
    t.set_phase(phase);
    t.set_cores(cores);
    t.set_span(span);
    c->SetBlockDim(cores);
    c->SetTilingKey(1);
    auto *r = c->GetRawTilingData();
    t.SaveToBuffer(r->GetData(), r->GetCapacity());
    r->SetDataSize(t.GetDataSize());
    c->GetWorkspaceSizes(1)[0] = platform.GetLibApiWorkSpaceSize();
    return ge::GRAPH_SUCCESS;
}
IMPL_OP_OPTILING(SelectedKvMetadata).Tiling(Tiling);
} // namespace optiling
