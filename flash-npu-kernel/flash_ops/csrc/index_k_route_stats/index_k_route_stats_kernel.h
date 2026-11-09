#pragma once
#include "kernel_operator.h"

namespace flash::IndexKRouteStats {
using namespace AscendC;
constexpr uint32_t TILE = 512;

struct Tiling {
    int64_t n;
    int64_t reserved;
    uint32_t groups;
    uint32_t cores;
    uint32_t stride;  // int64 elements, rounded to a complete 32-byte block
};

template <HardEvent event>
__aicore__ inline void Sync() {
    auto id = GetTPipePtr()->FetchEventID(event);
    SetFlag<event>(id);
    WaitFlag<event>(id);
}

// Per-core private histograms avoid atomics into the small group table.
// Only bounded UB tiles are allocated, independently of the input length.
__global__ __aicore__ __vector__ void partial_stats(
    GM_ADDR group_ptr, GM_ADDR length_ptr, GM_ADDR page_ptr,
    GM_ADDR partial_ptr, Tiling t) {
    TPipe pipe;
    TBuf<QuePosition::VECCALC> gb, lb, pb, hb;
    pipe.InitBuffer(gb, TILE * sizeof(int64_t));
    pipe.InitBuffer(lb, TILE * sizeof(int64_t));
    pipe.InitBuffer(pb, TILE * sizeof(int64_t));
    pipe.InitBuffer(hb, t.stride * sizeof(int64_t));
    auto gs = gb.Get<int64_t>();
    auto ls = lb.Get<int64_t>();
    auto ps = pb.Get<int64_t>();
    auto hist = hb.Get<int64_t>();
    GlobalTensor<int64_t> groups, lengths, pages, partial;
    groups.SetGlobalBuffer((__gm__ int64_t*)group_ptr);
    lengths.SetGlobalBuffer((__gm__ int64_t*)length_ptr);
    pages.SetGlobalBuffer((__gm__ int64_t*)page_ptr);
    partial.SetGlobalBuffer((__gm__ int64_t*)partial_ptr);
    for (uint32_t i = 0; i < t.stride; ++i) hist.SetValue(i, 0);
    const int64_t begin = t.n * GetBlockIdx() / t.cores;
    const int64_t end = t.n * (GetBlockIdx() + 1) / t.cores;
    for (int64_t off = begin; off < end; off += TILE) {
        const uint32_t size = end - off < TILE ? end - off : TILE;
        DataCopyExtParams copy{1, size * uint32_t(sizeof(int64_t)), 0, 0, 0};
        DataCopyPadExtParams<int64_t> pad{false, 0, 0, 0};
        DataCopyPad(gs, groups[off], copy, pad);
        DataCopyPad(ls, lengths[off], copy, pad);
        DataCopyPad(ps, pages[off], copy, pad);
        Sync<HardEvent::MTE2_S>();
        for (uint32_t i = 0; i < size; ++i) {
            const int64_t g = gs.GetValue(i);
            const int64_t len = ls.GetValue(i);
            if (ps.GetValue(i) < t.reserved || g < 0 || g >= t.groups || len < 0) {
                hist.SetValue(2 * t.groups, 1);
            }
            // Guard the UB index even for a malformed direct operator call.
            if (g >= 0 && g < t.groups && len >= 0) {
                hist.SetValue(g, hist.GetValue(g) + len);
                hist.SetValue(t.groups + g, hist.GetValue(t.groups + g) + 1);
            }
        }
        Sync<HardEvent::S_MTE2>();
    }
    Sync<HardEvent::S_MTE3>();
    const uint32_t written = t.cores == 1 ? 2 * t.groups + 1 : t.stride;
    DataCopyExtParams copy{1, written * uint32_t(sizeof(int64_t)), 0, 0, 0};
    DataCopyPad(partial[GetBlockIdx() * t.stride], hist, copy);
}

// A second launch on the same stream provides the inter-core completion
// boundary. There is no spin barrier, shared atomic or host synchronization.
__global__ __aicore__ __vector__ void finish_stats(
    GM_ADDR partial_ptr, GM_ADDR output_ptr, Tiling t) {
    TPipe pipe;
    TBuf<QuePosition::VECCALC> pb, ob;
    // Each block owns a complete output cache line. Strided DMA gathers
    // that line from each partial histogram; no two blocks share a line.
    constexpr uint32_t LINE = 4;
    const uint32_t first = GetBlockIdx() * LINE;
    const uint32_t remaining = 2 * t.groups + 1 - first;
    const uint32_t size = remaining < LINE ? remaining : LINE;
    pipe.InitBuffer(pb, LINE * t.cores * sizeof(int64_t));
    pipe.InitBuffer(ob, LINE * sizeof(int64_t));
    auto parts = pb.Get<int64_t>();
    auto out = ob.Get<int64_t>();
    GlobalTensor<int64_t> partial, output;
    partial.SetGlobalBuffer((__gm__ int64_t*)partial_ptr);
    output.SetGlobalBuffer((__gm__ int64_t*)output_ptr);
    DataCopyExtParams in_copy{uint16_t(t.cores), size * uint32_t(sizeof(int64_t)),
                             (t.stride - size) * uint32_t(sizeof(int64_t)), 0, 0};
    DataCopyPadExtParams<int64_t> pad{false, 0, 0, 0};
    DataCopyPad(parts, partial[first], in_copy, pad);
    Sync<HardEvent::MTE2_S>();
    for (uint32_t i = 0; i < size; ++i) {
        int64_t sum = 0;
        for (uint32_t core = 0; core < t.cores; ++core) {
            sum += parts.GetValue(core * LINE + i);
        }
        out.SetValue(i, first + i == 2 * t.groups ? int64_t(sum != 0) : sum);
    }
    Sync<HardEvent::S_MTE3>();
    DataCopyExtParams out_copy{1, size * uint32_t(sizeof(int64_t)), 0, 0, 0};
    DataCopyPad(output[first], out, out_copy);
}
}  // namespace flash::IndexKRouteStats
