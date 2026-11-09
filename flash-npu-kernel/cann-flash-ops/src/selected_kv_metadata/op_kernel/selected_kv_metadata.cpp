#include "kernel_operator.h"
using namespace AscendC;

// Four ordered device phases split across the start/finish Torch interfaces. Each partition has
// one writer: duplicate marks need neither inter-core atomics nor locks.
class SelectedKvMetadataKernel {
    static constexpr uint32_t K = 2048, TILE_PAGES = 4096, OWNER_SHIFT = 22, OWNER_STRIDE = 1u << OWNER_SHIFT;
    GlobalTensor<int32_t> topk, pages, seq, prefixes, offsets;
    GlobalTensor<int32_t> keys, mapping, temporary, counts, unique, indices, logicalMap, routeCounts;
    TBuf<> aBuf, bBuf, cBuf, dBuf, pageBuf, bitBuf, loBuf, hiBuf, countBuf, eBuf, fBuf, lutBuf, routeBuf;
    LocalTensor<int32_t> a, b, c, d, e, f, pt, co, lut, routed;
    LocalTensor<uint32_t> bits;
    LocalTensor<uint8_t> lo, hi;
    uint32_t Q, P, W, L, R, phase, cores, span, core, ownerSpan, B;
    __aicore__ inline void Load(LocalTensor<int32_t> dst, GlobalTensor<int32_t> src, uint32_t n) {
        DataCopyPad(dst, src, DataCopyExtParams{1, n * 4, 0, 0, 0}, DataCopyPadExtParams<int32_t>{false, 0, 0, 0});
        PipeBarrier<PIPE_ALL>();
    }
    __aicore__ inline void Store(GlobalTensor<int32_t> dst, LocalTensor<int32_t> src, uint32_t n) {
        PipeBarrier<PIPE_ALL>();
        if (n)
            DataCopyPad(dst, src, DataCopyExtParams{1, n * 4, 0, 0, 0});
        PipeBarrier<PIPE_ALL>();
    }
    __aicore__ inline void SelectI32(LocalTensor<int32_t> out, LocalTensor<uint8_t> mask, LocalTensor<int32_t> yes,
                                     LocalTensor<int32_t> no) {
        Select(out.ReinterpretCast<float>(), mask, yes.ReinterpretCast<float>(), no.ReinterpretCast<float>(),
               SELMODE::VSEL_TENSOR_TENSOR_MODE, K);
        PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void InRange(int32_t start, int32_t end) {
        Cast(f.ReinterpretCast<float>(), a, RoundMode::CAST_ROUND, K);
        PipeBarrier<PIPE_V>();
        CompareScalar(lo, f.ReinterpretCast<float>(), static_cast<float>(start), CMPMODE::GE, K);
        CompareScalar(hi, f.ReinterpretCast<float>(), static_cast<float>(end), CMPMODE::LT, K);
        PipeBarrier<PIPE_V>();
        And(hi.ReinterpretCast<uint16_t>(), hi.ReinterpretCast<uint16_t>(), lo.ReinterpretCast<uint16_t>(), K / 16);
        PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void Lookup() {
        uint32_t shift = 0;
        while ((1u << shift) < W)
            ++shift;
        Duplicate(co, static_cast<int32_t>(0), 8);
        PipeBarrier<PIPE_ALL>();
        uint32_t work = (Q + cores - 1) / cores, start = core * work, end = start + work < Q ? start + work : Q;
        for (uint32_t q = start; q < end; ++q) {
            int32_t s = seq.GetValue(q), pre = prefixes.GetValue(s), off = offsets.GetValue(s);
            Load(a, topk[static_cast<uint64_t>(q) * K], K);
            Load(pt, pages[static_cast<uint64_t>(s) * P], P);
            InRange(0, pre);
            Duplicate(e, static_cast<int32_t>(0), K);
            PipeBarrier<PIPE_V>();
            SelectI32(b, hi, a, e); // Padding and extend must not access unallocated pages.
            ShiftRight(b, b, static_cast<int32_t>(7), K);
            PipeBarrier<PIPE_V>();
            Muls(b, b, static_cast<int32_t>(4), K);
            PipeBarrier<PIPE_V>();
            Gather(c, pt, b.ReinterpretCast<uint32_t>(), 0, K);
            PipeBarrier<PIPE_V>();
            ShiftRight(c, c, static_cast<int32_t>(7), K);
            PipeBarrier<PIPE_V>();
            Adds(c, c, -static_cast<int32_t>(R), K);
            PipeBarrier<PIPE_V>();
            ShiftRight(b, c, static_cast<int32_t>(shift), K);
            PipeBarrier<PIPE_V>();
            Muls(d, b, static_cast<int32_t>(W), K);
            PipeBarrier<PIPE_V>();
            Sub(c, c, d, K);
            PipeBarrier<PIPE_V>();
            Muls(c, c, static_cast<int32_t>(OWNER_STRIDE), K);
            Adds(b, b, static_cast<int32_t>(R), K);
            PipeBarrier<PIPE_V>();
            // Stripe physical pages across the six owner cores. Contiguous
            // allocations otherwise put nearly all short-prefix work on one.
            // Local page ids are <=24575; float quotient conversion is exact
            // enough here and CAST_FLOOR implements nonnegative /6.
            Cast(f.ReinterpretCast<float>(), b, RoundMode::CAST_ROUND, K);
            PipeBarrier<PIPE_V>();
            Muls(f.ReinterpretCast<float>(), f.ReinterpretCast<float>(), 1.0f / 6.0f, K);
            PipeBarrier<PIPE_V>();
            Cast(d, f.ReinterpretCast<float>(), RoundMode::CAST_FLOOR, K);
            PipeBarrier<PIPE_V>();
            Muls(e, d, static_cast<int32_t>(6), K);
            PipeBarrier<PIPE_V>();
            Sub(e, b, e, K);
            PipeBarrier<PIPE_V>();
            Muls(e, e, static_cast<int32_t>(span), K);
            Muls(d, d, static_cast<int32_t>(128), K);
            PipeBarrier<PIPE_V>();
            Add(b, e, d, K);
            PipeBarrier<PIPE_V>();
            Add(c, c, b, K);
            ShiftRight(b, a, static_cast<int32_t>(7), K);
            PipeBarrier<PIPE_V>();
            Muls(b, b, static_cast<int32_t>(128), K);
            PipeBarrier<PIPE_V>();
            Sub(b, a, b, K);
            PipeBarrier<PIPE_V>();
            Add(c, c, b, K);
            Muls(e, a, static_cast<int32_t>(-1), K);
            PipeBarrier<PIPE_V>();
            Adds(e, e, pre - off - 2, K);
            PipeBarrier<PIPE_V>();
            SelectI32(c, hi, c, e);
            Duplicate(e, static_cast<int32_t>(-1), K);
            PipeBarrier<PIPE_V>();
            SelectI32(d, lo, c, e);
            ShiftRight(f, d, static_cast<int32_t>(OWNER_SHIFT), K);
            PipeBarrier<PIPE_V>();
            uint32_t cursor = 8;
            Duplicate(routed, static_cast<int32_t>(0), K + W * 8 + 8);
            PipeBarrier<PIPE_ALL>();
            for (uint32_t owner = 0; owner < W; ++owner) {
                CompareScalar(lo, f, static_cast<int32_t>(owner), CMPMODE::EQ, K);
                PipeBarrier<PIPE_V>();
                uint64_t n = 0;
                GatherMask(routed[cursor].ReinterpretCast<float>(), d.ReinterpretCast<float>(),
                           lo.ReinterpretCast<uint32_t>(), true, K, GatherMaskParams{1, 1, 8, 0}, n);
                PipeBarrier<PIPE_ALL>();
                if (n) {
                    Adds(routed[cursor], routed[cursor], -static_cast<int32_t>(owner * (OWNER_STRIDE - ownerSpan)),
                         static_cast<uint32_t>(n));
                    PipeBarrier<PIPE_V>();
                }
                routed.SetValue(owner, static_cast<int32_t>(n));
                co.SetValue(owner, co.GetValue(owner) + static_cast<int32_t>(n));
                cursor += (static_cast<uint32_t>(n) + 7) / 8 * 8;
            }
            // One query at a time: retain the first compaction, with each owner
            // segment aligned to 32 bytes. The first 8 int32 entries are counts.
            Store(keys[static_cast<uint64_t>(q) * (K + W * 8 + 8)], routed, K + W * 8 + 8);
        }
        Store(routeCounts[core * 8], co, 8);
    }
    __aicore__ inline void Route() {
        // Each producer owns a fixed contiguous query block. Bucket starts are
        // local prefix sums: no cross-core atomics or global scan is necessary.
        // Lookup and Route therefore run in the same device launch.
        Load(co, routeCounts[core * 8], 8);
        uint32_t prefix = 0;
        for (uint32_t owner = 0; owner < W; ++owner) {
            co.SetValue(8 + owner, prefix);
            prefix += co.GetValue(owner);
        }
        uint32_t work = (Q + cores - 1) / cores, start = core * work, end = start + work < Q ? start + work : Q;
        for (uint32_t q = start; q < end; ++q) {
            // Counts and compacted owner segments share one aligned DMA.
            Load(routed, keys[static_cast<uint64_t>(q) * (K + W * 8 + 8)], K + W * 8 + 8);
            uint32_t offset = 8;
            for (uint32_t owner = 0; owner < W; ++owner) {
                uint32_t n = routed.GetValue(owner);
                if (!n)
                    continue;
                uint32_t cursor = co.GetValue(8 + owner);
                // All owner segments stay live in routed until this query is
                // drained. Queue disjoint writes, then synchronize once.
                DataCopyPad(indices[static_cast<uint64_t>(start) * K + cursor], routed[offset],
                            DataCopyExtParams{1, n * 4, 0, 0, 0});
                co.SetValue(8 + owner, cursor + n);
                offset += (n + 7) / 8 * 8;
            }
            PipeBarrier<PIPE_ALL>(); // Complete MTE3 before reusing routed.
        }
    }
    __aicore__ inline void Mark() {
        Duplicate(bits, static_cast<uint32_t>(0), span / 32);
        PipeBarrier<PIPE_ALL>();
        for (uint32_t i = 0; i < 32; ++i)
            lut.SetValue(i, static_cast<int32_t>(1u << i));
        PipeBarrier<PIPE_ALL>();
        const int32_t begin = core * span;
        Load(co, routeCounts, cores * 8);
        uint32_t owner = core / 6, work = (Q + cores - 1) / cores;
        uint32_t touchedBegin = span, touchedEnd = 0;
        for (uint32_t producer = 0; producer < cores; ++producer) {
            uint32_t bucket = co.GetValue(producer * 8 + owner), offset = 0;
            for (uint32_t o = 0; o < owner; ++o)
                offset += co.GetValue(producer * 8 + o);
            for (uint32_t j = 0; j < bucket; j += K) {
                uint32_t take = bucket - j < K ? bucket - j : K;
                if (take < K) {
                    Duplicate(a, static_cast<int32_t>(-1), K);
                    PipeBarrier<PIPE_ALL>();
                }
                Load(a, indices[static_cast<uint64_t>(producer) * work * K + offset + j], take);
                Adds(b, a, -begin, K);
                PipeBarrier<PIPE_V>();
                Cast(c.ReinterpretCast<float>(), b, RoundMode::CAST_ROUND, K);
                PipeBarrier<PIPE_V>();
                CompareScalar(lo, c.ReinterpretCast<float>(), 0.0f, CMPMODE::GE, K);
                CompareScalar(hi, c.ReinterpretCast<float>(), static_cast<float>(static_cast<int32_t>(span)),
                              CMPMODE::LT, K);
                PipeBarrier<PIPE_V>();
                And(lo.ReinterpretCast<uint16_t>(), lo.ReinterpretCast<uint16_t>(), hi.ReinterpretCast<uint16_t>(),
                    K / 16);
                PipeBarrier<PIPE_V>();
                uint64_t valid = 0;
                GatherMask(d.ReinterpretCast<float>(), b.ReinterpretCast<float>(), lo.ReinterpretCast<uint32_t>(), true,
                           K, GatherMaskParams{1, 1, 8, 0}, valid);
                PipeBarrier<PIPE_ALL>();
                if (!valid)
                    continue;
                // Query the existing bitmap in vector lanes before scalar updates.
                // Filter only bits committed by previous address tiles. Survivors may
                // share a word (or a bit) within this row: merge those with scalar OR
                // against the latest word, never scatter stale read-modify-write data.
                uint32_t lanes = static_cast<uint32_t>(valid);
                ShiftRight(b, d, static_cast<int32_t>(5), lanes);
                PipeBarrier<PIPE_V>();
                Muls(e, b, static_cast<int32_t>(32), lanes);
                PipeBarrier<PIPE_V>();
                Sub(e, d, e, lanes);
                Muls(b, b, static_cast<int32_t>(4), lanes);
                PipeBarrier<PIPE_V>();
                Muls(e, e, static_cast<int32_t>(4), lanes);
                Gather(c, bits.ReinterpretCast<int32_t>(), b.ReinterpretCast<uint32_t>(), 0, lanes);
                PipeBarrier<PIPE_V>();
                Gather(f, lut, e.ReinterpretCast<uint32_t>(), 0, lanes);
                PipeBarrier<PIPE_V>();
                And(f.ReinterpretCast<uint16_t>(), f.ReinterpretCast<uint16_t>(), c.ReinterpretCast<uint16_t>(),
                    lanes * 2);
                PipeBarrier<PIPE_V>();
                // Native int32 equality avoids a cast (float reinterpretation would
                // incorrectly treat bit 31 as -0). CompareScalar packs complete
                // 64-element repeats; GatherMask below still consumes only lanes.
                CompareScalar(hi, f, static_cast<int32_t>(0), CMPMODE::EQ, ((lanes + 63) / 64) * 64);
                PipeBarrier<PIPE_V>();
                uint64_t unseen = 0;
                GatherMask(a.ReinterpretCast<float>(), d.ReinterpretCast<float>(), hi.ReinterpretCast<uint32_t>(), true,
                           lanes, GatherMaskParams{1, 1, 8, 0}, unseen);
                PipeBarrier<PIPE_ALL>();
                for (uint32_t j = 0; j < unseen; ++j) {
                    uint32_t pos = a.GetValue(j), word = pos / 32;
                    bits.SetValue(word, bits.GetValue(word) | (1u << (pos % 32)));
                    touchedBegin = pos < touchedBegin ? pos : touchedBegin;
                    touchedEnd = pos + 1 > touchedEnd ? pos + 1 : touchedEnd;
                }
            }
        }
        // The bitmap already is a GatherMask selection mask: compact a vector
        // of ascending local rows directly, without expanding/testing 32 bits
        // or constructing each physical address with scalar instructions.
        CreateVecIndex(d, static_cast<int32_t>(0), K);
        PipeBarrier<PIPE_V>();
        // Every tile starts on a 128-row page boundary. Construct the first
        // tile's owner-local address pattern once; later tiles differ by start*rowStride.
        // Emit owner-local request addresses directly for HCCL routing.
        const uint32_t rowStride = 6;
        ShiftRight(b, d, static_cast<int32_t>(7), K);
        PipeBarrier<PIPE_V>();
        Muls(b, b, static_cast<int32_t>((rowStride - 1) * 128), K);
        PipeBarrier<PIPE_V>();
        Add(d, d, b, K);
        PipeBarrier<PIPE_V>();
        int32_t physicalBase = static_cast<int32_t>((core % 6) * 128);
        Adds(d, d, physicalBase, K);
        PipeBarrier<PIPE_ALL>();
        // Project reads page entries outside the touched interval too. Clear
        // them in vector tiles, then scan rank bases only where a bit can exist.
        // Empty words need no rank base: Remap never consumes absent tokens.
        Duplicate(a, static_cast<int32_t>(0), K);
        PipeBarrier<PIPE_ALL>();
        for (uint32_t start = 0; start < span / 16; start += K) {
            uint32_t count = span / 16 - start < K ? span / 16 - start : K;
            Store(mapping[begin / 16 + start], a, count);
        }
        uint32_t n = 0;
        uint32_t firstTile = (touchedBegin / K) * K;
        uint32_t lastTile = ((touchedEnd + K - 1) / K) * K;
        for (uint32_t start = firstTile; start < lastTile; start += K) {
            uint32_t nmap = span - start < K ? span - start : K, first = n;
            for (uint32_t word = start / 32; word < (start + nmap) / 32; ++word) {
                uint32_t v = bits.GetValue(word);
                a.SetValue((word - start / 32) * 2, static_cast<int32_t>(v));
                a.SetValue((word - start / 32) * 2 + 1, n);
                // Only rank bases need the per-word scan. Fixed-cost unsigned
                // popcount handles bit31 without a per-set-bit scalar loop.
                if (v) {
                    uint32_t x = v - ((v >> 1) & 0x55555555u);
                    x = (x & 0x33333333u) + ((x >> 2) & 0x33333333u);
                    x = (x + (x >> 4)) & 0x0f0f0f0fu;
                    x += x >> 8;
                    x += x >> 16;
                    n += x & 63u;
                }
            }
            Store(mapping[(begin + start) / 16], a, nmap / 16);
            if (n == first)
                continue;
            uint64_t selected = 0;
            GatherMask(e.ReinterpretCast<float>(), d.ReinterpretCast<float>(), bits[start / 32], true, nmap,
                       GatherMaskParams{1, 1, 8, 0}, selected);
            PipeBarrier<PIPE_ALL>();
            uint32_t count = static_cast<uint32_t>(selected);
            Adds(b, e, static_cast<int32_t>(start * rowStride), count);
            PipeBarrier<PIPE_V>();
            Store(temporary[static_cast<uint64_t>(begin) + first], b, count);
        }
        Duplicate(a, static_cast<int32_t>(0), 8);
        PipeBarrier<PIPE_ALL>();
        a.SetValue(0, n);
        Store(counts[core * 8], a, 8);
    }
    __aicore__ inline uint32_t CountOffsets() {
        Load(co, counts, cores * 8);
        uint32_t total = 0;
        for (uint32_t i = 0; i < cores; ++i) {
            uint32_t n = co.GetValue(i * 8);
            co.SetValue(i * 8 + 1, total);
            total += n;
        }
        return total;
    }
    __aicore__ inline void Project() {
        CountOffsets();
        uint32_t n = co.GetValue(core * 8), start = co.GetValue(core * 8 + 1);
        for (uint32_t j = 0; j < n; j += K) {
            uint32_t take = n - j < K ? n - j : K;
            Load(a, temporary[static_cast<uint64_t>(core) * span + j], take);
            Store(unique[start + j], a, take);
        }
        // Batch consecutive pages per core. Queue the 32-byte mapping reads,
        // synchronize once, then write the complete projected block in one DMA.
        constexpr uint32_t PAGE_BATCH = 128;
        uint32_t work = (B * P + cores - 1) / cores, first = core * work;
        uint32_t end = first + work < B * P ? first + work : B * P;
        for (uint32_t pos = first; pos < end; pos += PAGE_BATCH) {
            uint32_t take = end - pos < PAGE_BATCH ? end - pos : PAGE_BATCH;
            Load(b, pages[pos], take);
            Duplicate(a, static_cast<int32_t>(0), take * 8);
            Duplicate(e, static_cast<int32_t>(0), take * 8);
            PipeBarrier<PIPE_ALL>();
            for (uint32_t j = 0; j < take; ++j) {
                uint32_t i = pos + j, s = i / P, p = i % P;
                if (p * 128 >= static_cast<uint32_t>(prefixes.GetValue(s)))
                    continue;
                uint32_t physical = b.GetValue(j) / 128;
                uint32_t owner = (physical - R) % W, local = ((physical - R) / W + R) * 128;
                uint32_t localPage = local / 128;
                uint32_t key = owner * ownerSpan + (localPage % 6) * span + (localPage / 6) * 128;
                int32_t base = co.GetValue((key / span) * 8 + 1);
                for (uint32_t k = 1; k < 8; k += 2)
                    e.SetValue(j * 8 + k, base);
                DataCopyPad(a[j * 8], mapping[key / 16], DataCopyExtParams{1, 32, 0, 0, 0},
                            DataCopyPadExtParams<int32_t>{false, 0, 0, 0});
            }
            PipeBarrier<PIPE_ALL>();
            Add(a, a, e, take * 8);
            PipeBarrier<PIPE_V>();
            Store(logicalMap[static_cast<uint64_t>(pos) * 8], a, take * 8);
        }
    }
    __aicore__ inline void And32(LocalTensor<int32_t> out, LocalTensor<int32_t> x, LocalTensor<int32_t> y) {
        And(out.ReinterpretCast<uint16_t>(), x.ReinterpretCast<uint16_t>(), y.ReinterpretCast<uint16_t>(), K * 2);
        PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void Popcount() {
        // SWAR rank within the 32-token bitmap; c never contains bit 31 after
        // applying the exclusive-prefix mask. b and f are scratch.
        ShiftRight(b, c, static_cast<int32_t>(1), K);
        Duplicate(f, static_cast<int32_t>(0x55555555), K);
        PipeBarrier<PIPE_V>();
        And32(b, b, f);
        Sub(c, c, b, K);
        PipeBarrier<PIPE_V>();
        ShiftRight(b, c, static_cast<int32_t>(2), K);
        Duplicate(f, static_cast<int32_t>(0x33333333), K);
        PipeBarrier<PIPE_V>();
        And32(c, c, f);
        And32(b, b, f);
        Add(c, c, b, K);
        PipeBarrier<PIPE_V>();
        ShiftRight(b, c, static_cast<int32_t>(4), K);
        PipeBarrier<PIPE_V>();
        Add(c, c, b, K);
        Duplicate(f, static_cast<int32_t>(0x0f0f0f0f), K);
        PipeBarrier<PIPE_V>();
        And32(c, c, f);
        ShiftRight(b, c, static_cast<int32_t>(8), K);
        PipeBarrier<PIPE_V>();
        Add(c, c, b, K);
        PipeBarrier<PIPE_V>();
        ShiftRight(b, c, static_cast<int32_t>(16), K);
        PipeBarrier<PIPE_V>();
        Add(c, c, b, K);
        Duplicate(f, static_cast<int32_t>(63), K);
        PipeBarrier<PIPE_V>();
        And32(c, c, f);
    }
    __aicore__ inline void Remap() {
        uint32_t total = CountOffsets(), perCore = (Q + cores - 1) / cores;
        uint32_t qstart = core * perCore, qend = qstart + perCore < Q ? qstart + perCore : Q;
        for (uint32_t i = 0; i < 32; ++i)
            lut.SetValue(i, static_cast<int32_t>((1u << i) - 1));
        PipeBarrier<PIPE_ALL>();
        for (uint32_t s = 0; s < B; ++s) {
            bool used = false;
            for (uint32_t q = qstart; q < qend; ++q)
                if (seq.GetValue(q) == static_cast<int32_t>(s))
                    used = true;
            if (!used)
                continue;
            int32_t pre = prefixes.GetValue(s), off = offsets.GetValue(s);
            uint32_t histPages = (pre + 127) / 128, tiles = histPages ? (histPages + TILE_PAGES - 1) / TILE_PAGES : 1;
            for (uint32_t tile = 0; tile < tiles; ++tile) {
                uint32_t first = tile * TILE_PAGES, take = histPages - first;
                if (take > TILE_PAGES)
                    take = TILE_PAGES;
                if (take)
                    Load(pt, logicalMap[(static_cast<uint64_t>(s) * P + first) * 8], take * 8);
                else {
                    Duplicate(pt, static_cast<int32_t>(0), 8);
                    PipeBarrier<PIPE_ALL>();
                }
                int32_t begin = first * 128, end = begin + take * 128;
                if (end > pre)
                    end = pre;
                for (uint32_t q = qstart; q < qend; ++q) {
                    if (seq.GetValue(q) != static_cast<int32_t>(s))
                        continue;
                    Load(a, topk[static_cast<uint64_t>(q) * K], K);
                    if (tile == 0) {
                        Adds(d, a, static_cast<int32_t>(total) + off - pre, K);
                        PipeBarrier<PIPE_V>();
                        InRange(0, pre);
                        Duplicate(e, static_cast<int32_t>(-1), K);
                        PipeBarrier<PIPE_V>();
                        SelectI32(d, lo, d, e);
                    } else
                        Load(d, indices[static_cast<uint64_t>(q) * K], K);
                    Adds(b, a, -begin, K);
                    PipeBarrier<PIPE_V>();
                    Maxs(b, b, static_cast<int32_t>(0), K);
                    PipeBarrier<PIPE_V>();
                    Mins(b, b, static_cast<int32_t>(take ? take * 128 - 1 : 0), K);
                    PipeBarrier<PIPE_V>();
                    ShiftRight(b, b, static_cast<int32_t>(5), K);
                    PipeBarrier<PIPE_V>();
                    Muls(b, b, static_cast<int32_t>(8), K);
                    PipeBarrier<PIPE_V>();
                    Gather(c, pt, b.ReinterpretCast<uint32_t>(), 0, K);
                    PipeBarrier<PIPE_V>();
                    Adds(b, b, static_cast<int32_t>(4), K);
                    PipeBarrier<PIPE_V>();
                    Gather(e, pt, b.ReinterpretCast<uint32_t>(), 0, K);
                    ShiftRight(f, a, static_cast<int32_t>(5), K);
                    PipeBarrier<PIPE_V>();
                    Muls(f, f, static_cast<int32_t>(32), K);
                    PipeBarrier<PIPE_V>();
                    Sub(f, a, f, K);
                    PipeBarrier<PIPE_V>();
                    Muls(f, f, static_cast<int32_t>(4), K);
                    PipeBarrier<PIPE_V>();
                    Gather(b, lut, f.ReinterpretCast<uint32_t>(), 0, K);
                    PipeBarrier<PIPE_V>();
                    And32(c, c, b);
                    Popcount();
                    Add(c, c, e, K);
                    PipeBarrier<PIPE_V>();
                    InRange(begin, end);
                    SelectI32(d, hi, c, d);
                    Store(indices[static_cast<uint64_t>(q) * K], d, K);
                }
            }
        }
    }

  public:
    __aicore__ inline void Init(GM_ADDR t, GM_ADDR p, GM_ADDR s, GM_ADDR pre, GM_ADDR off, GM_ADDR k, GM_ADDR map,
                                GM_ADDR tmp, GM_ADDR cnt, GM_ADDR uni, GM_ADDR idx, GM_ADDR lm, GM_ADDR rc, uint32_t bn,
                                uint32_t qn, uint32_t pn, uint32_t wn, uint32_t ln, uint32_t rn, uint32_t ph,
                                uint32_t cn, uint32_t sn, TPipe *pipe) {
        topk.SetGlobalBuffer((__gm__ int32_t *)t);
        pages.SetGlobalBuffer((__gm__ int32_t *)p);
        seq.SetGlobalBuffer((__gm__ int32_t *)s);
        prefixes.SetGlobalBuffer((__gm__ int32_t *)pre);
        offsets.SetGlobalBuffer((__gm__ int32_t *)off);
        keys.SetGlobalBuffer((__gm__ int32_t *)k);
        mapping.SetGlobalBuffer((__gm__ int32_t *)map);
        temporary.SetGlobalBuffer((__gm__ int32_t *)tmp);
        counts.SetGlobalBuffer((__gm__ int32_t *)cnt);
        unique.SetGlobalBuffer((__gm__ int32_t *)uni);
        indices.SetGlobalBuffer((__gm__ int32_t *)idx);
        routeCounts.SetGlobalBuffer((__gm__ int32_t *)rc);
        logicalMap.SetGlobalBuffer((__gm__ int32_t *)lm);
        B = bn;
        Q = qn;
        P = pn;
        W = wn;
        L = ln;
        R = rn;
        phase = ph;
        cores = cn;
        span = sn;
        core = GetBlockIdx();
        ownerSpan = 6 * span;
        pipe->InitBuffer(aBuf, K * 4);
        pipe->InitBuffer(bBuf, K * 4);
        pipe->InitBuffer(cBuf, K * 4);
        pipe->InitBuffer(dBuf, K * 4);
        pipe->InitBuffer(eBuf, K * 4);
        pipe->InitBuffer(fBuf, K * 4);
        pipe->InitBuffer(lutBuf, 128);
        pipe->InitBuffer(pageBuf, phase == 3 ? TILE_PAGES * 32 : ((P + 7) / 8) * 32);
        pipe->InitBuffer(bitBuf, (phase == 1) ? span / 8 : 32);
        pipe->InitBuffer(loBuf, K / 8);
        pipe->InitBuffer(hiBuf, K / 8);
        pipe->InitBuffer(countBuf, cores * 32);
        pipe->InitBuffer(routeBuf, phase == 0 ? (K + W * 8 + 8) * 4 : 32);
        a = aBuf.Get<int32_t>();
        b = bBuf.Get<int32_t>();
        c = cBuf.Get<int32_t>();
        d = dBuf.Get<int32_t>();
        pt = pageBuf.Get<int32_t>();
        bits = bitBuf.Get<uint32_t>();
        lo = loBuf.Get<uint8_t>();
        hi = hiBuf.Get<uint8_t>();
        co = countBuf.Get<int32_t>();
        e = eBuf.Get<int32_t>();
        f = fBuf.Get<int32_t>();
        lut = lutBuf.Get<int32_t>();
        routed = routeBuf.Get<int32_t>();
    }
    __aicore__ inline void Process() {
        if (phase == 0) {
            Lookup();
            Route();
        } else if (phase == 1)
            Mark();
        else if (phase == 2)
            Project();
        else
            Remap();
    }
};
extern "C" __global__ __aicore__ void selected_kv_metadata(GM_ADDR topk, GM_ADDR pages, GM_ADDR seq, GM_ADDR prefix,
                                                           GM_ADDR offsets, GM_ADDR keys, GM_ADDR mapping,
                                                           GM_ADDR temporary, GM_ADDR counts, GM_ADDR unique,
                                                           GM_ADDR indices, GM_ADDR logical_map, GM_ADDR route_counts,
                                                           GM_ADDR workspace, GM_ADDR tiling) {
    (void)workspace;
    GET_TILING_DATA(t, tiling);
    TPipe pipe;
    if (TILING_KEY_IS(1)) {
        SelectedKvMetadataKernel op;
        op.Init(topk, pages, seq, prefix, offsets, keys, mapping, temporary, counts, unique, indices, logical_map,
                route_counts, t.requests, t.queries, t.pages, t.world, t.localRows, t.reserved, t.phase, t.cores,
                t.span, &pipe);
        op.Process();
    }
}
