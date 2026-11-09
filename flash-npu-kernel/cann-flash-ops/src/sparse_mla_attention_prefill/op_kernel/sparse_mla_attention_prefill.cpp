#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "sparse_mla_attention_prefill_template_tiling_key.h"

using namespace AscendC;

constexpr SoftmaxConfig PREFILL_SOFTMAX_CFG = {
    false, 0, 0, SoftmaxMode::SOFTMAX_OUTPUT_WITHOUT_BRC};

template<typename T, bool KV_INT8, bool PACKED_QUERY, bool COMM_OUTPUT,
         bool FAST_PREP>
class SparseMlaAttentionPrefillKernel {
    static constexpr uint32_t HEADS = 64;
    static constexpr uint32_t HALF_HEADS = 32;
    static constexpr uint32_t LATENT = 512;
    static constexpr uint32_t ROPE = 64;
    static constexpr uint32_t WIDTH = LATENT + ROPE;
    static constexpr uint32_t PAGE = 128;
    static constexpr uint32_t CHUNK = 128;
    static constexpr uint32_t GROUP_CHUNKS = 4;
    static constexpr uint32_t GROUP_TOKENS = GROUP_CHUNKS * CHUNK;
    static constexpr uint32_t GROUP_SLOTS = 3;
    static constexpr uint32_t CAST_HEADS = 16;
    static constexpr uint32_t GATHER_PART = 32;
    static constexpr uint32_t INT8_PACKED_PAYLOAD_BYTES = 656;
    static constexpr uint32_t INT8_PACKED_ROW_ALIGN = 672;
    static constexpr uint32_t INT8_DEQUANT_ROWS = 13;
    static constexpr uint32_t INT8_FP32_ROWS = 13;
    static constexpr uint32_t MAX_SPARSE_TOKENS = 2048;
    static constexpr uint32_t ADDRESS_COUNT = MAX_SPARSE_TOKENS / 2;
    static constexpr uint32_t MAX_BLOCK_TABLE = 10000;
    static constexpr uint32_t GATHER_DATA_BYTES = GATHER_PART * WIDTH * sizeof(T);
    static constexpr uint32_t ADDRESS_BYTES = ADDRESS_COUNT * sizeof(int32_t);
    static constexpr uint32_t STATS_EXP_MAX = 128;
    static constexpr uint32_t STATS_GROUP_SCALE = 160;
    static constexpr uint32_t STATS_ROWS = 256;
    static constexpr uint32_t ROW_SLOTS = GROUP_SLOTS * GROUP_CHUNKS;
    static constexpr uint32_t Q_BUF = HEADS * WIDTH;
    static constexpr uint32_t KV_BUF = CHUNK * WIDTH;
    static constexpr uint32_t MM1_BYTES = HEADS * GROUP_TOKENS * sizeof(float);
    static constexpr uint32_t PROB_BYTES = HEADS * GROUP_TOKENS * sizeof(T);
    static constexpr uint32_t MM2_BYTES = HEADS * LATENT * sizeof(float);
    static constexpr uint32_t PACKED_KV_BYTES = KV_BUF * sizeof(T);
    static constexpr uint32_t ROWS_BYTES = ROW_SLOTS * 2 * 64;
    static constexpr uint32_t TASK_BYTES = 2 * 64;
    static constexpr uint32_t CORE_DATA_BYTES = ROWS_BYTES +
        GROUP_SLOTS * (MM1_BYTES + PROB_BYTES + MM2_BYTES +
                       GROUP_CHUNKS * PACKED_KV_BYTES);
    static constexpr uint32_t CORE_WORKSPACE = (CORE_DATA_BYTES + 511) / 512 * 512;
    static constexpr uint32_t L0A_BYTES = 32 * 1024;
    static constexpr uint32_t L0B_BYTES = 32 * 1024;
    static constexpr uint32_t L0C_BYTES = 64 * 1024;
    // Cross-core flags 11-14 are reserved by the CANN super-kernel task
    // boundary protocol. Both vector sub-blocks publish the slot-ready flag.
    static constexpr uint32_t SYNC_KV = 0;
    static constexpr uint32_t SYNC_MM1 = 1;
    static constexpr uint32_t SYNC_PROB = 4;
    static constexpr uint32_t SYNC_TASK = 5;
    static constexpr uint32_t SYNC_MM2 = 8;
    static constexpr uint32_t QUERY_READY_EVENT = 5;
    static constexpr IsResetLoad3dConfig LOAD3D_CONFIG = {true, true};

    GlobalTensor<T> queryGm, queryRopeGm, keyGm, keyRopeGm, probGm;
    GlobalTensor<int8_t> packedKeyGm;
    GlobalTensor<T> outputGm;
    GlobalTensor<T> packedKvGm;
    GlobalTensor<float> mm1Gm, mm2Gm, maxGm, sumGm;
    GlobalTensor<int32_t> indicesGm, validChunksGm, blocksGm, querySeqGm;
    GlobalTensor<int32_t> rowsGm, taskGm, scheduleGm;

    TBuf<TPosition::A1> qL1Buf, kvL1Buf, probL1Buf;
    TBuf<TPosition::A2> l0ABuf;
    TBuf<TPosition::B2> l0BBuf;
    TBuf<TPosition::CO1> l0CBuf;
    LocalTensor<T> qL1, kvL1, probL1, l0A, l0B;
    LocalTensor<float> l0C;

    TBuf<> logitsBuf, accumBuf, statsBuf, softmaxBuf, gatherBuf;
    LocalTensor<float> logitsUb, mm2Ub, accumUb, statsUb;
    LocalTensor<T> probUb, gatherUb;
    LocalTensor<int32_t> addressUb, addressTmpUb;
    LocalTensor<uint8_t> softmaxUb;
    TEventID gatherMte3ToV, gatherVToMte3;
    TEventID gatherVToMte2;
    TEventID probMte3ToMte2;
    TEventID countSToMte3, countMte2ToS, taskMte3ToS;
    TEventID addressMte2ToV, addressVToS;

    uint32_t totalQuery, sparseCount, pageSize, maxBlocks, usedCores, batchSize;
    uint32_t packedRowBytes;
    uint32_t queryRankSize, queryLocalHeads;
    uint32_t coreTaskBegin, coreTaskEnd;
    uint32_t currentTask;
    uint32_t sparseMode;
    uint32_t abL0BufIter, cL0BufIter;
    float scale;
    bool returnLse;


    __aicore__ inline uint32_t ProbFlag(uint32_t slot) {
        return slot == 0 ? SYNC_PROB : slot + 5;
    }

    struct GroupInfo {
        uint32_t task;
        uint32_t group;
        uint32_t slot;
        uint32_t chunks;
        bool last;
    };

    __aicore__ inline bool IsDirectOutput(const GroupInfo &info) {
        return info.group == 0 && info.last;
    }

    __aicore__ inline uint32_t TaskCost(
        LocalTensor<int32_t> liveChunks, uint32_t task) {
        int32_t raw = liveChunks.GetValue(task);
        if (raw <= 0) return 1;
        uint32_t valid = static_cast<uint32_t>(raw);
        if (valid > sparseCount) valid = sparseCount;
        uint32_t cost = (valid + CHUNK - 1) / CHUNK;
        uint32_t maximum = (sparseCount + CHUNK - 1) / CHUNK;
        return cost > maximum ? maximum : cost;
    }

    // Runtime values produced by SelectLocalTopkIndices must not be consumed
    // by host/AICPU tiling.  AIV0 reads the current invocation through MTE2,
    // computes one range for its whole AIC/AIV group, then publishes that
    // range to the other two lanes.  This avoids both scalar-cache staleness
    // across graph replays and lane-to-lane disagreement.
    __aicore__ inline void BuildTaskRange(
        uint32_t core, LocalTensor<int32_t> liveChunks) {
        uint32_t totalCost = 0;
        uint32_t maximumCost = 1;
        for (uint32_t task = 0; task < totalQuery; ++task) {
            uint32_t cost = TaskCost(liveChunks, task);
            totalCost += cost;
            if (cost > maximumCost) maximumCost = cost;
        }

        // Find the minimum capacity for a contiguous partition over the
        // available AICs. This is the same minimax partitioner that delivered
        // the original dynamic-load-balancing gain, now evaluated in-kernel.
        uint32_t low = maximumCost;
        uint32_t high = totalCost;
        while (low < high) {
            uint32_t capacity = low + (high - low) / 2;
            uint32_t partitions = 1;
            uint32_t load = 0;
            for (uint32_t task = 0; task < totalQuery; ++task) {
                uint32_t cost = TaskCost(liveChunks, task);
                if (load + cost > capacity) {
                    ++partitions;
                    load = 0;
                }
                load += cost;
            }
            if (partitions <= usedCores) high = capacity;
            else low = capacity + 1;
        }

        uint32_t boundary = 0;
        uint32_t begin = 0;
        uint32_t end = 0;
        for (uint32_t owner = 0; owner <= core; ++owner) {
            begin = boundary;
            uint32_t remainingCores = usedCores - owner - 1;
            uint32_t limit = totalQuery - remainingCores;
            uint32_t load = 0;
            while (boundary < limit) {
                uint32_t cost = TaskCost(liveChunks, boundary);
                if (load + cost > low) break;
                load += cost;
                ++boundary;
            }
            end = boundary;
        }
        coreTaskBegin = begin;
        coreTaskEnd = end;
    }

    __aicore__ inline void PublishTaskRange(uint32_t core) {
        LocalTensor<int32_t> record =
            statsUb.template ReinterpretCast<int32_t>();
        record.SetValue(0, static_cast<int32_t>(coreTaskBegin));
        record.SetValue(1, static_cast<int32_t>(coreTaskEnd));
        SetFlag<HardEvent::S_MTE3>(countSToMte3);
        WaitFlag<HardEvent::S_MTE3>(countSToMte3);
        DataCopy(scheduleGm[core * 16], record, 8);
        SetFlag<HardEvent::MTE3_S>(taskMte3ToS);
        WaitFlag<HardEvent::MTE3_S>(taskMte3ToS);
    }

    __aicore__ inline void LoadTaskRange(uint32_t core) {
        DataCacheCleanAndInvalid<
            int32_t, CacheLine::SINGLE_CACHE_LINE,
            DcciDst::CACHELINE_OUT>(scheduleGm[core * 16]);
        coreTaskBegin = static_cast<uint32_t>(
            scheduleGm.GetValue(core * 16));
        coreTaskEnd = static_cast<uint32_t>(
            scheduleGm.GetValue(core * 16 + 1));
    }

    __aicore__ inline void PrepareTaskRange(uint32_t core,
                                            uint32_t subBlock) {
        if (subBlock != 0) return;
        LocalTensor<int32_t> liveChunks =
            logitsUb.template ReinterpretCast<int32_t>();
        DataCopyExtParams copy{1,
            static_cast<uint32_t>(totalQuery * sizeof(int32_t)), 0, 0, 0};
        DataCopyPadExtParams<int32_t> noPad{false, 0, 0, 0};
        DataCopyPad(liveChunks, validChunksGm, copy, noPad);
        SetFlag<HardEvent::MTE2_S>(countMte2ToS);
        WaitFlag<HardEvent::MTE2_S>(countMte2ToS);
        BuildTaskRange(core, liveChunks);
        PublishTaskRange(core);
    }

    __aicore__ inline void CopyNd(LocalTensor<T> dst, GlobalTensor<T> src,
                                  uint32_t rows, uint32_t rowAlign,
                                  uint32_t cols, uint32_t stride) {
        Nd2NzParams p = {};
        p.ndNum = 1;
        p.nValue = rows;
        p.dValue = cols;
        p.srcDValue = stride;
        p.dstNzC0Stride = rowAlign;
        p.dstNzNStride = 1;
        p.srcNdMatrixStride = 0;
        p.dstNzMatrixStride = 0;
        DataCopy(dst, src, p);
    }

    __aicore__ inline void LoadQuery(uint32_t task, uint32_t buf) {
        LocalTensor<T> q = qL1[buf * Q_BUF];
        if constexpr (PACKED_QUERY) {
            constexpr uint32_t blockElements = 32 / sizeof(T);
            for (uint32_t rank = 0; rank < queryRankSize; ++rank) {
                uint64_t base =
                    (static_cast<uint64_t>(rank) * totalQuery + task) *
                    queryLocalHeads * WIDTH;
                uint32_t dstHead = rank * queryLocalHeads;
                CopyNd(q[dstHead * blockElements], queryGm[base],
                       queryLocalHeads, HEADS, WIDTH, WIDTH);
            }
            SetFlag<HardEvent::MTE2_MTE1>(QUERY_READY_EVENT + buf);
        } else {
            uint64_t base = static_cast<uint64_t>(task) * HEADS * LATENT;
            uint64_t ropeBase = static_cast<uint64_t>(task) * HEADS * ROPE;
            CopyNd(q, queryGm[base], HEADS, HEADS, LATENT, LATENT);
            CopyNd(q[HEADS * LATENT], queryRopeGm[ropeBase],
                   HEADS, HEADS, ROPE, ROPE);
        }
    }

    __aicore__ inline void WaitQuery(uint32_t buf) {
        if constexpr (PACKED_QUERY) {
            WaitFlag<HardEvent::MTE2_MTE1>(QUERY_READY_EVENT + buf);
        }
    }

    __aicore__ inline void WaitKv() {
        CrossCoreWaitFlag(SYNC_KV);
    }

    __aicore__ inline uint32_t WaitTask(uint32_t task) {
        CrossCoreWaitFlag(SYNC_TASK);
        return static_cast<uint32_t>(taskGm.GetValue(task * 2 * 16));
    }

    __aicore__ inline void LoadKv(uint32_t buf, uint32_t groupSlot,
                                  uint32_t chunkInGroup) {
        WaitFlag<HardEvent::MTE1_MTE2>(1 + buf);
        LocalTensor<T> kv = kvL1[buf * KV_BUF];
        GlobalTensor<T> packed = packedKvGm[
            (groupSlot * GROUP_CHUNKS + chunkInGroup) * KV_BUF];
        CopyNd(kv, packed, CHUNK / 2, CHUNK, LATENT, LATENT);
        CopyNd(kv[CHUNK * LATENT], packed[CHUNK * LATENT],
               CHUNK / 2, CHUNK, ROPE, ROPE);
        CopyNd(kv[(CHUNK / 2) * 16], packed[(CHUNK / 2) * LATENT],
               CHUNK / 2, CHUNK, LATENT, LATENT);
        CopyNd(kv[CHUNK * LATENT + (CHUNK / 2) * 16],
               packed[CHUNK * LATENT + (CHUNK / 2) * ROPE],
               CHUNK / 2, CHUNK, ROPE, ROPE);
        SetFlag<HardEvent::MTE2_MTE1>(1 + buf);
    }

    __aicore__ inline void LoadValue(uint32_t buf, uint32_t groupSlot,
                                     uint32_t chunkInGroup) {
        WaitFlag<HardEvent::MTE1_MTE2>(1 + buf);
        LocalTensor<T> kv = kvL1[buf * KV_BUF];
        GlobalTensor<T> packed = packedKvGm[
            (groupSlot * GROUP_CHUNKS + chunkInGroup) * KV_BUF];
        CopyNd(kv, packed, CHUNK, CHUNK, LATENT, LATENT);
        SetFlag<HardEvent::MTE2_MTE1>(1 + buf);
    }


    __aicore__ inline void LoadA(LocalTensor<T> dst, LocalTensor<T> src,
                                 uint32_t m, uint32_t k) {
        LoadData3DParamsV2<T> p = {};
        p.l1H = m / 16;
        p.l1W = 16;
        p.padList[3] = 255;
        p.mExtension = m;
        p.kExtension = k;
        p.strideW = p.strideH = p.filterW = p.filterH = 1;
        p.dilationFilterW = p.dilationFilterH = 1;
        p.channelSize = k;
        LoadData<T, LOAD3D_CONFIG>(dst, src, p);
    }

    __aicore__ inline void LoadB(LocalTensor<T> dst, LocalTensor<T> src,
                                 uint32_t k, uint32_t n, bool transpose) {
        if (!transpose) {
            LoadData2DParams p = {};
            p.startIndex = 0;
            p.repeatTimes = n / 16 * k / (32 / sizeof(T));
            p.srcStride = 1;
            p.dstGap = 0;
            p.ifTranspose = false;
            LoadData(dst, src, p);
            return;
        }
        LoadData3DParamsV2<T> p = {};
        p.l1H = k / 16;
        p.l1W = 16;
        p.padList[3] = 255;
        p.mExtension = k;
        p.kExtension = n;
        p.strideW = p.strideH = p.filterW = p.filterH = 1;
        p.dilationFilterW = p.dilationFilterH = 1;
        p.enTranspose = 1;
        p.channelSize = n;
        LoadData<T, LOAD3D_CONFIG>(dst, src, p);
    }

    __aicore__ inline void DrainL0AB() {
        WaitFlag<HardEvent::M_MTE1>(3);
        WaitFlag<HardEvent::M_MTE1>(4);
        WaitFlag<HardEvent::MTE1_MTE2>(1);
        WaitFlag<HardEvent::MTE1_MTE2>(2);
        WaitFlag<HardEvent::MTE1_MTE2>(3);
        WaitFlag<HardEvent::MTE1_MTE2>(4);
    }

    __aicore__ inline void MmadBlock(LocalTensor<float> c, LocalTensor<T> a,
                                     LocalTensor<T> b, uint32_t m, uint32_t n,
                                     uint32_t k, bool init, bool last,
                                     bool unitFlag = true) {
        MmadParams p = {};
        p.m = m;
        p.n = n;
        p.k = k;
        p.cmatrixInitVal = init;
        p.cmatrixSource = false;
        p.unitFlag = unitFlag ? (last ? 0b11 : 0b10) : 0;
        Mmad(c, a, b, p);
    }

    __aicore__ inline void Fix(LocalTensor<float> c, GlobalTensor<float> dst,
                              uint32_t m, uint32_t n, uint32_t stride,
                              bool unitFlag = true) {
        FixpipeParamsV220 p = {};
        p.nSize = n;
        p.mSize = m;
        p.srcStride = m;
        p.dstStride = stride;
        p.ndNum = 1;
        p.unitFlag = unitFlag ? 0b11 : 0;
        Fixpipe(dst, c, p);
    }

    __aicore__ inline void FixOutput(LocalTensor<float> c,
                                     GlobalTensor<T> dst,
                                     uint32_t m, uint32_t n,
                                     uint32_t stride,
                                     bool unitFlag = true) {
        FixpipeParamsV220 p = {};
        p.nSize = n;
        p.mSize = m;
        p.srcStride = m;
        p.dstStride = stride;
        p.ndNum = 1;
        p.unitFlag = unitFlag ? 0b11 : 0;
        p.quantPre = QuantMode_t::F322BF16;
        Fixpipe<T, float, CFG_ROW_MAJOR>(dst, c, p);
    }

    __aicore__ inline void LoadProb(uint32_t buf, uint32_t groupSlot,
                                    uint32_t chunkInGroup) {
        WaitFlag<HardEvent::MTE1_MTE2>(3 + buf);
        LocalTensor<T> prob = probL1[buf * HEADS * CHUNK];
        CopyNd(prob, probGm[groupSlot * HEADS * GROUP_TOKENS +
                            chunkInGroup * CHUNK],
               HEADS, HEADS, CHUNK, GROUP_TOKENS);
        SetFlag<HardEvent::MTE2_MTE1>(3 + buf);
    }

    __aicore__ inline void Mm1(uint32_t qBuf, uint32_t kvBuf,
                               uint32_t groupSlot,
                               uint32_t chunkInGroup) {
        WaitFlag<HardEvent::MTE2_MTE1>(1 + kvBuf);
        LocalTensor<T> q = qL1[qBuf * Q_BUF];
        LocalTensor<T> kv = kvL1[kvBuf * KV_BUF];
        uint32_t cSlot = cL0BufIter & 1;
        LocalTensor<float> c = l0C[cSlot * L0C_BYTES / sizeof(float)];
        for (uint32_t part = 0; part < 6; ++part) {
            uint32_t ping = abL0BufIter & 1;
            LocalTensor<T> a = l0A[ping * L0A_BYTES / sizeof(T)];
            LocalTensor<T> b = l0B[ping * L0B_BYTES / sizeof(T)];
            WaitFlag<HardEvent::M_MTE1>(3 + ping);
            LoadA(a, q[part * HEADS * 96], HEADS, 96);
            LoadB(b, kv[part * CHUNK * 96], 96, CHUNK, false);
            SetFlag<HardEvent::MTE1_M>(3 + ping);
            WaitFlag<HardEvent::MTE1_M>(3 + ping);
            MmadBlock(c, a, b, HEADS, CHUNK, 96,
                      part == 0, part == 5);
            SetFlag<HardEvent::M_MTE1>(3 + ping);
            ++abL0BufIter;
        }
        SetFlag<HardEvent::MTE1_MTE2>(1 + kvBuf);
        Fix(c, mm1Gm[groupSlot * HEADS * GROUP_TOKENS +
                     chunkInGroup * CHUNK],
            HEADS, CHUNK, GROUP_TOKENS);
        ++cL0BufIter;
    }


    __aicore__ inline void Mm2Chunk(uint32_t kvBuf, uint32_t probBuf,
                                    uint32_t chunkInGroup,
                                    uint32_t groupChunks,
                                    uint32_t groupSlot, uint32_t task,
                                    bool directOutput) {
        WaitFlag<HardEvent::MTE2_MTE1>(1 + kvBuf);
        WaitFlag<HardEvent::MTE2_MTE1>(3 + probBuf);
        LocalTensor<T> prob = probL1[probBuf * HEADS * CHUNK];
        LocalTensor<T> kv = kvL1[kvBuf * KV_BUF];
        for (uint32_t n = 0; n < 4; ++n) {
            uint32_t ping = abL0BufIter & 1;
            LocalTensor<T> a = l0A[ping * L0A_BYTES / sizeof(T)];
            LocalTensor<T> b = l0B[ping * L0B_BYTES / sizeof(T)];
            LocalTensor<float> c = l0C[n * HEADS * 128];
            uint32_t col = n * 128;
            WaitFlag<HardEvent::M_MTE1>(3 + ping);
            LoadA(a, prob, HEADS, CHUNK);
            LoadB(b, kv[col * CHUNK], CHUNK, 128, true);
            SetFlag<HardEvent::MTE1_M>(3 + ping);
            WaitFlag<HardEvent::MTE1_M>(3 + ping);
            MmadBlock(c, a, b, HEADS, 128, CHUNK,
                      chunkInGroup == 0,
                      chunkInGroup + 1 == groupChunks);
            SetFlag<HardEvent::M_MTE1>(3 + ping);
            // Consume each completed L0C tile immediately.  Deferring all four
            // Fixpipe operations overfills the two-slot MM/Fix unitFlag chain.
            if (chunkInGroup + 1 == groupChunks) {
                if (directOutput) {
                    if constexpr (COMM_OUTPUT) {
                        // Write [N,T,D] directly.  The public tensor can then
                        // be viewed as [kvp, N/kvp, T, D] by AllToAll without
                        // materializing the former TND -> NTD transpose.
                        FixOutput(c, outputGm[
                            static_cast<uint64_t>(task) * LATENT + n * 128],
                            HEADS, 128, totalQuery * LATENT);
                    } else {
                        FixOutput(c, outputGm[
                            static_cast<uint64_t>(task) * HEADS * LATENT +
                            n * 128], HEADS, 128, LATENT);
                    }
                } else {
                    Fix(c, mm2Gm[groupSlot * HEADS * LATENT + n * 128],
                        HEADS, 128, LATENT);
                }
            }
            ++abL0BufIter;
        }
        SetFlag<HardEvent::MTE1_MTE2>(1 + kvBuf);
        SetFlag<HardEvent::MTE1_MTE2>(3 + probBuf);
    }

    __aicore__ inline void InitCube(TPipe *pipe) {
        pipe->InitBuffer(qL1Buf, 2 * Q_BUF * sizeof(T));
        pipe->InitBuffer(kvL1Buf, 2 * KV_BUF * sizeof(T));
        pipe->InitBuffer(probL1Buf, 2 * HEADS * CHUNK * sizeof(T));
        pipe->InitBuffer(l0ABuf, 2 * L0A_BYTES);
        pipe->InitBuffer(l0BBuf, 2 * L0B_BYTES);
        pipe->InitBuffer(l0CBuf, 2 * L0C_BYTES);
        qL1 = qL1Buf.Get<T>();
        kvL1 = kvL1Buf.Get<T>();
        probL1 = probL1Buf.Get<T>();
        l0A = l0ABuf.Get<T>();
        l0B = l0BBuf.Get<T>();
        l0C = l0CBuf.Get<float>();
        abL0BufIter = 0;
        cL0BufIter = 0;
        SetFlag<HardEvent::M_MTE1>(3);
        SetFlag<HardEvent::M_MTE1>(4);
        SetFlag<HardEvent::MTE1_MTE2>(1);
        SetFlag<HardEvent::MTE1_MTE2>(2);
        SetFlag<HardEvent::MTE1_MTE2>(3);
        SetFlag<HardEvent::MTE1_MTE2>(4);
    }

    __aicore__ inline void FreeCubeEvents(TPipe *pipe) {
        (void)pipe;
    }

    __aicore__ inline void AlignVectorStartFromCube() {
        // Consume both AIV startup arrivals before any normal SYNC_KV use,
        // then release both sub-blocks through the normal AIC -> AIV path.
        CrossCoreWaitFlag(SYNC_KV);
        CrossCoreSetFlag<0x2, PIPE_FIX>(SYNC_MM1);
    }

    __aicore__ inline void StartCubeWithPublishedRange(uint32_t core) {
        // AIV0's schedule write and both AIV startup arrivals are ordered on
        // PIPE_MTE3.  Read the published record before releasing the pair.
        CrossCoreWaitFlag(SYNC_KV);
        LoadTaskRange(core);
        CrossCoreSetFlag<0x2, PIPE_FIX>(SYNC_MM1);
    }

    __aicore__ inline void CubeMm1Group(
        uint32_t qBuf, uint32_t groupSlot, uint32_t groupChunks,
        bool prefetchValue, uint32_t valueGroupSlot, uint32_t &valueBuf) {
        WaitKv();
        LoadKv(0, groupSlot, 0);
        for (uint32_t i = 0; i < groupChunks; ++i) {
            uint32_t kvBuf = i & 1;
            if (i + 1 < groupChunks) {
                WaitKv();
                LoadKv(kvBuf ^ 1, groupSlot, i + 1);
            } else if (prefetchValue) {
                // The other KV slot has already been consumed by the prior
                // MM1 chunk. Fill it with the previous group's first Value
                // chunk while this final MM1 runs.
                valueBuf = kvBuf ^ 1;
                LoadValue(valueBuf, valueGroupSlot, 0);
            }
            Mm1(qBuf, kvBuf, groupSlot, i);
        }
        CrossCoreSetFlag<0x2, PIPE_FIX>(SYNC_MM1 + groupSlot);
    }

    __aicore__ inline void CubeMm2Group(
        uint32_t task, uint32_t groupSlot, uint32_t groupChunks,
        bool directOutput, bool valuePrefetched = false,
        uint32_t firstValueBuf = 0) {
        if (!valuePrefetched) LoadValue(firstValueBuf, groupSlot, 0);
        CrossCoreWaitFlag(ProbFlag(groupSlot));
        LoadProb(0, groupSlot, 0);
        for (uint32_t i = 0; i < groupChunks; ++i) {
            uint32_t kvBuf = (firstValueBuf + i) & 1;
            uint32_t probBuf = i & 1;
            if (i + 1 < groupChunks) {
                LoadValue(kvBuf ^ 1, groupSlot, i + 1);
                LoadProb(probBuf ^ 1, groupSlot, i + 1);
            }
            Mm2Chunk(kvBuf, probBuf, i, groupChunks, groupSlot,
                     task, directOutput);
        }
        CrossCoreSetFlag<0x2, PIPE_FIX>(SYNC_MM2 + groupSlot);
    }

    __aicore__ inline void ProcessCube() {
        uint32_t core = GetBlockIdx();
        if (core >= usedCores) {
            DrainL0AB();
            return;
        }
        uint32_t item = 0;
        uint32_t prevTask = 0;
        uint32_t prevSlot = 0;
        uint32_t prevChunks = 0;
        bool prevDirect = false;
        bool hasPrev = false;
        // Rank-major Query needs several strided GM reads.  Start the first
        // read before waiting for AIV to publish its sparse task. Subsequent
        // reads are issued only after the current query's final MM1, because
        // MTE1 must not consume one L1 slot while MTE2 fills the other slot.
        if constexpr (PACKED_QUERY) {
            if (coreTaskBegin < coreTaskEnd) {
                LoadQuery(coreTaskBegin, coreTaskBegin & 1);
            }
        }
        for (uint32_t task = coreTaskBegin; task < coreTaskEnd; ++task) {
            uint32_t chunks = WaitTask(task);
            WaitQuery(task & 1);
            uint32_t groups = (chunks + GROUP_CHUNKS - 1) / GROUP_CHUNKS;
            if (groups == 0) {
                if (hasPrev) {
                    CubeMm2Group(prevTask, prevSlot, prevChunks, prevDirect);
                    hasPrev = false;
                }
                // A zero-topk task has no normal KV/MM1 handshake.  Realign
                // both vector sub-blocks with the cube before advancing to the
                // next task so a following non-empty task cannot inherit a
                // launch-skewed pipeline state.
                AlignVectorStartFromCube();
                if constexpr (PACKED_QUERY) {
                    if (task + 1 < coreTaskEnd) {
                        LoadQuery(task + 1, (task + 1) & 1);
                    }
                }
                continue;
            }
            uint32_t qBuf = task & 1;
            if constexpr (!PACKED_QUERY) LoadQuery(task, qBuf);
            for (uint32_t group = 0; group < groups; ++group, ++item) {
                uint32_t groupStart = group * GROUP_CHUNKS;
                uint32_t groupChunks = chunks - groupStart;
                if (groupChunks > GROUP_CHUNKS) groupChunks = GROUP_CHUNKS;
                uint32_t groupSlot = item % GROUP_SLOTS;
                uint32_t prefetchedValueBuf = 0;
                bool canPrefetchValue = hasPrev;
                CubeMm1Group(qBuf, groupSlot, groupChunks,
                             canPrefetchValue, prevSlot, prefetchedValueBuf);
                if (hasPrev) {
                    CubeMm2Group(prevTask, prevSlot, prevChunks, prevDirect,
                                 canPrefetchValue,
                                 prefetchedValueBuf);
                }
                prevTask = task;
                prevSlot = groupSlot;
                prevChunks = groupChunks;
                prevDirect = groups == 1;
                hasPrev = true;
            }
            if constexpr (PACKED_QUERY) {
                if (task + 1 < coreTaskEnd) {
                    LoadQuery(task + 1, (task + 1) & 1);
                }
            }
        }
        if (hasPrev) {
            CubeMm2Group(prevTask, prevSlot, prevChunks, prevDirect);
        }
        DrainL0AB();
    }

    __aicore__ inline void RowOp(LocalTensor<float> dst, LocalTensor<float> src,
                                 LocalTensor<float> rows, bool divide,
                                 uint32_t cols = LATENT) {
        BinaryRepeatParams p = {};
        p.src0BlkStride = 1;
        p.src1BlkStride = 0;
        p.dstBlkStride = 1;
        p.src0RepStride = 8;
        p.src1RepStride = 0;
        p.dstRepStride = 8;
        for (uint32_t h = 0; h < HALF_HEADS; ++h) {
            if (divide)
                Div(dst[h * cols], src[h * cols], rows[h * 8],
                    64, cols / 64, p);
            else
                Mul(dst[h * cols], src[h * cols], rows[h * 8],
                    64, cols / 64, p);
        }
    }

    __aicore__ inline void MaskInvalidSegment(uint32_t segmentStart,
                                              uint32_t validRows,
                                              uint32_t groupTokens) {
        if (validRows >= CHUNK / 2) return;
        uint32_t start = segmentStart + validRows;
        uint32_t end = segmentStart + CHUNK / 2;
        uint32_t alignedStart = start / 8 * 8;
        uint32_t firstBit = start - alignedStart;
        uint32_t span = end - alignedStart;
        uint64_t endBits = span == 64 ? UINT64_MAX : ((1ULL << span) - 1);
        uint64_t beforeBits = firstBit == 0 ? 0 : ((1ULL << firstBit) - 1);
        uint64_t mask[2] = {endBits & ~beforeBits, 0};
        // One repeat per head. groupTokens is always a multiple of one 32B
        // FP32 block, so every repeat starts aligned while the bit mask covers
        // the exact (possibly unaligned) invalid suffix.
        Duplicate(logitsUb[alignedStart], -2e38f, mask, HALF_HEADS,
                  1, groupTokens / 8);
    }

    __aicore__ inline void MaskRows(uint32_t rows0, uint32_t rows1,
                                    uint32_t chunkInGroup,
                                    uint32_t groupTokens) {
        uint32_t chunkStart = chunkInGroup * CHUNK;
        MaskInvalidSegment(chunkStart, rows0, groupTokens);
        MaskInvalidSegment(chunkStart + CHUNK / 2, rows1, groupTokens);
    }

    __aicore__ inline void CopyGatherRow(LocalTensor<T> key,
                                          LocalTensor<T> rope,
                                          uint32_t dstRow,
                                          uint64_t srcRow) {
        const uint64_t src = srcRow * WIDTH;
        DataCopy(key[dstRow * LATENT], keyGm[src], LATENT);
        DataCopy(rope[dstRow * ROPE], keyGm[src + LATENT], ROPE);
    }


    // The packed C8 row is [512 x int8 C-KV, 64 x bf16 RoPE,
    // 4 x fp32 per-128 scales] = 656 bytes (672 bytes after UB padding).
    // Expand up to thirteen rows from the tail after gathering the complete
    // part.  The Softmax temporary is dead during Gather and holds the FP16
    // expansion plus scales; the high 28 KiB of logitsBuf is then dedicated
    // to FP32.  This raises one exact dequant batch from 8 to 13 rows without
    // increasing UB, cutting Cast/Mul/Brcb launch and barrier count. All
    // packed bytes for a batch are consumed before its BF16 rows are written,
    // which keeps in-place expansion overlap-safe.  Scales are compacted and
    // broadcast in one vector launch.  The unused high 28 KiB of logitsBuf is
    // disjoint from both 36 KiB gather outputs, allowing part 1 MTE2 to overlap
    // part 0 dequantization without an artificial V->MTE2 dependency.
    // Scale multiplication remains FP32 and CAST_RINT matches the reference
    // dequantization exactly.
    __aicore__ inline void DequantGatherRows(LocalTensor<T> buffer,
                                              LocalTensor<uint8_t> scratch,
                                              uint32_t rows) {
        LocalTensor<int8_t> raw =
            buffer.template ReinterpretCast<int8_t>();
        LocalTensor<T> key = buffer;
        LocalTensor<T> rope = buffer[GATHER_PART * LATENT];
        LocalTensor<half> fp16 =
            softmaxUb.template ReinterpretCast<half>();
        constexpr uint32_t fp16Bytes =
            INT8_DEQUANT_ROWS * LATENT * sizeof(half);
        LocalTensor<float> fp32 = scratch.template ReinterpretCast<float>();
        LocalTensor<float> scaleUb = softmaxUb[fp16Bytes]
            .template ReinterpretCast<float>();
        constexpr uint32_t scaleBytes =
            INT8_DEQUANT_ROWS * 32 * sizeof(float);
        LocalTensor<float> compactScale = softmaxUb[
            fp16Bytes + scaleBytes]
            .template ReinterpretCast<float>();

        SetFlag<HardEvent::MTE2_V>(addressMte2ToV);
        WaitFlag<HardEvent::MTE2_V>(addressMte2ToV);
        uint32_t remaining = rows;
        while (remaining != 0) {
            uint32_t batch = remaining > INT8_DEQUANT_ROWS ?
                INT8_DEQUANT_ROWS : remaining;
            uint32_t start = remaining - batch;
            LocalTensor<int8_t> packed =
                raw[start * INT8_PACKED_ROW_ALIGN];

            for (uint32_t tile = 0; tile < 4; ++tile) {
                Cast(fp16[tile * 128], packed[tile * 128],
                     RoundMode::CAST_NONE, 128,
                     static_cast<uint8_t>(batch), {1, 1, 32, 21});
            }

            LocalTensor<T> packedRope =
                packed[512].template ReinterpretCast<T>();
            Copy(rope[start * ROPE], packedRope, ROPE,
                 static_cast<uint8_t>(batch), {1, 1, 4, 21});

            // Each packed scale block has four FP32 scales followed by 16
            // bytes of padding.  Compact the strided row blocks first so Brcb
            // can broadcast every row in one launch.
            LocalTensor<float> packedScale =
                packed[640].template ReinterpretCast<float>();
            DataCopyParams scaleCopy;
            scaleCopy.blockCount = static_cast<uint16_t>(batch);
            scaleCopy.blockLen = 1;
            scaleCopy.srcStride = 20;
            scaleCopy.dstStride = 0;
            DataCopy(compactScale, packedScale, scaleCopy);
            PipeBarrier<PIPE_V>();
            Brcb(scaleUb, compactScale, static_cast<uint8_t>(batch),
                 {1, 4});
            PipeBarrier<PIPE_V>();

            for (uint32_t rowBase = 0; rowBase < batch;
                 rowBase += INT8_FP32_ROWS) {
                uint32_t dealRows = batch - rowBase;
                if (dealRows > INT8_FP32_ROWS) {
                    dealRows = INT8_FP32_ROWS;
                }
                Cast(fp32, fp16[rowBase * LATENT],
                     RoundMode::CAST_NONE, dealRows * LATENT);
                PipeBarrier<PIPE_V>();
                for (uint32_t halfTile = 0; halfTile < 2; ++halfTile) {
                    Mul(fp32[halfTile * 64], fp32[halfTile * 64],
                        scaleUb[rowBase * 32], 64,
                        static_cast<uint8_t>(4 * dealRows),
                        {1, 1, 0, 16, 16, 1});
                }
                PipeBarrier<PIPE_V>();
                Cast(key[(start + rowBase) * LATENT], fp32,
                     RoundMode::CAST_RINT, dealRows * LATENT);
                PipeBarrier<PIPE_V>();
            }
            remaining = start;
        }
    }

    __aicore__ inline uint32_t PrepareGatherAddresses(
        uint32_t task, uint32_t sub) {
        uint64_t topkBase = static_cast<uint64_t>(task) * sparseCount;
        uint32_t line = task & ~15U;
        DataCacheCleanAndInvalid<
            int32_t, CacheLine::SINGLE_CACHE_LINE,
            DcciDst::CACHELINE_OUT>(validChunksGm[line]);
        int32_t rawValid = validChunksGm.GetValue(task);
        if (rawValid <= 0) return 0;
        uint32_t validCount = static_cast<uint32_t>(rawValid);
        if (validCount > sparseCount) validCount = sparseCount;
        uint32_t chunks = (validCount + CHUNK - 1) / CHUNK;

        const uint32_t scanCount = chunks * CHUNK;
        LocalTensor<int32_t> pageTableUb =
            logitsUb.template ReinterpretCast<int32_t>();
        LocalTensor<int32_t> blockIdUb = pageTableUb[MAX_BLOCK_TABLE];
        addressTmpUb = blockIdUb[ADDRESS_COUNT];
        LocalTensor<int32_t> topkUb = addressTmpUb[ADDRESS_COUNT];
        DataCopy(topkUb, indicesGm[topkBase], scanCount);
        DataCopyPadExtParams<int32_t> noPad{false, 0, 0, 0};
        DataCopyExtParams pageCopy{
            1, static_cast<uint32_t>(maxBlocks * sizeof(int32_t)),
            0, 0, 0};
        DataCacheCleanAndInvalid<
            int32_t, CacheLine::SINGLE_CACHE_LINE,
            DcciDst::CACHELINE_OUT>(querySeqGm[line]);
        uint32_t batch = static_cast<uint32_t>(querySeqGm.GetValue(task));
        DataCopyPad(pageTableUb,
                    blocksGm[static_cast<uint64_t>(batch) * maxBlocks],
                    pageCopy, noPad);
        SetFlag<HardEvent::MTE2_V>(addressMte2ToV);
        WaitFlag<HardEvent::MTE2_V>(addressMte2ToV);

        // Pattern 7 keeps all 64 int32 lanes.  A 16-block source repeat
        // stride skips the other AIV's half, packing this AIV's addresses
        // directly from the sole topk UB copy into addressUb.
        GatherMaskParams addressParams{
            1, static_cast<uint16_t>(chunks),
            static_cast<uint16_t>(CHUNK * sizeof(int32_t) / 32), 0};
        uint64_t addressCount = 0;
        GatherMask(addressUb, topkUb[sub * (CHUNK / 2)],
                   static_cast<uint8_t>(7), false, 0,
                   addressParams, addressCount);
        PipeBarrier<PIPE_V>();

        uint32_t fullChunks = validCount / CHUNK;
        uint32_t tail = validCount - fullChunks * CHUNK;
        uint32_t valid = fullChunks * (CHUNK / 2);
        uint32_t subStart = sub * (CHUNK / 2);
        if (tail > subStart) {
            uint32_t subValid = tail - subStart;
            if (subValid > CHUNK / 2) subValid = CHUNK / 2;
            valid += subValid;
        }
        if (valid != 0) {
            // Convert sequence-logical token rows to gathered-PA physical
            // rows in place.  This removes the former Q*2048 framework
            // pointwise chain and keeps mapping under the AIV gather stage.
            ShiftRight(addressTmpUb, addressUb,
                       static_cast<int32_t>(7), valid);
            PipeBarrier<PIPE_V>();
            Muls(addressTmpUb, addressTmpUb,
                 static_cast<int32_t>(sizeof(int32_t)), valid);
            PipeBarrier<PIPE_V>();
            Gather(blockIdUb, pageTableUb,
                   addressTmpUb.template ReinterpretCast<uint32_t>(),
                   0, valid);
            PipeBarrier<PIPE_V>();
            Muls(addressTmpUb, addressTmpUb,
                 static_cast<int32_t>(32), valid);
            Muls(blockIdUb, blockIdUb,
                 static_cast<int32_t>(PAGE), valid);
            PipeBarrier<PIPE_V>();
            Sub(addressUb, addressUb, addressTmpUb, valid);
            PipeBarrier<PIPE_V>();
            Add(addressUb, addressUb, blockIdUb, valid);
            PipeBarrier<PIPE_V>();
        }
        SetFlag<HardEvent::V_S>(addressVToS);
        WaitFlag<HardEvent::V_S>(addressVToS);
        return chunks;
    }

    __aicore__ inline void GatherKvPart(
        uint32_t chunk, uint32_t slot, uint32_t sub, uint32_t part,
        bool drainVector,
        uint32_t &rows) {
        if (drainVector) {
            SetFlag<HardEvent::V_MTE2>(gatherVToMte2);
            WaitFlag<HardEvent::V_MTE2>(gatherVToMte2);
        }
        LocalTensor<T> buffer = part == 0 ? gatherUb :
            logitsUb.template ReinterpretCast<T>();
        LocalTensor<T> key = buffer;
        LocalTensor<T> rope = buffer[GATHER_PART * LATENT];
        uint32_t partRows = 0;
        uint32_t cursor = chunk * (CHUNK / 2) + part * GATHER_PART;
        uint32_t end = cursor + GATHER_PART;
        while (cursor < end) {
            int32_t row = addressUb.GetValue(cursor++);
            if (row < 0) continue;
            if constexpr (KV_INT8) {
                LocalTensor<int8_t> raw =
                    buffer.template ReinterpretCast<int8_t>();
                if (packedRowBytes == INT8_PACKED_ROW_ALIGN) {
                    DataCopy(raw[partRows * INT8_PACKED_ROW_ALIGN],
                             packedKeyGm[static_cast<uint64_t>(row) *
                                         packedRowBytes],
                             INT8_PACKED_ROW_ALIGN);
                } else {
                    DataCopyExtParams copy{
                        1, INT8_PACKED_PAYLOAD_BYTES, 0, 0, 0};
                    DataCopyPadExtParams<int8_t> pad{
                        true, 0,
                        INT8_PACKED_ROW_ALIGN - INT8_PACKED_PAYLOAD_BYTES, 0};
                    DataCopyPad(raw[partRows * INT8_PACKED_ROW_ALIGN],
                                packedKeyGm[static_cast<uint64_t>(row) *
                                            packedRowBytes],
                                copy, pad);
                }
            } else {
                CopyGatherRow(key, rope, partRows,
                              static_cast<uint32_t>(row));
            }
            ++partRows;
        }
        if constexpr (KV_INT8) {
            if (partRows != 0) {
                LocalTensor<uint8_t> scratch =
                    logitsUb.template ReinterpretCast<uint8_t>()[
                        GATHER_DATA_BYTES];
                DequantGatherRows(buffer, scratch, partRows);
            }
        }
        uint32_t padding = part == 1 ? CHUNK / 2 - rows - partRows : 0;
        GlobalTensor<T> packed = packedKvGm[slot * KV_BUF];
        uint32_t dstRow = sub * (CHUNK / 2) + rows;
        bool wrote = false;
        if (partRows != 0) {
            if constexpr (KV_INT8) {
                SetFlag<HardEvent::V_MTE3>(gatherVToMte3);
                WaitFlag<HardEvent::V_MTE3>(gatherVToMte3);
            } else {
                SetFlag<HardEvent::MTE2_MTE3>(0);
                WaitFlag<HardEvent::MTE2_MTE3>(0);
            }
            DataCopy(packed[dstRow * LATENT], key, partRows * LATENT);
            DataCopy(packed[CHUNK * LATENT + dstRow * ROPE],
                     rope, partRows * ROPE);
            wrote = true;
        }
        for (uint32_t done = 0; done < padding; done += GATHER_PART) {
            if (wrote) {
                SetFlag<HardEvent::MTE3_V>(gatherMte3ToV);
                WaitFlag<HardEvent::MTE3_V>(gatherMte3ToV);
            }
            uint32_t zeroRows = padding - done;
            if (zeroRows > GATHER_PART) zeroRows = GATHER_PART;
            Duplicate(key, static_cast<T>(0), zeroRows * LATENT);
            Duplicate(rope, static_cast<T>(0), zeroRows * ROPE);
            SetFlag<HardEvent::V_MTE3>(gatherVToMte3);
            WaitFlag<HardEvent::V_MTE3>(gatherVToMte3);
            uint32_t zeroDst = dstRow + partRows + done;
            DataCopy(packed[zeroDst * LATENT], key, zeroRows * LATENT);
            DataCopy(packed[CHUNK * LATENT + zeroDst * ROPE],
                     rope, zeroRows * ROPE);
            wrote = true;
        }
        rows += partRows;
    }

    __aicore__ inline void GatherKv(uint32_t chunk, uint32_t slot,
                                    uint32_t sub, bool drainVector
        ) {
        uint32_t rows = 0;
        // BF16 first touches logitsUb in part 1, while INT8 uses it as part
        // 0's dequant scratch.  Drain the prior vector consumer exactly at
        // the first access in either path.
        GatherKvPart(chunk, slot, sub, 0,
                     KV_INT8 && drainVector, rows);
        GatherKvPart(chunk, slot, sub, 1,
                     !KV_INT8 && drainVector, rows);
        // Packed KV and row-count writes share one MTE3 queue.  Waiting on
        // the row-count tail below also completes both packed writes.
        LocalTensor<int32_t> count =
            softmaxUb[16 * 1024 - 128].template ReinterpretCast<int32_t>();
        count.SetValue(0, rows);
        SetFlag<HardEvent::S_MTE3>(countSToMte3);
        WaitFlag<HardEvent::S_MTE3>(countSToMte3);
        DataCopy(rowsGm[(slot * 2 + sub) * 16], count, 8);
        SetFlag<HardEvent::MTE3_S>(taskMte3ToS);
        WaitFlag<HardEvent::MTE3_S>(taskMte3ToS);
        CrossCoreSetFlag<0x2, PIPE_MTE3>(SYNC_KV);
    }

    __aicore__ inline void InitVector(TPipe *pipe) {
        gatherMte3ToV = pipe->AllocEventID<HardEvent::MTE3_V>();
        gatherVToMte3 = pipe->AllocEventID<HardEvent::V_MTE3>();
        gatherVToMte2 = pipe->AllocEventID<HardEvent::V_MTE2>();
        probMte3ToMte2 = pipe->AllocEventID<HardEvent::MTE3_MTE2>();
        countSToMte3 = pipe->AllocEventID<HardEvent::S_MTE3>();
        countMte2ToS = pipe->AllocEventID<HardEvent::MTE2_S>();
        taskMte3ToS = pipe->AllocEventID<HardEvent::MTE3_S>();
        addressMte2ToV = pipe->AllocEventID<HardEvent::MTE2_V>();
        addressVToS = pipe->AllocEventID<HardEvent::V_S>();
        pipe->InitBuffer(logitsBuf, HALF_HEADS * GROUP_TOKENS * sizeof(float));
        pipe->InitBuffer(accumBuf, HALF_HEADS * LATENT * sizeof(float));
        pipe->InitBuffer(statsBuf, 512 * sizeof(float));
        pipe->InitBuffer(softmaxBuf, CAST_HEADS * GROUP_TOKENS * sizeof(T));
        pipe->InitBuffer(gatherBuf,
                         GATHER_DATA_BYTES + ADDRESS_BYTES);
        logitsUb = logitsBuf.Get<float>();
        mm2Ub = logitsUb;
        accumUb = accumBuf.Get<float>();
        statsUb = statsBuf.Get<float>();
        softmaxUb = softmaxBuf.Get<uint8_t>();
        LocalTensor<uint8_t> gatherRaw = gatherBuf.Get<uint8_t>();
        gatherUb = gatherRaw.template ReinterpretCast<T>();
        addressUb = gatherRaw[GATHER_DATA_BYTES]
                        .template ReinterpretCast<int32_t>();
        // Keep the 16 KiB SoftmaxFlashV2 scratch isolated from probability
        // conversion.  Reuse the already-drained gather buffer instead of
        // overwriting all of the scratch with a 16-head Cast.
        probUb = gatherUb;
    }

    __aicore__ inline void FreeVectorEvents(TPipe *pipe) {
        pipe->ReleaseEventID<HardEvent::MTE3_V>(gatherMte3ToV);
        pipe->ReleaseEventID<HardEvent::V_MTE3>(gatherVToMte3);
        pipe->ReleaseEventID<HardEvent::V_MTE2>(gatherVToMte2);
        pipe->ReleaseEventID<HardEvent::MTE3_MTE2>(probMte3ToMte2);
        pipe->ReleaseEventID<HardEvent::S_MTE3>(countSToMte3);
        pipe->ReleaseEventID<HardEvent::MTE2_S>(countMte2ToS);
        pipe->ReleaseEventID<HardEvent::MTE3_S>(taskMte3ToS);
        pipe->ReleaseEventID<HardEvent::MTE2_V>(addressMte2ToV);
        pipe->ReleaseEventID<HardEvent::V_S>(addressVToS);
    }

    __aicore__ inline void AlignVectorStart() {
        CrossCoreSetFlag<0x2, PIPE_MTE3>(SYNC_KV);
        CrossCoreWaitFlag(SYNC_MM1);
    }

    __aicore__ inline void StartVectorWithPublishedRange(
        uint32_t core, uint32_t subBlock) {
        PrepareTaskRange(core, subBlock);
        CrossCoreSetFlag<0x2, PIPE_MTE3>(SYNC_KV);
        CrossCoreWaitFlag(SYNC_MM1);
        // AIV0 owns the calculation; AIV1 consumes the exact same record.
        if (subBlock != 0) LoadTaskRange(core);
    }

    __aicore__ inline void PublishTask(uint32_t task, uint32_t chunks,
                                       uint32_t sub) {
        LocalTensor<int32_t> count =
            softmaxUb[16 * 1024 - 256].template ReinterpretCast<int32_t>();
        count.SetValue(0, static_cast<int32_t>(chunks));
        SetFlag<HardEvent::S_MTE3>(countSToMte3);
        WaitFlag<HardEvent::S_MTE3>(countSToMte3);
        DataCopy(taskGm[(task * 2 + sub) * 16], count, 8);
        CrossCoreSetFlag<0x2, PIPE_MTE3>(SYNC_TASK);
        SetFlag<HardEvent::MTE3_S>(taskMte3ToS);
        WaitFlag<HardEvent::MTE3_S>(taskMte3ToS);
    }

    __aicore__ inline void StoreLse(uint32_t task, uint32_t sub,
                                    LocalTensor<float> sum,
                                    LocalTensor<float> max) {
        if (!returnLse) return;
        // Keep max/sum in the native contiguous [N2,T,G] layout even when O
        // is written directly in KVP communication order.  Writing
        // [N2,G,T] here turns each token into 32 four-byte strided MTE3
        // transfers plus scalar UB staging.  The tiny LSE layout conversion
        // runs on the independent LSE stream and is hidden by the much longer
        // O AllToAll, while this contiguous store shortens SFAD itself.
        const uint64_t lse = static_cast<uint64_t>(task) * HEADS +
                             sub * HALF_HEADS;
        SetFlag<HardEvent::V_MTE3>(5);
        WaitFlag<HardEvent::V_MTE3>(5);
        DataCopy(maxGm[lse], max, HALF_HEADS);
        DataCopy(sumGm[lse], sum, HALF_HEADS);
        SetFlag<HardEvent::MTE3_V>(5);
        WaitFlag<HardEvent::MTE3_V>(5);
    }

    __aicore__ inline void StoreOutput(uint32_t task, uint32_t sub,
                                       uint32_t valid) {
        LocalTensor<float> sum = statsUb;
        LocalTensor<float> max = statsUb[32];
        LocalTensor<float> rows = statsUb[STATS_ROWS];
        if (valid == 0) {
            Duplicate(accumUb, 0.0f, HALF_HEADS * LATENT);
            // A finite neutral LSE is required by the online KVP merge.  With
            // sum=0 every rank-empty row becomes -inf and the rank softmax is
            // NaN.  max=-2e38,sum=1 keeps this rank's weight at zero whenever
            // another rank is valid, and yields a finite uniform weighting of
            // zero outputs when all ranks are empty.
            Duplicate(sum, 1.0f, HALF_HEADS);
            Duplicate(max, -2e38f, HALF_HEADS);
        }
        if (valid != 0) {
            Brcb(rows, sum, 4, {1, 8});
            PipeBarrier<PIPE_V>();
            RowOp(accumUb, accumUb, rows, true);
        }
        PipeBarrier<PIPE_V>();
        // Match the existing Prefill SFA semantic boundary: attention is
        // accumulated and normalized in FP32, then rounded once to BF16
        // before W_UV.  gatherUb is dead after the final KV gather.
        LocalTensor<T> rounded = gatherUb;
        Cast(rounded, accumUb, RoundMode::CAST_RINT,
             HALF_HEADS * LATENT);
        PipeBarrier<PIPE_V>();
        // MTE3 consumes the BF16 cast, not just the FP32 normalization.
        // A V-only barrier cannot stop DMA from reading the previous gather
        // or probability contents while Cast is still writing this buffer.
        SetFlag<HardEvent::V_MTE3>(4);
        WaitFlag<HardEvent::V_MTE3>(4);
        if constexpr (COMM_OUTPUT) {
            // Scatter contiguous head rows directly to [H,T,D], removing the
            // standalone [T,H,D] -> [H,T,D] materialization before W_UV.
            // The extended DMA descriptor also supports T > 2048 and avoids
            // the intermittent small-T loss observed with legacy strides.
            const uint64_t outBase =
                static_cast<uint64_t>(sub) * HALF_HEADS * totalQuery * LATENT +
                static_cast<uint64_t>(task) * LATENT;
            DataCopyExtParams outputCopy{
                static_cast<uint16_t>(HALF_HEADS),
                static_cast<uint32_t>(LATENT * sizeof(T)), 0,
                static_cast<uint32_t>((totalQuery - 1) * LATENT * sizeof(T)),
                0};
            DataCopyPad(outputGm[outBase], rounded, outputCopy);
        } else {
            const uint64_t outBase =
                static_cast<uint64_t>(task) * HEADS * LATENT +
                sub * HALF_HEADS * LATENT;
            DataCopy(outputGm[outBase], rounded, HALF_HEADS * LATENT);
        }
        SetFlag<HardEvent::MTE3_V>(4);
        WaitFlag<HardEvent::MTE3_V>(4);
        StoreLse(task, sub, sum, max);
    }

    __aicore__ inline uint32_t SoftmaxGroup(
        uint32_t group, uint32_t groupSlot, uint32_t groupChunks, uint32_t sub,
        LocalTensor<float> sum, LocalTensor<float> max,
        LocalTensor<float> oldSum, LocalTensor<float> oldMax,
        bool directOutput
        ) {
        CrossCoreWaitFlag(SYNC_MM1 + groupSlot);
        uint32_t groupTokens = groupChunks * CHUNK;
        uint32_t valid = 0;
        uint32_t rows0[GROUP_CHUNKS] = {};
        uint32_t rows1[GROUP_CHUNKS] = {};
        LocalTensor<int32_t> count =
            softmaxUb.template ReinterpretCast<int32_t>();
        DataCopy(count, rowsGm[
            groupSlot * GROUP_CHUNKS * 2 * 16],
            groupChunks * 2 * 16);
        SetFlag<HardEvent::MTE2_S>(countMte2ToS);
        WaitFlag<HardEvent::MTE2_S>(countMte2ToS);
        for (uint32_t i = 0; i < groupChunks; ++i) {
            rows0[i] = count.GetValue(i * 2 * 16);
            rows1[i] = count.GetValue((i * 2 + 1) * 16);
            valid += rows0[i] + rows1[i];
        }

        DataCopyParams logitsCopy{
            HALF_HEADS,
            static_cast<uint16_t>(groupTokens * sizeof(float) / 32),
            static_cast<uint16_t>((GROUP_TOKENS - groupTokens) * sizeof(float) / 32),
            0};
        DataCopy(logitsUb,
                 mm1Gm[groupSlot * HEADS * GROUP_TOKENS +
                       sub * HALF_HEADS * GROUP_TOKENS], logitsCopy);
        SetFlag<HardEvent::MTE2_V>(2);
        WaitFlag<HardEvent::MTE2_V>(2);
        bool vectorPending = false;
        if (scale != 1.0f) {
            Muls(logitsUb, logitsUb, scale, HALF_HEADS * groupTokens);
            vectorPending = true;
        }
        for (uint32_t i = 0; i < groupChunks; ++i) {
            if (rows0[i] + rows1[i] < CHUNK) {
                if (vectorPending) {
                    PipeBarrier<PIPE_V>();
                    vectorPending = false;
                }
                MaskRows(rows0[i], rows1[i], i, groupTokens);
                vectorPending = true;
            }
        }
        if (vectorPending) PipeBarrier<PIPE_V>();

        LocalTensor<float> expMax = statsUb[STATS_EXP_MAX];
        SoftMaxShapeInfo shape{
            HALF_HEADS, groupTokens, HALF_HEADS, groupTokens};
        if (group == 0) {
            SoftMaxTiling tiling = SoftMaxFlashV2TilingFunc(
                shape, sizeof(float), sizeof(float), 16 * 1024, false, true);
            SoftmaxFlashV2<float, false, true, true, false,
                PREFILL_SOFTMAX_CFG>(
                logitsUb, sum, max, logitsUb, expMax, oldSum, oldMax,
                softmaxUb, tiling, shape);
        } else {
            SoftMaxTiling tiling = SoftMaxFlashV2TilingFunc(
                shape, sizeof(float), sizeof(float), 16 * 1024, true, true);
            SoftmaxFlashV2<float, true, true, true, false,
                PREFILL_SOFTMAX_CFG>(
                logitsUb, sum, max, logitsUb, expMax, oldSum, oldMax,
                softmaxUb, tiling, shape);
        }
        PipeBarrier<PIPE_V>();
        if (directOutput && valid != 0) {
            LocalTensor<float> denominator = statsUb[STATS_ROWS];
            Brcb(denominator, sum, 4, {1, 8});
            PipeBarrier<PIPE_V>();
            RowOp(logitsUb, logitsUb, denominator, true, groupTokens);
            PipeBarrier<PIPE_V>();
        }
        for (uint32_t wave = 0; wave < HALF_HEADS / CAST_HEADS; ++wave) {
            uint32_t head = sub * HALF_HEADS + wave * CAST_HEADS;
            Cast(probUb, logitsUb[wave * CAST_HEADS * groupTokens],
                 RoundMode::CAST_ROUND, CAST_HEADS * groupTokens);
            PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_MTE3>(4);
            WaitFlag<HardEvent::V_MTE3>(4);
            DataCopyParams probCopy{
                CAST_HEADS,
                static_cast<uint16_t>(groupTokens * sizeof(T) / 32),
                0,
                static_cast<uint16_t>((GROUP_TOKENS - groupTokens) * sizeof(T) / 32)};
            DataCopy(probGm[groupSlot * HEADS * GROUP_TOKENS +
                            head * GROUP_TOKENS], probUb, probCopy);
            // Only an intermediate wave needs MTE3 -> V before the next Cast
            // overwrites the probability staging buffer.  The final wave is
            // closed by the dependency selected below.
            if (wave + 1 < HALF_HEADS / CAST_HEADS) {
                SetFlag<HardEvent::MTE3_V>(4);
                WaitFlag<HardEvent::MTE3_V>(4);
            }
        }
        // The next Gather writes probUb/gatherUb from MTE2.  Close the
        // probability MTE3 read before that shared buffer is reused.
        SetFlag<HardEvent::MTE3_MTE2>(probMte3ToMte2);
        WaitFlag<HardEvent::MTE3_MTE2>(probMte3ToMte2);
        if (!directOutput) {
            Adds(oldSum, sum, 0.0f, HALF_HEADS);
            Adds(oldMax, max, 0.0f, HALF_HEADS);
            Adds(statsUb[STATS_GROUP_SCALE + groupSlot * HALF_HEADS],
                 expMax, 0.0f,
                 HALF_HEADS);
        }
        CrossCoreSetFlag<0x2, PIPE_MTE3>(ProbFlag(groupSlot));
        return valid;
    }

    __aicore__ inline uint32_t SoftmaxFinishedGroup(
        const GroupInfo &info, uint32_t sub,
        LocalTensor<float> sum, LocalTensor<float> max,
        LocalTensor<float> oldSum, LocalTensor<float> oldMax) {
        bool directOutput = IsDirectOutput(info);
        uint32_t valid = SoftmaxGroup(
            info.group, info.slot, info.chunks, sub,
            sum, max, oldSum, oldMax, directOutput
            );
        if (directOutput && valid != 0)
            StoreLse(info.task, sub, sum, max);
        return valid;
    }

    __aicore__ inline void MergeGroup(uint32_t group, uint32_t groupSlot,
                                      uint32_t sub, LocalTensor<float> rows) {
        LocalTensor<float> expMax =
            statsUb[STATS_GROUP_SCALE + groupSlot * HALF_HEADS];
        uint64_t subOffset = sub * HALF_HEADS * LATENT;
        LocalTensor<float> dst = group == 0 ? accumUb : mm2Ub;
        DataCopy(dst,
                 mm2Gm[groupSlot * HEADS * LATENT + subOffset],
                 HALF_HEADS * LATENT);
        SetFlag<HardEvent::MTE2_V>(2);
        WaitFlag<HardEvent::MTE2_V>(2);
        if (group != 0) {
            Brcb(rows, expMax, 4, {1, 8});
            PipeBarrier<PIPE_V>();
            RowOp(accumUb, accumUb, rows, false);
            PipeBarrier<PIPE_V>();
            Add(accumUb, accumUb, mm2Ub, HALF_HEADS * LATENT);
            // logitsUb aliases mm2Ub.  The next SoftmaxGroup starts an MTE2
            // copy into that storage, so finish the vector read before the
            // shared buffer is reused by MTE2.
            SetFlag<HardEvent::V_MTE2>(gatherVToMte2);
            WaitFlag<HardEvent::V_MTE2>(gatherVToMte2);
        }
    }

    __aicore__ inline void GatherGroup(uint32_t groupStart,
                                       uint32_t groupSlot,
                                       uint32_t groupChunks, uint32_t sub) {
        for (uint32_t i = 0; i < groupChunks; ++i) {
            GatherKv(groupStart + i, groupSlot * GROUP_CHUNKS + i,
                     sub, i == 0
                     );
        }
    }

    __aicore__ inline uint32_t MergeFinishedGroup(
        const GroupInfo &info, uint32_t sub, uint32_t valid,
        LocalTensor<float> rows) {
        CrossCoreWaitFlag(SYNC_MM2 + info.slot);
        if (IsDirectOutput(info)) {
            if (valid == 0) StoreOutput(info.task, sub, 0);
            return 0;
        }
        MergeGroup(info.group, info.slot, sub, rows);
        if (info.last) {
            StoreOutput(info.task, sub, valid);
            return 0;
        }
        return valid;
    }

    __aicore__ inline void ProcessVector() {
        uint32_t subBlock = GetBlockIdx() & 1;
        uint32_t core = GetBlockIdx() >> 1;
        if (core >= usedCores) return;
        LocalTensor<float> sum = statsUb;
        LocalTensor<float> max = statsUb[32];
        LocalTensor<float> oldSum = statsUb[64];
        LocalTensor<float> oldMax = statsUb[96];
        LocalTensor<float> rows = statsUb[STATS_ROWS];
        uint32_t item = 0;
        GroupInfo prev1 = {};
        GroupInfo prev2 = {};
        uint32_t valid = 0;
        bool hasPrev1 = false;
        bool hasPrev2 = false;
        for (uint32_t task = coreTaskBegin; task < coreTaskEnd; ++task) {
            uint32_t chunks = PrepareGatherAddresses(task, subBlock);
            currentTask = task;
            PublishTask(task, chunks, subBlock);
            uint32_t groups = (chunks + GROUP_CHUNKS - 1) / GROUP_CHUNKS;
            if (groups == 0) {
                if (hasPrev2) {
                    valid = MergeFinishedGroup(prev2, subBlock, valid, rows);
                }
                if (hasPrev1) {
                    valid += SoftmaxFinishedGroup(
                        prev1, subBlock, sum, max, oldSum, oldMax);
                    valid = MergeFinishedGroup(prev1, subBlock, valid, rows);
                }
                hasPrev1 = hasPrev2 = false;
                StoreOutput(task, subBlock, 0);
                AlignVectorStart();
                    continue;
            }
            for (uint32_t group = 0; group < groups; ++group, ++item) {
                uint32_t groupStart = group * GROUP_CHUNKS;
                uint32_t groupChunks = chunks - groupStart;
                if (groupChunks > GROUP_CHUNKS) groupChunks = GROUP_CHUNKS;
                uint32_t groupSlot = item % GROUP_SLOTS;
                GatherGroup(groupStart, groupSlot, groupChunks, subBlock);
                bool softmaxFirst = hasPrev1 &&
                    (!hasPrev2 || prev1.task == prev2.task);
                if (hasPrev2 && !softmaxFirst) {
                    valid = MergeFinishedGroup(prev2, subBlock, valid, rows);
                }
                if (hasPrev1) {
                    valid += SoftmaxFinishedGroup(
                        prev1, subBlock, sum, max, oldSum, oldMax);
                }
                if (hasPrev2 && softmaxFirst) {
                    valid = MergeFinishedGroup(prev2, subBlock, valid, rows);
                }
                prev2 = prev1;
                hasPrev2 = hasPrev1;
                prev1 = {task, group, groupSlot, groupChunks,
                         group + 1 == groups};
                hasPrev1 = true;
            }
        }
        bool softmaxFirst = hasPrev1 &&
            (!hasPrev2 || prev1.task == prev2.task);
        if (hasPrev2 && !softmaxFirst) {
            valid = MergeFinishedGroup(prev2, subBlock, valid, rows);
        }
        if (hasPrev1) {
            valid += SoftmaxFinishedGroup(
                prev1, subBlock, sum, max, oldSum, oldMax);
        }
        if (hasPrev2 && softmaxFirst) {
            valid = MergeFinishedGroup(prev2, subBlock, valid, rows);
        }
        if (hasPrev1) {
            MergeFinishedGroup(prev1, subBlock, valid, rows);
        }
    }

public:
    __aicore__ inline void Init(
        GM_ADDR query, GM_ADDR gatheredKvPages,
        GM_ADDR logicalSparseRows, GM_ADDR scanCount,
        GM_ADDR pageTable, GM_ADDR querySeq,
        GM_ADDR output, GM_ADDR workspace,
        uint32_t totalQ, uint32_t sparseK,
        uint32_t pageSize, uint32_t cores, uint32_t maxBlocks,
        float scaleValue, TPipe *pipe) {
        queryGm.SetGlobalBuffer((__gm__ T *)query);
        keyGm.SetGlobalBuffer((__gm__ T *)gatheredKvPages);
        indicesGm.SetGlobalBuffer((__gm__ int32_t *)logicalSparseRows);
        validChunksGm.SetGlobalBuffer((__gm__ int32_t *)scanCount);
        blocksGm.SetGlobalBuffer((__gm__ int32_t *)pageTable);
        querySeqGm.SetGlobalBuffer((__gm__ int32_t *)querySeq);
        outputGm.SetGlobalBuffer((__gm__ T *)output);
        totalQuery = totalQ;
        sparseCount = sparseK;
        this->pageSize = pageSize;
        this->maxBlocks = maxBlocks;
        usedCores = cores;
        scale = scaleValue;
        returnLse = false;
        queryRankSize = 1;
        queryLocalHeads = HEADS;
        uint32_t core = 0;
        if ASCEND_IS_AIC core = GetBlockIdx();
        else core = GetBlockIdx() / 2;
        __gm__ uint8_t *userBase = GetUserWorkspace(workspace);
        __gm__ uint8_t *base = userBase + core * CORE_WORKSPACE;
        mm1Gm.SetGlobalBuffer((__gm__ float *)base);
        probGm.SetGlobalBuffer((__gm__ T *)(base + GROUP_SLOTS * MM1_BYTES));
        mm2Gm.SetGlobalBuffer((__gm__ float *)(base +
            GROUP_SLOTS * (MM1_BYTES + PROB_BYTES)));
        rowsGm.SetGlobalBuffer((__gm__ int32_t *)(base +
            GROUP_SLOTS * (MM1_BYTES + PROB_BYTES + MM2_BYTES)));
        taskGm.SetGlobalBuffer((__gm__ int32_t *)(userBase +
            static_cast<uint64_t>(usedCores) * CORE_WORKSPACE));
        scheduleGm.SetGlobalBuffer((__gm__ int32_t *)(userBase +
            static_cast<uint64_t>(usedCores) * CORE_WORKSPACE +
            static_cast<uint64_t>(totalQuery) * TASK_BYTES));
        packedKvGm.SetGlobalBuffer((__gm__ T *)(base +
            GROUP_SLOTS * (MM1_BYTES + PROB_BYTES + MM2_BYTES) + ROWS_BYTES));
        if ASCEND_IS_AIC {
            InitCube(pipe);
            StartCubeWithPublishedRange(core);
            ProcessCube();
            FreeCubeEvents(pipe);
        } else {
            InitVector(pipe);
            StartVectorWithPublishedRange(core, GetBlockIdx() & 1);
            ProcessVector();
            FreeVectorEvents(pipe);
        }
    }
};

template<int PREFILL_IMPL>
__global__ __aicore__ void sparse_mla_attention_prefill(
    GM_ADDR query, GM_ADDR gatheredKvPages, GM_ADDR logicalSparseRows,
    GM_ADDR scanCount, GM_ADDR pageTable, GM_ADDR querySeq,
    GM_ADDR attentionOut,
    GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    GET_TILING_DATA(data, tiling);
    TPipe pipe;
    static_assert(PREFILL_IMPL == 1 || PREFILL_IMPL == 2,
                  "unsupported Prefill implementation");
    SparseMlaAttentionPrefillKernel<
        bfloat16_t, false, true, PREFILL_IMPL == 2, true> op;
    op.Init(query, gatheredKvPages, logicalSparseRows, scanCount,
            pageTable, querySeq,
            attentionOut, workspace, data.totalQuery, data.sparseCount,
            data.pageSize, data.usedCores, data.maxBlocks, data.scale, &pipe);
}
