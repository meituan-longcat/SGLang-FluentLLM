#include "kernel_operator.h"

using namespace AscendC;

template <typename CacheT, typename IndexT>
class KvCacheCopyRowsKernel {
public:
    __aicore__ inline void Init(GM_ADDR cache, GM_ADDR srcIndices,
                                GM_ADDR dstIndices, uint32_t rowCount,
                                uint32_t rowWidth, uint32_t totalRows,
                                uint32_t snapshotBytes, TPipe *pipe)
    {
        cacheGm_.SetGlobalBuffer(reinterpret_cast<__gm__ CacheT *>(cache));
        srcIndicesGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ IndexT *>(srcIndices));
        dstIndicesGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ IndexT *>(dstIndices));
        rowCount_ = rowCount;
        rowWidth_ = rowWidth;
        totalRows_ = totalRows;
        rowBytes_ = rowWidth * sizeof(CacheT);
        snapshotStride_ = snapshotBytes / rowCount / sizeof(CacheT);

        pipe->InitBuffer(snapshotBuf_, snapshotBytes);
        snapshot_ = snapshotBuf_.Get<CacheT>();
        mte2ToMte3_ = pipe->AllocEventID<HardEvent::MTE2_MTE3>();
    }

    __aicore__ inline void ProcessByteExact()
    {
        const DataCopyExtParams copyParams{1, rowBytes_, 0, 0, 0};
        const DataCopyPadExtParams<CacheT> padParams{false, 0, 0, 0};
        for (uint32_t i = 0; i < rowCount_; ++i) {
            const int64_t src = static_cast<int64_t>(srcIndicesGm_.GetValue(i));
            if (src < 0 || src >= static_cast<int64_t>(totalRows_)) {
                continue;
            }
            DataCopyPad(snapshot_[i * snapshotStride_],
                        cacheGm_[static_cast<uint64_t>(src) * rowWidth_],
                        copyParams, padParams);
        }

        SetFlag<HardEvent::MTE2_MTE3>(mte2ToMte3_);
        WaitFlag<HardEvent::MTE2_MTE3>(mte2ToMte3_);

        for (uint32_t i = 0; i < rowCount_; ++i) {
            const int64_t src = static_cast<int64_t>(srcIndicesGm_.GetValue(i));
            const int64_t dst = static_cast<int64_t>(dstIndicesGm_.GetValue(i));
            if (src < 0 || src >= static_cast<int64_t>(totalRows_) ||
                dst < 0 || dst >= static_cast<int64_t>(totalRows_)) {
                continue;
            }
            DataCopyPad(cacheGm_[static_cast<uint64_t>(dst) * rowWidth_],
                        snapshot_[i * snapshotStride_], copyParams);
        }
    }

    __aicore__ inline void Process()
    {
        for (uint32_t i = 0; i < rowCount_; ++i) {
            const int64_t src = static_cast<int64_t>(srcIndicesGm_.GetValue(i));
            if (src < 0 || src >= static_cast<int64_t>(totalRows_)) {
                continue;
            }
            DataCopy(snapshot_[i * rowWidth_],
                     cacheGm_[static_cast<uint64_t>(src) * rowWidth_],
                     rowWidth_);
        }

        SetFlag<HardEvent::MTE2_MTE3>(mte2ToMte3_);
        WaitFlag<HardEvent::MTE2_MTE3>(mte2ToMte3_);

        for (uint32_t i = 0; i < rowCount_; ++i) {
            const int64_t src = static_cast<int64_t>(srcIndicesGm_.GetValue(i));
            const int64_t dst = static_cast<int64_t>(dstIndicesGm_.GetValue(i));
            if (src < 0 || src >= static_cast<int64_t>(totalRows_) ||
                dst < 0 || dst >= static_cast<int64_t>(totalRows_)) {
                continue;
            }
            DataCopy(cacheGm_[static_cast<uint64_t>(dst) * rowWidth_],
                     snapshot_[i * rowWidth_], rowWidth_);
        }
    }

private:
    GlobalTensor<CacheT> cacheGm_;
    GlobalTensor<IndexT> srcIndicesGm_;
    GlobalTensor<IndexT> dstIndicesGm_;
    TBuf<> snapshotBuf_;
    LocalTensor<CacheT> snapshot_;
    TEventID mte2ToMte3_;
    uint32_t rowCount_ = 0;
    uint32_t rowWidth_ = 0;
    uint32_t totalRows_ = 0;
    uint32_t rowBytes_ = 0;
    uint32_t snapshotStride_ = 0;
};

template <typename CacheT, typename IndexT>
__aicore__ inline void RunKvCacheCopyRows(
    GM_ADDR cache, GM_ADDR srcIndices, GM_ADDR dstIndices,
    uint32_t rowCount, uint32_t rowWidth, uint32_t totalRows,
    uint32_t snapshotBytes, TPipe *pipe)
{
    KvCacheCopyRowsKernel<CacheT, IndexT> op;
    op.Init(cache, srcIndices, dstIndices, rowCount, rowWidth, totalRows,
            snapshotBytes, pipe);
    op.Process();
}

template <typename CacheT, typename IndexT>
__aicore__ inline void RunKvCacheCopyRowsByteExact(
    GM_ADDR cache, GM_ADDR srcIndices, GM_ADDR dstIndices,
    uint32_t rowCount, uint32_t rowWidth, uint32_t totalRows,
    uint32_t snapshotBytes, TPipe *pipe)
{
    KvCacheCopyRowsKernel<CacheT, IndexT> op;
    op.Init(cache, srcIndices, dstIndices, rowCount, rowWidth, totalRows,
            snapshotBytes, pipe);
    op.ProcessByteExact();
}

extern "C" __global__ __aicore__ void kv_cache_copy_rows(
    GM_ADDR cache, GM_ADDR srcIndices, GM_ADDR dstIndices, GM_ADDR cacheOut,
    GM_ADDR workspace, GM_ADDR tiling)
{
    (void)cacheOut;
    (void)workspace;
    GET_TILING_DATA(data, tiling);
    TPipe pipe;
    if (TILING_KEY_IS(2)) {
        if constexpr (ORIG_DTYPE_SRC_INDICES == DT_INT64) {
            RunKvCacheCopyRowsByteExact<int8_t, int64_t>(
                cache, srcIndices, dstIndices, data.rowCount,
                data.rowWidth, data.totalRows, data.snapshotBytes, &pipe);
        } else {
            RunKvCacheCopyRowsByteExact<int8_t, int32_t>(
                cache, srcIndices, dstIndices, data.rowCount,
                data.rowWidth, data.totalRows, data.snapshotBytes, &pipe);
        }
    } else if (TILING_KEY_IS(1)) {
        if constexpr (ORIG_DTYPE_CACHE == DT_FLOAT16) {
            if constexpr (ORIG_DTYPE_SRC_INDICES == DT_INT64) {
                RunKvCacheCopyRows<half, int64_t>(
                    cache, srcIndices, dstIndices, data.rowCount,
                    data.rowWidth, data.totalRows, data.snapshotBytes, &pipe);
            } else {
                RunKvCacheCopyRows<half, int32_t>(
                    cache, srcIndices, dstIndices, data.rowCount,
                    data.rowWidth, data.totalRows, data.snapshotBytes, &pipe);
            }
        } else if constexpr (ORIG_DTYPE_CACHE == DT_BF16) {
            if constexpr (ORIG_DTYPE_SRC_INDICES == DT_INT64) {
                RunKvCacheCopyRows<bfloat16_t, int64_t>(
                    cache, srcIndices, dstIndices, data.rowCount,
                    data.rowWidth, data.totalRows, data.snapshotBytes, &pipe);
            } else {
                RunKvCacheCopyRows<bfloat16_t, int32_t>(
                    cache, srcIndices, dstIndices, data.rowCount,
                    data.rowWidth, data.totalRows, data.snapshotBytes, &pipe);
            }
        } else {
            if constexpr (ORIG_DTYPE_SRC_INDICES == DT_INT64) {
                RunKvCacheCopyRows<int8_t, int64_t>(
                    cache, srcIndices, dstIndices, data.rowCount,
                    data.rowWidth, data.totalRows, data.snapshotBytes, &pipe);
            } else {
                RunKvCacheCopyRows<int8_t, int32_t>(
                    cache, srcIndices, dstIndices, data.rowCount,
                    data.rowWidth, data.totalRows, data.snapshotBytes, &pipe);
            }
        }
    }
}
