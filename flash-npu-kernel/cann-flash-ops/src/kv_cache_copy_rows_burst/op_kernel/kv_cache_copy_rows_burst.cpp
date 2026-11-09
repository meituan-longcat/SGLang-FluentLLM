#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
using namespace AscendC;

template <typename IndexT>
class KvCacheCopyRowsBurstKernel {
public:
    __aicore__ inline void Init(GM_ADDR caches, GM_ADDR src, GM_ADDR dst,
        uint32_t layers, uint32_t rows, uint32_t count, uint32_t rowBytes, uint32_t indexBytes, TPipe* pipe)
    {
        caches_ = caches;layers_ = layers;rows_ = rows;count_ = count;rowBytes_ = rowBytes;
        pipe->InitBuffer(snapshotBuffer_, count * rowBytes_);
        pipe->InitBuffer(sourceBuffer_, indexBytes);pipe->InitBuffer(destinationBuffer_, indexBytes);
        snapshot_ = snapshotBuffer_.Get<int8_t>();
        source_ = sourceBuffer_.Get<IndexT>();destination_ = destinationBuffer_.Get<IndexT>();
        GlobalTensor<IndexT> srcGm, dstGm;
        srcGm.SetGlobalBuffer(reinterpret_cast<__gm__ IndexT*>(src));
        dstGm.SetGlobalBuffer(reinterpret_cast<__gm__ IndexT*>(dst));
        mte2S_ = pipe->AllocEventID<HardEvent::MTE2_S>();
        mte2Mte3_ = pipe->AllocEventID<HardEvent::MTE2_MTE3>();
        mte3S_ = pipe->AllocEventID<HardEvent::MTE3_S>();
        mte3Mte2_ = pipe->AllocEventID<HardEvent::MTE3_MTE2>();
        const DataCopyExtParams bytes{1, count * static_cast<uint32_t>(sizeof(IndexT)), 0, 0, 0};
        const DataCopyPadExtParams<IndexT> pad{false, 0, 0, 0};
        DataCopyPad(source_, srcGm, bytes, pad);DataCopyPad(destination_, dstGm, bytes, pad);
        Sync<HardEvent::MTE2_S>(mte2S_);
    }
    __aicore__ inline void Process()
    {
        ListTensorDesc list(reinterpret_cast<__gm__ void*>(caches_));
        bool previousLayer = false;
        for (uint32_t layer = GetBlockIdx(); layer < layers_; layer += GetBlockNum()) {
            if (previousLayer) {
                // MTE3 still reads this exact snapshot UB. Complete those
                // reads before any following layer's MTE2 overwrites it.
                Sync<HardEvent::MTE3_S>(mte3S_);
                Sync<HardEvent::MTE3_MTE2>(mte3Mte2_);
            }
            GlobalTensor<int8_t> cache;
            cache.SetGlobalBuffer(list.GetDataPtr<int8_t>(layer));
            for (uint32_t i = 0; i < count_; ++i) {
                const int64_t src = static_cast<int64_t>(source_.GetValue(i));
                if (Valid(src)) { DataCopy(snapshot_[i * rowBytes_], cache[uint64_t(src) * rowBytes_], rowBytes_); }
            }
            // All valid sources, including rows whose destination is invalid,
            // are snapshotted before the first inplace write in this layer.
            Sync<HardEvent::MTE2_S>(mte2S_);
            Sync<HardEvent::MTE2_MTE3>(mte2Mte3_);
            int64_t previous = -1;
            for (uint32_t i = 0; i < count_; ++i) {
                const int64_t src = static_cast<int64_t>(source_.GetValue(i));
                const int64_t dst = static_cast<int64_t>(destination_.GetValue(i));
                if (!Valid(src) || !Valid(dst)) { continue; }
                // Increasing runs have disjoint destinations. Drain before
                // any repeated/backward address so input-order last wins.
                if (dst <= previous) { Sync<HardEvent::MTE3_S>(mte3S_); }
                DataCopy(cache[uint64_t(dst) * rowBytes_], snapshot_[i * rowBytes_], rowBytes_);
                previous = dst;
            }
            previousLayer = true;
        }
        Sync<HardEvent::MTE3_S>(mte3S_);
    }
private:
    template <HardEvent Event>
    __aicore__ inline void Sync(TEventID id) { SetFlag<Event>(id);WaitFlag<Event>(id); }
    __aicore__ inline bool Valid(int64_t index) { return index >= 0 && index < static_cast<int64_t>(rows_); }
    GM_ADDR caches_;
    uint32_t layers_ = 0, rows_ = 0, count_ = 0, rowBytes_ = 0;
    TBuf<TPosition::VECCALC> snapshotBuffer_, sourceBuffer_, destinationBuffer_;
    LocalTensor<int8_t> snapshot_;
    LocalTensor<IndexT> source_, destination_;
    TEventID mte2S_, mte2Mte3_, mte3S_, mte3Mte2_;
};

extern "C" __global__ __aicore__ void kv_cache_copy_rows_burst(
    GM_ADDR caches, GM_ADDR srcIndices, GM_ADDR dstIndices, GM_ADDR cachesOut,
    GM_ADDR workspace, GM_ADDR tiling)
{
    (void)cachesOut;(void)workspace;
    GET_TILING_DATA(data, tiling);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (data.rowCount == 0) { return; }
    TPipe pipe;
    if (TILING_KEY_IS(2)) {
        KvCacheCopyRowsBurstKernel<int64_t> op;
        op.Init(caches, srcIndices, dstIndices, data.layerCount, data.totalRows, data.rowCount, data.rowBytes, data.indexBufferBytes, &pipe);op.Process();
    } else if (TILING_KEY_IS(1)) {
        KvCacheCopyRowsBurstKernel<int32_t> op;
        op.Init(caches, srcIndices, dstIndices, data.layerCount, data.totalRows, data.rowCount, data.rowBytes, data.indexBufferBytes, &pipe);op.Process();
    }
}
