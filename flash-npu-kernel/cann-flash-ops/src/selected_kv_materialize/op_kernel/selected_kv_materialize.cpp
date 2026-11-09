#include "kernel_operator.h"

using namespace AscendC;

namespace {
constexpr uint32_t LATENT = 512;
constexpr uint32_t ROPE = 64;
constexpr uint32_t TILE_COUNT = 4;
constexpr uint32_t TILE = 128;
constexpr uint32_t PACKED_PAYLOAD = 656;
constexpr uint32_t PACKED_ROW_ALIGN = 672;
// Batch rows to amortize DMA and pipeline synchronization. 16 rows use
// bounded per-core UB; packed, latent and RoPE strides remain distinct.
constexpr uint32_t BATCH_ROWS = 16;
constexpr uint32_t FP32_ROWS = BATCH_ROWS;
} // namespace

class SelectedKvMaterializeKernel {
  public:
    __aicore__ inline void Init(GM_ADDR packedKv, GM_ADDR extend, GM_ADDR output, uint32_t rows, uint32_t rowsPerCore,
                                uint32_t packedRowBytes, uint32_t extendRows, TPipe *pipe) {
        packedGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t *>(packedKv));
        keyGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(output));
        ropeGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(output));
        rows_ = rows;
        extendRows_ = extendRows;
        extendGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(extend));
        rowsPerCore_ = rowsPerCore;
        packedRowBytes_ = packedRowBytes;

        pipe->InitBuffer(packedBuf_, BATCH_ROWS * PACKED_ROW_ALIGN);
        pipe->InitBuffer(keyBuf_, BATCH_ROWS * LATENT * sizeof(bfloat16_t));
        pipe->InitBuffer(ropeBuf_, BATCH_ROWS * ROPE * sizeof(bfloat16_t));
        pipe->InitBuffer(fp32Buf_, FP32_ROWS * LATENT * sizeof(float));
        pipe->InitBuffer(scaleBuf_, BATCH_ROWS * 64 * sizeof(float));
        pipe->InitBuffer(compactScaleBuf_, BATCH_ROWS * 8 * sizeof(float));

        packedUb_ = packedBuf_.Get<int8_t>();
        keyUb_ = keyBuf_.Get<bfloat16_t>();
        ropeUb_ = ropeBuf_.Get<bfloat16_t>();
        fp32Ub_ = fp32Buf_.Get<float>();
        scaleUb_ = scaleBuf_.Get<float>();
        compactScaleUb_ = compactScaleBuf_.Get<float>();
        mte2ToV_ = pipe->AllocEventID<HardEvent::MTE2_V>();
        vToMte3_ = pipe->AllocEventID<HardEvent::V_MTE3>();
        mte3ToV_ = pipe->AllocEventID<HardEvent::MTE3_V>();
        mte3ToMte2_ = pipe->AllocEventID<HardEvent::MTE3_MTE2>();
    }

    __aicore__ inline void Process() {
        const uint32_t first = GetBlockIdx() * rowsPerCore_;
        const uint32_t total = ((rows_ + extendRows_ + 127) / 128) * 128;
        uint32_t end = first + rowsPerCore_;
        if (end > total)
            end = total;
        for (uint32_t row = first; row < end;) {
            uint32_t batch = end - row;
            if (batch > BATCH_ROWS)
                batch = BATCH_ROWS;
            if (row < rows_) {
                if (batch > rows_ - row)
                    batch = rows_ - row;
                Load(row, batch);
                Dequant(batch);
                Store(row, batch);
            } else {
                auto buf = fp32Buf_.Get<bfloat16_t>();
                if (row < rows_ + extendRows_) {
                    if (batch > rows_ + extendRows_ - row)
                        batch = rows_ + extendRows_ - row;
                    DataCopy(buf, extendGm_[static_cast<uint64_t>(row - rows_) * 576], batch * 576);
                } else {
                    Duplicate(buf, static_cast<bfloat16_t>(0), batch * 576);
                }
                PipeBarrier<PIPE_ALL>();
                DataCopy(keyGm_[static_cast<uint64_t>(row) * 576], buf, batch * 576);
                PipeBarrier<PIPE_ALL>();
            }
            row += batch;
        }
    }

  private:
    __aicore__ inline void Load(uint32_t firstRow, uint32_t batch) {
        // Both ABIs carry the same 656-byte payload.  The optional 16-byte
        // tail is alignment padding and must not enter the dequant pipeline.
        const DataCopyExtParams copy{static_cast<uint16_t>(batch), PACKED_PAYLOAD, packedRowBytes_ - PACKED_PAYLOAD, 0,
                                     0};
        const DataCopyPadExtParams<int8_t> noPad{false, 0, 0, 0};
        DataCopyPad(packedUb_, packedGm_[static_cast<uint64_t>(firstRow) * packedRowBytes_], copy, noPad);
        SetFlag<HardEvent::MTE2_V>(mte2ToV_);
        WaitFlag<HardEvent::MTE2_V>(mte2ToV_);
    }

    __aicore__ inline void Dequant(uint32_t batch) {
        LocalTensor<half> fp16 = keyUb_.template ReinterpretCast<half>();
        for (uint32_t tile = 0; tile < TILE_COUNT; ++tile) {
            Cast(fp16[tile * TILE], packedUb_[tile * TILE], RoundMode::CAST_NONE, TILE, static_cast<uint8_t>(batch),
                 {1, 1, 32, 21});
        }

        LocalTensor<bfloat16_t> packedRope = packedUb_[LATENT].template ReinterpretCast<bfloat16_t>();
        Copy(ropeUb_, packedRope, ROPE, static_cast<uint8_t>(batch), {1, 1, 4, 21});

        LocalTensor<float> packedScale = packedUb_[640].template ReinterpretCast<float>();
        DataCopyParams scaleCopy;
        scaleCopy.blockCount = static_cast<uint16_t>(batch);
        scaleCopy.blockLen = 1;
        scaleCopy.srcStride = 20;
        scaleCopy.dstStride = 0;
        DataCopy(compactScaleUb_, packedScale, scaleCopy);
        PipeBarrier<PIPE_V>();
        Brcb(scaleUb_, compactScaleUb_, static_cast<uint8_t>(batch), {1, 8});
        PipeBarrier<PIPE_V>();

        Cast(fp32Ub_, fp16, RoundMode::CAST_NONE, batch * LATENT);
        PipeBarrier<PIPE_V>();
        // Each repeat advances one row: 512 floats (64 blocks), 64 expanded
        // scales (8 blocks). Each tile shares exactly one FP32 scale.
        for (uint32_t tile = 0; tile < TILE_COUNT; ++tile) {
            for (uint32_t halfTile = 0; halfTile < 2; ++halfTile) {
                const uint32_t offset = tile * TILE + halfTile * 64;
                Mul(fp32Ub_[offset], fp32Ub_[offset], scaleUb_[tile * 8], 64, static_cast<uint8_t>(batch),
                    {1, 1, 0, 64, 64, 8});
            }
        }
        PipeBarrier<PIPE_V>();
        Cast(keyUb_, fp32Ub_, RoundMode::CAST_RINT, batch * LATENT);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void Store(uint32_t firstRow, uint32_t batch) {
        SetFlag<HardEvent::V_MTE3>(vToMte3_);
        WaitFlag<HardEvent::V_MTE3>(vToMte3_);
        // Strided stores interleave latent/rope directly into [T,576].
        DataCopyParams keyCopy{static_cast<uint16_t>(batch), 32, 0, 4};
        DataCopyParams ropeCopy{static_cast<uint16_t>(batch), 4, 0, 32};
        DataCopy(keyGm_[static_cast<uint64_t>(firstRow) * 576], keyUb_, keyCopy);
        DataCopy(ropeGm_[static_cast<uint64_t>(firstRow) * 576 + LATENT], ropeUb_, ropeCopy);
        // The next batch reuses keyUb_ and ropeUb_.  Wait for both stores to
        // finish before allowing Vector to overwrite either source buffer.
        SetFlag<HardEvent::MTE3_V>(mte3ToV_);
        WaitFlag<HardEvent::MTE3_V>(mte3ToV_);
        // Also serialize the next GM->UB load behind the current GM stores.
        // This makes the per-core loop a closed MTE2->V->MTE3 transaction.
        SetFlag<HardEvent::MTE3_MTE2>(mte3ToMte2_);
        WaitFlag<HardEvent::MTE3_MTE2>(mte3ToMte2_);
    }

    GlobalTensor<int8_t> packedGm_;
    GlobalTensor<bfloat16_t> extendGm_;
    uint32_t extendRows_;
    GlobalTensor<bfloat16_t> keyGm_;
    GlobalTensor<bfloat16_t> ropeGm_;
    TBuf<> packedBuf_;
    TBuf<> keyBuf_;
    TBuf<> ropeBuf_;
    TBuf<> fp32Buf_;
    TBuf<> scaleBuf_;
    TBuf<> compactScaleBuf_;
    LocalTensor<int8_t> packedUb_;
    LocalTensor<bfloat16_t> keyUb_;
    LocalTensor<bfloat16_t> ropeUb_;
    LocalTensor<float> fp32Ub_;
    LocalTensor<float> scaleUb_;
    LocalTensor<float> compactScaleUb_;
    TEventID mte2ToV_;
    TEventID vToMte3_;
    TEventID mte3ToV_;
    TEventID mte3ToMte2_;
    uint32_t rows_ = 0;
    uint32_t rowsPerCore_ = 0;
    uint32_t packedRowBytes_ = PACKED_PAYLOAD;
};

extern "C" __global__ __aicore__ void selected_kv_materialize(GM_ADDR packedKv, GM_ADDR extend, GM_ADDR output,
                                                              GM_ADDR workspace, GM_ADDR tiling) {
    (void)workspace;
    GET_TILING_DATA(data, tiling);
    TPipe pipe;
    if (TILING_KEY_IS(1)) {
        SelectedKvMaterializeKernel op;
        op.Init(packedKv, extend, output, data.rows, data.rowsPerCore, data.packedRowBytes, data.extendRows, &pipe);
        op.Process();
    }
}
