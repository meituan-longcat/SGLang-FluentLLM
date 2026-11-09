#include "kernel_operator.h"

using namespace AscendC;

namespace {
constexpr uint32_t LATENT = 512;
constexpr uint32_t ROPE = 64;
constexpr uint32_t TILE_COUNT = 4;
constexpr uint32_t TILE = 128;
constexpr uint32_t PACKED_PAYLOAD = 656;
constexpr uint32_t PACKED_ROW_ALIGN = 672;
// Keep the first production gate deliberately conservative: one row per
// per-core pipeline iteration removes cross-row Vector repeat/UB reuse
// hazards while preserving the main win (one host launch for the whole TND
// tensor).  Forty-eight AIV cores still process independent rows in parallel.
constexpr uint32_t BATCH_ROWS = 1;
constexpr uint32_t FP32_ROWS = 1;
}

class PackedKvDequantKernel {
public:
    __aicore__ inline void Init(
        GM_ADDR packedKv, GM_ADDR key, GM_ADDR keyRope,
        uint32_t rows, uint32_t rowsPerCore, uint32_t packedRowBytes,
        TPipe *pipe)
    {
        packedGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ int8_t *>(packedKv));
        keyGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(key));
        ropeGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ bfloat16_t *>(keyRope));
        rows_ = rows;
        rowsPerCore_ = rowsPerCore;
        packedRowBytes_ = packedRowBytes;

        pipe->InitBuffer(packedBuf_, BATCH_ROWS * PACKED_ROW_ALIGN);
        pipe->InitBuffer(keyBuf_, BATCH_ROWS * LATENT * sizeof(bfloat16_t));
        pipe->InitBuffer(ropeBuf_, BATCH_ROWS * ROPE * sizeof(bfloat16_t));
        pipe->InitBuffer(fp32Buf_, FP32_ROWS * LATENT * sizeof(float));
        pipe->InitBuffer(scaleBuf_, BATCH_ROWS * 32 * sizeof(float));
        pipe->InitBuffer(compactScaleBuf_, BATCH_ROWS * TILE_COUNT * sizeof(float));

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

    __aicore__ inline void Process()
    {
        const uint32_t first = GetBlockIdx() * rowsPerCore_;
        if (first >= rows_) {
            return;
        }
        uint32_t localRows = rows_ - first;
        if (localRows > rowsPerCore_) {
            localRows = rowsPerCore_;
        }
        uint32_t done = 0;
        while (done < localRows) {
            uint32_t batch = localRows - done;
            if (batch > BATCH_ROWS) {
                batch = BATCH_ROWS;
            }
            Load(first + done, batch);
            Dequant(batch);
            Store(first + done, batch);
            done += batch;
        }
    }

private:
    __aicore__ inline void Load(uint32_t firstRow, uint32_t batch)
    {
        // Both ABIs carry the same 656-byte payload.  The optional 16-byte
        // tail is alignment padding and must not enter the dequant pipeline.
        const DataCopyExtParams copy{1, PACKED_PAYLOAD, 0, 0, 0};
        const DataCopyPadExtParams<int8_t> noPad{false, 0, 0, 0};
        for (uint32_t row = 0; row < batch; ++row) {
            DataCopyPad(
                packedUb_[row * PACKED_ROW_ALIGN],
                packedGm_[static_cast<uint64_t>(firstRow + row) *
                    packedRowBytes_],
                copy, noPad);
        }
        SetFlag<HardEvent::MTE2_V>(mte2ToV_);
        WaitFlag<HardEvent::MTE2_V>(mte2ToV_);
    }

    __aicore__ inline void Dequant(uint32_t batch)
    {
        LocalTensor<half> fp16 = keyUb_.template ReinterpretCast<half>();
        for (uint32_t tile = 0; tile < TILE_COUNT; ++tile) {
            Cast(fp16[tile * TILE], packedUb_[tile * TILE],
                 RoundMode::CAST_NONE, TILE, static_cast<uint8_t>(batch),
                 {1, 1, 32, 21});
        }

        LocalTensor<bfloat16_t> packedRope =
            packedUb_[LATENT].template ReinterpretCast<bfloat16_t>();
        Copy(ropeUb_, packedRope, ROPE, static_cast<uint8_t>(batch),
             {1, 1, 4, 21});

        LocalTensor<float> packedScale =
            packedUb_[640].template ReinterpretCast<float>();
        DataCopyParams scaleCopy;
        scaleCopy.blockCount = static_cast<uint16_t>(batch);
        scaleCopy.blockLen = 1;
        scaleCopy.srcStride = 20;
        scaleCopy.dstStride = 0;
        DataCopy(compactScaleUb_, packedScale, scaleCopy);
        PipeBarrier<PIPE_V>();
        Brcb(scaleUb_, compactScaleUb_, static_cast<uint8_t>(batch),
             {1, 4});
        PipeBarrier<PIPE_V>();

        for (uint32_t rowBase = 0; rowBase < batch;
             rowBase += FP32_ROWS) {
            uint32_t dealRows = batch - rowBase;
            if (dealRows > FP32_ROWS) {
                dealRows = FP32_ROWS;
            }
            Cast(fp32Ub_, fp16[rowBase * LATENT], RoundMode::CAST_NONE,
                 dealRows * LATENT);
            PipeBarrier<PIPE_V>();
            for (uint32_t halfTile = 0; halfTile < 2; ++halfTile) {
                Mul(fp32Ub_[halfTile * 64], fp32Ub_[halfTile * 64],
                    scaleUb_[rowBase * 32], 64,
                    static_cast<uint8_t>(TILE_COUNT * dealRows),
                    {1, 1, 0, 16, 16, 1});
            }
            PipeBarrier<PIPE_V>();
            Cast(keyUb_[rowBase * LATENT], fp32Ub_,
                 RoundMode::CAST_RINT, dealRows * LATENT);
            PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void Store(uint32_t firstRow, uint32_t batch)
    {
        SetFlag<HardEvent::V_MTE3>(vToMte3_);
        WaitFlag<HardEvent::V_MTE3>(vToMte3_);
        DataCopy(
            keyGm_[static_cast<uint64_t>(firstRow) * LATENT], keyUb_,
            batch * LATENT);
        DataCopy(
            ropeGm_[static_cast<uint64_t>(firstRow) * ROPE], ropeUb_,
            batch * ROPE);
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

extern "C" __global__ __aicore__ void packed_kv_dequant(
    GM_ADDR packedKv, GM_ADDR key, GM_ADDR keyRope,
    GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(data, tiling);
    TPipe pipe;
    if (TILING_KEY_IS(1)) {
        PackedKvDequantKernel op;
        op.Init(packedKv, key, keyRope, data.rows, data.rowsPerCore,
            data.packedRowBytes, &pipe);
        op.Process();
    }
}
