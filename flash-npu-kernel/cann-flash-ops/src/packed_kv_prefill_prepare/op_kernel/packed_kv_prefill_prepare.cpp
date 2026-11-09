#include "kernel_operator.h"
#include "mla_prolog_v3_gather_sin_cos.h"
#include "mla_prolog_v3_rms_norm.h"
#include "mla_prolog_v3_rotary_position_embedding.h"

using namespace AscendC;

namespace {
constexpr uint32_t LATENT = 512;
constexpr uint32_t ROPE = 64;
constexpr uint32_t INPUT_ROW = 576;
constexpr uint32_t TILE = 128;
constexpr uint32_t TILE_COUNT = 4;
constexpr uint32_t PACKED_PAYLOAD = 656;
constexpr uint32_t PACKED_ALIGN = 672;
}

class PackedKvPrefillPrepareKernel {
public:
    __aicore__ inline void Init(
        GM_ADDR latentCache, GM_ADDR gamma, GM_ADDR cos, GM_ADDR sin,
        GM_ADDR ownerRows, GM_ADDR key, GM_ADDR keyRope,
        GM_ADDR ownerPacked, const PackedKvPrefillPrepareTilingData &tiling,
        TPipe *pipe)
    {
        latentGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(latentCache));
        gammaGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(gamma));
        cosGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(cos));
        sinGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(sin));
        ownerRowsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(ownerRows));
        keyGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(key));
        ropeGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(keyRope));
        packedGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t *>(ownerPacked));

        rows_ = tiling.rows;
        ownerRows_ = tiling.ownerRows;
        rowsPerCore_ = tiling.rowsPerCore;
        ownersPerCore_ = tiling.ownersPerCore;
        packedRowBytes_ = tiling.packedRowBytes;
        epsilon_ = tiling.epsilon;

        pipe->InitBuffer(gammaBuf_, LATENT * sizeof(bfloat16_t));
        pipe->InitBuffer(normBuf_, LATENT * sizeof(bfloat16_t));
        pipe->InitBuffer(ropeBuf_, ROPE * sizeof(bfloat16_t));
        pipe->InitBuffer(rmsWorkBuf_, 5 * LATENT * sizeof(float));
        pipe->InitBuffer(sinCosBuf_, 2 * ROPE * sizeof(float));
        pipe->InitBuffer(sinCosWorkBuf_, 2 * ROPE * sizeof(bfloat16_t));
        pipe->InitBuffer(ropeWorkBuf_, 6 * ROPE * sizeof(float));
        pipe->InitBuffer(quantFp32Buf_, LATENT * sizeof(float));
        pipe->InitBuffer(quantWorkBuf_, 6 * LATENT * sizeof(float));
        pipe->InitBuffer(packedBuf_, PACKED_ALIGN);

        gammaUb_ = gammaBuf_.Get<bfloat16_t>();
        normUb_ = normBuf_.Get<bfloat16_t>();
        ropeUb_ = ropeBuf_.Get<bfloat16_t>();
        rmsWorkUb_ = rmsWorkBuf_.Get<uint8_t>();
        cosUb_ = sinCosBuf_.Get<float>();
        sinUb_ = cosUb_[ROPE];
        sinCosWorkUb_ = sinCosWorkBuf_.Get<uint8_t>();
        ropeWorkUb_ = ropeWorkBuf_.Get<uint8_t>();
        quantFp32Ub_ = quantFp32Buf_.Get<float>();
        quantWorkUb_ = quantWorkBuf_.Get<uint8_t>();
        packedUb_ = packedBuf_.Get<int8_t>();
    }

    __aicore__ inline void Process()
    {
        LoadGamma();
        ProcessTransientRows();
        ProcessOwnerRows();
    }

private:
    __aicore__ inline void LoadGamma()
    {
        DataCopy(gammaUb_, gammaGm_, LATENT);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID2);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID2);
    }

    __aicore__ inline void PrepareRow(uint32_t row)
    {
        MlaPrologV3::RmsNormParam rmsParams{
            0.001953125f, epsilon_, 1, LATENT, 1.0f, false};
        LocalTensor<float> unused;
        MlaPrologV3::RmsNormNormal<
            bfloat16_t, bfloat16_t, float, bfloat16_t>(
                normUb_, latentGm_[static_cast<uint64_t>(row) * INPUT_ROW],
                gammaUb_, unused, unused, rmsWorkUb_, rmsParams);

        MlaPrologV3::GatherSinCos<bfloat16_t, float>(
            cosUb_, sinUb_, cosGm_, sinGm_, row, 1,
            sinCosWorkUb_, 1, ROPE);
        MlaPrologV3::Rectangle ropeParams{1, ROPE, INPUT_ROW};
        MlaPrologV3::RotaryPosEmbPerTensor<
            bfloat16_t, float, bfloat16_t>(
                ropeUb_, latentGm_[static_cast<uint64_t>(row) * INPUT_ROW + LATENT],
                cosUb_, sinUb_, ropeWorkUb_, ropeParams);
    }

    __aicore__ inline void StoreTransient(uint32_t row)
    {
        SetFlag<HardEvent::V_MTE3>(EVENT_ID2);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID2);
        DataCopy(keyGm_[static_cast<uint64_t>(row) * LATENT], normUb_, LATENT);
        DataCopy(ropeGm_[static_cast<uint64_t>(row) * ROPE], ropeUb_, ROPE);
        SetFlag<HardEvent::MTE3_V>(EVENT_ID2);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID2);
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID2);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID2);
    }

    __aicore__ inline void PackOwner(uint32_t owner)
    {
        // The aligned ABI exposes the full 672-byte row. Clear the UB slot so
        // its 16-byte padding tail is deterministic and never leaks stale UB
        // contents. Quantization and packing overwrite the 656-byte payload.
        if (packedRowBytes_ == PACKED_ALIGN) {
            Duplicate(
                packedUb_.ReinterpretCast<uint32_t>(),
                static_cast<uint32_t>(0), PACKED_ALIGN / sizeof(uint32_t));
        }
        Cast(quantFp32Ub_, normUb_, RoundMode::CAST_NONE, LATENT);
        PipeBarrier<PIPE_V>();
        MlaPrologV3::PerTileQuantParams params{
            static_cast<int32_t>(TILE), static_cast<int32_t>(TILE_COUNT),
            1.0f, 1, LATENT};
        MlaPrologV3::QuantPerTile(packedUb_, quantFp32Ub_, quantWorkUb_, params);
        PipeBarrier<PIPE_V>();

        // PerTileQuant writes the four FP32 scales immediately after the
        // 512-byte quant payload.  Move that 32-byte-aligned block to the ABI
        // scale slot, then overwrite bytes [512, 640) with the BF16 rope.
        LocalTensor<float> scaleSrc = packedUb_[512].ReinterpretCast<float>();
        LocalTensor<float> scaleDst = packedUb_[640].ReinterpretCast<float>();
        Copy(scaleDst, scaleSrc, 8, 1, {1, 1, 8, 8});
        LocalTensor<bfloat16_t> packedRope = packedUb_[512].ReinterpretCast<bfloat16_t>();
        Copy(packedRope, ropeUb_, ROPE, 1, {1, 1, 8, 8});
        PipeBarrier<PIPE_V>();

        SetFlag<HardEvent::V_MTE3>(EVENT_ID2);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID2);
        if (packedRowBytes_ == PACKED_ALIGN) {
            DataCopy(
                packedGm_[static_cast<uint64_t>(owner) * packedRowBytes_],
                packedUb_, PACKED_ALIGN);
        } else {
            DataCopyExtParams out{1, PACKED_PAYLOAD, 0, 0, 0};
            DataCopyPad(
                packedGm_[static_cast<uint64_t>(owner) * packedRowBytes_],
                packedUb_, out);
        }
        SetFlag<HardEvent::MTE3_V>(EVENT_ID2);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID2);
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID2);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID2);
    }

    __aicore__ inline void ProcessTransientRows()
    {
        const uint32_t first = GetBlockIdx() * rowsPerCore_;
        const uint32_t end = first + rowsPerCore_ < rows_ ?
            first + rowsPerCore_ : rows_;
        for (uint32_t row = first; row < end; ++row) {
            PrepareRow(row);
            StoreTransient(row);
        }
    }

    __aicore__ inline void ProcessOwnerRows()
    {
        const uint32_t first = GetBlockIdx() * ownersPerCore_;
        const uint32_t end = first + ownersPerCore_ < ownerRows_ ?
            first + ownersPerCore_ : ownerRows_;
        for (uint32_t owner = first; owner < end; ++owner) {
            const int64_t row = ownerRowsGm_.GetValue(owner);
            if (row < 0 || static_cast<uint64_t>(row) >= rows_) {
                continue;
            }
            PrepareRow(static_cast<uint32_t>(row));
            PackOwner(owner);
        }
    }

    GlobalTensor<bfloat16_t> latentGm_;
    GlobalTensor<bfloat16_t> gammaGm_;
    GlobalTensor<bfloat16_t> cosGm_;
    GlobalTensor<bfloat16_t> sinGm_;
    GlobalTensor<int64_t> ownerRowsGm_;
    GlobalTensor<bfloat16_t> keyGm_;
    GlobalTensor<bfloat16_t> ropeGm_;
    GlobalTensor<int8_t> packedGm_;

    TBuf<> gammaBuf_;
    TBuf<> normBuf_;
    TBuf<> ropeBuf_;
    TBuf<> rmsWorkBuf_;
    TBuf<> sinCosBuf_;
    TBuf<> sinCosWorkBuf_;
    TBuf<> ropeWorkBuf_;
    TBuf<> quantFp32Buf_;
    TBuf<> quantWorkBuf_;
    TBuf<> packedBuf_;
    LocalTensor<bfloat16_t> gammaUb_;
    LocalTensor<bfloat16_t> normUb_;
    LocalTensor<bfloat16_t> ropeUb_;
    LocalTensor<uint8_t> rmsWorkUb_;
    LocalTensor<float> cosUb_;
    LocalTensor<float> sinUb_;
    LocalTensor<uint8_t> sinCosWorkUb_;
    LocalTensor<uint8_t> ropeWorkUb_;
    LocalTensor<float> quantFp32Ub_;
    LocalTensor<uint8_t> quantWorkUb_;
    LocalTensor<int8_t> packedUb_;
    uint32_t rows_ = 0;
    uint32_t ownerRows_ = 0;
    uint32_t rowsPerCore_ = 0;
    uint32_t ownersPerCore_ = 0;
    uint32_t packedRowBytes_ = PACKED_PAYLOAD;
    float epsilon_ = 1e-6f;
};

extern "C" __global__ __aicore__ void packed_kv_prefill_prepare(
    GM_ADDR latentCache, GM_ADDR gamma, GM_ADDR cos, GM_ADDR sin,
    GM_ADDR ownerRows, GM_ADDR key, GM_ADDR keyRope, GM_ADDR ownerPacked,
    GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(data, tiling);
    TPipe pipe;
    if (TILING_KEY_IS(1)) {
        PackedKvPrefillPrepareKernel op;
        op.Init(latentCache, gamma, cos, sin, ownerRows, key, keyRope,
                ownerPacked, data, &pipe);
        op.Process();
    }
}
