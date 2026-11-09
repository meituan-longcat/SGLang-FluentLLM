#include "kernel_operator.h"

using namespace AscendC;

namespace {
constexpr float EMPTY_LSE_THRESHOLD = -1.0e30f;
} // namespace

class KvpAttentionMergeKernel {
public:
    __aicore__ inline void Init(
        GM_ADDR attentionOut, GM_ADDR lse, GM_ADDR output,
        const KvpAttentionMergeTilingData &data, TPipe *pipe)
    {
        attentionGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ float *>(attentionOut));
        lseGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(lse));
        outputGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ bfloat16_t *>(output));

        rankCount_ = data.rankCount;
        rowCount_ = data.rowCount;
        hiddenSize_ = data.hiddenSize;
        rowsPerCore_ = data.rowsPerCore;
        alignedRowsPerCore_ = data.alignedRowsPerCore;

        const uint32_t matrixElements = rankCount_ * hiddenSize_;
        pipe->InitBuffer(
            attentionQueue_, 1, matrixElements * sizeof(float));
        pipe->InitBuffer(accumulatorBuf_, hiddenSize_ * sizeof(float));
        pipe->InitBuffer(
            outputQueue_, 1, hiddenSize_ * sizeof(bfloat16_t));
        pipe->InitBuffer(
            lseBuf_, rankCount_ * alignedRowsPerCore_ * sizeof(float));
        pipe->InitBuffer(maxBuf_, alignedRowsPerCore_ * sizeof(float));
        pipe->InitBuffer(sumBuf_, alignedRowsPerCore_ * sizeof(float));

        accumulatorUb_ = accumulatorBuf_.Get<float>();
        lseUb_ = lseBuf_.Get<float>();
        maxUb_ = maxBuf_.Get<float>();
        sumUb_ = sumBuf_.Get<float>();

        lseMte2ToV_ = pipe->AllocEventID<HardEvent::MTE2_V>();
        weightsVToS_ = pipe->AllocEventID<HardEvent::V_S>();
        attentionVToMte2_ = pipe->AllocEventID<HardEvent::V_MTE2>();
        outputMte3ToV_ = pipe->AllocEventID<HardEvent::MTE3_V>();
    }

    __aicore__ inline void Process()
    {
        const uint32_t firstRow = GetBlockIdx() * rowsPerCore_;
        // 910B schedules pure AIV kernels in sibling pairs.  An odd logical
        // block count is rounded up, so the padded AIV must not enter the
        // unsigned remaining-row calculation below.
        if (firstRow >= rowCount_) {
            return;
        }
        const uint32_t remainingRows = rowCount_ - firstRow;
        const uint32_t localRows =
            remainingRows < rowsPerCore_ ? remainingRows : rowsPerCore_;
        ComputeWeights(firstRow, localRows);
        for (uint32_t row = 0; row < localRows; ++row) {
            // SFAD/QSFA use a large negative finite value for an empty rank.
            // If every rank is empty, do not touch attention_out: a backend
            // is allowed to leave NaNs in an output row that has no keys.
            if (maxUb_.GetValue(row) <= EMPTY_LSE_THRESHOLD) {
                Duplicate(accumulatorUb_, 0.0f, hiddenSize_);
                StoreOutput(firstRow + row);
                continue;
            }

            LocalTensor<float> attention =
                CopyAttentionRow(firstRow + row);

            for (uint32_t rank = 0; rank < rankCount_; ++rank) {
                const float weight = lseUb_.GetValue(
                    rank * alignedRowsPerCore_ + row);
                const uint32_t rankOffset = rank * hiddenSize_;
                // Avoid 0 * NaN for an empty rank.  Its softmax weight is
                // exactly zero after exp underflow against any valid rank.
                if (weight == 0.0f) {
                    Duplicate(attention[rankOffset], 0.0f, hiddenSize_);
                } else {
                    Muls(attention[rankOffset], attention[rankOffset],
                         weight, hiddenSize_);
                }
            }
            PipeBarrier<PIPE_V>();
            Adds(accumulatorUb_, attention, 0.0f, hiddenSize_);
            PipeBarrier<PIPE_V>();
            for (uint32_t rank = 1; rank < rankCount_; ++rank) {
                Add(accumulatorUb_, accumulatorUb_,
                    attention[rank * hiddenSize_], hiddenSize_);
                PipeBarrier<PIPE_V>();
            }
            SetFlag<HardEvent::V_MTE2>(attentionVToMte2_);
            WaitFlag<HardEvent::V_MTE2>(attentionVToMte2_);
            attentionQueue_.FreeTensor(attention);
            StoreOutput(firstRow + row);
        }
    }

private:
    __aicore__ inline void StoreOutput(uint32_t row)
    {
        LocalTensor<bfloat16_t> output =
            outputQueue_.AllocTensor<bfloat16_t>();
        Cast(output, accumulatorUb_, RoundMode::CAST_RINT, hiddenSize_);
        outputQueue_.EnQue(output);
        output = outputQueue_.DeQue<bfloat16_t>();
        DataCopy(outputGm_[static_cast<uint64_t>(row) * hiddenSize_],
                 output, hiddenSize_);
        SetFlag<HardEvent::MTE3_V>(outputMte3ToV_);
        WaitFlag<HardEvent::MTE3_V>(outputMte3ToV_);
        outputQueue_.FreeTensor(output);
    }

    __aicore__ inline void ComputeWeights(
        uint32_t firstRow, uint32_t localRows)
    {
        const DataCopyExtParams copy{
            1, static_cast<uint32_t>(localRows * sizeof(float)), 0, 0, 0};
        const DataCopyPadExtParams<float> noPad{false, 0, 0, 0};
        for (uint32_t rank = 0; rank < rankCount_; ++rank) {
            const uint64_t gmOffset =
                static_cast<uint64_t>(rank) * rowCount_ + firstRow;
            DataCopyPad(
                lseUb_[rank * alignedRowsPerCore_], lseGm_[gmOffset],
                copy, noPad);
        }
        SetFlag<HardEvent::MTE2_V>(lseMte2ToV_);
        WaitFlag<HardEvent::MTE2_V>(lseMte2ToV_);

        Adds(maxUb_, lseUb_, 0.0f, localRows);
        PipeBarrier<PIPE_V>();
        for (uint32_t rank = 1; rank < rankCount_; ++rank) {
            Max(maxUb_, maxUb_, lseUb_[rank * alignedRowsPerCore_],
                localRows);
            PipeBarrier<PIPE_V>();
        }

        Duplicate(sumUb_, 0.0f, localRows);
        PipeBarrier<PIPE_V>();
        for (uint32_t rank = 0; rank < rankCount_; ++rank) {
            LocalTensor<float> rankLse =
                lseUb_[rank * alignedRowsPerCore_];
            Sub(rankLse, rankLse, maxUb_, localRows);
            PipeBarrier<PIPE_V>();
            Exp(rankLse, rankLse, localRows);
            PipeBarrier<PIPE_V>();
            Add(sumUb_, sumUb_, rankLse, localRows);
            PipeBarrier<PIPE_V>();
        }
        // This vector clamp is mathematically redundant for finite SFAD
        // inputs, but keeps the generated Exp/Add/Div pipeline faster than
        // feeding Div directly on the current compiler.
        Maxs(sumUb_, sumUb_, 1.0e-30f, localRows);
        PipeBarrier<PIPE_V>();
        for (uint32_t rank = 0; rank < rankCount_; ++rank) {
            LocalTensor<float> rankWeight =
                lseUb_[rank * alignedRowsPerCore_];
            Div(rankWeight, rankWeight, sumUb_, localRows);
        }
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_S>(weightsVToS_);
        WaitFlag<HardEvent::V_S>(weightsVToS_);
    }

    __aicore__ inline LocalTensor<float> CopyAttentionRow(uint32_t row)
    {
        LocalTensor<float> attention =
            attentionQueue_.AllocTensor<float>();
        for (uint32_t rank = 0; rank < rankCount_; ++rank) {
            const uint64_t gmOffset =
                (static_cast<uint64_t>(rank) * rowCount_ + row) *
                hiddenSize_;
            DataCopy(attention[rank * hiddenSize_],
                     attentionGm_[gmOffset], hiddenSize_);
        }
        attentionQueue_.EnQue(attention);
        return attentionQueue_.DeQue<float>();
    }

    GlobalTensor<float> attentionGm_;
    GlobalTensor<float> lseGm_;
    GlobalTensor<bfloat16_t> outputGm_;
    TQue<QuePosition::VECIN, 1> attentionQueue_;
    TBuf<> accumulatorBuf_;
    TQue<QuePosition::VECOUT, 1> outputQueue_;
    TBuf<> lseBuf_;
    TBuf<> maxBuf_;
    TBuf<> sumBuf_;
    LocalTensor<float> accumulatorUb_;
    LocalTensor<float> lseUb_;
    LocalTensor<float> maxUb_;
    LocalTensor<float> sumUb_;
    TEventID lseMte2ToV_;
    TEventID weightsVToS_;
    TEventID attentionVToMte2_;
    TEventID outputMte3ToV_;
    uint32_t rankCount_ = 0;
    uint32_t rowCount_ = 0;
    uint32_t hiddenSize_ = 0;
    uint32_t rowsPerCore_ = 0;
    uint32_t alignedRowsPerCore_ = 0;
};

extern "C" __global__ __aicore__ void kvp_attention_merge(
    GM_ADDR attentionOut, GM_ADDR lse, GM_ADDR output,
    GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(data, tiling);
    TPipe pipe;
    if (TILING_KEY_IS(1)) {
        KvpAttentionMergeKernel op;
        op.Init(attentionOut, lse, output, data, &pipe);
        op.Process();
    }
}
