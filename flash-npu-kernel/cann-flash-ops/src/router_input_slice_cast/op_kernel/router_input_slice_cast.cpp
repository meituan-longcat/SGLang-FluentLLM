#include "kernel_operator.h"

using namespace AscendC;

template <typename InputT>
class RouterInputSliceCastKernel {
public:
    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR y, uint32_t rows, uint32_t inputCols,
        uint32_t sliceStart, uint32_t sliceLength, uint32_t rowsPerCore,
        TPipe *pipe)
    {
        inputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ InputT *>(x));
        outputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(y));
        rows_ = rows;
        inputCols_ = inputCols;
        sliceStart_ = sliceStart;
        sliceLength_ = sliceLength;
        rowsPerCore_ = rowsPerCore;

        const uint32_t maxLocalElements = rowsPerCore_ * sliceLength_;
        pipe->InitBuffer(inputBuf_, maxLocalElements * sizeof(InputT));
        pipe->InitBuffer(outputBuf_, maxLocalElements * sizeof(float));
        inputUb_ = inputBuf_.Get<InputT>();
        outputUb_ = outputBuf_.Get<float>();
        mte2ToV_ = pipe->AllocEventID<HardEvent::MTE2_V>();
        vToMte3_ = pipe->AllocEventID<HardEvent::V_MTE3>();
    }

    __aicore__ inline void Process()
    {
        const uint32_t core = GetBlockIdx();
        const uint32_t firstRow = core * rowsPerCore_;
        const uint32_t remainingRows = rows_ - firstRow;
        const uint32_t localRows =
            remainingRows < rowsPerCore_ ? remainingRows : rowsPerCore_;
        const uint32_t localElements = localRows * sliceLength_;
        const uint64_t inputOffset =
            static_cast<uint64_t>(firstRow) * inputCols_ + sliceStart_;
        const uint64_t outputOffset =
            static_cast<uint64_t>(firstRow) * sliceLength_;

        // Gather all rows assigned to this core with one 2-D GM->UB transfer.
        // The UB layout is contiguous, so Cast and the final store each need
        // only one vector/DMA instruction sequence per core.
        DataCopyParams inputCopy{
            static_cast<uint16_t>(localRows),
            static_cast<uint16_t>(sliceLength_ * sizeof(InputT) / 32),
            static_cast<uint16_t>(
                (inputCols_ - sliceLength_) * sizeof(InputT) / 32),
            0};
        DataCopy(inputUb_, inputGm_[inputOffset], inputCopy);
        SetFlag<HardEvent::MTE2_V>(mte2ToV_);
        WaitFlag<HardEvent::MTE2_V>(mte2ToV_);

        Cast(outputUb_, inputUb_, RoundMode::CAST_NONE, localElements);
        SetFlag<HardEvent::V_MTE3>(vToMte3_);
        WaitFlag<HardEvent::V_MTE3>(vToMte3_);
        DataCopy(outputGm_[outputOffset], outputUb_, localElements);
    }

private:
    GlobalTensor<InputT> inputGm_;
    GlobalTensor<float> outputGm_;
    TBuf<> inputBuf_;
    TBuf<> outputBuf_;
    LocalTensor<InputT> inputUb_;
    LocalTensor<float> outputUb_;
    TEventID mte2ToV_;
    TEventID vToMte3_;
    uint32_t rows_ = 0;
    uint32_t inputCols_ = 0;
    uint32_t sliceStart_ = 0;
    uint32_t sliceLength_ = 0;
    uint32_t rowsPerCore_ = 0;
};

template <typename InputT>
__aicore__ inline void RunRouterInputSliceCast(
    GM_ADDR x, GM_ADDR y, const RouterInputSliceCastTilingData &data, TPipe *pipe)
{
    RouterInputSliceCastKernel<InputT> op;
    op.Init(x, y, data.rows, data.inputCols, data.sliceStart,
            data.sliceLength, data.rowsPerCore, pipe);
    op.Process();
}

extern "C" __global__ __aicore__ void router_input_slice_cast(
    GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(data, tiling);
    TPipe pipe;
    if (TILING_KEY_IS(1)) {
        if constexpr (ORIG_DTYPE_X == DT_FLOAT16) {
            RunRouterInputSliceCast<half>(x, y, data, &pipe);
        } else {
            RunRouterInputSliceCast<bfloat16_t>(x, y, data, &pipe);
        }
    }
}
