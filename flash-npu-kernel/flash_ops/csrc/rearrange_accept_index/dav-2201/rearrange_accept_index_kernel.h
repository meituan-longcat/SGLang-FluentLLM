/*!
 * \file rearrange_accept_index_kernel.h
 * \brief Migrated from MTP op_kernel/rearrange_accept_index/rearrange_accept_index.h.
 *
 * Class body reproduced verbatim. Only changes from the original:
 *   1. `RearrangeAcceptIndexTilingData` now comes from
 *      rearrange_accept_index_tilingdata.h (POD struct with identical field
 *      names/types to the original TILING_DATA_DEF-generated struct).
 *   2. All code wrapped in `flash::RearrangeAcceptIndex` namespace so the
 *      original `using namespace AscendC;` does not leak into the host
 *      translation unit.
 *
 * The direct-invoke kernel entry lives in rearrange_accept_index_entry.h.
 */

#ifndef REARRANGE_ACCEPT_INDEX_KERNEL_H
#define REARRANGE_ACCEPT_INDEX_KERNEL_H

#include "kernel_operator.h"
#include "rearrange_accept_index_tilingdata.h"

namespace flash {
namespace RearrangeAcceptIndex {

using namespace AscendC;

// ===== begin: verbatim copy from MTP op_kernel/.../rearrange_accept_index.h
template <typename T>
class RearrangeAcceptIndex {
    public:
        __aicore__ inline RearrangeAcceptIndex() {}
        __aicore__ inline RearrangeAcceptIndex(TPipe *pipe, const RearrangeAcceptIndexTilingData *tiling): pipe_(pipe), tiling_(tiling) {}
        __aicore__ inline void Init(GM_ADDR accept_index, GM_ADDR accept_length, GM_ADDR output);
        __aicore__ inline void Process();
    private:
        __aicore__ inline void CopyIn(const int64_t srcOffset, const int64_t copyLen);
        __aicore__ inline void CopyOut(const int64_t dstOffset, const int64_t copyLen);
        __aicore__ inline void LoadAcceptLength();

        TPipe *pipe_ = nullptr;
        const RearrangeAcceptIndexTilingData *tiling_;
        constexpr static int64_t BLOCK_SIZE = 32;
        constexpr static int64_t DOUBLE_BUFFER = 2;
        int64_t blockIdx_ = 0;
        int64_t blockFactor_ = 0;
        int64_t loopCount_ = 0;
        int64_t tailCopyLen = 0;
        int64_t inputGmOffset_ = 0;
        int64_t outputGmOffset_ = 0;
        GlobalTensor<T> acceptIndexGm_;
        GlobalTensor<T> acceptLengthGm_;
        GlobalTensor<T> outputGm_;
        TQue<QuePosition::VECIN, 1> acceptLengthInQue_;
        TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> queBind_;
};

template <typename T>
__aicore__ inline void RearrangeAcceptIndex<T>::Init(GM_ADDR accept_index, GM_ADDR accept_length, GM_ADDR output) {
    blockIdx_ = GetBlockIdx();
    blockFactor_ = tiling_->blockFactor;
    acceptIndexGm_.SetGlobalBuffer((__gm__ T*)accept_index);
    acceptLengthGm_.SetGlobalBuffer((__gm__ T*)accept_length);
    outputGm_.SetGlobalBuffer((__gm__ T*)output);
    this->pipe_->InitBuffer(acceptLengthInQue_, 1,  tiling_->batchSize * sizeof(T));
    this->pipe_->InitBuffer(queBind_, DOUBLE_BUFFER, tiling_->ubFactor * sizeof(T));
}

template <typename T>
__aicore__ inline void RearrangeAcceptIndex<T>::Process() {
    auto currBlockFactor = blockFactor_;
    if (blockIdx_ == tiling_->usedCoreNum - 1) {
        currBlockFactor = tiling_->tailBlockFactor;
    }
    int64_t dstOffset = 0;
    LoadAcceptLength();
    LocalTensor<T> acceptLengthTensor = acceptLengthInQue_.DeQue<T>();
    TEventID eventIdMTE2ToS = GetTPipePtr()->FetchEventID(HardEvent::MTE2_S);
    SetFlag<HardEvent::MTE2_S>(eventIdMTE2ToS);
    WaitFlag<HardEvent::MTE2_S>(eventIdMTE2ToS);
    for (uint32_t i = 0; i < blockIdx_ * blockFactor_; i++) {
        dstOffset += acceptLengthTensor.GetValue(i);
    }
    for (int64_t idx = 0; idx < currBlockFactor; idx++) {
        int64_t perBatchAcceptLen = acceptLengthGm_.GetValue(blockIdx_ * blockFactor_ + idx);
        int64_t srcOffset = (blockIdx_ * blockFactor_ + idx) * tiling_->poolLen;
        loopCount_ = perBatchAcceptLen / tiling_->ubFactor;
        tailCopyLen = perBatchAcceptLen % tiling_->ubFactor;
        for (int64_t i = 0; i < loopCount_; i++) {;
            CopyIn(srcOffset, tiling_->ubFactor);
            CopyOut(dstOffset, tiling_->ubFactor);
            srcOffset += tiling_->ubFactor;
            dstOffset += tiling_->ubFactor;
        }
        if (tailCopyLen) {
            CopyIn(srcOffset, tailCopyLen);
            CopyOut(dstOffset, tailCopyLen);
            dstOffset += tailCopyLen;
        }
    }
}

template <typename T>
__aicore__ inline void RearrangeAcceptIndex<T>::LoadAcceptLength()
{
    LocalTensor<T> acceptLengthTensor = acceptLengthInQue_.AllocTensor<T>();
    uint32_t copyLen = tiling_->batchSize;
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(copyLen * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    DataCopyPad(acceptLengthTensor, acceptLengthGm_[0], copyParams, padParams);
    acceptLengthInQue_.EnQue(acceptLengthTensor);
}

template <typename T>
__aicore__ inline void RearrangeAcceptIndex<T>::CopyIn(const int64_t srcOffset, const int64_t copyLen) {
    LocalTensor<T> srcLocal = queBind_.AllocTensor<T>();
    DataCopyExtParams copyParams(1, static_cast<uint32_t>(copyLen * sizeof(T)), 0, 0, 0);
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    DataCopyPad(srcLocal, acceptIndexGm_[srcOffset], copyParams, padParams);
    queBind_.EnQue<QuePosition::VECIN, QuePosition::VECOUT>(srcLocal);
}

template <typename T>
__aicore__ inline void RearrangeAcceptIndex<T>::CopyOut(const int64_t dstOffset, const int64_t copyLen) {
    LocalTensor<T> dstLocal = queBind_.DeQue<T>();
    DataCopyExtParams copyParams(1, static_cast<uint32_t>(copyLen * sizeof(T)), 0, 0, 0);
    DataCopyPad(outputGm_[dstOffset], dstLocal, copyParams);
    queBind_.FreeTensor(dstLocal);
}
// ===== end: verbatim copy =================================================

} // namespace RearrangeAcceptIndex
} // namespace flash

#endif // REARRANGE_ACCEPT_INDEX_KERNEL_H
