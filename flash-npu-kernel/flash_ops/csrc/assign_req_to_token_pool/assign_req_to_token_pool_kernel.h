/*!
 * \file assign_req_to_token_pool_kernel.h
 * \brief Migrated from MTP op_kernel/assign_req_to_token_pool/assign_req_to_token_pool.h.
 *
 * Body of the class is reproduced verbatim. Only the following changes are
 * applied to resolve direct-invoke dependencies (no kernel logic changes):
 *   1. `AssignReqToTokenPoolTilingData` now comes from
 *      assign_req_to_token_pool_tilingdata.h (POD struct with identical field
 *      names/types to the original TILING_DATA_DEF-generated struct).
 *   2. All code wrapped in `flash::AssignReqToTokenPool` namespace so the
 *      original `using namespace AscendC;` does not leak into the host
 *      translation unit that eventually pulls this header in.
 *
 * The direct-invoke kernel entry lives in assign_req_to_token_pool_entry.h.
 */

#ifndef ASSIGN_REQ_TO_TOKEN_POOL_KERNEL_H
#define ASSIGN_REQ_TO_TOKEN_POOL_KERNEL_H

#include "kernel_operator.h"
#include "assign_req_to_token_pool_tilingdata.h"

namespace flash {
namespace AssignReqToTokenPool {

using namespace AscendC;

template <typename T>
class AssignReqToTokenPool {
public:
    __aicore__ inline AssignReqToTokenPool()
    {}
    __aicore__ inline AssignReqToTokenPool(TPipe *pipe, const AssignReqToTokenPoolTilingData *tiling) : pipe_(pipe), tl_(tiling)
    {}
    __aicore__ inline void Init(GM_ADDR req_pool_indices, GM_ADDR extend_lens, GM_ADDR alloced_lens,
                                GM_ADDR out_cache_loc, GM_ADDR req_to_token_pool);

    __aicore__ inline void Process();

private:
    __aicore__ inline int64_t CeilDiv(int64_t a, int64_t b){
        if (b == 0) {
            return a;
        }
        return (a + b - 1) / b;
    }
    __aicore__ inline void CopyIn(const int64_t srcOffset, const int64_t copyLen);
    __aicore__ inline void CopyOut(const int64_t dstOffset, const int64_t copyLen);
    __aicore__ inline void LoadExtendLens();

    TPipe *pipe_ = nullptr;
    const AssignReqToTokenPoolTilingData *tl_;
    constexpr static int64_t BLOCK_SIZE = 32;
    constexpr static int64_t DOUBLE_BUFF = 2;
    GlobalTensor<T> reqPoolIndicesGm_;
    GlobalTensor<T> extendLensGm_;
    GlobalTensor<T> allocedLensGm_;
    GlobalTensor<T> outCacheLocGm_;
    GlobalTensor<T> reqToTokenPoolGm_;
    int64_t blockIdx_ = 0;
    int64_t blockFactor_ = 0;
    TQue<QuePosition::VECIN, 1> extendLensInQue_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> queBind_;
};

template <typename T>
__aicore__ inline void AssignReqToTokenPool<T>::Init(GM_ADDR req_pool_indices, GM_ADDR extend_lens,
    GM_ADDR alloced_lens, GM_ADDR out_cache_loc, GM_ADDR req_to_token_pool)
{
    blockIdx_ = GetBlockIdx();
    blockFactor_ = tl_->blockFactor;
    reqPoolIndicesGm_.SetGlobalBuffer((__gm__ T*)req_pool_indices + blockIdx_ * blockFactor_);
    extendLensGm_.SetGlobalBuffer((__gm__ T*)extend_lens);
    allocedLensGm_.SetGlobalBuffer((__gm__ T*)alloced_lens);
    outCacheLocGm_.SetGlobalBuffer((__gm__ T*)out_cache_loc);
    reqToTokenPoolGm_.SetGlobalBuffer((__gm__ T*)req_to_token_pool);
    this->pipe_->InitBuffer(extendLensInQue_, 1, tl_->batchSize * sizeof(T));
    this->pipe_->InitBuffer(queBind_, DOUBLE_BUFF, tl_->ubFactor * sizeof(T));
}

template <typename T>
__aicore__ inline void AssignReqToTokenPool<T>::Process()
{
    auto currBlockFactor = blockFactor_;
    if (blockIdx_ == tl_->usedCoreNum - 1) {
        currBlockFactor =  tl_->tailBlockFactor;
    }
    int64_t srcOffset = 0;
    LoadExtendLens();
    LocalTensor<T> extendLensTensor = extendLensInQue_.DeQue<T>();
    TEventID eventIdMTE2ToS = GetTPipePtr()->FetchEventID(HardEvent::MTE2_S);
    SetFlag<HardEvent::MTE2_S>(eventIdMTE2ToS);
    WaitFlag<HardEvent::MTE2_S>(eventIdMTE2ToS);
    for (uint32_t i = 0; i < blockIdx_ * blockFactor_; i++) {
        srcOffset += extendLensTensor.GetValue(i);
    }
    for (int64_t idx = 0; idx < currBlockFactor; idx++) {
        auto poolIdx = reqPoolIndicesGm_.GetValue(idx);
        auto extendLen = extendLensGm_.GetValue(blockIdx_ * blockFactor_ + idx);
        auto allocedLen = allocedLensGm_.GetValue(poolIdx);
        auto dstOffset = poolIdx * tl_->poolLen + allocedLen;
        int64_t ubLoopCnt = CeilDiv(extendLen, tl_->ubFactor);
        int64_t curCopyLen = tl_->ubFactor;
        if (extendLen == 0) {
            continue;
        }
        for (int64_t loopIdx = 0; loopIdx < ubLoopCnt; loopIdx++) {
            if (loopIdx == ubLoopCnt - 1 && extendLen % tl_->ubFactor != 0) {
                curCopyLen = extendLen % tl_->ubFactor;
            }
            CopyIn(srcOffset, curCopyLen);
            CopyOut(dstOffset, curCopyLen);
            srcOffset += curCopyLen;
            dstOffset += curCopyLen;
        }
    }
    extendLensInQue_.FreeTensor(extendLensTensor);
}

template <typename T>
__aicore__ inline void AssignReqToTokenPool<T>::LoadExtendLens()
{
    LocalTensor<T> extendLensTensor = extendLensInQue_.AllocTensor<T>();
    uint32_t copyLen = tl_->batchSize;
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(copyLen * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    DataCopyPad(extendLensTensor, extendLensGm_[0], copyParams, padParams);
    extendLensInQue_.EnQue(extendLensTensor);
}


template <typename T>
__aicore__ inline void AssignReqToTokenPool<T>::CopyIn(const int64_t srcOffset, const int64_t copyLen)
{
    LocalTensor<T> srcLocal = queBind_.AllocTensor<T>();
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(copyLen * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    DataCopyPad(srcLocal, outCacheLocGm_[srcOffset], copyParams, padParams);
    queBind_.EnQue(srcLocal);
}

template <typename T>
__aicore__ inline void AssignReqToTokenPool<T>::CopyOut(const int64_t dstOffset, const int64_t copyLen)
{
    LocalTensor<T> dstLocal = queBind_.DeQue<T>();
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(copyLen * sizeof(T)), 0, 0, 0};
    DataCopyPad(reqToTokenPoolGm_[dstOffset], dstLocal, copyParams);
    queBind_.FreeTensor(dstLocal);
}

} // namespace AssignReqToTokenPool
} // namespace flash

#endif // ASSIGN_REQ_TO_TOKEN_POOL_KERNEL_H
