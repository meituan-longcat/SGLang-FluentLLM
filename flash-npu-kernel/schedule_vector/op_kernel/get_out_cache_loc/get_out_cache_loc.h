/*!
 * \file get_out_cache_loc.h
 * \brief
 */

#ifndef __GET_OUT_CACHE_LOC_H
#define __GET_OUT_CACHE_LOC_H

#include "kernel_operator.h"
using namespace AscendC;

template <typename T>
class GetOutCacheLoc {
public:
    __aicore__ inline GetOutCacheLoc()
    {}
    __aicore__ inline GetOutCacheLoc(TPipe *pipe, const GetOutCacheLocTilingData *tiling) : pipe_(pipe), tl_(tiling)
    {}
    __aicore__ inline void Init(GM_ADDR req_to_token, GM_ADDR req_pool_indices, GM_ADDR new_compute_lens,
                                GM_ADDR cache_lens, GM_ADDR out_cache_loc);

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
    __aicore__ inline void LoadNewComputeLen();
    
    TPipe *pipe_ = nullptr;
    const GetOutCacheLocTilingData *tl_;
    constexpr static int64_t BLOCK_SIZE = 32;
    constexpr static int64_t DOUBLE_BUFF = 2;
    GlobalTensor<T> reqPoolIndicesGm_;
    GlobalTensor<T> newComputeLensGm_;
    GlobalTensor<T> cacheLensGm_;
    GlobalTensor<T> outCacheLocGm_;
    GlobalTensor<T> reqToTokenGm_;
    int64_t blockIdx_ = 0;
    int64_t blockFactor_ = 0;
    TQue<QuePosition::VECIN, 1> newComputeLensInQue_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> queBind_;
};

template <typename T>
__aicore__ inline void GetOutCacheLoc<T>::Init(GM_ADDR req_to_token, GM_ADDR req_pool_indices,
    GM_ADDR new_compute_lens, GM_ADDR cache_lens, GM_ADDR out_cache_loc)
{
    blockIdx_ = GetBlockIdx();
    blockFactor_ = tl_->blockFactor;
    reqToTokenGm_.SetGlobalBuffer((__gm__ T*)req_to_token);
    reqPoolIndicesGm_.SetGlobalBuffer((__gm__ T*)req_pool_indices + blockIdx_ * blockFactor_);
    newComputeLensGm_.SetGlobalBuffer((__gm__ T*)new_compute_lens);
    cacheLensGm_.SetGlobalBuffer((__gm__ T*)cache_lens);
    outCacheLocGm_.SetGlobalBuffer((__gm__ T*)out_cache_loc);
    this->pipe_->InitBuffer(newComputeLensInQue_, 1, tl_->batchSize * sizeof(T));
    this->pipe_->InitBuffer(queBind_, DOUBLE_BUFF, tl_->ubFactor * sizeof(T));
}

template <typename T>
__aicore__ inline void GetOutCacheLoc<T>::Process()
{
    auto currBlockFactor = blockFactor_;
    if (blockIdx_ == tl_->usedCoreNum -1) {
        currBlockFactor =  tl_->tailBlockFactor;
    }
    LoadNewComputeLen();
    LocalTensor<T> newComputeLenTensor = newComputeLensInQue_.DeQue<T>();
    TEventID eventIdMTE2ToS = GetTPipePtr()->FetchEventID(HardEvent::MTE2_S);
    SetFlag<HardEvent::MTE2_S>(eventIdMTE2ToS);
    WaitFlag<HardEvent::MTE2_S>(eventIdMTE2ToS);
    int64_t dstOffset = 0;
    for (uint32_t i = 0; i < blockIdx_ * blockFactor_; i++) {
        dstOffset += newComputeLenTensor.GetValue(i);
    }
    for (int64_t idx = 0; idx < currBlockFactor; idx++) {
        auto poolIdx = reqPoolIndicesGm_.GetValue(idx);
        int64_t newComputeLen = newComputeLensGm_.GetValue(blockIdx_ * blockFactor_ + idx);
        auto cacheLen = cacheLensGm_.GetValue(poolIdx);
        auto srcOffset = poolIdx * tl_->poolLen + cacheLen;
        int64_t ubLoopCnt = CeilDiv(newComputeLen, tl_->ubFactor);
        int64_t curCopyLen = tl_->ubFactor;
        for (int64_t loopIdx = 0; loopIdx < ubLoopCnt; loopIdx++) {
            if (loopIdx == ubLoopCnt - 1 && newComputeLen % tl_->ubFactor != 0) {
                curCopyLen = newComputeLen % tl_->ubFactor;
            }
            CopyIn(srcOffset, curCopyLen);
            CopyOut(dstOffset, curCopyLen);
            dstOffset += curCopyLen;
            srcOffset += curCopyLen;
        }
    }
}

template <typename T>
__aicore__ inline void GetOutCacheLoc<T>::LoadNewComputeLen()
{
    LocalTensor<T> newComputeLenTensor = newComputeLensInQue_.AllocTensor<T>();
    uint32_t copyLen = tl_->batchSize;
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(copyLen * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    DataCopyPad(newComputeLenTensor, newComputeLensGm_[0], copyParams, padParams);
    newComputeLensInQue_.EnQue(newComputeLenTensor);
}

template <typename T>
__aicore__ inline void GetOutCacheLoc<T>::CopyIn(const int64_t srcOffset, const int64_t copyLen)
{
    LocalTensor<T> srcLocal = queBind_.AllocTensor<T>();
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(copyLen * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    DataCopyPad(srcLocal, reqToTokenGm_[srcOffset], copyParams, padParams);
    queBind_.EnQue(srcLocal);
}

template <typename T>
__aicore__ inline void GetOutCacheLoc<T>::CopyOut(const int64_t dstOffset, const int64_t copyLen)
{
    LocalTensor<T> dstLocal = queBind_.DeQue<T>();
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(copyLen * sizeof(T)), 0, 0, 0};
    DataCopyPad(outCacheLocGm_[dstOffset], dstLocal, copyParams);
    queBind_.FreeTensor(dstLocal);
}

#endif  // __GET_OUT_CACHE_LOC_H__