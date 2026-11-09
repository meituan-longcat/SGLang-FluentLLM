# pragma once

#include "kernel_operator.h"

using namespace AscendC;

namespace mlp_split_swiglu_ops {

template <typename T>
class SplitSwiglu {
public:
    __aicore__ SplitSwiglu(const SplitSwigluTilingData& tiling): tiling_(tiling) {}

    __aicore__ ~SplitSwiglu() {
        GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE2>(eventIdVToMte2Ping_);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_V>(eventIdMte2ToVPing_);
        GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE3>(eventIdVToMte3Ping_);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_V>(eventIdMte3ToVPing_);
    }

    __aicore__ void inline InitLocalTensor(TPipe& pipe, const int tokenCapacity) {
        // ub layout: xfloatping(xping) - xfloatpong(xpong) - yfloatping(yping) - yfloatpong(ypong) - sigmodping - sigmoidpong
        pipe.InitBuffer(ubBuf_, MAX_UB_BYTES);
        xFloatPing_     = ubBuf_.template GetWithOffset<float>(tokenCapacity * tiling_.outHidden, 0);
        xPing_          = ubBuf_.template GetWithOffset<T>(tokenCapacity * tiling_.outHidden, UB_SPLIT_DB_BYTES / 2);
        xFloatPong_     = ubBuf_.template GetWithOffset<float>(tokenCapacity * tiling_.outHidden, UB_SPLIT_DB_BYTES);
        xPong_          = ubBuf_.template GetWithOffset<T>(tokenCapacity * tiling_.outHidden, UB_SPLIT_DB_BYTES + UB_SPLIT_DB_BYTES / 2);

        yFloatPing_     = ubBuf_.template GetWithOffset<float>(tokenCapacity * tiling_.outHidden, UB_SPLIT_DB_BYTES * 2);
        yPing_          = ubBuf_.template GetWithOffset<T>(tokenCapacity * tiling_.outHidden, UB_SPLIT_DB_BYTES * 2 + UB_SPLIT_DB_BYTES / 2);

        yFloatPong_     = ubBuf_.template GetWithOffset<float>(tokenCapacity * tiling_.outHidden, UB_SPLIT_DB_BYTES * 3);
        yPong_          = ubBuf_.template GetWithOffset<T>(tokenCapacity * tiling_.outHidden, UB_SPLIT_DB_BYTES * 3 + UB_SPLIT_DB_BYTES / 2);

        sigmoidDstPing_ = ubBuf_.template GetWithOffset<float>(tokenCapacity * tiling_.outHidden, UB_SPLIT_DB_BYTES * 4);
        outPing_        = ubBuf_.template GetWithOffset<T>(tokenCapacity * tiling_.outHidden, UB_SPLIT_DB_BYTES * 4);

        sigmoidDstPong_ = ubBuf_.template GetWithOffset<float>(tokenCapacity * tiling_.outHidden, UB_SPLIT_DB_BYTES * 5);
        outPong_        = ubBuf_.template GetWithOffset<T>(tokenCapacity * tiling_.outHidden, UB_SPLIT_DB_BYTES * 5);
    }

    __aicore__ void inline SplitCore() {
        blockIdx_ = GetBlockIdx();

        const int tokenPerCore = (validToken_ + tiling_.vecCoreNum - 1) / tiling_.vecCoreNum;
        const int tokenPerCoreTail = tokenPerCore - 1;
        const int tailCoreNum = tokenPerCore * tiling_.vecCoreNum - validToken_;
        const int formerCoreNum = tiling_.vecCoreNum - tailCoreNum;

        curToken_ = tokenPerCore;
        if (blockIdx_ >= formerCoreNum) {
            curToken_ = tokenPerCoreTail;
        }
        curTokenOffset_ = blockIdx_ * tokenPerCore;
        if (blockIdx_ > formerCoreNum) {
            curTokenOffset_ = formerCoreNum * tokenPerCore + (blockIdx_ - formerCoreNum) * tokenPerCoreTail;
        }
        curTokenOffset_ += startOffset_;
    }

    __aicore__ void inline Sigmoid(const LocalTensor<float>& src, const int& calcCount, LocalTensor<float>& dst) {
        PipeBarrier<PIPE_V>();
        Muls(dst, src, static_cast<float>(-1.0), calcCount);
        PipeBarrier<PIPE_V>();
        Exp(dst, dst, calcCount);
        PipeBarrier<PIPE_V>();
        Adds(dst, dst, static_cast<float>(1.0), calcCount);
        PipeBarrier<PIPE_V>();
        Div(dst, src, dst, calcCount);
        PipeBarrier<PIPE_V>();
    }
    
    __aicore__ void inline CalcValidTokenNum(GM_ADDR groupList) {
        groupListGm_.SetGlobalBuffer((__gm__ int32_t*)groupList);
        validToken_ = 0;
        startOffset_ = 0;
        for (int i = 0; i < tiling_.expertNum; ++i) {
            int token = groupListGm_.GetValue(i);
            if (i < tiling_.localExpertStart) {
                startOffset_ += token;
            } else if (i < tiling_.localExpertEnd) { // [localExpertStart, localExpertEnd)
                validToken_ += token;
            }
        }
    }

    __aicore__ void inline Process(GM_ADDR x, GM_ADDR groupList, GM_ADDR out, GM_ADDR workspace, TPipe& pipe) {
        // only vector can go through
        if (g_coreType == AIC) {
            return;
        }

        CalcValidTokenNum(groupList);
        if (validToken_ == 0) {
            return;
        }

        SplitCore();
        
        if (curToken_ <= 0) { // 当前core没有需要处理的数据，直接退出即可
            return;
        }

        // global tensor init
        xGm_.SetGlobalBuffer((__gm__ T*)x);
        outGm_.SetGlobalBuffer((__gm__ T*)out);

        // calc loop info
        const int tokenCapacity = UB_SPLIT_DB_BYTES / sizeof(float) / tiling_.outHidden; // TODO:确保能容纳一行;确保也是32B对齐
        const int loop = (curToken_ + tokenCapacity - 1) / tokenCapacity;
        const int tokenLastLoop = curToken_ - (loop - 1) * tokenCapacity;

        InitLocalTensor(pipe, tokenCapacity);

        // ping-pong tensor wrapper
        LocalTensor<T> curX;
        LocalTensor<float> curXFloat;
        LocalTensor<T> curY;
        LocalTensor<float> curYFloat;
        LocalTensor<float> curSigmoidDst;
        LocalTensor<T> curOut;
        
        SetFlag<HardEvent::V_MTE2>(eventIdVToMte2Ping_);
        SetFlag<HardEvent::V_MTE2>(eventIdVToMte2Pong_);

        SetFlag<HardEvent::MTE3_V>(eventIdMte3ToVPing_);
        SetFlag<HardEvent::MTE3_V>(eventIdMte3ToVPong_);

        for (int i = 0; i < loop; ++i) {
            if ((i & 1) == 0) {
                curX = xPing_;
                curXFloat = xFloatPing_;
                curY = yPing_;
                curYFloat = yFloatPing_;
                curSigmoidDst = sigmoidDstPing_;
                curOut = outPing_;
            } else {
                curX = xPong_;
                curXFloat = xFloatPong_;
                curY = yPong_;
                curYFloat = yFloatPong_;
                curSigmoidDst = sigmoidDstPong_;
                curOut = outPong_;
            }
            
            // 反向同步: 等上一次vec算完、再开始本次的MTE2搬入
            if ((i & 1) == 0) {
                WaitFlag<HardEvent::V_MTE2>(eventIdVToMte2Ping_);
            } else {
                WaitFlag<HardEvent::V_MTE2>(eventIdVToMte2Pong_);
            }

            const int start = curTokenOffset_ + i * tokenCapacity; // token维度
            const int token = (i == loop - 1 ? tokenLastLoop : tokenCapacity);
            const int calcCount = token * tiling_.outHidden;

            DataCopyPad(curX, xGm_[start * tiling_.inHidden],
                        {static_cast<uint16_t>(token),
                        static_cast<uint32_t>(tiling_.outHidden * sizeof(T)),
                        /*srcStride*/static_cast<uint32_t>(tiling_.outHidden * sizeof(T)),
                        /*dstStride*/0, 0},
                        {false, 0, 0, 0});
            DataCopyPad(curY, xGm_[start * tiling_.inHidden + tiling_.outHidden],
                        {static_cast<uint16_t>(token),
                        static_cast<uint32_t>(tiling_.outHidden * sizeof(T)),
                        /*srcStride*/static_cast<uint32_t>(tiling_.outHidden * sizeof(T)),
                        /*dstStride*/0, 0},
                        {false, 0, 0, 0});
       
            if ((i & 1) == 0) {
                SetFlag<HardEvent::MTE2_V>(eventIdMte2ToVPing_);
                WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToVPing_);
            } else {
                SetFlag<HardEvent::MTE2_V>(eventIdMte2ToVPong_);
                WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToVPong_);
            }

            PipeBarrier<PIPE_V>();
            Cast(curXFloat, curX, RoundMode::CAST_NONE, calcCount);
            PipeBarrier<PIPE_V>();

            Cast(curYFloat, curY, RoundMode::CAST_NONE, calcCount);
            PipeBarrier<PIPE_V>();
            
            // 反向同步: 等上一次MTE3搬完，本次vector才可以开始计算
            if ((i & 1) == 0) {
                WaitFlag<HardEvent::MTE3_V>(eventIdMte3ToVPing_);
            } else {
                WaitFlag<HardEvent::MTE3_V>(eventIdMte3ToVPong_);
            }

            Sigmoid(curXFloat, calcCount, curSigmoidDst); // 计算: x / (1 + exp(-x))
            PipeBarrier<PIPE_V>();
            
            Mul(curSigmoidDst, curSigmoidDst, curYFloat, calcCount);
            PipeBarrier<PIPE_V>();

            // x、y空间后面都不用了，可以通知mte2开始下次搬入了
            if ((i & 1) == 0) {
                SetFlag<HardEvent::V_MTE2>(eventIdVToMte2Ping_);
            } else {
                SetFlag<HardEvent::V_MTE2>(eventIdVToMte2Pong_);
            }

            Cast(curOut, curSigmoidDst, RoundMode::CAST_RINT, calcCount);
            PipeBarrier<PIPE_V>();

            if ((i & 1) == 0) {
                SetFlag<HardEvent::V_MTE3>(eventIdVToMte3Ping_);
                WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3Ping_);
            } else {
                SetFlag<HardEvent::V_MTE3>(eventIdVToMte3Pong_);
                WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3Pong_);
            }

            DataCopyPad(outGm_[start * tiling_.outHidden], curOut,
                        {static_cast<uint16_t>(token), static_cast<uint32_t>(tiling_.outHidden * sizeof(T)), /*srcStride*/0, /*dstStride*/0, 0});
            if ((i & 1) == 0) {
                SetFlag<HardEvent::MTE3_V>(eventIdMte3ToVPing_);
            } else {
                SetFlag<HardEvent::MTE3_V>(eventIdMte3ToVPong_);
            }
        }

        WaitFlag<HardEvent::V_MTE2>(eventIdVToMte2Ping_);
        WaitFlag<HardEvent::V_MTE2>(eventIdVToMte2Pong_);

        WaitFlag<HardEvent::MTE3_V>(eventIdMte3ToVPing_);
        WaitFlag<HardEvent::MTE3_V>(eventIdMte3ToVPong_);
    }

private:
    const SplitSwigluTilingData& tiling_;
    int blockIdx_;
    int validToken_;

    GlobalTensor<T> xGm_; // [token, hidden]
    GlobalTensor<int32_t> groupListGm_;
    GlobalTensor<T> outGm_; // [token, hidden / 2]

    TBuf<> ubBuf_;
    LocalTensor<T> xPing_;
    LocalTensor<float> xFloatPing_;
    LocalTensor<T> xPong_;
    LocalTensor<float> xFloatPong_;
    LocalTensor<T> yPing_;
    LocalTensor<float> yFloatPing_;
    LocalTensor<T> yPong_;
    LocalTensor<float> yFloatPong_;
    LocalTensor<float> sigmoidDstPing_;
    LocalTensor<float> sigmoidDstPong_;
    LocalTensor<T> outPing_;
    LocalTensor<T> outPong_;

    // core split info
    int curToken_;
    int curTokenOffset_;
    int startOffset_;

    // some consts 
    const uint32_t DB = 2;
    const uint32_t BLOCK_BYTES = 32;
    const uint32_t BLOCK_FLOAT_NUM = 8;
    const uint32_t BLOCK_16BIT_NUM = 16;
    const uint32_t MAX_UB_BYTES = 184 * 1024; // 184kb max for user
    const uint32_t UB_SPLIT_NUM = 3; // out = x * sigmoid(x) * y, 3份ub空间: x 1份、 y 1份、 sigmoid(x) 1份
    const uint32_t UB_SPLIT_BYTES = MAX_UB_BYTES / UB_SPLIT_NUM / 1024 * 1024;
    const uint32_t UB_SPLIT_DB_BYTES = UB_SPLIT_BYTES / DB / 1024 * 1024;

    event_t eventIdMte2ToVPing_ = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>());
    event_t eventIdVToMte2Ping_ = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>());
    event_t eventIdVToMte3Ping_ = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>());
    event_t eventIdMte3ToVPing_ = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>());

    event_t eventIdMte2ToVPong_ = static_cast<event_t>(GetTPipePtr()->FetchEventID<HardEvent::MTE2_V>());
    event_t eventIdVToMte2Pong_ = static_cast<event_t>(GetTPipePtr()->FetchEventID<HardEvent::V_MTE2>());
    event_t eventIdVToMte3Pong_ = static_cast<event_t>(GetTPipePtr()->FetchEventID<HardEvent::V_MTE3>());
    event_t eventIdMte3ToVPong_ = static_cast<event_t>(GetTPipePtr()->FetchEventID<HardEvent::MTE3_V>());
};

}