#include "kernel_operator.h"

using namespace AscendC;

class SelectLocalTopkIndicesKernel {
    GlobalTensor<int32_t> localIndicesGm;
    GlobalTensor<int32_t> topkPosGm;
    GlobalTensor<int32_t> outputGm;
    GlobalTensor<int32_t> validChunksGm;

    TBuf<> localBuf;
    TBuf<> positionBuf;
    TBuf<> floatBuf;
    TBuf<> offsetBuf;
    TBuf<> gatheredBuf;
    TBuf<> outputBuf;
    TBuf<> lowMaskBuf;
    TBuf<> highMaskBuf;

    LocalTensor<int32_t> localUb;
    LocalTensor<int32_t> positionUb;
    LocalTensor<float> floatUb;
    LocalTensor<int32_t> offsetUb;
    LocalTensor<int32_t> gatheredUb;
    LocalTensor<int32_t> outputUb;
    LocalTensor<uint8_t> lowMaskUb;
    LocalTensor<uint8_t> highMaskUb;

    TEventID mte2ToV;
    TEventID vToS;
    TEventID vToMte3;
    TEventID sToMte3;
    TEventID mte3ToMte2;

    uint32_t totalTokens;
    uint32_t localCount;
    uint32_t topkCount;
    uint32_t usedCores;
    int32_t rankStart;
    int32_t rankEnd;

    __aicore__ inline void ProcessToken(uint32_t token)
    {
        const uint64_t localBase =
            static_cast<uint64_t>(token) * localCount;
        const uint64_t topkBase =
            static_cast<uint64_t>(token) * topkCount;

        DataCopy(localUb, localIndicesGm[localBase], localCount);
        DataCopy(positionUb, topkPosGm[topkBase], topkCount);
        SetFlag<HardEvent::MTE2_V>(mte2ToV);
        WaitFlag<HardEvent::MTE2_V>(mte2ToV);

        Cast(floatUb, positionUb, RoundMode::CAST_ROUND, topkCount);
        PipeBarrier<PIPE_V>();
        CompareScalar(lowMaskUb, floatUb, static_cast<float>(rankStart),
                      CMPMODE::GE, topkCount);
        CompareScalar(highMaskUb, floatUb,
                      static_cast<float>(rankEnd),
                      CMPMODE::LT, topkCount);
        PipeBarrier<PIPE_V>();
        And(lowMaskUb.template ReinterpretCast<uint16_t>(),
            lowMaskUb.template ReinterpretCast<uint16_t>(),
            highMaskUb.template ReinterpretCast<uint16_t>(),
            topkCount / 16);

        Adds(offsetUb, positionUb, -rankStart, topkCount);
        PipeBarrier<PIPE_V>();
        // Invalid positions must not address outside the local-index UB row.
        Select(offsetUb.template ReinterpretCast<float>(), lowMaskUb,
               offsetUb.template ReinterpretCast<float>(), 0.0f,
               SELMODE::VSEL_TENSOR_SCALAR_MODE, topkCount);
        PipeBarrier<PIPE_V>();
        Muls(offsetUb, offsetUb, static_cast<int32_t>(sizeof(int32_t)),
             topkCount);
        PipeBarrier<PIPE_V>();
        Gather(gatheredUb, localUb,
               offsetUb.template ReinterpretCast<uint32_t>(), 0, topkCount);
        PipeBarrier<PIPE_V>();

        // A selected local slot may itself be padding (-1). Do not count or
        // emit it as a valid sparse index.
        Cast(floatUb, gatheredUb, RoundMode::CAST_ROUND, topkCount);
        PipeBarrier<PIPE_V>();
        CompareScalar(highMaskUb, floatUb, -1.0f, CMPMODE::GT, topkCount);
        PipeBarrier<PIPE_V>();
        And(lowMaskUb.template ReinterpretCast<uint16_t>(),
            lowMaskUb.template ReinterpretCast<uint16_t>(),
            highMaskUb.template ReinterpretCast<uint16_t>(),
            topkCount / 16);
        Duplicate(outputUb, static_cast<int32_t>(-1), topkCount);
        PipeBarrier<PIPE_V>();

        GatherMaskParams params{1, 1, 8, 0};
        uint64_t validCount = 0;
        GatherMask(outputUb.template ReinterpretCast<float>(),
                   gatheredUb.template ReinterpretCast<float>(),
                   lowMaskUb.template ReinterpretCast<uint32_t>(), true,
                   topkCount, params, validCount);
        SetFlag<HardEvent::V_S>(vToS);
        WaitFlag<HardEvent::V_S>(vToS);
        // valid_chunks is a scheduling cost, so a non-empty tail chunk must
        // contribute one unit as well.
        offsetUb.SetValue(0, static_cast<int32_t>((validCount + 127) / 128));

        SetFlag<HardEvent::V_MTE3>(vToMte3);
        SetFlag<HardEvent::S_MTE3>(sToMte3);
        WaitFlag<HardEvent::V_MTE3>(vToMte3);
        WaitFlag<HardEvent::S_MTE3>(sToMte3);
        DataCopy(outputGm[topkBase], outputUb, topkCount);
        DataCopyExtParams countCopy{
            1, static_cast<uint32_t>(sizeof(int32_t)), 0, 0, 0};
        DataCopyPad(validChunksGm[token], offsetUb, countCopy);
        SetFlag<HardEvent::MTE3_MTE2>(mte3ToMte2);
        WaitFlag<HardEvent::MTE3_MTE2>(mte3ToMte2);
    }

public:
    __aicore__ inline void Init(
        GM_ADDR localIndices, GM_ADDR topkPos, GM_ADDR output,
        GM_ADDR validChunks, uint32_t totalTokensIn,
        uint32_t localCountIn, uint32_t topkCountIn,
        uint32_t usedCoresIn, int32_t rankStartIn, TPipe *pipe)
    {
        localIndicesGm.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t *>(localIndices));
        topkPosGm.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t *>(topkPos));
        outputGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(output));
        validChunksGm.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t *>(validChunks));
        totalTokens = totalTokensIn;
        localCount = localCountIn;
        topkCount = topkCountIn;
        usedCores = usedCoresIn;
        rankStart = rankStartIn;
        rankEnd = rankStartIn + static_cast<int32_t>(localCountIn);

        pipe->InitBuffer(localBuf, localCount * sizeof(int32_t));
        pipe->InitBuffer(positionBuf, topkCount * sizeof(int32_t));
        pipe->InitBuffer(floatBuf, topkCount * sizeof(float));
        pipe->InitBuffer(offsetBuf, topkCount * sizeof(int32_t));
        pipe->InitBuffer(gatheredBuf, topkCount * sizeof(int32_t));
        pipe->InitBuffer(outputBuf, topkCount * sizeof(int32_t));
        pipe->InitBuffer(lowMaskBuf, topkCount / 8);
        pipe->InitBuffer(highMaskBuf, topkCount / 8);
        localUb = localBuf.Get<int32_t>();
        positionUb = positionBuf.Get<int32_t>();
        floatUb = floatBuf.Get<float>();
        offsetUb = offsetBuf.Get<int32_t>();
        gatheredUb = gatheredBuf.Get<int32_t>();
        outputUb = outputBuf.Get<int32_t>();
        lowMaskUb = lowMaskBuf.Get<uint8_t>();
        highMaskUb = highMaskBuf.Get<uint8_t>();

        mte2ToV = pipe->AllocEventID<HardEvent::MTE2_V>();
        vToS = pipe->AllocEventID<HardEvent::V_S>();
        vToMte3 = pipe->AllocEventID<HardEvent::V_MTE3>();
        sToMte3 = pipe->AllocEventID<HardEvent::S_MTE3>();
        mte3ToMte2 = pipe->AllocEventID<HardEvent::MTE3_MTE2>();
    }

    __aicore__ inline void Process()
    {
        const uint32_t core = GetBlockIdx();
        for (uint32_t token = core; token < totalTokens;
             token += usedCores) {
            ProcessToken(token);
        }
    }
};

extern "C" __global__ __aicore__ void select_local_topk_indices(
    GM_ADDR topkIndicesLocal, GM_ADDR topkPos, GM_ADDR topkIndices,
    GM_ADDR validChunks, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(data, tiling);
    TPipe pipe;
    SelectLocalTopkIndicesKernel op;
    op.Init(topkIndicesLocal, topkPos, topkIndices, validChunks,
            data.totalTokens, data.localCount, data.topkCount,
            data.usedCores, data.rankStart, &pipe);
    if (TILING_KEY_IS(1)) op.Process();
}
