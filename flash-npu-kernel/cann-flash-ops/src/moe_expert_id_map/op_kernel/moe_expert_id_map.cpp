#include "kernel_operator.h"

using namespace AscendC;

class MoeExpertIdMapKernel {
public:
    __aicore__ inline void Init(
        GM_ADDR logicalIds, GM_ADDR expertMap, GM_ADDR physicalIds,
        const MoeExpertIdMapTilingData &data, TPipe *pipe)
    {
        logicalIdsGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t *>(logicalIds));
        expertMapGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t *>(expertMap));
        physicalIdsGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t *>(physicalIds));
        idCount_ = data.idCount;
        mapSize_ = data.mapSize;

        pipe->InitBuffer(idsBuf_, data.alignedIdCount * sizeof(int32_t));
        pipe->InitBuffer(mapBuf_, data.alignedMapSize * sizeof(int32_t));
        pipe->InitBuffer(outputBuf_, data.alignedIdCount * sizeof(int32_t));
        idsUb_ = idsBuf_.Get<int32_t>();
        mapUb_ = mapBuf_.Get<int32_t>();
        outputUb_ = outputBuf_.Get<int32_t>();
        mte2ToS_ = pipe->AllocEventID<HardEvent::MTE2_S>();
        sToMte3_ = pipe->AllocEventID<HardEvent::S_MTE3>();
    }

    __aicore__ inline void Process()
    {
        if (idCount_ == 0) {
            return;
        }

        const DataCopyExtParams idsCopy{
            1, static_cast<uint32_t>(idCount_ * sizeof(int32_t)), 0, 0, 0};
        const DataCopyExtParams mapCopy{
            1, static_cast<uint32_t>(mapSize_ * sizeof(int32_t)), 0, 0, 0};
        const DataCopyPadExtParams noPad{false, 0, 0, 0};
        DataCopyPad(idsUb_, logicalIdsGm_, idsCopy, noPad);
        DataCopyPad(mapUb_, expertMapGm_, mapCopy, noPad);
        SetFlag<HardEvent::MTE2_S>(mte2ToS_);
        WaitFlag<HardEvent::MTE2_S>(mte2ToS_);

        const int32_t maxId = static_cast<int32_t>(mapSize_ - 1);
        for (uint32_t i = 0; i < idCount_; ++i) {
            int32_t id = idsUb_.GetValue(i);
            id = id < 0 ? 0 : id;
            id = id > maxId ? maxId : id;
            outputUb_.SetValue(i, mapUb_.GetValue(static_cast<uint32_t>(id)));
        }

        SetFlag<HardEvent::S_MTE3>(sToMte3_);
        WaitFlag<HardEvent::S_MTE3>(sToMte3_);
        DataCopyExtParams outputCopy{
            1, static_cast<uint32_t>(idCount_ * sizeof(int32_t)), 0, 0, 0};
        DataCopyPad(physicalIdsGm_, outputUb_, outputCopy);
    }

private:
    GlobalTensor<int32_t> logicalIdsGm_;
    GlobalTensor<int32_t> expertMapGm_;
    GlobalTensor<int32_t> physicalIdsGm_;
    TBuf<> idsBuf_;
    TBuf<> mapBuf_;
    TBuf<> outputBuf_;
    LocalTensor<int32_t> idsUb_;
    LocalTensor<int32_t> mapUb_;
    LocalTensor<int32_t> outputUb_;
    TEventID mte2ToS_;
    TEventID sToMte3_;
    uint32_t idCount_ = 0;
    uint32_t mapSize_ = 0;
};

extern "C" __global__ __aicore__ void moe_expert_id_map(
    GM_ADDR logicalIds, GM_ADDR expertMap, GM_ADDR physicalIds,
    GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(data, tiling);
    TPipe pipe;
    if (TILING_KEY_IS(1)) {
        MoeExpertIdMapKernel op;
        op.Init(logicalIds, expertMap, physicalIds, data, &pipe);
        op.Process();
    }
}
