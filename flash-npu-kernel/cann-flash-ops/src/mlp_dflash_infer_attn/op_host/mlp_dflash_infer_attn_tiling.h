/*!
 * \file mlp_dflash_infer_attn_tiling.h
 * \brief MlpDflashInferAttn tiling 数据与 split-fuse 任务划分。
 * 移植自 FusedInferAttentionScore 的 split-fuse 路径（sparse_mode=15，DFlash PageAttention），
 * 仅保留 mode15 所需：TND + paged cache + fp16/bf16 + 单 online-softmax 状态（无 FD/Decoding split）。
 * FAInferTilingData 字段布局与 FIA 保持一致，kernel 侧可直接对照 flash_attention_regular.h 消费。
 */

#ifndef MLP_DFLASH_INFER_ATTN_TILING_H
#define MLP_DFLASH_INFER_ATTN_TILING_H

#include <cstdint>
#include <algorithm>
#include <string>
#include "register/tilingdata_base.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {

constexpr int32_t MAX_CORE_NUM_FD = 26;   // 数组维度沿用 FIA（25 + 1 保证 8 字节对齐）

// FlashDecode 专用数组结构：mode15 不走 FD，但字段保留在 tiling data 内，维持与 FIA 一致的布局
BEGIN_TILING_DATA_DEF(coreNode)
TILING_DATA_FIELD_DEF_ARR(int32_t, MAX_CORE_NUM_FD, startBIdx)
TILING_DATA_FIELD_DEF_ARR(int32_t, MAX_CORE_NUM_FD, startN1Idx)
TILING_DATA_FIELD_DEF_ARR(int32_t, MAX_CORE_NUM_FD, startS1Idx)
TILING_DATA_FIELD_DEF_ARR(int32_t, MAX_CORE_NUM_FD, startS2Idx)
TILING_DATA_FIELD_DEF_ARR(int32_t, MAX_CORE_NUM_FD, endBIdx)
TILING_DATA_FIELD_DEF_ARR(int32_t, MAX_CORE_NUM_FD, endN1Idx)
TILING_DATA_FIELD_DEF_ARR(int32_t, MAX_CORE_NUM_FD, endS1Idx)
TILING_DATA_FIELD_DEF_ARR(int32_t, MAX_CORE_NUM_FD, endS2Idx)
TILING_DATA_FIELD_DEF_ARR(int64_t, MAX_CORE_NUM_FD, firstSplitKVTaskLseOffset)
TILING_DATA_FIELD_DEF_ARR(int64_t, MAX_CORE_NUM_FD, firstSplitKVTaskOOffset)
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(coreNodeOp, coreNode)

BEGIN_TILING_DATA_DEF(splitNode)
TILING_DATA_FIELD_DEF_ARR(int32_t, MAX_CORE_NUM_FD, batchIdx)
TILING_DATA_FIELD_DEF_ARR(int32_t, MAX_CORE_NUM_FD, headStartIdx)
TILING_DATA_FIELD_DEF_ARR(int32_t, MAX_CORE_NUM_FD, headEndIdx)
TILING_DATA_FIELD_DEF_ARR(int32_t, MAX_CORE_NUM_FD, qStartIdx)
TILING_DATA_FIELD_DEF_ARR(int32_t, MAX_CORE_NUM_FD, qEndIdx)
TILING_DATA_FIELD_DEF_ARR(int32_t, MAX_CORE_NUM_FD, splitNum)
TILING_DATA_FIELD_DEF_ARR(int64_t, MAX_CORE_NUM_FD, lseTaskOffset)
TILING_DATA_FIELD_DEF_ARR(int64_t, MAX_CORE_NUM_FD, oTaskOffset)
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(splitNodeOp, splitNode)

// split-fuse 主 tiling data：字段与 FIA flash_attention_infer_tiling.h 完全一致
BEGIN_TILING_DATA_DEF(FAInferTilingData)
TILING_DATA_FIELD_DEF(uint32_t, numHeads)
TILING_DATA_FIELD_DEF(uint32_t, embeddingSize)
TILING_DATA_FIELD_DEF(uint32_t, embeddingSizeV)
TILING_DATA_FIELD_DEF(uint32_t, numBlocks)
TILING_DATA_FIELD_DEF(uint32_t, blockSize)
TILING_DATA_FIELD_DEF(uint32_t, maxQSeqlen)
TILING_DATA_FIELD_DEF(uint32_t, maxKvSeqlen)
TILING_DATA_FIELD_DEF(uint32_t, kvHeads)
TILING_DATA_FIELD_DEF(uint32_t, batch)
TILING_DATA_FIELD_DEF(uint32_t, maxNumBlocksPerBatch)
TILING_DATA_FIELD_DEF(uint32_t, firstBatchTaskNum)
TILING_DATA_FIELD_DEF(uint32_t, totalTaskNum)
TILING_DATA_FIELD_DEF(uint32_t, maskType)
TILING_DATA_FIELD_DEF(uint64_t, mm1OutSize)
TILING_DATA_FIELD_DEF(uint64_t, smOnlineOutSize)
TILING_DATA_FIELD_DEF(uint64_t, mm2OutSize)
TILING_DATA_FIELD_DEF(uint64_t, UpdateSize)
TILING_DATA_FIELD_DEF(uint64_t, workSpaceSize)
TILING_DATA_FIELD_DEF(float, scaleValue)
TILING_DATA_FIELD_DEF(uint64_t, pseQ)
TILING_DATA_FIELD_DEF(uint64_t, pseKv)
TILING_DATA_FIELD_DEF(uint32_t, padding3)
TILING_DATA_FIELD_DEF(int64_t, preToken)
TILING_DATA_FIELD_DEF(int64_t, nextToken)
TILING_DATA_FIELD_DEF(uint32_t, sparseMode)
TILING_DATA_FIELD_DEF(uint64_t, splitLseTotalSize)
TILING_DATA_FIELD_DEF(uint64_t, splitOTotalSize)
TILING_DATA_FIELD_DEF(uint32_t, totalSplitNodeNum)
TILING_DATA_FIELD_DEF(uint32_t, needCoreNum)
TILING_DATA_FIELD_DEF(uint32_t, mainLoopTaskNum)
TILING_DATA_FIELD_DEF(uint32_t, tailLoopTaskNum)
TILING_DATA_FIELD_DEF(uint32_t, tailStartBatch)
TILING_DATA_FIELD_DEF(uint32_t, tailStartN2)
TILING_DATA_FIELD_DEF(uint32_t, tailKvNBlockTile)
TILING_DATA_FIELD_DEF_STRUCT(coreNode, coreInfo)
TILING_DATA_FIELD_DEF_STRUCT(splitNode, splitInfo)
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(MlpDflashInferAttn, FAInferTilingData)

const uint32_t SIZE_OF_16BIT = 2;
const uint32_t SIZE_OF_32BIT = 4;
const uint32_t N_SPLIT_HELPER = 2;
const uint32_t MAX_KV_STACK_LEN = 512;
const uint32_t Q_TILE_CEIL = 128;
const uint32_t WORKSPACE_BLOCK_SIZE_DB = Q_TILE_CEIL * MAX_KV_STACK_LEN;
const uint32_t PRELANCH_NUM = 3;
const int64_t SPARSE_MODE_INT_MAX = 2147483647;

enum class MaskType : uint32_t {
    NO_MASK = 0,     // mode15 专用：可见性由 DFlash 区间枚举保证，无 mask
};

enum class InferDataType : uint32_t {
    FP16 = 0,
    BF16 = 1
};

struct FAInferContext {
    int32_t numHeads = 0;
    int32_t embeddingSize = 0;
    int32_t embeddingSizeV = 0;
    int32_t numBlocks = 0;
    int32_t blockSize = 0;
    int32_t kvHeads = 0;
    int32_t batch = 0;
    int64_t maxQSeqlen = 0;
    int64_t maxKvSeqlen = 0;
    int64_t preToken = 0;
    int64_t nextToken = 0;
    int32_t sparseMode = 15;                 // 固定 DFlash paged 语义
    uint32_t maxNumBlocksPerBatch = 0;
    const int64_t *qSeqlenList{nullptr};
    const int64_t *kvSeqlenList{nullptr};
    float scaleValue = 0.0;
    size_t *workspaces{nullptr};
    MaskType maskType = MaskType::NO_MASK;
    InferDataType dataType = InferDataType::FP16;
    bool lseFlag = false;
    bool isTilingSink = false;               // seq 值 host 不可见（tiling 下沉/AICPU）时保守处理
};

class FAInferTiling {
public:
    FAInferTiling() = default;
    explicit FAInferTiling(const FAInferContext &faInfo) : faInfo_(faInfo) {}
    ge::graphStatus DoTiling(FAInferTilingData &tilingdata);
    void SetCoreNum(uint32_t blockNum)
    {
        this->blockNum_ = blockNum;
    }
    uint32_t GetCoreNum()
    {
        return this->blockNum_;
    }
    uint64_t GetTilingKey();

private:
    void FillSplitCoreTilingData(FAInferTilingData &tilingdata);
    void FillWorkSpaceTilingData(FAInferTilingData &faTilingData);
    uint32_t GetQSBlockTile(int64_t kvSeqlen);
    uint32_t GetKSBlockTile(int64_t kvSeqlen);
    uint32_t GetQNBlockTile(uint32_t qSeqlen, uint32_t groupSize);
    void FillBasicTilingData(FAInferTilingData &faTilingData);

private:
    FAInferContext faInfo_;
    uint32_t blockNum_;
};

inline uint32_t FAInferTiling::GetQNBlockTile(uint32_t qSeqlen, uint32_t groupSize)
{
    uint32_t qRowNumCeil = Q_TILE_CEIL;
    uint32_t qNBlockTile = (qSeqlen != 0) ?
        (qRowNumCeil / qSeqlen) / N_SPLIT_HELPER * N_SPLIT_HELPER : Q_TILE_CEIL;
    qNBlockTile = std::min(qNBlockTile, groupSize);
    qNBlockTile = std::max(qNBlockTile, static_cast<uint32_t>(1));
    return qNBlockTile;
}

inline uint32_t FAInferTiling::GetQSBlockTile(int64_t kvSeqlen)
{
    (void)kvSeqlen;
    return Q_TILE_CEIL;
}

inline uint32_t FAInferTiling::GetKSBlockTile(int64_t kvSeqlen)
{
    (void)kvSeqlen;
    return MAX_KV_STACK_LEN;
}

inline void FAInferTiling::FillBasicTilingData(FAInferTilingData &faTilingData)
{
    faTilingData.set_batch(static_cast<uint32_t>(faInfo_.batch));
    faTilingData.set_numHeads(static_cast<uint32_t>(faInfo_.numHeads));
    faTilingData.set_kvHeads(static_cast<uint32_t>(faInfo_.kvHeads));
    faTilingData.set_embeddingSize(static_cast<uint32_t>(faInfo_.embeddingSize));
    faTilingData.set_embeddingSizeV(static_cast<uint32_t>(faInfo_.embeddingSizeV));
    faTilingData.set_numBlocks(static_cast<uint32_t>(faInfo_.numBlocks));
    faTilingData.set_blockSize(static_cast<uint32_t>(faInfo_.blockSize));
    faTilingData.set_maxQSeqlen(faInfo_.maxQSeqlen);
    faTilingData.set_maxKvSeqlen(faInfo_.maxKvSeqlen);
    faTilingData.set_maxNumBlocksPerBatch(faInfo_.maxNumBlocksPerBatch);
    faTilingData.set_maskType(static_cast<uint32_t>(faInfo_.maskType));
    faTilingData.set_scaleValue(faInfo_.scaleValue);
    faTilingData.set_sparseMode(faInfo_.sparseMode);
    faTilingData.set_preToken(static_cast<int64_t>(faInfo_.preToken));
    faTilingData.set_nextToken(static_cast<int64_t>(faInfo_.nextToken));
    faTilingData.set_pseQ(0);
    faTilingData.set_pseKv(0);
}

inline uint64_t FAInferTiling::GetTilingKey()
{
    // 与 FIA split-fuse 的 key 构成一致：mode15 固定 paged + TND + NO_MASK，无 FD/Decoding/NZ/sink 项
    constexpr uint64_t SPLIT_FUSE_BASE_KEY = 5000000000000000000;
    constexpr uint64_t PAGED_CACHE_KEY = 10000000;
    constexpr uint64_t LAYOUTQ_TND_KEY = 200000;
    constexpr uint64_t DTYPE_FP16_KEY = 100;
    constexpr uint64_t DTYPE_BF16_KEY = 200;
    constexpr uint64_t LSE_OUT_ONLY_KEY = 1000;
    uint64_t tilingKey = SPLIT_FUSE_BASE_KEY + PAGED_CACHE_KEY + LAYOUTQ_TND_KEY;
    tilingKey += (faInfo_.dataType == InferDataType::FP16) ? DTYPE_FP16_KEY : DTYPE_BF16_KEY;
    if (faInfo_.lseFlag) {
        tilingKey += LSE_OUT_ONLY_KEY;
    }
    return tilingKey;
}

inline void FAInferTiling::FillWorkSpaceTilingData(FAInferTilingData &faTilingData)
{
    uint64_t mm1OutSize = static_cast<uint64_t>(blockNum_) * WORKSPACE_BLOCK_SIZE_DB *
        SIZE_OF_32BIT * PRELANCH_NUM;
    uint64_t smOnlineOutSize = static_cast<uint64_t>(blockNum_) * WORKSPACE_BLOCK_SIZE_DB *
        SIZE_OF_16BIT * PRELANCH_NUM;
    uint64_t mm2OutSize = static_cast<uint64_t>(blockNum_) * WORKSPACE_BLOCK_SIZE_DB *
        SIZE_OF_32BIT * PRELANCH_NUM;
    uint64_t UpdateSize = static_cast<uint64_t>(blockNum_) * WORKSPACE_BLOCK_SIZE_DB *
        SIZE_OF_32BIT * PRELANCH_NUM;
    uint64_t splitLseTotalSize = 0;
    uint64_t splitOTotalSize = 0;
    if (faInfo_.isTilingSink) {
        // seq 值不可见时按最大任务面保守预留 split 规约空间（kernel 不依赖其内容做任务划分）
        splitLseTotalSize = 2 * static_cast<uint64_t>(blockNum_) * Q_TILE_CEIL * SIZE_OF_32BIT * faInfo_.numHeads;
        uint32_t embeddingSizeV = static_cast<uint32_t>(faInfo_.embeddingSizeV);
        splitOTotalSize = 2 * static_cast<uint64_t>(blockNum_) * Q_TILE_CEIL * embeddingSizeV * SIZE_OF_32BIT *
            faInfo_.numHeads;
        faTilingData.set_splitLseTotalSize(splitLseTotalSize);
        faTilingData.set_splitOTotalSize(splitOTotalSize);
        faTilingData.set_needCoreNum(blockNum_);
    } else {
        splitLseTotalSize = faTilingData.get_splitLseTotalSize();
        splitOTotalSize = faTilingData.get_splitOTotalSize();
    }
    uint64_t workSpaceSize = mm1OutSize + smOnlineOutSize + mm2OutSize + UpdateSize + splitLseTotalSize +
        splitOTotalSize;
    faTilingData.set_mm1OutSize(mm1OutSize);
    faTilingData.set_smOnlineOutSize(smOnlineOutSize);
    faTilingData.set_mm2OutSize(mm2OutSize);
    faTilingData.set_UpdateSize(UpdateSize);
    faTilingData.set_workSpaceSize(workSpaceSize);
}

inline void FAInferTiling::FillSplitCoreTilingData(FAInferTilingData &faTilingData)
{
    // 任务 = batch × (GQA head-group × q-block)，task 号与 kernel 侧 decode 顺序一致
    uint32_t totalTaskNum = 0;
    uint32_t groupSize = faInfo_.numHeads / faInfo_.kvHeads;
    for (int32_t batchIdx = 0; batchIdx < faInfo_.batch; batchIdx++) {
        uint32_t qSeqlen = *(faInfo_.qSeqlenList + batchIdx);
        uint32_t kvSeqlen = *(faInfo_.kvSeqlenList + batchIdx);
        if (batchIdx > 0) {
            // TND：actual_seq_lengths_query 是前缀和；paged 下 kv 长度是逐 request 的绝对值
            uint64_t prevQSeqlenSum = *(faInfo_.qSeqlenList + batchIdx - 1);
            qSeqlen = qSeqlen - prevQSeqlenSum;
        }
        (void)kvSeqlen;
        uint32_t curQNBlockTile = GetQNBlockTile(qSeqlen, groupSize);
        uint32_t qNBlockNumPerGroup = (groupSize + curQNBlockTile - 1) / curQNBlockTile;
        uint32_t curQNBlockNum = qNBlockNumPerGroup * faInfo_.kvHeads;
        uint32_t curQSBlockTile = GetQSBlockTile(kvSeqlen);
        uint32_t curQSBlockNum = (qSeqlen + curQSBlockTile - 1) / curQSBlockTile;
        uint32_t curTaskNum = curQNBlockNum * curQSBlockNum;
        if (batchIdx == 0) {
            faTilingData.set_firstBatchTaskNum(curTaskNum);
        }
        totalTaskNum += curTaskNum;
    }
    faTilingData.set_totalTaskNum(totalTaskNum);
}

inline ge::graphStatus FAInferTiling::DoTiling(FAInferTilingData &tilingdata)
{
    FillBasicTilingData(tilingdata);
    if (!faInfo_.isTilingSink) {
        FillSplitCoreTilingData(tilingdata);
    }
    FillWorkSpaceTilingData(tilingdata);
    return ge::GRAPH_SUCCESS;
}

// MlpDflashInferAttn 编译信息
struct MlpDflashInferAttnCompileInfo {
    uint32_t aicNum = 0;
};

} // namespace optiling

#endif // MLP_DFLASH_INFER_ATTN_TILING_H
