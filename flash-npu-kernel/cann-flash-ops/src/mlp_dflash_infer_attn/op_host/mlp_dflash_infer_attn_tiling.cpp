/*!
 * \file mlp_dflash_infer_attn_tiling.cpp
 * \brief MlpDflashInferAttn tiling：DFlash PageAttention（FIA sparse_mode=15 的专用算子版）。
 * 仅 arch22 split-fuse 路径：契约校验（原 CheckDflashPagedParams 裁剪）+ FAInferTiling 任务划分。
 */

#include "mlp_dflash_infer_attn_tiling.h"
#ifdef DEVICE_OP_TILING_LIB
#include <register/device_op_impl_registry.h>
#else
#include <register/op_impl_registry.h>
#endif
#ifdef DEVICE_OP_TILING_LIB
// Device tiling reports through the AICPU dump logger; do not pull the
// host-only alog headers into the cross-compiled op-master.
#define ALOG_NOT_FOUND
#endif
#include "error/ops_error.h"

// 兼容层：源算子用 CANN 内部 err/ops_err.h 的 OP_*/OPS_REPORT 宏，本框架用 host_common/ops_log.h 的 OPS_* 宏，做等价映射
#ifndef ASCENDC_EXTERN_C
#define ASCENDC_EXTERN_C extern "C"
#endif
#ifndef OP_CHECK_IF
#define OP_CHECK_IF(COND, LOG_FUNC, EXPR) OPS_ERR_IF(COND, LOG_FUNC, EXPR)
#endif

namespace optiling {

// 输入下标（对齐 def.cpp）
constexpr int QUERY_INDEX = 0;
constexpr int KEY_INDEX = 1;
constexpr int VALUE_INDEX = 2;
constexpr int BLOCK_TABLE_INDEX = 3;
constexpr int ACTUAL_SEQ_Q_INDEX = 4;
constexpr int ACTUAL_SEQ_KV_INDEX = 5;
// 输出下标
constexpr int ATTENTION_OUT_INDEX = 0;
constexpr int SOFTMAX_LSE_INDEX = 1;
// 属性下标（对齐 def.cpp）
constexpr int ATTR_NUM_HEADS_INDEX = 0;
constexpr int ATTR_SCALE_INDEX = 1;
constexpr int ATTR_PRE_TOKENS_INDEX = 2;
constexpr int ATTR_NEXT_TOKENS_INDEX = 3;
constexpr int ATTR_INPUT_LAYOUT_INDEX = 4;
constexpr int ATTR_NUM_KV_HEADS_INDEX = 5;
constexpr int ATTR_BLOCK_SIZE_INDEX = 6;
constexpr int ATTR_SOFTMAX_LSE_FLAG_INDEX = 7;

constexpr int64_t DIM_0 = 0;
constexpr int64_t DIM_1 = 1;
constexpr int64_t DIM_2 = 2;
constexpr int64_t DIM_3 = 3;

static int64_t ClampTokenValue(int64_t token)
{
    if (token > SPARSE_MODE_INT_MAX) {
        return SPARSE_MODE_INT_MAX;
    } else if (token < -(SPARSE_MODE_INT_MAX)) {
        return -(SPARSE_MODE_INT_MAX);
    }
    return token;
}

// DFlash PageAttention 契约校验（自 FIA CheckDflashPagedParams 裁剪：mode 固定，无 pse/mask/learnableSink 输入）
static ge::graphStatus CheckDflashInferParams(gert::TilingContext *context)
{
    auto attrs = context->GetAttrs();
    OP_CHECK_IF(attrs == nullptr,
                OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "Attributes returned from GetAttrs() is nullptr"),
                return ge::GRAPH_FAILED);
    const std::string inputLayout = std::string(attrs->GetAttrPointer<char>(ATTR_INPUT_LAYOUT_INDEX));
    int64_t preToken = *(attrs->GetAttrPointer<int64_t>(ATTR_PRE_TOKENS_INDEX));
    int64_t nextToken = *(attrs->GetAttrPointer<int64_t>(ATTR_NEXT_TOKENS_INDEX));
    int64_t blockSize = *(attrs->GetAttrPointer<int64_t>(ATTR_BLOCK_SIZE_INDEX));
    int64_t numHeads = *(attrs->GetAttrPointer<int64_t>(ATTR_NUM_HEADS_INDEX));
    int64_t kvHeads = *(attrs->GetAttrPointer<int64_t>(ATTR_NUM_KV_HEADS_INDEX));

    OP_CHECK_IF(inputLayout != "TND",
                OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
                                            "MlpDflashInferAttn requires TND input layout, but got %s",
                                            inputLayout.c_str()),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(preToken < 0 || nextToken < 0,
                OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
                                            "preTokens/nextTokens must be non-negative, but got %ld and %ld",
                                            preToken, nextToken),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(preToken > SPARSE_MODE_INT_MAX,
                OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
                                            "preTokens must be <= INT32_MAX, but got %ld", preToken),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF((nextToken % 8) != 0 && nextToken != SPARSE_MODE_INT_MAX,
                OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
                                            "nextTokens must be 0, INT32_MAX(=不限), or a positive multiple of 8, but got %ld",
                                            nextToken),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(numHeads <= 0 || kvHeads < 0 || (kvHeads != 0 && (numHeads % kvHeads) != 0),
                OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
                                            "numHeads(%ld) must be positive and divisible by numKeyValueHeads(%ld)",
                                            numHeads, kvHeads),
                return ge::GRAPH_FAILED);

    // Q/K/V shape 与 dtype
    auto qShape = context->GetInputShape(QUERY_INDEX);
    auto kShape = context->GetInputShape(KEY_INDEX);
    auto vShape = context->GetInputShape(VALUE_INDEX);
    OP_CHECK_IF(qShape == nullptr || kShape == nullptr || vShape == nullptr,
                OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "query/key/value shape is nullptr"),
                return ge::GRAPH_FAILED);
    auto qDtype = context->GetInputDesc(QUERY_INDEX)->GetDataType();
    auto kDtype = context->GetInputDesc(KEY_INDEX)->GetDataType();
    auto vDtype = context->GetInputDesc(VALUE_INDEX)->GetDataType();
    OP_CHECK_IF(kDtype != qDtype || vDtype != qDtype,
                OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "key/value dtype must match query"),
                return ge::GRAPH_FAILED);
    // paged K/V cache：[numBlocks, blockSize, Nkv*D]（3 维）或 [numBlocks, blockSize, Nkv, D]（4 维），
    // 页内 token 连续，与 kernel 的偏移算术一致
    int64_t headDim = qShape->GetStorageShape().GetDim(DIM_2);
    int64_t effKvHeads = (kvHeads == 0) ? numHeads : kvHeads;
    auto checkCacheShape = [&](const gert::Shape &s, const char *tag) -> ge::graphStatus {
        size_t dn = s.GetDimNum();
        if (dn == 3U) {
            OP_CHECK_IF(s.GetDim(DIM_1) != blockSize || s.GetDim(DIM_2) != effKvHeads * headDim,
                        OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
                            "%s cache [numBlocks, blockSize=%ld, Nkv*D=%ld] mismatch block_size(%ld)/Nkv*D(%ld)",
                            tag, s.GetDim(DIM_1), s.GetDim(DIM_2), blockSize, effKvHeads * headDim),
                        return ge::GRAPH_FAILED);
        } else if (dn == 4U) {
            OP_CHECK_IF(s.GetDim(DIM_1) != blockSize || s.GetDim(DIM_2) != effKvHeads ||
                            s.GetDim(DIM_3) != headDim,
                        OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
                            "%s cache [numBlocks, blockSize=%ld, Nkv=%ld, D=%ld] mismatch "
                            "block_size(%ld)/numKeyValueHeads(%ld)/headDim(%ld)",
                            tag, s.GetDim(DIM_1), s.GetDim(DIM_2), s.GetDim(DIM_3),
                            blockSize, effKvHeads, headDim),
                        return ge::GRAPH_FAILED);
        } else {
            OPS_LOG_E(context->GetNodeName(),
                     "%s cache must be 3-dim [numBlocks, blockSize, Nkv*D] or 4-dim [numBlocks, blockSize, Nkv, D]",
                     tag);
            return ge::GRAPH_FAILED;
        }
        return ge::GRAPH_SUCCESS;
    };
    OP_CHECK_IF(blockSize <= 0,
                OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
                                            "block_size attr(%ld) must be positive", blockSize),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(checkCacheShape(kShape->GetStorageShape(), "K") != ge::GRAPH_SUCCESS ||
                    checkCacheShape(vShape->GetStorageShape(), "V") != ge::GRAPH_SUCCESS,
                OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "paged K/V cache shape check failed"),
                return ge::GRAPH_FAILED);
    // 当前只验证过 128 页：一个 512-token stack tile 恰为 4 页，跨页取数路径已按此覆盖
    OP_CHECK_IF(blockSize != 128,
                OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
                                            "MlpDflashInferAttn currently supports block_size 128 only, but got %ld",
                                            blockSize),
                return ge::GRAPH_FAILED);

    // actual_seq_lengths：DFlash 语义必须提供，且 batch 一致
    auto actualQSeq = context->GetOptionalInputTensor(ACTUAL_SEQ_Q_INDEX);
    auto actualKvSeq = context->GetOptionalInputTensor(ACTUAL_SEQ_KV_INDEX);
    OP_CHECK_IF(actualQSeq == nullptr || actualKvSeq == nullptr,
                OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
                                            "actual_seq_lengths and actual_seq_lengths_kv are required"),
                return ge::GRAPH_FAILED);
    int64_t batch = actualQSeq->GetShapeSize();
    OP_CHECK_IF(actualKvSeq->GetShapeSize() != batch,
                OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
                                            "Q/KV sequence-length batch mismatch: %ld vs %ld",
                                            batch, actualKvSeq->GetShapeSize()),
                return ge::GRAPH_FAILED);
    // block_table：[B, maxNumPagesPerReq]，batch 与容量校验
    auto blockTableShape = context->GetInputShape(BLOCK_TABLE_INDEX);
    OP_CHECK_IF(blockTableShape == nullptr || blockTableShape->GetStorageShape().GetDimNum() != 2U,
                OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "block_table must be two-dimensional"),
                return ge::GRAPH_FAILED);
    int64_t blockTableBatch = blockTableShape->GetStorageShape().GetDim(DIM_0);
    int64_t maxBlocksPerBatch = blockTableShape->GetStorageShape().GetDim(DIM_1);
    OP_CHECK_IF(blockTableBatch != batch,
                OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
                                            "block_table batch %ld must match sequence batch %ld",
                                            blockTableBatch, batch),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(maxBlocksPerBatch <= 0,
                OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
                                            "block_table width must be positive, but got %ld", maxBlocksPerBatch),
                return ge::GRAPH_FAILED);
    int64_t logicalKvCapacity = maxBlocksPerBatch * blockSize;
    const int64_t *actualQ = actualQSeq->GetData<int64_t>();
    const int64_t *actualKv = actualKvSeq->GetData<int64_t>();
    if (actualQ == nullptr || actualKv == nullptr) {
        // tiling 下沉场景：q_len<=kv_len 由 device 侧 Plan builder 兜底
        return ge::GRAPH_SUCCESS;
    }
    for (int64_t batchIdx = 0; batchIdx < batch; ++batchIdx) {
        int64_t prevQ = (batchIdx == 0) ? 0 : actualQ[batchIdx - 1];
        int64_t qLength = actualQ[batchIdx] - prevQ;
        int64_t kvLength = actualKv[batchIdx];
        OP_CHECK_IF(qLength <= 0 || kvLength < qLength || kvLength > logicalKvCapacity,
                    OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
                                                "require 0 < q_len <= kv_len <= block_table capacity, but batch %ld "
                                                "got q_len %ld, kv_len %ld, capacity %ld",
                                                batchIdx, qLength, kvLength, logicalKvCapacity),
                    return ge::GRAPH_FAILED);
    }

    // lse 输出 shape（TND 下 [T, N, 1]）
    bool lseFlag = *(attrs->GetAttrPointer<bool>(ATTR_SOFTMAX_LSE_FLAG_INDEX));
    if (lseFlag) {
        auto lseShape = context->GetOutputShape(SOFTMAX_LSE_INDEX);
        OP_CHECK_IF(lseShape == nullptr || lseShape->GetStorageShape().GetDimNum() != 3U,
                    OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "SoftmaxLse shape dim must be 3 in TND"),
                    return ge::GRAPH_FAILED);
        auto t = qShape->GetStorageShape().GetDim(DIM_0);
        auto n = qShape->GetStorageShape().GetDim(DIM_1);
        OP_CHECK_IF(lseShape->GetStorageShape().GetDim(DIM_0) != t ||
                        lseShape->GetStorageShape().GetDim(DIM_1) != n ||
                        lseShape->GetStorageShape().GetDim(DIM_2) != 1U,
                    OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
                                                "SoftmaxLse shape must be [T, N, 1] in TND"),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ConvertContextToParamsFAI(gert::TilingContext *context, FAInferContext &faInfo,
                                                 uint32_t aicoreNum)
{
    (void)aicoreNum;
    auto qDataType = context->GetInputDesc(QUERY_INDEX)->GetDataType();
    auto qShape = context->GetInputShape(QUERY_INDEX);
    auto kShape = context->GetInputShape(KEY_INDEX);
    auto blockTableShape = context->GetInputShape(BLOCK_TABLE_INDEX);
    auto actualQSeq = context->GetOptionalInputTensor(ACTUAL_SEQ_Q_INDEX);
    auto actualKvSeq = context->GetOptionalInputTensor(ACTUAL_SEQ_KV_INDEX);
    auto attrs = context->GetAttrs();

    faInfo.numHeads = static_cast<int32_t>(*(attrs->GetAttrPointer<int64_t>(ATTR_NUM_HEADS_INDEX)));
    int32_t tmpNKv = static_cast<int32_t>(*(attrs->GetAttrPointer<int64_t>(ATTR_NUM_KV_HEADS_INDEX)));
    faInfo.kvHeads = (tmpNKv == 0) ? faInfo.numHeads : tmpNKv;   // 0 = MHA
    faInfo.blockSize = static_cast<int32_t>(*(attrs->GetAttrPointer<int64_t>(ATTR_BLOCK_SIZE_INDEX)));
    faInfo.scaleValue = *(attrs->GetAttrPointer<float>(ATTR_SCALE_INDEX));
    faInfo.preToken = ClampTokenValue(*(attrs->GetAttrPointer<int64_t>(ATTR_PRE_TOKENS_INDEX)));
    faInfo.nextToken = ClampTokenValue(*(attrs->GetAttrPointer<int64_t>(ATTR_NEXT_TOKENS_INDEX)));
    faInfo.lseFlag = *(attrs->GetAttrPointer<bool>(ATTR_SOFTMAX_LSE_FLAG_INDEX));
    faInfo.sparseMode = 15;   // 固定 DFlash paged 语义
    faInfo.maskType = MaskType::NO_MASK;

    faInfo.numBlocks = kShape->GetStorageShape().GetDim(DIM_0);
    faInfo.maxNumBlocksPerBatch = blockTableShape->GetStorageShape().GetDim(DIM_1);
    faInfo.embeddingSize = qShape->GetStorageShape().GetDim(DIM_2);   // TND: [T, N, D]
    faInfo.embeddingSizeV = faInfo.embeddingSize;
    faInfo.dataType = (qDataType == ge::DT_BF16) ? InferDataType::BF16 : InferDataType::FP16;

    int32_t batch = actualQSeq->GetShapeSize();
    faInfo.batch = batch;
    const int64_t *actualSeqQTnd = actualQSeq->GetData<int64_t>();
    const int64_t *actualSeqKvTnd = actualKvSeq->GetData<int64_t>();
    if (actualSeqQTnd != nullptr && actualSeqKvTnd != nullptr) {
        faInfo.qSeqlenList = actualSeqQTnd;
        faInfo.kvSeqlenList = actualSeqKvTnd;
        int64_t maxQSeqlen = 0;
        int64_t maxKVSeqlen = 0;
        for (int32_t batchIdx = 0; batchIdx < batch; batchIdx++) {
            // TND：q 是前缀和；paged 下 kv 是逐 request 绝对长度
            int64_t prevQ = (batchIdx == 0) ? 0 : actualSeqQTnd[batchIdx - 1];
            int64_t qSeqlen = actualSeqQTnd[batchIdx] - prevQ;
            int64_t kvSeqlen = actualSeqKvTnd[batchIdx];
            maxQSeqlen = std::max(maxQSeqlen, qSeqlen);
            maxKVSeqlen = std::max(maxKVSeqlen, kvSeqlen);
        }
        faInfo.maxQSeqlen = maxQSeqlen;
        faInfo.maxKvSeqlen = maxKVSeqlen;
        faInfo.isTilingSink = false;
    } else {
        faInfo.isTilingSink = true;
    }
    faInfo.workspaces = context->GetWorkspaceSizes(1);
    return ge::GRAPH_SUCCESS;
}

ASCENDC_EXTERN_C ge::graphStatus TilingMlpDflashInferAttn(gert::TilingContext *context)
{
    OP_CHECK_IF(context == nullptr,
                OPS_REPORT_VECTOR_INNER_ERR("MlpDflashInferAttn", "Tiling context is null."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckDflashInferParams(context) != ge::GRAPH_SUCCESS,
                OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "DFlash paged contract check failed"),
                return ge::GRAPH_FAILED);
    auto platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_IF(platformInfoPtr == nullptr,
                OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "PlatformInfoPtr is null"),
                return ge::GRAPH_FAILED);
    // kernel 使用 SyncAll，需要 batchmode 模式所有核同时启动，否则多流下可能卡死
    constexpr uint32_t BATCH_MODE_SCHEDULE = 1;
    context->SetScheduleMode(BATCH_MODE_SCHEDULE);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    uint32_t aicoreNum = ascendcPlatform.GetCoreNumAic();

    FAInferTilingData faiTilingData;
    FAInferContext faiContext;
    ConvertContextToParamsFAI(context, faiContext, aicoreNum);
    FAInferTiling faiTiling(faiContext);
    faiTiling.SetCoreNum(aicoreNum);
    auto ret = faiTiling.DoTiling(faiTilingData);
    OP_CHECK_IF(ret != ge::GRAPH_SUCCESS,
                OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "Do fai tiling went wrong"),
                return ge::GRAPH_FAILED);
    faiTilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(faiTilingData.GetDataSize());
    faiContext.workspaces[0] = 16U * 1024U * 1024U +
        static_cast<uint64_t>(faiTiling.GetCoreNum()) * WORKSPACE_BLOCK_SIZE_DB * 4U * 3U * 4U +
        static_cast<uint64_t>(faiTilingData.get_splitLseTotalSize()) +
        static_cast<uint64_t>(faiTilingData.get_splitOTotalSize());
    context->SetBlockDim(faiTiling.GetCoreNum());
    context->SetTilingKey(faiTiling.GetTilingKey());
    return ge::GRAPH_SUCCESS;
}

ASCENDC_EXTERN_C ge::graphStatus TilingPrepareForMlpDflashInferAttn(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

#ifdef DEVICE_OP_TILING_LIB
DEVICE_IMPL_OP_OPTILING(MlpDflashInferAttn)
    .Tiling(TilingMlpDflashInferAttn);
#else
IMPL_OP_OPTILING(MlpDflashInferAttn)
    .TilingInputsDataDependency({ACTUAL_SEQ_Q_INDEX, ACTUAL_SEQ_KV_INDEX},
                                {gert::TilingPlacement::TILING_ON_HOST, gert::TilingPlacement::TILING_ON_AICPU})
    .Tiling(TilingMlpDflashInferAttn)
    .TilingParse<MlpDflashInferAttnCompileInfo>(TilingPrepareForMlpDflashInferAttn); // 向框架注册入口函数
#endif

} // namespace optiling
