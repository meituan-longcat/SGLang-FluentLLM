#include "sparse_flash_attention_decode_tiling.h"
#include "../op_kernel/sparse_flash_attention_decode_template_tiling_key.h"

#ifdef DEVICE_OP_TILING_LIB
#include <register/device_op_impl_registry.h>
#else
#include <register/op_impl_registry.h>
#endif
#include "tiling/platform/platform_ascendc.h"

#include <algorithm>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace optiling {

#ifndef ASCENDC_EXTERN_C
#define ASCENDC_EXTERN_C extern "C"
#endif

constexpr uint32_t VALID_CHUNKS_INDEX = 4;
constexpr uint32_t QUERY_LAYOUT_ATTR_INDEX = 2;
constexpr uint32_t QUERY_HEADS = 64;
constexpr uint32_t QUERY_WIDTH = 576;
constexpr uint32_t LATENT_WIDTH = 512;
constexpr uint32_t PACKED_INT8_WIDTH = 656;
constexpr uint32_t ALIGNED_PACKED_INT8_WIDTH = 672;
constexpr uint32_t ROPE_WIDTH = 64;
constexpr uint32_t SPARSE_COUNT = 2048;
// PrepareTaskRange temporarily stores one int32_t per query in the 64 KiB
// logits UB before that buffer is reused by the attention pipeline.
constexpr uint32_t MAX_QUERY_TOKENS = 64 * 1024 / sizeof(int32_t);
constexpr int64_t ATTENTION_MODE_MLA = 2;

static bool ParseQueryShape(const gert::Shape &q, const char *layout,
                            uint32_t &totalQuery, uint32_t &rankSize,
    uint32_t &localHeads, uint32_t &layoutKey)
{
    if (layout == nullptr) return false;
    const bool tnd = std::strcmp(layout, "TND") == 0;
    const bool tndComm = std::strcmp(layout, "TND_KVP_COMM") == 0;
    if (tnd || tndComm) {
        if (q.GetDimNum() != 3 || q.GetDim(0) <= 0 ||
            q.GetDim(0) > MAX_QUERY_TOKENS ||
            q.GetDim(1) != QUERY_HEADS ||
            q.GetDim(2) != 512) {
            return false;
        }
        totalQuery = static_cast<uint32_t>(q.GetDim(0));
        rankSize = 1;
        localHeads = QUERY_HEADS;
        layoutKey = tndComm ? SFAD_LAYOUT_TND_KVP_COMM : SFAD_LAYOUT_TND;
        return true;
    }
    const bool packedComm =
        std::strcmp(layout, "RANK_MAJOR_TND_PACKED_KVP_COMM") == 0;
    if (!packedComm || (q.GetDimNum() != 4 && q.GetDimNum() != 5)) {
        return false;
    }
    if (q.GetDim(1) <= 0 || q.GetDim(1) > MAX_QUERY_TOKENS) {
        return false;
    }
    rankSize = q.GetDim(0);
    totalQuery = static_cast<uint32_t>(q.GetDim(1));
    localHeads = q.GetDim(q.GetDimNum() - 2);
    if (q.GetDim(q.GetDimNum() - 1) != QUERY_WIDTH ||
        rankSize == 0 || localHeads == 0 ||
        rankSize * localHeads != QUERY_HEADS) {
        return false;
    }
    for (size_t dim = 2; dim + 2 < q.GetDimNum(); ++dim) {
        if (q.GetDim(dim) != 1) return false;
    }
    layoutKey = SFAD_LAYOUT_RANK_MAJOR_TND_PACKED_KVP_COMM;
    return true;
}

ASCENDC_EXTERN_C ge::graphStatus TilingSparseFlashAttentionDecode(
    gert::TilingContext *context)
{
    const auto &q = context->GetInputShape(0)->GetStorageShape();
    const auto &k = context->GetInputShape(1)->GetStorageShape();
    const auto &v = context->GetInputShape(2)->GetStorageShape();
    const auto &indices = context->GetInputShape(3)->GetStorageShape();
    const auto &validChunks =
        context->GetInputShape(VALID_CHUNKS_INDEX)->GetStorageShape();
    const auto *blocksShape = context->GetOptionalInputShape(5);
    const auto *seqQShape = context->GetOptionalInputShape(6);
    const auto *seqKvShape = context->GetOptionalInputShape(7);
    const auto *queryRopeShape = context->GetOptionalInputShape(8);
    const auto *keyRopeShape = context->GetOptionalInputShape(9);
    if (blocksShape == nullptr || seqQShape == nullptr ||
        seqKvShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const auto &blocks = blocksShape->GetStorageShape();
    const auto &seqQ = seqQShape->GetStorageShape();
    const auto &seqKv = seqKvShape->GetStorageShape();
    auto platform = platform_ascendc::PlatformAscendC(
        context->GetPlatformInfo());
    const int64_t sparseMode =
        *context->GetAttrs()->GetAttrPointer<int64_t>(4);
    const int64_t preTokens =
        *context->GetAttrs()->GetAttrPointer<int64_t>(5);
    const int64_t nextTokens =
        *context->GetAttrs()->GetAttrPointer<int64_t>(6);
    const int64_t attentionMode =
        *context->GetAttrs()->GetAttrPointer<int64_t>(7);
    const int64_t sparsePrepMode =
        *context->GetAttrs()->GetAttrPointer<int64_t>(9);
    uint32_t totalQuery = 0;
    uint32_t queryRankSize = 0;
    uint32_t queryLocalHeads = 0;
    uint32_t queryLayoutKey = 0;
    const char *queryLayout =
        context->GetAttrs()->GetAttrPointer<char>(QUERY_LAYOUT_ATTR_INDEX);
    const char *kvLayout = context->GetAttrs()->GetAttrPointer<char>(3);
    const auto queryDtype = context->GetInputDesc(0)->GetDataType();
    const auto keyDtype = context->GetInputDesc(1)->GetDataType();
    const auto valueDtype = context->GetInputDesc(2)->GetDataType();
    const bool packedInt8 = keyDtype == ge::DT_INT8;
    if (!ParseQueryShape(q, queryLayout, totalQuery, queryRankSize,
                         queryLocalHeads, queryLayoutKey)) {
        return ge::GRAPH_FAILED;
    }
    const bool packedQuery =
        queryLayoutKey == SFAD_LAYOUT_RANK_MAJOR_TND_PACKED_KVP_COMM;
    if (indices.GetDimNum() != 3 || indices.GetDim(0) != totalQuery ||
        indices.GetDim(1) != 1 || indices.GetDim(2) != SPARSE_COUNT ||
        validChunks.GetDimNum() != 1 ||
        validChunks.GetDim(0) != totalQuery ||
        blocks.GetDimNum() != 2 || blocks.GetDim(0) <= 0 ||
        blocks.GetDim(1) <= 0 || blocks.GetDim(1) >= 10000 ||
        seqQ.GetDimNum() != 1 || seqKv.GetDimNum() != 1 ||
        seqQ.GetDim(0) <= 0 || seqQ.GetDim(0) > 128 ||
        seqKv.GetDim(0) != seqQ.GetDim(0) ||
        blocks.GetDim(0) < seqQ.GetDim(0)) {
        return ge::GRAPH_FAILED;
    }
    const uint32_t batchSize = static_cast<uint32_t>(seqQ.GetDim(0));
    if (packedQuery) {
        if (queryRopeShape != nullptr) return ge::GRAPH_FAILED;
    } else {
        if (queryRopeShape == nullptr) return ge::GRAPH_FAILED;
        const auto &queryRope = queryRopeShape->GetStorageShape();
        if (queryRope.GetDimNum() != 3 ||
            queryRope.GetDim(0) != totalQuery ||
            queryRope.GetDim(1) != QUERY_HEADS ||
            queryRope.GetDim(2) != ROPE_WIDTH) {
            return ge::GRAPH_FAILED;
        }
    }
    if (kvLayout == nullptr || std::strcmp(kvLayout, "PA_BSND") != 0 ||
        k.GetDimNum() != 4 || v.GetDimNum() != 4 ||
        k.GetDim(1) != 128 || k.GetDim(2) != 1) {
        return ge::GRAPH_FAILED;
    }
    if (k.GetDim(0) != v.GetDim(0) ||
        k.GetDim(1) != v.GetDim(1) ||
        k.GetDim(2) != v.GetDim(2) ||
        k.GetDim(3) != v.GetDim(3) || keyDtype != valueDtype) {
        return ge::GRAPH_FAILED;
    }
    if (packedInt8) {
        if (queryDtype != ge::DT_BF16 ||
            (k.GetDim(3) != PACKED_INT8_WIDTH &&
             k.GetDim(3) != ALIGNED_PACKED_INT8_WIDTH) ||
            keyRopeShape != nullptr)
            return ge::GRAPH_FAILED;
    } else {
        if ((keyDtype != ge::DT_FLOAT16 && keyDtype != ge::DT_BF16) ||
            queryDtype != keyDtype || k.GetDim(3) != LATENT_WIDTH ||
            keyRopeShape == nullptr) {
            return ge::GRAPH_FAILED;
        }
        const auto &keyRope = keyRopeShape->GetStorageShape();
        if (keyRope.GetDimNum() != 4 ||
            keyRope.GetDim(0) != k.GetDim(0) ||
            keyRope.GetDim(1) != k.GetDim(1) ||
            keyRope.GetDim(2) != k.GetDim(2) ||
            keyRope.GetDim(3) != ROPE_WIDTH) {
            return ge::GRAPH_FAILED;
        }
    }
    if (*context->GetAttrs()->GetAttrPointer<int64_t>(1) != 1 ||
        totalQuery == 0 || totalQuery > MAX_QUERY_TOKENS ||
        (sparseMode != 0 && sparseMode != 3) ||
        preTokens != INT64_MAX || nextTokens != INT64_MAX ||
        attentionMode != ATTENTION_MODE_MLA ||
        (sparsePrepMode != SFAD_PREP_MODE_FULL_SCAN &&
         sparsePrepMode != SFAD_PREP_MODE_VALID_CHUNKS &&
         sparsePrepMode != SFAD_PREP_MODE_BOUNDED_HOLES) ||
        (sparsePrepMode == SFAD_PREP_MODE_BOUNDED_HOLES && !packedInt8)) {
        return ge::GRAPH_FAILED;
    }
    const uint32_t cores = std::min<uint32_t>(
        platform.GetCoreNumAic(), totalQuery);
    if (cores == 0 || cores > SPARSE_FLASH_ATTENTION_DECODE_MAX_CORES)
        return ge::GRAPH_FAILED;

    SparseFlashAttentionDecodeTilingData data;
    data.set_totalQuery(totalQuery);
    data.set_sparseCount(indices.GetDim(2));
    data.set_pageSize(k.GetDim(1));
    data.set_maxBlocks(blocks.GetDim(1));
    data.set_usedCores(cores);
    data.set_batchSize(batchSize);
    data.set_scale(*context->GetAttrs()->GetAttrPointer<float>(0));
    data.set_returnLse(*context->GetAttrs()->GetAttrPointer<bool>(8));
    data.set_sparseMode(static_cast<uint32_t>(sparseMode));
    data.set_packedRowBytes(
        packedInt8 ? static_cast<uint32_t>(k.GetDim(3)) : 0U);
    data.set_queryRankSize(queryRankSize);
    data.set_queryLocalHeads(queryLocalHeads);

    context->SetBlockDim(platform.CalcTschBlockDim(2 * cores, cores,
                                                   2 * cores));
    const uint64_t sparsePrepKey = static_cast<uint64_t>(sparsePrepMode);
    const uint64_t tilingKey = GET_TPL_TILING_KEY(
        queryLayoutKey, 2U, sparsePrepKey);
    context->SetTilingKey(tilingKey);
    context->SetScheduleMode(1);
    constexpr size_t coreData = 3 * 4 * 2 * 64 +
        3 * (64 * 512 * 4 + 64 * 512 * 2 + 64 * 512 * 4 +
             4 * 128 * (512 + 64) * 2);
    context->GetWorkspaceSizes(1)[0] = platform.GetLibApiWorkSpaceSize() +
        static_cast<size_t>(cores) * ((coreData + 511) / 512 * 512) +
        static_cast<size_t>(totalQuery) * 128 +
        static_cast<size_t>(cores) * 64;
    data.SaveToBuffer(context->GetRawTilingData()->GetData(),
                      context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(data.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

#ifdef DEVICE_OP_TILING_LIB
DEVICE_IMPL_OP_OPTILING(SparseFlashAttentionDecode)
    .Tiling(TilingSparseFlashAttentionDecode);
#else
ASCENDC_EXTERN_C ge::graphStatus PrepareSparseFlashAttentionDecode(
    gert::TilingParseContext *)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(SparseFlashAttentionDecode)
    .Tiling(TilingSparseFlashAttentionDecode)
    .TilingParse<SparseFlashAttentionDecodeCompileInfo>(
        PrepareSparseFlashAttentionDecode);
#endif

} // namespace optiling
