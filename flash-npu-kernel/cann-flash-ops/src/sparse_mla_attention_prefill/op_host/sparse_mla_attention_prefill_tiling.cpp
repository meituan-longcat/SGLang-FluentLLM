#include "sparse_mla_attention_prefill_tiling.h"
#include "../op_kernel/sparse_mla_attention_prefill_template_tiling_key.h"

#ifdef DEVICE_OP_TILING_LIB
#include <register/device_op_impl_registry.h>
#else
#include <register/op_impl_registry.h>
#endif
#include "tiling/platform/platform_ascendc.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace optiling {

#ifndef ASCENDC_EXTERN_C
#define ASCENDC_EXTERN_C extern "C"
#endif

constexpr uint32_t QUERY_HEADS = 64;
constexpr uint32_t QUERY_WIDTH = 576;
constexpr uint32_t PAGE_SIZE = 128;
constexpr uint32_t SPARSE_COUNT = 2048;
constexpr uint32_t MAX_QUERY_TOKENS = 16384;

ASCENDC_EXTERN_C ge::graphStatus TilingSparseMlaAttentionPrefill(
    gert::TilingContext *context)
{
    const auto &q = context->GetInputShape(0)->GetStorageShape();
    const auto &kv = context->GetInputShape(1)->GetStorageShape();
    const auto &indices = context->GetInputShape(2)->GetStorageShape();
    const auto &valid = context->GetInputShape(3)->GetStorageShape();
    const auto &pageTable = context->GetInputShape(4)->GetStorageShape();
    const auto &querySeq = context->GetInputShape(5)->GetStorageShape();

    if (context->GetInputDesc(0)->GetDataType() != ge::DT_BF16 ||
        context->GetInputDesc(1)->GetDataType() != ge::DT_BF16 ||
        context->GetInputDesc(2)->GetDataType() != ge::DT_INT32 ||
        context->GetInputDesc(3)->GetDataType() != ge::DT_INT32 ||
        context->GetInputDesc(4)->GetDataType() != ge::DT_INT32 ||
        context->GetInputDesc(5)->GetDataType() != ge::DT_INT32) {
        return ge::GRAPH_FAILED;
    }

    if (q.GetDimNum() != 3 || q.GetDim(0) <= 0 ||
        q.GetDim(0) > MAX_QUERY_TOKENS ||
        q.GetDim(1) != QUERY_HEADS || q.GetDim(2) != QUERY_WIDTH ||
        kv.GetDimNum() != 3 || kv.GetDim(0) <= 0 ||
        kv.GetDim(1) != PAGE_SIZE ||
        kv.GetDim(2) != QUERY_WIDTH ||
        indices.GetDimNum() != 2 || indices.GetDim(0) != q.GetDim(0) ||
        indices.GetDim(1) != SPARSE_COUNT ||
        valid.GetDimNum() != 1 || valid.GetDim(0) != q.GetDim(0) ||
        pageTable.GetDimNum() != 2 || pageTable.GetDim(0) <= 0 ||
        pageTable.GetDim(1) <= 0 || pageTable.GetDim(1) > 10000 ||
        querySeq.GetDimNum() != 1 || querySeq.GetDim(0) != q.GetDim(0)) {
        return ge::GRAPH_FAILED;
    }

    auto platform = platform_ascendc::PlatformAscendC(
        context->GetPlatformInfo());
    const uint32_t totalQuery = static_cast<uint32_t>(q.GetDim(0));
    const uint32_t cores = std::min<uint32_t>(
        platform.GetCoreNumAic(), totalQuery);
    if (cores == 0 || cores > SPARSE_MLA_ATTENTION_PREFILL_MAX_CORES) {
        return ge::GRAPH_FAILED;
    }

    SparseMlaAttentionPrefillTilingData data;
    data.set_totalQuery(totalQuery);
    data.set_sparseCount(SPARSE_COUNT);
    data.set_pageSize(PAGE_SIZE);
    data.set_usedCores(cores);
    data.set_maxBlocks(static_cast<uint32_t>(pageTable.GetDim(1)));
    data.set_scale(*context->GetAttrs()->GetAttrPointer<float>(0));
    const bool *headMajorPtr =
        context->GetAttrs()->GetAttrPointer<bool>(1);
    const bool headMajor = headMajorPtr != nullptr && *headMajorPtr;

    context->SetBlockDim(platform.CalcTschBlockDim(2 * cores, cores,
                                                   2 * cores));
    context->SetTilingKey(GET_TPL_TILING_KEY(headMajor ? 2U : 1U));
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
DEVICE_IMPL_OP_OPTILING(SparseMlaAttentionPrefill)
    .Tiling(TilingSparseMlaAttentionPrefill);
#else
ASCENDC_EXTERN_C ge::graphStatus PrepareSparseMlaAttentionPrefill(
    gert::TilingParseContext *)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(SparseMlaAttentionPrefill)
    .Tiling(TilingSparseMlaAttentionPrefill)
    .TilingParse<SparseMlaAttentionPrefillCompileInfo>(
        PrepareSparseMlaAttentionPrefill);
#endif

} // namespace optiling
