#include "kvp_attention_merge_tiling.h"

#include <cstdint>
#include <limits>

#include "error/ops_error.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
namespace {
constexpr uint32_t KVP_RANK_COUNT = 8;
constexpr uint32_t HIDDEN_SIZE = 512;
constexpr uint32_t FP32_BYTES = 4;
constexpr uint32_t BF16_BYTES = 2;
constexpr uint32_t FP32_BLOCK_ELEMENTS = 8;
constexpr uint64_t RESERVED_UB_BYTES = 8192;

constexpr uint64_t AlignUp(uint64_t value, uint64_t alignment)
{
    return (value + alignment - 1) / alignment * alignment;
}
} // namespace

static ge::graphStatus TilingKvpAttentionMerge(gert::TilingContext *context)
{
    if (context == nullptr) {
        OPS_LOG_E("KvpAttentionMerge", "tiling context is null.");
        return ge::GRAPH_FAILED;
    }
    const char *nodeName = context->GetNodeName();
    const auto *attentionTensor = context->GetInputShape(0);
    const auto *lseTensor = context->GetInputShape(1);
    const auto *attentionDesc = context->GetInputDesc(0);
    const auto *lseDesc = context->GetInputDesc(1);
    if (attentionTensor == nullptr || lseTensor == nullptr ||
        attentionDesc == nullptr || lseDesc == nullptr) {
        OPS_LOG_E(nodeName,
                  "attention_out or lse shape/descriptor is null.");
        return ge::GRAPH_FAILED;
    }

    const auto &attention = attentionTensor->GetStorageShape();
    const auto &lse = lseTensor->GetStorageShape();
    if (attention.GetDimNum() != 3 || lse.GetDimNum() != 2 ||
        attentionDesc->GetDataType() != ge::DT_FLOAT ||
        lseDesc->GetDataType() != ge::DT_FLOAT) {
        OPS_LOG_E(nodeName,
                  "attention_out must be 3-D FP32 and lse must be 2-D FP32.");
        return ge::GRAPH_FAILED;
    }

    const int64_t rankCount = attention.GetDim(0);
    const int64_t rowCount = attention.GetDim(1);
    const int64_t hiddenSize = attention.GetDim(2);
    if (rankCount != KVP_RANK_COUNT || hiddenSize != HIDDEN_SIZE ||
        lse.GetDim(0) != rankCount || lse.GetDim(1) != rowCount ||
        rowCount <= 0 ||
        static_cast<uint64_t>(rowCount) >
            std::numeric_limits<uint32_t>::max()) {
        OPS_LOG_E(nodeName,
                  "expected attention_out=[8, rows, 512] and lse=[8, rows] "
                  "with 0 < rows <= UINT32_MAX, but got rank count %ld, "
                  "row count %ld, hidden size %ld and lse shape [%ld, %ld].",
                  rankCount, rowCount, hiddenSize, lse.GetDim(0),
                  lse.GetDim(1));
        return ge::GRAPH_FAILED;
    }

    auto *platformInfo = context->GetPlatformInfo();
    if (platformInfo == nullptr) {
        OPS_LOG_E(nodeName, "platform info is null.");
        return ge::GRAPH_FAILED;
    }
    platform_ascendc::PlatformAscendC platform(platformInfo);
    const uint32_t availableCores = platform.GetCoreNumAiv();
    uint64_t ubBytes = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubBytes);
    if (availableCores == 0 || ubBytes <= RESERVED_UB_BYTES) {
        OPS_LOG_E(nodeName,
                  "invalid AIV/UB capacity: available cores=%u, UB=%lu bytes.",
                  availableCores, ubBytes);
        return ge::GRAPH_FAILED;
    }

    const uint64_t rows = static_cast<uint64_t>(rowCount);
    const uint64_t rowsPerCore =
        (rows + availableCores - 1) / availableCores;
    const uint64_t usedCores =
        (rows + rowsPerCore - 1) / rowsPerCore;
    const uint64_t alignedRows =
        AlignUp(rowsPerCore, FP32_BLOCK_ELEMENTS);

    // One row from all 8 ranks, one FP32 accumulator and one BF16 output
    // dominate UB.  The remaining buffers hold rank-major LSE and max/sum for
    // every row assigned to this core.  Keeping all LSE rows resident lets the
    // kernel calculate their softmax in one vector batch instead of repeating
    // tiny GM copies and event handshakes per row.
    const uint64_t matrixElements =
        static_cast<uint64_t>(KVP_RANK_COUNT) * HIDDEN_SIZE;
    const uint64_t requiredUb =
        matrixElements * FP32_BYTES + HIDDEN_SIZE * FP32_BYTES +
        HIDDEN_SIZE * BF16_BYTES +
        static_cast<uint64_t>(KVP_RANK_COUNT) * alignedRows * FP32_BYTES +
        2 * static_cast<uint64_t>(alignedRows) * FP32_BYTES;
    constexpr uint64_t U32_MAX = std::numeric_limits<uint32_t>::max();
    if (rowsPerCore > U32_MAX || usedCores > U32_MAX ||
        alignedRows > U32_MAX ||
        requiredUb > ubBytes - RESERVED_UB_BYTES) {
        OPS_LOG_E(nodeName,
                  "input does not fit the per-core UB design: required=%lu "
                  "bytes, available=%lu bytes, rowsPerCore=%lu and "
                  "alignedRowsPerCore=%lu.",
                  requiredUb, ubBytes - RESERVED_UB_BYTES, rowsPerCore,
                  alignedRows);
        return ge::GRAPH_FAILED;
    }

    auto *workspaceSizes = context->GetWorkspaceSizes(1);
    auto *rawTilingData = context->GetRawTilingData();
    if (workspaceSizes == nullptr || rawTilingData == nullptr) {
        OPS_LOG_E(nodeName,
                  "workspace sizes or raw tiling data is null.");
        return ge::GRAPH_FAILED;
    }

    KvpAttentionMergeTilingData data;
    data.set_rankCount(KVP_RANK_COUNT);
    data.set_rowCount(static_cast<uint32_t>(rows));
    data.set_hiddenSize(HIDDEN_SIZE);
    data.set_rowsPerCore(static_cast<uint32_t>(rowsPerCore));
    data.set_alignedRowsPerCore(static_cast<uint32_t>(alignedRows));
    if (data.GetDataSize() > rawTilingData->GetCapacity()) {
        OPS_LOG_E(nodeName, "tiling data size %zu exceeds capacity %zu.",
                  data.GetDataSize(), rawTilingData->GetCapacity());
        return ge::GRAPH_FAILED;
    }

    context->SetBlockDim(static_cast<uint32_t>(usedCores));
    context->SetTilingKey(1);
    // This is a pure AIV kernel using only basic vector/DMA instructions.
    // Advertising the platform matmul workspace makes the generated launcher
    // emit an unnecessary matmul::clearWorkspace path on some CANN versions.
    workspaceSizes[0] = 0;
    data.SaveToBuffer(rawTilingData->GetData(), rawTilingData->GetCapacity());
    rawTilingData->SetDataSize(data.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus PrepareKvpAttentionMerge(gert::TilingParseContext *)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(KvpAttentionMerge)
    .Tiling(TilingKvpAttentionMerge)
    .TilingParse<KvpAttentionMergeCompileInfo>(PrepareKvpAttentionMerge);

} // namespace optiling
