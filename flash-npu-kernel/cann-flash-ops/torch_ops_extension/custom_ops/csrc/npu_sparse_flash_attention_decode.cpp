#include <torch/library.h>
#include "ops_common.h"

namespace custom {
using Result = std::tuple<at::Tensor, at::Tensor, at::Tensor>;

constexpr const char *PACKED_COMM_QUERY_LAYOUT =
    "RANK_MAJOR_TND_PACKED_KVP_COMM";
constexpr const char *TND_QUERY_LAYOUT = "TND";
constexpr const char *TND_COMM_QUERY_LAYOUT = "TND_KVP_COMM";
constexpr int64_t QUERY_HEADS = 64;
constexpr int64_t QUERY_WIDTH = 576;
constexpr int64_t LATENT_WIDTH = 512;
constexpr int64_t ROPE_WIDTH = 64;
constexpr int64_t PAGE_SIZE = 128;
constexpr int64_t SPARSE_COUNT = 2048;
constexpr int64_t MAX_QUERY_TOKENS = 16 * 1024;
constexpr int64_t ATTENTION_MODE_MLA = 2;
constexpr int64_t SPARSE_PREP_MODE_FULL_SCAN = 0;
constexpr int64_t SPARSE_PREP_MODE_VALID_CHUNKS = 1;
constexpr int64_t SPARSE_PREP_MODE_BOUNDED_HOLES = 2;

void CheckExecutionModes(int64_t attentionMode, int64_t sparsePrepMode,
                         const at::Tensor &key)
{
    TORCH_CHECK(attentionMode == ATTENTION_MODE_MLA,
        "attention_mode must be 2 (MLA), but got ", attentionMode, ".");
    TORCH_CHECK(sparsePrepMode == SPARSE_PREP_MODE_FULL_SCAN ||
                sparsePrepMode == SPARSE_PREP_MODE_VALID_CHUNKS ||
                sparsePrepMode == SPARSE_PREP_MODE_BOUNDED_HOLES,
        "sparse_prep_mode must be 0 (full scan), 1 (valid_chunks bounded scan), "
        "or 2 (trusted-bound holes), but got ",
        sparsePrepMode, ".");
    TORCH_CHECK(sparsePrepMode != SPARSE_PREP_MODE_BOUNDED_HOLES ||
                key.scalar_type() == at::kChar,
        "sparse_prep_mode=2 requires packed INT8 KV cache.");
}

bool IsPackedLayout(c10::string_view layout)
{
    return layout == PACKED_COMM_QUERY_LAYOUT;
}

bool IsCommLayout(c10::string_view layout)
{
    return layout == TND_COMM_QUERY_LAYOUT ||
           layout == PACKED_COMM_QUERY_LAYOUT;
}

Result MakeDecodeOutputs(const at::Tensor &query, const at::Tensor &key,
                         c10::string_view layoutQuery, bool lse)
{
    TORCH_CHECK(layoutQuery == TND_QUERY_LAYOUT || IsCommLayout(layoutQuery),
        "layout_query must be TND, TND_KVP_COMM, or "
        "RANK_MAJOR_TND_PACKED_KVP_COMM, but got ", layoutQuery, ".");
    auto options = query.options().dtype(at::kFloat);
    const bool packed = IsPackedLayout(layoutQuery);
    TORCH_CHECK(key.dim() == 4 && key.size(2) == 1,
        "key must have a rank-4 PA_BSND shape with one KV head.");
    int64_t totalQuery = 0;
    int64_t queryHeads = QUERY_HEADS;
    at::Tensor out;
    if (packed) {
        TORCH_CHECK(query.dim() == 4 || query.dim() == 5,
            "RANK_MAJOR_TND_PACKED_KVP_COMM query must have rank 4 or 5.");
        TORCH_CHECK(query.size(-1) == QUERY_WIDTH,
            "RANK_MAJOR_TND_PACKED_KVP_COMM query width must be 576.");
        for (int64_t dim = 2; dim + 2 < query.dim(); ++dim) {
            TORCH_CHECK(query.size(dim) == 1,
                "RANK_MAJOR_TND_PACKED_KVP_COMM intermediate dimensions must be 1.");
        }
        totalQuery = query.size(1);
        queryHeads = query.size(0) * query.size(-2);
        TORCH_CHECK(queryHeads == QUERY_HEADS,
            "RANK_MAJOR_TND_PACKED_KVP_COMM query must contain 64 total heads.");
        out = at::empty({queryHeads, totalQuery, LATENT_WIDTH}, options);
    } else {
        TORCH_CHECK(query.dim() == 3 && query.size(1) == QUERY_HEADS &&
                    query.size(2) == LATENT_WIDTH,
            "TND query must have shape [T, 64, 512].");
        totalQuery = query.size(0);
        out = IsCommLayout(layoutQuery) ?
            at::empty({queryHeads, totalQuery, LATENT_WIDTH}, options) :
            at::empty(query.sizes(), options);
    }
    TORCH_CHECK(totalQuery > 0 && totalQuery <= MAX_QUERY_TOKENS,
        "query token count must be in [1, 16384], but got ", totalQuery, ".");
    c10::SmallVector<int64_t, SIZE> shape = {0};
    if (lse) {
        const int64_t kvHeads = key.size(2);
        // KVP_COMM changes only O.  max/sum remain contiguous [N2,T,G] so
        // SFAD does not pay for thousands of scalar-sized strided stores.
        shape = c10::SmallVector<int64_t, SIZE>{
            kvHeads, totalQuery, queryHeads / kvHeads};
    }
    return {out, at::empty(shape, options), at::empty(shape, options)};
}

Result NpuSparseFlashAttentionDecode(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &value,
    const at::Tensor &sparseIndices, double scaleValue,
    const at::Tensor &validChunks,
    const c10::optional<at::Tensor> &blockTable,
    const c10::optional<at::Tensor> &actualSeqLengthsQuery,
    const c10::optional<at::Tensor> &actualSeqLengthsKv,
    const c10::optional<at::Tensor> &queryRope,
    const c10::optional<at::Tensor> &keyRope, int64_t sparseBlockSize,
    c10::string_view layoutQuery, c10::string_view layoutKv, int64_t sparseMode,
    int64_t preTokens, int64_t nextTokens, int64_t attentionMode,
    bool returnLse, int64_t sparsePrepMode)
{
    CheckExecutionModes(attentionMode, sparsePrepMode, key);
    const bool packedQuery = IsPackedLayout(layoutQuery);
    const bool packedInt8 = key.scalar_type() == at::kChar;
    TORCH_CHECK(key.dim() == 4 && value.dim() == 4,
        "key and value must have rank-4 shapes.");
    TORCH_CHECK(value.scalar_type() == key.scalar_type(),
        "key and value dtypes must match.");
    TORCH_CHECK(key.sizes() == value.sizes(),
        "key and value must have identical PA_BSND shapes.");
    TORCH_CHECK(key.size(1) == PAGE_SIZE && key.size(2) == 1,
        "key/value page size must be 128 and KV heads must be 1.");
    TORCH_CHECK(layoutKv == "PA_BSND",
        "layout_kv must be PA_BSND.");
    if (packedInt8) {
        TORCH_CHECK(query.scalar_type() == at::kBFloat16,
            "packed INT8 KV cache requires a BF16 query.");
        TORCH_CHECK(key.size(3) == 656 || key.size(3) == 672,
            "packed INT8 KV rows must have width 656 or 672.");
        TORCH_CHECK(!keyRope.has_value(),
            "packed INT8 KV cache contains RoPE and scales; key_rope must be None.");
    } else {
        TORCH_CHECK((key.scalar_type() == at::kHalf ||
                     key.scalar_type() == at::kBFloat16) &&
                    key.scalar_type() == query.scalar_type(),
            "BF16/FP16 key, value, and query dtypes must match.");
        TORCH_CHECK(key.size(3) == 512,
            "BF16/FP16 key/value rows must have width 512.");
        TORCH_CHECK(keyRope.has_value(),
            "BF16/FP16 KV cache requires key_rope.");
    }
    auto [out, max, sum] =
        MakeDecodeOutputs(query, key, layoutQuery, returnLse);
    const int64_t totalQuery = packedQuery ? query.size(1) : query.size(0);
    TORCH_CHECK(sparseIndices.scalar_type() == at::kInt &&
                sparseIndices.dim() == 3 &&
                sparseIndices.size(0) == totalQuery &&
                sparseIndices.size(1) == 1 &&
                sparseIndices.size(2) == SPARSE_COUNT,
        "sparse_indices must be int32 with shape [T, 1, 2048].");
    TORCH_CHECK(validChunks.scalar_type() == at::kInt,
        "valid_chunks dtype must be int32.");
    TORCH_CHECK(validChunks.dim() == 1 &&
        validChunks.size(0) == totalQuery,
        "valid_chunks must have shape [T].");
    TORCH_CHECK(blockTable.has_value() && actualSeqLengthsQuery.has_value() &&
                actualSeqLengthsKv.has_value(),
        "block_table, actual_seq_lengths_query, and actual_seq_lengths_kv are required.");
    const auto &blocks = blockTable.value();
    const auto &seqQ = actualSeqLengthsQuery.value();
    const auto &seqKv = actualSeqLengthsKv.value();
    TORCH_CHECK(blocks.scalar_type() == at::kInt && blocks.dim() == 2 &&
                blocks.size(0) > 0 && blocks.size(1) > 0 &&
                blocks.size(1) < 10000,
        "block_table must be int32 with shape [B_table, max_blocks], where max_blocks is in [1, 9999].");
    TORCH_CHECK(seqQ.scalar_type() == at::kInt &&
                seqKv.scalar_type() == at::kInt &&
                seqQ.dim() == 1 && seqKv.dim() == 1 &&
                seqQ.size(0) > 0 && seqQ.size(0) <= 128 &&
                seqKv.size(0) == seqQ.size(0) &&
                blocks.size(0) >= seqQ.size(0),
        "sequence lengths must be matching int32 [B] tensors with B in [1, 128], and block_table must contain at least B rows.");
    if (packedQuery) {
        TORCH_CHECK(!queryRope.has_value(),
            "rank-major packed query already contains RoPE; query_rope must be None.");
    } else {
        TORCH_CHECK(queryRope.has_value(),
            "TND query requires query_rope.");
        const auto &qRope = queryRope.value();
        TORCH_CHECK(qRope.scalar_type() == query.scalar_type() &&
                    qRope.dim() == 3 && qRope.size(0) == totalQuery &&
                    qRope.size(1) == QUERY_HEADS &&
                    qRope.size(2) == ROPE_WIDTH,
            "query_rope must match query dtype and have shape [T, 64, 64].");
    }
    if (!packedInt8) {
        const auto &kRope = keyRope.value();
        TORCH_CHECK(kRope.scalar_type() == query.scalar_type() &&
                    kRope.dim() == 4 &&
                    kRope.size(0) == key.size(0) &&
                    kRope.size(1) == PAGE_SIZE && kRope.size(2) == 1 &&
                    kRope.size(3) == ROPE_WIDTH,
            "key_rope must match query dtype and have shape [num_pages, 128, 1, 64].");
    }
    std::string qLayout(layoutQuery), kvLayout(layoutKv);
    char *qLayoutPtr = qLayout.data();
    char *kvLayoutPtr = kvLayout.data();
    EXEC_NPU_CMD_V1(aclnnSparseFlashAttentionDecodeV2, query, key, value,
        sparseIndices, validChunks, blockTable, actualSeqLengthsQuery,
        actualSeqLengthsKv, queryRope, keyRope, scaleValue, sparseBlockSize,
        qLayoutPtr, kvLayoutPtr, sparseMode, preTokens, nextTokens,
        attentionMode, returnLse, sparsePrepMode, out, max, sum);
    return {out, max, sum};
}

Result NpuSparseFlashAttentionDecodeMeta(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &,
    const at::Tensor &, double, const at::Tensor &,
    const c10::optional<at::Tensor> &,
    const c10::optional<at::Tensor> &, const c10::optional<at::Tensor> &,
    const c10::optional<at::Tensor> &, const c10::optional<at::Tensor> &,
    int64_t, c10::string_view layoutQuery,
    c10::string_view, int64_t, int64_t, int64_t,
    int64_t attentionMode, bool returnLse, int64_t sparsePrepMode)
{
    CheckExecutionModes(attentionMode, sparsePrepMode, key);
    return MakeDecodeOutputs(query, key, layoutQuery, returnLse);
}
} // namespace custom

TORCH_LIBRARY_IMPL(custom, PrivateUse1, m) {
    m.impl("npu_sparse_flash_attention_decode", &custom::NpuSparseFlashAttentionDecode);
}
TORCH_LIBRARY_IMPL(custom, Meta, m) {
    m.impl("npu_sparse_flash_attention_decode", &custom::NpuSparseFlashAttentionDecodeMeta);
}
