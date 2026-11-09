#include <tuple>
#include <torch/library.h>

#include "ops_common.h"

namespace custom {
using Outputs = std::tuple<at::Tensor, at::Tensor, at::Tensor>;

namespace {
constexpr int64_t PACKED_PAYLOAD_BYTES = 656;
constexpr int64_t PACKED_ALIGNED_ROW_BYTES = 672;

void CheckPackedRowBytes(int64_t packedRowBytes)
{
    TORCH_CHECK(
        packedRowBytes == PACKED_PAYLOAD_BYTES ||
            packedRowBytes == PACKED_ALIGNED_ROW_BYTES,
        "packed_row_bytes must be 656 or 672, got ", packedRowBytes, ".");
}

void CheckInputs(const at::Tensor &latentCache, const at::Tensor &gamma,
                 const at::Tensor &cos, const at::Tensor &sin,
                 const at::Tensor &ownerRows)
{
    TORCH_CHECK(latentCache.scalar_type() == at::kBFloat16 &&
        latentCache.dim() == 2 && latentCache.size(1) == 576 &&
        latentCache.size(0) > 0 && latentCache.is_contiguous(),
        "latent_cache must be contiguous BF16 [T, 576].");
    TORCH_CHECK(gamma.scalar_type() == at::kBFloat16 && gamma.dim() == 1 &&
        gamma.size(0) == 512 && gamma.is_contiguous(),
        "gamma must be contiguous BF16 [512].");
    TORCH_CHECK(cos.scalar_type() == at::kBFloat16 && sin.scalar_type() == at::kBFloat16 &&
        cos.dim() == 2 && sin.dim() == 2 && cos.size(0) == latentCache.size(0) &&
        sin.sizes() == cos.sizes() && cos.size(1) == 64 &&
        cos.is_contiguous() && sin.is_contiguous(),
        "cos and sin must be contiguous BF16 [T, 64].");
    TORCH_CHECK(ownerRows.scalar_type() == at::kLong && ownerRows.dim() == 1 &&
        ownerRows.is_contiguous(), "owner_rows must be contiguous int64 [O].");
}

Outputs MakeOutputs(const at::Tensor &latentCache, const at::Tensor &ownerRows,
                    int64_t packedRowBytes)
{
    auto bf16 = latentCache.options().dtype(at::kBFloat16);
    auto int8 = latentCache.options().dtype(at::kChar);
    return {
        at::empty({latentCache.size(0), 1, 512}, bf16),
        at::empty({latentCache.size(0), 1, 64}, bf16),
        at::empty({ownerRows.size(0), 1, packedRowBytes}, int8)};
}
} // namespace

Outputs NpuPackedKvPrefillPrepare(
    const at::Tensor &latentCache, const at::Tensor &gamma,
    const at::Tensor &cos, const at::Tensor &sin,
    const at::Tensor &ownerRows, double epsilon, int64_t packedRowBytes)
{
    CheckInputs(latentCache, gamma, cos, sin, ownerRows);
    CheckPackedRowBytes(packedRowBytes);
    auto [key, keyRope, ownerPacked] =
        MakeOutputs(latentCache, ownerRows, packedRowBytes);
    EXEC_NPU_CMD_V1(aclnnPackedKvPrefillPrepare, latentCache, gamma, cos, sin,
        ownerRows, epsilon, packedRowBytes, key, keyRope, ownerPacked);
    return {key, keyRope, ownerPacked};
}

Outputs NpuPackedKvPrefillPrepareMeta(
    const at::Tensor &latentCache, const at::Tensor &gamma,
    const at::Tensor &cos, const at::Tensor &sin,
    const at::Tensor &ownerRows, double epsilon, int64_t packedRowBytes)
{
    (void)epsilon;
    CheckInputs(latentCache, gamma, cos, sin, ownerRows);
    CheckPackedRowBytes(packedRowBytes);
    return MakeOutputs(latentCache, ownerRows, packedRowBytes);
}
} // namespace custom

TORCH_LIBRARY_IMPL(custom, PrivateUse1, m) {
    m.impl("npu_packed_kv_prefill_prepare", &custom::NpuPackedKvPrefillPrepare);
}

TORCH_LIBRARY_IMPL(custom, Meta, m) {
    m.impl("npu_packed_kv_prefill_prepare", &custom::NpuPackedKvPrefillPrepareMeta);
}
