#include <tuple>
#include <ATen/MemoryOverlap.h>
#include <torch/library.h>

#include "ops_common.h"

namespace custom {
using PackedKvOutputs = std::tuple<at::Tensor, at::Tensor>;

namespace {
constexpr int64_t kPackedPayloadBytes = 656;
constexpr int64_t kPackedAlignedRowBytes = 672;

void CheckPackedKv(const at::Tensor &packedKv)
{
    TORCH_CHECK(packedKv.scalar_type() == at::kChar,
        "packed_kv dtype must be int8, but got ", packedKv.scalar_type(), ".");
    TORCH_CHECK(packedKv.dim() == 3 && packedKv.size(1) == 1 &&
        (packedKv.size(2) == kPackedPayloadBytes ||
         packedKv.size(2) == kPackedAlignedRowBytes) &&
        packedKv.size(0) > 0,
        "packed_kv must have shape [T, 1, 656] or [T, 1, 672], but got ",
        packedKv.sizes(), ".");
    TORCH_CHECK(packedKv.is_contiguous(), "packed_kv must be contiguous.");
}

PackedKvOutputs MakeOutputs(const at::Tensor &packedKv)
{
    auto options = packedKv.options().dtype(at::kBFloat16);
    return {
        at::empty({packedKv.size(0), 1, 512}, options),
        at::empty({packedKv.size(0), 1, 64}, options)};
}

void CheckOutputs(const at::Tensor &packedKv, const at::Tensor &key,
    const at::Tensor &keyRope)
{
    CheckPackedKv(packedKv);
    TORCH_CHECK(key.device() == packedKv.device() &&
        keyRope.device() == packedKv.device(),
        "output devices must match packed_kv.");
    TORCH_CHECK(key.scalar_type() == at::kBFloat16 &&
        keyRope.scalar_type() == at::kBFloat16, "outputs must be BF16.");
    TORCH_CHECK(key.dim() == 3 && key.size(0) == packedKv.size(0) &&
        key.size(1) == 1 && key.size(2) == 512, "key shape mismatch.");
    TORCH_CHECK(keyRope.dim() == 3 &&
        keyRope.size(0) == packedKv.size(0) && keyRope.size(1) == 1 &&
        keyRope.size(2) == 64, "rope shape mismatch.");
    TORCH_CHECK(key.is_contiguous() && keyRope.is_contiguous(),
        "outputs must be contiguous.");
}
} // namespace

PackedKvOutputs NpuPackedKvDequant(const at::Tensor &packedKv)
{
    CheckPackedKv(packedKv);
    auto [key, keyRope] = MakeOutputs(packedKv);
    EXEC_NPU_CMD_V1(aclnnPackedKvDequant, packedKv, key, keyRope);
    return {key, keyRope};
}

PackedKvOutputs NpuPackedKvDequantMeta(const at::Tensor &packedKv)
{
    CheckPackedKv(packedKv);
    return MakeOutputs(packedKv);
}

// Let callers allocate/reuse outputs on their owning stream while executing
// the unchanged kernel on a prefetch stream. The caller orders cross-stream use.
void NpuPackedKvDequantOut(const at::Tensor &packedKv, at::Tensor &key,
    at::Tensor &keyRope)
{
    CheckOutputs(packedKv, key, keyRope);
    at::assert_no_overlap(packedKv, key);
    at::assert_no_overlap(packedKv, keyRope);
    at::assert_no_overlap(key, keyRope);
    EXEC_NPU_CMD_V1(aclnnPackedKvDequant, packedKv, key, keyRope);
}

void NpuPackedKvDequantOutMeta(const at::Tensor &packedKv, at::Tensor &key,
    at::Tensor &keyRope)
{
    CheckOutputs(packedKv, key, keyRope);
}
} // namespace custom

TORCH_LIBRARY_IMPL(custom, PrivateUse1, m) {
    m.impl("npu_packed_kv_dequant", &custom::NpuPackedKvDequant);
    m.impl("npu_packed_kv_dequant_out", &custom::NpuPackedKvDequantOut);
}

TORCH_LIBRARY_IMPL(custom, Meta, m) {
    m.impl("npu_packed_kv_dequant", &custom::NpuPackedKvDequantMeta);
    m.impl("npu_packed_kv_dequant_out", &custom::NpuPackedKvDequantOutMeta);
}
