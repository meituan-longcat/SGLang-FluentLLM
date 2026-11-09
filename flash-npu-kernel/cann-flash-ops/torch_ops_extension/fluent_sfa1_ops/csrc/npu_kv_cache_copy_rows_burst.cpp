#include <torch/library.h>
#include <cstdint>
#include <limits>
#include <memory>
#include <vector>
#include <c10/util/ScopeExit.h>
#include "ops_common.h"
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"

namespace {
struct Interval { uintptr_t begin, end; };
Interval AddressRange(const at::Tensor& tensor)
{
    const auto begin = reinterpret_cast<uintptr_t>(tensor.data_ptr());
    TORCH_CHECK(tensor.nbytes() <= std::numeric_limits<uintptr_t>::max() - begin, "Tensor byte interval overflow");
    return {begin, begin + tensor.nbytes()};
}
bool Overlap(const Interval& a, const Interval& b)
{
    return a.begin < a.end && b.begin < b.end && a.begin < b.end && b.begin < a.end;
}
template <typename Params>
struct BurstApiLease {
    Params params;
    ReleaseHugeMem releaseHuge;
    bool released = false;
    BurstApiLease(Params value, ReleaseHugeMem release) : params(std::move(value)), releaseHuge(release) {}
    void Release()
    {
        if (released) { return; }
        released = true;
        ReleaseConvertTypes(params);
        if (releaseHuge) { releaseHuge(nullptr, false); }
    }
    ~BurstApiLease() { Release(); }
};
void RecordCachingTensor(const at::Tensor& tensor, c10_npu::NPUStream stream)
{
    if (!tensor.defined() || tensor.numel() == 0) { return; }
    auto* allocator = c10_npu::NPUCachingAllocator::get();
    const auto& ptr = tensor.storage().data_ptr();
    // External/swapped pool owners retain their existing completion lifetime.
    // Only hand native caching allocations to its stream recorder.
    const auto deleter = allocator == nullptr ? nullptr : allocator->raw_deleter();
    if (deleter != nullptr && ptr.get_deleter() == deleter) {
        c10_npu::NPUCachingAllocator::recordStream(ptr, stream);
    }
}
void LaunchBurst(at::TensorList caches, const at::Tensor& src, const at::Tensor& dst)
{
    static const auto workspaceApi = GetOpApiFuncAddr("aclnnKvCacheCopyRowsBurstGetWorkspaceSize");
    static const auto runApi = GetOpApiFuncAddr("aclnnKvCacheCopyRowsBurst");
    TORCH_CHECK(workspaceApi != nullptr && runApi != nullptr, "Missing paired layer-burst ACLNN APIs");
    static const auto initHuge = reinterpret_cast<InitHugeMemThreadLocal>(GetOpApiFuncAddr("InitHugeMemThreadLocal"));
    static const auto uninitHuge = reinterpret_cast<UnInitHugeMemThreadLocal>(GetOpApiFuncAddr("UnInitHugeMemThreadLocal"));
    static const auto releaseHuge = reinterpret_cast<ReleaseHugeMem>(GetOpApiFuncAddr("ReleaseHugeMem"));
    static const auto initCache = reinterpret_cast<InitPTACacheThreadLocal>(GetOpApiFuncAddr("InitPTACacheThreadLocal"));
    static const auto setHash = reinterpret_cast<SetPTAHashKey>(GetOpApiFuncAddr("SetPTAHashKey"));
    const auto stream = c10_npu::getCurrentNPUStream();
    const auto aclStream = stream.stream(false);
    if (c10_npu::check_enqueue_need_use(aclStream)) { GetUseStreamResFuncCoreNum()(aclStream); }
    if (initCache && setHash) { initCache();setHash(0); }
    if (initHuge) { initHuge(nullptr, false); }
    auto threadCleanup = c10::make_scope_exit([&] {
        if (uninitHuge) { uninitHuge(nullptr, false); }
        UnInitCacheThreadLocal();
    });
    uint64_t workspaceBytes = 0;
    aclOpExecutor* executor = nullptr;
    uint64_t* workspaceAddress = &workspaceBytes;
    aclOpExecutor** executorAddress = &executor;
    auto converted = ConvertTypes(caches, src, dst, workspaceAddress, executorAddress);
    auto lease = std::make_shared<BurstApiLease<decltype(converted)>>(std::move(converted), releaseHuge);
    static auto getWorkspace = ConvertToOpApiFunc(lease->params, workspaceApi);
    const auto status = call(getWorkspace, lease->params);
    TORCH_CHECK(status == 0, "aclnnKvCacheCopyRowsBurstGetWorkspaceSize failed: ", status);
    TORCH_CHECK(workspaceBytes <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
                "Layer-burst workspace size exceeds Tensor dimensions");
    at::Tensor workspaceTensor;
    if (workspaceBytes != 0) {
        workspaceTensor = at::empty({static_cast<int64_t>(workspaceBytes)},
            at::TensorOptions(torch_npu::utils::get_npu_device_type()).dtype(at::kByte));
    }
    std::vector<at::Tensor> keptCaches(caches.begin(), caches.end());
    auto handler = [lease, workspaceTensor, keptCaches, src, dst, stream, aclStream, workspaceBytes, executor]() -> int {
        auto release = c10::make_scope_exit([&] { lease->Release(); });
        if (c10_npu::check_dequeue_need_use(aclStream)) { GetUseStreamResFuncCoreNum()(aclStream); }
        RecordCachingTensor(workspaceTensor, stream);
        for (const auto& cache : keptCaches) { RecordCachingTensor(cache, stream); }
        RecordCachingTensor(src, stream);RecordCachingTensor(dst, stream);
        void* workspace = workspaceTensor.defined() ? workspaceTensor.data_ptr() : nullptr;
        const auto run = reinterpret_cast<OpApiFunc>(runApi);
        const auto result = run(workspace, workspaceBytes, executor, aclStream);
        TORCH_CHECK(result == 0, "aclnnKvCacheCopyRowsBurst failed: ", result);
        return result;
    };
    at_npu::native::OpCommand::RunOpApiV2("aclnnKvCacheCopyRowsBurst", handler);
}
void Check(at::TensorList caches, const at::Tensor& src, const at::Tensor& dst, bool addresses)
{
    TORCH_CHECK(!caches.empty() && caches.size() <= 38, "Burst requires1..38 caches");
    const auto& first = caches[0];
    const bool bf16 = first.scalar_type() == at::kBFloat16;
    TORCH_CHECK(first.dim() == 2 && first.size(0) > 0 && first.size(0) <= INT32_MAX &&
                    ((first.scalar_type() == at::kChar && first.size(1) == 672) ||
                     (bf16 && (first.size(1) == 512 || first.size(1) == 64))),
                "Burst caches require INT8[R,672] or BF16[R,512/64],1<=R<=INT32_MAX");
    TORCH_CHECK(src.dim() == 1 && dst.dim() == 1 && src.numel() == dst.numel() && src.numel() <= (bf16 ? 128 : 256),
                "Burst requires shared1-D source/destination indices,N<=128 for BF16 or256 for INT8");
    TORCH_CHECK((src.scalar_type() == at::kInt || src.scalar_type() == at::kLong) && src.scalar_type() == dst.scalar_type(),
                "Burst indices must have the same int32/int64 dtype");
    TORCH_CHECK(src.device() == first.device() && dst.device() == first.device() && src.is_contiguous() && dst.is_contiguous(),
                "Burst indices must be contiguous on the caches' device");
    std::vector<Interval> regions;
    for (const auto& cache : caches) {
        TORCH_CHECK(cache.device() == first.device() && cache.scalar_type() == first.scalar_type() && cache.is_contiguous() &&
                    cache.sizes() == first.sizes(), "Burst caches must have equal contiguous dtype/geometry/device");
        if (addresses) {
            const auto region = AddressRange(cache);
            TORCH_CHECK(region.begin % 32 == 0, "Burst cache address must be32-byte aligned");
            for (const auto& other : regions) { TORCH_CHECK(!Overlap(region, other), "Burst layers must not overlap"); }
            regions.push_back(region);
        }
    }
    if (addresses) {
        for (const auto* indices : {&src, &dst}) {
            const auto region = AddressRange(*indices);
            TORCH_CHECK(indices->numel() == 0 || region.begin % 32 == 0, "Burst index addresses must be32-byte aligned");
            for (const auto& cache : regions) { TORCH_CHECK(!Overlap(region, cache), "Burst indices must not overlap cache payload"); }
        }
    }
}
void CopyBurst(at::TensorList caches, const at::Tensor& src, const at::Tensor& dst)
{
    Check(caches, src, dst, true);
    TORCH_CHECK(caches[0].device().type() == c10::DeviceType::PrivateUse1, "Native burst requires NPU tensors");
    if (src.numel() == 0) { return; }
    const c10::OptionalDeviceGuard guard(caches[0].device());
    LaunchBurst(caches, src, dst);
}
void CopyBurstMeta(at::TensorList caches, const at::Tensor& src, const at::Tensor& dst)
{
    Check(caches, src, dst, false);
}
} // namespace

TORCH_LIBRARY_FRAGMENT(fluent_sfa1, m) {
    m.def("npu_kv_cache_copy_rows_burst_(Tensor(a!)[] caches, Tensor src_indices, Tensor dst_indices) -> ()");
}
TORCH_LIBRARY_IMPL(fluent_sfa1, PrivateUse1, m) { m.impl("npu_kv_cache_copy_rows_burst_", CopyBurst); }
TORCH_LIBRARY_IMPL(fluent_sfa1, Meta, m) { m.impl("npu_kv_cache_copy_rows_burst_", CopyBurstMeta); }
