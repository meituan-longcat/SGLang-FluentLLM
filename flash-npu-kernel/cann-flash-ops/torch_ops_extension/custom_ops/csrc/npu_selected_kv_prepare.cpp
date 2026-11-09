#include <torch/library.h>
#include "ops_common.h"

namespace custom {
namespace {
constexpr int64_t kTopK = 2048;
constexpr int64_t kPage = 128;
constexpr int64_t kCoresPerOwner = 6;

struct PlanLayout {
    int64_t cores;
    int64_t span;
};

PlanLayout CheckPlanInputs(const at::Tensor &topk, const at::Tensor &pages, const at::Tensor &seq,
                           const at::Tensor &prefix, const at::Tensor &offsets, int64_t world, int64_t localRows,
                           int64_t reserved) {
    for (const auto &tensor : {topk, pages, seq, prefix, offsets}) {
        TORCH_CHECK(tensor.scalar_type() == at::kInt && tensor.is_contiguous(), "expected contiguous int32 metadata");
        TORCH_CHECK(tensor.device() == topk.device(), "metadata tensors must share one device");
    }
    TORCH_CHECK(topk.dim() == 2 && topk.size(1) == kTopK && pages.dim() == 2, "invalid query/page shape");
    TORCH_CHECK(seq.dim() == 1 && prefix.dim() == 1 && offsets.dim() == 1, "metadata vectors must be one-dimensional");
    TORCH_CHECK(seq.numel() == topk.size(0), "query sequence count mismatch");
    TORCH_CHECK(world > 0 && world <= 8 && !(world & (world - 1)), "world must be 1/2/4/8");
    TORCH_CHECK(reserved >= 0 && localRows > 0 && localRows <= kCoresPerOwner * 524288,
                "metadata workspace capacity exceeded");
    TORCH_CHECK(localRows % kPage == 0 && reserved < localRows / kPage, "invalid local pool");
    TORCH_CHECK(pages.size(1) > 0 && pages.size(1) <= 16384 && prefix.numel() == pages.size(0) &&
                    offsets.numel() == pages.size(0),
                "invalid metadata");
    const int64_t cores = world * kCoresPerOwner;
    const int64_t span = ((localRows + kCoresPerOwner - 1) / kCoresPerOwner + 255) / 256 * 256;
    TORCH_CHECK(cores * span < INT32_MAX && topk.numel() < INT32_MAX, "metadata workspace capacity exceeded");
    return {cores, span};
}

std::vector<std::vector<int64_t>> WorkspaceShapes(const at::Tensor &topk, const at::Tensor &pages, int64_t world,
                                                  PlanLayout layout) {
    // keys, bitmap, unique-address scratch, counts, addresses, SFAP indices,
    // logical-page map, and per-producer routing counts.
    return {{topk.size(0), kTopK + world * 8 + 8},
            {layout.cores * layout.span / 16},
            {layout.cores * layout.span},
            {layout.cores, 8},
            {topk.numel()},
            {topk.size(0), kTopK},
            {pages.size(0), pages.size(1), 8},
            {layout.cores, 8}};
}

std::vector<at::Tensor> AllocateWorkspace(const at::Tensor &topk, const at::Tensor &pages, int64_t world,
                                          PlanLayout layout) {
    std::vector<at::Tensor> workspace;
    workspace.reserve(8);
    for (const auto &shape : WorkspaceShapes(topk, pages, world, layout)) {
        workspace.push_back(at::empty(shape, topk.options()));
    }
    if (topk.numel() == 0) {
        workspace[3].zero_();
    }
    return workspace;
}

void RunPhase(const at::Tensor &topk, const at::Tensor &pages, const at::Tensor &seq, const at::Tensor &prefix,
              const at::Tensor &offsets, int64_t world, int64_t localRows, int64_t reserved, int64_t phase,
              at::TensorList workspace) {
    if (topk.numel() == 0) {
        return;
    }
    // Keep named lvalues: the ACL dispatch macro retains argument references.
    auto keys = workspace[0], bitmap = workspace[1], temporary = workspace[2];
    auto counts = workspace[3], addresses = workspace[4], indices = workspace[5];
    auto logicalMap = workspace[6], routeCounts = workspace[7];
    EXEC_NPU_CMD_V1(aclnnSelectedKvMetadata, topk, pages, seq, prefix, offsets, world, localRows, reserved, phase, keys,
                    bitmap, temporary, counts, addresses, indices, logicalMap, routeCounts);
}
} // namespace

std::vector<at::Tensor> SelectedKvPlanStart(const at::Tensor &topk, const at::Tensor &pages, const at::Tensor &seq,
                                            const at::Tensor &prefix, const at::Tensor &offsets, int64_t world,
                                            int64_t localRows, int64_t reserved) {
    auto layout = CheckPlanInputs(topk, pages, seq, prefix, offsets, world, localRows, reserved);
    auto workspace = AllocateWorkspace(topk, pages, world, layout);
    // Lookup/mark emits owner-local addresses; count communication can overlap finish.
    for (int64_t phase : {0, 1}) {
        RunPhase(topk, pages, seq, prefix, offsets, world, localRows, reserved, phase, workspace);
    }
    return workspace;
}

std::tuple<at::Tensor, at::Tensor> SelectedKvPlanFinish(const at::Tensor &topk, const at::Tensor &pages,
                                                        const at::Tensor &seq, const at::Tensor &prefix,
                                                        const at::Tensor &offsets, int64_t world, int64_t localRows,
                                                        int64_t reserved, at::TensorList workspace) {
    auto layout = CheckPlanInputs(topk, pages, seq, prefix, offsets, world, localRows, reserved);
    auto shapes = WorkspaceShapes(topk, pages, world, layout);
    TORCH_CHECK(workspace.size() == shapes.size(), "expected build_index_start workspace");
    for (size_t i = 0; i < shapes.size(); ++i) {
        TORCH_CHECK(workspace[i].device() == topk.device() && workspace[i].scalar_type() == at::kInt &&
                        workspace[i].is_contiguous() && workspace[i].sizes().equals(shapes[i]),
                    "invalid build_index workspace at position ", i);
    }
    for (int64_t phase : {2, 3}) {
        RunPhase(topk, pages, seq, prefix, offsets, world, localRows, reserved, phase, workspace);
    }
    return {workspace[4], workspace[5]};
}

at::Tensor SelectedKvMaterialize(const at::Tensor &packed, const at::Tensor &extend) {
    TORCH_CHECK(packed.scalar_type() == at::kChar && packed.dim() == 3 && packed.size(1) == 1 &&
                    (packed.size(2) == 656 || packed.size(2) == 672) && packed.is_contiguous(),
                "packed must be contiguous int8 [N,1,656/672]");
    TORCH_CHECK(extend.scalar_type() == at::kBFloat16 && extend.dim() == 2 && extend.size(1) == 576 &&
                    extend.is_contiguous(),
                "extend must be contiguous BF16 [E,576]");
    TORCH_CHECK(packed.device() == extend.device(), "packed and extend must share one device");
    auto output = at::empty({(packed.size(0) + extend.size(0) + kPage - 1) / kPage, kPage, 576}, extend.options());
    if (output.numel()) {
        EXEC_NPU_CMD_V1(aclnnSelectedKvMaterialize, packed, extend, output);
    }
    return output;
}
} // namespace custom

TORCH_LIBRARY_FRAGMENT(custom, m) {
    m.def("npu_sparse_kv_build_index_start(Tensor topk, Tensor pages, Tensor seq, Tensor prefix, Tensor offsets, int "
          "world, int local_rows, int reserved) -> Tensor[]");
    m.def("npu_sparse_kv_build_index_finish(Tensor topk, Tensor pages, Tensor seq, Tensor prefix, Tensor offsets, int "
          "world, int local_rows, int reserved, Tensor(a!)[] workspace) -> (Tensor(a!), Tensor(a!))");
    m.def("npu_sparse_kv_materialize(Tensor packed, Tensor extend) -> Tensor");
}
TORCH_LIBRARY_IMPL(custom, PrivateUse1, m) {
    m.impl("npu_sparse_kv_build_index_start", &custom::SelectedKvPlanStart);
    m.impl("npu_sparse_kv_build_index_finish", &custom::SelectedKvPlanFinish);
    m.impl("npu_sparse_kv_materialize", &custom::SelectedKvMaterialize);
}
