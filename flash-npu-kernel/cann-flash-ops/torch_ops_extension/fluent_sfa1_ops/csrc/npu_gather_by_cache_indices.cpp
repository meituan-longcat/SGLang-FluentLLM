#include "../../custom_ops/csrc/offload/npu_gather_by_cache_indices.h"

// step3, 注册前向实现
TORCH_LIBRARY_IMPL(fluent_sfa1, PrivateUse1, m) {
    m.impl("npu_gather_by_cache_indices", &flash_offload::npu_gather_by_cache_indices_npu);
}

TORCH_LIBRARY_IMPL(fluent_sfa1, Meta, m) {
    m.impl("npu_gather_by_cache_indices", &flash_offload::npu_gather_by_cache_indices_meta);
}
