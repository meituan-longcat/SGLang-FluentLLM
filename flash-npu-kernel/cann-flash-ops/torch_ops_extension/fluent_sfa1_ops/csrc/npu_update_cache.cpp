#include "../../custom_ops/csrc/offload/npu_update_cache.h"

// step3, 注册前向实现
TORCH_LIBRARY_IMPL(fluent_sfa1, PrivateUse1, m) {
    m.impl("npu_update_cache", &flash_offload::npu_update_cache_npu);
}

TORCH_LIBRARY_IMPL(fluent_sfa1, Meta, m) {
    m.impl("npu_update_cache", &flash_offload::npu_update_cache_meta);
}
