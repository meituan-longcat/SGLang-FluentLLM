#include "../../custom_ops/csrc/offload/npu_update_cur_kv_to_selection.h"

// step3, 注册前向实现
TORCH_LIBRARY_IMPL(fluent_sfa1, PrivateUse1, m) {
    m.impl("npu_update_cur_kv_to_selection", &flash_offload::npu_update_cur_kv_to_selection_npu);
    m.impl("npu_update_cur_kv_to_selection_functional", &flash_offload::npu_update_cur_kv_to_selection_functional_npu);
}

TORCH_LIBRARY_IMPL(fluent_sfa1, Meta, m) {
    m.impl("npu_update_cur_kv_to_selection", &flash_offload::npu_update_cur_kv_to_selection_meta);
    m.impl("npu_update_cur_kv_to_selection_functional", &flash_offload::npu_update_cur_kv_to_selection_functional_meta);
}
