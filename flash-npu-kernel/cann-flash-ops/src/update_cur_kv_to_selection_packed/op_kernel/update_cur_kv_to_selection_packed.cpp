// Copyright (c) 2026. Licensed under the Apache License, Version 2.0.
// Packed-row adapter for the existing KV offload implementation.
#include "kernel_operator.h"
#if __has_include("../../update_cur_kv_to_selection/op_kernel/update_cur_kv_to_selection_impl.h")
#include "../../update_cur_kv_to_selection/op_kernel/update_cur_kv_to_selection_impl.h"
#else
#include "../update_cur_kv_to_selection/update_cur_kv_to_selection_impl.h"
#endif
using namespace AscendC;
extern "C" __global__ __aicore__ void update_cur_kv_to_selection_packed(GM_ADDR current_packed, GM_ADDR current_write_slots, GM_ADDR selection_packed, GM_ADDR selection_block_table, GM_ADDR selection_packed_out, GM_ADDR kernel_done, GM_ADDR workspace, GM_ADDR tiling) {
 if (g_coreType == AIC) return;
 // GE expands the same-name in-place output as an extra device argument.
 (void)selection_packed_out;
 TPipe pipe;
 GET_TILING_DATA(tilingData,tiling);
 if (TILING_KEY_IS(1)) {
 UpdateCurKvToSelectionNs::KernelUpdateCurKvToSelection<int8_t, true> op(&pipe);
 op.Init(current_packed, current_packed, current_write_slots, selection_packed,
         selection_packed, selection_block_table, kernel_done, &tilingData);
 op.Process();
 }
}
