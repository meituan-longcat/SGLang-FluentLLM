// Copyright (c) 2026. Licensed under the Apache License, Version 2.0.
// Packed-row adapter for the existing KV offload implementation.
#include "kernel_operator.h"
#if __has_include("../../update_cache/op_kernel/update_cache_impl.h")
#include "../../update_cache/op_kernel/update_cache_impl.h"
#else
#include "../update_cache/update_cache_impl.h"
#endif
using namespace AscendC;
extern "C" __global__ __aicore__ void update_cache_packed(GM_ADDR topk, GM_ADDR selection_packed, GM_ADDR cache_packed, GM_ADDR status, GM_ADDR age, GM_ADDR selection_block_table, GM_ADDR write_slot, GM_ADDR step, GM_ADDR new_cache_packed, GM_ADDR new_status, GM_ADDR new_age, GM_ADDR workspace, GM_ADDR tiling) {
 if (g_coreType == AIC) return;
 TPipe pipe;
 GET_TILING_DATA(tilingData,tiling);
 if (TILING_KEY_IS(1)) {
 UpdateCacheNs::KernelUpdateCache<int8_t, true> op(&pipe);
 op.Init(topk, selection_packed, selection_packed, cache_packed, cache_packed,
         status, age, selection_block_table, write_slot, step,
         new_cache_packed, new_cache_packed, new_status, new_age,
         workspace, &tilingData);
 op.Process();
 }
}
