// Copyright (c) 2026. Licensed under the Apache License, Version 2.0.
// Packed-row adapter for the existing KV offload implementation.
#include "kernel_operator.h"
#if __has_include("../../gather_by_cache_indices/op_kernel/gather_by_cache_indices_impl.h")
#include "../../gather_by_cache_indices/op_kernel/gather_by_cache_indices_impl.h"
#else
#include "../gather_by_cache_indices/gather_by_cache_indices_impl.h"
#endif
using namespace AscendC;
extern "C" __global__ __aicore__ void gather_by_cache_indices_packed(GM_ADDR cache_packed, GM_ADDR from_cache_indices, GM_ADDR topk, GM_ADDR full_packed, GM_ADDR full_block_table, GM_ADDR selection_block_table, GM_ADDR sources, GM_ADDR kv_lengths, GM_ADDR query_lengths, GM_ADDR selection_packed, GM_ADDR current_write_slots, GM_ADDR workspace, GM_ADDR tiling) {
 if (g_coreType == AIC) return;
 TPipe pipe;
 GET_TILING_DATA(tilingData,tiling);
 if (TILING_KEY_IS(1)) {
 GatherByCacheIndicesNs::KernelGatherByCacheIndices<int8_t, true> op(&pipe);
 op.Init(cache_packed, cache_packed, from_cache_indices, topk,
         full_packed, full_packed, full_block_table, selection_block_table,
         sources, kv_lengths, query_lengths, selection_packed, selection_packed,
         current_write_slots, &tilingData);
 op.Process();
 }
}
