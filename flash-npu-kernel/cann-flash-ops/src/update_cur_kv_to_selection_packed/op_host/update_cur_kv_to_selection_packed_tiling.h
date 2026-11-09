// Copyright (c) 2026. Licensed under the Apache License, Version 2.0.
// Packed-row adapter for the existing KV offload implementation.
#pragma once
#include "register/tilingdata_base.h"
namespace optiling {
BEGIN_TILING_DATA_DEF(UpdateCurKvToSelectionPackedTilingData)
TILING_DATA_FIELD_DEF(uint32_t, blockNum);        // 实际使用核数（= usedCore）
TILING_DATA_FIELD_DEF(uint32_t, totalUnits);      // B*S*S
TILING_DATA_FIELD_DEF(uint32_t, numPerCore);      // 每核基础单元数
TILING_DATA_FIELD_DEF(uint32_t, tailUnits);       // 余数核数（前 tailUnits 核多处理 1 个单元）
TILING_DATA_FIELD_DEF(uint32_t, batch);           // B
TILING_DATA_FIELD_DEF(uint32_t, seq);             // S（slots 的 s/j 两维）
TILING_DATA_FIELD_DEF(uint32_t, kNopeDim);        // cur_k_nope / selection_k_nope 末维
TILING_DATA_FIELD_DEF(uint32_t, kRopeDim);        // cur_k_rope / selection_k_rope 末维
TILING_DATA_FIELD_DEF(uint32_t, unitsPerBatch);   // P：单次 UB 批处理的单元数（P 自适应）
TILING_DATA_FIELD_DEF(uint32_t, maxPagePerReq);   // MAX_PAGE_PER_REQ：selection_kv_block_table 列数
TILING_DATA_FIELD_DEF(uint32_t, totalPageNum);    // TOTAL_PAGE_NUM：selection 物理页总数
TILING_DATA_FIELD_DEF(uint32_t, nopeRowPitchElems); // UB 侧 nope 行 pitch（元素数，= ceil(kNopeDim/16)*16）
TILING_DATA_FIELD_DEF(uint32_t, ropeRowPitchElems); // UB 侧 rope 行 pitch（元素数，= ceil(kRopeDim/16)*16）
TILING_DATA_FIELD_DEF(uint32_t, rowAlignedNope);  // kNopeDim*2 % 32 == 0 → 可连续段合并（nope）
TILING_DATA_FIELD_DEF(uint32_t, rowAlignedRope);  // kRopeDim*2 % 32 == 0 → 可连续段合并（rope）
TILING_DATA_FIELD_DEF(uint32_t, reserved);        // 0=数据双缓冲；1=单缓冲退化（P_ub<1 时）
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(UpdateCurKvToSelectionPacked, UpdateCurKvToSelectionPackedTilingData)
}
