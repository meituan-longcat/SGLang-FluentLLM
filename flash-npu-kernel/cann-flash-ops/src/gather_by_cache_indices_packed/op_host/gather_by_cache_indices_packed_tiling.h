// Copyright (c) 2026. Licensed under the Apache License, Version 2.0.
// Packed-row adapter for the existing KV offload implementation.
#pragma once
#include "register/tilingdata_base.h"
namespace optiling {
BEGIN_TILING_DATA_DEF(GatherByCacheIndicesPackedTilingData)
TILING_DATA_FIELD_DEF(uint32_t, blockNum);
TILING_DATA_FIELD_DEF(uint32_t, totalUnits);
TILING_DATA_FIELD_DEF(uint32_t, numPerCore);
TILING_DATA_FIELD_DEF(uint32_t, tailUnits);
TILING_DATA_FIELD_DEF(uint32_t, kRopeDim);
TILING_DATA_FIELD_DEF(uint32_t, kvDim);
TILING_DATA_FIELD_DEF(uint32_t, unitsPerBatch);
TILING_DATA_FIELD_DEF(uint32_t, cacheRows);
TILING_DATA_FIELD_DEF(uint32_t, unitsPerRealBatch);
// 分页（两级页表）新增
TILING_DATA_FIELD_DEF(uint32_t, seq);              // S
TILING_DATA_FIELD_DEF(uint32_t, topk);             // TOPK
TILING_DATA_FIELD_DEF(uint32_t, selTopKBlockSize); // 每 topk id token 数（本用例=1）
TILING_DATA_FIELD_DEF(uint32_t, fullKvBlockSize);  // full 物理块长（=128）
TILING_DATA_FIELD_DEF(uint32_t, fullMaxBlockNum);  // full_kv_block_table 列数
TILING_DATA_FIELD_DEF(uint32_t, fullKvBlockNum);   // full 物理块数（越界断言）
TILING_DATA_FIELD_DEF(uint32_t, selKvBlockSize);   // selection 物理块长（=128）
TILING_DATA_FIELD_DEF(uint32_t, selMaxBlockNum);   // selection_kv_block_table 列数
TILING_DATA_FIELD_DEF(uint32_t, enableSlotsContiguous); // 1=page id 连续，走线性偏移优化
TILING_DATA_FIELD_DEF(uint32_t, reserved);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(GatherByCacheIndicesPacked, GatherByCacheIndicesPackedTilingData)
}
