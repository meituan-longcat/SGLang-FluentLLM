// Copyright (c) 2026. Licensed under the Apache License, Version 2.0.
// Packed-row adapter for the existing KV offload implementation.
#pragma once
#include "register/tilingdata_base.h"
namespace optiling {
BEGIN_TILING_DATA_DEF(UpdateCachePackedTilingData)
TILING_DATA_FIELD_DEF(uint32_t, blockNum);
TILING_DATA_FIELD_DEF(uint32_t, B);
TILING_DATA_FIELD_DEF(uint32_t, S);
TILING_DATA_FIELD_DEF(uint32_t, topk);
TILING_DATA_FIELD_DEF(uint32_t, cache);
TILING_DATA_FIELD_DEF(uint32_t, kRopeDim);
TILING_DATA_FIELD_DEF(uint32_t, kvDim);
TILING_DATA_FIELD_DEF(uint32_t, rowsPerBatch);
// 分页：selection(NEW 源) 物理块寻址
TILING_DATA_FIELD_DEF(uint32_t, selKvBlockSize);   // selection 物理块长（=128）
TILING_DATA_FIELD_DEF(uint32_t, selMaxBlockNum);   // selection_kv_block_table 列数
TILING_DATA_FIELD_DEF(uint32_t, reserved);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(UpdateCachePacked, UpdateCachePackedTilingData)
}
