#ifndef SPARSE_FLASH_ATTENTION_DECODE_TILING_H
#define SPARSE_FLASH_ATTENTION_DECODE_TILING_H

#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"

namespace optiling {

constexpr uint32_t SPARSE_FLASH_ATTENTION_DECODE_MAX_CORES = 64;

BEGIN_TILING_DATA_DEF(SparseFlashAttentionDecodeTilingData)
TILING_DATA_FIELD_DEF(uint32_t, totalQuery);
TILING_DATA_FIELD_DEF(uint32_t, sparseCount);
TILING_DATA_FIELD_DEF(uint32_t, pageSize);
TILING_DATA_FIELD_DEF(uint32_t, maxBlocks);
TILING_DATA_FIELD_DEF(uint32_t, usedCores);
TILING_DATA_FIELD_DEF(uint32_t, batchSize);
TILING_DATA_FIELD_DEF(float, scale);
TILING_DATA_FIELD_DEF(uint32_t, returnLse);
TILING_DATA_FIELD_DEF(uint32_t, sparseMode);
TILING_DATA_FIELD_DEF(uint32_t, packedRowBytes);
TILING_DATA_FIELD_DEF(uint32_t, queryRankSize);
TILING_DATA_FIELD_DEF(uint32_t, queryLocalHeads);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(SparseFlashAttentionDecode, SparseFlashAttentionDecodeTilingData)

struct SparseFlashAttentionDecodeCompileInfo {};

} // namespace optiling
#endif
