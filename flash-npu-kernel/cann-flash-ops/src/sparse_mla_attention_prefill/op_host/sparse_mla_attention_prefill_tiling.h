#ifndef SPARSE_MLA_ATTENTION_PREFILL_TILING_H
#define SPARSE_MLA_ATTENTION_PREFILL_TILING_H

#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"

namespace optiling {

constexpr uint32_t SPARSE_MLA_ATTENTION_PREFILL_MAX_CORES = 64;

BEGIN_TILING_DATA_DEF(SparseMlaAttentionPrefillTilingData)
TILING_DATA_FIELD_DEF(uint32_t, totalQuery);
TILING_DATA_FIELD_DEF(uint32_t, sparseCount);
TILING_DATA_FIELD_DEF(uint32_t, pageSize);
TILING_DATA_FIELD_DEF(uint32_t, usedCores);
TILING_DATA_FIELD_DEF(uint32_t, maxBlocks);
TILING_DATA_FIELD_DEF(float, scale);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(SparseMlaAttentionPrefill,
                           SparseMlaAttentionPrefillTilingData)

struct SparseMlaAttentionPrefillCompileInfo {};

} // namespace optiling
#endif
