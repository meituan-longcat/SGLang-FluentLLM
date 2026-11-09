#ifndef KVP_ATTENTION_MERGE_TILING_H
#define KVP_ATTENTION_MERGE_TILING_H

#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(KvpAttentionMergeTilingData)
TILING_DATA_FIELD_DEF(uint32_t, rankCount);
TILING_DATA_FIELD_DEF(uint32_t, rowCount);
TILING_DATA_FIELD_DEF(uint32_t, hiddenSize);
TILING_DATA_FIELD_DEF(uint32_t, rowsPerCore);
TILING_DATA_FIELD_DEF(uint32_t, alignedRowsPerCore);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(KvpAttentionMerge, KvpAttentionMergeTilingData)

struct KvpAttentionMergeCompileInfo {};

} // namespace optiling

#endif
