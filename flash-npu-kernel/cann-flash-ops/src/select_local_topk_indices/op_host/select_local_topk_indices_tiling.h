#ifndef SELECT_LOCAL_TOPK_INDICES_TILING_H
#define SELECT_LOCAL_TOPK_INDICES_TILING_H

#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(SelectLocalTopkIndicesTilingData)
TILING_DATA_FIELD_DEF(uint32_t, totalTokens);
TILING_DATA_FIELD_DEF(uint32_t, localCount);
TILING_DATA_FIELD_DEF(uint32_t, topkCount);
TILING_DATA_FIELD_DEF(uint32_t, usedCores);
TILING_DATA_FIELD_DEF(int32_t, rankStart);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(SelectLocalTopkIndices,
                           SelectLocalTopkIndicesTilingData)
struct SelectLocalTopkIndicesCompileInfo {};
} // namespace optiling
#endif
