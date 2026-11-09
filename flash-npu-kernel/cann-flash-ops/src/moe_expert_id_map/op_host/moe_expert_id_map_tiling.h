#ifndef MOE_EXPERT_ID_MAP_TILING_H
#define MOE_EXPERT_ID_MAP_TILING_H

#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(MoeExpertIdMapTilingData)
TILING_DATA_FIELD_DEF(uint32_t, idCount);
TILING_DATA_FIELD_DEF(uint32_t, mapSize);
TILING_DATA_FIELD_DEF(uint32_t, alignedIdCount);
TILING_DATA_FIELD_DEF(uint32_t, alignedMapSize);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MoeExpertIdMap, MoeExpertIdMapTilingData)

struct MoeExpertIdMapCompileInfo {};

} // namespace optiling

#endif
