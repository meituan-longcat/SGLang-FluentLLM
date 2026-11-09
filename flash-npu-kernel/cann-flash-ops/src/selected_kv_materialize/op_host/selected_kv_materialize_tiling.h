#ifndef SELECTED_KV_MATERIALIZE_TILING_H
#define SELECTED_KV_MATERIALIZE_TILING_H

#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(SelectedKvMaterializeTilingData)
TILING_DATA_FIELD_DEF(uint32_t, rows);
TILING_DATA_FIELD_DEF(uint32_t, usedCores);
TILING_DATA_FIELD_DEF(uint32_t, rowsPerCore);
TILING_DATA_FIELD_DEF(uint32_t, packedRowBytes);
TILING_DATA_FIELD_DEF(uint32_t, extendRows);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(SelectedKvMaterialize, SelectedKvMaterializeTilingData)

struct SelectedKvMaterializeCompileInfo {};

} // namespace optiling

#endif
