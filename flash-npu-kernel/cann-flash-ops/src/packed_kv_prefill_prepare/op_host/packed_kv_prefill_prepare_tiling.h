#pragma once

#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(PackedKvPrefillPrepareTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, rows);
    TILING_DATA_FIELD_DEF(uint32_t, ownerRows);
    TILING_DATA_FIELD_DEF(uint32_t, rowsPerCore);
    TILING_DATA_FIELD_DEF(uint32_t, ownersPerCore);
    TILING_DATA_FIELD_DEF(uint32_t, packedRowBytes);
    TILING_DATA_FIELD_DEF(float, epsilon);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(PackedKvPrefillPrepare, PackedKvPrefillPrepareTilingData)
struct PackedKvPrefillPrepareCompileInfo {};
} // namespace optiling
