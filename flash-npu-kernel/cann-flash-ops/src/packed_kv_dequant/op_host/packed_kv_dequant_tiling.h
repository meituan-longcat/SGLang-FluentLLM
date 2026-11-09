#ifndef PACKED_KV_DEQUANT_TILING_H
#define PACKED_KV_DEQUANT_TILING_H

#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(PackedKvDequantTilingData)
TILING_DATA_FIELD_DEF(uint32_t, rows);
TILING_DATA_FIELD_DEF(uint32_t, usedCores);
TILING_DATA_FIELD_DEF(uint32_t, rowsPerCore);
TILING_DATA_FIELD_DEF(uint32_t, packedRowBytes);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(PackedKvDequant, PackedKvDequantTilingData)

struct PackedKvDequantCompileInfo {};

} // namespace optiling

#endif
