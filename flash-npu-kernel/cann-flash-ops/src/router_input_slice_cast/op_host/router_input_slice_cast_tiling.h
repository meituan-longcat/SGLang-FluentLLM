#ifndef ROUTER_INPUT_SLICE_CAST_TILING_H
#define ROUTER_INPUT_SLICE_CAST_TILING_H

#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(RouterInputSliceCastTilingData)
TILING_DATA_FIELD_DEF(uint32_t, rows);
TILING_DATA_FIELD_DEF(uint32_t, inputCols);
TILING_DATA_FIELD_DEF(uint32_t, sliceStart);
TILING_DATA_FIELD_DEF(uint32_t, sliceLength);
TILING_DATA_FIELD_DEF(uint32_t, usedCores);
TILING_DATA_FIELD_DEF(uint32_t, rowsPerCore);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(RouterInputSliceCast, RouterInputSliceCastTilingData)

struct RouterInputSliceCastCompileInfo {};

} // namespace optiling

#endif
