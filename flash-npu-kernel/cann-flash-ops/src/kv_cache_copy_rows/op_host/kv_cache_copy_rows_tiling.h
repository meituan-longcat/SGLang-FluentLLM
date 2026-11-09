#ifndef KV_CACHE_COPY_ROWS_TILING_H
#define KV_CACHE_COPY_ROWS_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(KvCacheCopyRowsTilingData)
TILING_DATA_FIELD_DEF(uint32_t, rowCount);
TILING_DATA_FIELD_DEF(uint32_t, rowWidth);
TILING_DATA_FIELD_DEF(uint32_t, totalRows);
TILING_DATA_FIELD_DEF(uint32_t, snapshotBytes);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(KvCacheCopyRows, KvCacheCopyRowsTilingData)

struct KvCacheCopyRowsCompileInfo {};

} // namespace optiling
#endif
