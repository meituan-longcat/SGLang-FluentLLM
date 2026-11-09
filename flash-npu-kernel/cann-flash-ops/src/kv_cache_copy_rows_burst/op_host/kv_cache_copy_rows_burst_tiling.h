#ifndef KV_CACHE_COPY_ROWS_BURST_TILING_H
#define KV_CACHE_COPY_ROWS_BURST_TILING_H
#include "register/tilingdata_base.h"
namespace optiling {
BEGIN_TILING_DATA_DEF(KvCacheCopyRowsBurstTilingData)
TILING_DATA_FIELD_DEF(uint32_t, layerCount);
TILING_DATA_FIELD_DEF(uint32_t, totalRows);
TILING_DATA_FIELD_DEF(uint32_t, rowCount);
TILING_DATA_FIELD_DEF(uint32_t, rowBytes);
TILING_DATA_FIELD_DEF(uint32_t, indexBufferBytes);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(KvCacheCopyRowsBurst, KvCacheCopyRowsBurstTilingData)
struct KvCacheCopyRowsBurstCompileInfo {};
} // namespace optiling
#endif
