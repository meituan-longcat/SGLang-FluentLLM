
#include "get_out_cache_loc_tiling.h"

namespace optiling {
static ge::graphStatus TilingForGetOutCacheLoc(gert::TilingContext* context)
{
    GetOutCacheLocTiling tiling(context);
    return tiling.DoOpTiling();
}

static ge::graphStatus TilingPrepareForGetOutCacheLoc(gert::TilingParseContext *context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(GetOutCacheLoc)
    .Tiling(TilingForGetOutCacheLoc)
    .TilingParse<GetOutCacheLocCompileInfo>(TilingPrepareForGetOutCacheLoc);
} // namespace optiling
