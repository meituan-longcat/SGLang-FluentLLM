#include "assign_req_to_token_pool_tiling.h"

namespace optiling {
static ge::graphStatus TilingForAssignReqToTokenPool(gert::TilingContext* context)
{
  AssignReqToTokenPoolTiling tiling(context);
  return tiling.DoOpTiling();
}

static ge::graphStatus TilingPrepareForAssignReqToTokenPool(gert::TilingParseContext *context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(AssignReqToTokenPool)
    .Tiling(TilingForAssignReqToTokenPool)
    .TilingParse<AssignReqToTokenPoolCompileInfo>(TilingPrepareForAssignReqToTokenPool);
} // namespace optiling