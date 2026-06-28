
#include "rearrange_accept_index_tiling.h"

namespace optiling {
static ge::graphStatus TilingForRearrangeAcceptIndex(gert::TilingContext* context)
{
  RearrangeAcceptIndexTiling tiling(context);
  return tiling.DoOpTiling();
}
static ge::graphStatus TilingPrepareForRearrangeAcceptIndex(gert::TilingParseContext *context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(RearrangeAcceptIndex)
    .Tiling(TilingForRearrangeAcceptIndex)
    .TilingParse<RearrangeAcceptIndexCompileInfo>(TilingPrepareForRearrangeAcceptIndex);
} // namespace optiling