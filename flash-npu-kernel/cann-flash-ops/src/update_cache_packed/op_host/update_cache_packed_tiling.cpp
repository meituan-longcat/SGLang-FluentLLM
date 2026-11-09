#include "update_cache_packed_tiling.h"
#include "../../common/packed_kv_offload_tiling.h"
namespace optiling {
using namespace PackedKvOffload;
static ge::graphStatus Tiling(gert::TilingContext* context)
{
    const auto* topk=Shape(context,0); const auto* selection=Shape(context,1);
    PACKED_REQUIRE(Is(topk,{-1,-1,TOPK}) && Is(selection,{-1,PAGE,ROW_BYTES}), "Packed Update expects TopK2048 and page128/row672 selection");
    const int64_t b=topk->GetDim(0),s=topk->GetDim(1);
    PACKED_REQUIRE(b>0 && s>0 && s<=4 && b*s<=128 && selection->GetDim(0)>=b*s*(TOPK/PAGE), "Packed Update dimensions exceed validated bounds");
    PACKED_REQUIRE(Is(Shape(context,2),{b,HOT,ROW_BYTES}) && Is(Shape(context,3),{b,HOT}) && Is(Shape(context,4),{b,HOT}), "Packed Update cache/tag/age shapes disagree");
    PACKED_REQUIRE(Is(Shape(context,5),{b*s,TOPK/PAGE}) && Is(Shape(context,6),{b,s,TOPK}) && Is(Shape(context,7),{b}), "Packed Update table/slot/step shapes disagree");
    const uint32_t cores=Cores(context,b);
    UpdateCachePackedTilingData data;
    data.set_blockNum(cores);data.set_B(b);data.set_S(s);data.set_topk(TOPK);data.set_cache(HOT);
    data.set_kRopeDim(0);data.set_kvDim(ROW_BYTES);data.set_rowsPerBatch(4);
    data.set_selKvBlockSize(PAGE);data.set_selMaxBlockNum(TOPK/PAGE);data.set_reserved(0);
    // Two status/age/source-map regions, eviction scratch, and alias-safe KV staging.
    const size_t workspace=2*3*b*HOT*sizeof(int32_t)+b*3*8192*sizeof(float)+b*HOT*ROW_BYTES+32UL*1024*1024;
    return Finish(context,data,cores,workspace);
}
struct PackedUpdateCompileInfo {};
static ge::graphStatus Prepare(gert::TilingParseContext*) { return ge::GRAPH_SUCCESS; }
IMPL_OP_OPTILING(UpdateCachePacked).Tiling(Tiling).TilingParse<PackedUpdateCompileInfo>(Prepare);
}
