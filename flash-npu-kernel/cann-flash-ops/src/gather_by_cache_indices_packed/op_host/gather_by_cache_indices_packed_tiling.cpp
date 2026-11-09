#include "gather_by_cache_indices_packed_tiling.h"
#include "../../common/packed_kv_offload_tiling.h"
namespace optiling {
using namespace PackedKvOffload;
static ge::graphStatus Tiling(gert::TilingContext* context)
{
    const auto* cache = Shape(context,0); const auto* topk = Shape(context,2);
    const auto* full = Shape(context,3); const auto* table = Shape(context,4);
    PACKED_REQUIRE(Is(topk,{-1,-1,TOPK}) && Is(cache,{-1,HOT,ROW_BYTES}) && Is(full,{-1,PAGE,ROW_BYTES}), "Packed Gather expects B/S/K2048, C4096 and page128/row672");
    const int64_t b=topk->GetDim(0), s=topk->GetDim(1);
    PACKED_REQUIRE(b>0 && s>0 && s<=4 && b*s<=128 && cache->GetDim(0)==b && full->GetDim(0)>0, "Packed Gather batch/query dimensions exceed validated bounds");
    PACKED_REQUIRE(Is(Shape(context,1),{b,s,TOPK}) && Is(Shape(context,6),{b,s,TOPK}) && Is(Shape(context,7),{b}) && Is(Shape(context,8),{b}), "Packed Gather index/length shapes disagree");
    PACKED_REQUIRE(Is(table,{b,-1}) && table->GetDim(1)>0 && table->GetDim(1)<=16384 && Is(Shape(context,5),{b*s,TOPK/PAGE}), "Packed Gather page tables disagree");
    auto* attrs=context->GetAttrs();
    PACKED_REQUIRE(attrs, "Missing packed Gather attributes");
    auto* block=attrs->GetAttrPointer<int64_t>(0); auto* pages=attrs->GetAttrPointer<int64_t>(1);
    auto* page=attrs->GetAttrPointer<int64_t>(2); auto* mode=attrs->GetAttrPointer<int64_t>(3);
    PACKED_REQUIRE(block && *block==1 && pages && *pages>=b*s*(TOPK/PAGE) && page && *page==PAGE && mode && *mode==2, "Packed Gather supports token selectors, page128 and mode2 only");
    const uint32_t cores=Cores(context,b);
    GatherByCacheIndicesPackedTilingData data;
    data.set_blockNum(cores); data.set_totalUnits(b*s*TOPK);
    data.set_numPerCore(((b*s+cores-1)/cores)*TOPK); data.set_tailUnits(0);
    data.set_kRopeDim(0); data.set_kvDim(ROW_BYTES); data.set_unitsPerBatch(32);
    data.set_cacheRows(HOT); data.set_unitsPerRealBatch(s*TOPK);
    data.set_seq(s); data.set_topk(TOPK); data.set_selTopKBlockSize(1);
    data.set_fullKvBlockSize(PAGE); data.set_fullMaxBlockNum(table->GetDim(1));
    data.set_fullKvBlockNum(full->GetDim(0)); data.set_selKvBlockSize(PAGE);
    data.set_selMaxBlockNum(TOPK/PAGE); data.set_enableSlotsContiguous(2); data.set_reserved(0);
    return Finish(context,data,cores,16UL*1024*1024);
}
struct PackedGatherCompileInfo {};
static ge::graphStatus Prepare(gert::TilingParseContext*) { return ge::GRAPH_SUCCESS; }
IMPL_OP_OPTILING(GatherByCacheIndicesPacked).Tiling(Tiling).TilingParse<PackedGatherCompileInfo>(Prepare);
}
