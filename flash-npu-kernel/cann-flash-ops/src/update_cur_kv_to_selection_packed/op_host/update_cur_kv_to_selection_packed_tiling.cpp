#include "update_cur_kv_to_selection_packed_tiling.h"
#include "../../common/packed_kv_offload_tiling.h"
namespace optiling {
using namespace PackedKvOffload;
static ge::graphStatus Tiling(gert::TilingContext* context)
{
    const auto* current=Shape(context,0);const auto* selection=Shape(context,2);
    PACKED_REQUIRE(Is(current,{-1,-1,ROW_BYTES}) && Is(selection,{-1,PAGE,ROW_BYTES}), "Packed current patch expects B/S/672 and page128/row672 selection");
    const int64_t b=current->GetDim(0),s=current->GetDim(1);
    PACKED_REQUIRE(b>0 && s>0 && s<=4 && b*s<=128 && selection->GetDim(0)>=b*s*(TOPK/PAGE), "Packed current dimensions exceed validated bounds");
    PACKED_REQUIRE(Is(Shape(context,1),{b,s,s}) && Is(Shape(context,3),{b,s,TOPK/PAGE}), "Packed current slots/table shapes disagree");
    const uint32_t total=b*s*s,cores=Cores(context,total);
    UpdateCurKvToSelectionPackedTilingData data;
    data.set_blockNum(cores);data.set_totalUnits(total);data.set_numPerCore(total/cores);data.set_tailUnits(total%cores);
    data.set_batch(b);data.set_seq(s);data.set_kNopeDim(ROW_BYTES);data.set_kRopeDim(0);
    data.set_unitsPerBatch(s);data.set_maxPagePerReq(TOPK/PAGE);data.set_totalPageNum(selection->GetDim(0));
    data.set_nopeRowPitchElems(ROW_BYTES);data.set_ropeRowPitchElems(0);
    data.set_rowAlignedNope(1);data.set_rowAlignedRope(1);data.set_reserved(0);
    return Finish(context,data,cores,16UL*1024*1024);
}
struct PackedCurrentCompileInfo {};
static ge::graphStatus Prepare(gert::TilingParseContext*) { return ge::GRAPH_SUCCESS; }
IMPL_OP_OPTILING(UpdateCurKvToSelectionPacked).Tiling(Tiling).TilingParse<PackedCurrentCompileInfo>(Prepare);
}
