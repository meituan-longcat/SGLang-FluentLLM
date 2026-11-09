#include <register/op_impl_registry.h>

#include <cstddef>
#include <cstring>

namespace ops {
ge::graphStatus InferShapeSparseFlashAttentionDecode(gert::InferShapeContext *context)
{
    const auto *query = context->GetInputShape(0);
    const auto *key = context->GetInputShape(1);
    if (query == nullptr || key == nullptr || key->GetDimNum() != 4 ||
        key->GetDim(2) != 1) {
        return ge::GRAPH_FAILED;
    }
    const char *layout = context->GetAttrs()->GetAttrPointer<char>(2);
    const bool packed = layout != nullptr &&
        std::strcmp(layout, "RANK_MAJOR_TND_PACKED_KVP_COMM") == 0;
    const bool tnd = layout != nullptr &&
        std::strcmp(layout, "TND") == 0;
    const bool tndComm = layout != nullptr &&
        std::strcmp(layout, "TND_KVP_COMM") == 0;
    if (!packed && !tnd && !tndComm) {
        return ge::GRAPH_FAILED;
    }
    int64_t totalQuery = 0;
    int64_t queryHeads = 64;
    if (packed) {
        if ((query->GetDimNum() != 4 && query->GetDimNum() != 5) ||
            query->GetDim(query->GetDimNum() - 1) != 576) {
            return ge::GRAPH_FAILED;
        }
        for (size_t dim = 2; dim + 2 < query->GetDimNum(); ++dim) {
            if (query->GetDim(dim) != 1) return ge::GRAPH_FAILED;
        }
        totalQuery = query->GetDim(1);
        queryHeads = query->GetDim(0) *
            query->GetDim(query->GetDimNum() - 2);
        if (queryHeads != 64) return ge::GRAPH_FAILED;
        *context->GetOutputShape(0) =
            gert::Shape({queryHeads, totalQuery, 512});
    } else {
        if (query->GetDimNum() != 3 || query->GetDim(1) != 64 ||
            query->GetDim(2) != 512) {
            return ge::GRAPH_FAILED;
        }
        totalQuery = query->GetDim(0);
        *context->GetOutputShape(0) = tndComm ?
            gert::Shape({queryHeads, totalQuery, 512}) : *query;
    }
    auto *maxShape = context->GetOutputShape(1);
    auto *sumShape = context->GetOutputShape(2);
    const bool lse = *context->GetAttrs()->GetAttrPointer<bool>(8);
    if (!lse) {
        *maxShape = gert::Shape({0});
        *sumShape = gert::Shape({0});
        return ge::GRAPH_SUCCESS;
    }
    const int64_t kvHeads = key->GetDim(2);
    // KVP_COMM changes only the attention output layout.  Keeping max/sum
    // in native [N2,T,G] order allows SFAD to use one contiguous MTE3
    // write per half-head group; LSE preparation remains asynchronous.
    *maxShape = gert::Shape(
        {kvHeads, totalQuery, queryHeads / kvHeads});
    *sumShape = *maxShape;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InferDataTypeSparseFlashAttentionDecode(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, ge::DT_FLOAT);
    context->SetOutputDataType(1, ge::DT_FLOAT);
    context->SetOutputDataType(2, ge::DT_FLOAT);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(SparseFlashAttentionDecode)
    .InferShape(InferShapeSparseFlashAttentionDecode)
    .InferDataType(InferDataTypeSparseFlashAttentionDecode);
} // namespace ops
