#include <register/op_impl_registry.h>

namespace ops {
ge::graphStatus InferShapeSparseMlaAttentionPrefill(
    gert::InferShapeContext *context)
{
    const auto *query = context->GetInputShape(0);
    if (query == nullptr || query->GetDimNum() != 3 ||
        query->GetDim(1) != 64 || query->GetDim(2) != 576) {
        return ge::GRAPH_FAILED;
    }
    const auto *attrs = context->GetAttrs();
    const bool *headMajorPtr =
        attrs == nullptr ? nullptr : attrs->GetAttrPointer<bool>(1);
    const bool headMajor = headMajorPtr != nullptr && *headMajorPtr;
    *context->GetOutputShape(0) = headMajor
        ? gert::Shape({64, query->GetDim(0), 512})
        : gert::Shape({query->GetDim(0), 64, 512});
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InferDataTypeSparseMlaAttentionPrefill(
    gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, ge::DT_BF16);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(SparseMlaAttentionPrefill)
    .InferShape(InferShapeSparseMlaAttentionPrefill)
    .InferDataType(InferDataTypeSparseMlaAttentionPrefill);
} // namespace ops
