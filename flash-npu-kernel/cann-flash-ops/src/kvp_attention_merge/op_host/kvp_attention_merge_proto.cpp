#include "register/op_impl_registry.h"

namespace ops {

static ge::graphStatus InferShapeKvpAttentionMerge(
    gert::InferShapeContext *context)
{
    const auto *attentionOut = context->GetInputShape(0);
    const auto *lse = context->GetInputShape(1);
    auto *output = context->GetOutputShape(0);
    if (attentionOut == nullptr || lse == nullptr || output == nullptr ||
        attentionOut->GetDimNum() != 3 || lse->GetDimNum() != 2) {
        return ge::GRAPH_FAILED;
    }
    if (attentionOut->GetDim(0) != lse->GetDim(0) ||
        attentionOut->GetDim(1) != lse->GetDim(1)) {
        return ge::GRAPH_FAILED;
    }
    *output = gert::Shape(
        {attentionOut->GetDim(1), attentionOut->GetDim(2)});
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeKvpAttentionMerge(
    gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, ge::DT_BF16);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(KvpAttentionMerge)
    .InferShape(InferShapeKvpAttentionMerge)
    .InferDataType(InferDataTypeKvpAttentionMerge);

} // namespace ops
