#include "register/op_impl_registry.h"

namespace ops {
ge::graphStatus InferShapeSelectLocalTopkIndices(gert::InferShapeContext *context)
{
    const auto *local = context->GetInputShape(0);
    const auto *position = context->GetInputShape(1);
    const int64_t tokens = position->GetDim(0);
    const int64_t topk = position->GetDim(1);
    *context->GetOutputShape(0) = gert::Shape({tokens, 1, topk});
    *context->GetOutputShape(1) = gert::Shape({tokens});
    (void)local;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InferDataTypeSelectLocalTopkIndices(
    gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, ge::DT_INT32);
    context->SetOutputDataType(1, ge::DT_INT32);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(SelectLocalTopkIndices)
    .InferShape(InferShapeSelectLocalTopkIndices)
    .InferDataType(InferDataTypeSelectLocalTopkIndices);
} // namespace ops
