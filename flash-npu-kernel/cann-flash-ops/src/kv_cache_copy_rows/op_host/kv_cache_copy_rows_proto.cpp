#include "register/op_impl_registry.h"

namespace ops {

static ge::graphStatus InferShapeKvCacheCopyRows(gert::InferShapeContext* context)
{
    const gert::Shape* inputShape = context->GetInputShape(0);
    gert::Shape* outputShape = context->GetOutputShape(0);
    if (inputShape == nullptr || outputShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    *outputShape = *inputShape;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeKvCacheCopyRows(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(KvCacheCopyRows)
    .InferShape(InferShapeKvCacheCopyRows)
    .InferDataType(InferDataTypeKvCacheCopyRows);

} // namespace ops
