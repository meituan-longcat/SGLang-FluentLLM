#include "register/op_impl_registry.h"

namespace ops {

static ge::graphStatus InferShapeRouterInputSliceCast(
    gert::InferShapeContext *context)
{
    const auto *input = context->GetInputShape(0);
    auto *output = context->GetOutputShape(0);
    const auto *attrs = context->GetAttrs();
    if (input == nullptr || output == nullptr || attrs == nullptr ||
        input->GetDimNum() != 2) {
        return ge::GRAPH_FAILED;
    }
    const auto *length = attrs->GetAttrPointer<int64_t>(1);
    if (length == nullptr || *length <= 0) {
        return ge::GRAPH_FAILED;
    }
    *output = gert::Shape({input->GetDim(0), *length});
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeRouterInputSliceCast(
    gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, ge::DT_FLOAT);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(RouterInputSliceCast)
    .InferShape(InferShapeRouterInputSliceCast)
    .InferDataType(InferDataTypeRouterInputSliceCast);

} // namespace ops
