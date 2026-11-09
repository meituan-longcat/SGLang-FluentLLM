#include "register/op_impl_registry.h"
namespace ops {
static ge::graphStatus InferShapeKvCacheCopyRowsBurst(gert::InferShapeContext* context)
{
    const size_t inputs = context->GetComputeNodeInputNum();
    if (inputs < 3 || inputs > 40 || context->GetComputeNodeOutputNum() != inputs - 2) {
        return ge::GRAPH_FAILED;
    }
    for (size_t i = 0; i < inputs - 2; ++i) {
        auto input = context->GetInputShape(i);
        auto output = context->GetOutputShape(i);
        if (input == nullptr || output == nullptr) { return ge::GRAPH_FAILED; }
        *output = *input;
    }
    return ge::GRAPH_SUCCESS;
}
static ge::graphStatus InferTypeKvCacheCopyRowsBurst(gert::InferDataTypeContext* context)
{
    const size_t inputs = context->GetComputeNodeInputNum();
    if (inputs < 3 || inputs > 40 || context->GetComputeNodeOutputNum() != inputs - 2) {
        return ge::GRAPH_FAILED;
    }
    for (size_t i = 0; i < inputs - 2; ++i) {
        context->SetOutputDataType(i, context->GetInputDataType(i));
    }
    return ge::GRAPH_SUCCESS;
}
IMPL_OP_INFERSHAPE(KvCacheCopyRowsBurst)
    .InferShape(InferShapeKvCacheCopyRowsBurst).InferDataType(InferTypeKvCacheCopyRowsBurst);
} // namespace ops
