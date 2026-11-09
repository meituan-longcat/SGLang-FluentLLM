#include "register/op_impl_registry.h"

namespace ops {

static ge::graphStatus InferShapeMoeExpertIdMap(
    gert::InferShapeContext *context)
{
    const auto *logicalIds = context->GetInputShape(0);
    auto *physicalIds = context->GetOutputShape(0);
    if (logicalIds == nullptr || physicalIds == nullptr) {
        return ge::GRAPH_FAILED;
    }
    *physicalIds = *logicalIds;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeMoeExpertIdMap(
    gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, ge::DT_INT32);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(MoeExpertIdMap)
    .InferShape(InferShapeMoeExpertIdMap)
    .InferDataType(InferDataTypeMoeExpertIdMap);

} // namespace ops
