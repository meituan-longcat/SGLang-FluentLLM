#include "runtime_util.h"
#include "op_log.h"

namespace ge {

static ge::graphStatus InferShape4RearrangeAcceptIndex(gert::InferShapeContext* context)
{
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType4RearrangeAcceptIndex(gert::InferDataTypeContext *context)
{
    const auto inputDataType = context->GetInputDataType(3);
    context->SetOutputDataType(0, inputDataType);
    return ge::GRAPH_SUCCESS;
}
IMPL_OP_INFERSHAPE(RearrangeAcceptIndex)
    .InferShape(InferShape4RearrangeAcceptIndex)
    .InferDataType(InferDataType4RearrangeAcceptIndex);
}