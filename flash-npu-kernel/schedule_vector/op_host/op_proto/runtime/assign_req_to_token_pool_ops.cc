#include "runtime_util.h"
#include "op_log.h"

namespace ge {

static ge::graphStatus InferShape4AssignReqToTokenPool(gert::InferShapeContext* context)
{
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType4AssignReqToTokenPool(gert::InferDataTypeContext *context)
{
    const auto inputDataType = context->GetInputDataType(3);
    context->SetOutputDataType(0, inputDataType);
    return ge::GRAPH_SUCCESS;
}
IMPL_OP_INFERSHAPE(AssignReqToTokenPool)
    .InferShape(InferShape4AssignReqToTokenPool)
    .InferDataType(InferDataType4AssignReqToTokenPool);

}