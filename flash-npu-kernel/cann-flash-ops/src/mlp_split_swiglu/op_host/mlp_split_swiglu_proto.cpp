#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>

#include "error/ops_error.h"

using namespace ge;

namespace {
constexpr int64_t INPUT_X_INDEX = 0;
constexpr int64_t DENSE_DIM_NUM = 2;
}  // namespace

namespace ops {
static ge::graphStatus InferShapeForMlpSplitSwiglu(gert::InferShapeContext* context)
{
    OPS_ERR_IF(context == nullptr,
               OPS_LOG_E("MlpSplitSwiglu", "infer shape context is null"),
               return ge::GRAPH_FAILED);
    const gert::Shape* xShape = context->GetInputShape(INPUT_X_INDEX);
    OPS_LOG_E_IF_NULL(context, xShape, return ge::GRAPH_FAILED);
    OPS_ERR_IF(xShape->GetDimNum() != DENSE_DIM_NUM,
               OPS_LOG_E(context, "x dim num must be 2, got %zu", xShape->GetDimNum()),
               return ge::GRAPH_FAILED);

    const int64_t token = xShape->GetDim(0);
    const int64_t hidden = xShape->GetDim(1);
    int64_t outHidden = -1;
    if (hidden == -2) {
        return ge::GRAPH_FAILED;
    }
    if (hidden != -1) {
        OPS_ERR_IF((hidden % 2) != 0,
                   OPS_LOG_E(context, "hidden must be even, got %ld", hidden),
                   return ge::GRAPH_FAILED);
        outHidden = hidden / 2;
    }

    gert::Shape* outShape = context->GetOutputShape(0);
    OPS_LOG_E_IF_NULL(context, outShape, return ge::GRAPH_FAILED);
    *outShape = gert::Shape({token, outHidden});
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeForMlpSplitSwiglu(gert::InferDataTypeContext* context)
{
    OPS_ERR_IF(context == nullptr,
               OPS_LOG_E("MlpSplitSwiglu", "infer dtype context is null"),
               return ge::GRAPH_FAILED);
    context->SetOutputDataType(0, context->GetInputDataType(INPUT_X_INDEX));
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(MlpSplitSwiglu)
    .InferShape(InferShapeForMlpSplitSwiglu)
    .InferDataType(InferDataTypeForMlpSplitSwiglu);

}  // namespace ops
