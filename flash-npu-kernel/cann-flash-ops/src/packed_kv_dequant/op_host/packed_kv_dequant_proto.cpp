#include "register/op_impl_registry.h"

namespace ops {

namespace {
constexpr int64_t PACKED_PAYLOAD_BYTES = 656;
constexpr int64_t PACKED_ALIGNED_ROW_BYTES = 672;
}

static ge::graphStatus InferShapePackedKvDequant(
    gert::InferShapeContext *context)
{
    const auto *input = context->GetInputShape(0);
    auto *key = context->GetOutputShape(0);
    auto *rope = context->GetOutputShape(1);
    if (input == nullptr || key == nullptr || rope == nullptr ||
        input->GetDimNum() != 3 || input->GetDim(1) != 1 ||
        (input->GetDim(2) != PACKED_PAYLOAD_BYTES &&
         input->GetDim(2) != PACKED_ALIGNED_ROW_BYTES)) {
        return ge::GRAPH_FAILED;
    }
    *key = gert::Shape({input->GetDim(0), 1, 512});
    *rope = gert::Shape({input->GetDim(0), 1, 64});
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypePackedKvDequant(
    gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, ge::DT_BF16);
    context->SetOutputDataType(1, ge::DT_BF16);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(PackedKvDequant)
    .InferShape(InferShapePackedKvDequant)
    .InferDataType(InferDataTypePackedKvDequant);

} // namespace ops
