#include "register/op_impl_registry.h"

namespace ops {
namespace {
constexpr int64_t PACKED_PAYLOAD_BYTES = 656;
constexpr int64_t PACKED_ALIGNED_ROW_BYTES = 672;

static ge::graphStatus InferShapePackedKvPrefillPrepare(gert::InferShapeContext *context)
{
    const auto *latent = context->GetInputShape(0);
    const auto *owners = context->GetInputShape(4);
    if (latent == nullptr || owners == nullptr || latent->GetDimNum() != 2 ||
        owners->GetDimNum() != 1) {
        return ge::GRAPH_FAILED;
    }
    const auto *attrs = context->GetAttrs();
    const int64_t *packedRowBytes =
        attrs == nullptr ? nullptr : attrs->GetAttrPointer<int64_t>(1);
    if (packedRowBytes == nullptr ||
        (*packedRowBytes != PACKED_PAYLOAD_BYTES &&
         *packedRowBytes != PACKED_ALIGNED_ROW_BYTES)) {
        return ge::GRAPH_FAILED;
    }
    *context->GetOutputShape(0) = gert::Shape({latent->GetDim(0), 1, 512});
    *context->GetOutputShape(1) = gert::Shape({latent->GetDim(0), 1, 64});
    *context->GetOutputShape(2) =
        gert::Shape({owners->GetDim(0), 1, *packedRowBytes});
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypePackedKvPrefillPrepare(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, ge::DT_BF16);
    context->SetOutputDataType(1, ge::DT_BF16);
    context->SetOutputDataType(2, ge::DT_INT8);
    return ge::GRAPH_SUCCESS;
}
} // namespace

IMPL_OP_INFERSHAPE(PackedKvPrefillPrepare)
    .InferShape(InferShapePackedKvPrefillPrepare)
    .InferDataType(InferDataTypePackedKvPrefillPrepare);
} // namespace ops
