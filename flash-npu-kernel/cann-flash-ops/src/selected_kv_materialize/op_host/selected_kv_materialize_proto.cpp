#include "register/op_impl_registry.h"
namespace ops {
static ge::graphStatus Shape(gert::InferShapeContext *c) {
    auto p = c->GetInputShape(0);
    auto e = c->GetInputShape(1);
    *c->GetOutputShape(0) = gert::Shape({(p->GetDim(0) + e->GetDim(0) + 127) / 128, 128, 576});
    return ge::GRAPH_SUCCESS;
}
static ge::graphStatus Dtype(gert::InferDataTypeContext *c) {
    c->SetOutputDataType(0, ge::DT_BF16);
    return ge::GRAPH_SUCCESS;
}
IMPL_OP_INFERSHAPE(SelectedKvMaterialize).InferShape(Shape).InferDataType(Dtype);
} // namespace ops
