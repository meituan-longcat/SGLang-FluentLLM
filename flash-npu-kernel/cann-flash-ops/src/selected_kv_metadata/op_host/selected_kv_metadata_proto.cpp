#include "register/op_impl_registry.h"
namespace ops {
static ge::graphStatus Shape(gert::InferShapeContext *c) {
    auto q = c->GetInputShape(0);
    auto a = c->GetAttrs();
    int64_t world = *a->GetInt(0), local = *a->GetInt(1);
    int64_t span = ((local + 5) / 6 + 255) / 256 * 256, cores = world * 6;
    *c->GetOutputShape(0) = gert::Shape({q->GetDim(0), 2048 + world * 8 + 8});
    *c->GetOutputShape(1) = gert::Shape({cores * span / 16});
    *c->GetOutputShape(2) = gert::Shape({cores * span});
    *c->GetOutputShape(3) = gert::Shape({cores, 8});
    *c->GetOutputShape(4) = gert::Shape({q->GetDim(0) * q->GetDim(1)});
    *c->GetOutputShape(5) = *q;
    auto p = c->GetInputShape(1);
    *c->GetOutputShape(6) = gert::Shape({p->GetDim(0), p->GetDim(1), 8});
    *c->GetOutputShape(7) = gert::Shape({cores, 8});
    return ge::GRAPH_SUCCESS;
}
static ge::graphStatus Dtype(gert::InferDataTypeContext *c) {
    for (int i = 0; i < 8; ++i)
        c->SetOutputDataType(i, ge::DT_INT32);
    return ge::GRAPH_SUCCESS;
}
IMPL_OP_INFERSHAPE(SelectedKvMetadata).InferShape(Shape).InferDataType(Dtype);
} // namespace ops
