// Copyright (c) 2026. Licensed under the Apache License, Version 2.0.
// Packed-row adapter for the existing KV offload implementation.
#include "exe_graph/runtime/infer_shape_context.h"
#include "exe_graph/runtime/infer_datatype_context.h"
#include "register/op_impl_registry.h"
#include "error/ops_error.h"
namespace ops {
static ge::graphStatus Shape(gert::InferShapeContext* context) {
 auto topk=context->GetInputShape(2);
 auto attrs=context->GetAttrs();
 if (!topk || !attrs || topk->GetDimNum()!=3) return ge::GRAPH_FAILED;
 auto pages=attrs->GetAttrPointer<int64_t>(1);
 auto out=context->GetOutputShape(0); auto slots=context->GetOutputShape(1);
 if (!pages || !out || !slots) return ge::GRAPH_FAILED;
 out->SetDimNum(3); out->SetDim(0,*pages); out->SetDim(1,128); out->SetDim(2,672);
 slots->SetDimNum(3); slots->SetDim(0,topk->GetDim(0)); slots->SetDim(1,topk->GetDim(1)); slots->SetDim(2,topk->GetDim(1));
 return ge::GRAPH_SUCCESS;
}
static ge::graphStatus Dtype(gert::InferDataTypeContext* context) {
 context->SetOutputDataType(0,ge::DT_INT8);
 context->SetOutputDataType(1,ge::DT_INT32);
 return ge::GRAPH_SUCCESS;
}
IMPL_OP_INFERSHAPE(GatherByCacheIndicesPacked).InferShape(Shape).InferDataType(Dtype);
}
