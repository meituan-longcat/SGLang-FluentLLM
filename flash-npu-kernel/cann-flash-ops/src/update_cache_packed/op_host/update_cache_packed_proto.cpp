// Copyright (c) 2026. Licensed under the Apache License, Version 2.0.
// Packed-row adapter for the existing KV offload implementation.
#include "exe_graph/runtime/infer_shape_context.h"
#include "exe_graph/runtime/infer_datatype_context.h"
#include "register/op_impl_registry.h"
#include "error/ops_error.h"
namespace ops {
static ge::graphStatus Shape(gert::InferShapeContext* context) {
 for (size_t i=0;i<3;++i) {
  auto input=context->GetInputShape(i+2); auto output=context->GetOutputShape(i);
  if (!input || !output) return ge::GRAPH_FAILED;
  *output=*input;
 }
 return ge::GRAPH_SUCCESS;
}
static ge::graphStatus Dtype(gert::InferDataTypeContext* context) {
 context->SetOutputDataType(0,ge::DT_INT8);
 context->SetOutputDataType(1,ge::DT_INT32);
 context->SetOutputDataType(2,ge::DT_INT32);
 return ge::GRAPH_SUCCESS;
}
IMPL_OP_INFERSHAPE(UpdateCachePacked).InferShape(Shape).InferDataType(Dtype);
}
