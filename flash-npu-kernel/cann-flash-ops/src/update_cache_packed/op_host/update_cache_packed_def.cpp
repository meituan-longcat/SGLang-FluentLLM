// Copyright (c) 2026. Licensed under the Apache License, Version 2.0.
// Packed-row adapter for the existing KV offload implementation.
#include "register/op_def_registry.h"
namespace ops {
class UpdateCachePacked : public OpDef {
public:
 explicit UpdateCachePacked(const char *name) : OpDef(name) {
  this->Input("topk").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
  this->Input("selection_packed").ParamType(REQUIRED).DataType({ge::DT_INT8}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
  this->Input("cache_packed").ParamType(REQUIRED).DataType({ge::DT_INT8}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
  this->Input("status").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
  this->Input("age").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
  this->Input("selection_block_table").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
  this->Input("write_slot").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
  this->Input("step").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
  this->Output("new_cache_packed").ParamType(REQUIRED).DataType({ge::DT_INT8}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
  this->Output("new_status").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
  this->Output("new_age").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
  this->AICore().AddConfig("ascend910b");
 }
};
OP_ADD(UpdateCachePacked);
}
