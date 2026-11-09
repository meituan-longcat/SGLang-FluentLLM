// Copyright (c) 2026. Licensed under the Apache License, Version 2.0.
// Packed-row adapter for the existing KV offload implementation.
#include "register/op_def_registry.h"
namespace ops {
class UpdateCurKvToSelectionPacked : public OpDef {
public:
 explicit UpdateCurKvToSelectionPacked(const char *name) : OpDef(name) {
  this->Input("current_packed").ParamType(REQUIRED).DataType({ge::DT_INT8}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
  this->Input("current_write_slots").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
  this->Input("selection_packed").ParamType(REQUIRED).DataType({ge::DT_INT8}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
  this->Input("selection_block_table").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
  this->Output("selection_packed").ParamType(REQUIRED).DataType({ge::DT_INT8}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
  this->Output("kernel_done").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
  this->AICore().AddConfig("ascend910b");
 }
};
OP_ADD(UpdateCurKvToSelectionPacked);
}
