// Copyright (c) 2026. Licensed under the Apache License, Version 2.0.
// Packed-row adapter for the existing KV offload implementation.
#include "register/op_def_registry.h"
namespace ops {
class GatherByCacheIndicesPacked : public OpDef {
public:
 explicit GatherByCacheIndicesPacked(const char *name) : OpDef(name) {
  this->Input("cache_packed").ParamType(REQUIRED).DataType({ge::DT_INT8}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
  this->Input("from_cache_indices").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
  this->Input("topk").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
  this->Input("full_packed").ParamType(REQUIRED).DataType({ge::DT_INT8}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
  this->Input("full_block_table").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
  this->Input("selection_block_table").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
  this->Input("sources").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
  this->Input("kv_lengths").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
  this->Input("query_lengths").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
  this->Output("selection_packed").ParamType(REQUIRED).DataType({ge::DT_INT8}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
  this->Output("current_write_slots").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
  this->Attr("selection_topk_block_size").AttrType(OPTIONAL).Int(1);
  this->Attr("selection_kv_block_num").AttrType(OPTIONAL).Int(0);
  this->Attr("selection_kv_block_size").AttrType(OPTIONAL).Int(128);
  this->Attr("enable_slots_contiguous").AttrType(OPTIONAL).Int(2);
  this->AICore().AddConfig("ascend910b");
 }
};
OP_ADD(GatherByCacheIndicesPacked);
}
