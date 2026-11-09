#include "register/op_def_registry.h"

namespace ops {
class KvCacheCopyRowsBurst : public OpDef {
public:
    explicit KvCacheCopyRowsBurst(const char* name) : OpDef(name)
    {
        this->Input("caches").ParamType(DYNAMIC)
            .DataType({ge::DT_INT8, ge::DT_INT8, ge::DT_BF16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("src_indices").ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT64, ge::DT_INT32, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("dst_indices").ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT64, ge::DT_INT32, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        // Same-name dynamic input/output is the inplace TensorList contract.
        this->Output("caches").ParamType(DYNAMIC)
            .DataType({ge::DT_INT8, ge::DT_INT8, ge::DT_BF16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->AICore().AddConfig("ascend910b");
    }
};
OP_ADD(KvCacheCopyRowsBurst);
} // namespace ops
