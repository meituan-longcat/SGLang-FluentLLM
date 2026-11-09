#include "register/op_def_registry.h"

namespace ops {
class SelectLocalTopkIndices : public OpDef {
public:
    explicit SelectLocalTopkIndices(const char *name) : OpDef(name)
    {
        this->Input("topk_indices_local").ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND}).AutoContiguous();
        this->Input("topk_pos").ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND}).AutoContiguous();
        this->Output("topk_indices").ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("valid_chunks").ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("kvp_rank").Int();

        OpAICoreConfig config;
        config.DynamicCompileStaticFlag(true).DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true).DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false).PrecisionReduceFlag(true);
        this->AICore().AddConfig("ascend910b", config);
        this->AICore().AddConfig("ascend910_93", config);
    }
};
OP_ADD(SelectLocalTopkIndices);
} // namespace ops
