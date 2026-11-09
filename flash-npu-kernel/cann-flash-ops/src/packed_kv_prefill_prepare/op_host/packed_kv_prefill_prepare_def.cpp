#include "register/op_def_registry.h"

namespace ops {
class PackedKvPrefillPrepare : public OpDef {
public:
    explicit PackedKvPrefillPrepare(const char *name) : OpDef(name)
    {
        this->Input("latent_cache").ParamType(REQUIRED).DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND}).AutoContiguous();
        this->Input("gamma").ParamType(REQUIRED).DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND}).AutoContiguous();
        this->Input("cos").ParamType(REQUIRED).DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND}).AutoContiguous();
        this->Input("sin").ParamType(REQUIRED).DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND}).AutoContiguous();
        this->Input("owner_rows").ParamType(REQUIRED).DataType({ge::DT_INT64})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND}).AutoContiguous();
        this->Output("key").ParamType(REQUIRED).DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("key_rope").ParamType(REQUIRED).DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("owner_packed").ParamType(REQUIRED).DataType({ge::DT_INT8})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("epsilon").AttrType(REQUIRED).Float();
        this->Attr("packed_row_bytes").AttrType(OPTIONAL).Int(656);

        OpAICoreConfig config;
        config.DynamicCompileStaticFlag(true).DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true).DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false).PrecisionReduceFlag(true)
            .ExtendCfgInfo("prebuildPattern.value", "Opaque")
            .ExtendCfgInfo("coreType.value", "VectorCore")
            .ExtendCfgInfo("jitCompile.flag", "false");
        this->AICore().AddConfig("ascend910b", config);
        this->AICore().AddConfig("ascend910_93", config);
    }
};
OP_ADD(PackedKvPrefillPrepare);
} // namespace ops
