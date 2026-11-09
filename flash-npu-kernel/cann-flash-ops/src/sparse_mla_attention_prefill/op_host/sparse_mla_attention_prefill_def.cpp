#include "register/op_def_registry.h"

namespace ops {
class SparseMlaAttentionPrefill : public OpDef {
public:
    explicit SparseMlaAttentionPrefill(const char *name) : OpDef(name)
    {
        // Prefill owns a replicated, rank-major gathered page workspace.  Q
        // and KV both carry latent + RoPE in one row so the kernel needs no
        // layout conversion or secondary RoPE gather.
        this->Input("query").ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND}).AutoContiguous();
        this->Input("gathered_kv_pages").ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND}).AutoContiguous();
        this->Input("logical_sparse_rows").ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND}).AutoContiguous();
        this->Input("scan_count").ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND}).AutoContiguous();
        this->Input("page_table").ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND}).AutoContiguous();
        this->Input("query_seq").ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND}).AutoContiguous();
        this->Output("attention_out").ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND});
        this->Attr("scale_value").AttrType(REQUIRED).Float(1.0);
        // false preserves [T,N,D]; true writes [N,T,D] so native
        // BatchMatMul can consume the attention result without an input
        // transpose/materialization.
        this->Attr("head_major_output").AttrType(OPTIONAL).Bool(false);

        OpAICoreConfig config;
        config.DynamicCompileStaticFlag(true).DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true).DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false).PrecisionReduceFlag(true);
        this->AICore().AddConfig("ascend910b", config);
        this->AICore().AddConfig("ascend910_93", config);
    }
};
OP_ADD(SparseMlaAttentionPrefill);
} // namespace ops
