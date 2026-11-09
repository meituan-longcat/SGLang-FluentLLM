#include "register/op_def_registry.h"

namespace ops {

// DFlash 推理 PageAttention（910B / arch22），自 FusedInferAttentionScore sparse_mode=15 裁剪而来。
// 复用 serving 框架的 PageAttention KV cache 与 block_table，不 gather/重建 KV。
// 每个 request 一个 draft block：context_end = kvLen - qLen 即 anchor，
// 可见逻辑 KV 集合 = [0, sink_end) U [tail_start, kvLen)，其中
//   tail_start = max(0, context_end - pre_tokens)（pre_tokens=INT32_MAX 表示全 context）
//   sink_end   = min(next_tokens, context_end)（next_tokens=0 关闭 sink；正数必须为 8 的倍数）
// draft 块（最后 qLen 个逻辑 token）双向可见。被跳过的区间不产生 tile、不读 GM、不参与计算。
class MlpDflashInferAttn : public OpDef {
public:
    explicit MlpDflashInferAttn(const char* name) : OpDef(name)
    {
        // 两组 dtype 配置：[0]=fp16, [1]=bf16。所有输入/输出的列表长度必须一致。
        this->Input("query")      // TND: [totalTokens, N, D]
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("key")        // PageAttention K cache：[numBlocks, blockSize, Nkv*D] 或 [numBlocks, blockSize, Nkv, D]，页内 token 连续，物理页经 block_table 索引
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("value")      // PageAttention V cache，同 key
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("block_table")   // [B, maxNumPagesPerReq] int32：逻辑页号 -> 物理页号
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("actual_seq_lengths")    // [B] int64：每个 request 的 draft block 长度 q_len（TND 前缀和）
            .ParamType(OPTIONAL)
            .ValueDepend(OPTIONAL)           // DFlash 语义必须提供；host tiling 需读值做任务划分
            .DataType({ge::DT_INT64, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("actual_seq_lengths_kv") // [B] int64：每个 request 的逻辑 KV 长度（含 draft 块）
            .ParamType(OPTIONAL)
            .ValueDepend(OPTIONAL)
            .DataType({ge::DT_INT64, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});

        this->Output("attention_out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("softmax_lse")  // [totalTokens, N] float
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});

        this->Attr("num_heads").AttrType(REQUIRED).Int(1);
        this->Attr("scale").AttrType(OPTIONAL).Float(1.0);
        // DFlash 滑窗（尾带长度）：INT32_MAX=全 context；否则保留 anchor 前 pre_tokens 个 token
        this->Attr("pre_tokens").AttrType(OPTIONAL).Int(2147483647);
        // DFlash prefix sink 长度：0=关闭；正数必须为 8 的倍数。默认 16
        this->Attr("next_tokens").AttrType(OPTIONAL).Int(16);
        // 仅支持 TND
        this->Attr("input_layout").AttrType(OPTIONAL).String("TND");
        this->Attr("num_key_value_heads").AttrType(OPTIONAL).Int(0);
        // PageAttention 页大小：必须等于 K/V cache 最后一维
        this->Attr("block_size").AttrType(OPTIONAL).Int(0);
        this->Attr("softmax_lse_flag").AttrType(OPTIONAL).Bool(false);

        OpAICoreConfig aicoreConfig;
        aicoreConfig.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true);
        this->AICore().AddConfig("ascend910b", aicoreConfig);
        this->AICore().AddConfig("ascend910_93", aicoreConfig);
    }
};

OP_ADD(MlpDflashInferAttn);

}  // namespace ops
