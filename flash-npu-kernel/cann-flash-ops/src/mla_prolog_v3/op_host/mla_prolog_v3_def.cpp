/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "../op_kernel/mla_prolog_v3_policy.h"

#include "register/op_def_registry.h"

namespace ops {
#if MLA_PROLOG_V3_SFA1
class FluentMlaPrologV3 : public OpDef {
#else
class MlaPrologV3 : public OpDef {
#endif
public:
#if MLA_PROLOG_V3_SFA1
    explicit FluentMlaPrologV3(const char *name) : OpDef(name)
#else
    explicit MlaPrologV3(const char *name) : OpDef(name)
#endif
    {
        this->Input("token_x")
            .ParamType(REQUIRED)
#if MLA_PROLOG_V3_SFA1
            .DataType({ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_BF16, ge::DT_INT8})
#else
            .DataType({ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_BF16, ge::DT_INT8, ge::DT_BF16})
#endif
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("weight_dq")
            .ParamType(REQUIRED)
#if MLA_PROLOG_V3_SFA1
            .DataType({ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_BF16, ge::DT_INT8})
#else
            .DataType({ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_BF16, ge::DT_INT8, ge::DT_BF16})
#endif
            .FormatList({ge::FORMAT_FRACTAL_NZ})
            .AutoContiguous();
        this->Input("weight_uq_qr")
            .ParamType(REQUIRED)
#if MLA_PROLOG_V3_SFA1
            .DataType({ge::DT_BF16, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8})
#else
            .DataType({ge::DT_BF16, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_BF16})
#endif
            .FormatList({ge::FORMAT_FRACTAL_NZ})
            .AutoContiguous();
        this->Input("weight_uk")
            .ParamType(REQUIRED)
#if MLA_PROLOG_V3_SFA1
            .DataType({ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16})
#else
            .DataType({ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16})
#endif
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("weight_dkv_kr")
            .ParamType(REQUIRED)
#if MLA_PROLOG_V3_SFA1
            .DataType({ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_BF16, ge::DT_INT8})
#else
            .DataType({ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_BF16, ge::DT_INT8, ge::DT_BF16})
#endif
            .FormatList({ge::FORMAT_FRACTAL_NZ})
            .AutoContiguous();
        this->Input("rmsnorm_gamma_cq")
            .ParamType(REQUIRED)
#if MLA_PROLOG_V3_SFA1
            .DataType({ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16})
#else
            .DataType({ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16})
#endif
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("rmsnorm_gamma_ckv")
            .ParamType(REQUIRED)
#if MLA_PROLOG_V3_SFA1
            .DataType({ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16})
#else
            .DataType({ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16})
#endif
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("rope_sin")
            .ParamType(REQUIRED)
#if MLA_PROLOG_V3_SFA1
            .DataType({ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16})
#else
            .DataType({ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16})
#endif
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("rope_cos")
            .ParamType(REQUIRED)
#if MLA_PROLOG_V3_SFA1
            .DataType({ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16})
#else
            .DataType({ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16})
#endif
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("kv_cache")
            .ParamType(REQUIRED)
#if MLA_PROLOG_V3_SFA1
            .DataType({ge::DT_BF16, ge::DT_INT8, ge::DT_BF16, ge::DT_INT8, ge::DT_BF16, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8})
#else
            .DataType({ge::DT_BF16, ge::DT_INT8, ge::DT_BF16, ge::DT_INT8, ge::DT_BF16, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8})
#endif
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("kr_cache")
            .ParamType(REQUIRED)
#if MLA_PROLOG_V3_SFA1
            .DataType({ge::DT_BF16, ge::DT_INT8, ge::DT_BF16, ge::DT_INT8, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16})
#else
            .DataType({ge::DT_BF16, ge::DT_INT8, ge::DT_BF16, ge::DT_INT8, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16})
#endif
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("cache_index")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_INT64})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("dequant_scale_x")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_FLOAT})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("dequant_scale_w_dq")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_FLOAT})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("dequant_scale_w_uq_qr")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_FLOAT})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("dequant_scale_w_dkv_kr")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_FLOAT})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("quant_scale_ckv")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_FLOAT})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("quant_scale_ckr")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_FLOAT})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("smooth_scales_cq")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_FLOAT})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("actual_seq_len")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("k_nope_clip_alpha")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_FLOAT})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Output("query")
            .ParamType(REQUIRED)
#if MLA_PROLOG_V3_SFA1
            .DataType({ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_INT8, ge::DT_BF16, ge::DT_BF16})
#else
            .DataType({ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_INT8, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16})
#endif
            .FormatList({ge::FORMAT_ND});
        this->Output("query_rope")
            .ParamType(REQUIRED)
#if MLA_PROLOG_V3_SFA1
            .DataType({ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16})
#else
            .DataType({ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16})
#endif
            .FormatList({ge::FORMAT_ND});
        this->Output("kv_cache")
            .ParamType(REQUIRED)
#if MLA_PROLOG_V3_SFA1
            .DataType({ge::DT_BF16, ge::DT_INT8, ge::DT_BF16, ge::DT_INT8, ge::DT_BF16, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8})
#else
            .DataType({ge::DT_BF16, ge::DT_INT8, ge::DT_BF16, ge::DT_INT8, ge::DT_BF16, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8})
#endif
            .FormatList({ge::FORMAT_ND});
        this->Output("kr_cache")
            .ParamType(REQUIRED)
#if MLA_PROLOG_V3_SFA1
            .DataType({ge::DT_BF16, ge::DT_INT8, ge::DT_BF16, ge::DT_INT8, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16})
#else
            .DataType({ge::DT_BF16, ge::DT_INT8, ge::DT_BF16, ge::DT_INT8, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16})
#endif
            .FormatList({ge::FORMAT_ND});
        this->Output("dequant_scale_q_nope")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_FLOAT})
            .FormatList({ge::FORMAT_ND});
        this->Output("query_norm")
            .ParamType(REQUIRED)
#if MLA_PROLOG_V3_SFA1
            .DataType({ge::DT_BF16, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8})
#else
            .DataType({ge::DT_BF16, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_BF16})
#endif
            .FormatList({ge::FORMAT_ND});
        this->Output("dequant_scale_q_norm")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_FLOAT})
#if MLA_PROLOG_V3_SFA1
            .FormatList({ge::FORMAT_ND});
        this->Output("cur_k_nope")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_INT8, ge::DT_BF16, ge::DT_INT8, ge::DT_BF16, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8})
            .FormatList({ge::FORMAT_ND});
        this->Output("cur_k_rope")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_INT8, ge::DT_BF16, ge::DT_INT8, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16})
#endif
            .FormatList({ge::FORMAT_ND});
        this->Attr("rmsnorm_epsilon_cq").AttrType(OPTIONAL).Float(1e-05f);
        this->Attr("rmsnorm_epsilon_ckv").AttrType(OPTIONAL).Float(1e-05f);
        this->Attr("cache_mode").AttrType(OPTIONAL).String("PA_BSND");
        this->Attr("query_norm_flag").AttrType(OPTIONAL).Bool(false);
        this->Attr("weight_quant_mode").AttrType(OPTIONAL).Int(0);
        this->Attr("kv_cache_quant_mode").AttrType(OPTIONAL).Int(0);
        this->Attr("query_quant_mode").AttrType(OPTIONAL).Int(0);
        this->Attr("ckvkr_repo_mode").AttrType(OPTIONAL).Int(0);
        this->Attr("quant_scale_repo_mode").AttrType(OPTIONAL).Int(0);
        this->Attr("tile_size").AttrType(OPTIONAL).Int(128); // 128 : set value of tile size
        this->Attr("qc_qr_scale").AttrType(OPTIONAL).Float(1.0f);
        this->Attr("kc_scale").AttrType(OPTIONAL).Float(1.0f);
#if MLA_PROLOG_V3_SFA1
        this->Attr("output_cur_k_nope_rope").AttrType(OPTIONAL).Bool(false);
#else
        this->Attr("enable_rope").AttrType(OPTIONAL).Bool(true);
#endif
        OpAICoreConfig aicore_config;
        aicore_config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("aclnnSupport.value", "support_aclnn");   // set value of aclnn support
        this->AICore().AddConfig("ascend910b", aicore_config);
        this->AICore().AddConfig("ascend910_93", aicore_config);
    }
};
#if MLA_PROLOG_V3_SFA1
OP_ADD(FluentMlaPrologV3, optiling::fluent_mla_prolog_v3::MlaPrologV3CompileInfo);
#else
OP_ADD(MlaPrologV3, optiling::MlaPrologV3CompileInfo);
#endif
} // namespace ops
