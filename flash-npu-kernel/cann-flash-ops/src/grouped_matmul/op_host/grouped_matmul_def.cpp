/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file grouped_matmul.cpp
 * \brief
 */

#include "register/op_def_registry.h"
namespace ops {
class GroupedMatmul : public OpDef {
public:
    explicit GroupedMatmul(const char *name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(DYNAMIC)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_INT8, ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_FLOAT,
                       ge::DT_INT8,    ge::DT_INT8, ge::DT_INT8, ge::DT_INT8,    ge::DT_FLOAT16, ge::DT_BF16,
                       ge::DT_FLOAT16, ge::DT_BF16, ge::DT_INT8, ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,
                       ge::DT_INT8,    ge::DT_INT8, ge::DT_INT4, ge::DT_INT4,    ge::DT_INT4,    ge::DT_INT4,
                       ge::DT_INT8,    ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("weight")
            .ParamType(DYNAMIC)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_FLOAT,
                       ge::DT_INT8,    ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT4, ge::DT_INT4,
                       ge::DT_FLOAT16, ge::DT_BF16, ge::DT_INT8, ge::DT_INT8, ge::DT_INT4, ge::DT_INT4,
                       ge::DT_INT4,    ge::DT_INT4, ge::DT_INT4, ge::DT_INT4, ge::DT_INT4, ge::DT_INT4,
                       ge::DT_INT8,    ge::DT_INT8})
            .Format({ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ,
                     ge::FORMAT_ND,         ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ,
                     ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ});
        this->Input("bias")
            .ParamType(DYNAMIC)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32,   ge::DT_FLOAT16, ge::DT_FLOAT,   ge::DT_FLOAT,
                       ge::DT_INT32,   ge::DT_INT32, ge::DT_INT32,   ge::DT_INT32,   ge::DT_FLOAT16, ge::DT_FLOAT,
                       ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32,   ge::DT_INT32,   ge::DT_FLOAT,   ge::DT_FLOAT,
                       ge::DT_FLOAT,   ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
                       ge::DT_BF16,    ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("scale")
            .ParamType(DYNAMIC)
            .DataType({ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64,
                       ge::DT_BF16,   ge::DT_BF16,   ge::DT_FLOAT,  ge::DT_FLOAT,  ge::DT_UINT64, ge::DT_UINT64,
                       ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64,
                       ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64,
                       ge::DT_BF16,   ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("offset")
            .ParamType(DYNAMIC)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("antiquant_scale")
            .ParamType(DYNAMIC)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_FLOAT16,
                       ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16,
                       ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
                       ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
                       ge::DT_FLOAT16, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("antiquant_offset")
            .ParamType(DYNAMIC)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_FLOAT16,
                       ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16,
                       ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
                       ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
                       ge::DT_FLOAT16, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("group_list")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64,
                       ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64,
                       ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64,
                       ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("per_token_scale")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(DYNAMIC)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_INT8,    ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_FLOAT,
                       ge::DT_BF16,    ge::DT_BF16,    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16,
                       ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_INT32,   ge::DT_INT32,   ge::DT_BF16,    ge::DT_FLOAT16,
                       ge::DT_BF16,    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_FLOAT16, ge::DT_BF16,
                       ge::DT_BF16,    ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("split_item").AttrType(OPTIONAL).Int(0);
        this->Attr("dtype").AttrType(OPTIONAL).Int(0); // Reserved parameter, indicate output data type, not enabled.
        this->Attr("transpose_weight")
            .AttrType(OPTIONAL)
            .Bool(false); // Reserved parameter, indicate wether input weight is transposed, not enabled.
        this->Attr("transpose_x")
            .AttrType(OPTIONAL)
            .Bool(false); // Reserved parameter, indicate wether input x is transposed, not enabled.
        this->Attr("group_type").AttrType(OPTIONAL).Int(-1); // Indicates the splited dimension.
        this->Attr("group_list_type")
            .AttrType(OPTIONAL)
            .Int(0); // Indicates whether the value in group_dist is cumsum or count.
        this->Attr("act_type").AttrType(OPTIONAL).Int(0);            // Indicate activation function type.
        this->Attr("tuning_config").AttrType(OPTIONAL).ListInt({0}); // Indicate activation function type.

        OpAICoreConfig aicore_config;
        aicore_config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("prebuildPattern.value", "Opaque")
            .ExtendCfgInfo("coreType.value", "AiCore")
            .ExtendCfgInfo("aclnnSupport.value", "support_aclnn");

        this->AICore().AddConfig("ascend910b", aicore_config);
        this->AICore().AddConfig("ascend910_93", aicore_config);

        OpAICoreConfig config310P;
        config310P.Input("x")
            .ParamType(DYNAMIC)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND});
        config310P.Input("weight")
            .ParamType(DYNAMIC)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_FRACTAL_NZ});
        config310P.Input("bias")
            .ParamType(DYNAMIC)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND});
        config310P.Input("scale")
            .ParamType(DYNAMIC)
            .DataType({ge::DT_UINT64})
            .Format({ge::FORMAT_ND});
        config310P.Input("offset")
            .ParamType(DYNAMIC)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND});
        config310P.Input("antiquant_scale")
            .ParamType(DYNAMIC)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND});
        config310P.Input("antiquant_offset")
            .ParamType(DYNAMIC)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND});
        config310P.Input("group_list")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT64})
            .Format({ge::FORMAT_ND});
        config310P.Input("per_token_scale")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND});
        config310P.Output("y")
            .ParamType(DYNAMIC)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND});
        config310P.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("prebuildPattern.value", "Opaque")
            .ExtendCfgInfo("coreType.value", "AiCore")
            .ExtendCfgInfo("aclnnSupport.value", "support_aclnn")
            .ExtendCfgInfo("jitCompile.flag", "static_false,dynamic_false");

        this->AICore().AddConfig("ascend310p", config310P);

        OpAICoreConfig config950;
        config950.Input("x")
            .ParamType(DYNAMIC)
            .DataType({
                ge::DT_BF16,          ge::DT_BF16,          ge::DT_FLOAT16,       ge::DT_FLOAT8_E5M2,
                ge::DT_FLOAT8_E5M2,   ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2,
                ge::DT_FLOAT8_E5M2,   ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2,
                ge::DT_FLOAT8_E5M2,   ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT4_E2M1,
                ge::DT_FLOAT4_E2M1,   ge::DT_FLOAT4_E2M1,   ge::DT_INT8,          ge::DT_INT8,
                ge::DT_FLOAT16,       ge::DT_FLOAT16,       ge::DT_BF16,          ge::DT_BF16,
                ge::DT_FLOAT16,       ge::DT_FLOAT16,       ge::DT_BF16,          ge::DT_BF16,
                ge::DT_INT8,          ge::DT_INT8,          ge::DT_INT8,          ge::DT_INT8,
                ge::DT_INT8,          ge::DT_INT8,          ge::DT_INT8,          ge::DT_INT8,
                ge::DT_INT8,          ge::DT_INT8,          ge::DT_INT8,          ge::DT_INT8,
                ge::DT_INT8,          ge::DT_INT8,          ge::DT_INT8,          ge::DT_INT8,
                ge::DT_INT8,          ge::DT_HIFLOAT8,      ge::DT_HIFLOAT8,      ge::DT_HIFLOAT8,
                ge::DT_HIFLOAT8,      ge::DT_HIFLOAT8,      ge::DT_HIFLOAT8,      ge::DT_HIFLOAT8,
                ge::DT_HIFLOAT8,      ge::DT_HIFLOAT8,      ge::DT_FLOAT8_E5M2,   ge::DT_FLOAT8_E5M2,
                ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2,   ge::DT_FLOAT8_E5M2,
                ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2,   ge::DT_FLOAT8_E5M2,
                ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2,   ge::DT_FLOAT8_E5M2,
                ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2,   ge::DT_FLOAT8_E5M2,
                ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2,   ge::DT_FLOAT8_E5M2,
                ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2,   ge::DT_FLOAT8_E5M2,
                ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2,   ge::DT_FLOAT8_E5M2,
                ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2,   ge::DT_FLOAT8_E5M2,
                ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT16,       ge::DT_FLOAT16,
                ge::DT_FLOAT16,       ge::DT_BF16,          ge::DT_BF16,          ge::DT_BF16,
                ge::DT_BF16,          ge::DT_BF16,          ge::DT_BF16,          ge::DT_FLOAT16,
                ge::DT_FLOAT16,       ge::DT_BF16,          ge::DT_BF16,          ge::DT_BF16,
                ge::DT_BF16,          ge::DT_FLOAT16,       ge::DT_BF16,          ge::DT_BF16,
                ge::DT_FLOAT16,       ge::DT_BF16,          ge::DT_BF16,          ge::DT_FLOAT8_E4M3FN,
                ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN,
                ge::DT_FLOAT8_E4M3FN, ge::DT_INT8,          ge::DT_INT8,          ge::DT_FLOAT16,
                ge::DT_BF16,          ge::DT_BF16,          ge::DT_FLOAT16,       ge::DT_BF16,
                ge::DT_BF16,          ge::DT_INT8,          ge::DT_INT8,          ge::DT_INT8,
                ge::DT_INT8,          ge::DT_INT8,          ge::DT_INT8,          ge::DT_INT8,
                ge::DT_INT8,          ge::DT_INT8,          ge::DT_INT8,          ge::DT_INT8,
                ge::DT_INT8,          ge::DT_INT8,          ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND});
        config950.Input("weight")
            .ParamType(DYNAMIC)
            .DataType({ge::DT_INT8,          ge::DT_INT8,        ge::DT_INT8,          ge::DT_FLOAT8_E5M2,
                       ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2,
                       ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2,
                       ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT4_E2M1,
                       ge::DT_FLOAT4_E2M1,
                       ge::DT_FLOAT4_E2M1,
                       ge::DT_INT4,
                       ge::DT_INT4,          ge::DT_FLOAT16,     ge::DT_FLOAT16,       ge::DT_BF16,
                       ge::DT_BF16,          ge::DT_FLOAT16,     ge::DT_FLOAT16,       ge::DT_BF16,
                       ge::DT_BF16,          ge::DT_INT8,        ge::DT_INT8,          ge::DT_INT8,
                       ge::DT_INT8,          ge::DT_INT8,        ge::DT_INT8,          ge::DT_INT8,
                       ge::DT_INT8,          ge::DT_INT8,        ge::DT_INT8,          ge::DT_INT8,
                       ge::DT_INT8,          ge::DT_INT8,        ge::DT_INT8,          ge::DT_INT8,
                       ge::DT_INT8,          ge::DT_INT8,        ge::DT_HIFLOAT8,      ge::DT_HIFLOAT8,
                       ge::DT_HIFLOAT8,      ge::DT_HIFLOAT8,    ge::DT_HIFLOAT8,      ge::DT_HIFLOAT8,
                       ge::DT_HIFLOAT8,      ge::DT_HIFLOAT8,    ge::DT_HIFLOAT8,      ge::DT_FLOAT8_E5M2,
                       ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2,
                       ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2,
                       ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2,
                       ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2,
                       ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2,
                       ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2,
                       ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2,
                       ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2,
                       ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2,
                       ge::DT_FLOAT8_E4M3FN, ge::DT_HIFLOAT8,    ge::DT_FLOAT8_E5M2,   ge::DT_FLOAT8_E4M3FN,
                       ge::DT_HIFLOAT8,      ge::DT_FLOAT8_E5M2, ge::DT_FLOAT8_E4M3FN, ge::DT_HIFLOAT8,
                       ge::DT_FLOAT4_E2M1,   ge::DT_FLOAT4_E1M2, ge::DT_FLOAT4_E2M1,   ge::DT_FLOAT4_E1M2,
                       ge::DT_FLOAT4_E2M1,   ge::DT_FLOAT4_E1M2, ge::DT_FLOAT4_E2M1,   ge::DT_FLOAT4_E2M1,
                       ge::DT_FLOAT4_E2M1,   ge::DT_FLOAT,       ge::DT_FLOAT,         ge::DT_FLOAT,
                       ge::DT_FLOAT4_E2M1,   ge::DT_FLOAT4_E2M1, ge::DT_FLOAT4_E2M1,   ge::DT_FLOAT4_E2M1,
                       ge::DT_FLOAT,         ge::DT_FLOAT,       ge::DT_INT32,         ge::DT_INT32,
                       ge::DT_INT4,          ge::DT_INT4,        ge::DT_INT4,          ge::DT_INT32,
                       ge::DT_INT32,         ge::DT_INT32,       ge::DT_INT8,          ge::DT_INT8,
                       ge::DT_INT8,          ge::DT_INT8,        ge::DT_INT8,          ge::DT_INT8,
                       ge::DT_INT8,          ge::DT_INT8,        ge::DT_INT8,          ge::DT_INT8,
                       ge::DT_INT8,          ge::DT_INT8,        ge::DT_INT8,          ge::DT_FLOAT})
            .Format({ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_ND,
                     ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_ND,
                     ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_ND,
                     ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_ND,
                     ge::FORMAT_ND,
                     ge::FORMAT_ND,
                     ge::FORMAT_FRACTAL_NZ,
                     ge::FORMAT_FRACTAL_NZ,       ge::FORMAT_ND,               ge::FORMAT_ND,               ge::FORMAT_ND,
                     ge::FORMAT_ND,               ge::FORMAT_FRACTAL_NZ,       ge::FORMAT_FRACTAL_NZ,       ge::FORMAT_FRACTAL_NZ,
                     ge::FORMAT_FRACTAL_NZ,       ge::FORMAT_ND,               ge::FORMAT_ND,               ge::FORMAT_ND,
                     ge::FORMAT_ND,               ge::FORMAT_ND,               ge::FORMAT_ND,               ge::FORMAT_ND,
                     ge::FORMAT_ND,               ge::FORMAT_ND,               ge::FORMAT_ND,               ge::FORMAT_ND,
                     ge::FORMAT_ND,               ge::FORMAT_ND,               ge::FORMAT_ND,               ge::FORMAT_ND,
                     ge::FORMAT_ND,               ge::FORMAT_FRACTAL_NZ,       ge::FORMAT_ND,               ge::FORMAT_ND,
                     ge::FORMAT_ND,               ge::FORMAT_ND,               ge::FORMAT_ND,               ge::FORMAT_ND,
                     ge::FORMAT_ND,               ge::FORMAT_ND,               ge::FORMAT_ND,               ge::FORMAT_ND,
                     ge::FORMAT_ND,               ge::FORMAT_ND,               ge::FORMAT_ND,               ge::FORMAT_ND,
                     ge::FORMAT_ND,               ge::FORMAT_ND,               ge::FORMAT_ND,               ge::FORMAT_ND,
                     ge::FORMAT_ND,               ge::FORMAT_ND,               ge::FORMAT_ND,               ge::FORMAT_ND,
                     ge::FORMAT_ND,               ge::FORMAT_ND,               ge::FORMAT_ND,               ge::FORMAT_ND,
                     ge::FORMAT_ND,               ge::FORMAT_ND,               ge::FORMAT_ND,               ge::FORMAT_ND,
                     ge::FORMAT_ND,               ge::FORMAT_ND,               ge::FORMAT_ND,               ge::FORMAT_ND,
                     ge::FORMAT_ND,               ge::FORMAT_ND,               ge::FORMAT_ND,               ge::FORMAT_ND,
                     ge::FORMAT_ND,               ge::FORMAT_ND,               ge::FORMAT_ND,               ge::FORMAT_ND,
                     ge::FORMAT_ND,               ge::FORMAT_ND,               ge::FORMAT_ND,               ge::FORMAT_ND,
                     ge::FORMAT_ND,               ge::FORMAT_ND,               ge::FORMAT_ND,               ge::FORMAT_ND,
                     ge::FORMAT_ND,               ge::FORMAT_ND,               ge::FORMAT_ND,               ge::FORMAT_ND,
                     ge::FORMAT_FRACTAL_NZ,       ge::FORMAT_FRACTAL_NZ,       ge::FORMAT_FRACTAL_NZ,       ge::FORMAT_FRACTAL_NZ,
                     ge::FORMAT_FRACTAL_NZ,       ge::FORMAT_FRACTAL_NZ,       ge::FORMAT_FRACTAL_NZ_C0_16, ge::FORMAT_FRACTAL_NZ_C0_16,
                     ge::FORMAT_FRACTAL_NZ_C0_16, ge::FORMAT_FRACTAL_NZ_C0_2,  ge::FORMAT_FRACTAL_NZ_C0_2,  ge::FORMAT_FRACTAL_NZ_C0_2,
                     ge::FORMAT_FRACTAL_NZ,       ge::FORMAT_FRACTAL_NZ,       ge::FORMAT_FRACTAL_NZ_C0_32, ge::FORMAT_FRACTAL_NZ_C0_32,
                     ge::FORMAT_FRACTAL_NZ_C0_4,  ge::FORMAT_FRACTAL_NZ_C0_4,  ge::FORMAT_FRACTAL_NZ_C0_4,  ge::FORMAT_FRACTAL_NZ_C0_4,
                     ge::FORMAT_ND,               ge::FORMAT_ND,               ge::FORMAT_ND,               ge::FORMAT_ND,
                     ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ,
                     ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ,
                     ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ,
                     ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND});
        config950.Input("bias")
            .ParamType(DYNAMIC)
            .DataType({ge::DT_FLOAT,   ge::DT_BF16,    ge::DT_FLOAT16, ge::DT_FLOAT,
                       ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_FLOAT,
                       ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_FLOAT,
                       ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_FLOAT,
                       ge::DT_FLOAT,
                       ge::DT_FLOAT,
                       ge::DT_FLOAT,
                       ge::DT_FLOAT,   ge::DT_FLOAT16, ge::DT_FLOAT,   ge::DT_FLOAT,
                       ge::DT_BF16,    ge::DT_FLOAT16, ge::DT_FLOAT,   ge::DT_FLOAT,
                       ge::DT_BF16,    ge::DT_INT32,   ge::DT_INT32,   ge::DT_INT32,
                       ge::DT_INT32,   ge::DT_INT32,   ge::DT_INT32,   ge::DT_BF16,
                       ge::DT_BF16,    ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_INT32,
                       ge::DT_FLOAT,   ge::DT_FLOAT16, ge::DT_INT32,   ge::DT_INT32,
                       ge::DT_INT32,   ge::DT_INT32,   ge::DT_FLOAT,   ge::DT_FLOAT,
                       ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_FLOAT,
                       ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_FLOAT,
                       ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_FLOAT,
                       ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_FLOAT,
                       ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_FLOAT,
                       ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_FLOAT,
                       ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_FLOAT,
                       ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_FLOAT,
                       ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_FLOAT,
                       ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_FLOAT,
                       ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_FLOAT16,
                       ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_BF16,
                       ge::DT_BF16,    ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_FLOAT,
                       ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_BF16,
                       ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_FLOAT16, ge::DT_BF16,
                       ge::DT_FLOAT,   ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_FLOAT,
                       ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_FLOAT16, ge::DT_BF16,
                       ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_FLOAT,   ge::DT_FLOAT,
                       ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_FLOAT,   ge::DT_FLOAT16,
                       ge::DT_BF16,    ge::DT_FLOAT,   ge::DT_INT32,   ge::DT_INT32,
                       ge::DT_INT32,   ge::DT_INT32,   ge::DT_INT32,   ge::DT_INT32,
                       ge::DT_INT32,   ge::DT_BF16,    ge::DT_BF16,    ge::DT_FLOAT,
                       ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND});
        config950.Input("scale")
            .ParamType(DYNAMIC)
            .DataType(
                {ge::DT_UINT64,      ge::DT_UINT64,      ge::DT_UINT64,      ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0,
                 ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0,
                 ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0,
                 ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0,
                 ge::DT_FLOAT8_E8M0,
                 ge::DT_FLOAT,       ge::DT_FLOAT,       ge::DT_UINT64,
                 ge::DT_UINT64,      ge::DT_UINT64,      ge::DT_UINT64,      ge::DT_UINT64,      ge::DT_UINT64,
                 ge::DT_UINT64,      ge::DT_UINT64,      ge::DT_UINT64,      ge::DT_INT64,       ge::DT_UINT64,
                 ge::DT_INT64,       ge::DT_BF16,        ge::DT_FLOAT,       ge::DT_BF16,        ge::DT_FLOAT,
                 ge::DT_BF16,        ge::DT_FLOAT,       ge::DT_FLOAT,       ge::DT_FLOAT,       ge::DT_FLOAT,
                 ge::DT_UINT64,      ge::DT_INT64,       ge::DT_UINT64,      ge::DT_UINT64,      ge::DT_UINT64,
                 ge::DT_INT64,       ge::DT_UINT64,      ge::DT_INT64,       ge::DT_UINT64,      ge::DT_INT64,
                 ge::DT_FLOAT,       ge::DT_FLOAT,       ge::DT_FLOAT,       ge::DT_INT64,       ge::DT_INT64,
                 ge::DT_INT64,       ge::DT_INT64,       ge::DT_UINT64,      ge::DT_UINT64,      ge::DT_UINT64,
                 ge::DT_UINT64,      ge::DT_INT64,       ge::DT_INT64,       ge::DT_INT64,       ge::DT_INT64,
                 ge::DT_UINT64,      ge::DT_UINT64,      ge::DT_UINT64,      ge::DT_UINT64,      ge::DT_INT64,
                 ge::DT_INT64,       ge::DT_INT64,       ge::DT_INT64,       ge::DT_UINT64,      ge::DT_UINT64,
                 ge::DT_UINT64,      ge::DT_UINT64,      ge::DT_FLOAT,       ge::DT_FLOAT,       ge::DT_FLOAT,
                 ge::DT_FLOAT,       ge::DT_FLOAT,       ge::DT_FLOAT,       ge::DT_FLOAT,       ge::DT_FLOAT,
                 ge::DT_FLOAT,       ge::DT_FLOAT,       ge::DT_FLOAT,       ge::DT_FLOAT,       ge::DT_UINT64,
                 ge::DT_UINT64,      ge::DT_UINT64,      ge::DT_UINT64,      ge::DT_UINT64,      ge::DT_UINT64,
                 ge::DT_UINT64,      ge::DT_UINT64,      ge::DT_UINT64,      ge::DT_UINT64,      ge::DT_UINT64,
                 ge::DT_UINT64,      ge::DT_UINT64,      ge::DT_UINT64,      ge::DT_UINT64,      ge::DT_UINT64,
                 ge::DT_UINT64,      ge::DT_UINT64,      ge::DT_UINT64,      ge::DT_UINT64,      ge::DT_UINT64,
                 ge::DT_UINT64,      ge::DT_UINT64,      ge::DT_UINT64,      ge::DT_UINT64,      ge::DT_UINT64,
                 ge::DT_UINT64,      ge::DT_FLOAT,       ge::DT_FLOAT,       ge::DT_UINT64,      ge::DT_UINT64,
                 ge::DT_UINT64,      ge::DT_UINT64,      ge::DT_UINT64,      ge::DT_UINT64,      ge::DT_UINT64,
                 ge::DT_INT64,       ge::DT_UINT64,      ge::DT_INT64,       ge::DT_BF16,        ge::DT_FLOAT,
                 ge::DT_FLOAT,       ge::DT_BF16,        ge::DT_FLOAT,       ge::DT_BF16,        ge::DT_FLOAT,
                 ge::DT_FLOAT,       ge::DT_FLOAT,       ge::DT_UINT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND});
        config950.Input("offset")
            .ParamType(DYNAMIC)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND});
        config950.Input("antiquant_scale")
            .ParamType(DYNAMIC)
            .DataType(
                {ge::DT_BF16,        ge::DT_BF16,        ge::DT_FLOAT16,     ge::DT_FLOAT16,     ge::DT_FLOAT16,
                 ge::DT_FLOAT16,     ge::DT_FLOAT16,     ge::DT_FLOAT16,     ge::DT_FLOAT16,     ge::DT_FLOAT16,
                 ge::DT_FLOAT16,     ge::DT_FLOAT16,     ge::DT_FLOAT16,     ge::DT_FLOAT16,     ge::DT_FLOAT16,
                 ge::DT_FLOAT16,     ge::DT_FLOAT16,     ge::DT_FLOAT16,     ge::DT_FLOAT16,     ge::DT_FLOAT16,
                 ge::DT_FLOAT16,     ge::DT_FLOAT16,     ge::DT_FLOAT16,     ge::DT_FLOAT16,     ge::DT_FLOAT16,
                 ge::DT_FLOAT16,     ge::DT_FLOAT16,     ge::DT_FLOAT16,     ge::DT_FLOAT16,     ge::DT_FLOAT16,
                 ge::DT_FLOAT16,     ge::DT_FLOAT16,     ge::DT_FLOAT16,     ge::DT_FLOAT16,     ge::DT_FLOAT16,
                 ge::DT_FLOAT16,     ge::DT_FLOAT16,     ge::DT_FLOAT16,     ge::DT_FLOAT16,     ge::DT_FLOAT16,
                 ge::DT_FLOAT16,     ge::DT_FLOAT16,     ge::DT_FLOAT16,     ge::DT_FLOAT16,     ge::DT_FLOAT16,
                 ge::DT_FLOAT16,     ge::DT_FLOAT16,     ge::DT_FLOAT16,     ge::DT_FLOAT16,     ge::DT_FLOAT16,
                 ge::DT_FLOAT16,     ge::DT_FLOAT16,     ge::DT_FLOAT16,     ge::DT_FLOAT16,     ge::DT_FLOAT16,
                 ge::DT_FLOAT16,     ge::DT_FLOAT16,     ge::DT_FLOAT16,     ge::DT_FLOAT16,     ge::DT_FLOAT16,
                 ge::DT_FLOAT16,     ge::DT_FLOAT16,     ge::DT_FLOAT16,     ge::DT_FLOAT16,     ge::DT_FLOAT16,
                 ge::DT_FLOAT16,     ge::DT_FLOAT16,     ge::DT_FLOAT16,     ge::DT_FLOAT16,     ge::DT_FLOAT16,
                 ge::DT_FLOAT16,     ge::DT_FLOAT16,     ge::DT_FLOAT16,     ge::DT_FLOAT16,     ge::DT_FLOAT16,
                 ge::DT_FLOAT16,     ge::DT_FLOAT16,     ge::DT_FLOAT16,     ge::DT_FLOAT16,     ge::DT_FLOAT16,
                 ge::DT_FLOAT16,     ge::DT_FLOAT16,     ge::DT_FLOAT16,     ge::DT_FLOAT16,     ge::DT_FLOAT16,
                 ge::DT_FLOAT16,     ge::DT_FLOAT16,     ge::DT_FLOAT16,     ge::DT_FLOAT16,     ge::DT_FLOAT16,
                 ge::DT_FLOAT16,     ge::DT_FLOAT16,     ge::DT_FLOAT16,     ge::DT_BF16,        ge::DT_BF16,
                 ge::DT_BF16,        ge::DT_BF16,        ge::DT_BF16,        ge::DT_BF16,        ge::DT_FLOAT8_E8M0,
                 ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0,
                 ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0,
                 ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0,
                 ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT16,     ge::DT_FLOAT16,     ge::DT_FLOAT16,
                 ge::DT_BF16,        ge::DT_BF16,        ge::DT_FLOAT16,     ge::DT_BF16,        ge::DT_BF16,
                 ge::DT_FLOAT16,     ge::DT_FLOAT16,     ge::DT_FLOAT16,     ge::DT_FLOAT16,     ge::DT_FLOAT16,
                 ge::DT_FLOAT16,     ge::DT_FLOAT16,     ge::DT_FLOAT16,     ge::DT_FLOAT16,     ge::DT_FLOAT16,
                 ge::DT_FLOAT16,     ge::DT_FLOAT16,     ge::DT_FLOAT16,     ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND});
        config950.Input("antiquant_offset")
            .ParamType(DYNAMIC)
            .DataType({ge::DT_BF16,    ge::DT_BF16,    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
                       ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
                       ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
                       ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
                       ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
                       ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
                       ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
                       ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
                       ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
                       ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
                       ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
                       ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
                       ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
                       ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
                       ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
                       ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_BF16,    ge::DT_BF16,
                       ge::DT_BF16,    ge::DT_BF16,    ge::DT_BF16,    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16,
                       ge::DT_BF16,    ge::DT_BF16,    ge::DT_BF16,    ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_BF16,
                       ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_BF16,    ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_FLOAT16,
                       ge::DT_BF16,    ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
                       ge::DT_BF16,    ge::DT_BF16,    ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_BF16,    ge::DT_FLOAT16,
                       ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
                       ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
                       ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND});
        config950.Input("group_list")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64,
                       ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64,
                       ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64,
                       ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64,
                       ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64,
                       ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64,
                       ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64,
                       ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64,
                       ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64,
                       ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64,
                       ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64,
                       ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64,
                       ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64,
                       ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64,
                       ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64,
                       ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64,
                       ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64,
                       ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64,
                       ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64,
                       ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND});
        config950.Input("per_token_scale")
            .ParamType(OPTIONAL)
            .DataType(
                {ge::DT_FLOAT,       ge::DT_FLOAT,       ge::DT_FLOAT,       ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0,
                 ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0,
                 ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0,
                 ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT,       ge::DT_FLOAT,
                 ge::DT_FLOAT,       ge::DT_FLOAT,       ge::DT_FLOAT,       ge::DT_FLOAT,       ge::DT_FLOAT,
                 ge::DT_FLOAT,       ge::DT_FLOAT,       ge::DT_FLOAT,       ge::DT_FLOAT,       ge::DT_FLOAT,
                 ge::DT_FLOAT,       ge::DT_FLOAT,       ge::DT_FLOAT,       ge::DT_FLOAT,       ge::DT_FLOAT,
                 ge::DT_FLOAT,       ge::DT_FLOAT,       ge::DT_FLOAT,       ge::DT_FLOAT,       ge::DT_FLOAT,
                 ge::DT_FLOAT,       ge::DT_FLOAT,       ge::DT_FLOAT,       ge::DT_FLOAT,       ge::DT_FLOAT,
                 ge::DT_FLOAT,       ge::DT_FLOAT,       ge::DT_FLOAT,       ge::DT_FLOAT,       ge::DT_FLOAT,
                 ge::DT_FLOAT,       ge::DT_FLOAT,       ge::DT_FLOAT,       ge::DT_FLOAT,       ge::DT_FLOAT,
                 ge::DT_FLOAT,       ge::DT_FLOAT,       ge::DT_FLOAT,       ge::DT_FLOAT,       ge::DT_FLOAT,
                 ge::DT_FLOAT,       ge::DT_FLOAT,       ge::DT_FLOAT,       ge::DT_FLOAT,       ge::DT_FLOAT,
                 ge::DT_FLOAT,       ge::DT_FLOAT,       ge::DT_FLOAT,       ge::DT_FLOAT,       ge::DT_FLOAT,
                 ge::DT_FLOAT,       ge::DT_FLOAT,       ge::DT_FLOAT,       ge::DT_FLOAT,       ge::DT_FLOAT,
                 ge::DT_FLOAT,       ge::DT_FLOAT,       ge::DT_FLOAT,       ge::DT_FLOAT,       ge::DT_FLOAT,
                 ge::DT_FLOAT,       ge::DT_FLOAT,       ge::DT_FLOAT,       ge::DT_FLOAT,       ge::DT_FLOAT,
                 ge::DT_FLOAT,       ge::DT_FLOAT,       ge::DT_FLOAT,       ge::DT_FLOAT,       ge::DT_FLOAT,
                 ge::DT_FLOAT,       ge::DT_FLOAT,       ge::DT_FLOAT,       ge::DT_FLOAT,       ge::DT_FLOAT,
                 ge::DT_FLOAT,       ge::DT_FLOAT,       ge::DT_FLOAT,       ge::DT_FLOAT,       ge::DT_FLOAT,
                 ge::DT_FLOAT,       ge::DT_FLOAT,       ge::DT_FLOAT,       ge::DT_FLOAT,       ge::DT_FLOAT,
                 ge::DT_FLOAT,       ge::DT_FLOAT,       ge::DT_FLOAT,       ge::DT_FLOAT,       ge::DT_FLOAT,
                 ge::DT_FLOAT,       ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0,
                 ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT,       ge::DT_FLOAT,       ge::DT_FLOAT,
                 ge::DT_FLOAT,       ge::DT_FLOAT,       ge::DT_FLOAT,       ge::DT_FLOAT,       ge::DT_FLOAT,
                 ge::DT_FLOAT,       ge::DT_FLOAT,       ge::DT_FLOAT,       ge::DT_FLOAT,       ge::DT_FLOAT,
                 ge::DT_FLOAT,       ge::DT_FLOAT,       ge::DT_FLOAT,       ge::DT_FLOAT,       ge::DT_FLOAT,
                 ge::DT_FLOAT,       ge::DT_FLOAT,       ge::DT_FLOAT,       ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND});
        config950.Output("y")
            .ParamType(DYNAMIC)
            .DataType({ge::DT_BF16,    ge::DT_BF16,    ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_BF16,    ge::DT_BF16,
                       ge::DT_BF16,    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT,
                       ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_BF16,
                       ge::DT_FLOAT16, ge::DT_FLOAT,
                       ge::DT_BF16,    ge::DT_FLOAT16, ge::DT_FLOAT16,
                       ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_BF16,    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16,
                       ge::DT_BF16,    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_BF16,    ge::DT_BF16,
                       ge::DT_BF16,    ge::DT_BF16,    ge::DT_BF16,    ge::DT_BF16,    ge::DT_BF16,    ge::DT_FLOAT16,
                       ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT32,   ge::DT_INT32,
                       ge::DT_BF16,    ge::DT_BF16,    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT,   ge::DT_FLOAT,
                       ge::DT_BF16,    ge::DT_FLOAT16, ge::DT_FLOAT,   ge::DT_BF16,    ge::DT_BF16,    ge::DT_BF16,
                       ge::DT_BF16,    ge::DT_BF16,    ge::DT_BF16,    ge::DT_BF16,    ge::DT_BF16,    ge::DT_FLOAT16,
                       ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
                       ge::DT_FLOAT16, ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_FLOAT,
                       ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_BF16,    ge::DT_BF16,    ge::DT_BF16,
                       ge::DT_BF16,    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT,
                       ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
                       ge::DT_BF16,    ge::DT_BF16,    ge::DT_BF16,    ge::DT_BF16,    ge::DT_BF16,    ge::DT_BF16,
                       ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_BF16,    ge::DT_BF16,    ge::DT_BF16,
                       ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_BF16,    ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_BF16,
                       ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_FLOAT16, ge::DT_BF16,
                       ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_BF16,    ge::DT_FLOAT16,
                       ge::DT_BF16,    ge::DT_BF16,    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_BF16,
                       ge::DT_BF16,    ge::DT_BF16,    ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_BF16,    ge::DT_BF16,
                       ge::DT_BF16,    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND});
        config950.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("prebuildPattern.value", "Opaque")
            .ExtendCfgInfo("coreType.value", "AiCore")
            .ExtendCfgInfo("aclnnSupport.value", "support_aclnn")
            .ExtendCfgInfo("opFile.value","grouped_matmul_apt");
        this->AICore().AddConfig("ascend950", config950);

        OpAICoreConfig config_kirin = GetKirinCoreConfig();
        this->AICore().AddConfig("kirinx90", config_kirin);
        this->AICore().AddConfig("kirin9030", config_kirin);
    }

private:
    OpAICoreConfig GetKirinCoreConfig() const
    {
        OpAICoreConfig config_kirin;
        config_kirin.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true);
        config_kirin.Input("x")
            .ParamType(DYNAMIC)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND});
        config_kirin.Input("weight")
            .ParamType(DYNAMIC)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_FRACTAL_NZ});
        config_kirin.Input("bias")
            .ParamType(DYNAMIC)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND});
        config_kirin.Input("scale")
            .ParamType(DYNAMIC)
            .DataType({ge::DT_UINT64})
            .Format({ge::FORMAT_ND});
        config_kirin.Input("offset")
            .ParamType(DYNAMIC)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND});
        config_kirin.Input("antiquant_scale")
            .ParamType(DYNAMIC)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND});
        config_kirin.Input("antiquant_offset")
            .ParamType(DYNAMIC)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND});
        config_kirin.Input("group_list")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT64})
            .Format({ge::FORMAT_ND});
        config_kirin.Input("per_token_scale")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND});
        config_kirin.Output("y")
            .ParamType(DYNAMIC)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND});
        return config_kirin;
    }
};

OP_ADD(GroupedMatmul);
} // namespace ops
