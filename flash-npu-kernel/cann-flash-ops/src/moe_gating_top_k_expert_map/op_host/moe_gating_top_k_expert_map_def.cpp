/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_gating_top_k_expert_map_def.cpp
 * \brief
 */
#include "register/op_def_registry.h"

namespace ops {
class MoeGatingTopKExpertMap : public OpDef {
public:
    explicit MoeGatingTopKExpertMap(const char *name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("bias")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("expert_map")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("expert_idx")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("k").Int();
        this->Attr("k_group").AttrType(OPTIONAL).Int(1);
        this->Attr("group_count").AttrType(OPTIONAL).Int(1);
        this->Attr("group_select_mode").AttrType(OPTIONAL).Int(0);
        this->Attr("renorm").AttrType(OPTIONAL).Int(0);
        this->Attr("norm_type").AttrType(OPTIONAL).Int(0);
        this->Attr("out_flag").AttrType(OPTIONAL).Bool(false);
        this->Attr("routed_scaling_factor").AttrType(OPTIONAL).Float(1.0);
        this->Attr("eps").AttrType(OPTIONAL).Float(1e-20f);
        OpAICoreConfig config;
        config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true);
        this->AICore().AddConfig("ascend910b", config);
        this->AICore().AddConfig("ascend910_93", config);
    }
};

OP_ADD(MoeGatingTopKExpertMap);
} // namespace ops
