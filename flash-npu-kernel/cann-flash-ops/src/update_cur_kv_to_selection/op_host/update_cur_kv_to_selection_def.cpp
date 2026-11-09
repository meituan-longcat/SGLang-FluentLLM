/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file update_cur_kv_to_selection_def.cpp
 * \brief UpdateCurKvToSelection op host config
 *        把当前请求新产生的 cur_k_nope/cur_k_rope 按 cur_kv_write_slots 槽位就地
 *        覆盖写入 selection 分页缓存（scatter 覆盖写，纯数据搬运，bit-exact）。
 *        selection_k_nope / selection_k_rope 为原地语义（Input/Output 同名）。
 *        kernel_done[1] int32 为图返回值标记，kernel 不修改其值。
 */

#include <cstdint>
#include "register/op_def_registry.h"

namespace ops {
class UpdateCurKvToSelection : public OpDef {
public:
    explicit UpdateCurKvToSelection(const char *name) : OpDef(name)
    {
        this->Input("cur_k_nope")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("cur_k_rope")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("cur_kv_write_slots")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("selection_k_nope")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("selection_k_rope")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("selection_kv_block_table")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        // 原地输出：与输入同名，GE 侧识别为 inplace（地址同一，kernel 只写不读）
        this->Output("selection_k_nope")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("selection_k_rope")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        // 图返回值标记：shape 固定 [1]，kernel 不修改其值
        this->Output("kernel_done")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->AICore()
            .AddConfig("ascend910_93")
            .AddConfig("ascend910b");
    }
};
OP_ADD(UpdateCurKvToSelection);
} // namespace ops
