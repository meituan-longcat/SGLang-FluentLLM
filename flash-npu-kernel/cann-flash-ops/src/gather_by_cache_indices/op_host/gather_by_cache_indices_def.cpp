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
 * \file gather_by_cache_indices_def.cpp
 * \brief GatherByCacheIndices op host config
 */

#include <cstdint>
#include "register/op_def_registry.h"

namespace ops {
class GatherByCacheIndices : public OpDef {
public:
    explicit GatherByCacheIndices(const char *name) : OpDef(name)
    {
        this->Input("cache_k_rope")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("cache_kv_cache")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("from_cache_indices")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("selection_topk_indices")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("full_k_rope")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("full_kv_cache")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        // full_kv_block_table [B, fullMaxBlockNum]：miss 时把逻辑 token 页翻译到 full 物理块（原算子同款分页）
        this->Input("full_kv_block_table")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        // selection_kv_block_table [B*S, selMaxBlockNum]：把 (b,s) 内 raw topk 位置翻译到 selection 输出物理块
        this->Input("selection_kv_block_table")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("write_slot")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        // actual_seq_lengths_kv [B]：每 batch 完整 KV 长度 cur_end；actual_seq_qlen [B]：当前 MTP token 数
        this->Input("actual_seq_lengths_kv")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("actual_seq_qlen")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("selection_k_rope")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("selection_kv_cache")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        // cur_kv_write_slots [B, S, S]（int32，初始化为 -1）：
        //   当前 MTP token 的扁平物理槽登记（selection KV 内 physical_page*PAGE_SIZE + page_offset），
        //   与 write_slot（历史 miss 写入持久化 cache pool 的位置）概念不同。
        this->Output("cur_kv_write_slots")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});

        this->Attr("selection_topk_block_size").AttrType(OPTIONAL).Int(1);   // 每 topk id 的 token 数
        this->Attr("selection_kv_block_num").AttrType(REQUIRED).Int(0);       // 输出 selection 缓冲块数 S_BLOCK_NUM
        this->Attr("selection_kv_block_size").AttrType(OPTIONAL).Int(128);    // 输出 selection 块长
        this->Attr("enable_slots_contiguous").AttrType(OPTIONAL).Int(0);      // 1=page id 连续，走线性偏移优化

        this->AICore()
            .AddConfig("ascend910_93")
            .AddConfig("ascend910b");
    }
};
OP_ADD(GatherByCacheIndices);
} // namespace ops
