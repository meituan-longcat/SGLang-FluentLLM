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
 * \file gen_from_cache_indices_def.cpp
 * \brief GenFromCacheIndices op host config
 *
 * 语义（因果筛选 + actual_seq 链）：
 *   对每个 (b,s)：由 full_q_actual_seq/full_kv_actual_seq 算 curKvModify（MTP 因果可见 KV 长度）
 *   与 maxSelectionId = CeilDiv(curKvModify, block_size) - 1；逐 topk id：
 *     - 越界/因果不可见（id<0 或 id>maxSelectionId，或整行 curKvModify<=0）→ from_cache_indices = -2
 *     - 有效且命中缓存池 → slot∈[0,CACHE)
 *     - 有效但未命中          → -1
 *   同时累加 selection_kv_actual_seq = Σ 有效 id 的 gatherBlockSize（尾块按 lastGatherBlockSize）。
 */

#include <cstdint>
#include "register/op_def_registry.h"

namespace ops {
class GenFromCacheIndices : public OpDef {
public:
    explicit GenFromCacheIndices(const char *name) : OpDef(name)
    {
        this->Input("selection_topk_indices")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("cache_status")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("full_kv_actual_seq")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("full_q_actual_seq")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        // full_kv_block_table [B, fullMaxBlockNum]：miss 时把逻辑 token 页翻译到 full 物理块
        // （enable_topk_indices_opt==2 时用于 unified_slots_in_lru_or_cache 的 miss 偏移计算）
        this->Input("full_kv_block_table")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("from_cache_indices")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("selection_kv_actual_seq")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("write_slot")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        // unified_slots_in_lru_or_cache [B,S,TOPK]（int32）：
        //   >0 = hit（lruRow+1）；<0 = miss（-(fullOffset+1)）；0 = skip/当前 token
        //   仅 enable_topk_indices_opt==2 时输出有效内容，其他模式为全 0。
        this->Output("unified_slots_in_lru_or_cache")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Attr("selection_topk_block_size").AttrType(OPTIONAL).Int(1);
        this->Attr("enable_topk_indices_opt").AttrType(OPTIONAL).Int(0);
        // full_kv_block_size：full cache 物理块长（通常=128），opt==2 时用于 page/poff 计算
        this->Attr("full_kv_block_size").AttrType(OPTIONAL).Int(128);

        this->AICore()
            .AddConfig("ascend910_93")
            .AddConfig("ascend910b");
    }
};
OP_ADD(GenFromCacheIndices);
} // namespace ops
