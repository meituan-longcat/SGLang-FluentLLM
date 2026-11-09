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
 * \file update_cur_kv_to_selection_tiling.h
 * \brief UpdateCurKvToSelection TilingData 结构体
 *        注册机制说明：CANN 8.5.0 工具链 tilingdata_base.h 的 REGISTER_TILING_DATA_CLASS
 *        仅支持继承 TilingDef 的类（BEGIN_TILING_DATA_DEF 生成），本仓全部算子同款；
 *        结构包含 16 个字段，reserved 承载单缓冲退化标志，见 tiling.cpp。
 */
#ifndef UPDATE_CUR_KV_TO_SELECTION_TILING_H_
#define UPDATE_CUR_KV_TO_SELECTION_TILING_H_

#include "exe_graph/runtime/tiling_context.h"
#include "tiling/platform/platform_ascendc.h"
#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "error/ops_error.h"
#include "platform/platform_info.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(UpdateCurKvToSelectionTilingData)
TILING_DATA_FIELD_DEF(uint32_t, blockNum);        // 实际使用核数（= usedCore）
TILING_DATA_FIELD_DEF(uint32_t, totalUnits);      // B*S*S
TILING_DATA_FIELD_DEF(uint32_t, numPerCore);      // 每核基础单元数
TILING_DATA_FIELD_DEF(uint32_t, tailUnits);       // 余数核数（前 tailUnits 核多处理 1 个单元）
TILING_DATA_FIELD_DEF(uint32_t, batch);           // B
TILING_DATA_FIELD_DEF(uint32_t, seq);             // S（slots 的 s/j 两维）
TILING_DATA_FIELD_DEF(uint32_t, kNopeDim);        // cur_k_nope / selection_k_nope 末维
TILING_DATA_FIELD_DEF(uint32_t, kRopeDim);        // cur_k_rope / selection_k_rope 末维
TILING_DATA_FIELD_DEF(uint32_t, unitsPerBatch);   // P：单次 UB 批处理的单元数（P 自适应）
TILING_DATA_FIELD_DEF(uint32_t, maxPagePerReq);   // MAX_PAGE_PER_REQ：selection_kv_block_table 列数
TILING_DATA_FIELD_DEF(uint32_t, totalPageNum);    // TOTAL_PAGE_NUM：selection 物理页总数
TILING_DATA_FIELD_DEF(uint32_t, nopeRowPitchElems); // UB 侧 nope 行 pitch（元素数，= ceil(kNopeDim/16)*16）
TILING_DATA_FIELD_DEF(uint32_t, ropeRowPitchElems); // UB 侧 rope 行 pitch（元素数，= ceil(kRopeDim/16)*16）
TILING_DATA_FIELD_DEF(uint32_t, rowAlignedNope);  // kNopeDim*2 % 32 == 0 → 可连续段合并（nope）
TILING_DATA_FIELD_DEF(uint32_t, rowAlignedRope);  // kRopeDim*2 % 32 == 0 → 可连续段合并（rope）
TILING_DATA_FIELD_DEF(uint32_t, reserved);        // 0=数据双缓冲；1=单缓冲退化（P_ub<1 时）
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(UpdateCurKvToSelection, UpdateCurKvToSelectionTilingData)

struct UpdateCurKvToSelectionCompileInfo {
};

class UpdateCurKvToSelectionTiling {
public:
    explicit UpdateCurKvToSelectionTiling(gert::TilingContext* context) : context_(context)
    {}
    ~UpdateCurKvToSelectionTiling()
    {}
    ge::graphStatus RunTiling();

protected:
    ge::graphStatus GetPlatformInfo();
    ge::graphStatus GetShapeAttrsInfo();
    ge::graphStatus DoOpTiling();
    ge::graphStatus PostTiling();

private:
    UpdateCurKvToSelectionTilingData tilingData_;

    int64_t coreNum_ = 0;
    int64_t ubSize_ = 0;

    uint32_t batch_ = 0;
    uint32_t seq_ = 0;
    uint32_t kNopeDim_ = 0;
    uint32_t kRopeDim_ = 0;
    uint32_t totalUnits_ = 0;
    uint32_t maxPagePerReq_ = 0;
    uint32_t totalPageNum_ = 0;

    gert::TilingContext *context_ = nullptr;
};

} // namespace optiling

#endif // UPDATE_CUR_KV_TO_SELECTION_TILING_H_
