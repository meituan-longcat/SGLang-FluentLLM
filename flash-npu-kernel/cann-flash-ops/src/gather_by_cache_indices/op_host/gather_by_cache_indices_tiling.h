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
 * \file gather_by_cache_indices_tiling.h
 * \brief
 */
#ifndef GATHER_BY_CACHE_INDICES_TILING_H_
#define GATHER_BY_CACHE_INDICES_TILING_H_

#include "exe_graph/runtime/tiling_context.h"
#include "tiling/platform/platform_ascendc.h"
#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "error/ops_error.h"
#include "platform/platform_info.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(GatherByCacheIndicesTilingData)
TILING_DATA_FIELD_DEF(uint32_t, blockNum);
TILING_DATA_FIELD_DEF(uint32_t, totalUnits);
TILING_DATA_FIELD_DEF(uint32_t, numPerCore);
TILING_DATA_FIELD_DEF(uint32_t, tailUnits);
TILING_DATA_FIELD_DEF(uint32_t, kRopeDim);
TILING_DATA_FIELD_DEF(uint32_t, kvDim);
TILING_DATA_FIELD_DEF(uint32_t, unitsPerBatch);
TILING_DATA_FIELD_DEF(uint32_t, cacheRows);
TILING_DATA_FIELD_DEF(uint32_t, unitsPerRealBatch);
// 分页（两级页表）新增
TILING_DATA_FIELD_DEF(uint32_t, seq);              // S
TILING_DATA_FIELD_DEF(uint32_t, topk);             // TOPK
TILING_DATA_FIELD_DEF(uint32_t, selTopKBlockSize); // 每 topk id token 数（本用例=1）
TILING_DATA_FIELD_DEF(uint32_t, fullKvBlockSize);  // full 物理块长（=128）
TILING_DATA_FIELD_DEF(uint32_t, fullMaxBlockNum);  // full_kv_block_table 列数
TILING_DATA_FIELD_DEF(uint32_t, fullKvBlockNum);   // full 物理块数（越界断言）
TILING_DATA_FIELD_DEF(uint32_t, selKvBlockSize);   // selection 物理块长（=128）
TILING_DATA_FIELD_DEF(uint32_t, selMaxBlockNum);   // selection_kv_block_table 列数
TILING_DATA_FIELD_DEF(uint32_t, enableSlotsContiguous); // 1=page id 连续，走线性偏移优化
TILING_DATA_FIELD_DEF(uint32_t, reserved);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(GatherByCacheIndices, GatherByCacheIndicesTilingData)

struct GatherByCacheIndicesCompileInfo {
};

class GatherByCacheIndicesTiling {
public:
    explicit GatherByCacheIndicesTiling(gert::TilingContext* context) : context_(context)
    {}
    ~GatherByCacheIndicesTiling()
    {}
    ge::graphStatus RunTiling();

protected:
    ge::graphStatus GetPlatformInfo();
    ge::graphStatus GetShapeAttrsInfo();
    ge::graphStatus DoOpTiling();
    ge::graphStatus PostTiling();

private:
    GatherByCacheIndicesTilingData tilingData_;

    int64_t coreNum_ = 0;
    int64_t ubSize_ = 0;

    uint32_t batch_ = 0;
    uint32_t cacheRows_ = 0;
    uint32_t kRopeDim_ = 0;
    uint32_t kvDim_ = 0;
    uint32_t totalUnits_ = 0;
    // 分页新增
    uint32_t seq_ = 0;
    uint32_t topk_ = 0;
    uint32_t selTopKBlockSize_ = 1;
    uint32_t fullKvBlockSize_ = 0;
    uint32_t fullMaxBlockNum_ = 0;
    uint32_t fullKvBlockNum_ = 0;
    uint32_t selKvBlockSize_ = 128;
    uint32_t selMaxBlockNum_ = 0;
    uint32_t enableSlotsContiguous_ = 0;

    gert::TilingContext *context_ = nullptr;
};

} // namespace optiling

#endif // GATHER_BY_CACHE_INDICES_TILING_H_
