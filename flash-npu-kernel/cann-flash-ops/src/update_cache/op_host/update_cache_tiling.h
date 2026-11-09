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
 * \file update_cache_tiling.h
 * \brief
 */
#ifndef UPDATE_CACHE_TILING_H_
#define UPDATE_CACHE_TILING_H_

#include "exe_graph/runtime/tiling_context.h"
#include "tiling/platform/platform_ascendc.h"
#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "error/ops_error.h"
#include "platform/platform_info.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(UpdateCacheTilingData)
TILING_DATA_FIELD_DEF(uint32_t, blockNum);
TILING_DATA_FIELD_DEF(uint32_t, B);
TILING_DATA_FIELD_DEF(uint32_t, S);
TILING_DATA_FIELD_DEF(uint32_t, topk);
TILING_DATA_FIELD_DEF(uint32_t, cache);
TILING_DATA_FIELD_DEF(uint32_t, kRopeDim);
TILING_DATA_FIELD_DEF(uint32_t, kvDim);
TILING_DATA_FIELD_DEF(uint32_t, rowsPerBatch);
// 分页：selection(NEW 源) 物理块寻址
TILING_DATA_FIELD_DEF(uint32_t, selKvBlockSize);   // selection 物理块长（=128）
TILING_DATA_FIELD_DEF(uint32_t, selMaxBlockNum);   // selection_kv_block_table 列数
TILING_DATA_FIELD_DEF(uint32_t, reserved);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(UpdateCache, UpdateCacheTilingData)

struct UpdateCacheCompileInfo {
};

class UpdateCacheTiling {
public:
    explicit UpdateCacheTiling(gert::TilingContext* context) : context_(context)
    {}
    ~UpdateCacheTiling()
    {}
    ge::graphStatus RunTiling();

protected:
    ge::graphStatus GetPlatformInfo();
    ge::graphStatus GetShapeAttrsInfo();
    ge::graphStatus DoOpTiling();
    ge::graphStatus PostTiling();

private:
    UpdateCacheTilingData tilingData_;

    int64_t coreNum_ = 0;
    int64_t ubSize_ = 0;

    uint32_t B_ = 0;
    uint32_t S_ = 0;
    uint32_t topk_ = 0;
    uint32_t cache_ = 0;
    uint32_t kRopeDim_ = 0;
    uint32_t kvDim_ = 0;
    int64_t dtypeSize_ = 2;
    uint32_t selKvBlockSize_ = 128;
    uint32_t selMaxBlockNum_ = 0;

    gert::TilingContext *context_ = nullptr;
};

} // namespace optiling

#endif // UPDATE_CACHE_TILING_H_
