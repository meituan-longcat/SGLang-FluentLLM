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
 * \file gen_from_cache_indices_tiling.h
 * \brief
 */
#ifndef GEN_FROM_CACHE_INDICES_TILING_H_
#define GEN_FROM_CACHE_INDICES_TILING_H_

#include "exe_graph/runtime/tiling_context.h"
#include "tiling/platform/platform_ascendc.h"
#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "error/ops_error.h"
#include "platform/platform_info.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(GenFromCacheIndicesTilingData)
TILING_DATA_FIELD_DEF(uint32_t, blockNum);
TILING_DATA_FIELD_DEF(uint32_t, B);
TILING_DATA_FIELD_DEF(uint32_t, S);
TILING_DATA_FIELD_DEF(uint32_t, topk);
TILING_DATA_FIELD_DEF(uint32_t, cache);
TILING_DATA_FIELD_DEF(uint32_t, blockSize);   // selection_topk_block_size（因果块长）
TILING_DATA_FIELD_DEF(uint32_t, enableTopkIndicesOpt);
TILING_DATA_FIELD_DEF(uint32_t, fullKvBlockSize);   // full cache 物理块长（opt==2 时用于 miss 偏移）
TILING_DATA_FIELD_DEF(uint32_t, fullMaxBlockNum);    // full_kv_block_table 列数
TILING_DATA_FIELD_DEF(uint32_t, reserved1);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(GenFromCacheIndices, GenFromCacheIndicesTilingData)

struct GenFromCacheIndicesCompileInfo {
};

class GenFromCacheIndicesTiling {
public:
    explicit GenFromCacheIndicesTiling(gert::TilingContext* context) : context_(context)
    {}
    ~GenFromCacheIndicesTiling()
    {}
    ge::graphStatus RunTiling();

protected:
    ge::graphStatus GetPlatformInfo();
    ge::graphStatus GetShapeAttrsInfo();
    ge::graphStatus DoOpTiling();
    ge::graphStatus PostTiling();

private:
    GenFromCacheIndicesTilingData tilingData_;

    int64_t coreNum_ = 0;
    int64_t ubSize_ = 0;

    uint32_t B_ = 0;
    uint32_t S_ = 0;
    uint32_t topk_ = 0;
    uint32_t cache_ = 0;
    uint32_t blockSize_ = 1;
    uint32_t enableOpt_ = 0;      // 原值直传（0/1/2），不截断为 bool
    uint32_t fullKvBlockSize_ = 128;
    uint32_t fullMaxBlockNum_ = 0;

    gert::TilingContext *context_ = nullptr;
};

} // namespace optiling

#endif // GEN_FROM_CACHE_INDICES_TILING_H_
