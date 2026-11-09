/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef COMPUTE_N_GRAM_IDS_TILING_H
#define COMPUTE_N_GRAM_IDS_TILING_H

#include <cstdint>
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "error/ops_error.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(ComputeNGramIdsTilingData)
TILING_DATA_FIELD_DEF(uint32_t, coreNum);
TILING_DATA_FIELD_DEF(uint32_t, batchSize);
TILING_DATA_FIELD_DEF(uint32_t, totalTask);
TILING_DATA_FIELD_DEF(int32_t, oeN);
TILING_DATA_FIELD_DEF(int32_t, oeK);
TILING_DATA_FIELD_DEF(int32_t, maxContextLen);
TILING_DATA_FIELD_DEF(int32_t, eosTokenId);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ComputeNGramIds, ComputeNGramIdsTilingData)

struct ComputeNGramIdsCompileInfo {
    uint64_t coreNum;
    uint64_t ubSize;
};

class ComputeNGramIdsTiling {
public:
    explicit ComputeNGramIdsTiling(gert::TilingContext* ctx) : context_(ctx) {}
    ~ComputeNGramIdsTiling() = default;

    ge::graphStatus DoTiling();

private:
    ge::graphStatus ResolvePlatform();
    ge::graphStatus PopulateTiling();
    ge::graphStatus Finalize();

    gert::TilingContext* context_ = nullptr;
    ComputeNGramIdsTilingData tilingData_;
    uint64_t coreNum_ = 0;
    uint64_t ubSize_ = 0;
    uint32_t usedCoreNum_ = 0;
    platform_ascendc::SocVersion socVersion_ = platform_ascendc::SocVersion::ASCEND910B;
};

}  // namespace optiling
#endif
