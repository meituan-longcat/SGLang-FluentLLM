/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 */

#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_FLAT_QUANT_TILING_H
#define OPS_BUILT_IN_OP_TILING_RUNTIME_FLAT_QUANT_TILING_H

#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "tiling_base/tiling_base.h"

namespace optiling {

// This layout is part of the host/kernel ABI and must remain synchronized
// with the FlatQuant kernel implementation.
struct FlatQuantCompileInfo {
    int64_t aicNum;
    int64_t aivNum;
    NpuArch npuArch;
};

BEGIN_TILING_DATA_DEF(FlatQuantTilingData)
TILING_DATA_FIELD_DEF(uint8_t, dataType);
TILING_DATA_FIELD_DEF(uint8_t, hasP2);
TILING_DATA_FIELD_DEF(int64_t, K);
TILING_DATA_FIELD_DEF(int64_t, M);
TILING_DATA_FIELD_DEF(int64_t, N);
TILING_DATA_FIELD_DEF(int64_t, iterBatch);
TILING_DATA_FIELD_DEF(float, clipRatio);
TILING_DATA_FIELD_DEF(float, dstTypeMax);
TILING_DATA_FIELD_DEF(float, invDstTypeMax);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, matmulTilingR);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, matmulTilingL);
TILING_DATA_FIELD_DEF(int64_t, groupNum);
TILING_DATA_FIELD_DEF(int64_t, groupListType);
TILING_DATA_FIELD_DEF(uint8_t, clearPadding);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(FlatQuant, FlatQuantTilingData)
} // namespace optiling

#endif // OPS_BUILT_IN_OP_TILING_RUNTIME_FLAT_QUANT_TILING_H
