/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef __grouped_matmul_TILING_DEF_H__
#define __grouped_matmul_TILING_DEF_H__

#include "kernel_tiling/kernel_tiling.h"
// #include "tiling/tiling_api.h"

#include <cstdint>
#include <cstring>

#define DT_BF16 bfloat16_t
#define ORIG_DTYPE_START DT_BF16
#define __CCE_UT_TEST__

constexpr uint16_t GMM_MAX_TENSOR_LIST_SIZE = 128;


inline void InitGMMTilingData(uint8_t* tiling, GMMTilingData* const_data)
{
    memcpy(const_data, tiling, sizeof(GMMTilingData));
}

#define GET_TILING_DATA(tilingData, tilingPointer) \
    GMMTilingData tilingData;         \
    InitGMMTilingData(tilingPointer, &tilingData)
#endif
