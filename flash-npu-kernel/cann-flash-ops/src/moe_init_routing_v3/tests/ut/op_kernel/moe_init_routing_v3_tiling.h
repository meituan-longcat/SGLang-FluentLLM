/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __MOE_INIT_ROUTING_V3_TILING_H__
#define __MOE_INIT_ROUTING_V3_TILING_H__

#include <cstdint>
#include <cstring>

#include "kernel_tiling/kernel_tiling.h"

#pragma pack(1)
struct MoeV3VBSComputeTilingData {
    int64_t needCoreNum = 0;
    int64_t perCoreElements = 0;
    int64_t perCoreLoops = 0;
    int64_t perCorePerLoopElements = 0;
    int64_t perCoreLastLoopElements = 0;
    int64_t lastCoreElements = 0;
    int64_t lastCoreLoops = 0;
    int64_t lastCorePerLoopElements = 0;
    int64_t lastCoreLastLoopElements = 0;
    int64_t oneLoopMaxElements = 0;
};
#pragma pack()

#ifdef __NPU_TILING__
inline[aicore] void InitTilingData(const __gm__ uint8_t *tiling, MoeV3VBSComputeTilingData *const_data)
{
    const __gm__ uint32_t *src = (const __gm__ uint32_t *)tiling;
    uint32_t *dst = (uint32_t *)const_data;
    for (auto i = 0; i < sizeof(MoeV3VBSComputeTilingData) / 4; i++)
        *(dst + i) = *(src + i);
}
#else
inline void InitTilingData(uint8_t *tiling, MoeV3VBSComputeTilingData *const_data)
{
    memcpy(const_data, tiling, sizeof(MoeV3VBSComputeTilingData));
}
#endif

#pragma pack(1)
struct MoeV3VMSMiddleComputeTilingData {
    int64_t needCoreNum = 0;
};
#pragma pack()

#ifdef __NPU_TILING__
inline[aicore] void InitTilingData(const __gm__ uint8_t *tiling, MoeV3VMSMiddleComputeTilingData *const_data)
{
    const __gm__ uint32_t *src = (const __gm__ uint32_t *)tiling;
    uint32_t *dst = (uint32_t *)const_data;
    for (auto i = 0; i < sizeof(MoeV3VMSMiddleComputeTilingData) / 4; i++)
        *(dst + i) = *(src + i);
}
#else
inline void InitTilingData(uint8_t *tiling, MoeV3VMSMiddleComputeTilingData *const_data)
{
    memcpy(const_data, tiling, sizeof(MoeV3VMSMiddleComputeTilingData));
}
#endif

#pragma pack(1)
struct MoeV3SortOutComputeTilingData {
    int64_t oneLoopMaxElements = 0;
};
#pragma pack()

#ifdef __NPU_TILING__
inline[aicore] void InitTilingData(const __gm__ uint8_t *tiling, MoeV3SortOutComputeTilingData *const_data)
{
    const __gm__ uint32_t *src = (const __gm__ uint32_t *)tiling;
    uint32_t *dst = (uint32_t *)const_data;
    for (auto i = 0; i < sizeof(MoeV3SortOutComputeTilingData) / 4; i++)
        *(dst + i) = *(src + i);
}
#else
inline void InitTilingData(uint8_t *tiling, MoeV3SortOutComputeTilingData *const_data)
{
    memcpy(const_data, tiling, sizeof(MoeV3SortOutComputeTilingData));
}
#endif

#pragma pack(1)
struct MoeV3GatherOutComputeTilingData {
    int64_t needCoreNum = 0;
    int64_t perCoreIndicesElements = 0;
    int64_t lastCoreIndicesElements = 0;
    int64_t perCoreIndicesLoops = 0;
    int64_t perCorePerLoopIndicesElements = 0;
    int64_t perCoreLastLoopIndicesElements = 0;
    int64_t lastCoreIndicesLoops = 0;
    int64_t lastCorePerLoopIndicesElements = 0;
    int64_t lastCoreLastLoopIndicesElements = 0;
    int64_t colsLoops = 0;
    int64_t perLoopCols = 0;
    int64_t lastLoopCols = 0;
    int64_t activeNum = 0;
};
#pragma pack()

#ifdef __NPU_TILING__
inline[aicore] void InitTilingData(const __gm__ uint8_t *tiling, MoeV3GatherOutComputeTilingData *const_data)
{
    const __gm__ uint32_t *src = (const __gm__ uint32_t *)tiling;
    uint32_t *dst = (uint32_t *)const_data;
    for (auto i = 0; i < sizeof(MoeV3GatherOutComputeTilingData) / 4; i++)
        *(dst + i) = *(src + i);
}
#else
inline void InitTilingData(uint8_t *tiling, MoeV3GatherOutComputeTilingData *const_data)
{
    memcpy(const_data, tiling, sizeof(MoeV3GatherOutComputeTilingData));
}
#endif

#pragma pack(1)
struct MoeV3ExpertTokensCountTilingData {
    int64_t needCoreNum = 0;
    int64_t perCoreElements = 0;
    int64_t lastCoreElements = 0;
    int64_t perCoreLoops = 0;
    int64_t perCorePerLoopElements = 0;
    int64_t perCoreLastLoopElements = 0;
    int64_t lastCoreLoops = 0;
    int64_t lastCorePerLoopElements = 0;
    int64_t lastCoreLastLoopElements = 0;
};
#pragma pack()

#ifdef __NPU_TILING__
inline[aicore] void InitTilingData(const __gm__ uint8_t *tiling, MoeV3ExpertTokensCountTilingData *const_data)
{
    const __gm__ uint32_t *src = (const __gm__ uint32_t *)tiling;
    uint32_t *dst = (uint32_t *)const_data;
    for (auto i = 0; i < sizeof(MoeV3ExpertTokensCountTilingData) / 4; i++)
        *(dst + i) = *(src + i);
}
#else
inline void InitTilingData(uint8_t *tiling, MoeV3ExpertTokensCountTilingData *const_data)
{
    memcpy(const_data, tiling, sizeof(MoeV3ExpertTokensCountTilingData));
}
#endif

#pragma pack(1)
struct MoeV3SrcToDstCapacityComputeTilingData {
  int64_t needCoreNum = 0;
  int64_t perCoreRows = 0;
  int64_t perCorePerLoopRows = 0;
  int64_t perCoreLastLoopRows = 0;
  int64_t lastCoreRows = 0;
  int64_t lastCorePerLoopRows = 0;
  int64_t lastCoreLastLoopRows = 0;
  int64_t perCoreLoops = 0;
  int64_t lastCoreLoops = 0;
  int64_t perLoopCols = 0;
  int64_t lastLoopCols = 0;
  int64_t colLoops = 0;
};
#pragma pack()

#ifdef __NPU_TILING__
inline[aicore] void InitTilingData(const __gm__ uint8_t* tiling, MoeV3SrcToDstCapacityComputeTilingData* const_data) {
  const __gm__ uint32_t* src = (const __gm__ uint32_t*)tiling;
  uint32_t* dst = (uint32_t*)const_data;
  for (auto i = 0; i < sizeof(MoeV3SrcToDstCapacityComputeTilingData) / 4; i++)
    *(dst + i) = *(src + i);
}
#else
inline void InitTilingData(uint8_t* tiling, MoeV3SrcToDstCapacityComputeTilingData* const_data) {
  memcpy(const_data, tiling, sizeof(MoeV3SrcToDstCapacityComputeTilingData));
}
#endif

#pragma pack(1)
struct MoeV3SrcToDstComputeTilingData {
  int64_t needCoreNum = 0;
  int64_t perCoreElements = 0;
  int64_t perCorePerLoopElements = 0;
  int64_t perCoreLastLoopElements = 0;
  int64_t lastCoreElements = 0;
  int64_t lastCorePerLoopElements = 0;
  int64_t lastCoreLastLoopElements = 0;
  int64_t perCoreLoops = 0;
  int64_t lastCoreLoops = 0;
};
#pragma pack()

#ifdef __NPU_TILING__
inline[aicore] void InitTilingData(const __gm__ uint8_t* tiling, MoeV3SrcToDstComputeTilingData* const_data) {
  const __gm__ uint32_t* src = (const __gm__ uint32_t*)tiling;
  uint32_t* dst = (uint32_t*)const_data;
  for (auto i = 0; i < sizeof(MoeV3SrcToDstComputeTilingData) / 4; i++)
    *(dst + i) = *(src + i);
}
#else
inline void InitTilingData(uint8_t* tiling, MoeV3SrcToDstComputeTilingData* const_data) {
  memcpy(const_data, tiling, sizeof(MoeV3SrcToDstComputeTilingData));
}
#endif

#pragma pack(1)
struct MoeInitRoutingV3TilingData {
    int64_t coreNum = 0;
    int64_t n = 0;
    int64_t cols = 0;
    int64_t k = 0;
    int64_t expertStart = 0;
    int64_t expertEnd = 0;
    int64_t actualExpertNum = 0;
    int64_t quantMode = -1;
    int64_t rowIdxType = 0;
    int64_t isInputScale = 0;
    int64_t isInputOffset = 0;
    int64_t expertNum = -1;
    int64_t expertTokensNumType = -1;
    int64_t expertTokensNumFlag = 1;
    int64_t gatherFirstFullload = 0;
    int64_t ep = 1;
    int64_t activeNum = 0;
    int64_t dropPadMode = 0;
    int64_t smoothType = 2;
    int64_t expertCountElements = 0;
    int64_t expertCapacity = 2;
    MoeV3VBSComputeTilingData vbsComputeParamsOp;
    MoeV3VMSMiddleComputeTilingData vmsMiddleComputeParamsOp;
    MoeV3SortOutComputeTilingData sortOutComputeParamsOp;
    MoeV3ExpertTokensCountTilingData expertTokensCountTilingDataOp;
    MoeV3GatherOutComputeTilingData gatherOutComputeParamsOp;
    MoeV3SrcToDstCapacityComputeTilingData srcToDstDropPadParamsOp;
    MoeV3SrcToDstCapacityComputeTilingData srcToDstDropPadDynamicParamsOp;
    MoeV3SrcToDstComputeTilingData srcToDstComputeParamsOp;
};
#pragma pack()

#ifdef __NPU_TILING__
inline[aicore] void InitTilingData(const __gm__ uint8_t *tiling, MoeInitRoutingV3TilingData *const_data)
{
    const __gm__ uint32_t *src = (const __gm__ uint32_t *)tiling;
    uint32_t *dst = (uint32_t *)const_data;
    for (auto i = 0; i < sizeof(MoeInitRoutingV3TilingData) / 4; i++)
        *(dst + i) = *(src + i);
}
#else
inline void InitTilingData(uint8_t *tiling, MoeInitRoutingV3TilingData *const_data)
{
    memcpy(const_data, tiling, sizeof(MoeInitRoutingV3TilingData));
}
#endif

#define GET_TILING_DATA_WITH_STRUCT(tiling_struct, tiling_data, tiling_arg)                                            \
    tiling_struct tiling_data;                                                                                         \
    InitTilingData(tiling_arg, &tiling_data)

#define GET_TILING_DATA(tiling_data, tiling_arg)                                                                       \
    MoeInitRoutingV3TilingData tiling_data;                                                                            \
    InitTilingData(tiling_arg, &tiling_data)

#define DTYPE_X float

#endif