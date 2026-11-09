/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef __TEST_GROUPED_MATMUL_TILING_H__
#define __TEST_GROUPED_MATMUL_TILING_H__

#include <cstdint>
#include <cstring>

#include "kernel_tiling/kernel_tiling.h"

#define DT_BF16 bfloat16_t
#define ORIG_DTYPE_START DT_BF16
#define __CCE_UT_TEST__
#define REGISTER_TILINGDATA_SIZE(tiling_struct, counter)
constexpr uint16_t GMM_MAX_TENSOR_LIST_SIZE = 128;

#pragma pack(push, 8)
struct GMMArray {
    int32_t mList[128] = {0};
    int32_t kList[128] = {0};
    int32_t nList[128] = {0};
};
#pragma pack(pop)
#pragma pack(push, 8)
struct GMMBaseParams {
    uint32_t groupNum = 0;
    uint32_t coreNum = 0;
    uint32_t activeType = 0;
    uint32_t ubBaseK = 0;
    uint32_t ubBaseN = 0;
    uint32_t ubCalSize = 0;
    uint32_t ubResetBytes = 0;
    uint32_t singleWeight = 0;
    uint32_t singleX = 0;
    uint32_t singleY = 0;
    int32_t groupType = 0;
    uint32_t singleN = 0;
    uint32_t quantParam = 0;
    uint32_t groupListType = 0;
    uint32_t m = 0;
    uint32_t hasBias = 0;
    uint64_t workspaceSize = 0;
    uint64_t totalInGroup = 0;
    uint64_t k = 0;
    uint64_t n = 0;
    uint64_t vBaseM = 0;
    uint64_t parallNum = 0;
    uint64_t quantGroupNum = 0;
    uint64_t isPreTiling = 0;
    uint32_t withOffset = 0;
    uint32_t isOutputDisableL2Cache = 0;
};
#pragma pack(pop)

#pragma pack(push,8)
struct A8W4HPTiling {
    uint32_t group_num = 0;
    int8_t group_type = 0;
    uint32_t required_core_num = 0;
    float format_in = 0;
    float format_out = 0;
    uint32_t numAic = 0;
    uint32_t numAiv = 0;
    uint64_t szUb = 0;
    uint64_t szL0A = 0;
    uint64_t szL0C = 0;
    uint8_t pattern = 0;
    uint8_t kernel_index = 0;
    uint32_t splitTimes = 0;
    int8_t output_type = 0;
    uint32_t ori_in0_shape[2] = {0};
    uint32_t ori_in1_shape[2] = {0};
    uint32_t ori_out_shape[2] = {0};
    uint32_t single_core_tiling[3] = {0};
    uint32_t single_core_base_tiling[2] = {0};
    uint32_t splitRecord[3] = {0};
    uint64_t workspaceOffset = 0;
};
#pragma pack(pop)

#pragma pack(push, 8)
struct GMMTilingData {
    GMMBaseParams gmmBaseParams;
    GMMArray gmmArray;
    TCubeTiling mmTilingData;
    A8W4HPTiling hpTilingData;
};
#pragma pack(pop)

template <class T>
void InitGmmMemberData(uint8_t* tiling, T* const_data)
{
    memcpy(const_data, tiling, sizeof(T));
}

inline void InitGMMTilingData(uint8_t* tiling, GMMTilingData* const_data)
{
    memcpy(const_data, tiling, sizeof(GMMTilingData));
}

#define GET_TILING_DATA(tilingData, tilingPointer) \
    GMMTilingData tilingData;         \
    InitGMMTilingData(tilingPointer, &tilingData)

#define GET_TILING_DATA_MEMBER(tiling_type, member, var, tiling) \
    REGISTER_TILINGDATA_SIZE(tiling_type, __COUNTER__); \
    decltype(((tiling_type *)0)->member) var; \
    size_t offset##var = (size_t)(&((tiling_type *)0)->member); \
    InitGmmMemberData<decltype(((tiling_type *)0)->member)>(tiling + offset##var, &var)

#endif