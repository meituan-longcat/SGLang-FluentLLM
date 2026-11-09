/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_distribute_combine_tiling.h
 * \brief
 */
#ifndef MOE_DISTRIBUTE_COMBINE_A2_TILING_H
#define MOE_DISTRIBUTE_COMBINE_A2_TILING_H

#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"

struct MoeDistributeCombineA2Info {
    uint32_t epWorldSize;                // epWorldSize
    uint32_t tpWorldSize;                // tpWorldSize
    uint32_t epRankId;                   // epRankId
    uint32_t tpRankId;                   // tpRankId
    uint32_t expertSharedType;           // expert type
    uint32_t sharedExpertRankNum;        // shared expert number
    uint32_t moeExpertNum;               // moe expert number
    uint32_t zeroExpertNum;              // zero expert number
    uint32_t copyExpertNum;              // copy expert number
    uint32_t constExpertNum;             // const expert number
    uint32_t globalBs;                   // globalBs = BS * worldSize
    uint32_t bs;                         // bs
    uint32_t k;                          // k
    uint32_t h;                          // h
    uint32_t aivNum;                     // aivNum
    uint64_t totalUbSize;                // epWorldSize
    bool isTokenMask;                    // input active mask 1dims or not
    bool isExpertMask;                   // input active mask 2dims or not
    int8_t reserved[7];                  // Pad 7 int8 for memory alignment
};

struct MoeDistributeCombineA2TilingData {
    Mc2InitTiling mc2InitTiling;
    Mc2CcTiling mc2CcTiling;
    MoeDistributeCombineA2Info moeDistributeCombineInfo;
};

// a3
struct MoeDistributeCombineInfo {
    uint32_t epWorldSize;
    uint32_t tpWorldSize;
    uint32_t epRankId;
    uint32_t tpRankId;
    uint32_t expertShardType;
    uint32_t sharedExpertRankNum;
    uint32_t moeExpertNum;
    uint32_t moeExpertPerRankNum;
    uint32_t globalBs;
    uint32_t bs;
    uint32_t k;
    uint32_t h;
    uint32_t a;
    uint32_t aivNum;
    uint64_t totalUbSize;
    uint64_t totalWinSizeEp;
    uint64_t totalWinSizeTp;
};
struct MoeDistributeCombineTilingData {
    Mc2InitTiling mc2InitTiling;
    Mc2CcTiling mc2CcTiling1;
    Mc2CcTiling mc2CcTiling2;
    MoeDistributeCombineInfo moeDistributeCombineInfo;
};

#endif //__MOE_DISTRIBUTE_COMBINE_A2_TILING_H__