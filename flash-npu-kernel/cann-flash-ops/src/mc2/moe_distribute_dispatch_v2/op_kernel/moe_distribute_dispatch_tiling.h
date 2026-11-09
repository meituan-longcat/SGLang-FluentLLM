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
 * \file moe_distribute_dispatch_tiling.h
 * \brief
 */

#ifndef ASCENDC_MOE_DISTRIBUTE_DISPATCH_TILING_H
#define ASCENDC_MOE_DISTRIBUTE_DISPATCH_TILING_H

#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"

struct MoeDistributeDispatchA2Info {
    uint32_t epWorldSize;                // epWorldSize
    uint32_t tpWorldSize;                // tpWorldSize
    uint32_t epRankId;                   // epRankId
    uint32_t tpRankId;                   // tpRankId
    uint32_t expertSharedType;           // expert type
    uint32_t sharedExpertRankNum;        // shared expert number
    uint32_t moeExpertNum;               // moe expert number
    uint32_t quantMode;                  // quant mode
    uint32_t globalBs;                   // globalBs = BS * worldSize
    uint32_t bs;                         // bs
    uint32_t k;                          // k
    uint32_t h;                          // h
    uint32_t aivNum;                     // aivNum
    bool isTokenMask;                    // input active mask 1dims or not
    bool isExpertMask;                   // input active mask 2dims or not
    bool isMc2Context;                      // reserved
    bool reserved2;                      // reserved
    bool reserved3;                      // reserved
    uint64_t totalUbSize;                // totalUbSize
    uint32_t expertTokenNumsType;        // expert token nums type, support 0: cumsum mode, 1: count mode
    int32_t zeroComputeExpertNum;        // sum of zero、copy and const expert nums
    uint32_t maxMoeExpertNum;            // maxMoeExpertNum
};

struct MoeDistributeDispatchA2TilingData {
    Mc2InitTiling mc2InitTiling;
    Mc2CcTiling mc2CcTiling;
    MoeDistributeDispatchA2Info moeDistributeDispatchInfo;
};

struct MoeDistributeDispatchInfo {
    uint32_t epWorldSize;                // epWorldSize
    uint32_t tpWorldSize;                // tpWorldSize
    uint32_t epRankId;                   // epRankId
    uint32_t tpRankId;                   // tpRankId
    uint32_t expertShardType;            // expert type
    uint32_t sharedExpertRankNum;        // shared expert number
    uint32_t moeExpertNum;               // moe expert number
    uint32_t quantMode;                  // quant mode
    uint32_t globalBs;                   // globalBs = BS * worldSize
    uint32_t bs;                         // bs
    uint32_t k;                          // k
    uint32_t h;                          // h
    uint32_t a;                          // a
    uint32_t aivNum;                     // aivNum
    bool isQuant;                        // whether quant or not
    bool reserved1;                      // reserved
    bool reserved2;                      // reserved
    bool reserved3;                      // reserved
    uint64_t totalUbSize;                // epWorldSize
    uint64_t totalWinSizeEp;
    uint64_t totalWinSizeTp;
    uint32_t expertTokenNumsType;        // expert token nums type, support 0: cumsum mode, 1: count mode
};

struct MoeDistributeDispatchTilingData {
    Mc2InitTiling mc2InitTiling;
    Mc2CcTiling mc2CcTiling1;
    Mc2CcTiling mc2CcTiling2;
    MoeDistributeDispatchInfo moeDistributeDispatchInfo;
};

#endif