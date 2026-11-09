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
 * \file moe_distribute_dispatch_v2_tiling.h
 * \brief
 */

#ifndef MOE_DISTRIBUTE_DISPATCH_V2_TILING_H
#define MOE_DISTRIBUTE_DISPATCH_V2_TILING_H

struct MoeDistributeDispatchV2Info {
    uint32_t epWorldSize;                // epWorldSize
    uint32_t tpWorldSize;                // tpWorldSize
    uint32_t epRankId;                   // epRankId
    uint32_t tpRankId;                   // tpRankId
    uint32_t expertShardType;            // expert type
    uint32_t sharedExpertNum;            // shared expert number
    uint32_t sharedExpertRankNum;        // shared expert rank number
    uint32_t moeExpertNum;               // moe expert number
    uint32_t quantMode;                  // quant mode
    uint32_t globalBs;                   // globalBs = BS * worldSize
    uint32_t bs;                         // bs
    uint32_t k;                          // k
    uint32_t h;                          // h
    uint32_t a;                          // a
    uint32_t aivNum;                     // aivNum
    bool isTokenMask;                    // input active mask 1dims or not
    bool isExpertMask;                   // input active mask 2dims or not
    bool hasElasticInfo;                 // has elasticinfo or not
    bool isQuant;                        // whether quant or not
    bool isMc2Context;
    bool reserved1;
    bool reserved2;
    uint64_t totalUbSize;                // epWorldSize
    uint64_t totalWinSizeEp;
    uint64_t totalWinSizeTp;
    uint32_t expertTokenNumsType;        // expert token nums type, support 0: cumsum mode, 1: count mode
    int32_t zeroComputeExpertNum;       // sum of zero, copy and const expert nums
    uint32_t maxSizeForUbBuffer;
    uint64_t scalesRow;
    uint64_t scalesCol;
    uint32_t scalesTypeSize;
    uint64_t scalesCount;
};

struct MoeDistributeDispatchV2TilingData {
    Mc2InitTiling mc2InitTiling;
    Mc2CcTiling mc2CcTiling1;
    Mc2CcTiling mc2CcTiling2;
    MoeDistributeDispatchV2Info moeDistributeDispatchV2Info;
};

#endif
