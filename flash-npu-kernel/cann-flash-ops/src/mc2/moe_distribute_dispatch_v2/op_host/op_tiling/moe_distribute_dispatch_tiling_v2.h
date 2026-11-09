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

#ifndef MOE_DISTRIBUTE_DISPATCH_TILING_V2
#define MOE_DISTRIBUTE_DISPATCH_TILING_V2

#include <cstdint>
#include "tiling/tiling_api.h"
#include "graph/utils/type_utils.h"
#include "register/tilingdata_base.h"
#include "tiling_base/tiling_base.h"
#include "tiling/mc2_opversion_manager.h"

namespace optiling {

struct DispatchV2Config {
    uint32_t contextIndex = 0U;  // 0: 根据dispatchV3算子原型标志位初始化context索引
    uint32_t xIndex = 0U; // 0: 根据dispatchV2算子原型标志位初始化groupEp索引
    uint32_t expertIdsIndex = 1U; // 1: 根据dispatchV2算子原型标志位初始化expertIds索引
    uint32_t scalesIndex = 2U; // 2: 根据dispatchV2算子原型标志位初始化scales索引
    uint32_t xActiveMaskIndex = 3U; // 3: 根据dispatchV2算子原型标志位初始化xActiveMask索引
    uint32_t expertScalesIndex = 4U; // 4: 根据dispatchV2算子原型标志位初始化expertScales索引
    uint32_t elasticInfoIndex = 5U; // 5: 根据dispatchV2算子原型标志位初始化elasticInfo索引
    uint32_t attrGroupEpIndex = 0; // 0: 根据dispatchV2算子原型标志位初始化groupEp索引
    uint32_t attrEpWorldSizeIndex = 1; // 1: 根据dispatchV2算子原型标志位初始化epWorldSize索引
    uint32_t attrEpRankIdIndex = 2; // 2: 根据dispatchV2算子原型标志位初始化epRankId索引
    uint32_t attrMoeExpertNumIndex = 3;  // 3: 根据dispatchV2算子原型标志位初始化moeExpertNum索引
    uint32_t attrCclBufferSizeIndex = 3; // 3: 根据dispatchV3算子原型标志位初始化cclBufferSize索引
    uint32_t attrGroupTpIndex = 4; // 4: 根据dispatchV2算子原型标志位初始化groupTp索引
    uint32_t attrTpWorldSizeIndex = 5; // 5: 根据dispatchV2算子原型标志位初始化tpWorldSize索引
    uint32_t attrTpRankIdIndex = 6; // 6: 根据dispatchV2算子原型标志位初始化tpRankId索引
    uint32_t attrExpertSharedTypeIndex = 7; // 7: 根据dispatchV2算子原型标志位初始化expertSharedType索引
    uint32_t attrSharedExpertNumIndex = 8; // 8: 根据dispatchV2算子原型标志位初始化sharedExpertNum索引
    uint32_t attrSharedExpertRankNumIndex = 9; // 9: 根据dispatchV2算子原型标志位初始化sharedExpertRankNum索引
    uint32_t attrQuantModeIndex = 10; // 10: 根据dispatchV2算子原型标志位初始化quantMode索引
    uint32_t attrGlobalBsIndex = 11; // 11: 根据dispatchV2算子原型标志位初始化globalBs索引
    uint32_t attrExpertTokenNumsTypeIndex = 12; // 12: 根据dispatchV2算子原型标志位初始化expertTokenNumType索引
    uint32_t attrCommAlgIndex = 13; // 13: 根据dispatchV2算子原型标志位初始化commAlg索引
    uint32_t attrZeroExpertNumIndex = 14; // 14: 根据dispatchV2算子原型标志位初始化zeroExpertNumIndex索引
    uint32_t attrCopyExpertNumIndex = 15; // 15: 根据dispatchV2算子原型标志位初始化copyExpertNumIndex索引
    uint32_t attrConstExpertNumIndex = 16; // 16: 根据dispatchV2算子原型标志位初始化constExpertNumIndex索引
    bool isMc2Context = false;
};
using namespace Ops::Transformer::OpTiling;

ge::graphStatus MoeDistributeDispatchA3TilingFuncImplPublic(gert::TilingContext* context, DispatchV2Config& config);

}

#endif
