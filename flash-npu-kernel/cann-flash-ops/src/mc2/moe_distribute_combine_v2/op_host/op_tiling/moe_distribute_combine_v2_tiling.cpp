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
 * \file moe_distribute_combine_v2_tiling.cpp
 * \brief
 */

#include <queue>
#include <vector>
#include <dlfcn.h>
#include <fcntl.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/types.h>
#include <unistd.h>
#include <cmath>
#include <cstdint>
#include <string>
#include <type_traits>
#include "tiling/mc2_tiling_utils.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "mc2_log.h"
#include "graph/utils/type_utils.h"
#include "register/op_def_registry.h"
#include "platform/platform_infos_def.h"
#include "../../op_kernel/moe_distribute_combine_tiling.h"
#include "arch35/moe_distribute_combine_tiling_arch35.h"
#include "../../op_kernel/moe_distribute_combine_v2_tiling.h"
#include "../../op_kernel/moe_distribute_combine_v2_tiling_key.h"
#include "mc2_hcom_topo_info.h"

#ifdef MC2_EXCEPTION_HANDLER
#include "mc2_exception_dump.h"
#endif
#ifdef MC2_EXCEPTION_HANDLER
using namespace Mc2Exception;
#endif

using namespace Mc2Tiling;
using namespace AscendC;
using namespace ge;

namespace {
    constexpr uint32_t OP_VERSION_2 = 2;
    constexpr uint32_t EXPAND_X_INDEX = 0;
    constexpr uint32_t EXPERT_IDS_INDEX = 1;
    constexpr uint32_t ASSIST_INFO_INDEX = 2;
    constexpr uint32_t EP_SEND_COUNTS_INDEX = 3;
    constexpr uint32_t EXPERT_SCALES_INDEX = 4;
    constexpr uint32_t EXPAND_SCALES_INDEX = 10;

    constexpr uint32_t ATTR_GROUP_EP_INDEX = 0;
    constexpr uint32_t ATTR_EP_WORLD_SIZE_INDEX = 1;
    constexpr uint32_t ATTR_EP_RANK_ID_INDEX = 2;
    constexpr uint32_t ATTR_MOE_EXPERT_NUM_INDEX = 3;
    constexpr uint32_t ATTR_GROUP_TP_INDEX = 4;
    constexpr uint32_t ATTR_TP_WORLD_SIZE_INDEX = 5;
    constexpr uint32_t ATTR_TP_RANK_ID_INDEX = 6;
    constexpr uint32_t ATTR_EXPERT_SHARD_TYPE_INDEX = 7;
    constexpr uint32_t ATTR_SHARED_EXPERT_NUM_INDEX = 8;
    constexpr uint32_t ATTR_SHARED_EXPERT_RANK_NUM_INDEX = 9;
    constexpr uint32_t ATTR_GLOBAL_BS_INDEX = 10;
    constexpr uint32_t ATTR_OUT_DTYPE_INDEX = 11;
    constexpr uint32_t ATTR_COMM_QUANT_MODE_INDEX = 12;
    constexpr uint32_t ATTR_GROUP_LIST_TYPE_INDEX = 13;
    constexpr uint32_t ATTR_COMM_ALG_INDEX = 14;

    constexpr uint32_t INT8_COMM_QUANT = 2U;
    constexpr uint64_t INIT_TILINGKEY = 10000;
    constexpr uint64_t TILINGKEY_TP_WORLD_SIZE = 100;
    constexpr uint64_t TP_WORLD_SIZE_TWO = 2;
    constexpr uint32_t TILINGKEY_INT8_COMM_QUANT = 20U;

    constexpr uint32_t THREE_DIMS = 3U;
    constexpr uint32_t TWO_DIMS = 2U;
    constexpr uint32_t ONE_DIM = 1U;
    constexpr uint32_t ASSIST_INFO_DIMS = 1U;
    constexpr uint64_t TILING_KEY_BASE_A2 = 2000UL;
    constexpr uint64_t TILING_KEY_LAYERED_COMM_A2 = 3000UL;
    constexpr uint64_t TILING_KEY_INT8_COMM_QUANT_A2 = 100UL;
    constexpr uint32_t ARR_LENGTH = 128U;
    constexpr uint32_t OP_TYPE_ALL_TO_ALL = 8U; // numeric representation of AlltoAll
    constexpr uint32_t OP_TYPE_REDUCE_SCATTER = 7U; // numeric representation of AlltoAll
    constexpr uint32_t STATE_OFFSET = 32U;
    constexpr uint32_t ALIGNED_LEN = 256U;
    constexpr uint32_t DTYPE_SIZE_HALF = 2;
    constexpr uint8_t BUFFER_SINGLE = 1;
    constexpr uint8_t BUFFER_NUM = 2;

    constexpr size_t MAX_GROUP_NAME_LENGTH = 128UL;
    constexpr int64_t MAX_SHARED_EXPERT_NUM = 4;
    constexpr int64_t MAX_EP_WORLD_SIZE = 768L; // 384 * 2
    constexpr int64_t MAX_EP_WORLD_SIZE_LAYERED = 256;
    constexpr int64_t MIN_EP_WORLD_SIZE = 2;
    constexpr int64_t EP_RESTRICT_8 = 8;
    constexpr int64_t MAX_TP_WORLD_SIZE_LAYERED = 1;
    constexpr int64_t MAX_TP_WORLD_SIZE = 2;
    constexpr int64_t BS_UPPER_BOUND = 512;
    constexpr int64_t BS_UPPER_BOUND_LAYERED = 256;
    constexpr uint32_t H_BASIC_BLOCK_LAYERED = 32;
    constexpr int64_t RESIDUAL_X_DIM2_SIZE = 1;
    constexpr uint32_t NUM_PER_REP_FP32 = 64U;
    constexpr uint32_t RANK_NUM_PER_NODE = 16U;
    constexpr uint32_t AIV_NUM_93 = 48U;

    constexpr size_t SYSTEM_NEED_WORKSPACE = 16UL * 1024UL * 1024UL;
    constexpr size_t MASK_CALC_NEED_WORKSPACE = 10UL * 1024UL;
    constexpr int32_t HCCL_BUFFER_SIZE_DEFAULT = 200 * 1024 * 1024; // Bytes
    constexpr uint32_t VERSION_2 = 2;
    constexpr uint32_t HCOMMCNT_2 = 2;
    constexpr uint32_t RANK_LIST_NUM = 2;
    constexpr int64_t MOE_EXPERT_MAX_NUM = 1024;
    constexpr int64_t MOE_EXPERT_MAX_NUM_LAYERED = 512;
    constexpr int64_t LOCAL_EXPERT_MAX_SIZE = 2048;
    constexpr int64_t K_MAX = 16;
    constexpr int64_t H_MIN = 1024;
    constexpr int64_t H_MAX = 8192;
    constexpr int64_t H_MAX_LAYERED = 7168;
    constexpr uint64_t MB_SIZE = 1024UL * 1024UL;
    constexpr uint64_t TRIPLE = 3;
    constexpr uint64_t ASSIST_NUM_PER_A = 128UL;
    constexpr uint64_t WIN_ADDR_ALIGN = 512UL;
    constexpr uint64_t SCALE_EXPAND_IDX_BUFFER = 44UL; // scale32B + 3*4expandIdx
    constexpr uint64_t DOUBLE_DATA_BUFFER = 2UL;
    constexpr uint64_t MAX_OUT_DTYPE_SIZE = 2UL;
    constexpr uint64_t UB_ALIGN = 32UL;
    constexpr int64_t ELASTIC_METAINFO_OFFSET = 4;
    constexpr uint32_t HAS_ADD_RMS_NORM = 1;
    // A2
    constexpr int32_t MAX_EP_WORLD_SIZE_A2 = 384;
    constexpr int32_t MAX_EP_WORLD_SIZE_A2_LAYERED = 128;
    constexpr int32_t MAX_MOE_EXPERT_NUMS_A2 = 1024;
    constexpr uint32_t MAX_HIDDEN_SIZE_A2 = 8192;
    constexpr uint32_t LAYERED_MAX_HIDDEN_SIZE_A2 = 10240;
    constexpr uint32_t MAX_BATCH_SIZE_A2 = 1024;
    constexpr uint32_t LAYERED_MAX_BATCH_SIZE_A2 = 512;
    constexpr uint32_t RANK_NUM_PER_NODE_A2 = 8;
    constexpr uint32_t BLOCK_SIZE_A2 = 32;
    constexpr uint32_t MAX_K_VALUE_A2 = 16;
    const char *K_INNER_DEBUG = "MoeDistributeCombineV2 Tiling Debug";

    enum class CommQuantMode : int32_t {
        NON_QUANT = 0,
        INT12_QUANT = 1,
        INT8_QUANT = 2
    };
    using CommQuantModeType = std::underlying_type_t<CommQuantMode>;
}

namespace optiling {
// a3专有
static void PrintTilingDataInfo(const char *nodeName, MoeDistributeCombineV2TilingData& tilingData)
{
    OP_LOGD(nodeName, "epWorldSize is %u.", tilingData.moeDistributeCombineV2Info.epWorldSize);
    OP_LOGD(nodeName, "tpWorldSize is %u.", tilingData.moeDistributeCombineV2Info.tpWorldSize);
    OP_LOGD(nodeName, "epRankId is %u.", tilingData.moeDistributeCombineV2Info.epRankId);
    OP_LOGD(nodeName, "tpRankId is %u.", tilingData.moeDistributeCombineV2Info.tpRankId);
    OP_LOGD(nodeName, "expertShardType is %u.", tilingData.moeDistributeCombineV2Info.expertShardType);
    OP_LOGD(nodeName, "sharedExpertNum is %u.", tilingData.moeDistributeCombineV2Info.sharedExpertNum);
    OP_LOGD(nodeName, "sharedExpertRankNum is %u.", tilingData.moeDistributeCombineV2Info.sharedExpertRankNum);
    OP_LOGD(nodeName, "moeExpertNum is %u.", tilingData.moeDistributeCombineV2Info.moeExpertNum);
    OP_LOGD(nodeName, "moeExpertPerRankNum is %u.", tilingData.moeDistributeCombineV2Info.moeExpertPerRankNum);
    OP_LOGD(nodeName, "globalBs is %u.", tilingData.moeDistributeCombineV2Info.globalBs);
    OP_LOGD(nodeName, "bs is %u.", tilingData.moeDistributeCombineV2Info.bs);
    OP_LOGD(nodeName, "k is %u.", tilingData.moeDistributeCombineV2Info.k);
    OP_LOGD(nodeName, "h is %u.", tilingData.moeDistributeCombineV2Info.h);
    OP_LOGD(nodeName, "aivNum is %u.", tilingData.moeDistributeCombineV2Info.aivNum);
    OP_LOGD(nodeName, "totalUbSize is %lu.", tilingData.moeDistributeCombineV2Info.totalUbSize);
    OP_LOGD(nodeName, "totalWinSizeEP is %lu.", tilingData.moeDistributeCombineV2Info.totalWinSizeEp);
    OP_LOGD(nodeName, "totalWinSizeTP is %lu.", tilingData.moeDistributeCombineV2Info.totalWinSizeTp);
    OP_LOGD(nodeName, "hasElastic is %d.", tilingData.moeDistributeCombineV2Info.hasElasticInfo);
}

static ge::graphStatus GetExpertsAttrAndSetTilingData(const gert::TilingContext *context,
    MoeDistributeCombineV2TilingData &tilingData, const char *nodeName, const CombineV2Config& config, const bool isLayered)
{
    auto attrs = context->GetAttrs();
    auto moeExpertNumPtr = attrs->GetAttrPointer<int64_t>(config.attrMoeExpertNumIndex);
    auto zeroExpertNumPtr = attrs->GetAttrPointer<int64_t>(static_cast<int>(config.attrZeroExpertNumIndex));
    auto copyExpertNumPtr = attrs->GetAttrPointer<int64_t>(static_cast<int>(config.attrCopyExpertNumIndex));
    auto constExpertNumPtr = attrs->GetAttrPointer<int64_t>(static_cast<int>(config.attrConstExpertNumIndex));
    auto expertShardPtr = attrs->GetAttrPointer<int64_t>((config.attrExpertSharedTypeIndex));
    OP_TILING_CHECK(moeExpertNumPtr == nullptr, OP_LOGE(nodeName, "moeExpertNum is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(zeroExpertNumPtr == nullptr, OP_LOGE(nodeName, "zeroExpertNum is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(copyExpertNumPtr == nullptr, OP_LOGE(nodeName, "copyExpertNum is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(constExpertNumPtr == nullptr, OP_LOGE(nodeName, "constExpertNum is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(expertShardPtr == nullptr, OP_LOGE(nodeName, "expertShardType is null."), return ge::GRAPH_FAILED);
    int64_t moeExpertNum = *moeExpertNumPtr;
    int64_t zeroExpertNum = *zeroExpertNumPtr;
    int64_t copyExpertNum = *copyExpertNumPtr;
    int64_t constExpertNum = *constExpertNumPtr;
    OP_TILING_CHECK(
        (moeExpertNum + zeroExpertNum + copyExpertNum + constExpertNum) > INT32_MAX,
        OP_LOGE(nodeName, "moeExpertNum + zeroExpertNum + copyExpertNum + constExpertNum exceeds MAX_INT32."),
        return ge::GRAPH_FAILED);
    int64_t moeExpertMaxNum = isLayered ? MOE_EXPERT_MAX_NUM_LAYERED : MOE_EXPERT_MAX_NUM;
    OP_TILING_CHECK((moeExpertNum <= 0) || (moeExpertNum > moeExpertMaxNum),
        OP_LOGE(nodeName, "moeExpertNum is invalid, only support (0, %ld], but got moeExpertNum=%ld.",
        moeExpertMaxNum, moeExpertNum), return ge::GRAPH_FAILED);
    OP_TILING_CHECK((zeroExpertNum < 0), OP_LOGE(nodeName,
        "zeroExpertNum less than 0, zeroExpertNum is %ld.", zeroExpertNum), return ge::GRAPH_FAILED);
    OP_TILING_CHECK((copyExpertNum < 0), OP_LOGE(nodeName,
        "copyExpertNum less than 0, copyExpertNum is %ld.", copyExpertNum), return ge::GRAPH_FAILED);
    OP_TILING_CHECK((constExpertNum < 0), OP_LOGE(nodeName,
        "constExpertNum less than 0, constExpertNum is %ld.", constExpertNum), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(*expertShardPtr != 0,
        OP_LOGE(nodeName, "expertShardType is invalid, only support 0, but got expertShardType=%ld.",
        *expertShardPtr), return ge::GRAPH_FAILED);
    tilingData.moeDistributeCombineV2Info.moeExpertNum = static_cast<uint32_t>(moeExpertNum);
    tilingData.moeDistributeCombineV2Info.zeroExpertNum = static_cast<uint32_t>(zeroExpertNum);
    tilingData.moeDistributeCombineV2Info.copyExpertNum = static_cast<uint32_t>(copyExpertNum);
    tilingData.moeDistributeCombineV2Info.constExpertNum = static_cast<uint32_t>(constExpertNum);
    tilingData.moeDistributeCombineV2Info.expertShardType = static_cast<uint32_t>(*expertShardPtr);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetSharedAttrAndSetTilingData(const gert::TilingContext *context,
    MoeDistributeCombineV2TilingData &tilingData, const char *nodeName, const CombineV2Config& config, const bool isLayered)
{
    auto attrs = context->GetAttrs();
    auto sharedExpertNumPtr = attrs->GetAttrPointer<int64_t>(static_cast<int>((config.attrSharedExpertNumIndex)));
    auto sharedExpertRankNumPtr = attrs->GetAttrPointer<int64_t>((config.attrSharedExpertRankNumIndex));
    auto epWorldSizePtr = attrs->GetAttrPointer<int64_t>((config.attrEpWorldSizeIndex));
    OP_TILING_CHECK(sharedExpertNumPtr == nullptr, OP_LOGE(nodeName, "sharedExpertNum is null."),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(sharedExpertRankNumPtr == nullptr, OP_LOGE(nodeName, "sharedExpertRankNum is null."),
        return ge::GRAPH_FAILED);
    int64_t sharedExpertRankNum = *sharedExpertRankNumPtr;
    int64_t epWorldSize = *epWorldSizePtr;
    if (isLayered) {
        OP_TILING_CHECK((sharedExpertRankNum != 0),
            OP_LOGE(nodeName, "sharedExpertNum is invalid, only support 0 in hierarchy mode, but got sharedExpertNum=%ld, sharedExpertRankNum=%ld",
            *sharedExpertNumPtr, sharedExpertRankNum), return ge::GRAPH_FAILED);
    } else {
        OP_TILING_CHECK((*sharedExpertNumPtr < 0) || (*sharedExpertNumPtr > MAX_SHARED_EXPERT_NUM),
            OP_LOGE(nodeName, "sharedExpertNum is invalid, only support [0, %ld], but got sharedExpertNum=%ld.",
            MAX_SHARED_EXPERT_NUM, *sharedExpertNumPtr), return ge::GRAPH_FAILED);
        OP_TILING_CHECK((sharedExpertRankNum < 0) || (sharedExpertRankNum >= epWorldSize),
            OP_LOGE(nodeName, "sharedExpertRankNum is invalid, only support [0, %ld), but got sharedExpertRankNum=%ld.",
            epWorldSize, sharedExpertRankNum), return ge::GRAPH_FAILED);
    }
    tilingData.moeDistributeCombineV2Info.sharedExpertNum = static_cast<uint32_t>(*sharedExpertNumPtr);
    tilingData.moeDistributeCombineV2Info.sharedExpertRankNum = static_cast<uint32_t>(sharedExpertRankNum);
    if (tilingData.moeDistributeCombineV2Info.sharedExpertRankNum == 0U) {
        if (tilingData.moeDistributeCombineV2Info.sharedExpertNum == 1U) {
            tilingData.moeDistributeCombineV2Info.sharedExpertNum = 0U;
        }
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetTpAndEpAttrAndSetTilingData(const gert::TilingContext *context,
    MoeDistributeCombineV2TilingData &tilingData, const char *nodeName, const CombineV2Config& config, const bool isLayered,
    const uint32_t commQuantMode, std::string &groupTp)
{
    auto attrs = context->GetAttrs();
    auto epWorldSizePtr = attrs->GetAttrPointer<int64_t>((config.attrEpWorldSizeIndex));
    auto tpWorldSizePtr = attrs->GetAttrPointer<int64_t>((config.attrTpWorldSizeIndex));
    auto epRankIdPtr = attrs->GetAttrPointer<int64_t>((config.attrEpRankIdIndex));
    auto tpRankIdPtr = attrs->GetAttrPointer<int64_t>((config.attrTpRankIdIndex));
    OP_TILING_CHECK(epWorldSizePtr == nullptr, OP_LOGE(nodeName, "epWorldSize is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(tpWorldSizePtr == nullptr, OP_LOGE(nodeName, "tpWorldSize is null."), return ge::GRAPH_FAILED);
    int64_t epWorldSize = *epWorldSizePtr;
    int64_t maxEpworldsize = isLayered ? MAX_EP_WORLD_SIZE_LAYERED : MAX_EP_WORLD_SIZE;
    int64_t maxTpworldsize = isLayered ? MAX_TP_WORLD_SIZE_LAYERED : MAX_TP_WORLD_SIZE;
    OP_TILING_CHECK((epWorldSize < MIN_EP_WORLD_SIZE) || (epWorldSize > maxEpworldsize),
        OP_LOGE(nodeName, "epWorldSize is invalid, only support [%ld, %ld], but got epWorldSize=%ld.",
        MIN_EP_WORLD_SIZE, maxEpworldsize, epWorldSize), return ge::GRAPH_FAILED);
    // 校验epWorldSize是否是16整数倍
    OP_TILING_CHECK((isLayered && (epWorldSize % RANK_NUM_PER_NODE != 0)),
        OP_LOGE(nodeName, "epWorldSize should be %u Aligned, but got %ld.", RANK_NUM_PER_NODE, epWorldSize), return ge::GRAPH_FAILED);
    OP_TILING_CHECK((*epRankIdPtr < 0) || (*epRankIdPtr >= epWorldSize),
        OP_LOGE(nodeName, "epRankId is invalid, only support [0, %ld), but got epRankId=%ld.",
        epWorldSize, *epRankIdPtr), return ge::GRAPH_FAILED);
    OP_TILING_CHECK((*tpWorldSizePtr < 0) || (*tpWorldSizePtr > maxTpworldsize),
        OP_LOGE(nodeName, "tpWorldSize is invalid, only support [0, %ld], but got tpWorldSize=%ld.",
        maxTpworldsize, *tpWorldSizePtr), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(epRankIdPtr == nullptr, OP_LOGE(nodeName, "epRankId is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(tpRankIdPtr == nullptr, OP_LOGE(nodeName, "tpRankId is null."), return ge::GRAPH_FAILED);
    
    if (*tpWorldSizePtr > 1) {
        auto groupTpPtr = attrs->GetAttrPointer<char>(static_cast<int>(config.attrGroupTpIndex));
        OP_TILING_CHECK((*tpRankIdPtr < 0) || (*tpRankIdPtr >= *tpWorldSizePtr),
            OP_LOGE(nodeName, "tpRankId is invalid, only support [0, %ld), but got tpRankId=%ld.",
            *tpWorldSizePtr, *tpRankIdPtr), return ge::GRAPH_FAILED);
        OP_TILING_CHECK((groupTpPtr == nullptr) || (strnlen(groupTpPtr, MAX_GROUP_NAME_LENGTH) == 0) ||
            (strnlen(groupTpPtr, MAX_GROUP_NAME_LENGTH) == MAX_GROUP_NAME_LENGTH),
            OP_LOGE(nodeName, "groupTpPtr is null."), return ge::GRAPH_FAILED);
        OP_TILING_CHECK((commQuantMode != 0), OP_LOGE(nodeName,
            "commQuantMode only supports 0 when tpWorldSize > 1, but got commQuantMode=%ld, tpWorldSize=%ld.",
            commQuantMode, *tpWorldSizePtr), return ge::GRAPH_FAILED);
        groupTp = std::string(groupTpPtr);
    } else {
        OP_TILING_CHECK(*tpRankIdPtr != 0,
            OP_LOGE(nodeName, "tpRankId is invalid, NoTp mode only support 0, but got tpRankId=%ld.", *tpRankIdPtr),
            return ge::GRAPH_FAILED);
    }
    tilingData.moeDistributeCombineV2Info.epWorldSize = static_cast<uint32_t>(epWorldSize);
    tilingData.moeDistributeCombineV2Info.tpWorldSize = static_cast<uint32_t>(*tpWorldSizePtr);
    tilingData.moeDistributeCombineV2Info.epRankId = static_cast<uint32_t>(*epRankIdPtr);
    tilingData.moeDistributeCombineV2Info.tpRankId = static_cast<uint32_t>(*tpRankIdPtr);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetAttrAndSetTilingData(const gert::TilingContext *context,
    MoeDistributeCombineV2TilingData &tilingData, const char *nodeName, std::string &groupEp, std::string &groupTp,
    uint32_t &commQuantMode, const CombineV2Config& config, bool &isLayered)
{
    auto attrs = context->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE(nodeName, "attrs is null."), return ge::GRAPH_FAILED);

    if (!config.isMc2Context) {
        auto groupEpPtr = attrs->GetAttrPointer<char>(static_cast<int>(config.attrGroupEpIndex));
        OP_TILING_CHECK((groupEpPtr == nullptr) || (strnlen(groupEpPtr, MAX_GROUP_NAME_LENGTH) == 0) ||
            (strnlen(groupEpPtr, MAX_GROUP_NAME_LENGTH) == MAX_GROUP_NAME_LENGTH), OP_LOGE(nodeName, "groupEp is invalid."),
            return ge::GRAPH_FAILED);
        groupEp = string(groupEpPtr);
    }
    auto commQuantModePtr = attrs->GetAttrPointer<int64_t>(static_cast<int>((config.attrCommQuantModeIndex)));
    auto commAlgPtr = attrs->GetAttrPointer<char>(static_cast<int>((config.attrCommAlgIndex)));

    OP_TILING_CHECK(commQuantModePtr == nullptr, OP_LOGE(nodeName, "commQuantMode is null."), return ge::GRAPH_FAILED);
    isLayered = strcmp(commAlgPtr, "hierarchy") == 0;
    if (config.isMc2Context) {
        OP_TILING_CHECK(isLayered,
            OP_LOGE(nodeName, "commAlgPtr %s doesn't support comm with context.", commAlgPtr),
            return ge::GRAPH_FAILED);
    }
    OP_TILING_CHECK((*commQuantModePtr != 0) && (*commQuantModePtr != INT8_COMM_QUANT),
        OP_LOGE(nodeName, "commQuantMode only support 0(default) or 2(int8 comm quant), but got commQuantMode=%ld.",
        *commQuantModePtr), return ge::GRAPH_FAILED);

    commQuantMode = static_cast<uint32_t>(*commQuantModePtr);
    OP_TILING_CHECK(GetExpertsAttrAndSetTilingData(context, tilingData, nodeName, config, isLayered) == ge::GRAPH_FAILED,
        OP_LOGE(nodeName, "Getting experts attr failed."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(GetSharedAttrAndSetTilingData(context, tilingData, nodeName, config, isLayered) == ge::GRAPH_FAILED,
        OP_LOGE(nodeName, "Getting shared expert attr failed."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(GetTpAndEpAttrAndSetTilingData(context, tilingData, nodeName, config, isLayered, commQuantMode,
        groupTp) == ge::GRAPH_FAILED, OP_LOGE(nodeName, "Getting tp and ep attr failed."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckARNAttrAndSetTilingData(const gert::TilingContext *context,
    MoeDistributeCombineV2TilingData &tilingData, const char *nodeName, const CombineV2Config& config)
{
    auto attrs = context->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE(nodeName, "attrs is null."), return ge::GRAPH_FAILED);
    auto epsilonPtr = attrs->GetAttrPointer<float>(config.attrNormEpsIndex);
    OP_TILING_CHECK(epsilonPtr == nullptr, OP_LOGE(nodeName, "epsilonPtr is null."), return ge::GRAPH_FAILED);
    tilingData.moeDistributeCombineV2Info.epsilon = *epsilonPtr;
    auto expandXDesc = context->GetInputDesc(config.expandXIndex);
    OP_TILING_CHECK(expandXDesc == nullptr, OP_LOGE(nodeName, "expandxDesc is null."), return false);

    auto residualXDesc = context->GetInputDesc(config.residualXIndex);
    OP_TILING_CHECK(residualXDesc == nullptr, OP_LOGE(nodeName, "residualXDesc is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK((residualXDesc->GetDataType() != ge::DT_BF16),
        OP_LOGE(nodeName, "residualX dataType is invalid, dataType should be bfloat16, but is %d",
        static_cast<ge::DataType>(residualXDesc->GetDataType())), return ge::GRAPH_FAILED);
    auto gammaDesc = context->GetInputDesc(config.gammaIndex);
    OP_TILING_CHECK(gammaDesc == nullptr, OP_LOGE(nodeName, "gammaDesc is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK((gammaDesc->GetDataType() != ge::DT_BF16),
        OP_LOGE(nodeName, "gamma dataType is invalid, dataType should be bfloat16, but is %d",
        static_cast<ge::DataType>(gammaDesc->GetDataType())), return ge::GRAPH_FAILED);
        auto yDesc = context->GetOutputDesc(config.outputYIndex);
    OP_TILING_CHECK(yDesc == nullptr, OP_LOGE(nodeName, "yDesc is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK((yDesc->GetDataType() != expandXDesc->GetDataType()), OP_LOGE(nodeName,
        "yOut dataType is invalid, dataType should be equal to expandX dataType %d, but is %d",
        static_cast<ge::DataType>(expandXDesc->GetDataType()), static_cast<ge::DataType>(yDesc->GetDataType())),
        return ge::GRAPH_FAILED);
    auto rstdDesc = context->GetOutputDesc(config.outputRstdIndex);
    OP_TILING_CHECK(rstdDesc->GetDataType() != ge::DT_FLOAT,
        OP_LOGE(nodeName, "rstdOut dataType is invalid, dataType should be float, but got %d",
        static_cast<ge::DataType>(rstdDesc->GetDataType())), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static bool CheckInputTensorDimARN(const gert::TilingContext *context, const char *nodeName, const CombineV2Config& config)
{
    const gert::StorageShape *residualXStorageShape = context->GetInputShape(config.residualXIndex);
    OP_TILING_CHECK(residualXStorageShape == nullptr, OP_LOGE(nodeName, "residualX is null."), return false);
    OP_TILING_CHECK(residualXStorageShape->GetStorageShape().GetDimNum() != THREE_DIMS,
        OP_LOGE(nodeName, "residualX must be 3-dimension, but got %lu dim",
        residualXStorageShape->GetStorageShape().GetDimNum()), return false);

    const gert::StorageShape *gammaStorageShape = context->GetInputShape(config.gammaIndex);
    OP_TILING_CHECK(gammaStorageShape == nullptr, OP_LOGE(nodeName, "gamma is null."), return false);
    OP_TILING_CHECK(gammaStorageShape->GetStorageShape().GetDimNum() != ONE_DIM,
        OP_LOGE(nodeName, "gamma must be 1-dimension, but got %lu dim",
        gammaStorageShape->GetStorageShape().GetDimNum()), return false);

    return true;
}

static bool CheckInputTensorDim(const gert::TilingContext *context, const char *nodeName, const CombineV2Config& config)
{
    const gert::StorageShape *expandXStorageShape = context->GetInputShape(config.expandXIndex);
    OP_TILING_CHECK(expandXStorageShape == nullptr, OP_LOGE(nodeName, "expandX is null."), return false);
    OP_TILING_CHECK(expandXStorageShape->GetStorageShape().GetDimNum() != TWO_DIMS,
        OP_LOGE(nodeName, "expandX must be 2-dimension, but got %lu dim",
        expandXStorageShape->GetStorageShape().GetDimNum()), return false);
    OP_LOGD(nodeName, "expandX dim0 = %ld", expandXStorageShape->GetStorageShape().GetDim(0));
    OP_LOGD(nodeName, "expandX dim1 = %ld", expandXStorageShape->GetStorageShape().GetDim(1));

    const gert::StorageShape *expertIdsStorageShape = context->GetInputShape(config.expertIdsIndex);
    OP_TILING_CHECK(expertIdsStorageShape == nullptr, OP_LOGE(nodeName, "expertIds is null."), return false);
    OP_TILING_CHECK(expertIdsStorageShape->GetStorageShape().GetDimNum() != TWO_DIMS,
        OP_LOGE(nodeName, "expertIds must be 2-dimension, but got %lu dim",
        expertIdsStorageShape->GetStorageShape().GetDimNum()), return false);
    int64_t expertIdsDim0 = expertIdsStorageShape->GetStorageShape().GetDim(0);
    int64_t expertIdsDim1 = expertIdsStorageShape->GetStorageShape().GetDim(1);
    OP_LOGD(nodeName, "expertIds dim0 = %ld", expertIdsDim0);
    OP_LOGD(nodeName, "expertIds dim1 = %ld", expertIdsDim1);

    const gert::StorageShape *assistInfoStorageShape = context->GetInputShape(config.assistInfoIndex);
    OP_TILING_CHECK(assistInfoStorageShape == nullptr, OP_LOGE(nodeName, "assistInfoForCombine is null."), return false);
    OP_TILING_CHECK(assistInfoStorageShape->GetStorageShape().GetDimNum() != ONE_DIM,
        OP_LOGE(nodeName, "assistInfoForCombine must be 1-dimension, but got %lu dim",
        assistInfoStorageShape->GetStorageShape().GetDimNum()), return false);
    OP_LOGD(nodeName, "assistInfoForCombine dim0 = %ld", assistInfoStorageShape->GetStorageShape().GetDim(0));

    const gert::StorageShape *epSendCountsStorageShape = context->GetInputShape(config.epSendCountIndex);
    OP_TILING_CHECK(epSendCountsStorageShape == nullptr, OP_LOGE(nodeName, "epSendCounts is null."), return false);
    OP_TILING_CHECK(epSendCountsStorageShape->GetStorageShape().GetDimNum() != ONE_DIM,
        OP_LOGE(nodeName, "epSendCounts must be 1-dimension, but got %lu dim",
        epSendCountsStorageShape->GetStorageShape().GetDimNum()), return false);
    OP_LOGD(nodeName, "epSendCounts dim0 = %ld", epSendCountsStorageShape->GetStorageShape().GetDim(0));

    const gert::StorageShape *expertScalesStorageShape = context->GetInputShape(config.expertScalesIndex);
    OP_TILING_CHECK(expertScalesStorageShape == nullptr, OP_LOGE(nodeName, "expertScales is null."), return false);
    OP_TILING_CHECK(expertScalesStorageShape->GetStorageShape().GetDimNum() != TWO_DIMS,
        OP_LOGE(nodeName, "expertScales must be 2-dimension, but got %lu dim",
        expertScalesStorageShape->GetStorageShape().GetDimNum()), return false);
    OP_LOGD(nodeName, "expertScales dim0 = %ld", expertScalesStorageShape->GetStorageShape().GetDim(0));
    OP_LOGD(nodeName, "expertScales dim1 = %ld", expertScalesStorageShape->GetStorageShape().GetDim(1));

    return true;
}

static bool CheckOptionalScalesTensorDim(const gert::TilingContext *context, const char *nodeName,
    const CombineV2Config& config, const bool isLayered)
{
    if (isLayered) {
        const gert::StorageShape *expandScaleStorageShape = context->GetOptionalInputShape(EXPAND_SCALES_INDEX);
        const int64_t expandScaleDim = expandScaleStorageShape->GetStorageShape().GetDimNum();
        OP_TILING_CHECK(expandScaleStorageShape == nullptr, OP_LOGE(nodeName, "expandScales is null."), return false);
        OP_TILING_CHECK(expandScaleDim != ONE_DIM,
            OP_LOGE(nodeName, "expandScales must be 1-dimension, but got %lu dim", expandScaleDim), return false);
        OP_LOGD(nodeName, "expandScales dim0 = %ld", expandScaleStorageShape->GetStorageShape().GetDim(0));
    }

    const gert::StorageShape* constExpertAlpha1StorageShape =
        context->GetOptionalInputShape(config.constExpertAlpha1Index);
    if (constExpertAlpha1StorageShape != nullptr) {
        OP_TILING_CHECK(constExpertAlpha1StorageShape->GetStorageShape().GetDimNum() != TWO_DIMS,
            OP_LOGE(nodeName, "const_expert_alpha_1 must be 2-dimension, but got %lu dim",
                constExpertAlpha1StorageShape->GetStorageShape().GetDimNum()), return false);
    }

    const gert::StorageShape* constExpertAlpha2StorageShape = context->GetOptionalInputShape(config.constExpertAlpha2Index);
    if (constExpertAlpha2StorageShape != nullptr) {
        OP_TILING_CHECK(constExpertAlpha2StorageShape->GetStorageShape().GetDimNum() != TWO_DIMS,
            OP_LOGE(nodeName, "const_expert_alpha_2 must be 2-dimension, but got %lu dim",
                constExpertAlpha2StorageShape->GetStorageShape().GetDimNum()), return false);
    }

    const gert::StorageShape* constExpertVStorageShape = context->GetOptionalInputShape(config.constExpertVIndex);
    if (constExpertVStorageShape != nullptr) {
        OP_TILING_CHECK(constExpertVStorageShape->GetStorageShape().GetDimNum() != TWO_DIMS,
            OP_LOGE(nodeName, "const_expert_v must be 2-dimension, but got %lu dim",
                constExpertVStorageShape->GetStorageShape().GetDimNum()), return false);
    }

    const gert::StorageShape *activationScaleStorageShape = context->GetOptionalInputShape(config.activationScaleIndex);
    OP_TILING_CHECK(activationScaleStorageShape != nullptr, OP_LOGE(nodeName, "activationScale is not null."), return false);

    const gert::StorageShape *weightScaleStorageShape = context->GetOptionalInputShape(config.weightScaleIndex);
    OP_TILING_CHECK(weightScaleStorageShape != nullptr, OP_LOGE(nodeName, "weightScale is not null."), return false);

    const gert::StorageShape *sharedExpertX = context->GetOptionalInputShape(config.sharedExpertXIndex);
    if (sharedExpertX != nullptr) {
        auto attrs = context->GetAttrs();
        auto sharedExpertRankNumPtr = attrs->GetAttrPointer<int64_t>((config.attrSharedExpertRankNumIndex));
        OP_TILING_CHECK(*sharedExpertRankNumPtr != 0, OP_LOGE(nodeName, "sharedExpertX only support input None "\
            "when sharedExpertRankNum is non-zero."), return false);
        OP_TILING_CHECK(((sharedExpertX->GetStorageShape().GetDimNum() != TWO_DIMS) &&
                        (sharedExpertX->GetStorageShape().GetDimNum() != THREE_DIMS)),
                        OP_LOGE(nodeName, "sharedExpertX must be 2-dimension or 3-dimension, but got %lu dim",
                                sharedExpertX->GetStorageShape().GetDimNum()), return false);
    }

    return true;
}

static bool CheckOptionalInputTensorDim(const gert::TilingContext *context, const char *nodeName,
    const bool isActiveMask, const bool hasElasticInfo, const uint32_t tpWorldSize,
    const CombineV2Config& config, const bool isLayered)
{
    const gert::StorageShape* oriXStorageShape = context->GetOptionalInputShape(config.oriXIndex);
    if (oriXStorageShape != nullptr) {
        OP_TILING_CHECK(oriXStorageShape->GetStorageShape().GetDimNum() != TWO_DIMS,
            OP_LOGE(nodeName, "ori_x must be 2-dimension, but got %lu dim",
                oriXStorageShape->GetStorageShape().GetDimNum()), return false);
    }

    OP_TILING_CHECK(!CheckOptionalScalesTensorDim(context, nodeName, config, isLayered),
        OP_LOGE(nodeName, "Param shape of optional scales input tensor is invalid"), return false);

    if (tpWorldSize == TP_WORLD_SIZE_TWO) {
        const gert::StorageShape *tpSendCountsStorageShape = context->GetOptionalInputShape(config.tpSendCountsIndex);
        OP_TILING_CHECK(tpSendCountsStorageShape == nullptr, OP_LOGE(nodeName, "tpSendCounts is null."), return false);
        OP_TILING_CHECK(tpSendCountsStorageShape->GetStorageShape().GetDimNum() != ONE_DIM,
            OP_LOGE(nodeName, "tpSendCounts must be 1-dimension, but got %lu dim",
            tpSendCountsStorageShape->GetStorageShape().GetDimNum()), return false);
        OP_LOGD(nodeName, "tpSendCounts dim0 = %ld", tpSendCountsStorageShape->GetStorageShape().GetDim(0));
    }

    if (isActiveMask) {
        const gert::StorageShape *xActiveMaskStorageShape = context->GetOptionalInputShape(config.xActiveMaskIndex);
        OP_TILING_CHECK(xActiveMaskStorageShape == nullptr, OP_LOGE(nodeName, "xActiveMask is null."), return false);
        const int64_t xActiveMaskDimNums = xActiveMaskStorageShape->GetStorageShape().GetDimNum();
        OP_TILING_CHECK(((xActiveMaskDimNums != ONE_DIM) && (xActiveMaskDimNums != TWO_DIMS)),
            OP_LOGE(nodeName, "xActiveMask must be 1-dimension or 2-dimension, but got %ld dim",
            xActiveMaskDimNums), return false);
    }
    if (hasElasticInfo) {
        const gert::StorageShape *elasticInfoStorageShape = context->GetOptionalInputShape(config.elasticInfoIndex);
        OP_TILING_CHECK(elasticInfoStorageShape == nullptr, OP_LOGE(nodeName, "elasticInfo is null."), return false);
        OP_TILING_CHECK(elasticInfoStorageShape->GetStorageShape().GetDimNum() != ONE_DIM,
            OP_LOGE(nodeName, "elasticInfo dim must be 1, but current dim num is %lu.",
            elasticInfoStorageShape->GetStorageShape().GetDimNum()), return false);
        OP_LOGD(nodeName, "elasticInfo dim0 = %ld", elasticInfoStorageShape->GetStorageShape().GetDim(0));
    }

    const gert::StorageShape *groupListStorageShape = context->GetOptionalInputShape(config.groupListIndex);
    OP_TILING_CHECK(groupListStorageShape != nullptr, OP_LOGE(nodeName, "groupList is not null."), return false);

    return true;
}

static bool CheckOutputTensorDimARN(const gert::TilingContext *context, const char *nodeName, const CombineV2Config& config)
{
    const gert::StorageShape *yStorageShape = context->GetOutputShape(config.outputYIndex);
    OP_TILING_CHECK(yStorageShape == nullptr, OP_LOGE(nodeName, "yOut is null."), return false);
    OP_TILING_CHECK(yStorageShape->GetStorageShape().GetDimNum() != THREE_DIMS,
        OP_LOGE(nodeName, "yOut must be 3-dimension, but got %lu dim", yStorageShape->GetStorageShape().GetDimNum()),
        return false);
    OP_LOGD(nodeName, "yOut dim0 = %ld", yStorageShape->GetStorageShape().GetDim(0));
    OP_LOGD(nodeName, "yOut dim1 = %ld", yStorageShape->GetStorageShape().GetDim(1));
    OP_LOGD(nodeName, "yOut dim2 = %ld", yStorageShape->GetStorageShape().GetDim(TWO_DIMS));
    
    const gert::StorageShape *rstdStorageShape = context->GetOutputShape(config.outputRstdIndex);
    OP_TILING_CHECK(rstdStorageShape == nullptr, OP_LOGE(nodeName, "rstdOut is null."), return false);
    OP_TILING_CHECK(rstdStorageShape->GetStorageShape().GetDimNum() != THREE_DIMS,
        OP_LOGE(nodeName, "rstdOut must be 3-dimension, but got %lu dim", rstdStorageShape->GetStorageShape().GetDimNum()),
        return false);
    OP_LOGD(nodeName, "rstdOut dim0 = %ld", rstdStorageShape->GetStorageShape().GetDim(0));
    OP_LOGD(nodeName, "rstdOut dim1 = %ld", rstdStorageShape->GetStorageShape().GetDim(1));
    OP_LOGD(nodeName, "rstdOut dim2 = %ld", rstdStorageShape->GetStorageShape().GetDim(TWO_DIMS));
    
    const gert::StorageShape *xStorageShape = context->GetOutputShape(config.outputXIndex);
    OP_TILING_CHECK(xStorageShape == nullptr, OP_LOGE(nodeName, "xOut is null."), return false);
    OP_TILING_CHECK(xStorageShape->GetStorageShape().GetDimNum() != THREE_DIMS,
        OP_LOGE(nodeName, "xOut must be 3-dimension, but got %lu dim", xStorageShape->GetStorageShape().GetDimNum()),
        return false);
    OP_LOGD(nodeName, "xOut dim0 = %ld", xStorageShape->GetStorageShape().GetDim(0));
    OP_LOGD(nodeName, "xOut dim1 = %ld", xStorageShape->GetStorageShape().GetDim(1));
    OP_LOGD(nodeName, "xOut dim2 = %ld", xStorageShape->GetStorageShape().GetDim(TWO_DIMS));

    return true;
}

static bool CheckOutputTensorDim(const gert::TilingContext *context, const char *nodeName, const CombineV2Config& config,
    int64_t expertIdsDim0, int64_t expandXDim1)
{
    const gert::StorageShape *xStorageShape = context->GetOutputShape(config.outputXIndex);
    OP_TILING_CHECK(xStorageShape == nullptr, OP_LOGE(nodeName, "x is null."), return false);
    OP_TILING_CHECK(xStorageShape->GetStorageShape().GetDimNum() != TWO_DIMS,
        OP_LOGE(nodeName, "x must be 2-dimension, but got %lu dim", xStorageShape->GetStorageShape().GetDimNum()),
        return false);
    int64_t xDim0 = xStorageShape->GetStorageShape().GetDim(0);
    int64_t xDim1 = xStorageShape->GetStorageShape().GetDim(1);
    OP_TILING_CHECK(xDim0 != expertIdsDim0, OP_LOGE(nodeName,
        "x's dim0 not equal to bs, bs = %ld, x's dim0 = %ld", expertIdsDim0, xDim0), return false);
    OP_TILING_CHECK(xDim1 != expandXDim1, OP_LOGE(nodeName,
        "x's dim1 not equal to h, x's dim1 = %ld, h = %ld", xDim1, expandXDim1), return false);

    return true;
}

static bool CheckTensorDim(gert::TilingContext *context, const char *nodeName, const bool isActiveMask,
                           const bool hasElasticInfo, uint32_t tpWorldSize,
                           const CombineV2Config& config, const bool isLayered)
{
    OP_TILING_CHECK(!CheckInputTensorDim(context, nodeName, config),
        OP_LOGE(nodeName, "param shape of input tensor is invalid"), return false);

    OP_TILING_CHECK(!CheckOptionalInputTensorDim(context, nodeName, isActiveMask, hasElasticInfo, tpWorldSize, config, isLayered),
        OP_LOGE(nodeName, "param shape of optional input tensor is invalid"), return false);

    return true;
}

static bool CheckExpertTensorDataType(const gert::TilingContext *context, const char *nodeName, const CombineV2Config& config)
{
    auto expandXDesc = context->GetInputDesc(config.expandXIndex);
    auto oriXDesc = context->GetOptionalInputDesc(config.oriXIndex);
    if (oriXDesc != nullptr) {
        OP_TILING_CHECK((oriXDesc->GetDataType() != expandXDesc->GetDataType()),
            OP_LOGE(nodeName, "ori_x dataType is invalid, dataType should be same as expandX dataType as %s, but now is %s",
                Ops::Base::ToString(expandXDesc->GetDataType()).c_str(), Ops::Base::ToString(oriXDesc->GetDataType()).c_str()), return false);
    }

    auto constExpertAlpha1Desc = context->GetOptionalInputDesc(config.constExpertAlpha1Index);
    if (constExpertAlpha1Desc != nullptr) {
        OP_TILING_CHECK((constExpertAlpha1Desc->GetDataType() != expandXDesc->GetDataType()),
            OP_LOGE(
                nodeName, "const_expert_alpha_1 dataType is invalid, dataType should be same as expandX dataType as %s, but now is %s",
                Ops::Base::ToString(expandXDesc->GetDataType()).c_str(),
                Ops::Base::ToString(constExpertAlpha1Desc->GetDataType()).c_str()), return false);
    }

    auto constExpertAlpha2Desc = context->GetOptionalInputDesc(config.constExpertAlpha2Index);
    if (constExpertAlpha2Desc != nullptr) {
        OP_TILING_CHECK(
            (constExpertAlpha2Desc->GetDataType() != expandXDesc->GetDataType()),
            OP_LOGE(
                nodeName, "const_expert_alpha_2 dataType is invalid, dataType should be same as expandX dataType as %s, but now is %s",
                Ops::Base::ToString(expandXDesc->GetDataType()).c_str(),
                Ops::Base::ToString(constExpertAlpha2Desc->GetDataType()).c_str()), return false);
    }

    auto constExpertVDesc = context->GetOptionalInputDesc(config.constExpertVIndex);
    if (constExpertVDesc != nullptr) {
        OP_TILING_CHECK((constExpertVDesc->GetDataType() != expandXDesc->GetDataType()),
            OP_LOGE(nodeName, "const_expert_v dataType is invalid, dataType should be same as expandX dataType as %s, but now is %s",
                Ops::Base::ToString(expandXDesc->GetDataType()).c_str(),
                Ops::Base::ToString(constExpertVDesc->GetDataType()).c_str()), return false);
    }

    auto expertIdsDesc = context->GetInputDesc(config.expertIdsIndex);
    OP_TILING_CHECK(expertIdsDesc == nullptr, OP_LOGE(nodeName, "expertIdsDesc is null."), return false);
    OP_TILING_CHECK((expertIdsDesc->GetDataType() != ge::DT_INT32), OP_LOGE(nodeName, "expertIds dataType is invalid, "
        "dataType should be int32, but is %s", Ops::Base::ToString(expertIdsDesc->GetDataType()).c_str()), return false);
    auto sharedExpertXDesc = context->GetOptionalInputDesc(config.sharedExpertXIndex);
    if (sharedExpertXDesc != nullptr) {
        OP_TILING_CHECK(sharedExpertXDesc->GetDataType() != expandXDesc->GetDataType(),
            OP_LOGE(nodeName, "sharedExpertX dataType should be the same as expandX dataType, but got sharedExpertX"
            "dataType %s, expandX dataType %s.", Ops::Base::ToString(sharedExpertXDesc->GetDataType()).c_str(),
            Ops::Base::ToString(expandXDesc->GetDataType()).c_str()), return false);
    }
    auto expertScalesDesc = context->GetInputDesc(config.expertScalesIndex);
    OP_TILING_CHECK(expertScalesDesc == nullptr, OP_LOGE(nodeName, "expertScalesDesc is null."), return false);
    OP_TILING_CHECK((expertScalesDesc->GetDataType() != ge::DT_FLOAT),
        OP_LOGE(nodeName, "expertScales dataType is invalid, dataType should be float, but is %s",
        Ops::Base::ToString(expertScalesDesc->GetDataType()).c_str()), return false);
    return true;
}

// 校验数据类型
static bool CheckTensorDataType(const gert::TilingContext *context, const char *nodeName, const bool isActiveMask,
                                const bool hasElasticInfo, const uint32_t tpWorldSize,
                                const CombineV2Config& config)
{
    auto expandXDesc = context->GetInputDesc(config.expandXIndex);
    OP_TILING_CHECK(expandXDesc == nullptr, OP_LOGE(nodeName, "expandxDesc is null."), return false);
    OP_TILING_CHECK((expandXDesc->GetDataType() != ge::DT_BF16) && (expandXDesc->GetDataType() != ge::DT_FLOAT16),
        OP_LOGE(nodeName, "expandX dataType is invalid, dataType should be bf16 or float16, but is %s",
        Ops::Base::ToString(expandXDesc->GetDataType()).c_str()), return false);
    auto assistInfoDesc = context->GetInputDesc(config.assistInfoIndex);
    OP_TILING_CHECK(assistInfoDesc == nullptr, OP_LOGE(nodeName, "assistInfoDesc is null."), return false);
    OP_TILING_CHECK((assistInfoDesc->GetDataType() != ge::DT_INT32), OP_LOGE(nodeName, "assistInfoForCombine dataType is invalid,"
        " dataType should be int32, but is %s", Ops::Base::ToString(assistInfoDesc->GetDataType()).c_str()), return false);
    auto epSendCountsDesc = context->GetInputDesc(config.epSendCountIndex);
    OP_TILING_CHECK(epSendCountsDesc == nullptr, OP_LOGE(nodeName, "epSendCountsDesc is null."), return false);
    OP_TILING_CHECK((epSendCountsDesc->GetDataType() != ge::DT_INT32),
        OP_LOGE(nodeName, "epSendCounts dataType is invalid, dataType should be int32, but is %s",
        Ops::Base::ToString(epSendCountsDesc->GetDataType()).c_str()), return false);
    if (tpWorldSize == TP_WORLD_SIZE_TWO) {
        auto tpSendCountsDesc = context->GetOptionalInputDesc(config.tpSendCountsIndex);
        OP_TILING_CHECK(tpSendCountsDesc == nullptr, OP_LOGE(nodeName, "tpSendCountsDesc is null."), return false);
        OP_TILING_CHECK((tpSendCountsDesc->GetDataType() != ge::DT_INT32),
            OP_LOGE(nodeName, "tpSendCounts dataType is invalid, dataType should be int32, but is %s",
            Ops::Base::ToString(tpSendCountsDesc->GetDataType()).c_str()), return false);
    }

    if (isActiveMask) {
        auto xActiveMaskDesc = context->GetOptionalInputDesc(config.xActiveMaskIndex);
        OP_TILING_CHECK(xActiveMaskDesc == nullptr, OP_LOGE(nodeName, "xActiveMaskDesc is null."), return false);
        OP_TILING_CHECK(xActiveMaskDesc->GetDataType() != ge::DT_BOOL, OP_LOGE(nodeName, "xActiveMask dataType is invalid,"
            " dataType should be bool, but is %s.", Ops::Base::ToString(xActiveMaskDesc->GetDataType()).c_str()), return false);
    }
    if (hasElasticInfo) {
        auto elasticInfoDesc = context->GetOptionalInputDesc(config.elasticInfoIndex);
        OP_TILING_CHECK(elasticInfoDesc == nullptr, OP_LOGE(nodeName, "elasticInfoDesc is null."), return false);
        OP_TILING_CHECK(elasticInfoDesc->GetDataType() != ge::DT_INT32, OP_LOGE(nodeName,
            "elasticInfoDesc dataType is invalid, dataType should be int32, but is %s.",
            Ops::Base::ToString(elasticInfoDesc->GetDataType()).c_str()), return false);
    }
    auto xDesc = context->GetOutputDesc(config.outputXIndex);
    OP_TILING_CHECK(xDesc == nullptr, OP_LOGE(nodeName, "xDesc is null."), return false);
    OP_TILING_CHECK((xDesc->GetDataType() != expandXDesc->GetDataType()), OP_LOGE(nodeName,
        "x dataType is invalid, dataType should be equal to expandX dataType %s, but is %s",
        Ops::Base::ToString(expandXDesc->GetDataType()).c_str(), Ops::Base::ToString(xDesc->GetDataType()).c_str()),
        return false);
    return true;
}

static bool CheckZeroComputeExpertTensorFormat(const gert::TilingContext *context, const char *nodeName, const CombineV2Config& config)
{
    auto oriXDesc = context->GetOptionalInputDesc(config.oriXIndex);
    if (oriXDesc != nullptr) {
        OP_TILING_CHECK(
            static_cast<ge::Format>(ge::GetPrimaryFormat(oriXDesc->GetStorageFormat())) == ge::FORMAT_FRACTAL_NZ,
            OP_LOGE(nodeName, "ori_x Format is invalid"), return false);
    }

    auto constExpertAlpha1Desc = context->GetOptionalInputDesc(config.constExpertAlpha1Index);
    if (constExpertAlpha1Desc != nullptr) {
        OP_TILING_CHECK(
            static_cast<ge::Format>(ge::GetPrimaryFormat(constExpertAlpha1Desc->GetStorageFormat())) ==
                ge::FORMAT_FRACTAL_NZ,
            OP_LOGE(nodeName, "const_expert_alpha_1 Format is invalid"), return false);
    }

    auto constExpertAlpha2Desc = context->GetOptionalInputDesc(config.constExpertAlpha2Index);
    if (constExpertAlpha2Desc != nullptr) {
        OP_TILING_CHECK(
            static_cast<ge::Format>(ge::GetPrimaryFormat(constExpertAlpha2Desc->GetStorageFormat())) ==
                ge::FORMAT_FRACTAL_NZ,
            OP_LOGE(nodeName, "const_expert_alpha_2 Format is invalid"), return false);
    }

    auto constExpertVDesc = context->GetOptionalInputDesc(config.constExpertVIndex);
    if (constExpertVDesc != nullptr) {
        OP_TILING_CHECK(
            static_cast<ge::Format>(ge::GetPrimaryFormat(constExpertVDesc->GetStorageFormat())) ==
                ge::FORMAT_FRACTAL_NZ,
            OP_LOGE(nodeName, "const_expert_v Format is invalid"), return false);
    }
    return true;
}

static bool CheckTensorFormat(const gert::TilingContext *context, const char *nodeName, const bool isActiveMask,
                              const bool hasElasticInfo, const uint32_t tpWorldSize,
                              const CombineV2Config& config)
{
    OP_TILING_CHECK(!CheckZeroComputeExpertTensorFormat(context, nodeName, config),
                    OP_LOGE(nodeName, "Zero_compute_expert Format is invalid"), return ge::GRAPH_FAILED);
    auto expandXDesc = context->GetInputDesc(config.expandXIndex);
    OP_TILING_CHECK(expandXDesc == nullptr, OP_LOGE(nodeName, "expandxDesc is null."), return false);
    OP_TILING_CHECK(static_cast<ge::Format>(ge::GetPrimaryFormat(expandXDesc->GetStorageFormat())) ==
        ge::FORMAT_FRACTAL_NZ, OP_LOGE(nodeName, "expandXFormat is invalid"), return false);

    auto expertIdsDesc = context->GetInputDesc(config.expertIdsIndex);
    OP_TILING_CHECK(expertIdsDesc == nullptr, OP_LOGE(nodeName, "expertIdsDesc is null."), return false);
    OP_TILING_CHECK(static_cast<ge::Format>(ge::GetPrimaryFormat(expertIdsDesc->GetStorageFormat())) ==
        ge::FORMAT_FRACTAL_NZ, OP_LOGE(nodeName, "expertIdsFormat is invalid"), return false);

    auto assistInfoDesc = context->GetInputDesc(config.assistInfoIndex);
    OP_TILING_CHECK(assistInfoDesc == nullptr, OP_LOGE(nodeName, "assistInfoDesc is null."), return false);
    OP_TILING_CHECK(static_cast<ge::Format>(ge::GetPrimaryFormat(assistInfoDesc->GetStorageFormat())) ==
        ge::FORMAT_FRACTAL_NZ, OP_LOGE(nodeName, "assistInfoFormat is invalid"), return false);

    auto epSendCountsDesc = context->GetInputDesc(config.epSendCountIndex);
    OP_TILING_CHECK(epSendCountsDesc == nullptr, OP_LOGE(nodeName, "epSendCountsDesc is null."), return false);
    OP_TILING_CHECK(static_cast<ge::Format>(ge::GetPrimaryFormat(epSendCountsDesc->GetStorageFormat())) ==
        ge::FORMAT_FRACTAL_NZ, OP_LOGE(nodeName, "epSendCountsFormat is invalid"), return false);

    if (tpWorldSize == TP_WORLD_SIZE_TWO) {
        auto tpSendCountsDesc = context->GetOptionalInputDesc(config.tpSendCountsIndex);
        OP_TILING_CHECK(tpSendCountsDesc == nullptr, OP_LOGE(nodeName, "tpSendCountsDesc is null."), return false);
        OP_TILING_CHECK(static_cast<ge::Format>(ge::GetPrimaryFormat(tpSendCountsDesc->GetStorageFormat())) ==
            ge::FORMAT_FRACTAL_NZ, OP_LOGE(nodeName, "tpSendCountsFormat is invalid"), return false);
    }

    auto expertScalesDesc = context->GetInputDesc(config.expertScalesIndex);
    OP_TILING_CHECK(expertScalesDesc == nullptr, OP_LOGE(nodeName, "expertScalesDesc is null."), return false);
    OP_TILING_CHECK(static_cast<ge::Format>(ge::GetPrimaryFormat(expertScalesDesc->GetStorageFormat())) ==
        ge::FORMAT_FRACTAL_NZ, OP_LOGE(nodeName, "expertScalesFormat is invalid"), return false);

    if (isActiveMask) {
        auto xActiveMaskDesc = context->GetOptionalInputDesc(config.xActiveMaskIndex);
        OP_TILING_CHECK(xActiveMaskDesc == nullptr, OP_LOGE(nodeName, "xActiveMaskDesc is null."), return false);
        OP_TILING_CHECK(static_cast<ge::Format>(ge::GetPrimaryFormat(xActiveMaskDesc->GetStorageFormat())) ==
            ge::FORMAT_FRACTAL_NZ, OP_LOGE(nodeName, "xActiveMaskFormat is invalid."), return false);
    }
    if (hasElasticInfo) {
        auto elasticInfoDesc = context->GetOptionalInputDesc(config.elasticInfoIndex);
        OP_TILING_CHECK(elasticInfoDesc == nullptr, OP_LOGE(nodeName, "elasticInfoDesc is null."), return false);
        OP_TILING_CHECK(static_cast<ge::Format>(ge::GetPrimaryFormat(elasticInfoDesc->GetStorageFormat())) ==
            ge::FORMAT_FRACTAL_NZ, OP_LOGE(nodeName, "elasticInfo format is invalid."), return false);
    }
    auto sharedExpertXDesc = context->GetOptionalInputDesc(config.sharedExpertXIndex);
    OP_TILING_CHECK((sharedExpertXDesc != nullptr) &&
        (static_cast<ge::Format>(ge::GetPrimaryFormat(sharedExpertXDesc->GetStorageFormat())) ==
        ge::FORMAT_FRACTAL_NZ), OP_LOGE(nodeName, "sharedExpertXFormat is invalid."), return false);

    auto xDesc = context->GetOutputDesc(config.outputXIndex);
    OP_TILING_CHECK(xDesc == nullptr, OP_LOGE(nodeName, "xDesc is null."), return false);
    OP_TILING_CHECK(static_cast<ge::Format>(ge::GetPrimaryFormat(xDesc->GetStorageFormat())) == ge::FORMAT_FRACTAL_NZ,
                    OP_LOGE(nodeName, "xFormat is invalid"), return false);

    return true;
}

static bool CheckTensorShapeARN(const gert::TilingContext *context,
    const char *nodeName, const CombineV2Config& config, int64_t expertIdsDim0, int64_t expandXDim1)
{
    // 校验residualX的维度
    const gert::StorageShape *residualXShape = context->GetOptionalInputShape(config.residualXIndex);
    if (residualXShape != nullptr) {
        int64_t residualXDimBs = residualXShape->GetStorageShape().GetDim(0);
        int64_t residualXDim1 = residualXShape->GetStorageShape().GetDim(1);
        int64_t residualXDimH = residualXShape->GetStorageShape().GetDim(TWO_DIMS);
        OP_TILING_CHECK(residualXDimBs != expertIdsDim0 || residualXDim1 != RESIDUAL_X_DIM2_SIZE ||
                residualXDimH != expandXDim1,
                OP_LOGE(nodeName, "Shape mismatch: residualX's dim0=%ld (bs=%ld), dim1=%ld (expected=%ld), dim2=%ld (h=%ld)",
                    residualXDimBs, expertIdsDim0, residualXDim1, RESIDUAL_X_DIM2_SIZE, residualXDimH, expandXDim1),
                return false);
    }
    // 校验gamma的维度
    const gert::StorageShape *gammaShape = context->GetOptionalInputShape(config.gammaIndex);
    if (gammaShape != nullptr) {
        int64_t gammaDimH = gammaShape->GetStorageShape().GetDim(0);
        OP_TILING_CHECK(gammaDimH != expandXDim1, OP_LOGE(nodeName,
            "gamma's dim0 not equal to h, gamma's dim0 = %ld, h = %ld",
            gammaDimH, expandXDim1), return false);
    }
    // 校验y的维度
    const gert::StorageShape *yStorageShape = context->GetOutputShape(config.outputYIndex);
    int64_t yDim0 = yStorageShape->GetStorageShape().GetDim(0);
    int64_t yDim1 = yStorageShape->GetStorageShape().GetDim(1);
    int64_t yDim2 = yStorageShape->GetStorageShape().GetDim(TWO_DIMS);
    OP_TILING_CHECK(yDim0 != expertIdsDim0 || yDim1 != 1 || yDim2 != expandXDim1,
                OP_LOGE(nodeName, "Shape mismatch: y's dim0=%ld (bs=%ld), dim1=%ld (expected=1), dim2=%ld (h=%ld)",
                    yDim0, expertIdsDim0, yDim1, yDim2, expandXDim1), return false);
    // 校验rstd的维度
    const gert::StorageShape *rstdStorageShape = context->GetOutputShape(config.outputRstdIndex);
    int64_t rstdDim0 = rstdStorageShape->GetStorageShape().GetDim(0);
    int64_t rstdDim1 = rstdStorageShape->GetStorageShape().GetDim(1);
    int64_t rstdDim2 = rstdStorageShape->GetStorageShape().GetDim(TWO_DIMS);
    OP_TILING_CHECK(rstdDim0 != expertIdsDim0 || rstdDim1 != 1 || rstdDim2 != 1,
                OP_LOGE(nodeName,
                "Shape mismatch: rstd's dim0=%ld (bs=%ld), dim1=%ld (expected=1), dim2=%ld (expected=1)",
                    rstdDim0, expertIdsDim0, rstdDim1, rstdDim2), return false);
    // 校验x的维度
    const gert::StorageShape *xStorageShape = context->GetOutputShape(config.outputXIndex);
    int64_t xDim0 = xStorageShape->GetStorageShape().GetDim(0);
    int64_t xDim1 = xStorageShape->GetStorageShape().GetDim(1);
    int64_t xDim2 = xStorageShape->GetStorageShape().GetDim(TWO_DIMS);
    OP_TILING_CHECK(xDim0 != expertIdsDim0 || xDim1 != 1 || xDim2 != expandXDim1,
                OP_LOGE(nodeName, "Shape mismatch: x's dim0=%ld (bs=%ld), dim1=%ld (expected=1), dim2=%ld (h=%ld)",
                    xDim0, expertIdsDim0, xDim1, xDim2, expandXDim1), return false);
    return true;
}

static bool CheckZeroComputeExpertsTensorShape(const gert::TilingContext *context, MoeDistributeCombineV2TilingData &tilingData,
    const char *nodeName, const CombineV2Config& config, const uint32_t expertIdsDim0)
{
    const gert::StorageShape* oriXShape = context->GetOptionalInputShape(config.oriXIndex);
    int64_t expandXDim1 = (context->GetInputShape(config.expandXIndex))->GetStorageShape().GetDim(1);
    if (oriXShape != nullptr) {
        int64_t oriXDim0 = oriXShape->GetStorageShape().GetDim(0);
        int64_t oriXDim1 = oriXShape->GetStorageShape().GetDim(1);
        OP_TILING_CHECK(oriXDim0 != expertIdsDim0, OP_LOGE(nodeName,
            "ori_x's dim0 not equal to bs, ori_x's dim0 = %ld, bs = %ld", oriXDim0, expertIdsDim0), return false);
        OP_TILING_CHECK(oriXDim1 != expandXDim1, OP_LOGE(nodeName,
            "ori_x's dim1 not equal to h, ori_x's dim1 = %ld, h = %ld", oriXDim1, expandXDim1), return false);
    }

    const gert::StorageShape* constExpertAlpha1Shape = context->GetOptionalInputShape(config.constExpertAlpha1Index);
    if (constExpertAlpha1Shape != nullptr) {
        int64_t constExpertAlpha1Dim0 = constExpertAlpha1Shape->GetStorageShape().GetDim(0);
        int64_t constExpertAlpha1Dim1 = constExpertAlpha1Shape->GetStorageShape().GetDim(1);
        OP_TILING_CHECK(
            constExpertAlpha1Dim0 != static_cast<int64_t>(tilingData.moeDistributeCombineV2Info.constExpertNum),
            OP_LOGE(nodeName, "const_expert_alpha_1's dim0 not equal to const_expert_num,"
                "const_expert_alpha_1's dim0 = %ld, const_expert_num = %u",
                constExpertAlpha1Dim0, tilingData.moeDistributeCombineV2Info.constExpertNum), return false);
        OP_TILING_CHECK(constExpertAlpha1Dim1 != expandXDim1, OP_LOGE(nodeName, "const_expert_alpha_1's dim1 not equal to h,"
                "const_expert_alpha_1's dim1 = %ld, h = %ld", constExpertAlpha1Dim1, expandXDim1), return false);
    }

    const gert::StorageShape* constExpertAlpha2Shape = context->GetOptionalInputShape(config.constExpertAlpha2Index);
    if (constExpertAlpha2Shape != nullptr) {
        int64_t constExpertAlpha2Dim0 = constExpertAlpha2Shape->GetStorageShape().GetDim(0);
        int64_t constExpertAlpha2Dim1 = constExpertAlpha2Shape->GetStorageShape().GetDim(1);
        OP_TILING_CHECK(
            constExpertAlpha2Dim0 != static_cast<int64_t>(tilingData.moeDistributeCombineV2Info.constExpertNum),
            OP_LOGE(nodeName, "const_expert_alpha_2's dim0 not equal to const_expert_num"
                "const_expert_alpha_2's dim0 = %ld, const_expert_num = %u",
                constExpertAlpha2Dim0, tilingData.moeDistributeCombineV2Info.constExpertNum), return false);
        OP_TILING_CHECK(constExpertAlpha2Dim1 != expandXDim1,
            OP_LOGE(nodeName, "const_expert_alpha_2's dim1 not equal to h, const_expert_alpha_2's dim1 = %ld, h = %ld",
                constExpertAlpha2Dim1, expandXDim1), return false);
    }

    const gert::StorageShape* constExpertVShape = context->GetOptionalInputShape(config.constExpertVIndex);
    if (constExpertVShape != nullptr) {
        int64_t constExpertVDim0 = constExpertVShape->GetStorageShape().GetDim(0);
        int64_t constExpertVDim1 = constExpertVShape->GetStorageShape().GetDim(1);
        OP_TILING_CHECK(
            constExpertVDim0 != static_cast<int64_t>(tilingData.moeDistributeCombineV2Info.constExpertNum), OP_LOGE(nodeName,
                "const_expert_v's dim0 not equal to const_expert_num, const_expert_v's dim0 = %ld, const_expert_num = %u.",
                constExpertVDim0, tilingData.moeDistributeCombineV2Info.constExpertNum), return false);
        OP_TILING_CHECK(constExpertVDim1 != expandXDim1,
            OP_LOGE(nodeName, "const_expert_v's dim1 not equal to h, const_expert_v's dim1 = %ld, h = %ld.",
                constExpertVDim1, expandXDim1), return false);
    }
    return true;
}

static bool CheckExpertsTensorShape(const gert::TilingContext *context, MoeDistributeCombineV2TilingData &tilingData,
    const char *nodeName, const CombineV2Config& config, const uint32_t expertIdsDim0, const uint32_t expertIdsDim1)
{
    int64_t moeExpertNum = static_cast<int64_t>(tilingData.moeDistributeCombineV2Info.moeExpertNum);
    int64_t zeroExpertNum = static_cast<int64_t>(tilingData.moeDistributeCombineV2Info.zeroExpertNum);
    int64_t copyExpertNum = static_cast<int64_t>(tilingData.moeDistributeCombineV2Info.copyExpertNum);
    int64_t constExpertNum = static_cast<int64_t>(tilingData.moeDistributeCombineV2Info.constExpertNum);
    OP_TILING_CHECK((expertIdsDim1 <= 0) || (expertIdsDim1 > K_MAX || (expertIdsDim1 > moeExpertNum
        + zeroExpertNum + copyExpertNum + constExpertNum)),
        OP_LOGE(nodeName, "expertIds's dim1(K) should be in (0, min(%ld, moeExpertNum"
        " + zeroExpertNum + copyExpertNum + constExpertNum = %ld)], "
        "but got expertIds's dim1=%ld.", K_MAX, moeExpertNum
        + zeroExpertNum + copyExpertNum + constExpertNum, expertIdsDim1), return false);
    tilingData.moeDistributeCombineV2Info.k = static_cast<uint32_t>(expertIdsDim1);

    // 校验expertScales的维度
    const gert::StorageShape *expertScalesStorageShape = context->GetInputShape(config.expertScalesIndex);
    int64_t expertScalesDim0 = expertScalesStorageShape->GetStorageShape().GetDim(0);
    int64_t expertScalesDim1 = expertScalesStorageShape->GetStorageShape().GetDim(1);
    OP_TILING_CHECK(expertScalesDim0 != expertIdsDim0,
        OP_LOGE(nodeName, "expertScales's dim0 not equal to bs, expertScales's dim0 = %ld, bs = %ld",
        expertScalesDim0, expertIdsDim0), return false);
    OP_TILING_CHECK(expertScalesDim1 != expertIdsDim1, OP_LOGE(nodeName,
        "expertScales's dim1 not equal to k, expertScales's dim1 = %ld, k = %ld",
        expertScalesDim1, expertIdsDim1), return false);

    // 校验sharedExpertX的维度
    const gert::StorageShape *sharedExpertXShape = context->GetOptionalInputShape(config.sharedExpertXIndex);
    tilingData.moeDistributeCombineV2Info.hasSharedExpertX = (sharedExpertXShape != nullptr);
    if (sharedExpertXShape != nullptr) {
        int64_t sharedExpertXDim0 = sharedExpertXShape->GetStorageShape().GetDim(0);
        int64_t sharedExpertXDim1 = sharedExpertXShape->GetStorageShape().GetDim(1);
        int64_t expandXDim1 = (context->GetInputShape(config.expandXIndex))->GetStorageShape().GetDim(1);
        if (sharedExpertXShape->GetStorageShape().GetDimNum() == TWO_DIMS) {
            OP_TILING_CHECK(sharedExpertXDim0 != expertIdsDim0,
                OP_LOGE(nodeName, "sharedExpertX's dim0 not equal to bs, sharedExpertX's dim0 = %ld, bs = %ld",
                sharedExpertXDim0, expertIdsDim0), return false);
            OP_TILING_CHECK(sharedExpertXDim1 != expandXDim1, OP_LOGE(nodeName,
                "sharedExpertX's dim1 not equal to h, sharedExpertX's dim1 = %ld, h = %ld",
                sharedExpertXDim1, expandXDim1), return false);
        } else {
            int64_t sharedExpertXDim2 = sharedExpertXShape->GetStorageShape().GetDim(TWO_DIMS);
            OP_TILING_CHECK(sharedExpertXDim0 * sharedExpertXDim1 != expertIdsDim0,
                OP_LOGE(nodeName, "sharedExpertX's dim0 * sharedExpertX's dim1 not equal to bs, sharedExpertX's dim0 * sharedExpertX's dim1 = %ld, bs = %ld",
                sharedExpertXDim0 * sharedExpertXDim1, expertIdsDim0), return false);
            OP_TILING_CHECK(sharedExpertXDim2 != expandXDim1, OP_LOGE(nodeName,
                "sharedExpertX's dim2 not equal to h, sharedExpertX's dim2 = %ld, h = %ld",
                sharedExpertXDim2, expandXDim1), return false);
        }
    }

    return true;
}

static bool CheckFromDispatchTensorShape(const gert::TilingContext *context, MoeDistributeCombineV2TilingData &tilingData,
    const char *nodeName, const CombineV2Config& config, const uint32_t A, const bool isLayered)
{
    // 校验expandX的维度并设h
    int64_t tpWorldSize = static_cast<int64_t>(tilingData.moeDistributeCombineV2Info.tpWorldSize);
    const int64_t epWorldSize = static_cast<int64_t>(tilingData.moeDistributeCombineV2Info.epWorldSize);
    const gert::StorageShape *expandXStorageShape = context->GetInputShape(config.expandXIndex);
    int64_t expandXDim0 = expandXStorageShape->GetStorageShape().GetDim(0);
    int64_t expandXDim1 = expandXStorageShape->GetStorageShape().GetDim(1);
    OP_TILING_CHECK(expandXDim0 < static_cast<int64_t>(A) * tpWorldSize, OP_LOGE(nodeName,
        "expandX's dim0 not greater than or equal to A * tpWorldSize, expandX's dim0 = %ld, A = %ld, tpWorldSize = %ld",
        expandXDim0, static_cast<int64_t>(A), tpWorldSize), return false);
    int64_t hMax = isLayered ? H_MAX_LAYERED : H_MAX;
    int64_t hMin = isLayered ? 0 : H_MIN;
    OP_TILING_CHECK((isLayered && (expandXDim1 % H_BASIC_BLOCK_LAYERED != 0)), OP_LOGE(nodeName,
        "xShape dims1(H) should be %u Aligned, but got %ld.", H_BASIC_BLOCK_LAYERED, expandXDim1), return false);
    OP_TILING_CHECK((expandXDim1 < hMin) || (expandXDim1 > hMax),
        OP_LOGE(nodeName, "expandX's dim1(H) should be in [%ld, %ld], but got %ld.", hMin, hMax, expandXDim1), return false); // 32对齐
    tilingData.moeDistributeCombineV2Info.h = static_cast<uint32_t>(expandXDim1);
    if (expandXDim1 != 0) {tilingData.moeDistributeCombineV2Info.armAvgFactor = 1.0f / expandXDim1;}

    // 校验assistInfo的维度
    const gert::StorageShape *assistInfoStorageShape = context->GetInputShape(config.assistInfoIndex);
    int64_t assistInfoDim0 = assistInfoStorageShape->GetStorageShape().GetDim(0);
    int64_t minAssistInfoDim0 = static_cast<int64_t>(A * ASSIST_NUM_PER_A);
    OP_TILING_CHECK(assistInfoDim0 < minAssistInfoDim0, OP_LOGE(nodeName, "assistInfoForCombine's dim0 <"
        " minAssistInfoDim0, assistInfoForCombine's dim0 is %ld, minAssistInfoDim0 is %ld.", assistInfoDim0,
        minAssistInfoDim0), return false);

    // 校验expandScales的维度
    if (isLayered) {
        const gert::StorageShape *expandScaleStorageShape = context->GetOptionalInputShape(EXPAND_SCALES_INDEX);
        int64_t expandScalesDim0 = expandScaleStorageShape->GetStorageShape().GetDim(0);
        int64_t minexpandScalesDim0 = static_cast<int64_t>(A);
        OP_TILING_CHECK(expandScalesDim0 < minexpandScalesDim0, OP_LOGE(nodeName, "expandScales's dim0 must >="
            " minexpandScales's dim0, expandScales's dim0 is %ld, minexpandScales's dim0 is %ld.", expandScalesDim0,
            minexpandScalesDim0), return false);
    }

    return true;
}

static uint32_t CalA(MoeDistributeCombineV2TilingData &tilingData, const bool isShared, const uint32_t localMoeExpertNum,
    const bool hasElasticInfo, const int64_t expertIdsDim1)
{
    uint32_t A = 0U;
    uint32_t rankNumPerSharedExpert = 0U;
    uint32_t maxSharedGroupNum = 0U;
    uint32_t globalBs = tilingData.moeDistributeCombineV2Info.globalBs;
    uint32_t sharedExpertNum = tilingData.moeDistributeCombineV2Info.sharedExpertNum;
    uint32_t sharedExpertRankNum = tilingData.moeDistributeCombineV2Info.sharedExpertRankNum;
    uint32_t epWorldSizeU32 = tilingData.moeDistributeCombineV2Info.epWorldSize;
    uint32_t maxBs = globalBs / epWorldSizeU32;
    if ((sharedExpertNum != 0U) && (sharedExpertRankNum != 0U)) { // 除零保护
        rankNumPerSharedExpert = sharedExpertRankNum / sharedExpertNum;
        maxSharedGroupNum = (epWorldSizeU32 + rankNumPerSharedExpert - 1U) / rankNumPerSharedExpert;
    }
    if (isShared) { // 本卡为共享专家
        A = maxBs * maxSharedGroupNum;
    } else { // 本卡为moe专家
        A = globalBs * std::min(static_cast<int64_t>(localMoeExpertNum), expertIdsDim1);
    }
    if (hasElasticInfo) {
        A = std::max(static_cast<int64_t>(maxBs * maxSharedGroupNum), globalBs * std::min(static_cast<int64_t>(localMoeExpertNum), expertIdsDim1));
    }
    tilingData.moeDistributeCombineV2Info.a = A;
    return A;
}

static bool CheckSendCountTensorShape(const gert::TilingContext *context, MoeDistributeCombineV2TilingData &tilingData,
    const char *nodeName, const bool isShared, const uint32_t localMoeExpertNum, const bool hasElasticInfo, const bool expertIdsDim1,
    const CombineV2Config& config, const bool isLayered)
{
    // 校验epSendCount和tpSendCount的维度
    int64_t tpWorldSize = static_cast<int64_t>(tilingData.moeDistributeCombineV2Info.tpWorldSize);
    const int64_t epWorldSize = static_cast<int64_t>(tilingData.moeDistributeCombineV2Info.epWorldSize);
    uint32_t globalBs = tilingData.moeDistributeCombineV2Info.globalBs;
    int64_t moeExpertPerRankNum = static_cast<int64_t>(tilingData.moeDistributeCombineV2Info.moeExpertPerRankNum);
    const gert::StorageShape *epSendCountStorageShape = context->GetInputShape(config.epSendCountIndex);
    const int64_t epSendCountDim0 = epSendCountStorageShape->GetStorageShape().GetDim(0);
    int64_t localEpSendCountSize = (isShared) ? epWorldSize : epWorldSize * moeExpertPerRankNum;

    if (hasElasticInfo) {localEpSendCountSize = std::max(epWorldSize, epWorldSize * moeExpertPerRankNum);}
    if (isLayered) {
        // 如果是分层方案，则需要校验全新的shape，额外的globalBs * 2 * k * epWorldSize / 8 用来存储token的cnt信息与offset信息，为了兼容A2&A5 这里取/8。
        localEpSendCountSize = epWorldSize * localMoeExpertNum + globalBs * 2 * expertIdsDim1 * epWorldSize / RANK_NUM_PER_NODE_A2;
    }
    OP_TILING_CHECK(epSendCountDim0 < localEpSendCountSize * tpWorldSize, OP_LOGE(nodeName, "epSendCount's dim0 not "
        "greater than or equal to localEpSendCountSize * tpWorldSize, epSendCount's dim0 is %ld, localEpSendCountSize "
        "is %ld, tpWorldSize is %ld.", epSendCountDim0, localEpSendCountSize, tpWorldSize), return false);
    if (tpWorldSize == TP_WORLD_SIZE_TWO) {
        const gert::StorageShape *tpSendCountStorageShape = context->GetOptionalInputShape(config.tpSendCountsIndex);
        const int64_t tpSendCountDim0 = tpSendCountStorageShape->GetStorageShape().GetDim(0);
        OP_TILING_CHECK(tpSendCountDim0 != tpWorldSize, OP_LOGE(nodeName, "tpSendCount's dim0 not equal to tpWorldSize, "
            "tpSendCount's dim0 is %ld, tpWorldSize is %ld.", tpSendCountDim0, tpWorldSize), return false);
    }
    return true;
}

static bool CheckTensorShape(const gert::TilingContext *context, MoeDistributeCombineV2TilingData &tilingData,
    const char *nodeName, bool isShared, bool isActiveMask, uint32_t localMoeExpertNum, const bool hasElasticInfo,
    const CombineV2Config& config, bool isLayered)
{
    // 校验输入expertIds的维度1并设k, bs已校验过
    const gert::StorageShape *expertIdsStorageShape = context->GetInputShape(config.expertIdsIndex);
    int64_t expertIdsDim0 = expertIdsStorageShape->GetStorageShape().GetDim(0);
    int64_t expertIdsDim1 = expertIdsStorageShape->GetStorageShape().GetDim(1);

    const int64_t epWorldSize = static_cast<int64_t>(tilingData.moeDistributeCombineV2Info.epWorldSize);
    if (hasElasticInfo) {
        const gert::StorageShape *elasticInfoStorageShape = context->GetOptionalInputShape(config.elasticInfoIndex);
        const int64_t elasticInfoDim0 = elasticInfoStorageShape->GetStorageShape().GetDim(0);

        OP_TILING_CHECK(elasticInfoDim0 != (ELASTIC_METAINFO_OFFSET + RANK_LIST_NUM * epWorldSize),
            OP_LOGE(nodeName, "elasticInfo's dim0 not equal to 4 + 2 * epWorldSize, "
            "elasticInfo's dim0 is %ld, epWorldSize is %ld.",
            elasticInfoDim0, epWorldSize), return false);
    }
    uint32_t A = CalA(tilingData, isShared, localMoeExpertNum, hasElasticInfo, expertIdsDim1);

    // 校验activeMask的维度
    if (isActiveMask) {
        const gert::StorageShape *xActiveMaskStorageShape = context->GetOptionalInputShape(config.xActiveMaskIndex);
        int64_t xActiveMaskDim0 = xActiveMaskStorageShape->GetStorageShape().GetDim(0);
        OP_TILING_CHECK(xActiveMaskDim0 != expertIdsDim0,
            OP_LOGE(nodeName, "xActiveMask's dim0 not equal to expertIds's dim0, xActiveMask's dim0 is %ld, "
            "expertIds's dim0 is %ld", xActiveMaskDim0, expertIdsDim0), return false);
        OP_TILING_CHECK(((xActiveMaskStorageShape->GetStorageShape().GetDimNum() == TWO_DIMS) &&
            (xActiveMaskStorageShape->GetStorageShape().GetDim(1) != expertIdsDim1)),
            OP_LOGE(nodeName, "xActiveMask's dim1 not equal to expertIds's dim1, xActiveMask's dim1 is %ld, "
            "expertIds's dim1 is %ld", xActiveMaskStorageShape->GetStorageShape().GetDim(1), expertIdsDim1), return false);
    }

    OP_TILING_CHECK(!CheckZeroComputeExpertsTensorShape(context, tilingData, nodeName, config, expertIdsDim0),
        OP_LOGE(nodeName, "Zero_compute_experts' param dim check failed."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(!CheckExpertsTensorShape(context, tilingData, nodeName, config, expertIdsDim0, expertIdsDim1),
        OP_LOGE(nodeName, "Experts' param dim check failed."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(!CheckFromDispatchTensorShape(context, tilingData, nodeName, config, A, isLayered),
        OP_LOGE(nodeName, "From dispatch's param dim check failed."), return ge::GRAPH_FAILED);
    
    OP_TILING_CHECK(!CheckSendCountTensorShape(context, tilingData, nodeName, isShared, localMoeExpertNum, hasElasticInfo, expertIdsDim1, config, isLayered),
        OP_LOGE(nodeName, "SendCount of dispatch's param dim check failed."), return ge::GRAPH_FAILED);
    return true;
}

static bool CheckSharedAttrs(const char *nodeName, const MoeDistributeCombineV2TilingData &tilingData)
{
    uint32_t sharedExpertNum = tilingData.moeDistributeCombineV2Info.sharedExpertNum;
    uint32_t sharedExpertRankNum = tilingData.moeDistributeCombineV2Info.sharedExpertRankNum;

    // 校验共享专家卡数和共享专家数是否只有一个为0
    OP_TILING_CHECK((sharedExpertNum == 0U) && (sharedExpertRankNum > 0U),
        OP_LOGE(nodeName, "sharedExpertRankNum is invalid, only support 0 when sharedExpertNum is 0, but got %u.",
        sharedExpertRankNum), return false);
    OP_TILING_CHECK((sharedExpertNum > 0U) && (sharedExpertRankNum == 0U),
        OP_LOGE(nodeName, "sharedExpertNum is invalid, only support 0 when sharedExpertRankNum is 0, but got %u.",
        sharedExpertNum), return false);

    if ((sharedExpertNum > 0U) && (sharedExpertRankNum > 0U)) {
        // 校验共享专家卡数能否整除共享专家数
        OP_TILING_CHECK(((sharedExpertRankNum % sharedExpertNum) != 0U),
            OP_LOGE(nodeName, "sharedExpertRankNum should be divisible by sharedExpertNum, but sharedExpertRankNum=%u, "
            "sharedExpertNum=%u.", sharedExpertRankNum, sharedExpertNum), return false);
    }

    return true;
}

static bool CheckCommAlgAttrs(const char *nodeName, const MoeDistributeCombineV2TilingData &tilingData, bool isLayered)
{
    bool hasElasticInfo = tilingData.moeDistributeCombineV2Info.hasElasticInfo;
    uint32_t zeroExpertNum = tilingData.moeDistributeCombineV2Info.zeroExpertNum;
    uint32_t copyExpertNum = tilingData.moeDistributeCombineV2Info.copyExpertNum;
    uint32_t constExpertNum = tilingData.moeDistributeCombineV2Info.constExpertNum;
    int32_t zeroComputeExpertNum = zeroExpertNum + copyExpertNum + constExpertNum;
    bool isExpertMask = tilingData.moeDistributeCombineV2Info.isExpertMask;

    // 校验动态缩容和分层不能同时启用
    OP_TILING_CHECK((isLayered && hasElasticInfo), OP_LOGE(nodeName, "Cannot support elasticInfo when comm_alg = hierarchy"),
        return false);
    // 校验特殊专家和分层不能同时启用
    OP_TILING_CHECK((isLayered && (zeroComputeExpertNum > 0)), OP_LOGE(nodeName, "Cannot support zeroComputeExpert when comm_alg = hierarchy"),
        return false);
    // 校验二维Mask和分层不能同时启用
    OP_TILING_CHECK((isLayered && isExpertMask), OP_LOGE(nodeName, "Cannot support 2D xActiveMask when comm_alg = hierarchy"),
        return false);
    return true;
}

static bool CheckZeroComputeExpert(const gert::TilingContext *context, MoeDistributeCombineV2TilingData &tilingData,
    const char *nodeName, const CombineV2Config& config)
{
    uint32_t copyExpertNum = tilingData.moeDistributeCombineV2Info.copyExpertNum;
    uint32_t constExpertNum = tilingData.moeDistributeCombineV2Info.constExpertNum;

    const gert::StorageShape *oriXStorageShape = context->GetOptionalInputShape(config.oriXIndex);
    const gert::StorageShape *constExpertAlpha1StorageShape = context->GetOptionalInputShape(config.constExpertAlpha1Index);
    const gert::StorageShape *constExpertAlpha2StorageShape = context->GetOptionalInputShape(config.constExpertAlpha2Index);
    const gert::StorageShape *constExpertVStorageShape = context->GetOptionalInputShape(config.constExpertVIndex);

    OP_TILING_CHECK(copyExpertNum > 0 && oriXStorageShape == nullptr,
        OP_LOGE(nodeName, "oriX must exist when copyExpertNum > 0"), return false);
    OP_TILING_CHECK(constExpertNum > 0 && (oriXStorageShape == nullptr || constExpertAlpha1StorageShape == nullptr ||
                    constExpertAlpha2StorageShape == nullptr || constExpertVStorageShape == nullptr),
        OP_LOGE(nodeName, "oriX、alpha1、alpha2、V must exist when constExpertNum > 0"), return false);
    return true;
}

static bool CheckAttrs(const gert::TilingContext *context, MoeDistributeCombineV2TilingData &tilingData,
    const char *nodeName, uint32_t &localMoeExpertNum, bool isActiveMask, const CombineV2Config& config, bool isLayered)
{
    uint32_t epWorldSize = tilingData.moeDistributeCombineV2Info.epWorldSize;
    uint32_t tpWorldSize = tilingData.moeDistributeCombineV2Info.tpWorldSize;
    uint32_t moeExpertNum = tilingData.moeDistributeCombineV2Info.moeExpertNum;
    uint32_t sharedExpertRankNum = tilingData.moeDistributeCombineV2Info.sharedExpertRankNum;

    OP_TILING_CHECK(!CheckSharedAttrs(nodeName, tilingData),
        OP_LOGE(nodeName, "Check shared expert related attributes failed."), return false);
    OP_TILING_CHECK(!CheckCommAlgAttrs(nodeName, tilingData, isLayered),
        OP_LOGE(nodeName, "Check comm_alg related attributes failed."), return false);
    // 校验moe专家数量能否均分给多机
    OP_TILING_CHECK(moeExpertNum % (epWorldSize - sharedExpertRankNum) != 0,
        OP_LOGE(nodeName, "moeExpertNum should be divisible by (epWorldSize - sharedExpertRankNum), "
        "but got moeExpertNum=%u, epWorldSize=%u, sharedExpertRankNum=%u.", moeExpertNum, epWorldSize,
        sharedExpertRankNum), return false);
    localMoeExpertNum = moeExpertNum / (epWorldSize - sharedExpertRankNum);
    OP_TILING_CHECK((localMoeExpertNum <= 0) || (localMoeExpertNum * epWorldSize > LOCAL_EXPERT_MAX_SIZE),OP_LOGE(nodeName, "localMoeExpertNum is invalid, "
        "localMoeExpertNum * epWorldSize must be less than or equal to 2048, and localMoeExpertNum must be greater than 0, "
        "but got localMoeExpertNum * epWorldSize = %u, localMoeExpertNum = %u", localMoeExpertNum * epWorldSize, localMoeExpertNum), return false);
    // 校验tp=2时单个moe卡上专家数是否等于1
    OP_TILING_CHECK((localMoeExpertNum > 1) && (tpWorldSize > 1), OP_LOGE(nodeName, "Cannot support multi-moeExpert %u "
        "in a rank when tpWorldSize = %u > 1", localMoeExpertNum, tpWorldSize), return false);
    OP_TILING_CHECK((tpWorldSize > 1) && (tilingData.moeDistributeCombineV2Info.hasElasticInfo), OP_LOGE(nodeName, "Cannot support elasticInfo "
        "when tpWorldSize = %u > 1", tpWorldSize), return false);
    tilingData.moeDistributeCombineV2Info.moeExpertPerRankNum = localMoeExpertNum;

    // 校验输入expertIds的维度0并设bs
    const gert::StorageShape *expertIdsStorageShape = context->GetInputShape(config.expertIdsIndex);
    int64_t expertIdsDim0 = expertIdsStorageShape->GetStorageShape().GetDim(0);
    int64_t bsUpperBound = isLayered ? BS_UPPER_BOUND_LAYERED : BS_UPPER_BOUND;
    OP_TILING_CHECK((expertIdsDim0 <= 0) || (expertIdsDim0 > bsUpperBound), OP_LOGE(nodeName, "Invalid expertIds dims0(BS) %ld. "
        "Should be between [1, %ld].", expertIdsDim0, bsUpperBound), return false);
    tilingData.moeDistributeCombineV2Info.bs = static_cast<uint32_t>(expertIdsDim0);

    // 校验globalBS
    auto attrs = context->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE(nodeName, "attrs is null."), return false);
    auto globalBsPtr = attrs->GetAttrPointer<int64_t>((config.attrGlobalBsIndex));
    OP_TILING_CHECK(globalBsPtr == nullptr, OP_LOGE(nodeName, "globalBs is null."), return false);
    OP_LOGD(nodeName, "MoeDistributeCombineV2 *globalBsPtr = %ld, bs = %ld, epWorldSize = %u\n",
        *globalBsPtr, expertIdsDim0, epWorldSize);

    OP_TILING_CHECK((*globalBsPtr != 0) && ((*globalBsPtr < static_cast<int64_t>(epWorldSize) * expertIdsDim0) ||
        ((*globalBsPtr) % (static_cast<int64_t>(epWorldSize)) != 0)), OP_LOGE(nodeName, "globalBS is invalid, only "
        "support 0 or maxBs(maxBs is the largest bs on all ranks) * epWorldSize, but got globalBS=%ld, "
        "bs=%ld, epWorldSize=%u.", *globalBsPtr, expertIdsDim0, epWorldSize),  return false);
    OP_TILING_CHECK(((*globalBsPtr > (expertIdsDim0 * static_cast<int64_t>(epWorldSize))) && isActiveMask),
        OP_LOGE(nodeName, "Different bs on different rank cannot work when isActiveMask=true, globalBS=%ld, "
        "bs=%ld, epWorldSize=%u.", *globalBsPtr, expertIdsDim0, epWorldSize), return false);

    tilingData.moeDistributeCombineV2Info.globalBs = static_cast<uint32_t>(*globalBsPtr);
    if (*globalBsPtr == 0) {
        tilingData.moeDistributeCombineV2Info.globalBs = static_cast<uint32_t>(expertIdsDim0) * epWorldSize;
    }
    OP_TILING_CHECK(!CheckZeroComputeExpert(context, tilingData, nodeName, config),
        OP_LOGE(nodeName, "Zero compute expert check failed."), return ge::GRAPH_FAILED);
    return true;
}

static ge::graphStatus TilingCheckMoeDistributeCombine(gert::TilingContext *context, const char *nodeName,
    const bool isActiveMask, const bool hasElasticInfo, uint32_t tpWorldSize,
    const CombineV2Config& config, const bool isLayered)
{
    // 检查参数shape信息
    OP_TILING_CHECK(!CheckTensorDim(context, nodeName, isActiveMask, hasElasticInfo, tpWorldSize, config, isLayered),
                    OP_LOGE(nodeName, "param shape is invalid"), return ge::GRAPH_FAILED);
    // 检查参数dataType信息
    OP_TILING_CHECK(!CheckExpertTensorDataType(context, nodeName, config),
                    OP_LOGE(nodeName, "Experts param dataType is invalid"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(!CheckTensorDataType(context, nodeName, isActiveMask, hasElasticInfo, tpWorldSize, config),
                    OP_LOGE(nodeName, "param dataType is invalid"), return ge::GRAPH_FAILED);
    // 检查参数format信息
    OP_TILING_CHECK(!CheckTensorFormat(context, nodeName, isActiveMask, hasElasticInfo, tpWorldSize, config),
                    OP_LOGE(nodeName, "param Format is invalid"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SetWorkspace(gert::TilingContext *context, const char *nodeName)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint64_t aivNum = ascendcPlatform.GetCoreNumAiv();
    size_t *workspace = context->GetWorkspaceSizes(1);
    OP_TILING_CHECK(workspace == nullptr, VECTOR_INNER_ERR_REPORT_TILING(nodeName, "get workspace failed"),
        return ge::GRAPH_FAILED);
    workspace[0] = SYSTEM_NEED_WORKSPACE + aivNum * MASK_CALC_NEED_WORKSPACE;
    OP_LOGD(nodeName, "workspace[0] size is %ld", workspace[0]);
    return ge::GRAPH_SUCCESS;
}

static uint64_t CalTilingKey(const uint32_t tpWorldSize, uint32_t commQuantMode, bool isLayered)
{
    bool tp = false;
    uint32_t quantMode = TILINGKEY_NO_QUANT;
    uint32_t hierarchy = TILINGKEY_TPL_MTE;
    if (tpWorldSize == MAX_TP_WORLD_SIZE) {
        tp = true;
    }
    if (commQuantMode == INT8_COMM_QUANT) {
        quantMode = TILINGKEY_INT8_QUANT;
    }
    if (isLayered) {
        hierarchy = TILINGKEY_TPL_HIERARCHY;
    }
    uint64_t tilingKey = GET_TPL_TILING_KEY(tp, quantMode, hierarchy, TILINGKEY_TPL_A3);
    return tilingKey;
}

static ge::graphStatus SetHCommCfg(const gert::TilingContext *context, MoeDistributeCombineV2TilingData *tiling,
    const std::string groupEp, const std::string groupTp, const uint32_t tpWorldSize, bool isLayered)
{
    const char* nodeName = context->GetNodeName();
    OP_LOGD(nodeName, "MoeDistributeCombineV2 groupEp = %s", groupEp.c_str());
    uint32_t opType1 = OP_TYPE_ALL_TO_ALL;
    uint32_t opType2 = OP_TYPE_REDUCE_SCATTER;
    std::string algConfigAllToAllStr = isLayered ? "AlltoAll=level1:hierarchy" : "AlltoAll=level0:fullmesh;level1:pairwise";
    std::string algConfigReduceScatterStr = "ReduceScatter=level0:ring";

    AscendC::Mc2CcTilingConfig mc2CcTilingConfig(groupEp, opType1, algConfigAllToAllStr);
    mc2CcTilingConfig.SetCommEngine(mc2tiling::AIV_ENGINE);   // 通过不拉起AICPU，提高算子退出性能
    
    OP_TILING_CHECK(mc2CcTilingConfig.GetTiling(tiling->mc2InitTiling) != 0,
        OP_LOGE(nodeName, "mc2CcTilingConfig mc2tiling GetTiling mc2InitTiling failed"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(mc2CcTilingConfig.GetTiling(tiling->mc2CcTiling1) != 0,
        OP_LOGE(nodeName, "mc2CcTilingConfig mc2tiling1 GetTiling mc2CcTiling1 failed"), return ge::GRAPH_FAILED);

    if (tpWorldSize > 1) {
        OP_LOGD(nodeName, "MoeDistributeCombineV2 groupTp = %s", groupTp.c_str());
        mc2CcTilingConfig.SetGroupName(groupTp);
        mc2CcTilingConfig.SetOpType(opType2);
        mc2CcTilingConfig.SetAlgConfig(algConfigReduceScatterStr);
        OP_TILING_CHECK(mc2CcTilingConfig.GetTiling(tiling->mc2CcTiling2) != 0,
            OP_LOGE(nodeName, "mc2CcTilingConfig mc2tiling2 GetTiling  mc2CcTiling2 failed"), return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

static void UbUsedCal(const uint64_t ubSize, const gert::TilingContext* context, MoeDistributeCombineV2TilingData *tilingData, const CombineV2Config& config)
{
    uint32_t axisH = tilingData->moeDistributeCombineV2Info.h;
    uint32_t axisBS = tilingData->moeDistributeCombineV2Info.bs;
    uint32_t axisK = tilingData->moeDistributeCombineV2Info.k;
    uint32_t zeroExpertNum = tilingData->moeDistributeCombineV2Info.zeroExpertNum;
    uint32_t copyExpertNum = tilingData->moeDistributeCombineV2Info.copyExpertNum;
    uint32_t constExpertNum = tilingData->moeDistributeCombineV2Info.constExpertNum;
    bool isInputExpertMaskFlag = tilingData->moeDistributeCombineV2Info.isExpertMask;
    bool isInputTokenMaskFlag = tilingData->moeDistributeCombineV2Info.isTokenMask;
    bool enableSpecialExpert = (constExpertNum + zeroExpertNum + copyExpertNum > 0U);
    auto expandXDesc = context->GetInputDesc(config.expandXIndex);
    auto attrs = context->GetAttrs();
    auto commQuantModePtr = attrs->GetAttrPointer<int64_t>((config.attrCommQuantModeIndex));
    uint32_t maxSizeTokenBuf = (axisH * sizeof(expandXDesc->GetDataType()) + UB_ALIGN - 1) / UB_ALIGN * UB_ALIGN;
    uint32_t hExpandXTypeSize = axisH * sizeof(expandXDesc->GetDataType());
    uint32_t activeMaskAlignSize = axisBS * ((axisK * sizeof(bool) + UB_ALIGN - 1) / UB_ALIGN * UB_ALIGN);
    uint32_t hExpandXAlign32Size = (hExpandXTypeSize + UB_ALIGN - 1) / UB_ALIGN * UB_ALIGN;
    uint32_t hFloatSize = axisH * static_cast<uint32_t>(sizeof(float));
    uint32_t hFloatAlign32Size = (hFloatSize + UB_ALIGN - 1) / UB_ALIGN * UB_ALIGN;
    uint32_t maxSizeRowTmpFloatBuf = hFloatAlign32Size;
    uint32_t flagRcvCount = axisK + tilingData->moeDistributeCombineV2Info.sharedExpertNum;
    uint32_t hFloatAlign256Size = (hFloatSize + ALIGNED_LEN - 1) / ALIGNED_LEN * ALIGNED_LEN;
    uint32_t bsKNum = axisBS * axisK;
    uint32_t bsKFloatAlign = (bsKNum * sizeof(float) + UB_ALIGN - 1) / UB_ALIGN * UB_ALIGN;
    uint32_t mulBufSize = hFloatAlign256Size > bsKFloatAlign ? hFloatAlign256Size : bsKFloatAlign;

    if (isInputExpertMaskFlag || enableSpecialExpert) {
        uint32_t activeMaskAlignHalfSize = activeMaskAlignSize * sizeof(DTYPE_SIZE_HALF);
        maxSizeTokenBuf = (activeMaskAlignSize > hExpandXAlign32Size ? activeMaskAlignSize : hExpandXAlign32Size);
        maxSizeRowTmpFloatBuf = (activeMaskAlignHalfSize > hFloatAlign32Size ? activeMaskAlignHalfSize : hFloatAlign32Size);
    }

    // LocalWindowCopy的ub使用总量
    uint32_t totalBufferSize = 0;
    if (config.hasAddRmsNorm) {
        // NUM_PER_REP_FP32 * sizeof(float) * 2 是为kernel侧ReduceSum操作申请的空间大小
        totalBufferSize = maxSizeTokenBuf + maxSizeRowTmpFloatBuf + mulBufSize + hFloatAlign32Size + hExpandXAlign32Size
        + NUM_PER_REP_FP32 * sizeof(float) * BUFFER_NUM + hExpandXAlign32Size * BUFFER_NUM + flagRcvCount * STATE_OFFSET * BUFFER_NUM + UB_ALIGN;
    } else {
        totalBufferSize = maxSizeTokenBuf + maxSizeRowTmpFloatBuf + mulBufSize + hFloatAlign32Size + hExpandXAlign32Size * BUFFER_NUM
        + flagRcvCount * STATE_OFFSET * BUFFER_NUM + UB_ALIGN;
    }
    if (*commQuantModePtr == INT8_COMM_QUANT) {
        uint32_t scaleNum = (hExpandXAlign32Size / sizeof(expandXDesc->GetDataType())) / static_cast<uint32_t>(UB_ALIGN / sizeof(float));
        uint32_t scaleNumAlignSize = (scaleNum * sizeof(float) + UB_ALIGN - 1) / UB_ALIGN * UB_ALIGN;
        totalBufferSize += scaleNumAlignSize;
    }
    if (isInputTokenMaskFlag) {
        uint32_t axisBsAlignSize = (axisBS * sizeof(bool) + UB_ALIGN - 1) / UB_ALIGN * UB_ALIGN;
        totalBufferSize += axisBsAlignSize + axisBsAlignSize * sizeof(DTYPE_SIZE_HALF) * BUFFER_NUM;
    }
    if (isInputExpertMaskFlag) {
        totalBufferSize += (axisBS * sizeof(DTYPE_SIZE_HALF) + UB_ALIGN - 1) / UB_ALIGN * UB_ALIGN + (axisBS * sizeof(int32_t) +
            UB_ALIGN - 1) / UB_ALIGN * UB_ALIGN + (axisBS * axisK * sizeof(bool) + UB_ALIGN - 1) / UB_ALIGN * UB_ALIGN;
    }
    if (enableSpecialExpert && !isInputExpertMaskFlag) {
        totalBufferSize += (axisBS * sizeof(DTYPE_SIZE_HALF) + UB_ALIGN - 1) / UB_ALIGN * UB_ALIGN;
    }
    tilingData->moeDistributeCombineV2Info.bufferNum = totalBufferSize > ubSize ? BUFFER_SINGLE : BUFFER_NUM;
}

static ge::graphStatus CheckTpWinSize(const gert::TilingContext *context, MoeDistributeCombineV2TilingData* tilingData, const char *nodeName,
    uint64_t tokenNeedSizeDispatch, uint64_t tokenNeedSizeCombine)
{
    auto attrs = context->GetAttrs();
    uint64_t tpWorldSize = static_cast<uint64_t>(tilingData->moeDistributeCombineV2Info.tpWorldSize);
    if (tpWorldSize == TP_WORLD_SIZE_TWO) {
        uint64_t maxWindowSizeTp = 0;
        auto groupTpHccl = attrs->GetAttrPointer<char>(static_cast<int>(ATTR_GROUP_TP_INDEX));
        OP_TILING_CHECK(mc2tiling::GetCclBufferSize(groupTpHccl, &maxWindowSizeTp, nodeName) != ge::GRAPH_SUCCESS,
            OP_LOGE(nodeName, "Get TP HcclBufferSize failed, HcclBufferSizeTP is %lu", maxWindowSizeTp),
            return ge::GRAPH_FAILED);
        uint64_t actualSize = static_cast<uint64_t>(tilingData->moeDistributeCombineV2Info.a) * (tokenNeedSizeDispatch +
        tokenNeedSizeCombine) * DOUBLE_DATA_BUFFER;
        OP_TILING_CHECK((actualSize > maxWindowSizeTp),
        OP_LOGE(nodeName, "TP HCCL_BUFFSIZE is too SMALL, A = %u, tokenNeedSizeDispatch = %lu, tokenNeedSizeCombine = %lu,"
            "NEEDED_HCCL_BUFFSIZE(A * (tokenNeedSizeDispatch + tokenNeedSizeCombine) * 2) = %luMB, TP HCCL_BUFFSIZE= %luMB.",
            tilingData->moeDistributeCombineV2Info.a, tokenNeedSizeDispatch, tokenNeedSizeCombine, actualSize / MB_SIZE + 1UL,
            maxWindowSizeTp / MB_SIZE), return ge::GRAPH_FAILED);
        tilingData->moeDistributeCombineV2Info.totalWinSizeTp = maxWindowSizeTp;
        OP_LOGD(nodeName, "TpwindowSize = %lu", maxWindowSizeTp);
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckWinSize(const gert::TilingContext *context, MoeDistributeCombineV2TilingData* tilingData,
    const char *nodeName, uint32_t localMoeExpertNum, bool isLayered)
{
    auto attrs = context->GetAttrs();
    uint64_t hcclBufferSizeEp = 0;
    uint64_t maxWindowSizeEp = 0;
    OP_TILING_CHECK(
        mc2tiling::GetEpWinSize(context, nodeName, hcclBufferSizeEp, maxWindowSizeEp, ATTR_GROUP_EP_INDEX, isLayered) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "Get EP WinSize failed"), return ge::GRAPH_FAILED);
    uint64_t h = static_cast<uint64_t>(tilingData->moeDistributeCombineV2Info.h);
    uint64_t epWorldSize = static_cast<uint64_t>(tilingData->moeDistributeCombineV2Info.epWorldSize);
    uint64_t k = static_cast<uint64_t>(tilingData->moeDistributeCombineV2Info.k);
    uint64_t bs = static_cast<uint64_t>(tilingData->moeDistributeCombineV2Info.bs);
    uint64_t sharedExpertNum = static_cast<uint64_t>(tilingData->moeDistributeCombineV2Info.sharedExpertNum);
    uint64_t maxBs = static_cast<uint64_t>(tilingData->moeDistributeCombineV2Info.globalBs)/ epWorldSize;
    // combine数据区 token首地址对齐512
    uint64_t tokenNeedSizeCombine = ((h * MAX_OUT_DTYPE_SIZE  + WIN_ADDR_ALIGN - 1UL) / WIN_ADDR_ALIGN) * WIN_ADDR_ALIGN;
    // dispatch数据区 token首对齐512，有效token长度h_align_32b + scale(32b) + 三元组(3*4b)
    uint64_t tokenActualLen = ((h * MAX_OUT_DTYPE_SIZE  + UB_ALIGN - 1UL) / UB_ALIGN) * UB_ALIGN + SCALE_EXPAND_IDX_BUFFER;
    uint64_t moeExpertNum = static_cast<uint64_t>(tilingData->moeDistributeCombineV2Info.moeExpertNum);
    uint64_t tokenNeedSizeDispatch = ((tokenActualLen + WIN_ADDR_ALIGN - 1UL) / WIN_ADDR_ALIGN) * WIN_ADDR_ALIGN;
    uint64_t actualSize = isLayered ? (moeExpertNum * maxBs * (h * MAX_OUT_DTYPE_SIZE + (3 * (k + 7) / 8 * 8) *
 	         sizeof(uint32_t) + 64) + 404 * MB_SIZE) : ((maxBs * tokenNeedSizeDispatch * epWorldSize * static_cast<uint64_t>(localMoeExpertNum))
             + (maxBs * tokenNeedSizeCombine * (k + static_cast<uint64_t>(sharedExpertNum)))) * DOUBLE_DATA_BUFFER;
    if (isLayered) {
        // 校验可变bs
        OP_TILING_CHECK((bs != maxBs), OP_LOGE(nodeName, "Cannot support variableBs."), return ge::GRAPH_FAILED);
        // 校验buffersize
        OP_TILING_CHECK((actualSize > maxWindowSizeEp),
        OP_LOGE(nodeName, "HCCL_BUFFSIZE_EP is too SMALL, maxBs = %lu, h = %lu,"
            "NEEDED_HCCL_BUFFSIZE_HIERARCHY((moeExpertNum * maxBs * (h * MAX_OUT_DTYPE_SIZE + (3 * (k + 7) / 8 * 8) *"
            "sizeof(uint32_t) + 64) + 404 * 1024 * 1024)) = %luMB, HCCL_BUFFSIZE=%luMB.", maxBs, h,
            actualSize / MB_SIZE + 1UL, hcclBufferSizeEp / MB_SIZE), return ge::GRAPH_FAILED);
    } else {
        OP_TILING_CHECK((actualSize > maxWindowSizeEp),
        OP_LOGE(nodeName, "HCCL_BUFFSIZE_EP is too SMALL, maxBs = %lu, h = %lu, epWorldSize = %lu,"
            " localMoeExpertNum = %u, sharedExpertNum = %u, tokenNeedSizeDispatch = %lu, tokenNeedSizeCombine = %lu,"
            " k = %lu, NEEDED_HCCL_BUFFSIZE(((maxBs * tokenNeedSizeDispatch * ep_worldsize * localMoeExpertNum) +"
            " (maxBs * tokenNeedSizeCombine * (k + sharedExpertNum))) * 2) = %luMB,"
            " HCCL_BUFFSIZE=%luMB.", maxBs, h, epWorldSize, localMoeExpertNum, sharedExpertNum,
            tokenNeedSizeDispatch, tokenNeedSizeCombine, k, actualSize / MB_SIZE + 1UL, hcclBufferSizeEp / MB_SIZE),
            return ge::GRAPH_FAILED);
    }
    tilingData->moeDistributeCombineV2Info.totalWinSizeEp = maxWindowSizeEp;
    OP_LOGD(nodeName, "EpwindowSize = %lu", maxWindowSizeEp);
    OP_TILING_CHECK(CheckTpWinSize(context, tilingData, nodeName, tokenNeedSizeDispatch, tokenNeedSizeCombine) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "Tiling check Tp window size failed."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckCombineOrARN(gert::TilingContext* context, MoeDistributeCombineV2TilingData &tilingData, const char *nodeName, const CombineV2Config& config)
{
    const gert::StorageShape *expertIdsStorageShape = context->GetInputShape(config.expertIdsIndex);
    const gert::StorageShape *expandXStorageShape = context->GetInputShape(config.expandXIndex);
    int64_t expertIdsDim0 = expertIdsStorageShape->GetStorageShape().GetDim(0);
    int64_t expandXDim1 = expandXStorageShape->GetStorageShape().GetDim(1);
    if (config.hasAddRmsNorm) {
        // 校验combineARN新增的输入与输出
        OP_TILING_CHECK(CheckARNAttrAndSetTilingData(context, tilingData, nodeName, config) != ge::GRAPH_SUCCESS,
            OP_LOGE(nodeName, "attr check or set tilingdata failed."), return ge::GRAPH_FAILED);
        // 校验combineARN的输入与输出shape
        OP_TILING_CHECK(!CheckTensorShapeARN(context, nodeName, config, expertIdsDim0, expandXDim1),
            OP_LOGE(nodeName, "CheckTensorShapeARN failed"), return ge::GRAPH_FAILED);
        // 校验combineARN新增的输入维数
        OP_TILING_CHECK(!CheckInputTensorDimARN(context, nodeName, config),
            OP_LOGE(nodeName, "CheckInputTensorDimARN failed"), return ge::GRAPH_FAILED);
        // 校验combineARN的输出维数
        OP_TILING_CHECK(!CheckOutputTensorDimARN(context, nodeName, config),
            OP_LOGE(nodeName, "param shape of output tensor is invalid"), return ge::GRAPH_FAILED);
    } else {
        // 校验combine输出的维数与shape
        OP_TILING_CHECK(!CheckOutputTensorDim(context, nodeName, config, expertIdsDim0, expandXDim1),
            OP_LOGE(nodeName, "param shape of output tensor is invalid"), return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus MoeDistributeCombineA3TilingFuncImpl(gert::TilingContext* context, const CombineV2Config& config)
{
    const char *nodeName = context->GetNodeName();
    OP_LOGD(nodeName, "Enter MoeDistributeCombineV2 Tiling func");
    MoeDistributeCombineV2TilingData *tilingData = context->GetTilingData<MoeDistributeCombineV2TilingData>();
    OP_TILING_CHECK(tilingData == nullptr, OP_LOGE(nodeName, "tilingData is nullptr."), return ge::GRAPH_FAILED);
    std::string groupEp = "", groupTp = "";
    bool isShared = true, isLayered = false, isActiveMask = false, hasElasticInfo = false;
    uint32_t localMoeExpertNum = 1, commQuantMode = 0U;

    // 获取入参属性
    OP_TILING_CHECK(GetAttrAndSetTilingData(context, *tilingData, nodeName, groupEp, groupTp, commQuantMode, config, isLayered) == ge::GRAPH_FAILED,
        OP_LOGE(nodeName, "Getting attr failed."), return ge::GRAPH_FAILED);
    const gert::StorageShape *xActiveMaskStorageShape = context->GetOptionalInputShape(config.xActiveMaskIndex);
    isActiveMask = (xActiveMaskStorageShape != nullptr);
    tilingData->moeDistributeCombineV2Info.isTokenMask = ((isActiveMask) &&
        (xActiveMaskStorageShape->GetStorageShape().GetDimNum() == ONE_DIM));
    tilingData->moeDistributeCombineV2Info.isExpertMask = ((isActiveMask) &&
        (xActiveMaskStorageShape->GetStorageShape().GetDimNum() == TWO_DIMS));

    // 获取elasticInfo
    const gert::StorageShape *elasticInfoStorageShape = context->GetOptionalInputShape(config.elasticInfoIndex);
    hasElasticInfo = (elasticInfoStorageShape != nullptr);
    tilingData->moeDistributeCombineV2Info.hasElasticInfo = hasElasticInfo;
    // 检查输入输出的dim、format、dataType
    uint32_t tpWorldSize = tilingData->moeDistributeCombineV2Info.tpWorldSize;
    OP_TILING_CHECK(TilingCheckMoeDistributeCombine(context, nodeName, isActiveMask, hasElasticInfo, tpWorldSize,
                    config, isLayered) !=
                    ge::GRAPH_SUCCESS, OP_LOGE(nodeName, "Tiling check params failed"), return ge::GRAPH_FAILED);

    // 检查属性的取值是否合法
    OP_TILING_CHECK(!CheckAttrs(context, *tilingData, nodeName, localMoeExpertNum, isActiveMask, config, isLayered),
        OP_LOGE(nodeName, "attr check failed."), return ge::GRAPH_FAILED);

    uint32_t sharedExpertRankNum = tilingData->moeDistributeCombineV2Info.sharedExpertRankNum;
    uint32_t epRankId = tilingData->moeDistributeCombineV2Info.epRankId;
    isShared = (epRankId < sharedExpertRankNum);

    // 检查shape各维度并赋值h,k
    OP_TILING_CHECK(!CheckTensorShape(context, *tilingData, nodeName, isShared, isActiveMask, localMoeExpertNum, hasElasticInfo, config, isLayered),
        OP_LOGE(nodeName, "param dim check failed."), return ge::GRAPH_FAILED);
    
    // 校验combine或combineARN有差异的参数
    OP_TILING_CHECK(CheckCombineOrARN(context, *tilingData, nodeName, config) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "CheckCombineOrARN failed."), return ge::GRAPH_FAILED);
    
    // 校验win区大小
    if (!config.isMc2Context) {
        OP_TILING_CHECK(CheckWinSize(context, tilingData, nodeName, localMoeExpertNum, isLayered) != ge::GRAPH_SUCCESS,
            OP_LOGE(nodeName, "Tiling check window size failed."), return ge::GRAPH_FAILED);
    } else {
        auto attrs = context->GetAttrs();
        auto cclBuffSizePtr = attrs->GetAttrPointer<int64_t>(static_cast<int>(config.attrCclBufferSizeIndex));
        OP_TILING_CHECK(cclBuffSizePtr == nullptr || *cclBuffSizePtr < 0,
            OP_LOGE(K_INNER_DEBUG, "cclBuffSizePtr is invalid."), return GRAPH_FAILED);
        tilingData->moeDistributeCombineV2Info.totalWinSizeEp = *cclBuffSizePtr;
    }

    OP_TILING_CHECK(SetWorkspace(context, nodeName) != ge::GRAPH_SUCCESS,
                    VECTOR_INNER_ERR_REPORT_TILING(context->GetNodeName(), "Tiling set workspace Failed"),
                    return ge::GRAPH_FAILED);

    if (!config.isMc2Context) {
        OP_TILING_CHECK(SetHCommCfg(context, tilingData, groupEp, groupTp, tpWorldSize, isLayered) != ge::GRAPH_SUCCESS,
            OP_LOGE(nodeName, "SetHCommCfg failed."), return ge::GRAPH_FAILED);
    }
    tilingData->moeDistributeCombineV2Info.isMc2Context = config.isMc2Context;
    uint64_t tilingKey = CalTilingKey(tpWorldSize, commQuantMode, isLayered);
    OP_LOGD(nodeName, "tilingKey is %lu", tilingKey);
    context->SetTilingKey(tilingKey);
    uint32_t numBlocks = 1U;

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint64_t aivNum = ascendcPlatform.GetCoreNumAiv();
    OP_TILING_CHECK((isLayered && (aivNum != AIV_NUM_93)), OP_LOGE(nodeName, "Layered must be fullCore."), return ge::GRAPH_FAILED);
    uint64_t ubSize = 0UL;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    numBlocks = ascendcPlatform.CalcTschBlockDim(aivNum, 0, aivNum);
    context->SetBlockDim(numBlocks);
    tilingData->moeDistributeCombineV2Info.aivNum = aivNum;
    tilingData->moeDistributeCombineV2Info.totalUbSize = ubSize;
    UbUsedCal(ubSize, context, tilingData, config);
    context->SetScheduleMode(1); // 设置为batch mode模式，所有核同时启动
    OP_LOGD(nodeName, "numBlocks = %u, aivNum = %lu, ubsize = %lu", numBlocks, aivNum, ubSize);
    PrintTilingDataInfo(nodeName, *tilingData);

    return ge::GRAPH_SUCCESS;
}

// a2专有
static void PrintA2TilingDataInfo(MoeDistributeCombineA2Info& info)
{
    OP_LOGD(K_INNER_DEBUG, "epWorldSize is %u.", info.epWorldSize);
    OP_LOGD(K_INNER_DEBUG, "tpWorldSize is %u.", info.tpWorldSize);
    OP_LOGD(K_INNER_DEBUG, "epRankId is %u.", info.epRankId);
    OP_LOGD(K_INNER_DEBUG, "tpRankId is %u.", info.tpRankId);
    OP_LOGD(K_INNER_DEBUG, "expertSharedType is %u.", info.expertSharedType);
    OP_LOGD(K_INNER_DEBUG, "sharedExpertRankNum is %u.", info.sharedExpertRankNum);
    OP_LOGD(K_INNER_DEBUG, "moeExpertNum is %u.", info.moeExpertNum);
    OP_LOGD(K_INNER_DEBUG, "globalBs is %u.", info.globalBs);
}

static ge::graphStatus MoeDistributeCombineA2CheckAttrAndSetTiling(const gert::TilingContext* context,
                                                                   MoeDistributeCombineA2Info& info,
                                                                   int32_t& commQuantMode,
                                                                   const bool isLayered, const CombineV2Config& config)
{
    auto attrs = context->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE(K_INNER_DEBUG, "attrs is null."), return ge::GRAPH_FAILED);

    auto groupEpPtr = attrs->GetAttrPointer<char>(static_cast<int>(ATTR_GROUP_EP_INDEX));
    auto epWorldSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_EP_WORLD_SIZE_INDEX);
    auto epRankIdPtr = attrs->GetAttrPointer<int64_t>(ATTR_EP_RANK_ID_INDEX);
    auto moeExpertNumPtr = attrs->GetAttrPointer<int64_t>(ATTR_MOE_EXPERT_NUM_INDEX);
    auto tpWorldSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_TP_WORLD_SIZE_INDEX);
    auto tpRankIdPtr = attrs->GetAttrPointer<int64_t>(ATTR_TP_RANK_ID_INDEX);
    auto expertSharedTypePtr = attrs->GetAttrPointer<int64_t>(ATTR_EXPERT_SHARD_TYPE_INDEX);
    auto sharedExpertRankNumPtr = attrs->GetAttrPointer<int64_t>(ATTR_SHARED_EXPERT_RANK_NUM_INDEX);
    auto globalBsPtr = attrs->GetAttrPointer<int64_t>(ATTR_GLOBAL_BS_INDEX);
    auto commQuantModePtr = attrs->GetAttrPointer<int64_t>(ATTR_COMM_QUANT_MODE_INDEX);

    auto zeroExpertNumPtr = attrs->GetAttrPointer<int64_t>(static_cast<int>(config.attrZeroExpertNumIndex));
    auto copyExpertNumPtr = attrs->GetAttrPointer<int64_t>(static_cast<int>(config.attrCopyExpertNumIndex));
    auto constExpertNumPtr = attrs->GetAttrPointer<int64_t>(static_cast<int>(config.attrConstExpertNumIndex));

    OP_TILING_CHECK(zeroExpertNumPtr == nullptr, OP_LOGE(K_INNER_DEBUG, "zeroExpertNum is invalid."), return GRAPH_FAILED);
    OP_TILING_CHECK(copyExpertNumPtr == nullptr, OP_LOGE(K_INNER_DEBUG, "copyExpertNum is invalid."), return GRAPH_FAILED);
    OP_TILING_CHECK(constExpertNumPtr == nullptr || *constExpertNumPtr != 0,
        OP_LOGE(K_INNER_DEBUG, "constExpertNum is invalid. Must be 0."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK((groupEpPtr == nullptr) || (strnlen(groupEpPtr, MAX_GROUP_NAME_LENGTH) == 0) ||
        (strnlen(groupEpPtr, MAX_GROUP_NAME_LENGTH) == MAX_GROUP_NAME_LENGTH),
        OP_LOGE(K_INNER_DEBUG, "groupEp is invalid."), return ge::GRAPH_FAILED);
    int32_t maxEpWorldSizeA2 = MAX_EP_WORLD_SIZE_A2;
    if (isLayered) {
        maxEpWorldSizeA2 = MAX_EP_WORLD_SIZE_A2_LAYERED;
    }
    OP_TILING_CHECK(epWorldSizePtr == nullptr || *epWorldSizePtr <= 0 || *epWorldSizePtr > maxEpWorldSizeA2 ||
        ((*epWorldSizePtr > RANK_NUM_PER_NODE_A2) && (*epWorldSizePtr % RANK_NUM_PER_NODE_A2 != 0)),
        OP_LOGE(K_INNER_DEBUG, "epWorldSize is invalid."), return GRAPH_FAILED);
    OP_TILING_CHECK(epRankIdPtr == nullptr || *epRankIdPtr < 0 || *epRankIdPtr >= *epWorldSizePtr,
        OP_LOGE(K_INNER_DEBUG, "epRankId is invalid."), return GRAPH_FAILED);
    OP_TILING_CHECK(moeExpertNumPtr == nullptr || *moeExpertNumPtr <= 0 || *moeExpertNumPtr > MAX_MOE_EXPERT_NUMS_A2 ||
        *moeExpertNumPtr % *epWorldSizePtr != 0,
        OP_LOGE(K_INNER_DEBUG, "moeExpertNum is invalid."), return GRAPH_FAILED);
    OP_TILING_CHECK(tpWorldSizePtr == nullptr, OP_LOGE(K_INNER_DEBUG, "tpWorldSize is null."), return GRAPH_FAILED);
    OP_TILING_CHECK(tpRankIdPtr == nullptr, OP_LOGE(K_INNER_DEBUG, "tpRankId is null."), return GRAPH_FAILED);
    OP_TILING_CHECK(expertSharedTypePtr == nullptr, OP_LOGE(K_INNER_DEBUG, "expertSharedType is null."), return GRAPH_FAILED);
    OP_TILING_CHECK(sharedExpertRankNumPtr == nullptr, OP_LOGE(K_INNER_DEBUG, "sharedExpertRankNum is null."), return GRAPH_FAILED);
    OP_TILING_CHECK(globalBsPtr == nullptr, OP_LOGE(K_INNER_DEBUG, "globalBs is null."), return GRAPH_FAILED);
    OP_TILING_CHECK(commQuantModePtr == nullptr, OP_LOGE(K_INNER_DEBUG, "commQuantMode is null."), return GRAPH_FAILED);
    OP_TILING_CHECK(!isLayered && *commQuantModePtr != static_cast<CommQuantModeType>(CommQuantMode::NON_QUANT),
        OP_LOGE(K_INNER_DEBUG, "commQuantMode is invalid."), return GRAPH_FAILED);
    OP_TILING_CHECK(isLayered && *commQuantModePtr != static_cast<CommQuantModeType>(CommQuantMode::NON_QUANT) &&
        *commQuantModePtr != static_cast<CommQuantModeType>(CommQuantMode::INT8_QUANT),
        OP_LOGE(K_INNER_DEBUG, "commQuantMode is invalid."), return GRAPH_FAILED);

    const gert::StorageShape *expertIdStorageShape = context->GetInputShape(config.expertIdsIndex);
    OP_TILING_CHECK(expertIdStorageShape == nullptr, OP_LOGE(K_INNER_DEBUG, "xShape is null."), return false);
    int32_t globalBs = *epWorldSizePtr * expertIdStorageShape->GetStorageShape().GetDim(0);

    // 判断是否满足uint32_t及其他限制
    int64_t moeExpertNum = static_cast<int64_t>(*moeExpertNumPtr);
    int64_t zeroExpertNum = *zeroExpertNumPtr;
    int64_t copyExpertNum = *copyExpertNumPtr;
    int64_t constExpertNum = 0LL;
    OP_TILING_CHECK(
        (moeExpertNum + zeroExpertNum + copyExpertNum + constExpertNum) > INT32_MAX,
        OP_LOGE(K_INNER_DEBUG, "moeExpertNum + zeroExpertNum + copyExpertNum + constExpertNum exceeds MAX_INT32."),
        return ge::GRAPH_FAILED);
    info.epWorldSize = *epWorldSizePtr;
    info.tpWorldSize = static_cast<uint32_t>(0);
    info.epRankId = *epRankIdPtr;
    info.tpRankId = static_cast<uint32_t>(0);
    info.expertSharedType = static_cast<uint32_t>(0);
    info.sharedExpertRankNum = static_cast<uint32_t>(0);
    info.moeExpertNum = *moeExpertNumPtr;

    info.zeroExpertNum = static_cast<uint32_t>(zeroExpertNum);
    info.copyExpertNum = static_cast<uint32_t>(copyExpertNum);
    info.constExpertNum = static_cast<uint32_t>(constExpertNum);

    if (*globalBsPtr == 0) {
        info.globalBs = static_cast<uint32_t>(globalBs);
    } else {
        info.globalBs = *globalBsPtr;
    }
    commQuantMode = *commQuantModePtr;
    PrintA2TilingDataInfo(info);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus MoeDistributeCombineA2CheckShapeAndSetTiling(const gert::TilingContext *context,
                                                                    MoeDistributeCombineA2Info &info, bool isLayered, const CombineV2Config& config)
{
    const gert::StorageShape *expandXStorageShape = context->GetInputShape(config.expandXIndex);
    const gert::StorageShape *expertIdStorageShape = context->GetInputShape(config.expertIdsIndex);
    const gert::StorageShape *xActiveMaskStorageShape = context->GetOptionalInputShape(config.xActiveMaskIndex);
    const gert::StorageShape *elasticInfoStorageShape = context->GetOptionalInputShape(config.elasticInfoIndex);
    OP_TILING_CHECK(expandXStorageShape == nullptr, OP_LOGE(K_INNER_DEBUG, "expandXShape is null."), return GRAPH_FAILED);
    OP_TILING_CHECK(expertIdStorageShape == nullptr, OP_LOGE(K_INNER_DEBUG, "expertIdShape is null."), return GRAPH_FAILED);
    OP_TILING_CHECK(elasticInfoStorageShape != nullptr, OP_LOGE(K_INNER_DEBUG, "current does not support elasticInfo as input"), return GRAPH_FAILED);

    // copy expert and const expert
    const gert::StorageShape* oriXStorageShape = context->GetOptionalInputShape(config.oriXIndex);
    const gert::StorageShape* constExpertAlpha1StorageShape = context->GetOptionalInputShape(config.constExpertAlpha1Index);
    const gert::StorageShape* constExpertAlpha2StorageShape = context->GetOptionalInputShape(config.constExpertAlpha2Index);
    const gert::StorageShape* constExpertVStorageShape = context->GetOptionalInputShape(config.constExpertVIndex);

    OP_TILING_CHECK(expandXStorageShape->GetStorageShape().GetDimNum() != TWO_DIMS,
        OP_LOGE(K_INNER_DEBUG, "expandXshape is invalid"), return GRAPH_FAILED);
    uint32_t h = expandXStorageShape->GetStorageShape().GetDim(1);
    uint32_t maxHiddenSizeA2 = isLayered ? LAYERED_MAX_HIDDEN_SIZE_A2 : MAX_HIDDEN_SIZE_A2;
    OP_TILING_CHECK(h == 0 || h > maxHiddenSizeA2 || h % BLOCK_SIZE_A2 != 0,
        OP_LOGE(K_INNER_DEBUG, "hiddensize is invalid."), return GRAPH_FAILED);
    OP_TILING_CHECK(expertIdStorageShape->GetStorageShape().GetDimNum() != TWO_DIMS,
        OP_LOGE(K_INNER_DEBUG, "expertIdshape is invalid"), return GRAPH_FAILED);
    uint32_t bs = expertIdStorageShape->GetStorageShape().GetDim(0);
    uint32_t maxBatchSizeA2 = isLayered ? LAYERED_MAX_BATCH_SIZE_A2 : MAX_BATCH_SIZE_A2;
    OP_TILING_CHECK(bs == 0 || bs > maxBatchSizeA2,
        OP_LOGE(K_INNER_DEBUG, "batchsize is invalid."), return GRAPH_FAILED);

    uint32_t k = expertIdStorageShape->GetStorageShape().GetDim(1);
    auto attrs = context->GetAttrs();
    auto moeExpertNumPtr = attrs->GetAttrPointer<int64_t>(ATTR_MOE_EXPERT_NUM_INDEX);
    auto zeroExpertNumPtr = attrs->GetAttrPointer<int64_t>(static_cast<int>(config.attrZeroExpertNumIndex));
    auto copyExpertNumPtr = attrs->GetAttrPointer<int64_t>(static_cast<int>(config.attrCopyExpertNumIndex));
    auto constExpertNumPtr = attrs->GetAttrPointer<int64_t>(static_cast<int>(config.attrConstExpertNumIndex));
    // 判断是否满足uint32_t及其他限制
    int32_t moeExpertNum = *moeExpertNumPtr;
    int32_t zeroExpertNum = static_cast<int32_t>(*zeroExpertNumPtr);
    int32_t copyExpertNum = static_cast<int32_t>(*copyExpertNumPtr);
    int32_t constExpertNum = 0;
    OP_TILING_CHECK(k == 0 || k > MAX_K_VALUE_A2 || k > moeExpertNum
        + zeroExpertNum + copyExpertNum + constExpertNum,
        OP_LOGE(K_INNER_DEBUG, "k is invalid."), return GRAPH_FAILED);

    bool isActiveMask = (xActiveMaskStorageShape != nullptr);
    if (isActiveMask) {
        const int64_t xActiveMaskDimNums = xActiveMaskStorageShape->GetStorageShape().GetDimNum();
        OP_TILING_CHECK(((xActiveMaskDimNums != ONE_DIM) && (xActiveMaskDimNums != TWO_DIMS)),
            OP_LOGE(K_INNER_DEBUG, "xActiveMask must be 1-dimension or 2-dimension, but got %lu dim",
            xActiveMaskDimNums), return GRAPH_FAILED);

        int64_t xActiveMaskDim0 = xActiveMaskStorageShape->GetStorageShape().GetDim(0);
        OP_TILING_CHECK(xActiveMaskDim0 != static_cast<int64_t>(bs),
            OP_LOGE(K_INNER_DEBUG, "xActiveMask's dim0 not equal to expertIds's dim0, xActiveMask's dim0 is %ld, "
            "expertIds's dim0 is %ld", xActiveMaskDim0, bs), return GRAPH_FAILED);

        OP_TILING_CHECK(((xActiveMaskStorageShape->GetStorageShape().GetDimNum() == TWO_DIMS) &&
            (xActiveMaskStorageShape->GetStorageShape().GetDim(1) != static_cast<int64_t>(k))),
            OP_LOGE(K_INNER_DEBUG, "xActiveMask's dim1 not equal to expertIds's dim1, xActiveMask's dim1 is %ld, "
            "expertIds's dim1 is %ld", xActiveMaskStorageShape->GetStorageShape().GetDim(1), k), return GRAPH_FAILED);
    }

    // copy expert and const expert
    OP_TILING_CHECK(copyExpertNum > 0 && oriXStorageShape == nullptr,
        OP_LOGE(K_INNER_DEBUG, "oriX must be exist when copyExpertNum > 0"), return GRAPH_FAILED);
    OP_TILING_CHECK(constExpertNum > 0 && (oriXStorageShape == nullptr || constExpertAlpha1StorageShape == nullptr ||
                    constExpertAlpha2StorageShape == nullptr || constExpertVStorageShape == nullptr),
        OP_LOGE(K_INNER_DEBUG, "oriX、alpha1、alpha2、V must be exist when constExpertNum > 0"), return GRAPH_FAILED);

    OP_TILING_CHECK(constExpertAlpha1StorageShape != nullptr || constExpertAlpha2StorageShape != nullptr || constExpertVStorageShape != nullptr,
        OP_LOGE(K_INNER_DEBUG, "current version does not support const_expert_alpha_1, const_expert_alpha_2, and const_expert_v."),
        return GRAPH_FAILED);

    if (oriXStorageShape != nullptr) {
        // 必须是2维
        OP_TILING_CHECK(
            oriXStorageShape->GetStorageShape().GetDimNum() != TWO_DIMS,
            OP_LOGE(
                K_INNER_DEBUG, "ori_x must be 2-dimension, but got %lu dim",
                oriXStorageShape->GetStorageShape().GetDimNum()),
            return GRAPH_FAILED);

        // shape为(bs, h)
        int64_t oriXDim0 = oriXStorageShape->GetStorageShape().GetDim(0);
        int64_t oriXDim1 = oriXStorageShape->GetStorageShape().GetDim(1);
        OP_TILING_CHECK(
            oriXDim0 != static_cast<int64_t>(bs),
            OP_LOGE(K_INNER_DEBUG, "ori_x's dim0 not equal to bs, ori_x's dim0 = %ld, bs = %ld",
                    oriXDim0, static_cast<int64_t>(bs)),
            return GRAPH_FAILED);
        OP_TILING_CHECK(
            oriXDim1 != static_cast<int64_t>(h),
            OP_LOGE(K_INNER_DEBUG, "ori_x's dim1 not equal to h, ori_x's dim1 = %ld, h = %ld",
                    oriXDim1, static_cast<int64_t>(h)),
            return GRAPH_FAILED);
    }

    if (constExpertAlpha1StorageShape != nullptr) {
        // 必须是1维
        OP_TILING_CHECK(
            constExpertAlpha1StorageShape->GetStorageShape().GetDimNum() != ONE_DIM,
            OP_LOGE(
                K_INNER_DEBUG, "const_expert_alpha_1 must be 1-dimension, but got %lu dim",
                constExpertAlpha1StorageShape->GetStorageShape().GetDimNum()),
            return GRAPH_FAILED);

        // shape为(constExpertNum)
        int64_t constExpertAlpha1Dim0 = constExpertAlpha1StorageShape->GetStorageShape().GetDim(0);
        OP_TILING_CHECK(
            constExpertAlpha1Dim0 != *constExpertNumPtr,
            OP_LOGE(
                K_INNER_DEBUG,
                "const_expert_alpha_1's dim0 not equal to const_expert_num, const_expert_alpha_1's dim0 = %ld, "
                "const_expert_num = %ld",
                constExpertAlpha1Dim0, *constExpertNumPtr),
            return GRAPH_FAILED);
    }

    if (constExpertAlpha2StorageShape != nullptr) {
        // 必须是1维
        OP_TILING_CHECK(
            constExpertAlpha2StorageShape->GetStorageShape().GetDimNum() != ONE_DIM,
            OP_LOGE(
                K_INNER_DEBUG, "const_expert_alpha_2 must be 1-dimension, but got %lu dim",
                constExpertAlpha2StorageShape->GetStorageShape().GetDimNum()),
            return GRAPH_FAILED);

        // shape为(constExpertNum)
        int64_t constExpertAlpha2Dim0 = constExpertAlpha2StorageShape->GetStorageShape().GetDim(0);
        OP_TILING_CHECK(
            constExpertAlpha2Dim0 != *constExpertNumPtr,
            OP_LOGE(
                K_INNER_DEBUG,
                "const_expert_alpha_2's dim0 not equal to const_expert_num, const_expert_alpha_2's dim0 = %ld, "
                "const_expert_num = %ld", constExpertAlpha2Dim0, *constExpertNumPtr),
            return GRAPH_FAILED);
    }

    if (constExpertVStorageShape != nullptr) {
        // 必须是2维
        OP_TILING_CHECK(
            constExpertVStorageShape->GetStorageShape().GetDimNum() != TWO_DIMS,
            OP_LOGE(
                K_INNER_DEBUG, "const_expert_v must be 2-dimension, but got %lu dim",
                constExpertVStorageShape->GetStorageShape().GetDimNum()),
            return GRAPH_FAILED);
        // 必须是2维(constExpertNum, H)
        int64_t constExpertVDim0 = constExpertVStorageShape->GetStorageShape().GetDim(0);
        int64_t constExpertVDim1 = constExpertVStorageShape->GetStorageShape().GetDim(1);
        OP_TILING_CHECK(constExpertVDim0 != *constExpertNumPtr,
            OP_LOGE(
                K_INNER_DEBUG,
                "const_expert_v's dim0 not equal to const_expert_num, const_expert_v's dim0 = %ld, const_expert_num = %ld",
                constExpertVDim0, *constExpertNumPtr), return GRAPH_FAILED);
        OP_TILING_CHECK(
            constExpertVDim1 != static_cast<int64_t>(h),
            OP_LOGE(
                K_INNER_DEBUG, "const_expert_v's dim1 not equal to h, const_expert_v's dim1 = %ld, h = %ld",
                constExpertVDim1, static_cast<int64_t>(h)),
            return GRAPH_FAILED);
    }

    attrs = context->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE(K_INNER_DEBUG, "attrs is null."), return ge::GRAPH_FAILED);

    info.isTokenMask =  ((isActiveMask) && (xActiveMaskStorageShape->GetStorageShape().GetDimNum() == ONE_DIM));
    info.isExpertMask = ((isActiveMask) && (xActiveMaskStorageShape->GetStorageShape().GetDimNum() == TWO_DIMS));
    info.bs = bs;
    info.k = k;
    info.h = h;

    OP_LOGD(K_INNER_DEBUG, "batchSize is %u", bs);
    OP_LOGD(K_INNER_DEBUG, "k is %u", k);
    OP_LOGD(K_INNER_DEBUG, "hiddenSize is %u", h);
    OP_LOGD(K_INNER_DEBUG, "isTokenMask is %d", static_cast<int32_t>(info.isTokenMask));
    OP_LOGD(K_INNER_DEBUG, "isExpertMask is %d", static_cast<int32_t>(info.isExpertMask));

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus MoeDistributeCombineA2GetPlatformInfoAndSetTiling(const gert::TilingContext *context, MoeDistributeCombineA2Info& info)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint64_t ubSize = 0U;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);

    info.aivNum = aivNum;
    info.totalUbSize = ubSize;

    OP_LOGD(K_INNER_DEBUG, "aivNum=%d", info.aivNum);
    OP_LOGD(K_INNER_DEBUG, "ubSize=%lu", info.totalUbSize);

    return ge::GRAPH_SUCCESS;
}

// 为了兼容老版本，在未配置commAlg参数时，读取环境变量；
// commAlg参数当前支持"fullmesh"和"hierarchy"两种，其余报错。
static ge::graphStatus MoeDistributeCombineCheckCommAlg(const gert::TilingContext *context, bool &isLayered,
    const CombineV2Config &config)
{
    isLayered = false;
    auto attrs = context->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE(K_INNER_DEBUG, "attrs is null."), return ge::GRAPH_FAILED);
    auto commAlg = attrs->GetAttrPointer<char>(static_cast<int>((config.attrCommAlgIndex)));
    auto epWorldSizePtr = attrs->GetAttrPointer<int64_t>((config.attrEpWorldSizeIndex));
    if ((epWorldSizePtr != nullptr) && (*epWorldSizePtr <= RANK_NUM_PER_NODE_A2)) {
        isLayered = false;
        OP_LOGD(K_INNER_DEBUG, "epWorldSize <= 8, use default fullmesh algorithm.");
        return ge::GRAPH_SUCCESS;
    }
    if (commAlg == nullptr || strlen(commAlg) == 0 || strcmp(commAlg, "0") == 0) {
        OP_LOGW(K_INNER_DEBUG, "Attr commAlg is invalid, please configure fullmesh or hierarchy.");

        const char* hcclIntraPcieEnable = getenv("HCCL_INTRA_PCIE_ENABLE");
        const char* hcclIntraRoceEnable = getenv("HCCL_INTRA_ROCE_ENABLE");
        if (hcclIntraPcieEnable != nullptr && hcclIntraRoceEnable != nullptr &&
            strcmp(hcclIntraPcieEnable, "1") == 0 && strcmp(hcclIntraRoceEnable, "0") == 0) {
            OP_LOGD(K_INNER_DEBUG, "ENV HCCL_INTRA_PCIE_ENABLE = 1 and HCCL_INTRA_ROCE_ENABLE = 0, use hierarchy algorithm.");
            isLayered = true;
        } else {
            OP_LOGD(K_INNER_DEBUG, "ENV HCCL_INTRA_PCIE_ENABLE != 1 or HCCL_INTRA_ROCE_ENABLE != 0, use default fullmesh algorithm.");
        }
        return ge::GRAPH_SUCCESS;
    }

    OP_LOGI(K_INNER_DEBUG, "commAlg is %s", commAlg);

    if (strcmp(commAlg, "fullmesh") == 0) {
        return ge::GRAPH_SUCCESS;
    } else if (strcmp(commAlg, "hierarchy") == 0) {
        isLayered = true;
        return ge::GRAPH_SUCCESS;
    } else {
        OP_LOGE(K_INNER_DEBUG, "commAlg is not support");
        return GRAPH_FAILED;
    }
}

static uint64_t MoeDistributeCombineA2CalcTilingKey(const bool isLayered, const int32_t commQuantMode)
{
    bool tp = false;
    uint32_t quantMode = TILINGKEY_NO_QUANT;
    uint32_t layeredMode = TILINGKEY_TPL_MTE;  // A2

    if (isLayered) {
        layeredMode = TILINGKEY_TPL_AICPU;
        if (commQuantMode == static_cast<CommQuantModeType>(CommQuantMode::INT8_QUANT)) {
            quantMode = TILINGKEY_INT8_QUANT;
        }
    }
    uint64_t tilingKey = GET_TPL_TILING_KEY(tp, quantMode, layeredMode, TILINGKEY_TPL_A2);
    OP_LOGD(K_INNER_DEBUG, "tilingKey=%lu", tilingKey);
    return tilingKey;
}

static std::string MoeDistributeCombineA2GetAlgConfig(int32_t epWorldSize, bool isLayered)
{
    if (epWorldSize <= RANK_NUM_PER_NODE_A2) {
        return "BatchWrite=level0:fullmesh";
    }
    return isLayered ? "BatchWrite=level1:hierarchy" : "BatchWrite=level1:fullmesh";
}

static ge::graphStatus MoeDistributeCombineA2CheckWinSize(const gert::TilingContext *context, const char *nodeName,
                                                          MoeDistributeCombineA2Info &info, bool isLayered)
{
    auto groupEp = context->GetAttrs()->GetAttrPointer<char>(ATTR_GROUP_EP_INDEX);
    uint64_t hcclBuffSize = 0ULL;
    auto ret = mc2tiling::GetCclBufferSize(groupEp, &hcclBuffSize, nodeName);
    OP_LOGD(nodeName, "HCCL_BUFFSIZE = %lu Bytes (%lu MB).", hcclBuffSize, ops::CeilDiv(hcclBuffSize, MB_SIZE));
    OP_TILING_CHECK(ret != ge::GRAPH_SUCCESS, OP_LOGE(nodeName, "Get Ep hcclBuffSize failed.", hcclBuffSize),
                    return ge::GRAPH_FAILED);
    uint32_t epWorldSize = info.epWorldSize;
    uint32_t localMoeExpertNum = info.moeExpertNum / epWorldSize;
    uint64_t maxBs = static_cast<uint64_t>(info.globalBs) / epWorldSize;
    uint64_t minHcclBuffSize = 0ULL;
    constexpr uint64_t sizeofDtypeX = 2ULL; // token数据类型为float16/bfloat16，每个元素字节数为2
    constexpr uint64_t BUFFER_NUM = 2UL;
    if (isLayered) {
        constexpr uint64_t BUFFER_ALIGN = 512UL;
        constexpr uint64_t flagBuffSize = 8 * MB_SIZE; // 固定8M空间作为存放同步Flag的区域
        // 每个token发往k个专家时额外需带上专家索引、topk权重、量化系数、到达标志位共4个信息，这些信息对齐到32字节
        const uint64_t extraTokenInfoSize = 4 * ((info.k + 7) / 8 * 8) * sizeof(uint32_t);
        const uint64_t perTokenSize = info.h * sizeofDtypeX + extraTokenInfoSize;
        uint64_t maxRecvTokenNum = maxBs * (info.moeExpertNum + epWorldSize / RANK_NUM_PER_NODE_A2 * BUFFER_NUM);
        maxRecvTokenNum = (maxRecvTokenNum + BUFFER_ALIGN - 1) / BUFFER_ALIGN * BUFFER_ALIGN;
        minHcclBuffSize = maxRecvTokenNum * perTokenSize + flagBuffSize;
        if (minHcclBuffSize > hcclBuffSize) {
            OP_LOGE(nodeName,
                    "HCCL_BUFFSIZE is too small, min required HCCL_BUFFSIZE ((moeExpertNum + epWorldSize / 4) * Align512(maxBs "
                    "* (h * 2 + 16 * Align8(k))) / 1MB + 8MB) = %luMB, actual HCCL_BUFFSIZE = %luMB, "
                    "moeExpertNum = %u, maxBs = %lu, h = %u, k = %u. AlignY(x) = (x + Y - 1) / Y * Y.",
                    ops::CeilDiv(minHcclBuffSize, MB_SIZE), ops::CeilDiv(hcclBuffSize, MB_SIZE), info.moeExpertNum,
                    maxBs, info.h, info.k);
            return ge::GRAPH_FAILED;
        }
    } else {
        constexpr uint64_t extraBuffSize = 2 * MB_SIZE; // 固定2M额外空间作为存储非数据信息的区域
        const uint64_t perTokenSize = info.h * sizeofDtypeX;
        const uint64_t maxRecvTokenNum = maxBs * epWorldSize * std::min(localMoeExpertNum, info.k);
        minHcclBuffSize = BUFFER_NUM * (maxRecvTokenNum * perTokenSize + extraBuffSize);
        if (minHcclBuffSize > hcclBuffSize) {
            OP_LOGE(nodeName,
                    "HCCL_BUFFSIZE is too small, min required HCCL_BUFFSIZE (%lu * (maxBs * epWorldSize * "
                    "min(localMoeExpertNum, k) * h * 2 / 1MB + 2MB)) = %luMB, actual HCCL_BUFFSIZE = %luMB, maxBs = "
                    "%lu, epWorldSize = %u, localMoeExpertNum = %u, k = %u, h = %u.",
                    BUFFER_NUM, ops::CeilDiv(minHcclBuffSize, MB_SIZE), ops::CeilDiv(hcclBuffSize, MB_SIZE), maxBs,
                    epWorldSize, localMoeExpertNum, info.k, info.h);
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus MoeDistributeCombineA2TilingFuncImpl(gert::TilingContext* context, const CombineV2Config& config)
{
    const char *nodeName = context->GetNodeName();
    OP_LOGI(nodeName, "Enter MoeDistributeCombineA2 tiling func.");
    // 涉及SyncAll，设置batch mode模式，所有核同时启动 
    uint32_t batch_mode = 1U; 
    auto ret = context->SetScheduleMode(batch_mode); 
    GE_ASSERT_GRAPH_SUCCESS(ret);

    // tilingData
    MoeDistributeCombineA2TilingData *tilingData = context->GetTilingData<MoeDistributeCombineA2TilingData>();
    OP_TILING_CHECK(tilingData == nullptr, VECTOR_INNER_ERR_REPORT_TILING(nodeName, "tilingData is nullptr."),
        return ge::GRAPH_FAILED);
    MoeDistributeCombineA2Info &info = tilingData->moeDistributeCombineInfo;

    bool isLayered = false;
    OP_TILING_CHECK(MoeDistributeCombineCheckCommAlg(context, isLayered, config) != ge::GRAPH_SUCCESS,
        VECTOR_INNER_ERR_REPORT_TILING(context->GetNodeName(), "MoeDistributeCombineA2 CheckCommAlg Failed"),
        return ge::GRAPH_FAILED);
    int32_t commQuantMode = 0;
    OP_TILING_CHECK(MoeDistributeCombineA2CheckShapeAndSetTiling(context, info, isLayered, config) != ge::GRAPH_SUCCESS,
        VECTOR_INNER_ERR_REPORT_TILING(context->GetNodeName(), "MoeDistributeCombineA2 CheckShapeAndSetTiling Failed"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(MoeDistributeCombineA2CheckAttrAndSetTiling(context, info, commQuantMode, isLayered, config) != ge::GRAPH_SUCCESS,
        VECTOR_INNER_ERR_REPORT_TILING(context->GetNodeName(), "MoeDistributeCombineA2 CheckAttrAndSetTiling Failed"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(MoeDistributeCombineA2GetPlatformInfoAndSetTiling(context, info) != ge::GRAPH_SUCCESS,
        VECTOR_INNER_ERR_REPORT_TILING(context->GetNodeName(), "MoeDistributeCombineA2 GetPlatformInfoAndSetTiling Failed"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(MoeDistributeCombineA2CheckWinSize(context, nodeName, info, isLayered) != ge::GRAPH_SUCCESS,
        VECTOR_INNER_ERR_REPORT_TILING(context->GetNodeName(), "MoeDistributeCombineA2 CheckWinSize Failed"),
        return ge::GRAPH_FAILED);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint32_t numBlocks = ascendcPlatform.CalcTschBlockDim(aivNum, 0, aivNum);
    context->SetBlockDim(numBlocks);
    uint32_t aicpuBlockDim = info.epWorldSize > RANK_NUM_PER_NODE_A2 ? mc2tiling::AICPU_NUM_BLOCKS_A2 : 1;
    context->SetAicpuBlockDim(aicpuBlockDim);

    uint64_t tilingKey = MoeDistributeCombineA2CalcTilingKey(isLayered, commQuantMode);
    context->SetTilingKey(tilingKey);
    // 2. workspace
    size_t *workSpaces = context->GetWorkspaceSizes(1);
    OP_TILING_CHECK(workSpaces == nullptr, VECTOR_INNER_ERR_REPORT_TILING(nodeName, "workSpaces is nullptr."),
        return ge::GRAPH_FAILED);
    size_t userWorkspaceSize = info.moeExpertNum * sizeof(uint32_t) * 2U;
    workSpaces[0] = SYSTEM_NEED_WORKSPACE + userWorkspaceSize;

    // 3. communication
    auto attrs = context->GetAttrs();
    auto group = attrs->GetAttrPointer<char>(static_cast<int>(ATTR_GROUP_EP_INDEX));
    auto epWorldSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_EP_WORLD_SIZE_INDEX);
    std::string algConfig = MoeDistributeCombineA2GetAlgConfig(*epWorldSizePtr, isLayered);
    uint32_t opType = 18; // DispatchCombine

    AscendC::Mc2CcTilingConfig mc2CcTilingConfig(group, opType, algConfig);
    OP_TILING_CHECK(mc2CcTilingConfig.GetTiling(tilingData->mc2InitTiling) != 0,
        OP_LOGE(nodeName, "mc2CcTilingConfig mc2tiling GetTiling mc2InitTiling failed"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(mc2CcTilingConfig.GetTiling(tilingData->mc2CcTiling) != 0,
        OP_LOGE(nodeName, "mc2CcTilingConfig mc2tiling GetTiling mc2CcTiling failed"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus MoeDistributeCombineA5TilingFuncImpl(gert::TilingContext* context, const CombineV2Config& config)
{
    auto attrs = context->GetAttrs();
    const char *nodeName = context->GetNodeName();
    auto commAlgPtr = attrs->GetAttrPointer<char>(static_cast<int>((config.attrCommAlgIndex)));
    // 检查 commAlg 参数合法性校验
    bool isNullOrEmpty = (commAlgPtr == nullptr) || (std::strlen(commAlgPtr) == 0);
    bool isCcu = std::strcmp(commAlgPtr, "ccu") == 0;
    OP_TILING_CHECK(!(isNullOrEmpty || isCcu),
        OP_LOGE(nodeName, "Invalid parameter: 'commAlg'='%s'."
            "'commAlg' is not supported on this SoC version. Nullptr (or empty char*) is acceptable.", commAlgPtr),
        return ge::GRAPH_FAILED);
    if (isCcu) {
        // CCU 调用 A5 tiling 实现
        return MoeDistributeCombineTilingImpl(context, OP_VERSION_2);
    }
    // 默认空指针和空字符走 MTE 方式，MTE 调用 A3 tiling 实现
    return MoeDistributeCombineA3TilingFuncImpl(context, config);
}

ge::graphStatus MoeDistributeCombineV2TilingFuncNew(gert::TilingContext* context, const CombineV2Config& config)
{
    // 不支持 expandX数据类型为int32 type
    auto expandXDesc = context->GetInputDesc(config.expandXIndex);
    const char *nodeName = context->GetNodeName();
    OP_TILING_CHECK(expandXDesc == nullptr, OP_LOGE(nodeName, "expandxDesc is null."), return ge::GRAPH_FAILED);
    // 检查expandX数据类型为DT_INT32
    OP_TILING_CHECK((expandXDesc->GetDataType() == ge::DT_INT32),
                    OP_LOGE(nodeName, "expandX dataType is invalid, dataType should be bf16 or float16, but is %d",
                    static_cast<ge::DataType>(expandXDesc->GetDataType())), return ge::GRAPH_FAILED);

    std::string socVersion = mc2tiling::GetSocVersion(context);
    NpuArch npuArch = mc2tiling::GetNpuArch(context);
    ge::graphStatus ret;
    if (socVersion == "Ascend910B") {
        ret = MoeDistributeCombineA2TilingFuncImpl(context, config);
    } else if (npuArch == NpuArch::DAV_3510) {
        ret = MoeDistributeCombineA5TilingFuncImpl(context, config);
    } else {
        ret = MoeDistributeCombineA3TilingFuncImpl(context, config);
    }

    return ret;
}

ge::graphStatus MoeDistributeCombineV2TilingFunc(gert::TilingContext* context)
{
    auto rstdOutDesc = context->GetOutputDesc(HAS_ADD_RMS_NORM);
    CombineV2Config config;
    ge::graphStatus ret;
    if (rstdOutDesc == nullptr) {
        config.tpSendCountsIndex = 5; // 根据combineV2算子原型标志位设置tpSendCounts索引为5
        config.xActiveMaskIndex = 6; // 根据combineV2算子原型标志位设置xActiveMask索引为6
        config.activationScaleIndex = 7; // 根据combineV2算子原型标志位设置activationScale索引为7
        config.weightScaleIndex = 8; // 根据combineV2算子原型标志位设置weightScale索引为8
        config.groupListIndex = 9; // 根据combineV2算子原型标志位设置groupList索引为9
        config.sharedExpertXIndex = 11; // 根据combineV2算子原型标志位设置sharedExpertX索引为11
        config.elasticInfoIndex = 12; // 根据combineV2算子原型标志位设置elasticInfo索引为12
        config.oriXIndex = 13; // 根据combineV2算子原型标志位设置oriX索引为13
        config.constExpertAlpha1Index = 14; // 根据combineV2算子原型标志位设置constExpertAlpha1索引为14
        config.constExpertAlpha2Index = 15; // 根据combineV2算子原型标志位设置constExpertAlpha2索引为15
        config.constExpertVIndex = 16; // 根据combineV2算子原型标志位设置constExpertV索引为16
        config.outputXIndex = 0; // 根据combineV2算子原型标志位设置outputX索引为0
        config.attrGroupEpIndex = 0;  // 0: 根据combineV2算子原型标志位初始化groupEp索引
        config.attrEpWorldSizeIndex = 1;  // 1: 根据combineV2算子原型标志位初始化epWorldSize索引
        config.attrEpRankIdIndex = 2; // 2: 根据combineV2算子原型标志位初始化epRankId索引
        config.attrMoeExpertNumIndex = 3; // 3: 根据combineV2算子原型标志位初始化moeExpertNum索引
        config.attrGroupTpIndex = 4; // 4: 根据combineV2算子原型标志位初始化attrGroupTpIndex索引
        config.attrTpWorldSizeIndex = 5; // 5: 根据combineV2算子原型标志位初始化attrTpWorldSizeIndex索引
        config.attrTpRankIdIndex = 6; // 6: 根据combineV2算子原型标志位初始化attrTpRankIdIndex索引
        config.attrExpertSharedTypeIndex = 7; // 7: 根据combineV2算子原型标志位初始化attrExpertSharedTypeIndex索引
        config.attrSharedExpertNumIndex = 8; // 8: 根据combineV2算子原型标志位初始化attrSharedExpertNumIndex索引
        config.attrSharedExpertRankNumIndex = 9; // 9: 根据combineV2算子原型标志位初始化attrSharedExpertRankNumIndex索引
        config.attrGlobalBsIndex  = 10; // 10: 根据combineV2算子原型标志位初始化attrGlobalBsIndex索引
        config.attrOutDTypeIndex = 11; // 11: 根据combineV2算子原型标志位初始化attrOutDTypeIndex索引
        config.attrCommQuantModeIndex = 12; // 12: 根据combineV2算子原型标志位初始化attrCommQuantModeIndex索引
        config.attrGroupListTypeIndex = 13; // 13: 根据combineV2算子原型标志位初始化attrGroupListTypeIndex索引
        config.attrCommAlgIndex = 14; // 14: 根据combineV2算子原型标志位初始化attrCommAlgIndex索引
        config.attrZeroExpertNumIndex = 15; // 根据combineV2算子原型标志位设置attrZeroExpertNum索引为15
        config.attrCopyExpertNumIndex = 16; // 根据combineV2算子原型标志位设置attrCopyExpertNum索引为16
        config.attrConstExpertNumIndex = 17; // 根据combineV2算子原型标志位设置attrConstExpertNum索引为17
        config.hasAddRmsNorm = false;
    } else {
        config.residualXIndex = 5; // 根据combineARN算子原型标志位设置residualX索引为5
        config.gammaIndex = 6; // 根据combineARN算子原型标志位设置gamma索引为6
        config.tpSendCountsIndex = 7; // 根据combineARN算子原型标志位设置tpSendCounts索引为7
        config.xActiveMaskIndex = 8; // 根据combineARN算子原型标志位设置xActiveMask索引为8
        config.activationScaleIndex = 9; // 根据combineARN算子原型标志位设置activationScale索引为9
        config.weightScaleIndex = 10; // 根据combineARN算子原型标志位设置weightScale索引为10
        config.groupListIndex = 11; // 根据combineARN算子原型标志位设置groupList索引为11
        config.sharedExpertXIndex = 13; // 根据combineARN算子原型标志位设置sharedExpertX索引为13
        config.elasticInfoIndex = 14; // 根据combineARN算子原型标志位设置elasticInfo索引为14
        config.oriXIndex = 15; // 根据combineARN算子原型标志位设置oriX索引为15
        config.constExpertAlpha1Index = 16; // 根据combineARN算子原型标志位设置constExpertAlpha1索引为16
        config.constExpertAlpha2Index = 17; // 根据combineARN算子原型标志位设置constExpertAlpha2索引为17
        config.constExpertVIndex = 18; // 根据combineARN算子原型标志位设置constExpertV索引为18
        config.outputYIndex = 0; // 根据combineARN算子原型标志位设置outputY索引为0
        config.outputRstdIndex = 1; // 根据combineARN算子原型标志位设置outputRstd索引为1
        config.outputXIndex = 2; // 根据combineARN算子原型标志位设置outputX索引为2
        config.attrGroupEpIndex = 0;  // 0: 根据combineV2算子原型标志位初始化groupEp索引
        config.attrEpWorldSizeIndex = 1;  // 1: 根据combineV2算子原型标志位初始化epWorldSize索引
        config.attrEpRankIdIndex = 2; // 2: 根据combineV2算子原型标志位初始化epRankId索引
        config.attrMoeExpertNumIndex = 3; // 3: 根据combineV2算子原型标志位初始化moeExpertNum索引
        config.attrGroupTpIndex = 4; // 4: 根据combineV2算子原型标志位初始化attrGroupTpIndex索引
        config.attrTpWorldSizeIndex = 5; // 5: 根据combineV2算子原型标志位初始化attrTpWorldSizeIndex索引
        config.attrTpRankIdIndex = 6; // 6: 根据combineV2算子原型标志位初始化attrTpRankIdIndex索引
        config.attrExpertSharedTypeIndex = 7; // 7: 根据combineV2算子原型标志位初始化attrExpertSharedTypeIndex索引
        config.attrSharedExpertNumIndex = 8; // 8: 根据combineV2算子原型标志位初始化attrSharedExpertNumIndex索引
        config.attrSharedExpertRankNumIndex = 9; // 9: 根据combineV2算子原型标志位初始化attrSharedExpertRankNumIndex索引
        config.attrGlobalBsIndex  = 10; // 10: 根据combineV2算子原型标志位初始化attrGlobalBsIndex索引
        config.attrOutDTypeIndex = 11; // 11: 根据combineV2算子原型标志位初始化attrOutDTypeIndex索引
        config.attrCommQuantModeIndex = 12; // 12: 根据combineV2算子原型标志位初始化attrCommQuantModeIndex索引
        config.attrGroupListTypeIndex = 13; // 13: 根据combineV2算子原型标志位初始化attrGroupListTypeIndex索引
        config.attrCommAlgIndex = 14; // 14: 根据combineV2算子原型标志位初始化attrCommAlgIndex索引
        config.attrNormEpsIndex = 15; // 根据combineARN算子原型标志位设置attrNormEps索引为15
        config.attrZeroExpertNumIndex = 16; // 根据combineARN算子原型标志位设置attrZeroExpertNum索引为16
        config.attrCopyExpertNumIndex = 17; // 根据combineARN算子原型标志位设置attrCopyExpertNum索引为17
        config.attrConstExpertNumIndex = 18; // 根据combineARN算子原型标志位设置attrConstExpertNum索引为18
        config.hasAddRmsNorm = true;
    }
    ret = MoeDistributeCombineV2TilingFuncNew(context, config);
    
    return ret;
}

struct MoeDistributeCombineCompileInfo {};
ge::graphStatus TilingParseForMoeDistributeCombineV2(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(MoeDistributeCombineV2)
    .Tiling(MoeDistributeCombineV2TilingFunc)
    .TilingParse<MoeDistributeCombineCompileInfo>(TilingParseForMoeDistributeCombineV2);

#ifdef MC2_EXCEPTION_HANDLER
// Register exception func
inline void MoeDistributeCombineV2ExceptionImplWrapper(aclrtExceptionInfo *args, void *userdata)
{
    Mc2ExceptionImpl(args, userdata, "MoeDistributeCombineV2");
}

IMPL_OP(MoeDistributeCombineV2)
    .ExceptionDumpParseFunc(MoeDistributeCombineV2ExceptionImplWrapper);
#endif

} // namespace optiling
