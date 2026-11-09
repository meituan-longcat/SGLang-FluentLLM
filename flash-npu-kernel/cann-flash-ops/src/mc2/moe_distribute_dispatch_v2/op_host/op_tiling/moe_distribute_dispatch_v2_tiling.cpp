/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_distribute_dispatch_v2_tiling.cpp
 * \brief
 */

#include "moe_distribute_dispatch_tiling_v2.h"

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

#include "tiling/mc2_tiling_utils.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "mc2_log.h"
#include "graph/utils/type_utils.h"
#include "register/op_def_registry.h"
#include "platform/platform_infos_def.h"
#include "../../op_kernel/moe_distribute_dispatch_tiling.h"
#include "arch35/moe_distribute_dispatch_tiling_arch35.h"
#include "../../op_kernel/moe_distribute_dispatch_v2_tiling.h"
#include "../../op_kernel/moe_distribute_dispatch_v2_tiling_key.h"
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
    constexpr uint32_t X_INDEX = 0U;
    constexpr uint32_t EXPERT_IDS_INDEX = 1U;
    constexpr uint32_t SCALES_INDEX = 2U;
    constexpr uint32_t X_ACTIVE_MASK_INDEX = 3U;
    constexpr uint32_t EXPERT_SCALES_INDEX = 4U;
    constexpr uint32_t ELASTIC_INFO_INDEX = 5U;
    constexpr uint32_t OUTPUT_EXPAND_X_INDEX = 0U;
    constexpr uint32_t OUTPUT_DYNAMIC_SCALES_INDEX = 1U;
    constexpr uint32_t OUTPUT_ASSIST_INFO_INDEX = 2U;
    constexpr uint32_t OUTPUT_EXPERT_TOKEN_NUMS_INDEX = 3U;
    constexpr uint32_t OUTPUT_EP_RECV_COUNTS_INDEX = 4U;
    constexpr uint32_t OUTPUT_TP_RECV_COUNTS_INDEX = 5U;
    constexpr uint32_t OUTPUT_EXPAND_SCALES_INDEX = 6U;

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
    constexpr uint32_t ATTR_QUANT_MODE_INDEX = 10;
    constexpr uint32_t ATTR_GLOBAL_BS_INDEX = 11;
    constexpr uint32_t ATTR_EXPERT_TOKEN_NUMS_TYPE_INDEX = 12;
    constexpr uint32_t ATTR_COMM_ALG_INDEX = 13;
    constexpr uint32_t ATTR_ZERO_EXPERT_NUM_INDEX = 14;
    constexpr uint32_t ATTR_COPY_EXPERT_NUM_INDEX = 15;
    constexpr uint32_t ATTR_CONST_EXPERT_NUM_INDEX = 16;

    constexpr uint32_t TWO_DIMS = 2;
    constexpr uint32_t HALF_NUM = 2;    // cumsum最多只能用一半的核
    constexpr uint32_t ONE_DIM = 1;
    constexpr uint32_t DYN_SCALE_DIMS = 1;
    constexpr uint32_t ASSIST_INFO_DIMS = 1;
    constexpr uint32_t DYNAMIC_SCALE_DIM_NUM = 1;
    constexpr uint64_t INIT_TILINGKEY = 10000;
    constexpr uint32_t ARR_LENGTH = 128;
    constexpr uint32_t OP_TYPE_ALL_TO_ALL = 8;
    constexpr uint32_t OP_TYPE_ALL_GATHER = 6;

    constexpr size_t MAX_GROUP_NAME_LENGTH = 128UL;
    constexpr int64_t MAX_SHARED_EXPERT_NUM = 4;
    constexpr int64_t MAX_EP_WORLD_SIZE = 768L; // 384 * 2
    constexpr int64_t MAX_EP_WORLD_SIZE_LAYERED = 256;
    constexpr int64_t MIN_EP_WORLD_SIZE = 2;
    constexpr int64_t EP_RESTRICT_8 = 8;
    constexpr int64_t MAX_TP_WORLD_SIZE = 2;
    constexpr int64_t MAX_TP_WORLD_SIZE_LAYERED = 1;
    constexpr int64_t BS_UPPER_BOUND = 512;
    constexpr int64_t BS_UPPER_BOUND_LAYERED = 256;
    constexpr int64_t FULLMESH_BS_UPPER_BOUND = 256;
    constexpr uint32_t H_BASIC_BLOCK_LAYERED = 32;
    constexpr uint32_t RANK_NUM_PER_NODE = 16U;
    constexpr uint32_t AIV_NUM_93 = 48U;

    constexpr uint64_t NUM_10 = 10ULL;
    constexpr uint32_t TILINGKEY_SCALES = 10;
    constexpr uint32_t TILINGKEY_TP_WORLD_SIZE = 100;
    constexpr uint32_t TILINGKEY_COMM_ALG = 1000;
    constexpr uint32_t TP_WORLD_SIZE_TWO = 2;
    constexpr uint32_t VERSION_2 = 2;
    constexpr uint32_t HCOMMCNT_2 = 2;
    constexpr uint32_t SIZE_OF_UNQUANT = 2;
    constexpr uint32_t SIZE_OF_HALF = 2;
    constexpr uint32_t SIZE_ALIGN_256 = 256;
    constexpr uint32_t SPLIT_BLOCK_DATA_SIZE = 480U;
    constexpr uint32_t SPLIT_BLOCK_SIZE = 512UL;
    constexpr uint32_t ELASTIC_INFO_OFFSET = 4U;
    constexpr uint32_t BUFFER_NUM = 2;
    constexpr int64_t MOE_EXPERT_MAX_NUM = 1024;
    constexpr int64_t MOE_EXPERT_MAX_NUM_LAYERED = 512;
    constexpr int64_t LOCAL_EXPERT_MAX_SIZE = 2048;
    constexpr int64_t K_MAX = 16;
    constexpr int64_t FULLMESH_K_MAX = 12;
    constexpr size_t SYSTEM_NEED_WORKSPACE = 16UL * 1024UL * 1024UL;
    constexpr uint32_t WORKSPACE_ELEMENT_OFFSET = 512;
    constexpr uint32_t RANK_LIST_NUM = 2;
    constexpr int32_t HCCL_BUFFER_SIZE_DEFAULT = 200 * 1024 * 1024; // Bytes
    constexpr int64_t H_MIN = 1024;
    constexpr int64_t H_MAX = 8192;
    constexpr int64_t H_MAX_LAYERED = 7168;
    constexpr uint64_t MB_SIZE = 1024UL * 1024UL;
    constexpr uint64_t TRIPLE = 3;
    constexpr uint64_t ASSIST_NUM_PER_A = 128;
    constexpr uint64_t WIN_ADDR_ALIGN = 512UL;
    constexpr uint64_t FULL_MESH_DATA_ALIGN = 480UL;
    constexpr uint64_t SCALE_EXPAND_IDX_BUFFER = 44UL; // scale32B + 3*4expandIdx
    constexpr uint64_t DOUBLE_DATA_BUFFER = 2UL;
    constexpr uint64_t MAX_OUT_DTYPE_SIZE = 2UL;
    constexpr uint64_t UB_ALIGN = 32UL;
    constexpr uint64_t EVEN_ALIGN = 2;
    constexpr int64_t ELASTIC_METAINFO_OFFSET = 4;
    constexpr int64_t CEIL_ALIGN32 = 8;

    constexpr uint64_t STATIC_SCALE_DIM_0 = 1;
    constexpr uint64_t ONE_DIM_SCALE_COL_NUM = 1;
    constexpr uint64_t MX_BLOCK_SIZE = 32U;
    constexpr uint64_t PERGROUP_BLOCK_SIZE = 128U;

    // A2定义
    const char *K_INNER_DEBUG = "MoeDistributeDispatchV2 Tiling Debug";
    constexpr uint32_t RANK_NUM_PER_NODE_A2 = 8;
    constexpr uint32_t BLOCK_SIZE_A2 = 32;
    constexpr uint32_t MAX_K_VALUE_A2 = 16;
    constexpr uint32_t MAX_HIDDEN_SIZE_A2 = 8192;
    constexpr uint32_t LAYERED_MAX_HIDDEN_SIZE_A2 = 10240;
    constexpr int32_t MAX_EP_WORLD_SIZE_A2 = 384;
    constexpr int32_t MAX_EP_WORLD_SIZE_A2_LAYERED = 128;
    constexpr int32_t MAX_MOE_EXPERT_NUMS_A2 = 1024;
    constexpr uint32_t MAX_BATCH_SIZE_A2 = 1024;
    constexpr uint32_t LAYERED_MAX_BATCH_SIZE_A2 = 512;
    constexpr uint32_t INT4_PACKED_H_A2 = 1024;
    constexpr uint64_t INT4_SCALE_SLOT_SIZE_A2 = 32;
    constexpr uint64_t ROUTE_ENCODING_SIZE_A2 = 32;
    constexpr uint64_t DATA_OFFSET_A2 = 512;
    constexpr uint64_t DATA_FLAG_SIZE_A2 = 128;
    constexpr uint64_t STATE_SIZE_A2 = 1UL * 1024UL * 1024UL;
    constexpr uint64_t BUFFER_NUM_A2 = 2;
    constexpr size_t USER_WORKSPACE_A2 = 1UL * 1024UL * 1024UL; // moeExpertNum_ * sizeof(uint32_t) + epWorldSize_ * 2 * 32
    constexpr uint64_t TILING_KEY_BASE_A2 = 2000000000;
    constexpr uint64_t TILING_KEY_LAYERED_COMM_A2 = 100000000;
    constexpr uint64_t INIT_TILINGKEY_A2 = 1000;

    // A5
    constexpr uint32_t OP_VERSION_2 = 2;
}

// Supported x datatype in nonquant mode, the same as expandX
static const std::set<ge::DataType> NON_QUANT_DTYPE = {
    ge::DT_FLOAT16, ge::DT_BF16, ge::DT_HIFLOAT8, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2};

namespace optiling {
static void PrintTilingDataInfo(const char *nodeName, MoeDistributeDispatchV2TilingData &tilingData)
{
    OP_LOGD(nodeName, "epWorldSize is %u.", tilingData.moeDistributeDispatchV2Info.epWorldSize);
    OP_LOGD(nodeName, "tpWorldSize is %u.", tilingData.moeDistributeDispatchV2Info.tpWorldSize);
    OP_LOGD(nodeName, "epRankId is %u.", tilingData.moeDistributeDispatchV2Info.epRankId);
    OP_LOGD(nodeName, "tpRankId is %u.", tilingData.moeDistributeDispatchV2Info.tpRankId);
    OP_LOGD(nodeName, "expertShardType is %u.", tilingData.moeDistributeDispatchV2Info.expertShardType);
    OP_LOGD(nodeName, "sharedExpertNum is %u.", tilingData.moeDistributeDispatchV2Info.sharedExpertNum);
    OP_LOGD(nodeName, "sharedExpertRankNum is %u.", tilingData.moeDistributeDispatchV2Info.sharedExpertRankNum);
    OP_LOGD(nodeName, "moeExpertNum is %u.", tilingData.moeDistributeDispatchV2Info.moeExpertNum);
    OP_LOGD(nodeName, "quantMode is %u.", tilingData.moeDistributeDispatchV2Info.quantMode);
    OP_LOGD(nodeName, "globalBs is %u.", tilingData.moeDistributeDispatchV2Info.globalBs);
    OP_LOGD(nodeName, "bs is %u.", tilingData.moeDistributeDispatchV2Info.bs);
    OP_LOGD(nodeName, "k is %u.", tilingData.moeDistributeDispatchV2Info.k);
    OP_LOGD(nodeName, "h is %u.", tilingData.moeDistributeDispatchV2Info.h);
    OP_LOGD(nodeName, "aivNum is %u.", tilingData.moeDistributeDispatchV2Info.aivNum);
    OP_LOGD(nodeName, "totalUbSize is %lu.", tilingData.moeDistributeDispatchV2Info.totalUbSize);
    OP_LOGD(nodeName, "totalWinSizeEP is %lu.", tilingData.moeDistributeDispatchV2Info.totalWinSizeEp);
    OP_LOGD(nodeName, "totalWinSizeTP is %lu.", tilingData.moeDistributeDispatchV2Info.totalWinSizeTp);
    OP_LOGD(nodeName, "hasElastic is %d.", tilingData.moeDistributeDispatchV2Info.hasElasticInfo);
    OP_LOGD(nodeName, "zeroComputeExpertNum is %d", tilingData.moeDistributeDispatchV2Info.zeroComputeExpertNum);
}

static bool CheckDynamicScalesDim(const gert::TilingContext *context,
    const char *nodeName, const uint32_t quantMode)
{
     const gert::StorageShape *dynamicScalesStorageShape = context->GetOutputShape(OUTPUT_DYNAMIC_SCALES_INDEX);
        OP_TILING_CHECK(dynamicScalesStorageShape == nullptr,
            OP_LOGE(nodeName, "dynamicScalesShape is null."), return false);
    if ((quantMode == static_cast<uint32_t>(QuantModeA5::PERTOKEN_DYNAMIC_QUANT))) {
        // quantMode 2: 1dim, the same in A2/A3/A5
        OP_TILING_CHECK(dynamicScalesStorageShape->GetStorageShape().GetDimNum() != DYNAMIC_SCALE_ONE_DIM_NUM,
            OP_LOGE(nodeName, "dynamicScalesShape dims must be %u when quantMode=%u, but current dim num is %lu.",
            DYNAMIC_SCALE_ONE_DIM_NUM, quantMode, dynamicScalesStorageShape->GetStorageShape().GetDimNum()), return false);
        OP_LOGD(nodeName, "dynamicScales dim0 = %ld", dynamicScalesStorageShape->GetStorageShape().GetDim(0));
    } else {
        // MX/PERGROUP
        OP_TILING_CHECK(dynamicScalesStorageShape->GetStorageShape().GetDimNum() != DYNAMIC_SCALE_TWO_DIM_NUM,
            OP_LOGE(nodeName, "dynamicScalesShape dims must be %u when quantMode=%u, but current dim num is %lu.",
            DYNAMIC_SCALE_TWO_DIM_NUM, quantMode, dynamicScalesStorageShape->GetStorageShape().GetDimNum()), return false);
        OP_LOGD(nodeName, "dynamicScales dim0=%ld, dim1=%ld", 
            dynamicScalesStorageShape->GetStorageShape().GetDim(0), 
            dynamicScalesStorageShape->GetStorageShape().GetDim(1));
    }
    return true;
}

static bool CheckScaleTensorDim(const gert::TilingContext *context, const char *nodeName,
    const bool isScales, const uint32_t quantMode, DispatchV2Config &config)
{
    if (isScales) {
        const gert::StorageShape *scalesStorageShape = context->GetOptionalInputShape(config.scalesIndex);
        OP_TILING_CHECK(scalesStorageShape == nullptr, OP_LOGE(nodeName, "scalesShape is null."), return false);
        if (quantMode != static_cast<uint32_t>(QuantModeA5::STATIC_QUANT)) {
            // the cond is compatible with A2/A3 because static quant is only supported on A5
            OP_TILING_CHECK(scalesStorageShape->GetStorageShape().GetDimNum() != TWO_DIMS,
                OP_LOGE(nodeName, "scales dims must be 2 when quantMode=%u, but current dim num is %lu.",
                quantMode, scalesStorageShape->GetStorageShape().GetDimNum()), return false);
            OP_LOGD(nodeName, "scales dim0 = %ld", scalesStorageShape->GetStorageShape().GetDim(0));
            OP_LOGD(nodeName, "scales dim1 = %ld", scalesStorageShape->GetStorageShape().GetDim(1));
        } else {
            OP_TILING_CHECK((scalesStorageShape->GetStorageShape().GetDimNum() != ONE_DIM)
                && (scalesStorageShape->GetStorageShape().GetDimNum() != TWO_DIMS),
                OP_LOGE(nodeName, "scalesShape dims must be 1 or 2 when quantMode is 1, but current dim num is %lu.",
                scalesStorageShape->GetStorageShape().GetDimNum()), return false);
            // additional check for hif8 quant
            auto expandXDesc = context->GetOutputDesc(OUTPUT_EXPAND_X_INDEX);
            OP_TILING_CHECK(expandXDesc == nullptr, OP_LOGE(nodeName, "expandXDesc is null."), return false);
            OP_TILING_CHECK((expandXDesc->GetDataType() == ge::DT_HIFLOAT8) && (scalesStorageShape->GetStorageShape().GetDimNum() != ONE_DIM),
                OP_LOGE(nodeName, "scalesShape dims must be 1 when x dtype is hif8 in static quant, but current dim num is %lu.",
                scalesStorageShape->GetStorageShape().GetDimNum()), return false);
            OP_LOGD(nodeName, "scales dim0 = %ld", scalesStorageShape->GetStorageShape().GetDim(0));
            if (scalesStorageShape->GetStorageShape().GetDimNum() == TWO_DIMS) {
                OP_LOGD(nodeName, "scales dim1 = %ld", scalesStorageShape->GetStorageShape().GetDim(1));
            }
        }
    }
    return true;
}

//x, expertIds, scales维度校验
static bool CheckInputTensorDim(const gert::TilingContext *context, const char *nodeName,
    const bool isScales, const uint32_t quantMode, const bool isLayered, DispatchV2Config &config)
{
    const gert::StorageShape *xStorageShape = context->GetInputShape(config.xIndex);
    OP_TILING_CHECK(xStorageShape == nullptr, OP_LOGE(nodeName, "xShape is null."), return false);
    OP_TILING_CHECK(xStorageShape->GetStorageShape().GetDimNum() != TWO_DIMS,
        OP_LOGE(nodeName, "xShape dims must be 2, but current dim num is %lu.",
        xStorageShape->GetStorageShape().GetDimNum()), return false);
    int64_t xDim0 = xStorageShape->GetStorageShape().GetDim(0);
    int64_t xDim1 = xStorageShape->GetStorageShape().GetDim(1);
    OP_LOGD(nodeName, "x dim0 = %ld", xDim0);
    OP_LOGD(nodeName, "x dim1 = %ld", xDim1);

    const gert::StorageShape *expertIdStorageShape = context->GetInputShape(config.expertIdsIndex);
    if (isLayered) {
        const gert::StorageShape *expertScaleStorageShape = context->GetOptionalInputShape(config.expertScalesIndex);
        OP_TILING_CHECK(expertScaleStorageShape == nullptr, OP_LOGE(nodeName, "expertScaleShape is null."), return false);
        OP_TILING_CHECK(expertScaleStorageShape->GetStorageShape().GetDimNum() != TWO_DIMS,
        OP_LOGE(nodeName, "expertScaleShape dims must be 2, but current dim num is %lu.",
        expertScaleStorageShape->GetStorageShape().GetDimNum()), return false);
        const int64_t expertScalesDim0 = expertScaleStorageShape->GetStorageShape().GetDim(0);
        const int64_t expertScalesDim1 = expertScaleStorageShape->GetStorageShape().GetDim(1);
        OP_LOGD(nodeName, "expertScales dim0 = %ld", expertScalesDim0);
        OP_LOGD(nodeName, "expertScales dim1 = %ld", expertScalesDim1);
    }
    OP_TILING_CHECK(expertIdStorageShape == nullptr, OP_LOGE(nodeName, "expertIdShape is null."), return false);
    OP_TILING_CHECK(expertIdStorageShape->GetStorageShape().GetDimNum() != TWO_DIMS,
        OP_LOGE(nodeName, "expertIdShape dims must be 2, but current dim num is %lu.",
        expertIdStorageShape->GetStorageShape().GetDimNum()), return false);
    const int64_t expertIdDim0 = expertIdStorageShape->GetStorageShape().GetDim(0);
    const int64_t expertIdDim1 = expertIdStorageShape->GetStorageShape().GetDim(1);
    OP_LOGD(nodeName, "expertId dim0 = %ld", expertIdDim0);
    OP_LOGD(nodeName, "expertId dim1 = %ld", expertIdDim1);
    OP_TILING_CHECK(!CheckScaleTensorDim(context, nodeName, isScales, quantMode, config), 
        OP_LOGE(nodeName, "isScale Input param shape is invalid."), return false);
    return true;
}

//expertX, assistInfo, expertTokenNums, epRecvCount, tpRecvCount维度校验 
static bool CheckCommonOutputTensorDim(const gert::TilingContext *context, const char *nodeName,
    const uint32_t quantMode)
{
    const gert::StorageShape *expandXStorageShape = context->GetOutputShape(OUTPUT_EXPAND_X_INDEX);
    OP_TILING_CHECK(expandXStorageShape == nullptr, OP_LOGE(nodeName, "expandXShape is null."), return false);
    OP_TILING_CHECK(expandXStorageShape->GetStorageShape().GetDimNum() != TWO_DIMS,
        OP_LOGE(nodeName, "expandXShape dims must be 2, but current dim num is %lu.",
        expandXStorageShape->GetStorageShape().GetDimNum()), return false);
    OP_LOGD(nodeName, "expandX dim0 = %ld", expandXStorageShape->GetStorageShape().GetDim(0));
    OP_LOGD(nodeName, "expandX dim1 = %ld", expandXStorageShape->GetStorageShape().GetDim(1));

    if ((quantMode != static_cast<uint32_t>(QuantModeA5::NON_QUANT)) && (quantMode != static_cast<uint32_t>(QuantModeA5::STATIC_QUANT))) {
        OP_TILING_CHECK(!CheckDynamicScalesDim(context, nodeName, quantMode),OP_LOGE(nodeName, "CheckDynamicScalesDim failed."), return false);
    }

    const gert::StorageShape *assistInfoStorageShape = context->GetOutputShape(OUTPUT_ASSIST_INFO_INDEX);
    OP_TILING_CHECK(assistInfoStorageShape == nullptr, OP_LOGE(nodeName, "assistInfoShape is null."), return false);
    OP_TILING_CHECK(assistInfoStorageShape->GetStorageShape().GetDimNum() != ONE_DIM,
        OP_LOGE(nodeName, "assistInfoShape dims must be 1, but current dim num is %lu.",
        assistInfoStorageShape->GetStorageShape().GetDimNum()), return false);
    OP_LOGD(nodeName, "assistInfoForCombine dim0 = %ld", assistInfoStorageShape->GetStorageShape().GetDim(0));

    const gert::StorageShape *expertTokenNumsStorageShape = context->GetOutputShape(OUTPUT_EXPERT_TOKEN_NUMS_INDEX);
    OP_TILING_CHECK(expertTokenNumsStorageShape == nullptr,
        OP_LOGE(nodeName, "expertTokenNumsShape is null."), return false);
    OP_TILING_CHECK(expertTokenNumsStorageShape->GetStorageShape().GetDimNum() != ONE_DIM,
        OP_LOGE(nodeName, "expertTokenNumsShape dims must be 1, but current dim num is %lu.",
        expertTokenNumsStorageShape->GetStorageShape().GetDimNum()), return false);
    OP_LOGD(nodeName, "expertTokenNums dim0 = %ld", expertTokenNumsStorageShape->GetStorageShape().GetDim(0));

    const gert::StorageShape *epRecvCountStorageShape = context->GetOutputShape(OUTPUT_EP_RECV_COUNTS_INDEX);
    OP_TILING_CHECK(epRecvCountStorageShape == nullptr, OP_LOGE(nodeName, "epRecvCountShape is null."), return false);
    OP_TILING_CHECK(epRecvCountStorageShape->GetStorageShape().GetDimNum() != ONE_DIM,
        OP_LOGE(nodeName, "epRecvCountShape dims must be 1, but current dim num is %lu.",
        epRecvCountStorageShape->GetStorageShape().GetDimNum()), return false);
    OP_LOGD(nodeName, "epRecvCount dim0 = %ld", epRecvCountStorageShape->GetStorageShape().GetDim(0));

    const gert::StorageShape *tpRecvCountStorageShape = context->GetOutputShape(OUTPUT_TP_RECV_COUNTS_INDEX);
    OP_TILING_CHECK(tpRecvCountStorageShape == nullptr,
        OP_LOGE(nodeName, "tpRecvCountShape is null."), return false);
    OP_TILING_CHECK(tpRecvCountStorageShape->GetStorageShape().GetDimNum() != ONE_DIM,
        OP_LOGE(nodeName, "tpRecvCountShape dims must be 1, but current dim num is %lu.",
        tpRecvCountStorageShape->GetStorageShape().GetDimNum()), return false);
    OP_LOGD(nodeName, "tpRecvCount dim0 = %ld", tpRecvCountStorageShape->GetStorageShape().GetDim(0));

    return true;
}    

static bool CheckTensorDim(const gert::TilingContext *context, const char *nodeName,
    const bool isScales, const uint32_t quantMode, const bool isActiveMask, const bool hasElasticInfo,
    const bool isLayered, DispatchV2Config &config)
{
    OP_TILING_CHECK(!CheckInputTensorDim(context, nodeName, isScales, quantMode, isLayered, config),
        OP_LOGE(nodeName, "Input param shape is invalid."), return false);
    if (isActiveMask) {
        const gert::StorageShape *xStorageShape = context->GetInputShape(config.xIndex);
        int64_t xDim0 = xStorageShape->GetStorageShape().GetDim(0);
        const gert::StorageShape *expertIdStorageShape = context->GetInputShape(config.expertIdsIndex);
        const int64_t expertIdDim1 = expertIdStorageShape->GetStorageShape().GetDim(1);
        const gert::StorageShape *xActiveMaskStorageShape = context->GetOptionalInputShape(config.xActiveMaskIndex);
        OP_TILING_CHECK(xActiveMaskStorageShape == nullptr, OP_LOGE(nodeName, "xActiveMask shape is null."),
            return false);
        const int64_t xActiveMaskDimNum = xActiveMaskStorageShape->GetStorageShape().GetDimNum();
        OP_TILING_CHECK(((xActiveMaskDimNum != ONE_DIM) && (xActiveMaskDimNum != TWO_DIMS)),
            OP_LOGE(nodeName, "xActiveMask shape dim must be 1 or 2, but current dim num is %ld.",
            xActiveMaskDimNum), return false);
        OP_TILING_CHECK((xActiveMaskStorageShape->GetStorageShape().GetDim(0) != xDim0), OP_LOGE(nodeName,
            "The input of xActiveMask dim0 = %ld is not equal to x dim0 = %ld.",
            xActiveMaskStorageShape->GetStorageShape().GetDim(0), xDim0), return false);
        OP_TILING_CHECK(((xActiveMaskDimNum == TWO_DIMS) &&
            (xActiveMaskStorageShape->GetStorageShape().GetDim(1) != expertIdDim1)), OP_LOGE(nodeName,
            "The input of xActiveMask dim1 = %ld is not equal to expertId dim1 = %ld.",
            xActiveMaskStorageShape->GetStorageShape().GetDim(1), expertIdDim1),
            return false);
    }
    if (hasElasticInfo) {
        const gert::StorageShape *elasticInfoStorageShape = context->GetOptionalInputShape(config.elasticInfoIndex);
        OP_TILING_CHECK(elasticInfoStorageShape == nullptr, OP_LOGE(nodeName, "elasticInfo is null."), return false);
        OP_TILING_CHECK(elasticInfoStorageShape->GetStorageShape().GetDimNum() != ONE_DIM,
            OP_LOGE(nodeName, "elasticInfo dim must be 1, but current dim num is %lu.",
            elasticInfoStorageShape->GetStorageShape().GetDimNum()), return false);
        OP_LOGD(nodeName, "elasticInfo dim0 = %ld", elasticInfoStorageShape->GetStorageShape().GetDim(0));
    }
    OP_TILING_CHECK(!CheckCommonOutputTensorDim(context, nodeName, quantMode), 
        OP_LOGE(nodeName, "Output param shape is invalid."), return false);

    return true;
}

static ge::graphStatus CheckQuantModeAndScales(const gert::TilingContext *context, const char *nodeName,
    bool isScales, const uint32_t quantMode, DispatchV2Config &config)
{
    OP_TILING_CHECK(!isScales && (quantMode == static_cast<uint32_t>(QuantModeA5::STATIC_QUANT)),
        OP_LOGE(nodeName, "The scales should not be nullptr when quantMode is %u.", 
        quantMode), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(isScales && (quantMode == static_cast<uint32_t>(QuantModeA5::MX_QUANT)),
        OP_LOGE(nodeName, "The scales should be nullptr when quantMode is %u.", 
        quantMode), return ge::GRAPH_FAILED);
    auto xDesc = context->GetInputDesc(config.xIndex);
    OP_TILING_CHECK(xDesc == nullptr, OP_LOGE(nodeName, "xDesc is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(!isScales && (quantMode == static_cast<uint32_t>(QuantModeA5::NON_QUANT)) 
        && ((xDesc->GetDataType() == ge::DT_HIFLOAT8) || (xDesc->GetDataType() == ge::DT_FLOAT8_E5M2) 
        || (xDesc->GetDataType() == ge::DT_FLOAT8_E4M3FN)),
        OP_LOGE(nodeName, "The scales should not be nullptr when quantMode is %u and X datatype is %s.",
        quantMode, Ops::Base::ToString(xDesc->GetDataType()).c_str()), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(isScales && (quantMode == static_cast<uint32_t>(QuantModeA5::NON_QUANT)) 
        && ((xDesc->GetDataType() == ge::DT_BF16) || (xDesc->GetDataType() == ge::DT_FLOAT16)),
        OP_LOGE(nodeName, "The scales should be nullptr when quantMode is %u and X datatype is %s.",
        quantMode, Ops::Base::ToString(xDesc->GetDataType()).c_str()), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static bool CheckTensorDataTypeNonQuant(const gert::TilingContext *context,
    const char *nodeName, const bool isScales, DispatchV2Config &config)
{
    auto xDesc = context->GetInputDesc(config.xIndex);
    auto expandXDesc = context->GetOutputDesc(OUTPUT_EXPAND_X_INDEX);
    OP_TILING_CHECK((NON_QUANT_DTYPE.find(static_cast<ge::DataType>(xDesc->GetDataType())) == NON_QUANT_DTYPE.end()),
        OP_LOGE(nodeName, 
        "x datatype is invalid, datatype should be one of bf16/fp16/e5m2/e4m3fn/hif8, but is %s.",
        Ops::Base::ToString(xDesc->GetDataType()).c_str()), return false);
    // ExpandX: the same as X
    OP_TILING_CHECK(expandXDesc->GetDataType() != xDesc->GetDataType(),
        OP_LOGE(nodeName, "expandX dataType is invalid, dataType should be equal to x dataType %s, but is %s.",
        Ops::Base::ToString(xDesc->GetDataType()).c_str(), Ops::Base::ToString(expandXDesc->GetDataType()).c_str()),
        return false);
    // Scales: bf16/fp16: nullptr; hif8: fp32; e5m2/e4m3fn: float/e8m0
    // Dynamic scales: the same as scales, and no validations for bf16/fp16
    // If X is bf16/fp16, the scales must be nullptr, which is validated in CheckQuantModeAndScales
    // Hence the datatype of X must be e5m2/e4m3fn/hif8 when isScales is true
    if (isScales) {
        auto scalesDesc = context->GetOptionalInputDesc(config.scalesIndex);
        auto dynamicScalesDesc = context->GetOutputDesc(OUTPUT_DYNAMIC_SCALES_INDEX);
        OP_TILING_CHECK(dynamicScalesDesc == nullptr, OP_LOGE(nodeName, "dynamicScalesDesc is null."),
            return false);
        OP_TILING_CHECK((xDesc->GetDataType() == ge::DT_HIFLOAT8) && 
            (scalesDesc->GetDataType() != ge::DT_FLOAT),
            OP_LOGE(nodeName, "scales datatype is invalid, datatype should be float, but is %s.",
            Ops::Base::ToString(scalesDesc->GetDataType()).c_str()), return false);
        OP_TILING_CHECK((scalesDesc->GetDataType() != ge::DT_FLOAT) && 
            (scalesDesc->GetDataType() != ge::DT_FLOAT8_E8M0),
            OP_LOGE(nodeName, "scales datatype is invalid, datatype should be float or e8m0, but is %s.",
            Ops::Base::ToString(scalesDesc->GetDataType()).c_str()), return false);
        OP_TILING_CHECK(dynamicScalesDesc->GetDataType() != scalesDesc->GetDataType(),
            OP_LOGE(nodeName, 
            "dynamicScales datatype is invalid, datatype should be equal to scales dataType %s, but is %s.",
            Ops::Base::ToString(scalesDesc->GetDataType()).c_str(), 
            Ops::Base::ToString(dynamicScalesDesc->GetDataType()).c_str()), return false);
    }
    return true;
}

static bool CheckTensorDataTypeStaticOrDynamic(
    const gert::TilingContext *context, const char *nodeName, bool isScales, DispatchV2Config &config)
{
    auto xDesc = context->GetInputDesc(config.xIndex);
    auto dynamicScalesDesc = context->GetOutputDesc(OUTPUT_DYNAMIC_SCALES_INDEX);
    OP_TILING_CHECK(dynamicScalesDesc == nullptr, OP_LOGE(nodeName, "dynamicScalesDesc is null."),
        return false);
    OP_TILING_CHECK((xDesc->GetDataType() != ge::DT_BF16) && (xDesc->GetDataType() != ge::DT_FLOAT16),
        OP_LOGE(nodeName, "x datatype is invalid, datatype should be bf16 or float16, but is %s.",
        Ops::Base::ToString(xDesc->GetDataType()).c_str()), return false);
    // Scales: fp32, optional for dynamic/pertoken/pertile, required for static/hif8
    // isScales has been checked in CheckQuantModeAndScales
    if (isScales) {
        auto scalesDesc = context->GetOptionalInputDesc(config.scalesIndex);
        OP_TILING_CHECK(scalesDesc == nullptr, OP_LOGE(nodeName, "scalesDesc is null."), return false);
        OP_TILING_CHECK((scalesDesc->GetDataType() != ge::DT_FLOAT),
            OP_LOGE(nodeName, "scales datatype is invalid, datatype should be float, but is %s.",
            Ops::Base::ToString(scalesDesc->GetDataType()).c_str()), return false);
    }
    OP_TILING_CHECK(dynamicScalesDesc->GetDataType() != ge::DT_FLOAT,
        OP_LOGE(nodeName, "dynamicScales datatype is invalid, datatype should be float, but is %s.",
        Ops::Base::ToString(dynamicScalesDesc->GetDataType()).c_str()), return false);
    return true;
}

static bool CheckTensorDataTypeMxfp8(
    const gert::TilingContext *context, const char *nodeName, DispatchV2Config &config)
{
    auto xDesc = context->GetInputDesc(config.xIndex);
    auto dynamicScalesDesc = context->GetOutputDesc(OUTPUT_DYNAMIC_SCALES_INDEX);
    OP_TILING_CHECK((xDesc->GetDataType() != ge::DT_BF16) && (xDesc->GetDataType() != ge::DT_FLOAT16),
        OP_LOGE(nodeName, "x datatype is invalid, datatype should be bf16 or float16, but is %s.",
        Ops::Base::ToString(xDesc->GetDataType()).c_str()), return false);
    // No Scales input
    OP_TILING_CHECK(dynamicScalesDesc->GetDataType() != ge::DT_FLOAT8_E8M0,
        OP_LOGE(nodeName, "dynamicScales datatype is invalid, datatype should be e8m0, but is %s.",
        Ops::Base::ToString(dynamicScalesDesc->GetDataType()).c_str()), return false);
    return true;
}

static bool CheckDistinctTensorDataType(const gert::TilingContext *context, const char *nodeName,
    const bool isScales, const uint32_t quantMode, DispatchV2Config &config)
{  
    if (quantMode == static_cast<uint32_t>(QuantModeA5::NON_QUANT)) {
        OP_TILING_CHECK(!CheckTensorDataTypeNonQuant(context, nodeName, isScales, config), 
            OP_LOGE(nodeName, "CheckTensorDataType for nonquant mode failed."), return false);
    } else if (quantMode == static_cast<uint32_t>(QuantModeA5::MX_QUANT)) {
        OP_TILING_CHECK(!CheckTensorDataTypeMxfp8(context, nodeName, config),
            OP_LOGE(nodeName, "CheckTensorDataType for mx quant mode failed."), return false);
    } else {
        // static/perToken/perGroup
        OP_TILING_CHECK(!CheckTensorDataTypeStaticOrDynamic(context, nodeName, isScales, config), 
            OP_LOGE(nodeName, "CheckTensorDataType for quantMode %u failed.", quantMode), return false);
    }
    return true;
}

static bool CheckCommomOutputTensorDataType(const gert::TilingContext *context, const char *nodeName)
{
    auto assistInfoDesc = context->GetOutputDesc(OUTPUT_ASSIST_INFO_INDEX);
    OP_TILING_CHECK(assistInfoDesc == nullptr, OP_LOGE(nodeName, "assistInfoDesc is null."), return false);
    OP_TILING_CHECK(assistInfoDesc->GetDataType() != ge::DT_INT32,
        OP_LOGE(nodeName, "assistInfoForCombine dataType is invalid, dataType should be int32, but is %s.",
        Ops::Base::ToString(assistInfoDesc->GetDataType()).c_str()), return false);

    auto expertTokenNumsDesc = context->GetOutputDesc(OUTPUT_EXPERT_TOKEN_NUMS_INDEX);
    OP_TILING_CHECK(expertTokenNumsDesc == nullptr, OP_LOGE(nodeName, "expertTokenNumsDesc is null."),
        return false);
    OP_TILING_CHECK(expertTokenNumsDesc->GetDataType() != ge::DT_INT64,
        OP_LOGE(nodeName, "expertTokenNums dataType is invalid, dataType should be int64, but is %s.",
        Ops::Base::ToString(expertTokenNumsDesc->GetDataType()).c_str()), return false);

    auto epRecvCountsDesc = context->GetOutputDesc(OUTPUT_EP_RECV_COUNTS_INDEX);
    OP_TILING_CHECK(epRecvCountsDesc == nullptr, OP_LOGE(nodeName, "epRecvCountsDesc is null."), return false);
    OP_TILING_CHECK(epRecvCountsDesc->GetDataType() != ge::DT_INT32,
        OP_LOGE(nodeName, "epRecvCounts dataType is invalid, dataType should be int32, but is %s.",
        Ops::Base::ToString(epRecvCountsDesc->GetDataType()).c_str()), return false);

    auto tpRecvCountsDesc = context->GetOutputDesc(OUTPUT_TP_RECV_COUNTS_INDEX);
    OP_TILING_CHECK(tpRecvCountsDesc == nullptr, OP_LOGE(nodeName, "tpRecvCountsDesc is null."), return false);
    OP_TILING_CHECK(tpRecvCountsDesc->GetDataType() != ge::DT_INT32,
        OP_LOGE(nodeName, "tpRecvCounts dataType is invalid, dataType should be int32, but is %s.",
        Ops::Base::ToString(tpRecvCountsDesc->GetDataType()).c_str()), return false);

    return true;
}

static bool CheckQuantModeAndExpandXType(const gert::TilingContext *context, const char *nodeName,
    DispatchV2Config &config)
{
    auto attrs = context->GetAttrs();
    auto quantModePtr = attrs-> GetAttrPointer<int64_t>(config.attrQuantModeIndex);
    auto expandXDesc = context-> GetOutputDesc(OUTPUT_EXPAND_X_INDEX);
    QuantModeA5 quantMode = static_cast<QuantModeA5>(*quantModePtr);
    auto modeToFind = QUANT_MODE_MAP.find({quantMode, static_cast<ge::DataType>(expandXDesc->GetDataType())});
    OP_TILING_CHECK(modeToFind == QUANT_MODE_MAP.end(),
        OP_LOGE(nodeName, "Failed to find real mode for quantMode=%u and expandx datatype=%s",
        static_cast<uint32_t>(quantMode), Ops::Base::ToString(expandXDesc->GetDataType()).c_str()),
        return false);
    OP_LOGD(nodeName, "quantMode=%u, expandx datatype=%s, get realMode=%u\n",
        static_cast<uint32_t>(quantMode), Ops::Base::ToString(expandXDesc->GetDataType()).c_str(),
        static_cast<uint32_t>(modeToFind->second));
    return true;
}

static bool CheckCommomOtherInputTensorDataType(const gert::TilingContext *context, const char *nodeName,
    const bool isActiveMask, const bool hasElasticInfo, DispatchV2Config &config)
{
    auto expertIdDesc = context->GetInputDesc(config.expertIdsIndex);
    OP_TILING_CHECK(expertIdDesc->GetDataType() != ge::DT_INT32,
        OP_LOGE(nodeName, "expertId dataType is invalid, dataType should be int32, but is %s.",
        Ops::Base::ToString(expertIdDesc->GetDataType()).c_str()), return false);

    if (isActiveMask) {
        auto xActiveMaskDesc = context->GetOptionalInputDesc(config.xActiveMaskIndex);
        OP_TILING_CHECK(xActiveMaskDesc->GetDataType() != ge::DT_BOOL, OP_LOGE(nodeName,
            "xActiveMask dataType is invalid, dataType should be bool, but is %s.",
            Ops::Base::ToString(xActiveMaskDesc->GetDataType()).c_str()), return false);
    }
    if (hasElasticInfo) {
        auto elasticInfoDesc = context->GetOptionalInputDesc(config.elasticInfoIndex);
        OP_TILING_CHECK(elasticInfoDesc->GetDataType() != ge::DT_INT32, OP_LOGE(nodeName,
            "elasticInfoDesc dataType is invalid, dataType should be int32, but is %s.",
            Ops::Base::ToString(elasticInfoDesc->GetDataType()).c_str()), return false);
    }

    return true;
}

static bool CheckTensorDataType(const gert::TilingContext *context, const char *nodeName,
    const bool isScales, const uint32_t quantMode, const bool isActiveMask, const bool hasElasticInfo,
    DispatchV2Config &config)
{
    if (mc2tiling::GetNpuArch(context) == NpuArch::DAV_3510) {
        OP_TILING_CHECK(!CheckQuantModeAndExpandXType(context, nodeName, config), 
            OP_LOGE(nodeName, "CheckQuantModeAndExpandXType failed."), return false);
        OP_TILING_CHECK(!CheckDistinctTensorDataType(context, nodeName, isScales, quantMode, config), 
            OP_LOGE(nodeName, "CheckDistinctTensorDataType failed."), return false);
    } else {
        auto xDesc = context->GetInputDesc(config.xIndex);
        OP_TILING_CHECK((xDesc->GetDataType() != ge::DT_BF16) && (xDesc->GetDataType() != ge::DT_FLOAT16),
            OP_LOGE(nodeName, "x dataType is invalid, dataType should be bf16 or float16, but is %s.",
            Ops::Base::ToString(xDesc->GetDataType()).c_str()), return false);

        if (quantMode == static_cast<uint32_t>(QuantModeA5::PERTOKEN_DYNAMIC_QUANT)) {
            auto dynamicScalesDesc = context->GetOutputDesc(OUTPUT_DYNAMIC_SCALES_INDEX);
            OP_TILING_CHECK(dynamicScalesDesc == nullptr, OP_LOGE(nodeName, "dynamicScalesDesc is null."),
                return false);
            OP_TILING_CHECK(dynamicScalesDesc->GetDataType() != ge::DT_FLOAT,
                OP_LOGE(nodeName, "dynamicScales dataType is invalid, dataType should be float, but is %s.",
                Ops::Base::ToString(dynamicScalesDesc->GetDataType()).c_str()), return false);
        }
        if (isScales) {
            auto scalesDesc = context->GetOptionalInputDesc(config.scalesIndex);
            OP_TILING_CHECK(scalesDesc->GetDataType() != ge::DT_FLOAT,
                OP_LOGE(nodeName, "scales dataType is invalid, dataType should be float, but is %s.",
                Ops::Base::ToString(scalesDesc->GetDataType()).c_str()), return false);
        }
        auto expandXDesc = context->GetOutputDesc(OUTPUT_EXPAND_X_INDEX);
        if (quantMode != static_cast<uint32_t>(QuantModeA5::NON_QUANT)) {
            OP_TILING_CHECK(expandXDesc->GetDataType() != ge::DT_INT8,
                OP_LOGE(nodeName, "expandX dataType is invalid, dataType should be int8, but is %s.",
                Ops::Base::ToString(expandXDesc->GetDataType()).c_str()), return false);
        } else {
            OP_TILING_CHECK(expandXDesc->GetDataType() != xDesc->GetDataType(),
                OP_LOGE(nodeName, "expandX dataType is invalid, dataType should be equal to x dataType %s, but is %s.",
                Ops::Base::ToString(xDesc->GetDataType()).c_str(), Ops::Base::ToString(expandXDesc->GetDataType()).c_str()),
                return false);
        }
    }

    OP_TILING_CHECK(!CheckCommomOtherInputTensorDataType(context, nodeName, isActiveMask, hasElasticInfo, config),
        OP_LOGE(nodeName, "CheckCommomOtherInputTensorDataType failed."), return false);
    OP_TILING_CHECK(!CheckCommomOutputTensorDataType(context, nodeName),
        OP_LOGE(nodeName, "CheckCommomOutputTensorDataType failed."), return false);

    return true;
}

//校验输入输出数据格式
static bool CheckTensorFormat(const gert::TilingContext *context, const char *nodeName,
    const bool isScales, const uint32_t quantMode, const bool isActiveMask, const bool hasElasticInfo,
    DispatchV2Config &config)
{
    auto xDesc = context->GetInputDesc(config.xIndex); // nullptr前面已check过
    OP_TILING_CHECK(static_cast<ge::Format>(ge::GetPrimaryFormat(xDesc->GetStorageFormat())) == ge::FORMAT_FRACTAL_NZ,
        OP_LOGE(nodeName, "x format is invalid."), return false);
    auto expertIdDesc = context->GetInputDesc(config.expertIdsIndex);
    OP_TILING_CHECK(static_cast<ge::Format>(ge::GetPrimaryFormat(expertIdDesc->GetStorageFormat())) ==
        ge::FORMAT_FRACTAL_NZ, OP_LOGE(nodeName, "expertId format is invalid."), return false);
    if (isScales) {
        auto scalesDesc = context->GetOptionalInputDesc(config.scalesIndex);
        OP_TILING_CHECK(static_cast<ge::Format>(ge::GetPrimaryFormat(scalesDesc->GetStorageFormat())) ==
            ge::FORMAT_FRACTAL_NZ, OP_LOGE(nodeName, "scales format is invalid."), return false);
    }
    if (isActiveMask) {
        auto xActiveMaskDesc = context->GetOptionalInputDesc(config.xActiveMaskIndex);
        OP_TILING_CHECK(static_cast<ge::Format>(ge::GetPrimaryFormat(xActiveMaskDesc->GetStorageFormat())) ==
            ge::FORMAT_FRACTAL_NZ, OP_LOGE(nodeName, "xActiveMask format is invalid."), return false);
    }
    if (hasElasticInfo) {
        auto elasticInfoDesc = context->GetOptionalInputDesc(config.elasticInfoIndex);
        OP_TILING_CHECK(static_cast<ge::Format>(ge::GetPrimaryFormat(elasticInfoDesc->GetStorageFormat())) ==
            ge::FORMAT_FRACTAL_NZ, OP_LOGE(nodeName, "elasticInfo format is invalid."), return false);
    }
    auto expandXDesc = context->GetOutputDesc(OUTPUT_EXPAND_X_INDEX);
    OP_TILING_CHECK(static_cast<ge::Format>(ge::GetPrimaryFormat(expandXDesc->GetStorageFormat())) ==
        ge::FORMAT_FRACTAL_NZ, OP_LOGE(nodeName, "expandX format is invalid."), return false);
    if (quantMode >= static_cast<uint32_t>(QuantModeA5::PERTOKEN_DYNAMIC_QUANT)) {
        auto dynamicScalesDesc = context->GetOutputDesc(OUTPUT_DYNAMIC_SCALES_INDEX);
        OP_TILING_CHECK(static_cast<ge::Format>(ge::GetPrimaryFormat(dynamicScalesDesc->GetStorageFormat())) ==
            ge::FORMAT_FRACTAL_NZ, OP_LOGE(nodeName, "dynamicScales format is invalid."), return false);
    }
    auto assistInfoDesc = context->GetOutputDesc(OUTPUT_ASSIST_INFO_INDEX);
    OP_TILING_CHECK(static_cast<ge::Format>(ge::GetPrimaryFormat(assistInfoDesc->GetStorageFormat())) ==
        ge::FORMAT_FRACTAL_NZ, OP_LOGE(nodeName, "assistInfoForCombine format is invalid."), return false);
    auto expertTokenNumsDesc = context->GetOutputDesc(OUTPUT_EXPERT_TOKEN_NUMS_INDEX);
    OP_TILING_CHECK(static_cast<ge::Format>(ge::GetPrimaryFormat(expertTokenNumsDesc->GetStorageFormat())) ==
        ge::FORMAT_FRACTAL_NZ, OP_LOGE(nodeName, "expertTokenNums format is invalid."), return false);
    auto epRecvCountsDesc = context->GetOutputDesc(OUTPUT_EP_RECV_COUNTS_INDEX);
    OP_TILING_CHECK(static_cast<ge::Format>(ge::GetPrimaryFormat(epRecvCountsDesc->GetStorageFormat())) ==
        ge::FORMAT_FRACTAL_NZ, OP_LOGE(nodeName, "epRecvCounts format is invalid."), return false);
    auto tpRecvCountsDesc = context->GetOutputDesc(OUTPUT_TP_RECV_COUNTS_INDEX);
    OP_TILING_CHECK(static_cast<ge::Format>(ge::GetPrimaryFormat(tpRecvCountsDesc->GetStorageFormat())) ==
        ge::FORMAT_FRACTAL_NZ, OP_LOGE(nodeName, "tpRecvCounts format is invalid."), return false);
    return true;
}

static ge::graphStatus CheckAttrPtrNullptr(const gert::TilingContext *context, const char *nodeName,
    std::string &groupEp, DispatchV2Config &config)
{
    auto attrs = context->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE(nodeName, "attrs is nullptr."), return ge::GRAPH_FAILED);

    if (!config.isMc2Context) {
        auto groupEpPtr = attrs->GetAttrPointer<char>(static_cast<int>(config.attrGroupEpIndex));
        OP_TILING_CHECK((groupEpPtr == nullptr) || (strnlen(groupEpPtr, MAX_GROUP_NAME_LENGTH) == 0) ||
            (strnlen(groupEpPtr, MAX_GROUP_NAME_LENGTH) == MAX_GROUP_NAME_LENGTH),
            OP_LOGE(nodeName, "groupEpPtr is null."), return ge::GRAPH_FAILED);
        groupEp = std::string(groupEpPtr);
    }
    auto epWorldSizePtr = attrs->GetAttrPointer<int64_t>(config.attrEpWorldSizeIndex);
    auto tpWorldSizePtr = attrs->GetAttrPointer<int64_t>(config.attrTpWorldSizeIndex);
    auto epRankIdPtr = attrs->GetAttrPointer<int64_t>(config.attrEpRankIdIndex);
    auto tpRankIdPtr = attrs->GetAttrPointer<int64_t>(config.attrTpRankIdIndex);
    auto expertShardPtr = attrs->GetAttrPointer<int64_t>(config.attrExpertSharedTypeIndex);
    auto sharedExpertNumPtr = attrs->GetAttrPointer<int64_t>(static_cast<int>(config.attrSharedExpertNumIndex));
    auto sharedExpertRankNumPtr = attrs->GetAttrPointer<int64_t>(config.attrSharedExpertRankNumIndex);
    auto moeExpertNumPtr = attrs->GetAttrPointer<int64_t>(config.attrMoeExpertNumIndex);
    auto quantModePtr = attrs->GetAttrPointer<int64_t>(config.attrQuantModeIndex);
    auto expertTokenNumsTypePtr = attrs->GetAttrPointer<int64_t>(static_cast<int>(config.attrExpertTokenNumsTypeIndex));
    auto commAlgPtr = attrs->GetAttrPointer<char>(static_cast<int>(config.attrCommAlgIndex));
    auto zeroExpertNumPtr = attrs->GetAttrPointer<int64_t>(static_cast<int>(config.attrZeroExpertNumIndex));
    auto copyExpertNumPtr = attrs->GetAttrPointer<int64_t>(static_cast<int>(config.attrCopyExpertNumIndex));
    auto constExpertNumPtr = attrs->GetAttrPointer<int64_t>(static_cast<int>(config.attrConstExpertNumIndex));

    // 判空
    OP_TILING_CHECK(commAlgPtr == nullptr, OP_LOGE(nodeName, "commAlgPtr is nullptr."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(epWorldSizePtr == nullptr, OP_LOGE(nodeName, "epWorldSizePtr is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(tpWorldSizePtr == nullptr, OP_LOGE(nodeName, "tpWorldSizePtr is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(epRankIdPtr == nullptr, OP_LOGE(nodeName, "epRankIdPtr is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(tpRankIdPtr == nullptr, OP_LOGE(nodeName, "tpRankIdPtr is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(expertShardPtr == nullptr, OP_LOGE(nodeName, "expertShardPtr is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(sharedExpertNumPtr == nullptr, OP_LOGE(nodeName, "sharedExpertNumPtr is null."),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(sharedExpertRankNumPtr == nullptr, OP_LOGE(nodeName, "sharedExpertRankNumPtr is null."),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(moeExpertNumPtr == nullptr, OP_LOGE(nodeName, "moeExpertNumPtr is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(quantModePtr == nullptr, OP_LOGE(nodeName, "quantModePtr is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(expertTokenNumsTypePtr == nullptr, OP_LOGE(nodeName, "expertTokenNumsTypePtr is null."),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(zeroExpertNumPtr == nullptr, OP_LOGE(nodeName, "zeroExpertNumPtr is null."),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(copyExpertNumPtr == nullptr, OP_LOGE(nodeName, "copyExpertNumPtr is null."),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(constExpertNumPtr == nullptr, OP_LOGE(nodeName, "constExpertNumPtr is null."),
        return ge::GRAPH_FAILED);
    
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckGroupAttrParams(const gert::TilingContext *context, const char *nodeName,
    std::string &groupTp, bool &isLayered, DispatchV2Config &config)
{
    auto attrs = context->GetAttrs();
    auto epWorldSizePtr = attrs->GetAttrPointer<int64_t>(config.attrEpWorldSizeIndex);
    auto tpWorldSizePtr = attrs->GetAttrPointer<int64_t>(config.attrTpWorldSizeIndex);
    auto epRankIdPtr = attrs->GetAttrPointer<int64_t>(config.attrEpRankIdIndex);
    auto tpRankIdPtr = attrs->GetAttrPointer<int64_t>(config.attrTpRankIdIndex);
    auto commAlgPtr = attrs->GetAttrPointer<char>(static_cast<int64_t>(config.attrCommAlgIndex));
    int64_t epWorldSize = *epWorldSizePtr;
    isLayered = strcmp(commAlgPtr, "hierarchy") == 0; // isLayered赋值
    int64_t maxEpworldsize = isLayered ? MAX_EP_WORLD_SIZE_LAYERED : MAX_EP_WORLD_SIZE;
    int64_t maxTpworldsize = isLayered ? MAX_TP_WORLD_SIZE_LAYERED : MAX_TP_WORLD_SIZE;
    OP_TILING_CHECK((epWorldSize < MIN_EP_WORLD_SIZE) || (epWorldSize > maxEpworldsize),
        OP_LOGE(nodeName, "epWorldSize is invalid, only support [%ld, %ld], but got epWorldSize=%ld.",
        MIN_EP_WORLD_SIZE, maxEpworldsize, epWorldSize), return ge::GRAPH_FAILED);
    OP_TILING_CHECK((isLayered && (epWorldSize % RANK_NUM_PER_NODE != 0)),     // 校验epWorldSize是否是16整数倍
        OP_LOGE(nodeName, "epWorldSize should be %u Aligned, but got %ld.", RANK_NUM_PER_NODE, epWorldSize), return ge::GRAPH_FAILED);
    OP_TILING_CHECK((*tpWorldSizePtr < 0) || (*tpWorldSizePtr > maxTpworldsize),
        OP_LOGE(nodeName, "tpWorldSize is invalid, only support [0, %ld], but got tpWorldSize=%ld.",
        maxTpworldsize, *tpWorldSizePtr), return ge::GRAPH_FAILED);
    OP_TILING_CHECK((*epRankIdPtr < 0) || (*epRankIdPtr >= epWorldSize),
        OP_LOGE(nodeName, "epRankId is invalid, only support [0, %ld), but got epRankId=%ld.",
        epWorldSize, *epRankIdPtr), return ge::GRAPH_FAILED);
    if (*tpWorldSizePtr > 1) {
        auto groupTpPtr = attrs->GetAttrPointer<char>(static_cast<int64_t>(config.attrGroupTpIndex));
        OP_TILING_CHECK((*tpRankIdPtr < 0) || (*tpRankIdPtr >= *tpWorldSizePtr),
            OP_LOGE(nodeName, "tpRankId is invalid, only support [0, %ld), but got tpRankId=%ld.",
            *tpWorldSizePtr, *tpRankIdPtr), return ge::GRAPH_FAILED);
        OP_TILING_CHECK((groupTpPtr == nullptr) || (strnlen(groupTpPtr, MAX_GROUP_NAME_LENGTH) == 0) ||
            (strnlen(groupTpPtr, MAX_GROUP_NAME_LENGTH) == MAX_GROUP_NAME_LENGTH),
            OP_LOGE(nodeName, "groupTpPtr is null."), return ge::GRAPH_FAILED);
        groupTp = std::string(groupTpPtr);
    } else {
        OP_TILING_CHECK(*tpRankIdPtr != 0,
            OP_LOGE(nodeName, "tpRankId is invalid, NoTp mode only support 0, but got tpRankId=%ld.", *tpRankIdPtr),
            return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckOtherAttrParams(const gert::TilingContext *context, const char *nodeName,
    bool &isSetFullMeshV2, DispatchV2Config &config)
{
    auto attrs = context->GetAttrs();
    auto expertShardPtr = attrs->GetAttrPointer<int64_t>(config.attrExpertSharedTypeIndex);
    auto quantModePtr = attrs->GetAttrPointer<int64_t>(config.attrQuantModeIndex);
    auto expertTokenNumsTypePtr = attrs->GetAttrPointer<int64_t>(static_cast<int>(config.attrExpertTokenNumsTypeIndex));
    auto commAlgPtr = attrs->GetAttrPointer<char>(static_cast<int>(config.attrCommAlgIndex));
    if (config.isMc2Context) {
        OP_TILING_CHECK((strcmp(commAlgPtr, "hierarchy") == 0),
            OP_LOGE(nodeName, "commAlgPtr %s doesn't support comm with context.", commAlgPtr),
            return ge::GRAPH_FAILED);
    }
    OP_TILING_CHECK(*expertShardPtr != 0,
        OP_LOGE(nodeName, "expertShardType is invalid, only support 0, but got expertShardType=%ld.",
        *expertShardPtr), return ge::GRAPH_FAILED);
    if (mc2tiling::GetNpuArch(context) == NpuArch::DAV_3510) {
        OP_TILING_CHECK((*quantModePtr < static_cast<int64_t>(QuantModeA5::NON_QUANT)) ||
        (*quantModePtr > static_cast<int64_t>(QuantModeA5::MX_QUANT)),
        OP_LOGE(nodeName, "quantMode is invalid, only support [0, %ld], but got quantMode=%ld.",
        static_cast<int64_t>(QuantModeA5::MX_QUANT), *quantModePtr), return ge::GRAPH_FAILED);
    } else {
        OP_TILING_CHECK((*quantModePtr < static_cast<int64_t>(QuantModeA5::NON_QUANT)) ||
        (*quantModePtr > static_cast<int64_t>(QuantModeA5::PERTOKEN_DYNAMIC_QUANT)),
        OP_LOGE(nodeName, "quantMode is invalid, only support [0, %ld], but got quantMode=%ld.",
        static_cast<int64_t>(QuantModeA5::PERTOKEN_DYNAMIC_QUANT), *quantModePtr), return ge::GRAPH_FAILED);
    }
    OP_TILING_CHECK((*expertTokenNumsTypePtr != 0) && (*expertTokenNumsTypePtr != 1),
        OP_LOGE(nodeName, "expertTokenNumsType only support 0 or 1, but got expertTokenNumsType=%ld.",
        *expertTokenNumsTypePtr), return ge::GRAPH_FAILED);
    // A5 已作校验，这里只校验 A3
    if (mc2tiling::GetNpuArch(context) != NpuArch::DAV_3510) {
        OP_TILING_CHECK(((strlen(commAlgPtr) != 0) && (strcmp(commAlgPtr, "fullmesh_v1") != 0) &&
            (strcmp(commAlgPtr, "fullmesh_v2")  != 0) && (strcmp(commAlgPtr, "hierarchy") != 0)),
            OP_LOGE(nodeName, "Attr commAlg is invalid, current only support hierarchy, fullmesh_v1 and fullmesh_v2, but got commAlg = %s.", 
            commAlgPtr), return ge::GRAPH_FAILED);
    }
    isSetFullMeshV2 = ((strcmp(commAlgPtr, "fullmesh_v2") == 0) ? true : false);
    OP_LOGD(nodeName, "MoeDistributeDispatchV2 isSetFullMeshV2 = %d\n", isSetFullMeshV2);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckCommAttrParams(const gert::TilingContext *context, const char *nodeName,
    std::string &groupTp, bool &isSetFullMeshV2, bool &isLayered, DispatchV2Config &config)
{
    // 校验 epWorldSize tpWorldSize epRankId tpRankId
    OP_TILING_CHECK(CheckGroupAttrParams(context, nodeName, groupTp, isLayered, config) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "CheckGroupAttrParams is failed."), return ge::GRAPH_FAILED);
    // 校验 expertSharedType quantMode expertTokenNumsType commAlg
    OP_TILING_CHECK(CheckOtherAttrParams(context, nodeName, isSetFullMeshV2, config) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "CheckGroupAttrParams is failed."), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckExpertAttrParams(const gert::TilingContext *context, const char *nodeName,
    bool isLayered, DispatchV2Config &config)
{
    auto attrs = context->GetAttrs();
    auto epWorldSizePtr = attrs->GetAttrPointer<int64_t>(config.attrEpWorldSizeIndex);
    auto sharedExpertNumPtr = attrs->GetAttrPointer<int64_t>(static_cast<int>(config.attrSharedExpertNumIndex));
    auto sharedExpertRankNumPtr = attrs->GetAttrPointer<int64_t>(config.attrSharedExpertRankNumIndex);
    auto moeExpertNumPtr = attrs->GetAttrPointer<int64_t>(config.attrMoeExpertNumIndex);
    auto zeroExpertNumPtr = attrs->GetAttrPointer<int64_t>(static_cast<int>(config.attrZeroExpertNumIndex));
    auto copyExpertNumPtr = attrs->GetAttrPointer<int64_t>(static_cast<int>(config.attrCopyExpertNumIndex));
    auto constExpertNumPtr = attrs->GetAttrPointer<int64_t>(static_cast<int>(config.attrConstExpertNumIndex));

    int64_t moeExpertNum = *moeExpertNumPtr;
    int64_t epWorldSize = *epWorldSizePtr;
    int64_t sharedExpertRankNum = *sharedExpertRankNumPtr;
    int64_t zeroExpertNum = *zeroExpertNumPtr;
    int64_t copyExpertNum = *copyExpertNumPtr;
    int64_t constExpertNum = *constExpertNumPtr;
    int64_t zeroComputeExpertNum = zeroExpertNum + copyExpertNum + constExpertNum;
    OP_TILING_CHECK((zeroExpertNum < 0), OP_LOGE(nodeName,
        "zeroExpertNum less than 0, zeroExpertNum is %ld.", zeroExpertNum), return ge::GRAPH_FAILED);
    OP_TILING_CHECK((copyExpertNum < 0), OP_LOGE(nodeName,
        "copyExpertNum less than 0, copyExpertNum is %ld.", copyExpertNum), return ge::GRAPH_FAILED);
    OP_TILING_CHECK((constExpertNum < 0), OP_LOGE(nodeName,
        "constExpertNum less than 0, constExpertNum is %ld.", constExpertNum), return ge::GRAPH_FAILED);
    OP_LOGD(nodeName, "zeroExpertNum=%ld,copyExpertNum= %ld, constExpertNum=%ld", zeroExpertNum, copyExpertNum,
        constExpertNum);
    OP_TILING_CHECK(zeroComputeExpertNum + moeExpertNum > INT32_MAX, OP_LOGE(nodeName,
        "zeroExpertNum[%ld] + copyExpertNum[%ld] + constExpertNum[%ld] + moeExpertNum[%ld] exceed INT32_MAX.",
         zeroExpertNum, copyExpertNum, constExpertNum, moeExpertNum), return ge::GRAPH_FAILED);
    if (isLayered) {
        OP_TILING_CHECK((sharedExpertRankNum != 0),
            OP_LOGE(nodeName, "sharedExpertNum is invalid in hierarchy mode, only support 0, but got sharedExpertNum=%ld, sharedExpertRankNum=%ld",
            *sharedExpertNumPtr, sharedExpertRankNum), return ge::GRAPH_FAILED);
    } else {
        OP_TILING_CHECK((*sharedExpertNumPtr < 0) || (*sharedExpertNumPtr > MAX_SHARED_EXPERT_NUM),
            OP_LOGE(nodeName, "sharedExpertNum is invalid, only support [0, %ld], but got sharedExpertNum=%ld.",
            MAX_SHARED_EXPERT_NUM, *sharedExpertNumPtr), return ge::GRAPH_FAILED);
        OP_TILING_CHECK((sharedExpertRankNum < 0) || (sharedExpertRankNum >= epWorldSize),
            OP_LOGE(nodeName, "sharedExpertRankNum is invalid, only support [0, %ld), but got sharedExpertRankNum=%ld.",
            epWorldSize, sharedExpertRankNum), return ge::GRAPH_FAILED);
    }
    int64_t moeExpertMaxNum = isLayered ? MOE_EXPERT_MAX_NUM_LAYERED : MOE_EXPERT_MAX_NUM;
    OP_TILING_CHECK((moeExpertNum <= 0) || (moeExpertNum > moeExpertMaxNum),
        OP_LOGE(nodeName, "moeExpertNum is invalid, only support (0, %ld], but got moeExpertNum=%ld.",
        moeExpertMaxNum, moeExpertNum), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SetAttrParams(const gert::TilingContext *context, const char *nodeName,
    DispatchV2Config &config, MoeDistributeDispatchV2TilingData &tilingData)
{
    auto attrs = context->GetAttrs();
    auto epWorldSizePtr = attrs->GetAttrPointer<int64_t>(config.attrEpWorldSizeIndex);
    auto sharedExpertNumPtr = attrs->GetAttrPointer<int64_t>(static_cast<int>(config.attrSharedExpertNumIndex));
    auto sharedExpertRankNumPtr = attrs->GetAttrPointer<int64_t>(config.attrSharedExpertRankNumIndex);
    auto moeExpertNumPtr = attrs->GetAttrPointer<int64_t>(config.attrMoeExpertNumIndex);
    auto zeroExpertNumPtr = attrs->GetAttrPointer<int64_t>(static_cast<int>(config.attrZeroExpertNumIndex));
    auto copyExpertNumPtr = attrs->GetAttrPointer<int64_t>(static_cast<int>(config.attrCopyExpertNumIndex));
    auto constExpertNumPtr = attrs->GetAttrPointer<int64_t>(static_cast<int>(config.attrConstExpertNumIndex));
    auto tpWorldSizePtr = attrs->GetAttrPointer<int64_t>(config.attrTpWorldSizeIndex);
    auto epRankIdPtr = attrs->GetAttrPointer<int64_t>(config.attrEpRankIdIndex);
    auto tpRankIdPtr = attrs->GetAttrPointer<int64_t>(config.attrTpRankIdIndex);
    auto expertShardPtr = attrs->GetAttrPointer<int64_t>(config.attrExpertSharedTypeIndex);
    auto quantModePtr = attrs->GetAttrPointer<int64_t>(config.attrQuantModeIndex);
    auto expertTokenNumsTypePtr = attrs->GetAttrPointer<int64_t>(static_cast<int>(config.attrExpertTokenNumsTypeIndex));

    int64_t zeroExpertNum = *zeroExpertNumPtr;
    int64_t copyExpertNum = *copyExpertNumPtr;
    int64_t constExpertNum = *constExpertNumPtr;
    int64_t zeroComputeExpertNum = zeroExpertNum + copyExpertNum + constExpertNum;
    tilingData.moeDistributeDispatchV2Info.epWorldSize = static_cast<uint32_t>(*epWorldSizePtr);
    tilingData.moeDistributeDispatchV2Info.tpWorldSize = static_cast<uint32_t>(*tpWorldSizePtr);
    tilingData.moeDistributeDispatchV2Info.epRankId = static_cast<uint32_t>(*epRankIdPtr);
    tilingData.moeDistributeDispatchV2Info.tpRankId = static_cast<uint32_t>(*tpRankIdPtr);
    tilingData.moeDistributeDispatchV2Info.expertShardType = static_cast<uint32_t>(*expertShardPtr);
    tilingData.moeDistributeDispatchV2Info.quantMode = static_cast<uint32_t>(*quantModePtr);
    tilingData.moeDistributeDispatchV2Info.expertTokenNumsType = static_cast<uint32_t>(*expertTokenNumsTypePtr);
    tilingData.moeDistributeDispatchV2Info.sharedExpertNum = static_cast<uint32_t>(*sharedExpertNumPtr);
    tilingData.moeDistributeDispatchV2Info.sharedExpertRankNum = static_cast<uint32_t>(*sharedExpertRankNumPtr);
    tilingData.moeDistributeDispatchV2Info.moeExpertNum = static_cast<uint32_t>(*moeExpertNumPtr);
    tilingData.moeDistributeDispatchV2Info.zeroComputeExpertNum = static_cast<int32_t>(zeroComputeExpertNum);
    if ((tilingData.moeDistributeDispatchV2Info.sharedExpertRankNum == 0U) && (tilingData.moeDistributeDispatchV2Info.sharedExpertNum == 1U)) {
        tilingData.moeDistributeDispatchV2Info.sharedExpertNum = 0U;
    }
    OP_LOGD(nodeName, "MoeDistributeDispatchV2 zeroComputeExpertNum = %d\n",
        tilingData.moeDistributeDispatchV2Info.zeroComputeExpertNum);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetAttrAndSetTilingData(const gert::TilingContext *context, const char *nodeName,
    MoeDistributeDispatchV2TilingData &tilingData, std::string &groupEp, std::string &groupTp, bool &isSetFullMeshV2,
    bool &isLayered, DispatchV2Config &config)
{
    OP_TILING_CHECK(CheckAttrPtrNullptr(context, nodeName, groupEp, config) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "params check nulld failed."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckCommAttrParams(context, nodeName, groupTp, isSetFullMeshV2,
        isLayered, config) != ge::GRAPH_SUCCESS,OP_LOGE(nodeName, "params shape is invalid."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckExpertAttrParams(context, nodeName, isLayered, config) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "params dataType is invalid."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(SetAttrParams(context, nodeName, config, tilingData) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "set attr params failed."), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static bool CheckSharedAttrs(const char *nodeName,
    const MoeDistributeDispatchV2TilingData &tilingData)
{
    uint32_t sharedExpertNum = tilingData.moeDistributeDispatchV2Info.sharedExpertNum;
    uint32_t sharedExpertRankNum = tilingData.moeDistributeDispatchV2Info.sharedExpertRankNum;

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

static bool CheckCommAlgAttrs(const gert::TilingContext *context, const char *nodeName,
    const MoeDistributeDispatchV2TilingData &tilingData, bool isSetFullMeshV2, DispatchV2Config &config, bool isLayered)
{
    uint32_t tpWorldSize = tilingData.moeDistributeDispatchV2Info.tpWorldSize;
    bool hasElasticInfo = tilingData.moeDistributeDispatchV2Info.hasElasticInfo;
    int32_t zeroComputeExpertNum = tilingData.moeDistributeDispatchV2Info.zeroComputeExpertNum;
    bool isExpertMask = tilingData.moeDistributeDispatchV2Info.isExpertMask;
    // 获取bs
    const gert::StorageShape *xStorageShape = context->GetInputShape(config.xIndex);
    const int64_t xDim0 = xStorageShape->GetStorageShape().GetDim(0);
    uint32_t bs = static_cast<uint32_t>(xDim0);

    // 获取topk
    const gert::StorageShape *expertIdStorageShape = context->GetInputShape(config.expertIdsIndex);
    const int64_t expertIdsDim1 = expertIdStorageShape->GetStorageShape().GetDim(1);
    uint32_t k = static_cast<uint32_t>(expertIdsDim1);

    // 检查comm_alg和tpWorldSize是否冲突
    OP_TILING_CHECK(isSetFullMeshV2 && (tpWorldSize == TP_WORLD_SIZE_TWO), OP_LOGE(nodeName, "When comm_alg is fullmesh_v2, tp_world_size cannot be 2."),
        return false);
    // 检查comm_alg和bs是否冲突
    OP_TILING_CHECK(isSetFullMeshV2 && (bs > FULLMESH_BS_UPPER_BOUND), OP_LOGE(nodeName, "When comm_alg is fullmesh_v2, bs should be between [1, %ld], but got %u.", FULLMESH_BS_UPPER_BOUND, bs),
        return false);
    // 检查comm_alg和topK是否冲突
    OP_TILING_CHECK(isSetFullMeshV2 && (k > FULLMESH_K_MAX), OP_LOGE(nodeName, "When comm_alg is fullmesh_v2, topK should be between [1, %ld], but got %u.", FULLMESH_K_MAX, k),
        return false);
    // 校验动态缩容和分层不能同时启用
    OP_TILING_CHECK((isLayered && hasElasticInfo), OP_LOGE(nodeName, "Cannot support elasticInfo when comm_alg is hierarchy"), 
        return false);
    // 校验特殊专家和分层不能同时启用
    OP_TILING_CHECK((isLayered && (zeroComputeExpertNum > 0)), OP_LOGE(nodeName, "Cannot support zeroComputeExpert when comm_alg is hierarchy"), 
        return false);
    // 校验二维Mask和分层不能同时启用
    OP_TILING_CHECK((isLayered && isExpertMask), OP_LOGE(nodeName, "Cannot support 2D xActiveMask when comm_alg is hierarchy"), 
        return false);
    return true;
}

static ge::graphStatus CheckAttrs(const gert::TilingContext *context, const char *nodeName,
    MoeDistributeDispatchV2TilingData &tilingData, uint32_t &localMoeExpertNum, bool isActiveMask, bool isSetFullMeshV2,
    bool isLayered, DispatchV2Config &config)
{
    uint32_t epWorldSize = tilingData.moeDistributeDispatchV2Info.epWorldSize;
    uint32_t tpWorldSize = tilingData.moeDistributeDispatchV2Info.tpWorldSize;
    uint32_t moeExpertNum = tilingData.moeDistributeDispatchV2Info.moeExpertNum;
    uint32_t sharedExpertRankNum = tilingData.moeDistributeDispatchV2Info.sharedExpertRankNum;

    OP_TILING_CHECK(!CheckSharedAttrs(nodeName, tilingData),
        OP_LOGE(nodeName, "Check shared expert related attributes failed."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(!CheckCommAlgAttrs(context, nodeName, tilingData, isSetFullMeshV2, config, isLayered),
        OP_LOGE(nodeName, "Check comm_alg related attributes failed."), return ge::GRAPH_FAILED);
    // 校验moe专家数量能否均分给多机
    localMoeExpertNum = moeExpertNum / (epWorldSize - sharedExpertRankNum);
    OP_TILING_CHECK(moeExpertNum % (epWorldSize - sharedExpertRankNum) != 0,
        OP_LOGE(nodeName, "moeExpertNum should be divisible by (epWorldSize - sharedExpertRankNum), "
        "but moeExpertNum=%u, epWorldSize=%u, sharedExpertRankNum=%u.", moeExpertNum, epWorldSize, sharedExpertRankNum),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK((localMoeExpertNum <= 0) || (localMoeExpertNum * epWorldSize > LOCAL_EXPERT_MAX_SIZE),OP_LOGE(nodeName, "localMoeExpertNum is invalid, "
        "localMoeExpertNum * epWorldSize must be less than or equal to 2048, and localMoeExpertNum must be greater than 0, "
        "but got localMoeExpertNum * epWorldSize = %u, localMoeExpertNum = %u", localMoeExpertNum * epWorldSize, localMoeExpertNum), return ge::GRAPH_FAILED);
    // 校验tp=2时单个moe卡上专家数是否等于1
    OP_TILING_CHECK((tpWorldSize > 1) && (localMoeExpertNum > 1), OP_LOGE(nodeName, "Cannot support multi-moeExpert %u "
        "in a rank when tpWorldSize = %u > 1", localMoeExpertNum, tpWorldSize), return ge::GRAPH_FAILED);
    OP_TILING_CHECK((tpWorldSize > 1) && (tilingData.moeDistributeDispatchV2Info.hasElasticInfo), OP_LOGE(nodeName, "Cannot support elasticInfo"
        " when tpWorldSize = %u > 1", tpWorldSize), return ge::GRAPH_FAILED);
    // 校验输入x的dim 0并设bs
    const gert::StorageShape *xStorageShape = context->GetInputShape(config.xIndex);
    const int64_t xDim0 = xStorageShape->GetStorageShape().GetDim(0);
    int64_t bsUpperBound = isLayered ? BS_UPPER_BOUND_LAYERED : BS_UPPER_BOUND;
    OP_TILING_CHECK((xDim0 > bsUpperBound) || (xDim0 <= 0), OP_LOGE(nodeName, "xDim0(BS) is invalid. Should be between "
        "[1, %ld], but got xDim0=%ld.", bsUpperBound, xDim0), return ge::GRAPH_FAILED);
    tilingData.moeDistributeDispatchV2Info.bs = static_cast<uint32_t>(xDim0);

    // 校验globalBS
    auto attrs = context->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE(nodeName, "attrs is nullptr."), return ge::GRAPH_FAILED);
    auto globalBsPtr = attrs->GetAttrPointer<int64_t>(config.attrGlobalBsIndex);
    OP_TILING_CHECK(globalBsPtr == nullptr, OP_LOGE(nodeName, "globalBsPtr is nullptr."), return ge::GRAPH_FAILED);
    OP_LOGD(nodeName, "MoeDistributeDispatchV2 *globalBsPtr = %ld, bs = %ld, epWorldSize = %u\n",
        *globalBsPtr, xDim0, epWorldSize);
    OP_TILING_CHECK((*globalBsPtr != 0) && ((*globalBsPtr < xDim0 * static_cast<int64_t>(epWorldSize)) ||
        ((*globalBsPtr) % (static_cast<int64_t>(epWorldSize)) != 0)), OP_LOGE(nodeName, "globalBS is invalid, only "
        "support 0 or maxBs(maxBs is the largest bs on all ranks) * epWorldSize, but got globalBS=%ld, "
        "bs=%ld, epWorldSize=%u.", *globalBsPtr, xDim0, epWorldSize), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(((*globalBsPtr > (xDim0 * static_cast<int64_t>(epWorldSize))) && isActiveMask),
        OP_LOGE(nodeName, "Different bs on different rank cannot work when isActiveMask=true, globalBS=%ld, "
        "bs=%ld, epWorldSize=%u.", *globalBsPtr, xDim0, epWorldSize), return ge::GRAPH_FAILED);
    if (*globalBsPtr == 0) {
        tilingData.moeDistributeDispatchV2Info.globalBs = static_cast<uint32_t>(xDim0) * epWorldSize;
    } else {
        tilingData.moeDistributeDispatchV2Info.globalBs = static_cast<uint32_t>(*globalBsPtr);
    }

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckTwoDimScalesShape(const gert::TilingContext *context, const char *nodeName,
    const MoeDistributeDispatchV2TilingData &tilingData, const int64_t scalesDim0, const int64_t scalesDim1,
    DispatchV2Config &config)
{
    uint32_t sharedExpertRankNum = tilingData.moeDistributeDispatchV2Info.sharedExpertRankNum;
    uint32_t sharedExpertNum = tilingData.moeDistributeDispatchV2Info.sharedExpertNum; 
    int64_t moeExpertNum = static_cast<int64_t>(tilingData.moeDistributeDispatchV2Info.moeExpertNum);
    const gert::StorageShape *xStorageShape = context->GetInputShape(config.xIndex);
    OP_TILING_CHECK(xStorageShape == nullptr, OP_LOGE(nodeName, "xShape is null."), return ge::GRAPH_FAILED);
    const int64_t xDim1 = xStorageShape->GetStorageShape().GetDim(1);
    if (sharedExpertRankNum != 0U) {
        OP_TILING_CHECK(scalesDim0 != (moeExpertNum + sharedExpertNum), OP_LOGE(nodeName,
            "scales's dim0 not equal to moeExpertNum + sharedExpertNum, scales's dim0=%ld, (moeExpertNum + sharedExpertNum)=%ld.",
            scalesDim0, moeExpertNum + sharedExpertNum), return ge::GRAPH_FAILED);
    } else {
        OP_TILING_CHECK(scalesDim0 != moeExpertNum, OP_LOGE(nodeName,
            "scales's dim0 not equal to moeExpertNum, scales's dim0=%ld, moeExpertNum=%ld.",
            scalesDim0, moeExpertNum), return ge::GRAPH_FAILED);
    }
    OP_TILING_CHECK(xDim1 != scalesDim1, OP_LOGE(nodeName, "scales's dim1 not equal to xShape's dim1, "
        "xShape's dim1=%ld, scales's dim1=%ld.", xDim1, scalesDim1), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckTensorShape(const gert::TilingContext *context, const char *nodeName,
    MoeDistributeDispatchV2TilingData &tilingData, const uint32_t quantMode, const bool isScales,
    const bool isSharedExpert, const bool hasElasticInfo, const int64_t localMoeExpertNum,
    bool isLayered, DispatchV2Config &config)
{
    auto attrs = context->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE(nodeName, "attrs is nullptr."), return ge::GRAPH_FAILED);

    auto zeroExpertNumPtr = attrs->GetAttrPointer<int64_t>(static_cast<int>(config.attrZeroExpertNumIndex));
    auto copyExpertNumPtr = attrs->GetAttrPointer<int64_t>(static_cast<int>(config.attrCopyExpertNumIndex));
    auto constExpertNumPtr = attrs->GetAttrPointer<int64_t>(static_cast<int>(config.attrConstExpertNumIndex));

    int64_t zeroExpertNum = *zeroExpertNumPtr;
    int64_t copyExpertNum = *copyExpertNumPtr;
    int64_t constExpertNum = *constExpertNumPtr;

    uint32_t A = 0U;
    uint32_t globalBs = tilingData.moeDistributeDispatchV2Info.globalBs;
    uint32_t sharedExpertNum = tilingData.moeDistributeDispatchV2Info.sharedExpertNum;
    uint32_t sharedExpertRankNum = tilingData.moeDistributeDispatchV2Info.sharedExpertRankNum;

    // 校验输入x的维度1并设h, bs已校验过
    const gert::StorageShape *xStorageShape = context->GetInputShape(config.xIndex);
    const int64_t xDim0 = xStorageShape->GetStorageShape().GetDim(0);
    const int64_t xDim1 = xStorageShape->GetStorageShape().GetDim(1);
    int64_t hMax = isLayered ? H_MAX_LAYERED : H_MAX;
    int64_t hMin = isLayered ? 0 : H_MIN;
    OP_TILING_CHECK((isLayered && (xDim1 % H_BASIC_BLOCK_LAYERED != 0)), OP_LOGE(nodeName,
        "xShape dims1(H) should be %u Aligned, but got %ld.", H_BASIC_BLOCK_LAYERED, xDim1), return ge::GRAPH_FAILED);
    OP_TILING_CHECK((xDim1 < hMin) || (xDim1 > hMax), OP_LOGE(nodeName,
        "xShape dims1(H) should be in [%ld, %ld], but got %ld.", hMin, hMax, xDim1), return ge::GRAPH_FAILED);
    tilingData.moeDistributeDispatchV2Info.h = static_cast<uint32_t>(xDim1);

    // 校验expert_id的维度并设k
    int64_t moeExpertNum = static_cast<int64_t>(tilingData.moeDistributeDispatchV2Info.moeExpertNum);
    const gert::StorageShape *expertIdStorageShape = context->GetInputShape(config.expertIdsIndex);
    const int64_t expertIdsDim0 = expertIdStorageShape->GetStorageShape().GetDim(0);
    const int64_t expertIdsDim1 = expertIdStorageShape->GetStorageShape().GetDim(1);
    OP_TILING_CHECK(xDim0 != expertIdsDim0, OP_LOGE(nodeName, "xShape's dim0 not equal to expertIdShape's dim0, "
        "xShape's dim0 is %ld, expertIdShape's dim0 is %ld.", xDim0, expertIdsDim0), return ge::GRAPH_FAILED);
    OP_TILING_CHECK((expertIdsDim1 <= 0) || (expertIdsDim1 > K_MAX) || (expertIdsDim1 > moeExpertNum + zeroExpertNum + copyExpertNum + constExpertNum),
        OP_LOGE(nodeName, "expertIdShape's dim1(k) should be in (0, min(%ld, moeExpertNum + zeroExpertNum + copyExpertNum + constExpertNum = %ld)], "
        "but got expertIdShape's dim1=%ld.", K_MAX, moeExpertNum + zeroExpertNum + copyExpertNum + constExpertNum, expertIdsDim1), return ge::GRAPH_FAILED);
    if (isLayered) {
        const gert::StorageShape *expertScaleStorageShape = context->GetOptionalInputShape(EXPERT_SCALES_INDEX);
        const int64_t expertScalesDim0 = expertScaleStorageShape->GetStorageShape().GetDim(0);
        const int64_t expertScalesDim1 = expertScaleStorageShape->GetStorageShape().GetDim(1);
        OP_TILING_CHECK((expertScalesDim0 != expertIdsDim0) || (expertScalesDim1 != expertIdsDim1),
            OP_LOGE(nodeName, "expertScaleShape's dim not equal to expertIdShape's dim, "
            "expertScaleShape's dim0 is %ld, expertIdShape's dim0 is %ld. expertScaleShape's dim1 is %ld, expertIdShape's dim1 is %ld.",
            expertScalesDim0, expertIdsDim0, expertScalesDim1, expertIdsDim1), return ge::GRAPH_FAILED);
    }
    tilingData.moeDistributeDispatchV2Info.k = static_cast<uint32_t>(expertIdsDim1);

    // 校验scales的维度
    uint64_t h = tilingData.moeDistributeDispatchV2Info.h;
    uint32_t bs = tilingData.moeDistributeDispatchV2Info.bs;
    uint64_t scalesRow = 0;
    uint64_t scalesCol = 0;
    uint32_t scalesTypeSize = 0;
    uint64_t scalesCount = 0;
    if (isScales) {
        auto scalesDesc = context->GetOptionalInputDesc(config.scalesIndex);
        const gert::StorageShape *scalesStorageShape = context->GetOptionalInputShape(config.scalesIndex);
        OP_TILING_CHECK(scalesStorageShape == nullptr, OP_LOGE(nodeName, "scalesShape is null."), return ge::GRAPH_FAILED);
        OP_TILING_CHECK(scalesDesc == nullptr, OP_LOGE(nodeName, "scalesDesc is null."), return ge::GRAPH_FAILED);
        size_t scalesDimNum = scalesStorageShape->GetStorageShape().GetDimNum();
        const int64_t scalesDim0 = scalesStorageShape->GetStorageShape().GetDim(0);
        scalesRow = static_cast<uint64_t>(scalesDim0);  
        scalesTypeSize = ge::GetSizeByDataType(scalesDesc->GetDataType());
        if (scalesDimNum == ONE_DIM) {
            //A3不会进此分支
            auto expandXDesc = context->GetOutputDesc(OUTPUT_EXPAND_X_INDEX);
            OP_TILING_CHECK((quantMode == static_cast<uint32_t>(QuantModeA5::STATIC_QUANT)) && (expandXDesc->GetDataType() == ge::DT_INT8) 
                && (scalesDim0 != static_cast<int64_t>(h)) && (scalesDim0 != static_cast<int64_t>(STATIC_SCALE_DIM_0)),
                OP_LOGE(nodeName, "The expected scalesDim0 is %lu or %lu in static quant, but got %ld", 
                h, STATIC_SCALE_DIM_0, scalesDim0), return ge::GRAPH_FAILED);
            OP_TILING_CHECK((quantMode == static_cast<uint32_t>(QuantModeA5::STATIC_QUANT))
                && (expandXDesc->GetDataType() == ge::DT_HIFLOAT8) && (scalesDim0 != STATIC_SCALE_DIM_0),
                OP_LOGE(nodeName, "The expected scalesDim0 is 1 when expandX datatype is hif8 in static quant, but got %ld", 
                scalesDim0), return ge::GRAPH_FAILED);
            scalesCol = ONE_DIM_SCALE_COL_NUM;
            scalesCount = static_cast<uint64_t>(scalesDim0);
        } else if (quantMode == static_cast<uint32_t>(QuantModeA5::NON_QUANT)) {
            OP_TILING_CHECK(scalesDim0 != bs,
                OP_LOGE(nodeName, "The expected scalesDim0 is %u when scales is not null in non-quant, but got %ld", 
                bs, scalesDim0), return ge::GRAPH_FAILED);
        } else {
            const int64_t scalesDim1 = scalesStorageShape->GetStorageShape().GetDim(1);
            OP_TILING_CHECK(
                CheckTwoDimScalesShape(context, nodeName, tilingData, scalesDim0, scalesDim1, config)
                    != ge::GRAPH_SUCCESS,
                OP_LOGE(nodeName, "CheckTwoDimScalesShape failed."), return ge::GRAPH_FAILED);
            scalesCol = static_cast<uint64_t>(scalesDim1);
            scalesCount = static_cast<uint64_t>(scalesDim0 * scalesDim1);
        }
    }
    tilingData.moeDistributeDispatchV2Info.scalesRow = scalesRow;
    tilingData.moeDistributeDispatchV2Info.scalesCol = scalesCol;
    tilingData.moeDistributeDispatchV2Info.scalesCount = scalesCount;
    tilingData.moeDistributeDispatchV2Info.scalesTypeSize = scalesTypeSize;

    uint32_t rankNumPerSharedExpert = 0;
    uint32_t epWorldSizeU32 = tilingData.moeDistributeDispatchV2Info.epWorldSize;
    uint32_t maxBs = globalBs / epWorldSizeU32;
    uint32_t maxSharedGroupNum = 0;
    if ((sharedExpertNum != 0U) && (sharedExpertRankNum != 0U)) { // 除零保护
        rankNumPerSharedExpert = sharedExpertRankNum / sharedExpertNum;
        maxSharedGroupNum = (epWorldSizeU32 + rankNumPerSharedExpert - 1U) / rankNumPerSharedExpert;
    }
    if (isSharedExpert) { // 本卡为共享专家
        A = maxBs * maxSharedGroupNum;
    } else {     // 本卡为moe专家
        A = globalBs * std::min(localMoeExpertNum, expertIdsDim1);
    }

    // 校验elasticInfo的维度，并更新一下最大输出的值
    if (hasElasticInfo) {
        const gert::StorageShape *elasticInfoStorageShape = context->GetOptionalInputShape(config.elasticInfoIndex);
        const int64_t elasticInfoDim0 = elasticInfoStorageShape->GetStorageShape().GetDim(0);
        const int64_t epWorldSize = static_cast<int64_t>(tilingData.moeDistributeDispatchV2Info.epWorldSize);

        OP_TILING_CHECK(elasticInfoDim0 != (ELASTIC_METAINFO_OFFSET + RANK_LIST_NUM * epWorldSize),
            OP_LOGE(nodeName, "elasticInfo's dim0 not equal to 4 + 2 * epWorldSize, "
            "elasticInfo's dim0 is %ld, epWorldSize is %ld.",
            elasticInfoDim0, epWorldSize), return ge::GRAPH_FAILED);
        A = std::max( static_cast<int64_t>(maxBs * maxSharedGroupNum) , globalBs * std::min(localMoeExpertNum, expertIdsDim1));
    }
    tilingData.moeDistributeDispatchV2Info.a = A;

    // 校验expandX的维度
    int64_t tpWorldSize = static_cast<int64_t>(tilingData.moeDistributeDispatchV2Info.tpWorldSize);
    const gert::StorageShape *expandXStorageShape = context->GetOutputShape(OUTPUT_EXPAND_X_INDEX);
    const int64_t expandXDim0 = expandXStorageShape->GetStorageShape().GetDim(0);
    const int64_t expandXDim1 = expandXStorageShape->GetStorageShape().GetDim(1);
    OP_TILING_CHECK(expandXDim0 < tpWorldSize * static_cast<int64_t>(A), OP_LOGE(nodeName,
        "expandX's dim0 not greater than or equal to A*tpWorldSize, "
        "expandX's dim0 is %ld, A*tpWorldSize is %ld.", expandXDim0, tpWorldSize * A), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(xDim1 != expandXDim1, OP_LOGE(nodeName, "expandX's dim1 not equal to xShape's dim1, "
        "xShape's dim1 is %ld, expandX's dim1 is %ld.", xDim1, expandXDim1), return ge::GRAPH_FAILED);

    // 校验dynamicScales的维度
    if ((quantMode != static_cast<uint32_t>(QuantModeA5::NON_QUANT)) && (quantMode != static_cast<uint32_t>(QuantModeA5::STATIC_QUANT))) {
        // Dim0
        const gert::StorageShape *dynamicScalesStorageShape = context->GetOutputShape(OUTPUT_DYNAMIC_SCALES_INDEX);
        const int64_t dynamicScalesDim0 = dynamicScalesStorageShape->GetStorageShape().GetDim(0);
        OP_TILING_CHECK(dynamicScalesDim0 < static_cast<int64_t>(A) * tpWorldSize, OP_LOGE(nodeName,
            "dynamicScales's dim0 should be equal to or greater than A*tpWorldSize, dynamicScales's dim0=%ld, A*tpWorldSize=%ld.",
            dynamicScalesDim0, A * tpWorldSize), return ge::GRAPH_FAILED);
        // Dim1, only for pergroup and mx
        if (quantMode != static_cast<uint32_t>(QuantModeA5::PERTOKEN_DYNAMIC_QUANT)) {
            const uint64_t dynamicScalesDim1 = static_cast<uint64_t>(dynamicScalesStorageShape->GetStorageShape().GetDim(1));
            OP_TILING_CHECK((quantMode == static_cast<uint32_t>(QuantModeA5::MX_QUANT)) 
                && (dynamicScalesDim1 != ops::CeilAlign(static_cast<uint64_t>(ops::CeilDiv(h, MX_BLOCK_SIZE)), EVEN_ALIGN)),
                OP_LOGE(nodeName, "dynamicScales's dim1 should be equal to %lu and even when quantMode=%u, but got %lu.",
                ops::CeilAlign(static_cast<uint64_t>(ops::CeilDiv(h, MX_BLOCK_SIZE)), EVEN_ALIGN), quantMode, dynamicScalesDim1), 
                return ge::GRAPH_FAILED);
            OP_TILING_CHECK((dynamicScalesDim1 != ops::CeilDiv(h, PERGROUP_BLOCK_SIZE)) && 
                (quantMode == static_cast<uint32_t>(QuantModeA5::PERGROUP_DYNAMIC_QUANT)), 
                OP_LOGE(nodeName, "dynamicScales's dim1 should be equal to %lu when quantMode=%u, but got %lu.",
                ops::CeilDiv(h, PERGROUP_BLOCK_SIZE), quantMode, dynamicScalesDim1), return ge::GRAPH_FAILED);
        }
    }

    // 校验assistInfo的维度
    const gert::StorageShape *assistInfoStorageShape = context->GetOutputShape(OUTPUT_ASSIST_INFO_INDEX);
    const int64_t assistInfoDim0 = assistInfoStorageShape->GetStorageShape().GetDim(0);
    int64_t minAssistInfoDim0 = static_cast<int64_t>(A * ASSIST_NUM_PER_A);
    OP_TILING_CHECK(assistInfoDim0 < minAssistInfoDim0, OP_LOGE(nodeName, "assistInfoDim0 < minAssistInfoDim0,"
        " assistInfoDim0 is %ld, minAssistInfoDim0 is %ld.", assistInfoDim0, minAssistInfoDim0), return ge::GRAPH_FAILED);

    // 校验expertTokenNums的维度
    const gert::StorageShape *expertTokenNumsStorageShape = context->GetOutputShape(OUTPUT_EXPERT_TOKEN_NUMS_INDEX);
    const int64_t expertTokenNumsDim0 = expertTokenNumsStorageShape->GetStorageShape().GetDim(0);

    if (hasElasticInfo) {
        OP_TILING_CHECK(expertTokenNumsDim0 != (localMoeExpertNum > 1 ? localMoeExpertNum : 1),  OP_LOGE(nodeName,
            "elastic scaling expertTokenNums's Dim0 not equal to max(localMoeExpertNum,1), expertTokenNumsDim0 is %ld, "
            "localMoeExpertNum is %ld.", expertTokenNumsDim0, localMoeExpertNum), return ge::GRAPH_FAILED);
    } else if (isSharedExpert) {
        OP_TILING_CHECK(expertTokenNumsDim0 != 1, OP_LOGE(nodeName, "shared expertTokenNums's dim0 %ld not equal to 1.",
            expertTokenNumsDim0), return ge::GRAPH_FAILED);
    } else {
        OP_TILING_CHECK(expertTokenNumsDim0 != localMoeExpertNum, OP_LOGE(nodeName,
            "moe expertTokenNums's Dim0 not equal to localMoeExpertNum, expertTokenNumsDim0 is %ld, "
            "localMoeExpertNum is %ld.", expertTokenNumsDim0, localMoeExpertNum), return ge::GRAPH_FAILED);
    }

    // 校验epRecvCount和tpRecvCount的维度
    int64_t epWorldSize = static_cast<int64_t>(tilingData.moeDistributeDispatchV2Info.epWorldSize);
    const gert::StorageShape *epRecvCountStorageShape = context->GetOutputShape(OUTPUT_EP_RECV_COUNTS_INDEX);
    const gert::StorageShape *tpRecvCountStorageShape = context->GetOutputShape(OUTPUT_TP_RECV_COUNTS_INDEX);
    const int64_t epRecvCountDim0 = epRecvCountStorageShape->GetStorageShape().GetDim(0);
    const int64_t tpRecvCountDim0 = tpRecvCountStorageShape->GetStorageShape().GetDim(0);
    int64_t epRecvCount = ((isSharedExpert) ? epWorldSize : epWorldSize * localMoeExpertNum);
    if (hasElasticInfo){
        epRecvCount = std::max(epWorldSize, epWorldSize * localMoeExpertNum);
    }
    if (tpWorldSize == MAX_TP_WORLD_SIZE) {
        epRecvCount *= tpWorldSize;
    }
    if (isLayered) {
        // 如果是分层方案，则需要校验全新的shape，额外的globalBs * 2 * k * epWorldSize / 8 用来存储token的cnt信息与offset信息，为了兼容A2&A5 这里取/8。
        epRecvCount = epWorldSize * localMoeExpertNum + globalBs * 2 * expertIdsDim1 * epWorldSize / RANK_NUM_PER_NODE_A2;
        OP_TILING_CHECK(epRecvCountDim0 < epRecvCount, OP_LOGE(nodeName,
        "dimension 0 of epRecvCount should be greater than or equal to epWorldSize * localMoeExpertNum + globalBs * 2 * k * epWorldSize / 8, "
        "but dimension 0 of epRecvCount is %ld, epWorldSize is %ld, localMoeExpertNum is %ld, k is %ld.",
        epRecvCountDim0, epWorldSize, localMoeExpertNum, expertIdsDim1), return ge::GRAPH_FAILED);
    } else {
        OP_TILING_CHECK(epRecvCountDim0 < epRecvCount, OP_LOGE(nodeName,
        "dimension 0 of epRecvCount should be greater than or equal to epWorldSize * localMoeExpertNum * tpWorldSize, "
        "but dimension 0 of epRecvCount is %ld, epWorldSize is %ld, localMoeExpertNum is %ld, tpWorldSize is %ld.",
        epRecvCountDim0, epWorldSize, localMoeExpertNum, tpWorldSize), return ge::GRAPH_FAILED);
    }
    OP_TILING_CHECK(tpRecvCountDim0 != tpWorldSize, OP_LOGE(nodeName,
        "dimension 0 of tpRecvCount should be equal to tpWorldSize, but dimension 0 of tpRecvCount is %ld, "
        "tpWorldSize is %ld.", tpRecvCountDim0, tpWorldSize), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

//校验输入输出数据格式
static bool CheckTensorPtrNullptr(const gert::TilingContext *context, const char *nodeName,
    const bool isScales, const uint32_t quantMode, const bool isActiveMask, const bool hasElasticInfo,
    DispatchV2Config &config)
{
    auto xDesc = context->GetInputDesc(config.xIndex);
    auto expertIdDesc = context->GetInputDesc(config.expertIdsIndex);
    OP_TILING_CHECK(xDesc == nullptr, OP_LOGE(nodeName, "xDesc is null."), return false);
    OP_TILING_CHECK(expertIdDesc == nullptr, OP_LOGE(nodeName, "expertIdDesc is null."), return false);
    if (isScales) {
        auto scalesDesc = context->GetOptionalInputDesc(config.scalesIndex);
        OP_TILING_CHECK(scalesDesc == nullptr, OP_LOGE(nodeName, "scalesDesc is null."), return false);
    }
    if (isActiveMask) {
        auto xActiveMaskDesc = context->GetOptionalInputDesc(config.xActiveMaskIndex);
        OP_TILING_CHECK(xActiveMaskDesc == nullptr, OP_LOGE(nodeName, "xActiveMaskDesc is null."), return false);
    }
    if (hasElasticInfo) {
        auto elasticInfoDesc = context->GetOptionalInputDesc(config.elasticInfoIndex);
        OP_TILING_CHECK(elasticInfoDesc == nullptr, OP_LOGE(nodeName, "elasticInfoDesc is null."), return false);
    }
    auto expandXDesc = context->GetOutputDesc(OUTPUT_EXPAND_X_INDEX);
    auto assistInfoDesc = context->GetOutputDesc(OUTPUT_ASSIST_INFO_INDEX);
    auto expertTokenNumsDesc = context->GetOutputDesc(OUTPUT_EXPERT_TOKEN_NUMS_INDEX);
    auto epRecvCountsDesc = context->GetOutputDesc(OUTPUT_EP_RECV_COUNTS_INDEX);
    auto tpRecvCountsDesc = context->GetOutputDesc(OUTPUT_TP_RECV_COUNTS_INDEX);
    OP_TILING_CHECK(expandXDesc == nullptr, OP_LOGE(nodeName, "expandXDesc is null."), return false);
    if (quantMode >= static_cast<uint32_t>(QuantModeA5::PERTOKEN_DYNAMIC_QUANT)) {
        auto dynamicScalesDesc = context->GetOutputDesc(OUTPUT_DYNAMIC_SCALES_INDEX);
        OP_TILING_CHECK(dynamicScalesDesc == nullptr, OP_LOGE(nodeName, "dynamicScalesDesc is null."),
            return false);
    }
    OP_TILING_CHECK(assistInfoDesc == nullptr, OP_LOGE(nodeName, "assistInfoDesc is null."), return false);
    OP_TILING_CHECK(expertTokenNumsDesc == nullptr, OP_LOGE(nodeName, "expertTokenNumsDesc is null."), return false);
    OP_TILING_CHECK(epRecvCountsDesc == nullptr, OP_LOGE(nodeName, "epRecvCountsDesc is null."), return false);
    OP_TILING_CHECK(tpRecvCountsDesc == nullptr, OP_LOGE(nodeName, "tpRecvCountsDesc is null."), return false);

    return true;
}

static ge::graphStatus TilingCheckMoeDistributeDispatch(gert::TilingContext *context, const char *nodeName,
    const bool isActiveMask, const bool isScales, const bool hasElasticInfo, const uint32_t quantMode,
    const bool isLayered, DispatchV2Config &config)
{
    OP_TILING_CHECK(!CheckTensorPtrNullptr(context, nodeName, isScales, quantMode, isActiveMask, hasElasticInfo,
                                    config),
        OP_LOGE(nodeName, "params check nulld failed."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(!CheckTensorDim(context, nodeName, isScales, quantMode, isActiveMask, hasElasticInfo,
                                    isLayered, config),
        OP_LOGE(nodeName, "params shape is invalid."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(!CheckTensorDataType(context, nodeName, isScales, quantMode, isActiveMask, hasElasticInfo,
                                         config),
        OP_LOGE(nodeName, "params dataType is invalid."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(!CheckTensorFormat(context, nodeName, isScales, quantMode, isActiveMask, hasElasticInfo,
                                       config),
        OP_LOGE(nodeName, "params format is invalid."), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static uint64_t CalTilingKey(const gert::TilingContext *context, const bool isScales, const uint32_t quantMode,
    const uint32_t tpWorldSize, const bool isSetFullMeshV2, bool isLayered)
{
    uint32_t templateDispatch = TILINGKEY_NO_FULLMESH;
    bool tp = false;
    uint32_t tilingKeyQuantMode = quantMode;
    bool scaleMode = false;
    uint64_t tilingKey;
    uint32_t commMode = TILINGKEY_TPL_MTE;
    if (isScales) {
        scaleMode = true;
    }
    if (isSetFullMeshV2) {
        templateDispatch = TILINGKEY_ENABLE_FULLMESH;
    }
    if (isLayered) {
        templateDispatch = TILINGKEY_ENABLE_HIERARCHY;
    }
    if (mc2tiling::GetNpuArch(context) == NpuArch::DAV_3510) {
        tilingKey = GET_TPL_TILING_KEY(tp, tilingKeyQuantMode, scaleMode,
                                                templateDispatch, commMode, TILINGKEY_TPL_A5);
    } else {
        if (tpWorldSize == MAX_TP_WORLD_SIZE) {
            tp = true;
        }
        tilingKey = GET_TPL_TILING_KEY(tp, tilingKeyQuantMode, scaleMode, 
                                                templateDispatch, commMode, TILINGKEY_TPL_A3);
    }
    return tilingKey;
}

static ge::graphStatus SetHcommCfg(const gert::TilingContext *context, MoeDistributeDispatchV2TilingData *tiling,
    const std::string groupEp, const std::string groupTp, const uint32_t tpWorldSize, bool isLayered)
{
    const char *nodeName = context->GetNodeName();
    OP_LOGD(nodeName, "MoeDistributeDispatchV2 groupEp = %s", groupEp.c_str());
    uint32_t opType1 = OP_TYPE_ALL_TO_ALL;
    uint32_t opType2 = OP_TYPE_ALL_GATHER;
    std::string algConfigAllToAllStr = isLayered ? "AlltoAll=level1:hierarchy" : "AlltoAll=level0:fullmesh;level1:pairwise";
    std::string algConfigAllGatherStr = "AllGather=level0:ring";

    AscendC::Mc2CcTilingConfig mc2CcTilingConfig(groupEp, opType1, algConfigAllToAllStr);
    mc2CcTilingConfig.SetCommEngine(mc2tiling::AIV_ENGINE);   // 通过不拉起AICPU，提高算子退出性能
    OP_TILING_CHECK(mc2CcTilingConfig.GetTiling(tiling->mc2InitTiling) != 0,
            OP_LOGE(nodeName, "mc2CcTilingConfig mc2InitTiling GetTiling failed"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(mc2CcTilingConfig.GetTiling(tiling->mc2CcTiling1) != 0,
            OP_LOGE(nodeName, "mc2CcTilingConfig mc2CcTiling1 GetTiling failed"), return ge::GRAPH_FAILED);

    if (tpWorldSize > 1) {
        OP_LOGD(nodeName, "MoeDistributeDispatchV2 groupTp = %s", groupTp.c_str());
        mc2CcTilingConfig.SetGroupName(groupTp);
        mc2CcTilingConfig.SetOpType(opType2);
        mc2CcTilingConfig.SetAlgConfig(algConfigAllGatherStr);
        OP_TILING_CHECK(mc2CcTilingConfig.GetTiling(tiling->mc2CcTiling2) != 0,
            OP_LOGE(nodeName, "mc2CcTilingConfig mc2CcTiling2 GetTiling failed"), return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckTpWinSize(const gert::TilingContext *context, MoeDistributeDispatchV2TilingData &tilingData, const char *nodeName,
    uint64_t tokenNeedSizeDispatch, uint64_t tokenNeedSizeCombine)
{
    auto attrs = context->GetAttrs();
    uint64_t tpWorldSize = static_cast<uint64_t>(tilingData.moeDistributeDispatchV2Info.tpWorldSize);
    if (tpWorldSize == TP_WORLD_SIZE_TWO) {
        uint64_t maxWindowSizeTp = 0;
        auto groupTpHccl = attrs->GetAttrPointer<char>(static_cast<int>(ATTR_GROUP_TP_INDEX));
        OP_TILING_CHECK(mc2tiling::GetCclBufferSize(groupTpHccl, &maxWindowSizeTp, nodeName) != ge::GRAPH_SUCCESS,
            OP_LOGE(nodeName, "Get Ep HcclBufferSizeTP failed, HcclBufferSizeTP is %lu", maxWindowSizeTp),
            return ge::GRAPH_FAILED);
        uint64_t actualSize = static_cast<uint64_t>(tilingData.moeDistributeDispatchV2Info.a) * (tokenNeedSizeDispatch +
        tokenNeedSizeCombine) * DOUBLE_DATA_BUFFER;
        OP_TILING_CHECK((actualSize > maxWindowSizeTp),
        OP_LOGE(nodeName, "TP HCCL_BUFFSIZE is too SMALL, A = %u, tokenNeedSizeDispatch = %lu, tokenNeedSizeCombine = %lu,"
            "NEEDED_HCCL_BUFFSIZE(A * (tokenNeedSizeDispatch + tokenNeedSizeCombine) * 2UL) = %luMB, TP HCCL_BUFFSIZE=%luMB.",
            tilingData.moeDistributeDispatchV2Info.a,  tokenNeedSizeDispatch, tokenNeedSizeCombine, actualSize / MB_SIZE + 1UL,
            maxWindowSizeTp / MB_SIZE), return ge::GRAPH_FAILED);
        tilingData.moeDistributeDispatchV2Info.totalWinSizeTp = maxWindowSizeTp;
        OP_LOGD(nodeName, "TpwindowSize = %lu", maxWindowSizeTp);
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckWinSize(const gert::TilingContext *context, MoeDistributeDispatchV2TilingData &tilingData,
    const char *nodeName, const bool isSetFullMeshV2, uint32_t &localMoeExpertNum, bool isLayered)
{
    auto attrs = context->GetAttrs();
    uint64_t hcclBufferSizeEp = 0;
    uint64_t maxWindowSizeEp = 0;
    OP_TILING_CHECK(
        mc2tiling::GetEpWinSize(context, nodeName, hcclBufferSizeEp, maxWindowSizeEp, ATTR_GROUP_EP_INDEX, isLayered) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "Get EP WinSize failed"), return ge::GRAPH_FAILED);
    uint32_t sharedExpertNum = tilingData.moeDistributeDispatchV2Info.sharedExpertNum;
    uint64_t h = static_cast<uint64_t>(tilingData.moeDistributeDispatchV2Info.h);
    uint64_t k = static_cast<uint64_t>(tilingData.moeDistributeDispatchV2Info.k);
    uint64_t bs = static_cast<uint64_t>(tilingData.moeDistributeDispatchV2Info.bs);
    uint64_t epWorldSize = static_cast<uint64_t>(tilingData.moeDistributeDispatchV2Info.epWorldSize);
    uint64_t maxBs = static_cast<uint64_t>(tilingData.moeDistributeDispatchV2Info.globalBs) / epWorldSize;
    // combine数据区 token首地址对齐512
    uint64_t tokenNeedSizeCombine = ((h * MAX_OUT_DTYPE_SIZE  + WIN_ADDR_ALIGN - 1UL) / WIN_ADDR_ALIGN) * WIN_ADDR_ALIGN;
    // dispatch数据区 token首对齐512，有效token长度h_align_32b + scale(32b) + 三元组(3*4b)
    uint64_t tokenActualLen = ((h * MAX_OUT_DTYPE_SIZE  + UB_ALIGN - 1UL) / UB_ALIGN) * UB_ALIGN + SCALE_EXPAND_IDX_BUFFER;
    uint64_t tokenNeedSizeDispatch = 0;
    uint64_t moeExpertNum = static_cast<uint64_t>(tilingData.moeDistributeDispatchV2Info.moeExpertNum);
    if (isSetFullMeshV2) {
        tokenNeedSizeDispatch = ((tokenActualLen + FULL_MESH_DATA_ALIGN - 1UL) / FULL_MESH_DATA_ALIGN) * WIN_ADDR_ALIGN;
    } else {
        tokenNeedSizeDispatch = ((tokenActualLen + WIN_ADDR_ALIGN - 1UL) / WIN_ADDR_ALIGN) * WIN_ADDR_ALIGN;
    }
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
    tilingData.moeDistributeDispatchV2Info.totalWinSizeEp = maxWindowSizeEp;
    OP_LOGD(nodeName, "windowSize = %lu", maxWindowSizeEp);
    OP_TILING_CHECK(CheckTpWinSize(context, tilingData, nodeName, tokenNeedSizeDispatch, tokenNeedSizeCombine) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "Tiling check Tp window size failed."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static uint32_t Ceil(uint32_t dividend, uint32_t divisor, const char *nodeName)
{
    if (divisor != 0){
        return ((dividend + divisor - 1 ) / divisor) * divisor;
    }
    OP_LOGE(nodeName, "ceil divisor should not be 0");
    return 0;
}

static uint32_t SendToMoeExpertUsedBuffer(const gert::TilingContext *context,
    const MoeDistributeDispatchV2TilingData &tilingData, uint32_t expertIdsBufSize, uint32_t kSize, const char *nodeName)
{
    uint32_t UbForMoe = 0;
    uint32_t moeExpertNum = tilingData.moeDistributeDispatchV2Info.moeExpertNum;
    uint32_t sharedExpertRankNum = tilingData.moeDistributeDispatchV2Info.sharedExpertRankNum;
    uint32_t sharedExpertNum = tilingData.moeDistributeDispatchV2Info.sharedExpertNum;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint32_t totalExpertNum = sharedExpertRankNum + moeExpertNum;
    uint32_t aivUsedCumSum = totalExpertNum / 32; // 单核处理32个专家cnt发送
    aivUsedCumSum = (aivUsedCumSum == 0) ? 1 : aivUsedCumSum;
    aivUsedCumSum = (aivUsedCumSum >= (aivNum / HALF_NUM)) ? (aivNum / HALF_NUM) : aivUsedCumSum;
    uint32_t aivUsedAllToAll = aivNum - aivUsedCumSum;
    uint32_t sharedUsedAivNum = 0;
    if (sharedExpertRankNum != 0U) {
        sharedUsedAivNum = (aivUsedAllToAll * sharedExpertNum) / (kSize + sharedExpertNum);
        if (sharedUsedAivNum == 0) {
            sharedUsedAivNum = 1;
        }
    }
    uint32_t moeUsedAivNum = aivUsedAllToAll - sharedUsedAivNum; 
    uint32_t maskSizePerExpert = Ceil((expertIdsBufSize / sizeof(int32_t)) / 8, UB_ALIGN, nodeName); // 8 is 1byte->8bit
    uint32_t expertMaskBufSize = maskSizePerExpert * Ceil(moeExpertNum, moeUsedAivNum, nodeName) / moeUsedAivNum;
    UbForMoe = UbForMoe + expertMaskBufSize; // expertMaskBuf
    UbForMoe = UbForMoe + Ceil(moeExpertNum * sizeof(int32_t), UB_ALIGN, nodeName); // tokenNumToExpertBuf
    return UbForMoe;
}

static uint32_t AllToAllBasicUsedBuffer(const gert::TilingContext *context, const MoeDistributeDispatchV2TilingData &tilingData,
    uint32_t quantMode, uint32_t &expertIdsBufSize, uint32_t hSize, uint32_t expertIdsCnt, uint32_t &hOutSizeAlign, const char *nodeName)
{
    uint32_t UbForBasic = 0;
    uint32_t epWorldSize = tilingData.moeDistributeDispatchV2Info.epWorldSize;
    if (tilingData.moeDistributeDispatchV2Info.hasElasticInfo) {
        uint32_t elasticInfoSize = (ELASTIC_INFO_OFFSET + RANK_LIST_NUM * epWorldSize) * sizeof(int32_t);
        uint32_t elasticInfoSizeAlign = Ceil(elasticInfoSize, UB_ALIGN, nodeName);
        UbForBasic = UbForBasic + elasticInfoSizeAlign; // elasticInfoBuf_
    }

    uint32_t sizeofX = 2U;
    uint32_t hAlignSize = Ceil(hSize * sizeofX, UB_ALIGN, nodeName);
    UbForBasic = UbForBasic + hAlignSize * BUFFER_NUM; //inQueue

    const gert::StorageShape *scalesStorageShape = context->GetOptionalInputShape(SCALES_INDEX);
    bool isScales = (scalesStorageShape != nullptr);
    uint32_t hOutSize = (quantMode == static_cast<uint32_t>(QuantModeA5::NON_QUANT)) ? hSize * SIZE_OF_UNQUANT : hSize;
    hOutSizeAlign = Ceil(hOutSize, UB_ALIGN, nodeName);
    if ((quantMode == static_cast<uint32_t>(QuantModeA5::NON_QUANT)) && isScales) {
        uint32_t scaleInBytes = tilingData.moeDistributeDispatchV2Info.scalesCol *
                    tilingData.moeDistributeDispatchV2Info.scalesTypeSize;
        hOutSizeAlign += scaleInBytes;
    } else if (quantMode == static_cast<uint32_t>(QuantModeA5::PERTOKEN_DYNAMIC_QUANT)) {
        hOutSizeAlign += sizeof(float);
    }
    uint32_t hScaleSizeAlign = Ceil(hOutSizeAlign, UB_ALIGN, nodeName);
    uint32_t tokenQuantAlign = hScaleSizeAlign / sizeof(int32_t);
    hOutSizeAlign = tokenQuantAlign * sizeof(int32_t) + UB_ALIGN;
    uint32_t blockCntPerToken = Ceil(hOutSizeAlign, SPLIT_BLOCK_DATA_SIZE, nodeName) / SPLIT_BLOCK_DATA_SIZE;
    uint32_t hCommuSize = blockCntPerToken * SPLIT_BLOCK_SIZE;
    UbForBasic = UbForBasic + hCommuSize * BUFFER_NUM; // outBuf

    expertIdsBufSize = Ceil(expertIdsCnt * sizeof(int32_t), SIZE_ALIGN_256, nodeName);
    UbForBasic = UbForBasic + expertIdsBufSize; //expertIdsBuf_
    return UbForBasic;
}

static ge::graphStatus CheckUBSize(const gert::TilingContext *context, MoeDistributeDispatchV2TilingData &tilingData,
    const char *nodeName)
{
    uint32_t ubSize = UB_ALIGN; // dataStateBuf

    uint32_t quantMode = tilingData.moeDistributeDispatchV2Info.quantMode;
    uint32_t hSize = tilingData.moeDistributeDispatchV2Info.h;
    uint32_t bsSize = tilingData.moeDistributeDispatchV2Info.bs;
    uint32_t kSize = tilingData.moeDistributeDispatchV2Info.k;
    uint32_t expertIdsCnt = bsSize * kSize;
    uint32_t expertIdsBufSize = 0;
    uint32_t hOutSizeAlign = 0;
    ubSize = ubSize + AllToAllBasicUsedBuffer(context, tilingData, quantMode, expertIdsBufSize, hSize, expertIdsCnt, hOutSizeAlign, nodeName);

    const gert::StorageShape *xActiveMaskStorageShape = context->GetOptionalInputShape(X_ACTIVE_MASK_INDEX);
    bool isActiveMask = (xActiveMaskStorageShape != nullptr);
    bool needMaskCalFlag = (isActiveMask || tilingData.moeDistributeDispatchV2Info.zeroComputeExpertNum != 0);
    if (needMaskCalFlag) {
        ubSize = ubSize + expertIdsBufSize; //gatherMaskTBuf_
    }

    uint32_t hFp32Size = Ceil(hSize * sizeof(float), UB_ALIGN, nodeName);
    uint32_t bsKAlign256 = Ceil(expertIdsCnt * SIZE_OF_HALF, SIZE_ALIGN_256, nodeName);
    uint32_t expertIdsSize = Ceil(expertIdsCnt * sizeof(int32_t), UB_ALIGN, nodeName);
    uint32_t xActivateMaskSize = bsSize * Ceil(kSize * sizeof(bool), UB_ALIGN, nodeName) * SIZE_OF_HALF;
    uint32_t maxSizeForUbBuffer = hFp32Size > expertIdsSize ? hFp32Size : expertIdsSize;
    maxSizeForUbBuffer = maxSizeForUbBuffer > xActivateMaskSize ? maxSizeForUbBuffer : xActivateMaskSize;
    maxSizeForUbBuffer = maxSizeForUbBuffer > bsKAlign256 ? maxSizeForUbBuffer : bsKAlign256;
    tilingData.moeDistributeDispatchV2Info.maxSizeForUbBuffer = maxSizeForUbBuffer;
    if (quantMode > static_cast<uint32_t>(QuantModeA5::NON_QUANT)) {
        uint32_t hOutAlignUbSize = Ceil(hOutSizeAlign, UB_ALIGN, nodeName);
        ubSize = ubSize + hOutAlignUbSize;
        ubSize = ubSize + BUFFER_NUM * maxSizeForUbBuffer; // receiveDataCastFloatBuf smoothScalesBuf
    } else if (needMaskCalFlag) {
        ubSize = ubSize + BUFFER_NUM * maxSizeForUbBuffer;
    }

    if (tilingData.moeDistributeDispatchV2Info.isExpertMask || (tilingData.moeDistributeDispatchV2Info.zeroComputeExpertNum != 0)) {
        uint32_t axisBSAlign = Ceil(bsSize * sizeof(int32_t), UB_ALIGN, nodeName);
        ubSize = ubSize +  axisBSAlign; //validBsIndexTBuf_
        uint32_t validBufferSize = expertIdsSize > xActivateMaskSize ? expertIdsSize : xActivateMaskSize;
        ubSize = ubSize + validBufferSize;  //validExpertIndexBuf_
    }

    ubSize = ubSize + SendToMoeExpertUsedBuffer(context, tilingData, expertIdsBufSize, kSize, nodeName);
    uint64_t ubRealSize = 0;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubRealSize);
    OP_TILING_CHECK(ubSize > ubRealSize, OP_LOGE(nodeName,
        "Current scenario is exceeds the size limit, BS = %u, H = %u, topK = %u, moeExpertNum = %u, aivNum available = %u, used UBsize = %u",
        bsSize, hSize, kSize, tilingData.moeDistributeDispatchV2Info.moeExpertNum, ascendcPlatform.GetCoreNumAiv(), ubSize), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SetWorkSpace(gert::TilingContext *context, const char *nodeName)
{
    size_t *workSpaces = context->GetWorkspaceSizes(1);
    OP_TILING_CHECK(workSpaces == nullptr, OP_LOGE(nodeName, "workSpaces is nullptr."),
        return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    workSpaces[0] = SYSTEM_NEED_WORKSPACE + static_cast<size_t>(WORKSPACE_ELEMENT_OFFSET * aivNum * aivNum);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeDistributeDispatchA3TilingFuncImplPublic(gert::TilingContext *context, DispatchV2Config &config)
{
    const char *nodeName = context->GetNodeName();
    MoeDistributeDispatchV2TilingData *tilingData = context->GetTilingData<MoeDistributeDispatchV2TilingData>();
    OP_TILING_CHECK(tilingData == nullptr, OP_LOGE(nodeName, "tilingData is nullptr."), return ge::GRAPH_FAILED);
    std::string groupEp = "";
    std::string groupTp = "";
    uint32_t quantMode = static_cast<uint32_t>(QuantModeA5::NON_QUANT);
    bool isScales = false;
    bool isActiveMask = false;
    bool hasElasticInfo = false;
    bool isSetFullMeshV2 = false;
    bool isLayered = false;
    uint32_t localMoeExpertNum = 1;
    OP_LOGI(nodeName, "Enter MoeDistributeDispatchV2 tiling check func.");

    // 获取入参属性
    OP_TILING_CHECK(GetAttrAndSetTilingData(context, nodeName, *tilingData, groupEp, groupTp, isSetFullMeshV2,
                                            isLayered, config) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "Get attr and set tiling data failed."), return ge::GRAPH_FAILED);

    // 获取scales
    const gert::StorageShape *scalesStorageShape = context->GetOptionalInputShape(config.scalesIndex);
    isScales = (scalesStorageShape != nullptr);

    // 获取xActiveMask
    const gert::StorageShape *xActiveMaskStorageShape = context->GetOptionalInputShape(config.xActiveMaskIndex);
    isActiveMask = (xActiveMaskStorageShape != nullptr);
    tilingData->moeDistributeDispatchV2Info.isTokenMask = ((isActiveMask) &&
        (xActiveMaskStorageShape->GetStorageShape().GetDimNum() == ONE_DIM));
    tilingData->moeDistributeDispatchV2Info.isExpertMask = ((isActiveMask) &&
        (xActiveMaskStorageShape->GetStorageShape().GetDimNum() == TWO_DIMS));

    // 获取elasticInfo
    const gert::StorageShape *elasticInfoStorageShape = context->GetOptionalInputShape(config.elasticInfoIndex);
    hasElasticInfo = (elasticInfoStorageShape != nullptr);
    tilingData->moeDistributeDispatchV2Info.hasElasticInfo = hasElasticInfo;

    quantMode = tilingData->moeDistributeDispatchV2Info.quantMode;

    // 检查quantMode和scales是否匹配
    if (mc2tiling::GetNpuArch(context) == NpuArch::DAV_3510) {
        OP_TILING_CHECK(CheckQuantModeAndScales(context, nodeName, isScales, quantMode, config) != ge::GRAPH_SUCCESS,
            OP_LOGE(nodeName, "quant mode and scales not match, isScales is %d,quantMode is %u.",
            static_cast<int32_t>(isScales),quantMode), return ge::GRAPH_FAILED);
    } else {
        OP_TILING_CHECK(quantMode == static_cast<uint32_t>(QuantModeA5::STATIC_QUANT), OP_LOGE(nodeName, "cannot support static quant now."),
            return ge::GRAPH_FAILED);
        OP_TILING_CHECK((isScales && (quantMode == static_cast<uint32_t>(QuantModeA5::NON_QUANT))) || ((!isScales) && (quantMode == static_cast<uint32_t>(QuantModeA5::STATIC_QUANT))),
            OP_LOGE(nodeName, "quant mode and scales not match, isScales is %d, quantMode is %u.",
            static_cast<int32_t>(isScales), quantMode), return ge::GRAPH_FAILED);
    }

    // 检查输入输出的dim、format、dataType
    OP_TILING_CHECK(
        TilingCheckMoeDistributeDispatch(context, nodeName, isActiveMask, isScales, hasElasticInfo,
                                         quantMode, isLayered, config) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "Tiling check param failed."), return ge::GRAPH_FAILED);

    // 检查属性的取值是否合法
    OP_TILING_CHECK(CheckAttrs(context, nodeName, *tilingData, localMoeExpertNum, isActiveMask, isSetFullMeshV2,
                               isLayered, config) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "Check attr failed."), return ge::GRAPH_FAILED);

    uint32_t epRankId = tilingData->moeDistributeDispatchV2Info.epRankId;
    uint32_t sharedExpertRankNum = tilingData->moeDistributeDispatchV2Info.sharedExpertRankNum;
    bool isSharedExpert = (epRankId < sharedExpertRankNum);

    // 检查shape各维度并赋值h,k
    OP_TILING_CHECK(CheckTensorShape(context, nodeName, *tilingData, quantMode, isScales,
        isSharedExpert, hasElasticInfo, static_cast<int64_t>(localMoeExpertNum), isLayered, config) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "Check tensor shape failed."), return ge::GRAPH_FAILED);

    // 校验UB大小
    if (isSetFullMeshV2) {
        OP_TILING_CHECK(CheckUBSize(context, *tilingData, nodeName) != ge::GRAPH_SUCCESS,
            OP_LOGE(nodeName, "Tiling check UB size failed."), return ge::GRAPH_FAILED);
    }

    // 校验win区大小
    if (!config.isMc2Context) {
        OP_TILING_CHECK(CheckWinSize(context, *tilingData, nodeName, isSetFullMeshV2, localMoeExpertNum, isLayered) != ge::GRAPH_SUCCESS,
            OP_LOGE(nodeName, "Tiling check window size failed."), return ge::GRAPH_FAILED);
    } else {
        auto attrs = context->GetAttrs();
        auto cclBuffSizePtr = attrs->GetAttrPointer<int64_t>(static_cast<int>(config.attrCclBufferSizeIndex));
        OP_TILING_CHECK(cclBuffSizePtr == nullptr || *cclBuffSizePtr < 0 ,
            OP_LOGE(nodeName, "cclBuffSizePtr is invalid."), return ge::GRAPH_FAILED);
        tilingData->moeDistributeDispatchV2Info.totalWinSizeEp = *cclBuffSizePtr;
    }
    OP_TILING_CHECK(SetWorkSpace(context, nodeName) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "Tiling set workspace failed."), return ge::GRAPH_FAILED);
    uint32_t tpWorldSize = tilingData->moeDistributeDispatchV2Info.tpWorldSize;
    if (!config.isMc2Context) {
        OP_TILING_CHECK(SetHcommCfg(context, tilingData, groupEp, groupTp, tpWorldSize, isLayered) != ge::GRAPH_SUCCESS,
            OP_LOGE(nodeName, "Tiling set hcomm cfg failed."), return ge::GRAPH_FAILED);
    }
    uint64_t tilingKey = CalTilingKey(context, isScales, quantMode, tpWorldSize, isSetFullMeshV2, isLayered);
    tilingData->moeDistributeDispatchV2Info.isMc2Context = config.isMc2Context;

    OP_LOGD(nodeName, "tilingKey is %lu", tilingKey);
    context->SetTilingKey(tilingKey);
    uint32_t numBlocks = 1U;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    OP_TILING_CHECK((isLayered && (aivNum != AIV_NUM_93)), OP_LOGE(nodeName, "Layered must be fullCore."), return ge::GRAPH_FAILED);
    uint64_t ubSize = 0UL;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    numBlocks = ascendcPlatform.CalcTschBlockDim(aivNum, 0, aivNum);
    context->SetBlockDim(numBlocks);
    context->SetScheduleMode(1); // 设置为batch mode模式, 所有核同时启动
    tilingData->moeDistributeDispatchV2Info.totalUbSize = ubSize;
    tilingData->moeDistributeDispatchV2Info.aivNum = aivNum;
    OP_LOGD(nodeName, "numBlocks=%u, aivNum=%u, ubSize=%lu", numBlocks, aivNum, ubSize);
    PrintTilingDataInfo(nodeName, *tilingData);
    return ge::GRAPH_SUCCESS;
}

// a2函数
static ge::graphStatus MoeDistributeDispatchA2CheckAttrAndSetTiling(const gert::TilingContext *context, MoeDistributeDispatchA2Info& info, const bool isLayered)
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
    auto quantModePtr = attrs->GetAttrPointer<int64_t>(ATTR_QUANT_MODE_INDEX);
    auto globalBsPtr = attrs->GetAttrPointer<int64_t>(ATTR_GLOBAL_BS_INDEX);
    auto expertTokenNumsTypePtr = attrs->GetAttrPointer<int64_t>(ATTR_EXPERT_TOKEN_NUMS_TYPE_INDEX);
    auto zeroExpertNumPtr = attrs->GetAttrPointer<int64_t>(static_cast<int>(ATTR_ZERO_EXPERT_NUM_INDEX));
    auto copyExpertNumPtr = attrs->GetAttrPointer<int64_t>(static_cast<int>(ATTR_COPY_EXPERT_NUM_INDEX));
    auto constExpertNumPtr = attrs->GetAttrPointer<int64_t>(static_cast<int>(ATTR_CONST_EXPERT_NUM_INDEX));

    const gert::StorageShape *expertIdStorageShape = context->GetInputShape(EXPERT_IDS_INDEX);
    OP_TILING_CHECK(expertIdStorageShape == nullptr, OP_LOGE(K_INNER_DEBUG, "expertIdShape is null."), return GRAPH_FAILED);
    int32_t bs = expertIdStorageShape->GetStorageShape().GetDim(0);

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
    OP_TILING_CHECK(moeExpertNumPtr == nullptr || *moeExpertNumPtr % *epWorldSizePtr != 0 ||
        *moeExpertNumPtr <= 0 || *moeExpertNumPtr > MAX_MOE_EXPERT_NUMS_A2,
        OP_LOGE(K_INNER_DEBUG, "moeExpertNum is invalid."), return GRAPH_FAILED);
    OP_TILING_CHECK(tpWorldSizePtr == nullptr,
        OP_LOGE(K_INNER_DEBUG, "tpWorldSize is null."), return GRAPH_FAILED);
    OP_TILING_CHECK(tpRankIdPtr == nullptr,
        OP_LOGE(K_INNER_DEBUG, "tpRankId is null."), return GRAPH_FAILED);
    OP_TILING_CHECK(expertSharedTypePtr == nullptr,
        OP_LOGE(K_INNER_DEBUG, "expertSharedType is null."), return GRAPH_FAILED);
    OP_TILING_CHECK(sharedExpertRankNumPtr == nullptr,
        OP_LOGE(K_INNER_DEBUG, "sharedExpertRankNum is null."), return GRAPH_FAILED);
    OP_TILING_CHECK(quantModePtr == nullptr || (*quantModePtr != static_cast<uint64_t>(QuantModeA5::NON_QUANT) && *quantModePtr != static_cast<uint64_t>(QuantModeA5::PERTOKEN_DYNAMIC_QUANT)),
        OP_LOGE(K_INNER_DEBUG, "quantMode is invalid."), return GRAPH_FAILED);
    OP_TILING_CHECK(globalBsPtr == nullptr,
        OP_LOGE(K_INNER_DEBUG, "globalBs is null."), return GRAPH_FAILED);
    auto xDesc = context->GetInputDesc(X_INDEX);
    bool isInt4Packed = (xDesc != nullptr) && (xDesc->GetDataType() == ge::DT_INT32);
    OP_TILING_CHECK(isInt4Packed && *globalBsPtr != 0 &&
        (*globalBsPtr % *epWorldSizePtr != 0 || *globalBsPtr < bs * *epWorldSizePtr),
        OP_LOGE(K_INNER_DEBUG,
            "INT32 packed globalBs must be 0 or a multiple of epWorldSize no smaller than bs * epWorldSize."),
        return GRAPH_FAILED);
    OP_TILING_CHECK(expertTokenNumsTypePtr == nullptr || *expertTokenNumsTypePtr < 0 || *expertTokenNumsTypePtr > 1,
        OP_LOGE(K_INNER_DEBUG, "expertTokenNumsType is invalid. Must be 0 or 1. "), return GRAPH_FAILED);
    OP_TILING_CHECK(zeroExpertNumPtr == nullptr, OP_LOGE(K_INNER_DEBUG, "zeroExpertNumPtr is null."),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(copyExpertNumPtr == nullptr, OP_LOGE(K_INNER_DEBUG, "copyExpertNumPtr is null."),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(constExpertNumPtr == nullptr || *constExpertNumPtr != 0,
        OP_LOGE(K_INNER_DEBUG, "constExpertNum is invalid. Must be 0."), return ge::GRAPH_FAILED);

    // 判断是否满足uint32_t及其他限制
    int64_t moeExpertNum = static_cast<int64_t>(*moeExpertNumPtr);
    int64_t zeroExpertNum = *zeroExpertNumPtr;
    int64_t copyExpertNum = *copyExpertNumPtr;
    int64_t constExpertNum = 0LL;
    int64_t zeroComputeExpertNum = zeroExpertNum + copyExpertNum + constExpertNum;

    OP_LOGD(K_INNER_DEBUG, "zeroExpertNum=%ld,copyExpertNum= %ld, constExpertNum=%ld", zeroExpertNum, copyExpertNum,
        constExpertNum);
    OP_TILING_CHECK(zeroComputeExpertNum + moeExpertNum > INT32_MAX,
        OP_LOGE(K_INNER_DEBUG,
        "zeroExpertNum[%ld] + copyExpertNum[%ld] + constExpertNum[%ld] + moeExpertNum[%ld] exceed INT32_MAX.",
         zeroExpertNum, copyExpertNum, constExpertNum, moeExpertNum), return ge::GRAPH_FAILED);

    info.epWorldSize = *epWorldSizePtr;
    info.tpWorldSize = static_cast<uint32_t>(0);
    info.epRankId = *epRankIdPtr;
    info.tpRankId = static_cast<uint32_t>(0);
    info.expertSharedType = static_cast<uint32_t>(0);
    info.sharedExpertRankNum = static_cast<uint32_t>(0);
    info.moeExpertNum = *moeExpertNumPtr;
    info.quantMode = *quantModePtr;
    info.maxMoeExpertNum = MAX_MOE_EXPERT_NUMS_A2;

    if (*globalBsPtr == 0) {
        info.globalBs = *epWorldSizePtr * bs;
    } else {
        info.globalBs = *globalBsPtr;
    }
    info.expertTokenNumsType = *expertTokenNumsTypePtr;
    info.zeroComputeExpertNum = static_cast<int32_t>(zeroComputeExpertNum);

    OP_LOGD(K_INNER_DEBUG, "quantMode=%d", info.quantMode);
    OP_LOGD(K_INNER_DEBUG, "globalBs=%d", info.globalBs);
    OP_LOGD(K_INNER_DEBUG, "expertTokenNumsType=%d", info.expertTokenNumsType);
    OP_LOGD(K_INNER_DEBUG, "expertSharedType=%d", info.expertSharedType);
    OP_LOGD(K_INNER_DEBUG, "sharedExpertRankNum=%d", info.sharedExpertRankNum);
    OP_LOGD(K_INNER_DEBUG, "moeExpertNum=%d", info.moeExpertNum);
    OP_LOGD(K_INNER_DEBUG, "epWorldSize=%d", info.epWorldSize);
    OP_LOGD(K_INNER_DEBUG, "tpWorldSize=%d", info.tpWorldSize);
    OP_LOGD(K_INNER_DEBUG, "epRankId=%d", info.epRankId);
    OP_LOGD(K_INNER_DEBUG, "tpRankId=%d", info.tpRankId);
    OP_LOGD(K_INNER_DEBUG, "zeroComputeExpertNum=%d", info.zeroComputeExpertNum);
    OP_LOGD(K_INNER_DEBUG, "maxMoeExpertNum=%d", info.maxMoeExpertNum);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus MoeDistributeDispatchA2CheckShapeAndSetTiling(const gert::TilingContext *context,
                                                                     MoeDistributeDispatchA2Info &info,
                                                                     bool isLayered)
{
    const char *nodeName = context->GetNodeName();
    const gert::StorageShape *xStorageShape = context->GetInputShape(X_INDEX);
    const gert::StorageShape *expertIdStorageShape = context->GetInputShape(EXPERT_IDS_INDEX);
    const gert::StorageShape *scalesStorageShape = context->GetOptionalInputShape(SCALES_INDEX);
    const gert::StorageShape *xActiveMaskStorageShape = context->GetOptionalInputShape(X_ACTIVE_MASK_INDEX);
    const gert::StorageShape *expertScalesStorageShape = context->GetOptionalInputShape(EXPERT_SCALES_INDEX);
    const gert::StorageShape *elasticInfoStorageShape = context->GetOptionalInputShape(ELASTIC_INFO_INDEX);
    const gert::StorageShape *expandScalesStorageShape = context->GetOutputShape(OUTPUT_EXPAND_SCALES_INDEX);
    auto xDesc = context->GetInputDesc(X_INDEX);
    auto scalesDesc = context->GetOptionalInputDesc(SCALES_INDEX);
    auto expandXDesc = context->GetOutputDesc(OUTPUT_EXPAND_X_INDEX);
    auto dynamicScalesDesc = context->GetOutputDesc(OUTPUT_DYNAMIC_SCALES_INDEX);

    OP_TILING_CHECK(xStorageShape == nullptr,
        OP_LOGE(K_INNER_DEBUG, "xShape is null."), return GRAPH_FAILED);
    OP_TILING_CHECK(expertIdStorageShape == nullptr,
        OP_LOGE(K_INNER_DEBUG, "expertIdShape is null."), return GRAPH_FAILED);
    OP_TILING_CHECK(isLayered && expertScalesStorageShape == nullptr,
        OP_LOGE(K_INNER_DEBUG, "expertScales is null."), return GRAPH_FAILED);
    OP_TILING_CHECK(isLayered && expandScalesStorageShape == nullptr,
        OP_LOGE(K_INNER_DEBUG, "expandScales is null."), return GRAPH_FAILED);
    OP_TILING_CHECK(elasticInfoStorageShape != nullptr,
        OP_LOGE(K_INNER_DEBUG, "current does not support elasticInfo as input"), return GRAPH_FAILED);
    OP_TILING_CHECK(xStorageShape->GetStorageShape().GetDimNum() != TWO_DIMS,
        OP_LOGE(K_INNER_DEBUG, "x dims is invalid."), return GRAPH_FAILED);
    OP_TILING_CHECK(expertIdStorageShape->GetStorageShape().GetDimNum() != TWO_DIMS,
        OP_LOGE(K_INNER_DEBUG, "expertId dims is invalid."), return GRAPH_FAILED);
    OP_LOGD(nodeName, "X dim0 = %ld", xStorageShape->GetStorageShape().GetDim(0));
    OP_LOGD(nodeName, "X dim1 = %ld", xStorageShape->GetStorageShape().GetDim(1));
    OP_LOGD(nodeName, "expertId dim0 = %ld", expertIdStorageShape->GetStorageShape().GetDim(0));
    OP_LOGD(nodeName, "expertId dim1 = %ld", expertIdStorageShape->GetStorageShape().GetDim(1));

    uint32_t h = xStorageShape->GetStorageShape().GetDim(1);
    uint32_t bs = expertIdStorageShape->GetStorageShape().GetDim(0);
    uint32_t k = expertIdStorageShape->GetStorageShape().GetDim(1);
    bool isScales = (scalesStorageShape != nullptr);
    bool isInt4Packed = (xDesc != nullptr) && (xDesc->GetDataType() == ge::DT_INT32);
    auto attrs = context->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE(K_INNER_DEBUG, "attrs is null."), return ge::GRAPH_FAILED);
    auto quantModePtr = attrs->GetAttrPointer<int64_t>(ATTR_QUANT_MODE_INDEX);
    auto epWorldSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_EP_WORLD_SIZE_INDEX);
    auto moeExpertNumPtr = attrs->GetAttrPointer<int64_t>(ATTR_MOE_EXPERT_NUM_INDEX);
    if (isInt4Packed) {
        OP_TILING_CHECK(!isLayered || epWorldSizePtr == nullptr || *epWorldSizePtr != MAX_EP_WORLD_SIZE_A2_LAYERED,
            OP_LOGE(K_INNER_DEBUG, "INT32 packed input only supports A2 EP128 hierarchy."), return GRAPH_FAILED);
        OP_TILING_CHECK(quantModePtr == nullptr || *quantModePtr != static_cast<uint64_t>(QuantModeA5::NON_QUANT),
            OP_LOGE(K_INNER_DEBUG, "INT32 packed input only supports quantMode=0."), return GRAPH_FAILED);
        OP_TILING_CHECK(!isScales || scalesDesc == nullptr || scalesDesc->GetDataType() != ge::DT_FLOAT,
            OP_LOGE(K_INNER_DEBUG, "INT32 packed input requires FP32 scales."), return GRAPH_FAILED);
        OP_TILING_CHECK(expandXDesc == nullptr || dynamicScalesDesc == nullptr ||
            expandXDesc->GetDataType() != ge::DT_INT32 || dynamicScalesDesc->GetDataType() != ge::DT_FLOAT,
            OP_LOGE(K_INNER_DEBUG, "INT32 packed output requires INT32 expandX and FP32 dynamicScales."),
            return GRAPH_FAILED);
        OP_TILING_CHECK(h != INT4_PACKED_H_A2,
            OP_LOGE(K_INNER_DEBUG, "INT32 packed h must be 1024 for logical H=8192, but got %u.", h),
            return GRAPH_FAILED);
        OP_TILING_CHECK(xStorageShape->GetStorageShape().GetDim(0) !=
            expertIdStorageShape->GetStorageShape().GetDim(0),
            OP_LOGE(K_INNER_DEBUG, "INT32 packed x dim0 must equal expertIds dim0."), return GRAPH_FAILED);
        OP_TILING_CHECK(scalesStorageShape->GetStorageShape().GetDimNum() != ONE_DIM ||
            scalesStorageShape->GetStorageShape().GetDim(0) != static_cast<int64_t>(bs),
            OP_LOGE(K_INNER_DEBUG, "INT32 packed scales shape must be [BS]."), return GRAPH_FAILED);
    }
    uint32_t maxHiddenSizeA2 = isLayered ? LAYERED_MAX_HIDDEN_SIZE_A2 : MAX_HIDDEN_SIZE_A2;
    OP_TILING_CHECK(h % BLOCK_SIZE_A2 != 0 || h == 0 || h > maxHiddenSizeA2,
        OP_LOGE(K_INNER_DEBUG, "hiddensize is invalid."), return GRAPH_FAILED);
    uint32_t maxBatchSizeA2 = isLayered ? LAYERED_MAX_BATCH_SIZE_A2 : MAX_BATCH_SIZE_A2;
    OP_TILING_CHECK(bs == 0 || bs > maxBatchSizeA2,
        OP_LOGE(K_INNER_DEBUG, "batchsize is invalid."), return GRAPH_FAILED);

    auto zeroExpertNumPtr = attrs->GetAttrPointer<int64_t>(static_cast<int>(ATTR_ZERO_EXPERT_NUM_INDEX));
    auto copyExpertNumPtr = attrs->GetAttrPointer<int64_t>(static_cast<int>(ATTR_COPY_EXPERT_NUM_INDEX));
    // 判断是否满足uint32_t及其他限制
    int32_t moeExpertNum = *moeExpertNumPtr;
    int32_t zeroExpertNum = static_cast<int32_t>(*zeroExpertNumPtr);
    int32_t copyExpertNum = static_cast<int32_t>(*copyExpertNumPtr);
    int32_t constExpertNum = 0;
    int32_t zeroComputeExpertNum = zeroExpertNum + copyExpertNum + constExpertNum;
    OP_TILING_CHECK(k == 0 || k > MAX_K_VALUE_A2 || k > moeExpertNum + zeroComputeExpertNum,
        OP_LOGE(K_INNER_DEBUG, "k is invalid."), return GRAPH_FAILED);
    OP_TILING_CHECK(!isInt4Packed && *quantModePtr == static_cast<uint64_t>(QuantModeA5::NON_QUANT) && isScales,
        OP_LOGE(K_INNER_DEBUG, "scales should be null when quantMode is unQuant."), return GRAPH_FAILED);

    bool isActiveMask = (xActiveMaskStorageShape != nullptr);
    if (isActiveMask) {
        const int64_t xActiveMaskDimNums = xActiveMaskStorageShape->GetStorageShape().GetDimNum();
        OP_TILING_CHECK(((xActiveMaskDimNums != ONE_DIM) && (xActiveMaskDimNums != TWO_DIMS)),
            OP_LOGE(nodeName, "xActiveMask must be 1-dimension or 2-dimension, but got %lu dim",
            xActiveMaskDimNums), return GRAPH_FAILED);

        int64_t xActiveMaskDim0 = xActiveMaskStorageShape->GetStorageShape().GetDim(0);
        OP_TILING_CHECK(xActiveMaskDim0 != static_cast<int64_t>(bs),
            OP_LOGE(nodeName, "xActiveMask's dim0 not equal to expertIds's dim0, xActiveMask's dim0 is %ld, "
            "expertIds's dim0 is %ld", xActiveMaskDim0, bs), return GRAPH_FAILED);

        OP_TILING_CHECK(((xActiveMaskStorageShape->GetStorageShape().GetDimNum() == TWO_DIMS) &&
            (xActiveMaskStorageShape->GetStorageShape().GetDim(1) != static_cast<int64_t>(k))),
            OP_LOGE(nodeName, "xActiveMask's dim1 not equal to expertIds's dim1, xActiveMask's dim1 is %ld, "
            "expertIds's dim1 is %ld", xActiveMaskStorageShape->GetStorageShape().GetDim(1), k), return GRAPH_FAILED);
    }

    info.isTokenMask = ((isActiveMask) && (xActiveMaskStorageShape->GetStorageShape().GetDimNum() == ONE_DIM));
    info.isExpertMask = ((isActiveMask) && (xActiveMaskStorageShape->GetStorageShape().GetDimNum() == TWO_DIMS));
    info.bs = bs;
    info.k = k;
    info.h = h;

    OP_LOGD(K_INNER_DEBUG, "isTokenMask is %d", static_cast<int32_t>(info.isTokenMask));
    OP_LOGD(K_INNER_DEBUG, "isExpertMask is %d", static_cast<int32_t>(info.isExpertMask));
    OP_LOGD(K_INNER_DEBUG, "batchSize is %u", info.bs);
    OP_LOGD(K_INNER_DEBUG, "k is %u", info.k);
    OP_LOGD(K_INNER_DEBUG, "hiddenSize is %u", info.h);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus MoeDistributeDispatchA2GetPlatformInfoAndSetTiling(const gert::TilingContext *context, MoeDistributeDispatchA2Info& info)
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
// commAlg参数当前支持"fullmesh"和"hierarchy"两种，其余使用默认fullmesh不分层方案。
static ge::graphStatus MoeDistributeDispatchA2CheckCommAlg(const gert::TilingContext *context, bool &isLayered)
{
    isLayered = false;
    auto attrs = context->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE(K_INNER_DEBUG, "attrs is null."), return ge::GRAPH_FAILED);
    auto epWorldSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_EP_WORLD_SIZE_INDEX);
    if ((epWorldSizePtr != nullptr) && (*epWorldSizePtr <= RANK_NUM_PER_NODE_A2)) {
        isLayered = false;
        OP_LOGD(K_INNER_DEBUG, "epWorldSize <= 8, use default fullmesh algorithm.");
        return ge::GRAPH_SUCCESS;
    }
    auto commAlg = attrs->GetAttrPointer<char>(static_cast<int>(ATTR_COMM_ALG_INDEX));
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

static uint64_t MoeDistributeDispatchA2CalcTilingKey(const gert::TilingContext *context, const bool isLayered)
{
    bool tp = false;
    bool scaleMode = false;
    uint32_t commMode = TILINGKEY_TPL_MTE;

    if (isLayered) {
        commMode = TILINGKEY_TPL_AICPU;
    }
    const gert::StorageShape *scalesStorageShape = context->GetOptionalInputShape(SCALES_INDEX);
    bool isScales = (scalesStorageShape != nullptr);
    if (isScales) {
        scaleMode = true;
    }
    uint64_t tilingKey = GET_TPL_TILING_KEY(tp, TILINGKEY_NO_QUANT, scaleMode, 
                                            TILINGKEY_NO_FULLMESH, commMode, TILINGKEY_TPL_A2);
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

static ge::graphStatus MoeDistributeDispatchA2CheckWinSize(const gert::TilingContext *context, const char *nodeName,
    MoeDistributeDispatchA2Info &info, bool isLayered)
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
    auto xDesc = context->GetInputDesc(X_INDEX);
    bool isInt4Packed = (xDesc != nullptr) && (xDesc->GetDataType() == ge::DT_INT32);
    if (isInt4Packed) {
        uint64_t localMoeExpertNumAlign = ops::CeilAlign(static_cast<uint64_t>(localMoeExpertNum), 8UL);
        uint64_t tokenEntrySize = (static_cast<uint64_t>(info.h) + localMoeExpertNumAlign) * sizeof(int32_t) +
            INT4_SCALE_SLOT_SIZE_A2 + ROUTE_ENCODING_SIZE_A2;
        uint64_t maxDedupTokens = ops::CeilDiv(static_cast<uint64_t>(info.globalBs),
            static_cast<uint64_t>(epWorldSize));
        uint64_t rankPartitionSize = DATA_OFFSET_A2 + maxDedupTokens * tokenEntrySize + DATA_FLAG_SIZE_A2;
        minHcclBuffSize = BUFFER_NUM_A2 * (rankPartitionSize * epWorldSize + STATE_SIZE_A2);
        if (minHcclBuffSize > hcclBuffSize) {
            OP_LOGE(nodeName,
                "HCCL_BUFFSIZE is too small for INT32 packed hierarchy, required=%luMB, actual=%luMB, "
                "entry=%luB, maxDedupTokens=%lu.",
                ops::CeilDiv(minHcclBuffSize, MB_SIZE), ops::CeilDiv(hcclBuffSize, MB_SIZE), tokenEntrySize,
                maxDedupTokens);
            return ge::GRAPH_FAILED;
        }
    } else if (isLayered) {
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

static ge::graphStatus MoeDistributeDispatchA2TilingFuncImpl(gert::TilingContext *context)
{
    const char *nodeName = context->GetNodeName();
    OP_LOGI(nodeName, "Enter MoeDistributeDispatchA2 tiling func.");

    // 涉及SyncAll，设置batch mode模式，所有核同时启动 
    uint32_t batch_mode = 1U; 
    auto ret = context->SetScheduleMode(batch_mode); 
    GE_ASSERT_GRAPH_SUCCESS(ret);

    // 1. tilingData
    MoeDistributeDispatchA2TilingData *tilingData = context->GetTilingData<MoeDistributeDispatchA2TilingData>();
    OP_TILING_CHECK(tilingData == nullptr, VECTOR_INNER_ERR_REPORT_TILING(nodeName, "tilingData is nullptr."),
        return ge::GRAPH_FAILED);
    MoeDistributeDispatchA2Info& info = tilingData->moeDistributeDispatchInfo;

    bool isLayered = false;
    OP_TILING_CHECK(MoeDistributeDispatchA2CheckCommAlg(context, isLayered) != ge::GRAPH_SUCCESS,
        VECTOR_INNER_ERR_REPORT_TILING(context->GetNodeName(), "MoeDistributeDispatchA2 CheckCommAlg Failed"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(MoeDistributeDispatchA2CheckShapeAndSetTiling(context, info, isLayered) != ge::GRAPH_SUCCESS,
        VECTOR_INNER_ERR_REPORT_TILING(context->GetNodeName(), "MoeDistributeDispatchA2 CheckShapeAndSetTiling Failed"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(MoeDistributeDispatchA2CheckAttrAndSetTiling(context, info, isLayered) != ge::GRAPH_SUCCESS,
        VECTOR_INNER_ERR_REPORT_TILING(context->GetNodeName(), "MoeDistributeDispatchA2 CheckAttrAndSetTiling Failed"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(MoeDistributeDispatchA2GetPlatformInfoAndSetTiling(context, info) != ge::GRAPH_SUCCESS,
        VECTOR_INNER_ERR_REPORT_TILING(context->GetNodeName(), "MoeDistributeDispatchA2 GetPlatformInfoAndSetTiling Failed"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(MoeDistributeDispatchA2CheckWinSize(context, nodeName, info, isLayered) != ge::GRAPH_SUCCESS,
        VECTOR_INNER_ERR_REPORT_TILING(context->GetNodeName(), "MoeDistributeDispatchA2 CheckWinSize Failed"),
        return ge::GRAPH_FAILED);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint32_t numBlocks = ascendcPlatform.CalcTschBlockDim(aivNum, 0, aivNum);
    context->SetBlockDim(numBlocks);
    uint32_t aicpuBlockDim = info.epWorldSize > RANK_NUM_PER_NODE_A2 ? mc2tiling::AICPU_NUM_BLOCKS_A2 : 1;
    context->SetAicpuBlockDim(aicpuBlockDim);

    uint64_t tilingKey = MoeDistributeDispatchA2CalcTilingKey(context, isLayered);
    context->SetTilingKey(tilingKey);
    // 2. workspace
    size_t *workSpaces = context->GetWorkspaceSizes(1);
    OP_TILING_CHECK(workSpaces == nullptr, VECTOR_INNER_ERR_REPORT_TILING(nodeName, "workSpaces is nullptr."),
        return ge::GRAPH_FAILED);
    workSpaces[0] = SYSTEM_NEED_WORKSPACE + USER_WORKSPACE_A2;

    // 3. communication
    auto attrs = context->GetAttrs();
    auto group = attrs->GetAttrPointer<char>(static_cast<int>(ATTR_GROUP_EP_INDEX));
    auto epWorldSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_EP_WORLD_SIZE_INDEX);
    std::string algConfig = MoeDistributeCombineA2GetAlgConfig(*epWorldSizePtr, isLayered);
    uint32_t opType = 18; // BatchWrite

    AscendC::Mc2CcTilingConfig mc2CcTilingConfig(group, opType, algConfig);
    OP_TILING_CHECK(mc2CcTilingConfig.GetTiling(tilingData->mc2InitTiling) != 0,
        OP_LOGE(nodeName, "mc2CcTilingConfig mc2InitTiling GetTiling failed"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(mc2CcTilingConfig.GetTiling(tilingData->mc2CcTiling) != 0,
        OP_LOGE(nodeName, "mc2CcTilingConfig mc2CcTiling GetTiling failed"), return ge::GRAPH_FAILED);

    OP_LOGI(nodeName, "Leave MoeDistributeDispatchA2 tiling func.");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus MoeDistributeDispatchA5TilingFuncImpl(gert::TilingContext* context)
{
    auto attrs = context->GetAttrs();
    const char *nodeName = context->GetNodeName();
    auto commAlgPtr = attrs->GetAttrPointer<char>(static_cast<int>(ATTR_COMM_ALG_INDEX));
    OP_LOGD(nodeName, "Set 'commAlg' as '%s'", commAlgPtr);
    // 检查 commAlg 参数合法性校验
    bool isNullOrEmpty = (commAlgPtr == nullptr) || (std::strlen(commAlgPtr) == 0);
    bool isFullmeshV1 = (std::strcmp(commAlgPtr, "fullmesh_v1") == 0);
    bool isFullmeshV2 = (std::strcmp(commAlgPtr, "fullmesh_v2") == 0);
    bool isMte = isFullmeshV1 || isFullmeshV2;
    bool isCcu = std::strcmp(commAlgPtr, "ccu") == 0;
    OP_TILING_CHECK(!(isNullOrEmpty || isMte || isCcu),
        OP_LOGE(nodeName, "Invalid parameter: 'commAlg'='%s'. "
            "Only 'fullmesh_v1' and 'fullmesh_v2' are supported. "
            "Nullptr and empty char* are also allowed but will be interpreted as 'fullmesh_v1'.", commAlgPtr),
        return ge::GRAPH_FAILED);
    if (isCcu) {
        // CCU 调用 A5 tiling 实现
        return MoeDistributeDispatchTilingImpl(context, OP_VERSION_2);
    }
    // 默认空指针和空字符走 MTE 方式
    if (isNullOrEmpty) {
        OP_LOGI(nodeName, "Parameter 'commAlg' is nullptr/empty, defaulting to 'fullmesh_v1'.");
    }
    DispatchV2Config config;
    config.xIndex = 0U; // 0: 根据dispatchV2算子原型标志位初始化groupEp索引
    config.expertIdsIndex = 1U; // 1: 根据dispatchV2算子原型标志位初始化expertIds索引
    config.scalesIndex = 2U; // 2: 根据dispatchV2算子原型标志位初始化scales索引
    config.xActiveMaskIndex = 3U; // 3: 根据dispatchV2算子原型标志位初始化xActiveMask索引
    config.expertScalesIndex = 4U; // 4: 根据dispatchV2算子原型标志位初始化expertScales索引
    config.elasticInfoIndex = 5U; // 5: 根据dispatchV2算子原型标志位初始化elasticInfo索引
    config.attrGroupEpIndex = 0; // 0: 根据dispatchV2算子原型标志位初始化groupEp索引
    config.attrEpWorldSizeIndex = 1; // 1: 根据dispatchV2算子原型标志位初始化epWorldSize索引
    config.attrEpRankIdIndex = 2; // 2: 根据dispatchV2算子原型标志位初始化epRankId索引
    config.attrMoeExpertNumIndex = 3;  // 3: 根据dispatchV2算子原型标志位初始化moeExpertNum索引
    config.attrGroupTpIndex = 4; // 4: 根据dispatchV2算子原型标志位初始化groupTp索引
    config.attrTpWorldSizeIndex = 5; // 5: 根据dispatchV2算子原型标志位初始化tpWorldSize索引
    config.attrTpRankIdIndex = 6; // 6: 根据dispatchV2算子原型标志位初始化tpRankId索引
    config.attrExpertSharedTypeIndex = 7; // 7: 根据dispatchV2算子原型标志位初始化expertSharedType索引
    config.attrSharedExpertNumIndex = 8; // 8: 根据dispatchV2算子原型标志位初始化sharedExpertNum索引
    config.attrSharedExpertRankNumIndex = 9; // 9: 根据dispatchV2算子原型标志位初始化sharedExpertRankNum索引
    config.attrQuantModeIndex = 10; // 10: 根据dispatchV2算子原型标志位初始化quantMode索引
    config.attrGlobalBsIndex = 11; // 11: 根据dispatchV2算子原型标志位初始化globalBs索引
    config.attrExpertTokenNumsTypeIndex = 12; // 12: 根据dispatchV2算子原型标志位初始化expertTokenNumType索引
    config.attrCommAlgIndex = 13; // 13: 根据dispatchV2算子原型标志位初始化commAlg索引
    config.attrZeroExpertNumIndex = 14; // 14: 根据dispatchV2算子原型标志位初始化zeroExpertNumIndex索引
    config.attrCopyExpertNumIndex = 15; // 15: 根据dispatchV2算子原型标志位初始化copyExpertNumIndex索引
    config.attrConstExpertNumIndex = 16; // 16: 根据dispatchV2算子原型标志位初始化constExpertNumIndex索引
    config.isMc2Context = false;
    // MTE 调用 A3 tiling 实现
    return MoeDistributeDispatchA3TilingFuncImplPublic(context, config);
}

static ge::graphStatus MoeDistributeDispatchV2TilingFunc(gert::TilingContext* context)
{
    std::string socVersion = mc2tiling::GetSocVersion(context);
    NpuArch npuArch = mc2tiling::GetNpuArch(context);
    ge::graphStatus ret;
    if (socVersion == "Ascend910B") {
        ret = MoeDistributeDispatchA2TilingFuncImpl(context);
    } else if (npuArch == NpuArch::DAV_3510) {
        ret = MoeDistributeDispatchA5TilingFuncImpl(context);
    } else {
        DispatchV2Config config;
        config.xIndex = 0U; // 0: 根据dispatchV2算子原型标志位初始化groupEp索引
        config.expertIdsIndex = 1U; // 1: 根据dispatchV2算子原型标志位初始化expertIds索引
        config.scalesIndex = 2U; // 2: 根据dispatchV2算子原型标志位初始化scales索引
        config.xActiveMaskIndex = 3U; // 3: 根据dispatchV2算子原型标志位初始化xActiveMask索引
        config.expertScalesIndex = 4U; // 4: 根据dispatchV2算子原型标志位初始化expertScales索引
        config.elasticInfoIndex = 5U; // 5: 根据dispatchV2算子原型标志位初始化elasticInfo索引
        config.attrGroupEpIndex = 0; // 0: 根据dispatchV2算子原型标志位初始化groupEp索引
        config.attrEpWorldSizeIndex = 1; // 1: 根据dispatchV2算子原型标志位初始化epWorldSize索引
        config.attrEpRankIdIndex = 2; // 2: 根据dispatchV2算子原型标志位初始化epRankId索引
        config.attrMoeExpertNumIndex = 3;  // 3: 根据dispatchV2算子原型标志位初始化moeExpertNum索引
        config.attrGroupTpIndex = 4; // 4: 根据dispatchV2算子原型标志位初始化groupTp索引
        config.attrTpWorldSizeIndex = 5; // 5: 根据dispatchV2算子原型标志位初始化tpWorldSize索引
        config.attrTpRankIdIndex = 6; // 6: 根据dispatchV2算子原型标志位初始化tpRankId索引
        config.attrExpertSharedTypeIndex = 7; // 7: 根据dispatchV2算子原型标志位初始化expertSharedType索引
        config.attrSharedExpertNumIndex = 8; // 8: 根据dispatchV2算子原型标志位初始化sharedExpertNum索引
        config.attrSharedExpertRankNumIndex = 9; // 9: 根据dispatchV2算子原型标志位初始化sharedExpertRankNum索引
        config.attrQuantModeIndex = 10; // 10: 根据dispatchV2算子原型标志位初始化quantMode索引
        config.attrGlobalBsIndex = 11; // 11: 根据dispatchV2算子原型标志位初始化globalBs索引
        config.attrExpertTokenNumsTypeIndex = 12; // 12: 根据dispatchV2算子原型标志位初始化expertTokenNumType索引
        config.attrCommAlgIndex = 13; // 13: 根据dispatchV2算子原型标志位初始化commAlg索引
        config.attrZeroExpertNumIndex = 14; // 14: 根据dispatchV2算子原型标志位初始化zeroExpertNumIndex索引
        config.attrCopyExpertNumIndex = 15; // 15: 根据dispatchV2算子原型标志位初始化copyExpertNumIndex索引
        config.attrConstExpertNumIndex = 16; // 16: 根据dispatchV2算子原型标志位初始化constExpertNumIndex索引
        config.isMc2Context = false;
        ret = MoeDistributeDispatchA3TilingFuncImplPublic(context, config);
    }
    return ret;
}

struct MoeDistributeDispatchCompileInfo {};
static ge::graphStatus TilingParseForMoeDistributeDispatchV2(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(MoeDistributeDispatchV2)
    .Tiling(MoeDistributeDispatchV2TilingFunc)
    .TilingParse<MoeDistributeDispatchCompileInfo>(TilingParseForMoeDistributeDispatchV2);

#ifdef MC2_EXCEPTION_HANDLER
// Register exception func
inline void MoeDistributeDispatchV2ExceptionImplWrapper(aclrtExceptionInfo *args, void *userdata)
{
    Mc2ExceptionImpl(args, userdata, "MoeDistributeDispatchV2");
}

IMPL_OP(MoeDistributeDispatchV2)
    .ExceptionDumpParseFunc(MoeDistributeDispatchV2ExceptionImplWrapper);
#endif

} // namespace optiling
