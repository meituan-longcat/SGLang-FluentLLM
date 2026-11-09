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
 * \file moe_distribute_dispatch_infer_v2.cpp
 * \brief
 */
#include "register/op_impl_registry.h"
#include "mc2_log.h"
#include "platform/platform_info.h"
#include "runtime/rt_external_base.h"
#include "platform/soc_spec.h"

using namespace ge;
namespace ops {

namespace {
enum QuantMode {
    QUANT_MODE_NO_QUANT = 0,
    QUANT_MODE_STATIC = 1,
    QUANT_MODE_PERTOKEN = 2,
    QUANT_MODE_PERGROUP = 3,
    QUANT_MODE_MX = 4,
};
}

static constexpr size_t DIM_ONE = 1UL;
static constexpr size_t DIM_TWO = 2UL;
static constexpr int64_t NEG_ONE = -1;
static constexpr int64_t RANK_NUM_PER_NODE = 8;
static constexpr int64_t ASSIST_INFO_NUM_PER_A = 128;
static constexpr int64_t PER_GROUP_SIZE = 128;
static constexpr int64_t MX_QUANT_SIZE = 32;
static constexpr int64_t NUM_EVEN = 2;
static constexpr int64_t SEND_COUNT_MEMORY_SIZE = 2;

static constexpr size_t DISPATCH_INPUT_X_INDEX = 0;
static constexpr size_t DISPATCH_INPUT_EXPERT_IDS_INDEX = 1;
static constexpr size_t DISPATCH_INPUT_SCALES_IDX_INDEX = 2;
static constexpr size_t DISPATCH_INPUT_EXPERT_SCALES_IDX_INDEX = 4;
static constexpr size_t DISPATCH_INPUT_ELASTIC_INFO_IDX_INDEX = 5;
static constexpr size_t DISPATCH_OUTPUT_EXPAND_X_INDEX = 0;
static constexpr size_t DISPATCH_OUTPUT_DYNAMIC_SCALES_INDEX = 1;
static constexpr size_t DISPATCH_OUTPUT_ASSIST_INFO_IDX_INDEX = 2;
static constexpr size_t DISPATCH_OUTPUT_EXPERT_TOKEN_NUMS_INDEX = 3;
static constexpr size_t DISPATCH_OUTPUT_EP_RECV_COUNTS_INDEX = 4;
static constexpr size_t DISPATCH_OUTPUT_TP_RECV_COUNTS_INDEX = 5;
static constexpr size_t DISPATCH_OUTPUT_EXPAND_SCALES = 6;
static constexpr size_t DISPATCH_INPUT_ATTR_EP_WORLD_SIZE_INDEX = 1;
static constexpr size_t DISPATCH_INPUT_ATTR_EP_RANK_ID_INDEX = 2;
static constexpr size_t DISPATCH_INPUT_ATTR_MOE_EXPERT_NUM_INDEX = 3;
static constexpr size_t DISPATCH_INPUT_ATTR_TP_WORLD_SIZE_INDEX = 5;
static constexpr size_t DISPATCH_INPUT_ATTR_TP_RANK_ID_INDEX = 6;
static constexpr size_t DISPATCH_INPUT_ATTR_EXPERT_SHARD_TYPE_INDEX = 7;
static constexpr size_t DISPATCH_INPUT_ATTR_SHARED_EXPERT_NUM_INDEX = 8;
static constexpr size_t DISPATCH_INPUT_ATTR_SHARED_EXPERT_RANK_NUM_INDEX = 9;
static constexpr size_t DISPATCH_INPUT_ATTR_QUANT_MODE_INDEX = 10;
static constexpr size_t DISPATCH_INPUT_ATTR_GLOBAL_BS_INDEX = 11;
static constexpr size_t DISPATCH_INPUT_ATTR_Y_DTYPE_INDEX = 17;

static constexpr uint32_t VERSION_SIZE = 32;
const std::set<std::string> PLATFORM_A2 = {"Ascend910B"};
const std::set<std::string> NPUARCH_A5 = {std::to_string(static_cast<uint32_t>(NpuArch::DAV_3510))};

static bool IsTargetSocVersionInfershape(const char *nodeName, const std::set<std::string> &targetPlatform)
{
    char versionValVersion[VERSION_SIZE];
    // rtGetSocSpec获取成功返回值是0，获取失败返回非0
    if (rtGetSocSpec("version", "Short_SoC_version", versionValVersion, VERSION_SIZE) != RT_ERROR_NONE) {
        OPS_LOG_E(nodeName, "Cannot get Short_SoC_version info in infershape!");
        return false;
    }
    OPS_LOG_D(nodeName, "(IsTargetSocVersionInfershape)Get Short_SoC_version %s", versionValVersion);
    return (targetPlatform.count(versionValVersion) > 0);
}

static bool IsTargetNpuArchInfershape(const char *nodeName, const std::set<std::string> &targetPlatform) 
{ 
    char versionValNpuArch[VERSION_SIZE]; 
    if (rtGetSocSpec("version", "NpuArch", versionValNpuArch, VERSION_SIZE) != RT_ERROR_NONE) { 
        OPS_LOG_E(nodeName, "Cannot get npuArch info in infershape!"); 
        return false; 
    } 
    OPS_LOG_D(nodeName, "(IsTargetNpuArchInfershape)Get NpuArch %s", versionValNpuArch); 
    return (targetPlatform.count(versionValNpuArch) > 0); 
}

static void InferShapeDynamicScalesA5(gert::Shape *dynamicScalesShape, int64_t quantMode, int64_t a, int64_t h)
{
    if (quantMode == QuantMode::QUANT_MODE_PERGROUP) {
        dynamicScalesShape->SetDimNum(DIM_TWO);
        dynamicScalesShape->SetDim(0U, a);
        dynamicScalesShape->SetDim(1U, (h + PER_GROUP_SIZE - 1) / PER_GROUP_SIZE);
    } else if (quantMode == QuantMode::QUANT_MODE_MX) {
        dynamicScalesShape->SetDimNum(DIM_TWO);
        dynamicScalesShape->SetDim(0U, a);
        dynamicScalesShape->SetDim(1U, ((h + MX_QUANT_SIZE - 1) / MX_QUANT_SIZE + 1) / NUM_EVEN * NUM_EVEN);
    } else {
        dynamicScalesShape->SetDimNum(DIM_ONE);
        dynamicScalesShape->SetDim(0U, a);
    }
}

static ge::DataType InferDataTypeDynamicScales(int64_t quantMode)
{
    ge::DataType dynamicScalesDtype = ge::DT_FLOAT;
    if (quantMode == QuantMode::QUANT_MODE_MX) {
        dynamicScalesDtype = ge::DT_FLOAT8_E8M0;
    }
    return dynamicScalesDtype;
}

static ge::graphStatus InferShapeMoeDistributeDispatchV2(gert::InferShapeContext *context)
{
    if (context == nullptr){
        return ge::GRAPH_FAILED;
    }
    OP_LOGD(context->GetNodeName(), "Begin to do InferShapeMoeDistributeDispatchV2.");
    // 获取输入shape
    const gert::Shape *xShape = context->GetInputShape(DISPATCH_INPUT_X_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, xShape);
    const gert::Shape *expertIdsShape = context->GetInputShape(DISPATCH_INPUT_EXPERT_IDS_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, expertIdsShape);
    const gert::Shape *expertScalesShape = context->GetOptionalInputShape(DISPATCH_INPUT_EXPERT_SCALES_IDX_INDEX);
    const gert::Shape *elasticInfoShape = context->GetOptionalInputShape(DISPATCH_INPUT_ELASTIC_INFO_IDX_INDEX);

    gert::Shape *expandXShape = context->GetOutputShape(DISPATCH_OUTPUT_EXPAND_X_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, expandXShape);
    gert::Shape *dynamicScalesShape = context->GetOutputShape(DISPATCH_OUTPUT_DYNAMIC_SCALES_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, dynamicScalesShape);
    gert::Shape *assistInfoShape = context->GetOutputShape(DISPATCH_OUTPUT_ASSIST_INFO_IDX_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, assistInfoShape);
    gert::Shape *expertTokenNumsShape = context->GetOutputShape(DISPATCH_OUTPUT_EXPERT_TOKEN_NUMS_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, expertTokenNumsShape);
    gert::Shape *epRecvCountShape = context->GetOutputShape(DISPATCH_OUTPUT_EP_RECV_COUNTS_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, epRecvCountShape);
    gert::Shape *tpRecvCountShape = context->GetOutputShape(DISPATCH_OUTPUT_TP_RECV_COUNTS_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, tpRecvCountShape);
    gert::Shape *expandScalesShape = context->GetOutputShape(DISPATCH_OUTPUT_EXPAND_SCALES);
    OPS_CHECK_NULL_WITH_CONTEXT(context, expandScalesShape);

    const auto attrs = context->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT(context, attrs);

    const auto epWorldSize = attrs->GetAttrPointer<int64_t>(DISPATCH_INPUT_ATTR_EP_WORLD_SIZE_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, epWorldSize);

    const auto epRankId = attrs->GetAttrPointer<int64_t>(DISPATCH_INPUT_ATTR_EP_RANK_ID_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, epRankId);

    const auto moeExpertNum = attrs->GetAttrPointer<int64_t>(DISPATCH_INPUT_ATTR_MOE_EXPERT_NUM_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, moeExpertNum);

    const auto tpWorldSize = attrs->GetAttrPointer<int64_t>(DISPATCH_INPUT_ATTR_TP_WORLD_SIZE_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, tpWorldSize);

    const auto tpRankId = attrs->GetAttrPointer<int64_t>(DISPATCH_INPUT_ATTR_TP_RANK_ID_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, tpRankId);

    const auto expertShardType = attrs->GetAttrPointer<int64_t>(DISPATCH_INPUT_ATTR_EXPERT_SHARD_TYPE_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, expertShardType);

    const auto sharedExpertNum = attrs->GetAttrPointer<int64_t>(DISPATCH_INPUT_ATTR_SHARED_EXPERT_NUM_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, sharedExpertNum);

    const auto sharedExpertRankNum = attrs->GetAttrPointer<int64_t>(DISPATCH_INPUT_ATTR_SHARED_EXPERT_RANK_NUM_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, sharedExpertRankNum);

    const auto quantMode = attrs->GetAttrPointer<int64_t>(DISPATCH_INPUT_ATTR_QUANT_MODE_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, quantMode);

    const auto globalBs = attrs->GetAttrPointer<int64_t>(DISPATCH_INPUT_ATTR_GLOBAL_BS_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, globalBs);

    OP_CHECK_IF((*epRankId < 0) || (*epRankId >= *epWorldSize),
        OP_LOGE(context->GetNodeName(), "epRankId shoule be in [0, epWorldSize), but got"
        " epWorldSize: %ld, epRankId: %ld.", *epWorldSize, *epRankId), return ge::GRAPH_FAILED);
    OP_CHECK_IF((*sharedExpertRankNum < 0) || (*sharedExpertRankNum >= *epWorldSize),
        OP_LOGE(context->GetNodeName(), "sharedExpertRankNum shoule be in [0, epWorldSize), but got"
        " epWorldSize: %ld, sharedExpertRankNum: %ld.", *epWorldSize, *sharedExpertRankNum), return ge::GRAPH_FAILED);
    bool isSharedDefault = ((*sharedExpertNum == 1) && (*sharedExpertRankNum == 0));
    bool isNoShared = ((*sharedExpertNum == 0) && (*sharedExpertRankNum == 0));
    bool isValidShared = ((*sharedExpertNum > 0)
                        && ((*sharedExpertRankNum / *sharedExpertNum) > 0)
                        && ((*sharedExpertRankNum % *sharedExpertNum) == 0));
    bool isSharedSettingValid = (isSharedDefault || isNoShared || isValidShared);
    OP_CHECK_IF(!isSharedSettingValid,
        OP_LOGE(context->GetNodeName(), "Shared expert setting invalid, got"
        " sharedExpertRankNum: %ld, sharedExpertNum: %ld.", *sharedExpertRankNum, *sharedExpertNum),
        return ge::GRAPH_FAILED);
    int64_t moeRankNum = *epWorldSize - *sharedExpertRankNum;
    OP_CHECK_IF(moeRankNum <= 0, OP_LOGE(context->GetNodeName(), "moeRankNum(epWorldSize - sharedExpertRankNum)"
        " should be larger than 0, but got %ld.", moeRankNum), return ge::GRAPH_FAILED);

    int64_t bs = ((xShape->GetDimNum() == 1U) ? NEG_ONE : xShape->GetDim(0));
    int64_t h = ((xShape->GetDimNum() == 1U) ? NEG_ONE : xShape->GetDim(1));
    int64_t bsTmp = expertIdsShape->GetDimNum() == 1U ? NEG_ONE : expertIdsShape->GetDim(0);
    int64_t k = ((expertIdsShape->GetDimNum() == 1U) ? NEG_ONE : expertIdsShape->GetDim(1));

    OP_CHECK_IF((bs <= 0) || (h <= 0) || (bsTmp <= 0) || (k <= 0),
        OP_LOGE(context->GetNodeName(), "Input shape of xShape or input shape of expertIdsShape is incorrect, "
        "xShape [%ld, %ld], expertIdsShape [%ld, %ld]", bs, h, bsTmp, k),
        return ge::GRAPH_FAILED);

    int64_t a;
    int64_t localExpertNum;
    int64_t localMoeExpertNum = *moeExpertNum / moeRankNum;
    int64_t globalBsReal = ((*globalBs == 0) ? (bs * *epWorldSize) : *globalBs);
    if (globalBsReal < 0) {
        globalBsReal = -1;
    }

    if (*epRankId < *sharedExpertRankNum) {
        localExpertNum = 1;
        int64_t maxBs = globalBsReal / *epWorldSize;
        int64_t rankNumPerSharedExpert = *sharedExpertRankNum / *sharedExpertNum;
        int64_t maxSharedGroupNum = (*epWorldSize + rankNumPerSharedExpert - 1) / rankNumPerSharedExpert;
        a = maxBs * maxSharedGroupNum;
    } else {
        localExpertNum = localMoeExpertNum;
        a = globalBsReal * std::min(localExpertNum, k);
    }
    if (!IsTargetSocVersionInfershape(context->GetNodeName(), PLATFORM_A2) && elasticInfoShape != nullptr) {
        localExpertNum = std::max(static_cast<int64_t>(1), localMoeExpertNum);
        if ((isSharedDefault) || (isNoShared)) {
            a = globalBsReal * std::min(localExpertNum, k);
        } else { // 除零保护
            int64_t maxBs = globalBsReal / *epWorldSize;
            int64_t rankNumPerSharedExpert = *sharedExpertRankNum / *sharedExpertNum;
            int64_t maxSharedGroupNum = (*epWorldSize + rankNumPerSharedExpert - 1) / rankNumPerSharedExpert;
            a = std::max(maxBs * maxSharedGroupNum, globalBsReal * std::min(localMoeExpertNum, k));
        }
    }
    if (globalBsReal < 0) {
        a = -1;
    }

    expandXShape->SetDimNum(DIM_TWO);
    auto realA = ((*tpWorldSize == 0) ? a : (a * *tpWorldSize));
    expandXShape->SetDim(0U, realA);
    expandXShape->SetDim(1U, h);
    OP_LOGD(context->GetNodeName(), "expandx shape is :%s after infershape.",
        Ops::Base::ToString(*expandXShape).c_str());

    if (IsTargetNpuArchInfershape(context->GetNodeName(), NPUARCH_A5)) {
        InferShapeDynamicScalesA5(dynamicScalesShape, *quantMode, a, h);
    } else {
        dynamicScalesShape->SetDimNum(DIM_ONE);
        dynamicScalesShape->SetDim(0U, realA);
    }
    OP_LOGD(context->GetNodeName(), "dynamicScalesShape shape is :%s after infershape.",
        Ops::Base::ToString(*dynamicScalesShape).c_str());

    assistInfoShape->SetDimNum(DIM_ONE);
    assistInfoShape->SetDim(0U, a * ASSIST_INFO_NUM_PER_A);
    OP_LOGD(context->GetNodeName(), "assistInfoShape shape is :%s after infershape.",
        Ops::Base::ToString(*assistInfoShape).c_str());

    expertTokenNumsShape->SetDimNum(DIM_ONE);
    expertTokenNumsShape->SetDim(0U, localExpertNum);
    OP_LOGD(context->GetNodeName(), "expertTokenNumsShape shape is :%s after infershape.",
        Ops::Base::ToString(*expertTokenNumsShape).c_str());

    epRecvCountShape->SetDimNum(DIM_ONE);
    if (IsTargetSocVersionInfershape(context->GetNodeName(), PLATFORM_A2)) {
        if (expertScalesShape != nullptr) {
            epRecvCountShape->SetDim(0U, *epWorldSize * localExpertNum + globalBsReal * 2 * k * ((*epWorldSize) / RANK_NUM_PER_NODE)); // 2：globalbs * 2kn memory size, to support different bs in ranks
        } else {
            epRecvCountShape->SetDim(0U, *epWorldSize * localExpertNum);
        }
    } else {
        if (*tpWorldSize == DIM_TWO)  {
            epRecvCountShape->SetDim(0U, (*epWorldSize) * localExpertNum * (*tpWorldSize));
        } else if (expertScalesShape != nullptr) {
            epRecvCountShape->SetDim(0U, *epWorldSize * localExpertNum + globalBsReal * SEND_COUNT_MEMORY_SIZE * k * (*epWorldSize) / RANK_NUM_PER_NODE);
        } else {
            epRecvCountShape->SetDim(0U, (*epWorldSize) * localExpertNum);
        }
    }
    OP_LOGD(context->GetNodeName(), "epRecvCountShape shape is :%s after infershape.",
        Ops::Base::ToString(*epRecvCountShape).c_str());

    tpRecvCountShape->SetDimNum(DIM_ONE);
    tpRecvCountShape->SetDim(0U, *tpWorldSize);
    OP_LOGD(context->GetNodeName(), "tpRecvCountShape shape is :%s after infershape.",
        Ops::Base::ToString(*tpRecvCountShape).c_str());

    expandScalesShape->SetDimNum(DIM_ONE);
    expandScalesShape->SetDim(0U, 0);
    if (expertScalesShape != nullptr) {
        expandScalesShape->SetDim(0U, a);
    }
    OP_LOGD(context->GetNodeName(), "expandScalesShape shape is :%s after infershape.",
        Ops::Base::ToString(*expandScalesShape).c_str());
    OP_LOGD(context->GetNodeName(), "End to do InferShapeMoeDistributeDispatchV2.");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckQuantMode(const gert::InferDataTypeContext *context, const int64_t *quantMode, int64_t yDtype)
{
    if (*quantMode == QuantMode::QUANT_MODE_STATIC) {
        OP_CHECK_IF((yDtype != static_cast<int64_t>(ge::DT_INT8)) &&
                    (yDtype != static_cast<int64_t>(ge::DT_HIFLOAT8)),
        OP_LOGE(context->GetNodeName(), "when quantmode is static quant, ydtype must be int8 or hifloat8"),
        return ge::GRAPH_FAILED);
    } else if (*quantMode == QuantMode::QUANT_MODE_PERTOKEN) {
        OP_CHECK_IF((yDtype != static_cast<int64_t>(ge::DT_INT8)) &&
                    (yDtype != static_cast<int64_t>(ge::DT_FLOAT8_E4M3FN)) &&
                    (yDtype != static_cast<int64_t>(ge::DT_FLOAT8_E5M2)),
        OP_LOGE(context->GetNodeName(), "when quantmode is dynamic quant, ydtype must be int8 or "
                "float8_e4m3fn or float8_e5m2"),
        return ge::GRAPH_FAILED);
    } else if ((*quantMode == QuantMode::QUANT_MODE_PERGROUP) ||
                (*quantMode == QuantMode::QUANT_MODE_MX)) {
        OP_CHECK_IF((yDtype != static_cast<int64_t>(ge::DT_FLOAT8_E4M3FN)) &&
                    (yDtype != static_cast<int64_t>(ge::DT_FLOAT8_E5M2)),
        OP_LOGE(context->GetNodeName(), "when quantmode is pergoup or mxfp8 quant, "
                "ydtype must be float8_e4m3fn or float8_e5m2"),
        return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeMoeDistributeDispatchV2(gert::InferDataTypeContext *context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferDataTypeMoeDistributeDispatchV2.");
    auto xDtype = context->GetInputDataType(DISPATCH_INPUT_X_INDEX);
    const auto attrs = context->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const auto quantMode = attrs->GetAttrPointer<int64_t>(DISPATCH_INPUT_ATTR_QUANT_MODE_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, quantMode);
    const auto scalesType = context->GetOptionalInputDataType(DISPATCH_INPUT_SCALES_IDX_INDEX);
    const int64_t *yDtypePtr = nullptr;
    if (IsTargetNpuArchInfershape(context->GetNodeName(), NPUARCH_A5)) {
        yDtypePtr = attrs->GetAttrPointer<int64_t>(DISPATCH_INPUT_ATTR_Y_DTYPE_INDEX);
    }
    bool quantFlag = (scalesType != ge::DT_UNDEFINED) ? true : false;
    OP_LOGD(context->GetNodeName(), "quantFlag id %d.", quantFlag);
    ge::DataType expandXDtype = ge::DT_INT8;
    // Packed INT4 uses an INT32 storage tensor plus an FP32 per-token scale.
    // It is already quantized, so quant_mode remains NO_QUANT even though the
    // scales input is present.  Treating every scales-present A2 invocation as
    // INT8 makes GE reinterpret packed INT4 as INT8 and corrupts the following
    // GroupedMatmul in graph mode.
    if ((xDtype == ge::DT_INT32) &&
        (*quantMode == QuantMode::QUANT_MODE_NO_QUANT)) {
        expandXDtype = ge::DT_INT32;
    } else if (!quantFlag && (*quantMode == QuantMode::QUANT_MODE_NO_QUANT)) {
        expandXDtype = xDtype;
    } else if ((yDtypePtr != nullptr) && (*yDtypePtr != ge::DT_UNDEFINED)) {
        int64_t yDtype = *yDtypePtr;
        OP_LOGD(context->GetNodeName(), "specified y_dtype = %lld.", yDtype);
        OP_CHECK_IF(CheckQuantMode(context, quantMode, yDtype) == ge::GRAPH_FAILED,
                    OP_LOGE(context->GetNodeName(), "CheckQuantMode fail"),
                    return ge::GRAPH_FAILED);
        expandXDtype = static_cast<ge::DataType>(yDtype);
    }
    context->SetOutputDataType(DISPATCH_OUTPUT_EXPAND_X_INDEX, expandXDtype);
    context->SetOutputDataType(DISPATCH_OUTPUT_DYNAMIC_SCALES_INDEX,
                               InferDataTypeDynamicScales(*quantMode));
    context->SetOutputDataType(DISPATCH_OUTPUT_ASSIST_INFO_IDX_INDEX, ge::DT_INT32);
    context->SetOutputDataType(DISPATCH_OUTPUT_EXPERT_TOKEN_NUMS_INDEX, ge::DT_INT64);
    context->SetOutputDataType(DISPATCH_OUTPUT_EP_RECV_COUNTS_INDEX, ge::DT_INT32);
    context->SetOutputDataType(DISPATCH_OUTPUT_TP_RECV_COUNTS_INDEX, ge::DT_INT32);
    OP_LOGD(context->GetNodeName(), "End to do InferDataTypeMoeDistributeDispatchV2.");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(MoeDistributeDispatchV2)
    .InferShape(InferShapeMoeDistributeDispatchV2)
    .InferDataType(InferDataTypeMoeDistributeDispatchV2);
}  // namespace ops
