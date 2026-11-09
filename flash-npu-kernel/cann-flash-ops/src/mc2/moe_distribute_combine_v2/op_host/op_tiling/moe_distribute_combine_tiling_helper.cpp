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
 * \file moe_distribute_combine_tiling_helper.cc
 * \brief
 */

#include "moe_distribute_combine_tiling_helper.h"
#include "mc2_log.h"

using namespace ge;

namespace optiling {
inline bool MoeDistributeCombineTilingHelper::CheckInputTensorDim(const gert::TilingContext *context,
                                                                  const char *nodeName)
{
    const gert::StorageShape *expandXStorageShape = context->GetInputShape(EXPAND_X_INDEX);
    OP_TILING_CHECK(expandXStorageShape == nullptr, OP_LOGE(nodeName, "expandX is null."), return false);
    OP_TILING_CHECK(expandXStorageShape->GetStorageShape().GetDimNum() != TWO_DIMS,
                    OP_LOGE(nodeName, "expandX must be 2-dimension, but got %lu dim",
                            expandXStorageShape->GetStorageShape().GetDimNum()),
                    return false);
    OP_LOGD(nodeName, "expandX dim0 = %ld", expandXStorageShape->GetStorageShape().GetDim(0));
    OP_LOGD(nodeName, "expandX dim1 = %ld", expandXStorageShape->GetStorageShape().GetDim(1));

    const gert::StorageShape *expertIdsStorageShape = context->GetInputShape(EXPERT_IDS_INDEX);
    OP_TILING_CHECK(expertIdsStorageShape == nullptr, OP_LOGE(nodeName, "expertIds is null."), return false);
    OP_TILING_CHECK(expertIdsStorageShape->GetStorageShape().GetDimNum() != TWO_DIMS,
                    OP_LOGE(nodeName, "expertIds must be 2-dimension, but got %lu dim",
                            expertIdsStorageShape->GetStorageShape().GetDimNum()),
                    return false);
    OP_LOGD(nodeName, "expertIds dim0 = %ld", expertIdsStorageShape->GetStorageShape().GetDim(0));
    OP_LOGD(nodeName, "expertIds dim1 = %ld", expertIdsStorageShape->GetStorageShape().GetDim(1));

    const gert::StorageShape *expandIdxStorageShape = context->GetInputShape(EXPAND_IDX_INDEX);
    OP_TILING_CHECK(expandIdxStorageShape == nullptr, OP_LOGE(nodeName, "expandIdx is null."), return false);
    OP_TILING_CHECK(expandIdxStorageShape->GetStorageShape().GetDimNum() != ONE_DIM,
                    OP_LOGE(nodeName, "expandIdx must be 1-dimension, but got %lu dim",
                            expandIdxStorageShape->GetStorageShape().GetDimNum()),
                    return false);
    OP_LOGD(nodeName, "expandIdx dim0 = %ld", expandIdxStorageShape->GetStorageShape().GetDim(0));
    
    const gert::StorageShape *sharedExpertX = context->GetOptionalInputShape(SHARED_EXPERT_X_INDEX);
    if (sharedExpertX != nullptr) {
        auto attrs = context->GetAttrs();
        auto sharedExpertRankNumPtr = attrs->GetAttrPointer<int64_t>(ATTR_SHARED_EXPERT_RANK_NUM_INDEX);
        OP_TILING_CHECK(*sharedExpertRankNumPtr != 0, OP_LOGE(nodeName, "sharedExpertX only support input None "\
            "when sharedExpertRankNum is non-zero."), return false);
        OP_TILING_CHECK(((sharedExpertX->GetStorageShape().GetDimNum() != TWO_DIMS) &&
                        (sharedExpertX->GetStorageShape().GetDimNum() != THREE_DIMS)),
                        OP_LOGE(nodeName, "sharedExpertX must be 2-dimension or 3-dimension, but got %lu dim",
                                sharedExpertX->GetStorageShape().GetDimNum()), return false);
    }
    return true;
}

inline bool MoeDistributeCombineTilingHelper::CheckInputSendCountsTensorDim(const gert::TilingContext *context,
                                                                            const char *nodeName)
{
    const gert::StorageShape *epSendCountsStorageShape = context->GetInputShape(EP_SEND_COUNTS_INDEX);
    OP_TILING_CHECK(epSendCountsStorageShape == nullptr, OP_LOGE(nodeName, "epSendCounts is null."), return false);
    OP_TILING_CHECK(epSendCountsStorageShape->GetStorageShape().GetDimNum() != ONE_DIM,
                    OP_LOGE(nodeName, "epSendCounts must be 1-dimension, but got %lu dim",
                            epSendCountsStorageShape->GetStorageShape().GetDimNum()),
                    return false);
    OP_LOGD(nodeName, "epSendCounts dim0 = %ld", epSendCountsStorageShape->GetStorageShape().GetDim(0));

    const gert::StorageShape *tpSendCountsStorageShape = context->GetOptionalInputShape(TP_SEND_COUNTS_INDEX);
    OP_TILING_CHECK(tpSendCountsStorageShape == nullptr, OP_LOGE(nodeName, "tpSendCounts is null."), return false);
    OP_TILING_CHECK(tpSendCountsStorageShape->GetStorageShape().GetDimNum() != ONE_DIM,
                    OP_LOGE(nodeName, "tpSendCounts must be 1-dimension, but got %lu dim",
                            tpSendCountsStorageShape->GetStorageShape().GetDimNum()),
                    return false);
    OP_LOGD(nodeName, "tpSendCounts dim0 = %ld", tpSendCountsStorageShape->GetStorageShape().GetDim(0));
    return true;
}

inline bool MoeDistributeCombineTilingHelper::CheckInputExpertScalesTensorDim(const gert::TilingContext *context,
                                                                              const char *nodeName)
{
    const gert::StorageShape *expertScalesStorageShape = context->GetInputShape(EXPERT_SCALES_INDEX);
    OP_TILING_CHECK(expertScalesStorageShape == nullptr, OP_LOGE(nodeName, "expertScales is null."), return false);
    OP_TILING_CHECK(expertScalesStorageShape->GetStorageShape().GetDimNum() != TWO_DIMS,
                    OP_LOGE(nodeName, "expertScale must be 2-dimension, but got %lu dim",
                            expertScalesStorageShape->GetStorageShape().GetDimNum()),
                    return false);
    OP_LOGD(nodeName, "expertScales dim0 = %ld", expertScalesStorageShape->GetStorageShape().GetDim(0));
    OP_LOGD(nodeName, "expertScales dim1 = %ld", expertScalesStorageShape->GetStorageShape().GetDim(1));
    return true;
}

inline bool MoeDistributeCombineTilingHelper::CheckOutputTensorDim(const gert::TilingContext *context,
                                                                   const char *nodeName)
{
    const gert::StorageShape *xStorageShape = context->GetOutputShape(OUTPUT_X_INDEX);
    OP_TILING_CHECK(xStorageShape == nullptr, OP_LOGE(nodeName, "x is null."), return false);
    OP_TILING_CHECK(
        xStorageShape->GetStorageShape().GetDimNum() != TWO_DIMS,
        OP_LOGE(nodeName, "x must be 2-dimension, but got %lu dim", xStorageShape->GetStorageShape().GetDimNum()),
        return false);
    OP_LOGD(nodeName, "x dim0 = %ld", xStorageShape->GetStorageShape().GetDim(0));
    OP_LOGD(nodeName, "x dim1 = %ld", xStorageShape->GetStorageShape().GetDim(1));
    return true;
}

bool MoeDistributeCombineTilingHelper::CheckTensorDim(gert::TilingContext *context, const char *nodeName)
{
    OP_TILING_CHECK(!CheckInputTensorDim(context, nodeName) || !CheckInputSendCountsTensorDim(context, nodeName) ||
                    !CheckInputExpertScalesTensorDim(context, nodeName) || !CheckOutputTensorDim(context, nodeName),
                    OP_LOGE(nodeName, "Input param shape is invalid."), return false);

    // x_active_mask当前不支持传入
    // A3/A5 v1接口的x_active_mask不支持传入
    if (OpVersionManager::GetInstance().GetVersion() == OP_VERSION_1) {
        const gert::StorageShape *xActiveMaskStorageShape = context->GetOptionalInputShape(X_ACTIVE_MASK_INDEX);
        OP_TILING_CHECK(xActiveMaskStorageShape != nullptr, OP_LOGE(nodeName, "x_active_mask only support input None."),
            return false);
    }

    OP_TILING_CHECK(!CheckOutputTensorDim(context, nodeName), OP_LOGE(nodeName, "Output param shape is invalid."),
                    return false);

    return true;
}

inline bool MoeDistributeCombineTilingHelper::CheckActiveMask(const gert::TilingContext *context, const char *nodeName)
{
    // Check Dim/DType/Format
    const gert::StorageShape *xActiveMaskStorageShape = context->GetOptionalInputShape(X_ACTIVE_MASK_INDEX);
    OP_TILING_CHECK(xActiveMaskStorageShape == nullptr, OP_LOGE(nodeName, "xActiveMaskStorageShape is null."),
                    return false);
    const int64_t xActiveMaskDimNums = xActiveMaskStorageShape->GetStorageShape().GetDimNum();
    OP_TILING_CHECK(xActiveMaskDimNums != ONE_DIM,
                    OP_LOGE(nodeName, "xActiveMask must be 1-dim, but current dim num is %ld.",
                            xActiveMaskDimNums),
                    return false);
    auto xActiveMaskDesc = context->GetOptionalInputDesc(X_ACTIVE_MASK_INDEX);
    OP_TILING_CHECK(xActiveMaskDesc == nullptr, OP_LOGE(nodeName, "xActiveMaskDesc is null."), return false);
    OP_TILING_CHECK(xActiveMaskDesc->GetDataType() != ge::DT_BOOL,
                    OP_LOGE(nodeName, "xActiveMask datatype is invalid, datatype should be bool, but is %s.",
                            Ops::Base::ToString(xActiveMaskDesc->GetDataType()).c_str()),
                    return false);
    OP_TILING_CHECK(static_cast<ge::Format>(ge::GetPrimaryFormat(xActiveMaskDesc->GetStorageFormat())) ==
                        ge::FORMAT_FRACTAL_NZ,
                    OP_LOGE(nodeName, "xActiveMask format is invalid."), return false);

    return true;
}

// 校验数据类型
bool MoeDistributeCombineTilingHelper::CheckTensorDataType(const gert::TilingContext *context, const char *nodeName)
{
    auto expandXDesc = context->GetInputDesc(EXPAND_X_INDEX);
    OP_TILING_CHECK(expandXDesc == nullptr, OP_LOGE(nodeName, "expandxDesc is null."), return false);
    OP_TILING_CHECK((expandXDesc->GetDataType() != ge::DT_BF16) && (expandXDesc->GetDataType() != ge::DT_FLOAT16),
                    OP_LOGE(nodeName, "expandX dataType is invalid, dataType should be bf16 or float16, but is %s",
                            Ops::Base::ToString(expandXDesc->GetDataType()).c_str()), return false);
    auto expertIdsDesc = context->GetInputDesc(EXPERT_IDS_INDEX);
    OP_TILING_CHECK(expertIdsDesc == nullptr, OP_LOGE(nodeName, "expertIdsDesc is null."), return false);
    OP_TILING_CHECK((expertIdsDesc->GetDataType() != ge::DT_INT32),
                    OP_LOGE(nodeName, "expertIds dataType is invalid, dataType should be int32, but is %s",
                            Ops::Base::ToString(expertIdsDesc->GetDataType()).c_str()), return false);
    auto expandIdxDesc = context->GetInputDesc(EXPAND_IDX_INDEX);
    OP_TILING_CHECK(expandIdxDesc == nullptr, OP_LOGE(nodeName, "expandIdxDesc is null."), return false);
    OP_TILING_CHECK((expandIdxDesc->GetDataType() != ge::DT_INT32),
                    OP_LOGE(nodeName, "expandIdx dataType is invalid, dataType should be int32, but is %s",
                            Ops::Base::ToString(expandIdxDesc->GetDataType()).c_str()), return false);
    auto epSendCountsDesc = context->GetInputDesc(EP_SEND_COUNTS_INDEX);
    OP_TILING_CHECK(epSendCountsDesc == nullptr, OP_LOGE(nodeName, "epSendCountsDesc is null."), return false);
    OP_TILING_CHECK((epSendCountsDesc->GetDataType() != ge::DT_INT32),
                    OP_LOGE(nodeName, "epSendCounts dataType is invalid, dataType should be int32, but is %s",
                            Ops::Base::ToString(epSendCountsDesc->GetDataType()).c_str()), return false);
    auto tpSendCountsDesc = context->GetOptionalInputDesc(TP_SEND_COUNTS_INDEX);
    OP_TILING_CHECK(tpSendCountsDesc == nullptr, OP_LOGE(nodeName, "tpSendCountsDesc is null."), return false);
    OP_TILING_CHECK((tpSendCountsDesc->GetDataType() != ge::DT_INT32),
                    OP_LOGE(nodeName, "tpSendCounts dataType is invalid, dataType should be int32, but is %s",
                            Ops::Base::ToString(tpSendCountsDesc->GetDataType()).c_str()), return false);
    auto expertScalesDesc = context->GetInputDesc(EXPERT_SCALES_INDEX);
    OP_TILING_CHECK(expertScalesDesc == nullptr, OP_LOGE(nodeName, "expertScalesDesc is null."), return false);
    OP_TILING_CHECK((expertScalesDesc->GetDataType() != ge::DT_FLOAT),
                    OP_LOGE(nodeName, "expertScales dataType is invalid, dataType should be float, but is %s",
                            Ops::Base::ToString(expertScalesDesc->GetDataType()).c_str()), return false);
    auto sharedExpertXDesc = context->GetOptionalInputDesc(SHARED_EXPERT_X_INDEX);
    if (sharedExpertXDesc != nullptr) {
        OP_TILING_CHECK(sharedExpertXDesc->GetDataType() != expandXDesc->GetDataType(),
            OP_LOGE(nodeName, "sharedExpertX dataType should be the same as expandX dataType, but got sharedExpertX"
            "dataType %s, expandX dataType %s.", Ops::Base::ToString(sharedExpertXDesc->GetDataType()).c_str(),
            Ops::Base::ToString(expandXDesc->GetDataType()).c_str()), return false);
    }
    auto xDesc = context->GetOutputDesc(OUTPUT_X_INDEX);
    OP_TILING_CHECK(xDesc == nullptr, OP_LOGE(nodeName, "xDesc is null."), return false);
    OP_TILING_CHECK((xDesc->GetDataType() != expandXDesc->GetDataType()),
                    OP_LOGE(nodeName, "x dataType is invalid, dataType should be equal expandX dataType %s, but is %s",
                            Ops::Base::ToString(expandXDesc->GetDataType()).c_str(),
                            Ops::Base::ToString(xDesc->GetDataType()).c_str()),
                    return false);
    return true;
}

bool MoeDistributeCombineTilingHelper::CheckTensorFormat(const gert::TilingContext *context, const char *nodeName)
{
    auto expandXDesc = context->GetInputDesc(EXPAND_X_INDEX);
    OP_TILING_CHECK(expandXDesc == nullptr, OP_LOGE(nodeName, "expandxDesc is null."), return false);
    OP_TILING_CHECK(static_cast<ge::Format>(ge::GetPrimaryFormat(expandXDesc->GetStorageFormat())) ==
                        ge::FORMAT_FRACTAL_NZ,
                    OP_LOGE(nodeName, "expandXFormat is invalid"), return false);

    auto expertIdsDesc = context->GetInputDesc(EXPERT_IDS_INDEX);
    OP_TILING_CHECK(expertIdsDesc == nullptr, OP_LOGE(nodeName, "expertIdsDesc is null."), return false);
    OP_TILING_CHECK(static_cast<ge::Format>(ge::GetPrimaryFormat(expertIdsDesc->GetStorageFormat())) ==
                        ge::FORMAT_FRACTAL_NZ,
                    OP_LOGE(nodeName, "expertIdsFormat is invalid"), return false);

    auto expandIdxDesc = context->GetInputDesc(EXPAND_IDX_INDEX);
    OP_TILING_CHECK(expandIdxDesc == nullptr, OP_LOGE(nodeName, "expandIdxDesc is null."), return false);
    OP_TILING_CHECK(static_cast<ge::Format>(ge::GetPrimaryFormat(expandIdxDesc->GetStorageFormat())) ==
                        ge::FORMAT_FRACTAL_NZ,
                    OP_LOGE(nodeName, "expandIdxFormat is invalid"), return false);

    auto epSendCountsDesc = context->GetInputDesc(EP_SEND_COUNTS_INDEX);
    OP_TILING_CHECK(epSendCountsDesc == nullptr, OP_LOGE(nodeName, "epSendCountsDesc is null."), return false);
    OP_TILING_CHECK(static_cast<ge::Format>(ge::GetPrimaryFormat(epSendCountsDesc->GetStorageFormat())) ==
                        ge::FORMAT_FRACTAL_NZ,
                    OP_LOGE(nodeName, "epSendCountsFormat is invalid"), return false);

    auto tpSendCountsDesc = context->GetOptionalInputDesc(TP_SEND_COUNTS_INDEX);
    OP_TILING_CHECK(tpSendCountsDesc == nullptr, OP_LOGE(nodeName, "tpSendCountsDesc is null."), return false);
    OP_TILING_CHECK(static_cast<ge::Format>(ge::GetPrimaryFormat(tpSendCountsDesc->GetStorageFormat())) ==
                        ge::FORMAT_FRACTAL_NZ,
                    OP_LOGE(nodeName, "tpSendCountsFormat is invalid"), return false);

    auto expertScalesDesc = context->GetInputDesc(EXPERT_SCALES_INDEX);
    OP_TILING_CHECK(expertScalesDesc == nullptr, OP_LOGE(nodeName, "expertScalesDesc is null."), return false);
    OP_TILING_CHECK(static_cast<ge::Format>(ge::GetPrimaryFormat(expertScalesDesc->GetStorageFormat())) ==
                        ge::FORMAT_FRACTAL_NZ,
                    OP_LOGE(nodeName, "expertScalesFormat is invalid"), return false);

    auto xDesc = context->GetOutputDesc(OUTPUT_X_INDEX);
    OP_TILING_CHECK(xDesc == nullptr, OP_LOGE(nodeName, "xDesc is null."), return false);
    OP_TILING_CHECK(static_cast<ge::Format>(ge::GetPrimaryFormat(xDesc->GetStorageFormat())) == ge::FORMAT_FRACTAL_NZ,
                    OP_LOGE(nodeName, "xFormat is invalid"), return false);
    return true;
}

ge::graphStatus MoeDistributeCombineTilingHelper::TilingCheckMoeDistributeCombine(gert::TilingContext *context,
                                                                                  const char *nodeName)
{
    // 检查参数shape信息
    OP_TILING_CHECK(!CheckTensorDim(context, nodeName), OP_LOGE(nodeName, "param shape is invalid"),
                    return ge::GRAPH_FAILED);
    // 检查参数dataType信息
    OP_TILING_CHECK(!CheckTensorDataType(context, nodeName), OP_LOGE(nodeName, "param dataType is invalid"),
                    return ge::GRAPH_FAILED);
    // 检查参数format信息
    OP_TILING_CHECK(!CheckTensorFormat(context, nodeName), OP_LOGE(nodeName, "param Format is invalid"),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeDistributeCombineTilingHelper::TilingCheckMoeDistributeCombineA5(gert::TilingContext *context,
    const char *nodeName, const bool isTokenMask)
{
    // 检查参数shape信息
    OP_TILING_CHECK(!CheckTensorDim(context, nodeName), OP_LOGE(nodeName, "param shape is invalid"),
                    return ge::GRAPH_FAILED);
    // 检查参数dataType信息
    OP_TILING_CHECK(!CheckTensorDataType(context, nodeName), OP_LOGE(nodeName, "param dataType is invalid"),
                    return ge::GRAPH_FAILED);
    // 检查参数format信息
    OP_TILING_CHECK(!CheckTensorFormat(context, nodeName), OP_LOGE(nodeName, "param Format is invalid"),
                    return ge::GRAPH_FAILED);
    // 检查ActiveMask信息
    if ((OpVersionManager::GetInstance().GetVersion() != OP_VERSION_1) && isTokenMask) {
        OP_TILING_CHECK(!CheckActiveMask(context, nodeName), OP_LOGE(nodeName, "xActiveMask is invalid."),
        return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling