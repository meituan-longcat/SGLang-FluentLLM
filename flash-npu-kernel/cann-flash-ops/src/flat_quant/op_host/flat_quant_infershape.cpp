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
 * \file flat_quant_infershape.cpp
 * \brief
 */
#include <cmath>
#include <limits>
#include "register/op_impl_registry.h"
#include "error/ops_error.h"

// Compatibility aliases for the host implementation.
#define OPS_LOGE OPS_LOG_E
#define OPS_LOGD OPS_LOG_D

using namespace ge;
namespace ops {
static const int32_t DTYPE_FLOAT4_E2M1 = 40;
static constexpr size_t FLATQUANT_K_IDX = 0;
static constexpr size_t FLATQUANT_M_IDX = 1;
static constexpr size_t FLATQUANT_N_IDX = 2;
static constexpr size_t FLATQUANT_ATTRS_NUM_TWO = 2;
static constexpr size_t FLATQUANT_ATTRS_NUM_THREE = 3;
static constexpr size_t FLATQUANT_MX_N_OUT_DIM = 2;
static constexpr size_t FLATQUANT_N_IS_EVEN = 2;
static constexpr size_t FLATQUANT_DOUBLE_CEIL_SIZE = 64;
static const size_t ATTR_INDEX_OF_DST_DTYPE = 1;
static const size_t ATTR_INDEX_OF_DST_TYPE_MAX = 2;
constexpr int64_t DIGIT_TWO = 2;
constexpr float ZERO_FLOAT = 0.0f;
constexpr float SIX_FLOAT = 6.0f;
constexpr float TWELVE_FLOAT = 12.0f;

static bool IsFloatEqual(float a, float b) { return std::abs(a - b) <= std::numeric_limits<float>::epsilon(); }

static bool IsFlatQuantMxFp4DavidSupport()
{
    // Fixed by BUILD_ARCH: only A5 (Ascend950) supports FLOAT4_E2M1/FLOAT8_E8M0.
#if defined(A5_COMPATIBLE)
    return true;
#else
    return false;
#endif
}

static ge::graphStatus InferShape4FlatQuant(gert::InferShapeContext* context)
{
    if (context == nullptr) {
        return GRAPH_FAILED;
    }
    const gert::Shape* xShape = context->GetInputShape(0);
    gert::Shape* outShape = context->GetOutputShape(0);
    gert::Shape* qScaleShape = context->GetOutputShape(1);
    if ((xShape == nullptr) || (outShape == nullptr) || (qScaleShape == nullptr)) {
        return ge::GRAPH_FAILED;
    }
    OPS_LOGD(context->GetNodeName(), "FlatQuantInferShape begin");
    auto* attrs = context->GetAttrs();
    if (attrs == nullptr) {
        return GRAPH_FAILED;
    }

    if (IsFlatQuantMxFp4DavidSupport()) { // Only for 950
        // Validate dst_type_max parameter if provided
        const float* dstTypeMax = nullptr;
        if (attrs->GetAttrNum() >= FLATQUANT_ATTRS_NUM_THREE) {
            dstTypeMax = attrs->GetAttrPointer<float>(ATTR_INDEX_OF_DST_TYPE_MAX);
        }
        if (dstTypeMax != nullptr) {
            OPS_ERR_IF(
                (!IsFloatEqual(*dstTypeMax, ZERO_FLOAT)) && (*dstTypeMax < SIX_FLOAT || *dstTypeMax > TWELVE_FLOAT),
                OPS_LOGE(context->GetNodeName(), "The dst_type_max[%f] must be 0 or in range [6, 12].", *dstTypeMax),
                return GRAPH_FAILED);
        }

        const int32_t* outxDtype = nullptr;
        if (attrs->GetAttrNum() >= FLATQUANT_ATTRS_NUM_TWO) {
            outxDtype = attrs->GetAttrPointer<int32_t>(ATTR_INDEX_OF_DST_DTYPE);
        }
        OPS_ERR_IF(xShape->GetDim(FLATQUANT_N_IDX) % FLATQUANT_N_IS_EVEN == 1,
                    OPS_LOGE(context->GetNodeName(), "dim N must be even number"), return ge::GRAPH_FAILED);

        if (outxDtype != nullptr) {
            int32_t dstDtype = *outxDtype;
            if (dstDtype == DTYPE_FLOAT4_E2M1) { // dst_dtype 为40
                size_t ceilMN = (xShape->GetDim(FLATQUANT_M_IDX) * xShape->GetDim(FLATQUANT_N_IDX) +
                                 FLATQUANT_DOUBLE_CEIL_SIZE - 1) /
                                FLATQUANT_DOUBLE_CEIL_SIZE;
                outShape->SetDimNum(0);
                outShape->AppendDim(xShape->GetDim(FLATQUANT_K_IDX));
                outShape->AppendDim(xShape->GetDim(FLATQUANT_M_IDX) * xShape->GetDim(FLATQUANT_N_IDX) / DIGIT_TWO);

                qScaleShape->SetDimNum(0);
                qScaleShape->AppendDim(xShape->GetDim(FLATQUANT_K_IDX));
                qScaleShape->AppendDim(ceilMN);
                qScaleShape->AppendDim(FLATQUANT_MX_N_OUT_DIM);
            } else { // 默认值
                outShape->SetDimNum(0);
                outShape->AppendDim(xShape->GetDim(FLATQUANT_K_IDX));
                outShape->AppendDim(xShape->GetDim(FLATQUANT_M_IDX));
                outShape->AppendDim(xShape->GetDim(FLATQUANT_N_IDX));

                qScaleShape->SetDimNum(0);
                qScaleShape->AppendDim(xShape->GetDim(FLATQUANT_K_IDX));
            }
        } else {
            outShape->SetDimNum(0);
            outShape->AppendDim(xShape->GetDim(FLATQUANT_K_IDX));
            outShape->AppendDim(xShape->GetDim(FLATQUANT_M_IDX));
            outShape->AppendDim(xShape->GetDim(FLATQUANT_N_IDX));

            qScaleShape->SetDimNum(0);
            qScaleShape->AppendDim(xShape->GetDim(FLATQUANT_K_IDX));
        }
    } else { // For 910B
        outShape->SetDimNum(0);
        outShape->AppendDim(xShape->GetDim(FLATQUANT_K_IDX));
        outShape->AppendDim(xShape->GetDim(FLATQUANT_M_IDX));
        outShape->AppendDim(xShape->GetDim(FLATQUANT_N_IDX));
        qScaleShape->SetDimNum(0);
        qScaleShape->AppendDim(xShape->GetDim(FLATQUANT_K_IDX));
    }

    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType4FlatQuant(gert::InferDataTypeContext* context)
{
    if (context == nullptr) {
        return GRAPH_FAILED;
    }
    OPS_LOGD(context->GetNodeName(), "FlatQuantInferDataType begin");
    auto* attrs = context->GetAttrs();
    if (attrs == nullptr) {
        return GRAPH_FAILED;
    }

    if (IsFlatQuantMxFp4DavidSupport()) { // For 950
        const int32_t* outxDtype = nullptr;
        if (attrs->GetAttrNum() >= FLATQUANT_ATTRS_NUM_TWO) {
            outxDtype = attrs->GetAttrPointer<int32_t>(ATTR_INDEX_OF_DST_DTYPE);
        }
        if (outxDtype != nullptr) {
            int32_t dstDtype = *outxDtype;
            if (dstDtype == DTYPE_FLOAT4_E2M1) {
                context->SetOutputDataType(0, ge::DT_FLOAT4_E2M1);
                context->SetOutputDataType(1, ge::DT_FLOAT8_E8M0);
            } else {
                context->SetOutputDataType(0, ge::DT_INT4);
                context->SetOutputDataType(1, ge::DT_FLOAT);
            }
        } else {
            context->SetOutputDataType(0, ge::DT_INT4);
            context->SetOutputDataType(1, ge::DT_FLOAT);
        }
    } else { // For 910B
        context->SetOutputDataType(0, ge::DT_INT4);
        context->SetOutputDataType(1, ge::DT_FLOAT);
    }

    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(FlatQuant).InferShape(InferShape4FlatQuant).InferDataType(InferDataType4FlatQuant);
} // namespace ops
