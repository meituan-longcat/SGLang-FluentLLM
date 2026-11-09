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
 * \file kv_quant_sparse_flash_attention_infershape.cpp
 * \brief
 */

#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "kv_quant_sparse_flash_attention_compat.h"

using namespace ge;

namespace ops {
constexpr size_t QUERY_INPUT_INDEX = 0;
constexpr size_t KEY_INPUT_INDEX = 1;
constexpr uint32_t LAYOUT_QUERY_ATTR_INDEX = 4;
constexpr uint32_t LAYOUT_KV_ATTR_INDEX = 5;
constexpr uint32_t ROPE_HEAD_DIM_ATTR_INDEX = 12;
constexpr uint32_t RETURN_SOFTMAX_LSE_INDEX = 13;
constexpr uint32_t SOFTMAX_LSE_TND_INDEX = 14;
constexpr uint32_t DIM_INDEX_0 = 0;
constexpr uint32_t DIM_INDEX_1 = 1;
constexpr uint32_t DIM_INDEX_2 = 2;
constexpr uint32_t DIM_INDEX_3 = 3;
constexpr uint32_t DIM_NUM_1 = 1;
constexpr uint32_t DIM_NUM_3 = 3;
constexpr uint32_t DIM_NUM_4 = 4;

constexpr uint32_t OUTPUT_INDEX_0 = 0;
constexpr uint32_t OUTPUT_INDEX_1 = 1;
constexpr uint32_t OUTPUT_INDEX_2 = 2;

static bool IsNonEmptyShape(const gert::Shape *shape)
{
    return shape != nullptr && (shape->GetDimNum() > DIM_NUM_1 ||
        (shape->GetDimNum() == DIM_NUM_1 && shape->GetDim(DIM_INDEX_0) > 0));
}

static bool GetReturnSoftmaxLse(const gert::RuntimeAttrs *attrs, const gert::Shape *softmaxMaxShape,
    const gert::Shape *softmaxSumShape, bool hasLseOutputs)
{
    // TorchAir custom_op can carry output meta before GE calls host infer. Use it as
    // a conservative fallback when RuntimeAttrs misses or defaults the new optional bool.
    if (IsNonEmptyShape(softmaxMaxShape) || IsNonEmptyShape(softmaxSumShape)) {
        return true;
    }

    if (attrs->GetAttrNum() > RETURN_SOFTMAX_LSE_INDEX) {
        const bool *lseFlag = attrs->GetAttrPointer<bool>(RETURN_SOFTMAX_LSE_INDEX);
        if (lseFlag != nullptr) {
            return *lseFlag;
        }
    }
    return hasLseOutputs;
}

ge::graphStatus InferShapeKvQuantSparseFlashAttention(gert::InferShapeContext *context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("KvQuantSparseFlashAttentionLse", "InferShapeContext is nullptr"),
               return ge::GRAPH_FAILED);
    const gert::Shape *queryShape = context->GetInputShape(QUERY_INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, queryShape);
    const gert::Shape *keyShape = context->GetInputShape(KEY_INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, keyShape);
    gert::Shape *attentionOutShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, attentionOutShape);
    bool hasLseOutputs = context->GetComputeNodeOutputNum() > OUTPUT_INDEX_2;
    gert::Shape *softmaxMaxShape = hasLseOutputs ? context->GetOutputShape(OUTPUT_INDEX_1) : nullptr;
    gert::Shape *softmaxSumShape = hasLseOutputs ? context->GetOutputShape(OUTPUT_INDEX_2) : nullptr;
    if (hasLseOutputs) {
        OP_CHECK_NULL_WITH_CONTEXT(context, softmaxMaxShape);
        OP_CHECK_NULL_WITH_CONTEXT(context, softmaxSumShape);
    }
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const char *inputLayoutQueryPtr = attrs->GetAttrPointer<char>(LAYOUT_QUERY_ATTR_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputLayoutQueryPtr);
    std::string inputLayoutQueryPtrStr = std::string(inputLayoutQueryPtr);
    const char *inputLayoutKvPtr = attrs->GetAttrPointer<char>(LAYOUT_KV_ATTR_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputLayoutKvPtr);
    std::string inputLayoutKvPtrStr = std::string(inputLayoutKvPtr);
    const int64_t ropeHeadDim = *attrs->GetAttrPointer<int64_t>(ROPE_HEAD_DIM_ATTR_INDEX);
    bool returnSoftmaxLse = GetReturnSoftmaxLse(attrs, softmaxMaxShape, softmaxSumShape, hasLseOutputs);
    bool softmaxLseTnd = false;
    if (attrs->GetAttrNum() > SOFTMAX_LSE_TND_INDEX) {
        const bool *tndFlag = attrs->GetAttrPointer<bool>(SOFTMAX_LSE_TND_INDEX);
        softmaxLseTnd = tndFlag != nullptr && *tndFlag;
    }
    int64_t kvHeadNum = (inputLayoutKvPtrStr == "TND") ? keyShape->GetDim(DIM_INDEX_1) : keyShape->GetDim(DIM_INDEX_2);

    *attentionOutShape = *queryShape;
    if (inputLayoutQueryPtrStr == "BSND") {
        attentionOutShape->SetDimNum(DIM_NUM_4);
        attentionOutShape->SetDim(DIM_INDEX_0, queryShape->GetDim(DIM_INDEX_0));
        attentionOutShape->SetDim(DIM_INDEX_1, queryShape->GetDim(DIM_INDEX_1));
        attentionOutShape->SetDim(DIM_INDEX_2, queryShape->GetDim(DIM_INDEX_2)); // 2:dim2
        if(queryShape->GetDim(DIM_INDEX_3) != -1){
            attentionOutShape->SetDim(DIM_INDEX_3, queryShape->GetDim(DIM_INDEX_3) - ropeHeadDim); // 3:dim3
        }
    } else { // TND
        attentionOutShape->SetDimNum(DIM_NUM_3);
        attentionOutShape->SetDim(DIM_INDEX_0, queryShape->GetDim(DIM_INDEX_0));
        attentionOutShape->SetDim(DIM_INDEX_1, queryShape->GetDim(DIM_INDEX_1));
        if(queryShape->GetDim(DIM_INDEX_2) != -1){
            attentionOutShape->SetDim(DIM_INDEX_2, queryShape->GetDim(DIM_INDEX_2) - ropeHeadDim); // 2:dim2
        }
    }
    if (!hasLseOutputs) {
        return GRAPH_SUCCESS;
    }
    if (returnSoftmaxLse) {
        if (inputLayoutQueryPtrStr == "TND") {
            softmaxMaxShape->SetDimNum(DIM_NUM_3);
            softmaxMaxShape->SetDim(DIM_INDEX_0, softmaxLseTnd ? queryShape->GetDim(DIM_INDEX_0) : kvHeadNum);
            softmaxMaxShape->SetDim(DIM_INDEX_1, softmaxLseTnd ? queryShape->GetDim(DIM_INDEX_1) :
                                                               queryShape->GetDim(DIM_INDEX_0));
            softmaxMaxShape->SetDim(DIM_INDEX_2, softmaxLseTnd ? 1 :
                                                               queryShape->GetDim(DIM_INDEX_1) / kvHeadNum);

            softmaxSumShape->SetDimNum(DIM_NUM_3);
            softmaxSumShape->SetDim(DIM_INDEX_0, softmaxLseTnd ? queryShape->GetDim(DIM_INDEX_0) : kvHeadNum);
            softmaxSumShape->SetDim(DIM_INDEX_1, softmaxLseTnd ? queryShape->GetDim(DIM_INDEX_1) :
                                                               queryShape->GetDim(DIM_INDEX_0));
            softmaxSumShape->SetDim(DIM_INDEX_2, softmaxLseTnd ? 1 :
                                                               queryShape->GetDim(DIM_INDEX_1) / kvHeadNum);
        } else {
            softmaxMaxShape->SetDimNum(DIM_NUM_4);
            softmaxMaxShape->SetDim(DIM_INDEX_0, queryShape->GetDim(DIM_INDEX_0));
            softmaxMaxShape->SetDim(DIM_INDEX_1, kvHeadNum);
            softmaxMaxShape->SetDim(DIM_INDEX_2, queryShape->GetDim(DIM_INDEX_1));
            softmaxMaxShape->SetDim(DIM_INDEX_3, queryShape->GetDim(DIM_INDEX_2) / kvHeadNum);

            softmaxSumShape->SetDimNum(DIM_NUM_4);
            softmaxSumShape->SetDim(DIM_INDEX_0, queryShape->GetDim(DIM_INDEX_0));
            softmaxSumShape->SetDim(DIM_INDEX_1, kvHeadNum);
            softmaxSumShape->SetDim(DIM_INDEX_2, queryShape->GetDim(DIM_INDEX_1));
            softmaxSumShape->SetDim(DIM_INDEX_3, queryShape->GetDim(DIM_INDEX_2) / kvHeadNum);
        }
    } else {
        softmaxMaxShape->SetDimNum(DIM_NUM_1);
        softmaxMaxShape->SetDim(DIM_INDEX_0, 0);
        softmaxSumShape->SetDimNum(DIM_NUM_1);
        softmaxSumShape->SetDim(DIM_INDEX_0, 0);
    }
    return GRAPH_SUCCESS;
}

ge::graphStatus InferDataTypeKvQuantSparseFlashAttention(gert::InferDataTypeContext *context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("KvQuantSparseFlashAttentionLse", "InferShapeContext is nullptr"),
               return ge::GRAPH_FAILED);
    const auto inputDataType = context->GetInputDataType(QUERY_INPUT_INDEX);
    context->SetOutputDataType(OUTPUT_INDEX_0, inputDataType);
    if (context->GetComputeNodeOutputNum() > OUTPUT_INDEX_1) {
        context->SetOutputDataType(OUTPUT_INDEX_1, ge::DT_FLOAT);
    }
    if (context->GetComputeNodeOutputNum() > OUTPUT_INDEX_2) {
        context->SetOutputDataType(OUTPUT_INDEX_2, ge::DT_FLOAT);
    }
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(KvQuantSparseFlashAttentionLse)
    .InferShape(InferShapeKvQuantSparseFlashAttention)
    .InferDataType(InferDataTypeKvQuantSparseFlashAttention);
} // namespace ops
