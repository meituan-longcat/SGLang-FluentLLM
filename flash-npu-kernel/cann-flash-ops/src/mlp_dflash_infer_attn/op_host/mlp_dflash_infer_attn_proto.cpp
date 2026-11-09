/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>

#include "error/ops_error.h"

namespace ops {
namespace {
constexpr size_t QUERY_INDEX = 0;
constexpr size_t ATTENTION_OUT_INDEX = 0;
constexpr size_t SOFTMAX_LSE_INDEX = 1;
constexpr size_t NUM_HEADS_ATTR_INDEX = 0;
constexpr size_t SOFTMAX_LSE_FLAG_ATTR_INDEX = 7;
}  // namespace

static ge::graphStatus InferShapeMlpDflashInferAttn(gert::InferShapeContext* context)
{
    OPS_ERR_IF(context == nullptr,
        OPS_LOG_E("MlpDflashInferAttn", "InferShapeContext is null."), return ge::GRAPH_FAILED);
    const gert::Shape* queryShape = context->GetInputShape(QUERY_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, queryShape);
    OPS_ERR_IF(queryShape->GetDimNum() != 3,
        OPS_LOG_E(context->GetNodeName(), "query must be TND (3-D), but got %zu-D.", queryShape->GetDimNum()),
        return ge::GRAPH_FAILED);

    gert::Shape* attentionOutShape = context->GetOutputShape(ATTENTION_OUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, attentionOutShape);
    *attentionOutShape = *queryShape;

    gert::Shape* softmaxLseShape = context->GetOutputShape(SOFTMAX_LSE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, softmaxLseShape);
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const bool* lseFlag = attrs->GetAttrPointer<bool>(SOFTMAX_LSE_FLAG_ATTR_INDEX);
    const int64_t* numHeads = attrs->GetAttrPointer<int64_t>(NUM_HEADS_ATTR_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, lseFlag);
    OP_CHECK_NULL_WITH_CONTEXT(context, numHeads);
    if (*lseFlag) {
        *softmaxLseShape = gert::Shape({queryShape->GetDim(0), *numHeads, 1});
    } else {
        *softmaxLseShape = gert::Shape({0});
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeMlpDflashInferAttn(gert::InferDataTypeContext* context)
{
    OPS_ERR_IF(context == nullptr,
        OPS_LOG_E("MlpDflashInferAttn", "InferDataTypeContext is null."), return ge::GRAPH_FAILED);
    context->SetOutputDataType(ATTENTION_OUT_INDEX, context->GetInputDataType(QUERY_INDEX));
    context->SetOutputDataType(SOFTMAX_LSE_INDEX, ge::DT_FLOAT);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(MlpDflashInferAttn)
    .InferShape(InferShapeMlpDflashInferAttn)
    .InferDataType(InferDataTypeMlpDflashInferAttn);
}  // namespace ops
