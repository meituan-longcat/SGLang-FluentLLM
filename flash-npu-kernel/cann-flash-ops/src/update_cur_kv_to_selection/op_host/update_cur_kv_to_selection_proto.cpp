/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file update_cur_kv_to_selection_proto.cpp
 * \brief UpdateCurKvToSelection InferShape / InferDtype
 *        - selection_k_nope / selection_k_rope：输出 shape/dtype 与对应输入完全一致（原地）
 *        - kernel_done：固定 [1] int32
 */

#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "error/ops_error.h"

using namespace ge;
namespace ops {
constexpr size_t INPUT_IDX_CUR_K_NOPE = 0;
constexpr size_t INPUT_IDX_CUR_K_ROPE = 1;
constexpr size_t INPUT_IDX_SLOTS = 2;
constexpr size_t INPUT_IDX_SEL_K_NOPE = 3;
constexpr size_t INPUT_IDX_SEL_K_ROPE = 4;
constexpr size_t INPUT_IDX_BLOCK_TABLE = 5;
constexpr size_t OUTPUT_IDX_SEL_K_NOPE = 0;
constexpr size_t OUTPUT_IDX_SEL_K_ROPE = 1;
constexpr size_t OUTPUT_IDX_KERNEL_DONE = 2;

static ge::graphStatus InferShape4UpdateCurKvToSelection(gert::InferShapeContext* context)
{
    OPS_LOG_I(context->GetNodeName(), "Begin to do InferShape4UpdateCurKvToSelection.");

    const gert::Shape* selNopeInShape = context->GetInputShape(INPUT_IDX_SEL_K_NOPE);
    OPS_LOG_E_IF_NULL(context, selNopeInShape, return ge::GRAPH_FAILED);
    const gert::Shape* selRopeInShape = context->GetInputShape(INPUT_IDX_SEL_K_ROPE);
    OPS_LOG_E_IF_NULL(context, selRopeInShape, return ge::GRAPH_FAILED);

    auto selNopeOutShape = context->GetOutputShape(OUTPUT_IDX_SEL_K_NOPE);
    OPS_LOG_E_IF_NULL(context, selNopeOutShape, return ge::GRAPH_FAILED);
    selNopeOutShape->SetDimNum(selNopeInShape->GetDimNum());
    for (size_t i = 0; i < selNopeInShape->GetDimNum(); ++i) {
        selNopeOutShape->SetDim(i, selNopeInShape->GetDim(i));
    }

    auto selRopeOutShape = context->GetOutputShape(OUTPUT_IDX_SEL_K_ROPE);
    OPS_LOG_E_IF_NULL(context, selRopeOutShape, return ge::GRAPH_FAILED);
    selRopeOutShape->SetDimNum(selRopeInShape->GetDimNum());
    for (size_t i = 0; i < selRopeInShape->GetDimNum(); ++i) {
        selRopeOutShape->SetDim(i, selRopeInShape->GetDim(i));
    }

    // kernel_done：固定 [1]
    auto kernelDoneShape = context->GetOutputShape(OUTPUT_IDX_KERNEL_DONE);
    OPS_LOG_E_IF_NULL(context, kernelDoneShape, return ge::GRAPH_FAILED);
    kernelDoneShape->SetDimNum(1);
    kernelDoneShape->SetDim(0, 1);

    OPS_LOG_I(context->GetNodeName(), "End to do InferShape4UpdateCurKvToSelection");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDtype4UpdateCurKvToSelection(gert::InferDataTypeContext* context)
{
    OPS_LOG_I(context->GetNodeName(), "InferDtype4UpdateCurKvToSelection enter");
    context->SetOutputDataType(OUTPUT_IDX_SEL_K_NOPE, context->GetInputDataType(INPUT_IDX_SEL_K_NOPE));
    context->SetOutputDataType(OUTPUT_IDX_SEL_K_ROPE, context->GetInputDataType(INPUT_IDX_SEL_K_ROPE));
    context->SetOutputDataType(OUTPUT_IDX_KERNEL_DONE, ge::DT_INT32);
    OPS_LOG_I(context->GetNodeName(), "InferDtype4UpdateCurKvToSelection end");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(UpdateCurKvToSelection)
    .InferShape(InferShape4UpdateCurKvToSelection)
    .InferDataType(InferDtype4UpdateCurKvToSelection);
}  // namespace ops
