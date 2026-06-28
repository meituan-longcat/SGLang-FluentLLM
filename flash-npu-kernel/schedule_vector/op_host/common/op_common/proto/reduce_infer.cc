/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
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

#include "reduce_infer.h"
#include "util/op_common_check.h"
#include "util/error_util.h"
#include "external/op_common/op_dev.h"

namespace opcommon {
ge::graphStatus InferShape4ReduceOp(gert::InferShapeContext* context) {
  OP_LOGD(context, "op common api : InferShape4ReduceOp start");
  auto inShape = context->GetInputShape(0);
  OPS_CHECK_NULL_WITH_CONTEXT(context, inShape);
  auto axesTensor = context->GetInputTensor(1);
  OPS_CHECK_NULL_WITH_CONTEXT(context, axesTensor);
  auto outShape = context->GetOutputShape(0);
  OPS_CHECK_NULL_WITH_CONTEXT(context, outShape);
  auto attrs = context->GetAttrs();
  OPS_CHECK_NULL_WITH_CONTEXT(context, attrs);

  const bool* keepDims = attrs->GetAttrPointer<bool>(0);
  OPS_CHECK_NULL_WITH_CONTEXT(context, keepDims);

  const int32_t axesSize = static_cast<int32_t>(axesTensor->GetShapeSize());

  OP_CHECK(axesSize < 0,
           VECTOR_INFER_SHAPE_INNER_ERR_REPORT(context, "axes num cannot be less than 0!"),
           return ge::GRAPH_FAILED);

  if (axesSize == 0) {
    *outShape = *inShape;
    OP_LOGD(context, "axes is empty tensor, will ignore infer, set output shape = input shape");
    return ge::GRAPH_SUCCESS;
  }

  const auto dtype = axesTensor->GetDataType();
  OP_CHECK(dtype != ge::DT_INT32 && dtype != ge::DT_INT64,
           VECTOR_INFER_SHAPE_INNER_ERR_REPORT(
               context, ConcatString("axes datatype ", ToString(dtype), " must in (int32, int64)")),
           return ge::GRAPH_FAILED);
  if (dtype == ge::DT_INT32) {
    return ReduceDims<int32_t>(inShape, axesTensor, axesSize, *keepDims, outShape);
  }
  return ReduceDims<int64_t>(inShape, axesTensor, axesSize, *keepDims, outShape);
}
} // namespace opcommon