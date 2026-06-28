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

#include "util/op_log.h"
#include "util/op_common_check.h"
#include "common_infershape.h"
#include "util/error_util.h"
#include "external/op_common/op_dev.h"

namespace opcommon {
ge::graphStatus InferShape4BroadcastOp(gert::InferShapeContext* context) {
  OP_LOGD(context, "op common api : InferShape4BroadcastOp start");
  auto inShape1 = context->GetInputShape(0);
  OPS_CHECK_NULL_WITH_CONTEXT(context, inShape1);
  auto inShape2 = context->GetInputShape(1);
  OPS_CHECK_NULL_WITH_CONTEXT(context, inShape2);
  auto outShape = context->GetOutputShape(0);
  OPS_CHECK_NULL_WITH_CONTEXT(context, outShape);

  OP_CHECK(!BroadcastShape(inShape1, inShape2, outShape),
           VECTOR_INFER_SHAPE_INNER_ERR_REPORT(context, ShapeCannotBroadcastMsg(*inShape2, *inShape1)),
           return ge::GRAPH_FAILED);

  OP_LOGD(context, "op common api : InferShape4BroadcastOp end");
  return ge::GRAPH_SUCCESS;
}
} // namespace opcommon