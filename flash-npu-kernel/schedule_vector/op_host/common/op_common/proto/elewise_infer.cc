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

#include "elewise_infer.h"
#include "util/op_common_check.h"
#include "util/error_util.h"
#include "util/op_util.h"
#include "external/op_common/op_dev.h"

namespace opcommon {
ge::graphStatus InferShape4ElewiseOp(gert::InferShapeContext* context) {
  OP_LOGD(context, "op common api : InferShape4ElewiseOp start");
  auto inShape = context->GetInputShape(0);
  OPS_CHECK_NULL_WITH_CONTEXT(context, inShape);
  auto outShape = context->GetOutputShape(0);
  OPS_CHECK_NULL_WITH_CONTEXT(context, outShape);

  if (IsUnknownRank(inShape)) {
    OP_LOGD(context, "input shape is UnknownRank, set output shape to (-2, )");
    return SetUnknownRank(outShape);
  }

  *outShape = *inShape;
  OP_LOGD(context, "op common api : InferShape4ElewiseOp end");
  return ge::GRAPH_SUCCESS;
}
} // namespace opcommon
