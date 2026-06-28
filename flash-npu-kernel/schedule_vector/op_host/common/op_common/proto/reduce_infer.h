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

/*!
 * \file proto_reduce_infer.h
 * \brief
 */

#ifndef PROTO_REDUCE_INFER_H_
#define PROTO_REDUCE_INFER_H_

#include <string>
#include "util/op_log.h"
#include "util/op_common_check.h"
#include "util/op_util.h"
#include "external/exe_graph/runtime/shape.h"
#include "external/exe_graph/runtime/infer_shape_context.h"

namespace opcommon {
template <typename T>
ge::graphStatus ReduceDimsWithKeepDims(const gert::Shape* xShape, const T* axesDims, int32_t axesSize,
                                       gert::Shape* outputShape) {
  const int64_t dimNum = xShape->GetDimNum();
  *outputShape = *xShape;
  for (int32_t i = 0; i < axesSize; i++) {
    OP_CHECK(!IsDimValid(dimNum, axesDims[i]), OP_LOGE("reduce", "axesDims is invalid"), return ge::GRAPH_FAILED);
    const int64_t dim = axesDims[i] < 0 ? axesDims[i] + dimNum : axesDims[i];
    outputShape->SetDim(dim, 1);
  }
  OP_LOGD("ReduceDimsWithKeepDims", "ReduceDimsWithKeepDims is SUCCESS");
  return ge::GRAPH_SUCCESS;
}

template <typename T>
ge::graphStatus ReduceDimsWithoutKeepDims(const gert::Shape* xShape, const T* axesDims, int32_t axesSize,
                                          gert::Shape* outputShape) {
  const int64_t dimNum = xShape->GetDimNum();
  outputShape->SetDimNum(0);
  for (int64_t j = 0; j < dimNum; j++) {
    bool reduceFlag = false;
    for (int32_t i = 0; i < axesSize; i++) {
      OP_CHECK(!IsDimValid(dimNum, axesDims[i]), OP_LOGE("reduce", "axesDims is invalid"), return ge::GRAPH_FAILED);
      const int64_t dim = axesDims[i] < 0 ? axesDims[i] + dimNum : axesDims[i];
      if (dim == j) {
        reduceFlag = true;
        break;
      }
    }
    if (!reduceFlag) {
      outputShape->AppendDim(xShape->GetDim(j));
    }
  }
  OP_LOGD("ReduceDimsWithoutKeepDims", "ReduceDimsWithoutKeepDims is SUCCESS");
  return ge::GRAPH_SUCCESS;
}

template <typename T>
ge::graphStatus ReduceDims(const gert::Shape* xShape, const gert::Tensor* axesTensor, int32_t axesSize,
                           const bool keepDims, gert::Shape* outputShape) {
  const T* axesDims = axesTensor->GetData<T>();
  if (keepDims) {
    return ReduceDimsWithKeepDims<T>(xShape, axesDims, axesSize, outputShape);
  }
  return ReduceDimsWithoutKeepDims<T>(xShape, axesDims, axesSize, outputShape);
}
} // namespace opcommon

#endif  // PROTO_REDUCE_INFER_H_