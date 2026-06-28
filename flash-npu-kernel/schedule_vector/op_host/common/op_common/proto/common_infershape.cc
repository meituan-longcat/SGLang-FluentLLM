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
#include "proto/common_infershape.h"
#include "util/error_util.h"
#include "util/op_util.h"

namespace opcommon {
bool BroadcastDim(int64_t& dim1, const int64_t dim2) {
  if (dim1 == dim2) {
    return true;
  }
  /* column is dim1, row is dim2, matrix value is broadcast(dim1, dim2)
  dim   0     1    d2
  0     0     0    E
  1     0     1    d2
  d1    E     d1   E
  */
  if ((dim1 != 1) && (dim2 != 1)) {
    string msg = ConcatString(dim1, " and ", dim2, " cannot broadcast!");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT("BroadcastDim", msg);
    return false;
  }
  dim1 = (dim1 == 1) ? dim2 : dim1;

  return true;
}

std::string ShapeCannotBroadcastMsg(const gert::Shape& shape1, const gert::Shape& shape2) {
  std::string res = "shape ";
  res += ToString(shape1);
  res += " and ";
  res += ToString(shape2);
  res += " cannot broadcast!";
  return res;
}

static bool BroadcastShapeToOutShape(const gert::Shape* shape, gert::Shape* shapeOutput) {
  OP_LOGD("BroadcastShapeToOutShape", "start broadcast %s to %s!", ToString(*shape).c_str(),
          ToString(*shapeOutput).c_str());
  const int64_t shapeLen = shape->GetDimNum();
  const int64_t shapeYLen = shapeOutput->GetDimNum();
  if (shapeLen > shapeYLen) {
    shapeOutput->SetDimNum(shapeLen);
    const int64_t lenSub = shapeLen - shapeYLen;
    for (int64_t i = shapeYLen; i > 0; i--) {
      int64_t dim1 = shape->GetDim(lenSub + i - 1);
      const int64_t dim2 = shapeOutput->GetDim(i - 1);
      if (!BroadcastDim(dim1, dim2)) {
        string msg = ConcatString(dim1, " and ", dim2, " cannot broadcast!");
        VECTOR_INFER_SHAPE_INNER_ERR_REPORT("BroadcastShapeToOutShape", msg);
        return false;
      }
      shapeOutput->SetDim(lenSub + i - 1, dim1);
    }
    for (int64_t i = 0; i < lenSub; i++) {
      shapeOutput->SetDim(i, shape->GetDim(i));
    }
  } else {
    const int64_t lenSub = shapeYLen - shapeLen;
    for (int64_t i = 0; i < shapeLen; i++) {
      int64_t dim1 = shapeOutput->GetDim(lenSub + i);
      const int64_t dim2 = shape->GetDim(i);
      if (!BroadcastDim(dim1, dim2)) {
        string msg = ConcatString(dim1, " and ", dim2, " cannot broadcast!");
        VECTOR_INFER_SHAPE_INNER_ERR_REPORT("BroadcastShapeToOutShape", msg);
        return false;
      }
      shapeOutput->SetDim(lenSub + i, dim1);
    }
  }
  return true;
}

bool BroadcastShape(const gert::Shape* in1Shape, const gert::Shape* in2Shape, gert::Shape* outShape) {
  *outShape = *in1Shape;

  OP_CHECK(!BroadcastShapeToOutShape(in2Shape, outShape),
           VECTOR_INFER_SHAPE_INNER_ERR_REPORT("BroadcastShape", ShapeCannotBroadcastMsg(*in2Shape, *in1Shape)),
           return false);
  return true;
}

bool BroadcastShape(const std::vector<const gert::Shape*>& inShapes, gert::Shape* outShape) {
  size_t size = inShapes.size();
  OP_CHECK(size == 0, VECTOR_INFER_SHAPE_INNER_ERR_REPORT("BroadcastShape", "inShapes is empty!"), return false);
  *outShape = *inShapes[0];

  for (size_t i = 1; i < size; i++) {
    OP_CHECK(!BroadcastShapeToOutShape(inShapes[i], outShape),
             VECTOR_INFER_SHAPE_INNER_ERR_REPORT("BroadcastShape", ShapeCannotBroadcastMsg(*inShapes[i], *outShape)),
             return false);
  }

  return true;
}

bool BroadcastShape(const gert::Shape** inShapes, size_t size, gert::Shape* outShape) {
  OP_CHECK(size == 0, VECTOR_INFER_SHAPE_INNER_ERR_REPORT("BroadcastShape", "inShapes is empty!"), return false);
  *outShape = *inShapes[0];

  for (size_t i = 1; i < size; i++) {
    OP_CHECK(!BroadcastShapeToOutShape(inShapes[i], outShape),
             VECTOR_INFER_SHAPE_INNER_ERR_REPORT("BroadcastShape", ShapeCannotBroadcastMsg(*inShapes[i], *outShape)),
             return false);
  }

  return true;
}
} // namespace opcommon