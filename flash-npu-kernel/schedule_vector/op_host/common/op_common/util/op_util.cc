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

#include <string>
#include <vector>
#include "op_util.h"
#include <graph/utils/type_utils.h>

namespace opcommon {
std::string ToString(const ge::DataType& type) {
  return ge::TypeUtils::DataTypeToSerialString(type);
}

/*
 * @brief: get format string from enum
 * @param [in] format: enum format
 * @return string: format string
 */
std::string ToString(const ge::Format& format) {
  return ge::TypeUtils::FormatToSerialString(format);
}

std::vector<int64_t> ToVector(const gert::Shape& shape) {
  size_t shapeSize = shape.GetDimNum();
  std::vector<int64_t> shapeVec(shapeSize, 0);

  for (size_t i = 0; i < shapeSize; i++) {
    shapeVec[i] = shape.GetDim(i);
  }
  return shapeVec;
}

std::string ToString(const gert::Shape& shape) {
  return DebugString(ToVector(shape));
}

std::string ToString(const gert::Shape* shape) {
  return DebugString(ToVector(*shape));
}

std::string ToString(const std::vector<int64_t>& shape) {
  return DebugString(shape);
}

std::string ToString(const std::vector<gert::Shape>& shapes) {
  std::string str = "[";
  for (gert::Shape shape : shapes) {
    str += ToString(shape);
    str += ", ";
  }
  str += "]";
  return str;
}
} // namespace opcommon