/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
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
 * \file shape_utils.h
 * \brief
 */
#ifndef OPS_COMMON_INC_SHAPE_UTILS_H
#define OPS_COMMON_INC_SHAPE_UTILS_H
#include <string>
#include <sstream>
#include "exe_graph/runtime/shape.h"
#include "graph/ge_error_codes.h"
#include "graph/types.h"

namespace ops {
const gert::Shape g_vec_1_shape = {1};

/**
 * Ensure that the returned shape is non-scalar.
 * When the dim num of shape is 0, this shape is considered to express a scalar.
 * This function returns the original shape when it receives a non-scalar shape, 
 * and returns the vector shape that returns a {1} when it receives a scalar shape
 * @param in_shape input shape
 * @return non-scalar shape
 */
inline const gert::Shape &EnsureNotScalar(const gert::Shape &in_shape) {
  if (in_shape.IsScalar()) {
    return g_vec_1_shape;
  }
  return in_shape;
}
}  // namespace ops
#endif  // OPS_COMMON_INC_SHAPE_UTILS_H
