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
 * \file common_infershape.h
 * \brief
 */

#ifndef COMMON_INFERSHAPE_H_
#define COMMON_INFERSHAPE_H_

#include <string>
#include "util/op_log.h"
#include "util/op_common_check.h"
#include "external/exe_graph/runtime/shape.h"
#include "external/exe_graph/runtime/infer_shape_context.h"

namespace opcommon {
bool BroadcastShape(const gert::Shape* in1Shape, const gert::Shape* in2Shape, gert::Shape* outShape);
bool BroadcastShape(const std::vector<const gert::Shape*>& inShapes, gert::Shape* outShape);
bool BroadcastShape(const gert::Shape** inShapes, size_t size, gert::Shape* outShape);
std::string ShapeCannotBroadcastMsg(const gert::Shape& shape1, const gert::Shape& shape2);
} // namespace opcommon

#endif  // COMMON_INFERSHAPE_H_