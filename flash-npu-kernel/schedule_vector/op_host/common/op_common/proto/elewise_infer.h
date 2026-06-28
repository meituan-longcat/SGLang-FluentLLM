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
 * \file proto_elewise_infer.h
 * \brief
 */

#ifndef PROTO_ELEWISE_INFER_H_
#define PROTO_ELEWISE_INFER_H_

#include <string>
#include "util/op_log.h"
#include "util/op_common_check.h"
#include "util/op_util.h"
#include "external/exe_graph/runtime/shape.h"
#include "external/exe_graph/runtime/infer_shape_context.h"

namespace opcommon {
constexpr int64_t UNKNOWN_RANK_DIM_VALUE = -2;

inline ge::graphStatus SetUnknownRank(gert::Shape* outputShape) {
  OP_CHECK(outputShape == nullptr, OP_LOGD("SetUnknownRank", "the outputShape is nullptr, return unsuccessful"),
           return ge::GRAPH_FAILED);
  outputShape->SetDimNum(0);
  outputShape->AppendDim(UNKNOWN_RANK_DIM_VALUE);

  OP_LOGD("SetUnknownRank", "set unknown rank = -2, output = %s", ToString(*outputShape).c_str());
  return ge::GRAPH_SUCCESS;
}

/*
 * @brief: check whether the output shape is unknown rank
 * @param [out] outputShape: the output shape ptr
 * @return ge::graphStatus
 */
inline bool IsUnknownRank(const gert::Shape* checkShape) {
  return checkShape->GetDimNum() == 1 && checkShape->GetDim(0) == UNKNOWN_RANK_DIM_VALUE;
}
} // namespace opcommon

#endif  // PROTO_ELEWISE_INFER_H_