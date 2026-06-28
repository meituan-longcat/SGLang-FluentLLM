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
 * \file getcdim.h
 * \brief
 */
#ifndef OPS_COMMON_INC_GETCDIM_H
#define OPS_COMMON_INC_GETCDIM_H

#include <cstdint>
#include "exe_graph/runtime/tiling_context.h"
#include "exe_graph/runtime/infer_shape_context.h"

namespace ops {
  int64_t GetInputCDim(gert::TilingContext *kernel_context, const size_t index);
  int64_t GetOutputCDim(gert::TilingContext *kernel_context, const size_t index);
}  // namespace ops
#endif  // OPS_COMMON_INC_GETCDIM_H