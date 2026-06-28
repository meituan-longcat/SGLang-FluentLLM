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
 * \file op_common_check.h
 * \brief
 */

#ifndef OP_COMMON_CHECK_H_
#define OP_COMMON_CHECK_H_

#include "op_log.h"
#include "util/error_manager/error_manager.h"
#include "runtime/infer_shape_context.h"

namespace opcommon {
#define OP_CHECK(cond, log_func, return_expr) \
  if (cond) {                                 \
    log_func;                                 \
    return_expr;                              \
  }

#define OPS_CHECK_NULL_WITH_CONTEXT(context, ptr)                                                \
  if ((ptr) == nullptr) {                                                                        \
    const char* name = ((context)->GetNodeName() == nullptr) ? "nil" : (context)->GetNodeName(); \
    OP_LOGE(name, "%s is nullptr!", #ptr);                                        \
    REPORT_CALL_ERROR("EZ9999", "op[%s], %s is nullptr!", name, #ptr);                           \
    return ge::GRAPH_FAILED;                                                                     \
  }
} // namespace opcommon

#endif // OP_COMMON_CHECK_H_