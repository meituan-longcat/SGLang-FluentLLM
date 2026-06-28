/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
 * \file error_log.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_TILING_ERROR_LOG_H_
#define OPS_BUILT_IN_OP_TILING_ERROR_LOG_H_

namespace optiling {

#define VECTOR_INNER_ERR_REPORT_TILIING(op_name, err_msg, ...)                                       \
  do {                                                                                               \
    OP_LOGE_WITHOUT_REPORT(op_name, err_msg, ##__VA_ARGS__);                                         \
    REPORT_INNER_ERROR("E89999", "op[%s], " err_msg, get_cstr(get_op_info(op_name)), ##__VA_ARGS__); \
  } while (0)

#define OP_TILING_CHECK(cond, log_func, expr) \
  do {                                        \
    if (cond) {                               \
      log_func;                               \
      expr;                                   \
    }                                         \
  } while (0)
}  // namespace optiling

#endif  // OPS_BUILT_IN_OP_TILING_ERROR_LOG_H_
