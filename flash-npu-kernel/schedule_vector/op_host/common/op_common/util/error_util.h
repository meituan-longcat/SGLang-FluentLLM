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
 * \file error_util.h
 * \brief
 */

#ifndef ERROR_UTIL_H_
#define ERROR_UTIL_H_

#include "op_log.h"
#include "util/error_manager/error_manager.h"
#include "op_util.h"
#include "external/op_common/op_error_code.h"

namespace opcommon {
#define VECTOR_INFER_SHAPE_INNER_ERR_REPORT(opName, errMsg)                                  \
  do {                                                                                       \
    OP_LOGE(opName, "%s", GetCstr(errMsg));                                                  \
    REPORT_INNER_ERROR(GetViewErrorCodeStr(ViewErrorCode::VECTOR_INNER_ERROR).c_str(), "%s", \
                       ConcatString("op[", opName, "], ", errMsg).c_str());                  \
  } while (0)

std::string GetViewErrorCodeStr(ViewErrorCode errCode);
} // namespace opcommon

#endif // ERROR_UTIL_H_