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
#ifndef OP_API_INC_HCCL_UTIL_H_
#define OP_API_INC_HCCL_UTIL_H_

#include "hccl/hccl_types.h"
#include "common/op_mc2_def.h"

#ifdef __cplusplus
extern "C" {
#endif

namespace op {
#define OP_API_CHECK(cond, exec_expr)          \
    do {                                       \
        if (cond) {                            \
            exec_expr;                         \
        }                                      \
    } while (0)
}

#ifdef __cplusplus
}
#endif
#endif // OP_API_INC_HCCL_UTIL_H_