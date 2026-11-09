/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include "error/ops_error.h"

#ifndef OP_LOGE
#define OP_LOGE(OPS_DESC, ...) OPS_LOG_E(OPS_DESC, __VA_ARGS__)
#endif

#ifndef OP_LOGD
#define OP_LOGD(OPS_DESC, ...) OPS_LOG_D(OPS_DESC, __VA_ARGS__)
#endif

#ifndef OP_LOGI
#define OP_LOGI(OPS_DESC, ...) OPS_LOG_I(OPS_DESC, __VA_ARGS__)
#endif

#ifndef OP_CHECK_IF
#define OP_CHECK_IF(COND, LOG_FUNC, EXPR) OPS_CHECK(COND, LOG_FUNC, EXPR)
#endif
