/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

#ifndef NPU_FORMAT_CAST_OP_ERROR_CHECK_LOCAL_H_
#define NPU_FORMAT_CAST_OP_ERROR_CHECK_LOCAL_H_

#include "opdev/op_log.h"

#define FORMAT_CAST_CHECK(condition, errorCode, message, ...) \
    do {                                                       \
        if (!(condition)) {                                    \
            OP_LOGE(errorCode, message, ##__VA_ARGS__);        \
            return errorCode;                                  \
        }                                                      \
    } while (false)

#endif // NPU_FORMAT_CAST_OP_ERROR_CHECK_LOCAL_H_
