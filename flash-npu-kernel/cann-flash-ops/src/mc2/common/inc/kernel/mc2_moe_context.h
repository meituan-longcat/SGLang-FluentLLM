/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file mc2_moe_context.h
 * \brief
 */

#ifndef MC2_MOE_CONTEXT_H
#define MC2_MOE_CONTEXT_H

struct Mc2MoeContext {
    uint64_t epRankId;
    uint64_t kfcContextAddr; // host kfc方案中，需要传递通信API所需的地址
    uint64_t epHcclBuffer_[1024];
};

#endif //MC2_MOE_CONTEXT_H