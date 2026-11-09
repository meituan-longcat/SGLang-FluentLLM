/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */


/*!
 * \file grouped_quant_matmul_info_factory.h
 * \brief
 */
#ifndef GROUPED_QUANT_MATMUL_INFO_FACTORY_H
#define GROUPED_QUANT_MATMUL_INFO_FACTORY_H

#include <pthread.h>
#include <shared_mutex>

#include "grouped_quant_matmul_tiling.h"
namespace optiling {
class GroupedQuantMatmulInfoFactory {
public:
    GroupedQuantMatmulInfoFactory() = default;
    ~GroupedQuantMatmulInfoFactory() = default;

    GQmmInputInfo *Get()
    {
        GQmmInputInfo *ptr = nullptr;
        auto threadId = pthread_self();
        lock_.lock_shared();
        auto it = inst_.find(threadId);
        if (it == inst_.end()) {
            lock_.unlock_shared();
            lock_.lock();
            ptr = &(inst_[threadId]);
            lock_.unlock();
        } else {
            ptr = &(it->second);
            lock_.unlock_shared();
        }

        return ptr;
    }

private:
    std::map<pthread_t, GQmmInputInfo> inst_;
    std::shared_mutex lock_;
};

} // namespace optiling
#endif // GROUPED_QUANT_MATMUL_INFO_FACTORY_H