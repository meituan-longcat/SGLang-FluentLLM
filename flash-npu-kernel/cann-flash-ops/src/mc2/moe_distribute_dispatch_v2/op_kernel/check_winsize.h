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
 * \file check_winsize.h
 * \brief
 */

#ifndef CHECK_WINSIZE_H
#define CHECK_WINSIZE_H

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "kernel_tiling/kernel_tiling.h"
#include "moe_distribute_dispatch_tiling.h"

__aicore__ inline void CheckWindowSize(
    uint64_t tilingWinSizeBytes, uint64_t realWinSizeBytes, AscendC::TPipe* tpipe_, GM_ADDR exceptionAddr)
{
    if (unlikely(realWinSizeBytes < tilingWinSizeBytes)) {
        constexpr uint64_t DATA_SIZE = 256; // 定义数据大小为256字节
        AscendC::GlobalTensor<int32_t> exceptionGlobal;
        exceptionGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(exceptionAddr), DATA_SIZE);
        AscendC::TBuf<AscendC::TPosition::VECCALC> exceptionBuf;
        tpipe_->InitBuffer(exceptionBuf, 1); // 初始化一个缓冲区
        AscendC::LocalTensor<int32_t> exceptionLocal = exceptionBuf.Get<int32_t>();
        AscendC::DataCopy(exceptionLocal[1], exceptionGlobal, 1); // 从全局地址复制数据到本地地址
    }
}
#endif // CHECK_WINSIZE_H