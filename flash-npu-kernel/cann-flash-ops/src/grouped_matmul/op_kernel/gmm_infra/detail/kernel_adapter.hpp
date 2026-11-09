/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_DETAIL_KERNEL_ADAPTER_HPP
#define CATLASS_DETAIL_KERNEL_ADAPTER_HPP

#include "../../gmm_infra/base_defs.hpp"

#if defined(ENABLE_ASCENDC_DUMP)
#include "../../gmm_infra/debug.hpp"
namespace Catlass {
/// Generic Catlass kernel template
template <class Operator>
CATLASS_GLOBAL void KernelAdapter(typename Operator::Params params, GM_ADDR ptrDump)
{
    Operator op;
    AscendC::InitDump(false, ptrDump, ALL_DUMPSIZE);
    bool mmLayoutMode = true;
    AscendC::SetMMLayoutTransform(mmLayoutMode);
    op(params);
}

template <class Operator>
CATLASS_GLOBAL void KernelAdapter(typename Operator::Params params, uint64_t fftsAddr, GM_ADDR ptrDump)
{
    AscendC::SetSyncBaseAddr(fftsAddr);
    Operator op;
    AscendC::InitDump(false, ptrDump, ALL_DUMPSIZE);
    bool mmLayoutMode = true;
    AscendC::SetMMLayoutTransform(mmLayoutMode);
    op(params);
}
} // namespace Catlass
#else
namespace Catlass {
/// Generic Catlass kernel template
template <class Operator>
CATLASS_GLOBAL void KernelAdapter(typename Operator::Params params)
{
    Operator op;
    bool mmLayoutMode = true;
    AscendC::SetMMLayoutTransform(mmLayoutMode);
    op(params);
}

template <class Operator>
CATLASS_GLOBAL void KernelAdapter(typename Operator::Params params, uint64_t fftsAddr)
{
    AscendC::SetSyncBaseAddr(fftsAddr);
    Operator op;
    bool mmLayoutMode = true;
    AscendC::SetMMLayoutTransform(mmLayoutMode);
    op(params);
}
} // namespace Catlass
#endif
#endif
