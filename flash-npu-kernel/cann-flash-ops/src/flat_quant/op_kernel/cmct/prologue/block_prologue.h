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
 * \file block_prologue.h
 * \brief
 */

#ifndef CMCT_INCLUDE_PROLOGUE_BLOCK_PROLOGUE_H
#define CMCT_INCLUDE_PROLOGUE_BLOCK_PROLOGUE_H
namespace Cmct::Prologue {
template <class DispatchPolicy, class... Args>
class BlockPrologue {
    static_assert(AscendC::Std::always_false_v<DispatchPolicy>, "Could not find an prologue specialization.");
};
}  // namespace Cmct::Prologue

#include "block_prologue_b_antiquant_scmc_nd_kn.h"
#include "block_prologue_b_antiquant_scmc_nd_nk_nz_kn.h"
#endif