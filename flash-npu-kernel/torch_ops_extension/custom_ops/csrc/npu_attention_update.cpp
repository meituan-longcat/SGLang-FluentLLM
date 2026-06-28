/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <torch/library.h>
#include "ops_common.h"

namespace custom {
using namespace at_npu::native;

// npu tensor max size
const int SIZE = 8;
const int DIM_0 = 0;
const int DIM_1 = 1;

std::tuple<at::Tensor, at::Tensor> npu_attention_update_npu(at::TensorList lse, at::TensorList local_out, int64_t update_type)
{
    at::SmallVector<int64_t, SIZE> out_size;
    at::SmallVector<int64_t, SIZE> lse_out_size;
    out_size.push_back(local_out[0].size(DIM_0));
    out_size.push_back(local_out[0].size(DIM_1));
    lse_out_size.push_back(lse[0].size(DIM_0));
    int64_t sp = local_out.size();
    at::Tensor out = at::empty(out_size, local_out[0].options().dtype(local_out[0].dtype()));
    at::Tensor lse_out = at::empty(lse_out_size, lse[0].options().dtype(lse[0].dtype()));

    EXEC_NPU_CMD_V1(aclnnAttentionUpdate, lse, local_out, update_type, sp, out, lse_out);

    return std::tuple<at::Tensor, at::Tensor>(out, lse_out);
}

// step3, 为META设备实现前向接口
std::tuple<at::Tensor, at::Tensor> npu_attention_update_meta(at::TensorList lse, at::TensorList local_out, int64_t update_type)
{
    at::SmallVector<int64_t, SIZE> out_size;
    at::SmallVector<int64_t, SIZE> lse_out_size;
    out_size.push_back(local_out[0].size(DIM_0));
    out_size.push_back(local_out[0].size(DIM_1));
    lse_out_size.push_back(lse[0].size(DIM_0));
    int64_t sp = local_out.size();
    at::Tensor out = at::empty(out_size, local_out[0].options().dtype(local_out[0].dtype()));
    at::Tensor lse_out = at::empty(lse_out_size, lse[0].options().dtype(lse[0].dtype()));

    return std::tuple<at::Tensor, at::Tensor>(out, lse_out);
}


// step4, 为NPU设备注册前向实现
TORCH_LIBRARY_IMPL(custom, PrivateUse1, m) {
    m.impl("npu_attention_update", &custom::npu_attention_update_npu);
}

// step5, 为META设备注册前向实现
TORCH_LIBRARY_IMPL(custom, Meta, m) {
    m.impl("npu_attention_update", &custom::npu_attention_update_meta);
}

}