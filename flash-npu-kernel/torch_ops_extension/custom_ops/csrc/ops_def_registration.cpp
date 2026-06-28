/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <torch/extension.h>
#include <torch/library.h>

// 在custom命名空间里注册add_custom和npu_sparse_flash_attention和后续的XXX算子，每次新增自定义aten ir都需先增加定义
// step1, 为新增自定义算子添加定义
TORCH_LIBRARY(custom, m) {
    m.def("npu_attention_update(Tensor[] lse, Tensor[] local_out, int update_type) -> (Tensor, Tensor)");
        
    m.def("npu_lightning_indexer(Tensor query, Tensor key, Tensor weights, *, Tensor? actual_seq_lengths_query=None, Tensor? actual_seq_lengths_key=None, "
        "Tensor? block_table=None, Tensor? init_tensor=None, Tensor? local_tensor=None, str layout_query='BSND', str layout_key='BSND', int sparse_count=2048, int sparse_mode=3, "
        "int pre_tokens=9223372036854775807, int next_tokens=9223372036854775807, bool return_value=False) -> (Tensor, Tensor)");
    }

// 通过pybind将c++接口和python接口绑定，这里绑定的是接口不是算子
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
}
