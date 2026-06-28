// Copyright (c) 2025 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

void npu_get_out_cache_loc(
    const at::Tensor &req_to_token,
    const at::Tensor &req_pool_indices,
    const at::Tensor &new_compute_lens,
    const at::Tensor &cache_lens,
    at::Tensor &out_cache_loc,
    int64_t bs
)
{
    TORCH_CHECK(req_to_token.dim() == 2,
        "npu_get_out_cache_loc: Input req_to_token should be 2D tensor"
        + OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(req_pool_indices.dim() == cache_lens.dim(),
        "npu_get_out_cache_loc: req_pool_indices and cache_lens must have same dimension");
    TORCH_CHECK(cache_lens.dim() == 1,
        "npu_get_out_cache_loc: Input cache_lens should be 1D tensor"
        + OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(out_cache_loc.dim() == 1,
        "npu_get_out_cache_loc: Input out_cache_loc should be 1D tensor"
        + OPS_ERROR(ErrCode::PARAM));
    EXEC_NPU_CMD(aclnnGetOutCacheLoc, req_to_token, req_pool_indices, new_compute_lens, cache_lens, bs, out_cache_loc);
}
} // namespace op_api