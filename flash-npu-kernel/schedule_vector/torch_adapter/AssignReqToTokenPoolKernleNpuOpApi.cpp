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

void npu_assign_req_to_token_pool(
    const at::Tensor &req_pool_indices,
    const at::Tensor &extend_lens,
    const at::Tensor &alloced_lens,
    const at::Tensor &out_cache_loc,
    at::Tensor &req_to_token_pool,
    int64_t bs
)
{
    TORCH_CHECK(req_pool_indices.dim() == 1,
        "npu_assign_req_to_token_pool: Input req_pool_indices should be 1D tensor"
        + OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(extend_lens.dim() == alloced_lens.dim(),
        "npu_assign_req_to_token_pool: extend_lens and alloced_lens must have same dim");
    TORCH_CHECK(out_cache_loc.dim() == 1,
        "npu_assign_req_to_token_pool: Input out_cache_loc should be 1D tensor"
        + OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(req_to_token_pool.dim() == 2,
        "npu_assign_req_to_token_pool: Input req_to_token_pool should be 2D tensor"
        + OPS_ERROR(ErrCode::PARAM));
    EXEC_NPU_CMD(aclnnAssignReqToTokenPool, req_pool_indices, extend_lens, alloced_lens, out_cache_loc,
       bs, req_to_token_pool);
}
} // namespace op_api