#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

void npu_rearrange_accept_index(
    const at::Tensor &accept_index,
    const at::Tensor &accept_length,
    int64_t bs,
    at::Tensor &output
) 
{
    TORCH_CHECK(accept_index.dim() == 2,
                "npu_rearrange_accept_index: accept_index must be 2D tensor");
    TORCH_CHECK(accept_length.dim() == 1,
        "npu_rearrange_accept_index: Input accept_length should be 1D tensor"
        + OPS_ERROR(ErrCode::PARAM));
    EXEC_NPU_CMD(aclnnRearrangeAcceptIndex, accept_index, accept_length, bs, output);
}

}