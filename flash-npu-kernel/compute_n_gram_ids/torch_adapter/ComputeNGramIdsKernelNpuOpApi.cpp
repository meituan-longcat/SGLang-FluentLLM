#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {

at::Tensor compute_n_gram_ids(
  const at::Tensor& oe_weights,
  const at::Tensor& oe_mods,
  const at::Tensor& exclusive_oe_embeder_size_sums,
  const at::Tensor& tokens,
  const at::Tensor& exclusive_req_len_sums,
  const at::Tensor& oe_token_table,
  const at::Tensor& row_indices,
  const at::Tensor& column_starts,
  int64_t batch_size,
  int64_t oe_n,
  int64_t oe_k,
  int64_t max_context_len)
{
  int64_t token_num = tokens.size(0);
  std::vector<int64_t> shape = {token_num, (oe_n - 1) * oe_k};
  at::Tensor result = at_npu::native::OpPreparation::apply_tensor_with_format(
    shape, oe_weights.options(), ACL_FORMAT_ND);
  // 调用NPU算子接口，完成输出结果的计算
  EXEC_NPU_CMD(aclnnComputeNGramIds,
    oe_weights,
    oe_mods,
    exclusive_oe_embeder_size_sums,
    tokens,
    exclusive_req_len_sums,
    oe_token_table,
    row_indices,
    column_starts,
    batch_size,
    oe_n,
    oe_k,
    max_context_len,
    result);

  return result;
}

}  // namespace op_api