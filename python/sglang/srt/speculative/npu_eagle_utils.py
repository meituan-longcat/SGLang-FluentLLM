from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from sglang.srt.env import global_server_args_dict
from sglang.srt.utils import get_colorful_logger

if TYPE_CHECKING:
    pass

top_k_white_list = {1: 4096, 2: 2048, 3: 2048, 4: 2048}
# xgrammar may leave an all-invalid probability row after the grammar terminates.
# Small inputs use the AICPU multinomial path and raise on that row; padding to
# this size selects the AIV path, which ignores the invalid-probability error.
_NPU_MULTINOMIAL_MIN_NUMEL = 8192

logger = get_colorful_logger(__name__)


def _pad_to_size(tensor: torch.Tensor, size: int, value=0):
    if tensor.shape[0] == size:
        return tensor
    if value == 0:
        return torch.cat(
            [tensor, tensor.new_zeros(size - tensor.shape[0], *tensor.shape[1:], dtype=tensor.dtype, device=tensor.device)],
            dim=0,
        )
    else:
        return torch.cat(
            [
                tensor,
                tensor.new_full((size - tensor.shape[0], *tensor.shape[1:]), value, dtype=tensor.dtype, device=tensor.device),
            ],
            dim=0,
        )


def _pad_for_npu_multinomial(probs: torch.Tensor) -> torch.Tensor:
    """Use the non-throwing NPU multinomial path for invalid masked rows."""
    min_rows = (_NPU_MULTINOMIAL_MIN_NUMEL + probs.shape[1] - 1) // probs.shape[1]
    return _pad_to_size(probs, max(probs.shape[0], min_rows))


def cummin_alternative_non_aicpu(accept_mask):
    """替代实现: arange + masked_fill + min — 纯 NPU 算子
    测试比accept_mask.to(torch.int32).cummin(dim=-1).values快

    思路:
    1. 生成 arange [0, 1, 2, ..., S-1]
    2. 把 False 位置设为索引值，True 位置设为大值 S
    3. 取每行 min = 第一个 False 的索引 (如果全 True 则为 S)
    4. arange < first_false_idx 就是 prefix-accept mask
    """
    B, S = accept_mask.shape
    arange = torch.arange(S, device=accept_mask.device, dtype=torch.int64)
    # False 位置保留索引，True 位置设为大值
    masked = torch.where(accept_mask, S, arange)
    # 每行第一个 False 的位置
    first_false = masked.min(dim=-1, keepdim=True).values
    # 位置 < 第一个 False 的位置 → True，否则 False
    return arange.unsqueeze(0) < first_false


def npu_verify_chain_greedy(
    candidates,
    target_predict,
    target_row_valid=None,
):
    """Greedy chain verification for NPU. Compares draft candidates against
    target argmax predictions and returns accepted token IDs with indices.

    Returns:
        predicts: Flat int32 tensor of accepted tokens (rejected positions zeroed).
        accept_index: Per-request indices of accepted positions (-1 for rejected).
        accept_token_num: Number of accepted draft tokens per request (excludes bonus).
    """
    batch_size, num_draft_tokens = candidates.shape

    comparison_result = candidates[:, 1:] == target_predict[:, : num_draft_tokens - 1]
    if target_row_valid is not None:
        comparison_result = comparison_result & target_row_valid.reshape(batch_size, num_draft_tokens)[:, :-1]
    comparison_result = cummin_alternative_non_aicpu(comparison_result)

    target_predict[:, 1:] = torch.where(comparison_result, target_predict[:, 1:], 0)
    predicts = target_predict.flatten().to(dtype=torch.int32)

    accept_index = torch.arange(0, num_draft_tokens * batch_size, device=candidates.device, dtype=torch.int32).reshape(batch_size, num_draft_tokens)

    accept_index[:, 1:] = torch.where(comparison_result, accept_index[:, 1:], -1)

    accept_token_num = comparison_result.int().sum(-1, dtype=torch.int32)
    return predicts, accept_index, accept_token_num


def npu_create_extend_spec_info(verified_id, accept_index, accept_length, new_verified_id):
    """Extract the last accepted token per request for NPU.
    Increments accept_length in-place (adds 1 to include the bonus token)."""
    accept_length.add_(1)
    accept_len_cum = torch.cumsum(accept_length, axis=0, dtype=torch.int)
    indices = (accept_len_cum - 1).clamp(min=0)
    torch.index_select(verified_id[accept_index].to(new_verified_id.dtype), 0, indices, out=new_verified_id)


def npu_rearrange_accept_index(accept_index, accept_length, bs, rearranged_accept_index):
    """Rearrange accept indices into a flat layout for NPU."""
    datatype = rearranged_accept_index.dtype
    import flash_npu_kernel  # noqa: F401  # Registers torch.ops.flash operators.

    torch.ops.flash.npu_rearrange_accept_index(accept_index.to(datatype), accept_length.to(datatype) + 1, bs, rearranged_accept_index)


def npu_sample_tokens(forward_batch, next_token_logits, sampling_top_k, only_return_probs):
    sampling_info = forward_batch.sampling_info
    batch_size = forward_batch.batch_size

    seq_len = next_token_logits.shape[0] // batch_size
    tokens = batch_size * seq_len

    next_token_logits_sort, probs_idx = torch.topk(next_token_logits, k=sampling_top_k, dim=-1, largest=True, sorted=True)
    # Grammar verification intentionally leaves unreachable/terminated tree rows
    # fully masked (-inf).  Keep that semantic state separate from the numeric
    # distribution: a finite dummy maximum makes softmax well-defined, while
    # target_row_valid prevents the dummy row from accepting draft tokens.
    target_row_valid = next_token_logits_sort[:, 0] != float("-inf")
    safe_first_logit = torch.where(
        target_row_valid,
        next_token_logits_sort[:, 0],
        torch.zeros_like(next_token_logits_sort[:, 0]),
    )
    next_token_logits_sort = torch.cat((safe_first_logit.unsqueeze(1), next_token_logits_sort[:, 1:]), dim=1)
    expanded_temperature = _pad_to_size(
        sampling_info.temperatures.view(-1, 1).expand(sampling_info.temperatures.numel(), seq_len).reshape(-1, 1), tokens, 1.0
    )

    probs_sort = F.softmax(next_token_logits_sort / expanded_temperature, dim=-1)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask_top_k = torch.arange(0, probs_sort.shape[-1], device=probs_sort.device).view(1, -1) >= _pad_to_size(
        sampling_info.top_ks.view(-1, 1).expand(sampling_info.top_ks.numel(), seq_len).reshape(-1, 1), tokens
    )
    mask_top_p = (probs_sum - probs_sort) > _pad_to_size(
        sampling_info.top_ps.view(-1, 1).expand(sampling_info.top_ps.numel(), seq_len).reshape(-1, 1), tokens
    )
    probs = torch.where(mask_top_k | mask_top_p, 0, probs_sort)  # [B*S, top_k]
    if global_server_args_dict["enable_speculative_sampling"]:
        probs = probs / probs.sum(dim=-1, keepdim=True)
    if only_return_probs:
        return None, probs, probs_idx, target_row_valid
    else:
        sampled_index = torch.multinomial(probs, num_samples=1)[:tokens]
        probs_idx = probs_idx.to(torch.int32)
        sampled_tokens = torch.gather(probs_idx, dim=1, index=sampled_index).view(-1)
        target_predict = _pad_to_size(sampled_tokens.reshape(tokens // seq_len, seq_len), batch_size)
        return target_predict, probs, probs_idx, target_row_valid


def npu_speculative_sampling_verify(
    candidates,
    probs,
    probs_idx,
    forward_batch,
    target_row_valid=None,
):
    with torch.profiler.record_function("npu_speculative_sampling_verify"):
        # Speculative sampling chain verification:
        # 1. accept draft token candidates[:, j + 1] at position j with prob min(1, p_j / q_j);
        # 2. keep the longest accepted prefix of length accept_token_num;
        # 3. sample bonus token from the target distribution at position accept_token_num
        #    (resampled from (p - q)+ on rejection, raw target probs if the whole chain
        #    is accepted, i.e. the bonus token);
        # 4. compose predicts: accepted draft tokens, then the sampled token, then zeros.
        batch_size = forward_batch.batch_size
        org_batch_size = getattr(forward_batch, "ori_bs", batch_size)  # real rows
        seq_len = probs.shape[0] // batch_size  # S

        # Target side: logits row j scores candidates[:, j + 1]; the last row is the bonus position.
        probs = probs.view(batch_size, seq_len, -1)  # [PB, S, K]
        probs_idx = probs_idx.view(batch_size, seq_len, -1)  # [PB, S, K]
        if target_row_valid is None:
            target_row_valid = torch.ones((batch_size, seq_len), dtype=torch.bool, device=probs.device)
        else:
            target_row_valid = target_row_valid.view(batch_size, seq_len)

        # Draft side (real rows only): slot j holds the distribution that produced
        # candidates[:, j + 1]; the last slot is a zero padding that reads as q = 0.
        req_idx: torch.Tensor = forward_batch.req_pool_indices[:org_batch_size]  # [B]
        p_draft = forward_batch.speculative_sampling_info.probs_selected[req_idx]  # [B, S-1]
        draft_token = candidates[:org_batch_size, 1:]  # [B, S-1]

        # Step 1: Accept — if uniform r < min(1, p_target(x)/p_draft(x)), accept draft token x.
        # p_target(x): the probs value where x appears in the target's candidate row
        # probs_idx, or 0 if it is not a candidate (row entries of probs_idx are unique).
        p_target = (probs[:org_batch_size, :-1] * (probs_idx[:org_batch_size, :-1] == draft_token.unsqueeze(-1))).sum(dim=-1)  # [B, S-1]
        r = torch.rand(org_batch_size, seq_len - 1, device=probs.device, dtype=probs.dtype)
        # r ~ U[0, 1), so r < min(1, p/q) is identical to r < p/q (no clamp needed).
        # q must be positive for a token that was actually sampled by the draft
        # model.  An inactive grammar row is not a probability distribution and
        # must terminate the accepted prefix rather than participate in p / q.
        valid_draft_prob = p_draft > 0
        safe_p_draft = torch.where(valid_draft_prob, p_draft, torch.ones_like(p_draft))
        accept_mask = target_row_valid[:org_batch_size, :-1] & valid_draft_prob & (r < (p_target / safe_p_draft))  # [B, S-1]

        # Step 2: accept_token_num — the length of the longest accepted prefix.
        # comparison_result=_pad_to_size(accept_mask.cummin(dim=-1).values, batch_size) # [PB, S-1], 0 for padded rows
        comparison_result = _pad_to_size(cummin_alternative_non_aicpu(accept_mask), batch_size)  # [PB, S-1], 0 for padded rows
        accept_token_num = comparison_result.int().sum(-1)  # [PB], in [0, S-1]

        # Step 3: Sample bonus token at position accept_token_num.
        arange_b = torch.arange(batch_size, device=probs.device)
        req_idx = forward_batch.req_pool_indices  # [PB]
        selected_probs = probs[arange_b, accept_token_num]  # [PB, K]
        selected_probs_idx = probs_idx[arange_b, accept_token_num]  # [PB, K]
        selected_row_valid = target_row_valid[arange_b, accept_token_num]  # [PB]
        # Resample from (p - q)+ on rejection; q is zero when the whole chain is
        # accepted (the padded draft slot), falling back to the raw target row (bonus).
        q_contrib = (
            forward_batch.speculative_sampling_info.probs[req_idx, accept_token_num].unsqueeze(1)
            * (forward_batch.speculative_sampling_info.probs_idx[req_idx, accept_token_num].unsqueeze(1) == selected_probs_idx.unsqueeze(2))
        ).sum(-1)  # [PB, K]
        # If grammar terminates immediately after the accepted prefix, the
        # fixed-shape verifier still needs a bonus slot although post-processing
        # will discard it.  Use the canonical dummy target row without subtracting
        # a draft distribution; normal reachable rows retain exact (p - q)+ math.
        q_contrib = torch.where(selected_row_valid.unsqueeze(1), q_contrib, torch.zeros_like(q_contrib))
        sampling_probs = (selected_probs - q_contrib).clamp(min=0)
        sampled_index = torch.multinomial(_pad_for_npu_multinomial(sampling_probs), num_samples=1)[:batch_size]
        sampled_token = selected_probs_idx.gather(1, sampled_index).view(-1).to(torch.int32)

        # Step 4: Compose predicts — accepted draft tokens+bonus token, other is zeros
        # accept_token_num, and zeros after it.
        predicts = torch.zeros(batch_size, seq_len, device=probs.device, dtype=torch.int32)  # [PB, S]
        predicts[:, :-1] = candidates[:, 1:].to(torch.int32) * comparison_result  # accepted draft tokens
        predicts.scatter_(1, accept_token_num.unsqueeze(1), sampled_token.unsqueeze(1))  # sampled token at L
        predicts = predicts.flatten()  # [PB * S]

        accept_index = torch.arange(0, seq_len * batch_size, device=candidates.device, dtype=torch.int32).reshape(batch_size, seq_len)
        accept_index[:, 1:] = torch.where(comparison_result, accept_index[:, 1:], -1)

        # Keep int64 for indexing above; match greedy/idle lengths at the verifier output.
        return predicts, accept_index, accept_token_num.to(torch.int32)


def npu_verify_chain_sample(spec_info, forward_batch, logits_output, candidates):
    """Top-k, top-p sampling verification for NPU using native pytorch operations."""
    sampling_top_k = top_k_white_list.get(forward_batch.batch_size, 1024)
    if global_server_args_dict["enable_speculative_sampling"] and not forward_batch.sampling_info.is_all_greedy:
        _, probs, probs_idx, target_row_valid = npu_sample_tokens(
            forward_batch,
            logits_output.next_token_logits,
            sampling_top_k,
            only_return_probs=True,
        )
        return npu_speculative_sampling_verify(
            candidates,
            probs,
            probs_idx,
            forward_batch,
            target_row_valid,
        )
    else:
        target_predict, _, _, target_row_valid = npu_sample_tokens(
            forward_batch,
            logits_output.next_token_logits,
            sampling_top_k,
            only_return_probs=False,
        )
        return npu_verify_chain_greedy(candidates, target_predict, target_row_valid)


def npu_verify(spec_info, forward_batch, logits_output, candidates, bs):
    """NPU verification dispatch: greedy or sampling."""
    if forward_batch.spec_info.is_all_greedy:
        target_values, target_predict = torch.max(logits_output.next_token_logits, dim=-1)
        target_row_valid = target_values != float("-inf")
        target_predict = target_predict.reshape(bs, spec_info.draft_token_num)
        predicts, accept_index, accept_length = npu_verify_chain_greedy(
            candidates,
            target_predict,
            target_row_valid,
        )
    else:
        predicts, accept_index, accept_length = npu_verify_chain_sample(
            spec_info,
            forward_batch,
            logits_output,
            candidates,
        )

    sampling_info = forward_batch.sampling_info
    if sampling_info.thinking_budgets is not None:
        sampling_info.apply_thinking_budgets_spec(
            forward_batch.seq_lens,
            predicts,
            accept_length,
            spec_info.draft_token_num,
        )

    # 支持 --speculative-force-accept-num：强制接受固定数量 token（用于 debug）
    # NPU 的 accept_index[i, j] 存全局 flat 索引 = i * num_draft_tokens + j
    # accept_length 此时不含 bonus（rearrange 内部会 +1），与 GPU 版语义一致
    force_accept_num: int = global_server_args_dict.get("speculative_force_accept_num", -1)
    if force_accept_num >= 0:
        num_draft_tokens: int = accept_index.shape[1]
        max_draft: int = num_draft_tokens - 1  # position 0 是 bonus，draft 最多 N-1 个
        forced: int = max(0, min(force_accept_num, max_draft))
        accept_length.fill_(forced)
        # 重建 accept_index：保留位置 0..forced（含 bonus），其余置 -1
        accept_index.fill_(-1)
        base = (torch.arange(bs, device=accept_index.device, dtype=accept_index.dtype) * num_draft_tokens).unsqueeze(1)  # [bs, 1]
        offsets = torch.arange(forced + 1, device=accept_index.device, dtype=accept_index.dtype).unsqueeze(0)  # [1, forced+1]
        accept_index[:, : forced + 1] = base + offsets  # [bs, forced+1]

    rearranged_accept_index = torch.zeros_like(predicts)
    npu_rearrange_accept_index(accept_index, accept_length, bs, rearranged_accept_index)
    return predicts, accept_length, rearranged_accept_index
