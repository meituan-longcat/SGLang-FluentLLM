from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
import torch_npu
import flash_npu_kernel

if TYPE_CHECKING:
    from sglang.srt.layers.logits_processor import LogitsProcessorOutput
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

top_k_white_list = {1: 4096, 2: 2048, 3: 2048, 4: 2048}


def _pad_to_size(tensor: torch.Tensor, size: int, value=0):
    if tensor.shape[0] == size:
        return tensor
    if value == 0:
        return torch.cat(
            [tensor, tensor.new_zeros(size - tensor.shape[0], *tensor.shape[1:],
                                      dtype=tensor.dtype, device=tensor.device)], dim=0,
        )
    else:
        return torch.cat(
            [
                tensor,
                tensor.new_full((size - tensor.shape[0], *tensor.shape[1:]), value,
                                dtype=tensor.dtype, device=tensor.device),
            ], dim=0,
        )


def npu_verify_chain_greedy(
    candidates,
    target_predict,
):
    """Greedy chain verification for NPU. Compares draft candidates against
    target argmax predictions and returns accepted token IDs with indices.

    Returns:
        predicts: Flat int32 tensor of accepted tokens (rejected positions zeroed).
        accept_index: Per-request indices of accepted positions (-1 for rejected).
        accept_token_num: Number of accepted draft tokens per request (excludes bonus).
    """
    batch_size, num_draft_tokens = candidates.shape

    comparison_result = candidates[:, 1:] == target_predict[:, :num_draft_tokens-1]
    comparison_result = comparison_result.cummin(dim=-1).values

    target_predict[:, 1:] = torch.where(comparison_result, target_predict[:, 1:], 0)
    predicts = target_predict.flatten().to(dtype=torch.int32)

    accept_index = torch.arange(
        0, num_draft_tokens * batch_size, device=candidates.device, dtype=torch.int32
    ).reshape(batch_size, num_draft_tokens)

    accept_index[:, 1:] = torch.where(comparison_result, accept_index[:, 1:], -1)

    accept_token_num = comparison_result.int().sum(-1)
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
    torch.ops.flash.npu_rearrange_accept_index(
        accept_index.to(datatype), accept_length.to(datatype) + 1, bs, rearranged_accept_index
    )


def npu_verify_chain_sample(spec_info, forward_batch, logits_output, candidates):
    """Top-k, top-p sampling verification for NPU using native pytorch operations."""
    sampling_info = forward_batch.sampling_info
    tokens = logits_output.next_token_logits.shape[0]
    expanded_temperature = _pad_to_size(
        sampling_info.temperatures.view(-1, 1).expand(sampling_info.temperatures.numel(), spec_info.draft_token_num).reshape(-1, 1), tokens, 1.0)

    probs = F.softmax(
        logits_output.next_token_logits / expanded_temperature, dim=-1
    )
    probs_sort, probs_idx = torch.topk(probs, k=top_k_white_list.get(forward_batch.batch_size, 1024), dim=-1, largest=True, sorted=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask_top_k = torch.arange(0, probs_sort.shape[-1], device=probs_sort.device).view(1, -1) >= _pad_to_size(
            sampling_info.top_ks.view(-1, 1).expand(sampling_info.top_ks.numel(), spec_info.draft_token_num).reshape(-1, 1), tokens)
    mask_top_p = (probs_sum - probs_sort) > _pad_to_size(
            sampling_info.top_ps.view(-1, 1).expand(sampling_info.top_ps.numel(), spec_info.draft_token_num).reshape(-1, 1), tokens)

    probs_sort = torch.where(mask_top_k | mask_top_p, 0, probs_sort)
    sampled_index = torch.multinomial(probs_sort, num_samples=1)
    probs_idx = probs_idx.to(torch.int32)
    sampled_tokens = torch.gather(probs_idx, dim=1, index=sampled_index).view(-1)
    target_predict = sampled_tokens.reshape(forward_batch.batch_size, spec_info.draft_token_num)
    return npu_verify_chain_greedy(candidates, target_predict)


def npu_verify(spec_info, forward_batch, logits_output, candidates, bs):
    """NPU verification dispatch: greedy or sampling."""
    if forward_batch.spec_info.is_all_greedy:
        target_predict = torch.argmax(
            logits_output.next_token_logits, dim=-1
        ).reshape(bs, spec_info.draft_token_num)
        predicts, accept_index, accept_length = npu_verify_chain_greedy(
            candidates, target_predict,
        )
    else:
        predicts, accept_index, accept_length = npu_verify_chain_sample(
            spec_info, forward_batch, logits_output, candidates,
        )

    rearranged_accept_index = torch.zeros_like(predicts)
    npu_rearrange_accept_index(accept_index, accept_length, bs, rearranged_accept_index)
    return predicts, accept_length, rearranged_accept_index
