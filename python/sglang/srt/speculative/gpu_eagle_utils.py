from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from flashinfer.sampling import (
    top_k_renorm_prob,
    top_p_renorm_probs,
    verify_chain_greedy,
    chain_speculative_sampling_target_only,
)

from sglang.srt.layers.attention.utils import create_flashinfer_kv_indices_triton
from sglang.srt.utils.common import get_device
from sglang.srt.env import global_server_args_dict

if TYPE_CHECKING:
    from sglang.srt.layers.logits_processor import LogitsProcessorOutput
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


# ---------------------------------------------------------------------------
# GPU verification functions (flashinfer-based)
# ---------------------------------------------------------------------------

def gpu_verify_greedy(spec_info, predicts, candidates, target_predict, bs):
    """Greedy verification for GPU using flashinfer."""
    accept_index = torch.full(
        (bs, spec_info.spec_steps + 1), -1, dtype=torch.int32, device=get_device()
    )
    accept_length = torch.empty((bs,), dtype=torch.int32, device=get_device())
    verify_chain_greedy(
        predicts=predicts,
        accept_index=accept_index,
        accept_token_num=accept_length,
        candidates=candidates.to(torch.int32),
        target_predict=target_predict,
        batch_size=bs,
        num_draft_tokens=spec_info.draft_token_num
    )
    return predicts, accept_index, accept_length


def gpu_verify_sample(spec_info, predicts, candidates, forward_batch, logits_output, bs):
    """Probabilistic sampling verification for GPU using flashinfer."""
    accept_index = torch.full(
        (bs, spec_info.spec_steps + 1), -1, dtype=torch.int32, device=get_device()
    )
    accept_length = torch.empty((bs,), dtype=torch.int32, device=get_device())
    sampling_info = forward_batch.sampling_info
    expanded_temperature = torch.repeat_interleave(
        sampling_info.temperatures, spec_info.draft_token_num, dim=0
    )

    target_probs = F.softmax(
        logits_output.next_token_logits / expanded_temperature, dim=-1
    )
    target_probs = top_k_renorm_prob(
        target_probs,
        torch.repeat_interleave(
            sampling_info.top_ks, spec_info.draft_token_num, dim=0
        ),
    )
    target_probs = top_p_renorm_probs(
        target_probs,
        torch.repeat_interleave(
            sampling_info.top_ps, spec_info.draft_token_num, dim=0
        ),
    )
    target_probs = target_probs.reshape(bs, spec_info.draft_token_num, -1)

    draft_probs = torch.zeros(
        target_probs.shape, dtype=torch.float32, device=get_device()
    )
    coins = torch.rand_like(candidates, dtype=torch.float32, device=get_device())
    coins_for_final_sampling = torch.rand(
        (bs,), dtype=torch.float32, device=get_device()
    )
    chain_speculative_sampling_target_only(
        predicts=predicts,
        accept_index=accept_index,
        accept_token_num=accept_length,
        candidates=candidates.to(torch.int32),
        uniform_samples=coins,
        uniform_samples_for_final_sampling=coins_for_final_sampling,
        target_probs=target_probs,
        draft_probs=draft_probs,
        threshold_single=global_server_args_dict[
            "speculative_accept_threshold_single"
        ],
        threshold_acc=global_server_args_dict[
            "speculative_accept_threshold_acc"
        ],
        deterministic=True,
    )
    return predicts, accept_index, accept_length


@triton.jit
def rearrange_accept_index(
    accept_index_ptr,
    accept_length_ptr,
    output_ptr,
    num_tokens_per_req_upper: tl.constexpr,
    accept_index_stride: tl.constexpr,
):
    pid = tl.program_id(0)
    accept_len = tl.load(accept_length_ptr + pid) + 1
    cum_accept_len = 0
    for i in range(pid):
        cum_accept_len += (tl.load(accept_length_ptr + i) + 1)
    store_offset = tl.arange(0, num_tokens_per_req_upper)
    accept_index_load_offset = (
        tl.arange(0, num_tokens_per_req_upper) + pid * accept_index_stride
    )
    accept_index = tl.load(accept_index_ptr + accept_index_load_offset)
    tl.store(
        output_ptr + store_offset + cum_accept_len,
        accept_index,
        mask=store_offset < accept_len,
    )


def gpu_verify(spec_info, forward_batch, logits_output, candidates, bs):
    """GPU verification dispatch: greedy or sampling, using flashinfer + triton rearrange."""
    predict_shape = list(logits_output.next_token_logits.shape)[:-1]
    predicts = torch.zeros(predict_shape, dtype=torch.int32, device=get_device())

    if forward_batch.spec_info.is_all_greedy:
        target_predict = torch.argmax(
            logits_output.next_token_logits, dim=-1
        ).reshape(bs, spec_info.draft_token_num)
        predicts, accept_index, accept_length = gpu_verify_greedy(
            spec_info, predicts, candidates, target_predict, bs,
        )
    else:
        predicts, accept_index, accept_length = gpu_verify_sample(
            spec_info, predicts, candidates, forward_batch, logits_output, bs,
        )

    rearranged_accept_index = torch.zeros_like(predicts)
    rearrange_accept_index[(bs,)](
        accept_index_ptr=accept_index,
        accept_length_ptr=accept_length,
        output_ptr=rearranged_accept_index,
        num_tokens_per_req_upper=triton.next_power_of_2(spec_info.draft_token_num),
        accept_index_stride=accept_index.shape[1],
    )
    return predicts, accept_length, rearranged_accept_index


@triton.jit(
    do_not_specialize_on_alignment=["num_seqs", "kv_indices_stride"],
)
def generate_draft_decode_kv_indices(
    req_pool_indices,
    req_to_token,
    paged_kernel_lens,
    kv_indices,  # shape: [self.speculative_num_steps, forward_batch.batch_size * self.topk * self.max_context_len], records slot address for topk at each position for each step
    kv_indptr,  # shape: [self.speculative_num_steps, max_batch_size * topk + 1], records starting address of topk for each step
    positions,
    num_seqs: int,
    kv_indices_stride: int,
    topk: tl.constexpr,
    pool_len: tl.constexpr,
    kv_indptr_stride: tl.constexpr,
    max_bs: tl.constexpr,
    iter_upper: tl.constexpr,
    max_num_tokens: tl.constexpr,
):
    """
    Rewrite req to token mapping from request-isolated to spec_step-isolated in kv_indices
    """
    BLOCK_SIZE: tl.constexpr = 128
    iters = tl.program_id(axis=0)  # Which round of draft
    bid = tl.program_id(axis=1)  # Specific seq in batch
    topk_id = tl.program_id(axis=2)  # Which one in topk

    kv_indices += kv_indices_stride * iters
    kv_indptr += kv_indptr_stride * iters
    iters += 1

    load_offset = tl.arange(0, max_bs)
    # Lengths of all seqs in batch
    seq_lens = tl.load(paged_kernel_lens + load_offset, mask=load_offset < bid)
    # Current seq length
    seq_len = tl.load(paged_kernel_lens + bid)
    cum_seq_len = tl.sum(seq_lens)

    kv_offset = cum_seq_len * topk + bid * iters * topk + topk_id * (seq_len + iters)
    # Write position
    kv_ptr = kv_indices + kv_offset
    token_pool_ptr = req_to_token + tl.load(req_pool_indices + bid) * pool_len

    kv_offset = tl.arange(0, BLOCK_SIZE)
    num_loop = tl.cdiv(seq_len, BLOCK_SIZE)
    # Block copy, copy info from original req_token_pool to buffer
    for _ in range(num_loop):
        mask = kv_offset < seq_len
        data = tl.load(token_pool_ptr + kv_offset, mask=mask)
        tl.store(kv_ptr + kv_offset, data, mask=mask)
        kv_offset += BLOCK_SIZE

    extend_offset = tl.arange(0, iter_upper)
    # Block copy, copy slot addresses from corresponding positions in req_to_token to kv_indices
    extend_data = tl.load(
        token_pool_ptr + seq_len + tl.arange(0, iter_upper) * topk + topk_id,
        mask=extend_offset < iters,
    )
    tl.store(kv_ptr + seq_len + extend_offset, extend_data, mask=extend_offset < iters)

    # Update kv_indptr
    bs_offset = tl.arange(0, max_num_tokens)

    zid = bid * topk + topk_id
    if zid == 0:
        zid = num_seqs * topk
    positions = tl.load(positions + bs_offset, mask=bs_offset < zid)
    base = tl.sum(positions)
    tl.store(kv_indptr + zid, base + zid * iters)


@triton.jit
def generate_attn_arg_v2_kernel(
    req_pool_indices_ptr,
    paged_kernel_lens_ptr,
    req_to_token_ptr,
    qo_indptr,
    cum_kv_seq_len_ptr,
    kv_indices_ptr,
    req_to_token_ptr_stride: tl.constexpr,
    draft_token_num: tl.constexpr,
    draft_decode_step: tl.constexpr,
    bs_upper: tl.constexpr,
    is_draft_decode: tl.constexpr,
):
    BLOCK_SIZE: tl.constexpr = 4096
    bx = tl.program_id(axis=0)
    if bx == 0:
        tl.store(qo_indptr, 0)
        tl.store(cum_kv_seq_len_ptr, 0)

    indices = tl.arange(0, bs_upper)
    paged_kernel_lens = tl.load(paged_kernel_lens_ptr + indices, mask=(indices <= bx), other=0)
    if is_draft_decode:
        paged_kernel_lens += tl.where(paged_kernel_lens != 0, draft_decode_step + 1, 0)
    else:
        paged_kernel_lens += tl.where(paged_kernel_lens != 0, draft_token_num, 0)

    cum_kv_len = tl.sum(paged_kernel_lens)
    if is_draft_decode:
        tl.store(qo_indptr + bx + 1, bx + 1)
    else:
        tl.store(qo_indptr + bx + 1, (bx + 1) * draft_token_num)

    tl.store(cum_kv_seq_len_ptr + bx + 1, cum_kv_len)

    req_pool_index = tl.load(req_pool_indices_ptr + bx)
    cur_paged_kernel_len = tl.sum(tl.where(indices==bx, paged_kernel_lens, 0))
    #kv_indices_offset = cum_kv_len - tl.load(paged_kernel_lens_ptr + bx)
    kv_indices_offset = cum_kv_len - cur_paged_kernel_len

    kv_start = 0
    kv_end = cur_paged_kernel_len
    #kv_end = tl.load(paged_kernel_lens_ptr + bx)

    num_loop = tl.cdiv(kv_end - kv_start, BLOCK_SIZE)
    for i in range(num_loop):
        offset = tl.arange(0, BLOCK_SIZE) + i * BLOCK_SIZE
        mask = offset < kv_end - kv_start
        data = tl.load(
            req_to_token_ptr
            + req_pool_index * req_to_token_ptr_stride
            + kv_start
            + offset,
            mask=mask,
        )
        tl.store(kv_indices_ptr + kv_indices_offset + offset, data, mask=mask)


def generate_attn_arg_v2(
    draft_token_num: int,
    req_pool_indices: torch.Tensor,
    paged_kernel_lens: torch.Tensor,
    req_to_token: torch.Tensor,
    kv_indices_buf: torch.Tensor,
    is_draft_decode: bool = False,
    draft_decode_step: int = None
):
    batch_size = req_pool_indices.shape[0]
    qo_indptr = torch.empty((batch_size + 1,), device="cuda", dtype=torch.int32)
    cum_kv_lens = torch.empty((batch_size + 1,), device="cuda", dtype=torch.int32)
    assert kv_indices_buf is not None
    generate_attn_arg_v2_kernel[(batch_size,)](
        req_pool_indices_ptr=req_pool_indices,
        paged_kernel_lens_ptr=paged_kernel_lens,
        req_to_token_ptr=req_to_token,
        qo_indptr=qo_indptr,
        cum_kv_seq_len_ptr=cum_kv_lens,
        kv_indices_ptr=kv_indices_buf,
        req_to_token_ptr_stride=req_to_token.size(1),
        draft_token_num=draft_token_num,
        draft_decode_step=draft_decode_step,
        is_draft_decode=is_draft_decode,
        bs_upper=triton.next_power_of_2(batch_size)
    )
    return kv_indices_buf, cum_kv_lens, qo_indptr


def generate_attn_arg_prefill(
    draft_token_num: int,
    req_pool_indices: torch.Tensor,
    paged_kernel_lens: torch.Tensor,
    req_to_token: torch.Tensor,
    kv_indices_buf: torch.Tensor = None,
    draft_decode_step: int = None
):
    batch_size = req_pool_indices.shape[0]
    if draft_decode_step is not None:
        qo_indptr = torch.arange(
            0,
            (1 + batch_size),
            step=1,
            dtype=torch.int32,
            device="cuda",
        )
    else:
        qo_indptr = torch.arange(
            0,
            (1 + batch_size) * draft_token_num,
            step=draft_token_num,
            dtype=torch.int32,
            device="cuda",
        )

    cum_kv_seq_len = torch.zeros((batch_size + 1,), dtype=torch.int32, device="cuda")

    if draft_decode_step is None:
        paged_kernel_lens = paged_kernel_lens + draft_token_num
    else:
        paged_kernel_lens = paged_kernel_lens + draft_decode_step + 1

    cum_kv_seq_len[1:] = torch.cumsum(paged_kernel_lens, dim=0)
    if kv_indices_buf is not None:
        kv_indices = kv_indices_buf
    else:
        # Prevent kv_indices out of bounds in large steps
        kv_indices = torch.empty(cum_kv_seq_len[-1] + 256, dtype=torch.int32, device="cuda")
    create_flashinfer_kv_indices_triton[(batch_size,)](
        req_to_token,
        req_pool_indices,
        paged_kernel_lens,
        cum_kv_seq_len,
        None,
        kv_indices,
        req_to_token.size(1),
    )
    return kv_indices, cum_kv_seq_len, qo_indptr, None

@triton.jit
def update_oe_metadata_kernel(
    oe_out_column_starts_ptr,
    oe_column_starts_ptr,
    oe_out_req_lens_ptr,
    oe_req_lens_ptr,
    verified_len_ptr,
    req_pool_indices_ptr,
    draft_decode_step: tl.constexpr,
    spec_num_steps: tl.constexpr
):
    pid = tl.program_id(axis=0)
    req_idx = tl.load(req_pool_indices_ptr + pid)
    veridied_len = tl.load(verified_len_ptr + req_idx)
    tl.store(oe_out_column_starts_ptr + pid, veridied_len + 1 + draft_decode_step)
    tl.store(oe_out_req_lens_ptr + pid, 1)
    if draft_decode_step < spec_num_steps - 1:
        tl.store(oe_column_starts_ptr + pid, veridied_len + 1 + draft_decode_step)
        tl.store(oe_req_lens_ptr + pid, 1)
