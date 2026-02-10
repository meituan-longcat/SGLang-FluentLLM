from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, List, Optional, Union

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from sglang.srt.constrained.base_grammar_backend import BaseGrammarObject
from sglang.srt.layers.attention.utils import create_flashinfer_kv_indices_triton
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
from sglang.srt.utils import get_colorful_logger, is_cuda_available

from sglang.srt.utils import is_npu
__is_npu__ = is_npu()

logger = get_colorful_logger(__name__)

if is_cuda_available():
    from flashinfer.sampling import top_k_renorm_prob, top_p_renorm_probs, verify_chain_greedy, chain_speculative_sampling_target_only


if TYPE_CHECKING:
    from sglang.srt.layers.logits_processor import LogitsProcessorOutput
    from sglang.srt.managers.schedule_batch import ModelWorkerBatch
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

from sglang.srt.env import global_server_args_dict


@dataclasses.dataclass
class EagleDraftInput:
    # The inputs for decode
    # shape: (b, topk)
    topk_p: torch.Tensor = None
    topk_index: torch.Tensor = None
    # shape: (b, hidden_size)
    hidden_states: torch.Tensor = None
    capture_hidden_mode: CaptureHiddenMode = CaptureHiddenMode.FULL

    # Inputs for extend
    # shape: (b,)
    verified_id: torch.Tensor = None
    accept_length: torch.Tensor = None
    accept_length_cpu: List[int] = None
    accept_index: torch.Tensor = None

    # Inputs for the attention backends
    # shape: (b + 1,)
    kv_indptr: torch.Tensor = None
    kv_indices: torch.Tensor = None

    # For draft extend fast plan
    qo_indptr_cpu: torch.Tensor = None
    kv_indptr_cpu: torch.Tensor = None
    kv_indices_for_extend: torch.Tensor = None
    kv_len_arr_cpu: torch.Tensor = None

    draft_token_num: int = 0

    def set_input_ids(self, forward_batch: ForwardBatch):
        pt = 0
        for i, extend_seq_len in enumerate(forward_batch.extend_seq_lens):
            input_ids = forward_batch.draft_input_ids[i]
            if input_ids[-1] == -1:
                input_ids[-1] = self.verified_id[i]
            forward_batch.input_ids[pt : pt + extend_seq_len] = input_ids

            pt += extend_seq_len

    def prepare_extend_after_decode(self, forward_batch: ForwardBatch, use_oe: bool=False):
        new_verified_id = torch.empty_like(self.accept_length, dtype=torch.long)
        create_extend_spec_info[(forward_batch.batch_size,)](
            self.verified_id,
            new_verified_id,
            self.accept_length,
            forward_batch.oe_column_starts,
            forward_batch.oe_req_lens,
            forward_batch.req_pool_indices,
            forward_batch.req_to_token_pool.verified_lens,
            self.draft_token_num,
            forward_batch.batch_size,
            use_oe
        )
        # Accepted tokens (padded)
        forward_batch.input_ids = self.verified_id
        # Extract the last accepted token for each request
        self.verified_id = new_verified_id
        return self.verified_id

    def filter_batch(self, new_indices: torch.Tensor):
        self.topk_p = self.topk_p[: len(new_indices)]
        self.topk_index = self.topk_index[: len(new_indices)]
        self.hidden_states = self.hidden_states[: len(new_indices)]
        self.verified_id = self.verified_id[: len(new_indices)]

    def merge_batch(self, spec_info: EagleDraftInput):
        if self.hidden_states is None:
            self.hidden_states = spec_info.hidden_states
            self.verified_id = spec_info.verified_id
            self.topk_p = spec_info.topk_p
            self.topk_index = spec_info.topk_index
            return
        if spec_info.hidden_states is None:
            return
        self.hidden_states = torch.cat(
            [self.hidden_states, spec_info.hidden_states], axis=0
        )
        self.verified_id = torch.cat([self.verified_id, spec_info.verified_id], axis=0)
        self.topk_p = torch.cat([self.topk_p, spec_info.topk_p])
        self.topk_index = torch.cat([self.topk_index, spec_info.topk_index])


@dataclasses.dataclass
class EagleDraftOutput:
    """
    Both prefill and decode batches end with draft. Used to store the previous draft's information,
    to construct verify's input at the next decode

    Args:
        last_verified_ids:
    """

    last_verified_ids: torch.Tensor
    token_list: Optional[torch.Tensor, List]

    def filter_batch(self, keep_indices: torch.Tensor):
        # 1. chunked prefill
        # 2. retract
        # 3. Check finished when updating running and getting new
        self.last_verified_ids = self.last_verified_ids[keep_indices]
        if isinstance(self.token_list, torch.Tensor):
            self.token_list = self.token_list[keep_indices, :]
        elif isinstance(self.token_list, list):
            self.token_list = [s[keep_indices] for s in self.token_list]
        else:
            raise RuntimeError(f"Not supported token_list type, {self.token_list=}")

    def merge_batch(self, spec_info):
        if spec_info.last_verified_ids is None:
            return
        if self.last_verified_ids is None:
            # May reach here when all requests in running batch are finished
            self.last_verified_ids = spec_info.last_verified_ids
            self.token_list = spec_info.token_list
            return
        self.last_verified_ids = torch.cat(
            [self.last_verified_ids, spec_info.last_verified_ids]
        )
        if isinstance(self.token_list, torch.Tensor):
            self.token_list = torch.cat([self.token_list, spec_info.token_list], dim=0)
        elif isinstance(self.token_list, list):
            self.token_list = [
                torch.cat([s1, s2], axis=0)
                for s1, s2 in zip(self.token_list, spec_info.token_list)
            ]
        else:
            raise RuntimeError(f"Not supported token_list type, {self.token_list=}")


@dataclasses.dataclass
class EagleVerifyInput:
    draft_token: torch.Tensor
    positions: torch.Tensor
    draft_token_num: int
    spec_steps: int
    capture_hidden_mode: CaptureHiddenMode
    is_all_greedy: bool
    grammar: BaseGrammarObject = None

    @classmethod
    def create(
        cls,
        verified_id: torch.Tensor,
        token_list: torch.Tensor,
        seq_lens: torch.Tensor,
        spec_steps: int,
        num_verify_tokens: int,
        is_all_greedy: bool,
        is_idle: bool,
    ):
        if is_idle:
            return cls(
                torch.empty(0, dtype=torch.int32, device="cuda"),
                torch.empty(0, dtype=torch.int32, device="cuda"),
                0,
                spec_steps,
                CaptureHiddenMode.LAST,
                True,
            )
        else:
            draft_tokens = torch.cat((verified_id.unsqueeze(1), token_list), dim=1).flatten()
            positions = (seq_lens.unsqueeze(1) + torch.arange(num_verify_tokens, device=draft_tokens.device)).flatten()
            return cls(
                draft_tokens,
                positions,
                num_verify_tokens,
                spec_steps,
                CaptureHiddenMode.FULL,
                is_all_greedy
            )

    def prepare_for_verify(self, model_worker_batch: ModelWorkerBatch):
        model_worker_batch.input_ids = self.draft_token
        model_worker_batch.capture_hidden_mode = CaptureHiddenMode.FULL

    def verify(
        self,
        forward_batch: ForwardBatch,
        logits_output: LogitsProcessorOutput,
        vocab_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bs = forward_batch.batch_size
        candidates = self.draft_token.reshape(bs, self.draft_token_num)
        predict_shape = list(logits_output.next_token_logits.shape)[:-1]
        predict = torch.zeros(predict_shape, dtype=torch.int32, device="cuda")
        accept_index = torch.full(
            (bs, self.spec_steps + 1), -1, dtype=torch.int32, device="cuda"
        )
        accept_length = torch.empty((bs,), dtype=torch.int32, device="cuda")

        # Apply grammar mask
        if vocab_mask is not None:
            assert self.grammar is not None
            self.grammar.apply_vocab_mask(
                logits=logits_output.next_token_logits, vocab_mask=vocab_mask
            )

        if forward_batch.spec_info.is_all_greedy:
            target_predict = torch.argmax(logits_output.next_token_logits, dim=-1)
            target_predict = target_predict.reshape(bs, self.draft_token_num)
            verify_chain_greedy(
                predicts=predict,
                accept_index=accept_index,
                accept_token_num=accept_length,
                candidates=candidates.to(torch.int32),
                target_predict=target_predict,
                batch_size=bs,
                num_draft_tokens=self.draft_token_num
            )
        else:
            sampling_info = forward_batch.sampling_info
            expanded_temperature = torch.repeat_interleave(
                sampling_info.temperatures, self.draft_token_num, dim=0
            )  # (bs * draft_token_num, 1)

            target_probs = F.softmax(
                logits_output.next_token_logits / expanded_temperature, dim=-1
            )  # (bs * draft_token_num, vocab_size)
            target_probs = top_k_renorm_prob(
                target_probs,
                torch.repeat_interleave(
                    sampling_info.top_ks, self.draft_token_num, dim=0
                ),
            )  # (bs * draft_token_num, vocab_size)
            target_probs = top_p_renorm_probs(
                target_probs,
                torch.repeat_interleave(
                    sampling_info.top_ps, self.draft_token_num, dim=0
                ),
            )
            target_probs = target_probs.reshape(bs, self.draft_token_num, -1)

            draft_probs = torch.zeros(
                target_probs.shape, dtype=torch.float32, device="cuda"
            )
            coins = torch.rand_like(candidates, dtype=torch.float32, device="cuda")
            coins_for_final_sampling = torch.rand(
                (bs,), dtype=torch.float32, device="cuda"
            )
            chain_speculative_sampling_target_only(
                predicts=predict,  # mutable
                accept_index=accept_index,  # mutable
                accept_token_num=accept_length,  # mutable
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
        rearranged_accept_index = torch.zeros_like(predict)
        rearrange_accept_index[(bs,)](
            accept_index_ptr=accept_index,
            accept_length_ptr=accept_length,
            output_ptr=rearranged_accept_index,
            num_tokens_per_req_upper=triton.next_power_of_2(self.draft_token_num),
            accept_index_stride=accept_index.shape[1],
        )
        return predict, logits_output, accept_length, rearranged_accept_index


def update_oe_metadata(forward_batch: ForwardBatch, draft_decode_step: int, spec_num_steps: int):
    bs = forward_batch.batch_size
    update_oe_metadata_kernel[(bs,)](
        oe_out_column_starts_ptr=forward_batch.oe_out_column_starts,
        oe_column_starts_ptr=forward_batch.oe_column_starts,
        oe_out_req_lens_ptr=forward_batch.oe_out_req_lens,
        oe_req_lens_ptr=forward_batch.oe_req_lens,
        verified_len_ptr=forward_batch.req_to_token_pool.verified_lens,
        req_pool_indices_ptr=forward_batch.req_pool_indices,
        draft_decode_step=draft_decode_step,
        spec_num_steps=spec_num_steps
    )

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

@triton.jit
def update_draft_decode_cache_kernel(
    out_cache_loc_ptr,
    out_cache_loc_out_ptr,
    draft_decode_step: tl.constexpr,
    stride: tl.constexpr,
    batch_size: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= batch_size:
        return
    idx = draft_decode_step + pid * stride
    val = tl.load(out_cache_loc_ptr + idx)
    tl.store(out_cache_loc_out_ptr + pid, val)

def update_draft_decode_cache(
    forward_batch:ForwardBatch,
    out_cache_loc: torch.Tensor,
    draft_decode_step: int,
    speculative_num_steps: int
):
    bs = forward_batch.batch_size
    out_cache_loc_new = torch.empty(bs, dtype=out_cache_loc.dtype, device=out_cache_loc.device)
    grid = (bs,)
    update_draft_decode_cache_kernel[grid](
        out_cache_loc_ptr=out_cache_loc,
        out_cache_loc_out_ptr=out_cache_loc_new,
        draft_decode_step=draft_decode_step,
        stride=speculative_num_steps - 1,
        batch_size=bs,
    )
    forward_batch.out_cache_loc = out_cache_loc_new


@triton.jit
def create_extend_spec_info(
    verified_id, # padded verified id
    new_verified_id,
    accept_length_ptr,
    oe_column_starts_ptr,
    oe_req_lens_ptr,
    req_pool_indices_ptr,
    verified_lens_ptr,
    spec_num_tokens: int,
    batch_size: int,
    use_oe: tl.constexpr 
):
    pid = tl.program_id(axis=0)
    if pid >= batch_size:
        return
    accept_len = tl.load(accept_length_ptr + pid)
    last_verified_id = tl.load(verified_id + pid * spec_num_tokens + accept_len)
    if use_oe:
        req_pool_index = tl.load(req_pool_indices_ptr + pid)
        verified_len = tl.load(verified_lens_ptr + req_pool_index)
        tl.store(oe_req_lens_ptr + pid, spec_num_tokens)
        tl.store(oe_column_starts_ptr + pid, verified_len + 1)
    tl.store(accept_length_ptr + pid, accept_len + 1)
    tl.store(new_verified_id + pid, last_verified_id)

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


@triton.jit
def assign_req_to_token_pool(
    req_pool_indices,
    req_to_token,
    start_offset,
    end_offset,
    out_cache_loc,
    pool_len: tl.constexpr,
    bs_upper: tl.constexpr,
):
    BLOCK_SIZE: tl.constexpr = 32
    pid = tl.program_id(axis=0)
    kv_start = tl.load(start_offset + pid)
    kv_end = tl.load(end_offset + pid)
    token_pool = req_to_token + tl.load(req_pool_indices + pid) * pool_len

    length_offset = tl.arange(0, bs_upper)
    start = tl.load(start_offset + length_offset, mask=length_offset < pid)
    end = tl.load(end_offset + length_offset, mask=length_offset < pid)
    out_offset = tl.sum(end - start, axis=0)

    out_cache_ptr = out_cache_loc + out_offset

    save_offset = tl.arange(0, BLOCK_SIZE) + kv_start
    load_offset = tl.arange(0, BLOCK_SIZE)

    num_loop = tl.cdiv(kv_end - kv_start, BLOCK_SIZE)
    for _ in range(num_loop):
        mask = save_offset < kv_end
        data = tl.load(out_cache_ptr + load_offset, mask=mask)
        tl.store(token_pool + save_offset, data, mask=mask)
        save_offset += BLOCK_SIZE
        load_offset += BLOCK_SIZE

if not __is_npu__:
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


def fast_topk(values, topk, dim):
    if topk == 1:
        # Use max along the specified dimension to get both value and index
        max_value, max_index = torch.max(values, dim=dim)
        return max_value.unsqueeze(1), max_index.unsqueeze(1)
    else:
        # Use topk for efficiency with larger k values
        return torch.topk(values, topk, dim=dim)

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


@triton.jit
def prepare_for_multi_step_draft_kernel(
    verified_lens_ptr,
    req_pool_indices_ptr,
    accept_lengths_ptr,
    seq_lens_ptr,
    seq_lens_sum_ptr,
    req_to_token_ptr,
    out_cache_loc_ptr,
    req_to_token_ptr_stride: tl.constexpr, 
    spec_num_steps: tl.constexpr,
    bs: tl.constexpr,
    bs_upper: tl.constexpr
):
    BLOCK_SIZE: tl.constexpr = 512

    bx = tl.program_id(axis=0)
    # add accept lens to verified lens
    req_idx = tl.load(req_pool_indices_ptr + bx)
    accept_len = tl.load(accept_lengths_ptr + bx)
    verified_len = tl.load(verified_lens_ptr + req_idx)
    seq_len = verified_len + accept_len
    if bx == 0:
        indices = tl.arange(0, bs_upper)
        verified_lens_all = tl.load(verified_lens_ptr + indices, mask=indices < bs, other=0)
        accept_len_all = tl.load(accept_lengths_ptr + indices, mask=indices < bs, other=0)
        tl.store(seq_lens_sum_ptr, tl.sum(verified_lens_all) + tl.sum(accept_len_all))
    tl.store(verified_lens_ptr + req_idx, seq_len)
    # save new seq_len
    tl.store(seq_lens_ptr + bx, seq_len)

    # set out_cache_loc for multi-step draft
    if spec_num_steps > 1:
        new_compute_len = spec_num_steps - 1
        cache_len = seq_len

        cumsum_start = bx * new_compute_len

        # req_idx == 0 means padding position
        if req_idx == 0:
            num_loop = tl.cdiv(new_compute_len, BLOCK_SIZE)
            for i in range(num_loop):
                offset = tl.arange(0, BLOCK_SIZE) + i * BLOCK_SIZE
                mask = offset < new_compute_len
                zero_values = tl.zeros((BLOCK_SIZE,), dtype=tl.int32)
                tl.store(out_cache_loc_ptr + cumsum_start + offset, zero_values, mask=mask)
        else:
            req_to_token_start_loc = req_idx * req_to_token_ptr_stride + cache_len
            num_loop = tl.cdiv(new_compute_len, BLOCK_SIZE)
            for i in range(num_loop):
                offset = tl.arange(0, BLOCK_SIZE) + i * BLOCK_SIZE
                mask = offset < new_compute_len
                data = tl.load(req_to_token_ptr + req_to_token_start_loc + offset, mask=mask)
                tl.store(
                    out_cache_loc_ptr + cumsum_start + offset,
                    data,
                    mask=mask,
                )


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


# copied from sglang: https://github.com/sgl-project/sglang
def traverse_tree(
    retrieve_next_token: torch.Tensor,
    retrieve_next_sibling: torch.Tensor,
    draft_tokens: torch.Tensor,
    grammar: BaseGrammarObject,
    allocate_token_bitmask: torch.Tensor,
):
    """
    Traverse the tree constructed by the draft model to generate the logits mask.
    """
    assert (
        retrieve_next_token.shape == retrieve_next_sibling.shape == draft_tokens.shape
    ), f"retrieve_next_token={retrieve_next_token.shape}, retrieve_next_sibling={retrieve_next_sibling.shape}, draft_tokens={draft_tokens.shape}"

    allocate_token_bitmask.fill_(0)

    def dfs(
        curr: int,
        retrieve_next_token: torch.Tensor,
        retrieve_next_sibling: torch.Tensor,
        parent_pos: int,
    ):
        if curr == 0:
            # the first token generated by the target model, and thus it is always
            # accepted from the previous iteration
            accepted = True
        else:
            parent_bitmask = allocate_token_bitmask[parent_pos]
            curr_token_id = draft_tokens[curr]
            # 32 boolean bitmask values are packed into 32-bit integers
            accepted = (
                parent_bitmask[curr_token_id // 32] & (1 << (curr_token_id % 32))
            ) != 0

        if accepted:
            if curr != 0:
                # Accept the current token
                grammar.accept_token(draft_tokens[curr])
            if not grammar.is_terminated():
                # Generate the bitmask for the current token
                grammar.fill_vocab_mask(allocate_token_bitmask, curr)
                if retrieve_next_token[curr] != -1:
                    # Visit the child node
                    dfs(
                        retrieve_next_token[curr],
                        retrieve_next_token,
                        retrieve_next_sibling,
                        curr,
                    )

            if curr != 0:
                # Rollback the current token
                grammar.rollback(1)

        if retrieve_next_sibling[curr] != -1:
            # Visit the sibling node
            dfs(
                retrieve_next_sibling[curr],
                retrieve_next_token,
                retrieve_next_sibling,
                parent_pos,
            )

    dfs(0, retrieve_next_token, retrieve_next_sibling, -1)


# copied from sglang: https://github.com/sgl-project/sglang
def generate_token_bitmask(
    grammars: List[Union[BaseGrammarObject, None]],
    verify_input: EagleVerifyInput,
    retrieve_next_token_cpu: torch.Tensor,
    retrieve_next_sibling_cpu: torch.Tensor,
    draft_tokens_cpu: torch.Tensor,
    vocab_size: int,
):
    """
    Generate the logit mask for structured output.
    Draft model's token can be either valid or invalid with respect to the grammar.
    We need to perform DFS to figure out:
    1. which tokens are accepted by the grammar
    2. what is the corresponding logit mask.
    """

    draft_tokens_cpu = draft_tokens_cpu.reshape_as(retrieve_next_token_cpu)

    num_draft_tokens = draft_tokens_cpu.shape[-1]

    allocate_token_bitmask = None
    assert len(grammars) == retrieve_next_token_cpu.shape[0]
    outer_grammar = None
    for i, grammar in enumerate(grammars):
        if grammar is not None:
            if allocate_token_bitmask is None:
                allocate_token_bitmask = grammar.allocate_vocab_mask(
                    vocab_size=vocab_size,
                    batch_size=draft_tokens_cpu.numel(),
                    device="cpu",
                )
            outer_grammar = grammar
            traverse_tree(
                retrieve_next_token_cpu[i],
                retrieve_next_sibling_cpu[i],
                draft_tokens_cpu[i],
                grammar,
                allocate_token_bitmask[
                    i * num_draft_tokens : (i + 1) * num_draft_tokens
                ],
            )

    verify_input.grammar = outer_grammar
    return allocate_token_bitmask
