from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import torch
import triton
import triton.language as tl

from sglang.srt.constrained.base_grammar_backend import (
    BaseGrammarObject,
    degrade_grammars_by_consensus,
)
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
from sglang.srt.sampling.penaltylib.repetition_penalty import apply_scaling_penalties
from sglang.srt.utils import get_colorful_logger, is_npu
from sglang.srt.utils.common import get_device

__is_npu__ = is_npu()

logger = get_colorful_logger(__name__)

if __is_npu__:
    from sglang.srt.speculative.npu_eagle_utils import (
        npu_create_extend_spec_info,
        npu_verify,
    )
else:
    from sglang.srt.speculative.gpu_eagle_utils import (
        gpu_verify,
        update_oe_metadata_kernel,
    )


if TYPE_CHECKING:
    from sglang.srt.layers.sampler.logits_processor import LogitsProcessorOutput
    from sglang.srt.managers.scheduling.schedule_batch import ModelWorkerBatch
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


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
    accept_length_cpu: list[int] = None
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
        if not is_npu():
            # Fuse the Python loop + torch.where into a single Triton kernel.
            # torch.cat on a list of GPU tensors is a pure GPU op (no sync).
            flat_draft = torch.cat(forward_batch.draft_input_ids)
            seq_lens_gpu = forward_batch.extend_seq_lens
            cumsum = torch.zeros(
                len(forward_batch.draft_input_ids) + 1,
                dtype=torch.int32,
                device=seq_lens_gpu.device,
            )
            cumsum[1:] = torch.cumsum(seq_lens_gpu, dim=0)
            bs = len(forward_batch.draft_input_ids)
            _set_input_ids_kernel[(bs,)](
                flat_draft,
                self.verified_id,
                forward_batch.input_ids,
                cumsum,
                bs,
            )
        else:
            pt = 0
            for i, extend_seq_len in enumerate(forward_batch.extend_seq_lens):
                input_ids = forward_batch.draft_input_ids[i]
                if input_ids[-1] == -1:
                    input_ids[-1] = self.verified_id[i]
                forward_batch.input_ids[pt : pt + extend_seq_len] = input_ids

                pt += extend_seq_len

    def prepare_extend_after_decode(self, forward_batch: ForwardBatch, use_oe: bool = False):
        new_verified_id = torch.empty_like(self.accept_length, dtype=torch.int32)
        if __is_npu__:
            npu_create_extend_spec_info(self.verified_id, self.accept_index, self.accept_length, new_verified_id)
        else:
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
                use_oe,
            )

        forward_batch.input_ids = self.verified_id
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
        self.hidden_states = torch.cat([self.hidden_states, spec_info.hidden_states], axis=0)
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
    token_list: torch.Tensor | list | None

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
        self.last_verified_ids = torch.cat([self.last_verified_ids, spec_info.last_verified_ids])
        if isinstance(self.token_list, torch.Tensor):
            self.token_list = torch.cat([self.token_list, spec_info.token_list], dim=0)
        elif isinstance(self.token_list, list):
            self.token_list = [torch.cat([s1, s2], axis=0) for s1, s2 in zip(self.token_list, spec_info.token_list)]
        else:
            raise RuntimeError(f"Not supported token_list type, {self.token_list=}")


@dataclasses.dataclass
class EagleVerifyInput:
    draft_token: torch.Tensor
    positions: torch.Tensor
    draft_token_num: int
    spec_steps: int  # TODO: 废弃tree模式后，此参数不再有意义
    capture_hidden_mode: CaptureHiddenMode
    is_all_greedy: bool
    grammar: BaseGrammarObject = None
    cumulated_scaling_penalties: torch.Tensor = None

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
                torch.empty(0, dtype=torch.int32, device=get_device()),
                torch.empty(0, dtype=torch.int32, device=get_device()),
                0,
                spec_steps,
                CaptureHiddenMode.LAST,
                True,
            )
        else:
            draft_tokens = torch.cat((verified_id.unsqueeze(1), token_list), dim=1).flatten()
            positions = (seq_lens.unsqueeze(1) + torch.arange(num_verify_tokens, device=draft_tokens.device)).flatten()
            return cls(draft_tokens, positions, num_verify_tokens, spec_steps, CaptureHiddenMode.FULL, is_all_greedy)

    def prepare_for_verify(self, model_worker_batch: ModelWorkerBatch):
        model_worker_batch.input_ids = self.draft_token
        model_worker_batch.capture_hidden_mode = CaptureHiddenMode.FULL

    def verify(
        self,
        forward_batch: ForwardBatch,
        logits_output: LogitsProcessorOutput,
        vocab_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:

        bs = forward_batch.batch_size
        candidates = self.draft_token.reshape(bs, self.draft_token_num)
        apply_scaling_penalties(
            logits_output.next_token_logits, torch.repeat_interleave(forward_batch.spec_info.cumulated_scaling_penalties, self.draft_token_num, dim=0)
        )

        if vocab_mask is not None:
            assert self.grammar is not None
            self.grammar.apply_vocab_mask(logits=logits_output.next_token_logits, vocab_mask=vocab_mask)

        if __is_npu__:
            predicts, accept_length, rearranged_accept_index = npu_verify(self, forward_batch, logits_output, candidates, bs)
        else:
            predicts, accept_length, rearranged_accept_index = gpu_verify(self, forward_batch, logits_output, candidates, bs)

        return predicts, logits_output, accept_length, rearranged_accept_index


@triton.jit
def _set_input_ids_kernel(
    flat_draft_ptr,  # [total_tokens], int32 – concatenated draft_input_ids
    verified_id_ptr,  # [batch_size], int64 – verified token for each request
    output_ids_ptr,  # [total_tokens], int32 – forward_batch.input_ids to write
    cumsum_ptr,  # [batch_size + 1], int32 – prefix-sum of seq lens
    batch_size: int,
):
    """Copy draft tokens into output_ids, replacing the last token with
    verified_id when it equals -1.  One program per request."""
    BLOCK_SIZE: tl.constexpr = 512
    pid = tl.program_id(0)
    if pid >= batch_size:
        return

    start = tl.load(cumsum_ptr + pid)
    end = tl.load(cumsum_ptr + pid + 1)
    seq_len = end - start
    verified = tl.load(verified_id_ptr + pid)

    num_loop = tl.cdiv(seq_len, BLOCK_SIZE)
    for i in range(num_loop):
        offsets = tl.arange(0, BLOCK_SIZE) + i * BLOCK_SIZE
        mask = offsets < seq_len
        tokens = tl.load(flat_draft_ptr + start + offsets, mask=mask)
        # Replace the sentinel -1 at the last position with verified_id
        is_last = offsets == (seq_len - 1)
        tokens = tl.where(is_last & (tokens == -1), verified, tokens)
        tl.store(output_ids_ptr + start + offsets, tokens, mask=mask)


def update_oe_metadata(forward_batch: ForwardBatch, draft_decode_step: int, spec_num_steps: int):
    if __is_npu__:
        verified_lens = forward_batch.req_to_token_pool.verified_lens[forward_batch.req_pool_indices]
        forward_batch.oe_out_column_starts[: forward_batch.batch_size] = verified_lens + 1 + draft_decode_step
        forward_batch.oe_out_req_lens[: forward_batch.batch_size] = 1
        if draft_decode_step < spec_num_steps - 1:
            forward_batch.oe_column_starts[: forward_batch.batch_size] = verified_lens + 1 + draft_decode_step
            forward_batch.oe_req_lens[: forward_batch.batch_size] = 1
    else:
        bs = forward_batch.batch_size
        update_oe_metadata_kernel[(bs,)](
            oe_out_column_starts_ptr=forward_batch.oe_out_column_starts,
            oe_column_starts_ptr=forward_batch.oe_column_starts,
            oe_out_req_lens_ptr=forward_batch.oe_out_req_lens,
            oe_req_lens_ptr=forward_batch.oe_req_lens,
            verified_len_ptr=forward_batch.req_to_token_pool.verified_lens,
            req_pool_indices_ptr=forward_batch.req_pool_indices,
            draft_decode_step=draft_decode_step,
            spec_num_steps=spec_num_steps,
        )


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


def update_draft_decode_cache(forward_batch: ForwardBatch, out_cache_loc: torch.Tensor, draft_decode_step: int, speculative_num_steps: int):
    bs = forward_batch.batch_size
    stride = speculative_num_steps - 1
    if __is_npu__:
        indices = torch.arange(bs, device=out_cache_loc.device) * stride + draft_decode_step
        forward_batch.out_cache_loc = out_cache_loc[indices]
        return

    out_cache_loc_new = torch.empty(bs, dtype=out_cache_loc.dtype, device=out_cache_loc.device)
    grid = (bs,)
    update_draft_decode_cache_kernel[grid](
        out_cache_loc_ptr=out_cache_loc,
        out_cache_loc_out_ptr=out_cache_loc_new,
        draft_decode_step=draft_decode_step,
        stride=stride,
        batch_size=bs,
    )
    forward_batch.out_cache_loc = out_cache_loc_new


@triton.jit
def create_extend_spec_info(
    verified_id,  # padded verified id
    new_verified_id,
    accept_length_ptr,
    oe_column_starts_ptr,
    oe_req_lens_ptr,
    req_pool_indices_ptr,
    verified_lens_ptr,
    spec_num_tokens: int,
    batch_size: int,
    use_oe: tl.constexpr,
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


def fast_topk(values, topk, dim):
    if topk == 1:
        # Use max along the specified dimension to get both value and index
        max_value, max_index = torch.max(values, dim=dim)
        return max_value.unsqueeze(1), max_index.unsqueeze(1)
    else:
        # Use topk for efficiency with larger k values
        return torch.topk(values, topk, dim=dim)


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
    bs_upper: tl.constexpr,
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
    assert retrieve_next_token.shape == retrieve_next_sibling.shape == draft_tokens.shape, (
        f"retrieve_next_token={retrieve_next_token.shape}, retrieve_next_sibling={retrieve_next_sibling.shape}, draft_tokens={draft_tokens.shape}"
    )

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
            accepted = (parent_bitmask[curr_token_id // 32] & (1 << (curr_token_id % 32))) != 0

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
    grammars: list[BaseGrammarObject | None],
    verify_input: EagleVerifyInput,
    retrieve_next_token_cpu: torch.Tensor,
    retrieve_next_sibling_cpu: torch.Tensor,
    draft_tokens_cpu: torch.Tensor,
    vocab_size: int,
    timeout_ms: float = 0.0,
):
    """
    Generate the logit mask for structured output.
    Draft model's token can be either valid or invalid with respect to the grammar.
    We need to perform DFS to figure out:
    1. which tokens are accepted by the grammar
    2. what is the corresponding logit mask.

    ``timeout_ms`` > 0 is a per-request budget for one draft tree. Overruns are
    only proposals: they are agreed on across the attention-TP group before any
    request is degraded, so every rank verifies with the same mask. Must be
    called by every rank of the group for the same batch.
    """

    draft_tokens_cpu = draft_tokens_cpu.reshape_as(retrieve_next_token_cpu)

    num_draft_tokens = draft_tokens_cpu.shape[-1]
    time_threshold_s = timeout_ms / 1000.0 if timeout_ms and timeout_ms > 0 else -1.0

    # traverse_draft_tree (xgrammar native path) requires int64 CPU tensors; the
    # Python fallback is dtype-agnostic. .to(int64) is a no-op when already int64.
    retrieve_next_token_cpu = retrieve_next_token_cpu.to(torch.int64)
    retrieve_next_sibling_cpu = retrieve_next_sibling_cpu.to(torch.int64)
    draft_tokens_cpu = draft_tokens_cpu.to(torch.int64)

    allocate_token_bitmask = None
    assert len(grammars) == retrieve_next_token_cpu.shape[0]
    outer_grammar = None
    overrun = [False] * len(grammars)
    for i, grammar in enumerate(grammars):
        if grammar is not None:
            if allocate_token_bitmask is None:
                allocate_token_bitmask = grammar.allocate_vocab_mask(
                    vocab_size=vocab_size,
                    batch_size=draft_tokens_cpu.numel(),
                    device="cpu",
                )
            outer_grammar = grammar
            # xgrammar's native traversal can stop in the middle of a
            # pathological tree once the per-request time budget is exceeded;
            # the Python DFS fallback checks the budget between nodes.
            overrun[i] = not grammar.fill_draft_tree_bitmask(
                retrieve_next_token_cpu[i],
                retrieve_next_sibling_cpu[i],
                draft_tokens_cpu[i],
                allocate_token_bitmask[i * num_draft_tokens : (i + 1) * num_draft_tokens],
                time_threshold_s=time_threshold_s,
            )

    if time_threshold_s > 0 and allocate_token_bitmask is not None:
        agreed = degrade_grammars_by_consensus(grammars, overrun, "draft tree mask budget exceeded")
        for i, grammar in enumerate(grammars):
            if not agreed[i] or grammar is None:
                continue
            rows = allocate_token_bitmask[i * num_draft_tokens : (i + 1) * num_draft_tokens]
            if getattr(grammar, "degraded", False):
                # Cheap now that the grammar is degraded: allow-all for the
                # constrained part, while a reasoner keeps its thinking-budget
                # mask. Same result on every rank.
                grammar.fill_draft_tree_bitmask(
                    retrieve_next_token_cpu[i],
                    retrieve_next_sibling_cpu[i],
                    draft_tokens_cpu[i],
                    rows,
                )
            else:
                rows.fill_(-1)

    verify_input.grammar = outer_grammar
    return allocate_token_bitmask


@triton.jit
def cumulate_output_tokens_kernel(
    output_ids_ptr,  # [batch_size * tokens_per_request]
    accept_lens_ptr,  # [batch_size]
    req_pool_indices_ptr,  # [batch_size]
    cumulated_penalty_ptr,  # [pool_size, vocab_size]
    scaling_penalties_ptr,  # [batch_size]
    tokens_per_request: tl.constexpr,
    vocab_size: tl.constexpr,
):
    """
    Triton kernel to cumulate output tokens for penalty calculation.

    Each program handles one request in the batch.
    For each request, we iterate through the accepted tokens and update the penalty matrix.
    """
    pid = tl.program_id(0)
    accept_len = tl.load(accept_lens_ptr + pid)
    pool_idx = tl.load(req_pool_indices_ptr + pid)
    output_start = pid * tokens_per_request
    scaling_penalty = tl.load(scaling_penalties_ptr + pid)

    for i in range(tokens_per_request):
        # 1 is for bouns token
        if i < accept_len + 1:
            # Load the token id
            token_id = tl.load(output_ids_ptr + output_start + i)

            # Calculate the position in the penalty matrix
            penalty_offset = pool_idx * vocab_size + token_id
            tl.store(cumulated_penalty_ptr + penalty_offset, scaling_penalty)


def cumulate_output_tokens_pytorch(
    output_ids,  # [batch_size * tokens_per_request]
    accept_lens,  # [batch_size]
    req_pool_indices,  # [batch_size]
    cumulated_penalty,  # [pool_size + 1, vocab_size]; last row is a dummy sink row
    scaling_penalties,  # [batch_size]
    tokens_per_request,
    valid_batch_size=None,
):
    """
    D2H-free implementation.

    Boolean-mask indexing (tensor[bool_mask]) internally calls nonzero(), whose
    output size must be known on the host, forcing a device-to-host sync on every
    loop iteration. Instead, we build static [batch_size, tokens_per_request]
    indices and redirect invalid (rejected/padding) positions to a dummy sink row
    at the end of cumulated_penalty, so everything stays on device.
    """
    device = output_ids.device
    dummy_row = cumulated_penalty.shape[0] - 1

    valid_lens = accept_lens + 1  # +1 for the bonus token

    # Flat static validity mask of shape [batch_size * tokens_per_request]
    # (element-wise ops only, no nonzero -> no D2H sync)
    positions = torch.arange(tokens_per_request, device=device)
    valid_mask = (positions.unsqueeze(0) < valid_lens.unsqueeze(1)).reshape(-1)

    # Whole graph-padding rows are invalid even if verify produced a bonus.
    # Their pool indices may alias a live request; never write them there.
    if valid_batch_size is not None:
        real_rows = torch.arange(accept_lens.shape[0], device=device) < valid_batch_size
        valid_mask = valid_mask & real_rows.repeat_interleave(tokens_per_request)

    # Redirect invalid positions to the dummy row. Rejected/padding slots in
    # output_ids are guaranteed to be 0 by npu verify, so cols stay in bounds.
    rows = req_pool_indices.repeat_interleave(tokens_per_request)
    rows = rows.masked_fill(~valid_mask, dummy_row)
    cols = output_ids
    values = scaling_penalties.repeat_interleave(tokens_per_request)

    cumulated_penalty.index_put_((rows.long(), cols.long()), values)
