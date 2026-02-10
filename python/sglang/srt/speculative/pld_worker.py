# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
from typing import List, Optional, Tuple

import torch

from sglang.srt.managers.schedule_batch import ModelWorkerBatch
from sglang.srt.model_executor.forward_batch_info import (
    ForwardBatch,
)
from sglang.srt.utils import get_colorful_logger
from sglang.srt.oe_utils import update_token_table
from sglang.srt.speculative.base_spec_worker import BaseSpecDeocdingWorker
from sglang.srt.speculative.pld_cuda_graph_runner import PLDCudaGraphRunner
from flashinfer import ngram_matching

logger = get_colorful_logger(__name__)


class PLDWorker(BaseSpecDeocdingWorker):
    """
    Prompt Lookup Decode worker that uses n-gram matching for speculative decoding.
    Unlike EAGLE, this doesn't require a separate draft model.
    """

    def __init__(
        self,
        server_args,
        gpu_id: int,
        attn_tp_rank: int,
        dp_rank: int,
        nccl_port: int,
        target_worker,
        global_rank: int,
    ):
        super().__init__(
            server_args=server_args,
            gpu_id=gpu_id,
            target_worker=target_worker,
            drafter_use_oe=False,  # PLD is a model-free method
        )
        logger.info(
            f"PLDWorker initialized with prompt_lookup_min={server_args.prompt_lookup_min}, "
            f"prompt_lookup_max={server_args.prompt_lookup_max}, "
            f"speculative_num_draft_tokens={server_args.speculative_num_draft_tokens}, "
            f"speculative_num_steps={server_args.speculative_num_steps}"
        )

        self._init_ngram_buffers()
        self.init_cuda_graphs(graph_runner_cls=PLDCudaGraphRunner)

    def _init_ngram_buffers(self):
        """Initialize buffers for n-gram matching kernel (shared by all code paths)."""

        # max_bs = self.target_worker.model_runner.max_running_requests
        # When get_batch_sizes_to_capture, if default max(capture_bs) > model_runner.req_to_token_pool.size,
        # capture_bs will align to model_runner.req_to_token_pool.size, and req_to_token_pool.size=max_running_request + 1.
        # (See model_runner.py for ReqToTokenPool creation logic)
        max_bs = self.target_worker.model_runner.req_to_token_pool.size
        draft_token_num = self.server_args.speculative_num_draft_tokens
        max_context_len = self.target_worker.model_runner.model_config.context_len

        with torch.device(self.device):
            self.context_tokens = torch.zeros(
                (max_bs, max_context_len), dtype=torch.int32
            )
            self.context_lens = torch.zeros((max_bs,), dtype=torch.int32)
            self.accept_lengths_buffer = torch.zeros((max_bs,), dtype=torch.int32)
            self.verified_tokens_buffer = torch.zeros(
                (max_bs * draft_token_num,), dtype=torch.int32
            )
            self.draft_tokens_output = torch.zeros(
                (max_bs, draft_token_num - 1), dtype=torch.int32
            )
            self.ngram_min_n = self.server_args.prompt_lookup_min
            self.ngram_max_n = self.server_args.prompt_lookup_max

    def forward_decode_spec(
        self, forward_batch: ForwardBatch, vocab_masks: Optional[torch.Tensor] = None
    ):
        if self.use_over_embedding:
            forward_batch.oe_column_starts[: forward_batch.batch_size] = (
                forward_batch.req_to_token_pool.verified_lens[
                    forward_batch.req_pool_indices
                ]
            )
            forward_batch.oe_req_lens[: forward_batch.batch_size] = (
                self.server_args.speculative_num_draft_tokens
            )
            bs = forward_batch.batch_size
            num_verify_tokens = self.server_args.speculative_num_draft_tokens
            forward_batch.oe_out_column_starts[:bs] = (
                forward_batch.req_to_token_pool.verified_lens[
                    forward_batch.req_pool_indices
                ]
            )
            forward_batch.oe_out_req_lens[:bs] = num_verify_tokens
            update_token_table(
                oe_token_table=forward_batch.oe_token_table,
                tokens=forward_batch.input_ids.to(torch.int32),
                row_indices=forward_batch.req_pool_indices,
                column_starts=forward_batch.oe_out_column_starts,
                oe_req_lens=forward_batch.oe_out_req_lens,
            )
        logits_output = self.target_worker.model_runner.forward_extend(
            forward_batch, skip_metadata_init=True
        )
        target_predict, logits_output, accept_length, accept_index = (
            self.rejection_sampling(forward_batch, logits_output, vocab_masks)
        )
        if self.use_over_embedding:
            bs = forward_batch.batch_size
            forward_batch.oe_out_column_starts[:bs] = (
                forward_batch.req_to_token_pool.verified_lens[
                    forward_batch.req_pool_indices
                ]
            )
            forward_batch.oe_out_req_lens[:bs] = accept_length

            update_token_table(
                oe_token_table=forward_batch.oe_token_table,
                tokens=target_predict,
                row_indices=forward_batch.req_pool_indices,
                column_starts=forward_batch.oe_out_column_starts,
                oe_req_lens=forward_batch.oe_out_req_lens,
            )
        new_verified_id = self.preprocess_for_draft_after_decode(
            forward_batch, accept_length, accept_index, target_predict, with_draft_model=False
        )

        output_ids = target_predict[accept_index]

        req_pool_indices = forward_batch.req_pool_indices
        self.target_worker.model_runner.req_to_token_pool.verified_lens[
            req_pool_indices
        ] += accept_length

        token_list = self.propose(
            forward_batch, target_predict, accept_length
        )
        return (
            logits_output,
            output_ids,
            accept_length,
            new_verified_id,
            token_list,
        )

    def forward_prefill_spec(
        self, model_worker_batch: ModelWorkerBatch, forward_batch: ForwardBatch
    ):
        """Handle prefill phase."""
        if self.use_over_embedding:
            forward_batch.oe_column_starts[: forward_batch.batch_size] = (
                forward_batch.extend_prefix_lens
            )
            forward_batch.oe_req_lens[: forward_batch.batch_size] = (
                forward_batch.extend_seq_lens
            )
        target_logits_output = self.target_worker.model_runner.forward_extend(
            forward_batch, skip_metadata_init=True
        )

        next_token_ids = self.target_worker.model_runner.sample(
            target_logits_output, forward_batch
        )

        if model_worker_batch.disagg_set_aux_fn is not None:
            model_worker_batch.disagg_set_aux_fn(next_token_ids, target_logits_output)

        bs = forward_batch.batch_size
        # TODO: _copy_context_to_buffers has performance overhead in large batches, consider optimizing
        self._copy_context_to_buffers(forward_batch, bs)

        req_pool_indices = forward_batch.req_pool_indices
        self.target_worker.model_runner.req_to_token_pool.verified_lens[
            req_pool_indices
        ] += forward_batch.new_tokens_to_compute
        accept_lengths_for_kernel = torch.ones(
            bs, dtype=torch.int32, device=self.device
        )

        # In prefill, next_token_ids is 1D [bs], but _ngram_matching_kernel expects
        # verified_tokens to be in the format that can be indexed by [bs * draft_token_num]
        # For prefill, we only have 1 verified token per request, so we need to reshape it
        # to match the expected format: [token1, pad, pad, ..., token2, pad, pad, ...]
        # where each request occupies draft_token_num slots
        draft_token_num = self.server_args.speculative_num_draft_tokens
        verified_tokens_for_kernel = torch.zeros(
            bs * draft_token_num,
            dtype=next_token_ids.dtype,
            device=next_token_ids.device,
        )
        # Place each next_token_id at the start of its draft_token_num-sized slot
        for i in range(bs):
            verified_tokens_for_kernel[i * draft_token_num] = next_token_ids[i]

        token_list = self.propose(
            forward_batch, verified_tokens_for_kernel, accept_lengths_for_kernel
        )

        return (
            target_logits_output,
            next_token_ids,
            None,  # accept_length (not used in prefill)
            next_token_ids,  # new_verified_id (same as next_token_ids in prefill)
            token_list,
        )

    def forward_idle(self, forward_batch: ForwardBatch):
        logits_output = self.target_worker.model_runner.forward(forward_batch)
        next_token_ids = self.target_worker.model_runner.sample(
            logits_output, forward_batch
        )

        from sglang.srt.speculative.eagle_utils import EagleDraftOutput

        forward_batch.spec_info = EagleDraftOutput(
            last_verified_ids=next_token_ids,
            token_list=[],
        )

        return None, None, None, None, None

    def propose(
        self,
        forward_batch: ForwardBatch,
        verified_tokens: torch.Tensor,
        accept_lengths: torch.Tensor,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """
        Unified n-gram matching using CUDA kernel.
        Returns EAGLE-compatible format: (scores_list, token_list, parents_list)
        """
        bs = forward_batch.batch_size

        self._ngram_matching_kernel(bs, verified_tokens, accept_lengths)

        token_list = self._construct_token_list(bs)
        return token_list

    def _copy_context_to_buffers(self, forward_batch: ForwardBatch, bs: int):
        """
        Copy context information (origin_input_ids + output_ids) to buffers.
        This prepares data for n-gram matching kernel.
        """
        for i, req in enumerate(forward_batch.reqs):
            # Build context: origin_input_ids + output_ids
            context = req.origin_input_ids + req.output_ids
            context_len = len(context)

            if context_len > 0:
                max_len = min(context_len, self.context_tokens.shape[1])
                self.context_tokens[i, :max_len] = torch.tensor(
                    context[:max_len],
                    dtype=torch.int32,
                    device=self.context_tokens.device,
                )
                self.context_lens[i] = max_len
            else:
                self.context_lens[i] = 0

    def _ngram_matching_kernel(
        self,
        bs: int,
        verified_tokens: torch.Tensor,
        accept_lengths: torch.Tensor,
    ):
        """
        N-gram matching CUDA kernel.

        This kernel will:
        1. Update context_tokens and context_lens based on verified tokens
        2. Perform n-gram matching to find draft tokens
        3. Write results to draft_tokens_output

        Args:
            bs: Batch size
            verified_tokens: Verified token IDs from target model
            accept_lengths: Number of accepted tokens per sequence [bs]

        Kernel inputs (from self):
            - context_tokens: [max_bs, max_context_len] - historical tokens
            - context_lens: [max_bs] - length of each context
            - ngram_min_n, ngram_max_n: n-gram matching parameters

        Kernel outputs (to self):
            - draft_tokens_output: [max_bs, draft_token_num-1] - proposed draft tokens
        """
        self.accept_lengths_buffer[:bs].copy_(accept_lengths)

        draft_token_num = self.server_args.speculative_num_draft_tokens
        max_context_len = self.context_tokens.shape[1]
        ngram_min_n = self.ngram_min_n
        ngram_max_n = self.ngram_max_n
        # Verified_tokens is 1D
        num_tokens = min(len(verified_tokens), bs * draft_token_num)
        self.verified_tokens_buffer[:num_tokens].copy_(verified_tokens[:num_tokens])

        ngram_matching(
            self.context_tokens,
            self.context_lens,
            self.verified_tokens_buffer,
            self.accept_lengths_buffer,
            self.draft_tokens_output,
            ngram_min_n,
            ngram_max_n,
            bs,
            draft_token_num,
            max_context_len,
        )

    def _construct_token_list(self, bs: int):
        """
        Construct token_list from draft_tokens_output.

        This converts the 2D draft_tokens_output tensor into the EAGLE-compatible
        list format expected by the rest of the system.

        Args:
            bs: Batch size

        Returns:
            token_list: List of tensors, each [bs, 1]
        """
        fixed_draft_len = self.server_args.speculative_num_draft_tokens - 1
        token_list = []

        for step in range(fixed_draft_len):
            tokens = self.draft_tokens_output[:bs, step].view(bs, 1)
            token_list.append(tokens)

        return token_list

    def preprocess_for_verify(self, forward_batch: ForwardBatch):
        bs = forward_batch.batch_size
        self._copy_context_to_buffers(forward_batch, bs)
        return super().preprocess_for_verify(forward_batch)
