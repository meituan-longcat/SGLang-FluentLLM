import triton
from typing import Optional

import torch

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.schedule_batch import ModelWorkerBatch
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.eagle_utils import (
    fast_topk,
    EagleDraftInput,
    update_oe_metadata,
    update_draft_decode_cache,
    prepare_for_multi_step_draft_kernel,
)
from sglang.srt.speculative.spec_decoding_cuda_graph_runner import (
    SpecDecodeCudaGraphRunner,
)
from sglang.srt.speculative.base_spec_worker import BaseSpecDeocdingWorker
from sglang.srt.oe_utils import update_token_table
from sglang.srt.utils import get_colorful_logger

from flashinfer.sampling import softmax

logger = get_colorful_logger(__name__)


class EAGLEWorker(BaseSpecDeocdingWorker, TpModelWorker):
    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        attn_tp_rank: int,
        moe_ep_rank: Optional[int],
        nccl_port: int,
        target_worker: TpModelWorker,
        global_rank: int,
    ):
        # Do not capture cuda graph in `TpModelWorker.__init__()`
        # We will capture it later
        backup_disable_cuda_graph = server_args.disable_cuda_graph
        server_args.disable_cuda_graph = True

        TpModelWorker.__init__(
            self,
            server_args=server_args,
            gpu_id=gpu_id,
            attn_tp_rank=attn_tp_rank,
            moe_ep_rank=moe_ep_rank,
            global_rank=global_rank,
            nccl_port=nccl_port,
            is_draft_worker=True,
            req_to_token_pool=target_worker.model_runner.req_to_token_pool,
            kv_allocator=target_worker.model_runner.kv_allocator,
            oe_token_table=target_worker.model_runner.oe_token_table,
        )

        BaseSpecDeocdingWorker.__init__(
            self,
            server_args=server_args,
            gpu_id=gpu_id,
            target_worker=target_worker,
            drafter_use_oe=self.use_over_embedding,
        )
        self.init_drafter_embedding(drafter_model_runner=self.model_runner)
        self.model_runner.server_args.disable_cuda_graph = backup_disable_cuda_graph
        self.init_drafter_attention_backends(draft_model_runner=self.model_runner)
        self.init_cuda_graphs(graph_runner_cls=SpecDecodeCudaGraphRunner)

    def forward_target_verify(self, forward_batch: ForwardBatch):
        assert forward_batch.forward_mode.is_target_verify()
        forward_batch.attn_backend = self.target_worker.model_runner.attn_backend
        logits_output = self.target_worker.model_runner.forward_extend(
            forward_batch, skip_metadata_init=True
        )
        return logits_output

    def forward_draft_extend(self, forward_batch: ForwardBatch):
        forward_batch.attn_backend = self.model_runner.attn_backend
        logits_output = self.model_runner.forward_extend(
            forward_batch, skip_metadata_init=True
        )
        self.capture_for_decode(logits_output, forward_batch)

    def prepare_for_multi_step_draft(
        self, forward_batch: ForwardBatch, accept_lengths: torch.Tensor
    ):
        bs = forward_batch.batch_size
        out_cache_loc_for_draft_decode = torch.empty(
            size=(bs * (self.speculative_num_steps - 1),),
            dtype=torch.int32,
            device=self.device,
        )
        seq_lens = torch.empty(bs, dtype=torch.int32, device=self.device)
        seq_lens_sum = torch.empty(1, dtype=torch.int32, device=self.device)
        prepare_for_multi_step_draft_kernel[(bs,)](
            out_cache_loc_ptr=out_cache_loc_for_draft_decode,
            verified_lens_ptr=self.req_to_token_pool.verified_lens,
            req_pool_indices_ptr=forward_batch.req_pool_indices,
            accept_lengths_ptr=accept_lengths,
            seq_lens_ptr=seq_lens,
            seq_lens_sum_ptr=seq_lens_sum,
            req_to_token_ptr=self.req_to_token_pool.req_to_token,
            req_to_token_ptr_stride=self.req_to_token_pool.req_to_token.shape[1],
            spec_num_steps=self.speculative_num_steps,
            bs=bs,
            bs_upper=triton.next_power_of_2(bs),
        )
        if self.speculative_num_steps > 1:
            forward_batch.seq_lens = seq_lens
            forward_batch.seq_lens_sum = seq_lens_sum
            forward_batch.positions = seq_lens
            forward_batch.capture_hidden_mode = CaptureHiddenMode.LAST
            forward_batch.forward_mode = ForwardMode.DECODE
            forward_batch.out_cache_loc = out_cache_loc_for_draft_decode

    def propose(self, forward_batch: ForwardBatch, accept_lengths: torch.Tensor):
        forward_batch.attn_backend = self.model_runner.attn_backend

        self.forward_draft_extend(forward_batch)
        self.prepare_for_multi_step_draft(forward_batch, accept_lengths)
        token_list = self.draft(forward_batch)
        return token_list

    def prepare_for_draft_prefill(
        self,
        forward_batch: ForwardBatch,
        target_logits_output: LogitsProcessorOutput,
        next_token_ids: torch.Tensor,
    ):
        forward_batch.forward_mode = ForwardMode.EXTEND
        if self.use_over_embedding:
            forward_batch.oe_column_starts[: forward_batch.batch_size] = (
                forward_batch.extend_prefix_lens + 1
            )
            forward_batch.oe_req_lens[: forward_batch.batch_size] = (
                forward_batch.extend_seq_lens
            )
        forward_batch.capture_hidden_mode = CaptureHiddenMode.LAST
        forward_batch.spec_info = EagleDraftInput(
            hidden_states=target_logits_output.hidden_states,
            verified_id=next_token_ids,
        )
        forward_batch.spec_info.set_input_ids(forward_batch)
        forward_batch.token_to_kv_pool = self.model_runner.token_to_kv_pool
        forward_batch.attn_backend = self.model_runner.attn_backend

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
        logits_output = self.forward_target_verify(forward_batch)
        target_predict, logits_output, accept_length, accept_index = (
            self.rejection_sampling(forward_batch, logits_output, vocab_masks)
        )
        # Results from target_predict, verify_lens not updated yet, just write at current verify length + 1 and continue
        if self.use_over_embedding:
            forward_batch.oe_out_column_starts[: forward_batch.batch_size] = (
                self.req_to_token_pool.verified_lens[forward_batch.req_pool_indices] + 1
            )
            forward_batch.oe_out_req_lens[: forward_batch.batch_size] = (
                self.server_args.speculative_num_draft_tokens
            )
            update_token_table(
                oe_token_table=forward_batch.oe_token_table,
                tokens=target_predict,
                row_indices=forward_batch.req_pool_indices,
                column_starts=forward_batch.oe_out_column_starts,
                oe_req_lens=forward_batch.oe_out_req_lens,
            )
        new_verified_id = self.preprocess_for_draft_after_decode(
            forward_batch, accept_length, accept_index, target_predict, with_draft_model=True
        )
        token_list = self.propose(forward_batch, accept_length)
        output_ids = target_predict[accept_index]
        return (logits_output, output_ids, accept_length, new_verified_id, token_list)

    def forward_prefill_spec(
        self, model_worker_batch: ModelWorkerBatch, forward_batch: ForwardBatch
    ):
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
        self.prepare_for_draft_prefill(
            forward_batch, target_logits_output, next_token_ids
        )
        token_list = self.propose(forward_batch, forward_batch.new_tokens_to_compute)
        return (
            target_logits_output,
            next_token_ids,
            None,
            next_token_ids,
            token_list,
        )

    def forward_idle(self, forward_batch: ForwardBatch):
        assert forward_batch.forward_mode.is_idle()
        forward_batch.capture_hidden_mode = CaptureHiddenMode.LAST
        logits_output = self.target_worker.model_runner.forward(forward_batch)
        next_token_ids = self.target_worker.model_runner.sample(
            logits_output, forward_batch
        )
        forward_batch.spec_info = EagleDraftInput(
            hidden_states=logits_output.hidden_states,
            verified_id=next_token_ids,
        )
        self.model_runner.forward_idle(forward_batch)
        for _ in range(self.speculative_num_steps - 1):
            self.model_runner.forward_idle(forward_batch)
        return None, None, None, None, None

    def draft(self, forward_batch: ForwardBatch):
        # Initialize attention backend
        if not forward_batch.forward_mode.is_idle() and self.speculative_num_steps > 1:
            self.draft_attn_backend.init_forward_metadata(forward_batch)
        # Run forward steps
        token_list = self.draft_forward(forward_batch)
        return token_list

    def draft_forward(self, forward_batch: ForwardBatch):
        # Parse args
        spec_info = forward_batch.spec_info
        # out_cache_loc here:
        # <-- req 1 --> <-- req 2 --> <-- req 3 --> .....
        # [step1, step2, step1, step2, step1, step2]
        # Need to select step-wise cache loc when doing multi-step decode
        out_cache_loc = forward_batch.out_cache_loc
        if self.server_args.enable_dp_attention:
            forward_batch.global_num_tokens = forward_batch.global_batch_size
        _, topk_index, hidden_states = (
            spec_info.topk_p,
            spec_info.topk_index,
            spec_info.hidden_states,
        )
        if self.hot_token_id is not None:
            topk_index = self.hot_token_id[topk_index]

        # Return values
        token_list: torch.Tensor = torch.empty(
            (forward_batch.batch_size, self.server_args.speculative_num_steps),
            dtype=torch.int32,
            device="cuda",
        )

        # Forward multiple steps
        for i in range(self.speculative_num_steps):
            input_ids = topk_index.flatten()
            if self.use_over_embedding:
                update_oe_metadata(forward_batch, i, self.speculative_num_steps)
                # OE needs to update token_table and corresponding table_column_starts and req_lens
                update_token_table(
                    forward_batch.oe_token_table,
                    input_ids.to(torch.int32),
                    forward_batch.req_pool_indices,
                    forward_batch.oe_out_column_starts,
                    forward_batch.oe_out_req_lens,
                )
            token_list[:, i] = input_ids

            # we don't need to run the last forward. we get 1 token from draft prefill and (#spec steps - 1) tokens here
            if i == self.speculative_num_steps - 1:
                break

            # Set inputs
            forward_batch.input_ids = input_ids.to(torch.int32)
            update_draft_decode_cache(
                out_cache_loc=out_cache_loc,
                forward_batch=forward_batch,
                draft_decode_step=i,
                speculative_num_steps=self.speculative_num_steps,
            )
            if self.drafter_backend == "flashinfer":
                forward_batch.attn_backend = self.draft_attn_backend
                forward_batch.attn_backend.set_draft_step(i)
            else:
                forward_batch.attn_backend = self.draft_attn_backend.attn_backends[i]
            spec_info.hidden_states = hidden_states
            logits_output = self.model_runner.model.forward(
                forward_batch.input_ids, forward_batch.positions, forward_batch
            )
            forward_batch.positions.add_(1)
            probs = softmax(logits_output.next_token_logits)
            # Get topk tokens for next position
            _, topk_index = fast_topk(probs, self.topk, dim=-1)
            if self.hot_token_id is not None:
                topk_index = self.hot_token_id[topk_index]

            # Update last hidden_states
            hidden_states = logits_output.hidden_states

        return token_list

    def capture_for_decode(
        self, logits_output: LogitsProcessorOutput, forward_batch: ForwardBatch
    ):
        probs = softmax(logits_output.next_token_logits)
        spec_info = forward_batch.spec_info
        spec_info.topk_p, spec_info.topk_index = fast_topk(probs, self.topk, dim=-1)
        spec_info.hidden_states = logits_output.hidden_states
