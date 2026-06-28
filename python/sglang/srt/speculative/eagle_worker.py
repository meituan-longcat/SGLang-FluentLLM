import functools
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
from sglang.srt.utils import is_npu,get_colorful_logger,MultiprocessingSerializer

__is_npu__ = is_npu()
if __is_npu__:
    softmax = functools.partial(torch.softmax, dim=-1)
else:
    from flashinfer.sampling import softmax

from sglang.srt.managers.io_struct import UpdateWeightsFromTensorReqInput
from sglang.srt.patch_torch import monkey_patch_torch_reductions
from sglang.srt.model_executor.weight_mixin import unwrap_ipc_tensors

logger = get_colorful_logger(__name__)
from sglang.srt.oe_utils import update_token_table

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
            is_multi_head_eagle=server_args.is_multi_head_eagle
        )

        BaseSpecDeocdingWorker.__init__(
            self,
            server_args=server_args,
            gpu_id=gpu_id,
            target_worker=target_worker,
            drafter_use_oe=server_args.draft_use_oe
        )

        assert hasattr(self, "model_runner_list")
        for runner in self.model_runner_list:
            self.init_drafter_embedding(drafter_model_runner=runner)

        self.model_runner.server_args.disable_cuda_graph = backup_disable_cuda_graph
        if (not self.is_multi_head_eagle) and self.speculative_num_steps > 1:
            self.init_drafter_attention_backends(draft_model_runner=self.model_runner)
        self.init_cuda_graphs(graph_runner_cls=SpecDecodeCudaGraphRunner)

    def forward_target_verify(self, forward_batch: ForwardBatch):
        assert forward_batch.forward_mode.is_target_verify()
        forward_batch.attn_backend = self.target_worker.model_runner.attn_backend
        if __is_npu__:
            logits_output = self.target_worker.model_runner.forward(
                forward_batch
            )
        else:
            logits_output = self.target_worker.model_runner.forward_extend(
                forward_batch, skip_metadata_init=True
            )
        return logits_output

    def forward_draft_extend(self, forward_batch: ForwardBatch):
        forward_batch.attn_backend = self.model_runner.attn_backend
        if __is_npu__:
            can_npu_graph = (
                self.npu_graph_runner_for_draft_extend
                and self.npu_graph_runner_for_draft_extend.can_run(forward_batch)
            )
            if can_npu_graph:
                logits_output = self.npu_graph_runner_for_draft_extend.replay(forward_batch)
            else:
                logits_output = self.model_runner.forward(
                    forward_batch
                )
        else:
            logits_output = self.model_runner.forward_extend(
                forward_batch, skip_metadata_init=True
            )
        self.capture_for_decode(logits_output, forward_batch)

    def prepare_for_multi_step_draft(
        self, forward_batch: ForwardBatch, accept_lengths: torch.Tensor
    ):
        if __is_npu__:
            self.req_to_token_pool.verified_lens[
                forward_batch.req_pool_indices
            ] += accept_lengths
            num_seqs = forward_batch.batch_size
            if self.speculative_num_steps > 1:
                verified_lens = self.req_to_token_pool.verified_lens[
                    forward_batch.req_pool_indices
                ]
                forward_batch.seq_lens = verified_lens
                forward_batch.seq_lens_sum = verified_lens.sum()
                forward_batch.positions = verified_lens.repeat_interleave(self.topk, dim=0)
                forward_batch.new_tokens_total = self.speculative_num_steps * num_seqs
                forward_batch.new_tokens_to_compute.fill_(self.speculative_num_steps - 1)
                forward_batch.capture_hidden_mode = CaptureHiddenMode.LAST
                forward_batch.forward_mode = ForwardMode.DECODE
                forward_batch.set_out_cache_loc()
                forward_batch.attn_metadata.seq_lens_tensor += accept_lengths
        else:
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

    @torch.inference_mode()
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
        if __is_npu__:
            target_logits_output = self.target_worker.model_runner.forward(
                forward_batch
            )
        else:
            target_logits_output = self.target_worker.model_runner.forward_extend(
                forward_batch, skip_metadata_init=True
            )

        if self.pp_group.is_last_rank:
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
        else:
            return (
                target_logits_output,
                None,
                None,
                None,
                None
            )

    def forward_idle(self, forward_batch: ForwardBatch):
        assert forward_batch.forward_mode.is_idle()
        forward_batch.capture_hidden_mode = CaptureHiddenMode.LAST
        logits_output = self.target_worker.model_runner.forward(forward_batch)
        if self.pp_group.is_last_rank:
            next_token_ids = self.target_worker.model_runner.sample(
                logits_output, forward_batch
            )
            forward_batch.spec_info = EagleDraftInput(
                hidden_states=logits_output.hidden_states,
                verified_id=next_token_ids,
            )
            self.model_runner.forward_idle(forward_batch)
            for _ in range(self.speculative_num_steps - 1):
                if self.server_args.enable_dp_attention and forward_batch.batch_size == 0:
                    # Empty DP ranks need to follow draft decode collectives after the first idle/prefill participation
                    forward_batch.global_num_tokens = forward_batch.global_batch_size
                    self.model_runner.forward_idle(forward_batch)
                else:
                    self.model_runner.forward_idle(forward_batch)
        return logits_output, None, None, None, None

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
            device=self.device,
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
            if __is_npu__:
                # update slot_mapping, actual_q_len, actual_kv_len
                forward_batch.attn_metadata.slot_mapping = forward_batch.out_cache_loc.clone().to(torch.int64)
                forward_batch.attn_metadata.seq_lens_tensor = (forward_batch.positions + 1).to(torch.int64)
                # incre len fix to 1
                forward_batch.attn_metadata.query_len_tensor = torch.arange(1, forward_batch.batch_size + 1, 1, dtype=torch.int64, device='npu')
                forward_batch.set_npu_ep_metadata(self.target_worker.model_runner)
                can_npu_graph = (
                    self.npu_graph_runner_for_draft_extend
                    and self.npu_graph_runner_for_draft_extend.can_run(forward_batch)
                )
                if can_npu_graph:
                    logits_output = self.npu_graph_runner_for_draft_extend.replay(forward_batch)
                else:
                    logits_output = self.model_runner.model.forward(
                        forward_batch.input_ids, forward_batch.positions, forward_batch
                    )
            else:
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

    def multi_batch_select(self, max_input_batch, compile_bs_list):
        max_compile_bs = max(compile_bs_list)
        assert max_input_batch <= max_compile_bs, f"max input batch ({max_input_batch}) should less equal than max compile bs({max_compile_bs}) in graph mode"
        return min(bs for bs in compile_bs_list if bs >= max_input_batch)

    def update_weights_from_tensor(self, recv_req: UpdateWeightsFromTensorReqInput):
        monkey_patch_torch_reductions()
        named_tensors = MultiprocessingSerializer.deserialize(
            recv_req.serialized_named_tensors[self.global_rank]
        )
        unwrapped_tensors = unwrap_ipc_tensors(
            named_tensors, self.global_rank, torch.device(self.device)
        )
        success, message = self.model_runner.update_weights_from_tensor(
            named_tensors=unwrapped_tensors,
            load_format=recv_req.load_format,
        )
        if not success:
            return success, message
        success, message = self.target_worker.model_runner.update_weights_from_tensor(
            named_tensors=unwrapped_tensors,
            load_format=recv_req.load_format,
        )
        return success, message
