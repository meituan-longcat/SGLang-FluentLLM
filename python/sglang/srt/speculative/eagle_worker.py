import functools
from typing import Optional

import torch
import triton

from sglang.srt.configs.model_config import dsa_mtp_is_shared_indexer
from sglang.srt.env import global_server_args_dict
from sglang.srt.layers.sampler.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.scheduling.multimodal_data import (
    resolve_mtp_mm_placeholder_ids,
    substitute_mm_pad_values_for_mtp,
)
from sglang.srt.managers.scheduling.schedule_batch import ModelWorkerBatch
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.base_spec_worker import BaseSpecDeocdingWorker
from sglang.srt.speculative.eagle_utils import (
    EagleDraftInput,
    fast_topk,
    prepare_for_multi_step_draft_kernel,
    update_draft_decode_cache,
    update_oe_metadata,
)
from sglang.srt.speculative.npu_eagle_utils import _pad_to_size
from sglang.srt.speculative.spec_decoding_cuda_graph_runner import (
    SpecDecodeCudaGraphRunner,
)
from sglang.srt.utils import MultiprocessingSerializer, get_colorful_logger, is_npu

__is_npu__ = is_npu()
if __is_npu__:
    softmax = functools.partial(torch.softmax, dim=-1)
else:
    from flashinfer.sampling import softmax

from sglang.srt.managers.io_struct import UpdateWeightsFromTensorReqInput
from sglang.srt.model_executor.weight_mixin import unwrap_ipc_tensors
from sglang.srt.utils.oe_utils import update_token_table
from sglang.srt.utils.patch_torch import monkey_patch_torch_reductions

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
            speculative_sampling_info=target_worker.model_runner.speculative_sampling_info,
            is_multi_head_eagle=server_args.is_multi_head_eagle,
        )

        BaseSpecDeocdingWorker.__init__(
            self, server_args=server_args, gpu_id=gpu_id, target_worker=target_worker, drafter_use_oe=server_args.draft_use_oe
        )
        target_model_config = target_worker.model_runner.model_config
        target_hf_config = target_model_config.hf_config
        self.mtp_mm_placeholder_ids = resolve_mtp_mm_placeholder_ids(target_hf_config)
        if target_model_config.is_multimodal and not self.mtp_mm_placeholder_ids:
            raise ValueError("The multimodal target model does not configure any in-vocabulary placeholder token ID for the MTP drafter")
        target_vocab_size = int(target_model_config.vocab_size)
        invalid_ids = {
            modality.name: token_id for modality, token_id in self.mtp_mm_placeholder_ids.items() if token_id < 0 or token_id >= target_vocab_size
        }
        if invalid_ids:
            raise ValueError(f"MTP multimodal placeholder IDs must be inside the target vocabulary (vocab_size={target_vocab_size}): {invalid_ids}")

        if __is_npu__:
            # Run post-verify grammar updates (verified-token D2H copy sync +
            # CPU grammar accept loop) in a dedicated thread so that the
            # forward thread is not blocked by them.
            self.start_grammar_update_thread()

        assert hasattr(self, "model_runner_list")
        for runner in self.model_runner_list:
            self.init_drafter_embedding(drafter_model_runner=runner)

        self.index_share_for_mtp_iteration = dsa_mtp_is_shared_indexer(self.model_runner.model_config.hf_config)

        self.model_runner.server_args.disable_cuda_graph = backup_disable_cuda_graph
        if (not self.is_multi_head_eagle) and self.speculative_num_steps > 1:
            self.init_drafter_attention_backends(draft_model_runner=self.model_runner)
        # Build the grammar overlap machinery BEFORE cuda graph capture: the
        # runner references these worker-owned objects, and they must exist even
        # when cuda graph is disabled (no runner is created then).
        if not __is_npu__:
            self._init_grammar_overlap_for_cuda_graph()
            self.init_cuda_graphs(graph_runner_cls=SpecDecodeCudaGraphRunner)

    def set_hicache_consumer(self, consumer_index: int):
        self.target_worker.set_hicache_consumer(consumer_index)
        super().set_hicache_consumer(consumer_index)

    def _masked_update_token_table(self, forward_batch: ForwardBatch, token_ids: torch.Tensor):
        """Update OE token table with mask support for chunked prefill requests."""
        if forward_batch.oe_skip_token_table_update is not None:
            indices = (~forward_batch.oe_skip_token_table_update).nonzero(as_tuple=True)[0]
            if indices.numel() > 0:
                update_token_table(
                    forward_batch.oe_token_table,
                    token_ids[indices].to(torch.int32),
                    forward_batch.req_pool_indices[indices],
                    forward_batch.oe_out_column_starts[indices],
                    forward_batch.oe_out_req_lens[indices],
                )
        else:
            update_token_table(
                forward_batch.oe_token_table,
                token_ids.to(torch.int32),
                forward_batch.req_pool_indices,
                forward_batch.oe_out_column_starts,
                forward_batch.oe_out_req_lens,
            )

    def forward_target_verify(self, forward_batch: ForwardBatch):
        assert forward_batch.forward_mode.is_target_verify()
        forward_batch.attn_backend = self.target_worker.model_runner.attn_backend
        if __is_npu__:
            # Pin logprob attrs to their torchair-warmup values around the
            # compiled target forward so dynamo guards stay hit and no runtime
            # recompile desyncs the cross-DP HCCL load (EI0006). See
            # _stable_graph_logprob_flags. Verify logprobs are computed post-hoc
            # by compute_spec_logprobs, so masking is result-neutral.
            with self._stable_graph_logprob_flags(forward_batch):
                logits_output = self.target_worker.model_runner.forward(forward_batch)
        else:
            logits_output = self.target_worker.model_runner.forward_extend(forward_batch, skip_metadata_init=True)
        return logits_output

    def forward_draft_extend(self, forward_batch: ForwardBatch):
        forward_batch.attn_backend = self.model_runner.attn_backend
        forward_batch.topk_indices = None
        with self._stable_graph_logprob_flags(forward_batch):
            if __is_npu__:
                can_npu_graph = self.npu_graph_runner_for_draft_extend and self.npu_graph_runner_for_draft_extend.can_run(forward_batch)
                if can_npu_graph:
                    logits_output = self.npu_graph_runner_for_draft_extend.replay(forward_batch)
                else:
                    logits_output = self.model_runner.forward(forward_batch)
            else:
                logits_output = self.model_runner.forward_extend(forward_batch, skip_metadata_init=True)
        self.capture_for_decode(logits_output, forward_batch)

    def prepare_for_multi_step_draft(self, forward_batch: ForwardBatch, accept_lengths: torch.Tensor):
        if __is_npu__:
            self.req_to_token_pool.verified_lens[forward_batch.req_pool_indices] += accept_lengths
            num_seqs = forward_batch.batch_size
            if self.speculative_num_steps > 1:
                verified_lens = self.req_to_token_pool.verified_lens[forward_batch.req_pool_indices]
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
            if self.server_args.enable_mla_l1_5_cache:
                self.token_to_kv_pool.remap_tokens_for_kvp_inplace(out_cache_loc_for_draft_decode)
            if self.speculative_num_steps > 1:
                forward_batch.seq_lens = seq_lens
                forward_batch.seq_lens_sum = seq_lens_sum
                forward_batch.positions = seq_lens
                forward_batch.capture_hidden_mode = CaptureHiddenMode.LAST
                if self.index_share_for_mtp_iteration:
                    if forward_batch.forward_mode == ForwardMode.EXTEND:
                        # [q_tokens, 2048]
                        cu_seqlens_q = torch.cumsum(forward_batch.extend_seq_lens, dim=0)
                        forward_batch.topk_indices = forward_batch.topk_indices[cu_seqlens_q - 1]
                    elif forward_batch.forward_mode == ForwardMode.DRAFT_EXTEND:
                        # [q_tokens, 2048] -> [bs, draft_tokens, 2048] -> [:, acc-1]
                        idxs_shaped = forward_batch.topk_indices.reshape(bs, self.model_runner.server_args.speculative_num_draft_tokens, -1)
                        forward_batch.topk_indices = idxs_shaped[torch.arange(bs, device=idxs_shaped.device), accept_lengths - 1]
                    else:
                        raise RuntimeError()
                forward_batch.forward_mode = ForwardMode.DECODE
                forward_batch.out_cache_loc = out_cache_loc_for_draft_decode

    @torch.inference_mode()
    def propose(self, forward_batch: ForwardBatch, accept_lengths: torch.Tensor):
        forward_batch.attn_backend = self.model_runner.attn_backend

        with self._stable_graph_logprob_flags(forward_batch):
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
            forward_batch.oe_column_starts[: forward_batch.batch_size] = forward_batch.extend_prefix_lens + 1
            forward_batch.oe_req_lens[: forward_batch.batch_size] = forward_batch.extend_seq_lens
        forward_batch.capture_hidden_mode = CaptureHiddenMode.LAST
        forward_batch.spec_info = EagleDraftInput(
            hidden_states=target_logits_output.hidden_states,
            verified_id=next_token_ids,
        )
        forward_batch.spec_info.set_input_ids(forward_batch)
        # The drafter always follows the token embedding path.  Discard any
        # target-side/preallocated embedding buffer before substituting the
        # batch-local hash pad IDs.
        forward_batch.input_embeds = None
        forward_batch.input_ids = substitute_mm_pad_values_for_mtp(forward_batch.input_ids, self.mtp_mm_placeholder_ids)
        forward_batch.token_to_kv_pool = self.model_runner.token_to_kv_pool
        forward_batch.attn_backend = self.model_runner.attn_backend

    def forward_decode_spec(
        self,
        forward_batch: ForwardBatch,
        vocab_masks: Optional[torch.Tensor] = None,
        pre_propose_hook=None,
        post_target_verify_hook=None,
    ):
        if self.use_over_embedding:
            forward_batch.oe_column_starts[: forward_batch.batch_size] = forward_batch.req_to_token_pool.verified_lens[forward_batch.req_pool_indices]
            forward_batch.oe_req_lens[: forward_batch.batch_size] = self.server_args.speculative_num_draft_tokens
        logits_output = self.forward_target_verify(forward_batch)
        # In-graph grammar (overlap path): the DFS that builds vocab_masks runs on
        # the grammar stream, forked off BEFORE forward_target_verify so it
        # overlaps the target forward. vocab_masks is consumed by
        # rejection_sampling below (not by forward_target_verify), so the runner
        # joins the grammar stream back here -- after the target forward, before
        # the mask is read.
        if post_target_verify_hook is not None:
            post_target_verify_hook()
        target_predict, logits_output, accept_length, accept_index = self.rejection_sampling(forward_batch, logits_output, vocab_masks)
        # Results from target_predict, verify_lens not updated yet, just write at current verify length + 1 and continue
        if self.use_over_embedding:
            forward_batch.oe_out_column_starts[: forward_batch.batch_size] = self.req_to_token_pool.verified_lens[forward_batch.req_pool_indices] + 1
            forward_batch.oe_out_req_lens[: forward_batch.batch_size] = self.server_args.speculative_num_draft_tokens
            update_token_table(
                oe_token_table=forward_batch.oe_token_table,
                tokens=target_predict,
                row_indices=forward_batch.req_pool_indices,
                column_starts=forward_batch.oe_out_column_starts,
                oe_req_lens=forward_batch.oe_out_req_lens,
            )
        output_ids = target_predict[accept_index]
        # Capture the logprob accept_length (+1 bonus token) before preprocess_for_draft_after_decode
        # mutates accept_length in-place. The logprob computation itself is deferred until after
        # propose(): it does host-blocking D2H syncs (.tolist) that must NOT sit between the
        # target-verify and draft collectives, otherwise DP groups whose local batch has no
        # return_logprob request launch propose's 16-card draft collective un-skewed while groups
        # with a request stall on the sync, wedging the collective under NPU graph replay.
        accept_length_for_logprob = accept_length + 1 if forward_batch.return_logprob else None
        new_verified_id = self.preprocess_for_draft_after_decode(forward_batch, accept_length, accept_index, target_predict, with_draft_model=True)
        output_ids = target_predict[accept_index]
        if __is_npu__:
            # make verified token d2h transfer overlap with draft forward
            output_ids_copy_done = torch.get_device_module(self.device).Event(enable_timing=True)
            output_ids_cpu = output_ids.clone().to("cpu", non_blocking=True, copy=True)
            accept_length_cpu = accept_length.clone().to("cpu", non_blocking=True, copy=True)
            output_ids_copy_done.record()

        # In-graph grammar (overlap path): the accept host node is forked onto a
        # dedicated grammar stream HERE -- after output_ids / (the +1'd)
        # accept_length are ready, but before propose() -- so the permanent
        # accept overlaps the draft steps. The runner supplies the hook (it owns
        # the grammar stream + ctx); accept reads only matcher state, draft reads
        # only new_verified_id, so they share no mutable state. The runner joins
        # the grammar stream at graph end.
        if pre_propose_hook is not None:
            pre_propose_hook(output_ids, accept_length)
        token_list = self.propose(forward_batch, accept_length)

        grammar_terminated_list_or_event = None
        if __is_npu__:
            # update_grammar with output_ids_cpu, offloaded to the dedicated
            # grammar-update thread so the forward thread is not blocked by the
            # D2H copy sync + CPU grammar accept loop. The forward thread waits
            # for it before the next step's vocab mask generation (see
            # rejection_sampling); the scheduler thread waits on the event
            # before reading grammar_terminated_list (see resolve_batch_result).
            grammar_update_done_event = self.submit_grammar_update(forward_batch, output_ids_copy_done, output_ids_cpu, accept_length_cpu)
            grammar_terminated_list_or_event = grammar_update_done_event

        # Compute logprobs only after all collectives (target-verify + draft propose) have been
        # launched, so the host-blocking D2H syncs it performs cannot skew collective launch order
        # across DP groups. propose() does not mutate the state this reads, so ordering is safe.
        self.compute_spec_logprobs(forward_batch, logits_output, accept_length_for_logprob, output_ids)

        # On NPU the last element is the grammar-done threading.Event carrying
        # the grammar_terminated_list attribute (resolved in resolve_batch_result);
        # elsewhere it is None.
        return (logits_output, output_ids, accept_length, new_verified_id, token_list, grammar_terminated_list_or_event)

    def forward_prefill_spec(self, model_worker_batch: ModelWorkerBatch, forward_batch: ForwardBatch):
        if self.use_over_embedding:
            forward_batch.oe_column_starts[: forward_batch.batch_size] = forward_batch.extend_prefix_lens
            forward_batch.oe_req_lens[: forward_batch.batch_size] = forward_batch.extend_seq_lens
        if __is_npu__:
            target_logits_output = self.target_worker.model_runner.forward(forward_batch)
        else:
            target_logits_output = self.target_worker.model_runner.forward_extend(forward_batch, skip_metadata_init=True)
        grammar_terminated_list = None

        if self.pp_group.is_last_rank:
            next_token_ids = self.target_worker.model_runner.sample(target_logits_output, forward_batch)
            if model_worker_batch.disagg_set_aux_fn is not None:
                model_worker_batch.disagg_set_aux_fn(next_token_ids, target_logits_output)
            self.prepare_for_draft_prefill(forward_batch, target_logits_output, next_token_ids)
            token_list = self.propose(forward_batch, forward_batch.new_tokens_to_compute)
            return (
                target_logits_output,
                next_token_ids,
                None,
                next_token_ids,
                token_list,
                grammar_terminated_list,
            )
        else:
            return (
                target_logits_output,
                None,
                None,
                None,
                None,
                grammar_terminated_list,
            )

    def forward_idle(self, forward_batch: ForwardBatch):
        assert forward_batch.forward_mode.is_idle()
        forward_batch.capture_hidden_mode = CaptureHiddenMode.LAST
        logits_output = self.target_worker.model_runner.forward_idle(forward_batch)
        if self.pp_group.is_last_rank:
            next_token_ids = self.target_worker.model_runner.sample(logits_output, forward_batch)
            forward_batch.spec_info = EagleDraftInput(
                hidden_states=logits_output.hidden_states,
                verified_id=next_token_ids,
            )
            self.model_runner.forward_idle(forward_batch)
            for _ in range(self.speculative_num_steps - 1):
                # For normal MTP path, draft decode uses per-request one token; use global batch size.
                # For multi-head EAGLE, peer ranks may still run EXTEND-style draft steps
                if not self.server_args.is_multi_head_eagle:
                    forward_batch.global_num_tokens = forward_batch.global_batch_size
                self.model_runner.forward_idle(forward_batch)
                # TODO: below is npu logic, check if needed
                # if self.server_args.enable_dp_attention and forward_batch.batch_size == 0:
                #     # Empty DP ranks need to follow draft decode collectives after the first idle/prefill participation
                #     forward_batch.global_num_tokens = forward_batch.global_batch_size
                #     self.model_runner.forward_idle(forward_batch)
                # else:
                #     self.model_runner.forward_idle(forward_batch)
        return logits_output, None, None, None, None, None

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
                self._masked_update_token_table(forward_batch, input_ids)
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
            if self.drafter_backend in ["flashinfer", "hybrid_linear_attn"]:
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
                forward_batch.attn_metadata.query_len_tensor = torch.arange(1, forward_batch.batch_size + 1, 1, dtype=torch.int64, device="npu")
                forward_batch.set_npu_ep_metadata(self.target_worker.model_runner)
                can_npu_graph = self.npu_graph_runner_for_draft_extend and self.npu_graph_runner_for_draft_extend.can_run(forward_batch)
                if can_npu_graph:
                    logits_output = self.npu_graph_runner_for_draft_extend.replay(forward_batch)
                else:
                    logits_output = self.model_runner.model.forward(forward_batch.input_ids, forward_batch.positions, forward_batch)
            else:
                logits_output = self.model_runner.model.forward(forward_batch.input_ids, forward_batch.positions, forward_batch)
            forward_batch.positions.add_(1)
            topk_index = self.draft_sample(forward_batch, logits_output, i + 1)
            if self.hot_token_id is not None:
                topk_index = self.hot_token_id[topk_index]

            # Update last hidden_states
            hidden_states = logits_output.hidden_states

        return token_list

    def capture_for_decode(self, logits_output: LogitsProcessorOutput, forward_batch: ForwardBatch):
        spec_info = forward_batch.spec_info
        spec_info.topk_index = self.draft_sample(forward_batch, logits_output, 0)
        spec_info.hidden_states = logits_output.hidden_states

    def draft_sample(self, forward_batch: ForwardBatch, logits_output: LogitsProcessorOutput, draft_step):
        if global_server_args_dict["enable_speculative_sampling"] and not forward_batch.sampling_info.is_all_greedy:
            with torch.profiler.record_function("draft_speculative_sample"):
                sampling_top_k = forward_batch.speculative_sampling_info.probs.shape[-1]
                sampling_info = forward_batch.sampling_info
                batch_size = forward_batch.batch_size
                next_token_logits = logits_output.next_token_logits
                seq_len = next_token_logits.shape[0] // batch_size
                tokens = batch_size * seq_len
                next_token_logits = next_token_logits[:tokens]
                expanded_temperature = _pad_to_size(
                    sampling_info.temperatures.view(-1, 1).expand(sampling_info.temperatures.numel(), seq_len).reshape(-1, 1), tokens, 1.0
                )
                next_token_logits_sort, probs_idx = torch.topk(next_token_logits, k=sampling_top_k, dim=-1, largest=True, sorted=True)
                probs_sort = softmax(next_token_logits_sort / expanded_temperature, dim=-1)
                # probs_sum=torch.cumsum(probs_sort, dim=-1)
                # mask_top_k=torch.arange(0, probs_sort.shape[-1], device=probs_sort.device).view(1, -1)>=_pad_to_size(
                #     sampling_info.top_ks.view(-1, 1).expand(sampling_info.top_ks.numel(), seq_len).reshape(-1, 1), tokens)
                # mask_top_p=(probs_sum-probs_sort)>_pad_to_size(
                #     sampling_info.top_ps.view(-1, 1).expand(sampling_info.top_ps.numel(), seq_len).reshape(-1, 1), tokens)
                # probs_sort=torch.where(mask_top_k|mask_top_p, 0, probs_sort)  # [B*S, top_k]
                # probs_sort=probs_sort/(probs_sort.sum(dim=-1, keepdim=True) + 1e-8)
                probs_sort_selected_index = torch.multinomial(probs_sort, num_samples=1)
                ori_bs = getattr(forward_batch, "ori_bs", None)
                req_pool_indices = forward_batch.req_pool_indices[:ori_bs]
                forward_batch.speculative_sampling_info.update_draft(
                    req_pool_indices,
                    draft_step,
                    probs_sort[:ori_bs],
                    probs_idx[:ori_bs],
                    probs_sort_selected_index[:ori_bs],
                )
                topk_index = torch.gather(probs_idx, dim=1, index=probs_sort_selected_index).view(-1).to(torch.int32)
        else:
            probs = softmax(logits_output.next_token_logits)
            _, topk_index = fast_topk(probs, self.topk, dim=-1)
        return topk_index

    def multi_batch_select(self, max_input_batch, compile_bs_list):
        max_compile_bs = max(compile_bs_list)
        assert max_input_batch <= max_compile_bs, (
            f"max input batch ({max_input_batch}) should less equal than max compile bs({max_compile_bs}) in graph mode"
        )
        return min(bs for bs in compile_bs_list if bs >= max_input_batch)

    def update_weights_from_tensor(self, recv_req: UpdateWeightsFromTensorReqInput):
        monkey_patch_torch_reductions()
        named_tensors = MultiprocessingSerializer.deserialize(recv_req.serialized_named_tensors[self.global_rank])
        unwrapped_tensors = unwrap_ipc_tensors(named_tensors, self.global_rank, torch.device(self.device))
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
