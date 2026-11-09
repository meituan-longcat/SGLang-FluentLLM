import bisect
from contextlib import contextmanager
from typing import TYPE_CHECKING, Optional

import torch
import tqdm

from sglang.srt.distributed import get_tensor_model_parallel_rank
from sglang.srt.distributed.parallel_state import graph_capture
from sglang.srt.hostfunc import launch_hostfunc
from sglang.srt.layers.moe.fused_moe_native import fused_moe_forward_native
from sglang.srt.layers.sampler.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.cuda_graph_runner import get_batch_sizes_to_capture
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.ops.custom_op import CustomOp
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.speculative.grammar_hostfunc import (
    accept_hostfunc,
    dfs_hostfunc,
)

if TYPE_CHECKING:
    from sglang.srt.speculative.eagle_worker import EAGLEWorker

from sglang.srt.utils import get_colorful_logger

logger = get_colorful_logger(__name__)

global_graph_memory_pool = None


def _to_torch(model: torch.nn.Module, reverse: bool, num_tokens: int, draft_token_num: int):
    for sub in model._modules.values():
        if isinstance(sub, CustomOp):
            if reverse:
                sub._forward_method = sub.forward_cuda
                sub.is_torch_compile = False
            else:
                # NOTE: Temporarily workaround MoE
                if "FusedMoE" in sub.__class__.__name__:
                    if num_tokens <= 1 * draft_token_num:
                        sub._forward_method = fused_moe_forward_native
                else:
                    sub._forward_method = sub.forward_native
                sub.is_torch_compile = True
        if isinstance(sub, torch.nn.Module):
            _to_torch(sub, reverse, num_tokens, draft_token_num)


@contextmanager
def patch_model(model: torch.nn.Module, enable_compile: bool, num_tokens: int, tp_group, draft_token_num: int):
    """Patch the model to make it compatible with with torch.compile"""
    backup_ca_comm = None

    try:
        if enable_compile:
            _to_torch(model, reverse=False, num_tokens=num_tokens, draft_token_num=draft_token_num)
            backup_ca_comm = tp_group.ca_comm
            # Use custom-allreduce here.
            # We found the custom allreduce is much faster than the built-in allreduce in torch,
            # even with ENABLE_INTRA_NODE_COMM=1.
            # tp_group.ca_comm = None
            yield torch.compile(
                torch.no_grad()(model.forward),
                mode="max-autotune-no-cudagraphs",
                dynamic=False,
            )
        else:
            yield model.forward
    finally:
        if enable_compile:
            _to_torch(model, reverse=True, num_tokens=num_tokens, draft_token_num=draft_token_num)
            tp_group.ca_comm = backup_ca_comm


def set_torch_compile_config():
    import torch._inductor.config

    torch._inductor.config.coordinate_descent_tuning = True
    torch._inductor.config.triton.unique_kernel_names = True
    torch._inductor.config.fx_graph_cache = True  # Experimental feature to reduce compilation times, will be on by default in future

    # FIXME: tmp workaround
    torch._dynamo.config.accumulated_cache_size_limit = 1024
    if hasattr(torch._dynamo.config, "cache_size_limit"):
        torch._dynamo.config.cache_size_limit = 1024


class SpecDecodeCudaGraphRunner:
    def __init__(self, spec_decode_worker: "EAGLEWorker"):
        self.output_buffers = {}
        self.graphs = {}
        self.spec_decode_worker = spec_decode_worker
        self.enable_torch_compile = spec_decode_worker.model_runner.server_args.enable_torch_compile
        self.capture_bs, self.complile_bs = get_batch_sizes_to_capture(spec_decode_worker.model_runner)
        logger.info(f"{self.capture_bs=} {self.complile_bs=}")

        self.max_bs = max(self.capture_bs)
        self.draft_token_num = spec_decode_worker.server_args.speculative_num_draft_tokens
        self.max_num_token = self.max_bs * self.draft_token_num
        self.enable_dp_attention = spec_decode_worker.model_runner.server_args.enable_dp_attention
        self.world_size = spec_decode_worker.model_runner.server_args.world_size
        self.dp_size = spec_decode_worker.model_runner.server_args.dp_size
        self.capture_sample_graph = spec_decode_worker.model_runner.server_args.capture_sample_graph

        self.spec_decode_worker.target_worker.model_runner.attn_backend.init_cuda_graph_state(self.max_num_token)
        assert hasattr(self.spec_decode_worker, "model_runner_list")
        for runner in self.spec_decode_worker.model_runner_list:
            runner.attn_backend.init_cuda_graph_state(self.max_num_token)

        self.disable_padding = self.spec_decode_worker.model_runner.server_args.disable_cuda_graph_padding
        self.seq_lens_cpu = torch.full((self.max_bs,), 1, dtype=torch.int32).cpu()

        hf_config = spec_decode_worker.model_runner.model_config.hf_config
        if getattr(hf_config, "use_over_embedding", False):
            self.over_embedding_n = hf_config.oe_neighbor_num
            self.over_embedding_k = hf_config.oe_split_num
        self.use_over_embedding = getattr(hf_config, "use_over_embedding", False) or self.spec_decode_worker.target_worker.use_over_embedding

        if self.use_over_embedding:
            self.token_table = spec_decode_worker.oe_token_table

        if self.enable_torch_compile:
            set_torch_compile_config()

        with torch.device("cuda"):
            self.req_pool_indices = torch.zeros((self.max_bs,), dtype=torch.int64)
            self.draft_token = torch.ones((self.max_bs * self.draft_token_num,), dtype=torch.int32)
            self.oe_column_starts = torch.empty([self.max_bs], dtype=torch.int32) if self.use_over_embedding else None
            self.oe_req_lens = torch.empty([self.max_bs], dtype=torch.int32) if self.use_over_embedding else None
            self.oe_out_column_starts = torch.empty([self.max_bs], dtype=torch.int32) if self.use_over_embedding else None
            self.oe_out_req_lens = torch.empty([self.max_bs], dtype=torch.int32) if self.use_over_embedding else None
            self.out_cache_loc_buffer = torch.arange(0, self.max_num_token, dtype=torch.int64)
            self.positions = torch.arange(0, self.max_num_token, dtype=torch.int64)
            self.mrope_positions = torch.zeros((3, self.max_bs), dtype=torch.int64)
            self.hidden_states = torch.zeros(
                (
                    self.max_num_token,
                    spec_decode_worker.model_runner.model_config.hidden_size,
                ),
                dtype=spec_decode_worker.model_runner.dtype,
            )
            self.seq_lens = torch.full((self.max_bs,), 1, dtype=torch.int32)
            self.new_tokens_to_compute = torch.full(
                (self.max_bs,),
                self.spec_decode_worker.speculative_num_steps - 1,
                device="cuda",
                dtype=torch.int32,
            )
            self.temperature_buffer = torch.zeros((self.max_bs, 1), dtype=torch.float32)
            self.topk_buffer = torch.zeros((self.max_bs,), dtype=torch.int32)
            self.topp_buffer = torch.zeros((self.max_bs,), dtype=torch.float32)
            self.minp_buffer = torch.zeros((self.max_bs,), dtype=torch.float32)
            self.thinking_budgets_buffer = torch.zeros((self.max_bs,), dtype=torch.int32)
            self.think_end_ids_buffer = torch.full((self.max_bs, 1), -1, dtype=torch.int32)
            if self.enable_dp_attention:
                self.gathered_buffer = torch.zeros(
                    (
                        self.max_num_token * self.world_size,
                        self.spec_decode_worker.model_runner.model_config.hidden_size,
                    ),
                    dtype=self.spec_decode_worker.model_runner.dtype,
                )
            self.scaling_penalties = torch.ones((self.max_bs,), dtype=torch.float32)
            self.cumulated_scaling_penalties = torch.ones(
                (self.max_bs, self.spec_decode_worker.target_worker.model_runner.model_config.vocab_size), dtype=torch.float32
            )

        # Grammar machinery is owned by the worker (built in
        # _init_grammar_overlap before cuda graph capture) so it exists even when
        # cuda graph is disabled and is shared with the eager verify path. The
        # runner just references it. ``grammar_overlap_enabled`` gates whether
        # grammar host nodes (DFS + accept) are captured into the verify graph:
        # they run on the grammar stream only under the overlap scheduler.
        # Per-batch signal field (last_grammar_slot) and the slot allocator live
        # on the worker so the graph and eager paths share one monotonic slot
        # counter.
        w = self.spec_decode_worker
        self.grammar_backend = w.grammar_backend
        self.vocab_masks = w.vocab_masks
        self.grammar_ctx = w.grammar_ctx
        self.grammar_stream = w.grammar_stream
        self.slot_scalar = w.slot_scalar
        self.grammar_overlap_enabled = w.grammar_overlap_enabled
        if self.grammar_overlap_enabled:
            self.grammar_ring_depth = w.grammar_ring_depth

        # Capture
        try:
            self.capture()
        except RuntimeError as e:
            raise Exception(
                f"Capture cuda graph failed: {e}\n"
                "Possible solutions:\n"
                "1. disable cuda graph by --disable-cuda-graph\n"
                "2. set --mem-fraction-static to a smaller value (e.g., 0.8 or 0.7)\n"
                "3. disable torch compile by not using --enable-torch-compile\n"
                "4. set --cuda-graph-max-bs to a smaller value (e.g., 32)\n"
                "Open an issue on GitHub https://github.com/sgl-project/sglang/issues/new/choose \n"
            ) from e

    def can_run(self, forward_batch: ForwardBatch):
        if not (forward_batch.forward_mode.is_target_verify() or forward_batch.forward_mode.is_idle()):
            return False
        if self.enable_dp_attention:
            min_num_tokens, max_num_tokens = (
                min(forward_batch.global_num_tokens),
                max(forward_batch.global_num_tokens),
            )
            max_bsz = int(max_num_tokens / self.draft_token_num)
            is_bs_supported = forward_batch.all_decode_or_idle and (
                (min_num_tokens == max_num_tokens and max_bsz in self.graphs) if self.disable_padding else max_bsz <= self.max_bs
            )
        else:
            is_bs_supported = forward_batch.batch_size in self.graphs if self.disable_padding else forward_batch.batch_size <= self.max_bs

        return is_bs_supported

    def replay(self, forward_batch: ForwardBatch, vocab_masks: Optional[torch.Tensor]):
        """
        ATTENTION! The returned tensors (new_verified_id, token_list, logits_output, ...) are
        views into this graph's static output buffers, which live in the shared CUDA
        graph memory pool.
        """
        assert forward_batch.out_cache_loc is not None
        raw_bs = forward_batch.batch_size
        raw_num_token = raw_bs * self.draft_token_num

        # Pad
        if self.enable_dp_attention:
            index = bisect.bisect_left(
                self.capture_bs,
                max(forward_batch.global_num_tokens) // self.draft_token_num,
            )
        else:
            index = bisect.bisect_left(self.capture_bs, raw_bs)
        bs = self.capture_bs[index]
        if bs != raw_bs:
            self.seq_lens.fill_(1)
            self.out_cache_loc_buffer.zero_()
            self.positions.zero_()
        self.req_pool_indices.zero_()

        if forward_batch.forward_mode.is_idle():
            forward_batch.spec_info = self.init_verify_spec_info(bs=0)
            forward_batch.forward_mode = ForwardMode.TARGET_VERIFY

        assert forward_batch.forward_mode.is_target_verify()

        # Common inputs
        self.draft_token[:raw_num_token].copy_(forward_batch.spec_info.draft_token)
        # Whether this batch carries a grammar list. The host nodes are always
        # captured, so for no-grammar slots the in-graph DFS emits allow-all and
        # accept no-ops; but if a grammar list exists, forward thread must consume
        # the graph-side termination snapshot instead of re-running accept.
        grammar_active = False
        if self.grammar_overlap_enabled:
            # The in-graph DFS host node builds the vocab mask this step; we only
            # need to hand it the per-step grammars + batch size. The draft tokens
            # are read back D2H *inside* the graph from self.draft_token (filled
            # just above), so nothing else to stage here.
            #
            # Pick this replay's ring slot and stage it into the device scalar so
            # the captured graph can copy it D2H into ctx.slot_host -- letting the
            # host nodes pick the right slot at execution time under launch-ahead.
            slot = self.spec_decode_worker.alloc_slot()
            self.slot_scalar.fill_(slot)
            grammars = forward_batch.sampling_info.grammars or []
            # Fill/H2D range is the padded bucket (bs*L), matching the captured
            # mask_host[:capture_n] copy below; padding rows raw_bs..bs stay
            # allow-all so padded requests carry no constraint.
            self.grammar_ctx.set_step_inputs(grammars, raw_bs, bs * self.draft_token_num, slot)
            grammar_active = forward_batch.sampling_info.grammars is not None
            # Record the slot for the consumer (forward thread -> output queue ->
            # resolve_batch_result). -1 means "no grammar this step, skip
            # collect". Do NOT collect here: that needs the graph to have run,
            # which would force a sync and break the overlap. The signal field
            # lives on the worker (shared with the eager verify path).
            self.spec_decode_worker.last_grammar_slot = slot if grammar_active else -1
        elif vocab_masks is not None:
            logger.debug("[SpecDecodeCudaGraphRUnner] copy vocab masks")
            self.vocab_masks[:raw_num_token].copy_(vocab_masks)
        elif self.grammar_backend:
            self.grammar_backend.reset_vocab_masks(self.vocab_masks[:raw_num_token])
        self.req_pool_indices[:raw_bs].copy_(forward_batch.req_pool_indices)
        self.seq_lens[:raw_bs].copy_(forward_batch.seq_lens)
        if forward_batch.seq_lens_cpu is not None:
            if bs != raw_bs:
                self.seq_lens_cpu.fill_(1)
            self.seq_lens_cpu[:raw_bs].copy_(forward_batch.seq_lens_cpu)
        self.out_cache_loc_buffer[:raw_num_token].copy_(forward_batch.out_cache_loc)
        self.positions[:raw_num_token].copy_(forward_batch.positions)
        if self.use_over_embedding:
            forward_batch.oe_column_starts[:raw_num_token].copy_(forward_batch.oe_column_starts[:raw_num_token])
            forward_batch.oe_req_lens[:raw_bs].copy_(forward_batch.oe_req_lens[:raw_bs])

        assert forward_batch.forward_mode.is_target_verify()

        sampling_info = forward_batch.sampling_info
        self.scaling_penalties[:raw_bs].copy_(sampling_info.repetition_penalties)
        self.cumulated_scaling_penalties[:raw_bs].copy_(forward_batch.spec_info.cumulated_scaling_penalties)

        self.thinking_budgets_buffer.zero_()
        if sampling_info.thinking_budgets is not None:
            self.thinking_budgets_buffer[:raw_bs].copy_(sampling_info.thinking_budgets)
            self.think_end_ids_buffer[:raw_bs].copy_(sampling_info.think_end_ids)

        if self.capture_sample_graph:
            self.temperature_buffer[:raw_bs].copy_(sampling_info.temperatures)
            self.topk_buffer[:raw_bs].copy_(sampling_info.top_ks)
            self.topp_buffer[:raw_bs].copy_(sampling_info.top_ps)
            self.minp_buffer[:raw_bs].copy_(sampling_info.min_ps)

        # Attention backend
        if len(self.spec_decode_worker.model_runner_list):
            # other runner share the first runner's attn_backend
            runner = self.spec_decode_worker.model_runner_list[0]
            runner.attn_backend.init_forward_metadata_replay_cuda_graph(
                bs,
                self.req_pool_indices,
                self.seq_lens,
                forward_batch.seq_lens_sum + (bs - raw_bs),
                forward_batch.forward_mode,
                forward_batch.spec_info,
                seq_lens_cpu=self.seq_lens_cpu,
            )
        self.spec_decode_worker.target_worker.model_runner.attn_backend.init_forward_metadata_replay_cuda_graph(
            bs,
            self.req_pool_indices,
            self.seq_lens,
            forward_batch.seq_lens_sum + (bs - raw_bs),
            forward_batch.forward_mode,
            forward_batch.spec_info,
            seq_lens_cpu=self.seq_lens_cpu,
        )

        # Replay
        self.graphs[bs].replay()
        # Persist graph-side budget latches before the next replay reloads the buffer.
        if sampling_info.thinking_budgets is not None:
            sampling_info.thinking_budgets.copy_(self.thinking_budgets_buffer[:raw_bs])
        out = self.output_buffers[bs]

        # NOTE: no per-replay synchronize() here. The grammar host nodes (DFS +
        # accept) are captured into the graph and run on the grammar stream,
        # which the graph joins back before its end. We deliberately do NOT block
        # the forward thread to read their results -- that full sync is exactly
        # what collapsed launch-ahead to pipeline depth 1. Instead:
        #   * the per-replay slot was already recorded above
        #     (self.last_grammar_slot),
        #   * the termination snapshot + error flags live in the ring at that
        #     slot, and
        #   * the consumer thread (resolve_batch_result) collects them after
        #     copy_done.synchronize() -- copy_done is recorded after the graph
        #     (so after the grammar-stream join) and is awaited unconditionally
        #     on every rank, giving the rank-symmetric GIL release the in-graph
        #     host callbacks need without serializing the forward thread.
        if not self.grammar_overlap_enabled:
            self.spec_decode_worker.last_grammar_slot = -1

        if bs != raw_bs:
            forward_batch.batch_size = raw_bs
            forward_batch.positions = self.positions[:raw_num_token]
            forward_batch.seq_lens = self.seq_lens[:raw_bs]
            forward_batch.req_pool_indices = self.req_pool_indices[:raw_bs]
            out = self._postprocess_output_to_raw_bs(out, raw_bs)
        else:
            # bs == raw_bs: always create a new logits_output to avoid object reuse.
            # output_buffers[bs][0] (logits_output) is a fixed object reused across replays.
            # In overlap mode:
            #   - Forward thread executes batch_{N-1} → batch_N sequentially
            #   - Scheduler thread consumes output_queue asynchronously
            #   - batch_{N-1} puts logits_output ref into output_queue
            #   - batch_N's compute_spec_logprobs() overwrites the same object's
            #     next_token_logprobs field (dynamic attribute addition)
            #   - If Scheduler delays reading batch_{N-1}, it gets batch_N's data
            #   - Race condition: IndexError when accept_lengths mismatch
            # _postprocess_output_to_raw_bs already creates new_logits_output, so just call it.
            out = self._postprocess_output_to_raw_bs(out, raw_bs)
        return out

    def init_verify_spec_info(self, bs: int):
        from sglang.srt.speculative.eagle_utils import EagleVerifyInput

        grammar = None
        if self.grammar_backend:
            grammar = self.grammar_backend.init_value_impl(("regex", r".*"))

        spec_info = EagleVerifyInput(
            draft_token=self.draft_token[: bs * self.draft_token_num],
            positions=self.positions[: bs * self.draft_token_num],
            draft_token_num=self.spec_decode_worker.model_runner.server_args.speculative_num_draft_tokens,
            spec_steps=self.spec_decode_worker.model_runner.server_args.speculative_num_steps,
            capture_hidden_mode=CaptureHiddenMode.FULL,
            is_all_greedy=not self.capture_sample_graph,
            grammar=grammar,
            cumulated_scaling_penalties=self.cumulated_scaling_penalties[:bs, :],
        )
        return spec_info

    def capture(self):
        with graph_capture() as graph_capture_context:
            self.stream = graph_capture_context.stream
            capture_range = tqdm.tqdm(self.capture_bs) if get_tensor_model_parallel_rank() == 0 else self.capture_bs
            for bs in capture_range:
                if bs in self.complile_bs:
                    with patch_model(
                        self.spec_decode_worker.target_worker.model_runner.model,
                        True,
                        num_tokens=bs * self.draft_token_num,
                        tp_group=self.spec_decode_worker.target_worker.model_runner.tp_group,
                        draft_token_num=self.draft_token_num,
                    ) as forward:
                        origin_forward_func = self.spec_decode_worker.target_worker.model_runner.model.forward
                        self.spec_decode_worker.target_worker.model_runner.model.forward = forward
                        (
                            graph,
                            output_buffers,
                        ) = self.capture_one_batch_size(bs)
                        self.spec_decode_worker.target_worker.model_runner.model.forward = origin_forward_func
                else:
                    (
                        graph,
                        output_buffers,
                    ) = self.capture_one_batch_size(bs)
                self.graphs[bs] = graph
                self.output_buffers[bs] = output_buffers

    def get_forward_batch(self, bs):
        input_ids = self.draft_token[: bs * self.draft_token_num]
        req_pool_indices = self.req_pool_indices[:bs]
        seq_lens = self.seq_lens[:bs]
        out_cache_loc = self.out_cache_loc_buffer[: bs * self.draft_token_num]
        positions = self.positions[: bs * self.draft_token_num]

        if self.enable_dp_attention:
            global_num_tokens = [bs * self.draft_token_num] * self.world_size
            gathered_buffer = self.gathered_buffer[: bs * self.world_size * self.draft_token_num]
            global_batch_size = [bs] * self.world_size
        else:
            global_num_tokens = None
            gathered_buffer = None
            global_batch_size = None

        verify_spec_info = self.init_verify_spec_info(bs)
        forward_batch = ForwardBatch(
            forward_mode=ForwardMode.TARGET_VERIFY,
            batch_size=bs,
            input_ids=input_ids,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            req_to_token_pool=self.spec_decode_worker.req_to_token_pool,
            token_to_kv_pool=self.spec_decode_worker.target_worker.model_runner.token_to_kv_pool,
            attn_backend=self.spec_decode_worker.target_worker.model_runner.attn_backend,
            out_cache_loc=out_cache_loc,
            seq_lens_sum=seq_lens.sum(),
            oe_token_table=self.token_table if self.use_over_embedding else None,
            oe_column_starts=self.oe_column_starts[:bs] if self.use_over_embedding else None,
            oe_req_lens=self.oe_req_lens[:bs] if self.use_over_embedding else None,
            oe_out_column_starts=self.oe_out_column_starts[:bs] if self.use_over_embedding else None,
            oe_out_req_lens=self.oe_out_req_lens[:bs] if self.use_over_embedding else None,
            return_logprob=False,
            positions=positions,
            global_num_tokens=global_num_tokens,
            gathered_buffer=gathered_buffer,
            mrope_positions=None,
            spec_algorithm=self.spec_decode_worker.model_runner.spec_algorithm,
            spec_info=verify_spec_info,
            capture_hidden_mode=CaptureHiddenMode.FULL,
            all_decode_or_idle=True,
            draft_input_ids=None,
            new_tokens_to_compute=self.new_tokens_to_compute[:bs],
            global_batch_size=global_batch_size,
        )

        forward_batch.sampling_info = SamplingBatchInfo(
            temperatures=self.temperature_buffer[:bs],
            top_ks=self.topk_buffer[:bs],
            top_ps=self.topp_buffer[:bs],
            min_ps=self.minp_buffer[:bs],
            repetition_penalties=self.scaling_penalties[:bs],
            is_all_greedy=not self.capture_sample_graph,
            need_min_p_sampling=False,
            vocab_size=self.spec_decode_worker.model_runner.model_config.vocab_size,
            thinking_budgets=self.thinking_budgets_buffer[:bs],
            think_end_ids=self.think_end_ids_buffer[:bs],
        )
        return forward_batch

    def capture_one_batch_size(self, bs: int):
        graph = torch.cuda.CUDAGraph()
        stream = self.stream

        forward_batch = self.get_forward_batch(bs)
        # Attention backend
        self.spec_decode_worker.target_worker.model_runner.attn_backend.init_forward_metadata_capture_cuda_graph(
            bs,
            bs * self.draft_token_num,
            forward_batch.req_pool_indices,
            forward_batch.seq_lens,
            forward_batch.forward_mode,
            forward_batch.spec_info,
        )
        if len(self.spec_decode_worker.model_runner_list):
            runner = self.spec_decode_worker.model_runner_list[0]
            runner.attn_backend.init_forward_metadata_capture_cuda_graph(
                bs,
                bs * self.draft_token_num,
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                forward_batch.forward_mode,
                forward_batch.spec_info,
            )
        for runner in self.spec_decode_worker.model_runner_list[1:]:
            runner.attn_backend = self.spec_decode_worker.model_runner_list[0].attn_backend

        # Run and capture
        def run_once():
            if self.grammar_overlap_enabled:
                capture_n = bs * self.draft_token_num
                ctx = self.grammar_ctx
                # Capture-time dummy inputs so the grammar host nodes execute
                # safely while the graph is recorded. replay() overwrites these
                # with the real per-step inputs before replaying the nodes; the
                # dummy `.*` grammar (from init_verify_spec_info) accepts any
                # token and never terminates, so advancing it here is harmless.
                ctx.set_step_inputs([forward_batch.spec_info.grammar] * bs, bs, capture_n, 0)
                # NOTE: slot_scalar must NOT be filled here -- this runs inside
                # torch.cuda.graph(), so any fill_() would be captured as a graph
                # node and re-run on every replay, clobbering the per-step slot
                # that replay() stages eagerly before graph.replay(). The captured
                # D2H copy below reads slot_scalar's (fixed-address) device memory
                # at replay time, so replay()'s eager fill_(slot) is what picks the
                # slot. slot_scalar is staged to 0 eagerly before capture (below).
                vocab_masks = ctx.vocab_masks_device[:capture_n]

                main_stream = torch.cuda.current_stream()

                # DFS host node, forked onto the grammar stream so the bitmask
                # build (D2H draft -> traverse_tree on host -> H2D mask) overlaps
                # the target forward. vocab_masks is consumed by rejection_sampling
                # (NOT by forward_target_verify), so we fork here -- before the
                # target forward -- and join via post_target_verify_hook below,
                # after the target forward but before the mask is read. Both DFS
                # and the target forward only READ draft_token, so the cross-stream
                # read is safe. fork/join captured into the graph, validated in
                # test_hostfunc.py.
                self.grammar_stream.wait_stream(main_stream)
                with torch.cuda.stream(self.grammar_stream):
                    # Stage the slot read-back FIRST on the grammar stream so both
                    # host nodes (DFS here, accept later on this same stream) read
                    # the slot of this replay. Single buffer is safe: the copy and
                    # both host nodes are stream-ordered within the graph, and the
                    # next replay's copy is serialized after this graph's nodes.
                    ctx.slot_host.copy_(self.slot_scalar, non_blocking=True)
                    ctx.draft_token_host[:capture_n].copy_(self.draft_token[:capture_n], non_blocking=True)
                    launch_hostfunc(dfs_hostfunc, ctx)
                    vocab_masks.copy_(ctx.mask_host[:capture_n], non_blocking=True)

                def _join_dfs():
                    # Join the DFS (grammar stream) back before rejection_sampling
                    # reads vocab_masks.
                    main_stream.wait_stream(self.grammar_stream)

                # Accept host node, forked onto the SAME grammar stream so the
                # permanent accept overlaps the draft steps. The hook fires
                # inside forward_decode_spec AFTER output_ids / (+1'd)
                # accept_length are ready but BEFORE propose() (the draft steps).
                # accept reads only matcher state; draft reads only
                # new_verified_id -- no shared mutable state -- so they run
                # concurrently. Sharing one grammar stream keeps DFS -> accept
                # serialized on the matcher (DFS joined before rejection; accept
                # forked after it). Joined at graph end (below) so the accept host
                # writes are visible once copy_done is synchronized downstream.
                def _fork_accept(output_ids, accept_length):
                    self.grammar_stream.wait_stream(main_stream)
                    with torch.cuda.stream(self.grammar_stream):
                        ctx.accepted_host[:capture_n].copy_(output_ids, non_blocking=True)
                        ctx.accept_length_host[:bs].copy_(accept_length, non_blocking=True)
                        launch_hostfunc(accept_hostfunc, ctx)

                output = self.spec_decode_worker.forward_decode_spec(
                    forward_batch,
                    vocab_masks,
                    pre_propose_hook=_fork_accept,
                    post_target_verify_hook=_join_dfs,
                )

                # Join the grammar stream so the accept host node (and its host
                # writes to ctx.terminated_ring / error_ring) complete before the
                # graph ends -- so they are visible once copy_done (recorded after
                # the graph on the consumer side) is synchronized.
                main_stream.wait_stream(self.grammar_stream)
                return output

            vocab_masks = None
            if self.enable_grammar_backend():
                # Non-overlap grammar path: the mask is built outside the graph
                # and copied into self.vocab_masks by replay() before replaying.
                vocab_masks = self.vocab_masks[: (bs * self.draft_token_num)]
            output = self.spec_decode_worker.forward_decode_spec(forward_batch, vocab_masks)
            return output

        for _ in range(4):
            torch.cuda.synchronize()
            self.spec_decode_worker.model_runner.tp_group.barrier()
            run_once()
            forward_batch = self.get_forward_batch(bs)

        torch.cuda.synchronize()
        self.spec_decode_worker.model_runner.tp_group.barrier()

        # Reset DeepEP dispatch mode to NORMAL before capture, so that
        # clean_low_latency_buffer is recorded into the graph. During runtime,
        # prefill (normal mode) may dirty the low-latency buffer; the recorded
        # clean kernel ensures correctness on graph replay.
        try:
            from sglang.srt.layers.moe.dispatcher.deep_ep import DeepEPBuffer

            DeepEPBuffer.set_dispatch_mode_as_normal()
        except ImportError:
            pass

        global global_graph_memory_pool
        # Stage slot 0 eagerly (outside the captured region) so the captured D2H
        # copy of slot_scalar records a valid slot. replay() re-stages the real
        # per-step slot eagerly before each graph.replay().
        if self.grammar_overlap_enabled:
            self.slot_scalar.fill_(0)
        with torch.cuda.graph(graph, pool=global_graph_memory_pool, stream=stream):
            out = run_once()

        torch.cuda.synchronize()
        self.spec_decode_worker.model_runner.tp_group.barrier()

        global_graph_memory_pool = graph.pool()
        return graph, out

    def _postprocess_output_to_raw_bs(self, out, raw_bs):
        (
            logits_output,
            output_ids,
            accept_length,
            new_verified_id,
            token_list,
            grammar_terminated_list,
        ) = out
        n = raw_bs * self.draft_token_num
        new_logits_output = LogitsProcessorOutput(
            next_token_logits=logits_output.next_token_logits[:n],
            hidden_states=logits_output.hidden_states[:n],
            spec_accept_index=(logits_output.spec_accept_index[:n] if logits_output.spec_accept_index is not None else None),
        )
        output_ids = output_ids[:n]
        accept_length = accept_length[:raw_bs]
        new_verified_id = new_verified_id[:raw_bs]
        token_list = token_list[:raw_bs]
        return (
            new_logits_output,
            output_ids,
            accept_length,
            new_verified_id,
            token_list,
            grammar_terminated_list,
        )

    def enable_grammar_backend(self):
        server_args = self.spec_decode_worker.model_runner.server_args
        return server_args.grammar_backend is not None
