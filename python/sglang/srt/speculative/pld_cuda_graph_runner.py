# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project

import bisect
from typing import TYPE_CHECKING, Optional

import torch
import tqdm

from sglang.srt.constrained.base_grammar_backend import create_grammar_backend
from sglang.srt.distributed import get_tensor_model_parallel_rank
from sglang.srt.distributed.parallel_state import graph_capture
from sglang.srt.hf_transformers_utils import get_tokenizer
from sglang.srt.model_executor.cuda_graph_runner import (
    get_batch_sizes_to_capture,
    patch_model,
    set_torch_compile_config,
)
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.oe_utils import update_token_table
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo

if TYPE_CHECKING:
    from sglang.srt.speculative.pld_worker import PLDWorker

from sglang.srt.utils import get_colorful_logger

logger = get_colorful_logger(__name__)

global_graph_memory_pool = None


class PLDCudaGraphRunner:
    """
    CUDA Graph runner specifically designed for PLD (Prompt Lookup Decode).

    Key design principle: CUDA Graph ONLY captures target model verification.
    All n-gram matching and draft token generation happens OUTSIDE the graph.
    """

    def __init__(self, pld_worker: "PLDWorker"):
        self.output_buffers = {}
        self.graphs = {}
        self.pld_worker = pld_worker
        self.target_worker = pld_worker.target_worker

        self.capture_bs, self.compile_bs = get_batch_sizes_to_capture(self.target_worker.model_runner)
        self.max_bs = max(self.capture_bs)
        self.draft_token_num = pld_worker.server_args.speculative_num_draft_tokens
        self.max_num_token = self.max_bs * self.draft_token_num

        server_args = self.target_worker.model_runner.server_args
        self.enable_dp_attention = server_args.enable_dp_attention
        self.tp_size = server_args.tp_size
        self.dp_size = server_args.dp_size
        self.capture_sample_graph = server_args.capture_sample_graph
        self.disable_padding = server_args.disable_cuda_graph_padding
        self.enable_torch_compile = server_args.enable_torch_compile

        self.target_worker.model_runner.attn_backend.init_cuda_graph_state(self.max_num_token)

        hf_config = pld_worker.target_worker.model_runner.model_config.hf_config
        self.use_over_embedding = getattr(hf_config, "use_over_embedding", False)
        if self.use_over_embedding:
            self.token_table = pld_worker.target_worker.model_runner.oe_token_table

        self._init_cuda_tensors()

        self._init_grammar_backend()

        try:
            self.capture()
        except RuntimeError as e:
            raise Exception(
                f"PLD CUDA Graph capture failed: {e}\n"
                "Possible solutions:\n"
                "1. disable cuda graph by --disable-cuda-graph\n"
                "2. set --mem-fraction-static to a smaller value (e.g., 0.8 or 0.7)\n"
                "3. set --cuda-graph-max-bs to a smaller value (e.g., 32)\n"
            )

    def _init_cuda_tensors(self):
        """Initialize pre-allocated CUDA tensors for graph capture."""
        with torch.device("cuda"):
            self.req_pool_indices = torch.zeros((self.max_bs,), dtype=torch.int64)
            self.draft_token = torch.ones((self.max_bs * self.draft_token_num,), dtype=torch.int64)
            self.out_cache_loc_buffer = torch.arange(0, self.max_num_token, dtype=torch.int64)
            self.positions = torch.arange(0, self.max_num_token, dtype=torch.int64)
            self.seq_lens = torch.full((self.max_bs,), 1, dtype=torch.int32)

            # Sampling tensors if needed
            if self.capture_sample_graph:
                self.temperature_buffer = torch.zeros((self.max_bs, 1), dtype=torch.float32)
                self.topk_buffer = torch.zeros((self.max_bs,), dtype=torch.int32)
                self.topp_buffer = torch.zeros((self.max_bs,), dtype=torch.float32)
                self.minp_buffer = torch.zeros((self.max_bs,), dtype=torch.float32)

            # DP attention tensors if needed
            if self.enable_dp_attention:
                self.gathered_buffer = torch.zeros(
                    (self.max_num_token * self.tp_size,
                     self.target_worker.model_runner.model_config.hidden_size),
                    dtype=self.target_worker.model_runner.dtype,
                )
            if self.use_over_embedding:
                self.oe_column_starts = torch.empty([self.max_bs], dtype=torch.int32)
                self.oe_req_lens = torch.empty([self.max_bs], dtype=torch.int32)
                self.oe_out_column_starts = torch.empty([self.max_bs], dtype=torch.int32)
                self.oe_out_req_lens = torch.empty([self.max_bs], dtype=torch.int32)
                token_positions = torch.arange(self.draft_token_num, dtype=torch.int32).view(1, -1)
                self._token_positions_2d = token_positions.repeat(self.max_bs, 1)
                self._order_keys = torch.empty(
                    (self.max_bs, self.draft_token_num),
                    dtype=torch.int32,
                )

    def _init_grammar_backend(self):
        """Initialize grammar backend if enabled."""
        self.grammar_backend = None
        if self._enable_grammar_backend():
            server_args = self.target_worker.model_runner.server_args
            self.tokenizer = get_tokenizer(
                server_args.tokenizer_path,
                tokenizer_mode=server_args.tokenizer_mode,
                trust_remote_code=server_args.trust_remote_code,
                revision=server_args.revision,
            )
            self.grammar_backend = create_grammar_backend(
                server_args,
                self.tokenizer,
                self.target_worker.model_runner.model_config.vocab_size,
            )
            dummy_grammar = self.grammar_backend.init_value(("regex", r"[a-z]+"))
            self.vocab_masks = dummy_grammar.allocate_vocab_mask(
                vocab_size=self.target_worker.model_runner.model_config.vocab_size,
                batch_size=self.max_bs * self.draft_token_num,
                device="cpu",
            )
            self.vocab_masks = self.vocab_masks.to("cuda")

    def can_run(self, forward_batch: ForwardBatch) -> bool:
        """Check if CUDA Graph can be used for this batch."""
        if not (forward_batch.forward_mode.is_target_verify() or forward_batch.forward_mode.is_idle()):
            return False

        # Check batch size support
        if self.enable_dp_attention:
            min_num_tokens, max_num_tokens = (
                min(forward_batch.global_num_tokens),
                max(forward_batch.global_num_tokens),
            )
            max_bsz = int(max_num_tokens / self.draft_token_num)
            is_bs_supported = forward_batch.all_decode_or_idle and (
                (min_num_tokens == max_num_tokens and max_bsz in self.graphs)
                if self.disable_padding
                else max_bsz <= self.max_bs
            )
        else:
            is_bs_supported = (
                forward_batch.batch_size in self.graphs
                if self.disable_padding
                else forward_batch.batch_size <= self.max_bs
            )

        return is_bs_supported

    def replay(self, forward_batch: ForwardBatch, vocab_masks: Optional[torch.Tensor]):
        """
        Replay CUDA Graph for target model verification.

        NOTE: This ONLY does target model verification. Draft token generation
        happens outside in _postprocess_cuda_graph_output().
        """
        assert forward_batch.out_cache_loc is not None
        raw_bs = forward_batch.batch_size
        raw_num_token = raw_bs * self.draft_token_num

        if self.enable_dp_attention:
            index = bisect.bisect_left(
                self.capture_bs,
                max(forward_batch.global_num_tokens) // self.draft_token_num,
            )
        else:
            index = bisect.bisect_left(self.capture_bs, raw_bs)
        bs = self.capture_bs[index]

        self.req_pool_indices.zero_()
        if bs != raw_bs:
            self.seq_lens.fill_(1)
            self.out_cache_loc_buffer.zero_()
            self.positions.zero_()
            self.pld_worker.context_tokens[raw_bs:bs].zero_()
            self.pld_worker.context_lens[raw_bs:bs].zero_()

        if forward_batch.forward_mode.is_idle():
            forward_batch.spec_info = self._init_verify_spec_info(bs=0)
            forward_batch.forward_mode = ForwardMode.TARGET_VERIFY

        assert forward_batch.forward_mode.is_target_verify()

        if self.use_over_embedding:
            forward_batch.oe_column_starts[:raw_bs] = forward_batch.req_to_token_pool.verified_lens[
                forward_batch.req_pool_indices
            ]
            forward_batch.oe_req_lens[:raw_bs] = self.draft_token_num

        self._copy_inputs_to_graph_tensors(forward_batch, vocab_masks, raw_bs, raw_num_token)
        self._copy_context_to_graph_tensors(forward_batch, raw_bs)
        self._init_attention_backend_for_replay(forward_batch, bs)

        self.graphs[bs].replay()
        out = self.output_buffers[bs]

        # Post-process output if padding was used
        if bs != raw_bs:
            forward_batch.batch_size = raw_bs
            forward_batch.positions = self.positions[:raw_num_token]
            forward_batch.seq_lens = self.seq_lens[:raw_bs]
            forward_batch.req_pool_indices = self.req_pool_indices[:raw_bs]
            out = self._postprocess_output_to_raw_bs(out, raw_bs)

        return out

    def _copy_inputs_to_graph_tensors(self, forward_batch: ForwardBatch, vocab_masks: Optional[torch.Tensor],
                                    raw_bs: int, raw_num_token: int):
        """Copy inputs from forward_batch to pre-allocated graph tensors."""
        self.draft_token[:raw_num_token].copy_(forward_batch.spec_info.draft_token)
        self.req_pool_indices[:raw_bs].copy_(forward_batch.req_pool_indices)
        self.seq_lens[:raw_bs].copy_(forward_batch.seq_lens)
        self.out_cache_loc_buffer[:raw_num_token].copy_(forward_batch.out_cache_loc)
        self.positions[:raw_num_token].copy_(forward_batch.positions)
        if self.use_over_embedding:
            self.oe_column_starts[:raw_bs].copy_(forward_batch.oe_column_starts[:raw_bs])
            self.oe_req_lens[:raw_bs].copy_(forward_batch.oe_req_lens[:raw_bs])

        if vocab_masks is not None:
            self.vocab_masks[:raw_num_token].copy_(vocab_masks)
        elif self.grammar_backend:
            self.grammar_backend.reset_vocab_masks(self.vocab_masks[:raw_num_token])

        if self.capture_sample_graph:
            sampling_info = forward_batch.sampling_info
            self.temperature_buffer[:raw_bs].copy_(sampling_info.temperatures)
            self.topk_buffer[:raw_bs].copy_(sampling_info.top_ks)
            self.topp_buffer[:raw_bs].copy_(sampling_info.top_ps)
            self.minp_buffer[:raw_bs].copy_(sampling_info.min_ps)

    def _init_attention_backend_for_replay(self, forward_batch: ForwardBatch, bs: int):
        """Initialize attention backend for CUDA Graph replay."""
        self.target_worker.model_runner.attn_backend.init_forward_metadata_replay_cuda_graph(
            bs,
            self.req_pool_indices,
            self.seq_lens,
            forward_batch.seq_lens_sum + (bs - forward_batch.batch_size),
            forward_batch.forward_mode,
            forward_batch.spec_info,
            seq_lens_cpu=None,
        )

    def capture(self):
        """Capture CUDA Graphs for different batch sizes."""
        if self.enable_torch_compile:
            set_torch_compile_config()

        with graph_capture() as graph_capture_context:
            self.stream = graph_capture_context.stream
            capture_range = (
                tqdm.tqdm(self.capture_bs)
                if get_tensor_model_parallel_rank() == 0
                else self.capture_bs
            )
            for bs in capture_range:
                with patch_model(
                    self.target_worker.model_runner.model,
                    bs in self.compile_bs,
                    num_tokens=bs * self.draft_token_num,
                    tp_group=self.target_worker.model_runner.tp_group,
                ) as forward:
                    graph, output_buffers = self._capture_one_batch_size(bs, forward)
                    self.graphs[bs] = graph
                    self.output_buffers[bs] = output_buffers

    def _capture_one_batch_size(self, bs: int, forward):
        """Capture CUDA Graph for a specific batch size."""
        graph = torch.cuda.CUDAGraph()
        stream = self.stream

        forward_batch = self._get_forward_batch(bs)

        # Initialize attention backend for capture
        self.target_worker.model_runner.attn_backend.init_forward_metadata_capture_cuda_graph(
            bs,
            bs * self.draft_token_num,
            forward_batch.req_pool_indices,
            forward_batch.seq_lens,
            forward_batch.forward_mode,
            forward_batch.spec_info,
        )

        # Define the function to capture
        def run_once():
            vocab_masks = None
            if self._enable_grammar_backend():
                vocab_masks = self.vocab_masks[: (bs * self.draft_token_num)]

            if self.use_over_embedding:
                # Prewrite verify tokens so OE n-gram for token2 can see token1.
                self.oe_out_column_starts[:bs] = self.target_worker.model_runner.req_to_token_pool.verified_lens[
                    forward_batch.req_pool_indices
                ]
                self.oe_out_req_lens[:bs] = self.draft_token_num
                update_token_table(
                    oe_token_table=forward_batch.oe_token_table,
                    tokens=forward_batch.input_ids.to(torch.int32),
                    row_indices=forward_batch.req_pool_indices,
                    column_starts=self.oe_out_column_starts[:bs],
                    oe_req_lens=self.oe_out_req_lens[:bs],
                )

            logits_output = forward(
                forward_batch.input_ids,
                forward_batch.positions,
                forward_batch
            )

            target_predict, logits_output, accept_length, accept_index = (
                self.pld_worker.rejection_sampling(forward_batch, logits_output, vocab_masks)
            )
            if self.use_over_embedding:
                # Only commit accepted tokens only
                self.oe_out_column_starts[:bs] = self.target_worker.model_runner.req_to_token_pool.verified_lens[
                    forward_batch.req_pool_indices
                ]
                self.oe_out_req_lens[:bs] = accept_length
                update_token_table(
                    oe_token_table=forward_batch.oe_token_table,
                    tokens=target_predict,
                    row_indices=forward_batch.req_pool_indices,
                    column_starts=self.oe_out_column_starts[:bs],
                    oe_req_lens=self.oe_out_req_lens[:bs],
                )

            new_verified_id = self.pld_worker.preprocess_for_draft_after_decode(
                forward_batch, accept_length, accept_index, target_predict, with_draft_model=False
            )
            output_ids = target_predict[accept_index]

            req_pool_indices = forward_batch.req_pool_indices
            self.target_worker.model_runner.req_to_token_pool.verified_lens[req_pool_indices] += accept_length

            token_list = self.pld_worker.propose(
                forward_batch, target_predict, accept_length
            )

            return (
                logits_output,
                output_ids,
                accept_length,
                new_verified_id,
                token_list,
            )

        # Warm up runs
        for _ in range(4):
            torch.cuda.synchronize()
            self.target_worker.model_runner.tp_group.barrier()
            # Clear buffers before each warm-up run
            run_once()
            forward_batch = self._get_forward_batch(bs)

        torch.cuda.synchronize()
        self.target_worker.model_runner.tp_group.barrier()

        # Capture the graph
        global global_graph_memory_pool
        with torch.cuda.graph(graph, pool=global_graph_memory_pool, stream=stream):
            out = run_once()

        torch.cuda.synchronize()
        self.target_worker.model_runner.tp_group.barrier()

        global_graph_memory_pool = graph.pool()
        return graph, out

    def _get_forward_batch(self, bs: int) -> ForwardBatch:
        """Create a ForwardBatch for graph capture/replay."""
        input_ids = self.draft_token[: bs * self.draft_token_num]
        req_pool_indices = self.req_pool_indices[:bs]
        seq_lens = self.seq_lens[:bs]
        out_cache_loc = self.out_cache_loc_buffer[: bs * self.draft_token_num]
        positions = self.positions[: bs * self.draft_token_num]
        if self.use_over_embedding:
            self.oe_column_starts[:bs].zero_()
            self.oe_req_lens[:bs].fill_(self.draft_token_num)

        if self.enable_dp_attention:
            global_num_tokens = [bs * self.draft_token_num] * self.tp_size
            gathered_buffer = self.gathered_buffer[: bs * self.tp_size * self.draft_token_num]
            global_batch_size = [bs] * self.tp_size
        else:
            global_num_tokens = None
            gathered_buffer = None
            global_batch_size = None

        verify_spec_info = self._init_verify_spec_info(bs)

        forward_batch = ForwardBatch(
            forward_mode=ForwardMode.TARGET_VERIFY,
            batch_size=bs,
            input_ids=input_ids,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            req_to_token_pool=self.target_worker.model_runner.req_to_token_pool,
            token_to_kv_pool=self.target_worker.model_runner.token_to_kv_pool,
            attn_backend=self.target_worker.model_runner.attn_backend,
            out_cache_loc=out_cache_loc,
            seq_lens_sum=seq_lens.sum(),
            oe_token_table=self.token_table if self.use_over_embedding else None,
            oe_column_starts=self.oe_column_starts[: bs] if self.use_over_embedding else None,
            oe_req_lens=self.oe_req_lens[: bs] if self.use_over_embedding else None,
            oe_out_column_starts=self.oe_out_column_starts[: bs] if self.use_over_embedding else None,
            oe_out_req_lens=self.oe_out_req_lens[: bs] if self.use_over_embedding else None,
            return_logprob=False,
            positions=positions,
            global_num_tokens=global_num_tokens,
            gathered_buffer=gathered_buffer,
            mrope_positions=None,
            spec_algorithm=self.target_worker.model_runner.spec_algorithm,
            spec_info=verify_spec_info,
            capture_hidden_mode=CaptureHiddenMode.FULL,
            all_decode_or_idle=True,
            draft_input_ids=None,
            new_tokens_to_compute=torch.full((bs,), 1, device="cuda", dtype=torch.int32),
            global_batch_size=global_batch_size,
        )

        if self.capture_sample_graph:
            forward_batch.sampling_info = SamplingBatchInfo(
                temperatures=self.temperature_buffer[:bs],
                top_ks=self.topk_buffer[:bs],
                top_ps=self.topp_buffer[:bs],
                min_ps=self.minp_buffer[:bs],
                is_all_greedy=False,
                need_min_p_sampling=False,
                vocab_size=self.target_worker.model_runner.model_config.vocab_size,
            )

        return forward_batch

    def _init_verify_spec_info(self, bs: int):
        """Initialize EagleVerifyInput for PLD compatibility."""
        from sglang.srt.speculative.eagle_utils import EagleVerifyInput

        grammar = None
        if self.grammar_backend:
            grammar = self.grammar_backend.init_value(("regex", r".*"))

        spec_info = EagleVerifyInput(
            draft_token=self.draft_token[: bs * self.draft_token_num],
            positions=self.positions[: bs * self.draft_token_num],
            draft_token_num=self.draft_token_num,
            spec_steps=self.pld_worker.speculative_num_steps,
            capture_hidden_mode=CaptureHiddenMode.FULL,
            is_all_greedy=False if self.capture_sample_graph else True,
            grammar=grammar,
        )
        return spec_info

    def _postprocess_output_to_raw_bs(self, out, raw_bs: int):
        """Post-process output to remove padding."""
        (
            logits_output,
            output_ids,
            accept_length,
            new_verified_id,
            token_list,
        ) = out

        # Trim outputs to raw batch size
        logits_output.next_token_logits = logits_output.next_token_logits[: raw_bs * self.draft_token_num]
        logits_output.hidden_states = logits_output.hidden_states[: raw_bs * self.draft_token_num]
        output_ids = output_ids[: raw_bs * self.draft_token_num]
        accept_length = accept_length[:raw_bs]
        new_verified_id = new_verified_id[:raw_bs]

        # Trim token_list if it exists (from n-gram matching)
        if token_list is not None:
            token_list = [t[:raw_bs] for t in token_list]

        return (
            logits_output,
            output_ids,
            accept_length,
            new_verified_id,
            token_list,
        )

    def _enable_grammar_backend(self) -> bool:
        """Check if grammar backend is enabled."""
        server_args = self.target_worker.model_runner.server_args
        return server_args.grammar_backend is not None

    def _copy_context_to_graph_tensors(self, forward_batch: ForwardBatch, raw_bs: int):
        """
        Copy context information (origin_input_ids + output_ids) to graph tensors.
        This is called during replay to prepare data for n-gram matching.

        Args:
            forward_batch: The forward batch containing request information
            raw_bs: Raw batch size (before padding)
        """
        self.pld_worker.context_tokens[:raw_bs].zero_()
        self.pld_worker.context_lens[:raw_bs].zero_()

        self.pld_worker._copy_context_to_buffers(forward_batch, raw_bs)
