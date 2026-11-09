from __future__ import annotations

import dataclasses
import os
import threading
import time
from abc import ABC, abstractmethod
from contextlib import contextmanager
from queue import Queue
from typing import TYPE_CHECKING

import torch
from huggingface_hub import snapshot_download

from sglang.srt.env import ENV, global_server_args_dict
from sglang.srt.layers.sampler.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.eagle_utils import (
    EagleDraftInput,
    EagleDraftOutput,
    EagleVerifyInput,
    cumulate_output_tokens_kernel,
    cumulate_output_tokens_pytorch,
    generate_token_bitmask,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.utils import get_available_gpu_memory, get_colorful_logger, is_npu

if TYPE_CHECKING:
    from sglang.srt.managers.scheduling.schedule_batch import ModelWorkerBatch
    from sglang.srt.managers.tp_worker import TpModelWorker
    from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
    from sglang.srt.mem_cache.memory_pool.kv_allocator import KVAllocator
    from sglang.srt.mem_cache.memory_pool.memory_pool_base import BaseTokenToKVPool
    from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.constrained.base_grammar_backend import create_grammar_backend
from sglang.srt.hostfunc import launch_hostfunc
from sglang.srt.managers.eplb.expert_distribution import (
    get_global_expert_distribution_recorder,
)
from sglang.srt.speculative.grammar_hostfunc import (
    GrammarGraphCtx,
    accept_hostfunc_eager,
    dfs_hostfunc_eager,
)
from sglang.srt.utils.hf_transformers_utils import get_tokenizer

logger = get_colorful_logger(__name__)

__is_npu__ = is_npu()

if __is_npu__:
    from sglang.srt.speculative.npu_draft_graph_runner import (
        NpuDraftGraphRunner,
    )

    # NPU 侧不编译 mamba/GDN 混合 backend, hybrid_linear_attn 分支永远走不到,
    # 用 () 占位使 isinstance 恒 False, 避免 NameError。
    HybridLinearAttnBackend = ()
else:
    from sglang.srt.layers.attention.hybrid_linear_attn_backend import HybridLinearAttnBackend


class BaseSpecDeocdingWorker(ABC):
    """
    Base class for speculative decoding workers.

    This abstract base class provides the foundation for implementing speculative decoding
    algorithms (e.g., EAGLE, EAGLE3, PLD, NEXTN).

    The worker manages:
    - Draft model initialization
    - CUDA graph capture
    - Attention backend configuration for multi-step decoding
    - Token proposal and verification workflow
    - Memory pool sharing with the target worker (KV cache and token pools)

    Attributes:
        server_args: Server configuration including speculative decoding parameters
        target_worker: The main model worker that performs final token verification
        device: GPU device for computation
        speculative_algorithm: The specific algorithm being used (EAGLE, EAGLE3, PDL, etc.)
        topk: Top-k candidates for tree attention (currently fixed at 1)
        speculative_num_steps: Number of speculative decoding steps
        use_over_embedding: Whether to use over-embedding
        req_to_token_pool: Shared token pool with target worker
        kv_allocator: Shared KV cache allocator with target worker
        oe_token_table: Over-embedding token lookup table
        cuda_graph_runner: CUDA graph runner for optimized execution
        draft_attn_backend: Attention backend for draft model (may be different with main model)

    Subclasses must implement:
        - propose(): Generate draft tokens using the draft model
        - forward_prefill_spec(): Handle prefill stage for speculative decoding
        - forward_decode_spec(): Handle decode stage with token verification
        - forward_idle(): Handle idle forward passes (DP Attention)
    """

    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        target_worker: TpModelWorker,
        drafter_use_oe: bool = False,
    ) -> None:
        self.server_args = server_args
        self.target_worker = target_worker
        self.device = target_worker.device
        self.speculative_algorithm = SpeculativeAlgorithm.from_string(server_args.speculative_algorithm)
        self.topk = server_args.speculative_eagle_topk
        assert self.topk == 1, "Tree Attention is abandoned for now."
        self.speculative_num_steps = server_args.speculative_num_steps
        self.drafter_use_oe = drafter_use_oe
        self.use_over_embedding = drafter_use_oe or self.target_worker.use_over_embedding
        # Share req_to_token_pool and kv_allocator with target worker
        self.req_to_token_pool: ReqToTokenPool = self.target_worker.model_runner.req_to_token_pool
        self.kv_allocator: KVAllocator = self.target_worker.model_runner.kv_allocator
        self.token_to_kv_pool: BaseTokenToKVPool = self.target_worker.model_runner.token_to_kv_pool
        self.oe_token_table = self.target_worker.model_runner.oe_token_table
        self.vocab_size = self.target_worker.model_config.vocab_size
        self._init_penalties()

        self.npu_graph_runner_for_draft_extend = None
        self.npu_graph_runner_for_draft_extends = []

        self.cuda_graph_runner = None

        # Grammar overlap machinery -- safe defaults so all subclasses have these
        # fields. EAGLEWorker._init_grammar_overlap_for_cuda_graph() builds the
        # real objects when a grammar backend + overlap scheduler are configured;
        # other workers (e.g. PLD) leave them disabled.
        self.grammar_overlap_enabled = False
        self.grammar_backend = None
        self.grammar_ctx = None
        self.grammar_stream = None
        self.slot_scalar = None
        self.vocab_masks = None
        self._grammar_slot_count = 0
        self.grammar_ring_depth = 1
        self.last_grammar_slot = -1

        # Async grammar update (NPU only): a dedicated thread runs
        # update_grammar_after_verify so the forward thread is not blocked by
        # the verified-token D2H copy sync and the CPU grammar accept loop.
        # The queues/thread are created by start_grammar_update_thread();
        # until then wait_grammar_update_done() is a no-op.
        self.grammar_update_queue = None
        self.grammar_update_done_queue = None
        self.grammar_update_inflight = 0

    def _init_grammar_overlap_for_cuda_graph(self):
        """Build the grammar machinery shared by the graph and eager verify
        paths, owned by the worker so it exists even when cuda graph is disabled.

        Creates (all on ``self``): the grammar backend + tokenizer, the device
        ``vocab_masks`` buffer, the stable ``grammar_ctx`` (host-node buffers),
        the dedicated ``grammar_stream``, the ``slot_scalar`` device scalar (graph
        path slot read-back), and the shared monotonic slot allocator. Gated on
        ``grammar_overlap_enabled`` (grammar configured AND overlap scheduler);
        the non-overlap / disagg paths keep the out-of-graph grammar flow.

        Defined here (not on a specific worker) because the machinery and its
        consumer (_prepare_eager_grammar_verify) are grammar-generic, not
        EAGLE-specific. Subclasses that want grammar overlap call this at the
        right time (e.g. EAGLEWorker before cuda graph capture); workers that do
        not call it keep the disabled defaults set in __init__.
        """
        server_args = self.model_runner.server_args
        if __is_npu__:
            return
        # Grammar host nodes (DFS + accept) run on the grammar stream only under
        # the overlap scheduler -- that is the path whose accept this reroutes.
        self.grammar_overlap_enabled = server_args.grammar_backend is not None and not server_args.disable_overlap_schedule
        if server_args.grammar_backend is None:
            return

        draft_token_num = server_args.speculative_num_draft_tokens
        # Size shared buffers for the max decode batch per DP rank -- the
        # scheduler's hard cap, already divided by dp_size in ModelRunner. This
        # covers eager-fallback bs that exceeds cuda_graph_max_bs.
        grammar_max_bs = self.model_runner.max_running_requests

        self.tokenizer = get_tokenizer(
            server_args.tokenizer_path,
            tokenizer_mode=server_args.tokenizer_mode,
            trust_remote_code=server_args.trust_remote_code,
            revision=server_args.revision,
        )
        self.grammar_backend = create_grammar_backend(
            server_args,
            self.tokenizer,
            self.model_runner.model_config.vocab_size,
        )
        dummy_grammar = self.grammar_backend.init_value_impl(("regex", r"[a-z]+"))
        self.vocab_masks = dummy_grammar.allocate_vocab_mask(
            vocab_size=self.model_runner.model_config.vocab_size,
            batch_size=grammar_max_bs * draft_token_num,
            device="cpu",
        ).to("cuda")

        if self.grammar_overlap_enabled:
            # Ring depth for the per-step grammar state under launch-ahead: the
            # consumer reads replay(N)'s snapshot while the forward thread has
            # already launched replay(N+1..); the slot must not be reused until
            # the consumer is done. 2*num_continuous_decode_steps+1 leaves margin.
            ncds = getattr(server_args, "num_continuous_decode_steps", 1) or 1
            self.grammar_ring_depth = 2 * ncds + 1
            # Device scalar holding the slot of the graph replay being launched;
            # the captured graph copies it D2H into ctx.slot_host so each host
            # node picks its own replay's slot. Eager passes slot via closure and
            # does not use this.
            self.slot_scalar = torch.zeros((1,), dtype=torch.int32, device="cuda")
            self.grammar_ctx = GrammarGraphCtx(
                max_bs=grammar_max_bs,
                draft_token_num=draft_token_num,
                vocab_masks_device=self.vocab_masks,
                ring_depth=self.grammar_ring_depth,
            )
            # Dedicated grammar stream: every grammar host node (graph & eager)
            # is forked onto this ONE stream so DFS/accept are FIFO-serialized on
            # the matcher regardless of graph vs eager path.
            self.grammar_stream = torch.cuda.Stream()

    def alloc_slot(self) -> int:
        """Return the next ring slot from a single monotonic counter shared by
        the graph and eager paths, so their slots never collide."""
        slot = self._grammar_slot_count % self.grammar_ring_depth
        self._grammar_slot_count += 1
        return slot

    @property
    def draft_model_runner(self):
        return self.model_runner

    def init_npu_graphs(self):
        """Capture npu graphs."""
        self.npu_graph_runner_for_draft_extend = None

        def create_npu_graph_runner(idx=0):
            tic = time.perf_counter()
            before_mem = get_available_gpu_memory(self.device, self.model_runner.gpu_id)
            logger.info(f"Capture draft extend npu graph {idx} begin. This can take up to several minutes. avail mem={before_mem:.2f} GB")
            npu_graph_runner_for_draft_extend = NpuDraftGraphRunner(self, draft_model_idx=idx)

            after_mem = get_available_gpu_memory(self.device, self.model_runner.gpu_id)
            logger.info(
                f"Capture draft extend npu graph {idx} end. "
                f"Time elapsed: {time.perf_counter() - tic:.2f} s. "
                f"mem usage={(before_mem - after_mem):.2f} GB. "
                f"avail mem={after_mem:.2f} GB."
            )
            return npu_graph_runner_for_draft_extend

        self.npu_graph_runner_for_draft_extend = create_npu_graph_runner()

        if self.server_args.is_multi_head_eagle:
            self.npu_graph_runner_for_draft_extends.append(self.npu_graph_runner_for_draft_extend)
            for idx in range(1, self.speculative_num_steps):
                npu_graph_runner_for_draft_extend = create_npu_graph_runner(idx)
                self.npu_graph_runner_for_draft_extends.append(npu_graph_runner_for_draft_extend)

    def init_cuda_graphs(self, graph_runner_cls):
        """
        Initialize and capture CUDA graphs.

        Args:
            graph_runner_cls: The CUDA graph runner class to instantiate for this worker.

        Note:
            - This method can take several minutes to complete as it captures all graph variants
            - CUDA graphs are disabled if server_args.disable_cuda_graph is True
            - Memory usage before and after capture is logged for monitoring
        """
        self.cuda_graph_runner = None

        if self.server_args.disable_cuda_graph:
            return

        tic = time.time()
        logger.info("Capture cuda graph begin. This can take up to several minutes.")
        before_capture_available_gpu_memory = get_available_gpu_memory(self.device, self.target_worker.model_runner.gpu_id)
        logger.info(
            f"Capture cuda graph begin. This can take up to several minutes. avail mem={before_capture_available_gpu_memory:.2f} GB in model runner!"
        )
        self.cuda_graph_runner = graph_runner_cls(self)
        after_capture_available_gpu_memory = get_available_gpu_memory(self.device, self.target_worker.model_runner.gpu_id)
        logger.info(f"Capture cuda graph end. Time elapsed: {time.time() - tic:.2f} s. avail mem={after_capture_available_gpu_memory:.2f} GB")
        logger.info(
            f"{len(self.cuda_graph_runner.graphs)} graphs used "
            f"mem={(before_capture_available_gpu_memory - after_capture_available_gpu_memory):.2f} GB"
        )
        self.target_worker.model_runner.graph_mem_usage_gb += before_capture_available_gpu_memory - after_capture_available_gpu_memory

    def init_drafter_embedding(self, drafter_model_runner: ModelRunner):
        """
        Initialize the draft model's embedding layers and lm head.

        Args:
            drafter_model_runner: The model runner for the draft model.

        Note:
            - Over-embedding is applied when enabled in server args
        """
        if self.server_args.speculative_token_map is not None:
            if os.path.exists(self.server_args.speculative_token_map):
                self.hot_token_id = torch.load(self.server_args.speculative_token_map)
            else:
                cache_dir = snapshot_download(
                    os.path.dirname(self.server_args.speculative_token_map),
                    ignore_patterns=["*.bin", "*.safetensors"],
                )
                file_path = os.path.join(cache_dir, os.path.basename(self.server_args.speculative_token_map))
                self.hot_token_id = torch.load(file_path)
            self.server_args.json_model_override_args = f'{{"hot_vocab_size": {len(self.hot_token_id)}}}'

        embed, head = self.target_worker.model_runner.model.get_embed_and_head()
        if self.speculative_algorithm.is_eagle3():
            if self.server_args.speculative_token_map is not None:
                logger.warning("Speculative token map specified, but EAGLE3 models already have this. Ignoring the specified token map.")
            self.hot_token_id = None
        elif self.server_args.speculative_token_map is not None:
            head = head.clone()
            self.hot_token_id = torch.tensor(self.hot_token_id, dtype=torch.int32, device=head.device)
            head.data = head.data[self.hot_token_id]
        else:
            self.hot_token_id = None

        if self.speculative_algorithm.is_eagle3():
            word_embed = embed.word_embeder.weight if self.target_worker.use_over_embedding else embed
            drafter_model_runner.model.set_embed(word_embed)
            self.hot_token_id = drafter_model_runner.model.get_hot_token_id().to(word_embed.device)
        else:
            if self.drafter_use_oe:
                assert self.target_worker.use_over_embedding
                drafter_model_runner.model.set_oe_and_head(embed, head)
            else:
                if self.target_worker.use_over_embedding and hasattr(drafter_model_runner.model, "set_embed_module_and_head"):
                    drafter_model_runner.model.set_embed_module_and_head(embed.word_embeder, head)
                else:
                    word_embed = embed.word_embeder.weight if self.target_worker.use_over_embedding else embed

                    drafter_model_runner.model.set_embed_and_head(word_embed, head)

    def init_drafter_attention_backends(self, draft_model_runner: ModelRunner) -> None:
        """
        Init attention backend for multi-step decode of draft model, model-based algorithms
        will need this when spec_steps > 1 (like NEXTN and EAGLE3). For model-free algo(like
        PLD), this method should be skipped.
        """
        # Create multi-step attn backends and cuda graph runners
        drafter_backend = self.server_args.drafter_attention_backend
        self.drafter_backend = drafter_backend
        if drafter_backend == "flashinfer":
            from sglang.srt.layers.attention.flashinfer_backend import (
                FlashInferMultiStepDraftBackendForTVDGraph,
            )

            self.draft_attn_backend = FlashInferMultiStepDraftBackendForTVDGraph(
                draft_model_runner,
                self.speculative_num_steps,
            )
        elif drafter_backend == "triton":
            from sglang.srt.layers.attention.triton_backend import (
                TritonMultiStepDraftBackend,
            )

            self.draft_attn_backend = TritonMultiStepDraftBackend(
                draft_model_runner,
                self.topk,
                self.speculative_num_steps,
            )
        elif drafter_backend == "flashmla":
            from sglang.srt.layers.attention.flashmla_backend import (
                FlashMLAMultiStepDecodeBackend,
            )

            self.draft_attn_backend = FlashMLAMultiStepDecodeBackend(draft_model_runner, self.speculative_num_steps)
        elif drafter_backend == "flashinfer_mla":
            from sglang.srt.layers.attention.flashinfer_mla_backend import (
                FlashInferMLAMultiStepDraftBackend,
            )

            self.draft_attn_backend = FlashInferMLAMultiStepDraftBackend(
                draft_model_runner,
                self.topk,
                self.speculative_num_steps,
            )
        elif drafter_backend == "hybrid_linear_attn":
            from sglang.srt.layers.attention.flashinfer_backend import (
                FlashInferMultiStepDraftBackendForTVDGraph,
            )

            self.draft_attn_backend = FlashInferMultiStepDraftBackendForTVDGraph(
                draft_model_runner,
                self.speculative_num_steps,
            )
        elif drafter_backend == "dsa":
            from sglang.srt.layers.attention.dsa_backend import (
                DpskSparseAttnMultiStepBackend,
            )

            self.draft_attn_backend = DpskSparseAttnMultiStepBackend(draft_model_runner, self.topk, self.speculative_num_steps)
        elif drafter_backend == "npu":
            logger.info("========eagle select NpuAttnBackend=============")
            from sglang.srt.layers.attention.npu_mla_backend import NPUMultiStepDecodeBackend

            self.draft_attn_backend = NPUMultiStepDecodeBackend(draft_model_runner, self.topk, self.speculative_num_steps)
        elif self.server_args.attention_backend == "npu_mla":
            logger.info("========eagle select NpuMLAAttnBackend=============")
            from sglang.srt.layers.attention.npu_mla_backend import NPUMLAMultiStepDecodeBackend

            self.draft_attn_backend = NPUMLAMultiStepDecodeBackend(draft_model_runner, self.topk, self.speculative_num_steps)
        else:
            raise ValueError(f"EAGLE is not supported with drafter attention backend {drafter_backend}")

    @torch.inference_mode()
    def rejection_sampling(
        self,
        forward_batch: ForwardBatch,
        logits_output: LogitsProcessorOutput,
        vocab_masks: torch.Tensor | None = None,
    ):
        """
        Perform rejection sampling to verify draft tokens against target model predictions.

        This method compares the draft tokens proposed by the speculative model with the
        target model's logits to determine which tokens should be accepted or rejected.

        Args:
            forward_batch: The forward batch containing draft tokens and sequence information.
            logits_output: The logits output from the target model.
            vocab_masks: Optional vocabulary masks for constrained generation (e.g., grammar).

        Returns:
            A tuple containing:
                - predict: The predicted tokens after rejection sampling
                - logits_output: Updated logits output
                - accept_length: Number of tokens accepted for each sequence (bonus token is not counted for now!)
                - accept_index: Indices of accepted tokens
        """
        assert isinstance(forward_batch.spec_info, EagleVerifyInput)
        forward_batch.spec_info.hidden_states = logits_output.hidden_states
        if vocab_masks is None and __is_npu__:
            # Caution! grammars vocab_mask generation on CPU, overlap with NPU
            # Wait for the previous step's async grammar update (if any) so
            # that the vocab masks are built on up-to-date grammar state.
            self.wait_grammar_update_done()
            vocab_masks = self._generate_vocab_mask(forward_batch, forward_batch.spec_info)
            # vocab_masks was generated with ori_bs; pad to match padded batch_size.
            # Use -1 (all bits set = all tokens allowed) for padded dummy rows.
            if vocab_masks is not None:
                ori_bs = getattr(forward_batch, "ori_bs", forward_batch.batch_size)
                padding_size = forward_batch.batch_size
                draft_token_num = self.model_runner.server_args.speculative_num_draft_tokens
                padded_rows = padding_size * draft_token_num
                ori_rows = ori_bs * draft_token_num
                if padded_rows > ori_rows:
                    pad_mask = torch.full(
                        (padded_rows - ori_rows, vocab_masks.shape[1]),
                        -1,
                        dtype=vocab_masks.dtype,
                        device=vocab_masks.device,
                    )
                    vocab_masks = torch.cat([vocab_masks, pad_mask], dim=0)

        predict, logits_output, accept_length, accept_index = forward_batch.spec_info.verify(forward_batch, logits_output, vocab_masks)
        self._cumulate_output_tokens(forward_batch, predict, accept_length)
        # Store accept_index on logits_output for CUDA graph output buffer path
        logits_output.spec_accept_index = accept_index
        return predict, logits_output, accept_length, accept_index

    @torch.inference_mode()
    def preprocess_for_draft_after_decode(
        self,
        forward_batch: ForwardBatch,
        accept_length: torch.Tensor,
        accept_index: torch.Tensor,
        target_predict: torch.Tensor,
        with_draft_model: bool = True,
    ):
        """
        Preprocess the forward batch for draft model execution after token verification.

        After the target model verifies draft tokens through rejection sampling, this method
        prepares the batch for the next round of draft token generation. It updates the batch
        state with accepted tokens and configures the appropriate KV cache and attention backend.

        Args:
            forward_batch: The forward batch to preprocess.
            accept_length: Number of tokens accepted for each sequence in the batch.
            accept_index: Indices of accepted tokens in the draft sequence.
            target_predict: The predicted tokens from the target model after verification.
            with_draft_model: If True, use the draft model's KV cache and attention backend;
                            if False, use the target model's resources (for model-free algorithms).

        Returns:
            new_verified_id: The newly verified token IDs to be used as input for draft generation.

        Note:
            - Sets forward mode to DRAFT_EXTEND for draft model execution
            - Captures only the last hidden state (LAST mode) for efficiency
            - Configures the appropriate token-to-KV pool mapping based on with_draft_model flag
            - Creates EagleDraftInput with rearranged hidden states and acceptance information
        """
        forward_batch.forward_mode = ForwardMode.DRAFT_EXTEND
        forward_batch.capture_hidden_mode = CaptureHiddenMode.LAST
        forward_batch.extend_seq_lens = accept_length
        if with_draft_model:
            forward_batch.token_to_kv_pool = self.model_runner.token_to_kv_pool
            forward_batch.attn_backend = self.model_runner.attn_backend
        else:
            forward_batch.token_to_kv_pool = self.target_worker.model_runner.token_to_kv_pool
        draft_input = EagleDraftInput()
        # rearranged hidden states
        draft_input.hidden_states = forward_batch.spec_info.hidden_states
        draft_input.accept_length = accept_length
        draft_input.verified_id = target_predict
        draft_input.draft_token_num = self.server_args.speculative_num_draft_tokens
        draft_input.accept_index = accept_index
        new_verified_id = draft_input.prepare_extend_after_decode(forward_batch, self.use_over_embedding)
        forward_batch.spec_info = draft_input
        return new_verified_id

    def preprocess_for_verify(self, forward_batch: ForwardBatch) -> None:
        """
        Preprocess the forward batch to prepare for token verification.

        This method transforms the draft output into a format suitable for verification by
        the target model. It sets up sequence lengths, positions, and input tokens, and
        resolves the grammar matcher if grammar constraints are present.

        Args:
            forward_batch: The forward batch containing draft output from the proposal stage.

        Note:
            - Converts EagleDraftOutput to EagleVerifyInput for the verification stage
            - Resolves the async grammar matcher onto the verify spec info, but does
              NOT build the vocab bitmask -- the caller decides that (via
              ``_build_vocab_bitmask`` for the sync path, or the DFS host node for
              the overlap path), so the skip decision is visible where it is made.
            - Updates batch metadata (seq_lens, positions, input_ids) for verification
        """
        assert isinstance(forward_batch.spec_info, EagleDraftOutput)
        forward_batch.seq_lens = self.req_to_token_pool.verified_lens[forward_batch.req_pool_indices]
        forward_batch.extend_seq_lens = forward_batch.new_tokens_to_compute
        token_list = forward_batch.spec_info.token_list
        if isinstance(token_list, list):
            token_list = torch.cat(token_list, dim=1)
        verify_spec_info = EagleVerifyInput.create(
            forward_batch.spec_info.last_verified_ids,
            token_list,
            forward_batch.seq_lens,
            self.speculative_num_steps,
            self.server_args.speculative_num_draft_tokens,
            forward_batch.sampling_info.is_all_greedy,
            False,
        )

        if __is_npu__:
            if forward_batch.sampling_info.grammars:
                # make draft token D2H transfer overlap with the CPU tensor construction
                assert verify_spec_info.draft_token is not None
                verify_spec_info.draft_token_cpu_copy_done = torch.get_device_module(self.device).Event()
                verify_spec_info.draft_token_cpu = verify_spec_info.draft_token.to("cpu", non_blocking=True)
                verify_spec_info.draft_token_cpu_copy_done.record()
        else:
            # Attach the resolved grammar matcher onto the verify spec info. The
            # bitmask itself is built at the verify call site (via
            # _build_vocab_bitmask for the sync path, or the DFS host node for the
            # overlap path) -- preprocess never builds it.
            self._resolve_grammar(forward_batch, verify_spec_info)

        verify_spec_info.cumulated_scaling_penalties = self._generate_penalties(forward_batch)
        forward_batch.spec_info = verify_spec_info
        forward_batch.input_ids = forward_batch.spec_info.draft_token
        forward_batch.capture_hidden_mode = CaptureHiddenMode.FULL
        forward_batch.positions = forward_batch.spec_info.positions
        if __is_npu__:
            forward_batch.set_npu_ep_metadata(self.target_worker.model_runner)
        return

    def _init_penalties(self):
        # NOTE: allocate one extra row (size +1) as a dummy sink row. The D2H-free
        # cumulate_output_tokens_pytorch redirects invalid (rejected/padding)
        # token positions to this row, so no boolean-mask indexing (nonzero)
        # is needed. All accessors use req_pool_indices which never point to
        # this row, so it is write-only and never read.
        self.cumulated_scaling_penalty = torch.ones(
            (self.req_to_token_pool.size + 1, self.vocab_size),
            dtype=torch.float32,
            device=self.device,
        )

    def _cumulate_output_tokens(
        self,
        forward_batch: ForwardBatch,
        output_ids: torch.Tensor,
        accept_lens: torch.Tensor,
    ):
        """
        Cumulate output tokens for penalty calculation in speculative decoding.

        In speculative decoding, the output_ids tensor contains accepted tokens followed by padding.
        We need to extract only the accepted tokens for each request and update the penalty state.

        Args:
            forward_batch: The forward batch containing request pool indices
            output_ids: Flattened tensor of output tokens [a, b, x, x, c, x, x, x]
                       where x represents padding/rejected tokens
            accept_lens: Number of accepted tokens per request [1, 0, ...]

        Example:
            If we have 2 requests with speculative_num_draft_tokens=4:
            - output_ids: [a, b, x, x, c, x, x, x] (8 elements)
            - accept_lens: [1, 0] (request 0 accepted 1 draft token 'a' and 'b' is bouns token, request 1 accepted 0 draft tokens)
            We should only update penalties for token 'a' and 'b' in request 0's penalty state.
        """
        batch_size = forward_batch.batch_size
        req_pool_indices = forward_batch.req_pool_indices
        tokens_per_request = self.server_args.speculative_num_draft_tokens
        scaling_penalties = forward_batch.sampling_info.repetition_penalties

        if scaling_penalties is None or scaling_penalties.shape[0] != batch_size:
            return

        if __is_npu__:
            cumulate_output_tokens_pytorch(
                output_ids,
                accept_lens,
                req_pool_indices,
                self.cumulated_scaling_penalty,
                scaling_penalties,
                tokens_per_request,
                valid_batch_size=getattr(forward_batch, "ori_bs", batch_size),
            )
        else:
            cumulate_output_tokens_kernel[(batch_size,)](
                output_ids,
                accept_lens,
                req_pool_indices,
                self.cumulated_scaling_penalty,
                scaling_penalties,
                tokens_per_request,
                self.vocab_size,
            )

    def renew_scaling_penalty(self, req_pool_indices):
        # When a new req arrives, renew it's penalty
        self.cumulated_scaling_penalty[req_pool_indices] = 1.0

    def _generate_penalties(self, forward_batch: ForwardBatch):
        scaling_penalties = self.cumulated_scaling_penalty[forward_batch.req_pool_indices]
        return scaling_penalties

    def _resolve_grammar(
        self,
        forward_batch: ForwardBatch,
        verify_spec_info: EagleVerifyInput,
    ) -> None:
        """Wait for the async grammar parse and attach the matcher to the verify
        input.

        This is the shared prerequisite for BOTH grammar paths: the synchronous
        path then builds the bitmask via ``_build_vocab_bitmask``; the overlap
        path lets the DFS host node build it. verify() applies the bitmask through
        ``verify_spec_info.grammar`` -- and apply_vocab_mask is a @staticmethod, so
        any non-None grammar of the right backend is a valid dispatch target.
        """
        grammars = forward_batch.sampling_info.grammars
        if not grammars:
            return
        verify_spec_info.grammar = next((g for g in grammars if g is not None), None)

    def _build_vocab_bitmask(
        self,
        forward_batch: ForwardBatch,
        verify_spec_info: EagleVerifyInput,
    ):
        """Build the grammar vocab bitmask for the SYNCHRONOUS verify path.

        Runs the DFS over the draft-token chain out-of-graph. Assumes
        ``_resolve_grammar`` already ran (grammar attached, async parse awaited).
        Returns the device bitmask, or None when the batch has no grammar
        constraints. The overlap path never calls this -- its DFS host node builds
        the mask instead.
        """
        grammars = forward_batch.sampling_info.grammars
        if not grammars:
            return None
        if __is_npu__:
            # NPU may have padded batch_size up to a graph bucket size, stashing
            # the real request count in ori_bs. Left as-is for the NPU merge.
            bs = getattr(forward_batch, "ori_bs", forward_batch.batch_size)
        else:
            # GPU never sets ori_bs, so batch_size is already the real count.
            bs = forward_batch.batch_size
        num_nodes = self.speculative_num_steps + 1
        # MTP produces a linear chain (not a tree): for each request the
        # draft tokens form a single path root -> t1 -> t2 -> ... -> tL-1.
        # Encode that chain for traverse_tree: next_token[i] = i+1 (last
        # position stays -1, no child), and there are never any siblings.
        # NOTE: the previous all--1 arrays made the DFS visit only the root
        # token, so draft tokens were never grammar-masked.
        retrive_next_sibling = torch.full(
            (bs, num_nodes),
            -1,
            device="cpu",
            dtype=torch.long,
        )
        # Build chain topology: node 0 -> 1 -> 2 -> ... -> N-1 -> -1
        # Each node's next_token points to the next node in the chain;
        # the last node has no child (-1).
        retrive_next_token = torch.full(
            (bs, num_nodes),
            -1,
            device="cpu",
            dtype=torch.long,
        )
        retrive_next_token[:, :-1] = torch.arange(1, num_nodes, dtype=torch.long)
        if __is_npu__:
            # Use the pre-fetched CPU tensor (started in preprocess_for_verify
            # for maximum D2H overlap); fall back to non-blocking copy otherwise.
            assert hasattr(verify_spec_info, "draft_token_cpu"), "draft_token_cpu should be pre-fetched on NPU"
            verify_spec_info.draft_token_cpu_copy_done.synchronize()
            draft_tokens_cpu = verify_spec_info.draft_token_cpu
        else:
            draft_tokens_cpu = verify_spec_info.draft_token.to("cpu")

        vocab_masks = generate_token_bitmask(
            grammars,
            verify_spec_info,
            retrive_next_token,
            retrive_next_sibling,
            draft_tokens_cpu,
            forward_batch.sampling_info.vocab_size,
            timeout_ms=getattr(self.server_args, "grammar_mask_timeout_ms", 0.0),
        ).pin_memory()

        if vocab_masks is not None:
            assert verify_spec_info.grammar is not None
            vocab_masks = vocab_masks.to(verify_spec_info.draft_token.device, non_blocking=True)
            # otherwise, this vocab mask will be the one from the previous extend stage
            # and will be applied to produce wrong results
            forward_batch.sampling_info.vocab_mask = None

        return vocab_masks

    def _generate_vocab_mask(
        self,
        forward_batch: ForwardBatch,
        verify_spec_info: EagleVerifyInput,
    ):
        """Resolve grammars and build the bitmask in one call.

        Used by the NPU rejection-sampling path, which builds the mask lazily
        (overlapping the CPU DFS with the NPU forward). The GPU path instead calls
        ``_resolve_grammar`` (in preprocess_for_verify) and ``_build_vocab_bitmask``
        (at the verify call site) separately, so the decision to skip the
        out-of-graph build is visible where it is made rather than hidden behind a
        flag.
        """
        self._resolve_grammar(forward_batch, verify_spec_info)
        return self._build_vocab_bitmask(forward_batch, verify_spec_info)

    @contextmanager
    def _stable_graph_logprob_flags(self, forward_batch):
        """Pin logprob-related Python attrs to their graph-warmup values around a
        compiled spec forward, then restore them.

        On NPU the target/draft forwards are torchair-compiled with
        ``dynamic=False``. dynamo installs guards on every Python attr read during
        tracing, and ``LogitsMetadata.from_forward_batch`` unconditionally reads
        ``top_logprobs_nums`` / ``token_ids_logprobs`` (plus ``return_logprob``).
        Warmup batches carry ``return_logprob=False`` and both lists ``None``; a
        real request with ``top_logprobs_nums=[10]`` misses those guards and
        triggers a runtime recompile on only the DP group that received it,
        desyncing the cross-DP HcomAllGather graph load into an EI0006 timeout.
        Forcing all three to the warmup values keeps the guards hit so no runtime
        recompile happens. Logprobs are computed post-hoc by compute_spec_logprobs
        from the restored forward_batch, so masking is result-neutral.

        GPU (npu=False) captures CUDA graphs with no value guards, so only
        ``return_logprob`` is masked there -- byte-for-byte the prior behavior."""
        npu = __is_npu__
        saved_return_logprob = forward_batch.return_logprob
        forward_batch.return_logprob = False
        if npu:
            saved_top_logprobs_nums = forward_batch.top_logprobs_nums
            saved_token_ids_logprobs = forward_batch.token_ids_logprobs
            forward_batch.top_logprobs_nums = None
            forward_batch.token_ids_logprobs = None
        try:
            yield
        finally:
            forward_batch.return_logprob = saved_return_logprob
            if npu:
                forward_batch.top_logprobs_nums = saved_top_logprobs_nums
                forward_batch.token_ids_logprobs = saved_token_ids_logprobs

    def start_grammar_update_thread(self):
        """
        Start the dedicated grammar-update thread and its queues (NPU only).

        The thread shares the device context with the forward thread. It waits
        for the verified-token D2H copy to complete and then runs
        update_grammar_after_verify off the forward thread's critical path.
        """
        self.grammar_update_queue = Queue()
        self.grammar_update_done_queue = Queue()
        self.grammar_update_inflight = 0
        self.grammar_update_thread = threading.Thread(
            target=self._grammar_update_thread_func,
            daemon=True,
            name="grammar-update-thread",
        )
        self.grammar_update_thread.start()

    def _grammar_update_thread_func(self):
        # Share the device context with the forward thread.
        torch.get_device_module(self.device).set_device(self.model_runner.gpu_id)
        while True:
            task = self.grammar_update_queue.get()
            if task is None:  # shutdown sentinel
                logger.error("grammar_update_queue get None")
                break
            (
                output_ids_copy_done,
                forward_batch,
                output_ids_cpu,
                accept_length_cpu,
                grammar_update_done_event,
            ) = task
            try:
                output_ids_copy_done.synchronize()

                grammar_update_done_event.grammar_terminated_list = self.update_grammar_after_verify(forward_batch, output_ids_cpu, accept_length_cpu)
            except Exception:
                grammar_update_done_event.grammar_terminated_list = None
                logger.exception("Async grammar update failed")
            grammar_update_done_event.set()
            self.grammar_update_done_queue.put(grammar_update_done_event)

    def submit_grammar_update(self, forward_batch, output_ids_copy_done, output_ids_cpu, accept_length_cpu):
        """
        Enqueue a grammar-update task for the grammar-update thread.

        Returns a threading.Event that is set once update_grammar_after_verify
        completes; the resulting grammar_terminated_list is attached to it as
        the `grammar_terminated_list` attribute.
        """
        grammar_update_done_event = threading.Event()
        grammar_update_done_event.grammar_terminated_list = None
        self.grammar_update_queue.put(
            (
                output_ids_copy_done,
                forward_batch,
                output_ids_cpu,
                accept_length_cpu,
                grammar_update_done_event,
            )
        )
        self.grammar_update_inflight += 1
        return grammar_update_done_event

    def wait_grammar_update_done(self):
        """
        Wait for the previous step's async grammar update, if there is one.

        Called in the forward thread right before vocab mask generation so the
        masks are built on up-to-date grammar state. Reads the completion event
        from the done queue: if it is there it is already set and we proceed,
        otherwise the get() blocks until the grammar-update thread finishes it.
        """
        if self.grammar_update_done_queue is None or self.grammar_update_inflight == 0:
            return
        self.grammar_update_inflight -= 1
        grammar_update_done_event = self.grammar_update_done_queue.get()
        # The event is set before being put into the done queue, so this
        # returns immediately; wait() is kept as a safety net.
        grammar_update_done_event.wait()

    def update_grammar_after_verify(self, forward_batch: ForwardBatch, output_ids_cpu, accept_lengths):
        # Grammar accept for multi-step decode: accept tokens in forward thread
        # so that the next step's vocab_mask is based on up-to-date grammar state.
        # Record per-TOKEN terminated state (list of lists) so post-process
        # knows exactly which token caused termination in spec decoding.
        grammar_terminated_list = None
        assert output_ids_cpu is not None
        assert accept_lengths is not None
        forward_mode = getattr(forward_batch, "ori_forward_mode", forward_batch.forward_mode)

        if not forward_mode.is_idle() and forward_batch.sampling_info and forward_batch.sampling_info.grammars:
            token_ids_cpu = output_ids_cpu.tolist()
            accept_lens_cpu = accept_lengths.tolist()
            grammar_terminated_list = []
            pt = 0
            for i, grammar in enumerate(forward_batch.sampling_info.grammars):
                n = accept_lens_cpu[i]
                req_terminated = []
                for j in range(n):
                    if grammar is not None and not grammar.finished and not grammar.is_terminated():
                        try:
                            grammar.accept_token(token_ids_cpu[pt + j])
                        except Exception as e:
                            # Keep accepting the rest of the batch: stopping here
                            # would leave every later request's grammar one step
                            # behind its output. Every rank sees the same tokens
                            # and grammar state, so they all degrade it together.
                            logger.error(f"Grammar accept failed after verify: req={i} token={token_ids_cpu[pt + j]}: {e}")
                            degrade_fn = getattr(grammar, "degrade", None)
                            if degrade_fn is None:
                                raise
                            degrade_fn(f"accept failed after verify: {e}")
                    req_terminated.append(grammar.is_terminated() if grammar is not None else False)
                grammar_terminated_list.append(req_terminated)
                pt += n
        return grammar_terminated_list

    def compute_spec_logprobs(
        self,
        forward_batch: ForwardBatch,
        logits_output: LogitsProcessorOutput,
        accept_length: torch.Tensor,
        output_ids: torch.Tensor,
    ):
        """Compute logprobs for speculative decoding accepted tokens."""
        if not forward_batch.return_logprob:
            return

        # NPU graph replay pads the batch to a fixed size; dummy entries produce
        # undefined accept_length values. Truncate to the original batch to exclude them.
        if __is_npu__:
            ori_bs = getattr(forward_batch, "ori_bs", forward_batch.batch_size)
            accept_length = accept_length[:ori_bs]

        accept_lengths_list = accept_length.tolist()
        total_accepted = sum(accept_lengths_list)
        if total_accepted == 0:
            return

        draft_token_num = self.server_args.speculative_num_draft_tokens
        device = logits_output.next_token_logits.device

        bs = len(accept_lengths_list)

        offsets = torch.arange(bs, device=device) * draft_token_num
        repeated_offsets = torch.repeat_interleave(offsets, accept_length)
        cumsum = torch.cumsum(accept_length, dim=0)
        within_block = torch.arange(total_accepted, device=device) - torch.repeat_interleave(cumsum - accept_length, accept_length)
        accepted_indices = repeated_offsets + within_block

        accepted_logits = logits_output.next_token_logits[accepted_indices]
        accepted_token_ids = output_ids[:total_accepted]

        temp_output = LogitsProcessorOutput(next_token_logits=accepted_logits)

        expanded_sampling_info = self._expand_sampling_info(forward_batch.sampling_info, accept_lengths_list)

        top_logprobs_nums_expanded = []
        token_ids_logprobs_expanded = []
        top_logprobs_nums = forward_batch.top_logprobs_nums or [0] * bs
        token_ids_logprobs = forward_batch.token_ids_logprobs or [None] * bs
        for num, tid, n_tok in zip(top_logprobs_nums, token_ids_logprobs, accept_lengths_list):
            top_logprobs_nums_expanded.extend([num] * n_tok)
            token_ids_logprobs_expanded.extend([tid] * n_tok)

        self.target_worker.model_runner.sampler(
            temp_output,
            expanded_sampling_info,
            True,
            top_logprobs_nums_expanded,
            token_ids_logprobs_expanded,
            batch_next_token_ids=accepted_token_ids,
        )

        logits_output.next_token_logprobs = temp_output.next_token_logprobs
        logits_output.next_token_top_logprobs_val = temp_output.next_token_top_logprobs_val
        logits_output.next_token_top_logprobs_idx = temp_output.next_token_top_logprobs_idx
        logits_output.next_token_token_ids_logprobs_val = temp_output.next_token_token_ids_logprobs_val
        logits_output.next_token_token_ids_logprobs_idx = temp_output.next_token_token_ids_logprobs_idx

    def _expand_sampling_info(self, sampling_info, accept_lengths_list):
        """Repeat-interleave sampling params to match [total_accepted] shape."""
        repeats = torch.tensor(
            accept_lengths_list,
            device=sampling_info.temperatures.device,
            dtype=torch.long,
        )
        return dataclasses.replace(
            sampling_info,
            temperatures=torch.repeat_interleave(sampling_info.temperatures, repeats, dim=0),
            top_ks=torch.repeat_interleave(sampling_info.top_ks, repeats, dim=0),
            top_ps=torch.repeat_interleave(sampling_info.top_ps, repeats, dim=0),
            min_ps=torch.repeat_interleave(sampling_info.min_ps, repeats, dim=0),
        )

    def forward_batch_speculative_generation(self, model_worker_batch: ModelWorkerBatch, launch_done: threading.Event = None):
        """
        Main entry point for speculative generation on a batch of requests.

        This method orchestrates the speculative decoding workflow by routing to the appropriate
        forward method based on the batch's forward mode (verify, extend, or idle). It handles
        both CUDA graph execution and regular forward passes.

        Args:
            model_worker_batch: The batch of requests to process.
            launch_done: Optional threading event, used in overlap-schedule.

        Returns:
            The output from the corresponding forward method (verify, prefill, or idle).

        Note:
            - Uses CUDA graphs when available and applicable for optimal performance
            - Handles special state updates for hybrid linear attention backends
            - Clears padding positions in the token pool after CUDA graph execution
        """
        self.set_hicache_consumer(model_worker_batch.hicache_consumer_index)
        forward_batch = ForwardBatch.init_new(model_worker_batch, self.target_worker.model_runner)
        # Reset the per-batch grammar signal on the worker. Whichever verify path
        # runs the in-stream accept (graph replay() or the eager verify branch
        # below) sets last_grammar_slot to a real slot again; if this batch falls
        # back to a path that does NOT run the in-stream accept, the slot stays -1
        # so the forward thread does not reuse a stale terminated snapshot. Reset
        # unconditionally -- the field exists even when cuda_graph_runner is None
        # (disable_cuda_graph).
        self.last_grammar_slot = -1
        vocab_masks = None
        if (not __is_npu__) and self.cuda_graph_runner and self.cuda_graph_runner.can_run(forward_batch):
            vocab_masks = None
            if forward_batch.forward_mode.is_target_verify():
                self.preprocess_for_verify(forward_batch)
                # When grammar runs inside the graph (overlap path) the DFS that
                # builds the vocab mask is a captured host node, so the
                # out-of-graph build is skipped here. Otherwise build it now for
                # the synchronous verify.
                if not (self.grammar_overlap_enabled and forward_batch.sampling_info.grammars):
                    vocab_masks = self._build_vocab_bitmask(forward_batch, forward_batch.spec_info)
            self.target_worker.model_runner.forward_pass_id += 1
            with get_global_expert_distribution_recorder().with_forward_pass(
                self.target_worker.model_runner.forward_pass_id,
                forward_batch,
            ):
                out = self.cuda_graph_runner.replay(forward_batch, vocab_masks)
            if self.target_worker.model_runner.eplb_manager is not None:
                self.target_worker.model_runner.eplb_manager.on_forward_pass_end()
            # set padding position to zero in case of illegal memory
            # when cuda graph padding happens
            self.req_to_token_pool.verified_lens[0].zero_()
            target_attn_backend = self.target_worker.model_runner.attn_backend
            if isinstance(target_attn_backend, HybridLinearAttnBackend):
                accept_length = out[2]
                target_attn_backend.update_mamba_state_after_mtp_verify(accept_length, None)
            # CUDA graph was captured with return_logprob=False, so compute logprobs
            # post-replay. accept_length from graph output is already +1'd (total output
            # tokens) because the graph captures create_extend_spec_info's in-place +1.
            if forward_batch.return_logprob:
                logits_output, output_ids, accept_length, _, _, _ = out
                self.compute_spec_logprobs(
                    forward_batch,
                    logits_output,
                    accept_length,
                    output_ids,
                )
        elif bool(
            __is_npu__
            and self.npu_graph_runner_for_draft_extend
            and (forward_batch.all_decode_or_idle or global_server_args_dict["npu_disable_all_gather"])
        ):
            # 图模式和单算子模式无法进行集合通信
            # prefill + idle 场景：大小模型都走eager
            # decode + idle 场景：大小模型都走图
            self.req_to_token_pool.verified_lens[0] = 0
            if forward_batch.forward_mode.is_target_verify():
                vocab_masks = self.preprocess_for_verify(forward_batch)
            if global_server_args_dict["npu_disable_all_gather"]:
                padding_size = max(self.npu_graph_runner_for_draft_extend.compile_bs)
            else:
                max_input_bs = max(forward_batch.global_num_tokens) / self.model_runner.server_args.speculative_num_draft_tokens
                padding_size = self.multi_batch_select(max_input_bs, self.npu_graph_runner_for_draft_extend.compile_bs)
            forward_batch.padding_for_graph_mtp(self.draft_model_runner, padding_size)
            forward_batch.can_run_all2all = ENV.npu_enable_all2all_comm
            self.init_attn_backends(forward_batch)
            out = self.forward_decode_spec(forward_batch, vocab_masks)

            bs = getattr(forward_batch, "ori_bs", forward_batch.batch_size)
            (
                logits_output,
                next_token_ids,
                accept_length,
                new_verified_id,
                token_list,
                grammar_terminated_list,
            ) = out

            def unpad(ori, bs):
                if isinstance(ori, torch.Tensor):
                    return ori[:bs]
                if isinstance(ori, list):
                    tmp = []
                    for i in ori:
                        tmp.append(i[:bs])
                    return tmp

            mtpn_factor = self.target_worker.model_runner.server_args.speculative_num_steps + 1
            logits_output.next_token_logits = unpad(logits_output.next_token_logits, bs * mtpn_factor)
            logits_output.hidden_states = unpad(logits_output.hidden_states, bs * mtpn_factor)

            out = (
                logits_output,
                unpad(next_token_ids, bs * mtpn_factor),
                unpad(accept_length, bs),
                unpad(new_verified_id, bs),
                unpad(token_list, bs),
                grammar_terminated_list,
            )
        elif forward_batch.forward_mode.is_target_verify():
            grammars = forward_batch.sampling_info.grammars if forward_batch.sampling_info is not None else None
            # Both paths run the SAME verify -- forward_decode_spec. They differ
            # only in pre/post processing: with grammar overlap the DFS/accept run
            # async on grammar_stream, injected via hooks and joined by finalize();
            # otherwise the vocab mask is built synchronously and no hooks are used.
            if self.grammar_overlap_enabled and grammars:
                vocab_masks, pre_hook, post_hook, finalize = self._prepare_eager_grammar_verify(forward_batch, grammars)
                out = self.forward_decode_spec(
                    forward_batch,
                    vocab_masks,
                    pre_propose_hook=pre_hook,
                    post_target_verify_hook=post_hook,
                )
                finalize()
            else:
                self.preprocess_for_verify(forward_batch)
                vocab_masks = self._build_vocab_bitmask(forward_batch, forward_batch.spec_info)
                self.init_attn_backends(forward_batch)
                out = self.forward_decode_spec(forward_batch, vocab_masks)

            target_attn_backend = self.target_worker.model_runner.attn_backend
            if isinstance(target_attn_backend, HybridLinearAttnBackend):
                accept_length = out[2]
                target_attn_backend.update_mamba_state_after_mtp_verify(accept_length, None)
        elif forward_batch.forward_mode.is_extend():
            self.init_attn_backends(forward_batch)
            forward_batch.capture_hidden_mode = CaptureHiddenMode.FULL
            out = self.forward_prefill_spec(model_worker_batch, forward_batch)
        elif forward_batch.forward_mode.is_idle():
            out = self.forward_idle(forward_batch)
        if launch_done:
            launch_done.set()
        return out

    def _prepare_eager_grammar_verify(self, forward_batch: ForwardBatch, grammars):
        """Set up async grammar for an eager (non-graph) target-verify, mirroring
        the in-graph fork/join (SpecDecodeCudaGraphRunner.capture_one_batch_size).

        This does NOT run the verify itself -- the caller runs the shared
        forward_decode_spec. It returns everything needed to weave the grammar
        overlap into that verify:
            (vocab_masks, pre_propose_hook, post_target_verify_hook, finalize)
        The DFS (vocab-mask build) and the permanent accept run as host nodes on
        the shared ``grammar_stream`` -- DFS forked here before the target forward
        so it overlaps it (DFS||target_forward); accept forked in pre_propose_hook
        before the draft steps so it overlaps them (accept||draft). ``finalize``
        joins grammar_stream back after the verify and signals the consumer.

        Unlike the graph path the stream ops are NOT captured, so the ring ``slot``
        is bound to the eager wrappers via closure (no slot_scalar / slot_host
        device read-back). Only the forward thread runs this, and every grammar
        host node is issued on the single ``grammar_stream`` -- so graph and eager
        accepts are FIFO-serialized on the matcher and never interleave.
        """
        ctx = self.grammar_ctx
        raw_bs = forward_batch.batch_size
        capture_n = raw_bs * ctx.draft_token_num
        slot = self.alloc_slot()
        ctx.set_step_inputs(grammars, raw_bs, capture_n, slot)

        # Resolve the grammars (waits on sampling_info_done) and set up verify
        # spec info WITHOUT building the vocab mask -- the DFS host node builds it.
        self.preprocess_for_verify(forward_batch)
        self.init_attn_backends(forward_batch)

        main_stream = torch.cuda.current_stream()
        grammar_stream = self.grammar_stream
        # draft_token now lives on the verify spec info (preprocess reassigned it).
        draft_token = forward_batch.spec_info.draft_token
        vocab_masks = ctx.vocab_masks_device[:capture_n]

        # DFS host node: fork before the target forward. Both DFS and the target
        # forward only READ draft_token, so the cross-stream read is safe; the
        # mask is consumed by rejection_sampling (joined via _join_dfs below).
        grammar_stream.wait_stream(main_stream)
        with torch.cuda.stream(grammar_stream):
            ctx.draft_token_host[:capture_n].copy_(draft_token[:capture_n], non_blocking=True)
            launch_hostfunc(dfs_hostfunc_eager, ctx, slot)
            vocab_masks.copy_(ctx.mask_host[:capture_n], non_blocking=True)

        def _join_dfs():
            main_stream.wait_stream(grammar_stream)

        def _fork_accept(output_ids, accept_length):
            grammar_stream.wait_stream(main_stream)
            with torch.cuda.stream(grammar_stream):
                ctx.accepted_host[:capture_n].copy_(output_ids, non_blocking=True)
                ctx.accept_length_host[:raw_bs].copy_(accept_length, non_blocking=True)
                launch_hostfunc(accept_hostfunc_eager, ctx, slot)

        def _finalize():
            # Join the grammar stream so the accept host node (and its writes to
            # terminated_ring / error_ring) complete before copy_done is recorded
            # on the forward thread -- so they are visible once the consumer syncs
            # it. Then signal the consumer to collect this slot's ring snapshot
            # (shared with the graph path).
            main_stream.wait_stream(grammar_stream)
            self.last_grammar_slot = slot

        return vocab_masks, _fork_accept, _join_dfs, _finalize

    def init_attn_backends(self, forward_batch: ForwardBatch):
        """
        Initialize attention backend metadata for both target and draft models.

        This method prepares the attention backends by computing and caching forward metadata
        such as sequence lengths, attention masks, and KV cache pointers.

        Args:
            forward_batch: The forward batch containing sequence and attention information.

        Note:
            - Always initializes the target model's attention backend
            - Also initializes the draft model's backend if using a model-based algorithm
        """
        self.target_worker.model_runner.attn_backend.init_forward_metadata(forward_batch)
        # This is a model-based algo and the model has attention module
        if hasattr(self, "model_runner_list") and hasattr(self.model_runner_list[0], "attn_backend"):
            if len(self.model_runner_list):
                runner = self.model_runner_list[0]
                runner.attn_backend.init_forward_metadata(forward_batch)
            for runner in self.model_runner_list[1:]:
                runner.attn_backend = self.model_runner_list[0].attn_backend

    @abstractmethod
    def propose(self):
        """
        Generate draft tokens using the speculative model.

        This abstract method must be implemented by subclasses to define how draft tokens
        are proposed. The specific implementation depends on the speculative algorithm
        (e.g., EAGLE uses a draft model, PLD uses prompt lookup).

        Returns:
            Draft tokens and associated metadata for verification.
        """
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def forward_prefill_spec(self):
        """
        Handle the prefill (extend) stage for speculative decoding.

        This abstract method processes new prompt tokens and prepares the KV cache for
        subsequent speculative decoding steps. It typically involves running both the
        target and draft models on the input prompt.

        Args:
            Implementation-specific arguments (defined in subclasses).

        Returns:
            Output from the prefill stage, including hidden states and logits.
        """
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def forward_decode_spec(self):
        """
        Handle the decode stage with token verification for speculative decoding.

        This abstract method performs the core speculative decoding loop: the draft model
        proposes tokens, and the target model verifies them through rejection sampling.

        Args:
            Implementation-specific arguments (defined in subclasses).

        Returns:
            Verified tokens, acceptance statistics, and updated hidden states.
        """
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def forward_idle(self):
        """
        Handle idle forward passes when no active generation is occurring.

        This abstract method is called when the batch is in idle mode, typically for
        maintaining state or performing background operations.

        Args:
            Implementation-specific arguments (defined in subclasses).

        Returns:
            Idle forward output (implementation-specific).
        """
        raise NotImplementedError("Not implemented")
