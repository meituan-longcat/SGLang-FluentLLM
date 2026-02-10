from __future__ import annotations

from abc import abstractmethod, ABC
import os
import time
from typing import Optional, TYPE_CHECKING
import threading

import torch
from huggingface_hub import snapshot_download


from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
    HybridLinearAttnBackend,
)
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode
)
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.eagle_utils import (
    EagleDraftInput,
    EagleDraftOutput,
    EagleVerifyInput,
    generate_token_bitmask,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.utils import get_available_gpu_memory, get_colorful_logger

from sglang.srt.configs.model_config import AttentionArch

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.layers.logits_processor import LogitsProcessorOutput
    from sglang.srt.managers.schedule_batch import ModelWorkerBatch
    from sglang.srt.managers.tp_worker import TpModelWorker
    from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
    from sglang.srt.mem_cache.allocator import KVAllocator

logger = get_colorful_logger(__name__)


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
        self.speculative_algorithm = SpeculativeAlgorithm.from_string(
            server_args.speculative_algorithm
        )
        self.topk = server_args.speculative_eagle_topk
        assert self.topk == 1, "Tree Attention is abandoned for now."
        self.speculative_num_steps = server_args.speculative_num_steps
        self.use_over_embedding = (
            drafter_use_oe or self.target_worker.use_over_embedding
        )
        # Share req_to_token_pool and kv_allocator with target worker
        self.req_to_token_pool: ReqToTokenPool = (
            self.target_worker.model_runner.req_to_token_pool
        )
        self.kv_allocator: KVAllocator = self.target_worker.model_runner.kv_allocator
        self.oe_token_table = self.target_worker.model_runner.oe_token_table

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
        before_capture_available_gpu_memory = get_available_gpu_memory(
            self.device, self.target_worker.model_runner.gpu_id
        )
        logger.info(
            "Capture cuda graph begin. This can take up to several minutes. "
            f"avail mem={before_capture_available_gpu_memory:.2f} GB in model runner!"
        )
        self.cuda_graph_runner = graph_runner_cls(self)
        after_capture_available_gpu_memory = get_available_gpu_memory(
            self.device, self.target_worker.model_runner.gpu_id
        )
        logger.info(
            f"Capture cuda graph end. Time elapsed: {time.time() - tic:.2f} s. "
            f"avail mem={after_capture_available_gpu_memory:.2f} GB"
        )
        logger.info(
            f"{len(self.cuda_graph_runner.graphs)} graphs used "
            f"mem={(before_capture_available_gpu_memory - after_capture_available_gpu_memory):.2f} GB"
        )

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
                file_path = os.path.join(
                    cache_dir, os.path.basename(self.server_args.speculative_token_map)
                )
                self.hot_token_id = torch.load(file_path)
            self.server_args.json_model_override_args = (
                f'{{"hot_vocab_size": {len(self.hot_token_id)}}}'
            )

        embed, head = self.target_worker.model_runner.model.get_embed_and_head()
        if self.speculative_algorithm.is_eagle3():
            if self.server_args.speculative_token_map is not None:
                logger.warning(
                    "Speculative token map specified, but EAGLE3 models already have this. Ignoring the specified token map."
                )
            self.hot_token_id = None
        elif self.server_args.speculative_token_map is not None:
            head = head.clone()
            self.hot_token_id = torch.tensor(
                self.hot_token_id, dtype=torch.int32, device=head.device
            )
            head.data = head.data[self.hot_token_id]
        else:
            self.hot_token_id = None

        self.use_over_embedding = (
            self.use_over_embedding or self.target_worker.use_over_embedding
        )

        if self.speculative_algorithm.is_eagle3():
            if self.target_worker.use_over_embedding:
                word_embed = embed.word_embeder.weight
            else:
                word_embed = embed
            drafter_model_runner.model.set_embed(word_embed)
            self.hot_token_id = drafter_model_runner.model.get_hot_token_id().to(
                word_embed.device
            )
        else:
            if self.use_over_embedding:
                drafter_model_runner.model.set_oe_and_head(embed, head)
            else:
                drafter_model_runner.model.set_embed_and_head(embed, head)

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

            self.draft_attn_backend = FlashMLAMultiStepDecodeBackend(
                draft_model_runner, self.speculative_num_steps
            )
        elif drafter_backend == "flashinfer_mla":
            from sglang.srt.layers.attention.flashinfer_mla_backend import (
                FlashInferMLAMultiStepDraftBackend,
            )

            self.draft_attn_backend = FlashInferMLAMultiStepDraftBackend(
                draft_model_runner,
                self.topk,
                self.speculative_num_steps,
            )
        elif drafter_backend == "duo_attn":
            if draft_model_runner.model_config.attention_arch != AttentionArch.MLA:
                from sglang.srt.layers.attention.flashinfer_backend import (
                    FlashInferMultiStepDraftBackend,
                )

                self.draft_attn_backend = FlashInferMultiStepDraftBackend(
                    draft_model_runner,
                    self.topk,
                    self.speculative_num_steps,
                )
            else:
                from sglang.srt.layers.attention.duo_attn_backend import (
                    DuoAttnMultiStepBackend,
                )

                self.draft_attn_backend = DuoAttnMultiStepBackend(
                    draft_model_runner,
                    self.topk,
                    self.speculative_num_steps,
                )
        elif drafter_backend == "hybrid_linear_attn":
            from sglang.srt.layers.attention.flashinfer_backend import (
                FlashInferMultiStepDraftBackend,
            )

            self.draft_attn_backend = FlashInferMultiStepDraftBackend(
                draft_model_runner, self.topk, self.speculative_num_steps
            )
        elif drafter_backend == "dsa":
            from sglang.srt.layers.attention.dsa_backend import (
                DpskSparseAttnMultiStepBackend,
            )

            self.draft_attn_backend = DpskSparseAttnMultiStepBackend(
                draft_model_runner, self.topk, self.speculative_num_steps
            )
        else:
            raise ValueError(
                f"EAGLE is not supported with drafter attention backend {drafter_backend}"
            )

    def rejection_sampling(
        self,
        forward_batch: ForwardBatch,
        logits_output: LogitsProcessorOutput,
        vocab_masks: Optional[torch.Tensor] = None,
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
        predict, logits_output, accept_length, accept_index = (
            forward_batch.spec_info.verify(forward_batch, logits_output, vocab_masks)
        )
        return predict, logits_output, accept_length, accept_index

    def preprocess_for_draft_after_decode(
        self,
        forward_batch: ForwardBatch,
        accept_length: torch.Tensor,
        accept_index: torch.Tensor,
        target_predict: torch.Tensor,
        with_draft_model: bool = True
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
            forward_batch.token_to_kv_pool = (
                self.target_worker.model_runner.token_to_kv_pool
            )
        draft_input = EagleDraftInput()
        # rearranged hidden states
        draft_input.hidden_states = forward_batch.spec_info.hidden_states
        draft_input.accept_length = accept_length
        draft_input.verified_id = target_predict
        draft_input.draft_token_num = self.server_args.speculative_num_draft_tokens
        draft_input.accept_index = accept_index
        new_verified_id = draft_input.prepare_extend_after_decode(
            forward_batch, self.use_over_embedding
        )
        forward_batch.spec_info = draft_input
        return new_verified_id

    def preprocess_for_verify(
        self, forward_batch: ForwardBatch
    ) -> Optional[torch.tensor]:
        """
        Preprocess the forward batch to prepare for token verification.

        This method transforms the draft output into a format suitable for verification by
        the target model. It sets up sequence lengths, positions, and input tokens, and
        generates vocabulary masks if grammar constraints are present.

        Args:
            forward_batch: The forward batch containing draft output from the proposal stage.

        Returns:
            Optional vocabulary masks for constrained generation, or None if no constraints.

        Note:
            - Converts EagleDraftOutput to EagleVerifyInput for the verification stage
            - Handles grammar-based constrained generation by creating token bitmasks
            - Updates batch metadata (seq_lens, positions, input_ids) for verification
        """
        assert isinstance(forward_batch.spec_info, EagleDraftOutput)
        forward_batch.seq_lens = self.req_to_token_pool.verified_lens[
            forward_batch.req_pool_indices
        ]
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

        vocab_masks = self._generate_vocab_mask(forward_batch, verify_spec_info)
        forward_batch.spec_info = verify_spec_info
        forward_batch.input_ids = forward_batch.spec_info.draft_token
        forward_batch.capture_hidden_mode = CaptureHiddenMode.FULL
        forward_batch.positions = forward_batch.spec_info.positions

        return vocab_masks

    def _generate_vocab_mask(
        self, forward_batch: ForwardBatch, verify_spec_info: EagleVerifyInput
    ):
        """
        Generate vocabulary masks for grammar-constrained generation.

        Creates bitmasks that restrict which tokens are valid at each position according to
        the grammar constraints specified in the sampling info.

        Args:
            forward_batch: The forward batch containing sampling info with grammar constraints.
            verify_spec_info: The verification input containing draft tokens and positions.

        Returns:
            Token bitmasks as a tensor, or None if no grammar constraints are present.

        Note:
            - Only generates masks when grammars are specified in sampling_info
            - Waits for sampling info to be ready if processing is still in progress
            - Clears the forward_batch vocab_mask to prevent stale masks from being applied
        """
        vocab_masks = None
        if forward_batch.sampling_info.grammars:
            grammars = forward_batch.sampling_info.grammars
            if forward_batch.sampling_info.sampling_info_done:
                forward_batch.sampling_info.sampling_info_done.wait()
            retrive_next_sibling = torch.full(
                (forward_batch.batch_size, self.speculative_num_steps + 1),
                -1,
                device="cpu",
                dtype=torch.long,
            )
            retrive_next_token = torch.full(
                (forward_batch.batch_size, self.speculative_num_steps + 1),
                -1,
                device="cpu",
                dtype=torch.long,
            )
            vocab_masks = generate_token_bitmask(
                grammars,
                verify_spec_info,
                retrive_next_token,
                retrive_next_sibling,
                verify_spec_info.draft_token.cpu(),
                forward_batch.sampling_info.vocab_size,
            )

            if vocab_masks is not None:
                assert verify_spec_info.grammar is not None
                vocab_masks = vocab_masks.to(verify_spec_info.draft_token.device)
                # otherwise, this vocab mask will be the one from the previous extend stage
                # and will be applied to produce wrong results
                forward_batch.sampling_info.vocab_mask = None

        return vocab_masks

    def forward_batch_speculative_generation(
        self, model_worker_batch: ModelWorkerBatch, launch_done: threading.Event = None
    ):
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
        forward_batch = ForwardBatch.init_new(
            model_worker_batch, self.target_worker.model_runner
        )
        vocab_masks = None
        if self.cuda_graph_runner and self.cuda_graph_runner.can_run(forward_batch):
            vocab_masks = None
            if forward_batch.forward_mode.is_target_verify():
                vocab_masks = self.preprocess_for_verify(forward_batch)
            out = self.cuda_graph_runner.replay(forward_batch, vocab_masks)
            # set padding position to zero in case of illegal memory
            # when cuda graph padding happens
            self.req_to_token_pool.verified_lens[0].zero_()
            if isinstance(forward_batch.attn_backend, HybridLinearAttnBackend):
                accept_length = out[2]
                forward_batch.attn_backend.update_mamba_state_after_mtp_verify(
                    accept_length, None
                )
        elif forward_batch.forward_mode.is_target_verify():
            vocab_masks = self.preprocess_for_verify(forward_batch)
            self.init_attn_backends(forward_batch)
            out = self.forward_decode_spec(forward_batch, vocab_masks)
            if isinstance(forward_batch.attn_backend, HybridLinearAttnBackend):
                accept_length = out[2]
                forward_batch.attn_backend.update_mamba_state_after_mtp_verify(
                    accept_length, None
                )
        elif forward_batch.forward_mode.is_extend():
            self.init_attn_backends(forward_batch)
            forward_batch.capture_hidden_mode = CaptureHiddenMode.FULL
            out = self.forward_prefill_spec(model_worker_batch, forward_batch)
        elif forward_batch.forward_mode.is_idle():
            out = self.forward_idle(forward_batch)
        if launch_done:
            launch_done.set()
        return out

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
        self.target_worker.model_runner.attn_backend.init_forward_metadata(
            forward_batch
        )
        # This is a model-based algo and the model has attention module
        if hasattr(self, "model_runner") and hasattr(self.model_runner, "attn_backend"):
            self.model_runner.attn_backend.init_forward_metadata(forward_batch)

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
