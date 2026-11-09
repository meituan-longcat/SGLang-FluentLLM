"""Scheduler multimodal mixin.

Encapsulates all multimodal-related logic for the Scheduler so that the main
Scheduler class stays focused on scheduling concerns.

Usage:
    class Scheduler(SchedulerMultimodalMixin, ...):
        def __init__(self, ...):
            ...
            self.init_multimodal_tokenizer(server_args)
            ...
            self.init_multimodal_runtime(server_args)
"""

from typing import List

from sglang.srt.env import envs
from sglang.srt.server_args import ServerArgs
from sglang.srt.multimodal.mm_feature_transport import get_mm_transport_strategy
from sglang.srt.multimodal.mm_utils import init_mm_embedding_cache
from sglang.srt.utils.hf_transformers_utils import get_processor, get_tokenizer_from_processor


class SchedulerMultimodalMixin:
    """Mixin that encapsulates all multimodal-related logic for the Scheduler.

    The main Scheduler class must select the final multimodal tokenizer before
    initializing reasoning and grammar state, then initialize runtime resources
    near the end of __init__ after workers are ready.
    """

    def init_multimodal_tokenizer(self, server_args: ServerArgs) -> None:
        """Select the final tokenizer before reasoning and grammar initialization."""
        self.is_multimodal = server_args.enable_multimodal and self.model_config.is_multimodal
        if not self.is_multimodal or self.tokenizer is None:
            return

        self.tokenizer = get_tokenizer_from_processor(
            get_processor(
                server_args.tokenizer_path,
                tokenizer_mode=server_args.tokenizer_mode,
                trust_remote_code=server_args.trust_remote_code,
                revision=server_args.revision,
            )
        )

    def init_multimodal_runtime(self, server_args: ServerArgs) -> None:
        """Initialize multimodal caches and transport after workers are ready."""
        if not self.is_multimodal:
            return

        if server_args.disaggregation_mode != "decode":
            embedding_cache_size = envs.SGLANG_VLM_CACHE_SIZE_MB.get()
            init_mm_embedding_cache(embedding_cache_size * 1024 * 1024)

        self._mm_transport_strategy = get_mm_transport_strategy(server_args)

    def _unwrap_mm_features(self, recv_reqs: List) -> None:
        """Unwrap shm/cuda_ipc-wrapped multimodal features after broadcast_pyobj.

        Called at the end of recv_requests(), after all TP ranks have received
        the broadcast data. Delegates to the transport strategy.
        """
        if not recv_reqs or not self.is_multimodal:
            return
        self._mm_transport_strategy.unwrap(recv_reqs, self)
