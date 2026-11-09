from __future__ import annotations

import copy
import dataclasses
import time
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sglang.srt.mem_cache.hicache_operation import S2HLoad

import numpy as np
import torch

from sglang.srt.constrained.base_grammar_backend import BaseGrammarObject
from sglang.srt.disaggregation.base import BaseKVSender
from sglang.srt.env import global_server_args_dict
from sglang.srt.managers.scheduling.multimodal_data import MultimodalInputs
from sglang.srt.mem_cache.memory_pool import BaseTokenToKVPool, ReqToTokenPool, ReqToTokenPoolInfo
from sglang.srt.mem_cache.radix_cache import RadixCache
from sglang.srt.observability.metrics.collector import SchedulerMetricsCollector, TimeStats
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.utils import get_colorful_logger

logger = get_colorful_logger(__name__)

INIT_INCREMENTAL_DETOKENIZATION_OFFSET = 5

# How far back to look in the prompt for a think start token left there by the chat template.
REASONING_PROMPT_SCAN_TOKENS = 8


class BaseFinishReason:
    def __init__(self, is_error: bool = False):
        self.is_error = is_error

    def to_json(self):
        raise NotImplementedError()


class FinishMatchedToken(BaseFinishReason):
    def __init__(self, matched: int | list[int]):
        super().__init__()
        self.matched = matched

    def to_json(self):
        return {
            "type": "stop",  # to match OpenAI API's return value
            "matched": self.matched,
        }


class FinishMatchedStr(BaseFinishReason):
    def __init__(self, matched: str):
        super().__init__()
        self.matched = matched

    def to_json(self):
        return {
            "type": "stop",  # to match OpenAI API's return value
            "matched": self.matched,
        }


class FinishLength(BaseFinishReason):
    def __init__(self, length: int):
        super().__init__()
        self.length = length

    def to_json(self):
        return {
            "type": "length",  # to match OpenAI API's return value
            "length": self.length,
        }


class AbortCode(Enum):
    TransferFailed = 521
    UnknownError = 522


class FinishAbort(BaseFinishReason):
    def __init__(self, message="Unknown error", status_code=None, err_type=AbortCode.UnknownError):
        super().__init__(is_error=True)
        self.message = message
        self.status_code = status_code
        self.err_type = err_type

    def to_json(self):
        return {
            "type": "abort",
            "message": self.message,
            "status_code": self.status_code,
            "err_type": self.err_type,
        }


@dataclasses.dataclass
class ImageInputs:
    """The image related inputs."""

    pixel_values: torch.Tensor | np.array
    image_hashes: list | None = None
    image_sizes: list | None = None
    image_offsets: list | None = None
    image_pad_len: list | None = None
    pad_values: list | None = None
    modalities: list | None = None
    num_image_tokens: int | None = None

    # Llava related
    aspect_ratio_ids: list[torch.Tensor] | None = None
    aspect_ratio_mask: list[torch.Tensor] | None = None

    # QWen2-VL related
    image_grid_thws: list[tuple[int, int, int]] = None
    mrope_position_delta: torch.Tensor | None = None

    # MiniCPMV related
    # All the images in the batch should share the same special image
    # bound token ids.
    im_start_id: torch.Tensor | None = None
    im_end_id: torch.Tensor | None = None
    slice_start_id: torch.Tensor | None = None
    slice_end_id: torch.Tensor | None = None
    tgt_sizes: list | None = None

    @staticmethod
    def from_dict(obj: dict):
        ret = ImageInputs(
            pixel_values=obj["pixel_values"],
            image_hashes=obj["image_hashes"],
        )

        # Use image hash as fake token_ids. We use this as the key for prefix matching in the radix cache.
        # Please note that if the `input_ids` is later used in the model forward,
        # you also need to clamp the values within the range of [0, vocab_size) to avoid out-of-bound
        # errors in cuda kernels. See also llava.py for example.
        ret.pad_values = [x % (1 << 30) for x in ret.image_hashes]

        optional_args = [
            "image_sizes",
            "modalities",
            "aspect_ratio_ids",
            "aspect_ratio_mask",
            "image_grid_thws",
            "im_start_id",
            "im_end_id",
            "slice_start_id",
            "slice_end_id",
            "tgt_sizes",
        ]
        for arg in optional_args:
            if arg in obj:
                setattr(ret, arg, obj[arg])

        return ret

    def merge(self, other):
        assert self.pixel_values.shape[1:] == other.pixel_values.shape[1:]
        self.pixel_values = np.concatenate([self.pixel_values, other.pixel_values])

        # Use image hash as fake token_ids. We use this as the key for prefix matching in the radix cache.
        # Please note that if the `input_ids` is later used in the model forward,
        # you also need to clamp the values within the range of [0, vocab_size) to avoid out-of-bound
        # errors in cuda kernels. See also llava.py for example.
        self.image_hashes += other.image_hashes
        self.pad_values = [x % (1 << 30) for x in self.image_hashes]

        optional_args = [
            "image_sizes",
            "image_offsets",
            "image_pad_len",
            # "modalities", # modalities should be ["multi-images"] (one entry) even for multiple images
            "aspect_ratio_ids",
            "aspect_ratio_mask",
            "image_grid_thws",
        ]
        for arg in optional_args:
            if getattr(self, arg, None) is not None:
                setattr(self, arg, getattr(self, arg) + getattr(other, arg))


class RequestStage(str, Enum):
    # prefill
    PREFILL_WAITING = "prefill_waiting"

    # disaggregation prefill
    PREFILL_PREPARE = "prefill_prepare"
    PREFILL_BOOTSTRAP = "prefill_bootstrap"
    PREFILL_FORWARD = "prefill_forward"
    PREFILL_TRANSFER_KV_CACHE = "prefill_transfer_kv_cache"

    # disaggregation decode
    DECODE_PREPARE = "decode_prepare"
    DECODE_BOOTSTRAP = "decode_bootstrap"
    DECODE_WAITING = "decode_waiting"
    DECODE_TRANSFERRED = "decode_transferred"


class Req:
    """The input and output status of a request."""

    def __init__(
        self,
        rid: str,
        origin_input_text: str,
        origin_input_ids: tuple[int],
        sampling_params: SamplingParams,
        return_logprob: bool = False,
        top_logprobs_num: int = 0,
        token_ids_logprob: list[int] = None,
        stream: bool = False,
        origin_input_ids_unpadded: tuple[int] | None = None,
        input_embeds: list[list[float]] | None = None,
        input_extra_infos: list[dict] | None = None,
        session_id: str | None = None,
        custom_logit_processor: str | None = None,
        return_hidden_states: bool = False,
        eos_token_ids: set[int] | None = None,
        bootstrap_host: str | None = None,
        bootstrap_port: int | None = None,
        bootstrap_room: int | None = None,
        origin_input_multi_ids: list[list[int]] | None = None,
        metrics_collector: SchedulerMetricsCollector | None = None,
        created_time: float | None = None,
        mm_inputs: MultimodalInputs | None = None,
    ):
        # Input and output info
        self.rid = rid
        self.origin_input_text = origin_input_text
        self.origin_input_ids_unpadded = (
            origin_input_ids_unpadded if origin_input_ids_unpadded else origin_input_ids  # Before image padding
        )
        self.origin_input_ids = origin_input_ids
        self.origin_input_multi_ids = origin_input_multi_ids
        # Each decode stage's output ids
        self.output_ids = []
        self.output_multi_ids = []
        # fill_ids = origin_input_ids + output_ids. Updated if chunked.
        self.fill_ids = None
        self.fill_multi_ids = None
        self.fill_input_embeds = None
        # For Eagle and chunked prefill, remove first token when chunked prefill
        self.draft_fill_ids = None
        self.session_id = session_id
        self.input_embeds = input_embeds
        self.input_extra_infos = input_extra_infos

        # Contains mm_items with pixel_values, hashes, pad_values, offsets, etc.
        # None for text-only requests.
        self.mm_inputs: MultimodalInputs | None = mm_inputs

        # for oe init
        self.oe_init = None

        # Sampling info
        if isinstance(sampling_params.custom_params, dict):
            sampling_params = copy.copy(sampling_params)
            sampling_params.custom_params = sampling_params.custom_params | {"__req__": self}
        self.sampling_params = sampling_params

        self.custom_logit_processor = custom_logit_processor
        self.return_hidden_states = return_hidden_states

        # Memory pool info
        self.req_pool_idx: int | None = None
        self.req_to_token_pool_info: ReqToTokenPoolInfo | None = None
        # substitute for prefix_indices
        self.prefix_page_ids = []
        self.prefix_len = 0
        # Check finish
        self.tokenizer = None
        # Cached tokenizer-related ids to avoid repeated HF attribute lookups in check_finished().
        self._eos_token_id_cached: int | None = None
        self._additional_stop_token_ids_cached: set[int] | None = None
        self.finished_reason = None
        # Whether this request has finished output
        self.finished_output = None
        # If we want to abort the request in the middle of the event loop, set this to true
        # Note: We should never set finished_reason in the middle, the req will get filtered and never respond
        self.to_abort = False
        # This carries the error message for `.to_abort` and will be attached to the finished_reason at the end of the event loop
        self.to_abort_message: str = "Unknown error"
        # Set when grammar compilation failed (grammar is None); propagated via
        # disagg abort_flags so the decode side aborts the request.
        self.to_abort_grammar = False
        self.stream = stream
        self.eos_token_ids = eos_token_ids

        # For incremental decoding
        # ----- | --------- read_ids -------|
        # ----- |   surr_ids  |
        # xxxxx | xxxxxxxxxxx | xxxxxxxxxxx |
        # ----- ^ ----------- ^ ----------- ^
        # ----- 1 ----------- 2 ----------- 3
        # 1: surr_offset
        # 2: read_offset
        # 3: last token
        self.surr_offset = None  # Surrounding offset to defeat the cleanup algorithm
        self.read_offset = None
        self.decoded_text = ""

        # Prefix info
        # The indices to kv cache for the shared prefix.
        self.prefix_indices = []
        # Number of tokens to run prefill.
        self.extend_input_len = 0

        # The relative logprob_start_len in an extend batch
        self.extend_logprob_start_len = 0
        self.last_node = None

        # Whether or not if it is chunked. It increments whenever
        # it is chunked, and decrement whenever chunked request is
        # processed.
        self.is_chunked = 0

        # For retraction
        self.is_retracted = False

        # Incremental streamining
        self.send_token_offset: int = 0
        self.send_decode_id_offset: int = 0
        # TODO (Byron): send_output_token_logprobs_offset and send_decode_id_offset can be different in disaggregation mode
        # because the decode server does not have the first output token logprobs
        self.send_output_token_logprobs_offset: int = 0

        # Logprobs (arguments)
        self.return_logprob = return_logprob
        # Start index to compute logprob from.
        self.logprob_start_len = 0
        self.top_logprobs_num = top_logprobs_num
        self.token_ids_logprob = token_ids_logprob

        # Logprobs (return values)
        self.input_logprob_sent: bool = False
        self.input_token_logprobs_val: list[float] | None = None
        self.input_token_logprobs_idx: list[int] | None = None
        self.input_top_logprobs_val: list[float] | None = None
        self.input_top_logprobs_idx: list[int] | None = None
        self.input_token_ids_logprobs_val: list[float] | None = None
        self.input_token_ids_logprobs_idx: list[int] | None = None
        # Temporary holder to store input_token_logprobs.
        self.input_token_logprobs: list[tuple[int]] | None = None
        self.temp_input_top_logprobs_val: list[torch.Tensor] | None = None
        self.temp_input_top_logprobs_idx: list[int] | None = None
        self.temp_input_token_ids_logprobs_val: list[float] | None = None
        self.temp_input_token_ids_logprobs_idx: list[int] | None = None

        if return_logprob:
            self.output_token_logprobs_val = []
            self.output_token_logprobs_idx = []
            self.output_top_logprobs_val = []
            self.output_top_logprobs_idx = []
            self.output_token_ids_logprobs_val = []
            self.output_token_ids_logprobs_idx = []
        else:
            self.output_token_logprobs_val = self.output_token_logprobs_idx = self.output_top_logprobs_val = self.output_top_logprobs_idx = (
                self.output_token_ids_logprobs_val
            ) = self.output_token_ids_logprobs_idx = None
        self.hidden_states = []

        # Embedding (return values)
        self.embedding = None

        # Constrained decoding
        self.grammar_key: tuple[str, str] | None = None
        self.grammar: BaseGrammarObject | None = None
        self.grammar_done: bool = False

        # The number of cached tokens that were already cached in the KV cache
        self.cached_tokens = 0
        self.already_computed = 0
        # Indicates if the req has ever been retracted. Once set, never cleared.
        self.retracted_stain = False
        self.last_host_node: Any = None
        self.host_hit_length = 0
        self.l2_cache_hit_len = None  # Track L2 cache hits (before prefetch)

        # L3 prefetch state
        self.s2h_load: S2HLoad | None = None
        self.l3_prefetch_ready_tokens = 0
        # Per-stage prefetch latencies are observed in the scheduler; only
        # e2e/tokens cross to the tokenizer for the TTFT-percentage metric.
        self.l3_prefetch_e2e_s: float | None = None
        self.l3_prefetch_tokens: int | None = None

        # The number of verification forward passes in the speculative decoding.
        # This is used to compute the average acceptance length per request.
        self.spec_verify_ct = 0

        # Time of obj created
        # Use the created_time from tokenizer if provided, otherwise use current time
        if created_time is not None:
            self.created_time = created_time
        else:
            self.created_time = time.time()
        # Calculate the time from receiving the request at TokenizerManager to reaching process_input_requests in the scheduling process
        self.tokenizer_to_scheduler_latency = time.time() - self.created_time
        # For metrics
        self.metrics_collector = metrics_collector
        self.time_stats: TimeStats = TimeStats()
        self.has_log_time_stats: bool = False
        self.queue_time_start = None
        self.queue_time_end = None
        self.dequeue_time = None
        self.last_tic = time.monotonic()
        self.first_latency_recorded = False  # Flag to track if first latency has been recorded
        self.prefill_waiting_recorded = False
        self.first_chunk_forward_start_time = None

        self.reserve_num_tokens = 0
        # For disaggregation
        self.bootstrap_host: str = bootstrap_host
        self.bootstrap_port: int | None = bootstrap_port
        self.bootstrap_room: int | None = bootstrap_room
        self.disagg_kv_sender: BaseKVSender | None = None
        # Decode only: whether the non-prefix KV arrived or was restored.
        # Until then only the locked prefix [0, prefix_len) may be cached.
        self.pd_kv_ready: bool = True

        # for epd
        self.need_wait_for_mm_inputs: bool = False
        self.mm_meta: Any | None = None
        # Tokenization delayed to Scheduler admission.
        self.epd_media_urls: list | None = None
        self.epd_input_text: str | None = None
        # Resolved encoder URLs for this request (list[str]); Scheduler consumes
        # it as a plain list[str], isomorphic to server_args.encoder_urls.
        self.epd_encoder_urls: list | None = None

        # the start index of the sent kv cache
        # We want to send it chunk by chunk for chunked prefill.
        # After every chunk forward, we do the following:
        # kv_send(req.input_ids[req.start_send_idx:len(req.fill_ids)])
        # start_send_idx = len(req.fill_ids)
        self.start_send_idx: int = 0

        self.metadata_buffer_index: int = -1
        # Only meaningful in speculative reasoning.
        self.accept_draft_tokens: float | None = None

        # Reasoning token tracking
        self.reasoning_tokens = 0
        self._is_reasoning_over = False
        self._pending_reasoning_tokens = 0
        # Whether the request is currently inside a reasoning block. Only when this is
        # true do the pending tokens count as reasoning if generation stops early.
        self._in_reasoning = False
        self._reasoning_tracking_inited = False

        self.output_extra_info: dict[str, Any] = {}

    def set_tokenizer(self, tokenizer):
        """Assign tokenizer and cache ids needed by check_finished()."""
        self.tokenizer = tokenizer
        if tokenizer is None:
            self._eos_token_id_cached = None
            self._additional_stop_token_ids_cached = None
            return
        eos_id = getattr(tokenizer, "eos_token_id", None)
        self._eos_token_id_cached = int(eos_id) if eos_id is not None else None
        extra = getattr(tokenizer, "additional_stop_token_ids", None)
        self._additional_stop_token_ids_cached = set(int(x) for x in extra) if extra else None

    @property
    def seqlen(self):
        return len(self.origin_input_ids) + len(self.output_ids)

    @property
    def is_prefill_only(self) -> bool:
        """Check if this request is prefill-only (no token generation needed)."""
        return self.sampling_params.max_new_tokens == 0

    def add_latency(self, stage: RequestStage):
        if self.metrics_collector is None:
            return
        assert stage.name in RequestStage.__members__, f"{stage=} is invalid"
        now = time.monotonic()
        latency = now - self.last_tic

        # For the first latency record, add the tokenizer_to_scheduler latency
        if not self.first_latency_recorded:
            latency += self.tokenizer_to_scheduler_latency
            self.first_latency_recorded = True

        self.metrics_collector.observe_request_latency_seconds(stage.value, latency)
        self.last_tic = now

    def finished(self) -> bool:
        # Whether request reached finished condition
        return self.finished_reason is not None

    def init_next_round_input(self, tree_cache: RadixCache | None = None):
        self.fill_ids = self.origin_input_ids + self.output_ids
        if "multi_ids" in global_server_args_dict["mm_mode"]:
            self.fill_multi_ids = self.origin_input_multi_ids + self.output_multi_ids
        if self.input_embeds is not None:
            self.fill_input_embeds = self.input_embeds
        self.draft_fill_ids = self.fill_ids[1:] + [-1]
        if tree_cache is not None:
            match_result = tree_cache.match_prefix(key=self.adjust_max_prefix_ids())
            (self.prefix_page_ids, self.prefix_len, self.last_node, self.last_host_node, self.host_hit_length) = (
                match_result.device_indices,
                match_result.device_prefix_length,
                match_result.last_device_node,
                match_result.last_host_node,
                match_result.host_hit_length,
            )
            self.l2_cache_hit_len = self.host_hit_length if self.l2_cache_hit_len is None else self.l2_cache_hit_len
            logger.debug(
                "init_next_round_input after match rid=%s req_pool_idx=%s prefix_len=%s prefix_page_count=%s host_hit_length=%s l2_cache_hit_len=%s",
                self.rid,
                self.req_pool_idx,
                self.prefix_len,
                self.prefix_page_ids.numel(),
                self.host_hit_length,
                self.l2_cache_hit_len,
            )
        self.extend_input_len = len(self.fill_ids) - self.prefix_len

    def chunk_split(self, new_extend_input_len: int):
        # first will init_next_round_input, then chunk_split
        self.extend_input_len = new_extend_input_len
        self.fill_ids = self.fill_ids[: self.prefix_len + self.extend_input_len]
        self.draft_fill_ids = self.draft_fill_ids[: self.prefix_len + self.extend_input_len]
        if "multi_ids" in global_server_args_dict["mm_mode"]:
            self.fill_multi_ids = self.fill_multi_ids[: self.prefix_len + self.extend_input_len]
        if self.input_embeds is not None:
            # input_embeds only appear in input, ids/multi_ids will be used during autoregression
            self.fill_input_embeds = self.input_embeds[: self.prefix_len + self.extend_input_len]

    def adjust_max_prefix_ids(self):
        self.fill_ids = self.origin_input_ids + self.output_ids
        if "multi_ids" in global_server_args_dict["mm_mode"]:
            self.fill_multi_ids = self.origin_input_multi_ids + self.output_multi_ids
        if self.input_embeds is not None:
            self.fill_input_embeds = self.input_embeds
        input_len = len(self.fill_ids)

        # FIXME: To work around some bugs in logprob computation, we need to ensure each
        # request has at least one token. Later, we can relax this requirement and use `input_len`.
        max_prefix_len = input_len - 1

        if self.sampling_params.max_new_tokens > 0:
            # Need at least one token to compute logits
            max_prefix_len = min(max_prefix_len, input_len - 1)

        if self.return_logprob:
            max_prefix_len = min(max_prefix_len, self.logprob_start_len)

        max_prefix_len = max(max_prefix_len, 0)
        return self.fill_ids[:max_prefix_len]

    def init_incremental_detokenize(self):
        first_iter = self.surr_offset is None or self.read_offset is None

        if first_iter:
            # EPD+PD deferred tokenization: origin_input_ids_unpadded may be
            # None until the encoder returns ViT embeddings and the TM fills
            # in the token ids.  In that case skip detokenize init this iter;
            # stream_output will retry on the next step.
            if self.origin_input_ids_unpadded is None:
                self.surr_and_decode_ids = list(self.output_ids)
                self.cur_decode_ids_len = len(self.output_ids)
                self.read_offset = 0
                self.surr_offset = 0
                return self.surr_and_decode_ids, len(self.output_ids)
            self.read_offset = len(self.origin_input_ids_unpadded)
            # TODO[lifengcun]: Is this necessary?
            self.surr_offset = max(self.read_offset - INIT_INCREMENTAL_DETOKENIZATION_OFFSET, 0)
            # self.surr_offset = self.read_offset
            self.surr_and_decode_ids = self.origin_input_ids_unpadded[self.surr_offset :] + self.output_ids
            self.cur_decode_ids_len = len(self.output_ids)
        else:
            # copied from https://github.com/sgl-project/sglang/pull/10412/ to optimize post-processing performance
            self.surr_and_decode_ids.extend(self.output_ids[self.cur_decode_ids_len :])
            self.cur_decode_ids_len = len(self.output_ids)

        return self.surr_and_decode_ids, self.read_offset - self.surr_offset

    def check_finished(self, grammar_terminated=None):
        if self.finished():
            return

        if self.to_abort:
            self.finished_reason = FinishAbort(
                message=self.to_abort_message,
            )
            return

        if len(self.output_ids) >= self.sampling_params.max_new_tokens:
            self.finished_reason = FinishLength(length=self.sampling_params.max_new_tokens)
            return

        if self.grammar is not None:
            # Use per-step snapshot when provided (multi-step decode),
            # otherwise fall back to live grammar state.
            is_terminated = grammar_terminated if grammar_terminated is not None else self.grammar.is_terminated()
            if is_terminated:
                self.finished_reason = FinishMatchedToken(matched=self.output_ids[-1])
                return

        last_token_id = self.output_ids[-1]

        if not self.sampling_params.ignore_eos:
            matched_eos = False

            # Check stop token ids
            if self.sampling_params.stop_token_ids:
                matched_eos = last_token_id in self.sampling_params.stop_token_ids
            if self.eos_token_ids:
                matched_eos |= last_token_id in self.eos_token_ids
            if self.tokenizer is not None and self._eos_token_id_cached is None:
                self.set_tokenizer(self.tokenizer)
            if self._eos_token_id_cached is not None:
                matched_eos |= last_token_id == self._eos_token_id_cached
            if self._additional_stop_token_ids_cached:
                matched_eos |= last_token_id in self._additional_stop_token_ids_cached
            if matched_eos:
                self.finished_reason = FinishMatchedToken(matched=last_token_id)
                return

        # Check stop strings
        if len(self.sampling_params.stop_strs) > 0:
            tail_str = self.tokenizer.decode(self.output_ids[-(self.sampling_params.stop_str_max_len + 1) :])

            for stop_str in self.sampling_params.stop_strs:
                if stop_str in tail_str or stop_str in self.decoded_text:
                    self.finished_reason = FinishMatchedStr(matched=stop_str)
                    return

    def offload_kv_cache(self, req_to_token_pool: ReqToTokenPool, token_to_kv_pool: BaseTokenToKVPool):
        req_page_info = req_to_token_pool.get_req_pool_info(self.req_pool_idx)
        offload_len = len(self.origin_input_ids) + max(len(self.output_ids) - 1, 0)
        # The allocated slots may be more than origin_input_ids + output_ids - 1
        token_indices = req_page_info.alloced_slots[:offload_len]
        logger.info(f"[Req] {self} offload page_indices: {len(token_indices)}")
        self.kv_cache_cpu = token_to_kv_pool.get_cpu_copy(token_indices)

    def load_kv_cache(self, req_to_token_pool: ReqToTokenPool, token_to_kv_pool: BaseTokenToKVPool):
        from sglang.srt.mem_cache.memory_pool.host_memory_pool import RankLocalCpuCopy

        req_page_info = req_to_token_pool.get_req_pool_info(self.req_pool_idx)
        token_indices = req_page_info.alloced_slots
        logger.info(f"[Req] {self} load page_indices: {len(token_indices)}")
        if isinstance(self.kv_cache_cpu, RankLocalCpuCopy):
            page_size = token_to_kv_pool.page_size
            assert self.prefix_len % page_size == 0, f"prefix_len {self.prefix_len} must be page-aligned on rank-local resume path"
            prefix_page_count = self.prefix_len // page_size
        else:
            prefix_page_count = 0
        token_to_kv_pool.load_cpu_copy(self.kv_cache_cpu, token_indices, prefix_page_count=prefix_page_count)
        del self.kv_cache_cpu

    def reset_for_retract(self, delay_req_pool_release=False):
        self.prefix_indices = []
        self.last_node = None
        self.last_host_node = None
        self.host_hit_length = 0
        self.l2_cache_hit_len = None
        self.extend_input_len = 0
        self.prefill_cache_hit_tokens = 0
        self.prefill_cache_input_tokens = 0
        self.prefill_cache_l1_hit_tokens = 0
        self.prefill_cache_l2_hit_tokens = 0
        self.prefill_cache_l3_hit_tokens = 0
        self.is_retracted = True
        self.retracted_stain = True
        self.input_token_logprobs = None
        self.temp_input_top_logprobs_val = None
        self.temp_input_top_logprobs_idx = None
        self.extend_logprob_start_len = 0
        self.is_chunked = 0
        if not delay_req_pool_release:
            self.req_pool_idx = None
        self.req_to_token_pool_info = None

    def _init_reasoning_tracking(self, think_end_id, think_start_id, force_reasoning):
        """Decide whether the request starts out inside a reasoning block.

        The chat template may leave a think block at the end of the prompt. A trailing
        `</think>` means the template pre-closed an empty block to disable thinking
        (Qwen3 with enable_thinking=False emits `<think>\\n\\n</think>`), so reasoning is
        already over and must win over a `<think>` seen in the same window. Otherwise a
        trailing `<think>` means the template opened a block for the model to continue.
        Failing both, the parser may force reasoning from the first token (DeepSeek-R1).
        """
        self._reasoning_tracking_inited = True
        tail = self.origin_input_ids[-REASONING_PROMPT_SCAN_TOKENS:]
        if think_end_id is not None and think_end_id in tail:
            self._is_reasoning_over = True
        elif think_start_id is not None and think_start_id in tail or force_reasoning:
            self._in_reasoning = True

    def update_reasoning_tokens(self, token_id, think_end_id, think_start_id=None, force_reasoning=False):
        if self._is_reasoning_over:
            return
        if not self._reasoning_tracking_inited:
            self._init_reasoning_tracking(think_end_id, think_start_id, force_reasoning)
            if self._is_reasoning_over:
                # Template pre-closed the block; nothing counts as reasoning.
                return
        if not isinstance(token_id, list):
            token_id = [token_id]
        if not self._in_reasoning and think_start_id is not None and think_start_id in token_id:
            # The model opened the think block itself, mid-generation.
            self._in_reasoning = True
        if think_end_id in token_id:
            end_pos = token_id.index(think_end_id)
            self.reasoning_tokens = self._pending_reasoning_tokens + end_pos + 1
            self._is_reasoning_over = True
        else:
            self._pending_reasoning_tokens += len(token_id)

    def finalize_reasoning_tokens(self):
        """Settle the count when generation ends without a think end token.

        Reasoning was truncated (max_new_tokens, stop token, abort), so everything
        generated so far was reasoning. Requests that never entered a reasoning block
        keep a count of 0.
        """
        if self._is_reasoning_over or not self._in_reasoning:
            return
        self.reasoning_tokens = self._pending_reasoning_tokens
        self._is_reasoning_over = True

    def __repr__(self):
        return f"Req(rid={self.rid}, input_ids={len(self.origin_input_ids)}, output_ids={len(self.output_ids)})"
