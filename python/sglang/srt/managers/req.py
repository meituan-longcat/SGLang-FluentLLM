import copy
import time
import dataclasses
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import torch

from sglang.srt.constrained.base_grammar_backend import BaseGrammarObject
from sglang.srt.disaggregation.base import BaseKVSender
from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache
from sglang.srt.mem_cache.memory_pool import BaseTokenToKVPool, ReqToTokenPool, ReqToTokenPoolInfo
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.metrics.collector import SchedulerMetricsCollector, TimeStats
from sglang.srt.utils import get_colorful_logger
from sglang.srt.env import global_server_args_dict

logger = get_colorful_logger(__name__)

INIT_INCREMENTAL_DETOKENIZATION_OFFSET = 5


class BaseFinishReason:
    def __init__(self, is_error: bool = False):
        self.is_error = is_error

    def to_json(self):
        raise NotImplementedError()


class FINISH_MATCHED_TOKEN(BaseFinishReason):
    def __init__(self, matched: Union[int, List[int]]):
        super().__init__()
        self.matched = matched

    def to_json(self):
        return {
            "type": "stop",  # to match OpenAI API's return value
            "matched": self.matched,
        }


class FINISH_MATCHED_STR(BaseFinishReason):
    def __init__(self, matched: str):
        super().__init__()
        self.matched = matched

    def to_json(self):
        return {
            "type": "stop",  # to match OpenAI API's return value
            "matched": self.matched,
        }


class FINISH_LENGTH(BaseFinishReason):
    def __init__(self, length: int):
        super().__init__()
        self.length = length

    def to_json(self):
        return {
            "type": "length",  # to match OpenAI API's return value
            "length": self.length,
        }


class ABORT_CODE(Enum):
    TransferFailed = 521
    UnknownError = 522


class FINISH_ABORT(BaseFinishReason):
    def __init__(self, message="Unknown error", status_code=None, err_type=ABORT_CODE.UnknownError):
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

    pixel_values: Union[torch.Tensor, np.array]
    image_hashes: Optional[list] = None
    image_sizes: Optional[list] = None
    image_offsets: Optional[list] = None
    image_pad_len: Optional[list] = None
    pad_values: Optional[list] = None
    modalities: Optional[list] = None
    num_image_tokens: Optional[int] = None

    # Llava related
    aspect_ratio_ids: Optional[List[torch.Tensor]] = None
    aspect_ratio_mask: Optional[List[torch.Tensor]] = None

    # QWen2-VL related
    image_grid_thws: List[Tuple[int, int, int]] = None
    mrope_position_delta: Optional[torch.Tensor] = None

    # MiniCPMV related
    # All the images in the batch should share the same special image
    # bound token ids.
    im_start_id: Optional[torch.Tensor] = None
    im_end_id: Optional[torch.Tensor] = None
    slice_start_id: Optional[torch.Tensor] = None
    slice_end_id: Optional[torch.Tensor] = None
    tgt_sizes: Optional[list] = None

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
        origin_input_ids: Tuple[int],
        sampling_params: SamplingParams,
        return_logprob: bool = False,
        top_logprobs_num: int = 0,
        token_ids_logprob: List[int] = None,
        stream: bool = False,
        origin_input_ids_unpadded: Optional[Tuple[int]] = None,
        input_embeds: Optional[List[List[float]]] = None,
        input_extra_infos: Optional[List[Dict]] = None,
        session_id: Optional[str] = None,
        custom_logit_processor: Optional[str] = None,
        return_hidden_states: bool = False,
        eos_token_ids: Optional[Set[int]] = None,
        bootstrap_host: Optional[str] = None,
        bootstrap_port: Optional[int] = None,
        bootstrap_room: Optional[int] = None,
        data_parallel_rank: Optional[int] = None,
        origin_input_multi_ids: Optional[List[List[int]]] = None,
        metrics_collector: Optional[SchedulerMetricsCollector] = None,
        created_time: Optional[float] = None,
    ):
        # Input and output info
        self.rid = rid
        self.origin_input_text = origin_input_text
        self.origin_input_ids_unpadded = (
            origin_input_ids_unpadded
            if origin_input_ids_unpadded
            else origin_input_ids  # Before image padding
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
        self.state_info_dict = {}

        # for oe init
        self.oe_init = None

        # Sampling info
        if isinstance(sampling_params.custom_params, dict):
            sampling_params = copy.copy(sampling_params)
            sampling_params.custom_params = sampling_params.custom_params | {
                "__req__": self
            }
        self.sampling_params = sampling_params

        self.custom_logit_processor = custom_logit_processor
        self.return_hidden_states = return_hidden_states

        # Memory pool info
        self.req_pool_idx: Optional[int] = None
        self.req_to_token_pool_info: Optional[ReqToTokenPoolInfo] = None
        # substitute for prefix_indices
        self.prefix_page_ids = []
        self.prefix_len = 0
        # Check finish
        self.tokenizer = None
        # Cached tokenizer-related ids to avoid repeated HF attribute lookups in check_finished().
        self._eos_token_id_cached: Optional[int] = None
        self._additional_stop_token_ids_cached: Optional[Set[int]] = None
        self.finished_reason = None
        # Whether this request has finished output
        self.finished_output = None
        # If we want to abort the request in the middle of the event loop, set this to true
        # Note: We should never set finished_reason in the middle, the req will get filtered and never respond
        self.to_abort = False
        # This carries the error message for `.to_abort` and will be attached to the finished_reason at the end of the event loop
        self.to_abort_message: str = "Unknown error"
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
        self.input_token_logprobs_val: Optional[List[float]] = None
        self.input_token_logprobs_idx: Optional[List[int]] = None
        self.input_top_logprobs_val: Optional[List[float]] = None
        self.input_top_logprobs_idx: Optional[List[int]] = None
        self.input_token_ids_logprobs_val: Optional[List[float]] = None
        self.input_token_ids_logprobs_idx: Optional[List[int]] = None
        # Temporary holder to store input_token_logprobs.
        self.input_token_logprobs: Optional[List[Tuple[int]]] = None
        self.temp_input_top_logprobs_val: Optional[List[torch.Tensor]] = None
        self.temp_input_top_logprobs_idx: Optional[List[int]] = None
        self.temp_input_token_ids_logprobs_val: Optional[List[float]] = None
        self.temp_input_token_ids_logprobs_idx: Optional[List[int]] = None

        if return_logprob:
            self.output_token_logprobs_val = []
            self.output_token_logprobs_idx = []
            self.output_top_logprobs_val = []
            self.output_top_logprobs_idx = []
            self.output_token_ids_logprobs_val = []
            self.output_token_ids_logprobs_idx = []
        else:
            self.output_token_logprobs_val = self.output_token_logprobs_idx = (
                self.output_top_logprobs_val
            ) = self.output_top_logprobs_idx = self.output_token_ids_logprobs_val = (
                self.output_token_ids_logprobs_idx
            ) = None
        self.hidden_states = []

        # Embedding (return values)
        self.embedding = None

        # Constrained decoding
        self.grammar: Optional[BaseGrammarObject] = None

        # The number of cached tokens that were already cached in the KV cache
        self.cached_tokens = 0
        self.already_computed = 0
        self.last_host_node: Any = None
        self.host_hit_length = 0

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
        self.last_tic = time.monotonic()
        self.first_latency_recorded = False  # Flag to track if first latency has been recorded
        self.prefill_waiting_recorded = False
        self.first_chunk_forward_start_time = None

        self.reserve_num_tokens = 0
        # For disaggregation
        self.bootstrap_host: str = bootstrap_host
        self.bootstrap_port: Optional[int] = bootstrap_port
        self.bootstrap_room: Optional[int] = bootstrap_room
        self.data_parallel_rank: Optional[int] = data_parallel_rank
        self.disagg_kv_sender: Optional[BaseKVSender] = None

        # the start index of the sent kv cache
        # We want to send it chunk by chunk for chunked prefill.
        # After every chunk forward, we do the following:
        # kv_send(req.input_ids[req.start_send_idx:len(req.fill_ids)])
        # start_send_idx = len(req.fill_ids)
        self.start_send_idx: int = 0

        # For overlap schedule, we delay the kv transfer until `process_batch_result_disagg_prefill` rather than `process_prefill_chunk` in non-overlap
        # This is because kv is not ready in `process_prefill_chunk`.
        # We use `tmp_end_idx` to store the end index of the kv cache to send.
        self.tmp_end_idx: int = -1
        self.metadata_buffer_index: int = -1
        # Only meaningful in speculative reasoning.
        self.accept_draft_tokens: Optional[float] = None

        self.output_extra_info: Dict[str, Any] = {}
        self.output_cache_dict = {}

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
        self._additional_stop_token_ids_cached = (
            set(int(x) for x in extra) if extra else None
        )

    def get_group_specs(self):
        if not self.input_extra_infos:
            return None, None, None
        group_name = self.input_extra_infos[0].get("group_name", None)
        if group_name is not None and self.input_extra_infos[0].get("group_size") is None:
            group_size = 1
        else:
            group_size = self.input_extra_infos[0].get("group_size")
        group_extra_info = self.input_extra_infos[0].get("group_extra_info", {})
        return group_name, group_size, group_extra_info

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

        self.metrics_collector.observe_request_latency_seconds(
            stage.value, latency
        )
        self.last_tic = now

    def finished(self) -> bool:
        # Whether request reached finished condition
        return self.finished_reason is not None

    def init_next_round_input(self, tree_cache: Optional[BasePrefixCache] = None):
        self.fill_ids = self.origin_input_ids + self.output_ids
        if "multi_ids" in global_server_args_dict["mm_mode"]:
            self.fill_multi_ids = self.origin_input_multi_ids + self.output_multi_ids
        if self.input_embeds is not None:
            self.fill_input_embeds = self.input_embeds
        self.draft_fill_ids = []
        self.draft_fill_ids.extend(self.fill_ids[1:])
        self.draft_fill_ids += [-1]
        if tree_cache is not None:
            match_result = tree_cache.match_prefix(key=self.adjust_max_prefix_ids(), req=self)
            (self.prefix_page_ids, self.prefix_len, self.last_node, self.last_host_node, self.host_hit_length) = (
                match_result.device_indices, match_result.device_prefix_length, match_result.last_device_node, match_result.last_host_node, match_result.host_hit_length
            )
            logger.debug(
                f"init_next_round_input after  match rid={self.rid} req_pool_idx={self.req_pool_idx} prefix_len={self.prefix_len} prefix_page_ids={self.prefix_page_ids}"
            )
        self.extend_input_len = len(self.fill_ids) - self.prefix_len

    def init_next_round_input_chunk(self):
        self.fill_ids = self.origin_input_ids + self.output_ids
        # Recalculate extend_input_len based on current fill_ids and prefix_len
        self.extend_input_len = len(self.fill_ids) - self.prefix_len
        self.draft_fill_ids = []
        self.draft_fill_ids.extend(self.fill_ids[1:])
        self.draft_fill_ids += [-1]
        logger.debug(
            f"[init_next_round_input_chunk] rid={self.rid}, fill_ids={len(self.fill_ids)}, prefix_len={self.prefix_len}, extend_input_len={self.extend_input_len}\n"
            f"origin_input_ids={len(self.origin_input_ids)} output_ids={len(self.output_ids)}"
        )

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

    # Based on https://github.com/vllm-project/vllm/blob/7a64d24aad69e4d2548aa0bf528d9fe63428ab01/vllm/transformers_utils/detokenizer.py#L194-L313
    def init_incremental_detokenize(self):
        first_iter = self.surr_offset is None or self.read_offset is None

        if first_iter:
            self.read_offset = len(self.origin_input_ids_unpadded)
            # TODO[lifengcun]: Is this necessary?
            self.surr_offset = max(
                self.read_offset - INIT_INCREMENTAL_DETOKENIZATION_OFFSET, 0
            )
            # self.surr_offset = self.read_offset

        all_ids = self.origin_input_ids_unpadded + self.output_ids
        return all_ids[self.surr_offset :], self.read_offset - self.surr_offset

    def check_finished(self):
        if self.finished():
            return

        if self.to_abort:
            self.finished_reason = FINISH_ABORT(
                message=self.to_abort_message,
            )
            return

        if len(self.output_ids) >= self.sampling_params.max_new_tokens:
            self.finished_reason = FINISH_LENGTH(
                length=self.sampling_params.max_new_tokens
            )
            return

        if self.grammar is not None:
            if self.grammar.is_terminated():
                self.finished_reason = FINISH_MATCHED_TOKEN(matched=self.output_ids[-1])
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
                self.finished_reason = FINISH_MATCHED_TOKEN(matched=last_token_id)
                return

        # Check stop strings
        if len(self.sampling_params.stop_strs) > 0:
            tail_str = self.tokenizer.decode(
                self.output_ids[-(self.sampling_params.stop_str_max_len + 1) :]
            )

            for stop_str in self.sampling_params.stop_strs:
                if stop_str in tail_str or stop_str in self.decoded_text:
                    self.finished_reason = FINISH_MATCHED_STR(matched=stop_str)
                    return

    def offload_kv_cache(
        self, req_to_token_pool: ReqToTokenPool, token_to_kv_pool: BaseTokenToKVPool
    ):
        req_page_info = req_to_token_pool.get_req_pool_info(self.req_pool_idx)
        offload_len = len(self.origin_input_ids) + max(len(self.output_ids) - 1, 0)
        # The allocated slots may be more than origin_input_ids + output_ids - 1
        token_indices = req_page_info.alloced_slots[:offload_len]
        logger.debug(f"[Req] {self} offload page_indices: {len(token_indices)}")
        self.kv_cache_cpu = token_to_kv_pool.get_cpu_copy(token_indices)

    def load_kv_cache(
        self, req_to_token_pool: ReqToTokenPool, token_to_kv_pool: BaseTokenToKVPool
    ):
        req_page_info = req_to_token_pool.get_req_pool_info(self.req_pool_idx)
        token_indices = req_page_info.alloced_slots
        logger.debug(f"[Req] {self} load page_indices: {len(token_indices)}")
        token_to_kv_pool.load_cpu_copy(self.kv_cache_cpu, token_indices)
        del self.kv_cache_cpu

    def reset_for_retract(self, delay_req_pool_release=False):
        self.prefix_indices = []
        self.last_node = None
        self.last_host_node = None
        self.extend_input_len = 0
        self.is_retracted = True
        self.input_token_logprobs = None
        self.temp_input_top_logprobs_val = None
        self.temp_input_top_logprobs_idx = None
        self.extend_logprob_start_len = 0
        self.is_chunked = 0
        if not delay_req_pool_release:
            self.req_pool_idx = None
        self.req_to_token_pool_info = None

    def __repr__(self):
        return (
            f"Req(rid={self.rid}, "
            f"input_ids={len(self.origin_input_ids)}, output_ids={len(self.output_ids)})"
        )
