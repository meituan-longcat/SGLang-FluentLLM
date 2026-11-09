# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""TokenizerManager is a process that tokenizes the text."""

import asyncio
import concurrent.futures
import copy
import dataclasses
import inspect
import logging
import multiprocessing as mp
import os
import pickle
import signal
import sys
import threading
import time
import uuid
from collections import deque
from collections.abc import Awaitable
from datetime import datetime
from enum import Enum
from http import HTTPStatus
from typing import (
    Any,
    Generic,
    Optional,
    TypeVar,
    Union,
)

import fastapi
import setproctitle
import transformers
import uvloop
import zmq
import zmq.asyncio
from fastapi import BackgroundTasks
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.disaggregation.utils import (
    DisaggregationMode,
    KVClassType,
    TransferBackend,
    get_kv_class,
)
from sglang.srt.managers.io_struct import (
    AbortReq,
    BatchEmbeddingOut,
    BatchMultimodalOut,
    BatchStrOut,
    BatchTokenIDOut,
    CloseSessionReqInput,
    ConfigureLoggingReq,
    EmbeddingReqInput,
    FlushCacheReqInput,
    FlushCacheReqOutput,
    GenerateReqInput,
    GetLoadReqInput,
    HealthCheckOutput,
    OpenSessionReqInput,
    OpenSessionReqOutput,
    SessionParams,
    SetSchedulerMaxRunningRequestsInput,
    TokenizedEmbeddingReqInput,
    TokenizedGenerateReqInput,
    UpdateWeightFromDiskReqInput,
    UpdateWeightFromDiskReqOutput,
    WatchLoadUpdateReq,
)
from sglang.srt.managers.scheduling.multimodal_data import (
    Modality,
)
from sglang.srt.managers.tokenizer_communicator_mixin import TokenizerCommunicatorMixin
from sglang.srt.managers.tokenizer_multimodal_mixin import TokenizerMultimodalMixin
from sglang.srt.observability.metrics.collector import ErrorMetricsCollector, TokenizerMetricsCollector
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.utils import (
    dataclass_to_string_truncated,
    get_colorful_logger,
    get_zmq_socket,
    kill_process_tree,
)
from sglang.srt.utils.aio_rwlock import RWLock
from sglang.srt.utils.hf_transformers_utils import get_processor, get_tokenizer, get_tokenizer_from_processor
from sglang.utils import TypeBasedDispatcher, get_exception_traceback

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

logger = get_colorful_logger(__name__)

global global_tokenizer


def init_global_tokenizer(server_args: ServerArgs):
    """Init the global tokenizer for process pool."""
    global global_tokenizer
    setproctitle.setproctitle("sglang::tokenizer_worker")
    global_tokenizer = get_tokenizer(
        server_args.tokenizer_path,
        tokenizer_mode=server_args.tokenizer_mode,
        trust_remote_code=server_args.trust_remote_code,
        revision=server_args.revision,
    )


def tokenizer_encode_task(
    input_text: str | list[str],
    encode_kwargs: dict[str, Any] | None = None,
    remove_bos: bool = False,
    /,
    *,
    tokenizer: transformers.PreTrainedTokenizerBase | None = None,
) -> list[int]:
    global global_tokenizer
    if tokenizer is None:
        tokenizer = global_tokenizer
    if encode_kwargs is None:
        encode_kwargs = {}
    encoded = tokenizer.encode(input_text, **encode_kwargs)
    if hasattr(encoded, "ids"):
        encoded = list(encoded.ids)
    elif not isinstance(encoded, list):
        encoded = list(encoded)
    if remove_bos and encoded and encoded[0] == tokenizer.bos_token_id:
        encoded = encoded[1:]
    return encoded


def tokenizer_apply_chat_template_task(
    openai_compatible_messages: list[dict],
    tools: Optional[list[dict]],
    reasoning_effort: Optional[str],
    chat_template_kwargs: Optional[dict],
    /,
    *,
    tokenizer: Optional[transformers.PreTrainedTokenizerBase] = None,
):
    global global_tokenizer
    if tokenizer is None:
        tokenizer = global_tokenizer
    if chat_template_kwargs is None:
        chat_template_kwargs = {}
    else:
        # chat_template_kwargs comes from the request; drop keys that would
        # collide with the explicit keyword arguments below.
        chat_template_kwargs = dict(chat_template_kwargs)
        reasoning_effort = chat_template_kwargs.pop("reasoning_effort", reasoning_effort)
        for key in ("tokenize", "add_generation_prompt", "tools", "builtin_tools"):
            chat_template_kwargs.pop(key, None)
    encoded = tokenizer.apply_chat_template(
        openai_compatible_messages,
        tokenize=True,
        add_generation_prompt=True,
        tools=tools,
        reasoning_effort=reasoning_effort,
        builtin_tools=[],
        **chat_template_kwargs,
    )
    if hasattr(encoded, "ids"):
        encoded = list(encoded.ids)
    elif hasattr(encoded, "input_ids"):
        ids = encoded.input_ids
        if len(ids) > 0 and isinstance(ids[0], (list, tuple)):
            ids = ids[0]
        encoded = list(ids)
    elif not isinstance(encoded, list):
        encoded = list(encoded)
    return encoded


def tokenizer_apply_chat_template_render_task(
    openai_compatible_messages: list[dict],
    tools: Optional[list[dict]],
    reasoning_effort: Optional[str],
    chat_template_kwargs: Optional[dict],
    /,
    *,
    tokenizer: Optional[transformers.PreTrainedTokenizerBase] = None,
) -> str:
    """Apply chat template with tokenize=False, returning the rendered text string."""
    global global_tokenizer
    if tokenizer is None:
        tokenizer = global_tokenizer
    if chat_template_kwargs is None:
        chat_template_kwargs = {}
    return tokenizer.apply_chat_template(
        openai_compatible_messages,
        tokenize=False,
        add_generation_prompt=True,
        tools=tools,
        reasoning_effort=reasoning_effort,
        builtin_tools=[],
        **chat_template_kwargs,
    )


def tokenizer_decode_task(
    prompt_ids: list[int], decode_kwargs: Optional[dict[str, Any]] = None, /, *, tokenizer: Optional[transformers.PreTrainedTokenizerBase] = None
) -> str:
    global global_tokenizer
    if tokenizer is None:
        tokenizer = global_tokenizer
    if decode_kwargs is None:
        decode_kwargs = {}
    return tokenizer.decode(prompt_ids, **decode_kwargs)


def tokenizer_batch_decode_task(
    sequences: Union[list[int], list[list[int]]],
    decode_kwargs: Optional[dict[str, Any]] = None,
    /,
    *,
    tokenizer: Optional[transformers.PreTrainedTokenizerBase] = None,
) -> list[str]:
    global global_tokenizer
    if tokenizer is None:
        tokenizer = global_tokenizer
    if decode_kwargs is None:
        decode_kwargs = {}
    return tokenizer.batch_decode(sequences, **decode_kwargs)


def tokenizer_query_chat_template_task(*, tokenizer: Optional[transformers.PreTrainedTokenizerBase] = None) -> Optional[str]:
    global global_tokenizer
    if tokenizer is None:
        tokenizer = global_tokenizer
    return tokenizer.chat_template


@dataclasses.dataclass
class ReqState:
    """Store the state a request."""

    out_list: list
    finished: bool
    event: asyncio.Event
    obj: Any

    # For metrics
    created_time: float
    metric_request_type: Optional[str] = None
    tokenized_time: float = 0.0
    finished_time: float = 0.0
    first_token_time: float = 0.0
    first_completion_tokens: int = 1
    last_time: float = 0.0
    last_pure_time: float = 0.0
    last_completion_tokens: int = 1

    # For streaming output
    last_output_offset: int = 0
    # Set by the message handler, never inferred from token contents.
    output_ids_are_incremental: bool = False

    # For incremental state update.
    text: str = ""
    output_ids: list[int] = dataclasses.field(default_factory=list)
    logprobs_info: dict = dataclasses.field(default_factory=dict)
    last_meta_info: dict = dataclasses.field(default_factory=dict)

    # Set when abort_request is waiting for final usage from scheduler.
    abort_future: Optional[asyncio.Future] = None


class ServerStatus(Enum):
    Up = "Up"
    Starting = "Starting"
    UnHealthy = "UnHealthy"
    Crashed = "Crashed"

    def is_healthy(self) -> bool:
        return self == ServerStatus.Up


class TTFTCounter:
    """Count TTFT by different input length."""

    def __init__(self, ttft_by_input_len_list: str):
        ttft_by_input_len_list = ttft_by_input_len_list.split(",")
        self.ttft_by_input_len_list = [int(input_len) for input_len in ttft_by_input_len_list]

    def counter_name(self, input_len: int):
        for i in range(len(self.ttft_by_input_len_list)):
            if input_len <= self.ttft_by_input_len_list[i]:
                return f"TTFT_{self.ttft_by_input_len_list[i]}"
        return None


class TokenizerManager(TokenizerCommunicatorMixin, TokenizerMultimodalMixin):
    """TokenizerManager is a process that tokenizes the text."""

    def __init__(
        self,
        server_args: ServerArgs,
        port_args: PortArgs,
    ):
        # Parse args
        self.server_args = server_args
        self.enable_metrics = server_args.enable_metrics
        self.log_requests = server_args.log_requests
        self.log_requests_level = server_args.log_requests_level

        # Init inter-process communication
        context = zmq.asyncio.Context(2)
        self.recv_from_detokenizer = get_zmq_socket(context, zmq.PULL, port_args.tokenizer_ipc_name, True)
        self.send_to_scheduler = get_zmq_socket(context, zmq.PUSH, port_args.scheduler_input_ipc_name, True)

        # Read model args
        self.model_path = server_args.model_path
        self.served_model_name = server_args.served_model_name
        self.model_config = ModelConfig(
            server_args.model_path,
            trust_remote_code=server_args.trust_remote_code,
            revision=server_args.revision,
            context_length=server_args.context_length,
            model_override_args=server_args.json_model_override_args,
            is_embedding=server_args.is_embedding,
            dtype=server_args.dtype,
            quantization=server_args.quantization,
            server_args=server_args,
        )

        self.is_generation = self.model_config.is_generation
        self.is_image_gen = self.model_config.is_image_gen
        self.context_len = self.model_config.context_len
        self.image_token_id = self.model_config.image_token_id
        self.is_longcat_model = self.model_config.is_longcat_model
        self.is_think_model = self.server_args.served_model_name is not None and "thinking" in self.server_args.served_model_name.lower()
        # Create tokenizer
        if server_args.skip_tokenizer_init:
            self.tokenizer = self.processor = self.tokenizer_executor = None
        else:
            if self.model_config.is_multimodal:
                self.processor = get_processor(
                    server_args.tokenizer_path,
                    tokenizer_mode=server_args.tokenizer_mode,
                    trust_remote_code=server_args.trust_remote_code,
                    revision=server_args.revision,
                )
                self.tokenizer = get_tokenizer_from_processor(self.processor)
                # Image processing (self.mm_processor -> _apply_mm_processing) always runs
                # in-process; the ProcessPool only handles text tokenization tasks
                # (encode/apply_chat_template/decode via run_tokenizer_task), so the two are
                # orthogonal. self.processor/self.tokenizer stay set for mm processing + as the
                # in-process fallback; when a pool is configured, run_tokenizer_task prefers it.
                os.environ["TOKENIZERS_PARALLELISM"] = "false"
                if server_args.tokenizer_executor_num_processes is not None:
                    self.tokenizer_executor = concurrent.futures.ProcessPoolExecutor(
                        initializer=init_global_tokenizer,
                        mp_context=mp.get_context("fork"),
                        initargs=(server_args,),
                        max_workers=server_args.tokenizer_executor_num_processes,
                    )
                else:
                    self.tokenizer_executor = None
            else:
                self.processor = None
                # Non-multimodal models
                if server_args.tokenizer_executor_num_processes is not None:
                    # Set environment variable before forking
                    os.environ["TOKENIZERS_PARALLELISM"] = "false"
                    # Use process pool executor for tokenizer
                    self.tokenizer = None
                    self.tokenizer_executor = concurrent.futures.ProcessPoolExecutor(
                        initializer=init_global_tokenizer,
                        mp_context=mp.get_context("fork"),
                        initargs=(server_args,),
                        max_workers=server_args.tokenizer_executor_num_processes,
                    )
                else:
                    # Traditional single-process tokenizer
                    self.tokenizer = get_tokenizer(
                        server_args.tokenizer_path,
                        tokenizer_mode=server_args.tokenizer_mode,
                        trust_remote_code=server_args.trust_remote_code,
                        revision=server_args.revision,
                    )
                    self.tokenizer_executor = None

        # mm_processor and transport strategy are initialized by TokenizerMultimodalMixin
        self.init_multimodal(server_args)

        # Store states
        self.no_create_loop = False
        self.rid_to_state: dict[str, ReqState] = {}
        self.gracefully_exit = False
        self.last_receive_tstamp = 0
        self.dump_requests_folder = ""  # By default do not dump
        self.dump_requests_threshold = 1000
        self.dump_request_list: list[tuple] = []
        self.log_request_metadata = self.get_log_request_metadata()
        self.server_status = ServerStatus.Starting

        # The event to notify the weight sync is finished.
        self.model_update_lock = RWLock()
        self.model_update_result: Optional[Awaitable[UpdateWeightFromDiskReqOutput]] = None
        self.asyncio_tasks = set()

        # For session info
        self.session_futures = {}  # session_id -> asyncio event

        # Set after scheduler is initialized
        self.max_req_input_len = None

        # Metrics
        if self.enable_metrics:
            self.metrics_collector = TokenizerMetricsCollector(
                labels={
                    "model_name": self.server_args.served_model_name,
                    "app_key": self.server_args.app_key,
                },
                metrics_reporters=server_args.metrics_reporters,
                enable_multimodal=server_args.enable_multimodal,
            )
            self.error_collector = ErrorMetricsCollector(
                labels={
                    "model_name": self.server_args.served_model_name,
                    "app_key": self.server_args.app_key,
                },
                metrics_reporters=server_args.metrics_reporters,
            )

        self._result_dispatcher = TypeBasedDispatcher(
            [
                (
                    (
                        BatchStrOut,
                        BatchEmbeddingOut,
                        BatchTokenIDOut,
                        BatchMultimodalOut,
                    ),
                    self._handle_batch_output,
                ),
                (OpenSessionReqOutput, self._handle_open_session_req_output),
                (
                    UpdateWeightFromDiskReqOutput,
                    self._handle_update_weights_from_disk_req_output,
                ),
                (HealthCheckOutput, lambda x: None),
            ]
        )

        self.disaggregation_mode = DisaggregationMode(self.server_args.disaggregation_mode)
        self.transfer_backend = TransferBackend(self.server_args.disaggregation_transfer_backend)
        # for disaggregtion, start kv boostrap server on prefill
        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            # only start bootstrap server on prefill tm
            kv_bootstrap_server_class = get_kv_class(self.transfer_backend, KVClassType.BOOTSTRAP_SERVER)
            self.bootstrap_server = kv_bootstrap_server_class(self.server_args.disaggregation_bootstrap_port)

        self.init_communicators(server_args)
        self.ttft_counter = None
        if self.server_args.ttft_by_input_len_list is not None:
            self.ttft_counter = TTFTCounter(self.server_args.ttft_by_input_len_list)

    async def set_scheduler_max_running_req(self, max_running_req: int):
        """Set the maximum concurrent requests for the scheduler"""
        self.auto_create_handle_loop()

        req = SetSchedulerMaxRunningRequestsInput(max_running_requests=max_running_req)
        results = await self.set_scheduler_max_running_communicator(req)

        if self.server_args.dp_size == 1:
            return results[0].success, results[0].message
        else:
            all_success = all(r.success for r in results)
            all_messages = " | ".join(r.message for r in results)
            return all_success, all_messages

    async def generate_request(
        self,
        obj: Union[GenerateReqInput, EmbeddingReqInput],
        request: Optional[fastapi.Request] = None,
    ):
        created_time = time.time()

        self.auto_create_handle_loop()

        if isinstance(obj, EmbeddingReqInput) and self.is_generation:
            raise ValueError(
                "This model does not appear to be an embedding model by default. "
                "Please add `--is-embedding` when launching the server or try another model."
            )

        obj.normalize_batch_and_arguments()

        if self.enable_metrics:
            batch_size = obj.batch_size if hasattr(obj, "batch_size") else 1
            self.metrics_collector.observe_request_arrival(batch_size)

        if self.log_requests:
            max_length, skip_names, _ = self.log_request_metadata
            logger.info(f"Receive: obj={dataclass_to_string_truncated(obj, max_length, skip_names=skip_names)}")

        async with self.model_update_lock.reader_lock:
            is_single = obj.is_single
            if is_single:
                tokenized_obj = await self._tokenize_one_request(obj)
                self._send_one_request(obj, tokenized_obj, created_time)
                async for response in self._wait_one_response(obj, request):
                    yield response
            else:
                async for response in self._handle_batch_request(obj, request, created_time):
                    yield response

    async def _normalize_sampling_params(self, sampling_params: SamplingParams):
        # Process stop strings
        if sampling_params.stop_strs is None:
            sampling_params.stop_strs = []
            sampling_params.stop_str_max_len = 0
        else:
            if isinstance(sampling_params.stop_strs, str):
                sampling_params.stop_strs = [sampling_params.stop_strs]

            stop_str_max_len = 0
            for stop_str in sampling_params.stop_strs:
                if not (self.tokenizer is None and self.tokenizer_executor is None):
                    stop_str_ids = await self.run_tokenizer_task(tokenizer_encode_task, stop_str, {"add_special_tokens": False})
                    stop_str_max_len = max(stop_str_max_len, len(stop_str_ids))
                else:
                    stop_str_max_len = max(stop_str_max_len, len(stop_str))
            sampling_params.stop_str_max_len = stop_str_max_len

    async def _tokenize_one_request(
        self,
        obj: Union[GenerateReqInput, EmbeddingReqInput],
    ):
        """Tokenize one request."""
        # Tokenize
        input_embeds = None
        mm_inputs = None
        input_ids = None
        input_text = obj.text

        modality = self._get_mm_modality(obj, input_text) if isinstance(obj, GenerateReqInput) else None
        has_mm_data = modality is not None
        use_mm_processing = isinstance(obj, GenerateReqInput) and self.mm_processor is not None and has_mm_data
        if obj.input_embeds is not None:
            if not self.server_args.disable_radix_cache:
                raise ValueError(
                    "input_embeds is provided while disable_radix_cache is False. "
                    "Please add `--disable-radix-cache` when you launch the server "
                    "if you want to use input_embeds as inputs."
                )
            input_embeds: list[float] = obj.input_embeds
            input_ids = obj.input_ids
        elif obj.input_ids is not None:
            input_ids = obj.input_ids
        else:
            if use_mm_processing:
                # pd disaggregation, decode call apply_mm_processing to hash feature and pad input_ids
                input_ids, mm_inputs = await self._apply_mm_processing(obj, input_text, input_ids, modality)
            else:
                input_ids = await self.run_tokenizer_task(tokenizer_encode_task, input_text)

        modality = self._resolve_mm_modality(mm_inputs, modality)

        if self.is_generation:
            return_logprob = obj.return_logprob
            logprob_start_len = obj.logprob_start_len
            top_logprobs_num = obj.top_logprobs_num
            token_ids_logprob = obj.token_ids_logprob
            session_params = SessionParams(**obj.session_params) if obj.session_params else None

        input_token_num = len(input_ids) if input_ids is not None else 0
        if input_token_num >= (self.context_len - 6):
            raise ValueError(
                f"The input ({input_token_num} tokens) is longer than the model's context length ({(self.context_len - 6)} tokens(todo fix))."
            )

        if obj.sampling_params.get("max_new_tokens") is not None and obj.sampling_params.get("max_new_tokens") + input_token_num >= self.context_len:
            adjusted_max_new_tokens = self.context_len - input_token_num
            msg = (
                f"Requested(rid={obj.rid}) token count exceeds the model's maximum context length "
                f"of {self.context_len} tokens. You requested a total of "
                f"{obj.sampling_params.get('max_new_tokens') + input_token_num} "
                f"tokens: {input_token_num} tokens from the input messages and "
                f"{obj.sampling_params.get('max_new_tokens')} tokens for the "
                f"completion. The max_new_tokens will be truncated to {adjusted_max_new_tokens}."
            )
            logger.warning(msg)

            obj.sampling_params.update({"max_new_tokens": adjusted_max_new_tokens})

        # Parse sampling parameters
        sampling_params = SamplingParams(**obj.sampling_params)
        await self._normalize_sampling_params(sampling_params)
        sampling_params.verify(self.model_config.vocab_size)

        # Build return object
        if isinstance(obj, GenerateReqInput):
            need_wait_for_mm_inputs = False
            mm_meta = None
            epd_media_urls = None
            epd_input_text = None
            epd_encoder_urls = None
            # Carry media_urls + input_text so the Scheduler can run process_mm_data_async once mm_meta arrives on the MMResult.
            if self._epd_language_only and mm_inputs is not None and mm_inputs.epd_media_urls is not None:
                epd_media_urls = mm_inputs.epd_media_urls
                epd_input_text = mm_inputs.epd_input_text
                epd_encoder_urls = mm_inputs.epd_encoder_urls
                need_wait_for_mm_inputs = True

            tokenized_obj = TokenizedGenerateReqInput(
                obj.rid,
                input_text,
                input_ids,
                sampling_params,
                return_logprob,
                logprob_start_len,
                top_logprobs_num,
                token_ids_logprob,
                obj.stream,
                bootstrap_host=obj.bootstrap_host,
                bootstrap_port=obj.bootstrap_port,
                bootstrap_room=obj.bootstrap_room,
                routed_dp_rank=obj.routed_dp_rank,
                input_embeds=input_embeds,
                session_params=session_params,
                custom_logit_processor=obj.custom_logit_processor,
                return_hidden_states=obj.return_hidden_states,
                created_time=time.time(),
                input_multi_ids=obj.input_multi_ids,
                input_extra_infos=obj.input_extra_infos,
                modality=modality,
                mm_inputs=mm_inputs,
                # EPD fields: tell the Scheduler to wait for Encoder output.
                need_wait_for_mm_inputs=need_wait_for_mm_inputs,
                mm_meta=mm_meta,
                # epd deferred tokenization data.
                epd_media_urls=epd_media_urls,
                epd_input_text=epd_input_text,
                epd_encoder_urls=epd_encoder_urls,
            )
        elif isinstance(obj, EmbeddingReqInput):
            tokenized_obj = TokenizedEmbeddingReqInput(
                obj.rid,
                input_text,
                input_ids,
                sampling_params,
                created_time=time.time(),
            )

        return tokenized_obj

    async def run_tokenizer_task(self, task, *args):
        if self.tokenizer_executor is not None:
            loop = asyncio.get_event_loop()
            input_ids = await loop.run_in_executor(self.tokenizer_executor, task, *args)
        elif self.tokenizer is not None:
            input_ids = task(*args, tokenizer=self.tokenizer)
        else:
            raise ValueError(
                "The engine initialized with skip_tokenizer_init=True cannot "
                "accept text prompts. Please provide input_ids or re-initialize "
                "the engine with skip_tokenizer_init=False."
            )
        return input_ids

    def _get_metric_request_type(
        self,
        obj: Union[GenerateReqInput, EmbeddingReqInput],
        tokenized_obj: Union[TokenizedGenerateReqInput, TokenizedEmbeddingReqInput],
    ) -> str:
        if not isinstance(obj, GenerateReqInput) or tokenized_obj.modality is None:
            return "text"

        return "mm.VIDEO" if tokenized_obj.modality is Modality.VIDEO else "mm.IMAGE"

    def _send_one_request(
        self,
        obj: Union[GenerateReqInput, EmbeddingReqInput],
        tokenized_obj: Union[TokenizedGenerateReqInput, TokenizedEmbeddingReqInput],
        created_time: Optional[float] = None,
    ):
        metric_request_type = self._get_metric_request_type(obj, tokenized_obj) if self.server_args.enable_multimodal else None
        state = ReqState(
            [],
            False,
            asyncio.Event(),
            obj,
            created_time=created_time,
            metric_request_type=metric_request_type,
            tokenized_time=tokenized_obj.created_time,
        )
        self.rid_to_state[obj.rid] = state
        tokenized_obj = self.wrap_mm_features(tokenized_obj)
        self.send_to_scheduler.send_pyobj(tokenized_obj)

    async def _wait_one_response(
        self,
        obj: Union[GenerateReqInput, EmbeddingReqInput],
        request: Optional[fastapi.Request] = None,
    ):
        """Wait for the response of one request."""
        state = self.rid_to_state[obj.rid]

        while True:
            try:
                await asyncio.wait_for(state.event.wait(), timeout=4)
            except asyncio.TimeoutError as e:
                if request is not None and await request.is_disconnected():
                    await self.abort_request(obj.rid)
                    raise ValueError(f"Request is disconnected from the client side. Abort request {obj.rid}") from e
                continue

            out = state.out_list[-1]

            # The handler defines the ID mode from the message type and request
            # flags. Repeated/prefix-equal deltas are still distinct tokens.
            # Text and usage counters are cumulative: keep the last snapshot.
            if len(state.out_list) > 1:
                out = dict(out)
                if state.output_ids_are_incremental:
                    for key in ("output_ids", "output_multi_ids"):
                        if key in out:
                            out[key] = [
                                item
                                for entry in state.out_list
                                for item in entry.get(key, [])
                            ]
                # Streaming logprobs are deltas for BOTH message types, even
                # when BatchTokenIDOut exposes cumulative IDs. Preserve input
                # logprobs sent once and every output probability in the queue.
                if obj.stream and getattr(obj, "return_logprob", False):
                    out["meta_info"] = dict(out["meta_info"])
                    for key in (
                        "input_token_logprobs", "output_token_logprobs",
                        "input_top_logprobs", "output_top_logprobs",
                        "input_token_ids_logprobs", "output_token_ids_logprobs",
                    ):
                        if any(key in entry["meta_info"] for entry in state.out_list):
                            out["meta_info"][key] = [
                                item
                                for entry in state.out_list
                                for item in entry["meta_info"].get(key, [])
                            ]

            state.out_list = []
            if state.finished:
                if self.log_requests:
                    max_length, skip_names, out_skip_names = self.log_request_metadata
                    if self.model_config.is_multimodal_gen:
                        msg = f"Finish: obj={dataclass_to_string_truncated(obj, max_length, skip_names=skip_names)}"
                    else:
                        # Do not decode a long prompt when its text is omitted from the log.
                        if (
                            (skip_names is None or "text" not in skip_names)
                            and isinstance(obj, GenerateReqInput)
                            and obj.input_ids is not None
                            and obj.text is None
                        ):
                            if self.tokenizer is not None:
                                obj.text = self.tokenizer.decode(
                                    obj.input_ids, skip_special_tokens=getattr(obj.sampling_params, "skip_special_tokens", False)
                                )
                            else:
                                obj.text = ""
                        msg = (
                            f"Finish: obj={dataclass_to_string_truncated(obj, max_length, skip_names=skip_names)}, "
                            f"out={dataclass_to_string_truncated(out, max_length, skip_names=out_skip_names)}"
                        )
                    logger.info(msg)
                self.rid_to_state.pop(obj.rid, None)

                # Check if this was an abort/error created by scheduler
                if isinstance(out["meta_info"].get("finish_reason"), dict):
                    finish_reason = out["meta_info"]["finish_reason"]
                    if finish_reason.get("type") == "abort":
                        if self.enable_metrics:
                            self.error_collector.record_error(finish_reason.get("message") or "")
                        if finish_reason.get("status_code") == HTTPStatus.BAD_REQUEST:
                            raise ValueError(finish_reason["message"])

                yield out
                break

            state.event.clear()

            if obj.stream:
                yield out
            else:
                if request is not None and await request.is_disconnected():
                    await self.abort_request(obj.rid)
                    raise ValueError(f"Request is disconnected from the client side. Abort request {obj.rid}")

    async def _handle_batch_request(
        self,
        obj: Union[GenerateReqInput, EmbeddingReqInput],
        request: Optional[fastapi.Request] = None,
        created_time: Optional[float] = None,
    ):
        batch_size = obj.batch_size

        generators = []
        rids = []
        if getattr(obj, "parallel_sample_num", 1) == 1:
            # Send all requests
            for i in range(batch_size):
                tmp_obj = obj[i]
                tokenized_obj = await self._tokenize_one_request(tmp_obj)
                self._send_one_request(tmp_obj, tokenized_obj, created_time)
                generators.append(self._wait_one_response(tmp_obj, request))
                rids.append(tmp_obj.rid)
        else:
            # FIXME: When using batch and parallel_sample_num together, the perf is not optimal.
            if batch_size > 128:
                logger.warning(
                    "Sending a single large batch with parallel sampling (n > 1) has not been well optimized. "
                    "The performance might be better if you just duplicate the requests n times or use "
                    "many threads to send them one by one with parallel sampling (n > 1)."
                )

            # Tokenize all requests
            objs = [obj[i] for i in range(batch_size)]
            tokenized_objs = await asyncio.gather(*(self._tokenize_one_request(obj) for obj in objs))

            # Cache the common prefix for parallel sampling
            for i in range(batch_size):
                tmp_obj = copy.copy(objs[i])
                tokenized_obj = copy.copy(tokenized_objs[i])
                tokenized_obj.rid = tmp_obj.regenerate_rid()
                tokenized_obj.sampling_params = copy.copy(tokenized_obj.sampling_params)
                tokenized_obj.sampling_params.max_new_tokens = 0
                tokenized_obj.stream = False
                self._send_one_request(tmp_obj, tokenized_obj, created_time)
                await self._wait_one_response(tmp_obj, request).__anext__()

            # Expand requests, assign new rids for them, and send them
            for i in range(batch_size):
                for _ in range(obj.parallel_sample_num):
                    tmp_obj = copy.copy(objs[i])
                    tokenized_obj = copy.copy(tokenized_objs[i])
                    tokenized_obj.rid = tmp_obj.regenerate_rid()
                    self._send_one_request(tmp_obj, tokenized_obj, created_time)
                    generators.append(self._wait_one_response(tmp_obj, request))
                    rids.append(tmp_obj.rid)

        # Wait for all requests
        is_stream = hasattr(obj, "stream") and obj.stream
        if not is_stream:
            outputs = await asyncio.gather(*(gen.__anext__() for gen in generators))
            yield outputs
        else:
            rid_to_index = {rid: i for i, rid in enumerate(rids)}
            task_map = {asyncio.create_task(gen.__anext__()): gen for gen in generators}
            while task_map:
                done, _ = await asyncio.wait(task_map.keys(), return_when=asyncio.FIRST_COMPLETED)

                for task in done:
                    gen = task_map.pop(task)
                    try:
                        result = task.result()
                        result["index"] = rid_to_index[result["meta_info"]["id"]]
                        yield result
                        new_task = asyncio.create_task(gen.__anext__())
                        task_map[new_task] = gen
                    except StopAsyncIteration:
                        pass

    async def flush_cache(self) -> FlushCacheReqOutput:
        return (await self.flush_cache_communicator(FlushCacheReqInput()))[0]

    def _build_usage_dict(self, meta_info: dict) -> dict:
        from sglang.srt.entrypoints.openai.usage_processor import UsageProcessor

        enable_cache_report = self.server_args.enable_cache_report
        cached_tokens = meta_info.get("cached_tokens", 0)
        cached_details = UsageProcessor._details_if_cached(cached_tokens, enable_cache_report) if enable_cache_report else None
        usage_kwargs = {
            "prompt_tokens": meta_info.get("prompt_tokens", 0),
            "completion_tokens": meta_info.get("completion_tokens", 0),
            "cached_tokens": cached_details,
        }
        if "reasoning_tokens" in inspect.signature(UsageProcessor.calculate_token_usage).parameters:
            usage_kwargs["reasoning_tokens"] = meta_info.get("reasoning_tokens", 0)
        usage = UsageProcessor.calculate_token_usage(**usage_kwargs)
        return usage.model_dump(exclude={"prompt_tokens_details"} if not enable_cache_report else None)

    def _extract_usage_from_state(self, state: ReqState) -> dict:
        if state.last_meta_info:
            return self._build_usage_dict(state.last_meta_info)
        completion_tokens = len(state.output_ids)
        return self._build_usage_dict(
            {
                "prompt_tokens": 0,
                "completion_tokens": completion_tokens,
                "reasoning_tokens": 0,
                "cached_tokens": 0,
            }
        )

    async def abort_request(self, rid: str, abort_all: bool = False, wait_timeout: float = 30.0) -> Optional[Union[dict, list[dict]]]:
        ## Abort all requests if abort_all is True
        if abort_all:
            rids = list(self.rid_to_state.keys())
            logger.info("Abort all requests: %s", rids)
            results = []
            for req_rid in rids:
                usage = await self.abort_request(req_rid, abort_all=False, wait_timeout=wait_timeout)
                results.append({"rid": req_rid, "usage": usage})
            return results
        if rid not in self.rid_to_state:
            return None

        state = self.rid_to_state[rid]
        if state.finished:
            return self._extract_usage_from_state(state)

        if state.abort_future is None or state.abort_future.done():
            state.abort_future = asyncio.get_running_loop().create_future()

        self.send_to_scheduler.send_pyobj(AbortReq(rid))

        try:
            return await asyncio.wait_for(state.abort_future, timeout=wait_timeout)
        except asyncio.TimeoutError:
            logger.warning(
                "Abort request timed out for rid=%s, returning best-effort usage",
                rid,
            )
            return self._extract_usage_from_state(state)
        finally:
            if rid in self.rid_to_state and (state.finished or not state.abort_future.done()):
                del self.rid_to_state[rid]

    async def update_weights_from_disk(
        self,
        obj: UpdateWeightFromDiskReqInput,
        request: Optional[fastapi.Request] = None,
    ) -> tuple[bool, str]:
        self.auto_create_handle_loop()

        # default the load format to the server_args
        if obj.load_format is None:
            obj.load_format = self.server_args.load_format
        logger.info("Start update_weights. Load format=%s", obj.load_format)

        if True:
            # Hold the lock if it is not async. This means that weight sync
            # cannot run while requests are in progress.
            async with self.model_update_lock.writer_lock:
                return await self._wait_for_model_update_from_disk(obj)

    async def _wait_for_model_update_from_disk(self, obj: UpdateWeightFromDiskReqInput) -> tuple[bool, str]:
        self.send_to_scheduler.send_pyobj(obj)
        self.model_update_result = asyncio.Future()
        if self.server_args.dp_size == 1:
            result = await self.model_update_result
            if result.success:
                self.served_model_name = obj.model_path
                self.server_args.model_path = obj.model_path
                self.server_args.load_format = obj.load_format
                self.model_path = obj.model_path
            return result.success, result.message, result.num_paused_requests
        else:  # self.server_args.dp_size > 1
            self.model_update_tmp = []
            result = await self.model_update_result

            all_success = all([r.success for r in result])
            if all_success is True:
                self.server_args.model_path = obj.model_path
                self.server_args.load_format = obj.load_format
                self.model_path = obj.model_path
            all_message = [r.message for r in result]
            all_message = " | ".join(all_message)
            all_paused_requests = [r.num_paused_requests for r in result]
            return all_success, all_message, all_paused_requests

    async def open_session(self, obj: OpenSessionReqInput, request: Optional[fastapi.Request] = None):
        self.auto_create_handle_loop()

        if obj.session_id is None:
            obj.session_id = uuid.uuid4().hex
        elif obj.session_id in self.session_futures:
            return None

        self.send_to_scheduler.send_pyobj(obj)

        self.session_futures[obj.session_id] = asyncio.Future()
        session_id = await self.session_futures[obj.session_id]
        del self.session_futures[obj.session_id]
        return session_id

    async def close_session(self, obj: CloseSessionReqInput, request: Optional[fastapi.Request] = None):
        await self.send_to_scheduler.send_pyobj(obj)

    async def watch_load_thread(self):
        # Only for dp_controller when dp_size > 1
        if self.server_args.dp_size == 1 or self.server_args.load_balance_method == "round_robin":
            return

        while True:
            await asyncio.sleep(self.server_args.load_watch_interval)
            loads = await self.get_load_communicator(GetLoadReqInput())
            load_udpate_req = WatchLoadUpdateReq(loads=loads)
            self.send_to_scheduler.send_pyobj(load_udpate_req)

    def get_log_request_metadata(self):
        max_length = None
        skip_names = None
        out_skip_names = None
        if self.log_requests:
            if self.log_requests_level == 0:
                max_length = 1 << 30
                skip_names = set(
                    [
                        "text",
                        "input_ids",
                        "input_embeds",
                        "image_data",
                        "audio_data",
                    ]
                )
                out_skip_names = set(
                    [
                        "text",
                        "output_ids",
                    ]
                )
            elif self.log_requests_level == 1:
                max_length = 2048
            elif self.log_requests_level == 2:
                max_length = 1 << 30
            else:
                raise ValueError(f"Invalid --log-requests-level: {self.log_requests_level=}")
        return max_length, skip_names, out_skip_names

    def configure_logging(self, obj: ConfigureLoggingReq):
        if obj.log_requests is not None:
            self.log_requests = obj.log_requests
        if obj.log_requests_level is not None:
            self.log_requests_level = obj.log_requests_level
        if obj.dump_requests_folder is not None:
            self.dump_requests_folder = obj.dump_requests_folder
        if obj.dump_requests_threshold is not None:
            self.dump_requests_threshold = obj.dump_requests_threshold
        logging.info(f"Config logging: {obj=}")
        self.log_request_metadata = self.get_log_request_metadata()

    def create_abort_task(self, obj: GenerateReqInput):
        # Abort the request if the client is disconnected.
        async def abort_request():
            await asyncio.sleep(1)
            if obj.is_single:
                await self.abort_request(obj.rid)
            else:
                for rid in obj.rid:
                    await self.abort_request(rid)

        background_tasks = BackgroundTasks()
        background_tasks.add_task(abort_request)
        return background_tasks

    def auto_create_handle_loop(self):
        if self.no_create_loop:
            return

        self.no_create_loop = True
        loop = asyncio.get_event_loop()
        self.asyncio_tasks.add(loop.create_task(print_exception_wrapper(self.handle_loop)))

        # We cannot add signal handler when the tokenizer manager is not in
        # the main thread due to the CPython limitation.
        if threading.current_thread() is threading.main_thread():
            signal_handler = SignalHandler(self)
            loop.add_signal_handler(signal.SIGTERM, signal_handler.signal_handler)
        else:
            logger.warning(
                "Signal handler is not added because the tokenizer manager is "
                "not in the main thread. This disables graceful shutdown of the "
                "tokenizer manager when SIGTERM is received."
            )
        self.asyncio_tasks.add(loop.create_task(print_exception_wrapper(self.sigterm_watchdog)))
        self.asyncio_tasks.add(loop.create_task(print_exception_wrapper(self.watch_load_thread)))

    async def sigterm_watchdog(self):
        while not self.gracefully_exit:
            await asyncio.sleep(5)

        # Drain requests
        while True:
            remain_num_req = len(self.rid_to_state)
            logger.info(f"Gracefully exiting... remaining number of requests {remain_num_req}")
            if remain_num_req > 0:
                await asyncio.sleep(5)
            else:
                break

        kill_process_tree(os.getpid(), include_parent=True)
        sys.exit(0)

    async def handle_loop(self):
        """The event loop that handles requests"""

        while True:
            recv_obj = await self.recv_from_detokenizer.recv_pyobj()
            dispatched = self._result_dispatcher(recv_obj)
            if inspect.isawaitable(dispatched):
                await dispatched
            self.last_receive_tstamp = time.time()

    async def _handle_batch_output(
        self,
        recv_obj: Union[BatchStrOut, BatchEmbeddingOut, BatchMultimodalOut, BatchTokenIDOut],
    ):
        for i, rid in enumerate(recv_obj.rids):
            state: ReqState = self.rid_to_state.get(rid, None)
            if state is None:
                logger.error(f"Received output for {rid=} but the state was deleted in TokenizerManager.")
                continue

            # Build meta_info and return value
            meta_info = {
                "id": rid,
                "finish_reason": recv_obj.finished_reasons[i],
                "prompt_tokens": recv_obj.prompt_tokens[i],
            }
            logprobs_info = state.logprobs_info if not state.obj.stream else {}

            if getattr(state.obj, "return_logprob", False):
                await self.convert_logprob_style(
                    logprobs_info,
                    state.obj.top_logprobs_num,
                    state.obj.token_ids_logprob,
                    state.obj.return_text_in_logprobs,
                    recv_obj,
                    i,
                )
                meta_info.update(logprobs_info)

            if not isinstance(recv_obj, BatchEmbeddingOut):
                meta_info.update(
                    {
                        "reasoning_tokens": recv_obj.reasoning_tokens[i],
                        "completion_tokens": recv_obj.completion_tokens[i],
                        "cached_tokens": recv_obj.cached_tokens[i],
                        "dequeue_time": recv_obj.dequeue_times[i],
                    }
                )

            if getattr(recv_obj, "output_hidden_states", None):
                meta_info["hidden_states"] = recv_obj.output_hidden_states[i]

            if isinstance(recv_obj, BatchStrOut):
                state.output_ids_are_incremental = bool(state.obj.stream)
                if len(recv_obj.batch_accept_draft_tokens) > 0:
                    meta_info.update({"accept_draft_tokens": recv_obj.batch_accept_draft_tokens[i]})
                state.text += recv_obj.output_strs[i]
                if state.obj.stream:
                    state.logprobs_info = logprobs_info
                    state.output_ids.extend(recv_obj.output_ids[i])
                    output_token_ids = state.output_ids[state.last_output_offset :]
                    state.last_output_offset = len(state.output_ids)
                else:
                    state.logprobs_info.update(logprobs_info)
                    state.output_ids.extend(recv_obj.output_ids[i])
                    output_token_ids = state.output_ids.copy()

                out_dict = {
                    "text": state.text,
                    "output_ids": output_token_ids,
                    "meta_info": meta_info,
                }
                if len(recv_obj.output_extra_infos):
                    out_dict["output_extra_info"] = recv_obj.output_extra_infos[i]
            elif isinstance(recv_obj, BatchTokenIDOut):
                state.output_ids_are_incremental = bool(
                    self.server_args.stream_output and state.obj.stream
                )
                output_multi_ids = None
                if state.output_ids_are_incremental:
                    state.output_ids.extend(recv_obj.output_ids[i])
                    output_token_ids = state.output_ids[state.last_output_offset :]
                    if recv_obj.output_multi_ids is not None:
                        output_multi_ids = recv_obj.output_multi_ids[i][state.last_output_offset :]
                    state.last_output_offset = len(state.output_ids)
                else:
                    state.output_ids.extend(recv_obj.output_ids[i])
                    output_token_ids = state.output_ids.copy()
                    if recv_obj.output_multi_ids is not None:
                        output_multi_ids = recv_obj.output_multi_ids[i]

                out_dict = {
                    "output_ids": output_token_ids,
                    "meta_info": meta_info,
                }
                if len(recv_obj.output_extra_infos):
                    out_dict["output_extra_info"] = recv_obj.output_extra_infos[i]
                if output_multi_ids is not None:
                    out_dict["output_multi_ids"] = output_multi_ids
            elif isinstance(recv_obj, BatchMultimodalOut):
                raise NotImplementedError()
            else:
                assert isinstance(recv_obj, BatchEmbeddingOut)
                out_dict = {
                    "embedding": recv_obj.embeddings[i],
                    "meta_info": meta_info,
                }

            state.finished = recv_obj.finished_reasons[i] is not None
            if state.finished:
                if self.server_args.speculative_algorithm:
                    meta_info["spec_verify_ct"] = recv_obj.spec_verify_ct[i]
                state.finished_time = time.time()
                meta_info["e2e_latency"] = state.finished_time - state.created_time

            state.last_meta_info = meta_info
            if state.abort_future is not None and not state.abort_future.done() and state.finished:
                state.abort_future.set_result(self._build_usage_dict(meta_info))

            state.out_list.append(out_dict)
            state.event.set()

            # Log metrics and dump
            if self.enable_metrics and state.obj.log_metrics:
                self.collect_metrics(state, recv_obj, i)
            if self.dump_requests_folder and state.finished and state.obj.log_metrics:
                self.dump_requests(state, out_dict)

    async def convert_logprob_style(
        self,
        logprobs_info: dict,
        top_logprobs_num: int,
        token_ids_logprob: list[int],
        return_text_in_logprobs: bool,
        recv_obj: BatchStrOut,
        recv_obj_index: int,
    ):
        input_token_logprobs = logprobs_info.get("input_token_logprobs", [])
        output_token_logprobs = logprobs_info.get("output_token_logprobs", [])

        input_token_logprobs.extend(
            await self.detokenize_logprob_tokens(
                recv_obj.input_token_logprobs_val[recv_obj_index],
                recv_obj.input_token_logprobs_idx[recv_obj_index],
                return_text_in_logprobs,
            )
        )

        output_token_logprobs.extend(
            await self.detokenize_logprob_tokens(
                recv_obj.output_token_logprobs_val[recv_obj_index],
                recv_obj.output_token_logprobs_idx[recv_obj_index],
                return_text_in_logprobs,
            )
        )

        logprobs_info["input_token_logprobs"] = input_token_logprobs
        logprobs_info["output_token_logprobs"] = output_token_logprobs

        if top_logprobs_num > 0:
            input_top_logprobs = logprobs_info.get("input_top_logprobs", [])
            output_top_logprobs = logprobs_info.get("output_top_logprobs", [])

            input_top_logprobs.extend(
                await self.detokenize_top_logprobs_tokens(
                    recv_obj.input_top_logprobs_val[recv_obj_index],
                    recv_obj.input_top_logprobs_idx[recv_obj_index],
                    return_text_in_logprobs,
                )
            )
            output_top_logprobs.extend(
                await self.detokenize_top_logprobs_tokens(
                    recv_obj.output_top_logprobs_val[recv_obj_index],
                    recv_obj.output_top_logprobs_idx[recv_obj_index],
                    return_text_in_logprobs,
                )
            )

            logprobs_info["input_top_logprobs"] = input_top_logprobs
            logprobs_info["output_top_logprobs"] = output_top_logprobs

        if token_ids_logprob is not None:
            input_token_ids_logprobs = logprobs_info.get("input_token_ids_logprobs", [])
            output_token_ids_logprobs = logprobs_info.get("output_token_ids_logprobs", [])

            input_token_ids_logprobs.extend(
                await self.detokenize_top_logprobs_tokens(
                    recv_obj.input_token_ids_logprobs_val[recv_obj_index],
                    recv_obj.input_token_ids_logprobs_idx[recv_obj_index],
                    return_text_in_logprobs,
                )
            )
            output_token_ids_logprobs.extend(
                await self.detokenize_top_logprobs_tokens(
                    recv_obj.output_token_ids_logprobs_val[recv_obj_index],
                    recv_obj.output_token_ids_logprobs_idx[recv_obj_index],
                    return_text_in_logprobs,
                )
            )

            logprobs_info["input_token_ids_logprobs"] = input_token_ids_logprobs
            logprobs_info["output_token_ids_logprobs"] = output_token_ids_logprobs

    async def detokenize_logprob_tokens(
        self,
        token_logprobs_val: list[float],
        token_logprobs_idx: list[int],
        decode_to_text: bool,
    ):
        if not decode_to_text:
            return [(logprob, token_id, None) for logprob, token_id in zip(token_logprobs_val, token_logprobs_idx)]
        else:
            token_texts = await self.run_tokenizer_task(
                tokenizer_batch_decode_task,
                [[token_id] for token_id in token_logprobs_idx],
            )
            return list(zip(token_logprobs_val, token_logprobs_idx, token_texts))

    async def detokenize_top_logprobs_tokens(
        self,
        token_logprobs_val: list[float],
        token_logprobs_idx: list[int],
        decode_to_text: bool,
    ):
        # TODO: The current implementation only batches the detokenization for top-k tokens per single position.
        # We should batch all top-k tokens in all positions.
        ret = [None] * len(token_logprobs_val)
        awaitables = []
        ret_idxs = []
        for i, (val, idx) in enumerate(zip(token_logprobs_val, token_logprobs_idx)):
            if val:
                awaitables.append(self.detokenize_logprob_tokens(val, idx, decode_to_text))
                ret_idxs.append(i)
        for i, tokens in zip(ret_idxs, await asyncio.gather(*awaitables)):
            ret[i] = tokens
        return ret

    def collect_metrics(self, state: ReqState, recv_obj: BatchStrOut, i: int):
        completion_tokens = recv_obj.completion_tokens[i] if getattr(recv_obj, "completion_tokens", None) else 0

        if state.first_token_time == 0.0 and (completion_tokens > 0 or isinstance(recv_obj, BatchEmbeddingOut)):
            state.first_token_time = state.last_time = time.time()
            state.last_pure_time = recv_obj.generated_time
            state.last_completion_tokens = completion_tokens
            state.first_completion_tokens = completion_tokens
            ttft_s = state.first_token_time - state.created_time
            cat_aliases = (f"{state.metric_request_type}.TTFT",) if state.metric_request_type is not None else ()
            self.metrics_collector.observe_time_to_first_token(ttft_s, cat_aliases=cat_aliases)
            l3_prefetch_e2e_s = recv_obj.l3_prefetch_e2e_s[i]
            l3_prefetch_tokens = recv_obj.l3_prefetch_tokens[i]
            if completion_tokens > 0 and ttft_s > 0.0:
                if l3_prefetch_e2e_s is not None and l3_prefetch_tokens is not None:
                    l3_prefetch_ttft_pct = 100.0 * l3_prefetch_e2e_s / ttft_s
                    logger.debug(
                        "[L3-TTFT-Metric] rid=%s tokens=%d l3_e2e_ms=%.1f ttft_ms=%.1f pct=%.2f",
                        recv_obj.rids[i],
                        l3_prefetch_tokens,
                        l3_prefetch_e2e_s * 1000.0,
                        ttft_s * 1000.0,
                        l3_prefetch_ttft_pct,
                    )
                self.metrics_collector.observe_l3_prefetch_ttft(
                    l3_prefetch_e2e_s,
                    l3_prefetch_tokens,
                    ttft_s,
                )
            if self.ttft_counter is not None:
                prompt_tokens = recv_obj.prompt_tokens[i] if getattr(recv_obj, "prompt_tokens", None) else 0
                counter_name = self.ttft_counter.counter_name(prompt_tokens)
                if counter_name is not None:
                    self.metrics_collector.observe_time_to_first_token(ttft_s, name=counter_name)
        elif state.first_token_time != 0.0:
            num_new_tokens = completion_tokens - state.last_completion_tokens
            if num_new_tokens:
                new_time = time.time()
                interval = new_time - state.last_time
                pure_interval = recv_obj.generated_time - state.last_pure_time
                cat_aliases = (f"{state.metric_request_type}.TPOT",) if state.metric_request_type is not None else ()
                pure_cat_aliases = (f"{state.metric_request_type}.pure_TPOT",) if state.metric_request_type is not None else ()
                self.metrics_collector.observe_inter_token_latency(
                    interval,
                    num_new_tokens,
                    cat_aliases=cat_aliases,
                )
                self.metrics_collector.observe_inter_token_latency(
                    pure_interval,
                    num_new_tokens,
                    name="pure_TPOT",
                    cat_aliases=pure_cat_aliases,
                )
                state.last_pure_time = recv_obj.generated_time
                state.last_time = new_time
                state.last_completion_tokens = completion_tokens

        if state.finished:
            reasoning_tokens = recv_obj.reasoning_tokens[i] if getattr(recv_obj, "reasoning_tokens", None) else 0
            prompt_token_cat_aliases = (f"{state.metric_request_type}.PromptTokens",) if state.metric_request_type is not None else ()
            self.metrics_collector.observe_one_finished_request(
                recv_obj.prompt_tokens[i],
                completion_tokens,
                state.finished_time - state.created_time,
                state.tokenized_time - state.created_time,
                reasoning_tokens=reasoning_tokens,
                prompt_token_cat_aliases=prompt_token_cat_aliases,
            )
            num_tpot_tokens = completion_tokens - state.first_completion_tokens
            if num_tpot_tokens > 0:
                self.metrics_collector.observe_inter_token_latency(
                    state.finished_time - state.first_token_time,
                    num_tpot_tokens,
                    name="e2e_TPOT",
                    cat_aliases=((f"{state.metric_request_type}.e2e_TPOT",) if state.metric_request_type is not None else ()),
                )
            if recv_obj.batch_accept_draft_tokens:
                accept_length = recv_obj.batch_accept_draft_tokens[i]
                if accept_length is not None:
                    self.metrics_collector.observe_accept_length(
                        accept_length,
                        cat_aliases=((f"{state.metric_request_type}.acceptLength",) if state.metric_request_type is not None else ()),
                    )

    def dump_requests(self, state: ReqState, out_dict: dict):
        self.dump_request_list.append((state.obj, out_dict, state.created_time, time.time()))

        if len(self.dump_request_list) >= self.dump_requests_threshold:
            filename = os.path.join(
                self.dump_requests_folder,
                datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".pkl",
            )
            logger.info(f"Dump {len(self.dump_request_list)} requests to {filename}")

            to_dump = self.dump_request_list
            self.dump_request_list = []

            def background_task():
                os.makedirs(self.dump_requests_folder, exist_ok=True)
                with open(filename, "wb") as f:
                    pickle.dump(to_dump, f)

            # Schedule the task to run in the background without awaiting it
            asyncio.create_task(asyncio.to_thread(background_task))

    def _handle_open_session_req_output(self, recv_obj):
        self.session_futures[recv_obj.session_id].set_result(recv_obj.session_id if recv_obj.success else None)

    def _handle_update_weights_from_disk_req_output(self, recv_obj):
        if self.server_args.dp_size == 1:
            self.model_update_result.set_result(recv_obj)
        else:  # self.server_args.dp_size > 1
            self.model_update_tmp.append(recv_obj)
            # set future if the all results are recevied
            if len(self.model_update_tmp) == self.server_args.dp_size:
                self.model_update_result.set_result(self.model_update_tmp)

    def query_actual_chat_template(self) -> Optional[str]:
        if self.tokenizer_executor is not None:
            return self.tokenizer_executor.submit(tokenizer_query_chat_template_task).result()
        elif self.tokenizer is not None:
            return tokenizer_query_chat_template_task(tokenizer=self.tokenizer)
        else:
            return None


async def print_exception_wrapper(func):
    """
    Sometimes an asyncio function does not print exception.
    We do another wrapper to handle the exception.
    """
    try:
        await func()
    except Exception:
        traceback = get_exception_traceback()
        logger.error(f"TokenizerManager hit an exception: {traceback}")
        kill_process_tree(os.getpid(), include_parent=True)
        sys.exit(1)


class SignalHandler:
    def __init__(self, tokenizer_manager):
        self.tokenizer_manager = tokenizer_manager

    def signal_handler(self, signum=None, frame=None):
        logger.warning(f"SIGTERM received. {signum=} {frame=}. Draining requests and shutting down...")
        self.tokenizer_manager.gracefully_exit = True


T = TypeVar("T")


class _Communicator(Generic[T]):
    """Note: The communicator now only run up to 1 in-flight request at any time."""

    def __init__(self, sender, fan_out: int):
        self._sender = sender
        self._fan_out = fan_out
        self._result_event: Optional[asyncio.Event] = None
        self._result_values: Optional[list[T]] = None
        self._ready_queue: deque[asyncio.Future] = deque()

    async def __call__(self, obj):
        ready_event = asyncio.Event()
        if self._result_event is not None or len(self._ready_queue) > 0:
            self._ready_queue.append(ready_event)
            await ready_event.wait()
            assert self._result_event is None
            assert self._result_values is None

        if obj:
            self._sender.send_pyobj(obj)

        self._result_event = asyncio.Event()
        self._result_values = []
        await self._result_event.wait()
        result_values = self._result_values
        self._result_event = self._result_values = None

        if len(self._ready_queue) > 0:
            self._ready_queue.popleft().set()

        return result_values

    def handle_recv(self, recv_obj: T):
        self._result_values.append(recv_obj)
        if len(self._result_values) == self._fan_out:
            self._result_event.set()
