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
"""A scheduler that manages a tensor parallel GPU worker."""

import faulthandler
import os
import signal
import sys
import threading
import time
import math
from collections import defaultdict, deque
from concurrent import futures
from dataclasses import dataclass
from http import HTTPStatus
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple, Union

import psutil
import setproctitle
import torch
import zmq

from sglang.srt.managers.scheduler_profiler_mixin import SchedulerProfilerMixin
from sglang.srt.managers.scheduler_stats_mixin import SchedulerStatsMixin
from sglang.srt.managers.scheduler_post_process_mixin import SchedulerPostProcessMixin
from sglang.srt.parser.reasoning_parser import ReasoningParser
from sglang.global_config import global_config
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.constrained.base_grammar_backend import create_grammar_backend
from sglang.srt.disaggregation.decode import (
    DecodePreallocQueue,
    DecodeTransferQueue,
    SchedulerDisaggregationDecodeMixin,
)
from sglang.srt.disaggregation.kv_events import KVEventBatch
from sglang.srt.disaggregation.prefill import (
    PrefillBootstrapQueue,
    SchedulerDisaggregationPrefillMixin,
)
from sglang.srt.disaggregation.utils import (
    DisaggregationMode,
    MetadataBuffers,
    ReqToMetadataIdxAllocator,
    StepCounter,
    TransferBackend,
    prepare_abort,
)
from sglang.srt.hf_transformers_utils import get_processor, get_tokenizer
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.expert_distribution import (
    get_global_expert_distribution_recorder,
)
from sglang.srt.managers.io_struct import (
    AbortReq,
    CloseSessionReqInput,
    ExpertDistributionReq,
    ExpertDistributionReqOutput,
    FlushCacheReqInput,
    FlushCacheReqOutput,
    GetInternalStateReq,
    GetInternalStateReqOutput,
    GetWeightsByNameReqInput,
    GetWeightsByNameReqOutput,
    InitWeightsUpdateGroupReqInput,
    InitWeightsUpdateGroupReqOutput,
    OpenSessionReqInput,
    OpenSessionReqOutput,
    ProfileReq,
    ReleaseMemoryOccupationReqInput,
    ReleaseMemoryOccupationReqOutput,
    ResumeMemoryOccupationReqInput,
    ResumeMemoryOccupationReqOutput,
    SetInternalStateReq,
    SetInternalStateReqOutput,
    TokenizedEmbeddingReqInput,
    TokenizedGenerateReqInput,
    UpdateWeightFromDiskReqInput,
    UpdateWeightFromDiskReqOutput,
    UpdateWeightsFromDistributedReqInput,
    UpdateWeightsFromDistributedReqOutput,
    UpdateWeightsFromTensorReqInput,
    UpdateWeightsFromTensorReqOutput,
    EmbeddingLookupReqInput,
    EmbeddingLookupReqOutput,
    GetLoadReqInput,
    GetLoadReqOutput,
    ClearHiCacheReqInput,
    ClearHiCacheReqOutput,
)
from sglang.srt.managers.req import (
    FINISH_ABORT,
    Req,
    RequestStage,
)
from sglang.srt.managers.schedule_batch import ScheduleBatch, collect_group_specs
from collections import defaultdict

from sglang.srt.managers.schedule_policy import (
    AddReqResult,
    PrefillAdder,
    SchedulePolicy,
)
from sglang.srt.managers.session_controller import Session
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.managers.tp_worker_overlap_thread import TpModelWorkerClient
from sglang.srt.managers.utils import validate_input_length
from sglang.srt.mem_cache.hiradix_cache import HiRadixCache
from sglang.srt.mem_cache.radix_cache import RadixCache, CacheInitParams
from sglang.srt.metrics.collector import SchedulerMetricsCollector, SchedulerStats
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.env import global_server_args_dict
from sglang.srt.speculative.eagle_utils import EagleDraftOutput
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.oe_utils import update_token_table
from sglang.srt.torch_memory_saver_adapter import TorchMemorySaverAdapter
from sglang.srt.utils import (
    broadcast_pyobj,
    configure_logger,
    get_bool_env_var,
    get_colorful_logger,
    get_int_env_var,
    get_str_env_var,
    get_zmq_socket,
    is_npu,
    pyspy_dump_schedulers,
    register_usr_signal,
    set_gpu_proc_affinity,
    set_random_seed,
    suppress_other_loggers,
)

if not is_npu():
    from sglang.srt.speculative.eagle_utils import EagleDraftOutput

from sglang.utils import TypeBasedDispatcher, get_exception_traceback

logger = get_colorful_logger(__name__)

# Test retract decode for debugging purposes
TEST_RETRACT = get_bool_env_var("SGLANG_TEST_RETRACT")
RECORD_STEP_TIME = get_bool_env_var("SGLANG_RECORD_STEP_TIME")


@dataclass
class GenerationBatchResult:
    logits_output: LogitsProcessorOutput
    next_token_ids: List[int]
    extend_input_len_per_req: List[int]
    extend_logprob_start_len_per_req: List[int]
    bid: int
    accept_lengths_cpu: List[int]
    next_token_multi_ids: Optional[List[int]] = None


@dataclass
class EmbeddingBatchResult:
    embeddings: Union[torch.Tensor, Dict[str, torch.Tensor]]
    bid: int


class Scheduler(
    SchedulerStatsMixin,
    SchedulerProfilerMixin,
    SchedulerPostProcessMixin,
    SchedulerDisaggregationDecodeMixin,
    SchedulerDisaggregationPrefillMixin,
):
    """A scheduler that manages a tensor parallel GPU worker."""

    def __init__(
        self,
        server_args: ServerArgs,
        port_args: PortArgs,
        gpu_id: int,
        attn_tp_rank: int,
        moe_ep_rank: int,
        dp_rank: Optional[int],
        global_rank: int,
    ):
        # Parse args
        self.server_args = server_args
        self.attn_tp_rank = attn_tp_rank
        self.attn_tp_size = server_args.attn_tp_size
        self.moe_ep_rank = moe_ep_rank
        self.global_rank = global_rank
        self.schedule_policy = server_args.schedule_policy
        self.enable_overlap = not server_args.disable_overlap_schedule
        self.skip_tokenizer_init = server_args.skip_tokenizer_init
        self.enable_metrics = server_args.enable_metrics
        self.stream_interval = server_args.stream_interval
        self.spec_algorithm = SpeculativeAlgorithm.from_string(
            server_args.speculative_algorithm
        )
        self.gpu_id = gpu_id
        self.enable_hierarchical_cache = server_args.enable_hierarchical_cache

        # KV Events configuration
        self.enable_kv_cache_events = bool(
            server_args.kv_events_config and attn_tp_rank == 0
        )
        self.attn_dp_rank = dp_rank if dp_rank is not None else 0
        if self.spec_algorithm.is_eagle():
            self.decode_mem_cache_buf_multiplier = (
                self.server_args.speculative_num_draft_tokens
                + (
                    self.server_args.speculative_eagle_topk
                    * self.server_args.speculative_num_steps
                )
            )
        elif self.spec_algorithm.is_PLD():
            # PLD only needs memory for speculative_num_draft_tokens
            self.decode_mem_cache_buf_multiplier = self.server_args.speculative_num_draft_tokens
        else:
            self.decode_mem_cache_buf_multiplier = 1
        self.stream_interval = server_args.stream_interval

        # Distributed rank info
        self.dp_size = server_args.dp_size
        self.dp_rank = dp_rank
        # world_size is currently just world_size
        self.world_size = server_args.world_size

        # Currently no pp, for non-attn parts, global rank is tp rank
        self.tp_rank = self.global_rank

        # Init inter-process communication
        context = zmq.Context(2)
        if self.attn_tp_rank == 0:
            self.recv_from_tokenizer = get_zmq_socket(
                context, zmq.PULL, port_args.scheduler_input_ipc_name, False
            )
            self.send_to_tokenizer = get_zmq_socket(
                context, zmq.PUSH, port_args.tokenizer_ipc_name, False
            )

            if server_args.skip_tokenizer_init:
                # Directly send to the TokenizerManager
                self.send_to_detokenizer = get_zmq_socket(
                    context, zmq.PUSH, port_args.tokenizer_ipc_name, False
                )
            else:
                # Send to the DetokenizerManager
                self.send_to_detokenizer = get_zmq_socket(
                    context, zmq.PUSH, port_args.detokenizer_ipc_name, False
                )
        else:
            self.recv_from_tokenizer = None
            self.send_to_tokenizer = SimpleNamespace(send_pyobj=lambda x: None)
            self.send_to_detokenizer = SimpleNamespace(send_pyobj=lambda x: None)

        # Init tokenizer
        self.model_config = ModelConfig(
            server_args.model_path,
            trust_remote_code=server_args.trust_remote_code,
            revision=server_args.revision,
            context_length=server_args.context_length,
            model_override_args=server_args.json_model_override_args,
            is_embedding=server_args.is_embedding,
            dtype=server_args.dtype,
            quantization=server_args.quantization,
            enable_tbo=server_args.enable_tbo,
            enable_sbo=server_args.enable_sbo,
            server_args=server_args,
        )
        self.is_generation = self.model_config.is_generation

        if server_args.skip_tokenizer_init:
            self.tokenizer = self.processor = None
        else:
            if self.model_config.is_multimodal:
                self.processor = get_processor(
                    server_args.tokenizer_path,
                    tokenizer_mode=server_args.tokenizer_mode,
                    trust_remote_code=server_args.trust_remote_code,
                    revision=server_args.revision,
                )
                self.tokenizer = self.processor.tokenizer
            else:
                self.tokenizer = get_tokenizer(
                    server_args.tokenizer_path,
                    tokenizer_mode=server_args.tokenizer_mode,
                    trust_remote_code=server_args.trust_remote_code,
                    revision=server_args.revision,
                )
            if server_args.think_end_token:
                think_end_token = server_args.think_end_token.encode("utf-8").decode(
                    "unicode_escape"
                )
                os.environ["TOKENIZERS_PARALLELISM"] = "false"
                self.tokenizer.think_end_id = self.tokenizer.encode(
                    think_end_token, add_special_tokens=False
                )

        # Set reasoning_parser and think_end_id if --reasoning_parser is enabled
        if self.server_args.reasoning_parser and self.tokenizer:
            reasoning_parser = ReasoningParser(
                model_type=self.server_args.reasoning_parser,
                stream_reasoning=False,
                force_reasoning=False
            )
            self.tokenizer.think_end_id = self.tokenizer.encode(
                reasoning_parser.detector.think_end_token, add_special_tokens=False
            )[0]

        # Check whether overlap can be enabled
        if not self.is_generation:
            self.enable_overlap = False
            logger.info("Overlap scheduler is disabled for embedding models.")

        if self.model_config.is_multimodal:
            self.enable_overlap = False
            logger.info("Overlap scheduler is disabled for multimodal models.")

        # Launch a tensor parallel worker
        if self.enable_overlap and self.server_args.speculative_algorithm is None:
            TpWorkerClass = TpModelWorkerClient
        else:
            TpWorkerClass = TpModelWorker

        self.tp_worker = TpWorkerClass(
            server_args=server_args,
            gpu_id=gpu_id,
            attn_tp_rank=attn_tp_rank,
            moe_ep_rank=moe_ep_rank,
            global_rank=global_rank,
            nccl_port=port_args.nccl_port,
        )

        # Launch a draft worker for speculative decoding
        if self.spec_algorithm.is_eagle():
            from sglang.srt.speculative.eagle_worker import EAGLEWorker
            from sglang.srt.speculative.eagle_worker_overlap import (
                EagleWorkerOverlapped,
            )

            if self.enable_overlap:
                WorkerClass = EagleWorkerOverlapped
            else:
                WorkerClass = EAGLEWorker

            self.draft_worker = WorkerClass(
                server_args=server_args,
                gpu_id=gpu_id,
                attn_tp_rank=attn_tp_rank,
                moe_ep_rank=moe_ep_rank,
                nccl_port=port_args.nccl_port,
                target_worker=self.tp_worker,
                global_rank=global_rank,
            )
            # Extra allocated token_slots, these slots might be overwritten in next forward due to rejection
            # This is an initial value, subsequent batches will keep a list, and update with accept length
            self.reserve_num_tokens = self.server_args.speculative_num_draft_tokens
        elif self.spec_algorithm.is_PLD():
            from sglang.srt.speculative.pld_worker import PLDWorker
            from sglang.srt.speculative.pld_worker_overlap import PLDWorkerOverlapped

            if self.enable_overlap:
                WorkerClass = PLDWorkerOverlapped
            else:
                WorkerClass = PLDWorker

            self.draft_worker = WorkerClass(
                gpu_id=gpu_id,
                attn_tp_rank=attn_tp_rank,
                global_rank=global_rank,
                server_args=server_args,
                nccl_port=port_args.nccl_port,
                target_worker=self.tp_worker,
                dp_rank=dp_rank,
            )
            self.reserve_num_tokens = self.server_args.speculative_num_draft_tokens
        else:
            self.draft_worker = None
            # When not using speculative inference, no extra allocation
            self.reserve_num_tokens = 0

        # Get token and memory info from the model worker
        (
            self.max_total_num_tokens,
            self.max_prefill_tokens,
            self.max_running_requests,
            self.max_req_len,
            self.max_req_input_len,
            self.random_seed,
            self.device,
            _,
            _,
            _,
        ) = self.tp_worker.get_worker_info()
        self.init_max_running_requests = self.max_running_requests
        self.tp_cpu_group = self.tp_worker.get_tp_cpu_group()
        self.tp_group = self.tp_worker.get_tp_group()
        self.attn_tp_cpu_group = self.tp_worker.get_attention_tp_cpu_group()
        self.attn_tp_group = self.tp_worker.get_attention_tp_group()
        self.pad_input_ids_func = self.tp_worker.get_pad_input_ids_func()
        set_random_seed(self.random_seed)

        # Print debug info
        logger.info(
            f"max_total_num_tokens={self.max_total_num_tokens}, "
            f"chunked_prefill_size={server_args.chunked_prefill_size}, "
            f"max_prefill_tokens={self.max_prefill_tokens}, "
            f"max_running_requests={self.max_running_requests}, "
            f"context_len={self.model_config.context_len}"
        )

        # Init npu profiling if activate
        self._init_npu_profiling()

        # Init memory pool and cache
        self.req_to_token_pool, self.token_to_kv_pool, self.kv_allocator = (
            self.tp_worker.get_memory_pool()
        )
        self.enable_hicache_storage = server_args.hicache_storage_backend is not None
        params = CacheInitParams(
            disable=server_args.disable_radix_cache,
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool=self.token_to_kv_pool,
            token_to_kv_pool_allocator=self.kv_allocator,
            page_size=self.server_args.page_size,
            tp_cache_group=(
                self.attn_tp_cpu_group
                if self.server_args.enable_dp_attention
                else self.tp_cpu_group
            ),
            eviction_policy=server_args.radix_eviction_policy,
            enable_metrics=self.enable_metrics,
            enable_kv_cache_events=self.enable_kv_cache_events,
        )
        if (
            server_args.chunked_prefill_size is not None
            and server_args.disable_radix_cache
        ):
            self.tree_cache = RadixCache(params=params)
        else:
            if self.enable_hierarchical_cache and not server_args.disable_radix_cache:

                self.tree_cache = HiRadixCache(params=params, server_args=server_args)
                self.tp_worker.register_hicache_layer_transfer_counter(
                    self.tree_cache.cache_controller.layer_done_counter
                )
                self.tree_cache.is_decode = self.server_args.disaggregation_mode == DisaggregationMode.DECODE
            else:
                self.tree_cache = RadixCache(params=params)
        self.max_total_page_num = self.kv_allocator.free_slots.shape[0]
        logger.debug(
            f"[Scheduler init] tree_cache={self.tree_cache}\n"
            f"max_total_page_num={self.max_total_page_num}"
        )

        self.policy = SchedulePolicy(self.schedule_policy, self.tree_cache, self.enable_hierarchical_cache)

        # Init running status
        self.waiting_queue: List[Req] = []
        # The running decoding batch for continuous batching
        self.running_batch: Optional[ScheduleBatch] = ScheduleBatch(reqs=[])
        # The current forward batch
        self.cur_batch: Optional[ScheduleBatch] = None
        # The current forward batch
        self.last_batch: Optional[ScheduleBatch] = None
        self.forward_ct = 0
        self.forward_ct_decode = 0
        self.num_generated_tokens = 0
        self.spec_num_total_accepted_tokens = 0
        self.spec_num_total_forward_ct = 0
        self.cum_spec_accept_length = 0
        self.cum_spec_accept_count = 0
        self.last_decode_stats_tic = time.time()
        self.return_health_check_ct = 0
        self.current_stream = torch.get_device_module(self.device).current_stream()
        if self.device == "cpu":
            self.current_stream.synchronize = lambda: None  # No-op for CPU
        self.forward_sleep_time = None

        # For metrics only.
        # The largest prefill length of a single request
        self._largest_prefill_len: int = 0
        # The largest context length (prefill + generation) of a single request
        self._largest_prefill_decode_len: int = 0
        self.last_gen_throughput: float = 0.0
        self.step_time_dict = defaultdict(list)  # Dict[batch size -> step time]
        self.total_retracted_reqs = 0
        self.num_retracted_reqs: int = 0
        self.num_paused_reqs: int = 0
        self.kv_transfer_speed_gb_s: float = 0.0
        self.kv_transfer_latency_ms: float = 0.0

        # Session info
        self.sessions: Dict[str, Session] = {}

        # Init chunked prefill
        self.chunked_prefill_size = server_args.chunked_prefill_size
        if self.chunked_prefill_size <= 0:  # -1 means disable
            self.chunked_prefill_size = None
        self.chunked_req = None
        self.is_mixed_chunk = (
            self.chunked_prefill_size is not None and server_args.enable_mixed_chunk
        )
        if self.server_args.speculative_algorithm is not None:
            assert not self.is_mixed_chunk, (
                "speculative decoding does not support mixed chunk for now"
            )

        # Init the grammar backend for constrained generation
        self.grammar_queue: List[Req] = []
        if not server_args.skip_tokenizer_init and server_args.grammar_backend is not None:
            self.grammar_backend = create_grammar_backend(
                server_args, self.tokenizer, self.model_config.vocab_size
            )
        else:
            self.grammar_backend = None

        # Init new token estimation
        assert server_args.schedule_conservativeness >= 0, (
            "Invalid schedule_conservativeness"
        )

        self.init_new_token_ratio = min(
            global_config.default_init_new_token_ratio
            * server_args.schedule_conservativeness,
            1.0,
        )
        self.min_new_token_ratio = min(
            self.init_new_token_ratio
            * global_config.default_min_new_token_ratio_factor,
            1.0,
        )
        self.new_token_ratio_decay = (
            self.init_new_token_ratio - self.min_new_token_ratio
        ) / global_config.default_new_token_ratio_decay_steps
        self.new_token_ratio = self.init_new_token_ratio

        # Tell whether the current running batch is full so that we can skip
        # the check of whether to prefill new requests.
        # This is an optimization to reduce the overhead of the prefill check.
        self.batch_is_full = False

        # Init watchdog thread
        self.watchdog_timeout = server_args.watchdog_timeout
        self.parent_process = psutil.Process().parent()
        t = threading.Thread(target=self.watchdog_thread, daemon=True)
        t.start()

        # Init memory saver
        self.memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=server_args.enable_memory_saver
        )

        # Init profiler
        self.init_profier()

        # Init metrics stats
        self.stats = SchedulerStats()
        if self.enable_metrics:
            self.metrics_collector = SchedulerMetricsCollector(
                labels={
                    "model_name": self.server_args.served_model_name,
                    "process_name": f"TP{self.tp_rank}-DP{self.dp_rank}",
                    "app_key": self.server_args.app_key,
                },
                metrics_reporters=server_args.metrics_reporters,
            )

        # Init KV Events for Dynamo compatibility
        self.init_kv_events(server_args.kv_events_config)

        # Init tracing for Dynamo compatibility
        if server_args.enable_trace:
            try:
                from sglang.srt.tracing.trace import process_tracing_init
                process_tracing_init(server_args.otlp_traces_endpoint, "sglang")
            except ImportError:
                logger.warning("Failed to import process_tracing_init for tracing")

        # Init request dispatcher
        self._request_dispatcher = TypeBasedDispatcher(
            [
                (TokenizedGenerateReqInput, self.handle_generate_request),
                (TokenizedEmbeddingReqInput, self.handle_embedding_request),
                (FlushCacheReqInput, self.flush_cache_wrapped),
                (ClearHiCacheReqInput, self.clear_hicache_storage_wrapped),
                (AbortReq, self.abort_request),
                (EmbeddingLookupReqInput, self.handle_embedding_lookup),
                (OpenSessionReqInput, self.open_session),
                (CloseSessionReqInput, self.close_session),
                (UpdateWeightFromDiskReqInput, self.update_weights_from_disk),
                (InitWeightsUpdateGroupReqInput, self.init_weights_update_group),
                (
                    UpdateWeightsFromDistributedReqInput,
                    self.update_weights_from_distributed,
                ),
                (UpdateWeightsFromTensorReqInput, self.update_weights_from_tensor),
                (GetWeightsByNameReqInput, self.get_weights_by_name),
                (ReleaseMemoryOccupationReqInput, self.release_memory_occupation),
                (ResumeMemoryOccupationReqInput, self.resume_memory_occupation),
                (ProfileReq, self.profile),
                (GetInternalStateReq, self.get_internal_state),
                (SetInternalStateReq, self.set_internal_state),
                (ExpertDistributionReq, self.expert_distribution_handle),
                (GetLoadReqInput, self.get_load),
            ]
        )

        self.disaggregation_mode = DisaggregationMode(
            self.server_args.disaggregation_mode
        )
        self.init_disaggregation()
        # Determine if over_embedding is used based on ckpt, if so need to pay attention to n value, need to supplement extra token info during scheduling
        if self.tp_worker.model_config.use_over_embedding:
            self.use_over_embedding = True
            self.token_table = self.tp_worker.model_runner.oe_token_table
            self.over_embedding_n = self.tp_worker.model_config.hf_config.oe_neighbor_num
            self.over_embedding_k = self.tp_worker.model_config.hf_config.oe_split_num
        else:
            self.use_over_embedding = False

    def init_kv_events(self, kv_events_config: Optional[str]):
        """Initialize KV event publisher for Dynamo compatibility."""
        self.kv_event_publisher = None  # Initialize to None by default
        if self.enable_kv_cache_events:
            try:
                from sglang.srt.disaggregation.kv_events import EventPublisherFactory
                self.kv_event_publisher = EventPublisherFactory.create(
                    kv_events_config, self.attn_dp_rank
                )
            except ImportError:
                logger.warning("Failed to import EventPublisherFactory for KV events")
                self.kv_event_publisher = None

    def _publish_kv_events(self):
        if not self.enable_kv_cache_events or not self.kv_event_publisher:
            return

        events = self.tree_cache.take_events()
        if events:
            batch = KVEventBatch(ts=time.time(), events=events)
            self.kv_event_publisher.publish(batch)

    def _init_npu_profiling(self):
        npu_profiling_save_path = get_str_env_var("NPU_PROF_SAVE_DIR", "")
        if is_npu() and npu_profiling_save_path:
            logger.info("NPU Profiling is activated.")
            import torch_npu

            experimental_config = torch_npu.profiler._ExperimentalConfig(
                profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
                aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
                l2_cache=False,
                data_simplification=False,
            )
            self.npu_custom_profiler = torch_npu.profiler.profile(
                activities=[
                    torch_npu.profiler.ProfilerActivity.NPU,
                    torch_npu.profiler.ProfilerActivity.CPU,
                ],
                with_stack=False,  # Whether to record the call stack
                record_shapes=True,  # Whether to record operator shapes
                profile_memory=False,  # Whether to profile GPU memory
                experimental_config=experimental_config,
                schedule=torch_npu.profiler.schedule(
                    wait=get_int_env_var("NPU_PROF_WAIT_TIME", 10),  # Number of steps to skip
                    warmup=get_int_env_var(
                        "NPU_PROF_WARMUP_TIME", 1
                    ),  # Number of steps for warmup after wait
                    active=get_int_env_var(
                        "NPU_PROF_ACTIVE_TIME", 5
                    ),  # Number of steps to profile after warmup
                    repeat=get_int_env_var(
                        "NPU_PROF_REPEAT_TIME", 1
                    ),  # How many times to repeat the above process
                ),
                # Save path for profile log, must use tensorboard_trace_handler for viewing with tb
                on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
                    npu_profiling_save_path
                ),
            )
        else:
            self.npu_custom_profiler = None

    def init_disaggregation(self):
        self.transfer_backend = TransferBackend(
            self.server_args.disaggregation_transfer_backend
        )
        self.enable_layerwise_transfer = False

        if (
            self.disaggregation_mode == DisaggregationMode.DECODE
        ):  # *2 for the headroom.
            buffer_size = (self.req_to_token_pool.size) * 2
            req_to_metadata_buffer_idx_allocator = ReqToMetadataIdxAllocator(
                buffer_size
            )
            self.disagg_metadata_buffers = MetadataBuffers(buffer_size,
                                            transfer_hidden_states_max_size=self.server_args.disaggregation_transfer_hidden_states_max_size,
                                            hidden_states_dim=self.model_config.hidden_size,
                                            hidden_states_dtype=self.model_config.dtype,
                                        )
            # The decode requests polling kv cache
            self.disagg_decode_transfer_queue = DecodeTransferQueue(
                gloo_group=self.attn_tp_cpu_group,
                req_to_metadata_buffer_idx_allocator=req_to_metadata_buffer_idx_allocator,
                metadata_buffers=self.disagg_metadata_buffers,
                scheduler=self,
                tree_cache=self.tree_cache,
            )

            # The decode requests pending for pre-allocation
            self.disagg_decode_prealloc_queue = DecodePreallocQueue(
                req_to_token_pool=self.req_to_token_pool,
                token_to_kv_pool=self.token_to_kv_pool,
                kv_allocator=self.kv_allocator,
                draft_token_to_kv_pool=(
                    None
                    if self.draft_worker is None or self.spec_algorithm.is_PLD()
                    else self.draft_worker.model_runner.token_to_kv_pool
                ),
                req_to_metadata_buffer_idx_allocator=req_to_metadata_buffer_idx_allocator,
                metadata_buffers=self.disagg_metadata_buffers,
                scheduler=self,
                transfer_queue=self.disagg_decode_transfer_queue,
                tree_cache=self.tree_cache,
                gloo_group=self.attn_tp_cpu_group,
                tp_rank=self.tp_rank,
                world_size=self.world_size,
                bootstrap_port=self.server_args.disaggregation_bootstrap_port,
                max_total_num_tokens=self.max_total_num_tokens,
                transfer_backend=self.transfer_backend,
            )

            # Metric for pre-allocation
            self.num_tokens_pre_allocated = 0

        elif self.disaggregation_mode == DisaggregationMode.PREFILL:
            # *2 for the headroom.
            buffer_size = self.max_running_requests * 2
            req_to_metadata_buffer_idx_allocator = ReqToMetadataIdxAllocator(
                buffer_size
            )
            self.enable_layerwise_transfer = (
                self.transfer_backend == TransferBackend.MOONCAKE_ASYNC
            )
            self.disagg_metadata_buffers = MetadataBuffers(
                buffer_size,
                device=(
                    f"cuda:{self.gpu_id}" if self.enable_layerwise_transfer else "cpu"
                ),
                transfer_hidden_states_max_size=self.server_args.disaggregation_transfer_hidden_states_max_size,
                hidden_states_dim=self.model_config.hidden_size,
                hidden_states_dtype=self.model_config.dtype,
            )

            self.disagg_prefill_bootstrap_queue = PrefillBootstrapQueue(
                token_to_kv_pool=self.token_to_kv_pool,
                kv_allocator=self.kv_allocator,
                draft_token_to_kv_pool=(
                    None
                    if self.draft_worker is None or self.spec_algorithm.is_PLD()
                    else self.draft_worker.model_runner.token_to_kv_pool
                ),
                req_to_metadata_buffer_idx_allocator=req_to_metadata_buffer_idx_allocator,
                metadata_buffers=self.disagg_metadata_buffers,
                tp_rank=self.tp_rank,
                world_size=self.world_size,
                bootstrap_port=self.server_args.disaggregation_bootstrap_port,
                gloo_group=self.attn_tp_cpu_group,
                transfer_backend=self.transfer_backend,
                scheduler=self,
            )

            # register step_counter for async mode
            if self.enable_layerwise_transfer:
                self.step_counter = StepCounter(gpu_id=self.gpu_id)
                self.kv_transfer_manager = (
                    self.disagg_prefill_bootstrap_queue.kv_manager
                )
                self.kv_transfer_manager.register_step_counter(self.step_counter)
                self.tp_worker.model_runner.attn_backend.register_step_counter(
                    self.step_counter
                )
                if not self.spec_algorithm.is_none() and not self.spec_algorithm.is_PLD():
                    self.draft_worker.model_runner.attn_backend.register_step_counter(
                        self.step_counter
                    )
            logger.info(f"enable_layerwise_transfer: {self.enable_layerwise_transfer}")

            # The prefill requests that are in the middle of kv sending
            self.disagg_prefill_inflight_queue: List[Req] = []

    def set_next_batch_sampling_info_done(self, batch: ScheduleBatch):
        if batch.next_batch_sampling_info:
            if batch.next_batch_sampling_info.grammars is not None:
                batch.next_batch_sampling_info.update_regex_vocab_mask()
                self.current_stream.synchronize()
            batch.next_batch_sampling_info.sampling_info_done.set()

    def watchdog_thread(self):
        """A watch dog thread that will try to kill the server itself if one forward batch takes too long."""
        self.watchdog_last_forward_ct = 0
        self.watchdog_last_time = time.time()

        while True:
            current = time.time()
            if self.cur_batch is not None:
                if self.watchdog_last_forward_ct == self.forward_ct:
                    if current > self.watchdog_last_time + self.watchdog_timeout:
                        logger.error(f"Watchdog timeout ({self.watchdog_timeout=})")
                        break
                else:
                    self.watchdog_last_forward_ct = self.forward_ct
                    self.watchdog_last_time = current
            time.sleep(self.watchdog_timeout // 2)

        # Print batch size and memory pool info to check whether there are de-sync issues.
        logger.error(
            f"{self.cur_batch.batch_size()=}, "
            f"{self.cur_batch.reqs=}, "
            f"{self.kv_allocator.available_size()=}, "
            f"{self.tree_cache.evictable_size()=}, "
        )
        # Wait for some time so that the parent process can print the error.
        pyspy_dump_schedulers()
        print(file=sys.stderr, flush=True)
        print(file=sys.stdout, flush=True)
        time.sleep(5)
        self.parent_process.send_signal(signal.SIGUSR1)

    @torch.no_grad()
    def event_loop_normal(self):
        """A normal scheduler loop."""

        def _loop_process(step_func=None):
            while True:
                recv_reqs = self.recv_requests()
                self.process_input_requests(recv_reqs)

                batch = self.get_next_batch_to_run()
                self.cur_batch = batch

                if batch:
                    result = self.run_batch(batch)
                    self.process_batch_result(batch, result)
                    if step_func:
                        step_func()  # p.step()
                else:
                    # When the server is idle, so self-check and re-init some states
                    self.check_memory()
                    self.new_token_ratio = self.init_new_token_ratio

                if batch is None:
                    self.log_idle_stats()

                self.last_batch = batch

        if self.npu_custom_profiler:
            with self.npu_custom_profiler as p:
                p.start()
                _loop_process(p.step)  # do npu process
                p.stop()
        else:
            _loop_process()

    @torch.no_grad()
    def event_loop_overlap(self):
        """A scheduler loop that overlaps the CPU processing and GPU computation."""

        def _loop_process(step_func=None):
            self.result_queue = deque()

            while True:
                # this thread not does actual kernel launch
                recv_reqs = self.recv_requests()
                self.process_input_requests(recv_reqs)

                batch = self.get_next_batch_to_run()
                self.cur_batch = batch

                if batch:
                    batch.launch_done = threading.Event()
                    # put batch into input_queue
                    result = self.run_batch(batch)
                    self.result_queue.append((batch.copy(), result))

                    if self.last_batch is None:
                        # Create a dummy first batch to start the pipeline for overlap schedule.
                        # It is now used for triggering the sampling_info_done event.
                        tmp_batch = ScheduleBatch(
                            reqs=None,
                            forward_mode=ForwardMode.DUMMY_FIRST,
                            next_batch_sampling_info=(
                                self.tp_worker.cur_sampling_info
                                if self.draft_worker is None
                                else self.draft_worker.cur_sampling_info
                            ),
                        )
                        self.process_batch_result(tmp_batch, None, batch.launch_done)
                        if step_func:
                            step_func()  # p.step()

                if self.last_batch:
                    # Process the results of the last batch
                    tmp_batch, tmp_result = self.result_queue.popleft()
                    if self.draft_worker is None:
                        tmp_batch.next_batch_sampling_info = (
                            self.tp_worker.cur_sampling_info if batch else None
                        )
                    else:
                        tmp_batch.next_batch_sampling_info = (
                            self.draft_worker.cur_sampling_info if batch else None
                        )
                    self.process_batch_result(
                        tmp_batch, tmp_result, batch.launch_done if batch else None
                    )
                    if step_func:
                        step_func()  # p.step()
                elif batch is None:
                    # When the server is idle, so self-check and re-init some states
                    self.check_memory()
                    self.log_idle_stats()
                    self.new_token_ratio = self.init_new_token_ratio

                self.last_batch = batch

        if self.npu_custom_profiler:
            with self.npu_custom_profiler as p:
                p.start()
                _loop_process(p.step)  # do npu process
                p.stop()
        else:
            _loop_process()

    def recv_requests(self) -> List[Req]:
        """Receive results at attn_tp_rank = 0 and broadcast it to all other TP ranks."""
        if self.attn_tp_rank == 0:
            recv_reqs = []

            while True:
                try:
                    recv_req = self.recv_from_tokenizer.recv_pyobj(zmq.NOBLOCK)
                except zmq.ZMQError:
                    break
                recv_reqs.append(recv_req)
        else:
            recv_reqs = None

        if self.server_args.enable_dp_attention:
            if self.attn_tp_rank == 0:
                work_reqs = [
                    req
                    for req in recv_reqs
                    if isinstance(
                        req, (TokenizedGenerateReqInput, TokenizedEmbeddingReqInput)
                    )
                ]
                control_reqs = [
                    req
                    for req in recv_reqs
                    if not isinstance(
                        req, (TokenizedGenerateReqInput, TokenizedEmbeddingReqInput)
                    )
                ]
            else:
                work_reqs = None
                control_reqs = None

            if self.attn_tp_size != 1:
                work_reqs = broadcast_pyobj(
                    work_reqs,
                    self.attn_tp_group.rank,
                    self.attn_tp_cpu_group,
                    src=self.attn_tp_group.ranks[0],
                )
            if self.world_size != 1:
                control_reqs = broadcast_pyobj(
                    control_reqs, self.tp_group.rank, self.tp_cpu_group, src=self.tp_group.ranks[0]
                )
            recv_reqs = work_reqs + control_reqs
        elif self.world_size != 1:
            recv_reqs = broadcast_pyobj(recv_reqs, self.tp_group.rank, self.tp_cpu_group, src=self.tp_group.ranks[0])
        return recv_reqs

    def process_input_requests(self, recv_reqs: List):
        for recv_req in recv_reqs:
            # If it is a health check generation request and there are running requests, ignore it.
            # In disaggregation prefill mode, we should not ignore health check requests based on running_batch
            # because prefill server doesn't maintain a running_batch in the same way as normal/decode mode
            should_ignore_health_check = (
                is_health_check_generate_req(recv_req) and
                (self.chunked_req is not None or
                 (self.running_batch is not None and self.disaggregation_mode != DisaggregationMode.PREFILL))
            )

            if should_ignore_health_check:
                if self.global_rank == 0:
                    logger.debug(f"[SCHEDULER] Ignoring health check request because chunked_req={self.chunked_req is not None}, running_batch={self.running_batch is not None}")
                self.return_health_check_ct += 1
                continue

            output = self._request_dispatcher(recv_req)
            if output is not None:
                self.send_to_tokenizer.send_pyobj(output)
    
    def handle_embedding_lookup(
        self,
        recv_req: EmbeddingLookupReqInput,
    ):
        output_dict = self.tp_worker.model_runner.embedding_lookup(recv_req.rid, recv_req.input_ids_list, recv_req.aux_info)
        # if self.attn_tp_rank == 0:
        res = EmbeddingLookupReqOutput(rid=recv_req.rid, output_dict=output_dict)
        self.send_to_tokenizer.send_pyobj(res)

    def handle_generate_request(
        self,
        recv_req: TokenizedGenerateReqInput,
    ):
        # Create a new request

        self.tp_worker.model_runner.patch_req_info(recv_req)
        if (
            recv_req.session_params is None
            or recv_req.session_params.id is None
            or recv_req.session_params.id not in self.sessions
        ):
            if recv_req.input_embeds is not None:
                # Generate fake input_ids based on the length of input_embeds
                seq_length = len(recv_req.input_embeds)
                fake_input_ids = [999991] * seq_length
                recv_req.input_ids = fake_input_ids
                if "multi_ids" in global_server_args_dict["mm_mode"]:
                    fake_input_multi_ids = [[999992]] * seq_length
                    recv_req.input_multi_ids = fake_input_multi_ids
            if "multi_ids" in global_server_args_dict["mm_mode"] and recv_req.input_multi_ids is None:
                print(f"{recv_req.input_ids=}")
                seq_length = len(recv_req.input_ids)
                # TODO: 8 is the default multi_ids length
                fake_input_multi_ids = [[999992] * 8] * seq_length
                recv_req.input_multi_ids = fake_input_multi_ids

            # Handle custom logit processor passed to the request
            custom_logit_processor = recv_req.custom_logit_processor
            if (
                not self.server_args.enable_custom_logit_processor
                and custom_logit_processor is not None
            ):
                logger.warning(
                    "The SGLang server is not configured to enable custom logit processor."
                    "The custom logit processor passed in will be ignored."
                    "Please set --enable-custom-logits-processor to enable this feature."
                )
                custom_logit_processor = None

            if recv_req.bootstrap_port is None:
                # Use default bootstrap port
                recv_req.bootstrap_port = self.server_args.disaggregation_bootstrap_port

            req = Req(
                recv_req.rid,
                recv_req.input_text,
                recv_req.input_ids,
                recv_req.sampling_params,
                return_logprob=recv_req.return_logprob,
                top_logprobs_num=recv_req.top_logprobs_num,
                token_ids_logprob=recv_req.token_ids_logprob,
                stream=recv_req.stream,
                input_embeds=recv_req.input_embeds,
                input_extra_infos=recv_req.input_extra_infos,
                custom_logit_processor=custom_logit_processor,
                return_hidden_states=recv_req.return_hidden_states,
                eos_token_ids=self.model_config.hf_eos_token_id,
                bootstrap_host=recv_req.bootstrap_host,
                bootstrap_port=recv_req.bootstrap_port,
                bootstrap_room=recv_req.bootstrap_room,
                data_parallel_rank=recv_req.data_parallel_rank,
                origin_input_multi_ids=recv_req.input_multi_ids,
                metrics_collector=(
                    self.metrics_collector if self.enable_metrics else None
                ),
                created_time=recv_req.created_time,
            )
            req.set_tokenizer(self.tokenizer)

            if self.disaggregation_mode != DisaggregationMode.NULL:
                # Invalid request for disaggregated mode
                if recv_req.bootstrap_room is None:
                    error_message = (
                        f"Invalid request: Disaggregated request received without "
                        f"boostrap room id. {req.rid=}"
                    )
                    logger.error(error_message)
                    prepare_abort(req, error_message)
                    self.stream_output([req], req.return_logprob)
                    return
            if len(req.origin_input_ids) == 0:
                error_message = (
                    f"Invalid request: empty input_ids. {req.rid=} "
                )
                logger.error(error_message)
                prepare_abort(req, error_message)
                self.stream_output([req], req.return_logprob)
                return

            if (
                recv_req.session_params is not None
                and recv_req.session_params.id is not None
            ):
                req.finished_reason = FINISH_ABORT(
                    f"Invalid request: session id {recv_req.session_params.id} does not exist"
                )
                self._add_request_to_queue(req)
                return
        else:
            # Create a new request from a previous session
            session = self.sessions[recv_req.session_params.id]
            req = session.create_req(recv_req, self.tokenizer)
            if isinstance(req.finished_reason, FINISH_ABORT):
                self._add_request_to_queue(req)
                return

        # Validate prompts length
        error_msg = validate_input_length(
            req,
            self.max_req_input_len,
            self.server_args.allow_auto_truncate,
        )
        if error_msg:
            req.origin_input_ids = [0]
            req.sampling_params.max_new_tokens = 0
            self._add_request_to_queue(req)
            return

        # Copy more attributes
        if recv_req.logprob_start_len == -1 or not recv_req.return_logprob:
            # By default, only return the logprobs for output tokens
            req.logprob_start_len = len(req.origin_input_ids) - 1
        else:
            req.logprob_start_len = recv_req.logprob_start_len

        if req.logprob_start_len >= len(req.origin_input_ids):
            req.finished_reason = FINISH_ABORT(
                f"logprob_start_len, ({req.logprob_start_len}) is higher than the number of input tokens ({len(req.origin_input_ids)}). Request with a lower logprob_start_len.",
                HTTPStatus.BAD_REQUEST,
                "BadRequestError",
            )
            req.logprob_start_len = len(req.origin_input_ids) - 1
            self._add_request_to_queue(req)
            return

        req.sampling_params.max_new_tokens = min(
            (
                req.sampling_params.max_new_tokens
                if req.sampling_params.max_new_tokens is not None
                else 1 << 30
            ),
            self.max_req_len - len(req.origin_input_ids) - 1,
        )

        # Init grammar cache for this request
        add_to_grammar_queue = False
        if (
            req.sampling_params.json_schema is not None
            or req.sampling_params.regex is not None
            or req.sampling_params.ebnf is not None
            or req.sampling_params.structural_tag is not None
        ) and self.grammar_backend is not None:
            # assert self.grammar_backend is not None
            if req.sampling_params.json_schema is not None:
                key = ("json", req.sampling_params.json_schema)
            elif req.sampling_params.regex is not None:
                key = ("regex", req.sampling_params.regex)
            elif req.sampling_params.ebnf is not None:
                key = ("ebnf", req.sampling_params.ebnf)
            elif req.sampling_params.structural_tag:
                key = ("structural_tag", req.sampling_params.structural_tag)

            req.grammar = self.grammar_backend.get_cached_value(key)
            if not req.grammar:
                req.grammar = self.grammar_backend.get_future_value(key)
                add_to_grammar_queue = True

        if add_to_grammar_queue:
            self.grammar_queue.append(req)
        else:
            self._add_request_to_queue(req)

    def _prefetch_kvcache(self, req: Req):
        if self.enable_hicache_storage:
            req.init_next_round_input(self.tree_cache)
            logger.debug(f"Init next round input for {req.rid}. {req.last_node.backuped=}, {req.last_node.key=}")
            if req.last_node.backuped:
                # only to initiate the prefetch if the last node is backuped
                # otherwise, the allocated GPU memory must be locked for integrity
                last_hash = req.last_host_node.get_last_hash_value()
                matched_len = req.prefix_len + req.host_hit_length
                new_input_tokens = req.fill_ids[matched_len:]
                logger.debug(f"[_prefetch_kvcache] req={req.rid} matched_len={matched_len} fill_ids_len={len(req.fill_ids)} new_input_tokens_len={len(new_input_tokens)} prefix_indices={req.prefix_len} host_hit_length={req.host_hit_length}")

                prefix_keys = (
                    req.last_node.get_prefix_hash_values(req.last_node.parent)
                    if self.tree_cache.hicache_storage_pass_prefix_keys
                    else None
                )
                self.tree_cache.prefetch_from_storage(
                    req.rid,
                    req.last_host_node,
                    new_input_tokens,
                    last_hash,
                    prefix_keys,
                )

    def _add_request_to_queue(self, req: Req):
        req.queue_time_start = time.time()
        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            self._prefetch_kvcache(req)
            self.disagg_prefill_bootstrap_queue.add(req)
        elif self.disaggregation_mode == DisaggregationMode.DECODE:
            self.disagg_decode_prealloc_queue.add(req)
        else:
            self._prefetch_kvcache(req)
            self.waiting_queue.append(req)

    def _extend_requests_to_queue(self, reqs: List[Req], is_retracted: bool = False):
        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            self.disagg_prefill_bootstrap_queue.extend(reqs)
        elif self.disaggregation_mode == DisaggregationMode.DECODE:
            # If this is a decode server, we put the request to the decode pending prealloc queue
            self.disagg_decode_prealloc_queue.extend(reqs, is_retracted)
        else:
            self.waiting_queue.extend(reqs)

    def _get_num_used_pages(self):
        return self.max_total_page_num - (
            self.kv_allocator.available_size() + self.tree_cache.evictable_size()
        )

    def handle_embedding_request(
        self,
        recv_req: TokenizedEmbeddingReqInput,
    ):
        req = Req(
            recv_req.rid,
            recv_req.input_text,
            recv_req.input_ids,
            recv_req.sampling_params,
        )
        req.set_tokenizer(self.tokenizer)

        # Validate prompts length
        error_msg = validate_input_length(
            req,
            self.max_req_input_len,
            self.server_args.allow_auto_truncate,
        )
        if error_msg:
            self._add_request_to_queue(req)
            return

        # Copy more attributes
        req.logprob_start_len = len(req.origin_input_ids) - 1
        self._add_request_to_queue(req)


    def get_next_batch_to_run(self) -> Optional[ScheduleBatch]:
        # Merge the prefill batch into the running batch
        if self.last_batch and self.last_batch.forward_mode.is_extend():
            if self.chunked_req:
                # Move the chunked request out of the batch so that we can merge
                # only finished requests to running_batch.
                self.last_batch.filter_batch(chunked_req_to_exclude=self.chunked_req)
                self.tree_cache.cache_unfinished_req(self.chunked_req)
                self.batch_is_full = False
                # Note: Don't discard req_pool_idx after caching. Keep it without caching or discarding.
                # Cannot skip setting prefix_len and prefix_page_ids, they are used in `init_next_round_input_chunk` to calculate extend_len
                req = self.chunked_req
                req.prefix_len = self.req_to_token_pool.alloced_lens[req.req_pool_idx].item()
                page_size = self.kv_allocator.page_size
                req.prefix_page_ids = self.kv_allocator.req_to_page[
                    req.req_pool_idx, 0 : (req.prefix_len + page_size - 1) // page_size
                ]

            self.last_batch.filter_batch()
            if not self.last_batch.is_empty():
                if self.running_batch is None:
                    self.running_batch = self.last_batch
                else:
                    # merge running_batch with prefill batch
                    self.running_batch.merge_batch(self.last_batch)
            elif self.running_batch is None or self.running_batch.is_empty():
                # if max_new_token=1 in last batch(prefill), self.last_batch = empty, self.running_batch always is None
                # if self.running_batch is None, never update_running_batch to update batch_is_full -> hung prefill
                # should update batch_is_full here if no running_batch(decode) and no prefill batch
                self.batch_is_full = False

        new_batch = self.get_new_batch_prefill()
        if new_batch is not None:
            # Run prefill first if possible
            ret = new_batch
            ret.filter_batch()
        else:
            # Run decode
            if self.running_batch is None:
                ret = None
            else:
                self.running_batch = self.update_running_batch(self.running_batch, self.last_batch)
                ret = self.running_batch

        # Handle DP attention
        if self.server_args.enable_dp_attention:
            ret = self.prepare_dp_attn_batch(ret)
        self.update_oe_info(ret)
        return ret

    def update_oe_info(self, ret: ScheduleBatch):
        # Update over embedding info
        if ret is not None and self.use_over_embedding:
            ret.token_table = self.token_table
            self.init_token_table(ret)

    def init_token_table(self, schedule_batch: ScheduleBatch):
        '''
        Initialize requests from schedule_batch into token table when needed:
        1. In prefill case, initialization is needed
        2. For PD disaggregation decode nodes, initialization is needed on first scheduling
        3. For PD disaggregation decode nodes, when recovering retracted requests, initialization is needed
           (In non-PD case, retract recovery triggers re-prefill, but in PD case, kvcache is restored directly from CPU)
        Iterate through each request, merge all update cases, and update token table in one batch
        Currently only consider prefill
        '''
        tokens=[]
        column_starts=[]
        request_lengths=[]
        req_pool_indices=[]
        for req in schedule_batch.reqs:
            if not req.oe_init:
                fill_ids=req.origin_input_ids+req.output_ids
                tokens.extend(fill_ids)
                column_starts.append(0)
                req_pool_indices.append(req.req_pool_idx)
                request_lengths.append(len(fill_ids))
                req.oe_init=True

        if len(req_pool_indices) > 0:
            dtype=schedule_batch.token_table.dtype
            device=schedule_batch.token_table.device
            update_token_table(
                oe_token_table=schedule_batch.token_table,
                tokens=torch.tensor(tokens, dtype=dtype, device=device),
                row_indices=torch.tensor(req_pool_indices, dtype=torch.int64, device=device),
                column_starts=torch.tensor(column_starts, dtype=torch.int32, device=device),
                oe_req_lens=torch.tensor(request_lengths, dtype=torch.int32, device=device),
            )

    def _group_first_iterator(self, l):
        group_sizes = collect_group_specs(l)
        group_reqs = defaultdict(list)
        non_group_reqs = []

        for req in l:
            g_name, group_size, _ = req.get_group_specs()
            if g_name is not None:
                if g_name in group_sizes:
                    if group_sizes[g_name] != group_size:
                        logger.warning(
                            f"Group '{g_name}' has inconsistent group_size: "
                            f"expected {group_sizes[g_name]}, got {group_size}"
                        )
                group_reqs[g_name].append(req)
            else:
                non_group_reqs.append(req)

        for g_name, g_reqs in group_reqs.items():
            expected_size = group_sizes[g_name]
            if len(g_reqs) == expected_size:
                logger.info(f"_group_first_iterator: Group '{g_name}' is ready, yielding {len(g_reqs)} requests")
                yield from g_reqs
            else:
                logger.warning(f"_group_first_iterator: Group '{g_name}' is not ready, {g_reqs=}")

        yield from non_group_reqs

    def get_new_batch_prefill(self) -> Optional[ScheduleBatch]:
        # Check if the grammar is ready in the grammar queue
        if self.grammar_queue:
            self.move_ready_grammar_requests()

        # Handle the cases where prefill is not allowed
        if (
            self.batch_is_full or len(self.waiting_queue) == 0
        ) and self.chunked_req is None:
            return None

        running_bs = len(self.running_batch.reqs) if self.running_batch else 0
        if running_bs >= self.max_running_requests:
            self.batch_is_full = True
            return None

        if self.enable_hierarchical_cache:
            self.tree_cache.check_hicache_events()
        # Get priority queue
        prefix_computed = self.policy.calc_priority(self.waiting_queue)

        # Prefill policy
        adder = PrefillAdder(
            self.tree_cache,
            self.kv_allocator,
            self.running_batch,
            self.new_token_ratio,
            self.max_prefill_tokens,
            self.chunked_prefill_size,
            running_bs if self.is_mixed_chunk else 0,
        )

        is_chunked = self.chunked_req is not None
        if is_chunked:
            self.chunked_req.init_next_round_input_chunk()
            # Add chunk_req to batch here, return None if it's the last prefill for this req
            self.chunked_req = adder.add_chunked_req(self.chunked_req)


        dequeue_durations: List[float] = []
        # Get requests from the waiting queue to a new prefill batch

        for req in self._group_first_iterator(self.waiting_queue):
            if self.req_to_token_pool.available_size() <= 0:
                self.batch_is_full = True
                break

            if running_bs + len(adder.can_run_list) >= self.max_running_requests:
                self.batch_is_full = True
                break

            if len(adder.can_run_list) >= self.req_to_token_pool.available_size():
                self.batch_is_full = True
                break

            if self.disaggregation_mode == DisaggregationMode.PREFILL:
                # In prefill mode, prealloc queue and transfer queue can also take memory,
                # so we need to check if the available size for the actual available size.
                if len(adder.can_run_list) >= self.req_to_token_pool.available_size():
                    self.running_batch.batch_is_full = True
                    break

            if self.enable_hicache_storage:
                prefetch_done = self.tree_cache.check_prefetch_progress(req.rid)
                if not prefetch_done:
                    # skip staging requests that are ongoing prefetch
                    continue
            req.init_next_round_input(None if prefix_computed else self.tree_cache)

            res = adder.add_one_req(req, self.chunked_req)
            if res != AddReqResult.CONTINUE:
                if res == AddReqResult.NO_TOKEN:
                    if self.enable_hierarchical_cache:
                        # Set batch_is_full after making sure there are requests that can be served
                        self.batch_is_full = len(adder.can_run_list) > 0 or (
                            self.running_batch is not None
                            and not self.running_batch.is_empty()
                        )
                    else:
                        self.batch_is_full = True
                break

            dequeue_durations.append(time.time() - req.created_time)

        # Update waiting queue
        can_run_list: List[Req] = adder.can_run_list
        if len(can_run_list) == 0:
            return None

        if self.enable_metrics:
            # only record queue time when enable_metrics is True to avoid overhead
            for req in can_run_list:
                req.queue_time_end = time.perf_counter()
                if self.global_rank == 0 and not req.prefill_waiting_recorded:
                    req.add_latency(RequestStage.PREFILL_WAITING)
                    req.prefill_waiting_recorded = True

        self.waiting_queue = [
            x for x in self.waiting_queue if x not in set(can_run_list)
        ]

        if adder.new_chunked_req is not None:
            assert self.chunked_req is None
            self.chunked_req = adder.new_chunked_req

        if self.chunked_req:
            self.chunked_req.is_chunked += 1

        # Print stats
        if self.attn_tp_rank == 0:
            self.log_prefill_stats(adder, can_run_list, running_bs, dequeue_durations)

        # Create a new batch
        draft_token_num = (
            self.server_args.speculative_num_draft_tokens if self.draft_worker else 0
        )
        spec_num_steps = (
            self.server_args.speculative_num_steps if self.draft_worker else 0
        )
        new_batch = ScheduleBatch.init_new(
            can_run_list,
            self.req_to_token_pool,
            self.kv_allocator,
            self.token_to_kv_pool,
            self.tree_cache,
            self.model_config,
            self.enable_overlap,
            self.spec_algorithm,
            self.server_args.enable_custom_logit_processor,
            self.reserve_num_tokens,
            draft_token_num,
            spec_num_steps,
        )
        if self.enable_hierarchical_cache:
            # todo：disable cuda graph execution if hicache loading triggered
            new_batch.hicache_consumer_index = (
                self.tree_cache.ready_to_load_host_cache()
            )
        new_batch.prepare_for_extend()

        # Mixed-style chunked prefill
        if (
            self.is_mixed_chunk
            and self.running_batch is not None
            and not (new_batch.return_logprob or self.running_batch.return_logprob)
        ):
            # TODO (lianmin): support return_logprob + mixed chunked prefill
            self.running_batch.filter_batch()
            if not self.running_batch.is_empty():
                self.running_batch.prepare_for_decode()
                new_batch.mix_with_running(self.running_batch)
                new_batch.decoding_reqs = self.running_batch.reqs
            self.running_batch = None
        else:
            new_batch.decoding_reqs = None

        return new_batch

    def update_running_batch(self, batch: ScheduleBatch, last_batch: ScheduleBatch) -> Optional[ScheduleBatch]:
        """Update the current running decoding batch."""
        initial_bs = batch.batch_size()

        batch.filter_batch()
        if batch.is_empty():
            self.batch_is_full = False
            return None

        # Check if decode out of memory

        if not batch.check_decode_mem(self.decode_mem_cache_buf_multiplier) or (
            TEST_RETRACT and batch.batch_size() > 1
        ):
            old_ratio = self.new_token_ratio

            retracted_reqs, new_token_ratio = batch.retract_decode(self.server_args, self.enable_overlap)
            num_retracted_reqs = len(retracted_reqs)

            # Non-PD disaggregation overlap scheduling: if running Prefill batch but retract a req from Decode batch,
            # this req won't enter post-processing, and is not on GPU, should be released immediately.

            # In PD disaggregation: Prefill instance never retracts, but on Decode instance with overlap scheduling,
            # if retract happens when last batch is new_prebuilt_batch, then the last Decode Batch's post-processing
            # has ended and GPU is actually idle, result queue is empty, retracted req can't enter post-processing,
            # should be released immediately.

            if last_batch and last_batch.forward_mode == ForwardMode.EXTEND and self.enable_overlap:
                if self.disaggregation_mode == DisaggregationMode.DECODE:
                    for req in retracted_reqs:
                        logger.info(f"Release Retracted req {req.req_pool_idx}")
                        # Clear any existing tree node references and re-match prefix
                        if req.last_node is not None:
                            self.tree_cache.dec_lock_ref(req.last_node)
                            req.last_node = None
                        self.req_to_token_pool.free(req.req_pool_idx)
                        req.req_pool_idx = None
                else:
                    for req in retracted_reqs:
                        if req not in last_batch.reqs:
                            logger.info(f"Release Retracted req {req.req_pool_idx}")
                            if req.last_node is not None:
                                self.tree_cache.dec_lock_ref(req.last_node)
                                req.last_node = None
                            self.req_to_token_pool.free(req.req_pool_idx)
                            req.req_pool_idx = None

            self.new_token_ratio = new_token_ratio
            logger.info(
                "Decode out of memory happened. "
                f"#retracted_reqs: {len(retracted_reqs)}, "
                f"#new_token_ratio: {old_ratio:.4f} -> {self.new_token_ratio:.4f}"
            )
            self._extend_requests_to_queue(retracted_reqs, is_retracted=True)
            self.total_retracted_reqs += num_retracted_reqs
        else:
            self.new_token_ratio = max(
                self.new_token_ratio - self.new_token_ratio_decay,
                self.min_new_token_ratio,
            )

        if batch.batch_size() < initial_bs and self.req_to_token_pool.available_size() >= 0:
            self.batch_is_full = False

        # Update batch tensors
        batch.prepare_for_decode()
        return batch

    def run_batch(
        self, batch: ScheduleBatch
    ) -> Union[GenerationBatchResult, EmbeddingBatchResult]:
        """Run a batch."""
        self.forward_ct += 1

        # Whether to run the profiler
        self._profile_batch_predicate(batch)
        if self.forward_sleep_time is not None:
            logger.info(f"Scheduler.run_batch sleep {self.forward_sleep_time}s")
            time.sleep(self.forward_sleep_time)

        if self.is_generation:
            if self.spec_algorithm.is_none():
                model_worker_batch = batch.get_model_worker_batch()
                logits_output, next_token_ids, next_token_multi_ids, _ = self.tp_worker.forward_batch_generation(
                    model_worker_batch
                )
                if not self.enable_overlap:
                    self.req_to_token_pool.verified_lens[
                        model_worker_batch.req_pool_indices
                    ] += model_worker_batch.new_tokens_to_compute
            else:
                if batch.forward_mode.is_decode():
                    batch.forward_mode = ForwardMode.TARGET_VERIFY

                model_worker_batch = batch.get_model_worker_batch()
                (
                    logits_output,
                    next_token_ids,
                    accept_length,
                    new_verified_id,
                    token_list,
                ) = self.draft_worker.forward_batch_speculative_generation(
                    model_worker_batch,
                    launch_done=None,
                )
                next_token_multi_ids = None

                batch.spec_info = EagleDraftOutput(
                    last_verified_ids=new_verified_id,
                    token_list=token_list,
                )
                if not self.enable_overlap:
                    if accept_length is not None:
                        accept_lengths_cpu = accept_length.tolist()
                        batch.seq_lens.add_(accept_length)
                        batch.seq_lens_sum = sum(batch.seq_lens.tolist())
                    else:
                        accept_lengths_cpu = [1 for _ in range(len(batch.reqs))]
                else:
                    accept_lengths_cpu = None
                if batch.forward_mode.is_target_verify():
                    batch.forward_mode = ForwardMode.DECODE
            batch.output_ids = next_token_ids
            batch.output_multi_ids = next_token_multi_ids

            # These 2 values are needed for processing the output, but the values can be
            # modified by overlap schedule. So we have to copy them here so that
            # we can use the correct values in output processing.
            if batch.return_logprob:
                extend_input_len_per_req = [req.extend_input_len for req in batch.reqs]
                extend_logprob_start_len_per_req = [
                    req.extend_logprob_start_len for req in batch.reqs
                ]
            else:
                extend_input_len_per_req = None
                extend_logprob_start_len_per_req = None

            ret = GenerationBatchResult(
                logits_output=logits_output,
                next_token_ids=next_token_ids,
                extend_input_len_per_req=extend_input_len_per_req,
                extend_logprob_start_len_per_req=extend_logprob_start_len_per_req,
                bid=model_worker_batch.bid,
                accept_lengths_cpu=(
                    accept_lengths_cpu
                    if self.draft_worker
                    else [1 for _ in range(len(batch.reqs))]
                ),
                next_token_multi_ids=next_token_multi_ids,
            )
        else:  # embedding or reward model
            model_worker_batch = batch.get_model_worker_batch()
            embeddings = self.tp_worker.forward_batch_embedding(model_worker_batch)
            ret = EmbeddingBatchResult(
                embeddings=embeddings, bid=model_worker_batch.bid
            )
        return ret

    def prepare_dp_attn_batch(self, local_batch: ScheduleBatch):
        # Check if other DP workers have running batches
        if local_batch is None:
            num_tokens = 0
        elif local_batch.forward_mode.is_decode():
            if local_batch.spec_algorithm.is_none():
                num_tokens = local_batch.batch_size()
            else:
                # Current eagle decode starts from verify, num_tokens should be multiplied by candidate set size
                num_tokens = (
                    local_batch.batch_size()
                    * self.server_args.speculative_num_draft_tokens
                )
        else:
            num_tokens = local_batch.extend_num_tokens

        local_num_tokens = num_tokens
        local_batch_size = local_batch.batch_size() if local_batch is not None else 0
        local_forward_mode = (
            local_batch.forward_mode if local_batch is not None else ForwardMode.IDLE
        )

        local_communication_info = torch.tensor(
            [[local_num_tokens, local_batch_size, local_forward_mode]],
            dtype=torch.int32,
        )

        global_communication_info = torch.empty((self.world_size, 3), dtype=torch.int32)

        torch.distributed.all_gather_into_tensor(
            global_communication_info,
            local_communication_info,
            group=self.tp_cpu_group,
        )

        global_num_tokens = global_communication_info[:, 0].tolist()

        if local_batch is None and max(global_num_tokens) > 0:
            local_batch = self.get_idle_batch()

        if local_batch is not None:
            local_batch.global_num_tokens = global_num_tokens

            global_forward_mode = global_communication_info[:, 2].tolist()

            all_decode_or_idle = all(
                mode
                in (ForwardMode.DECODE, ForwardMode.IDLE, ForwardMode.TARGET_VERIFY)
                for mode in global_forward_mode
            )

            local_batch.all_decode_or_idle = all_decode_or_idle

            can_run_tbo = False
            global_batch_size = global_communication_info[:, 1].tolist()
            local_batch.global_batch_size = global_batch_size
            min_batch_size = min(global_batch_size)
            all_prefill = all(
                mode == ForwardMode.EXTEND for mode in global_forward_mode
            )
            if (
                self.server_args.enable_tbo
                and all_decode_or_idle
                and min_batch_size >= self.server_args.tbo_min_bs
            ):
                can_run_tbo = True
            elif self.server_args.enable_tbo and all_prefill and min_batch_size >= 2:
                can_run_tbo = True
            local_batch.can_run_tbo = can_run_tbo
        return local_batch

    def get_idle_batch(self):
        idle_batch = ScheduleBatch.init_new(
            [],
            self.req_to_token_pool,
            self.kv_allocator,
            self.token_to_kv_pool,
            self.tree_cache,
            self.model_config,
            self.enable_overlap,
            self.spec_algorithm,
            self.server_args.enable_custom_logit_processor,
            0,
        )
        idle_batch.prepare_for_idle()
        return idle_batch

    def move_ready_grammar_requests(self):
        """Move requests whose grammar objects are ready from grammar_queue to waiting_queue."""
        num_ready_reqs = 0
        for req in self.grammar_queue:
            try:
                req.grammar = req.grammar.result(timeout=0.05)
                num_ready_reqs += 1
            except futures._base.TimeoutError:
                break

        if self.server_args.enable_dp_attention:
            if self.attn_tp_size > 1:
                # Sync across attn TP ranks to make sure they have the same number of ready requests
                tensor = torch.tensor(num_ready_reqs, dtype=torch.int32)
                torch.distributed.all_reduce(
                    tensor,
                    op=torch.distributed.ReduceOp.MAX,
                    group=self.attn_tp_cpu_group,
                )
                num_ready_reqs_max = tensor.item()
                for i in range(num_ready_reqs, num_ready_reqs_max):
                    self.grammar_queue[i].grammar = self.grammar_queue[
                        i
                    ].grammar.result()
                num_ready_reqs = num_ready_reqs_max
        else:
            if self.world_size > 1:
                # Sync across TP ranks to make sure they have the same number of ready requests
                tensor = torch.tensor(num_ready_reqs, dtype=torch.int32)
                torch.distributed.all_reduce(
                    tensor, op=torch.distributed.ReduceOp.MAX, group=self.tp_cpu_group
                )
                num_ready_reqs_max = tensor.item()
                for i in range(num_ready_reqs, num_ready_reqs_max):
                    self.grammar_queue[i].grammar = self.grammar_queue[
                        i
                    ].grammar.result()
                num_ready_reqs = num_ready_reqs_max

        self._extend_requests_to_queue(self.grammar_queue[:num_ready_reqs])
        self.grammar_queue = self.grammar_queue[num_ready_reqs:]


    def clear_hicache_storage_wrapped(self, recv_req: ClearHiCacheReqInput):
        if self.enable_hierarchical_cache:
            self.tree_cache.clear_storage_backend()
            logger.info("Hierarchical cache cleared successfully!")
            if_success = True
        else:
            logger.warning("Hierarchical cache is not enabled.")
            if_success = False
        return ClearHiCacheReqOutput(success=if_success)

    def flush_cache_wrapped(self, recv_req: FlushCacheReqInput):
        success = self.flush_cache()
        return FlushCacheReqOutput(success=success)

    def flush_cache(self):
        """Flush the memory pool and cache."""
        if len(self.waiting_queue) == 0 and (
            self.running_batch is None or len(self.running_batch.reqs) == 0
        ):
            self.cur_batch = None
            self.last_batch = None
            self.tree_cache.reset()
            self.tree_cache_metrics = {"total": 0, "hit": 0}
            if self.grammar_backend:
                self.grammar_backend.reset()
            self.req_to_token_pool.clear()
            self.kv_allocator.clear()

            if not self.spec_algorithm.is_none():
                self.draft_worker.model_runner.req_to_token_pool.clear()
                self.draft_worker.model_runner.kv_allocator.clear()

            self.num_generated_tokens = 0
            self.forward_ct_decode = 0
            self.spec_num_total_accepted_tokens = 0
            self.spec_num_total_forward_ct = 0
            self.cum_spec_accept_length = 0
            self.cum_spec_accept_count = 0
            torch.cuda.empty_cache()
            logger.info("Cache flushed successfully!")
            if_success = True
        else:
            logger.warning(
                f"Cache not flushed because there are pending requests. "
                f"#queue-req: {len(self.waiting_queue)}, "
                f"#running-req: {0 if self.running_batch is None else len(self.running_batch.reqs)}"
            )
            if_success = False
        return if_success

    def get_load(self, recv_req: GetLoadReqInput = None) -> GetLoadReqOutput:
        # TODO(lsyin): use dynamically maintained num_waiting_tokens

        num_pages = self._get_num_used_pages()

        # Pages to use in waiting queue, bootstrap queue, prealloc queue
        def get_page_num(input_ids):
            return math.ceil(len(input_ids) / self.server_args.page_size)
        num_pages += sum(get_page_num(req.origin_input_ids) for req in self.waiting_queue)
        num_waiting_reqs = len(self.waiting_queue)
        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            num_pages += sum(
                get_page_num(req.origin_input_ids)
                for req in self.disagg_prefill_bootstrap_queue.queue
            )
            num_waiting_reqs += len(self.disagg_prefill_bootstrap_queue.queue)
        elif self.disaggregation_mode == DisaggregationMode.DECODE:
            num_pages += sum(
                get_page_num(req.req.origin_input_ids)
                for req in self.disagg_decode_prealloc_queue.queue
            )
            num_waiting_reqs += len(self.disagg_decode_prealloc_queue.queue)
        num_running_reqs = 0 if self.running_batch is None else len(self.running_batch.reqs)

        return GetLoadReqOutput(
            dp_rank=self.dp_rank,
            num_reqs=num_running_reqs + num_waiting_reqs,
            num_waiting_reqs=num_waiting_reqs,
            num_pages=num_pages,
        )

    def get_internal_state(self, recv_req: GetInternalStateReq):
        ret = dict(global_server_args_dict)
        ret["last_gen_throughput"] = self.last_gen_throughput
        if not self.spec_algorithm.is_none() and self.cum_spec_accept_count > 0:
            ret["avg_spec_accept_length"] = (
                self.cum_spec_accept_length / self.cum_spec_accept_count
            )

        if RECORD_STEP_TIME:
            ret["step_time_dict"] = self.step_time_dict
        return GetInternalStateReqOutput(
            internal_state=ret,
        )

    def set_internal_state(self, recv_req: SetInternalStateReq):
        server_args_dict = recv_req.server_args
        args_allow_update = set(
            [
                "speculative_accept_threshold_single",
                "speculative_accept_threshold_acc",
            ]
        )
        if_success = True
        for k, v in server_args_dict.items():
            if k not in args_allow_update:
                logger.warning(f"Updating {k} is not supported.")
                if_success = False
                break
        if if_success:
            if not self.spec_algorithm.is_none() and self.cum_spec_accept_count > 0:
                avg_spec_accept_length = (
                    self.cum_spec_accept_length / self.cum_spec_accept_count
                )
                logger.info(f"{avg_spec_accept_length=}")
            self.cum_spec_accept_length = self.cum_spec_accept_count = 0
            for k, v in server_args_dict.items():
                global_server_args_dict[k] = v
            logger.info(f"Global server args updated! {global_server_args_dict=}")
        return SetInternalStateReqOutput(
            updated=True,
            server_args=global_server_args_dict,
        )

    def abort_request(self, recv_req: AbortReq):
        # Delete requests in the waiting queue
        to_del = None
        for i, req in enumerate(self.waiting_queue):
            if req.rid == recv_req.rid:
                to_del = i
                if req.last_node is not None:
                    self.tree_cache.dec_lock_ref(req.last_node)
                    req.last_node = None
                    logger.debug(f"Cleaned last_node for aborted waiting req {req.rid}")
                break

        if to_del is not None:
            del self.waiting_queue[to_del]
            if self.enable_hicache_storage:
                # to release prefetch events associated with the request
                self.tree_cache.release_aborted_request(req.rid)
            logger.debug(f"Abort queued request. {req.rid=}")
            return

        # Delete requests in the running batch
        if self.running_batch:
            for req in self.running_batch.reqs:
                if req.rid == recv_req.rid and not req.finished():
                    logger.debug(f"Abort running request. {req.rid=}")
                    req.to_abort = True
                    if req.last_node is not None:
                        self.tree_cache.dec_lock_ref(req.last_node)
                        req.last_node = None
                        logger.debug(f"Force cleaned last_node for aborted req {req.rid}")
                    break

        if self.disaggregation_mode == DisaggregationMode.DECODE:
            for req in self.disagg_decode_prealloc_queue.queue:
                if req.req.rid == recv_req.rid:
                    req.req.to_abort = True
                    logger.debug(f"Abort disagg decode prealloc req. {req.req.rid=}")
                    return
            for req in self.disagg_decode_prealloc_queue.retracted_queue:
                if req.rid == recv_req.rid:
                    req.to_abort = True
                    logger.debug(f"Abort disagg decode retracted prealloc req. {req.req.rid=}")
                    return
            for req in self.disagg_decode_transfer_queue.queue:
                if req.req.rid == recv_req.rid:
                    req.req.to_abort = True
                    logger.debug(f"Abort disagg decode transfer req. {req.req.rid=}")
                    return
        elif self.disaggregation_mode == DisaggregationMode.PREFILL:
            for req in self.disagg_prefill_bootstrap_queue.queue:
                if req.rid == recv_req.rid:
                    req.to_abort = True
                    logger.debug(f"Abort disagg prefill bootstrap req. {req.rid=}")
                    return
            for req in self.disagg_prefill_inflight_queue:
                if req.rid == recv_req.rid:
                    req.to_abort = True
                    logger.debug(f"Abort disagg prefill inflight req. {req.rid=}")
                    return

    def update_weights_from_disk(self, recv_req: UpdateWeightFromDiskReqInput):
        """In-place update of the weights from disk."""
        success, message = self.tp_worker.update_weights_from_disk(recv_req)
        if success:
            flash_cache_success = self.flush_cache()
            assert flash_cache_success, "Cache flush failed after updating weights"
        else:
            logger.error(message)
        return UpdateWeightFromDiskReqOutput(success, message, 0)

    def init_weights_update_group(self, recv_req: InitWeightsUpdateGroupReqInput):
        """Initialize the online model parameter update group."""
        success, message = self.tp_worker.init_weights_update_group(recv_req)
        return InitWeightsUpdateGroupReqOutput(success, message)

    def update_weights_from_distributed(
        self,
        recv_req: UpdateWeightsFromDistributedReqInput,
    ) -> Tuple[bool, str]:
        """Update the online model parameter."""
        success, message = self.tp_worker.update_weights_from_distributed(recv_req)
        if success:
            flash_cache_success = self.flush_cache()
            assert flash_cache_success, "Cache flush failed after updating weights"
        else:
            logger.error(message)
        return UpdateWeightsFromDistributedReqOutput(success, message)

    def update_weights_from_tensor(self, recv_req: UpdateWeightsFromTensorReqInput):
        """Update the online model parameter from tensors."""
        success, message = self.tp_worker.update_weights_from_tensor(recv_req)
        # TODO extract common code b/t update_weights_from_distributed and update_weights_from_tensor later
        if success:
            if recv_req.flush_cache:
                flash_cache_success = self.flush_cache()
                assert flash_cache_success, "Cache flush failed after updating weights"
        else:
            logger.error(message)
        return UpdateWeightsFromTensorReqOutput(success, message)

    def get_weights_by_name(self, recv_req: GetWeightsByNameReqInput):
        parameter = self.tp_worker.get_weights_by_name(recv_req)
        return GetWeightsByNameReqOutput(parameter)

    def release_memory_occupation(self, recv_req: ReleaseMemoryOccupationReqInput):
        self.stashed_model_static_state = _export_static_state(
            self.tp_worker.worker.model_runner.model
        )
        self.memory_saver_adapter.pause()
        self.flush_cache()
        return ReleaseMemoryOccupationReqOutput()

    def resume_memory_occupation(self, recv_req: ResumeMemoryOccupationReqInput):
        self.memory_saver_adapter.resume()
        _import_static_state(
            self.tp_worker.worker.model_runner.model, self.stashed_model_static_state
        )
        del self.stashed_model_static_state
        return ResumeMemoryOccupationReqOutput()

    def expert_distribution_handle(self, recv_req: ExpertDistributionReq):
        if recv_req == ExpertDistributionReq.START_RECORD:
            get_global_expert_distribution_recorder().start_record()
        elif recv_req == ExpertDistributionReq.STOP_RECORD:
            get_global_expert_distribution_recorder().stop_record()
        elif recv_req == ExpertDistributionReq.DUMP_RECORD:
            get_global_expert_distribution_recorder().dump_record()
        else:
            raise ValueError("Unrecognized ExpertDistributionReq value")
        return ExpertDistributionReqOutput()


    def open_session(self, recv_req: OpenSessionReqInput):
        # handle error
        session_id = recv_req.session_id
        if session_id in self.sessions:
            logger.warning(f"session id {session_id} already exist, cannot open.")
            return OpenSessionReqOutput(session_id, False)
        elif session_id is None:
            logger.warning("session id is None, cannot open.")
            return OpenSessionReqOutput(session_id, False)
        else:
            self.sessions[session_id] = Session(
                recv_req.capacity_of_str_len, session_id
            )
            return OpenSessionReqOutput(session_id, True)

    def close_session(self, recv_req: CloseSessionReqInput):
        # handle error
        session_id = recv_req.session_id
        if session_id not in self.sessions:
            logger.warning(f"session id {session_id} does not exist, cannot delete.")
        else:
            del self.sessions[session_id]


def is_health_check_generate_req(recv_req):
    rid = getattr(recv_req, "rid", None)
    return rid is not None and rid.startswith("HEALTH_CHECK")


def _export_static_state(model):
    return dict(
        buffers=[
            (name, buffer.detach().clone()) for name, buffer in model.named_buffers()
        ]
    )


def _import_static_state(model, static_params):
    self_named_buffers = dict(model.named_buffers())
    for name, tensor in static_params["buffers"]:
        self_named_buffers[name][...] = tensor


def run_scheduler_process(
    server_args: ServerArgs,
    port_args: PortArgs,
    gpu_id: int,
    attn_tp_rank: int,
    moe_ep_rank: int,
    dp_rank: Optional[int],
    global_rank: int,
    pipe_writer,
):
    # Config the process
    # kill_itself_when_parent_died()  # This is disabled because it does not work for `--dp 2`
    setproctitle.setproctitle(f"sglang::scheduler_{dp_rank}")
    faulthandler.enable()
    parent_process = psutil.Process().parent()
    register_usr_signal()

    # [For Router] if env var "SGLANG_DP_RANK" exist, set dp_rank to the value of the env var
    if dp_rank is None and "SGLANG_DP_RANK" in os.environ:
        dp_rank = int(os.environ["SGLANG_DP_RANK"])

    # Configure the logger
    if dp_rank is None:
        prefix = f" ATTN TP RANK {attn_tp_rank}"
    else:
        prefix = f" DP_RANK {dp_rank} ATTN_TP_RANK {attn_tp_rank}"
    configure_logger(server_args, prefix=prefix)
    suppress_other_loggers()

    # Set cpu affinity to this gpu process
    if get_bool_env_var("SGLANG_SET_CPU_AFFINITY"):
        set_gpu_proc_affinity(server_args.world_size, server_args.nnodes, gpu_id)

    # Create a scheduler and run the event loop
    try:
        scheduler = Scheduler(
            server_args, port_args, gpu_id, attn_tp_rank, moe_ep_rank, dp_rank, global_rank
        )
        pipe_writer.send(
            {
                "status": "ready",
                "max_total_num_tokens": scheduler.max_total_num_tokens,
                "max_req_input_len": scheduler.max_req_input_len,
                "max_running_requests": scheduler.max_running_requests,
                "chunked_prefill_size": scheduler.chunked_prefill_size,
                "context_length": scheduler.model_config.context_len,
            }
        )
        disaggregation_mode: DisaggregationMode = scheduler.disaggregation_mode

        if disaggregation_mode == DisaggregationMode.NULL:
            if scheduler.enable_overlap:
                scheduler.event_loop_overlap()
            else:
                scheduler.event_loop_normal()
        elif disaggregation_mode == DisaggregationMode.PREFILL:
            if scheduler.enable_overlap:
                scheduler.event_loop_overlap_disagg_prefill()
            else:
                scheduler.event_loop_normal_disagg_prefill()
        elif disaggregation_mode == DisaggregationMode.DECODE:
            if scheduler.enable_overlap:
                scheduler.event_loop_overlap_disagg_decode()
            else:
                scheduler.event_loop_normal_disagg_decode()

    except Exception:
        traceback = get_exception_traceback()
        logger.error(f"Scheduler hit an exception: {traceback}")
        parent_process.send_signal(signal.SIGUSR1)