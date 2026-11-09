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

import math
import os
import signal
import sys
import threading
import time
from collections import defaultdict, deque
from concurrent import futures
from http import HTTPStatus
from types import SimpleNamespace
from typing import Optional, Union

import psutil
import setproctitle
import torch
import zmq

# noinspection PyUnusedImports
from sglang.global_config import global_config
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.constants import GPU_MEMORY_ALL_TYPES, GPU_MEMORY_TYPE_CUDA_GRAPH, GPU_MEMORY_TYPE_KV_CACHE, GPU_MEMORY_TYPE_WEIGHTS
from sglang.srt.constrained.base_grammar_backend import create_grammar_backend
from sglang.srt.constrained.reasoner_grammar_backend import ReasonerGrammarObject
from sglang.srt.disaggregation.decode import (
    DecodePreallocQueue,
    DecodeTransferQueue,
    SchedulerDisaggregationDecodeMixin,
)
from sglang.srt.disaggregation.encode.encode_data import (
    CanonicalRequestTokenDescriptor,
)
from sglang.srt.disaggregation.encode.encoder_disagg_mixin import (
    SchedulerEncoderDisaggMixin,
)
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
from sglang.srt.distributed import get_attn_tp_pp_group, get_pp_group, get_scheduler_attn_tp_group, get_scheduler_comm_group, get_world_group
from sglang.srt.env import global_server_args_dict
from sglang.srt.managers.eplb.expert_distribution import (
    get_global_expert_distribution_recorder,
)
from sglang.srt.managers.io_struct import (
    AbortReq,
    ClearHiCacheReqInput,
    ClearHiCacheReqOutput,
    CloseSessionReqInput,
    ExpertDistributionReq,
    ExpertDistributionReqOutput,
    FlushCacheReqInput,
    FlushCacheReqOutput,
    GetInternalStateReq,
    GetInternalStateReqOutput,
    GetLoadReqInput,
    GetLoadReqOutput,
    GetWeightsByNameReqInput,
    GetWeightsByNameReqOutput,
    HeapDumpReq,
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
)
from sglang.srt.managers.kv_event_publisher import SchedulerKvEventsPublisher
from sglang.srt.managers.req import (
    FinishAbort,
    Req,
    RequestStage,
)
from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin
from sglang.srt.managers.scheduling.schedule_batch import EmbeddingBatchResult, GenerationBatchResult, ScheduleBatch
from sglang.srt.managers.scheduling.schedule_policy import (
    CLIP_MAX_NEW_TOKENS_ESTIMATION,
    AddReqResult,
    PrefillAdder,
    SchedulePolicy,
)
from sglang.srt.managers.scheduling.scheduler_multimodal_mixin import (
    SchedulerMultimodalMixin,
)
from sglang.srt.managers.scheduling.scheduler_post_process_mixin import SchedulerPostProcessMixin
from sglang.srt.managers.scheduling.scheduler_profiler_mixin import SchedulerProfilerMixin
from sglang.srt.managers.scheduling.scheduler_stats_mixin import SchedulerStatsMixin
from sglang.srt.managers.session_controller import Session
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.managers.tp_worker_overlap_thread import TpModelWorkerClient
from sglang.srt.managers.utils import validate_input_length
from sglang.srt.mem_cache.hicache_engine import HiCacheEngine
from sglang.srt.mem_cache.hicache_operation import PrefetchStatus
from sglang.srt.mem_cache.radix_cache import HiCacheInitParams, RadixCache
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.observability.metrics.collector import SchedulerLatencyMetrics, SchedulerMetricsCollector, SchedulerStats
from sglang.srt.parser.reasoning_parser import ReasoningParser
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.speculative.eagle_utils import EagleDraftOutput
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
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
from sglang.srt.utils.hf_transformers_utils import get_tokenizer
from sglang.srt.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter

__is_npu__ = is_npu()

from sglang.srt.utils.oe_utils import update_token_table
from sglang.utils import TypeBasedDispatcher, get_exception_traceback

logger = get_colorful_logger(__name__)

# Test retract decode for debugging purposes
TEST_RETRACT = get_bool_env_var("SGLANG_TEST_RETRACT")
RECORD_STEP_TIME = get_bool_env_var("SGLANG_RECORD_STEP_TIME")


class Scheduler(
    SchedulerStatsMixin,
    SchedulerProfilerMixin,
    SchedulerPostProcessMixin,
    SchedulerMultimodalMixin,
    SchedulerDisaggregationDecodeMixin,
    SchedulerDisaggregationPrefillMixin,
    SchedulerEncoderDisaggMixin,
    SchedulerPPMixin,
):
    """A scheduler that manages a tensor parallel GPU worker."""

    def __init__(
        self,
        server_args: ServerArgs,
        port_args: PortArgs,
        gpu_id: int,
        pp_rank: int,
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
        self.pp_rank = pp_rank
        self.global_rank = global_rank
        self.schedule_policy = server_args.schedule_policy
        self.enable_overlap = not server_args.disable_overlap_schedule
        self.skip_tokenizer_init = server_args.skip_tokenizer_init
        self.enable_metrics = server_args.enable_metrics
        self.stream_interval = server_args.stream_interval
        self.spec_algorithm = SpeculativeAlgorithm.from_string(server_args.speculative_algorithm)
        self.gpu_id = gpu_id
        self.enable_hierarchical_cache = server_args.enable_hierarchical_cache

        # KV Events configuration
        self.enable_kv_cache_events = bool(server_args.kv_events_config and attn_tp_rank == 0 and self.pp_rank == 0)
        self.attn_dp_rank = dp_rank if dp_rank is not None else 0
        if self.spec_algorithm.is_eagle():
            self.decode_mem_cache_buf_multiplier = 1
        elif self.spec_algorithm.is_pld():
            # PLD only needs memory for speculative_num_draft_tokens
            self.decode_mem_cache_buf_multiplier = self.server_args.speculative_num_draft_tokens
        else:
            self.decode_mem_cache_buf_multiplier = 1
        self.stream_interval = server_args.stream_interval
        if __is_npu__:
            torch.get_device_module("npu").set_device(self.gpu_id)

        # Distributed rank info
        self.dp_size = server_args.dp_size
        self.dp_rank = dp_rank
        # world_size is currently just world_size
        self.world_size = server_args.world_size

        self.tp_rank = self.global_rank % server_args.tp_size
        self.tp_size = server_args.tp_size
        self.pp_size = server_args.pp_size
        self.attn_tp_pp_size = self.attn_tp_size * self.pp_size

        # Init inter-process communication
        context = zmq.Context(2)
        if self.pp_rank == 0 and self.attn_tp_rank == 0:
            self.recv_from_tokenizer = get_zmq_socket(context, zmq.PULL, port_args.scheduler_input_ipc_name, False)
        else:
            self.recv_from_tokenizer = None

        if self.pp_rank == self.pp_size - 1 and self.attn_tp_rank == 0:
            self.send_to_tokenizer = get_zmq_socket(context, zmq.PUSH, port_args.tokenizer_ipc_name, False)

            if server_args.skip_tokenizer_init:
                # Directly send to the TokenizerManager
                self.send_to_detokenizer = get_zmq_socket(context, zmq.PUSH, port_args.tokenizer_ipc_name, False)
            else:
                # Send to the DetokenizerManager
                self.send_to_detokenizer = get_zmq_socket(context, zmq.PUSH, port_args.detokenizer_ipc_name, False)
        else:
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

        # Load a plain tokenizer first. Multimodal checkpoints replace it below,
        # before reasoning IDs and the grammar backend are initialized.
        if server_args.skip_tokenizer_init:
            self.tokenizer = None
        else:
            self.tokenizer = get_tokenizer(
                server_args.tokenizer_path,
                tokenizer_mode=server_args.tokenizer_mode,
                trust_remote_code=server_args.trust_remote_code,
                revision=server_args.revision,
            )

        # Finalize the tokenizer identity before attaching reasoning state. In
        # particular, multimodal processor tokenizers must not replace it later.
        # (mm tokenizer 替换必须前移到 think_end_id/reasoning 初始化之前,
        # 否则 think_end_id 用 plain tokenizer 编码、与 processor tokenizer 词表错位,
        # thinking_budget 强制截断(apply_thinking_budgets 注入 think_end_ids)在多模下不生效)
        self.init_multimodal_tokenizer(server_args)

        if self.tokenizer is not None and server_args.think_end_token:
            think_end_token = server_args.think_end_token.encode("utf-8").decode("unicode_escape")
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            self.tokenizer.think_end_id = self.tokenizer.encode(think_end_token, add_special_tokens=False)

        # Set reasoning_parser and think_end_id if --reasoning_parser is enabled
        if self.server_args.reasoning_parser and self.tokenizer:
            reasoning_parser = ReasoningParser(model_type=self.server_args.reasoning_parser, stream_reasoning=False, force_reasoning=False)
            self.tokenizer.think_end_id = self.tokenizer.encode(reasoning_parser.detector.think_end_token, add_special_tokens=False)[0]
            # Used to tell whether a request is inside a reasoning block, so that an
            # unterminated one can still be counted. Detectors may force reasoning on
            # from the first token regardless of the flag passed above.
            self.tokenizer.think_force_reasoning = reasoning_parser.detector._in_reasoning
            think_start_ids = self.tokenizer.encode(reasoning_parser.detector.think_start_token, add_special_tokens=False)
            # Only usable as a marker when the start tag is a single token; multi-token
            # start tags would need sequence matching, so skip them.
            self.tokenizer.think_start_id = think_start_ids[0] if len(think_start_ids) == 1 else None

        # Cache think_end_id to avoid expensive getattr on tokenizer every decode step
        self.think_end_id = getattr(self.tokenizer, "think_end_id", None) if self.tokenizer else None
        self.think_start_id = getattr(self.tokenizer, "think_start_id", None) if self.tokenizer else None
        self.think_force_reasoning = getattr(self.tokenizer, "think_force_reasoning", False) if self.tokenizer else False

        # Check whether overlap can be enabled
        if not self.is_generation:
            self.enable_overlap = False
            logger.info("Overlap scheduler is disabled for embedding models.")

        # Launch a tensor parallel worker
        TpWorkerClass = TpModelWorkerClient if self.enable_overlap and self.server_args.speculative_algorithm is None else TpModelWorker

        self.tp_worker = TpWorkerClass(
            server_args=server_args,
            gpu_id=gpu_id,
            attn_tp_rank=attn_tp_rank,
            moe_ep_rank=moe_ep_rank,
            global_rank=global_rank,
            nccl_port=port_args.nccl_port,
        )
        self.speculative_sampling_info = self.tp_worker.model_runner.speculative_sampling_info
        # Launch a draft worker for speculative decoding
        if self.spec_algorithm.is_eagle():
            if self.enable_overlap:
                from sglang.srt.speculative.eagle_worker_overlap import (
                    EagleWorkerOverlapped,
                )

                WorkerClass = EagleWorkerOverlapped
            else:
                if server_args.is_multi_head_eagle:
                    from sglang.srt.speculative.multi_head_eagle_worker import MultiHeadEAGLEWorker

                    WorkerClass = MultiHeadEAGLEWorker
                else:
                    from sglang.srt.speculative.eagle_worker import EAGLEWorker

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
        elif self.spec_algorithm.is_dflash():
            # DFLASH draft 只在末 PP stage 构造, 非末 stage draft_worker=None。
            if self.pp_size == 1 or self.pp_rank == self.pp_size - 1:
                if self.enable_overlap:
                    from sglang.srt.speculative.eagle_worker_overlap import (
                        EagleWorkerOverlapped,
                    )

                    WorkerClass = EagleWorkerOverlapped
                else:
                    from sglang.srt.speculative.dflash_worker import DFlashWorker

                    WorkerClass = DFlashWorker

                self.draft_worker = WorkerClass(
                    server_args=server_args,
                    gpu_id=gpu_id,
                    attn_tp_rank=attn_tp_rank,
                    moe_ep_rank=moe_ep_rank,
                    nccl_port=port_args.nccl_port,
                    target_worker=self.tp_worker,
                    global_rank=global_rank,
                )
                self.reserve_num_tokens = self.server_args.speculative_num_draft_tokens
            else:
                self.draft_worker = None
                # 预留量与末 stage 保持一致, 见 get_new_batch_prefill 的 DFLASH 分支。
                self.reserve_num_tokens = self.server_args.speculative_num_draft_tokens
        elif self.spec_algorithm.is_pld():
            if self.enable_overlap:
                from sglang.srt.speculative.pld_worker_overlap import PLDWorkerOverlapped

                WorkerClass = PLDWorkerOverlapped
            else:
                from sglang.srt.speculative.pld_worker import PLDWorker

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
        self.pp_group = get_pp_group()
        self.world_group = get_world_group()
        self.attn_tp_cpu_group = self.tp_worker.get_attention_tp_cpu_group()
        self.attn_tp_group = self.tp_worker.get_attention_tp_group()
        self.attn_tp_pp_group = get_attn_tp_pp_group() if self.pp_size > 1 else self.attn_tp_group
        self.attn_tp_pp_size = self.attn_tp_pp_group.world_size
        self.attn_tp_pp_cpu_group = get_attn_tp_pp_group().cpu_group if self.pp_size > 1 else self.attn_tp_cpu_group
        self.pp_cpu_group = self.pp_group.cpu_group
        self.pad_input_ids_func = self.tp_worker.get_pad_input_ids_func()
        set_random_seed(self.random_seed)
        self.scheduler_comm_group = self.world_group.cpu_group if self.pp_size > 1 else self.tp_cpu_group
        self.scheduler_comm_force_cpu_device = True
        self.scheduler_comm_device = None
        if server_args.npu_scheduler_comm and __is_npu__:
            assert self.pp_size == 1
            # get comm group after tp_worker init
            self.scheduler_comm_force_cpu_device = False
            self.scheduler_comm_group = get_scheduler_comm_group().device_group
            self.scheduler_comm_device = self.device
            self.attn_tp_cpu_group = get_scheduler_attn_tp_group().device_group

        # Print debug info
        logger.info(
            f"max_total_num_tokens={self.max_total_num_tokens}, "
            f"chunked_prefill_size={server_args.chunked_prefill_size}, "
            f"max_prefill_tokens={self.max_prefill_tokens}, "
            f"max_running_requests={self.max_running_requests}, "
            f"context_len={self.model_config.context_len}"
        )
        self._log_gpu_memory_summary()

        # Added page_size is reserved pad page. Checked here (moved from
        # AttnInitializer.init_memory_pool) so the memory summary above is
        # already logged when the KV pool turns out too small.
        page_size = self.tp_worker.model_runner.page_size
        assert self.max_total_num_tokens >= self.model_config.context_len + page_size, (
            f"KV Cache allocation too small: {self.max_total_num_tokens=}, "
            f"{self.model_config.context_len=}, {page_size=}, may cause requests "
            "to enter Scheduler but cannot be scheduled, causing service to hang"
        )

        # Init npu profiling if activate
        self._init_npu_profiling()

        # Init memory pool and cache
        self.req_to_token_pool, self.token_to_kv_pool, self.kv_allocator = self.tp_worker.get_memory_pool()
        self.enable_hicache_storage = server_args.hicache_storage_backend is not None
        draft_pools = None
        if self.draft_worker and not self.spec_algorithm.is_pld() and (self.pp_size == 1 or self.pp_rank == self.pp_size - 1):
            draft_pools = [runner.token_to_kv_pool for runner in self.draft_worker.model_runner_list]

        if self.enable_hicache_storage:
            self.prefetch_hicache_reqs = []

        params = HiCacheInitParams(
            disable=server_args.disable_radix_cache,
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool=self.token_to_kv_pool,
            token_to_kv_pool_allocator=self.kv_allocator,
            page_size=self.server_args.page_size,
            tp_pp_group=self.attn_tp_pp_cpu_group,  # just for tick
            eviction_policy=server_args.radix_eviction_policy,
            enable_metrics=self.enable_metrics,
            enable_kv_cache_events=self.enable_kv_cache_events,
            draft_token_to_kv_pools=draft_pools,
            attn_tp_rank=self.attn_tp_rank,
            dp_rank=self.dp_rank,
            pp_rank=self.pp_rank,
        )
        if server_args.chunked_prefill_size is not None and server_args.disable_radix_cache:
            self.tree_cache = RadixCache(params=params)
        else:
            if self.enable_hierarchical_cache and not server_args.disable_radix_cache:
                self.tree_cache = HiCacheEngine(params=params, server_args=server_args)
                self.tp_worker.register_hicache_layer_transfer_counter(self.tree_cache.l2_ctrl.layer_done_counter)
                if self.draft_worker:
                    for entry in self.tree_cache.l2_ctrl.pool_entries[1:]:
                        self.draft_worker.register_hicache_layer_transfer_counter(entry.layer_done_counter)
                self.tree_cache.is_decode = self.server_args.disaggregation_mode == DisaggregationMode.DECODE
                if self.tree_cache.l3_ctrl is not None:
                    self.tree_cache.l3_ctrl.is_decode = self.tree_cache.is_decode
            else:
                self.tree_cache = RadixCache(params=params)
        self.max_total_page_num = self.kv_allocator.free_slots.shape[0]
        logger.debug(f"[Scheduler init] tree_cache={self.tree_cache}\nmax_total_page_num={self.max_total_page_num}")

        self.policy = SchedulePolicy(self.schedule_policy, self.tree_cache, self.enable_hierarchical_cache)

        # Init running status
        self.waiting_queue: list[Req] = []
        # The running decoding batch for continuous batching
        self.running_batch: Optional[ScheduleBatch] = ScheduleBatch(reqs=[])
        # The current forward batch
        self.cur_batch: Optional[ScheduleBatch] = None
        # The last forward batch
        self.last_batch: Optional[ScheduleBatch] = None
        self.forward_ct = 0
        self.forward_ct_decode = 0
        self.schedule_forward_ct = -1  # other schedule die, kill all
        self.num_generated_tokens = 0
        self.spec_num_total_accepted_tokens = 0
        self.spec_num_total_forward_ct = 0
        self.cum_spec_accept_length = 0
        self.cum_spec_accept_count = 0
        self.last_decode_stats_tic = time.time()
        self.return_health_check_ct = 0
        self.current_stream = torch.get_device_module(self.device).current_stream()
        self.disaggregation_mode = DisaggregationMode(self.server_args.disaggregation_mode)
        assert not (self.pp_size > 1 and self.disaggregation_mode != DisaggregationMode.PREFILL), "pp_size > 1 requires DisaggregationMode.PREFILL"
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
        self.sessions: dict[str, Session] = {}

        # Init chunked prefill
        self.chunked_prefill_size = server_args.chunked_prefill_size
        if self.chunked_prefill_size <= 0:  # -1 means disable
            self.chunked_prefill_size = None
        self.chunked_req = None
        self.is_mixed_chunk = self.chunked_prefill_size is not None and server_args.enable_mixed_chunk
        if self.server_args.speculative_algorithm is not None:
            assert not self.is_mixed_chunk, "speculative decoding does not support mixed chunk for now"

        # Init the grammar backend for constrained generation
        self.grammar_queue: list[Req] = []
        if not server_args.skip_tokenizer_init and server_args.grammar_backend is not None:
            self.grammar_backend = create_grammar_backend(server_args, self.tokenizer, self.model_config.vocab_size)
        else:
            self.grammar_backend = None

        # Init new token estimation
        assert server_args.schedule_conservativeness >= 0, "Invalid schedule_conservativeness"

        self.init_new_token_ratio = min(
            global_config.default_init_new_token_ratio * server_args.schedule_conservativeness,
            1.0,
        )
        self.min_new_token_ratio = min(
            self.init_new_token_ratio * global_config.default_min_new_token_ratio_factor,
            1.0,
        )
        self.new_token_ratio_decay = (self.init_new_token_ratio - self.min_new_token_ratio) / global_config.default_new_token_ratio_decay_steps
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
        self.memory_saver_adapter = TorchMemorySaverAdapter.create(enable=server_args.enable_memory_saver)

        self.offload_tags = set()

        # Init profiler
        self.init_profiler()

        # Init metrics stats
        self.stats = SchedulerStats()
        if self.enable_metrics:
            self.stats.scheduler_latency_metrics = SchedulerLatencyMetrics()
            self.tree_cache.scheduler_latency_metrics = self.stats.scheduler_latency_metrics
            self.metrics_collector = SchedulerMetricsCollector(
                labels={
                    "model_name": self.server_args.served_model_name,
                    "process_name": f"TP{self.tp_rank}-DP{self.dp_rank}",
                    "app_key": self.server_args.app_key,
                },
                metrics_reporters=server_args.metrics_reporters,
                enable_multimodal=server_args.enable_multimodal,
                language_only=server_args.language_only,
            )
            if (
                self.grammar_backend is not None
                and self.pp_rank == self.pp_size - 1
                and self.attn_tp_rank == 0
            ):
                # Sampling (and therefore degrade application) happens on the
                # last PP stage; all attention-TP ranks agree on the same
                # degrades, so report only on its rank 0 to count a request once.
                self.grammar_backend.set_degrade_reporter(
                    self.metrics_collector.observe_grammar_degrade
                )

        # Init optional MFU profiling (enabled when SGLANG_MFU_HARDWARE + SGLANG_MFU_MODEL are set)
        self.mfu_runner = None
        try:
            from sglang.srt.metrics.mfu import MFURunner

            self.mfu_runner = MFURunner.maybe_from_env(
                hf_config=self.model_config.hf_config,
                world_size=self.server_args.world_size,
            )
            if self.mfu_runner is not None:
                logger.info(
                    f"MFU profiling enabled: hardware={os.environ.get('SGLANG_MFU_HARDWARE')}, "
                    f"model={os.environ.get('SGLANG_MFU_MODEL')}, "
                    f"world_size={self.server_args.world_size}"
                )
        except Exception as e:
            logger.warning(f"Failed to enable MFU profiling: {e}")
            self.mfu_runner = None

        # (launch_time, extend_lens, seq_lens) per in-flight prefill batch,
        # consumed FIFO in log_prefill_mfu so overlap scheduling stays correct.
        self._prefill_mfu_queue = deque()

        # Init KV Events for Dynamo compatibility
        self.send_metrics_from_scheduler = get_zmq_socket(context, zmq.PUSH, port_args.metrics_ipc_name, False)
        self.kv_event_publisher = SchedulerKvEventsPublisher(
            kv_events_config=server_args.kv_events_config,
            attn_tp_rank=self.attn_tp_rank,
            attn_dp_rank=self.attn_dp_rank,
            dp_rank=self.dp_rank,
            pp_rank=self.pp_rank,
            tree_cache=self.tree_cache,
            send_metrics_from_scheduler=self.send_metrics_from_scheduler,
            max_running_requests=self.max_running_requests,
            max_total_num_tokens=self.max_total_num_tokens,
        )

        self.pp_max_running_requests = self.max_running_requests // self.pp_size
        self.preparing_schedule_batch: Optional[ScheduleBatch] = None
        self.preparing_adder: Optional[PrefillAdder] = None

        # Init tracing for Dynamo compatibility
        if server_args.enable_trace:
            try:
                from sglang.srt.observability.tracing.trace import process_tracing_init

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
                (HeapDumpReq, self.heap_dump),
                (GetInternalStateReq, self.get_internal_state),
                (SetInternalStateReq, self.set_internal_state),
                (ExpertDistributionReq, self.expert_distribution_handle),
                (GetLoadReqInput, self.get_load),
            ]
        )

        self.init_disaggregation()
        # Determine if over_embedding is used based on ckpt, if so need to pay attention to
        # n value, need to supplement extra token info during scheduling
        if self.tp_worker.model_config.use_over_embedding:
            self.use_over_embedding = True
            self.token_table = self.tp_worker.model_runner.oe_token_table
            self.over_embedding_n = self.tp_worker.model_config.hf_config.oe_neighbor_num
            self.over_embedding_k = self.tp_worker.model_config.hf_config.oe_split_num
            self.oe_ignore_tokens = self.tp_worker.model_config.oe_ignore_tokens
        else:
            self.use_over_embedding = False

        self.init_multimodal_runtime(server_args)  # SchedulerMultimodalMixin
        self.init_epd_receiver()  # SchedulerEncoderDisaggMixin

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
                    warmup=get_int_env_var("NPU_PROF_WARMUP_TIME", 1),  # Number of steps for warmup after wait
                    active=get_int_env_var("NPU_PROF_ACTIVE_TIME", 5),  # Number of steps to profile after warmup
                    repeat=get_int_env_var("NPU_PROF_REPEAT_TIME", 1),  # How many times to repeat the above process
                ),
                # Save path for profile log, must use tensorboard_trace_handler for viewing with tb
                on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(npu_profiling_save_path),
            )
            self.npu_custom_profiler.prof_if.prof_if = self.npu_custom_profiler.prof_if
        else:
            self.npu_custom_profiler = None

    def init_disaggregation(self):
        self.decode_offload_manager = None
        self.transfer_backend = TransferBackend(self.server_args.disaggregation_transfer_backend)
        self.enable_layerwise_transfer = False

        # EPD defaults (set by init_epd_receiver() in Scheduler.__init__;
        # re-asserted here for safety).
        self.epd_prefill_receiver = None
        self._epd_waiting_reqs: dict[str, Req] = {}
        self._epd_descriptor_outbox: list[CanonicalRequestTokenDescriptor] = []
        self._epd_descriptor_generation = 0
        self._epd_descriptor_last_applied = 0
        # EPD admission state (SchedulerEncoderDisaggMixin): Decode coordinator
        # and Prefill PP0 attention-TP ranks.
        self._epd_admission_budget = self.server_args.epd_max_admissions_per_tick
        self._epd_admitted_backlog: deque[str] = deque()
        self._epd_prepare_worker = None
        self._epd_preparing: dict[str, Req] = {}
        self._epd_prefill_inflight = deque()
        self._epd_prefill_inflight_by_rid = {}

        if self.disaggregation_mode == DisaggregationMode.DECODE:  # *2 for the headroom.
            buffer_size = (self.req_to_token_pool.size) * 2
            req_to_metadata_buffer_idx_allocator = ReqToMetadataIdxAllocator(buffer_size)
            if self.device == "npu":
                self.disagg_metadata_buffers = MetadataBuffers(buffer_size, device=self.device)
            else:
                self.disagg_metadata_buffers = MetadataBuffers(buffer_size)

            # The decode requests polling kv cache
            self.disagg_decode_transfer_queue = DecodeTransferQueue(
                gloo_group=self.attn_tp_cpu_group,
                req_to_metadata_buffer_idx_allocator=req_to_metadata_buffer_idx_allocator,
                metadata_buffers=self.disagg_metadata_buffers,
                scheduler=self,
                tree_cache=self.tree_cache,
                scheduler_comm_device=self.scheduler_comm_device,
            )

            # The decode requests pending for pre-allocation
            self.disagg_decode_prealloc_queue = DecodePreallocQueue(
                req_to_token_pool=self.req_to_token_pool,
                token_to_kv_pool=self.token_to_kv_pool,
                kv_allocator=self.kv_allocator,
                draft_token_to_kv_pools=(
                    None
                    if self.draft_worker is None or self.spec_algorithm.is_pld()
                    else [model_runner.token_to_kv_pool for model_runner in self.draft_worker.model_runner_list]
                ),
                req_to_metadata_buffer_idx_allocator=req_to_metadata_buffer_idx_allocator,
                metadata_buffers=self.disagg_metadata_buffers,
                scheduler=self,
                transfer_queue=self.disagg_decode_transfer_queue,
                tree_cache=self.tree_cache,
                gloo_group=self.attn_tp_cpu_group,
                tp_rank=self.tp_rank,
                tp_size=self.tp_size,
                bootstrap_port=self.server_args.disaggregation_bootstrap_port,
                max_total_num_tokens=self.max_total_num_tokens,
                transfer_backend=self.transfer_backend,
                scheduler_comm_device=self.scheduler_comm_device,
                pp_rank=self.pp_rank,
                pp_size=self.pp_size,
            )

            if self.server_args.disaggregation_decode_enable_offload_kvcache and getattr(self.tree_cache, "l3_ctrl", None) is not None:
                from sglang.srt.disaggregation.decode_kvcache_offload_manager import (
                    DecodeKVCacheOffloadManager,
                )

                self.decode_offload_manager = DecodeKVCacheOffloadManager(
                    self.tree_cache,
                    self.disagg_decode_prealloc_queue.kv_manager,
                )

            # Metric for pre-allocation
            self.num_tokens_pre_allocated = 0

        elif self.disaggregation_mode == DisaggregationMode.PREFILL:
            # *2 for the headroom.
            buffer_size = self.max_running_requests * 2
            req_to_metadata_buffer_idx_allocator = ReqToMetadataIdxAllocator(buffer_size)
            self.enable_layerwise_transfer = self.transfer_backend == TransferBackend.MOONCAKE_ASYNC
            if self.device == "npu":
                self.disagg_metadata_buffers = MetadataBuffers(
                    buffer_size, device=(f"npu:{self.gpu_id}" if self.enable_layerwise_transfer else self.device)
                )
            else:
                self.disagg_metadata_buffers = MetadataBuffers(
                    buffer_size,
                    device=(f"cuda:{self.gpu_id}" if self.enable_layerwise_transfer else "cpu"),
                )

            self.disagg_prefill_bootstrap_queue = PrefillBootstrapQueue(
                token_to_kv_pool=self.token_to_kv_pool,
                kv_allocator=self.kv_allocator,
                draft_token_to_kv_pools=(
                    None
                    if self.draft_worker is None or self.spec_algorithm.is_pld()
                    else [model_runner.token_to_kv_pool for model_runner in self.draft_worker.model_runner_list]
                ),
                req_to_metadata_buffer_idx_allocator=req_to_metadata_buffer_idx_allocator,
                metadata_buffers=self.disagg_metadata_buffers,
                tp_rank=self.tp_rank,
                tp_size=self.tp_size,
                bootstrap_port=self.server_args.disaggregation_bootstrap_port,
                gloo_group=self.attn_tp_pp_cpu_group,
                transfer_backend=self.transfer_backend,
                scheduler=self,
                pp_rank=self.pp_rank,
                pp_size=self.pp_size,
            )

            # register step_counter for async mode
            if self.enable_layerwise_transfer:
                self.step_counter = StepCounter(self.device, gpu_id=self.gpu_id)
                self.kv_transfer_manager = self.disagg_prefill_bootstrap_queue.kv_manager
                self.kv_transfer_manager.register_step_counter(self.step_counter)
                self.tp_worker.model_runner.attn_backend.register_step_counter(self.step_counter)
                if not self.spec_algorithm.is_none() and not self.spec_algorithm.is_pld() and self.draft_worker is not None:
                    for model_runner in self.draft_worker.model_runner_list:
                        model_runner.attn_backend.register_step_counter(self.step_counter)
            logger.info(f"enable_layerwise_transfer: {self.enable_layerwise_transfer}")

            # The prefill requests that are in the middle of kv sending
            self.disagg_prefill_inflight_queue: list[Req] = []

    def cache_finished_req(self, req: Req, delay_req_pool_release: bool = False):
        finalizer = self.decode_offload_manager or self.tree_cache
        finalizer.cache_finished_req(req, delay_req_pool_release=delay_req_pool_release)

    def tick_cache(self):
        (self.decode_offload_manager or self.tree_cache).tick()

    def set_next_batch_sampling_info_done(self, batch: ScheduleBatch):
        if batch.next_batch_sampling_info:
            # For MTP the draft worker sets sampling_info_done=None: all grammar
            # operations live on the forward thread, so the scheduler must neither
            # compute the (discarded) vocab mask nor signal an event here.
            if batch.next_batch_sampling_info.sampling_info_done is None:
                return
            if batch.next_batch_sampling_info.grammars is not None:
                batch.next_batch_sampling_info.update_regex_vocab_mask()
                self.current_stream.synchronize()
            batch.next_batch_sampling_info.sampling_info_done.set()

    def watchdog_thread(self):
        """A watch dog thread that will try to kill the server itself if one forward batch takes too long."""
        self.watchdog_last_forward_ct = 0
        self.watchdog_last_time = time.time()

        self.watchdog_last_schedule_forward_ct = self.schedule_forward_ct
        self.watchdog_last_schedule_time = time.time()

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

            if self.schedule_forward_ct != -1:  # Forward thread not hung，But scheduler hung
                if self.watchdog_last_schedule_forward_ct == self.schedule_forward_ct:
                    if current > self.watchdog_last_schedule_time + self.watchdog_timeout:
                        logger.error(f"Watchdog timeout ({self.watchdog_timeout=})")
                        break
                else:
                    self.watchdog_last_schedule_forward_ct = self.schedule_forward_ct
                    self.watchdog_last_schedule_time = current
            time.sleep(self.watchdog_timeout // 5)

        # Print batch size and memory pool info to check whether there are de-sync issues.
        if self.cur_batch is not None:
            logger.error(
                f"{self.cur_batch.batch_size()=}, "
                f"{self.cur_batch.reqs=}, "
                f"{self.kv_allocator.available_size()=}, "
                f"{self.tree_cache.evictable_size()=}, "
            )
        else:
            logger.error(
                f"{self.watchdog_last_schedule_forward_ct=}, "
                f"{self.schedule_forward_ct=}, "
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
                self.poll_epd_admissions()

                recv_reqs = self.recv_requests()
                self.process_input_requests(recv_reqs)
                self.process_prefetches()

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
                self.poll_epd_admissions()

                # this thread not does actual kernel launch
                recv_reqs = self.recv_requests()
                self.process_input_requests(recv_reqs)
                self.process_prefetches()

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
                                self.tp_worker.cur_sampling_info if self.draft_worker is None else self.draft_worker.cur_sampling_info
                            ),
                        )
                        self.process_batch_result(tmp_batch, None, batch.launch_done)
                        if step_func:
                            step_func()  # p.step()

                if self.last_batch:
                    # Process the results of the last batch
                    tmp_batch, tmp_result = self.result_queue.popleft()
                    if self.draft_worker is None:
                        tmp_batch.next_batch_sampling_info = self.tp_worker.cur_sampling_info if batch else None
                    else:
                        tmp_batch.next_batch_sampling_info = self.draft_worker.cur_sampling_info if batch else None
                    self.process_batch_result(tmp_batch, tmp_result, batch.launch_done if batch else None)
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

    def recv_requests(self) -> list[Req]:
        """Receive results at attn_tp_rank = 0 and broadcast it to all other TP ranks."""
        if self.pp_rank == 0 and self.attn_tp_rank == 0:
            recv_reqs = []
            while True:
                try:
                    recv_req = self.recv_from_tokenizer.recv_pyobj(zmq.NOBLOCK)
                except zmq.ZMQError:
                    break
                recv_reqs.append(recv_req)
            # EPD descriptors share the existing ordered Scheduler broadcast
            # boundary.  Injecting them here prevents background collectives
            # and guarantees every PP/TP participant commits in the same frame.
            # poll_epd_admissions() happened before this receive.  Commit its
            # descriptors before later external control messages (notably an
            # AbortReq drained in the same frame), preserving event-loop order.
            recv_reqs = self.drain_epd_descriptors() + recv_reqs
        else:
            recv_reqs = None

        if self.server_args.enable_dp_attention:
            if self.attn_tp_rank == 0 and self.pp_rank == 0:
                work_req_types = (
                    TokenizedGenerateReqInput,
                    TokenizedEmbeddingReqInput,
                    CanonicalRequestTokenDescriptor,
                )
                work_reqs = [req for req in recv_reqs if isinstance(req, work_req_types)]
                control_reqs = [req for req in recv_reqs if not isinstance(req, work_req_types)]
            else:
                work_reqs = None
                control_reqs = None

            if self.attn_tp_pp_size != 1:
                work_reqs = broadcast_pyobj(
                    work_reqs,
                    self.attn_tp_pp_group.rank,
                    self.attn_tp_pp_cpu_group,
                    src=self.attn_tp_pp_group.ranks[0],
                    force_cpu_device=self.scheduler_comm_force_cpu_device,
                )
            if self.world_size != 1:
                broadcast_pyobj_time_start = time.time()
                control_reqs = broadcast_pyobj(
                    control_reqs,
                    self.world_group.rank,
                    self.scheduler_comm_group,
                    src=self.world_group.ranks[0],
                    force_cpu_device=self.scheduler_comm_force_cpu_device,
                )
                if self.enable_metrics:
                    self.metrics_collector.observe_module_time_in_scheduler(time.time() - broadcast_pyobj_time_start, "broadcast_pyobj_time")
            recv_reqs = work_reqs + control_reqs
        elif self.world_size != 1:
            recv_reqs = broadcast_pyobj(recv_reqs, self.world_group.rank, self.world_group.cpu_group, src=self.world_group.ranks[0])

        # Unwrap shared memory features AFTER all broadcasts complete
        if recv_reqs:
            self._unwrap_mm_features(recv_reqs)

        return recv_reqs

    def process_input_requests(self, recv_reqs: list):
        for recv_req in recv_reqs:
            if isinstance(recv_req, CanonicalRequestTokenDescriptor):
                self.apply_epd_descriptor(recv_req)
                continue

            # If it is a health check generation request and there are running requests, ignore it.
            # In disaggregation prefill mode, we should not ignore health check requests based on running_batch
            # because prefill server doesn't maintain a running_batch in the same way as normal/decode mode
            should_ignore_health_check = is_health_check_generate_req(recv_req) and (
                self.chunked_req is not None or (self.running_batch is not None and self.disaggregation_mode != DisaggregationMode.PREFILL)
            )

            if should_ignore_health_check:
                if self.global_rank == 0:
                    logger.debug(
                        f"[SCHEDULER] Ignoring health check request because "
                        f"chunked_req={self.chunked_req is not None}, "
                        f"running_batch={self.running_batch is not None}"
                    )
                self.return_health_check_ct += 1
                continue

            output = self._request_dispatcher(recv_req)
            if output is not None:
                self.send_to_tokenizer.send_pyobj(output)

    def handle_generate_request(
        self,
        recv_req: TokenizedGenerateReqInput,
    ):
        # Preserve the existing EPD deferred-tokenization contract.
        missing_input_ids = (
            recv_req.input_ids is None
            and recv_req.input_embeds is None
            and not (recv_req.need_wait_for_mm_inputs and recv_req.epd_media_urls is not None)
        )
        # Invalid inputs must also bypass session merging, which requires IDs.
        if (
            missing_input_ids
            or recv_req.session_params is None
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

            # Handle custom logit processor passed to the request
            custom_logit_processor = recv_req.custom_logit_processor
            if not self.server_args.enable_custom_logit_processor and custom_logit_processor is not None:
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
                origin_input_multi_ids=recv_req.input_multi_ids,
                metrics_collector=(self.metrics_collector if self.enable_metrics else None),
                created_time=recv_req.created_time,
                # mm_inputs produced by tokenizer_manager (hash+pad_value already set).
                mm_inputs=recv_req.mm_inputs,
            )
            # EPD: carry forward the flag and meta so the Scheduler routes
            # the request to EpdPrefillReceiver.enqueue() instead of the normal queue.
            # need_wait_for_mm_inputs / mm_meta are set by TokenizerManager when it dispatches to the remote Encoder.
            req.mm_meta = recv_req.mm_meta
            req.need_wait_for_mm_inputs = recv_req.need_wait_for_mm_inputs

            req.epd_media_urls = recv_req.epd_media_urls
            req.epd_input_text = recv_req.epd_input_text
            req.epd_encoder_urls = recv_req.epd_encoder_urls
            req.set_tokenizer(self.tokenizer)

            if missing_input_ids:
                error_message = f"Invalid request: missing input_ids. rid={req.rid}"
                logger.error(error_message)
                prepare_abort(req, error_message, HTTPStatus.BAD_REQUEST, "BadRequestError")
                self.stream_output([req], req.return_logprob)
                return

            if req.need_wait_for_mm_inputs and req.epd_media_urls is not None:
                # Save recv_req.logprob_start_len so that _finalize_request_after_tokenization
                # (called from _process_epd_admitted after deferred tokenization) can read it.
                req.logprob_start_len = recv_req.logprob_start_len
                self._add_request_to_queue(req)
                return

            if self.disaggregation_mode != DisaggregationMode.NULL:
                # Invalid request for disaggregated mode
                if recv_req.bootstrap_room is None:
                    error_message = f"Invalid request: Disaggregated request received without boostrap room id. {req.rid=}"
                    logger.error(error_message)
                    prepare_abort(req, error_message)
                    self.stream_output([req], req.return_logprob)
                    return

            if len(req.origin_input_ids) == 0:
                error_message = f"Invalid request: empty input_ids. {req.rid=} "
                logger.error(error_message)
                prepare_abort(req, error_message)
                self.stream_output([req], req.return_logprob)
                return

            if recv_req.session_params is not None and recv_req.session_params.id is not None:
                req.finished_reason = FinishAbort(f"Invalid request: session id {recv_req.session_params.id} does not exist")
                self._add_request_to_queue(req)
                return
        else:
            # Create a new request from a previous session
            session = self.sessions[recv_req.session_params.id]
            req = session.create_req(recv_req, self.tokenizer)
            if isinstance(req.finished_reason, FinishAbort):
                self._add_request_to_queue(req)
                return

        # Save recv_req.logprob_start_len onto req for _finalize_request_after_tokenization.
        # (The EPD early-return branch above already saved it and returned, so
        # control reaching here means this is the normal/session path.)
        req.logprob_start_len = recv_req.logprob_start_len
        self._finalize_request_after_tokenization(req)

    def _finalize_request_after_tokenization(self, req: Req) -> None:
        """Validate + set logprob_start_len + clip max_new_tokens + init grammar + enqueue.

        Shared by:
        - handle_generate_request (normal path, after Req creation + session handling).
        - _process_epd_admitted (EPD path, after deferred tokenization sets
          origin_input_ids / input_ids / fill_ids / mm_inputs).

        Ensures EPD-admitted requests pass through the SAME validation +
        parameter-setting pipeline as normal requests, so logprob_start_len /
        max_new_tokens clipping / validate_input_length / grammar init are never
        skipped.  Caller must set ``req.logprob_start_len`` (from
        ``recv_req.logprob_start_len``) before calling this method.
        """
        # 1. Validate prompts length (truncate or abort).
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

        # 2. logprob_start_len: req.logprob_start_len was set by the caller
        #    (from recv_req.logprob_start_len). -1 or no return_logprob → default
        #    to last input token (only output logprobs).
        if req.logprob_start_len == -1 or not req.return_logprob:
            req.logprob_start_len = len(req.origin_input_ids) - 1

        if req.logprob_start_len >= len(req.origin_input_ids):
            req.finished_reason = FinishAbort(
                f"logprob_start_len, ({req.logprob_start_len}) is higher than the number "
                f"of input tokens ({len(req.origin_input_ids)}). "
                f"Request with a lower logprob_start_len.",
                HTTPStatus.BAD_REQUEST,
                "BadRequestError",
            )
            req.logprob_start_len = len(req.origin_input_ids) - 1
            self._add_request_to_queue(req)
            return

        # 3. max_new_tokens clipping against max_req_len.
        req.sampling_params.max_new_tokens = min(
            (req.sampling_params.max_new_tokens if req.sampling_params.max_new_tokens is not None else 1 << 30),
            self.max_req_len - len(req.origin_input_ids) - 1,
        )

        # 4. Init grammar cache for this request.
        add_to_grammar_queue = False
        if (
            req.sampling_params.json_schema is not None
            or req.sampling_params.regex is not None
            or req.sampling_params.ebnf is not None
            or req.sampling_params.structural_tag is not None
        ) and self.grammar_backend is not None:
            if req.sampling_params.json_schema is not None:
                key = ("json", req.sampling_params.json_schema)
            elif req.sampling_params.regex is not None:
                key = ("regex", req.sampling_params.regex)
            elif req.sampling_params.ebnf is not None:
                key = ("ebnf", req.sampling_params.ebnf)
            elif req.sampling_params.structural_tag:
                key = ("structural_tag", req.sampling_params.structural_tag)

            req.grammar, cache_hit = self.grammar_backend.get_cached_or_future_value(key)
            if not cache_hit:
                req.grammar_key = key
                add_to_grammar_queue = True
            else:
                # Hit an invalid grammar cached earlier: mark for abort now,
                # since this req skips the grammar_queue / _check_grammar_compilation path.
                self._check_grammar_compilation(req)
                self._apply_reasoning_grammar_state(req)

        if add_to_grammar_queue:
            self.grammar_queue.append(req)
        else:
            self._add_request_to_queue(req)

    def _prepare_prefetch_kvcache(self, req: Req) -> bool:
        if self.enable_hicache_storage:
            req.init_next_round_input(self.tree_cache)
            # Determine tokens to query L3 for:
            # - When host is present (last_host_node != None): tokens after L1+L2
            # - When host is evicted (last_host_node is None): tokens after device
            #   prefix only, since host data needs to be re-fetched from L3
            if req.last_host_node is not None:
                matched_len = req.prefix_len + req.host_hit_length
            else:
                matched_len = req.prefix_len
            new_input_tokens = req.fill_ids[matched_len:]

            prefetch_length = len(new_input_tokens) - (len(new_input_tokens) % self.tree_cache.page_size)
            if prefetch_length == 0 or prefetch_length < self.tree_cache.prefetch_threshold:
                logger.debug(
                    f"[L3-Prefetch-Query-Skip] req={req.rid} prefetch_length={prefetch_length} threshold={self.tree_cache.prefetch_threshold}"
                )
                return

            # Get the anchor node for prefetch insertion and hash chain
            prefetch_parent = req.last_host_node if req.last_host_node is not None else req.last_node
            # When host is evicted AND node is deleted from tree, both are None.
            # In that case, start fresh: query L3 for all tokens with last_hash=None.
            if prefetch_parent is None:
                last_hash = None
                lease_prefix_keys = None
            else:
                last_hash = prefetch_parent.get_last_hash_value()
                # These hashes are already available in L1/L2. Include them in
                # Mooncake's existence request so an L3 copy, when present, has
                # its lease renewed too. A missing L3 copy in this known-local
                # prefix must not break the queried L3 suffix.
                # get_prefix_hash_values() is cached on TreeNode, while the
                # L3 controller extends prefix_keys as batches are loaded.
                # Copy here so controller updates cannot corrupt that cache.
                lease_prefix_keys = list(prefetch_parent.get_prefix_hash_values(prefetch_parent))

            logger.debug(
                f"[L3-Prefetch-Try] req={req.rid} prefix_len={req.prefix_len} "
                f"host_hit_length={req.host_hit_length} matched_len={matched_len} "
                f"new_tokens={len(new_input_tokens)} "
                f"prefetch_parent={'host' if req.last_host_node else 'device'} "
                f"last_hash={'present' if last_hash else 'None'}"
            )

            if self.tree_cache.prepare_prefetch(
                req,
                prefetch_parent,
                new_input_tokens,
                last_hash,
                lease_prefix_keys,
            ):
                self.prefetch_hicache_reqs.append(req)

    def _release_l3_prefetch_ready(self, req: Req, detach: bool = False):
        if not self.enable_hicache_storage:
            return
        self.tree_cache.l3_ctrl.release_prefetch_ready(req)
        if detach and req.s2h_load is not None:
            req.s2h_load.mark_detached()

    def _add_request_to_queue(self, req: Req):
        req.queue_time_start = time.time()

        # EPD: route MM-pending requests to EpdPrefillReceiver (SchedulerEncoderDisaggMixin).
        # Returns True if intercepted; the caller must NOT enqueue it normally.
        if self.route_epd_request(req):
            return

        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            self._prepare_prefetch_kvcache(req)
            self.disagg_prefill_bootstrap_queue.add(req)
        elif self.disaggregation_mode == DisaggregationMode.DECODE:
            self.disagg_decode_prealloc_queue.add(req)
        else:
            self._prepare_prefetch_kvcache(req)
            self.waiting_queue.append(req)

    def _extend_requests_to_queue(self, reqs: list[Req], is_retracted: bool = False):
        # EPD: route MM-pending requests to EpdPrefillReceiver (SchedulerEncoderDisaggMixin).
        # route_epd_requests_batch stages EPD-pending reqs and returns the rest.
        reqs = self.route_epd_requests_batch(reqs)
        if not reqs:
            return

        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            self.disagg_prefill_bootstrap_queue.extend(reqs)
        elif self.disaggregation_mode == DisaggregationMode.DECODE:
            # If this is a decode server, we put the request to the decode pending prealloc queue
            self.disagg_decode_prealloc_queue.extend(reqs, is_retracted)
        else:
            self.waiting_queue.extend(reqs)

    def _get_num_used_pages(self):
        return self.max_total_page_num - (self.kv_allocator.available_size() + self.tree_cache.evictable_size())

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
                req.prefix_page_ids = self.kv_allocator.req_to_page[req.req_pool_idx, 0 : (req.prefix_len + page_size - 1) // page_size]

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

        # Check if the grammar is ready in the grammar queue
        if self.grammar_queue:
            self.move_ready_grammar_requests()
        if self.enable_hierarchical_cache:
            self.tree_cache.tick()
        new_batch = self.get_new_batch_prefill()
        if new_batch is not None:
            # Run prefill first if possible
            ret = new_batch
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
            # Mark non-last-chunk chunked prefill requests to skip token_table update in sample.
            # Use self.chunked_req identity (set during scheduling) instead of req.is_chunked
            # to avoid timing issues with overlap scheduling.
            if ret.forward_mode.is_extend() and self.chunked_req is not None:
                skip_mask = [req is self.chunked_req for req in ret.reqs]
                if any(skip_mask):
                    ret.oe_skip_token_table_update = torch.tensor(skip_mask, dtype=torch.bool, device=self.device)
                else:
                    ret.oe_skip_token_table_update = None
            else:
                ret.oe_skip_token_table_update = None

    def init_token_table(self, schedule_batch: ScheduleBatch):
        """
        Initialize requests from schedule_batch into token table when needed:
        1. In prefill case, initialization is needed
        2. For PD disaggregation decode nodes, initialization is needed on first scheduling
        3. For PD disaggregation decode nodes, when recovering retracted requests, initialization is needed
           (In non-PD case, retract recovery triggers re-prefill, but in PD case, kvcache is restored directly from CPU)
        Iterate through each request, merge all update cases, and update token table in one batch
        Currently only consider prefill
        """
        tokens = []
        column_starts = []
        request_lengths = []
        req_pool_indices = []
        for req in schedule_batch.reqs:
            if not req.oe_init:
                fill_ids = req.origin_input_ids + req.output_ids
                tokens.extend(fill_ids)
                column_starts.append(0)
                req_pool_indices.append(req.req_pool_idx)
                request_lengths.append(len(fill_ids))
                req.oe_init = True

        if len(req_pool_indices) > 0:
            dtype = schedule_batch.token_table.dtype
            device = schedule_batch.token_table.device
            ignore_tokens = None
            if self.oe_ignore_tokens:
                ignore_tokens = torch.tensor(self.oe_ignore_tokens, dtype=torch.int32, device=device)
            tokens_tensor = torch.tensor(tokens, dtype=dtype, pin_memory=True).to(device, non_blocking=True)
            # RadixCache pad values are n-gram boundaries.
            from sglang.srt.managers.scheduling.multimodal_data import MM_PAD_SHIFT_VALUE

            pad_mask = tokens_tensor >= MM_PAD_SHIFT_VALUE
            tokens_tensor[pad_mask] = -tokens_tensor[pad_mask].abs()
            row_indices_tensor = torch.tensor(req_pool_indices, dtype=torch.int64, pin_memory=True).to(device, non_blocking=True)
            column_starts_tensor = torch.tensor(column_starts, dtype=torch.int32, pin_memory=True).to(device, non_blocking=True)
            oe_req_lens_tensor = torch.tensor(request_lengths, dtype=torch.int32, pin_memory=True).to(device, non_blocking=True)
            update_token_table(
                oe_token_table=schedule_batch.token_table,
                tokens=tokens_tensor,
                row_indices=row_indices_tensor,
                column_starts=column_starts_tensor,
                oe_req_lens=oe_req_lens_tensor,
                ignore_tokens=ignore_tokens,
            )

    def process_prefetches(self):
        if not self.enable_hicache_storage:
            return

        if len(self.prefetch_hicache_reqs) == 0:
            return

        reqs = self.prefetch_hicache_reqs
        l3 = self.tree_cache.l3_ctrl
        now = time.monotonic()
        status_tensor = torch.tensor(
            [req.s2h_load.get_prefetch_status_for_sync(now) for req in reqs],
            dtype=torch.int32,
        )
        if self.attn_tp_pp_size > 1:
            torch.distributed.all_reduce(
                status_tensor,
                op=torch.distributed.ReduceOp.MIN,
                group=self.attn_tp_pp_cpu_group,
            )

        slm = self.stats.scheduler_latency_metrics
        if slm is not None:
            for req in reqs:
                s = req.s2h_load
                if not s._pf_query_recorded and s.t_query_done > 0.0 and s.t_query_enq > 0.0:
                    slm.add_pf_query((s.t_query_done - s.t_query_enq) * 1000.0)
                    s._pf_query_recorded = True
                if not s._pf_queue_recorded and s.t_read_start > 0.0 and s.t_read_enq > 0.0:
                    slm.add_pf_queue((s.t_read_start - s.t_read_enq) * 1000.0)
                    s._pf_queue_recorded = True

        remaining_reqs = []
        io_done_reqs = []
        for i, req in enumerate(reqs):
            status = status_tensor[i].item()
            if status in (
                PrefetchStatus.SoftTimeout,
                PrefetchStatus.QueryMiss,
            ):
                req.s2h_load.mark_detached()
                remaining_reqs.append(req)
                if status == PrefetchStatus.SoftTimeout and self.attn_tp_rank == 0 and self.pp_rank == 0:
                    logger.warning(
                        "[L3-Prefetch-Soft-Timeout] rid=%s; continuing without L3",
                        req.rid,
                    )
            elif status == PrefetchStatus.WaitAllocate:
                # todo add l2 cache watermark
                if self.tree_cache.start_prefetch(req):
                    remaining_reqs.append(req)
            elif status == PrefetchStatus.IODone:
                io_done_reqs.append(req)
            else:
                remaining_reqs.append(req)

        if io_done_reqs:
            detached_reqs = [req for req in io_done_reqs if req.s2h_load.is_detached()]
            completed_reqs = [req for req in io_done_reqs if not req.s2h_load.is_detached()]
            for req in detached_reqs:
                req.s2h_load.set_synced_completed_tokens(-1)
            if l3.enable_l15:
                l3.sync_l15_masks_batched([req.s2h_load for req in completed_reqs])
            else:
                for req in completed_reqs:
                    req.s2h_load.set_synced_completed_tokens(l3.sync_completed_tokens(req.s2h_load))
            for req in io_done_reqs:
                s = req.s2h_load
                self.tree_cache.finalize_prefetch(req)
                if (
                    self.enable_metrics
                    and self.attn_tp_rank == 0
                    and (self.pp_rank == 0 or self.pp_rank == self.pp_size - 1)
                    and s.t_query_enq > 0.0
                    and s.t_finish_done >= s.t_query_enq
                ):
                    completed_tokens = s.get_synced_completed_tokens()
                    if completed_tokens is not None and completed_tokens >= 0:
                        query_queue_s = s.t_query_start - s.t_query_enq
                        storage_query_s = s.t_query_storage_done - s.t_query_start
                        query_sync_s = s.t_query_done - s.t_query_storage_done
                        query_s = s.t_query_done - s.t_query_enq
                        query_to_read_s = s.t_read_enq - s.t_query_done
                        queue_s = s.t_read_start - s.t_read_enq
                        read_s = s.t_io_done - s.t_read_start
                        finish_s = s.t_finish_done - s.t_io_done
                        l3_prefetch_e2e_s = s.t_finish_done - s.t_query_enq
                        # Per-stage latencies are reported here, where the
                        # timestamps live. Report on PP rank 0 so the series
                        # lands on the master node's /metrics endpoint (the
                        # tokenizer process also lives there). Only e2e/tokens
                        # travel to the tokenizer via the last PP rank's output
                        # batch (for the TTFT-percentage metric).
                        if self.pp_rank == 0:
                            self.metrics_collector.observe_l3_prefetch_stages(
                                query_queue_s=query_queue_s,
                                storage_query_s=storage_query_s,
                                query_sync_s=query_sync_s,
                                query_s=query_s,
                                query_to_read_s=query_to_read_s,
                                queue_s=queue_s,
                                read_s=read_s,
                                finish_s=finish_s,
                            )
                            if l3_prefetch_e2e_s > 5.0:
                                logger.warning(
                                    "[L3-Prefetch-Slow] rid=%s e2e_ms=%.1f "
                                    "query_ms=%.1f query_queue_ms=%.1f "
                                    "storage_hit_query_ms=%.1f query_sync_ms=%.1f "
                                    "query_to_read_ms=%.1f "
                                    "queue_ms=%.1f read_ms=%.1f finish_ms=%.1f "
                                    "completed_tokens=%d",
                                    req.rid,
                                    l3_prefetch_e2e_s * 1000.0,
                                    query_s * 1000.0,
                                    query_queue_s * 1000.0,
                                    storage_query_s * 1000.0,
                                    query_sync_s * 1000.0,
                                    query_to_read_s * 1000.0,
                                    queue_s * 1000.0,
                                    read_s * 1000.0,
                                    finish_s * 1000.0,
                                    completed_tokens,
                                )
                        if self.pp_rank == self.pp_size - 1:
                            req.l3_prefetch_e2e_s = l3_prefetch_e2e_s
                            req.l3_prefetch_tokens = completed_tokens
                if slm is not None and s.t_finish_done > 0.0 and s.t_io_done > 0.0:
                    slm.add_pf_finish((s.t_finish_done - s.t_io_done) * 1000.0)
        self.prefetch_hicache_reqs = remaining_reqs

    def get_running_batch_reserved_token_info(self):
        # reserved token for running_batch or mixed_decode
        reserved_input_tokens = 0
        reserved_kv_pages = 0
        reserved_req_num = 0
        if self.running_batch is not None:
            for req in self.running_batch.reqs:
                if req.sampling_params.ignore_eos and global_server_args_dict["disaggregation_mode"] != "prefill":
                    req_reserved_kv_tokens = req.sampling_params.max_new_tokens - len(req.output_ids)
                else:
                    req_reserved_kv_tokens = (
                        min(req.sampling_params.max_new_tokens - len(req.output_ids), CLIP_MAX_NEW_TOKENS_ESTIMATION) * self.new_token_ratio
                    )
                page_size = self.kv_allocator.page_size
                reserved_kv_pages += (req_reserved_kv_tokens + page_size - 1) // page_size

            if self.is_mixed_chunk:
                reserved_input_tokens += len(self.running_batch.reqs)
                reserved_kv_pages += len(self.running_batch.reqs)
            reserved_req_num += len(self.running_batch.reqs)
        return reserved_input_tokens, reserved_kv_pages, reserved_req_num

    def get_new_batch_prefill(self, log=True) -> Optional[ScheduleBatch]:
        # Handle the cases where prefill is not allowed
        if (self.batch_is_full or len(self.waiting_queue) == 0) and self.chunked_req is None:
            return None

        running_bs = len(self.running_batch.reqs) if self.running_batch else 0
        if running_bs >= self.pp_max_running_requests:
            self.batch_is_full = True
            return None

        # Ignore the check if self.chunked_req is not None.
        # In the non-PP case, when self.chunked_req is not None, num_allocatable_reqs should always be
        # greater than 0, as the space for the chunked requests has just been released.
        # In PP case, chunked requests (or dllm requests) can start in one microbatch and end in another
        # microbatch, so the max_running_requests per microbatch should not be strict.
        # Instead, we should always allow chunked requests to be added, otherwise, there will be a memory leak.
        if self.req_to_token_pool.available_size() <= 0 and self.chunked_req is None:
            self.batch_is_full = True
            return None

        # Get priority queue
        prefix_computed = self.policy.calc_priority(self.waiting_queue)
        if self.preparing_schedule_batch is not None:
            assert self.preparing_adder
            adder = self.preparing_adder
            old_can_run_list_len = len(adder.can_run_list)
        else:
            if self.pp_size > 1:
                assert self.max_prefill_tokens == self.chunked_prefill_size
            reserved_input_tokens, reserved_kv_pages, reserved_req_num = self.get_running_batch_reserved_token_info()
            adder = PrefillAdder(
                self.tree_cache,
                self.kv_allocator,
                self.chunked_prefill_size,
                reserved_input_tokens,
                reserved_kv_pages,
                self.pp_max_running_requests - reserved_req_num,
                self.stats.cache_hit_metrics,
            )
            old_can_run_list_len = 0

        if self.chunked_req is not None:
            chunked_req = self.chunked_req
            chunked_req.prefix_len = self.req_to_token_pool.alloced_lens[chunked_req.req_pool_idx].item()
            page_size = self.kv_allocator.page_size
            chunked_req.prefix_page_ids = self.kv_allocator.req_to_page[
                chunked_req.req_pool_idx, 0 : (chunked_req.prefix_len + page_size - 1) // page_size
            ]
            self.chunked_req.init_next_round_input()
            res = adder.add_chunked_req(self.chunked_req)
            if res == AddReqResult.NO_TOKEN:
                self.batch_is_full = True
        else:
            res = adder.budget_state()

        if res == AddReqResult.NO_TOKEN:
            self.batch_is_full = True
        elif res == AddReqResult.CONTINUE:
            for req in self.waiting_queue:
                if len(adder.can_run_list) >= self.req_to_token_pool.available_size():
                    self.batch_is_full = True
                    break

                if self.enable_hicache_storage and req.s2h_load is not None and not req.s2h_load.is_detached():
                    continue

                req.init_next_round_input(None if prefix_computed else self.tree_cache)

                res = adder.add_one_req(req)
                if res != AddReqResult.CONTINUE:
                    if res == AddReqResult.NO_TOKEN:
                        self.batch_is_full = True
                    break

        # Update waiting queue
        can_run_list: list[Req] = adder.can_run_list[old_can_run_list_len:]
        if len(can_run_list) == 0:
            return None
        self.preparing_adder = adder

        # Decrement inflight retracted counter when retracted requests are re-scheduled
        for req in can_run_list:
            if req.is_retracted:
                self.num_retracted_reqs -= 1
            self._release_l3_prefetch_ready(req)

        for req in can_run_list:
            if req.dequeue_time is None:
                req.dequeue_time = time.time() - req.created_time
            adder.dequeue_durations.append(req.dequeue_time)

        if self.enable_metrics:
            # only record queue time when enable_metrics is True to avoid overhead
            for req in can_run_list:
                req.queue_time_end = time.perf_counter()
                if self.global_rank == 0 and not req.prefill_waiting_recorded:
                    req.add_latency(RequestStage.PREFILL_WAITING)
                    req.prefill_waiting_recorded = True

        self.waiting_queue = [x for x in self.waiting_queue if x not in set(can_run_list)]

        self.chunked_req = adder.chunked_req

        if self.chunked_req:
            self.chunked_req.is_chunked += 1

        if self.attn_tp_rank == 0 and log:
            self.log_prefill_stats(adder, running_bs, self.forward_ct)

        # Create a new batch
        # DFLASH: draft_worker 只在末 PP stage 构造, 但 slot 预留量须全 stage 一致,
        # 否则各 stage 的 KV 水位分叉, 准入的请求数不同。
        if self.spec_algorithm.is_dflash():
            draft_token_num = self.server_args.speculative_num_draft_tokens
            spec_num_steps = self.server_args.speculative_num_steps
        else:
            draft_token_num = self.server_args.speculative_num_draft_tokens if self.draft_worker else 0
            spec_num_steps = self.server_args.speculative_num_steps if self.draft_worker else 0
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

        new_batch.prefill_batch_info = adder.batch_info[old_can_run_list_len:]
        if self.enable_hierarchical_cache:
            # todo：disable cuda graph execution if hicache loading triggered
            old_consumer_idx = self.preparing_schedule_batch.hicache_consumer_index if self.preparing_schedule_batch else -1
            new_batch.hicache_consumer_index = self.tree_cache.ready_to_load_host_cache(old_consumer_idx)
        new_batch.prepare_for_extend()

        if self.draft_worker:
            self.draft_worker.renew_scaling_penalty(new_batch.req_pool_indices)

        # Mixed-style chunked prefill
        if self.is_mixed_chunk and self.running_batch is not None and not (new_batch.return_logprob or self.running_batch.return_logprob):
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

        if not batch.check_decode_mem(self.decode_mem_cache_buf_multiplier) or (TEST_RETRACT and batch.batch_size() > 1):
            old_ratio = self.new_token_ratio

            retracted_reqs, new_token_ratio, reqs_to_abort = batch.retract_decode(self.server_args, self.enable_overlap)
            num_retracted_reqs = len(retracted_reqs)

            # Non-PD disaggregation overlap scheduling: if running Prefill batch but retract a req from Decode batch,
            # this req won't enter post-processing, and is not on GPU, should be released immediately.

            # In PD disaggregation: Prefill instance never retracts, but on Decode instance with overlap scheduling,
            # if retract happens when last batch is new_prebuilt_batch, then the last Decode Batch's post-processing
            # has ended and GPU is actually idle, result queue is empty, retracted req can't enter post-processing,
            # should be released immediately.

            if last_batch and last_batch.forward_mode == ForwardMode.EXTEND and self.enable_overlap:
                for req in retracted_reqs:
                    if (self.disaggregation_mode == DisaggregationMode.DECODE) or (req not in last_batch.reqs):
                        logger.info(f"Release Retracted req rid={req.rid} req_pool_idx={req.req_pool_idx}")
                        self.req_to_token_pool.free(req.req_pool_idx)
                        req.req_pool_idx = None

            self.new_token_ratio = new_token_ratio
            logger.info(
                "Decode out of memory happened. "
                f"#retracted_reqs: {len(retracted_reqs)}, "
                f"#new_token_ratio: {old_ratio:.4f} -> {self.new_token_ratio:.4f}"
            )
            for req in reqs_to_abort:
                self.stream_output([req], req.return_logprob)
            self._extend_requests_to_queue(retracted_reqs, is_retracted=True)
            self.num_retracted_reqs += num_retracted_reqs
            self.total_retracted_reqs += num_retracted_reqs
            self.num_retracted_reqs += num_retracted_reqs
        else:
            self.new_token_ratio = max(
                self.new_token_ratio - self.new_token_ratio_decay,
                self.min_new_token_ratio,
            )

        if batch.batch_size() < initial_bs and self.req_to_token_pool.available_size() >= 0:
            self.batch_is_full = False

        # The last request can be aborted on decode OOM, leaving no requests
        # to prepare for the next decode step.
        if batch.is_empty():
            self.batch_is_full = False
            return None

        # Update batch tensors
        batch.prepare_for_decode()
        return batch

    def run_batch(
        self,
        batch: ScheduleBatch,
    ) -> Union[GenerationBatchResult, EmbeddingBatchResult]:
        """Run a batch."""
        self.forward_ct += 1

        # Whether to run the profiler
        self.profile_batch_predicate(batch)
        if self.forward_sleep_time is not None:
            logger.info(f"Scheduler.run_batch sleep {self.forward_sleep_time}s")
            time.sleep(self.forward_sleep_time)

        self.profile_mfu(batch)
        if batch.return_logprob:
            batch.extend_input_len_per_req = [req.extend_input_len for req in batch.reqs]
            batch.extend_logprob_start_len_per_req = [req.extend_logprob_start_len for req in batch.reqs]
        return self.run_batch_inner(batch)

    def profile_mfu(
        self,
        batch: ScheduleBatch,
    ):
        if self.mfu_runner is not None and self.is_generation and batch.forward_mode.is_extend():
            # prepare_for_extend already computes both values on the CPU and
            # guarantees seq_len == prefix_len + extend_len.  Pulling
            # batch.seq_lens back from the device here introduces a synchronous
            # D2H fence in the scheduler thread.  In PP prefill that fence can
            # wait behind outstanding async P2P sends and stop result draining.
            seq_lens = [prefix_len + extend_len for prefix_len, extend_len in zip(batch.prefix_lens, batch.extend_lens)]
            self._prefill_mfu_queue.append(
                (
                    time.perf_counter(),
                    list(batch.extend_lens) if batch.extend_lens else [],
                    seq_lens,
                )
            )

    def run_batch_inner(
        self,
        batch: ScheduleBatch,
    ) -> Union[GenerationBatchResult, EmbeddingBatchResult]:
        if self.is_generation:
            # 非末 PP stage 的 DFLASH draft_worker=None, 走普通 forward。
            if self.spec_algorithm.is_none() or self.draft_worker is None:
                model_worker_batch = batch.get_model_worker_batch()
                logits_output, next_token_ids, next_token_multi_ids = self.tp_worker.forward_batch_generation(model_worker_batch)
                if not self.enable_overlap:
                    self.req_to_token_pool.verified_lens[model_worker_batch.req_pool_indices] += model_worker_batch.new_tokens_to_compute
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
                    grammar_terminated_list,
                ) = self.draft_worker.forward_batch_speculative_generation(
                    model_worker_batch,
                    launch_done=None,
                )
                next_token_multi_ids = None

                batch.spec_info = EagleDraftOutput(
                    last_verified_ids=None if new_verified_id is None else new_verified_id.clone(),
                    token_list=None if token_list is None else token_list.clone(),
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
                extend_input_len_per_req = batch.extend_input_len_per_req
                extend_logprob_start_len_per_req = batch.extend_logprob_start_len_per_req
            else:
                extend_input_len_per_req = None
                extend_logprob_start_len_per_req = None

            ret = GenerationBatchResult(
                logits_output=logits_output,
                next_token_ids=next_token_ids,
                extend_input_len_per_req=extend_input_len_per_req,
                extend_logprob_start_len_per_req=extend_logprob_start_len_per_req,
                bid=model_worker_batch.bid,
                accept_lengths_cpu=(accept_lengths_cpu if self.draft_worker else [1 for _ in range(len(batch.reqs))]),
                next_token_multi_ids=next_token_multi_ids,
            )
        else:  # embedding or reward model
            model_worker_batch = batch.get_model_worker_batch()
            embeddings = self.tp_worker.forward_batch_embedding(model_worker_batch)
            ret = EmbeddingBatchResult(embeddings=embeddings, bid=model_worker_batch.bid)
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
                num_tokens = local_batch.batch_size() * self.server_args.speculative_num_draft_tokens
        else:
            num_tokens = local_batch.extend_num_tokens

        local_num_tokens = num_tokens
        local_batch_size = local_batch.batch_size() if local_batch is not None else 0
        local_forward_mode = local_batch.forward_mode if local_batch is not None else ForwardMode.IDLE

        local_communication_info = torch.tensor(
            [[local_num_tokens, local_batch_size, local_forward_mode]],
            dtype=torch.int32,
        )
        if global_server_args_dict["npu_disable_all_gather"]:
            global_communication_info = torch.zeros((self.world_size, 3), dtype=torch.int32)
            global_communication_info[:, 0] = 1
            global_communication_info[:, 2] = 4
        else:
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

            all_decode_or_idle = all(mode in (ForwardMode.DECODE, ForwardMode.IDLE, ForwardMode.TARGET_VERIFY) for mode in global_forward_mode)

            local_batch.all_decode_or_idle = all_decode_or_idle

            can_run_tbo = False
            global_batch_size = global_communication_info[:, 1].tolist()
            local_batch.global_batch_size = global_batch_size
            min_batch_size = min(global_batch_size)
            all_prefill = all(mode == ForwardMode.EXTEND for mode in global_forward_mode)
            if (
                self.server_args.enable_tbo
                and all_decode_or_idle
                and min_batch_size >= self.server_args.tbo_min_bs
                or self.server_args.enable_tbo
                and all_prefill
                and min_batch_size >= 2
            ):
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
            self.server_args.num_continuous_decode_steps,
        )
        idle_batch.prepare_for_idle()
        return idle_batch

    def _check_grammar_compilation(self, req):
        """Check if grammar compilation failed and mark request for abort if so."""
        if req.grammar is None:
            req.to_abort_grammar = True
            logger.warning(f"Grammar compilation failed for req {req.rid}, marking for abort")

    def _apply_reasoning_grammar_state(self, req):
        """Initialize a ReasonerGrammarObject's thinking state for this request.

        The grammar is cached by grammar key only, so per-request thinking state
        must be set on the resolved object here. Whether the request is in a
        thinking phase is driven by enable_thinking (the same signal that built
        the prompt): with it on, the prompt ends with an open think block and the
        model generates think_end, so the wrapper starts in THINKING and grammar
        only engages on the post-think content. With it off, the chat template
        already appended think_end to the prompt (the model never generates one),
        so the wrapper must start in GENERATION or grammar would never engage.
        thinking_budget only caps the thinking length; it does not decide state.
        """
        if not isinstance(req.grammar, ReasonerGrammarObject):
            return
        thinking_budget = req.sampling_params.thinking_budget
        req.grammar.max_think_tokens = thinking_budget if thinking_budget is not None else -1
        req.grammar.set_reasoning(bool(req.sampling_params.enable_thinking))

    def move_ready_grammar_requests(self):
        """Move requests whose grammar objects are ready from grammar_queue to waiting_queue."""
        num_ready_reqs = 0
        for req in self.grammar_queue:
            try:
                if req.grammar_done is False:
                    req.grammar = req.grammar.result(timeout=0.001)
                    req.grammar_done = True
                self._check_grammar_compilation(req)
                num_ready_reqs += 1
            except futures._base.TimeoutError:
                break
        if self.attn_tp_pp_size > 1:
            tensor = torch.tensor(num_ready_reqs, dtype=torch.int32)
            torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.MIN, group=self.attn_tp_pp_cpu_group)
            num_ready_reqs = tensor.item()

        for i in range(num_ready_reqs):
            req = self.grammar_queue[i]
            # Cache both success and failure. A failed/degenerate grammar
            # resolves to None (see init_value_impl); caching None lets identical
            # invalid requests hit the cache and be aborted without recompiling.
            grammar = req.grammar
            self.grammar_backend.set_cache(req.grammar_key, grammar.copy() if grammar is not None else None)
            # Apply per-req reasoning state AFTER caching so the cached template
            # stays free of this request's thinking_budget / reasoning state.
            self._apply_reasoning_grammar_state(req)
        self._extend_requests_to_queue(self.grammar_queue[:num_ready_reqs])
        self.grammar_queue = self.grammar_queue[num_ready_reqs:]
        return num_ready_reqs

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
        if self._is_no_request():
            self.cur_batch = None
            self.last_batch = None
            self.tree_cache.reset()
            self.tree_cache_metrics = {"total": 0, "hit": 0}
            if self.grammar_backend:
                self.grammar_backend.reset()
            self.req_to_token_pool.clear()
            self.kv_allocator.clear()

            if not self.spec_algorithm.is_none() and self.draft_worker is not None:
                self.draft_worker.model_runner.req_to_token_pool.clear()
                self.draft_worker.model_runner.kv_allocator.clear()

            self.num_generated_tokens = 0
            self.forward_ct_decode = 0
            self.spec_num_total_accepted_tokens = 0
            self.spec_num_total_forward_ct = 0
            self.cum_spec_accept_length = 0
            self.cum_spec_accept_count = 0

            try:
                from sglang.srt.multimodal.mm_utils import clear_mm_embedding_cache

                clear_mm_embedding_cache()
            except Exception as e:
                logger.warning(f"Failed to clear ViT embedding cache: {e}")

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

        # Add estimated pages only for queues that have not allocated KV yet.
        def get_page_num(input_ids):
            return math.ceil(len(input_ids) / self.server_args.page_size)

        # Decode waiting/transfer pages are already in _get_num_used_pages().
        if self.disaggregation_mode != DisaggregationMode.DECODE:
            num_pages += sum(get_page_num(req.origin_input_ids) for req in self.waiting_queue)
        num_waiting_reqs = len(self.waiting_queue)  # todo p overlap
        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            num_pages += sum(get_page_num(req.origin_input_ids) for req in self.disagg_prefill_bootstrap_queue.queue)
            num_waiting_reqs += len(self.disagg_prefill_bootstrap_queue.queue)
        elif self.disaggregation_mode == DisaggregationMode.DECODE:
            num_pages += sum(get_page_num(req.req.origin_input_ids) for req in self.disagg_decode_prealloc_queue.queue)
            num_waiting_reqs += len(self.disagg_decode_prealloc_queue.queue)
            num_waiting_reqs += len(self.disagg_decode_transfer_queue.queue)
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
            ret["avg_spec_accept_length"] = self.cum_spec_accept_length / self.cum_spec_accept_count

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
        for k, _v in server_args_dict.items():
            if k not in args_allow_update:
                logger.warning(f"Updating {k} is not supported.")
                if_success = False
                break
        if if_success:
            if not self.spec_algorithm.is_none() and self.cum_spec_accept_count > 0:
                avg_spec_accept_length = self.cum_spec_accept_length / self.cum_spec_accept_count
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
        # EPD requests that have not committed their canonical descriptor live
        # outside the normal Prefill/Decode queues.  Every rank removes the same
        # waiting entry because AbortReq uses Scheduler's ordered broadcast.
        epd_req = self._epd_waiting_reqs.pop(recv_req.rid, None)
        if epd_req is not None:
            self._finish_epd_admission_failed(recv_req.rid)
            if self.epd_prefill_receiver is not None:
                self.epd_prefill_receiver.cancel(recv_req.rid)
            logger.debug("Abort EPD waiting request. rid=%s", recv_req.rid)
            return

        # Delete requests in the waiting queue
        to_del = None
        for i, req in enumerate(self.waiting_queue):
            if req.rid == recv_req.rid:
                to_del = i
                break

        if to_del is not None:
            self._release_l3_prefetch_ready(req, detach=True)
            del self.waiting_queue[to_del]
            logger.debug(f"Abort queued request. {req.rid=}")
            return

        # Delete requests in the running batch
        if self.running_batch:
            for req in self.running_batch.reqs:
                if req.rid == recv_req.rid and not req.finished():
                    logger.debug(f"Abort running request. {req.rid=}")
                    req.to_abort = True
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
                    logger.debug(f"Abort disagg decode retracted prealloc req. {req.rid=}")
                    return
            for req in self.disagg_decode_transfer_queue.queue:
                if req.req.rid == recv_req.rid:
                    req.req.to_abort = True
                    logger.debug(f"Abort disagg decode transfer req. {req.req.rid=}")
                    return
        elif self.disaggregation_mode == DisaggregationMode.PREFILL:
            for req in self.disagg_prefill_bootstrap_queue.queue:
                if req.rid == recv_req.rid:
                    self._release_l3_prefetch_ready(req, detach=True)
                    req.to_abort = True
                    logger.debug(f"Abort disagg prefill bootstrap req. {req.rid=}")
                    return
            for req in self.disagg_prefill_inflight_queue:
                if req.rid == recv_req.rid:
                    self._release_l3_prefetch_ready(req, detach=True)
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
    ) -> tuple[bool, str]:
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
        worker = self.draft_worker or self.tp_worker
        success, message = worker.update_weights_from_tensor(recv_req)
        # TODO extract common code b/t update_weights_from_distributed and update_weights_from_tensor later
        if success:
            if recv_req.flush_cache:
                flash_cache_success = self.flush_cache()
                assert flash_cache_success, "Cache flush failed after updating weights"
        else:
            logger.error(message)
        torch.distributed.barrier(group=self.tp_cpu_group)
        return UpdateWeightsFromTensorReqOutput(success, message)

    def get_weights_by_name(self, recv_req: GetWeightsByNameReqInput):
        parameter = self.tp_worker.get_weights_by_name(recv_req)
        return GetWeightsByNameReqOutput(parameter)

    def release_memory_occupation(self, recv_req: ReleaseMemoryOccupationReqInput):
        assert self._is_no_request(), "release_memory_occupation should be called only when no ongoing request."

        tags = recv_req.tags

        if tags is None or len(tags) == 0:
            tags = GPU_MEMORY_ALL_TYPES

        for tag in tags:
            self.offload_tags.add(tag)

        if GPU_MEMORY_TYPE_KV_CACHE in tags:
            self.memory_saver_adapter.pause(GPU_MEMORY_TYPE_KV_CACHE)
            self.flush_cache()

        if GPU_MEMORY_TYPE_WEIGHTS in tags:
            self.stashed_model_static_state = _export_static_state(self.tp_worker.model_runner.model)
            torch.distributed.barrier(self.tp_cpu_group)
            self.memory_saver_adapter.pause(GPU_MEMORY_TYPE_WEIGHTS)

        if GPU_MEMORY_TYPE_CUDA_GRAPH in tags:
            self.memory_saver_adapter.pause(GPU_MEMORY_TYPE_CUDA_GRAPH)

        torch.get_device_module().synchronize()

        return ReleaseMemoryOccupationReqOutput()

    def _is_no_request(self):
        no_request = (
            len(self.waiting_queue) == 0
            and (self.running_batch is None or self.running_batch.is_empty())
            and (self.last_batch is None or self.last_batch.is_empty())
            and (self.cur_batch is None or self.cur_batch.is_empty())
            and (not self.enable_overlap or len(self.result_queue) == 0)
            and self.chunked_req is None
        )
        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            no_request &= len(self.disagg_prefill_bootstrap_queue.queue) == 0 and len(self.disagg_prefill_inflight_queue) == 0
        if self.disaggregation_mode == DisaggregationMode.DECODE:
            no_request &= len(self.disagg_decode_prealloc_queue.queue) == 0 and len(self.disagg_decode_transfer_queue.queue) == 0
        return no_request

    def resume_memory_occupation(self, recv_req: ResumeMemoryOccupationReqInput):
        tags = recv_req.tags

        if tags is None or len(tags) == 0:
            tags = GPU_MEMORY_ALL_TYPES

        for tag in tags:
            self.offload_tags.remove(tag)

        if GPU_MEMORY_TYPE_CUDA_GRAPH in tags:
            self.memory_saver_adapter.resume(GPU_MEMORY_TYPE_CUDA_GRAPH)

        if GPU_MEMORY_TYPE_WEIGHTS in tags:
            self.memory_saver_adapter.resume(GPU_MEMORY_TYPE_WEIGHTS)
            torch.distributed.barrier(self.tp_cpu_group)
            _import_static_state(
                self.tp_worker.model_runner.model,
                self.stashed_model_static_state,
            )
            del self.stashed_model_static_state

        if GPU_MEMORY_TYPE_KV_CACHE in tags:
            self.memory_saver_adapter.resume(GPU_MEMORY_TYPE_KV_CACHE)

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
            self.sessions[session_id] = Session(recv_req.capacity_of_str_len, session_id)
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
    return dict(buffers=[(name, buffer.detach().clone()) for name, buffer in model.named_buffers()])


def _import_static_state(model, static_params):
    self_named_buffers = dict(model.named_buffers())
    for name, tensor in static_params["buffers"]:
        self_named_buffers[name][...] = tensor


def run_scheduler_process(
    server_args: ServerArgs,
    port_args: PortArgs,
    gpu_id: int,
    pp_rank: int,
    attn_tp_rank: int,
    moe_ep_rank: int,
    dp_rank: Optional[int],
    global_rank: int,
    pipe_writer,
):
    from sglang.srt.npu.utils import init_npu

    init_npu()

    # Config the process
    # kill_itself_when_parent_died()  # This is disabled because it does not work for `--dp 2`
    if dp_rank is not None:
        setproctitle.setproctitle(f"sglang::scheduler_{pp_rank}_{dp_rank}_{attn_tp_rank}")
    else:
        setproctitle.setproctitle(f"sglang::scheduler_{pp_rank}_{attn_tp_rank}")
    # faulthandler.enable()
    parent_process = psutil.Process().parent()
    register_usr_signal()

    # [For Router] if env var "SGLANG_DP_RANK" exist, set dp_rank to the value of the env var
    if dp_rank is None and "SGLANG_DP_RANK" in os.environ:
        dp_rank = int(os.environ["SGLANG_DP_RANK"])

    # Configure the logger
    pp_size = server_args.pp_size
    if pp_size == 1:
        prefix = f" ATTN TP RANK {attn_tp_rank}" if dp_rank is None else f" DP_RANK {dp_rank} ATTN_TP_RANK {attn_tp_rank}"
    else:
        prefix = f" PP {pp_rank} ATTN TP RANK {attn_tp_rank}" if dp_rank is None else f" PP {pp_rank} DP_RANK {dp_rank} ATTN_TP_RANK {attn_tp_rank}"
    configure_logger(server_args, prefix=prefix)
    suppress_other_loggers()

    # Set cpu affinity to this gpu process
    if get_bool_env_var("SGLANG_SET_CPU_AFFINITY"):
        set_gpu_proc_affinity(server_args.world_size, server_args.nnodes, gpu_id)

    # Create a scheduler and run the event loop
    try:
        scheduler = Scheduler(server_args, port_args, gpu_id, pp_rank, attn_tp_rank, moe_ep_rank, dp_rank, global_rank)
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
            if scheduler.pp_size == 1:
                if scheduler.enable_overlap:
                    scheduler.event_loop_overlap_disagg_prefill()
                else:
                    scheduler.event_loop_normal_disagg_prefill()
            else:
                scheduler.event_loop_pp_disagg_prefill()
        elif disaggregation_mode == DisaggregationMode.DECODE:
            if scheduler.enable_overlap:
                if scheduler.server_args.num_continuous_decode_steps > 1:
                    scheduler.event_loop_overlap_disagg_multi_step_decode()
                else:
                    scheduler.event_loop_overlap_disagg_decode()
            else:
                scheduler.event_loop_normal_disagg_decode()

    except KeyboardInterrupt:
        logger.info("Scheduler received KeyboardInterrupt, exiting.")
    except Exception:
        traceback = get_exception_traceback()
        logger.error(f"Scheduler hit an exception: {traceback}")
        parent_process.send_signal(signal.SIGUSR1)
