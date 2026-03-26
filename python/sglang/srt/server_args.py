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
"""The arguments of the server."""

import argparse
import dataclasses
import os
import random
from typing import List, Optional, Literal

import torch

from sglang.srt.parser.reasoning_parser import ReasoningParser
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.srt.distributed.parallel_strategy import (
    AttnParallelStrategy,
    DenseParallelStategy,
    MoeParallelStrategy,
)
from sglang.srt.utils import (
    get_amdgpu_memory_capacity,
    get_colorful_logger,
    get_hpu_memory_capacity,
    get_nvgpu_memory_capacity,
    is_flashinfer_available,
    is_hip,
    is_npu,
    is_port_available,
    is_valid_ipv6_address,
    maybe_model_redirect,
    nullable_str,
)

logger = get_colorful_logger(__name__)

RADIX_EVICTION_POLICY_CHOICES = ["lru", "lfu"]

@dataclasses.dataclass
class ServerArgs:
    # Model and tokenizer
    model_path: str
    tokenizer_path: Optional[str] = None
    tokenizer_mode: str = "auto"
    skip_tokenizer_init: bool = False
    load_format: str = "auto"
    trust_remote_code: bool = True
    dtype: str = "auto"
    kv_cache_dtype: str = "auto"
    kv_cache_quant_method: str = "none"
    quantization: Optional[str] = None
    quantization_param_path: nullable_str = None
    context_length: Optional[int] = None
    device: str = "cuda"
    served_model_name: Optional[str] = None
    chat_template: Optional[str] = None
    completion_template: Optional[str] = None
    is_embedding: bool = False
    revision: Optional[str] = None
    think_end_token: Optional[str] = None

    # Port for the HTTP server
    host: str = "127.0.0.1"
    port: int = 30000

    # Memory and scheduling
    mem_fraction_static: Optional[float] = None
    max_running_requests: Optional[int] = None
    max_total_tokens: Optional[int] = None
    chunked_prefill_size: Optional[int] = None
    max_prefill_tokens: int = 16384
    schedule_policy: str = "fcfs"
    schedule_conservativeness: float = 1.0
    cpu_offload_gb: int = 0
    page_size: int = 64
    # special kv cache
    max_mamba_cache_size: Optional[int] = None
    mamba_ssm_dtype: str = "float32"
    disable_hybrid_swa_memory: bool = False

    # Other runtime options
    stream_interval: int = 1
    stream_output: bool = False
    random_seed: Optional[int] = None
    constrained_json_whitespace_pattern: Optional[str] = None
    watchdog_timeout: float = 300
    dist_timeout: Optional[int] = None  # timeout for torch.distributed
    download_dir: Optional[str] = None
    # Used for customizing extensible models
    ext_yaml: Optional[str] = None
    base_gpu_id: int = 0
    gpu_id_step: int = 1

    # Logging
    log_level: str = "info"
    log_level_http: Optional[str] = None
    log_requests: bool = False
    log_requests_level: int = 0
    show_time_cost: bool = False
    enable_metrics: bool = False
    decode_log_interval: int = 40
    enable_request_time_stats_logging: bool = False
    kv_events_config: Optional[str] = None
    enable_trace: bool = False
    otlp_traces_endpoint: str = "localhost:4317"
    metrics_reporters: Optional[List[str]] = None
    app_key: Optional[str] = None

    # API related
    api_key: Optional[str] = None
    file_storage_path: str = "sglang_storage"
    enable_cache_report: bool = False

    # Data parallelism
    dp_size: int = 1
    dp_spmd_mode: bool = False
    load_balance_method: str = "shortest_queue"
    load_watch_interval: float = 0.02

    # Expert parallelism
    ep_size: int = 1
    init_expert_location: str = "trivial"
    ep_num_redundant_experts: int = 0
    ep_dispatch_algorithm: Optional[Literal["static", "dynamic", "fake", "static_with_zero_expert", "dynamic_with_zero_expert"]] = None
    eplb_algorithm: str = "auto"
    eplb_rebalance_num_iterations: int = 1000
    eplb_rebalance_layers_per_chunk: Optional[int] = None
    expert_distribution_recorder_mode: Optional[
        Literal["stat", "stat_approx", "per_pass", "per_token"]
    ] = None
    expert_distribution_recorder_buffer_size: Optional[int] = None
    enable_expert_distribution_metrics: bool = False
    enable_eplb: bool = False

    # Hierarchical cache
    radix_eviction_policy: str = "lru"
    enable_hierarchical_cache: bool = False
    hicache_ratio: float = 2.0
    hicache_size: int = 0
    hicache_write_policy: str = "write_through"
    hicache_io_backend: str = "kernel"
    hicache_mem_layout: str = "layer_first"
    hicache_storage_backend: Optional[str] = None
    hicache_storage_prefetch_policy: str = "best_effort"
    hicache_storage_backend_extra_config: Optional[str] = None

    # Multi-node distributed serving
    dist_init_addr: Optional[str] = None
    nnodes: int = 1
    node_rank: int = 0

    # Model override args in JSON
    json_model_override_args: str = "{}"
    preferred_sampling_params: Optional[str] = None

    # Kernel backend
    attention_backend: Optional[str] = None
    drafter_attention_backend: Optional[str] = None
    chunker_backend: Optional[str] = None
    sampling_backend: Optional[str] = None
    grammar_backend: Optional[str] = None

    # Speculative decoding
    capture_sample_graph: Optional[bool] = False
    draft_model_path_use_base: Optional[bool] = False
    speculative_algorithm: Optional[str] = None
    speculative_draft_model_path: Optional[str] = None
    speculative_num_steps: int = 5
    speculative_eagle_topk: int = 4
    speculative_num_draft_tokens: int = 8
    speculative_accept_threshold_single: float = 1.0
    speculative_accept_threshold_acc: float = 1.0
    speculative_token_map: Optional[str] = None
    prompt_lookup_min: int = 0
    prompt_lookup_max: int = 0
    eagle3_layers_to_capture: Optional[str] = None

    # Optimization/debug options
    disable_pdl: bool = False
    disable_radix_cache: bool = False
    disable_cuda_graph: bool = False
    disable_cuda_graph_padding: bool = False
    enable_cudagraph_gc: bool = False
    enable_nccl_nvls: bool = False
    enable_symm_mem: bool = False
    disable_outlines_disk_cache: bool = False
    disable_custom_all_reduce: bool = False
    disable_overlap_schedule: bool = False
    enable_mixed_chunk: bool = False
    enable_dp_attention: bool = False
    enable_ep_moe: bool = False
    enable_deep_ep: bool = False
    force_deterministic_rsag: bool = False
    low_latency_max_num_tokens_per_gpu: int = 256
    enable_torch_compile: bool = False
    torch_compile_max_bs: int = 32
    cuda_graph_max_bs: Optional[int] = None
    disable_prefill_graph: Optional[bool] = False
    prefill_graph_max_tokens: Optional[int] = 128
    prefill_graph_max_bs: Optional[int] = 4
    cuda_graph_bs: Optional[List[int]] = None
    torchao_config: str = ""
    enable_nan_detection: bool = False
    enable_p2p_check: bool = False
    triton_attention_reduce_in_fp32: bool = False
    triton_attention_num_kv_splits: int = 8
    num_continuous_decode_steps: int = 1
    delete_ckpt_after_loading: bool = False
    enable_memory_saver: bool = False
    allow_auto_truncate: bool = False
    enable_custom_logit_processor: bool = False
    tool_call_parser: str = None
    reasoning_parser: Optional[str] = None
    enable_flashinfer_mla: bool = False
    flashinfer_mla_disable_ragged: bool = False
    warmups: Optional[str] = None

    # Debug tensor dumps
    debug_tensor_dump_output_folder: Optional[str] = None
    debug_tensor_dump_input_file: Optional[str] = None
    debug_tensor_dump_inject: bool = False

    # parallel strategy
    nprocs_per_node: int = 1
    world_size: int = 1
    attn_tp_size: int = 1
    dense_tp_size: int = -1
    moe_parallel_strategy: str = "tp"
    attn_parallel_strategy: str = "tp"
    dense_parallel_strategy: str = "tp"

    enable_tbo: Optional[bool] = False
    enable_sbo: Optional[bool] = False
    tbo_min_bs: int = 64
    mla_max_chunk_capacity: int = 16 * 1024
    mm_mode: str = "none"

    # For PD disaggregation: can be "null" (not disaggregated), "prefill" (prefill-only), or "decode" (decode-only)
    disaggregation_mode: str = "null"
    disaggregation_bootstrap_port: int = 8998
    disaggregation_transfer_backend: str = "mooncake"
    disaggregation_ib_device: Optional[str] = None
    disaggregation_layerwise_interval: int = 1
    disaggregation_transfer_hidden_states_max_size: int = 0
    pdlb_url: Optional[str] = None

    # For tool server
    tool_server: Optional[str] = None

    skip_server_warmup: bool = False

    # For NPU
    npu_enable_weight_nz: bool = False
    npu_enable_mc2: bool = False
    npu_enable_mlp_matmul: bool = False
    npu_enable_opt_rope: bool = False

    request_max_input_len: int = 0
    request_max_output_len: int = 0
    request_cache_size: int = 0
    request_cache_config: str = "{}"
    
    # For flashinfer reduce norm fusion
    flashinfer_comm_max_num_tokens: int = 64

    def __post_init__(self):
        self.model_path = maybe_model_redirect(self.model_path)
        self.world_size = self.nprocs_per_node * self.nnodes

        # Set missing default values
        if self.tokenizer_path is None:
            self.tokenizer_path = self.model_path

        if self.served_model_name is None:
            self.served_model_name = self.model_path

        if self.random_seed is None:
            self.random_seed = random.randint(0, 1 << 30)

        if is_hip():
            gpu_mem = get_amdgpu_memory_capacity()
        elif not is_npu() and torch.cuda.is_available():
            # TODO for npu
            gpu_mem = get_nvgpu_memory_capacity()
        elif self.device == "hpu":
            gpu_mem = get_hpu_memory_capacity()
        else:
            # GPU memory is not known yet or no GPU is available.
            gpu_mem = None

        # Set mem fraction static, which depends on the tensor parallelism size
        if self.mem_fraction_static is None:
            if self.world_size >= 16:
                self.mem_fraction_static = 0.79
            elif self.world_size >= 8:
                self.mem_fraction_static = 0.81
            elif self.world_size >= 4:
                self.mem_fraction_static = 0.85
            elif self.world_size >= 2:
                self.mem_fraction_static = 0.87
            else:
                self.mem_fraction_static = 0.88

        # Set chunked prefill size, which depends on the gpu memory capacity
        if self.chunked_prefill_size is None:
            if gpu_mem is not None and gpu_mem < 25_000:
                self.chunked_prefill_size = 2048
            else:
                self.chunked_prefill_size = 8192

        # Set cuda graph max batch size
        if self.cuda_graph_max_bs is None:
            # Based on detailed statistics, when serving TP1/TP2 models on lower-end GPUs with HBM<25G, you can either disable cuda graph or set `cuda_graph_max_bs` to a very small value to reduce the memory overhead of creating cuda graphs, with almost no impact on performance. However, when serving models with TP4 or TP8, we need to enable cuda graph to maintain high performance. In this case, we can set `cuda_graph_max_bs` to 80 (half of the default value 160) to reduce the memory overhead of creating cuda graphs. Looking at the logs from TP4 serving of qwen2-72b, a value of 80 is sufficient and can reduce the memory overhead of creating cuda graphs on lower-end GPUs compared to the original 160, avoiding OOM issues.
            if gpu_mem is not None and gpu_mem < 25_000:
                if self.world_size < 4:
                    self.cuda_graph_max_bs = 8
                else:
                    self.cuda_graph_max_bs = 80
            else:
                self.cuda_graph_max_bs = 160

        # Choose kernel backends
        if self.device == "hpu":
            self.attention_backend = "torch_native"
            self.sampling_backend = "pytorch"

        if self.attention_backend is None:
            self.attention_backend = (
                "flashinfer" if is_flashinfer_available() else "triton"
            )
        if self.sampling_backend is None:
            self.sampling_backend = (
                "flashinfer" if is_flashinfer_available() else "pytorch"
            )

        if self.attention_backend == "torch_native":
            logger.warning(
                "Cuda graph is disabled because of using torch native attention backend"
            )
            self.disable_cuda_graph = True

        if self.chunker_backend is None:
            self.chunker_backend = "fa3"

        assert (
            self.dp_size * self.attn_tp_size == self.world_size
        ), f"{self.dp_size} * {self.attn_tp_size} == {self.world_size}"

        assert self.max_running_requests >= self.dp_size, f"{self.max_running_requests=} < {self.dp_size=}"

        FLLM_IS_CP = os.environ.get("FLLM_IS_CP", "0") in ["True", "true", "1"]
        assert not FLLM_IS_CP or self.max_running_requests == 1, "FLLM DSA CP enable, only support batchsize=1"

        if self.dp_size == self.world_size:
            self.enable_dp_attention = True
            self.attn_parallel_strategy = AttnParallelStrategy.DATA_PARALLEL
        else:
            self.attn_parallel_strategy = AttnParallelStrategy.TENSOR_PARALLEL
            if self.attn_tp_size == self.world_size:
                self.enable_dp_attention = False
            else:
                self.enable_dp_attention = True

        # Currently equals world_size
        self.tp_size = self.attn_tp_size * self.dp_size
        # dense parallel
        if self.world_size == 1:
            self.dense_parallel_strategy = DenseParallelStategy.REPLICATED

        if self.dense_parallel_strategy == DenseParallelStategy.REPLICATED:
            self.dense_tp_size = 1
        else:
            # Default dense_tp_size == attn_tp_size, if dense_tp_size is 1, then modify to nprocs_per_node and raise a warning
            if self.dense_tp_size <= 0:
                self.dense_tp_size = self.attn_tp_size
            if self.dense_tp_size == 1:
                # attn is pure dp
                self.dense_tp_size = self.nprocs_per_node
                logger.warning(
                    f"TP dense is enabled. The dense tp size must be greater than 1, but got dense_tp_size={self.dense_tp_size}." + \
                    f"The final dense tp size will be changed to {self.nprocs_per_node}."
                )
            else:
                if self.attn_tp_size > 1:
                    # attn is dp tp combination, dense dp tp must be the same as attn
                    assert self.dense_tp_size == self.attn_tp_size, \
                        f"When attn is not only dp," + \
                        f"the dp tp of dense must be the same as that of attn," + \
                        f"does not support the hybrid communication mode for the time being:" + \
                        f"{self.dense_tp_size=} | {self.attn_tp_size=}"

        if self.attn_parallel_strategy == AttnParallelStrategy.DATA_PARALLEL or self.attn_tp_size != self.world_size:
            # Originally divided by 2, not sure why yet, leave it for now
            # chunked_prefill_size refers to the maximum number of tokens for a single dp scheduler to do prefill
            # self.chunked_prefill_size = self.chunked_prefill_size // 2
            assert self.chunked_prefill_size <= self.max_prefill_tokens
            self.schedule_conservativeness = self.schedule_conservativeness * 0.3
            logger.warning(
                f"DP attention is enabled. The chunked prefill size is adjusted to {self.chunked_prefill_size} to avoid MoE kernel issues. "
                f"The schedule conservativeness is adjusted to {self.schedule_conservativeness}."
            )

        # Handle Hicache settings.
        self._handle_hicache()
        self._handle_cache_compatibility()
        if self.moe_parallel_strategy == MoeParallelStrategy.EXPERT_PARALLEL:
            self.enable_ep_moe = True
        elif self.moe_parallel_strategy == MoeParallelStrategy.TENSOR_PARALLEL:
            self.enable_ep_moe = False
        else:
            print(f"self.moe_parallel_strategy: {self.moe_parallel_strategy}")
            print(
                f"MoeParallelStrategy.EXPERT_PARALLEL:{MoeParallelStrategy.EXPERT_PARALLEL}"
            )
            raise NotImplementedError("Oh no")

        # Expert parallelism
        if self.enable_ep_moe:
            self.ep_size = self.world_size
            logger.info(
                f"EP MoE is enabled. The expert parallel size is adjusted to be the same as the world size[{self.world_size}]."
            )

        if self.enable_eplb and (self.expert_distribution_recorder_mode is None):
            self.expert_distribution_recorder_mode = "stat"
            logger.info(
                "EPLB is enabled. The expert_distribution_recorder_mode is automatically set."
            )

        if (self.enable_eplb or (self.init_expert_location is not None)) and (
            self.ep_dispatch_algorithm is None
        ):
            self.ep_dispatch_algorithm = "static"
            logger.info(
                "EPLB is enabled or init_expert_location is provided. ep_dispatch_algorithm is configured."
            )

        logger.info(
            f"Model layout configs: world_size: {self.world_size} attn_tp: {self.attn_tp_size} dp: {self.dp_size} ep: {self.ep_size}"
        )

        os.environ["SGLANG_ENABLE_TORCH_COMPILE"] = (
            "1" if self.enable_torch_compile else "0"
        )
        os.environ["SGLANG_MAMBA_SSM_DTYPE"] = self.mamba_ssm_dtype

        # Speculative Decoding
        if self.speculative_algorithm == "NEXTN":
            # NEXTN shares the same implementation of EAGLE
            self.speculative_algorithm = "EAGLE"

        if self.draft_model_path_use_base:
            self.speculative_draft_model_path = self.model_path

        if self.speculative_draft_model_path == self.model_path:
            self.draft_model_path_use_base = True

        if self.eagle3_layers_to_capture is not None:
            self.eagle3_layers_to_capture = [int(x) for x in self.eagle3_layers_to_capture.split(",")]

        # PLD parameter validation
        if self.speculative_algorithm == "PLD":
            expected_steps = self.speculative_num_draft_tokens - 1
            self.speculative_eagle_topk = 1
            if self.speculative_num_steps != expected_steps:
                logger.warning(
                    f"PLD requires speculative_num_steps = speculative_num_draft_tokens - 1. "
                    f"Adjusting speculative_num_steps from {self.speculative_num_steps} to {expected_steps}."
                )
                self.speculative_num_steps = expected_steps

        # AMD-specific Triton attention KV splits default number
        if is_hip():
            self.triton_attention_num_kv_splits = 16

        if self.attn_tp_size != self.dense_tp_size:
            self.flashinfer_comm_max_num_tokens = -1
            logger.info(f"flashinfer_comm_fusion is forbidden due to different attn_tp_size: {self.attn_tp_size} and dense_tp_size: {self.dense_tp_size}!")
        elif self.enable_sbo:
            if self.flashinfer_comm_max_num_tokens > self.low_latency_max_num_tokens_per_gpu:
                self.flashinfer_comm_max_num_tokens = self.low_latency_max_num_tokens_per_gpu
                logger.info("self.flashinfer_comm_max_num_tokens has been changed to low_latency_max_num_tokens_per_gpu!")

        if self.enable_sbo:
            assert self.dp_size == 1, "dp not supported yet for sbo"

        # PD disaggregation
        if self.disaggregation_mode == "prefill":
            self.disable_cuda_graph = True
            logger.warning("Cuda graph is disabled for prefill server")
        elif self.disaggregation_mode == "decode":
            # Enable RadixCache for decode server to support prefix sharing
            self.disable_radix_cache = self.disable_radix_cache
            logger.info(f"{self.disable_radix_cache=} for decode server to support prefix sharing")
            
        if self.attention_backend != "flashinfer":
            self.disable_prefill_graph = True
            logger.warning("Prefill graph is disabled for non-flashinfer backend")
        elif not self.dp_size == 1:
            self.disable_prefill_graph = True
            logger.warning("Prefill graph is not supported for DP Server currently")
        elif self.disaggregation_mode == "decode":
            self.disable_prefill_graph = True
            logger.warning("Prefill graph is disabled for decode server")

        if self.disaggregation_mode == "prefill" and self.load_balance_method != "round_robin":
            assert self.dp_size == 1, (f"Not Supported when {self.disaggregation_mode=} {self.load_balance_method=} {self.dp_size=}")
            


    def _handle_hicache(self):
        if self.hicache_storage_backend == "mooncake":
            if self.hicache_mem_layout == "layer_first":
                if self.hicache_io_backend == "direct":
                    self.hicache_mem_layout = "page_first_direct"
                elif self.hicache_io_backend == "kernel":
                    self.hicache_mem_layout = "page_first"
                logger.warning(
                    f"Mooncake storage backend does not support layer_first layout, "
                    f"switching to {self.hicache_mem_layout} layout for {self.hicache_io_backend} io backend"
                )

        if self.hicache_mem_layout == "page_first_direct":
            if self.hicache_io_backend != "direct":
                self.hicache_io_backend = "direct"
                logger.warning(
                    "Page first direct layout only support direct io backend"
                )

        # if self.enable_hierarchical_cache and self.hicache_io_backend == "kernel":
        #     # fix for the compatibility issue with FlashAttention3 decoding and HiCache kernel backend
        #     if self.decode_attention_backend is None:
        #         if not self.use_mla_backend():
        #             self.decode_attention_backend = (
        #                 "flashinfer" if is_flashinfer_available() else "triton"
        #             )
        #         else:
        #             self.decode_attention_backend = (
        #                 "flashinfer" if is_sm100_supported() else "triton"
        #             )
        #     elif self.decode_attention_backend == "fa3":
        #         self.hicache_io_backend = "direct"
        #         logger.warning(
        #             "FlashAttention3 decode backend is not compatible with hierarchical cache. "
        #             "Setting hicache_io_backend to vanilla I/O, which may lead to suboptimal performance with small page sizes."
        #         )

        # Below are the only parameters currently supported on Ascend
        if self.enable_hierarchical_cache and is_npu():
            # FIXME(iforgetmyname) fix decode_attention_backend on ascend
            self.decode_attention_backend = "ascend"
            self.hicache_io_backend = "kernel_ascend"
            if self.use_mla_backend():
                self.hicache_mem_layout = "page_first_kv_split"
            else:
                self.hicache_mem_layout = "page_first_direct"
            logger.warning(
                f"Ascend NPU Platform detected, change `hicache_io_backend` to `kernel_ascend` and "
                f"`hicache_mem_layout` to `{self.hicache_mem_layout}`"
            )

    def _handle_cache_compatibility(self):
        if self.enable_hierarchical_cache and self.disable_radix_cache:
            raise ValueError(
                "The arguments enable-hierarchical-cache and disable-radix-cache are mutually exclusive "
                "and cannot be used at the same time. Please use only one of them."
            )
    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser):
        # Model and port args
        parser.add_argument(
            "--model-path",
            type=str,
            help="The path of the model weights. This can be a local folder or a Hugging Face repo ID.",
            required=True,
        )
        parser.add_argument(
            "--tokenizer-path",
            type=str,
            default=ServerArgs.tokenizer_path,
            help="The path of the tokenizer.",
        )
        parser.add_argument(
            "--host", type=str, default=ServerArgs.host, help="The host of the server."
        )
        parser.add_argument(
            "--port", type=int, default=ServerArgs.port, help="The port of the server."
        )
        parser.add_argument(
            "--tokenizer-mode",
            type=str,
            default=ServerArgs.tokenizer_mode,
            choices=["auto", "slow"],
            help="Tokenizer mode. 'auto' will use the fast "
            "tokenizer if available, and 'slow' will "
            "always use the slow tokenizer.",
        )
        parser.add_argument(
            "--skip-tokenizer-init",
            action="store_true",
            help="If set, skip init tokenizer and pass input_ids in generate request",
        )
        parser.add_argument("--ext-yaml", type=str, default=None)
        parser.add_argument(
            "--load-format",
            type=str,
            default=ServerArgs.load_format,
            choices=[
                "auto",
                "pt",
                "safetensors",
                "npcache",
                "dummy",
                "bitsandbytes",
                "layered",
                "extensible",
            ],
            help="The format of the model weights to load. "
            '"auto" will try to load the weights in the safetensors format '
            "and fall back to the pytorch bin format if safetensors format "
            "is not available. "
            '"pt" will load the weights in the pytorch bin format. '
            '"safetensors" will load the weights in the safetensors format. '
            '"npcache" will load the weights in pytorch format and store '
            "a numpy cache to speed up the loading. "
            '"dummy" will initialize the weights with random values, '
            "which is mainly for profiling."
            '"bitsandbytes" will load the weights using bitsandbytes '
            "quantization."
            '"layered" loads weights layer by layer so that one can quantize a '
            "layer before loading another to make the peak memory envelope "
            "smaller.",
        )
        parser.add_argument(
            "--trust-remote-code",
            action="store_true",
            help="Whether or not to allow for custom models defined on the Hub in their own modeling files.",
        )
        parser.add_argument(
            "--dtype",
            type=str,
            default=ServerArgs.dtype,
            choices=["auto", "half", "float16", "bfloat16", "float", "float32"],
            help="Data type for model weights and activations.\n\n"
            '* "auto" will use FP16 precision for FP32 and FP16 models, and '
            "BF16 precision for BF16 models.\n"
            '* "half" for FP16. Recommended for AWQ quantization.\n'
            '* "float16" is the same as "half".\n'
            '* "bfloat16" for a balance between precision and range.\n'
            '* "float" is shorthand for FP32 precision.\n'
            '* "float32" for FP32 precision.',
        )
        parser.add_argument(
            "--kv-cache-dtype",
            type=str,
            default=ServerArgs.kv_cache_dtype,
            choices=["auto", "fp8_e5m2", "fp8_e4m3"],
            help='Data type for kv cache storage. "auto" will use model data type. "fp8_e5m2" and "fp8_e4m3" is supported for CUDA 11.8+.',
        )
        parser.add_argument(
            "--kv-cache-quant-method",
            type=str,
            default=ServerArgs.kv_cache_quant_method,
            choices=["none", "per_token_head"],
            help='kv cache quant method',
        )
        parser.add_argument(
            "--quantization",
            type=str,
            default=ServerArgs.quantization,
            choices=[
                "awq",
                "fp8",
                "gptq",
                "marlin",
                "gptq_marlin",
                "awq_marlin",
                "bitsandbytes",
                "modelopt",
                "w8a8_int8",
                "w8a8_fp8",
                "compressed-tensors",
            ],
            help="The quantization method.",
        )
        parser.add_argument(
            "--quantization-param-path",
            type=nullable_str,
            default=None,
            help="Path to the JSON file containing the KV cache "
            "scaling factors. This should generally be supplied, when "
            "KV cache dtype is FP8. Otherwise, KV cache scaling factors "
            "default to 1.0, which may cause accuracy issues. ",
        )
        parser.add_argument(
            "--context-length",
            type=int,
            default=ServerArgs.context_length,
            help="The model's maximum context length. Defaults to None (will use the value from the model's config.json instead).",
        )
        parser.add_argument(
            "--device",
            type=str,
            default="cuda",
            choices=["cuda", "xpu", "hpu", "cpu", "npu"],
            help="The device type.",
        )
        parser.add_argument(
            "--served-model-name",
            type=str,
            default=ServerArgs.served_model_name,
            help="Override the model name returned by the v1/models endpoint in OpenAI API server.",
        )
        parser.add_argument(
            "--chat-template",
            type=str,
            default=ServerArgs.chat_template,
            help="The buliltin chat template name or the path of the chat template file. This is only used for OpenAI-compatible API server.",
        )
        parser.add_argument(
            "--completion-template",
            type=str,
            default=ServerArgs.completion_template,
            help="The buliltin completion template name or the path of the completion template file. This is only used for OpenAI-compatible API server. only for code completion currently.",
        )
        parser.add_argument(
            "--is-embedding",
            action="store_true",
            help="Whether to use a CausalLM as an embedding model.",
        )
        parser.add_argument(
            "--dp-spmd-mode",
            action="store_true",
            help="Whether to use spmd mode for dp.",
        )
        parser.add_argument(
            "--revision",
            type=str,
            default=None,
            help="The specific model version to use. It can be a branch "
            "name, a tag name, or a commit id. If unspecified, will use "
            "the default version.",
        )
        parser.add_argument(
            "--think-end-token",
            type=str,
            default=ServerArgs.think_end_token,
            help="The think end token of a thinking model, such as '</think>' for DeepSeek R1.",
        )
        # Memory and scheduling
        parser.add_argument(
            "--mem-fraction-static",
            type=float,
            default=ServerArgs.mem_fraction_static,
            help="The fraction of the memory used for static allocation (model weights and KV cache memory pool). Use a smaller value if you see out-of-memory errors.",
        )
        parser.add_argument(
            "--max-running-requests",
            type=int,
            default=ServerArgs.max_running_requests,
            help="The maximum number of running requests.",
        )
        parser.add_argument(
            "--max-total-tokens",
            type=int,
            default=ServerArgs.max_total_tokens,
            help="The maximum number of tokens in the memory pool. If not specified, it will be automatically calculated based on the memory usage fraction. "
            "This option is typically used for development and debugging purposes.",
        )
        parser.add_argument(
            "--chunked-prefill-size",
            type=int,
            default=ServerArgs.chunked_prefill_size,
            help="The maximum number of tokens in a chunk for the chunked prefill. Setting this to -1 means disabling chunked prefill",
        )
        parser.add_argument(
            "--page-size",
            type=int,
            default=ServerArgs.page_size,
        )

        # Hierarchical cache
        parser.add_argument(
            "--enable-hierarchical-cache",
            action="store_true",
            help="Enable hierarchical cache",
        )
        parser.add_argument(
            "--hicache-ratio",
            type=float,
            default=ServerArgs.hicache_ratio,
            help="The ratio of the size of host KV cache memory pool to the size of device pool.",
        )
        parser.add_argument(
            "--hicache-size",
            type=int,
            default=ServerArgs.hicache_size,
            help="The size of host KV cache memory pool in gigabytes, which will override the hicache_ratio if set.",
        )
        parser.add_argument(
            "--hicache-write-policy",
            type=str,
            choices=["write_back", "write_through", "write_through_selective"],
            default=ServerArgs.hicache_write_policy,
            help="The write policy of hierarchical cache.",
        )
        parser.add_argument(
            "--hicache-io-backend",
            type=str,
            choices=["direct", "kernel", "kernel_ascend"],
            default=ServerArgs.hicache_io_backend,
            help="The IO backend for KV cache transfer between CPU and GPU",
        )
        parser.add_argument(
            "--hicache-mem-layout",
            type=str,
            choices=[
                "layer_first",
                "page_first",
                "page_first_direct",
                "page_first_kv_split",
                "page_head",
            ],
            default=ServerArgs.hicache_mem_layout,
            help="The layout of host memory pool for hierarchical cache.",
        )
        parser.add_argument(
            "--hicache-storage-backend",
            type=str,
            choices=["file", "mooncake"],
            default=ServerArgs.hicache_storage_backend,
            help="The storage backend for hierarchical KV cache. "
            "Built-in backends: file, mooncake "
            "For dynamic backend, use --hicache-storage-backend-extra-config to specify: "
            "backend_name (custom name), module_path (Python module path), class_name (backend class name).",
        )
        parser.add_argument(
            "--hicache-storage-prefetch-policy",
            type=str,
            choices=["best_effort", "wait_complete", "timeout"],
            default=ServerArgs.hicache_storage_prefetch_policy,
            help="Control when prefetching from the storage backend should stop.",
        )
        parser.add_argument(
            "--hicache-storage-backend-extra-config",
            type=str,
            default=ServerArgs.hicache_storage_backend_extra_config,
            help="A dictionary in JSON string format containing extra configuration for the storage backend.",
        )
        # Mamba Cache
        parser.add_argument(
            "--max-mamba-cache-size",
            type=int,
            default=ServerArgs.max_mamba_cache_size,
            help="It is used for mamba cache memory static allocation.",
        )
        parser.add_argument(
            "--mamba-ssm-dtype",
            type=str,
            default=ServerArgs.mamba_ssm_dtype,
            choices=["float32", "bfloat16"],
            help="It is used to tune mamba ssm dtype",
        )

        parser.add_argument(
            "--max-prefill-tokens",
            type=int,
            default=ServerArgs.max_prefill_tokens,
            help="The maximum number of tokens in a prefill batch. The real bound will be the maximum of this value and the model's maximum context length.",
        )
        parser.add_argument(
            "--schedule-policy",
            type=str,
            default=ServerArgs.schedule_policy,
            choices=["lpm", "random", "fcfs", "dfs-weight"],
            help="The scheduling policy of the requests.",
        )
        parser.add_argument(
            "--schedule-conservativeness",
            type=float,
            default=ServerArgs.schedule_conservativeness,
            help="How conservative the schedule policy is. A larger value means more conservative scheduling. Use a larger value if you see requests being retracted frequently.",
        )
        parser.add_argument(
            "--cpu-offload-gb",
            type=int,
            default=ServerArgs.cpu_offload_gb,
            help="How many GBs of RAM to reserve for CPU offloading",
        )

        # Other runtime options
        parser.add_argument(
            "--stream-interval",
            type=int,
            default=ServerArgs.stream_interval,
            help="The interval (or buffer size) for streaming in terms of the token length. A smaller value makes streaming smoother, while a larger value makes the throughput higher",
        )
        parser.add_argument(
            "--stream-output",
            action="store_true",
            help="Whether to output as a sequence of disjoint segments.",
        )
        parser.add_argument(
            "--random-seed",
            type=int,
            default=ServerArgs.random_seed,
            help="The random seed.",
        )
        parser.add_argument(
            "--constrained-json-whitespace-pattern",
            type=str,
            default=ServerArgs.constrained_json_whitespace_pattern,
            help=r"Regex pattern for syntactic whitespaces allowed in JSON constrained output. For example, to allow the model generate consecutive whitespaces, set the pattern to [\n\t ]*",
        )
        parser.add_argument(
            "--watchdog-timeout",
            type=float,
            default=ServerArgs.watchdog_timeout,
            help="Set watchdog timeout in seconds. If a forward batch takes longer than this, the server will crash to prevent hanging.",
        )
        parser.add_argument(
            "--dist-timeout",
            type=int,
            default=ServerArgs.dist_timeout,
            help="Set timeout for torch.distributed initialization.",
        )
        parser.add_argument(
            "--download-dir",
            type=str,
            default=ServerArgs.download_dir,
            help="Model download directory for huggingface.",
        )
        parser.add_argument(
            "--base-gpu-id",
            type=int,
            default=ServerArgs.base_gpu_id,
            help="The base GPU ID to start allocating GPUs from. Useful when running multiple instances on the same machine.",
        )
        parser.add_argument(
            "--gpu-id-step",
            type=int,
            default=ServerArgs.gpu_id_step,
            help="The delta between consecutive GPU IDs that are used. For example, setting it to 2 will use GPU 0,2,4,...",
        )

        # Logging
        parser.add_argument(
            "--log-level",
            type=str,
            default=ServerArgs.log_level,
            help="The logging level of all loggers.",
        )
        parser.add_argument(
            "--log-level-http",
            type=str,
            default=ServerArgs.log_level_http,
            help="The logging level of HTTP server. If not set, reuse --log-level by default.",
        )
        parser.add_argument(
            "--log-requests",
            action="store_true",
            help="Log metadata, inputs, outputs of all requests. The verbosity is decided by --log-requests-level",
        )
        parser.add_argument(
            "--log-requests-level",
            type=int,
            default=0,
            help="0: Log metadata. 1. Log metadata and partial input/output. 2. Log every input/output.",
            choices=[0, 1, 2],
        )
        parser.add_argument(
            "--show-time-cost",
            action="store_true",
            help="Show time cost of custom marks.",
        )
        parser.add_argument(
            "--enable-metrics",
            action="store_true",
            help="Enable log metrics.",
        )
        parser.add_argument(
            "--metrics-reporters",
            action="append",
            choices=["cat", "prometheus", "llm-platform"],
            default=["cat"],
            help="Select metrics reporter(can be specified multiple times)",
        )

        parser.add_argument(
            "--app-key",
            type=str,
            default=ServerArgs.app_key,
            help="Set app key of the server",
        )

        parser.add_argument(
            "--decode-log-interval",
            type=int,
            default=ServerArgs.decode_log_interval,
            help="The log interval of decode batch.",
        )
        parser.add_argument(
            "--enable-request-time-stats-logging",
            action="store_true",
            default=ServerArgs.enable_request_time_stats_logging,
            help="Enable per request time stats logging",
        )
        parser.add_argument(
            "--kv-events-config",
            type=str,
            default=None,
            help="Config in json format for NVIDIA dynamo KV event publishing. Publishing will be enabled if this flag is used.",
        )
        parser.add_argument(
            "--enable-trace",
            action="store_true",
            help="Enable opentelemetry trace",
        )
        parser.add_argument(
            "--otlp-traces-endpoint",
            type=str,
            default="localhost:4317",
            help="Config opentelemetry collector endpoint if --enable-trace is set. format: <ip>:<port>",
        )

        # API related
        parser.add_argument(
            "--api-key",
            type=str,
            default=ServerArgs.api_key,
            help="Set API key of the server. It is also used in the OpenAI API compatible server.",
        )
        parser.add_argument(
            "--file-storage-path",
            type=str,
            default=ServerArgs.file_storage_path,
            help="The path of the file storage in backend.",
        )
        parser.add_argument(
            "--enable-cache-report",
            action="store_true",
            help="Return number of cached tokens in usage.prompt_tokens_details for each openai request.",
        )

        # Data parallelism
        parser.add_argument(
            "--data-parallel-size",
            "--dp-size",
            type=int,
            default=ServerArgs.dp_size,
            help="The data parallelism size.",
        )
        parser.add_argument(
            "--load-balance-method",
            type=str,
            default=ServerArgs.load_balance_method,
            help="The load balancing strategy for data parallelism.",
            choices=[
                "round_robin",
                "shortest_queue",
                "minimum_cache_usage",
            ],
        )
        parser.add_argument(
            "--load-watch-interval",
            type=float,
            default=ServerArgs.load_watch_interval,
            help="The interval of load watching in seconds.",
        )

        # Expert parallelism
        parser.add_argument(
            "--expert-parallel-size",
            "--ep-size",
            type=int,
            default=ServerArgs.ep_size,
            help="The expert parallelism size.",
        )
        parser.add_argument(
            "--init-expert-location",
            type=str,
            default=ServerArgs.init_expert_location,
            help="Initial location of EP experts.",
        )
        parser.add_argument(
            "--ep-num-redundant-experts",
            type=int,
            default=ServerArgs.ep_num_redundant_experts,
            help="Allocate this number of redundant experts in expert parallel.",
        )
        parser.add_argument(
            "--ep-dispatch-algorithm",
            type=str,
            default=ServerArgs.ep_dispatch_algorithm,
            help="The algorithm to choose ranks for redundant experts in expert parallel.",
        )
        parser.add_argument(
            "--eplb-algorithm",
            type=str,
            default=ServerArgs.eplb_algorithm,
            help="Chosen EPLB algorithm",
        )
        parser.add_argument(
            "--eplb-rebalance-num-iterations",
            type=int,
            default=ServerArgs.eplb_rebalance_num_iterations,
            help="Number of iterations to automatically trigger a EPLB re-balance.",
        )
        parser.add_argument(
            "--eplb-rebalance-layers-per-chunk",
            type=int,
            default=ServerArgs.eplb_rebalance_layers_per_chunk,
            help="Number of layers to rebalance per forward pass.",
        )
        parser.add_argument(
            "--expert-distribution-recorder-mode",
            type=str,
            default=ServerArgs.expert_distribution_recorder_mode,
            help="Mode of expert distribution recorder.",
        )
        parser.add_argument(
            "--expert-distribution-recorder-buffer-size",
            type=int,
            default=ServerArgs.expert_distribution_recorder_buffer_size,
            help="Circular buffer size of expert distribution recorder. Set to -1 to denote infinite buffer.",
        )
        parser.add_argument(
            "--enable-expert-distribution-metrics",
            action="store_true",
            help="Enable logging metrics for expert balancedness",
        )
        parser.add_argument(
            "--enable-eplb",
            action="store_true",
            help="Enable EPLB algorithm",
        )

        # Multi-node distributed serving
        parser.add_argument(
            "--dist-init-addr",
            "--nccl-init-addr",  # For backward compatbility. This will be removed in the future.
            type=str,
            help="The host address for initializing distributed backend (e.g., `192.168.0.2:25000`).",
        )
        parser.add_argument(
            "--nnodes", type=int, default=ServerArgs.nnodes, help="The number of nodes."
        )
        parser.add_argument(
            "--node-rank", type=int, default=ServerArgs.node_rank, help="The node rank."
        )

        # Model override args
        parser.add_argument(
            "--json-model-override-args",
            type=str,
            help="A dictionary in JSON string format used to override default model configurations.",
            default=ServerArgs.json_model_override_args,
        )
        parser.add_argument(
            "--preferred-sampling-params",
            type=str,
            help="json-formatted sampling settings that will be returned in /get_model_info",
        )

        # Kernel backend
        parser.add_argument(
            "--attention-backend",
            type=str,
            choices=[
                "flashinfer",
                "triton",
                "torch_native",
                "flashmla",
                "npu_mla",
                "torch_native_mla",
                "fa3",
                "duo_attn",
                "hybrid_linear_attn",
                "dsa",
            ],
            default=ServerArgs.attention_backend,
            help="Choose the kernels for attention layers.",
        )
        parser.add_argument(
            "--drafter-attention-backend",
            type=str,
            choices=[
                "flashinfer",
                "triton",
                "torch_native",
                "flashmla",
                "flashinfer_mla",
                "duo_attn",
                "hybrid_linear_attn",
            ],
            help="Attention backend for drafter model in speculative decoding. "
                 "If not specified, uses the same backend as the main model (attention_backend). "
                 "Supported backends: flashinfer, triton, flashmla, flashinfer_mla, "
                 "duo_attn, hybrid_linear_attn.",
        )
        parser.add_argument(
            "--chunker-backend",
            type=str,
            choices=["fa3", "flashinfer"],
            default=ServerArgs.chunker_backend,
            help="Attention backed for mla model to use in Prefill."
        )
        parser.add_argument(
            "--sampling-backend",
            type=str,
            choices=["flashinfer", "pytorch"],
            default=ServerArgs.sampling_backend,
            help="Choose the kernels for sampling layers.",
        )
        parser.add_argument(
            "--grammar-backend",
            type=str,
            choices=["xgrammar", "outlines", "llguidance"],
            default=ServerArgs.grammar_backend,
            help="Choose the backend for grammar-guided decoding.",
        )
        parser.add_argument(
            "--enable-flashinfer-mla",
            action="store_true",
            help="Enable FlashInfer MLA optimization",
        )
        parser.add_argument(
            "--flashinfer-mla-disable-ragged",
            action="store_true",
            help="Not using ragged prefill wrapper when running flashinfer mla",
        )

        # Speculative decoding
        parser.add_argument("--capture-sample-graph", action="store_true")
        parser.add_argument(
            "--draft-model-path-use-base",
            action="store_true",
            help="The path of the draft model weights use the path of the base model",
        )
        parser.add_argument(
            "--speculative-algorithm",
            type=str,
            choices=["EAGLE", "EAGLE3", "NEXTN", "PLD"],
            help="Speculative algorithm.",
        )
        parser.add_argument(
            "--prompt-lookup-min",
            type=int,
            help="The minimum number of tokens to lookup in the prompt embedding table.",
            default=ServerArgs.prompt_lookup_min,
        )
        parser.add_argument(
            "--prompt-lookup-max",
            type=int,
            help="The maximum number of tokens to lookup in the prompt embedding table.",
            default=ServerArgs.prompt_lookup_max,
        )        
        parser.add_argument(
            "--speculative-draft-model-path",
            type=str,
            help="The path of the draft model weights. This can be a local folder or a Hugging Face repo ID.",
        )
        parser.add_argument(
            "--speculative-num-steps",
            type=int,
            help="The number of steps sampled from draft model in Speculative Decoding.",
            default=ServerArgs.speculative_num_steps,
        )
        parser.add_argument(
            "--speculative-eagle-topk",
            type=int,
            help="The number of tokens sampled from the draft model in eagle2 each step.",
            choices=[1, 2, 4, 8],
            default=ServerArgs.speculative_eagle_topk,
        )
        parser.add_argument(
            "--speculative-num-draft-tokens",
            type=int,
            help="The number of tokens sampled from the draft model in Speculative Decoding.",
            default=ServerArgs.speculative_num_draft_tokens,
        )
        parser.add_argument(
            "--speculative-accept-threshold-single",
            type=float,
            help="Accept a draft token if its probability in the target model is greater than this threshold.",
            default=ServerArgs.speculative_accept_threshold_single,
        )
        parser.add_argument(
            "--speculative-accept-threshold-acc",
            type=float,
            help="The accept probability of a draft token is raised from its target probability p to min(1, p / threshold_acc).",
            default=ServerArgs.speculative_accept_threshold_acc,
        )
        parser.add_argument(
            "--speculative-token-map",
            type=str,
            help="The path of the draft model's small vocab table.",
            default=ServerArgs.speculative_token_map,
        )
        parser.add_argument(
            "--eagle3-layers-to-capture",
            type=str,
            help="The layers of Eagle3 to capture.",
            default=ServerArgs.eagle3_layers_to_capture,
        )

        # Optimization/debug options
        parser.add_argument(
            "--disable-pdl",
            action="store_true",
            help="Disable PDL launch.",
        )
        parser.add_argument(
            "--disable-radix-cache",
            action="store_true",
            help="Disable RadixAttention for prefix caching.",
        )
        parser.add_argument(
            "--radix-eviction-policy",
            type=str,
            choices=RADIX_EVICTION_POLICY_CHOICES,
            default=ServerArgs.radix_eviction_policy,
            help="The eviction policy of radix trees. 'lru' stands for Least Recently Used, 'lfu' stands for Least Frequently Used.",
        )
        parser.add_argument(
            "--disable-cuda-graph",
            action="store_true",
            help="Disable cuda graph.",
        )
        parser.add_argument(
            "--disable-cuda-graph-padding",
            action="store_true",
            help="Disable cuda graph when padding is needed. Still uses cuda graph when padding is not needed.",
        )
        parser.add_argument(
            "--enable-cudagraph-gc",
            action="store_true",
            help="Enable garbage collection during CUDA graph capture. If disabled (default), GC is frozen during capture to speed up the process.",
        )
        parser.add_argument(
            "--enable-nccl-nvls",
            action="store_true",
            help="Enable NCCL NVLS for prefill heavy requests when available.",
        )
        parser.add_argument(
            "--enable-symm-mem",
            action="store_true",
            help="Enable NCCL symmetric memory for fast collectives.",
        )
        parser.add_argument(
            "--disable-outlines-disk-cache",
            action="store_true",
            help="Disable disk cache of outlines to avoid possible crashes related to file system or high concurrency.",
        )
        parser.add_argument(
            "--disable-custom-all-reduce",
            action="store_true",
            help="Disable the custom all-reduce kernel and fall back to NCCL.",
        )
        parser.add_argument(
            "--disable-mla",
            action="store_true",
            help="Disable Multi-head Latent Attention (MLA) for DeepSeek V2/V3/R1 series models.",
        )
        parser.add_argument(
            "--disable-overlap-schedule",
            action="store_true",
            help="Disable the overlap scheduler, which overlaps the CPU scheduler with GPU model worker.",
        )
        parser.add_argument(
            "--enable-mixed-chunk",
            action="store_true",
            help="Enabling mixing prefill and decode in a batch when using chunked prefill.",
        )
        parser.add_argument(
            "--enable-torch-compile",
            action="store_true",
            help="Optimize the model with torch.compile. Experimental feature.",
        )
        parser.add_argument(
            "--torch-compile-max-bs",
            type=int,
            default=ServerArgs.torch_compile_max_bs,
            help="Set the maximum batch size when using torch compile.",
        )
        parser.add_argument(
            "--cuda-graph-max-bs",
            type=int,
            default=ServerArgs.cuda_graph_max_bs,
            help="Set the maximum batch size for cuda graph.",
        )
        parser.add_argument(
            "--cuda-graph-bs",
            type=int,
            nargs="+",
            help="Set the list of batch sizes for cuda graph.",
        )
        parser.add_argument(
            "--disable-prefill-graph",
            action="store_true",
            help="Disable cuda graph for prefill."
        )
        parser.add_argument(
            "--prefill-graph-max-tokens",
            type=int,
            default=ServerArgs.prefill_graph_max_tokens,
            help="Max query tokens to capture when enable prefill graph"
        )
        parser.add_argument(
            "--prefill-graph-max-bs",
            type=int,
            default=ServerArgs.prefill_graph_max_bs,
            help="Max batch size to capture when enable prefill graph, it relates to query tile of attention part"
        )
        parser.add_argument(
            "--torchao-config",
            type=str,
            default=ServerArgs.torchao_config,
            help="Optimize the model with torchao. Experimental feature. Current choices are: int8dq, int8wo, int4wo-<group_size>, fp8wo, fp8dq-per_tensor, fp8dq-per_row",
        )
        parser.add_argument(
            "--enable-nan-detection",
            action="store_true",
            help="Enable the NaN detection for debugging purposes.",
        )
        parser.add_argument(
            "--enable-p2p-check",
            action="store_true",
            help="Enable P2P check for GPU access, otherwise the p2p access is allowed by default.",
        )
        parser.add_argument(
            "--triton-attention-reduce-in-fp32",
            action="store_true",
            help="Cast the intermidiate attention results to fp32 to avoid possible crashes related to fp16."
            "This only affects Triton attention kernels.",
        )
        parser.add_argument(
            "--triton-attention-num-kv-splits",
            type=int,
            default=ServerArgs.triton_attention_num_kv_splits,
            help="The number of KV splits in flash decoding Triton kernel. Larger value is better in longer context scenarios. The default value is 8.",
        )
        parser.add_argument(
            "--num-continuous-decode-steps",
            type=int,
            default=ServerArgs.num_continuous_decode_steps,
            help="Run multiple continuous decoding steps to reduce scheduling overhead. "
            "This can potentially increase throughput but may also increase time-to-first-token latency. "
            "The default value is 1, meaning only run one decoding step at a time.",
        )
        parser.add_argument(
            "--delete-ckpt-after-loading",
            action="store_true",
            help="Delete the model checkpoint after loading the model.",
        )
        parser.add_argument(
            "--enable-memory-saver",
            action="store_true",
            help="Allow saving memory using release_memory_occupation and resume_memory_occupation",
        )
        parser.add_argument(
            "--allow-auto-truncate",
            action="store_true",
            help="Allow automatically truncating requests that exceed the maximum input length instead of returning an error.",
        )
        parser.add_argument(
            "--enable-custom-logit-processor",
            action="store_true",
            help="Enable users to pass custom logit processors to the server (disabled by default for security)",
        )
        tool_call_parser_choices = list(FunctionCallParser.ToolCallParserEnum.keys())
        parser.add_argument(
            "--tool-call-parser",
            type=str,
            choices=tool_call_parser_choices,
            default=ServerArgs.tool_call_parser,
            help=f"Specify the parser for handling tool-call interactions. Options include: {tool_call_parser_choices}.",
        )
        parser.add_argument(
            "--reasoning-parser",
            type=str,
            choices=list(ReasoningParser.DetectorMap.keys()),
            default=ServerArgs.reasoning_parser,
            help=f"Specify the parser for reasoning models, supported parsers are: {list(ReasoningParser.DetectorMap.keys())}.",
        )
        
        parser.add_argument(
            "--force-reasoning",
            action="store_true",
            help="Enable force-reasoning",
        )

        # For tool server
        parser.add_argument(
            "--tool-server",
            type=str,
            default=None,
            help="Either 'demo' or a comma-separated list of tool server urls to use for the model. If not specified, no tool server will be used.",
        )

        # Server warmups
        parser.add_argument(
            "--skip-server-warmup",
            action="store_true",
            help="If set, skip warmup.",
        )
        parser.add_argument(
            "--warmups",
            type=str,
            required=False,
            help="Specify custom warmup functions (csv) to run before server starts eg. --warmups=warmup_name1,warmup_name2 "
            "will run the functions `warmup_name1` and `warmup_name2` specified in warmup.py before the server starts listening for requests",
        )

        # Debug tensor dumps
        parser.add_argument(
            "--debug-tensor-dump-output-folder",
            type=str,
            default=ServerArgs.debug_tensor_dump_output_folder,
            help="The output folder for dumping tensors.",
        )
        parser.add_argument(
            "--debug-tensor-dump-input-file",
            type=str,
            default=ServerArgs.debug_tensor_dump_input_file,
            help="The input filename for dumping tensors",
        )
        parser.add_argument(
            "--debug-tensor-dump-inject",
            type=str,
            default=ServerArgs.debug_tensor_dump_inject,
            help="Inject the outputs from jax as the input of every layer.",
        )

        # Specify different parallel strategies, different combinations correspond to different communication groups and weight partitioning, as well as different communication methods
        parser.add_argument(
            "--moe-parallel-strategy",
            type=str,
            default=ServerArgs.moe_parallel_strategy,
            choices=["tp", "ep"],
            help="Specify the model parallel strategy used by moe layer",
        )
        parser.add_argument(
            "--dense-parallel-strategy",
            type=str,
            default=ServerArgs.dense_parallel_strategy,
            choices=["tp", "rep", "combine"],
            help="Specify the model parallel strategy used by attn part",
        )
        parser.add_argument(
            "--attn-tp-size",
            type=int,
            default=ServerArgs.attn_tp_size,
            help="Specify tp size for attn part",
        )
        parser.add_argument(
            "--dense-tp-size",
            type=int,
            default=ServerArgs.dense_tp_size,
            help="Specify tp size for dense part, default equals nprocs-per-node, if non dp_attn && combine_dense mode, this parameter will be overridden by attn_tp_size",
        )
        parser.add_argument(
            "--nprocs-per-node",
            type=int,
            default=ServerArgs.nprocs_per_node,
            help="Number of processes to start per node",
        )
        parser.add_argument(
            "--world-size",
            type=int,
            default=ServerArgs.nprocs_per_node,
            help="Number of processes to start per node",
        )
        parser.add_argument(
            "--enable-deep-ep",
            action="store_true",
            help="Enable DeepEp.",
        )
        parser.add_argument(
            "--force-deterministic-rsag",
            action="store_true",
            help="Enable force deterministic rsag.",
        )
        parser.add_argument(
            "--enable-tbo",
            action="store_true",
            help="Enable two micro batches overlap.",
        )
        parser.add_argument(
            "--enable-sbo",
            action="store_true",
            help="Enable overlap of a single batch.",
        )

        parser.add_argument(
            "--tbo-min-bs",
            type=int,
            default=ServerArgs.tbo_min_bs,
            help="Min batch size to enable two batches overlap.",
        )

        parser.add_argument(
            "--low-latency-max-num-tokens-per-gpu",
            type=int,
            default=ServerArgs.low_latency_max_num_tokens_per_gpu,
            help="Low latency max num tokens per gpu",
        )

        parser.add_argument(
            "--mla-max-chunk-capacity",
            type=int,
            default=ServerArgs.mla_max_chunk_capacity,
            help="Mla max chunk capacity",
        )

        # Disaggregation
        parser.add_argument(
            "--disaggregation-mode",
            type=str,
            default="null",
            choices=["null", "prefill", "decode"],
            help='Only used for PD disaggregation. "prefill" for prefill-only server, and "decode" for decode-only server. If not specified, it is not PD disaggregated',
        )
        parser.add_argument(
            "--flashinfer-comm-max-num-tokens",
            type=int,
            default=ServerArgs.flashinfer_comm_max_num_tokens,
            help="Max num tokens for flashinfer communication fusion workspace"
        )
        parser.add_argument(
            "--disaggregation-bootstrap-port",
            type=int,
            default=ServerArgs.disaggregation_bootstrap_port,
            help="Bootstrap server port on the prefill server. Default is 8998.",
        )
        parser.add_argument(
            "--disaggregation-transfer-backend",
            type=str,
            default=ServerArgs.disaggregation_transfer_backend,
            choices=["mooncake", "mooncake_async", "nixl"],
            help="The backend for disaggregation transfer. Default is mooncake.",
        )
        parser.add_argument(
            "--disaggregation-ib-device",
            type=str,
            default=ServerArgs.disaggregation_ib_device,
            help="The InfiniBand devices for disaggregation transfer, accepts single device (e.g., --disaggregation-ib-device mlx5_0) "
            "or multiple comma-separated devices (e.g., --disaggregation-ib-device mlx5_0,mlx5_1). "
            "Default is None, which triggers automatic device detection when mooncake backend is enabled.",
        )
        parser.add_argument(
            "--disaggregation-layerwise-interval",
            type=int,
            default=ServerArgs.disaggregation_layerwise_interval,
            help="The interval of layerwise transfer for disaggregation. Default is 1.",
        )
        parser.add_argument(
            "--disaggregation-transfer-hidden-states-max-size",
            type=int,
            default=ServerArgs.disaggregation_transfer_hidden_states_max_size,
            help="Transfer hidden_states max size for disaggregation. Default is 0.",
        )
        parser.add_argument(
            "--pdlb-url",
            type=str,
            default=None,
            help="The URL of the PD disaggregation load balancer. If set, the prefill/decode server will register with the load balancer.",
        )

        # Multi-modal inference mode
        parser.add_argument("--mm-mode", type=str, default=ServerArgs.mm_mode)

        # NPU Flag
        parser.add_argument(
            "--npu-enable-weight-nz",
            action="store_true",
            help="[NPU] enable weight nz",
        )
        parser.add_argument(
            "--npu-enable-mc2",
            action="store_true",
            help="[NPU] enable mc2",
        )
        parser.add_argument(
            "--npu-enable-mlp-matmul",
            action="store_true",
            help="[NPU] enable mlp_matmul",
        )
        parser.add_argument(
            "--npu-enable-opt-rope",
            action="store_true",
            help="[NPU] enable optimization on rope",
        )
        parser.add_argument(
            "--request-max-input-len",
            type=int,
            default=0,
            help="request_max_input_len. Default is 0.",
        )
        parser.add_argument(
            "--request-max-output-len",
            type=int,
            default=0,
            help="request_max_output_len. Default is 0.",
        )
        parser.add_argument(
            "--request-cache-size",
            type=int,
            help="request_cache_size. Default is 0.",
            default=0,
        )
        parser.add_argument(
            "--request-cache-config",
            type=str,
            default=ServerArgs.request_cache_config,
            help='request_cache_config json format',
        )

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace):
        args.dp_size = args.data_parallel_size
        args.ep_size = args.expert_parallel_size
        attrs = [attr.name for attr in dataclasses.fields(cls)]
        return cls(**{attr: getattr(args, attr, None) for attr in attrs})

    def url(self):
        if is_valid_ipv6_address(self.host):
            return f"http://[{self.host}]:{self.port}"
        else:
            return f"http://{self.host}:{self.port}"

    def check_server_args(self):
        assert (
            self.world_size % self.nnodes == 0
        ), "world_size must be divisible by number of nodes"
        assert not (
            self.dp_size > 1 and self.nnodes != 1 and not self.enable_dp_attention
        ), "multi-node data parallel is not supported unless dp attention!"
        assert self.base_gpu_id >= 0, "base_gpu_id must be non-negative"
        assert self.gpu_id_step >= 1, "gpu_id_step must be positive"

        assert (not self.enable_sbo or self.enable_deep_ep), "SBO requires DeepEP to be enabled now."


def prepare_server_args(argv: List[str]) -> ServerArgs:
    """
    Prepare the server arguments from the command line arguments.

    Args:
        args: The command line arguments. Typically, it should be `sys.argv[1:]`
            to ensure compatibility with `parse_args` when no arguments are passed.

    Returns:
        The server arguments.
    """
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    raw_args = parser.parse_args(argv)
    server_args = ServerArgs.from_cli_args(raw_args)
    return server_args


ZMQ_TCP_PORT_DELTA = 233


@dataclasses.dataclass
class PortArgs:
    # The ipc filename for tokenizer to receive inputs from detokenizer (zmq)
    tokenizer_ipc_name: str
    # The ipc filename for scheduler (rank 0) to receive inputs from tokenizer (zmq)
    scheduler_input_ipc_name: str
    # The ipc filename for detokenizer to receive inputs from scheduler (zmq)
    detokenizer_ipc_name: str

    # The port for nccl initialization (torch.dist)
    nccl_port: int
    
    # The ipc filename for rpc call between Engine and Scheduler
    rpc_ipc_name: str
    
    # The ipc filename for Scheduler to send metrics
    metrics_ipc_name: str
    
    # The ipc filename for Tokenizer and worker tokenizer
    tokenizer_worker_ipc_name: Optional[str]

    @staticmethod
    def init_new(server_args, dp_rank: Optional[int] = None) -> "PortArgs":
        port = server_args.port + random.randint(100, 1000)
        while True:
            if is_port_available(port):
                break
            if port < 60000:
                port += 42
            else:
                port -= 43

        # DP attention. Use TCP + port to handle both single-node and multi-node.
        if server_args.dp_spmd_mode or (server_args.nnodes == 1 and server_args.dist_init_addr is None):
            # Only use default port fallback when dp_size == 1
            # For dp_size > 1, we need explicit dist_init_addr to avoid port conflicts
            if not server_args.dp_spmd_mode and server_args.dp_size > 1:
                raise ValueError(
                    f"When dp_size > 1 (dp_size={server_args.dp_size}), you must provide --dist-init-addr. "
                    f"Example: --dist-init-addr 127.0.0.1:4000"
                )
            dist_init_addr = ("127.0.0.1", server_args.port + ZMQ_TCP_PORT_DELTA)
        else:
            dist_init_addr = server_args.dist_init_addr.split(":")
        assert (
            len(dist_init_addr) == 2
        ), "please provide --dist-init-addr as host:port of head node"

        dist_init_host, dist_init_port = dist_init_addr
        dist_init_port = int(dist_init_port)
        port_base = dist_init_port + 1
        detokenizer_port = port_base + 1
        rpc_port = port_base + 2
        metrics_ipc_port = port_base + 3
        if dp_rank is None:
            # TokenizerManager to DataParallelController
            scheduler_input_port = port_base + 4
        else:
            scheduler_input_port = port_base + 2 + 1 + dp_rank
        if not is_port_available(scheduler_input_port):
            raise Exception(f"{scheduler_input_port=} is in use")
        rpc_ipc_port = scheduler_input_port + 1

        return PortArgs(
            tokenizer_ipc_name=f"tcp://{dist_init_host}:{port_base}",
            scheduler_input_ipc_name=f"tcp://{dist_init_host}:{scheduler_input_port}",
            detokenizer_ipc_name=f"tcp://{dist_init_host}:{detokenizer_port}",
            nccl_port=port,
            rpc_ipc_name=f"tcp://{dist_init_host}:{rpc_port}",
            metrics_ipc_name=f"tcp://{dist_init_host}:{metrics_ipc_port}",
            tokenizer_worker_ipc_name=None,
        )


class DeprecatedAction(argparse.Action):
    def __init__(self, option_strings, dest, nargs=0, **kwargs):
        super(DeprecatedAction, self).__init__(
            option_strings, dest, nargs=nargs, **kwargs
        )

    def __call__(self, parser, namespace, values, option_string=None):
        raise ValueError(self.help)
