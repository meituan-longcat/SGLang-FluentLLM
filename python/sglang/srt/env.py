import os
from sglang.srt.utils import is_npu

import subprocess
import warnings
from contextlib import ExitStack, contextmanager
from enum import IntEnum
from typing import Any

__is_npu__ = is_npu()

# NOTE: Initialized with None defaults to avoid circular import with
# sglang.srt.server_args (server_args -> distributed -> parallel_state -> env -> server_args).
# The real values are populated at runtime when global_server_args_dict_update() is called.
global_server_args_dict: dict = {}

class EnvVar:
    npu_enable_all2all_comm: bool = os.getenv("NPU_ENABLE_ALL2ALL_COMM", "1") == "1"
    npu_enable_graph: bool = os.getenv("NPU_ENABLE_GRAPH", "0") == "1"
    npu_enable_mla_split_kv_kr: bool = __is_npu__ and os.getenv("NPU_ENABLE_MLA_SPLIT_KV_KR", "1") == "1"
    npu_enable_get_out_cache: bool=os.getenv("NPU_ENABLE_GET_OUT_CACHE", "0")=="1"
    npu_moe_core_num: bool = int(os.getenv("NPU_MOE_CORE_NUM", "12"))

    def __init__(self):
        """
        logger.info(f"npu_enable_all2all_comm:{self.npu_enable_all2all_comm}")
        logger.info(f"npu_enable_graph:{self.npu_enable_graph}")
        logger.info(f"npu_enable_mla_split_kv_kr:{self.npu_enable_mla_split_kv_kr}")
        """
        pass




ENV = EnvVar()

def global_server_args_dict_update(server_args):
    global_server_args_dict.update(
        {
            "attention_backend": server_args.attention_backend,
            "sampling_backend": server_args.sampling_backend,
            "chunker_backend": server_args.chunker_backend,
            "triton_attention_reduce_in_fp32": server_args.triton_attention_reduce_in_fp32,
            "torchao_config": server_args.torchao_config,
            "kv_cache_dtype": server_args.kv_cache_dtype,
            "enable_nan_detection": server_args.enable_nan_detection,
            "enable_dp_attention": server_args.enable_dp_attention,
            "enable_ep_moe": server_args.enable_ep_moe,
            "enable_deep_ep": server_args.enable_deep_ep,
            "force_deterministic_rsag": server_args.force_deterministic_rsag,
            "low_latency_max_num_tokens_per_gpu": server_args.low_latency_max_num_tokens_per_gpu,
            "device": server_args.device,
            "draft_model_path_use_base": server_args.draft_model_path_use_base,
            "speculative_accept_threshold_single": server_args.speculative_accept_threshold_single,
            "speculative_accept_threshold_acc": server_args.speculative_accept_threshold_acc,
            "speculative_algorithm": server_args.speculative_algorithm,
            "is_multi_head_eagle": server_args.is_multi_head_eagle,
            "draft_use_oe": server_args.draft_use_oe,
            "enable_flashinfer_mla": server_args.enable_flashinfer_mla,
            "disable_pdl": server_args.disable_pdl,
            "disable_radix_cache": server_args.disable_radix_cache,
            "flashinfer_mla_disable_ragged": server_args.flashinfer_mla_disable_ragged,
            "debug_tensor_dump_output_folder": server_args.debug_tensor_dump_output_folder,
            "debug_tensor_dump_inject": server_args.debug_tensor_dump_inject,
            "dp_size": server_args.dp_size,
            "attn_tp_size": server_args.attn_tp_size,
            "kvp_size": server_args.kvp_size,
            "attn_parallel_strategy": server_args.attn_parallel_strategy,
            "moe_parallel_strategy": server_args.moe_parallel_strategy,
            "dense_parallel_strategy": server_args.dense_parallel_strategy,
            "chunked_prefill_size": server_args.chunked_prefill_size,
            "mla_max_chunk_capacity": server_args.mla_max_chunk_capacity,
            "ep_num_redundant_experts": server_args.ep_num_redundant_experts,
            "ep_dispatch_algorithm": server_args.ep_dispatch_algorithm,
            "enable_eplb": server_args.enable_eplb,
            "mm_mode": server_args.mm_mode,
            "npu_enable_weight_nz": server_args.npu_enable_weight_nz,
            "npu_enable_opt_rope": server_args.npu_enable_opt_rope,
            "flashinfer_comm_max_num_tokens": server_args.flashinfer_comm_max_num_tokens,
            "enable_deep_ep": server_args.enable_deep_ep,
            "max_prefill_tokens": server_args.max_prefill_tokens,
            "max_running_requests": server_args.max_running_requests,
            "enable_mla_l1_5_cache": server_args.enable_mla_l1_5_cache,
            "max_total_tokens": server_args.max_total_tokens,
            "npu_hccl_buffsize_a2a": server_args.npu_hccl_buffsize_a2a,
            "npu_o_proj_tp_size": server_args.npu_o_proj_tp_size,
            "npu_smooth_quant": server_args.npu_smooth_quant,
            "npu_kvp_accuracy_fix": server_args.npu_kvp_accuracy_fix,
            "npu_enable_super_kernel": server_args.npu_enable_super_kernel,
            "npu_disable_kv_nz": server_args.npu_disable_kv_nz,
            "model_path": server_args.model_path,
            "npu_enable_graph_cache": server_args.npu_enable_graph_cache,
            "npu_graph_cache_path": server_args.npu_graph_cache_path,
            "npu_compile_bs": server_args.npu_compile_bs,
            "disaggregation_mode": server_args.disaggregation_mode,
            "npu_disable_all_gather": server_args.npu_disable_all_gather,
            "npu_scheduler_comm": server_args.npu_scheduler_comm,
            "npu_enable_sp_for_indexer": server_args.npu_enable_sp_for_indexer,
            "npu_disable_dsa_head_parallel": server_args.npu_disable_dsa_head_parallel,
            "npu_enable_a2_dispatch_combine_opt": server_args.npu_enable_a2_dispatch_combine_opt,
            "npu_limit_max_new_tokens": server_args.npu_limit_max_new_tokens,
            "npu_moe_chunked_prefill_size": server_args.npu_moe_chunked_prefill_size,
            "npu_lmhead_tp_size": server_args.npu_lmhead_tp_size,
            "npu_enable_oe_cpu_offload": server_args.npu_enable_oe_cpu_offload,
            "pp_layer_nums": server_args.pp_layer_nums,
        }
    )


class EnvField:
    def __init__(self, default: Any):
        self.default = default
        # NOTE: we use None to indicate whether the value is set or not
        # If the value is manually set to None, we need mark it as _set_to_none.
        # Always use clear() to reset the value, which leads to the default fallback.
        self._set_to_none = False

    def __set_name__(self, owner, name):
        self.name = name

    def parse(self, value: str) -> Any:
        raise NotImplementedError()

    def get(self) -> Any:
        value = os.getenv(self.name)
        if self._set_to_none:
            assert value is None
            return None

        if value is None:
            return self.default

        try:
            return self.parse(value)
        except ValueError as e:
            warnings.warn(
                f'Invalid value for {self.name}: {e}, using default "{self.default}"'
            )
            return self.default

    def is_set(self):
        # NOTE: If None is manually set, it is considered as set.
        return self.name in os.environ or self._set_to_none

    def get_set_value_or(self, or_value: Any):
        # NOTE: Ugly usage, but only way to get custom default value.
        return self.get() if self.is_set() else or_value

    def set(self, value: Any):
        if value is None:
            self._set_to_none = True
            os.environ.pop(self.name, None)
        else:
            self._set_to_none = False
            os.environ[self.name] = str(value)

    @contextmanager
    def override(self, value: Any):
        backup_present = self.name in os.environ
        backup_value = os.environ.get(self.name)
        backup_set_to_none = self._set_to_none
        self.set(value)
        yield
        if backup_present:
            os.environ[self.name] = backup_value
        else:
            os.environ.pop(self.name, None)
        self._set_to_none = backup_set_to_none

    def clear(self):
        os.environ.pop(self.name, None)
        self._set_to_none = False

    @property
    def value(self):
        return self.get()

    def __bool__(self):
        raise RuntimeError(
            "Please use `envs.YOUR_FLAG.get()` instead of `envs.YOUR_FLAG`"
        )

    def __len__(self):
        raise RuntimeError(
            "Please use `envs.YOUR_FLAG.get()` instead of `envs.YOUR_FLAG`"
        )


class EnvTuple(EnvField):
    def parse(self, value: str) -> tuple[str, ...]:
        return tuple(s.strip() for s in value.split(",") if s.strip())


class EnvStr(EnvField):
    def parse(self, value: str) -> str:
        return value


class EnvBool(EnvField):
    def parse(self, value: str) -> bool:
        value = value.lower()
        if value in ["true", "1", "yes", "y"]:
            return True
        if value in ["false", "0", "no", "n"]:
            return False
        raise ValueError(f'"{value}" is not a valid boolean value')


class EnvInt(EnvField):
    def parse(self, value: str) -> int:
        try:
            return int(value)
        except ValueError:
            raise ValueError(f'"{value}" is not a valid integer value')


class EnvFloat(EnvField):
    def parse(self, value: str) -> float:
        try:
            return float(value)
        except ValueError:
            raise ValueError(f'"{value}" is not a valid float value')


class ToolStrictLevel(IntEnum):
    """
    Defines the strictness levels for tool call parsing and validation.

    OFF: No strict validation
    FUNCTION: Enables structural tag constraints for all tools
    PARAMETER: Enforces strict parameter validation for all tools
    """

    OFF = 0
    FUNCTION = 1
    PARAMETER = 2


class Envs:
    # fmt: off

    # Model & File Download
    SGLANG_USE_MODELSCOPE = EnvBool(False)
    SGLANG_DISABLED_MODEL_ARCHS = EnvTuple(tuple())

    # Logging Options
    SGLANG_LOG_GC = EnvBool(False)
    SGLANG_LOG_FORWARD_ITERS = EnvBool(False)
    SGLANG_LOG_MS = EnvBool(False)
    SGLANG_DISABLE_REQUEST_LOGGING = EnvBool(False)

    # Test & Debug
    SGLANG_IS_IN_CI = EnvBool(False)
    SGLANG_IS_IN_CI_AMD = EnvBool(False)
    IS_BLACKWELL = EnvBool(False)
    SGLANG_SET_CPU_AFFINITY = EnvBool(False)
    SGLANG_PROFILE_WITH_STACK = EnvBool(True)
    SGLANG_PROFILE_RECORD_SHAPES = EnvBool(True)
    SGLANG_PROFILE_V2 = EnvBool(False)
    SGLANG_RECORD_STEP_TIME = EnvBool(False)
    SGLANG_FORCE_SHUTDOWN = EnvBool(False)
    SGLANG_DEBUG_MEMORY_POOL = EnvBool(False)
    SGLANG_TEST_REQUEST_TIME_STATS = EnvBool(False)
    SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK = EnvBool(False)
    SGLANG_SIMULATE_ACC_LEN = EnvFloat(-1)
    SGLANG_SIMULATE_ACC_METHOD = EnvStr("multinomial")
    SGLANG_TORCH_PROFILER_DIR = EnvStr("/tmp")
    SGLANG_OTLP_EXPORTER_SCHEDULE_DELAY_MILLIS = EnvInt(500)
    SGLANG_OTLP_EXPORTER_MAX_EXPORT_BATCH_SIZE = EnvInt(64)
    SGLANG_NATIVE_MOVE_KV_CACHE = EnvBool(False)

    # Scheduler: memory leak test
    SGLANG_TEST_RETRACT = EnvBool(False)
    SGLANG_TEST_RETRACT_INTERVAL = EnvInt(3)
    SGLANG_TEST_RETRACT_NO_PREFILL_BS = EnvInt(2 ** 31)
    SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY = EnvInt(0)
    SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE = EnvBool(True)
    SGLANG_CI_SMALL_KV_SIZE = EnvInt(-1)

    # Scheduler: new token ratio hyperparameters
    SGLANG_INIT_NEW_TOKEN_RATIO = EnvFloat(0.7)
    SGLANG_MIN_NEW_TOKEN_RATIO_FACTOR = EnvFloat(0.14)
    SGLANG_NEW_TOKEN_RATIO_DECAY_STEPS = EnvInt(600)
    SGLANG_RETRACT_DECODE_STEPS = EnvInt(20)
    SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION = EnvInt(4096)

    # Scheduler: recv interval
    SGLANG_SCHEDULER_RECV_SKIPPER_WEIGHT_DEFAULT = EnvInt(1000)
    SGLANG_SCHEDULER_RECV_SKIPPER_WEIGHT_DECODE = EnvInt(1)
    SGLANG_SCHEDULER_RECV_SKIPPER_WEIGHT_TARGET_VERIFY = EnvInt(1)
    SGLANG_SCHEDULER_RECV_SKIPPER_WEIGHT_NONE = EnvInt(1)


    # Scheduler: others:
    SGLANG_EMPTY_CACHE_INTERVAL = EnvFloat(-1)  # in seconds. Set if you observe high memory accumulation over a long serving period.
    SGLANG_DISABLE_CONSECUTIVE_PREFILL_OVERLAP = EnvBool(False)
    SGLANG_SCHEDULER_MAX_RECV_PER_POLL = EnvInt(-1)
    SGLANG_EXPERIMENTAL_CPP_RADIX_TREE = EnvBool(False)
    SGLANG_DYNAMIC_CHUNKING_SMOOTH_FACTOR = EnvFloat(0.75)
    SGLANG_SCHEDULER_SKIP_ALL_GATHER = EnvBool(False)
    SGLANG_SCHEDULER_DECREASE_PREFILL_IDLE = EnvBool(False)

    # Test: pd-disaggregation
    SGLANG_TEST_PD_DISAGG_BACKEND = EnvStr("mooncake")
    SGLANG_TEST_PD_DISAGG_DEVICES = EnvStr(None)

    # Model Parallel
    SGLANG_USE_MESSAGE_QUEUE_BROADCASTER = EnvBool(True)
    SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS = EnvBool(False)

    # Constrained Decoding
    SGLANG_DISABLE_OUTLINES_DISK_CACHE = EnvBool(True)
    SGLANG_GRAMMAR_TIMEOUT = EnvFloat(300)

    # Tool Calling
    SGLANG_FORWARD_UNKNOWN_TOOLS = EnvBool(False)

    # Hi-Cache
    SGLANG_HICACHE_HF3FS_CONFIG_PATH = EnvStr(None)

    # Mooncake KV Transfer
    SGLANG_MOONCAKE_CUSTOM_MEM_POOL = EnvStr(None)
    ENABLE_ASCEND_TRANSFER_WITH_MOONCAKE = EnvBool(False)
    ASCEND_NPU_PHY_ID = EnvInt(-1)

    # Mooncake Store
    SGLANG_HICACHE_MOONCAKE_CONFIG_PATH = EnvStr(None)
    MOONCAKE_MASTER = EnvStr(None)
    MOONCAKE_LOCAL_HOSTNAME = EnvStr("localhost")
    MOONCAKE_TE_META_DATA_SERVER = EnvStr("P2PHANDSHAKE")
    MOONCAKE_GLOBAL_SEGMENT_SIZE = EnvStr("4gb")
    MOONCAKE_PROTOCOL = EnvStr("tcp")
    MOONCAKE_DEVICE = EnvStr("")
    MOONCAKE_MASTER_METRICS_PORT = EnvInt(9003)
    MOONCAKE_CHECK_SERVER = EnvBool(False)

    # AMD & ROCm
    SGLANG_USE_AITER = EnvBool(False)
    SGLANG_ROCM_FUSED_DECODE_MLA = EnvBool(False)
    SGLANG_ROCM_DISABLE_LINEARQUANT = EnvBool(False)

    # NPU
    SGLANG_NPU_DISABLE_ACL_FORMAT_WEIGHT = EnvBool(False)
    SGLANG_NPU_USE_MULTI_STREAM = EnvBool(False)

    # Quantization
    SGLANG_INT4_WEIGHT = EnvBool(False)
    SGLANG_CPU_QUANTIZATION = EnvBool(False)
    SGLANG_USE_DYNAMIC_MXFP4_LINEAR = EnvBool(False)
    SGLANG_FORCE_FP8_MARLIN = EnvBool(False)
    SGLANG_MOE_NVFP4_DISPATCH = EnvBool(False)
    SGLANG_NVFP4_CKPT_FP8_GEMM_IN_ATTN = EnvBool(False)
    SGLANG_PER_TOKEN_GROUP_QUANT_8BIT_V2 = EnvBool(False)

    # Flashinfer
    SGLANG_IS_FLASHINFER_AVAILABLE = EnvBool(True)
    SGLANG_ENABLE_FLASHINFER_FP8_GEMM = EnvBool(False)
    # Default to the pick from flashinfer
    SGLANG_FLASHINFER_FP4_GEMM_BACKEND = EnvStr("")
    SGLANG_FLASHINFER_WORKSPACE_SIZE = EnvInt(384 * 1024 * 1024)

    # Triton
    SGLANG_TRITON_DECODE_ATTN_STATIC_KV_SPLITS = EnvBool(False)
    SGLANG_USE_CUSTOM_TRITON_KERNEL_CACHE = EnvBool(False)

    # Torch Compile
    SGLANG_ENABLE_TORCH_COMPILE = EnvBool(False)

    # EPLB
    SGLANG_EXPERT_LOCATION_UPDATER_LOG_INPUT = EnvBool(False)
    SGLANG_EXPERT_LOCATION_UPDATER_CANARY = EnvBool(False)
    SGLANG_EXPERT_LOCATION_UPDATER_LOG_METRICS = EnvBool(False)
    SGLANG_LOG_EXPERT_LOCATION_METADATA = EnvBool(False)
    SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR = EnvStr("/tmp")
    SGLANG_EPLB_HEATMAP_COLLECTION_INTERVAL = EnvInt(0)

    # TBO
    SGLANG_TBO_DEBUG = EnvBool(False)

    # DeepGemm
    SGLANG_ENABLE_JIT_DEEPGEMM = EnvBool(True)
    SGLANG_JIT_DEEPGEMM_PRECOMPILE = EnvBool(True)
    SGLANG_JIT_DEEPGEMM_COMPILE_WORKERS = EnvInt(4)
    SGLANG_IN_DEEPGEMM_PRECOMPILE_STAGE = EnvBool(False)
    SGLANG_DG_CACHE_DIR = EnvStr(os.path.expanduser("~/.cache/deep_gemm"))
    SGLANG_DG_USE_NVRTC = EnvBool(False)
    SGLANG_USE_DEEPGEMM_BMM = EnvBool(False)

    # DeepEP
    SGLANG_DEEPEP_BF16_DISPATCH = EnvBool(False)
    SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK = EnvInt(128)
    SGLANG_DEEPEP_LL_COMBINE_SEND_NUM_SMS = EnvInt(32)

    # sgl-kernel
    SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK = EnvBool(False)

    # vLLM dependencies (TODO: they have been deprecated, we can remove them safely)
    USE_VLLM_CUTLASS_W8A8_FP8_KERNEL = EnvBool(False)

    USE_TRITON_W8A8_FP8_KERNEL = EnvBool(False)
    SGLANG_RETURN_ORIGINAL_LOGPROB = EnvBool(False)
    SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN = EnvBool(False)
    SGLANG_MOE_PADDING = EnvBool(False)
    SGLANG_CUTLASS_MOE = EnvBool(False)
    HF_HUB_DISABLE_XET = EnvBool(False)
    DISABLE_OPENAPI_DOC = EnvBool(False)
    SGLANG_ENABLE_TORCH_INFERENCE_MODE = EnvBool(False)
    SGLANG_IS_FIRST_RANK_ON_NODE = EnvBool(True)
    SGLANG_SUPPORT_CUTLASS_BLOCK_FP8 = EnvBool(False)
    SGLANG_SYNC_TOKEN_IDS_ACROSS_TP = EnvBool(False)
    SGLANG_ENABLE_COLOCATED_BATCH_GEN = EnvBool(False)

    # Deterministic inference
    SGLANG_ENABLE_DETERMINISTIC_INFERENCE = EnvBool(False)
    SGLANG_FLASHINFER_PREFILL_SPLIT_TILE_SIZE = EnvInt(4096)
    SGLANG_FLASHINFER_DECODE_SPLIT_TILE_SIZE = EnvInt(2048)
    SGLANG_TRITON_PREFILL_TRUNCATION_ALIGN_SIZE = EnvInt(4096)
    SGLANG_TRITON_DECODE_SPLIT_TILE_SIZE = EnvInt(256)

    # RoPE cache configuration
    SGLANG_SPEC_EXPANSION_SAFETY_FACTOR = EnvInt(2)
    SGLANG_ROPE_CACHE_SAFETY_MARGIN = EnvInt(256)
    SGLANG_ROPE_CACHE_ALIGN = EnvInt(128)

    # Overlap Spec V2
    SGLANG_ENABLE_SPEC_V2 = EnvBool(False)
    SGLANG_ENABLE_OVERLAP_PLAN_STREAM = EnvBool(False)

    # Spec Config
    SGLANG_SPEC_ENABLE_STRICT_FILTER_CHECK = EnvBool(True)

    # VLM
    SGLANG_VLM_CACHE_SIZE_MB = EnvInt(100)
    SGLANG_IMAGE_MAX_PIXELS = EnvInt(16384 * 28 * 28)
    SGLANG_RESIZE_RESAMPLE = EnvStr("")
    SGLANG_MM_BUFFER_SIZE_MB = EnvInt(0)

    # Release & Resume Memory
    SGLANG_MEMORY_SAVER_CUDA_GRAPH = EnvBool(False)

    # Sparse Embeddings
    SGLANG_EMBEDDINGS_SPARSE_HEAD = EnvStr(None)

    # Logits processor
    SGLANG_ENABLE_LOGITS_PROCESSER_CHUNK = EnvBool(False)
    SGLANG_LOGITS_PROCESSER_CHUNK_SIZE = EnvInt(2048)

    # Tool-Call behavior
    SGLANG_TOOL_STRICT_LEVEL = EnvInt(ToolStrictLevel.OFF)

    # Ngram
    SGLANG_NGRAM_FORCE_GREEDY_VERIFY = EnvBool(False)

    # Warmup
    SGLANG_WARMUP_TIMEOUT = EnvFloat(-1) # in seconds. If a warmup forward batch takes longer than this, the server will crash to prevent hanging. Recommend to increase warmup timeout to 1800 to accommodate some kernel JIT precache e.g. deep gemm

    # Health Check
    SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION = EnvBool(True)

    # External models
    SGLANG_EXTERNAL_MODEL_PACKAGE = EnvStr("")
    SGLANG_EXTERNAL_MM_MODEL_ARCH = EnvStr("")
    SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE = EnvStr("")

    # Numa
    SGLANG_NUMA_BIND_V2 = EnvBool(True)

    # fmt: on


envs = Envs()


def _print_deprecated_env(new_name: str, old_name: str):
    if old_name in os.environ:
        warnings.warn(
            f"Environment variable {old_name} will be deprecated, please use {new_name} instead"
        )
        os.environ[new_name] = os.environ[old_name]


def _warn_deprecated_env_to_cli_flag(env_name: str, suggestion: str):
    """Warn when a deprecated environment variable is used.

    This is for env vars that are deprecated in favor of CLI flags.
    """
    if env_name in os.environ:
        warnings.warn(f"Environment variable {env_name} is deprecated. {suggestion}")


def _convert_SGL_to_SGLANG():
    _print_deprecated_env("SGLANG_LOG_GC", "SGLANG_GC_LOG")
    _print_deprecated_env(
        "SGLANG_ENABLE_FLASHINFER_FP8_GEMM", "SGLANG_ENABLE_FLASHINFER_GEMM"
    )
    _print_deprecated_env(
        "SGLANG_MOE_NVFP4_DISPATCH", "SGLANG_CUTEDSL_MOE_NVFP4_DISPATCH"
    )

    for key, value in os.environ.items():
        if key.startswith("SGL_"):
            new_key = key.replace("SGL_", "SGLANG_", 1)
            warnings.warn(
                f"Environment variable {key} is deprecated, please use {new_key}"
            )
            os.environ[new_key] = value


_convert_SGL_to_SGLANG()

_warn_deprecated_env_to_cli_flag(
    "SGLANG_ENABLE_FLASHINFER_FP8_GEMM",
    "It will be completely removed in 0.5.7. Please use '--fp8-gemm-backend=flashinfer_trtllm' instead.",
)
_warn_deprecated_env_to_cli_flag(
    "SGLANG_ENABLE_FLASHINFER_GEMM",
    "It will be completely removed in 0.5.7. Please use '--fp8-gemm-backend=flashinfer_trtllm' instead.",
)
_warn_deprecated_env_to_cli_flag(
    "SGLANG_SUPPORT_CUTLASS_BLOCK_FP8",
    "It will be completely removed in 0.5.7. Please use '--fp8-gemm-backend=cutlass' instead.",
)


def example_with_exit_stack():
    # Use this style of context manager in unit test
    exit_stack = ExitStack()
    exit_stack.enter_context(envs.SGLANG_TEST_RETRACT.override(False))
    assert envs.SGLANG_TEST_RETRACT.value is False
    exit_stack.close()
    assert envs.SGLANG_TEST_RETRACT.value is None


def example_with_subprocess():
    command = ["python", "-c", "import os; print(os.getenv('SGLANG_TEST_RETRACT'))"]
    with envs.SGLANG_TEST_RETRACT.override(True):
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        process.wait()
        output = process.stdout.read().decode("utf-8").strip()
        assert output == "True"

    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output = process.stdout.read().decode("utf-8").strip()
    assert output == "None"


def example_with_implicit_bool_avoidance():
    @contextmanager
    def assert_throws(message_matcher: str):
        try:
            yield
        except Exception as e:
            assert message_matcher in str(e), f"{e=}"
            print(f"assert_throws find expected error: {e}")
            return
        raise AssertionError(f"assert_throws do not see exceptions")

    with assert_throws("Please use `envs.YOUR_FLAG.get()` instead of `envs.YOUR_FLAG`"):
        if envs.SGLANG_TEST_RETRACT:
            pass

    with assert_throws("Please use `envs.YOUR_FLAG.get()` instead of `envs.YOUR_FLAG`"):
        if (1 != 1) or envs.SGLANG_TEST_RETRACT:
            pass

    with assert_throws("Please use `envs.YOUR_FLAG.get()` instead of `envs.YOUR_FLAG`"):
        if envs.SGLANG_TEST_RETRACT or (1 == 1):
            pass


def examples():
    # Example usage for envs
    envs.SGLANG_TEST_RETRACT.clear()
    assert envs.SGLANG_TEST_RETRACT.value is False

    envs.SGLANG_TEST_RETRACT.set(None)
    assert envs.SGLANG_TEST_RETRACT.is_set() and envs.SGLANG_TEST_RETRACT.value is None

    envs.SGLANG_TEST_RETRACT.clear()
    assert not envs.SGLANG_TEST_RETRACT.is_set()

    envs.SGLANG_TEST_RETRACT.set(True)
    assert envs.SGLANG_TEST_RETRACT.value is True

    with envs.SGLANG_TEST_RETRACT.override(None):
        assert (
            envs.SGLANG_TEST_RETRACT.is_set() and envs.SGLANG_TEST_RETRACT.value is None
        )

    assert envs.SGLANG_TEST_RETRACT.value is True

    envs.SGLANG_TEST_RETRACT.set(None)
    with envs.SGLANG_TEST_RETRACT.override(True):
        assert envs.SGLANG_TEST_RETRACT.value is True

    assert envs.SGLANG_TEST_RETRACT.is_set() and envs.SGLANG_TEST_RETRACT.value is None

    example_with_exit_stack()
    example_with_subprocess()
    example_with_implicit_bool_avoidance()


if __name__ == "__main__":
    examples()
