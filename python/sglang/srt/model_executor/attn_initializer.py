import json
import os
from typing import Optional

import torch

from sglang.srt.configs.model_config import AttentionArch, get_nsa_index_head_dim, get_nsa_index_topk, is_dsa
from sglang.srt.distributed.parallel_state import get_world_group
from sglang.srt.env import ENV, global_server_args_dict
from sglang.srt.layers.attention.torch_native_backend import TorchNativeAttnBackend
from sglang.srt.mem_cache.dsa_indexer_layout import cli_indexer_mask
from sglang.srt.mem_cache.kv_offload.budget import (
    OffloadBudget,
    available_host_bytes,
    packed_coordinator_fixed_bytes,
    packed_offload_workspace_bytes,
)
from sglang.srt.mem_cache.kv_offload.packed_host import packed_host_enabled
from sglang.srt.mem_cache.memory_pool import (
    DSATokenToKVPool,
    HybridLinearKVPool,
    HybridReqToTokenPool,
    MHATokenToKVPool,
    MLATokenToKVPool,
    NativeSparseMHATokenToKVPool,
    NativeSparseMLATokenToKVPool,
    NPUMHATokenToKVPool,
    ReqToTokenPool,
    SWAKVPool,
    get_dsa_packed_kv_cache_dim,
)
from sglang.srt.mem_cache.memory_pool.kv_allocator import KVAllocator
from sglang.srt.mem_cache.memory_pool.memory_pool_l15_wrapper import TokenToKVPoolL15Wrapper
from sglang.srt.npu.offload_runtime import host_offload_enabled
from sglang.srt.utils import get_colorful_logger, is_npu, is_sm90_supported

__is_npu__ = is_npu()
if not __is_npu__:
    from sglang.srt.layers.attention.flashinfer_backend import FlashInferAttnBackend

    if is_sm90_supported():
        from sglang.srt.layers.attention.dsa_backend import DpskSparseAttnBackend
        from sglang.srt.layers.attention.flash_attention_backend import FlashAttentionBackend
        from sglang.srt.layers.attention.flashmla_backend import FlashMLABackend
    from sglang.srt.layers.attention.flashinfer_mla_backend import FlashInferMLAAttnBackend, FlashInferMLATBOBackend
    from sglang.srt.layers.attention.torch_native_mla_backend import TorchNativeMlaAttnBackend
    from sglang.srt.layers.attention.triton_backend import TritonAttnBackend
else:
    from sglang.srt.layers.attention.npu_mla_backend import NpuAttnBackend, NpuMLAAttnBackend

from sglang.srt.layers.attention.dp_attention import get_attention_tp_size, get_draft_attention_tp_size
from sglang.srt.utils import get_available_gpu_memory

logger = get_colorful_logger(__name__)


SGLANG_CI_SMALL_KV_SIZE = os.getenv("SGLANG_CI_SMALL_KV_SIZE", None)


def _needs_kvp_ownership_wrapper(args) -> bool:
    """Return whether the KV pool must expose KVP ownership metadata."""
    return bool(args.enable_mla_l1_5_cache or getattr(args, "kvp_size", 1) > 1)


class AttnInitializer:
    @staticmethod
    def validate_after_sfad(model_runner):
        args = model_runner.server_args
        if (getattr(args, "offload_cache_update_timing", "before_attention") != "after_sfad"
                or model_runner.is_draft_worker):
            return
        config = model_runner.model_config.hf_text_config
        if not (
            __is_npu__ and host_offload_enabled(args) and is_dsa(config)
            and args.disaggregation_mode == "decode"
            and getattr(config, "cli_factor", 1) == 2
            and args.attention_backend == "npu_mla" and args.kvp_size == 8
            and args.page_size == 128
            and (not args.npu_dsa_kv_int8 or args.npu_dsa_packed_kv_row_bytes == 672)
            and args.pp_size == 1 and args.npu_disable_kv_nz
            and args.offload_cache_update_policy in ("all", "last_query")
            and model_runner.model_config.kv_lora_rank == 512
            and model_runner.model_config.qk_rope_head_dim == 64
            and get_nsa_index_topk(config) == 2048
        ):
            raise ValueError("after_sfad requires target Host CLI2/KVP8/page128, PP1 npu_mla Decode")

    @staticmethod
    def _get_full_attention_layer_ids(model_runner):
        hf_config = model_runner.model_config.hf_config
        full_attn_layers = getattr(hf_config, "full_attention_layer_ids", None)
        if full_attn_layers is not None:
            return full_attn_layers
        if model_runner.is_longcat_hybrid_linear:
            total_attention_layers = model_runner.model_config.num_attention_layers
            fa_interval = max(int(getattr(hf_config, "fa_interval", 4)), 1)
            return [i for i in range(total_attention_layers) if (i + 1) % fa_interval == 0]
        return list(range(model_runner.model_config.num_attention_layers))

    @staticmethod
    def indexer_layer_mask(model_runner, start_layer, end_layer):
        args = model_runner.server_args
        if not getattr(args, "enable_dsa_indexer_placeholders", False) or model_runner.is_draft_worker:
            return (True,) * (end_layer - start_layer)
        config = model_runner.model_config.hf_text_config
        packed = bool(args.npu_dsa_kv_int8)
        packed_layout = (
            packed
            and args.page_size == 128
            and args.npu_dsa_packed_kv_row_bytes == 672
            and model_runner.dtype == torch.bfloat16
            and model_runner.model_config.kv_lora_rank == 512
            and model_runner.model_config.qk_rope_head_dim == 64
        )
        if (
            model_runner.device != "npu"
            or not is_dsa(config)
            or getattr(config, "cli_factor", 1) != 2
            or not (packed_layout or (not packed and ENV.npu_enable_mla_split_kv_kr))
            or model_runner.index_k_dtype != torch.bfloat16
            or args.attention_backend != "npu_mla"
        ):
            raise ValueError(
                "Indexer placeholders require NPU CLI2 DSA with BF16 IndexK, "
                "npu_mla and split BF16 or page128/row672 packed BF16-activation KV"
            )
        if args.disaggregation_mode != "null" and args.disaggregation_transfer_backend != "mooncake_async":
            raise ValueError("Indexer placeholders require mooncake_async for P/D")
        if args.pp_size > 1 and args.pp_layer_nums is None:
            raise ValueError("Indexer placeholders with PP require explicit pp_layer_nums")
        return cli_indexer_mask(start_layer, end_layer, config.cli_factor)

    @staticmethod
    def modify_args(model_runner):
        # Model-specific adjustment
        if model_runner.model_config.attention_arch == AttentionArch.MLA and not __is_npu__:
            if model_runner.is_kimi_linear or model_runner.is_longcat_hybrid_linear:
                # Resolve full-attn MLA backend with the same rules as plain MLA.
                # `--attention-backend hybrid_linear_attn` means "unspecified" for full attn.
                req = model_runner.server_args.attention_backend
                if req == "hybrid_linear_attn":
                    req = None
                if is_dsa(model_runner.model_config.hf_config):
                    full = "dsa"
                elif model_runner.server_args.enable_flashinfer_mla:
                    if req == "flashmla":
                        full = "flashmla"
                    else:
                        full = "flashinfer_mla"
                else:
                    full = req if req in ("flashmla", "flashinfer_mla", "triton") else "triton"
                model_runner.full_attention_backend = full
                model_runner.server_args.full_attention_backend = full
                model_runner.server_args.attention_backend = "hybrid_linear_attn"
                logger.info(f"Hybrid linear mode is turned on. Use hybrid_linear_attn backend (full attention: {full}).")
            elif is_dsa(model_runner.model_config.hf_config):
                model_runner.server_args.attention_backend = "dsa"
            elif model_runner.server_args.enable_flashinfer_mla:
                if model_runner.server_args.attention_backend == "duo_attn":
                    logger.info("MLA duo-attention is turned on. Use duo attn backend.")
                elif model_runner.server_args.attention_backend != "flashmla":
                    model_runner.server_args.attention_backend = "flashinfer_mla"
                    logger.info("MLA optimization is turned on. Use flashinfer mla backend.")
                else:
                    logger.info("MLA optimization is turned on. Use flashmla backend.")
            else:
                logger.info("MLA optimization is turned on. Use triton backend.")
                model_runner.server_args.attention_backend = "triton"

        # Set drafter backend default after MLA optimization logic
        if model_runner.server_args.drafter_attention_backend is None:
            model_runner.server_args.drafter_attention_backend = model_runner.server_args.attention_backend
            logger.info(f"Drafter attention backend set to: {model_runner.server_args.drafter_attention_backend}")

        # Check if the model is using hybrid SWA
        # if (
        #     not model_runner.server_args.disable_hybrid_swa_memory
        #     and model_runner.sliding_window_size is not None
        #     and model_runner.sliding_window_size > 0
        # ):
        #     architectures = model_runner.model_config.hf_config.architectures
        #     if architectures and not any("Llama4" in arch for arch in architectures):
        #         model_runner.is_hybrid = model_runner.model_config.is_hybrid = True

        if model_runner.mambaish_config:
            logger.warning("Hybrid GDN model detected, disable radix cache")
            model_runner.server_args.disable_radix_cache = True
            model_runner.server_args.attention_backend = "hybrid_linear_attn"
            if model_runner.server_args.max_mamba_cache_size is None:
                if model_runner.server_args.max_running_requests is not None:
                    model_runner.server_args.max_mamba_cache_size = model_runner.server_args.max_running_requests
                else:
                    model_runner.server_args.max_mamba_cache_size = 512
            if model_runner.is_hybrid_gdn:
                model_runner.server_args.max_mamba_cache_size = model_runner.server_args.max_mamba_cache_size // (
                    model_runner.server_args.dp_size if model_runner.server_args.enable_dp_attention else 1
                )

    @staticmethod
    def init_memory_pool(
        model_runner,
        total_gpu_memory: int,
        max_num_reqs: Optional[int] = None,
        max_total_tokens: Optional[int] = None,
        page_size: int = 64,
    ):
        AttnInitializer.validate_after_sfad(model_runner)
        args = model_runner.server_args
        if (args.offload_cache and args.offload_cache_update_policy == "last_query"
                and not model_runner.is_draft_worker):
            config = model_runner.model_config.hf_text_config
            if not (
                __is_npu__ and host_offload_enabled(args) and is_dsa(config)
                and getattr(config, "cli_factor", 1) == 2
                and (not args.npu_dsa_kv_int8 or args.npu_dsa_packed_kv_row_bytes == 672)
                and model_runner.dtype == torch.bfloat16
                and args.index_k_dtype == "bf16"
                and model_runner.model_config.kv_lora_rank == 512
                and model_runner.model_config.qk_rope_head_dim == 64
                and get_nsa_index_topk(config) == 2048
            ):
                raise ValueError("last_query requires Host CLI2 target with BF16 activation/IndexK and KV512+rope64/TopK2048")
        if model_runner.server_args.kv_cache_dtype == "auto":
            model_runner.kv_cache_dtype = model_runner.dtype
        elif model_runner.server_args.kv_cache_dtype == "fp8_e4m3":
            model_runner.kv_cache_dtype = torch.float8_e4m3fn
        else:
            raise ValueError(f"Unsupported kv_cache_dtype: {model_runner.server_args.kv_cache_dtype}.")
        if model_runner.is_draft_worker and model_runner.spec_algorithm.is_dflash():
            model_runner.kv_cache_dtype = torch.bfloat16
        model_runner.kv_cache_quant_method = model_runner.server_args.kv_cache_quant_method
        if model_runner.server_args.index_k_dtype == "fp8_e4m3":
            model_runner.index_k_dtype = torch.float8_e4m3fn
        elif model_runner.server_args.index_k_dtype == "bf16":
            model_runner.index_k_dtype = torch.bfloat16
        else:
            raise ValueError(f"{model_runner.server_args.index_k_dtype=}")

        if model_runner.server_args.npu_dsa_kv_int8 and not (
            __is_npu__ and model_runner.model_config.attention_arch == AttentionArch.MLA and is_dsa(model_runner.model_config.hf_config)
        ):
            raise ValueError("--npu-dsa-kv-int8 is only supported for NPU DSA MLA models.")

        model_runner.max_total_num_tokens = AttnInitializer.profile_max_num_token(
            model_runner, total_gpu_memory, max_num_reqs=max_num_reqs
        )
        logger.info(f"profiled max_total_num_tokens={model_runner.max_total_num_tokens}")
        # The profiled value may vary slightly across runs (e.g., due to dynamic GPU memory or driver.
        # When graph cache is enabled, restore max_total_num_tokens from file
        # to ensure KV pool size matches the cached graph's shape.
        if global_server_args_dict["npu_enable_graph_cache"]:
            max_total_num_tokens_file_path = os.path.join(
                global_server_args_dict["npu_graph_cache_path"], "num_token_file_path", str(model_runner.global_rank)
            )
            os.makedirs(max_total_num_tokens_file_path, exist_ok=True)
            max_total_num_tokens_file = os.path.join(max_total_num_tokens_file_path, "max_total_num_tokens.txt")
            if os.path.exists(max_total_num_tokens_file):
                with open(max_total_num_tokens_file) as f:
                    model_runner.max_total_num_tokens = int(f.read().strip())
                logger.info(f"graph cache: restored max_total_num_tokens={model_runner.max_total_num_tokens} from {max_total_num_tokens_file}")
            else:
                with open(max_total_num_tokens_file, "w") as f:
                    f.write(str(model_runner.max_total_num_tokens))
                logger.info(f"graph cache: saved max_total_num_tokens={model_runner.max_total_num_tokens} to {max_total_num_tokens_file}")

        if SGLANG_CI_SMALL_KV_SIZE:
            model_runner.max_total_num_tokens = int(SGLANG_CI_SMALL_KV_SIZE)
        # this is the max token can be stored locally, will be different when enable_mla_l1_5_cache
        if model_runner.server_args.enable_mla_l1_5_cache and not model_runner.is_draft_worker:
            assert model_runner.model_config.attention_arch == AttentionArch.MLA, "should be mla model"
            assert not getattr(model_runner.model_config.hf_config, "use_nsa_mla", False)
            model_runner.max_total_num_tokens = model_runner.max_total_num_tokens * model_runner.server_args.attn_tp_size

        if max_num_reqs is None:
            max_num_reqs = min(
                max(
                    int(model_runner.max_total_num_tokens / model_runner.model_config.context_len * 512),
                    2048,
                ),
                4096,
            )

        if not model_runner.spec_algorithm.is_none():
            # Both target and draft models call `init_memory_pool`
            # The target model runs first and profiles max_total_pages.
            # The draft model just use it, avoiding redundant profiling.
            if model_runner.is_draft_worker:
                if model_runner.spec_algorithm.is_dflash():
                    # draft pool 须覆盖 target 发放的 slot 值域 (draft 与 target 共享 kv_allocator)。
                    slot_domain_size: int = max(
                        model_runner.kv_allocator.size,
                        model_runner.server_args.draft_runner_cache_size,
                    )
                    if model_runner.max_total_num_tokens < slot_domain_size:
                        # 不可静默截断: 越界写在 NPU 上不报错, 只会静默掉 accept 率。
                        raise RuntimeError(
                            f"DFLASH draft pool 无法覆盖 target slot 值域: "
                            f"profile={model_runner.max_total_num_tokens} < {slot_domain_size}。"
                            f"请调低 --mem-fraction-static 或 --max-running-requests。"
                        )
                    logger.info(f"DFLASH draft pool: {model_runner.max_total_num_tokens} -> {slot_domain_size} (target slot 值域)")
                    model_runner.max_total_num_tokens = slot_domain_size
                else:
                    model_runner.max_total_num_tokens = model_runner.server_args.draft_runner_cache_size
            else:
                model_runner.server_args.draft_runner_cache_size = (
                    model_runner.max_total_num_tokens + max_num_reqs * model_runner.server_args.speculative_num_steps + 100
                )

        # Below, max_num_reqs sizes slot-indexed storage and KVP reserved pages,
        # not the running/graph batch. Target and draft runners each receive
        # the original per-DP limit, so add the same extra capacity once here.
        if model_runner.server_args.disaggregation_mode == "decode":
            extra_slots = model_runner.server_args.disaggregation_decode_extra_slots
            logger.info(
                "Decode request capacity: running=%s, extra_slots=%s, total_slots=%s",
                max_num_reqs,
                extra_slots,
                max_num_reqs + extra_slots,
            )
            max_num_reqs += extra_slots

        if max_total_tokens is not None:
            if max_total_tokens > model_runner.max_total_num_tokens:
                logger.warning(
                    f"max_total_tokens={max_total_tokens} is larger than the profiled value "
                    f"{model_runner.max_total_num_tokens}. "
                    f"Use the profiled value instead."
                )
            model_runner.max_total_num_tokens = min(model_runner.max_total_num_tokens, max_total_tokens)

        if model_runner.max_total_num_tokens <= 0:
            raise RuntimeError("Not enough memory. Please try to increase --mem-fraction-static.")

        model_runner.page_size = page_size
        if __is_npu__:
            # 加参数--page-size 128
            assert model_runner.page_size == 128

        model_runner.max_total_pages = model_runner.max_total_num_tokens // model_runner.page_size
        model_runner.max_total_num_tokens = model_runner.max_total_pages * model_runner.page_size
        model_runner.local_max_num_tokens = model_runner.max_total_num_tokens
        # each card reserve its 0th page for cuda graph padding
        if model_runner.server_args.enable_mla_l1_5_cache and model_runner.max_total_pages > 0:
            # max_batch_size is setting to max_num_reqs + 1
            # reserved_block_num is 1 + max_batch_size
            reserved_block_num = 1 + max_num_reqs + 1
            # Align to attn_tp_size so local page indices never exceed local pool size.
            model_runner.local_total_pages = model_runner.max_total_num_tokens // model_runner.server_args.attn_tp_size // model_runner.page_size
            model_runner.max_total_pages = model_runner.local_total_pages * model_runner.server_args.attn_tp_size
            model_runner.max_total_pages += reserved_block_num
            model_runner.max_total_num_tokens = model_runner.max_total_pages * model_runner.page_size
            model_runner.local_max_num_tokens = (model_runner.local_total_pages + reserved_block_num) * model_runner.page_size
        # KV-pool-vs-context-len check happens in Scheduler after the GPU
        # memory summary is logged, so the summary is visible when it fails.
        if hasattr(model_runner, "offload_budget"):
            usage = model_runner.offload_budget.usage(
                model_runner.local_max_num_tokens
            )
            if (
                usage["hbm_bytes"] > model_runner.offload_hbm_budget_bytes
                or usage["host_bytes"] > model_runner.offload_host_budget_bytes
            ):
                raise ValueError(
                    "Final offload pool exceeds profiled Host/HBM budget after "
                    "alignment: "
                    + json.dumps(
                        {
                            "local_max_num_tokens": model_runner.local_max_num_tokens,
                            "max_total_num_tokens": model_runner.max_total_num_tokens,
                            "hbm_bytes": usage["hbm_bytes"],
                            "hbm_budget": model_runner.offload_hbm_budget_bytes,
                            "host_bytes": usage["host_bytes"],
                            "host_budget": model_runner.offload_host_budget_bytes,
                        },
                        sort_keys=True,
                    )
                )
        # Added page_size is reserved pad page
        assert model_runner.max_total_num_tokens >= model_runner.model_config.context_len + model_runner.page_size, (
            f"KV Cache allocation too small: {model_runner.max_total_num_tokens=}, "
            f"{model_runner.model_config.context_len=}, {model_runner.page_size=}, "
            "may cause requests to enter Scheduler but cannot be scheduled, causing service to hang"
        )

        if model_runner.req_to_token_pool is None:
            if model_runner.mambaish_config:
                config = model_runner.model_config.hf_config
                (
                    conv_state_shape,
                    temporal_state_shape,
                    conv_dtype,
                    ssm_dtype,
                    mamba_layers,
                ) = config.mamba2_cache_params
                extra_max_context_len = 4
                if model_runner.server_args.speculative_num_draft_tokens is not None:
                    extra_max_context_len += model_runner.server_args.speculative_num_draft_tokens
                model_runner.req_to_token_pool = HybridReqToTokenPool(
                    size=max_num_reqs + 1,
                    max_context_len=model_runner.model_config.context_len + extra_max_context_len,
                    device=model_runner.device,
                    enable_memory_saver=model_runner.server_args.enable_memory_saver,
                    conv_state_shape=conv_state_shape,
                    temporal_state_shape=temporal_state_shape,
                    conv_dtype=conv_dtype,
                    ssm_dtype=ssm_dtype,
                    mamba_layers=mamba_layers,
                    speculative_num_draft_tokens=model_runner.server_args.speculative_num_draft_tokens,
                )
            else:
                # If there is a draft model, let draft model and target model share
                # req_to_token_pool and allocator
                # Add 1 is for spec cuda graph padding.
                model_runner.req_to_token_pool = ReqToTokenPool(
                    size=max_num_reqs + 1,
                    max_context_len=model_runner.model_config.context_len + model_runner.page_size * 4,
                    device=model_runner.device,
                    enable_memory_saver=model_runner.server_args.enable_memory_saver,
                )
        else:
            assert model_runner.is_draft_worker

        if model_runner.kv_allocator is None:
            model_runner.kv_allocator = KVAllocator(
                size=model_runner.max_total_num_tokens,
                device=model_runner.device,
                max_batch_size=max_num_reqs + 1,
                max_context_len=model_runner.model_config.context_len + model_runner.page_size * 4,
                page_size=model_runner.page_size,
            )
        else:
            assert model_runner.is_draft_worker

        if model_runner.is_draft_worker:
            # DFLASH draft 层数取 num_hidden_layers (完整多层 GQA)。
            if "DFlash" in getattr(model_runner.model_config.hf_config, "architectures", [""])[0]:
                model_runner.model_config.num_attention_layers = getattr(model_runner.model_config.hf_config, "num_hidden_layers", 1)
            else:
                model_runner.model_config.num_attention_layers = getattr(model_runner.model_config.hf_config, "num_nextn_predict_layers", 1)

        if model_runner.server_args.pp_size > 1:
            model_runner.attn_start_layer = model_runner.model.attn_start_layer
            model_runner.attn_end_layer = model_runner.model.attn_end_layer
        else:
            model_runner.attn_start_layer = 0
            model_runner.attn_end_layer = model_runner.model_config.num_attention_layers

        if model_runner.model_config.attention_arch == AttentionArch.MLA and not model_runner.mambaish_config:
            # for flash nsa+
            if getattr(model_runner.model_config.hf_config, "use_nsa_mla", False):
                model_runner.token_to_kv_pool = NativeSparseMLATokenToKVPool(
                    size=model_runner.max_total_num_tokens,
                    model_dtype=model_runner.dtype,
                    dtype=model_runner.kv_cache_dtype,
                    quant_method=model_runner.kv_cache_quant_method,
                    kv_lora_rank=model_runner.model_config.hf_config.kv_lora_rank,
                    qk_rope_head_dim=model_runner.model_config.hf_config.qk_rope_head_dim,
                    layer_num=model_runner.model_config.num_attention_layers,
                    device=model_runner.device,
                    enable_memory_saver=model_runner.server_args.enable_memory_saver,
                    max_batch_size=max_num_reqs + 1,
                    max_context_len=model_runner.model_config.context_len + model_runner.page_size * 4,
                    page_size=model_runner.page_size,
                    rank=model_runner.global_rank,
                    compressed_block_stride=model_runner.model_config.hf_config.stride,
                )
            elif is_dsa(model_runner.model_config.hf_text_config):
                model_runner.num_effective_layers = model_runner.attn_end_layer - model_runner.attn_start_layer
                use_offload = bool(
                    model_runner.server_args.offload_cache
                    and not model_runner.is_draft_worker
                    and model_runner.server_args.disaggregation_mode != "prefill"
                )
                sparse_coordinator = None
                if use_offload:
                    from sglang.srt.mem_cache.kv_offload.sparse_coordinator import (
                        SparseCoordinator,
                    )

                    sparse_coordinator = SparseCoordinator(
                        config=model_runner.model_config.hf_text_config,
                        page_size=model_runner.page_size,
                        kv_lora_rank=model_runner.model_config.kv_lora_rank,
                        qk_rope_head_dim=model_runner.model_config.qk_rope_head_dim,
                        layer_num=model_runner.num_effective_layers,
                        max_batch_size=model_runner.server_args.offload_cache_bs,
                        speculative_num_draft_tokens=(
                            model_runner.server_args.speculative_num_draft_tokens
                            if model_runner.server_args.speculative_algorithm
                            else 0
                        ),
                        device=model_runner.device,
                        dtype=model_runner.kv_cache_dtype,
                        start_layer=model_runner.attn_start_layer,
                        packed_row_bytes=(
                            model_runner.server_args.npu_dsa_packed_kv_row_bytes
                            if packed_host_enabled(model_runner.server_args) else 0
                        ),
                    )
                model_runner.token_to_kv_pool = DSATokenToKVPool(
                    model_runner.local_max_num_tokens,
                    model_dtype=model_runner.dtype,
                    dtype=model_runner.kv_cache_dtype,
                    kv_lora_rank=model_runner.model_config.kv_lora_rank,
                    qk_rope_head_dim=model_runner.model_config.qk_rope_head_dim,
                    layer_num=model_runner.num_effective_layers,
                    device=model_runner.device,
                    enable_memory_saver=model_runner.server_args.enable_memory_saver,
                    max_batch_size=max_num_reqs + 1,
                    max_context_len=model_runner.model_config.context_len + model_runner.page_size * 4,
                    page_size=model_runner.page_size,
                    rank=model_runner.global_rank,
                    index_head_dim=get_nsa_index_head_dim(model_runner.model_config.hf_text_config),
                    index_dtype=model_runner.index_k_dtype,
                    npu_dsa_kv_int8=model_runner.server_args.npu_dsa_kv_int8,
                    npu_dsa_packed_kv_row_bytes=model_runner.server_args.npu_dsa_packed_kv_row_bytes,
                    start_layer=model_runner.attn_start_layer,
                    end_layer=model_runner.attn_end_layer,
                    offload_cache=use_offload,
                    sparse_coordinator=sparse_coordinator,
                    indexer_layer_mask=AttnInitializer.indexer_layer_mask(model_runner, model_runner.attn_start_layer, model_runner.attn_end_layer),
                )
                if _needs_kvp_ownership_wrapper(model_runner.server_args):
                    model_runner.token_to_kv_pool = TokenToKVPoolL15Wrapper(model_runner.token_to_kv_pool)
            else:
                model_runner.num_effective_layers = model_runner.attn_end_layer - model_runner.attn_start_layer
                model_runner.token_to_kv_pool = MLATokenToKVPool(
                    model_runner.local_max_num_tokens,
                    model_dtype=model_runner.dtype,
                    dtype=model_runner.kv_cache_dtype,
                    quant_method=model_runner.kv_cache_quant_method,
                    kv_lora_rank=model_runner.model_config.kv_lora_rank,
                    qk_rope_head_dim=model_runner.model_config.qk_rope_head_dim,
                    layer_num=model_runner.num_effective_layers,
                    device=model_runner.device,
                    enable_memory_saver=model_runner.server_args.enable_memory_saver,
                    max_batch_size=max_num_reqs + 1,
                    max_context_len=model_runner.model_config.context_len + model_runner.page_size * 4,
                    page_size=model_runner.page_size,
                    rank=model_runner.global_rank,
                    start_layer=model_runner.attn_start_layer,
                    end_layer=model_runner.attn_end_layer,
                )
                if model_runner.server_args.enable_mla_l1_5_cache:
                    model_runner.token_to_kv_pool = TokenToKVPoolL15Wrapper(model_runner.token_to_kv_pool)
        elif model_runner.mambaish_config:
            extra_args = {}
            if model_runner.model_config.attention_arch == AttentionArch.MLA:
                extra_args = {
                    "model_dtype": model_runner.dtype,
                    "quant_method": model_runner.kv_cache_quant_method,
                    "use_mla": True,
                    "kv_lora_rank": model_runner.model_config.kv_lora_rank,
                    "qk_rope_head_dim": model_runner.model_config.qk_rope_head_dim,
                }
                if is_dsa(model_runner.model_config.hf_config):
                    extra_args.update(
                        {
                            "use_dsa": True,
                            "index_head_dim": get_nsa_index_head_dim(model_runner.model_config.hf_config),
                            "index_dtype": model_runner.index_k_dtype,
                            "npu_dsa_kv_int8": model_runner.server_args.npu_dsa_kv_int8,
                            "npu_dsa_packed_kv_row_bytes": model_runner.server_args.npu_dsa_packed_kv_row_bytes,
                        }
                    )
            model_runner.token_to_kv_pool = HybridLinearKVPool(
                size=model_runner.max_total_num_tokens,
                dtype=model_runner.kv_cache_dtype,
                head_num=model_runner.model_config.get_num_kv_heads(get_attention_tp_size()),
                head_dim=model_runner.model_config.head_dim,
                # if draft worker, we only need 1 attention layer's kv pool
                full_attention_layer_ids=([0] if model_runner.is_draft_worker else AttnInitializer._get_full_attention_layer_ids(model_runner)),
                enable_kvcache_transpose=False,
                device=model_runner.device,
                max_batch_size=max_num_reqs * 2 + 1,
                max_context_len=model_runner.model_config.context_len + model_runner.page_size * 4,
                page_size=model_runner.page_size,
                rank=model_runner.global_rank,
                **extra_args,
            )
        else:
            # for dpsk lite attn experiment
            if getattr(model_runner.model_config.hf_config, "use_nsa", False):
                model_runner.token_to_kv_pool = NativeSparseMHATokenToKVPool(
                    model_runner.max_total_num_tokens,
                    dtype=model_runner.kv_cache_dtype,
                    head_num=model_runner.model_config.get_num_kv_heads(get_attention_tp_size()),
                    head_dim=model_runner.model_config.head_dim,
                    compressed_block_stride=model_runner.model_config.hf_config.stride,
                    layer_num=model_runner.model_config.num_attention_layers,
                    device=model_runner.device,
                    enable_memory_saver=model_runner.server_args.enable_memory_saver,
                    max_batch_size=max_num_reqs + 1,
                    max_context_len=model_runner.model_config.context_len + model_runner.page_size * 4,
                    page_size=model_runner.page_size,
                    rank=model_runner.global_rank,
                )
            elif getattr(model_runner.model_config.hf_config, "use_nsa_mla", False):
                model_runner.token_to_kv_pool = NativeSparseMLATokenToKVPool(
                    size=model_runner.max_total_num_tokens,
                    model_dtype=model_runner.dtype,
                    dtype=model_runner.kv_cache_dtype,
                    quant_method=model_runner.kv_cache_quant_method,
                    kv_lora_rank=model_runner.model_config.hf_config.kv_lora_rank,
                    qk_rope_head_dim=model_runner.model_config.hf_config.qk_rope_head_dim,
                    layer_num=model_runner.model_config.num_attention_layers,
                    device=model_runner.device,
                    enable_memory_saver=model_runner.server_args.enable_memory_saver,
                    max_batch_size=max_num_reqs + 1,
                    max_context_len=model_runner.model_config.context_len + model_runner.page_size * 4,
                    page_size=model_runner.page_size,
                    rank=model_runner.global_rank,
                    compressed_block_stride=model_runner.model_config.hf_config.stride,
                )
            else:
                if model_runner.is_hybrid:
                    model_runner.token_to_kv_pool = SWAKVPool(
                        size=model_runner.full_max_total_num_tokens,
                        size_swa=model_runner.swa_max_total_num_tokens,
                        dtype=model_runner.kv_cache_dtype,
                        head_num=model_runner.model_config.get_num_kv_heads(get_attention_tp_size()),
                        head_dim=model_runner.model_config.head_dim,
                        swa_attention_layer_ids=model_runner.model_config.swa_attention_layer_ids,
                        full_attention_layer_ids=model_runner.model_config.full_attention_layer_ids,
                        enable_kvcache_transpose=False,
                        device=model_runner.device,
                    )
                elif model_runner.server_args.drafter_attention_backend == "npu":
                    model_runner.token_to_kv_pool = NPUMHATokenToKVPool(
                        model_runner.max_total_num_tokens,
                        dtype=model_runner.kv_cache_dtype,
                        # head_num 口径须与 DFlashAttention 的 num_kv_heads 一致 (get_attention_tp_size)
                        head_num=model_runner.model_config.get_num_kv_heads(
                            get_attention_tp_size() if model_runner.spec_algorithm.is_dflash() else get_draft_attention_tp_size()
                        ),
                        head_dim=model_runner.model_config.head_dim,
                        layer_num=model_runner.model_config.num_attention_layers,
                        device=model_runner.device,
                        enable_memory_saver=model_runner.server_args.enable_memory_saver,
                        max_batch_size=max_num_reqs * 2 + 1,
                        max_context_len=model_runner.model_config.context_len + model_runner.page_size * 4,
                        page_size=model_runner.page_size,
                        rank=model_runner.global_rank,
                        use_nd_layout=model_runner.spec_algorithm.is_dflash(),
                    )
                else:
                    model_runner.num_effective_layers = model_runner.attn_end_layer - model_runner.attn_start_layer
                    model_runner.token_to_kv_pool = MHATokenToKVPool(
                        model_runner.max_total_num_tokens,
                        dtype=model_runner.kv_cache_dtype,
                        head_num=model_runner.model_config.get_num_kv_heads(get_attention_tp_size()),
                        head_dim=model_runner.model_config.head_dim,
                        layer_num=model_runner.model_config.num_attention_layers,
                        device=model_runner.device,
                        enable_memory_saver=model_runner.server_args.enable_memory_saver,
                        max_batch_size=max_num_reqs + 1,
                        max_context_len=model_runner.model_config.context_len + model_runner.page_size * 4,
                        page_size=model_runner.page_size,
                        rank=model_runner.global_rank,
                        start_layer=model_runner.attn_start_layer,
                        end_layer=model_runner.attn_end_layer,
                    )

        # Add a ref count to slots, used for checking when server is idle
        model_runner.token_to_kv_pool.set_token_slot_refs(model_runner.kv_allocator.token_slot_refs)
        logger.info(
            f"Init Memory pool {model_runner.token_to_kv_pool.__class__.__name__} end. "
            f"avail mem={get_available_gpu_memory(model_runner.device, model_runner.gpu_id):.2f} GB. "
            f"model_runner.max_total_num_tokens={model_runner.max_total_num_tokens}"
        )

    @staticmethod
    def init_attention_backend(model_runner):
        """Init attention kernel backend."""
        # Select appropriate backend based on worker type
        if model_runner.is_draft_worker:
            backend_name = model_runner.server_args.drafter_attention_backend
        else:
            backend_name = model_runner.server_args.attention_backend

        if backend_name == "flashinfer":
            model_runner.attn_backend = FlashInferAttnBackend(model_runner)
        elif backend_name == "triton":
            model_runner.attn_backend = TritonAttnBackend(model_runner)
        elif backend_name == "torch_native":
            model_runner.attn_backend = TorchNativeAttnBackend(model_runner)
        elif backend_name == "torch_native_mla":
            model_runner.attn_backend = TorchNativeMlaAttnBackend(model_runner)
        elif backend_name == "flashmla":
            model_runner.attn_backend = FlashMLABackend(model_runner)
        elif backend_name == "fa3":
            model_runner.attn_backend = FlashAttentionBackend(model_runner)
        elif backend_name == "flashinfer_mla":
            if model_runner.server_args.enable_tbo:
                model_runner.attn_backend = FlashInferMLATBOBackend(model_runner)
            else:
                model_runner.attn_backend = FlashInferMLAAttnBackend(model_runner)
        elif backend_name == "npu_mla":
            logger.info("========select NpuMLAAttnBackend=============")
            model_runner.attn_backend = NpuMLAAttnBackend(model_runner)
        elif backend_name == "npu":
            logger.info("========select NpuAttnBackend=============")
            model_runner.attn_backend = NpuAttnBackend(model_runner)
        elif backend_name == "dsa":
            model_runner.attn_backend = DpskSparseAttnBackend(model_runner)
        elif backend_name == "hybrid_linear_attn":
            assert model_runner.mambaish_config, "hybrid_linear_attn backend can only be used with mambaish models."
            from sglang.srt.layers.attention.hybrid_linear_attn_backend import HybridLinearAttnBackend, KimiLinearAttnBackend, MambaAttnBackend

            linear_attn_backend_cls = (
                MambaAttnBackend
                if model_runner.is_hybrid_gdn
                else KimiLinearAttnBackend
                if (model_runner.is_kimi_linear or model_runner.is_longcat_hybrid_linear)
                else None
            )
            if model_runner.is_hybrid_gdn:
                full_attn_backend_cls = FlashInferAttnBackend
            elif model_runner.is_longcat_hybrid_linear and is_dsa(model_runner.model_config.hf_config) and not is_npu():
                full_attn_backend_cls = DpskSparseAttnBackend
            elif model_runner.is_longcat_hybrid_linear and is_npu():
                full_attn_backend_cls = NpuMLAAttnBackend
            elif model_runner.is_kimi_linear or model_runner.is_longcat_hybrid_linear:
                # Resolved in modify_args to a concrete MLA backend name.
                name = getattr(model_runner, "full_attention_backend", "triton")
                if name == "flashmla":
                    full_attn_backend_cls = FlashMLABackend
                elif name == "flashinfer_mla":
                    full_attn_backend_cls = FlashInferMLAAttnBackend
                elif name == "dsa":
                    full_attn_backend_cls = DpskSparseAttnBackend
                else:
                    full_attn_backend_cls = TritonAttnBackend
            else:
                full_attn_backend_cls = None
            linear_attn_backend = linear_attn_backend_cls(model_runner)
            full_attn_backend = full_attn_backend_cls(model_runner)
            full_attn_layers = AttnInitializer._get_full_attention_layer_ids(model_runner)
            model_runner.attn_backend = HybridLinearAttnBackend(full_attn_backend, linear_attn_backend, full_attn_layers)
        else:
            raise ValueError(f"Invalid attention backend: {backend_name}")

    @staticmethod
    def _estimate_dflash_draft_weight_gb(model_runner) -> float:
        """估算 DFLASH draft 权重在本卡的显存占用 (GB), 供 target profile 预扣。

        直读 safetensors header 按真实 dtype 与各模块分片方式累加 (不用 index.json 的
        total_size, 它按 bf16 记账会高估量化 ckpt), 加上 __init__ 另行分配的
        embed_tokens/lm_head, 再乘系数覆盖加载期开销。
        系数默认 2, 可用 SGLANG_DFLASH_DRAFT_WEIGHT_FACTOR 按日志的实测值校准。
        """
        draft_path = model_runner.server_args.speculative_draft_model_path
        if not draft_path:
            return 0.0
        try:
            import glob
            import json
            import struct

            dtype_bytes = {
                "I8": 1,
                "U8": 1,
                "F8_E4M3": 1,
                "F8_E5M2": 1,
                "BF16": 2,
                "F16": 2,
                "I16": 2,
                "F32": 4,
                "I32": 4,
                "I64": 8,
                "F64": 8,
            }
            attn_tp = max(1, get_attention_tp_size())
            dense_tp = max(1, model_runner.server_args.dense_tp_size)
            ckpt_bytes = 0.0
            for path in sorted(glob.glob(os.path.join(draft_path, "*.safetensors"))):
                with open(path, "rb") as f:
                    header_len = struct.unpack("<Q", f.read(8))[0]
                    header = json.loads(f.read(header_len))
                for name, meta in header.items():
                    if name == "__metadata__":
                        continue
                    elems = 1
                    for d in meta["shape"]:
                        elems *= d
                    nbytes = elems * dtype_bytes.get(meta["dtype"], 2)
                    if ".mlp." in name:
                        ckpt_bytes += nbytes / dense_tp
                    elif ".self_attn." in name and "_norm" not in name:
                        ckpt_bytes += nbytes / attn_tp
                    else:  # fc / norm / 各类 scale: 不分片
                        ckpt_bytes += nbytes
            if ckpt_bytes <= 0:
                return 0.0

            from sglang.srt.utils.hf_transformers_utils import get_config

            cfg = get_config(
                draft_path,
                trust_remote_code=model_runner.server_args.trust_remote_code,
                revision=model_runner.server_args.revision,
                is_draft_worker=True,
            )
            vocab = int(getattr(cfg, "vocab_size", 0))
            hidden = int(getattr(cfg, "hidden_size", 0))
            embed_bytes = vocab * hidden * 2
            if not global_server_args_dict["enable_dp_attention"]:
                embed_bytes //= attn_tp
            lmhead_tp = model_runner.server_args.npu_lmhead_tp_size
            if lmhead_tp and lmhead_tp > 1:
                lm_head_bytes = vocab * hidden * 2 // lmhead_tp
            elif global_server_args_dict["enable_dp_attention"]:
                lm_head_bytes = vocab * hidden * 2
            else:
                lm_head_bytes = vocab * hidden * 2 // attn_tp

            static_gb = (ckpt_bytes + embed_bytes + lm_head_bytes) / (1 << 30)
            factor = float(os.environ.get("SGLANG_DFLASH_DRAFT_WEIGHT_FACTOR", 2))
            return static_gb * factor
        except Exception as e:  # noqa: BLE001 - 估算失败不应阻断启动
            logger.warning(f"DFLASH draft 权重估算失败 ({e}), 跳过预扣")
            return 0.0

    @staticmethod
    def profile_max_num_token(
        model_runner,
        total_gpu_memory: int,
        max_num_reqs: Optional[int] = None,
    ):
        placeholder_bytes = 0
        # draft worker 跳过 world all_reduce (非末 PP stage 不构造 draft, 会等不到伙伴)。
        available_gpu_memory = get_available_gpu_memory(
            model_runner.device,
            model_runner.gpu_id,
            distributed=(model_runner.tp_size > 1) and not model_runner.is_draft_worker,
            cpu_group=get_world_group().cpu_group,
        )

        if model_runner.mambaish_config:
            num_attention_layers = len(AttnInitializer._get_full_attention_layer_ids(model_runner))
        else:
            num_attention_layers = model_runner.model_config.num_attention_layers

        if model_runner.server_args.pp_size > 1:
            if model_runner.server_args.pp_layer_nums is not None:
                # 本 stage 真实层数 (×2 同 num_attention_layers 的 SCMoE 口径)。
                num_attention_layers = model_runner.server_args.pp_layer_nums[model_runner.pp_rank] * 2
            else:
                num_attention_layers = (num_attention_layers + model_runner.server_args.pp_size) // model_runner.server_args.pp_size * 2
            logger.info(f"will creating {num_attention_layers=} attention layers kv cache")

        # Include the draft KV pools that are allocated on this PP stage.
        # DFLASH draft 是 GQA, 单价与 target MLA/DSA 不同, 按真实字节数在 cell_size 上单独累加。
        dflash_draft_cell_size: int = 0
        if model_runner.server_args.speculative_algorithm is not None and (
            model_runner.server_args.pp_size == 1 or model_runner.pp_rank == model_runner.server_args.pp_size - 1
        ):
            if model_runner.spec_algorithm.is_dflash():
                if not model_runner.is_draft_worker:
                    from sglang.srt.utils.hf_transformers_utils import get_config

                    draft_config = get_config(
                        model_runner.server_args.speculative_draft_model_path,
                        trust_remote_code=model_runner.server_args.trust_remote_code,
                        revision=model_runner.server_args.revision,
                        is_draft_worker=True,
                    )
                    draft_layers = int(getattr(draft_config, "num_hidden_layers", 1))
                    draft_total_kv_heads = int(
                        getattr(
                            draft_config,
                            "num_key_value_heads",
                            getattr(draft_config, "num_attention_heads", 1),
                        )
                    )
                    draft_kv_heads = max(1, draft_total_kv_heads // get_attention_tp_size())
                    draft_head_dim = int(getattr(draft_config, "head_dim", model_runner.model_config.head_dim))
                    # k + v 各一份; l1.5 下 target KV 按 attn_tp 分片而 draft pool 不分片, 单价乘 attn_tp
                    dflash_draft_cell_size = draft_kv_heads * draft_head_dim * draft_layers * 2 * torch._utils._element_size(torch.bfloat16)
                    if model_runner.server_args.enable_mla_l1_5_cache and not model_runner.is_draft_worker:
                        dflash_draft_cell_size *= model_runner.server_args.attn_tp_size
                    logger.info(f"DFLASH draft cell_size={dflash_draft_cell_size} bytes/token (layers={draft_layers}, kv_heads={draft_kv_heads})")
            else:
                # Multi-head EAGLE creates one independent KV pool per draft step.
                draft_pool_count = (
                    model_runner.server_args.speculative_num_steps
                    if model_runner.server_args.is_multi_head_eagle
                    else 1
                )
                num_attention_layers += draft_pool_count
        if model_runner.model_config.attention_arch == AttentionArch.MLA:
            if is_dsa(model_runner.model_config.hf_config):
                index_head_dim = get_nsa_index_head_dim(model_runner.model_config.hf_config)
                if model_runner.index_k_dtype == torch.float8_e4m3fn:
                    index_cell_size = index_head_dim + index_head_dim // DSATokenToKVPool.quant_block_size * torch._utils._element_size(torch.float32)
                else:
                    index_cell_size = index_head_dim * torch._utils._element_size(model_runner.index_k_dtype)

                if model_runner.server_args.npu_dsa_kv_int8:
                    kv_cell_size = get_dsa_packed_kv_cache_dim(
                        model_runner.model_config.kv_lora_rank,
                        model_runner.model_config.qk_rope_head_dim,
                        model_runner.dtype,
                        DSATokenToKVPool.quant_block_size,
                        model_runner.server_args.npu_dsa_packed_kv_row_bytes,
                    )
                elif model_runner.kv_cache_quant_method == "per_token_head":
                    kv_cell_size = (
                        model_runner.model_config.kv_lora_rank * torch._utils._element_size(model_runner.kv_cache_dtype)
                        + model_runner.model_config.qk_rope_head_dim * torch._utils._element_size(model_runner.dtype)
                        + 1 * torch._utils._element_size(torch.float32)
                    )
                else:
                    kv_cell_size = (model_runner.model_config.kv_lora_rank + model_runner.model_config.qk_rope_head_dim) * torch._utils._element_size(
                        model_runner.kv_cache_dtype
                    )

                cell_size = (kv_cell_size + index_cell_size) * num_attention_layers
                if getattr(model_runner.server_args, "enable_dsa_indexer_placeholders", False):
                    args = model_runner.server_args
                    if args.pp_size > 1:
                        if args.pp_layer_nums is None:
                            raise ValueError("Indexer placeholders with PP require explicit pp_layer_nums")
                        first = 2 * sum(args.pp_layer_nums[: model_runner.pp_rank])
                        count = 2 * args.pp_layer_nums[model_runner.pp_rank]
                    else:
                        first = 0
                        count = model_runner.model_config.num_attention_layers
                    mask = AttnInitializer.indexer_layer_mask(model_runner, first, first + count)
                    inactive = len(mask) - sum(mask)
                    cell_size -= inactive * index_cell_size
                    # profile precedes init_memory_pool's page_size assignment.
                    placeholder_bytes = inactive * index_cell_size * args.page_size
                logger.info(
                    "DSA profile cell_size=%s bytes/token (kv=%s, index=%s, layers=%s, npu_dsa_kv_int8=%s, kv_cache_dtype=%s, index_k_dtype=%s).",
                    cell_size,
                    kv_cell_size,
                    index_cell_size,
                    num_attention_layers,
                    model_runner.server_args.npu_dsa_kv_int8,
                    model_runner.kv_cache_dtype,
                    model_runner.index_k_dtype,
                )
            elif model_runner.kv_cache_quant_method == "per_token_head":
                cell_size = (
                    model_runner.model_config.kv_lora_rank * torch._utils._element_size(model_runner.kv_cache_dtype)
                    + model_runner.model_config.qk_rope_head_dim * torch._utils._element_size(model_runner.dtype)
                    + 1 * torch._utils._element_size(torch.float32)
                ) * num_attention_layers
            else:
                cell_size = (
                    (model_runner.model_config.kv_lora_rank + model_runner.model_config.qk_rope_head_dim)
                    * num_attention_layers
                    * torch._utils._element_size(model_runner.kv_cache_dtype)
                )
        else:
            cell_size = (
                model_runner.model_config.get_num_kv_heads(get_attention_tp_size())
                * model_runner.model_config.head_dim
                * num_attention_layers
                * 2
                * torch._utils._element_size(model_runner.kv_cache_dtype)
            )
        # DFLASH draft 与 target 共用这一次 profile, 二者 max_total_num_tokens 天然一致。
        cell_size += dflash_draft_cell_size
        rest_memory = available_gpu_memory - total_gpu_memory * (1 - model_runner.mem_fraction_static)
        logger.info(f"{rest_memory=} {available_gpu_memory=} {total_gpu_memory=} {model_runner.mem_fraction_static}")
        if model_runner.mambaish_config:
            rest_memory -= model_runner.server_args.max_mamba_cache_size * model_runner.model_config.hf_config.mamba_cache_per_req / (1 << 30)
        # draft 权重与 target 共享同卡显存, 须在 profile 时预扣
        if dflash_draft_cell_size > 0:
            draft_weight_gb = AttnInitializer._estimate_dflash_draft_weight_gb(model_runner)
            if draft_weight_gb > 0:
                rest_memory -= draft_weight_gb
                logger.info(f"DFLASH 预扣 draft 权重 {draft_weight_gb:.2f} GB, rest_memory={rest_memory:.2f} GB")

        use_offload = bool(
            __is_npu__
            and model_runner.server_args.offload_cache
            and not model_runner.is_draft_worker
            and model_runner.server_args.disaggregation_mode != "prefill"
        )
        if use_offload:
            from sglang.srt.model_executor.cuda_graph_runner import get_npu_compile_batch_sizes

            args = model_runner.server_args
            text_config = model_runner.model_config.hf_text_config
            packed_host = packed_host_enabled(args)
            if packed_host and (
                getattr(text_config, "cli_factor", 1) != 2
                or args.kvp_size != 8 or args.page_size != 128
                or args.npu_dsa_packed_kv_row_bytes != 672
            ):
                raise ValueError("Packed INT8 host offload requires CLI2/KVP8/page128/row672")
            if (
                not is_dsa(text_config)
                or not ENV.npu_enable_mla_split_kv_kr
                or (args.npu_dsa_kv_int8 and not packed_host)
                or model_runner.kv_cache_quant_method not in (None, "none")
                or model_runner.kv_cache_dtype != torch.bfloat16
            ):
                raise ValueError("Offload budget supports split BF16 DSA KV or packed INT8 host KV")
            if args.pp_size != 1:
                raise ValueError(
                    f"Offload budget requires PP1 Decode (pp_size={args.pp_size})"
                )
            if (
                args.offload_cache_bs <= 0
                or args.offload_cache_workspace_mb < 0
                or args.offload_cache_host_gb < 0
            ):
                raise ValueError("Invalid offload batch or memory budget")
            capture_bs = get_npu_compile_batch_sizes(args)
            if max(capture_bs) > args.offload_cache_bs:
                raise ValueError("NPU compile batch exceeds --offload-cache-bs")
            queries = (
                args.speculative_num_draft_tokens if args.speculative_algorithm else 1
            )
            topk = get_nsa_index_topk(text_config)
            if packed_host and (topk != 2048 or not 1 <= queries <= 4 or args.offload_cache_bs * queries > 64):
                raise ValueError("Packed INT8 hot offload requires TopK2048, 1 <= queries <= 4 and batch x queries <= 64")
            selected_records = (
                args.offload_cache_bs
                * queries
                * ((topk + args.page_size - 1) // args.page_size * args.page_size)
            )
            if packed_host:
                minimum_workspace = packed_offload_workspace_bytes(
                    args.offload_cache_bs, queries, topk, args.page_size, kv_cell_size
                )
            else:
                minimum_workspace = (
                    2 * selected_records * kv_cell_size
                    + args.offload_cache_bs * 4096 * (kv_cell_size + 8)
                    + selected_records * 4 * 16
                )
            if (args.offload_cache_workspace_mb << 20) < minimum_workspace:
                raise ValueError(
                    "Offload workspace reserve is too small: "
                    f"need at least {minimum_workspace} bytes"
                )
            draft_layers = 0
            if args.speculative_algorithm is not None:
                if not args.is_multi_head_eagle or getattr(
                    text_config, "nextn_use_scmoe", False
                ):
                    raise ValueError(
                        "Offload budget validates single-Attention multi-head NEXTN drafts"
                    )
                draft_layers = args.speculative_num_steps
            # Match init_memory_pool's explicit request limit. If it will be
            # inferred later, reserve the existing conservative upper bound.
            max_reqs = max_num_reqs if max_num_reqs is not None else 4096
            # Match the Decode-only spare slots that init_memory_pool adds
            # after determining the running limit and draft token expansion.
            storage_reqs = max_reqs + (
                args.disaggregation_decode_extra_slots
                if args.disaggregation_mode == "decode" else 0
            )
            reserved_pages = storage_reqs + 2 if args.enable_mla_l1_5_cache else 0
            budget = OffloadBudget(
                layer_count=model_runner.model_config.num_attention_layers,
                indexer_layer_mask=AttnInitializer.indexer_layer_mask(
                    model_runner, 0, model_runner.model_config.num_attention_layers
                ),
                kv_bytes_per_token_layer=kv_cell_size,
                index_bytes_per_token_layer=index_cell_size,
                kv_layout="packed_int8" if packed_host else "split_bf16",
                batch_size=args.offload_cache_bs,
                draft_layers=draft_layers,
                page_size=args.page_size,
                workspace_bytes=args.offload_cache_workspace_mb << 20,
                coordinator_fixed_bytes=(
                    packed_coordinator_fixed_bytes(
                        args.offload_cache_bs, queries, topk, args.page_size,
                        last_query=args.offload_cache_update_policy == "last_query",
                    ) if packed_host else 0
                ),
                host_backed=args.offload_cache_storage == "host",
                reserved_pages=reserved_pages,
                other_fixed_hbm_bytes=(storage_reqs + 1)
                * (model_runner.model_config.context_len + args.page_size * 4)
                * 4,
                draft_extra_tokens=(
                    max_reqs * args.speculative_num_steps + 100
                    if draft_layers
                    else 0
                ),
                compile_batch_sizes=tuple(capture_bs),
                graph_extra_bytes=(
                    args.offload_cache_graph_extra_mb << 20
                    if args.offload_cache_graph_extra_mb is not None else None
                ),
                graph_capture_order=tuple(
                    int(x) for x in args.offload_cache_graph_capture_order.split(',')
                ) if args.offload_cache_graph_capture_order else (),
            )
            local_ranks = getattr(args, "nprocs_per_node", None) or torch.npu.device_count()
            host_budget = int(available_host_bytes() * 0.8 / local_ranks)
            if args.offload_cache_host_gb:
                host_budget = min(
                    host_budget, int(args.offload_cache_host_gb * (1 << 30))
                )
            hbm_budget = int(rest_memory * (1 << 30))

            def _ledger(solved_tokens):
                return json.dumps(
                    {
                        **budget.describe(),
                        "hbm_budget_bytes": hbm_budget,
                        "host_budget_bytes": host_budget,
                        "profiled_local_tokens": solved_tokens,
                        "reserved_local_tokens": reserved_pages * args.page_size,
                        "local_ranks": local_ranks,
                    },
                    sort_keys=True,
                )

            # A graph captured below offload_cache_bs allocates its own GE
            # arena instead of sharing the offload_cache_bs graph's, and that
            # cost cannot be derived from the offload parameters. Refuse rather
            # than hand the pool headroom graph capture will take back, and
            # print the ledger first: the numbers are the point of refusing.
            if budget.padded_graph_count and budget.graph_extra_bytes is None:
                logger.error("Offload KV budget: %s", _ledger(None))
                raise ValueError(
                    "Offload budget cannot size the pool: "
                    f"{budget.padded_graph_count} graph(s) are captured below "
                    f"--offload-cache-bs {args.offload_cache_bs} "
                    f"({list(budget.padded_graph_batch_sizes)} out of "
                    f"--npu-compile-bs {capture_bs}). Either capture graphs only "
                    "at --offload-cache-bs, where the hand-off is a no-op and "
                    "this term is zero, or pass the measured "
                    "--offload-cache-graph-extra-mb with its capture order."
                )
            capacity = budget.capacity(hbm_budget, host_budget)
            if capacity <= 0:
                logger.error("Offload KV budget: %s", _ledger(capacity))
                raise ValueError(
                    "Offload budget leaves no KV pool: "
                    + json.dumps(
                        {
                            "hbm_budget_bytes": hbm_budget,
                            "host_budget_bytes": host_budget,
                            "fixed_hbm_bytes": budget.fixed_hbm_bytes,
                            "padded_graph_bytes": budget.padded_graph_bytes,
                            "padded_graph_batch_sizes": list(
                                budget.padded_graph_batch_sizes
                            ),
                            "hbm_bytes_per_token": budget.hbm_bytes_per_token,
                            "remedy": (
                                "capture graphs only at --offload-cache-bs, or "
                                "lower --offload-cache-bs, or raise "
                                "--mem-fraction-static"
                            ),
                        },
                        sort_keys=True,
                    )
                )
            if model_runner.tp_size > 1:
                common_capacity = torch.tensor(capacity, dtype=torch.int64)
                torch.distributed.all_reduce(
                    common_capacity,
                    op=torch.distributed.ReduceOp.MIN,
                    group=get_world_group().cpu_group,
                )
                capacity = int(common_capacity.item())
            model_runner.offload_budget = budget
            model_runner.offload_hbm_budget_bytes = hbm_budget
            model_runner.offload_host_budget_bytes = host_budget
            logger.info("Offload KV budget: %s", _ledger(capacity))
            return capacity

        max_num_token = int((rest_memory * (1 << 30) - placeholder_bytes) // cell_size)
        # PP 各 stage 的 pool 尺寸必须相同 (PrefillAdder 读本地 available_size 决定准入)。
        if model_runner.server_args.pp_size > 1 and not model_runner.is_draft_worker:
            token_tensor = torch.tensor([max_num_token], dtype=torch.int64)
            torch.distributed.all_reduce(
                token_tensor,
                op=torch.distributed.ReduceOp.MIN,
                group=get_world_group().cpu_group,
            )
            max_num_token = int(token_tensor.item())
        return max_num_token
