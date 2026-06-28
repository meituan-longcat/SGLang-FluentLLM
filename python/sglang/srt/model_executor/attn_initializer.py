import os
from typing import Optional

import torch

from sglang.srt.distributed import get_pp_group
from sglang.srt.utils import get_colorful_logger, is_npu, is_sm90_supported

from sglang.srt.env import global_server_args_dict
from sglang.srt.configs.model_config import AttentionArch, is_dsa, get_nsa_index_head_dim
from sglang.srt.mem_cache.memory_pool import (
    NativeSparseMHATokenToKVPool,
    NativeSparseMLATokenToKVPool,
    MHATokenToKVPool,
    NPUMHATokenToKVPool,
    MLATokenToKVPool,
    MLAL1HalfTokenToKVPool,
    ReqToTokenPool,
    SWAKVPool,
    HybridLinearKVPool,
    HybridReqToTokenPool,
    DSATokenToKVPool,
    L1HalfDSATokenToKVPool,
)
from sglang.srt.mem_cache.allocator import KVAllocator
from sglang.srt.distributed.parallel_state import get_world_group
from sglang.srt.layers.attention.torch_native_backend import TorchNativeAttnBackend

__is_npu__ = is_npu()
if not __is_npu__:
    from sglang.srt.layers.attention.flashinfer_backend import FlashInferAttnBackend
    if is_sm90_supported():
        from sglang.srt.layers.attention.flashmla_backend import FlashMLABackend
        from sglang.srt.layers.attention.flash_attention_backend import FlashAttentionBackend
        from sglang.srt.layers.attention.duo_attn_backend import DuoAttnBackend
        from sglang.srt.layers.attention.dsa_backend import DpskSparseAttnBackend
    from sglang.srt.layers.attention.flashinfer_mla_backend import FlashInferMLAAttnBackend, FlashInferMLATBOBackend
    from sglang.srt.layers.attention.triton_backend import TritonAttnBackend
    from sglang.srt.layers.attention.torch_native_mla_backend import TorchNativeMlaAttnBackend
else:
    from sglang.srt.layers.attention.npu_mla_backend import NpuMLAAttnBackend, NpuAttnBackend

from sglang.srt.layers.dp_attention import get_attention_tp_size, get_draft_attention_tp_size

from sglang.srt.utils import get_available_gpu_memory

logger = get_colorful_logger(__name__)


SGLANG_CI_SMALL_KV_SIZE = os.getenv("SGLANG_CI_SMALL_KV_SIZE", None)


class AttnInitializer(object):
    @staticmethod
    def modify_args(model_runner):
        # Model-specific adjustment
        if (
            model_runner.model_config.attention_arch == AttentionArch.MLA
        ):
            if not __is_npu__:
                if model_runner.is_kimi_linear:
                    model_runner.server_args.attention_backend = "hybrid_linear_attn"
                    logger.info("MLA optimization is turned on. Use hybrid_linear_attn backend.")
                elif is_dsa(model_runner.model_config.hf_config):
                    model_runner.server_args.attention_backend = "dsa"
                elif model_runner.server_args.enable_flashinfer_mla:
                    if model_runner.server_args.attention_backend == "duo_attn":
                        logger.info("MLA duo-attention is turned on. Use duo attn backend.")
                    elif not model_runner.server_args.attention_backend == "flashmla":
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
                    model_runner.server_args.max_mamba_cache_size = (
                        model_runner.server_args.max_running_requests
                    )
                else:
                    model_runner.server_args.max_mamba_cache_size = 512
            if model_runner.is_hybrid_gdn:
                model_runner.server_args.max_mamba_cache_size = (
                    model_runner.server_args.max_mamba_cache_size
                    // (
                        model_runner.server_args.dp_size
                        if model_runner.server_args.enable_dp_attention
                        else 1
                    )
                )

    @staticmethod
    def init_memory_pool(
        model_runner,
        total_gpu_memory: int,
        max_num_reqs: Optional[int] = None,
        max_total_tokens: Optional[int] = None,
        page_size: int = 64,
    ):
        if model_runner.server_args.kv_cache_dtype == "auto":
            model_runner.kv_cache_dtype = model_runner.dtype
        elif model_runner.server_args.kv_cache_dtype == "fp8_e4m3":
            model_runner.kv_cache_dtype = torch.float8_e4m3fn
        else:
            raise ValueError(
                f"Unsupported kv_cache_dtype: {model_runner.server_args.kv_cache_dtype}."
            )
        model_runner.kv_cache_quant_method = model_runner.server_args.kv_cache_quant_method
        if model_runner.server_args.index_k_dtype == "fp8_e4m3":
            model_runner.index_k_dtype = torch.float8_e4m3fn
        elif model_runner.server_args.index_k_dtype == "bf16":
            model_runner.index_k_dtype = torch.bfloat16
        else:
            raise ValueError(f"{model_runner.server_args.index_k_dtype=}")

        model_runner.max_total_num_tokens = AttnInitializer.profile_max_num_token(model_runner, total_gpu_memory)
        if SGLANG_CI_SMALL_KV_SIZE:
            model_runner.max_total_num_tokens = int(SGLANG_CI_SMALL_KV_SIZE)

        # this is the max token can be stored locally, will be different when enable_mla_l1_5_cache
        if model_runner.server_args.enable_mla_l1_5_cache:
            assert model_runner.model_config.attention_arch == AttentionArch.MLA, f"should be mla model"
            assert getattr(model_runner.model_config.hf_config, "use_nsa_mla", False) == False
            model_runner.max_total_num_tokens = model_runner.max_total_num_tokens * model_runner.server_args.attn_tp_size

        if max_num_reqs is None:
            max_num_reqs = min(
                max(
                    int(
                        model_runner.max_total_num_tokens / model_runner.model_config.context_len * 512
                    ),
                    2048,
                ),
                4096,
            )

        if not model_runner.spec_algorithm.is_none():
            if model_runner.is_draft_worker:
                model_runner.max_total_num_tokens = model_runner.server_args.draft_runner_cache_size
            else:
                model_runner.server_args.draft_runner_cache_size = (
                    model_runner.max_total_num_tokens
                    + max_num_reqs * model_runner.server_args.speculative_num_steps
                    + 100
                )

        if max_total_tokens is not None:
            if max_total_tokens > model_runner.max_total_num_tokens:
                logger.warning(
                    f"max_total_tokens={max_total_tokens} is larger than the profiled value "
                    f"{model_runner.max_total_num_tokens}. "
                    f"Use the profiled value instead."
                )
            model_runner.max_total_num_tokens = min(model_runner.max_total_num_tokens, max_total_tokens)

        if model_runner.max_total_num_tokens <= 0:
            raise RuntimeError(
                "Not enough memory. Please try to increase --mem-fraction-static."
            )

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
        # Added page_size is reserved pad page
        assert model_runner.max_total_num_tokens >= model_runner.model_config.context_len + model_runner.page_size, \
            f"KV Cache allocation too small: {model_runner.max_total_num_tokens=}, {model_runner.model_config.context_len=}, {model_runner.page_size=}, may cause requests to enter Scheduler but cannot be scheduled, causing service to hang"
        if global_server_args_dict['npu_enable_graph_cache']:
            max_total_num_tokens_file_path = os.path.join(model_runner.server_args.model_path,
                                                          "num_token_file_path",
                                                          str(model_runner.global_rank))
            os.makedirs(max_total_num_tokens_file_path, exist_ok=True)
            max_total_num_tokens_file = os.path.join(max_total_num_tokens_file_path, "max_total_num_tokens.txt")
            if os.path.exists(max_total_num_tokens_file):
                with open(max_total_num_tokens_file, "r") as f:
                    model_runner.max_total_num_tokens = int(f.read().strip())
            else:
                with open(max_total_num_tokens_file, "w") as f:
                    f.write(str(model_runner.max_total_num_tokens))

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
                    size=max_num_reqs,
                    max_context_len=model_runner.model_config.context_len
                        + extra_max_context_len,
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
            model_runner.model_config.num_attention_layers = getattr(model_runner.model_config.hf_config, "num_nextn_predict_layers", 1)

        if model_runner.server_args.pp_size > 1:
            model_runner.attn_start_layer = model_runner.model.attn_start_layer
            model_runner.attn_end_layer = model_runner.model.attn_end_layer
        else:
            model_runner.attn_start_layer = 0
            model_runner.attn_end_layer = model_runner.model_config.num_attention_layers

        if (
            model_runner.model_config.attention_arch == AttentionArch.MLA
            and not model_runner.mambaish_config
        ):
            # for flash nsa+
            # todo chlpp
            if getattr(model_runner.model_config.hf_config, "use_nsa_mla", False):
                model_runner.token_to_kv_pool = NativeSparseMLATokenToKVPool(
                    size=model_runner.max_total_num_tokens,
                    dtype=model_runner.kv_cache_dtype,
                    kv_lora_rank=model_runner.model_config.hf_config.kv_lora_rank,
                    qk_rope_head_dim=model_runner.model_config.hf_config.qk_rope_head_dim,
                    layer_num=model_runner.model_config.num_attention_layers,
                    device=model_runner.device,
                    enable_memory_saver=model_runner.server_args.enable_memory_saver,
                    max_batch_size=max_num_reqs + 1,
                    max_context_len=model_runner.model_config.context_len + model_runner.page_size * 4,
                    page_size=model_runner.page_size,
                    rank=model_runner.global_rank,
                    compressed_block_stride=model_runner.model_config.hf_config.stride
                )
            elif is_dsa(model_runner.model_config.hf_config):
                model_runner.num_effective_layers=model_runner.attn_end_layer-model_runner.attn_start_layer
                token_to_kv_pool_class = L1HalfDSATokenToKVPool if model_runner.server_args.enable_mla_l1_5_cache else DSATokenToKVPool
                model_runner.token_to_kv_pool = token_to_kv_pool_class(
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
                    index_head_dim=get_nsa_index_head_dim(model_runner.model_config.hf_config),
                    index_dtype=model_runner.index_k_dtype,
                    start_layer=model_runner.attn_start_layer,
                    end_layer=model_runner.attn_end_layer,
                )
            else:
                model_runner.num_effective_layers=model_runner.attn_end_layer-model_runner.attn_start_layer
                token_to_kv_pool_class = MLAL1HalfTokenToKVPool if model_runner.server_args.enable_mla_l1_5_cache else MLATokenToKVPool
                model_runner.token_to_kv_pool = token_to_kv_pool_class(
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
        elif model_runner.mambaish_config:
            extra_args = {}
            if model_runner.model_config.attention_arch == AttentionArch.MLA:
                extra_args = {
                    "model_dtype": model_runner.dtype,
                    "quant_method": model_runner.kv_cache_quant_method,
                    "use_mla":True,
                    "kv_lora_rank": model_runner.model_config.kv_lora_rank,
                    "qk_rope_head_dim": model_runner.model_config.qk_rope_head_dim,
                }
            model_runner.token_to_kv_pool = HybridLinearKVPool(
                size=model_runner.max_total_num_tokens,
                dtype=model_runner.kv_cache_dtype,
                head_num=model_runner.model_config.get_num_kv_heads(
                    get_attention_tp_size()
                ),
                head_dim=model_runner.model_config.head_dim,
                # if draft worker, we only need 1 attention layer's kv pool
                full_attention_layer_ids=(
                    [0]
                    if model_runner.is_draft_worker
                    else model_runner.model_config.hf_config.full_attention_layer_ids
                ),
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
                    rank=model_runner.global_rank
                )
            elif getattr(model_runner.model_config.hf_config, "use_nsa_mla", False):
                model_runner.token_to_kv_pool = NativeSparseMLATokenToKVPool(
                    size=model_runner.max_total_num_tokens,
                    dtype=model_runner.kv_cache_dtype,
                    kv_lora_rank=model_runner.model_config.hf_config.kv_lora_rank,
                    qk_rope_head_dim=model_runner.model_config.hf_config.qk_rope_head_dim,
                    layer_num=model_runner.model_config.num_attention_layers,
                    device=model_runner.device,
                    enable_memory_saver=model_runner.server_args.enable_memory_saver,
                    max_batch_size=max_num_reqs + 1,
                    max_context_len=model_runner.model_config.context_len + model_runner.page_size * 4,
                    page_size=model_runner.page_size,
                    rank=model_runner.global_rank,
                    compressed_block_stride=model_runner.model_config.hf_config.stride
                )
            else:
                if model_runner.is_hybrid:
                    model_runner.token_to_kv_pool = SWAKVPool(
                        size=model_runner.full_max_total_num_tokens,
                        size_swa=model_runner.swa_max_total_num_tokens,
                        dtype=model_runner.kv_cache_dtype,
                        head_num=model_runner.model_config.get_num_kv_heads(
                            get_attention_tp_size()
                        ),
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
                        head_num=model_runner.model_config.get_num_kv_heads(get_draft_attention_tp_size()),
                        head_dim=model_runner.model_config.head_dim,
                        layer_num=model_runner.model_config.num_attention_layers,
                        device=model_runner.device,
                        enable_memory_saver=model_runner.server_args.enable_memory_saver,
                        max_batch_size=max_num_reqs * 2 + 1,
                        max_context_len=model_runner.model_config.context_len + model_runner.page_size * 4,
                        page_size=model_runner.page_size,
                        rank=model_runner.global_rank
                    )
                else:
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
                        rank=model_runner.global_rank
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
            from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
                HybridLinearAttnBackend,
                MambaAttnBackend,
                KimiLinearAttnBackend
            )
            linear_attn_backend_cls = MambaAttnBackend if model_runner.is_hybrid_gdn \
                else KimiLinearAttnBackend if model_runner.is_kimi_linear else None
            full_attn_backend_cls = FlashInferAttnBackend if model_runner.is_hybrid_gdn \
                else TritonAttnBackend if model_runner.is_kimi_linear else None
            linear_attn_backend = linear_attn_backend_cls(model_runner)
            full_attn_backend = full_attn_backend_cls(model_runner)
            full_attn_layers = model_runner.model_config.hf_config.full_attention_layer_ids
            model_runner.attn_backend = HybridLinearAttnBackend(
                full_attn_backend, linear_attn_backend, full_attn_layers
            )
        elif backend_name == "duo_attn":
            if model_runner.model_config.attention_arch != AttentionArch.MLA:
                if model_runner.is_draft_worker:
                    model_runner.attn_backend = FlashInferAttnBackend(model_runner)
                else:
                    model_runner.attn_backend = DuoAttnBackend(model_runner)
            else:
                model_runner.attn_backend = DuoAttnBackend(model_runner)
        else:
            raise ValueError(
                f"Invalid attention backend: {backend_name}"
            )

    @staticmethod
    def profile_max_num_token(model_runner, total_gpu_memory: int):
        available_gpu_memory = get_available_gpu_memory(
            model_runner.device,
            model_runner.gpu_id,
            distributed=model_runner.tp_size > 1,
            cpu_group=get_world_group().cpu_group,
        )

        if model_runner.is_hybrid_gdn:
            num_attention_layers = len(model_runner.model_config.hf_config.full_attention_layer_ids)
        else:
            num_attention_layers = model_runner.model_config.num_attention_layers

        if model_runner.server_args.pp_size > 1:
            if model_runner.server_args.pp_layer_nums is not None:
                num_attention_layers = max(model_runner.server_args.pp_layer_nums) * 2
            else:
                num_attention_layers = (num_attention_layers + model_runner.server_args.pp_size) // model_runner.server_args.pp_size * 2
            logger.info(f"will creating {num_attention_layers=} attention layers kv cache")

        # If using MTP or EAGLE, add one layer during profile (default to one for now)
        if model_runner.server_args.speculative_algorithm is not None:
            num_attention_layers += 1
        if (
            model_runner.model_config.attention_arch == AttentionArch.MLA
        ):
            if model_runner.kv_cache_quant_method == "per_token_head":
                cell_size = (
                    (model_runner.model_config.kv_lora_rank * torch._utils._element_size(model_runner.kv_cache_dtype)
                     + model_runner.model_config.qk_rope_head_dim * torch._utils._element_size(model_runner.dtype)
                     + 1 * torch._utils._element_size(torch.float32))
                     * num_attention_layers
                )
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
        rest_memory = available_gpu_memory - total_gpu_memory * (
            1 - model_runner.mem_fraction_static
        )
        logger.info(f"{rest_memory=} {available_gpu_memory=} {total_gpu_memory=} {model_runner.mem_fraction_static}")
        if model_runner.is_hybrid_gdn:
            rest_memory -= (
                model_runner.server_args.max_mamba_cache_size
                * model_runner.model_config.hf_config.mamba_cache_per_req
                / (1 << 30)
            )
        max_num_token = int(rest_memory * (1 << 30) // cell_size)
        return max_num_token
