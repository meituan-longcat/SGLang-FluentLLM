import copy
from typing import Any, NamedTuple, Optional, Union

import torch
import torch_npu  # noqa: F401 - registers device/operator implementations
import torchair as tng

from sglang.srt.npu.sfa1_runtime import get_sfa1_ops, initialize_sfa1_runtime

import flash_ops  # noqa: F401 - registers torch.ops.custom flash operators  # isort: skip

from sglang.srt.configs import FLASHConfig
from sglang.srt.configs.model_config import get_nsa_index_head_dim, get_nsa_index_n_heads, is_deepseek_nsa
from sglang.srt.distributed import (
    get_attn_o_proj_tp_group,
    get_attn_sp_token_slice_pos,
    get_attn_tp_group,
    get_attn_tp_world_size,
)
from sglang.srt.env import ENV, global_server_args_dict
from sglang.srt.layers.attention.dsa.nsa_indexer import Indexer
from sglang.srt.layers.attention.npu_attn.flash_attn import CacheConfig
from sglang.srt.layers.attention.npu_attn.npu_selected_kv_prefill import (
    SelectedKvPrefillState,
    begin_native_selected_plan,
    prepare_native_selected_metadata,
    prepare_selected_kv,
    selected_kv_attention,
)
from sglang.srt.layers.attention.npu_attn.npu_sfad_prefill import (
    SfadPrefillPrefixState,
    SfadPrefillState,
    get_sfad_prefill_stage_stream,
    prefetch_prefill_prefix_pages,
    prepare_mtp_prefill_pages,
    prepare_prefill_pages_local,
    prepare_prefill_prefix_pages_local,
    reserve_prefill_workspace,
    sfad_topk_metadata,
    submit_prefill_page_gather,
    submit_prefill_prefix_page_gather,
)
from sglang.srt.layers.attention.radix_attention import RadixAttention
from sglang.srt.layers.dense.layouts.unquant import UnquantizedLinearMethod
from sglang.srt.layers.linear import ColumnParallelLinear, LinearBase, ReplicatedLinear, RowParallelLinear
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.npu.utils import npu_limit_core, npu_stream_switch, npu_super_kernel
from sglang.srt.ops.layernorm import RMSNorm
from sglang.srt.utils import get_colorful_logger

logger = get_colorful_logger(__name__)
_NATIVE_SFAD_PREFILL_OP = getattr(torch.ops.custom, "npu_sparse_mla_attention_prefill", None)


class _PrefillLatentCacheGather(NamedTuple):
    """New-token KV gathered on the staging stream; Q remains independent."""

    value: torch.Tensor
    ready: Any

    def wait(self) -> torch.Tensor:
        stream = torch.npu.current_stream()
        stream.wait_event(self.ready)
        self.value.record_stream(stream)
        return self.value


def _gather_prefill_latent_cache(value, global_sp_num_tokens):
    compute = torch.npu.current_stream()
    stage = get_sfad_prefill_stage_stream()
    produced = torch.npu.Event()
    produced.record(compute)
    with torch.npu.stream(stage):
        stage.wait_event(produced)
        value.record_stream(stage)
        gathered = get_attn_tp_group().gather_tensor(value, global_sp_num_tokens)
        ready = torch.npu.Event()
        ready.record(stage)
    return _PrefillLatentCacheGather(gathered, ready)


def yarn_get_mscale(scale: float = 1, mscale: float = 1) -> float:
    import math

    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


class DeepseekV2MLAAttention(torch.nn.Module):
    """
    Main reference: DeepseekV2 paper, and FlashInfer Implementation
    (https://arxiv.org/abs/2405.04434 and https://github.com/flashinfer-ai/flashinfer/pull/551).

    For more info see MLACommonImpl in: vllm/attention/backends/mla/utils.py
    """

    def __init__(
        self,
        config: Union[FLASHConfig, Any],
        hidden_size: int,
        num_heads: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: Optional[int],
        kv_lora_rank: int,
        bias: bool = False,
        rope_theta: float = 10000,
        rms_norm_eps: float = 1.0e-6,
        rope_scaling: Optional[dict[str, Any]] = None,
        max_position_embeddings: int = 8192,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        layer_id=None,
        attn_layer_id=None,
        reduce_results: bool = True,
        has_mla_scale=True,
    ) -> None:
        super().__init__()
        global v_head_dim_global
        global hidden_size_global
        self.layer_id = layer_id
        self.attn_layer_id = attn_layer_id
        self.hidden_size = hidden_size
        hidden_size_global = hidden_size
        self.num_heads = num_heads
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        v_head_dim_global = v_head_dim
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.bias = bias
        self.rope_theta = rope_theta
        self.rms_norm_eps = rms_norm_eps
        self.rope_scaling = rope_scaling  # None
        self.max_position_embeddings = max_position_embeddings
        self.cache_config = cache_config
        self.quant_config = quant_config
        self.reduce_results = reduce_results
        self.config = config
        self.enable_nsa = is_deepseek_nsa(config)
        self.npu_disable_dsa_head_parallel = (
            self.enable_nsa
            and global_server_args_dict["npu_disable_dsa_head_parallel"]
            and global_server_args_dict["disaggregation_mode"] == "prefill"
        )
        self.npu_dsa_kv_int8 = bool(global_server_args_dict.get("npu_dsa_kv_int8", False))
        self.use_sfa1 = initialize_sfa1_runtime(
            eligible=(
                self.enable_nsa and ENV.npu_enable_mla_split_kv_kr
                and not self.npu_dsa_kv_int8
                and not ENV.npu_enable_sfad_rank_major_query
            ),
            offload_enabled=bool(global_server_args_dict.get("offload_cache", False)),
            # ModelRunner.__init__ already made the authoritative call, before
            # this model was built; a layer only needs the route. The draft and
            # target models share this process, so there is no global that
            # could tell a layer which of the two it belongs to.
            owns_offload_pool=False,
        )
        self.npu_dsa_packed_kv_row_bytes = int(global_server_args_dict.get("npu_dsa_packed_kv_row_bytes", 656))
        if self._npu_dsa_kv_int8_enabled():
            logger.info("NPU DSA KV int8 packed C8 path is enabled.")

        self.attn_tp_size = get_attn_tp_world_size()
        assert num_heads % self.attn_tp_size == 0
        self.num_local_heads = num_heads // self.attn_tp_size
        self.scaling = self.qk_head_dim**-0.5
        if rope_scaling is not None:
            self.rope_scaling["rope_type"] = "deepseek_yarn"
            mscale_all_dim = rope_scaling.get("mscale_all_dim", False)
            scaling_factor = rope_scaling["factor"]
            mscale = yarn_get_mscale(scaling_factor, float(mscale_all_dim))
            self.scaling = self.scaling * mscale * mscale

        attn_tp_group = None if self.attn_tp_size == 1 else get_attn_tp_group()
        assert self.q_lora_rank is not None  # assert MLA
        # no merged qkv_a when using npu, need add for gpu
        self.q_a_proj = ReplicatedLinear(
            self.hidden_size, self.q_lora_rank, bias=False, quant_config=self.quant_config, enable_weight_transpsoe=True, enable_weight_nz=True
        )
        self.q_a_layernorm = RMSNorm(self.q_lora_rank, eps=self.rms_norm_eps)
        self.q_b_proj = ColumnParallelLinear(
            q_lora_rank,
            self.num_heads * self.qk_head_dim,
            bias=bias,
            quant_config=quant_config,
            enable_weight_transpsoe=True,
            enable_weight_nz=True,
            outside_tp_group=None if self.npu_disable_dsa_head_parallel else attn_tp_group,
            disable_parallel=True,
        )

        self.kv_a_proj_with_mqa = ReplicatedLinear(
            self.hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=False,
            quant_config=self.quant_config,
            enable_weight_transpsoe=True,
            enable_weight_nz=True,
        )
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank, eps=self.rms_norm_eps)
        self.kv_b_proj = ColumnParallelLinear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=bias,
            quant_config=quant_config,
            outside_tp_group=None if self.npu_disable_dsa_head_parallel else attn_tp_group,
            disable_parallel=True,
        )

        # different o_proj tp mode
        # todo
        self.o_proj_tp_group = None
        if global_server_args_dict["npu_o_proj_tp_size"] > 0:
            self.o_proj_tp_group = get_attn_o_proj_tp_group()
        else:
            self.o_proj_tp_group = attn_tp_group

        split_n = False  # Set True only for attn tp size == 1
        if split_n and self.attn_tp_size == 1:
            assert global_server_args_dict["npu_o_proj_tp_size"] > 0
            # when tp_size > 1, need add all2all before o_proj, not support now
            self.o_proj = ColumnParallelLinear(
                self.num_heads * self.v_head_dim,
                self.hidden_size,
                bias=bias,
                quant_config=quant_config,
                outside_tp_group=None if self.npu_disable_dsa_head_parallel else self.o_proj_tp_group,
                disable_parallel=True,
                enable_weight_nz=True,
                enable_weight_transpsoe=True,
                gather_output=False,  # use all2all
            )
        else:
            self.o_proj = RowParallelLinear(
                self.num_heads * self.v_head_dim,
                self.hidden_size,
                bias=bias,
                quant_config=quant_config,
                reduce_results=reduce_results,  # need skip in forward process
                outside_tp_group=None if self.npu_disable_dsa_head_parallel else self.o_proj_tp_group,
                disable_parallel=True,
                enable_weight_nz=True,
                enable_weight_transpsoe=True,
            )

        # for longcat extra prolog scale parameter
        if has_mla_scale:
            self.mla_scale_q_lora = (self.hidden_size / self.q_lora_rank) ** 0.5
            self.mla_scale_kv_lora = (self.hidden_size / self.kv_lora_rank) ** 0.5
            self.mla_scale_v = None
        else:
            self.mla_scale_q_lora = 1.0
            self.mla_scale_kv_lora = 1.0
            self.mla_scale_v = None

        self.enable_nsa = is_deepseek_nsa(config)
        self.attn_mqa = RadixAttention(
            self.num_local_heads,
            self.kv_lora_rank + self.qk_rope_head_dim,
            self.scaling,
            num_kv_heads=1,
            layer_id=layer_id,
            v_head_dim=self.kv_lora_rank,
            is_npu_dsa=bool(self.enable_nsa),
            attn_layer_id=attn_layer_id,
        )

        self.attn_mha = RadixAttention(
            self.num_local_heads,
            self.qk_nope_head_dim + self.qk_rope_head_dim,
            self.scaling,
            num_kv_heads=self.num_local_heads,
            layer_id=layer_id,
            v_head_dim=self.v_head_dim,
            is_npu_dsa=bool(self.enable_nsa),
            attn_layer_id=attn_layer_id,
        )

        self.w_kc = None
        self.w_vc = None

        if self.enable_nsa:
            self.index_topk = getattr(config, "index_topk", 128)
            self.cli_factor = getattr(config, "cli_factor", 1)
            self.indexer = Indexer(
                hidden_size=self.hidden_size,
                index_n_heads=get_nsa_index_n_heads(config),
                index_head_dim=get_nsa_index_head_dim(config),
                rope_head_dim=self.qk_rope_head_dim,
                index_topk=self.index_topk,
                index_k_norm_type="rms",
                q_lora_rank=self.q_lora_rank,
                max_position_embeddings=self.max_position_embeddings,
                rope_theta=self.rope_theta,
                layer_id=layer_id,
                rope_scaling=self.rope_scaling,
                config=config,
                quant_config=quant_config,
                scale_fmt=None,
                npu_disable_dsa_head_parallel=self.npu_disable_dsa_head_parallel,
                skip_rope=bool(getattr(config, "dsa_idxer_nope", False)),
            )

    def _npu_dsa_kv_int8_enabled(self) -> bool:
        return self.enable_nsa and self.npu_dsa_kv_int8

    def _validate_packed_c8_cache(self, packed_kv_cache: torch.Tensor, expected_dim: int) -> None:
        if packed_kv_cache.dtype != torch.int8:
            raise RuntimeError(f"NPU DSA KV int8 expects int8 packed cache, got {packed_kv_cache.dtype}.")
        if packed_kv_cache.shape[-1] != expected_dim:
            raise RuntimeError(f"NPU DSA KV int8 expects packed dim {expected_dim}, got {tuple(packed_kv_cache.shape)}.")

    def _run_packed_c8_mla_prolog(
        self,
        rotary_emb,
        hidden_states: torch.Tensor,
        packed_kv_cache: torch.Tensor,
        forward_batch: ForwardBatch,
        contiguous_cache: bool = False,
    ):
        kv_pool = forward_batch.token_to_kv_pool
        packed_kv_cache_dim = kv_pool.kv_cache_dim
        tile_size = kv_pool.quant_block_size
        self._validate_packed_c8_cache(packed_kv_cache, packed_kv_cache_dim)

        token_x = hidden_states.reshape(-1, self.hidden_size).contiguous()
        cos, sin = rotary_emb
        rope_cos = cos.reshape(-1, self.qk_rope_head_dim).contiguous()
        rope_sin = sin.reshape(-1, self.qk_rope_head_dim).contiguous()
        if contiguous_cache:
            cache_index_arg = None
        else:
            cache_index = forward_batch.attn_metadata.slot_mapping
            cache_index_arg = cache_index.reshape(-1).to(torch.int64).contiguous()
            if cache_index_arg.numel() != token_x.shape[0]:
                raise RuntimeError(f"packed C8 MLA prolog cache_index/token count mismatch: {cache_index_arg.numel()} vs {token_x.shape[0]}.")

        empty_kr_cache = torch.empty((1, 2, 1, 0), dtype=torch.bfloat16, device=packed_kv_cache.device)
        k_nope_clip_alpha = torch.ones((1,), dtype=torch.float32, device=packed_kv_cache.device)
        cache_mode = "PA_BSND" if packed_kv_cache.dim() == 4 else "TND"
        mla_output = torch.ops.custom.npu_mla_prolog_v3(
            token_x=token_x,
            weight_dq=self.q_a_proj.weight,
            weight_uq_qr=self.q_b_proj.weight,
            weight_uk=self.kv_b_proj.W_UK_T,
            weight_dkv_kr=self.kv_a_proj_with_mqa.weight,
            rmsnorm_gamma_cq=self.q_a_layernorm.weight,
            rmsnorm_gamma_ckv=self.kv_a_layernorm.weight,
            rmsnorm_epsilon_cq=self.rms_norm_eps,
            rmsnorm_epsilon_ckv=self.rms_norm_eps,
            rope_sin=rope_sin,
            rope_cos=rope_cos,
            cache_index=cache_index_arg,
            kv_cache=packed_kv_cache,
            kr_cache=empty_kr_cache,
            dequant_scale_x=None,
            dequant_scale_w_dq=None,
            dequant_scale_w_uq_qr=None,
            dequant_scale_w_dkv_kr=None,
            smooth_scales_cq=None,
            quant_scale_ckv=None,
            cache_mode=cache_mode,
            query_norm_flag=True,
            weight_quant_mode=0,
            kv_cache_quant_mode=3,
            query_quant_mode=0,
            ckvkr_repo_mode=1,
            quant_scale_repo_mode=1,
            tile_size=tile_size,
            k_nope_clip_alpha=k_nope_clip_alpha,
            qc_qr_scale=self.mla_scale_q_lora,
            kc_scale=self.mla_scale_kv_lora,
        )
        q_nope, q_rope = mla_output[:2]
        q_lora = mla_output[3]
        return q_nope, q_rope, q_lora

    def _ensure_absorbed_packed_c8_query(self, q_nope: torch.Tensor) -> torch.Tensor:
        """Accept both V3 query-output ABIs and return kv_lora_rank width."""
        if q_nope.shape[-1] == self.kv_lora_rank:
            return q_nope
        if q_nope.shape[-1] != self.qk_nope_head_dim:
            raise RuntimeError(
                f"packed C8 MLA Prolog returned unsupported q_nope dim: {q_nope.shape[-1]}; expected {self.qk_nope_head_dim} or {self.kv_lora_rank}."
            )
        return torch.bmm(
            q_nope.transpose(0, 1),
            self.kv_b_proj.W_UK_T,
        ).transpose(0, 1)

    def _can_use_local_packed_c8_prefill(
        self,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> bool:
        kv_pool = forward_batch.token_to_kv_pool
        quant_block_size = getattr(kv_pool, "quant_block_size", 0)
        if quant_block_size != 128:
            return False
        packed_payload_dim = (
            self.kv_lora_rank
            + self.qk_rope_head_dim * torch.bfloat16.itemsize
            + ((self.kv_lora_rank + quant_block_size - 1) // quant_block_size) * torch.float32.itemsize
        )
        expected_packed_dim = self.npu_dsa_packed_kv_row_bytes
        if expected_packed_dim not in (
            packed_payload_dim,
            (packed_payload_dim + 31) // 32 * 32,
        ):
            return False
        if not (
            self._npu_dsa_kv_int8_enabled()
            and getattr(kv_pool, "dsa_kv_int8_packed", False)
            and getattr(kv_pool, "kv_cache_dim", 0) == expected_packed_dim
            and self.npu_disable_dsa_head_parallel
            and global_server_args_dict.get("enable_mla_l1_5_cache", False)
            and global_server_args_dict.get("npu_disable_kv_nz", False)
            and forward_batch.global_sp_num_tokens is not None
            and forward_batch.attn_metadata.dense_slot_mapping_idx is not None
            and forward_batch.attn_metadata.dense_slot_mapping is not None
        ):
            return False

        slice_start, slice_end = get_attn_sp_token_slice_pos(forward_batch)
        local_token_count = hidden_states.reshape(-1, self.hidden_size).shape[0]
        return local_token_count == slice_end - slice_start

    @staticmethod
    def _is_cold_prefill(forward_batch: ForwardBatch) -> bool:
        """Return true when the attention KV contains only this extend batch."""
        # KVP shards slot_mapping/kv_index_list by cache owner, so their
        # lengths are not a valid cold/hit discriminator.  The scheduler's
        # CPU prefix lengths are request-global and avoid a device sync here.
        prefix_lens = forward_batch.extend_prefix_lens_cpu
        return prefix_lens is not None and not any(prefix_lens)

    def _run_local_packed_c8_prefill(
        self,
        rotary_emb,
        hidden_states: torch.Tensor,
        packed_kv_pool: torch.Tensor,
        forward_batch: ForwardBatch,
    ):
        """Run packed-C8 prolog once per SP token and publish KV."""
        logger.info_once("NPU DSA KV int8 local TND Prefill path is active.")
        attn_metadata = forward_batch.attn_metadata
        packed_kv_cache_dim = forward_batch.token_to_kv_pool.kv_cache_dim
        self._validate_packed_c8_cache(packed_kv_pool, packed_kv_cache_dim)

        token_x = hidden_states.reshape(-1, self.hidden_size)
        local_token_count = token_x.shape[0]
        slice_start, slice_end = get_attn_sp_token_slice_pos(forward_batch)
        if local_token_count != slice_end - slice_start:
            raise RuntimeError(f"packed C8 local prefill hidden/SP slice mismatch: {local_token_count} vs {slice_end - slice_start}.")

        global_token_count = attn_metadata.slot_mapping.reshape(-1).numel()
        cos, sin = rotary_emb
        rope_cos = cos.reshape(-1, self.qk_rope_head_dim)
        rope_sin = sin.reshape(-1, self.qk_rope_head_dim)
        if rope_sin.shape[0] != rope_cos.shape[0]:
            raise RuntimeError(f"packed C8 local prefill rotary sin/cos count mismatch: {rope_sin.shape[0]} vs {rope_cos.shape[0]}.")
        if rope_cos.shape[0] == global_token_count:
            local_rotary_emb = (
                rope_cos[slice_start:slice_end],
                rope_sin[slice_start:slice_end],
            )
        elif rope_cos.shape[0] == local_token_count:
            local_rotary_emb = (rope_cos, rope_sin)
        else:
            raise RuntimeError(
                "packed C8 local prefill rotary/token count mismatch: "
                f"rope={rope_cos.shape[0]}, local={local_token_count}, "
                f"global={global_token_count}."
            )

        local_packed_kv = torch.empty(
            (local_token_count, 1, packed_kv_cache_dim),
            dtype=torch.int8,
            device=packed_kv_pool.device,
        )
        if packed_kv_cache_dim > 656:
            local_packed_kv[..., 656:].zero_()
        if local_token_count > 0:
            q_nope, q_pe, q_lora = self._run_packed_c8_mla_prolog(
                local_rotary_emb,
                hidden_states,
                local_packed_kv,
                forward_batch,
                contiguous_cache=True,
            )
        else:
            q_nope = hidden_states.new_empty((0, self.num_heads, self.qk_nope_head_dim))
            q_pe = hidden_states.new_empty((0, self.num_heads, self.qk_rope_head_dim))
            q_lora = hidden_states.new_empty((0, self.q_lora_rank))

        global_packed_kv = self._publish_local_packed_c8_prefill(
            local_packed_kv,
            packed_kv_pool,
            forward_batch,
        )

        return q_nope, q_pe, q_lora, global_packed_kv

    def _publish_local_packed_c8_prefill(
        self,
        local_packed_kv: torch.Tensor,
        packed_kv_pool: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        """Gather local packed rows once and publish this rank's PA owners."""
        attn_metadata = forward_batch.attn_metadata
        packed_kv_cache_dim = forward_batch.token_to_kv_pool.kv_cache_dim
        global_token_count = attn_metadata.slot_mapping.reshape(-1).numel()
        global_packed_kv = get_attn_tp_group().gather_tensor(
            local_packed_kv,
            forward_batch.global_sp_num_tokens,
        )
        if global_packed_kv.shape[0] != global_token_count:
            raise RuntimeError(f"packed C8 prefill gathered KV/token count mismatch: {global_packed_kv.shape[0]} vs {global_token_count}.")

        owner_rows = attn_metadata.dense_slot_mapping_idx
        owner_slots = attn_metadata.dense_slot_mapping
        if owner_rows is None or owner_slots is None:
            raise RuntimeError("packed C8 local prefill requires dense KVP slot metadata.")
        if owner_rows.numel() != owner_slots.numel():
            raise RuntimeError(f"packed C8 local prefill owner row/slot count mismatch: {owner_rows.numel()} vs {owner_slots.numel()}.")
        if owner_rows.numel() > 0:
            owner_packed_kv = global_packed_kv.index_select(0, owner_rows.long())
            torch_npu.npu_scatter_nd_update_(
                packed_kv_pool.reshape(-1, packed_kv_cache_dim),
                owner_slots.reshape(-1, 1),
                owner_packed_kv.reshape(-1, packed_kv_cache_dim),
            )

        # Keep the first SP AllGather result as the attention staging tensor.
        # The owner scatter above is still required to publish the new rows to
        # the persistent PA cache, but rebuilding the same TND tensor from that
        # cache would otherwise introduce a second AllGather in every layer.
        return global_packed_kv

    @staticmethod
    def _gather_packed_prefill_history(
        packed_kv_pool: torch.Tensor,
        packed_kv_cache_dim: int,
        attn_metadata,
    ) -> torch.Tensor:
        """Rebuild the full sequence-ordered TND KV from the sharded pool."""
        if attn_metadata.kv_index_list is None:
            raise RuntimeError("packed C8 Prefill history requires kv_index_list.")
        packed_kv = packed_kv_pool.reshape(-1, 1, packed_kv_cache_dim).index_select(0, attn_metadata.kv_index_list.long()).contiguous()
        if global_server_args_dict.get("enable_mla_l1_5_cache", False):
            rank_counts = attn_metadata.per_rank_count
            inv_perm = attn_metadata.inv_perm
            if rank_counts is None or inv_perm is None:
                raise RuntimeError("packed C8 KVP Prefill history requires per_rank_count and inv_perm.")
            # HCCL's variable-count INT8 transport is substantially slower
            # than its BF16 path on Atlas A2 for this 656-byte packed row.
            # Reinterpret pairs of bytes as BF16 for transport only; view()
            # does not launch a conversion kernel and preserves every bit.
            if packed_kv_cache_dim % torch.bfloat16.itemsize != 0:
                raise RuntimeError(f"packed C8 BF16 transport requires an even row width, got {packed_kv_cache_dim}.")
            packed_transport = packed_kv.view(torch.bfloat16)
            packed_kv = (
                get_attn_tp_group()
                .gather_tensor(packed_transport, rank_counts)
                .view(torch.int8)
                .reshape(-1, 1, packed_kv_cache_dim)
                .index_select(0, inv_perm.long())
                .contiguous()
            )
        return packed_kv

    def forword_absorb_prepare(
            self,
            rotary_emb,
            positions: torch.Tensor,
            hidden_states: torch.Tensor,
            kv_cache: torch.Tensor,
            forward_batch: ForwardBatch,
            output_cur_k_nope_rope: bool = False,
            atten_id: int = 0):
        cur_k_nope = None
        cur_k_rope = None
        if self._npu_dsa_kv_int8_enabled():
            packed_kv_cache = kv_cache[0] if isinstance(kv_cache, (list, tuple)) else kv_cache
            q_nope, q_rope, q_lora = self._run_packed_c8_mla_prolog(
                rotary_emb,
                hidden_states,
                packed_kv_cache,
                forward_batch,
            )
            return (
                q_nope,
                q_rope,
                packed_kv_cache,
                packed_kv_cache,
                forward_batch,
                q_lora,
                cur_k_nope,
                cur_k_rope,
            )
        if len(hidden_states.size()) == 2:
            hidden_states = torch.unsqueeze(hidden_states, dim=1)
        kv_cache, kr_cache = kv_cache
        if len(kv_cache.size()) == 3:
            kv_cache = torch.unsqueeze(kv_cache, dim=2)
            kr_cache = torch.unsqueeze(kr_cache, dim=2)

        cache_mode = "PA_BSND" if global_server_args_dict["npu_disable_kv_nz"] else "PA_NZ"
        use_sfa1_prolog = (
            self.use_sfa1
            and self.layer_id * 2 + atten_id
            not in forward_batch.attn_backend.inline_current_keyrope_layers(forward_batch)
        )
        prolog_v3 = use_sfa1_prolog or hasattr(torch_npu, "npu_mla_prolog_v3")
        mla_prolog = torch.ops.npu.npu_mla_prolog
        kwargs = dict()
        if prolog_v3:
            # Keep the legacy route on the operator the mainline framework
            # calls. The SFA1 7/9-output ABI lives in its own namespace; the
            # guard above tests torch_npu, so the call must agree with it.
            mla_prolog = (get_sfa1_ops().npu_mla_prolog_v3 if use_sfa1_prolog
                          else torch_npu.npu_mla_prolog_v3)
            kwargs = dict(qc_qr_scale=self.mla_scale_q_lora, kc_scale=self.mla_scale_kv_lora)
        if use_sfa1_prolog:
            # initialize_sfa1_runtime already refuses offload without SFA1, so
            # only that route can ask for the current-KV outputs.
            kwargs["output_cur_k_nope_rope"] = output_cur_k_nope_rope
        mla_output = mla_prolog(
            token_x=hidden_states,  # (B,S,He)
            weight_dq=self.q_a_proj.weight,  # (He,Hcq), (7168,1536)
            weight_uq_qr=self.q_b_proj.weight,  # (Hcq,N*(D+Dr)), (1536, 64*(128+64))
            weight_uk=self.kv_b_proj.W_UK_T,  # (N,D,Hckv), (64, 128, 512)
            weight_dkv_kr=self.kv_a_proj_with_mqa.weight,  # (He,Hckv+Dr), (7168, 576)
            rmsnorm_gamma_cq=self.q_a_layernorm.weight,
            rmsnorm_gamma_ckv=self.kv_a_layernorm.weight,
            rmsnorm_epsilon_cq=self.rms_norm_eps,
            rmsnorm_epsilon_ckv=self.rms_norm_eps,
            rope_sin=rotary_emb[1],  # (B,S,Dr)
            rope_cos=rotary_emb[0],  # (B,S,Dr)
            cache_index=forward_batch.attn_metadata.slot_mapping.reshape(-1, 1),  # (B,S)
            kv_cache=kv_cache,  # (BlockNum, BlockSize, Nkv, Hckv)
            kr_cache=kr_cache,  # (BlockNum, BlockSize, Nkv, Dr)
            dequant_scale_x=None,
            dequant_scale_w_dq=None,
            dequant_scale_w_uq_qr=None,
            dequant_scale_w_dkv_kr=None,
            smooth_scales_cq=None,
            quant_scale_ckv=None,
            cache_mode=cache_mode,
            query_norm_flag=True,
            **kwargs,
        )
        q_nope, q_rope = mla_output[:2]
        q_lora = mla_output[3] if prolog_v3 else None
        if output_cur_k_nope_rope:
            cur_k_nope, cur_k_rope = mla_output[-2], mla_output[-1]
        return (
            q_nope,
            q_rope,
            kv_cache,
            kr_cache,
            forward_batch,
            q_lora,
            cur_k_nope,
            cur_k_rope,
        )

    def forward_absorb_core(
        self,
        atten_id,
        q_nope,
        q_rope,
        k_nope,
        k_rope,
        forward_batch,
        return_ctrl=False,
        topk_indices=None,
        skip_gather_q=False,
        sfad_packed_query=None,
    ):
        # Stage 1 (main, main_aic_limit): index_k -> set_index_k -> index_w.
        # set_index_k stays on main so lightning_indexer reads the cache
        # in-stream (no need to wait on past_key_states for the cache write).
        with npu_super_kernel(f"attn{atten_id}_{self.layer_id}", "stream-fusion=1", flag=forward_batch.can_run_with_graph):
            with npu_limit_core("main", flag=forward_batch.can_run_with_graph and atten_id % 2 == 1):
                layout = "TND_NTD"
                attn_output = self.attn_mqa(
                    q_nope,
                    k_nope,
                    k_nope,
                    forward_batch,
                    q_pe=q_rope,
                    k_pe=k_rope,
                    topk_indices=topk_indices,
                    layout=layout,
                    absorbed=True,
                    atten_id=atten_id,
                    skip_gather_q=skip_gather_q,
                    sfad_packed_query=sfad_packed_query,
                )

            if layout not in ["TND_NTD", "BSND_NBSD"]:
                attn_output = attn_output.view(-1, self.num_local_heads, self.kv_lora_rank).transpose(0, 1)
            # Multiply (N, B, L) x (N, L, V) -> (N, B, V)
            attn_output = torch.bmm(attn_output, self.kv_b_proj.W_UV).transpose(0, 1)
            # Convert from (N, B, V) to (B, N * V)
            output = self.forward_attn_dp_o_proj(attn_output, return_ctrl=return_ctrl)
            return output

    def forward_normal_chunked_kv_prepare(
        self,
        q,
        latent_cache,
        rotary_emb,
        kv_cache: torch.Tensor,
        forward_batch: ForwardBatch,
        reduce_type: str = "reduce_scatter",
        hidden_states: Optional[torch.Tensor] = None,
        packed_prolog_output: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        packed_prefill_kv: Optional[torch.Tensor] = None,
        use_bf16_prefill_staging: bool = False,
        use_native_sfad_prefill: bool = False,
        prefetched_sfad_prefix: Optional[SfadPrefillPrefixState] = None,
    ):
        attn_metadata = forward_batch.attn_metadata
        num_prefill_tokens = attn_metadata.num_prefill_tokens
        num_decode_tokens = attn_metadata.num_decode_tokens

        if use_native_sfad_prefill and (latent_cache is None or num_decode_tokens != 0):
            raise RuntimeError("Native SFAD Prefill requires BF16 extend KV and a Prefill-only batch.")

        cos, sin = rotary_emb
        cos = cos.view(-1, 1, 1, self.qk_rope_head_dim)
        sin = sin.view(-1, 1, 1, self.qk_rope_head_dim)

        if self._npu_dsa_kv_int8_enabled() and not use_bf16_prefill_staging:
            packed_kv_pool = kv_cache[0] if isinstance(kv_cache, (list, tuple)) else kv_cache
            packed_kv_cache_dim = forward_batch.token_to_kv_pool.kv_cache_dim
            if packed_prolog_output is not None:
                q_nope, q_pe = packed_prolog_output
            else:
                if hidden_states is None:
                    raise RuntimeError("NPU DSA KV int8 prefill requires hidden_states.")
                prolog_hidden_states = hidden_states
                local_slice = None
                cache_token_count = attn_metadata.slot_mapping.reshape(-1).numel()
                hidden_token_count = hidden_states.reshape(-1, self.hidden_size).shape[0]
                l15_cache_enabled = global_server_args_dict.get("enable_mla_l1_5_cache", False)
                if self.npu_disable_dsa_head_parallel and (l15_cache_enabled or hidden_token_count != cache_token_count):
                    slice_start, slice_end = get_attn_sp_token_slice_pos(forward_batch)
                    prolog_hidden_states = get_attn_tp_group().gather_tensor(
                        hidden_states,
                        forward_batch.global_sp_num_tokens,
                    )
                    local_slice = (slice_start, slice_end)
                q_nope, q_pe, _ = self._run_packed_c8_mla_prolog(
                    rotary_emb,
                    prolog_hidden_states,
                    packed_kv_pool,
                    forward_batch,
                )
                if local_slice is not None:
                    slice_start, slice_end = local_slice
                    q_nope = q_nope[slice_start:slice_end]
                    q_pe = q_pe[slice_start:slice_end]
            forward_batch.token_to_kv_pool.wait_hicache_load(self.attn_layer_id)

            output = [(num_prefill_tokens, num_decode_tokens)]
            if num_prefill_tokens > 0:
                prefill_q_nope = q_nope[:num_prefill_tokens]
                prefill_q_pe = q_pe[:num_prefill_tokens]
                # Deployed V3 packages may return either an unabsorbed
                # 128-wide query or an already absorbed 512-wide query.
                prefill_q_nope = self._ensure_absorbed_packed_c8_query(prefill_q_nope)
                if packed_prefill_kv is not None:
                    packed_kv = packed_prefill_kv
                    expected_tokens = attn_metadata.slot_mapping.reshape(-1).numel()
                    if (
                        packed_kv.dtype != torch.int8
                        or packed_kv.dim() != 3
                        or packed_kv.shape[0] != expected_tokens
                        or packed_kv.shape[1] != 1
                        or packed_kv.shape[2] != packed_kv_cache_dim
                    ):
                        raise RuntimeError(
                            "packed C8 Prefill staging tensor mismatch: "
                            f"expected ({expected_tokens}, 1, "
                            f"{packed_kv_cache_dim}) int8, got "
                            f"dtype={packed_kv.dtype}, shape={tuple(packed_kv.shape)}."
                        )
                    # Do not feed the HCCL receive buffer directly into the
                    # asynchronous dequant/attention pipeline.  Some deployed
                    # collectives do not retain that buffer until downstream
                    # custom ops finish, which can lead to allocator reuse and
                    # an eventual AIV MTE out-of-range fault.  An explicit
                    # device copy both establishes the stream dependency and
                    # decouples the consumer lifetime, while still avoiding a
                    # second cross-rank AllGather.
                    packed_kv = packed_kv.clone(memory_format=torch.contiguous_format)
                else:
                    packed_kv = self._gather_packed_prefill_history(
                        packed_kv_pool,
                        packed_kv_cache_dim,
                        attn_metadata,
                    )
                output.append(
                    (
                        prefill_q_nope,
                        packed_kv,
                        packed_kv,
                        prefill_q_pe,
                        None,
                    )
                )

            if num_decode_tokens > 0:
                decode_q_nope = q_nope[num_prefill_tokens : num_prefill_tokens + num_decode_tokens]
                decode_q_pe = q_pe[num_prefill_tokens : num_prefill_tokens + num_decode_tokens]
                decode_q_nope = self._ensure_absorbed_packed_c8_query(decode_q_nope)
                output.append(
                    (
                        decode_q_nope,
                        packed_kv_pool,
                        packed_kv_pool,
                        decode_q_pe,
                        None,
                    )
                )
            return output, forward_batch, reduce_type

        q_num_heads = self.num_local_heads
        q_cos = cos
        q_sin = sin

        if self.npu_disable_dsa_head_parallel:
            q_num_heads = self.num_heads
            slice_start, slice_end = get_attn_sp_token_slice_pos(forward_batch)
            q_cos = cos[slice_start:slice_end]
            q_sin = sin[slice_start:slice_end]

        # mla去掉padding
        q = q * self.mla_scale_q_lora
        q = self.q_b_proj(q)[0].view(-1, q_num_heads, self.qk_head_dim)

        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        q_pe = q_pe.unsqueeze(2)
        q_pe = torch_npu.npu_interleave_rope(q_pe, q_cos, q_sin)  # BNSD
        q_pe = q_pe.squeeze(2)  # BSH

        forward_batch.token_to_kv_pool.wait_hicache_load(self.attn_layer_id)

        if use_bf16_prefill_staging:
            prefill_q_nope = None
            if isinstance(latent_cache, _PrefillLatentCacheGather):
                # Finish all independent Q work before joining new-token KV.
                # Keep the original GEMM/normalization/RoPE arithmetic intact.
                prefill_q_nope = torch.bmm(
                    q_nope[:num_prefill_tokens].transpose(0, 1),
                    self.kv_b_proj.W_UK_T,
                ).transpose(0, 1)
                latent_cache = latent_cache.wait()
            if latent_cache is None or num_decode_tokens != 0:
                raise RuntimeError("BF16 Prefill staging requires latent_cache and a Prefill-only batch.")
            if latent_cache.dtype != torch.bfloat16:
                raise RuntimeError(f"BF16 Prefill staging expects BF16 latent_cache, got {latent_cache.dtype}.")
            packed_kv_pool = kv_cache[0] if isinstance(kv_cache, (list, tuple)) else kv_cache
            packed_kv_cache_dim = forward_batch.token_to_kv_pool.kv_cache_dim
            self._validate_packed_c8_cache(packed_kv_pool, packed_kv_cache_dim)

            owner_rows = attn_metadata.dense_slot_mapping_idx
            owner_slots = attn_metadata.dense_slot_mapping
            if owner_rows is None or owner_slots is None:
                raise RuntimeError("BF16 Prefill staging requires dense KVP slot metadata.")
            if owner_rows.numel() != owner_slots.numel():
                raise RuntimeError(f"BF16 Prefill staging owner row/slot count mismatch: {owner_rows.numel()} vs {owner_slots.numel()}.")
            owner_rows = owner_rows.to(torch.int64).contiguous()
            kv_nope, k_pe, owner_packed_kv = torch.ops.custom.npu_packed_kv_prefill_prepare(
                latent_cache.view(-1, self.kv_lora_rank + self.qk_rope_head_dim).contiguous(),
                self.kv_a_layernorm_absorb_scale.weight.contiguous(),
                cos.reshape(-1, self.qk_rope_head_dim).contiguous(),
                sin.reshape(-1, self.qk_rope_head_dim).contiguous(),
                owner_rows,
                epsilon=self.kv_a_layernorm.variance_epsilon,
                packed_row_bytes=packed_kv_cache_dim,
            )
            kv_nope = kv_nope.view(-1, self.kv_lora_rank)
            k_pe = k_pe.view(-1, self.qk_rope_head_dim)

            if owner_rows.numel() > 0:
                if owner_packed_kv.shape[-1] != packed_kv_cache_dim:
                    raise RuntimeError(f"BF16 Prefill staging packed width mismatch: {owner_packed_kv.shape[-1]} vs {packed_kv_cache_dim}.")
                torch_npu.npu_scatter_nd_update_(
                    packed_kv_pool.reshape(-1, packed_kv_cache_dim),
                    owner_slots.reshape(-1, 1),
                    owner_packed_kv.reshape(-1, packed_kv_cache_dim),
                )

            if prefill_q_nope is None:
                prefill_q_nope = torch.bmm(
                    q_nope[:num_prefill_tokens].transpose(0, 1),
                    self.kv_b_proj.W_UK_T,
                ).transpose(0, 1)
            if use_native_sfad_prefill:
                state = prepare_selected_kv(
                    prefill_q_nope,
                    q_pe[:num_prefill_tokens],
                    kv_cache,
                    kv_nope,
                    k_pe,
                )
            elif any(p + e > 1_280_000 for p, e in zip(forward_batch.extend_prefix_lens_cpu, forward_batch.extend_seq_lens_cpu)):
                # MTP SFAP stores the full context in a 10,000 x 128 page table.
                # All ranks use the same batch lengths to select ordinary SFA.
                if any(forward_batch.extend_prefix_lens_cpu):
                    packed_history = self._gather_packed_prefill_history(packed_kv_pool, packed_kv_cache_dim, attn_metadata)
                    attention_k, attention_k_pe = torch.ops.custom.npu_packed_kv_dequant(packed_history)
                else:
                    # Preserve cold-prefill BF16 values, without a KV8 round trip.
                    attention_k = kv_nope.contiguous().view(-1, 1, self.kv_lora_rank)
                    attention_k_pe = k_pe.contiguous().view(-1, 1, self.qk_rope_head_dim)
                state = (prefill_q_nope, attention_k, attention_k, q_pe[:num_prefill_tokens], attention_k_pe)
            else:
                # MTP owns different KV values and uses its page-table path.
                state = prepare_mtp_prefill_pages(
                    prefill_q_nope,
                    q_pe[:num_prefill_tokens],
                    kv_cache,
                    kv_nope,
                    k_pe,
                    forward_batch,
                )
            return [(num_prefill_tokens, num_decode_tokens), state], forward_batch, reduce_type

        # kv [N, 1, 576]
        if len(kv_cache) == 1:
            kv_pool = kv_cache[0]
            kv_nope_pool = kv_pool[..., : self.kv_lora_rank].view(-1, 128, 1, 512)
            k_pe_pool = kv_pool[..., self.kv_lora_rank :].view(-1, 128, 1, 64)
        else:
            kv_nope_pool = kv_cache[0].view(-1, 128, 1, 512)
            k_pe_pool = kv_cache[1].view(-1, 128, 1, 64)

        # k_cache形状为 [block_num, block_size, 1, rope_size]，ckv_cache形状为[block_num, block_size, 1, rms_size]
        # index形状为[batch_size * seq_len]，index里的值表示每个token的偏移。
        # k_pe:BNS,64 kv_a:BNS, 512, kv_states:bnsd, cos,sin:bnsd,kv cache:bsnd
        if len(kv_cache) == 1:
            kv_nope, k_pe = torch.split(latent_cache.view(-1, 576), [512, 64], dim=-1)
            kv_nope = self.kv_a_layernorm_absorb_scale(kv_nope)
            k_pe = k_pe.view(-1, 1, 1, 64)
            k_pe = torch_npu.npu_interleave_rope(k_pe, cos, sin)
            kv = torch.concat((kv_nope, k_pe.view(-1, 64)), dim=-1)
            torch_npu.npu_scatter_nd_update_(kv_pool.reshape(-1, 576), attn_metadata.slot_mapping.reshape(-1, 1), kv.reshape(-1, 576))
        else:
            cache_mode = "PA" if global_server_args_dict["npu_disable_kv_nz"] else "PA_NZ"
            (
                _,
                _,
                output_k_pe,
                output_kv_nope,
            ) = torch_npu.npu_kv_rmsnorm_rope_cache(
                latent_cache.view(-1, 1, 1, 576),  # bnsd
                self.kv_a_layernorm_absorb_scale.weight,
                cos.view(-1, 1, 1, self.qk_rope_head_dim),
                sin.view(-1, 1, 1, self.qk_rope_head_dim),
                attn_metadata.slot_mapping,
                k_pe_pool,
                kv_nope_pool,
                k_rope_scale=None,
                c_kv_scale=None,
                k_rope_offset=None,
                c_kv_offset=None,
                epsilon=self.kv_a_layernorm.variance_epsilon,
                cache_mode=cache_mode,
                is_output_kv=use_native_sfad_prefill,
            )
            if use_native_sfad_prefill:
                k_pe = output_k_pe
                kv_nope = output_kv_nope

        if use_native_sfad_prefill:
            if not self.enable_nsa:
                raise RuntimeError("Native SFAD Prefill is only valid for NSA models.")
            logger.info_once("Native SparseMlaAttentionPrefill BF16-cache path is active.")
            prefill_q_nope = torch.bmm(
                q_nope[:num_prefill_tokens].transpose(0, 1),
                self.kv_b_proj.W_UK_T,
            ).transpose(0, 1)
            sfad_state = prepare_prefill_pages_local(
                prefill_q_nope,
                q_pe[:num_prefill_tokens],
                kv_cache,
                kv_nope.reshape(-1, self.kv_lora_rank),
                k_pe.reshape(-1, self.qk_rope_head_dim),
                forward_batch,
                prefetched_prefix=prefetched_sfad_prefix,
            )
            return (
                [
                    (num_prefill_tokens, num_decode_tokens),
                    sfad_state,
                ],
                forward_batch,
                reduce_type,
            )

        output = [(num_prefill_tokens, num_decode_tokens)]
        if num_prefill_tokens > 0:
            prefill_q_nope = q_nope[:num_prefill_tokens]
            prefill_q_pe = q_pe[:num_prefill_tokens]
            if len(kv_cache) == 1:
                kv = kv_pool.reshape(-1, self.kv_lora_rank + self.qk_rope_head_dim).index_select(0, attn_metadata.kv_index_list)
                kv_a, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
            else:
                if not global_server_args_dict["npu_disable_kv_nz"]:
                    KVCACHE_NZ_DIM = 16  # for half
                    block_num, block_size, _, _ = kv_nope_pool.shape
                    if attn_metadata.kv_block_index_list is not None:
                        mini_kv_nope_pool = kv_nope_pool.index_select(0, attn_metadata.kv_block_index_list)
                        mini_kv_pe_pool = k_pe_pool.index_select(0, attn_metadata.kv_block_index_list)
                        kv_cache_a = mini_kv_nope_pool.view(-1, 1, self.kv_lora_rank // KVCACHE_NZ_DIM, block_size, KVCACHE_NZ_DIM)
                        kv_cache_pe = mini_kv_pe_pool.view(-1, 1, self.qk_rope_head_dim // KVCACHE_NZ_DIM, block_size, KVCACHE_NZ_DIM)
                    else:
                        kv_cache_a = kv_nope_pool.view(block_num, 1, self.kv_lora_rank // KVCACHE_NZ_DIM, block_size, KVCACHE_NZ_DIM)
                        kv_cache_pe = k_pe_pool.view(block_num, 1, self.qk_rope_head_dim // KVCACHE_NZ_DIM, block_size, KVCACHE_NZ_DIM)
                    kv_nope_pool = kv_cache_a.transpose(1, 3)
                    k_pe_pool = kv_cache_pe.transpose(1, 3)
                kv_a = kv_nope_pool.reshape(-1, self.kv_lora_rank).index_select(0, attn_metadata.kv_index_list)
                k_pe = k_pe_pool.reshape(-1, self.qk_rope_head_dim).index_select(0, attn_metadata.kv_index_list)

                if global_server_args_dict.get("enable_mla_l1_5_cache", False):
                    rank_counts = attn_metadata.per_rank_count
                    inv_perm = attn_metadata.inv_perm
                    kv_a = get_attn_tp_group().gather_tensor(kv_a, rank_counts).view(-1, self.kv_lora_rank).index_select(0, inv_perm)
                    k_pe = get_attn_tp_group().gather_tensor(k_pe, rank_counts).view(-1, self.qk_rope_head_dim).index_select(0, inv_perm)

            if self.enable_nsa:
                # TND [bs,kv_n,dim]
                k_nope = kv_a.contiguous().view(-1, 1, self.kv_lora_rank)
                k_pe = k_pe.contiguous().view(-1, 1, self.qk_rope_head_dim)
                v = k_nope
                prefill_q_nope = prefill_q_nope.transpose(0, 1)
                prefill_q_nope = torch.bmm(prefill_q_nope, self.kv_b_proj.W_UK_T)
                prefill_q_nope = prefill_q_nope.transpose(0, 1)
            else:
                kv_a = kv_a.contiguous().view(-1, self.kv_lora_rank)
                k_pe = k_pe.contiguous().view(-1, 1, self.qk_rope_head_dim).repeat(1, self.num_local_heads, 1)

                # TODO group attention
                kv = self.kv_b_proj(kv_a)[0].view(-1, self.num_local_heads, self.qk_nope_head_dim + self.v_head_dim)
                k_nope, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)

            output.append((prefill_q_nope, k_nope, v, prefill_q_pe, k_pe))

        if num_decode_tokens > 0:
            decode_q_nope = q_nope[num_prefill_tokens : num_prefill_tokens + num_decode_tokens]
            decode_q_pe = q_pe[num_prefill_tokens : num_prefill_tokens + num_decode_tokens]
            decode_q_nope = decode_q_nope.transpose(0, 1)
            decode_ql_nope = torch.bmm(decode_q_nope, self.kv_b_proj.W_UK_T)
            decode_ql_nope = decode_ql_nope.transpose(0, 1)
            output.append((decode_ql_nope, kv_cache[0], kv_cache[0], decode_q_pe, kv_cache[1]))
        return output, forward_batch, reduce_type

    def forward_normal_chunked_kv_core(self, kv_data, forward_batch: ForwardBatch, reduce_type, topk_indices=None, selected_route_cache=None):
        # after attention, add padding
        assert len(kv_data) >= 1
        num_prefill_tokens, num_decode_tokens = kv_data[0]
        output_size = num_prefill_tokens + num_decode_tokens
        if len(kv_data) >= 2 and isinstance(kv_data[1], SelectedKvPrefillState):
            if num_decode_tokens != 0 or topk_indices is None:
                raise RuntimeError("Selected-KV Prefill requires sparse Prefill TopK")
            state = kv_data[1]
            step_counter = getattr(forward_batch.attn_backend, "step_counter", None)
            if step_counter is not None:
                step_counter.record_cache()
            attn_output = selected_kv_attention(
                state,
                topk_indices,
                forward_batch,
                self.scaling,
                get_attn_tp_group(),
                get_attn_sp_token_slice_pos(forward_batch),
                get_sfad_prefill_stage_stream(),
                route_cache=selected_route_cache,
            )
            if state.query.shape[0] == 0:
                return state.query.new_empty((0, self.hidden_size))
            attn_output = torch.bmm(attn_output, self.kv_b_proj.W_UV).transpose(0, 1).contiguous()
            output = attn_output.view(-1, self.v_head_dim * state.query.shape[1])
            token_num = None if reduce_type == "skip" else get_attn_tp_group().get_local_sp_token_num(forward_batch.global_sp_num_tokens)
            return self.forward_attn_tp_o_proj(output, reduce_type, token_num)
        if len(kv_data) >= 2 and isinstance(kv_data[1], SfadPrefillState):
            if num_decode_tokens != 0:
                raise RuntimeError("Native SFAD Prefill only accepts a Prefill-only batch.")
            if topk_indices is None:
                raise RuntimeError("Native SFAD Prefill requires LightningIndexer TopK output.")
            sfad_state = kv_data[1]
            if sfad_state.gathered_pages is None:
                raise RuntimeError("Native SFAD Prefill page AllGather was not submitted.")
            # The native SFAD path bypasses RadixAttention and therefore also
            # bypasses AttentionBackend.forward(), which normally advances the
            # Mooncake layerwise-transfer cache-ready counter.  Page AllGather
            # has already joined the cache-write dependency onto this stream,
            # so it is safe to publish the current layer here.  Do this before
            # the empty-SP-rank return as every rank must advance the counter.
            step_counter = getattr(forward_batch.attn_backend, "step_counter", None)
            if step_counter is not None:
                step_counter.record_cache()
            logical_rows, scan_count, page_table, query_seq = sfad_topk_metadata(topk_indices, sfad_state.plan)
            if sfad_state.query.shape[0] == 0:
                # Every rank has already participated in Indexer/Page-Gather
                # collectives. Skip only local kernels on an empty SP rank;
                # this path is gated to Prefill OProj TP1.
                return sfad_state.query.new_empty((0, self.hidden_size))
            sfad_args = (
                sfad_state.query,
                sfad_state.gathered_pages,
                logical_rows,
                scan_count,
                page_table,
                query_seq,
                self.scaling,
            )
            # Request contiguous [H,T,512] output and feed it directly to the
            # BF16 projection BMM. Only the final [H,T,128] -> [T,H,128]
            # conversion remains.
            attn_output = _NATIVE_SFAD_PREFILL_OP(*sfad_args, True)
            attn_output = torch.bmm(attn_output, self.kv_b_proj.W_UV).transpose(0, 1).contiguous()
            output = attn_output.view(-1, self.v_head_dim * sfad_state.query.shape[1])
            attn_sp_token_num = None if reduce_type == "skip" else get_attn_tp_group().get_local_sp_token_num(forward_batch.global_sp_num_tokens)
            return self.forward_attn_tp_o_proj(output, reduce_type, attn_sp_token_num)
        if num_decode_tokens > 0:
            output = torch.empty(output_size, self.v_head_dim * self.num_local_heads, device=kv_data[1][0].device, dtype=kv_data[1][0].dtype)
        if num_prefill_tokens > 0:
            assert len(kv_data) >= 2
            q, k, v, q_pe, k_pe = kv_data[1]
            layout = "TND"
            q_num_heads = q.shape[1]
            attn_output = self.attn_mha(q, k, v, forward_batch, q_pe=q_pe, k_pe=k_pe, topk_indices=topk_indices, layout=layout)
            if self.enable_nsa:
                # 512 -> 128
                attn_output_transposed = attn_output.transpose(0, 1)  # [N, B*S, L]
                attn_output = torch.bmm(attn_output_transposed, self.kv_b_proj.W_UV)  # [N, B*S, V]
                attn_output = attn_output.transpose(0, 1).contiguous()
            if num_decode_tokens > 0:
                output[:num_prefill_tokens] = attn_output.view(-1, self.v_head_dim * q_num_heads)
            else:
                output = attn_output.view(-1, self.v_head_dim * q_num_heads)
        if num_decode_tokens > 0:
            assert len(kv_data) >= 3
            q, k, v, q_pe, k_pe = kv_data[2]
            layout = "TND"
            attn_output = self.attn_mqa(q, k, v, forward_batch, q_pe=q_pe, k_pe=k_pe, layout=layout, absorbed=True)

            if layout not in ["TND_NTD", "BSND_NBSD"]:
                attn_output = attn_output.view(-1, self.num_local_heads, self.kv_lora_rank).transpose(0, 1)
            # Multiply (N, B, L) x (N, L, V) -> (N, B, V)
            output[num_prefill_tokens : num_prefill_tokens + num_decode_tokens] = torch.bmm(attn_output, self.kv_b_proj.W_UV)
        attn_sp_token_num = None if reduce_type == "skip" else get_attn_tp_group().get_local_sp_token_num(forward_batch.global_sp_num_tokens)
        o_proj_output = self.forward_attn_tp_o_proj(output, reduce_type, attn_sp_token_num)
        return o_proj_output

    def forward_attn_tp_o_proj(self, attn_out, reduce_type, attn_sp_token_num=None):
        if global_server_args_dict["npu_o_proj_tp_size"] == 0 or global_server_args_dict["npu_o_proj_tp_size"] == self.attn_tp_size:
            if reduce_type == "skip":
                return self.o_proj(attn_out, force_skip_reduce_results=True)[0]
            elif reduce_type == "reduce_scatter":
                out = self.o_proj(attn_out, force_skip_reduce_results=True)[0]
                return self.post_comm(out, reduce_type, input_split_sizes=attn_sp_token_num)
            else:
                raise ValueError(f"o_project tp not support reduce type {reduce_type}")

        assert get_attn_o_proj_tp_group().world_size == 1, "only support o_proj tp1 on prefill"
        if self.npu_disable_dsa_head_parallel:
            return self.o_proj(attn_out, force_skip_reduce_results=True)[0]
        h = attn_out.shape[1]
        attn_tp_size = get_attn_tp_group().world_size
        attn_tp_rank = get_attn_tp_group().rank_in_group
        output_flat = attn_out.reshape(-1)
        output_all2all = torch.empty(
            [
                attn_sp_token_num[attn_tp_rank] * h * attn_tp_size,
            ],
            dtype=output_flat.dtype,
            device=output_flat.device,
        )

        input_split_sizes = [x * h for x in attn_sp_token_num]
        output_split_sizes = [h * attn_sp_token_num[attn_tp_rank]] * attn_tp_size
        # logger.info(f"{output_flat.shape=} {input_split_sizes=} {output_split_sizes=} {attn_sp_token_num=}")
        torch.distributed.all_to_all_single(
            output_all2all,
            output_flat,
            group=get_attn_tp_group().device_group,
            input_split_sizes=input_split_sizes,
            output_split_sizes=output_split_sizes,
        )
        # logger.info(f"{output_all2all.shape} {input_split_sizes=} {output_split_sizes=}")

        #  shape '[8, 1, 1024]' is invalid for input of size 4096
        output = (
            output_all2all.reshape(attn_tp_size, attn_sp_token_num[attn_tp_rank], h)
            .permute(1, 0, 2)
            .contiguous()
            .reshape(attn_sp_token_num[attn_tp_rank], h * attn_tp_size)
        )
        o_proj_output = self.o_proj(output, force_skip_reduce_results=True)[0]
        return o_proj_output

    def post_comm(self, o_proj_output, reduce_type="reduce_scatter", input_split_sizes=None):
        if reduce_type == "reduce_scatter":
            if not self.npu_disable_dsa_head_parallel:
                return self.o_proj_tp_group.reduce_scatter(o_proj_output, input_split_sizes=input_split_sizes)
            else:
                return o_proj_output
        else:
            raise ValueError(f"o_project tp not support reduce type {reduce_type}")

    def forward_attn_dp_o_proj(self, attn_out, return_ctrl=False):
        if global_server_args_dict["npu_o_proj_tp_size"] <= 0:
            o_proj_output = self.o_proj(attn_out.reshape(attn_out.shape[0], -1))[0]
            return o_proj_output, None
        bs = attn_out.shape[0]

        if isinstance(self.o_proj, ColumnParallelLinear):
            # dp -> tp n -> dp: all_gather(bs) -> o_proj(k, n/tp) -> all2all(split n -> split bs)
            attn_out = self.o_proj_tp_group.all_gather(attn_out, dim=0)

            o_proj_output = self.o_proj(attn_out.reshape(attn_out.shape[0], -1))[0]

            output = torch.empty([bs * self.hidden_size], dtype=attn_out.dtype, device=attn_out.device)
            torch.distributed.all_to_all_single(output, o_proj_output.reshape(-1), group=self.o_proj_tp_group.device_group)
            output = output.view(-1, bs, o_proj_output.shape[-1]).transpose(0, 1).reshape(bs, hidden_size_global)
        else:
            # dp -> tp k -> dp: all2all(split bs -> split k) -> o_proj(k/tp, n) -> reduce_scatter(partial output -> split bs)
            # reshape for matmul+tranpose fusion
            if get_attn_tp_group().world_size == 1:
                attn_out_tmp = (
                    attn_out.reshape(bs, 1, -1)
                    .view(
                        bs,
                        global_server_args_dict["npu_o_proj_tp_size"],
                        (self.num_local_heads * v_head_dim_global) // global_server_args_dict["npu_o_proj_tp_size"],
                    )
                    .transpose(0, 1)
                    .reshape(-1)
                )
                attn_out_k = torch.empty([bs * self.num_local_heads * v_head_dim_global], dtype=attn_out.dtype, device=attn_out.device)
                torch.distributed.all_to_all_single(attn_out_k, attn_out_tmp, group=self.o_proj_tp_group.device_group)
                attn_out = attn_out_k.view(bs * global_server_args_dict["npu_o_proj_tp_size"], -1)

                o_proj_output = self.o_proj(attn_out, force_skip_reduce_results=True)[0]

                output = torch.empty([bs, hidden_size_global], dtype=attn_out.dtype, device=attn_out.device)
                torch.distributed.reduce_scatter_tensor(output, o_proj_output, group=self.o_proj_tp_group.device_group)
            elif get_attn_tp_group().world_size == get_attn_o_proj_tp_group().world_size:
                output = self.o_proj(attn_out.reshape(attn_out.shape[0], self.num_local_heads * v_head_dim_global), force_skip_reduce_results=True)[0]
                torch.distributed.all_reduce(output, group=self.o_proj_tp_group.device_group)
                o_proj_output = output
            else:
                raise AssertionError("not support now")
        if return_ctrl:
            return output, o_proj_output
        return output

    def forward_indexer_mla_prolog(
        self,
        rotary_emb,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        forward_batch: ForwardBatch,
        atten_id: int = 0,
        output_cur_k_nope_rope: bool = False,
    ):
        # Pipeline:
        #   stage 1 (main, main_aic_limit) || stage 2 (side, side_aic_limit)
        #     main: index_k + set_index_k + index_w   (k before w)
        #     side: q_lora + index_q                  (q produced last on side)
        #   stage 3 (main, main_aic_limit) || stage 3b (side, side_aic_limit)
        #     main: lightning_indexer  (waits on q to cover all side ops)
        #     side: mla_prolog (forword_absorb_prepare)
        if not self.indexer.skip_rope:
            cos, sin = forward_batch.nsa_rotary_emb
        slot_mapping = forward_batch.attn_metadata.slot_mapping
        block_table = forward_batch.attn_metadata.block_table
        bs = hidden_states.shape[0]
        layer_id = self.layer_id * 2 + atten_id

        # Stage 1 (main, main_aic_limit): index_k -> set_index_k -> index_w.
        # set_index_k stays on main so lightning_indexer reads the cache
        # in-stream (no need to wait on past_key_states for the cache write).
        with (
            npu_super_kernel(f"attn{atten_id}_{self.layer_id}", "stream-fusion=1", flag=forward_batch.can_run_with_graph),
            npu_limit_core("indexer_prolog", flag=forward_batch.can_run_with_graph),
        ):
            k_proj = self.indexer.wk(hidden_states)[0]
            k = self.indexer.k_norm(k_proj).to(k_proj.dtype)
            k_pe, k_nope = torch.split(
                k,
                [self.indexer.rope_head_dim, self.indexer.head_dim - self.indexer.rope_head_dim],
                dim=-1,
            )
            if not self.indexer.skip_rope:
                k_pe = k_pe.view(-1, 1, 1, self.indexer.rope_head_dim)
                k_pe = torch_npu.npu_interleave_rope(k_pe, cos, sin).view(bs, 1, self.indexer.rope_head_dim)
                k = torch.cat([k_pe, k_nope.unsqueeze(1)], dim=-1)
            else:
                k = k.unsqueeze(1)

        with npu_limit_core("indexer_prolog", flag=forward_batch.can_run_with_graph):
            forward_batch.token_to_kv_pool.set_index_k_and_scale_buffer(layer_id, slot_mapping, k)

        with (
            npu_super_kernel(f"attn{atten_id}_{self.layer_id}", "stream-fusion=1", flag=forward_batch.can_run_with_graph),
            npu_limit_core("indexer_prolog", flag=forward_batch.can_run_with_graph),
        ):
            hidden_states_v = hidden_states.view(-1, self.hidden_size)
            weights = self.indexer.weights_proj(hidden_states_v.to(torch.float32))[0]

        with (
            npu_super_kernel(f"mla_prolog_{self.layer_id}", "stream-fusion=1", flag=forward_batch.can_run_with_graph),
            npu_stream_switch(forward_batch.can_run_with_graph, "scmoe_steam"),
            npu_limit_core("mla_prolog", flag=forward_batch.can_run_with_graph),
        ):
            s = self.forword_absorb_prepare(
                rotary_emb,
                positions,
                hidden_states,
                kv_cache,
                forward_batch,
                output_cur_k_nope_rope=output_cur_k_nope_rope,
                atten_id=atten_id,
            )
            # mla_prolog outputs q_lora at mla_output[3] (prolog_v3)
            q_nope_prolog, q_rope_prolog, kv_cache_prolog, kr_cache_prolog, _, q_lora = s[:6]

        # Stage 2 (side, side_aic_limit): index_q using prolog's q_lora.
        # No need to recompute q_a_proj + q_a_layernorm since mla_prolog already
        # produced q_lora.
        with npu_super_kernel(f"mla_prolog_{self.layer_id}", "stream-fusion=1", flag=forward_batch.can_run_with_graph):
            with (
                npu_stream_switch(forward_batch.can_run_with_graph, "scmoe_steam"),
                npu_limit_core("mla_prolog", flag=forward_batch.can_run_with_graph),
            ):
                q = self.indexer.wq_b(q_lora)[0]
                q = q.view(-1, self.indexer.n_heads, self.indexer.head_dim)
                q_pe, q_nope = torch.split(
                    q,
                    [self.indexer.rope_head_dim, self.indexer.head_dim - self.indexer.rope_head_dim],
                    dim=-1,
                )
                if not self.indexer.skip_rope:
                    q_pe = q_pe.view(-1, self.indexer.n_heads, 1, self.indexer.rope_head_dim)
                    q_pe = torch_npu.npu_interleave_rope(q_pe, cos, sin).view(-1, self.indexer.n_heads, self.indexer.rope_head_dim)
                    q = torch.cat([q_pe, q_nope], dim=-1)

                layout_key = "PA_BSND"
                actual_seq_lengths_q = forward_batch.attn_metadata.query_len_tensor
                actual_seq_lengths_kv = forward_batch.attn_metadata.cu_seq_lens_tensor_int32
                past_key_states = forward_batch.token_to_kv_pool.get_index_k_with_scale_buffer(layer_id)

            # Stage 3a (main): single wait on q (last side op) covers the entire side
            # stream. Target `weights` (a main-stream tensor consumed by
            # lightning_indexer) so we don't touch past_key_states.
            if forward_batch.can_run_with_graph:
                tng.scope.npu_wait_tensor(weights, q)

        use_dcp = global_server_args_dict["kvp_size"] > 1
        sfad_packed_query = None
        if use_dcp:
            skip_gather_q = True
            (
                topk_indices,
                g_q_nope_prolog,
                g_q_rope_prolog,
                sfad_packed_query,
            ) = self.indexer.npu_indexer_dcp(
                q, k, actual_seq_lengths_q, layout_key, past_key_states, weights, forward_batch, layer_id, [q_nope_prolog, q_rope_prolog]
            )
            # 用 kvp 聚合后的 q 替换 prolog 输出，下游无需再 gather q
            s = (g_q_nope_prolog, g_q_rope_prolog) + s[2:]
        else:
            with (
                npu_super_kernel(f"attn{atten_id}_{self.layer_id}", "stream-fusion=1", flag=forward_batch.can_run_with_graph),
                npu_limit_core("indexer_prolog", flag=forward_batch.can_run_with_graph),
            ):
                topk_indices = self.indexer.npu_indexer_normal(
                    q,
                    k,
                    layout_key,
                    actual_seq_lengths_q,
                    actual_seq_lengths_kv,
                    past_key_states,
                    weights,
                    forward_batch,
                    block_table,
                    layer_id,
                )
            skip_gather_q = False
        return topk_indices, s, skip_gather_q, sfad_packed_query

    def can_overlap_indexer_prolog(
        self,
        forward_batch: ForwardBatch,
        atten_id: int = 0,
        gather_qkv: bool = False,
        extra_input: Any = None,
    ) -> bool:
        if not self.enable_nsa:
            return False
        if not forward_batch.can_run_with_graph:
            return False
        if gather_qkv or extra_input is not None:
            return False
        if self.indexer.npu_disable_dsa_head_parallel:
            return False
        # The prolog path needs precomputed nsa rope. Draft (NextN decode) leaves
        # nsa_rotary_emb=None, so fall back to the legacy path there.
        if forward_batch.nsa_rotary_emb is None:
            return False

        if "FLASHForCausalLMNextN" in self.config.architectures[0]:
            if global_server_args_dict["is_multi_head_eagle"]:
                return forward_batch.topk_indices is None
            return True

        layer_id = self.layer_id * 2 + atten_id
        return self.cli_factor <= 1 or layer_id % self.cli_factor == 0

    def _can_use_native_sfad_prefill(
        self,
        kv_cache,
        forward_batch: ForwardBatch,
    ) -> bool:
        """Return whether this Prefill batch has a supported SFAD KV layout."""
        architectures = getattr(self.config, "architectures", ()) or ()
        cache_tensors = tuple(kv_cache) if isinstance(kv_cache, (list, tuple)) else (kv_cache,)
        if self._npu_dsa_kv_int8_enabled():
            supported_cache = cache_tensors[0].dtype == torch.int8 and getattr(
                forward_batch.token_to_kv_pool,
                "dsa_kv_int8_packed",
                False,
            )
        else:
            # Native SFAD always consumes a BF16 page workspace.  A BF16 PA
            # cache can be staged directly; only packed INT8 needs dequant.
            supported_cache = (
                (
                    global_server_args_dict.get("npu_enable_prefill_two_stream", False)
                    or global_server_args_dict.get("npu_enable_w4a4_prefill_two_stream", False)
                )
                and global_server_args_dict.get("npu_disable_kv_nz", False)
                and all(x.dtype == torch.bfloat16 for x in cache_tensors)
            )

        return (
            _NATIVE_SFAD_PREFILL_OP is not None
            and self.enable_nsa
            and supported_cache
            and forward_batch.attn_metadata.num_decode_tokens == 0
            and global_server_args_dict.get("enable_mla_l1_5_cache", False)
            and global_server_args_dict.get("npu_enable_sp_for_indexer", False)
            and global_server_args_dict.get("npu_disable_dsa_head_parallel", False)
            and global_server_args_dict.get("npu_o_proj_tp_size", 0) == 1
            and self.kv_lora_rank == 512
            and self.qk_rope_head_dim == 64
            and self.index_topk == 2048
            # MTP has an independent page-gather path; it must never enter
            # target Selected-KV dispatch or reuse a target CLI route.
            and not any("FLASHForCausalLMNextN" in x for x in architectures)
            and forward_batch.extend_prefix_lens_cpu is not None
        )

    def _validate_packed_prefill_sfap(self):
        """Validate the optimized packed-cache Prefill configuration."""
        if _NATIVE_SFAD_PREFILL_OP is None:
            raise RuntimeError("Packed Prefill requires the native SFAP operator to be installed.")
        if (self.kv_lora_rank, self.qk_rope_head_dim, self.index_topk) != (512, 64, 2048):
            raise ValueError("Packed Prefill SFAP requires KV dimensions 512+64 and TopK=2048.")
        if not global_server_args_dict.get("npu_enable_sp_for_indexer", False) or global_server_args_dict.get("npu_o_proj_tp_size", 0) != 1:
            raise ValueError("Packed Prefill SFAP requires Indexer SP and npu_o_proj_tp_size=1.")

    def can_prefetch_native_sfad_prefix(
        self,
        kv_cache,
        forward_batch: ForwardBatch,
        atten_id: int,
    ) -> bool:
        """Whether a CLI-reuse Attention may prefetch cached prefix pages."""
        logical_attention_id = self.layer_id * 2 + atten_id
        return (
            self._can_use_native_sfad_prefill(kv_cache, forward_batch)
            and not self._npu_dsa_kv_int8_enabled()
            and self.cli_factor > 1
            and logical_attention_id % self.cli_factor != 0
            and forward_batch.extend_prefix_lens_cpu is not None
            and any(int(x) > 0 for x in forward_batch.extend_prefix_lens_cpu)
        )

    def prefetch_native_sfad_prefix(
        self,
        kv_cache,
        forward_batch: ForwardBatch,
        producer_ready_event,
    ) -> tuple[SfadPrefillPrefixState, Any]:
        """Stage cached-prefix BF16 pages and submit their Page AllGather."""
        stage_stream = get_sfad_prefill_stage_stream()
        prefix_ready = torch.npu.Event()
        with torch.npu.stream(stage_stream):
            stage_stream.wait_event(producer_ready_event)
            forward_batch.token_to_kv_pool.wait_hicache_load(self.attn_layer_id)
            prefix_state = prefetch_prefill_prefix_pages(kv_cache, forward_batch)
            prefix_ready.record(stage_stream)
        return prefix_state, prefix_ready

    def _prepare_native_sfad_prefill(
        self,
        q_lora: torch.Tensor,
        latent_cache: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        rotary_emb,
        kv_cache,
        forward_batch: ForwardBatch,
        reduce_type: str,
        atten_id: int,
        prefill_sfad_prefix_state: Optional[SfadPrefillPrefixState],
        prefill_sfad_prefix_ready_event: Optional[Any],
        *,
        packed_cache_staging: bool,
    ):
        """Prepare Native SFAD while preserving the shared collective order."""
        if packed_cache_staging:
            # Selected-KV does not use a full-history workspace or page prefetch.
            slice_pos = get_attn_sp_token_slice_pos(forward_batch)
            prepare_native_selected_metadata(forward_batch, slice_pos)
            topk_indices = self.prefill_indexer(
                q_lora,
                positions,
                hidden_states,
                forward_batch,
                atten_id,
            )
            if self.cli_factor == 2 and atten_id == 1:
                pair = forward_batch._selected_kv_route_pair
                if pair[0] != self.layer_id or pair[1] is not topk_indices:
                    raise RuntimeError("Selected-KV CLI pair must share the same TopK")
                route_cache = pair[2]
            else:
                route_cache = {}
                forward_batch._selected_kv_route_pair = (self.layer_id, topk_indices, route_cache)
            pool = kv_cache[0] if isinstance(kv_cache, (tuple, list)) else kv_cache
            begin_native_selected_plan(
                topk_indices,
                pool.reshape(-1, pool.shape[-1]),
                forward_batch,
                get_attn_tp_group(),
                slice_pos,
                get_sfad_prefill_stage_stream(),
                route_cache,
            )
            prepared = self.forward_normal_chunked_kv_prepare(
                q_lora,
                latent_cache,
                rotary_emb,
                kv_cache,
                forward_batch,
                reduce_type,
                use_bf16_prefill_staging=True,
                use_native_sfad_prefill=True,
            )
            return topk_indices, prepared
        compute_stream = torch.npu.current_stream()
        stage_stream = get_sfad_prefill_stage_stream()
        reserve_prefill_workspace(forward_batch)
        update_prefill_topk = self._should_update_prefill_topk(atten_id)
        input_ready = torch.npu.Event()
        page_collective_ready = torch.npu.Event()
        page_ready = torch.npu.Event()
        input_ready.record(compute_stream)

        if update_prefill_topk:
            if prefill_sfad_prefix_state is not None or prefill_sfad_prefix_ready_event is not None:
                raise RuntimeError("Native SFAD CLI prefix prefetch was provided to an Indexer-update Attention.")
            local_prefix_ready = torch.npu.Event()
            with torch.npu.stream(stage_stream):
                stage_stream.wait_event(input_ready)
                forward_batch.token_to_kv_pool.wait_hicache_load(self.attn_layer_id)
                local_prefix_state = prepare_prefill_prefix_pages_local(kv_cache, forward_batch)
                local_prefix_ready.record(stage_stream)

            page_gather_submitted = [False]
            gathered_prefix_state = [None]
            indexer_inputs_ready = torch.npu.Event()

            def record_indexer_inputs_ready():
                current_stream = torch.npu.current_stream()
                indexer_inputs_ready.record(current_stream)
                # Avoid running two AIV-heavy local kernels concurrently.
                # Page AG can still overlap the LightningIndexer kernel.
                current_stream.wait_event(local_prefix_ready)

            def submit_page_gather():
                if page_gather_submitted[0]:
                    raise RuntimeError("Native SFAD Prefill Page AllGather was submitted more than once.")
                with torch.npu.stream(stage_stream):
                    stage_stream.wait_event(indexer_inputs_ready)
                    gathered_prefix_state[0] = submit_prefill_prefix_page_gather(
                        local_prefix_state,
                        collective_done_event=page_collective_ready,
                    )
                page_gather_submitted[0] = True

            def wait_page_collective_before_reverse():
                if not page_gather_submitted[0]:
                    raise RuntimeError("Native SFAD Prefill Page AllGather was not submitted before reverse TopK A2A.")
                # Attention Page AG and reverse TopK A2A share a communicator.
                torch.npu.current_stream().wait_event(page_collective_ready)

            topk_indices = self.prefill_indexer(
                q_lora,
                positions,
                hidden_states,
                forward_batch,
                atten_id,
                after_input_redistribute_callback=record_indexer_inputs_ready,
                between_input_and_core_callback=submit_page_gather,
                before_output_redistribute_callback=(wait_page_collective_before_reverse),
            )
            if gathered_prefix_state[0] is None:
                raise RuntimeError("Native SFAD Prefill Page AllGather did not produce a gathered-prefix state.")
            with torch.npu.stream(stage_stream):
                s = self.forward_normal_chunked_kv_prepare(
                    q_lora,
                    latent_cache,
                    rotary_emb,
                    kv_cache,
                    forward_batch,
                    reduce_type,
                    use_bf16_prefill_staging=packed_cache_staging,
                    use_native_sfad_prefill=True,
                    prefetched_sfad_prefix=gathered_prefix_state[0],
                )
                sfad_state = s[0][1]
                submit_prefill_page_gather(sfad_state)
                page_ready.record(stage_stream)
            # Native SFAD is the first consumer of the assembled workspace.
            compute_stream.wait_event(page_ready)
            return topk_indices, s

        # A CLI-reuse layer skips Indexer callbacks. Its optional gathered
        # prefix was prefetched across the preceding Dense/MoE work.
        with torch.npu.stream(stage_stream):
            stage_stream.wait_event(input_ready)
            if prefill_sfad_prefix_ready_event is not None:
                stage_stream.wait_event(prefill_sfad_prefix_ready_event)
            s = self.forward_normal_chunked_kv_prepare(
                q_lora,
                latent_cache,
                rotary_emb,
                kv_cache,
                forward_batch,
                reduce_type,
                use_bf16_prefill_staging=packed_cache_staging,
                use_native_sfad_prefill=True,
                prefetched_sfad_prefix=prefill_sfad_prefix_state,
            )
            sfad_state = s[0][1]
            submit_prefill_page_gather(sfad_state)
            page_ready.record(stage_stream)
        topk_indices = self.prefill_indexer(
            q_lora,
            positions,
            hidden_states,
            forward_batch,
            atten_id,
        )
        if topk_indices is None:
            raise RuntimeError("Native SFAD Prefill CLI reuse layer has no cached TopK output.")
        compute_stream.wait_event(page_ready)
        return topk_indices, s

    def forward(
        self,
        rotary_emb,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        forward_batch: ForwardBatch,
        gather_qkv: bool = False,
        reduce_type: str = "reduce_scatter",
        return_ctrl: bool = False,
        extra_input: Any = None,
        atten_id: int = 0,
        prefill_core_ready_event: Optional[Any] = None,
        prefill_before_core_callback: Optional[Any] = None,
        prefill_sfad_prefix_state: Optional[SfadPrefillPrefixState] = None,
        prefill_sfad_prefix_ready_event: Optional[Any] = None,
        prefill_selected_next_pool=None,
    ):
        if forward_batch.all_decode_or_idle:
            skip_gather_q = False
            sfad_packed_query = None
            coordinator = getattr(
                forward_batch.token_to_kv_pool, "sparse_coordinator", None
            )
            attention_offloaded = bool(
                coordinator is not None
                and coordinator.is_attention_offloaded(self.layer_id, atten_id)
            )
            # Packed mode9 writes its complete row to the backing pool. The
            # coordinator patches selection KV from that row; only split BF16
            # offload needs the Fluent Prolog's extra current-KV outputs.
            output_cur_k_nope_rope = attention_offloaded and not self._npu_dsa_kv_int8_enabled()
            if self.can_overlap_indexer_prolog(forward_batch, atten_id, gather_qkv, extra_input):
                (
                    topk_indices,
                    s,
                    skip_gather_q,
                    sfad_packed_query,
                ) = self.forward_indexer_mla_prolog(
                    rotary_emb,
                    positions,
                    hidden_states,
                    kv_cache,
                    forward_batch,
                    atten_id,
                    output_cur_k_nope_rope=output_cur_k_nope_rope,
                )
                forward_batch.topk_indices = topk_indices
                q_nope, q_rope, kv_cache, kr_cache, forward_batch, q_lora, cur_k_nope, cur_k_rope = s
            else:
                s = self.forword_absorb_prepare(
                    rotary_emb,
                    positions,
                    hidden_states,
                    kv_cache,
                    forward_batch,
                    output_cur_k_nope_rope=output_cur_k_nope_rope,
                    atten_id=atten_id,
                )
                q_nope, q_rope, kv_cache, kr_cache, forward_batch, q_lora, cur_k_nope, cur_k_rope = s
                topk_indices = None
                if self.enable_nsa:
                    topk_indices = self.indexer_fn(positions, hidden_states, forward_batch, gather_qkv, extra_input, atten_id, q_lora=q_lora)
            if coordinator is not None:
                if atten_id == 0:
                    # SFA0 selects; prefetch the offloaded SFA1 KV it chose.
                    if coordinator.is_attention_offloaded(self.layer_id, 1):
                        coordinator.decode_load_kv_h2d_async(
                            forward_batch, topk_indices, self.layer_id, 1
                        )
                if attention_offloaded:
                    # Packed current KV is read back from the native Prolog's
                    # backing-pool write. Its query output carries that write's
                    # GE dependency, including the overlapped/rank-major route.
                    current_kv_kwargs = (
                        {"prolog_dependency": q_nope}
                        if getattr(coordinator, "packed_kv", False)
                        else {}
                    )
                    coordinator.decode_update_cur_kv_to_selection(
                        forward_batch,
                        cur_k_nope,
                        cur_k_rope,
                        self.layer_id,
                        atten_id,
                        **current_kv_kwargs,
                    )
                    if (
                        global_server_args_dict.get(
                            "offload_cache_update_timing", "before_attention"
                        )
                        == "before_attention"
                    ):
                        coordinator.decode_update_kv_lru_cache_async(
                            forward_batch,
                            topk_indices,
                            self.layer_id,
                            atten_id,
                        )

            output = self.forward_absorb_core(
                atten_id,
                q_nope,
                q_rope,
                kv_cache,
                kr_cache,
                forward_batch,
                return_ctrl=return_ctrl,
                topk_indices=topk_indices,
                skip_gather_q=skip_gather_q,
                sfad_packed_query=sfad_packed_query,
            )
            if (
                attention_offloaded
                and coordinator is not None
                and global_server_args_dict.get(
                    "offload_cache_update_timing", "before_attention"
                )
                == "after_attention"
            ):
                attention_output = output[0] if isinstance(output, tuple) else output
                coordinator.decode_update_kv_lru_cache_async(
                    forward_batch,
                    topk_indices,
                    self.layer_id,
                    atten_id,
                    update_after=attention_output,
                )
                coordinator.clear_temp_tensors(atten_id)
            return output
        else:
            if forward_batch.forward_mode.is_idle():
                return hidden_states
            use_local_packed_c8_prefill = extra_input is None and self._can_use_local_packed_c8_prefill(hidden_states, forward_batch)
            if use_local_packed_c8_prefill:
                # Preserve the original Indexer input exactly. The fused V3
                # q_lora differs from q_a_proj + RMSNorm in a very small
                # number of BF16 values because of fusion rounding.
                use_bf16_prefill_staging = forward_batch.attn_metadata.num_decode_tokens == 0
                if use_bf16_prefill_staging:
                    self._validate_packed_prefill_sfap()
                q_lora, latent_cache, _ = self.prefill_prepare_attn(
                    hidden_states,
                    forward_batch,
                    gather_qkv,
                    None,
                    skip_latent_cache=not use_bf16_prefill_staging,
                    overlap_latent_cache_gather=(
                        use_bf16_prefill_staging
                        and atten_id == 1
                        and self.cli_factor == 2
                        and self._can_use_native_sfad_prefill(kv_cache, forward_batch)
                        and self._npu_dsa_kv_int8_enabled()
                    ),
                )
                packed_kv_pool = kv_cache[0] if isinstance(kv_cache, (list, tuple)) else kv_cache
                if use_bf16_prefill_staging:
                    logger.info_once("NPU DSA KV int8 BF16-compute/packed-cache Prefill path is active.")
                    use_native_sfad_prefill = not any("FLASHForCausalLMNextN" in x for x in self.config.architectures)
                    if use_native_sfad_prefill:
                        topk_indices, s = self._prepare_native_sfad_prefill(
                            q_lora,
                            latent_cache,
                            positions,
                            hidden_states,
                            rotary_emb,
                            kv_cache,
                            forward_batch,
                            reduce_type,
                            atten_id,
                            prefill_sfad_prefix_state,
                            prefill_sfad_prefix_ready_event,
                            packed_cache_staging=True,
                        )
                    else:
                        topk_indices = self.prefill_indexer(
                            q_lora,
                            positions,
                            hidden_states,
                            forward_batch,
                            atten_id,
                        )
                        s = self.forward_normal_chunked_kv_prepare(
                            q_lora,
                            latent_cache,
                            rotary_emb,
                            kv_cache,
                            forward_batch,
                            reduce_type,
                            use_bf16_prefill_staging=True,
                        )
                else:
                    (
                        q_nope,
                        q_pe,
                        _,
                        packed_prefill_kv,
                    ) = self._run_local_packed_c8_prefill(
                        rotary_emb,
                        hidden_states,
                        packed_kv_pool,
                        forward_batch,
                    )
                    topk_indices = self.prefill_indexer(
                        q_lora,
                        positions,
                        hidden_states,
                        forward_batch,
                        atten_id,
                    )
                    s = self.forward_normal_chunked_kv_prepare(
                        q_lora,
                        None,
                        rotary_emb,
                        kv_cache,
                        forward_batch,
                        reduce_type,
                        packed_prolog_output=(q_nope, q_pe),
                        packed_prefill_kv=(packed_prefill_kv if self._is_cold_prefill(forward_batch) else None),
                    )
            else:
                q_lora, latent_cache, hidden_states = self.prefill_prepare_attn(
                    hidden_states,
                    forward_batch,
                    gather_qkv,
                    extra_input,
                    skip_latent_cache=self._npu_dsa_kv_int8_enabled(),
                )
                use_native_sfad_prefill = (
                    extra_input is None and not self._npu_dsa_kv_int8_enabled() and self._can_use_native_sfad_prefill(kv_cache, forward_batch)
                )
                if use_native_sfad_prefill:
                    logger.info_once("NPU DSA KV BF16 Native SFAD Prefill path is active.")
                    topk_indices, s = self._prepare_native_sfad_prefill(
                        q_lora,
                        latent_cache,
                        positions,
                        hidden_states,
                        rotary_emb,
                        kv_cache,
                        forward_batch,
                        reduce_type,
                        atten_id,
                        prefill_sfad_prefix_state,
                        prefill_sfad_prefix_ready_event,
                        packed_cache_staging=False,
                    )
                else:
                    topk_indices = None
                    if self.enable_nsa:
                        topk_indices = self.prefill_indexer(
                            q_lora,
                            positions,
                            hidden_states,
                            forward_batch,
                            atten_id,
                        )
                    if self._npu_dsa_kv_int8_enabled():
                        s = self.forward_normal_chunked_kv_prepare(
                            q_lora,
                            latent_cache,
                            rotary_emb,
                            kv_cache,
                            forward_batch,
                            reduce_type,
                            hidden_states=hidden_states,
                        )
                    else:
                        s = self.forward_normal_chunked_kv_prepare(
                            q_lora,
                            latent_cache,
                            rotary_emb,
                            kv_cache,
                            forward_batch,
                            reduce_type,
                        )
            if prefill_core_ready_event is not None:
                # All Prefill preparation and its input collectives are ordered
                # before this event. The caller can make another stream wait on
                # it and safely launch communication alongside the attention
                # core without placing that communication ahead of Attention's
                # own collectives on a shared HCCL communicator.
                prefill_core_ready_event.record(stream=torch.npu.current_stream())
            if prefill_before_core_callback is not None:
                # Submit side-stream work at this exact host-side boundary. A
                # callback after self.forward() returns would be too late: the
                # attention core and its output collective would already have
                # been submitted ahead of that work.
                prefill_before_core_callback()
            # CLI=2 shares TopK within this transformer only. Keep routing
            # addresses across the pair, but always fetch each attention's KV.
            # Explicit pair lifetime avoids tensor-address caches across batches.
            uses_selected_kv = len(s[0]) >= 2 and isinstance(s[0][1], SelectedKvPrefillState)
            selected_pair = uses_selected_kv and self.cli_factor == 2
            route_cache = None
            if uses_selected_kv:
                pair = getattr(forward_batch, "_selected_kv_route_pair", None)
                if pair is None or pair[0] != self.layer_id or pair[1] is not topk_indices:
                    raise RuntimeError("Selected-KV CLI pair must share the same TopK")
                route_cache = pair[2]
                if selected_pair and atten_id == 0 and prefill_selected_next_pool is not None:
                    route_cache["next_pool"] = prefill_selected_next_pool
            try:
                output = self.forward_normal_chunked_kv_core(*s, topk_indices=topk_indices, selected_route_cache=route_cache)
            finally:
                if (selected_pair and atten_id == 1) or (uses_selected_kv and not selected_pair):
                    del forward_batch._selected_kv_route_pair
            if return_ctrl:
                # prefill has no ctrl tensor to return
                return output, None
            return output

    def indexer_fn(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        gather_qkv: bool = False,
        extra_input: Any = None,
        atten_id: int = 0,
        q_lora: Optional[torch.Tensor] = None,
    ):
        if "FLASHForCausalLMNextN" in self.config.architectures[0]:
            # hack for MTP3H
            if global_server_args_dict["is_multi_head_eagle"]:
                if forward_batch.topk_indices is None:
                    topk_indices = self.get_topk_indcies(
                        positions, hidden_states, forward_batch, self.layer_id, gather_qkv, extra_input, q_lora=q_lora
                    )
                    forward_batch.topk_indices = topk_indices
                else:
                    topk_indices = forward_batch.topk_indices
            else:
                topk_indices = self.get_topk_indcies(positions, hidden_states, forward_batch, self.layer_id, gather_qkv, extra_input, q_lora=q_lora)
        else:
            # add for cli. two ds-atten
            layer_id = self.layer_id * 2 if not atten_id else self.layer_id * 2 + 1
            if self.cli_factor <= 1 or layer_id % self.cli_factor == 0:
                topk_indices = self.get_topk_indcies(positions, hidden_states, forward_batch, layer_id, gather_qkv, extra_input, q_lora=q_lora)
            else:
                topk_indices = forward_batch.topk_indices
            if self.cli_factor > 1:
                forward_batch.topk_indices = topk_indices
        return topk_indices

    def get_topk_indcies(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        layer_id: int,
        gather_qkv: bool = False,
        extra_input: Any = None,
        q_lora: Optional[torch.Tensor] = None,
    ):
        if q_lora is not None:
            # q_lora provided externally (e.g., from npu_mla_prolog_v3 output),
            # skip redundant q_a_proj + q_a_layernorm computation
            if gather_qkv:
                hidden_states = get_attn_tp_group().gather_tensor(hidden_states, forward_batch.global_sp_num_tokens)
                q_lora = get_attn_tp_group().gather_tensor(q_lora, forward_batch.global_sp_num_tokens)
        elif extra_input is not None:
            q_lora, _ = extra_input
            global_sp_num_tokens = forward_batch.global_sp_num_tokens
            attn_sp_token_nums = get_attn_tp_group().get_local_sp_token_num(global_sp_num_tokens)
            hidden_states = get_attn_tp_group().all_gather(hidden_states, dim=0, output_split_sizes=attn_sp_token_nums)
        else:
            q_lora = self.q_a_proj(hidden_states)[0]
            if gather_qkv:
                hidden_states = get_attn_tp_group().gather_tensor(hidden_states, forward_batch.global_sp_num_tokens)
                q_lora = get_attn_tp_group().gather_tensor(q_lora, forward_batch.global_sp_num_tokens)
            q_lora = self.q_a_layernorm(q_lora)
        if not forward_batch.all_decode_or_idle:
            attn_metadata = forward_batch.attn_metadata
            num_prefill_tokens = attn_metadata.num_prefill_tokens
            num_decode_tokens = attn_metadata.num_decode_tokens
            no_padding_input_len = num_prefill_tokens + num_decode_tokens
            hidden_states = hidden_states[:no_padding_input_len]
            q_lora = q_lora[:no_padding_input_len]

        topk_indices = self.indexer(
            x=hidden_states,
            q_lora=q_lora,
            positions=positions,
            forward_batch=forward_batch,
            layer_id=layer_id,  # Use stored layer_id if available
        )

        return topk_indices

    def process_weights_after_loading(self, not_used: torch.nn.Module):

        def get_layer_weight(layer):
            WEIGHT_NAMES = ("weight", "qweight", "weight_packed")
            for attr in WEIGHT_NAMES:
                if hasattr(layer, attr):
                    return getattr(layer, attr)
            raise AttributeError(f"Layer '{layer}' has no recognized weight attribute: {WEIGHT_NAMES}.")

        def get_and_maybe_dequant_weights(layer: LinearBase):
            if not isinstance(layer.quant_method, UnquantizedLinearMethod):
                # NOTE: This should only be used offline, since it's O(N^3)
                act_dtype = self.quant_config.get_supported_act_dtypes()[0] if self.quant_config else torch.float16
                eye = torch.eye(layer.input_size_per_partition, dtype=act_dtype, device=get_layer_weight(layer).device)
                dequant_weights = layer.quant_method.apply(layer, eye, bias=None)
                del eye
                # standardize to (output, input)
                return dequant_weights.T
            return layer.weight

        # we currently do not have quantized bmm's which are needed for
        # `W_UV` and `W_UK_T`, we we just store fp16/bf16 copies and perform
        # the bmm's in 16-bit, the extra memory overhead of this is fairly low
        num_heads = self.num_local_heads
        if self.npu_disable_dsa_head_parallel:
            num_heads = self.num_heads
        kv_b_proj_weight = get_and_maybe_dequant_weights(self.kv_b_proj).T
        assert kv_b_proj_weight.shape == (self.kv_lora_rank, num_heads * (self.qk_nope_head_dim + self.v_head_dim)), (
            f"{kv_b_proj_weight.shape=}, {self.kv_lora_rank=}, {num_heads=}, {self.qk_nope_head_dim=}, {self.v_head_dim=}"
        )
        kv_b_proj_weight = kv_b_proj_weight.view(
            self.kv_lora_rank,
            num_heads,
            self.qk_nope_head_dim + self.v_head_dim,
        )

        W_UK, W_UV = kv_b_proj_weight.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        # Convert from (L, N, V) to (N, L, V)
        W_UV = W_UV.transpose(0, 1).contiguous()
        # Convert from (L, N, P) to (N, P, L)
        W_UK_T = W_UK.permute(1, 2, 0).contiguous()

        # tensor here need to be capsulated adapting for graph mode
        # Capsulated as parameter may cause additional weight update in p2p
        if not hasattr(self.kv_b_proj, "W_UV"):
            self.kv_b_proj.register_parameter("W_UV", torch.nn.Parameter(W_UV))
        else:
            self.kv_b_proj.W_UV.data = W_UV
        if not hasattr(self.kv_b_proj, "W_UK_T"):
            self.kv_b_proj.register_parameter("W_UK_T", torch.nn.Parameter(W_UK_T))
        else:
            self.kv_b_proj.W_UK_T.data = W_UK_T
        self.kv_a_layernorm_absorb_scale = copy.deepcopy(self.kv_a_layernorm)
        self.kv_a_layernorm_absorb_scale.weight.data = self.kv_a_layernorm.weight.data * self.mla_scale_kv_lora

    def prefetch_full(self, dependency):
        torch_npu.npu_prefetch(
            self.kv_a_proj_with_mqa.weight, dependency, self.kv_a_proj_with_mqa.weight.numel() * self.kv_a_proj_with_mqa.weight.element_size()
        )
        torch_npu.npu_prefetch(self.q_a_proj.weight, dependency, self.q_a_proj.weight.numel() * self.q_a_proj.weight.element_size())
        torch_npu.npu_prefetch(self.q_b_proj.weight, dependency, self.q_b_proj.weight.numel() * self.q_b_proj.weight.element_size())

    def prefetch_half(self, dependency):
        torch_npu.npu_prefetch(self.q_a_proj.weight, dependency, self.q_a_proj.weight.numel() * self.q_a_proj.weight.element_size())
        torch_npu.npu_prefetch(self.q_b_proj.weight, dependency, self.q_b_proj.weight.numel() * self.q_b_proj.weight.element_size() // 2)

    def prefill_prepare_attn(
        self,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        gather_qkv: bool = False,
        extra_input: Any = None,
        skip_latent_cache: bool = False,
        overlap_latent_cache_gather: bool = False,
    ):
        if extra_input is not None:
            q, latent_cache = extra_input
            if self.npu_disable_dsa_head_parallel:
                slice_start, slice_end = get_attn_sp_token_slice_pos(forward_batch)
                q = q[slice_start:slice_end]
                hidden_states = hidden_states[slice_start:slice_end]
            else:
                hidden_states = get_attn_tp_group().gather_tensor(hidden_states, forward_batch.global_sp_num_tokens)
        else:
            q = self.q_a_proj(hidden_states)[0]
            latent_cache = None if skip_latent_cache else self.kv_a_proj_with_mqa(hidden_states)[0]
            if gather_qkv:
                if not self.npu_disable_dsa_head_parallel:
                    q = get_attn_tp_group().gather_tensor(q, forward_batch.global_sp_num_tokens)
                    hidden_states = get_attn_tp_group().gather_tensor(hidden_states, forward_batch.global_sp_num_tokens)
                if latent_cache is not None:
                    if overlap_latent_cache_gather:
                        latent_cache = _gather_prefill_latent_cache(latent_cache, forward_batch.global_sp_num_tokens)
                    else:
                        latent_cache = get_attn_tp_group().gather_tensor(latent_cache, forward_batch.global_sp_num_tokens)
        q = self.q_a_layernorm(q).contiguous()
        return q, latent_cache, hidden_states

    def _should_update_prefill_topk(self, atten_id: int) -> bool:
        layer_id = self.layer_id * 2 if not atten_id else self.layer_id * 2 + 1
        return self.cli_factor <= 1 or layer_id % self.cli_factor == 0

    def prefill_indexer(
        self,
        q_lora,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        atten_id: int = 0,
        after_input_redistribute_callback=None,
        between_input_and_core_callback=None,
        before_output_redistribute_callback=None,
    ):
        if "FLASHForCausalLMNextN" in self.config.architectures[0]:
            # hack for MTP3H
            if global_server_args_dict["is_multi_head_eagle"]:
                if forward_batch.topk_indices is None:
                    topk_indices = self.prefill_get_topk_indcies(
                        q_lora,
                        positions,
                        hidden_states,
                        forward_batch,
                        self.layer_id,
                        after_input_redistribute_callback,
                        between_input_and_core_callback,
                        before_output_redistribute_callback,
                    )
                    forward_batch.topk_indices = topk_indices
                else:
                    topk_indices = forward_batch.topk_indices
            else:
                topk_indices = self.prefill_get_topk_indcies(
                    q_lora,
                    positions,
                    hidden_states,
                    forward_batch,
                    self.layer_id,
                    after_input_redistribute_callback,
                    between_input_and_core_callback,
                    before_output_redistribute_callback,
                )
        else:
            # add for cli. two ds-atten
            layer_id = self.layer_id * 2 if not atten_id else self.layer_id * 2 + 1
            if self._should_update_prefill_topk(atten_id):
                topk_indices = self.prefill_get_topk_indcies(
                    q_lora,
                    positions,
                    hidden_states,
                    forward_batch,
                    layer_id,
                    after_input_redistribute_callback,
                    between_input_and_core_callback,
                    before_output_redistribute_callback,
                )
            else:
                topk_indices = forward_batch.topk_indices
            if self.cli_factor > 1:
                forward_batch.topk_indices = topk_indices
        return topk_indices

    def prefill_get_topk_indcies(
        self,
        q_lora,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        layer_id: int,
        after_input_redistribute_callback=None,
        between_input_and_core_callback=None,
        before_output_redistribute_callback=None,
    ):

        topk_indices = self.indexer(
            x=hidden_states,
            q_lora=q_lora,
            positions=positions,
            forward_batch=forward_batch,
            layer_id=layer_id,  # Use stored layer_id if available
            prefill_after_input_redistribute_callback=(after_input_redistribute_callback),
            prefill_between_input_and_core_callback=(between_input_and_core_callback),
            prefill_before_output_redistribute_callback=(before_output_redistribute_callback),
        )

        return topk_indices
