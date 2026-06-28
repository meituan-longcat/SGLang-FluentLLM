"""Attention layer."""
from typing import Any, Dict, List, Optional, Generic, TypeVar, Set
from dataclasses import dataclass, fields
from abc import ABC, abstractmethod
from enum import Enum, auto

import psutil
import torch
import torch.nn as nn
import torchair as tng
from torch.nn.parameter import Parameter
from sglang.srt.layers.dense.layouts.unquant import UnquantizedLinearMethod
from sglang.srt.layers.linear import ColumnParallelLinear, LinearBase

from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.env import ENV, global_server_args_dict

from sglang.srt.utils import get_colorful_logger, is_npu

logger = get_colorful_logger(__name__)

__is_npu__ = is_npu()
if __is_npu__:
    import torch_npu

GiB_bytes = 1 << 30

def get_cpu_memory() -> int:
    """Returns the total CPU memory of the node in bytes."""
    return psutil.virtual_memory().total

def is_quantized_kv_cache(kv_cache_dtype: str) -> bool:
    return kv_cache_dtype != "auto"

class CacheConfig:
    """Configuration for the KV cache.

    Args:
        block_size: Size of a cache block in number of tokens.
        gpu_memory_utilization: Fraction of GPU memory to use for the
            vLLM execution.
        swap_space: Size of the CPU swap space per GPU (in GiB).
        cache_dtype: Data type for kv cache storage.
        num_gpu_blocks_override: Number of GPU blocks to use. This overrides the
            profiled num_gpu_blocks if specified. Does nothing if None.
    """

    def __init__(
        self,
        block_size: int,
        gpu_memory_utilization: float,
        swap_space: float,
        cache_dtype: str,
        is_attention_free: bool = False,
        num_gpu_blocks_override: Optional[int] = None,
        sliding_window: Optional[int] = None,
        enable_prefix_caching: bool = False,
        cpu_offload_gb: float = 0,
    ) -> None:
        self.block_size = block_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.swap_space_bytes = swap_space * GiB_bytes
        self.num_gpu_blocks_override = num_gpu_blocks_override
        self.cache_dtype = cache_dtype
        self.is_attention_free = is_attention_free
        self.sliding_window = sliding_window
        self.enable_prefix_caching = enable_prefix_caching
        self.cpu_offload_gb = cpu_offload_gb
        self._verify_args()
        self._verify_cache_dtype()
        self._verify_prefix_caching()

        # Will be set after profiling.
        self.num_gpu_blocks = None
        self.num_cpu_blocks = None
    """
    def compute_hash(self) -> str:
        factors: List[Any] = []
        factors.append(self.block_size)
        factors.append(self.cache_dtype)

        hash_str = hashlib.md5(str(factors).encode()).hexdigest()
        return hash_str
    """
    def metrics_info(self):
        # convert cache_config to dict(key: str, value: str) for prometheus
        # metrics info
        return {key: str(value) for key, value in self.__dict__.items()}

    def _verify_args(self) -> None:
        if self.gpu_memory_utilization > 1.0:
            raise ValueError(
                "GPU memory utilization must be less than 1.0. Got "
                f"{self.gpu_memory_utilization}.")

    def _verify_cache_dtype(self) -> None:
        if self.cache_dtype == "auto":
            pass
        elif self.cache_dtype in ("fp8", "fp8_e4m3", "fp8_e5m2"):
            logger.info(
                "Using fp8 data type to store kv cache. It reduces the GPU "
                "memory footprint and boosts the performance. "
                "Meanwhile, it may cause accuracy drop without a proper "
                "scaling factor")
        else:
            raise ValueError(f"Unknown kv cache dtype: {self.cache_dtype}")

    def _verify_prefix_caching(self) -> None:
        if not self.enable_prefix_caching:
            return

        if self.sliding_window is not None:
            raise NotImplementedError(
                "Prefix caching is not supported with sliding window. "
                "Run with --disable-sliding-window to use prefix caching.")
    """
    def verify_with_parallel_config(
        self,
        parallel_config: "ParallelConfig",
    ) -> None:
        total_cpu_memory = get_cpu_memory()
        # FIXME(woosuk): Here, it is assumed that the GPUs in a tensor parallel
        # group are in the same node. However, the GPUs may span multiple nodes.
        num_gpus_per_node = parallel_config.tensor_parallel_size
        cpu_memory_usage = self.swap_space_bytes * num_gpus_per_node

        msg = (f"{cpu_memory_usage / GiB_bytes:.2f} GiB out of the "
               f"{total_cpu_memory / GiB_bytes:.2f} GiB total CPU memory "
               "is allocated for the swap space.")
        if cpu_memory_usage > 0.7 * total_cpu_memory:
            raise ValueError("Too large swap space. " + msg)
        elif cpu_memory_usage > 0.4 * total_cpu_memory:
            logger.warning("Possibly too large swap space. %s", msg)
    """

class AttentionType(Enum):
    DECODER = auto()  # Decoder attention between previous layer Q/K/V
    ENCODER = auto()  # Encoder attention between previous layer Q/K/V
    ENCODER_DECODER = auto()  # Attention between dec. Q and enc. K/V

class AttentionMetadata:
    is_profile_run: bool = False
    context_chunk_workspace: Optional[torch.Tensor] = None

    #prefill_metadata: Optional['AttentionMetadata'] = None
    #decode_metadata: Optional['AttentionMetadata'] = None

    num_prefill_tokens: int = 0
    num_decode_tokens: int = 0
    slot_mapping: Optional[torch.Tensor] = None
    actual_seq_lengths: List[int] = None
    actual_seq_lengths_kv: List[int] = None
    kv_index_list: torch.Tensor = None
    kv_block_index_list: torch.Tensor = None

    all_decode_or_idle: bool = False
    #is_idle: bool = False

    block_table: Optional[torch.Tensor] = None

    seq_lens_tensor: Optional[torch.Tensor] = None
    seq_lens: Optional[List[int]] = None
    max_seq_len: Optional[int] = 0

    context_lens_tensor: Optional[torch.Tensor] = None
    context_lens: Optional[List[int]] = None

    query_len_tensor: Optional[torch.Tensor] = None
    query_len: Optional[List[int]] = None
    max_query_len: Optional[int] = 0

    # for kvp
    kvp_block_table: Optional[torch.Tensor] = None
    kvp_context_lens_tensor: Optional[torch.Tensor] = None
    kvp_current_slots_mapping: Optional[torch.Tensor] = None
    # for 3s
    init_cnt_req: Optional[torch.Tensor] = None
    local_cnt_req: Optional[torch.Tensor] = None

    # def __repr__(self) -> str:
    #     return (f"AttentionMetadata("
    #             f"is_profile_run={self.is_profile_run}, "
    #             f"context_chunk_workspace={self.context_chunk_workspace}, "
    #             f"num_prefill_tokens={self.num_prefill_tokens}, "
    #             f"num_decode_tokens={self.num_decode_tokens}, "
    #             f"slot_mapping={self.slot_mapping}, "
    #             f"all_decode_or_idle={self.all_decode_or_idle}, "
    #             f"block_table={self.block_table}, "
    #             f"seq_lens_tensor={self.seq_lens_tensor}, "
    #             f"seq_lens={self.seq_lens}, "
    #             f"max_seq_len={self.max_seq_len}, "
    #             f"context_lens_tensor={self.context_lens_tensor}, "
    #             f"context_lens={self.context_lens}, "
    #             f"query_len_tensor={self.query_len_tensor}, "
    #             f"query_len={self.query_len}, "
    #             f"max_query_len={self.max_query_len}, "
    #             f")")


T = TypeVar("T", bound=AttentionMetadata)


class AttentionImpl(ABC, Generic[T]):

    @abstractmethod
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: Optional[int] = None,
        alibi_slopes: Optional[List[float]] = None,
        sliding_window: Optional[int] = None,
        kv_cache_dtype: str = "auto",
        blocksparse_params: Optional[Dict[str, Any]] = None,
        logits_soft_cap: Optional[float] = None,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: T,
        k_scale: float = 1.0,
        v_scale: float = 1.0,
        attn_type: AttentionType = AttentionType.DECODER,
    ) -> torch.Tensor:
        raise NotImplementedError


class MLAAttentionImpl(AttentionImpl[T], Generic[T]):

    @abstractmethod
    def forward(
        self,
        hidden_states_or_cq: torch.Tensor,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: T,
        k_scale: float,
        v_scale: float,
        attn_type: AttentionType = AttentionType.DECODER,
        output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        raise NotImplementedError


class MLACommonImpl(MLAAttentionImpl[T], Generic[T]):
    """
    NOTE: Please read the comment at the top of the file before trying to
    understand this class
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: Optional[List[float]],
        sliding_window: Optional[int],
        kv_cache_dtype: str,
        blocksparse_params: Optional[Dict[str, Any]],
        logits_soft_cap: Optional[float],
        #attn_type: str,
        # MLA Specific Arguments
        q_lora_rank: Optional[int],
        kv_lora_rank: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        qk_head_dim: int,
        v_head_dim: int,
        kv_b_proj: ColumnParallelLinear,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        self.kv_cache_dtype = kv_cache_dtype

        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_head_dim
        self.v_head_dim = v_head_dim
        self.kv_b_proj = kv_b_proj


    def _flash_attn_varlen_diff_headdims(self, q, k, v, softmax_scale, attn_metadata: AttentionMetadata):
        attn_output = torch_npu.mlp_prefill_flash_attention(
            q, # [BS, Nq, qkH]
            k, # [BS, Nkv, qkH]
            None, # k_scale
            v, # [BS, Nkv, vH]
            None, # v_scale
            None, # block_nums_prefix
            attn_metadata.query_len_tensor, #seqlens_q_tensor
            attn_metadata.seq_lens_tensor, #seqlens_kv_tensor
            attn_metadata.max_query_len, # max_seqlen_q
            attn_metadata.max_seq_len, # max_seqlen_k
            len(attn_metadata.seq_lens), # batch_size
            self.num_heads, # q_head_num
            self.num_heads, # kv_head_dnum
            self.qk_head_dim,
            self.v_head_dim,
            softmax_scale,
            [-1, -1],
            3, # kv_cache_type 3 for continuous kvcache
            0, # block_size
            0, # layer_offset
            0, # scale_layer_offset
            attn_metadata.query_len, # actual_seq_qlen
            attn_metadata.seq_lens, # actual_seq_kvlen
            True # causal
            )
        attn_output = attn_output.reshape(-1, self.num_heads, self.v_head_dim)
        return attn_output

    def _v_up_proj(self, x, transposed=False):
        # Convert from (B, N, L) to (N, B, L)
        if not transposed:
            x = x.view(-1, self.num_heads, self.kv_lora_rank).transpose(0, 1)
        # Multiply (N, B, L) x (N, L, V) -> (N, B, V)
        x = torch.bmm(x, self.kv_b_proj.W_UV)
        # Convert from (N, B, V) to (B, N * V)
        return x.transpose(0, 1)

    def process_weights_after_loading(self, not_used: torch.nn.Module):

        def get_layer_weight(layer):
            WEIGHT_NAMES = ("weight", "qweight", "weight_packed")
            for attr in WEIGHT_NAMES:
                if hasattr(layer, attr):
                    return getattr(layer, attr)
            raise AttributeError(
                f"Layer '{layer}' has no recognized weight attribute:"
                f" {WEIGHT_NAMES}.")

        def get_and_maybe_dequant_weights(layer: LinearBase):
            if not isinstance(layer.quant_method, UnquantizedLinearMethod):
                # NOTE: This should only be used offline, since it's O(N^3)
                eye = torch.eye(layer.input_size_per_partition,
                                dtype=act_dtype,
                                device=get_layer_weight(layer).device)
                dequant_weights = layer.quant_method.apply(layer,
                                                           eye,
                                                           bias=None)
                del eye
                # standardize to (output, input)
                return dequant_weights.T
            return layer.weight

        # we currently do not have quantized bmm's which are needed for
        # `W_UV` and `W_UK_T`, we we just store fp16/bf16 copies and perform
        # the bmm's in 16-bit, the extra memory overhead of this is fairly low
        kv_b_proj_weight = get_and_maybe_dequant_weights(self.kv_b_proj).T
        assert kv_b_proj_weight.shape == (
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim)), (
                f"{kv_b_proj_weight.shape=}, "
                f"{self.kv_lora_rank=}, "
                f"{self.num_heads=}, "
                f"{self.qk_nope_head_dim=}, "
                f"{self.v_head_dim=}")
        kv_b_proj_weight = kv_b_proj_weight.view(
            self.kv_lora_rank,
            self.num_heads,
            self.qk_nope_head_dim + self.v_head_dim,
        )

        W_UK, W_UV = kv_b_proj_weight.split(
            [self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        # Convert from (L, N, V) to (N, L, V)
        W_UV = W_UV.transpose(0, 1).contiguous()
        # Convert from (L, N, P) to (N, P, L)
        W_UK_T = W_UK.permute(1, 2, 0).contiguous()

        # tensor here need to be capsulated adapting for graph mode
        # Capsulated as parameter may cause additional weight update in p2p
        if not hasattr(self.kv_b_proj, 'W_UV'):
            self.kv_b_proj.register_parameter("W_UV", Parameter(W_UV))
        else:
            self.kv_b_proj.W_UV.data = W_UV
        if not hasattr(self.kv_b_proj, 'W_UK_T'):
            self.kv_b_proj.register_parameter("W_UK_T", Parameter(W_UK_T))
        else:
            self.kv_b_proj.W_UK_T.data = W_UK_T


    def _forward_prefill(
        self,
        q: torch.Tensor,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:

        prefill_metadata = attn_metadata
        assert prefill_metadata is not None

        has_context = prefill_metadata.context_lens_tensor is not None \
            and max(prefill_metadata.context_lens) > 0

        kv_nope = self.kv_b_proj(kv_c_normed)[0].view(\
            -1, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
        k_nope, v = kv_nope\
            .split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        k = torch.cat((k_nope, k_pe.expand((*k_nope.shape[:-1], -1))), dim=-1)

        if has_context:
            # recompute kv and concat
            chunked_prefill_block_num = prefill_metadata.chunked_prefill_block_num
            chunked_prefill_remained_token_num = prefill_metadata.chunked_prefill_remained_token_num

            if not ENV.npu_enable_mla_split_kv_kr:
                kv_c_and_k_pe_cache_list = [kv_c_and_k_pe_cache[idx] for idx in prefill_metadata.block_tables[-1][:chunked_prefill_block_num]]
                kv_c_and_k_pe_cache_list.append(kv_c_and_k_pe_cache[prefill_metadata.block_tables[-1][chunked_prefill_block_num]][:chunked_prefill_remained_token_num])
                kv_c_and_k_pe_cache = torch.cat(kv_c_and_k_pe_cache_list, dim=0)

                k_c_cache = kv_c_and_k_pe_cache[..., :self.kv_lora_rank]
                k_pe_cache = kv_c_and_k_pe_cache[..., self.kv_lora_rank:].unsqueeze(1)
            else:
                kv_c_and_k_pe_cache_list = [kv_c_and_k_pe_cache[0][idx] for idx in prefill_metadata.block_tables[-1][:chunked_prefill_block_num]]
                kv_c_and_k_pe_cache_list.append(kv_c_and_k_pe_cache[0][prefill_metadata.block_tables[-1][chunked_prefill_block_num]][:chunked_prefill_remained_token_num])
                k_c_cache = torch.cat(kv_c_and_k_pe_cache_list, dim=0)

                kv_c_and_k_pe_cache_list = [kv_c_and_k_pe_cache[1][idx] for idx in prefill_metadata.block_tables[-1][:chunked_prefill_block_num]]
                kv_c_and_k_pe_cache_list.append(kv_c_and_k_pe_cache[1][prefill_metadata.block_tables[-1][chunked_prefill_block_num]][:chunked_prefill_remained_token_num])
                k_pe_cache = torch.cat(kv_c_and_k_pe_cache_list, dim=0).unsqueeze(1)


            kv_nope_cache = self.kv_b_proj(k_c_cache)[0].view(\
                -1, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
            k_nope_cache, v_cache = kv_nope_cache\
                .split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)
            k_pe_cache = k_pe_cache.view(-1, self.num_kv_heads, self.qk_rope_head_dim)
            k_cache = torch.cat((k_nope_cache, k_pe_cache.expand((*k_nope_cache.shape[:-1], -1))), dim=-1)

            k = torch.cat([k[:prefill_metadata.query_start_loc[-2]], k_cache, k[prefill_metadata.query_start_loc[-2]:]], dim=0)
            v = torch.cat([v[:prefill_metadata.query_start_loc[-2]], v_cache, v[prefill_metadata.query_start_loc[-2]:]], dim=0)

        output = self._flash_attn_varlen_diff_headdims(
            q=q,
            k=k,
            v=v,
            softmax_scale=self.scale,
            attn_metadata=prefill_metadata,
        )

        return output.flatten(start_dim=-2)

    @abstractmethod
    def _forward_decode(
        self,
        ql_nope: torch.Tensor,
        q_pe: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: T,
    ) -> torch.Tensor:
        raise NotImplementedError

    def forward(
        self,
        q: torch.Tensor,  # query in unified attn
        k_c_normed: torch.Tensor,  # key in unified attn
        k_pe: torch.Tensor,  # value in unified attn
        kv_cache: torch.Tensor,
        attn_metadata: T,
        k_scale: float,
        v_scale: float,
        attn_type: AttentionType = AttentionType.DECODER,
        output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if output is not None:
            raise NotImplementedError(
                "output is not yet supported for MLAImplBase")
        if attn_metadata.is_profile_run and \
            attn_metadata.context_chunk_workspace is not None:
            # During the profile run try to simulate to worse case output size
            # for `self.kv_b_proj(kv_c_normed)` in `_compute_prefill_context`
            # since this can be large
            _ = torch.empty(
                (attn_metadata.context_chunk_workspace.shape[0],
                 self.num_heads, self.qk_nope_head_dim + self.v_head_dim),
                device=k_c_normed.device,
                dtype=k_c_normed.dtype,
            )

        # TODO: support other forward mode
        has_decode = attn_metadata.num_decode_tokens != 0
        has_prefill = attn_metadata.num_prefill_tokens != 0

        num_prefill_tokens: int = attn_metadata.num_prefill_tokens
        q = q.view(-1, self.num_heads, self.qk_head_dim)

        decode_q = q[num_prefill_tokens:]

        prefill_q = q[:num_prefill_tokens]
        prefill_k_pe = k_pe[:num_prefill_tokens]
        prefill_k_c_normed = k_c_normed[:num_prefill_tokens]

        # write the latent and rope to kv cache
        if kv_cache is not None and ((isinstance(kv_cache, torch.Tensor) and kv_cache.numel() > 0) or (isinstance(kv_cache, list) and kv_cache[0].numel() > 0)):
            if __is_npu__:
                if not ENV.npu_enable_mla_split_kv_kr:
                    kv = torch.concat((k_c_normed, k_pe.squeeze(1)), dim = -1)
                    torch_npu.npu_scatter_nd_update_(kv_cache.reshape(-1, kv_cache.shape[-1]), attn_metadata.slot_mapping.reshape(-1, 1), kv.reshape(-1, kv.shape[-1]))
                else:
                    k_pe = k_pe.squeeze(1)
                    torch_npu.npu_scatter_nd_update_(kv_cache[0].reshape(-1, kv_cache[0].shape[-1]), attn_metadata.slot_mapping.reshape(-1, 1), k_c_normed.reshape(-1, k_c_normed.shape[-1]))
                    torch_npu.npu_scatter_nd_update_(kv_cache[1].reshape(-1, kv_cache[1].shape[-1]), attn_metadata.slot_mapping.reshape(-1, 1), k_pe.reshape(-1, k_pe.shape[-1]))

        output = torch.empty(attn_metadata.num_prefill_tokens +
                             attn_metadata.num_decode_tokens,
                             self.v_head_dim * self.num_heads,
                             device=q.device,
                             dtype=q.dtype)
        if has_prefill:
            output[:num_prefill_tokens] = self._forward_prefill(
                prefill_q, prefill_k_c_normed, prefill_k_pe, kv_cache,
                attn_metadata)

        if has_decode:
            decode_q_nope, decode_q_pe = decode_q.split(
                [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
            # Convert from (B, N, P) to (N, B, P)
            decode_q_nope = decode_q_nope.transpose(0, 1)
            # Multiply (N, B, P) x (N, P, L) -> (N, B, L)
            decode_ql_nope = torch.bmm(decode_q_nope, self.kv_b_proj.W_UK_T)
            # Convert from (N, B, L) to (B, N, L)
            decode_ql_nope = decode_ql_nope.transpose(0, 1)

            output[num_prefill_tokens:] = self._forward_decode(
                decode_ql_nope, decode_q_pe, kv_cache, attn_metadata)

        return output


class TritonMLAImpl(MLACommonImpl[AttentionMetadata]):

    def __init__(
            self,
            num_heads: int,
            head_size: int,
            scale: float,
            num_kv_heads: int,
            alibi_slopes: Optional[List[float]],
            sliding_window: Optional[int],
            kv_cache_dtype: str,
            blocksparse_params: Optional[Dict[str, Any]],
            logits_soft_cap: Optional[float],
            # MLA Specific Arguments
            **mla_args) -> None:
        super().__init__(num_heads, head_size, scale, num_kv_heads,
                         alibi_slopes, sliding_window, kv_cache_dtype,
                         blocksparse_params, logits_soft_cap,
                         **mla_args)

        unsupported_features = [
            alibi_slopes, sliding_window, blocksparse_params, logits_soft_cap
        ]
        if any(unsupported_features):
            raise NotImplementedError(
                "TritonMLAImpl does not support one of the following: "
                "alibi_slopes, sliding_window, blocksparse_params, "
                "logits_soft_cap")

        attn_type: str = AttentionType.DECODER
        if attn_type != AttentionType.DECODER:
            raise NotImplementedError("Encoder self-attention and "
                                      "encoder/decoder cross-attention "
                                      "are not implemented for "
                                      "TritonMLAImpl")

        if is_quantized_kv_cache(self.kv_cache_dtype):
            raise NotImplementedError(
                "TritonMLA with FP8 KV cache not yet supported")

    def mlp_ifa_call(self,
                    q_nope: torch.Tensor,
                    q_pe: torch.Tensor,
                    kv_c_and_k_pe_cache: torch.Tensor,
                    attn_metadata: AttentionMetadata):

        decode_meta = attn_metadata
        assert decode_meta is not None
        torchair_enable = decode_meta.all_decode_or_idle and ENV.npu_enable_graph

        B = q_nope.shape[0]
        if not ENV.npu_enable_mla_split_kv_kr:
            assert kv_c_and_k_pe_cache.numel() > 0
            block_size = kv_c_and_k_pe_cache.shape[1]
            q = torch.cat([q_nope, q_pe], dim=-1)
            if len(kv_c_and_k_pe_cache.size())==3:
                kv_c_and_k_pe_cache = kv_c_and_k_pe_cache.unsqueeze(2)
            q_nope = q
            q_pe = None
            kv_c = kv_c_and_k_pe_cache
            k_pe = None
        else:
            assert kv_c_and_k_pe_cache[0].numel() > 0
            assert kv_c_and_k_pe_cache[1].numel() > 0
            block_size = kv_c_and_k_pe_cache[0].shape[1]
            if len(kv_c_and_k_pe_cache[0].size())==3:
                kv_c_and_k_pe_cache[0] = kv_c_and_k_pe_cache[0].unsqueeze(2)
                kv_c_and_k_pe_cache[1] = kv_c_and_k_pe_cache[1].unsqueeze(2)
            kv_c = kv_c_and_k_pe_cache[0]
            k_pe = kv_c_and_k_pe_cache[1]

        if torchair_enable:
            o = torch_npu.mlp_incre_flash_attention_graph(
                                q_nope, # [q_nope, q_pe]
                                kv_c,
                                None, # k_scale
                                kv_c,
                                None, # v_scale
                                decode_meta.query_len, #_actual_seq_lens,
                                decode_meta.seq_lens,
                                decode_meta.block_tables,
                                B,
                                self.num_heads,
                                self.num_kv_heads,
                                self.kv_lora_rank + self.qk_rope_head_dim,
                                self.kv_lora_rank,
                                self.scale,
                                kv_cache_type = 1, # kv_cache_type 0:fp16 1:bf16 2:int8
                                block_size = block_size,
                                layer_offset = 0, # layer_offset
                                scale_layer_offset = 0, # layer_scale_offset
                                q_rope = q_pe, # None
                                k_rope = k_pe, # None
                                query_length_array=[],
                                context_length_array=[],
                                causal=False,
                                window_size=[-1, -1],
                                mla_flag=True
                                )
        else:
            o = torch_npu.mlp_incre_flash_attention(
                                q_nope,
                                kv_c,
                                None, # k_scale
                                kv_c,
                                None, # v_scale
                                decode_meta.seq_lens_tensor - decode_meta.context_lens_tensor, # q_length_tensor
                                decode_meta.seq_lens_tensor,
                                decode_meta.block_tables,
                                B,
                                self.num_heads,
                                self.num_kv_heads,
                                self.kv_lora_rank + self.qk_rope_head_dim,
                                self.kv_lora_rank,
                                self.scale,
                                kv_cache_type = 1, # kv_cache_type 0:fp16 1:bf16 2:int8
                                block_size = block_size,
                                layer_offset = 0, # layer_offset
                                scale_layer_offset = 0, # layer_scale_offset
                                q_rope = q_pe,
                                k_rope = k_pe,
                                query_length_array = [ a - b for a, b in zip(decode_meta.seq_lens, decode_meta.context_lens)],
                                context_length_array = decode_meta.seq_lens,
                                causal = False,
                                window_size = [-1, -1],
                                mla_flag = True # mla_flag kv同buf
                                )
        return o

    def _forward_decode(
        self,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_mask: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        if not ENV.npu_enable_mla_split_kv_kr:
            assert kv_c_and_k_pe_cache.numel() > 0
        else:
            assert kv_c_and_k_pe_cache[0].numel() > 0
            assert kv_c_and_k_pe_cache[1].numel() > 0

        decode_meta = attn_metadata
        assert decode_meta is not None
        B = q_nope.shape[0]

        if __is_npu__:
            if ENV.npu_enable_mla_split_kv_kr:
                if len(kv_c_and_k_pe_cache[0].size())==4:
                    kv_c_and_k_pe_cache[0] = kv_c_and_k_pe_cache[0].squeeze(dim=2)
                    kv_c_and_k_pe_cache[1] = kv_c_and_k_pe_cache[1].squeeze(dim=2)
                block_size = kv_c_and_k_pe_cache[0].shape[1]
                layout = 'TND' # or TND_NTD, BSND, BSND_NBSD
                if 'BSND' in layout:
                    q_nope = q_nope.unsqueeze(dim=1)
                    q_pe = q_pe.unsqueeze(dim=1)

            torchair_enable = decode_meta.all_decode_or_idle and ENV.npu_enable_graph
            if torchair_enable:
                op_scope = tng.ops
                kwargs = dict(
                    actual_seq_lengths_kv=decode_meta.seq_lens_tensor.to(torch.int64),
                    actual_seq_lengths=decode_meta.query_len_tensor.to(torch.int64),
                )
            else:
                op_scope = torch.ops.npu
                kwargs = dict(
                    actual_seq_lengths_kv=decode_meta.seq_lens_tensor.to(torch.int64),
                    actual_seq_lengths=decode_meta.query_len_tensor.to(torch.int64),
                    # actual_seq_lengths_kv=decode_meta.seq_lens,
                    # actual_seq_lengths=decode_meta.actual_seq_lengths,
                )
            if not global_server_args_dict["npu_disable_kv_nz"]:
                kv_c_and_k_pe_cache[0] = kv_c_and_k_pe_cache[0].unsqueeze(dim=1)
                kv_c_and_k_pe_cache[0] = kv_c_and_k_pe_cache[0].view(kv_c_and_k_pe_cache[0].shape[0], kv_c_and_k_pe_cache[0].shape[1], kv_c_and_k_pe_cache[0].shape[3] // 16, kv_c_and_k_pe_cache[0].shape[2], 16)
                kv_c_and_k_pe_cache[1] = kv_c_and_k_pe_cache[1].unsqueeze(dim=1)
                kv_c_and_k_pe_cache[1] = kv_c_and_k_pe_cache[1].view(kv_c_and_k_pe_cache[1].shape[0], kv_c_and_k_pe_cache[1].shape[1], kv_c_and_k_pe_cache[1].shape[3] // 16, kv_c_and_k_pe_cache[1].shape[2], 16)
            o, _ = op_scope.npu_fused_infer_attention_score(
                    q_nope, kv_c_and_k_pe_cache[0], kv_c_and_k_pe_cache[0], query_rope=q_pe, key_rope=kv_c_and_k_pe_cache[1],
                    dequant_scale1=None,
                    dequant_scale2=None,
                    num_heads= self.num_heads,
                    num_key_value_heads=self.num_kv_heads,
                    input_layout=layout,
                    atten_mask=attn_mask,
                    scale=self.scale,
                    antiquant_mode=0,
                    antiquant_scale=None,
                    block_table=decode_meta.block_table,
                    block_size=block_size,
                    sparse_mode=3,
                    next_tokens=0,
                    **kwargs
                    )
            o = o.squeeze(dim=1)
            attn_out = self._v_up_proj(o, layout in ['TND_NTD', 'BSND_NBSD'])
        return attn_out

class Attention(nn.Module):
    """Attention layer.

    This class takes query, key, and value tensors as input. The input tensors
    can either contain prompt tokens or generation tokens.
    The class does the following:

    1. Store the input key and value tensors in the KV cache.
    2. Perform (multi-head/multi-query/grouped-query) attention.
    3. Return the output tensor.
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: Optional[int] = None,
        alibi_slopes: Optional[List[float]] = None,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        blocksparse_params: Optional[Dict[str, Any]] = None,
        logits_soft_cap: Optional[float] = None,
        prefix: str = "",
        sliding_window: Optional[int] = None,
        mla_flash_attn_version: int = 0,
        **extra_impl_args,
    ) -> None:
        super().__init__()
        if sliding_window:
            self.sliding_window = sliding_window
        elif cache_config is not None:
            self.sliding_window = cache_config.sliding_window
        else:
            self.sliding_window = None

        self.mla_flash_attn_version = mla_flash_attn_version
        if cache_config is not None:
            kv_cache_dtype = cache_config.cache_dtype
            block_size = cache_config.block_size
            is_attention_free = cache_config.is_attention_free
        else:
            kv_cache_dtype = "auto"
            block_size = 16
            is_attention_free = False
        if num_kv_heads is None:
            num_kv_heads = num_heads

        # The default k/v_scale is set to 1.0. This is ignored
        # when kv-cache is not fp8, and should be used with
        # kv-cache in fp8_e5m2. For kv-cache in fp8_e4m3, we
        # expect the pre-quantized k/v_scale to be loaded along
        # with the model weights.
        self.kv_cache_dtype = kv_cache_dtype
        self._k_scale = 1.0
        self._v_scale = 1.0
        # TODO support quant
        """
        quant_method = quant_config.get_quant_method(
            self, prefix=prefix) if quant_config else None
        if quant_method is not None:
            assert isinstance(quant_method, BaseKVCacheMethod)
            # TODO (mgoin): kv cache dtype should be specified in the FP8
            # checkpoint config and become the "auto" behavior
            if self.kv_cache_dtype == "fp8_e5m2":
                raise ValueError("fp8_e5m2 kv-cache is not supported with "
                                 "fp8 checkpoints.")
            # If quantization is enabled, we make "k_scale" and "v_scale"
            # parameters so that it can be loaded from the model checkpoint.
            # The k/v_scale will then be converted back to native float32
            # values after weight loading.
            self.quant_method = quant_method
            self.quant_method.create_weights(self)
        """
        # During model initialization, the default dtype is set as the model
        # weight and activation dtype.
        dtype = torch.get_default_dtype()
        """
        attn_backend = get_attn_backend(head_size, self.sliding_window, dtype,
                                        kv_cache_dtype, block_size,
                                        is_attention_free, blocksparse_params
                                        is not None, mla_flash_attn_version=mla_flash_attn_version)
        """
        self.impl = TritonMLAImpl(num_heads, head_size, scale, num_kv_heads,
                             alibi_slopes, sliding_window, kv_cache_dtype,
                             blocksparse_params, logits_soft_cap, **extra_impl_args)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: Optional[torch.Tensor],
        attn_metadata: AttentionMetadata,
        attn_type: AttentionType = AttentionType.DECODER,
    ) -> torch.Tensor:

        return self.impl.forward(query,
                                 key,
                                 value,
                                 kv_cache,
                                 attn_metadata,
                                 self._k_scale,
                                 self._v_scale,
                                 attn_type=attn_type)

    def apply_kv_cache(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: Optional[torch.Tensor],
        attn_metadata: AttentionMetadata,
    ):
        return self.impl.apply_kv_cache(key,
                                       value,
                                       kv_cache,
                                       attn_metadata,
                                       self._k_scale,
                                       self._v_scale)

    def extra_repr(self) -> str:
        s = f"head_size={self.impl.head_size}"  # type: ignore
        s += f", num_heads={self.impl.num_heads}"  # type: ignore
        s += f", num_kv_heads={self.impl.num_kv_heads}"  # type: ignore
        s += f", scale={self.impl.scale}"  # type: ignore
        s += f", backend={self.impl.__class__.__name__}"
        return s

    def process_weights_after_loading(self, layer: torch.nn.Module):
        if hasattr(self.impl, "process_weights_after_loading"):
            self.impl.process_weights_after_loading(layer)


class DeepseekNSAWithMLA(nn.Module):

    def __init__(
        self,
    ) -> None:
        super().__init__()
        raise RuntimeError("DeepseekNSAWithMLA is not implemented on NPU.")


