from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, Optional

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn
from transformers import PretrainedConfig

from sglang.srt.custom_op import CustomOp
from sglang.srt.utils import add_prefix, align, is_cuda, is_npu

if is_cuda():
    try:
        import deep_gemm_oss
    except ImportError as e:
        deep_gemm_oss = e
from sglang.srt.layers.attention.dsa.utils import NSA_USE_REAL_INDEXER
from sglang.srt.layers.dp_attention import (
    get_attention_tp_group, get_attention_tp_rank, get_attention_tp_size, get_attn_tp_dp_convertor
)
from sglang.srt.layers.utils import (
    CP_METADATA, cp_all_gather_rerange_output,
)
from sglang.srt.layers.linear import ReplicatedLinear
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.rotary_embedding import get_rope_wrapper
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.distributed.decoder_comm_manager import DecoderCommMananger

if TYPE_CHECKING:
    from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool
    from sglang.srt.layers.attention.dsa_backend import DpskSparseAttnBackend

from sglang.srt.utils import get_colorful_logger
logger = get_colorful_logger(__name__)

DUAL_STREAM_TOKEN_THRESHOLD = 1024 if is_cuda() else 0

class BaseIndexerMetadata(ABC):
    @abstractmethod
    def get_seqlens_int32(self) -> torch.Tensor:
        """
        Return: (batch_size,) int32 tensor
        """

    @abstractmethod
    def get_page_table_64(self) -> torch.Tensor:
        """
        Return: (batch_size, num_blocks) int32, page table.
                The page size of the table is 64.
        """

    @abstractmethod
    def get_seqlens_expanded(self) -> torch.Tensor:
        """
        Return: (sum_extend_seq_len,) int32 tensor
        """

    @abstractmethod
    def topk_transform(
        self,
        logits: torch.Tensor,
        topk: int,
        lengths: Optional[torch.Tensor] = None,
        cu_seqlens_q: Optional[torch.Tensor] = None,
        row_starts: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Perform topk selection on the logits and possibly transform the result.

        NOTE that attention backend may override this function to do some
        transformation, which means the result of this topk_transform may not
        be the topk indices of the input logits.

        Return: Anything, since it will be passed to the attention backend
                for further processing on sparse attention computation.
                Don't assume it is the topk indices of the input logits.
        """


def rotate_activation(x: torch.Tensor) -> torch.Tensor:
    assert x.dtype == torch.bfloat16
    if x.shape[0] == 0:
        return x
    from fast_hadamard_transform import hadamard_transform

    hidden_size = x.size(-1)
    assert (
        hidden_size & (hidden_size - 1)
    ) == 0, "Hidden size must be a power of 2 for Hadamard transform."
    return hadamard_transform(x, scale=hidden_size**-0.5)


class V32LayerNorm(nn.Module):
    """
    Layer Normalization.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.bias = nn.Parameter(torch.zeros(dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor):
        return F.layer_norm(
            x.float(), (self.dim,), self.weight, self.bias, self.eps
        ).type_as(x)

def compute_local_lens(
    extend_lens:     torch.Tensor,   # (B,) 当前 token 数
    history_kv_lens: torch.Tensor,   # (B,) 历史 kv 长度
    sp_num_tokens:   torch.Tensor,   # (cp_size,) 每个 rank 的 token 数
    cp_rank:         int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
        local_extend_lens:   (B,) 当前 rank 负责的每个请求的 token 数
        local_total_kv_lens: (B,) 当前 rank 视角下每个请求的总 kv 长度
    """
    # ---- 当前 rank 负责的全局 token 范围 ----
    rank_start = sp_num_tokens[:cp_rank].sum()          # scalar
    rank_end   = rank_start + sp_num_tokens[cp_rank]    # scalar

    # ---- 每个请求在全局 token 序号中的范围 ----
    # req_start[b] = sum(extend_lens[:b])
    # req_end[b]   = sum(extend_lens[:b+1])
    cumsum    = torch.cumsum(extend_lens, dim=0)        # (B,)
    req_end   = cumsum                                  # (B,)
    req_start = cumsum - extend_lens                    # (B,)

    # ---- 与当前 rank 范围取交集 ----
    inter_start = torch.maximum(req_start, rank_start)  # (B,)
    inter_end   = torch.minimum(req_end,   rank_end)    # (B,)

    local_extend_lens = torch.clamp(inter_end - inter_start, min=0)  # (B,)

    # ---- local_total_kv_lens ----
    # 当前 rank 视角下，每个请求的 kv 长度 =
    #   历史 kv 长度
    #   + 当前 rank 之前所有 rank 处理的该请求的 token 数（已写入 kv）
    #   + 当前 rank 负责的该请求的 token 数
    #
    # 即：history_kv_lens + min(req_end, rank_end) - req_start
    #   （从请求开始到当前 rank 末尾，累计写入的 kv 数）
    #   但不超过请求本身的总长度
    tokens_up_to_rank_end = torch.clamp(
        torch.minimum(req_end, rank_end) - req_start,
        min=0
    )                                                   # (B,)

    local_total_kv_lens = history_kv_lens + tokens_up_to_rank_end  # (B,)

    return local_extend_lens, local_total_kv_lens

def get_local_past_key_states(past_key_states, global_cu_kv_lens_cpu, cu_local_total_kv_lens_cpu):
    global_cu_kv_lens_cpu_list = global_cu_kv_lens_cpu.tolist()
    cu_local_total_kv_lens_cpu_list = cu_local_total_kv_lens_cpu.tolist()
    bs = len(global_cu_kv_lens_cpu_list) - 1
    res_list = list()
    for bi in range(bs):
        global_start = global_cu_kv_lens_cpu_list[bi]
        global_end = global_cu_kv_lens_cpu_list[bi+1]
        cur_len = cu_local_total_kv_lens_cpu_list[bi+1] - cu_local_total_kv_lens_cpu_list[bi]
        cur = past_key_states[global_start:global_end][:cur_len]
        res_list.append(cur)
    # TODO: remove contiguous()
    return torch.cat(res_list, dim = 0).contiguous()

def get_non_zero(past_key_states, global_cu_kv_lens_cpu, local_extend_lens, local_total_kv_lens):
    global_cu_kv_lens_cpu_list = global_cu_kv_lens_cpu.tolist()
    local_extend_lens_list = local_extend_lens.tolist()
    local_total_kv_lens_list = local_total_kv_lens.tolist()
    bs = len(local_extend_lens_list)
    res_local_extend_lens_list = list()
    res_local_total_kv_lens_list = list()
    res_list = list()
    for i in range(bs):
        if local_extend_lens_list[i]>0:
            res_local_extend_lens_list.append(local_extend_lens_list[i])
            cur_len = local_total_kv_lens_list[i]
            res_local_total_kv_lens_list.append(cur_len)
            global_start = global_cu_kv_lens_cpu_list[i]
            global_end = global_cu_kv_lens_cpu_list[i+1]
            cur = past_key_states[global_start:global_end][:cur_len]
            res_list.append(cur)

    return torch.cat(res_list, dim = 0).contiguous(), torch.tensor(res_local_extend_lens_list, dtype =torch.int32), torch.tensor(res_local_total_kv_lens_list, dtype =torch.int32)



class Indexer(CustomOp):
    def __init__(
        self,
        hidden_size: int,
        index_n_heads: int,
        index_head_dim: int,
        rope_head_dim: int,
        index_topk: int,
        index_k_norm_type: str,
        q_lora_rank: int,
        max_position_embeddings: int,
        rope_theta: float,
        layer_id: int,
        scale_fmt: Optional[str],
        block_size: int = 128,
        rope_scaling: Optional[Dict[str, Any]] = None,
        is_neox_style: bool = True,
        prefix: str = "",
        config: Optional[PretrainedConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        alt_stream: Optional[torch.cuda.Stream] = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_heads = index_n_heads
        self.head_dim = index_head_dim
        self.rope_head_dim = rope_head_dim
        self.index_topk = index_topk
        self.q_lora_rank = q_lora_rank
        self.layer_id = layer_id
        self.alt_stream = alt_stream

        self.cp_size = get_attention_tp_size()
        self.cp_rank = get_attention_tp_rank()
        if CP_METADATA:
            self.comm_convertor = get_attn_tp_dp_convertor()
            assert self.comm_convertor is not None

        if is_cuda():
            self.sm_count = deep_gemm_oss.get_num_sms()
            self.half_device_sm_count = align(self.sm_count // 2, 8)

        self.wq_b = ReplicatedLinear(
            self.q_lora_rank,
            self.n_heads * self.head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("wq_b", prefix),
        )
        self.wk = ReplicatedLinear(
            self.hidden_size,
            self.head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("wk", prefix),
        )
        self.k_norm = V32LayerNorm(self.head_dim)
        # NOTE: weight_proj is not quantized
        self.weights_proj = ReplicatedLinear(
            self.hidden_size,
            self.n_heads,
            bias=False,
            params_dtype=torch.float32,
            prefix=add_prefix("weights_proj", prefix),
        )
        self.rotary_emb = get_rope_wrapper(
            rope_head_dim,
            rotary_dim=rope_head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,  # type: ignore
            rope_scaling=rope_scaling,
            is_neox_style=is_neox_style,
            device="cuda" if not is_npu() else "npu",
        )
        self.block_size = block_size
        self.scale_fmt = scale_fmt
        self.softmax_scale = self.head_dim**-0.5

        self.kv_block_size = getattr(config, "kv_block_size", 1)
        self.q_block_size = getattr(config, "q_block_size", 1)
        self.num_init_tokens = getattr(config, "index_init_tokens", 0)
        self.num_local_tokens = getattr(config, "index_local_tokens", 0)

    def _forward_fake(
        self,
        x: torch.Tensor,
        q_lora: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        layer_id: int,
    ):
        bs = x.shape[0]
        assert self.index_topk == 2048
        ans = torch.arange(0, self.index_topk, dtype=torch.int32, device=x.device)[
            None, ...
        ].repeat(bs, 1)
        if forward_batch.forward_mode.is_extend():
            assert (
                forward_batch.extend_seq_lens_cpu is not None
                and forward_batch.seq_lens_cpu is not None
            )
            which = 0
            for i, (kv_len, qo_len) in enumerate(
                zip(
                    forward_batch.seq_lens_cpu.tolist(),
                    forward_batch.extend_seq_lens_cpu,
                    strict=True,
                )
            ):
                for j in range(kv_len - qo_len, kv_len):
                    ans[which, j + 1 :] = -1
                    which += 1
            assert which == ans.shape[0]
        else:
            assert forward_batch.seq_lens_cpu is not None
            for i, seq_len in enumerate(forward_batch.seq_lens_cpu.tolist()):
                ans[i, seq_len:] = -1

        return ans

    @torch.compile(dynamic=True)
    def _get_logits_head_gate(self, x: torch.Tensor, q_scale: torch.Tensor):
        weights, _ = self.weights_proj(x.float())
        weights = weights * self.n_heads**-0.5
        weights = weights.unsqueeze(-1) * q_scale * self.softmax_scale
        return weights

    def get_head_gate(self, x: torch.Tensor, out: torch.Tensor):
        torch.matmul(x, self.weights_proj.weight.T, out=out)

    def get_q_k(
        self,
        q_lora: torch.Tensor,
        x: torch.Tensor,
        positions: torch.Tensor,
    ):
        query, _ = self.wq_b(q_lora)
        query = rearrange(query, "l (h d) -> l h d", d=self.head_dim)

        q_rope, _ = torch.split(
            query, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1
        )
        key, _ = self.wk(x)
        key = self.k_norm(key)
        k_rope, _ = torch.split(
            key, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1
        )

        if q_rope.shape[0] > 0:
            q_rope, k_rope = self.rotary_emb(positions, q_rope, k_rope)
            query[..., : self.rope_head_dim] = q_rope
            key[..., : self.rope_head_dim] = k_rope

        if CP_METADATA:
            key = cp_all_gather_rerange_output(
                key.contiguous(),
                CP_METADATA.value,
                self.comm_convertor
            )

        from fast_hadamard_transform import hadamard_transform_fuse_quant

        qd = query.shape[-1]
        kd = key.shape[-1]
        assert qd == kd and (qd & (qd - 1)) == 0, \
            "Dimension must be a power of 2 for Hadamard transform."
        q_fp8, q_scale, k_fp8, k_scale = hadamard_transform_fuse_quant(query, key, qd**-0.5)

        return q_fp8, q_scale, k_fp8, k_scale

    def _get_topk_paged(
        self,
        forward_batch: ForwardBatch,
        layer_id: int,
        q_fp8: torch.Tensor,
        weights: torch.Tensor,
        metadata: BaseIndexerMetadata,
    ) -> torch.Tensor:
        if TYPE_CHECKING:
            assert isinstance(forward_batch.token_to_kv_pool, DSATokenToKVPool)

        page_size = forward_batch.token_to_kv_pool.page_size
        # NOTE(dark): blocksize = 64 is hardcoded in deep_gemm
        assert page_size == 64, "only support page size 64"

        # NOTE(dark): this support extend/decode/decode+graph
        block_tables = metadata.get_page_table_64()

        # align to page_size
        max_seq_len = block_tables.shape[1] * page_size
        kv_cache_fp8 = forward_batch.token_to_kv_pool.get_index_k_with_scale_buffer(
            layer_id=layer_id
        )

        blocksize = page_size
        if (
            forward_batch.forward_mode.is_target_verify()
            or forward_batch.forward_mode.is_draft_extend()
        ):
            seqlens_32 = metadata.get_seqlens_expanded()
        else:
            seqlens_32 = metadata.get_seqlens_int32()

        schedule_metadata = deep_gemm_oss.get_paged_mqa_logits_metadata(
            seqlens_32, blocksize, self.sm_count
        )

        assert len(q_fp8.shape) == 3
        # the next_n dim is always 1, and block_tables corresponding to it
        q_fp8 = q_fp8.unsqueeze(1)
        assert len(kv_cache_fp8.shape) == 2
        block_kv = 64
        num_heads_kv = 1
        head_dim_with_sf = 132
        kv_cache_fp8 = kv_cache_fp8.view(
            kv_cache_fp8.shape[0], block_kv, num_heads_kv, head_dim_with_sf
        )
        assert len(weights.shape) == 3
        weights = weights.squeeze(2)
        # [b, max_seq_len]
        logits = deep_gemm_oss.fp8_paged_mqa_logits(
            q_fp8,
            kv_cache_fp8,
            weights,
            seqlens_32,
            block_tables,
            schedule_metadata,
            max_seq_len,
            clean_logits=False,
        )
        # NOTE(dark): logits should be cleaned in topk_transform
        topk_result = metadata.topk_transform(logits, self.index_topk)
        return topk_result

    def _get_topk_ragged(
        self,
        forward_batch: ForwardBatch,
        layer_id: int,
        q_fp8: torch.Tensor,
        weights: torch.Tensor,
        metadata: BaseIndexerMetadata,
    ) -> torch.Tensor:
        if TYPE_CHECKING:
            assert isinstance(forward_batch.token_to_kv_pool, DSATokenToKVPool)

        page_size = forward_batch.token_to_kv_pool.page_size
        assert page_size == 64, "only support page size 64"
        assert len(weights.shape) == 3
        weights = weights.squeeze(-1)
        k_fp8_list = []
        k_scale_list = []
        ks_list = []
        ke_list = []
        q_offset = 0
        k_offset = 0

        block_tables = metadata.get_page_table_64()
        seq_lens_expanded = metadata.get_seqlens_expanded()

        assert (
            forward_batch.seq_lens_cpu is not None
            and forward_batch.extend_seq_lens_cpu is not None
        )

        for i in range(forward_batch.batch_size):
            seq_len = forward_batch.seq_lens_cpu[i].item()
            assert isinstance(seq_len, int)
            k_fp8 = forward_batch.token_to_kv_pool.get_index_k_continuous(
                layer_id,
                seq_len,
                block_tables[i],
            )
            k_scale = forward_batch.token_to_kv_pool.get_index_k_scale_continuous(
                layer_id,
                seq_len,
                block_tables[i],
            )
            seq_len = forward_batch.seq_lens_cpu[i]
            extend_seq_len = forward_batch.extend_seq_lens_cpu[i]
            ks = torch.full((extend_seq_len,), k_offset, dtype=torch.int32, device="cuda")
            ke = ks + seq_lens_expanded[q_offset : q_offset + extend_seq_len]
            k_fp8_list.append(k_fp8)
            k_scale_list.append(k_scale)
            ks_list.append(ks)
            ke_list.append(ke)
            q_offset += extend_seq_len
            k_offset += seq_len

        k_fp8 = torch.cat(k_fp8_list, dim=0).view(torch.float8_e4m3fn)
        k_scale = torch.cat(k_scale_list, dim=0).view(torch.float32).squeeze(-1)
        kv_fp8 = (k_fp8, k_scale)
        ks = torch.cat(ks_list, dim=0)
        ke = torch.cat(ke_list, dim=0)
        logits = deep_gemm_oss.fp8_mqa_logits(
            q_fp8,      # [s_q, nh, hd]
            kv_fp8,     # tuple: ([s_k, hd], [s_k])
            weights,    # [s_q, nh]
            ks,         # cu_seq_len_k_start, [s_q]
            ke,         # cu_seq_len_k_end, [s_q]
            clean_logits=False, # not clean the unfilled logits into -inf
        )

        assert logits.shape[0] == len(seq_lens_expanded)
        topk_result = metadata.topk_transform(logits, self.index_topk, row_starts=ks)

        return topk_result

    def _get_topk_ragged_cp(
        self,
        forward_batch: ForwardBatch,
        layer_id: int,
        q_fp8: torch.Tensor,
        weights: torch.Tensor,
        metadata: BaseIndexerMetadata,
        seq_lens_expanded: torch.Tensor,
    ) -> torch.Tensor:
        if TYPE_CHECKING:
            assert isinstance(forward_batch.token_to_kv_pool, DSATokenToKVPool)

        page_size = forward_batch.token_to_kv_pool.page_size
        assert page_size == 64, "only support page size 64"
        assert len(weights.shape) == 3
        weights = weights.squeeze(-1)
        k_fp8_list = []
        k_scale_list = []
        ks_list = []
        ke_list = []
        q_offset = 0
        k_offset = 0

        block_tables = metadata.get_page_table_64()

        assert (
            forward_batch.seq_lens_cpu is not None
            and forward_batch.extend_seq_lens_cpu is not None
        )
        assert forward_batch.batch_size <= 1, f"CP only support bs<=1 for now"

        seq_len = forward_batch.seq_lens_cpu[0].item()
        k_fp8 = forward_batch.token_to_kv_pool.get_index_k_continuous(
            layer_id,
            seq_len,
            block_tables[0],
        )
        k_scale = forward_batch.token_to_kv_pool.get_index_k_scale_continuous(
            layer_id,
            seq_len,
            block_tables[0],
        )
        # extend_seq_len = forward_batch.extend_seq_lens_cpu[0]
        q_seq_len = q_fp8.shape[0]
        ks = torch.full((q_seq_len,), k_offset, dtype=torch.int32, device="cuda")
        ke = ks + seq_lens_expanded[q_offset : q_offset + q_seq_len]
        k_fp8_list.append(k_fp8)
        k_scale_list.append(k_scale)
        ks_list.append(ks)
        ke_list.append(ke)
        # q_offset += extend_seq_len
        # k_offset += seq_len

        k_fp8 = torch.cat(k_fp8_list, dim=0).view(torch.float8_e4m3fn)
        k_scale = torch.cat(k_scale_list, dim=0).view(torch.float32).squeeze(-1)
        kv_fp8 = (k_fp8, k_scale)
        ks = torch.cat(ks_list, dim=0)
        ke = torch.cat(ke_list, dim=0)
        logits = deep_gemm_oss.fp8_mqa_logits(
            q_fp8,      # [s_q, nh, hd]
            kv_fp8,     # tuple: ([s_k, hd], [s_k])
            weights,    # [s_q, nh]
            ks,         # cu_seq_len_k_start, [s_q]
            ke,         # cu_seq_len_k_end, [s_q]
            clean_logits=False, # not clean the unfilled logits into -inf
        )

        assert logits.shape[0] == len(seq_lens_expanded)
        topk_result = metadata.topk_transform(
            logits, self.index_topk,
            lengths=seq_lens_expanded,
            cu_seqlens_q=seq_lens_expanded.new_tensor([0, q_seq_len])
        )

        return topk_result

    def forward_cuda(
        self,
        x: torch.Tensor,
        q_lora: torch.Tensor,
        index_k: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        layer_id: int,
        **kwargs, # comm_manager: Optional[DecoderCommMananger] = None
    ) -> Optional[torch.Tensor]:
        if TYPE_CHECKING:
            assert isinstance(forward_batch.token_to_kv_pool, DSATokenToKVPool)

        metadata = forward_batch.attn_backend.get_indexer_metadata(
            layer_id, forward_batch
        )
        # skip NSA if attention backend choose to skip this batch
        if metadata is None:
            return None

        query, _ = self.wq_b(q_lora)
        query = rearrange(query, "l (h d) -> l h d", d=self.head_dim)

        q_rope, _ = torch.split(
            query, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1
        )
        key = index_k
        key = self.k_norm(key)
        k_rope, _ = torch.split(
            key, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1
        )

        if q_rope.shape[0] > 0:
            q_rope, k_rope = self.rotary_emb(positions, q_rope, k_rope)
            query[..., : self.rope_head_dim] = q_rope
            key[..., : self.rope_head_dim] = k_rope

        from fast_hadamard_transform import hadamard_transform_fuse_quant

        qd = query.shape[-1]
        kd = key.shape[-1]
        assert qd == kd and (qd & (qd - 1)) == 0, \
            "Dimension must be a power of 2 for Hadamard transform."
        q_fp8, q_scale, k_fp8, k_scale = hadamard_transform_fuse_quant(query, key, qd**-0.5)

        weights = self._get_logits_head_gate(x, q_scale)

        # k_fp8: (seq_len, head_dim) fp8_e4m3fn
        # k_buffer: (num_total_tokens + page_size, head_dim) fp8_e4m3fn
        # k_scale: (seq_len, head_dim // block_size = 1) fp8_e4m3fn
        # k_scale_cache: (num_total_tokens + page_size, head_dim // block_size = 1) fp8_e4m3fn
        forward_batch.token_to_kv_pool.set_index_k_and_scale_buffer(
            layer_id=layer_id,
            loc=forward_batch.out_cache_loc,
            index_k=k_fp8,
            index_k_scale=k_scale,
        )

        if x.shape[0] == 0:
            return x.new_zeros((0, self.index_topk), dtype=torch.int32)

        if is_cuda():
            if (
                forward_batch.forward_mode.is_decode_or_idle()
                or forward_batch.forward_mode.is_target_verify()
                or forward_batch.forward_mode.is_draft_extend()
            ):
                topk_result = self._get_topk_paged(
                    forward_batch, layer_id, q_fp8, weights, metadata
                )
            else:
                assert forward_batch.seq_lens_cpu is not None
                if CP_METADATA:
                    seqlens_expanded = metadata.get_seqlens_expanded()
                    topk_result = self._get_topk_ragged_cp(
                        forward_batch, layer_id, q_fp8, weights, metadata, seqlens_expanded
                    )
                else:
                    topk_result = self._get_topk_ragged(
                        forward_batch, layer_id, q_fp8, weights, metadata
                    )

        return topk_result

    def forward_npu(
        self,
        x: torch.Tensor,
        q_lora: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        layer_id: int,
    ) -> torch.Tensor:
        import torch_npu

        from sglang.srt.distributed import (
            get_attn_tp_world_size,
            get_attn_tp_group,
        )
        from sglang.srt.env import global_server_args_dict
        from sglang.srt.utils import get_bool_env_var

        is_prefill = (
            forward_batch.forward_mode.is_extend()
            and not forward_batch.forward_mode.is_draft_extend()
            and not forward_batch.forward_mode.is_target_verify()
        )

        if hasattr(self.rotary_emb, 'cos_sin_cache'):
            cos_sin = self.rotary_emb.cos_sin_cache[positions]
            cos, sin = cos_sin.chunk(2, dim=-1)
            cos = cos.repeat(1, 2).view(-1, 1, 1, self.rope_head_dim)
            sin = sin.repeat(1, 2).view(-1, 1, 1, self.rope_head_dim)
        else:
            cos, sin = self.rotary_emb.cos_cached[positions], self.rotary_emb.sin_cached[positions]
            cos = cos.view(-1, 1, 1, self.rope_head_dim)
            sin = sin.view(-1, 1, 1, self.rope_head_dim)

        slot_mapping = forward_batch.attn_metadata.slot_mapping
        block_table = forward_batch.attn_metadata.block_table

        bs = x.shape[0]

        q = self.wq_b(q_lora)[0]  # [bs, 1536] @ [1536, 64 * 128] = [bs, 64 * 128]
        q = q.view(bs, self.n_heads, self.head_dim)  # [bs, 64, 128]
        q_pe, q_nope = torch.split(
            q,
            [self.rope_head_dim, self.head_dim - self.rope_head_dim],
            dim=-1,
        )  # [bs, 64, 64 + 64]

        q_pe = q_pe.view(bs, self.n_heads, 1, self.rope_head_dim)
        q_pe = torch_npu.npu_interleave_rope(q_pe, cos, sin).view(
            bs, self.n_heads, self.rope_head_dim
        )  # [bs, n, d]
        q = torch.cat([q_pe, q_nope], dim=-1)

        k_proj = self.wk(x)[0]  # [b, s, 7168] @ [7168, 128] = [b, s, 128]
        k = self.k_norm(k_proj)
        k_pe, k_nope = torch.split(
            k,
            [self.rope_head_dim, self.head_dim - self.rope_head_dim],
            dim=-1,
        )  # [bs, 64 + 64]

        k_pe = k_pe.view(-1, 1, 1, self.rope_head_dim)
        k_pe = torch_npu.npu_interleave_rope(k_pe, cos, sin).view(
            bs, 1, self.rope_head_dim
        )  # [bs, 1, d]
        k = torch.cat([k_pe, k_nope.unsqueeze(1)], dim=-1)  # [bs, 1, 128]

        forward_batch.token_to_kv_pool.set_index_k_and_scale_buffer(layer_id, slot_mapping, k)

        layout_key = "PA_BSND"
        if is_prefill:
            actual_seq_lengths_kv = torch.tensor(forward_batch.attn_metadata.actual_seq_lengths_kv).to(device=q.device)
            actual_seq_lengths_q = torch.tensor(forward_batch.attn_metadata.actual_seq_lengths).to(
                device=q.device
            )
        else:
            if forward_batch.attn_metadata.query_len_tensor is None:
                actual_seq_lengths_q = torch.tensor(
                    [1 + i * 1 for i in range(bs)], dtype=torch.int32, device=k.device
                )
            else:
                actual_seq_lengths_q = (
                    forward_batch.attn_metadata.query_len_tensor
                )
            actual_seq_lengths_kv = forward_batch.attn_metadata.seq_lens_tensor.cumsum(dim=0)

        past_key_states = forward_batch.token_to_kv_pool.get_index_k_with_scale_buffer(layer_id)
        block_table = (
            block_table[: actual_seq_lengths_q.size()[0]] if is_prefill else block_table
        )
        x = x.view(-1, self.hidden_size)
        weights = self.weights_proj(x)[0]
        if is_prefill and global_server_args_dict.get("enable_mla_l1_5_cache", False):
            # pcp
            attn_metadata = forward_batch.attn_metadata
            past_key_state_local = past_key_states.reshape(-1, self.head_dim).index_select(0, attn_metadata.kv_index_list)
            rank_counts = attn_metadata.per_rank_count
            inv_perm = attn_metadata.inv_perm

            attn_tp_group = get_attn_tp_group()
            tp_size = attn_tp_group.world_size
            device_group = attn_tp_group.device_group

            past_key_state_gathered = [
                torch.empty(rank_counts[r], self.head_dim,
                            dtype=past_key_state_local.dtype, device=past_key_state_local.device)
                for r in range(tp_size)
            ]

            torch.distributed.all_gather(past_key_state_gathered, past_key_state_local, group=device_group)

            past_key_states = torch.cat(past_key_state_gathered, dim=0)[inv_perm].view(-1, 1, self.head_dim)
            block_table = None
            layout_key = "TND"

        use_dcp = (global_server_args_dict["kvp_size"] > 1)
        if not is_prefill and use_dcp:
            #dcp
            block_table = forward_batch.attn_metadata.kvp_block_table
            kvp_group = get_attn_tp_group()
            kvp_rank = kvp_group.rank_in_group
            kvp_size = global_server_args_dict["kvp_size"]
            actual_seq_lengths_query=actual_seq_lengths_q.to(k.device).to(torch.int32)
            actual_seq_lengths_key=forward_batch.attn_metadata.kvp_context_lens_tensor.to(k.device).to(torch.int32)
            cur_seq_lengths_query = torch.concat((torch.tensor([0], device=k.device, dtype=torch.int32), actual_seq_lengths_q.to(k.device).to(torch.int32)))
            sparse_mode = 0
            if kvp_rank == 0:
                sparse_mode = 3
            init_cnt_req = forward_batch.attn_metadata.init_cnt_req
            local_cnt_req = forward_batch.attn_metadata.local_cnt_req
            topk_indices_local, topk_values = torch_npu.npu_lightning_indexer(
                q.view(-1, self.n_heads, self.head_dim), 
                past_key_states, weights.to(torch.bfloat16),
                actual_seq_lengths_query=actual_seq_lengths_query,   # (B+1,)
                actual_seq_lengths_key=actual_seq_lengths_key,       # (B+1,)
                block_table=block_table,                        # (B, max_blocks)
                layout_query="TND",
                layout_key=layout_key,
                sparse_count=self.index_topk,
                sparse_mode=sparse_mode,
                return_value=True,
            )

            q_len = cur_seq_lengths_query[1:] - cur_seq_lengths_query[:-1]
            kv_len = forward_batch.attn_metadata.kvp_context_lens_tensor
            T, D = topk_values.shape[0], topk_values.shape[2]
            seq_range = torch.arange(T).to(k.device)                 
            cumsum = torch.cumsum(q_len, dim=0)         
            batch_id  = (seq_range.unsqueeze(1) >= cumsum.unsqueeze(0)).sum(dim=1)  # [T]
            token_kv_len = kv_len[batch_id]                 # [T]

            mask = torch.arange(D).to(k.device)[None,:]>=token_kv_len[:,None]
            mask = mask[:,None,:]
            topk_values.masked_fill_(mask, float('-inf'))
 
            sparse_topk = self.index_topk - self.num_init_tokens - self.num_local_tokens
            idx = topk_indices_local.squeeze(1).to(torch.int64)   # [T, D]
            val = topk_values.squeeze(1)                    # [T, D]
            T, D = idx.shape
            device = idx.device
            q_lens = (cur_seq_lengths_query[1:] - cur_seq_lengths_query[:-1]).to(torch.int64) 
            global_hist_lens = forward_batch.attn_metadata.seq_lens_tensor - q_lens
            q_cumsum = actual_seq_lengths_q.to(k.device).to(torch.int32)
            q_start  = torch.cat([torch.tensor([0]).to(k.device).to(torch.int32), q_cumsum[:-1]])
            kv_len = forward_batch.attn_metadata.kvp_context_lens_tensor
            seq_id = torch.zeros(T, dtype=torch.int32).to(k.device)
            seq_id.scatter_(0, q_start.to(torch.int64), torch.ones(len(q_lens), dtype=torch.int32).to(k.device))
            seq_id = seq_id.cumsum(0) - 1
            # 计算每个token对应的kv len.
            if kvp_rank == 0:
                token_kv_len = kv_len[seq_id]
                token_q_len = q_lens[seq_id]
                token_start = q_start[seq_id]
                local_pos = torch.arange(T).to(k.device) - token_start
                kv_pos = token_kv_len - token_q_len + local_pos + 1 # 每个token对应的kv len
                local_curr = local_cnt_req[seq_id] + local_pos + 1 # 每个token对应的local token数
                kv_local_start = kv_pos - local_curr # 每个token对应的local开始的index
                kv_init_end = init_cnt_req[seq_id] # 每个token对应的init的个数
            else:
                kv_pos = kv_len[seq_id]
                kv_local_start = kv_pos - local_cnt_req[seq_id]
                kv_init_end = init_cnt_req[seq_id]
            
            # mask
            # init&local
            valid_idx = idx >= 0
            val = val.masked_fill_(~valid_idx, float("-inf"))
            forced = (
                (idx < kv_init_end.unsqueeze(1)) |
                (idx >= kv_local_start.unsqueeze(1))
            )
            forced = forced & valid_idx
            val = val.masked_fill_(forced, float("-inf"))

            val_flat = val.contiguous().view(-1)
            idx_flat = idx.contiguous().view(-1)
            gather_flat = torch.empty(
                kvp_size*val_flat.numel(),
                dtype=topk_values.dtype,
                device=topk_values.device,
            )
            gather_flat_idx = torch.empty(
                kvp_size*idx_flat.numel(),
                dtype=idx.dtype,
                device=idx.device,
            )
            torch.distributed.all_gather_into_tensor(
                gather_flat,
                val_flat,
                group=kvp_group.device_group,
            )
            torch.distributed.all_gather_into_tensor(
                gather_flat_idx,
                idx_flat,
                group=kvp_group.device_group,
            )
            all_val = (
                gather_flat.view(kvp_size, T, D)
                .transpose(0, 1)            # [T, kvp_size, D]
                .contiguous()
                .view(T, kvp_size * D)      # [T, kvp_size*D]
            )
            all_idx = (
                gather_flat_idx.view(kvp_size, T, D)
                .transpose(0, 1)            # [T, kvp_size, D]
                .contiguous()
                .view(T, kvp_size * D)      # [T, kvp_size*D]
            )
            #计算 非init和非local的值
            _, indcies = all_val.topk(sparse_topk, dim=1)
            gather_idx = torch.gather(all_idx, dim=1, index=indcies)
            mask_indcies = (indcies//self.index_topk == kvp_rank)
            gather_idx = gather_idx.masked_fill_(~mask_indcies,-1)

            #合并
            base_init = torch.arange(self.num_init_tokens).unsqueeze(0).expand(T, -1).to(k.device)
            mask_init = (base_init < kv_init_end.unsqueeze(1)).to(k.device)
            # 超出部分填-1
            init_res = torch.where(mask_init, base_init, -1)

            offset = torch.arange(self.num_local_tokens).unsqueeze(0).expand(T, -1).to(k.device)
            values = kv_local_start.unsqueeze(1) + offset
            local_lens = kv_pos - kv_local_start 
            mask_local = offset < local_lens.unsqueeze(1)
            init_local = torch.where(mask_local, values, torch.tensor(-1))
            # init+local+sparse
            topk_indices = torch.cat([init_res, init_local, gather_idx], dim=1)
            mask_res = (topk_indices == -1).to(torch.float32).to(k.device)
            _, order = torch.sort(mask_res, dim=1)
            topk_indices = topk_indices.gather(1, order)
      

            if global_server_args_dict["npu_kvp_accuracy_fix"] and kvp_rank == 0:
                assert global_server_args_dict["npu_disable_kv_nz"]
                from sglang.srt.env import ENV
                torchair_enable = forward_batch.all_decode_or_idle and ENV.npu_enable_graph
                import torchair as tng
                if torchair_enable:
                    tng.scope.npu_wait_tensor(past_key_states, topk_indices_local)
                slot_mapping = forward_batch.attn_metadata.kvp_current_slots_mapping
                torch_npu.npu_scatter_nd_update_(past_key_states.view(-1, 1, self.head_dim), slot_mapping.reshape(-1, 1), k)
            return topk_indices.to(torch.int32).unsqueeze(1)


        enable_sp_for_indexer = is_prefill and global_server_args_dict.get("enable_mla_l1_5_cache", False) and global_server_args_dict["npu_enable_sp_for_indexer"]

        if not enable_sp_for_indexer:
            cur_seq_lengths_query = torch.concat((torch.tensor([0], device=k.device, dtype=torch.int32), actual_seq_lengths_q.to(k.device).to(torch.int32)))
            cur_seq_lengths_key = torch.concat((torch.tensor([0], device=k.device, dtype=torch.int32), actual_seq_lengths_kv.to(k.device).to(torch.int32)))
            topk_indices, _ = torch_npu.mlp_lightning_indexer(
                q.view(-1, self.n_heads, self.head_dim),
                past_key_states, weights.to(torch.float32),
                cur_seq_lengths_query=cur_seq_lengths_query,   # (B+1,)
                cur_seq_lengths_key=cur_seq_lengths_key,       # (B+1,)
                block_table=block_table,                        # (B, max_blocks)
                layout_query="TND",
                layout_key=layout_key,
                sparse_count=self.index_topk,
                kv_block_len=self.kv_block_size,
                q_block_len=self.q_block_size,
                init_num=self.num_init_tokens,
                local_num=self.num_local_tokens,
                sparse_mode=3,
            )
        else:
            attn_sp_token_nums = get_attn_tp_group().get_local_sp_token_num(forward_batch.global_sp_num_tokens)
            cu_start_cpu = torch.tensor([0], dtype = torch.int32)
            global_cu_kv_lens_cpu = torch.concat((cu_start_cpu, torch.tensor(forward_batch.attn_metadata.actual_seq_lengths_kv, dtype =torch.int32)))
            cur_cu_extend_lens_cpu = torch.concat((cu_start_cpu, torch.tensor(forward_batch.attn_metadata.actual_seq_lengths, dtype = torch.int32)))
            attn_sp_token_nums_cpu = torch.tensor(attn_sp_token_nums, dtype =torch.int32)
            cu_attn_sp_token_nums_cpu = torch.concat((cu_start_cpu, torch.cumsum(attn_sp_token_nums_cpu, dim = 0)))
            global_kv_lens_cpu = global_cu_kv_lens_cpu[1:] - global_cu_kv_lens_cpu[:-1]
            cur_extend_lens_cpu = cur_cu_extend_lens_cpu[1:] - cur_cu_extend_lens_cpu[:-1]
            cp_rank = get_attn_tp_group().rank_in_group
            cp_size = get_attn_tp_group().world_size
            local_extend_lens_with_zero, local_total_kv_lens_with_zero = compute_local_lens(cur_extend_lens_cpu, global_kv_lens_cpu - cur_extend_lens_cpu, attn_sp_token_nums_cpu, cp_rank)
            
            forward_batch.local_extend_lens_with_zero = local_extend_lens_with_zero
            forward_batch.local_total_kv_lens_with_zero = local_total_kv_lens_with_zero
            forward_batch.global_cu_kv_lens_cpu = global_cu_kv_lens_cpu
            forward_batch.cu_attn_sp_token_nums_cpu = cu_attn_sp_token_nums_cpu

            if local_extend_lens_with_zero.sum().item() > 0:
                past_key_states, local_extend_lens, local_total_kv_lens = get_non_zero(past_key_states, global_cu_kv_lens_cpu, local_extend_lens_with_zero, local_total_kv_lens_with_zero)
                cur_seq_lengths_query_cpu = torch.concat((cu_start_cpu, torch.cumsum(local_extend_lens, dim = 0)))
                cu_local_total_kv_lens_cpu = torch.concat((cu_start_cpu, torch.cumsum(local_total_kv_lens, dim = 0)))
                slice_start = cu_attn_sp_token_nums_cpu[cp_rank].item()
                slice_end = cu_attn_sp_token_nums_cpu[cp_rank+1].item()
                q = q.view(-1, self.n_heads, self.head_dim)[slice_start:slice_end].contiguous()
                weights = weights[slice_start:slice_end].contiguous()
                #past_key_states = get_local_past_key_states(past_key_states, global_cu_kv_lens_cpu, cu_local_total_kv_lens_cpu)

                cur_seq_lengths_query = cur_seq_lengths_query_cpu.to(k.device).to(torch.int32)
                cur_seq_lengths_key = cu_local_total_kv_lens_cpu.to(k.device).to(torch.int32)

                forward_batch.cur_seq_lengths_query = cur_seq_lengths_query
                forward_batch.cur_seq_lengths_key = cur_seq_lengths_key

                topk_indices, _ = torch_npu.mlp_lightning_indexer(
                    q.view(-1, self.n_heads, self.head_dim),
                    past_key_states, weights.to(torch.float32),
                    cur_seq_lengths_query=cur_seq_lengths_query,   # (B+1,)
                    cur_seq_lengths_key=cur_seq_lengths_key,       # (B+1,)
                    block_table=block_table,                        # (B, max_blocks)
                    layout_query="TND",
                    layout_key=layout_key,
                    sparse_count=self.index_topk,
                    kv_block_len=self.kv_block_size,
                    q_block_len=self.q_block_size,
                    init_num=self.num_init_tokens,
                    local_num=self.num_local_tokens,
                    sparse_mode=3,
                )
            else:
                topk_indices = torch.empty((0, 1, self.index_topk), dtype = torch.int32, device = k.device)
            #topk_indices = get_attn_tp_group().all_gather(topk_indices, dim=0, output_split_sizes=attn_sp_token_nums)

        return topk_indices

class IndexerBf16(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        index_n_heads: int,
        index_head_dim: int,
        rope_head_dim: int,
        index_topk: int,
        index_k_norm_type: str,
        q_lora_rank: int,
        max_position_embeddings: int,
        rope_theta: float,
        layer_id: int,
        scale_fmt: Optional[str],
        block_size: int = 128,
        rope_scaling: Optional[Dict[str, Any]] = None,
        is_neox_style = False,
        prefix: str = "",
        config: Optional[PretrainedConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        alt_stream: Optional[torch.cuda.Stream] = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_heads = index_n_heads
        self.head_dim = index_head_dim
        self.rope_head_dim = rope_head_dim
        self.index_topk = index_topk
        self.q_lora_rank = q_lora_rank
        self.layer_id = layer_id
        self.alt_stream = alt_stream
        if is_cuda():
            self.sm_count = deep_gemm_oss.get_num_sms()
            self.half_device_sm_count = align(self.sm_count // 2, 8)

        self.wq_b = ReplicatedLinear(
            self.q_lora_rank,
            self.n_heads * self.head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("wq_b", prefix),
        )
        self.wk = ReplicatedLinear(
            self.hidden_size,
            self.head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("wk", prefix),
        )
        if index_k_norm_type == "rms":
            self.k_norm = RMSNorm(self.head_dim)
        else:
            self.k_norm = V32LayerNorm(self.head_dim)
        # NOTE: weight_proj is not quantized
        self.weights_proj = ReplicatedLinear(
            self.hidden_size,
            self.n_heads,
            bias=False,
            params_dtype=torch.float32,
            prefix=add_prefix("weights_proj", prefix),
        )
        self.rotary_emb = get_rope_wrapper(
            rope_head_dim,
            rotary_dim=rope_head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,  # type: ignore
            rope_scaling=rope_scaling,
            is_neox_style=is_neox_style,
            device="cuda" if not is_npu() else "npu",
        )
        self.block_size = block_size
        self.scale_fmt = scale_fmt
        self.softmax_scale = self.head_dim**-0.5
        self.topk = 2048
        self.kv_block_size = getattr(config, "kv_block_size", 1)
        self.q_block_size = getattr(config, "q_block_size", 1)
        self.num_init_tokens = getattr(config, "index_init_tokens", 0)
        self.num_local_tokens = getattr(config, "index_local_tokens", 0)

    def _get_topk_ragged(
        self,
        forward_batch: ForwardBatch,
        layer_id: int,
        query: torch.Tensor,   # [l, h, d]
        weights: torch.Tensor, # [l, h, 1]
        key: torch.Tensor,     # [l, d]
        metadata: BaseIndexerMetadata,
    ):
        assert query.shape[0] == key.shape[0]
        bs = forward_batch.batch_size
        block_tables = metadata.get_page_table_64()
        page_size = 64
        max_seq_len = block_tables.shape[1] * page_size
        all_qk_logits = torch.full((query.shape[0], max_seq_len), float("-inf"), dtype=torch.float32, device=query.device)
        for i in range(bs):
            q_st = metadata.attn_metadata.cu_seqlens_q[i]
            q_ed = metadata.attn_metadata.cu_seqlens_q[i+1]
            q = query[q_st:q_ed]
            k = key[q_st:q_ed]
            w = weights[q_st:q_ed]

            l = q.shape[0]
            # [l, d] -> [h, l, d]
            k = k.unsqueeze(0).repeat_interleave(self.n_heads, 0)
            # [h, l, d] @ [h, d, l] -> [h, l, l]
            index_score = q.transpose(0, 1) @ k.transpose(-2, -1)
            # [h, l, l] -> [l, l]
            index_score = (w.transpose(0, 1) * F.relu(index_score)).sum(0)
            causal_mask = torch.triu(torch.full((l, l), float("-inf"), device=query.device), diagonal=1)
            index_score = index_score + causal_mask
            all_qk_logits[q_st:q_ed, :index_score.shape[1]] = index_score

        return all_qk_logits

    def _get_topk_paged(
        self,
        forward_batch: ForwardBatch,
        layer_id: int,
        query: torch.Tensor,   # [b, h, d]
        weights: torch.Tensor, # [b, h, 1]
        metadata: BaseIndexerMetadata,
    ):
        if TYPE_CHECKING:
            assert isinstance(forward_batch.token_to_kv_pool, DSATokenToKVPool)
        page_size = 64
        block_tables = metadata.get_page_table_64()
        seqlens_32 = metadata.get_seqlens_expanded()
        index_k_cache = forward_batch.token_to_kv_pool.get_index_k_with_scale_buffer(
            layer_id=layer_id
        )
        index_k_cache = index_k_cache.view(
            index_k_cache.shape[0], page_size, -1
        )
        max_seq_len = block_tables.shape[1] * page_size
        logits = triton_mqa_logits(query, weights.squeeze(2), index_k_cache, block_tables, seqlens_32, max_seq_len)
        return logits

    def _get_topk_paged_extend(
        self,
        forward_batch: ForwardBatch,
        layer_id: int,
        query: torch.Tensor,   # [s, h, d]
        weights: torch.Tensor, # [s, h, 1]
        metadata: BaseIndexerMetadata,
    ):
        if TYPE_CHECKING:
            assert isinstance(forward_batch.token_to_kv_pool, DSATokenToKVPool)
        page_size = 64
        block_tables = metadata.get_page_table_64()
        seqlens_32 = metadata.get_seqlens_int32()
        seq_lens_expanded = metadata.get_seqlens_expanded()
        index_k_cache = forward_batch.token_to_kv_pool.get_index_k_with_scale_buffer(
            layer_id=layer_id
        )
        index_k_cache = index_k_cache.view(
            index_k_cache.shape[0], page_size, -1
        )
        bs = forward_batch.batch_size
        token_to_batch = torch.arange(bs, dtype=torch.int32, device=query.device).repeat_interleave(forward_batch.extend_seq_lens)
        max_seq_len = block_tables.shape[1] * page_size
        weights = weights.squeeze(2)
        logits = triton_mqa_extend_logits(
            query, weights, index_k_cache, block_tables, token_to_batch, seqlens_32, seq_lens_expanded, max_seq_len
        )
        return logits

    def forward(
        self,
        x: torch.Tensor,
        q_lora: torch.Tensor,
        index_k: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        layer_id: int,
        **kwargs,
    ):
        if TYPE_CHECKING:
            assert isinstance(forward_batch.token_to_kv_pool, DSATokenToKVPool)
            assert isinstance(forward_batch.attn_backend, DpskSparseAttnBackend)

        metadata = forward_batch.attn_backend.get_indexer_metadata(
            layer_id, forward_batch
        )

        # skip NSA if attention backend choose to skip this batch
        if metadata is None:
            return None

        if not NSA_USE_REAL_INDEXER:  # temporary
            return self._forward_fake(x, q_lora, positions, forward_batch, layer_id)

        query, _ = self.wq_b(q_lora)
        query = rearrange(query, "l (h d) -> l h d", d=self.head_dim)

        q_rope, _ = torch.split(
            query, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1
        )
        key = index_k
        key = self.k_norm(key)
        k_rope, _ = torch.split(
            key, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1
        )
        q_rope, k_rope = self.rotary_emb(positions, q_rope, k_rope)

        query[..., : self.rope_head_dim] = q_rope
        key[..., : self.rope_head_dim] = k_rope

        forward_batch.token_to_kv_pool.set_index_k_and_scale_buffer(
            layer_id=layer_id,
            loc=forward_batch.out_cache_loc,
            index_k=key,
        )

        weights, _ = self.weights_proj(x.float())
        weights = weights * self.n_heads**-0.5
        weights = weights.unsqueeze(-1) * self.softmax_scale

        if is_cuda():
            if (
                forward_batch.forward_mode.is_decode_or_idle()
                or forward_batch.forward_mode.is_target_verify()
                or forward_batch.forward_mode.is_draft_extend()
            ):
                qk_logits = self._get_topk_paged(
                    forward_batch, layer_id, query, weights, metadata
                )
            else:
                assert forward_batch.seq_lens_cpu is not None
                if torch.all(forward_batch.extend_prefix_lens == 0):
                    qk_logits = self._get_topk_ragged(
                        forward_batch, layer_id, query, weights, key, metadata
                    )
                else:
                    qk_logits = self._get_topk_paged_extend(
                        forward_batch, layer_id, query, weights, metadata
                    )
        topk_result = metadata.topk_transform(
            qk_logits, self.index_topk,
            num_init_and_local_tokens=[self.num_init_tokens, self.num_local_tokens],
        )
        if kwargs.get("unit_test", False):
            return qk_logits, topk_result
        return topk_result

def compute_cu_seqlens(seqlens: torch.Tensor) -> torch.Tensor:
    assert seqlens.dtype == torch.int32 and seqlens.is_cuda
    return torch.nn.functional.pad(
        torch.cumsum(seqlens, dim=0, dtype=torch.int32), (1, 0)
    )

import triton
import triton.language as tl

@triton.jit
def triton_mqa_logits_kernel(
    q_ptr, # [b, h, d]
    w_ptr, # [b, h]
    k_cache_ptr, # [l, 64, d]
    r_ptr,
    block_table,
    seq_lens,
    q_stride_b, q_stride_h, q_stride_d,
    w_stride_b, w_stride_h,
    k_c_stride_l, k_c_stride_p, k_c_stride_d,
    r_stride_b, r_stride_l,
    block_table_b, block_table_d,
    HEAD_DIM: tl.constexpr = 128,
    BLOCK_M: tl.constexpr = 64,
    BLOCK_N: tl.constexpr = 64
):
    pid_b = tl.program_id(0)
    cur_seq_len = tl.load(seq_lens + pid_b)
    block_table += pid_b * block_table_b
    r_ptr += pid_b * r_stride_b
    off_m = tl.arange(0, BLOCK_M)
    off_d = tl.arange(0, HEAD_DIM)
    # [BM, d]
    q = tl.load(
        q_ptr + pid_b * q_stride_b +\
        off_m[:,None]*q_stride_h + off_d[None,:]
    )
    # [BM]
    w = tl.load(
        w_ptr + pid_b * w_stride_b + off_m
    )

    for i in range(0, tl.cdiv(cur_seq_len, BLOCK_N)):
        page_id = tl.load(block_table+i)
        k_st = i * BLOCK_N
        off_n = tl.arange(0, BLOCK_N)
        mask = (off_n + k_st) < cur_seq_len
        # [BN, d]
        k = tl.load(
            k_cache_ptr + page_id * k_c_stride_l +\
            off_n[:,None]*k_c_stride_p + off_d[None,:],
            mask=mask[:,None], other=0
        )
        # [BM, BN]
        s = tl.dot(q, k.trans())
        s = tl.maximum(s, 0.0)
        # [BN]
        r = tl.sum(s * w[:,None], axis=0)
        tl.store(r_ptr + k_st+off_n, r)


def triton_mqa_logits(
    query: torch.Tensor, # [b, h, d]
    weights, # [b, h]
    index_k_cache,
    block_tables, # [b, s]
    seqlens_32,
    max_seq_len,
):
    b, h, d = query.shape
    ret = query.new_empty(b, max_seq_len, dtype=torch.float32)
    triton_mqa_logits_kernel[(b,)](
        query,
        weights,
        index_k_cache,
        ret,
        block_tables,
        seqlens_32,
        *query.stride(),
        *weights.stride(),
        *index_k_cache.stride(),
        max_seq_len, 1,
        *block_tables.stride(),
        BLOCK_M=h
    )
    return ret

@triton.jit
def triton_mqa_extend_logits_kernel(
    q_ptr, # [s, h, d]
    w_ptr, # [s, h]
    k_cache_ptr, # [l, 64, d]
    r_ptr,
    block_table,
    token_to_batch, # [s]
    seq_lens, # [b]
    seq_lens_expanded, # [s]
    bs,
    q_stride_s, q_stride_h, q_stride_d,
    w_stride_s, w_stride_h,
    k_c_stride_l, k_c_stride_p, k_c_stride_d,
    r_stride_s, r_stride_l,
    block_table_b, block_table_d,
    HEAD_DIM: tl.constexpr = 128,
    BLOCK_M: tl.constexpr = 64,
    BLOCK_N: tl.constexpr = 64
):
    # One CTA per q token
    pid = tl.program_id(0)
    pid_b = tl.load(token_to_batch + pid)
    cur_seq_len = tl.load(seq_lens_expanded + pid)
    block_table += pid_b * block_table_b
    r_ptr += pid * r_stride_s
    off_m = tl.arange(0, BLOCK_M)
    off_d = tl.arange(0, HEAD_DIM)
    # [BM, d]
    q = tl.load(
        q_ptr + pid * q_stride_s + \
        off_m[:,None]*q_stride_h + off_d[None,:]
    )
    # [BM]
    w = tl.load(
        w_ptr + pid * w_stride_s + off_m
    )

    for i in range(0, tl.cdiv(cur_seq_len, BLOCK_N)):
        page_id = tl.load(block_table+i)
        k_st = i * BLOCK_N
        off_n = tl.arange(0, BLOCK_N)
        mask = (off_n + k_st) < cur_seq_len
        # [BN, d]
        k = tl.load(
            k_cache_ptr + page_id * k_c_stride_l + \
            off_n[:,None]*k_c_stride_p + off_d[None,:],
            mask=mask[:,None], other=0
        )
        # [BM, BN]
        s = tl.dot(q, k.trans())
        # relu
        s = tl.maximum(s, 0.0)
        # [BN]
        r = tl.sum(s * w[:,None], axis=0)
        tl.store(r_ptr + k_st+off_n, r)


def triton_mqa_extend_logits(
    query: torch.Tensor,    # [s, h, d]
    weights,                # [s, h]
    index_k_cache,          # [l, 64, d]
    block_tables,           # [b, s]
    token_to_batch,         # [s]
    seqlens_32,
    seq_lens_expanded,      # [s]
    max_seq_len,
):
    s, h, d = query.shape
    bs = block_tables.shape[0]
    ret = query.new_empty(s, max_seq_len, dtype=torch.float32)
    triton_mqa_extend_logits_kernel[(s,)](
        query,
        weights,
        index_k_cache,
        ret,
        block_tables,
        token_to_batch,
        seqlens_32,
        seq_lens_expanded,
        bs,
        *query.stride(),
        *weights.stride(),
        *index_k_cache.stride(),
        *ret.stride(),
        *block_tables.stride(),
        BLOCK_M=h
    )
    return ret
