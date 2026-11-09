from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn
from transformers import PretrainedConfig

from sglang.srt.npu.utils import npu_limit_core, npu_stream_switch, npu_super_kernel
from sglang.srt.ops.custom_op import CustomOp
from sglang.srt.utils import add_prefix, align, is_cuda, is_npu

if is_npu():
    import flash_ops  # noqa: F401  # Registers torch.ops.custom operators.
    import torch_npu  # noqa: F401 - registers device/operator implementations
    import torchair as tng

    npu_ops = torch.ops.custom

    try:
        import cp_core
    except ImportError:
        cp_core = None
else:
    torch_npu = None
    tng = None
    cp_core = None

if is_cuda():
    try:
        import deep_gemm_oss
    except ImportError as e:
        deep_gemm_oss = e
    import aok_sfa
    import flashinfer
    from fast_hadamard_transform import hadamard_transform_fuse_quant
from sglang.srt.distributed import (
    get_attn_sp_token_slice_pos,
    get_attn_tp_group,
)
from sglang.srt.env import ENV, global_server_args_dict
from sglang.srt.layers.attention.dp_attention import get_attention_tp_group, get_attention_tp_rank, get_attention_tp_size, get_attn_tp_dp_convertor
from sglang.srt.layers.attention.dsa.utils import NSA_USE_REAL_INDEXER
from sglang.srt.layers.embedding.rotary_embedding import get_rope_wrapper
from sglang.srt.layers.linear import ReplicatedLinear
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.utils import CP_METADATA, cp_all_gather_rerange_output, cp_all_gather_torch_rerange_output
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.ops.layernorm import RMSNorm

if TYPE_CHECKING:
    from sglang.srt.distributed.decoder_comm_manager import DecoderCommMananger
    from sglang.srt.layers.attention.dsa.triton_kernel import indexer_qk_rag
    from sglang.srt.layers.attention.dsa_backend import DpskSparseAttnBackend, NSAIndexerMetadata, NSAMetadata
    from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool, L1HalfDSATokenToKVPool


from sglang.srt.utils import get_colorful_logger

logger = get_colorful_logger(__name__)

DUAL_STREAM_TOKEN_THRESHOLD = 1024 if is_cuda() else 0


def indexer_cache_copy_rows_(
    cache: torch.Tensor,
    src_slots: torch.Tensor,
    dst_slots: torch.Tensor,
    dependency: torch.Tensor | None = None,
) -> torch.Tensor:
    if dependency is not None:
        dst_slots = tng.scope.npu_wait_tensor(dst_slots, dependency)
    return torch.ops.custom.npu_kv_cache_copy_rows_(
        cache.view(-1, cache.shape[-1]),
        src_slots.reshape(-1),
        dst_slots.reshape(-1),
    )


def rotate_activation(x: torch.Tensor) -> torch.Tensor:
    assert x.dtype == torch.bfloat16
    if x.shape[0] == 0:
        return x
    from fast_hadamard_transform import hadamard_transform

    hidden_size = x.size(-1)
    assert (hidden_size & (hidden_size - 1)) == 0, "Hidden size must be a power of 2 for Hadamard transform."
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
        return F.layer_norm(x.float(), (self.dim,), self.weight, self.bias, self.eps).type_as(x)


def ceil_div(a: int, b: int):
    return (a + b - 1) // b


class FP32RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        x_fp32 = x.to(torch.float32)
        w_fp32 = self.weight.to(torch.float32)

        return torch_npu.npu_rms_norm(x_fp32, w_fp32, self.eps)[0]


def get_non_zero_q_meta(past_key_states, global_cu_kv_lens_cpu, local_extend_lens, local_total_kv_lens):
    global_cu_kv_lens_cpu_list = global_cu_kv_lens_cpu.tolist()
    local_extend_lens_list = local_extend_lens.tolist()
    local_total_kv_lens_list = local_total_kv_lens.tolist()
    bs = len(local_extend_lens_list)
    res_local_extend_lens_list = list()
    res_local_total_kv_lens_list = list()
    res_list = list()
    for i in range(bs):
        if local_extend_lens_list[i] > 0:
            res_local_extend_lens_list.append(local_extend_lens_list[i])
            cur_len = local_total_kv_lens_list[i]
            res_local_total_kv_lens_list.append(cur_len)
            global_start = global_cu_kv_lens_cpu_list[i]
            global_end = global_cu_kv_lens_cpu_list[i + 1]
            if past_key_states is not None:
                cur = past_key_states[global_start:global_end][:cur_len]
                res_list.append(cur)
    res_local_extend_lens_cpu = torch.tensor(res_local_extend_lens_list, dtype=torch.int32)
    res_local_total_kv_lens_cpu = torch.tensor(res_local_total_kv_lens_list, dtype=torch.int32)
    if len(res_list) == 0:
        return None, res_local_extend_lens_cpu, res_local_total_kv_lens_cpu
    if len(res_local_extend_lens_list) == 1:
        return res_list[0], res_local_extend_lens_cpu, res_local_total_kv_lens_cpu
    else:
        return torch.cat(res_list, dim=0).contiguous(), res_local_extend_lens_cpu, res_local_total_kv_lens_cpu


def compute_block_scores_and_topk(
    q: torch.Tensor,  # [T_q, N, D] query
    pooled_max: torch.Tensor,  # [T_pooled, N, D] max pooled key
    pooled_min: torch.Tensor,  # [T_pooled, N, D] min pooled key
    cu_seq_lengths_query: torch.Tensor,  # [B+1] query 的 cumsum 长度
    cu_seq_lengths_key: torch.Tensor,  # [B+1] key (pooled) 的 cumsum 长度
    weights: torch.Tensor,  # [T_q] 每个 query token 的权重
    num_init_blocks: int,
    num_local_blocks: int,
    topk: int,
) -> torch.Tensor:
    T_q, N, D = q.shape
    B = cu_seq_lengths_query.shape[0] - 1

    block_indices = torch.full((T_q, topk), -1, dtype=torch.int32, device=q.device)

    # 按 batch 处理
    for b in range(B):
        q_start = cu_seq_lengths_query[b].item()
        q_end = cu_seq_lengths_query[b + 1].item()
        q_len = q_end - q_start

        k_start = cu_seq_lengths_key[b].item()
        k_end = cu_seq_lengths_key[b + 1].item()
        k_len = k_end - k_start

        if q_len == 0 or k_len == 0:
            continue
        q_batch = q[q_start:q_end]  # [q_len, N, D]
        k_max_batch = pooled_max[k_start:k_end]  # [k_len, N, D]
        k_min_batch = pooled_min[k_start:k_end]  # [k_len, N, D]
        weights_batch = weights[q_start:q_end]  # [q_len]

        q_pos = torch.clamp(q_batch, min=0)  # [q_len, N, D]
        q_neg = torch.clamp(q_batch, max=0)  # [q_len, N, D]

        scores_pos = torch.einsum("qhd,khd->qkh", q_pos, k_max_batch)  # [q_len, k_len, N]
        scores_neg = torch.einsum("qhd,khd->qkh", q_neg, k_min_batch)  # [q_len, k_len, N]

        scores = scores_pos + scores_neg  # [q_len, k_len, N]
        scores = torch.clamp(scores, min=0)  # ReLU: [q_len, k_len, N]
        scores = scores.sum(dim=2)  # 对所有 head 求和: [q_len, k_len]

        scores = scores * weights_batch.unsqueeze(1)  # [q_len, k_len]

        for i in range(q_len):
            scores_i = scores[i]  # [k_len]

            # 强制保留 init 和 local 块
            init_end = min(num_init_blocks, k_len)
            local_start = max(0, k_len - num_local_blocks)

            forced_mask = torch.zeros(k_len, dtype=torch.bool, device=q.device)
            forced_mask[:init_end] = True
            forced_mask[local_start:] = True

            # 设置强制块的分数为最大值（确保被选中）
            scores_i_masked = scores_i.clone()
            scores_i_masked[forced_mask] = float("inf")

            topk_val, topk_idx = torch.topk(scores_i_masked, k=min(topk, k_len), largest=True)

            topk_idx_global = topk_idx + k_start  # 转换为全局池化 key 索引
            block_indices[q_start + i, : len(topk_idx_global)] = topk_idx_global

    return block_indices


def split_to_kv_cache(input_tensor, block_size, cur_kv_seqlen, max_seq_len):
    """将连续 KV tensor 拆分到 PA block cache 格式"""
    import math

    device = input_tensor.device
    B = len(cur_kv_seqlen)
    total_seq_block_cnt = 0
    for i in range(B):
        seq_block_cnt = math.ceil(cur_kv_seqlen[i] / block_size)
        if seq_block_cnt == 0:
            seq_block_cnt = 1
        total_seq_block_cnt += seq_block_cnt
    block_table_list = torch.arange(0, total_seq_block_cnt, 1)

    block_tables = torch.zeros(B, math.ceil(max_seq_len / block_size)).to(torch.int32)
    input_shape = input_tensor.shape
    cache_tensor = (
        torch.zeros(B * math.ceil(max_seq_len / block_size), block_size, input_shape[-2], input_shape[-1]).to(input_tensor.dtype).to(device)
    )

    start = 0
    total_seq_block_cnt = 0
    for i in range(B):
        seq_block_cnt = math.ceil(cur_kv_seqlen[i] / block_size)
        if seq_block_cnt == 0:
            seq_block_cnt = 1
        block_tables[i, :seq_block_cnt] = block_table_list[total_seq_block_cnt : seq_block_cnt + total_seq_block_cnt]
        kv_len = cur_kv_seqlen[i]
        if kv_len == 0:
            kv_len = 1
        for block_idx in range(seq_block_cnt):
            block_offset = block_idx * block_size
            seq_offset = block_offset + start
            if block_idx == seq_block_cnt - 1:
                remain_size = kv_len - block_idx * block_size
                cache_tensor[block_tables[i][block_idx], :remain_size] = input_tensor[seq_offset : seq_offset + remain_size]
            else:
                cache_tensor[block_tables[i][block_idx], :block_size] = input_tensor[seq_offset : seq_offset + block_size]
        total_seq_block_cnt += seq_block_cnt
        start += kv_len
    return cache_tensor, block_tables.to(device).detach()


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
        scale_fmt: str | None,
        block_size: int = 128,
        rope_scaling: dict[str, Any] | None = None,
        is_neox_style: bool = True,
        prefix: str = "",
        config: PretrainedConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        npu_disable_dsa_head_parallel: bool = False,
        alt_stream: torch.cuda.Stream | None = None,
        skip_rope: bool = False,
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
        self.skip_rope = skip_rope
        self.npu_disable_dsa_head_parallel = npu_disable_dsa_head_parallel
        self.cp_size = get_attention_tp_size()
        self.cp_rank = get_attention_tp_rank()
        if global_server_args_dict["enable_dsa_qcp"]:
            self.comm_convertor = get_attn_tp_dp_convertor()
            assert self.comm_convertor is not None

        if is_cuda():
            self.sm_count = deep_gemm_oss.get_num_sms()
            self.half_device_sm_count = align(self.sm_count // 2, 8)
        device = "cuda" if not is_npu() else "npu"

        self.wq_b = ReplicatedLinear(
            self.q_lora_rank,
            self.n_heads * self.head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("wq_b", prefix),
        )
        if is_npu():
            # On CUDA the wk weight lives in the attention's fused
            # fused_qkv_a_proj_with_mqa and index_k is passed in from there;
            # only the NPU path projects index_k inside the indexer.
            self.wk = ReplicatedLinear(
                self.hidden_size,
                self.head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=add_prefix("wk", prefix),
            )
        if index_k_norm_type == "rms":
            if is_npu():
                self.k_norm = FP32RMSNorm(self.head_dim)
            else:
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
        if not self.skip_rope and (not is_npu() or global_server_args_dict["disaggregation_mode"] == "decode"):
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
        self.base_init = torch.arange(self.num_init_tokens, device=device).unsqueeze(0)
        self.local_init = torch.arange(self.num_local_tokens, device=device).unsqueeze(0)

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
        ans = torch.arange(0, self.index_topk, dtype=torch.int32, device=x.device)[None, ...].repeat(bs, 1)
        if forward_batch.forward_mode.is_extend():
            assert forward_batch.extend_seq_lens_cpu is not None and forward_batch.seq_lens_cpu is not None
            which = 0
            for _, (kv_len, qo_len) in enumerate(
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

        q_rope, _ = torch.split(query, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1)
        key, _ = self.wk(x)
        key = self.k_norm(key)
        k_rope, _ = torch.split(key, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1)

        if not self.skip_rope and q_rope.shape[0] > 0:
            q_rope, k_rope = self.rotary_emb(positions, q_rope, k_rope)
            query[..., : self.rope_head_dim] = q_rope
            key[..., : self.rope_head_dim] = k_rope

        if CP_METADATA:
            key = cp_all_gather_rerange_output(key.contiguous(), CP_METADATA.value, self.comm_convertor)

        qd = query.shape[-1]
        kd = key.shape[-1]
        assert qd == kd and (qd & (qd - 1)) == 0, "Dimension must be a power of 2 for Hadamard transform."
        q_fp8, q_scale, k_fp8, k_scale = hadamard_transform_fuse_quant(query, key, qd**-0.5)

        return q_fp8, q_scale, k_fp8, k_scale

    def _get_topk_paged(
        self,
        forward_batch: ForwardBatch,
        layer_id: int,
        q_fp8: torch.Tensor,
        weights: torch.Tensor,
        metadata: NSAIndexerMetadata,
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
        kv_cache_fp8 = forward_batch.token_to_kv_pool.get_index_k_with_scale_buffer(layer_id=layer_id)

        if forward_batch.forward_mode.is_target_verify() or forward_batch.forward_mode.is_draft_extend():
            seqlens_32 = metadata.get_seqlens_expanded()
        else:
            seqlens_32 = metadata.get_seqlens_int32()

        schedule_metadata = deep_gemm_oss.get_paged_mqa_logits_metadata(seqlens_32, page_size, self.sm_count)

        assert len(q_fp8.shape) == 3
        # the next_n dim is always 1, and block_tables corresponding to it
        q_fp8 = q_fp8.unsqueeze(1)
        num_heads_kv = 1
        head_dim_with_sf = 132
        kv_cache_fp8 = kv_cache_fp8.view(kv_cache_fp8.shape[0], page_size, num_heads_kv, head_dim_with_sf)
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
        topk_result = metadata.topk_transform(
            logits,
            self.index_topk,
            num_init_and_local_tokens=[self.num_init_tokens, self.num_local_tokens],
        )
        return topk_result

    def _get_topk_ragged(
        self,
        forward_batch: ForwardBatch,
        layer_id: int,
        q_fp8: torch.Tensor,
        weights: torch.Tensor,
        metadata: NSAIndexerMetadata,
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

        assert forward_batch.seq_lens_cpu is not None and forward_batch.extend_seq_lens_cpu is not None

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
            q_fp8,  # [s_q, nh, hd]
            kv_fp8,  # tuple: ([s_k, hd], [s_k])
            weights,  # [s_q, nh]
            ks,  # cu_seq_len_k_start, [s_q]
            ke,  # cu_seq_len_k_end, [s_q]
            clean_logits=False,  # not clean the unfilled logits into -inf
        )

        assert logits.shape[0] == len(seq_lens_expanded)
        topk_result = metadata.topk_transform(
            logits,
            self.index_topk,
            row_starts=ks,
            num_init_and_local_tokens=[self.num_init_tokens, self.num_local_tokens],
        )

        return topk_result

    def _get_topk_ragged_cp(
        self,
        forward_batch: ForwardBatch,
        layer_id: int,
        q_fp8: torch.Tensor,
        weights: torch.Tensor,
        metadata: NSAIndexerMetadata,
        seq_lens_expanded: torch.Tensor,
    ) -> torch.Tensor:
        if TYPE_CHECKING:
            assert isinstance(forward_batch.token_to_kv_pool, DSATokenToKVPool)

        if q_fp8.shape[0] == 0:
            return q_fp8.new_zeros((0, self.index_topk), dtype=torch.int32)

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

        assert forward_batch.seq_lens_cpu is not None and forward_batch.extend_seq_lens_cpu is not None
        attn_metadata = metadata.attn_metadata
        if self.layer_id == 0 or not hasattr(attn_metadata, "seqlens_q_cpu"):
            cu_seqlens_q_cpu = attn_metadata.cu_seqlens_q.cpu()
            attn_metadata.extend_seq_lens_cpu = cu_seqlens_q_cpu[1:] - cu_seqlens_q_cpu[:-1]

        for i in range(forward_batch.batch_size):
            seq_len = forward_batch.seq_lens_cpu[i].item()
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
            q_seq_len = attn_metadata.extend_seq_lens_cpu[i]
            ks = torch.full((q_seq_len,), k_offset, dtype=torch.int32, device="cuda")
            ke = ks + seq_lens_expanded[q_offset : q_offset + q_seq_len]
            k_fp8_list.append(k_fp8)
            k_scale_list.append(k_scale)
            ks_list.append(ks)
            ke_list.append(ke)
            q_offset += q_seq_len
            k_offset += seq_len

        k_fp8 = torch.cat(k_fp8_list, dim=0).view(torch.float8_e4m3fn)
        k_scale = torch.cat(k_scale_list, dim=0).view(torch.float32).squeeze(-1)
        kv_fp8 = (k_fp8, k_scale)
        ks = torch.cat(ks_list, dim=0)
        ke = torch.cat(ke_list, dim=0)
        logits = deep_gemm_oss.fp8_mqa_logits(
            q_fp8,  # [s_q, nh, hd]
            kv_fp8,  # tuple: ([s_k, hd], [s_k])
            weights,  # [s_q, nh]
            ks,  # cu_seq_len_k_start, [s_q]
            ke,  # cu_seq_len_k_end, [s_q]
            clean_logits=False,  # not clean the unfilled logits into -inf
        )

        assert logits.shape[0] == len(seq_lens_expanded)
        topk_result = metadata.topk_transform(
            logits,
            self.index_topk,
            lengths=seq_lens_expanded,
            cu_seqlens_q=attn_metadata.cu_seqlens_q,
            row_starts=ks,
            num_init_and_local_tokens=[self.num_init_tokens, self.num_local_tokens],
        )

        return topk_result

    def _get_topk_by_ag_kvcache(
        self,
        forward_batch: ForwardBatch,
        layer_id: int,
        q_fp8: torch.Tensor,
        weights: torch.Tensor,
        metadata: NSAIndexerMetadata,
    ):
        if TYPE_CHECKING:
            assert isinstance(forward_batch.token_to_kv_pool, DSATokenToKVPool)

        page_size = forward_batch.token_to_kv_pool.page_size
        assert page_size == 64, "only support page size 64"
        assert len(weights.shape) == 3
        weights = weights.squeeze(-1)

        page_table_1 = metadata.attn_metadata.page_table_1
        seq_lens_expanded = metadata.get_seqlens_expanded()
        seqlens_32 = metadata.get_seqlens_int32()
        attn_metadata = metadata.attn_metadata
        bs = forward_batch.batch_size

        if self.layer_id == 0 or not hasattr(attn_metadata, "cp_gather_k_meta"):
            cu_seqlens_q_cpu = attn_metadata.cu_seqlens_q.cpu()
            attn_metadata.cu_seqlens_q_cpu = cu_seqlens_q_cpu
            if global_server_args_dict["enable_dsa_qcp"]:
                attn_metadata.extend_seq_lens_cpu = cu_seqlens_q_cpu[1:] - cu_seqlens_q_cpu[:-1]
            else:
                attn_metadata.extend_seq_lens_cpu = forward_batch.extend_seq_lens.cpu()
            attn_metadata.seqlens_cpu = seqlens_32.cpu()
            attn_metadata.cp_gather_k_meta = [
                forward_batch.token_to_kv_pool.remap_token_idxs_for_gather(page_table_1[i][: attn_metadata.seqlens_cpu[i]]) for i in range(bs)
            ]
            attn_metadata.flat_idxs_meta = [
                forward_batch.token_to_kv_pool.get_index_k_flat_idxs_by_token_idxs(layer_id, 0, attn_metadata.cp_gather_k_meta[i][0])
                for i in range(bs)
            ]
            attn_metadata.batch_groups = self.group_batch_by_sum_seqlen(forward_batch, attn_metadata)

        def ag_and_permute(local_tensor: torch.Tensor, seqlen: int, local_indices, per_rank_count, inv_perm):
            gathered_tensor = local_tensor.new_empty(seqlen, local_tensor.shape[-1])
            tensor_list = list(gathered_tensor.split(per_rank_count))
            torch.distributed.all_gather(tensor_list, local_tensor, group=get_attention_tp_group().device_group)
            permuted_tensor = gathered_tensor[inv_perm]
            return permuted_tensor

        def _get_logits_and_topk():
            # delay return because other rank need to allgather
            if q_ed == q_st:
                return q_fp8.new_zeros((0, self.index_topk), dtype=torch.int32)

            logits = deep_gemm_oss.fp8_mqa_logits(
                q_fp8[q_st:q_ed],  # [s_q, nh, hd]
                kv_fp8,  # tuple: ([s_k, hd], [s_k])
                weights[q_st:q_ed],  # [s_q, nh]
                ks,  # cu_seq_len_k_start, [s_q]
                ke,  # cu_seq_len_k_end, [s_q]
                clean_logits=False,  # not clean the unfilled logits into -inf
            )
            if global_server_args_dict["enable_dsa_qcp"]:
                topk_result = get_rag_topk(
                    logits,
                    seq_lens_expanded[q_st:q_ed],
                    self.num_init_tokens,
                    self.num_local_tokens,
                    row_starts=ks,
                )
            else:
                topk_result = metadata.topk_transform(
                    logits,
                    self.index_topk,
                    lengths=seq_lens_expanded[q_st:q_ed],
                    row_starts=ks,
                    num_init_and_local_tokens=[self.num_init_tokens, self.num_local_tokens],
                )
            return topk_result

        k_fp8_list = []
        k_scale_list = []
        ks_list = []
        ke_list = []
        q_offset = 0
        k_offset = 0
        topk_result_list = []

        for group in attn_metadata.batch_groups:
            q_st = q_offset
            for i in group:
                seqlen = attn_metadata.seqlens_cpu[i].item()
                local_indices, per_rank_count, inv_perm = attn_metadata.cp_gather_k_meta[i]
                k_flat_idxs, s_flat_idxs = attn_metadata.flat_idxs_meta[i]
                buf = forward_batch.token_to_kv_pool.get_index_k_with_scale_buffer(layer_id)
                k_fp8 = buf.flatten()[k_flat_idxs]
                k_scale = buf.flatten()[s_flat_idxs]

                all_k_fp8 = ag_and_permute(k_fp8, seqlen, local_indices, per_rank_count, inv_perm)
                all_k_scale = ag_and_permute(k_scale, seqlen, local_indices, per_rank_count, inv_perm)

                q_seq_len = attn_metadata.extend_seq_lens_cpu[i].item()
                ks = torch.full((q_seq_len,), k_offset, dtype=torch.int32, device="cuda")
                ke = ks + seq_lens_expanded[q_offset : q_offset + q_seq_len]
                k_fp8_list.append(all_k_fp8)
                k_scale_list.append(all_k_scale)
                ks_list.append(ks)
                ke_list.append(ke)
                q_offset += q_seq_len
                k_offset += seqlen

            q_ed = q_offset
            k_fp8 = torch.cat(k_fp8_list, dim=0).view(torch.float8_e4m3fn)
            k_scale = torch.cat(k_scale_list, dim=0).view(torch.float32).squeeze(-1)
            kv_fp8 = (k_fp8, k_scale)
            ks = torch.cat(ks_list, dim=0)
            ke = torch.cat(ke_list, dim=0)

            topk_result = _get_logits_and_topk()
            topk_result_list.append(topk_result)

            k_fp8_list = []
            k_scale_list = []
            ks_list = []
            ke_list = []
            k_offset = 0

        return torch.cat(topk_result_list)

    def group_batch_by_sum_seqlen(
        self,
        forward_batch: ForwardBatch,
        attn_metadata: NSAMetadata,
        max_sum_seqlen_in_group: int = 1024 * 1024,
    ):
        """
        Split the batch into several groups
        to prevent `fp8_mqa_logits` from consuming too much GPU memory
        """
        groups = []
        cur_group = []
        cur_sum = 0
        for i in range(forward_batch.batch_size):
            seqlen = attn_metadata.seqlens_cpu[i].item()
            if cur_group and cur_sum + seqlen > max_sum_seqlen_in_group:
                groups.append(cur_group)
                cur_group = []
                cur_sum = 0
            cur_group.append(i)
            cur_sum += seqlen
        if cur_group:
            groups.append(cur_group)
        return groups

    def _get_topk_gather_topk(
        self,
        forward_batch: ForwardBatch,
        layer_id: int,
        q_fp8: torch.Tensor,  # [s, h, d]
        weights: torch.Tensor,  # [s, h, 1]
        metadata: NSAIndexerMetadata,
    ):
        if TYPE_CHECKING:
            assert isinstance(
                forward_batch.token_to_kv_pool,
                (DSATokenToKVPool, L1HalfDSATokenToKVPool),
            )
        attn_metadata = metadata.attn_metadata
        page_size = 64
        cp_l1half_meta = attn_metadata.cp_l1half_meta
        local_seqlens_k = cp_l1half_meta.local_seqlens_k
        local_table_64 = cp_l1half_meta.local_page_table_64
        global_valid_indices = cp_l1half_meta.global_valid_indices
        swa_mask = cp_l1half_meta.swa_mask
        max_seq_len = local_table_64.shape[1] * page_size
        if self.layer_id == 0 or not hasattr(attn_metadata, "paged_mqa_logits_schedule_metadata"):
            attn_metadata.paged_mqa_logits_schedule_metadata = deep_gemm_oss.get_paged_mqa_logits_metadata(local_seqlens_k, page_size, self.sm_count)

        kv_cache_fp8 = forward_batch.token_to_kv_pool.get_index_k_with_scale_buffer(layer_id=layer_id)
        kv_cache_fp8 = kv_cache_fp8.view(kv_cache_fp8.shape[0], page_size, 1, forward_batch.token_to_kv_pool.index_k_cache_dim)
        assert len(weights.shape) == 3
        weights = weights.squeeze(2)
        # every q token is treated as an independent token for calculation, ignore of s_q
        logits = deep_gemm_oss.fp8_paged_mqa_logits(
            q_fp8.unsqueeze(1),
            kv_cache_fp8,
            weights,
            local_seqlens_k,
            local_table_64,
            attn_metadata.paged_mqa_logits_schedule_metadata,
            max_seq_len,
            clean_logits=False,
        )
        logits = logits.view(q_fp8.shape[0], -1)
        if swa_mask is not None:
            logits = torch.where(swa_mask, float("inf"), logits)

        indices, vals = flashinfer.flexible_topk_trans_and_out_val(
            logits,
            global_valid_indices,  # global indices
            local_seqlens_k,
        )
        all_inds = get_attention_tp_group().all_gather(indices, dim=1)
        all_vals = get_attention_tp_group().all_gather(vals, dim=1)
        assert self.index_topk == 2048
        # the `indices` lengths may less than 2048, so need topk with sort
        vals, raw_idxs = torch.topk(all_vals, self.index_topk)

        global_topk_idxs = all_inds.gather(-1, raw_idxs)
        return global_topk_idxs

    def forward_cuda(
        self,
        x: torch.Tensor,
        q_lora: torch.Tensor,
        index_k: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        layer_id: int,
        **kwargs,  # comm_manager: Optional[DecoderCommMananger] = None
    ) -> torch.Tensor | None:
        if TYPE_CHECKING:
            assert isinstance(forward_batch.token_to_kv_pool, DSATokenToKVPool)

        metadata = forward_batch.attn_backend.get_indexer_metadata(layer_id, forward_batch)
        # skip NSA if attention backend choose to skip this batch
        if metadata is None:
            return None

        query, _ = self.wq_b(q_lora)
        query = rearrange(query, "l (h d) -> l h d", d=self.head_dim)

        q_rope, _ = torch.split(query, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1)
        key = index_k
        key = self.k_norm(key)
        k_rope, _ = torch.split(key, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1)

        if not self.skip_rope and q_rope.shape[0] > 0:
            q_rope, k_rope = self.rotary_emb(positions, q_rope, k_rope)
            query[..., : self.rope_head_dim] = q_rope
            key[..., : self.rope_head_dim] = k_rope

        if CP_METADATA:
            key = cp_all_gather_torch_rerange_output(key.contiguous(), CP_METADATA.value, get_attention_tp_group())

        qd = query.shape[-1]
        kd = key.shape[-1]
        assert qd == kd and (qd & (qd - 1)) == 0, "Dimension must be a power of 2 for Hadamard transform."
        q_fp8, q_scale, k_fp8, k_scale = hadamard_transform_fuse_quant(query, key, qd**-0.5)

        # if CP_METADATA:
        #     k_fp8 = cp_all_gather_torch_rerange_output(k_fp8.contiguous(), CP_METADATA.value, get_attention_tp_group())
        #     k_scale = cp_all_gather_torch_rerange_output(k_scale.contiguous(), CP_METADATA.value, get_attention_tp_group())

        weights = self._get_logits_head_gate(x, q_scale)

        # deep_gemm mqa_logits not support heads less than 32
        if q_fp8.shape[1] < 32:
            # q_fp8: [len, nh, hd], weights: [len, nh, 1]
            pad_heads = 32 - q_fp8.shape[1]
            q_fp8 = torch.nn.functional.pad(q_fp8, [0, 0, 0, pad_heads])
            weights = torch.nn.functional.pad(weights, [0, 0, 0, pad_heads])

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

        if global_server_args_dict["enable_mla_l1_5_cache"]:
            if (
                forward_batch.forward_mode.is_decode_or_idle()
                or forward_batch.forward_mode.is_target_verify()
                or forward_batch.forward_mode.is_draft_extend()
            ):
                topk_result = self._get_topk_gather_topk(forward_batch, layer_id, q_fp8, weights, metadata)
            else:
                topk_result = self._get_topk_by_ag_kvcache(forward_batch, layer_id, q_fp8, weights, metadata)
        else:
            if (
                forward_batch.forward_mode.is_decode_or_idle()
                or forward_batch.forward_mode.is_target_verify()
                or forward_batch.forward_mode.is_draft_extend()
            ):
                topk_result = self._get_topk_paged(forward_batch, layer_id, q_fp8, weights, metadata)
            else:
                assert forward_batch.seq_lens_cpu is not None
                if CP_METADATA:
                    seqlens_expanded = metadata.get_seqlens_expanded()
                    topk_result = self._get_topk_ragged_cp(forward_batch, layer_id, q_fp8, weights, metadata, seqlens_expanded)
                else:
                    topk_result = self._get_topk_ragged(forward_batch, layer_id, q_fp8, weights, metadata)

        return topk_result

    def npu_indexer_dcp(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        actual_seq_lengths_q: torch.Tensor,
        layout_key: str,
        past_key_states: torch.Tensor,
        weights: torch.Tensor,
        forward_batch: ForwardBatch,
        layer_id: int,
        prolog_qkv: list[torch.Tensor] | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        # dcp
        block_table = forward_batch.attn_metadata.kvp_block_table
        kvp_group = get_attn_tp_group()
        kvp_rank = kvp_group.rank_in_group
        kvp_size = global_server_args_dict["kvp_size"]
        actual_seq_lengths_query = actual_seq_lengths_q.to(k.device).to(torch.int32)
        actual_seq_lengths_key = forward_batch.attn_metadata.kvp_context_lens_tensor.to(k.device).to(torch.int32)
        sparse_mode = 0
        if kvp_rank == 0:
            sparse_mode = 3
        init_cnt_req = forward_batch.attn_metadata.init_cnt_req
        local_cnt_req = forward_batch.attn_metadata.local_cnt_req
        topk_indices_local, topk_values = npu_ops.npu_lightning_indexer(
            q.view(-1, self.n_heads, self.head_dim),
            past_key_states,
            weights.to(torch.bfloat16),
            actual_seq_lengths_query=actual_seq_lengths_query,  # (B+1,)
            actual_seq_lengths_key=actual_seq_lengths_key,  # (B+1,)
            block_table=block_table,  # (B, max_blocks)
            layout_query="TND",
            layout_key=layout_key,
            init_tensor=init_cnt_req,
            local_tensor=local_cnt_req,
            sparse_count=self.index_topk,
            sparse_mode=sparse_mode,
            return_value=True,
        )
        T, D = topk_values.shape[0], topk_values.shape[2]

        q_nope_prolog, q_rope_prolog = None, None
        sfad_packed_query = None
        q_total, q_shape = None, None
        index_cache_write = None
        use_graph_cache_copy = False
        if kvp_rank == 0:
            if not global_server_args_dict["npu_disable_kv_nz"]:
                raise RuntimeError("NPU Indexer KVP cache update requires --npu-disable-kv-nz.")
            use_graph_cache_copy = (
                bool(forward_batch.can_run_with_graph)
                and forward_batch.all_decode_or_idle
                and ENV.npu_enable_graph
                and not ENV.npu_disable_graph_mode
            )

        # Keep the collective order identical on every KVP rank:
        # topk-values AllToAll -> query AllGather -> topk-position AllGather.
        with npu_super_kernel("", "", flag=forward_batch.can_run_with_graph), npu_limit_core("indexer_topk", flag=forward_batch.can_run_with_graph):
            # ---- all_to_all: 把 topk_values 沿 T 维切到各 rank ----
            # query 在 kvp 组内是复制的，所有 rank 的 T 相同，切分方案可本地确定。
            # sk 不支持 all2all_v，要求 T 能被 kvp_size 整除，走等长 all2all
            T_pad = (T + kvp_size - 1) // kvp_size * kvp_size
            T_local = T_pad // kvp_size
            if T_pad != T:
                pad_rows = torch.full(
                    (T_pad - T, *topk_values.shape[1:]),
                    float("-inf"),
                    dtype=topk_values.dtype,
                    device=topk_values.device,
                )
                topk_values = torch.cat([topk_values, pad_rows], dim=0)

            recv_val = torch.empty(kvp_size * T_local * D, dtype=topk_values.dtype, device=topk_values.device)
            torch.distributed.all_to_all_single(
                recv_val,
                topk_values.squeeze(1).reshape(-1),
                group=kvp_group.device_group,
            )
        # Query preparation and the graph cache update use the existing side
        # stream while topk-values AllToAll is in flight.
        with (
            npu_super_kernel(
                f"indexer_query_{self.layer_id}",
                "stream-fusion=1",
                flag=forward_batch.can_run_with_graph,
            ),
            npu_stream_switch(forward_batch.can_run_with_graph, "scmoe_steam"),
            npu_limit_core("indexer_kvp", flag=forward_batch.can_run_with_graph),
        ):
            if prolog_qkv is not None:
                q_nope_prolog, q_rope_prolog = prolog_qkv
                if forward_batch.can_run_with_graph:
                    q_nope_prolog = tng.scope.npu_wait_tensor(q_nope_prolog, topk_indices_local)
                q_total = torch.cat([q_nope_prolog, q_rope_prolog], dim=-1)
                q_shape = q_total.shape

            if use_graph_cache_copy:
                src_slots = forward_batch.attn_metadata.slot_mapping
                dst_slots = forward_batch.attn_metadata.kvp_current_slots_mapping
                with tng.scope.super_kernel(None, ""):
                    index_cache_write = indexer_cache_copy_rows_(
                        past_key_states,
                        src_slots,
                        dst_slots,
                        topk_indices_local,
                    )

            if q_total is not None:
                if forward_batch.can_run_with_graph:
                    q_total = tng.scope.npu_wait_tensor(q_total, recv_val)
                q_total = kvp_group.all_gather(q_total, dim=0)

        with npu_super_kernel("", "", flag=forward_batch.can_run_with_graph), npu_limit_core("indexer_topk", flag=forward_batch.can_run_with_graph):
            # 收到的 buffer 按 source rank 分块: [kvp_size, T_local, D]，
            # 转置后 [T_local, kvp_size*D]，列块 r 即 rank r 的 value
            all_val = recv_val.view(kvp_size, T_local, D).transpose(0, 1).contiguous().view(T_local, kvp_size * D)

            _, topk_pos = all_val.topk(self.index_topk, dim=1)
            topk_pos = topk_pos.to(torch.int32).contiguous()
            if q_total is not None and forward_batch.can_run_with_graph:
                topk_pos = tng.scope.npu_wait_tensor(topk_pos, q_total)
            topk_pos = kvp_group.all_gather(topk_pos, dim=0)[:T]

            topk_indices, valid_chunks = torch.ops.custom.npu_select_local_topk_indices(topk_indices_local, topk_pos, kvp_rank)
            forward_batch.topk_valid_chunks = valid_chunks

        with (
            npu_super_kernel(f"indexer_kvp_{self.layer_id}", "stream-fusion=1", flag=forward_batch.can_run_with_graph),
            npu_stream_switch(forward_batch.can_run_with_graph, "scmoe_steam"),
            npu_limit_core("indexer_kvp", flag=forward_batch.can_run_with_graph),
        ):
            if kvp_rank == 0 and not use_graph_cache_copy:
                dst_slots = forward_batch.attn_metadata.kvp_current_slots_mapping
                torch_npu.npu_scatter_nd_update_(
                    past_key_states.view(-1, 1, self.head_dim),
                    dst_slots.reshape(-1, 1),
                    k,
                )

            if q_total is not None:
                packed_query = q_total.view(kvp_size, *q_shape)
                use_packed_query = (
                    ENV.npu_enable_sfad_rank_major_query
                    and packed_query.shape[1] > 0
                    and packed_query.dim() in (4, 5)
                    and packed_query.shape[-1] == 576
                    and packed_query.shape[0] * packed_query.shape[-2] == 64
                    and (packed_query.dim() == 4 or packed_query.shape[2] == 1)
                    and packed_query.dtype in (torch.bfloat16, torch.float16)
                    and layout_key == "PA_BSND"
                )
                if use_packed_query:
                    sfad_packed_query = packed_query
                else:
                    q_total = packed_query.movedim(0, -3).flatten(-3, -2).contiguous()
                    q_nope_prolog, q_rope_prolog = torch.split(
                        q_total,
                        [
                            q_nope_prolog.shape[-1],
                            q_rope_prolog.shape[-1],
                        ],
                        dim=-1,
                    )

        if use_graph_cache_copy:
            topk_indices = tng.scope.npu_wait_tensor(topk_indices, index_cache_write)

        return (
            topk_indices,
            q_nope_prolog,
            q_rope_prolog,
            sfad_packed_query,
        )

    def npu_indexer_normal(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        layout_key: str,
        actual_seq_lengths_q: torch.Tensor,
        actual_seq_lengths_kv: torch.Tensor,
        past_key_states: torch.Tensor,
        weights: torch.Tensor,
        forward_batch: ForwardBatch,
        block_table: torch.Tensor,
        layer_id: int,
    ):
        torch.concat((torch.tensor([0], device=k.device, dtype=torch.int32), actual_seq_lengths_q.to(k.device).to(torch.int32)))
        cur_seq_lengths_key = torch.concat(
            (torch.tensor([0], device=k.device, dtype=torch.int32), actual_seq_lengths_kv.to(k.device).to(torch.int32))
        )

        bs = actual_seq_lengths_q.shape[0]
        init_tensor = torch.tensor([self.num_init_tokens] * bs, device=k.device, dtype=torch.int32)
        # local_tensor 只支持 per-token [T]: 逐行传 num_local(与逐请求定宽同语义, 常量展开)
        num_q_tokens = q.numel() // (self.n_heads * self.head_dim)
        local_tensor = torch.full((num_q_tokens,), self.num_local_tokens, device=k.device, dtype=torch.int32)
        actual_seq_lengths_kv = cur_seq_lengths_key[1:] - cur_seq_lengths_key[:-1]

        topk_indices, _ = npu_ops.npu_lightning_indexer(
            q.view(-1, self.n_heads, self.head_dim),
            past_key_states,
            weights.to(torch.bfloat16),
            actual_seq_lengths_query=actual_seq_lengths_q.to(k.device).to(torch.int32),
            actual_seq_lengths_key=actual_seq_lengths_kv.to(k.device).to(torch.int32),
            block_table=block_table,  # (B, max_blocks)
            layout_query="TND",
            layout_key=layout_key,
            sparse_count=self.index_topk,
            init_tensor=init_tensor,
            local_tensor=local_tensor,
            sparse_mode=3,
        )
        return topk_indices

    def forward_npu(
        self,
        x: torch.Tensor,
        q_lora: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        layer_id: int,
        prefill_after_input_redistribute_callback=None,
        prefill_between_input_and_core_callback=None,
        prefill_before_output_redistribute_callback=None,
    ) -> torch.Tensor:
        is_prefill = (
            forward_batch.forward_mode.is_extend()
            and not forward_batch.forward_mode.is_draft_extend()
            and not forward_batch.forward_mode.is_target_verify()
        )
        if not self.skip_rope:
            if forward_batch.nsa_rotary_emb is not None:
                cos, sin = forward_batch.nsa_rotary_emb
            else:
                if self.npu_disable_dsa_head_parallel:
                    slice_start, slice_end = get_attn_sp_token_slice_pos(forward_batch)
                    positions = positions[slice_start:slice_end]

                if hasattr(self.rotary_emb, "cos_sin_cache"):
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
        q = q.view(-1, self.n_heads, self.head_dim)  # [bs, 64, 128]
        q_pe, q_nope = torch.split(
            q,
            [self.rope_head_dim, self.head_dim - self.rope_head_dim],
            dim=-1,
        )  # [bs, 64, 64 + 64]

        if not self.skip_rope:
            q_pe = q_pe.view(-1, self.n_heads, 1, self.rope_head_dim)
            q_pe = torch_npu.npu_interleave_rope(q_pe, cos, sin).view(-1, self.n_heads, self.rope_head_dim)  # [bs, n, d]
            q = torch.cat([q_pe, q_nope], dim=-1)

        k_proj = self.wk(x)[0]  # [b, s, 7168] @ [7168, 128] = [b, s, 128]
        k = self.k_norm(k_proj).to(k_proj.dtype)
        k_pe, k_nope = torch.split(
            k,
            [self.rope_head_dim, self.head_dim - self.rope_head_dim],
            dim=-1,
        )  # [bs, 64 + 64]

        if not self.skip_rope:
            k_pe = k_pe.view(-1, 1, 1, self.rope_head_dim)
            k_pe = torch_npu.npu_interleave_rope(k_pe, cos, sin).view(bs, 1, self.rope_head_dim)  # [bs, 1, d]
            k = torch.cat([k_pe, k_nope.unsqueeze(1)], dim=-1)  # [bs, 1, 128]
        else:
            k = k.unsqueeze(1)
        if self.npu_disable_dsa_head_parallel:
            k = get_attn_tp_group().gather_tensor(k, forward_batch.global_sp_num_tokens)

        dense_slot_mapping = forward_batch.attn_metadata.dense_slot_mapping
        if dense_slot_mapping is not None:
            # l1.5 cache: slot_mapping 是 global 的，仅将本 rank 的有效 token 写入本地 index buffer
            dense_slot_mapping_idx = forward_batch.attn_metadata.dense_slot_mapping_idx
            if dense_slot_mapping_idx.numel() > 0:
                forward_batch.token_to_kv_pool.set_index_k_and_scale_buffer(layer_id, dense_slot_mapping, k[dense_slot_mapping_idx])
        else:
            forward_batch.token_to_kv_pool.set_index_k_and_scale_buffer(layer_id, slot_mapping, k)

        layout_key = "PA_BSND"
        if is_prefill:
            actual_seq_lengths_kv = forward_batch.attn_metadata.actual_seq_lengths_kv_tensor
            actual_seq_lengths_q = forward_batch.attn_metadata.actual_seq_lengths_tensor
        else:
            actual_seq_lengths_q = forward_batch.attn_metadata.query_len_tensor
            actual_seq_lengths_kv = forward_batch.attn_metadata.cu_seq_lens_tensor_int32

        past_key_states = forward_batch.token_to_kv_pool.get_index_k_with_scale_buffer(layer_id)
        block_table = block_table[: actual_seq_lengths_q.size()[0]] if is_prefill else block_table
        x = x.view(-1, self.hidden_size)
        weights = self.weights_proj(x.to(torch.float32))[0]
        index_k_route = getattr(forward_batch.attn_metadata, "index_k_route", None) if is_prefill else None
        if is_prefill and global_server_args_dict.get("enable_mla_l1_5_cache", False):
            # pcp
            attn_metadata = forward_batch.attn_metadata
            rank_counts = attn_metadata.per_rank_count
            inv_perm = attn_metadata.inv_perm
            if index_k_route is not None:
                past_key_states = index_k_route.exchange(past_key_states, get_attn_tp_group().device_group)
            else:
                past_key_states = past_key_states.reshape(-1, self.head_dim).index_select(0, attn_metadata.kv_index_list)
                past_key_states = get_attn_tp_group().gather_tensor(past_key_states, rank_counts).index_select(0, inv_perm).view(-1, 1, self.head_dim)

            block_table = None
            layout_key = "TND"

        use_dcp = global_server_args_dict["kvp_size"] > 1
        if not is_prefill and use_dcp:
            return self.npu_indexer_dcp(
                q,
                k,
                actual_seq_lengths_q,
                layout_key,
                past_key_states,
                weights,
                forward_batch,
                layer_id,
            )[0]

        enable_sp_for_indexer = (
            is_prefill and global_server_args_dict.get("enable_mla_l1_5_cache", False) and global_server_args_dict["npu_enable_sp_for_indexer"]
        )
        enable_level2_indexer = is_prefill and global_server_args_dict["npu_enable_level2_indexer"]
        is_pass = layer_id < (2 * global_server_args_dict["level2_pass_layers"])
        enable_level2_indexer = enable_level2_indexer and not is_pass

        # Use Level2 only when every request's current KV length exceeds both
        # the candidate capacity and the configured threshold. A long prompt's
        # first chunk can share a batch with another request's late chunk.
        # Keep that entire mixed batch on the dense mlp_lightning_indexer path.
        if enable_level2_indexer:
            level2_capacity = global_server_args_dict["level1_topk"] * global_server_args_dict["level2_block_page_len"]
            if forward_batch.seq_lens_cpu is not None and forward_batch.seq_lens_cpu.numel() > 0:
                min_kvseqlen = int(forward_batch.seq_lens_cpu.min().item())
            else:
                # Missing lengths cannot establish that every request qualifies.
                min_kvseqlen = 0
            if min_kvseqlen <= level2_capacity:
                enable_level2_indexer = False

            level2_min_kv_len = global_server_args_dict.get("level2_min_kv_len")
            if enable_level2_indexer and level2_min_kv_len is not None and min_kvseqlen <= level2_min_kv_len:
                enable_level2_indexer = False

        if not enable_sp_for_indexer:
            return self.npu_indexer_normal(
                q, k, layout_key, actual_seq_lengths_q, actual_seq_lengths_kv, past_key_states, weights, forward_batch, block_table, layer_id
            )
        else:
            if forward_batch.ib_local_extend_lens_with_zero is None:
                q = q.view(-1, self.n_heads, self.head_dim)
                local_extend_lens_with_zero = forward_batch.local_extend_lens_with_zero
                local_total_kv_lens_with_zero = forward_batch.local_total_kv_lens_with_zero
                cur_seq_lengths_query = forward_batch.cur_seq_lengths_query
                cur_seq_lengths_key = forward_batch.cur_seq_lengths_key
            else:
                h = self.n_heads * self.head_dim
                ib_input_split_sizes = [x * h for x in forward_batch.ib_input_split_sizes]
                ib_output_split_sizes = [x * h for x in forward_batch.ib_output_split_sizes]
                q_all2all = torch.empty(
                    [forward_batch.ib_attn_sp_token_nums[get_attn_tp_group().rank_in_group] * h],
                    dtype=q.dtype,
                    device=q.device,
                )
                torch.distributed.all_to_all_single(
                    q_all2all,
                    q.reshape(-1),
                    input_split_sizes=ib_input_split_sizes,
                    output_split_sizes=ib_output_split_sizes,
                    group=get_attn_tp_group().device_group,
                )
                q = q_all2all.view(-1, self.n_heads, self.head_dim)

                # all2all weights: each token has n_heads elements
                w_h = self.n_heads
                ib_w_input_split_sizes = [x * w_h for x in forward_batch.ib_input_split_sizes]
                ib_w_output_split_sizes = [x * w_h for x in forward_batch.ib_output_split_sizes]
                weights_all2all = torch.empty(
                    [forward_batch.ib_attn_sp_token_nums[get_attn_tp_group().rank_in_group] * w_h],
                    dtype=weights.dtype,
                    device=weights.device,
                )
                torch.distributed.all_to_all_single(
                    weights_all2all,
                    weights.reshape(-1),
                    input_split_sizes=ib_w_input_split_sizes,
                    output_split_sizes=ib_w_output_split_sizes,
                    group=get_attn_tp_group().device_group,
                )
                weights = weights_all2all.view(-1, self.n_heads)

                local_extend_lens_with_zero = forward_batch.ib_local_extend_lens_with_zero
                local_total_kv_lens_with_zero = forward_batch.ib_local_total_kv_lens_with_zero
                cur_seq_lengths_query = forward_batch.ib_cur_seq_lengths_query
                cur_seq_lengths_key = forward_batch.ib_cur_seq_lengths_key

            # Record only a lightweight dependency here.  Expensive Page-AG
            # host submission stays after the LightningIndexer enqueue, while
            # its communication stream can still wait for both input A2As.
            if prefill_after_input_redistribute_callback is not None:
                prefill_after_input_redistribute_callback()

            if local_extend_lens_with_zero.sum().item() > 0:
                if index_k_route is None:
                    past_key_states, _, _ = get_non_zero_q_meta(
                        past_key_states, forward_batch.global_cu_kv_lens_cpu, local_extend_lens_with_zero, local_total_kv_lens_with_zero
                    )
                if enable_level2_indexer:
                    from sglang.srt.layers.attention.dsa.lightning_indexer import block_pool_past_key_states

                    level1_size = global_server_args_dict["level2_kv_block_len"]
                    level1_topk = global_server_args_dict["level1_topk"]
                    block_page_size = global_server_args_dict["level2_block_page_len"]
                    kv_block_len = int(block_page_size // level1_size)
                    k_avg_npu, _, cur_block_lengths_key = block_pool_past_key_states(
                        past_key_states, cur_seq_lengths_key, level1_size, block_page_size, is_minmax=False
                    )
                    init_pages = ceil_div(self.num_init_tokens, block_page_size)  # 1
                    local_pages = self.num_local_tokens // block_page_size + 1  # 9
                    cur_q = q.view(-1, self.n_heads, self.head_dim)
                    # The current ascendc op uses the avg-pool version of the key,
                    # but its interface has not been updated to match, so q_pos/q_neg
                    # are not split here and cur_q / k_avg_npu are passed directly.
                    block_indcies, _ = npu_ops.mlp_compressed_lightning_indexer(
                        cur_q,
                        cur_q,
                        k_avg_npu,
                        k_avg_npu,
                        weights.to(torch.float32),
                        cur_seq_lengths_query=cur_seq_lengths_query,
                        cur_seq_lengths_key=cur_seq_lengths_key,
                        block_table=None,
                        layout_query="TND",
                        layout_key="TND",
                        sparse_count=level1_topk,
                        kv_block_len=kv_block_len,
                        q_block_len=1,
                        init_num=init_pages * kv_block_len,
                        local_num=local_pages * kv_block_len,
                        sparse_mode=3,
                        pre_tokens=2147483647,
                        next_tokens=2147483647,
                        return_value=True,
                    )
                    topk_indices, _ = npu_ops.mlp_sparse_lightning_indexer(
                        q.view(-1, self.n_heads, self.head_dim),
                        past_key_states,
                        weights.to(torch.float32),
                        cur_seq_lengths_query=cur_seq_lengths_query,
                        cur_seq_lengths_key=cur_seq_lengths_key,
                        sparse_index=block_indcies.squeeze(1),
                        layout_query="TND",
                        layout_key="TND",
                        sparse_count=self.index_topk,
                        init_num=self.num_init_tokens,
                        local_num=self.num_local_tokens,
                        sparse_index_block_size=block_page_size,
                        sparse_mode=3,
                        return_value=True,
                    )
                else:
                    topk_indices, _ = npu_ops.mlp_lightning_indexer(
                        q,
                        past_key_states,
                        weights,
                        cur_seq_lengths_query=cur_seq_lengths_query,  # (B+1,)
                        cur_seq_lengths_key=cur_seq_lengths_key,  # (B+1,)
                        block_table=block_table,  # (B, max_blocks)
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
                topk_indices = torch.empty((0, 1, self.index_topk), dtype=torch.int32, device=k.device)

            # Submit LightningIndexer to the compute stream first, then pay
            # the Page-AllGather host enqueue cost.  The collective is placed
            # on the staging stream and can execute under LightningIndexer;
            # both Indexer input A2As have already been submitted, so HCCL
            # ordering remains deterministic across ranks.
            if prefill_between_input_and_core_callback is not None:
                prefill_between_input_and_core_callback()

            # Enqueue a stream dependency after LightningIndexer and before
            # reverse TopK A2A.  This preserves one communicator order on all
            # ranks without host synchronization:
            # input A2A -> KV Page AG -> reverse TopK A2A.
            if prefill_before_output_redistribute_callback is not None:
                prefill_before_output_redistribute_callback()

            if forward_batch.ib_local_extend_lens_with_zero is not None:
                # reverse all2all: re-shard topk_indices from ib_attn_sp_token_nums back to attn_sp_token_nums
                topk_h = self.index_topk
                # reverse: new_splits=old_splits, old_splits=new_splits
                rev_input_split_sizes = [x * topk_h for x in forward_batch.ib_output_split_sizes]
                rev_output_split_sizes = [x * topk_h for x in forward_batch.ib_input_split_sizes]
                topk_all2all = torch.empty(
                    forward_batch.attn_sp_token_nums[get_attn_tp_group().rank_in_group] * topk_h,
                    dtype=topk_indices.dtype,
                    device=topk_indices.device,
                )
                torch.distributed.all_to_all_single(
                    topk_all2all,
                    topk_indices.reshape(-1),
                    input_split_sizes=rev_input_split_sizes,
                    output_split_sizes=rev_output_split_sizes,
                    group=get_attn_tp_group().device_group,
                )
                topk_indices = topk_all2all.view(-1, 1, self.index_topk)
            # topk_indices = get_attn_tp_group().all_gather(topk_indices, dim=0, output_split_sizes=attn_sp_token_nums)

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
        scale_fmt: str | None,
        block_size: int = 128,
        rope_scaling: dict[str, Any] | None = None,
        is_neox_style=False,
        prefix: str = "",
        config: PretrainedConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        alt_stream: torch.cuda.Stream | None = None,
        skip_rope: bool = False,
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
        self.skip_rope = skip_rope
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
        if is_npu():
            # On CUDA the wk weight lives in the attention's fused
            # fused_qkv_a_proj_with_mqa and index_k is passed in from there;
            # only the NPU path projects index_k inside the indexer.
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
        if not self.skip_rope:
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
        if global_server_args_dict["enable_dsa_qcp"]:
            self.comm_convertor = get_attn_tp_dp_convertor()
            assert self.comm_convertor is not None

    def _get_topk_ragged(
        self,
        forward_batch: ForwardBatch,
        layer_id: int,
        query: torch.Tensor,  # [l, h, d]
        weights: torch.Tensor,  # [l, h, 1]
        key: torch.Tensor,  # [l, d]
        metadata: NSAIndexerMetadata,
    ):
        assert query.shape[0] == key.shape[0]
        bs = forward_batch.batch_size
        block_tables = metadata.get_page_table_64()
        page_size = 64
        max_seq_len = block_tables.shape[1] * page_size
        all_qk_logits = torch.full((query.shape[0], max_seq_len), float("-inf"), dtype=torch.float32, device=query.device)
        attn_metadata = metadata.attn_metadata
        if layer_id == 0 or not hasattr(attn_metadata, "cu_seqlens_q_cpu"):
            attn_metadata.cu_seqlens_q_cpu = attn_metadata.cu_seqlens_q.cpu()
        for i in range(bs):
            q_st = attn_metadata.cu_seqlens_q_cpu[i]
            q_ed = attn_metadata.cu_seqlens_q_cpu[i + 1]
            q = query[q_st:q_ed]
            k = key[q_st:q_ed]
            w = weights[q_st:q_ed]

            l = q.shape[0]  # noqa: E741
            # [l, d] -> [h, l, d]
            k = k.unsqueeze(0).repeat_interleave(self.n_heads, 0)
            # [h, l, d] @ [h, d, l] -> [h, l, l]
            index_score = q.transpose(0, 1) @ k.transpose(-2, -1)
            # [h, l, l] -> [l, l]
            index_score = (w.transpose(0, 1) * F.relu(index_score)).sum(0)
            causal_mask = torch.triu(torch.full((l, l), float("-inf"), device=query.device), diagonal=1)
            index_score = index_score + causal_mask
            all_qk_logits[q_st:q_ed, : index_score.shape[1]] = index_score

        return all_qk_logits

    def _get_topk_paged(
        self,
        forward_batch: ForwardBatch,
        layer_id: int,
        query: torch.Tensor,  # [b, h, d]
        weights: torch.Tensor,  # [b, h, 1]
        metadata: NSAIndexerMetadata,
    ):
        if TYPE_CHECKING:
            assert isinstance(forward_batch.token_to_kv_pool, DSATokenToKVPool)

        page_size = 64
        block_tables = metadata.get_page_table_64()
        seqlens_32 = metadata.get_seqlens_expanded()
        index_k_cache = forward_batch.token_to_kv_pool.get_index_k_with_scale_buffer(layer_id=layer_id)
        index_k_cache = index_k_cache.view(index_k_cache.shape[0], page_size, -1)
        max_seq_len = block_tables.shape[1] * page_size
        bs = query.shape[0]
        attn_metadata = metadata.attn_metadata
        if not hasattr(attn_metadata, "aok_li_metadata"):
            attn_metadata.aok_li_metadata = aok_sfa.get_paged_lightning_indexer_metatdata(
                seqlens_32, page_size, torch.cuda.get_device_properties().multi_processor_count
            )
        logits = aok_sfa.paged_lightning_indexer(
            query.view(bs, -1, self.n_heads, self.head_dim),
            index_k_cache.unsqueeze(2),
            weights.view(bs, -1, self.n_heads),
            seqlens_32,
            block_tables,
            max_context_len=max_seq_len,
            schedule_meta=attn_metadata.aok_li_metadata,
            clean_logits=False,
        )
        return logits.view(query.shape[0], -1)

    def _get_topk_paged_extend(
        self,
        forward_batch: ForwardBatch,
        layer_id: int,
        query: torch.Tensor,  # [s, h, d]
        weights: torch.Tensor,  # [s, h, 1]
        metadata: NSAIndexerMetadata,
    ):
        if TYPE_CHECKING:
            assert isinstance(forward_batch.token_to_kv_pool, DSATokenToKVPool)
        page_size = 64
        block_tables = metadata.get_page_table_64()
        seq_lens_expanded = metadata.get_seqlens_expanded()
        index_k_cache = forward_batch.token_to_kv_pool.get_index_k_with_scale_buffer(layer_id=layer_id)
        index_k_cache = index_k_cache.view(index_k_cache.shape[0], page_size, -1)
        max_seq_len = block_tables.shape[1] * page_size
        weights = weights.squeeze(2)

        attn_metadata = metadata.attn_metadata
        if self.layer_id == 0 or not hasattr(attn_metadata, "aok_li_metadata"):
            attn_metadata.block_tables_expanded = block_tables.repeat_interleave(forward_batch.extend_seq_lens, dim=0)
            attn_metadata.aok_li_metadata = aok_sfa.get_paged_lightning_indexer_metatdata(
                seq_lens_expanded, page_size, torch.cuda.get_device_properties().multi_processor_count
            )
        logits = aok_sfa.paged_lightning_indexer(
            query.view(-1, 1, self.n_heads, self.head_dim),
            index_k_cache.unsqueeze(2),
            weights.view(-1, 1, self.n_heads),
            seq_lens_expanded,
            attn_metadata.block_tables_expanded,
            max_context_len=max_seq_len,
            schedule_meta=attn_metadata.aok_li_metadata,
            clean_logits=False,
        ).view(query.shape[0], -1)
        return logits

    def _get_score_extend_chunked(
        self,
        forward_batch: ForwardBatch,
        layer_id: int,
        query: torch.Tensor,  # [s, h, d]
        weights: torch.Tensor,  # [s, h, 1]
        metadata: NSAIndexerMetadata,
        comm_manager: DecoderCommMananger,
    ):
        if TYPE_CHECKING:
            assert isinstance(forward_batch.token_to_kv_pool, (DSATokenToKVPool, L1HalfDSATokenToKVPool))
        page_size = 64
        block_tables = metadata.get_page_table_64()
        page_table_1 = metadata.attn_metadata.page_table_1
        seqlens_32 = metadata.get_seqlens_int32()
        index_k_cache = forward_batch.token_to_kv_pool.get_index_k_with_scale_buffer(layer_id=layer_id)
        index_k_cache = index_k_cache.view(index_k_cache.shape[0], page_size, -1)
        bs = forward_batch.batch_size
        # token_to_batch = torch.arange(bs, dtype=torch.int32, device=query.device).repeat_interleave(forward_batch.extend_seq_lens)
        max_seq_len = block_tables.shape[1] * page_size
        weights = weights.squeeze(2)
        logits = torch.empty((query.shape[0], max_seq_len), dtype=torch.float32, device=query.device)
        attn_metadata = metadata.attn_metadata
        if self.layer_id == 0 or not hasattr(attn_metadata, "cp_gather_k_meta"):
            attn_metadata.seqlens_cpu = seqlens_32.cpu()
            attn_metadata.cp_gather_k_meta = [
                forward_batch.token_to_kv_pool.remap_token_idxs_for_gather(page_table_1[i][: attn_metadata.seqlens_cpu[i]]) for i in range(bs)
            ]
            attn_metadata.cu_seqlens_q_cpu = attn_metadata.cu_seqlens_q.cpu()

        for i in range(bs):
            # todo: multi seqs
            q_st = attn_metadata.cu_seqlens_q_cpu[i]
            q_ed = attn_metadata.cu_seqlens_q_cpu[i + 1]
            # [s, h, d]
            q = query[q_st:q_ed]
            # [s, h]
            w = weights[q_st:q_ed]
            local_indices, per_rank_count, inv_perm = attn_metadata.cp_gather_k_meta[i]
            local_k = (
                forward_batch.token_to_kv_pool.get_index_k_with_scale_buffer(self.layer_id).view(-1, self.head_dim).index_select(0, local_indices)
            )
            gathered_tensor = local_k.new_empty(attn_metadata.seqlens_cpu[i], local_k.shape[-1])
            tensor_list = list(gathered_tensor.split(per_rank_count))
            torch.distributed.all_gather(tensor_list, local_k, group=get_attention_tp_group().device_group)
            all_k = gathered_tensor[inv_perm]
            # result save to logits
            indexer_qk_rag(q, w, all_k, logits[q_st:q_ed])

        return logits

    def _get_topk_gather_topk(
        self,
        forward_batch: ForwardBatch,
        layer_id: int,
        query: torch.Tensor,  # [s, h, d]
        weights: torch.Tensor,  # [s, h, 1]
        metadata: NSAIndexerMetadata,
        comm_manager: DecoderCommMananger,
    ):
        if TYPE_CHECKING:
            assert isinstance(forward_batch.token_to_kv_pool, (DSATokenToKVPool, L1HalfDSATokenToKVPool))
        page_size = 64
        cp_l1half_meta = metadata.attn_metadata.cp_l1half_meta
        local_seqlens_k = cp_l1half_meta.local_seqlens_k
        local_table_64 = cp_l1half_meta.local_page_table_64
        global_valid_indices = cp_l1half_meta.global_valid_indices
        swa_mask = cp_l1half_meta.swa_mask
        index_k_cache = forward_batch.token_to_kv_pool.get_index_k_with_scale_buffer(layer_id=layer_id)
        index_k_cache = index_k_cache.view(index_k_cache.shape[0], page_size, -1)
        max_seq_len = local_table_64.shape[1] * page_size
        weights = weights.squeeze(2)

        # every q token is treated as an independent token for calculation, ignore of s_q
        logits = aok_sfa.paged_lightning_indexer(
            query.view(-1, 1, self.n_heads, self.head_dim),
            index_k_cache,
            weights.view(-1, 1, self.n_heads),
            local_seqlens_k,
            local_table_64,
            max_context_len=max_seq_len,
            clean_logits=False,
        )
        logits = logits.view(query.shape[0], -1)
        if swa_mask is not None:
            logits = torch.where(swa_mask, float("inf"), logits)

        indices, vals = flashinfer.flexible_topk_trans_and_out_val(
            logits,
            global_valid_indices,  # global indices
            local_seqlens_k,
        )
        all_inds = get_attention_tp_group().all_gather(indices, dim=1)
        all_vals = get_attention_tp_group().all_gather(vals, dim=1)
        assert self.index_topk == 2048
        # the `indices` lengths may less than 2048, so need topk with sort
        vals, raw_idxs = torch.topk(all_vals, self.index_topk)

        global_topk_idxs = all_inds.gather(-1, raw_idxs)
        return global_topk_idxs

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
        comm_manager = kwargs["comm_manager"]
        metadata = forward_batch.attn_backend.get_indexer_metadata(layer_id, forward_batch)

        # skip NSA if attention backend choose to skip this batch
        if metadata is None:
            return None

        if not NSA_USE_REAL_INDEXER:  # temporary
            return self._forward_fake(x, q_lora, positions, forward_batch, layer_id)

        query, _ = self.wq_b(q_lora)
        query = rearrange(query, "l (h d) -> l h d", d=self.head_dim)

        q_rope, _ = torch.split(query, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1)
        key = index_k
        key = self.k_norm(key)
        k_rope, _ = torch.split(key, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1)
        if not self.skip_rope and q_rope.shape[0] > 0:
            q_rope, k_rope = self.rotary_emb(positions, q_rope, k_rope)
            query[..., : self.rope_head_dim] = q_rope
            key[..., : self.rope_head_dim] = k_rope

        if CP_METADATA:
            key = cp_all_gather_rerange_output(key.contiguous(), CP_METADATA.value, self.comm_convertor)

        out_cache_loc = forward_batch.out_cache_loc
        forward_batch.token_to_kv_pool.set_index_k_and_scale_buffer(
            layer_id=layer_id,
            loc=out_cache_loc,
            index_k=key,
        )

        weights, _ = self.weights_proj(x.float())
        weights = weights * self.n_heads**-0.5
        weights = weights.unsqueeze(-1) * self.softmax_scale

        return self.qk_and_topk_dispatcher(forward_batch, layer_id, query, key, weights, comm_manager)

    def qk_and_topk_dispatcher(
        self,
        forward_batch: ForwardBatch,
        layer_id: int,
        query: torch.Tensor,
        key: torch.Tensor,
        weights: torch.Tensor,
        comm_manager: DecoderCommMananger,
    ):
        if TYPE_CHECKING:
            assert isinstance(forward_batch.attn_backend, DpskSparseAttnBackend)
        metadata = forward_batch.attn_backend.get_indexer_metadata(layer_id, forward_batch)
        # if is_cuda():
        topk_result = None
        if global_server_args_dict["enable_mla_l1_5_cache"]:
            if (
                forward_batch.forward_mode.is_decode_or_idle()
                or forward_batch.forward_mode.is_target_verify()
                or forward_batch.forward_mode.is_draft_extend()
            ):
                topk_result = self._get_topk_gather_topk(forward_batch, layer_id, query, weights, metadata, comm_manager)
                # decode can also use `_get_score_extend_chunked` for debug
            else:
                assert forward_batch.seq_lens_cpu is not None
                qk_logits = self._get_score_extend_chunked(forward_batch, layer_id, query, weights, metadata, comm_manager)
                if global_server_args_dict["enable_dsa_qcp"]:
                    topk_result = get_rag_topk(qk_logits, metadata.get_seqlens_expanded(), self.num_init_tokens, self.num_local_tokens)
        else:
            if (
                forward_batch.forward_mode.is_decode_or_idle()
                or forward_batch.forward_mode.is_target_verify()
                or forward_batch.forward_mode.is_draft_extend()
            ):
                qk_logits = self._get_topk_paged(forward_batch, layer_id, query, weights, metadata)
            else:
                assert forward_batch.seq_lens_cpu is not None
                attn_metadata = metadata.attn_metadata
                if self.layer_id == 0 or not hasattr(attn_metadata, "is_extend_rag"):
                    attn_metadata.is_extend_rag = torch.all(forward_batch.extend_prefix_lens == 0).item()
                if attn_metadata.is_extend_rag:
                    qk_logits = self._get_topk_ragged(forward_batch, layer_id, query, weights, key, metadata)
                else:
                    qk_logits = self._get_topk_paged_extend(forward_batch, layer_id, query, weights, metadata)
        if topk_result is None:
            topk_result = metadata.topk_transform(
                qk_logits,
                self.index_topk,
                num_init_and_local_tokens=[self.num_init_tokens, self.num_local_tokens],
            )
        return topk_result


def get_rag_topk(
    logits,
    seq_lens_expanded,
    num_init_tokens,
    num_local_tokens,
    row_starts=None,
):
    lengths = seq_lens_expanded
    if row_starts is None:
        row_starts = lengths.new_zeros(logits.shape[0])
    if num_init_tokens > 0:
        init_idxs = row_starts[:, None] + torch.arange(num_init_tokens, dtype=lengths.dtype, device=lengths.device)[None, :]
        init_idxs.clamp_max_(logits.shape[-1] - 1)
        logits.scatter_(dim=1, index=init_idxs.expand(logits.shape[0], -1), value=float("inf"))
    if num_local_tokens > 0:
        local_idxs = (row_starts + lengths)[:, None] - 1 - torch.arange(num_local_tokens, dtype=lengths.dtype, device=lengths.device)[None, :]
        local_idxs.clamp_min_(0)
        logits.scatter_(dim=1, index=local_idxs, value=float("inf"))
    topk_result = flashinfer.flexible_topk(logits, seq_lens_expanded, row_starts)
    return topk_result
