from __future__ import annotations

# ============================================================
# Vision Attention Module
#
# Adapted from sglang's vision.py, with import paths modified
# for the fluentllm project.
#
# Key differences from upstream sglang:
# - `sglang.srt.environ` → `sglang.srt.env`
# - `sglang.srt.layers.attention.dp_attention` → `sglang.srt.layers.attention.dp_attention`
# - `sglang.srt.models.utils.apply_qk_norm` → inlined
# - `sglang.jit_kernel.norm.can_use_fused_inplace_qknorm` → stubbed (always False)
# - `sglang.srt.utils.multi_stream_utils` → inlined simplified version
# - `is_blackwell_supported` → derived from `get_device_capability`
# - `RMSNorm` → `sglang.srt.ops.layernorm.RMSNorm` (simplified params)
# - `apply_rotary_pos_emb` → `sglang.srt.layers.embedding.rotary_embedding`
# - `server_args.mm_attention_backend` → read from `global_server_args_dict` (env.py)
# ============================================================
import dataclasses
import functools
import math
from contextlib import contextmanager, suppress
from functools import lru_cache, partial
from typing import Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from sglang.srt.env import envs, global_server_args_dict
from sglang.srt.layers.attention.dp_attention import (
    get_attention_tp_group,
    get_attention_tp_rank,
    get_attention_tp_size,
)
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from sglang.srt.layers.quantization import QuantizationConfig
from sglang.srt.ops.layernorm import RMSNorm
from sglang.srt.utils import (
    add_prefix,
    divide,
    get_device_capability,
    is_cuda,
    is_npu,
    print_info_once,
    split_tensor_along_last_dim,
)

_is_cuda = is_cuda()
_is_npu = is_npu()

# ------------------------------------------------------------------
# Optional heavy imports guarded by platform checks
# ------------------------------------------------------------------
if _is_cuda:
    try:
        # TODO benchmark flashinfer_cudnn, fa3
        from flashinfer.prefill import cudnn_batch_prefill_with_kv_cache

        _has_flashinfer_cudnn = True
    except ImportError:
        _has_flashinfer_cudnn = False

    try:
        from flash_attn_3.flash_attn_interface import flash_attn_varlen_func

        _has_flash_attn = True
    except ImportError:
        _has_flash_attn = False

    try:
        from flash_attn import flash_attn_varlen_func as flash_attn_varlen_func_v2

        _has_flash_attn_2 = True
    except ImportError:
        _has_flash_attn_2 = False

else:
    _has_flashinfer_cudnn = False
    _has_flash_attn = False
    _has_flash_attn_2 = False


try:
    from sglang.srt.layers.attention.triton_ops.prefill_attention import (
        context_attention_fwd,
    )

    _has_triton_attn = True
except ImportError:
    context_attention_fwd = None
    _has_triton_attn = False


if _is_npu:
    with suppress(ImportError):
        import torch_npu


try:
    from sglang.srt.layers.embedding.rotary_embedding import (
        apply_rope_triton,
        can_use_triton_rope,
    )
    from sglang.srt.layers.embedding.rotary_embedding import (
        apply_rotary_pos_emb_native as apply_rotary_pos_emb,
    )
except ImportError:
    # Fallback: define a minimal pure-torch implementation
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        q_embed = (q * cos) + (_rotate_half(q) * sin)
        k_embed = (k * cos) + (_rotate_half(k) * sin)
        return q_embed, k_embed

    def can_use_triton_rope(q, cos, head_size):
        return False

    def apply_rope_triton(q, k, cos, sin, head_size):
        raise RuntimeError("Triton RoPE not available")


def _is_blackwell_supported() -> bool:
    """Check if current GPU is Blackwell (SM100)."""
    if not _is_cuda:
        return False
    try:
        major, minor = get_device_capability()
        return major >= 10
    except Exception:
        return False


# ------------------------------------------------------------------
# Stub for jit qk norm (not available in fluentllm)
# ------------------------------------------------------------------
def _can_use_jit_qk_norm(head_size: int, dtype: torch.dtype) -> bool:
    """fluentllm does not ship the JIT qk-norm kernel; always use fallback."""
    return False


# ------------------------------------------------------------------
# Inline multi-stream utilities (simplified, no-op when no aux stream)
# ------------------------------------------------------------------
@contextmanager
def _with_multi_stream(enabled: bool):
    yield


def _maybe_execute_in_parallel(fn_a, fn_b, events, aux_stream):
    """Execute fn_a and fn_b, optionally in parallel on aux_stream."""
    if aux_stream is not None:
        # Execute fn_b on aux stream, fn_a on main stream
        event_a, event_b = events[0], events[1]
        result_a = fn_a()
        event_a.record()
        with torch.cuda.stream(aux_stream):
            result_b = fn_b()
            event_b.record()
        # Sync: make sure main stream waits for aux result
        torch.cuda.current_stream().wait_event(event_b)
    else:
        result_a = fn_a()
        result_b = fn_b()
    return result_a, result_b


# ------------------------------------------------------------------
# Inline apply_qk_norm (simplified version)
# ------------------------------------------------------------------
def _apply_qk_norm_jit(
    q: torch.Tensor,
    k: torch.Tensor,
    q_norm: nn.Module,
    k_norm: nn.Module,
    head_dim: int,
    alt_stream=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply QK normalization (simplified, no JIT kernel path)."""
    q_ = q.flatten(1, 2) if q.dim() == 3 else q.reshape(q.shape[0], -1)
    k_ = k.flatten(1, 2) if k.dim() == 3 else k.reshape(k.shape[0], -1)
    q_ = q_norm(q_)
    k_ = k_norm(k_)
    return q_.view(q.shape), k_.view(k.shape)


# ------------------------------------------------------------------
# Vision-specific rotary embedding helper (QwenVL-style freqs tensor)
# ------------------------------------------------------------------
def _rotate_half_simple(x: torch.Tensor) -> torch.Tensor:
    """Rotate the tensor by half for rotary embeddings."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_emb_torch_simple(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    Apply rotary embeddings (pure PyTorch).
    x: (..., headdim)
    cos, sin: (..., rotary_dim)
    """
    ro_dim = cos.shape[-1] * 2
    assert ro_dim <= x.shape[-1]
    cos = torch.cat([cos, cos], dim=-1)  # (..., rotary_dim * 2)
    sin = torch.cat([sin, sin], dim=-1)
    return torch.cat(
        [
            x[..., :ro_dim] * cos + _rotate_half_simple(x[..., :ro_dim]) * sin,
            x[..., ro_dim:],
        ],
        dim=-1,
    )


def _apply_rotary_pos_emb_vision(t: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """
    Apply rotary position embeddings for vision (QwenVL-style).
    t: (..., headdim)
    freqs: (..., rotary_dim) - raw frequencies (not yet cos/sin)
    """
    t_ = t.float()
    cos = freqs.cos()
    sin = freqs.sin()
    output = _apply_rotary_emb_torch_simple(t_, cos, sin).type_as(t)
    return output


# ------------------------------------------------------------------
# ROTARY_EMBED_CLASSES registry
# ------------------------------------------------------------------
ROTARY_EMBED_CLASSES = {
    "normal": apply_rotary_pos_emb,
}


# === Vision Encoder === #
FLASHINFER_WORKSPACE_SIZE_BYTES = 128 * 1024 * 1024

# Batch buckets for cuDNN graph caching
BATCH_BUCKETS = [8, 16, 32, 64]

# Bucketized max seqlens to reduce cuDNN recompilation frequency
FLASHINFER_MAX_SEQLEN_BUCKETS = [
    4 * 1024,
    8 * 1024,
    16 * 1024,
    32 * 1024,
    64 * 1024,
    128 * 1024,
]


@dataclasses.dataclass
class SingletonCache:
    data: Any = None

    def set_data(self, value: Any) -> None:
        self.data = value

    def get_data(self) -> Any | None:
        return self.data

    def empty(self) -> bool:
        return self.get_data() is None


@functools.lru_cache(maxsize=128)
def _get_cu_seqlens_for_shape(batch_size: int, seqlen: int, device) -> torch.Tensor:
    """
    Generates cumulative sequence lengths (cu_seqlens) for a given batch_size, seqlen, and device.
    Caches the result based on these parameters.
    """
    cu_seqlens = torch.arange(
        0,
        (batch_size + 1) * seqlen,
        step=seqlen,
        dtype=torch.int32,
        device=device,
    )
    return cu_seqlens


def resolve_seqlens(
    cu_seqlens: torch.Tensor | SingletonCache | None,
    bsz: int,
    seq_len: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    if cu_seqlens is None:
        resolved_seqlens = _get_cu_seqlens_for_shape(bsz, seq_len, device=device)
    elif isinstance(cu_seqlens, SingletonCache):
        if cu_seqlens.empty():
            cu_seqlens.set_data(_get_cu_seqlens_for_shape(bsz, seq_len, device=device))
        resolved_seqlens = cu_seqlens.get_data()
    else:
        resolved_seqlens = cu_seqlens
    assert isinstance(resolved_seqlens, torch.Tensor), "cu_seqlens must be a torch.Tensor"
    return resolved_seqlens


class VisionSdpaAttention(nn.Module):
    r"""
    Scaled Dot Product Attention inner product

    """

    def __init__(
        self,
        head_dim: int,
        num_heads: int,
        num_kv_heads: int,
        dropout: float = 0.0,
        flatten_batch: bool = False,
        softmax_in_single_precision: bool = False,
        softmax_scale: float | None = None,
        **kwargs,
    ):
        super().__init__()
        self.head_size = head_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.flatten_batch = flatten_batch
        self.softmax_in_single_precision = softmax_in_single_precision
        self.dropout = dropout
        self.scale = softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(self.head_size)

    @staticmethod
    @lru_cache(maxsize=128)
    def _generate_mask_cache(s: int, flatten_batch: bool, cu_seqlens: tuple) -> torch.BoolTensor:
        """
        Generate a boolean attention mask with caching mechanism.
        Args:
            s: sequence length
            flatten_batch: whether to flatten batch dimension
            cu_seqlens: tuple of cumulative sequence lengths
        Returns:
            attention mask tensor of shape [b, 1, s, s] or [1, s, s]
        """
        if flatten_batch:
            mask = torch.zeros([1, s, s], dtype=torch.bool)
            for i in range(1, len(cu_seqlens)):
                start = cu_seqlens[i - 1]
                end = cu_seqlens[i]
                mask[..., start:end, start:end] = True
        else:
            # [1, 1, 1, s]
            row_indices = torch.arange(s).view(1, 1, 1, s)
            # [1, 1, s, 1]
            col_indices = torch.arange(s).view(1, 1, s, 1)
            # [b, 1, 1, 1]
            seq_lens = torch.tensor(
                [end - start for start, end in zip(cu_seqlens[:-1], cu_seqlens[1:])],
            ).view(-1, 1, 1, 1)

            mask = (row_indices < seq_lens) & (col_indices < seq_lens)

        return mask

    def generate_patch_attention_mask(
        self,
        s: int,
        cu_seqlens: torch.Tensor | None,
        flatten_batch: bool = False,
    ) -> torch.Tensor | None:
        r"""
        Creates a non-causal 4D mask of shape `(b, 1, s, s)` or `(1, 1, s, s)`.
        Args:
            s: sequence length
            cu_seqlens: cumulative sequence lengths tensor. If not, returns an empty mask
            flatten_batch: whether to flatten batch dimension
        Returns:
            attention mask tensor or None
        """
        if cu_seqlens is None:
            return None

        cu_seqlens_tuple = tuple(cu_seqlens.cpu().tolist())

        return self._generate_mask_cache(s, flatten_batch, cu_seqlens_tuple)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        bsz: int,
        cu_seqlens: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        softmax_scale: float | None = None,
        **kwargs,
    ) -> torch.Tensor:
        r"""
        Args:
            cu_seqlens: [b]
        Returns:
             [b * s, h, head_size]
        """
        if self.flatten_batch:
            assert bsz == 1, "flatten_batch is True, bsz must be 1"

        assert q.dim() == 3, q.shape

        s = q.shape[0] // bsz

        # [b, 1, s, s]
        if attention_mask is None:
            attention_mask = self.generate_patch_attention_mask(s, cu_seqlens, flatten_batch=self.flatten_batch)

        if attention_mask is None:
            if self.softmax_in_single_precision:
                raise RuntimeError("Empty attention mask")
        else:
            attention_mask = attention_mask.to(device=q.device)

        q, k, v = [rearrange(x, "(b s) h d -> b h s d", b=bsz) for x in [q, k, v]]

        if self.softmax_in_single_precision:
            k = rearrange(k, "b h s d -> b h d s")
            attn_weights = torch.matmul(q, k) * self.scale
            del k
            # masking
            attention_mask = (~attention_mask) * torch.finfo(q.dtype).min
            attn_weights = attn_weights + attention_mask
            del attention_mask
            # full-precision
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
            attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=False)
            output = torch.matmul(attn_weights, v)
            del attn_weights, v
        else:
            # SDPA
            # [b, h, s, head_size]
            output = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attention_mask,
                dropout_p=self.dropout,
                is_causal=False,
                scale=self.scale,
            )

        # [b, h, s, head_size] --> [b * s, h, head_size]
        output = rearrange(output, "b h s d -> (b s) h d")

        return output


class VisionTritonAttention(nn.Module):
    """
    Triton-implemented attention without a causal mask
    """

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__()
        use_data_parallel = kwargs.get("use_data_parallel", False)
        self.tp_size = 1 if use_data_parallel else get_attention_tp_size()

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens: torch.Tensor | SingletonCache | None,
        bsz: int,
        seq_len: int,
        softmax_scale: float | None = None,
        **kwargs,
    ) -> torch.Tensor:
        r"""
        Args:
            cu_seqlens: [b]
            softmax_scale: override softmax scale (default 1/sqrt(head_dim))
        Returns:
             [b * s, h, head_size]
        """
        if isinstance(cu_seqlens, list):
            # CUDA-graph path: cu_seqlens = [cu_indptr, b_seq_len, max_seqlen]
            if "output_ws" not in kwargs:
                raise RuntimeError("output_ws should be prepared for cuda-graph mode")

            output = kwargs["output_ws"]
            context_attention_fwd(
                q,
                k,
                v,
                output,
                cu_seqlens[0],
                cu_seqlens[1],
                cu_seqlens[2],
                is_causal=False,
                sm_scale=softmax_scale,
            )
        else:
            # Eager path (or CG fallback): cu_seqlens is a plain 1D int32 tensor.
            cu_seqlens = resolve_seqlens(cu_seqlens, bsz, seq_len, device=q.device)

            # [b * s, head, head_size]
            output = torch.empty_like(q)

            seq_lens = cu_seqlens[1:] - cu_seqlens[:-1]
            max_seqlen = seq_lens.max().item()
            context_attention_fwd(
                q,
                k,
                v,
                output,
                cu_seqlens.to(q.device),
                seq_lens.to(q.device),
                max_seqlen,
                is_causal=False,
                sm_scale=softmax_scale,
            )

        return output


class VisionFlash3Attention(nn.Module):
    def __init__(
        self,
        **kwargs,
    ):
        if not _is_cuda:
            raise Exception("VisionFlash3Attention is only available for cuda")
        if not _has_flash_attn:
            raise Exception("VisionFlash3Attention requires sglang.jit_kernel.flash_attention")
        super().__init__()
        use_data_parallel = kwargs.get("use_data_parallel", False)
        self.tp_size = 1 if use_data_parallel else get_attention_tp_size()

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens: torch.Tensor | SingletonCache | None,
        bsz: int,
        seq_len: int,
        softmax_scale: float | None = None,
        **kwargs,
    ) -> torch.Tensor:
        r"""
        Args:
            cu_seqlens: [b]
        Returns:
             [b * s, h, head_size]
        """
        if isinstance(cu_seqlens, list):
            # CUDA-graph path: cu_seqlens = [cu_seqlens_tensor, max_seqlen]
            max_seqlen = cu_seqlens[1]
            output = flash_attn_varlen_func(
                q,
                k,
                v,
                cu_seqlens_q=cu_seqlens[0],
                cu_seqlens_k=cu_seqlens[0],
                max_seqlen_q=max_seqlen,
                max_seqlen_k=max_seqlen,
                softmax_scale=softmax_scale,
            )
        else:
            # Eager path (or CG fallback): cu_seqlens is a plain 1D int32 tensor.
            cu_seqlens = resolve_seqlens(cu_seqlens, bsz, seq_len, device=q.device)
            cu_seqlens = cu_seqlens.to(dtype=torch.int32).to(q.device)
            seq_lens = cu_seqlens[1:] - cu_seqlens[:-1]
            max_seqlen = seq_lens.max().item()

            output = flash_attn_varlen_func(
                q,
                k,
                v,
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seqlen,
                max_seqlen_k=max_seqlen,
                softmax_scale=softmax_scale,
            )

        return output


class VisionFlash4Attention(nn.Module):
    def __init__(
        self,
        **kwargs,
    ):
        if not _is_cuda:
            raise Exception("VisionFlash4Attention is only available for cuda")
        if not _has_flash_attn:
            raise Exception("VisionFlash4Attention requires sglang.jit_kernel.flash_attention")
        super().__init__()

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens: torch.Tensor | SingletonCache | None,
        bsz: int,
        seq_len: int,
        softmax_scale: float | None = None,
        **kwargs,
    ) -> torch.Tensor:
        r"""
        Args:
            cu_seqlens: [b]
        Returns:
             [b * s, h, head_size]
        """
        if cu_seqlens is None:
            cu_seqlens = _get_cu_seqlens_for_shape(bsz, seq_len, device=q.device)
        elif isinstance(cu_seqlens, SingletonCache):
            if cu_seqlens.empty():
                cu_seqlens.set_data(_get_cu_seqlens_for_shape(bsz, seq_len, device=q.device))
            cu_seqlens = cu_seqlens.get_data()

        cu_seqlens = cu_seqlens.to(dtype=torch.int32).to(q.device)
        seq_lens = cu_seqlens[1:] - cu_seqlens[:-1]
        max_seqlen = seq_lens.max().item()

        output = flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            softmax_scale=softmax_scale,
            ver=4,
        )

        return output


class VisionFlashInferAttention(nn.Module):
    def __init__(
        self,
        **kwargs,
    ):
        if not _is_cuda:
            raise Exception("VisionFlashInferAttention is only available for cuda")
        if not _has_flashinfer_cudnn:
            raise Exception("VisionFlashInferAttention requires flashinfer.prefill.cudnn_batch_prefill_with_kv_cache")
        super().__init__()
        self.workspace_buffer = kwargs.get("workspace_buffer")

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens: torch.Tensor | SingletonCache | None,
        bsz: int,
        seq_len: int,
        softmax_scale: float | None = None,
        **kwargs,
    ) -> torch.Tensor:
        r"""
        Args:
            cu_seqlens: [b]
        Returns:
             [b * s, h, head_size]
        """
        if "sequence_lengths" not in kwargs:
            raise RuntimeError("sequence_lengths should be prepared for vision flashinfer_cudnn attention backend")
        if "max_seqlen" not in kwargs:
            raise RuntimeError("max_seqlen should be prepared for vision flashinfer_cudnn attention backend")

        sequence_lengths = kwargs["sequence_lengths"]
        max_seqlen = kwargs["max_seqlen"]

        # max_seqlen must be python int
        if isinstance(max_seqlen, torch.Tensor):
            if max_seqlen.is_cuda:
                max_seqlen = int(max_seqlen.detach().cpu().item())
            else:
                max_seqlen = int(max_seqlen.item())
        else:
            max_seqlen = int(max_seqlen)

        # flatten if caller gives (b, s, h, d)
        is_reshaped = q.dim() == 4
        if is_reshaped:
            reshape_batch_size = q.shape[0]
            q, k, v = (rearrange(x, "b s ... -> (b s) ...") for x in [q, k, v])

        if not isinstance(cu_seqlens, torch.Tensor):
            raise RuntimeError("flashinfer_cudnn expects packed indptrs as a torch.Tensor")

        if not isinstance(sequence_lengths, torch.Tensor):
            raise RuntimeError("sequence_lengths must be a torch.Tensor")
        seq_lens_1d = sequence_lengths.view(-1).to(device=q.device, dtype=torch.int32)
        B = int(seq_lens_1d.numel())

        cu_seqlens_1d = cu_seqlens.view(-1).to(device=q.device, dtype=torch.int32)
        expected = 3 * (B + 1)
        if int(cu_seqlens_1d.numel()) != expected:
            raise RuntimeError(f"packed indptr numel mismatch: got {cu_seqlens_1d.numel()}, expected {expected} (= 3*(B+1))")

        split = B + 1
        indptr_qk = cu_seqlens_1d[:split].view(split, 1, 1, 1)
        indptr_v = cu_seqlens_1d[split : 2 * split].view(split, 1, 1, 1)
        indptr_o = cu_seqlens_1d[2 * split :].view(split, 1, 1, 1)

        seq_lens_4d = seq_lens_1d.view(B, 1, 1, 1)

        token_width_q = int(q.shape[1] * q.shape[2])
        total_elems_q = int(q.numel())

        start_elems = indptr_qk.view(-1)[:-1]
        end_elems = start_elems + seq_lens_1d * token_width_q
        if (end_elems > total_elems_q).any():
            raise RuntimeError("offset + len out of bounds; packed indptr is wrong")

        _, _, head_size = q.shape
        scale = softmax_scale if softmax_scale is not None else head_size**-0.5

        output, _ = cudnn_batch_prefill_with_kv_cache(
            q,
            k,
            v,
            scale,
            self.workspace_buffer,
            max_token_per_sequence=max_seqlen,
            max_sequence_kv=max_seqlen,
            actual_seq_lens_q=seq_lens_4d,
            actual_seq_lens_kv=seq_lens_4d,
            causal=False,
            return_lse=True,
            batch_offsets_q=indptr_qk,
            batch_offsets_k=indptr_qk,
            batch_offsets_v=indptr_v,
            batch_offsets_o=indptr_o,
            is_cuda_graph_compatible=True,
        )

        if is_reshaped:
            output = rearrange(output, "(b s) h d -> b s h d", b=reshape_batch_size)

        return output


class VisionAscendAttention(nn.Module):
    def __init__(
        self,
        **kwargs,
    ):
        if not _is_npu:
            raise Exception("VisionAscendAttention is only available for ascend npu")
        super().__init__()

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens: torch.Tensor | SingletonCache | None,
        bsz: int,
        seq_len: int,
        softmax_scale: float | None = None,
        **kwargs,
    ) -> torch.Tensor:
        r"""
        Args:
            cu_seqlens: [b]
        Returns:
             [b * s, h, head_size]
        """
        if envs.SGLANG_VIT_ENABLE_CUDA_GRAPH.get():
            if "output_ws" not in kwargs:
                raise RuntimeError("output_ws should be prepared for npu-graph mode")
            output = kwargs["output_ws"]
            seq_len_arg = cu_seqlens
        else:
            cu_seqlens = resolve_seqlens(cu_seqlens, bsz, seq_len, device="cpu")
            seq_lens = cu_seqlens[1:] - cu_seqlens[:-1]
            if seq_lens.is_npu:
                seq_lens = seq_lens.to("cpu")
            output = torch.empty_like(q)
            seq_len_arg = seq_lens.to(torch.int32)

        _, num_heads, head_size = q.shape
        num_kv_heads = k.shape[1]

        scale_value = softmax_scale if softmax_scale is not None else head_size**-0.5

        torch_npu._npu_flash_attention_unpad(
            query=q,
            key=k,
            value=v,
            seq_len=seq_len_arg,
            scale_value=scale_value,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            out=output,
        )
        return output


class VisionFlash2Attention(nn.Module):
    """Flash Attention 2 backend (flash_attn package, varlen interface)."""

    def __init__(self, **kwargs):
        if not _is_cuda:
            raise Exception("VisionFlashAttn2Attention is only available for cuda")
        if not _has_flash_attn_2:
            raise Exception("VisionFlashAttn2Attention requires flash_attn package (pip install flash-attn)")
        super().__init__()
        use_data_parallel = kwargs.get("use_data_parallel", False)
        self.tp_size = 1 if use_data_parallel else get_attention_tp_size()

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens: torch.Tensor | SingletonCache | None,
        bsz: int,
        seq_len: int,
        softmax_scale: float | None = None,
        **kwargs,
    ) -> torch.Tensor:
        cu_seqlens = resolve_seqlens(cu_seqlens, bsz, seq_len, device=q.device)
        cu_seqlens = cu_seqlens.to(dtype=torch.int32, device=q.device)
        seq_lens = cu_seqlens[1:] - cu_seqlens[:-1]
        max_seqlen = seq_lens.max().item()

        output = flash_attn_varlen_func_v2(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            softmax_scale=softmax_scale,
            causal=False,
        )
        return output


QKV_BACKEND_IMPL = {
    "triton_attn": VisionTritonAttention,
    "sdpa": VisionSdpaAttention,
    "fa2": VisionFlash2Attention,
    "fa3": VisionFlash3Attention,
    "fa4": VisionFlash4Attention,
    "flashinfer_cudnn": VisionFlashInferAttention,
    "ascend_attn": VisionAscendAttention,
}


class VisionAttention(nn.Module):
    r"""
        Multi-headed attention without any cache, mostly used for multimodal transformers.

    Supports multiple attention backends:
        - "triton_attn": Flash-attention style via custom Triton kernel (default on CUDA)
        - "sdpa": PyTorch scaled_dot_product_attention (universal fallback)
        - "fa3": Flash Attention 3 (CUDA Hopper)
        - "fa4": Flash Attention 4 (CUDA Blackwell)
        - "flashinfer_cudnn": FlashInfer cuDNN backend
        - "ascend_attn": Ascend NPU backend

    Backend selection priority:
        1. server_args.mm_attention_backend override (if set)
        2. qkv_backend constructor argument
        3. Platform auto-detect: CUDA SM90→fa3, SM100+→fa4, others→triton_attn;
           non-CUDA→sdpa

    Args:
        embed_dim: total embedding dimension
        num_heads: number of attention heads
        projection_size: output projection size (usually == embed_dim)
        use_qkv_parallel: if True, use QKVParallelLinear; else ColumnParallelLinear
        qkv_backend: explicit backend name, or None for auto-detection
        quant_config: optional quantization config
        dropout: attention dropout rate
        softmax_in_single_precision: use FP32 softmax
        softmax_scale: custom softmax scale (default: 1/sqrt(head_dim))
        flatten_batch: if True, all sequences are flattened into one batch dimension
        prefix: module name prefix for weight loading
        proj_bias: whether projection layer has bias
        qkv_bias: whether QKV projection has bias
        num_dummy_heads: extra heads for TP divisibility
        qk_normalization: apply RMSNorm to Q and K over full attn dim (InternVL-style)
        qk_normalization_by_head_size: apply RMSNorm per-head (GLM-OCR-style)
        layer_norm_eps: epsilon for RMSNorm
        customized_position_embedding_applier: custom RoPE applier
        use_data_parallel: if True, disable tensor parallelism for attention
        use_dp_attention_reduce: enable all-reduce in RowParallelLinear
        aux_stream: optional CUDA stream for parallelising QK-norm computation
        workspace_buffer: pre-allocated workspace for flashinfer_cudnn backend
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        projection_size: int,
        use_qkv_parallel: bool,
        qkv_backend: str | None = None,
        quant_config: QuantizationConfig | None = None,
        dropout: float = 0.0,
        softmax_in_single_precision: bool = False,
        softmax_scale: float | None = None,
        flatten_batch: bool = False,
        prefix: str = "",
        proj_bias: bool = True,
        num_dummy_heads: int = 0,
        qkv_bias: bool = True,
        qk_normalization: bool = False,
        qk_normalization_by_head_size: bool = False,
        layer_norm_eps: float = 1e-06,
        customized_position_embedding_applier: Callable[[torch.Tensor, torch.Tensor, Any, Any], tuple[torch.Tensor, torch.Tensor]] = None,
        use_data_parallel: bool = False,
        use_dp_attention_reduce: bool = False,
        aux_stream: torch.cuda.Stream | None = None,
        workspace_buffer: torch.Tensor | None = None,
        # Legacy compatibility
        use_context_forward: bool = True,
        use_full_precision_softmax: bool = False,
        **kwargs,
    ):
        super().__init__()
        if use_data_parallel:
            self.tp_size = 1
            self.tp_rank = 0
            self.tp_group = None
        else:
            self.tp_size = get_attention_tp_size()
            self.tp_rank = get_attention_tp_rank()
            self.tp_group = get_attention_tp_group()
        self.dropout = dropout
        self.head_size = embed_dim // num_heads
        self.hidden_size_per_attention_head = divide(projection_size, num_heads)
        self.num_attention_heads_per_partition = divide(num_dummy_heads + num_heads, self.tp_size)
        self.num_attention_kv_heads_per_partition = divide(num_dummy_heads + num_heads, self.tp_size)

        self.q_size = self.num_attention_heads_per_partition * self.head_size
        self.kv_size = self.num_attention_kv_heads_per_partition * self.head_size

        self.qk_normalization = qk_normalization
        self.qk_normalization_by_head_size = qk_normalization_by_head_size

        # Additional dummy heads are used to enable TP for common GPU counts.
        self.dummy_dim = (num_dummy_heads + num_heads) * self.head_size

        if self.qk_normalization:
            self.q_norm, self.k_norm = self._init_qk_norm(self.dummy_dim, layer_norm_eps)

        elif self.qk_normalization_by_head_size:
            self.q_norm, self.k_norm = self._init_qk_norm(self.head_size, layer_norm_eps)

        # Handle legacy softmax precision flags
        effective_softmax_fp32 = use_full_precision_softmax or softmax_in_single_precision

        # Select attention backend via a unified method
        _passed_backend = qkv_backend
        qkv_backend = self._determine_attention_backend(_passed_backend, use_context_forward)
        if global_server_args_dict.get("mm_attention_backend") is None and _passed_backend is None:
            print_info_once(f"Multimodal attention backend not set. Use {qkv_backend}.")
        print_info_once(f"Using {qkv_backend} as multimodal attention backend.")

        self.customized_position_embedding_applier = customized_position_embedding_applier
        self.softmax_scale = softmax_scale
        self.qkv_backend = QKV_BACKEND_IMPL[qkv_backend](
            head_dim=self.head_size,
            num_heads=self.num_attention_heads_per_partition,
            num_kv_heads=self.num_attention_kv_heads_per_partition,
            dropout=dropout,
            flatten_batch=flatten_batch,
            softmax_in_single_precision=effective_softmax_fp32,
            softmax_scale=softmax_scale,
            use_data_parallel=use_data_parallel,
            workspace_buffer=workspace_buffer,
        )

        self.use_qkv_parallel = use_qkv_parallel
        if use_qkv_parallel:
            self.qkv_proj = QKVParallelLinear(
                hidden_size=embed_dim,
                head_size=self.head_size,
                total_num_heads=num_dummy_heads + num_heads,
                total_num_kv_heads=num_dummy_heads + num_heads,
                bias=qkv_bias,
                quant_config=quant_config,
                outside_tp_group=self.tp_group,
                disable_parallel=use_data_parallel,
                prefix=add_prefix("qkv_proj", prefix),
            )
        else:
            self.qkv_proj = ColumnParallelLinear(
                input_size=embed_dim,
                output_size=3 * self.dummy_dim,
                bias=qkv_bias,
                quant_config=quant_config,
                outside_tp_group=self.tp_group,
                disable_parallel=use_data_parallel,
                prefix=add_prefix("qkv_proj", prefix),
            )
        self.proj = RowParallelLinear(
            input_size=self.dummy_dim,
            output_size=embed_dim,
            bias=proj_bias,
            quant_config=quant_config,
            outside_tp_group=self.tp_group,
            disable_parallel=use_data_parallel,
            prefix=add_prefix("proj", prefix),
            # use_dp_attention_reduce=use_dp_attention_reduce,
        )

        self.workspace_buffer = workspace_buffer
        self.aux_stream = aux_stream
        self.ln_events = [torch.cuda.Event(), torch.cuda.Event()] if aux_stream else []

    def _init_qk_norm(self, norm_dim: int, eps: float):
        """Initialize Q and K normalization layers."""
        q_norm = RMSNorm(norm_dim, eps=eps)
        k_norm = RMSNorm(norm_dim, eps=eps)
        return q_norm, k_norm

    def _determine_attention_backend(self, passed_backend: str | None, use_context_forward: bool = True) -> str:
        """Decide the multimodal attention backend string.

        Priority: server args override > constructor arg > legacy flag > platform default.

        Platform defaults:
        - CUDA (Hopper SM90): "fa3"
        - CUDA (Blackwell SM100+): "fa4"
        - CUDA (other): "triton_attn"
        - Ascend NPU: "ascend_attn"
        - Non-CUDA: "sdpa"
        """
        override_backend = global_server_args_dict.get("mm_attention_backend")
        if override_backend is not None:
            return override_backend
        if passed_backend is not None:
            return passed_backend

        # Legacy flag: use_context_forward=False means user wants sdpa
        if not use_context_forward:
            return "sdpa"

        if _is_cuda:
            try:
                major, minor = get_device_capability()
                if major == 9 and _has_flash_attn:
                    backend = "fa3"
                elif major >= 10 and _has_flash_attn:
                    backend = "fa4"
                else:
                    backend = "triton_attn"
            except Exception:
                backend = "triton_attn"
        elif _is_npu:
            backend = "ascend_attn"
        else:
            backend = "sdpa"

        # Blackwell guard: fa3 is not supported on Blackwell (SM100+)
        if backend == "fa3" and _is_blackwell_supported():
            raise ValueError("The 'fa3' backend is not supported on Blackwell GPUs")

        return backend

    @staticmethod
    def _select_backend(qkv_backend: str | None, use_context_forward: bool = True) -> str:
        """Backward-compatible static method for backend selection.

        Kept for unit-test and external code compatibility.
        New code should use _determine_attention_backend() instead.

        Priority:
        1. server_args.mm_attention_backend (if set)
        2. Explicit qkv_backend argument
        3. Legacy use_context_forward flag
        4. Auto-detect: CUDA → triton_attn, non-CUDA → sdpa
        """
        # Check global server args override first
        override = global_server_args_dict.get("mm_attention_backend")
        if override is not None:
            return override

        if qkv_backend is not None:
            return qkv_backend

        # Legacy flag support
        if not use_context_forward:
            return "sdpa"

        # Auto-detect: triton on CUDA (using torch.cuda.is_available for testability),
        # sdpa elsewhere
        with suppress(Exception):
            import torch as _torch

            if _torch.cuda.is_available():
                return "triton_attn"

        return "sdpa"

    def _apply_qk_norm_head_size(self, q: torch.Tensor, k: torch.Tensor):
        """Apply qk norm per-head (GLM-OCR ViT attn style)."""
        q_by_head = q.reshape(-1, self.head_size)
        q_by_head = self.q_norm(q_by_head)
        k_by_head = k.reshape(-1, self.head_size)
        k_by_head = self.k_norm(k_by_head)
        q = q_by_head.view(q.shape)
        k = k_by_head.view(k.shape)
        return q, k

    def _apply_qk_norm(self, q: torch.Tensor, k: torch.Tensor):
        """Apply qk norm for InternVL ViT attn (full attention-dim norm)."""

        def q_l2norm():
            q_ = q.flatten(1, 2)
            if self.tp_size > 1:
                q_ = self.tp_group.all_gather(q_.contiguous())
                # q_ = tensor_model_parallel_all_gather(q_.contiguous())
            q_ = self.q_norm(q_)
            if self.tp_size > 1:
                splitter = partial(split_tensor_along_last_dim, num_partitions=self.tp_size)
                q_ = splitter(q_)[self.tp_rank]
            q_ = q_.unflatten(-1, (-1, self.head_size))
            return q_

        def k_l2norm():
            k_ = k.flatten(1, 2)
            if self.tp_size > 1:
                k_ = self.tp_group.all_gather(k_.contiguous())
                # k_ = tensor_model_parallel_all_gather(k_.contiguous())
            k_ = self.k_norm(k_)
            if self.tp_size > 1:
                splitter = partial(split_tensor_along_last_dim, num_partitions=self.tp_size)
                k_ = splitter(k_)[self.tp_rank]
            k_ = k_.unflatten(-1, (-1, self.head_size))
            return k_

        with _with_multi_stream(True):
            q, k = _maybe_execute_in_parallel(
                q_l2norm,
                k_l2norm,
                self.ln_events,
                self.aux_stream,
            )
        return q, k

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        rotary_pos_emb: torch.Tensor | None = None,
        rotary_pos_emb_cos: torch.Tensor | None = None,
        rotary_pos_emb_sin: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        r"""
        Args:
            x: [b, s, embed_dim] or [total_tokens, embed_dim] (if flattened)
            cu_seqlens: cumulative sequence lengths [b+1] for varlen attention
            position_embeddings: (cos, sin) tuple for standard RoPE
            rotary_pos_emb: rotary position embeddings (QwenVL-style freqs tensor)
            rotary_pos_emb_cos: explicit cos tensor for RoPE
            rotary_pos_emb_sin: explicit sin tensor for RoPE
            attention_mask: explicit attention mask
        Returns:
            [b, s, embed_dim] attention output
        """
        if x.dim() == 2:
            x = x.unsqueeze(0)
        assert x.dim() == 3, x.shape
        x_shape = x.shape
        bsz, s, _ = x_shape
        head = self.num_attention_heads_per_partition
        kv_head = self.num_attention_kv_heads_per_partition

        attn_output_ws = kwargs.get("output_ws")
        max_seqlen = kwargs.get("max_seqlen")
        sequence_lengths = kwargs.get("sequence_lengths")

        if self.use_qkv_parallel:
            # [b, s, embed_dim] --> [b, s, embed_dim]
            qkv, _ = self.qkv_proj(x)
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

            # [b, s, embed_dim] --> [b * s, head, head_size]
            q = q.reshape(bsz * s, head, -1).contiguous()
            k = k.reshape(bsz * s, kv_head, -1).contiguous()
            v = v.reshape(bsz * s, kv_head, -1).contiguous()
            if self.qk_normalization_by_head_size:
                q, k = self._apply_qk_norm_head_size(q, k)
        else:
            # [b, s, embed_dim] --> [s, b, embed_dim]
            x = rearrange(x, "b s ... -> s b ...")
            # [s, b, embed_dim] --> [s, b, head * 3 * head_size]
            qkv, _ = self.qkv_proj(x)

            # [s, b, head, head_dim_sum]
            new_x_shape = qkv.size()[:-1] + (
                head,
                self.q_size + 2 * self.kv_size,
            )
            qkv = qkv.view(*new_x_shape)

            # [s, b, head, 3 * head_size] --> 3 [s, b, head, head_size]
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

            # [s, b, head, head_size] --> [b, s, head, head_size]
            q, k, v = [rearrange(x, "s b ... -> b s ...").contiguous() for x in (q, k, v)]

            if self.qk_normalization_by_head_size:
                q, k = self._apply_qk_norm_head_size(q, k)

        cos = None
        sin = None

        if position_embeddings is not None:
            if self.customized_position_embedding_applier is not None:
                q, k = self.customized_position_embedding_applier(q, k, position_embeddings, x_shape)
            else:
                cos, sin = position_embeddings
        elif rotary_pos_emb_cos is not None and rotary_pos_emb_sin is not None:
            cos = rotary_pos_emb_cos
            sin = rotary_pos_emb_sin
        elif rotary_pos_emb is not None:
            # QwenVL-style: rotary_pos_emb is a freqs tensor
            # Apply using the vision-specific applier (handles float cast internally)
            q = _apply_rotary_pos_emb_vision(q, rotary_pos_emb)
            k = _apply_rotary_pos_emb_vision(k, rotary_pos_emb)

        if cos is not None and sin is not None:
            original_shape = q.shape

            # [total_tokens, head, head_size]
            q = q.view(-1, head, self.head_size)
            k = k.view(-1, head, self.head_size)

            if can_use_triton_rope(q, cos, self.head_size):
                if not k.is_contiguous():
                    k = k.contiguous()
                q, k = apply_rope_triton(q, k, cos, sin, self.head_size)
            else:
                if cos.size(-1) * 2 == self.head_size:
                    cos = torch.cat([cos, cos], dim=-1)
                    sin = torch.cat([sin, sin], dim=-1)

                q, k = apply_rotary_pos_emb(q, k, cos, sin)

            q = q.view(original_shape)
            k = k.view(original_shape)

        if q.dim() == 4:
            # [b, s, head, head_size] --> [b * s, head, head_size]
            q = rearrange(q, "b s ... -> (b s) ...")
        if k.dim() == 4:
            k = rearrange(k, "b s ... -> (b s) ...")
        if v.dim() == 4:
            v = rearrange(v, "b s ... -> (b s) ...")

        assert q.dim() == 3, q.dim()
        assert k.dim() == 3, k.dim()
        assert v.dim() == 3, v.dim()

        # InternVL-style full-dim QK norm
        if self.qk_normalization and not self.qk_normalization_by_head_size:
            if _can_use_jit_qk_norm(self.head_size, q.dtype):
                # JIT kernel path (not available in fluentllm, but kept for future)
                head_dim_for_norm = head * self.head_size
                q, k = _apply_qk_norm_jit(
                    q=q,
                    k=k,
                    q_norm=self.q_norm,
                    k_norm=self.k_norm,
                    head_dim=head_dim_for_norm,
                    alt_stream=self.aux_stream,
                )
            else:
                q, k = self._apply_qk_norm(q, k)

        output = self.qkv_backend.forward(
            q=q,
            k=k,
            v=v,
            bsz=bsz,
            seq_len=s,
            cu_seqlens=cu_seqlens,
            attention_mask=attention_mask,
            sequence_lengths=sequence_lengths,
            max_seqlen=max_seqlen,
            output_ws=attn_output_ws,
            softmax_scale=self.softmax_scale,
        )

        assert output.dim() == 3, output.shape

        if self.use_qkv_parallel:
            # [b * s, h, head_size] --> [b, s, h * head_size]
            output = rearrange(output, "(b s) ... h d -> b s ... (h d)", b=bsz)

            # [b, s, h * head_size] --> [b, s, h * head_size]
            output, _ = self.proj(output)
        else:
            # [b * s, h, head_size] --> [s, b, h * head_size]
            context_layer = rearrange(output, "(b s) h d -> s b (h d)", b=bsz, s=s).contiguous()

            # [s, b, h * head_size] --> [s, b, h * head_size]
            output, _ = self.proj(context_layer)

            # [s, b, h * head_size] --> [b, s, h * head_size]
            output = output.view(bsz, s, -1)

        return output
