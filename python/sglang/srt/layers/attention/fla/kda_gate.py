# Adapted from https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/kda/gate.py
# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# This file is modified and supported by the Moonshot AI Team

import torch
import triton
import triton.language as tl

from sglang.srt.layers.attention.fla.index import prepare_chunk_indices
from sglang.srt.layers.attention.fla.op import exp
from sglang.srt.layers.attention.fla.utils import check_shared_mem, input_guard, is_amd

BS_LIST = [32, 64] if check_shared_mem() else [16, 32]
BT_LIST_AUTOTUNE = [32, 64, 128]
NUM_WARPS_AUTOTUNE = [2, 4, 8, 16] if is_amd else [4, 8, 16, 32]


@triton.jit
def softplus(x):
    return tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(x)))


@triton.heuristics({
    "HAS_BIAS": lambda args: args["dt_bias"] is not None,
    "HAS_BETA": lambda args: args["beta"] is not None,
    'USE_LOWER_BOUND': lambda args: args['lower_bound'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({"BT": BT}, num_warps=num_warps, num_stages=num_stages)
        for BT in BT_LIST_AUTOTUNE
        for num_warps in NUM_WARPS_AUTOTUNE
        for num_stages in [2, 3]
    ],
    key=["H", "D"],
)
@triton.jit(do_not_specialize=['T'])
def kda_gate_fwd_kernel(
    g,
    A_log,
    dt_bias,
    beta,
    yg,
    yb,
    lower_bound,
    T,
    H: tl.constexpr,
    D: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_BETA: tl.constexpr,
    USE_LOWER_BOUND: tl.constexpr,
):
    i_t, i_h = tl.program_id(0), tl.program_id(1)

    b_A = tl.load(A_log + i_h).to(tl.float32)

    p_g = tl.make_block_ptr(g + i_h * D, (T, D), (H * D, 1), (i_t * BT, 0), (BT, BD), (1, 0))
    p_yg = tl.make_block_ptr(yg + i_h * D, (T, D), (H * D, 1), (i_t * BT, 0), (BT, BD), (1, 0))
    # [BT, BD]
    b_g = tl.load(p_g, boundary_check=(0, 1)).to(tl.float32)
    if HAS_BIAS:
        p_b = tl.make_block_ptr(dt_bias, (H * D,), (1,), (i_h * D,), (BD,), (0,))
        b_g = b_g + tl.load(p_b, boundary_check=(0,)).to(tl.float32)
    if not USE_LOWER_BOUND:
        b_yg = -exp(b_A) * softplus(b_g)
    else:
        b_yg = lower_bound * tl.sigmoid(exp(b_A) * b_g)
    tl.store(p_yg, b_yg.to(p_yg.dtype.element_ty), boundary_check=(0, 1))

    if HAS_BETA:
        p_b = tl.make_block_ptr(beta + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
        p_yb = tl.make_block_ptr(yb + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
        b_yb = tl.sigmoid(tl.load(p_b, boundary_check=(0,)).to(tl.float32))
        tl.store(p_yb, b_yb.to(p_yb.dtype.element_ty), boundary_check=(0,))


def kda_gate_fwd(
    g: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor | None = None,
    lower_bound: float | None = None,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    H, K = g.shape[-2:]
    T = g.numel() // (H * K)

    yg = torch.empty_like(g, dtype=output_dtype)

    def grid(meta):
        return (triton.cdiv(T, meta["BT"]), H)

    kda_gate_fwd_kernel[grid](
        g=g,
        A_log=A_log,
        dt_bias=dt_bias,
        beta=None,
        yg=yg,
        yb=None,
        T=T,
        H=H,
        D=K,
        BD=triton.next_power_of_2(K),
        lower_bound=lower_bound,
    )
    return yg


def fused_kda_gate(
    g: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor | None = None,
    lower_bound: float | None = None,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Fused KDA gate computation (forward-only, no autograd).

    Computes: g = -A_log.exp().unsqueeze(-1) * softplus(g + dt_bias.view(g.shape[-2:]))
    Or with lower_bound: g = lower_bound * sigmoid(A_log.exp() * (g + dt_bias))

    Args:
        g (torch.Tensor):
            Input tensor of shape `[..., H, K]`.
        A_log (torch.Tensor):
            Parameter tensor with `H` elements.
        dt_bias (torch.Tensor | None):
            Optional bias tensor added to `g` before activation, shape `[H * K]`.
        lower_bound (float | None):
            If not None, use sigmoid mode with this lower bound.
        output_dtype (torch.dtype):
            Output dtype.

    Returns:
        Output tensor of shape `[..., H, K]`.
    """
    return kda_gate_fwd(g, A_log, dt_bias, lower_bound, output_dtype)


@triton.jit
def beta_sigmoid_fwd_kernel(
    x,
    y,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE).to(tl.int64)
    mask = offs < n_elements
    b_x = tl.load(x + offs, mask=mask, other=0).to(tl.float32)
    b_y = tl.sigmoid(b_x)
    tl.store(y + offs, b_y.to(y.dtype.element_ty), mask=mask)


_BETA_SIGMOID_BLOCK_SIZE = 2048
_BETA_SIGMOID_NUM_WARPS = 8


def beta_sigmoid_fwd(x: torch.Tensor) -> torch.Tensor:
    y = torch.empty_like(x, dtype=torch.float32)
    n_elements = x.numel()
    grid = (triton.cdiv(n_elements, _BETA_SIGMOID_BLOCK_SIZE),)
    beta_sigmoid_fwd_kernel[grid](
        x,
        y,
        n_elements,
        BLOCK_SIZE=_BETA_SIGMOID_BLOCK_SIZE,
        num_warps=_BETA_SIGMOID_NUM_WARPS,
    )
    return y


@triton.heuristics({
    "HAS_BIAS": lambda args: args["dt_bias"] is not None,
    'HAS_SCALE': lambda args: args['scale'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
    'USE_LOWER_BOUND': lambda args: args['lower_bound'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'BS': BS}, num_warps=num_warps)
        for BS in BS_LIST
        for num_warps in [2, 4, 8]
    ],
    key=['H', 'S', 'BT', 'IS_VARLEN', 'REVERSE'],
)
@triton.jit(do_not_specialize=['T'])
def kda_gate_chunk_cumsum_vector_kernel(
    s,
    A_log,
    dt_bias,
    o,
    scale,
    cu_seqlens,
    chunk_indices,
    lower_bound,
    T,
    H: tl.constexpr,
    S: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    REVERSE: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_LOWER_BOUND: tl.constexpr,
):
    i_s, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    p_s = tl.make_block_ptr(s + (bos * H + i_h) * S, (T, S), (H*S, 1), (i_t * BT, i_s * BS), (BT, BS), (1, 0))
    p_o = tl.make_block_ptr(o + (bos * H + i_h) * S, (T, S), (H*S, 1), (i_t * BT, i_s * BS), (BT, BS), (1, 0))
    # [BT, BS]
    b_s = tl.load(p_s, boundary_check=(0, 1)).to(tl.float32)

    # Apply dt_bias if exists
    if HAS_BIAS:
        p_b = tl.make_block_ptr(dt_bias + i_h * S, (S,), (1,), (i_s * BS,), (BS,), (0,))
        b_bias = tl.load(p_b, boundary_check=(0,)).to(tl.float32)
        b_s = b_s + b_bias[None, :]

    b_A = tl.load(A_log + i_h).to(tl.float32)
    if not USE_LOWER_BOUND:
        # Apply gate: -exp(A_log) * softplus(g + bias)
        b_gate = -exp(b_A) * softplus(b_s)
    else:
        b_gate = lower_bound * tl.sigmoid(exp(b_A) * b_s)

    # Apply chunk local cumsum
    if REVERSE:
        b_o = tl.cumsum(b_gate, axis=0, reverse=True)
    else:
        b_o = tl.cumsum(b_gate, axis=0)

    if HAS_SCALE:
        b_o *= scale
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))


@input_guard
def kda_gate_chunk_cumsum(
    g: torch.Tensor,
    A_log: torch.Tensor,
    chunk_size: int,
    scale: float = None,
    dt_bias: torch.Tensor | None = None,
    cu_seqlens: torch.Tensor | None = None,
    output_dtype: torch.dtype | None = torch.float,
    chunk_indices: torch.LongTensor | None = None,
    lower_bound: float | None = None,
    **kwargs,
) -> torch.Tensor:
    if cu_seqlens is not None:
        assert g.shape[0] == 1, "Only batch size 1 is supported when cu_seqlens are provided"
    assert len(g.shape) == 4
    B, T, H, S = g.shape
    BT = chunk_size
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    assert chunk_size == 2**(chunk_size.bit_length()-1), "chunk_size must be a power of 2"

    g_org, g = g, torch.empty_like(g, dtype=output_dtype or g.dtype)
    def grid(meta): return (triton.cdiv(meta['S'], meta['BS']), NT, B * H)
    kda_gate_chunk_cumsum_vector_kernel[grid](
        s=g_org,
        A_log=A_log,
        dt_bias=dt_bias,
        o=g,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        lower_bound=lower_bound,
        T=T,
        H=H,
        S=S,
        BT=BT,
        REVERSE=False,
    )
    return g
