from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Union

import torch
import triton
import triton.language as tl

from sglang.srt.distributed import (
    GroupCoordinator,
    get_tensor_model_parallel_world_size,
    get_tp_group,
    get_mlp_tp_group,
    init_model_parallel_group,
    tensor_model_parallel_all_reduce,
    get_world_group
)
from sglang.srt.distributed.device_communicators.custom_all_reduce import _can_p2p
from sglang.srt.utils import get_colorful_logger, is_npu, is_sm90_supported
if TYPE_CHECKING:
    from sglang.srt.layers.logits_processor import LogitsMetadata
logger = get_colorful_logger(__name__)

_is_npu__ = is_npu()


if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

_ATTN_TP_GROUP = None
_ATTN_TP_RANK = None
_ATTN_TP_SIZE = None
_DP_RANK = None
_DP_SIZE = None

_ATTN_TP_DP_CONVERTOR = None
_DENSE_TP_DP_CONVERTOR = None

"""
============================================= initialize dp tp convertor =============================================

|============|================|==============================|======================|
| ATTN DP TP |   POST ATTN    |          PRE DENSE           |     DENSE DP TP      |
|============|================|==============================|======================|
|            |                | all gather in dense dp group |       DP1 TP8        |
|            |                |==============================|======================|
|  DP8 TP1   |   No comm      | all gather in dense dp group |   DP2 TP4 | DP4 TP2  |
|            |                |==============================|======================|
|            |                |          No comm              |       DP8 TP1        |
|============|================|==============================|======================|
|  DP2 TP4   | reduce_scatter | all gather in dense dp group |       DP2 TP4        |
|============|================|==============================|======================|
|  DP4 TP2   | reduce_scatter | all gather in dense dp group |       DP4 TP2        |
|============|================|==============================|======================|
|  DP1 TP8   | reduce_scatter |          No comm              |       DP8 TP1        |
|============|================|==============================|======================|

"""
"""
    Communication module in eps, used for uneven reduce_scatter and all_gather within attn tp group
"""
def init_attn_tp_dp_convertor_v1(global_rank, max_num_tokens, attn_tp_size, hidden_size):
    global _ATTN_TP_DP_CONVERTOR
    from sglang.srt.distributed.parallel_state import _EPS_COMMUNICATOR
    from eps.communication import TPDPConvertor
    p = TPDPConvertor.Params(
        global_rank,
        max_num_tokens,
        attn_tp_size,
        hidden_size,
        _EPS_COMMUNICATOR,
    )
    convertor = TPDPConvertor(p)
    _ATTN_TP_DP_CONVERTOR = convertor

"""
    Custom triton multimem communication module, used for uneven reduce_scatter and all_gather within tp group
    eps establishes global communication groups, but for completing reduce_scatter and all_gather within tp group, here we only establish in-group communication connections
"""
def init_attn_tp_dp_convertor_v2(max_num_tokens, hidden_size):

    global _ATTN_TP_DP_CONVERTOR
    from sglang.srt.distributed.device_communicators.custom_triton_rsag.triton_rsag import TritonRSAG
    convertor = TritonRSAG(get_attention_tp_group(), get_attention_tp_rank(), max_num_tokens, hidden_size)
    _ATTN_TP_DP_CONVERTOR = convertor


def init_dense_tp_dp_convertor(max_num_tokens, hidden_size):
    global _DENSE_TP_DP_CONVERTOR
    from sglang.srt.distributed.device_communicators.custom_triton_rsag.triton_rsag import TritonRSAG
    convertor = TritonRSAG(get_dense_tp_group(), get_dense_tp_rank(), max_num_tokens, hidden_size)
    _DENSE_TP_DP_CONVERTOR = convertor

"""
============================================= initialize dp tp group =============================================
"""
def initialize_dp_attention(attn_tp_rank, attn_tp_size, dp_size, dp_rank, global_rank, local_rank, hidden_size, max_num_tokens, force_deterministic_rsag):
    global _ATTN_TP_GROUP, _ATTN_TP_RANK, _ATTN_TP_SIZE, _DP_RANK, _DP_SIZE

    from sglang.srt.layers.sampler import SYNC_TOKEN_IDS_ACROSS_TP

    _ATTN_TP_RANK, _ATTN_TP_SIZE, _DP_RANK = attn_tp_rank, attn_tp_size, dp_rank
    _DP_SIZE = dp_size

    world_group = get_world_group()
    world_size = world_group.world_size

    _ATTN_TP_GROUP = GroupCoordinator(
        [
            list(range(head, head + _ATTN_TP_SIZE))
            for head in range(0, world_size, _ATTN_TP_SIZE)
        ],
        local_rank,
        torch.distributed.get_backend(world_group.device_group),
        SYNC_TOKEN_IDS_ACROSS_TP,
        False,
        False,
        False,
        False,
        group_name="attention_tp",
    )

    assert max_num_tokens is not None

    if not _is_npu__ and _can_p2p(local_rank, attn_tp_size):
        if attn_tp_size > 1 and is_sm90_supported() and not force_deterministic_rsag:
            init_attn_tp_dp_convertor_v2(max_num_tokens=max_num_tokens, hidden_size=hidden_size)
        else:
            init_attn_tp_dp_convertor_v1(global_rank=global_rank, max_num_tokens=max_num_tokens, attn_tp_size=attn_tp_size, hidden_size=hidden_size)


def initialize_dp_dense(dense_tp_rank, dense_tp_size, dense_dp_size, dense_dp_rank, max_num_tokens, hidden_size, local_rank):
    global _ATTN_TP_DP_CONVERTOR, _DENSE_TP_DP_CONVERTOR, _ATTN_TP_GROUP, _DP_SIZE, _ATTN_TP_SIZE, _DENSE_TP_GROUP, _DENSE_TP_RANK, _DENSE_TP_SIZE, _DENSE_DP_RANK, _DENSE_DP_SIZE

    _DENSE_TP_RANK, _DENSE_TP_SIZE, _DENSE_DP_RANK, _DENSE_DP_SIZE = dense_tp_rank, dense_tp_size, dense_dp_rank, dense_dp_size

    if _DENSE_TP_SIZE == _ATTN_TP_SIZE:
        logger.info(f"dense_dp_tp_group reuse attn_dp_tp_group")
        _DENSE_TP_GROUP = _ATTN_TP_GROUP
        _DENSE_TP_DP_CONVERTOR = _ATTN_TP_DP_CONVERTOR
    else:
        tp_group = get_tp_group()
        dense_tp_group_ranks = [
            list(range(i * _DENSE_TP_SIZE, (i + 1) * _DENSE_TP_SIZE))
            for i in range(_DENSE_DP_SIZE)
        ]
        for ranks in dense_tp_group_ranks:
            logger.info(f"dense_tp_group_ranks: {ranks}")
        _DENSE_TP_GROUP = init_model_parallel_group(
            dense_tp_group_ranks,
            local_rank,
            torch.distributed.get_backend(tp_group.device_group),
            group_name="dense_tp",
        )

        if _is_npu__:
            _DENSE_TP_GROUP = get_mlp_tp_group()

        assert max_num_tokens is not None

        if not _is_npu__ and _can_p2p(local_rank, dense_tp_size):
            init_dense_tp_dp_convertor(
                max_num_tokens=max_num_tokens * max(int(_DP_SIZE / _DENSE_DP_SIZE), 1),
                hidden_size=hidden_size
            )

"""
============================================= get dp tp convertor && info =============================================
"""
def get_dense_tp_dp_convertor():
    return _DENSE_TP_DP_CONVERTOR


def get_attn_tp_dp_convertor():
    assert _ATTN_TP_DP_CONVERTOR is not None, "attn_tp_dp_convertor is not initialized!"
    return _ATTN_TP_DP_CONVERTOR


def get_attention_tp_group():
    assert _ATTN_TP_GROUP is not None, "dp attention not initialized!"
    return _ATTN_TP_GROUP


def get_attention_tp_rank():
    assert _ATTN_TP_RANK is not None, "dp attention not initialized!"
    return _ATTN_TP_RANK


def get_attention_tp_size():
    assert _ATTN_TP_SIZE is not None, "dp attention not initialized!"
    return _ATTN_TP_SIZE


def get_attention_dp_rank():
    assert _DP_RANK is not None, "dp attention not initialized!"
    return _DP_RANK


def get_attention_dp_size():
    assert _DP_SIZE is not None, "dp attention not initialized!"
    return _DP_SIZE


def get_dense_tp_group():
    assert _DENSE_TP_GROUP is not None, "dp dense not initialized!"
    if isinstance(_DENSE_TP_GROUP, list):
        return _DENSE_TP_GROUP[_DENSE_DP_RANK]
    else:
        return _DENSE_TP_GROUP


def get_dense_tp_rank():
    assert _DENSE_TP_RANK is not None, "dp dense not initialized!"
    return _DENSE_TP_RANK


def get_dense_tp_size():
    assert _DENSE_TP_SIZE is not None, "dp dense not initialized!"
    return _DENSE_TP_SIZE


def get_dense_dp_rank():
    assert _DENSE_DP_RANK is not None, "dp dense not initialized!"
    return _DENSE_DP_RANK


def get_dense_dp_size():
    assert _DENSE_DP_SIZE is not None, "dp dense not initialized!"
    return _DENSE_DP_SIZE


_DRAFT_ATTN_TP_GROUP = None
_DRAFT_ATTN_TP_RANK = None
_DRAFT_ATTN_TP_SIZE = None

def get_draft_attention_tp_group():
    assert _DRAFT_ATTN_TP_GROUP is not None, "draft dp attention not initialized!"
    return _DRAFT_ATTN_TP_GROUP


def get_draft_attention_tp_rank():
    assert _DRAFT_ATTN_TP_RANK is not None, "draft dp attention not initialized!"
    return _DRAFT_ATTN_TP_RANK


def get_draft_attention_tp_size():
    assert _DRAFT_ATTN_TP_SIZE is not None, "draft dp attention not initialized!"
    return _DRAFT_ATTN_TP_SIZE

def initialize_dp_draft_model():
    global _DRAFT_ATTN_TP_GROUP, _DRAFT_ATTN_TP_RANK, _DRAFT_ATTN_TP_SIZE

    from sglang.srt.layers.sampler import SYNC_TOKEN_IDS_ACROSS_TP

    # set to TP1
    _DRAFT_ATTN_TP_RANK, _DRAFT_ATTN_TP_SIZE = 0, 1

    tp_group = get_tp_group()

    world_size = torch.distributed.get_world_size()

    _DRAFT_ATTN_TP_GROUP = GroupCoordinator(
        [
            list(range(head, head + _DRAFT_ATTN_TP_SIZE))
            for head in range(0, world_size, _DRAFT_ATTN_TP_SIZE)
        ],
        0,
        torch.distributed.get_backend(tp_group.device_group),
        SYNC_TOKEN_IDS_ACROSS_TP,
        False,
        False,
        False,
        False,
        group_name="draft_attention_tp",
    )


def compute_dp_attention_world_info(enable_dp_attention, tp_rank, tp_size, dp_size):
    if not enable_dp_attention:
        return tp_rank, tp_size, 0

    attn_tp_size = tp_size // dp_size
    dp_rank = tp_rank // attn_tp_size
    attn_tp_rank = tp_rank % attn_tp_size
    return attn_tp_rank, attn_tp_size, dp_rank

def get_dp_local_info(forward_batch: LogitsMetadata):
    dp_rank = get_attention_dp_rank()

    if forward_batch.dp_local_start_pos is None:
        cumtokens = torch.cumsum(forward_batch.global_num_tokens_gpu, dim=0)
        if dp_rank == 0:
            local_start_pos = torch.zeros_like(cumtokens[0])
        else:
            local_start_pos = cumtokens[dp_rank - 1]
        local_num_tokens = forward_batch.global_num_tokens_gpu[dp_rank]

        forward_batch.dp_local_start_pos = local_start_pos
        forward_batch.dp_local_num_tokens = local_num_tokens

    return forward_batch.dp_local_start_pos, forward_batch.dp_local_num_tokens


@triton.jit
def memcpy_triton_kernel(
    dst_ptr,
    src_ptr,
    offset_ptr,
    sz_ptr,
    offset_src,
    chunk_size,  # multiplied for offset and sz
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0).to(tl.int64)
    offset = tl.load(offset_ptr).to(tl.int64) * chunk_size
    sz = tl.load(sz_ptr).to(tl.int64) * chunk_size

    start_index = pid * BLOCK_SIZE
    offs = tl.arange(0, BLOCK_SIZE)
    mask = start_index + offs < sz

    if offset_src:
        data = tl.load(src_ptr + offset + start_index + offs, mask=mask)
        tl.store(dst_ptr + start_index + offs, data, mask=mask)
    else:
        data = tl.load(src_ptr + start_index + offs, mask=mask)
        tl.store(dst_ptr + offset + start_index + offs, data, mask=mask)


def prod(x):
    return functools.reduce(lambda a, b: a * b, x, 1)


def memcpy_triton(dst, src, dim, offset, sz, offset_src):
    max_size = min(src.numel(), dst.numel())
    assert dim == 0, "dim != 0 unsupported"
    assert src.shape[1:] == dst.shape[1:], "src and dst must have same shape"
    chunk_size = prod(src.shape[1:])
    BLOCK_SIZE = 8192
    grid = (triton.cdiv(max_size, BLOCK_SIZE),)

    memcpy_triton_kernel[grid](dst, src, offset, sz, offset_src, chunk_size, BLOCK_SIZE)


def memcpy_npu(dst, src, dim, offset, sz, offset_src):
    """
    NPU-compatible memcpy_triton alternative implementation.
    dst: Destination tensor
    src: Source tensor
    dim: Currently only supports 0
    offset: Starting position for writing to destination tensor
    sz: Number of elements to copy (main dimension)
    offset_src: Whether to offset src (True means src[offset:offset+sz], False means src[:sz])
    """
    assert dim == 0, "Only supports dimension 0"
    assert src.shape[1:] == dst.shape[1:], "src and dst must have the same shape except for the main dimension"
    if offset_src:
        # src offset
        dst[:sz].copy_(src[offset:offset+sz])
    else:
        # dst offset
        dst[offset:offset+sz].copy_(src[:sz])

def memcpy_generic(dst, src, dim, offset, sz, offset_src):
    if _is_npu:
        memcpy_npu(dst, src, dim, offset, sz, offset_src)
    else:
        memcpy_triton(dst, src, dim, offset, sz, offset_src)

def dp_gather(
    global_tokens: torch.Tensor,
    local_tokens: torch.Tensor,
    forward_batch: ForwardBatch,
    layer_id: Union[str, int],
):
    local_start_pos, local_num_tokens = get_dp_local_info(forward_batch)

    global_tokens.fill_(0)
    assert local_tokens.is_contiguous()
    assert global_tokens.is_contiguous()
    if local_tokens.shape[0] > 0 and (
        layer_id != "embedding" or get_attention_tp_rank() == 0
    ):
        assert (
            global_tokens.storage().data_ptr() != local_tokens.storage().data_ptr()
        ), "aliasing between global_tokens and local_tokens not allowed"
        memcpy_generic(
            global_tokens, local_tokens, 0, local_start_pos, local_num_tokens, False
        )

    # Input IDs are in int 32. We should use inplace_all_reduce for local case becaues of custom all reduce.
    NUM_GPUS_PER_NODE = 8
    if (
        not local_tokens.dtype.is_floating_point
        and get_tensor_model_parallel_world_size() <= NUM_GPUS_PER_NODE
    ):
        torch.ops.sglang.inplace_all_reduce(
            global_tokens, group_name=get_tp_group().unique_name
        )
    else:
        global_tokens = tensor_model_parallel_all_reduce(global_tokens)


def dp_scatter(
    local_tokens: torch.Tensor,  # output
    global_tokens: torch.Tensor,  # input
    forward_batch: ForwardBatch,
):
    # local_num_tokens is not necessarily the same as local_tokens.shape[0],
    # since local_tokens may be padded for cuda graph
    local_start_pos, local_num_tokens = get_dp_local_info(forward_batch)
    local_tokens.fill_(0)
    assert local_tokens.is_contiguous()
    assert global_tokens.is_contiguous()
    if local_tokens.shape[0] > 0:
        assert (
            local_tokens.untyped_storage().data_ptr()
            != global_tokens.untyped_storage().data_ptr()
        ), "aliasing between local_tokens and global_tokens not allowed"
        memcpy_generic(
            local_tokens, global_tokens, 0, local_start_pos, local_num_tokens, True
        )


def get_do_logits_dp_scatter(forward_batch: ForwardBatch):
    def do_logits_dp_scatter(logits: torch.Tensor):
        local_logits = torch.empty(
            (forward_batch.input_ids.shape[0], *logits.shape[1:]),
            dtype=logits.dtype,
            device=logits.device,
        )
        dp_scatter(local_logits, logits, forward_batch)
        return local_logits

    return do_logits_dp_scatter
