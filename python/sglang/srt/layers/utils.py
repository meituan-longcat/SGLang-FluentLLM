from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from sglang.srt.distributed import GroupCoordinator

if TYPE_CHECKING:
    from sglang.srt.distributed.device_communicators.custom_triton_rsag.triton_rsag import TritonRSAG

import torch

logger = logging.getLogger(__name__)


def get_layer_id(weight_name):
    # example weight name: model.layers.10.self_attn.qkv_proj.weight
    match = re.search(r"layers\.(\d+)\.", weight_name)
    if match:
        return int(match.group(1))
    return None


def maybe_get_longcat_attn_layer_id(weight_name):
    # example: model.layers.4.self_attn.1.fused_qkv_a_proj_with_mqa.weight
    match = re.search(r"layers\.(\d+)\.self_attn\.(\d+)\.", weight_name)
    if match:
        layer_id = int(match.group(1))
        attn_num = int(match.group(2))
        return layer_id * 2 + attn_num
    match = re.search(r"layers\.(\d+)\.", weight_name)
    if match:
        return int(match.group(1))
    return None


# Attention CP utils
@dataclass
class ContextParallelMetadata:
    split_list: list[int] = None
    inverse_split_list: list[int] = None
    zigzag_index: list[int] = None
    inverse_index: list[int] = None
    per_rank_actual_token: list[int] = None
    cur_split_list: list[int] = None


class CPMetadataContainer:
    """Container class for storing global CP_METADATA"""

    def __init__(self):
        self.value: ContextParallelMetadata | None = None

    def set(self, metadata: ContextParallelMetadata | None):
        self.value = metadata

    def get(self) -> ContextParallelMetadata | None:
        return self.value

    def __bool__(self):
        """Support if CP_METADATA: syntax"""
        return self.value is not None


CP_METADATA = CPMetadataContainer()


def reset_cp_metadata():
    """Clear the global QCP metadata before a forward.

    CP_METADATA is only valid for the extend batch that set it; decode /
    target-verify / draft-extend / idle forwards in the same process must not
    observe a stale value from a previous prefill.
    """
    CP_METADATA.set(None)


def get_cp_metadata(
    token_len,
    cp_rank,
    cp_size,
):
    """prepare_input_dp_with_cp_dsa-zigzag index
    Example (DP_ATTENT_TP == CP_SIZE == 4):
    Description:
    1. Start with a full-length request.
    2. Split the request into multiple blocks (block0 to block7).
    3. Rearrange these blocks to balance computational
        load across different DP ranks.
    4. Assign the rearranged blocks to different DP attention
        time points (dp_atten_tp0 to dp_atten_tp3).
    +---------------------------------+
    |        cp_split_tokens         |
    +---------------------------------+
    |                                 |
    |   request_with_full_length     |
    |             | split (cp_size * 2) |
    |   +-------------------------+  |
    |   | block0 | block1 | block2 | block3 | block4 | block5 | block6 | block7 |
    |   +-------------------------+  |
    |             | rerange          |
    |   +---------------------------------+
    |   | block0 | block7 | block1 | block6 | block2 | block5 | block3 | block4 |
    |   +---------------------------------+
    |             |
    |   +-------------------------+
    |   | dp_atten_tp0: block0, block7 |
    |   | dp_atten_tp1: block1, block6 |
    |   | dp_atten_tp2: block2, block5 |
    |   | dp_atten_tp3: block3, block4 |
    |   +-------------------------+

    Why zigzag rearrange?
    - Attention calculations must follow causal attention principles.
    - Simply slicing by rank order can lead to computational load imbalance:
        * First rank may focus on fewer historical key-value tokens (less computation)
        * Last rank may focus on more tokens (more computation)
    - To mitigate uneven load, the input hissenstate needs to be sliced by cp_size*2 and rearranged.
    """
    # just support batch = 1
    # get zigzag index
    cp_segment_num = cp_size * 2
    segment_tokens = token_len // cp_segment_num
    remainder = token_len % cp_segment_num
    split_list = [segment_tokens + 1] * remainder + [segment_tokens] * (cp_segment_num - remainder)

    zigzag_index = [cp_rank, cp_segment_num - 1 - cp_rank]

    inverse_split_list = [e for i in range(cp_size) for e in (split_list[i], split_list[cp_segment_num - 1 - i])]

    per_rank_actual_token = list(split_list[i] + split_list[cp_segment_num - 1 - i] for i in range(cp_size))

    # TODO Support multi-batch-cp-split, multi-batch-cp support has accuracy issues
    # cp_seq_index = calculate_cp_seq_idx(split_list[:], seqs_len[:])
    cp_metadata = ContextParallelMetadata(
        split_list=split_list,
        inverse_split_list=inverse_split_list,
        zigzag_index=zigzag_index,
        # inverse_index=inverse_index,
        per_rank_actual_token=per_rank_actual_token,
        tokens_prev=split_list[cp_rank],
        tokens_cur=split_list[cp_segment_num - cp_rank - 1],
    )
    return cp_metadata


def get_cp_metadata_v2(
    cp_rank: int,
    cp_size: int,
    seqlens_q: torch.Tensor | list,
):
    bs = len(seqlens_q)
    cp_segment_num = cp_size * 2
    split_list = []
    for s in seqlens_q:
        base = s // cp_segment_num
        rem = s % cp_segment_num
        blk = [base + 1] * rem + [base] * (cp_segment_num - rem)
        split_list.extend(blk)

    all_zigzag_index = []
    for r in range(cp_size):
        for b in range(bs):
            base = b * cp_segment_num
            all_zigzag_index.extend([base + r, base + cp_segment_num - 1 - r])

    zigzag_index = all_zigzag_index[cp_rank * bs * 2 : (cp_rank + 1) * bs * 2]

    inverse_split_list = [split_list[i] for i in all_zigzag_index]
    inverse_index = [0] * len(all_zigzag_index)
    for i, a in enumerate(all_zigzag_index):
        inverse_index[a] = i

    per_rank_actual_token = [sum(inverse_split_list[r * bs * 2 : (r + 1) * bs * 2]) for r in range(cp_size)]

    cur_split_list = [split_list[i] for i in zigzag_index]

    cp_metadata = ContextParallelMetadata(
        split_list=split_list,
        inverse_split_list=inverse_split_list,
        zigzag_index=zigzag_index,
        inverse_index=inverse_index,
        per_rank_actual_token=per_rank_actual_token,
        cur_split_list=cur_split_list,
    )
    return cp_metadata


def cp_split_and_rebuild_data(x: torch.Tensor, split_list, zigzag_index):
    x_list = list(torch.split(x, split_list, dim=0))
    result = torch.cat([x_list[i] for i in zigzag_index], dim=0)
    return result


def cp_split_to_tuple(x: torch.Tensor, split_list, zigzag_index):
    x_list = torch.split(x, split_list, dim=0)
    result = tuple(x_list[i] for i in zigzag_index)
    return result


def cp_all_gather_rerange_output(x, cp_metadata: ContextParallelMetadata, comm_convertor: TritonRSAG):
    """
    |   +-----------before allgather------------+|
    |   | dp_atten_tp0: block0, block7 |
    |   | dp_atten_tp1: block1, block6 |
    |   | dp_atten_tp2: block2, block5 |
    |   | dp_atten_tp3: block3, block4 |
    |
    |   +----------before rerange---------------+|
    | block0 | block7 | block1 | block6 | block2 | block5 | block3 | block4 |
    |
    |   +--------------result-------------------+
    | block0 | block1 | block2 | block3 | block4 | block5 | block6 | block7 |
    |   +-------------------------+
    """
    num_tokens = sum(cp_metadata.per_rank_actual_token)
    x = comm_convertor.all_gather(x, num_tokens, cp_metadata.per_rank_actual_token)
    inverse_index = cp_metadata.inverse_index
    x_list = torch.split(x, cp_metadata.inverse_split_list)
    output = torch.cat([x_list[i] for i in inverse_index])
    return output


def cp_all_gather_torch_rerange_output(
    x: torch.Tensor,
    cp_metadata: ContextParallelMetadata,
    comm_group: GroupCoordinator,
):
    num_tokens = sum(cp_metadata.per_rank_actual_token)
    gx = x.new_empty(num_tokens, x.shape[-1])
    tensor_list = list(gx.split(cp_metadata.per_rank_actual_token))
    torch.distributed.all_gather(tensor_list, x, group=comm_group.device_group)
    inverse_index = cp_metadata.inverse_index
    x_list = torch.split(gx, cp_metadata.inverse_split_list)
    output = torch.cat([x_list[i] for i in inverse_index])
    return output


class PPMissingLayer(torch.nn.Identity):
    # Adapted from
    # https://github.com/vllm-project/vllm/blob/18ed3132d2bfe1df9a74729457b69243955221e8/vllm/model_executor/models/utils.py#L468C1-L486C1
    """
    A placeholder layer for missing layers in a pipeline parallel model.

    Attribute access on this placeholder returns 'self' so that common
    patterns like ``isinstance(layer.mlp, SomeMoE)`` safely evaluate to
    ``False`` instead of raising ``AttributeError``.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.return_tuple = kwargs.get("return_tuple", False)
        self.return_index = kwargs.get("return_index", 0)
        self.return_indices = kwargs.get("return_indices")
        self.return_kwarg_names = kwargs.get("return_kwarg_names")

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except (AttributeError, KeyError):
            return self

    def __setattr__(self, name: str, value):
        if name in {"return_tuple", "return_index", "return_indices", "return_kwarg_names"} or name.startswith("_"):
            super().__setattr__(name, value)
        else:
            pass

    def forward(self, *args, **kwargs):
        """
        Return the first arg from args or the first value from kwargs.

        Wraps the input in a tuple if `self.return_tuple` is True.
        If `return_indices` or `return_kwarg_names` is set, returns those
        values in order as a tuple.
        """
        if len(args) == 1 and len(kwargs.keys()) == 0:
            return args[0]
        if self.return_tuple:
            outputs = []
            for index in self.return_indices or ():
                outputs.append(args[index] if args else next(iter(kwargs.values())))
            for name in self.return_kwarg_names or ():
                outputs.append(kwargs.get(name))
            return tuple(outputs)
        else:
            input = args[self.return_index] if args else next(iter(kwargs.values()))
            return input
