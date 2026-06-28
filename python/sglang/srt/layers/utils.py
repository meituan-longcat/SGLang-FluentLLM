from __future__ import annotations
import logging
import re
from itertools import accumulate
from dataclasses import dataclass
from typing import List, Optional, TYPE_CHECKING
if TYPE_CHECKING:
    from sglang.srt.distributed.device_communicators.custom_triton_rsag.triton_rsag import TritonRSAG

import torch
from sglang.srt.utils import get_bool_env_var

logger = logging.getLogger(__name__)


def get_layer_id(weight_name):
    # example weight name: model.layers.10.self_attn.qkv_proj.weight
    match = re.search(r"layers\.(\d+)\.", weight_name)
    if match:
        return int(match.group(1))
    return None

# Attention CP utils
@dataclass
class ContextParallelMetadata:
    split_list: List[int] = None
    inverse_split_list: List[int] = None
    max_token_len_in_block: List[int] = None
    zigzag_index: List[int] = None
    per_rank_actual_token: List[int] = None
    prefix_sum_tokens_prev: int = -1
    prefix_sum_tokens_cur: int = -1
    tokens_prev: int = -1
    tokens_cur: int = -1
    total_token_len: int = -1

class CPMetadataContainer:
    """Container class for storing global CP_METADATA"""
    def __init__(self):
        self.value: Optional[ContextParallelMetadata] = None
    def set(self, metadata: Optional[ContextParallelMetadata]):
        self.value = metadata
    def get(self) -> Optional[ContextParallelMetadata]:
        return self.value
    def __bool__(self):
        """Support if CP_METADATA: syntax"""
        return self.value is not None

CP_METADATA = CPMetadataContainer()
FLLM_IS_CP = get_bool_env_var("FLLM_IS_CP", "false")

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
    bs_per_cp_group = 1
    # get zigzag index
    cp_segment_num = cp_size * 2
    segment_tokens = token_len // cp_segment_num
    remainder = token_len % cp_segment_num
    split_list = [segment_tokens + 1] * remainder + [segment_tokens] * (cp_segment_num - remainder)

    max_token_len_in_block = (token_len + cp_size - 1) // cp_size

    zigzag_index = [cp_rank, cp_segment_num-1-cp_rank]

    inverse_split_list = [
        e for i in range(cp_size)
        for e in (split_list[i], split_list[cp_segment_num - 1 - i])
    ]

    per_rank_actual_token = list(
        split_list[i] + split_list[cp_segment_num - 1 - i] for i in range(cp_size)
    )
    prefix_sum_list = list(accumulate(split_list))

    # TODO Support multi-batch-cp-split, multi-batch-cp support has accuracy issues
    # cp_seq_index = calculate_cp_seq_idx(split_list[:], seqs_len[:])
    cp_metadata = ContextParallelMetadata(
        split_list=split_list,
        inverse_split_list=inverse_split_list,
        max_token_len_in_block=max_token_len_in_block,
        zigzag_index=zigzag_index,
        per_rank_actual_token=per_rank_actual_token,
        prefix_sum_tokens_prev=prefix_sum_list[cp_rank],
        prefix_sum_tokens_cur=prefix_sum_list[cp_segment_num - cp_rank - 1],
        tokens_prev=split_list[cp_rank],
        tokens_cur=split_list[cp_segment_num - cp_rank - 1],
        total_token_len=token_len,
    )
    return cp_metadata


def cp_split_and_rebuild_data(
    x: torch.Tensor, split_list, zigzag_index
):
    x_list = list(
        torch.split(x, split_list, dim=0)
    )
    result = torch.cat(
        [x_list[i] for i in zigzag_index], dim=0
    )
    return result

def cp_split_to_tuple(
    x: torch.Tensor, split_list, zigzag_index
):
    x_list = torch.split(x, split_list, dim=0)
    result = tuple(x_list[i] for i in zigzag_index)
    return result

def cp_all_gather_rerange_output(
    x,
    cp_metadata: ContextParallelMetadata,
    comm_convertor: TritonRSAG
):
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
    cp_segment_num = len(cp_metadata.split_list)
    inverse_index = list(range(0, cp_segment_num, 2)) + list(range(cp_segment_num-1, 0, -2))
    x_list = torch.split(x, cp_metadata.inverse_split_list)
    output = torch.cat(
        [x_list[i] for i in inverse_index]
    )
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

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except (AttributeError, KeyError):
            return self

    def __setattr__(self, name: str, value):
        if name == "return_tuple" or name.startswith("_"):
            super().__setattr__(name, value)
        else:
            pass

    def forward(self, *args, **kwargs):
        """
        Return the first arg from args or the first value from kwargs.

        Wraps the input in a tuple if `self.return_tuple` is True.
        """
        input = args[0] if args else next(iter(kwargs.values()))
        return (input,) if self.return_tuple else input
