from typing import List, Optional
import itertools

import torch
import torch.distributed as dist
from dataclasses import dataclass, fields

from sglang.srt.distributed import (
    TensorMetadata, get_ep_group, get_mlp_tp_group, get_attn_cross_group, get_attn_tp_world_size
)

from sglang.srt.env import ENV
from sglang.srt.utils import get_colorful_logger
logger = get_colorful_logger(__name__)

@dataclass
class DPAllGatherBuffer:
    # hidden states
    padded_hs: torch.Tensor = None
    partial_hs: torch.Tensor = None
    partial_global_padded_hs: torch.Tensor = None
    global_padded_hs_trans: torch.Tensor = None
    global_padded_hs: torch.Tensor = None

    @torch.inference_mode()
    def __init__(self, ep_size, max_token_num_across_ep: int, tensor_meta: TensorMetadata):
        self.padded_hs = torch.empty((max_token_num_across_ep, tensor_meta.size),
                                      dtype=tensor_meta.dtype, device=tensor_meta.device)
        self.global_padded_hs = torch.empty((ep_size, max_token_num_across_ep, tensor_meta.size),
                                             dtype=tensor_meta.dtype, device=tensor_meta.device)

    def mark_static(self):
        for field in fields(self):
            if (field_value := getattr(self, field.name)) is not None:
                torch._dynamo.mark_static(field_value)

class EPMetadata:
    # todo remove this
    """Expert metadata."""
    # [ep_size + 2], -2 value for decode_only & graph pad info, -1 value for decode only info (No matter if graph mode is turned on or off)
    query_lens_across_ep_tensor: Optional[torch.Tensor] = None # create once

    @torch.inference_mode()
    def __init__(self, dtype, hidden_size, query_len: int, decode_only: bool, decode_only_without_graph: bool, device: torch.device):
        self.tensor_meta = TensorMetadata(device, dtype, hidden_size)
        ep_group = get_ep_group()
        self.ep_rank = ep_group.rank_in_group
        self.ep_size = ep_group.world_size

        if self.ep_size > 1:
            if EPMetadata.query_lens_across_ep_tensor is not None:
                EPMetadata.query_lens_across_ep_tensor.zero_()
                EPMetadata.query_lens_across_ep_tensor[self.ep_rank] = query_len
                EPMetadata.query_lens_across_ep_tensor[-2] = decode_only
                EPMetadata.query_lens_across_ep_tensor[-1] = decode_only_without_graph
            else:
                num_tokens_across_ep = [0] * (self.ep_size + 2)
                num_tokens_across_ep[self.ep_rank] = query_len
                num_tokens_across_ep[-2] = decode_only
                num_tokens_across_ep[-1] = decode_only_without_graph
                EPMetadata.query_lens_across_ep_tensor = torch.tensor(num_tokens_across_ep,
                                                        device=device,
                                                        dtype=torch.int32)
            self.work = dist.all_reduce(EPMetadata.query_lens_across_ep_tensor, group=ep_group.device_group, async_op=True)
        else:
            self.work = None
        # local ep rank query start location across ep
        self.local_query_start_loc = 0
        # local ep rank query end location across ep
        self.local_query_end_loc = -1

        self.max_token_num_across_ep = query_len
        self.decode_only_across_ep: bool = ENV.npu_enable_graph and decode_only
        self.ep_world_query_lens: List[int] = None

        # all gather buffer
        self.all_gather_buffer: Optional[DPAllGatherBuffer] = None
        self.all2all_available = None
        self.exclude_prefill = None

        if True: # ENV.npu_enable_graph?
            self.post_init()

    @torch.inference_mode()
    def post_init(self):
        """as late as possible to overlay comm"""
        if self.work is None:
            return

        self.work.wait()
        self.work = None
        tp_size = 1

        # torch compile requires cpu list
        origin_ep_world_q_len = EPMetadata.query_lens_across_ep_tensor.tolist()
        self.ep_world_query_lens = origin_ep_world_q_len.copy()
        if self.decode_only_across_ep:
            self.decode_only_across_ep = self.ep_world_query_lens[-2] == self.ep_size
        self.ep_world_query_lens = self.ep_world_query_lens[:-2]
        self.max_token_num_across_ep = max(self.ep_world_query_lens)
        if self.decode_only_across_ep:
            self.ep_world_query_lens = [self.max_token_num_across_ep] * self.ep_size
        attn_tp_size = get_attn_tp_world_size()
        cu_tokens_across_ep_cpu = list(itertools.accumulate(self.ep_world_query_lens[::attn_tp_size]))
        query_index = self.ep_rank // attn_tp_size
        self.local_query_start_loc = 0 if query_index == 0 else cu_tokens_across_ep_cpu[query_index - 1]
        self.local_query_end_loc = cu_tokens_across_ep_cpu[query_index]

        # for mlp tp
        mlp_tp_size = get_mlp_tp_group().world_size
        mlp_group_start_in_ep = self.ep_rank //  mlp_tp_size *  mlp_tp_size // attn_tp_size
        mlp_group_end_in_ep =  (self.ep_rank // mlp_tp_size + 1) *  mlp_tp_size // attn_tp_size
        mlp_local_query_offset = 0 if mlp_group_start_in_ep == 0 else cu_tokens_across_ep_cpu[int(mlp_group_start_in_ep) - 1]
        self.mlp_local_query_start_loc = 0 if query_index == mlp_group_start_in_ep else cu_tokens_across_ep_cpu[query_index - 1] - mlp_local_query_offset
        self.mlp_local_query_end_loc = cu_tokens_across_ep_cpu[query_index] - mlp_local_query_offset
        self.mlp_group_query_start_loc = 0 if mlp_group_start_in_ep == 0 else cu_tokens_across_ep_cpu[int(mlp_group_start_in_ep) - 1]
        self.mlp_group_query_end_loc = cu_tokens_across_ep_cpu[int(mlp_group_end_in_ep) - 1]

        self.exclude_prefill = all(element <= 256 for element in self.ep_world_query_lens) and origin_ep_world_q_len[-1] == self.ep_size

        self.all2all_available = all(element <= 256 for element in self.ep_world_query_lens) and origin_ep_world_q_len[-1] == self.ep_size and ENV.npu_enable_all2all_comm

        self.ep_world_query_token_num = sum(self.ep_world_query_lens)
        self.ep_world_query_token_num_dis = max(self.ep_world_query_lens)*get_ep_group().world_size

        # 两个条件：全局等长，能被tp整除
        self.no_padding = all(element == self.max_token_num_across_ep for element in self.ep_world_query_lens) and (self.max_token_num_across_ep % tp_size == 0)
        all_gather_word_size = get_attn_cross_group().world_size if attn_tp_size < self.ep_size else 1
        self.all_gather_buffer = DPAllGatherBuffer(
            all_gather_word_size, self.max_token_num_across_ep, self.tensor_meta,
        )

    def __repr__(self) -> str:
        return (f"EPMetadata("
                f"ep_rank={self.ep_rank}, "
                f"ep_size={self.ep_size}, "
                f"local_query_start_loc={self.local_query_start_loc}, "
                f"local_query_end_loc={self.local_query_end_loc}, "
                f"max_token_num_across_ep={self.max_token_num_across_ep}, "
                f"decode_only_across_ep={self.decode_only_across_ep}, "
                f"ep_world_query_lens={self.ep_world_query_lens}"
                f")")

    def get_exclude_prefill(self) -> bool:
        return self.exclude_prefill

    def get_all2all_available(self) -> bool:
        return self.all2all_available

    def get_ep_world_query_token_num(self):
        return self.ep_world_query_token_num

    def get_all_gather_buffer(self):
        return (self.all_gather_buffer.padded_hs, \
                self.all_gather_buffer.partial_hs, \
                self.all_gather_buffer.partial_global_padded_hs, \
                self.all_gather_buffer.global_padded_hs_trans, \
                self.all_gather_buffer.global_padded_hs)

    def get_ep_world_query_token_num(self):
        return self.ep_world_query_token_num

    def get_ep_world_query_token_num_dis(self):
        return self.ep_world_query_token_num_dis
