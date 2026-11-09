import torch
import torch.distributed._symmetric_memory as symm_mem
import torch.distributed as dist
from typing import Tuple, List
import triton
import triton.language as tl
from sglang.srt.distributed.device_communicators.custom_triton_rsag.triton_barrier import blockwise_barrier
from sglang.srt.distributed.device_communicators.custom_triton_rsag.triton_utils import (
    get_flat_tid,
    local_ld_128,
    local_st_128,
    multimem_ld_reduce_128,
    multimem_st_128,
    sync_threads,
)
from sglang.srt.distributed import GroupCoordinator
from sglang.srt.utils import get_available_gpu_memory, get_colorful_logger


logger = get_colorful_logger(__file__)

""" in-group reduce_scatter and all_gather, similar to torch.distributed.reduce_scatter and torch.distributed.all_gather """


class TritonRSAG(object):
    def __init__(self, group: dist.ProcessGroup, rank_in_group: int, max_tokens: int, hidden_size: int, device: torch.device = None) -> None:
        if type(group) == dist.ProcessGroup:
            self.group = group
        elif type(group) == GroupCoordinator:
            self.group = group.cpu_group  # TODO: why cpu_group
        if device == None:
            self.device = torch.device(f"cuda:{torch.cuda.current_device()}")
        else:
            self.device = device
        self.rank_in_group = rank_in_group
        self.max_tokens = max_tokens
        self.hidden_size = hidden_size
        # Only bfloat16 is supported for now
        self.dtype = torch.bfloat16
        free_gpu_memory_begin = get_available_gpu_memory("cuda", torch.cuda.current_device())
        self.comm_buff = symm_mem.empty((max_tokens, hidden_size), dtype=torch.bfloat16, device=self.device)
        free_gpu_memory_after = get_available_gpu_memory("cuda", torch.cuda.current_device())
        logger.info(f"Custom Triton RSAG buffer allocated: {free_gpu_memory_begin - free_gpu_memory_after} GB")
        symm_mem.rendezvous(self.comm_buff, self.group.group_name)

    def get_token_dist(self, total_tokens_in_group: int) -> list:
        world_size = self.group.size()
        token_list_in_group = []
        for rank in range(0, world_size):
            num_tokens_per_rank = total_tokens_in_group // world_size + (1 if (rank < total_tokens_in_group % world_size) else 0)
            token_list_in_group.append(num_tokens_per_rank)
        return token_list_in_group

    def get_context(
        self,
        token_list_in_group: list,
    ) -> Tuple[int, int, int]:
        """token_list_in_group records tokens per rank in the comm group"""

        total_num_tokens = sum(token_list_in_group)
        assert total_num_tokens <= self.max_tokens, f"The inner comm buffer is too small: {total_num_tokens=} is not <= {self.max_tokens=}"

        local_num_tokens = token_list_in_group[self.rank_in_group]
        local_token_offset = sum(token_list_in_group[: self.rank_in_group])

        return total_num_tokens, local_num_tokens, local_token_offset

    def get_luanch_config(self, local_numel: int) -> Tuple[int, int, int, int]:
        WARP_SIZE = 32
        MAX_NUM_BLOCKS = 4
        MAX_BLOCK_SIZE = 1024
        BYTES_PER_THREAD = 16

        numel_per_thread = BYTES_PER_THREAD // self.dtype.itemsize

        """ Assume 16 bytes alignment now """
        assert local_numel % numel_per_thread == 0, f"The number of elements must be {BYTES_PER_THREAD} bytes aligned"

        """ If token's distribution is unbalanced, we must force each rank has the same number of thread blocks to fix hang """

        block_size = MAX_BLOCK_SIZE
        num_warps = MAX_BLOCK_SIZE // WARP_SIZE
        num_blocks = MAX_NUM_BLOCKS

        return num_blocks, block_size, num_warps, numel_per_thread

    @triton.jit
    def multimem_reduce_scatter_kernel(
        output_ptr,
        multicast_ptr,
        signal_pad_ptr,
        numel,
        offset,
        BLOCK_SIZE: tl.constexpr,
        NUMEL_PER_THREAD: tl.constexpr,
        RANK: tl.constexpr,
        WORLD_SIZE: tl.constexpr,
    ) -> None:
        """input data ready"""
        blockwise_barrier(signal_pad_ptr, None, RANK, WORLD_SIZE, sem="relaxed")
        sync_threads()

        numel = numel // NUMEL_PER_THREAD

        pid = tl.program_id(axis=0)
        tid = get_flat_tid()
        block_start = pid * BLOCK_SIZE

        while block_start < numel:
            thread_offset = block_start + tid
            mask = thread_offset < numel

            in_ptr = multicast_ptr.to(tl.pointer_type(tl.uint64)) + (offset // NUMEL_PER_THREAD + thread_offset) * 2
            out_ptr = output_ptr.to(tl.pointer_type(tl.uint64)) + (offset // NUMEL_PER_THREAD + thread_offset) * 2
            # load global comm buff, store local
            (x, y, z, w) = multimem_ld_reduce_128(in_ptr, mask)
            local_st_128(out_ptr, x, y, z, w, mask)

            block_start += tl.num_programs(axis=0) * BLOCK_SIZE

        # beacause each rank has its own local buff to store
        sync_threads()
        blockwise_barrier(signal_pad_ptr, None, RANK, WORLD_SIZE, sem="acq_rel")

    @triton.jit
    def multimem_all_gather_kernel(
        input_ptr,
        multicast_ptr,
        signal_pad_ptr,
        numel,
        offset,
        BLOCK_SIZE: tl.constexpr,
        NUMEL_PER_THREAD: tl.constexpr,
        RANK: tl.constexpr,
        WORLD_SIZE: tl.constexpr,
    ) -> None:
        # because each rank has its own local buff to load
        blockwise_barrier(signal_pad_ptr, None, RANK, WORLD_SIZE, sem="relaxed")
        sync_threads()

        numel = numel // NUMEL_PER_THREAD

        pid = tl.program_id(axis=0)
        tid = get_flat_tid()
        block_start = pid * BLOCK_SIZE

        while block_start < numel:
            thread_offset = block_start + tid
            mask = thread_offset < numel

            in_ptr = input_ptr.to(tl.pointer_type(tl.uint64)) + (offset // NUMEL_PER_THREAD + thread_offset) * 2
            out_ptr = multicast_ptr.to(tl.pointer_type(tl.uint64)) + (offset // NUMEL_PER_THREAD + thread_offset) * 2
            x, y, z, w = local_ld_128(in_ptr, mask)
            multimem_st_128(out_ptr, x, y, z, w, mask)

            block_start += tl.num_programs(axis=0) * BLOCK_SIZE

        """ all_gather done """
        sync_threads()
        blockwise_barrier(signal_pad_ptr, None, RANK, WORLD_SIZE, sem="acq_rel")

    def multimem_reduce_scatter(self, local_num_tokens: int, local_token_offset: int) -> None:
        num_elts = local_num_tokens * self.hidden_size
        num_blocks, block_size, num_warps, numel_per_thread = self.get_luanch_config(num_elts)

        symm_mem_hdl = symm_mem.rendezvous(self.comm_buff, group=self.group)
        assert self.rank_in_group == symm_mem_hdl.rank, "Mismatched rank id"

        self.multimem_reduce_scatter_kernel[(num_blocks, 1, 1)](
            output_ptr=self.comm_buff,
            multicast_ptr=symm_mem_hdl.multicast_ptr,
            signal_pad_ptr=symm_mem_hdl.signal_pad_ptrs_dev,
            numel=local_num_tokens * self.hidden_size,
            offset=local_token_offset * self.hidden_size,
            BLOCK_SIZE=block_size,
            NUMEL_PER_THREAD=numel_per_thread,
            RANK=symm_mem_hdl.rank,
            WORLD_SIZE=symm_mem_hdl.world_size,
            num_warps=num_warps,
        )

    def multimem_all_gather(self, local_num_tokens: int, local_token_offset: int) -> None:
        num_elts = local_num_tokens * self.hidden_size
        num_blocks, block_size, num_warps, numel_per_thread = self.get_luanch_config(num_elts)

        symm_mem_hdl = symm_mem.rendezvous(self.comm_buff, group=self.group)
        assert self.rank_in_group == symm_mem_hdl.rank, "Mismatched rank id"

        self.multimem_all_gather_kernel[(num_blocks, 1, 1)](
            input_ptr=self.comm_buff,
            multicast_ptr=symm_mem_hdl.multicast_ptr,
            signal_pad_ptr=symm_mem_hdl.signal_pad_ptrs_dev,
            numel=local_num_tokens * self.hidden_size,
            offset=local_token_offset * self.hidden_size,
            BLOCK_SIZE=block_size,
            NUMEL_PER_THREAD=numel_per_thread,
            RANK=symm_mem_hdl.rank,
            WORLD_SIZE=symm_mem_hdl.world_size,
            num_warps=num_warps,
        )

    def reduce_scatter(
        self, hidden_states: torch.tensor, tp_num_tokens: int = None, token_list_in_group: List[int] = None, safe=True
    ) -> Tuple[torch.Tensor, int]:
        assert tp_num_tokens is not None or token_list_in_group is not None, "Either tp_num_tokens or token_list_in_group must be provided"

        if token_list_in_group is None:
            token_list_in_group = self.get_token_dist(tp_num_tokens)

        assert hidden_states.dtype == torch.bfloat16, "Only bfloat16 is supported for now"

        total_num_tokens, local_num_tokens, local_token_offset = self.get_context(token_list_in_group)

        assert (hidden_states.shape[0] == total_num_tokens) and (hidden_states.shape[-1] == self.hidden_size), (
            f"Dismatched shape, {hidden_states.shape[0]=} != {total_num_tokens=} or {hidden_states.shape[-1]=} != {self.hidden_size=} {hidden_states.shape=}"
        )

        self.comm_buff[:total_num_tokens, :].copy_(hidden_states)
        self.multimem_reduce_scatter(local_num_tokens, local_token_offset)

        """ a slice view of comm buff """
        output = self.comm_buff[local_token_offset : (local_token_offset + local_num_tokens), :]

        if safe:
            output = output.clone()

        return output, local_token_offset

    def all_gather(self, hidden_states: torch.tensor, tp_num_tokens: int = None, token_list_in_group: List[int] = None, safe=True) -> torch.Tensor:
        assert tp_num_tokens is not None or token_list_in_group is not None, "Either tp_num_tokens or token_list_in_group must be provided"

        if token_list_in_group is None:
            token_list_in_group = self.get_token_dist(tp_num_tokens)

        assert hidden_states.dtype == torch.bfloat16, "Only bfloat16 is supported for now"

        total_num_tokens, local_num_tokens, local_token_offset = self.get_context(token_list_in_group)

        assert (hidden_states.shape[0] == local_num_tokens) and (hidden_states.shape[-1] <= self.hidden_size), (
            f"{hidden_states.shape=}|{local_num_tokens=}|{hidden_states.device=} Dismatched shape, {token_list_in_group=}"
        )

        hidden_size_bak, comm_buff_bak = self.hidden_size, self.comm_buff
        if hidden_states.shape[-1] < hidden_size_bak:
            self.hidden_size = hidden_states.shape[-1]
            self.comm_buff = comm_buff_bak.reshape(-1)[: self.max_tokens * self.hidden_size].reshape(self.max_tokens, self.hidden_size)

        # TODO: the copy operation is cheap
        self.comm_buff[local_token_offset : (local_token_offset + local_num_tokens), :].copy_(hidden_states)
        self.multimem_all_gather(local_num_tokens, local_token_offset)

        """ a slice view of comm buff """
        output = self.comm_buff[:total_num_tokens, :]

        if safe:
            output = output.clone()

        if hidden_states.shape[-1] < hidden_size_bak:
            self.hidden_size = hidden_size_bak
            self.comm_buff = comm_buff_bak

        return output
