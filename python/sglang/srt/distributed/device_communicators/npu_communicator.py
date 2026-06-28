import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from sglang.srt.utils import is_npu, get_prefix_sum

__is_npu__ = is_npu()
if __is_npu__:
    import torch_npu

class NpuCommunicator:

    def __init__(self, group: ProcessGroup):
        if not is_npu():
            self.disabled = True
            return
        self.disabled = False
        self.group = group
        self.rank_in_group = dist.get_rank(group)
        self.world_size = dist.get_world_size(self.group)

    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        dist.all_reduce(x, group=self.group)
        return x

    def is_even(self, input_split_sizes):
        if max(input_split_sizes) == min(input_split_sizes):
            return False
        else:
            return True

    def all_gather(self, x: torch.Tensor, dim: int = -1, output_split_sizes=None) -> torch.Tensor:
        world_size = self.world_size
        if dim < 0:
            # Convert negative dim to positive.
            dim += x.dim()

        if output_split_sizes is None or not self.is_even(output_split_sizes):
            input_size = x.size()
            output_size = (input_size[0] * world_size,) + input_size[1:]
            # Allocate output tensor.
            output_tensor = torch.empty(output_size, dtype=x.dtype, device=x.device)
            # All-gather.
            dist.all_gather_into_tensor(output_tensor, x, group=self.group)
            # Reshape
            output_tensor=output_tensor.reshape((world_size,)+input_size)
            output_tensor=output_tensor.movedim(0, dim)
            output_tensor=output_tensor.reshape(
                input_size[:dim]+(world_size*input_size[dim],)+input_size[dim+1:]
            )
        else:
            assert dim==0, "dim must be 0 when use output_split_sizes"
            output_tensor=torch.empty((sum(output_split_sizes),)+x.size()[1:], dtype=x.dtype, device=x.device)
            torch_npu.distributed.all_gather_into_tensor_uneven(output_tensor, x, output_split_sizes, group=self.group)
        return output_tensor

    def all_gather_into_tensor_uneven(
        self,
        output_tensor: torch.Tensor,
        input_tensor: torch.Tensor,
        output_split_sizes: list,
        group=None,
    ):
        max_size=max(output_split_sizes)
        my_size=input_tensor.size(0)
        if my_size<max_size:
            pad_size=max_size-my_size
            # 在第0维后面填充 (batch维度)
            pad=torch.zeros((pad_size,)+input_tensor.size()[1:], dtype=input_tensor.dtype, device=input_tensor.device)
            padded_input=torch.cat([input_tensor, pad], dim=0)
        else:
            padded_input=input_tensor

        # 分配接收缓冲区 [world_size, max_size, ...]
        gathered=torch.empty(
            (self.world_size, max_size)+input_tensor.size()[1:],
            dtype=input_tensor.dtype,
            device=input_tensor.device
        )

        dist.all_gather_into_tensor(gathered, padded_input, group=group)
        offset=0
        for i, size in enumerate(output_split_sizes):
            output_tensor[offset:offset+size].copy_(gathered[i, :size])
            offset+=size

        return output_tensor

    def all_gather_into_tensor(self, output: torch.Tensor, input: torch.Tensor):
        dist.all_gather_into_tensor(output, input, group=self.group)

    def reduce_scatter(self, x: torch.Tensor, input_split_sizes=None) -> torch.Tensor:
        x_size = tuple(x.size())
        if input_split_sizes is None or not self.is_even(input_split_sizes):
            output_tensor=torch.empty((x_size[0]//self.world_size,)+x_size[1:],
                                      dtype=x.dtype,
                                      device=x.device)
            torch.distributed.reduce_scatter_tensor(output_tensor,
                                                    x,
                                                    group=self.group)
        else:
            output_tensor=torch.empty((input_split_sizes[self.rank_in_group],)+x_size[1:],
                                   dtype=x.dtype, device=x.device)
            torch_npu.distributed.reduce_scatter_tensor_uneven(output_tensor, x, input_split_sizes,
                                                               group=self.group)
        return output_tensor

    def reduce_scatter_tensor_uneven(
        self,
        output_tensor: torch.Tensor,
        input_tensor: torch.Tensor,
        input_split_sizes: list,
        group=None,
    ):
        max_size=max(input_split_sizes)
        my_size=input_split_sizes[self.rank_in_group]

        input_offsets=get_prefix_sum(input_split_sizes)

        # 将输入按 split_sizes 分块，每块填充到 max_size
        padded_chunks=[]
        for i, size in enumerate(input_split_sizes):
            start=input_offsets[i]
            end=input_offsets[i+1]
            chunk=input_tensor[start:end]
            if size<max_size:
                pad_size=max_size-size
                pad=torch.zeros((pad_size,)+input_tensor.size()[1:],
                                dtype=input_tensor.dtype,
                                device=input_tensor.device)
                chunk=torch.cat([chunk, pad], dim=0)
            padded_chunks.append(chunk)

        # 拼接成填充后的输入 [world_size * max_size, ...]
        padded_input=torch.cat(padded_chunks, dim=0)

        dist.reduce_scatter_tensor(output_tensor, padded_input, group=group)

        # 截取实际需要的部分
        return output_tensor[:my_size]
