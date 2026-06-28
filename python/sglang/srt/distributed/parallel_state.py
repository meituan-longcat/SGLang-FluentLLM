# Adapted from https://github.com/vllm-project/vllm/blob/v0.6.4.post1/vllm/distributed/parallel_state.py

# Copyright 2023 The vLLM team.
# Adapted from
# https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/parallel_state.py
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
"""vLLM distributed state.
It takes over the control of the distributed environment from PyTorch.
The typical workflow is:

- call `init_distributed_environment` to initialize the distributed environment.
- call `initialize_model_parallel` or `ensure_model_parallel_initialized` to
 initialize the model parallel groups.

- any code dealing with the distributed stuff

- call `destroy_model_parallel` to destroy the model parallel groups.
- call `destroy_distributed_environment` to destroy the distributed environment.

If you only need to use the distributed environment without model/pipeline
 parallelism, you can skip the model parallel initialization and destruction
 steps.
"""
import contextlib
import gc

from sglang.srt.utils import get_colorful_logger, get_prefix_sum
import os
import pickle
import weakref
from collections import namedtuple
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from datetime import timedelta
from multiprocessing import shared_memory
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from unittest.mock import patch

import torch
import torch.distributed
from torch.distributed import Backend, ProcessGroup
from sglang.srt.env import ENV
from sglang.srt.env import global_server_args_dict
from sglang.srt.utils import (
    direct_register_custom_op,
    is_cuda_alike,
    is_npu,
    supports_custom_op,
)

__is_npu__ = is_npu()

if not __is_npu__:
    from eps.communication import MscclppCommunicator, MscclppCommunicatorParams
else:
    import torch_npu


@dataclass
class GraphCaptureContext:
    stream: torch.cuda.Stream

@dataclass
class P2PWork:
    work: Optional[torch.distributed.Work]
    payload: Optional[torch.Tensor]

TensorMetadata = namedtuple("TensorMetadata", ["device", "dtype", "size"])


def _split_tensor_dict(
    tensor_dict: Dict[str, Union[torch.Tensor, Any]]
) -> Tuple[List[Tuple[str, Any]], List[torch.Tensor]]:
    """Split the tensor dictionary into two parts:
    1. A list of (key, value) pairs. If the value is a tensor, it is replaced
         by its metadata.
    2. A list of tensors.
    """
    metadata_list: List[Tuple[str, Any]] = []
    tensor_list: List[torch.Tensor] = []
    for key, value in tensor_dict.items():
        if isinstance(value, torch.Tensor):
            # Note: we cannot use `value.device` here,
            # because it contains not only the device type but also the device
            # index (e.g. "cuda:0"). We only need the device type.
            # receiving side will set the device index.
            device = value.device.type
            metadata_list.append(
                (key, TensorMetadata(device, value.dtype, value.size()))
            )
            tensor_list.append(value)
        else:
            metadata_list.append((key, value))
    return metadata_list, tensor_list


_group_name_counter: Dict[str, int] = {}


def _get_unique_name(name: str) -> str:
    """Get a unique name for the group.
    Example:
    _get_unique_name("tp") -> "tp:0"
    _get_unique_name("tp") -> "tp:1"
    """
    if name not in _group_name_counter:
        _group_name_counter[name] = 0
    newname = f"{name}:{_group_name_counter[name]}"
    _group_name_counter[name] += 1
    return newname


_groups: Dict[str, Callable[[], Optional["GroupCoordinator"]]] = {}


def _register_group(group: "GroupCoordinator") -> None:
    _groups[group.unique_name] = weakref.ref(group)


if supports_custom_op():

    def inplace_all_reduce(tensor: torch.Tensor, group_name: str) -> None:
        assert group_name in _groups, f"Group {group_name} is not found."
        group = _groups[group_name]()
        if group is None:
            raise ValueError(f"Group {group_name} is destroyed.")
        group._all_reduce_in_place(tensor)

    def inplace_all_reduce_fake(tensor: torch.Tensor, group_name: str) -> None:
        return

    direct_register_custom_op(
        op_name="inplace_all_reduce",
        op_func=inplace_all_reduce,
        mutates_args=["tensor"],
        fake_impl=inplace_all_reduce_fake,
    )

    def outplace_all_reduce(tensor: torch.Tensor, group_name: str) -> torch.Tensor:
        assert group_name in _groups, f"Group {group_name} is not found."
        group = _groups[group_name]()
        if group is None:
            raise ValueError(f"Group {group_name} is destroyed.")
        return group._all_reduce_out_place(tensor)

    def outplace_all_reduce_fake(tensor: torch.Tensor, group_name: str) -> torch.Tensor:
        return torch.empty_like(tensor)

    direct_register_custom_op(
        op_name="outplace_all_reduce",
        op_func=outplace_all_reduce,
        mutates_args=[],
        fake_impl=outplace_all_reduce_fake,
    )

    def reg_all_gather_into_tensor(
        output: torch.Tensor, input: torch.Tensor, group_name: str
    ) -> None:
        assert group_name in _groups, f"Group {group_name} is not found."
        group = _groups[group_name]()
        if group is None:
            raise ValueError(f"Group {group_name} is destroyed.")
        group._all_gather_into_tensor(output, input)

    def reg_all_gather_into_tensor_fake(
        output: torch.Tensor, input: torch.Tensor, group_name: str
    ) -> None:
        pass

    direct_register_custom_op(
        op_name="reg_all_gather_into_tensor",
        op_func=reg_all_gather_into_tensor,
        mutates_args=["output"],
        fake_impl=reg_all_gather_into_tensor_fake,
    )


class GroupCoordinator:
    """
    PyTorch ProcessGroup wrapper for a group of processes.
    PyTorch ProcessGroup is bound to one specific communication backend,
        e.g. NCCL, Gloo, MPI, etc.
    GroupCoordinator takes charge of all the communication operations among
        the processes in the group. It can route the communication to
        a specific implementation (e.g. switch allreduce implementation
        based on the tensor size and cuda graph mode).
    """

    # available attributes:
    rank: int  # global rank
    ranks: List[int]  # global ranks in the group
    world_size: int  # size of the group
    # difference between `local_rank` and `rank_in_group`:
    # if we have a group of size 4 across two nodes:
    # Process | Node | Rank | Local Rank | Rank in Group
    #   0     |   0  |  0   |     0      |       0
    #   1     |   0  |  1   |     1      |       1
    #   2     |   1  |  2   |     0      |       2
    #   3     |   1  |  3   |     1      |       3
    local_rank: int  # local rank used to assign devices
    rank_in_group: int  # rank inside the group
    cpu_group: ProcessGroup  # group for CPU communication
    device_group: ProcessGroup  # group for device communication
    use_pynccl: bool  # a hint of whether to use PyNccl
    use_custom_allreduce: bool  # a hint of whether to use CustomAllreduce
    # communicators are only created for world size > 1
    pynccl_comm: Optional[Any]  # PyNccl communicator
    ca_comm: Optional[Any]  # Custom allreduce communicator
    mq_broadcaster: Optional[Any]  # shared memory broadcaster

    def __init__(
        self,
        group_ranks: List[List[int]],
        local_rank: int,
        torch_distributed_backend: Union[str, Backend],
        use_pynccl: bool,
        use_custom_allreduce: bool,
        use_hpu_communicator: bool,
        use_xpu_communicator: bool,
        use_npu_communicator: bool,
        use_message_queue_broadcaster: bool = False,
        group_name: Optional[str] = None,
        pg_options: Any = None,
    ):
        group_name = group_name or "anonymous"
        self.unique_name = _get_unique_name(group_name)
        _register_group(self)

        self.rank = torch.distributed.get_rank()
        self.local_rank = local_rank
        self.device_group = None
        self.cpu_group = None

        if __is_npu__ and global_server_args_dict["disaggregation_mode"] == "decode":
            if pg_options is None:
                pg_options=torch_npu._C._distributed_c10d.ProcessGroupHCCL.Options()
            if not isinstance(pg_options.hccl_config, dict):
                configs=dict(pg_options.hccl_config)
            else:
                configs=pg_options.hccl_config

            if "hccl_op_expansion_mode" not in configs.keys():
                configs["hccl_op_expansion_mode"]=3
            pg_options.hccl_config=configs

        for ranks in group_ranks:
            device_group = torch.distributed.new_group(
                ranks, backend=torch_distributed_backend, pg_options=pg_options
            )
            # a group with `gloo` backend, to allow direct coordination between
            # processes through the CPU.
            cpu_group = torch.distributed.new_group(ranks, backend="gloo")
            if self.rank in ranks:
                self.ranks = ranks
                self.world_size = len(ranks)
                self.rank_in_group = ranks.index(self.rank)
                self.device_group = device_group
                self.cpu_group = cpu_group

        assert self.cpu_group is not None
        assert self.device_group is not None

        if is_cuda_alike():
            self.device = torch.device(f"cuda:{local_rank}")
        elif is_npu():
            self.device = torch.device(f"npu:{local_rank}")
        else:
            self.device = torch.device("cpu")

        self.use_pynccl = use_pynccl
        self.use_custom_allreduce = use_custom_allreduce
        self.use_hpu_communicator = use_hpu_communicator
        self.use_xpu_communicator = use_xpu_communicator

        if not is_npu():
            # lazy import to avoid documentation build error
            from sglang.srt.distributed.device_communicators.custom_all_reduce import (
                CustomAllreduce,
            )
            from sglang.srt.distributed.device_communicators.pynccl import (
                PyNcclCommunicator,
            )

        self.pynccl_comm: Optional[PyNcclCommunicator] = None
        if use_pynccl and self.world_size > 1:
            self.pynccl_comm = PyNcclCommunicator(
                group=self.cpu_group,
                device=self.device,
            )

        self.ca_comm: Optional[CustomAllreduce] = None
        if use_custom_allreduce and self.world_size > 1:
            # Initialize a custom fast all-reduce implementation.
            self.ca_comm = CustomAllreduce(
                group=self.cpu_group,
                device=self.device,
            )

        from sglang.srt.distributed.device_communicators.hpu_communicator import (
            HpuCommunicator,
        )

        self.hpu_communicator: Optional[HpuCommunicator] = None
        if use_hpu_communicator and self.world_size > 1:
            self.hpu_communicator = HpuCommunicator(group=self.device_group)

        from sglang.srt.distributed.device_communicators.xpu_communicator import (
            XpuCommunicator,
        )

        self.xpu_communicator: Optional[XpuCommunicator] = None
        if use_xpu_communicator and self.world_size > 1:
            self.xpu_communicator = XpuCommunicator(group=self.device_group)

        from sglang.srt.distributed.device_communicators.npu_communicator import (
            NpuCommunicator,
        )

        self.npu_communicator: Optional[NpuCommunicator] = None
        if use_npu_communicator and self.world_size > 1:
            self.npu_communicator = NpuCommunicator(group=self.device_group)

        from sglang.srt.distributed.device_communicators.shm_broadcast import (
            MessageQueue,
        )

        self.mq_broadcaster: Optional[MessageQueue] = None
        if use_message_queue_broadcaster and self.world_size > 1:
            self.mq_broadcaster = MessageQueue.create_from_process_group(
                self.cpu_group, 1 << 22, 6
            )

    @property
    def first_rank(self):
        """Return the global rank of the first process in the group"""
        return self.ranks[0]

    @property
    def last_rank(self):
        """Return the global rank of the last process in the group"""
        return self.ranks[-1]

    @property
    def is_first_rank(self):
        """Return whether the caller is the first process in the group"""
        return self.rank == self.first_rank

    @property
    def is_last_rank(self):
        """Return whether the caller is the last process in the group"""
        return self.rank == self.last_rank

    @property
    def next_rank(self):
        """Return the global rank of the process that follows the caller"""
        rank_in_group = self.rank_in_group
        world_size = self.world_size
        return self.ranks[(rank_in_group + 1) % world_size]

    @property
    def prev_rank(self):
        """Return the global rank of the process that precedes the caller"""
        rank_in_group = self.rank_in_group
        world_size = self.world_size
        return self.ranks[(rank_in_group - 1) % world_size]

    @contextmanager
    def graph_capture(
        self, graph_capture_context: Optional[GraphCaptureContext] = None
    ):
        if graph_capture_context is None:
            stream = torch.cuda.Stream()
            graph_capture_context = GraphCaptureContext(stream)
        else:
            stream = graph_capture_context.stream

        ca_comm = self.ca_comm
        maybe_ca_context = nullcontext() if ca_comm is None else ca_comm.capture()

        # ensure all initialization operations complete before attempting to
        # capture the graph on another stream
        curr_stream = torch.cuda.current_stream()
        if curr_stream != stream:
            stream.wait_stream(curr_stream)

        with torch.cuda.stream(stream), maybe_ca_context:
            # In graph mode, we have to be very careful about the collective
            # operations. The current status is:
            #     allreduce \ Mode   |  Eager  |  Graph  |
            # --------------------------------------------
            # custom allreduce       | enabled | enabled |
            # PyNccl                 | disabled| enabled |
            # torch.distributed      | enabled | disabled|
            #
            # Note that custom allreduce will have a runtime check, if the
            #  tensor size is too large, it will fallback to the next
            #  available option.
            # In summary: When using CUDA graph, we use
            #  either custom all-reduce kernel or pynccl. When not using
            #  CUDA graph, we use either custom all-reduce kernel or
            #  PyTorch NCCL. We always prioritize using custom all-reduce
            #  kernel but fall back to PyTorch or pynccl if it is
            #  disabled or not supported.
            pynccl_comm = self.pynccl_comm
            maybe_pynccl_context: Any
            if not pynccl_comm:
                maybe_pynccl_context = nullcontext()
            else:
                maybe_pynccl_context = pynccl_comm.change_state(
                    enable=True, stream=torch.cuda.current_stream()
                )
            with maybe_pynccl_context:
                yield graph_capture_context

    def all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
        """
        User-facing all-reduce function before we actually call the
        all-reduce operation.

        We need this because Dynamo does not support passing an arbitrary
        object (`self` in this case) to a custom op. We need to pass the
         group name as a string, and then look up the group coordinator from
         the group name, dispatch the all-reduce operation to the group
         coordinator.

        In addition, PyTorch custom ops do not support mutation or returning
        a new tensor in the same op. So we need to figure out if the op is
        in-place or out-of-place ahead of time.
        """
        # Bypass the function if we are using only 1 GPU.
        if self.world_size == 1:
            return input_

        if input_.is_cpu:
            import intel_extension_for_pytorch as ipex

            ipex.distributed.all_reduce(input_, group=self.device_group)
            return input_

        if not supports_custom_op():
            self._all_reduce_in_place(input_)
            return input_

        if self.hpu_communicator is not None and not self.hpu_communicator.disabled:
            return self.hpu_communicator.all_reduce(input_)

        if self.xpu_communicator is not None and not self.xpu_communicator.disabled:
            return self.xpu_communicator.all_reduce(input_)

        if self.npu_communicator is not None and not self.npu_communicator.disabled:
            return self.npu_communicator.all_reduce(input_)

        if (
            self.ca_comm is not None
            and not self.ca_comm.disabled
            and self.ca_comm.should_custom_ar(input_)
        ):
            result = torch.ops.sglang.outplace_all_reduce(
                input_, group_name=self.unique_name
            )
            return result
        else:
            torch.ops.sglang.inplace_all_reduce(input_, group_name=self.unique_name)
            return input_

    def _all_reduce_out_place(self, input_: torch.Tensor) -> torch.Tensor:
        ca_comm = self.ca_comm
        assert ca_comm is not None
        assert not ca_comm.disabled
        out = ca_comm.custom_all_reduce(input_)
        assert out is not None
        return out

    def _all_reduce_in_place(self, input_: torch.Tensor) -> None:
        pynccl_comm = self.pynccl_comm
        if pynccl_comm is not None and not pynccl_comm.disabled:
            pynccl_comm.all_reduce(input_)
        else:
            torch.distributed.all_reduce(input_, group=self.device_group)

    def _all_gather_into_tensor(self, output: torch.Tensor, input: torch.Tensor):
        pynccl_comm = self.pynccl_comm
        if pynccl_comm is not None and not pynccl_comm.disabled:
            pynccl_comm.all_gather(output, input)
        else:
            torch.distributed.all_gather_into_tensor(
                output, input, group=self.device_group
            )

    def all_gather_into_tensor(self, output: torch.Tensor, input: torch.Tensor):
        if self.npu_communicator is not None and not self.npu_communicator.disabled:
            return self.npu_communicator.all_gather_into_tensor(output, input)

        if not supports_custom_op():
            self._all_gather_into_tensor(output, input)
        else:
            torch.ops.sglang.reg_all_gather_into_tensor(
                output, input, group_name=self.unique_name
            )

    def all_gather(self, input_: torch.Tensor, dim: int = -1, output_split_sizes=None) -> torch.Tensor:
        world_size = self.world_size
        # Bypass the function if we are using only 1 GPU.
        if world_size == 1:
            return input_
        assert (
            -input_.dim() <= dim < input_.dim()
        ), f"Invalid dim ({dim}) for input tensor with shape {input_.size()}"

        # For HPUs, use HPU communicator.
        hpu_comm = self.hpu_communicator
        if hpu_comm is not None and not hpu_comm.disabled:
            return hpu_comm.all_gather(input_, dim)

        # For NPUs, use NPU communicator.
        npu_comm = self.npu_communicator
        if npu_comm is not None and not npu_comm.disabled:
            return npu_comm.all_gather(input_, dim, output_split_sizes)

        if dim < 0:
            # Convert negative dim to positive.
            dim += input_.dim()
        input_size = input_.size()
        # NOTE: we have to use concat-style all-gather here,
        # stack-style all-gather has compatibility issues with
        # torch.compile . see https://github.com/pytorch/pytorch/issues/138795
        output_size = (input_size[0] * world_size,) + input_size[1:]
        # Allocate output tensor.
        output_tensor = torch.empty(
            output_size, dtype=input_.dtype, device=input_.device
        )
        # All-gather.
        self.all_gather_into_tensor(output_tensor, input_)
        # Reshape
        output_tensor = output_tensor.reshape((world_size,) + input_size)
        output_tensor = output_tensor.movedim(0, dim)
        output_tensor = output_tensor.reshape(
            input_size[:dim] + (world_size * input_size[dim],) + input_size[dim + 1 :]
        )
        return output_tensor

    def reduce_scatter(self, input_: torch.Tensor, input_split_sizes=None) -> torch.Tensor:
        if self.world_size == 1:
            return input_

        input_size = tuple(input_.size())
        if input_split_sizes is None:
            output_tensor = torch.empty((input_size[0] // self.world_size,) + input_size[1:],
                                        dtype=input_.dtype,
                                        device=input_.device)
            torch.distributed.reduce_scatter_tensor(output_tensor,
                                                    input_,
                                                    group=self.device_group)
        else:
            assert self.npu_communicator is not None
            return self.npu_communicator.reduce_scatter(input_, input_split_sizes)

        return output_tensor

    def reduce_scatter_tensor(self, output : torch.Tensor, input: torch.Tensor) -> None:
        if self.world_size == 1:
            output.copy_(input)
        else:
            torch.distributed.reduce_scatter_tensor(output, input, group=self.device_group)
        return None

    def gather(
        self, input_: torch.Tensor, dst: int = 0, dim: int = -1
    ) -> Optional[torch.Tensor]:
        """
        NOTE: We assume that the input tensor is on the same device across
        all the ranks.
        NOTE: `dst` is the local rank of the destination rank.
        """
        world_size = self.world_size
        # Bypass the function if we are using only 1 GPU.
        if world_size == 1:
            return input_
        assert (
            -input_.dim() <= dim < input_.dim()
        ), f"Invalid dim ({dim}) for input tensor with shape {input_.size()}"
        if dim < 0:
            # Convert negative dim to positive.
            dim += input_.dim()
        if self.xpu_communicator is not None and not self.xpu_communicator.disabled:
            return self.xpu_communicator.gather(input_, self.rank_in_group, dst, dim)
        # Allocate output tensor.
        if self.rank_in_group == dst:
            gather_list = [torch.empty_like(input_) for _ in range(world_size)]
        else:
            gather_list = None
        # Gather.
        torch.distributed.gather(
            input_, gather_list, dst=self.ranks[dst], group=self.device_group
        )
        if self.rank_in_group == dst:
            output_tensor = torch.cat(gather_list, dim=dim)
        else:
            output_tensor = None
        return output_tensor

    def broadcast(self, input_: torch.Tensor, src: int = 0):
        """Broadcast the input tensor.
        NOTE: `src` is the local rank of the source rank.
        """
        assert src < self.world_size, f"Invalid src rank ({src})"

        # Bypass the function if we are using only 1 GPU.
        if self.world_size == 1:
            return input_
        # Broadcast.
        torch.distributed.broadcast(
            input_, src=self.ranks[src], group=self.device_group
        )
        return input_

    def broadcast_object(self, obj: Optional[Any] = None, src: int = 0):
        """Broadcast the input object.
        NOTE: `src` is the local rank of the source rank.
        """
        assert src < self.world_size, f"Invalid src rank ({src})"

        # Bypass the function if we are using only 1 GPU.
        if self.world_size == 1:
            return obj
        if self.mq_broadcaster is not None:
            assert src == 0, "Message queue broadcaster only supports src=0"
            return self.mq_broadcaster.broadcast_object(obj)
        if self.rank_in_group == src:
            torch.distributed.broadcast_object_list(
                [obj], src=self.ranks[src], group=self.cpu_group
            )
            return obj
        else:
            recv = [None]
            torch.distributed.broadcast_object_list(
                recv, src=self.ranks[src], group=self.cpu_group
            )
            return recv[0]

    def broadcast_object_list(
        self, obj_list: List[Any], src: int = 0, group: Optional[ProcessGroup] = None
    ):
        """Broadcast the input object list.
        NOTE: `src` is the local rank of the source rank.
        """
        assert src < self.world_size, f"Invalid src rank ({src})"

        # Bypass the function if we are using only 1 GPU.
        if self.world_size == 1:
            return obj_list
        # Broadcast.
        torch.distributed.broadcast_object_list(
            obj_list, src=self.ranks[src], group=self.device_group
        )
        return obj_list


    def send_object(
        self,
        obj: Any,
        dst: int,
        async_send: bool = False,
    ) -> List[P2PWork]:
        """
        Send the input object list to the destination rank.
        This function uses the CPU group for all communications.

        TODO: If you want to use GPU communication, please add a new argument (e.g., data_group, group),
        use other functions (e.g., send), or implement a new function (e.g., send_object_device).

        NOTE: `dst` is the local rank of the destination rank.
        """

        assert dst < self.world_size, f"Invalid dst rank ({dst})"
        assert dst != self.rank_in_group, (
            "Invalid destination rank. Destination rank is the same "
            "as the current rank."
        )
        send_func = torch.distributed.isend if async_send else torch.distributed.send

        # Serialize object to tensor and get the size as well
        object_tensor = torch.frombuffer(pickle.dumps(obj), dtype=torch.uint8)
        size_tensor = torch.tensor(
            [object_tensor.numel()], dtype=torch.long, device="cpu"
        )

        # Send object size
        p2p_work = []
        size_work = send_func(
            size_tensor,
            self.ranks[dst],
            group=self.cpu_group,
        )
        if async_send:
            p2p_work.append(P2PWork(size_work, size_tensor))

        object_work = send_func(
            object_tensor,
            self.ranks[dst],
            group=self.cpu_group,
        )
        if async_send:
            p2p_work.append(P2PWork(object_work, object_tensor))

        return p2p_work

    def recv_object(self, src: int) -> Any:
        """Receive the input object list from the source rank."""
        """NOTE: `src` is the local rank of the source rank."""

        assert src < self.world_size, f"Invalid src rank ({src})"

        assert (
            src != self.rank_in_group
        ), "Invalid source rank. Source rank is the same as the current rank."

        size_tensor = torch.empty(1, dtype=torch.long, device="cpu")

        # Receive object size
        rank_size = torch.distributed.recv(
            size_tensor, src=self.ranks[src], group=self.cpu_group
        )

        # Tensor to receive serialized objects into.
        object_tensor = torch.empty(  # type: ignore[call-overload]
            size_tensor.item(),  # type: ignore[arg-type]
            dtype=torch.uint8,
            device="cpu",
        )

        rank_object = torch.distributed.recv(
            object_tensor, src=self.ranks[src], group=self.cpu_group
        )

        assert (
            rank_object == rank_size
        ), "Received object sender rank does not match the size sender rank."

        obj = pickle.loads(object_tensor.numpy().tobytes())

        return obj

    def broadcast_tensor_dict(
        self,
        tensor_dict: Optional[Dict[str, Union[torch.Tensor, Any]]] = None,
        src: int = 0,
        group: Optional[ProcessGroup] = None,
        metadata_group: Optional[ProcessGroup] = None,
    ) -> Optional[Dict[str, Union[torch.Tensor, Any]]]:
        """Broadcast the input tensor dictionary.
        NOTE: `src` is the local rank of the source rank.
        """
        # Bypass the function if we are using only 1 GPU.
        if not torch.distributed.is_initialized() or self.world_size == 1:
            return tensor_dict

        group = self.device_group
        metadata_group = self.cpu_group
        assert src < self.world_size, f"Invalid src rank ({src})"

        rank_in_group = self.rank_in_group
        if rank_in_group == src:
            metadata_list: List[Tuple[Any, Any]] = []
            assert isinstance(
                tensor_dict, dict
            ), f"Expecting a dictionary, got {type(tensor_dict)}"
            metadata_list, tensor_list = _split_tensor_dict(tensor_dict)
            # `metadata_list` lives in CPU memory.
            # `broadcast_object_list` has serialization & deserialization,
            # all happening on CPU. Therefore, we can use the CPU group.
            self.broadcast_object(metadata_list, src=src)
            async_handles = []
            for tensor in tensor_list:
                if tensor.numel() == 0:
                    # Skip broadcasting empty tensors.
                    continue
                if tensor.is_cpu:
                    # use metadata_group for CPU tensors
                    handle = torch.distributed.broadcast(
                        tensor, src=self.ranks[src], group=metadata_group, async_op=True
                    )
                else:
                    # use group for GPU tensors
                    handle = torch.distributed.broadcast(
                        tensor, src=self.ranks[src], group=group, async_op=True
                    )
                async_handles.append(handle)
            for async_handle in async_handles:
                async_handle.wait()

        else:
            metadata_list = self.broadcast_object(None, src=src)
            tensor_dict = {}
            async_handles = []
            for key, value in metadata_list:
                if isinstance(value, TensorMetadata):
                    tensor = torch.empty(
                        value.size, dtype=value.dtype, device=value.device
                    )
                    if tensor.numel() == 0:
                        # Skip broadcasting empty tensors.
                        tensor_dict[key] = tensor
                        continue
                    if tensor.is_cpu:
                        # use metadata_group for CPU tensors
                        handle = torch.distributed.broadcast(
                            tensor,
                            src=self.ranks[src],
                            group=metadata_group,
                            async_op=True,
                        )
                    else:
                        # use group for GPU tensors
                        handle = torch.distributed.broadcast(
                            tensor, src=self.ranks[src], group=group, async_op=True
                        )
                    async_handles.append(handle)
                    tensor_dict[key] = tensor
                else:
                    tensor_dict[key] = value
            for async_handle in async_handles:
                async_handle.wait()
        return tensor_dict

    def send_tensor_dict(
        self,
        tensor_dict: Dict[str, Union[torch.Tensor, Any]],
        dst: Optional[int] = None,
        all_gather_group: Optional["GroupCoordinator"] = None,
        async_send: bool = False,
    ) -> Optional[List[P2PWork]]:
        """Send the input tensor dictionary.
        NOTE: `dst` is the local rank of the source rank.
        """
        # Bypass the function if we are using only 1 GPU.
        if self.world_size == 1:
            return tensor_dict

        all_gather_size = 1 if all_gather_group is None else all_gather_group.world_size
        all_gather_rank = (
            0 if all_gather_group is None else all_gather_group.rank_in_group
        )

        group = self.device_group
        metadata_group = self.cpu_group

        if dst is None:
            dst = (self.rank_in_group + 1) % self.world_size
        assert dst < self.world_size, f"Invalid dst rank ({dst})"

        assert isinstance(
            tensor_dict, dict
        ), f"Expecting a dictionary, got {type(tensor_dict)}"
        metadata_list, tensor_list = _split_tensor_dict(tensor_dict)
        # Note: While switching to Device-to-Device (D2D) would introduce an extra
        # Device-to-Host (D2H) memory copy overhead for serialization, our benchmarks
        # show better overall transmission performance with D2D due to:
        # 1. Superior D2D transfer bandwidth
        # 2. Ability to overlap send and recv operations
        # Thus the net performance gain justifies this approach.

        send_func = torch.distributed.isend if async_send else torch.distributed.send
        p2p_works = self.send_object(metadata_list, dst=dst, async_send=async_send)

        for tensor in tensor_list:
            if tensor.numel() == 0:
                # Skip sending empty tensors.
                continue

            # send-allgather: send only a slice, then do allgather.
            if all_gather_group is not None and tensor.numel() % all_gather_size == 0:
                tensor = tensor.reshape(all_gather_size, -1)[all_gather_rank]

            comm_group = metadata_group if tensor.is_cpu else group
            work = send_func(tensor, self.ranks[dst], group=comm_group)
            if async_send:
                p2p_works.append(P2PWork(work, tensor))
        return p2p_works

    def recv_tensor_dict(
        self,
        src: Optional[int] = None,
        all_gather_group: Optional["GroupCoordinator"] = None,
    ) -> Optional[Dict[str, Union[torch.Tensor, Any]]]:
        """Recv the input tensor dictionary.
        NOTE: `src` is the local rank of the source rank.
        """
        # Bypass the function if we are using only 1 GPU.
        if not torch.distributed.is_initialized() or self.world_size == 1:
            return None

        all_gather_size = 1 if all_gather_group is None else all_gather_group.world_size
        all_gather_rank = (
            0 if all_gather_group is None else all_gather_group.rank_in_group
        )

        group = self.device_group
        metadata_group = self.cpu_group

        if src is None:
            src = (self.rank_in_group - 1) % self.world_size
        assert src < self.world_size, f"Invalid src rank ({src})"

        recv_metadata_list = self.recv_object(src=src)
        tensor_dict: Dict[str, Any] = {}
        for key, value in recv_metadata_list:
            if isinstance(value, TensorMetadata):
                tensor = torch.empty(value.size, dtype=value.dtype, device=value.device)
                if tensor.numel() == 0:
                    # Skip broadcasting empty tensors.
                    tensor_dict[key] = tensor
                    continue

                # send-allgather: send only a slice, then do allgather.
                use_all_gather = (
                    all_gather_group is not None
                    and tensor.numel() % all_gather_size == 0
                )

                if use_all_gather:
                    orig_shape = tensor.shape
                    tensor = tensor.reshape(all_gather_size, -1)[all_gather_rank]

                if tensor.is_cpu:
                    # use metadata_group for CPU tensors
                    torch.distributed.recv(
                        tensor, src=self.ranks[src], group=metadata_group
                    )
                else:
                    # use group for GPU tensors
                    torch.distributed.recv(tensor, src=self.ranks[src], group=group)
                if use_all_gather:
                    # do the allgather
                    tensor = all_gather_group.all_gather(tensor, dim=0)  # type: ignore
                    tensor = tensor.reshape(orig_shape)

                tensor_dict[key] = tensor
            else:
                tensor_dict[key] = value
        return tensor_dict

    def barrier(self):
        """Barrier synchronization among the group.
        NOTE: don't use `device_group` here! `barrier` in NCCL is
        terrible because it is internally a broadcast operation with
        secretly created GPU tensors. It is easy to mess up the current
        device. Use the CPU group instead.
        """
        torch.distributed.barrier(group=self.cpu_group)

    def send(self, tensor: torch.Tensor, dst: Optional[int] = None) -> None:
        """Sends a tensor to the destination rank in a non-blocking way"""
        """NOTE: `dst` is the local rank of the destination rank."""
        if dst is None:
            dst = (self.rank_in_group + 1) % self.world_size

        pynccl_comm = self.pynccl_comm
        if pynccl_comm is not None and not pynccl_comm.disabled:
            pynccl_comm.send(tensor, dst)
        else:
            torch.distributed.send(tensor, self.ranks[dst], self.device_group)

    def recv(
        self, size: torch.Size, dtype: torch.dtype, src: Optional[int] = None
    ) -> torch.Tensor:
        """Receives a tensor from the source rank."""
        """NOTE: `src` is the local rank of the source rank."""
        if src is None:
            src = (self.rank_in_group - 1) % self.world_size

        tensor = torch.empty(size, dtype=dtype, device=self.device)
        pynccl_comm = self.pynccl_comm
        if pynccl_comm is not None and not pynccl_comm.disabled:
            pynccl_comm.recv(tensor, src)
        else:
            torch.distributed.recv(tensor, self.ranks[src], self.device_group)
        return tensor

    def destroy(self):
        if self.device_group is not None:
            torch.distributed.destroy_process_group(self.device_group)
            self.device_group = None
        if self.cpu_group is not None:
            torch.distributed.destroy_process_group(self.cpu_group)
            self.cpu_group = None
        if self.pynccl_comm is not None:
            self.pynccl_comm = None
        if self.ca_comm is not None:
            self.ca_comm = None
        if self.mq_broadcaster is not None:
            self.mq_broadcaster = None

    def get_local_sp_token_num(self, global_sp_token_num: List[int]) -> List[int]:
        if global_sp_token_num is None:
            return None
        tp_rank=get_tp_group().rank_in_group
        group_size=self.world_size
        group_rank=tp_rank//self.world_size
        return global_sp_token_num[group_rank*group_size: (group_rank+1)*group_size]

    def split_tensor(self, input: torch.Tensor, global_sp_token_num: List[int]) -> torch.Tensor:
        local_sp_token_num = self.get_local_sp_token_num(global_sp_token_num)
        if local_sp_token_num is None:
            local_sp_token_num = [input.shape[0]//self.world_size] * self.world_size
        sp_prefix_sum = get_prefix_sum(local_sp_token_num)
        return input[sp_prefix_sum[self.rank_in_group]: sp_prefix_sum[self.rank_in_group+1]]

    def gather_tensor(self, input: torch.Tensor, global_sp_token_num: List[int]) -> torch.Tensor:
        local_sp_token_num = self.get_local_sp_token_num(global_sp_token_num)
        return self.all_gather(input, dim=0, output_split_sizes=local_sp_token_num)

    def scatter_tensor(self, input: torch.Tensor, global_sp_token_num: List[int]) -> torch.Tensor:
        local_sp_token_num = self.get_local_sp_token_num(global_sp_token_num)
        return self.reduce_scatter(input, input_split_sizes=local_sp_token_num)

_WORLD: Optional[GroupCoordinator] = None


def get_world_group() -> GroupCoordinator:
    assert _WORLD is not None, "world group is not initialized"
    return _WORLD


def init_world_group(
    ranks: List[int], local_rank: int, backend: str
) -> GroupCoordinator:
    return GroupCoordinator(
        group_ranks=[ranks],
        local_rank=local_rank,
        torch_distributed_backend=backend,
        use_pynccl=False,
        use_custom_allreduce=False,
        use_hpu_communicator=False,
        use_xpu_communicator=False,
        use_npu_communicator=False,
        group_name="world",
    )


_EPS_COMMUNICATOR = None


def init_eps_communicator(num_nodes: int):
    from eps.communication import MscclppCommunicator, MscclppCommunicatorParams

    rank = _WORLD.rank
    world_size = _WORLD.world_size
    num_ranks_per_node = world_size // num_nodes

    if rank == 0:
        unique_id = [MscclppCommunicator.createUniqueId()]
    else:
        unique_id = [None]

    torch.distributed.broadcast_object_list(unique_id, src=0, group=_WORLD.cpu_group)
    p = MscclppCommunicatorParams(rank, world_size, num_ranks_per_node)
    return MscclppCommunicator(unique_id[0], p)


def get_eps_communicator():
    return _EPS_COMMUNICATOR


def init_model_parallel_group(
    group_ranks: List[List[int]],
    local_rank: int,
    backend: str,
    use_custom_allreduce: Optional[bool] = None,
    use_message_queue_broadcaster: bool = False,
    group_name: Optional[str] = None,
    pg_options: Any = None,
) -> GroupCoordinator:
    if use_custom_allreduce is None:
        use_custom_allreduce = _ENABLE_CUSTOM_ALL_REDUCE
    return GroupCoordinator(
        group_ranks=group_ranks,
        local_rank=local_rank,
        torch_distributed_backend=backend,
        use_pynccl=not is_npu(),
        use_custom_allreduce=use_custom_allreduce,
        use_hpu_communicator=True,
        use_xpu_communicator=True,
        use_npu_communicator=is_npu(),
        use_message_queue_broadcaster=use_message_queue_broadcaster,
        group_name=group_name,
        pg_options=pg_options
    )


_TP: Optional[GroupCoordinator] = None


_ALL2ALL_EP: Optional[GroupCoordinator] = None

_ATTN_EP: Optional[GroupCoordinator] = None

def get_tp_group() -> GroupCoordinator:
    assert _TP is not None, "tensor model parallel group is not initialized"
    return _TP


def get_all2all_ep_group():
    assert not ENV.npu_enable_all2all_comm or _ALL2ALL_EP is not None, "All2All expert parallel group is not initialized"
    return _ALL2ALL_EP

def get_attn_ep_group():
    assert _ATTN_EP is not None, "Attn expert parallel group is not initialized"
    return _ATTN_EP

# kept for backward compatibility
get_tensor_model_parallel_group = get_tp_group


_MOE_EP: Optional[GroupCoordinator] = None
_MOE_TP: Optional[GroupCoordinator] = None


def get_moe_ep_group() -> GroupCoordinator:
    assert _MOE_EP is not None, "expert model parallel group is not initialized"
    return _MOE_EP


def get_moe_tp_group() -> GroupCoordinator:
    assert _MOE_TP is not None, "expert model parallel group is not initialized"
    return _MOE_TP


def get_moe_expert_parallel_world_size():
    """Return world size for the moe expert parallel group."""
    return get_moe_ep_group().world_size


def get_moe_expert_parallel_rank():
    """Return my rank for the moe expert parallel group."""
    return get_moe_ep_group().rank_in_group


def get_moe_tensor_parallel_world_size():
    """Return world size for the moe tensor parallel group."""
    return get_moe_tp_group().world_size


def get_moe_tensor_parallel_rank():
    """Return my rank for the moe tensor parallel group."""
    return get_moe_tp_group().rank_in_group


_PP: Optional[GroupCoordinator] = None
_PP_REVERSE: Optional[GroupCoordinator] = None

def get_pp_group() -> GroupCoordinator:
    assert _PP is not None, "pipeline model parallel group is not initialized"
    return _PP

def get_pp_reverse_group() -> GroupCoordinator:
    assert _PP_REVERSE is not None, "pipeline reverse model parallel group is not initialized"
    return _PP_REVERSE


# kept for backward compatibility
get_pipeline_model_parallel_group = get_pp_group


_MLP_TP: Optional[GroupCoordinator] = None

_MLP_TP_CROSS: Optional[GroupCoordinator] = None

ALL_COMMS: Dict[int, GroupCoordinator] = {}
ALL_COMMS_CROSS: Dict[int, GroupCoordinator] = {}

def get_mlp_tp_group() -> GroupCoordinator:
    assert _MLP_TP is not None, ("mlp tensor model parallel group is not initialized")
    return _MLP_TP


def get_mlp_tp_group_cross() -> GroupCoordinator:
    assert _MLP_TP_CROSS is not None, ("mlp tensor model parallel group is not initialized")
    return _MLP_TP_CROSS

_ATTN_TP: Optional[GroupCoordinator] = None


def get_attn_tp_group() -> GroupCoordinator:
    assert _ATTN_TP is not None, ("attn tensor model parallel group is not initialized")
    return _ATTN_TP


_ATTN_TP_CROSS: Optional[GroupCoordinator] = None


def get_attn_cross_group() -> GroupCoordinator:
    assert _ATTN_TP_CROSS is not None, ("attn tp parallel dp cross group is not initialized")
    return _ATTN_TP_CROSS

_ATTN_O_PROJ_TP: Optional[GroupCoordinator] = None

def get_attn_o_proj_tp_group() -> GroupCoordinator:
    assert _ATTN_O_PROJ_TP is not None, ("attn o_proj tensor parallel group is not initialized")
    return _ATTN_O_PROJ_TP

_ATTN_LM_HEAD_TP: Optional[GroupCoordinator] = None

def get_lm_head_tp_group() -> GroupCoordinator:
    assert _ATTN_LM_HEAD_TP is not None, ("lm head tensor parallel group is not initialized")
    return _ATTN_LM_HEAD_TP
# kept for backward compatibility
get_mlp_tensor_model_parallel_group = get_mlp_tp_group


SCHEDULER_COMM_GROUP: Optional[GroupCoordinator] = None
SCHEDULER_ATTN_TP_GROUP: Optional[GroupCoordinator] = None

def get_scheduler_comm_group() -> GroupCoordinator:
    assert SCHEDULER_COMM_GROUP is not None, ("Scheduler comm group is not initialized")
    return SCHEDULER_COMM_GROUP

def get_scheduler_attn_tp_group() -> GroupCoordinator:
    assert SCHEDULER_ATTN_TP_GROUP is not None, ("Scheduler attn tp group is not initialized")
    return SCHEDULER_ATTN_TP_GROUP

@contextmanager
def graph_capture():
    """
    `graph_capture` is a context manager which should surround the code that
    is capturing the CUDA graph. Its main purpose is to ensure that
    some operations will run after the graph is captured, before the graph
    is replayed. It returns a `GraphCaptureContext` object which contains the
    necessary data for the graph capture. Currently, it only contains the
    stream that the graph capture is running on. This stream is set to the
    current CUDA stream when the context manager is entered and reset to the
    default stream when the context manager is exited. This is to ensure that
    the graph capture is running on a separate stream from the default stream,
    in order to explicitly distinguish the kernels to capture
    from other kernels possibly launched on background in the default stream.
    """
    with get_tp_group().graph_capture() as context, get_pp_group().graph_capture(
        context
    ):
        yield context


logger = get_colorful_logger(__name__)

_ENABLE_CUSTOM_ALL_REDUCE = True


def set_custom_all_reduce(enable: bool):
    global _ENABLE_CUSTOM_ALL_REDUCE
    _ENABLE_CUSTOM_ALL_REDUCE = enable

def full_mesh_warmup():
    """
    执行一次 Full Mesh (All-to-All) 通信，用于建立所有节点间的连接。
    """
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()

    device = global_server_args_dict['device']
    send_tensor = torch.tensor([float(rank)], device=device)

    recv_buffers = [torch.zeros(1, device=device) for _ in range(world_size)]

    ops = []
    for peer_rank in range(world_size):
        if peer_rank == rank:
            continue

        ops.append(torch.distributed.P2POp(torch.distributed.isend, send_tensor, peer_rank))
        ops.append(torch.distributed.P2POp(torch.distributed.irecv, recv_buffers[peer_rank], peer_rank))

    if ops:
        reqs = torch.distributed.batch_isend_irecv(ops)
        for req in reqs:
            req.wait()

def init_distributed_environment(
    world_size: int = -1,
    rank: int = -1,
    distributed_init_method: str = "env://",
    local_rank: int = -1,
    backend: str = "nccl",
    timeout: Optional[int] = None,
    num_nodes: int = 1
):
    logger.info(
        "world_size=%d rank=%d local_rank=%d " "distributed_init_method=%s backend=%s",
        world_size,
        rank,
        local_rank,
        distributed_init_method,
        backend,
    )
    if not torch.distributed.is_initialized():
        assert distributed_init_method is not None, (
            "distributed_init_method must be provided when initializing "
            "distributed environment"
        )
        if timeout is not None:
            assert isinstance(timeout, (int)), "timeout must be a number"
            assert timeout > 0, "timeout must be positive"
            timeout = timedelta(seconds=timeout)

        # this backend is used for WORLD
        # TODO chl for a3
        # if global_server_args_dict["disaggregation_mode"] == "prefill":
        #     os.environ["LOCAL_RANK"] = str(local_rank)
        #     os.environ["GROUP_RANK"] = "0"
        #     os.environ["RANK"] = str(rank)
        #     os.environ["LOCAL_WORLD_SIZE"] = "16"
        #     print(f"LOCAL_WORLD_SIZE    {local_rank}   {rank} {world_size}")
        torch.distributed.init_process_group(
            backend=backend,
            init_method=distributed_init_method,
            world_size=world_size,
            rank=rank,
            timeout=timeout,
        )
        # `batch_isend_irecv` cannot be the first collective call
        # Refer to PyTorch's doc of `batch_isend_irecv`
        torch.distributed.barrier()

        # Note(lyk, 12/29/2025)
        # Force full mesh communication to prevent random hangs due to
        # socket establishment timeouts in specific topologies during batch_isend_irecv.
        # Temporary workaround. Remove it when the issue is resolved.
        full_mesh_warmup()


    # set the local rank
    # local_rank is not available in torch ProcessGroup,
    # see https://github.com/pytorch/pytorch/issues/122816
    if local_rank == -1:
        # local rank not set, this usually happens in single-node
        # setting, where we can use rank as local rank
        if distributed_init_method == "env://":
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        else:
            local_rank = rank

    global _WORLD
    global _EPS_COMMUNICATOR
    if _WORLD is None:
        ranks = list(range(torch.distributed.get_world_size()))
        _WORLD = init_world_group(ranks, local_rank, backend)
        if not __is_npu__:
            _EPS_COMMUNICATOR = init_eps_communicator(num_nodes)
    else:
        assert (
            _WORLD.world_size == torch.distributed.get_world_size()
        ), "world group already initialized with a different world size"

def initialize_communicator(tp_size, backend, world_size=None, with_cross=False):
    global ALL_COMMS, ALL_COMMS_CROSS
    if world_size is None:
        assert torch.distributed.is_initialized()
        world_size = torch.distributed.get_world_size()

    if tp_size not in ALL_COMMS:
        assert world_size%tp_size==0
        num_tp_groups=world_size//tp_size
        tp_group_ranks=[]
        for i in range(num_tp_groups):
            ranks=list(range(i*tp_size, (i+1)*tp_size))
            tp_group_ranks.append(ranks)
        comm=init_model_parallel_group(tp_group_ranks,
                                                  get_world_group().local_rank,
                                                  backend,
                                                  use_message_queue_broadcaster=False,
                                                  group_name="tp"+str(tp_size))
        ALL_COMMS[tp_size]=comm

    if with_cross and tp_size not in ALL_COMMS_CROSS:
        assert world_size%tp_size==0
        tp_group_ranks=[]
        for i in range(tp_size):
            ranks=list(range(i, world_size, tp_size))
            tp_group_ranks.append(ranks)
        comm=init_model_parallel_group(tp_group_ranks,
                                                  get_world_group().local_rank,
                                                  backend,
                                                  use_message_queue_broadcaster=False,
                                                  group_name="tp_cross"+str(tp_size))
        ALL_COMMS_CROSS[tp_size]=comm

def initialize_attn_tp(attn_tp_size, backend, world_size=None):
    initialize_communicator(attn_tp_size, backend, world_size=world_size, with_cross=True)
    global ALL_COMMS, ALL_COMMS_CROSS
    global _ATTN_TP
    assert _ATTN_TP is None, ("attn tensor model parallel group is already initialized")
    global _ATTN_TP_CROSS
    assert _ATTN_TP_CROSS is None, ("attn tensor model parallel cross group is already initialized")
    _ATTN_TP = ALL_COMMS[attn_tp_size]
    _ATTN_TP_CROSS = ALL_COMMS_CROSS[attn_tp_size]
    o_proj_tp=global_server_args_dict["npu_o_proj_tp_size"] or attn_tp_size
    if o_proj_tp > 0:
        global _ATTN_O_PROJ_TP, _ATTN_O_PROJ_TP_CROSS
        initialize_communicator(o_proj_tp, backend, world_size=world_size, with_cross=False)
        _ATTN_O_PROJ_TP=ALL_COMMS[o_proj_tp]

def initialize_model_parallel(
    tensor_model_parallel_size: int = 1,
    expert_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    mlp_tensor_model_parallel_size: int = 1,
    attn_tp_size: int = 1,
    backend: Optional[str] = None,
) -> None:
    """
    Initialize model parallel groups.

    Arguments:
        tensor_model_parallel_size: number of GPUs used for tensor model
            parallelism.
        pipeline_model_parallel_size: number of GPUs used for pipeline model
            parallelism.

    Let's say we have a total of 8 GPUs denoted by g0 ... g7 and we
    use 2 GPUs to parallelize the model tensor, and 4 GPUs to parallelize
    the model pipeline. The present function will
    create 4 tensor model-parallel groups and 2 pipeline model-parallel groups:
        4 tensor model-parallel groups:
            [g0, g1], [g2, g3], [g4, g5], [g6, g7]
        2 pipeline model-parallel groups:
            [g0, g2, g4, g6], [g1, g3, g5, g7]
    Note that for efficiency, the caller should make sure adjacent ranks
    are on the same DGX box. For example if we are using 2 DGX-1 boxes
    with a total of 16 GPUs, rank 0 to 7 belong to the first box and
    ranks 8 to 15 belong to the second box.
    """
    # Get world size and rank. Ensure some consistencies.
    assert torch.distributed.is_initialized()
    world_size: int = torch.distributed.get_world_size()
    backend = backend or torch.distributed.get_backend(get_world_group().device_group)

    if world_size != tensor_model_parallel_size * pipeline_model_parallel_size:
        raise RuntimeError(
            f"world_size ({world_size}) is not equal to "
            f"tensor_model_parallel_size ({tensor_model_parallel_size}) x "
            f"pipeline_model_parallel_size ({pipeline_model_parallel_size})"
        )

    # Build the tensor model-parallel groups.
    num_tensor_model_parallel_groups: int = world_size // tensor_model_parallel_size
    global _TP
    assert _TP is None, "tensor model parallel group is already initialized"
    group_ranks = []
    for i in range(num_tensor_model_parallel_groups):
        ranks = list(
            range(i * tensor_model_parallel_size, (i + 1) * tensor_model_parallel_size)
        )
        group_ranks.append(ranks)

    # message queue broadcaster is only used in tensor model parallel group
    _TP = init_model_parallel_group(
        group_ranks,
        get_world_group().local_rank,
        backend,
        use_message_queue_broadcaster=True,
        group_name="tp",
    )
    if ENV.npu_enable_all2all_comm and __is_npu__:
        global _ALL2ALL_EP
        assert _ALL2ALL_EP is None, "All2All expert parallel group is already initialized"
        import torch_npu
        options = torch_npu._C._distributed_c10d.ProcessGroupHCCL.Options()
        options.hccl_config = {"hccl_buffer_size": global_server_args_dict["npu_hccl_buffsize_a2a"]}
        _ALL2ALL_EP = init_model_parallel_group(
            group_ranks,
            get_world_group().local_rank,
            backend,
            use_message_queue_broadcaster=True,
            group_name="all2all_ep",
            pg_options=options
        )

        group_ranks=[]
        global _ATTN_EP
        assert _ATTN_EP is None, "attn expert parallel group is already initialized"
        for i in range(world_size//attn_tp_size):
            ranks=list(
                range(i*attn_tp_size, (i+1)*attn_tp_size)
            )
            group_ranks.append(ranks)
        _ATTN_EP=init_model_parallel_group(
            group_ranks,
            get_world_group().local_rank,
            backend,
            use_message_queue_broadcaster=True,
            group_name="attn_ep")

    moe_ep_size = expert_model_parallel_size
    moe_tp_size = tensor_model_parallel_size // moe_ep_size

    global _MOE_EP
    assert _MOE_EP is None, "expert model parallel group is already initialized"

    if moe_ep_size == tensor_model_parallel_size:
        _MOE_EP = _TP
    else:
        # TODO(ch-wan): use split_group to save memory
        group_ranks = []
        for i in range(num_tensor_model_parallel_groups):
            for j in range(moe_tp_size):
                st = i * tensor_model_parallel_size + j
                en = (i + 1) * tensor_model_parallel_size + j
                ranks = list(range(st, en, moe_tp_size))
                group_ranks.append(ranks)
        _MOE_EP = init_model_parallel_group(
            group_ranks,
            get_world_group().local_rank,
            backend,
            group_name="moe_ep",
        )

    global _MOE_TP
    assert _MOE_TP is None, "expert model parallel group is already initialized"

    if moe_tp_size == tensor_model_parallel_size:
        _MOE_TP = _TP
    else:
        # TODO(ch-wan): use split_group to save memory
        group_ranks = []
        for i in range(num_tensor_model_parallel_groups):
            for j in range(moe_ep_size):
                st = i * tensor_model_parallel_size + j * moe_tp_size
                en = i * tensor_model_parallel_size + (j + 1) * moe_tp_size
                ranks = list(range(st, en))
                group_ranks.append(ranks)
        _MOE_TP = init_model_parallel_group(
            group_ranks,
            get_world_group().local_rank,
            backend,
            group_name="moe_tp",
        )

    # Build the pipeline model-parallel groups.
    num_pipeline_model_parallel_groups: int = world_size // pipeline_model_parallel_size
    global _PP, _PP_REVERSE
    assert _PP is None, "pipeline model parallel group is already initialized"
    group_ranks = []
    for i in range(num_pipeline_model_parallel_groups):
        ranks = list(range(i, world_size, num_pipeline_model_parallel_groups))
        group_ranks.append(ranks)
    # pipeline parallel does not need custom allreduce
    _PP = init_model_parallel_group(
        group_ranks,
        get_world_group().local_rank,
        backend,
        use_custom_allreduce=False,
        group_name="pp",
    )

    _PP_REVERSE = init_model_parallel_group(
            group_ranks,
            get_world_group().local_rank,
            backend,
            use_message_queue_broadcaster=True,
            group_name="pp_reverse",
        )

    global ALL_COMMS, ALL_COMMS_CROSS
    ALL_COMMS[tensor_model_parallel_size] = _TP
    ALL_COMMS_CROSS[tensor_model_parallel_size] = _PP

    if pipeline_model_parallel_size == 1:
        ALL_COMMS_CROSS[pipeline_model_parallel_size] = _TP

    global _MLP_TP
    assert _MLP_TP is None, ("mlp tensor model parallel group is already initialized")
    global _MLP_TP_CROSS
    assert _MLP_TP_CROSS is None, ("mlp tensor model parallel cross group is already initialized")
    initialize_communicator(mlp_tensor_model_parallel_size, backend, world_size=world_size, with_cross=True)
    _MLP_TP = ALL_COMMS[mlp_tensor_model_parallel_size]
    _MLP_TP_CROSS = ALL_COMMS_CROSS[mlp_tensor_model_parallel_size]

    initialize_attn_tp(attn_tp_size, backend, world_size)

    # todo: 后续重构，将已经创建的通信域放在字典里，有相同大小的通信域复用，不能复用的单独创建，另外需要释放通信域
    npu_lmhead_tp_size=global_server_args_dict["npu_lmhead_tp_size"]
    if npu_lmhead_tp_size > 0:
        if global_server_args_dict['disaggregation_mode']=='prefill':
            assert npu_lmhead_tp_size==attn_tp_size, "force lmhead_tp_size==attn_tp_size"
        global _ATTN_LM_HEAD_TP
        assert _ATTN_LM_HEAD_TP is None, ("lm head tensor parallel group is already initialized")
        assert world_size % npu_lmhead_tp_size == 0, f'({world_size=}) % ({npu_lmhead_tp_size=}) != 0'
        initialize_communicator(npu_lmhead_tp_size, backend, world_size=world_size, with_cross=False)
        _ATTN_LM_HEAD_TP = ALL_COMMS[npu_lmhead_tp_size]

    if global_server_args_dict['npu_scheduler_comm'] and __is_npu__:
        global SCHEDULER_COMM_GROUP
        assert SCHEDULER_COMM_GROUP is None, ("Scheduler comm group is already initialized")
        import torch_npu
        options = torch_npu._C._distributed_c10d.ProcessGroupHCCL.Options()
        options.hccl_config ={"hccl_op_expansion_mode": 2}
        SCHEDULER_COMM_GROUP = init_model_parallel_group(
            [list(range(torch.distributed.get_world_size()))],
            get_world_group().local_rank,
            backend,
            use_message_queue_broadcaster=True,
            group_name="scheduler_comm",
            pg_options=options
        )

        global SCHEDULER_ATTN_TP_GROUP
        assert SCHEDULER_ATTN_TP_GROUP is None, ("Scheduler attn tp group is already initialized")
        group_ranks=[]
        for i in range(world_size//attn_tp_size):
            ranks=list(
                range(i*attn_tp_size, (i+1)*attn_tp_size)
            )
            group_ranks.append(ranks)
        SCHEDULER_ATTN_TP_GROUP = init_model_parallel_group(
            group_ranks,
            get_world_group().local_rank,
            backend,
            use_message_queue_broadcaster=True,
            group_name="scheduler_attn_tp",
            pg_options=options
        )


def ensure_model_parallel_initialized(
    tensor_model_parallel_size: int,
    expert_model_parallel_size: int,
    pipeline_model_parallel_size: int,
    backend: Optional[str] = None,
) -> None:
    """Helper to initialize model parallel groups if they are not initialized,
    or ensure tensor-parallel and pipeline-parallel sizes are equal to expected
    values if the model parallel groups are initialized.
    """
    backend = backend or torch.distributed.get_backend(get_world_group().device_group)
    if not model_parallel_is_initialized():
        initialize_model_parallel(
            tensor_model_parallel_size,
            expert_model_parallel_size,
            pipeline_model_parallel_size,
            backend
        )
        return

    assert get_tensor_model_parallel_world_size() == tensor_model_parallel_size, (
        "tensor parallel group already initialized, but of unexpected size: "
        f"{get_tensor_model_parallel_world_size()=} vs. "
        f"{tensor_model_parallel_size=}"
    )

    pp_world_size = get_pp_group().world_size
    assert pp_world_size == pipeline_model_parallel_size, (
        "pipeline parallel group already initialized, but of unexpected size: "
        f"{pp_world_size=} vs. "
        f"{pipeline_model_parallel_size=}"
    )


def model_parallel_is_initialized():
    """Check if tensor and pipeline parallel groups are initialized."""
    return _TP is not None and _PP is not None and (not ENV.npu_enable_all2all_com or _ALL2ALL_EP is not None)


_TP_STATE_PATCHED = False


@contextmanager
def patch_tensor_parallel_group(tp_group: GroupCoordinator):
    """Patch the tp group temporarily until this function ends.

    This method is for draft workers of speculative decoding to run draft model
    with different tp degree from that of target model workers.

    Args:
        tp_group (GroupCoordinator): the tp group coordinator
    """
    global _TP_STATE_PATCHED
    assert not _TP_STATE_PATCHED, "Should not call when it's already patched"

    _TP_STATE_PATCHED = True
    old_tp_group = get_tp_group()
    global _TP
    _TP = tp_group
    try:
        yield
    finally:
        # restore the original state
        _TP_STATE_PATCHED = False
        _TP = old_tp_group

def get_tensor_model_parallel_world_size():
    """Return world size for the tensor model parallel group."""
    return get_tp_group().world_size


def get_tensor_model_parallel_rank():
    """Return my rank for the tensor model parallel group."""
    return get_tp_group().rank_in_group


def destroy_model_parallel():
    """Set the groups to none and destroy them."""
    global _TP
    if _TP:
        _TP.destroy()
    _TP = None

    global _ALL2ALL_EP
    if _ALL2ALL_EP:
        _ALL2ALL_EP.destroy()
    _ALL2ALL_EP = None

    global _PP
    if _PP:
        _PP.destroy()
    _PP = None


def destroy_distributed_environment():
    global _WORLD
    if _WORLD:
        _WORLD.destroy()
    _WORLD = None
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def cleanup_dist_env_and_memory(shutdown_ray: bool = False):
    destroy_model_parallel()
    destroy_distributed_environment()
    with contextlib.suppress(AssertionError):
        torch.distributed.destroy_process_group()
    if shutdown_ray:
        import ray  # Lazy import Ray

        ray.shutdown()
    gc.collect()
    if is_cuda_alike():
        torch.cuda.empty_cache()


def in_the_same_node_as(pg: ProcessGroup, source_rank: int = 0) -> List[bool]:
    """
    This is a collective operation that returns if each rank is in the same node
    as the source rank. It tests if processes are attached to the same
    memory system (shared access to shared memory).
    """
    assert (
        torch.distributed.get_backend(pg) != torch.distributed.Backend.NCCL
    ), "in_the_same_node_as should be tested with a non-NCCL group."
    # local rank inside the group
    rank = torch.distributed.get_rank(group=pg)
    world_size = torch.distributed.get_world_size(group=pg)

    # local tensor in each process to store the result
    is_in_the_same_node = torch.tensor([0] * world_size, dtype=torch.int32)

    # global ranks of the processes in the group
    ranks = torch.distributed.get_process_group_ranks(pg)

    magic_message = b"magic_message"
    shm = None

    try:
        with contextlib.suppress(OSError):
            if rank == source_rank:
                # create a shared memory segment
                shm = shared_memory.SharedMemory(create=True, size=128)
                shm.buf[: len(magic_message)] = magic_message
                torch.distributed.broadcast_object_list(
                    [shm.name], src=ranks[source_rank], group=pg
                )
                is_in_the_same_node[rank] = 1
            else:
                # try to open the shared memory segment
                recv = [None]
                torch.distributed.broadcast_object_list(
                    recv, src=ranks[source_rank], group=pg
                )
                name = recv[0]
                # fix to https://stackoverflow.com/q/62748654/9191338
                # Python incorrectly tracks shared memory even if it is not
                # created by the process. The following patch is a workaround.
                with patch(
                    "multiprocessing.resource_tracker.register",
                    lambda *args, **kwargs: None,
                ):
                    shm = shared_memory.SharedMemory(name=name)
                if shm.buf[: len(magic_message)] == magic_message:
                    is_in_the_same_node[rank] = 1
    except Exception as e:
        logger.error("Error ignored in is_in_the_same_node: %s", e)
    finally:
        if shm:
            shm.close()

    torch.distributed.barrier(group=pg)

    # clean up the shared memory segment
    with contextlib.suppress(OSError):
        if rank == source_rank and shm:
            shm.unlink()
    torch.distributed.all_reduce(is_in_the_same_node, group=pg)

    return [x == 1 for x in is_in_the_same_node.tolist()]

def get_etp_group():
    return get_tp_group()

def get_expert_and_tensor_model_parallel_world_size():
    return get_etp_group().world_size

def get_expert_and_tensor_model_parallel_rank():
    return get_etp_group().rank_in_group

def get_ep_group():
    return get_tp_group()

def get_expert_model_parallel_world_size():
    return get_ep_group().world_size

def get_attn_tp_world_size():
    if _ATTN_TP is None:
        return 1
    return get_attn_tp_group().world_size

def get_expert_model_parallel_rank():
    return get_ep_group().rank_in_group

def expert_and_tensor_model_parallel_all_reduce(input_: torch.Tensor) -> torch.Tensor:
    """All-reduce the input tensor across model parallel group."""
    return get_etp_group().all_reduce(input_)

def expert_model_parallel_all_reduce(input_: torch.Tensor) -> torch.Tensor:
    """All-reduce the input tensor across model parallel group."""
    return get_ep_group().all_reduce(input_)

def tensor_model_parallel_all_gather_into_tensor(output_: torch.Tensor,
                                                 input_: torch.Tensor) -> None:
    """All-gather the input tensor into output tensor across model parallel group."""
    return get_tp_group().all_gather_into_tensor(output_, input_)

def get_attn_sp_token_slice_pos(forward_batch):
    cu_start_cpu = torch.tensor([0], dtype = torch.int32)
    attn_sp_token_nums = get_attn_tp_group().get_local_sp_token_num(forward_batch.global_sp_num_tokens)
    attn_sp_token_nums_cpu = torch.tensor(attn_sp_token_nums, dtype =torch.int32)
    cu_attn_sp_token_nums_cpu = torch.concat((cu_start_cpu, torch.cumsum(attn_sp_token_nums_cpu, dim = 0)))
    cp_rank = get_attn_tp_group().rank_in_group
    slice_start = cu_attn_sp_token_nums_cpu[cp_rank].item()
    slice_end = cu_attn_sp_token_nums_cpu[cp_rank+1].item()
    return slice_start, slice_end
