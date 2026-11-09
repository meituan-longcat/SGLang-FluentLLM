"""Multimodal feature transport strategies.

Provides a strategy pattern for transferring multimodal feature tensors
(pixel_values, audio_features, etc.) from TokenizerManager to Scheduler via ZMQ.

Four strategies are available, selected by server_args.mm_feature_transport:

  "default"    -- Pickle serialization: tensor bytes travel inside the ZMQ message.
  "shared_mem" -- POSIX shared memory: only metadata (shm_name, shape, dtype)
                  travels in ZMQ; single-node only.
  "cuda_ipc"   -- CUDA IPC memory pool: only IPC handle + byte offset travel
                  in ZMQ; single-node only.
  "auto"       -- Single-node GPU tensor -> cuda_ipc; CPU tensor -> shared_mem;
                  multi-node -> default (Pickle).

Usage (TokenizerManager side):
    strategy = get_mm_transport_strategy(server_args)
    tokenized_obj = strategy.wrap(tokenized_obj)
    zmq_socket.send_pyobj(tokenized_obj)

Usage (Scheduler side, after broadcast_pyobj):
    strategy.unwrap(recv_reqs, scheduler)
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.env import envs, global_server_args_dict
from sglang.srt.managers.io_struct import TokenizedGenerateReqInput
from sglang.srt.multimodal.mm_shared_memory_transport import (
    has_shm_features,
    unwrap_shm_features,
    wrap_shm_features,
)
from sglang.srt.utils.cuda_ipc_transport_utils import (
    CudaIpcTensorTransportProxy,
    MmItemMemoryPool,
)

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs

_IPC_POOL_HANDLE_CACHE = envs.SGLANG_USE_IPC_POOL_HANDLE_CACHE.get()


class MMFeatureTransportStrategy(ABC):
    """Abstract base class for multimodal feature transport strategies."""

    @abstractmethod
    def wrap(self, tokenized_obj) -> object:
        """Tokenizer-side: wrap feature tensors before ZMQ send."""

    @abstractmethod
    def unwrap(self, recv_reqs: list, scheduler) -> None:
        """Scheduler-side: restore feature tensors after broadcast_pyobj."""


class DefaultTransportStrategy(MMFeatureTransportStrategy):
    """Pickle serialization: tensor bytes travel inside the ZMQ message.
    """

    @staticmethod
    def _move_gpu_tensors_to_cpu(tokenized_obj):
        """Move any GPU feature tensor to CPU so pickle does not carry a cuda device tag."""
        if not (hasattr(tokenized_obj, "mm_inputs") and tokenized_obj.mm_inputs):
            return tokenized_obj
        for item in tokenized_obj.mm_inputs.mm_items:
            if isinstance(item.feature, torch.Tensor) and item.feature.is_cuda:
                item.feature = item.feature.cpu()
            elif (
                isinstance(item.precomputed_embeddings, torch.Tensor)
                and item.precomputed_embeddings.is_cuda
            ):
                item.precomputed_embeddings = item.precomputed_embeddings.cpu()
        return tokenized_obj

    def wrap(self, tokenized_obj):
        return self._move_gpu_tensors_to_cpu(tokenized_obj)

    def unwrap(self, recv_reqs, scheduler) -> None:
        pass


class SharedMemTransportStrategy(MMFeatureTransportStrategy):
    """POSIX shared-memory strategy for CPU tensors (single-node only).

    wrap: moves CPU feature tensors into /dev/shm segments; only tiny metadata
          (shm_name, shape, dtype) travels inside the ZMQ message.
    unwrap: after broadcast_pyobj, each TP rank mmaps the segment, clones the
            tensor to private memory, and unlinks the segment. A distributed
            barrier is inserted when world_size > 1 to guarantee all ranks have
            called shm_open before the source rank unlinks the segment.
    """

    def wrap(self, tokenized_obj):
        return wrap_shm_features(tokenized_obj)

    def unwrap(self, recv_reqs, scheduler) -> None:
        if (
            not scheduler.server_args.enable_dp_attention
            and scheduler.world_size > 1
            and has_shm_features(recv_reqs)
        ):
            torch.distributed.barrier(group=scheduler.tp_cpu_group)
        for req in recv_reqs:
            unwrap_shm_features(req)


class CudaIpcTransportStrategy(MMFeatureTransportStrategy):
    """CUDA IPC memory-pool strategy for GPU tensors (single-node only).

    This strategy is the single authoritative owner of the MmItemMemoryPool.
    wrap() and unwrap() are the only entry points — no hooks into base_processor.

    Precondition for wrap(): item.hash must already be set on every mm_item
    (base_processor.process_and_combine_mm_data() pre-hashes unconditionally),
    so set_pad_value() has already run without touching these tensors.

    wrap (called in _send_one_request, after from_processor_output):
        Replaces each GPU tensor with a CudaIpcTensorTransportProxy.
        If the pool is full, falls back to tensor.cpu() so the SHM path handles it.

    unwrap (called in recv_requests):
        Reconstructs each CudaIpcTensorTransportProxy back to a real GPU tensor
        on the scheduler's device, then increments the sync counter so the
        producer can recycle the pool chunk.
    """

    def __init__(self, server_args: "ServerArgs"):
        pool_bytes = server_args.mm_feature_cache_mb * 1024 * 1024
        recycle_interval = envs.SGLANG_MM_ITEM_MEMORY_POOL_RECYCLE_INTERVAL.get()
        self._pool = MmItemMemoryPool(pool_bytes, recycle_interval)

    def _wrap_gpu_tensor(self, tensor: torch.Tensor) -> object:
        """Wrap one GPU tensor with a CUDA IPC proxy, or fall back to CPU on pool-full."""
        sync_flag, available_slice, byte_offset = (
            self._pool.return_a_slice_tensor_with_flag(tensor)
        )
        if isinstance(available_slice, torch.Tensor):
            available_slice.copy_(
                tensor.view(torch.int8).view(-1), non_blocking=True
            )
            return CudaIpcTensorTransportProxy(
                data=available_slice,
                info_data=tensor,
                sync_buffer_meta=sync_flag,
                pool_ipc_handle=(
                    self._pool._pool_ipc_handle if _IPC_POOL_HANDLE_CACHE else None
                ),
                pool_byte_offset=byte_offset,
                pool_device_index=self._pool._pool_device_index,
            )
        # Pool full — fall back to CPU tensor for the SHM path
        return tensor.cpu()

    def wrap(self, tokenized_obj) -> object:
        """Replace GPU feature tensors with CUDA IPC proxies before ZMQ send."""
        if not (hasattr(tokenized_obj, "mm_inputs") and tokenized_obj.mm_inputs):
            return tokenized_obj
        for item in tokenized_obj.mm_inputs.mm_items:
            if isinstance(item.feature, torch.Tensor) and item.feature.is_cuda:
                item.feature = self._wrap_gpu_tensor(item.feature)
            elif (
                isinstance(item.precomputed_embeddings, torch.Tensor)
                and item.precomputed_embeddings.is_cuda
            ):
                item.precomputed_embeddings = self._wrap_gpu_tensor(
                    item.precomputed_embeddings
                )
        return tokenized_obj

    def unwrap(self, recv_reqs, scheduler) -> None:
        for req in recv_reqs:
            if isinstance(req, TokenizedGenerateReqInput) and req.mm_inputs:
                req.mm_inputs.reconstruct_features()


class AutoTransportStrategy(MMFeatureTransportStrategy):
    """Automatically selects the best strategy based on the runtime environment.

    Single-node: GPU tensors -> CudaIpc; remaining CPU tensors -> SharedMem.
    Multi-node:  Degrades to Default (no-op) — tensors are fully serialized via Pickle.

    wrap() applies IPC wrapping first, then SHM wrapping. Pool-full fallback inside
    CudaIpcTransportStrategy moves GPU tensors to CPU, which SharedMem then picks up.
    """

    def __init__(self, server_args: "ServerArgs"):
        self._is_single_node = global_server_args_dict.get("nnodes", 1) == 1
        self._shm = SharedMemTransportStrategy()
        self._ipc = CudaIpcTransportStrategy(server_args) if self._is_single_node else None
        self._default = DefaultTransportStrategy()

    def wrap(self, tokenized_obj):
        if not self._is_single_node:
            return self._default.wrap(tokenized_obj)
        # Wrap GPU tensors as CUDA IPC proxies first; pool-full fallback moves
        # them to CPU, which SharedMem then picks up in the second pass.
        tokenized_obj = self._ipc.wrap(tokenized_obj)
        return self._shm.wrap(tokenized_obj)

    def unwrap(self, recv_reqs, scheduler) -> None:
        if not self._is_single_node:
            return
        self._shm.unwrap(recv_reqs, scheduler)
        self._ipc.unwrap(recv_reqs, scheduler)


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

from sglang.srt.utils import is_npu


class NpuDefaultTransportStrategy(MMFeatureTransportStrategy):
    """NPU "default" mode: shared_mem on single-node, default (Pickle) multi-node.

    NPU has no CUDA IPC, so unlike GPU's DefaultTransportStrategy (always Pickle),
    this uses shared_mem (single-node) or default (multi-node). Outer routing:
    NPU dispatches here via the is_npu() check in get_mm_transport_strategy when
    mode=="default"; GPU keeps DefaultTransportStrategy / CudaIpcTransportStrategy.
    """

    def __init__(self, server_args: "ServerArgs"):
        self._is_single_node = global_server_args_dict.get("nnodes", 1) == 1
        self._shm = SharedMemTransportStrategy()
        self._default = DefaultTransportStrategy()

    def wrap(self, tokenized_obj):
        if not self._is_single_node:
            return self._default.wrap(tokenized_obj)
        return self._shm.wrap(tokenized_obj)

    def unwrap(self, recv_reqs, scheduler) -> None:
        if not self._is_single_node:
            return
        self._shm.unwrap(recv_reqs, scheduler)


def get_mm_transport_strategy(server_args: "ServerArgs") -> MMFeatureTransportStrategy:
    """Instantiate and return the transport strategy for the given server_args."""
    mode = getattr(server_args, "mm_feature_transport", "auto")
    # On NPU, "default" routes to NpuDefaultTransportStrategy (shared_mem single-node,
    # Pickle multi-node) instead of GPU's plain DefaultTransportStrategy (always Pickle).
    # server_args forces mm_feature_transport="default" on decode nodes (no vision
    # encoder) and asserts "default" on NPU prefill, so NPU PD-disagg uniformly uses
    # NpuDefaultTransportStrategy.
    if mode == "default":
        if is_npu():
            return NpuDefaultTransportStrategy(server_args)
        return DefaultTransportStrategy()
    if mode == "shared_mem":
        return SharedMemTransportStrategy()
    elif mode == "cuda_ipc":
        return CudaIpcTransportStrategy(server_args)
    elif mode == "auto":
        return AutoTransportStrategy(server_args)
    else:
        raise ValueError(
            f"Unknown mm_feature_transport mode: {mode!r}. "
            "Valid values: 'default', 'shared_mem', 'cuda_ipc', 'auto'."
        )
