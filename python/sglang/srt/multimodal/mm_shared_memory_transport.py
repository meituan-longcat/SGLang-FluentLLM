"""
POSIX Shared Memory tensor transport utilities.

Pure utilities used by SharedMemTransportStrategy in mm_feature_transport.py.
Transport-mode routing (single-node vs multi-node) lives in the strategy layer.

Public API
----------
ShmPointerMMData
    Wraps a CPU tensor as a POSIX /dev/shm pointer. Only the tiny metadata
    (shm_name, shape, dtype) travels inside the ZMQ message.

wrap_shm_features(obj)
    Called by SharedMemTransportStrategy.wrap() before ZMQ send.

has_shm_features(recv_reqs) -> bool
    Returns True if any request contains a ShmPointerMMData feature.

unwrap_shm_features(obj)
    Called by SharedMemTransportStrategy.unwrap() after broadcast_pyobj(),
    materialises ShmPointerMMData back to torch.Tensor.
"""

import torch
from multiprocessing import shared_memory as _shared_memory

from sglang.srt.env import global_server_args_dict


# ---------------------------------------------------------------------------
# ShmPointerMMData
# ---------------------------------------------------------------------------
class ShmPointerMMData:
    """
    Wraps a CPU tensor as a POSIX shared-memory "pointer".

    Producer side (__init__):
      1. Allocates a /dev/shm segment large enough for the tensor.
      2. Copies tensor bytes into the segment (one copy).
      3. Closes the local fd without deleting the segment.
      4. Only shm_name / shape / dtype are serialised into the ZMQ message.

    Consumer side (__setstate__):
      1. Opens the segment by name (mmap – zero copy).
      2. Creates a tensor view over the mapped memory.

    materialize():
      1. Clones the view to private memory (one copy).
      2. Closes and unlinks the segment.
      FileNotFoundError from unlink is silently ignored (another TP rank
      may have already unlinked it).
    """

    def __init__(self, tensor: torch.Tensor):
        if not tensor.is_cpu:
            tensor = tensor.cpu()
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        self.shape = tensor.shape
        self.dtype = tensor.dtype
        nbytes = tensor.numel() * tensor.element_size()
        shm = _shared_memory.SharedMemory(create=True, size=nbytes)
        try:
            dst = torch.frombuffer(shm.buf, dtype=torch.uint8)
            dst.copy_(tensor.view(torch.uint8).reshape(-1))
        except BaseException:
            shm.close()
            shm.unlink()
            raise
        self.shm_name = shm.name
        shm.close()  # close local fd; segment persists until unlink
        self._shm_handle = None

    def __getstate__(self):
        return {
            "shm_name": self.shm_name,
            "shape": self.shape,
            "dtype": self.dtype,
        }

    def __setstate__(self, state):
        self.shm_name = state["shm_name"]
        self.shape = state["shape"]
        self.dtype = state["dtype"]
        self.shm = None
        # Open existing segment (mmap – zero copy)
        self._shm_handle = _shared_memory.SharedMemory(name=self.shm_name)
        self.tensor = torch.frombuffer(self._shm_handle.buf, dtype=self.dtype).reshape(
            self.shape
        )

    def materialize(self) -> torch.Tensor:
        """Clone tensor from shm to owned memory, then release shm handle."""
        tensor = self.tensor.clone()
        if self._shm_handle is not None:
            self._shm_handle.close()
            try:
                self._shm_handle.unlink()
            except FileNotFoundError:
                pass  # Another TP rank already unlinked
            self._shm_handle = None
        return tensor

    def __del__(self):
        # Only close; never unlink here – that is materialize()'s responsibility.
        if getattr(self, "_shm_handle", None) is not None:
            self._shm_handle.close()
            self._shm_handle = None


# ---------------------------------------------------------------------------
# Producer-side helper: wrap CPU features before ZMQ send
# ---------------------------------------------------------------------------
def wrap_shm_features(obj):
    """Wrap CPU tensor features in ShmPointerMMData before ZMQ serialization.

    Pure utility — contains no transport-mode routing. The caller
    (SharedMemTransportStrategy) is responsible for deciding whether to invoke
    this function.
    """
    if global_server_args_dict.get("skip_tokenizer_init", False):
        return obj

    if hasattr(obj, "mm_inputs") and obj.mm_inputs:
        for item in obj.mm_inputs.mm_items:
            if isinstance(item.feature, torch.Tensor) and item.feature.is_cpu:
                item.feature = ShmPointerMMData(item.feature)
    return obj


# ---------------------------------------------------------------------------
# Consumer-side helpers: detect and unwrap after broadcast
# ---------------------------------------------------------------------------
def has_shm_features(recv_reqs) -> bool:
    """Return True if any request in the list contains a ShmPointerMMData feature."""
    for req in recv_reqs:
        if hasattr(req, "batch"):
            if has_shm_features(req.batch):
                return True
        elif hasattr(req, "mm_inputs") and req.mm_inputs:
            for item in req.mm_inputs.mm_items:
                if isinstance(item.feature, ShmPointerMMData):
                    return True
    return False


def unwrap_shm_features(obj):
    """Materialise ShmPointerMMData back into standard torch.Tensors.

    Called by SharedMemTransportStrategy.unwrap() after barrier() ensures every
    TP rank has opened the shared-memory segment (shm_open via __setstate__).
    Handles both single requests and batch requests.
    """
    # Handle batch requests
    if hasattr(obj, "batch"):
        for sub_obj in obj.batch:
            unwrap_shm_features(sub_obj)
        return obj
    # Handle single requests
    if hasattr(obj, "mm_inputs") and obj.mm_inputs:
        for item in obj.mm_inputs.mm_items:
            if isinstance(item.feature, ShmPointerMMData):
                item.feature = item.feature.materialize()
    return obj
