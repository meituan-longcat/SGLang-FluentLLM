"""Physical expert views used by EPLB weight migration.

Ascend tensors may have a physical layout and storage size that do not match
their logical ``dtype`` and ``shape``.  In particular this is true for the
FRACTAL_NZ packed INT4 weights used by W4A4.  EPLB therefore transports every
NPU expert tensor as physical ``uint8`` bytes.  Besides preserving opaque NPU
layouts, this keeps packing/communication operators independent of model dtypes
such as ``uint64`` (which Ascend ``aclnnCat`` does not support).
"""

import torch

FRACTAL_NZ_FORMAT = 29


def _validate_num_local_physical_experts(
    num_local_physical_experts: int,
) -> None:
    if num_local_physical_experts <= 0:
        raise ValueError(f"Invalid {num_local_physical_experts=}; expected > 0")


def is_npu_nz_tensor(tensor: torch.Tensor) -> bool:
    """Return whether ``tensor`` uses the Ascend FRACTAL_NZ storage format."""
    if tensor.device.type != "npu":
        return False

    import torch_npu

    return torch_npu.get_npu_format(tensor) == FRACTAL_NZ_FORMAT


def get_npu_storage_nbytes(tensor: torch.Tensor) -> int:
    """Return the physical NPU storage size in bytes."""
    if tensor.device.type != "npu":
        raise ValueError(f"Expected an NPU tensor, but got {tensor.device=}")

    import torch_npu

    return int(torch_npu.get_storage_size(tensor)) * tensor.element_size()


def get_physical_expert_nbytes(
    tensor: torch.Tensor, num_local_physical_experts: int
) -> int:
    """Return one expert's physical byte size in a leading-dimension tensor."""
    _validate_num_local_physical_experts(num_local_physical_experts)
    if tensor.ndim == 0 or tensor.shape[0] != num_local_physical_experts:
        raise ValueError(
            f"Leading dimension {tuple(tensor.shape)} does not match "
            f"{num_local_physical_experts=}"
        )
    if tensor.storage_offset() != 0:
        raise ValueError(
            "EPLB physical transfer requires a tensor starting at storage offset 0"
        )

    storage_nbytes = get_npu_storage_nbytes(tensor)
    if storage_nbytes % num_local_physical_experts != 0:
        raise ValueError(
            f"Physical storage {storage_nbytes} bytes is not divisible by "
            f"{num_local_physical_experts=}"
        )
    return storage_nbytes // num_local_physical_experts


def get_expert_transfer_view(
    tensor: torch.Tensor,
    local_expert_index: int,
    num_local_physical_experts: int,
) -> torch.Tensor:
    """Return the exact buffer EPLB should copy/send for one expert.

    Every NPU tensor is exposed as a flat uint8 view over one contiguous
    physical expert block.  CPU tensors retain the logical expert view so this
    helper remains usable by CPU-only scheduling tests.
    """
    _validate_num_local_physical_experts(num_local_physical_experts)
    if not 0 <= local_expert_index < num_local_physical_experts:
        raise IndexError(
            f"{local_expert_index=} is outside "
            f"[0, {num_local_physical_experts})"
        )
    if tensor.ndim == 0 or tensor.shape[0] != num_local_physical_experts:
        raise ValueError(
            f"Leading dimension {tuple(tensor.shape)} does not match "
            f"{num_local_physical_experts=}"
        )

    if tensor.device.type != "npu":
        return tensor.narrow(0, local_expert_index, 1).squeeze(0)

    # For ordinary NPU tensors the leading expert dimension must map to
    # consecutive storage blocks.  FRACTAL_NZ is intentionally exempt: its
    # physical storage is opaque, and get_npu_storage_nbytes() below is the
    # source of truth for its per-expert block size.
    if not is_npu_nz_tensor(tensor) and not tensor.is_contiguous():
        raise ValueError(
            "EPLB physical transfer requires non-NZ NPU tensors to be "
            f"contiguous, but got shape={tuple(tensor.shape)} "
            f"stride={tuple(tensor.stride())}"
        )

    import torchair

    expert_nbytes = get_physical_expert_nbytes(tensor, num_local_physical_experts)
    expert_data_ptr = tensor.data_ptr() + local_expert_index * expert_nbytes
    return torchair.llm_datadist.create_npu_tensors(
        [expert_nbytes], torch.uint8, [expert_data_ptr]
    )[0]


def create_expert_transfer_buffers(
    tensor: torch.Tensor, num_local_physical_experts: int
) -> list[torch.Tensor]:
    """Allocate contiguous staging buffers matching each expert view."""
    sample_view = get_expert_transfer_view(
        tensor,
        local_expert_index=0,
        num_local_physical_experts=num_local_physical_experts,
    )
    return [
        torch.empty(
            sample_view.shape,
            dtype=sample_view.dtype,
            layout=sample_view.layout,
            device=sample_view.device,
        )
        for _ in range(num_local_physical_experts)
    ]
