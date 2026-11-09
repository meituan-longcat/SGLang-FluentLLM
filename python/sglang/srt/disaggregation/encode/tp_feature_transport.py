"""Send Encoder patch tensors directly through the existing CPU TP group.

Only small, tensor-free metadata is pickled. Receivers keep feature/grid tensors
on CPU; each rank's DeviceWorker/model entry point performs device placement.
"""

import copy
import pickle
from typing import NamedTuple, Optional

import torch
import torch.distributed as dist


class GridMetadata(NamedTuple):
    shape: tuple
    dtype: torch.dtype
    values: list


class FeatureMetadata(NamedTuple):
    request_id: str
    output_format: str
    item_index: int
    cost: int
    item: object
    feature_shape: tuple
    feature_dtype: torch.dtype
    grid: Optional[GridMetadata]


def _validate_metadata(value, active=None):
    """Allow only known scalar/container types, including dictionary keys.

    Rejecting arbitrary objects also prevents their pickle reducers or hidden
    attributes from reintroducing a tensor storage into the metadata channel.
    Exact type checks deliberately exclude container/scalar subclasses.
    """
    from sglang.srt.managers.scheduling.multimodal_data import (
        Modality,
        MultimodalInputFormat,
    )

    value_type = type(value)
    if value_type in (
        type(None), bool, int, float, str, bytes, Modality, MultimodalInputFormat
    ):
        return
    if value_type not in (dict, list, tuple):
        raise TypeError(f"Unsupported Encoder TP metadata type: {value_type.__name__}")
    if active is None:
        active = set()
    identity = id(value)
    if identity in active:
        raise TypeError("Encoder TP metadata must not contain reference cycles")
    active.add(identity)
    try:
        if value_type is dict:
            for key, item in value.items():
                _validate_metadata(key, active)
                _validate_metadata(item, active)
        else:
            for item in value:
                _validate_metadata(item, active)
    finally:
        active.remove(identity)


def _split_entry(entry):
    from sglang.srt.managers.scheduling.multimodal_data import MultimodalDataItem

    request_id, output_format, item_index, cost, item = entry
    if type(request_id) is not str or type(output_format) is not str:
        raise TypeError("Encoder TP request_id and output_format must be strings")
    if (
        type(item_index) is not int
        or item_index < 0
        or type(cost) is not int
        or cost < 0
    ):
        raise ValueError("Encoder TP item_index and cost must be nonnegative integers")
    if type(item) is not MultimodalDataItem:
        raise TypeError("Encoder TP item must be a MultimodalDataItem")
    if not isinstance(item.feature, torch.Tensor):
        raise TypeError("Encoder TP feature must be a torch.Tensor")
    if item.feature.layout != torch.strided or item.feature.dtype not in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
        torch.uint8,
        torch.int8,
        torch.int32,
        torch.int64,
        torch.bool,
    ):
        raise TypeError("Encoder TP feature layout/dtype is not supported by Gloo")
    if item.precomputed_embeddings is not None:
        raise TypeError("Encoder TP feature cannot also have precomputed_embeddings")
    feature = item.feature.detach().to(device="cpu").contiguous()
    item.feature = feature
    metadata_item = copy.copy(item)
    metadata_item.feature = None
    if type(item.model_specific_data) is not dict:
        raise TypeError("Encoder TP model_specific_data must be a metadata dictionary")
    metadata_item.model_specific_data = dict(item.model_specific_data)
    grid = metadata_item.model_specific_data.pop("image_grid_thw", None)
    grid_descriptor = None
    if grid is not None:
        if (
            not isinstance(grid, torch.Tensor)
            or grid.layout != torch.strided
            or grid.dtype not in (torch.int32, torch.int64)
            or grid.ndim != 2
            or grid.shape[1] != 3
        ):
            raise TypeError("Encoder TP image_grid_thw must be an integer [N, 3] tensor")
        grid = grid.detach().to(device="cpu")
        if bool((grid <= 0).any()):
            raise ValueError("Encoder TP image_grid_thw values must be positive")
        item.model_specific_data = dict(item.model_specific_data)
        item.model_specific_data["image_grid_thw"] = grid
        grid_descriptor = GridMetadata(tuple(grid.shape), grid.dtype, grid.tolist())
    # Cover every field, including dynamically attached item attributes.
    _validate_metadata(vars(metadata_item))
    return (
        FeatureMetadata(
            request_id, output_format, item_index, cost, metadata_item,
            tuple(feature.shape), feature.dtype, grid_descriptor,
        ),
        feature,
    )


def _check_all_ranks(error, group, phase):
    ready = torch.tensor([int(error is None)], dtype=torch.int32, device="cpu")
    dist.all_reduce(ready, op=dist.ReduceOp.MIN, group=group)
    if not ready.item():
        detail = error if error is not None else "failure on another rank"
        raise RuntimeError(
            f"Encoder TP {phase} failed on at least one rank: {detail}"
        )


def broadcast_features(entries, *, rank, group, src=0):
    """Broadcast scheduler cache-miss entries; return restored receiver entries.

    Readiness checks keep allocation/restoration failures from making ranks
    execute different numbers of tensor collectives. This function does not
    change which requests the scheduler drains or batches. Communication
    exceptions propagate to the caller, as with broadcast_pyobj; this helper
    does not change the service's failure or worker lifecycle policy.
    """
    from sglang.srt.utils import broadcast_pyobj

    features = []
    header = None
    if rank == src:
        try:
            metadata = []
            for entry in entries:
                descriptor, feature = _split_entry(entry)
                metadata.append(descriptor)
                features.append(feature)
            header = {"entries": metadata, "error": None}
            # A source pickle failure must be communicated before peers wait
            # for tensor data. This preflight contains no feature-sized bytes.
            pickle.dumps(header)
        except Exception as error:
            features = []
            header = {"entries": [], "error": f"{type(error).__name__}: {error}"}
    header = broadcast_pyobj(header, rank=rank, dist_group=group, src=src)
    if type(header) is not dict or set(header) != {"entries", "error"}:
        raise TypeError("Invalid Encoder TP feature header")
    error = header["error"]
    metadata = header["entries"]
    if error is not None and type(error) is not str:
        raise TypeError("Invalid Encoder TP preparation error")
    if type(metadata) is not list or any(
        type(m) is not FeatureMetadata for m in metadata
    ):
        raise TypeError("Invalid Encoder TP feature descriptors")
    if error is not None:
        raise RuntimeError(f"Encoder TP feature preparation failed: {error}")
    if not metadata:
        return []

    allocation_error = None
    if rank != src:
        try:
            features = [
                torch.empty(m.feature_shape, dtype=m.feature_dtype, device="cpu")
                for m in metadata
            ]
        except Exception as error:
            allocation_error = f"{type(error).__name__}: {error}"
    _check_all_ranks(allocation_error, group, "allocation")

    for feature in features:
        dist.broadcast(feature, src=src, group=group)

    received = []
    restore_error = None
    if rank != src:
        try:
            for m, feature in zip(metadata, features):
                item = m.item
                item.feature = feature
                if m.grid is not None:
                    item.model_specific_data["image_grid_thw"] = torch.tensor(
                        m.grid.values, dtype=m.grid.dtype, device="cpu"
                    ).reshape(m.grid.shape)
                received.append((m.request_id, m.output_format, m.item_index, m.cost, item))
        except Exception as error:
            restore_error = f"{type(error).__name__}: {error}"
    _check_all_ranks(restore_error, group, "restore")
    return received
