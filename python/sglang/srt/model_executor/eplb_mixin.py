from collections import defaultdict
from typing import NamedTuple, Optional

import einops
import torch
import torch.distributed

from sglang.srt.distributed import get_moe_ep_group
from sglang.srt.env import global_server_args_dict
from sglang.srt.managers.eplb.eplb_manager import EPLBManager
from sglang.srt.managers.eplb.expert_distribution import (
    ExpertDistributionRecorder,
    set_global_expert_distribution_recorder,
)
from sglang.srt.managers.eplb.expert_location import (
    ExpertLocationMetadata,
    compute_initial_expert_location_metadata,
    get_global_expert_location_metadata,
    set_global_expert_location_metadata,
)
from sglang.srt.model_executor.eplb_physical_transfer import (
    create_expert_transfer_buffers,
    get_expert_transfer_view,
)
from sglang.srt.utils import get_bool_env_var, get_colorful_logger, get_device_module

logger = get_colorful_logger(__name__)

device_module = get_device_module()


class _EplbTransfer(NamedTuple):
    tensor_index: int
    direction: str
    tensor: torch.Tensor
    peer: int


def flatten_2d_list(nested_list):
    if len(nested_list) == 0:
        return []
    if not any(isinstance(i, list) for i in nested_list):
        return nested_list
    return [x for sublist in nested_list for x in sublist]


def list_depth(data):
    if not isinstance(data, list):
        return 0
    if all(isinstance(item, list) for item in data):
        return 2
    return 1


def inplace_perm_helper(x, perm):
    new_tensor = x[perm].clone()
    x.copy_(new_tensor)


def construct_perm_from_phy_to_logic_mapping(old_mapping: list[int], new_mapping: list[int]):
    map_dict = dict(zip(old_mapping, range(len(old_mapping))))
    gather_indices = [map_dict[lid] for lid in new_mapping]
    return gather_indices


class ExpertLocationUpdater:
    def __init__(self):
        self._first_execution = True

    def update(
        self,
        routed_experts_weights_of_layer: dict[int, list[torch.Tensor]],
        device_local_routed_experts_weights_of_layer: dict[int, list[torch.Tensor]],
        new_expert_location_metadata: ExpertLocationMetadata,
        update_layer_ids: list[int],
        nnodes_per_moe_ep_group: int,
        moe_ep_rank: int,
        moe_ep_size: int,
        moe_ep_group_ranks: list[int],
    ):
        if self._first_execution:
            self._first_execution = False
            device_module.empty_cache()

        old_expert_location_metadata = get_global_expert_location_metadata()
        _update_expert_weights(
            routed_experts_weights_of_layer=routed_experts_weights_of_layer,
            device_local_routed_experts_weights_of_layer=device_local_routed_experts_weights_of_layer,
            old_expert_location_metadata=old_expert_location_metadata,
            new_expert_location_metadata=new_expert_location_metadata,
            update_layer_ids=update_layer_ids,
            nnodes_per_moe_ep_group=nnodes_per_moe_ep_group,
            moe_ep_rank=moe_ep_rank,
            moe_ep_size=moe_ep_size,
            moe_ep_group_ranks=moe_ep_group_ranks,
        )
        old_expert_location_metadata.update(
            new_expert_location_metadata,
            update_layer_ids=update_layer_ids,
        )


def _update_expert_weights(**kwargs):
    if get_bool_env_var("SGLANG_EXPERT_LOCATION_UPDATER_CANARY"):
        return _update_expert_weights_with_canary(**kwargs)
    else:
        return _update_expert_weights_raw(**kwargs)


# can add watchdog as well
def _update_expert_weights_with_canary(
    routed_experts_weights_of_layer: dict[int, list[torch.Tensor]],
    device_local_routed_experts_weights_of_layer: dict[int, list[torch.Tensor]],
    old_expert_location_metadata: ExpertLocationMetadata,
    new_expert_location_metadata: ExpertLocationMetadata,
    update_layer_ids: list[int],
    nnodes_per_moe_ep_group: int,
    moe_ep_rank: int,
    moe_ep_size: int,
    moe_ep_group_ranks: list[int],
):
    num_local_physical_experts = old_expert_location_metadata.num_local_physical_experts

    def _get_canary_value(meta: ExpertLocationMetadata, layer_id: int):
        return meta.physical_to_logical_map_cpu[
            layer_id,
            num_local_physical_experts * moe_ep_rank : num_local_physical_experts * (moe_ep_rank + 1),
        ]

    routed_experts_weights_of_layer = {k: [x for x in v] for k, v in routed_experts_weights_of_layer.items()}
    for layer_id in update_layer_ids:
        canary_tensor = (
            _get_canary_value(old_expert_location_metadata, layer_id).clone().to(device=global_server_args_dict["device"], non_blocking=True)
        )
        routed_experts_weights_of_layer[layer_id].append(canary_tensor)

    _update_expert_weights_raw(
        routed_experts_weights_of_layer=routed_experts_weights_of_layer,
        device_local_routed_experts_weights_of_layer=device_local_routed_experts_weights_of_layer,
        old_expert_location_metadata=old_expert_location_metadata,
        new_expert_location_metadata=new_expert_location_metadata,
        update_layer_ids=update_layer_ids,
        nnodes_per_moe_ep_group=nnodes_per_moe_ep_group,
        moe_ep_rank=moe_ep_rank,
        moe_ep_size=moe_ep_size,
        moe_ep_group_ranks=moe_ep_group_ranks,
    )

    for layer_id in update_layer_ids:
        # can optimize speed if needed
        expect_value = _get_canary_value(new_expert_location_metadata, layer_id)
        actual_value = routed_experts_weights_of_layer[layer_id][-1].cpu()
        assert torch.all(expect_value == actual_value), (
            f"{expect_value=} {actual_value=} {layer_id=} "
            f"{old_expert_location_metadata.physical_to_logical_map_cpu.tolist()=} "
            f"{new_expert_location_metadata.physical_to_logical_map_cpu.tolist()=} "
        )


def _update_expert_weights_raw(
    routed_experts_weights_of_layer: dict[int, list[torch.Tensor]],
    device_local_routed_experts_weights_of_layer: dict[int, list[torch.Tensor]],
    old_expert_location_metadata: ExpertLocationMetadata,
    new_expert_location_metadata: ExpertLocationMetadata,
    update_layer_ids: list[int],
    nnodes_per_moe_ep_group: int,
    moe_ep_rank: int,
    moe_ep_size: int,
    moe_ep_group_ranks: list[int],
):
    log_metrics = get_bool_env_var("SGLANG_EXPERT_LOCATION_UPDATER_LOG_METRICS")

    num_local_physical_experts = old_expert_location_metadata.num_local_physical_experts
    num_gpu_per_node = moe_ep_size // nnodes_per_moe_ep_group

    for layer_id in update_layer_ids:
        temp_buffers = create_temp_buffers(
            routed_experts_weights_of_layer[layer_id],
            num_local_physical_experts=num_local_physical_experts,
        )

        update_expert_weights_single_layer(
            routed_experts_weights=routed_experts_weights_of_layer[layer_id],
            temp_buffers=temp_buffers,
            old_physical_to_logical_map=old_expert_location_metadata.physical_to_logical_map_cpu[layer_id].tolist(),
            new_physical_to_logical_map=new_expert_location_metadata.physical_to_logical_map_cpu[layer_id].tolist(),
            num_local_physical_experts=num_local_physical_experts,
            num_gpu_per_node=num_gpu_per_node,
            rank=moe_ep_rank,
            world_size=moe_ep_size,
            moe_ep_group_ranks=moe_ep_group_ranks,
            log_metrics=log_metrics,
            debug=False,
        )

        old_mapping = old_expert_location_metadata.physical_to_logical_map_cpu[layer_id].tolist()
        new_mapping = new_expert_location_metadata.physical_to_logical_map_cpu[layer_id].tolist()
        perm = construct_perm_from_phy_to_logic_mapping(old_mapping, new_mapping)
        if device_local_routed_experts_weights_of_layer is not None:
            for local_routed_experts_weights in device_local_routed_experts_weights_of_layer[layer_id]:
                inplace_perm_helper(local_routed_experts_weights, perm)


def create_temp_buffers(sample_tensors, num_local_physical_experts=None):
    if num_local_physical_experts is None:
        return [torch.empty_like(tensor) for tensor in sample_tensors]

    return [create_expert_transfer_buffers(tensor, num_local_physical_experts) for tensor in sample_tensors]


def update_expert_weights_single_layer(
    routed_experts_weights: list[torch.Tensor],
    temp_buffers: list[torch.Tensor],
    old_physical_to_logical_map: list[int],  # (num_physical_Experts,)
    new_physical_to_logical_map: list[int],  # (num_physical_Experts,)
    num_local_physical_experts: int,
    num_gpu_per_node: int,
    rank: int,
    moe_ep_group_ranks: list[int],
    world_size: Optional[int] = None,
    debug: bool = False,
    log_metrics: bool = False,
):
    def _to_global_rank(ep_local_rank: int) -> int:
        return moe_ep_group_ranks[ep_local_rank]

    if num_local_physical_experts <= 0:
        raise ValueError(f"Invalid {num_local_physical_experts=}")
    if num_gpu_per_node <= 0:
        raise ValueError(f"Invalid {num_gpu_per_node=}")
    if not routed_experts_weights:
        raise ValueError("EPLB requires at least one routed expert tensor")
    if not all(tensor.ndim > 0 and tensor.shape[0] == num_local_physical_experts for tensor in routed_experts_weights):
        raise ValueError(f"{num_local_physical_experts=} does not match routed tensor shapes: {[tuple(x.shape) for x in routed_experts_weights]}")
    if not isinstance(old_physical_to_logical_map, list) or not isinstance(new_physical_to_logical_map, list):
        raise TypeError("EPLB physical-to-logical mappings must be lists")
    if len(old_physical_to_logical_map) != len(new_physical_to_logical_map):
        raise ValueError(
            f"Old and new EPLB mappings must have equal lengths: old={len(old_physical_to_logical_map)} new={len(new_physical_to_logical_map)}"
        )
    inferred_world_size, remainder = divmod(len(old_physical_to_logical_map), num_local_physical_experts)
    if remainder or inferred_world_size <= 0:
        raise ValueError(
            f"EPLB mapping length must be a positive multiple of {num_local_physical_experts=}: mapping_length={len(old_physical_to_logical_map)}"
        )
    if world_size is not None and world_size != inferred_world_size:
        raise ValueError(f"{world_size=} does not match mapping-derived world size {inferred_world_size}")
    if len(moe_ep_group_ranks) != inferred_world_size:
        raise ValueError(f"MoE EP group rank count does not match mapping-derived world size: ranks={moe_ep_group_ranks}, {inferred_world_size=}")
    if not 0 <= rank < inferred_world_size:
        raise ValueError(f"Invalid {rank=} for world size {inferred_world_size}")
    if inferred_world_size % num_gpu_per_node != 0:
        raise ValueError(f"World size {inferred_world_size} is not divisible by {num_gpu_per_node=}")
    missing_logical_experts = set(new_physical_to_logical_map).difference(old_physical_to_logical_map)
    if missing_logical_experts:
        raise ValueError(f"New EPLB mapping references logical experts with no source: {sorted(missing_logical_experts)}")

    output_logs = [] if debug else None

    num_physical_experts = len(old_physical_to_logical_map)
    num_tensors = len(routed_experts_weights)

    self_node_id = rank // num_gpu_per_node

    local_expert_location_range = (
        rank * num_local_physical_experts,
        (rank + 1) * num_local_physical_experts,
    )

    def _entrypoint():
        # list[tuple[logical_expert_id, list[_EplbTransfer]]]
        p2p_op_infos: list[tuple[int, list[_EplbTransfer]]] = []
        # List[Tuple[temp_buffers_expert_location, routed_experts_weights_expert_location]]
        buffer2weight_copy_infos: list[tuple[int, int]] = []

        _handle_recv(buffer2weight_copy_infos, p2p_op_infos)
        _create_isend_ops(p2p_op_infos)
        _execute_transfer_collectives(p2p_op_infos)
        _execute_buffer2weight_copies(buffer2weight_copy_infos)

        if log_metrics:
            _log_p2p_op_metrics(
                p2p_op_infos,
                world_size=(world_size if world_size is not None else inferred_world_size),
                num_gpu_per_node=num_gpu_per_node,
                self_node_id=self_node_id,
                moe_ep_group_ranks=moe_ep_group_ranks,
            )

        if debug:
            output_logs.append(f"{p2p_op_infos=}")
            output_logs.append(f"{buffer2weight_copy_infos=}")

    def _handle_recv(buffer2weight_copy_infos, p2p_op_infos):
        for dst_expert_location in range(*local_expert_location_range):
            _handle_recv_of_dst_expert_location(dst_expert_location, buffer2weight_copy_infos, p2p_op_infos)

    def _handle_recv_of_dst_expert_location(dst_expert_location: int, buffer2weight_copy_infos, p2p_op_infos):
        logical_expert_id = new_physical_to_logical_map[dst_expert_location]

        # case 1: unchanged
        if old_physical_to_logical_map[dst_expert_location] == logical_expert_id:
            if debug:
                output_logs.append(f"handle_recv_of_dst_expert_location {dst_expert_location=} case=unchanged")
            return

        # case 2: same-gpu
        for src_expert_location in range(*local_expert_location_range):
            if old_physical_to_logical_map[src_expert_location] == logical_expert_id:
                for i in range(num_tensors):
                    _get_tensor(temp_buffers, i, dst_expert_location).copy_(_get_transfer_tensor(routed_experts_weights, i, src_expert_location))
                buffer2weight_copy_infos.append((dst_expert_location, dst_expert_location))
                if debug:
                    output_logs.append(f"handle_recv_of_dst_expert_location {dst_expert_location=} case=same-gpu {src_expert_location=}")
                return

        # case 3: free-rider
        for src_expert_location in range(rank * num_local_physical_experts, dst_expert_location):
            if new_physical_to_logical_map[src_expert_location] == logical_expert_id:
                buffer2weight_copy_infos.append((src_expert_location, dst_expert_location))
                if debug:
                    output_logs.append(f"handle_recv_of_dst_expert_location {dst_expert_location=} case=free-rider {src_expert_location=}")
                return

        same_node_mapping, cross_node_mapping, need_comm_self_node_dst_ranks = _compute_comm_info(logical_expert_id=logical_expert_id)

        # case 4: same-node
        if rank in need_comm_self_node_dst_ranks:
            chosen_src_rank = same_node_mapping.chunk_value_from_element_value(element_value=rank)
            _create_p2p_recv_and_buffer2weight_copy(
                buffer2weight_copy_infos,
                p2p_op_infos,
                src_rank=chosen_src_rank,
                logical_expert_id=logical_expert_id,
                dst_expert_location=dst_expert_location,
            )
            if debug:
                output_logs.append(f"handle_recv_of_dst_expert_location {dst_expert_location=} case=same-node {chosen_src_rank=}")
            return

        # case 5: cross-node
        # Future work: can optimize when there are multiple ranks in the same dst node that uses the same logical expert
        chosen_src_rank = cross_node_mapping.chunk_value_from_element_value(element_value=rank)
        _create_p2p_recv_and_buffer2weight_copy(
            buffer2weight_copy_infos,
            p2p_op_infos,
            src_rank=chosen_src_rank,
            logical_expert_id=logical_expert_id,
            dst_expert_location=dst_expert_location,
        )
        if debug:
            output_logs.append(f"handle_recv_of_dst_expert_location {dst_expert_location=} case=cross-node {chosen_src_rank=}")
        return

    def _create_p2p_recv_and_buffer2weight_copy(
        buffer2weight_copy_infos,
        p2p_op_infos,
        *,
        logical_expert_id: int,
        src_rank: int,
        dst_expert_location: int,
    ):
        p2p_op_infos.append(
            (
                logical_expert_id,
                [
                    _EplbTransfer(
                        tensor_index=i,
                        direction="recv",
                        tensor=_get_tensor(temp_buffers, i, dst_expert_location),
                        peer=_to_global_rank(src_rank),
                    )
                    for i in range(num_tensors)
                ],
            )
        )
        buffer2weight_copy_infos.append((dst_expert_location, dst_expert_location))

    def _create_isend_ops(p2p_op_infos):
        handled_logical_expert_ids = set()
        for src_expert_location in range(*local_expert_location_range):
            logical_expert_id = old_physical_to_logical_map[src_expert_location]

            if logical_expert_id in handled_logical_expert_ids:
                continue
            handled_logical_expert_ids.add(logical_expert_id)

            _create_isend_ops_of_logical_expert_id(logical_expert_id, src_expert_location, p2p_op_infos)

    def _create_isend_ops_of_logical_expert_id(logical_expert_id, src_expert_location, p2p_op_infos):
        same_node_mapping, cross_node_mapping, need_comm_self_node_dst_ranks = _compute_comm_info(logical_expert_id=logical_expert_id)

        same_node_dst_ranks = same_node_mapping.element_values_from_chunk_value(chunk_value=rank)
        cross_node_dst_ranks = cross_node_mapping.element_values_from_chunk_value(chunk_value=rank)
        all_dst_ranks = same_node_dst_ranks + cross_node_dst_ranks

        if debug:
            output_logs.append(
                f"create_isend_ops_of_logical_expert_id {logical_expert_id=} {src_expert_location=} {same_node_dst_ranks=} {cross_node_dst_ranks=}"
            )

        if not all_dst_ranks:
            p2p_op_infos.append((logical_expert_id, []))
            return

        p2p_op_infos.append(
            (
                logical_expert_id,
                [
                    _EplbTransfer(
                        tensor_index=i,
                        direction="send",
                        tensor=_get_transfer_tensor(routed_experts_weights, i, src_expert_location),
                        peer=_to_global_rank(dst_rank),
                    )
                    for dst_rank in all_dst_ranks
                    for i in range(num_tensors)
                ],
            )
        )

    def _compute_comm_info(logical_expert_id: int):
        all_src_ranks = _deduplicate_ordered(
            [x // num_local_physical_experts for x in range(num_physical_experts) if old_physical_to_logical_map[x] == logical_expert_id]
        )
        all_src_nodes = [x // num_gpu_per_node for x in all_src_ranks]
        self_node_src_ranks = [x for x in all_src_ranks if x // num_gpu_per_node == self_node_id]

        need_comm_dst_ranks = _deduplicate_ordered(
            [
                x // num_local_physical_experts
                for x in range(num_physical_experts)
                if new_physical_to_logical_map[x] == logical_expert_id and x // num_local_physical_experts not in all_src_ranks
            ]
        )
        need_comm_self_node_dst_ranks = (
            [x for x in need_comm_dst_ranks if x // num_gpu_per_node == self_node_id] if len(self_node_src_ranks) > 0 else []
        )
        need_comm_cross_node_dst_ranks = [x for x in need_comm_dst_ranks if (x // num_gpu_per_node) not in all_src_nodes]

        same_node_mapping = _ChunkUtils(
            chunk_values=self_node_src_ranks,
            element_values=need_comm_self_node_dst_ranks,
        )

        cross_node_mapping = _ChunkUtils(
            chunk_values=all_src_ranks,
            element_values=need_comm_cross_node_dst_ranks,
        )

        return same_node_mapping, cross_node_mapping, need_comm_self_node_dst_ranks

    def _execute_transfer_collectives(p2p_op_infos):
        """Execute sparse expert transfers as ordered all-to-all collectives.

        HCCL P2P work on a large world group can report host-side completion
        while cross-topology device work is still stuck.  The next NPU
        synchronization then hangs indefinitely.  EPLB already has the full
        deterministic migration map on every rank, so represent that sparse
        traffic with variable split sizes in all_to_all_single instead.  Every
        rank executes exactly one collective per model tensor, including ranks
        with zero send/receive elements, which also keeps process-group epochs
        aligned without synthetic P2P traffic.
        """
        sorted_transfers = [transfer for _, transfers in sorted(p2p_op_infos, key=lambda info: info[0]) for transfer in transfers]
        effective_world_size = world_size if world_size is not None else inferred_world_size
        if effective_world_size == 1:
            return

        ep_group = get_moe_ep_group().device_group
        distributed_world_size = torch.distributed.get_world_size(group=ep_group)
        if effective_world_size != distributed_world_size:
            raise ValueError(f"EPLB all-to-all world size does not match the MoE EP group: but {effective_world_size=} != {distributed_world_size=}")
        global_to_ep_local_rank = {global_rank: ep_local_rank for ep_local_rank, global_rank in enumerate(moe_ep_group_ranks)}

        for tensor_index in range(num_tensors):
            sample = _get_transfer_tensor(
                routed_experts_weights,
                tensor_index,
                local_expert_location_range[0],
            )
            transfers = [transfer for transfer in sorted_transfers if transfer.tensor_index == tensor_index]
            sends_by_peer = [[] for _ in range(effective_world_size)]
            recvs_by_peer = [[] for _ in range(effective_world_size)]
            for transfer in transfers:
                if transfer.peer not in global_to_ep_local_rank:
                    raise ValueError(f"Invalid EPLB transfer peer: {transfer}")
                peer = global_to_ep_local_rank[transfer.peer]
                if transfer.tensor.dtype != sample.dtype:
                    raise TypeError(f"All EPLB transfers for one model tensor must use the same dtype: {sample.dtype=} {transfer=}")
                if transfer.tensor.device != sample.device:
                    raise ValueError(f"All EPLB transfers for one model tensor must use the same device: {sample.device=} {transfer=}")
                if transfer.direction == "send":
                    sends_by_peer[peer].append(transfer.tensor.reshape(-1))
                elif transfer.direction == "recv":
                    try:
                        # A receive destination must be a zero-copy flat view.
                        # reshape() is deliberately not used here: for a
                        # non-contiguous tensor it can silently allocate a copy,
                        # causing a successful collective to update the copy
                        # instead of the staging buffer.
                        chunk = transfer.tensor.view(-1)
                    except RuntimeError as exc:
                        raise ValueError(f"EPLB receive buffers must have a zero-copy flat view: {transfer=}") from exc
                    recvs_by_peer[peer].append(chunk)
                else:
                    raise ValueError(f"Invalid EPLB transfer direction: {transfer}")

            input_chunks = [chunk for chunks in sends_by_peer for chunk in chunks]
            input_split_sizes = [sum(chunk.numel() for chunk in chunks) for chunks in sends_by_peer]
            output_split_sizes = [sum(chunk.numel() for chunk in chunks) for chunks in recvs_by_peer]
            input_buffer = torch.cat(input_chunks) if input_chunks else torch.empty(0, dtype=sample.dtype, device=sample.device)
            output_buffer = torch.empty(
                sum(output_split_sizes),
                dtype=sample.dtype,
                device=sample.device,
            )

            work = torch.distributed.all_to_all_single(
                output_buffer,
                input_buffer,
                output_split_sizes=output_split_sizes,
                input_split_sizes=input_split_sizes,
                group=ep_group,
                async_op=True,
            )
            work.wait()

            offset = 0
            for chunks in recvs_by_peer:
                for destination in chunks:
                    next_offset = offset + destination.numel()
                    destination.copy_(output_buffer[offset:next_offset])
                    offset = next_offset
            if offset != output_buffer.numel():
                raise RuntimeError(f"EPLB all-to-all unpack size mismatch: {offset=} output_numel={output_buffer.numel()}")

    def _execute_buffer2weight_copies(buffer2weight_copy_infos):
        for (
            temp_buffers_expert_location,
            routed_experts_weights_expert_location,
        ) in buffer2weight_copy_infos:
            for i in range(num_tensors):
                _get_transfer_tensor(routed_experts_weights, i, routed_experts_weights_expert_location).copy_(
                    _get_tensor(temp_buffers, i, temp_buffers_expert_location)
                )

    def _get_tensor(tensors, tensor_index: int, expert_location: int) -> torch.Tensor:
        if list_depth(tensors) == 1:
            local_loc = _get_local_expert_location(expert_location)
            return tensors[tensor_index].narrow(0, local_loc, 1).squeeze(0)
        elif list_depth(tensors) == 2:
            return tensors[tensor_index][_get_local_expert_location(expert_location)]

    def _get_transfer_tensor(tensors, tensor_index: int, expert_location: int) -> torch.Tensor:
        return get_expert_transfer_view(
            tensors[tensor_index],
            _get_local_expert_location(expert_location),
            num_local_physical_experts,
        )

    def _get_local_expert_location(expert_location: int) -> int:
        assert local_expert_location_range[0] <= expert_location < local_expert_location_range[1]
        return expert_location % num_local_physical_experts

    _entrypoint()

    return output_logs


class _ChunkUtils:
    def __init__(self, *, chunk_values: list, element_values: list):
        self.chunk_values = chunk_values
        self.element_values = element_values

    def chunk_value_from_element_value(self, element_value):
        chunk_index = self._chunk_index_from_element_index(
            num_elements=len(self.element_values),
            num_chunks=len(self.chunk_values),
            element_index=self.element_values.index(element_value),
        )
        return self.chunk_values[chunk_index]

    def element_values_from_chunk_value(self, chunk_value) -> list:
        if len(self.element_values) == 0:
            return []
        element_slice = self._element_slice_from_chunk_index(
            num_elements=len(self.element_values),
            num_chunks=len(self.chunk_values),
            chunk_index=self.chunk_values.index(chunk_value),
        )
        return self.element_values[element_slice]

    @staticmethod
    def _chunk_index_from_element_index(num_elements: int, num_chunks: int, element_index: int) -> int:
        short_chunk_size, num_long_chunks = divmod(num_elements, num_chunks)
        num_elements_for_long_chunks = num_long_chunks * (short_chunk_size + 1)
        if element_index < num_elements_for_long_chunks:
            return element_index // (short_chunk_size + 1)
        else:
            return num_long_chunks + (element_index - num_elements_for_long_chunks) // short_chunk_size

    @staticmethod
    def _element_slice_from_chunk_index(num_elements: int, num_chunks: int, chunk_index: int) -> slice:
        short_chunk_size, num_long_chunks = divmod(num_elements, num_chunks)
        start = chunk_index * short_chunk_size + min(chunk_index, num_long_chunks)
        end = start + short_chunk_size + int(chunk_index < num_long_chunks)
        return slice(start, end)

    def __repr__(self):
        return f"[{self.__class__.__name__}]({self.chunk_values=}, {self.element_values=})"


def _deduplicate_ordered(arr: list[int]):
    output = []
    for item in arr:
        if len(output) == 0 or item != output[-1]:
            output.append(item)
    return output


def _log_p2p_op_metrics(
    p2p_op_infos: list[tuple[int, list[_EplbTransfer]]],
    num_gpu_per_node: int,
    world_size: int,
    self_node_id: int,
    moe_ep_group_ranks: list[int],
):
    text = ""
    all_ops = [op for _, ops in p2p_op_infos for op in ops]
    global_to_ep_local_rank = {global_rank: ep_local_rank for ep_local_rank, global_rank in enumerate(moe_ep_group_ranks)}

    for direction, ops in _group_by(all_ops, _get_direction_from_op).items():
        nbytes_of_gpu = [0] * world_size
        for op in ops:
            nbytes_of_gpu[global_to_ep_local_rank[op.peer]] += op.tensor.nbytes
        nbytes_of_gpu = torch.tensor(nbytes_of_gpu, dtype=torch.int64)

        nbytes_of_node = einops.reduce(
            nbytes_of_gpu,
            "(num_nodes num_gpu_per_node) -> num_nodes",
            num_gpu_per_node=num_gpu_per_node,
            reduction="sum",
        )

        nbytes_curr_node = nbytes_of_node[self_node_id]
        nbytes_cross_node = torch.sum(nbytes_of_node) - nbytes_curr_node

        text += (
            f"{direction}_nbytes_of_gpu={nbytes_of_gpu.tolist()} "
            f"{direction}_nbytes_of_node={nbytes_of_node.tolist()} "
            f"{direction}_nbytes_curr_node={nbytes_curr_node.item()} "
            f"{direction}_nbytes_cross_node={nbytes_cross_node.item()} "
        )

    logger.info(f"[ExpertLocationUpdater] {text}")


def _get_direction_from_op(op: _EplbTransfer):
    if op.direction == "send":
        return "isend"
    if op.direction == "recv":
        return "irecv"
    raise NotImplementedError


def _group_by(items, keyfunc):
    ans = defaultdict(list)
    for item in items:
        ans[keyfunc(item)].append(item)
    return dict(ans)


class EPLBMixin:
    def setup_eplb(self):
        if not self.is_draft_worker:
            set_global_expert_location_metadata(
                compute_initial_expert_location_metadata(self.server_args, self.model_config, moe_ep_rank=self.moe_ep_rank)
            )
            if self.attn_tp_rank == 0 and get_bool_env_var("SGLANG_LOG_EXPERT_LOCATION_METADATA"):
                logger.info(f"Initial expert_location_metadata: {get_global_expert_location_metadata().debug_str()}")

            set_global_expert_distribution_recorder(
                ExpertDistributionRecorder.init_new(
                    self.server_args,
                    get_global_expert_location_metadata(),
                    rank=self.global_rank,
                    ep_rank=self.moe_ep_rank,
                )
            )

        self.eplb_manager = EPLBManager(self) if self.server_args.enable_eplb and (not self.is_draft_worker) else None
        self.expert_location_updater = ExpertLocationUpdater()

    def update_expert_location(
        self,
        new_expert_location_metadata: ExpertLocationMetadata,
        update_layer_ids: list[int],
    ):
        nnodes_per_moe_ep_group = max(self.server_args.nnodes // self.pp_size, 1)
        self.expert_location_updater.update(
            self.model.routed_experts_weights_of_layer,
            self.model.device_local_routed_experts_weights_of_layer,
            new_expert_location_metadata,
            update_layer_ids=update_layer_ids,
            nnodes_per_moe_ep_group=nnodes_per_moe_ep_group,
            moe_ep_rank=self.moe_ep_rank,
            moe_ep_size=self.moe_ep_size,
            moe_ep_group_ranks=get_moe_ep_group().ranks,
        )
