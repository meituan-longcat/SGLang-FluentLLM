from __future__ import annotations

from typing import TYPE_CHECKING

from sglang.srt.disaggregation.base import BaseKVManager, KVArgs
from sglang.srt.disaggregation.utils import (
    DisaggregationMode,
    KVClassType,
    MetadataBuffers,
    TransferBackend,
    get_kv_class,
    is_mla_backend,
)
from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool

if TYPE_CHECKING:
    from sglang.srt.managers.scheduling.scheduler import Scheduler
    from sglang.srt.mem_cache.memory_pool import BaseTokenToKVPool


def get_transfer_buffer_device_types(pool, buffer_count):
    """Read optional registration metadata without changing the buffer API."""
    getter = getattr(pool, "get_buffer_device_types", None)
    device_types = getter() if getter is not None else None
    if device_types is None:
        device_types = [str(pool.device).split(":", 1)[0]] * buffer_count
    return list(device_types)


class KVManagerQueueMixin:
    def __init__(
        self,
        token_to_kv_pool: BaseTokenToKVPool,
        draft_token_to_kv_pool: BaseTokenToKVPool | None,
        metadata_buffers: MetadataBuffers,
        tp_rank: int,
        world_size: int,
        transfer_backend: TransferBackend,
        scheduler: Scheduler,
        disaggregation_mode: DisaggregationMode,
    ):
        self.token_to_kv_pool = token_to_kv_pool
        self.draft_token_to_kv_pool = draft_token_to_kv_pool
        self.is_mla_backend = is_mla_backend(token_to_kv_pool)
        self.draft_is_mla_backend = is_mla_backend(draft_token_to_kv_pool)
        if self.draft_token_to_kv_pool is None:
            self.draft_is_mla_backend = self.is_mla_backend

        self.metadata_buffers = metadata_buffers
        self.tp_rank = tp_rank
        self.tp_size = world_size
        self.transfer_backend = transfer_backend
        self.scheduler = scheduler
        self.disaggregation_mode = disaggregation_mode
        self.kv_manager = self._init_kv_manager()

    def _init_kv_manager(self) -> BaseKVManager:
        kv_args = KVArgs()
        kv_args.engine_rank = self.tp_rank
        kv_data_ptrs, kv_data_lens, kv_item_lens = self.token_to_kv_pool.get_contiguous_buf_infos()
        kv_buffer_device_types = get_transfer_buffer_device_types(self.token_to_kv_pool, len(kv_data_ptrs))
        # [[layer0buf0, layer0buf1...], [layer1buf0, layer1buf1...], ...]
        offsets = self.token_to_kv_pool.get_layerwise_buf_info_offsets()
        target_layer_num = self.token_to_kv_pool.layer_num
        draft_layer_num = 0 if self.draft_token_to_kv_pool is None else self.draft_token_to_kv_pool.layer_num
        if self.draft_token_to_kv_pool is not None:
            # We should also transfer draft model kv cache. The indices are
            # always shared with a target model.
            draft_kv_data_ptrs, draft_kv_data_lens, draft_kv_item_lens = self.draft_token_to_kv_pool.get_contiguous_buf_infos()
            kv_buffer_device_types += get_transfer_buffer_device_types(self.draft_token_to_kv_pool, len(draft_kv_data_ptrs))
            draft_offsets = self.draft_token_to_kv_pool.get_layerwise_buf_info_offsets(len(kv_data_ptrs))
            kv_data_ptrs += draft_kv_data_ptrs
            kv_data_lens += draft_kv_data_lens
            kv_item_lens += draft_kv_item_lens
            offsets += draft_offsets

        kv_args.kv_data_ptrs = kv_data_ptrs
        kv_args.kv_data_lens = kv_data_lens
        kv_args.kv_item_lens = kv_item_lens
        kv_args.buffer_device_types = kv_buffer_device_types
        kv_args.target_layer_num = target_layer_num
        kv_args.draft_layer_num = draft_layer_num
        kv_args.offsets = offsets

        # Define req -> input ids buffer
        kv_args.aux_data_ptrs, kv_args.aux_data_lens, kv_args.aux_item_lens = self.metadata_buffers.get_buf_infos()
        kv_args.ib_device = self.scheduler.server_args.disaggregation_ib_device
        kv_args.gpu_id = self.scheduler.gpu_id

        if isinstance(self.token_to_kv_pool, HybridLinearKVPool):
            state_data_ptrs, state_data_lens, state_item_lens = self.token_to_kv_pool.get_state_buf_infos()
            kv_args.state_data_ptrs = state_data_ptrs
            kv_args.state_data_lens = state_data_lens
            kv_args.state_item_lens = state_item_lens

            kv_args.state_type = "mamba"
            # Get state dimension info for cross-TP slice transfer
            if hasattr(self.token_to_kv_pool, "get_state_dim_per_tensor"):
                kv_args.state_dim_per_tensor = self.token_to_kv_pool.get_state_dim_per_tensor()
        else:
            kv_args.state_data_ptrs = []
            kv_args.state_data_lens = []
            kv_args.state_item_lens = []
            kv_args.state_type = "none"

        kv_manager_class = get_kv_class(self.transfer_backend, KVClassType.MANAGER)
        kv_manager = kv_manager_class(kv_args, self.disaggregation_mode, self.scheduler.server_args, self.is_mla_backend, self.draft_is_mla_backend)
        return kv_manager
