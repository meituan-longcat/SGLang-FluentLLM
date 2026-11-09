"""Optional eager Host writeback for split-BF16 SFA1 target layers."""

import importlib
import os

import torch

from sglang.srt.mem_cache.kv_offload.current_copy import host_attention_ids
from sglang.srt.npu.offload_runtime import host_offload_enabled


class HostBF16Burst:
    max_rows = 128

    def __init__(self, pool, host_ids, operator):
        pairs = [pool.get_kv_buffer(i) for i in sorted(host_ids)]
        self.keys = tuple(k.view(-1, k.shape[-1]) for k, _ in pairs)
        self.ropes = tuple(r.view(-1, r.shape[-1]) for _, r in pairs)
        self.operator = operator
        self._last_indices = None

    def copy(self, source, destination):
        if source.numel() > self.max_rows:
            return False
        src, dst = source.clone().contiguous(), destination.clone().contiguous()
        stream = torch.npu.current_stream(source.device)
        src.record_stream(stream)
        dst.record_stream(stream)
        self._last_indices = (src, dst)
        self.operator(self.keys, src, dst)
        self.operator(self.ropes, src, dst)
        return True


def initialize_bf16_layer_burst(model_runner):
    if os.environ.get("NPU_OFFLOAD_COPY_BURST", "0") != "1":
        return
    if model_runner.is_draft_worker or not host_offload_enabled(model_runner.server_args):
        return
    pool = model_runner.token_to_kv_pool
    if getattr(pool, "dsa_kv_int8_packed", False):
        return
    getter = getattr(pool, "get_buffer_device_types", None)
    kinds = getter() if getter is not None else None
    host_ids = host_attention_ids(kinds, pool.layer_num, hasattr(pool, "get_index_k_with_scale_buffer"))
    module = importlib.import_module("fluent_sfa1_ops")
    receipt = module.validate_layer_copy_burst()
    if (
        getattr(module, "LAYER_COPY_BURST_ABI_VERSION", None) != 1
        or receipt.get("layer_copy_burst_row_bytes_supported") != [672, 1024, 128]
        or receipt.get("layer_copy_burst_bf16_max_rows") != HostBF16Burst.max_rows
    ):
        raise RuntimeError("BF16 Host burst package does not support the split-BF16 layout")
    target = getattr(model_runner.model, "language_model", model_runner.model)
    target._bf16_host_burst = HostBF16Burst(pool, host_ids, module.ops.npu_kv_cache_copy_rows_burst_.default)
