from __future__ import annotations

import os

import numpy as np
import torch

try:
    import torch_npu  # 模块级: 函数内 import 会让 torch.compile graph break
except ImportError:  # 非 NPU 环境无此包, 相关路径不会被走到
    torch_npu = None

from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE
from sglang.srt.layers.attention.radix_attention import RadixAttention
from sglang.srt.mem_cache.memory_pool.memory_pool_base import BaseTokenToKVPool
from sglang.srt.mem_cache.memory_pool.memory_pool_kernel import (
    _KV_COPY_NUM_WARPS_LARGE_TILE,
    _KV_COPY_NUM_WARPS_SMALL_TILE,
    _KV_COPY_STRIDE_THRESHOLD_LARGE,
    _KV_COPY_STRIDE_THRESHOLD_MEDIUM,
    _KV_COPY_TILE_SIZE_LARGE,
    _KV_COPY_TILE_SIZE_MEDIUM,
    _KV_COPY_TILE_SIZE_SMALL,
    GB,
    copy_all_layer_kv_cache_tiled,
    move_kv_cache_native,
)
from sglang.srt.utils import (
    debug_timing,
    get_colorful_logger,
    get_page_slices_by_layer_and_page_idx,
)
from sglang.srt.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter

logger = get_colorful_logger(__name__)


class MHATokenToKVPool(BaseTokenToKVPool):
    def __init__(
        self,
        size: int,
        dtype: torch.dtype,
        head_num: int,
        head_dim: int,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
        max_batch_size: int,
        max_context_len: int,
        page_size: int,
        rank: int,
        enable_kv_cache_copy: bool = False,
        enable_alt_stream: bool = True,
        start_layer: int | None = None,
        end_layer: int | None = None,
    ):
        super().__init__(size, dtype, device, max_batch_size, max_context_len, page_size, rank, start_layer=start_layer, end_layer=end_layer)

        self.memory_saver_adapter = TorchMemorySaverAdapter.create(enable=enable_memory_saver)

        self.head_num = head_num
        self.head_dim = head_dim
        self.layer_num = layer_num
        self.page_size_bytes = self.get_page_size_bytes()
        self.create_buffers()

        self.device_module = torch.get_device_module(self.device)
        self.alt_stream = self.device_module.Stream() if torch.cuda.is_available() and enable_alt_stream else None

        if enable_kv_cache_copy:
            self.init_kv_copy_and_warmup()
        else:
            self._kv_copy_config = None

        k_size, v_size = self.get_kv_size_bytes()
        logger.info(f"KV Cache is allocated. K size: {k_size / GB:.2f} GB, V size: {v_size / GB:.2f} GB.")

    def get_page_size_bytes(self):
        return 2 * self.page_size * self.layer_num * self.head_num * self.head_dim * torch._utils._element_size(self.dtype)

    def create_buffers(self):
        with self.memory_saver_adapter.region(tag=GPU_MEMORY_TYPE_KV_CACHE):
            logger.info(f"create_buffers {self.size=}, {self.page_size=}, {self.head_num=}, {self.head_dim=}, {self.layer_num=}")
            self.k_buffer = [
                torch.empty(
                    (self.size + self.page_size, self.head_num, self.head_dim),
                    dtype=self.store_dtype,
                    device=self.device,
                )
                for _ in range(self.layer_num)
            ]
            self.v_buffer = [
                torch.empty(
                    (self.size + self.page_size, self.head_num, self.head_dim),
                    dtype=self.store_dtype,
                    device=self.device,
                )
                for _ in range(self.layer_num)
            ]
            self.k_data_ptrs = torch.tensor([x.data_ptr() for x in self.k_buffer], dtype=torch.uint64, device=self.device)
            self.v_data_ptrs = torch.tensor([x.data_ptr() for x in self.v_buffer], dtype=torch.uint64, device=self.device)
            self.data_ptrs = torch.cat([self.k_data_ptrs, self.v_data_ptrs], dim=0)
            self.data_strides = torch.tensor(
                [np.prod(x.shape[1:]) * x.dtype.itemsize for x in self.k_buffer + self.v_buffer],
                device=self.device,
            )

    def clear_buffers(self):
        del self.k_buffer
        del self.v_buffer
        if hasattr(self, "k_data_ptrs"):
            del self.k_data_ptrs
        if hasattr(self, "v_data_ptrs"):
            del self.v_data_ptrs
        if hasattr(self, "data_ptrs"):
            del self.data_ptrs
        if hasattr(self, "data_strides"):
            del self.data_strides

    def init_kv_copy_and_warmup(self):
        stride_bytes = int(self.data_strides[0].item())
        if stride_bytes >= _KV_COPY_STRIDE_THRESHOLD_LARGE:
            bytes_per_tile = _KV_COPY_TILE_SIZE_LARGE
        elif stride_bytes >= _KV_COPY_STRIDE_THRESHOLD_MEDIUM:
            bytes_per_tile = _KV_COPY_TILE_SIZE_MEDIUM
        else:
            bytes_per_tile = _KV_COPY_TILE_SIZE_SMALL

        self._kv_copy_config = {
            "bytes_per_tile": bytes_per_tile,
            "byte_tiles": (stride_bytes + bytes_per_tile - 1) // bytes_per_tile,
            "num_warps": (_KV_COPY_NUM_WARPS_SMALL_TILE if bytes_per_tile <= _KV_COPY_TILE_SIZE_MEDIUM else _KV_COPY_NUM_WARPS_LARGE_TILE),
        }

        dummy_loc = torch.zeros(1, dtype=torch.int32, device=self.device)
        grid = (self.data_ptrs.numel(), self._kv_copy_config["byte_tiles"])

        copy_all_layer_kv_cache_tiled[grid](
            self.data_ptrs,
            self.data_strides,
            dummy_loc,
            dummy_loc,
            1,
            1,
            BYTES_PER_TILE=self._kv_copy_config["bytes_per_tile"],
            num_warps=self._kv_copy_config["num_warps"],
            num_stages=2,
        )

    def move_kv_cache(self, tgt_loc: torch.Tensor, src_loc: torch.Tensor):
        if self._kv_copy_config is None:
            move_kv_cache_native(self.k_buffer, self.v_buffer, tgt_loc, src_loc)
        else:
            grid = (self.data_ptrs.numel(), self._kv_copy_config["byte_tiles"])
            copy_all_layer_kv_cache_tiled[grid](
                self.data_ptrs,
                self.data_strides,
                tgt_loc,
                src_loc,
                tgt_loc.numel(),
                tgt_loc.numel(),
                BYTES_PER_TILE=self._kv_copy_config["bytes_per_tile"],
                num_warps=self._kv_copy_config["num_warps"],
                num_stages=2,
            )

    def get_kv_size_bytes(self):
        assert hasattr(self, "k_buffer")
        assert hasattr(self, "v_buffer")
        k_size_bytes = 0
        for k_cache in self.k_buffer:
            k_size_bytes += np.prod(k_cache.shape) * k_cache.dtype.itemsize
        v_size_bytes = 0
        for v_cache in self.v_buffer:
            v_size_bytes += np.prod(v_cache.shape) * v_cache.dtype.itemsize
        return k_size_bytes, v_size_bytes

    def get_contiguous_buf_infos(self):
        kv_data_ptrs = [self._get_key_buffer(i).data_ptr() for i in range(self.layer_num)] + [
            self._get_value_buffer(i).data_ptr() for i in range(self.layer_num)
        ]
        kv_data_lens = [self._get_key_buffer(i).nbytes for i in range(self.layer_num)] + [
            self._get_value_buffer(i).nbytes for i in range(self.layer_num)
        ]
        kv_item_lens = [self._get_key_buffer(i)[0].nbytes * self.page_size for i in range(self.layer_num)] + [
            self._get_value_buffer(i)[0].nbytes * self.page_size for i in range(self.layer_num)
        ]
        return kv_data_ptrs, kv_data_lens, kv_item_lens

    def get_layerwise_buf_info_offsets(self, start_idx=0):
        return [[start_idx + i * self.layer_num + layer_id for i in range(2)] for layer_id in range(self.layer_num)]

    def get_cpu_copy(self, indices):
        torch.cuda.synchronize()
        kv_cache_cpu = []
        for layer_id in range(self.layer_num):
            kv_cache_cpu.append([])
            for i in range(0, len(indices), self.offload_chunk_page_num):
                chunk_indices = indices[i : i + self.offload_chunk_page_num]
                k_cpu = self.k_buffer[layer_id][chunk_indices].to("cpu", non_blocking=True)
                v_cpu = self.v_buffer[layer_id][chunk_indices].to("cpu", non_blocking=True)
                kv_cache_cpu[-1].append([k_cpu, v_cpu])
        torch.cuda.synchronize()
        return kv_cache_cpu

    def load_cpu_copy(self, kv_cache_cpu, indices, prefix_page_count: int = 0):
        del prefix_page_count
        torch.cuda.synchronize()
        for layer_id in range(self.layer_num):
            for i in range(0, len(indices), self.offload_chunk_page_num):
                chunk_indices = indices[i : i + self.offload_chunk_page_num]
                k_cpu, v_cpu = (
                    kv_cache_cpu[layer_id][i // self.offload_chunk_page_num][0],
                    kv_cache_cpu[layer_id][i // self.offload_chunk_page_num][1],
                )
                assert k_cpu.shape[0] == v_cpu.shape[0] == len(chunk_indices)
                k_chunk = k_cpu.to(self.k_buffer[0].device, non_blocking=True)
                v_chunk = v_cpu.to(self.v_buffer[0].device, non_blocking=True)
                self.k_buffer[layer_id][chunk_indices] = k_chunk
                self.v_buffer[layer_id][chunk_indices] = v_chunk
        torch.cuda.synchronize()

    def get_flat_data(self, indices):
        flatten = torch.stack(
            [
                torch.stack([self.k_buffer[i][indices] for i in range(self.layer_num)]),
                torch.stack([self.v_buffer[i][indices] for i in range(self.layer_num)]),
            ]
        )
        return flatten

    @debug_timing
    def transfer(self, indices, flat_data):
        flat_data = flat_data.to(device=self.device, non_blocking=False)
        k_data, v_data = flat_data[0], flat_data[1]
        for i in range(self.layer_num):
            self.k_buffer[i][indices] = k_data[i]
            self.v_buffer[i][indices] = v_data[i]

    def _get_key_buffer(self, layer_id: int):
        if self.store_dtype != self.dtype:
            return self.k_buffer[layer_id].view(self.dtype)
        return self.k_buffer[layer_id]

    def get_key_buffer(self, layer_id: int):
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id)
        return self._get_key_buffer(layer_id)

    def _get_value_buffer(self, layer_id: int):
        if self.store_dtype != self.dtype:
            return self.v_buffer[layer_id].view(self.dtype)
        return self.v_buffer[layer_id]

    def get_value_buffer(self, layer_id: int):
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id)
        return self._get_value_buffer(layer_id)

    def get_kv_buffer(self, layer_id: int):
        return self.get_key_buffer(layer_id), self.get_value_buffer(layer_id)

    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        k_scale: float | None = None,
        v_scale: float | None = None,
        layer_id_override: int = None,
    ):
        layer_id = layer_id_override if layer_id_override is not None else layer.layer_id
        if cache_k.dtype != self.dtype:
            if k_scale is not None:
                cache_k.div_(k_scale)
            if v_scale is not None:
                cache_v.div_(v_scale)
            cache_k = cache_k.to(self.dtype)
            cache_v = cache_v.to(self.dtype)
        if self.store_dtype != self.dtype:
            self.k_buffer[layer_id][loc] = cache_k.view(self.store_dtype)
            self.v_buffer[layer_id][loc] = cache_v.view(self.store_dtype)
        else:
            self.k_buffer[layer_id][loc] = cache_k
            self.v_buffer[layer_id][loc] = cache_v


class NPUMHATokenToKVPool(MHATokenToKVPool):
    # npu_scatter_pa_kv_cache 在 ge 图内会被丢弃, 须用 npu_scatter_nd_update_。回退: NPU_KV_SCATTER_PA=1。
    _USE_SCATTER_PA = os.environ.get("NPU_KV_SCATTER_PA") == "1"

    def __init__(self, *args, use_nd_layout: bool = False, **kwargs):
        self.use_nd_layout = use_nd_layout
        super().__init__(*args, **kwargs)

    @staticmethod
    def _scatter_nd_write(buf: torch.Tensor, val: torch.Tensor, loc: torch.Tensor) -> None:
        """按 slot 写 NZ buffer; ND channel c -> buf[c//16, off, c%16]。"""
        page_size = buf.shape[2]
        hd16 = buf.shape[1]
        n = loc.numel()
        page_idx = torch.div(loc, page_size, rounding_mode="floor")
        off_idx = loc % page_size
        h_off = torch.arange(hd16, device=loc.device, dtype=loc.dtype) * page_size
        row_idx = (page_idx.unsqueeze(1) * (hd16 * page_size) + h_off.unsqueeze(0) + off_idx.unsqueeze(1)).reshape(-1, 1)
        torch_npu.npu_scatter_nd_update_(buf.view(-1, 16), row_idx, val.reshape(n, hd16, 16).reshape(-1, 16))

    @staticmethod
    def _scatter_nd_write_nd(buf: torch.Tensor, val: torch.Tensor, loc: torch.Tensor) -> None:
        """按 slot 写 ND buffer, 展平行号 = page*page_size + off。"""
        page_size = buf.shape[1]
        n = loc.numel()
        page_idx = torch.div(loc, page_size, rounding_mode="floor")
        row_idx = (page_idx * page_size + loc % page_size).reshape(-1, 1)
        torch_npu.npu_scatter_nd_update_(buf.view(-1, buf.shape[-1]), row_idx, val.reshape(n, -1))

    def create_buffers(self):
        with self.memory_saver_adapter.region(tag=GPU_MEMORY_TYPE_KV_CACHE):
            logger.info(
                f"NPUMHA create_buffers {self.size=}, {self.page_size=}, {self.head_num=}, "
                f"{self.head_dim=}, {self.layer_num=}, nd_layout={self.use_nd_layout}"
            )
            page_num = self.size // self.page_size + 1
            if self.use_nd_layout:
                dims = (page_num, self.page_size, self.head_num * self.head_dim)
            else:
                dims = (page_num, self.head_num * self.head_dim // 16, self.page_size, 16)

            self.k_buffer = [torch.empty(dims, dtype=self.store_dtype, device=self.device) for _ in range(self.layer_num)]
            self.v_buffer = [torch.empty(dims, dtype=self.store_dtype, device=self.device) for _ in range(self.layer_num)]
            # HiCache D2H/H2D 按 page 逐片 _foreach_copy_, 预切分避免每次传输重复建 view。
            self.k_buffer_page_slices = get_page_slices_by_layer_and_page_idx(self.k_buffer)
            self.v_buffer_page_slices = get_page_slices_by_layer_and_page_idx(self.v_buffer)

    def clear_buffers(self):
        del self.k_buffer
        del self.v_buffer
        self.k_buffer_page_slices = None
        self.v_buffer_page_slices = None

    def move_kv_cache(self, tgt_loc: torch.Tensor, src_loc: torch.Tensor):
        raise NotImplementedError("NPU KV buffer内存布局和GPU不同，暂未实现HiCache")

    def get_contiguous_buf_infos(self):
        kv_data_ptrs = [self.get_key_buffer(i).data_ptr() for i in range(self.layer_num)] + [
            self.get_value_buffer(i).data_ptr() for i in range(self.layer_num)
        ]
        kv_data_lens = [self.get_key_buffer(i).nbytes for i in range(self.layer_num)] + [
            self.get_value_buffer(i).nbytes for i in range(self.layer_num)
        ]
        kv_item_lens = [self.get_key_buffer(i)[0].nbytes for i in range(self.layer_num)] + [
            self.get_value_buffer(i)[0].nbytes for i in range(self.layer_num)
        ]
        return kv_data_ptrs, kv_data_lens, kv_item_lens

    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        k_scale: float | None = None,
        v_scale: float | None = None,
    ):
        layer_id = layer.layer_id
        if cache_k.dtype != self.dtype:
            if k_scale is not None:
                cache_k.div_(k_scale)
            if v_scale is not None:
                cache_v.div_(v_scale)
            cache_k = cache_k.to(self.dtype)
            cache_v = cache_v.to(self.dtype)

        if self.store_dtype != self.dtype:
            cache_k = cache_k.view(self.store_dtype)
            cache_v = cache_v.view(self.store_dtype)

        if self._USE_SCATTER_PA:
            torch_npu.npu_scatter_pa_kv_cache(
                key=cache_k,
                value=cache_v,
                key_cache=self.k_buffer[layer_id],
                value_cache=self.v_buffer[layer_id],
                slot_mapping=loc,
            )
        elif self.use_nd_layout:
            self._scatter_nd_write_nd(self.k_buffer[layer_id], cache_k, loc)
            self._scatter_nd_write_nd(self.v_buffer[layer_id], cache_v, loc)
        else:
            self._scatter_nd_write(self.k_buffer[layer_id], cache_k, loc)
            self._scatter_nd_write(self.v_buffer[layer_id], cache_v, loc)
