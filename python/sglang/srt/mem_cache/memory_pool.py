"""
Copyright 2023-2024 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
from __future__ import annotations

from dataclasses import dataclass

from contextlib import contextmanager
from sglang.srt.torch_memory_saver_adapter import TorchMemorySaverAdapter

"""
Memory pool.

SGLang has two levels of memory pool.
ReqToTokenPool maps a a request to its token locations.
BaseTokenToKVPool maps a token location to its KV cache data.
"""

import threading
from enum import IntEnum
from functools import wraps
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

import numpy as np
import psutil
import torch
import triton
import triton.language as tl

from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.utils import is_npu, is_sm90_supported
if not is_npu():
    if is_sm90_supported():
        from flash_mla_fp8 import quantize_and_cache_k, dequantize_ckv_fused_indexed
from sglang.srt.utils import debug_timing, get_colorful_logger, get_compiler_backend

if TYPE_CHECKING:
    from sglang.srt.managers.cache_controller import LayerDoneCounter
    from sglang.srt.managers.req import Req

logger = get_colorful_logger(__name__)

GB = 1024 * 1024 * 1024

def get_tensor_size_bytes(t: Union[torch.Tensor, List[torch.Tensor], tuple]):
    if isinstance(t, (list, tuple)):
        return sum(get_tensor_size_bytes(x) for x in t)
    return np.prod(t.shape) * t.dtype.itemsize

@dataclass
class ReqToTokenPoolInfo:
    """For chunked prefill"""

    verified_len: int
    alloced_len: int
    alloced_slots: torch.Tensor


class ReqToTokenPool:
    """A memory pool that maps a request to its token locations."""

    def __init__(
        self,
        size: int,
        max_context_len: int,
        device: str,
        enable_memory_saver: bool,
    ):
        memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=enable_memory_saver
        )

        self.size = size
        self.max_context_len = max_context_len
        self.device = device
        with memory_saver_adapter.region():
            self.req_to_token = torch.zeros(
                (size, max_context_len), dtype=torch.int32, device=device
            )
            # verified_lens records the valid historical KV cache length for each request,
            # mainly used to determine the KV cache position to write for this computation
            self.verified_lens = torch.zeros(size, dtype=torch.int32, device=device)
            # alloced_lens records the allocated KV cache length for each request,
            # which can be larger than verified_lens, mainly used to determine the KV cache position for this allocation
            self.alloced_lens = torch.zeros(size, dtype=torch.int32, device=device)
        self.free_slots = list(range(size))[1:]

    def get_req_pool_info(self, req_pool_idx: int):
        alloced_len = self.alloced_lens[req_pool_idx].item()
        return ReqToTokenPoolInfo(
            self.verified_lens[req_pool_idx].item(),
            alloced_len,
            self.req_to_token[req_pool_idx, :alloced_len].clone(),
        )

    def set_req_pool_info(self, req_pool_idx: int, metadata: ReqToTokenPoolInfo):
        self.verified_lens[req_pool_idx] = metadata.verified_len
        self.alloced_lens[req_pool_idx] = metadata.alloced_len
        self.req_to_token[req_pool_idx, : metadata.alloced_len] = metadata.alloced_slots

    def write(self, indices, values):
        self.req_to_token[indices] = values

    def available_size(self):
        return len(self.free_slots)

    def alloc(self, need_size: int) -> List[int]:
        if need_size > len(self.free_slots):
            return None

        select_index = self.free_slots[:need_size]
        self.free_slots = self.free_slots[need_size:]

        # During overlap scheduling, after a retracted request frees its req_pool,
        # the forward_thread may still modify its verified_lens, causing errors when
        # reusing this position. Here we ensure that when req_idx is reused, the corresponding resource is empty.
        self.verified_lens[select_index] = 0
        self.alloced_lens[select_index] = 0

        return select_index

    def free(self, free_index: Union[int, List[int]]):
        if isinstance(free_index, (int,)):
            self.free_slots.append(free_index)
            self.verified_lens[free_index] = 0
            self.alloced_lens[free_index] = 0
        else:
            self.free_slots.extend(free_index)
            for index in free_index:
                self.verified_lens[index] = 0
                self.alloced_lens[index] = 0

    def clear(self):
        # clear method is called during flush_cache
        # slot 0 is used as padding in spec_cuda_graph and is not allocated externally
        self.free_slots = list(range(self.size))[1:]
        self.verified_lens.zero_()
        self.alloced_lens.zero_()


class BaseTokenToKVPool:
    """A memory pool that maps a token location to its kv cache data."""

    def __init__(
        self,
        size: int,
        dtype: torch.dtype,
        device: str,
        max_batch_size: int,
        max_context_len: int,
        page_size: int,
        rank: int,
    ):
        self.dtype = dtype
        self.rank = rank
        self.size = size
        self.page_size = page_size
        if dtype in (torch.float8_e5m2, torch.float8_e4m3fn):
            # NOTE: Store as torch.uint8 because Tensor.index_put is not implemented for torch.float8_e5m2
            self.store_dtype = torch.uint8
        else:
            self.store_dtype = dtype
        self.device = device
        self.offload_chunk_page_num = 1024
        self.token_slot_refs = None
        
        # default state for optional layer-wise transfer control
        self.layer_transfer_counter = None
        logger.info(f"Initialized token to kv pool with size {size}, dtype {dtype}, device {device}, page size {page_size}, rank {rank}")

    def register_layer_transfer_counter(self, layer_transfer_counter: LayerDoneCounter):
        self.layer_transfer_counter = layer_transfer_counter

    def set_token_slot_refs(self, token_slot_refs: torch.Tensor):
        self.token_slot_refs = token_slot_refs

    def get_key_buffer(self, layer_id: int) -> torch.Tensor:
        raise NotImplementedError()

    def get_value_buffer(self, layer_id: int) -> torch.Tensor:
        raise NotImplementedError()

    def get_kv_buffer(self, layer_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError()

    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
    ) -> None:
        raise NotImplementedError()

    def get_cpu_copy(self, page_indices: list[int]) -> torch.Tensor:
        raise NotImplementedError()

    def load_cpu_copy(
        self, kv_cache_cpu: torch.Tensor, page_indices: list[int]
    ) -> None:
        raise NotImplementedError()
    
    # for pd disaggregation
    def get_contiguous_buf_infos(self):
        raise NotImplementedError()

    # for pd disaggregation
    def get_layerwise_buf_info_offsets(self, start_idx=0):
        raise NotImplementedError()


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
    ):
        super().__init__(
            size, dtype, device, max_batch_size, max_context_len, page_size, rank
        )

        self.memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=enable_memory_saver
        )

        self.head_num = head_num
        self.head_dim = head_dim
        self.layer_num = layer_num
        self.page_size_bytes = self._get_page_size_bytes()
        self._create_buffers()

        self.device_module = torch.get_device_module(self.device)
        self.alt_stream = (
            self.device_module.Stream() if torch.cuda.is_available() and enable_alt_stream else None
        )

        if enable_kv_cache_copy:
            self._init_kv_copy_and_warmup()
        else:
            self._kv_copy_config = None

        k_size, v_size = self.get_kv_size_bytes()
        logger.info(
            f"KV Cache is allocated. K size: {k_size / GB:.2f} GB, V size: {v_size / GB:.2f} GB."
        )

    def _get_page_size_bytes(self):
        return 2 * self.page_size * self.layer_num * self.head_num * self.head_dim * torch._utils._element_size(self.dtype)

    def _create_buffers(self):
        with self.memory_saver_adapter.region():
            # [size, head_num, head_dim] for each layer
            # The padded page 0 is used for writing dummy outputs from padded tokens.
            logger.info(f"_create_buffers {self.size=}, {self.page_size=}, {self.head_num=}, {self.head_dim=}, {self.layer_num=}")
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
                [
                    np.prod(x.shape[1:]) * x.dtype.itemsize
                    for x in self.k_buffer + self.v_buffer
                ],
                device=self.device,
            )

    def _clear_buffers(self):
        del self.k_buffer
        del self.v_buffer
        if hasattr(self, 'k_data_ptrs'):
            del self.k_data_ptrs
        if hasattr(self, 'v_data_ptrs'):
            del self.v_data_ptrs
        if hasattr(self, 'data_ptrs'):
            del self.data_ptrs
        if hasattr(self, 'data_strides'):
            del self.data_strides

        _KV_COPY_STRIDE_THRESHOLD_LARGE = 8192
        _KV_COPY_STRIDE_THRESHOLD_MEDIUM = 4096
        _KV_COPY_TILE_SIZE_LARGE = 512
        _KV_COPY_TILE_SIZE_MEDIUM = 256
        _KV_COPY_TILE_SIZE_SMALL = 128
        _KV_COPY_NUM_WARPS_LARGE_TILE = 8
        _KV_COPY_NUM_WARPS_SMALL_TILE = 4

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
            "num_warps": (
                _KV_COPY_NUM_WARPS_SMALL_TILE
                if bytes_per_tile <= _KV_COPY_TILE_SIZE_MEDIUM
                else _KV_COPY_NUM_WARPS_LARGE_TILE
            ),
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

    # for disagg
    def get_contiguous_buf_infos(self):
        # layer_num x [seq_len, head_num, head_dim]
        # layer_num x [page_num, page_size, head_num, head_dim]
        kv_data_ptrs = [
            self._get_key_buffer(i).data_ptr() for i in range(self.layer_num)
        ] + [self._get_value_buffer(i).data_ptr() for i in range(self.layer_num)]
        kv_data_lens = [
            self._get_key_buffer(i).nbytes for i in range(self.layer_num)
        ] + [self._get_value_buffer(i).nbytes for i in range(self.layer_num)]
        kv_item_lens = [
            self._get_key_buffer(i)[0].nbytes * self.page_size
            for i in range(self.layer_num)
        ] + [
            self._get_value_buffer(i)[0].nbytes * self.page_size
            for i in range(self.layer_num)
        ]
        return kv_data_ptrs, kv_data_lens, kv_item_lens

    def get_layerwise_buf_info_offsets(self, start_idx=0):
        return [[start_idx + i*self.layer_num + layer_id for i in range(2)] for layer_id in range(self.layer_num)]

    def get_cpu_copy(self, indices):
        torch.cuda.synchronize()
        kv_cache_cpu = []
        for layer_id in range(self.layer_num):
            kv_cache_cpu.append([])
            for i in range(0, len(indices), self.offload_chunk_page_num):
                chunk_indices = indices[i : i + self.offload_chunk_page_num]
                k_cpu = self.k_buffer[layer_id][chunk_indices].to(
                    "cpu", non_blocking=True
                )
                v_cpu = self.v_buffer[layer_id][chunk_indices].to(
                    "cpu", non_blocking=True
                )
                kv_cache_cpu[-1].append([k_cpu, v_cpu])
        torch.cuda.synchronize()
        return kv_cache_cpu

    def load_cpu_copy(self, kv_cache_cpu, indices):
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

    # Todo: different memory layout
    def get_flat_data(self, indices):
        # prepare a large chunk of contiguous data for efficient transfer
        flatten = torch.stack(
            [
                torch.stack([self.k_buffer[i][indices] for i in range(self.layer_num)]),
                torch.stack([self.v_buffer[i][indices] for i in range(self.layer_num)]),
            ]
        )
        return flatten

    @debug_timing
    def transfer(self, indices, flat_data):
        # transfer prepared data from host to device
        flat_data = flat_data.to(device=self.device, non_blocking=False)
        k_data, v_data = flat_data[0], flat_data[1]
        for i in range(self.layer_num):
            self.k_buffer[i][indices] = k_data[i]
            self.v_buffer[i][indices] = v_data[i]

    def _get_key_buffer(self, layer_id: int):
        # for internal use of referencing
        if self.store_dtype != self.dtype:
            return self.k_buffer[layer_id].view(self.dtype)
        return self.k_buffer[layer_id]

    def get_key_buffer(self, layer_id: int):
        # note: get_key_buffer is hooked with synchronization for layer-wise KV cache loading
        # it is supposed to be used only by attention backend not for information purpose
        # same applies to get_value_buffer and get_kv_buffer
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id)
        return self._get_key_buffer(layer_id)

    def _get_value_buffer(self, layer_id: int):
        # for internal use of referencing
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
        k_scale: Optional[float] = None,
        v_scale: Optional[float] = None,
        layer_id_override: int = None
    ):
        if layer_id_override is not None:
            layer_id = layer_id_override
        else:
            layer_id = layer.layer_id
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


# This compiled version is slower in the unit test
# python3 -m unittest test_bench_serving.TestBenchServing.test_offline_throughput_non_stream_small_batch_size
@torch.compile(dynamic=True, backend=get_compiler_backend())
def copy_two_array(loc, dst_1, src_1, dst_2, src_2, dtype, store_dtype):
    dst_1[loc] = src_1.to(dtype).view(store_dtype)
    dst_2[loc] = src_2.to(dtype).view(store_dtype)


@triton.jit
def set_mla_kv_buffer_kernel(
    kv_buffer_ptr,
    cache_k_nope_ptr,
    cache_k_rope_ptr,
    loc_ptr,
    buffer_stride: tl.constexpr,
    nope_stride: tl.constexpr,
    rope_stride: tl.constexpr,
    nope_dim: tl.constexpr,
    rope_dim: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid_loc = tl.program_id(0)
    pid_blk = tl.program_id(1)

    base = pid_blk * BLOCK
    offs = base + tl.arange(0, BLOCK)
    total_dim = nope_dim + rope_dim
    mask = offs < total_dim

    loc = tl.load(loc_ptr + pid_loc)
    dst_ptr = kv_buffer_ptr + loc * buffer_stride + offs

    if base + BLOCK <= nope_dim:
        src = tl.load(
            cache_k_nope_ptr + pid_loc * nope_stride + offs,
            mask=mask,
        )
    else:
        offs_rope = offs - nope_dim
        src = tl.load(
            cache_k_rope_ptr + pid_loc * rope_stride + offs_rope,
            mask=mask,
        )

    tl.store(dst_ptr, src, mask=mask)


def set_mla_kv_buffer_triton(
    kv_buffer: torch.Tensor,
    loc: torch.Tensor,
    cache_k_nope: torch.Tensor,
    cache_k_rope: torch.Tensor,
):
    nope_dim = cache_k_nope.shape[-1]
    rope_dim = cache_k_rope.shape[-1]
    total_dim = nope_dim + rope_dim
    BLOCK = 128
    n_loc = loc.numel()
    grid = (n_loc, triton.cdiv(total_dim, BLOCK))

    set_mla_kv_buffer_kernel[grid](
        kv_buffer,
        cache_k_nope,
        cache_k_rope,
        loc,
        kv_buffer.stride(0),
        cache_k_nope.stride(0),
        cache_k_rope.stride(0),
        nope_dim,
        rope_dim,
        BLOCK=BLOCK,
    )

from sglang.srt.layers.attention.dsa.quant_k_cache import quantize_k_cache
class MLATokenToKVPool(BaseTokenToKVPool):
    def __init__(
        self,
        size: int,
        model_dtype: torch.dtype,
        dtype: torch.dtype,
        quant_method: str,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
        max_batch_size: int,
        max_context_len: int,
        page_size: int,
        rank: int,
        use_dsa: bool = False,
        enable_kv_cache_copy: bool = False,
        enable_alt_stream: bool = True,
    ):
        super().__init__(
            size, dtype, device, max_batch_size, max_context_len, page_size, rank
        )
        self.model_dtype = model_dtype
        self.quant_method = quant_method

        self.kv_lora_rank = kv_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.layer_num = layer_num
        self.use_dsa = use_dsa
        self.nsa_kv_cache_store_fp8 = use_dsa and dtype == torch.float8_e4m3fn
        self.kv_cache_dim = (
            656
            if self.use_dsa and self.nsa_kv_cache_store_fp8
            else (kv_lora_rank + qk_rope_head_dim)
        )

        self.memory_saver_adapter = memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=enable_memory_saver
        )
        self.page_size_bytes = self._get_page_size_bytes()

        with memory_saver_adapter.region():
            # The padded page 0 is used for writing dummy outputs from padded tokens.
            if self.quant_method == "per_token_head":
                self.kv_buffer = [
                    (
                        torch.empty((self.size + self.page_size, 1, kv_lora_rank),
                                     dtype=self.store_dtype,
                                     device=device),
                        torch.empty((self.size + self.page_size, 1, 1),
                                     dtype=torch.float32,
                                     device=device),
                        torch.empty((self.size + self.page_size, 1, qk_rope_head_dim),
                                     dtype=self.model_dtype,
                                     device=device),
                    )
                    for _ in range(layer_num)
                ]
            else:
                self.kv_buffer = [
                    torch.empty(
                        (self.size + self.page_size, 1, self.kv_cache_dim),
                        dtype=self.store_dtype,
                        device=device,
                    )
                    for _ in range(layer_num)
                ]
        
        # Calculate data pointers and strides for all buffers
        all_buffers = []
        if self.quant_method == "per_token_head":
            # kv_buffer is a list of tuples (k_lora_cache, k_scale_cache, k_rope_cache)
            for layer_buffers in self.kv_buffer:
                # Each layer has 3 tensors
                all_buffers.extend(layer_buffers)
        else:
            # kv_buffer is a list of single tensors
            all_buffers = self.kv_buffer
        
        self.data_ptrs = torch.tensor(
            [buf.data_ptr() for buf in all_buffers],
            dtype=torch.uint64,
            device=self.device,
        )
        self.data_strides = torch.tensor(
            [
                np.prod(buf.shape[1:]) * buf.dtype.itemsize
                for buf in all_buffers
            ],
            device=self.device,
        )
        
        self.device_module = torch.get_device_module(self.device)
        self.alt_stream = (
            self.device_module.Stream() if torch.cuda.is_available() and enable_alt_stream else None
        )

        if enable_kv_cache_copy:
            self._init_kv_copy_and_warmup()
        else:
            self._kv_copy_config = None

        kv_size = sum(get_tensor_size_bytes(buf) for buf in all_buffers)
        
        logger.info(f"KV Cache is allocated. KV size: {kv_size / GB:.2f} GB.")
        if self.quant_method == "per_token_head":
            # kv_buffer contains tuples of 3 tensors
            first_buffer = self.kv_buffer[0]
            device_info = first_buffer[0].device
            shape_info = f"({first_buffer[0].shape}, {first_buffer[1].shape}, {first_buffer[2].shape})"
        else:
            # kv_buffer contains single tensors
            device_info = self.kv_buffer[-1].device
            shape_info = self.kv_buffer[0].shape
        logger.info(f"MLATokenToKVPool: {len(self.kv_buffer)=} device={device_info} shape={shape_info} {self.size=} {self.page_size=} {self.kv_cache_dim=} {self.store_dtype=}")

    def _get_page_size_bytes(self):
        if self.quant_method ==  "per_token_head":
            dim_size_bytes = self.kv_lora_rank * torch._utils._element_size(self.dtype) + \
                self.qk_rope_head_dim * torch._utils._element_size(self.model_dtype) + \
                1 * torch._utils._element_size(torch.float32)
        else:
            dim_size_bytes = (self.kv_lora_rank + self.qk_rope_head_dim) * torch._utils._element_size(self.dtype)
        return self.page_size * self.layer_num * dim_size_bytes

    def _init_kv_copy_and_warmup(self):
        # Heuristics for KV copy tiling
        _KV_COPY_STRIDE_THRESHOLD_LARGE = 8192
        _KV_COPY_STRIDE_THRESHOLD_MEDIUM = 4096
        _KV_COPY_TILE_SIZE_LARGE = 512
        _KV_COPY_TILE_SIZE_MEDIUM = 256
        _KV_COPY_TILE_SIZE_SMALL = 128
        _KV_COPY_NUM_WARPS_LARGE_TILE = 8
        _KV_COPY_NUM_WARPS_SMALL_TILE = 4

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
            "num_warps": (
                _KV_COPY_NUM_WARPS_SMALL_TILE
                if bytes_per_tile <= _KV_COPY_TILE_SIZE_MEDIUM
                else _KV_COPY_NUM_WARPS_LARGE_TILE
            ),
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
            # Native implementation for MLA
            if tgt_loc.numel() == 0:
                return
            
            tgt_loc_flat = tgt_loc.view(-1).long()
            src_loc_flat = src_loc.view(-1).long()
            
            if self.quant_method == "per_token_head":
                # kv_buffer is a list of tuples
                for layer_buffers in self.kv_buffer:
                    # Each layer has 3 tensors: k_lora_cache, k_scale_cache, k_rope_cache
                    for buf in layer_buffers:
                        buf[tgt_loc_flat] = buf[src_loc_flat]
            else:
                # kv_buffer is a list of single tensors
                for buf in self.kv_buffer:
                    buf[tgt_loc_flat] = buf[src_loc_flat]
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
        assert hasattr(self, "kv_buffer")
        kv_size_bytes = 0
        for kv_cache in self.kv_buffer:
            kv_size_bytes += get_tensor_size_bytes(kv_cache)
        return kv_size_bytes
    
    # for disagg
    def get_contiguous_buf_infos(self):
        if self.quant_method ==  "per_token_head":
            kv_data_ptrs = [sub_tuple[i].data_ptr() for i in range(3) for sub_tuple in self.kv_buffer]
            kv_data_lens = [sub_tuple[i].nbytes for i in range(3) for sub_tuple in self.kv_buffer]
            kv_item_lens = [sub_tuple[i][0].nbytes*self.page_size for i in range(3) for sub_tuple in self.kv_buffer]
        else:
            # MLA has only one kv_buffer, so only the information of this buffer needs to be returned.
            kv_data_ptrs = [self.kv_buffer[i].data_ptr() for i in range(self.layer_num)]
            kv_data_lens = [self.kv_buffer[i].nbytes for i in range(self.layer_num)]
            kv_item_lens = [
                self.kv_buffer[i][0].nbytes * self.page_size for i in range(self.layer_num)
            ]
        return kv_data_ptrs, kv_data_lens, kv_item_lens

    def get_layerwise_buf_info_offsets(self, start_idx=0):
        if self.quant_method == "per_token_head":
            return [[start_idx + i*self.layer_num + layer_id for i in range(3)] for layer_id in range(self.layer_num)]
        else:
            return [[start_idx + layer_id] for layer_id in range(self.layer_num)]

    def get_key_buffer(self, layer_id: int):
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id)
        if self.quant_method ==  "per_token_head":
            return self.kv_buffer[layer_id]
        elif self.store_dtype != self.dtype:
            return self.kv_buffer[layer_id].view(self.dtype)
        else:
            return self.kv_buffer[layer_id]

    def get_key_split_contiguous(self, layer_id: int, indices: torch.Tensor):
        if self.quant_method ==  "per_token_head":
            k_lora_cache, k_scale_cache, k_rope_cache = self.kv_buffer[layer_id]
            if not is_npu() and is_sm90_supported():
                k_lora_deq, k_rope_deq = dequantize_ckv_fused_indexed(
                    k_lora_cache.view(self.dtype), k_rope_cache, k_scale_cache, indices
                )
                return k_lora_deq, k_rope_deq
            else:
                k_lora = k_lora_cache[indices].view(self.dtype).float()
                k_scale = k_scale_cache[indices]
                k_rope = k_rope_cache[indices].float()
                k_lora_deq = (k_lora * k_scale).to(self.model_dtype).contiguous()
                k_rope_deq = (k_rope * k_scale).to(self.model_dtype).contiguous()
                return k_lora_deq, k_rope_deq
        elif self.store_dtype != self.dtype:
            latent_cache = self.kv_buffer[layer_id].view(self.dtype)[indices].contiguous()
        else:
            latent_cache = self.kv_buffer[layer_id][indices].contiguous()
        kv_a_normed, k_pe = latent_cache.split(
            [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )
        return kv_a_normed, k_pe

    def get_value_buffer(self, layer_id: int):
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id)
        if self.quant_method == "per_token_head":
            return self.kv_buffer[layer_id][:2]
        elif self.store_dtype != self.dtype:
            return self.kv_buffer[layer_id][..., : self.kv_lora_rank].view(self.dtype)
        else:
            return self.kv_buffer[layer_id][..., : self.kv_lora_rank]

    def get_kv_buffer(self, layer_id: int):
        return self.get_key_buffer(layer_id), self.get_value_buffer(layer_id)

    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
    ):
        layer_id = layer.layer_id
        if self.quant_method == "per_token_head":
            if not is_npu() and is_sm90_supported():
                quantize_and_cache_k(
                    key=cache_k.contiguous(),
                    k_lora_cache=self.kv_buffer[layer_id][0],
                    k_lora_scale_cache=self.kv_buffer[layer_id][1],
                    k_rope_cache=self.kv_buffer[layer_id][2],
                    indices=loc.to(torch.int32),
                    head_dim_v=self.kv_lora_rank
                )
            else:
                k_lora = cache_k[..., :self.kv_lora_rank].float()
                k_rope = cache_k[..., self.kv_lora_rank:].float()
                scale = k_lora.abs().amax(dim=-1, keepdim=True).clamp(1e-26) / 448.0
                k_lora = (k_lora / scale).to(torch.float8_e4m3fn)
                k_rope = (k_rope / scale).to(self.model_dtype)
                self.kv_buffer[layer_id][0][loc] = k_lora.view(self.store_dtype)
                self.kv_buffer[layer_id][1][loc] = scale
                self.kv_buffer[layer_id][2][loc] = k_rope
        else:
            self.kv_buffer[layer_id][loc] = cache_k

    def set_mla_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k_nope: torch.Tensor,
        cache_k_rope: torch.Tensor,
    ):
        layer_id = layer.layer_id
        if self.use_dsa and self.nsa_kv_cache_store_fp8:
            # original cache_k: (num_tokens, num_heads 1, hidden 576); we unsqueeze the page_size=1 dim here
            # TODO no need to cat
            cache_k = torch.cat([cache_k_nope, cache_k_rope], dim=-1)
            cache_k = quantize_k_cache(cache_k.unsqueeze(1)).squeeze(1)
            cache_k = cache_k.view(self.store_dtype)
            self.kv_buffer[layer_id][loc] = cache_k
        elif self.quant_method == "per_token_head":
            if not is_npu() and is_sm90_supported():
                cache_k_combined = torch.cat([cache_k_nope, cache_k_rope], dim=-1)
                quantize_and_cache_k(
                    key=cache_k_combined.contiguous(),
                    k_lora_cache=self.kv_buffer[layer_id][0],
                    k_lora_scale_cache=self.kv_buffer[layer_id][1],
                    k_rope_cache=self.kv_buffer[layer_id][2],
                    indices=loc.to(torch.int32),
                    head_dim_v=self.kv_lora_rank
                )
            else:
                k_lora = cache_k_nope.float()
                k_rope = cache_k_rope.float()
                scale = k_lora.abs().amax(dim=-1, keepdim=True).clamp(1e-26) / 448.0
                k_lora = (k_lora / scale).to(torch.float8_e4m3fn)
                k_rope = (k_rope / scale).to(self.model_dtype)
                self.kv_buffer[layer_id][0][loc] = k_lora.view(self.store_dtype)
                self.kv_buffer[layer_id][1][loc] = scale
                self.kv_buffer[layer_id][2][loc] = k_rope
        else:
            if cache_k_nope.dtype != self.dtype:
                cache_k_nope = cache_k_nope.to(self.dtype)
                cache_k_rope = cache_k_rope.to(self.dtype)
            if self.store_dtype != self.dtype:
                cache_k_nope = cache_k_nope.view(self.store_dtype)
                cache_k_rope = cache_k_rope.view(self.store_dtype)

            set_mla_kv_buffer_triton(
                self.kv_buffer[layer_id], loc, cache_k_nope, cache_k_rope
            )

    def get_cpu_copy(self, token_indices: list[int]) -> torch.Tensor:
        torch.cuda.synchronize()
        kv_cache_cpu = []
        for layer_id in range(self.layer_num):
            kv_cache_cpu.append([])
            for i in range(0, len(token_indices), self.offload_chunk_page_num):
                chunk_indices = token_indices[i : i + self.offload_chunk_page_num]
                if self.quant_method == "per_token_head":
                    kv_cache_cpu[-1].append([
                        buffer[chunk_indices].to("cpu", non_blocking=True) for buffer in self.kv_buffer[layer_id]
                    ])
                else:
                    kv_cpu = self.kv_buffer[layer_id][chunk_indices].to(
                        "cpu", non_blocking=True
                    )
                    kv_cache_cpu[-1].append([kv_cpu])
        torch.cuda.synchronize()
        return kv_cache_cpu

    def load_cpu_copy(
        self, kv_cache_cpu: torch.Tensor, token_indices: list[int]
    ) -> None:
        torch.cuda.synchronize()
        for layer_id in range(self.layer_num):
            for i in range(0, len(token_indices), self.offload_chunk_page_num):
                chunk_indices = token_indices[i : i + self.offload_chunk_page_num]
                if self.quant_method == "per_token_head":
                    for j in range(3):
                        t = kv_cache_cpu[layer_id][i // self.offload_chunk_page_num][j]
                        assert t.shape[0] == len(chunk_indices)
                        self.kv_buffer[layer_id][j][chunk_indices] = t.to(self.kv_buffer[0][0].device, non_blocking=True)
                else:
                    kv_cpu = kv_cache_cpu[layer_id][i // self.offload_chunk_page_num][0]
                    assert kv_cpu.shape[0] == len(chunk_indices), f"kv_cpu.shape[0] {kv_cpu.shape[0]} != len(chunk_indices) {len(chunk_indices)}"
                    kv_chunk = kv_cpu.to(self.kv_buffer[0].device, non_blocking=True)
                    self.kv_buffer[layer_id][chunk_indices] = kv_chunk
        torch.cuda.synchronize()


class MemoryStateInt(IntEnum):
    IDLE = 0
    RESERVED = 1
    PROTECTED = 2
    SYNCED = 3
    BACKUP = 4


def synchronized(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        with self.lock:
            return func(self, *args, **kwargs)

    return wrapper


class MLATokenToKVPoolHost:
    def __init__(
        self,
        device_pool: MHATokenToKVPool,
        host_to_device_ratio: float = 4.0,
        pin_memory: bool = False,  # no need to use pin memory with the double buffering
        device: str = "cpu",
    ):
        assert host_to_device_ratio >= 1, (
            "The host memory should be larger than the device memory with the current protocol"
        )
        # todo, other ways of configuring the size

        self.device_pool = device_pool
        self.host_to_device_ratio = host_to_device_ratio
        self.pin_memory = pin_memory
        self.device = device

        self.size = int(device_pool.size * host_to_device_ratio)
        self.dtype = device_pool.store_dtype
        self.head_num = device_pool.head_num
        self.head_dim = device_pool.head_dim
        self.layer_num = device_pool.layer_num
        self.size_per_token = (
            self.head_dim * self.head_num * self.layer_num * self.dtype.itemsize * 2
        )

        # Verify there is enough available host memory.
        host_mem = psutil.virtual_memory()
        requested_bytes = self.size * self.size_per_token
        # preserve at least 10GB for other usage
        ten_gb = 10 * (1024**3)
        if requested_bytes > host_mem.available - ten_gb:
            raise ValueError(
                f"Not enough host memory available. Requesting "
                f"{requested_bytes / 1e9:.2f} GB but only have "
                f"{host_mem.available / 1e9:.2f} GB free. Please reduce the "
                f"size of the hierarchical cache."
            )
        else:
            logger.info(
                f"Allocating {requested_bytes / 1e9:.2f} GB host memory for hierarchical KV cache."
            )

        self.kv_buffer = torch.empty(
            (2, self.layer_num, self.size, self.head_num, self.head_dim),
            dtype=self.dtype,
            device=self.device,
            pin_memory=self.pin_memory,
        )

        # Initialize memory states and tracking structures.
        self.mem_state = torch.zeros(
            (self.size,), dtype=torch.uint8, device=self.device
        )
        self.free_slots = torch.arange(self.size, dtype=torch.int32)
        self.can_use_mem_size = self.size

        # A lock for synchronized operations on memory allocation and state transitions.
        self.lock = threading.RLock()

    def get_flat_data(self, indices):
        return self.kv_buffer[:, :, indices]

    def assign_flat_data(self, indices, flat_data):
        self.kv_buffer[:, :, indices] = flat_data

    @debug_timing
    def transfer(self, indices, flat_data):
        # backup prepared data from device to host
        self.kv_buffer[:, :, indices] = flat_data.to(
            device=self.device, non_blocking=False
        )

    @synchronized
    def clear(self):
        self.mem_state.fill_(0)
        self.can_use_mem_size = self.size
        self.free_slots = torch.arange(self.size, dtype=torch.int32)

    @synchronized
    def get_state(self, indices: torch.Tensor) -> MemoryStateInt:
        assert len(indices) > 0, "The indices should not be empty"
        states = self.mem_state[indices]
        assert (states == states[0]).all(), (
            "The memory slots should have the same state {}".format(states)
        )
        return MemoryStateInt(states[0].item())

    @synchronized
    def alloc(self, need_size: int) -> torch.Tensor:
        if need_size > self.can_use_mem_size:
            return None

        # todo: de-fragementation
        select_index = self.free_slots[:need_size]
        self.free_slots = self.free_slots[need_size:]

        self.mem_state[select_index] = MemoryStateInt.RESERVED
        self.can_use_mem_size -= need_size

        return select_index

    @synchronized
    def is_reserved(self, indices: torch.Tensor) -> bool:
        return self.get_state(indices) == MemoryStateInt.RESERVED

    @synchronized
    def is_protected(self, indices: torch.Tensor) -> bool:
        return self.get_state(indices) == MemoryStateInt.PROTECTED

    @synchronized
    def is_synced(self, indices: torch.Tensor) -> bool:
        return self.get_state(indices) == MemoryStateInt.SYNCED

    @synchronized
    def is_backup(self, indices: torch.Tensor) -> bool:
        return self.get_state(indices) == MemoryStateInt.BACKUP

    @synchronized
    def update_backup(self, indices: torch.Tensor):
        assert self.is_synced(indices), (
            f"The host memory slots should be in SYNCED state before turning into BACKUP. "
            f"Current state: {self.get_state(indices)}"
        )
        self.mem_state[indices] = MemoryStateInt.BACKUP

    @synchronized
    def update_synced(self, indices: torch.Tensor):
        self.mem_state[indices] = MemoryStateInt.SYNCED

    @synchronized
    def protect_write(self, indices: torch.Tensor):
        assert self.is_reserved(indices), (
            f"The host memory slots should be RESERVED before write operations. "
            f"Current state: {self.get_state(indices)}"
        )
        self.mem_state[indices] = MemoryStateInt.PROTECTED

    @synchronized
    def protect_load(self, indices: torch.Tensor):
        assert self.is_backup(indices), (
            f"The host memory slots should be in BACKUP state before load operations. "
            f"Current state: {self.get_state(indices)}"
        )
        self.mem_state[indices] = MemoryStateInt.PROTECTED

    @synchronized
    def complete_io(self, indices: torch.Tensor):
        assert self.is_protected(indices), (
            f"The host memory slots should be PROTECTED during I/O operations. "
            f"Current state: {self.get_state(indices)}"
        )
        self.mem_state[indices] = MemoryStateInt.SYNCED

    def available_size(self):
        return len(self.free_slots)

    @synchronized
    def free(self, indices: torch.Tensor) -> int:
        self.mem_state[indices] = MemoryStateInt.IDLE
        self.free_slots = torch.concat([self.free_slots, indices])
        self.can_use_mem_size += len(indices)
        return len(indices)

class NativeSparseMHATokenToKVPool(MHATokenToKVPool):

    def __init__(
        self,
        size: int,
        dtype: torch.dtype,
        head_num: int,
        head_dim: int,
        compressed_block_stride: int,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
        max_batch_size: int,
        max_context_len: int,
        page_size: int,
        rank: int
    ):
        super().__init__(
            size=size,
            dtype=dtype,
            head_num=head_num,
            head_dim=head_dim,
            layer_num=layer_num,
            device=device,
            enable_memory_saver=enable_memory_saver,
            max_batch_size=max_batch_size,
            max_context_len=max_context_len,
            page_size=page_size,
            rank=rank,
        )
        assert page_size % compressed_block_stride == 0, f"The page size should be divisible by the kernel stride. Page size: {page_size}, compressed_block_stride: {compressed_block_stride}"
        assert self.store_dtype == self.dtype
        self.compressed_block_stride = compressed_block_stride
        self._create_compressed_buffers()

        compressed_k_size, compressed_v_size = self.get_compressed_kv_size_bytes()
        logger.info(
            f"Compressed KV Cache is allocated. K size: {compressed_k_size / GB:.2f} GB, V size: {compressed_v_size / GB:.2f} GB."
        )

    def _create_compressed_buffers(self):
        compress_block_size = (self.size + self.page_size) // self.compressed_block_stride
        with self.memory_saver_adapter.region():
            # [compress_block_size, head_num, head_dim] for each layer
            # The padded page 0 is used for writing dummy outputs from padded tokens.
            self.compressed_k_buffer = [
                torch.empty(
                    (compress_block_size, self.head_num, self.head_dim),
                    dtype=self.store_dtype,
                    device=self.device,
                )
                for _ in range(self.layer_num)
            ]
            self.compressed_v_buffer = [
                torch.empty(
                    (compress_block_size, self.head_num, self.head_dim),
                    dtype=self.store_dtype,
                    device=self.device,
                )
                for _ in range(self.layer_num)
            ]

    def _clear_compressed_buffers(self):
        del self.compressed_k_buffer
        del self.compressed_v_buffer

    def get_compressed_kv_size_bytes(self):
        assert hasattr(self, "compressed_k_buffer")
        assert hasattr(self, "compressed_v_buffer")
        compressed_k_size_bytes = 0
        for compressed_k_cache in self.compressed_k_buffer:
            compressed_k_size_bytes += np.prod(compressed_k_cache.shape) * compressed_k_cache.dtype.itemsize
        compressed_v_size_bytes = 0
        for compressed_v_cache in self.compressed_v_buffer:
            compressed_v_size_bytes += np.prod(compressed_v_cache.shape) * compressed_v_cache.dtype.itemsize
        return compressed_k_size_bytes, compressed_v_size_bytes

    def get_compressed_key_buffer(self, layer_id: int):
        if self.store_dtype != self.dtype:
            return self.compressed_k_buffer[layer_id].view(self.dtype)
        return self.compressed_k_buffer[layer_id]

    def get_compressed_value_buffer(self, layer_id: int):
        if self.store_dtype != self.dtype:
            return self.compressed_v_buffer[layer_id].view(self.dtype)
        return self.compressed_v_buffer[layer_id]

    def get_compressed_kv_buffer(self, layer_id: int):
        return self.get_compressed_key_buffer(layer_id), self.get_compressed_value_buffer(layer_id)

    def get_contiguous_buf_infos(self):
        raise NotImplementedError

    def get_flat_data(self, indices):
        raise NotImplementedError

    def transfer(self, indices, flat_data):
        raise NotImplementedError


class NativeSparseMLATokenToKVPool(MLATokenToKVPool):
    def __init__(
        self,
        size: int,
        model_dtype: torch.dtype,
        dtype: torch.dtype,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
        max_batch_size: int,
        max_context_len: int,
        page_size: int,
        rank: int,
        compressed_block_stride: int,
    ):
        super().__init__(
            size=size,
            dtype=dtype,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            layer_num=layer_num,
            device=device,
            enable_memory_saver=enable_memory_saver,
            max_batch_size=max_batch_size,
            max_context_len=max_context_len,
            page_size=page_size,
            rank=rank,
        )
        assert page_size % compressed_block_stride == 0, f"The page size should be divisible by the kernel stride. Page size: {page_size}, compressed_block_stride: {compressed_block_stride}"
        assert self.store_dtype == self.dtype
        self.kv_lora_rank = kv_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.compressed_block_stride = compressed_block_stride
        self._create_compressed_buffers()

        compressed_kv_size = self.get_compressed_kv_size_bytes()
        logger.info(
            f"Compressed KV Cache is allocated. latent KV size: {compressed_kv_size / GB:.2f} GB."
        )

    def _create_compressed_buffers(self):
        compress_block_size = (self.size + self.page_size) // self.compressed_block_stride
        with self.memory_saver_adapter.region():
            # [compress_block_size, head_num, head_dim] for each layer
            # The padded page 0 is used for writing dummy outputs from padded tokens.
            self.compressed_kv_buffer = [
                torch.empty(
                    (compress_block_size, 1, self.kv_lora_rank + self.qk_rope_head_dim),
                    dtype=self.store_dtype,
                    device=self.device,
                )
                for _ in range(self.layer_num)
            ]

    def _clear_compressed_buffers(self):
        del self.compressed_k_buffer
        del self.compressed_v_buffer

    def get_compressed_kv_size_bytes(self):
        assert hasattr(self, "compressed_kv_buffer")
        compressed_kv_size_bytes = 0
        for compressed_kv_cache in self.compressed_kv_buffer:
            compressed_kv_size_bytes += np.prod(compressed_kv_cache.shape) * compressed_kv_cache.dtype.itemsize
        return compressed_kv_size_bytes

    def get_compressed_key_buffer(self, layer_id: int):
        if self.store_dtype != self.dtype:
            return self.compressed_kv_buffer[layer_id].view(self.dtype)
        return self.compressed_kv_buffer[layer_id]

    def get_compressed_value_buffer(self, layer_id: int):
        if self.store_dtype != self.dtype:
            return self.compressed_kv_buffer[layer_id][..., : self.kv_lora_rank].view(self.dtype)
        return self.compressed_kv_buffer[layer_id][..., : self.kv_lora_rank]

    def get_compressed_kv_buffer(self, layer_id: int):
        return self.get_compressed_key_buffer(layer_id), self.get_compressed_value_buffer(layer_id)


class SWAKVPool(BaseTokenToKVPool):
    """KV cache with separate pools for full and SWA attention layers."""
    def __init__(
        self,
        size: int,
        size_swa: int,
        dtype: torch.dtype,
        head_num: int,
        head_dim: int,
        swa_attention_layer_ids: List[int],
        full_attention_layer_ids: List[int],
        enable_kvcache_transpose: bool,
        device: str,
    ):
        self.size = size
        self.size_swa = size_swa
        self.dtype = dtype
        self.device = device
        self.swa_layer_nums = len(swa_attention_layer_ids)
        self.full_layer_nums = len(full_attention_layer_ids)
        self.page_size = 1
        # TODO MHATransposedTokenToKVPool if enable_kvcache_transpose is True
        assert not enable_kvcache_transpose
        TokenToKVPoolClass = MHATokenToKVPool
        self.swa_kv_pool = TokenToKVPoolClass(
            size=size_swa,
            page_size=self.page_size,
            dtype=dtype,
            head_num=head_num,
            head_dim=head_dim,
            layer_num=self.swa_layer_nums,
            device=device,
            enable_memory_saver=False,
        )
        self.full_kv_pool = TokenToKVPoolClass(
            size=size,
            page_size=self.page_size,
            dtype=dtype,
            head_num=head_num,
            head_dim=head_dim,
            layer_num=self.full_layer_nums,
            device=device,
            enable_memory_saver=False,
        )
        self.layers_mapping: Dict[int, Tuple[int, bool]] = {}
        for full_attn_layer_id, global_layer_id in enumerate(full_attention_layer_ids):
            self.layers_mapping[global_layer_id] = (full_attn_layer_id, False)
        for swa_layer_id, global_layer_id in enumerate(swa_attention_layer_ids):
            self.layers_mapping[global_layer_id] = (swa_layer_id, True)
        self.full_to_swa_index_mapping: Optional[torch.Tensor] = None

        k_size, v_size = self.get_kv_size_bytes()
        self.mem_usage = (k_size + v_size) / GB

    def get_kv_size_bytes(self):
        k_size, v_size = self.full_kv_pool.get_kv_size_bytes()
        k_size_swa, v_size_swa = self.swa_kv_pool.get_kv_size_bytes()
        return k_size + k_size_swa, v_size + v_size_swa

    def get_contiguous_buf_infos(self):
        full_kv_data_ptrs, full_kv_data_lens, full_kv_item_lens = (
            self.full_kv_pool.get_contiguous_buf_infos()
        )
        swa_kv_data_ptrs, swa_kv_data_lens, swa_kv_item_lens = (
            self.swa_kv_pool.get_contiguous_buf_infos()
        )

        kv_data_ptrs = full_kv_data_ptrs + swa_kv_data_ptrs
        kv_data_lens = full_kv_data_lens + swa_kv_data_lens
        kv_item_lens = full_kv_item_lens + swa_kv_item_lens

        return kv_data_ptrs, kv_data_lens, kv_item_lens

    def get_key_buffer(self, layer_id: int):
        layer_id_pool, is_swa = self.layers_mapping[layer_id]
        if is_swa:
            return self.swa_kv_pool.get_key_buffer(layer_id_pool)
        else:
            return self.full_kv_pool.get_key_buffer(layer_id_pool)

    def get_value_buffer(self, layer_id: int):
        layer_id_pool, is_swa = self.layers_mapping[layer_id]
        if is_swa:
            return self.swa_kv_pool.get_value_buffer(layer_id_pool)
        else:
            return self.full_kv_pool.get_value_buffer(layer_id_pool)

    def get_kv_buffer(self, layer_id: int):
        layer_id_pool, is_swa = self.layers_mapping[layer_id]
        if is_swa:
            return self.swa_kv_pool.get_kv_buffer(layer_id_pool)
        else:
            return self.full_kv_pool.get_kv_buffer(layer_id_pool)

    def translate_loc_from_full_to_swa(self, kv_indices: torch.Tensor):
        assert self.full_to_swa_index_mapping is not None
        return self.full_to_swa_index_mapping[kv_indices].to(torch.int32)

    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        k_scale: float = 1.0,
        v_scale: float = 1.0,
    ):

        layer_id = layer.layer_id
        layer_id_pool, is_swa = self.layers_mapping[layer_id]
        if is_swa:
            if self.full_to_swa_index_mapping is not None:
                loc = self.translate_loc_from_full_to_swa(loc)
            self.swa_kv_pool.set_kv_buffer(
                None,
                loc,
                cache_k,
                cache_v,
                k_scale,
                v_scale,
                layer_id_override=layer_id_pool,
            )
        else:
            self.full_kv_pool.set_kv_buffer(
                None,
                loc,
                cache_k,
                cache_v,
                k_scale,
                v_scale,
                layer_id_override=layer_id_pool,
            )

class MambaPool:
    def __init__(
        self,
        size: int,
        conv_dtype: torch.dtype,
        ssm_dtype: torch.dtype,
        num_mamba_layers: int,
        conv_state_shape: Tuple[int, int],
        temporal_state_shape: Tuple[int, int],
        device: str,
        speculative_num_draft_tokens: Optional[int] = None,
    ):  
        self.is_kda_cache = isinstance(conv_state_shape, List)
        if self.is_kda_cache:
            conv_state = [
                torch.zeros(
                    size=(num_mamba_layers, size + 1) + conv_shape,
                    dtype=conv_dtype,
                    device=device,
                )
                for conv_shape in conv_state_shape
            ]
        else:
            # assume conv_state = (dim, state_len)
            assert conv_state_shape[0] > conv_state_shape[1]
            conv_state = torch.zeros(
                size=(num_mamba_layers, size + 1) + conv_state_shape,
                dtype=conv_dtype,
                device=device,
            )

        temporal_state = torch.zeros(
            size=(num_mamba_layers, size + 1) + temporal_state_shape,
            dtype=ssm_dtype,
            device=device,
        )
        if speculative_num_draft_tokens is not None:
            if self.is_kda_cache:
                intermediate_conv_window_cache = [
                    torch.zeros(
                        size=(
                            num_mamba_layers,
                            size + 1,
                            speculative_num_draft_tokens,
                            conv_shape[0],
                            conv_shape[1],
                        ),
                        dtype=conv_dtype,
                        device="cuda",
                    )
                    for conv_shape in conv_state_shape
                ]
            else:
                intermediate_conv_window_cache = torch.zeros(
                    size=(
                        num_mamba_layers,
                        size + 1,
                        speculative_num_draft_tokens,
                        conv_state_shape[0],
                        conv_state_shape[1],
                    ),
                    dtype=conv_dtype,
                    device="cuda",
                )
            # Cache intermediate SSM states per draft token during target verify
            # Shape: [num_layers, size + 1, speculative_num_draft_tokens, HV, K, V]
            intermediate_ssm_state_cache = torch.empty(
                size=(
                    num_mamba_layers,
                    size + 1,
                    speculative_num_draft_tokens,
                    temporal_state_shape[0],
                    temporal_state_shape[1],
                    temporal_state_shape[2],
                ),
                dtype=ssm_dtype,
                device="cuda",
            )
            self.mamba_cache = (
                conv_state,
                temporal_state,
                intermediate_ssm_state_cache,
                intermediate_conv_window_cache,
            )
            logger.info(
                f"Mamba Cache is allocated. "
                f"conv_state size: {get_tensor_size_bytes(conv_state) / GB:.2f}GB, "
                f"ssm_state size: {get_tensor_size_bytes(temporal_state) / GB:.2f}GB "
                f"intermediate_ssm_state_cache size: {get_tensor_size_bytes(intermediate_ssm_state_cache) / GB:.2f}GB "
                f"intermediate_conv_window_cache size: {get_tensor_size_bytes(intermediate_conv_window_cache) / GB:.2f}GB "
            )
        else:
            self.mamba_cache = (conv_state, temporal_state)
            logger.info(
                f"Mamba Cache is allocated. "
                f"conv_state size: {get_tensor_size_bytes(conv_state) / GB:.2f}GB, "
                f"ssm_state size: {get_tensor_size_bytes(temporal_state) / GB:.2f}GB "
            )
        self.size = size
        self.free_slots = list(range(size))
        self.mem_usage = self.get_mamba_size() / GB

    def get_mamba_params_all_layers(self):
        return [self.mamba_cache[i] for i in range(len(self.mamba_cache))]

    def get_mamba_params(self, layer_id: int):
        return [self.mamba_cache[i][layer_id] for i in range(len(self.mamba_cache))]

    def get_mamba_size(self):
        return sum(get_tensor_size_bytes(t) for t in self.mamba_cache)

    def mamba2_layer_cache(self, layer_id: int):
        return self.at_layer_idx(layer_id)
    
    def at_layer_idx(self, layer: int):
        if isinstance(self.mamba_cache[0], list):
            return {
                "conv": [v[layer] for v in self.mamba_cache[0]],
                "temporal": self.mamba_cache[1][layer],
            }
        return {
            "conv": self.mamba_cache[0][layer],
            "temporal": self.mamba_cache[1][layer],
        }

    def available_size(self):
        return len(self.free_slots)

    def alloc(self, need_size: int) -> Optional[List[int]]:
        if need_size > len(self.free_slots):
            return None

        select_index = self.free_slots[:need_size]
        self.free_slots = self.free_slots[need_size:]

        return select_index

    def free(self, free_index: Union[int, List[int]]):
        if isinstance(free_index, (int,)):
            self.free_slots.append(free_index)
        else:
            self.free_slots.extend(free_index)
        if self.is_kda_cache:
            for i in range(len(self.mamba_cache[0])):
                self.mamba_cache[0][i][:, free_index] = 0
        else:
            self.mamba_cache[0][:, free_index] = 0
        self.mamba_cache[1][:, free_index] = 0

    def clear(self):
        self.free_slots = list(range(self.size))


class HybridReqToTokenPool(ReqToTokenPool):
    """A memory pool that maps a request to its token locations."""

    def __init__(
        self,
        size: int,
        max_context_len: int,
        device: str,
        enable_memory_saver: bool,
        conv_dtype: torch.dtype,
        ssm_dtype: torch.dtype,
        mamba_layers: List[int],
        conv_state_shape: Tuple[int, int],
        temporal_state_shape: Tuple[int, int],
        speculative_num_draft_tokens: int,
    ):
        super().__init__(
            size=size,
            max_context_len=max_context_len,
            device=device,
            enable_memory_saver=enable_memory_saver,
        )

        self.mamba_pool = MambaPool(
            size,
            conv_dtype,
            ssm_dtype,
            len(mamba_layers),
            conv_state_shape,
            temporal_state_shape,
            device,
            speculative_num_draft_tokens,
        )
        self.mamba_map = {layer_id: i for i, layer_id in enumerate(mamba_layers)}

        self.device = device
        self.req_index_to_mamba_index_mapping: torch.Tensor = torch.empty(
            size, dtype=torch.int32, device=self.device
        )

        self.rid_to_mamba_index_mapping: Dict[str, int] = {}
        self.mamba_index_to_rid_mapping: Dict[int, str] = {}

    # For chunk prefill req, we do not need to allocate mamba cache,
    # We could use allocated mamba cache instead.
    def alloc(
        self, need_size: int, reqs: Optional[List["Req"]] = None
    ) -> Optional[List[int]]:
        # need_size not include chunked-req
        select_index = super().alloc(need_size)
        if select_index == None:
            return None

        req_index = select_index[:]
        mamba_index = []
        for i, req in enumerate(reqs):
            # req_index need to correspond 1-to-1 with mamba_index
            if req.req_pool_idx is not None:
                req_index.insert(i, req.req_pool_idx)
            rid = req.rid
            if rid in self.rid_to_mamba_index_mapping:
                mid = self.rid_to_mamba_index_mapping[rid]
            elif (mid := self.mamba_pool.alloc(1)) is not None:
                mid = mid[0]
                self.rid_to_mamba_index_mapping[rid] = mid
                self.mamba_index_to_rid_mapping[mid] = rid
            mamba_index.append(mid)
        assert len(req_index) == len(
            mamba_index
        ), f"Not enough space for mamba cache, try to increase --max-mamba-cache-size."
        self.req_index_to_mamba_index_mapping[req_index] = torch.tensor(
            mamba_index, dtype=torch.int32, device=self.device
        )
        return select_index

    def get_mamba_indices(self, req_indices: torch.Tensor) -> torch.Tensor:
        return self.req_index_to_mamba_index_mapping[req_indices]

    def get_mamba_params(self, layer_id: int):
        assert layer_id in self.mamba_map
        return self.mamba_pool.get_mamba_params(self.mamba_map[layer_id])

    def get_mamba_params_all_layers(self):
        return self.mamba_pool.get_mamba_params_all_layers()

    def mamba2_layer_cache(self, layer_id: int):
        assert layer_id in self.mamba_map
        return self.mamba_pool.mamba2_layer_cache(self.mamba_map[layer_id])

    # For chunk prefill, we can not free mamba cache, we need use it in the future
    def free(self, free_index: Union[int, List[int]], free_mamba_cache: bool = True):
        super().free(free_index)
        if free_mamba_cache:
            mamba_index = self.req_index_to_mamba_index_mapping[free_index]
            mamba_index_list = mamba_index.tolist()
            if isinstance(mamba_index_list, int):
                mamba_index_list = [mamba_index_list]
            self.mamba_pool.free(mamba_index_list)
            for mid in mamba_index_list:
                rid = self.mamba_index_to_rid_mapping[mid]
                self.mamba_index_to_rid_mapping.pop(mid)
                self.rid_to_mamba_index_mapping.pop(rid)

    def clear(self):
        super().clear()
        self.mamba_pool.clear()


class HybridLinearKVPool(BaseTokenToKVPool):
    """KV cache with separate pools for full and linear attention layers."""

    def __init__(
        self,
        size: int,
        dtype: torch.dtype,
        head_num: int,
        head_dim: int,
        full_attention_layer_ids: List[int],
        enable_kvcache_transpose: bool,
        device: str,
        max_batch_size: int,
        max_context_len: int,
        page_size: int,
        rank: int,
        # TODO: refactor mla related args
        model_dtype: torch.dtype = torch.bfloat16,
        quant_method: str = "",
        use_mla: bool = False,
        kv_lora_rank: int = None,
        qk_rope_head_dim: int = None,
    ):
        self.size = size
        self.dtype = dtype
        self.device = device
        self.full_layer_nums = len(full_attention_layer_ids)
        self.page_size = 1
        self.use_mla = use_mla
        # TODO MHATransposedTokenToKVPool if enable_kvcache_transpose is True
        assert not enable_kvcache_transpose

        if not use_mla:
            self.full_kv_pool = MHATokenToKVPool(
                size=size,
                page_size=self.page_size,
                dtype=dtype,
                head_num=head_num,
                head_dim=head_dim,
                layer_num=self.full_layer_nums,
                device=device,
                enable_memory_saver=False,
                max_batch_size=max_batch_size,
                max_context_len=max_context_len,
                rank=rank,
            )
        else:
            self.full_kv_pool = MLATokenToKVPool(
                size=size,
                page_size=self.page_size,
                dtype=dtype,
                kv_lora_rank=kv_lora_rank,
                qk_rope_head_dim=qk_rope_head_dim,
                layer_num=self.full_layer_nums,
                device=device,
                enable_memory_saver=False,
                max_batch_size=max_batch_size,
                max_context_len=max_context_len,
                rank=rank,
                model_dtype=model_dtype,
                quant_method=quant_method,
            )
        self.full_attention_layer_id_mapping = {
            id: i for i, id in enumerate(full_attention_layer_ids)
        }
        if use_mla:
            self.mem_usage = self.get_kv_size_bytes() / GB
        else:
            k_size, v_size = self.get_kv_size_bytes()
            self.mem_usage = (k_size + v_size) / GB

    def get_kv_size_bytes(self):
        return self.full_kv_pool.get_kv_size_bytes()

    def get_contiguous_buf_infos(self):
        return self.full_kv_pool.get_contiguous_buf_infos()

    def _transfer_full_attention_id(self, layer_id: int):
        if layer_id not in self.full_attention_layer_id_mapping:
            raise ValueError(
                f"{layer_id=} not in full attention layers: {self.full_attention_layer_id_mapping.keys()}"
            )
        return self.full_attention_layer_id_mapping[layer_id]

    def get_key_buffer(self, layer_id: int):
        layer_id = self._transfer_full_attention_id(layer_id)
        return self.full_kv_pool.get_key_buffer(layer_id)

    def get_value_buffer(self, layer_id: int):
        layer_id = self._transfer_full_attention_id(layer_id)
        return self.full_kv_pool.get_value_buffer(layer_id)

    def get_kv_buffer(self, layer_id: int):
        layer_id = self._transfer_full_attention_id(layer_id)
        return self.full_kv_pool.get_kv_buffer(layer_id)

    @contextmanager
    def _transfer_id_context(self, layer: RadixAttention):

        @contextmanager
        def _patch_layer_id(layer):
            original_layer_id = layer.layer_id
            layer.layer_id = self._transfer_full_attention_id(layer.layer_id)
            try:
                yield
            finally:
                layer.layer_id = original_layer_id

        with _patch_layer_id(layer):
            yield

    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        k_scale: float = 1.0,
        v_scale: float = 1.0,
    ):
        layer_id = self._transfer_full_attention_id(layer.layer_id)
        if not self.use_mla:
            self.full_kv_pool.set_kv_buffer(
                None,
                loc,
                cache_k,
                cache_v,
                k_scale,
                v_scale,
                layer_id_override=layer_id,
            )
        else:
            with self._transfer_id_context(layer):
                self.full_kv_pool.set_kv_buffer(
                    layer,
                    loc,
                    cache_k,
                    cache_v,
                )
    def set_mla_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k_nope: torch.Tensor,
        cache_k_rope: torch.Tensor,
    ):
        assert self.use_mla, "set_mla_kv_buffer called when use_mla is False"
        with self._transfer_id_context(layer):
            self.full_kv_pool.set_mla_kv_buffer(layer, loc, cache_k_nope, cache_k_rope)

    def get_mla_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        dst_dtype: Optional[torch.dtype] = None,
    ):
        assert self.use_mla, "get_mla_kv_buffer called when use_mla is False"
        with self._transfer_id_context(layer):
            return self.full_kv_pool.get_mla_kv_buffer(layer, loc, dst_dtype)


from sglang.srt.layers.attention.dsa import index_buf_accessor
class DSATokenToKVPool(MLATokenToKVPool):
    quant_block_size = 128

    def __init__(
        self,
        size: int,
        model_dtype: torch.dtype,
        dtype: torch.dtype,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
        max_batch_size: int,
        max_context_len: int,
        page_size: int,
        rank: int,
        index_head_dim: int,
        index_dtype: torch.dtype
    ):
        super().__init__(
            size,
            model_dtype,
            dtype,
            None, # quant_method
            kv_lora_rank,
            qk_rope_head_dim,
            layer_num,
            device,
            enable_memory_saver,
            max_batch_size,
            max_context_len,
            page_size,
            rank,
            True,
        )
        # self.index_k_dtype = torch.float8_e4m3fn
        # self.index_k_scale_dtype = torch.float32
        assert index_dtype in [torch.float8_e4m3fn, torch.bfloat16]
        # num head == 1 and head dim == 128 for index_k in NSA
        assert index_head_dim == 128
        assert self.page_size == 64
        self.index_head_dim = index_head_dim
        self.index_dtype = index_dtype
        if index_dtype == torch.float8_e4m3fn:
            self.index_k_cache_dim = index_head_dim + index_head_dim // self.quant_block_size * 4
            self.index_k_with_scale_buffer_dtype = torch.uint8
        else:
            self.index_k_cache_dim = index_head_dim
            self.index_k_with_scale_buffer_dtype = torch.bfloat16

        with self.memory_saver_adapter.region():
            self.index_k_with_scale_buffer = [
                torch.zeros(
                    # Layout:
                    #     ref: test_attention.py :: kv_cache_cast_to_fp8
                    #     shape: (num_pages, page_size 64 * head_dim 128 + page_size 64 * fp32_nbytes 4)
                    #     data: for page i,
                    #         * buf[i, :page_size * head_dim] for fp8 data
                    #         * buf[i, page_size * head_dim:].view(float32) for scale
                    (
                        (size + page_size + 1) // self.page_size,
                        self.page_size * self.index_k_cache_dim,
                    ),
                    dtype=self.index_k_with_scale_buffer_dtype,
                    device=device,
                )
                for _ in range(layer_num)
            ]

    def get_contiguous_buf_infos(self):
        kv_data_ptrs, kv_data_lens, kv_item_lens = (
            super().get_contiguous_buf_infos()
        )
        index_data_ptrs, index_data_lens, index_item_lens = self.get_state_buf_infos()

        return (
            kv_data_ptrs + index_data_ptrs,
            kv_data_lens + index_data_lens,
            kv_item_lens + index_item_lens
        )

    def get_layerwise_buf_info_offsets(self, start_idx=0):
        return [[start_idx + i*self.layer_num + layer_id for i in range(2)] for layer_id in range(self.layer_num)]

    def get_index_k_with_scale_buffer(self, layer_id: int) -> torch.Tensor:
        if getattr(self, "layer_transfer_counter", None) is not None:
            self.layer_transfer_counter.wait_until(layer_id)
        return self.index_k_with_scale_buffer[layer_id]

    def get_index_k_continuous(
        self,
        layer_id: int,
        seq_len: int,
        page_indices: torch.Tensor,
    ):
        buf = self.index_k_with_scale_buffer[layer_id]
        return index_buf_accessor.GetK.execute(
            self, buf, seq_len=seq_len, page_indices=page_indices
        )

    def get_index_k_scale_continuous(
        self,
        layer_id: int,
        seq_len: int,
        page_indices: torch.Tensor,
    ):
        buf = self.index_k_with_scale_buffer[layer_id]
        if self.index_dtype == torch.float8_e4m3fn:
            return index_buf_accessor.GetS.execute(
                self, buf, seq_len=seq_len, page_indices=page_indices
            )
        else:
            return buf[page_indices]

    # TODO rename later (currently use diff name to avoid confusion)
    def set_index_k_and_scale_buffer(
        self,
        layer_id: int,
        loc: torch.Tensor,
        index_k: torch.Tensor,
        index_k_scale: torch.Tensor = None,
    ) -> None:
        buf = self.index_k_with_scale_buffer[layer_id]
        if self.index_dtype == torch.float8_e4m3fn:
            index_buf_accessor.SetKAndS.execute(
                pool=self, buf=buf, loc=loc, index_k=index_k, index_k_scale=index_k_scale
            )
        else:
            buf = buf.view(-1, self.index_k_cache_dim)
            buf[loc] = index_k

    def get_state_buf_infos(self):
        data_ptrs = [
            self.index_k_with_scale_buffer[i].data_ptr() for i in range(self.layer_num)
        ]
        data_lens = [
            self.index_k_with_scale_buffer[i].nbytes for i in range(self.layer_num)
        ]
        item_lens = [
            self.index_k_with_scale_buffer[i][0].nbytes for i in range(self.layer_num)
        ]
        return data_ptrs, data_lens, item_lens

    def get_kv_size_bytes(self):
        kv_size_bytes = super().get_kv_size_bytes()
        for index_k_cache in self.index_k_with_scale_buffer:
            kv_size_bytes += get_tensor_size_bytes(index_k_cache)
        return kv_size_bytes

def move_kv_cache_native(
    k_buffer: List[torch.Tensor],
    v_buffer: List[torch.Tensor],
    tgt_loc: torch.Tensor,
    src_loc: torch.Tensor,
):
    if tgt_loc.numel() == 0:
        return

    tgt_loc_flat = tgt_loc.view(-1).long()
    src_loc_flat = src_loc.view(-1).long()
    for k_cache, v_cache in zip(k_buffer, v_buffer):
        k_cache[tgt_loc_flat] = k_cache[src_loc_flat]
        v_cache[tgt_loc_flat] = v_cache[src_loc_flat]

@triton.jit
def copy_all_layer_kv_cache_tiled(
    data_ptrs,
    strides,
    tgt_loc_ptr,
    src_loc_ptr,
    num_locs,
    num_locs_upper: tl.constexpr,
    BYTES_PER_TILE: tl.constexpr,
):
    """2D tiled kernel. Safe for in-place copy."""
    bid = tl.program_id(0)
    tid = tl.program_id(1)

    stride = tl.load(strides + bid)
    base_ptr = tl.load(data_ptrs + bid)
    base_ptr = tl.cast(base_ptr, tl.pointer_type(tl.uint8))

    byte_off = tid * BYTES_PER_TILE + tl.arange(0, BYTES_PER_TILE)
    mask_byte = byte_off < stride
    tl.multiple_of(byte_off, 16)

    loc_idx = tl.arange(0, num_locs_upper)
    mask_loc = loc_idx < num_locs

    src = tl.load(src_loc_ptr + loc_idx, mask=mask_loc, other=0)
    tgt = tl.load(tgt_loc_ptr + loc_idx, mask=mask_loc, other=0)

    src_ptr = base_ptr + src[:, None] * stride + byte_off[None, :]
    tgt_ptr = base_ptr + tgt[:, None] * stride + byte_off[None, :]

    mask = mask_loc[:, None] & mask_byte[None, :]
    vals = tl.load(src_ptr, mask=mask)
    tl.store(tgt_ptr, vals, mask=mask)
