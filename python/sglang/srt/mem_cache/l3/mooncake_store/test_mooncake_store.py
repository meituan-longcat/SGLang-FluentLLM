import logging
import os
import uuid
from collections.abc import Sequence

import numpy as np
import pytest
import torch
from mooncake_store import MooncakeStore

from sglang.srt.mem_cache.l3.storage import (
    HiCacheStorageConfig,
    HiCacheStorageExtraInfo,
)
from sglang.srt.utils import get_colorful_logger

logger = get_colorful_logger(__name__)


def test_model_prefix_prefers_l3_namespace(monkeypatch, caplog):
    monkeypatch.setenv("L3_NAMESPACE", "explicit_model")
    monkeypatch.setenv("MOONCAKE_MODEL_NAME", "ignored_legacy_model")
    monkeypatch.setenv("ARGS_MODEL_NAME", "args_model")

    with caplog.at_level(logging.INFO):
        assert MooncakeStore._resolve_model_prefix(None) == "explicit_model"
    assert "L3_NAMESPACE='explicit_model' namespace_applied=True" in caplog.text
    assert "source=L3_NAMESPACE model_prefix='explicit_model'" in caplog.text


def test_model_prefix_falls_back_to_args_model_name(monkeypatch, caplog):
    monkeypatch.delenv("L3_NAMESPACE", raising=False)
    monkeypatch.setenv("MOONCAKE_MODEL_NAME", "ignored_legacy_model")
    monkeypatch.setenv("ARGS_MODEL_NAME", "args_model")

    with caplog.at_level(logging.INFO):
        assert MooncakeStore._resolve_model_prefix(None) == "args_model"
    assert "L3_NAMESPACE=None namespace_applied=False" in caplog.text
    assert "source=ARGS_MODEL_NAME model_prefix='args_model'" in caplog.text


def test_model_prefix_treats_empty_explicit_name_as_missing(monkeypatch, caplog):
    monkeypatch.setenv("L3_NAMESPACE", "")
    monkeypatch.setenv("ARGS_MODEL_NAME", "args_model")

    with caplog.at_level(logging.INFO):
        assert MooncakeStore._resolve_model_prefix(None) == "args_model"
    assert "L3_NAMESPACE='' namespace_applied=False" in caplog.text
    assert "source=ARGS_MODEL_NAME model_prefix='args_model'" in caplog.text


def test_model_prefix_falls_back_to_storage_config(monkeypatch):
    monkeypatch.delenv("L3_NAMESPACE", raising=False)
    monkeypatch.delenv("ARGS_MODEL_NAME", raising=False)
    config = HiCacheStorageConfig(0, 1, True, False, "served_model")

    assert MooncakeStore._resolve_model_prefix(config) == "served_model"


def test_model_prefix_rejects_empty_namespace(monkeypatch):
    monkeypatch.delenv("L3_NAMESPACE", raising=False)
    monkeypatch.setenv("MOONCAKE_MODEL_NAME", "ignored_legacy_model")
    monkeypatch.delenv("ARGS_MODEL_NAME", raising=False)

    with pytest.raises(ValueError, match="L3 model namespace is required"):
        MooncakeStore._resolve_model_prefix(None)


def test_model_prefix_applies_to_keys_without_repeated_logging(monkeypatch, caplog):
    monkeypatch.setenv("L3_NAMESPACE", "shared_model_")
    backend = MooncakeStore.__new__(MooncakeStore)
    backend.pp_prefix = "pp0of8_"
    backend.extra_backend_tag = None
    backend.is_mla_backend = True
    captured_keys = []

    def fake_batch_exist(keys, **kwargs):
        captured_keys.append(keys)
        return [1] * len(keys)

    backend._batch_exist = fake_batch_exist
    with caplog.at_level(logging.INFO):
        backend.model_prefix = backend._resolve_model_prefix(None)
        for _ in range(3):
            assert backend.batch_exists(["page_hash"]) == 1
        assert backend._resolve_pp_prefix(HiCacheStorageExtraInfo(pp_prefix="pp7of8_")) == "shared_model_pp7of8_"

    assert captured_keys == [["shared_model_pp0of8_page_hash_k"]] * 3
    assert caplog.text.count("[L3-Key-Namespace]") == 1


class _RegistrationRecorder:
    def __init__(self, result=0):
        self.result = result
        self.calls = []

    def register_buffers(self, pointers, sizes):
        self.calls.append((pointers, sizes))
        return self.result


def test_register_mem_pool_host_publishes_once(monkeypatch):
    backend = MooncakeStore.__new__(MooncakeStore)
    backend.store = _RegistrationRecorder()
    pool = type("Pool", (), {"layout": "layer_first", "buffers": object()})()
    tensors = [torch.empty(4), torch.empty(8), torch.empty(16)]
    monkeypatch.setattr(
        backend,
        "_iter_registration_targets",
        lambda _: iter((f"buffer_{i}", tensor) for i, tensor in enumerate(tensors)),
    )

    backend.register_mem_pool_host(pool)

    assert len(backend.store.calls) == 1
    pointers, sizes = backend.store.calls[0]
    assert pointers == [tensor.data_ptr() for tensor in tensors]
    assert sizes == [tensor.numel() * tensor.element_size() for tensor in tensors]


def test_register_mem_pool_host_fails_before_ready(monkeypatch):
    backend = MooncakeStore.__new__(MooncakeStore)
    backend.store = _RegistrationRecorder(result=-1)
    pool = type("Pool", (), {"layout": "layer_first", "buffers": object()})()
    tensor = torch.empty(4)
    monkeypatch.setattr(
        backend,
        "_iter_registration_targets",
        lambda _: iter((("buffer", tensor),)),
    )

    with pytest.raises(RuntimeError, match="batch host buffer registration failed"):
        backend.register_mem_pool_host(pool)

    assert len(backend.store.calls) == 1


def test_batch_exists_renews_local_prefix_and_orders_suffix():
    class FakeStore:
        def __init__(self):
            self.call = None

        def batch_is_exist(self, keys, ordered_chain=False, lease_prefix_length=0):
            self.call = (keys, ordered_chain, lease_prefix_length)
            # Two logical MHA prefix pages, then two queried pages. The second
            # queried page misses on its K object.
            return [1, 0, 1, 1, 1, 1, 0, 1]

    backend = MooncakeStore.__new__(MooncakeStore)
    backend.store = FakeStore()
    backend.is_mla_backend = False
    backend.local_rank = 3
    backend.model_prefix = "model_"
    backend.pp_prefix = "pp0_"
    backend.extra_backend_tag = None

    hit_pages = backend.batch_exists(
        ["q0", "q1"],
        HiCacheStorageExtraInfo(prefix_keys=["p0", "p1"]),
    )

    assert hit_pages == 1
    keys, ordered_chain, lease_prefix_length = backend.store.call
    assert ordered_chain is True
    assert lease_prefix_length == 4
    assert keys == [
        "model_pp0_p0_3_k",
        "model_pp0_p0_3_v",
        "model_pp0_p1_3_k",
        "model_pp0_p1_3_v",
        "model_pp0_q0_3_k",
        "model_pp0_q0_3_v",
        "model_pp0_q1_3_k",
        "model_pp0_q1_3_v",
    ]


def test_batch_exist_default_path_keeps_legacy_call_shape():
    class LegacyStore:
        def __init__(self):
            self.keys = None

        def batch_is_exist(self, keys):
            self.keys = keys
            return [1] * len(keys)

    backend = MooncakeStore.__new__(MooncakeStore)
    backend.store = LegacyStore()

    assert backend._batch_exist(["k0", "k1"]) == [1, 1]
    assert backend.store.keys == ["k0", "k1"]


def test_backup_exist_query_renews_prefix_without_affecting_put_results():
    backend = MooncakeStore.__new__(MooncakeStore)
    backend.is_mla_backend = False
    backend.local_rank = 3
    backend.model_prefix = "model_"
    backend.pp_prefix = "pp0of2_"
    backend.extra_backend_tag = "tag"

    exist_calls = []
    put_calls = []

    def fake_preprocess(keys, _host_indices, _layer_range):
        physical = []
        for key in keys:
            physical.extend([f"{key}_3_k", f"{key}_3_v"])
        return physical, list(range(len(physical))), [64] * len(physical)

    def fake_batch_exist(keys, ordered_chain=False, lease_prefix_length=0):
        exist_calls.append((keys, ordered_chain, lease_prefix_length))
        # Prefix results are deliberately mixed and must not affect backup.
        # q0 already exists; only q1's K/V objects need to be written.
        return [1, 0, 1, 1, 0, 0]

    def fake_put(keys, ptrs, sizes):
        put_calls.append((keys, ptrs, sizes))
        return [0] * len(keys)

    backend._batch_preprocess = fake_preprocess
    backend._batch_exist = fake_batch_exist
    backend._put_batch_zero_copy_impl = fake_put

    results = backend.batch_set_zero_copy(
        ["q0", "q1"],
        [0],
        HiCacheStorageExtraInfo(prefix_keys=["pp0", "p0"], pp_prefix="pp1of2_"),
    )

    assert results == [True, True]
    assert exist_calls == [
        (
            [
                "tag_model_pp1of2_p0_3_k",
                "tag_model_pp1of2_p0_3_v",
                "tag_model_pp1of2_q0_3_k",
                "tag_model_pp1of2_q0_3_v",
                "tag_model_pp1of2_q1_3_k",
                "tag_model_pp1of2_q1_3_v",
            ],
            False,
            0,
        )
    ]
    assert put_calls == [
        (
            ["tag_model_pp1of2_q1_3_k", "tag_model_pp1of2_q1_3_v"],
            [2, 3],
            [64, 64],
        )
    ]


def test_expand_exist_query_keys_for_mla():
    backend = MooncakeStore.__new__(MooncakeStore)
    backend.is_mla_backend = True
    assert backend._expand_exist_query_keys(["p0", "p1"]) == (
        ["p0_k", "p1_k"],
        1,
    )


def generate_batch_query_keys(kv_num: int, config: HiCacheStorageConfig):
    keys = []
    for _ in range(kv_num):
        key = "test_" + str(uuid.uuid4())
        keys.append(key)
    set_keys = []
    for key in keys:
        if config.is_mla_model:
            set_keys.append(key + "_k")
        else:
            set_keys.append(key + f"_{config.tp_rank}_k")
            set_keys.append(key + f"_{config.tp_rank}_v")
    get_keys = set_keys
    exist_keys = keys
    return set_keys, get_keys, exist_keys


def create_mock_host_kv_cache(buffer_size, dtype=torch.float32):
    buffer = torch.randn(buffer_size, dtype=dtype)

    class MockHostTokenToKVPool:
        def __init__(self, buffer):
            self.kv_buffer = buffer
            self.layout = "page_first"
            self.page_size = 1
            self.buffers = type("Buffers", (), {"kv_buffer": buffer})()

        def get_page_buffer_meta(self, indices):
            ptr_list = []
            element_size_list = []
            elem_bytes = self.kv_buffer.element_size()
            base_ptr = self.kv_buffer.data_ptr()
            for idx in indices.tolist():
                ptr_list.append(base_ptr + idx * self.page_size * elem_bytes)
                element_size_list.append(self.page_size * elem_bytes)
            return ptr_list, element_size_list

    return MockHostTokenToKVPool(buffer), buffer


def create_mock_layer_first_mla_pool(page_num=4, page_size=2, layer_num=3, dtype=torch.float32):
    kv_lora_rank = 4
    qk_rope_head_dim = 6
    k_buffer = []
    v_buffer = []
    k_buffer_chunks = []
    v_buffer_chunks = []
    chunk_size = 2
    for _ in range(layer_num):
        layer_k = []
        layer_v = []
        layer_k_chunks = []
        layer_v_chunks = []
        for chunk_start in range(0, page_num, chunk_size):
            chunk_len = min(chunk_size, page_num - chunk_start)
            chunk_k = torch.randn(chunk_len, page_size, 1, kv_lora_rank, dtype=dtype)
            chunk_v = torch.randn(chunk_len, page_size, 1, qk_rope_head_dim, dtype=dtype)
            layer_k_chunks.append(chunk_k)
            layer_v_chunks.append(chunk_v)
            for p in range(chunk_len):
                layer_k.append(chunk_k[p])
                layer_v.append(chunk_v[p])
        k_buffer.append(layer_k)
        v_buffer.append(layer_v)
        k_buffer_chunks.append(layer_k_chunks)
        v_buffer_chunks.append(layer_v_chunks)

    class MockLayerFirstMLAPool:
        def __init__(self):
            self.layout = "layer_first"
            self.page_size = page_size
            self.layer_num = layer_num
            self.kv_lora_rank = kv_lora_rank
            self.qk_rope_head_dim = qk_rope_head_dim
            self.dtype = dtype
            self.buffers = type(
                "Buffers",
                (),
                {
                    "kv_buffer": None,
                    "k_buffer": k_buffer,
                    "v_buffer": v_buffer,
                    "index_k_buffer": None,
                    "k_buffer_chunks": k_buffer_chunks,
                    "v_buffer_chunks": v_buffer_chunks,
                    "index_k_buffer_chunks": None,
                },
            )()

        def get_page_buffer_meta(self, indices):
            assert len(indices) % self.page_size == 0
            ptr_list = []
            size_list = []
            k_bytes = self.page_size * self.kv_lora_rank * self.dtype.itemsize
            v_bytes = self.page_size * self.qk_rope_head_dim * self.dtype.itemsize
            for i in range(0, len(indices), self.page_size):
                page_idx = int(indices[i]) // self.page_size
                for layer_id in range(self.layer_num):
                    ptr_list.append(self.buffers.k_buffer[layer_id][page_idx].data_ptr())
                    size_list.append(k_bytes)
                    ptr_list.append(self.buffers.v_buffer[layer_id][page_idx].data_ptr())
                    size_list.append(v_bytes)
            return ptr_list, size_list

    return MockLayerFirstMLAPool()


def create_mock_layer_first_dsa_pool(page_num=4, page_size=2, layer_num=3, dtype=torch.float32):
    pool = create_mock_layer_first_mla_pool(page_num, page_size, layer_num, dtype)
    index_head_dim = 5
    index_k_buffer = []
    index_k_buffer_chunks = []
    chunk_size = 2
    for _ in range(layer_num):
        layer_idx = []
        layer_idx_chunks = []
        for chunk_start in range(0, page_num, chunk_size):
            chunk_len = min(chunk_size, page_num - chunk_start)
            chunk = torch.randn(chunk_len, page_size, 1, index_head_dim, dtype=dtype)
            layer_idx_chunks.append(chunk)
            for p in range(chunk_len):
                layer_idx.append(chunk[p])
        index_k_buffer.append(layer_idx)
        index_k_buffer_chunks.append(layer_idx_chunks)
    pool.buffers.index_k_buffer = index_k_buffer
    pool.buffers.index_k_buffer_page_slices = index_k_buffer
    pool.buffers.index_k_buffer_chunks = index_k_buffer_chunks
    pool.device_pool = type("DevicePool", (), {"index_head_dim": index_head_dim})()

    def _get_page_buffer_meta(indices):
        base_ptrs, base_sizes = create_mock_layer_first_mla_pool(page_num, page_size, layer_num, dtype).get_page_buffer_meta(indices)
        ptr_list = []
        size_list = []
        stride = layer_num * 2
        idx_bytes = page_size * index_head_dim * dtype.itemsize
        for page_idx in range(0, len(indices), page_size):
            base = (page_idx // page_size) * stride
            host_page_idx = int(indices[page_idx]) // page_size
            for layer_id in range(layer_num):
                kv_base = base + layer_id * 2
                ptr_list.append(base_ptrs[kv_base])
                size_list.append(base_sizes[kv_base])
                ptr_list.append(base_ptrs[kv_base + 1])
                size_list.append(base_sizes[kv_base + 1])
                ptr_list.append(index_k_buffer[layer_id][host_page_idx].data_ptr())
                size_list.append(idx_bytes)
        return ptr_list, size_list

    pool.get_page_buffer_meta = _get_page_buffer_meta
    return pool


def test_single_operation():
    print("=" * 100)
    print("Testing single operation")

    buffer_size = 1024 * 1024 * 16
    value_elements = 1024
    store = MooncakeStore()
    mock_host_kv_cache, buffer = create_mock_host_kv_cache(buffer_size)

    store.register_mem_pool_host(mock_host_kv_cache)

    value_size = value_elements * buffer.element_size()

    key = str(uuid.uuid4())
    set_slice = buffer[:value_elements]
    get_slice = buffer[value_elements : 2 * value_elements]
    set_location = set_slice.data_ptr()
    get_location = get_slice.data_ptr()

    result = store.set(key, target_location=set_location, target_sizes=value_size)
    assert result is True, f"❌set operation failed for key: {key}"

    assert store.exists(key), f"❌key {key} should exist after set operation"

    result = store.get(key, target_location=get_location, target_sizes=value_size)
    assert result is True, f"❌get operation failed for key: {key}"

    assert torch.allclose(set_slice, get_slice, atol=1e-6), f"❌get operation failed for key: {key}"

    logger.info("✅ Single operation passed")


def test_batch_operation(config: HiCacheStorageConfig):
    print("=" * 100)
    print(f"Testing batch operation with config: {config}")

    buffer_size = 1024 * 1024 * 16
    value_elements = 256
    kv_num = 13
    store = MooncakeStore(config)
    mock_host_kv_cache, buffer = create_mock_host_kv_cache(buffer_size)

    store.register_mem_pool_host(mock_host_kv_cache)

    value_size = value_elements * buffer.element_size()

    set_keys, get_keys, exist_keys = generate_batch_query_keys(kv_num, config)
    set_slices = [buffer[i * value_elements : (i + 1) * value_elements] for i in range(len(set_keys))]
    set_locations = [set_slice.data_ptr() for set_slice in set_slices]
    target_sizes = [value_size for _ in range(len(set_keys))]

    result = store.batch_set(set_keys, target_locations=set_locations, target_sizes=target_sizes)
    assert result is True, "❌batch set operation failed"

    assert store.batch_exists(exist_keys), "❌keys should exist after batch set operation"

    get_slices = [buffer[(len(set_keys) + i) * value_elements : (len(set_keys) + i + 1) * value_elements] for i in range(len(get_keys))]
    get_locations = [get_slice.data_ptr() for get_slice in get_slices]
    result = store.batch_get(get_keys, target_locations=get_locations, target_sizes=target_sizes)
    assert result == kv_num, "❌batch get operation failed"
    for i in range(len(get_keys)):
        assert torch.allclose(set_slices[i], get_slices[i], atol=1e-6), f"❌batch get operation failed for key: {get_keys[i]}"

    logger.info("✅ Batch operation passed")


def _get_mooncake_test_ip() -> str:
    test_ip = os.environ.get("MOONCAKE_TEST_IP", "").strip()
    if not test_ip:
        raise ValueError("Set MOONCAKE_TEST_IP to this test node's IP address before running Mooncake integration tests.")
    return test_ip


def test_layer_first_registration_target_diff():
    print("=" * 100)
    print("Testing layer_first registration targets")

    store = create_store(f"{_get_mooncake_test_ip()}:{13001 + (os.getpid() % 1000)}")
    pool = create_mock_layer_first_mla_pool()

    reg_targets = list(store._iter_registration_targets(pool.buffers))
    assert len(reg_targets) == pool.layer_num * len(pool.buffers.k_buffer_chunks[0]) * 2
    assert all("buffer_chunks" in name for name, _ in reg_targets)
    assert all(isinstance(tensor, torch.Tensor) and tensor.dim() == 4 for _, tensor in reg_targets)

    bad_targets = []
    for name in ["k_buffer", "v_buffer"]:
        bad_targets.extend(list(store._iter_registered_tensors(name, getattr(pool.buffers, name))))
    assert bad_targets, "expected legacy page-slice registration targets"
    assert all(isinstance(tensor, torch.Tensor) and tensor.dim() == 3 for _, tensor in bad_targets)
    assert bad_targets[0][0].startswith("k_buffer[0][0]")

    logger.info("✅ layer_first registration target diff passed")


def test_layer_first_meta_pack():
    print("=" * 100)
    print("Testing layer_first MLA meta packing")

    store = create_store(f"{_get_mooncake_test_ip()}:{14001 + (os.getpid() % 1000)}")
    pool = create_mock_layer_first_mla_pool()
    store.register_mem_pool_host(pool)
    keys = ["page0", "page1"]
    indices = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
    key_strs, buffer_ptrs, buffer_sizes = store._batch_preprocess(keys, indices)

    assert key_strs == ["page0_k", "page1_k"]
    assert len(buffer_ptrs) == len(keys)
    assert len(buffer_sizes) == len(keys)
    assert all(isinstance(group, Sequence) for group in buffer_ptrs)
    assert all(len(group) == pool.layer_num * 2 for group in buffer_ptrs)
    assert all(len(group) == pool.layer_num * 2 for group in buffer_sizes)
    assert store._uses_multi_buffer(buffer_ptrs)

    logger.info("✅ layer_first MLA meta packing passed")


def test_layer_first_dsa_meta_pack():
    print("=" * 100)
    print("Testing layer_first DSA meta packing")

    store = create_store(f"{_get_mooncake_test_ip()}:{14001 + (os.getpid() % 1000)}")
    pool = create_mock_layer_first_dsa_pool()
    store.register_mem_pool_host(pool)
    keys = ["page0", "page1"]
    indices = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
    key_strs, buffer_ptrs, buffer_sizes = store._batch_preprocess(keys, indices)

    assert key_strs == ["page0_k", "page1_k"]
    assert len(buffer_ptrs) == len(keys)
    assert len(buffer_sizes) == len(keys)
    assert all(isinstance(group, Sequence) for group in buffer_ptrs)
    assert all(len(group) == pool.layer_num * 3 for group in buffer_ptrs)
    assert all(len(group) == pool.layer_num * 3 for group in buffer_sizes)
    assert store._uses_multi_buffer(buffer_ptrs)

    reg_targets = list(store._iter_registration_targets(pool.buffers))
    expected_chunks = pool.layer_num * len(pool.buffers.k_buffer_chunks[0]) * 2 + pool.layer_num * len(pool.buffers.index_k_buffer_chunks[0])
    assert len(reg_targets) == expected_chunks
    assert all("buffer_chunks" in name for name, _ in reg_targets)

    logger.info("✅ layer_first DSA meta packing passed")


def create_store(local_hostname: str = None):
    config = HiCacheStorageConfig(
        is_mla_model=True,
        tp_rank=0,
        tp_size=1,
        model_name=None,
        is_page_first_layout=False,
    )
    if local_hostname is not None:
        config.extra_config = {
            "master_server_address": os.environ["MOONCAKE_MASTER"],
            "metadata_server": os.environ["MOONCAKE_TE_META_DATA_SERVER"],
            "local_hostname": local_hostname,
            "protocol": os.environ.get("MOONCAKE_PROTOCOL", "rdma"),
            "device_name": os.environ.get("MOONCAKE_DEVICE", ""),
            "global_segment_size": os.environ.get("MOONCAKE_GLOBAL_SEGMENT_SIZE", "900gb"),
            "check_server": False,
        }
    return MooncakeStore(config)


def test_host_buffer_registration_matrix():
    print("=" * 100)
    print("Testing host buffer registration matrix")

    unique_host = f"{_get_mooncake_test_ip()}:{12001 + (os.getpid() % 1000)}"
    store = create_store(unique_host)

    plain_torch = torch.empty(1024 * 1024, dtype=torch.uint8, device="cpu")
    plain_numpy = np.empty(1024 * 1024, dtype=np.uint8)
    tensors = [
        ("plain_torch", plain_torch.data_ptr(), plain_torch.numel() * plain_torch.element_size()),
        ("plain_numpy", plain_numpy.ctypes.data, int(plain_numpy.nbytes)),
    ]

    if hasattr(torch.Tensor, "pin_memory"):
        try:
            pinned_torch = torch.empty(1024 * 1024, dtype=torch.uint8, device="cpu").pin_memory()
            tensors.append(
                (
                    "pinned_torch",
                    pinned_torch.data_ptr(),
                    pinned_torch.numel() * pinned_torch.element_size(),
                )
            )
        except Exception as e:
            logger.info(f"skip pinned_torch registration case: {e}")
            pinned_torch = None
    else:
        pinned_torch = None

    pool = create_mock_layer_first_mla_pool(page_num=8, page_size=16, layer_num=2, dtype=torch.float16)
    chunk_tensor = pool.buffers.k_buffer_chunks[0][0]
    page_slice_tensor = pool.buffers.k_buffer[0][0]
    tensors.extend(
        [
            (
                "mock_host_pool_chunk",
                chunk_tensor.data_ptr(),
                chunk_tensor.numel() * chunk_tensor.element_size(),
            ),
            (
                "mock_host_pool_page_slice",
                page_slice_tensor.data_ptr(),
                page_slice_tensor.numel() * page_slice_tensor.element_size(),
            ),
        ]
    )

    try:
        from sglang.srt.mem_cache.memory_pool.host_memory_pool import HostBufferSpec
        from sglang.srt.mem_cache.memory_pool.host_memory_pool_dsa import HostDSABufferLayout
        from sglang.srt.mem_cache.memory_pool.host_memory_pool_mla import HostMLABufferLayoutLayerFirstSplitKV

        spec = HostBufferSpec(
            size=128,
            page_num=8,
            page_size=16,
            layer_num=2,
            dtype=torch.float16,
            pin_memory=True,
            device="cpu",
            kv_cache_dim=10,
            kv_lora_rank=4,
            qk_rope_head_dim=6,
            index_head_dim=5,
        )
        real_mla_layout = HostMLABufferLayoutLayerFirstSplitKV(io_backend="direct")
        real_mla_buffers = real_mla_layout.alloc_buffers(spec)
        real_dsa_buffers = HostDSABufferLayout(real_mla_layout).alloc_buffers(spec)
        tensors.extend(
            [
                (
                    "real_mla_k_chunk",
                    real_mla_buffers.k_buffer_chunks[0][0].data_ptr(),
                    real_mla_buffers.k_buffer_chunks[0][0].numel() * real_mla_buffers.k_buffer_chunks[0][0].element_size(),
                ),
                (
                    "real_mla_k_page",
                    real_mla_buffers.k_buffer[0][0].data_ptr(),
                    real_mla_buffers.k_buffer[0][0].numel() * real_mla_buffers.k_buffer[0][0].element_size(),
                ),
                (
                    "real_dsa_index_chunk",
                    real_dsa_buffers.index_k_buffer_chunks[0][0].data_ptr(),
                    real_dsa_buffers.index_k_buffer_chunks[0][0].numel() * real_dsa_buffers.index_k_buffer_chunks[0][0].element_size(),
                ),
                (
                    "real_dsa_index_page",
                    real_dsa_buffers.index_k_buffer_page_slices[0][0].data_ptr(),
                    real_dsa_buffers.index_k_buffer_page_slices[0][0].numel() * real_dsa_buffers.index_k_buffer_page_slices[0][0].element_size(),
                ),
            ]
        )
    except Exception as e:
        logger.info(f"skip real host pool allocation cases: {e}")

    failed_cases = []
    for name, ptr, size in tensors:
        ret = store.store.register_buffer(ptr, size)
        logger.info(f"register case {name}: ptr=0x{ptr:x}, size={size}, ret={ret}")
        if ret == 0:
            store.store.unregister_buffer(ptr)
        else:
            failed_cases.append((name, ret, size))

    logger.info(f"host registration failures: {failed_cases}")
    return failed_cases


if __name__ == "__main__":
    test_layer_first_registration_target_diff()
    test_host_buffer_registration_matrix()
    test_layer_first_meta_pack()
    test_layer_first_dsa_meta_pack()
    test_single_operation()
    test_batch_operation(
        HiCacheStorageConfig(
            is_mla_model=False,
            tp_rank=0,
            tp_size=1,
            model_name=None,
            is_page_first_layout=True,
        )
    )
    test_batch_operation(
        HiCacheStorageConfig(
            is_mla_model=True,
            tp_rank=0,
            tp_size=1,
            model_name=None,
            is_page_first_layout=True,
        )
    )
    test_batch_operation(
        HiCacheStorageConfig(
            is_mla_model=False,
            tp_rank=1,
            tp_size=4,
            model_name=None,
            is_page_first_layout=True,
        )
    )
    test_batch_operation(
        HiCacheStorageConfig(
            is_mla_model=True,
            tp_rank=3,
            tp_size=8,
            model_name=None,
            is_page_first_layout=True,
        )
    )
    logger.info("✅ All tests passed")
    print("all passed")
