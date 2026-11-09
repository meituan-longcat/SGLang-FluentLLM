from sglang.srt.mem_cache.memory_pool.host_kv_allocator import HostKVAllocator  # noqa: F401
from sglang.srt.mem_cache.memory_pool.kv_allocator import (  # noqa: F401
    KVAllocator,
    PagesInfo,
)
from sglang.srt.mem_cache.memory_pool.memory_pool import *  # noqa: F401, F403
from sglang.srt.mem_cache.memory_pool.memory_pool import __all__  # noqa: F401


def create_host_token_to_kv_pool(
    token_to_kv_pool,
    page_size: int,
    hicache_ratio: float,
    hicache_size: int,
    hicache_mem_layout: str,
    hicache_io_backend: str,
    extra_layer_num: int = 0,
    host_size_tokens: int = 0,
):
    from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool, MHATokenToKVPool, MLATokenToKVPool
    from sglang.srt.mem_cache.memory_pool.host_memory_pool_dsa import HostDSATokenToKVPool
    from sglang.srt.mem_cache.memory_pool.host_memory_pool_l15_wrapper import HostTokenToKVPoolL15Wrapper
    from sglang.srt.mem_cache.memory_pool.host_memory_pool_mha import HostMHATokenToKVPool
    from sglang.srt.mem_cache.memory_pool.host_memory_pool_mla import HostMLATokenToKVPool
    from sglang.srt.mem_cache.memory_pool.memory_pool_l15_wrapper import TokenToKVPoolL15Wrapper

    real_pool = token_to_kv_pool._inner if isinstance(token_to_kv_pool, TokenToKVPoolL15Wrapper) else token_to_kv_pool

    if isinstance(real_pool, MHATokenToKVPool):
        pool = HostMHATokenToKVPool(
            token_to_kv_pool,
            hicache_ratio,
            hicache_size,
            page_size,
            hicache_mem_layout,
            io_backend=hicache_io_backend,
            extra_layer_num=extra_layer_num,
            host_size_tokens=host_size_tokens,
        )
    elif isinstance(real_pool, DSATokenToKVPool):
        pool = HostDSATokenToKVPool(
            token_to_kv_pool,
            hicache_ratio,
            hicache_size,
            page_size,
            hicache_mem_layout,
            io_backend=hicache_io_backend,
            extra_layer_num=extra_layer_num,
        )
    elif isinstance(real_pool, MLATokenToKVPool):
        pool = HostMLATokenToKVPool(
            token_to_kv_pool,
            hicache_ratio,
            hicache_size,
            page_size,
            hicache_mem_layout,
            io_backend=hicache_io_backend,
            extra_layer_num=extra_layer_num,
        )
    else:
        raise ValueError("HiCacheEngine only supports MHA, MLA and DSA yet")

    if getattr(token_to_kv_pool, "enable_mla_l1_5_cache", False):
        pool = HostTokenToKVPoolL15Wrapper(pool)
    return pool
