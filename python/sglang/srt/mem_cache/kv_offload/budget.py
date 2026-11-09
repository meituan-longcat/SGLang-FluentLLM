"""Byte accounting for split BF16 and packed INT8 DSA KV offload, per rank.

Odd global attention IDs are offloaded; indexer and draft KV stay on device.
The padded graph hand-off cost is explained at ``padded_graph_batch_sizes``.
No capacity formula is changed here.
"""
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

HOT_TOKENS = 4096


def is_offloaded_attention(layer_id: int) -> bool:
    return layer_id % 2 == 1


def offloaded_layer_ids(start_layer: int, layer_count: int) -> tuple[int, ...]:
    return tuple(i for i in range(layer_count) if is_offloaded_attention(start_layer + i))


def packed_coordinator_fixed_bytes(
    batch_size: int, queries: int, topk: int, page_size: int, *, last_query=False,
) -> int:
    """Eager INT32 storage outside hot pools, workspace and graph growth.

    Padding views share one maximum TopK tensor and one lengths tensor.
    The selection page table and per-request step each have their own storage.
    """
    if min(batch_size, queries, topk, page_size) <= 0:
        raise ValueError("Packed Coordinator buffers require positive geometry")
    padding_elements = (batch_size - 1) * (queries * topk + 1)
    table_elements = batch_size * queries * ((topk + page_size - 1) // page_size)
    # S1 reuses the original contiguous table prefix. Each other supported S
    # owns one contiguous [B,K/page] last-query row map, outside graph capture.
    last_query_elements = ((queries - 1) * batch_size * ((topk + page_size - 1) // page_size)
                           if last_query else 0)
    return (padding_elements + table_elements + batch_size + last_query_elements) * 4


def packed_offload_workspace_bytes(
    batch_size: int,
    queries: int,
    topk: int,
    page_size: int,
    packed_row_bytes: int,
    cache_tokens: int = HOT_TOKENS,
) -> int:
    """Conservative single-step reserve for the packed hot-cache kernels.

    Keep two selections live, a functional hot-pool output and its alias-safe
    staging, plus the output metadata and UpdateCache's metadata scratch.
    This excludes resident per-layer hot pools and graph arenas: the latter
    remain covered by the separately calibrated graph reserve.
    """
    if min(batch_size, queries, topk, page_size, packed_row_bytes, cache_tokens) <= 0:
        raise ValueError("Packed offload workspace requires positive geometry")
    selected_records = batch_size * queries * ((topk + page_size - 1) // page_size * page_size)
    hot_records = batch_size * cache_tokens
    return (
        2 * selected_records * packed_row_bytes
        + 2 * hot_records * packed_row_bytes
        + hot_records * 8  # functional status/age outputs
        + hot_records * 24  # two status/age/resolved workspace regions
        + 3 * batch_size * 8192 * 4  # UpdateCache eviction scratch
        + (16 << 20)  # GatherByCacheIndices workspace
        + (32 << 20)  # UpdateCache system reserve
    )


@dataclass(frozen=True)
class OffloadBudget:
    layer_count: int
    kv_bytes_per_token_layer: int
    index_bytes_per_token_layer: int
    batch_size: int
    draft_layers: int = 0
    start_layer: int = 0
    hot_tokens: int = HOT_TOKENS
    page_size: int = 128
    workspace_bytes: int = 256 << 20
    other_fixed_hbm_bytes: int = 0
    draft_extra_tokens: int = 0
    host_backed: bool = True
    # Replicated L1.5 pages, beyond the ordinary padding page. capacity()
    # returns usable tokens excluding these pages; usage() receives the final
    # local allocation including them.
    reserved_pages: int = 0
    # The captured graph batch sizes (``--npu-compile-bs``). Only entries
    # strictly below ``batch_size`` carry the padded hand-off bill; see the
    # module docstring.
    compile_batch_sizes: tuple = ()
    # Measured reserve for ALL main-graph net growth after the first captured
    # graph; the first graph stays covered by the usual non-static runtime
    # reserve. Positive only, with an exact capture permutation and a runtime
    # guard. Required whenever a graph is captured below ``batch_size``.
    graph_extra_bytes: Optional[int] = None
    graph_capture_order: tuple = ()
    indexer_layer_mask: Optional[tuple] = None
    kv_layout: str = "split_bf16"
    coordinator_fixed_bytes: int = 0

    def __post_init__(self):
        mask = ((True,) * self.layer_count if self.indexer_layer_mask is None
                else tuple(self.indexer_layer_mask))
        object.__setattr__(self, 'indexer_layer_mask', mask)
        # Normalise to a deduplicated, ascending tuple so the ledger and the
        # hash are stable whatever order the CLI listed the graphs in.
        object.__setattr__(
            self, 'compile_batch_sizes', tuple(sorted(set(self.compile_batch_sizes)))
        )
        # AttnInitializer validates the CLI values these are derived from
        # before building a budget, and refuses a compile batch above
        # offload_cache_bs there, where the message can name the flags.

        object.__setattr__(self, 'graph_capture_order', tuple(self.graph_capture_order))
        if self.graph_extra_bytes is not None:
            if self.graph_extra_bytes <= 0:
                raise ValueError('calibrated graph reserve must be positive')
            if (len(self.graph_capture_order) != len(self.compile_batch_sizes)
                    or set(self.graph_capture_order) != set(self.compile_batch_sizes)):
                raise ValueError('calibrated graph reserve requires an exact capture permutation')
        elif self.graph_capture_order:
            raise ValueError('custom capture order requires a calibrated graph reserve')

    @property
    def hot_managed_layers(self):
        return len(offloaded_layer_ids(self.start_layer, self.layer_count))

    @property
    def offloaded_layers(self):
        return self.hot_managed_layers if self.host_backed else 0

    @property
    def hbm_bytes_per_token(self):
        return ((self.layer_count - self.offloaded_layers) * self.kv_bytes_per_token_layer
                + sum(self.indexer_layer_mask) * self.index_bytes_per_token_layer
                + self.draft_layers * (self.kv_bytes_per_token_layer + self.index_bytes_per_token_layer))

    @property
    def indexer_placeholder_bytes(self):
        return (self.layer_count - sum(self.indexer_layer_mask)) * self.page_size * self.index_bytes_per_token_layer

    @property
    def host_bytes_per_token(self):
        return self.offloaded_layers * self.kv_bytes_per_token_layer

    @property
    def hot_bytes(self):
        # Split NoPE/RoPE or a whole packed row, plus int32 tags and ages.
        return self.hot_managed_layers * self.batch_size * self.hot_tokens * (self.kv_bytes_per_token_layer + 8)

    @property
    def packed_hot_bytes(self):
        return self.hot_bytes if self.kv_layout == "packed_int8" else 0

    @property
    def padded_graph_batch_sizes(self):
        """Captured graphs that pay for the padded hand-off.

        ``hot_cache_operands`` returns early at ``bs == offload_cache_bs``, so a
        graph captured at exactly ``batch_size`` is unchanged by the fix -
        measured: its capture cost is identical on both arms.
        """
        if self.hot_managed_layers == 0:
            return ()
        return tuple(bs for bs in self.compile_batch_sizes if bs < self.batch_size)

    @property
    def padded_graph_count(self):
        return len(self.padded_graph_batch_sizes)

    @property
    def padded_graph_bytes(self):
        """Zero when every graph is captured at ``batch_size``; see above."""
        return self.graph_extra_bytes or 0

    @property
    def fixed_hbm_bytes(self):
        return (self.hot_bytes + self.workspace_bytes + self.other_fixed_hbm_bytes
                + self.coordinator_fixed_bytes
                + self.indexer_placeholder_bytes
                + self.padded_graph_bytes
                + self.draft_extra_tokens * self.draft_layers * (self.kv_bytes_per_token_layer + self.index_bytes_per_token_layer)
                + self.page_size * self.hbm_bytes_per_token)

    def usage(self, tokens: int) -> dict:
        return {'hbm_bytes': self.fixed_hbm_bytes + tokens * self.hbm_bytes_per_token,
                'host_bytes': (tokens + self.page_size) * self.host_bytes_per_token}

    def capacity(self, hbm_budget_bytes: int, host_budget_bytes: int) -> int:
        by_hbm = (hbm_budget_bytes - self.fixed_hbm_bytes) // self.hbm_bytes_per_token
        by_host = (host_budget_bytes // self.host_bytes_per_token - self.page_size
                   if self.host_bytes_per_token else by_hbm)
        usable = min(by_hbm, by_host) - self.reserved_pages * self.page_size
        return max(0, usable // self.page_size * self.page_size)

    def describe(self):
        return {**asdict(self), 'offloaded_layers': self.offloaded_layers, 'hot_managed_layers': self.hot_managed_layers,
                'active_indexer_layers': sum(self.indexer_layer_mask),
                'indexer_placeholder_bytes': self.indexer_placeholder_bytes,
                'resident_kv_layers': self.layer_count - self.offloaded_layers,
                'hbm_bytes_per_token': self.hbm_bytes_per_token,
                'host_bytes_per_token': self.host_bytes_per_token,
                'hot_bytes': self.hot_bytes,
                'packed_hot_bytes': self.packed_hot_bytes,
                'padded_graph_batch_sizes': list(self.padded_graph_batch_sizes),
                'padded_graph_count': self.padded_graph_count,
                'padded_graph_bytes': self.padded_graph_bytes,
                'fixed_hbm_bytes': self.fixed_hbm_bytes}


def _inactive_file_reclaim_estimate(cgroup: Path, limit: int, *, v1: bool) -> int:
    """Conservative file-cache estimate, not a guarantee for DMA-pinned pages.

    The existing fixed cgroup paths must refer to the container's memory
    controller. Do not count active cache, tmpfs/shmem or mapped/dirty pages.
    GUP/FOLL_PIN is not fully represented by these counters; runtime ownership
    and registered-file mappings still need to satisfy that assumption.
    """
    try:
        stats = dict(line.split() for line in (cgroup / 'memory.stat').read_text().splitlines())
        if v1:
            # A stricter ancestor needs its own usage; a child's counters alone
            # cannot account for siblings. Keep the previous strict headroom.
            if int(stats['hierarchical_memory_limit']) != limit:
                return 0
            keys = ('total_inactive_file', 'total_cache', 'total_shmem',
                    'total_dirty', 'total_writeback', 'total_mapped_file', 'total_unevictable')
        else:
            keys = ('inactive_file', 'file', 'shmem', 'file_dirty',
                    'file_writeback', 'file_mapped', 'unevictable')
        inactive, file, shmem, dirty, writeback, mapped, unevictable = (
            int(stats[key]) for key in keys
        )
        if min(inactive, file, shmem, dirty, writeback, mapped, unevictable) < 0:
            return 0
        # Exclusions may overlap; subtracting twice is deliberately conservative.
        return max(0, min(inactive, max(0, file - shmem))
                   - dirty - writeback - mapped - unevictable)
    except (OSError, ValueError, KeyError):
        return 0


def available_host_bytes() -> int:
    """Cap MemAvailable by conservative cgroup headroom including cold files."""
    mem = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
    available = int(mem['MemAvailable'].split()[0]) * 1024
    for limit_path, usage_path in [('/sys/fs/cgroup/memory.max', '/sys/fs/cgroup/memory.current'),
                                   ('/sys/fs/cgroup/memory/memory.limit_in_bytes', '/sys/fs/cgroup/memory/memory.usage_in_bytes')]:
        try:
            limit = Path(limit_path).read_text().strip()
            if limit != 'max':
                limit = int(limit)
                usage = int(Path(usage_path).read_text())
                if min(limit, usage) < 0:
                    return 0
                reclaim = min(usage, _inactive_file_reclaim_estimate(
                    Path(limit_path).parent, limit, v1=limit_path.endswith('memory.limit_in_bytes')
                ))
                available = min(available, max(0, limit - usage + reclaim))
        except FileNotFoundError:
            pass
    return available
