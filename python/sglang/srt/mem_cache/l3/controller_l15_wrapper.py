from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.srt.layers.attention.dp_attention import get_attention_tp_rank
from sglang.srt.mem_cache.hicache_operation import H2SWrite, S2HLoad
from sglang.srt.mem_cache.l3.controller import L3QueryResult
from sglang.srt.mem_cache.l3.storage import HiCacheStorageExtraInfo
from sglang.srt.utils import get_colorful_logger

if TYPE_CHECKING:
    from sglang.srt.mem_cache.l3.controller import L3CacheController

logger = get_colorful_logger(__name__)


class L3CacheControllerL15Wrapper:
    def __init__(self, inner: L3CacheController, is_mooncake: bool = True):
        self._inner = inner
        self._is_mooncake = is_mooncake
        self._inner.sync_tp_fn = self.sync_tp_l15
        self._inner.page_get_zero_copy_fn = self.page_get_zero_copy_l15
        self._inner.storage_hit_query_local_fn = self.storage_hit_query_local
        self._inner.sync_storage_hit_query_fn = self.sync_storage_hit_query

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def get_local_token_indices(self, global_host_page: torch.Tensor):
        local_host_page, local_page_mask = self._inner.mem_pool_host.global_pages_to_local_page_indices(global_host_page)
        local_token_indices = self._inner.mem_pool_host.pages_to_tokens(local_host_page)
        return local_token_indices, local_page_mask

    def get_local_page_hashes(self, hash_value, local_page_mask) -> list:
        page_hashes = hash_value if hash_value is not None else []
        return [h for h, m in zip(page_hashes, local_page_mask.tolist()) if m]

    def add_h2s_write(self, node, prefix_keys=None, pp_prefix=None, layer_range=None) -> H2SWrite:
        local_token_indices, local_page_mask = self.get_local_token_indices(node.host_value)
        local_page_hashes = self.get_local_page_hashes(node.hash_value, local_page_mask)
        logger.debug(f"[add_h2s_write] num_pages={len(node.host_value)} num_tokens={len(local_token_indices)}")
        return self._inner.enqueue_h2s_write(node, local_token_indices, local_page_hashes, prefix_keys, pp_prefix, layer_range)

    def create_s2h_load(self, rid, tokens, last_hash, prefix_keys, last_host_node):
        s2h_load = self._inner.create_s2h_load(rid, tokens, last_hash, prefix_keys, last_host_node)
        return s2h_load

    def storage_hit_query_local(self, tokens, last_hash, prefix_keys):
        inner = self._inner
        prefix_keys_copy = prefix_keys.copy() if prefix_keys else None

        hit_pages = 0
        hash_value = []
        page2rank: list[int] = []
        storage_batch_size = inner.storage_batch_size * 10  # tmp trick for lower reduce cost
        query_pages = (len(tokens) + inner.page_size - 1) // inner.page_size
        storage_batch_size = max(storage_batch_size, query_pages)
        for start in range(0, len(tokens), inner.page_size * storage_batch_size):
            end = min(start + inner.page_size * storage_batch_size, len(tokens))
            batch_tokens = tokens[start:end]
            batch_hashes = []
            for i in range(0, len(batch_tokens), inner.page_size):
                last_hash = inner.get_hash_str(batch_tokens[i : i + inner.page_size], last_hash)
                batch_hashes.append(last_hash)

            extra_info = HiCacheStorageExtraInfo(prefix_keys=prefix_keys_copy)

            if self._is_mooncake:
                monitor = inner.l3_status_monitor
                token = monitor.begin_operation()
                if token is None:
                    hit_in_batch = 0
                    logger.warning("[L3-Status-Skip] Mooncake L1.5 query is unavailable")
                else:
                    try:
                        hit_in_batch = inner.storage_backend.batch_exists(batch_hashes, extra_info)
                        if not isinstance(hit_in_batch, int) or not (0 <= hit_in_batch <= len(batch_hashes)):
                            raise RuntimeError(f"invalid batch_exists result: {hit_in_batch!r}")
                    except Exception:
                        unavailable = monitor.report_failure(token)
                        logger.exception(
                            "[L3-L1.5-Query-Failed] treating batch as miss%s",
                            "; Mooncake marked unavailable" if unavailable else "",
                        )
                        hit_in_batch = 0
                    else:
                        if monitor.report_success(token):
                            logger.info("[L3-Status-Recovered] Mooncake L1.5 query check succeeded")
                local_hit_pages = hit_in_batch
                hash_value.extend(batch_hashes[:local_hit_pages])
                hit_pages += local_hit_pages
                if local_hit_pages < len(batch_hashes):
                    break
            else:
                kvp_rank = get_attention_tp_rank()
                batch_mask = inner.storage_backend.batch_exists_mask(batch_hashes, kvp_rank, extra_info)
                if len(batch_mask) != len(batch_hashes):
                    raise RuntimeError("batch_exists_mask returned an unexpected length")
                hash_value.extend(batch_hashes)
                page2rank.extend(batch_mask)
            if prefix_keys_copy and len(prefix_keys_copy) > 0:
                prefix_keys_copy += batch_hashes

        return L3QueryResult(hash_value, hit_pages, page2rank or None)

    def sync_storage_hit_query(self, result: L3QueryResult) -> L3QueryResult:
        inner = self._inner
        inner.assert_query_coordinator_thread()
        if self._is_mooncake:
            hit_tensor = torch.tensor([result.hit_pages], dtype=torch.int)
            if inner.tp_world_size > 1:
                torch.distributed.all_reduce(
                    hit_tensor,
                    op=torch.distributed.ReduceOp.MIN,
                    group=inner.query_tp_group,
                )
            hit_pages = int(hit_tensor.item())
            tp_size = inner.tp_world_size if inner.tp_world_size > 0 else 1
            page2rank = [i % tp_size for i in range(hit_pages)]
        else:
            page2rank = result.page2rank or []
            mask_tensor = torch.tensor(page2rank, dtype=torch.int)
            if page2rank and inner.tp_world_size > 1:
                torch.distributed.all_reduce(
                    mask_tensor,
                    op=torch.distributed.ReduceOp.MAX,
                    group=inner.query_tp_group,
                )
            page2rank = mask_tensor.tolist() if page2rank else []
            hit_pages = 0
            for rank in page2rank:
                if rank == -1:
                    break
                hit_pages += 1

        if inner.pp_world_size > 1:
            hit_tensor = torch.tensor([hit_pages], dtype=torch.int)
            torch.distributed.all_reduce(
                hit_tensor,
                op=torch.distributed.ReduceOp.MIN,
                group=inner.query_pp_group,
            )
            hit_pages = int(hit_tensor.item())

        return L3QueryResult(
            result.hash_values[:hit_pages],
            hit_pages,
            page2rank[:hit_pages],
        )

    def page_get_zero_copy_l15(self, s2h_load: S2HLoad, hash_values, host_token_indices, extra_info=None):
        global_pages = host_token_indices[:: self.page_size] // self.page_size
        local_host_page, local_page_mask = self._inner.mem_pool_host.global_pages_to_local_page_indices(global_pages)
        local_hashes = self.get_local_page_hashes(hash_values, local_page_mask)
        local_token_indices = self._inner.mem_pool_host.pages_to_tokens(local_host_page)

        batch_offset = s2h_load.completed_tokens // self._inner.page_size
        s2h_load.increment(len(hash_values) * self._inner.page_size)

        if not local_hashes:
            return False

        results = self._inner.storage_backend.batch_get_zero_copy(local_hashes, local_token_indices, extra_info)
        if self._is_mooncake and (len(results) != len(local_hashes) or not all(results)):
            raise RuntimeError("L1.5 storage prefetch returned failed pages")

        result_iter = iter(results)
        for i, is_local in enumerate(local_page_mask.tolist()):
            if is_local:
                if next(result_iter, None):
                    s2h_load.page_mask[batch_offset + i] = self._inner.mem_pool_host.kvp_rank
        return True

    def sync_tp_l15(self, s2h_load):
        if s2h_load.page_mask is None:
            return 0
        global_page_mask = torch.tensor(s2h_load.page_mask, dtype=torch.int)
        if self._inner.tp_world_size > 1:
            torch.distributed.all_reduce(
                global_page_mask,
                op=torch.distributed.ReduceOp.MAX,
                group=self._inner.prefetch_tp_group,
            )
        min_completed_pages = 0
        for v in global_page_mask.tolist():
            if v == -1:
                break
            min_completed_pages += 1
        return min_completed_pages * self._inner.page_size

    def sync_l15_masks_batched(self, s2h_loads):
        inner = self._inner
        if not s2h_loads:
            return
        lengths = []
        flat = []
        for s in s2h_loads:
            mask = s.page_mask if s.page_mask is not None else []
            lengths.append(len(mask))
            flat.extend(mask)

        if len(flat) > 0 and inner.tp_world_size > 1:
            flat_tensor = torch.tensor(flat, dtype=torch.int)
            torch.distributed.all_reduce(
                flat_tensor,
                op=torch.distributed.ReduceOp.MAX,
                group=inner.prefetch_tp_group,
            )
            flat = flat_tensor.tolist()

        completed_tokens_list = []
        off = 0
        for length in lengths:
            seg = flat[off : off + length]
            off += length
            completed_pages = 0
            for v in seg:
                if v == -1:
                    break
                completed_pages += 1
            completed_tokens_list.append(completed_pages * inner.page_size)

        if inner.pp_world_size > 1:
            ct_tensor = torch.tensor(completed_tokens_list, dtype=torch.int64)
            torch.distributed.all_reduce(
                ct_tensor,
                op=torch.distributed.ReduceOp.MIN,
                group=inner.prefetch_pp_group,
            )
            completed_tokens_list = ct_tensor.tolist()

        for s, ct in zip(s2h_loads, completed_tokens_list):
            s.set_synced_completed_tokens(ct)
