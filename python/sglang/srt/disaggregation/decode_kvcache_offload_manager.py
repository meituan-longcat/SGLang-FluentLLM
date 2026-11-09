from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

from sglang.srt.mem_cache.utils import get_hash_str

if TYPE_CHECKING:
    from sglang.srt.managers.req import Req
    from sglang.srt.mem_cache.hicache_engine import HiCacheEngine
    from sglang.srt.mem_cache.hicache_operation import H2SWrite

logger = logging.getLogger(__name__)


class OffloadNode:
    """Lightweight stand-in for a radix TreeNode. It carries everything the L2/L3
    but is NEVER inserted into the radix tree.

    Carries:
    - hash_value      : per-page hash chain (storage keys, read by prefill).
    - host_value      : the transit host pages, freed once backup completes.
    - host_ref_counter: 1 per in-flight pp{i} H2SWrite; back to 0 == backup done.
    - req             : the finished request, so D2H completion can release its
                        device KV + req_pool slot + prefix lock.
    """

    counter = 0

    def __init__(self, hash_value: list[str], req: Req):
        OffloadNode.counter += 1
        self.id = OffloadNode.counter
        self.hash_value = hash_value
        self.req = req
        self.host_value: torch.Tensor | None = None
        self.host_ref_counter = 0
        # Bypass node is never in the radix tree, so it has no parent and no
        # prefix-hash chain.
        self.parent = None
        self.lock_ref = 0

    def get_prefix_hash_values(self, parent):
        return None

    def protect_host(self):
        self.host_ref_counter += 1

    def release_host(self):
        if self.host_ref_counter > 0:
            self.host_ref_counter -= 1
        else:
            raise RuntimeError("Host reference counter is already zero.")


class DecodeKVCacheOffloadManager:
    """Decode-side KV cache offload to L3 storage via a bypass that does NOT go
    through the radix tree logic.

    - Triggered only when a request is finished (no mid-decode increments), so
      there is no race with slots the request is still attending.
    - Reuses the existing HiCacheEngine's l2_ctrl / l3_ctrl and host pool (no
      second controller / host pool), but feeds them an OffloadNode placeholder
      so nothing is inserted into the radix tree.
    - Host pages are transit-only: allocated for the D2H staging, and freed as
      soon as the L3 backup of that node completes (not retained as L2 cache).
    """

    def __init__(self, hicache_engine: HiCacheEngine, kv_manager):
        self.hicache_engine = hicache_engine
        self.d2h_controller = hicache_engine.l2_ctrl
        self.h2s_controller = hicache_engine.l3_ctrl
        self.page_size = hicache_engine.page_size
        self.token_to_kv_pool_allocator = hicache_engine.token_to_kv_pool_allocator
        self.req_to_token_pool = hicache_engine.req_to_token_pool

        self.kv_manager = kv_manager
        hicache_engine.prefill_kv_manager = kv_manager
        hicache_engine.decode_offload_bypass_l2 = True
        self.inflight_host_pages = 0
        logger.info("DecodeKVCacheOffloadManager enabled (bypass, finish-then-offload)")

    def cache_finished_req(self, req: Req, delay_req_pool_release: bool = False):
        if self.offload_finished_req(req):
            return
        self.release_device(req, delay_req_pool_release=delay_req_pool_release)

    def offload_finished_req(self, req: Req) -> bool:
        assert self.h2s_controller is not None
        assert req.req_pool_idx is not None

        token_ids = req.origin_input_ids + req.output_ids
        seq_len = len(token_ids) - 1
        full_page_num = seq_len // self.page_size
        prefill_pages = len(req.origin_input_ids) // self.page_size
        if full_page_num <= prefill_pages:
            return False

        device_pages = self.token_to_kv_pool_allocator.req_to_page[req.req_pool_idx, prefill_pages:full_page_num].clone()
        if device_pages.numel() == 0:
            return False

        prefix = req.last_node.get_prefix_hash_values(req.last_node) if req.last_node is not None else []
        page_hashes: list[str] = list(prefix[prefill_pages:full_page_num])
        prior = prefix[-1] if prefix else None
        for p in range(len(prefix), full_page_num):
            prior = get_hash_str(token_ids[p * self.page_size : (p + 1) * self.page_size], prior)
            if p >= prefill_pages:
                page_hashes.append(prior)
        node = OffloadNode(hash_value=page_hashes, req=req)
        page2rank = self.d2h_controller.device_kv_pool.get_ranks_from_global_pages(device_pages)
        host_pages = self.alloc_host_pages(page2rank)
        if host_pages is None:
            logger.error(f"[decode-offload] no host memory for req {req.rid}")
            return False

        node.host_value = host_pages
        self.inflight_host_pages += len(host_pages)
        logger.debug(
            f"[offload_finished_req] Added a D2H Offload: {node.req.rid=} {len(device_pages)=} {len(host_pages)=} {prefill_pages=} {full_page_num=}"
        )
        self.d2h_controller.write(
            device_page_indices=device_pages,
            host_page_indices=host_pages,
            node=node,
        )
        return True

    def alloc_host_pages(self, page2rank) -> torch.Tensor | None:
        # Decode-Side Host Mem Cache bypasses Radix Tree, no eviction needed.
        host_kv_allocator = self.d2h_controller.host_kv_allocator
        return host_kv_allocator.rank_alloc_pages(page2rank)

    def has_inflight(self) -> bool:
        d2h = self.d2h_controller
        if d2h.d2h_write_q_todo or d2h.d2h_write_q_ongoing:
            return True
        h2s = self.h2s_controller
        return bool(h2s is not None and h2s.h2s_write_inflight > 0)

    def tick(self):
        finish_counts = []
        finish_counts.append(self.hicache_engine.check_l2_writing_progress())
        finish_counts.extend(self.hicache_engine.check_l3_progress())
        if self.hicache_engine.tp_pp_size > 1:
            finish_counts_tensor = torch.tensor(finish_counts, dtype=torch.int, device="cpu")
            torch.distributed.all_reduce(
                finish_counts_tensor,
                op=torch.distributed.ReduceOp.MIN,
                group=self.hicache_engine.tp_pp_group,
            )
            finish_counts = finish_counts_tensor.tolist()
        logger.debug(f"[decode offloader tick] {finish_counts=}")
        self.process_finished_d2h_decode(finish_counts[0])
        self.process_finished_h2s_decode(finish_counts[1:][0])

    def process_finished_d2h_decode(self, finish_cnt: int):
        while finish_cnt > 0:
            w = self.d2h_controller.d2h_write_q_ongoing.pop(0)
            for node in w.nodes:
                # immediately release device cause d2h is done
                # need to make sure the decode-side incremental
                # pages is not in radix tree (do not insert when
                # cache finished)
                self.release_device(node.req)
                self.hicache_engine.add_h2s_write(node)
            finish_cnt -= 1

    def process_finished_h2s_decode(self, finish_cnt: int):
        for _ in range(finish_cnt):
            h2s_write: H2SWrite = self.h2s_controller.h2s_write_q_done.get()
            self.h2s_controller.complete_h2s_write()
            if h2s_write.node is not None:
                h2s_write.node.release_host()
                # free host pages immediately
                if h2s_write.node.host_ref_counter == 0:
                    self.inflight_host_pages -= len(h2s_write.node.host_value)
                    self.d2h_controller.host_kv_allocator.free_pages(h2s_write.node.host_value)

    def release_non_prefix(self, req: Req, delay_req_pool_release: bool = False):
        assert req.req_pool_idx is not None
        allocator = self.token_to_kv_pool_allocator
        alloced_len = self.req_to_token_pool.alloced_lens[req.req_pool_idx].item()

        page_size = self.page_size
        token_ids = req.origin_input_ids + req.output_ids
        seq_len = len(token_ids) - 1
        full_page_num = seq_len // page_size
        paged_token_ids = [tuple(token_ids[i * page_size : (i + 1) * page_size]) for i in range(full_page_num)]
        matched_pages = 0
        if paged_token_ids:
            matched = self.hicache_engine.match_prefix(paged_token_ids)
            matched_pages = len(matched.device_indices)
            if matched_pages > 0:
                own_pages = allocator.req_to_page[req.req_pool_idx, :matched_pages]
                allocator.free_with_diff(matched.device_indices.to(own_pages.device), own_pages)
        cached_len = matched_pages * page_size
        allocator.free_extra_pages_not_cached(req.req_pool_idx, cached_len, alloced_len)
        allocator.free_group_flush()
        if not delay_req_pool_release:
            self.req_to_token_pool.free(req.req_pool_idx)
            req.req_pool_idx = None

    # TODO: delay_req_pool_release is reserved for retraction, but right now it's useless.
    def release_device(self, req: Req, delay_req_pool_release: bool = False):
        self.release_non_prefix(req, delay_req_pool_release=delay_req_pool_release)
        if req.last_node is not None:
            self.hicache_engine.dec_lock_ref(req.last_node)
            req.last_node = None
