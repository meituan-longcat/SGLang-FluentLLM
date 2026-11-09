import logging

import torch

from sglang.srt.configs.model_config import get_nsa_index_topk
from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE
from sglang.srt.distributed import get_attn_tp_group
from sglang.srt.env import global_server_args_dict
from sglang.srt.mem_cache.kv_offload.budget import HOT_TOKENS, offloaded_layer_ids
from sglang.srt.mem_cache.kv_offload.request_init import (
    build_req_id_status,
    has_unknown_identity,
    request_identities,
)
from sglang.srt.mem_cache.kv_offload.selection_context import (
    SelectionContext,
    SelectionContextStore,
)

#from sglang.srt.managers.schedule_batch import Req
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.npu.packed_offload_runtime import get_packed_offload_ops
from sglang.srt.npu.sfa1_runtime import get_sfa1_ops
from sglang.srt.npu.utils import KV_OFFLOAD_STREAM, npu_limit_core, npu_stream_switch, npu_wait_tensor
from sglang.srt.utils import is_npu
from sglang.srt.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter

__is_npu__ = is_npu()
logger = logging.getLogger(__name__)

def build_hot_cache_pads(max_batch_size, queries, topk, device):
    """Build inert padding once, outside captured graphs.

    Padding uses topk=-1 and lengths=0, so Gen/Gather skip it and
    UpdateCache admits no rows. Never give a padding row a positive length.
    All batch sizes share immutable prefixes of the maximum padding storage.
    """
    if max_batch_size <= 1:
        return {}
    topk_pad = torch.full(
        (max_batch_size - 1, queries, topk), -1,
        dtype=torch.int32, device=device,
    )
    length_pad = torch.zeros(
        (max_batch_size - 1,), dtype=torch.int32, device=device,
    )
    return {
        bs: (
            topk_pad[: max_batch_size - bs],
            length_pad[: max_batch_size - bs],
        )
        for bs in range(1, max_batch_size)
    }


def pad_block_table(block_table, rows):
    """Pad the full-pool table to the same batch size as Gen's operands.

    Zero-length padding requests never read these page-zero entries.
    """
    bs = block_table.shape[0]
    if bs >= rows:
        return block_table
    return torch.cat(
        [
            block_table,
            block_table.new_zeros((rows - bs,) + tuple(block_table.shape[1:])),
        ],
        dim=0,
    )


def select_offload_kv_metadata(forward_batch):
    """Match topk ids to their block table and KV-length index space.

    DCP topk ids are rank-local, with -1 for tokens owned by another rank.
    Pair them with kvp_block_table and kvp_context_lens_tensor; using global
    metadata would silently resolve different tokens.
    """
    meta = forward_batch.attn_metadata
    if global_server_args_dict.get("kvp_size", 1) <= 1:
        return meta.block_table, meta.seq_lens_tensor.to(torch.int32)

    block_table = getattr(meta, "kvp_block_table", None)
    kv_lens = getattr(meta, "kvp_context_lens_tensor", None)
    assert block_table is not None and kv_lens is not None, (
        "kvp_size > 1 but attn_metadata carries no kvp_block_table / "
        "kvp_context_lens_tensor; the offload chain would receive global "
        "metadata against rank-local topk ids"
    )
    return block_table, kv_lens.to(torch.int32)


def dcp_history_only_write_slots(topk, write_slots, kv_lens, q_lens, kvp_rank):
    """Do not persist KVP0's temporary MTP ids as rank-local history ids.

    KVP0 temporarily appends *all* current queries after its local history.
    Across a page/owner boundary only some of those tokens become KVP0
    history, so the same numeric id can refer to a different token next step.
    Mask LRU admission only: attention still consumes every selected current
    KV, and historical entries keep their original selection-buffer offsets.
    """
    if kvp_rank != 0:
        return write_slots
    history = kv_lens.to(write_slots.dtype) - q_lens.to(write_slots.dtype)
    padding = topk.shape[0] - history.shape[0]
    assert padding >= 0 and topk.shape == write_slots.shape
    if padding:
        history = torch.cat([history, history.new_zeros(padding)])
    return write_slots.masked_fill(topk >= history[:, None, None], -1)


def dcp_align_current_token_window(kv_lens, actual_seq_qlen, kvp_rank):
    """Place the operators' current-token window after rank-local history.

    Gen/Gather interpret [kv_len-q_len, kv_len) as current tokens. Only
    KVP rank 0 includes current queries in its local length. Other ranks
    therefore reserve q_len positions after history, which their topk ids
    cannot select. Preserve kv_len=0 so empty owners remain invalid and
    receive no cross-rank attention weight.
    """
    if kvp_rank == 0:
        return kv_lens
    # Elementwise, no host sync: safe to trace.
    return torch.where(
        kv_lens > 0, kv_lens + actual_seq_qlen.to(kv_lens.dtype), kv_lens
    )


def group_offload_topk_indices(topk_indices, batch_size):
    """Group TND index rows by request without a GE Reshape node.

    TorchAir/GE mis-infers the reshape that follows the DCP indexer in the full
    graph even though the same shape works in isolation.  The static split and
    stack is value-equivalent and keeps the request/query grouping explicit.
    """
    rows_per_request = topk_indices.shape[0] // batch_size
    rows = topk_indices.squeeze(1).split(rows_per_request, dim=0)
    return torch.stack(rows, dim=0)


def build_last_query_update_tables(table, max_batch_size, max_queries):
    """Freeze each query layout's final row map before graph capture.

    Selection still contains every query. Update's S=1 must resolve through
    the original last-query pages, not the first B rows of the full table.
    The S=1 entry shares its contiguous prefix; other entries own a small copy.
    """
    return {
        queries: table[:max_batch_size * queries].reshape(max_batch_size, queries, -1)[:, -1, :].contiguous()
        for queries in range(1, max_queries + 1)
    }


def last_query_update_operands(topk, write_slots, table):
    """Select final-query admission metadata after original-query masking."""
    return topk[:, -1:, :].contiguous(), write_slots[:, -1:, :].contiguous(), table




class SparseCoordinator:
    def __init__(self, config, page_size, kv_lora_rank,
                 qk_rope_head_dim, layer_num, max_batch_size, speculative_num_draft_tokens, device, dtype, start_layer=0,
                 packed_row_bytes=0):
        if packed_row_bytes not in (0, 672):
            raise ValueError("Packed KV offload requires complete 672-byte rows")
        self.config = config
        # The NPU offload kernels use 128-token pages; keep their existing layout.
        self.page_size = 128 if __is_npu__ else page_size
        self.kv_lora_rank = kv_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.layer_num = layer_num
        self.start_layer = start_layer
        self.offloaded_layer_ids = offloaded_layer_ids(start_layer, layer_num)
        self._offloaded_layer_id_set = set(self.offloaded_layer_ids)
        self.dtype = dtype
        self.packed_row_bytes = packed_row_bytes
        self.packed_kv = bool(packed_row_bytes)
        self.update_policy = global_server_args_dict.get("offload_cache_update_policy", "all")
        if self.update_policy not in ("all", "last_query"):
            raise ValueError("Unknown offload cache update policy")
        self.max_decode_token = max_batch_size * speculative_num_draft_tokens if speculative_num_draft_tokens else max_batch_size
        self.max_batch_size = max_batch_size
        self.device = device
        self.index_topk = get_nsa_index_topk(config)
        self.selection_num_blocks = (self.index_topk + self.page_size - 1) // self.page_size * self.max_decode_token
        self.prev_decode_req_ids = None
        self.warned_missing_uid = False
        # TODO: batch变化时需要设置为0
        self.decode_req_step = torch.zeros((max_batch_size),dtype=torch.int32,device=device)
        self.LRU_CACHE_SIZE = HOT_TOKENS
        self.cache_status = [
            torch.full((max_batch_size, self.LRU_CACHE_SIZE), -1, dtype=torch.int32, device=device)
            if i in self.offloaded_layer_ids else None
            for i in range(layer_num)
        ]
        # cache_age 与 cache_status 一一对应，记录每个 cache 槽位的年龄，INT_MIN 表示该槽位未使用。
        self.cache_age = [
            torch.full((max_batch_size, self.LRU_CACHE_SIZE), torch.iinfo(torch.int32).min, dtype=torch.int32, device=device)
            if i in self.offloaded_layer_ids else None
            for i in range(layer_num)
        ]
        # TODO: update cache async
        self.selection_contexts = SelectionContextStore()

    def sync_request_slots(self, forward_batch: ForwardBatch):
        """Carry surviving hot-cache rows to their new physical batch slots.

        Match request instance ids, not reusable pool indices or bare rids.
        New, unknown and padding slots must not reuse another request's KV.
        Run once per main Decode/target-verify step, outside graph capture.
        """
        if not global_server_args_dict.get("offload_cache", False):
            return
        if not self.offloaded_layer_ids:
            return
        if not getattr(forward_batch.token_to_kv_pool, "offload_cache", False):
            return
        mode = forward_batch.forward_mode
        if mode.is_idle():
            # An idle batch owns no requests and carries no attn_metadata.
            return
        if not (mode.is_decode() or mode.is_target_verify()):
            # The step that consumes the hot cache is DECODE without
            # speculation and TARGET_VERIFY with it: ForwardMode.is_decode()
            # is strict equality, and the MTP main-model step is
            # TARGET_VERIFY (cuda_graph_runner.py:247, scheduler.py:2175,
            # forward_batch_info.py:755/848 all set it on the existing batch,
            # so rids survives). Guarding on is_decode() alone made this
            # method return before doing anything on every single step of an
            # MTP service - silently, which is the failure mode this whole
            # change exists to remove. Draft steps (DRAFT_EXTEND) run the 3
            # MTP layers, which are not offloaded, and prefill never reaches
            # here because its pool reports offload_cache False.
            return

        rids = getattr(forward_batch, "rids", None)
        if rids is None:
            # Never degrade to "skip the reset": that is exactly the failure
            # this method exists to remove. ForwardBatch.split_micro_batch()
            # does not propagate rids, so two-batch overlap reaches here with
            # no identity and must fail loudly rather than serve stale KV.
            raise AssertionError(
                "offload cache is on but ForwardBatch carries no rids; the hot "
                "cache cannot tell a new request from a survivor. Propagate "
                "rids (see ForwardBatch.split_micro_batch) or disable offload."
            )
        rids = list(rids)
        # Identity is the (rid, instance id) pair, never the rid alone: a
        # client may supply its own rid and may reuse it, and a reused rid in
        # a recycled slot is exactly the case this whole method exists to
        # catch. See request_identities.
        identities = request_identities(rids, getattr(forward_batch, "req_uids", None))
        if has_unknown_identity(identities) and not self.warned_missing_uid:
            self.warned_missing_uid = True
            logger.warning(
                "kv-offload: batch carries no per-request instance id "
                "(bootstrap_room); those slots are treated as new every "
                "step, so they keep no hot cache. rids=%s", rids,
            )
        if identities == self.prev_decode_req_ids:
            return

        meta = getattr(forward_batch, "attn_metadata", None)
        if meta is None or (
            global_server_args_dict.get("kvp_size", 1) > 1
            and getattr(meta, "kvp_block_table", None) is None
        ):
            # Too early in this step. The speculative worker reaches
            # init_forward_metadata from base_spec_worker.init_attn_backends
            # before get_attn_meta_npu has filled the rank-local kvp fields,
            # so the metadata the reset needs does not exist yet.
            #
            # Defer WITHOUT recording prev_decode_req_ids: the later call in
            # the same step - the one under model_runner, after attn_metadata
            # is complete - still sees the same mismatch and does the work.
            # Counted so a validation round can show deferrals and handovers
            # together instead of leaving "deferred forever" indistinguishable
            # from "never needed".
            return

        req_id_status_cpu = build_req_id_status(
            identities, self.prev_decode_req_ids, self.max_batch_size
        )
        moved = [
            (destination, source)
            for destination, source in enumerate(req_id_status_cpu)
            if source >= 0 and source != destination
        ]
        if moved:
            runs = []
            for destination_slot, source_slot in moved:
                if (runs and destination_slot == runs[-1][0] + runs[-1][2]
                        and source_slot == runs[-1][1] + runs[-1][2]):
                    runs[-1][2] += 1
                else:
                    runs.append([destination_slot, source_slot, 1])
            destination = torch.tensor(
                [pair[0] for pair in moved], dtype=torch.int64, device=self.device
            )
            source = torch.tensor(
                [pair[1] for pair in moved], dtype=torch.int64, device=self.device
            )
            # Snapshot one buffer before its writes, including slot cycles.
            # Keep every cache tensor's identity and process one layer at a time.
            self.decode_req_step.index_copy_(
                0, destination, self.decode_req_step.index_select(0, source)
            )
            for layer_id in self.offloaded_layer_ids:
                for cache in (self.cache_nope[layer_id], self.cache_rope[layer_id]):
                    if cache is not None:
                        # Materialize all source runs before any overlapping write.
                        snapshots = [cache.narrow(0, src, count).clone()
                                     for _, src, count in runs]
                        for (dst, _, count), snapshot in zip(runs, snapshots):
                            cache.narrow(0, dst, count).copy_(snapshot)
                        del snapshots, snapshot
                for cache in (
                    self.cache_status[layer_id],
                    self.cache_age[layer_id],
                ):
                    if cache is not None:
                        cache.index_copy_(0, destination, cache.index_select(0, source))

        invalid = [
            destination for destination, source in enumerate(req_id_status_cpu)
            if source < 0
        ]
        invalid.extend(range(len(req_id_status_cpu), self.max_batch_size))
        if invalid:
            index = torch.tensor(invalid, dtype=torch.int64, device=self.device)
            self.decode_req_step.index_fill_(0, index, 0)
            for layer_id in self.offloaded_layer_ids:
                self.cache_status[layer_id].index_fill_(0, index, -1)
                self.cache_age[layer_id].index_fill_(
                    0, index, torch.iinfo(torch.int32).min
                )
        self.prev_decode_req_ids = identities

    def _local_attention_id(self, layer_id, attn_idx):
        return 2 * layer_id + attn_idx - self.start_layer

    def is_attention_offloaded(self, layer_id, attn_idx):
        return self._local_attention_id(layer_id, attn_idx) in self._offloaded_layer_id_set

    def _offload_active(self, layer_id, attn_idx, forward_batch):
        """Whether this attention of this layer offloads on this batch's pool.

        The flag and the pool attribute are both implied by a coordinator
        existing at all, but the per-attention term is not, so the test stays.
        """
        return bool(
            global_server_args_dict["offload_cache"]
            and self.is_attention_offloaded(layer_id, attn_idx)
            and getattr(forward_batch.token_to_kv_pool, "offload_cache", False)
        )

    def get_selection_context(self, attn_idx, required=True):
        return self.selection_contexts.get(attn_idx, required=required)

    def has_selection_context(self, attn_idx):
        return self.get_selection_context(attn_idx, required=False) is not None

    def clear_temp_tensors(self, attn_idx=None):
        self.selection_contexts.clear(attn_idx)


    def selection_block_table(self, rows):
        """Slice only the selection table's active request-row dimension.

        SFA1 TND requires one block-table row per query-length entry. Keep the
        page-width dimension and the underlying selection KV pool unchanged.
        """
        table = self.selection_kv_block_table
        if rows > table.shape[0]:
            # Static shapes only - safe inside a captured graph, unlike a
            # value-dependent check, and this is exactly the moment (capture)
            # when the mismatch would otherwise surface as a GE tiling error.
            raise AssertionError(
                f"selection block table holds {table.shape[0]} rows "
                f"(max_decode_token) but {rows} active selection rows were "
                "requested; the offload cache BS is smaller than this graph's "
                "batch x draft tokens"
            )
        if rows == table.shape[0]:
            return table
        return table[:rows]

    def hot_cache_operands(self, layer_id, bs, topk_indices_op,
                           kv_lens=None, q_lens=None):
        """Pass whole hot-cache buffers and pad per-request operands to capacity.

        A prefix view can functionalize writes into full-tensor slice_scatter
        read-modify-write operations. Whole buffers retain the validated
        writeback contract. Padding uses topk=-1 and lengths=0, so it neither
        gathers nor admits KV. Callers narrow outputs to active selection rows
        before SFA and its cross-rank merge. Reject batches above capacity.
        """
        rows = self.max_batch_size
        if bs > rows:
            raise AssertionError(
                f"{bs} active requests but the hot cache holds {rows} "
                "(offload_cache_bs)"
            )
        cache = (
            self.cache_status[layer_id],
            self.cache_age[layer_id],
            self.cache_rope[layer_id],
            self.cache_nope[layer_id],
            self.decode_req_step,
        )
        if bs == rows:
            # Already whole: nothing to pad, and the write-back is already a
            # pure write. This is the configuration that has always been clean.
            return topk_indices_op, kv_lens, q_lens, cache
        # create_selected_buffer pre-builds one pad per captured batch; building
        # it here would be traced INTO the captured graph as a Fill per layer.
        topk_pad, len_pad = self.hot_cache_pad[bs]
        if self.update_policy == "last_query":
            topk_pad = topk_pad[:, :topk_indices_op.shape[1]]
        # UpdateCache takes no length operand at all, so its caller passes none
        # and relies on write_slot (which Gen already set to -1 for the pad
        # rows) to keep them inert.
        return (
            torch.cat([topk_indices_op, topk_pad], dim=0),
            None if kv_lens is None
            else torch.cat([kv_lens, len_pad.to(kv_lens.dtype)], dim=0),
            None if q_lens is None
            else torch.cat([q_lens, len_pad.to(q_lens.dtype)], dim=0),
            cache,
        )

    def create_selected_buffer(self, memory_saver_adapter: TorchMemorySaverAdapter):
        # create buffer
        with memory_saver_adapter.region(tag=GPU_MEMORY_TYPE_KV_CACHE):
            self.selection_kv_block_table = torch.arange(0, self.selection_num_blocks
                                                    ).reshape(self.max_decode_token, -1).to(device=self.device, dtype=torch.int32)
            if self.update_policy == "last_query":
                self.last_query_update_tables = build_last_query_update_tables(
                    self.selection_kv_block_table, self.max_batch_size,
                    self.max_decode_token // self.max_batch_size,
                )
            # Pad rows for `hot_cache_operands`, one set per possible active
            # batch size. Built here because it must happen EAGERLY, before any
            # graph capture. `-1` topk ids and `0` lengths are what make a pad
            # row provably inert - see hot_cache_operands for the chain.
            # Immutable prefix views share one allocation per operand; at
            # offload_cache_bs=4, S=4, topk=2048 this holds about 96 KiB.
            self.hot_cache_pad = build_hot_cache_pads(
                self.max_batch_size,
                self.max_decode_token // self.max_batch_size,
                self.index_topk,
                self.device,
            )
            #create cache pool
            self.cache_rope = [
                torch.zeros((self.max_batch_size, self.LRU_CACHE_SIZE, self.qk_rope_head_dim), dtype=self.dtype, device=self.device)
                if i in self.offloaded_layer_ids and not self.packed_kv else None
                for i in range(self.layer_num)
            ]
            self.cache_nope = [
                torch.zeros(
                    (self.max_batch_size, self.LRU_CACHE_SIZE, self.packed_row_bytes if self.packed_kv else self.kv_lora_rank),
                    dtype=torch.int8 if self.packed_kv else self.dtype,
                    device=self.device,
                )
                if i in self.offloaded_layer_ids else None
                for i in range(self.layer_num)
            ]

    def decode_load_kv_h2d_async(
        self,
        forward_batch: ForwardBatch,
        topk_indices: torch.Tensor,
        layer_id: int,
        attn_idx: int,
        ):
        """
        依赖npu_gather_selection_kv_cache算子进行dsa在decode阶段的kv cache的加载。actual_seq_qlen和actual_seq_lengths_kv非前缀和。
        """
        if not self._offload_active(layer_id, attn_idx, forward_batch):
            return
        self.clear_temp_tensors(attn_idx)
        enable_topk_indices_opt = 2
        with npu_stream_switch(forward_batch.can_run_with_graph, KV_OFFLOAD_STREAM):
            with npu_limit_core("offload_gen", flag=forward_batch.can_run_with_graph):
                layer_id = self._local_attention_id(layer_id, attn_idx)
                actual_seq_qlen_cumsum = forward_batch.attn_metadata.query_len_tensor.to(torch.int32)
                actual_seq_qlen = torch.diff(
                    actual_seq_qlen_cumsum,
                    prepend=torch.tensor([0], device=actual_seq_qlen_cumsum.device, dtype=actual_seq_qlen_cumsum.dtype),
                )
                # DCP: rank-local topk ids must be paired with rank-local KV
                # metadata, never the global block table (see
                # select_offload_kv_metadata).
                block_table, local_kv_lens = select_offload_kv_metadata(forward_batch)
                # DCP: only kvp_rank 0 carries the current tokens in its local
                # KV length, but the operators derive their current-token window
                # from (kv_len, q_len) on every rank. See
                # dcp_align_current_token_window.
                if global_server_args_dict.get("kvp_size", 1) > 1:
                    actual_seq_lengths_kv = dcp_align_current_token_window(
                        local_kv_lens,
                        actual_seq_qlen,
                        get_attn_tp_group().rank_in_group,
                    )
                else:
                    actual_seq_lengths_kv = local_kv_lens

                k_nope, k_pe = forward_batch.token_to_kv_pool.get_kv_buffer(layer_id + self.start_layer)
                full_kv_nope = k_nope.squeeze(dim=2)
                full_kv_rope = None if self.packed_kv else k_pe.squeeze(dim=2)

                bsz, _, index_topk = topk_indices.shape #tnd
                device = topk_indices.device
                selection_kv_block_table = self.selection_kv_block_table[:bsz]#shape=【T，seq-len】
                # shape = TND,
                #[t0,t1,t2]->t0,t2,t3 status [t0,t1,t2,t3]  status [t0,t2,t3] -> status [index] -> status []
                # batchsize=4,mtp=3,总共16token。
                # topk_indices：[16,1,2048]
                # selection_k_nope:[16,128,512]-》bf16
                # selection_k_rope:[16,128,64]
                # block_table:[4,8192]
                # actual_seq_lengths_kv:[B]
                # actual_seq_qlen:[B]
                # full_kv_nope:[1294,128,512]-》bf16
                # full_kv_rope:[1294,128,64]
                # 1、16，1，2048 steps 16 step 0 16个token
                # 2、4，1，2048 复制4份 step 0 stauts -1 step 1 基于前一步的status的 batch 0 [seq0,seq1,seq2,seq3] batch[batch0,batch1,batch2,batch3]
                #get indices
                bs = actual_seq_qlen.shape[0]
                # cache_status.fill_(-1)
                S_BLOCK_NUM = self.selection_num_blocks
                topk_indices_op = group_offload_topk_indices(topk_indices, bs)
                # Hand the hot cache over WHOLE and pad the batch instead.
                (
                    topk_indices_op, op_kv_lens, op_q_lens,
                    (cache_status, _age, cache_kr, cache_kv, _step),
                ) = self.hot_cache_operands(
                    layer_id, bs, topk_indices_op,
                    actual_seq_lengths_kv, actual_seq_qlen,
                )
                # Gen/Gather cross-check every per-request operand against
                # the topk batch, so the full-pool table pads with them.
                block_table = pad_block_table(block_table, self.max_batch_size)
                # Match Gen/Gather's padded batch and actual query extent.
                selection_kv_block_table = (
                    self.selection_block_table(topk_indices_op.shape[0] * topk_indices_op.shape[1])
                    if self.update_policy == "last_query" else self.selection_kv_block_table
                )
                kwargs = dict()
                if enable_topk_indices_opt == 2:
                    kwargs = dict(
                        full_kv_block_table=block_table
                    )
                ops = get_packed_offload_ops() if self.packed_kv else get_sfa1_ops()
                from_cache_indices, selection_actual_kv_seq, write_slot, unified_slots_in_lru_or_cache = ops.npu_gen_from_cache_indices(
                    topk_indices_op,
                    cache_status,
                    op_kv_lens,
                    op_q_lens,
                    enable_topk_indices_opt = enable_topk_indices_opt,
                    selection_topk_block_size=1,
                    full_kv_block_size=128,
                    **kwargs)

            with npu_limit_core("offload_gather", flag=forward_batch.can_run_with_graph):
                tmp_write_slot = write_slot
                if enable_topk_indices_opt == 2:
                    tmp_write_slot = unified_slots_in_lru_or_cache
                if self.packed_kv:
                    selection_k_nope, cur_kv_write_slots = ops.npu_gather_by_cache_indices_packed(
                        cache_kv, from_cache_indices, topk_indices_op, full_kv_nope,
                        block_table, selection_kv_block_table, tmp_write_slot,
                        op_kv_lens, op_q_lens,
                        selection_topk_block_size=1, selection_kv_block_num=S_BLOCK_NUM,
                        selection_kv_block_size=128, enable_slots_contiguous=enable_topk_indices_opt,
                    )
                    selection_k_rope = None
                else:
                    selection_k_rope, selection_k_nope, cur_kv_write_slots = ops.npu_gather_by_cache_indices(
                        cache_kr, cache_kv, from_cache_indices, topk_indices_op, full_kv_rope,
                        full_kv_nope, block_table, selection_kv_block_table, tmp_write_slot,
                        op_kv_lens, op_q_lens,
                        selection_topk_block_size=1, selection_kv_block_num=S_BLOCK_NUM,
                        selection_kv_block_size=128, enable_slots_contiguous=enable_topk_indices_opt,
                    )
                if bs != self.max_batch_size:
                    # Narrow the padded path back down before anything
                    # downstream sees it. SFA's batch is bsz = B * S from the
                    # SELECTION space, and the current-KV patch takes B from
                    # cur_kv_write_slots.shape, so a pad row must not survive
                    # here - see hot_cache_operands.
                    selection_actual_kv_seq = selection_actual_kv_seq[:bsz]
                    cur_kv_write_slots = cur_kv_write_slots[:bs]
                if self.packed_kv:
                    # Gen owns the selection address mapping, including causal
                    # holes. Counting valid rows cannot bound the last live slot.
                    selection_topk_indices = write_slot[:bs].reshape(bsz, 1, index_topk)
                    selection_actual_kv_seq = torch.full_like(selection_actual_kv_seq, index_topk)
                else:
                    mask = (topk_indices == -1)
                    selection_topk_indices = torch.arange(index_topk, dtype=torch.int32, device=device)\
                                            .view(1, -1).repeat(bsz, 1)
                    selection_topk_indices.masked_fill_(mask.squeeze(1), -1)
                    selection_topk_indices = selection_topk_indices.unsqueeze(1)
                    selection_k_rope = selection_k_rope.unsqueeze(dim=2)

                selection_k_nope = selection_k_nope.unsqueeze(dim=2)
                selection_actual_seq_qlen = torch.arange(1, bsz + 1, dtype=torch.int32, device=device) # shape=[T]

                selection_actual_seq_qlen = selection_actual_seq_qlen.to(torch.int32)
                selection_actual_kv_seq = selection_actual_kv_seq.to(torch.int32)

        self.selection_contexts.publish(
            SelectionContext(
                attention_id=attn_idx,
                topk_indices=selection_topk_indices,
                k_nope=selection_k_nope,
                k_rope=selection_k_rope,
                write_slot=write_slot,
                cur_kv_write_slots=cur_kv_write_slots,
                actual_seq_qlen=selection_actual_seq_qlen,
                actual_kv_seq=selection_actual_kv_seq,
                source_topk=topk_indices,
                packed_row_bytes=self.packed_row_bytes,
                cache_topk_indices=topk_indices_op,
                source_valid_chunks=forward_batch.topk_valid_chunks,
            )
        )
        return

    def decode_update_cur_kv_to_selection(
        self,
        forward_batch: ForwardBatch,
        cur_k_nope: torch.Tensor,
        cur_k_rope: torch.Tensor,
        layer_id: int,
        attn_idx: int,
        *,
        prolog_dependency: torch.Tensor = None,
        ):
        """
        依赖npu_gather_selection_kv_cache算子进行dsa在decode阶段的kv cache的加载。actual_seq_qlen和actual_seq_lengths_kv非前缀和。
        """
        if not self._offload_active(layer_id, attn_idx, forward_batch):
            return
        if self.packed_kv and get_attn_tp_group().rank_in_group != 0:
            return
        context = self.get_selection_context(attn_idx)

        B, S, S2 = context.cur_kv_write_slots.shape
        assert S == S2
        if self.packed_kv:
            packed, _ = forward_batch.token_to_kv_pool.get_kv_buffer(
                self._local_attention_id(layer_id, attn_idx) + self.start_layer
            )
            if forward_batch.can_run_with_graph:
                if prolog_dependency is None:
                    raise RuntimeError("Packed current-KV graph patch requires the Prolog output dependency")
                packed = npu_wait_tensor(packed, prolog_dependency)
            source_slots = forward_batch.attn_metadata.slot_mapping.reshape(-1).long()
            current = torch.index_select(packed.reshape(-1, self.packed_row_bytes), 0, source_slots)
            current = current.reshape(B, S, self.packed_row_bytes)
            selection = context.k_nope.squeeze(dim=2)
            table = self.selection_kv_block_table[: B * S].reshape(B, S, -1)
            context.update_cur_kv_done = get_packed_offload_ops().npu_update_cur_kv_to_selection_packed(
                current, context.cur_kv_write_slots, selection, table,
            )
            if forward_batch.can_run_with_graph:
                context.k_nope = npu_wait_tensor(context.k_nope, context.update_cur_kv_done)
            return
        cur_k_nope = cur_k_nope.reshape(B, S, cur_k_nope.shape[-1])
        cur_k_rope = cur_k_rope.reshape(B, S, cur_k_rope.shape[-1])
        selection_k_nope = context.k_nope.squeeze(dim=2)
        selection_k_rope = context.k_rope.squeeze(dim=2)
        selection_kv_block_table = self.selection_kv_block_table[: B * S].reshape(
            B, S, self.selection_kv_block_table.shape[-1]
        )
        update_cur_kv_done = get_sfa1_ops().npu_update_cur_kv_to_selection(
            cur_k_nope, cur_k_rope, context.cur_kv_write_slots, selection_k_nope, selection_k_rope, selection_kv_block_table)

        context.update_cur_kv_done = update_cur_kv_done


    def decode_update_kv_lru_cache_async(
        self,
        forward_batch: ForwardBatch,
        topk_indices: torch.Tensor,
        layer_id: int,
        attn_idx: int,
        update_after: torch.Tensor = None,
        ):
        """
        依赖npu_gather_selection_kv_cache算子进行dsa在decode阶段的kv cache的加载。actual_seq_qlen和actual_seq_lengths_kv非前缀和。
        """
        if not self._offload_active(layer_id, attn_idx, forward_batch):
            return
        context = self.get_selection_context(attn_idx)

        # Keep hot-cache writes off the prefetch stream. Gen/Gather retain
        # their existing stream and core budgets. Packed metadata Update uses
        # its own vector budget; the split BF16 path retains the gather budget.
        # The two scopes are deliberately separate concerns, so they stay nested.
        with npu_stream_switch(forward_batch.can_run_with_graph, "kv_offload_update_stream"):  # noqa: SIM117
            with npu_limit_core("offload_update" if self.packed_kv else "offload_gather", flag=forward_batch.can_run_with_graph):
                layer_id = self._local_attention_id(layer_id, attn_idx)
                actual_seq_qlen_cumsum = forward_batch.attn_metadata.query_len_tensor.to(torch.int32)
                #actual_seq_qlen = torch.diff(actual_seq_qlen_cumsum,
                #    prepend=torch.tensor([0], device=actual_seq_qlen_cumsum.device, dtype=actual_seq_qlen_cumsum.dtype))
                #bs = actual_seq_qlen.shape[0]

                bsz, _, index_topk = topk_indices.shape #tnd
                selection_kv_block_table = self.selection_kv_block_table[:bsz]#shape=【T，seq-len】
                # shape = TND,
                #[t0,t1,t2]->t0,t2,t3 status [t0,t1,t2,t3]  status [t0,t2,t3] -> status [index] -> status []
                # batchsize=4,mtp=3,总共16token。
                # topk_indices：[16,1,2048]
                # selection_k_nope:[16,128,512]-》bf16
                # selection_k_rope:[16,128,64]
                # block_table:[4,8192]
                # actual_seq_lengths_kv:[B]
                # actual_seq_qlen:[B]
                # full_kv_nope:[1294,128,512]-》bf16
                # full_kv_rope:[1294,128,64]
                # 1、16，1，2048 steps 16 step 0 16个token
                # 2、4，1，2048 复制4份 step 0 stauts -1 step 1 基于前一步的status的 batch 0 [seq0,seq1,seq2,seq3] batch[batch0,batch1,batch2,batch3]
                #get indices
                # cache_status.fill_(-1)
                topk_indices_op = context.cache_topk_indices
                if self.packed_kv:
                    if context.source_topk is not topk_indices:
                        raise RuntimeError("Packed LRU update must reuse its gathered source TopK")
                # THE write-back this fix exists for: with the whole tensor
                # the four `copy_` calls below are pure whole-tensor writes,
                # so the `slice_scatter(arg, new, 0, 0, B)` + `copy_(arg,..)`
                # read-modify-write disappears. `step` is advanced for every
                # slot rather than only the active ones - deliberate and
                # harmless: an inactive slot admits nothing (its pad row is
                # inert) and its step is zeroed when it next takes an
                # occupant (`:554`, `:411-416`).
                (
                    topk_indices_op, _kv_lens, _q_lens,
                    (cache_status, cache_age, cache_kr, cache_kv, step),
                ) = self.hot_cache_operands(
                    layer_id, self.max_batch_size, topk_indices_op
                )
                selection_kv_block_table = self.selection_kv_block_table
                write_slot = context.write_slot
                if global_server_args_dict.get("kvp_size", 1) > 1:
                    _, local_kv_lens = select_offload_kv_metadata(forward_batch)
                    query_lens = torch.diff(
                        actual_seq_qlen_cumsum,
                        prepend=actual_seq_qlen_cumsum.new_zeros(1),
                    )
                    write_slot = dcp_history_only_write_slots(
                        topk_indices_op, write_slot, local_kv_lens, query_lens,
                        get_attn_tp_group().rank_in_group,
                    )
                if self.update_policy == "last_query":
                    # Keep the original query count for the history mask above;
                    # only hot-cache admission uses each request's final query.
                    topk_indices_op, write_slot, selection_kv_block_table = last_query_update_operands(
                        topk_indices_op, write_slot,
                        self.last_query_update_tables[topk_indices_op.shape[1]],
                    )

                if update_after is not None and forward_batch.can_run_with_graph:
                    # wait_tensor is a pure Tensor -> Tensor op. Its returned
                    # Identity carries the control edge and must be consumed.
                    # Use the small index tensor, not a full KV tensor.
                    topk_indices_op = npu_wait_tensor(topk_indices_op, update_after)
                selection_k_nope = context.k_nope.squeeze(dim=2)
                if self.packed_kv:
                    # The current-row patch publishes k_nope with its GE
                    # completion dependency; all packed consumers share it.
                    new_cache_kv, new_cache_status, new_cache_age = get_packed_offload_ops().npu_update_cache_packed(
                        topk_indices_op, selection_k_nope, cache_kv,
                        cache_status, cache_age, selection_kv_block_table, write_slot,
                        step=step,
                    )
                else:
                    selection_k_rope = context.k_rope.squeeze(dim=2)
                    if forward_batch.can_run_with_graph and context.update_cur_kv_done is not None:
                        selection_k_rope = npu_wait_tensor(selection_k_rope, context.update_cur_kv_done)
                    new_cache_kr, new_cache_kv, new_cache_status, new_cache_age = get_sfa1_ops().npu_update_cache(
                        topk_indices_op, selection_k_rope, selection_k_nope, cache_kr, cache_kv,
                        cache_status, cache_age, selection_kv_block_table, write_slot,
                        step=step,
                    )
                step.add_(1)
                #update pool
                cache_status.copy_(new_cache_status)
                cache_age.copy_(new_cache_age)
                if not self.packed_kv:
                    cache_kr.copy_(new_cache_kr)
                cache_kv.copy_(new_cache_kv)

        return
