from __future__ import annotations
from sglang.srt.npu.sfa1_runtime import get_attention_ops

import math
from itertools import accumulate
from typing import TYPE_CHECKING

import flash_npu_kernel  # noqa: F401  # Registers torch.ops.flash operators.
import flash_ops  # noqa: F401  # Registers torch.ops.custom operators.
import torch
import torch_npu  # noqa: F401 - registers device/operator implementations
import torchair as tng
from sglang.srt.configs.model_config import (
    is_deepseek_nsa,
)
from sglang.srt.distributed import (
    get_attn_tp_group,
    get_lse_comm_group,
)
from sglang.srt.env import ENV, global_server_args_dict
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.dsa.indexer_balancer import IndexerBalancer, IndexerLatencyPredictor, get_all2all_params_for_index_balance
from sglang.srt.layers.attention.dsa.nsa_indexer import get_non_zero_q_meta
from sglang.srt.layers.attention.npu_attn.flash_attn import AttentionMetadata
from sglang.srt.layers.attention.radix_attention import AttentionType
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.npu.utils import npu_limit_core, npu_stream_switch, npu_super_kernel
from sglang.srt.npu.offload_runtime import (
    host_offload_enabled, offload_sfad_bounded_holes_config,
)
from sglang.srt.utils import get_colorful_logger

logger = get_colorful_logger(__name__)

if TYPE_CHECKING:
    from sglang.srt.layers.attention.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.speculative.spec_info import SpecInfo

BLOCK_SIZE = 128
_PACKED_KV_DEQUANT_OP = getattr(torch.ops.custom, "npu_packed_kv_dequant", None)


def unpack_packed_c8_tnd(
    packed_kv: torch.Tensor,
    latent_dim: int,
    rope_head_dim: int,
    tile_size: int,
    output_dtype: torch.dtype,
):
    """Decode a contiguous packed-C8 row for the mature BF16 SFA path."""
    if packed_kv.dtype != torch.int8:
        raise RuntimeError(f"packed C8 Prefill expects int8 KV, got {packed_kv.dtype}.")
    if output_dtype != torch.bfloat16:
        raise RuntimeError(f"packed C8 Prefill SFA currently requires BF16 queries, got {output_dtype}.")
    if latent_dim % tile_size != 0:
        raise RuntimeError(f"packed C8 latent dim {latent_dim} is not divisible by {tile_size}.")
    rope_bytes = rope_head_dim * torch.bfloat16.itemsize
    scale_count = latent_dim // tile_size
    scale_bytes = scale_count * torch.float32.itemsize
    payload_width = latent_dim + rope_bytes + scale_bytes
    aligned_width = (payload_width + 31) // 32 * 32
    row_width = packed_kv.shape[-1]
    if row_width not in (payload_width, aligned_width):
        raise RuntimeError(f"packed C8 Prefill row width mismatch: expected {payload_width} or {aligned_width}, got {row_width}.")

    # The production operator package fuses slicing, INT8->FP16->FP32,
    # per-128 FP32 scaling, BF16 conversion, and RoPE extraction into one AIV
    # launch.  Keep the eager expression as an ABI-compatible fallback so an
    # older operator package remains functionally usable during rolling
    # deployment, albeit without the cached-Prefill performance optimization.
    if (
        _PACKED_KV_DEQUANT_OP is not None
        and latent_dim == 512
        and rope_head_dim == 64
        and tile_size == 128
        and packed_kv.dim() == 3
        and packed_kv.shape[1] == 1
    ):
        logger.info_once("NPU DSA KV int8 fused packed-KV dequant operator is active.")
        return _PACKED_KV_DEQUANT_OP(packed_kv.contiguous())

    prefix_shape = packed_kv.shape[:-1]
    quant = packed_kv[..., :latent_dim].contiguous().reshape(*prefix_shape, scale_count, tile_size)
    rope_end = latent_dim + rope_bytes
    scale_end = rope_end + scale_bytes
    key_rope = packed_kv[..., latent_dim:rope_end].contiguous().view(torch.bfloat16).reshape(*prefix_shape, rope_head_dim)
    scales = packed_kv[..., rope_end:scale_end].contiguous().view(torch.float32).reshape(*prefix_shape, scale_count, 1)
    key = (quant.float() * scales).to(output_dtype).reshape(*prefix_shape, latent_dim)
    return key, key_rope


def kvp_cache_copy_rows_(
    cache: torch.Tensor,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    dependency: torch.Tensor | None = None,
) -> torch.Tensor:
    if dependency is not None:
        dst_indices = tng.scope.npu_wait_tensor(dst_indices, dependency)
    return torch.ops.custom.npu_kv_cache_copy_rows_(
        cache,
        src_indices.reshape(-1),
        dst_indices.reshape(-1),
    )


def all2all_wrapper(x, group=None):
    x_shape = x.shape
    x = x.reshape(-1)  # for GE graph
    o_all2all = torch.empty_like(x)
    torch.distributed.all_to_all_single(o_all2all, x, group=group)
    o_all2all = o_all2all.reshape(x_shape)
    return o_all2all


index_balancer = None


def get_index_balancer():
    global index_balancer
    if index_balancer is None:
        index_balancer = IndexerBalancer(get_attn_tp_group().world_size, IndexerLatencyPredictor(use_mlp=True))
    return index_balancer


class NPURotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, inv_freq_persistent=True):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.device = device
        self.initialized = False
        self.inv_freq_persistent = inv_freq_persistent
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(self.device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, self.inv_freq_persistent)
        if not self.initialized:
            self.initialized_first_time(device, torch.bfloat16)

    def initialized_first_time(self, device, dtype):
        """gaoxi:
            inv_freq可能会在加载时被覆盖
            那么第一次运行需要重置inv_freq
        TODO:
            在加载时不要覆盖inv_freq
        """
        self.initialized = True
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(self.device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, self.inv_freq_persistent)
        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(seq_len=self.max_position_embeddings, device=device, dtype=dtype)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq.to(device))
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        # This `if` block is unlikely to be run after we build sin/cos in `__init__`. Keep the logic here just in case.
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)
        return (self.cos_cached[:seq_len].to(dtype=x.dtype), self.sin_cached[:seq_len].to(dtype=x.dtype))


class NpuMLAAttnBackend(AttentionBackend):
    def __init__(self, model_runner: ModelRunner):
        super().__init__()
        self.forward_metadata = None
        self.use_sfa1 = model_runner.use_sfa1
        self.is_draft_worker = model_runner.is_draft_worker
        bounded_holes = offload_sfad_bounded_holes_config(model_runner.server_args)
        if (bounded_holes["effective"]
                and getattr(model_runner.server_args, "_npu_offload_sfad_bounded_holes", None) is None):
            raise RuntimeError("SFAD bounded holes requires ServerArgs startup cache qualification")
        self.sfad_bounded_holes = bounded_holes["effective"] and not self.is_draft_worker
        self.device = model_runner.device
        self._sfad_pool = model_runner.token_to_kv_pool
        coordinator = getattr(self._sfad_pool, "sparse_coordinator", None)
        self._sfad_offload_enabled = (
            self.use_sfa1
            and not self.is_draft_worker
            and host_offload_enabled(global_server_args_dict)
            and global_server_args_dict.get("kvp_size") == 8
            and global_server_args_dict.get("npu_disable_kv_nz", False)
            and getattr(model_runner.model_config.hf_config, "cli_factor", 1) == 2
            and global_server_args_dict.get("offload_cache_update_timing", "before_attention") in ("before_attention", "after_sfad")
            and coordinator is not None
        )
        # The pool and coordinator share the same local attention IDs.
        self._native_keyrope_layers = (
            frozenset(range(self._sfad_pool.layer_num)) - frozenset(coordinator.offloaded_layer_ids)
            if self._sfad_offload_enabled else frozenset()
        )
        self.attn_mask = ~torch.tril(torch.ones((2048, 2048), dtype=torch.bool, device=self.device))
        if ENV.npu_enable_graph:
            torch._dynamo.mark_static(self.attn_mask)

    def sfad_offload_step(self, forward_batch):
        """Shared target route for Prolog, attention and persistence ownership."""
        return (
            self._sfad_offload_enabled
            and forward_batch.token_to_kv_pool is self._sfad_pool
            and forward_batch.all_decode_or_idle
            and (forward_batch.forward_mode.is_decode()
                 or forward_batch.forward_mode.is_target_verify())
            and 0 < forward_batch.input_ids.numel() <= 64
        )

    def inline_current_keyrope_layers(self, forward_batch):
        if self.sfad_offload_step(forward_batch):
            return self._native_keyrope_layers
        # Multi-head draft shares the first head's backend but selects its HBM
        # pool per head. Use the batch pool, not the backend's frozen first pool.
        if (
            self.is_draft_worker
            and self.use_sfa1
            and (forward_batch.forward_mode.is_decode()
                 or forward_batch.forward_mode.is_draft_extend())
        ):
            return frozenset(range(forward_batch.token_to_kv_pool.layer_num))
        return frozenset()

    def init_cuda_graph_state(
        self,
        max_bs: int,
        block_kv_indices: torch.Tensor | None = None,
    ):
        pass

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        """Init the metadata for a forward pass."""
        # Hand the offload hot cache over before any layer reads it.
        # This is the only per-step host-side hook that runs outside
        # the captured graph, which is where a value-dependent slot
        # decision has to be made.
        pool = getattr(forward_batch, "token_to_kv_pool", None)
        coordinator = getattr(pool, "sparse_coordinator", None)
        if coordinator is not None and forward_batch.attn_metadata is not None:
            coordinator.sync_request_slots(forward_batch)

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        encoder_lens: torch.Tensor | None,
        forward_mode: ForwardMode,
        spec_info: SpecInfo | None,
        forward_batch: ForwardBatch,
    ):
        pass

    def get_cuda_graph_seq_len_fill_value(self):
        return 1

    @staticmethod
    def bind_offload_selection_context(
        coordinator,
        atten_id,
        topk_indices,
        k_nope,
        k_rope,
        block_table,
        actual_seq_qlen,
        actual_seq_lengths_kv,
        update_timing="before_attention",
    ):
        """Bind SFA1 to gathered selection KV when this attention has a context.

        Selection ids are positional, retaining DCP's -1 sentinels. Each TND
        row is one query, so lengths, topk and block-table rows must agree.
        Keep the backing pool references separate for current-KV writeback.
        Return original operands when no selected context exists.
        """
        if coordinator is None or not coordinator.has_selection_context(atten_id):
            return (
                topk_indices,
                k_nope,
                k_rope,
                block_table,
                actual_seq_qlen,
                actual_seq_lengths_kv,
            )
        selection_context = coordinator.get_selection_context(atten_id)
        # The gather for this attention is issued while the PREVIOUS attention
        # runs, so it is planned from whatever TopK existed then, and the
        # context's topk_indices is positional over the buffer it produced.
        # Returning it to a caller holding a different TopK means attending
        # that other selection's tokens. Identity, not equality: with
        # cli_factor == 2 the CLI pair shares one tensor by construction, which
        # is the same contract deepseek_mla asserts on the prefill route.
        #
        # Reported, not enforced, until a device run has shown it does not fire
        # on a CLI2 checkpoint. Raising here would turn a working service into
        # a failing one if that identity reasoning is wrong, and it has never
        # been exercised; warning finds the same condition at no such risk.
        # Promote to a raise once a run has confirmed it stays quiet.
        if (selection_context.source_topk is not None
                and selection_context.source_topk is not topk_indices
                and not getattr(coordinator, "_warned_topk_mismatch", False)):
            coordinator._warned_topk_mismatch = True
            logger.warning(
                "selected-KV context for attention %s was gathered from a "
                "different TopK than this attention computed; the CLI pair "
                "must share one TopK (cli_factor == 2) for the offload decode "
                "path. Attention will use the gathered selection.", atten_id,
            )
        # `actual_seq_qlen` is arange(1, rows+1): one query per selection row, so
        # its length IS the batch the operator will compute. Only actual_kv_seq
        # is worth asserting against it: the publisher derives topk_indices and
        # actual_seq_qlen from one local batch size a few lines apart, so that
        # term was `bsz != bsz`, while actual_kv_seq comes back from the vendor
        # gather and is narrowed by hand on the padded branch. Static shapes
        # only - no host sync and no value-dependent branch.
        rows = selection_context.actual_seq_qlen.shape[0]
        if selection_context.actual_kv_seq.numel() != rows:
            raise AssertionError(
                f"selection metadata disagrees with actual_seq_lengths_query "
                f"({rows} rows): actual_kv_seq="
                f"{selection_context.actual_kv_seq.numel()}"
            )
        bound = (
            selection_context.topk_indices,
            selection_context.k_nope,
            selection_context.k_rope,
            coordinator.selection_block_table(rows),
            selection_context.actual_seq_qlen,
            selection_context.actual_kv_seq,
        )
        if update_timing == "before_attention":
            coordinator.clear_temp_tensors(atten_id)
        return bound

    def _forward_kvp_sfad(
        self, *, q, q_pe, k_nope, k_pe, layer, forward_batch,
        actual_seq_qlen, layout_kv, topk_indices, skip_gather_q,
        sfad_packed_query, packed_int8_kv, absorbed, torchair_enable,
        selection_context=None,
    ):
        """Native SFAD math/communication; selected KV only changes operands.

        Native callers retain the original Prolog KV views and inline writer.
        Selected materialization owns only temporary KV; host persistence stays
        with its existing post-hook, never the native writer below.
        """
        kvp_k_write = None
        kvp_k_pe_write = None
        write_current = selection_context is None
        if write_current:
            block_table = forward_batch.attn_metadata.kvp_block_table
            actual_seq_lengths_kv = forward_batch.attn_metadata.kvp_context_lens_tensor.to(torch.int32)
            valid_chunks = forward_batch.topk_valid_chunks
        else:
            valid_chunks = selection_context.source_valid_chunks
            update_done = selection_context.update_cur_kv_done
            coordinator = forward_batch.token_to_kv_pool.sparse_coordinator
            topk_indices = selection_context.topk_indices
            k_nope, k_pe = selection_context.k_nope, selection_context.k_rope
            actual_seq_qlen = selection_context.actual_seq_qlen
            actual_seq_lengths_kv = selection_context.actual_kv_seq
            block_table = coordinator.selection_block_table(actual_seq_qlen.shape[0])
            if global_server_args_dict.get("offload_cache_update_timing", "before_attention") == "before_attention":
                coordinator.clear_temp_tensors(selection_context.attention_id)
            if forward_batch.can_run_with_graph:
                k_pe = tng.scope.npu_wait_tensor(k_pe, update_done)
        kvp_group = get_attn_tp_group()
        kvp_rank = kvp_group.rank_in_group
        kvp_size = global_server_args_dict["kvp_size"]
        kv_lora_rank = q.shape[-1]
        qk_rope_head_dim = q_pe.shape[-1]
        packed_query = sfad_packed_query
        if packed_query is not None:
            valid_packed_shape = (
                packed_query.dim() in (4, 5)
                and packed_query.shape[1] > 0
                and packed_query.shape[-1] == kv_lora_rank + qk_rope_head_dim
                and packed_query.shape[0] * packed_query.shape[-2] == 64
                and (packed_query.dim() == 4 or packed_query.shape[2] == 1)
                and packed_query.dtype in (torch.bfloat16, torch.float16)
            )
            if not valid_packed_shape:
                raise RuntimeError(f"Invalid rank-major SFAD query shape: {packed_query.shape}.")
        elif not skip_gather_q:
            local_query = torch.cat([q, q_pe], dim=-1)
            use_packed_query = (
                ENV.npu_enable_sfad_rank_major_query
                and local_query.shape[0] > 0
                and local_query.dim() in (3, 4)
                and local_query.shape[-1] == 576
                and kvp_size * local_query.shape[-2] == 64
                and (local_query.dim() == 3 or local_query.shape[1] == 1)
                and local_query.dtype in (torch.bfloat16, torch.float16)
                and layout_kv == "PA_BSND"
            )
            if use_packed_query:
                packed_flat = torch.empty(
                    (kvp_size * local_query.shape[0],) + local_query.shape[1:],
                    dtype=local_query.dtype,
                    device=local_query.device,
                )
                kvp_group.all_gather_into_tensor(packed_flat, local_query)
                packed_query = packed_flat.view(kvp_size, *local_query.shape)
            else:
                q_total = kvp_group.all_gather(local_query, dim=1)
                q, q_pe = torch.split(
                    q_total,
                    [kv_lora_rank, qk_rope_head_dim],
                    dim=2,
                )

        if packed_query is None:
            sfad_query = q
            sfad_query_rope = q_pe
            sfad_layout_query = "TND_KVP_COMM"
            T = q.shape[0]
            N = q.shape[1]
        else:
            sfad_query = packed_query
            sfad_query_rope = None
            sfad_layout_query = "RANK_MAJOR_TND_PACKED_KVP_COMM"
            T = packed_query.shape[1]
            N = packed_query.shape[0] * packed_query.shape[-2]
        sparse_mode = 0
        if kvp_rank == 0:
            sparse_mode = 3
        use_bounded_sparse_prep = packed_int8_kv and absorbed
        use_bounded_holes_prep = False
        if valid_chunks is None:
            raise RuntimeError("SFAD requires topk_valid_chunks paired with topk_indices.")
        sfad_key = k_nope
        sfad_topk = topk_indices
        sfad_query_lengths = actual_seq_qlen
        packed_selection_coordinator = None
        update_timing = global_server_args_dict.get("offload_cache_update_timing", "before_attention")
        if (
            packed_int8_kv
            and getattr(forward_batch.token_to_kv_pool, "packed_host_offload", False)
            and layer.attn_layer_id % 2 == 1
        ):
            from sglang.srt.mem_cache.kv_offload.packed_host import consume_packed_selection
            packed_selection_coordinator = getattr(
                forward_batch.token_to_kv_pool, "sparse_coordinator", None
            )
            if update_timing == "after_sfad" and not forward_batch.all_decode_or_idle:
                raise RuntimeError("after_sfad requires a target Decode/verify or padded-idle batch")

            (
                sfad_key, sfad_topk, block_table, sfad_query_lengths,
                actual_seq_lengths_kv, valid_chunks,
            ) = consume_packed_selection(
                packed_selection_coordinator,
                layer.attn_layer_id % 2,
                topk_indices,
                update_timing,
                original_valid_chunks=valid_chunks,
            )
            sparse_mode = 0
            # Gen preserves the original prefix positions while masking
            # entries to -1. Retain its chunk plan and positional holes.
            use_bounded_holes_prep = self.sfad_bounded_holes and use_bounded_sparse_prep
            use_bounded_sparse_prep = False
        o_ori, max_o, sum_o = torch.ops.custom.npu_sparse_flash_attention_decode(
            query=sfad_query,
            key=sfad_key,
            value=sfad_key,
            query_rope=sfad_query_rope,
            key_rope=None if packed_int8_kv else k_pe,
            sparse_indices=sfad_topk,
            valid_chunks=valid_chunks,
            scale_value=layer.scaling,
            actual_seq_lengths_query=sfad_query_lengths,
            actual_seq_lengths_kv=actual_seq_lengths_kv,
            block_table=block_table,
            sparse_block_size=1,
            layout_query=sfad_layout_query,
            layout_kv=layout_kv,
            sparse_mode=sparse_mode,
            # attention_mode describes MLA semantics only. The bounded
            # sparse-address preparation remains limited to the packed
            # INT8 Model path where it was measured to help.
            attention_mode=2,
            sparse_prep_mode=2 if use_bounded_holes_prep else 1 if use_bounded_sparse_prep else 0,
            return_softmax_lse=True,
        )

        if selection_context is not None and update_timing == "after_sfad":
            coordinator.decode_update_kv_lru_cache_async(
                forward_batch, selection_context.source_topk,
                layer.layer_id, selection_context.attention_id, update_after=sum_o,
            )
            coordinator.clear_temp_tensors(selection_context.attention_id)
        elif packed_selection_coordinator is not None and update_timing == "after_sfad":
            context = packed_selection_coordinator.get_selection_context(layer.attn_layer_id % 2)
            packed_selection_coordinator.decode_update_kv_lru_cache_async(
                forward_batch, context.source_topk, layer.layer_id,
                context.attention_id, update_after=sum_o,
            )
            packed_selection_coordinator.clear_temp_tensors(context.attention_id)

        with (
            npu_stream_switch(forward_batch.can_run_with_graph, "lse_stream"),
            npu_limit_core(f"kvp_lse_{layer.attn_layer_id % 2}", flag=forward_batch.can_run_with_graph),
            npu_super_kernel("", "", flag=forward_batch.can_run_with_graph),
        ):
            # O is already in communication order, while max/sum
            # stay in SFAD's contiguous [N2,T,G] layout.  Prepare
            # the tiny LSE branch asynchronously behind O AllToAll.
            lse = max_o + torch.log(sum_o)  # [1=N2, T, N]
            lse = lse.permute(1, 0, 2).reshape(T, -1).unsqueeze(-1)  # [T, N , 1]
            lse = lse.transpose(0, 1).reshape(kvp_size, N // kvp_size, T).contiguous()
            with npu_stream_switch(forward_batch.can_run_with_graph, "lse_all2all_stream"):
                lse_all2all = all2all_wrapper(lse, group=get_lse_comm_group().device_group)  # [kvp, local_head, T]

            if write_current and kvp_rank == 0 and not packed_int8_kv:
                assert global_server_args_dict["npu_disable_kv_nz"]
                k_nope = k_nope.view(-1, kv_lora_rank)
                kvp_k_write = kvp_cache_copy_rows_(
                    k_nope,
                    forward_batch.attn_metadata.slot_mapping,
                    forward_batch.attn_metadata.kvp_current_slots_mapping,
                    o_ori if torchair_enable else None,
                )

            lse_all2all = lse_all2all.reshape(kvp_size, (N // kvp_size) * T)

        # The packed INT8 cache has a single row write.  Keep it off the
        # LSE stream: writer rank 0 may otherwise queue cache writes ahead
        # of LogAdd/Transpose and make every peer wait in LSE AllToAll.
        # Waiting for the prepared LSE prioritizes the short local branch;
        # the copy can still overlap both AllToAll operations.
        packed_host_writeback = (
            packed_int8_kv
            and getattr(forward_batch.token_to_kv_pool, "packed_host_offload", False)
            and layer.attn_layer_id % 2 == 1
            and global_server_args_dict.get("offload_cache_storage", "host") == "host"
        )
        # Host current rows become persistent in the model's post-forward
        # hook. Keeping this mutation outside GE avoids functionalizing a
        # few-row update into a full Host pool TensorMove. Attention and LRU
        # admission already consume their staging/selection rows above.
        if write_current and kvp_rank == 0 and packed_int8_kv and not packed_host_writeback:
            assert global_server_args_dict["npu_disable_kv_nz"]
            with npu_stream_switch(
                forward_batch.can_run_with_graph,
                "kvp_cache_stream",
            ):
                cache_row_width = forward_batch.token_to_kv_pool.kv_cache_dim
                k_nope = k_nope.view(-1, cache_row_width)
                kvp_k_write = kvp_cache_copy_rows_(
                    k_nope,
                    forward_batch.attn_metadata.slot_mapping,
                    forward_batch.attn_metadata.kvp_current_slots_mapping,
                    lse if torchair_enable else None,
                )

        with (
            npu_limit_core(f"kvp_o_{layer.attn_layer_id % 2}", flag=forward_batch.can_run_with_graph),
            npu_super_kernel("", "", flag=forward_batch.can_run_with_graph),
        ):
            # SFAD writes [N,T,D] directly in per-peer order.
            o = o_ori.reshape(kvp_size, N // kvp_size, T, -1)
            o_all2all = all2all_wrapper(o, group=kvp_group.device_group)  # (8=kvp, 64 //8 = local_head, T, 512=DIM)

            if write_current and kvp_rank == 0 and not packed_int8_kv:
                k_pe = k_pe.view(-1, qk_rope_head_dim)
                kvp_k_pe_write = kvp_cache_copy_rows_(
                    k_pe,
                    forward_batch.attn_metadata.slot_mapping,
                    forward_batch.attn_metadata.kvp_current_slots_mapping,
                    o_ori if torchair_enable else None,
                )

            o_all2all = o_all2all.reshape(kvp_size, (N // kvp_size) * T, -1)  # (8=kvp, 64 //8 = local_head * T, 512=DIM)
            o_all2all = o_all2all.to(max_o.dtype)  # to float

        if o_all2all.shape[-1] != 512:
            raise RuntimeError("KvpAttentionMerge currently requires head_dim=512.")

        if forward_batch.can_run_with_graph:
            o_all2all = tng.scope.npu_wait_tensor(o_all2all, lse_all2all)
        o = torch.ops.custom.npu_kvp_attention_merge(o_all2all, lse_all2all)
        attn_out = o.reshape(N // kvp_size, T, -1)
        if write_current and torchair_enable and kvp_rank == 0 and kvp_k_write is not None:
            attn_out = tng.scope.npu_wait_tensor(attn_out, kvp_k_write)
            if kvp_k_pe_write is not None:
                attn_out = tng.scope.npu_wait_tensor(attn_out, kvp_k_pe_write)
        return attn_out

    def forward_sparse(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        # For multi_head latent attention
        q_rope: torch.Tensor | None = None,
        k_rope: torch.Tensor | None = None,
        topk_indices: torch.Tensor = None,
        absorbed: bool = False,
        layout: str = "TND",
        skip_gather_q: bool = False,
        atten_id: int = 0,
        sfad_packed_query: torch.Tensor | None = None,
    ):
        is_prefill = (
            forward_batch.forward_mode.is_extend()
            and not forward_batch.forward_mode.is_draft_extend()
            and not forward_batch.forward_mode.is_target_verify()
        )

        if save_kv_cache:
            forward_batch.token_to_kv_pool.set_kv_buffer(layer, forward_batch.out_cache_loc, k, v)

        if absorbed:
            q = q.squeeze(dim=1)
            q_rope = q_rope.squeeze(dim=1)
            block_size = forward_batch.token_to_kv_pool.page_size
            assert layer.tp_k_head_num == 1, "num of k head must be 1 in mla absorbed mode"
            if not global_server_args_dict["npu_disable_kv_nz"]:
                k = k.squeeze(dim=2)  # head num must be 1
                k_rope = k_rope.squeeze(dim=2)  # head num must be 1
                k = (
                    k.view(k.shape[0], 1, k.shape[2] // 16, k.shape[1], 16)
                    .transpose(1, 3)
                    .reshape(-1, k.shape[2])
                    .reshape(-1, block_size, 1, k.shape[2])
                )
                k_rope = (
                    k_rope.view(k_rope.shape[0], 1, k_rope.shape[2] // 16, k_rope.shape[1], 16)
                    .transpose(1, 3)
                    .reshape(-1, k_rope.shape[2])
                    .reshape(-1, block_size, 1, k_rope.shape[2])
                )

        q_nope, q_pe = q, q_rope
        k_nope, k_pe = k, k_rope

        if is_prefill:
            if forward_batch.attn_metadata.actual_seq_lengths is not None:
                actual_seq_qlen = forward_batch.attn_metadata.actual_seq_lengths_tensor_int32
            else:
                actual_seq_qlen = forward_batch.attn_metadata.cu_seq_lens_tensor_int32
        else:
            if forward_batch.attn_metadata.query_len_tensor is None:
                if forward_batch.forward_mode.is_target_verify():
                    actual_seq_qlen = (
                        torch.arange(
                            forward_batch.spec_num_steps,
                            forward_batch.spec_num_steps + q.shape[0],
                            forward_batch.spec_num_steps,
                            dtype=torch.int32,
                        )
                        .to(q.device)
                        .to(torch.int32)
                    )
                elif forward_batch.forward_mode.is_draft_extend():
                    actual_seq_qlen = torch.cumsum(
                        forward_batch.extend_seq_lens,
                        dim=0,
                        dtype=torch.int32,
                    ).to(q.device)
                else:
                    actual_seq_qlen = torch.arange(1, q.shape[0] + 1).to(q.device).to(torch.int32)
            else:
                actual_seq_qlen = forward_batch.attn_metadata.query_len_tensor.to(torch.int32)

        if is_prefill:
            actual_seq_lengths_kv = forward_batch.attn_metadata.seq_lens_tensor_int32
            if "TND" in layout:
                # sfa is not support block table with TND layout.
                block_table = None
                layout_kv = "TND"
                actual_seq_lengths_kv = forward_batch.attn_metadata.actual_seq_lengths_kv_tensor_int32
        else:
            actual_seq_lengths_kv = forward_batch.attn_metadata.seq_lens_tensor.to(torch.int32)
            block_table = forward_batch.attn_metadata.block_table
            layout_kv = "PA_BSND"

        use_kvp = global_server_args_dict["kvp_size"] > 1
        torchair_enable = forward_batch.all_decode_or_idle and ENV.npu_enable_graph and not ENV.npu_disable_graph_mode
        packed_int8_kv = getattr(forward_batch.token_to_kv_pool, "dsa_kv_int8_packed", False) and k_nope.dtype == torch.int8
        use_packed_prefill_sfa = packed_int8_kv and is_prefill and not use_kvp and layout_kv == "TND" and not forward_batch.can_run_with_graph
        if use_packed_prefill_sfa:
            logger.info_once("NPU DSA KV int8 Prefill dequant-to-BF16 SFA path is active.")
        # Native INT8 SFAD is the KVP decode path. The validated eager Prefill
        # path stages BF16 KV for SFA; keep QSFA for unsupported fallbacks.
        if packed_int8_kv and (is_prefill or not use_kvp):
            if q_pe is None:
                raise RuntimeError("NPU DSA KV int8 requires q_pe for MLA attention.")
            kv_pool = forward_batch.token_to_kv_pool
            packed_kv_dim = kv_pool.kv_cache_dim
            tile_size = kv_pool.quant_block_size
            rope_head_dim = q_pe.shape[-1]
            q_total = None
            if not use_packed_prefill_sfa:
                q_total = torch.cat([q_nope, q_pe], dim=-1).contiguous()
            packed_kv = k_nope.contiguous()
            if packed_kv.shape[-1] != packed_kv_dim:
                packed_kv = packed_kv.view(-1, 1, packed_kv_dim)
            if packed_kv.dtype != torch.int8 or packed_kv.shape[-1] != packed_kv_dim:
                raise RuntimeError(
                    "NPU DSA KV int8 attention expects int8 KV rows with dim "
                    f"{packed_kv_dim}, got dtype={packed_kv.dtype}, "
                    f"shape={tuple(packed_kv.shape)}."
                )
            if topk_indices is None:
                raise RuntimeError("NPU DSA KV int8 attention requires sparse topk indices.")

            expected_shape = q_nope.shape
            enable_sp_for_indexer = (
                is_prefill and global_server_args_dict.get("enable_mla_l1_5_cache", False) and global_server_args_dict["npu_enable_sp_for_indexer"]
            )
            if use_kvp:
                block_table = forward_batch.attn_metadata.kvp_block_table
                actual_seq_lengths_kv = forward_batch.attn_metadata.kvp_context_lens_tensor.to(torch.int32)
                kvp_group = get_attn_tp_group()
                kvp_rank = kvp_group.rank_in_group
                kvp_size = global_server_args_dict["kvp_size"]
                # The overlapped indexer/prolog path has already gathered Q
                # while the TopK-value AllToAll was in flight.
                if not skip_gather_q:
                    q_total = kvp_group.all_gather(q_total, dim=1)
                T = q_total.shape[0]
                N = q_total.shape[1]
                sparse_mode = 0
                if kvp_rank == 0:
                    sparse_mode = 3
                o_ori, max_o, sum_o = torch.ops.custom.npu_kv_quant_sparse_flash_attention_lse(
                    q_total,
                    packed_kv,
                    packed_kv,
                    topk_indices.contiguous(),
                    scale_value=layer.scaling,
                    sparse_block_size=1,
                    actual_seq_lengths_query=actual_seq_qlen.to(torch.int32),
                    actual_seq_lengths_kv=actual_seq_lengths_kv,
                    layout_query="TND",
                    layout_kv=layout_kv,
                    sparse_mode=sparse_mode,
                    block_table=block_table,
                    attention_mode=2,
                    quant_scale_repo_mode=1,
                    tile_size=tile_size,
                    key_quant_mode=2,
                    value_quant_mode=2,
                    rope_head_dim=rope_head_dim,
                    return_softmax_lse=True,
                )
                if o_ori.shape[0] != T or o_ori.shape[1] != N:
                    raise RuntimeError(
                        "NPU DSA KV int8 KVP attention output layout mismatch: "
                        "expected output in TND layout from "
                        "torch.ops.custom.npu_kv_quant_sparse_flash_attention_lse, got "
                        f"{o_ori.shape=} for {T=} {N=}."
                    )
                kv_heads = packed_kv.shape[1] if layout_kv == "TND" else packed_kv.shape[2]
                if N % kv_heads != 0:
                    raise RuntimeError(f"NPU DSA KV int8 KVP query heads must be divisible by KV heads, got {N=} {kv_heads=}.")
                expected_stats_shape = (kv_heads, T, N // kv_heads)
                if max_o.shape != sum_o.shape or max_o.shape != expected_stats_shape:
                    raise RuntimeError(
                        "NPU DSA KV int8 KVP attention softmax stats layout "
                        "mismatch: expected custom softmax_max and softmax_sum "
                        f"in {expected_stats_shape} layout, got {max_o.shape=} "
                        f"{sum_o.shape=}."
                    )

                kvp_kv_write = None
                with (
                    npu_stream_switch(forward_batch.can_run_with_graph, "lse_stream"),
                    npu_limit_core(
                        f"kvp_lse_{layer.attn_layer_id % 2}",
                        flag=forward_batch.can_run_with_graph,
                    ),
                    npu_super_kernel("", "", flag=forward_batch.can_run_with_graph),
                ):
                    max_o = max_o.permute(0, 2, 1).reshape(N, T, 1)
                    sum_o = sum_o.permute(0, 2, 1).reshape(N, T, 1)
                    lse = max_o + torch.log(sum_o)
                    lse = lse.permute(1, 0, 2).reshape(T, -1).unsqueeze(-1)
                    bs = actual_seq_lengths_kv.shape[0]
                    zero_kv_mask = (actual_seq_lengths_kv == 0)[:, None].expand(bs, T // bs).reshape(T)[:, None, None]
                    lse = lse.masked_fill(
                        zero_kv_mask.expand(-1, N, -1),
                        -2.0e38,
                    )
                    lse = lse.transpose(0, 1).reshape(kvp_size, N // kvp_size, T).contiguous()
                    with npu_stream_switch(
                        forward_batch.can_run_with_graph,
                        "lse_all2all_stream",
                    ):
                        lse_all2all = all2all_wrapper(
                            lse,
                            group=get_lse_comm_group().device_group,
                        )
                    lse_all2all = lse_all2all.reshape(kvp_size, (N // kvp_size) * T)

                    if kvp_rank == 0:
                        packed_kv_flat = packed_kv.view(-1, 1, packed_kv_dim)
                        if torchair_enable:
                            packed_kv_flat = tng.scope.npu_wait_tensor(packed_kv_flat, o_ori)
                        current_kv = packed_kv_flat[forward_batch.attn_metadata.slot_mapping]
                        kvp_kv_write = torch_npu.npu_scatter_nd_update_(
                            packed_kv_flat,
                            forward_batch.attn_metadata.kvp_current_slots_mapping.reshape(-1, 1),
                            current_kv,
                        )

                with (
                    npu_limit_core(
                        f"kvp_o_{layer.attn_layer_id % 2}",
                        flag=forward_batch.can_run_with_graph,
                    ),
                    npu_super_kernel("", "", flag=forward_batch.can_run_with_graph),
                ):
                    o = o_ori.transpose(0, 1).contiguous()
                    o = o.reshape(kvp_size, N // kvp_size, T, -1)
                    o_all2all = all2all_wrapper(o, group=kvp_group.device_group)
                    o_all2all = o_all2all.reshape(kvp_size, (N // kvp_size) * T, -1).to(max_o.dtype)

                if forward_batch.can_run_with_graph:
                    o_all2all = tng.scope.npu_wait_tensor(o_all2all, lse_all2all)
                o = torch.ops.custom.npu_kvp_attention_merge(o_all2all, lse_all2all)
                attn_out = o.reshape(N // kvp_size, T, -1).transpose(0, 1).contiguous()
                if torchair_enable and kvp_rank == 0:
                    attn_out = tng.scope.npu_wait_tensor(attn_out, kvp_kv_write)
            elif not enable_sp_for_indexer:
                if use_packed_prefill_sfa:
                    sfa_key, sfa_key_rope = unpack_packed_c8_tnd(
                        packed_kv,
                        q_nope.shape[-1],
                        rope_head_dim,
                        tile_size,
                        q_nope.dtype,
                    )
                    attn_out, _, _ = torch.ops.custom.npu_sparse_flash_attention(
                        query=q_nope,
                        key=sfa_key,
                        value=sfa_key,
                        query_rope=q_pe,
                        key_rope=sfa_key_rope,
                        sparse_indices=topk_indices,
                        scale_value=layer.scaling,
                        actual_seq_lengths_query=actual_seq_qlen.to(torch.int32),
                        actual_seq_lengths_kv=actual_seq_lengths_kv.to(torch.int32),
                        block_table=None,
                        sparse_block_size=1,
                        layout_query="TND",
                        layout_kv="TND",
                        sparse_mode=3,
                        attention_mode=2,
                        return_softmax_lse=False,
                    )
                else:
                    attn_result = torch.ops.custom.npu_kv_quant_sparse_flash_attention_lse(
                        q_total,
                        packed_kv,
                        packed_kv,
                        topk_indices.contiguous(),
                        scale_value=layer.scaling,
                        sparse_block_size=1,
                        actual_seq_lengths_query=actual_seq_qlen.to(torch.int32),
                        actual_seq_lengths_kv=actual_seq_lengths_kv.to(torch.int32),
                        layout_query="TND",
                        layout_kv=layout_kv,
                        sparse_mode=3,
                        block_table=block_table,
                        attention_mode=2,
                        quant_scale_repo_mode=1,
                        tile_size=tile_size,
                        key_quant_mode=2,
                        value_quant_mode=2,
                        rope_head_dim=rope_head_dim,
                        return_softmax_lse=False,
                    )
                    attn_out = attn_result[0] if isinstance(attn_result, tuple) else attn_result
            else:
                cp_size = get_attn_tp_group().world_size
                cp_rank = get_attn_tp_group().rank_in_group
                if not global_server_args_dict["npu_disable_dsa_head_parallel"]:
                    if use_packed_prefill_sfa:
                        q_nope = get_attn_tp_group().all_gather(q_nope.contiguous(), dim=1)
                        q_pe = get_attn_tp_group().all_gather(q_pe.contiguous(), dim=1)
                    else:
                        q_total = get_attn_tp_group().all_gather(q_total.contiguous(), dim=1)
                global_q_head = q_nope.shape[1] if use_packed_prefill_sfa else q_total.shape[1]

                if forward_batch.local_extend_lens_with_zero.sum().item() > 0:
                    if not global_server_args_dict["npu_disable_dsa_head_parallel"]:
                        slice_start = forward_batch.cu_attn_sp_token_nums_cpu[cp_rank].item()
                        slice_end = forward_batch.cu_attn_sp_token_nums_cpu[cp_rank + 1].item()
                        if use_packed_prefill_sfa:
                            q_nope = q_nope[slice_start:slice_end]
                            q_pe = q_pe[slice_start:slice_end]
                        else:
                            q_total = q_total[slice_start:slice_end]

                    packed_kv, _, _ = get_non_zero_q_meta(
                        packed_kv,
                        forward_batch.global_cu_kv_lens_cpu,
                        forward_batch.local_extend_lens_with_zero,
                        forward_batch.local_total_kv_lens_with_zero,
                    )
                    actual_seq_qlen = forward_batch.cur_seq_lengths_query_int32[1:]
                    actual_seq_lengths_kv = forward_batch.cur_seq_lengths_key_int32[1:]
                    if use_packed_prefill_sfa:
                        sfa_key, sfa_key_rope = unpack_packed_c8_tnd(
                            packed_kv,
                            q_nope.shape[-1],
                            rope_head_dim,
                            tile_size,
                            q_nope.dtype,
                        )
                        attn_out, _, _ = torch.ops.custom.npu_sparse_flash_attention(
                            query=q_nope,
                            key=sfa_key,
                            value=sfa_key,
                            query_rope=q_pe,
                            key_rope=sfa_key_rope,
                            sparse_indices=topk_indices,
                            scale_value=layer.scaling,
                            actual_seq_lengths_query=actual_seq_qlen,
                            actual_seq_lengths_kv=actual_seq_lengths_kv,
                            block_table=None,
                            sparse_block_size=1,
                            layout_query="TND",
                            layout_kv="TND",
                            sparse_mode=3,
                            attention_mode=2,
                            return_softmax_lse=False,
                        )
                    else:
                        attn_result = torch.ops.custom.npu_kv_quant_sparse_flash_attention_lse(
                            q_total,
                            packed_kv,
                            packed_kv,
                            topk_indices.contiguous(),
                            scale_value=layer.scaling,
                            sparse_block_size=1,
                            actual_seq_lengths_query=actual_seq_qlen,
                            actual_seq_lengths_kv=actual_seq_lengths_kv,
                            layout_query="TND",
                            layout_kv="TND",
                            sparse_mode=3,
                            block_table=block_table,
                            attention_mode=2,
                            quant_scale_repo_mode=1,
                            tile_size=tile_size,
                            key_quant_mode=2,
                            value_quant_mode=2,
                            rope_head_dim=rope_head_dim,
                            return_softmax_lse=False,
                        )
                        attn_out = attn_result[0] if isinstance(attn_result, tuple) else attn_result
                else:
                    attn_out = torch.empty(
                        (0, global_q_head, q_nope.shape[-1]),
                        dtype=q_nope.dtype,
                        device=q_nope.device,
                    )
                if not global_server_args_dict["npu_disable_dsa_head_parallel"]:
                    attn_sp_token_nums = get_attn_tp_group().get_local_sp_token_num(forward_batch.global_sp_num_tokens)
                    attn_out = get_attn_tp_group().all_gather(attn_out, dim=0, output_split_sizes=attn_sp_token_nums)
                    head_start = cp_rank * (global_q_head // cp_size)
                    head_end = (cp_rank + 1) * (global_q_head // cp_size)
                    attn_out = attn_out[:, head_start:head_end, :]

            if layout == "TND_NTD":
                attn_out = attn_out.transpose(0, 1).contiguous()
            if layout == "TND_NTD":
                expected_shape = (
                    expected_shape[1],
                    expected_shape[0],
                    expected_shape[2],
                )
            if attn_out.shape != expected_shape:
                raise RuntimeError(f"NPU DSA KV int8 attention output shape mismatch: {attn_out.shape=} {expected_shape=}.")
            return attn_out
        coordinator = getattr(
            forward_batch.token_to_kv_pool, "sparse_coordinator", None
        )
        selected_sfad = (
            use_kvp and self.sfad_offload_step(forward_batch)
            and coordinator.is_attention_offloaded(layer.layer_id, atten_id)
        )
        native_sfad = use_kvp and layer.attn_layer_id in self.inline_current_keyrope_layers(forward_batch)
        if use_kvp and (not self.use_sfa1 or selected_sfad or native_sfad):
            context = None
            if selected_sfad:
                context = coordinator.get_selection_context(atten_id)
            return self._forward_kvp_sfad(
                q=q, q_pe=q_pe, k_nope=k_nope, k_pe=k_pe, layer=layer,
                forward_batch=forward_batch, actual_seq_qlen=actual_seq_qlen,
                layout_kv=layout_kv, topk_indices=topk_indices,
                skip_gather_q=skip_gather_q, sfad_packed_query=sfad_packed_query,
                packed_int8_kv=packed_int8_kv, absorbed=absorbed,
                torchair_enable=torchair_enable,
                selection_context=context,
            )
        if use_kvp:
            # This split-BF16 path takes unpacked TND queries. Both ways of
            # getting a rank-major packed query here are already excluded:
            # NPU_ENABLE_SFAD_RANK_MAJOR_QUERY is a conjunct of `eligible`, so
            # it makes use_sfa1 False and this arm unreachable, and it also
            # gates the only producer of sfad_packed_query. The owner of that
            # invariant is initialize_sfa1_runtime, which fails at startup.
            block_table = forward_batch.attn_metadata.kvp_block_table
            actual_seq_lengths_kv = forward_batch.attn_metadata.kvp_context_lens_tensor.to(torch.int32)
            kvp_group = get_attn_tp_group()
            kvp_rank = kvp_group.rank_in_group
            kvp_size = global_server_args_dict["kvp_size"]
            kv_lora_rank = q.shape[-1]
            qk_rope_head_dim = q_pe.shape[-1]
            if not skip_gather_q:
                local_query = torch.cat([q, q_pe], dim=-1)
                q_total = kvp_group.all_gather(local_query, dim=1)
                q, q_pe = torch.split(
                    q_total, [kv_lora_rank, qk_rope_head_dim], dim=2
                )
            T, N = q.shape[:2]
            sparse_mode = 0
            if kvp_rank == 0:
                sparse_mode = 3
            (
                topk_indices,
                k_nope,
                k_pe,
                block_table,
                actual_seq_qlen,
                actual_seq_lengths_kv,
            ) = self.bind_offload_selection_context(
                coordinator,
                atten_id,
                topk_indices,
                k_nope,
                k_pe,
                block_table,
                actual_seq_qlen,
                actual_seq_lengths_kv,
                global_server_args_dict.get(
                    "offload_cache_update_timing", "before_attention"
                ),
            )
            o_ori, max_o, sum_o = get_attention_ops(self.use_sfa1).npu_sparse_flash_attention(
                query=q,
                key=k_nope,
                value=k_nope,
                query_rope=q_pe,
                key_rope=k_pe,
                sparse_indices=topk_indices,
                scale_value=layer.scaling,
                actual_seq_lengths_query=actual_seq_qlen,
                actual_seq_lengths_kv=actual_seq_lengths_kv,
                block_table=block_table,
                sparse_block_size=1,
                layout_query="TND",
                layout_kv=layout_kv,
                sparse_mode=sparse_mode,
                attention_mode=2,
                return_softmax_lse=True,
            )

            with (
                npu_stream_switch(forward_batch.can_run_with_graph, "lse_stream"),
                npu_limit_core(f"kvp_lse_{layer.attn_layer_id % 2}", flag=forward_batch.can_run_with_graph),
                npu_super_kernel("", "", flag=forward_batch.can_run_with_graph),
            ):
                lse = max_o + torch.log(sum_o)  # [1=N2, T, N]
                lse = lse.permute(1, 0, 2).reshape(T, -1).unsqueeze(-1)
                # Backing pools have per-request lengths; selected KV has
                # one length per query row. Keep the empty-owner LSE mask in
                # either representation before the cross-rank softmax.
                kv_lens_flat = actual_seq_lengths_kv.reshape(-1)
                bs = kv_lens_flat.shape[0]
                zero_kv_mask = (kv_lens_flat == 0)[:, None].expand(bs, T // bs).reshape(T)[:, None, None]
                lse = lse.masked_fill(zero_kv_mask.expand(-1, N, -1), float("-inf"))
                lse = lse.transpose(0, 1).squeeze(-1).contiguous()
                with npu_stream_switch(forward_batch.can_run_with_graph, "lse_all2all_stream"):
                    lse = lse.reshape(-1)
                    lse_all2all = torch.empty_like(lse)
                    torch.distributed.all_to_all_single(
                        lse_all2all, lse, group=get_lse_comm_group().device_group,
                    )


                lse_all2all = lse_all2all.reshape(kvp_size, -1)
                weights = torch.softmax(lse_all2all, dim=0).unsqueeze(-1)

            with (
                npu_limit_core(f"kvp_o_{layer.attn_layer_id % 2}", flag=forward_batch.can_run_with_graph),
                npu_super_kernel("", "", flag=forward_batch.can_run_with_graph),
            ):
                # SFA1 returns [T,N,D], unlike the legacy SFAD comm layout.
                o = o_ori.transpose(0, 1).contiguous()
                o = o.reshape(kvp_size, N // kvp_size, T, -1)
                o_all2all = all2all_wrapper(o, group=kvp_group.device_group)


                o_all2all = o_all2all.reshape(kvp_size, (N // kvp_size) * T, -1)
                o_all2all = o_all2all.to(max_o.dtype)

            if torchair_enable:
                o_all2all = tng.scope.npu_wait_tensor(o_all2all, weights)
            o = (o_all2all * weights).sum(dim=0)  # (local_head * T, D)
            attn_out = o.reshape(N // kvp_size, T, -1).to(o_ori.dtype)
        else:
            enable_sp_for_indexer = (
                is_prefill
                and global_server_args_dict.get("enable_mla_l1_5_cache", False)
                and global_server_args_dict["npu_enable_sp_for_indexer"]
            )
            (
                topk_indices,
                k_nope,
                k_pe,
                block_table,
                actual_seq_qlen,
                actual_seq_lengths_kv,
            ) = self.bind_offload_selection_context(
                coordinator,
                atten_id,
                topk_indices,
                k_nope,
                k_pe,
                block_table,
                actual_seq_qlen,
                actual_seq_lengths_kv,
                global_server_args_dict.get(
                    "offload_cache_update_timing", "before_attention"
                ),
            )
            if not enable_sp_for_indexer:
                attn_out, _, _ = get_attention_ops(self.use_sfa1).npu_sparse_flash_attention(
                    query=q_nope,
                    key=k_nope,
                    value=k_nope,
                    query_rope=q_pe,
                    key_rope=k_pe,
                    sparse_indices=topk_indices,
                    scale_value=layer.scaling,
                    actual_seq_lengths_query=actual_seq_qlen,
                    actual_seq_lengths_kv=actual_seq_lengths_kv,
                    block_table=block_table,
                    sparse_block_size=1,
                    layout_query="TND",
                    layout_kv=layout_kv,
                    sparse_mode=3,
                    attention_mode=2,
                    return_softmax_lse=False,
                )
            else:
                cp_size = get_attn_tp_group().world_size
                cp_rank = get_attn_tp_group().rank_in_group
                if not global_server_args_dict["npu_disable_dsa_head_parallel"]:
                    q_nope = get_attn_tp_group().all_gather(q_nope.contiguous(), dim=1)
                    q_pe = get_attn_tp_group().all_gather(q_pe.contiguous(), dim=1)
                global_q_head = q_nope.shape[1]

                if forward_batch.local_extend_lens_with_zero.sum().item() > 0:
                    if not global_server_args_dict["npu_disable_dsa_head_parallel"]:
                        slice_start = forward_batch.cu_attn_sp_token_nums_cpu[cp_rank].item()
                        slice_end = forward_batch.cu_attn_sp_token_nums_cpu[cp_rank + 1].item()
                        q_nope = q_nope[slice_start:slice_end]
                        q_pe = q_pe[slice_start:slice_end]

                    k_nope, _, _ = get_non_zero_q_meta(
                        k_nope,
                        forward_batch.global_cu_kv_lens_cpu,
                        forward_batch.local_extend_lens_with_zero,
                        forward_batch.local_total_kv_lens_with_zero,
                    )
                    k_pe, _, _ = get_non_zero_q_meta(
                        k_pe,
                        forward_batch.global_cu_kv_lens_cpu,
                        forward_batch.local_extend_lens_with_zero,
                        forward_batch.local_total_kv_lens_with_zero,
                    )
                    actual_seq_qlen = forward_batch.cur_seq_lengths_query_int32[1:]
                    actual_seq_lengths_kv = forward_batch.cur_seq_lengths_key_int32[1:]

                    attn_out, _, _ = get_attention_ops(self.use_sfa1).npu_sparse_flash_attention(
                        query=q_nope,
                        key=k_nope,
                        value=k_nope,
                        query_rope=q_pe,
                        key_rope=k_pe,
                        sparse_indices=topk_indices,
                        scale_value=layer.scaling,
                        actual_seq_lengths_query=actual_seq_qlen,
                        actual_seq_lengths_kv=actual_seq_lengths_kv,
                        block_table=block_table,
                        sparse_block_size=1,
                        layout_query="TND",
                        layout_kv=layout_kv,
                        sparse_mode=3,
                        attention_mode=2,
                        return_softmax_lse=False,
                    )
                else:
                    attn_out = torch.empty((0, global_q_head, k_nope.shape[2]), dtype=q_nope.dtype, device=q_nope.device)
                if not global_server_args_dict["npu_disable_dsa_head_parallel"]:
                    attn_sp_token_nums = get_attn_tp_group().get_local_sp_token_num(forward_batch.global_sp_num_tokens)
                    attn_out = get_attn_tp_group().all_gather(attn_out, dim=0, output_split_sizes=attn_sp_token_nums)
                    head_start = cp_rank * (global_q_head // cp_size)
                    head_end = (cp_rank + 1) * (global_q_head // cp_size)
                    attn_out = attn_out[:, head_start:head_end, :]
            if layout == "TND_NTD":
                attn_out = attn_out.transpose(0, 1).contiguous()
        return attn_out

    def forward_extend(
        self,
        q,
        k,
        v,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=False,
        q_pe: torch.Tensor | None = None,
        k_pe: torch.Tensor | None = None,
        topk_indices: torch.Tensor | None = None,
        layout="TND",
        absorbed=False,
        skip_gather_q: bool = False,
        atten_id: int = 0,
        sfad_packed_query: torch.Tensor | None = None,
    ):
        save_kv_cache = False  # kv cache saved in mla prolog on npu
        if topk_indices is not None:
            return self.forward_sparse(
                q,
                k,
                v,
                layer,
                forward_batch,
                save_kv_cache,
                q_pe,
                k_pe,
                topk_indices,
                absorbed,
                layout,
                skip_gather_q=skip_gather_q,
                atten_id=atten_id,
                sfad_packed_query=sfad_packed_query,
            )
        if absorbed:
            return self.forward_absorbed(q, k, v, layer, forward_batch, save_kv_cache, q_pe, k_pe, layout)
        return self.forward_norm_extend(q, k, v, layer, forward_batch, save_kv_cache, q_pe, k_pe, layout)

    def forward_norm_extend(
        self,
        q,
        k,
        v,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=False,
        q_pe: torch.Tensor | None = None,
        k_pe: torch.Tensor | None = None,
        layout="TND",
    ):
        save_kv_cache = False  # kv cache saved in mla prolog on npu
        if save_kv_cache:
            forward_batch.token_to_kv_pool.set_kv_buffer(layer, forward_batch.out_cache_loc, k, v)

        o = torch_npu.npu_fused_infer_attention_score(
            q,
            k,
            v,
            query_rope=q_pe,
            key_rope=k_pe,
            num_heads=layer.tp_q_head_num,
            num_key_value_heads=layer.tp_v_head_num,
            input_layout=layout,
            atten_mask=self.attn_mask,
            sparse_mode=3,
            actual_seq_lengths=forward_batch.attn_metadata.actual_seq_lengths,
            actual_seq_lengths_kv=forward_batch.attn_metadata.actual_seq_lengths_kv,
            scale=layer.scaling,
            next_tokens=0,
        )[0]
        return o

    def forward_decode(
        self,
        q,
        k,
        v,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=False,
        q_pe: torch.Tensor | None = None,
        k_pe: torch.Tensor | None = None,
        topk_indices: torch.Tensor | None = None,
        layout="TND",
        absorbed=True,
        skip_gather_q=False,
        atten_id: int = 0,
        sfad_packed_query: torch.Tensor | None = None,
    ):
        save_kv_cache = False  # kv cache saved in mla prolog on npu
        if topk_indices is not None:
            return self.forward_sparse(
                q,
                k,
                v,
                layer,
                forward_batch,
                save_kv_cache,
                q_pe,
                k_pe,
                topk_indices,
                absorbed,
                layout,
                skip_gather_q=skip_gather_q,
                atten_id=atten_id,
                sfad_packed_query=sfad_packed_query,
            )
        assert absorbed, "decode only support absorbed mode"
        return self.forward_absorbed(q, k, v, layer, forward_batch, save_kv_cache, q_pe, k_pe, layout)

    def forward_absorbed(
        self,
        q,
        k,
        v,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=False,
        q_pe: torch.Tensor | None = None,
        k_pe: torch.Tensor | None = None,
        # layout指输入数据的内存排布，比如BSND意思是(batch, seq_len, head_num, head_dim)。
        # TND中的T是指BS合并为T，同时不带padding, ND的含义不变。
        layout="TND",
    ):
        save_kv_cache = False  # kv cache saved in mla prolog on npu
        if save_kv_cache:
            forward_batch.token_to_kv_pool.set_kv_buffer(layer, forward_batch.out_cache_loc, k, v)
        block_size = forward_batch.token_to_kv_pool.page_size
        assert layer.tp_k_head_num == 1, "num of k head must be 1 in mla absorbed mode"
        k_head_dim = layer.v_head_dim if k_pe is not None else layer.qk_head_dim
        k = k.view(-1, block_size, k_head_dim)
        k_pe = k_pe.squeeze(dim=2)  # head num must be 1
        if "TND" in layout:
            q = q.squeeze(dim=1)
            q_pe = q_pe.squeeze(dim=1)

        use_kvp = global_server_args_dict["kvp_size"] > 1
        torchair_enable = forward_batch.all_decode_or_idle and ENV.npu_enable_graph and not ENV.npu_disable_graph_mode
        actual_seq_lengths_kv = forward_batch.attn_metadata.seq_lens_tensor.to(torch.int64)
        actual_seq_lengths = forward_batch.attn_metadata.query_len_tensor.to(torch.int64)
        block_table = forward_batch.attn_metadata.block_table
        if use_kvp:
            block_table = forward_batch.attn_metadata.kvp_block_table
            actual_seq_lengths_kv = forward_batch.attn_metadata.kvp_context_lens_tensor

        if not torchair_enable:
            actual_seq_lengths_kv = actual_seq_lengths_kv.cpu().tolist()
            actual_seq_lengths = actual_seq_lengths.cpu().tolist()
        op_scope = tng.ops if torchair_enable else torch.ops.npu
        if not global_server_args_dict["npu_disable_kv_nz"]:
            k = k.view(k.shape[0], 1, k.shape[2] // 16, k.shape[1], 16)
            k_pe = k_pe.view(k_pe.shape[0], 1, k_pe.shape[2] // 16, k_pe.shape[1], 16)

        if use_kvp:
            kvp_k_write = None
            kvp_k_pe_write = None
            kvp_group = get_attn_tp_group()
            kvp_rank = kvp_group.rank_in_group
            kvp_size = global_server_args_dict["kvp_size"]
            kv_lora_rank = q.shape[-1]
            qk_rope_head_dim = q_pe.shape[-1]
            q_total = torch.cat([q, q_pe], dim=-1)
            q_total = kvp_group.all_gather(q_total, dim=1)
            q, q_pe = torch.split(q_total, [kv_lora_rank, qk_rope_head_dim], dim=2)
            T = q.shape[0]
            N = q.shape[1]
            sparse_mode = 0
            atten_mask = None
            if kvp_rank == 0:
                sparse_mode = 3
                atten_mask = self.attn_mask
            o_ori, lse = op_scope.npu_fused_infer_attention_score(
                q,
                k,
                k,
                query_rope=q_pe,
                key_rope=k_pe,
                dequant_scale1=None,
                dequant_scale2=None,
                num_heads=N,
                num_key_value_heads=layer.tp_v_head_num,
                input_layout=layout,
                atten_mask=atten_mask,
                scale=layer.scaling,
                antiquant_mode=0,
                antiquant_scale=None,
                block_table=block_table,
                block_size=block_size,
                sparse_mode=sparse_mode,
                actual_seq_lengths_kv=actual_seq_lengths_kv,
                actual_seq_lengths=actual_seq_lengths,
                softmax_lse_flag=True,
            )
            lse = lse.masked_fill(torch.isinf(lse), -1e9)
            # TODO: use two stream to process o and lse in parallel
            # o: (N,T,D)
            o = o_ori.reshape(kvp_size, N // kvp_size, T, -1)
            o_all2all = all2all_wrapper(o, group=kvp_group.device_group)
            o_all2all = o_all2all.reshape(N, T, -1)
            # lse: (T, N, 1)
            lse = lse.transpose(0, 1).reshape(kvp_size, N // kvp_size, T).contiguous()
            lse_all2all = all2all_wrapper(lse, group=kvp_group.device_group)
            lse_all2all = lse_all2all.reshape(N, T, -1)

            o_all2all = o_all2all.reshape(kvp_size, (N // kvp_size) * T, -1)
            lse_all2all = lse_all2all.reshape(kvp_size, (N // kvp_size) * T)

            o_all2all_list = torch.chunk(o_all2all, kvp_size, dim=0)
            o_all2all_list = [t.squeeze(0) for t in o_all2all_list]
            lse_all2all_list = torch.chunk(lse_all2all, kvp_size, dim=0)
            lse_all2all_list = [t.squeeze(0) for t in lse_all2all_list]

            o, _ = torch_npu.npu_attention_update(lse_all2all_list, o_all2all_list, update_type=0)

            o = o.reshape(N // kvp_size, T, -1)
            # TODO: Following code can fix accurary bug, however it will bring many TensorMove operators in NPU graph.
            # Don't open this pass before HW fix the issue.
            # Host offload writes the current KV eagerly after the forward via
            # the fused host-copy path; every other configuration keeps this
            # in-graph write, which also covers the non-graph path. The test is
            # --offload-cache and not the pool's coordinator, because on THIS
            # path it is the complement of the guard on
            # fix_kvp_current_kv_cache in npu_longcat_flash and
            # npu_flash_nextn: exactly one of the two performs the write, and
            # both key on the same flag.
            #
            # This describes forward_absorbed only. forward_sparse divides the
            # same work by ARM, not by flag: _forward_kvp_sfad writes on
            # kvp_rank 0 with no offload term at all, and the SFA1 arm has no
            # in-graph write and leans on the hook. What keeps that safe is the
            # invariant that use_sfa1 implies --offload-cache, so the armless
            # write is only reached when the hook is enabled. Do not carry this
            # paragraph's reasoning over to the sparse path.
            if not global_server_args_dict["offload_cache"] and kvp_rank == 0:
                assert global_server_args_dict["npu_disable_kv_nz"]
                k = k.view(-1, kv_lora_rank)
                k_pe = k_pe.view(-1, qk_rope_head_dim)
                with tng.scope.super_kernel(None, ""):
                    kvp_k_write = kvp_cache_copy_rows_(
                        k,
                        forward_batch.attn_metadata.slot_mapping,
                        forward_batch.attn_metadata.kvp_current_slots_mapping,
                        o_ori if torchair_enable else None,
                    )
                    kvp_k_pe_write = kvp_cache_copy_rows_(
                        k_pe,
                        forward_batch.attn_metadata.slot_mapping,
                        forward_batch.attn_metadata.kvp_current_slots_mapping,
                        o_ori if torchair_enable else None,
                    )

            # Mainline wrote unconditionally on rank 0, so these were always
            # set here. The write above is now skipped when host offload owns
            # it, which leaves both None; _forward_kvp_sfad already guards its
            # own pe edge the same way.
            if torchair_enable and kvp_rank == 0:
                if kvp_k_write is not None:
                    o = tng.scope.npu_wait_tensor(o, kvp_k_write)
                if kvp_k_pe_write is not None:
                    o = tng.scope.npu_wait_tensor(o, kvp_k_pe_write)

        else:
            o, _ = op_scope.npu_fused_infer_attention_score(
                q,
                k,
                k,
                query_rope=q_pe,
                key_rope=k_pe,
                dequant_scale1=None,
                dequant_scale2=None,
                num_heads=layer.tp_q_head_num,
                num_key_value_heads=layer.tp_v_head_num,
                input_layout=layout,
                atten_mask=self.attn_mask,
                scale=layer.scaling,
                antiquant_mode=0,
                antiquant_scale=None,
                block_table=block_table,
                block_size=block_size,
                sparse_mode=3,
                next_tokens=0,
                actual_seq_lengths_kv=actual_seq_lengths_kv,
                actual_seq_lengths=actual_seq_lengths,
            )
        return o


def get_or_build_block_table(forward_batch: ForwardBatch, block_size: int, capture_graph: bool | None = False) -> torch.Tensor:

    req_pool_indices = forward_batch.req_pool_indices
    seq_lens = forward_batch.seq_lens
    if forward_batch.forward_mode.is_target_verify():
        seq_lens = seq_lens + forward_batch.spec_num_steps + 1
    forward_batch.seq_lens_cpu = forward_batch.seq_lens.cpu()
    req_pool_indices_cpu = req_pool_indices.cpu()
    req_pool_indices_cpu = req_pool_indices_cpu.tolist()

    if capture_graph:
        block_table = torch.full(
            (len(seq_lens), 2048),
            0,
            dtype=torch.int32,
            device=seq_lens.device,
        )
    else:
        block_table = torch.full(size=(len(seq_lens), 2048), fill_value=-1, dtype=torch.int32, device="npu")

        for seq_idx in range(len(seq_lens)):
            req_pool_info = forward_batch.req_to_token_pool.get_req_pool_info(req_pool_indices_cpu[seq_idx])
            req_num_pages = (seq_lens[seq_idx] + block_size - 1) // block_size
            page_indices = req_pool_info.alloced_slots[::block_size] // block_size
            if page_indices.numel() == 0:
                page_indices = torch.zeros((1,), dtype=torch.int32, device="npu")
            if req_num_pages > len(page_indices[:req_num_pages]):
                tmp = forward_batch.req_to_token_pool
                rid = req_pool_indices_cpu[seq_idx]
                logger.info(
                    f"page_indices err, {seq_lens=}, {req_pool_info.alloced_slots.shape=}, "
                    f"{req_pool_info.alloced_slots=}, {tmp.verified_lens[rid]=}, "
                    f"{tmp.alloced_lens[rid]=}, {tmp.req_to_token[rid]=}"
                )
                req_num_pages = len(page_indices[:req_num_pages])
            block_table[seq_idx, :req_num_pages] = page_indices[:req_num_pages]

    return block_table


_INT32_MAX = 2147483647

# query_len 只由 (block, bs) 决定; 缓存复用使图 capture/replay 拿到同一 tensor。
_DFLASH_QUERY_LEN_CACHE: dict = {}


def _dflash_query_len_tensor(block: int, bs: int, device) -> torch.Tensor:
    """累积形式 [block, 2*block, ..., bs*block]。"""
    key = (block, bs, str(device))
    tensor = _DFLASH_QUERY_LEN_CACHE.get(key)
    if tensor is None:
        tensor = torch.arange(block, bs * block + 1, block, dtype=torch.int64, device=device)
        _DFLASH_QUERY_LEN_CACHE[key] = tensor
    return tensor


def update_dflash_extend_meta(
    forward_batch: ForwardBatch,
    model_runner: ModelRunner,
) -> None:
    """按 draft 块前向的语义重建 attn_metadata 的相关字段。"""
    attn_metadata = forward_batch.attn_metadata
    assert attn_metadata is not None, "DFLASH_EXTEND replay 前 target verify 应已建好 attn_metadata"
    block = forward_batch.spec_num_steps + 1  # DFlash 强制 num_steps = block-1
    bs = forward_batch.batch_size

    # seq_lens 已是 KV 总长 (verified_lens_new + block), 不能像 TARGET_VERIFY 再 +block。
    attn_metadata.seq_lens_tensor = forward_batch.seq_lens.to(torch.int64)
    attn_metadata.query_len_tensor = _dflash_query_len_tensor(block, bs, forward_batch.seq_lens.device)
    attn_metadata.context_lens_tensor = attn_metadata.seq_lens_tensor - block
    attn_metadata.slot_mapping = forward_batch.out_cache_loc.clone().to(torch.int64)
    attn_metadata.num_prefill_tokens = forward_batch.new_tokens_total
    attn_metadata.num_decode_tokens = 0

    # block_table: 全局页号全量表, dtype 须 int32 (算子硬约束)。
    page_size = model_runner.page_size

    context_len = model_runner.model_config.context_len
    block_table_size = (context_len + page_size - 1) // page_size + 1 + 1
    block_table = torch.full((bs, block_table_size), -1, dtype=torch.int32, device="npu")
    block_len = min(model_runner.kv_allocator.req_to_page.size(-1), block_table_size)
    block_table[:, :block_len] = model_runner.kv_allocator.req_to_page[forward_batch.req_pool_indices, :block_len]
    attn_metadata.block_table = block_table

    # DFlash SWA sink/window 由 forward_extend 传 pre_tokens/next_tokens 给专用算子表达。


def get_local_kv_state(
    global_block_table: torch.Tensor,  # (B, PS), device
    global_kv_lens: torch.Tensor,  # (B,),    device, 写入前的长度, int64
    reserved_block_num: int,
    page_size: int,
    kvp_size: int,
    current_rank: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    B, PS = global_block_table.shape
    device = global_block_table.device

    # 写入前/后的长度
    pre_seq_lens = global_kv_lens  # (B,) 写入前
    kvp_size_shift = int(math.log(kvp_size, 2))
    page_size_shift = int(math.log(page_size, 2))

    # ------------------------------------------------------------------ #
    # 1. local_block_table
    # ------------------------------------------------------------------ #
    is_reserved = global_block_table < reserved_block_num  # (B, PS)
    adjusted = global_block_table - reserved_block_num  # (B, PS)
    belongs_to_rank = (adjusted % kvp_size) == current_rank  # (B, PS)
    local_idx = (adjusted >> kvp_size_shift) + reserved_block_num  # (B, PS)
    is_padding = global_block_table == 0  # (B, PS)

    # 每个位置映射后的值（不考虑是否属于本rank）
    mapped = torch.where(
        is_padding,
        torch.zeros_like(global_block_table),
        torch.where(
            is_reserved,
            global_block_table,  # 保留块不变
            torch.where(
                belongs_to_rank,
                local_idx,  # 本rank映射
                torch.zeros_like(global_block_table),  # 其他rank置0
            ),
        ),
    )
    # 有效位置：保留块 或 属于本rank的非padding块
    valid = (is_reserved & ~is_padding) | (belongs_to_rank & ~is_padding)  # (B, PS)

    # 利用 sort trick：对 ~valid 升序排序，False(有效)排前，True(无效)排后
    # stable=True 保证有效位置内部的相对顺序不变
    order = (~valid).to(torch.float32)  # (B, PS) 0=有效,1=无效
    perm = order.argsort(dim=1, stable=True)  # (B, PS)

    local_block_table = mapped.gather(1, perm)  # (B, PS)

    # ------------------------------------------------------------------ #
    # 2. local_kv_lens，基于写入前的长度
    # ------------------------------------------------------------------ #
    pre_total_blocks = (pre_seq_lens + page_size - 1) >> page_size_shift  # (B,)

    col_idx = torch.arange(PS, device=device).unsqueeze(0)  # (1, PS)
    valid_mask = col_idx < pre_total_blocks.unsqueeze(1)  # (B, PS)
    rank_mask = belongs_to_rank & valid_mask & (~is_reserved)  # (B, PS)

    # 最后一个有效 block 是否属于当前 rank
    last_col = (pre_total_blocks - 1).clamp(0, PS - 1)  # (B,)
    last_block_global = global_block_table.gather(1, last_col.unsqueeze(1)).squeeze(1)  # (B,)
    last_adjusted = last_block_global - reserved_block_num  # (B,)
    last_belongs = (last_adjusted % kvp_size == current_rank) & (last_block_global >= reserved_block_num)  # (B,)

    local_block_count = rank_mask.to(torch.int32).sum(dim=1)  # (B,)

    remainder = pre_seq_lens % page_size  # (B,)
    last_tokens = torch.where(remainder == 0, torch.full_like(remainder, page_size), remainder)  # (B,)

    local_kv_lens = torch.where(
        last_belongs & (local_block_count > 0), (local_block_count - 1) * page_size + last_tokens, local_block_count * page_size
    )  # (B,) 写入前

    return local_block_table, local_kv_lens  # int32 & int64


def build_req_level_owner_stats(
    forward_batch: ForwardBatch,
    global_hist_lens: torch.Tensor,  # [B], global history length before current decode tokens
    local_req_lens: torch.Tensor,  # [B], local kv length on this rank (rank0 already includes current)
    q_lens: torch.Tensor,  # [B], query length on this rank
    num_init: int,
    num_local: int,
    max_q_len: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
        init_cnt_req:  [B], number of init tokens owned by this rank.
        local_cnt_req: [T] per-token (TND flat, token (b,j) at b*max_q_len+j).
                       rank0: value = w_j + min(j+1, num_local), w_j = 第 j 行全局滑动窗口内
                              归本 rank 的历史 token 数;
                       非 rank0: value = w_j，无当前 token，但历史窗口同样逐行移动。
    """
    device = global_hist_lens.device
    B = global_hist_lens.numel()
    global_hist_lens = torch.clamp(global_hist_lens, min=0)

    kv_pool = forward_batch.token_to_kv_pool
    from sglang.srt.env import global_server_args_dict

    reserved_block_num = 1
    if global_server_args_dict["kvp_size"] > 1:
        reserved_block_num = 1 + kv_pool.max_batch_size

    req_to_token_table = forward_batch.req_to_token_pool.req_to_token
    req_row_indices = forward_batch.req_pool_indices
    C = req_to_token_table.shape[1]
    batch_req_to_token = req_to_token_table.index_select(0, req_row_indices)

    def _gather_req_tokens(col_indices: torch.Tensor) -> torch.Tensor:
        col_indices = torch.clamp(col_indices, min=0, max=C - 1)
        return batch_req_to_token.gather(1, col_indices)

    def _is_local_token(token_indices: torch.Tensor) -> torch.Tensor:
        page_size = 1 << kv_pool.page_shift
        group_size = kv_pool.group_mask + 1
        pages = torch.div(token_indices, page_size, rounding_mode="trunc")
        return (pages < reserved_block_num) | (((pages - reserved_block_num) % group_size) == kv_pool.kvp_rank)

    init_prefix_len = torch.minimum(
        global_hist_lens,
        torch.full_like(global_hist_lens, num_init),
    )

    if num_init > 0:
        init_pos = torch.arange(num_init, device=device, dtype=torch.int64)[None, :]
        init_tokens = _gather_req_tokens(init_pos.expand(B, -1))
        init_valid = init_pos < init_prefix_len[:, None]
        init_cnt_req = (_is_local_token(init_tokens) & init_valid).sum(dim=1, dtype=torch.int32)
    else:
        init_cnt_req = torch.zeros(B, device=device, dtype=torch.int32)

    local_start = torch.minimum(
        global_hist_lens,
        torch.maximum(
            init_prefix_len,
            torch.clamp(global_hist_lens + 1 - num_local, min=0),
        ),
    )
    if num_local > 0:
        local_offsets = torch.arange(num_local, device=device, dtype=torch.int64)[None, :]
        local_pos = local_start[:, None] + local_offsets
        local_tokens = _gather_req_tokens(local_pos)
        local_valid = local_pos < global_hist_lens[:, None]
        local_cnt_req = (_is_local_token(local_tokens) & local_valid).sum(dim=1, dtype=torch.int32)
    else:
        local_cnt_req = torch.zeros(B, device=device, dtype=torch.int32)

    # local_tensor 只支持 per-token [T](TND 扁平, token (b,j) 位于 b*max_q_len+j)。
    # rank0: 第 j 行传精确计数 w_j + min(j+1, num_local),
    #   w_j = 全局窗口 [max(init_prefix, hist+j+1-num_local), hist) 内归本 rank 的历史 token 数,
    #   min(j+1, num_local) = 段尾当前 token(全在 rank0, j+1>num_local 时意图窗口只剩当前 token)。
    # 非 rank0: 虽无当前 KV，历史窗口仍随全局 query 位置逐行移动。
    # q_len 在该路径恒等于 max_q_len(decode=1 / verify=spec_num_steps+1, 同 L1307 的既有假设)。
    if num_local > 0 and max_q_len > 1:
        # verify chunk 内的 query 行号 j=0..J-1；第 j 行可见 q0..qj 对应的 j+1 个当前 KV。
        j_range = torch.arange(max_q_len, device=device, dtype=torch.int64)[None, :]
        # 第 j 行 local window 的历史起点 hist+j+1-num_local，先限制为非负。
        win_start = torch.clamp(global_hist_lens[:, None] + 1 + j_range - num_local, min=0)
        # local 强选不与 init prefix 重叠，历史起点不能早于 init_prefix_len。
        win_start = torch.maximum(init_prefix_len[:, None], win_start)
        # 当当前 KV 已占满 local window 时历史区间为空，将起点截断到 hist。
        win_start = torch.minimum(global_hist_lens[:, None], win_start)  # (B, J)
        # 从每行的历史起点开始，枚举最多 num_local 个 req_to_token 位置。
        win_pos = win_start[:, :, None] + local_offsets[:, None, :]  # (B, J, num_local)
        # 只保留 [win_start, hist) 内的历史位置，屏蔽枚举到 hist 之后的位置。
        win_valid = win_pos < global_hist_lens[:, None, None]
        # 查出各历史位置的全局 token id，并恢复为 [B, J, num_local]。
        win_tokens = _gather_req_tokens(win_pos.reshape(B, -1)).reshape(B, max_q_len, num_local)
        # 统计每个 query 行的历史 local window 中实际属于本 rank 的 KV 数 w_j。
        w_j = (_is_local_token(win_tokens) & win_valid).sum(dim=2, dtype=torch.int32)  # (B, J)
        # 加上 rank0 中该行可见的 min(j+1, num_local) 个当前 KV，再展平为 per-token [T]。
        if kv_pool.kvp_rank == 0:
            w_j = w_j + (j_range + 1).clamp(max=num_local).to(torch.int32)
        local_cnt_req = w_j.reshape(-1)  # [T=B*J]
    elif kv_pool.kvp_rank == 0 and num_local > 0:
        # decode(q_len=1): per-token 退化为 w_0+1, [T]=[B]
        local_cnt_req = local_cnt_req + 1
    else:
        # 非 rank0 decode / num_local==0: [B] 展开为 [T]
        local_cnt_req = local_cnt_req.repeat_interleave(max_q_len)

    return init_cnt_req, local_cnt_req


def compute_local_lens(
    extend_lens: torch.Tensor,  # (B,) 当前 token 数
    history_kv_lens: torch.Tensor,  # (B,) 历史 kv 长度
    sp_num_tokens: torch.Tensor,  # (cp_size,) 每个 rank 的 token 数
    cp_rank: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
        local_extend_lens:   (B,) 当前 rank 负责的每个请求的 token 数
        local_total_kv_lens: (B,) 当前 rank 视角下每个请求的总 kv 长度
    """
    # ---- 当前 rank 负责的全局 token 范围 ----
    rank_start = sp_num_tokens[:cp_rank].sum()  # scalar
    rank_end = rank_start + sp_num_tokens[cp_rank]  # scalar

    # ---- 每个请求在全局 token 序号中的范围 ----
    # req_start[b] = sum(extend_lens[:b])
    # req_end[b]   = sum(extend_lens[:b+1])
    cumsum = torch.cumsum(extend_lens, dim=0)  # (B,)
    req_end = cumsum  # (B,)
    req_start = cumsum - extend_lens  # (B,)

    # ---- 与当前 rank 范围取交集 ----
    inter_start = torch.maximum(req_start, rank_start)  # (B,)
    inter_end = torch.minimum(req_end, rank_end)  # (B,)

    local_extend_lens = torch.clamp(inter_end - inter_start, min=0)  # (B,)

    # ---- local_total_kv_lens ----
    # 当前 rank 视角下，每个请求的 kv 长度 =
    #   历史 kv 长度
    #   + 当前 rank 之前所有 rank 处理的该请求的 token 数（已写入 kv）
    #   + 当前 rank 负责的该请求的 token 数
    #
    # 即：history_kv_lens + min(req_end, rank_end) - req_start
    #   （从请求开始到当前 rank 末尾，累计写入的 kv 数）
    #   但不超过请求本身的总长度
    tokens_up_to_rank_end = torch.clamp(torch.minimum(req_end, rank_end) - req_start, min=0)  # (B,)

    local_total_kv_lens = history_kv_lens + tokens_up_to_rank_end  # (B,)

    return local_extend_lens, local_total_kv_lens


def get_attn_sp_meta_npu(forward_batch: ForwardBatch, attn_metadata: AttentionMetadata):
    device = "npu"
    attn_sp_token_nums = get_attn_tp_group().get_local_sp_token_num(forward_batch.global_sp_num_tokens)
    cu_start_cpu = torch.tensor([0], dtype=torch.int32)
    global_cu_kv_lens_cpu = torch.concat((cu_start_cpu, torch.tensor(attn_metadata.actual_seq_lengths_kv, dtype=torch.int32)))
    cur_cu_extend_lens_cpu = torch.concat((cu_start_cpu, torch.tensor(attn_metadata.actual_seq_lengths, dtype=torch.int32)))
    attn_sp_token_nums_cpu = torch.tensor(attn_sp_token_nums, dtype=torch.int32)
    cu_attn_sp_token_nums_cpu = torch.concat((cu_start_cpu, torch.cumsum(attn_sp_token_nums_cpu, dim=0)))
    global_kv_lens_cpu = global_cu_kv_lens_cpu[1:] - global_cu_kv_lens_cpu[:-1]
    cur_extend_lens_cpu = cur_cu_extend_lens_cpu[1:] - cur_cu_extend_lens_cpu[:-1]
    cp_rank = get_attn_tp_group().rank_in_group

    local_extend_lens_with_zero, local_total_kv_lens_with_zero = compute_local_lens(
        cur_extend_lens_cpu, global_kv_lens_cpu - cur_extend_lens_cpu, attn_sp_token_nums_cpu, cp_rank
    )
    forward_batch.local_extend_lens_with_zero = local_extend_lens_with_zero
    forward_batch.local_total_kv_lens_with_zero = local_total_kv_lens_with_zero
    forward_batch.global_cu_kv_lens_cpu = global_cu_kv_lens_cpu
    forward_batch.cu_attn_sp_token_nums_cpu = cu_attn_sp_token_nums_cpu

    if local_extend_lens_with_zero.sum().item() > 0:
        _, local_extend_lens, local_total_kv_lens = get_non_zero_q_meta(
            None, global_cu_kv_lens_cpu, local_extend_lens_with_zero, local_total_kv_lens_with_zero
        )
        cur_seq_lengths_query_cpu = torch.concat((cu_start_cpu, torch.cumsum(local_extend_lens, dim=0)))
        cu_local_total_kv_lens_cpu = torch.concat((cu_start_cpu, torch.cumsum(local_total_kv_lens, dim=0)))
        # past_key_states = get_local_past_key_states(past_key_states, global_cu_kv_lens_cpu, cu_local_total_kv_lens_cpu)

        cur_seq_lengths_query = cur_seq_lengths_query_cpu.to(device, non_blocking=True)
        cur_seq_lengths_key = cu_local_total_kv_lens_cpu.to(device, non_blocking=True)

        forward_batch.cur_seq_lengths_query_int32 = cur_seq_lengths_query.to(torch.int32)
        forward_batch.cur_seq_lengths_key_int32 = cur_seq_lengths_key.to(torch.int32)
        forward_batch.cur_seq_lengths_query = cur_seq_lengths_query
        forward_batch.cur_seq_lengths_key = cur_seq_lengths_key

    # use indexer_balance
    cur_extend_lens = cur_extend_lens_cpu.tolist()
    global_kv_lens = global_kv_lens_cpu.tolist()
    if get_index_balancer().need_balance(cur_extend_lens, global_kv_lens):
        indexer_balance_attn_sp_token_nums = get_index_balancer().balance(cur_extend_lens, global_kv_lens)
        ib_attn_sp_token_nums_cpu = torch.tensor(indexer_balance_attn_sp_token_nums, dtype=torch.int32)
        ib_local_extend_lens_with_zero, ib_local_total_kv_lens_with_zero = compute_local_lens(
            cur_extend_lens_cpu, global_kv_lens_cpu - cur_extend_lens_cpu, ib_attn_sp_token_nums_cpu, cp_rank
        )
        forward_batch.attn_sp_token_nums = attn_sp_token_nums
        forward_batch.ib_attn_sp_token_nums = indexer_balance_attn_sp_token_nums
        forward_batch.ib_local_extend_lens_with_zero = ib_local_extend_lens_with_zero
        forward_batch.ib_local_total_kv_lens_with_zero = ib_local_total_kv_lens_with_zero
        ib_input_split_sizes, ib_output_split_sizes = get_all2all_params_for_index_balance(
            attn_sp_token_nums, indexer_balance_attn_sp_token_nums, cp_rank
        )
        forward_batch.ib_input_split_sizes = ib_input_split_sizes
        forward_batch.ib_output_split_sizes = ib_output_split_sizes
        # if get_attn_tp_group().rank_in_group == 0:
        #     logger.info(f"{cur_extend_lens=} {global_kv_lens=} {attn_sp_token_nums=} {indexer_balance_attn_sp_token_nums=}")

        if ib_local_total_kv_lens_with_zero.sum().item() > 0:
            _, ib_local_extend_lens, ib_local_total_kv_lens = get_non_zero_q_meta(
                None, global_cu_kv_lens_cpu, ib_local_extend_lens_with_zero, ib_local_total_kv_lens_with_zero
            )
            ib_cur_seq_lengths_query_cpu = torch.concat((cu_start_cpu, torch.cumsum(ib_local_extend_lens, dim=0)))
            ib_cur_local_total_kv_lens_cpu = torch.concat((cu_start_cpu, torch.cumsum(ib_local_total_kv_lens, dim=0)))
            ib_cur_seq_lengths_query = ib_cur_seq_lengths_query_cpu.to(device, non_blocking=True)
            ib_cur_seq_lengths_key = ib_cur_local_total_kv_lens_cpu.to(device, non_blocking=True)
            forward_batch.ib_cur_seq_lengths_query = ib_cur_seq_lengths_query
            forward_batch.ib_cur_seq_lengths_key = ib_cur_seq_lengths_key


def get_attn_meta_npu(forward_batch: ForwardBatch, model_runner: ModelRunner, capture_graph: bool | None = False):
    attn_metadata = AttentionMetadata()
    attn_metadata.is_profile_run = False
    attn_metadata.context_chunk_workspace = None
    attn_metadata.seq_lens_tensor = forward_batch.seq_lens.to(torch.int64)
    attn_metadata.slot_mapping = forward_batch.out_cache_loc.clone().to(torch.int64)
    attn_metadata.all_decode_or_idle = forward_batch.all_decode_or_idle
    decode_seq_per_batch = 1
    config = model_runner.model_config.hf_text_config
    enable_nsa = is_deepseek_nsa(config)

    if forward_batch.forward_mode.is_extend():
        if forward_batch.forward_mode.is_dflash_extend():
            # seq_lens 已是 KV 总长, 不能像 TARGET_VERIFY 分支那样再 +block。
            block = forward_batch.spec_num_steps + 1  # DFlash 强制 num_steps = block-1
            attn_metadata.num_prefill_tokens = forward_batch.new_tokens_total
            attn_metadata.query_len_tensor = _dflash_query_len_tensor(block, forward_batch.batch_size, attn_metadata.seq_lens_tensor.device)
            attn_metadata.context_lens_tensor = attn_metadata.seq_lens_tensor - block
            decode_seq_per_batch = block
        elif forward_batch.forward_mode.is_target_verify() or forward_batch.forward_mode.is_draft_extend():
            attn_metadata.num_prefill_tokens = forward_batch.new_tokens_total
            attn_metadata.query_len_tensor = torch.tensor(
                list(accumulate([(forward_batch.spec_num_steps + 1) for _ in range(forward_batch.batch_size)])), dtype=torch.int64, pin_memory=True
            ).to(device="npu", non_blocking=True)
            attn_metadata.context_lens_tensor = attn_metadata.seq_lens_tensor
            attn_metadata.seq_lens_tensor = attn_metadata.seq_lens_tensor + forward_batch.spec_num_steps + 1
            decode_seq_per_batch = forward_batch.spec_num_steps + 1
        else:
            attn_metadata.seq_lens = attn_metadata.seq_lens_tensor.cpu()
            attn_metadata.num_prefill_tokens = forward_batch.extend_num_tokens
            attn_metadata.context_lens_tensor = forward_batch.extend_prefix_lens
            attn_metadata.context_lens = forward_batch.extend_prefix_lens_cpu
            attn_metadata.query_len_tensor = forward_batch.extend_seq_lens.to(dtype=torch.int64)
            attn_metadata.actual_seq_lengths = list(accumulate(forward_batch.extend_seq_lens_cpu))
            seq_lens = [x + y for x, y in zip(forward_batch.extend_prefix_lens_cpu, forward_batch.extend_seq_lens_cpu)]
            attn_metadata.actual_seq_lengths_kv = list(accumulate(seq_lens))
            attn_metadata.query_len_tensor = torch.tensor(attn_metadata.actual_seq_lengths, dtype=torch.int64, pin_memory=True).to(
                device="npu", non_blocking=True
            )
            attn_metadata.seq_lens_tensor = torch.tensor(seq_lens, dtype=torch.int64, pin_memory=True).to(device="npu", non_blocking=True)
            kv_index_list = []
            enable_mla_l1_5_cache = global_server_args_dict.get("enable_mla_l1_5_cache", False)
            if not global_server_args_dict["npu_disable_kv_nz"]:
                if not enable_mla_l1_5_cache or (model_runner.is_draft_worker and model_runner.spec_algorithm.is_dflash()):
                    kv_block_index_list = []
                    bias = 0
                    for seq_idx, seq_len in enumerate(seq_lens):
                        alloced_slots = forward_batch.req_to_token_pool.req_to_token[forward_batch.req_pool_indices[seq_idx]]
                        kv_block_index_list.append(alloced_slots[:seq_len:128] // 128)
                        kv_index_list.append(torch.arange(seq_len, dtype=torch.int32, device=alloced_slots.device) + bias)
                        bias = bias + (seq_len + 127) // 128 * 128
                    attn_metadata.kv_index_list = torch.cat(kv_index_list, dim=0)
                    attn_metadata.kv_block_index_list = torch.cat(kv_block_index_list, dim=0)

                else:
                    block_size = forward_batch.token_to_kv_pool.page_size
                    # 1. Keep the original global token order for all-gather restore metadata.
                    # 2. Reuse the same mapping result to rebuild mini pool indices per sequence
                    #    using each sequence's local token count.
                    if enable_nsa:
                        from sglang.srt.layers.attention.npu_attn.npu_prefill_kvp import build_prefill_kvp_metadata

                        local_indices, per_rank_count, inv_perm, local_seq_lens = build_prefill_kvp_metadata(
                            forward_batch.token_to_kv_pool,
                            [forward_batch.req_to_token_pool.req_to_token[forward_batch.req_pool_indices[i]] for i in range(len(seq_lens))],
                            seq_lens,
                        )
                    else:
                        kv_slots = torch.cat(
                            [
                                forward_batch.req_to_token_pool.req_to_token[forward_batch.req_pool_indices[seq_idx]][:seq_len]
                                for seq_idx, seq_len in enumerate(seq_lens)
                            ],
                            dim=0,
                        )
                        mask, local_indices, per_rank_count, _, inv_perm = forward_batch.token_to_kv_pool.remap_tokens_for_kvp(
                            kv_slots,
                        )
                        local_seq_lens = [int(seq_mask.sum().item()) for seq_mask in mask.split(seq_lens)]
                    attn_metadata.per_rank_count = per_rank_count
                    attn_metadata.inv_perm = inv_perm
                    kv_block_index_list = []
                    kv_index_list = []
                    bias = 0
                    local_offset = 0
                    for local_seq_len in local_seq_lens:
                        seq_local_indices = local_indices[local_offset : local_offset + local_seq_len]
                        if local_seq_len > 0:
                            kv_block_index_list.append(seq_local_indices[::block_size] // block_size)
                            kv_index_list.append(
                                torch.arange(
                                    local_seq_len,
                                    dtype=torch.int32,
                                    device=local_indices.device,
                                )
                                + bias
                            )
                        bias = bias + (local_seq_len + block_size - 1) // block_size * block_size
                        local_offset += local_seq_len
                    attn_metadata.kv_block_index_list = (
                        torch.cat(kv_block_index_list, dim=0)
                        if kv_block_index_list
                        else torch.empty((0,), dtype=torch.int32, device=local_indices.device)
                    )
                    attn_metadata.kv_index_list = (
                        torch.cat(kv_index_list, dim=0) if kv_index_list else torch.empty((0,), dtype=torch.int32, device=local_indices.device)
                    )
            else:
                if not enable_mla_l1_5_cache or (model_runner.is_draft_worker and model_runner.spec_algorithm.is_dflash()):
                    for seq_idx, seq_len in enumerate(seq_lens):
                        alloced_slots = forward_batch.req_to_token_pool.req_to_token[forward_batch.req_pool_indices[seq_idx]]
                        kv_index_list.append(alloced_slots[:seq_len])
                    attn_metadata.kv_index_list = torch.cat(kv_index_list, dim=0)
                else:
                    kv_slots = torch.cat(
                        [
                            forward_batch.req_to_token_pool.req_to_token[forward_batch.req_pool_indices[seq_idx]][:seq_len]
                            for seq_idx, seq_len in enumerate(seq_lens)
                        ],
                        dim=0,
                    )
                    mask, local_indices, per_rank_count, _, inv_perm = forward_batch.token_to_kv_pool.remap_tokens_for_kvp(
                        kv_slots,
                    )
                    attn_metadata.per_rank_count = per_rank_count
                    attn_metadata.inv_perm = inv_perm
                    attn_metadata.kv_index_list = local_indices

                    attn_metadata.dense_slot_mapping_idx = (attn_metadata.slot_mapping > 0).nonzero().squeeze(-1)
                    attn_metadata.dense_slot_mapping = attn_metadata.slot_mapping[attn_metadata.dense_slot_mapping_idx]
            if isinstance(attn_metadata.seq_lens, torch.Tensor):
                attn_metadata.seq_lens = attn_metadata.seq_lens.tolist()
            if isinstance(attn_metadata.context_lens, torch.Tensor):
                attn_metadata.context_lens = attn_metadata.context_lens.tolist()
                attn_metadata.query_len = [a - b for a, b in zip(attn_metadata.seq_lens, attn_metadata.context_lens)]
                attn_metadata.max_query_len = max(attn_metadata.query_len)
                attn_metadata.max_seq_len = max(attn_metadata.seq_lens)
            if enable_nsa:
                attn_metadata.actual_seq_lengths_kv_tensor = torch.tensor(attn_metadata.actual_seq_lengths_kv, pin_memory=True).to(
                    device="npu", non_blocking=True
                )
                attn_metadata.actual_seq_lengths_kv_tensor_int32 = attn_metadata.actual_seq_lengths_kv_tensor.to(dtype=torch.int32)
                attn_metadata.actual_seq_lengths_tensor = torch.tensor(attn_metadata.actual_seq_lengths, pin_memory=True).to(
                    device="npu", non_blocking=True
                )
                attn_metadata.actual_seq_lengths_tensor_int32 = attn_metadata.actual_seq_lengths_tensor.to(torch.int32)

    elif forward_batch.forward_mode.is_decode():
        attn_metadata.num_prefill_tokens = 0
        attn_metadata.context_lens_tensor = attn_metadata.seq_lens_tensor - 1
        attn_metadata.query_len_tensor = torch.tensor(
            list(accumulate([1 for _ in range(forward_batch.batch_size)])), dtype=torch.int64, pin_memory=True
        ).to(device="npu", non_blocking=True)
    else:
        raise ValueError(f"Unsupported forward_mode:{forward_batch.forward_mode}")
    if enable_nsa:
        attn_metadata.seq_lens_tensor_int32 = attn_metadata.seq_lens_tensor.to(dtype=torch.int32)
        attn_metadata.cu_seq_lens_tensor_int32 = torch.cumsum(
            attn_metadata.seq_lens_tensor_int32,
            dim=0,
            dtype=torch.int32,
        )

    context_len = model_runner.model_config.context_len
    page_size = model_runner.page_size
    # 除以page_size并向上取整，额外增加一个page以应对attention溢出最大上文长度的部分（疑是bug，但是难以定位）
    # 额外加1 for npu and kvp_size > 1, used for current token
    block_table_size = (context_len + page_size - 1) // page_size + 1 + 1
    block_len = 0
    if capture_graph:
        attn_metadata.num_decode_tokens = len(forward_batch.seq_lens)
        attn_metadata.block_table = torch.full(
            (len(forward_batch.seq_lens), block_table_size),
            0,
            dtype=torch.int32,
            device=forward_batch.seq_lens.device,
        )
    else:
        ### build block table
        attn_metadata.block_table = torch.full(size=(len(forward_batch.seq_lens), block_table_size), fill_value=-1, dtype=torch.int32, device="npu")
        block_len = min(model_runner.kv_allocator.req_to_page.size(-1), attn_metadata.block_table.size(-1))
        attn_metadata.block_table[:, :block_len] = model_runner.kv_allocator.req_to_page[forward_batch.req_pool_indices, :block_len]
        attn_metadata.num_decode_tokens = forward_batch.new_tokens_total - attn_metadata.num_prefill_tokens
    kvp_size = global_server_args_dict["kvp_size"]

    # DFLASH draft 是 GQA 且 pool 不分片 (全局页号), 豁免 kvp 切分。
    dflash_draft = model_runner.is_draft_worker and model_runner.spec_algorithm.is_dflash()

    if kvp_size > 1 and not dflash_draft:
        kvp_rank = get_attn_tp_group().rank_in_group
        global_block_table = attn_metadata.block_table
        global_kv_lens = attn_metadata.context_lens_tensor
        reserved_block_num = model_runner.kv_allocator.max_batch_size + 1

        kvp_block_table, kvp_context_lens_tensor = get_local_kv_state(
            global_block_table, global_kv_lens, reserved_block_num, page_size, kvp_size, kvp_rank
        )
        attn_metadata.kvp_current_slots_mapping = attn_metadata.slot_mapping
        if kvp_rank == 0:
            batch_size = kvp_block_table.shape[0]
            device = kvp_block_table.device
            actual_block_num_tensor = ((kvp_context_lens_tensor + page_size - 1) // page_size).to(torch.int32)  # [batch_size]

            # assert (actual_block_num_tensor + 1).max().item() <= block_table_size

            bs_idx_tensor = torch.arange(batch_size, dtype=torch.int32, device=device)  # [batch_size]

            pad_block_idx_tensor = torch.arange(1, batch_size + 1, dtype=torch.int32, device=device)  # [batch_size]

            kvp_block_table[bs_idx_tensor, actual_block_num_tensor] = pad_block_idx_tensor

            kv_len_expanded = kvp_context_lens_tensor.unsqueeze(1).expand(batch_size, decode_seq_per_batch)

            si = torch.arange(decode_seq_per_batch, dtype=kvp_context_lens_tensor.dtype, device=device).unsqueeze(0)  # [1, decode_seq_per_batch]

            pos = kv_len_expanded + si
            kvp_block_talbe_idx_tensor = (pos // page_size).to(torch.int32)
            kvp_slot_mapping_idx_in_page = (pos % page_size).to(torch.int32)
            kvp_slot_mapping_page_idx = kvp_block_table.gather(1, kvp_block_talbe_idx_tensor)  # [batch_size, decode_seq_per_batch]

            kvp_current_slots_mapping = (
                (kvp_slot_mapping_page_idx * page_size + kvp_slot_mapping_idx_in_page).flatten().to(torch.int64)
            )  # [batch_size * decode_seq_per_batch]

            kvp_context_lens_tensor = kvp_context_lens_tensor + decode_seq_per_batch
            attn_metadata.slot_mapping = kvp_current_slots_mapping

        attn_metadata.kvp_block_table = kvp_block_table
        attn_metadata.kvp_context_lens_tensor = kvp_context_lens_tensor

        if enable_nsa:
            # add for 3s indexer
            num_init_tokens = getattr(config, "index_init_tokens", 0)
            num_local_tokens = getattr(config, "index_local_tokens", 0)
            index_topk = getattr(config, "index_topk", 0)

            device = attn_metadata.query_len_tensor.device
            non_cu_query_len_tensor = attn_metadata.seq_lens_tensor - attn_metadata.context_lens_tensor

            init_cnt_req, local_cnt_req = build_req_level_owner_stats(
                forward_batch,
                attn_metadata.context_lens_tensor,
                attn_metadata.kvp_context_lens_tensor,
                non_cu_query_len_tensor,
                num_init_tokens,
                num_local_tokens,
                decode_seq_per_batch,
            )

            token_to_batch_id = (
                torch.arange(forward_batch.batch_size, dtype=torch.int32, device=device)
                .view(forward_batch.batch_size, 1)
                .repeat(1, decode_seq_per_batch)
                .flatten()
            )
            token_kv_len = attn_metadata.kvp_context_lens_tensor.to(torch.int32)[token_to_batch_id]

            mask = torch.arange(index_topk, dtype=torch.int32, device=device)[None, :] >= token_kv_len[:, None]
            attn_metadata.out_of_kv_len_mask = mask[:, None, :]

            attn_metadata.init_cnt_req = init_cnt_req
            attn_metadata.local_cnt_req = local_cnt_req
            # 计算每个token对应的kv len.
            # if kvp_rank == 0:
            #     T = forward_batch.batch_size * decode_seq_per_batch
            #     cu0_query_len_tensor_int32 = attn_metadata.query_len_tensor.to(torch.int32)
            #     cu0_query_len_tensor_int32 = torch.cat([torch.zeros(1, dtype=torch.int32, device=device), cu0_query_len_tensor_int32])
            #     token_q_len = non_cu_query_len_tensor_int32[token_to_batch_id]
            #     token_start = cu0_query_len_tensor_int32[token_to_batch_id]
            #     local_pos = torch.arange(T, dtype=torch.int32, device=device) - token_start
            #     kv_pos = token_kv_len - token_q_len + local_pos + 1 # 每个token对应的kv len
            #     local_curr = local_cnt_req[token_to_batch_id] + local_pos + 1 # 每个token对应的local token数
            #     attn_metadata.kv_local_start = kv_pos - local_curr # 每个token对应的local开始的index
            #     attn_metadata.kv_init_end = init_cnt_req[token_to_batch_id] # 每个token对应的init的个数
            #     attn_metadata.kv_pos = kv_pos
            # else:
            #     kv_pos = token_kv_len
            #     attn_metadata.kv_local_start = kv_pos - local_cnt_req[token_to_batch_id]
            #     attn_metadata.kv_init_end = init_cnt_req[token_to_batch_id]
            #     attn_metadata.kv_pos = kv_pos

    if not forward_batch.all_decode_or_idle:

        def get_sp_token_num(x, rank):
            base = x // get_attn_tp_group().world_size
            rem = x % get_attn_tp_group().world_size
            if rank < rem:
                return base + 1
            else:
                return base

        if forward_batch.global_num_tokens is None:
            forward_batch.global_num_tokens = [forward_batch.extend_num_tokens] * get_attn_tp_group().world_size
        forward_batch.global_sp_num_tokens = [
            get_sp_token_num(x, i % get_attn_tp_group().world_size) for i, x in enumerate(forward_batch.global_num_tokens)
        ]

    if enable_nsa and (not forward_batch.all_decode_or_idle):
        get_attn_sp_meta_npu(forward_batch, attn_metadata)
        if (
            not model_runner.is_draft_worker
            and forward_batch.forward_mode.is_extend()
            and not forward_batch.forward_mode.is_draft_extend()
            and not forward_batch.forward_mode.is_target_verify()
            and global_server_args_dict.get("enable_mla_l1_5_cache", False)
            and global_server_args_dict.get("npu_enable_sp_for_indexer", False)
        ):
            from sglang.srt.layers.attention.npu_attn.npu_index_k_route import build_index_k_route

            pool = forward_batch.token_to_kv_pool
            group = get_attn_tp_group()
            query_splits = (
                forward_batch.ib_attn_sp_token_nums
                if forward_batch.ib_local_extend_lens_with_zero is not None
                else group.get_local_sp_token_num(forward_batch.global_sp_num_tokens)
            )
            attn_metadata.index_k_route = build_index_k_route(
                [forward_batch.req_to_token_pool.req_to_token[idx] for idx in forward_batch.req_pool_indices],
                [p + q for p, q in zip(forward_batch.extend_prefix_lens_cpu, forward_batch.extend_seq_lens_cpu)],
                forward_batch.extend_seq_lens_cpu,
                query_splits,
                pool.page_size,
                pool.reserved_page_num,
                group.rank_in_group,
            )
    return attn_metadata


class NPUMLAMultiStepDecodeBackend:
    """
    Wrap multiple NPU attention backends as one for multiple consecutive
    draft decoding steps
    """

    def __init__(
        self,
        model_runner: ModelRunner,
        topk: int,
        speculative_num_steps: int,
    ):
        self.topk = topk
        self.speculative_num_steps = speculative_num_steps

        self.attn_backends = []
        for _ in range(self.speculative_num_steps):
            self.attn_backends.append(NpuMLAAttnBackend(model_runner))

    def common_template(self, forward_batch: ForwardBatch, call_fn: int):
        assert forward_batch.spec_info is not None

        for i in range(self.speculative_num_steps - 1):
            call_fn(i, forward_batch)

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        def call_fn(i, forward_batch):
            assert forward_batch.spec_info is not None
            self.attn_backends[i].init_forward_metadata(forward_batch)

        self.common_template(forward_batch, call_fn)

    def init_cuda_graph_state(self, max_bs, max_num_tokens):
        for i in range(self.speculative_num_steps):
            self.attn_backends[i].init_cuda_graph_state(max_bs, max_num_tokens)

    def init_forward_metadata_capture_cuda_graph(self, forward_batch: ForwardBatch):
        def call_fn(i, forward_batch):
            self.attn_backends[i].init_forward_metadata_capture_cuda_graph(
                forward_batch.batch_size,
                forward_batch.batch_size * self.topk,
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                encoder_lens=None,
                forward_mode=ForwardMode.DECODE,
                spec_info=forward_batch.spec_info,
            )

        self.common_template(forward_batch, call_fn)

    def init_forward_metadata_replay_cuda_graph(self, forward_batch: ForwardBatch, bs: int):
        def call_fn(i, forward_batch):
            self.attn_backends[i].init_forward_metadata_replay_cuda_graph(
                bs,
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                seq_lens_sum=-1,
                encoder_lens=None,
                forward_mode=ForwardMode.DECODE,
                spec_info=forward_batch.spec_info,
                seq_lens_cpu=None,
            )

        self.common_template(forward_batch, call_fn)


class NpuAttnBackend(AttentionBackend):
    def __init__(self, model_runner: ModelRunner):
        super().__init__()
        self.forward_metadata = None
        self.device = model_runner.device
        self.page_size = model_runner.page_size
        self.attn_mask = ~torch.tril(torch.ones((2048, 2048), dtype=torch.bool, device=self.device))
        # DFlash SWA sink: 仅 draft 且 config 开启时由 DFlashWorker 置真
        self.dflash_swa_enabled: bool = False
        self.dflash_swa_first_kv: int = 0
        self.dflash_swa_window: int = 0

    def init_cuda_graph_state(
        self,
        max_bs: int,
        block_kv_indices: torch.Tensor | None = None,
    ):
        pass

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        """Init the metadata for a forward pass."""
        pass

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        encoder_lens: torch.Tensor | None,
        forward_mode: ForwardMode,
        spec_info: SpecInfo | None,
        forward_batch: ForwardBatch,
    ):
        pass

    def get_cuda_graph_seq_len_fill_value(self):
        return 1

    def forward_extend(
        self,
        q,
        k,
        v,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
    ):
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()

        if save_kv_cache:
            forward_batch.token_to_kv_pool.set_kv_buffer(layer, forward_batch.out_cache_loc, k, v)

        decode_meta = forward_batch.attn_metadata
        assert decode_meta is not None
        assert decode_meta.seq_lens_tensor.dtype is torch.int64
        assert decode_meta.query_len_tensor.dtype is torch.int64
        assert decode_meta.seq_lens_tensor.device.type == "npu"
        assert decode_meta.query_len_tensor.device.type == "npu"

        # View q to TND.
        q = q.view(-1, layer.tp_q_head_num, layer.head_dim)

        # ENCODER_ONLY (DFLASH draft 块内双向): 专用算子, KV 为 ND [numBlocks, page_size, Nkv*D]。
        if getattr(layer, "attn_type", None) == AttentionType.ENCODER_ONLY:
            mlp_k = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)
            mlp_v = forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id)
            mlp_k = mlp_k.view(mlp_k.shape[0], self.page_size, -1)
            mlp_v = mlp_v.view(mlp_v.shape[0], self.page_size, -1)
            window = self.dflash_swa_window if self.dflash_swa_enabled else _INT32_MAX
            sink = self.dflash_swa_first_kv if self.dflash_swa_enabled else 0
            o_mlp, _ = torch.ops.custom.mlp_dflash_infer_attn(
                q,
                mlp_k,
                mlp_v,
                actual_seq_lengths=decode_meta.query_len_tensor,
                actual_seq_lengths_kv=decode_meta.seq_lens_tensor,
                block_table=decode_meta.block_table,
                num_heads=layer.tp_q_head_num,
                scale=layer.scaling,
                pre_tokens=window,
                next_tokens=sink,
                input_layout="TND",
                num_key_value_heads=layer.tp_k_head_num,
                block_size=self.page_size,
                softmax_lse_flag=False,
            )
            return o_mlp.view(-1, layer.tp_q_head_num * layer.head_dim)

        # kv shape: [blockNum, blockSize, N,D]
        k_cache = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)
        v_cache = forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id)
        k_cache = k_cache.view(-1, layer.tp_k_head_num, layer.head_dim // 16, self.page_size, 16)
        v_cache = v_cache.view(-1, layer.tp_k_head_num, layer.head_dim // 16, self.page_size, 16)

        torchair_enable = decode_meta.all_decode_or_idle and ENV.npu_enable_graph and not ENV.npu_disable_graph_mode
        if torchair_enable:
            op_scope = tng.ops
        else:
            op_scope = torch.ops.npu
        o, _ = op_scope.npu_fused_infer_attention_score(
            q,
            k_cache,
            v_cache,
            num_heads=layer.tp_q_head_num,
            num_key_value_heads=layer.tp_k_head_num,
            input_layout="TND",
            atten_mask=self.attn_mask,
            sparse_mode=3,
            scale=layer.scaling,
            block_table=decode_meta.block_table,
            block_size=self.page_size,
            actual_seq_lengths_kv=decode_meta.seq_lens_tensor,
            actual_seq_lengths=decode_meta.query_len_tensor,
        )
        return o.view(-1, layer.tp_q_head_num * layer.head_dim)

    def forward_decode(
        self,
        q,
        k,
        v,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
    ):
        if save_kv_cache:
            forward_batch.token_to_kv_pool.set_kv_buffer(layer, forward_batch.out_cache_loc, k, v)

        decode_meta = forward_batch.attn_metadata
        assert decode_meta is not None
        assert decode_meta.seq_lens_tensor.dtype is torch.int64
        assert decode_meta.query_len_tensor.dtype is torch.int64

        torchair_enable = decode_meta.all_decode_or_idle and ENV.npu_enable_graph and not ENV.npu_disable_graph_mode
        if torchair_enable:
            op_scope = tng.ops
            kwargs = dict(
                actual_seq_lengths_kv=decode_meta.seq_lens_tensor,
                actual_seq_lengths=decode_meta.query_len_tensor,
            )
        else:
            op_scope = torch.ops.npu
            kwargs = dict(
                actual_seq_lengths_kv=decode_meta.seq_lens_tensor,
                actual_seq_lengths=decode_meta.query_len_tensor,
            )

        # kv shape: [blockNum, blockSize, N,D]
        k_cache = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)
        v_cache = forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id)

        # View q k v to TND.
        q = q.view(-1, layer.tp_q_head_num, layer.head_dim).contiguous()
        k_cache = k_cache.view(-1, layer.tp_k_head_num, layer.head_dim // 16, self.page_size, 16)
        v_cache = v_cache.view(-1, layer.tp_k_head_num, layer.head_dim // 16, self.page_size, 16)

        # TODO kv_nz: [blockNum, KV_N, D/16, blockSize, 16]
        # if not global_server_args_dict["npu_disable_kv_nz"]:
        #     k_cache = k_cache.view(k_cache.shape[0], k_cache.shape[1], k_cache.shape[3] // 16, k_cache.shape[2], 16)
        #     v_cache = v_cache.view(v_cache.shape[0], v_cache.shape[1], v_cache.shape[3] // 16, v_cache.shape[2], 16)

        # ENCODER_ONLY (DFLASH draft 块内双向): 不传 atten_mask + sparse_mode=0; 因果路径维持上三角掩码 + sparse_mode=3。
        if getattr(layer, "attn_type", None) == AttentionType.ENCODER_ONLY:
            atten_mask, sparse_mode = None, 0
        else:
            atten_mask, sparse_mode = self.attn_mask, 3
        o, _ = op_scope.npu_fused_infer_attention_score(
            q,
            k_cache,
            v_cache,
            num_heads=layer.tp_q_head_num,
            num_key_value_heads=layer.tp_k_head_num,
            input_layout="TND",
            atten_mask=atten_mask,
            sparse_mode=sparse_mode,
            scale=layer.scaling,
            block_table=forward_batch.attn_metadata.block_table,
            block_size=self.page_size,
            **kwargs,
        )

        return o.view(-1, layer.tp_q_head_num * layer.head_dim)


class NPUMultiStepDecodeBackend:
    """
    Wrap multiple NPU attention backends as one for multiple consecutive
    draft decoding steps
    """

    def __init__(
        self,
        model_runner: ModelRunner,
        topk: int,
        speculative_num_steps: int,
    ):
        self.topk = topk
        self.speculative_num_steps = speculative_num_steps

        self.attn_backends = []
        for _ in range(self.speculative_num_steps):
            self.attn_backends.append(NpuAttnBackend(model_runner))

    def common_template(self, forward_batch: ForwardBatch, call_fn: int):
        assert forward_batch.spec_info is not None

        for i in range(self.speculative_num_steps - 1):
            call_fn(i, forward_batch)

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        def call_fn(i, forward_batch):
            assert forward_batch.spec_info is not None
            self.attn_backends[i].init_forward_metadata(forward_batch)

        self.common_template(forward_batch, call_fn)

    def init_cuda_graph_state(self, max_bs, max_num_tokens):
        for i in range(self.speculative_num_steps):
            self.attn_backends[i].init_cuda_graph_state(max_bs, max_num_tokens)

    def init_forward_metadata_capture_cuda_graph(self, forward_batch: ForwardBatch):
        def call_fn(i, forward_batch):
            self.attn_backends[i].init_forward_metadata_capture_cuda_graph(
                forward_batch.batch_size,
                forward_batch.batch_size * self.topk,
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                encoder_lens=None,
                forward_mode=ForwardMode.DECODE,
                spec_info=forward_batch.spec_info,
            )

        self.common_template(forward_batch, call_fn)

    def init_forward_metadata_replay_cuda_graph(self, forward_batch: ForwardBatch, bs: int):
        def call_fn(i, forward_batch):
            self.attn_backends[i].init_forward_metadata_replay_cuda_graph(
                bs,
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                seq_lens_sum=-1,
                encoder_lens=None,
                forward_mode=ForwardMode.DECODE,
                spec_info=forward_batch.spec_info,
                seq_lens_cpu=None,
            )

        self.common_template(forward_batch, call_fn)
