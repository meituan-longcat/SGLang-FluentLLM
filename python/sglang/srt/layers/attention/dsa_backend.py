from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import torch
from flash_attn_3.flash_attn_interface import flash_attn_with_kvcache

from sglang.srt.configs.model_config import get_nsa_index_topk, is_dsa
from sglang.srt.env import global_server_args_dict
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.dp_attention import get_attention_tp_group, get_attention_tp_rank, get_attention_tp_size
from sglang.srt.layers.attention.dsa.transform_index import (
    transform_index_page_table_decode,
    transform_index_page_table_prefill,
)
from sglang.srt.layers.attention.dsa.utils import (
    NSA_FLASHMLA_BACKEND_DECODE_COMPUTE_FP8,
    NSA_FUSE_TOPK,
    compute_nsa_seqlens,
)
from sglang.srt.layers.utils import CP_METADATA, cp_split_and_rebuild_data, get_cp_metadata_v2, reset_cp_metadata
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode

if TYPE_CHECKING:
    from sglang.srt.layers.attention.radix_attention import RadixAttention
    from sglang.srt.mem_cache.memory_pool.memory_pool_l15_wrapper import TokenToKVPoolL15Wrapper
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.speculative.spec_info import SpecInput

from sglang.srt.utils import get_colorful_logger

logger = get_colorful_logger(__name__)


@dataclass(frozen=True)
class NSAFlashMLAMetadata:
    """Metadata only needed by FlashMLA"""

    flashmla_metadata: torch.Tensor
    num_splits: torch.Tensor

    def slice(self, sli):
        return NSAFlashMLAMetadata(
            flashmla_metadata=self.flashmla_metadata,
            num_splits=self.num_splits[sli],
        )

    def copy_(self, other: NSAFlashMLAMetadata):
        self.flashmla_metadata.copy_(other.flashmla_metadata)
        self.num_splits.copy_(other.num_splits)


@dataclass
class CPL1halfMetadata:
    local_page_table_64: torch.Tensor | None = None
    local_page_table_1: torch.Tensor | None = None
    global_valid_indices: torch.Tensor | None = None
    local_seqlens_k: torch.Tensor | None = None
    swa_mask: torch.Tensor | None = None
    out_cache_loc: torch.Tensor | None = None


@dataclass
class NSAMetadata:
    page_size: int

    # Sequence lengths for the forward batch
    cache_seqlens_int32: torch.Tensor
    # Maximum sequence length for query
    max_seq_len_q: int
    # Maximum sequence length for key
    max_seq_len_k: int
    # Cumulative sequence lengths for query
    cu_seqlens_q: torch.Tensor
    # Cumulative sequence lengths for key
    cu_seqlens_k: torch.Tensor
    # Page table, the index of KV Cache Tables/Blocks
    # this table is always with page_size = 1, use for FA3(kv cache dtype=bf16)
    page_table_1: torch.Tensor

    # NOTE(dark): This will property be used in:
    # 1. dense decode/prefill, we use paged flash attention, need real_page_table
    # 2. sparse decode/prefill, indexer need real_page_table to compute the score
    real_page_table: torch.Tensor

    # NSA metadata (nsa prefill are expanded)
    nsa_cache_seqlens_int32: torch.Tensor  # this seqlens is clipped to `topk`
    nsa_cu_seqlens_q: torch.Tensor  # must be arange(0, len(nsa_cu_seqlens_k))
    nsa_cu_seqlens_k: torch.Tensor  # cumsum of `nsa_cache_seqlens_int32`
    nsa_extend_seq_lens_list: list[int]
    # expanded, unclipped `seqlens`, [s_q] represents the length of attention calculation for each q token
    nsa_seqlens_expanded: torch.Tensor
    row_to_batch: torch.Tensor = None

    cp_l1half_meta: CPL1halfMetadata | None = None
    nsa_max_seqlen_q: Literal[1] = 1  # always 1 for decode, variable for extend

    flashmla_metadata: NSAFlashMLAMetadata | None = None


@dataclass
class NSAIndexerMetadata:
    attn_metadata: NSAMetadata

    def get_seqlens_int32(self) -> torch.Tensor:
        return self.attn_metadata.cache_seqlens_int32

    def get_page_table_64(self) -> torch.Tensor:
        return self.attn_metadata.real_page_table

    def get_seqlens_expanded(self) -> torch.Tensor:
        return self.attn_metadata.nsa_seqlens_expanded

    def topk_transform(
        self,
        logits: torch.Tensor,
        topk: int,
        lengths: torch.Tensor | None = None,
        cu_seqlens_q: torch.Tensor | None = None,
        row_starts: torch.Tensor | None = None,
        num_init_and_local_tokens: list[int] | None = None,
    ) -> torch.Tensor:
        from flashinfer import top_k_page_table_transform, top_k_ragged_transform

        if num_init_and_local_tokens is None:
            num_init_and_local_tokens = [0, 0]

        if lengths is None:
            lengths = self.get_seqlens_expanded()

        if not NSA_FUSE_TOPK:
            offsets = torch.zeros(logits.shape[0], dtype=torch.int32, device=logits.device)
            return top_k_ragged_transform(
                input=logits.contiguous(),
                offsets=offsets,
                lengths=lengths,
                k=topk,
                deterministic=global_server_args_dict["use_deterministic_topk"],
                row_starts=row_starts,
            )

        if cu_seqlens_q is None:
            cu_seqlens_q = self.attn_metadata.cu_seqlens_q
        num_init_tokens, num_local_tokens = num_init_and_local_tokens
        if row_starts is None:
            row_starts = lengths.new_zeros(logits.shape[0])
        if num_init_tokens > 0:
            init_idxs = row_starts[:, None] + torch.arange(num_init_tokens, dtype=lengths.dtype, device=lengths.device)[None, :]
            init_idxs.clamp_max_(logits.shape[-1] - 1)
            logits.scatter_(dim=1, index=init_idxs.expand(logits.shape[0], -1), value=float("inf"))
        if num_local_tokens > 0:
            local_idxs = (row_starts + lengths)[:, None] - 1 - torch.arange(num_local_tokens, dtype=lengths.dtype, device=lengths.device)[None, :]
            local_idxs.clamp_min_(0)
            logits.scatter_(dim=1, index=local_idxs, value=float("inf"))
        # NOTE(dark): if fused, we return a transformed page table directly
        return top_k_page_table_transform(
            input=logits.contiguous(),
            src_page_table=self.attn_metadata.page_table_1,
            lengths=lengths,
            k=topk,
            row_to_batch=self.attn_metadata.row_to_batch,
            deterministic=global_server_args_dict["use_deterministic_topk"],
            row_starts=row_starts,
        )


def topk_transform_torch(
    score: torch.Tensor,
    lengths: torch.Tensor,
    page_table_1: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    row_starts: torch.Tensor,
    topk=2048,
):
    """
    deteministic topk
    """
    bs = score.shape[0]
    seqlens_q = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
    real_bs = seqlens_q.shape[0]
    if row_starts is None:
        row_starts = lengths.new_zeros(bs)
    token2bs = torch.arange(real_bs, dtype=torch.int32, device=score.device).repeat_interleave(seqlens_q)
    indices = torch.full((bs, topk), -1, dtype=torch.int32, device=score.device)
    if lengths.min() >= topk:
        _, idxs = score.topk(topk)
        idxs = idxs - row_starts[:, None]
        return page_table_1[token2bs[:, None], idxs]

    for i in range(score.shape[0]):
        _, idxs = score[i].topk(min(topk, lengths[i]))
        idxs = idxs - row_starts[i]
        ind_i = page_table_1[token2bs[i], idxs]
        indices[i, : len(idxs)] = ind_i
    return indices


def compute_cu_seqlens(seqlens: torch.Tensor) -> torch.Tensor:
    assert seqlens.dtype == torch.int32 and seqlens.is_cuda, f"{seqlens.dtype=} {seqlens.is_cuda=}"
    return torch.nn.functional.pad(torch.cumsum(seqlens, dim=0, dtype=torch.int32), (1, 0))


# _NSA_IMPL_T: TypeAlias = Literal[
#     "flashmla_prefill", "flashmla_decode", "fa3", "tilelang"
# ]
# NSA_PREFILL_IMPL: _NSA_IMPL_T
# NSA_DECODE_IMPL: _NSA_IMPL_T


class DpskSparseAttnBackend(AttentionBackend):
    def __init__(
        self,
        model_runner: ModelRunner,
        speculative_step_id=0,
        speculative_num_steps=0,
    ):
        super().__init__()
        self.forward_metadata: NSAMetadata
        self.is_draft = model_runner.is_draft_worker
        self.device = model_runner.device
        assert isinstance(model_runner.page_size, int)
        self.real_page_size = model_runner.page_size
        self.num_splits = 0  # not model_runner.server_args.enable_deterministic_inference
        self.use_dsa = is_dsa(model_runner.model_config.hf_config)
        assert self.use_dsa, "NSA backend only supports DeepSeek NSA"
        self.kv_cache_store_fp8 = model_runner.kv_cache_dtype == torch.float8_e4m3fn
        self.nsa_index_topk = get_nsa_index_topk(model_runner.model_config.hf_config)

        # TP and CP must be the same for now
        self.tp_rank = self.cp_rank = get_attention_tp_rank()
        self.tp_size = self.cp_size = get_attention_tp_size()
        if global_server_args_dict["enable_mla_l1_5_cache"] or global_server_args_dict["enable_dsa_qcp"]:
            self.num_q_heads = model_runner.model_config.num_attention_heads
        else:
            self.num_q_heads = model_runner.model_config.num_attention_heads // self.tp_size
        self.kv_cache_dim = model_runner.token_to_kv_pool.kv_cache_dim

        assert model_runner.req_to_token_pool is not None
        self.req_to_token = model_runner.req_to_token_pool.req_to_token
        self.max_context_len = self.req_to_token.shape[1]
        self.token_to_kv_pool = model_runner.token_to_kv_pool
        # kv_allocator maintains block_table
        self.kv_allocator = model_runner.kv_allocator
        if self.kv_cache_store_fp8:
            self.nsa_prefill_impl = "flashmla_decode"
            self.nsa_decode_impl = "flashmla_decode"
        else:
            self.nsa_prefill_impl = "fa3"
            self.nsa_decode_impl = "fa3"
        # be careful 16k is enough. max bs maybe greater then it?
        self._arange_buf = torch.arange(16384, device=self.device, dtype=torch.int32)
        # Speculative decoding
        self.speculative_num_draft_tokens = model_runner.server_args.speculative_num_draft_tokens
        self.speculative_num_steps = speculative_num_steps
        self.speculative_step_id = speculative_step_id
        self.num_init_tokens = getattr(model_runner.model_config.hf_config, "index_init_tokens", 0)
        self.num_local_tokens = getattr(model_runner.model_config.hf_config, "index_local_tokens", 0)

    def get_device_int32_arange(self, length: int) -> torch.Tensor:
        if length > len(self._arange_buf):
            next_pow_of_2 = 1 << (length - 1).bit_length()
            self._arange_buf = torch.arange(next_pow_of_2, device=self.device, dtype=torch.int32)
        return self._arange_buf[:length]

    def _transform_table_1_to_real(self, page_table: torch.Tensor) -> torch.Tensor:
        page_size = self.real_page_size
        if page_size == 1:
            return page_table
        max_seqlen_k = page_table.shape[1]
        strided_indices = torch.arange(0, max_seqlen_k, page_size, device=page_table.device, dtype=torch.int32)
        return page_table[:, strided_indices] // page_size

    def _transform_page_table_to_1(self, page_table: torch.Tensor) -> torch.Tensor:
        page_size = self.real_page_size
        # [bs, page_num, 1] + [1, 1, pagesize]
        token_table = (page_table * page_size).unsqueeze(2) + torch.arange(0, page_size, device=page_table.device, dtype=torch.int32)[None, None, :]
        token_table = token_table.flatten(1)
        return token_table

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        """Init the metadata for a forward pass."""
        if global_server_args_dict["enable_dsa_qcp"]:
            # Drop any stale QCP metadata from a previous extend batch; the
            # extend branch below re-sets it for QCP prefill batches.
            reset_cp_metadata()
        batch_size = forward_batch.batch_size
        device = forward_batch.seq_lens.device
        draft_token_num = (
            self.speculative_num_draft_tokens if forward_batch.forward_mode.is_target_verify() or forward_batch.forward_mode.is_draft_extend() else 0
        )

        cache_seqlens_int32 = (forward_batch.seq_lens + draft_token_num).to(torch.int32)
        cu_seqlens_k = compute_cu_seqlens(cache_seqlens_int32)
        if forward_batch.forward_mode.is_decode_or_idle():
            # special treatment due to multi step draft in cuda-graph
            max_seqlen_k = forward_batch.seq_lens.max()
            page_table = forward_batch.req_to_token_pool.req_to_token[forward_batch.req_pool_indices]
        else:
            # in speculative decode, the seq_lens_cpu may not be updated
            forward_batch.seq_lens_cpu = forward_batch.seq_lens.cpu()
            max_seqlen_k = int(forward_batch.seq_lens_cpu.max().item() + draft_token_num)
            page_table = forward_batch.req_to_token_pool.req_to_token[forward_batch.req_pool_indices, :max_seqlen_k]

        if forward_batch.forward_mode.is_decode_or_idle():
            extend_seq_lens_cpu = [1] * batch_size
            max_seqlen_q = 1
            cu_seqlens_q = self.get_device_int32_arange(batch_size + 1)
            seqlens_expanded = cache_seqlens_int32
        elif forward_batch.forward_mode.is_target_verify() or forward_batch.forward_mode.is_draft_extend():
            max_seqlen_q = 1
            cu_seqlens_q = torch.arange(
                0,
                batch_size * self.speculative_num_draft_tokens + 1,
                1,
                dtype=torch.int32,
                device=device,
            )
            extend_seq_lens_cpu = [self.speculative_num_draft_tokens] * batch_size
            forward_batch.extend_seq_lens_cpu = extend_seq_lens_cpu
            seqlens_int32_cpu = [self.speculative_num_draft_tokens + kv_len for kv_len in forward_batch.seq_lens_cpu.tolist()]
            seqlens_expanded = torch.cat(
                [
                    torch.arange(
                        kv_len - qo_len + 1,
                        kv_len + 1,
                        dtype=torch.int32,
                        device=device,
                    )
                    for qo_len, kv_len in zip(
                        extend_seq_lens_cpu,
                        seqlens_int32_cpu,
                    )
                ]
            )
            page_table = torch.repeat_interleave(page_table, repeats=self.speculative_num_draft_tokens, dim=0)
        elif forward_batch.forward_mode.is_extend():
            assert (
                forward_batch.extend_seq_lens_cpu is not None
                and forward_batch.extend_seq_lens is not None
                and forward_batch.extend_prefix_lens_cpu is not None
            ), "All of them must not be None"
            extend_seq_lens_cpu = forward_batch.extend_seq_lens_cpu
            if global_server_args_dict["enable_dsa_qcp"]:
                CP_METADATA.set(get_cp_metadata_v2(self.cp_rank, self.cp_size, extend_seq_lens_cpu))
                max_seqlen_q = None  # not used
                cur_split_list = cu_seqlens_k.new_tensor(CP_METADATA.value.cur_split_list)
                cur_split_list = cur_split_list.reshape(-1, 2).sum(-1, dtype=torch.int32)
                cu_seqlens_q = compute_cu_seqlens(cur_split_list)
            elif any(forward_batch.extend_prefix_lens_cpu):
                max_seqlen_q = max(extend_seq_lens_cpu)
                cu_seqlens_q = compute_cu_seqlens(forward_batch.extend_seq_lens.to(torch.int32))
            else:
                max_seqlen_q = max_seqlen_k
                cu_seqlens_q = cu_seqlens_k
            seqlens_expanded = torch.cat(
                [
                    torch.arange(
                        kv_len - qo_len + 1,
                        kv_len + 1,
                        dtype=torch.int32,
                        device=device,
                    )
                    for qo_len, kv_len in zip(
                        forward_batch.extend_seq_lens_cpu,
                        forward_batch.seq_lens_cpu.tolist(),
                    )
                ]
            )
        else:
            raise AssertionError(f"Unsupported {forward_batch.forward_mode = }")

        # 1D, expanded seqlens (1D means cheap to compute, so always compute it)
        if CP_METADATA:
            seqlens_expanded = cp_split_and_rebuild_data(seqlens_expanded, CP_METADATA.value.split_list, CP_METADATA.value.zigzag_index)
        nsa_cache_seqlens_int32 = compute_nsa_seqlens(
            original_seq_lens=seqlens_expanded,
            nsa_index_topk=self.nsa_index_topk,
        )
        nsa_cu_seqlens_k = compute_cu_seqlens(nsa_cache_seqlens_int32)
        nsa_cu_seqlens_q = self.get_device_int32_arange(len(nsa_cu_seqlens_k))
        flashmla_metadata = (
            self._compute_flashmla_metadata(
                cache_seqlens=nsa_cache_seqlens_int32,
                seq_len_q=1,
            )
            if self.nsa_decode_impl == "flashmla_decode"
            else None
        )
        if CP_METADATA and global_server_args_dict["enable_mla_l1_5_cache"]:
            flashmla_metadata = (
                [
                    self._compute_flashmla_metadata(cache_seqlens=nsa_cache_seqlens_int32[qs:qe], seq_len_q=1)
                    for qs, qe in zip(cu_seqlens_q[:-1], cu_seqlens_q[1:])
                ]
                if self.nsa_decode_impl == "flashmla_decode"
                else [None] * (len(cu_seqlens_q) - 1)
            )
        metadata = NSAMetadata(
            page_size=self.real_page_size,
            cache_seqlens_int32=cache_seqlens_int32,
            max_seq_len_q=max_seqlen_q,
            max_seq_len_k=max_seqlen_k,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            page_table_1=page_table,
            flashmla_metadata=flashmla_metadata,
            nsa_cache_seqlens_int32=nsa_cache_seqlens_int32,
            nsa_cu_seqlens_q=nsa_cu_seqlens_q,
            nsa_cu_seqlens_k=nsa_cu_seqlens_k,
            nsa_seqlens_expanded=seqlens_expanded,
            nsa_extend_seq_lens_list=extend_seq_lens_cpu,
            real_page_table=(
                self.kv_allocator.req_to_page[forward_batch.req_pool_indices]
                if forward_batch.forward_mode.is_decode_or_idle()
                else self._transform_table_1_to_real(page_table)
            ),
            row_to_batch=(
                None
                if forward_batch.forward_mode.is_decode_or_idle()
                else torch.arange(
                    len(cu_seqlens_q) - 1,
                    device=cu_seqlens_q.device,
                    dtype=torch.int32,
                ).repeat_interleave(torch.diff(cu_seqlens_q))
            ),
        )
        if global_server_args_dict["enable_mla_l1_5_cache"]:
            if forward_batch.forward_mode.is_decode_or_idle():
                if TYPE_CHECKING:
                    assert isinstance(self.token_to_kv_pool, TokenToKVPoolL15Wrapper)
                metadata.cp_l1half_meta = self.get_cp_l1half_meta(
                    forward_batch.seq_lens.int(),
                    self.kv_allocator.req_to_page[forward_batch.req_pool_indices],
                    self.req_to_token[forward_batch.req_pool_indices],
                )
            elif forward_batch.forward_mode.is_target_verify() or forward_batch.forward_mode.is_draft_extend():
                draft_token_num = self.speculative_num_draft_tokens
                page_table_64 = torch.repeat_interleave(self.kv_allocator.req_to_page[forward_batch.req_pool_indices], repeats=draft_token_num, dim=0)
                page_table_1 = torch.repeat_interleave(self.req_to_token[forward_batch.req_pool_indices], repeats=draft_token_num, dim=0)
                metadata.cp_l1half_meta = self.get_cp_l1half_meta(seqlens_expanded, page_table_64, page_table_1)

        self.forward_metadata = metadata

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int = 0):
        """Initialize CUDA graph state for the attention backend.

        Args:
            max_bs (int): Max token in batch to support in CUDA graphs

        This creates fixed-size tensors that will be reused during CUDA graph replay
        to avoid memory allocations.
        """
        self.decode_cuda_graph_metadata: dict = {
            "cache_seqlens": torch.zeros(max_bs, dtype=torch.int32, device=self.device),
            "cu_seqlens_q": torch.arange(0, max_bs + 1, dtype=torch.int32, device=self.device),
            "cu_seqlens_k": torch.zeros(max_bs + 1, dtype=torch.int32, device=self.device),
            # fake page_table for sparse_prefill
            "page_table": torch.zeros(
                max_bs,
                self.max_context_len,
                dtype=torch.int32,
                device=self.device,
            ),
            "flashmla_metadata": (
                self._compute_flashmla_metadata(
                    cache_seqlens=torch.ones(max_bs, dtype=torch.int32, device=self.device),
                    seq_len_q=1,  # TODO handle MTP which is not 1
                )
                if self.nsa_decode_impl == "flashmla_decode"
                else None
            ),
            "row_to_batch": torch.zeros(max_bs, dtype=torch.int32, device=self.device),
        }
        if global_server_args_dict["enable_mla_l1_5_cache"]:
            draft_token_num = self.speculative_num_draft_tokens
            page_table_64 = self.kv_allocator.req_to_page
            page_table_1 = self.req_to_token
            if draft_token_num > 1:
                page_table_64 = torch.repeat_interleave(page_table_64, repeats=draft_token_num, dim=0)
                page_table_1 = torch.repeat_interleave(page_table_1, repeats=draft_token_num, dim=0)
            cp_l1half_meta = self.get_cp_l1half_meta(
                torch.ones(page_table_64.shape[0], dtype=torch.int32, device=self.device), page_table_64, page_table_1
            )
            # table: [max_bs, max_local_len or max_page_num]
            self.local_page_table_64 = cp_l1half_meta.local_page_table_64
            self.local_page_table_1 = cp_l1half_meta.local_page_table_1
            self.global_valid_indices = cp_l1half_meta.global_valid_indices
            self.local_seqlens_k = cp_l1half_meta.local_seqlens_k
            self.swa_mask = cp_l1half_meta.swa_mask

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        forward_mode: ForwardMode,
        spec_info: SpecInput | None,
    ):
        """Initialize forward metadata for capturing CUDA graph."""
        if global_server_args_dict["enable_dsa_qcp"]:
            # Graph capture replays model python code; make sure it does not
            # observe QCP metadata left over from a prefill batch.
            reset_cp_metadata()
        assert forward_mode.is_decode_or_idle() or forward_mode.is_target_verify() or forward_mode.is_draft_extend(), f"{forward_mode=}"
        if forward_mode.is_decode_or_idle():
            # Normal Decode
            # Get sequence information
            cache_seqlens_int32 = seq_lens.to(torch.int32)
            cu_seqlens_k = compute_cu_seqlens(cache_seqlens_int32)

            # Use max context length for seq_len_k
            page_table_1 = self.decode_cuda_graph_metadata["page_table"][:bs, :]
            max_seq_len_q = 1

            max_seq_len_k = page_table_1.shape[1]

            # Precompute page table
            # Precompute cumulative sequence lengths

            # NOTE(dark): this is always arange, since we are decoding
            cu_seqlens_q = self.decode_cuda_graph_metadata["cu_seqlens_q"][: bs + 1]
            seqlens_expanded = cache_seqlens_int32
            nsa_cache_seqlens_int32 = compute_nsa_seqlens(cache_seqlens_int32, nsa_index_topk=self.nsa_index_topk)
            nsa_extend_seq_lens_list = [1] * bs

            if self.nsa_decode_impl == "flashmla_decode":
                flashmla_metadata = self.decode_cuda_graph_metadata["flashmla_metadata"].slice(slice(0, bs + 1))
                flashmla_metadata.copy_(
                    self._compute_flashmla_metadata(
                        cache_seqlens=nsa_cache_seqlens_int32,
                        seq_len_q=1,  # TODO handle MTP which is not 1
                    )
                )
            else:
                flashmla_metadata = None

        else:  # forward_mode.is_target_verify() or forward_mode.is_draft_extend()
            cache_seqlens_int32 = (seq_lens + self.speculative_num_draft_tokens).to(torch.int32)
            cu_seqlens_k = compute_cu_seqlens(cache_seqlens_int32)
            max_seq_len_q = 1
            page_table_1 = self.decode_cuda_graph_metadata["page_table"][: bs * self.speculative_num_draft_tokens, :]
            max_seq_len_k = page_table_1.shape[1]

            cu_seqlens_q = torch.arange(
                0,
                bs * self.speculative_num_draft_tokens + 1,
                1,
                dtype=torch.int32,
                device=self.device,
            )

            extend_seq_lens_cpu = [self.speculative_num_draft_tokens] * bs

            seqlens_int32_cpu = [self.speculative_num_draft_tokens + kv_len for kv_len in seq_lens.tolist()]
            seqlens_expanded = torch.cat(
                [
                    torch.arange(
                        kv_len - qo_len + 1,
                        kv_len + 1,
                        dtype=torch.int32,
                        device=self.device,
                    )
                    for qo_len, kv_len in zip(
                        extend_seq_lens_cpu,
                        seqlens_int32_cpu,
                    )
                ]
            )
            nsa_cache_seqlens_int32 = compute_nsa_seqlens(seqlens_expanded, nsa_index_topk=self.nsa_index_topk)
            nsa_extend_seq_lens_list = [1] * bs * self.speculative_num_draft_tokens

            if self.nsa_decode_impl == "flashmla_decode":
                flashmla_metadata = self.decode_cuda_graph_metadata["flashmla_metadata"].slice(slice(0, bs * self.speculative_num_draft_tokens + 1))

                flashmla_metadata.copy_(
                    self._compute_flashmla_metadata(
                        cache_seqlens=nsa_cache_seqlens_int32,
                        seq_len_q=1,
                    )
                )
            else:
                flashmla_metadata = None

        nsa_cu_seqlens_k = compute_cu_seqlens(nsa_cache_seqlens_int32)
        nsa_cu_seqlens_q = self.get_device_int32_arange(len(nsa_cu_seqlens_k))
        real_page_table = self._transform_table_1_to_real(page_table_1)
        row_to_batch = self.decode_cuda_graph_metadata["row_to_batch"][: cu_seqlens_q[-1]]
        row_to_batch.copy_(
            torch.arange(
                len(cu_seqlens_q) - 1,
                device=cu_seqlens_q.device,
                dtype=torch.int32,
            ).repeat_interleave(torch.diff(cu_seqlens_q))
        )
        metadata = NSAMetadata(
            page_size=self.real_page_size,
            cache_seqlens_int32=cache_seqlens_int32,
            max_seq_len_q=max_seq_len_q,
            max_seq_len_k=max_seq_len_k,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            page_table_1=page_table_1,
            flashmla_metadata=flashmla_metadata,
            nsa_cache_seqlens_int32=nsa_cache_seqlens_int32,
            nsa_cu_seqlens_q=nsa_cu_seqlens_q,
            nsa_cu_seqlens_k=nsa_cu_seqlens_k,
            nsa_seqlens_expanded=seqlens_expanded,
            real_page_table=real_page_table,
            nsa_extend_seq_lens_list=nsa_extend_seq_lens_list,
            row_to_batch=row_to_batch,
        )
        if global_server_args_dict["enable_mla_l1_5_cache"]:
            if forward_mode.is_decode_or_idle():
                if TYPE_CHECKING:
                    assert isinstance(self.token_to_kv_pool, TokenToKVPoolL15Wrapper)
                cp_l1half_meta = self.get_cp_l1half_meta(
                    seq_lens.to(torch.int32), self.kv_allocator.req_to_page[req_pool_indices], self.req_to_token[req_pool_indices]
                )
                metadata.cp_l1half_meta = self.copy_l1half_meta_to_self_and_ret_meta(cp_l1half_meta, bs)
            elif forward_mode.is_target_verify() or forward_mode.is_draft_extend():
                draft_token_num = self.speculative_num_draft_tokens
                page_table_64 = torch.repeat_interleave(self.kv_allocator.req_to_page[req_pool_indices], repeats=draft_token_num, dim=0)
                page_table_1 = torch.repeat_interleave(self.req_to_token[req_pool_indices], repeats=draft_token_num, dim=0)
                cp_l1half_meta = self.get_cp_l1half_meta(seqlens_expanded, page_table_64, page_table_1)
                metadata.cp_l1half_meta = self.copy_l1half_meta_to_self_and_ret_meta(cp_l1half_meta, bs * draft_token_num)

        self.decode_cuda_graph_metadata[bs] = metadata
        self.forward_metadata = metadata

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        forward_mode: ForwardMode,
        spec_info: SpecInput | None,
        seq_lens_cpu: torch.Tensor | None,
        out_cache_loc: torch.Tensor | None = None,
    ):
        """Initialize forward metadata for replaying CUDA graph."""
        if global_server_args_dict["enable_dsa_qcp"]:
            reset_cp_metadata()
        assert seq_lens_cpu is not None
        assert forward_mode.is_decode_or_idle() or forward_mode.is_target_verify() or forward_mode.is_draft_extend(), f"{forward_mode=}"
        seq_lens = seq_lens[:bs]
        req_pool_indices = req_pool_indices[:bs]

        # Normal Decode
        metadata: NSAMetadata = self.decode_cuda_graph_metadata[bs]
        if forward_mode.is_decode_or_idle():
            seq_lens_cpu = seq_lens.cpu()
            max_len = int(seq_lens_cpu.max().item())

            cache_seqlens = seq_lens.to(torch.int32)
            metadata.cache_seqlens_int32.copy_(cache_seqlens)
            metadata.cu_seqlens_k[1:].copy_(torch.cumsum(cache_seqlens, dim=0, dtype=torch.int32))
            page_indices = self.req_to_token[req_pool_indices, :max_len]
            metadata.page_table_1[:, :max_len].copy_(page_indices)
            metadata.nsa_seqlens_expanded.copy_(cache_seqlens)
            seqlens_expanded = cache_seqlens
            nsa_cache_seqlens = compute_nsa_seqlens(cache_seqlens, nsa_index_topk=self.nsa_index_topk)
            metadata.nsa_cache_seqlens_int32.copy_(nsa_cache_seqlens)
        elif forward_mode.is_target_verify():
            cache_seqlens = (seq_lens + self.speculative_num_draft_tokens).to(torch.int32)
            metadata.cache_seqlens_int32.copy_(cache_seqlens)
            metadata.cu_seqlens_k[1:].copy_(torch.cumsum(cache_seqlens, dim=0, dtype=torch.int32))
            page_indices = self.req_to_token[req_pool_indices]
            page_indices = torch.repeat_interleave(page_indices, repeats=self.speculative_num_draft_tokens, dim=0)
            metadata.page_table_1.copy_(page_indices)

            seqlens_expanded = (
                seq_lens[:, None] + 1 + torch.arange(0, self.speculative_num_draft_tokens, dtype=torch.int32, device=self.device)[None, :]
            ).reshape(-1)
            metadata.nsa_seqlens_expanded.copy_(seqlens_expanded)
            nsa_cache_seqlens = compute_nsa_seqlens(seqlens_expanded, self.nsa_index_topk)
            metadata.nsa_cache_seqlens_int32.copy_(nsa_cache_seqlens)
        elif forward_mode.is_draft_extend():
            max_seqlen_k = int(seq_lens_cpu.max().item())
            cache_seqlens = seq_lens.to(torch.int32)
            metadata.cache_seqlens_int32.copy_(cache_seqlens)
            metadata.cu_seqlens_k[1:].copy_(torch.cumsum(cache_seqlens, dim=0, dtype=torch.int32))

            extend_seq_lens = spec_info.accept_length[:bs]
            extend_seq_lens_cpu = extend_seq_lens.tolist()

            page_indices = self.req_to_token[req_pool_indices, :max_seqlen_k]
            page_indices = torch.repeat_interleave(page_indices, repeats=extend_seq_lens, dim=0)
            metadata.page_table_1[: page_indices.shape[0], :max_seqlen_k].copy_(page_indices)

            seqlens_expanded = torch.cat(
                [
                    torch.arange(
                        kv_len - qo_len + 1,
                        kv_len + 1,
                        dtype=torch.int32,
                        device=self.device,
                    )
                    for qo_len, kv_len in zip(
                        extend_seq_lens_cpu,
                        seq_lens_cpu.tolist(),
                    )
                ]
            )
            metadata.nsa_seqlens_expanded[: seqlens_expanded.shape[0]].copy_(seqlens_expanded)
            nsa_cache_seqlens = compute_nsa_seqlens(seqlens_expanded, self.nsa_index_topk)
            metadata.nsa_cache_seqlens_int32[: seqlens_expanded.shape[0]].copy_(nsa_cache_seqlens)
        seqlens_expanded_size = seqlens_expanded.shape[0]
        assert metadata.nsa_cache_seqlens_int32 is not None and metadata.nsa_cu_seqlens_k is not None and self.nsa_index_topk is not None
        metadata.nsa_cu_seqlens_k[1 : 1 + seqlens_expanded_size].copy_(torch.cumsum(nsa_cache_seqlens, dim=0, dtype=torch.int32))
        # NOTE(dark): (nsa-) cu_seqlens_q is always arange, no need to copy
        assert self.real_page_size == metadata.page_size
        if self.real_page_size > 1:
            real_table = self._transform_table_1_to_real(page_indices)
            new_rows = real_table.shape[0]
            new_cols = real_table.shape[1]
            metadata.real_page_table[:new_rows, :new_cols].copy_(real_table)
        else:
            assert metadata.real_page_table is metadata.page_table_1

        if self.nsa_decode_impl == "flashmla_decode":
            flashmla_metadata = metadata.flashmla_metadata.slice(slice(0, seqlens_expanded_size + 1))
            flashmla_metadata.copy_(
                self._compute_flashmla_metadata(
                    cache_seqlens=nsa_cache_seqlens,
                    seq_len_q=1,
                )
            )
        if global_server_args_dict["enable_mla_l1_5_cache"]:
            if forward_mode.is_decode_or_idle():
                if TYPE_CHECKING:
                    assert isinstance(self.token_to_kv_pool, TokenToKVPoolL15Wrapper)
                cp_l1half_meta = self.get_cp_l1half_meta(
                    seq_lens.to(torch.int32), self.kv_allocator.req_to_page[req_pool_indices], self.req_to_token[req_pool_indices]
                )
                metadata.cp_l1half_meta = self.copy_l1half_meta_to_self_and_ret_meta(cp_l1half_meta, bs)
            elif forward_mode.is_target_verify() or forward_mode.is_draft_extend():
                draft_token_num = self.speculative_num_draft_tokens
                page_table_64 = torch.repeat_interleave(self.kv_allocator.req_to_page[req_pool_indices], repeats=draft_token_num, dim=0)
                page_table_1 = torch.repeat_interleave(self.req_to_token[req_pool_indices], repeats=draft_token_num, dim=0)
                cp_l1half_meta = self.get_cp_l1half_meta(seqlens_expanded, page_table_64, page_table_1)
                metadata.cp_l1half_meta = self.copy_l1half_meta_to_self_and_ret_meta(cp_l1half_meta, bs * draft_token_num)
        self.forward_metadata = metadata

    def get_cp_l1half_meta(
        self,
        seq_lens_32: torch.Tensor,
        global_page_table,
        global_req_to_token,
    ):
        bs = seq_lens_32.shape[0]
        device = seq_lens_32.device
        local_page_table_64, _ = self.token_to_kv_pool.remap_page_table_to_local(global_page_table, seq_lens_32)
        local_page_table_1, global_valid_indices, local_seqlens_k, local_mask, safe_pos = self.token_to_kv_pool.remap_token_idxs_to_local_return_mask(
            global_req_to_token, seq_lens_32
        )
        swa_mask = None
        if self.num_init_tokens > 0:
            max_seq_len = global_page_table.shape[1] * self.real_page_size
            pos = torch.arange(max_seq_len, device=device)[None, :]
            swa_global_mask = (pos < seq_lens_32[:, None]) & ((pos < self.num_init_tokens) | (pos >= seq_lens_32[:, None] - self.num_local_tokens))
            swa_global_mask = swa_global_mask.bool()
            max_shape_l = local_page_table_64.shape[1] * self.real_page_size
            swa_mask = torch.zeros(bs, max_shape_l + 1, dtype=torch.bool, device=device)
            swa_mask.scatter_(dim=1, index=safe_pos, src=torch.where(local_mask, swa_global_mask, False))
            swa_mask = swa_mask[:, :max_shape_l]

        cp_l1half_meta = CPL1halfMetadata(local_page_table_64, local_page_table_1, global_valid_indices, local_seqlens_k, swa_mask)
        return cp_l1half_meta

    def copy_l1half_meta_to_self_and_ret_meta(
        self,
        cp_l1half_meta: CPL1halfMetadata,
        num_valid: int,
    ):
        self.local_seqlens_k[:num_valid].copy_(cp_l1half_meta.local_seqlens_k)
        self.local_page_table_64[:num_valid].copy_(cp_l1half_meta.local_page_table_64)
        self.local_page_table_1[:num_valid].copy_(cp_l1half_meta.local_page_table_1)
        self.global_valid_indices[:num_valid].copy_(cp_l1half_meta.global_valid_indices)
        if self.swa_mask is not None:
            self.swa_mask[:num_valid].copy_(cp_l1half_meta.swa_mask)
        return CPL1halfMetadata(
            self.local_page_table_64[:num_valid],
            self.local_page_table_1[:num_valid],
            self.global_valid_indices[:num_valid],
            self.local_seqlens_k[:num_valid],
            None if self.swa_mask is None else self.swa_mask[:num_valid],
        )

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
        # For multi-head latent attention
        q_pe: torch.Tensor | None = None,
        k_pe: torch.Tensor | None = None,
        topk_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if k is not None:
            assert v is not None
            if save_kv_cache:
                cache_loc = forward_batch.out_cache_loc
                forward_batch.token_to_kv_pool.set_mla_kv_buffer(  # type: ignore
                    layer,
                    cache_loc,
                    k,
                    k_pe,
                )

        metadata = self.forward_metadata
        causal = not layer.is_cross_attention
        assert causal, "DSA is causal only"

        # Do absorbed multi-latent attention
        assert q_pe is not None
        kv_cache = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)

        if q_pe is not None:
            q_nope = q.view(-1, self.num_q_heads, layer.v_head_dim)
            q_pe = q_pe.view(-1, self.num_q_heads, layer.head_dim - layer.v_head_dim)
        else:
            q_all = q.contiguous().view(-1, self.num_q_heads, layer.head_dim)
            q_nope = q_all[:, :, : layer.v_head_dim]
            q_pe = q_all[:, :, layer.v_head_dim :]

        # NOTE(dark): here, we use page size = 1
        if NSA_FUSE_TOPK:
            page_table_1 = topk_indices
        else:
            assert metadata.nsa_extend_seq_lens_list is not None
            page_table_1 = transform_index_page_table_prefill(
                page_table=metadata.page_table_1,
                topk_indices=topk_indices,
                extend_lens_cpu=metadata.nsa_extend_seq_lens_list,
                page_size=1,
            )
        # tilelang not support for now
        # if self.nsa_prefill_impl == "tilelang":
        #     if q_rope is not None:
        #         q_all = torch.cat([q_nope, q_rope], dim=-1)
        #     return self._forward_tilelang(
        #         q_all=q_all,
        #         kv_cache=kv_cache,
        #         page_table_1=page_table_1,
        #         sm_scale=layer.scaling,
        #         v_head_dim=layer.v_head_dim,
        #     )
        if self.nsa_prefill_impl == "flashmla_prefill":
            if q_pe is not None:
                q_all = torch.cat([q_nope, q_pe], dim=-1)
            return self._forward_flashmla_prefill(
                q_all=q_all,
                kv_cache=kv_cache,
                page_table_1=page_table_1,
                sm_scale=layer.scaling,
                v_head_dim=layer.v_head_dim,
            )
        elif self.nsa_prefill_impl == "flashmla_decode":
            if q_pe is not None:
                q_all = torch.cat([q_nope, q_pe], dim=-1)

            # Core attention op with a kernel-agnostic signature so the
            # context-parallel l1.5-cache strategies can be reused.
            # Returns o when return_lse is False, else (o, lse).
            def flashmla_decode_core(q_all, kv, page_table, cache_seqlens, return_lse, flashmla_metadata=None):
                if q_all.shape[0] == 0:
                    assert not return_lse
                    return q_all.new_zeros((0, 1, self.num_q_heads, layer.v_head_dim))
                ret = self._forward_flashmla_decode(
                    q_all=q_all,
                    kv_cache=kv,
                    sm_scale=layer.scaling,
                    v_head_dim=layer.v_head_dim,
                    # TODO optimize args
                    layer=layer,
                    metadata=metadata,
                    page_table_1=page_table,
                    cache_seqlens=cache_seqlens,
                    return_softmax_lse=return_lse,
                    flashmla_metadata=flashmla_metadata,
                )
                if return_lse:
                    o, lse = ret
                    # [bs*sq, 1, num_q_heads, dim] -> [bs*sq, num_q_heads, dim]
                    o.squeeze_(1)
                    # flashmla lse: (batch, num_q_heads, seq_len_q) ->
                    # (q_len, num_q_heads) to match the merge contract.
                    lse = lse.permute(0, 2, 1).reshape(-1, self.num_q_heads)
                    return o, lse
                return ret

            if global_server_args_dict["enable_mla_l1_5_cache"]:
                return self._forward_l1_5_cache(
                    flashmla_decode_core,
                    forward_batch,
                    q_all=q_all,
                    kv_cache=kv_cache,
                    page_table_1=page_table_1,
                    cache_seqlens=metadata.nsa_cache_seqlens_int32,
                    layer_id=layer.layer_id,
                )
            else:
                o = flashmla_decode_core(
                    q_all,
                    kv_cache,
                    page_table_1,
                    metadata.nsa_cache_seqlens_int32,
                    return_lse=False,
                )
                return o
        elif self.nsa_prefill_impl == "fa3":
            if q_pe is not None:
                q_all = torch.cat([q_nope, q_pe], dim=-1)

            def fa3_core(q_all, kv, page_table, cache_seqlens, return_lse, flashmla_metadata=None):
                if q_all.shape[0] == 0:
                    assert not return_lse
                    return q_all.new_zeros((0, self.num_q_heads, layer.v_head_dim))
                q_nope = q_all[:, :, : layer.v_head_dim]
                q_pe = q_all[:, :, layer.v_head_dim :]
                ret = self._forward_fa3(
                    q_rope=q_pe,
                    kv_cache=kv,
                    v_head_dim=layer.v_head_dim,
                    q_nope=q_nope,
                    page_table=page_table,
                    cache_seqlens=cache_seqlens,
                    cu_seqlens_q=metadata.nsa_cu_seqlens_q,
                    cu_seqlens_k=metadata.nsa_cu_seqlens_k,
                    max_seqlen_q=metadata.nsa_max_seqlen_q,
                    sm_scale=layer.scaling,
                    logit_cap=layer.logit_cap,
                    page_size=1,
                    return_softmax_lse=return_lse,
                )
                if return_lse:
                    o, lse = ret[0], ret[1]
                    # fa3 lse: (num_q_heads, q_len) -> (q_len, num_q_heads)
                    # to match the merge contract.
                    lse = torch.transpose(lse, 0, 1).contiguous()
                    return o, lse
                return ret

            if global_server_args_dict["enable_mla_l1_5_cache"]:
                return self._forward_l1_5_cache(
                    fa3_core,
                    forward_batch,
                    q_all,
                    kv_cache=kv_cache,
                    page_table_1=page_table_1,
                    cache_seqlens=metadata.nsa_cache_seqlens_int32,
                )
            else:
                return fa3_core(
                    q_all,
                    kv_cache,
                    page_table_1,
                    metadata.nsa_cache_seqlens_int32,
                    return_lse=False,
                )
        else:
            raise ValueError(f"Unsupported {self.nsa_prefill_impl = }")

    def _cp_allgather_kv(self, raw_kv_buffer: torch.Tensor, seq_i: int) -> torch.Tensor:
        """All-gather the per-rank KV shards into the full KV on every rank."""
        # assume that only have one req
        local_indices, per_rank_count, inv_perm = self.forward_metadata.cp_gather_k_meta[seq_i]
        local_k = raw_kv_buffer.view(-1, self.kv_cache_dim).index_select(0, local_indices)
        gathered_tensor = local_k.new_empty(self.forward_metadata.seqlens_cpu[seq_i], local_k.shape[-1])
        tensor_list = list(gathered_tensor.split(per_rank_count))
        torch.distributed.all_gather(tensor_list, local_k, group=get_attention_tp_group().device_group)
        all_k = gathered_tensor[inv_perm]
        return all_k.view(-1, 1, self.kv_cache_dim)

    def _forward_l1_5_cache(
        self,
        core,
        forward_batch: ForwardBatch,
        q_all: torch.Tensor,
        kv_cache: torch.Tensor,
        page_table_1: torch.Tensor,
        cache_seqlens: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Context-parallel wrapper for the l1.5 cache, kernel-agnostic.

        `core(kv, page_table, cache_seqlens, return_lse)` is the attention op:
        it returns o when return_lse is False, else (o, lse).
        """
        # TODO use len and other conditions to determine use `AG kv` or `comm lse`
        if global_server_args_dict["enable_dsa_qcp"]:
            assert self.nsa_prefill_impl == "flashmla_decode", "only work for flashmla_decode, not work for FA3 for now!"
            # Strategy A: all-gather KV to every rank, then full local attention.
            # Uses the raw key buffer (not dtype-cast) to match the original path.
            olist = []
            for i in range(forward_batch.batch_size):
                qs, qe = self.forward_metadata.cu_seqlens_q_cpu[i], self.forward_metadata.cu_seqlens_q_cpu[i + 1]
                all_k = self._cp_allgather_kv(kv_cache, i)
                oi = core(
                    q_all[qs:qe],
                    all_k,
                    page_table_1[qs:qe],
                    cache_seqlens[qs:qe],
                    return_lse=False,
                    flashmla_metadata=self.forward_metadata.flashmla_metadata[i],
                )
                olist.append(oi)
            return torch.cat(olist)
        else:
            # Strategy B: local-shard partial attention merged via lse.
            local_indices, local_lens = indices_trans_local(
                page_table_1,
                cache_seqlens,
                get_attention_tp_rank(),
                get_attention_tp_size(),
                self.real_page_size,
                self.token_to_kv_pool.reserved_page_num,
            )
            # flashmla-only metadata; other cores (e.g. fa3) ignore it.
            cur_metadata = None
            if self.nsa_prefill_impl == "flashmla_decode":
                raise NotImplementedError()
                from flash_mla import get_mla_metadata

                flashmla_metadata, num_splits = get_mla_metadata(
                    cache_seqlens=local_lens,
                    num_q_tokens_per_head_k=1 * self.num_q_heads // 1,
                    num_heads_k=1,
                    num_heads_q=self.num_q_heads,
                    is_fp8_kvcache=NSA_FLASHMLA_BACKEND_DECODE_COMPUTE_FP8,
                    topk=self.nsa_index_topk,
                )
                cur_metadata = NSAFlashMLAMetadata(
                    flashmla_metadata=flashmla_metadata,
                    num_splits=num_splits,
                )

            o, lse = core(
                q_all,
                kv_cache,
                local_indices,
                local_lens,
                return_lse=True,
                flashmla_metadata=cur_metadata,
            )
            merge_o = self.merge_attn_o_and_lse(o, lse)
            return merge_o

    def merge_attn_o_and_lse(
        self,
        o: torch.Tensor,  # (s_q, q_h, d)
        lse: torch.Tensor,  # (s_q, q_h)
    ):
        attn_tp_group = get_attention_tp_group()
        q_len, num_q_heads = lse.shape
        gathered_lse = torch.zeros(
            (self.tp_size, q_len, num_q_heads),
            device=lse.device,
            dtype=lse.dtype,
        )
        torch.distributed.all_gather_into_tensor(gathered_lse, lse, group=attn_tp_group.device_group)

        # Global logsumexp over TP ranks for mathematically correct attention merge.
        # local lse = log(sum_j exp(score_j)_local_rank)
        # global lse = log(sum_r exp(local_lse_r))
        new_lse = torch.logsumexp(gathered_lse, dim=0)

        # use full_seq_lse to correct attn_output
        lse_correction = torch.exp(lse - new_lse)
        o *= lse_correction.unsqueeze(-1)
        attn_output = o

        # Reduce-scatter must shard by head dimension, not by flattened
        # token-head order. Otherwise q_len > 1 can mix tokens across shards.
        # d = attn_output.shape[-1]
        # attn_output = attn_output.transpose(0, 1).contiguous()  # (num_q_heads, q_len, h)
        # attn_output = attn_output.reshape(num_q_heads, q_len * d)
        # attn_output = attn_tp_group.reduce_scatter(attn_output)
        # attn_output = attn_output.reshape(num_q_heads // self.tp_size, q_len, d).transpose(0, 1).contiguous()
        torch.distributed.all_reduce(attn_output, group=attn_tp_group.device_group)
        local_heads = num_q_heads // self.tp_size
        attn_output = attn_output[:, self.cp_rank * local_heads : (self.cp_rank + 1) * local_heads]
        return attn_output

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
        # For multi-head latent attention
        q_pe: torch.Tensor | None = None,
        k_pe: torch.Tensor | None = None,
        topk_indices: torch.Tensor | None = None,  # (batch_size, seq_len_q, topk)
    ) -> torch.Tensor:
        if k is not None:
            assert v is not None
            if save_kv_cache:
                cache_loc = forward_batch.out_cache_loc
                forward_batch.token_to_kv_pool.set_mla_kv_buffer(  # type: ignore
                    layer,
                    cache_loc,
                    k,
                    k_pe,
                )

        metadata = self.forward_metadata
        causal = not layer.is_cross_attention
        assert causal, "DSA is causal only"

        # Do absorbed multi-latent attention
        kv_cache = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)
        if q_pe is not None:
            q_nope = q.view(-1, self.num_q_heads, layer.v_head_dim)
            q_pe = q_pe.view(-1, self.num_q_heads, layer.head_dim - layer.v_head_dim)
        else:
            q_all = q.contiguous().view(-1, self.num_q_heads, layer.head_dim)
            q_nope = q_all[:, :, : layer.v_head_dim]
            q_pe = q_all[:, :, layer.v_head_dim :]

        if NSA_FUSE_TOPK:
            page_table_1 = topk_indices
        else:
            page_table_1 = transform_index_page_table_decode(
                page_table=metadata.page_table_1,
                topk_indices=topk_indices,
                page_size=1,
            )
        if global_server_args_dict["enable_mla_l1_5_cache"]:
            local_indices, local_lens = indices_trans_local(
                page_table_1,
                metadata.nsa_cache_seqlens_int32,
                get_attention_tp_rank(),
                get_attention_tp_size(),
                self.real_page_size,
                self.token_to_kv_pool.reserved_page_num,
            )

        if self.nsa_decode_impl == "flashmla_prefill":
            if q_pe is not None:
                q_all = torch.cat([q_nope, q_pe], dim=-1)
            return self._forward_flashmla_prefill(
                q_all=q_all,
                kv_cache=kv_cache,
                page_table_1=page_table_1,
                sm_scale=layer.scaling,
                v_head_dim=layer.v_head_dim,
            )
        elif self.nsa_decode_impl == "flashmla_decode":
            if q_pe is not None:
                q_all = torch.cat([q_nope, q_pe], dim=-1)
            if global_server_args_dict["enable_mla_l1_5_cache"]:
                raise NotImplementedError("kv fp8 and enable_mla_l1_5_cache have precision issue")
            else:
                o = self._forward_flashmla_decode(
                    q_all=q_all,
                    kv_cache=kv_cache,
                    sm_scale=layer.scaling,
                    v_head_dim=layer.v_head_dim,
                    # TODO optimize args
                    layer=layer,
                    metadata=metadata,
                    page_table_1=page_table_1,
                )
                return o
        elif self.nsa_decode_impl == "fa3":
            if global_server_args_dict["enable_mla_l1_5_cache"]:
                fa_ret = self._forward_fa3(
                    q_rope=q_pe,
                    kv_cache=kv_cache,
                    v_head_dim=layer.v_head_dim,
                    q_nope=q_nope,
                    page_table=local_indices,
                    cache_seqlens=local_lens,
                    cu_seqlens_q=metadata.nsa_cu_seqlens_q,
                    cu_seqlens_k=metadata.nsa_cu_seqlens_k,
                    max_seqlen_q=metadata.nsa_max_seqlen_q,
                    sm_scale=layer.scaling,
                    logit_cap=layer.logit_cap,
                    page_size=1,
                    return_softmax_lse=True,
                )
                o, lse = fa_ret[0], fa_ret[1]
                lse = torch.transpose(lse, 0, 1).contiguous()
                o = self.merge_attn_o_and_lse(o, lse)
                return o
            else:
                return self._forward_fa3(
                    q_rope=q_pe,
                    kv_cache=kv_cache,
                    v_head_dim=layer.v_head_dim,
                    q_nope=q_nope,
                    page_table=page_table_1,
                    cache_seqlens=metadata.nsa_cache_seqlens_int32,
                    cu_seqlens_q=metadata.nsa_cu_seqlens_q,
                    cu_seqlens_k=metadata.nsa_cu_seqlens_k,
                    max_seqlen_q=metadata.nsa_max_seqlen_q,
                    sm_scale=layer.scaling,
                    logit_cap=layer.logit_cap,
                    page_size=1,
                )
        else:
            raise AssertionError(f"Unsupported {self.nsa_decode_impl = }")

    def _forward_fa3(
        self,
        q_rope: torch.Tensor,
        kv_cache: torch.Tensor,
        v_head_dim: int,
        q_nope: torch.Tensor,
        page_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int,
        sm_scale: float,
        logit_cap: float,
        page_size: int,
        return_softmax_lse: bool = False,
    ):
        k_rope_cache = kv_cache[..., v_head_dim:]
        c_kv_cache = kv_cache[..., :v_head_dim]
        qk_rope_dim = k_rope_cache.shape[-1]
        k_rope_cache = k_rope_cache.view(-1, page_size, 1, qk_rope_dim)
        c_kv_cache = c_kv_cache.view(-1, page_size, 1, v_head_dim)
        res = flash_attn_with_kvcache(
            q=q_rope,
            k_cache=k_rope_cache,
            v_cache=c_kv_cache,
            qv=q_nope,
            page_table=page_table,
            cache_seqlens=cache_seqlens,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k_new=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            softmax_scale=sm_scale,
            causal=True,
            softcap=logit_cap,
            return_softmax_lse=return_softmax_lse,
            num_splits=self.num_splits,
        )
        return res  # type: ignore

    def _forward_flashmla_prefill(
        self,
        q_all: torch.Tensor,
        kv_cache: torch.Tensor,
        v_head_dim: int,
        page_table_1: torch.Tensor,
        sm_scale: float,
    ) -> torch.Tensor:
        from flash_mla import flash_mla_sparse_fwd

        o, _, _ = flash_mla_sparse_fwd(
            q=q_all,
            kv=kv_cache,
            indices=page_table_1.unsqueeze(1),
            sm_scale=sm_scale,
            d_v=v_head_dim,
        )
        return o

    def _forward_flashmla_decode(
        self,
        q_all: torch.Tensor,
        kv_cache: torch.Tensor,
        v_head_dim: int,
        sm_scale: float,
        layer,
        metadata: NSAMetadata,
        page_table_1,
        cache_seqlens: torch.Tensor | None = None,
        return_softmax_lse: bool = False,
        flashmla_metadata=None,
    ) -> torch.Tensor:
        from flash_mla import flash_mla_with_kvcache

        assert self.real_page_size == 64, "only page size 64 is supported"

        if cache_seqlens is None:
            cache_seqlens = metadata.nsa_cache_seqlens_int32

        # s_q is always 1, each q token attent to diffent kv
        q_all = q_all.view(-1, 1, self.num_q_heads, layer.head_dim)

        # if allgather kv_cache
        # the cache does not fill a whole page, pad up to a full page before view,
        page_numel = self.real_page_size * self.kv_cache_dim
        flat_kv_cache = kv_cache.reshape(-1)
        remainder = flat_kv_cache.numel() % page_numel
        if remainder != 0:
            pad_numel = page_numel - remainder
            flat_kv_cache = torch.nn.functional.pad(flat_kv_cache, (0, pad_numel))
        kv_cache = flat_kv_cache.view(-1, self.real_page_size, 1, self.kv_cache_dim)

        if NSA_FLASHMLA_BACKEND_DECODE_COMPUTE_FP8 and not self.kv_cache_store_fp8:
            # inefficiently quantize the whole cache
            # kv_cache = quantize_k_cache(kv_cache)
            raise ValueError(f"{NSA_FLASHMLA_BACKEND_DECODE_COMPUTE_FP8=} {self.kv_cache_store_fp8=}")

        indices = page_table_1.unsqueeze(1)
        assert indices.shape[-1] == self.nsa_index_topk  # requirement of FlashMLA decode kernel
        if flashmla_metadata is None:
            flashmla_metadata = metadata.flashmla_metadata
        o, lse = flash_mla_with_kvcache(
            q=q_all,
            k_cache=kv_cache,
            cache_seqlens=cache_seqlens,
            head_dim_v=v_head_dim,
            tile_scheduler_metadata=flashmla_metadata.flashmla_metadata,
            num_splits=flashmla_metadata.num_splits,
            softmax_scale=sm_scale,
            indices=indices,
            # doc says it is not used, but if pass in None then error
            block_table=torch.empty((q_all.shape[0], 0), dtype=torch.int32, device=q_all.device),
            is_fp8_kvcache=NSA_FLASHMLA_BACKEND_DECODE_COMPUTE_FP8,
        )
        if return_softmax_lse:
            return o, lse
        return o

    def get_cuda_graph_seq_len_fill_value(self):
        """Get the fill value for sequence length in CUDA graph."""
        return 1

    def get_indexer_metadata(self, layer_id: int, forward_batch: ForwardBatch) -> NSAIndexerMetadata:
        return NSAIndexerMetadata(attn_metadata=self.forward_metadata)

    def _compute_flashmla_metadata(self, cache_seqlens: torch.Tensor, seq_len_q: int):
        from flash_mla import get_mla_metadata

        flashmla_metadata, num_splits = get_mla_metadata(
            cache_seqlens=cache_seqlens,
            # TODO doc says `num_q_tokens_per_q_seq * num_heads_q // num_heads_k`
            #      but the name looks like need seq_len_q?
            num_q_tokens_per_head_k=seq_len_q * self.num_q_heads // 1,
            num_heads_k=1,
            num_heads_q=self.num_q_heads,
            is_fp8_kvcache=NSA_FLASHMLA_BACKEND_DECODE_COMPUTE_FP8,
            topk=self.nsa_index_topk,
        )

        return NSAFlashMLAMetadata(
            flashmla_metadata=flashmla_metadata,
            num_splits=num_splits,
        )

    def get_naive_flashmla_metadata(self, cache_seqlens: torch.Tensor, seq_len_q: int):
        bs = len(cache_seqlens)
        naive_metadata = torch.zeros(39, 8).to(cache_seqlens)
        if bs <= 39:
            naive_metadata[:bs, 0] = torch.arange(bs, dtype=torch.int32)
            naive_metadata[:bs, 2] = torch.arange(bs, dtype=torch.int32)
            naive_metadata[bs:, 0] = bs
            naive_metadata[:, 3] = 32
            naive_split = torch.arange(bs + 1).to(cache_seqlens)
        else:
            seqs_per_sm = (bs + 39 - 1) // 39
            naive_metadata[:, 0] = torch.arange(0, 39 * seqs_per_sm, seqs_per_sm, dtype=torch.int32)
            naive_metadata[:, 2] = torch.clamp_max(naive_metadata[:, 0] + seqs_per_sm - 1, bs - 1)
            naive_metadata[:, 3] = 32
            naive_split = torch.arange(bs + 1).to(cache_seqlens)
        return NSAFlashMLAMetadata(
            flashmla_metadata=naive_metadata,
            num_splits=naive_split,
        )


class DpskSparseAttnMultiStepBackend:
    def __init__(self, model_runner: ModelRunner, topk: int, speculative_num_steps: int):
        self.model_runner = model_runner
        self.topk = topk
        self.speculative_num_steps = speculative_num_steps
        self.attn_backends: list[DpskSparseAttnBackend] = []
        for i in range(self.speculative_num_steps):
            self.attn_backends.append(
                DpskSparseAttnBackend(
                    model_runner,
                    speculative_step_id=i,
                    speculative_num_steps=self.speculative_num_steps,
                )
            )

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        for i in range(self.speculative_num_steps - 1):
            self.attn_backends[i].init_forward_metadata(forward_batch)

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        for i in range(self.speculative_num_steps):
            self.attn_backends[i].init_cuda_graph_state(max_bs, max_num_tokens)

    def init_forward_metadata_capture_cuda_graph(self, forward_batch: ForwardBatch):
        for i in range(self.speculative_num_steps):
            self.attn_backends[i].init_forward_metadata_capture_cuda_graph(
                forward_batch.batch_size,
                forward_batch.batch_size * self.topk,
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                forward_mode=ForwardMode.DECODE,
                spec_info=forward_batch.spec_info,
            )

    def init_forward_metadata_replay_cuda_graph(self, forward_batch: ForwardBatch, bs: int):
        for i in range(self.speculative_num_steps):
            self.attn_backends[i].init_forward_metadata_replay_cuda_graph(
                bs,
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                seq_lens_sum=-1,
                forward_mode=ForwardMode.DECODE,
                spec_info=forward_batch.spec_info,
                seq_lens_cpu=forward_batch.seq_lens_cpu,
            )


import triton  # noqa: E402
import triton.language as tl  # noqa: E402


@triton.jit
def indices_trans_local_kernel(
    indices_ptr,  # [b, 2048]
    seqlens_ptr,  # [b]
    out_indices_ptr,  # [b, 2048]
    out_lens_ptr,  # [b]
    tp_rank,
    tp_size,
    page_size,
    reserved_pages,
):
    BK: tl.constexpr = 2048
    pid = tl.program_id(0)
    seqlen = tl.load(seqlens_ptr + pid)

    k_off = tl.arange(0, BK)
    mask = k_off < seqlen
    # [2048]
    indices = tl.load(indices_ptr + BK * pid + k_off, mask=mask, other=-1)
    page_ids = indices // page_size
    cond = ((page_ids - reserved_pages) % tp_size) == tp_rank
    cond = cond & mask
    new_page_ids = ((page_ids - reserved_pages) // tp_size) + reserved_pages
    new_indices = new_page_ids * page_size + (indices % page_size)
    filtered = tl.where(cond, new_indices, -1)
    keep = cond.to(tl.int32)
    cumsum = tl.cumsum(keep, axis=0)
    write_pos = cumsum - 1
    out_offsets = pid * BK + write_pos
    tl.store(out_indices_ptr + out_offsets, filtered, mask=cond)
    new_len = tl.sum(keep, axis=0)
    tl.store(out_lens_ptr + pid, new_len)


def indices_trans_local(
    indices: torch.Tensor,  # [b, 2048]
    seqlens: torch.Tensor,  # [b]
    tp_rank: int,
    tp_size: int,
    page_size: int,
    reserved_pages: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    b = indices.shape[0]
    local_indices = indices.clone()
    compute_lens = seqlens.clone()
    indices_trans_local_kernel[(b,)](indices, seqlens, local_indices, compute_lens, tp_rank, tp_size, page_size, reserved_pages)
    return local_indices, compute_lens


def indices_trans_local_pt(
    indices: torch.Tensor,  # [b, 2048]
    seqlens: torch.Tensor,  # [b]
    cp_rank: int,
    cp_size: int,
    page_size: int,
    reserved_pages: int,
):
    # for other topk shape
    b = indices.shape[0]
    topk_len = indices.shape[1]
    local_indices = indices.clone()
    compute_lens = seqlens.clone()

    local_indices = indices.new_zeros(b, topk_len + 1)
    page_ids = indices // page_size
    mask = (page_ids >= reserved_pages) & ((page_ids - reserved_pages) % cp_size == cp_rank)
    # [bs, s]
    len_mask = torch.arange(topk_len, device=seqlens.device)[None, :] < seqlens[:, None]
    mask = mask & len_mask
    pos = torch.cumsum(mask.int(), dim=1) - 1

    safe_pos = torch.where(mask, pos, topk_len)
    new_pages = (page_ids - reserved_pages) // cp_size + reserved_pages
    new_indices = new_pages * page_size + indices % page_size
    safe_val = torch.where(mask, new_indices, 0)

    local_indices.scatter_(dim=1, index=safe_pos, src=safe_val)
    compute_lens = mask.sum(dim=1, dtype=torch.int32)

    return local_indices, compute_lens
