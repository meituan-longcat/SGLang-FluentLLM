from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

import flash_mla_fp8
import flash_mla_swap
import torch

from sglang.srt.env import global_server_args_dict
from sglang.srt.layers.attention.dp_attention import get_attention_tp_rank, get_attention_tp_size
from sglang.srt.layers.attention.flashinfer_mla_backend import FlashInferMLAAttnBackend
from sglang.srt.layers.attention.radix_attention import RadixAttention
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode


def get_flash_mla_module(M, quant_method):
    if M <= 56 and quant_method != "per_token_head":
        return flash_mla_swap
    else:
        return flash_mla_fp8


PAGE_SIZE = 64

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.speculative.eagle_utils import EagleVerifyInput


@dataclass
class FlashMLADecodeMetadata:
    flashmla_metadata: tuple[torch.Tensor, torch.Tensor] | None = None
    num_splits: torch.Tensor | None = None
    block_table: torch.Tensor | None = None
    seq_lens_k: torch.Tensor | None = None


class FlashMLABackend(FlashInferMLAAttnBackend):
    def __init__(
        self,
        model_runner: ModelRunner,
        skip_prefill: bool = False,
        kv_indptr_buf: torch.Tensor = None,
        q_indptr_decode_buf: torch.Tensor = None,
        multi_step_seqlen_incr=0,
    ):
        super().__init__(model_runner, skip_prefill, kv_indptr_buf, q_indptr_decode_buf)
        self.softmax_scale = model_runner.model_config.scaling
        if model_runner.server_args.speculative_algorithm is None:
            self.draft_token_num = 0
        else:
            self.draft_token_num = model_runner.server_args.speculative_num_draft_tokens
        self.multi_step_seqlen_incr = multi_step_seqlen_incr
        self.cache_dtype = model_runner.kv_cache_dtype
        self.cache_quant_method = model_runner.kv_cache_quant_method
        if global_server_args_dict["enable_mla_l1_5_cache"]:
            self.num_q_heads = model_runner.model_config.num_attention_heads
        else:
            self.num_q_heads = model_runner.model_config.num_attention_heads // get_attention_tp_size()
        self.tp_rank = get_attention_tp_rank()
        # kv_allocator maintains block_table
        self.kv_allocator = model_runner.kv_allocator
        self.token_to_kv_pool = model_runner.token_to_kv_pool

        self.forward_metadata: FlashMLADecodeMetadata = None
        self.kv_lora_rank = model_runner.model_config.kv_lora_rank
        self.qk_nope_head_dim = model_runner.model_config.qk_nope_head_dim
        self.qk_rope_head_dim = model_runner.model_config.qk_rope_head_dim
        self.v_head_dim = model_runner.model_config.v_head_dim
        self.scaling = model_runner.model_config.scaling
        self.q_data_type = model_runner.dtype
        self.kv_cache_dim = self.kv_lora_rank + self.qk_rope_head_dim

        self.out_cache_loc = None

    def get_flash_mla_module(self, M):
        return get_flash_mla_module(M, self.cache_quant_method)

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
        q_pe: torch.Tensor | None = None,
        k_pe: torch.Tensor | None = None,
        **kwargs,
    ):
        if forward_batch.forward_mode == ForwardMode.EXTEND:
            # Prefill of base model and draft model still use flashinfer_mla as backend.
            # And draft_extend after verify is now effectively with padding, both use flashmla with verify,
            # their metadata should be the same.
            return super().forward_extend(q, k, v, layer, forward_batch, save_kv_cache, q_pe, k_pe)
        else:
            assert forward_batch.forward_mode.is_target_verify() or forward_batch.forward_mode.is_draft_extend()
            cache_loc = forward_batch.out_cache_loc if getattr(self, "out_cache_loc", None) is None else self.out_cache_loc
            if k is not None:
                assert v is not None
                if save_kv_cache:
                    forward_batch.token_to_kv_pool.set_kv_buffer(layer, cache_loc, k, v)

            bs = forward_batch.batch_size
            k_cache = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)

            assert layer.tp_q_head_num == self.num_q_heads, f"{layer.tp_q_head_num=} != {self.num_q_heads=}"
            reshape_q = q.view(bs, -1, self.num_q_heads, layer.head_dim)
            flash_mla_module = self.get_flash_mla_module(reshape_q.shape[1] * reshape_q.shape[2])

            if self.cache_quant_method == "per_token_head":
                reshape_q_contiguous = reshape_q.contiguous()
                q_nope, q_scale, q_rope = flash_mla_module.quantize_ckv_per_token_head(reshape_q_contiguous, self.kv_lora_rank)
                k_cache_lora, k_scale, k_cache_rope = k_cache
                o, lse = flash_mla_module.flash_mla_ckv_fp8_per_token(
                    q_nope=q_nope,
                    q_rope=q_rope,
                    k_cache_lora=k_cache_lora.view(-1, PAGE_SIZE, 1, self.kv_lora_rank),
                    k_cache_rope=k_cache_rope.view(-1, PAGE_SIZE, 1, self.qk_rope_head_dim),
                    q_scale=q_scale,
                    k_scale=k_scale.view(-1, PAGE_SIZE, 1, 1),
                    block_table=self.forward_metadata.block_table[:bs],
                    cache_seqlens=forward_batch.seq_lens.to(torch.int32) + self.draft_token_num,
                    head_dim_v=self.kv_lora_rank,  # TODO Retrieve from config.
                    tile_scheduler_metadata=self.forward_metadata.flashmla_metadata,
                    num_splits=self.forward_metadata.num_splits,
                    softmax_scale=layer.scaling,
                    causal=True,
                )
            elif self.cache_dtype == torch.float8_e4m3fn:
                reshape_q_fp8 = reshape_q.to(torch.float8_e4m3fn)
                o, lse = flash_mla_module.flash_mla_with_kvcache(
                    q=reshape_q_fp8,
                    k_cache=k_cache.view(-1, PAGE_SIZE, 1, self.kv_cache_dim),
                    block_table=self.forward_metadata.block_table[:bs],
                    cache_seqlens=forward_batch.seq_lens.to(torch.int32) + self.draft_token_num,
                    head_dim_v=self.kv_lora_rank,
                    tile_scheduler_metadata=self.forward_metadata.flashmla_metadata,
                    num_splits=self.forward_metadata.num_splits,
                    softmax_scale=layer.scaling,
                    causal=True,
                    descale_q=torch.ones((1), dtype=torch.float32, device=reshape_q.device),
                    descale_k=torch.ones((1), dtype=torch.float32, device=reshape_q.device),
                )
            else:
                causal = True
                if global_server_args_dict["enable_mla_l1_5_cache"] and self.tp_rank != 0:
                    causal = False
                o, lse = flash_mla_module.flash_mla_with_kvcache(
                    q=reshape_q,
                    k_cache=k_cache.view(-1, PAGE_SIZE, 1, self.kv_cache_dim),
                    block_table=self.forward_metadata.block_table[:bs],
                    cache_seqlens=self.forward_metadata.seq_lens_k[:bs],
                    head_dim_v=self.kv_lora_rank,
                    tile_scheduler_metadata=self.forward_metadata.flashmla_metadata,
                    num_splits=self.forward_metadata.num_splits,
                    softmax_scale=layer.scaling,
                    causal=causal,
                )
                if global_server_args_dict["enable_mla_l1_5_cache"]:
                    forward_batch.token_to_kv_pool.set_kv_buffer(layer, forward_batch.out_cache_loc, k, v)
            return o.view(-1, layer.tp_q_head_num * layer.v_head_dim), lse.transpose(1, 2).reshape(-1, layer.tp_q_head_num)

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
        **kwargs,
    ) -> torch.Tensor:
        cache_loc = forward_batch.out_cache_loc
        if k is not None:
            assert v is not None
            if save_kv_cache:
                forward_batch.token_to_kv_pool.set_kv_buffer(
                    layer,
                    cache_loc,
                    k,
                    v,
                )
        bs = forward_batch.batch_size
        k_cache = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)
        assert layer.tp_q_head_num == self.num_q_heads, f"{layer.tp_q_head_num=} != {self.num_q_heads=}"
        reshape_q = q.view(bs, -1, self.num_q_heads, layer.head_dim)
        flash_mla_module = self.get_flash_mla_module(reshape_q.shape[1] * reshape_q.shape[2])
        if self.cache_quant_method == "per_token_head":
            reshape_q_contiguous = reshape_q.contiguous()
            q_nope, q_scale, q_rope = flash_mla_module.quantize_ckv_per_token_head(reshape_q_contiguous, self.kv_lora_rank)
            k_cache_lora, k_scale, k_cache_rope = k_cache
            o, lse = flash_mla_module.flash_mla_ckv_fp8_per_token(
                q_nope=q_nope,
                q_rope=q_rope,
                k_cache_lora=k_cache_lora.view(-1, PAGE_SIZE, 1, self.kv_lora_rank),
                k_cache_rope=k_cache_rope.view(-1, PAGE_SIZE, 1, self.qk_rope_head_dim),
                q_scale=q_scale,
                k_scale=k_scale.view(-1, PAGE_SIZE, 1, 1),
                block_table=self.forward_metadata.block_table[:bs],
                cache_seqlens=self.forward_metadata.seq_lens_k,
                head_dim_v=self.kv_lora_rank,  # TODO Retrieve from config.
                tile_scheduler_metadata=self.forward_metadata.flashmla_metadata,
                num_splits=self.forward_metadata.num_splits,
                softmax_scale=layer.scaling,
                causal=True,
            )

            return o.view(-1, layer.tp_q_head_num * layer.v_head_dim), lse
        elif self.cache_dtype == torch.float8_e4m3fn:
            reshape_q_fp8 = reshape_q.to(torch.float8_e4m3fn)
            o, lse = flash_mla_module.flash_mla_with_kvcache(
                q=reshape_q_fp8,
                k_cache=k_cache.view(-1, PAGE_SIZE, 1, self.kv_cache_dim),
                block_table=self.forward_metadata.block_table[:bs],
                cache_seqlens=self.forward_metadata.seq_lens_k,
                head_dim_v=self.kv_lora_rank,  # TODO Retrieve from config.
                tile_scheduler_metadata=self.forward_metadata.flashmla_metadata,
                num_splits=self.forward_metadata.num_splits,
                softmax_scale=layer.scaling,
                causal=True,
                descale_q=torch.ones((1), dtype=torch.float32, device=reshape_q.device),
                descale_k=torch.ones((1), dtype=torch.float32, device=reshape_q.device),
            )

            return o.view(-1, layer.tp_q_head_num * layer.v_head_dim), lse
        else:
            # todo: need check all causal True or False?
            o, lse = flash_mla_module.flash_mla_with_kvcache(
                q=reshape_q,
                k_cache=k_cache.view(-1, PAGE_SIZE, 1, self.kv_cache_dim),
                block_table=self.forward_metadata.block_table[:bs],
                cache_seqlens=self.forward_metadata.seq_lens_k,
                head_dim_v=self.kv_lora_rank,  # TODO Retrieve from config.
                tile_scheduler_metadata=self.forward_metadata.flashmla_metadata,
                num_splits=self.forward_metadata.num_splits,
                softmax_scale=layer.scaling,
                causal=True,
            )

            return o.view(-1, layer.tp_q_head_num * layer.v_head_dim), lse.transpose(1, 2).reshape(-1, layer.tp_q_head_num)

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        block_table = self.kv_allocator.req_to_page[forward_batch.req_pool_indices]
        if forward_batch.forward_mode.is_decode_or_idle():
            seqlens = forward_batch.seq_lens.to(torch.int32) + self.multi_step_seqlen_incr
            if global_server_args_dict["enable_mla_l1_5_cache"]:
                local_table, local_lens = self.token_to_kv_pool.page_table_to_local(block_table, seqlens)
            else:
                local_table, local_lens = block_table, seqlens
            mla_metadata, num_splits = self.get_flash_mla_module(self.num_q_heads).get_mla_metadata(
                local_lens,
                self.num_q_heads,
                1,
            )
            self.forward_metadata = FlashMLADecodeMetadata(mla_metadata, num_splits, local_table, local_lens)
            self.out_cache_loc = None
        elif forward_batch.forward_mode.is_target_verify() or forward_batch.forward_mode.is_draft_extend():
            if global_server_args_dict["enable_mla_l1_5_cache"]:
                local_table, local_lens, out_cache_loc = self.get_dcp_meta_for_mtp(forward_batch.seq_lens.int(), block_table)
                self.out_cache_loc = out_cache_loc
                mla_metadata, num_splits = self.get_flash_mla_module(self.draft_token_num * self.num_q_heads).get_mla_metadata(
                    local_lens,
                    self.draft_token_num * self.num_q_heads,
                    1,
                )
                self.forward_metadata = FlashMLADecodeMetadata(mla_metadata, num_splits, local_table, local_lens)
            else:
                seq_lens = forward_batch.seq_lens + self.draft_token_num
                seq_lens = seq_lens.to(torch.int32)
                mla_metadata, num_splits = self.get_flash_mla_module(self.draft_token_num * self.num_q_heads).get_mla_metadata(
                    seq_lens.to(torch.int32),
                    self.draft_token_num * self.num_q_heads,
                    1,
                )
                self.forward_metadata = FlashMLADecodeMetadata(mla_metadata, num_splits, block_table, seq_lens)
        else:
            super().init_forward_metadata(forward_batch)

    def init_cuda_graph_state(
        self,
        max_bs: int,
        block_kv_indices: torch.Tensor | None = None,
    ):
        if block_kv_indices is None:
            max_context_len = self.max_context_len + PAGE_SIZE - 1
            # 4 PAGES are reserved for speculation
            cuda_graph_kv_indices = torch.full(
                (max_bs, (max_context_len + 4 * PAGE_SIZE) // PAGE_SIZE),
                1,
                dtype=torch.int32,
                device="cuda",
            )
        else:
            cuda_graph_kv_indices = block_kv_indices

        device = cuda_graph_kv_indices.device
        compute_q_heads = self.num_q_heads
        max_q_token = max_bs
        if self.draft_token_num:
            compute_q_heads *= self.draft_token_num
            max_q_token *= self.draft_token_num
        self.cuda_graph_mla_metadata, self.cuda_graph_num_splits = self.get_flash_mla_module(compute_q_heads).get_mla_metadata(
            torch.ones(max_bs, dtype=torch.int32, device=device),
            compute_q_heads,
            1,
        )
        self.cuda_graph_kv_indices = cuda_graph_kv_indices
        self.cuda_graph_seq_lens_k = torch.ones(max_bs, dtype=torch.int32, device=device)
        if global_server_args_dict["enable_mla_l1_5_cache"]:
            self.cuda_graph_cache_loc = torch.ones(max_q_token, dtype=torch.int32, device=device)

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        forward_mode: ForwardMode,
        spec_info: EagleVerifyInput | None = None,
    ):
        block_table = self.kv_allocator.req_to_page[req_pool_indices]
        if forward_mode.is_decode_or_idle():
            mla_metadata, num_splits = self.get_flash_mla_module(self.num_q_heads).get_mla_metadata(
                (seq_lens + self.multi_step_seqlen_incr).to(torch.int32),
                self.num_q_heads,
                1,
            )
            self.cuda_graph_mla_metadata.copy_(mla_metadata)
            self.cuda_graph_num_splits[: bs + 1].copy_(num_splits)
            self.cuda_graph_kv_indices[:bs].copy_(block_table)
            self.cuda_graph_seq_lens_k[:bs].copy_(seq_lens[:bs])
            self.forward_metadata = FlashMLADecodeMetadata(
                self.cuda_graph_mla_metadata,
                self.cuda_graph_num_splits[: bs + 1],
                self.cuda_graph_kv_indices[:bs, :],
                self.cuda_graph_seq_lens_k[:bs],
            )
        elif forward_mode.is_target_verify() or forward_mode.is_draft_extend():
            seq_lens = seq_lens + self.draft_token_num
            mla_metadata, num_splits = self.get_flash_mla_module(self.draft_token_num * self.num_q_heads).get_mla_metadata(
                seq_lens.to(torch.int32),
                self.draft_token_num * self.num_q_heads,
                1,
            )
            self.cuda_graph_mla_metadata.copy_(mla_metadata)
            self.cuda_graph_num_splits[: bs + 1].copy_(num_splits)
            self.cuda_graph_kv_indices[:bs].copy_(block_table)
            self.cuda_graph_seq_lens_k[:bs].copy_(seq_lens[:bs])
            self.forward_metadata = FlashMLADecodeMetadata(
                self.cuda_graph_mla_metadata,
                self.cuda_graph_num_splits[: bs + 1],
                self.cuda_graph_kv_indices[:bs],
                self.cuda_graph_seq_lens_k[:bs],
            )
            if global_server_args_dict["enable_mla_l1_5_cache"] and self.tp_rank == 0:
                self.out_cache_loc = self.cuda_graph_cache_loc[: bs * self.draft_token_num]
        else:
            raise RuntimeError(f"Not supported forward mode: {forward_mode}")

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        forward_mode: ForwardMode,
        spec_info: EagleVerifyInput | None,
        seq_lens_cpu: torch.Tensor | None,
    ):
        req_pool_indices = req_pool_indices[:bs]
        block_table = self.kv_allocator.req_to_page[req_pool_indices]
        seq_lens = seq_lens[:bs]
        if forward_mode.is_decode_or_idle():
            seq_lens = seq_lens.int() + self.multi_step_seqlen_incr
            if global_server_args_dict["enable_mla_l1_5_cache"]:
                local_table, local_lens = self.token_to_kv_pool.page_table_to_local(block_table, seq_lens)
            else:
                local_table, local_lens = block_table, seq_lens
            mla_metadata, num_splits = self.get_flash_mla_module(self.num_q_heads).get_mla_metadata(
                local_lens,
                self.num_q_heads,
                1,
            )
            self.cuda_graph_mla_metadata.copy_(mla_metadata)
            self.cuda_graph_num_splits[: bs + 1].copy_(num_splits)
            self.cuda_graph_kv_indices[:bs, : local_table.shape[1]].copy_(local_table)
            self.cuda_graph_seq_lens_k[:bs].copy_(local_lens)
            self.forward_metadata.flashmla_metadata = self.cuda_graph_mla_metadata
            self.forward_metadata.num_splits = self.cuda_graph_num_splits[: bs + 1]
            self.forward_metadata.block_table = self.cuda_graph_kv_indices[:bs]
            self.forward_metadata.seq_lens_k = self.cuda_graph_seq_lens_k[:bs]
        elif forward_mode.is_target_verify() or forward_mode.is_draft_extend():
            if global_server_args_dict["enable_mla_l1_5_cache"]:
                local_table, local_lens, out_cache_loc = self.get_dcp_meta_for_mtp(seq_lens.int(), block_table)
                if out_cache_loc is not None:
                    self.cuda_graph_cache_loc[: out_cache_loc.shape[0]].copy_(out_cache_loc)
                    self.out_cache_loc = self.cuda_graph_cache_loc[: out_cache_loc.shape[0]]
                mla_metadata, num_splits = self.get_flash_mla_module(self.draft_token_num * self.num_q_heads).get_mla_metadata(
                    local_lens,
                    self.draft_token_num * self.num_q_heads,
                    1,
                )
                self.cuda_graph_mla_metadata.copy_(mla_metadata)
                self.cuda_graph_num_splits[: bs + 1].copy_(num_splits)
                self.cuda_graph_kv_indices[:bs, : local_table.shape[1]].copy_(local_table)
                self.cuda_graph_seq_lens_k[:bs].copy_(local_lens)
                self.forward_metadata.flashmla_metadata = self.cuda_graph_mla_metadata
                self.forward_metadata.num_splits = self.cuda_graph_num_splits[: bs + 1]
                self.forward_metadata.block_table = self.cuda_graph_kv_indices[:bs]
                self.forward_metadata.seq_lens_k = self.cuda_graph_seq_lens_k[:bs]
            else:
                seq_lens = seq_lens[:bs] + self.draft_token_num
                mla_metadata, num_splits = self.get_flash_mla_module(self.draft_token_num * self.num_q_heads).get_mla_metadata(
                    seq_lens.to(torch.int32),
                    self.draft_token_num * self.num_q_heads,
                    1,
                )
                self.cuda_graph_mla_metadata.copy_(mla_metadata)
                self.cuda_graph_num_splits[: bs + 1].copy_(num_splits)
                self.cuda_graph_kv_indices[:bs].copy_(block_table)
                self.cuda_graph_seq_lens_k[:bs].copy_(seq_lens)
                self.forward_metadata.flashmla_metadata = self.cuda_graph_mla_metadata
                self.forward_metadata.num_splits = self.cuda_graph_num_splits[: bs + 1]
                self.forward_metadata.block_table = self.cuda_graph_kv_indices[:bs]
                self.forward_metadata.seq_lens_k = self.cuda_graph_seq_lens_k[:bs]
        else:
            raise RuntimeError(f"Not supported forward mode: {forward_mode}")

    def get_cuda_graph_seq_len_fill_value(self):
        return 1

    def get_dcp_meta_for_mtp(
        self,
        seq_lens: torch.Tensor,
        block_table: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        bs = seq_lens.shape[0]
        local_causal_table, local_causal_lens = self.token_to_kv_pool.remap_page_table_to_local(block_table, seq_lens + self.draft_token_num)
        local_tight_table, local_tight_lens = self.token_to_kv_pool.remap_page_table_to_local(block_table, seq_lens)
        rank = get_attention_tp_rank()
        device = seq_lens.device
        if rank == 0:
            page_size = self.token_to_kv_pool.page_size
            pad_page_ids = torch.arange(1, bs + 1, dtype=torch.int32, device=device)
            pad_page_pos = (local_causal_lens - 1) // page_size + 1
            # pad_page_pos is safe, because page_table already have 4 spare pages reserved
            torch._assert_async(
                torch.all((pad_page_pos >= 0) & (pad_page_pos < local_causal_table.shape[1])), "get_dcp_meta_for_mtp pad_page_pos illegal."
            )
            local_causal_table.scatter_(dim=1, index=pad_page_pos[:, None], src=pad_page_ids[:, None])
            compute_lens = local_tight_lens + self.draft_token_num
            # [bs, draft_token_num]
            token_pos = local_tight_lens[:, None] + torch.arange(self.draft_token_num, dtype=torch.int32, device=device)[None, :]
            page_ids = local_causal_table.gather(dim=1, index=token_pos // page_size)
            out_cache_loc = page_ids * page_size + token_pos % page_size
            out_cache_loc = out_cache_loc.flatten()
            local_table = local_causal_table
            local_lens = compute_lens
        else:
            out_cache_loc = None
            local_table = local_tight_table
            local_lens = local_tight_lens
        return local_table, local_lens, out_cache_loc


class FlashMLAMultiStepDecodeBackend:
    def __init__(self, model_runner, spec_num_steps: int):
        self.attn_backends = []
        self.spec_num_steps = spec_num_steps
        for i in range(self.spec_num_steps):
            self.attn_backends.append(
                FlashMLABackend(
                    model_runner=model_runner,
                    multi_step_seqlen_incr=i + 1,
                    skip_prefill=True,
                )
            )

    def common_template(
        self,
        forward_batch: ForwardBatch,
        call_fn: Callable,
    ):
        assert forward_batch.spec_info is not None

        for i in range(self.spec_num_steps - 1):
            call_fn(i, forward_batch)

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        def call_fn(i, forward_batch):
            assert forward_batch.spec_info is not None
            self.attn_backends[i].init_forward_metadata(forward_batch)

        self.common_template(forward_batch, call_fn)

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        for i in range(self.speculative_num_steps):
            self.attn_backends[i].init_cuda_graph_state(max_bs, max_num_tokens, block_kv_indices=None)

    def init_forward_metadata_capture_cuda_graph(self, forward_batch: ForwardBatch):
        def call_fn(i, forward_batch):
            self.attn_backends[i].init_forward_metadata_capture_cuda_graph(
                forward_batch.batch_size,
                forward_batch.batch_size * self.topk,
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
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
                forward_mode=ForwardMode.DECODE,
                spec_info=forward_batch.spec_info,
                seq_lens_cpu=None,
            )

        self.common_template(forward_batch, call_fn)

    def get_cuda_graph_seq_len_fill_value(self):
        return 1
