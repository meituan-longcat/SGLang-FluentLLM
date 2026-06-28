from __future__ import annotations

import torch
from dataclasses import dataclass
from typing import Optional, Tuple, Union, Callable, TYPE_CHECKING

from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.layers.attention.flashinfer_mla_backend import FlashInferMLAAttnBackend

from sglang.srt.layers.dp_attention import (
    get_attention_tp_size,
)
import flash_mla_fp8
import flash_mla_swap


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
    flashmla_metadata: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    num_splits: Optional[torch.Tensor] = None
    block_table: Optional[torch.Tensor] = None

    def __init__(
        self,
        flashmla_metadata: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        num_splits: Optional[torch.Tensor] = None,
        block_table: Optional[torch.Tensor] = None,
    ):
        self.flashmla_metadata = flashmla_metadata
        self.num_splits = num_splits
        self.block_table = block_table


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
        self.num_q_heads = (
            model_runner.model_config.num_attention_heads // get_attention_tp_size()
        )
        # kv_allocator maintains block_table
        self.kv_allocator = model_runner.kv_allocator

        self.num_local_heads = (
            model_runner.model_config.num_attention_heads // get_attention_tp_size()
        )
        self.forward_metadata: Union[FlashMLADecodeMetadata] = None
        self.kv_lora_rank = model_runner.model_config.kv_lora_rank
        self.qk_nope_head_dim = model_runner.model_config.qk_nope_head_dim
        self.qk_rope_head_dim = model_runner.model_config.qk_rope_head_dim
        self.v_head_dim = model_runner.model_config.v_head_dim
        self.scaling = model_runner.model_config.scaling
        self.q_data_type = model_runner.dtype
        self.kv_cache_dim = self.kv_lora_rank + self.qk_rope_head_dim

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
        q_pe: Optional[torch.Tensor] = None,
        k_pe: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        if forward_batch.forward_mode == ForwardMode.EXTEND:
            # Prefill of base model and draft model still use flashinfer_mla as backend.
            # And draft_extend after verify is now effectively with padding, both use flashmla with verify,
            # their metadata should be the same.
            return super().forward_extend(q, k, v, layer, forward_batch, save_kv_cache, q_pe, k_pe)
        else:
            assert (
                forward_batch.forward_mode.is_target_verify()
                or forward_batch.forward_mode.is_draft_extend()
            )
            cache_loc = forward_batch.out_cache_loc
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
                    cache_seqlens=forward_batch.seq_lens.to(torch.int32)
                    + self.draft_token_num,
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
                    cache_seqlens=forward_batch.seq_lens.to(torch.int32)
                    + self.draft_token_num,
                    head_dim_v=self.kv_lora_rank,
                    tile_scheduler_metadata=self.forward_metadata.flashmla_metadata,
                    num_splits=self.forward_metadata.num_splits,
                    softmax_scale=layer.scaling,
                    causal=True,
                    descale_q=torch.ones(
                        (1), dtype=torch.float32, device=reshape_q.device
                    ),
                    descale_k=torch.ones(
                        (1), dtype=torch.float32, device=reshape_q.device
                    ),
                )
            else:
                o, lse = flash_mla_module.flash_mla_with_kvcache(
                    q=reshape_q,
                    k_cache=k_cache.view(-1, PAGE_SIZE, 1, self.kv_cache_dim),
                    block_table=self.forward_metadata.block_table[:bs],
                    cache_seqlens=forward_batch.seq_lens.to(torch.int32)
                    + self.draft_token_num,
                    head_dim_v=self.kv_lora_rank,
                    tile_scheduler_metadata=self.forward_metadata.flashmla_metadata,
                    num_splits=self.forward_metadata.num_splits,
                    softmax_scale=layer.scaling,
                    causal=True,
                )
            return o.view(-1, layer.tp_q_head_num * layer.v_head_dim), lse

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
        cache_lens = forward_batch.seq_lens + self.multi_step_seqlen_incr
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
                cache_seqlens=cache_lens.to(torch.int32),
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
                cache_seqlens=cache_lens.to(torch.int32),
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
                cache_seqlens=cache_lens.to(torch.int32),
                head_dim_v=self.kv_lora_rank,  # TODO Retrieve from config.
                tile_scheduler_metadata=self.forward_metadata.flashmla_metadata,
                num_splits=self.forward_metadata.num_splits,
                softmax_scale=layer.scaling,
                causal=True,
            )

            return o.view(-1, layer.tp_q_head_num * layer.v_head_dim), lse

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        block_table = self.kv_allocator.req_to_page[forward_batch.req_pool_indices]
        if forward_batch.forward_mode.is_decode_or_idle():
            mla_metadata, num_splits = self.get_flash_mla_module(self.num_q_heads).get_mla_metadata(
                forward_batch.seq_lens.to(torch.int32) + self.multi_step_seqlen_incr,
                self.num_q_heads,
                1,
            )
            self.forward_metadata = FlashMLADecodeMetadata(
                mla_metadata,
                num_splits,
                block_table,
            )
        elif (
            forward_batch.forward_mode.is_target_verify()
            or forward_batch.forward_mode.is_draft_extend()
        ):
            seq_lens = forward_batch.seq_lens + self.draft_token_num
            mla_metadata, num_splits = self.get_flash_mla_module(self.draft_token_num * self.num_q_heads).get_mla_metadata(
                seq_lens.to(torch.int32),
                self.draft_token_num * self.num_q_heads,
                1,
            )
            self.forward_metadata = FlashMLADecodeMetadata(
                mla_metadata,
                num_splits,
                block_table,
            )
        else:
            super().init_forward_metadata(forward_batch)

    def init_cuda_graph_state(
        self,
        max_bs: int,
        block_kv_indices: Optional[torch.Tensor] = None,
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

        if self.draft_token_num:
            self.cuda_graph_mla_metadata, self.cuda_graph_num_splits = self.get_flash_mla_module(self.draft_token_num * self.num_q_heads).get_mla_metadata(
                torch.ones(
                    max_bs, dtype=torch.int32, device=cuda_graph_kv_indices.device
                ),
                self.draft_token_num * self.num_q_heads,
                1,
            )
        else:
            self.cuda_graph_mla_metadata, self.cuda_graph_num_splits = self.get_flash_mla_module(self.num_q_heads).get_mla_metadata(
                torch.ones(
                    max_bs, dtype=torch.int32, device=cuda_graph_kv_indices.device
                ),
                self.num_q_heads,
                1,
            )
        self.cuda_graph_kv_indices = cuda_graph_kv_indices

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        forward_mode: ForwardMode,
        spec_info: Optional[EagleVerifyInput] = None,
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
            self.forward_metadata = FlashMLADecodeMetadata(
                self.cuda_graph_mla_metadata,
                self.cuda_graph_num_splits[: bs + 1],
                self.cuda_graph_kv_indices[:bs, :],
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
            self.forward_metadata = FlashMLADecodeMetadata(
                self.cuda_graph_mla_metadata,
                self.cuda_graph_num_splits[: bs + 1],
                self.cuda_graph_kv_indices[:bs],
            )
        else:
            raise RuntimeError(f"Not supported forward mode: {forward_mode}")

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        forward_mode: ForwardMode,
        spec_info: Optional[EagleVerifyInput],
        seq_lens_cpu: Optional[torch.Tensor],
    ):
        req_pool_indices = req_pool_indices[:bs]
        block_table = self.kv_allocator.req_to_page[req_pool_indices]
        seq_lens = seq_lens[:bs]
        if forward_mode.is_decode_or_idle():
            mla_metadata, num_splits = self.get_flash_mla_module(self.num_q_heads).get_mla_metadata(
                (seq_lens + self.multi_step_seqlen_incr).to(torch.int32),
                self.num_q_heads,
                1,
            )
            self.cuda_graph_mla_metadata.copy_(mla_metadata)
            self.cuda_graph_num_splits[: bs + 1].copy_(num_splits)
            self.cuda_graph_kv_indices[:bs].copy_(block_table)
            self.forward_metadata.flashmla_metadata = self.cuda_graph_mla_metadata
            self.forward_metadata.num_splits = self.cuda_graph_num_splits[: bs + 1]
            self.forward_metadata.block_table = self.cuda_graph_kv_indices[:bs]
        elif forward_mode.is_target_verify() or forward_mode.is_draft_extend():
            seq_lens = seq_lens[:bs] + self.draft_token_num
            mla_metadata, num_splits = self.get_flash_mla_module(self.draft_token_num * self.num_q_heads).get_mla_metadata(
                seq_lens.to(torch.int32),
                self.draft_token_num * self.num_q_heads,
                1,
            )
            self.cuda_graph_mla_metadata.copy_(mla_metadata)
            self.cuda_graph_num_splits[: bs + 1].copy_(num_splits)
            self.cuda_graph_kv_indices[:bs].copy_(block_table)
            self.forward_metadata.flashmla_metadata = self.cuda_graph_mla_metadata
            self.forward_metadata.num_splits = self.cuda_graph_num_splits[: bs + 1]
            self.forward_metadata.block_table = self.cuda_graph_kv_indices[:bs]
        else:
            raise RuntimeError(f"Not supported forward mode: {forward_mode}")

    def get_cuda_graph_seq_len_fill_value(self):
        return 1


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
            self.attn_backends[i].init_cuda_graph_state(
                max_bs, max_num_tokens, block_kv_indices=None
            )

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

    def init_forward_metadata_replay_cuda_graph(
        self, forward_batch: ForwardBatch, bs: int
    ):
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
