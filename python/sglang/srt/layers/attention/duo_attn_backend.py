
from __future__ import annotations
import os
from typing import TYPE_CHECKING, Optional, List

import torch
import numpy as np
import math
import inspect

from sglang.srt.layers.attention.flash_attention_backend import FlashAttentionBackend
from sglang.srt.layers.dp_attention import get_attention_tp_size, get_attention_tp_rank
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
# from block_sparse_attn import block_streaming_attn_func
from duo_flash_attn_interface import flash_attn_varlen_func, flash_attn_with_kvcache, get_scheduler_metadata
from duo_flash_mla_swap import flash_mla_with_kvcache, get_mla_metadata

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner


def use_flash_mla_swap(M):
    return M <= 56


class DuoAttnBackend(FlashAttentionBackend):
    def __init__(
        self,
        model_runner: ModelRunner,
        skip_prefill: bool = False,
        kv_indptr_buf: Optional[torch.Tensor] = None,
        speculative_step_id: int = 0,
        topk: int = 0,
        speculative_num_steps: int = 0,
    ):
        super().__init__(
            model_runner,
            skip_prefill,
            speculative_step_id,
            topk,
            speculative_num_steps,
        )
        model_config = model_runner.model_config
        self.num_q_head = model_config.num_attention_heads
        self.num_k_head = model_config.num_key_value_heads
        self.tp_size = get_attention_tp_size()
        self.local_q_head = self.num_q_head // self.tp_size
        self.local_k_head = self.num_k_head // self.tp_size
        self.max_seq_len = model_runner.server_args.context_length
        if model_runner.server_args.speculative_algorithm is None:
            self.draft_token_num = 0
        else:
            self.draft_token_num = model_runner.server_args.speculative_num_draft_tokens
        # duo attn config
        duo_attention_config = model_config.hf_config.streaming_sparse_attention
        self.sink_size = duo_attention_config.get('sink_size', 128)
        self.recent_size = duo_attention_config.get('recent_size', 256)
        self.sparsity = duo_attention_config.get('sparsity', 0.5)
        head_score_path = duo_attention_config.get('head_score', None)
        if (
            head_score_path is not None and
            not os.path.isabs(head_score_path)
        ):
            head_score_path = f"{model_config.model_path}/{head_score_path}"
        self.duo_attn_heads_pattern = head_score_path
        self.block_size = 128
        # self.layers_full_q_head_indices = []
        # self.layers_stream_q_head_indices = []
        num_sink_blocks = math.ceil(self.sink_size / self.block_size)
        num_recent_blocks = math.ceil(self.recent_size / self.block_size)
        self.streaming_info = torch.tensor(
            [num_sink_blocks, num_recent_blocks] * self.local_q_head,
            device=model_runner.device,
            dtype=torch.int32,
        )
        if not model_runner.is_draft_worker:
            self.layers_head_mask_type = [
                torch.zeros(
                    self.local_q_head, device=model_runner.device, dtype=torch.int32
                )
                for _ in range(model_config.num_attention_layers)
            ]
            self._initialize_head_classification()
            self.layers_type = [0] * model_config.num_attention_layers
            for i in range(model_config.num_attention_layers):
                if torch.all(self.layers_head_mask_type[i] == 0):
                    self.layers_type[i] = 1
        else:
            self.layers_head_mask_type = [
                torch.full(
                    (self.local_q_head,), -1, device=model_runner.device, dtype=torch.int32
                )
                for _ in range(model_config.num_attention_layers)
            ]
            self.layers_type = [0] * model_config.num_attention_layers
    
    def _load_attention_heads_score(self):
        """Load attention heads pattern from file"""
        if self.duo_attn_heads_pattern is None:
            # If no pattern provided, use all heads as full attention
            return torch.ones((1,)) # Will be resized later based on actual head count

        try:
            head_scores = np.loadtxt(
                self.duo_attn_heads_pattern,
                dtype=float,
                delimiter="\t"
            )
            head_scores = np.clip(head_scores, 0, 1)
            return torch.tensor(head_scores)
        except Exception as e:
            print(f"load attention heads pattern failed: {e}, use all full attention instead.")
            # Fallback to all full attention if file loading fails
            exit(0)
    
    def _initialize_head_classification(self):
        head_scores = self._load_attention_heads_score()    
        threshold = torch.quantile(head_scores, self.sparsity)
        full_head_mask = head_scores >= threshold
        tp_size = get_attention_tp_size()
        tp_rank = get_attention_tp_rank()

        total_kv_heads = full_head_mask.shape[1]
        kv_heads_per_rank = total_kv_heads // tp_size
        # Calculate the head indices for this tp_rank
        start_head_idx = tp_rank * kv_heads_per_rank
        end_head_idx = start_head_idx + kv_heads_per_rank
        full_head_mask = full_head_mask[:, start_head_idx: end_head_idx]
        if full_head_mask.shape[1] < self.num_q_head:
            assert self.local_q_head % full_head_mask.shape[1] == 0
            g = self.local_q_head // full_head_mask.shape[1]
            full_head_mask = full_head_mask.repeat_interleave(g, dim=1)
        for layer_idx in range(full_head_mask.shape[0]):
            # full_head_indices = full_head_mask[layer_idx].nonzero().squeeze()
            stream_head_indices = (full_head_mask[layer_idx] == 0).nonzero().squeeze()
            # self.layers_full_q_head_indices.append(full_head_indices)
            # self.layers_stream_q_head_indices.append(stream_head_indices)
            # 0 for full, 1 for blocksparse need basemask, -1 for streaming
            self.layers_head_mask_type[layer_idx][stream_head_indices] = -1
        
    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
        sinks=None,
    ):
        if save_kv_cache:
            if not self.use_mla:
                forward_batch.token_to_kv_pool.set_kv_buffer(
                    layer, forward_batch.out_cache_loc, k, v
                )
            else:
                # For MLA, k is the latent cache which is shared by the key and value.
                # Plus, it encompasses the `kv_lora_rank` and `k_rope`.
                forward_batch.token_to_kv_pool.set_kv_buffer(
                    layer, forward_batch.out_cache_loc, k, None, # The None input here is not used within the function anyway.
                )
        
        q_head = layer.tp_q_head_num
        kv_head = layer.tp_k_head_num
        head_mask_type = self.layers_head_mask_type[layer.layer_id]
        
        if not self.use_mla: # GQA.
            if forward_batch.forward_mode == ForwardMode.EXTEND:
                if not any(forward_batch.extend_prefix_lens_cpu):
                    # No prefix prefill.
                    bs = forward_batch.batch_size
                    cu_seqlens_q = self.forward_metadata.cu_seqlens_q
                    cu_seqlens_k = self.forward_metadata.cu_seqlens_k
                    q = q.view(-1, q_head, layer.qk_head_dim)
                    o = flash_attn_varlen_func(
                        q,
                        k,
                        v,
                        cu_seqlens_q=cu_seqlens_q,
                        cu_seqlens_k=cu_seqlens_k,
                        head_mask_type=head_mask_type,
                        streaming_info=self.streaming_info,
                        max_seqlen_q=self.forward_metadata.max_seq_len_q,
                        max_seqlen_k=self.forward_metadata.max_seq_len_q,
                        softmax_scale=layer.scaling,
                        causal=True,
                    )
                    o = o.view(-1, q_head * layer.head_dim)
                else:
                    # Prefix extend.
                    k_buffer = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)
                    v_buffer = forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id)
                    k_buffer, v_buffer = map(
                        lambda e: e.unflatten(0, (-1, forward_batch.token_to_kv_pool.page_size)),
                        (k_buffer, v_buffer)
                    )
                    page_table = self.forward_metadata.page_table
                    q = q.view(-1, q_head, layer.qk_head_dim)
                    o = flash_attn_with_kvcache(
                        q=q,
                        k_cache=k_buffer,
                        v_cache=v_buffer,
                        streaming_info=self.streaming_info,
                        head_mask_type=head_mask_type,
                        cu_seqlens_q=self.forward_metadata.cu_seqlens_q,
                        max_seqlen_q=self.forward_metadata.max_seq_len_q,
                        page_table=page_table,
                        cache_seqlens=self.forward_metadata.cache_seqlens_int32,
                        softmax_scale=layer.scaling,
                        causal=True
                    )
                    o = o.view(-1, q_head * layer.head_dim)
            else:
                # MTP.
                assert (
                    forward_batch.forward_mode.is_target_verify()
                    or forward_batch.forward_mode.is_draft_extend()
                ), f"{forward_batch.forward_mode=}"
                k_buffer = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)
                v_buffer = forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id)
                k_buffer, v_buffer = map(
                    lambda e: e.unflatten(0, (-1, forward_batch.token_to_kv_pool.page_size)),
                    (k_buffer, v_buffer)
                )
                bs = forward_batch.batch_size
                page_table = self.forward_metadata.page_table
                q = q.view(bs, -1, q_head, layer.head_dim)
                o = flash_attn_with_kvcache(
                    q=q,
                    k_cache=k_buffer,
                    v_cache=v_buffer,
                    streaming_info=self.streaming_info,
                    head_mask_type=head_mask_type,
                    page_table=page_table,
                    # cache_seqlens = seq_lens + s_q
                    cache_seqlens=self.forward_metadata.cache_seqlens_int32,
                    softmax_scale=layer.scaling,
                    causal=True
                )
                o = o.view(-1, q_head * layer.head_dim)
        else: # MLA.
            if forward_batch.forward_mode == ForwardMode.EXTEND:
                if not any(forward_batch.extend_prefix_lens_cpu):
                    # No prefix prefill, no absorb, MHA.
                    # `save_kv_cache` op has been done before entering the backend,
                    # since it requires accessing the latent cache which would better 
                    # not be involved in the backend during prefill.
                    assert not save_kv_cache
                    bs = forward_batch.batch_size
                    cu_seqlens_q = self.forward_metadata.cu_seqlens_q
                    cu_seqlens_k = self.forward_metadata.cu_seqlens_k
                    q = q.view(-1, q_head, layer.qk_head_dim) # E.g., 128 + 64.
                    o = flash_attn_varlen_func(
                        q,
                        k,
                        v,
                        cu_seqlens_q=cu_seqlens_q,
                        cu_seqlens_k=cu_seqlens_k,
                        head_mask_type=head_mask_type,
                        streaming_info=self.streaming_info,
                        max_seqlen_q=self.forward_metadata.max_seq_len_q,
                        max_seqlen_k=self.forward_metadata.max_seq_len_q,
                        softmax_scale=layer.scaling,
                        causal=True,
                    )
                    o = o.view(-1, q_head * layer.v_head_dim) # E.g., 128.
                else:
                    # Prefix extend, absorb, MQA.
                    k_buffer = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)
                    k_buffer = k_buffer.unflatten(0, (-1, forward_batch.token_to_kv_pool.page_size)) # Recover the shape [num_pages, page_size, ...].
                    page_table = self.forward_metadata.page_table
                    q = q.view(-1, q_head, layer.qk_head_dim) # E.g., 512 + 64.
                    o = flash_attn_with_kvcache(
                        q=q[:, :, layer.v_head_dim :], # q_rope, 64.
                        k_cache=k_buffer[:, :, :, layer.v_head_dim :], # k_rope, 64.
                        v_cache=k_buffer[:, :, :, : layer.v_head_dim], # kv_lora_rank, 512.
                        qv=q[:, :, : layer.v_head_dim], # q_nope_absorb, 512
                        streaming_info=self.streaming_info,
                        head_mask_type=head_mask_type,
                        cu_seqlens_q=self.forward_metadata.cu_seqlens_q,
                        max_seqlen_q=self.forward_metadata.max_seq_len_q,
                        page_table=page_table,
                        cache_seqlens=self.forward_metadata.cache_seqlens_int32,
                        softmax_scale=layer.scaling,
                        causal=True,
                    )
                    o = o.view(-1, q_head * layer.v_head_dim) # E.g., 512.
            else:
                # MTP.
                assert (
                    forward_batch.forward_mode.is_target_verify()
                    or forward_batch.forward_mode.is_draft_extend()
                ), f"{forward_batch.forward_mode=}"
                k_buffer = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)
                k_buffer = k_buffer.unflatten(0, (-1, forward_batch.token_to_kv_pool.page_size)) # Recover the shape [num_pages, page_size, ...].
                bs = forward_batch.batch_size
                page_table = self.forward_metadata.page_table
                q = q.view(bs, -1, q_head, layer.head_dim)
                q_seqlen = q.shape[1]
                if use_flash_mla_swap(q_seqlen * q_head):
                    if self.layers_type[layer.layer_id]:
                        tile_scheduler_metadata, num_splits = self.running_full_tile_scheduler_metadata, self.running_full_num_splits
                    else:
                        tile_scheduler_metadata, num_splits = self.running_stream_tile_scheduler_metadata, self.running_stream_num_splits                    
                    o, _ = flash_mla_with_kvcache(
                        q=q,
                        k_cache=k_buffer,
                        head_dim_v=layer.v_head_dim,
                        streaming_info=self.streaming_info, 
                        head_mask_type=head_mask_type,
                        block_table=page_table,
                        cache_seqlens=self.forward_metadata.cache_seqlens_int32,
                        softmax_scale=layer.scaling,
                        causal=True,
                        tile_scheduler_metadata=tile_scheduler_metadata,
                        num_splits=num_splits[: bs + 1],
                    )
                else:
                    o = flash_attn_with_kvcache(
                        q=q[:, :, :, layer.v_head_dim :], # q_rope, 64.
                        k_cache=k_buffer[:, :, :, layer.v_head_dim :], # k_rope, 64.
                        v_cache=k_buffer[:, :, :, : layer.v_head_dim], # kv_lora_rank, 512.
                        qv=q[:, :, :, : layer.v_head_dim], # q_nope_absorb, 512
                        streaming_info=self.streaming_info,
                        head_mask_type=head_mask_type,
                        page_table=page_table,
                        # cache_seqlens = seq_lens + s_q
                        cache_seqlens=self.forward_metadata.cache_seqlens_int32,
                        softmax_scale=layer.scaling,
                        causal=True
                    )
                o = o.view(-1, q_head * layer.v_head_dim)
        return o

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
        sinks=None,
    ):
        if save_kv_cache:
            if not self.use_mla:
                forward_batch.token_to_kv_pool.set_kv_buffer(
                    layer, forward_batch.out_cache_loc, k, v
                )
            else:
                # For MLA, k is the latent cache which is shared by the key and value.
                # Plus, it encompasses the `kv_lora_rank` and `k_rope`.
                forward_batch.token_to_kv_pool.set_kv_buffer(
                    layer, forward_batch.out_cache_loc, k, None, # The None input here is not used within the function anyway.
                )

        bs = forward_batch.batch_size
        q_head = layer.tp_q_head_num
        kv_head = layer.tp_k_head_num
        head_mask_type = self.layers_head_mask_type[layer.layer_id]

        if not self.use_mla: # GQA.
            k_buffer = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)
            v_buffer = forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id)
            k_buffer, v_buffer = map(
                lambda e: e.unflatten(0, (-1, forward_batch.token_to_kv_pool.page_size)),
                (k_buffer, v_buffer)
            )
            page_table = self.forward_metadata.page_table
            q = q.view(-1, 1, q_head, layer.qk_head_dim)
            # GQA ver always requires just-in-time scheduler_metadata as there are diverse attention patterns.
            scheduler_metadata = get_scheduler_metadata(
                batch_size=bs,
                max_seqlen_q=1,
                max_seqlen_k=self.forward_metadata.max_seq_len_k,
                num_heads_q=q_head,
                num_heads_kv=kv_head,
                headdim=layer.qk_head_dim, # query and key dimension
                cache_seqlens=self.forward_metadata.cache_seqlens_int32, # [b] actual cache length per batch
                qkv_dtype=q.dtype, # input data type
                headdim_v=layer.v_head_dim, # value dimension (can be different)
                page_size=forward_batch.token_to_kv_pool.page_size, # block size
                causal=True, # optional parameter, depends on implementation whether to pass
            )
            o = flash_attn_with_kvcache(
                q=q,
                k_cache=k_buffer,
                v_cache=v_buffer,
                streaming_info=self.streaming_info,
                head_mask_type=head_mask_type,
                page_table=page_table,
                cache_seqlens=self.forward_metadata.cache_seqlens_int32,
                softmax_scale=layer.scaling,
                causal=True,
                scheduler_metadata=scheduler_metadata,
            )
            o = o.view(-1, q_head * layer.head_dim)
        else: # MLA.
            k_buffer = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)
            k_buffer = k_buffer.unflatten(0, (-1, forward_batch.token_to_kv_pool.page_size))
            page_table = self.forward_metadata.page_table
            q = q.view(-1, 1, q_head, layer.qk_head_dim)
            if use_flash_mla_swap(1 * q_head):
                if self.layers_type[layer.layer_id]:
                    tile_scheduler_metadata, num_splits = self.running_full_tile_scheduler_metadata, self.running_full_num_splits
                else:
                    tile_scheduler_metadata, num_splits = self.running_stream_tile_scheduler_metadata, self.running_stream_num_splits
                o, _ = flash_mla_with_kvcache(
                    q=q,
                    k_cache=k_buffer,
                    head_dim_v=layer.v_head_dim,
                    streaming_info=self.streaming_info, 
                    head_mask_type=head_mask_type,
                    block_table=page_table,
                    cache_seqlens=self.forward_metadata.cache_seqlens_int32,
                    softmax_scale=layer.scaling,
                    causal=True,
                    tile_scheduler_metadata=tile_scheduler_metadata,
                    num_splits=num_splits[: bs + 1],
                )
            else:
                scheduler_metadata = get_scheduler_metadata(
                    batch_size=bs,
                    max_seqlen_q=1,
                    max_seqlen_k=self.forward_metadata.max_seq_len_k,
                    num_heads_q=q_head,
                    num_heads_kv=kv_head,
                    headdim=layer.qk_head_dim, # query and key dimension
                    cache_seqlens=self.forward_metadata.cache_seqlens_int32, # [b] actual cache length per batch
                    qkv_dtype=q.dtype, # input data type
                    headdim_v=layer.v_head_dim, # value dimension (can be different)
                    page_size=forward_batch.token_to_kv_pool.page_size, # block size
                    causal=True, # optional parameter, depends on implementation whether to pass
                )
                o = flash_attn_with_kvcache(
                    q=q[:, :, :, layer.v_head_dim :], # q_rope, 64.
                    k_cache=k_buffer[:, :, :, layer.v_head_dim :], # k_rope, 64.
                    v_cache=k_buffer[:, :, :, : layer.v_head_dim], # kv_lora_rank, 512.
                    qv=q[:, :, :, : layer.v_head_dim], # q_nope_absorb, 512
                    streaming_info=self.streaming_info,
                    head_mask_type=head_mask_type,
                    page_table=page_table,
                    cache_seqlens=self.forward_metadata.cache_seqlens_int32,
                    softmax_scale=layer.scaling,
                    causal=True,
                    scheduler_metadata=scheduler_metadata,
                )
            o = o.view(-1, q_head * layer.v_head_dim)
        return o


    def init_forward_metadata(self, forward_batch: ForwardBatch):
        super().init_forward_metadata(forward_batch)
        if self.use_mla:
            if forward_batch.forward_mode.is_decode_or_idle() and use_flash_mla_swap(1 * self.local_q_head):
                full_head_mask_type = torch.zeros(
                    self.local_q_head, device=self.device, dtype=torch.int32
                )
                self.full_tile_scheduler_metadata, self.full_num_splits = get_mla_metadata(
                    self.forward_metadata.cache_seqlens_int32, 1 * self.local_q_head, # q_group
                    1, # kv_head
                    1, # s_q
                    self.streaming_info,
                    full_head_mask_type,
                )
                stream_head_mask_type = torch.full(
                    (self.local_q_head,), -1, device=self.device, dtype=torch.int32
                )
                self.stream_tile_scheduler_metadata, self.stream_num_splits = get_mla_metadata(
                    self.forward_metadata.cache_seqlens_int32, 1 * self.local_q_head, # q_group
                    1,
                    1,
                    self.streaming_info,
                    stream_head_mask_type,
                )
                self.running_full_tile_scheduler_metadata = self.full_tile_scheduler_metadata
                self.running_full_num_splits = self.full_num_splits
                self.running_stream_tile_scheduler_metadata = self.stream_tile_scheduler_metadata
                self.running_stream_num_splits = self.stream_num_splits
            elif (forward_batch.forward_mode.is_target_verify() or forward_batch.forward_mode.is_draft_extend()) and use_flash_mla_swap(self.draft_token_num * self.local_q_head):
                full_head_mask_type = torch.zeros(
                    self.local_q_head, device=self.device, dtype=torch.int32
                )
                self.full_tile_scheduler_metadata, self.full_num_splits = get_mla_metadata(
                    self.forward_metadata.cache_seqlens_int32, self.draft_token_num * self.local_q_head, # q_group
                    1, # kv_head
                    self.draft_token_num, # s_q
                    self.streaming_info,
                    full_head_mask_type,
                )
                stream_head_mask_type = torch.full(
                    (self.local_q_head,), -1, device=self.device, dtype=torch.int32
                )
                self.stream_tile_scheduler_metadata, self.stream_num_splits = get_mla_metadata(
                    self.forward_metadata.cache_seqlens_int32, self.draft_token_num * self.local_q_head, # q_group
                    1,
                    self.draft_token_num,
                    self.streaming_info,
                    stream_head_mask_type,
                )
                self.running_full_tile_scheduler_metadata = self.full_tile_scheduler_metadata
                self.running_full_num_splits = self.full_num_splits
                self.running_stream_tile_scheduler_metadata = self.stream_tile_scheduler_metadata
                self.running_stream_num_splits = self.stream_num_splits

    def init_cuda_graph_state(
        self,
        max_bs: int,
    ):
        super().init_cuda_graph_state(max_bs)
        if self.use_mla:
            if self.draft_token_num and use_flash_mla_swap(self.draft_token_num * self.local_q_head):
                full_head_mask_type = torch.zeros(
                    self.local_q_head, device=self.device, dtype=torch.int32
                )
                self.cuda_graph_full_tile_scheduler_metadata, self.cuda_graph_full_num_splits = get_mla_metadata(
                    torch.ones(
                        max_bs, dtype=torch.int32, device=self.device
                    ),
                    self.draft_token_num * self.local_q_head, # q_group
                    1, # kv_head
                    self.draft_token_num, # s_q
                    self.streaming_info,
                    full_head_mask_type,
                )
                stream_head_mask_type = torch.full(
                    (self.local_q_head,), -1, device=self.device, dtype=torch.int32
                )
                self.cuda_graph_stream_tile_scheduler_metadata, self.cuda_graph_stream_num_splits = get_mla_metadata(
                    torch.ones(
                        max_bs, dtype=torch.int32, device=self.device
                    ),
                    self.draft_token_num * self.local_q_head, # q_group
                    1,
                    self.draft_token_num,
                    self.streaming_info,
                    stream_head_mask_type,
                )
            elif use_flash_mla_swap(1 * self.local_q_head):
                full_head_mask_type = torch.zeros(
                    self.local_q_head, device=self.device, dtype=torch.int32
                )
                self.cuda_graph_full_tile_scheduler_metadata, self.cuda_graph_full_num_splits = get_mla_metadata(
                    torch.ones(
                        max_bs, dtype=torch.int32, device=self.device
                    ),
                    1 * self.local_q_head, # q_group
                    1, # kv_head
                    1, # s_q
                    self.streaming_info,
                    full_head_mask_type,
                )
                stream_head_mask_type = torch.full(
                    (self.local_q_head,), -1, device=self.device, dtype=torch.int32
                )
                self.cuda_graph_stream_tile_scheduler_metadata, self.cuda_graph_stream_num_splits = get_mla_metadata(
                    torch.ones(
                        max_bs, dtype=torch.int32, device=self.device
                    ),
                    1 * self.local_q_head, # q_group
                    1,
                    1,
                    self.streaming_info,
                    stream_head_mask_type,
                )
    
    def init_forward_metadata_capture_cuda_graph(
        self, *args, **kwargs
    ):
        sig = inspect.signature(super().init_forward_metadata_capture_cuda_graph)
        bound_args = sig.bind_partial(*args, **kwargs)
        bound_args.apply_defaults()
        super().init_forward_metadata_capture_cuda_graph(*args, **kwargs)
        bs = bound_args.arguments.get("bs")
        if self.use_mla:
            if self.draft_token_num and use_flash_mla_swap(self.draft_token_num * self.local_q_head):
                full_head_mask_type = torch.zeros(
                    self.local_q_head, device=self.device, dtype=torch.int32
                )
                t1, t2 = get_mla_metadata(
                    self.forward_metadata.cache_seqlens_int32, self.draft_token_num * self.local_q_head, # q_group
                    1, # kv_head
                    self.draft_token_num, # s_q
                    self.streaming_info,
                    full_head_mask_type,
                )
                self.cuda_graph_full_tile_scheduler_metadata.copy_(t1)
                self.cuda_graph_full_num_splits[: bs + 1].copy_(t2)
                stream_head_mask_type = torch.full(
                    (self.local_q_head,), -1, device=self.device, dtype=torch.int32
                )
                t1, t2 = get_mla_metadata(
                    self.forward_metadata.cache_seqlens_int32, self.draft_token_num * self.local_q_head,
                    1,
                    self.draft_token_num,
                    self.streaming_info,
                    stream_head_mask_type,
                )
                self.cuda_graph_stream_tile_scheduler_metadata.copy_(t1)
                self.cuda_graph_stream_num_splits[: bs + 1].copy_(t2)
                self.running_full_tile_scheduler_metadata = self.cuda_graph_full_tile_scheduler_metadata
                self.running_full_num_splits = self.cuda_graph_full_num_splits[: bs + 1]
                self.running_stream_tile_scheduler_metadata = self.cuda_graph_stream_tile_scheduler_metadata
                self.running_stream_num_splits = self.cuda_graph_stream_num_splits[: bs + 1]
            elif use_flash_mla_swap(1 * self.local_q_head):
                full_head_mask_type = torch.zeros(
                    self.local_q_head, device=self.device, dtype=torch.int32
                )
                t1, t2 = get_mla_metadata(
                    self.forward_metadata.cache_seqlens_int32, 1 * self.local_q_head, # q_group
                    1, # kv_head
                    1, # s_q
                    self.streaming_info,
                    full_head_mask_type,
                )
                self.cuda_graph_full_tile_scheduler_metadata.copy_(t1)
                self.cuda_graph_full_num_splits[: bs + 1].copy_(t2)
                stream_head_mask_type = torch.full(
                    (self.local_q_head,), -1, device=self.device, dtype=torch.int32
                )
                t1, t2 = get_mla_metadata(
                    self.forward_metadata.cache_seqlens_int32, 1 * self.local_q_head,
                    1,
                    1,
                    self.streaming_info,
                    stream_head_mask_type,
                )
                self.cuda_graph_stream_tile_scheduler_metadata.copy_(t1)
                self.cuda_graph_stream_num_splits[: bs + 1].copy_(t2)
                self.running_full_tile_scheduler_metadata = self.cuda_graph_full_tile_scheduler_metadata
                self.running_full_num_splits = self.cuda_graph_full_num_splits[: bs + 1]
                self.running_stream_tile_scheduler_metadata = self.cuda_graph_stream_tile_scheduler_metadata
                self.running_stream_num_splits = self.cuda_graph_stream_num_splits[: bs + 1]

    def init_forward_metadata_replay_cuda_graph(
        self, *args, **kwargs
    ):
        sig = inspect.signature(super().init_forward_metadata_replay_cuda_graph)
        bound_args = sig.bind_partial(*args, **kwargs)
        bound_args.apply_defaults()
        super().init_forward_metadata_replay_cuda_graph(*args, **kwargs)
        bs = bound_args.arguments.get("bs")
        if self.use_mla:
            if self.draft_token_num and use_flash_mla_swap(self.draft_token_num * self.local_q_head):
                full_head_mask_type = torch.zeros(
                    self.local_q_head, device=self.device, dtype=torch.int32
                )
                t1, t2 = get_mla_metadata(
                    self.forward_metadata.cache_seqlens_int32[: bs], self.draft_token_num * self.local_q_head, # q_group
                    1, # kv_head
                    self.draft_token_num, # s_q
                    self.streaming_info,
                    full_head_mask_type,
                )
                self.cuda_graph_full_tile_scheduler_metadata.copy_(t1)
                self.cuda_graph_full_num_splits[: bs + 1].copy_(t2)
                stream_head_mask_type = torch.full(
                    (self.local_q_head,), -1, device=self.device, dtype=torch.int32
                )
                t1, t2 = get_mla_metadata(
                    self.forward_metadata.cache_seqlens_int32[: bs], self.draft_token_num * self.local_q_head,
                    1,
                    self.draft_token_num,
                    self.streaming_info,
                    stream_head_mask_type,
                )
                self.cuda_graph_stream_tile_scheduler_metadata.copy_(t1)
                self.cuda_graph_stream_num_splits[: bs + 1].copy_(t2)
                self.running_full_tile_scheduler_metadata = self.cuda_graph_full_tile_scheduler_metadata
                self.running_full_num_splits = self.cuda_graph_full_num_splits[: bs + 1]
                self.running_stream_tile_scheduler_metadata = self.cuda_graph_stream_tile_scheduler_metadata
                self.running_stream_num_splits = self.cuda_graph_stream_num_splits[: bs + 1]
            elif use_flash_mla_swap(1 * self.local_q_head):
                full_head_mask_type = torch.zeros(
                    self.local_q_head, device=self.device, dtype=torch.int32
                )
                t1, t2 = get_mla_metadata(
                    self.forward_metadata.cache_seqlens_int32[: bs], 1 * self.local_q_head, # q_group
                    1, # kv_head
                    1, # s_q
                    self.streaming_info,
                    full_head_mask_type,
                )
                self.cuda_graph_full_tile_scheduler_metadata.copy_(t1)
                self.cuda_graph_full_num_splits[: bs + 1].copy_(t2)
                stream_head_mask_type = torch.full(
                    (self.local_q_head,), -1, device=self.device, dtype=torch.int32
                )
                t1, t2 = get_mla_metadata(
                    self.forward_metadata.cache_seqlens_int32[: bs], 1 * self.local_q_head,
                    1,
                    1,
                    self.streaming_info,
                    stream_head_mask_type,
                )
                self.cuda_graph_stream_tile_scheduler_metadata.copy_(t1)
                self.cuda_graph_stream_num_splits[: bs + 1].copy_(t2)
                self.running_full_tile_scheduler_metadata = self.cuda_graph_full_tile_scheduler_metadata
                self.running_full_num_splits = self.cuda_graph_full_num_splits[: bs + 1]
                self.running_stream_tile_scheduler_metadata = self.cuda_graph_stream_tile_scheduler_metadata
                self.running_stream_num_splits = self.cuda_graph_stream_num_splits[: bs + 1]

class DuoAttnMultiStepBackend:
    def __init__(
        self, model_runner: ModelRunner, topk: int, speculative_num_steps: int
    ):
        self.model_runner = model_runner
        self.topk = topk
        self.speculative_num_steps = speculative_num_steps
        self.attn_backends: List[DuoAttnBackend] = []
        for i in range(self.speculative_num_steps):
            self.attn_backends.append(
                DuoAttnBackend(
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

    def init_forward_metadata_replay_cuda_graph(
        self, forward_batch: ForwardBatch, bs: int
    ):
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
