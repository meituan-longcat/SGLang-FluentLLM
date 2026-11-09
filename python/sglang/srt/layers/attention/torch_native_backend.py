from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch.nn.functional import scaled_dot_product_attention

from sglang.srt.utils import is_npu

__is_npu__ = is_npu()
if __is_npu__:
    import torch_npu

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.model_executor.forward_batch_info import ForwardBatch

if TYPE_CHECKING:
    from sglang.srt.layers.attention.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner


class TorchNativeAttnBackend(AttentionBackend):
    def __init__(self, model_runner: ModelRunner):
        super().__init__()
        self.forward_metadata = None
        self.device = model_runner.device
        self.casual_mask = None

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        """Init the metadata for a forward pass."""
        pass

    def _get_or_build_block_table(self, forward_batch: ForwardBatch, req_to_token_pool, block_size: int) -> torch.Tensor:
        """get or build block_table of forwardbatch"""
        # block_table has been built, return
        if forward_batch.block_table_cache is not None:
            assert forward_batch.seq_lens_cpu is not None
            return forward_batch.block_table_cache

        # block_table has not been built, build new block_table
        req_pool_indices = forward_batch.req_pool_indices
        seq_lens = forward_batch.seq_lens
        forward_batch.seq_lens_cpu = forward_batch.seq_lens.cpu()
        req_pool_indices_cpu = req_pool_indices.cpu()
        torch.npu.synchronize()
        req_pool_indices_cpu = req_pool_indices_cpu.tolist()

        block_table = torch.full(size=(len(seq_lens), 2048), fill_value=-1, dtype=torch.int32, device="npu")

        for seq_idx in range(len(seq_lens)):
            req_pool_info = req_to_token_pool.get_req_pool_info(req_pool_indices_cpu[seq_idx])
            req_num_pages = (seq_lens[seq_idx] + block_size - 1) // block_size
            page_indices = req_pool_info.alloced_slots[::block_size] // block_size
            block_table[seq_idx, :req_num_pages] = page_indices[:req_num_pages]

        # cache block_table into forwardbatch
        forward_batch.block_table_cache = block_table

        return block_table

    def _init_casual_mask(self, query: torch.Tensor):
        self.casual_mask = ~torch.tril(torch.ones((1, 2048, 2048), dtype=torch.bool, device=query.device))

    def _run_sdpa_forward_extend(
        self,
        query: torch.Tensor,
        output: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        req_to_token: torch.Tensor,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        extend_prefix_lens: torch.Tensor,
        extend_seq_lens: torch.Tensor,
        scaling=None,
        enable_gqa=False,
        causal=False,
    ):
        """Run the extend forward by using torch native sdpa op.

        Args:
            query: [num_tokens, num_heads, head_size]
            output: [num_tokens, num_heads, head_size]
            k_cache: [max_total_num_tokens, num_heads, head_size]
            v_cache: [max_total_num_tokens, num_heads, head_size]
            req_to_token: [max_num_reqs, max_context_len]
            req_pool_indices: [num_seqs]
            seq_lens: [num_seqs]
            extend_prefix_lens: [num_seqs]
            extend_seq_lens: [num_seqs]
            scaling: float or None
            enable_gqa: bool
            causal: bool

        Returns:
            output: [num_tokens, num_heads, head_size]
        """

        assert seq_lens.shape[0] == extend_prefix_lens.shape[0]
        assert seq_lens.shape[0] == extend_seq_lens.shape[0]

        # [num_tokens, num_heads, head_size] -> [num_heads, num_tokens, head_size]
        query = query.movedim(0, query.dim() - 2)

        start_q, start_kv = 0, 0
        for seq_idx in range(seq_lens.shape[0]):
            # TODO: this loop process a sequence per iter, this is inefficient.
            # Need optimize the performance later.

            extend_seq_len_q = extend_seq_lens[seq_idx]
            prefill_seq_len_q = extend_prefix_lens[seq_idx]

            seq_len_kv = seq_lens[seq_idx]
            end_q = start_q + extend_seq_len_q
            end_kv = start_kv + seq_len_kv

            per_req_query = query[:, start_q:end_q, :]
            per_req_query_redudant = torch.empty(
                (per_req_query.shape[0], seq_len_kv, per_req_query.shape[2]),
                dtype=per_req_query.dtype,
                device=per_req_query.device,
            )

            per_req_query_redudant[:, prefill_seq_len_q:, :] = per_req_query

            # get key and value from cache. per_req_tokens contains the kv cache
            # index for each token in the sequence.
            req_pool_idx = req_pool_indices[seq_idx]
            per_req_tokens = req_to_token[req_pool_idx, :seq_len_kv]
            per_req_key = k_cache[per_req_tokens].movedim(0, query.dim() - 2)
            per_req_value = v_cache[per_req_tokens].movedim(0, query.dim() - 2)

            per_req_out_redudant = (
                scaled_dot_product_attention(
                    per_req_query_redudant.unsqueeze(0),
                    per_req_key.unsqueeze(0),
                    per_req_value.unsqueeze(0),
                    enable_gqa=enable_gqa,
                    scale=scaling,
                    is_causal=causal,
                )
                .squeeze(0)
                .movedim(query.dim() - 2, 0)
            )
            output[start_q:end_q, :, :] = per_req_out_redudant[prefill_seq_len_q:, :, :]
            start_q, start_kv = end_q, end_kv
        return output

    def _run_sdpa_forward_extend_npu(
        self,
        query: torch.Tensor,
        output: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        req_to_token: torch.Tensor,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        extend_prefix_lens: torch.Tensor,
        extend_seq_lens: torch.Tensor,
        scaling=None,
        enable_gqa=False,
        causal=False,
        forward_batch: ForwardBatch = None,
    ):
        """Run the extend forward by using NPU fused attention op.

        Args:
            query: [num_tokens, num_heads, head_size]
            output: [num_tokens, num_heads, head_size]
            k_cache: [max_total_num_tokens, num_heads, head_size]
            v_cache: [max_total_num_tokens, num_heads, head_size]
            req_to_token: [max_num_reqs, max_context_len]
            req_pool_indices: [num_seqs]
            seq_lens: [num_seqs]
            extend_prefix_lens: [num_seqs]
            extend_seq_lens: [num_seqs]
            scaling: float or None
            enable_gqa: bool
            causal: bool

        Returns:
            output: [num_tokens, num_heads, head_size]
        """
        if forward_batch.req_pool_indices_cpu is None:
            forward_batch.seq_lens_cpu = forward_batch.seq_lens.cpu()
            forward_batch.extend_seq_lens_cpu = forward_batch.extend_seq_lens.cpu()
            forward_batch.req_pool_indices_cpu = forward_batch.req_pool_indices.cpu()

        seq_lens_cpu_list = forward_batch.seq_lens_cpu
        extend_seq_lens_cpu_list = forward_batch.extend_seq_lens_cpu
        req_pool_indices_cpu_list = forward_batch.req_pool_indices_cpu

        batch_size = len(seq_lens_cpu_list)
        max_seq_len = max(seq_lens_cpu_list)
        max_extend_len = max(extend_seq_lens_cpu_list)

        if max_seq_len > 1 and max_extend_len == 1:
            return self._run_sdpa_forward_extend(
                query,
                output,
                k_cache,
                v_cache,
                req_to_token,
                req_pool_indices,
                seq_lens,
                extend_prefix_lens,
                extend_seq_lens,
                scaling,
                enable_gqa,
                causal,
            )

        batched_q, batched_k, batched_v = [], [], []
        start_q = 0

        # build batched query, key, value
        for seq_idx in range(batch_size):
            extend_len = extend_seq_lens_cpu_list[seq_idx]
            seq_len = seq_lens_cpu_list[seq_idx]

            # get query, key, value of current seq
            end_q = start_q + extend_len
            q_i = query[start_q:end_q, :, :]  # [extend_len, num_heads, head_dim]

            req_pool_idx = req_pool_indices_cpu_list[seq_idx]
            per_req_tokens = req_to_token[req_pool_idx, :seq_len]
            k_i = k_cache[per_req_tokens]  # [seq_len, num_heads, head_dim]
            v_i = v_cache[per_req_tokens]  # [seq_len, num_heads, head_dim]

            # padding KV by max_seq_len
            if seq_len < max_seq_len:
                pad_len = max_seq_len - seq_len
                pad_tensor_kv = torch.zeros((pad_len, k_cache.shape[1], k_cache.shape[2]), dtype=k_cache.dtype, device=k_cache.device)
                k_i = torch.cat([k_i, pad_tensor_kv], dim=0)
                v_i = torch.cat([v_i, pad_tensor_kv], dim=0)

            # padding query by max_extend_len（rather than max_seq_len）
            if extend_len < max_extend_len:
                pad_len = max_extend_len - extend_len
                pad_tensor_q = torch.zeros((pad_len, query.shape[1], query.shape[2]), dtype=query.dtype, device=query.device)
                q_i = torch.cat([q_i, pad_tensor_q], dim=0)

            batched_q.append(q_i)
            batched_k.append(k_i)
            batched_v.append(v_i)
            start_q = end_q

        # BSND
        batched_q = torch.stack(batched_q, dim=0)  # [B, S, N, D]
        batched_k = torch.stack(batched_k, dim=0)  # [B, S, N, D]
        batched_v = torch.stack(batched_v, dim=0)  # [B, S, N, D]

        # get causal attention mask
        atten_mask = None
        if causal:
            if self.casual_mask is None:
                self._init_casual_mask(query)
            atten_mask = self.casual_mask

        npu_output, _ = torch_npu.npu_fused_infer_attention_score(
            batched_q,
            batched_k,
            batched_v,
            num_heads=batched_q.shape[2],
            num_key_value_heads=batched_k.shape[2],
            input_layout="BSND",
            atten_mask=atten_mask,
            sparse_mode=3 if causal else 0,
            scale=scaling,
            actual_seq_lengths=extend_seq_lens_cpu_list,
            actual_seq_lengths_kv=seq_lens_cpu_list,
        )

        start_q = 0
        for seq_idx in range(batch_size):
            extend_len = extend_seq_lens_cpu_list[seq_idx]
            end_q = start_q + extend_len
            seq_output = npu_output[seq_idx, :extend_len, :, :]
            output[start_q:end_q, :, :] = seq_output
            start_q = end_q

        return output

    def _run_sdpa_forward_decode(
        self,
        query: torch.Tensor,
        output: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        req_to_token: torch.Tensor,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        scaling=None,
        enable_gqa=False,
        causal=False,
    ):
        """Run the decode forward by using torch native sdpa op.

        Args:
            query: [num_tokens, num_heads, head_size]
            output: [num_tokens, num_heads, head_size]
            k_cache: [max_total_num_tokens, num_heads, head_size]
            v_cache: [max_total_num_tokens, num_heads, head_size]
            req_to_token: [max_num_reqs, max_context_len]
            req_pool_indices: [num_seqs]
            seq_lens: [num_seqs]
            scaling: float or None
            enable_gqa: bool
            causal: bool

        Returns:
            output: [num_tokens, num_heads, head_size]
        """

        # [num_tokens, num_heads, head_size] -> [num_heads, num_tokens, head_size]
        query = query.movedim(0, query.dim() - 2)

        start_q, start_kv = 0, 0
        for seq_idx in range(seq_lens.shape[0]):
            # TODO: this loop process a sequence per iter, this is inefficient.
            # Need optimize the performance later.

            seq_len_q = 1
            seq_len_kv = seq_lens[seq_idx]
            end_q = start_q + seq_len_q
            end_kv = start_kv + seq_len_kv

            per_req_query = query[:, start_q:end_q, :]

            # get key and value from cache. per_req_tokens contains the kv cache
            # index for each token in the sequence.
            req_pool_idx = req_pool_indices[seq_idx]
            per_req_tokens = req_to_token[req_pool_idx, :seq_len_kv]
            per_req_key = k_cache[per_req_tokens].movedim(0, query.dim() - 2)
            per_req_value = v_cache[per_req_tokens].movedim(0, query.dim() - 2)

            per_req_out = (
                scaled_dot_product_attention(
                    per_req_query.unsqueeze(0),
                    per_req_key.unsqueeze(0),
                    per_req_value.unsqueeze(0),
                    enable_gqa=enable_gqa,
                    scale=scaling,
                    is_causal=causal,
                )
                .squeeze(0)
                .movedim(query.dim() - 2, 0)
            )
            output[start_q:end_q, :, :] = per_req_out
            start_q, start_kv = end_q, end_kv

        return output

    def _run_sdpa_forward_decode_npu(
        self,
        query: torch.Tensor,
        output: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        req_to_token: torch.Tensor,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        req_to_token_pool,
        forward_batch: ForwardBatch,
        scaling=None,
        enable_gqa=False,
        causal=False,
    ):
        """Run the decode forward by using NPU fused attention op.

        Args:
            query: [num_tokens, num_heads, head_size]
            output: [num_tokens, num_heads, head_size]
            k_cache: [max_total_num_tokens, num_heads, head_size]
            v_cache: [max_total_num_tokens, num_heads, head_size]
            req_to_token: [max_num_reqs, max_context_len]
            req_pool_indices: [num_seqs]
            seq_lens: [num_seqs]
            scaling: float or None
            enable_gqa: bool
            causal: bool

        Returns:
            output: [num_tokens, num_heads, head_size]
        """

        batch_size = seq_lens.shape[0]
        num_kv_heads = k_cache.shape[1]
        head_dim = query.shape[2]

        # BSND
        q_batch = query.view(batch_size, 1, query.shape[1], query.shape[2])

        block_size = 128
        k_cache = k_cache.view(-1, block_size, k_cache.shape[1] * k_cache.shape[2])
        v_cache = v_cache.view(-1, block_size, v_cache.shape[1] * v_cache.shape[2])

        # get block_table to enable PA in npu_fused_infer_attention_score
        block_table = self._get_or_build_block_table(forward_batch, req_to_token_pool, block_size)

        npu_output, _ = torch_npu.npu_fused_infer_attention_score(
            q_batch,
            k_cache,
            v_cache,
            dequant_scale1=None,
            dequant_scale2=None,
            block_table=block_table,
            block_size=block_size,
            num_heads=q_batch.shape[2],
            num_key_value_heads=num_kv_heads,
            input_layout="BSND",
            atten_mask=None,
            scale=scaling,
            antiquant_mode=0,
            antiquant_scale=None,
            actual_seq_lengths_kv=forward_batch.seq_lens_cpu.tolist(),
            sparse_mode=0,
        )

        output.copy_(npu_output.squeeze(2).view(-1, query.shape[1], head_dim))

        return output

    def forward_extend(
        self,
        q,
        k,
        v,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
    ):
        o = q.new_empty((q.shape[0], layer.tp_q_head_num * layer.v_head_dim)) if layer.qk_head_dim != layer.v_head_dim else torch.empty_like(q)

        if save_kv_cache:
            forward_batch.token_to_kv_pool.set_kv_buffer(layer, forward_batch.out_cache_loc, k, v)

        use_gqa = layer.tp_q_head_num != layer.tp_k_head_num

        q_ = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim)
        o_ = o.view(-1, layer.tp_q_head_num, layer.v_head_dim)

        if not __is_npu__:
            self._run_sdpa_forward_extend(
                q_,
                o_,
                forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id),
                forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id),
                forward_batch.req_to_token_pool.req_to_token,
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                forward_batch.extend_prefix_lens,
                forward_batch.extend_seq_lens,
                scaling=layer.scaling,
                enable_gqa=use_gqa,
                causal=not layer.is_cross_attention,
            )
        else:
            self._run_sdpa_forward_extend_npu(
                q_,
                o_,
                forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id),
                forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id),
                forward_batch.req_to_token_pool.req_to_token,
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                forward_batch.extend_prefix_lens,
                forward_batch.extend_seq_lens,
                scaling=layer.scaling,
                enable_gqa=use_gqa,
                causal=not layer.is_cross_attention,
                forward_batch=forward_batch,
            )
        return o

    def forward_decode(
        self,
        q,
        k,
        v,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
    ):
        # During torch.compile, there is a bug in rotary_emb that causes the
        # output value to have a 3D tensor shape. This reshapes the output correctly.
        q = q.reshape(-1, layer.tp_q_head_num * layer.qk_head_dim)

        o = q.new_empty((q.shape[0], layer.tp_q_head_num * layer.v_head_dim)) if layer.qk_head_dim != layer.v_head_dim else torch.empty_like(q)

        if save_kv_cache:
            forward_batch.token_to_kv_pool.set_kv_buffer(layer, forward_batch.out_cache_loc, k, v)

        use_gqa = layer.tp_q_head_num != layer.tp_k_head_num

        q_ = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim)
        o_ = o.view(-1, layer.tp_q_head_num, layer.v_head_dim)

        if not __is_npu__:
            self._run_sdpa_forward_decode(
                q_,
                o_,
                forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id),
                forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id),
                forward_batch.req_to_token_pool.req_to_token,
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                scaling=layer.scaling,
                enable_gqa=use_gqa,
                causal=False,
            )
        else:
            self._run_sdpa_forward_decode_npu(
                q_,
                o_,
                forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id),
                forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id),
                forward_batch.req_to_token_pool.req_to_token,
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                forward_batch.req_to_token_pool,
                forward_batch,
                scaling=layer.scaling,
                enable_gqa=use_gqa,
                causal=False,
            )
        return o
