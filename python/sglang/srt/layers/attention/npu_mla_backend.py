from __future__ import annotations

from itertools import accumulate
from typing import TYPE_CHECKING, Optional

import torch
import torch_npu
import torchair as tng
import cp_core

from sglang.srt.distributed import get_attn_tp_group
from sglang.srt.env import ENV
from sglang.srt.layers.attention.npu_attn.flash_attn import AttentionMetadata
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.distributed import (
    get_attn_tp_world_size,
    get_attn_tp_group,
)
from sglang.srt.managers.schedule_batch import global_server_args_dict
from sglang.srt.utils import get_colorful_logger
logger = get_colorful_logger(__name__)

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner

BLOCK_SIZE = 128

def all2all_wrapper(x, group = None):
    x_shape = x.shape
    x = x.reshape(-1) # for GE graph
    o_all2all = torch.empty_like(x)
    torch.distributed.all_to_all_single(o_all2all, x, group=group)
    o_all2all = o_all2all.reshape(x_shape)
    return o_all2all


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
        self._set_cos_sin_cache(
            seq_len=self.max_position_embeddings, device=device, dtype=dtype
        )

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
        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype)
        )

class NpuMLAAttnBackend(AttentionBackend):
    def __init__(self, model_runner: ModelRunner):
        super().__init__()
        self.forward_metadata = None
        self.device = model_runner.device
        self.attn_mask = ~torch.tril(
            torch.ones((2048, 2048), dtype=torch.bool, device=self.device)
        )
        if ENV.npu_enable_graph:
            torch._dynamo.mark_static(self.attn_mask)

    def init_cuda_graph_state(
        self,
        max_bs: int,
        block_kv_indices: Optional[torch.Tensor] = None,
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
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInfo],
        forward_batch: ForwardBatch,
    ):
        pass

    def get_cuda_graph_seq_len_fill_value(self):
        return 1

    def forward_sparse(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        # For multi_head latent attention
        q_rope: Optional[torch.Tensor] = None,
        k_rope: Optional[torch.Tensor] = None,
        topk_indices: torch.Tensor = None,
        absorbed: bool = False,
        layout: str = "TND",
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
            assert layer.tp_k_head_num == 1, 'num of k head must be 1 in mla absorbed mode'
            if not global_server_args_dict["npu_disable_kv_nz"]:
                k = k.squeeze(dim=2)  # head num must be 1
                k_rope = k_rope.squeeze(dim=2)  # head num must be 1
                k = k.view(k.shape[0], 1, k.shape[2] // 16, k.shape[1], 16).transpose(1, 3).reshape(-1, k.shape[2]).reshape(-1,block_size,1,k.shape[2])
                k_rope = k_rope.view(k_rope.shape[0], 1, k_rope.shape[2] // 16, k_rope.shape[1], 16).transpose(1, 3).reshape(-1, k_rope.shape[2]).reshape(-1,block_size,1,k_rope.shape[2])

        q_nope, q_pe = q, q_rope
        k_nope, k_pe = k, k_rope

        if is_prefill:
            if  forward_batch.attn_metadata.actual_seq_lengths is not None:
                actual_seq_qlen = torch.tensor(forward_batch.attn_metadata.actual_seq_lengths).to(q.device).to(torch.int32)
            else:
                actual_seq_qlen = torch.cumsum(forward_batch.seq_lens, dim=0)
        else:
            if forward_batch.attn_metadata.query_len_tensor is None:
                if (forward_batch.forward_mode.is_target_verify()):
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
                    actual_seq_qlen = (
                        forward_batch.extend_seq_lens.cumsum()
                        .to(q.device)
                        .to(torch.int32)
                    )
                else:
                    actual_seq_qlen = (
                        torch.arange(1, q.shape[0] + 1).to(q.device).to(torch.int32)
                    )
            else:
                actual_seq_qlen = forward_batch.attn_metadata.query_len_tensor.to(torch.int32)

        if is_prefill:
            actual_seq_lengths_kv = forward_batch.attn_metadata.seq_lens_tensor.to(q.device).to(torch.int32)
            if "TND" in layout:
                # sfa is not support block table with TND layout.
                block_table = None
                layout_kv = "TND"
                actual_seq_lengths_kv = torch.tensor(forward_batch.attn_metadata.actual_seq_lengths_kv).to(q.device).to(torch.int32)
        else:
            actual_seq_lengths_kv = forward_batch.attn_metadata.seq_lens_tensor.to(torch.int32)
            block_table = forward_batch.attn_metadata.block_table
            layout_kv = "PA_BSND"

        use_kvp = (global_server_args_dict["kvp_size"] > 1)
        torchair_enable = forward_batch.all_decode_or_idle and ENV.npu_enable_graph
        if use_kvp:
            block_table = forward_batch.attn_metadata.kvp_block_table
            actual_seq_lengths_kv = forward_batch.attn_metadata.kvp_context_lens_tensor.to(torch.int32)
            kvp_group = get_attn_tp_group()
            kvp_rank = kvp_group.rank_in_group
            kvp_size = global_server_args_dict["kvp_size"]
            kv_lora_rank = q.shape[-1]
            qk_rope_head_dim = q_pe.shape[-1]
            q_total = torch.cat([q, q_pe], dim = -1)
            q_total = kvp_group.all_gather(q_total, dim=1)
            q, q_pe = torch.split(q_total, [kv_lora_rank, qk_rope_head_dim], dim=2)
            T = q.shape[0]
            N = q.shape[1]
            sparse_mode = 0
            if kvp_rank == 0:
                sparse_mode = 3
            o_ori, max_o, sum_o = torch_npu.npu_sparse_flash_attention(
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
            #N,T,1->(128,T,1)
            lse = max_o + torch.log(sum_o)
            lse = lse.permute(1, 0, 2).reshape(T, -1).unsqueeze(-1)
            lse = lse.masked_fill((lse != lse) | torch.isinf(lse) | (lse < -1e9), -1e9)
            # TODO: use two stream to process o and lse in parallel
            # o: (T,N,D)
            o = o_ori.transpose(0, 1).contiguous()#NTD
            o = o.reshape(kvp_size, N//kvp_size, T, -1) # (4, 32, 10, 512)
            o_all2all = all2all_wrapper(o, group=kvp_group.device_group)
            o_all2all = o_all2all.reshape(N, T, -1)
            # lse: (T, N, 1)
            lse = lse.transpose(0, 1).reshape(kvp_size, N//kvp_size, T).contiguous()
            lse_all2all = all2all_wrapper(lse, group=kvp_group.device_group)
            lse_all2all = lse_all2all.reshape(N, T, -1)

            o_all2all = o_all2all.reshape(kvp_size, (N//kvp_size)*T, -1)
            lse_all2all = lse_all2all.reshape(kvp_size, (N//kvp_size)*T)

            o_all2all_list = torch.chunk(o_all2all, kvp_size, dim=0)
            o_all2all_list = [t.squeeze(0) for t in o_all2all_list]
            lse_all2all_list = torch.chunk(lse_all2all, kvp_size, dim=0)
            lse_all2all_list = [t.squeeze(0) for t in lse_all2all_list]

            o, _ = torch_npu.npu_attention_update(lse_all2all_list, o_all2all_list, update_type=0)

            attn_out = o.reshape(N//kvp_size, T, -1)
            # TODO: Following code can fix accurary bug, however it will bring many TensorMove operators in NPU graph.
            # Don't open this pass before HW fix the issue.
            if global_server_args_dict["npu_kvp_accuracy_fix"] and kvp_rank == 0:
                assert global_server_args_dict["npu_disable_kv_nz"]
                k_nope = k_nope.view(-1, kv_lora_rank)
                k_pe = k_pe.view(-1, qk_rope_head_dim)
                current_k = k_nope[forward_batch.attn_metadata.slot_mapping]
                current_k_pe = k_pe[forward_batch.attn_metadata.slot_mapping]
                if torchair_enable:
                    tng.scope.npu_wait_tensor(k_nope, o_ori)
                    tng.scope.npu_wait_tensor(k_pe, o_ori)
                torch_npu.npu_scatter_nd_update_(k_nope, forward_batch.attn_metadata.kvp_current_slots_mapping.reshape(-1, 1), current_k)
                torch_npu.npu_scatter_nd_update_(k_pe, forward_batch.attn_metadata.kvp_current_slots_mapping.reshape(-1, 1), current_k_pe)
        else:
            enable_sp_for_indexer = is_prefill and global_server_args_dict.get("enable_mla_l1_5_cache", False) and global_server_args_dict["npu_enable_sp_for_indexer"]
            if not enable_sp_for_indexer:
                attn_out, _, _ = torch_npu.npu_sparse_flash_attention(
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
                        slice_end = forward_batch.cu_attn_sp_token_nums_cpu[cp_rank+1].item()
                        q_nope = q_nope[slice_start:slice_end]
                        q_pe = q_pe[slice_start:slice_end]

                    from sglang.srt.layers.attention.dsa.nsa_indexer import get_non_zero
                    k_nope, _, _ = get_non_zero(k_nope, forward_batch.global_cu_kv_lens_cpu, forward_batch.local_extend_lens_with_zero, forward_batch.local_total_kv_lens_with_zero)
                    k_pe, _, _ = get_non_zero(k_pe, forward_batch.global_cu_kv_lens_cpu, forward_batch.local_extend_lens_with_zero, forward_batch.local_total_kv_lens_with_zero)
                    actual_seq_qlen = forward_batch.cur_seq_lengths_query[1:]
                    actual_seq_lengths_kv = forward_batch.cur_seq_lengths_key[1:]

                    attn_out, _, _ = torch_npu.npu_sparse_flash_attention(
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
                    attn_out = torch.empty((0, global_q_head, k_nope.shape[2]), dtype = q_nope.dtype, device = q_nope.device)
                if not global_server_args_dict["npu_disable_dsa_head_parallel"]:
                    attn_sp_token_nums = get_attn_tp_group().get_local_sp_token_num(forward_batch.global_sp_num_tokens)
                    attn_out = get_attn_tp_group().all_gather(attn_out, dim=0, output_split_sizes=attn_sp_token_nums)
                    head_start = cp_rank * (global_q_head // cp_size)
                    head_end = (cp_rank + 1) * (global_q_head // cp_size)
                    attn_out = attn_out[:,head_start:head_end,:]
            if layout == 'TND_NTD':
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
        q_pe: Optional[torch.Tensor] = None,
        k_pe: Optional[torch.Tensor] = None,
        topk_indices: Optional[torch.Tensor] = None,
        layout='TND',
        absorbed=False
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
        q_pe: Optional[torch.Tensor] = None,
        k_pe: Optional[torch.Tensor] = None,
        layout='TND',
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
                next_tokens=0)[0]
        return o

    def forward_decode(
        self,
        q,
        k,
        v,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=False,
        q_pe: Optional[torch.Tensor] = None,
        k_pe: Optional[torch.Tensor] = None,
        topk_indices: Optional[torch.Tensor] = None,
        layout='TND',
        absorbed=True
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
            )
        assert absorbed, 'decode only support absorbed mode'
        return self.forward_absorbed(q, k, v, layer, forward_batch, save_kv_cache, q_pe, k_pe, layout)

    def forward_absorbed(
        self,
        q,
        k,
        v,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=False,
        q_pe: Optional[torch.Tensor] = None,
        k_pe: Optional[torch.Tensor] = None,
        layout='TND', # layout指输入数据的内存排布，比如BSND意思是(batch, seq_len, head_num, head_dim)。TND中的T是指BS合并为T，同时不带padding, ND的含义不变。
    ):
        save_kv_cache = False  # kv cache saved in mla prolog on npu
        if save_kv_cache:
            forward_batch.token_to_kv_pool.set_kv_buffer(layer, forward_batch.out_cache_loc, k, v)
        block_size = forward_batch.token_to_kv_pool.page_size
        assert layer.tp_k_head_num == 1, 'num of k head must be 1 in mla absorbed mode'
        k_head_dim = layer.v_head_dim if k_pe is not None else layer.qk_head_dim
        k = k.view(-1, block_size, k_head_dim)
        k_pe = k_pe.squeeze(dim=2)  # head num must be 1
        if 'TND' in layout:
            q = q.squeeze(dim=1)
            q_pe = q_pe.squeeze(dim=1)

        use_kvp = (global_server_args_dict["kvp_size"] > 1)
        torchair_enable = forward_batch.all_decode_or_idle and ENV.npu_enable_graph
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
            kvp_group = get_attn_tp_group()
            kvp_rank = kvp_group.rank_in_group
            kvp_size = global_server_args_dict["kvp_size"]
            kv_lora_rank = q.shape[-1]
            qk_rope_head_dim = q_pe.shape[-1]
            q_total = torch.cat([q, q_pe], dim = -1)
            q_total = kvp_group.all_gather(q_total, dim=1)
            q, q_pe = torch.split(q_total, [kv_lora_rank, qk_rope_head_dim], dim=2)
            T = q.shape[0]
            N = q.shape[1]
            sparse_mode = 0
            atten_mask = None
            if kvp_rank == 0:
                sparse_mode = 3
                atten_mask=self.attn_mask
            o_ori, lse = op_scope.npu_fused_infer_attention_score(
                    q, k, k, query_rope=q_pe, key_rope=k_pe,
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
                    actual_seq_lengths_kv= actual_seq_lengths_kv,
                    actual_seq_lengths=actual_seq_lengths,
                    softmax_lse_flag = True,
                    )
            lse = lse.masked_fill(torch.isinf(lse), -1e9)
            # TODO: use two stream to process o and lse in parallel
            # o: (N,T,D)
            o = o_ori.reshape(kvp_size, N//kvp_size, T, -1)
            o_all2all = all2all_wrapper(o, group=kvp_group.device_group)
            o_all2all = o_all2all.reshape(N, T, -1)
            # lse: (T, N, 1)
            lse = lse.transpose(0, 1).reshape(kvp_size, N//kvp_size, T).contiguous()
            lse_all2all = all2all_wrapper(lse, group=kvp_group.device_group)
            lse_all2all = lse_all2all.reshape(N, T, -1)

            o_all2all = o_all2all.reshape(kvp_size, (N//kvp_size)*T, -1)
            lse_all2all = lse_all2all.reshape(kvp_size, (N//kvp_size)*T)

            o_all2all_list = torch.chunk(o_all2all, kvp_size, dim=0)
            o_all2all_list = [t.squeeze(0) for t in o_all2all_list]
            lse_all2all_list = torch.chunk(lse_all2all, kvp_size, dim=0)
            lse_all2all_list = [t.squeeze(0) for t in lse_all2all_list]

            o, _ = torch_npu.npu_attention_update(lse_all2all_list, o_all2all_list, update_type=0)

            o = o.reshape(N//kvp_size, T, -1)
            # TODO: Following code can fix accurary bug, however it will bring many TensorMove operators in NPU graph.
            # Don't open this pass before HW fix the issue.
            if global_server_args_dict["npu_kvp_accuracy_fix"] and kvp_rank == 0:
                assert global_server_args_dict["npu_disable_kv_nz"]
                k = k.view(-1, kv_lora_rank)
                k_pe = k_pe.view(-1, qk_rope_head_dim)
                current_k = k[forward_batch.attn_metadata.slot_mapping]
                current_k_pe = k_pe[forward_batch.attn_metadata.slot_mapping]
                if torchair_enable:
                    tng.scope.npu_wait_tensor(k, o_ori)
                    tng.scope.npu_wait_tensor(k_pe, o_ori)
                torch_npu.npu_scatter_nd_update_(k, forward_batch.attn_metadata.kvp_current_slots_mapping.reshape(-1, 1), current_k)
                torch_npu.npu_scatter_nd_update_(k_pe, forward_batch.attn_metadata.kvp_current_slots_mapping.reshape(-1, 1), current_k_pe)

        else:
            o, _ = op_scope.npu_fused_infer_attention_score(
                    q, k, k, query_rope=q_pe, key_rope=k_pe,
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


def get_or_build_block_table(forward_batch: ForwardBatch, block_size: int, capture_graph: Optional[bool] = False) -> torch.Tensor:

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
        block_table = torch.full(
            size=(len(seq_lens), 2048),
            fill_value=-1,
            dtype=torch.int32,
            device="npu")

        for seq_idx in range(len(seq_lens)):
            req_pool_info = forward_batch.req_to_token_pool.get_req_pool_info(req_pool_indices_cpu[seq_idx])
            req_num_pages = (seq_lens[seq_idx] + block_size - 1) // block_size
            page_indices = req_pool_info.alloced_slots[::block_size] // block_size
            if page_indices.numel() == 0:
                page_indices = torch.zeros((1,), dtype=torch.int32, device='npu')
            if req_num_pages > len(page_indices[:req_num_pages]):
                tmp = forward_batch.req_to_token_pool
                rid = req_pool_indices_cpu[seq_idx]
                logger.info(f'page_indices err, {seq_lens=}, {req_pool_info.alloced_slots.shape=}, {req_pool_info.alloced_slots=}, {tmp.verified_lens[rid]=}, {tmp.alloced_lens[rid]=}, {tmp.req_to_token[rid]=}')
                req_num_pages = len(page_indices[:req_num_pages])
            block_table[seq_idx, :req_num_pages] = page_indices[:req_num_pages]

    return block_table

def get_local_kv_state(
    global_block_table: torch.Tensor,  # (B, PS), device
    global_kv_lens: torch.Tensor,      # (B,),    device, 写入前的长度
    reserved_block_num: int,
    page_size: int,
    kvp_size: int,
    current_rank: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    B, PS  = global_block_table.shape
    device = global_block_table.device

    # 写入前/后的长度
    pre_seq_lens  = global_kv_lens.long()                              # (B,) 写入前

    # ------------------------------------------------------------------ #
    # 1. local_block_table
    # ------------------------------------------------------------------ #
    is_reserved     = global_block_table < reserved_block_num          # (B, PS)
    adjusted        = global_block_table - reserved_block_num          # (B, PS)
    belongs_to_rank = ((adjusted % kvp_size) == current_rank)          # (B, PS)
    local_idx       = adjusted // kvp_size + reserved_block_num        # (B, PS)
    is_padding      = (global_block_table == 0)                        # (B, PS)

    # 每个位置映射后的值（不考虑是否属于本rank）
    mapped = torch.where(
        is_padding,
        torch.zeros_like(global_block_table),
        torch.where(
            is_reserved,
            global_block_table,                                        # 保留块不变
            torch.where(
                belongs_to_rank,
                local_idx,                                             # 本rank映射
                torch.zeros_like(global_block_table)                   # 其他rank置0
            )
        )
    )
    # 有效位置：保留块 或 属于本rank的非padding块
    valid = (is_reserved & ~is_padding) | (belongs_to_rank & ~is_padding)  # (B, PS)

    # 利用 sort trick：对 ~valid 升序排序，False(有效)排前，True(无效)排后
    # stable=True 保证有效位置内部的相对顺序不变
    order = (~valid).long()                                            # (B, PS) 0=有效,1=无效
    perm  = order.argsort(dim=1, stable=True)                         # (B, PS)

    local_block_table = mapped.gather(1, perm)                        # (B, PS)

    # ------------------------------------------------------------------ #
    # 2. local_kv_lens，基于写入前的长度
    # ------------------------------------------------------------------ #
    pre_total_blocks = (pre_seq_lens + page_size - 1) // page_size     # (B,)

    col_idx    = torch.arange(PS, device=device).unsqueeze(0)          # (1, PS)
    valid_mask = col_idx < pre_total_blocks.unsqueeze(1)               # (B, PS)
    rank_mask  = belongs_to_rank & valid_mask & (~is_reserved)         # (B, PS)

    # 最后一个有效 block 是否属于当前 rank
    last_col         = (pre_total_blocks - 1).clamp(0, PS - 1)         # (B,)
    last_block_global = global_block_table.gather(
        1, last_col.unsqueeze(1)
    ).squeeze(1)                                                        # (B,)
    last_adjusted    = last_block_global - reserved_block_num          # (B,)
    last_belongs     = (
        (last_adjusted % kvp_size == current_rank)
        & (last_block_global >= reserved_block_num)
    )                                                                   # (B,)

    local_block_count = rank_mask.long().sum(dim=1)                    # (B,)

    remainder   = pre_seq_lens % page_size                             # (B,)
    last_tokens = torch.where(
        remainder == 0,
        torch.full_like(remainder, page_size),
        remainder
    )                                                                   # (B,)

    local_kv_lens = torch.where(
        last_belongs & (local_block_count > 0),
        (local_block_count - 1) * page_size + last_tokens,
        local_block_count * page_size
    )                                                                   # (B,) 写入前

    return local_block_table, local_kv_lens

def build_req_level_owner_stats(
    forward_batch: ForwardBatch,
    global_hist_lens: torch.Tensor,   # [B], global history length before current decode tokens
    local_req_lens: torch.Tensor,     # [B], local kv length on this rank (rank0 already includes current)
    q_lens: torch.Tensor,             # [B], query length on this rank
    num_init: int,
    num_local: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
        init_cnt_req:  [B], number of init tokens owned by this rank.
        local_cnt_req: [B], number of historical local-window tokens owned by this rank.
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
        pages = torch.div(token_indices, page_size, rounding_mode='trunc')
        return (pages < reserved_block_num) | (
            ((pages - reserved_block_num) % group_size) == kv_pool.kvp_rank
        )

    init_prefix_len = torch.minimum(
        global_hist_lens,
        torch.full_like(global_hist_lens, num_init),
    )

    if num_init > 0:
        init_pos = torch.arange(num_init, device=device, dtype=torch.int64)[None, :]
        init_tokens = _gather_req_tokens(init_pos.expand(B, -1))
        init_valid = init_pos < init_prefix_len[:, None]
        init_cnt_req = (_is_local_token(init_tokens) & init_valid).sum(
            dim=1, dtype=torch.int32
        )
    else:
        init_cnt_req = torch.zeros(B, device=device, dtype=torch.int32)

    if num_local > 0:
        local_offsets = torch.arange(num_local, device=device, dtype=torch.int64)[None, :]
        local_start = torch.minimum(
            global_hist_lens,
            torch.maximum(
                init_prefix_len,
                torch.clamp(global_hist_lens + 1 - num_local, min=0),
            )
        )
        local_pos = local_start[:, None] + local_offsets
        local_tokens = _gather_req_tokens(local_pos)
        local_valid = local_pos < global_hist_lens[:, None]
        local_cnt_req = (_is_local_token(local_tokens) & local_valid).sum(
            dim=1, dtype=torch.int32
        )
    else:
        local_cnt_req = torch.zeros(B, device=device, dtype=torch.int32)

    return init_cnt_req, local_cnt_req

def get_attn_meta_npu(forward_batch: ForwardBatch, model_runner: ModelRunner, capture_graph: Optional[bool] = False):
    attn_metadata =  AttentionMetadata()
    attn_metadata.is_profile_run = False
    attn_metadata.context_chunk_workspace  = None
    attn_metadata.seq_lens_tensor = forward_batch.seq_lens.to(torch.int64)
    attn_metadata.slot_mapping = forward_batch.out_cache_loc.clone().to(torch.int64)
    attn_metadata.all_decode_or_idle = forward_batch.all_decode_or_idle
    decode_seq_per_batch = 1

    if forward_batch.forward_mode.is_extend():
        if forward_batch.forward_mode.is_target_verify() or forward_batch.forward_mode.is_draft_extend():
            attn_metadata.num_prefill_tokens = forward_batch.new_tokens_total
            attn_metadata.query_len_tensor = torch.tensor(
                list(accumulate([(forward_batch.spec_num_steps + 1) for _ in range(forward_batch.batch_size)])), dtype=torch.int64, pin_memory=True).to(device='npu', non_blocking=True)
            attn_metadata.context_lens_tensor = attn_metadata.seq_lens_tensor
            attn_metadata.seq_lens_tensor = attn_metadata.seq_lens_tensor + forward_batch.spec_num_steps + 1
            decode_seq_per_batch = forward_batch.spec_num_steps + 1
        else:
            attn_metadata.seq_lens = attn_metadata.seq_lens_tensor.cpu()
            attn_metadata.num_prefill_tokens = forward_batch.extend_num_tokens
            attn_metadata.context_lens_tensor = forward_batch.extend_prefix_lens
            attn_metadata.context_lens = forward_batch.extend_prefix_lens_cpu
            attn_metadata.query_len_tensor = forward_batch.extend_seq_lens.to(dtype=torch.int64)
            attn_metadata.actual_seq_lengths=list(accumulate(forward_batch.extend_seq_lens_cpu))
            seq_lens=[x+y for x, y in
                      zip(forward_batch.extend_prefix_lens_cpu, forward_batch.extend_seq_lens_cpu)]
            attn_metadata.actual_seq_lengths_kv=list(accumulate(seq_lens))
            attn_metadata.query_len_tensor=torch.tensor(attn_metadata.actual_seq_lengths, dtype=torch.int64,
                                                        pin_memory=True).to(device='npu', non_blocking=True)
            attn_metadata.seq_lens_tensor=torch.tensor(seq_lens, dtype=torch.int64, pin_memory=True).to(device='npu',
                                                                                                        non_blocking=True)
            kv_index_list=[]
            enable_mla_l1_5_cache = global_server_args_dict.get("enable_mla_l1_5_cache", False)
            if not global_server_args_dict["npu_disable_kv_nz"]:
                if not enable_mla_l1_5_cache:
                    kv_block_index_list=[]
                    bias=0
                    for seq_idx, seq_len in enumerate(seq_lens):
                        alloced_slots=forward_batch.req_to_token_pool.req_to_token[
                            forward_batch.req_pool_indices[seq_idx]]
                        kv_block_index_list.append(alloced_slots[:seq_len:128]//128)
                        kv_index_list.append(torch.arange(seq_len, dtype=torch.int32, device=alloced_slots.device)+bias)
                        bias=bias+(seq_len+127)//128*128
                    attn_metadata.kv_index_list=torch.cat(kv_index_list, dim=0)
                    attn_metadata.kv_block_index_list=torch.cat(kv_block_index_list, dim=0)

                else:
                    block_size = forward_batch.token_to_kv_pool.page_size
                    # 1. Keep the original global token order for all-gather restore metadata.
                    # 2. Reuse the same mapping result to rebuild mini pool indices per sequence
                    #    using each sequence's local token count.
                    kv_slots = torch.cat([
                            forward_batch.req_to_token_pool.req_to_token[forward_batch.req_pool_indices[seq_idx]][:seq_len]
                            for seq_idx, seq_len in enumerate(seq_lens)
                        ],
                        dim=0
                    )
                    mask, local_indices, per_rank_count, _, inv_perm = (
                        forward_batch.token_to_kv_pool.global_loc_to_local_mapping(
                            kv_slots,
                        )
                    )
                    attn_metadata.per_rank_count = per_rank_count
                    attn_metadata.inv_perm = inv_perm
                    kv_block_index_list = []
                    kv_index_list = []
                    bias = 0
                    global_offset = 0
                    local_offset = 0
                    for seq_len in seq_lens:
                        seq_mask = mask[global_offset:global_offset + seq_len]
                        local_seq_len = int(seq_mask.sum().item())
                        seq_local_indices = local_indices[local_offset:local_offset + local_seq_len]
                        local_seq_len = seq_local_indices.numel()
                        if local_seq_len > 0:
                            kv_block_index_list.append(seq_local_indices[::block_size] // block_size)
                            kv_index_list.append(
                                torch.arange(
                                    local_seq_len,
                                    dtype=torch.int32,
                                    device=kv_slots.device,
                                ) + bias
                            )
                        bias = bias + (local_seq_len + block_size - 1) // block_size * block_size
                        global_offset += seq_len
                        local_offset += local_seq_len
                    attn_metadata.kv_block_index_list = (
                        torch.cat(kv_block_index_list, dim=0)
                        if kv_block_index_list
                        else torch.empty((0,), dtype=torch.int32, device=kv_slots.device)
                    )
                    attn_metadata.kv_index_list = (
                        torch.cat(kv_index_list, dim=0)
                        if kv_index_list
                        else torch.empty((0,), dtype=torch.int32, device=kv_slots.device)
                    )
            else:
                if not enable_mla_l1_5_cache:
                    for seq_idx, seq_len in enumerate(seq_lens):
                        alloced_slots=forward_batch.req_to_token_pool.req_to_token[
                            forward_batch.req_pool_indices[seq_idx]]
                        kv_index_list.append(alloced_slots[:seq_len])
                    attn_metadata.kv_index_list=torch.cat(kv_index_list, dim=0)
                else:
                    kv_slots = torch.cat([
                            forward_batch.req_to_token_pool.req_to_token[forward_batch.req_pool_indices[seq_idx]][:seq_len]
                            for seq_idx, seq_len in enumerate(seq_lens)
                        ],
                        dim=0
                    )
                    mask, local_indices, per_rank_count, _, inv_perm = (
                        forward_batch.token_to_kv_pool.global_loc_to_local_mapping(
                            kv_slots,
                        )
                    )
                    attn_metadata.per_rank_count = per_rank_count
                    attn_metadata.inv_perm = inv_perm
                    attn_metadata.kv_index_list = local_indices
            if isinstance(attn_metadata.seq_lens, torch.Tensor):
                attn_metadata.seq_lens = attn_metadata.seq_lens.tolist()
            if isinstance(attn_metadata.context_lens, torch.Tensor):
                attn_metadata.context_lens = attn_metadata.context_lens.tolist()
                attn_metadata.query_len = [a - b for a, b in zip(attn_metadata.seq_lens, attn_metadata.context_lens)]
                attn_metadata.max_query_len = max(attn_metadata.query_len)
                attn_metadata.max_seq_len = max(attn_metadata.seq_lens)

    elif forward_batch.forward_mode.is_decode():
        attn_metadata.num_prefill_tokens = 0
        attn_metadata.context_lens_tensor = attn_metadata.seq_lens_tensor - 1
        attn_metadata.query_len_tensor = torch.tensor(
                list(accumulate([1 for _ in range(forward_batch.batch_size)])), dtype=torch.int64, pin_memory=True).to(device='npu', non_blocking=True)
    else:
        raise ValueError(f"Unsupported forward_mode:{forward_batch.forward_mode}")
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
        attn_metadata.block_table = torch.full(
            size=(len(forward_batch.seq_lens), block_table_size),
            fill_value=-1,
            dtype=torch.int32,
            device="npu")
        block_len = min(model_runner.kv_allocator.req_to_page.size(-1), attn_metadata.block_table.size(-1))
        attn_metadata.block_table[:, :block_len] = model_runner.kv_allocator.req_to_page[forward_batch.req_pool_indices, :block_len]
        attn_metadata.num_decode_tokens = forward_batch.new_tokens_total - attn_metadata.num_prefill_tokens
    kvp_size = global_server_args_dict["kvp_size"]


    if kvp_size > 1:
        from sglang.srt.distributed import get_attn_tp_group
        kvp_rank = get_attn_tp_group().rank_in_group
        global_block_table = attn_metadata.block_table
        global_kv_lens = attn_metadata.context_lens_tensor
        reserved_block_num = model_runner.kv_allocator.max_batch_size + 1

        kvp_block_table, kvp_context_lens_tensor = get_local_kv_state(global_block_table,
                                                                      global_kv_lens,
                                                                      reserved_block_num,
                                                                      page_size,
                                                                      kvp_size,
                                                                      kvp_rank)
        attn_metadata.kvp_current_slots_mapping = attn_metadata.slot_mapping
        if kvp_rank == 0:
            batch_size = kvp_block_table.shape[0]
            device = kvp_block_table.device
            actual_block_num_tensor = (
                (kvp_context_lens_tensor + page_size - 1) // page_size
            ).to(torch.int32)                                     # [batch_size]

            #assert (actual_block_num_tensor + 1).max().item() <= block_table_size

            bs_idx_tensor = torch.arange(
                batch_size, dtype=torch.int32, device=device
            )                                                     # [batch_size]

            pad_block_idx_tensor = torch.arange(
                1, batch_size + 1, dtype=torch.int32, device=device
            )                                                     # [batch_size]

            kvp_block_table[bs_idx_tensor, actual_block_num_tensor] = pad_block_idx_tensor

            kv_len_expanded = kvp_context_lens_tensor.unsqueeze(1).expand(
                batch_size, decode_seq_per_batch
            )

            si = torch.arange(
                decode_seq_per_batch, dtype=kvp_context_lens_tensor.dtype, device=device
            ).unsqueeze(0)                                        # [1, decode_seq_per_batch]

            pos = kv_len_expanded + si
            kvp_block_talbe_idx_tensor = (pos // page_size).to(torch.int32)
            kvp_slot_mapping_idx_in_page = (pos % page_size).to(torch.int32)
            kvp_slot_mapping_page_idx = kvp_block_table.gather(
                1, kvp_block_talbe_idx_tensor
            )                                                     # [batch_size, decode_seq_per_batch]

            kvp_current_slots_mapping = (
                kvp_slot_mapping_page_idx * page_size + kvp_slot_mapping_idx_in_page
            ).flatten().to(torch.int64)                           # [batch_size * decode_seq_per_batch]

            kvp_context_lens_tensor = kvp_context_lens_tensor + decode_seq_per_batch
            attn_metadata.slot_mapping = kvp_current_slots_mapping

        attn_metadata.kvp_block_table = kvp_block_table
        attn_metadata.kvp_context_lens_tensor = kvp_context_lens_tensor

        # add for 3s indexer
        config = model_runner.model_config.hf_config
        num_init_tokens = getattr(config, "index_init_tokens", 0)
        num_local_tokens = getattr(config, "index_local_tokens", 0)
        device = attn_metadata.query_len_tensor.device
        cur_seq_lengths_query = torch.concat((torch.tensor([0], device=device, dtype=torch.int64), attn_metadata.query_len_tensor))
        q_lens = cur_seq_lengths_query[1:] - cur_seq_lengths_query[:-1]
        global_hist_lens = attn_metadata.seq_lens_tensor.to(device) - q_lens
        init_cnt_req, local_cnt_req = build_req_level_owner_stats(
                  forward_batch, global_hist_lens, kvp_context_lens_tensor, q_lens, num_init_tokens, num_local_tokens)
        attn_metadata.init_cnt_req = init_cnt_req
        attn_metadata.local_cnt_req = local_cnt_req

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

    def init_forward_metadata_replay_cuda_graph(
        self, forward_batch: ForwardBatch, bs: int
    ):
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
        self.attn_mask = ~torch.tril(
            torch.ones((2048, 2048), dtype=torch.bool, device=self.device)
        )

    def init_cuda_graph_state(
        self,
        max_bs: int,
        block_kv_indices: Optional[torch.Tensor] = None,
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
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInfo],
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

        # kv shape: [blockNum, blockSize, N,D]
        k_cache = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)
        v_cache = forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id)

        # View q k v to TND.
        q = q.view(-1, layer.tp_q_head_num, layer.head_dim)
        k_cache = k_cache.view(-1, layer.tp_k_head_num, layer.head_dim // 16, self.page_size, 16)
        v_cache = v_cache.view(-1, layer.tp_k_head_num, layer.head_dim // 16, self.page_size, 16)

        # todo kv_nz: [blockNum, KV_N, D/16, blockSize, 16]
        # if not global_server_args_dict["npu_disable_kv_nz"]:
        #     k_cache = k_cache.view(k_cache.shape[0], k_cache.shape[1], k_cache.shape[3] // 16, k_cache.shape[2], 16)
        #     v_cache = v_cache.view(v_cache.shape[0], v_cache.shape[1], v_cache.shape[3] // 16, v_cache.shape[2], 16)
        decode_meta = forward_batch.attn_metadata
        assert decode_meta is not None
        assert decode_meta.seq_lens_tensor.dtype is torch.int64
        assert decode_meta.query_len_tensor.dtype is torch.int64
        torchair_enable = decode_meta.all_decode_or_idle and ENV.npu_enable_graph
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
        o, _ = op_scope.npu_fused_infer_attention_score(
            q, k_cache, v_cache,
            num_heads=layer.tp_q_head_num,
            num_key_value_heads=layer.tp_k_head_num,
            input_layout="TND",
            atten_mask=self.attn_mask,
            sparse_mode=3,
            scale=layer.scaling,
            block_table=forward_batch.attn_metadata.block_table,
            block_size=self.page_size,
            **kwargs)
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

        torchair_enable = decode_meta.all_decode_or_idle and ENV.npu_enable_graph
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

        o, _ = op_scope.npu_fused_infer_attention_score(
            q, k_cache, v_cache,
            num_heads=layer.tp_q_head_num,
            num_key_value_heads=layer.tp_k_head_num,
            input_layout="TND",
            atten_mask=self.attn_mask,
            sparse_mode=3,
            scale=layer.scaling,
            block_table=forward_batch.attn_metadata.block_table,
            block_size=self.page_size,
            **kwargs)

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

    def init_forward_metadata_replay_cuda_graph(
        self, forward_batch: ForwardBatch, bs: int
    ):
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
