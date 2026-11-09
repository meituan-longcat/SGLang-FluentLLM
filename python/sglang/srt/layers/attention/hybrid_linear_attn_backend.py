import logging
import os
from dataclasses import dataclass
from typing import Optional, Union

import torch
import torch.nn.functional as F
from einops import rearrange

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.fla.chunk import chunk_gated_delta_rule
from sglang.srt.layers.attention.fla.fused_recurrent import (
    fused_recurrent_gated_delta_rule_update,
)
from sglang.srt.layers.attention.fla.fused_sigmoid_gating_recurrent import (
    fused_sigmoid_gating_delta_rule_update,
)
from sglang.srt.layers.attention.fla.kda import chunk_kda
from sglang.srt.layers.attention.mamba.causal_conv1d_triton import (
    causal_conv1d_fn,
    causal_conv1d_update,
)
from sglang.srt.layers.attention.radix_attention import RadixAttention
from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.models.qwen3_next import fused_gdn_gating
from sglang.srt.speculative.eagle_utils import EagleDraftInput, EagleVerifyInput

logger = logging.getLogger(__name__)

try:
    import flash_kda

    _FLASH_KDA_AVAILABLE = True
except ImportError:
    _FLASH_KDA_AVAILABLE = False

_FLASH_KDA_ENABLED = _FLASH_KDA_AVAILABLE and os.getenv("FLA_FLASH_KDA", "1") != "0"


@dataclass
class ForwardMetadata:
    query_start_loc: Optional[torch.Tensor]
    mamba_cache_indices: torch.Tensor


class MambaAttnBackend(AttentionBackend):
    """Attention backend using Mamba kernel."""

    def __init__(self, model_runner: ModelRunner):
        super().__init__()
        self.pad_slot_id = -1  # Default pad slot id
        self.device = model_runner.device
        self.speculative_num_draft_tokens = model_runner.server_args.speculative_num_draft_tokens
        self.req_to_token_pool: HybridReqToTokenPool = model_runner.req_to_token_pool
        self.forward_metadata: ForwardMetadata = None
        self.state_indices_list = []
        self.query_start_loc_list = []
        self.cached_cuda_graph_decode_query_start_loc: torch.Tensor = None
        self.cached_cuda_graph_verify_query_start_loc: torch.Tensor = None

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        bs = forward_batch.batch_size
        if forward_batch.forward_mode.is_decode_or_idle():
            query_start_loc = torch.arange(0, bs + 1, dtype=torch.int32, device=self.device)
        elif forward_batch.forward_mode.is_extend():
            if forward_batch.forward_mode.is_target_verify():
                query_start_loc = torch.arange(
                    0,
                    forward_batch.input_ids.shape[0] + 1,
                    step=forward_batch.spec_info.draft_token_num,
                    dtype=torch.int32,
                    device=forward_batch.input_ids.device,
                )
            else:
                query_start_loc = torch.empty((bs + 1,), dtype=torch.int32, device=self.device)
                query_start_loc[:bs] = forward_batch.extend_start_loc
                query_start_loc[bs] = forward_batch.extend_start_loc[-1] + forward_batch.extend_seq_lens[-1]
        else:
            raise ValueError(f"Invalid forward mode: {forward_batch.forward_mode=}")
        mamba_cache_indices = self.req_to_token_pool.get_mamba_indices(forward_batch.req_pool_indices)
        self.forward_metadata = ForwardMetadata(
            query_start_loc=query_start_loc,
            mamba_cache_indices=mamba_cache_indices,
        )

    def init_cuda_graph_state(self, max_num_tokens: int):
        # self.speculative_num_draft_tokens should equal to (max_num_tokens // max_bs)
        for i in range(max_num_tokens):
            self.state_indices_list.append(torch.full((i + 1,), self.pad_slot_id, dtype=torch.int32, device=self.device))
            self.query_start_loc_list.append(torch.empty((i + 2,), dtype=torch.int32, device=self.device))
        self.cached_cuda_graph_decode_query_start_loc = torch.arange(0, max_num_tokens + 1, dtype=torch.int32, device=self.device)
        if self.speculative_num_draft_tokens > 0:
            self.cached_cuda_graph_verify_query_start_loc = torch.arange(
                0,
                max_num_tokens + 1,
                step=self.speculative_num_draft_tokens,
                dtype=torch.int32,
                device=self.device,
            )

    def init_forward_metadata_prefill_capture_cuda_graph(
        self,
        forward_batch,
        bs: int,
        num_tokens: int,
        seq_lens: torch.Tensor,
        prefix_lens: torch.Tensor,
        req_pool_indices: torch.Tensor,
    ):
        # For prefill cuda graph capture, build query_start_loc from extend_seq_lens
        # extend_seq_lens = seq_lens - prefix_lens (tokens to actually process)
        extend_seq_lens = seq_lens - prefix_lens
        query_start_loc = torch.zeros(bs + 1, dtype=torch.int32, device=self.device)
        query_start_loc[1:] = torch.cumsum(extend_seq_lens[:bs], dim=0)

        mamba_indices = self.req_to_token_pool.get_mamba_indices(req_pool_indices)
        self.forward_metadata = ForwardMetadata(
            query_start_loc=query_start_loc,
            mamba_cache_indices=mamba_indices,
        )

    def init_forward_metadata_replay_prefill_cuda_graph(
        self,
        forward_batch,
        bs: int,
        num_tokens: int,
        seq_lens: torch.Tensor,
        req_pool_indices: torch.Tensor,
        prefix_lens: torch.Tensor,
    ):
        # For prefill cuda graph replay, rebuild query_start_loc
        extend_seq_lens = seq_lens - prefix_lens
        query_start_loc = torch.zeros(bs + 1, dtype=torch.int32, device=self.device)
        query_start_loc[1:] = torch.cumsum(extend_seq_lens[:bs], dim=0)

        mamba_indices = self.req_to_token_pool.get_mamba_indices(req_pool_indices)
        self.forward_metadata = ForwardMetadata(
            query_start_loc=query_start_loc,
            mamba_cache_indices=mamba_indices,
        )

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        forward_mode: ForwardMode,
        spec_info: Optional[Union[EagleDraftInput, EagleVerifyInput]],
    ):
        if forward_mode.is_decode_or_idle():
            self.query_start_loc_list[bs - 1].copy_(self.cached_cuda_graph_decode_query_start_loc[: bs + 1])
        elif forward_mode.is_target_verify():
            self.query_start_loc_list[bs - 1].copy_(self.cached_cuda_graph_verify_query_start_loc[: bs + 1])
        else:
            raise ValueError(f"Invalid forward mode: {forward_mode=}")
        mamba_indices = self.req_to_token_pool.get_mamba_indices(req_pool_indices)
        self.state_indices_list[bs - 1][: len(mamba_indices)].copy_(mamba_indices)
        self.forward_metadata = ForwardMetadata(
            query_start_loc=self.query_start_loc_list[bs - 1],
            mamba_cache_indices=self.state_indices_list[bs - 1],
        )

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        forward_mode: ForwardMode,
        spec_info: Optional[Union[EagleDraftInput, EagleVerifyInput]],
        seq_lens_cpu: Optional[torch.Tensor],
    ):
        req_pool_indices = req_pool_indices[:bs]
        seq_lens_cpu = seq_lens_cpu[:bs]
        num_padding = torch.count_nonzero(seq_lens_cpu == self.get_cuda_graph_seq_len_fill_value())
        # Make sure forward metadata is correctly handled for padding reqs
        req_pool_indices[bs - num_padding :] = 0
        mamba_indices = self.req_to_token_pool.get_mamba_indices(req_pool_indices)
        mamba_indices[bs - num_padding :] = -1
        self.state_indices_list[bs - 1][: len(mamba_indices)].copy_(mamba_indices)
        if forward_mode.is_decode_or_idle():
            if num_padding == 0:
                self.query_start_loc_list[bs - 1].copy_(self.cached_cuda_graph_decode_query_start_loc[: bs + 1])
            else:
                self.query_start_loc_list[bs - 1][: bs - num_padding].copy_(self.cached_cuda_graph_decode_query_start_loc[: bs - num_padding])
                self.query_start_loc_list[bs - 1][bs - num_padding :].copy_(bs - num_padding)
        elif forward_mode.is_target_verify():
            if num_padding == 0:
                self.query_start_loc_list[bs - 1].copy_(self.cached_cuda_graph_verify_query_start_loc[: bs + 1])
            else:
                self.query_start_loc_list[bs - 1][: bs - num_padding].copy_(self.cached_cuda_graph_verify_query_start_loc[: bs - num_padding])
                self.query_start_loc_list[bs - 1][bs - num_padding :].copy_((bs - num_padding) * spec_info.draft_token_num)
        else:
            raise ValueError(f"Invalid forward mode: {forward_mode=}")

        self.forward_metadata = ForwardMetadata(
            query_start_loc=self.query_start_loc_list[bs - 1],
            mamba_cache_indices=self.state_indices_list[bs - 1],
        )

    def get_cuda_graph_seq_len_fill_value(self):
        return 1  # Mamba attn does not use seq lens to index kv cache

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        **kwargs,
    ):
        mixed_qkv = kwargs["mixed_qkv"]
        conv_weights = kwargs["conv_weights"]
        bias = kwargs["bias"]
        activation = kwargs["activation"]
        key_dim = kwargs["key_dim"]
        value_dim = kwargs["value_dim"]
        attn_tp_size = kwargs["attention_tp_size"]
        head_k_dim = kwargs["head_k_dim"]
        head_v_dim = kwargs["head_v_dim"]
        a = kwargs["a"]
        b = kwargs["b"]
        A_log = kwargs["A_log"]
        dt_bias = kwargs["dt_bias"]
        layer_id = kwargs["layer_id"]

        conv_states, ssm_states, *rest = self.req_to_token_pool.get_mamba_params(layer_id)
        query_start_loc = self.forward_metadata.query_start_loc
        cache_indices = self.forward_metadata.mamba_cache_indices

        mixed_qkv = causal_conv1d_update(
            mixed_qkv,
            conv_states,
            conv_weights,
            bias,
            activation,
            conv_state_indices=cache_indices,
        )

        query, key, value = torch.split(
            mixed_qkv,
            [
                key_dim // attn_tp_size,
                key_dim // attn_tp_size,
                value_dim // attn_tp_size,
            ],
            dim=-1,
        )
        # Reshape from [l, h*d] to [1, l, h, d]
        seq_len = query.shape[0]
        num_heads = query.shape[1] // head_k_dim
        query = query.view(1, seq_len, num_heads, head_k_dim)
        key = key.view(1, seq_len, num_heads, head_k_dim)
        value = value.view(1, seq_len, value.shape[1] // head_v_dim, head_v_dim)

        core_attn_out = fused_sigmoid_gating_delta_rule_update(
            A_log=A_log,
            dt_bias=dt_bias,
            q=query,
            k=key,
            v=value,
            a=a,
            b=b,
            initial_state_source=ssm_states,
            initial_state_indices=cache_indices,
            cu_seqlens=query_start_loc,
            use_qk_l2norm_in_kernel=True,
            softplus_beta=1.0,
            softplus_threshold=20.0,
        )

        return core_attn_out

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        **kwargs,
    ):
        mixed_qkv = kwargs["mixed_qkv"]
        conv_weights = kwargs["conv_weights"]
        bias = kwargs["bias"]
        activation = kwargs["activation"]
        key_dim = kwargs["key_dim"]
        value_dim = kwargs["value_dim"]
        attn_tp_size = kwargs["attention_tp_size"]
        head_k_dim = kwargs["head_k_dim"]
        head_v_dim = kwargs["head_v_dim"]
        a = kwargs["a"].contiguous()
        b = kwargs["b"].contiguous()
        A_log = kwargs["A_log"]
        dt_bias = kwargs["dt_bias"]
        layer_id = kwargs["layer_id"]
        seq_len = kwargs["seq_len"]

        is_target_verify = forward_batch.forward_mode.is_target_verify()

        query_start_loc = self.forward_metadata.query_start_loc
        cache_indices = self.forward_metadata.mamba_cache_indices

        if is_target_verify:
            (
                conv_states,
                ssm_states,
                intermediate_state_cache,
                intermediate_conv_window_cache,
            ) = self.req_to_token_pool.get_mamba_params(layer_id)
            has_initial_states = torch.ones(
                seq_len // forward_batch.spec_info.draft_token_num,
                dtype=torch.bool,
                device=forward_batch.input_ids.device,
            )
            conv_states_to_use = conv_states.clone()
        else:
            conv_states, ssm_states, *rest = self.req_to_token_pool.get_mamba_params(layer_id)
            has_initial_states = forward_batch.extend_prefix_lens > 0
            conv_states_to_use = conv_states

        if is_target_verify:
            batch_size = seq_len // forward_batch.spec_info.draft_token_num
            draft_token_num = forward_batch.spec_info.draft_token_num
            mixed_qkv_reshaped = mixed_qkv.view(batch_size, draft_token_num, -1).transpose(1, 2).contiguous()
            mixed_qkv_processed = causal_conv1d_update(
                mixed_qkv_reshaped,
                conv_states_to_use,
                conv_weights,
                bias,
                activation,
                conv_state_indices=cache_indices[:batch_size],
                intermediate_conv_window=intermediate_conv_window_cache,
            )
            mixed_qkv = mixed_qkv_processed.transpose(1, 2).contiguous().view(seq_len, -1)
        else:
            mixed_qkv = (
                causal_conv1d_fn(
                    mixed_qkv.transpose(0, 1).contiguous(),
                    conv_weights,
                    bias,
                    activation=activation,
                    conv_states=conv_states_to_use,
                    has_initial_state=has_initial_states,
                    cache_indices=cache_indices,
                    query_start_loc=query_start_loc,
                    seq_lens_cpu=forward_batch.extend_seq_lens_cpu,
                )
                .transpose(0, 1)
                .contiguous()[:seq_len]
            )

        key_split_dim = key_dim // attn_tp_size
        value_split_dim = value_dim // attn_tp_size

        query, key, value = torch.split(
            mixed_qkv,
            [key_split_dim, key_split_dim, value_split_dim],
            dim=-1,
        )

        actual_seq_len = query.shape[0]
        num_heads = query.shape[1] // head_k_dim
        num_value_heads = value.shape[1] // head_v_dim

        query = query.view(1, actual_seq_len, num_heads, head_k_dim)
        key = key.view(1, actual_seq_len, num_heads, head_k_dim)
        value = value.view(1, actual_seq_len, num_value_heads, head_v_dim)

        g, beta = fused_gdn_gating(A_log, a, b, dt_bias)

        if is_target_verify:
            core_attn_out = fused_recurrent_gated_delta_rule_update(
                q=query,
                k=key,
                v=value,
                g=g,
                beta=beta,
                initial_state_source=ssm_states,
                initial_state_indices=cache_indices,
                cu_seqlens=query_start_loc,
                use_qk_l2norm_in_kernel=True,
                disable_state_update=True,
                intermediate_states_buffer=intermediate_state_cache,
                cache_steps=forward_batch.spec_info.draft_token_num,
            )
        else:
            core_attn_out = chunk_gated_delta_rule(
                q=query,
                k=key,
                v=value,
                g=g,
                beta=beta,
                initial_state=ssm_states,
                initial_state_indices=cache_indices,
                cu_seqlens=query_start_loc,
                head_first=False,
                use_qk_l2norm_in_kernel=True,
            )

        return core_attn_out


class HybridLinearAttnBackend(AttentionBackend):
    """Support different backends for prefill and decode."""

    def __init__(
        self,
        full_attn_backend: AttentionBackend,
        linear_attn_backend: AttentionBackend,
        full_attn_layers: list[int],
    ):
        self.full_attn_layers = full_attn_layers
        self.attn_backend_list = [full_attn_backend, linear_attn_backend]

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        for attn_backend in self.attn_backend_list:
            attn_backend.init_forward_metadata(forward_batch)

    def init_cuda_graph_state_prefill(self, max_bs: int, kv_indices_buf=None):
        for attn_backend in self.attn_backend_list:
            attn_backend.init_cuda_graph_state_prefill(max_bs, kv_indices_buf)

    def init_forward_metadata_prefill_capture_cuda_graph(
        self,
        forward_batch,
        bs: int,
        num_tokens: int,
        seq_lens: torch.Tensor,
        prefix_lens: torch.Tensor,
        req_pool_indices: torch.Tensor,
    ):
        for attn_backend in self.attn_backend_list:
            attn_backend.init_forward_metadata_prefill_capture_cuda_graph(forward_batch, bs, num_tokens, seq_lens, prefix_lens, req_pool_indices)

    def init_forward_metadata_replay_prefill_cuda_graph(
        self,
        forward_batch,
        bs: int,
        num_tokens: int,
        seq_lens: torch.Tensor,
        req_pool_indices: torch.Tensor,
        prefix_lens: torch.Tensor,
    ):
        for attn_backend in self.attn_backend_list:
            attn_backend.init_forward_metadata_replay_prefill_cuda_graph(forward_batch, bs, num_tokens, seq_lens, req_pool_indices, prefix_lens)

    def init_cuda_graph_state(self, max_bs: int):
        for attn_backend in self.attn_backend_list:
            attn_backend.init_cuda_graph_state(max_bs)

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        forward_mode: ForwardMode,
        spec_info: Optional[Union[EagleDraftInput, EagleVerifyInput]],
    ):
        for attn_backend in self.attn_backend_list:
            attn_backend.init_forward_metadata_capture_cuda_graph(
                bs,
                num_tokens,
                req_pool_indices,
                seq_lens,
                forward_mode,
                spec_info,
            )

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        forward_mode: ForwardMode,
        spec_info: Optional[Union[EagleDraftInput, EagleVerifyInput]],
        seq_lens_cpu: Optional[torch.Tensor],
    ):
        for attn_backend in self.attn_backend_list:
            attn_backend.init_forward_metadata_replay_cuda_graph(
                bs,
                req_pool_indices,
                seq_lens,
                seq_lens_sum,
                forward_mode,
                spec_info,
                seq_lens_cpu,
            )

    def get_cuda_graph_seq_len_fill_value(self):
        return self.attn_backend_list[0].get_cuda_graph_seq_len_fill_value()

    def get_indexer_metadata(self, layer_id: int, forward_batch: ForwardBatch):
        if layer_id not in self.full_attn_layers:
            return None
        return self.attn_backend_list[0].get_indexer_metadata(layer_id, forward_batch)

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        **kwargs,
    ):
        layer_id = layer.layer_id if layer else kwargs["layer_id"]
        if layer_id in self.full_attn_layers:
            return self.attn_backend_list[0].forward_decode(q, k, v, layer, forward_batch, save_kv_cache, **kwargs)
        return self.attn_backend_list[1].forward_decode(q, k, v, layer, forward_batch, save_kv_cache, **kwargs)

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        **kwargs,
    ):
        layer_id = layer.layer_id if layer else kwargs["layer_id"]
        if layer_id in self.full_attn_layers:
            return self.attn_backend_list[0].forward_extend(q, k, v, layer, forward_batch, save_kv_cache, **kwargs)
        return self.attn_backend_list[1].forward_extend(q, k, v, layer, forward_batch, save_kv_cache, **kwargs)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        **kwargs,
    ):
        """Run forward on an attention layer."""
        if forward_batch.forward_mode.is_idle():
            if layer is None:
                return torch.empty_like(kwargs["z"])
            return q.new_empty(q.shape[0], layer.tp_q_head_num * layer.v_head_dim)
        elif forward_batch.forward_mode.is_decode():
            return self.forward_decode(
                q,
                k,
                v,
                layer,
                forward_batch,
                save_kv_cache=save_kv_cache,
                **kwargs,
            )
        else:
            layer_id = layer.layer_id if layer else kwargs["layer_id"]
            o = self.forward_extend(
                q,
                k,
                v,
                layer,
                forward_batch,
                save_kv_cache=save_kv_cache,
                **kwargs,
            )
            if not forward_batch.forward_mode.is_idle() and getattr(self, "step_counter", None) and layer_id in self.full_attn_layers:
                self.step_counter.record_cache()
            return o

    def update_mamba_state_after_mtp_verify(self, accepted_length, model):
        request_number = accepted_length.shape[0]

        state_indices_tensor = self.attn_backend_list[1].forward_metadata.mamba_cache_indices[:request_number]

        mamba_caches = self.attn_backend_list[1].req_to_token_pool.get_mamba_params_all_layers()

        (
            conv_states,
            ssm_states,
            intermediate_state_cache,
            intermediate_conv_window_cache,
        ) = mamba_caches

        # SSM state updates (chunked to reduce peak memory)
        valid_mask = accepted_length > 0

        # Compute common indices once to avoid duplication
        last_steps_all = (accepted_length - 1).to(torch.int64)
        valid_state_indices = state_indices_tensor[valid_mask].to(torch.int64)  # [N]
        last_steps = last_steps_all[valid_mask].to(torch.int64)  # [N]

        # scatter into ssm_states at the chosen cache lines
        ssm_states[:, valid_state_indices, :] = intermediate_state_cache[:, valid_state_indices, last_steps].to(ssm_states.dtype, copy=False)

        # Scatter into conv_states at the chosen cache lines
        conv_states[:, valid_state_indices, :, :] = intermediate_conv_window_cache[:, valid_state_indices, last_steps].to(
            conv_states.dtype, copy=False
        )


def _flash_kda_prefill(q, k, v, g, beta, scale, A_log, dt_bias, lower_bound, ssm_states, cache_indices, cu_seqlens):
    """FlashKDA prefill with copy-based state management.

    Args:
        q: [1, T, H, K] bf16
        k: [1, T, H, K] bf16
        v: [1, T, H, V] bf16
        g: [1, T, H, K] bf16 (raw, pre-activation)
        beta: [1, T, H] bf16 (logits, pre-sigmoid)
        scale: float
        A_log: [1, 1, H, 1] fp32
        dt_bias: [H*K] fp32
        lower_bound: float
        ssm_states: [pool_size, H, V, K] state pool
        cache_indices: [N] int32
        cu_seqlens: [N+1] int32/int64
    """
    H, K = q.shape[2], q.shape[3]
    N = cu_seqlens.shape[0] - 1

    # Reshape A_log: [1,1,H,1] -> [H]
    A_log_flat = A_log.reshape(-1)
    # Reshape dt_bias: [H*K] -> [H, K]
    dt_bias_2d = dt_bias.reshape(H, K)

    # Copy initial state from pool: [N, H, V, K]
    initial_state = ssm_states[cache_indices[:N]].contiguous()
    final_state = torch.empty_like(initial_state)
    out = torch.empty_like(v)  # [1, T, H, V]

    flash_kda.fwd(
        q.contiguous().to(torch.bfloat16),
        k.contiguous().to(torch.bfloat16),
        v.contiguous().to(torch.bfloat16),
        g.contiguous().to(torch.bfloat16),
        beta.contiguous().to(torch.bfloat16),
        scale,
        out,
        A_log=A_log_flat,
        dt_bias=dt_bias_2d,
        lower_bound=lower_bound,
        initial_state=initial_state,
        final_state=final_state,
        cu_seqlens=cu_seqlens.to(torch.int64),
    )

    # Write back final state to pool
    ssm_states[cache_indices[:N]] = final_state.to(ssm_states.dtype)
    return out


class KimiLinearAttnBackend(MambaAttnBackend):
    """Attention backend using Mamba kernel."""

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        **kwargs,
    ):
        q_proj_states = kwargs["q_proj_states"]
        k_proj_states = kwargs["k_proj_states"]
        v_proj_states = kwargs["v_proj_states"]
        q_conv_weights = kwargs["q_conv_weights"]
        k_conv_weights = kwargs["k_conv_weights"]
        v_conv_weights = kwargs["v_conv_weights"]

        q_conv_bias = kwargs["q_conv_bias"]
        k_conv_bias = kwargs["k_conv_bias"]
        v_conv_bias = kwargs["v_conv_bias"]

        A_log = kwargs["A_log"]
        dt_bias = kwargs["dt_bias"]
        b_proj = kwargs["b_proj"]
        f_a_proj = kwargs["f_a_proj"]
        f_b_proj = kwargs["f_b_proj"]
        hidden_states = kwargs["hidden_states"]
        head_dim = kwargs["head_dim"]
        layer_id = kwargs["layer_id"]
        use_fgb_kda = kwargs.get("use_fgb_kda", False)

        layer_cache = self.req_to_token_pool.mamba2_layer_cache(layer_id)
        q_conv_state, k_conv_state, v_conv_state = layer_cache["conv"]
        ssm_states = layer_cache["temporal"]
        query_start_loc = self.forward_metadata.query_start_loc
        cache_indices = self.forward_metadata.mamba_cache_indices

        q_conv_state = q_conv_state.transpose(-1, -2)
        k_conv_state = k_conv_state.transpose(-1, -2)
        v_conv_state = v_conv_state.transpose(-1, -2)

        q = causal_conv1d_update(
            q_proj_states,
            q_conv_state,
            q_conv_weights,
            q_conv_bias,
            activation="silu",
            conv_state_indices=cache_indices,
        )
        k = causal_conv1d_update(
            k_proj_states,
            k_conv_state,
            k_conv_weights,
            k_conv_bias,
            activation="silu",
            conv_state_indices=cache_indices,
        )
        v = causal_conv1d_update(
            v_proj_states,
            v_conv_state,
            v_conv_weights,
            v_conv_bias,
            activation="silu",
            conv_state_indices=cache_indices,
        )

        q, k, v = map(lambda x: rearrange(x, "n (h d) -> 1 n h d", d=head_dim), (q, k, v))

        if use_fgb_kda:
            if isinstance(b_proj, torch.nn.Sequential):
                beta = b_proj[1](b_proj[0](hidden_states)[0])[0].sigmoid()
            else:
                beta = b_proj(hidden_states)[0].sigmoid()
            beta = rearrange(beta, "n (h d) -> 1 n h d", d=head_dim)
            beta = torch.sqrt(beta + 1e-10)
            q = F.normalize(q, p=2, dim=-1)
            k = F.normalize(k, p=2, dim=-1)
            k = k * beta
            v = v * beta
            # Construct dummy b so sigmoid(b)≈1.0 (beta already absorbed into k/v)
            b_for_kernel = torch.full((*q.shape[:3],), 20.0, dtype=q.dtype, device=q.device)
            use_qk_l2norm_in_kernel = False
        else:
            b_for_kernel = b_proj(hidden_states)[0].unsqueeze(0)  # raw logits (pre-sigmoid)
            use_qk_l2norm_in_kernel = True

        g = f_b_proj(f_a_proj(hidden_states)[0])[0]
        g = rearrange(g, "n (h d) -> 1 n h d", d=head_dim)

        lower_bound = kwargs.get("lower_bound")

        core_attn_out = fused_sigmoid_gating_delta_rule_update(
            A_log=A_log,
            dt_bias=dt_bias,
            q=q,
            k=k,
            v=v,
            a=g,
            b=b_for_kernel,
            initial_state_source=ssm_states,
            initial_state_indices=cache_indices,
            cu_seqlens=query_start_loc,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            softplus_beta=1.0,
            softplus_threshold=20.0,
            is_kda=True,
            lower_bound=lower_bound,
        )
        return core_attn_out

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        **kwargs,
    ):
        from sglang.srt.layers.attention.mamba.causal_conv1d_triton import (
            causal_conv1d_fn,
        )

        q_proj_states = kwargs["q_proj_states"]
        k_proj_states = kwargs["k_proj_states"]
        v_proj_states = kwargs["v_proj_states"]
        q_conv_weights = kwargs["q_conv_weights"]
        k_conv_weights = kwargs["k_conv_weights"]
        v_conv_weights = kwargs["v_conv_weights"]

        q_conv_bias = kwargs["q_conv_bias"]
        k_conv_bias = kwargs["k_conv_bias"]
        v_conv_bias = kwargs["v_conv_bias"]

        A_log = kwargs["A_log"]
        dt_bias = kwargs["dt_bias"]
        b_proj = kwargs["b_proj"]
        f_a_proj = kwargs["f_a_proj"]
        f_b_proj = kwargs["f_b_proj"]
        hidden_states = kwargs["hidden_states"]
        head_dim = kwargs["head_dim"]
        layer_id = kwargs["layer_id"]
        use_fgb_kda = kwargs.get("use_fgb_kda", False)

        query_start_loc = self.forward_metadata.query_start_loc
        cache_indices = self.forward_metadata.mamba_cache_indices

        mamba_cache_params = self.req_to_token_pool.mamba2_layer_cache(layer_id)
        conv_state_q, conv_state_k, conv_state_v = mamba_cache_params["conv"]
        # deal with strides
        conv_state_q = conv_state_q.transpose(-1, -2)
        conv_state_k = conv_state_k.transpose(-1, -2)
        conv_state_v = conv_state_v.transpose(-1, -2)

        ssm_states = mamba_cache_params["temporal"]

        has_initial_state = forward_batch.extend_prefix_lens > 0

        q_proj_states = q_proj_states.transpose(0, 1)
        k_proj_states = k_proj_states.transpose(0, 1)
        v_proj_states = v_proj_states.transpose(0, 1)

        q = causal_conv1d_fn(
            q_proj_states,
            q_conv_weights,
            q_conv_bias,
            activation="silu",
            conv_states=conv_state_q,
            has_initial_state=has_initial_state,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
            seq_lens_cpu=forward_batch.extend_seq_lens_cpu,
        ).transpose(0, 1)

        k = causal_conv1d_fn(
            k_proj_states,
            k_conv_weights,
            k_conv_bias,
            activation="silu",
            conv_states=conv_state_k,
            has_initial_state=has_initial_state,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
            seq_lens_cpu=forward_batch.extend_seq_lens_cpu,
        ).transpose(0, 1)

        v = causal_conv1d_fn(
            v_proj_states,
            v_conv_weights,
            v_conv_bias,
            activation="silu",
            conv_states=conv_state_v,
            has_initial_state=has_initial_state,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
            seq_lens_cpu=forward_batch.extend_seq_lens_cpu,
        ).transpose(0, 1)

        q, k, v = map(lambda x: rearrange(x, "n (h d) -> 1 n h d", d=head_dim), (q, k, v))

        lower_bound = kwargs.get("lower_bound")

        g = f_b_proj(f_a_proj(hidden_states)[0])[0]
        g = rearrange(g, "n (h d) -> 1 n h d", d=head_dim)

        if _FLASH_KDA_ENABLED and not use_fgb_kda and lower_bound is not None:
            # FlashKDA path: pass raw beta logits (pre-sigmoid)
            beta_logits = b_proj(hidden_states)[0].unsqueeze(0)  # [1, T, H] bf16

            core_attn_out = _flash_kda_prefill(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta_logits,
                scale=head_dim**-0.5,
                A_log=A_log,
                dt_bias=dt_bias,
                lower_bound=lower_bound,
                ssm_states=ssm_states,
                cache_indices=cache_indices,
                cu_seqlens=query_start_loc,
            )
        else:
            # Triton fallback
            if use_fgb_kda:
                if isinstance(b_proj, torch.nn.Sequential):
                    beta = b_proj[1](b_proj[0](hidden_states)[0])[0].sigmoid()
                else:
                    beta = b_proj(hidden_states)[0].sigmoid()
                beta = rearrange(beta, "n (h d) -> 1 n h d", d=head_dim)
                beta = torch.sqrt(beta + 1e-10)
                q = F.normalize(q, p=2, dim=-1)
                k = F.normalize(k, p=2, dim=-1)
                k = k * beta
                v = v * beta
                beta = torch.ones(beta.shape[:-1], dtype=beta.dtype, device=beta.device)
                use_qk_l2norm_in_kernel = False
            else:
                beta = b_proj(hidden_states)[0].float().sigmoid().unsqueeze(0)
                use_qk_l2norm_in_kernel = True

            core_attn_out = chunk_kda(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                initial_state=ssm_states,
                initial_state_indices=cache_indices,
                use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                cu_seqlens=query_start_loc,
                A_log=A_log,
                dt_bias=dt_bias,
                lower_bound=lower_bound,
            )

        return core_attn_out
