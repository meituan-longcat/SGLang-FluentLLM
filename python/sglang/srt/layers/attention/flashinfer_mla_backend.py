from __future__ import annotations

"""
Support attention backend for flashinfer MLA.
The flashinfer_mla_disable_ragged flag controls whether to use ragged prefill wrapper and defaults to be false.
When it's set to false, all wrappers are BatchMLAPaged wrapper.
When it's set to true, the backend uses BatchRagged and BatchMLAPaged wrapper for prefilling,
and uses BatchMLAPaged wrapper for decoding.
More details can be found in https://docs.flashinfer.ai/api/mla.html
"""

import os
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Callable, Optional, Union

import torch
import triton

if os.getenv("SGLANG_ENABLE_TORCH_COMPILE", "0") == "1":
    import logging

    torch._logging.set_logs(dynamo=logging.ERROR)
    torch._dynamo.config.suppress_errors = True

from sglang.global_config import global_config
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.flashinfer_backend import (
    create_flashinfer_kv_indices_triton,
)
from sglang.srt.layers.dp_attention import get_attention_tp_size
from sglang.srt.env import global_server_args_dict
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.speculative.eagle_utils import EagleDraftInput, EagleVerifyInput, generate_attn_arg_prefill
from sglang.srt.utils import is_flashinfer_available, is_sm90_supported
from sglang.srt.utils import get_colorful_logger

logger = get_colorful_logger(__name__)


def _remap_flashinfer_kv_indices(
    token_to_kv_pool,
    kv_indices: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_lens: torch.Tensor,
):
    if not getattr(token_to_kv_pool, "enable_mla_l1_5_cache", False):
        return kv_indices, kv_indptr, kv_lens
    if kv_indices.numel() == 0:
        return kv_indices, kv_indptr, kv_lens

    mask_out, _, _, _, _ = token_to_kv_pool.global_loc_to_local_mapping(kv_indices)
    if mask_out.numel() == 0:
        return kv_indices, kv_indptr, kv_lens

    bs = kv_lens.numel()
    seg_ids = torch.repeat_interleave(
        torch.arange(bs, device=kv_indices.device), kv_lens
    )
    local_kv_lens = torch.zeros_like(kv_lens)
    if seg_ids.numel() > 0:
        local_kv_lens.scatter_add_(0, seg_ids, mask_out.to(kv_lens.dtype))

    local_kv_indptr = kv_indptr.clone()
    if bs > 0:
        local_kv_indptr[1:] = torch.cumsum(local_kv_lens, dim=0)

    local_kv_indices = kv_indices[mask_out]
    kv_indices[: local_kv_indices.numel()].copy_(local_kv_indices)
    kv_indices = kv_indices[: local_kv_indices.numel()]
    return kv_indices, local_kv_indptr, local_kv_lens

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.speculative.spec_info import SpecInfo

if is_flashinfer_available():
    from flashinfer import (
        BatchMLAPagedAttentionWrapper,
        BatchPrefillWithRaggedKVCacheWrapper,
    )
if is_sm90_supported():
    from flash_mla_fp8 import dequantize_ckv_fused_indexed

@dataclass
class DecodeMetadata:
    decode_wrapper: BatchMLAPagedAttentionWrapper


@dataclass
class PrefillMetadata:
    prefill_wrapper: BatchMLAPagedAttentionWrapper
    use_ragged: bool


# Reuse this workspace buffer across all flashinfer wrappers
global_workspace_buffer = None


class FlashInferMLAAttnBackend(AttentionBackend):
    """Flashinfer attention kernels."""

    def __init__(
        self,
        model_runner: ModelRunner,
        skip_prefill: bool = False,
        kv_indptr_buf: Optional[torch.Tensor] = None,
        q_indptr_decode_buf: Optional[torch.Tensor] = None,
    ):
        super().__init__()

        # Parse constants
        self.max_context_len = model_runner.model_config.context_len
        self.kv_cache_quant_method = model_runner.kv_cache_quant_method
        self.device = model_runner.device
        self.skip_prefill = skip_prefill

        # Allocate buffers
        global global_workspace_buffer
        if global_workspace_buffer is None:
            global_workspace_buffer = torch.empty(
                global_config.flashinfer_workspace_size,
                dtype=torch.uint8,
                device=model_runner.device,
            )
        self.workspace_buffer = global_workspace_buffer

        max_bs = model_runner.req_to_token_pool.size
        if kv_indptr_buf is None:
            self.kv_indptr = torch.zeros(
                (max_bs + 1,), dtype=torch.int32, device=model_runner.device
            )
        else:
            self.kv_indptr = kv_indptr_buf
        if self.kv_cache_quant_method == "per_token_head":
            self.kv_fetch_indices = torch.empty(
                model_runner.req_to_token_pool.req_to_token.numel(),
                dtype=torch.int32,
                device=model_runner.device,
            )
        else:
            self.kv_fetch_indices = None

        if not self.skip_prefill:
            self.qo_indptr = torch.zeros(
                (max_bs + 1,), dtype=torch.int32, device=model_runner.device
            )

        if q_indptr_decode_buf is None:
            self.q_indptr_decode = torch.arange(
                0, max_bs + 1, dtype=torch.int32, device=model_runner.device
            )
        else:
            self.q_indptr_decode = q_indptr_decode_buf

        self.prefill_wrapper_ragged = BatchPrefillWithRaggedKVCacheWrapper(
            self.workspace_buffer, "NHD"
        )

        if not self.skip_prefill:
            self.prefill_wrapper_paged = BatchMLAPagedAttentionWrapper(
                self.workspace_buffer,
                backend="auto",
            )

            # FlashinferMLA backend uses mla wrapper for target verify
            self.prefill_wrapper_verify = BatchMLAPagedAttentionWrapper(
                self.workspace_buffer,
                backend="auto",
            )

        self.decode_wrapper = BatchMLAPagedAttentionWrapper(
            self.workspace_buffer, backend="auto"
        )

        # Create indices updater
        if not skip_prefill:
            self.indices_updater_prefill = FlashInferMLAIndicesUpdaterPrefill(
                model_runner, self
            )

        self.indices_updater_decode = FlashInferMLAIndicesUpdaterDecode(
            model_runner, self
        )

        # Other metadata
        self.forward_metadata: Union[PrefillMetadata, DecodeMetadata] = None
        self.decode_cuda_graph_metadata = {}
        self.verify_cuda_graph_metadata = {}  # For verify
        self.prefill_cuda_graph_metadata = {}
        if not model_runner.server_args.disable_prefill_graph:
            prefill_graph_batch_sizes = model_runner.server_args.prefill_graph_max_bs
            for i in range(prefill_graph_batch_sizes):
                self.prefill_cuda_graph_metadata[i + 1] = {}

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        if forward_batch.forward_mode.is_decode_or_idle():
            self.indices_updater_decode.update(
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                forward_batch.seq_lens_sum,
                decode_wrapper=self.decode_wrapper,
                init_metadata_replay=False,
            )
            self.forward_metadata = DecodeMetadata(self.decode_wrapper)
        elif forward_batch.forward_mode.is_target_verify() or forward_batch.forward_mode.is_draft_extend():
            self.indices_updater_prefill.update(
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                forward_batch.seq_lens_sum,
                self.kv_fetch_indices,
                prefix_lens=None,
                prefill_wrapper_paged=self.prefill_wrapper_verify,
                use_ragged=False,
                spec_info=forward_batch.spec_info,
            )
            self.forward_metadata = PrefillMetadata(self.prefill_wrapper_verify, False)
        else:
            prefix_lens = forward_batch.extend_prefix_lens
            extend_no_prefix = not any(forward_batch.extend_prefix_lens_cpu)
            use_ragged = (
                not global_server_args_dict["flashinfer_mla_disable_ragged"]
                and extend_no_prefix
            )

            self.indices_updater_prefill.update(
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                forward_batch.seq_lens_sum,
                self.kv_fetch_indices,
                prefix_lens,
                prefill_wrapper_paged=self.prefill_wrapper_paged,
                use_ragged=use_ragged,
            )
            self.forward_metadata = PrefillMetadata(
                self.prefill_wrapper_paged, use_ragged
            )

    def init_cuda_graph_state_prefill(
        self, max_bs: int, kv_indices_buf: Optional[torch.Tensor] = None
    ):
        if kv_indices_buf is None:
            cuda_graph_kv_indices = torch.zeros(
                (max_bs * self.max_context_len,),
                dtype=torch.int32,
                device="cuda",
            )
        else:
            cuda_graph_kv_indices = kv_indices_buf

        self.cuda_graph_kv_indices_prefill = cuda_graph_kv_indices
        self.cuda_graph_qo_indptr_prefill = self.q_indptr_decode.clone()
        self.cuda_graph_kv_indptr_prefill = self.kv_indptr.clone()
        self.cuda_graph_kv_lens_prefill = torch.ones(
            (max_bs,), dtype=torch.int32, device=self.device
        )

    def init_cuda_graph_state(
        self, max_bs: int, kv_indices_buf: Optional[torch.Tensor] = None
    ):
        if kv_indices_buf is None:
            cuda_graph_kv_indices = torch.zeros(
                (max_bs * self.max_context_len,),
                dtype=torch.int32,
                device="cuda",
            )
        else:
            cuda_graph_kv_indices = kv_indices_buf

        self.cuda_graph_kv_indices = cuda_graph_kv_indices
        self.cuda_graph_qo_indptr = self.q_indptr_decode.clone()
        self.cuda_graph_kv_indptr = self.kv_indptr.clone()
        self.cuda_graph_kv_lens = torch.ones(
            (max_bs,), dtype=torch.int32, device=self.device
        )

        # For fast decode plan in graph replaying
        self.cuda_graph_qo_indptr_cpu = self.cuda_graph_qo_indptr.to("cpu")
        self.cuda_graph_kv_indptr_cpu = self.cuda_graph_kv_indptr.to("cpu")
        self.cuda_graph_kv_lens_cpu = self.cuda_graph_kv_lens.to("cpu")
        self.fast_decode_kwargs = {
            "qo_indptr_cpu": self.cuda_graph_qo_indptr_cpu,
            "kv_indptr_cpu": self.cuda_graph_kv_indptr_cpu,
            "kv_len_arr_cpu": self.cuda_graph_kv_lens_cpu,
            "kv_indices": self.cuda_graph_kv_indices,
        }

    def init_forward_metadata_prefill_capture_cuda_graph(
        self,
        forward_batch: ForwardBatch,
        bs: int,
        num_tokens: int,
        seq_lens: torch.Tensor,
        prefix_lens: torch.Tensor,
        req_pool_indices: torch.Tensor,
    ):
        if forward_batch.forward_mode == ForwardMode.EXTEND:
            wrapper = BatchMLAPagedAttentionWrapper(
                self.workspace_buffer,
                use_cuda_graph=True,
                qo_indptr=self.cuda_graph_qo_indptr_prefill[: bs + 1],
                kv_indptr=self.cuda_graph_kv_indptr_prefill[: bs + 1],
                kv_indices=self.cuda_graph_kv_indices_prefill,
                kv_len_arr=self.cuda_graph_kv_lens_prefill[:bs],
                backend="auto"
            )
            seq_lens_sum = seq_lens.sum().item()
            self.indices_updater_prefill.update(
                req_pool_indices,
                seq_lens,
                seq_lens_sum,
                self.kv_fetch_indices,
                prefix_lens=prefix_lens,
                prefill_wrapper_paged=wrapper,
                use_ragged=False,
                spec_info=None,
            )
            self.prefill_cuda_graph_metadata[bs][num_tokens] = wrapper
            self.forward_metadata = PrefillMetadata(wrapper, False)
        else:
            raise ValueError(f"Invalid mode: {forward_batch.forward_mode=}")

    def init_forward_metadata_replay_prefill_cuda_graph(
        self,
        forward_batch: ForwardBatch,
        bs: int,
        num_tokens: int,
        seq_lens: torch.Tensor,
        req_pool_indices: torch.Tensor,
        prefix_lens: torch.Tensor,
    ):
        if forward_batch.forward_mode == ForwardMode.EXTEND:
            seq_lens_sum = seq_lens.sum().item()
            self.indices_updater_prefill.update(
                req_pool_indices[:bs],
                seq_lens[:bs],
                seq_lens_sum,
                self.kv_fetch_indices,
                prefix_lens=prefix_lens[:bs],
                prefill_wrapper_paged=self.prefill_cuda_graph_metadata[bs][num_tokens],
                use_ragged=False,
                spec_info=None,
            )
        else:
            raise ValueError(f"Invalid forward mode: {forward_batch.forward_mode=}")

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInfo],
    ):
        if forward_mode.is_decode_or_idle():
            decode_wrapper = BatchMLAPagedAttentionWrapper(
                self.workspace_buffer,
                use_cuda_graph=True,
                qo_indptr=self.cuda_graph_qo_indptr[: num_tokens + 1],
                kv_indptr=self.cuda_graph_kv_indptr[: num_tokens + 1],
                kv_indices=self.cuda_graph_kv_indices,
                kv_len_arr=self.cuda_graph_kv_lens[:num_tokens],
                backend="auto",
            )

            seq_lens_sum = seq_lens.sum().item()
            self.indices_updater_decode.update(
                req_pool_indices,
                seq_lens,
                seq_lens_sum,
                decode_wrapper=decode_wrapper,
                init_metadata_replay=False,
                spec_info=spec_info,
            )
            self.decode_cuda_graph_metadata[bs] = decode_wrapper
            self.forward_metadata = DecodeMetadata(decode_wrapper)
            decode_wrapper.plan = partial(fast_mla_decode_plan, decode_wrapper)
        elif forward_mode.is_target_verify():
            verify_wrapper = BatchMLAPagedAttentionWrapper(
                self.workspace_buffer,
                use_cuda_graph=True,
                qo_indptr=self.cuda_graph_qo_indptr[: bs + 1],
                kv_indptr=self.cuda_graph_kv_indptr[: bs + 1],
                kv_indices=self.cuda_graph_kv_indices,
                kv_len_arr=self.cuda_graph_kv_lens[:bs],
                backend="auto",
            )
            seq_lens_sum = seq_lens.sum().item()
            self.indices_updater_prefill.update(
                req_pool_indices,
                seq_lens,
                seq_lens_sum,
                self.kv_fetch_indices,
                prefix_lens=None,
                prefill_wrapper_paged=verify_wrapper,
                use_ragged=False,
                spec_info=spec_info,
            )
            self.verify_cuda_graph_metadata[bs] = verify_wrapper
            self.forward_metadata = PrefillMetadata(verify_wrapper, False)
        else:
            raise ValueError(f"Invalid mode: {forward_mode=}")

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInfo],
        seq_lens_cpu: Optional[torch.Tensor],
    ):
        if forward_mode.is_decode_or_idle():
            self.fast_decode_kwargs.update(
                {
                    "qo_indptr_cpu": self.cuda_graph_qo_indptr_cpu[: bs + 1],
                    "kv_indptr_cpu": self.cuda_graph_kv_indptr_cpu[: bs + 1],
                    "kv_len_arr_cpu": self.cuda_graph_kv_lens_cpu[:bs],
                }
            )

            self.indices_updater_decode.update(
                req_pool_indices[:bs],
                seq_lens[:bs],
                seq_lens_sum,
                decode_wrapper=self.decode_cuda_graph_metadata[bs],
                init_metadata_replay=True,
                spec_info=spec_info,
                **self.fast_decode_kwargs,
            )
        elif forward_mode.is_target_verify():
            self.indices_updater_prefill.update(
                req_pool_indices[:bs],
                seq_lens[:bs],
                seq_lens_sum,
                self.kv_fetch_indices,
                prefix_lens=None,
                prefill_wrapper_paged=self.verify_cuda_graph_metadata[bs],
                use_ragged=False,
                spec_info=spec_info,
            )
        else:
            raise ValueError(f"Invalid forward mode: {forward_mode=}")

    def get_cuda_graph_seq_len_fill_value(self):
        return 1

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        q_pe: Optional[torch.Tensor] = None,
        k_pe: Optional[torch.Tensor] = None,
    ):
        if self.forward_metadata.use_ragged:
            # MHA
            return self.forward_normal_extend(q, k, v, layer, save_kv_cache)
        else:
            # MQA
            return self.forward_absorbed_extend(q, k, v, layer, forward_batch, save_kv_cache, q_pe, k_pe)

    def forward_normal_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        save_kv_cache: bool = True,
    ):
        assert not save_kv_cache

        o = self.prefill_wrapper_ragged.forward(
            q,
            k.view(-1, layer.tp_k_head_num, layer.head_dim),
            v.view(-1, layer.tp_k_head_num, layer.v_head_dim),
            causal=True,
            sm_scale=layer.scaling,
            logits_soft_cap=layer.logit_cap,
        )
        return o.view(-1, layer.tp_q_head_num * layer.v_head_dim)

    def forward_absorbed_extend(
        self,
        q_nope: torch.Tensor,
        k_nope: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        q_pe: Optional[torch.Tensor] = None,
        k_pe: Optional[torch.Tensor] = None,
    ):
        return self.forward_absorbed(self.forward_metadata.prefill_wrapper, q_nope, k_nope, v, layer, forward_batch, save_kv_cache, q_pe, k_pe)

    def forward_decode(
        self,
        q_nope: torch.Tensor,
        k_nope: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        q_pe: Optional[torch.Tensor] = None,
        k_pe: Optional[torch.Tensor] = None,
    ):
        return self.forward_absorbed(self.forward_metadata.decode_wrapper, q_nope, k_nope, v, layer, forward_batch, save_kv_cache, q_pe, k_pe)

    # MQA
    def forward_absorbed(
        self,
        wrapper,
        q_nope: torch.Tensor,
        k_nope: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        q_pe: Optional[torch.Tensor] = None,
        k_pe: Optional[torch.Tensor] = None,
    ):
        assert k_nope is not None
        assert k_pe is not None
        assert v is not None
        if save_kv_cache:
            forward_batch.token_to_kv_pool.set_mla_kv_buffer(
                layer, forward_batch.out_cache_loc, k_nope, k_pe
            )

        q_nope = q_nope.view(-1, layer.tp_q_head_num, layer.v_head_dim)
        q_pe = q_pe.view(-1, layer.tp_q_head_num, layer.head_dim - layer.v_head_dim)
        o = q_nope.new_empty(q_nope.shape)

        if self.kv_cache_quant_method == "per_token_head":
            k_nope_cache, k_nope_scale_cache, k_rope_cache = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)
            kv_fetch_indices = self.kv_fetch_indices[:forward_batch.seq_lens_sum]
            if is_sm90_supported():
                k_nope_deq, k_rope = dequantize_ckv_fused_indexed(
                    k_nope_cache.view(torch.float8_e4m3fn), k_rope_cache, k_nope_scale_cache, kv_fetch_indices
                )
            else:
                k_nope_fetch = k_nope_cache[kv_fetch_indices].view(torch.float8_e4m3fn).float()
                k_nope_scale_fetch = k_nope_scale_cache[kv_fetch_indices]
                k_rope_fetch = k_rope_cache[kv_fetch_indices].float()
                k_nope_deq = (k_nope_fetch * k_nope_scale_fetch).to(q_nope.dtype)
                k_rope = (k_rope_fetch * k_nope_scale_fetch).to(q_nope.dtype)
            o, lse = wrapper.run(
                q_nope,
                q_pe,
                k_nope_deq,
                k_rope,
                out=o,
                return_lse=True,
            )
        else:
            k_buf = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id).to(
                q_nope.dtype
            )
            o, lse = wrapper.run(
                q_nope,
                q_pe,
                k_buf[:, :, : layer.v_head_dim],
                k_buf[:, :, layer.v_head_dim :],
                out=o,
                return_lse=True,
            )
        return o, lse


class FlashInferMLAIndicesUpdaterDecode:
    def __init__(self, model_runner: ModelRunner, attn_backend: AttentionBackend):
        # Parse Constants
        self.num_local_heads = (
            model_runner.model_config.num_attention_heads // get_attention_tp_size()
        )
        if global_server_args_dict["enable_mla_l1_5_cache"]:
            self.num_local_heads = model_runner.model_config.num_attention_heads
        self.kv_lora_rank = model_runner.model_config.kv_lora_rank
        self.qk_nope_head_dim = model_runner.model_config.qk_nope_head_dim
        self.qk_rope_head_dim = model_runner.model_config.qk_rope_head_dim
        self.scaling = model_runner.model_config.scaling
        self.data_type = model_runner.kv_cache_dtype
        self.kv_cache_quant_method = model_runner.kv_cache_quant_method
        self.attn_backend = attn_backend

        # Buffers and wrappers
        self.kv_indptr = attn_backend.kv_indptr
        self.req_to_token = model_runner.req_to_token_pool.req_to_token
        self.token_to_kv_pool = model_runner.token_to_kv_pool
        self.q_indptr = attn_backend.q_indptr_decode

    def update(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        decode_wrapper: BatchMLAPagedAttentionWrapper,
        init_metadata_replay: bool = False,
        spec_info: Optional[Union[EagleDraftInput, EagleVerifyInput]] = None,
        **fast_decode_kwargs,
    ):
        decode_wrapper = decode_wrapper or self.decode_wrapper
        self.call_begin_forward(
            decode_wrapper,
            req_pool_indices,
            seq_lens,
            seq_lens_sum,
            self.q_indptr,
            self.kv_indptr,
            init_metadata_replay,
            spec_info,
            **fast_decode_kwargs,
        )

    def call_begin_forward(
        self,
        wrapper: BatchMLAPagedAttentionWrapper,
        req_pool_indices: torch.Tensor,
        paged_kernel_lens: torch.Tensor,
        paged_kernel_lens_sum: int,
        q_indptr: torch.Tensor,
        kv_indptr: torch.Tensor,
        init_metadata_replay: bool = False,
        spec_info: Optional[Union[EagleDraftInput, EagleVerifyInput]] = None,
        **fast_decode_kwargs,
    ):
        bs = len(req_pool_indices)
        q_indptr = q_indptr[: bs + 1]
        kv_lens = paged_kernel_lens.to(torch.int32)
        sm_scale = self.scaling
        if spec_info is None:
            kv_indptr[1 : bs + 1] = torch.cumsum(paged_kernel_lens, dim=0)
            kv_indptr = kv_indptr[: bs + 1]
            kv_indices = (
                torch.empty(paged_kernel_lens_sum, dtype=torch.int32, device="cuda")
                if not init_metadata_replay
                else fast_decode_kwargs["kv_indices"][:paged_kernel_lens_sum]
            )
            create_flashinfer_kv_indices_triton[(bs,)](
                self.req_to_token,
                req_pool_indices,
                paged_kernel_lens,
                kv_indptr,
                None,
                kv_indices,
                self.req_to_token.shape[1],
            )
            if self.kv_cache_quant_method != "per_token_head":
                kv_indices, kv_indptr, kv_lens = _remap_flashinfer_kv_indices(
                    self.token_to_kv_pool, kv_indices, kv_indptr, kv_lens
                )
        else:
            kv_indptr, kv_indices = spec_info.kv_indptr, spec_info.kv_indices

        if init_metadata_replay:
            assert "qo_indptr_cpu" in fast_decode_kwargs
            assert "kv_indptr_cpu" in fast_decode_kwargs
            assert "kv_len_arr_cpu" in fast_decode_kwargs

            # correctness guard for remap path
            assert kv_indptr.numel() == bs + 1, f"{kv_indptr.numel()=} vs {bs + 1=}"
            assert kv_lens.numel() == bs, f"{kv_indptr.numel()=} vs {bs + 1=}, {kv_lens.numel()=} vs {bs=}"
            assert kv_indptr[-1].item() == kv_indices.numel(), f"{kv_indptr[-1].item()=} vs {kv_indices.numel()=}"

            # keep fast-plan CPU inputs consistent with remapped kv tensors
            fast_decode_kwargs["kv_indptr_cpu"][1 : bs + 1].copy_(
                kv_indptr[1:].to("cpu")
            )
            fast_decode_kwargs["kv_len_arr_cpu"][:bs].copy_(kv_lens.to("cpu"))

            wrapper.plan(
                fast_decode_kwargs["qo_indptr_cpu"],
                fast_decode_kwargs["kv_indptr_cpu"],
                kv_indices,
                fast_decode_kwargs["kv_len_arr_cpu"],
                self.num_local_heads,
                self.kv_lora_rank,
                self.qk_rope_head_dim,
                1,
                False,
                sm_scale,
                self.data_type,
                self.data_type,
            )
        else:
            wrapper.plan(
                q_indptr,
                kv_indptr,
                kv_indices,
                kv_lens,
                self.num_local_heads,
                self.kv_lora_rank,
                self.qk_rope_head_dim,
                1,
                False,
                sm_scale,
                self.data_type,
                self.data_type,
            )


class FlashInferMLAIndicesUpdaterPrefill:
    def __init__(self, model_runner: ModelRunner, attn_backend: AttentionBackend):
        # Parse Constants
        self.num_local_heads = (
            model_runner.model_config.num_attention_heads // get_attention_tp_size()
        )
        if global_server_args_dict["enable_mla_l1_5_cache"]:
            self.num_local_heads = model_runner.model_config.num_attention_heads
        self.kv_cache_quant_method = model_runner.kv_cache_quant_method
        self.kv_lora_rank = model_runner.model_config.kv_lora_rank
        self.qk_nope_head_dim = model_runner.model_config.qk_nope_head_dim
        self.qk_rope_head_dim = model_runner.model_config.qk_rope_head_dim
        self.v_head_dim = model_runner.model_config.v_head_dim
        self.scaling = model_runner.model_config.scaling
        self.data_type = model_runner.kv_cache_dtype
        self.q_data_type = model_runner.dtype
        if self.kv_cache_quant_method == "per_token_head":
            self.data_type = model_runner.dtype
        self.attn_backend = attn_backend

        # Buffers and wrappers
        self.kv_indptr = attn_backend.kv_indptr
        self.qo_indptr = attn_backend.qo_indptr
        self.req_to_token = model_runner.req_to_token_pool.req_to_token
        self.token_to_kv_pool = model_runner.token_to_kv_pool
        self.prefill_wrapper_ragged = attn_backend.prefill_wrapper_ragged

    def update(
        self,
        req_pool_indices: torch.Tnesor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        kv_fetch_indices: torch.Tensor,
        prefix_lens: torch.Tensor,
        prefill_wrapper_paged: BatchMLAPagedAttentionWrapper,
        use_ragged: bool,
        spec_info: Optional[Union[EagleDraftInput, EagleVerifyInput]] = None,
    ):
        if use_ragged:
            paged_kernel_lens = prefix_lens
            paged_kernel_lens_sum = paged_kernel_lens.sum().item()
        else:
            paged_kernel_lens = seq_lens
            paged_kernel_lens_sum = seq_lens_sum

        self.call_begin_forward(
            self.prefill_wrapper_ragged,
            prefill_wrapper_paged,
            req_pool_indices,
            paged_kernel_lens,
            paged_kernel_lens_sum,
            kv_fetch_indices,
            seq_lens,
            prefix_lens,
            self.kv_indptr,
            self.qo_indptr,
            use_ragged,
            spec_info,
        )

    def call_begin_forward(
        self,
        wrapper_ragged: BatchPrefillWithRaggedKVCacheWrapper,
        wrapper_paged: BatchMLAPagedAttentionWrapper,
        req_pool_indices: torch.Tensor,
        paged_kernel_lens: torch.Tensor,
        paged_kernel_lens_sum: int,
        kv_fetch_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        prefix_lens: torch.Tensor,
        kv_indptr: torch.Tensor,
        qo_indptr: torch.Tensor,
        use_ragged: bool,
        spec_info: Optional[Union[EagleDraftInput, EagleVerifyInput]] = None,
    ):
        bs = len(seq_lens)
        sm_scale = self.scaling

        if spec_info is None:
            assert len(seq_lens) == len(req_pool_indices)
            kv_indptr[1 : bs + 1] = torch.cumsum(paged_kernel_lens, dim=0)
            kv_indptr = kv_indptr[: bs + 1]
            if self.kv_cache_quant_method == "per_token_head":
                kv_indices = torch.arange(0, paged_kernel_lens_sum, dtype=torch.int32, device=req_pool_indices.device)
                create_flashinfer_kv_indices_triton[(bs,)](
                    self.req_to_token,
                    req_pool_indices,
                    paged_kernel_lens,
                    kv_indptr,
                    None,
                    kv_fetch_indices,
                    self.req_to_token.shape[1],
                )
            else:
                if wrapper_paged._use_cuda_graph:
                    kv_indices = wrapper_paged._kv_indices_buf
                else:
                    kv_indices = torch.empty(
                        paged_kernel_lens_sum,
                        dtype=torch.int32,
                        device=req_pool_indices.device,
                    )
                create_flashinfer_kv_indices_triton[(bs,)](
                    self.req_to_token,
                    req_pool_indices,
                    paged_kernel_lens,
                    kv_indptr,
                    None,
                    kv_indices,
                    self.req_to_token.shape[1],
                )
            qo_indptr[1 : bs + 1] = torch.cumsum(seq_lens - prefix_lens, dim=0)
            qo_indptr = qo_indptr[: bs + 1]
            custom_mask = None
        else:
            # TODO: Support topk > 1 with custom mask
            kv_indices, kv_indptr, qo_indptr, custom_mask = (
                generate_attn_arg_prefill(
                    spec_info.draft_token_num,
                    req_pool_indices,
                    paged_kernel_lens,
                    self.req_to_token,
                )
            )

        if use_ragged:
            # ragged prefill
            wrapper_ragged.begin_forward(
                qo_indptr=qo_indptr,
                kv_indptr=qo_indptr,
                num_qo_heads=self.num_local_heads,
                num_kv_heads=self.num_local_heads,
                head_dim_qk=self.qk_nope_head_dim + self.qk_rope_head_dim,
                head_dim_vo=self.v_head_dim,
                q_data_type=self.q_data_type,
            )
        else:
            # mla paged prefill
            kv_len_arr = kv_indptr[1:] - kv_indptr[:-1]
            wrapper_paged.plan(
                qo_indptr,
                kv_indptr,
                kv_indices,
                kv_len_arr,
                self.num_local_heads,
                self.kv_lora_rank,
                self.qk_rope_head_dim,
                1,
                True,
                sm_scale,
                self.q_data_type,
                self.data_type,
            )


class FlashInferMLAMultiStepDraftBackend:
    """
    Wrap multiple flashinfer mla attention backends as one for multiple consecutive
    draft decoding steps.
    """

    def __init__(
        self,
        model_runner: ModelRunner,
        topk: int,
        speculative_num_steps: int,
    ):
        from sglang.srt.speculative.eagle_utils import generate_draft_decode_kv_indices

        if topk > 1:
            raise ValueError(
                f"Currently Flashinfer MLA only supports topk=1 for speculative decoding"
            )
        self.topk = topk
        self.speculative_num_steps = speculative_num_steps
        self.generate_draft_decode_kv_indices = generate_draft_decode_kv_indices

        max_bs = model_runner.req_to_token_pool.size * self.topk
        self.kv_indptr = torch.zeros(
            (
                self.speculative_num_steps,
                max_bs + 1,
            ),
            dtype=torch.int32,
            device=model_runner.device,
        )
        self.q_indptr_decode = torch.arange(
            0, max_bs + 1, dtype=torch.int32, device=model_runner.device
        )

        self.attn_backends = []
        for i in range(self.speculative_num_steps):
            self.attn_backends.append(
                FlashInferMLAAttnBackend(
                    model_runner,
                    skip_prefill=True,
                    kv_indptr_buf=self.kv_indptr[i],
                    q_indptr_decode_buf=self.q_indptr_decode,
                )
            )

        self.max_context_len = self.attn_backends[0].max_context_len

        # Cached variables for generate_draft_decode_kv_indices
        self.pool_len = model_runner.req_to_token_pool.req_to_token.shape[1]
        self.max_bs = max_bs
        self.max_num_seqs = model_runner.req_to_token_pool.size

    def common_template(
        self,
        forward_batch: ForwardBatch,
        kv_indices_buffer: torch.Tensor,
        call_fn: Callable,
    ):
        num_seqs = forward_batch.batch_size
        bs = self.topk * num_seqs
        seq_lens_sum = forward_batch.seq_lens_sum

        self.generate_draft_decode_kv_indices[
            (self.speculative_num_steps, num_seqs, self.topk)
        ](
            forward_batch.req_pool_indices,
            forward_batch.req_to_token_pool.req_to_token,
            forward_batch.seq_lens,
            kv_indices_buffer,
            self.kv_indptr,
            forward_batch.positions,
            num_seqs,
            kv_indices_buffer.shape[1],
            self.topk,
            self.pool_len,
            self.kv_indptr.shape[1],
            triton.next_power_of_2(self.max_num_seqs),
            triton.next_power_of_2(self.speculative_num_steps),
            triton.next_power_of_2(self.max_bs),
        )

        assert forward_batch.spec_info is not None
        assert isinstance(forward_batch.spec_info, EagleDraftInput)

        for i in range(self.speculative_num_steps - 1):
            forward_batch.spec_info.kv_indptr = self.kv_indptr[i, : bs + 1]
            forward_batch.spec_info.kv_indices = kv_indices_buffer[i][
                : seq_lens_sum * self.topk + bs * (i + 1)
            ]
            call_fn(i, forward_batch)

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        kv_indices = torch.zeros(
            (
                self.speculative_num_steps,
                forward_batch.batch_size * self.topk * self.max_context_len,
            ),
            dtype=torch.int32,
            device="cuda",
        )

        def call_fn(i, forward_batch):
            assert forward_batch.spec_info is not None
            assert isinstance(forward_batch.spec_info, EagleDraftInput)
            forward_batch.spec_info.kv_indptr = (
                forward_batch.spec_info.kv_indptr.clone()
            )
            forward_batch.spec_info.kv_indices = (
                forward_batch.spec_info.kv_indices.clone()
            )
            self.attn_backends[i].init_forward_metadata(forward_batch)

        self.common_template(forward_batch, kv_indices, call_fn)

    def init_cuda_graph_state(self, max_bs: int):
        self.cuda_graph_kv_indices = torch.zeros(
            (self.speculative_num_steps, max_bs * self.max_context_len),
            dtype=torch.int32,
            device="cuda",
        )

        for i in range(self.speculative_num_steps):
            self.attn_backends[i].init_cuda_graph_state(
                max_bs, kv_indices_buf=self.cuda_graph_kv_indices[i]
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

        self.common_template(forward_batch, self.cuda_graph_kv_indices, call_fn)

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
                seq_lens_cpu=forward_batch.seq_lens_cpu,
            )

        self.common_template(forward_batch, self.cuda_graph_kv_indices, call_fn)


class FlashInferMLATBOBackend(FlashInferMLAAttnBackend):
    def __init__(self, model_runner: ModelRunner):
        super().__init__(model_runner)

        self.micro_batch_size = 2
        self.tbo_attn_backends = []
        for i in range(self.micro_batch_size):
            self.tbo_attn_backends.append(
                FlashInferMLAAttnBackend(model_runner)
            )

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        super().init_forward_metadata(forward_batch)
        micro_batches = forward_batch.split_micro_batch()
        if micro_batches:
            self.tbo_attn_backends[0].init_forward_metadata(micro_batches[0].forward_batch)
            self.tbo_attn_backends[1].init_forward_metadata(micro_batches[1].forward_batch)

    def init_cuda_graph_state(self, max_bs: int, kv_indices_buf: Optional[torch.Tensor] = None):
        super().init_cuda_graph_state(max_bs, kv_indices_buf)
        bs_0 = max_bs // 2
        bs_1 = max_bs - bs_0
        self.tbo_attn_backends[0].init_cuda_graph_state(bs_0, kv_indices_buf)
        self.tbo_attn_backends[1].init_cuda_graph_state(bs_1, kv_indices_buf)

    def init_forward_metadata_capture_cuda_graph(self, bs: int, num_tokens: int, req_pool_indices: torch.Tensor,
                                                 seq_lens: torch.Tensor,
                                                 forward_mode: ForwardMode, spec_info: Optional[SpecInfo]):
        super().init_forward_metadata_capture_cuda_graph(bs, num_tokens, req_pool_indices, seq_lens,
                                                         forward_mode, spec_info)

        if forward_mode.is_decode_or_target_verify() and bs >= global_server_args_dict["tbo_min_bs"]:
            bs_0 = bs // 2
            num_tokens_0 = num_tokens // 2
            split_index = bs_0
            req_pool_indices_0 = req_pool_indices[:split_index]
            seq_lens_0 = seq_lens[:split_index]
            self.tbo_attn_backends[0].init_forward_metadata_capture_cuda_graph(bs_0, num_tokens_0, req_pool_indices_0,
                                                                               seq_lens_0, forward_mode,
                                                                               spec_info)

            bs_1 = bs - bs_0
            num_tokens_1 = num_tokens - num_tokens_0
            req_pool_indices_1 = req_pool_indices[split_index:]
            seq_lens_1 = seq_lens[split_index:]
            self.tbo_attn_backends[1].init_forward_metadata_capture_cuda_graph(bs_1, num_tokens_1, req_pool_indices_1,
                                                                               seq_lens_1, forward_mode,
                                                                               spec_info)

    def init_forward_metadata_replay_cuda_graph(self, bs: int, req_pool_indices: torch.Tensor, seq_lens: torch.Tensor,
                                                seq_lens_sum: int,
                                                forward_mode: ForwardMode, spec_info: Optional[SpecInfo],
                                                seq_lens_cpu: Optional[torch.Tensor]):
        super().init_forward_metadata_replay_cuda_graph(bs, req_pool_indices, seq_lens, seq_lens_sum,
                                                        forward_mode, spec_info, seq_lens_cpu)

        if forward_mode.is_decode_or_target_verify() and bs >= global_server_args_dict["tbo_min_bs"]:
            bs_0 = bs // 2
            split_index = bs_0
            req_pool_indices_0 = req_pool_indices[:split_index]
            seq_lens_0 = seq_lens[:split_index]
            seq_lens_cpu_0 = seq_lens_cpu[:split_index] if seq_lens_cpu is not None else None
            seq_lens_sum_0 = seq_lens_0.sum()
            self.tbo_attn_backends[0].init_forward_metadata_replay_cuda_graph(bs_0, req_pool_indices_0, seq_lens_0,
                                                                              seq_lens_sum_0, forward_mode,
                                                                              spec_info, seq_lens_cpu_0)

            bs_1 = bs - bs_0
            req_pool_indices_1 = req_pool_indices[split_index:]
            seq_lens_1 = seq_lens[split_index:]
            seq_lens_cpu_1 = seq_lens_cpu[split_index:] if seq_lens_cpu is not None else None
            seq_lens_sum_1 = seq_lens_1.sum()
            self.tbo_attn_backends[1].init_forward_metadata_replay_cuda_graph(bs_1, req_pool_indices_1, seq_lens_1,
                                                                              seq_lens_sum_1, forward_mode,
                                                                              spec_info, seq_lens_cpu_1)


def fast_mla_decode_plan(
    self,
    qo_indptr_cpu: torch.Tensor,
    kv_indptr_cpu: torch.Tensor,
    kv_indices: torch.Tensor,
    kv_len_arr_cpu: torch.Tensor,
    num_heads: int,
    head_dim_ckv: int,
    head_dim_kpe: int,
    page_size: int,
    causal: bool,
    sm_scale: float,
    q_data_type: torch.dtype,
    kv_data_type: torch.dtype,
) -> None:
    """A faster version of BatchMLAPagedAttentionWrapper::plan,
    for skipping the stream synchronization in original plan function during
    cuda graph replaying.
    """
    self._causal = causal
    self._page_size = page_size
    self._sm_scale = sm_scale
    with self.device as device:
        try:
            # Standard version with just the required arguments (no use_profiler)
            self._cached_module.plan(
                self._float_workspace_buffer,
                self._int_workspace_buffer,
                self._pin_memory_int_workspace_buffer,
                qo_indptr_cpu,
                kv_indptr_cpu,
                kv_len_arr_cpu,
                num_heads,
                head_dim_ckv,
                causal,
            )
        except Exception as e:
            raise RuntimeError(f"Error in alternate MLA plan: {e}")
