from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from sglang.srt.disaggregation.utils import StepCounter
    from sglang.srt.layers.attention.radix_attention import RadixAttention
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
    from sglang.srt.speculative.eagle_utils import EagleDraftInput, EagleVerifyInput


class AttentionBackend(ABC):
    """The base class of attention backends"""

    @abstractmethod
    def init_forward_metadata(self, forward_batch: ForwardBatch):
        """Init the metadata for a forward pass."""
        raise NotImplementedError()

    def init_cuda_graph_state_prefill(self, max_bs: int, kv_indices_buf=None):
        """Init the global shared states for prefill cuda graph.
        Default no-op; backends that support prefill cuda graphs should override."""
        pass

    def init_forward_metadata_prefill_capture_cuda_graph(
        self,
        forward_batch: ForwardBatch,
        bs: int,
        num_tokens: int,
        seq_lens: torch.Tensor,
        prefix_lens: torch.Tensor,
        req_pool_indices: torch.Tensor,
    ):
        """Init the metadata for a prefill forward pass for capturing a cuda graph.
        Default no-op; backends that support prefill cuda graphs should override."""
        pass

    def init_forward_metadata_replay_prefill_cuda_graph(
        self,
        forward_batch: ForwardBatch,
        bs: int,
        num_tokens: int,
        seq_lens: torch.Tensor,
        req_pool_indices: torch.Tensor,
        prefix_lens: torch.Tensor,
    ):
        """Init the metadata for a prefill forward pass for replaying a cuda graph.
        Default no-op; backends that support prefill cuda graphs should override."""
        pass

    def init_cuda_graph_state(self, max_bs: int):
        """Init the global shared states for cuda graph."""
        raise NotImplementedError()

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        forward_mode: ForwardMode,
        spec_info: EagleDraftInput | EagleVerifyInput | None,
    ):
        """Init the metadata for a forward pass for capturing a cuda graph."""
        raise NotImplementedError()

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        forward_mode: ForwardMode,
        spec_info: EagleDraftInput | EagleVerifyInput | None,
    ):
        """Init the metadata for a forward pass for replying a cuda graph."""
        raise NotImplementedError()

    def get_cuda_graph_seq_len_fill_value(self):
        """Get the fill value for padded seq lens. Typically, it is 0 or 1."""
        raise NotImplementedError()

    def register_step_counter(self, step_counter: StepCounter):
        self.step_counter = step_counter

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
        if forward_batch.forward_mode.is_decode():
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
            if not forward_batch.forward_mode.is_idle() and getattr(self, "step_counter", None) and not save_kv_cache:
                self.step_counter.record_cache()

            ret = self.forward_extend(
                q,
                k,
                v,
                layer,
                forward_batch,
                save_kv_cache=save_kv_cache,
                **kwargs,
            )

            if not forward_batch.forward_mode.is_idle() and getattr(self, "step_counter", None) and save_kv_cache:
                self.step_counter.record_cache()
            return ret

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
        """Run a forward for decode."""
        raise NotImplementedError()

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
        """Run a forward for extend."""
        raise NotImplementedError()
