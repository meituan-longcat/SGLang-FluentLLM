# Copyright 2023-2025 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""ViT CUDA Graph Runner class.
"""

from __future__ import annotations

import bisect
import inspect
import logging

from typing import Any, Dict, Hashable, List, Optional, Tuple

import torch
import torch.nn as nn

from sglang.srt.distributed.parallel_state import get_tp_group, graph_capture
from sglang.srt.layers.attention.vision import VisionAttention
from sglang.srt.env import global_server_args_dict
from sglang.srt.model_executor.cuda_graph_runner import freeze_gc
from sglang.srt.utils import get_available_gpu_memory

logger = logging.getLogger(__name__)


# Default capture seq-lens for ViT CUDA graphs.
_DEFAULT_CAPTURE_SEQ_LENS = [128, 256, 512, 1024, 2048, 4096]


def get_vit_capture_seq_lens(
    max_seq_len: Optional[int] = None,
    base_ladder: Optional[List[int]] = None,
) -> List[int]:
    """Return the sorted list of ViT token-count ladder to capture.

    Args:
        max_seq_len: upper bound (inclusive).  Entries above this are dropped.
            If None, all entries in *base_ladder* are kept.
        base_ladder: explicit list; defaults to ``_DEFAULT_CAPTURE_SEQ_LENS``.

    Returns:
        Sorted ascending list of seq-lens to capture.
    """
    ladder = sorted(base_ladder if base_ladder is not None else _DEFAULT_CAPTURE_SEQ_LENS)
    if max_seq_len is not None:
        ladder = [s for s in ladder if s <= max_seq_len]
        # Always include max_seq_len itself so the maximum possible input has a graph.
        if not ladder or ladder[-1] < max_seq_len:
            ladder.append(max_seq_len)
    return ladder


class ViTCudaGraphRunner:
    """Generic ViT CUDA Graph Runner with padding support.

    This runner captures the "blocks + merger/final_layernorm + deepstack merger
    (optional)" part of a vision transformer into CUDA graphs keyed by padded
    token counts, and replays the nearest larger (or equal) graph for any input.

    Padding scheme
    --------------
    A fixed ladder of ``capture_seq_lens`` (e.g. [256, 512, 1024, ...]) is
    captured at construction time.  When ``run()`` receives an input of actual token count S:

    1. ``padded_S = capture_seq_lens[bisect_left(capture_seq_lens, S)]``
    2. ``hidden_states`` is zero-padded to ``[padded_S, ...]``.
    3. ``rope_freqs_cis`` (or cos/sin) is zero-padded to ``[padded_S, ...]``.
    4. ``cu_seqlens`` gets an extra "padding image" appended:
       ``[0, s1, ..., S, padded_S]`` so attention only touches valid tokens.
    5. After graph replay, only ``output[:S]`` is returned (unpad).

    The cu_seqlens buffer for each padded_key is pre-allocated with size
    ``max_images + 2`` (one extra slot for the padding image sentinel).
    ``max_images`` is configurable; defaults to ``_DEFAULT_MAX_IMAGES``.

    Supported RoPE formats (mutually exclusive, pass exactly one per call):
      1. ``position_embeddings``: ``(cos, sin)`` tuple of real-valued tensors
         ``[S, D]`` — used by Qwen2.5-VL / Qwen3-VL.
      2. ``rotary_pos_emb_cos`` / ``rotary_pos_emb_sin``: separate real-valued
         tensors — alternative Qwen2.5-VL interface.
      3. ``rope_freqs_cis``: complex64 tensor ``[S, head_dim//2]`` — used by
         Kimi K2.5 (MoonViT).

    Optional for Qwen2.5 windowed attention:
      - vit.fullatt_block_indexes: Sequence[int]
      - run() provides both cu_seqlens and cu_window_seqlens

    Optional for Qwen3 deepstack:
      - vit.deepstack_visual_indexes: Sequence[int]
      - vit.deepstack_merger_list: nn.ModuleList

    Output layer (auto-detected via ``getattr``):
      - If ``vit`` has a ``merger`` attribute, ``merger(y)`` is called.
      - Otherwise ``final_layernorm(y)`` is called (MoonViT style).
    """

    # Maximum number of real images (sequences) in a single batch.
    # The cu_seqlens buffer is allocated with ``max_images + 2`` slots so there
    # is always room for the trailing padding-image entry.
    _DEFAULT_MAX_IMAGES: int = 32

    def __init__(
        self,
        vit: nn.Module,
        warmup_iters: int = 3,
        capture_seq_lens: Optional[List[int]] = None,
        max_images: int = _DEFAULT_MAX_IMAGES,
        lazy_capture: bool = True,
    ) -> None:
        """
        Args:
            vit: Vision encoder module.
            warmup_iters: Number of eager warm-up passes before graph capture.
            capture_seq_lens: Explicit list of token-count ladder to capture.
                If None, uses ``_DEFAULT_CAPTURE_SEQ_LENS``.
            max_images: Maximum number of images (sequences) per batch.  Used
                to pre-allocate the cu_seqlens buffer with enough slots.
            lazy_capture: If True (default), graphs are captured lazily on the
                first ``run()`` call for each padded_key.  If False, all graphs
                in ``capture_seq_lens`` are captured at construction time.
        """
        self.vit = vit
        self._warmup_iters = warmup_iters
        self._max_images = max_images
        self._lazy_capture = lazy_capture

        # Sorted ascending list of padded seq-lens to capture.
        self.capture_seq_lens: List[int] = sorted(
            capture_seq_lens if capture_seq_lens is not None else _DEFAULT_CAPTURE_SEQ_LENS
        )

        # graph_key (= padded seq_len) -> buffers / graphs
        self.block_input: Dict[int, torch.Tensor] = {}
        self.block_ws: Dict[int, torch.Tensor] = {}
        self.block_graphs: Dict[int, torch.cuda.CUDAGraph] = {}
        self.block_output: Dict[int, torch.Tensor] = {}

        # cu_seqlens stable-address buffers (size = max_images + 2 per padded_key).
        # The "+2" leaves room for: the leading 0, real images, and one padding sentinel.
        # Content is refreshed via copy_() before every replay.
        self.cu_full_len: Dict[int, torch.Tensor] = {}
        self.cu_window_len: Dict[int, torch.Tensor] = {}
        self.cu_full_len_kk: Dict[int, torch.Tensor] = {}
        self.cu_window_len_kk: Dict[int, torch.Tensor] = {}
        # max_seqlen is fixed at capture time (= padded_key for the padding graph).
        self.max_full_len: Dict[int, int] = {}
        self.max_window_len: Dict[int, int] = {}

        # rotary position buffers -----------------------------------------------
        # Format A: real-valued (cos, sin) — Qwen2.5 / Qwen3.
        # Pre-allocated with shape [max(capture_seq_lens), head_dim] once.
        self.sin_cos_ws: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        self.max_context_len = getattr(vit, "max_context_len", None)

        # Format B: complex64 rope_freqs_cis — Kimi K2.5 / MoonViT.
        # Keyed by padded_key; pre-allocated with shape [padded_key, head_dim//2].
        self.rope_freqs_ws: Dict[int, torch.Tensor] = {}

        # Qwen2.5-VL specific variable.
        self._fullatt_block_indexes = set(getattr(vit, "fullatt_block_indexes", ()))

        # Qwen3-VL specific variables.
        self._deepstack_visual_indexes = list(
            getattr(vit, "deepstack_visual_indexes", []) or []
        )
        self._deepstack_merger_list = getattr(vit, "deepstack_merger_list", None)

        first_blk = vit.blocks[0]
        self._blk_accepts_output_ws = (
            "output_ws" in inspect.signature(first_blk.forward).parameters
        )
        self._blk_accepts_max_seqlen = (
            "max_seqlen" in inspect.signature(first_blk.forward).parameters
        )
        self._blk_accepts_rope_freqs_cis = (
            "rope_freqs_cis" in inspect.signature(first_blk.forward).parameters
        )

        self._attn: Optional[VisionAttention] = getattr(first_blk, "attn", None)
        self._attn_backend = getattr(self._attn, "qkv_backend", None)

        # Output layer: prefer merger (Qwen style), fall back to final_layernorm (MoonViT style).
        self._merger = getattr(vit, "merger", None)
        self._final_layernorm = getattr(vit, "final_layernorm", None)
        if self._merger is None and self._final_layernorm is None:
            raise ValueError(
                "ViTCudaGraphRunner: vit must have either a 'merger' or a "
                "'final_layernorm' attribute as the output projection layer."
            )

        # Shared CUDA Graph memory pool: reused across all graph_keys
        self._graph_memory_pool: Optional[Any] = None

        self._capture_stream: Optional[torch.cuda.Stream] = None

    @property
    def device(self) -> torch.device:
        return self.vit.device

    @property
    def dtype(self) -> torch.dtype:
        return self.vit.dtype

    # ------------------------------------------------------------------
    # Graph-key / padding helpers
    # ------------------------------------------------------------------

    def _find_padded_key(self, seq_len: int) -> int:
        """Return the smallest capture_seq_len >= seq_len.

        If seq_len exceeds all captured sizes, return seq_len itself (will be
        captured lazily on first use, just like the old per-shape behaviour).
        """
        idx = bisect.bisect_left(self.capture_seq_lens, seq_len)
        if idx < len(self.capture_seq_lens):
            return self.capture_seq_lens[idx]
        # Fallback: capture a new graph for this exact size.
        return seq_len

    def _pad_1d(self, t: torch.Tensor, padded_len: int) -> torch.Tensor:
        """Zero-pad a 1-D or 2-D tensor along dim-0 to ``padded_len``."""
        s = t.shape[0]
        if s == padded_len:
            return t
        pad_size = padded_len - s
        return torch.nn.functional.pad(t, (0,) * (2 * (t.dim() - 1)) + (0, pad_size))

    def _pad_cu_seqlens(
        self,
        cu_seqlens: torch.Tensor,
        padded_S: int,
    ) -> torch.Tensor:
        """Append a padding-image entry to cu_seqlens.

        Given real cu_seqlens = [0, s1, s2, ..., S] (B+1 elements), returns
        [0, s1, s2, ..., S, padded_S] (B+2 elements) on CPU (int32).

        If S == padded_S no padding image is needed; returns cu_seqlens as-is.
        """
        real_S = int(cu_seqlens[-1].item())
        if real_S == padded_S:
            return cu_seqlens
        pad_entry = torch.tensor([padded_S], dtype=torch.int32, device=cu_seqlens.device)
        return torch.cat([cu_seqlens, pad_entry], dim=0)

    # ------------------------------------------------------------------
    # Rotary embedding helpers
    # ------------------------------------------------------------------

    def _ensure_sin_cos_ws(self, seq_len: int, head_dim: int):
        """Ensure sin_cos_ws is large enough for seq_len tokens."""
        if self.sin_cos_ws is None:
            max_shape = self.max_context_len or seq_len
            max_shape = max(max_shape, seq_len)
            cos_ws = torch.empty(max_shape, head_dim, dtype=self.dtype, device=self.device)
            sin_ws = torch.empty(max_shape, head_dim, dtype=self.dtype, device=self.device)
            self.sin_cos_ws = (cos_ws, sin_ws)
        elif self.sin_cos_ws[0].size(0) < seq_len:
            max_shape = max(self.sin_cos_ws[0].size(0) * 2, seq_len)
            cos_ws = torch.empty(max_shape, head_dim, dtype=self.dtype, device=self.device)
            sin_ws = torch.empty(max_shape, head_dim, dtype=self.dtype, device=self.device)
            self.sin_cos_ws = (cos_ws, sin_ws)

    # ------------------------------------------------------------------
    # cu_seqlens buffer helpers
    # ------------------------------------------------------------------

    def _alloc_cu_seqlens_buf(self, padded_key: int) -> None:
        """Pre-allocate stable-address cu_seqlens buffers for *padded_key*.

        The buffer is sized ``max_images + 2`` (room for B real images + the
        padding sentinel).  Initial content: a single "virtual image" that
        spans the entire padded window:  [0, padded_key, padded_key, ...].
        """
        buf_len = self._max_images + 2
        # cu_full_len: indptr [0, padded_key, padded_key, ...]
        cu = torch.zeros(buf_len, dtype=torch.int32, device=self.device)
        cu[1:] = padded_key  # all slots point to padded_key
        self.cu_full_len[padded_key] = cu

        # cu_full_len_kk: per-seq lengths [padded_key, 0, 0, ...]
        cu_kk = torch.zeros(buf_len - 1, dtype=torch.int32, device=self.device)
        cu_kk[0] = padded_key
        self.cu_full_len_kk[padded_key] = cu_kk

        # max_full_len: fixed at padded_key (the padding image can be as large as padded_key)
        self.max_full_len[padded_key] = padded_key

        if self._fullatt_block_indexes:
            # Qwen2.5-VL: also alloc window seqlens buffer (same shape, same init)
            cu_w = cu.clone()
            cu_kk_w = cu_kk.clone()
            self.cu_window_len[padded_key] = cu_w
            self.cu_window_len_kk[padded_key] = cu_kk_w
            self.max_window_len[padded_key] = padded_key

    def _update_cu_seqlens(
        self,
        padded_key: int,
        cu_seqlens_padded: torch.Tensor,  # [B+2] on device, already has padding sentinel
        window: bool = False,
    ) -> None:
        """Copy fresh cu_seqlens (and derived kk) into stable buffer before replay.

        ``cu_seqlens_padded`` has shape ``[B+2]`` where the last element is padded_key.
        The stable buffer has shape ``[max_images+2]``.  We copy the first ``B+2``
        elements and then fill the rest with ``padded_key`` (harmless sentinels).
        """
        n = cu_seqlens_padded.shape[0]  # B+2
        buf_len = self._max_images + 2

        if n > buf_len:
            raise RuntimeError(
                f"ViTCudaGraphRunner: cu_seqlens length {n} exceeds the pre-allocated "
                f"buffer size {buf_len} (max_images={self._max_images}). "
                "Increase max_images when constructing ViTCudaGraphRunner."
            )

        if window:
            cu_buf = self.cu_window_len[padded_key]
            cu_kk_buf = self.cu_window_len_kk[padded_key]
        else:
            cu_buf = self.cu_full_len[padded_key]
            cu_kk_buf = self.cu_full_len_kk[padded_key]

        # Fill leading B+2 entries with real + padding data.
        cu_buf[:n].copy_(cu_seqlens_padded)
        # Fill remaining slots with padded_key (acts as zero-length sequences).
        if n < buf_len:
            cu_buf[n:].fill_(padded_key)

        # Recompute kk (diff)
        cu_kk_all = cu_buf[1:buf_len] - cu_buf[0:buf_len - 1]
        cu_kk_buf.copy_(cu_kk_all)

    # ------------------------------------------------------------------
    # Block call / output helpers
    # ------------------------------------------------------------------

    def _build_cu_seq_len_ws(
        self,
        graph_key: int,
        override_backend: str,
        full: bool = True,
    ) -> list:
        """Build the cu_seq_len_ws list passed to each block's forward method."""
        if full:
            cu = self.cu_full_len[graph_key]
            cu_kk = self.cu_full_len_kk[graph_key]
            max_len = self.max_full_len[graph_key]
        else:
            cu = self.cu_window_len[graph_key]
            cu_kk = self.cu_window_len_kk[graph_key]
            max_len = self.max_window_len[graph_key]

        if override_backend == "triton_attn":
            return [cu, cu_kk, max_len]
        elif override_backend == "fa3":
            return [cu, max_len]
        else:
            raise RuntimeError(
                f"ViTCudaGraphRunner: unsupported attention backend '{override_backend}'. "
                "Set mm_attention_backend to 'triton_attn' or 'fa3'."
            )

    def _call_block(
        self,
        blk: nn.Module,
        x: torch.Tensor,
        graph_key: int,
        layer_num: int,
        override_backend: str,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        rotary_pos_emb_cos: Optional[torch.Tensor] = None,
        rotary_pos_emb_sin: Optional[torch.Tensor] = None,
        rope_freqs_cis: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Call a single transformer block with the appropriate arguments."""
        if self._fullatt_block_indexes:
            use_full = layer_num in self._fullatt_block_indexes
        else:
            use_full = True
        cu_seq_len_ws = self._build_cu_seq_len_ws(graph_key, override_backend, full=use_full)
        max_len = self.max_full_len[graph_key] if use_full else self.max_window_len[graph_key]

        kwargs: dict = {"cu_seqlens": cu_seq_len_ws}
        if self._blk_accepts_max_seqlen:
            kwargs["max_seqlen"] = max_len
        if self._blk_accepts_output_ws:
            kwargs["output_ws"] = self.block_ws[graph_key]

        if rope_freqs_cis is not None:
            kwargs["rope_freqs_cis"] = rope_freqs_cis
        elif position_embeddings is not None:
            kwargs["position_embeddings"] = position_embeddings
        elif rotary_pos_emb_cos is not None and rotary_pos_emb_sin is not None:
            kwargs["rotary_pos_emb_cos"] = rotary_pos_emb_cos
            kwargs["rotary_pos_emb_sin"] = rotary_pos_emb_sin

        return blk(x, **kwargs)

    def _apply_output_layer(self, y: torch.Tensor, graph_key: int) -> torch.Tensor:
        """Apply merger or final_layernorm."""
        if self._merger is not None:
            return self._merger(y)
        return self._final_layernorm(y)

    # ------------------------------------------------------------------
    # Graph capture internals
    # ------------------------------------------------------------------

    def _warmup(
        self,
        graph_key: int,
        override_backend: str,
        stream: torch.cuda.Stream,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        rotary_pos_emb_cos: Optional[torch.Tensor] = None,
        rotary_pos_emb_sin: Optional[torch.Tensor] = None,
        rope_freqs_cis: Optional[torch.Tensor] = None,
    ) -> None:
        """Run ``_warmup_iters`` eager passes on the capture stream.

        ``stream`` must be the same stream used for graph capture (obtained from
        ``graph_capture()``), so that CUDA sees all prior kernels before
        starting to record and all TP collectives (pynccl) stay on one stream.
        """
        if self._warmup_iters <= 0:
            return

        with torch.cuda.stream(stream):
            for _ in range(self._warmup_iters):
                y = None
                for layer_num, blk in enumerate(self.vit.blocks):
                    x_in = self.block_input[graph_key] if layer_num == 0 else y
                    y = self._call_block(
                        blk, x_in, graph_key, layer_num, override_backend,
                        position_embeddings=position_embeddings,
                        rotary_pos_emb_cos=rotary_pos_emb_cos,
                        rotary_pos_emb_sin=rotary_pos_emb_sin,
                        rope_freqs_cis=rope_freqs_cis,
                    )
        stream.synchronize()

    def _create_graph(
        self,
        graph_key: int,
        stream: torch.cuda.Stream,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        rotary_pos_emb_cos: Optional[torch.Tensor] = None,
        rotary_pos_emb_sin: Optional[torch.Tensor] = None,
        rope_freqs_cis: Optional[torch.Tensor] = None,
    ) -> None:
        """Capture a CUDA graph for *graph_key*.
        """
        graph = torch.cuda.CUDAGraph()
        vit = self.vit

        override_backend = global_server_args_dict.get("mm_attention_backend")

        with torch.cuda.graph(
            graph,
            pool=self._graph_memory_pool,
            stream=stream,
        ):
            y = None
            deepstack_outs: List[torch.Tensor] = []
            deepstack_capture_idx = 0

            for layer_num, blk in enumerate(vit.blocks):
                x_in = self.block_input[graph_key] if layer_num == 0 else y
                y = self._call_block(
                    blk, x_in, graph_key, layer_num, override_backend,
                    position_embeddings=position_embeddings,
                    rotary_pos_emb_cos=rotary_pos_emb_cos,
                    rotary_pos_emb_sin=rotary_pos_emb_sin,
                    rope_freqs_cis=rope_freqs_cis,
                )

                # Optional deepstack support (Qwen3-VL)
                if (
                    self._deepstack_visual_indexes
                    and layer_num in self._deepstack_visual_indexes
                ):
                    if self._deepstack_merger_list is None:
                        raise RuntimeError(
                            "deepstack_visual_indexes exists but deepstack_merger_list is missing."
                        )
                    deepstack_out = self._deepstack_merger_list[deepstack_capture_idx](y)
                    deepstack_outs.append(deepstack_out)
                    deepstack_capture_idx += 1

            main_out = self._apply_output_layer(y, graph_key)

            # Write results into the pre-allocated stable-address block_output buffer
            if deepstack_outs:
                tmp = torch.cat([main_out] + deepstack_outs, dim=1)
                self.block_output[graph_key].copy_(tmp.reshape(self.block_output[graph_key].shape))
            else:
                self.block_output[graph_key].copy_(main_out.reshape(self.block_output[graph_key].shape))

        # Update the shared pool handle so the next (smaller) graph reuses it.
        self._graph_memory_pool = graph.pool()
        self.block_graphs[graph_key] = graph

    def _alloc_buffers(self, graph_key: int, x_3d: torch.Tensor) -> None:
        """Pre-allocate all stable-address buffers for *graph_key* (padded_key).

        ``x_3d`` is used only for shape/dtype/device inference; its *shape[0]* is
        the *actual* seq_len and may differ from *graph_key*.
        """
        if graph_key in self.block_input:
            return  # already allocated

        vit = self.vit
        attn_module: VisionAttention = vit.blocks[0].attn
        num_heads = attn_module.num_attention_heads_per_partition
        attn_head_dim = attn_module.head_size

        # block_input / block_output / block_ws use graph_key (padded) as dim-0
        extra_dims = x_3d.shape[1:]  # () for 1-D, (H,) for 2-D, (1, H) for 3-D
        padded_shape = (graph_key,) + extra_dims
        self.block_input[graph_key] = torch.zeros(
            padded_shape, dtype=x_3d.dtype, device=self.device
        ).contiguous()
        self.block_output[graph_key] = torch.zeros(
            padded_shape, dtype=x_3d.dtype, device=self.device
        ).contiguous()
        # block_ws is only needed by triton_attn backend (output_ws path).
        # fa3 backend does not use output_ws, so skip the allocation to save memory.
        if self._blk_accepts_output_ws:
            self.block_ws[graph_key] = torch.zeros(
                graph_key, num_heads, attn_head_dim,
                device=self.device, dtype=self.dtype,
            )

        # Allocate cu_seqlens buffers (fixed size = max_images + 2)
        self._alloc_cu_seqlens_buf(graph_key)

    def create_graph(
        self,
        x_3d: torch.Tensor,
        cu_seqlens: torch.Tensor,
        cu_window_seqlens: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        rotary_pos_emb_cos: Optional[torch.Tensor] = None,
        rotary_pos_emb_sin: Optional[torch.Tensor] = None,
        rope_freqs_cis: Optional[torch.Tensor] = None,
    ) -> int:
        """Capture a graph for the padded seq-len corresponding to x_3d.shape[0].

        Returns the graph_key (= padded seq-len).
        """
        real_S = x_3d.shape[0]
        graph_key = self._find_padded_key(real_S)

        if graph_key in self.block_graphs:
            return graph_key

        if torch.cuda.is_available():
            avail_mem_before = get_available_gpu_memory(
                self.device.type, torch.cuda.current_device(), empty_cache=False
            )
            logger.info(
                f"[ViTCudaGraphRunner] before capture graph_key={graph_key}: "
                f"available GPU memory = {avail_mem_before:.2f} GB"
            )

        # Allocate all buffers (sized for graph_key, not real_S)
        self._alloc_buffers(graph_key, x_3d)

        # Prime cu_seqlens buffer with a single virtual image spanning padded_key.
        # (This is the content that will be recorded into the graph.)
        # Actual per-replay content is written by _update_cu_seqlens in run().
        # Already done by _alloc_cu_seqlens_buf; no extra action needed here.

        override_backend = global_server_args_dict.get("mm_attention_backend")

        if rope_freqs_cis is not None:
            # Kimi K2.5 / MoonViT: complex64 rope_freqs_cis — shape [S, head_dim//2]
            head_dim_half = rope_freqs_cis.shape[1]
            if graph_key not in self.rope_freqs_ws:
                self.rope_freqs_ws[graph_key] = torch.zeros(
                    graph_key, head_dim_half,
                    dtype=rope_freqs_cis.dtype, device=self.device,
                ).contiguous()
            # Fill first real_S rows; rest stay zero (padding tokens)
            self.rope_freqs_ws[graph_key][:real_S].copy_(rope_freqs_cis[:real_S])
            stable_rope = self.rope_freqs_ws[graph_key]
            self._warmup(graph_key, override_backend, rope_freqs_cis=stable_rope)
            self._create_graph(graph_key=graph_key, rope_freqs_cis=stable_rope)

        elif position_embeddings is not None:
            head_dim = position_embeddings[0].shape[1]
            self._ensure_sin_cos_ws(graph_key, head_dim)
            used_cos_ws = self.sin_cos_ws[0][:graph_key, :]
            used_sin_ws = self.sin_cos_ws[1][:graph_key, :]
            used_cos_ws[:real_S].copy_(position_embeddings[0])
            used_cos_ws[real_S:].zero_()
            used_sin_ws[:real_S].copy_(position_embeddings[1])
            used_sin_ws[real_S:].zero_()
            persist_pe = (used_cos_ws, used_sin_ws)
            self._warmup(graph_key, override_backend, position_embeddings=persist_pe)
            self._create_graph(graph_key=graph_key, position_embeddings=persist_pe)

        elif rotary_pos_emb_cos is not None and rotary_pos_emb_sin is not None:
            head_dim = rotary_pos_emb_cos.shape[1]
            self._ensure_sin_cos_ws(graph_key, head_dim)
            used_cos_ws = self.sin_cos_ws[0][:graph_key, :]
            used_sin_ws = self.sin_cos_ws[1][:graph_key, :]
            used_cos_ws[:real_S].copy_(rotary_pos_emb_cos)
            used_cos_ws[real_S:].zero_()
            used_sin_ws[:real_S].copy_(rotary_pos_emb_sin)
            used_sin_ws[real_S:].zero_()
            self._warmup(
                graph_key, override_backend,
                rotary_pos_emb_cos=used_cos_ws, rotary_pos_emb_sin=used_sin_ws,
            )
            self._create_graph(
                graph_key=graph_key,
                rotary_pos_emb_cos=used_cos_ws, rotary_pos_emb_sin=used_sin_ws,
            )

        if torch.cuda.is_available():
            avail_mem_after = get_available_gpu_memory(
                self.device.type, torch.cuda.current_device(), empty_cache=False
            )
            logger.info(
                f"[ViTCudaGraphRunner] after  capture graph_key={graph_key}: "
                f"available GPU memory = {avail_mem_after:.2f} GB"
            )

        return graph_key

    # ------------------------------------------------------------------
    # Eager pre-capture (参考 CudaGraphRunner.capture())
    # ------------------------------------------------------------------

    def warmup_capture(
        self,
        *,
        # RoPE format A: real-valued (cos, sin) — Qwen2.5 / Qwen3
        # 提供 head_dim 即可；cos/sin 用 dummy zeros 填充。
        rope_head_dim: Optional[int] = None,
        use_rotary_pos_emb_cos_sin: bool = False,  # True → cos/sin 分开传；False → position_embeddings tuple
        # RoPE format B: complex64 rope_freqs_cis — Kimi K2.5
        rope_freqs_cis_head_dim_half: Optional[int] = None,
        # hidden state 的额外维度（除 seq_len 外），用于决定 block_input shape
        # Qwen: [S, 1, H]  → extra_dims=(1, hidden_size)
        # Kimi: [S, H]     → extra_dims=(hidden_size,)
        hidden_size: Optional[int] = None,
        extra_dims: Optional[Tuple[int, ...]] = None,
    ) -> None:
        """提前录制所有 ``capture_seq_lens`` 档位的 CUDA Graph。

        与 ``CudaGraphRunner.capture()`` 类似，按**降序**录制（大档位先录，
        使小档位能复用大档位申请的 memory pool），从而减少显存碎片。

        只需提供 RoPE 的格式描述（head_dim 等），无需真实输入数据；
        所有 buffer 用 zeros/dummy 值填充，与录图时一致。

        Args:
            rope_head_dim            : (Format A) attention head dim（cos/sin 的 last dim）。
            use_rotary_pos_emb_cos_sin: True → 以 ``rotary_pos_emb_cos/sin`` 两个 tensor
                                        分开传入；False → 以 ``position_embeddings=(cos,sin)``
                                        tuple 传入。仅在 rope_head_dim 不为 None 时有效。
            rope_freqs_cis_head_dim_half: (Format B) complex64 rope 的 head_dim // 2。
            hidden_size              : ViT 隐层宽度，用于推断 ``extra_dims``（若未直接给出）。
            extra_dims               : block_input 在 seq_len 之后的维度 tuple。
                                       Qwen 系列为 (1, hidden_size)，Kimi 系列为 (hidden_size,)。
                                       若提供了 hidden_size 且未提供 extra_dims，则根据模型类型自动推断：
                                         - 有 merger → (1, hidden_size)
                                         - 无 merger → (hidden_size,)

        Example (Kimi K2.5 / MoonViT)::

            runner.warmup_capture(
                rope_freqs_cis_head_dim_half=vit.rope_2d.dim // 2,
                hidden_size=vit.blocks[0].hidden_dim,
            )

        Example (Qwen2.5-VL)::

            runner.warmup_capture(
                rope_head_dim=vit.blocks[0].attn.head_size,
                hidden_size=vit.config.hidden_size,
            )
        """
        # ---- 推断 extra_dims ----
        if extra_dims is None:
            if hidden_size is None:
                # 尝试从 vit 模块自动推断
                attn = self.vit.blocks[0].attn
                hidden_size = (
                    attn.num_attention_heads_per_partition * self.tp_size_for_warmup
                    * attn.head_size
                ) if hasattr(attn, "num_attention_heads_per_partition") else None
            if hidden_size is not None:
                extra_dims = (1, hidden_size) if self._merger is not None else (hidden_size,)
            else:
                raise ValueError(
                    "warmup_capture: cannot infer extra_dims. "
                    "Please pass hidden_size or extra_dims explicitly."
                )

        sorted_lens = sorted(self.capture_seq_lens, reverse=True)
        pending_lens = [k for k in sorted_lens if k not in self.block_graphs]

        logger.info(
            f"[ViTCudaGraphRunner] warmup_capture: pre-capturing {len(pending_lens)} graphs "
            f"for seq_lens={pending_lens}"
        )

        override_backend = global_server_args_dict.get("mm_attention_backend")

        enable_cudagraph_gc = global_server_args_dict.get("enable_cudagraph_gc", False)
        with graph_capture() as capture_ctx, freeze_gc(enable_cudagraph_gc):
            stream = capture_ctx.stream
            for graph_key in pending_lens:
                dummy_x = torch.zeros(
                    (graph_key,) + tuple(extra_dims),
                    dtype=self.dtype, device=self.device,
                )
                # Allocate all stable buffers (block_input, block_output, cu_seqlens, etc.)
                self._alloc_buffers(graph_key, dummy_x)

                if torch.cuda.is_available():
                    avail_mem = get_available_gpu_memory(
                        self.device.type, torch.cuda.current_device(), empty_cache=False
                    )
                    logger.info(
                        f"[ViTCudaGraphRunner] warmup_capture graph_key={graph_key}: "
                        f"available GPU memory = {avail_mem:.2f} GB"
                    )

                if rope_freqs_cis_head_dim_half is not None:
                    head_dim_half = rope_freqs_cis_head_dim_half
                    if graph_key not in self.rope_freqs_ws:
                        self.rope_freqs_ws[graph_key] = torch.zeros(
                            graph_key, head_dim_half,
                            dtype=torch.complex64, device=self.device,
                        ).contiguous()
                    stable_rope = self.rope_freqs_ws[graph_key]
                    self._warmup(graph_key, override_backend, stream=stream, rope_freqs_cis=stable_rope)
                    self._create_graph(graph_key=graph_key, stream=stream, rope_freqs_cis=stable_rope)
                elif rope_head_dim is not None:
                    self._ensure_sin_cos_ws(graph_key, rope_head_dim)
                    used_cos_ws = self.sin_cos_ws[0][:graph_key, :]
                    used_sin_ws = self.sin_cos_ws[1][:graph_key, :]
                    if use_rotary_pos_emb_cos_sin:
                        self._warmup(
                            graph_key, override_backend, stream=stream,
                            rotary_pos_emb_cos=used_cos_ws,
                            rotary_pos_emb_sin=used_sin_ws,
                        )
                        self._create_graph(
                            graph_key=graph_key, stream=stream,
                            rotary_pos_emb_cos=used_cos_ws,
                            rotary_pos_emb_sin=used_sin_ws,
                        )
                    else:
                        self._warmup(
                            graph_key, override_backend, stream=stream,
                            position_embeddings=(used_cos_ws, used_sin_ws),
                        )
                        self._create_graph(
                            graph_key=graph_key, stream=stream,
                            position_embeddings=(used_cos_ws, used_sin_ws),
                        )
                else:
                    raise ValueError(
                        "warmup_capture: must provide either rope_freqs_cis_head_dim_half "
                        "(Format B, complex64) or rope_head_dim (Format A, real-valued)."
                    )

        logger.info(
            f"[ViTCudaGraphRunner] warmup_capture done: "
            f"{len(self.block_graphs)} graphs captured."
        )

    @property
    def tp_size_for_warmup(self) -> int:
        """TP size used only for hidden_size inference in warmup_capture."""
        try:
            from sglang.srt.distributed.parallel_state import get_tensor_model_parallel_world_size
            return get_tensor_model_parallel_world_size()
        except Exception:
            return 1

    def replay(
        self,
        graph_key: int,
        real_S: int,
        x_3d: torch.Tensor,
        cu_seqlens: torch.Tensor,
        cu_window_seqlens: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        rotary_pos_emb_cos: Optional[torch.Tensor] = None,
        rotary_pos_emb_sin: Optional[torch.Tensor] = None,
        rope_freqs_cis: Optional[torch.Tensor] = None,
        output_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Replay graph for *graph_key*, copying padded inputs and returning
        only the first *real_S* rows of the output.

        All tensor arguments refer to the *padded* versions (size = graph_key).
        """
        # ------ Update cu_seqlens buffers (with padding sentinel appended) ------
        # cu_seqlens here is already the padded version [0,..,S,padded_S] computed
        # in run() — shape [B+2] or [B+1] if no padding needed.
        self._update_cu_seqlens(graph_key, cu_seqlens, window=False)
        if cu_window_seqlens is not None and self._fullatt_block_indexes:
            self._update_cu_seqlens(graph_key, cu_window_seqlens, window=True)

        # ------ Update rotary embeddings ------
        if rope_freqs_cis is not None:
            self.rope_freqs_ws[graph_key].copy_(rope_freqs_cis)
        elif position_embeddings is not None:
            head_dim = position_embeddings[0].shape[1]
            self._ensure_sin_cos_ws(graph_key, head_dim)
            self.sin_cos_ws[0][:graph_key].copy_(position_embeddings[0])
            self.sin_cos_ws[1][:graph_key].copy_(position_embeddings[1])
        elif rotary_pos_emb_cos is not None and rotary_pos_emb_sin is not None:
            head_dim = rotary_pos_emb_cos.shape[1]
            self._ensure_sin_cos_ws(graph_key, head_dim)
            self.sin_cos_ws[0][:graph_key].copy_(rotary_pos_emb_cos)
            self.sin_cos_ws[1][:graph_key].copy_(rotary_pos_emb_sin)

        # ------ Update hidden states input buffer ------
        self.block_input[graph_key].copy_(x_3d)

        # ------ Replay ------
        self.block_graphs[graph_key].replay()

        # ------ Unpad output ------
        out = self.block_output[graph_key]
        out = out[:real_S]

        # Optional output reordering (Qwen2.5-VL window permutation inverse)
        if output_indices is not None:
            out = out.index_select(0, output_indices)

        return out

    def run(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        cu_window_seqlens: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        rotary_pos_emb_cos: Optional[torch.Tensor] = None,
        rotary_pos_emb_sin: Optional[torch.Tensor] = None,
        rope_freqs_cis: Optional[torch.Tensor] = None,
        output_indices: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """Run the vision encoder with CUDA-graph acceleration and padding.

        Input ``x`` (shape ``[S, ...]``) is padded to the nearest ``capture_seq_lens``
        entry >= S.  After graph replay the output is truncated back to ``[:S]``.

        Returns:
            Output tensor ``[S, ...]``, or **None** when ``lazy_capture=False`` and
            the required graph has not been pre-captured (caller should fall back to
            the eager path).
        """
        # Normalise shape: Qwen models expect [S, 1, H], MoonViT uses [S, H].
        x_in = x.unsqueeze(1) if x.dim() == 2 and self._merger is not None else x

        real_S: int = x_in.shape[0]
        graph_key: int = self._find_padded_key(real_S)

        if graph_key not in self.block_graphs:
            if not self._lazy_capture:
                # lazy_capture=False
                logger.warning(
                    f"[ViTCudaGraphRunner] graph_key={graph_key} not pre-captured "
                    f"(real_S={real_S}, capture_seq_lens={self.capture_seq_lens}). "
                    "Falling back to eager (non-CUDA-graph) execution."
                )
                return None  # caller should fall back to the eager path

            # lazy_capture=True
            logger.info(
                f"[ViTCudaGraphRunner] lazy-capturing graph for graph_key={graph_key} "
                f"(real_S={real_S})"
            )
            self.create_graph(
                x_3d=x_in,
                cu_seqlens=cu_seqlens,
                cu_window_seqlens=cu_window_seqlens,
                position_embeddings=position_embeddings,
                rotary_pos_emb_cos=rotary_pos_emb_cos,
                rotary_pos_emb_sin=rotary_pos_emb_sin,
                rope_freqs_cis=rope_freqs_cis,
            )

        # ---------- Pad inputs to graph_key ----------
        needs_pad = (real_S < graph_key)

        # Pad hidden states (dim-0 only)
        if needs_pad:
            x_padded = self._pad_1d(x_in, graph_key)
        else:
            x_padded = x_in

        # Pad cu_seqlens: append padding-image sentinel
        cu_seqlens_padded = self._pad_cu_seqlens(cu_seqlens, graph_key)
        cu_window_seqlens_padded: Optional[torch.Tensor] = None
        if cu_window_seqlens is not None:
            cu_window_seqlens_padded = self._pad_cu_seqlens(cu_window_seqlens, graph_key)

        # Pad rope embeddings (dim-0 to graph_key; padding rows = 0)
        pe_padded: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        cos_padded: Optional[torch.Tensor] = None
        sin_padded: Optional[torch.Tensor] = None
        rope_padded: Optional[torch.Tensor] = None

        if rope_freqs_cis is not None:
            if needs_pad:
                rope_padded = self._pad_1d(rope_freqs_cis, graph_key)
            else:
                rope_padded = rope_freqs_cis
        elif position_embeddings is not None:
            if needs_pad:
                pe_padded = (
                    self._pad_1d(position_embeddings[0], graph_key),
                    self._pad_1d(position_embeddings[1], graph_key),
                )
            else:
                pe_padded = position_embeddings
        elif rotary_pos_emb_cos is not None and rotary_pos_emb_sin is not None:
            if needs_pad:
                cos_padded = self._pad_1d(rotary_pos_emb_cos, graph_key)
                sin_padded = self._pad_1d(rotary_pos_emb_sin, graph_key)
            else:
                cos_padded = rotary_pos_emb_cos
                sin_padded = rotary_pos_emb_sin

        # ---------- Replay ----------
        return self.replay(
            graph_key=graph_key,
            real_S=real_S,
            x_3d=x_padded,
            cu_seqlens=cu_seqlens_padded,
            cu_window_seqlens=cu_window_seqlens_padded,
            position_embeddings=pe_padded,
            rotary_pos_emb_cos=cos_padded,
            rotary_pos_emb_sin=sin_padded,
            rope_freqs_cis=rope_padded,
            output_indices=output_indices,
        )
