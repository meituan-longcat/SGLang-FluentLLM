"""
Inference-only Longcat Flash VL model.

Integrates Visual Encoder (Qwen2.5-VL ViT), Visual Bridge, Visual Quantizer,
and Visual Embedding Bridge with FLASHForCausalLM.
"""

import logging
from typing import Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VisionPatchEmbed,
    Qwen2_5_VisionRotaryEmbedding,
)

from sglang.srt.ops.layernorm import RMSNorm

from sglang.srt.configs.longcat_flash_vl import (
    LongcatFlashVLConfig,
    LongcatVLQuantizerConfig,
    LongcatVLVisualConfig,
)
from sglang.srt.layers.attention.vision import VisionAttention
from sglang.srt.layers.linear import MergedColumnParallelLinear, RowParallelLinear
from sglang.srt.layers.utils import CP_METADATA, cp_split_and_rebuild_data
from sglang.srt.models.utils import RotaryPosMixin
from sglang.srt.ops.activation import SiluAndMul
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.managers.scheduling.multimodal_data import (
    Modality,
    MultimodalDataItem,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.npu_longcat_flash import FLASHForCausalLM
from sglang.srt.multimodal.mm_utils import embed_mm_inputs
from sglang.srt.env import envs, global_server_args_dict
from sglang.srt.layers.attention.dp_attention import (
    get_attention_tp_group,
    is_dp_attention_enabled,
)
from sglang.srt.utils import is_npu

_is_npu = is_npu()
logger = logging.getLogger(__name__)

class LongcatVisionMLP(nn.Module):
    # flashinfer silu_and_mul vec_size = 16 / sizeof(dtype);
    _SILU_VEC_BYTES = 16

    def __init__(self, hidden_size: int, intermediate_size: int,
                 hidden_act: str = "silu", bias: bool = True,
                 use_data_parallel: bool = False, prefix: str = ""):
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=hidden_size,
            output_sizes=[intermediate_size, intermediate_size],
            bias=bias,
            prefix=prefix,
            disable_parallel=use_data_parallel,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=bias,
            prefix=prefix,
            disable_parallel=use_data_parallel,
        )
        self.act_fn = self._build_act_fn(hidden_act, intermediate_size)

    def _build_act_fn(self, hidden_act: str, intermediate_size: int):
        """
        flashinfer ``silu_and_mul`` requires ``intermediate_size % vec_size == 0``
        when unaligned, fall back to torch elementwise (no per-forward branch).
        """
        # bf16 assumption — ViT inference dtype.
        vec_size = self._SILU_VEC_BYTES // torch.bfloat16.itemsize  # 8
        silu_aligned = intermediate_size % vec_size == 0
        if hidden_act == "silu" and silu_aligned:
            return SiluAndMul()
        base_act = ACT2FN[hidden_act] if hidden_act != "silu" else F.silu

        def _act_fn(x: torch.Tensor) -> torch.Tensor:
            d = x.shape[-1] // 2
            return base_act(x[..., :d]) * x[..., d:]

        return _act_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x_down, _ = self.down_proj(x)
        return x_down

class LongcatVisionBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, intermediate_size: int,
                 hidden_act: str = "silu",
                 use_data_parallel: bool = False, prefix: str = ""):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size, eps=1e-6)
        self.norm2 = RMSNorm(hidden_size, eps=1e-6)

        self.attn = VisionAttention(
            qkv_backend="fa2", # aligned with offline
            embed_dim=hidden_size,
            num_heads=num_heads,
            projection_size=hidden_size,
            use_qkv_parallel=True,
            qkv_bias=True,
            proj_bias=True,
            flatten_batch=True,
            prefix=f"{prefix}.attn" if prefix else "attn",
            use_data_parallel=use_data_parallel,
            use_dp_attention_reduce=is_dp_attention_enabled(),
        )

        self.mlp = LongcatVisionMLP(
            hidden_size, intermediate_size, hidden_act=hidden_act, bias=True,
            use_data_parallel=use_data_parallel,
            prefix=f"{prefix}.mlp" if prefix else "mlp",
        )

    def forward(self, hidden_states: torch.Tensor, cu_seqlens: torch.Tensor,
                position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):

        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.attn(hidden_states, cu_seqlens=cu_seqlens,
                                  position_embeddings=position_embeddings)
        # attn returns 3D [1, seq, hidden]; residual is 2D [seq, hidden] —
        # broadcast add yields 3D, same as the original implementation. The
        # flashinfer rmsnorm kernel accepts 3D inputs, so norm2 sees 3D.
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states

class LongcatVisualEncoder(nn.Module, RotaryPosMixin):
    def __init__(self, config: LongcatVLVisualConfig, use_data_parallel: bool = False):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.depth = config.depth
        self.spatial_merge_size = config.spatial_merge_size
        self.fullatt_block_indexes = config.fullatt_block_indexes
        self.window_size = config.window_size
        self.patch_size = config.patch_size

        self.spatial_merge_unit = self.spatial_merge_size ** 2

        self.patch_embed = Qwen2_5_VisionPatchEmbed(
            patch_size=config.patch_size,
            temporal_patch_size=config.temporal_patch_size,
            in_channels=config.in_channels,
            embed_dim=config.hidden_size,
        )

        self.rotary_pos_emb = Qwen2_5_VisionRotaryEmbedding(
            config.hidden_size // config.num_heads // 2
        )

        hidden_act = getattr(config, "hidden_act", "silu")
        self.blocks = nn.ModuleList([
            LongcatVisionBlock(
                hidden_size=config.hidden_size,
                num_heads=config.num_heads,
                intermediate_size=config.intermediate_size,
                hidden_act=hidden_act,
                use_data_parallel=use_data_parallel,
                prefix=f"blocks.{i}",
            )
            for i in range(config.depth)
        ])

    def get_window_index(self, grid_thw: torch.Tensor):
        cu_window_seqlens: list = [0]
        window_index_id = 0
        vit_merger_window_size = (
            self.window_size // self.spatial_merge_size // self.patch_size
        )
        window_index: list = []
        for grid_t, grid_h, grid_w in grid_thw:
            llm_grid_h, llm_grid_w = (
                grid_h // self.spatial_merge_size,
                grid_w // self.spatial_merge_size,
            )
            index = torch.arange(grid_t * llm_grid_h * llm_grid_w).reshape(
                grid_t, llm_grid_h, llm_grid_w
            )
            pad_h = vit_merger_window_size - llm_grid_h % vit_merger_window_size
            pad_w = vit_merger_window_size - llm_grid_w % vit_merger_window_size
            num_windows_h = (llm_grid_h + pad_h) // vit_merger_window_size
            num_windows_w = (llm_grid_w + pad_w) // vit_merger_window_size
            index_padded = F.pad(index, (0, pad_w, 0, pad_h), "constant", -100)
            index_padded = index_padded.reshape(
                grid_t,
                num_windows_h,
                vit_merger_window_size,
                num_windows_w,
                vit_merger_window_size,
            )
            index_padded = index_padded.permute(0, 1, 3, 2, 4).reshape(
                grid_t,
                num_windows_h * num_windows_w,
                vit_merger_window_size,
                vit_merger_window_size,
            )
            seqlens = (index_padded != -100).sum([2, 3]).reshape(-1)
            index_padded = index_padded.reshape(-1)
            index_new = index_padded[index_padded != -100]
            window_index.append(index_new + window_index_id)
            cu_seqlens_tmp = (
                seqlens.cumsum(0) * self.spatial_merge_unit + cu_window_seqlens[-1]
            )
            cu_window_seqlens.extend(cu_seqlens_tmp.tolist())
            window_index_id += (grid_t * llm_grid_h * llm_grid_w).item()
        window_index = torch.cat(window_index, dim=0)
        return window_index, cu_window_seqlens

    def rot_pos_emb(self, grid_thw: torch.Tensor):
        pos_ids = []
        for t, h, w in grid_thw.tolist():
            base = self.rot_pos_ids(int(h), int(w), self.spatial_merge_size)
            pos_ids.append(base if t == 1 else base.repeat(int(t), 1))

        pos_ids = torch.cat(pos_ids, dim=0).to(grid_thw.device)

        max_grid_size_t = pos_ids.max().item() + 1
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size_t)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)
        return rotary_pos_emb

    @torch.no_grad()
    def forward(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor):
        hidden_states = self.patch_embed(pixel_values.to(self.patch_embed.proj.weight.dtype))
        grid_thw = grid_thw.to(hidden_states.device)

        rotary_pos_emb = self.rot_pos_emb(grid_thw)

        window_index, cu_window_seqlens = self.get_window_index(grid_thw)
        cu_window_seqlens = torch.tensor(
            cu_window_seqlens, device=hidden_states.device, dtype=torch.int32
        )
        cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)

        seq_len = hidden_states.shape[0]
        hidden_states = hidden_states.reshape(
            seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1
        )
        hidden_states = hidden_states[window_index.to(hidden_states.device)]
        hidden_states = hidden_states.reshape(seq_len, -1)

        rotary_pos_emb = rotary_pos_emb.reshape(
            seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1
        )
        rotary_pos_emb = rotary_pos_emb[window_index.to(rotary_pos_emb.device)]
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)

        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        cu_seqlens = torch.repeat_interleave(
            grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
        ).cumsum(dim=0, dtype=torch.int32)
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        for layer_num, blk in enumerate(self.blocks):
            if layer_num in self.fullatt_block_indexes:
                cu_seqlens_now = cu_seqlens
            else:
                cu_seqlens_now = cu_window_seqlens

            hidden_states = blk(hidden_states, cu_seqlens_now, position_embeddings)

        return hidden_states, window_index


# ---------------------------------------------------------------------------
# Visual Bridge (spatial merge + MLP)
# ---------------------------------------------------------------------------
class LongcatVisualBridge(nn.Module):
    def __init__(self, config: LongcatVLVisualConfig):
        super().__init__()
        self.merge_size = config.merge_size
        self.hidden_size = config.hidden_size * (self.merge_size ** 2)

        self.ln_q = RMSNorm(config.hidden_size, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, config.out_hidden_size),
        )

    @torch.no_grad()
    def forward(self, x: torch.Tensor, window_index: torch.Tensor) -> torch.Tensor:
        x = self.mlp(self.ln_q(x).view(-1, self.hidden_size))
        reverse_indices = torch.argsort(window_index)
        x = x[reverse_indices]
        return x


# ---------------------------------------------------------------------------
# Visual Quantizer (RQ encode path only)
# ---------------------------------------------------------------------------

class VQEmbeddingInference(nn.Module):
    """Inference-only VQ embedding (no EMA update).
    Codebook is pinned to float32 — survives model-wide .to(bfloat16) calls."""

    def __init__(self, n_embed: int, embed_dim: int):
        super().__init__()
        self.n_embed = n_embed
        self.embed = nn.Parameter(torch.zeros(n_embed + 1, embed_dim, dtype=torch.float32))

    def _apply(self, fn):
        super()._apply(fn)
        # Restore codebook to float32 after any model-wide dtype cast (e.g. .to(bfloat16))
        self.embed.data = self.embed.data.float()
        return self

    @torch.no_grad()
    def find_nearest_embedding(self, inputs: torch.Tensor) -> torch.Tensor:
        codebook = self.embed[:-1].float()
        inputs_norm_sq = inputs.pow(2.0).sum(dim=-1, keepdim=True)
        codebook_norm_sq = codebook.pow(2.0).sum(dim=-1, keepdim=True).t()
        distances = inputs_norm_sq + codebook_norm_sq - 2.0 * inputs @ codebook.t()
        return distances.argmin(dim=-1)

    @torch.no_grad()
    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if inputs.dtype != torch.float32:
            inputs = inputs.float()
        embed_idxs = self.find_nearest_embedding(inputs)
        embeds = self.embed[embed_idxs]
        return embeds, embed_idxs


class LongcatVisualQuantizer(nn.Module):
    def __init__(self, config: LongcatVLQuantizerConfig):
        super().__init__()
        self.config = config
        self.depth = config.depth
        self.codebook_size = config.codebook_size
        self.codebook_dim = config.codebook_dim

        if config.quant_conv:
            self.quant_conv = nn.Sequential(
                nn.LayerNorm(config.in_channels),
                nn.Linear(config.in_channels, config.in_channels),
                nn.GELU(),
                nn.Linear(config.in_channels, config.codebook_dim),
            )
        else:
            self.quant_conv = None

        self.codebook = VQEmbeddingInference(config.codebook_size, config.codebook_dim)

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode features to VQ indices. Returns [L, depth] indices."""
        if self.quant_conv is not None:
            x = self.quant_conv(x.unsqueeze(0)).squeeze(0)

        x = x.float()
        all_indices = []
        residual = x.clone()

        for _ in range(self.depth):
            _, indices = self.codebook(residual)
            quantized = self.codebook.embed[indices]
            residual = residual - quantized
            all_indices.append(indices)

        return torch.stack(all_indices, dim=-1).to(torch.int32)


# ---------------------------------------------------------------------------
# Visual Embedding Bridge (VQ indices → LLM embeddings)
# ---------------------------------------------------------------------------

class LongcatVisualEmbeddingBridge(nn.Module):
    def __init__(self, codebook_sizes: List[int], hidden_size: int,
                 intermediate_size: int, hidden_act: str, rms_norm_eps: float,
                 use_data_parallel: bool = False, prefix: str = "",
                 level_parallel_group=None, bridge_parallel_group=None):
        super().__init__()
        self.level_parallel_group = level_parallel_group
        self.bridge_parallel_group = bridge_parallel_group
        self.level_parallel = (
            level_parallel_group is not None
            and level_parallel_group.world_size > 1
        )
        if self.level_parallel:
            level_parallel_size = level_parallel_group.world_size
            level_parallel_rank = level_parallel_group.rank_in_group
            if level_parallel_size > len(codebook_sizes):
                raise ValueError(
                    "visual embedding level-parallel size cannot exceed the "
                    f"number of RVQ levels: {level_parallel_size=} "
                    f"num_levels={len(codebook_sizes)}"
                )
            if len(codebook_sizes) % level_parallel_size != 0:
                raise ValueError(
                    "the number of RVQ levels must be divisible by the visual "
                    f"embedding level-parallel size: num_levels={len(codebook_sizes)} "
                    f"{level_parallel_size=}"
                )
            self.owned_levels = tuple(
                level
                for level in range(len(codebook_sizes))
                if level % level_parallel_size == level_parallel_rank
            )
        else:
            self.owned_levels = tuple(range(len(codebook_sizes)))

        owned_level_set = set(self.owned_levels)
        self.embedding_layers = nn.ModuleList([
            (
                nn.Embedding(codebook_size, hidden_size)
                if level in owned_level_set
                else None
            )
            for level, codebook_size in enumerate(codebook_sizes)
        ])
        self.hidden_size = hidden_size
        self.codebook_num = len(codebook_sizes)
        # Keep the bounds as a non-persistent buffer so they follow the module
        # device without becoming part of the checkpoint. Remote EPD encoders
        # are outside this process' trust boundary: malformed/corrupted RVQ
        # indices must never reach nn.Embedding, where an out-of-range lookup
        # aborts the whole NPU process instead of raising a recoverable error.
        self.register_buffer(
            "rvq_codebook_sizes",
            torch.tensor(codebook_sizes, dtype=torch.int32),
            persistent=False,
        )

        self.post_attention_layernorm = nn.LayerNorm(hidden_size, eps=rms_norm_eps)

        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=hidden_size,
            output_sizes=[intermediate_size, intermediate_size],
            bias=False,
            prefix=prefix,
            outside_tp_group=bridge_parallel_group,
            disable_parallel=use_data_parallel,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            prefix=prefix,
            outside_tp_group=bridge_parallel_group,
            disable_parallel=use_data_parallel,
        )
        if hidden_act == "silu":
            self.act_fn = SiluAndMul()
        else:
            base_act = ACT2FN[hidden_act]

            def _act_fn(x: torch.Tensor) -> torch.Tensor:
                gate, up = x.chunk(2, dim=-1)
                return base_act(gate) * up

            self.act_fn = _act_fn

        logger.info(
            "Visual embedding bridge layout: level_parallel=%s "
            "owned_levels=%s group_size=%d group_rank=%d",
            self.level_parallel,
            self.owned_levels,
            (
                self.level_parallel_group.world_size
                if self.level_parallel_group is not None
                else 1
            ),
            (
                self.level_parallel_group.rank_in_group
                if self.level_parallel_group is not None
                else 0
            ),
        )

    @property
    def device(self) -> torch.device:
        for level in self.owned_levels:
            embedding_layer = self.embedding_layers[level]
            if embedding_layer is not None:
                return embedding_layer.weight.device
        raise RuntimeError("visual embedding bridge does not own any RVQ level")

    def owns_level(self, level: int) -> bool:
        return level in self.owned_levels

    @torch.no_grad()
    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        """indices: [N, num_levels] → embeddings: [N, hidden_size]"""
        if indices.ndim == 0 or indices.shape[-1] != self.codebook_num:
            raise RuntimeError(
                "Invalid rvq_indices shape before visual embedding lookup: "
                f"shape={tuple(indices.shape)}, expected_last_dim={self.codebook_num}"
            )

        # Clamp each RVQ level to its own codebook range.
        codebook_sizes = self.rvq_codebook_sizes.to(device=indices.device)
        indices = torch.minimum(indices.clamp_min(0), codebook_sizes - 1)
        embedding = None
        for i in self.owned_levels:
            emb_layer = self.embedding_layers[i]
            if embedding is None:
                embedding = emb_layer(indices[..., i])
            else:
                embedding = embedding + emb_layer(indices[..., i])

        if self.level_parallel:
            embedding = self.level_parallel_group.all_reduce(embedding)

        residual = embedding
        hidden_states = self.post_attention_layernorm(embedding)
        gate_up, _ = self.gate_up_proj(hidden_states)
        hidden_states = self.act_fn(gate_up)
        hidden_states, _ = self.down_proj(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


# ---------------------------------------------------------------------------
# Vision Tower (combines Encoder + Bridge + Quantizer)
# ---------------------------------------------------------------------------

class LongcatVisionTower(nn.Module):
    def __init__(self, vision_config: LongcatVLVisualConfig,
                 quantizer_config: LongcatVLQuantizerConfig,
                 use_data_parallel: bool = False):
        super().__init__()
        self.encoder = LongcatVisualEncoder(vision_config, use_data_parallel)
        self.bridge = LongcatVisualBridge(vision_config)
        self.quantizer = LongcatVisualQuantizer(quantizer_config)
        self.vision_config = vision_config

    @property
    def device(self) -> torch.device:
        return self.encoder.patch_embed.proj.weight.device

    @property
    def dtype(self) -> torch.dtype:
        return self.encoder.patch_embed.proj.weight.dtype

    @torch.no_grad()
    def fake_input(self) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Generate minimal fake video input.

        Returns:
            fake_pixels: [merge_size*merge_size, patch_dim] zero tensor
            fake_grid: [1, 3] tensor with (t=1, h=merge_size, w=merge_size)
            n_merged_tokens: number of merged tokens the fake input produces (always 1)
        """
        merge_size = max(self.vision_config.merge_size, self.vision_config.spatial_merge_size)
        temporal_patch_size = self.vision_config.temporal_patch_size
        patch_size = self.vision_config.patch_size
        spatial_merge_size = self.vision_config.spatial_merge_size

        fake_image = torch.zeros([
            1,
            temporal_patch_size,
            3,
            merge_size // spatial_merge_size,
            spatial_merge_size,
            patch_size,
            merge_size // spatial_merge_size,
            spatial_merge_size,
            patch_size,
        ], dtype=torch.float32)
        # HACK: build fake_image on CPU (not self.device) so the 9-dim permute below
        # runs on CPU. NPU aclnnInplaceCopy rejects tensors >8 dims (EZ1001), and
        # .reshape() forces .contiguous() on the 9-dim permuted view. CPU has no
        # 8-dim cap. Callers move the 2-dim result to device before cat. Remove
        # this comment+CPU pin once CANN lifts the 8-dim limit on aclnnInplaceCopy.
        patches = fake_image.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
        flatten_patches = patches.reshape(
            merge_size * merge_size, 3 * temporal_patch_size * patch_size * patch_size
        )
        fake_grid = torch.tensor(
            [[1, merge_size, merge_size]], dtype=torch.long
        )
        n_merged_tokens = 1
        # Move the 2-dim results to the vision tower's device+dtype here so callers can
        # cat without a per-call .to() — the CPU pin above is only for the 9-dim permute.
        return (
            flatten_patches.to(device=self.device, dtype=self.dtype),
            fake_grid.to(device=self.device),
            n_merged_tokens,
        )

    @torch.no_grad()
    def forward(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor,
                n_image_patches: Optional[int] = None) -> torch.Tensor:
        """Full pipeline: pixel_values → VQ indices [N_merged, depth].

        Args:
            n_image_patches: if set, only return the first n_image_patches indices
                (used to discard appended fake video tokens).
        """
        hidden_states, window_index = self.encoder(pixel_values, grid_thw)
        hidden_states = self.bridge(hidden_states, window_index)
        indices = self.quantizer.encode(hidden_states)
        if n_image_patches is not None:
            indices = indices[:n_image_patches]
        return indices


# ---------------------------------------------------------------------------
# Main Model: LongcatFlashVLForConditionalGeneration
# ---------------------------------------------------------------------------

class LongcatFlashVLForConditionalGeneration(nn.Module):

    def __init__(
        self,
        config: LongcatFlashVLConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        **kwargs,
    ) -> None:
        super().__init__()
        self.config = config
        self.quant_config = quant_config

        # EPD encoder_only / language_only flags (from global_server_args_dict)
        self.encoder_only = global_server_args_dict.get("encoder_only", False)
        self.language_only = global_server_args_dict.get("language_only", False)
        self.encoder_output_format = global_server_args_dict.get(
            "encoder_output_format", "rvq_indices"
        )
        self._is_disaggregation_decode = (
            global_server_args_dict.get("disaggregation_mode", "null") == "decode"
        )

        # Keep the original NPU safety rule: the vision MLP is replicated on
        # NPU because its TP all-reduce cannot be gathered by EncoderDispatcher.
        if torch.distributed.is_initialized():
            from sglang.srt.layers.attention.dp_attention import get_attention_tp_size

            try:
                attn_tp_size = get_attention_tp_size()
            except AssertionError:
                attn_tp_size = 1
        else:
            attn_tp_size = 1
        self.use_data_parallel = (
            global_server_args_dict.get("mm_enable_dp_encoder", False)
            or attn_tp_size <= 1
        )

        # vision_tower (encoder + bridge + quantizer):
        #   - encoder_only=True: load (Encoder needs it for ViT forward)
        #   - language_only=True: skip (Prefill doesn't run ViT)
        #   - joint mode: load
        if not self.language_only:
            self.vision_tower = LongcatVisionTower(
                config.vision_config,
                config.quantizer_config,
                use_data_parallel=self.use_data_parallel,
            )
        else:
            self.vision_tower = None

        # language_model:
        #   - encoder_only=True: skip (Encoder has no LM)
        #   - language_only=True: load
        #   - joint mode: load
        if not self.encoder_only:
            self.language_model = FLASHForCausalLM(
                config.text_config,
                quant_config=quant_config,
            )
        else:
            self.language_model = None

        # visual_embedding_bridge (rvq_indices → LLM embedding lookup + MLP):
        #   - encoder_only=True (rvq_indices mode): skip (Encoder only outputs rvq_indices)
        #   - encoder_only=True (vit_embedding mode): keep on Encoder workers
        #   - EPD Prefill: only PP0 consumes it
        #   - EPD Decode: skip on every PP stage (Prefill KV is transferred in)
        #   - joint mode: only PP0 consumes it
        bridge_required_by_role = (
            (
                not self.encoder_only
                and not self._is_disaggregation_decode
            )
            or (
                self.encoder_only
                and self.encoder_output_format == "vit_embedding"
            )
        )
        bridge_owned_by_pp_stage = (
            self.language_model is None
            or self.language_model.pp_group.is_first_rank
        )
        if bridge_required_by_role and bridge_owned_by_pp_stage:
            level_parallel_group = None
            bridge_parallel_group = None
            # This switch controls only RVQ-level ownership and the level SUM.
            # The bridge MLP keeps its existing Attention-TP parallel layout.
            enable_level_parallel = (
                envs.ENABLE_VISUAL_EMBEDDING_BRIDGE_PARALLEL.get()
            )
            if (
                self.language_model is not None
                and not self.use_data_parallel
                and torch.distributed.is_initialized()
            ):
                attention_tp_group = get_attention_tp_group()
                bridge_parallel_group = attention_tp_group
                num_levels = len(config.codebook_sizes)
                if (
                    enable_level_parallel
                    and attention_tp_group.world_size > 1
                    and attention_tp_group.world_size <= num_levels
                    and num_levels % attention_tp_group.world_size == 0
                ):
                    level_parallel_group = attention_tp_group
                elif enable_level_parallel and attention_tp_group.world_size > 1:
                    logger.warning(
                        "Visual embedding level-parallel is disabled because "
                        "the Attention-TP layout does not evenly partition RVQ "
                        "levels: attention_tp_size=%d num_levels=%d",
                        attention_tp_group.world_size,
                        num_levels,
                    )
                elif attention_tp_group.world_size > 1:
                    logger.info(
                        "Visual embedding level-parallel is disabled by "
                        "ENABLE_VISUAL_EMBEDDING_BRIDGE_PARALLEL; all RVQ "
                        "levels are replicated on each Attention-TP rank"
                    )

            self.visual_embedding_bridge = LongcatVisualEmbeddingBridge(
                codebook_sizes=config.codebook_sizes,
                hidden_size=config.bridge_hidden_size,
                intermediate_size=config.bridge_intermediate_size,
                hidden_act=config.bridge_hidden_act,
                rms_norm_eps=config.bridge_rms_norm_eps,
                use_data_parallel=self.use_data_parallel,
                prefix="visual_embedding_bridge",
                level_parallel_group=level_parallel_group,
                bridge_parallel_group=bridge_parallel_group,
            )
        else:
            self.visual_embedding_bridge = None
            logger.info(
                "Visual embedding bridge is not constructed on this rank: "
                "required_by_role=%s first_pp_stage=%s "
                "disaggregation_decode=%s",
                bridge_required_by_role,
                bridge_owned_by_pp_stage,
                self._is_disaggregation_decode,
            )

    def get_rvq_indices(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        """Run vision_tower only (encoder + bridge + quantizer) and return
        raw rvq_indices [num_tokens, depth] int32, WITHOUT running
        visual_embedding_bridge.

        Used by the EPD Encoder server in rvq_indices output mode.
        """
        assert self.vision_tower is not None, "vision_tower not loaded (language_only=True?)"
        device = self.vision_tower.device
        target_dtype = self.vision_tower.dtype

        pixel_values = torch.cat([item.feature for item in items], dim=0).to(
            device=device, dtype=target_dtype
        )
        grid_thws = torch.cat([item.image_grid_thw for item in items], dim=0).to(device)

        n_image_patches = (
            (grid_thws[:, 1] // self.vision_tower.vision_config.spatial_merge_size)
            * (grid_thws[:, 2] // self.vision_tower.vision_config.spatial_merge_size)
            * grid_thws[:, 0]
        ).sum().item()

        fake_pixels, fake_grid, _ = self.vision_tower.fake_input()
        pixel_values = torch.cat([pixel_values, fake_pixels.to(dtype=pixel_values.dtype)], dim=0)
        grid_thws = torch.cat([grid_thws, fake_grid], dim=0)

        rvq_indices = self.vision_tower(pixel_values, grid_thws, n_image_patches=n_image_patches)
        return rvq_indices.to(torch.int32)

    def item_token_count(self, item: MultimodalDataItem) -> int:
        """Compute ViT output token count for an item (EncoderFeature)."""
        grid_thw = item.image_grid_thw
        t, h, w = grid_thw[0].tolist()
        merge = self.vision_tower.vision_config.spatial_merge_size
        return int(t * (h // merge) * (w // merge))

    def get_encoder_preprocessor_class(self):
        """Return the ``EncoderPreprocessor`` subclass for this model.

        The model only *declares* which preprocessor class implements it;
        the preprocessor class itself knows how to build an instance (via
        its ``EncoderPreprocessor.from_hf_config`` classmethod factory).
        This keeps construction logic on the interface that owns it
        (``EncoderPreprocessor``), not on ``EncoderFeature``.
        """
        from sglang.srt.multimodal.processors.longcat_flash_vl import (
            LongcatFlashVLEncoderPreprocessor,
        )
        return LongcatFlashVLEncoderPreprocessor

    def get_image_feature(self, items: List[MultimodalDataItem],
                          output_format: str = "vit_embedding") -> torch.Tensor:
        # EPD Encoder rvq_indices mode: run vision_tower only, return raw indices.
        if output_format == "rvq_indices" and self.vision_tower is not None:
            return self.get_rvq_indices(items)
        # EPD language_only path: items carry rvq_indices (or vit_embedding)
        # from the Encoder server in item.feature.  Decode rvq_indices via
        # visual_embedding_bridge (no ViT needed).
        if self.vision_tower is None and self.visual_embedding_bridge is not None:
            indices_list = []
            for item in items:
                emb = item.feature
                if emb is None:
                    raise RuntimeError(
                        "EPD language_only: item has no feature (rvq_indices/vit_embedding)"
                    )
                indices_list.append(emb)
            vq_indices = torch.cat(indices_list, dim=0)
            # If the data is already float embeddings (vit_embedding mode),
            # return as-is; otherwise decode rvq_indices via the bridge.
            bridge_device = self.visual_embedding_bridge.device
            if vq_indices.dtype in (torch.float16, torch.bfloat16, torch.float32):
                return vq_indices.to(bridge_device)
            vq_indices = vq_indices.to(device=bridge_device, dtype=torch.int32)
            visual_embeds = self.visual_embedding_bridge(vq_indices)
            return visual_embeds

        # Standard path: run ViT + quantizer + bridge from pixel_values.
        device = self.vision_tower.device
        target_dtype = self.vision_tower.dtype

        pixel_values = torch.cat([item.feature for item in items], dim=0).to(
            device=device, dtype=target_dtype
        )
        grid_thws = torch.cat([item.image_grid_thw for item in items], dim=0).to(device)

        n_image_patches = (
            (grid_thws[:, 1] // self.vision_tower.vision_config.spatial_merge_size)
            * (grid_thws[:, 2] // self.vision_tower.vision_config.spatial_merge_size)
            * grid_thws[:, 0]
        ).sum().item()

        fake_pixels, fake_grid, _ = self.vision_tower.fake_input()
        pixel_values = torch.cat([pixel_values, fake_pixels.to(dtype=pixel_values.dtype)], dim=0)
        grid_thws = torch.cat([grid_thws, fake_grid], dim=0)

        vq_indices = self.vision_tower(pixel_values, grid_thws, n_image_patches=n_image_patches)
        visual_embeds = self.visual_embedding_bridge(vq_indices)

        return visual_embeds

    def get_video_feature(self, items: List[MultimodalDataItem],
                          output_format: str = "vit_embedding") -> torch.Tensor:
        """Process video inputs through the same ViT pipeline (EncoderFeature).

        Video items use image_grid_thw (with temporal > 1) and are handled identically
        to images in terms of the ViT + quantizer + embedding bridge pipeline.
        A fake image input is prepended.

        In EPD rvq_indices mode (encoder_only), returns raw rvq_indices without
        running visual_embedding_bridge; otherwise returns full visual embeds.
        """
        if output_format == "rvq_indices" and self.vision_tower is not None:
            # Preserve the original joint video path exactly: fake sample is
            # prepended and removed from the front after ViT/quantization.
            device = self.vision_tower.device
            target_dtype = self.vision_tower.dtype
            pixel_values = torch.cat(
                [item.feature for item in items], dim=0
            ).to(device=device, dtype=target_dtype)
            grid_thws = torch.cat(
                [item.image_grid_thw for item in items], dim=0
            ).to(device)
            fake_pixels, fake_grid, n_fake_patches = self.vision_tower.fake_input()
            all_pixel_values = torch.cat([fake_pixels.to(dtype=target_dtype), pixel_values], dim=0)
            all_grid_thws = torch.cat([fake_grid, grid_thws], dim=0)
            vq_indices = self.vision_tower(all_pixel_values, all_grid_thws)
            return vq_indices[n_fake_patches:].to(torch.int32)
        # EPD language_only path: items carry rvq_indices (or vit_embedding)
        # from the Encoder server in item.feature.  Decode rvq_indices via
        # visual_embedding_bridge (no ViT needed).  Same logic as
        # get_image_feature's language_only branch.
        if self.vision_tower is None and self.visual_embedding_bridge is not None:
            indices_list = []
            for item in items:
                emb = item.feature
                if emb is None:
                    raise RuntimeError(
                        "EPD language_only: video item has no feature (rvq_indices/vit_embedding)"
                    )
                indices_list.append(emb)
            vq_indices = torch.cat(indices_list, dim=0)
            bridge_device = self.visual_embedding_bridge.device
            if vq_indices.dtype in (torch.float16, torch.bfloat16, torch.float32):
                return vq_indices.to(bridge_device)
            vq_indices = vq_indices.to(device=bridge_device, dtype=torch.int32)
            visual_embeds = self.visual_embedding_bridge(vq_indices)
            return visual_embeds

        device = self.vision_tower.device
        target_dtype = self.vision_tower.dtype

        pixel_values = torch.cat([item.feature for item in items], dim=0).to(
            device=device, dtype=target_dtype
        )
        grid_thws = torch.cat([item.image_grid_thw for item in items], dim=0).to(device)

        fake_pixels, fake_grid, n_fake_patches = self.vision_tower.fake_input()
        all_pixel_values = torch.cat([fake_pixels.to(dtype=target_dtype), pixel_values], dim=0)
        all_grid_thws = torch.cat([fake_grid, grid_thws], dim=0)

        vq_indices = self.vision_tower(all_pixel_values, all_grid_thws)
        vq_indices = vq_indices[n_fake_patches:]
        visual_embeds = self.visual_embedding_bridge(vq_indices)

        return visual_embeds

    @classmethod
    def get_model_config_for_expert_location(cls, config):
        # Delegate to FLASHForCausalLM so that EPLB can discover the real
        # n_routed_experts (768) instead of falling back to init_dummy() (1).
        return FLASHForCausalLM.get_model_config_for_expert_location(config.text_config)

    @property
    def routed_experts_weights_of_layer(self):
        return self.language_model._routed_experts_weights_of_layer.value

    @property
    def device_local_routed_experts_weights_of_layer(self):
        """Expose LM device-local expert tensors to the EPLB updater."""
        return self.language_model.device_local_routed_experts_weights_of_layer

    def get_embed_and_head(self):
        return self.language_model.get_embed_and_head()

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        lm = self.language_model

        input_embeds = None
        # Multimodal embedding only runs on the first PP rank, where embed_tokens /
        # over_embedding and the vision tower live. Later ranks receive hidden_states
        # through forward_batch.pp_proxy_tensors inside FLASHModel.forward.
        if lm.pp_group.is_first_rank:
            if lm.model.enable_over_embedding:
                embed_tokens = lm.model.over_embedding
            else:
                embed_tokens = lm.model.embed_tokens

            contains_mm = (
                forward_batch.mm_inputs is not None
                and any(m is not None for m in forward_batch.mm_inputs)
            )

            # Only embed multimodal inputs on a real prefill (EXTEND / MIXED). TARGET_VERIFY
            # and DRAFT_EXTEND are post-prefill speculative steps
            if (
                not forward_batch.forward_mode.is_decode()
                and not forward_batch.forward_mode.is_target_verify()
                and not forward_batch.forward_mode.is_draft_extend()
                and contains_mm
            ):
                mm_inputs_list = [
                    mm_input
                    for mm_input in forward_batch.mm_inputs
                    if mm_input is not None
                ]
                extend_prefix_lens = [
                    prefix_len
                    for i, prefix_len in enumerate(forward_batch.extend_prefix_lens_cpu)
                    if forward_batch.mm_inputs[i] is not None
                ]
                extend_seq_lens = [
                    seq_len
                    for i, seq_len in enumerate(forward_batch.extend_seq_lens_cpu)
                    if forward_batch.mm_inputs[i] is not None
                ]

                input_embeds, _ = embed_mm_inputs(
                    mm_inputs_list=mm_inputs_list,
                    extend_prefix_lens=extend_prefix_lens,
                    extend_seq_lens=extend_seq_lens,
                    input_ids=input_ids,
                    input_embedding=embed_tokens,
                    forward_batch=forward_batch,
                    data_embedding_func_mapping={
                        Modality.IMAGE: self.get_image_feature,
                        Modality.VIDEO: self.get_video_feature,
                    },
                )

                for mm_input_obj in mm_inputs_list:
                    for mm_item in mm_input_obj.mm_items:
                        if (
                            isinstance(mm_item.feature, torch.Tensor)
                            and mm_item.feature.is_npu
                        ):
                            mm_item.feature = mm_item.feature.to("cpu", non_blocking=True)
                forward_batch.mm_inputs = None

                if forward_batch.input_embeds is not None and not CP_METADATA:
                    forward_batch.input_embeds.copy_(input_embeds)
                    input_embeds = forward_batch.input_embeds

        if input_embeds is not None and CP_METADATA:
            expected_local_tokens = sum(CP_METADATA.value.cur_split_list)
            if input_embeds.shape[0] != expected_local_tokens:
                input_embeds = cp_split_and_rebuild_data(
                    input_embeds,
                    CP_METADATA.value.split_list,
                    CP_METADATA.value.zigzag_index,
                )
        hidden_states = lm.model(input_ids, positions, forward_batch, input_embeds)

        aux_hidden_states = None
        if lm.capture_aux_hidden_states and isinstance(hidden_states, tuple):
            hidden_states, aux_hidden_states = hidden_states

        if lm.pp_group.is_last_rank:
            return lm.logits_processor(
                input_ids, hidden_states, lm.lm_head, forward_batch, aux_hidden_states
            )
        else:
            return hidden_states

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """Route and load checkpoint weights based on the running mode.

        Three modes (mutually exclusive):
          - **encoder_only** (EPD Encoder): loads only ``vision_tower``
            (encoder + bridge + quantizer).  Skips LM and visual_embedding_bridge
            MLP/embedding_layers — the Encoder outputs rvq_indices, not embeddings.
          - **language_only** (EPD LM): always loads ``language_model`` and skips
            ``vision_tower``.  Only Prefill PP0 additionally owns/loads
            ``visual_embedding_bridge``; Decode and Prefill PP1+ skip it.
          - **joint** (non-EPD): loads everything.

        Checkpoint key prefixes (merged ckpt):
          - ``model.visual_model.*``                → vision_tower (encoder + quantizer)
          - ``model.visual_bridge_model.embedding_layers.*`` → bridge embedding_layers
          - ``model.visual_bridge_model.transformer_block.*`` → bridge MLP
          - ``model.embed_tokens.weight``           → LM embed_tokens (sliced to text vocab)
          - everything else                         → LM weights
        """
        # ── Phase 1: route weights into buckets ──────────────────────────
        vision_encoder_weights = {}    # discrete_visualModel.* / visual_quantizer.*
        bridge_embedding_weights = {}  # embedding_layers.X.weight
        bridge_mlp_weights = {}        # transformer_block.*
        language_weights = []          # LM weights (including sliced embed_tokens)
        skipped = 0

        for name, loaded_weight in weights:
            if name.startswith("model.visual_model."):
                # vision_tower: only needed in encoder_only / joint mode
                if self.language_only:
                    skipped += 1
                    continue
                vision_encoder_weights[name[len("model.visual_model."):]] = loaded_weight

            elif name.startswith("model.visual_bridge_model.embedding_layers."):
                # Loaded only on the role/PP stage that constructed the bridge.
                if self.encoder_only and self.encoder_output_format == "rvq_indices":
                    skipped += 1
                    continue
                short_name = name[len("model.visual_bridge_model."):]
                if self.visual_embedding_bridge is None:
                    skipped += 1
                    continue
                key_parts = short_name.split(".")
                if len(key_parts) != 3 or not key_parts[1].isdigit():
                    raise ValueError(
                        f"Invalid visual bridge embedding checkpoint key: {name}"
                    )
                level = int(key_parts[1])
                if not self.visual_embedding_bridge.owns_level(level):
                    skipped += 1
                    continue
                bridge_embedding_weights[short_name] = loaded_weight

            elif name.startswith("model.visual_bridge_model.transformer_block."):
                # Loaded only on the role/PP stage that constructed the bridge.
                if self.encoder_only and self.encoder_output_format == "rvq_indices":
                    skipped += 1
                    continue
                if self.visual_embedding_bridge is None:
                    skipped += 1
                    continue
                short_name = name[len("model.visual_bridge_model.transformer_block."):]
                bridge_mlp_weights[short_name] = loaded_weight

            elif name == "model.embed_tokens.weight":
                # LM embed_tokens: not needed in encoder_only
                if self.encoder_only:
                    skipped += 1
                    continue
                # A merged checkpoint may contain rows beyond the configured LLM
                # vocabulary (for example visual bridge weights), so slice to the
                # embedding size declared by the text config.
                text_vocab_size = self.config.text_config.vocab_size
                sliced = loaded_weight[:text_vocab_size]
                logger.info(
                    "[load_weights] embed_tokens: full=%s sliced_to_text=%s "
                    "(text_vocab_size=%d)",
                    tuple(loaded_weight.shape), tuple(sliced.shape),
                    text_vocab_size,
                )
                language_weights.append((name, sliced))

            else:
                # LM weights: not needed in encoder_only
                if self.encoder_only:
                    skipped += 1
                    continue
                language_weights.append((name, loaded_weight))

        logger.info(
            "[load_weights] routing summary: encoder_only=%s language_only=%s "
            "skipped=%d vision_encoder=%d bridge_embedding=%d bridge_mlp=%d "
            "language=%d",
            self.encoder_only, self.language_only, skipped,
            len(vision_encoder_weights), len(bridge_embedding_weights),
            len(bridge_mlp_weights), len(language_weights),
        )

        # ── Phase 2: load each bucket ────────────────────────────────────
        # vision_tower (encoder + quantizer): encoder_only / joint
        if vision_encoder_weights:
            self._load_vision_encoder(vision_encoder_weights)

        # visual_embedding_bridge: Prefill PP0 / joint PP0 / optional Encoder
        if bridge_embedding_weights:
            self._load_bridge_embedding_layers(bridge_embedding_weights)
        if bridge_mlp_weights:
            self._load_emb_bridge_mlp_weights(bridge_mlp_weights)

        # language_model: language_only / joint
        if language_weights:
            logger.info(
                "[load_weights] calling language_model.load_weights with %d tensors",
                len(language_weights),
            )
            self.language_model.load_weights(language_weights)

    def _load_vision_encoder(self, encoder_weights: dict):
        """Load vision_tower weights (encoder + bridge + quantizer).

        Maps checkpoint key prefixes to vision_tower parameter names:
          - ``discrete_visualModel.*``    → ``encoder.*`` (QKV fused → qkv_proj)
          - ``discrete_visualBridgeModel.*`` → ``bridge.*``
          - ``visual_quantizer.*``        → ``quantizer.*`` (codebook rename)
        """
        key_mapping = {}
        for key in encoder_weights:
            if key.startswith("discrete_visualModel."):
                new_key = "encoder." + key[len("discrete_visualModel."):]
                # checkpoint stores fused QKV as "attn.qkv.*",
                # model uses "attn.qkv_proj.*"
                if ".attn.qkv." in new_key:
                    new_key = new_key.replace(".attn.qkv.", ".attn.qkv_proj.")
                key_mapping[key] = new_key
            elif key.startswith("discrete_visualBridgeModel."):
                new_key = "bridge." + key[len("discrete_visualBridgeModel."):]
                key_mapping[key] = new_key
            elif key.startswith("visual_quantizer."):
                new_key = "quantizer." + key[len("visual_quantizer."):]
                if "embed_ema" in new_key or "cluster_size_ema" in new_key:
                    continue
                if "quantize.codebooks.0.embed" in new_key:
                    new_key = new_key.replace(
                        "quantize.codebooks.0.embed", "codebook.embed"
                    )
                key_mapping[key] = new_key

        params_dict = dict(self.vision_tower.named_parameters())
        loaded_count = 0
        for old_key, new_key in key_mapping.items():
            if self._load_vision_weight(new_key, encoder_weights[old_key], params_dict):
                loaded_count += 1

        logger.info(
            "[load_weights] loaded vision encoder: %d/%d params",
            loaded_count, len(key_mapping),
        )

    def _load_bridge_embedding_layers(self, embedding_weights: dict):
        """Load visual_embedding_bridge.embedding_layers.X.weight.

        These are nn.Embedding tables (one per RVQ level) that map rvq_indices
        to embeddings.  Required by language_only / joint mode to decode
        rvq_indices produced by the Encoder.
        """
        assert self.visual_embedding_bridge is not None
        params_dict = dict(self.visual_embedding_bridge.named_parameters())
        expected_keys = {
            f"embedding_layers.{level}.weight"
            for level in self.visual_embedding_bridge.owned_levels
        }
        missing_keys = expected_keys - embedding_weights.keys()
        if missing_keys:
            raise ValueError(
                "Missing owned visual bridge embedding weights: "
                f"{sorted(missing_keys)}"
            )
        logger.info(
            "[load_weights] loading owned bridge embedding levels=%s keys=%s",
            self.visual_embedding_bridge.owned_levels,
            {k: tuple(v.shape) for k, v in embedding_weights.items()},
        )
        for key, weight in embedding_weights.items():
            if key in params_dict:
                param = params_dict[key]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, weight)
                logger.info(
                    "[load_weights] loaded bridge embedding: %s shape=%s",
                    key, tuple(weight.shape),
                )
            else:
                logger.warning(
                    "[load_weights] bridge embedding key not found: %s", key
                )

    def _load_vision_weight(
        self, name: str, weight: torch.Tensor, params_dict: dict
    ) -> bool:
        """Load one vision weight, applying stacked_params_mapping.
        """

        stacked_params_mapping = [
            ("gate_up_proj", ".gate_proj", 0),
            ("gate_up_proj", ".up_proj", 1),
        ]
        for param_name, shard_name, shard_id in stacked_params_mapping:
            if shard_name not in name:
                continue
            fused_name = name.replace(shard_name, "." + param_name)
            if fused_name not in params_dict:
                continue
            param = params_dict[fused_name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, weight, shard_id)
            return True

        if name not in params_dict:
            logger.warning(f"Vision weight {name} not found in model parameters")
            return False

        param = params_dict[name]
        # Codebook must stay float32 for VQ distance precision.
        # Checkpoint stores [codebook_size+1, dim] (extra EMA slot);
        # model param is [codebook_size, dim] — slice to fit.
        if name == "quantizer.codebook.embed":
            weight_fp32 = weight.float().to(param.device)
            if weight_fp32.shape[0] > param.shape[0]:
                logger.info(
                    "[load_weights] codebook embed: checkpoint rows=%d > param rows=%d, "
                    "truncating extra EMA slot",
                    weight_fp32.shape[0], param.shape[0],
                )
                weight_fp32 = weight_fp32[:param.shape[0]]
            param.data = weight_fp32
        else:
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, weight)
        return True

    def _load_emb_bridge_mlp_weights(self, state_dict: dict):
        assert self.visual_embedding_bridge is not None
        params_dict = dict(self.visual_embedding_bridge.named_parameters())

        stacked_params_mapping = [
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        prefix_mapping = {
            "mlp.down_proj.": "down_proj.",
            "post_attention_layernorm.": "post_attention_layernorm.",
        }
        for key, weight in state_dict.items():
            # stacked: mlp.gate_proj.* / mlp.up_proj.* -> gate_up_proj.* (shard 0/1)
            shard_hit = False
            for param_name, shard_name, shard_id in stacked_params_mapping:
                cp_key = f"mlp.{shard_name}."
                if key.startswith(cp_key):
                    fused_key = param_name + "." + key[len(cp_key):]
                    if fused_key in params_dict:
                        param = params_dict[fused_key]
                        weight_loader = getattr(param, "weight_loader", default_weight_loader)
                        weight_loader(param, weight, shard_id)
                        shard_hit = True
                    else:
                        logger.warning(f"EmbBridge weight {key} → {fused_key} not found")
                    break
            if shard_hit:
                continue

            mapped_key = key
            for old_prefix, new_prefix in prefix_mapping.items():
                if key.startswith(old_prefix):
                    mapped_key = new_prefix + key[len(old_prefix):]
                    break
            if mapped_key in params_dict:
                param = params_dict[mapped_key]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, weight)
            else:
                logger.warning(f"EmbBridge weight {key} → {mapped_key} not found")
    @property
    def attn_start_layer(self):
        return self.language_model.attn_start_layer

    @property
    def attn_end_layer(self):
        return self.language_model.attn_end_layer





EntryClass = [LongcatFlashVLForConditionalGeneration]
