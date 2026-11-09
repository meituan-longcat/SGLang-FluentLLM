"""
Longcat Flash VL Model Configuration.

Supports: LongcatFlashVLForConditionalGeneration
  - model_type = "omni"  -> FlashA3B VL checkpoints
  - model_type = "pro_v" -> Pro-V VL checkpoints (same architecture, larger LLM)

Config.json layout (flat, no text_config sub-dict):
  - LLM params at top level: hidden_size, intermediate_size, rms_norm_eps, ...
  - visual_config: ViT + Bridge config
  - video_config:  Video processing config
  - visual_quantizer_config: RVQ config
  - visual_offset: embedding offset for VQ levels
  - vocab_size: LLM vocab (lm_head / embed_tokens size and OE extension base)

Bridge params are NOT in config.json; derived from text_config at runtime.
"""

from typing import Optional

from transformers.configuration_utils import PretrainedConfig

from sglang.srt.configs.configuration_flash import FLASHConfig


class LongcatVLVisualConfig(PretrainedConfig):
    """Visual encoder configuration (Qwen2.5-VL ViT + Bridge)."""

    model_type = "longcat_vl_visual"

    def __init__(
        self,
        hidden_size: int = 1280,
        depth: int = 32,
        patch_size: int = 14,
        temporal_patch_size: int = 2,
        merge_size: int = 2,
        spatial_merge_size: int = 2,
        num_heads: int = 16,
        mlp_ratio: int = 4,
        intermediate_size: int = None,
        out_hidden_size: int = 3584,
        in_channels: int = 3,
        fullatt_block_indexes=None,
        window_size: int = 112,
        enable: bool = True,
        bridge_intermediate_size: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.depth = depth
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.merge_size = merge_size
        self.spatial_merge_size = spatial_merge_size
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.intermediate_size = intermediate_size if intermediate_size is not None else hidden_size * mlp_ratio
        self.out_hidden_size = out_hidden_size
        self.in_channels = in_channels
        self.fullatt_block_indexes = fullatt_block_indexes or [7, 15, 23, 31]
        self.window_size = window_size
        self.enable = enable
        self.bridge_intermediate_size = bridge_intermediate_size

class LongcatVLQuantizerConfig(PretrainedConfig):
    """Visual quantizer configuration (8-level RVQ)."""

    model_type = "longcat_vl_quantizer"

    def __init__(
        self,
        depth: int = 8,
        codebook_size: int = 16384,
        codebook_dim: int = 3584,
        in_channels: int = 3584,
        shared_codebook: bool = True,
        decay: float = 0.99,
        quant_conv: bool = True,
        quantizer_type: str = "rq",
        codebook_sizes=None,
        enable: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.depth = depth
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.in_channels = in_channels
        self.shared_codebook = shared_codebook
        self.decay = decay
        self.quant_conv = quant_conv
        self.quantizer_type = quantizer_type
        self.enable = enable
        # codebook_sizes: list of per-level sizes; default to depth copies of codebook_size
        self.codebook_sizes = codebook_sizes or [codebook_size] * depth


class LongcatVLVideoConfig(PretrainedConfig):
    """Video processing configuration."""

    model_type = "longcat_vl_video"

    def __init__(
        self,
        # Video special token IDs (verified against tokenizer.json)
        video_start_token_id: int = 131118,
        video_end_token_id: int = 131119,
        video_place_token_id: int = 131117,  # <video_palce> (typo in tokenizer)
        # Video processing parameters (from config.json video_config)
        max_frame_num: int = 128,
        min_pixels: int = 78400,       # 100*28*28 / spatial_merge_size^2 ≈ 78400
        max_pixels: int = 150528,      # from config.json video_config.max_pixels
        decode_way: str = "1fps",
        split_video: bool = True,
        frame_pattern: str = "<frame><TIMEIDX>",
        temporal_patch_size: int = 2,
        spatial_merge_size: int = 2,
        patch_size: int = 14,
        merge_size: int = 2,
        enable: bool = True,
        dynamic_frame_pixel: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.video_start_token_id = video_start_token_id
        self.video_end_token_id = video_end_token_id
        self.video_place_token_id = video_place_token_id
        self.max_frame_num = max_frame_num
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.decode_way = decode_way
        self.split_video = split_video
        self.frame_pattern = frame_pattern
        self.temporal_patch_size = temporal_patch_size
        self.spatial_merge_size = spatial_merge_size
        self.patch_size = patch_size
        self.merge_size = merge_size
        self.enable = enable
        self.dynamic_frame_pixel = dynamic_frame_pixel


class LongcatFlashVLConfig(PretrainedConfig):
    """Longcat Flash VL model configuration (VL wrapper over FLASHForCausalLM).

    config.json is flat — all LLM params sit at the top level with no
    'text_config' sub-dict.  The LLM uses ``vocab_size`` for its embedding,
    lm_head, and OE extension base.

    Bridge attributes are derived at runtime from text_config.
    """

    model_type = "omni"

    MODEL_TYPES = ("omni", "pro_v")

    def __init__(
        self,
        text_config=None,
        visual_config=None,
        visual_quantizer_config=None,
        video_config=None,
        visual_offset: int = 150581, # FIXME，权重拆分后去掉
        codebook_sizes=None,
        oe_ignore_tokens=None,
        **kwargs,
    ):
        # --- text_config ---
        if text_config is None:
            flash_kwargs = dict(kwargs)
            # Forward oe_ignore_tokens to FLASHConfig so text_config.oe_ignore_tokens
            # stays consistent with ours.
            if oe_ignore_tokens is not None and "oe_ignore_tokens" not in flash_kwargs:
                flash_kwargs["oe_ignore_tokens"] = oe_ignore_tokens

            text_config = FLASHConfig(**flash_kwargs)

        elif isinstance(text_config, dict):
            text_config = FLASHConfig(**text_config)

        if getattr(text_config, "architectures", None) in (
            ["LongcatCausalLM"],
            ["LongcatFlashForCausalLM"],
            ["LongcatFlashNgramForCausalLM"],
        ):
            text_config.architectures = ["FLASHForCausalLM"]

        # --- visual_config ---
        if visual_config is None:
            visual_config = LongcatVLVisualConfig()
        elif isinstance(visual_config, dict):
            visual_config = LongcatVLVisualConfig(**visual_config)

        # --- quantizer_config ---
        if visual_quantizer_config is None:
            visual_quantizer_config = LongcatVLQuantizerConfig()
        elif isinstance(visual_quantizer_config, dict):
            visual_quantizer_config = LongcatVLQuantizerConfig(**visual_quantizer_config)

        # --- video_config ---
        if video_config is None:
            video_config = LongcatVLVideoConfig()
        elif isinstance(video_config, dict):
            video_config = LongcatVLVideoConfig(**video_config)

        self.text_config = text_config
        self.visual_config = visual_config
        self.visual_quantizer_config = visual_quantizer_config
        self.video_config = video_config

        self.visual_offset = visual_offset
        if codebook_sizes is not None:
            self.codebook_sizes = codebook_sizes
        else:
            self.codebook_sizes = visual_quantizer_config.codebook_sizes

        # oe_ignore_tokens is fully resolved inside FLASHConfig
        self.oe_ignore_tokens = getattr(text_config, "oe_ignore_tokens", None)

        super().__init__(**kwargs)

    # ------------------------------------------------------------------
    # Convenience aliases used by model code
    # ------------------------------------------------------------------

    @classmethod
    def get_model_config_for_expert_location(cls, config):
        return FLASHForCausalLM.get_model_config_for_expert_location(config.text_config)

    @property
    def vision_config(self):
        return self.visual_config

    @property
    def quantizer_config(self):
        return self.visual_quantizer_config

    @property
    def use_over_embedding(self) -> bool:
        # model_config.py line 133 reads hf_config.use_over_embedding directly
        return getattr(self.text_config, "use_over_embedding", False)

    # ------------------------------------------------------------------
    # Bridge properties — derived from text_config, not stored in JSON.
    # EmbeddingBridge outputs into LLM embedding space.
    # ------------------------------------------------------------------

    @property
    def bridge_hidden_size(self) -> int:
        return self.text_config.hidden_size

    @property
    def bridge_intermediate_size(self) -> int:
        val = getattr(self.visual_config, "bridge_intermediate_size", None)
        if val is not None:
            return val
        return self.text_config.intermediate_size

    @property
    def bridge_hidden_act(self) -> str:
        return getattr(self.text_config, "hidden_act", "silu")

    @property
    def bridge_rms_norm_eps(self) -> float:
        return self.text_config.rms_norm_eps


class LongcatFlashVLProVConfig(LongcatFlashVLConfig):
    """Alias config for Pro-V VL checkpoints.

    Pro-V shares the exact same architecture and config.json layout as the
    omni FlashA3B VL model; only the ``model_type`` string differs ("pro_v"
    vs "omni").  Subclassing lets us register a distinct ``model_type`` so
    HuggingFace's ``from_pretrained`` model_type check passes cleanly, without
    duplicating any config logic.
    """

    model_type = "pro_v"
