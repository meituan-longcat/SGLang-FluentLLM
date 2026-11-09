from typing import Any, Dict, List

import torch

from sglang.srt.layers.quantization import QuantizationConfig


class W8A8Fp8Config(QuantizationConfig):
    """Config class for W8A8 FP8 Quantization.

    Weight Quantization:
    - Method: Static quantization
    - Granularity: Per-channel
    - Type: Symmetric

    Activation Quantization:
    - Method: Dynamic quantization
    - Granularity: Per-token
    - Type: Symmetric

    Note:
    - For models without offline quantization, weights will be quantized during model loading:
        - If CUTLASS is supported: Per-channel weight quantization is used
        - If CUTLASS is not supported: Falls back to per-tensor weight quantization
    """

    def __init__(self, is_checkpoint_fp8_serialized: bool = False):
        self.is_checkpoint_fp8_serialized = is_checkpoint_fp8_serialized

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.float16, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 89

    @classmethod
    def get_name(self) -> str:
        return "w8a8_fp8"

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return []

    @classmethod
    def from_config(cls, config: Dict[str, Any]):
        quant_method = cls.get_from_keys(config, ["quant_method"])
        is_checkpoint_fp8_serialized = "compressed-tensors" in quant_method or "w8a8_fp8" in quant_method
        return cls(is_checkpoint_fp8_serialized=is_checkpoint_fp8_serialized)

    def get_scaled_act_names(self) -> List[str]:
        return []
