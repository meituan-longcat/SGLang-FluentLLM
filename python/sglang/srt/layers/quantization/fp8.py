# Adapted from https://github.com/vllm-project/vllm/blob/v0.6.4.post1/vllm/model_executor/layers/quantization/fp8.py

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import torch

MARLIN_FP8_AVAILABLE = False


def dummy_func(*args, **kwargs):
    raise ImportError("marlin FP8 requires some operators from vllm. Please install vllm.")


apply_fp8_marlin_linear = prepare_fp8_layer_for_marlin = dummy_func


from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.utils import (
    cpu_has_amx_support,
    get_bool_env_var,
    is_cpu,
    is_cuda,
    is_hip,
    is_npu,
    log_info_on_rank0,
)

_is_hip = is_hip()
_is_cuda = is_cuda()
_is_npu = is_npu()
_is_cpu_amx_available = cpu_has_amx_support()
_is_cpu = is_cpu()

_use_hip_int4 = get_bool_env_var("SGLANG_INT4_WEIGHT")
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip


ACTIVATION_SCHEMES = ["static", "dynamic"]

logger = logging.getLogger(__name__)


class Fp8Config(QuantizationConfig):
    """Config class for FP8."""

    def __init__(
        self,
        is_checkpoint_fp8_serialized: bool = False,
        activation_scheme: str = "dynamic",
        ignored_layers: Optional[List[str]] = None,
        weight_block_size: List[int] = None,
    ) -> None:
        self.is_checkpoint_fp8_serialized = is_checkpoint_fp8_serialized
        if is_checkpoint_fp8_serialized:
            log_info_on_rank0(logger, "Detected fp8 checkpoint.")
        if activation_scheme not in ACTIVATION_SCHEMES:
            raise ValueError(f"Unsupported activation scheme {activation_scheme}")
        self.activation_scheme = activation_scheme
        self.ignored_layers = ignored_layers or []
        if weight_block_size is not None:
            if not is_checkpoint_fp8_serialized:
                raise ValueError(f"The block-wise quantization only supports fp8-serialized checkpoint for now.")
            if len(weight_block_size) != 2:
                raise ValueError(f"The quantization block size of weight must have 2 dimensions, but got {len(weight_block_size)} dimensions.")
            if activation_scheme != "dynamic":
                raise ValueError(
                    f"The block-wise quantization only supports dynamic activation scheme for now, but got {activation_scheme} activation scheme."
                )
        self.weight_block_size = weight_block_size

    @classmethod
    def get_name(cls) -> str:
        return "fp8"

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.bfloat16, torch.half]

    @classmethod
    def get_min_capability(cls) -> int:
        return 80

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return []

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> Fp8Config:
        quant_method = cls.get_from_keys(config, ["quant_method"])
        is_checkpoint_fp8_serialized = "fp8" in quant_method
        activation_scheme = cls.get_from_keys(config, ["activation_scheme"])
        ignored_layers = cls.get_from_keys_or(config, ["ignored_layers"], None)
        weight_block_size = cls.get_from_keys_or(config, ["weight_block_size"], None)
        return cls(
            is_checkpoint_fp8_serialized=is_checkpoint_fp8_serialized,
            activation_scheme=activation_scheme,
            ignored_layers=ignored_layers,
            weight_block_size=weight_block_size,
        )

    def get_scaled_act_names(self) -> List[str]:
        return []
