from __future__ import annotations

import functools
import json
import os
from typing import Any, Dict, List, Optional, Tuple
from contextlib import contextmanager

import torch

from sglang.srt.utils import get_colorful_logger, get_device_name, is_hip

logger = get_colorful_logger(__name__)
_is_hip = is_hip()

_config: Optional[Dict[str, Any]] = None


@contextmanager
def override_config(config):
    global _config
    old_config = _config
    _config = config
    yield
    _config = old_config


def get_config() -> Optional[Dict[str, Any]]:
    return _config


def get_config_file_name(
    E: int,
    inter_size: int,
    hidden_size: int,
    dtype: Optional[str],
    block_shape: Optional[int] = None,
    per_channel_quant: bool = False,
    down_moe: bool = False,
) -> str:
    device_name = get_device_name().replace(" ", "_")
    dtype_selector = "" if not dtype else f",dtype={dtype}"
    block_shape_selector = "" if not block_shape or not all(block_shape) else f",block_shape={block_shape}"
    per_channel_quant_selector = ",per_channel_quant=True" if per_channel_quant else ""
    down_moe_selector = "_down" if down_moe else ""
    return f"{E=},{inter_size=},{hidden_size=},device_name={device_name}{dtype_selector}{block_shape_selector}{per_channel_quant_selector}{down_moe_selector}.json"


def get_config_file_path(
    E: int,
    inter_size: int,
    hidden_size: int,
    dtype: Optional[str],
    block_n: Optional[int] = 0,
    block_k: Optional[int] = 0,
    per_channel_quant: bool = False,
    down_moe: bool = False,
):
    # First look up if an optimized configuration is available in the configs
    # directory
    json_file_name = get_config_file_name(
        E,
        inter_size,
        hidden_size,
        dtype,
        [block_n, block_k],
        per_channel_quant,
        down_moe=down_moe,
    )

    # We found that using the fused_moe_kernel config from Triton 3.1.0 with Triton 3.2.0 results in negative performance gains,
    # so we also include the Triton version as a key for finding the fused_moe_kernel config to achieve the best performance.
    config_dir = os.environ.get("SGLANG_MOE_CONFIG_DIR", os.path.dirname(os.path.realpath(__file__)))

    config_file_path = os.path.join(
        config_dir,
        json_file_name,
    )
    return config_file_path


@functools.lru_cache
def get_moe_configs(
    E: int,
    inter_size: int,
    hidden_size: int,
    dtype: Optional[str],
    block_n: Optional[int] = 0,
    block_k: Optional[int] = 0,
    per_channel_quant: bool = False,
    down_moe: bool = False,
) -> Optional[Dict[int, Any]]:
    """
    Return optimized configurations for the fused MoE kernel.

    The return value will be a dictionary that maps an irregular grid of
    batch sizes to configurations of the fused_moe kernel. To evaluate the
    kernel on a given batch size bs, the closest batch size in the grid should
    be picked and the associated configuration chosen to invoke the kernel.
    """
    config_file_path = get_config_file_path(
        E,
        inter_size,
        hidden_size,
        dtype,
        block_n,
        block_k,
        per_channel_quant,
        down_moe,
    )
    if os.path.exists(config_file_path):
        with open(config_file_path) as f:
            # Please note that although we find the config files, performance might still be suboptimal.
            # This is because the tuning environment might differ from your current environment.
            # For example, updating the Triton version might cause all old configs to become suboptimal.
            # To achieve the best performance, consider re-tuning the Triton fused MOE kernel in your environment.
            # For the tuning method, refer to: https://github.com/sgl-project/sglang/tree/main/benchmark/kernels/fused_moe_triton
            logger.info(f"Using MoE kernel config from {config_file_path}.")
            # If a configuration has been found, return it
            return {int(key): val for key, val in json.load(f).items()}

    # If no optimized configuration is available, we will use the default
    # configuration
    logger.warning(
        (
            "Using default MoE kernel config. Performance might be sub-optimal! "
            "Config file not found at %s, you can create them with https://github.com/sgl-project/sglang/tree/main/benchmark/kernels/fused_moe_triton"
        ),
        config_file_path,
    )
    return None


def get_default_config(
    M: int,
    E: int,
    inter_size: int,
    hidden_size: int,
    topk: int,
    dtype: Optional[str],
    is_marlin: bool,
    block_shape: Optional[List[int]] = None,
) -> Dict[str, int]:
    if dtype == "fp8_w8a8":
        if block_shape is None:
            config = {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 256,
                "BLOCK_SIZE_K": 128,
                "GROUP_SIZE_M": 32,
                "num_warps": 8,
                "num_stages": 2 if _is_hip else 4,
            }
            if M <= E:
                config = {
                    "BLOCK_SIZE_M": 64,
                    "BLOCK_SIZE_N": 128,
                    "BLOCK_SIZE_K": 128,
                    "GROUP_SIZE_M": 1,
                    "num_warps": 4,
                    "num_stages": 2 if _is_hip else 4,
                }
        else:
            # Block-wise quant: BLOCK_SIZE_K must be divisible by block_shape[1]
            config = {
                "BLOCK_SIZE_M": 64,
                "BLOCK_SIZE_N": block_shape[0],
                "BLOCK_SIZE_K": block_shape[1],
                "GROUP_SIZE_M": 32,
                "num_warps": 4,
                "num_stages": 2 if _is_hip else 3,
            }
    else:
        config = {
            "BLOCK_SIZE_M": 64,
            "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32,
            "GROUP_SIZE_M": 8,
        }
        # A heuristic: fused marlin works faster with this config for small M
        if M <= E or (is_marlin and M <= 32):
            config = {
                "BLOCK_SIZE_M": 16,
                "BLOCK_SIZE_N": 32,
                "BLOCK_SIZE_K": 64,
                "GROUP_SIZE_M": 1,
            }
    return config


def try_get_optimal_moe_config(
    w1_shape: Tuple[int, ...],
    w2_shape: Tuple[int, ...],
    top_k: int,
    dtype: Optional[str],
    M: int,
    is_marlin: bool = False,
    block_shape: Optional[List[int]] = None,
    return_down_config: bool = False,
):
    down_config = None
    max_block_m = None
    override_config = get_config()
    if override_config:
        config = override_config
    else:
        # First try to load optimal config from the file
        E, hidden_size, inter_size = w2_shape
        block_n = block_shape[0] if block_shape else 0
        block_k = block_shape[1] if block_shape else 0

        configs = get_moe_configs(E, inter_size, hidden_size, dtype, block_n, block_k, down_moe=False)
        if configs:
            # If an optimal configuration map has been found, look up the
            # optimal config
            config = configs[min(configs.keys(), key=lambda x: abs(x - M))]
        else:
            # Else use the default config
            config = get_default_config(M, E, inter_size, hidden_size, top_k, dtype, is_marlin, block_shape)

        if return_down_config:
            down_configs = get_moe_configs(E, inter_size, hidden_size, dtype, block_n, block_k, down_moe=True)
            if down_configs:
                down_config = down_configs[min(down_configs.keys(), key=lambda x: abs(x - M))]
                down_config = dict(**down_config)
                max_block_m = max([cfg["BLOCK_SIZE_M"] for cfg in down_configs.values()])
            else:
                # FIXME: lifengcun
                down_config = get_default_config(M, E, inter_size, hidden_size, top_k, dtype, is_marlin, block_shape)
                max_block_m = down_config["BLOCK_SIZE_M"]

    if return_down_config:
        assert down_config is None or config["BLOCK_SIZE_M"] == down_config["BLOCK_SIZE_M"]
        return config, (down_config, max_block_m)
    return config


def get_config_dtype_str(
    dtype: torch.dtype,
    use_int8_w8a16: Optional[bool] = False,
    use_int4_w4a16: Optional[bool] = False,
    use_fp8_w8a8: Optional[bool] = False,
    use_int8_w8a8: Optional[bool] = False,
):
    if use_fp8_w8a8:
        return "fp8_w8a8"
    elif use_int8_w8a8:
        return "int8_w8a8"
    elif use_int4_w4a16:
        return "int4_w4a16"
    elif use_int8_w8a16:
        return "int8_w8a16"
    elif dtype == torch.float:
        # avoiding cases where kernel fails when float32 MoE
        # use fp16/bfloat16 configs
        return "float32"
    return None
