from sglang.srt.layers.quantization import (
    CompressedTensorsConfig,
    Fp8Config,
    W8A8Fp8Config,
    W8A8Int8Config,
)
from sglang.srt.utils import is_sm90_supported

if is_sm90_supported():
    from sglang.srt.layers.moe.gemms.bf16.aok import AokGroupedGemmWrapper

from sglang.srt.layers.quantization.utils import should_ignore_quant_layer


def get_gemm_cls(quant_config, is_ep: bool, prefix: str = ""):
    """Return the appropriate GEMM wrapper class based on quantization config and EP mode.

    Args:
        quant_config: Quantization configuration (Fp8Config, W8A8Fp8Config, etc.)
        is_ep: Whether expert parallelism is enabled
        prefix: Layer prefix for ignored_layers check

    Returns:
        A GEMM wrapper class (e.g., AokGroupedGemmWrapper, TritonGemmWrapper, etc.)

    Raises:
        RuntimeError: If the quantization configuration is not supported
    """
    # Handle ignored layers or no quantization
    if quant_config is None or should_ignore_quant_layer(prefix=prefix, ignored_layers=getattr(quant_config, "ignored_layers", [])):
        return AokGroupedGemmWrapper if is_ep and is_sm90_supported() else (_import_triton_gemm("bf16"))

    # CompressedTensorsConfig
    if isinstance(quant_config, CompressedTensorsConfig):
        weight_quant = quant_config.target_scheme_map["Linear"].get("weights")
        input_quant = quant_config.target_scheme_map["Linear"].get("input_activations")
        if quant_config._is_wNa16_group_channel(weight_quant, input_quant):
            from sglang.srt.layers.moe.gemms.wna16 import WNA16MoEGemmWrapper

            return WNA16MoEGemmWrapper
        raise RuntimeError(f"Unsupported FusedMoe scheme: {weight_quant}, {input_quant}")

    # Fp8Config with block quantization
    if isinstance(quant_config, Fp8Config) and quant_config.weight_block_size is not None:
        return _import_fire_grouped_gemm() if is_ep else _import_triton_gemm("fp8")

    # W8A8 quantization configs
    if isinstance(quant_config, W8A8Fp8Config):
        return _import_triton_gemm("w8a8_fp8")

    if isinstance(quant_config, W8A8Int8Config):
        return _import_triton_gemm("w8a8_int8")

    raise RuntimeError(f"Unsupported quant_config: {quant_config}")


def _import_triton_gemm(module: str):
    from sglang.srt.layers.moe.gemms.bf16.triton import TritonGemmWrapper as BF16Wrapper
    from sglang.srt.layers.moe.gemms.fp8.triton import TritonGemmWrapper as FP8Wrapper
    from sglang.srt.layers.moe.gemms.w8a8_fp8.triton import TritonGemmWrapper as W8A8FP8Wrapper
    from sglang.srt.layers.moe.gemms.w8a8_int8.triton import TritonGemmWrapper as W8A8Int8Wrapper

    wrappers = {
        "bf16": BF16Wrapper,
        "fp8": FP8Wrapper,
        "w8a8_fp8": W8A8FP8Wrapper,
        "w8a8_int8": W8A8Int8Wrapper,
    }
    return wrappers[module]


def _import_fire_grouped_gemm():
    from sglang.srt.layers.moe.gemms.fp8.fire import FireGroupedGemmWrapper

    return FireGroupedGemmWrapper
