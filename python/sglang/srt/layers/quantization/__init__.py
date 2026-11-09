# from typing import Dict, Type
#
#
# class DummyConfig:
#     def override_quantization_method(self, *args, **kwargs):
#         return None
#
# AQLMConfig = BitsAndBytesConfig = CompressedTensorsConfig = DeepSpeedFPConfig = (
#     ExpertsInt8Config
# ) = GGUFConfig = GPTQMarlin24Config = MarlinConfig = QQQConfig = Int8TpuConfig = (
#     DummyConfig
# )
#
#
# from sglang.srt.layers.quantization.awq import AWQConfig, AWQMarlinConfig, AWQMoEMethod
# from sglang.srt.layers.quantization.base_config import QuantizationConfig
# from sglang.srt.layers.quantization.awq import AWQConfig, AWQMarlinConfig
# from sglang.srt.layers.quantization.blockwise_int8 import BlockInt8Config
# from sglang.srt.layers.quantization.gptq import GPTQConfig, GPTQMarlinConfig
# from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
#     CompressedTensorsConfig,
# )
#
from enum import Enum
from typing import Dict, Type


from sglang.srt.layers.quantization.base_config import QuantizationConfig, QuantizeMethodBase
from sglang.srt.layers.quantization.fp8 import Fp8Config
from sglang.srt.layers.quantization.w8a8_fp8 import W8A8Fp8Config
from sglang.srt.layers.quantization.w8a8_int8 import W8A8Int8Config
from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
)


BASE_QUANTIZATION_METHODS: Dict[str, Type[QuantizationConfig]] = {
    "fp8": Fp8Config,
    # "blockwise_int8": BlockInt8Config,
    "w8a8_int8": W8A8Int8Config,
    "w8a8_fp8": W8A8Fp8Config,
    "compressed-tensors": CompressedTensorsConfig,
    # "awq": AWQConfig,
    # "awq_marlin": AWQMarlinConfig,
    # "gptq": GPTQConfig,
    # "gptq_marlin": GPTQMarlinConfig,
}

QUANTIZATION_METHODS = BASE_QUANTIZATION_METHODS


#
#
def get_quantization_config(quantization: str) -> Type[QuantizationConfig]:
    if quantization not in QUANTIZATION_METHODS:
        raise ValueError(f"Invalid quantization method: {quantization}. Available methods: {list(QUANTIZATION_METHODS.keys())}")
    return QUANTIZATION_METHODS[quantization]


#
#
# __all__ = [
#     "QuantizationConfig",
#     "get_quantization_config",
#     "QUANTIZATION_METHODS",
# ]
#


# TODO: lifengcun, move to
class FusedMoeWeightScaleSupported(Enum):
    TENSOR = "tensor"
    CHANNEL = "channel"
    GROUP = "group"
    BLOCK = "block"
