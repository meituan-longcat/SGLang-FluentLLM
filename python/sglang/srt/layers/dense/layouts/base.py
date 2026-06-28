import torch
from typing import Optional

from sglang.srt.layers.dense.layouts.w8a8_int8 import W8A8Int8LinearMethod
from sglang.srt.layers.quantization import Fp8Config, QuantizationConfig, QuantizeMethodBase, W8A8Fp8Config, W8A8Int8Config, CompressedTensorsConfig
from sglang.srt.layers.dense.layouts.unquant import UnquantizedLinearMethod
from sglang.srt.layers.quantization.utils import should_ignore_quant_layer


def get_layout(config, prefix):
    if isinstance(config, Fp8Config):
        from sglang.srt.layers.dense.layouts.fp8 import Fp8LinearMethod
        return Fp8LinearMethod(config)
    if isinstance(config, W8A8Fp8Config):
        from sglang.srt.layers.dense.layouts.w8a8_fp8 import W8A8Fp8LinearMethod
        return W8A8Fp8LinearMethod(config)
    if isinstance(config, W8A8Int8Config):
        return W8A8Int8LinearMethod(config)

    raise RuntimeError(f"Unsupported config: {config}, prefix: {prefix}")


class LinearBase(torch.nn.Module):
    """Base linear layer.

    Args:
        input_size: input dimension of the linear layer.
        output_size: output dimension of the linear layer.
        bias: If true, add bias.
        skip_bias_add: If true, skip adding bias but instead return it.
        params_dtype: Data type for the parameters.
        quant_config: Quantization configure.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        skip_bias_add: bool = False,
        params_dtype: Optional[torch.dtype] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()

        # Keep input parameters
        self.input_size = input_size
        self.output_size = output_size
        self.skip_bias_add = skip_bias_add
        if params_dtype is None:
            params_dtype = torch.get_default_dtype()
        self.params_dtype = params_dtype
        self.prefix = prefix

        self.quant_config = quant_config
        if quant_config is None or should_ignore_quant_layer(
            prefix=prefix,
            ignored_layers=getattr(quant_config, "ignored_layers", [])
        ):
            self.quant_method: Optional[QuantizeMethodBase] = UnquantizedLinearMethod()
        elif isinstance(quant_config, CompressedTensorsConfig):
            self.quant_method = quant_config.get_quant_method(self, prefix)
        else:
            self.quant_method = get_layout(quant_config, prefix)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
