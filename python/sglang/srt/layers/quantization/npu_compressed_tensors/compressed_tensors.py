from typing import Any, Dict, List, Optional, cast

import torch
from vllm.model_executor.layers.linear import LinearMethodBase

from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsLinearMethod as GPUCompressedTensorsLinearMethod,
    CompressedTensorsLinearMethod, CompressedTensorsKVCacheMethod)
#from vllm.model_executor.layers.quantization.compressed_tensors.utils import should_ignore_layer

#from sglang.srt.layers.linear import UnquantizedLinearMethod, LinearBase
#from sglang.srt.layers.moe.npu_moe.layer import NpuEPMoE, NpuUnquantizedFusedMoEMethod
#from sglang.srt.layers.quantization.npu_compressed_tensors.compressed_tensors_moe import NpuCompressedTensorsMoEMethod

"""
def get_quant_method(
    self,
    layer: torch.nn.Module,
    prefix: str,
) -> Optional["QuantizeMethodBase"]:
    from vllm.attention.layer import Attention  # Avoid circular import

    # Check if the layer is skipped for quantization.
    # TODO (@robertgshaw2): support module names
    if should_ignore_layer(prefix, ignore=self.ignore):
        return UnquantizedLinearMethod()
    if isinstance(layer, LinearBase):
        scheme = self.get_scheme(layer=layer, layer_name=prefix)
        if scheme is None:
            return UnquantizedLinearMethod()
        layer.scheme = scheme
        return NpuCompressedTensorsLinearMethod(self)
    if isinstance(layer, Attention):
        return CompressedTensorsKVCacheMethod(self)
    if isinstance(layer, NpuEPMoE):
        return NpuCompressedTensorsMoEMethod.get_moe_method(quant_config=layer.quant_config)
    return None
"""

class NpuCompressedTensorsLinearMethod(LinearMethodBase):
    process_weights_after_loading = GPUCompressedTensorsLinearMethod.process_weights_after_loading
    create_weights = GPUCompressedTensorsLinearMethod.create_weights

    def __init__(self, quantization_config: "CompressedTensorsConfig"):
        self.quantization_config = quantization_config

    def apply(self,
              layer: torch.nn.Module,
              x: torch.Tensor,
              bias: Optional[torch.Tensor] = None,
              hcom: Optional[str] = None,
              use_mm_all_reduce_op: bool = False,
              inner_gather: bool = False,
              enable_weight_transpsoe: bool = False,
              ):
        """
        Use the output of create_weights and the CompressedTensorsScheme
        associated with the layer to apply the forward pass with the
        layer input.  See LinearMethodBase for param details

        """

        scheme = layer.scheme
        if scheme is None:
            raise ValueError("A scheme must be defined for each layer")
        return scheme.apply_weights(layer, x, bias=bias, hcom=hcom, use_mm_all_reduce_op=use_mm_all_reduce_op,
                                    inner_gather=inner_gather, enable_weight_transpsoe=enable_weight_transpsoe)
