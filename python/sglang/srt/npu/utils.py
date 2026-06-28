import torch
from contextlib import nullcontext
from sglang.srt.utils import is_npu

__is_npu__ = is_npu()
if __is_npu__:
    import os
    import torchair as tng
    os.environ['USE_EPS_TOPK_SIGMOID'] = 'false'


def init_npu():
    if __is_npu__:
        print("npu initialized")
        """
        import vllm.model_executor.layers.quantization.compressed_tensors as vllm_compressed_tensors
        import vllm.model_executor.layers.quantization.compressed_tensors.schemes as vllm_schemes
        import sglang.srt.layers.quantization.npu_compressed_tensors as npu_compressed_tensors
        import sglang.srt.layers.quantization.npu_compressed_tensors.schemes as npu_schemes

        vllm_schemes.CompressedTensorsW8A8Int8 = npu_schemes.CompressedTensorsW8A8Int8
        #vllm_schemes.CompressedTensorsW8A8Int8.create_weights = npu_schemes.create_weights
        #vllm_schemes.CompressedTensorsW8A8Int8.process_weights_after_loading = npu_schemes.process_weights_after_loading
        #vllm_schemes.CompressedTensorsW8A8Int8.apply_weights = npu_schemes.apply_weights

        vllm_compressed_tensors.compressed_tensors.CompressedTensorsConfig.get_quant_method = npu_compressed_tensors.compressed_tensors.get_quant_method
        vllm_compressed_tensors.compressed_tensors.CompressedTensorsLinearMethod = npu_compressed_tensors.NpuCompressedTensorsLinearMethod
        vllm_compressed_tensors.compressed_tensors.CompressedTensorsMoEMethod = npu_compressed_tensors.NpuCompressedTensorsW8A8Int8MoEMethod
        """
        import vllm.model_executor.layers.quantization.compressed_tensors as vllm_compressed_tensors
        import sglang.srt.layers.quantization.npu_compressed_tensors as npu_compressed_tensors
        vllm_compressed_tensors.compressed_tensors.CompressedTensorsMoEMethod = npu_compressed_tensors.NpuCompressedTensorsW8A8Int8MoEMethod


def npu_limit_core(aicore, aiv, flag=True):
    if flag:
        return tng.scope.limit_core_num(aicore, aiv)
    else:
        return nullcontext()

def npu_super_kernel(*args, flag=True, **kwargs):
    from sglang.srt.env import global_server_args_dict
    if global_server_args_dict['npu_enable_super_kernel'] and flag:
        return tng.scope.super_kernel(*args, **kwargs)
    else:
        return nullcontext()

def npu_stream_switch(flag, scope):
    if flag:
        return tng.scope.npu_stream_switch(scope)
    else:
        return nullcontext()
