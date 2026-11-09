from contextlib import nullcontext

import torch

from sglang.srt.utils import get_colorful_logger, is_npu

__is_npu__ = is_npu()
if __is_npu__:
    import os

    import torchair as tng

    os.environ["USE_EPS_TOPK_SIGMOID"] = "false"

logger = get_colorful_logger(__name__)

# Full core budget of the device (910B: 24 AIC / 48 AIV).
_FULL_CORE_NUM = (24, 48)
_core_num_map = None
KV_OFFLOAD_STREAM = "kv_offload_stream"


def _make_core_num_map(moe_aic):
    """The existing graph core budgets, shared by all named scope callers."""
    aic, aiv = _FULL_CORE_NUM
    ratio = aiv // aic
    moe_aiv = ratio * (aic - moe_aic)
    main_aic = aic - moe_aic
    main_aiv = aiv - moe_aiv
    return {
        "full": (aic, aiv),
        "indexer_prolog": (moe_aic, moe_aiv),
        "mla_prolog": (main_aic, main_aiv),
        "indexer_topk": (16, 32),
        "indexer_kvp": (aic - 16, aiv - 32),
        "main": (main_aic, main_aiv),
        "moe_side": (moe_aic, moe_aiv),
        "kvp_lse_0": (aic // 2, aiv // 2),
        "kvp_o_0": (aic // 2, aiv // 2),
        "kvp_lse_1": (main_aic // 2, main_aiv // 2),
        "kvp_o_1": (main_aic // 2, main_aiv // 2),
        "offload_gen": (1, aiv - main_aiv),
        "offload_gather": (1, 4),
        "offload_update": (1, 8),
    }


def init_npu():
    if __is_npu__:
        print("npu initialized")
        import vllm.model_executor.layers.quantization.compressed_tensors as vllm_compressed_tensors

        import sglang.srt.layers.quantization.npu_compressed_tensors as npu_compressed_tensors

        vllm_compressed_tensors.compressed_tensors.CompressedTensorsMoEMethod = npu_compressed_tensors.NpuCompressedTensorsW8A8Int8MoEMethod

        global _core_num_map
        if _core_num_map is None:
            from sglang.srt.env import ENV

            _core_num_map = _make_core_num_map(ENV.npu_moe_core_num)
            logger.info(f"core_num_map={_core_num_map}")


def _get_core_num_map():
    """Central registry of all core-limit configurations, so that the core
    numbers are managed in one place instead of being computed at every
    npu_limit_core call site."""
    return _core_num_map


def npu_limit_core(name, flag=True):
    """Limit the core num of the enclosed ops. name must be a key of the
    central core-num map in _get_core_num_map()."""
    if not flag:
        return nullcontext()
    return tng.scope.limit_core_num(*_get_core_num_map()[name])


def npu_super_kernel(*args, flag=True, **kwargs):
    from sglang.srt.env import global_server_args_dict

    if global_server_args_dict["npu_enable_super_kernel"] and flag:
        return tng.scope.super_kernel(*args, **kwargs)
    else:
        return nullcontext()


def npu_stream_switch(flag, scope):
    if flag:
        return tng.scope.npu_stream_switch(scope)
    else:
        return nullcontext()


def npu_wait_tensor(self_tensor: torch.Tensor, dependency_tensor: torch.Tensor, flag=True):
    if flag:
        return tng.scope.npu_wait_tensor(self_tensor, dependency_tensor)
    return self_tensor
