# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import ctypes
import os
import importlib
import pkgutil
import warnings

__all__ = list(module for _, module, _ in pkgutil.iter_modules([os.path.dirname(__file__)]))

# The generated custom op-api library references a transformer helper that is
# exported by CANN's transformer op-api library but is not recorded in its
# DT_NEEDED entries. Make that symbol globally visible before loading the
# extension; newer CANN packages that already link it correctly are unaffected.
try:
    ctypes.CDLL("libopapi_transformer.so", mode=ctypes.RTLD_GLOBAL)
except OSError:
    pass

# 导入so 和 python
from . import custom_ops_lib

# Consumers must install this extension together with its matching custom OPP.
CAUSAL_CONV1D_STATE_SLOT_STRIDE = True

_CONVERTER_MODULES = (
    "flat_quant",
    "mlp_dflash_infer_attn",
    "mlp_compressed_lightning_indexer",
    "mlp_lightning_indexer",
    "mlp_sparse_lightning_indexer",
    "mlp_split_swiglu",
    "npu_moe_distribute_dispatch_v2",
    "npu_moe_distribute_combine_v2",
    "npu_causal_conv1d",
    "npu_compute_n_gram_ids",
    "npu_gather_element",
    "npu_select_local_topk_indices",
    "npu_router_input_slice_cast",
    "npu_kvp_attention_merge",
    "npu_packed_kv_prefill_prepare",
    "npu_moe_expert_id_map",
    "npu_moe_gating_top_k_expert_map",
    "npu_gather_selection_kv_cache",
    "npu_kv_cache_copy_rows",
    "npu_kv_quant_sparse_flash_attention_lse",
    "npu_lightning_indexer",
    "npu_mla_prolog_v3",
    "npu_sparse_flash_attention",
    "npu_sparse_flash_attention_decode",
    "npu_sparse_mla_attention_prefill",
)

for _module_name in _CONVERTER_MODULES:
    try:
        importlib.import_module(f".converter.{_module_name}", __name__)
    except AttributeError as exc:
        if os.getenv("FLASH_OPS_WARN_SKIPPED_CONVERTERS") == "1":
            warnings.warn(f"Skip flash_ops converter {_module_name}: {exc}")

"""
import custom ops as torch_npu ops to support the following usage:
'torch.ops.custom.npu_selected_flash_attention()'
'torch_npu.npu_selected_flash_attention()'
"""

# Ensure that the torch and torch_npu has been successfully imported to avoid subsequent mount operation failures
import torch
import torch_npu

# get torch.ops.custom module
custom_ops_module = getattr(torch.ops, 'custom', None)

if custom_ops_module is not None:
    for op_name in dir(custom_ops_module):
        if op_name.startswith('_'):
            # skip built-in method, such as __name__, __doc__
            continue

        # get custom ops and set to torch_npu
        custom_op_func = getattr(custom_ops_module, op_name)
        setattr(torch_npu, op_name, custom_op_func)

else:
    warn_msg = "torch.ops.custom module is not found, mount custom ops to torch_npu failed." \
               "Calling by torch_npu.xxx for custom ops is unsupported, please use torch.ops.custom.xxx."
    warnings.warn(warn_msg)
    warnings.filterwarnings("ignore", message=warn_msg)
