"""Private SFA1 ABI, shipped alongside the legacy flash_ops ABI in one wheel.

The private ABI preserves flash_ops' normal bindings and does not initialize an
NPU or explicitly preload an OpProto library.
"""
import importlib

import torch
import torch_npu  # noqa: F401
import flash_ops  # noqa: F401 - retain master's custom wrappers and namespace

from . import _C
from .runtime import validate_vendor

ABI_VERSION = 1
PACKED_KV_ABI_VERSION = _C.PACKED_KV_ABI_VERSION
LAYER_COPY_BURST_ABI_VERSION = getattr(_C, 'LAYER_COPY_BURST_ABI_VERSION', 0)
if _C.SFA1_ABI_VERSION != ABI_VERSION:
    raise RuntimeError("The SFA1 Python package and native extension do not match")
PROVIDER = validate_vendor(_C)

_CONTRACTS = {
    "npu_mla_prolog_v3": 7,
    "npu_mla_prolog_v3_functional": 9,
    "npu_gen_from_cache_indices": 4,
    "npu_gather_by_cache_indices_packed": 2,
    "npu_update_cache_packed": 3,
    "npu_update_cur_kv_to_selection_packed": 1,
    "npu_update_cur_kv_to_selection_packed_functional": 2,
    "npu_gather_by_cache_indices": 3,
    "npu_update_cache": 4,
    "npu_update_cur_kv_to_selection": 1,
    "npu_update_cur_kv_to_selection_functional": 3,
}
for _name, _returns in _CONTRACTS.items():
    _op = getattr(torch.ops.fluent_sfa1, _name).default
    if len(_op._schema.returns) != _returns:
        raise RuntimeError(f"Unexpected SFA1 schema: {_op._schema}")
    for _key in ("PrivateUse1", "Meta"):
        if not torch._C._dispatch_has_kernel_for_dispatch_key(
            "fluent_sfa1::" + _name, _key
        ):
            raise RuntimeError(f"Missing SFA1 {_name} {_key} implementation")

for _module in (
    "npu_packed_kv_offload", "npu_mla_prolog_v3",
    "npu_gen_from_cache_indices", "npu_gather_by_cache_indices",
    "npu_update_cache", "npu_update_cur_kv_to_selection",
):
    importlib.import_module(".converter." + _module, __name__)

ops = torch.ops.fluent_sfa1

if LAYER_COPY_BURST_ABI_VERSION:
    from .layer_copy_burst import register_cpu as _register_layer_copy_burst_cpu
    _register_layer_copy_burst_cpu()
    importlib.import_module('.converter.npu_kv_cache_copy_rows_burst', __name__)


def validate_layer_copy_burst():
    """Opt-in paired-runtime proof; no allocation, NPU call or old-op fallback."""
    from .layer_copy_burst import validate_pair
    return validate_pair(_C, PROVIDER)
