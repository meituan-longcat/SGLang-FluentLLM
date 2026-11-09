"""Operator capability check for packed INT8 hot-cache offload.

This selects cache-management operators only. Packed Prolog and SFAD retain
their existing native ABI. Resolve before tracing; never fall back to a cold
Host gather when an installation lacks the paired packed kernels.
"""

import importlib

_ops = None
PACKED_KV_ABI_VERSION = 1
_SCHEMAS = {
    "npu_gen_from_cache_indices": (
        "(Tensor selection_topk_indices, Tensor cache_status, Tensor full_kv_actual_seq, Tensor full_q_actual_seq, *, "
        "int selection_topk_block_size=1, int enable_topk_indices_opt=0, Tensor? full_kv_block_table=None, "
        "int full_kv_block_size=128) -> (Tensor, Tensor, Tensor, Tensor)"
    ),
    "npu_gather_by_cache_indices_packed": (
        "(Tensor cache_packed, Tensor from_cache_indices, Tensor topk, Tensor full_packed, Tensor full_block_table, "
        "Tensor selection_block_table, Tensor sources, Tensor kv_lengths, Tensor query_lengths, *, "
        "int selection_topk_block_size=1, int selection_kv_block_num=0, int selection_kv_block_size=128, "
        "int enable_slots_contiguous=2) -> (Tensor, Tensor)"
    ),
    "npu_update_cache_packed": (
        "(Tensor topk, Tensor selection_packed, Tensor cache_packed, Tensor status, Tensor age, "
        "Tensor selection_block_table, Tensor write_slot, *, Tensor step) -> (Tensor, Tensor, Tensor)"
    ),
    "npu_update_cur_kv_to_selection_packed": (
        "(Tensor current_packed, Tensor current_write_slots, Tensor(a!) selection_packed, Tensor selection_block_table) -> Tensor"
    ),
    "npu_update_cur_kv_to_selection_packed_functional": (
        "(Tensor current_packed, Tensor current_write_slots, Tensor selection_packed, Tensor selection_block_table) -> (Tensor, Tensor)"
    ),
}


def get_packed_offload_ops():
    """Require the documented packed ABI and its native/Meta implementations."""
    global _ops
    if _ops is not None:
        return _ops
    try:
        module = importlib.import_module("fluent_sfa1_ops")
    except ModuleNotFoundError as error:
        if error.name != "fluent_sfa1_ops":
            raise
        raise RuntimeError("packed INT8 hot-cache offload requires the paired operator package; see docs/npu_int8_offload.md") from error
    if getattr(module, "PACKED_KV_ABI_VERSION", None) != PACKED_KV_ABI_VERSION:
        raise RuntimeError(
            "packed INT8 offload requires fluent_sfa1_ops.PACKED_KV_ABI_VERSION=1; "
            "the BF16-only wheel cannot serve this branch. See docs/npu_int8_offload.md"
        )

    import torch

    for name, signature in _SCHEMAS.items():
        try:
            operator = getattr(module.ops, name).default
        except AttributeError as error:
            raise RuntimeError(f"Missing packed offload operator fluent_sfa1::{name}") from error
        schema = operator._schema
        expected_schema = torch._C.parse_schema("fluent_sfa1::" + name + signature)
        if (
            schema.name != expected_schema.name
            or len(schema.arguments) != len(expected_schema.arguments)
            or any(
                (arg.name, arg.type, arg.kwarg_only) != (expected.name, expected.type, expected.kwarg_only)
                for arg, expected in zip(schema.arguments, expected_schema.arguments)
            )
            or len(schema.returns) != len(expected_schema.returns)
            or any(result.type != expected.type or result.alias_info is not None for result, expected in zip(schema.returns, expected_schema.returns))
        ):
            raise RuntimeError(f"Incompatible packed offload schema: {schema}")
        for key in ("PrivateUse1", "Meta"):
            if not torch._C._dispatch_has_kernel_for_dispatch_key(schema.name, key):
                raise RuntimeError(f"Missing {schema.name} {key} implementation")
        mutations = {arg.name for arg in schema.arguments if arg.alias_info is not None and arg.alias_info.is_write}
        expected = {"selection_packed"} if name == "npu_update_cur_kv_to_selection_packed" else set()
        if mutations != expected:
            raise RuntimeError(f"Incompatible packed offload mutation contract: {schema}")
    patch_name = "fluent_sfa1::npu_update_cur_kv_to_selection_packed"
    if not torch._C._dispatch_has_kernel_for_dispatch_key(patch_name, "Functionalize"):
        raise RuntimeError(f"Missing {patch_name} Functionalize implementation")
    _ops = module.ops
    return _ops
