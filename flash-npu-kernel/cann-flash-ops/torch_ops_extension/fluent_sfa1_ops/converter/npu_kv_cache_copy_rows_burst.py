"""The mutable TensorList burst is an eager posthook, never a GE node."""
import torch
from torchair._ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter


@register_fx_node_ge_converter(torch.ops.fluent_sfa1.npu_kv_cache_copy_rows_burst_.default)
def reject_ge(caches, src_indices, dst_indices, *, meta_outputs=None):
    raise RuntimeError('npu_kv_cache_copy_rows_burst_ is eager-only; call it after graph replay')
