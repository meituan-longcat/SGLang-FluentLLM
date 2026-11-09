import torch
import torchair
from torch.library import Library, impl
from torchair._ge_concrete_graph import ge_apis as ge
from torchair._ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair.ge._ge_graph import Tensor, TensorSpec, dont_prune_me
from torchair.patterns._recover_view_inplace_pattern import (
    InplaceOpInfo,
    _INPLACE_OPS_MAP,
)


def _kv_cache_copy_rows(cache: Tensor, src_indices: Tensor,
                        dst_indices: Tensor):
    return torchair.ge.custom_op(
        "KvCacheCopyRows",
        inputs={
            "cache": cache,
            "src_indices": src_indices,
            "dst_indices": dst_indices,
        },
        outputs=["cache"],
    )


_fragment = Library("custom", "FRAGMENT")


@impl(_fragment, "npu_kv_cache_copy_rows_", "Functionalize")
def functionalize_npu_kv_cache_copy_rows_(cache, src_indices, dst_indices):
    result = torch.ops.custom.npu_kv_cache_copy_rows(
        cache, src_indices, dst_indices)
    cache.copy_(result)
    return cache


@register_fx_node_ge_converter(torch.ops.custom.npu_kv_cache_copy_rows.default)
def convert_npu_kv_cache_copy_rows(
    cache: Tensor,
    src_indices: Tensor,
    dst_indices: Tensor,
    *,
    meta_outputs: TensorSpec = None,
):
    return _kv_cache_copy_rows(
        ge.TensorMove(cache), src_indices, dst_indices
    )


@register_fx_node_ge_converter(torch.ops.custom.npu_kv_cache_copy_rows_.default)
def convert_npu_kv_cache_copy_rows_(
    cache: Tensor,
    src_indices: Tensor,
    dst_indices: Tensor,
    *,
    meta_outputs: TensorSpec = None,
):
    result = _kv_cache_copy_rows(cache, src_indices, dst_indices)
    dont_prune_me(result)
    return result


_INPLACE_OPS_MAP[torch.ops.custom.npu_kv_cache_copy_rows.default] = InplaceOpInfo(
    inplace_op=torch.ops.custom.npu_kv_cache_copy_rows_.default,
    output_to_ref_input_idx_mapping={0: 0},
)
