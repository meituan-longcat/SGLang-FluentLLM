"""Packed row offload GE adapters; payload bytes are never interpreted here."""
import torch
import torchair
from torch.library import Library, impl
from torchair._ge_concrete_graph import ge_apis as ge
from torchair._ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair.ge import attr
from torchair.ge._ge_graph import DataType


@register_fx_node_ge_converter(torch.ops.fluent_sfa1.npu_gather_by_cache_indices_packed.default)
def gather(cache_packed, from_cache_indices, topk, full_packed, full_block_table,
           selection_block_table, sources, kv_lengths, query_lengths, *,
           selection_topk_block_size=1, selection_kv_block_num=0,
           selection_kv_block_size=128, enable_slots_contiguous=2, meta_outputs=None):
    return torchair.ge.custom_op(
        "GatherByCacheIndicesPacked",
        inputs=dict(cache_packed=cache_packed, from_cache_indices=from_cache_indices,
                    topk=topk, full_packed=full_packed, full_block_table=full_block_table,
                    selection_block_table=selection_block_table, sources=sources,
                    kv_lengths=kv_lengths, query_lengths=query_lengths),
        attrs=dict(selection_topk_block_size=attr.Int(selection_topk_block_size),
                   selection_kv_block_num=attr.Int(selection_kv_block_num),
                   selection_kv_block_size=attr.Int(selection_kv_block_size),
                   enable_slots_contiguous=attr.Int(enable_slots_contiguous)),
        outputs=["selection_packed", "current_write_slots"],
    )


@register_fx_node_ge_converter(torch.ops.fluent_sfa1.npu_update_cache_packed.default)
def update(topk, selection_packed, cache_packed, status, age,
           selection_block_table, write_slot, *, step, meta_outputs=None):
    return torchair.ge.custom_op(
        "UpdateCachePacked",
        inputs=dict(topk=topk, selection_packed=selection_packed, cache_packed=cache_packed,
                    status=status, age=age, selection_block_table=selection_block_table,
                    write_slot=write_slot, step=step),
        attrs={}, outputs=["new_cache_packed", "new_status", "new_age"],
    )


def _patch(current_packed, current_write_slots, selection_packed, selection_block_table):
    output, _ = torchair.ge.custom_op(
        "UpdateCurKvToSelectionPacked",
        inputs=dict(current_packed=current_packed, current_write_slots=current_write_slots,
                    selection_packed=selection_packed, selection_block_table=selection_block_table),
        attrs={}, outputs=["selection_packed", "kernel_done"],
    )
    # The kernel does not write the completion token: order it after the real write.
    done = ge.Identity(ge.Const([0], dtype=DataType.DT_INT32), dependencies=[output])
    return done, output


@register_fx_node_ge_converter(torch.ops.fluent_sfa1.npu_update_cur_kv_to_selection_packed.default)
def patch(current_packed, current_write_slots, selection_packed, selection_block_table, *, meta_outputs=None):
    return _patch(current_packed, current_write_slots, selection_packed, selection_block_table)[0]


@register_fx_node_ge_converter(torch.ops.fluent_sfa1.npu_update_cur_kv_to_selection_packed_functional.default)
def patch_functional(current_packed, current_write_slots, selection_packed, selection_block_table, *, meta_outputs=None):
    # Functional callers retain their original selection. AOT may reuse this
    # buffer when lowering the mutable entry, but direct functional calls must
    # preserve the input as the eager and Meta implementations do.
    selection_copy = ge.TensorMove(selection_packed)
    return _patch(current_packed, current_write_slots, selection_copy, selection_block_table)


_library = Library("fluent_sfa1", "FRAGMENT")


@impl(_library, "npu_update_cur_kv_to_selection_packed", "Functionalize")
def functionalize_patch(current_packed, current_write_slots, selection_packed, selection_block_table):
    done, output = torch.ops.fluent_sfa1.npu_update_cur_kv_to_selection_packed_functional(
        current_packed, current_write_slots, selection_packed, selection_block_table)
    selection_packed.copy_(output)
    return done
