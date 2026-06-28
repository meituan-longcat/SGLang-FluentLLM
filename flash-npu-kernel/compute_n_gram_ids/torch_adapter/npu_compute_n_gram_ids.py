
import torch
from torch import Generator, contiguous_format, inf, strided, SymInt
from torch.types import Device, Number, _bool, _complex, _device, _dtype, _float, _int, _layout, _qscheme, _size
from torchair._ge_concrete_graph import ge_apis as ge
from torchair._ge_concrete_graph.fx2ge_converter import declare_supported, register_fx_node_ge_converter
from torchair._ge_concrete_graph.supported_declaration import _TypedTensor, F32, F16, BF16, F64, I32, I16, I64, \
    I8, U8, BOOL, Support
from torchair._ge_concrete_graph.utils import dtype_promote
from torchair.ge._ge_graph import Tensor, TensorSpec
from typing import (
    Any, Callable, ContextManager, Iterator, List, Literal, NamedTuple, Optional, Sequence, Tuple, TypeVar,
    Union, overload, Optional
)


# ... 省略前面不变的代码 ...

@register_fx_node_ge_converter(torch.ops.npu.compute_n_gram_ids.default)
def convert_npu_compute_n_gram_ids(
        oe_weights: Tensor,
        oe_mods: Tensor,
        exclusive_oe_embeder_size_sums: Tensor,
        tokens: Tensor,
        exclusive_req_len_sums: Tensor,
        oe_token_table: Tensor,
        row_indices: Tensor,
        column_starts: Tensor,
        batch_size: int,
        oe_n: int,
        oe_k: int,
        max_context_len: int
):
    # 调用自定义算子 ComputeNGramIds
    return ge.ComputeNGramIds(
        oe_weights,
        oe_mods,
        exclusive_oe_embeder_size_sums,
        tokens,
        exclusive_req_len_sums,
        oe_token_table,
        row_indices,
        column_starts,
        batch_size=batch_size,
        oe_n=oe_n,
        oe_k=oe_k,
        max_context_len=max_context_len
    )