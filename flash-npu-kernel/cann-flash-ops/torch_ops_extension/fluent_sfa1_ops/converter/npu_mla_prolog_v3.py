# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from typing import (
    Any, Callable, ContextManager, Iterator, List, Literal, NamedTuple, Optional, Sequence, Tuple, TypeVar,
    Union, overload,
)
import torch
import torch_npu
import torchair
from torch import Generator, contiguous_format, inf, strided, SymInt
from torch.types import Device, Number, _bool, _complex, _device, _dtype, _float, _int, _layout, _qscheme, _size
from torchair._ge_concrete_graph import ge_apis as ge
from torchair._ge_concrete_graph.fx2ge_converter import declare_supported, register_fx_node_ge_converter
from torchair.ge._ge_graph import Tensor, TensorSpec, DataType, TensorType
from torchair._ge_concrete_graph.supported_declaration import _TypedTensor, F32, F16, F64, I32, I16, I64, I8, U8, \
    BOOL, Support
from torchair._ge_concrete_graph.utils import dtype_promote
from torchair.ge import attr
from torch.library import Library, impl
from typing import Any, List, Tuple, Union, Callable, Optional
from torchair.ge._ge_graph import get_default_ge_graph, next_unique_name, auto_convert_to_tensor, compat_as_bytes, \
    get_invalid_desc, dont_prune_me


# This api is auto-generated from IR MlaPrologV3
@auto_convert_to_tensor(
        [False, False, False, False, False, False, False, False, False, False,
         False, False, False, False, False, False, False, False, False, False, False],
        [False, False, False, False, False, False, False, False, False, False,
         False, True, True, True, True, True, True, True, True, True, True])
def MlaPrologV3(token_x: Tensor, weight_dq: Tensor, weight_uq_qr: Tensor, weight_uk: Tensor,
    weight_dkv_kr: Tensor, rmsnorm_gamma_cq: Tensor, rmsnorm_gamma_ckv: Tensor, rope_sin: Tensor,
    rope_cos: Tensor, kv_cache: Tensor, kr_cache: Tensor, cache_index: Optional[Tensor],
    dequant_scale_x: Optional[Tensor], dequant_scale_w_dq: Optional[Tensor], dequant_scale_w_uq_qr: Optional[Tensor],
    dequant_scale_w_dkv_kr: Optional[Tensor], quant_scale_ckv: Optional[Tensor], quant_scale_ckr: Optional[Tensor],
    smooth_scales_cq: Optional[Tensor], actual_seq_len: Optional[Tensor], k_nope_clip_alpha: Optional[Tensor], *,
    rmsnorm_epsilon_cq: float = 0.000010, rmsnorm_epsilon_ckv: float = 0.000010, cache_mode: str = "PA_BSND",
    query_norm_flag: bool = False, weight_quant_mode: int = 0, kv_cache_quant_mode: int = 0, query_quant_mode: int = 0,
    ckvkr_repo_mode: int = 0, quant_scale_repo_mode: int = 0, tile_size: int = 128, qc_qr_scale: float = 1.0,
    kc_scale: float = 1.0, output_cur_k_nope_rope: bool = False, dependencies=[], node_name=None):
    """REG_OP(MlaPrologV3)\n
.INPUT(token_x, TensorType({DT_INT8, DT_BF16}))\n
.INPUT(weight_dq, TensorType({DT_INT8, DT_BF16}))\n
.INPUT(weight_uq_qr, TensorType({DT_INT8, DT_BF16}))\n
.INPUT(weight_uk, TensorType({DT_FLOAT16, DT_BF16}))\n
.INPUT(weight_dkv_kr, TensorType({DT_INT8, DT_BF16}))\n
.INPUT(rmsnorm_gamma_cq, TensorType({DT_FLOAT16, DT_BF16}))\n
.INPUT(rmsnorm_gamma_ckv, TensorType({DT_FLOAT16, DT_BF16}))\n
.INPUT(rope_sin, TensorType({DT_FLOAT16, DT_BF16}))\n
.INPUT(rope_cos, TensorType({DT_FLOAT16, DT_BF16}))\n
.INPUT(kv_cache, TensorType({DT_FLOAT16, DT_BF16, DT_INT8}))\n
.INPUT(kr_cache, TensorType({DT_FLOAT16, DT_BF16, DT_INT8}))\n
.OPTIONAL_INPUT(cache_index, TensorType({DT_INT64}))\n
.OPTIONAL_INPUT(dequant_scale_x, TensorType({DT_FLOAT}))\n
.OPTIONAL_INPUT(dequant_scale_w_dq, TensorType({DT_FLOAT}))\n
.OPTIONAL_INPUT(dequant_scale_w_uq_qr, TensorType({DT_FLOAT}))\n
.OPTIONAL_INPUT(dequant_scale_w_dkv_kr, TensorType({DT_FLOAT}))\n
.OPTIONAL_INPUT(quant_scale_ckv, TensorType({DT_FLOAT}))\n
.OPTIONAL_INPUT(quant_scale_ckr, TensorType({DT_FLOAT}))\n
.OPTIONAL_INPUT(smooth_scales_cq, TensorType({DT_FLOAT}))\n
.OPTIONAL_INPUT(actual_seq_len, TensorType({DT_INT32}))\n
.OPTIONAL_INPUT(k_nope_clip_alpha, TensorType({DT_FLOAT}))\n
.OUTPUT(query, TensorType({DT_FLOAT16, DT_BF16, DT_INT8}))\n
.OUTPUT(query_rope, TensorType({DT_FLOAT16, DT_BF16, DT_INT8}))\n
.OUTPUT(kv_cache, TensorType({DT_FLOAT16, DT_BF16, DT_INT8}))\n
.OUTPUT(kr_cache, TensorType({DT_FLOAT16, DT_BF16, DT_INT8}))\n
.OUTPUT(dequant_scale_q_nope, TensorType({DT_FLOAT}))\n
.OUTPUT(query_norm, TensorType({DT_BF16, DT_INT8}))\n
.OUTPUT(dequant_scale_q_norm, TensorType({DT_FLOAT}))\n
.OUTPUT(cur_k_nope, TensorType({DT_FLOAT16, DT_BF16, DT_INT8}))\n
.OUTPUT(cur_k_rope, TensorType({DT_FLOAT16, DT_BF16, DT_INT8}))\n
.ATTR(rmsnorm_epsilon_cq, Float, 1e-05)\n
.ATTR(rmsnorm_epsilon_ckv, Float, 1e-05)\n
.ATTR(cache_mode, String, "PA_BSND")\n
.ATTR(query_norm_flag, Bool, 0)\n
.ATTR(weight_quant_mode, Int, 0)\n
.ATTR(query_quant_mode, Int, 0)\n
.ATTR(ckvkr_repo_mode, Int, 0)\n
.ATTR(quant_scale_repo_mode, Int, 0)\n
.ATTR(tile_size, Int, 0)\n
.ATTR(qc_qr_scale, Float, 1.0)\n
.ATTR(kc_scale, Float, 1.0)\n
.ATTR(output_cur_k_nope_rope, Bool, 0)\n
"""

    op = get_default_ge_graph().op.add()
    op.type = "FluentMlaPrologV3"
    op.name = next_unique_name(node_name, "FluentMlaPrologV3")

    # process dependices
    for dependency in dependencies:
        op.input.append(dependency.controller)

    # process inputs
    op.input.append(token_x.tensor)
    op.input_desc.add().CopyFrom(token_x.desc)
    op.input_desc[-1].name = "token_x"

    op.input.append(weight_dq.tensor)
    op.input_desc.add().CopyFrom(weight_dq.desc)
    op.input_desc[-1].name = "weight_dq"

    op.input.append(weight_uq_qr.tensor)
    op.input_desc.add().CopyFrom(weight_uq_qr.desc)
    op.input_desc[-1].name = "weight_uq_qr"

    op.input.append(weight_uk.tensor)
    op.input_desc.add().CopyFrom(weight_uk.desc)
    op.input_desc[-1].name = "weight_uk"

    op.input.append(weight_dkv_kr.tensor)
    op.input_desc.add().CopyFrom(weight_dkv_kr.desc)
    op.input_desc[-1].name = "weight_dkv_kr"

    op.input.append(rmsnorm_gamma_cq.tensor)
    op.input_desc.add().CopyFrom(rmsnorm_gamma_cq.desc)
    op.input_desc[-1].name = "rmsnorm_gamma_cq"

    op.input.append(rmsnorm_gamma_ckv.tensor)
    op.input_desc.add().CopyFrom(rmsnorm_gamma_ckv.desc)
    op.input_desc[-1].name = "rmsnorm_gamma_ckv"

    op.input.append(rope_sin.tensor)
    op.input_desc.add().CopyFrom(rope_sin.desc)
    op.input_desc[-1].name = "rope_sin"

    op.input.append(rope_cos.tensor)
    op.input_desc.add().CopyFrom(rope_cos.desc)
    op.input_desc[-1].name = "rope_cos"

    op.input.append(kv_cache.tensor)
    op.input_desc.add().CopyFrom(kv_cache.desc)
    op.input_desc[-1].name = "kv_cache"

    op.input.append(kr_cache.tensor)
    op.input_desc.add().CopyFrom(kr_cache.desc)
    op.input_desc[-1].name = "kr_cache"

    if cache_index is not None:
        op.input.append(cache_index.tensor)
        op.input_desc.add().CopyFrom(cache_index.desc)
        op.input_desc[-1].name = "cache_index"
    else:
        op.input.append('')
        op.input_desc.add().CopyFrom(get_invalid_desc())
        op.input_desc[-1].name = "cache_index"

    if dequant_scale_x is not None:
        op.input.append(dequant_scale_x.tensor)
        op.input_desc.add().CopyFrom(dequant_scale_x.desc)
        op.input_desc[-1].name = "dequant_scale_x"
    else:
        op.input.append('')
        op.input_desc.add().CopyFrom(get_invalid_desc())
        op.input_desc[-1].name = "dequant_scale_x"

    if dequant_scale_w_dq is not None:
        op.input.append(dequant_scale_w_dq.tensor)
        op.input_desc.add().CopyFrom(dequant_scale_w_dq.desc)
        op.input_desc[-1].name = "dequant_scale_w_dq"
    else:
        op.input.append('')
        op.input_desc.add().CopyFrom(get_invalid_desc())
        op.input_desc[-1].name = "dequant_scale_w_dq"

    if dequant_scale_w_uq_qr is not None:
        op.input.append(dequant_scale_w_uq_qr.tensor)
        op.input_desc.add().CopyFrom(dequant_scale_w_uq_qr.desc)
        op.input_desc[-1].name = "dequant_scale_w_uq_qr"
    else:
        op.input.append('')
        op.input_desc.add().CopyFrom(get_invalid_desc())
        op.input_desc[-1].name = "dequant_scale_w_uq_qr"

    if dequant_scale_w_dkv_kr is not None:
        op.input.append(dequant_scale_w_dkv_kr.tensor)
        op.input_desc.add().CopyFrom(dequant_scale_w_dkv_kr.desc)
        op.input_desc[-1].name = "dequant_scale_w_dkv_kr"
    else:
        op.input.append('')
        op.input_desc.add().CopyFrom(get_invalid_desc())
        op.input_desc[-1].name = "dequant_scale_w_dkv_kr"

    if quant_scale_ckv is not None:
        op.input.append(quant_scale_ckv.tensor)
        op.input_desc.add().CopyFrom(quant_scale_ckv.desc)
        op.input_desc[-1].name = "quant_scale_ckv"
    else:
        op.input.append('')
        op.input_desc.add().CopyFrom(get_invalid_desc())
        op.input_desc[-1].name = "quant_scale_ckv"

    if quant_scale_ckr is not None:
        op.input.append(quant_scale_ckr.tensor)
        op.input_desc.add().CopyFrom(quant_scale_ckr.desc)
        op.input_desc[-1].name = "quant_scale_ckr"
    else:
        op.input.append('')
        op.input_desc.add().CopyFrom(get_invalid_desc())
        op.input_desc[-1].name = "quant_scale_ckr"

    if smooth_scales_cq is not None:
        op.input.append(smooth_scales_cq.tensor)
        op.input_desc.add().CopyFrom(smooth_scales_cq.desc)
        op.input_desc[-1].name = "smooth_scales_cq"
    else:
        op.input.append('')
        op.input_desc.add().CopyFrom(get_invalid_desc())
        op.input_desc[-1].name = "smooth_scales_cq"

    if actual_seq_len is not None:
        op.input.append(actual_seq_len.tensor)
        op.input_desc.add().CopyFrom(actual_seq_len.desc)
        op.input_desc[-1].name = "actual_seq_len"
    else:
        op.input.append('')
        op.input_desc.add().CopyFrom(get_invalid_desc())
        op.input_desc[-1].name = "actual_seq_len"

    if k_nope_clip_alpha is not None:
        op.input.append(k_nope_clip_alpha.tensor)
        op.input_desc.add().CopyFrom(k_nope_clip_alpha.desc)
        op.input_desc[-1].name = "k_nope_clip_alpha"
    else:
        op.input.append('')
        op.input_desc.add().CopyFrom(get_invalid_desc())
        op.input_desc[-1].name = "k_nope_clip_alpha"

    # process attrs
    op.attr["rmsnorm_epsilon_cq"].f = rmsnorm_epsilon_cq
    op.attr["rmsnorm_epsilon_ckv"].f = rmsnorm_epsilon_ckv
    op.attr["cache_mode"].s = compat_as_bytes(cache_mode)
    op.attr["query_norm_flag"].b = query_norm_flag
    op.attr["weight_quant_mode"].i = weight_quant_mode
    op.attr["kv_cache_quant_mode"].i = kv_cache_quant_mode
    op.attr["query_quant_mode"].i = query_quant_mode
    op.attr["ckvkr_repo_mode"].i = ckvkr_repo_mode
    op.attr["quant_scale_repo_mode"].i = quant_scale_repo_mode
    op.attr["tile_size"].i = tile_size
    op.attr["qc_qr_scale"].f = qc_qr_scale
    op.attr["kc_scale"].f = kc_scale
    op.attr["output_cur_k_nope_rope"].b = output_cur_k_nope_rope

    # process outputs
    output_index = 0
    op.output_desc.add().name = "query"
    query = Tensor(op, output_index)
    output_index += 1
    op.output_desc.add().name = "query_rope"
    query_rope = Tensor(op, output_index)
    output_index += 1
    op.output_desc.add().name = "kv_cache"
    kv_cache = Tensor(op, output_index)
    output_index += 1
    op.output_desc.add().name = "kr_cache"
    kr_cache = Tensor(op, output_index)
    output_index += 1
    op.output_desc.add().name = "dequant_scale_q_nope"
    dequant_scale_q_nope = Tensor(op, output_index)
    output_index += 1
    op.output_desc.add().name = "query_norm"
    query_norm = Tensor(op, output_index)
    output_index += 1
    op.output_desc.add().name = "dequant_scale_q_norm"
    dequant_scale_q_norm = Tensor(op, output_index)
    output_index += 1
    op.output_desc.add().name = "cur_k_nope"
    cur_k_nope = Tensor(op, output_index)
    output_index += 1
    op.output_desc.add().name = "cur_k_rope"
    cur_k_rope = Tensor(op, output_index)
    output_index += 1

    return query, query_rope, kv_cache, kr_cache, dequant_scale_q_nope, query_norm, dequant_scale_q_norm, \
        cur_k_nope, cur_k_rope


m = Library("fluent_sfa1", "FRAGMENT")
@impl(m, "npu_mla_prolog_v3", "Functionalize")
def custom_npu_mla_prolog_v3_func(
    token_x: Tensor,
    weight_dq: Tensor,
    weight_uq_qr: Tensor,
    weight_uk: Tensor,
    weight_dkv_kr: Tensor,
    rmsnorm_gamma_cq: Tensor,
    rmsnorm_gamma_ckv: Tensor,
    rope_sin: Tensor,
    rope_cos: Tensor,
    kv_cache: Tensor,
    kr_cache: Tensor,
    *,
    cache_index: Optional[Tensor] = None,
    dequant_scale_x: Optional[Tensor] = None,
    dequant_scale_w_dq: Optional[Tensor] = None,
    dequant_scale_w_uq_qr: Optional[Tensor] = None,
    dequant_scale_w_dkv_kr: Optional[Tensor] = None,
    quant_scale_ckv: Optional[Tensor] = None,
    quant_scale_ckr: Optional[Tensor] = None,
    smooth_scales_cq: Optional[Tensor] = None,
    actual_seq_len: Optional[Tensor] = None,
    k_nope_clip_alpha: Optional[Tensor] = None,
    rmsnorm_epsilon_cq: float = 1e-5,
    rmsnorm_epsilon_ckv: float = 1e-5,
    cache_mode: str = "PA_BSND",
    query_norm_flag: bool = False,
    weight_quant_mode: int = 0,
    kv_cache_quant_mode: int = 0,
    query_quant_mode: int = 0,
    ckvkr_repo_mode: int = 0,
    quant_scale_repo_mode: int = 0,
    tile_size: int = 128,
    qc_qr_scale: float = 1.0,
    kc_scale: float = 1.0,
    output_cur_k_nope_rope: bool = False
):
    query, query_rope, dequant_scale_q_nope, query_norm, dequant_scale_q_norm, cur_k_nope, cur_k_rope, kv_cache_out, kr_cache_out = torch.ops.fluent_sfa1.npu_mla_prolog_v3_functional(
        token_x, weight_dq, weight_uq_qr, weight_uk, weight_dkv_kr, rmsnorm_gamma_cq,
        rmsnorm_gamma_ckv, rope_sin, rope_cos, kv_cache, kr_cache, cache_index=cache_index,
        rmsnorm_epsilon_cq=rmsnorm_epsilon_cq, rmsnorm_epsilon_ckv=rmsnorm_epsilon_ckv,
        dequant_scale_x=dequant_scale_x, dequant_scale_w_dq=dequant_scale_w_dq,
        dequant_scale_w_uq_qr=dequant_scale_w_uq_qr, dequant_scale_w_dkv_kr=dequant_scale_w_dkv_kr,
        quant_scale_ckv=quant_scale_ckv, quant_scale_ckr=quant_scale_ckr,
        smooth_scales_cq=smooth_scales_cq, actual_seq_len=actual_seq_len, k_nope_clip_alpha=k_nope_clip_alpha,
        cache_mode=cache_mode, query_norm_flag=query_norm_flag, weight_quant_mode=weight_quant_mode,
        kv_cache_quant_mode=kv_cache_quant_mode, query_quant_mode=query_quant_mode, ckvkr_repo_mode=ckvkr_repo_mode,
        quant_scale_repo_mode=quant_scale_repo_mode, tile_size=tile_size, qc_qr_scale=qc_qr_scale, kc_scale=kc_scale,
        output_cur_k_nope_rope=output_cur_k_nope_rope)
    kv_cache.copy_(kv_cache_out)
    kr_cache.copy_(kr_cache_out)
    return query, query_rope, dequant_scale_q_nope, query_norm, dequant_scale_q_norm, cur_k_nope, cur_k_rope

# 为自定义算子注册converter，用于torch.compile 场景成图

# 注意： meta_outputs形参名为固定写法，若写错会影响ge节点的输出dtype与shape推导
@register_fx_node_ge_converter(torch.ops.fluent_sfa1.npu_mla_prolog_v3.default)
def convert_npu_npu_mla_prolog_v3(
    token_x: Tensor,
    weight_dq: Tensor,
    weight_uq_qr: Tensor,
    weight_uk: Tensor,
    weight_dkv_kr: Tensor,
    rmsnorm_gamma_cq: Tensor,
    rmsnorm_gamma_ckv: Tensor,
    rope_sin: Tensor,
    rope_cos: Tensor,
    kv_cache: Tensor,
    kr_cache: Tensor,
    *,
    cache_index: Optional[Tensor] = None,
    dequant_scale_x: Optional[Tensor] = None,
    dequant_scale_w_dq: Optional[Tensor] = None,
    dequant_scale_w_uq_qr: Optional[Tensor] = None,
    dequant_scale_w_dkv_kr: Optional[Tensor] = None,
    quant_scale_ckv: Optional[Tensor] = None,
    quant_scale_ckr: Optional[Tensor] = None,
    smooth_scales_cq: Optional[Tensor] = None,
    actual_seq_len: Optional[Tensor] = None,
    k_nope_clip_alpha: Optional[Tensor] = None,
    rmsnorm_epsilon_cq: float = 1e-5,
    rmsnorm_epsilon_ckv: float = 1e-5,
    cache_mode: str = "PA_BSND",
    query_norm_flag: bool = False,
    weight_quant_mode: int = 0,
    kv_cache_quant_mode: int = 0,
    query_quant_mode: int = 0,
    ckvkr_repo_mode: int = 0,
    quant_scale_repo_mode: int = 0,
    tile_size: int = 128,
    qc_qr_scale: float = 1.0,
    kc_scale: float = 1.0,
    output_cur_k_nope_rope: bool = False,
    meta_outputs: TensorSpec = None,
):
    (query, query_rope, kv_cache_out, kr_cache_out, dequant_scale_q_nope,
        query_norm, dequant_scale_q_norm, cur_k_nope, cur_k_rope) = torchair.ge.custom_op(
        "FluentMlaPrologV3",
        inputs={"token_x": token_x,
                "weight_dq": weight_dq,
                "weight_uq_qr": weight_uq_qr,
                "weight_uk": weight_uk,
                "weight_dkv_kr": weight_dkv_kr,
                "rmsnorm_gamma_cq": rmsnorm_gamma_cq,
                "rmsnorm_gamma_ckv": rmsnorm_gamma_ckv,
                "rope_sin": rope_sin,
                "rope_cos": rope_cos,
                "kv_cache": kv_cache,
                "kr_cache": kr_cache,
                "cache_index": cache_index,
                "dequant_scale_x": dequant_scale_x,
                "dequant_scale_w_dq": dequant_scale_w_dq,
                "dequant_scale_w_uq_qr": dequant_scale_w_uq_qr,
                "dequant_scale_w_dkv_kr": dequant_scale_w_dkv_kr,
                "quant_scale_ckv": quant_scale_ckv,
                "quant_scale_ckr": quant_scale_ckr,
                "smooth_scales_cq": smooth_scales_cq,
                "actual_seq_len": actual_seq_len,
                "k_nope_clip_alpha": k_nope_clip_alpha,
                },
        attrs={"rmsnorm_epsilon_cq": attr.Float(rmsnorm_epsilon_cq),
               "rmsnorm_epsilon_ckv": attr.Float(rmsnorm_epsilon_ckv),
               "cache_mode": attr.Str(cache_mode),
               "query_norm_flag": attr.Bool(query_norm_flag),
               "weight_quant_mode": attr.Int(weight_quant_mode),
               "kv_cache_quant_mode": attr.Int(kv_cache_quant_mode),
               "query_quant_mode": attr.Int(query_quant_mode),
               "ckvkr_repo_mode": attr.Int(ckvkr_repo_mode),
               "quant_scale_repo_mode": attr.Int(quant_scale_repo_mode),
               "tile_size": attr.Int(tile_size),
               "qc_qr_scale": attr.Float(qc_qr_scale),
               "kc_scale": attr.Float(kc_scale),
               "output_cur_k_nope_rope": attr.Bool(output_cur_k_nope_rope),
               },
        outputs=['query', 'query_rope', 'kv_cache', 'kr_cache', 'dequant_scale_q_nope',
                 'query_norm', 'dequant_scale_q_norm', 'cur_k_nope', 'cur_k_rope']
    )
    dont_prune_me(kv_cache_out)
    dont_prune_me(kr_cache_out)
    return query, query_rope, dequant_scale_q_nope, query_norm, dequant_scale_q_norm, cur_k_nope, cur_k_rope

@register_fx_node_ge_converter(torch.ops.fluent_sfa1.npu_mla_prolog_v3_functional.default)
def convert_npu_npu_mla_prolog_v3_functional(
    token_x: Tensor,
    weight_dq: Tensor,
    weight_uq_qr: Tensor,
    weight_uk: Tensor,
    weight_dkv_kr: Tensor,
    rmsnorm_gamma_cq: Tensor,
    rmsnorm_gamma_ckv: Tensor,
    rope_sin: Tensor,
    rope_cos: Tensor,
    kv_cache: Tensor,
    kr_cache: Tensor,
    *,
    cache_index: Optional[Tensor] = None,
    dequant_scale_x: Optional[Tensor] = None,
    dequant_scale_w_dq: Optional[Tensor] = None,
    dequant_scale_w_uq_qr: Optional[Tensor] = None,
    dequant_scale_w_dkv_kr: Optional[Tensor] = None,
    quant_scale_ckv: Optional[Tensor] = None,
    quant_scale_ckr: Optional[Tensor] = None,
    smooth_scales_cq: Optional[Tensor] = None,
    actual_seq_len: Optional[Tensor] = None,
    k_nope_clip_alpha: Optional[Tensor] = None,
    rmsnorm_epsilon_cq: float = 1e-5,
    rmsnorm_epsilon_ckv: float = 1e-5,
    cache_mode: str = "PA_BSND",
    query_norm_flag: bool = False,
    weight_quant_mode: int = 0,
    kv_cache_quant_mode: int = 0,
    query_quant_mode: int = 0,
    ckvkr_repo_mode: int = 0,
    quant_scale_repo_mode: int = 0,
    tile_size: int = 128,
    qc_qr_scale: float = 1.0,
    kc_scale: float = 1.0,
    output_cur_k_nope_rope: bool = False,
    meta_outputs: TensorSpec = None
):
    kv_cache_copy = ge.TensorMove(kv_cache)
    kr_cache_copy = ge.TensorMove(kr_cache)
    (
        query,
        query_rope,
        kv_cache_out,
        kr_cache_out,
        dequant_scale_q_nope,
        query_norm,
        dequant_scale_q_norm,
        cur_k_nope,
        cur_k_rope,
    ) = MlaPrologV3(
        token_x,
        weight_dq,
        weight_uq_qr,
        weight_uk,
        weight_dkv_kr,
        rmsnorm_gamma_cq,
        rmsnorm_gamma_ckv,
        rope_sin,
        rope_cos,
        kv_cache_copy,
        kr_cache_copy,
        cache_index=cache_index,
        dequant_scale_x=dequant_scale_x,
        dequant_scale_w_dq=dequant_scale_w_dq,
        dequant_scale_w_uq_qr=dequant_scale_w_uq_qr,
        dequant_scale_w_dkv_kr=dequant_scale_w_dkv_kr,
        quant_scale_ckv=quant_scale_ckv,
        quant_scale_ckr=quant_scale_ckr,
        smooth_scales_cq=smooth_scales_cq,
        actual_seq_len=actual_seq_len,
        k_nope_clip_alpha=k_nope_clip_alpha,
        rmsnorm_epsilon_cq=rmsnorm_epsilon_cq,
        rmsnorm_epsilon_ckv=rmsnorm_epsilon_ckv,
        cache_mode=cache_mode,
        query_norm_flag=query_norm_flag,
        weight_quant_mode=weight_quant_mode,
        kv_cache_quant_mode=kv_cache_quant_mode,
        query_quant_mode=query_quant_mode,
        ckvkr_repo_mode=ckvkr_repo_mode,
        quant_scale_repo_mode=quant_scale_repo_mode,
        tile_size=tile_size,
        qc_qr_scale=qc_qr_scale,
        kc_scale=kc_scale,
        output_cur_k_nope_rope=output_cur_k_nope_rope
    )
    return (
        query,
        query_rope,
        dequant_scale_q_nope,
        query_norm,
        dequant_scale_q_norm,
        cur_k_nope,
        cur_k_rope,
        kv_cache_out,
        kr_cache_out,
    )
