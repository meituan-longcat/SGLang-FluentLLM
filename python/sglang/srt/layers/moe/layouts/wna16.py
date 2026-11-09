# Adapted from https://github.com/vllm-project/vllm/tree/v0.8.2/vllm/model_executor/layers/quantization/compressed_tensors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import torch
import enum
import logging
from functools import partial
from enum import Enum
from typing import TYPE_CHECKING
from sglang.srt.layers.quantization.utils import (
    replace_parameter,
)
from sglang.srt.utils import (
    get_compiler_backend,
    set_weight_attrs,
)
from sglang.srt.distributed.parallel_state import get_moe_tensor_parallel_rank
from sglang.srt.layers.quantization.compressed_tensors.gptq_marlin_moe import gptq_marlin_moe_repack, marlin_moe_permute_scales
from sglang.srt.layers.moe.layouts.compress_tensor_load_functions import load_model_weight


if TYPE_CHECKING:
    from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
        CompressedTensorsConfig,
    )

logger = logging.getLogger(__name__)


def _mask_topk_ids_cpu_experts(topk_ids: torch.Tensor, num_gpu_experts: int):
    """Mask topk_ids >= num_gpu_experts by setting them to -1."""
    topk_ids[topk_ids >= num_gpu_experts] = -1


@torch.compile(dynamic=True, backend=get_compiler_backend())
def mask_cpu_expert_ids(topk_ids: torch.Tensor, num_gpu_experts: int):
    """mask CPU expert IDs."""
    _mask_topk_ids_cpu_experts(topk_ids, num_gpu_experts)
    return topk_ids


class GPTQMarlinState(Enum):
    REPACK = enum.auto()
    READY = enum.auto()


class CompressedTensorsWNA16MoELayout:
    def __init__(self, quant_config: CompressedTensorsConfig):
        self.quant_config = quant_config
        # TODO: @dsikka: refactor this to use schemes as other kernels
        # are supported + check if the layer is being ignored.
        config = self.quant_config.target_scheme_map["Linear"].get("weights")
        self.num_bits = config.num_bits
        self.packed_factor = 32 // config.num_bits
        self.strategy = config.strategy
        self.group_size = config.group_size
        self.actorder = config.actorder
        assert config.symmetric, "Only symmetric quantization is supported for MoE"

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        # Will transpose the loaded weight along the
        # intermediate and hidden dim sizes. Will
        # shard for TP along the transposed dims
        extra_weight_attrs.update({"is_transposed": True, "quant_method": self.strategy})
        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size // self.packed_factor,
                2 * intermediate_size_per_partition,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_packed", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)
        weight_loader = partial(
            load_model_weight,
            weight_name="w13_weight_packed",
            tp_rank=get_moe_tensor_parallel_rank(),
        )  # FIXME
        set_weight_attrs(w13_weight, {"weight_loader": weight_loader})

        w2_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                intermediate_size_per_partition // self.packed_factor,
                hidden_size,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_packed", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)
        weight_loader = partial(
            load_model_weight,
            weight_name="w2_weight_packed",
            tp_rank=get_moe_tensor_parallel_rank(),
        )  # FIXME
        set_weight_attrs(w2_weight, {"weight_loader": weight_loader})

        # In the case where we have actorder/g_idx,
        # we do not partition the w2 scales
        load_full_w2 = self.actorder and self.group_size != -1

        if load_full_w2:
            w2_scales_size = intermediate_size_per_partition * layer.moe_tp_size
        else:
            w2_scales_size = intermediate_size_per_partition

        self.is_k_full = (not self.actorder) or layer.moe_tp_size == 1

        if self.strategy == "channel":
            num_groups_w2 = num_groups_w13 = 1
            self.group_size = -1
        else:
            num_groups_w2 = w2_scales_size // self.group_size
            num_groups_w13 = hidden_size // self.group_size

        w13_scale = torch.nn.Parameter(
            torch.ones(
                num_experts,
                num_groups_w13,
                2 * intermediate_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale", w13_scale)
        set_weight_attrs(w13_scale, extra_weight_attrs)
        weight_loader = partial(
            load_model_weight,
            weight_name="w13_weight_scale",
            tp_rank=get_moe_tensor_parallel_rank(),
        )  # FIXME
        set_weight_attrs(w13_scale, {"weight_loader": weight_loader})

        w2_scale = torch.nn.Parameter(
            torch.ones(num_experts, num_groups_w2, hidden_size, dtype=params_dtype),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_scale", w2_scale)
        set_weight_attrs(w2_scale, extra_weight_attrs)
        set_weight_attrs(w2_scale, {"load_full_w2": load_full_w2})
        weight_loader = partial(
            load_model_weight,
            weight_name="w2_weight_scale",
            tp_rank=get_moe_tensor_parallel_rank(),
        )  # FIXME
        set_weight_attrs(w2_scale, {"weight_loader": weight_loader})

        w2_weight_shape = torch.nn.Parameter(torch.empty(num_experts, 2), requires_grad=False)
        layer.register_parameter("w2_weight_shape", w2_weight_shape)
        set_weight_attrs(w2_weight_shape, extra_weight_attrs)
        weight_loader = partial(
            load_model_weight,
            weight_name="w2_weight_shape",
            tp_rank=get_moe_tensor_parallel_rank(),
        )  # FIXME
        set_weight_attrs(w2_weight_shape, {"weight_loader": weight_loader})

        w13_weight_shape = torch.nn.Parameter(torch.empty(num_experts, 2), requires_grad=False)
        layer.register_parameter("w13_weight_shape", w13_weight_shape)
        set_weight_attrs(w13_weight_shape, extra_weight_attrs)
        weight_loader = partial(
            load_model_weight,
            weight_name="w13_weight_shape",
            tp_rank=get_moe_tensor_parallel_rank(),
        )  # FIXME
        set_weight_attrs(w13_weight_shape, {"weight_loader": weight_loader})

        w13_g_idx = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_g_idx", w13_g_idx)
        set_weight_attrs(w13_g_idx, extra_weight_attrs)
        weight_loader = partial(
            load_model_weight,
            weight_name="w13_weight_g_idx",
            tp_rank=get_moe_tensor_parallel_rank(),
        )  # FIXME
        set_weight_attrs(w13_g_idx, {"weight_loader": weight_loader})

        w2_g_idx = torch.nn.Parameter(
            torch.empty(
                num_experts,
                intermediate_size_per_partition,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_g_idx", w2_g_idx)
        set_weight_attrs(w2_g_idx, extra_weight_attrs)
        weight_loader = partial(
            load_model_weight,
            weight_name="w2_weight_g_idx",
            tp_rank=get_moe_tensor_parallel_rank(),
        )  # FIXME
        set_weight_attrs(w2_g_idx, {"weight_loader": weight_loader})

        w13_g_idx_sort_indices = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_g_idx_sort_indices", w13_g_idx_sort_indices)
        set_weight_attrs(w13_g_idx_sort_indices, extra_weight_attrs)
        weight_loader = partial(
            load_model_weight,
            weight_name="w13_g_idx_sort_indices",
            tp_rank=get_moe_tensor_parallel_rank(),
        )  # FIXME
        set_weight_attrs(w13_g_idx_sort_indices, {"weight_loader": weight_loader})

        w2_g_idx_sort_indices = torch.nn.Parameter(
            torch.empty(
                num_experts,
                intermediate_size_per_partition,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_g_idx_sort_indices", w2_g_idx_sort_indices)
        set_weight_attrs(w2_g_idx_sort_indices, extra_weight_attrs)
        weight_loader = partial(
            load_model_weight,
            weight_name="w2_g_idx_sort_indices",
            tp_rank=get_moe_tensor_parallel_rank(),
        )  # FIXME
        set_weight_attrs(w2_g_idx_sort_indices, {"weight_loader": weight_loader})

        layer.a13_scale = None
        layer.a2_scale = None
        layer.marlin_state = GPTQMarlinState.REPACK

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        def replace_tensor(name, new_t):
            # It is important to use resize_() here since it ensures
            # the same buffer is reused
            getattr(layer, name).resize_(new_t.shape)
            getattr(layer, name).copy_(new_t)
            del new_t

        num_experts = layer.w13_weight_g_idx.shape[0]
        device = layer.w13_weight_g_idx.device

        # when running models with grouped act order,
        # resort to g_idx values provided in checkpoint
        if self.actorder == "group":
            w13_g_idx_sort_indices = torch.empty_like(layer.w13_weight_g_idx)
            w2_g_idx_sort_indices = torch.empty_like(layer.w2_weight_g_idx)
            w13_sorted_g_idx = torch.empty_like(layer.w13_weight_g_idx)
            w2_sorted_g_idx = torch.empty_like(layer.w2_weight_g_idx)

            for e in range(num_experts):
                w13_g_idx_sort_indices[e] = torch.argsort(layer.w13_weight_g_idx[e]).to(torch.int32)
                w2_g_idx_sort_indices[e] = torch.argsort(layer.w2_weight_g_idx[e]).to(torch.int32)
                w13_sorted_g_idx[e] = layer.w13_weight_g_idx[e][w13_g_idx_sort_indices[e]]
                w2_sorted_g_idx[e] = layer.w2_weight_g_idx[e][w2_g_idx_sort_indices[e]]

            replace_parameter(layer, "w13_weight_g_idx", w13_sorted_g_idx)
            replace_parameter(layer, "w2_weight_g_idx", w2_sorted_g_idx)
            replace_parameter(layer, "w13_g_idx_sort_indices", w13_g_idx_sort_indices)
            replace_parameter(layer, "w2_g_idx_sort_indices", w2_g_idx_sort_indices)

        else:
            layer.w13_weight_g_idx = torch.nn.Parameter(
                torch.empty((num_experts, 0), dtype=torch.int32, device=device),
                requires_grad=False,
            )
            layer.w2_weight_g_idx = torch.nn.Parameter(
                torch.empty((num_experts, 0), dtype=torch.int32, device=device),
                requires_grad=False,
            )
            layer.w13_g_idx_sort_indices = torch.nn.Parameter(
                torch.empty((num_experts, 0), dtype=torch.int32, device=device),
                requires_grad=False,
            )
            layer.w2_g_idx_sort_indices = torch.nn.Parameter(
                torch.empty((num_experts, 0), dtype=torch.int32, device=device),
                requires_grad=False,
            )

        marlin_w13_qweight = gptq_marlin_moe_repack(
            layer.w13_weight_packed,
            layer.w13_g_idx_sort_indices,
            layer.w13_weight_packed.shape[1] * self.packed_factor,
            layer.w13_weight_packed.shape[2],
            self.num_bits,
        )
        replace_parameter(layer, "w13_weight_packed", marlin_w13_qweight)
        marlin_w2_qweight = gptq_marlin_moe_repack(
            layer.w2_weight_packed,
            layer.w2_g_idx_sort_indices,
            layer.w2_weight_packed.shape[1] * self.packed_factor,
            layer.w2_weight_packed.shape[2],
            self.num_bits,
        )
        replace_parameter(layer, "w2_weight_packed", marlin_w2_qweight)

        # TODO(Not using the latest sgl-kernel operators yet, skipping scale flipping for now)
        # Repack scales
        marlin_w13_scales = marlin_moe_permute_scales(
            layer.w13_weight_scale,
            layer.w13_weight_packed.shape[2],
            layer.w13_weight_scale.shape[2],
            self.group_size,
        )
        replace_parameter(layer, "w13_weight_scale", marlin_w13_scales)

        marlin_w2_scales = marlin_moe_permute_scales(
            layer.w2_weight_scale,
            layer.w2_weight_scale.shape[1] * (self.group_size if self.group_size != -1 else self.packed_factor),
            layer.w2_weight_scale.shape[2],
            self.group_size,
        )
        replace_parameter(layer, "w2_weight_scale", marlin_w2_scales)
