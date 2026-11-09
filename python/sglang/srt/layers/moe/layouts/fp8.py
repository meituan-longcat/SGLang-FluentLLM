from __future__ import annotations

from functools import partial
import logging

import torch
from torch.nn import Module

from sglang.srt.layers.moe.layouts.common import MoELayout
from sglang.srt.layers.quantization import FusedMoeWeightScaleSupported
from sglang.srt.layers.quantization.fp8 import Fp8Config
from sglang.srt.layers.moe.layouts.load_functions import load_model_weight, load_per_tensor_weight_scale, load_single_value
from sglang.srt.distributed import get_moe_tensor_parallel_rank, get_tensor_model_parallel_world_size
from sglang.srt.layers.dense.gemms.fp8.fp8_kernel import (
    fp8_dtype,
    scaled_fp8_quant,
)
from sglang.srt.utils import (
    set_weight_attrs,
)

ACTIVATION_SCHEMES = ["static", "dynamic"]

logger = logging.getLogger(__name__)


class Fp8MoELayout(MoELayout):
    def __init__(self, quant_config: Fp8Config):
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: Module,
        num_local_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        assert params_dtype == torch.bfloat16, f"{params_dtype} != {torch.bfloat16}"
        # if self.quant_config.is_checkpoint_fp8_serialized:
        #     params_dtype = torch.float8_e4m3fn
        super().create_weights(
            layer,
            num_local_experts,
            hidden_size,
            intermediate_size_per_partition,
            torch.float8_e4m3fn,
            **extra_weight_attrs,
        )


class Fp8MoETensorQuantLayout(Fp8MoELayout):
    def __init__(self, quant_config: Fp8Config):
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: Module,
        num_local_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        super().create_weights(
            layer,
            num_local_experts,
            hidden_size,
            intermediate_size_per_partition,
            params_dtype,
            **extra_weight_attrs,
        )
        self.create_weight_scales(
            layer,
            num_local_experts,
            **extra_weight_attrs,
        )
        self.create_input_scales(
            layer,
            num_local_experts,
            **extra_weight_attrs,
        )

    def create_weight_scales(
        self,
        layer: Module,
        num_local_experts: int,
        **extra_weight_attrs,
    ):
        w13_weight_scale = torch.nn.Parameter(torch.ones(num_local_experts, 2, dtype=torch.float32), requires_grad=False)
        w2_weight_scale = torch.nn.Parameter(torch.ones(num_local_experts, dtype=torch.float32), requires_grad=False)

        layer.register_parameter("w13_weight_scale", w13_weight_scale)
        layer.register_parameter("w2_weight_scale", w2_weight_scale)

        if self.quant_config.is_checkpoint_fp8_serialized:
            set_weight_attrs(w13_weight_scale, extra_weight_attrs)
            set_weight_attrs(w2_weight_scale, extra_weight_attrs)

            set_weight_attrs(w13_weight_scale, {"weight_loader": load_per_tensor_weight_scale})
            set_weight_attrs(w2_weight_scale, {"weight_loader": load_per_tensor_weight_scale})

    def create_input_scales(
        self,
        layer: Module,
        num_local_experts: int,
        **extra_weight_attrs,
    ):
        assert self.quant_config.activation_scheme == "static"
        assert self.quant_config.is_checkpoint_fp8_serialized

        w13_input_scale = torch.nn.Parameter(torch.ones(num_local_experts, dtype=torch.float32), requires_grad=False)
        w2_input_scale = torch.nn.Parameter(torch.ones(num_local_experts, dtype=torch.float32), requires_grad=False)

        layer.register_parameter("w13_input_scale", w13_input_scale)
        layer.register_parameter("w2_input_scale", w2_input_scale)

        set_weight_attrs(w13_input_scale, extra_weight_attrs)
        set_weight_attrs(w2_input_scale, extra_weight_attrs)

        set_weight_attrs(w13_input_scale, {"weight_loader": load_single_value})
        set_weight_attrs(w2_input_scale, {"weight_loader": load_single_value})

    def process_weights_after_loading(self, layer: Module) -> None:
        if not self.quant_config.is_checkpoint_fp8_serialized:
            w13_weight = torch.empty_like(layer.w13_weight.data, dtype=fp8_dtype)
            w2_weight = torch.empty_like(layer.w2_weight.data, dtype=fp8_dtype)

            for expert in range(layer.num_local_experts):
                w13_weight[expert, :, :], layer.w13_weight_scale[expert] = scaled_fp8_quant(layer.w13_weight.data[expert, :, :])
                w2_weight[expert, :, :], layer.w2_weight_scale[expert] = scaled_fp8_quant(layer.w2_weight.data[expert, :, :])
            layer.w13_weight = torch.nn.Parameter(w13_weight, requires_grad=False)
            layer.w2_weight = torch.nn.Parameter(w2_weight, requires_grad=False)
            return
        else:
            raise RuntimeError("Not supported")


class Fp8MoEBlockQuantLayout(Fp8MoELayout):
    def __init__(self, quant_config: Fp8Config):
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: Module,
        num_local_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        super().create_weights(
            layer,
            num_local_experts,
            hidden_size,
            intermediate_size_per_partition,
            params_dtype,
            **extra_weight_attrs,
        )
        self.create_weight_scales(
            layer,
            num_local_experts,
            hidden_size,
            intermediate_size_per_partition,
            **extra_weight_attrs,
        )

    def create_weight_scales(
        self,
        layer: Module,
        num_local_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        **extra_weight_attrs,
    ):
        tp_size = get_tensor_model_parallel_world_size()
        block_n, block_k = (
            self.quant_config.weight_block_size[0],
            self.quant_config.weight_block_size[1],
        )
        # NOTE(HandH1998): To ensure proper alignment of the block-wise quantization scales, the output_size of the weights for both the gate and up layers must be divisible by block_n.
        # Required by column parallel or enabling merged weights
        if intermediate_size_per_partition % block_n != 0:
            raise ValueError(
                f"The output_size of gate's and up's weight = "
                f"{intermediate_size_per_partition} is not divisible by "
                f"weight quantization block_n = {block_n}."
            )
        if tp_size > 1:
            # Required by row parallel
            if intermediate_size_per_partition % block_k != 0:
                raise ValueError(
                    f"The input_size of down's weight = "
                    f"{intermediate_size_per_partition} is not divisible by "
                    f"weight quantization block_k = {block_k}."
                )

        w13_weight_scale = torch.nn.Parameter(
            torch.ones(
                num_local_experts,
                2 * ((intermediate_size_per_partition + block_n - 1) // block_n),
                (hidden_size + block_k - 1) // block_k,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        w2_weight_scale = torch.nn.Parameter(
            torch.ones(
                num_local_experts,
                (hidden_size + block_n - 1) // block_n,
                (intermediate_size_per_partition + block_k - 1) // block_k,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale_inv", w13_weight_scale)
        layer.register_parameter("w2_weight_scale_inv", w2_weight_scale)

        set_weight_attrs(w13_weight_scale, extra_weight_attrs)
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)

        weight_loader = partial(
            load_model_weight, tp_rank=get_moe_tensor_parallel_rank(), is_bias=False, use_presharded_weights=False, do_transpose=False
        )  # FIXME
        set_weight_attrs(w13_weight_scale, {"weight_loader": weight_loader})
        set_weight_attrs(w2_weight_scale, {"weight_loader": weight_loader})

        assert self.quant_config.is_checkpoint_fp8_serialized
        assert self.quant_config.activation_scheme == "dynamic"

        extra_weight_attrs.update({"quant_method": FusedMoeWeightScaleSupported.BLOCK.value})

    def process_weights_after_loading(self, layer: Module) -> None:
        return


class Fp8MoEBlockQuantUseCutlassLayout(Fp8MoEBlockQuantLayout):
    def __init__(self, quant_config: Fp8Config):
        self.quant_config = quant_config

    def create_aux(
        self,
        num_local_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        device,
    ):
        self.ab_strides1 = torch.full(
            (num_local_experts,),
            hidden_size,
            device=device,
            dtype=torch.int64,
        )
        self.c_strides1 = torch.full(
            (num_local_experts,),
            2 * intermediate_size_per_partition,
            device=device,
            dtype=torch.int64,
        )
        self.ab_strides2 = torch.full(
            (num_local_experts,),
            intermediate_size_per_partition,
            device=device,
            dtype=torch.int64,
        )
        self.c_strides2 = torch.full(
            (num_local_experts,),
            hidden_size,
            device=device,
            dtype=torch.int64,
        )
        self.workspace = torch.empty(90000, device=device, dtype=torch.uint8)
        self.a_ptr = torch.empty(num_local_experts, device=device, dtype=torch.int64)
        self.b_ptr = torch.empty(num_local_experts, device=device, dtype=torch.int64)
        self.out_ptr = torch.empty(num_local_experts, device=device, dtype=torch.int64)
        self.a_scales_ptr = torch.empty(num_local_experts, device=device, dtype=torch.int64)
        self.b_scales_ptr = torch.empty(num_local_experts, device=device, dtype=torch.int64)
        self.expert_offsets = torch.empty(num_local_experts + 1, device=device, dtype=torch.int32)
        self.problem_sizes1 = torch.empty(num_local_experts, 3, device=device, dtype=torch.int32)
        self.problem_sizes2 = torch.empty(num_local_experts, 3, device=device, dtype=torch.int32)

    def create_weights(
        self,
        layer: Module,
        num_local_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        super().create_weights(
            layer,
            num_local_experts,
            hidden_size,
            intermediate_size_per_partition,
            params_dtype,
            **extra_weight_attrs,
        )
        self.create_aux(
            num_local_experts,
            hidden_size,
            intermediate_size_per_partition,
            layer.w13_weight.device,
        )

    def process_weights_after_loading(self, layer: Module) -> None:
        return
