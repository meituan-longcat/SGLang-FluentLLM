from abc import abstractmethod
from functools import partial

import torch

from sglang.srt.distributed.parallel_state import get_moe_tensor_parallel_rank
from sglang.srt.layers.moe.layouts.load_functions import load_model_weight
from sglang.srt.utils import set_weight_attrs


class MoELayout:
    def create_weights(
        self,
        layer: torch.nn.Module,
        num_local_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        with_bias: bool = False,
        **extra_weight_attrs,
    ):
        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_local_experts,
                2 * intermediate_size_per_partition,
                hidden_size,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        w2_weight = torch.nn.Parameter(
            torch.empty(
                num_local_experts,
                hidden_size,
                intermediate_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )

        layer.register_parameter("w13_weight", w13_weight)
        layer.register_parameter("w2_weight", w2_weight)

        set_weight_attrs(w13_weight, extra_weight_attrs)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        if with_bias:
            w13_weight_bias = torch.nn.Parameter(
                torch.zeros(
                    num_local_experts,
                    2 * intermediate_size_per_partition,
                    dtype=params_dtype,
                ),
                requires_grad=False,
            )
            layer.register_parameter("w13_weight_bias", w13_weight_bias)
            set_weight_attrs(w13_weight_bias, extra_weight_attrs)

            w2_weight_bias = torch.nn.Parameter(
                torch.zeros(
                    num_local_experts,
                    hidden_size,
                    dtype=params_dtype,
                ),
                requires_grad=False,
            )
            layer.register_parameter("w2_weight_bias", w2_weight_bias)
            set_weight_attrs(w2_weight_bias, extra_weight_attrs)

        weight_loader = partial(
            load_model_weight,
            tp_rank=get_moe_tensor_parallel_rank(),
            is_bias=False,
            use_presharded_weights=False,
            do_transpose=False
        )  # FIXME
        set_weight_attrs(w13_weight, {"weight_loader": weight_loader})
        set_weight_attrs(w2_weight, {"weight_loader": weight_loader})
        if with_bias:
            set_weight_attrs(w13_weight_bias, {"weight_loader": weight_loader})
            set_weight_attrs(w2_weight_bias, {"weight_loader": weight_loader})

    @abstractmethod
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Process the weight after loading.

        This can be used for example, to transpose weights for computation.
        """
