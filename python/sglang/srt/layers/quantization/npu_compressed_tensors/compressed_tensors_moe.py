import torch, torch_npu

from torch.nn import Parameter

from compressed_tensors import CompressionFormat
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import CompressedTensorsMoEMethod
from vllm.model_executor.utils import set_weight_attrs

from sglang.srt.env import global_server_args_dict

SUPPORTED_BITS = 8
SEQ_SPLIT_LENGTH = 4096
torch.npu.config.allow_internal_format = True

class NpuCompressedTensorsMoEMethod(CompressedTensorsMoEMethod):
    @staticmethod
    def get_moe_method(
        quant_config: "CompressedTensorsConfig"  # type: ignore # noqa E501
    ) -> "CompressedTensorsMoEMethod":
        # TODO: @dsikka: refactor this to use schemes as other kernels
        # are supported + check if the layer is being ignored.
        weight_quant = quant_config.target_scheme_map["Linear"].get("weights")
        input_quant = quant_config.target_scheme_map["Linear"].get(
            "input_activations")

        if quant_config._is_dynamic_token_w8a8(weight_quant, input_quant):
            return NpuCompressedTensorsW8A8Int8MoEMethod(quant_config)
        else:
            raise RuntimeError(
                f"Unsupported FusedMoe scheme: {weight_quant}, {input_quant}")

class NpuCompressedTensorsW8A8Int8MoEMethod(NpuCompressedTensorsMoEMethod):
    LAST_SEQ_LEN = None
    BEST_EXPERT_TOKENS = None

    def __init__(
            self,
            quant_config: "CompressedTensorsConfig"  # type: ignore # noqa E501
    ):
        self.quant_config = quant_config
        self.weight_quant = self.quant_config.target_scheme_map["Linear"].get(
            "weights")
        self.input_quant = self.quant_config.target_scheme_map["Linear"].get(
            "input_activations")
        self.is_transposed = False

        if not (self.quant_config.quant_format
                == CompressionFormat.int_quantized.value
                and self.weight_quant.num_bits == SUPPORTED_BITS):
            raise ValueError("For Fused MoE layers, only ",
                             f"{CompressionFormat.int_quantized.value} ",
                             "is supported for the following bits: ",
                             f"{SUPPORTED_BITS}")

    def create_weights(self, layer: torch.nn.Module, num_experts: int,
                       hidden_size: int, intermediate_size: int,
                       params_dtype: torch.dtype, **extra_weight_attrs):
        layer.use_presharded_weights = False
        extra_weight_attrs.update(
            {"is_transposed": self.is_transposed, "quant_method": self.weight_quant.strategy}
        )
        w13_weight = Parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size,
                hidden_size,
                device=torch.npu.current_device(),
                dtype=torch.int8,
            ), requires_grad=False
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                intermediate_size,
                device=torch.npu.current_device(),
                dtype=torch.int8,
            ), requires_grad=False
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        w13_weight_scale = Parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size,
                device=torch.npu.current_device(),
                dtype=torch.float,
            ), requires_grad=False
        )
        layer.register_parameter("w13_weight_scale", w13_weight_scale)
        set_weight_attrs(w13_weight_scale, extra_weight_attrs)

        w2_weight_scale = Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                device=torch.npu.current_device(),
                dtype=torch.bfloat16,
            ), requires_grad=False
        )
        layer.register_parameter("w2_weight_scale", w2_weight_scale)
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)

        if global_server_args_dict['npu_smooth_quant']:
            w13_smooth_scale = Parameter(torch.ones((layer.num_experts, hidden_size),
                                                    device=torch.npu.current_device(), dtype=torch.float32,),
                                         requires_grad=False)
            w2_smooth_scale = Parameter(torch.ones((layer.num_local_experts, intermediate_size),
                                                   device=torch.npu.current_device(), dtype=torch.float32,),
                                        requires_grad=False)
            layer.register_parameter("w13_smooth_scale", w13_smooth_scale)
            set_weight_attrs(w13_smooth_scale, extra_weight_attrs)
            layer.register_parameter("w2_smooth_scale", w2_smooth_scale)
            set_weight_attrs(w2_smooth_scale, extra_weight_attrs)

    def process_weights_after_loading(self, layer: torch.nn.Module):
        if not hasattr(layer, 'w13_smooth_scale'):
            layer.w13_smooth_scale = None
            layer.w2_smooth_scale = None
            layer.w13_smooth_scale_total = None
        if layer.w13_smooth_scale is not None:
            w13_smooth_scale = layer.w13_smooth_scale
            # w13_smooth_scale cast to torch.bfloat16 for npu_dynamic_quant and init_routing
            layer.w13_smooth_scale = Parameter(w13_smooth_scale.data[self.expert_idx_min:self.expert_idx_max, :].to(torch.float32),
                                               requires_grad=False)
            # w13_smooth_scale_total for dispatch
            layer.w13_smooth_scale_total = w13_smooth_scale
        if self.is_transposed:
            layer.w13_weight.data = torch_npu.npu_format_cast(layer.w13_weight.data, 29)
            layer.w2_weight.data = torch_npu.npu_format_cast(layer.w2_weight.data, 29)
        else:
            layer.w13_weight.data = torch_npu.npu_format_cast(layer.w13_weight.data.transpose(-1, -2).contiguous(), 29)
            layer.w2_weight.data = torch_npu.npu_format_cast(layer.w2_weight.data.transpose(-1, -2).contiguous(), 29)

    def apply(self, layer: torch.nn.Module, x, tokens_per_expert) -> torch.Tensor:
        dynamic_quant = None
        if isinstance(x, (list, tuple)):
            x, dynamic_quant = x
        w1 = layer.w13_weight.transpose(-1, -2) if self.is_transposed else layer.w13_weight
        w2 = layer.w2_weight.transpose(-1, -2) if self.is_transposed else layer.w2_weight
        expert_tokens = tokens_per_expert.to(torch.int64)
        if dynamic_quant is None:
            if layer.w13_smooth_scale is not None:
                group_index = torch.cumsum(tokens_per_expert, dim=0)
            else:
                group_index = None
            x, dynamic_quant = torch_npu.npu_dynamic_quant(
                x, smooth_scales=layer.w13_smooth_scale, group_index=group_index)
        out = torch_npu.npu_grouped_matmul(
            [x],
            [w1],
            bias=None,
            group_list=expert_tokens,
            split_item=3,  # 3 means combine result tokens belong to diff expert to a single tensor
            output_dtype=torch.int32,
            group_type=0,  # 0 means expand_x is grouped in m-axis (1 for k-axis is used in training backward)
            group_list_type=1  # 1 means group_list[i] is count for expert i (count mode), otherwise 0 is comsum mode
        )[0]

        gate_up_proj, pertoken_scale = torch_npu.npu_dequant_swiglu_quant(
            out,
            weight_scale=layer.w13_weight_scale.to(torch.float),
            group_index=expert_tokens,
            activation_scale=dynamic_quant,
            activate_left=True,
            quant_mode=1,
            quant_scale=layer.w2_smooth_scale
        )
        out = torch_npu.npu_grouped_matmul(
            [gate_up_proj],
            [w2],
            scale=[layer.w2_weight_scale],
            per_token_scale=[pertoken_scale.float()],
            bias=None,
            group_list=expert_tokens,
            split_item=3,
            output_dtype=torch.bfloat16,
            group_type=0,
            group_list_type=1
        )[0]

        return out
