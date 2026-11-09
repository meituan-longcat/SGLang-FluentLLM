from typing import Any

import numpy as np
import torch, torch_npu
import flash_ops  # noqa: F401  # Registers torch.ops.custom.flat_quant.

from torch.nn import Parameter

from compressed_tensors import CompressionFormat
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import CompressedTensorsMoEMethod
from vllm.model_executor.utils import set_weight_attrs

from sglang.srt.env import global_server_args_dict
from sglang.srt.utils import dispose_tensor
from sglang.srt.utils import get_colorful_logger
logger = get_colorful_logger(__name__)

SUPPORTED_BITS = 8
SEQ_SPLIT_LENGTH = 4096
torch.npu.config.allow_internal_format = True


def kronecker_matmul(x, hadL, hadR):
    """equivalent to
        had = torch.kron(hadL, hadR)
        x = x.reshape(-1, had.shape[0])
        x = x.matmul(had).reshape(init_shape)
    """
    init_shape = x.shape
    x = x.reshape(-1, hadL.shape[0], hadR.shape[0])
    x = torch.matmul(x, hadR)
    x = torch.matmul(hadL.T, x)
    return x.reshape(init_shape)

def clip_x_ratio(x, clip_ratio=1.0):
    xmax = x.abs().amax(1, keepdim=True)
    xmax.mul_(clip_ratio)
    x.clamp_(min=-xmax, max=xmax)
    return x

class InvDecomposeTransMatrix(torch.nn.Module):
    def __init__(self, left_size, right_size, add_diag=False):
        super(InvDecomposeTransMatrix, self).__init__()
        self.matrix_left = torch.nn.Parameter(torch.empty(left_size, left_size, device=torch.npu.current_device(), dtype=torch.bfloat16), requires_grad=False)
        self.matrix_right = torch.nn.Parameter(torch.empty(right_size, right_size, device=torch.npu.current_device(), dtype=torch.bfloat16), requires_grad=False)
        if add_diag:
            self.diag_scale = torch.nn.Parameter(torch.empty(left_size * right_size, device=torch.npu.current_device(), dtype=torch.bfloat16), requires_grad=False)

    def forward(self, inp):
        if hasattr(self, "diag_scale"):
            inp = inp * self.diag_scale
        matrix_left, matrix_right = self.matrix_left, self.matrix_right
        return kronecker_matmul(inp, matrix_left, matrix_right)

    def __repr__(self):
        res = f"InvDecomposeTransMatrix(matrix.shape={self.matrix_right.shape}, matrix_right.shape={self.matrix_right.shape}, )"
        return res

class NpuCompressedTensorsMoEMethod(CompressedTensorsMoEMethod):
    @staticmethod
    def get_moe_method(
        quant_config: "CompressedTensorsConfig",  # type: ignore # noqa E501
    ) -> "CompressedTensorsMoEMethod":
        # TODO: @dsikka: refactor this to use schemes as other kernels
        # are supported + check if the layer is being ignored.
        weight_quant = quant_config.target_scheme_map["Linear"].get("weights")
        input_quant = quant_config.target_scheme_map["Linear"].get(
            "input_activations")
        if "Moe" in quant_config.target_scheme_map:
            weight_quant = quant_config.target_scheme_map["Moe"].get("weights")
            input_quant = quant_config.target_scheme_map["Moe"].get(
                "input_activations")
        if quant_config._is_dynamic_token_w8a8(weight_quant, input_quant):
            return NpuCompressedTensorsW8A8Int8MoEMethod(quant_config)
        elif quant_config._is_w4a16(weight_quant, input_quant):
            return NpuCompressedTensorsW4A16MoEMethod(quant_config)
        elif quant_config._is_dynamic_token_w4a4(weight_quant, input_quant):
            return NpuCompressedTensorsW4A4Int4MoEMethod(quant_config)
        else:
            raise RuntimeError(f"Unsupported FusedMoe scheme: {weight_quant}, {input_quant}")


class NpuCompressedTensorsW4A4Int4MoEMethod(NpuCompressedTensorsMoEMethod):
    LAST_SEQ_LEN = None
    BEST_EXPERT_TOKENS = None

    def __init__(
            self,
            quant_config: "CompressedTensorsConfig"  # type: ignore # noqa E501
    ):
        self.quant_config = quant_config
        if "Linear" in self.quant_config.target_scheme_map:
            scheme = quant_config.target_scheme_map["Linear"]
        if "Moe" in quant_config.target_scheme_map:
            scheme = quant_config.target_scheme_map["Moe"]
        self.weight_quant = scheme.get(
            "weights")
        self.input_quant = scheme.get(
            "input_activations")
        self.is_transposed = False
        self.w4a4_gmm_is_transposed = bool((self.quant_config.config or {}).get("w4a4_gmm_is_transposed", False))

        if not (self.quant_config.quant_format
                == CompressionFormat.int_quantized.value
                and self.weight_quant.num_bits == 4):
            raise ValueError("For Fused MoE layers, only ",
                             f"{CompressionFormat.int_quantized.value} ",
                             "is supported for the following bits: 4")

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
                hidden_size // 8,
                device=torch.npu.current_device(),
                dtype=torch.int32,
            ) if self.w4a4_gmm_is_transposed else torch.empty(
                num_experts,
                2 * intermediate_size // 8,
                hidden_size,
                device=torch.npu.current_device(),
                dtype=torch.int32,
            ), requires_grad=False
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                intermediate_size // 8,
                device=torch.npu.current_device(),
                dtype=torch.int32,
            ) if self.w4a4_gmm_is_transposed else torch.empty(
                num_experts,
                hidden_size // 8,
                intermediate_size,
                device=torch.npu.current_device(),
                dtype=torch.int32,
            ), requires_grad=False
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        w13_weight_scale = Parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size,
                device=torch.npu.current_device(),
                dtype=torch.float32,
            ), requires_grad=False
        )
        layer.register_parameter("w13_weight_scale", w13_weight_scale)
        set_weight_attrs(w13_weight_scale, extra_weight_attrs)

        w2_weight_scale = Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                device=torch.npu.current_device(),
                dtype=torch.float32,
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

    @staticmethod
    def pack_scale_to_uint64(scale):
        E = scale.shape[0]
        N = scale.shape[1]
        current_device = scale.device
        scale_np = scale.cpu().float().numpy().reshape(E, 1, N)
        scale_np.dtype = np.uint32
        scale_uint64 = np.zeros((E, 1, N * 2), dtype=np.uint32)
        scale_uint64[..., ::2] = scale_np
        scale_uint64.dtype = np.uint64
        return torch.from_numpy(scale_uint64).to(current_device)

    def process_weights_after_loading(self, layer: torch.nn.Module):
        if not self.w4a4_gmm_is_transposed:
            layer.w13_weight.data = layer.w13_weight.data.transpose(-1, -2).contiguous()
            layer.w2_weight.data = layer.w2_weight.data.transpose(-1, -2).contiguous()
        layer.w13_weight.data = torch_npu.npu_format_cast(layer.w13_weight.data, 29, customize_dtype=3)
        layer.w2_weight.data = torch_npu.npu_format_cast(layer.w2_weight.data, 29, customize_dtype=3)

        layer.w13_weight_scale.data = self.pack_scale_to_uint64(layer.w13_weight_scale.data)
        layer.w2_weight_scale.data = self.pack_scale_to_uint64(layer.w2_weight_scale.data)

        layer.w1_trans.matrix_left.data = layer.w1_trans.matrix_left.data.T.contiguous()
        layer.w2_trans.matrix_left.data = layer.w2_trans.matrix_left.data.T.contiguous()

        # npu_kronecker_quant 的 clip_ratio 形参是 Optional[float]，torch.compile 下
        # 无法从 FakeTensor 提取 float，故在此把常量 clip 张量转为 Python float 缓存。
        layer.w1_clip_factor_a = layer.act_quantizer_w1.clip_factor_a.item()
        layer.w2_clip_factor_a = layer.act_quantizer_w2.clip_factor_a.item()

    def quantize_moe_input(self, experts: torch.nn.Module, x, group_list=None):
        # w1 输入的 FlatQuant。per-token 且与 expert 无关（w1_trans/clip 全 expert 共享），
        # 因此可提到 pre_comm(AllGather) 之前执行，用 int4 通信，数值等价。
        # prefill(int4通信)：量化在 AllGather 前，此时未按 expert 分组，group_list=None；
        # decode：量化在 dispatch 后，已按 expert permute，传 expert 的 group_list。
        # mlp_flat_quant 同时支持 group_list=None 和 0 行输入，故两条路及空卡统一走此算子。
        x_q, pertoken_scale = torch.ops.custom.flat_quant(
            x.view(x.shape[0], experts.w1_trans_matrix_left_shape, experts.w1_trans_matrix_right_shape),
            experts.w1_trans.matrix_left,
            experts.w1_trans.matrix_right,
            group_list=group_list,
            clip_ratio=experts.w1_clip_factor_a,
            group_list_type=1,
        )
        # 用 flatten(1) 而非 view(shape[0], -1)：空卡(0 token)时 numel=0，
        # view 的 -1 无法推断（0*k==0 对任意 k 成立）会报 ambiguous；flatten 按已知维度展开，安全。
        x_q = x_q.flatten(1)
        return x_q, pertoken_scale

    def apply(
        self,
        experts: torch.nn.Module,
        x,
        tokens_per_expert,
        *,
        gmm2_tuning_config=None,
    ) -> torch.Tensor:
        w1 = experts.w13_weight
        w2 = experts.w2_weight
        if self.w4a4_gmm_is_transposed:
            # (E, N, K//8) -> (E, K//8, N)，与旧布局 process_weights 后的排布对齐。
            w1 = w1.transpose(1, 2)
            w2 = w2.transpose(1, 2)
        w1_weight_scale = experts.w13_weight_scale
        w2_weight_scale = experts.w2_weight_scale

        expert_tokens = tokens_per_expert.to(torch.int64)

        if isinstance(x, (list, tuple)):
            # int4 通信路径：输入已在 pre_comm 前量化并随 dispatch permute。
            x, pertoken_scale = x
        else:
            # decode 路径：x 是 dispatch 后的 expand_x，已按 expert permute，
            # 需传 expert 的 group_list 才能正确分组做 FlatQuant。
            x, pertoken_scale = self.quantize_moe_input(experts, x, group_list=expert_tokens)
        out = torch_npu.npu_grouped_matmul(
            [x],
            [w1],
            scale=[w1_weight_scale],
            per_token_scale=[pertoken_scale],
            bias=None,
            group_list=expert_tokens,
            split_item=3,
            output_dtype=torch.bfloat16,
            group_type=0,
            group_list_type=1
        )[0]

        out = torch_npu.mlp_split_swiglu(out, tokens_per_expert.to(torch.int32), local_exp_start=0, local_exp_end=experts.num_local_experts)

        # out, pertoken_scale = torch_npu.npu_kronecker_quant(
        #     out.view(out.shape[0], experts.w2_trans_matrix_left_shape, experts.w2_trans_matrix_right_shape),
        #     experts.w2_trans.matrix_left,
        #     experts.w2_trans.matrix_right,
        #     clip_ratio=experts.w2_clip_factor_a
        # )
        out, pertoken_scale = torch.ops.custom.flat_quant(
            out.view(out.shape[0], experts.w2_trans_matrix_left_shape, experts.w2_trans_matrix_right_shape),
            experts.w2_trans.matrix_left,
            experts.w2_trans.matrix_right,
            group_list=expert_tokens,          # 复用上面的 expert_tokens(int64, count 语义)
            clip_ratio=experts.w2_clip_factor_a,
            group_list_type=1,                  # 与 grouped_matmul 的 group_list_type=1 对齐
        )
        # 同上：flatten(1) 避免空卡 0 元素时 view(-1) 歧义报错。
        out = out.flatten(1)
        use_gmm2_tuning = (
            gmm2_tuning_config is not None
            and gmm2_tuning_config[0] <= out.shape[0]
        )
        if not use_gmm2_tuning:
            out = torch_npu.npu_grouped_matmul(
                [out],
                [w2],
                scale=[w2_weight_scale],
                per_token_scale=[pertoken_scale],
                bias=None,
                group_list=expert_tokens,
                split_item=3,
                output_dtype=torch.bfloat16,
                group_type=0,
                group_list_type=1,
            )[0]
        else:
            out = torch_npu.npu_grouped_matmul(
                [out],
                [w2],
                scale=[w2_weight_scale],
                per_token_scale=[pertoken_scale],
                bias=None,
                group_list=expert_tokens,
                split_item=3,
                output_dtype=torch.bfloat16,
                group_type=0,
                group_list_type=1,
                tuning_config=gmm2_tuning_config,
            )[0]

        return out


class NpuCompressedTensorsW8A8Int8MoEMethod(NpuCompressedTensorsMoEMethod):
    LAST_SEQ_LEN = None
    BEST_EXPERT_TOKENS = None

    def __init__(
        self,
        quant_config: "CompressedTensorsConfig",  # type: ignore # noqa E501
    ):
        self.quant_config = quant_config
        self.weight_quant = self.quant_config.target_scheme_map["Linear"].get("weights")
        self.input_quant = self.quant_config.target_scheme_map["Linear"].get("input_activations")
        self.is_transposed = False

        if not (self.quant_config.quant_format == CompressionFormat.int_quantized.value and self.weight_quant.num_bits == SUPPORTED_BITS):
            raise ValueError(
                "For Fused MoE layers, only ",
                f"{CompressionFormat.int_quantized.value} ",
                "is supported for the following bits: ",
                f"{SUPPORTED_BITS}",
            )

    def create_weights(
        self, layer: torch.nn.Module, num_experts: int, hidden_size: int, intermediate_size: int, params_dtype: torch.dtype, **extra_weight_attrs
    ):
        layer.use_presharded_weights = False
        extra_weight_attrs.update({"is_transposed": self.is_transposed, "quant_method": self.weight_quant.strategy})
        w13_weight = Parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size,
                hidden_size,
                device=torch.npu.current_device(),
                dtype=torch.int8,
            ),
            requires_grad=False,
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
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        w13_weight_scale = Parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size,
                device=torch.npu.current_device(),
                dtype=torch.float,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale", w13_weight_scale)
        set_weight_attrs(w13_weight_scale, extra_weight_attrs)

        w2_weight_scale = Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                device=torch.npu.current_device(),
                dtype=torch.bfloat16,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_scale", w2_weight_scale)
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)

        if global_server_args_dict["npu_smooth_quant"]:
            w13_smooth_scale = Parameter(
                torch.ones(
                    (layer.num_experts, hidden_size),
                    device=torch.npu.current_device(),
                    dtype=torch.float32,
                ),
                requires_grad=False,
            )
            w2_smooth_scale = Parameter(
                torch.ones(
                    (layer.num_local_experts, intermediate_size),
                    device=torch.npu.current_device(),
                    dtype=torch.float32,
                ),
                requires_grad=False,
            )
            layer.register_parameter("w13_smooth_scale", w13_smooth_scale)
            set_weight_attrs(w13_smooth_scale, extra_weight_attrs)
            layer.register_parameter("w2_smooth_scale", w2_smooth_scale)
            set_weight_attrs(w2_smooth_scale, extra_weight_attrs)

    def process_weights_after_loading(self, layer: torch.nn.Module):
        if not hasattr(layer, "w13_smooth_scale"):
            layer.w13_smooth_scale = None
            layer.w2_smooth_scale = None
            layer.w13_smooth_scale_total = None
        if layer.w13_smooth_scale is not None:
            w13_smooth_scale = layer.w13_smooth_scale
            # w13_smooth_scale cast to torch.bfloat16 for npu_dynamic_quant and init_routing
            layer.w13_smooth_scale = Parameter(
                w13_smooth_scale.data[self.expert_idx_min : self.expert_idx_max, :].to(torch.float32), requires_grad=False
            )
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
            x, dynamic_quant = torch_npu.npu_dynamic_quant(x, smooth_scales=layer.w13_smooth_scale, group_index=group_index)
        out = torch_npu.npu_grouped_matmul(
            [x],
            [w1],
            bias=None,
            group_list=expert_tokens,
            split_item=3,  # 3 means combine result tokens belong to diff expert to a single tensor
            output_dtype=torch.int32,
            group_type=0,  # 0 means expand_x is grouped in m-axis (1 for k-axis is used in training backward)
            group_list_type=1,  # 1 means group_list[i] is count for expert i (count mode), otherwise 0 is comsum mode
        )[0]
        dispose_tensor(x)

        gate_up_proj, pertoken_scale = torch_npu.npu_dequant_swiglu_quant(
            out,
            weight_scale=layer.w13_weight_scale.to(torch.float),
            group_index=expert_tokens,
            activation_scale=dynamic_quant,
            activate_left=True,
            quant_mode=1,
            quant_scale=layer.w2_smooth_scale,
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
            group_list_type=1,
        )[0]

        return out


def unpack_from_int32(
    weight: torch.Tensor,
    shape: torch.Size,
    num_bits: int,
    packed_dim: int = 1,
) -> torch.Tensor:
    """Unpacks quantized weights from int32 format back to original bits.

    :param weight: The packed int32 tensor containing quantized weights
    :param shape: Original shape to restore, defaults to None
    :param num_bits: The number of bits used for quantization (<= 8)
    :param packed_dim: Dimension along which weights are packed (0 or 1), defaults to 1
    :return: Unpacked tensor with int8 dtype after applying offset correction
    """
    assert weight.dtype == torch.int32, f"Expecting `weight.dtype` is torch.int32 but got {weight.dtype}."
    assert num_bits <= 8, f"Expecting `num_bits` should not be larger than 8 but got {num_bits}."

    pack_factor = 32 // num_bits
    mask = (1 << num_bits) - 1

    if packed_dim == 1:
        unpacked_weight = torch.zeros(
            (weight.shape[0], weight.shape[1] * pack_factor),
            device=weight.device,
            dtype=torch.int32,
        )
        for i in range(pack_factor):
            unpacked_weight[:, i::pack_factor] = (weight >> (num_bits * i)) & mask
        original_row_size = int(shape[1])
        unpacked_weight = unpacked_weight[:, :original_row_size]
    else:
        unpacked_weight = torch.zeros(
            (weight.shape[0] * pack_factor, weight.shape[1]),
            device=weight.device,
            dtype=torch.int32,
        )
        for i in range(pack_factor):
            unpacked_weight[i::pack_factor, :] = (weight >> (num_bits * i)) & mask
        original_row_size = int(shape[0])
        unpacked_weight = unpacked_weight[:original_row_size, :]

    offset = pow(2, num_bits) // 2
    unpacked_weight = (unpacked_weight - offset).to(torch.int8)

    return unpacked_weight


def pack_to_int32(weight: torch.Tensor) -> torch.Tensor:
    """Packs quantized weights into int32 format for storage.

    :param weight: The 3D tensor to pack, must be int8 or int32 dtype
    :return: Packed tensor with int32 dtype optimized for storage
    """
    assert weight.dim() == 3, f"Expecting `weight.dim()` is 3 ([e, n, k] or [e, k, n]) but got {weight.dim()}."
    assert weight.dtype in [torch.int8, torch.int32], f"Expecting `weight.dtype` is torch.int8 or torch.int32 bug got {weight.dtype}."

    if weight.dtype == torch.int32:
        assert weight.shape[-1] % 8 == 0, "the last dim of weight needs to be divided by 8."
        packed_weight = torch_npu.npu_convert_weight_to_int4pack(weight.flatten(0, 1))
        packed_weight = packed_weight.view(weight.shape[0], weight.shape[1], -1)
    else:
        assert weight.shape[-1] % 4 == 0, "the last dim of weight needs to be divided by 4."
        packed_weight = weight.view(torch.int32).contiguous()

    return packed_weight


class NpuCompressedTensorsW4A16MoEMethod(NpuCompressedTensorsMoEMethod):
    """FusedMoE method for Ascend W4A16."""

    def __init__(self, quant_config: "CompressedTensorsConfig") -> None:
        self.transpose_weight = True
        self.quant_config = quant_config
        self.weight_quant = self.quant_config.target_scheme_map["Linear"].get("weights")
        self.input_quant = self.quant_config.target_scheme_map["Linear"].get("input_activations")
        self.output_quant = self.quant_config.target_scheme_map["Linear"].get("output_activations")
        self.type = "int"

        self.group_size = self.weight_quant.group_size
        self.num_bits = self.weight_quant.num_bits
        self.pack_factor = 8  # pack 8 of torch.int4 tensors to torch.int32
        assert self.num_bits == 4
        assert self.input_quant is None and self.output_quant is None
        assert self.group_size > 0

    def get_weight(
        self,
        num_experts: int,
        intermediate_size_per_partition: int,
        hidden_sizes: int,
        params_dtype: torch.dtype,
    ) -> dict[str, Any]:
        assert intermediate_size_per_partition % self.pack_factor == 0, (
            f"Expecting `intermediate_size_per_partition` {intermediate_size_per_partition} can be divided by `pack_factor` {self.pack_factor}"
        )
        assert hidden_sizes % self.pack_factor == 0, f"Expecting `hidden_sizes` {hidden_sizes} can be divided by `pack_factor` {self.pack_factor}"

        param_dict = {}

        param_dict["w13_weight_packed"] = torch.zeros(
            num_experts, 2 * intermediate_size_per_partition, hidden_sizes // self.pack_factor, dtype=torch.int32
        )
        param_dict["w2_weight_packed"] = torch.zeros(
            num_experts, hidden_sizes, intermediate_size_per_partition // self.pack_factor, dtype=torch.int32
        )

        return param_dict

    def get_dynamic_quant_param(
        self,
        num_experts: int,
        intermediate_size_per_partition: int,
        hidden_sizes: int,
        params_dtype: torch.dtype,
    ) -> dict[str, Any]:
        assert intermediate_size_per_partition % self.group_size == 0, (
            f"Expecting `intermediate_size_per_partition` {intermediate_size_per_partition} can be divided by `group_size` {self.group_size}"
        )
        assert hidden_sizes % self.group_size == 0, f"Expecting `hidden_sizes` {hidden_sizes} can be divided by `group_size` {self.group_size}"

        param_dict = {}

        param_dict["w13_weight_scale"] = torch.zeros(
            num_experts, 2 * intermediate_size_per_partition, hidden_sizes // self.group_size, dtype=torch.bfloat16
        )
        param_dict["w2_weight_scale"] = torch.zeros(
            num_experts, hidden_sizes, intermediate_size_per_partition // self.group_size, dtype=torch.bfloat16
        )
        param_dict["w13_weight_shape"] = torch.zeros(num_experts, 2, dtype=torch.int32)
        param_dict["w2_weight_shape"] = torch.zeros(num_experts, 2, dtype=torch.int32)
        param_dict["w13_weight_offset"] = torch.zeros(
            num_experts, 2 * intermediate_size_per_partition, hidden_sizes // self.group_size, dtype=torch.bfloat16
        )
        param_dict["w2_weight_offset"] = torch.zeros(
            num_experts, hidden_sizes, intermediate_size_per_partition // self.group_size, dtype=torch.bfloat16
        )

        return param_dict

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        weight_param = self.get_weight(num_experts, intermediate_size, hidden_size, params_dtype)
        for param_key, param_value in weight_param.items():
            param = torch.nn.Parameter(param_value, requires_grad=False)
            layer.register_parameter(param_key, param)
            set_weight_attrs(param, extra_weight_attrs)

        extra_weight_attrs.update({"quant_method": "channel"})
        per_group_param = ["weight_scale_second", "weight_offset_second", "scale_bias"] + (
            ["weight_scale", "weight_offset"] if self.group_size > 0 else []
        )
        dynamic_quant_param = self.get_dynamic_quant_param(num_experts, intermediate_size, hidden_size, params_dtype)
        for param_key, param_value in dynamic_quant_param.items():
            param = torch.nn.Parameter(param_value, requires_grad=False)
            layer.register_parameter(param_key, param)
            set_weight_attrs(param, extra_weight_attrs)
            if any(fields in param_key for fields in per_group_param):
                param.quant_method = "group"

    def apply(self, layer: torch.nn.Module, x, tokens_per_expert) -> torch.Tensor:
        hidden_states = torch_npu.npu_grouped_matmul(
            x=[x],
            weight=[layer.w13_weight_packed],
            antiquant_scale=[layer.w13_weight_scale],
            antiquant_offset=[layer.w13_weight_offset],
            split_item=3,
            group_list_type=1,
            group_type=0,
            group_list=tokens_per_expert,
            output_dtype=layer.w2_weight_scale.dtype,
        )[0]
        dispose_tensor(x)
        hidden_states = torch_npu.npu_swiglu(hidden_states)
        hidden_states = torch_npu.npu_grouped_matmul(
            x=[hidden_states],
            weight=[layer.w2_weight_packed],
            antiquant_scale=[layer.w2_weight_scale],
            antiquant_offset=[layer.w2_weight_offset],
            split_item=3,
            group_list_type=1,
            group_type=0,
            group_list=tokens_per_expert,
            output_dtype=layer.w2_weight_scale.dtype,  # why not x.dtype
        )[0]
        return hidden_states

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if self.transpose_weight:
            w13_shape = layer.w13_weight_packed.data.shape
            w2_shape = layer.w2_weight_packed.data.shape
            unpacked_w13_weight = (
                unpack_from_int32(
                    layer.w13_weight_packed.data.flatten(0, 1),
                    torch.Size([w13_shape[0] * w13_shape[1], w13_shape[2] * self.pack_factor]),
                    self.num_bits,
                )
                .view(w13_shape[0], w13_shape[1], -1)
                .transpose(1, 2)
                .contiguous()
                .int()
            )
            unpacked_w2_weight = (
                unpack_from_int32(
                    layer.w2_weight_packed.data.flatten(0, 1),
                    torch.Size([w2_shape[0] * w2_shape[1], w2_shape[2] * self.pack_factor]),
                    self.num_bits,
                )
                .view(w2_shape[0], w2_shape[1], -1)
                .transpose(1, 2)
                .contiguous()
                .int()
            )
            layer.w13_weight_packed.data = pack_to_int32(unpacked_w13_weight)
            layer.w2_weight_packed.data = pack_to_int32(unpacked_w2_weight)

            layer.w13_weight_scale.data = layer.w13_weight_scale.data.transpose(1, 2).contiguous()
            layer.w2_weight_scale.data = layer.w2_weight_scale.data.transpose(1, 2).contiguous()

            layer.w13_weight_offset.data = layer.w13_weight_offset.data.transpose(1, 2).contiguous()
            layer.w2_weight_offset.data = layer.w2_weight_offset.data.transpose(1, 2).contiguous()
