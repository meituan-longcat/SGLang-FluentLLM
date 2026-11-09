import math
import os
from typing import Optional

import flash_ops  # noqa: F401  # Registers torch.ops.custom operators.
import torch
import torch_npu

from sglang.srt.distributed import (
    get_attn_tp_group,
    get_expert_and_tensor_model_parallel_rank,
    get_expert_and_tensor_model_parallel_world_size,
    get_tensor_model_parallel_rank,
)
from sglang.srt.env import global_server_args_dict
from sglang.srt.layers.moe.layouts.compress_tensor_load_functions import (
    _load_model_weight_or_group_weight_scale,
    _load_per_channel_weight_scale,
    _load_per_tensor_weight_scale,
    _load_single_value,
)
from sglang.srt.layers.moe.npu_moe.token_dispatcher import TokensDispatcherAgRs, TokensDispatcherAll2All, TokensDispatcherPA2A
from sglang.srt.layers.quantization import FusedMoeWeightScaleSupported
from sglang.srt.layers.quantization.base_config import (
    FusedMoEMethodBase,
    QuantizationConfig,
    QuantizeMethodBase,
)
from sglang.srt.layers.quantization.npu_compressed_tensors.compressed_tensors_moe import InvDecomposeTransMatrix, NpuCompressedTensorsW4A16MoEMethod
from sglang.srt.managers.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.managers.eplb.expert_location import get_global_expert_location_metadata
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.ops.custom_op import CustomOp
from sglang.srt.utils import dispose_tensor, get_colorful_logger, set_weight_attrs

logger = get_colorful_logger(__name__)

UNQUANT_MODE = 0
STATIC_QUANT_MODE = 1
DYNAMIC_QUANT_MODE = 2


def get_decompose_dim(n: int) -> tuple[int, int]:
    """Return the closest positive factor pair in ascending order."""
    if n <= 0:
        raise ValueError(f"Dimension must be positive, got {n}")

    for left in range(math.isqrt(n), 1, -1):
        if n % left == 0:
            return left, n // left

    return 1, n


class NpuUnquantizedFusedMoEMethod(FusedMoEMethodBase, CustomOp):
    LAST_SEQ_LEN = None
    BEST_EXPERT_TOKENS = None

    def __init__(self):
        super().__init__()
        self.initialized = False
        self.transpose = False

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        # Fused gate_up_proj (column parallel)
        w13_weight = torch.nn.Parameter(
            torch.empty(num_experts, 2 * intermediate_size, hidden_size, dtype=params_dtype),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        # down_proj (row parallel)
        w2_weight = torch.nn.Parameter(
            torch.empty(num_experts, hidden_size, intermediate_size, dtype=params_dtype),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if self.initialized:
            return
        if self.transpose:
            layer.w13_weight.data = torch_npu.npu_format_cast(layer.w13_weight.data, 29)
            layer.w2_weight.data = torch_npu.npu_format_cast(layer.w2_weight.data, 29)
        elif torch.npu.mem_get_info(layer.w13_weight.device)[0] < layer.w13_weight.numel() * layer.w13_weight.element_size():
            # BF16 EP8 weights can leave less HBM than a single expert tensor.
            # Stage this layer on Host during startup so ND and NZ fit together.
            device = layer.w13_weight.device
            dtype = layer.w13_weight.dtype
            w13_host = layer.w13_weight.data.cpu()
            w2_host = layer.w2_weight.data.cpu()
            layer.w13_weight.data = torch.empty(0, device=device, dtype=dtype)
            layer.w2_weight.data = torch.empty(0, device=device, dtype=dtype)
            torch.npu.empty_cache()
            w13_device = w13_host.transpose(-1, -2).contiguous().to(device)
            layer.w13_weight.data = torch_npu.npu_format_cast(w13_device, 29)
            del w13_device, w13_host
            w2_device = w2_host.transpose(-1, -2).contiguous().to(device)
            layer.w2_weight.data = torch_npu.npu_format_cast(w2_device, 29)
            del w2_device, w2_host
        else:
            # Release the original ND storage before allocating the NZ result.
            # Keeping all three full expert tensors alive can exceed 910B HBM.
            layer.w13_weight.data = layer.w13_weight.data.transpose(-1, -2).contiguous()
            layer.w13_weight.data = torch_npu.npu_format_cast(layer.w13_weight.data, 29)
            layer.w2_weight.data = layer.w2_weight.data.transpose(-1, -2).contiguous()
            layer.w2_weight.data = torch_npu.npu_format_cast(layer.w2_weight.data, 29)
        self.n_routed_experts = len(layer.w13_weight)
        self.local_expert_indices_offset = get_tensor_model_parallel_rank() * self.n_routed_experts
        self.local_expert_indices = [self.local_expert_indices_offset + i for i in range(self.n_routed_experts)]
        self.initialized = True

    def apply(self, layer: torch.nn.Module, x, tokens_per_expert) -> torch.Tensor:
        if not self.initialized:
            raise RuntimeError("MoE weights must be initialized before apply().")
        # w1: [ffn_hidden_size, hidden_size], w2: [hidden_size, ffn_hidden_size/2]
        w1 = layer.w13_weight.transpose(-1, -2) if self.transpose else layer.w13_weight
        w2 = layer.w2_weight.transpose(-1, -2) if self.transpose else layer.w2_weight
        expert_tokens = tokens_per_expert.to(torch.int64)
        hidden_states = torch_npu.npu_grouped_matmul(
            [x],
            [w1],
            bias=None,
            group_list=expert_tokens,
            split_item=3,  # 3 means combine result tokens belong to diff expert to a single tensor
            output_dtype=torch.bfloat16,
            group_type=0,  # 0 means expand_x is grouped in m-axis (1 for k-axis is used in training backward)
            group_list_type=1,  # 1 means group_list[i] is count for expert i (count mode), otherwise 0 is comsum mode
        )[0]
        dispose_tensor(x)
        hidden_states = torch.ops.custom.mlp_split_swiglu(
            hidden_states, tokens_per_expert.to(torch.int32), local_exp_start=0, local_exp_end=layer.num_local_experts
        )
        hidden_states = torch.ops.custom.npu_grouped_matmul(
            [hidden_states], [w2], bias=None, group_list=expert_tokens, split_item=3, output_dtype=torch.bfloat16, group_type=0, group_list_type=1
        )[0]
        return hidden_states


_RANDOM_ROUTER = None
if _RANDOM_ROUTER is None:
    _RANDOM_ROUTER = os.getenv("RANDOM_ROUTER", "0") == "1"




class NpuEPMoE(torch.nn.Module):
    def __init__(
        self,
        layer_id: int,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        num_experts: int,
        zero_expert_num: int = 0,
        zero_expert_type=None,
        reduce_results: bool = True,
        params_dtype: Optional[torch.dtype] = None,
        quant_config: Optional[QuantizationConfig] = None,
        use_presharded_weights: bool = False,
        prefix: str = "",
        moe_chunked_prefill_size=-1,
    ):
        super().__init__()
        if params_dtype is None:
            params_dtype = torch.get_default_dtype()

        self.layer_id = layer_id
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.zero_expert_num = zero_expert_num
        self.zero_expert_type = zero_expert_type
        self.has_identity_zero_expert = self.zero_expert_num is not None and self.zero_expert_num > 0 and self.zero_expert_type == "identity"
        if self.zero_expert_num and not self.has_identity_zero_expert:
            raise ValueError("NPU MoE only supports zero_expert_type='identity' when zero_expert_num is positive.")
        self.reduce_results = reduce_results
        self.use_presharded_weights = use_presharded_weights
        # cac = available_npu_mem * 1024 * 1024 * 1024 / (self.hidden_size * min(self.topk, local_expert_num) * gmm_1_out_dtype_size * 3))
        self.moe_chunked_prefill_size = moe_chunked_prefill_size

        self.moe_ep_size = get_expert_and_tensor_model_parallel_world_size()
        self.moe_ep_rank = get_expert_and_tensor_model_parallel_rank()

        self.num_experts += global_server_args_dict["ep_num_redundant_experts"]
        if self.num_experts % self.moe_ep_size != 0:
            raise ValueError(f"num_experts ({self.num_experts}) must be divisible by moe_ep_size ({self.moe_ep_size}).")
        self.num_local_experts = self.num_experts // self.moe_ep_size

        self.quant_config = quant_config
        self.quant_method: Optional[FusedMoEMethodBase] = None
        if quant_config is None:
            self.quant_method: Optional[QuantizeMethodBase] = NpuUnquantizedFusedMoEMethod()
        else:
            self.quant_method = quant_config.get_quant_method(self, prefix=prefix)
            self.quant_method.expert_idx_min = self.moe_ep_rank * self.num_local_experts
            self.quant_method.expert_idx_max = self.quant_method.expert_idx_min + self.num_local_experts

        self.weight_quant = None
        self.input_quant = None
        self.is_dynamic_w4a4 = self._is_dynamic_w4a4_quantization()
        if self.is_dynamic_w4a4:
            self._init_w4a4_auxiliary_modules()

        self.quant_method.create_weights(
            layer=self,
            num_experts=self.num_local_experts,
            hidden_size=hidden_size,
            intermediate_size=self.intermediate_size,
            params_dtype=params_dtype,
            weight_loader=self.weight_loader,
            with_bias=False,
        )
        is_dispatcher_quant = False if isinstance(self.quant_method, NpuCompressedTensorsW4A16MoEMethod) else self.quant_config is not None
        self.dispatcher_all2all = TokensDispatcherAll2All(
            self, self.num_experts, self.zero_expert_num, ep_size=self.moe_ep_size, ep_rank=self.moe_ep_rank, quant=is_dispatcher_quant
        )
        self.dispatcher_p_all2all = TokensDispatcherPA2A(
            self, self.num_experts, self.zero_expert_num, ep_size=self.moe_ep_size, ep_rank=self.moe_ep_rank, quant=is_dispatcher_quant
        )
        self.dispatcher_ag_rs = TokensDispatcherAgRs(
            self, self.num_experts, self.zero_expert_num, ep_size=self.moe_ep_size, ep_rank=self.moe_ep_rank, quant=is_dispatcher_quant
        )
        if self.moe_chunked_prefill_size != -1:
            logger.info(f"prefill use all2all: {self.moe_chunked_prefill_size=}")

    def _is_dynamic_w4a4_quantization(self) -> bool:
        if self.quant_config is None:
            return False

        if getattr(self.quant_method, "is_dynamic_w4a4", False):
            return True

        scheme_map = getattr(self.quant_config, "target_scheme_map", {}) or {}
        scheme = scheme_map.get("Moe", scheme_map.get("Linear"))
        if scheme is None:
            return False

        self.weight_quant = scheme.get("weights")
        self.input_quant = scheme.get("input_activations")
        is_dynamic_token_w4a4 = getattr(self.quant_config, "_is_dynamic_token_w4a4", None)
        return (
            self.weight_quant is not None
            and self.input_quant is not None
            and callable(is_dynamic_token_w4a4)
            and is_dynamic_token_w4a4(self.weight_quant, self.input_quant)
        )

    def _init_w4a4_auxiliary_modules(self) -> None:
        (
            self.w1_trans_matrix_left_shape,
            self.w1_trans_matrix_right_shape,
        ) = get_decompose_dim(self.hidden_size)
        (
            self.w2_trans_matrix_left_shape,
            self.w2_trans_matrix_right_shape,
        ) = get_decompose_dim(self.intermediate_size)
        self.w1_trans = InvDecomposeTransMatrix(
            self.w1_trans_matrix_left_shape,
            self.w1_trans_matrix_right_shape,
            False,
        )
        self.w2_trans = InvDecomposeTransMatrix(
            self.w2_trans_matrix_left_shape,
            self.w2_trans_matrix_right_shape,
            False,
        )
        self.act_quantizer_w1 = self._create_clip_quantizer()
        self.act_quantizer_w2 = self._create_clip_quantizer()

        for module in (
            self.w1_trans,
            self.w2_trans,
            self.act_quantizer_w1,
            self.act_quantizer_w2,
        ):
            for param in module.parameters():
                set_weight_attrs(param, {"exclude_from_eplb": True})

    @staticmethod
    def _create_clip_quantizer() -> torch.nn.Module:
        quantizer = torch.nn.Module()
        clip_factor_a = torch.nn.Parameter(
            torch.empty(1, device=torch.npu.current_device(), dtype=torch.float32),
            requires_grad=False,
        )
        quantizer.register_parameter("clip_factor_a", clip_factor_a)
        return quantizer

    def _prepare_dispatch_input(self, hidden_states: torch.Tensor):
        if not (global_server_args_dict.get("npu_enable_int4_moe_comm", False) and self.is_dynamic_w4a4):
            return hidden_states

        quantize_moe_input = getattr(self.quant_method, "quantize_moe_input", None)
        if not callable(quantize_moe_input):
            raise RuntimeError("Dynamic W4A4 quantization method must implement quantize_moe_input() when INT4 MoE communication is enabled.")
        return quantize_moe_input(self, hidden_states)

    def zero_expert(self, hidden_states, original_top_experts=None, expert_weights=None):
        if not self.has_identity_zero_expert:
            raise RuntimeError("Identity zero experts are not enabled.")
        if original_top_experts is None or expert_weights is None:
            raise ValueError("original_top_experts and expert_weights are required for zero experts.")

        expert_weights = expert_weights.to(hidden_states.dtype)
        hidden_size = hidden_states.shape[-1]
        identity_output = hidden_states.view(-1, hidden_size)
        zero_expert_mask = torch.logical_and(
            original_top_experts >= self.num_experts,
            original_top_experts < self.num_experts + self.zero_expert_num,
        )
        zero_expert_weights = expert_weights * zero_expert_mask
        expert_weights_sum = torch.sum(zero_expert_weights, dim=-1, keepdim=True)
        return identity_output * expert_weights_sum

    def forward_all2all(self, hidden_states: torch.Tensor, expert_topks, expert_weights, dispatch_input=None, active_mask=None):
        if self.quant_method is None:
            raise RuntimeError("MoE quantization method is not initialized.")
        # Combine still needs the original BF16 hidden states as ori_x.
        if dispatch_input is None:
            dispatch_input = self._prepare_dispatch_input(hidden_states)
        mask_kwargs = {} if active_mask is None else {"active_mask": active_mask}
        (expand_x, expert_token_count, dispatch_output) = self.dispatcher_all2all.dispatch(
            dispatch_input, expert_topks, expert_weights, **mask_kwargs
        )
        get_global_expert_distribution_recorder().on_local_expert_counts(expert_token_count)
        combine_input = self.quant_method.apply(self, expand_x, expert_token_count)
        result = self.dispatcher_all2all.combine(
            combine_input, expert_topks, expert_weights, hidden_states, dispatch_output, **mask_kwargs
        )
        # Masked kernels may leave padded rows unwritten. Define them before all_gather.
        if active_mask is not None:
            result = torch.where(active_mask[:, None], result, torch.zeros_like(result))
        return result

    def forward_ag_rs(self, hidden_states: torch.Tensor, expert_topks, expert_weights, sp_num_tokens):
        # dispatcher.precomm -> dispatcher.dispatch -> ffn -> dispatcher.combine -> dispatcher.post_comm
        if self.quant_method is None:
            raise RuntimeError("MoE quantization method is not initialized.")
        (global_expert_topks, global_expert_weights) = self.dispatcher_ag_rs.pre_comm([expert_topks, expert_weights], sp_num_tokens=sp_num_tokens)
        (expand_x, expert_token_count, dispatch_output) = self.dispatcher_ag_rs.dispatch(
            hidden_states, global_expert_topks, skip_comm=False, sp_num_tokens=sp_num_tokens
        )
        combine_input = self.quant_method.apply(self, expand_x, expert_token_count)
        return self.dispatcher_ag_rs.combine(combine_input, global_expert_topks, global_expert_weights, dispatch_output, sp_token_num=sp_num_tokens)

    def forward_p_all2all(self, hidden_states: torch.Tensor, expert_topks, expert_weights, global_sp_num_tokens):
        global_chunk_size = self.moe_chunked_prefill_size
        total_sp_num_tokens = sum(global_sp_num_tokens)
        loop = int((total_sp_num_tokens + global_chunk_size - 1) // global_chunk_size)
        chunk_size = int(global_sp_num_tokens[self.moe_ep_rank] + loop - 1) // loop
        moe_output_chunks = []
        for i in range(loop):
            expert_weights_chunk = expert_weights[i * chunk_size : (i + 1) * chunk_size]
            top_experts_chunk = expert_topks[i * chunk_size : (i + 1) * chunk_size]
            if isinstance(hidden_states, (list, tuple)):
                moe_chunk = (hidden_states[0][i * chunk_size : (i + 1) * chunk_size], hidden_states[1][i * chunk_size : (i + 1) * chunk_size])
            else:
                moe_chunk = hidden_states[i * chunk_size : (i + 1) * chunk_size]
            expand_x, expert_token_count, dispatch_output = self.dispatcher_p_all2all.dispatch(moe_chunk, top_experts_chunk)
            combine_input = self.quant_method.apply(self, expand_x, expert_token_count)
            moe_output = self.dispatcher_p_all2all.combine(combine_input, top_experts_chunk, expert_weights_chunk, dispatch_output)
            moe_output_chunks.append(moe_output)
        return torch.concat(moe_output_chunks, dim=0) if len(moe_output_chunks) > 1 else moe_output_chunks[0]

    def forward(self, hidden_states: torch.Tensor, expert_topks, expert_weights, forward_batch: ForwardBatch, dispatch_input=None):
        if self.quant_method is None:
            raise RuntimeError("MoE quantization method is not initialized.")
        if forward_batch.all_decode_or_idle:
            # A verify batch can be reused for draft execution with a different layout.
            active_mask = (
                forward_batch.moe_valid_token_mask
                if forward_batch.can_run_with_graph and forward_batch.forward_mode.is_target_verify()
                else None
            )
            if active_mask is not None:
                tp = get_attn_tp_group()
                sp_len = hidden_states.shape[0]
                # Match LongCat forward_decode_3b: pad to TP, then take a contiguous slice.
                missing = sp_len * tp.world_size - active_mask.shape[0]
                assert 0 <= missing < tp.world_size
                active_mask = torch.cat((active_mask, active_mask.new_zeros((missing,))))
                active_mask = active_mask[tp.rank_in_group * sp_len : (tp.rank_in_group + 1) * sp_len]
            return self.forward_all2all(
                hidden_states, expert_topks, expert_weights, dispatch_input=dispatch_input, active_mask=active_mask
            )
        else:
            if dispatch_input is not None:
                raise ValueError("Pre-quantized dispatch_input is only supported by the decode AllToAll path.")
            if self.moe_chunked_prefill_size == -1:
                moe_output = self.forward_ag_rs(hidden_states, expert_topks, expert_weights, forward_batch.global_sp_num_tokens)
            else:
                moe_output = self.forward_p_all2all(hidden_states, expert_topks, expert_weights, forward_batch.global_sp_num_tokens)
            if self.has_identity_zero_expert:
                moe_output_zero = self.zero_expert(hidden_states, expert_topks, expert_weights)
                return moe_output + moe_output_zero
            else:
                return moe_output

    def weight_loader(self, param: torch.nn.Parameter, loaded_weight: torch.Tensor, weight_name: str, shard_id: str, expert_id: int) -> None:
        physical_expert_ids = get_global_expert_location_metadata().logical_to_all_physical(self.layer_id, expert_id)
        for physical_expert_id in physical_expert_ids:
            self._weight_loader_physical(
                param=param,
                loaded_weight=loaded_weight,
                weight_name=weight_name,
                shard_id=shard_id,
                expert_id=physical_expert_id,
            )

    def _weight_loader_physical(self, param: torch.nn.Parameter, loaded_weight: torch.Tensor, weight_name: str, shard_id: str, expert_id: int):
        expert_id_min = self.moe_ep_rank * self.num_local_experts
        expert_id_max = expert_id_min + self.num_local_experts
        if "smooth_scale" in weight_name:
            if "w13_smooth_scale" in weight_name:
                param.data[expert_id].copy_(loaded_weight)  # no tp for moe
            elif expert_id_min <= expert_id < expert_id_max:
                param.data[expert_id - expert_id_min].copy_(loaded_weight)  # no tp for moe
            return
        if expert_id < expert_id_min or expert_id >= expert_id_max:
            return

        tp_rank = 0
        expert_id -= expert_id_min

        if shard_id not in ("w1", "w2", "w3"):
            raise ValueError(f"shard_id must be ['w1','w2','w3'] but got {shard_id}.")

        WEIGHT_SCALE_SUPPORTED = [e.value for e in FusedMoeWeightScaleSupported]
        # Fetch the dim to shard the parameter/loaded weight
        # based on the shard id. This will be whatever
        # dimension intermediate_size is used.
        SHARD_ID_TO_SHARDED_DIM = {"w1": 0, "w2": 1, "w3": 0}

        expert_data = param.data[expert_id]

        # is_transposed: if the dim to shard the weight
        # should be flipped. Required by GPTQ, compressed-tensors
        # should be whatever dimension intermediate_size is
        is_transposed = getattr(param, "is_transposed", False)
        shard_dim = SHARD_ID_TO_SHARDED_DIM[shard_id]
        if is_transposed:
            shard_dim = ~shard_dim

        # Case weight scales and zero_points
        if "scale" in weight_name or "zero" in weight_name or "offset" in weight_name:
            # load the weight scales and zp based on the quantization scheme
            # supported weight scales/zp can be found in
            # FusedMoeWeightScaleSupported
            # TODO @dsikka: once hardened, refactor to use vLLM Parameters
            # specific to each case
            quant_method = getattr(param, "quant_method", None)
            if quant_method == FusedMoeWeightScaleSupported.CHANNEL.value:
                _load_per_channel_weight_scale(
                    shard_id=shard_id,
                    shard_dim=shard_dim,
                    loaded_weight=loaded_weight.squeeze(-1),
                    expert_data=expert_data,
                    tp_rank=tp_rank,
                    use_presharded_weights=self.use_presharded_weights,
                )
            elif quant_method == FusedMoeWeightScaleSupported.GROUP.value:
                _load_model_weight_or_group_weight_scale(
                    shard_id=shard_id,
                    shard_dim=shard_dim,
                    loaded_weight=loaded_weight,
                    expert_data=expert_data,
                    tp_rank=tp_rank,
                    use_presharded_weights=self.use_presharded_weights,
                )
            elif quant_method == FusedMoeWeightScaleSupported.TENSOR.value:
                _load_per_tensor_weight_scale(shard_id=shard_id, param=param, loaded_weight=loaded_weight, local_expert_id=expert_id)
            else:
                raise ValueError(f"quant method must be one of {WEIGHT_SCALE_SUPPORTED}")
            return

        # Case weight_shape
        if "weight_shape" in weight_name:
            # only required by compressed-tensors
            _load_single_value(param=param, loaded_weight=loaded_weight, local_expert_id=expert_id)
            return

        # Case model weights
        if "weight" in weight_name:
            _load_model_weight_or_group_weight_scale(
                shard_id=shard_id,
                shard_dim=shard_dim,
                loaded_weight=loaded_weight,
                expert_data=expert_data,
                tp_rank=tp_rank,
                use_presharded_weights=self.use_presharded_weights,
            )
            return

    @classmethod
    def make_expert_params_mapping(
        cls,
        ckpt_gate_proj_name: str,
        ckpt_down_proj_name: str,
        ckpt_up_proj_name: str,
        num_experts: int,
    ) -> list[tuple[str, str, int, str]]:

        return [
            # (param_name, weight_name, expert_id, shard_id)
            (
                ("experts.w13_" if weight_name in [ckpt_gate_proj_name, ckpt_up_proj_name] else "experts.w2_"),
                f"experts.{expert_id}.{weight_name}.",
                expert_id,
                shard_id,
            )
            for expert_id in range(num_experts)
            for shard_id, weight_name in [
                ("w1", ckpt_gate_proj_name),
                ("w2", ckpt_down_proj_name),
                ("w3", ckpt_up_proj_name),
            ]
        ]
