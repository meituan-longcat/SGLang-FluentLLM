from enum import Enum
import torch

class FusedMoeWeightScaleSupported(Enum):
    TENSOR = "tensor"
    CHANNEL = "channel"
    GROUP = "group"
    BLOCK = "block"


def _load_per_tensor_weight_scale(
    shard_id: str,
    param: torch.nn.Parameter,
    loaded_weight: torch.Tensor,
    local_expert_id: int,
):
    param_data = param.data
    # for per tensor weight quantization
    if shard_id in ("w1", "w3"):
        # We have to keep the weight scales of w1 and w3 because
        # we need to re-quantize w1/w3 weights after weight loading.
        idx = 0 if shard_id == "w1" else 1
        param_data[local_expert_id][idx] = loaded_weight
    # If we are in the row parallel case (down_proj)
    elif shard_id == "w2":
        param_data[local_expert_id] = loaded_weight

def _load_model_weight_or_group_weight_scale(
    shard_dim: int,
    expert_data: torch.Tensor,
    shard_id: str,
    loaded_weight: torch.tensor,
    tp_rank: int,
    use_presharded_weights: bool = False,
):
    # Load grouped weight scales for group quantization
    # or model weights
    if shard_id == "w2":
        _load_w2(
            shard_id=shard_id,
            shard_dim=shard_dim,
            loaded_weight=loaded_weight,
            expert_data=expert_data,
            tp_rank=tp_rank,
            use_presharded_weights=use_presharded_weights,
        )
    elif shard_id in ("w1", "w3"):
        _load_w13(
            shard_id=shard_id,
            shard_dim=shard_dim,
            loaded_weight=loaded_weight,
            expert_data=expert_data,
            tp_rank=tp_rank,
            use_presharded_weights=use_presharded_weights,
        )

def _load_per_channel_weight_scale(
    expert_data: torch.Tensor,
    shard_dim: int,
    shard_id: str,
    loaded_weight: torch.tensor,
    tp_rank: int,
    use_presharded_weights: bool = False,
):
    # for per channel weight quantization
    if shard_id == "w2":
        expert_data.copy_(loaded_weight)
    elif shard_id in ("w1", "w3"):
        _load_w13(
            shard_id=shard_id,
            shard_dim=shard_dim,
            loaded_weight=loaded_weight,
            expert_data=expert_data,
            tp_rank=tp_rank,
            use_presharded_weights=use_presharded_weights,
        )

def _load_w13(
    expert_data: torch.Tensor,
    shard_dim: int,
    shard_id: str,
    loaded_weight: torch.tensor,
    tp_rank: int,
    use_presharded_weights: bool = False,
):

    # Index the loaded weight for tp sharding.
    # gate_up_proj: "MergedColumnParallel", so tp sharding on output_dim
    shard_size = expert_data.shape[shard_dim] // 2

    if not use_presharded_weights:
        loaded_weight = loaded_weight.narrow(
            shard_dim, shard_size * tp_rank, shard_size
        )

    # Narrow parameter and load.
    # w1, gate_proj: Load into first logical weight of w13.
    if shard_id == "w1":
        expert_data = expert_data.narrow(shard_dim, 0, shard_size)
    # w3, up_proj: Load into second logical weight of w13.
    else:
        assert shard_id == "w3"
        expert_data = expert_data.narrow(shard_dim, shard_size, shard_size)
    expert_data.copy_(loaded_weight)

def _load_w2(
    expert_data: torch.Tensor,
    shard_dim: int,
    shard_id: str,
    loaded_weight: torch.tensor,
    tp_rank: int,
    use_presharded_weights: bool = False,
):

    # Index the loaded weight for tp sharding.
    # down_proj: "RowParallel" so tp sharding on input_dim
    # Narrow parameter and load.
    shard_size = expert_data.shape[shard_dim]

    if not use_presharded_weights:
        loaded_weight = loaded_weight.narrow(
            shard_dim, shard_size * tp_rank, shard_size
        )

    # w2, down_proj: Load into only logical weight of w2.
    expert_data.copy_(loaded_weight)

def _load_single_value(
    param: torch.nn.Parameter, loaded_weight: torch.Tensor, local_expert_id: int
):
    param_data = param.data

    # Input scales can be loaded directly and should be equal.
    param_data[local_expert_id] = loaded_weight

def _load_g_idx(
    shard_id: str,
    expert_data: torch.Tensor,
    shard_dim: int,
    loaded_weight: torch.tensor,
    tp_rank: int,
):

    if shard_id == "w2":
        _load_w2(
            shard_id=shard_id,
            shard_dim=shard_dim,
            loaded_weight=loaded_weight,
            expert_data=expert_data,
            tp_rank=tp_rank,
        )
    else:
        assert shard_id in ("w1", "w3")
        expert_data.copy_(loaded_weight)

def load_model_weight(
    param: torch.nn.Parameter,
    loaded_weight: torch.Tensor,
    weight_name: str,
    shard_id: str,
    local_expert_id: int,
    tp_rank: int
) -> None:
    # compressed-tensors checkpoints with packed weights are stored flipped
    # TODO (mgoin): check quant_method.quant_config.quant_format
    # against known CompressionFormat enum values that have this quality
    loaded_weight = loaded_weight.t().contiguous()

    if shard_id not in ("w1", "w2", "w3"):
        raise ValueError(
            f"shard_id must be ['w1','w2','w3'] but " f"got {shard_id}."
        )

    WEIGHT_SCALE_SUPPORTED = [e.value for e in FusedMoeWeightScaleSupported]
    # Fetch the dim to shard the parameter/loaded weight
    # based on the shard id. This will be whatever
    # dimension intermediate_size is used.
    SHARD_ID_TO_SHARDED_DIM = {"w1": 0, "w2": 1, "w3": 0}

    expert_data = param.data[local_expert_id]

    # is_transposed: if the dim to shard the weight
    # should be flipped. Required by GPTQ, compressed-tensors
    # should be whatever dimension intermediate_size is
    is_transposed = getattr(param, "is_transposed", False)
    shard_dim = SHARD_ID_TO_SHARDED_DIM[shard_id]
    if is_transposed:
        shard_dim = ~shard_dim

    # Case input scale: input_scale loading is only supported for fp8
    if "input_scale" in weight_name:
        # this is needed for compressed-tensors only
        loaded_weight = loaded_weight.to(param.data.device)

        if (
            param.data[local_expert_id] != 1
            and (param.data[local_expert_id] - loaded_weight).abs() > 1e-5
        ):
            raise ValueError(
                "input_scales of w1 and w3 of a layer "
                f"must be equal. But got {param.data[local_expert_id]} "
                f"vs. {loaded_weight}"
            )

        _load_single_value(
            param=param, loaded_weight=loaded_weight, local_expert_id=local_expert_id
        )
        return

    # Case g_idx
    if "g_idx" in weight_name:
        _load_g_idx(
            shard_dim=0,
            shard_id=shard_id,
            loaded_weight=loaded_weight,
            expert_data=expert_data,
            tp_rank=tp_rank,
        )
        return

    # Case weight scales and zero_points
    if "scale" in weight_name or "zero" in weight_name:
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
                loaded_weight=loaded_weight,
                expert_data=expert_data,
                tp_rank=tp_rank,
            )
        elif quant_method in [
            FusedMoeWeightScaleSupported.GROUP.value,
            FusedMoeWeightScaleSupported.BLOCK.value,
        ]:
            _load_model_weight_or_group_weight_scale(
                shard_id=shard_id,
                shard_dim=shard_dim,
                loaded_weight=loaded_weight,
                expert_data=expert_data,
                tp_rank=tp_rank,
            )
        elif quant_method == FusedMoeWeightScaleSupported.TENSOR.value:
            _load_per_tensor_weight_scale(
                shard_id=shard_id,
                param=param,
                loaded_weight=loaded_weight,
                local_expert_id=local_expert_id,
            )
        else:
            raise ValueError(
                f"quant method must be one of {WEIGHT_SCALE_SUPPORTED}"
            )
        return

    # Case weight_shape
    if "weight_shape" in weight_name:
        # only required by compressed-tensors
        _load_single_value(
            param=param, loaded_weight=loaded_weight, local_expert_id=local_expert_id
        )
        return

    # Case model weights
    if "weight" in weight_name:
        _load_model_weight_or_group_weight_scale(
            shard_id=shard_id,
            shard_dim=shard_dim,
            loaded_weight=loaded_weight,
            expert_data=expert_data,
            tp_rank=tp_rank,
        )
        return
