from functools import partial

import torch


def _load_w13(
    expert_data: torch.Tensor,
    loaded_weight: torch.Tensor,
    shard_id: str,
    shard_dim: int,
    tp_rank: int,
    is_bias: bool,
    use_presharded_weights: bool,
    do_transpose: bool,
    load_up_proj_weight_first: bool,
):
    # Index the loaded weight for tp sharding.
    # gate_up_proj: "MergedColumnParallel", so tp sharding on output_dim
    assert shard_id in {"w1", "w3", "w13"}

    if is_bias:
        # if this weight is a bias, the last dimension must be the sharded dimension
        shard_dim = -1

    if shard_id in {"w1", "w3"}:
        # non-fused version
        shard_size = expert_data.shape[shard_dim] // 2
    elif shard_id in {"w13"}:
        # fused version
        shard_size = expert_data.shape[shard_dim]
    else:
        raise NotImplementedError

    # Narrow parameter and load.
    # w1, gate_proj: Load into first logical weight of w13.
    # w3, up_proj: Load into second logical weight of w13.
    # trtllm cutlass kernel assumes differently
    switch_w13 = load_up_proj_weight_first
    if (switch_w13 and shard_id == "w1") or (not switch_w13 and shard_id == "w3"):
        start = shard_size
    else:
        start = 0

    if not use_presharded_weights:
        if not is_bias and do_transpose:
            # do not transpose for bias
            loaded_weight = loaded_weight.transpose(-2, -1)
        loaded_weight = loaded_weight.narrow(shard_dim, shard_size * tp_rank, shard_size)

    expert_data = expert_data.narrow(shard_dim, start, shard_size)
    expert_data.copy_(loaded_weight)


def _load_w2(
    expert_data: torch.Tensor,
    loaded_weight: torch.Tensor,
    shard_id: str,
    shard_dim: int,
    tp_rank: int,
    is_bias: bool,
    use_presharded_weights: bool,
    do_transpose: bool,
):
    if not isinstance(expert_data, torch.Tensor) or not isinstance(loaded_weight, torch.Tensor):
        raise ValueError("expert_data and loaded_weight must be torch.Tensor")

    if shard_id != "w2":
        raise ValueError(f"shard_id must be 'w2', got {shard_id}")

    # Index the loaded weight for tp sharding.
    # down_proj: "RowParallel" so tp sharding on input_dim
    # Narrow parameter and load.
    if is_bias:
        # this expert_data is a bias, not weight,
        # for w2_weight_bias in TP, it does not need to be sharded
        shard_size = expert_data.shape[-1]
    else:
        # this parameter is a weight matrix
        # for w2 in TP, it shards the input_features, i.e., shard_dim=2
        shard_size = expert_data.shape[shard_dim]

    if not use_presharded_weights:
        if not is_bias and do_transpose:
            # do not transpose for bias
            loaded_weight = loaded_weight.transpose(-2, -1)
        loaded_weight = loaded_weight.narrow(shard_dim, shard_size * tp_rank, shard_size)

    # w2, down_proj: Load into only logical weight of w2.
    expert_data.copy_(loaded_weight)


def get_shard_dim(param, shard_id, do_transpose):
    is_transposed = getattr(param, "is_transposed", False)
    if do_transpose:
        is_transposed = True

    SHARD_ID_TO_SHARDED_DIM = {"w1": 0, "w2": 1, "w3": 0}
    shard_dim = SHARD_ID_TO_SHARDED_DIM[shard_id]
    if is_transposed:
        shard_dim = int(not shard_dim)
    return shard_dim


def load_model_weight(
    param: torch.Tensor,
    loaded_weight: torch.Tensor,
    shard_id: str,
    local_expert_id: int,
    tp_rank: int,
    is_bias: bool,
    use_presharded_weights: bool,
    do_transpose: bool,
):

    expert_data = param.data[local_expert_id]
    shard_dim = get_shard_dim(param, shard_id, do_transpose)

    if shard_id == "w2":
        load_func = _load_w2
    elif shard_id in ("w1", "w3", "w13"):
        load_func = partial(_load_w13, load_up_proj_weight_first=False)  # FIXME

    load_func(
        expert_data,
        loaded_weight,
        shard_id,
        shard_dim,
        tp_rank,
        is_bias,
        use_presharded_weights,
        do_transpose,
    )


# for per tensor weight quantization
def load_per_tensor_weight_scale(
    param: torch.nn.Parameter,
    loaded_weight: torch.Tensor,
    shard_id: str,
    local_expert_id: int,
):
    if shard_id in ("w1", "w3"):
        # We have to keep the weight scales of w1 and w3 because
        # we need to re-quantize w1/w3 weights after weight loading.
        idx = 0 if shard_id == "w1" else 1
        param.data[local_expert_id][idx] = loaded_weight
    # If we are in the row parallel case (down_proj)
    elif shard_id == "w2":
        param.data[local_expert_id] = loaded_weight
    else:
        raise RuntimeError(f"Unrecognized shard_id: {shard_id}")


def load_group_weight_scale(param: torch.Tensor, loaded_weight: torch.Tensor, local_expert_id: int, shard_id: str, tp_rank: int, do_transpose: bool):
    expert_data = param.data[local_expert_id]
    shard_dim = get_shard_dim(param, shard_id, do_transpose)

    if shard_id == "w2":
        load_func = _load_w2
    elif shard_id in ("w1", "w3", "w13"):
        load_func = partial(_load_w13, load_up_proj_weight_first=False)  # FIXME

    load_func(expert_data, loaded_weight, shard_id, shard_dim, tp_rank, False, False, do_transpose, False)


def load_per_channel_weight_scale(
    param: torch.Tensor, loaded_weight: torch.Tensor, local_expert_id: int, shard_id: str, tp_rank: int, do_transpose: bool
):
    expert_data = param.data[local_expert_id]
    shard_dim = get_shard_dim(param, shard_id, do_transpose)

    # for per channel weight quantization
    if shard_id == "w2":
        expert_data.copy_(loaded_weight)
    elif shard_id in ("w1", "w3"):
        _load_w13(expert_data, loaded_weight, shard_id, shard_dim, tp_rank, False, False, do_transpose, False)


def load_g_idx(
    shard_id: str,
    expert_data: torch.Tensor,
    shard_dim: int,
    loaded_weight: torch.Tensor,
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


def load_single_value(param: torch.nn.Parameter, loaded_weight: torch.Tensor, local_expert_id: int):
    # Input scales can be loaded directly and should be equal.
    param.data[local_expert_id] = loaded_weight
