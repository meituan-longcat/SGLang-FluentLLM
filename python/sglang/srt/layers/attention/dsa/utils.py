import torch
import torch.nn.functional as F

from sglang.srt.layers.attention.dp_attention import get_attention_tp_group

# temp NSA debugging environ
from sglang.srt.utils import get_bool_env_var

NSA_USE_REAL_INDEXER = get_bool_env_var("SGLANG_NSA_USE_REAL_INDEXER", "true")
NSA_DUAL_STREAM = get_bool_env_var("SGLANG_NSA_DUAL_STREAM", "true")
NSA_FUSE_TOPK = get_bool_env_var("SGLANG_NSA_FUSE_TOPK", "true")

NSA_FLASHMLA_BACKEND_DECODE_COMPUTE_FP8 = get_bool_env_var("SGLANG_NSA_FLASHMLA_BACKEND_DECODE_COMPUTE_FP8", "true")
NSA_QUANT_K_CACHE_FAST = get_bool_env_var("SGLANG_NSA_QUANT_K_CACHE_FAST", "true")
NSA_DEQUANT_K_CACHE_FAST = get_bool_env_var("SGLANG_NSA_DEQUANT_K_CACHE_FAST", "true")


def print_nsa_bool_env_vars():
    msg = ""
    for k, v in globals().items():
        if k.startswith("NSA_") and isinstance(v, bool):
            msg += f"{k}={v} "
    print(msg, flush=True)


def compute_nsa_seqlens(original_seq_lens, nsa_index_topk: int):
    return original_seq_lens.clamp(max=nsa_index_topk)


def cp_attn_tp_all_gather_reorganazied_into_tensor(input_: torch.Tensor, total_len, attn_tp_size, forward_batch, stream_op):
    """
    Allgather communication for context_parallel(kv_cache, index_k, hidden_states).
    This implementation mainly consists of three parts:
    Step 1, padding the input shape to unify the shape for allgather communication (the shape must be the same).
    Step 2, allgather communication(async).
    Step 3, removing the padding and reassembling the data according to the actual tokens.
    """
    # step1
    max_len = (total_len + attn_tp_size - 1) // attn_tp_size
    pad_size = max_len - input_.shape[0]
    if pad_size > 0:
        input_ = F.pad(input_, (0, 0, 0, pad_size), mode="constant", value=0)
    input_tensor_all = torch.empty(
        max_len * attn_tp_size,
        input_.shape[1],
        device=input_.device,
        dtype=input_.dtype,
    )
    # step2
    get_attention_tp_group().cp_all_gather_into_tensor_async(input_tensor_all, input_, stream_op)
    # step3
    outputs_list_max = list(torch.split(input_tensor_all, forward_batch.nsa_cp_metadata.max_rank_len, dim=0))
    outputs = torch.cat(
        [outputs_list_max[index][:per_rank_len] for index, per_rank_len in enumerate(forward_batch.nsa_cp_metadata.per_rank_actual_token)],
        dim=0,
    )
    return outputs


def cp_all_gather_rerange_output(input_tensor, cp_size, forward_batch, stream):
    """
    |   +-----------before allgather------------+|
    |   | dp_atten_tp0: block0, block7 |
    |   | dp_atten_tp1: block1, block6 |
    |   | dp_atten_tp2: block2, block5 |
    |   | dp_atten_tp3: block3, block4 |
    |
    |   +----------before rerange---------------+|
    | block0 | block7 | block1 | block6 | block2 | block5 | block3 | block4 |
    |
    |   +--------------result-------------------+
    | block0 | block1 | block2 | block3 | block4 | block5 | block6 | block7 |
    |   +-------------------------+
    """
    bs_seq_len, hidden_size = input_tensor.shape
    output_tensor = cp_attn_tp_all_gather_reorganazied_into_tensor(
        input_tensor,
        forward_batch.nsa_cp_metadata.total_seq_lens,
        cp_size,
        forward_batch,
        stream,
    )
    outputs_list = list(torch.split(output_tensor, forward_batch.nsa_cp_metadata.reverse_split_len, dim=0))
    output_tensor = torch.cat([outputs_list[i] for i in forward_batch.nsa_cp_metadata.cp_reverse_index], dim=0)
    output_tensor = output_tensor.view(-1, hidden_size)
    return output_tensor
