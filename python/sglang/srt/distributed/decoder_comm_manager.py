import torch
import contextlib
from typing import Tuple

from .parallel_strategy import AttnParallelStrategy, DenseParallelStategy, MoeParallelStrategy

from sglang.srt.env import global_server_args_dict
from sglang.srt.layers.dp_attention import (
    get_attention_tp_group,
    get_attention_tp_size,
    get_attention_tp_rank,
    get_dense_tp_group,
    get_dense_tp_rank,
    get_dense_tp_size,
    get_dense_dp_rank,
    get_attn_tp_dp_convertor,
    get_dense_tp_dp_convertor,
)
from sglang.srt.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
    tensor_model_parallel_all_reduce,
)
from sglang.srt.layers.utils import FLLM_IS_CP

from sglang.srt.utils import get_colorful_logger, is_npu, is_sm90_supported
__is_npu__ = is_npu()

logger = get_colorful_logger(__name__)


@contextlib.contextmanager
def split_and_join(current_stream, alt_stream):
    alt_stream.wait_stream(current_stream)
    try:
        yield
    finally:
        current_stream.wait_stream(alt_stream)


class RSAG(object):
    def __init__(self):
        self.tp_dp_convertor = get_attn_tp_dp_convertor()
        self.rank_in_group = get_attention_tp_rank()

        self.alt_stream = torch.cuda.Stream()
        if global_server_args_dict["enable_deep_ep"]:
            self.reduce_scatter_num_blocks = self.all_gather_num_blocks = 14
        else:
            self.reduce_scatter_num_blocks = 45
            self.all_gather_num_blocks = 28

    def reduce_scatter(self, hidden_states, tp_num_tokens, safe=True) ->  Tuple[torch.Tensor, int]:
        with split_and_join(torch.cuda.current_stream(), self.alt_stream):
            with torch.cuda.stream(self.alt_stream):
                dtype = hidden_states.dtype

                rs_context = self.tp_dp_convertor.get_reduce_scatter_context(tp_num_tokens, self.reduce_scatter_num_blocks)
                rs_input = rs_context.input().view(dtype=dtype)
                rs_input.copy_(hidden_states)
                self.tp_dp_convertor.reduce_scatter(rs_context, self.alt_stream.cuda_stream)
                hidden_states = rs_context.output()
                hidden_states = hidden_states.view(dtype=dtype)

                if safe:
                    hidden_states = hidden_states.clone()
                return hidden_states, rs_context.output_row_offset

    def all_gather(self, hidden_states, tp_num_tokens, safe=True) -> torch.Tensor:
        with split_and_join(torch.cuda.current_stream(), self.alt_stream):
            with torch.cuda.stream(self.alt_stream):
                dtype = hidden_states.dtype

                ag_context = self.tp_dp_convertor.get_all_gather_context(tp_num_tokens, hidden_states.shape[-1], self.all_gather_num_blocks)
                if hidden_states.shape[0] > 0:
                    ag_input = ag_context.input().view(dtype=dtype)
                    ag_input.copy_(hidden_states)
                self.tp_dp_convertor.all_gather(ag_context, self.alt_stream.cuda_stream)
                hidden_states = ag_context.output()
                hidden_states = hidden_states.view(dtype=dtype)

                if safe:
                    hidden_states = hidden_states.clone()
                return hidden_states


_ATTN_RSAG = None
_DENSE_RSAG = None


def get_attn_rsag_v1():
    global _ATTN_RSAG
    if _ATTN_RSAG is None:
        _ATTN_RSAG = RSAG()
    return _ATTN_RSAG

def get_attn_rsag_v2():
    global _ATTN_RSAG
    if _ATTN_RSAG is None:
        _ATTN_RSAG = get_attn_tp_dp_convertor()
    return _ATTN_RSAG

def get_dense_rsag():
    global _ATTN_RSAG
    global _DENSE_RSAG
    if is_sm90_supported() and not global_server_args_dict["force_deterministic_rsag"]:
        if _DENSE_RSAG is None:
            _DENSE_RSAG = get_dense_tp_dp_convertor()
    else:
        if _DENSE_RSAG is None:
            _DENSE_RSAG = _ATTN_RSAG
    return _DENSE_RSAG

def get_attn_rsag():
    if is_sm90_supported() and not global_server_args_dict["force_deterministic_rsag"]:
        return get_attn_rsag_v2()
    else:
        return get_attn_rsag_v1()


class DecoderCommMananger(object):
    def __init__(self, layer_id, attn_parallel_strategy, dense_parallel_strategy, moe_parallel_strategy, is_moe_layer, num_layers) -> None:
        self.attn_parallel_strategy = attn_parallel_strategy
        self.dense_parallel_strategy = dense_parallel_strategy
        self.moe_parallel_strategy = moe_parallel_strategy
        self.is_moe_layer = is_moe_layer
        self.num_layers = num_layers
        self.layer_id = layer_id

        self.attn_tp_rank = get_attention_tp_rank()
        self.attn_tp_group = get_attention_tp_group()
        self.attn_tp_size = get_attention_tp_size()

        self.dense_tp_group = get_dense_tp_group()
        self.dense_dp_rank = get_dense_dp_rank()
        self.dense_tp_size = get_dense_tp_size()
        self.dense_tp_rank = get_dense_tp_rank()

        self.tp_rank = get_tensor_model_parallel_rank()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_group = get_tp_group()

        self.attn_rsag = None
        self.dense_rsag = None
        if not __is_npu__:
            if self.attn_tp_size > 1:
                self.attn_rsag = get_attn_rsag()
            if self.dense_tp_size > 1:
                self.dense_rsag = get_dense_rsag()

    def pre_attn_comm(self, hidden_states, tp_num_tokens, is_second_attn=False):
        if FLLM_IS_CP:
            return hidden_states
        if self.layer_id > 0 or is_second_attn:
            if (
                self.is_moe_layer
                and
                self.attn_parallel_strategy == AttnParallelStrategy.TENSOR_PARALLEL
                and
                self.moe_parallel_strategy == MoeParallelStrategy.EXPERT_PARALLEL
            ):
                hidden_states = self.attn_rsag.all_gather(hidden_states, tp_num_tokens)
                return hidden_states
            elif (
                not self.is_moe_layer
                and
                self.attn_parallel_strategy == AttnParallelStrategy.TENSOR_PARALLEL
                and
                self.dense_parallel_strategy == DenseParallelStategy.REPLICATED
            ):
                hidden_states = self.attn_rsag.all_gather(hidden_states, tp_num_tokens)
                return hidden_states
            elif (
                not self.is_moe_layer
                and
                self.attn_parallel_strategy == AttnParallelStrategy.TENSOR_PARALLEL
                and
                self.dense_parallel_strategy == DenseParallelStategy.TENSOR_PARALLEL
            ):
                if self.moe_parallel_strategy == MoeParallelStrategy.EXPERT_PARALLEL or self.attn_tp_size != self.tp_size:
                    hidden_states = self.attn_rsag.all_gather(hidden_states, tp_num_tokens)
                return hidden_states
            else:
                return hidden_states
        else:
            return hidden_states

    def post_attn_comm(self, hidden_states, residual, tp_num_tokens, is_first_attn=True):
        if FLLM_IS_CP:
            return hidden_states, residual
        if self.is_moe_layer:
            if (
                self.attn_parallel_strategy == AttnParallelStrategy.TENSOR_PARALLEL
                and
                self.moe_parallel_strategy == MoeParallelStrategy.TENSOR_PARALLEL
            ):
                return tensor_model_parallel_all_reduce(hidden_states), residual
            elif (
                self.attn_parallel_strategy == AttnParallelStrategy.TENSOR_PARALLEL
                and
                self.moe_parallel_strategy == MoeParallelStrategy.EXPERT_PARALLEL
            ):
                hidden_states, offset = self.attn_rsag.reduce_scatter(hidden_states, tp_num_tokens)
                if (self.layer_id == 0 and is_first_attn):
                    residual = residual[offset: offset + hidden_states.shape[0]]
                return hidden_states, residual
            elif (
                self.attn_parallel_strategy == AttnParallelStrategy.DATA_PARALLEL
                and
                self.moe_parallel_strategy == MoeParallelStrategy.EXPERT_PARALLEL
            ):
                return hidden_states, residual
            elif (
                self.attn_parallel_strategy == AttnParallelStrategy.DATA_PARALLEL
                and
                self.moe_parallel_strategy == MoeParallelStrategy.TENSOR_PARALLEL
            ):
                return hidden_states, residual
            else:
                raise NotImplementedError("NotImplementedError")
        else:
            if (
                self.attn_parallel_strategy == AttnParallelStrategy.TENSOR_PARALLEL
                and
                self.dense_parallel_strategy == DenseParallelStategy.TENSOR_PARALLEL
            ):
                if self.moe_parallel_strategy == MoeParallelStrategy.EXPERT_PARALLEL or self.attn_tp_size != self.tp_size:
                    hidden_states, offset = self.attn_rsag.reduce_scatter(hidden_states, tp_num_tokens)
                    if is_first_attn and self.layer_id == 0:
                        residual = residual[offset: offset + hidden_states.shape[0]]
                    return hidden_states, residual
                else:
                    return tensor_model_parallel_all_reduce(hidden_states), residual
            elif(
                self.attn_parallel_strategy == AttnParallelStrategy.TENSOR_PARALLEL
                and
                self.dense_parallel_strategy == DenseParallelStategy.REPLICATED
            ):
                hidden_states, offset = self.attn_rsag.reduce_scatter(hidden_states, tp_num_tokens)
                if (self.layer_id == 0 and is_first_attn):
                    residual = residual[offset: offset + hidden_states.shape[0]]
                return hidden_states, residual
            elif (
                self.attn_parallel_strategy == AttnParallelStrategy.DATA_PARALLEL
                and
                self.dense_parallel_strategy == DenseParallelStategy.REPLICATED
            ):
                return hidden_states, residual
            elif (
                self.attn_parallel_strategy == AttnParallelStrategy.DATA_PARALLEL
                and
                self.dense_parallel_strategy == DenseParallelStategy.TENSOR_PARALLEL
            ):
                return hidden_states, residual
            else:
                logger.error(
                    f"attn_parallel_strategy: {self.attn_parallel_strategy}, "
                    f"dense_parallel_strategy: {self.dense_parallel_strategy}"
                )
                raise NotImplementedError("NotImplementedError")

    def post_attn_comm_for_tbo(self, hidden_states, residual, tp_num_tokens, is_first_tbo_attn=False):
        assert self.is_moe_layer
        if (
            self.attn_parallel_strategy == AttnParallelStrategy.TENSOR_PARALLEL
            and
            self.moe_parallel_strategy == MoeParallelStrategy.TENSOR_PARALLEL
        ):
            return tensor_model_parallel_all_reduce(hidden_states), residual
        elif (
            self.attn_parallel_strategy == AttnParallelStrategy.TENSOR_PARALLEL
            and
            self.moe_parallel_strategy == MoeParallelStrategy.EXPERT_PARALLEL
        ):
            hidden_states, offset = self.attn_rsag.reduce_scatter(hidden_states, tp_num_tokens)
            if is_first_tbo_attn:
                residual = residual[offset: offset + hidden_states.shape[0]]
            return hidden_states, residual
        elif (
            self.attn_parallel_strategy == AttnParallelStrategy.DATA_PARALLEL
            and
            self.moe_parallel_strategy == MoeParallelStrategy.EXPERT_PARALLEL
        ):
            return hidden_states, residual
        elif (
            self.attn_parallel_strategy == AttnParallelStrategy.DATA_PARALLEL
            and
            self.moe_parallel_strategy == MoeParallelStrategy.TENSOR_PARALLEL
        ):
            return hidden_states, residual
        else:
            raise NotImplementedError("NotImplementedError")

    def post_final_norm_comm(self, hidden_states, residual, tp_num_tokens):
        if (
            self.attn_parallel_strategy == AttnParallelStrategy.TENSOR_PARALLEL
            and
            self.moe_parallel_strategy == MoeParallelStrategy.EXPERT_PARALLEL
        ):
            return self.attn_rsag.all_gather(hidden_states, tp_num_tokens), residual
        elif(
            self.attn_parallel_strategy == AttnParallelStrategy.TENSOR_PARALLEL
            and
            self.moe_parallel_strategy == DenseParallelStategy.TENSOR_PARALLEL
        ):
            if self.moe_parallel_strategy == MoeParallelStrategy.EXPERT_PARALLEL or self.dense_tp_size != self.tp_size:
                return self.attn_rsag.all_gather(hidden_states, tp_num_tokens), residual
            else:
                return hidden_states, residual
        else:
            return hidden_states, residual

    def pre_mlp_comm(self, hidden_states, forward_batch, tp_num_tokens):
        if self.is_moe_layer:
            if (
                self.attn_parallel_strategy == AttnParallelStrategy.DATA_PARALLEL
                and
                self.moe_parallel_strategy == MoeParallelStrategy.TENSOR_PARALLEL
            ):
                return self.all_gather_origin(hidden_states, forward_batch)
            else:
                return hidden_states, None, None
        else:
            if (
                self.attn_parallel_strategy == AttnParallelStrategy.TENSOR_PARALLEL
                and
                self.dense_parallel_strategy == DenseParallelStategy.TENSOR_PARALLEL
            ):
                if self.moe_parallel_strategy == MoeParallelStrategy.EXPERT_PARALLEL or self.dense_tp_size != self.tp_size:
                    hidden_states = self.dense_rsag.all_gather(hidden_states, tp_num_tokens)
                return hidden_states, None, None
            elif(
                self.attn_parallel_strategy == AttnParallelStrategy.DATA_PARALLEL
                and
                self.dense_parallel_strategy == DenseParallelStategy.TENSOR_PARALLEL
            ):
                if self.dense_rsag is None:
                    hidden_states = self.all_gather_torch(hidden_states, forward_batch)
                else:
                    local_global_num_tokens = self.get_dense_group_num_tokens(forward_batch)
                    hidden_states = self.dense_rsag.all_gather(hidden_states, tp_num_tokens, local_global_num_tokens)
                return hidden_states, None, None
            else:
                return hidden_states, None, None

    def post_mlp_comm(self, hidden_states, residual, tp_num_tokens, forward_batch=None):
        if self.is_moe_layer:
            if self.moe_parallel_strategy == MoeParallelStrategy.TENSOR_PARALLEL:
                return tensor_model_parallel_all_reduce(hidden_states), residual
            else:
                return hidden_states, residual
        else:
            if (
                self.attn_parallel_strategy == AttnParallelStrategy.TENSOR_PARALLEL
                and
                self.dense_parallel_strategy == DenseParallelStategy.TENSOR_PARALLEL
            ):
                if self.moe_parallel_strategy == MoeParallelStrategy.EXPERT_PARALLEL or self.dense_tp_size != self.tp_size:
                    hidden_states, _ = self.dense_rsag.reduce_scatter(hidden_states, tp_num_tokens)
                else:
                    hidden_states = tensor_model_parallel_all_reduce(hidden_states)
                return hidden_states, residual
            elif(
                self.attn_parallel_strategy == AttnParallelStrategy.DATA_PARALLEL
                and
                self.dense_parallel_strategy == DenseParallelStategy.TENSOR_PARALLEL
            ):
                if self.dense_rsag is None:
                    hidden_states = self.reduce_scatter_torch(hidden_states, forward_batch)
                else:
                    local_global_num_tokens = self.get_dense_group_num_tokens(forward_batch)
                    hidden_states, _ = self.dense_rsag.reduce_scatter(hidden_states, tp_num_tokens, local_global_num_tokens)
                return hidden_states, residual
            else:
                return hidden_states, residual

    def get_dense_group_num_tokens(self, forward_batch):
        global_num_tokens = forward_batch.global_num_tokens
        local_global_num_tokens = global_num_tokens[self.dense_dp_rank * self.dense_tp_size: (self.dense_dp_rank + 1) * self.dense_tp_size]
        return local_global_num_tokens

    def reduce_scatter_torch(self, hidden_states, forward_batch):
        local_global_num_tokens = self.get_dense_group_num_tokens(forward_batch)
        device = hidden_states.device
        dtype = hidden_states.dtype
        output = torch.empty(local_global_num_tokens[self.dense_tp_rank], hidden_states.shape[-1], device=device, dtype=dtype)
        split_tensors = torch.split(hidden_states, local_global_num_tokens, dim=0)
        input_list = list(split_tensors)
        torch.distributed.reduce_scatter(
            output,
            input_list,
            op=torch.distributed.ReduceOp.SUM,
            group=self.dense_tp_group.device_group,
        )
        return output

    def all_gather_torch(self, hidden_states: torch.Tensor, forward_batch):
        local_global_num_tokens = self.get_dense_group_num_tokens(forward_batch)
        device = hidden_states.device
        dtype = hidden_states.dtype
        gathered_tensors = [torch.empty(num_tokens, hidden_states.shape[-1], dtype=dtype, device=device) for num_tokens in local_global_num_tokens]
        torch.distributed.all_gather(
            gathered_tensors, hidden_states, self.dense_tp_group.device_group
        )
        gathered_tensors = torch.concat(gathered_tensors)
        return gathered_tensors

    def all_gather_origin(
        self, input_tensor: torch.Tensor, forward_batch
    ):
        if self.tp_size == 1:
            return input_tensor, None, None

        all_lens = forward_batch.global_num_tokens
        max_len = max(forward_batch.global_num_tokens)
        rank = self.tp_rank

        padded_tensor = torch.nn.functional.pad(
            input_tensor, (0, 0, 0, max_len - input_tensor.shape[0])
        )

        self.tp_group.all_gather_into_tensor(forward_batch.gathered_buffer, padded_tensor)

        gathered_tensors = torch.concat(
            [
                forward_batch.gathered_buffer[i * max_len: i * max_len + all_lens[i]]
                for i in range(self.tp_size)
            ]
        )
        start_index = 0 if rank == 0 else sum(all_lens[:rank])
        end_index = start_index + all_lens[rank]
        return gathered_tensors, start_index, end_index
