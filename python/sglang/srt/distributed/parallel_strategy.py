from enum import Enum


class MoeParallelStrategy(str, Enum):
    TENSOR_PARALLEL = "tp"
    EXPERT_PARALLEL = "ep"


class AttnParallelStrategy(str, Enum):
    TENSOR_PARALLEL = "tp"
    DATA_PARALLEL = "dp"
    COMBINE = "combine"


class DenseParallelStategy(str, Enum):
    TENSOR_PARALLEL = "tp"
    REPLICATED = "rep"
    COMBINE = "combine"  # dp+tp, tp size consistent with attn
