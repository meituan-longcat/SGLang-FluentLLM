# Adapted from https://github.com/vllm-project/vllm/blob/v0.6.4.post1/vllm/_custom_ops.py
from typing import List, Tuple

import torch

from sglang.srt.utils import get_colorful_logger, is_npu

logger = get_colorful_logger(__name__)
__is_npu__ = is_npu()


if not __is_npu__:
    try:
        from flashinfer.comm import vllm_ar as flashinfer_ar
    except ImportError:
        raise ImportError("flashinfer not correctly installed!")

    custom_op = flashinfer_ar

    # custom allreduce
    def init_custom_ar(
        ipc_tensors: List[int],
        rank_data: torch.Tensor,
        rank: int,
        full_nvlink: bool,
    ) -> int:
        return custom_op.init_custom_ar(ipc_tensors, rank_data, rank, full_nvlink)

    def all_reduce(
        fa: int,
        inp: torch.Tensor,
        out: torch.Tensor,
        reg_buffer: int,
        reg_buffer_sz_bytes: int,
        num_ctas: int = 4,
    ) -> None:
        custom_op.all_reduce(fa, inp, out, reg_buffer, reg_buffer_sz_bytes, num_ctas)

    def dispose(fa: int) -> None:
        custom_op.dispose(fa)

    def meta_size() -> int:
        return custom_op.meta_size()

    def register_buffer(fa: int, ipc_tensors: List[int]) -> None:
        return custom_op.register_buffer(fa, ipc_tensors)

    def get_graph_buffer_ipc_meta(fa: int) -> Tuple[List[int], List[int]]:
        return custom_op.get_graph_buffer_ipc_meta(fa)

    def register_graph_buffers(
        fa: int, handles: List[List[int]], offsets: List[List[int]]
    ) -> None:
        custom_op.register_graph_buffers(fa, handles, offsets)
