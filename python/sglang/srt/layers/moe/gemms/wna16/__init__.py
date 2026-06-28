from __future__ import annotations

import os
import re
import functools
import torch
import enum
import logging
import json
from enum import Enum
from typing import TYPE_CHECKING, Callable, List, Optional, Tuple
from sglang.srt.layers.moe.config import DispatcherType
from sglang.srt.layers.activation import SwigluArg
from sglang.srt.utils import (
    get_compiler_backend,
)
logger = logging.getLogger(__name__)


def _mask_topk_ids_cpu_experts(topk_ids: torch.Tensor, num_gpu_experts: int):
    """Mask topk_ids >= num_gpu_experts by setting them to -1."""
    topk_ids[topk_ids >= num_gpu_experts] = -1


@torch.compile(dynamic=True, backend=get_compiler_backend())
def mask_cpu_expert_ids(topk_ids: torch.Tensor, num_gpu_experts: int):
    """mask CPU expert IDs."""
    _mask_topk_ids_cpu_experts(topk_ids, num_gpu_experts)
    return topk_ids


class GPTQMarlinState(Enum):
    REPACK = enum.auto()
    READY = enum.auto()


class WNA16MoEGemmWrapper():

    def __init__(self, top_k, num_experts, layer, quant_config):
        self.top_k= top_k
        self.num_experts= num_experts
        self.quant_config = quant_config

    @staticmethod
    def get_layout_cls():
        from sglang.srt.layers.moe.layouts.wna16 import CompressedTensorsWNA16MoELayout
        return CompressedTensorsWNA16MoELayout

    def get_executor(self, dispatcher_type : DispatcherType, activation: str, swiglu_arg: Optional[SwigluArg] = None):
        if dispatcher_type == DispatcherType.TP:
            from sglang.srt.layers.moe.executors.wna16_executor import WNA16MoeExecutor
            return WNA16MoeExecutor(self.quant_config)
        else:
            raise RuntimeError("WNA16MoEGemmWrapper only Support Moe EP")
