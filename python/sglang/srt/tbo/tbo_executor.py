from __future__ import annotations

import inspect
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable

import deep_gemm
import torch

from sglang.srt.distributed import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size, get_world_group
from sglang.srt.env import global_server_args_dict
from sglang.srt.layers.moe.config import EPConfig
from sglang.srt.layers.moe.dispatcher.deep_ep import DeepEPDispatcher
from sglang.srt.utils import get_colorful_logger

logger = get_colorful_logger(__name__)


@dataclass
class Orchestration:
    stages: list[Callable]
    delta_stage: int
    pre_process: Callable | None
    post_process: Callable
    deep_gemm_num_sms: int | None = None


def tbo_stage(
    provides: str | list[str] | None = None,
    requires: str | list[str] | None = None,
    temp: bool = False,
    use_globals: str | list[str] | None = None,
):
    def decorator(func: Callable) -> Callable:
        func.tbo_metadata = {"provides": provides, "requires": requires, "temp": temp, "use_globals": use_globals}
        return func

    return decorator


class TBOExecutor:
    def __init__(self, orchestration: Orchestration, layers):
        super().__init__()
        self.stages = []
        self.context = []
        self.temp_keys = set()
        self.permanent_keys = set()
        self.global_vars = {}
        self.orchestration = orchestration
        self.layers = layers
        self.total_stages = len(orchestration.stages) * len(layers)

        self._stage_indices = [0, 0]
        self._layer_indices = [0, 0]

        for stage in orchestration.stages:
            self.register_stage(stage)
        logger.info(f"TBOExecutor registered stages: {self.stages}")

    def set_global_var(self, key, value):
        self.global_vars[key] = value
        return self

    def get_global_var(self, key, default=None):
        return self.global_vars.get(key, default)

    def update_global_vars(self, **kwargs):
        self.global_vars.update(kwargs)
        return self

    def register_stage(self, func):
        metadata = getattr(func, "tbo_metadata", {})
        provides = metadata.get("provides")
        requires = metadata.get("requires")
        temp = metadata.get("temp", False)
        use_globals = metadata.get("use_globals")

        inferred_requires = requires
        if inferred_requires is None:
            sig = inspect.signature(func)
            inferred_requires = list(sig.parameters.keys())

        self.stages.append(
            {
                "func": func,
                "provides": [provides] if isinstance(provides, str) else provides,
                "requires": [inferred_requires] if isinstance(inferred_requires, str) else inferred_requires,
                "temp": temp,
                "use_globals": [use_globals] if isinstance(use_globals, str) else use_globals,
            }
        )

        if provides:
            provides_list = [provides] if isinstance(provides, str) else provides
            if temp:
                self.temp_keys.update(provides_list)
            else:
                self.permanent_keys.update(provides_list)

    def forward_overlap(self, **kwargs) -> dict[str, Any]:
        with self.configure_deep_gemm_num_sms(self.orchestration.deep_gemm_num_sms):
            self.clear_states()

            if self.orchestration.pre_process:
                self.context = self.orchestration.pre_process(**kwargs)

            for _ in range(self.orchestration.delta_stage):
                self._execute_one_stage(micro_batch_index=0)

            for _ in range(self.total_stages - self.orchestration.delta_stage):
                self._execute_one_stage(micro_batch_index=0)
                self._execute_one_stage(micro_batch_index=1)

            for _ in range(self.orchestration.delta_stage):
                self._execute_one_stage(micro_batch_index=1)

            result = self.orchestration.post_process(self.context, self.layers)
        return result

    @contextmanager
    def configure_deep_gemm_num_sms(self, num_sms):
        if num_sms is None:
            yield
        else:
            original_num_sms = deep_gemm.get_num_sms()
            deep_gemm.set_num_sms(num_sms)
            try:
                yield
            finally:
                deep_gemm.set_num_sms(original_num_sms)

    def _execute_one_stage(self, micro_batch_index):
        assert not self.done(micro_batch_index)

        current_stage_index = self._stage_indices[micro_batch_index]
        stage = self.stages[current_stage_index]

        current_layer_index = self._layer_indices[micro_batch_index]
        layer = self.layers[current_layer_index]

        kwargs = {"layer": layer}
        for key in stage["requires"]:
            if key in self.context[micro_batch_index]:
                kwargs[key] = self.context[micro_batch_index][key]

        try:
            results = stage["func"](**kwargs)
        except Exception as e:
            logger.error(
                f"TBO failed to execute stage {stage}, "
                f"micro_batch_index: {micro_batch_index}, "
                f"current_stage_index: {current_stage_index}, "
                f"current_layer_index: {current_layer_index}, "
                f"context: {self.context}, e: {e}"
            )
            raise e

        for key in self.temp_keys:
            if key in self.context[micro_batch_index] and key not in self.permanent_keys:
                del self.context[micro_batch_index][key]

        if results is not None:
            if len(stage["provides"]) == 1:
                self.context[micro_batch_index][stage["provides"][0]] = results
            else:
                if isinstance(results, (tuple, list)) and len(results) == len(stage["provides"]):
                    for i, key in enumerate(stage["provides"]):
                        self.context[micro_batch_index][key] = results[i]
                else:
                    self.context[micro_batch_index][stage["provides"][0]] = results

        self._stage_indices[micro_batch_index] += 1
        if self._stage_indices[micro_batch_index] == len(self.stages):
            self._layer_indices[micro_batch_index] += 1
            self._stage_indices[micro_batch_index] = 0

    def done(self, batch_index):
        return self._layer_indices[batch_index] >= len(self.layers)

    def clear_states(self):
        self.context.clear()
        self._stage_indices = [0, 0]
        self._layer_indices = [0, 0]


class TBODeepEPDispatchers:
    def __init__(self, top_k, num_experts, hidden_size):
        config = EPConfig(
            top_k=top_k,
            num_experts=num_experts,
            low_latency_max_num_tokens_per_gpu=global_server_args_dict["low_latency_max_num_tokens_per_gpu"],
            max_num_tokens_per_gpu=1024,
            hidden_size=hidden_size,
            rank=get_tensor_model_parallel_rank(),
            world_size=get_tensor_model_parallel_world_size(),
            group=get_world_group().device_group,
            params_dtype=torch.bfloat16,
        )

        self.deepep_dispatchers = [DeepEPDispatcher(config), DeepEPDispatcher(config)]

    def __getitem__(self, index):
        return self.deepep_dispatchers[index]

    def __len__(self):
        return len(self.deepep_dispatchers)
