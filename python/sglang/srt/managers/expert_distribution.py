import logging
import os
import time
from abc import ABC, abstractmethod
from collections import deque
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Type

import einops
import torch
import torch.nn.functional as F
import torch.distributed

from sglang.srt.managers.expert_location import ExpertLocationMetadata
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import Withable, get_device_name, get_colorful_logger, get_device_module, is_npu

logger = get_colorful_logger(__name__)
_is_npu = is_npu()

# --------------------------------------- Entrypoint -----------------------------------------

_OutputMode = Literal["file", "object"]


class ExpertDistributionRecorder(ABC):
    """Global expert distribution recording"""

    @staticmethod
    def init_new(
        server_args: ServerArgs,
        expert_location_metadata: "ExpertLocationMetadata",
        rank: int,
    ):
        if server_args.expert_distribution_recorder_mode is not None:
            return _ExpertDistributionRecorderReal(
                server_args, expert_location_metadata, rank
            )
        else:
            return _ExpertDistributionRecorderNoop()

    def set_current_layer(self, layer_idx):
        pass

    @contextmanager
    def with_current_layer(self, layer_idx):
        yield

    @contextmanager
    def with_debug_name(self, debug_name):
        yield

    @contextmanager
    def with_forward_pass(self, forward_pass_id: int, forward_batch: ForwardBatch):
        yield
    
    def on_local_expert_counts(self, local_expert_counts: torch.Tensor):
        pass

    def on_select_experts(self, topk_ids: torch.Tensor, num_experts: Optional[int] = None):
        pass

    def on_deepep_dispatch_normal(
        self,
        local_physical_count_of_layer: List[int],
        num_tokens_per_rank,
        num_tokens_per_rdma_rank,
        num_tokens_per_expert,
    ):
        pass

    def on_deepep_dispatch_low_latency(
        self, local_physical_count_of_layer: torch.Tensor
    ):
        pass

    def start_record(self):
        self._on_not_implemented()

    def stop_record(self):
        self._on_not_implemented()

    def dump_record(self, output_mode: _OutputMode = "file", reset: bool = True):
        self._on_not_implemented()

    @property
    def recording(self):
        return False

    def _on_not_implemented(self):
        raise Exception(
            "Please set ServerArgs.expert_distribution_recorder_mode to use ExpertDistributionRecorder."
        )


class _ExpertDistributionRecorderNoop(ExpertDistributionRecorder):
    pass


class _ExpertDistributionRecorderReal(ExpertDistributionRecorder):
    def __init__(
        self,
        server_args: ServerArgs,
        expert_location_metadata: "ExpertLocationMetadata",
        rank: int,
    ):
        self._server_args = server_args
        self._expert_location_metadata = expert_location_metadata

        self._recording = False
        self._current_forward_pass_id = Withable()
        self._current_layer_idx = Withable()
        self._current_debug_name = Withable()
        self._accumulator = _Accumulator.init_new(
            server_args, expert_location_metadata, rank
        )
        self._single_pass_gatherers = {
            k: _SinglePassGatherer.init_new(server_args, expert_location_metadata, rank)
            for k in self._accumulator.get_single_pass_gatherer_keys()
        }

        if server_args.enable_expert_distribution_metrics:
            logger.info(
                "ExpertDistributionRecorder auto start record since enable_expert_distribution_metrics"
            )
            self.start_record()
    
    def set_current_layer(self, layer_idx):
        self._current_layer_idx.value = layer_idx

    def with_current_layer(self, layer_idx):
        return self._current_layer_idx.with_value(layer_idx)

    def with_debug_name(self, debug_name):
        return self._current_debug_name.with_value(debug_name)

    @contextmanager
    def with_forward_pass(self, forward_pass_id: int, forward_batch: ForwardBatch):
        with self._current_forward_pass_id.with_value(forward_pass_id):
            self._on_forward_pass_start(forward_batch)
            try:
                yield
            finally:
                self._on_forward_pass_end(forward_pass_id)

    def _on_forward_pass_start(self, forward_batch: ForwardBatch):
        if not self._recording:
            return
        for gatherer_key, gatherer in self._single_pass_gatherers.items():
            gatherer.reset()
            gatherer.on_forward_pass_start(forward_batch)

    def _on_forward_pass_end(self, forward_pass_id: int):
        if not self._recording:
            return
        for gatherer_key, gatherer in self._single_pass_gatherers.items():
            single_pass_data = gatherer.collect()
            self._accumulator.append(forward_pass_id, gatherer_key, single_pass_data)

    def on_select_experts(self, topk_ids: torch.Tensor, num_experts: Optional[int] = None):
        return self._on_hook("on_select_experts", topk_ids=topk_ids, num_experts=num_experts)

    def on_local_expert_counts(self, local_expert_counts: torch.Tensor):
        return self._on_hook("on_local_expert_counts", local_expert_counts=local_expert_counts)

    def on_deepep_dispatch_normal(
        self,
        local_physical_count_of_layer: List[int],
        num_tokens_per_rank,
        num_tokens_per_rdma_rank,
        num_tokens_per_expert,
    ):
        self._on_hook(
            "on_deepep_dispatch_normal",
            local_physical_count_of_layer=local_physical_count_of_layer,
            num_tokens_per_rank=num_tokens_per_rank,
            num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
            num_tokens_per_expert=num_tokens_per_expert,
        )

    def on_deepep_dispatch_low_latency(
        self, local_physical_count_of_layer: torch.Tensor
    ):
        self._on_hook(
            "on_deepep_dispatch_low_latency",
            local_physical_count_of_layer=local_physical_count_of_layer,
        )

    def _on_hook(self, hook_name: str, **kwargs):
        if not (self._recording or torch.cuda.is_current_stream_capturing()):
            return
        gatherer = self._single_pass_gatherers[
            self._accumulator.get_single_pass_gatherer_key(
                self._current_debug_name.value
            )
        ]
        return getattr(gatherer, hook_name)(layer_idx=self._current_layer_idx.value, **kwargs)

    def _reset(self):
        """Reset the expert distribution recorder."""
        logger.info("Resetting ExpertDistributionRecorder...")
        assert (
            self._current_layer_idx.value is None
        ), f"{self._current_layer_idx.value=}"
        for gatherer in self._single_pass_gatherers.values():
            gatherer.reset()
        self._accumulator.reset()

    def start_record(self):
        """Start recording the expert distribution."""
        if self._recording:
            logger.warning(
                "SGLang server is already recording expert ids. Did you forget to dump the expert ids recorded so far by sending requests to the `/stop_expert_distribution_record` and `/dump_expert_distribution_record` endpoints?"
            )
        self._reset()
        self._recording = True

    def stop_record(self):
        """Stop recording the expert distribution."""
        if not self._recording:
            logger.warning(
                "SGLang server has not been recording expert ids. Did you forget to start recording by sending request to the `/start_expert_distribution_record` endpoint?"
            )
        self._recording = False

    def dump_record(self, output_mode: _OutputMode = "file"):
        """Dump the expert distribution record and reset the recorder after dumping."""
        output = self._accumulator.dump(output_mode=output_mode)
        self._reset()
        return output

    @property
    def recording(self):
        return self._recording


_global_expert_distribution_recorder: Optional[ExpertDistributionRecorder] = (
    _ExpertDistributionRecorderNoop()
)


def get_global_expert_distribution_recorder():
    return _global_expert_distribution_recorder


def set_global_expert_distribution_recorder(value):
    global _global_expert_distribution_recorder
    _global_expert_distribution_recorder = value


# --------------------------------------- SinglePassGatherer -----------------------------------------


class _SinglePassGatherer(ABC):
    @staticmethod
    def init_new(
        server_args: ServerArgs,
        expert_location_metadata: "ExpertLocationMetadata",
        rank: int,
    ) -> "_SinglePassGatherer":
        if server_args.expert_distribution_recorder_mode == "per_token":
            return _DetailSinglePassGatherer(
                server_args, expert_location_metadata, rank
            )

        if server_args.expert_distribution_recorder_mode == "stat_approx":
            if server_args.enable_deepep_moe and (server_args.deepep_mode == "normal"):
                return _DeepepNormalSinglePassGatherer(expert_location_metadata, rank)
            else:
                raise NotImplementedError

        # TODO: this is not important for Flash model, will do this later.
        # if server_args.enable_deepep_moe:
        #     if server_args.deepep_mode == "normal":
        #         return _SelectExpertsSinglePassGatherer(expert_location_metadata, rank)
        #     elif server_args.deepep_mode == "low_latency":
        #         return _DeepepLowLatencySinglePassGatherer(
        #             expert_location_metadata, rank
        #         )
        #     else:
        #         raise NotImplementedError
        if _is_npu:
            return _SelectExpertsSinglePassGathererNPU(server_args, expert_location_metadata, rank)
        else:
            return _SelectExpertsSinglePassGatherer(server_args, expert_location_metadata, rank)

    def __init__(self, server_args, expert_location_metadata: "ExpertLocationMetadata", rank: int):
        self._server_args = server_args
        self._expert_location_metadata = expert_location_metadata
        self._rank = rank

    def on_forward_pass_start(self, forward_batch: ForwardBatch):
        pass

    def on_select_experts(self, layer_idx: int, topk_ids: torch.Tensor):
        pass

    def on_local_expert_counts(self, layer_idx: int, local_expert_counts: torch.Tensor):
        pass

    def on_deepep_dispatch_normal(
        self,
        layer_idx: int,
        local_physical_count_of_layer: List[int],
        num_tokens_per_rank,
        num_tokens_per_rdma_rank,
        num_tokens_per_expert,
    ):
        pass

    def on_deepep_dispatch_low_latency(
        self, layer_idx: int, local_physical_count_of_layer: torch.Tensor
    ):
        pass

    def reset(self):
        raise NotImplementedError

    def collect(self) -> Dict:
        raise NotImplementedError


class _DetailSinglePassGatherer(_SinglePassGatherer):
    # DeepSeek V3 has this value; should generalize later
    _TOP_K_NUM = 8

    def __init__(
        self,
        server_args: ServerArgs,
        expert_location_metadata: "ExpertLocationMetadata",
        rank: int,
    ):
        super().__init__(expert_location_metadata, rank)
        self._metadata: Optional[Dict[str, Any]] = None
        self._topk_ids_of_layer = torch.zeros(
            (
                expert_location_metadata.num_layers,
                # TODO determine the max number
                server_args.chunked_prefill_size * 8,
                self._TOP_K_NUM,
            ),
            dtype=torch.int32,
            device=server_args.device,
        )
        self._misc_objects: List[Dict[str, Any]] = []
        # assert (
        #     not server_args.enable_two_batch_overlap
        # ), "DetailSinglePassGatherer does not support TBO yet"
        # TODO assert shared experts fusion is disabled, o/w data is wrong

    def on_forward_pass_start(self, forward_batch: ForwardBatch):
        assert self._metadata is None
        self._metadata = dict(
            # TODO pr-chain
            # rids=forward_batch.rids,
            input_ids=forward_batch.input_ids.cpu().tolist(),
            positions=forward_batch.positions.cpu().tolist(),
            extend_seq_lens=forward_batch.extend_seq_lens_cpu,
            forward_mode=forward_batch.forward_mode.value,
        )

    def on_select_experts(self, layer_idx: int, topk_ids: torch.Tensor):
        self._topk_ids_of_layer[layer_idx, : topk_ids.shape[0], : topk_ids.shape[1]] = (
            topk_ids
        )
        return self._topk_ids_of_layer

    def on_deepep_dispatch_normal(
        self,
        layer_idx: int,
        local_physical_count_of_layer: List[int],
        num_tokens_per_rank,
        num_tokens_per_rdma_rank,
        num_tokens_per_expert,
    ):
        self._misc_objects.append(
            dict(
                layer_id=layer_idx,
                num_tokens_per_rank=num_tokens_per_rank.cpu().tolist(),
                num_tokens_per_rdma_rank=num_tokens_per_rdma_rank.cpu().tolist(),
                num_tokens_per_expert=num_tokens_per_expert.cpu().tolist(),
            )
        )

    def reset(self):
        self._topk_ids_of_layer[...] = -1
        self._misc_objects.clear()
        self._metadata = None

    def collect(self) -> Dict:
        num_tokens = len(self._metadata["input_ids"])
        return dict(
            **self._metadata,
            topk_ids_of_layer=self._topk_ids_of_layer[:, :num_tokens, :].clone().cpu(),
            misc_objects=self._misc_objects,
        )


class _LayerBasedCpuSinglePassGatherer(_SinglePassGatherer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._objects_of_layer = {}

    def _on_layer_data(self, layer_idx: int, objects: List[int]):
        assert 0 <= layer_idx < self._expert_location_metadata.num_layers
        if layer_idx in self._objects_of_layer:
            self._objects_of_layer[layer_idx] = _list_sum(
                self._objects_of_layer[layer_idx], objects
            )
        else:
            self._objects_of_layer[layer_idx] = objects

    def reset(self):
        self._objects_of_layer.clear()

    def _collect_objects(self, pad_len: int) -> torch.Tensor:
        data = [
            self._objects_of_layer.get(layer_index) or ([0] * pad_len)
            for layer_index in range(self._expert_location_metadata.num_layers)
        ]
        return torch.tensor(data)


def _list_sum(a: List, b: List) -> List:
    return [x + y for x, y in zip(a, b, strict=True)]


class _LayerBasedGpuSinglePassGatherer(_SinglePassGatherer):
    def __init__(self, *args, enable_global_physical_experts: bool, **kwargs):
        super().__init__(*args, **kwargs)
        self._enable_global_physical_experts = enable_global_physical_experts
        self._data = torch.zeros(
            (
                self._expert_location_metadata.num_layers,
                (
                    self._expert_location_metadata.num_physical_experts
                    if enable_global_physical_experts
                    else self._expert_location_metadata.num_local_physical_experts
                ),
            ),
            dtype=torch.int,
            device=self._server_args.device,
        )
        torch._dynamo.mark_static(self._data)

    def reset(self):
        self._data[...] = 0

    def collect(self) -> Dict:
        if self._enable_global_physical_experts:
            global_physical_count = self._data
        else:
            # Can optimize if bottleneck
            global_physical_count = _convert_local_to_global_physical_count(
                self._data,
                rank=self._rank,
                num_local_physical_experts=self._expert_location_metadata.num_local_physical_experts,
                num_physical_experts=self._expert_location_metadata.num_physical_experts,
            )

        return dict(global_physical_count=global_physical_count)


class _SelectExpertsSinglePassGatherer(_LayerBasedGpuSinglePassGatherer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, enable_global_physical_experts=True)

    # can optimize (e.g. fuse / compile)
    def on_select_experts(self, layer_idx: int, topk_ids: torch.Tensor, num_experts: Optional[int] = None):
        topk_ids = topk_ids.flatten()
        if num_experts is None:
            mask = topk_ids != -1
        else:
            mask = (topk_ids != -1) & (topk_ids < num_experts)
        self._data[layer_idx, :].scatter_add_(
            dim=0, index=topk_ids.masked_fill(~mask, 0).long(), src=mask.int()
        )
        return self._data


class _SelectExpertsSinglePassGathererNPU(_SinglePassGatherer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._enable_global_physical_experts = False
        self._data = [
            torch.zeros(
                (
                    (
                        self._expert_location_metadata.num_physical_experts
                        if self._enable_global_physical_experts
                        else self._expert_location_metadata.num_local_physical_experts
                    ),
                ),
                dtype=torch.int64,
                device=self._server_args.device,
            )
            for _ in range(self._expert_location_metadata.num_layers)
        ]
        for t in self._data:
            torch._dynamo.mark_static(t)

    def reset(self):
        for t in self._data:
            t.zero_()

    def collect(self) -> Dict:
        if self._enable_global_physical_experts:
            global_physical_count = torch.stack(self._data, dim=0)
        else:
            # Can optimize if bottleneck
            global_physical_count = _convert_local_to_global_physical_count(
                torch.stack(self._data, dim=0),
                rank=self._rank,
                num_local_physical_experts=self._expert_location_metadata.num_local_physical_experts,
                num_physical_experts=self._expert_location_metadata.num_physical_experts,
            )

        return dict(global_physical_count=global_physical_count)

    def on_select_experts(self, layer_idx: int, topk_ids: torch.Tensor, num_experts: Optional[int] = None):
        topk_ids = topk_ids.flatten()
        if num_experts is None:
            mask = topk_ids != -1
        elif _is_npu:
            mask = (topk_ids < num_experts)
        else:
            mask = (topk_ids != -1) & (topk_ids < num_experts)
        safe_ids = torch.where(
            mask,
            topk_ids,
            torch.tensor(0, device=topk_ids.device, dtype=topk_ids.dtype)
        )
        self._data[layer_idx].index_add_(
            dim=0, index=safe_ids, source=mask.int(),
        )
        return self._data[layer_idx]
    
    def on_local_expert_counts(self, layer_idx: int, local_expert_counts: torch.tensor):
        assert (not self._enable_global_physical_experts == True)
        self._data[layer_idx] += local_expert_counts
        return self._data[layer_idx]


class _DeepepNormalSinglePassGatherer(_LayerBasedCpuSinglePassGatherer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if torch.distributed.get_rank() == 0:
            logger.info(
                "DeepepNormalSinglePassGatherer gathers approximate statistics. "
                "If used with small batch size, consider using expert_distribution_recorder_mode=stat."
            )

    def on_deepep_dispatch_normal(
        self,
        layer_idx: int,
        local_physical_count_of_layer: List[int],
        num_tokens_per_rank,
        num_tokens_per_rdma_rank,
        num_tokens_per_expert,
    ):
        assert isinstance(local_physical_count_of_layer, list)
        self._on_layer_data(layer_idx, local_physical_count_of_layer)

    def collect(self) -> Dict:
        local_physical_count = super()._collect_objects(
            pad_len=self._expert_location_metadata.num_local_physical_experts
        )
        global_physical_count = _convert_local_to_global_physical_count(
            local_physical_count,
            rank=self._rank,
            num_local_physical_experts=self._expert_location_metadata.num_local_physical_experts,
            num_physical_experts=self._expert_location_metadata.num_physical_experts,
        )
        return dict(global_physical_count=global_physical_count)


class _DeepepLowLatencySinglePassGatherer(_LayerBasedGpuSinglePassGatherer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, enable_global_physical_experts=False)

    def on_deepep_dispatch_low_latency(
        self, layer_idx: int, local_physical_count_of_layer: torch.Tensor
    ):
        # Most naive implementation, can optimize later
        self._data[layer_idx, :] += local_physical_count_of_layer


def _convert_local_to_global_physical_count(
    local_physical_count: torch.Tensor,
    rank: int,
    num_local_physical_experts: int,
    num_physical_experts: int,
) -> torch.Tensor:
    dtype = local_physical_count.dtype
    device = local_physical_count.device
    num_layers, _ = local_physical_count.shape

    ans = torch.zeros((num_layers, num_physical_experts), dtype=dtype, device=device)
    ans[
        :, num_local_physical_experts * rank : num_local_physical_experts * (rank + 1)
    ] = local_physical_count
    return ans


# --- Unbalancedness Metrics ---

class BaseMetric(ABC):
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @abstractmethod
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """输入: [layers, experts], 输出: [layers]"""
        pass

class MetricRegistry:
    _registry: Dict[str, type] = {}

    @classmethod
    def register(cls, name: str):
        def decorator(subclass):
            cls._registry[name] = subclass
            return subclass
        return decorator

    @classmethod
    def create(cls, name: str, **kwargs) -> BaseMetric:
        return cls._registry[name](**kwargs)

@MetricRegistry.register("peak_to_mean")
class PeakToMean(BaseMetric):
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # x: [layers, experts]
        peak = x.max(dim=-1).values
        mean = x.mean(dim=-1)
        return peak / (mean + 1e-9)

@MetricRegistry.register("gini")
class GiniCoefficient(BaseMetric):
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        sorted_x, _ = torch.sort(x, dim=-1)
        n = x.size(-1)
        index = torch.arange(1, n + 1, device=x.device, dtype=x.dtype)
        numerator = 2 * torch.sum(index * sorted_x, dim=-1)
        denominator = n * torch.sum(sorted_x, dim=-1)
        return (numerator / (denominator + 1e-9)) - (n + 1) / n

@MetricRegistry.register("top_k_load")
class TopKLoadPercentage(BaseMetric):
    def __init__(self, k: int = 1, **kwargs):
        super().__init__(**kwargs)
        self.k = k

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        total = x.sum(dim=-1)
        top_k_vals = torch.topk(x, k=min(self.k, x.size(-1)), dim=-1).values
        return top_k_vals.sum(dim=-1) / (total + 1e-9)

@MetricRegistry.register("hoover")
class HooverIndex(BaseMetric):
    """
    Hoover 指数 (Robin Hood Index)
    衡量需要移动多少比例的负载才能达到平衡。范围 [0, 1]。
    """
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        total = x.sum(dim=-1)
        
        # sum(|xi - mean|) / (2 * sum(xi))
        diff_sum = torch.sum(torch.abs(x - mean), dim=-1)
        return 0.5 * diff_sum / (total + 1e-9)

@MetricRegistry.register("entropy")
class NormalizedEntropy(BaseMetric):
    """
    归一化熵 (Normalized Entropy)
    衡量分布的无序程度。范围 [0, 1]。
    0 = 绝对均匀 (最大熵), 1 = 绝对集中 (最小熵)
    注：为了符合"越大越不均"的直觉，这里返回 (1 - 归一化熵)
    """
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # 转换为概率分布
        probs = x / (x.sum(dim=-1, keepdim=True) + 1e-9)
        
        # 避免 log(0)
        probs = probs + 1e-12
        
        entropy = -torch.sum(probs * torch.log(probs), dim=-1)
        max_entropy = torch.log(torch.tensor(x.size(-1), device=x.device, dtype=x.dtype))
        
        # normalized_entropy = entropy / max_entropy (1是均匀，0是不均)
        return 1.0 - (entropy / max_entropy)


# --- 2. 结果封装 ---

class AnalysisResult:
    def __init__(self, results: Dict[str, torch.Tensor], num_layers: int):
        self._results = results
        self.num_layers = num_layers

    def __getitem__(self, key: str) -> torch.Tensor:
        return self._results[key]

    def __str__(self) -> str:
        """
        生成适合日志打印的表格。
        格式：
        Layer ID | Gini   | PeakMean | Top1Load
        ---------------------------------------
        0        | 0.1234 | 1.0500   | 0.0500
        1        | 0.8800 | 7.5000   | 0.6000
        """
        if not self._results:
            return "No Metrics"

        metric_names = list(self._results.keys())
        
        # 1. 定义列宽
        layer_col_width = 8
        metric_col_width = 12
        
        # 2. 构建表头
        header = f"{'Layer':<{layer_col_width}}" \
            + "".join([f"{name:>{metric_col_width}}" for name in metric_names])
        separator = "-" * len(header)
        
        lines = [separator, header, separator]

        # 3. 构建每一层的数据行
        for layer_idx in range(self.num_layers):
            row_str = f"{layer_idx:<{layer_col_width}}"
            for name in metric_names:
                # 获取该 metric 在该 layer 的值
                val = self._results[name][layer_idx].item()
                row_str += f"{val:>{metric_col_width}.4f}"
            lines.append(row_str)
            
        lines.append(separator)
        return "\n".join(lines)

    def summarization(self):
        if not self._results:
            return "No Metrics"

        metric_names = list(self._results.keys())
        summarization = f"[MetricsSummarization]:" \
            + "".join([f"{name}:{self._results[name].mean().item():.4f} " for name in metric_names])
        return summarization
    
    def dump(self) -> Dict[str, torch.Tensor]:
        """
        导出内部数据字典。
        用法: torch.save(result.dump(), 'metrics.pt')
        """
        return self._results

# --- 3. 分析器 (逻辑增强) ---

class MoEAnalyzer:
    def __init__(self):
        self.metrics: List[BaseMetric] = []

    def add_metric(self, metric_name: str, **kwargs) -> 'MoEAnalyzer':
        self.metrics.append(MetricRegistry.create(metric_name, **kwargs))
        return self

    def analyze(self, activations: torch.Tensor) -> AnalysisResult:
        """
        输入 activations: 
          - 2D Tensor [num_layers, num_experts] (标准情况)
          - 1D Tensor [num_experts] (会自动升维处理)
        """
        # 1. 维度检查与标准化
        x = activations.float()
        if x.dim() == 1:
            x = x.unsqueeze(0) # 变成 [1, num_experts]
        elif x.dim() != 2:
            raise ValueError(f"Expected 1D or 2D tensor, got {x.dim()}D")
            
        num_layers = x.size(0)
        results = {}

        # 2. 批量计算
        for metric in self.metrics:
            results[metric.name] = metric(x)
        
        return AnalysisResult(results, num_layers)

# --------------------------------------- Accumulator -----------------------------------------

_SINGLE_PASS_GATHERER_KEY_PRIMARY = "primary"


class _Accumulator(ABC):
    @staticmethod
    def init_new(
        server_args: ServerArgs,
        expert_location_metadata: "ExpertLocationMetadata",
        rank: int,
    ) -> "_Accumulator":
        return _Accumulator.get_class(server_args)(
            server_args, expert_location_metadata, rank
        )

    @staticmethod
    def get_class(server_args: ServerArgs) -> Type["_Accumulator"]:
        return {
            "stat": _StatAccumulator,
            "stat_approx": _StatAccumulator,
            "per_pass": _DetailAccumulator,
            "per_token": _DetailAccumulator,
        }[server_args.expert_distribution_recorder_mode]

    def __init__(
        self,
        server_args: ServerArgs,
        expert_location_metadata: "ExpertLocationMetadata",
        rank: int,
    ):
        self._server_args = server_args
        self._expert_location_metadata = expert_location_metadata
        self._rank = rank

    def get_single_pass_gatherer_keys(self):
        return [_SINGLE_PASS_GATHERER_KEY_PRIMARY]

    def get_single_pass_gatherer_key(self, debug_name: Optional[str]):
        return _SINGLE_PASS_GATHERER_KEY_PRIMARY

    def append(
        self,
        forward_pass_id: int,
        gatherer_key: str,
        single_pass_data: Dict,
    ):
        pass

    def reset(self):
        pass

    def dump(self, output_mode: _OutputMode):
        pass


class _UtilizationRateAccumulatorMixin(_Accumulator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._enable = self._server_args.enable_expert_distribution_metrics

        if self._enable:
            window_sizes = [10, 100, 1000]
            self._history = _DequeCollection(maxlens=window_sizes)
            self._rank = torch.distributed.get_rank()

    def append(
        self,
        forward_pass_id: int,
        gatherer_key: str,
        single_pass_data: Dict,
    ):
        super().append(forward_pass_id, gatherer_key, single_pass_data)
        if self._enable:
            self._append_utilization_rate(
                forward_pass_id, single_pass_data["global_physical_count"]
            )

    def reset(self):
        super().reset()
        if self._enable:
            self._history.clear()

    def _append_utilization_rate(
        self, forward_pass_id: int, single_pass_global_physical_count: torch.Tensor
    ):
        gpu_physical_count = compute_gpu_physical_count(
            single_pass_global_physical_count,
            num_gpu=self._expert_location_metadata.ep_size,
        )
        gpu_physical_count = gpu_physical_count.to(self._server_args.device)
        torch.distributed.reduce(
            gpu_physical_count, dst=0, op=torch.distributed.ReduceOp.SUM
        )

        if self._rank == 0:
            utilization_rate_tensor = compute_utilization_rate(gpu_physical_count)
            utilization_rate = torch.mean(utilization_rate_tensor).item()
            self._history.append(utilization_rate)

            gpu_physical_count_sum = gpu_physical_count.sum().item()

            logger.info(
                f"[Expert Balancedness] "
                f"forward_pass_id={forward_pass_id} "
                f"current_pass_balancedness={utilization_rate:.03f} "
                f"{''.join(f'last_{size}_average_balancedness={value:.03f} ' for size, value in self._history.mean().items())} "
                f"gpu_physical_count_sum={gpu_physical_count_sum}"
                # f"current_pass_per_layer={[round(x, 2) for x in utilization_rate_tensor.cpu().tolist()]}"
            )


class _DequeCollection:
    def __init__(self, maxlens: List[int]):
        self._dequeues = [deque(maxlen=maxlen) for maxlen in maxlens]

    def append(self, value):
        for d in self._dequeues:
            d.append(value)

    def clear(self):
        for d in self._dequeues:
            d.clear()

    def mean(self) -> Dict[int, float]:
        return {d.maxlen: sum(d) / len(d) for d in self._dequeues}


class _DetailAccumulator(_UtilizationRateAccumulatorMixin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._records = []

    def get_single_pass_gatherer_keys(self):
        if False:  # TODO `server_args.enable_two_batch_overlap`
            return [_SINGLE_PASS_GATHERER_KEY_PRIMARY, "child_a", "child_b"]
        return super().get_single_pass_gatherer_keys()

    def get_single_pass_gatherer_key(self, debug_name: Optional[str]):
        if False:  # TODO `server_args.enable_two_batch_overlap`
            return debug_name or _SINGLE_PASS_GATHERER_KEY_PRIMARY
        return super().get_single_pass_gatherer_key(debug_name)

    def append(
        self,
        forward_pass_id: int,
        gatherer_key: str,
        single_pass_data: Dict,
    ):
        super().append(forward_pass_id, gatherer_key, single_pass_data)

        def _process_object(obj):
            if isinstance(obj, torch.Tensor):
                return obj.cpu().clone()
            return obj

        single_pass_data_processed = {
            k: _process_object(v) for k, v in single_pass_data.items()
        }

        self._records.append(
            dict(
                forward_pass_id=forward_pass_id,
                rank=self._rank,
                gatherer_key=gatherer_key,
                **single_pass_data_processed,
            )
        )

    def reset(self):
        super().reset()
        self._records.clear()

    def dump(self, output_mode: _OutputMode):
        assert output_mode == "file"
        output = dict(
            records=self._records,
            # NOTE: This may change during recording, so here we say it is the "last" one
            last_physical_to_logical_map=self._expert_location_metadata.physical_to_logical_map,
        )
        _dump_to_file(
            f"expert_distribution_recorder_{time.time()}_{self._rank}.pt", output
        )


class _StatAccumulator(_UtilizationRateAccumulatorMixin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._use_running_sum_only = (
            self._server_args.enable_eplb
            and os.environ.get("SGLANG_EXPERT_DISTRIBUTION_RECORDER_USE_RUNNING_SUM_ONLY", "1") == "1"
        )
        self._global_physical_count_sum = torch.zeros(
            (
                self._expert_location_metadata.num_layers,
                self._expert_location_metadata.num_physical_experts,
            ),
            dtype=torch.int64,
            device=self._server_args.device,
        )
        self._global_physical_count_of_buffered_step = None
        if not self._use_running_sum_only:
            self._global_physical_count_of_buffered_step = _Buffer.init_new(
                item_shape=(
                    self._expert_location_metadata.num_layers,
                    # Cannot use local_physical_count to support select_experts
                    self._expert_location_metadata.num_physical_experts,
                ),
                buffer_size=self._server_args.expert_distribution_recorder_buffer_size,
                dtype=torch.int32,
                device=self._server_args.device,
            )
        self._first_dump = True
        self.history_stats = []

    def append(
        self,
        forward_pass_id: int,
        gatherer_key: str,
        single_pass_data: Dict,
    ):
        super().append(forward_pass_id, gatherer_key, single_pass_data)
        self._global_physical_count_sum += single_pass_data["global_physical_count"].to(
            self._global_physical_count_sum.dtype
        )
        if self._global_physical_count_of_buffered_step is not None:
            # Can optimize if overhead here is large
            self._global_physical_count_of_buffered_step.append(
                single_pass_data["global_physical_count"]
            )

    def reset(self):
        super().reset()
        self._global_physical_count_sum.zero_()
        if self._global_physical_count_of_buffered_step is not None:
            self._global_physical_count_of_buffered_step.reset()


    def _get_buffered_physical_count(self) -> torch.Tensor:
        if self._global_physical_count_of_buffered_step is not None:
            return self._global_physical_count_of_buffered_step.get_all()
        return self._global_physical_count_sum.unsqueeze(0)

    def dump(self, output_mode: _OutputMode):
        buffered_physical_count = self._get_buffered_physical_count()


        logical_count_of_buffered_step = _convert_global_physical_count_to_logical_count(
            buffered_physical_count,
            num_layers=self._expert_location_metadata.num_layers,
            num_logical_experts=self._expert_location_metadata.num_logical_experts,
            physical_to_logical_map=self._expert_location_metadata.physical_to_logical_map,
        )
        if self._use_running_sum_only:
            logical_count_of_buffered_step = logical_count_of_buffered_step.squeeze(0)

        if self._first_dump:
            self._first_dump = False
            get_device_module().empty_cache()

        torch.distributed.all_reduce(
            logical_count_of_buffered_step, op=torch.distributed.ReduceOp.SUM
        )

        global_count_of_buffered_step = self._global_physical_count_sum.clone()
        if not self._use_running_sum_only:
            global_count_of_buffered_step = buffered_physical_count.sum(dim=0)

        torch.distributed.all_reduce(
            global_count_of_buffered_step, op=torch.distributed.ReduceOp.SUM
        )
        analyzer = MoEAnalyzer() \
                .add_metric("peak_to_mean") \
                .add_metric("gini") \
                .add_metric("top_k_load", k=1) \
                .add_metric("entropy") \
                .add_metric("hoover")
        metrics_result = analyzer.analyze(global_count_of_buffered_step)
        per_gpu_metrics_result = analyzer.analyze(
            global_count_of_buffered_step \
            .clone() \
            .view(
                self._expert_location_metadata.num_layers,
                self._expert_location_metadata.num_physical_experts // self._expert_location_metadata.num_local_physical_experts,
                self._expert_location_metadata.num_local_physical_experts
            ) \
            .sum(dim=-1)
        )
        if self._rank == 0:
            logger.info(f"[METRICS][Unbalancedness metrics] {metrics_result.summarization()}")
            logger.info(f"[METRICS][Unbalancedness per gpu metrics] {per_gpu_metrics_result.summarization()}")
            # self.history_stats.append(dict(
            #     logic_count=logical_count_of_buffered_step.sum(dim=0),
            #     physical_count=global_count_of_buffered_step,
            #     metrics_result=metrics_result.dump(),
            # ))

        output = dict(
            rank=self._rank,
            logical_count=logical_count_of_buffered_step,
            physical_count=global_count_of_buffered_step,
            metrics=metrics_result.dump(),
            per_expert_metrics=metrics_result['GiniCoefficient'].mean().item(),
            per_device_metrics=per_gpu_metrics_result['GiniCoefficient'].mean().item(),
        )

        if output_mode == "file":
            if self._rank == 0:
                output['historys'] = self.history_stats
                _dump_to_file(f"expert_distribution_recorder_{time.time()}.pt", output)
        elif output_mode == "object":
            return output
        else:
            raise NotImplementedError


def _dump_to_file(name, data):
    save_dir = Path(os.environ.get("SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR", "/tmp"))
    path_output = save_dir / name
    logger.info(f"Write expert distribution to {path_output}")
    if not save_dir.exists():
        save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(data, str(path_output))


class _Buffer:
    @staticmethod
    def init_new(item_shape: Tuple, buffer_size: int, dtype, device):
        if buffer_size < 0:
            return _InfiniteBuffer(item_shape, dtype=dtype, device=device)
        else:
            return _CircularBuffer(item_shape, buffer_size, dtype=dtype, device=device)

    def append(self, value: torch.Tensor):
        raise NotImplementedError

    def get_all(self) -> torch.Tensor:
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError


class _CircularBuffer(_Buffer):
    def __init__(self, item_shape: Tuple, buffer_size: int, dtype, device):
        self._buffer = torch.zeros(
            (buffer_size, *item_shape), dtype=dtype, device=device
        )
        self._curr_index = 0

    def append(self, value: torch.Tensor):
        self._buffer[self._curr_index] = value
        self._curr_index = (self._curr_index + 1) % len(self._buffer)

    def get_all(self) -> torch.Tensor:
        return self._buffer

    def reset(self):
        self._buffer[...] = 0


class _InfiniteBuffer(_Buffer):
    def __init__(self, item_shape: Tuple, dtype, device):
        self._item_shape = item_shape
        self._buffer = torch.zeros((128, *item_shape), dtype=dtype, device=device)
        self._size = 0

    def append(self, value: torch.Tensor):
        curr_buffer_size = len(self._buffer)
        dtype = self._buffer.dtype
        device = self._buffer.device

        if self._size == curr_buffer_size:
            new_buffer = torch.zeros(
                (2 * curr_buffer_size, *self._item_shape), dtype=dtype, device=device
            )
            new_buffer[:curr_buffer_size] = self._buffer
            self._buffer = new_buffer

        self._buffer[self._size] = value
        self._size += 1

    def get_all(self) -> torch.Tensor:
        return self._buffer[: self._size]

    def reset(self):
        self._buffer[...] = 0
        self._size = 0


def _convert_global_physical_count_to_logical_count(
    # (whatever, num_layers, num_physical_experts)
    global_physical_count: torch.Tensor,
    num_layers: int,
    num_logical_experts: int,
    physical_to_logical_map: torch.Tensor,
):
    dim_extra, _, _ = global_physical_count.shape
    dtype = global_physical_count.dtype
    device = global_physical_count.device
    logical_count = torch.zeros(
        (dim_extra, num_layers, num_logical_experts), dtype=dtype, device=device
    )
    logical_count.scatter_add_(
        dim=2,
        index=physical_to_logical_map.unsqueeze(0)
        .expand(dim_extra, -1, -1)
        .to(torch.int64),
        src=global_physical_count,
    )
    return logical_count


def compute_gpu_physical_count(
    physical_count_of_whatever: torch.Tensor,  # (..., num_layer, num_physical_expert)
    num_gpu: int,
):
    """output: gpu_physical_count_of_batch (..., num_layer, num_gpu)"""
    return einops.reduce(
        physical_count_of_whatever,
        "... num_layer (num_gpu num_expert_per_gpu) -> ... num_layer num_gpu",
        "sum",
        num_gpu=num_gpu,
    )


def compute_utilization_rate(
    gpu_physical_count_of_batch: torch.Tensor,  # (..., num_layer, num_gpu)
):
    """output: utilization_rate (..., num_layer)"""
    gpu_physical_count_of_batch = gpu_physical_count_of_batch.float()
    max_gpu_physical_count = einops.reduce(
        gpu_physical_count_of_batch,
        "... num_layer num_gpu -> ... num_layer",
        "max",
    )
    avg_gpu_physical_count = einops.reduce(
        gpu_physical_count_of_batch,
        "... num_layer num_gpu -> ... num_layer",
        "mean",
    )
    return (avg_gpu_physical_count + 1e-5) / (max_gpu_physical_count + 1e-5)