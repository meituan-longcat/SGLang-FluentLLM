import os
import tempfile

import torch

from sglang.srt.distributed import get_tensor_model_parallel_world_size
from sglang.srt.distributed.parallel_strategy import DenseParallelStategy
from sglang.srt.env import global_server_args_dict
from sglang.srt.layers.attention.dp_attention import (
    get_attention_dp_rank,
    get_attention_dp_size,
    get_attention_tp_rank,
    get_attention_tp_size,
)
from sglang.srt.utils import get_colorful_logger

logger = get_colorful_logger(__name__)


class ModelTensorTracer:
    def __init__(self):
        self.dp_rank = get_attention_dp_rank()
        self.tp_rank = get_attention_tp_rank()
        self.tp_size = get_attention_tp_size()
        self.dump_tensor_file_path = os.path.join(
            os.getenv("DUMP_TENSOR_DIR", tempfile.gettempdir()), f"fluentllm_model_tensor_tp{self.tp_rank}-{self.tp_size}.pt"
        )
        self.dump_tensor_stats: dict = {}

        self.local_tensor_stats = {}
        if self.dp_rank == 0:
            load_local_tensor_path = os.getenv("LOAD_LOCAL_TENSOR_DIR", None)
            if load_local_tensor_path:
                self.load_local_tensor_file_path = None
                if os.path.isfile(load_local_tensor_path):
                    # Load single pt file, used for comparison with huggingface, only supports pure dp except for moe, does not support tp
                    assert get_attention_dp_size() == get_tensor_model_parallel_world_size(), (
                        "IF you enable ENABLE_DUMP_TENSOR and load single *.pt, the attention dp size must be equal to the world size."
                    )
                    assert global_server_args_dict["dense_parallel_strategy"] == DenseParallelStategy.REPLICATED, (
                        "IF you enable ENABLE_DUMP_TENSOR and load single *.pt, the dense_parallel_strategy must be 'rep'."
                    )
                    self.load_local_tensor_file_path = load_local_tensor_path

                if os.path.isdir(load_local_tensor_path):
                    # If loading multiple files, each tp load corresponds to tp_rank file
                    load_local_tensor_file_list = os.listdir(load_local_tensor_path)
                    for local_tensor_file in load_local_tensor_file_list:
                        if f"_tp{self.tp_rank}-{self.tp_size}.pt" in local_tensor_file:
                            self.load_local_tensor_file_path = os.path.join(load_local_tensor_path, local_tensor_file)
                            break

                if self.load_local_tensor_file_path is None:
                    raise FileNotFoundError(f"The file '{load_local_tensor_path}' does not exist or is not a valid path.")

                try:
                    self.local_tensor_stats = torch.load(self.load_local_tensor_file_path)
                    logger.info(f"Load load_local_tensor from {self.load_local_tensor_file_path}")
                except Exception as e:
                    raise RuntimeError(f"Failed to load local tensor from '{self.load_local_tensor_file_path}': {e}") from e

        self.dump_tensor_stats_keys = list(self.dump_tensor_stats.keys())
        self.local_tensor_stats_keys = list(self.local_tensor_stats.keys())

    def __len___(self):
        return len(self.dump_tensor_stats)

    def trace(self, tensor_name: str, tensor: torch.Tensor):
        assert os.environ.get("ENABLE_DUMP_TENSOR", "0") == "1", "Please enable ENABLE_DUMP_TENSOR"
        if self.dp_rank == 0:
            self.refresh_keys()
            assert tensor_name not in self.dump_tensor_stats_keys, (
                f"Tensor '{tensor_name}' already exists in dump_tensor_stats. Available keys: {self.dump_tensor_stats_keys}"
            )
            self.dump_tensor_stats[tensor_name] = tensor.clone().to("cpu")
            if len(self.local_tensor_stats_keys) != 0:
                return self.replace_tensor(tensor_name, tensor)
            else:
                return tensor
        else:
            return tensor

    def dump(self):
        if self.dp_rank == 0:
            self.refresh_keys()
            torch.save(self.dump_tensor_stats, self.dump_tensor_file_path)
            logger.info(f"Dump the model_tensor is {self.dump_tensor_stats_keys}")
            logger.info(f"Dump the model_tensor to {self.dump_tensor_file_path}")

    def get_tensor(self, tensor_name: str, pop=False):
        self.refresh_keys()
        assert tensor_name in self.dump_tensor_stats_keys, (
            f"The {tensor_name} not in self.dump_tensor_stats. Available keys: {self.dump_tensor_stats_keys}"
        )
        if pop:
            return self.dump_tensor_stats.pop(tensor_name)
        else:
            return self.dump_tensor_stats[tensor_name]

    def get_local_tensor(self, tensor_name: str, pop=True):
        assert len(self.local_tensor_stats_keys) != 0, "The local_tensor is None, please check the env LOAD_LOCAL_TENSOR_DIR"
        assert tensor_name in self.local_tensor_stats_keys, (
            f"The {tensor_name} not in self.local_tensor. Available keys: {self.local_tensor_stats_keys}"
        )
        if pop:
            return self.local_tensor_stats.pop(tensor_name)
        else:
            return self.local_tensor_stats[tensor_name]

    def replace_tensor(self, tensor_name: str, tensor: torch.Tensor, pop=False):
        self.refresh_keys()
        if tensor_name in self.local_tensor_stats_keys:
            local_tensor = self.get_local_tensor(tensor_name, pop).squeeze()
            local_shape = local_tensor.shape
            shape = tensor.shape
            assert local_shape == shape, f"The shape of {tensor_name} is not equal, local_shape: {local_shape}, shape: {shape}"
            device = tensor.device
            dtype = tensor.dtype
            del tensor
            return local_tensor.to(dtype).to(device).contiguous()
        else:
            return tensor

    def refresh_keys(self):
        self.dump_tensor_stats_keys = list(self.dump_tensor_stats.keys())
        self.local_tensor_stats_keys = list(self.local_tensor_stats.keys())


global MODEL_TENSOR_TRACER
MODEL_TENSOR_TRACER = ModelTensorTracer()


def get_model_tensor_tracer():
    global MODEL_TENSOR_TRACER
    return MODEL_TENSOR_TRACER


global LOAD_NUMBER_LAYERS
LOAD_NUMBER_LAYERS = int(os.environ.get("LOAD_NUMBER_LAYERS", "0"))


def get_load_number_layers():
    global LOAD_NUMBER_LAYERS
    return LOAD_NUMBER_LAYERS
