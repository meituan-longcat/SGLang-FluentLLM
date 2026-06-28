import gc

from sglang.srt.distributed import get_tp_group
from sglang.srt.utils import get_colorful_logger, is_npu
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.distributed as dist

from sglang.srt.configs.device_config import DeviceConfig
from sglang.srt.configs.load_config import LoadConfig

from sglang.srt.model_loader import get_model
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.patch_torch import monkey_patch_torch_reductions
from sglang.srt.utils import (
    MultiprocessingSerializer,
    get_available_gpu_memory,
    init_custom_process_group,
    set_cuda_arch,
)
from sglang.srt.constants import (
    GPU_MEMORY_ALL_TYPES,
    GPU_MEMORY_TYPE_CUDA_GRAPH,
    GPU_MEMORY_TYPE_KV_CACHE,
    GPU_MEMORY_TYPE_WEIGHTS
)

logger = get_colorful_logger(__name__)


class WeightMixin:
    def load_model(self, draft_model_idx: int = None):
        logger.info(
            f"Load weight begin. avail mem={get_available_gpu_memory(self.device, self.gpu_id):.2f} GB"
        )

        # This can reduce thread conflicts and speed up weight loading.
        if self.device != "cpu":
            torch.set_num_threads(1)
        if self.device == "cuda":
            if torch.cuda.get_device_capability()[0] < 8:
                logger.info(
                    "Compute capability below sm80. Use float16 due to lack of bfloat16 support."
                )
                self.server_args.dtype = "float16"
                self.model_config.dtype = torch.float16
                if torch.cuda.get_device_capability()[1] < 5:
                    raise RuntimeError("SGLang only supports sm75 and above.")

        set_cuda_arch()

        # Prepare the model config
        self.load_config = LoadConfig(
            load_format=self.server_args.load_format,
            download_dir=self.server_args.download_dir,
            ext_yaml=self.server_args.ext_yaml,
            draft_model_idx=draft_model_idx
        )

        with self.memory_saver_adapter.region(tag=GPU_MEMORY_TYPE_WEIGHTS):
            self.model = get_model(
                model_config=self.model_config,
                load_config=self.load_config,
                device_config=DeviceConfig(self.device),
            )

        if self.server_args.kv_cache_dtype == "fp8_e4m3":
            if self.server_args.quantization_param_path is not None:
                if callable(getattr(self.model, "load_kv_cache_scales", None)):
                    self.model.load_kv_cache_scales(
                        self.server_args.quantization_param_path
                    )
                    logger.info(
                        "Loaded KV cache scaling factors from %s",
                        self.server_args.quantization_param_path,
                    )
                else:
                    raise RuntimeError(
                        "Using FP8 KV cache and scaling factors provided but "
                        "model %s does not support loading scaling factors.",
                        self.model.__class__,
                    )
            else:
                logger.warning(
                    "Using FP8 KV cache but no scaling factors "
                    "provided. Defaulting to scaling factors of 1.0. "
                    "This may lead to less accurate results!"
                )

        # Parse other args
        self.sliding_window_size = (
            self.model.get_attention_sliding_window_size()
            if hasattr(self.model, "get_attention_sliding_window_size")
            else None
        )
        self.dtype = self.model_config.dtype

        def print_model_weights_memory(model):
            """打印模型所有参数的权重占用"""
            total_params=0
            total_memory=0

            print("="*80)
            print(f"{'参数名':<60} {'参数量':>12} {'大小(MB)':>12}")
            print("="*80)

            for name, param in model.named_parameters():
                param_count=param.numel()
                param_memory=param.numel()*param.element_size()  # 字节数

                total_params+=param_count
                total_memory+=param_memory

                print(f"{name:<60} {param_count:>12,} {param_memory/1024**2:>12.2f}")

            print("="*80)
            print(f"{'总计':<60} {total_params:>12,} {total_memory/1024**2:>12.2f}")
            print(f"总参数量: {total_params:,}")
            print(f"总显存占用: {total_memory/1024**2:.2f} MB ({total_memory/1024**3:.2f} GB)")
            print("="*80)

        if is_npu():
            torch.npu.empty_cache()
            torch.npu.synchronize()
            if get_tp_group().is_first_rank:
                print_model_weights_memory(self.model)

        logger.info(
            f"Load weight end. "
            f"type={type(self.model).__name__}, "
            f"dtype={self.dtype}, "
            f"avail mem={get_available_gpu_memory(self.device, self.gpu_id):.2f} GB"
        )

    def update_weights_from_disk(
        self, model_path: str, load_format: str
    ) -> tuple[bool, str]:
        """Update engine weights in-place from the disk."""
        from sglang.srt.model_loader.loader import (
            DefaultModelLoader,
            device_loading_context,
            get_model_loader,
        )
        from sglang.srt.model_loader.utils import set_default_torch_dtype

        logger.info(
            f"Update engine weights online from disk begin. "
            f"avail mem={get_available_gpu_memory(self.device, self.gpu_id):.2f} GB"
        )

        target_device = torch.device(self.device)
        self.model_config.model_path = model_path
        load_config = LoadConfig(load_format=load_format)

        # Only support vllm DefaultModelLoader for now
        loader = get_model_loader(load_config)
        if not isinstance(loader, DefaultModelLoader):
            message = f"Failed to get model loader: {loader}."
            return False, message

        def get_weight_iter(config):
            iter = loader._get_weights_iterator(
                DefaultModelLoader.Source(
                    config.model_path,
                    revision=config.revision,
                    fall_back_to_pt=getattr(
                        self.model, "fall_back_to_pt_during_load", True
                    ),
                )
            )
            return iter

        def model_load_weights(model, iter):
            model.load_weights(iter)
            for _, module in self.model.named_modules():
                quant_method = getattr(module, "quant_method", None)
                if quant_method is not None:
                    with device_loading_context(module, target_device):
                        quant_method.process_weights_after_loading(module)
                process_method = getattr(module, "process_weights_after_loading", None)
                if process_method is not None:
                    with device_loading_context(module, target_device):
                        module.process_weights_after_loading(module)
            return model

        with set_default_torch_dtype(self.model_config.dtype):
            try:
                iter = get_weight_iter(self.model_config)
            except Exception as e:
                message = f"Failed to get weights iterator: {e}."
                return False, message
            try:
                model = model_load_weights(self.model, iter)
            except Exception as e:
                message = (
                    f"Failed to update weights: {e}.\nRolling back to original weights."
                )
                del iter
                gc.collect()
                iter = get_weight_iter(self.model_config)
                self.model = model_load_weights(self.model, iter)
                return False, message

        self.model = model
        self.server_args.model_path = model_path
        self.server_args.load_format = load_format
        self.load_config = load_config

        logger.info("Update weights end.")
        return True, "Succeeded to update model weights."

    def init_weights_update_group(
        self,
        master_address,
        master_port,
        rank_offset,
        world_size,
        group_name,
        backend="nccl",
    ):
        """Initialize the Torch process group for model parameter updates.

        `_model_update_group` is used in the RLHF workflow, where rank
        0 is the actor model in the training engine, and the other ranks are
        the inference engine, which is used for rollout.

        In the RLHF workflow, the training engine updates the model
        weights/parameters online, and broadcasts them to the inference
        engine through the `_model_update_group` process group.
        """
        assert (
            torch.distributed.is_initialized()
        ), "Default torch process group must be initialized"
        assert group_name != "", "Group name cannot be empty"

        rank = rank_offset + self.tp_rank

        logger.info(
            f"init custom process group: master_address={master_address}, master_port={master_port}, "
            f"rank_offset={rank_offset}, rank={rank}, world_size={world_size}, group_name={group_name}, backend={backend}"
        )

        try:
            self._model_update_group = init_custom_process_group(
                backend=backend,
                init_method=f"tcp://{master_address}:{master_port}",
                world_size=world_size,
                rank=rank,
                group_name=group_name,
            )
            dist.barrier(group=self._model_update_group, device_ids=[rank])
            return True, "Succeeded to initialize custom process group."
        except Exception as e:
            message = f"Failed to initialize custom process group: {e}."
            logger.error(message)
            return False, message

    def update_weights_from_distributed(self, name, dtype, shape):
        """
        Update specific parameter in the model weights online
        through `_model_update_group` process group.

        Args:
            name: the name of the parameter to be updated.
            dtype: the data type of the parameter to be updated.
            shape: the shape of the parameter to be updated.
        """
        target_dtype = (
            dtype if isinstance(dtype, torch.dtype) else getattr(torch, dtype)
        )

        assert (
            self._model_update_group is not None
        ), "model update group must be initialized"

        try:
            weights = torch.empty(shape, dtype=target_dtype, device=self.device)
            torch.distributed.broadcast(weights, src=0, group=self._model_update_group)
            self.model.load_weights([(name, weights)])
            return True, f"Succeeded to update parameter {name} online."

        except Exception as e:
            error_msg = (
                f"Failed to update parameter online: {e}. "
                f"The full weights of the ModelRunner are partially updated. "
                f"Please discard the whole weights."
            )
            logger.error(error_msg)
            return False, error_msg

    def update_weights_from_tensor(
        self,
        named_tensors: List[Tuple[str, Union[torch.Tensor, "LocalSerializedTensor"]]],
        load_format: Optional[str] = None,
    ):
        monkey_patch_torch_reductions()
        # We need to get device after patch otherwise the device would be wrong
        self.device_module = torch.get_device_module(self.device)
        infered_device = self.device_module.current_device()

        named_tensors = [
            (name, _unwrap_tensor(tensor, tp_rank=self.global_rank, device=infered_device))
            for name, tensor in named_tensors
        ]
        if load_format == "direct":
            _model_load_weights_direct(self.model, named_tensors)
        elif load_format is None:
            self.model.load_weights(named_tensors)
        else:
            raise NotImplementedError(f"Unknown load_format={load_format}")
        return True, "Success"

    def get_weights_by_name(
        self, name: str, truncate_size: int = 100
    ) -> Optional[torch.Tensor]:
        """Get the weights of the parameter by its name. Similar to `get_parameter` in Hugging Face.

        Only used for unit test with an unoptimized performance.
        For optimized performance, please use torch.save and torch.load.
        """
        # TODO: (chenyang) Add support for Qwen models.
        try:
            return self.model.get_weights_by_name(
                name, truncate_size, tp_size=self.tp_size
            )
        except Exception as e:
            logger.error(f"Error when getting parameter {name}: {e}")
            return None

def _model_load_weights_direct(model, named_tensors: List[Tuple[str, torch.Tensor]]):
    params_dict = dict(model.named_parameters())
    for name, tensor in named_tensors:
        default_weight_loader(params_dict[name], tensor)


def _unwrap_tensor(tensor, tp_rank, device):
    if isinstance(tensor, LocalSerializedTensor):
        tensor = tensor.get(tp_rank)
    return tensor.to(device)


@dataclass
class LocalSerializedTensor:
    """torch.Tensor that gets serialized by MultiprocessingSerializer (which only serializes a pointer and not the data).
    The i-th element in the list corresponds to i-th rank's GPU."""

    values: List[bytes]

    def get(self, rank: int):
        return MultiprocessingSerializer.deserialize(self.values[rank])

def unwrap_ipc_tensors(
    tensors: List[Tuple[str, torch.Tensor]], tp_rank: int, device: torch.device
) -> List[Tuple[str, torch.Tensor]]:
    """Unwrap IPC (Inter-Process Communication) tensors and move them to the target device.

    This function handles tensors that may be serialized using IPC handles and converts them
    to regular tensors on the specified device. It's primarily used in distributed training
    scenarios where tensors need to be shared across processes.

    Args:
        tensors: List of tuples containing tensor names and tensor objects.
        tp_rank: Tensor parallelism rank to retrieve the correct tensor from LocalSerializedTensor.
        device: Target device to move tensors to (e.g., 'cuda:0', 'cpu').

    Returns:
        List of tuples with the same structure as input, but with tensors moved to target device.
    """
    result = []
    for name, tensor in tensors:
        if isinstance(tensor, LocalSerializedTensor):
            # Deserialize the IPC tensor for the current rank
            ipc_tensor = tensor.get(tp_rank)
            # Move the deserialized tensor to the target device
            tensor_on_device = ipc_tensor.to(device)
            # Explicitly delete the tensor created from the IPC handle to trigger
            # garbage collection and release the handle, freeing IPC resources.
            del ipc_tensor
        else:
            # For regular tensors, simply move them to the target device
            tensor_on_device = tensor.to(device)
        # Append the processed tensor to the result list
        result.append((name, tensor_on_device))
    return result
