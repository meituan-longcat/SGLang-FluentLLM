from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Optional, Union

import torch


class EngineBase(ABC):
    """
    Abstract base class for engine interfaces that support generation, weight updating, and memory control.
    This base class provides a unified API for both HTTP-based engines and engines.
    """

    @abstractmethod
    def generate(
        self,
        prompt: Optional[Union[list[str], str]] = None,
        sampling_params: Optional[Union[list[dict], dict]] = None,
        input_ids: Optional[Union[list[list[int]], list[int]]] = None,
        image_data: Optional[Union[list[str], str]] = None,
        return_logprob: Optional[Union[list[bool], bool]] = False,
        logprob_start_len: Optional[Union[list[int], int]] = None,
        top_logprobs_num: Optional[Union[list[int], int]] = None,
        token_ids_logprob: Optional[Union[list[list[int]], list[int]]] = None,
        custom_logit_processor: Optional[Union[list[str], str]] = None,
        return_hidden_states: Optional[bool] = None,
        stream: Optional[bool] = None,
        bootstrap_host: Optional[Union[list[str], str]] = None,
        bootstrap_port: Optional[Union[list[int], int]] = None,
        bootstrap_room: Optional[Union[list[int], int]] = None,
        data_parallel_rank: Optional[int] = None,
    ) -> Union[dict, Iterator[dict]]:
        """Generate outputs based on given inputs."""
        pass

    @abstractmethod
    def flush_cache(self):
        """Flush the cache of the engine."""
        pass

    @abstractmethod
    def update_weights_from_tensor(
        self,
        named_tensors: list[tuple[str, torch.Tensor]],
        load_format: Optional[str] = None,
        flush_cache: bool = True,
    ):
        """Update model weights with in-memory tensor data."""
        pass

    @abstractmethod
    def release_memory_occupation(self):
        """Release GPU memory occupation temporarily."""
        pass

    @abstractmethod
    def resume_memory_occupation(self):
        """Resume GPU memory occupation which is previously released."""
        pass

    @abstractmethod
    def shutdown(self):
        """Shutdown the engine and clean up resources."""
        pass
