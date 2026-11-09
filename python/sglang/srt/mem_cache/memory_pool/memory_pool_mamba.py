from __future__ import annotations

import torch

from sglang.srt.mem_cache.memory_pool.memory_pool_kernel import GB, get_tensor_size_bytes
from sglang.srt.utils import get_colorful_logger

logger = get_colorful_logger(__name__)


class MambaPool:
    def __init__(
        self,
        size: int,
        conv_dtype: torch.dtype,
        ssm_dtype: torch.dtype,
        num_mamba_layers: int,
        conv_state_shape: tuple[int, int],
        temporal_state_shape: tuple[int, int],
        device: str,
        speculative_num_draft_tokens: int | None = None,
    ):
        self.is_kda_cache = isinstance(conv_state_shape, list)
        if self.is_kda_cache:
            conv_state = [
                torch.zeros(
                    size=(num_mamba_layers, size + 1) + conv_shape,
                    dtype=conv_dtype,
                    device=device,
                )
                for conv_shape in conv_state_shape
            ]
        else:
            assert conv_state_shape[0] > conv_state_shape[1]
            conv_state = torch.zeros(
                size=(num_mamba_layers, size + 1) + conv_state_shape,
                dtype=conv_dtype,
                device=device,
            )

        temporal_state = torch.zeros(
            size=(num_mamba_layers, size + 1) + temporal_state_shape,
            dtype=ssm_dtype,
            device=device,
        )
        if speculative_num_draft_tokens is not None:
            if self.is_kda_cache:
                intermediate_conv_window_cache = [
                    torch.zeros(
                        size=(
                            num_mamba_layers,
                            size + 1,
                            speculative_num_draft_tokens,
                            conv_shape[0],
                            conv_shape[1],
                        ),
                        dtype=conv_dtype,
                        device="cuda",
                    )
                    for conv_shape in conv_state_shape
                ]
            else:
                intermediate_conv_window_cache = torch.zeros(
                    size=(
                        num_mamba_layers,
                        size + 1,
                        speculative_num_draft_tokens,
                        conv_state_shape[0],
                        conv_state_shape[1],
                    ),
                    dtype=conv_dtype,
                    device="cuda",
                )
            intermediate_ssm_state_cache = torch.empty(
                size=(
                    num_mamba_layers,
                    size + 1,
                    speculative_num_draft_tokens,
                    temporal_state_shape[0],
                    temporal_state_shape[1],
                    temporal_state_shape[2],
                ),
                dtype=ssm_dtype,
                device="cuda",
            )
            self.mamba_cache = (
                conv_state,
                temporal_state,
                intermediate_ssm_state_cache,
                intermediate_conv_window_cache,
            )
            logger.info(
                f"Mamba Cache is allocated. "
                f"conv_state size: {get_tensor_size_bytes(conv_state) / GB:.2f}GB, "
                f"ssm_state size: {get_tensor_size_bytes(temporal_state) / GB:.2f}GB "
                f"intermediate_ssm_state_cache size: {get_tensor_size_bytes(intermediate_ssm_state_cache) / GB:.2f}GB "
                f"intermediate_conv_window_cache size: {get_tensor_size_bytes(intermediate_conv_window_cache) / GB:.2f}GB "
            )
        else:
            self.mamba_cache = (conv_state, temporal_state)
            logger.info(
                f"Mamba Cache is allocated. "
                f"conv_state size: {get_tensor_size_bytes(conv_state) / GB:.2f}GB, "
                f"ssm_state size: {get_tensor_size_bytes(temporal_state) / GB:.2f}GB "
            )
        self.size = size
        self.free_slots = list(range(size))
        self.mem_usage = self.get_mamba_size() / GB

    def get_mamba_params_all_layers(self):
        return [self.mamba_cache[i] for i in range(len(self.mamba_cache))]

    def get_mamba_params(self, layer_id: int):
        return [self.mamba_cache[i][layer_id] for i in range(len(self.mamba_cache))]

    def get_mamba_size(self):
        return sum(get_tensor_size_bytes(t) for t in self.mamba_cache)

    def mamba2_layer_cache(self, layer_id: int):
        return self.at_layer_idx(layer_id)

    def at_layer_idx(self, layer: int):
        if isinstance(self.mamba_cache[0], list):
            return {
                "conv": [v[layer] for v in self.mamba_cache[0]],
                "temporal": self.mamba_cache[1][layer],
            }
        return {
            "conv": self.mamba_cache[0][layer],
            "temporal": self.mamba_cache[1][layer],
        }

    def available_size(self):
        return len(self.free_slots)

    def alloc(self, need_size: int) -> list[int] | None:
        if need_size > len(self.free_slots):
            return None

        select_index = self.free_slots[:need_size]
        self.free_slots = self.free_slots[need_size:]

        return select_index

    def free(self, free_index: int | list[int]):
        if isinstance(free_index, (int,)):
            self.free_slots.append(free_index)
        else:
            self.free_slots.extend(free_index)
        if self.is_kda_cache:
            for i in range(len(self.mamba_cache[0])):
                self.mamba_cache[0][i][:, free_index] = 0
        else:
            self.mamba_cache[0][:, free_index] = 0
        self.mamba_cache[1][:, free_index] = 0

    def clear(self):
        self.free_slots = list(range(self.size))
