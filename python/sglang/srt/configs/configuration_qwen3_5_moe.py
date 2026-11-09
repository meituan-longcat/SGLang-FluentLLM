# Copyright 2024 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ruff: noqa: N801

"""Qwen3Hybrid model configuration"""

import enum
import os

import numpy as np
import torch
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

from sglang.srt.utils.common import divide

logger = logging.get_logger(__name__)


# NOTE: HybridLayerType
class HybridLayerType(enum.Enum):
    full_attention = "attention"
    swa_attention = "swa_attention"
    linear_attention = "linear_attention"
    mamba2 = "mamba"


class Qwen3_5MoeTextConfig(PretrainedConfig):
    model_type = "qwen3_5_moe_text"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=151936,
        hidden_size=2048,
        intermediate_size=5632,
        num_hidden_layers=48,
        num_attention_heads=16,
        num_key_value_heads=2,
        hidden_act="silu",
        max_position_embeddings=32768,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        rope_parameters=None,
        partial_rotary_factor=0.25,
        attention_bias=False,
        attention_dropout=0.0,
        head_dim=256,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_num_key_heads=16,
        linear_num_value_heads=32,
        decoder_sparse_step=1,
        moe_intermediate_size=512,
        shared_expert_intermediate_size=512,
        num_experts_per_tok=10,
        num_experts=512,
        norm_topk_prob=True,
        output_router_logits=False,
        router_aux_loss_coef=0.001,
        mlp_only_layers=None,
        layer_types=None,
        **kwargs,
    ):
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.rope_parameters = rope_parameters
        self.partial_rotary_factor = partial_rotary_factor
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.head_dim = head_dim

        # linear attention (gdn now part)
        self.layer_types = layer_types
        self.linear_conv_kernel_dim = linear_conv_kernel_dim
        self.linear_key_head_dim = linear_key_head_dim
        self.linear_value_head_dim = linear_value_head_dim
        self.linear_num_key_heads = linear_num_key_heads
        self.linear_num_value_heads = linear_num_value_heads

        # MoE arguments
        self.decoder_sparse_step = decoder_sparse_step
        self.moe_intermediate_size = moe_intermediate_size
        self.shared_expert_intermediate_size = shared_expert_intermediate_size
        self.num_experts_per_tok = num_experts_per_tok
        self.num_experts = num_experts
        self.norm_topk_prob = norm_topk_prob
        self.output_router_logits = output_router_logits
        self.router_aux_loss_coef = router_aux_loss_coef
        self.mlp_only_layers = mlp_only_layers if mlp_only_layers is not None else []

    @property
    def layers_block_type(self):
        layer_type_list = []

        for layer_idx in range(self.num_hidden_layers):
            if hasattr(self, "full_attention_interval") and self.full_attention_interval == 1:
                layer_type_list.append(HybridLayerType.full_attention.value)
            else:
                if self.layer_types[layer_idx] == "full_attention":
                    self.layer_types[layer_idx] = HybridLayerType.full_attention.value
                layer_type_list.append(self.layer_types[layer_idx])

        return layer_type_list

    @property
    def linear_layer_ids(self):
        return [i for i, type_value in enumerate(self.layers_block_type) if type_value == HybridLayerType.linear_attention.value]

    @property
    def full_attention_layer_ids(self):
        return [i for i, type_value in enumerate(self.layers_block_type) if type_value == HybridLayerType.full_attention.value]

    @property
    def mamba2_cache_params(self):
        from sglang.srt.layers.attention.dp_attention import get_attention_tp_size

        world_size = get_attention_tp_size()
        conv_dim = self.linear_key_head_dim * self.linear_num_key_heads * 2 + self.linear_value_head_dim * self.linear_num_value_heads
        conv_state_shape = (
            divide(conv_dim, world_size),
            self.linear_conv_kernel_dim - 1,
        )

        temporal_state_shape = (
            divide(self.linear_num_value_heads, world_size),
            self.linear_key_head_dim,
            self.linear_value_head_dim,
        )
        conv_dtype = torch.bfloat16
        dtype_map = {
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
        }
        ssm_dtype = dtype_map[os.environ["SGLANG_MAMBA_SSM_DTYPE"]]
        mamba_layers = self.linear_layer_ids
        return (
            conv_state_shape,
            temporal_state_shape,
            conv_dtype,
            ssm_dtype,
            mamba_layers,
        )

    @property
    def mamba_cache_per_req(self):
        conv_state_shape, temporal_state_shape, conv_dtype, ssm_dtype, mamba_layers = self.mamba2_cache_params
        mamba_layers_len = len(mamba_layers)

        return (int(np.prod(conv_state_shape)) * conv_dtype.itemsize + int(np.prod(temporal_state_shape)) * ssm_dtype.itemsize) * mamba_layers_len


class Qwen3_5MoeVisionConfig(PretrainedConfig):
    model_type = "qwen3_5_moe"
    base_config_key = "vision_config"

    def __init__(
        self,
        depth=27,
        hidden_size=1152,
        hidden_act="gelu_pytorch_tanh",
        intermediate_size=4304,
        num_heads=16,
        in_channels=3,
        patch_size=16,
        spatial_merge_size=2,
        temporal_patch_size=2,
        out_hidden_size=3584,
        num_position_embeddings=2304,
        initializer_range=0.02,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.depth = depth
        self.hidden_size = hidden_size
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.num_heads = num_heads
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size
        self.temporal_patch_size = temporal_patch_size
        self.out_hidden_size = out_hidden_size
        self.num_position_embeddings = num_position_embeddings
        self.initializer_range = initializer_range


class Qwen3_5MoeConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Qwen3_5MoeModel`]. It is used to instantiate a
    Qwen3.5-MoE model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of
    Qwen3.5-35B-A3B-Instruct [Qwen/Qwen3.5-35B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3.5-35B-A3B-Instruct).

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        text_config (`Union[PretrainedConfig, dict]`, *optional*, defaults to `Qwen3_5TextConfig`):
            The config object or dictionary of the text backbone.
        vision_config (`Union[PretrainedConfig, dict]`,  *optional*, defaults to `Qwen3_5VisionConfig`):
            The config object or dictionary of the vision backbone.
        image_token_id (`int`, *optional*, defaults to 248056):
            The image token index to encode the image prompt.
        video_token_id (`int`, *optional*, defaults to 248057):
            The video token index to encode the image prompt.
        vision_start_token_id (`int`, *optional*, defaults to 248053):
            The start token index to encode the image prompt.
        vision_end_token_id (`int`, *optional*, defaults to 248054):
            The end token index to encode the image prompt.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to tie the word embeddings.

    ```python
    >>> from transformers import Qwen3_5MoeForConditionalGeneration, Qwen3_5MoeConfig

    >>> # Initializing a Qwen3.5-MoE style configuration
    >>> configuration = Qwen3_5MoeConfig()

    >>> # Initializing a model from the Qwen3.5-35B-A3B style configuration
    >>> model = Qwen3_5MoeForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "qwen3_5_moe"
    sub_configs = {"vision_config": Qwen3_5MoeVisionConfig, "text_config": Qwen3_5MoeTextConfig}
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        text_config=None,
        vision_config=None,
        image_token_id=248056,
        video_token_id=248057,
        vision_start_token_id=248053,
        vision_end_token_id=248054,
        tie_word_embeddings=False,
        **kwargs,
    ):
        if isinstance(vision_config, dict):
            self.vision_config = self.sub_configs["vision_config"](**vision_config)
        elif vision_config is None:
            self.vision_config = self.sub_configs["vision_config"]()

        if isinstance(text_config, dict):
            self.text_config = self.sub_configs["text_config"](**text_config)
        elif text_config is None:
            self.text_config = self.sub_configs["text_config"]()

        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.vision_start_token_id = vision_start_token_id
        self.vision_end_token_id = vision_end_token_id
        self.tie_word_embeddings = tie_word_embeddings
        super().__init__(**kwargs)
