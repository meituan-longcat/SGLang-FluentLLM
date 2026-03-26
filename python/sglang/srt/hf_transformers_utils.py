# Copyright 2023-2024 SGLang Team
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
# ==============================================================================
"""Utilities for Huggingface Transformers."""

import contextlib
import logging
import os
import tempfile
import json
import warnings
from typing import Dict, Optional, Type, Union

from huggingface_hub import snapshot_download
from transformers import (
    AutoConfig,
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig,
    PretrainedConfig,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)

from sglang.srt.utils import lru_cache_frozenset
from sglang.srt.configs import (
    Qwen3MoeConfig,
    FLASHConfig,
    Qwen3Config,
    DeepseekMhaNsaConfig,
    Glm4MoeConfig,
    Qwen3NextConfig,
    KimiLinearConfig
)

_CONFIG_REGISTRY: Dict[str, Type[PretrainedConfig]] = {
    Qwen3MoeConfig.model_type: Qwen3MoeConfig,
    FLASHConfig.model_type: FLASHConfig,
    Glm4MoeConfig.model_type: Glm4MoeConfig,
    Qwen3Config.model_type: Qwen3Config,
    DeepseekMhaNsaConfig.model_type: DeepseekMhaNsaConfig,
    Qwen3NextConfig.model_type: Qwen3NextConfig,
    KimiLinearConfig.model_type: KimiLinearConfig
}

for name, cls in _CONFIG_REGISTRY.items():
    with contextlib.suppress(ValueError):
        AutoConfig.register(name, cls)


def download_from_hf(model_path: str):
    if os.path.exists(model_path):
        return model_path

    return snapshot_download(model_path, allow_patterns=["*.json", "*.bin", "*.model"])

def get_hf_text_config(config: PretrainedConfig):
    """Get the "sub" config relevant to llm for multi modal models.
    No op for pure text models.
    """
    if config.architectures is not None:
        class_name = config.architectures[0]
        if class_name.startswith("Llava") and class_name.endswith("ForCausalLM"):
            # We support non-hf version of llava models, so we do not want to
            # read the wrong values from the unused default text_config.
            # NOTE(HandH1998): We set `torch_dtype` of config to `torch.float16` for the weights, as
            # `torch.float16` is default used for image features in `python/sglang/srt/models/llava.py`.
            setattr(config, "torch_dtype", torch.float16)
            return config

    if hasattr(config, "text_config"):
        # The code operates under the assumption that text_config should have
        # `num_attention_heads` (among others). Assert here to fail early
        # if transformers config doesn't align with this assumption.
        assert hasattr(config.text_config, "num_attention_heads")
        return config.text_config
    if hasattr(config, "language_config"):
        return config.language_config
    if hasattr(config, "thinker_config"):
        # qwen2.5 omni
        thinker_config = config.thinker_config
        if hasattr(thinker_config, "text_config"):
            setattr(
                thinker_config.text_config,
                "torch_dtype",
                getattr(thinker_config, "torch_dtype", None),
            )
            return thinker_config.text_config
        return thinker_config
    else:
        return config

# Temporary hack for DeepSeek-V3.2 model
def _load_deepseek_v32_model(
    model_path: str,
    trust_remote_code: bool = False,
    revision: Optional[str] = None,
    **kwargs,
):
    # first get the local path
    local_path = download_from_hf(model_path)
    # then load the config file in json
    config_file = os.path.join(local_path, "config.json")
    if not os.path.exists(config_file):
        raise RuntimeError(f"Can't find config file in {local_path}.")

    with open(config_file, "r") as f:
        config_json = json.load(f)

    # config_json["architectures"] = ["DeepseekV3ForCausalLM"]
    config_json["model_type"] = "deepseek_v3"

    tmp_path = os.path.join(tempfile.gettempdir(), "_tmp_config_folder")
    os.makedirs(tmp_path, exist_ok=True)

    unique_path = os.path.join(tmp_path, f"deepseek_v32_{os.getpid()}")
    with open(unique_path, "w") as f:
        json.dump(config_json, f)

    return AutoConfig.from_pretrained(
        unique_path, trust_remote_code=trust_remote_code, revision=revision, **kwargs
    )


def get_config(
    model: str,
    trust_remote_code: bool,
    revision: Optional[str] = None,
    model_override_args: Optional[dict] = None,
    is_draft_worker: Optional[bool] = False,
    **kwargs,
):
    try:
        with open(os.path.join(model, "config.json"), 'r') as file:
            config = json.load(file)
    except FileNotFoundError:
        raise RuntimeError(f"Config file not found in {model}. Please check the path.")
    except json.JSONDecodeError:
        raise RuntimeError(f"Failed to decode JSON from config file in {model}. Please ensure the file is valid JSON.")
    # Zero computation expert
    if config["architectures"] in [["LongcatCausalLM"], ["LongcatFlashForCausalLM"], ["LongcatFlashNgramForCausalLM"],["LongcatNextForCausalLM"]]:
        config["model_type"] = "flash"

    if config.get("model_type", "llama") in _CONFIG_REGISTRY:
        config_class = _CONFIG_REGISTRY[config["model_type"]]
        config = config_class.from_pretrained(model, revision=revision)
        setattr(config, "_name_or_path", model)
    else:
        try:
            config = AutoConfig.from_pretrained(
                model, trust_remote_code=trust_remote_code, revision=revision, **kwargs
            )
        except ValueError as e:
            if not "deepseek_v32" in str(e):
                raise e
            config = _load_deepseek_v32_model(
                model, trust_remote_code=trust_remote_code, revision=revision, **kwargs
            )
    config = get_hf_text_config(config)

    if hasattr(config, "quantization_config"):
        if "modules_to_not_convert" in config.quantization_config:
            config.quantization_config["ignored_layers"] = config.quantization_config["modules_to_not_convert"]
            del config.quantization_config["modules_to_not_convert"]

    if config.architectures in [["LongcatCausalLM"], ["LongcatFlashForCausalLM"], ["LongcatFlashNgramForCausalLM"],["LongcatNextForCausalLM"]]:
        config.architectures = ["FLASHForCausalLM"]

    # Adapt to the case where mtp and base model are placed together
    if is_draft_worker and "NextN" not in config.architectures[0] and "Eagle" not in config.architectures[0]:
        config.architectures[0] += "NextN"

    # Intercept MTP of longcat flash, directly reuse deepseek, Only 1 nextn layer is supported
    if config.architectures == ["LongcatCausalLMNextN"] \
        or config.architectures == ["LlamaForCausalLMNextN"] \
        or config.architectures == ["MeituanQwen3ForCausalLMNextN"] \
        or config.architectures == ["FLASHForCausalLMNextN"]:
        config.num_hidden_layers = 1

    if model_override_args:
        config.update(model_override_args)

    return config


@lru_cache_frozenset(maxsize=32)
def get_generation_config(
    model: str,
    trust_remote_code: bool,
    revision: Optional[str] = None,
    **kwargs,
):
    try:
        return GenerationConfig.from_pretrained(
            model, trust_remote_code=trust_remote_code, revision=revision, **kwargs
        )
    except OSError as e:
        logging.info("model doesn't have generation_config.json")
        return None


# Models don't use the same configuration key for determining the maximum
# context length.  Store them here so we can sanely check them.
# NOTE: The ordering here is important. Some models have two of these and we
# have a preference for which value gets used.
CONTEXT_LENGTH_KEYS = [
    "max_sequence_length",
    "seq_length",
    "max_seq_len",
    "model_max_length",
    "max_position_embeddings",
]


def get_context_length(config):
    """Get the context length of a model from a huggingface model configs."""
    text_config = config
    rope_scaling = getattr(text_config, "rope_scaling", None)
    if rope_scaling:
        rope_scaling_factor = rope_scaling.get("factor", 1)
        if "original_max_position_embeddings" in rope_scaling:
            rope_scaling_factor = 1
        if rope_scaling.get("rope_type", None) == "llama3":
            rope_scaling_factor = 1
    else:
        rope_scaling_factor = 1

    for key in CONTEXT_LENGTH_KEYS:
        val = getattr(text_config, key, None)
        if val is not None:
            return int(rope_scaling_factor * val)
    return 2048


# A fast LLaMA tokenizer with the pre-processed `tokenizer.json` file.
_FAST_LLAMA_TOKENIZER = "hf-internal-testing/llama-tokenizer"


def get_tokenizer(
    tokenizer_name: str,
    *args,
    tokenizer_mode: str = "auto",
    trust_remote_code: bool = False,
    tokenizer_revision: Optional[str] = None,
    **kwargs,
) -> Union[PreTrainedTokenizer, PreTrainedTokenizerFast]:
    """Gets a tokenizer for the given model name via Huggingface."""
    if tokenizer_mode == "slow":
        if kwargs.get("use_fast", False):
            raise ValueError("Cannot use the fast tokenizer in slow tokenizer mode.")
        kwargs["use_fast"] = False

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            *args,
            trust_remote_code=trust_remote_code,
            tokenizer_revision=tokenizer_revision,
            clean_up_tokenization_spaces=False,
            **kwargs,
        )
    except TypeError as e:
        # The LLaMA tokenizer causes a protobuf error in some environments.
        err_msg = (
            "Failed to load the tokenizer. If you are using a LLaMA V1 model "
            f"consider using '{_FAST_LLAMA_TOKENIZER}' instead of the "
            "original tokenizer."
        )
        raise RuntimeError(err_msg) from e
    except ValueError as e:
        # If the error pertains to the tokenizer class not existing or not
        # currently being imported, suggest using the --trust-remote-code flag.
        if not trust_remote_code and (
            "does not exist or is not currently imported." in str(e)
            or "requires you to execute the tokenizer file" in str(e)
        ):
            err_msg = (
                "Failed to load the tokenizer. If the tokenizer is a custom "
                "tokenizer not yet available in the HuggingFace transformers "
                "library, consider setting `trust_remote_code=True` in LLM "
                "or using the `--trust-remote-code` flag in the CLI."
            )
            raise RuntimeError(err_msg) from e
        else:
            raise e

    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        warnings.warn(
            "Using a slow tokenizer. This might cause a significant "
            "slowdown. Consider using a fast tokenizer instead."
        )

    attach_additional_stop_token_ids(tokenizer)
    return tokenizer


def get_processor(
    tokenizer_name: str,
    *args,
    tokenizer_mode: str = "auto",
    trust_remote_code: bool = False,
    tokenizer_revision: Optional[str] = None,
    **kwargs,
):
    processor = AutoProcessor.from_pretrained(
        tokenizer_name,
        *args,
        trust_remote_code=trust_remote_code,
        tokenizer_revision=tokenizer_revision,
        **kwargs,
    )

    attach_additional_stop_token_ids(processor.tokenizer)
    return processor


def attach_additional_stop_token_ids(tokenizer):
    # Special handling for stop token <|eom_id|> generated by llama 3 tool use.
    if "<|eom_id|>" in tokenizer.get_added_vocab():
        tokenizer.additional_stop_token_ids = set(
            [tokenizer.get_added_vocab()["<|eom_id|>"]]
        )
    else:
        tokenizer.additional_stop_token_ids = None
