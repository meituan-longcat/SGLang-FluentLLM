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
import json
import logging
import os
import tempfile
import warnings
from typing import Optional, Union

import torch
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

from sglang.srt.configs import (
    DeepseekMhaNsaConfig,
    FLASHConfig,
    GlmMoeDsaConfig,
    KimiLinearConfig,
    LongcatFlashVLConfig,
    LongcatFlashVLProVConfig,
    Qwen3_5MoeConfig,
    Qwen3Config,
    Qwen3MoeConfig,
    Qwen3NextConfig,
)
from sglang.srt.utils import lru_cache_frozenset

_CONFIG_REGISTRY: dict[str, type[PretrainedConfig]] = {
    Qwen3MoeConfig.model_type: Qwen3MoeConfig,
    Qwen3_5MoeConfig.model_type: Qwen3_5MoeConfig,
    FLASHConfig.model_type: FLASHConfig,
    Qwen3Config.model_type: Qwen3Config,
    DeepseekMhaNsaConfig.model_type: DeepseekMhaNsaConfig,
    Qwen3NextConfig.model_type: Qwen3NextConfig,
    KimiLinearConfig.model_type: KimiLinearConfig,
    GlmMoeDsaConfig.model_type: GlmMoeDsaConfig,
    LongcatFlashVLConfig.model_type: LongcatFlashVLConfig,
    LongcatFlashVLProVConfig.model_type: LongcatFlashVLProVConfig,
}

for name, cls in _CONFIG_REGISTRY.items():
    with contextlib.suppress(ValueError):
        AutoConfig.register(name, cls)


def download_from_hf(model_path: str):
    if os.path.exists(model_path):
        return model_path

    return snapshot_download(model_path, allow_patterns=["*.json", "*.bin", "*.model"])


def _patch_text_config(parent_config: PretrainedConfig, text_config: PretrainedConfig) -> PretrainedConfig:
    """Propagate standard token/embedding attributes between parent and text sub-config.

    In newer transformers versions the "untangle config" refactor removed automatic
    inheritance of top-level PretrainedConfig attributes from sub-configs. Downstream
    code may access these attributes on either the parent or the text config, so we
    synchronise them in both directions.
    """
    _ATTRS_TO_PROPAGATE = [
        "pad_token_id",
        "bos_token_id",
        "eos_token_id",
        "tie_word_embeddings",
    ]
    for attr in _ATTRS_TO_PROPAGATE:
        parent_val = getattr(parent_config, attr, None)
        text_val = getattr(text_config, attr, None)
        if parent_val is not None and text_val is None:
            setattr(text_config, attr, parent_val)
        elif text_val is not None and parent_val is None:
            setattr(parent_config, attr, text_val)
    return text_config


def get_hf_text_config(config: PretrainedConfig) -> PretrainedConfig:
    """Return the text/LLM sub-config for multimodal models; no-op for pure text models.

    Supports the following sub-config attribute names (in priority order):
    thinker_config > language_config > text_config

    For each case, attributes are synchronised between the parent config and the
    text sub-config so that callers can safely read token IDs / dtype from either.
    """
    if config.architectures is not None:
        class_name = config.architectures[0]
        if class_name.startswith("Llava") and class_name.endswith("ForCausalLM"):
            # Non-HF LLaVA models: keep the parent config as-is.
            # NOTE(HandH1998): Set torch_dtype to float16 as used by image features.
            config.torch_dtype = torch.float16
            return config

    parent_dtype = getattr(config, "torch_dtype", None)

    # Convert any dict-valued sub-configs to PretrainedConfig objects early so
    # that hasattr() checks and attribute propagation work correctly.
    for attr in ("text_config", "language_config", "thinker_config"):
        sub = getattr(config, attr, None)
        if isinstance(sub, dict):
            converted = PretrainedConfig(**sub)
            if getattr(converted, "torch_dtype", None) is None and parent_dtype is not None:
                converted.torch_dtype = parent_dtype
            setattr(config, attr, converted)

    # Priority: thinker_config (Qwen2.5-Omni) > language_config > text_config
    if hasattr(config, "thinker_config"):
        thinker_config = config.thinker_config
        if hasattr(thinker_config, "text_config"):
            thinker_config.text_config.torch_dtype = getattr(thinker_config, "torch_dtype", None)
            return _patch_text_config(config, thinker_config.text_config)
        return _patch_text_config(config, thinker_config)

    if hasattr(config, "language_config"):
        return _patch_text_config(config, config.language_config)

    if hasattr(config, "text_config"):
        assert hasattr(config.text_config, "num_attention_heads"), (
            "text_config is expected to have 'num_attention_heads'. Check that the transformers config aligns with this assumption."
        )
        return _patch_text_config(config, config.text_config)

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

    with open(config_file) as f:
        config_json = json.load(f)

    config_json["architectures"] = ["DeepseekV3ForCausalLM"]
    config_json["model_type"] = "deepseek_v3"

    tmp_path = os.path.join(tempfile.gettempdir(), "_tmp_config_folder")
    os.makedirs(tmp_path, exist_ok=True)

    unique_path = os.path.join(tmp_path, f"deepseek_v32_{os.getpid()}")
    with open(unique_path, "w") as f:
        json.dump(config_json, f)

    return AutoConfig.from_pretrained(unique_path, trust_remote_code=trust_remote_code, revision=revision, **kwargs)


def get_config(
    model: str,
    trust_remote_code: bool,
    revision: Optional[str] = None,
    model_override_args: Optional[dict] = None,
    is_draft_worker: Optional[bool] = False,
    **kwargs,
):
    """Load a HuggingFace model config.

    For multimodal models (e.g. Qwen3-VL) the *top-level* config object is
    returned so that callers can access ``architectures``, ``vision_config``,
    ``image_token_id``, etc.  Attributes from the text sub-config are
    flattened onto the top-level config so that existing code that reads
    ``config.num_attention_heads`` and similar text-model fields continues to
    work without modification.
    """
    try:
        with open(os.path.join(model, "config.json")) as file:
            raw_json = json.load(file)
    except FileNotFoundError as e:
        raise RuntimeError(f"Config file not found in {model}. Please check the path.") from e
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Failed to decode JSON from config file in {model}. Please ensure the file is valid JSON.") from e
    # Zero computation expert / Longcat remapping
    if raw_json.get("architectures") in [["LongcatCausalLM"], ["LongcatFlashForCausalLM"], ["LongcatFlashNgramForCausalLM"]]:
        if raw_json.get("visual_config", {}).get("enable", False):
            pass  # Keep model_type ("omni" or "pro_v"), route to LongcatFlashVLConfig family
        else:
            raw_json["model_type"] = "flash"

    model_type = raw_json.get("model_type", "llama")
    if model_type in _CONFIG_REGISTRY:
        config = _CONFIG_REGISTRY[model_type].from_pretrained(model, revision=revision)
        config._name_or_path = model
    else:
        try:
            config = AutoConfig.from_pretrained(model, trust_remote_code=trust_remote_code, revision=revision, **kwargs)
        except ValueError as e:
            if not any(k in str(e) for k in ["deepseek_v32", "glm_moe_dsa"]):
                raise e
            config = _load_deepseek_v32_model(model, trust_remote_code=trust_remote_code, revision=revision, **kwargs)

    # ------------------------------------------------------------------ #
    # quantization_config fixup (must happen on the top-level config)
    # ------------------------------------------------------------------ #
    if hasattr(config, "quantization_config"):
        if isinstance(config.quantization_config, dict):
            if "modules_to_not_convert" in config.quantization_config:
                config.quantization_config["ignored_layers"] = config.quantization_config["modules_to_not_convert"]
                del config.quantization_config["modules_to_not_convert"]

    # ------------------------------------------------------------------ #
    # Multimodal support: flatten text sub-config attributes onto the
    # top-level config so legacy callers work unchanged.
    #
    # The top-level config is still the object we return, preserving
    # fields like architectures, vision_config, image_token_id, etc.
    # ------------------------------------------------------------------ #
    text_config = get_hf_text_config(config)
    if text_config is not config:
        # Propagate text-model fields to the top-level config where absent.
        for key, val in vars(text_config).items():
            if not hasattr(config, key) and val is not None:
                setattr(config, key, val)

    # ------------------------------------------------------------------ #
    # Architecture / model-type fixups
    # ------------------------------------------------------------------ #
    if config.architectures in [["LongcatCausalLM"], ["LongcatFlashForCausalLM"], ["LongcatFlashNgramForCausalLM"]]:
        if getattr(config, "model_type", None) in LongcatFlashVLConfig.MODEL_TYPES:
            config.architectures = ["LongcatFlashVLForConditionalGeneration"]
        else:
            config.architectures = ["FLASHForCausalLM"]

    if getattr(config, "model_type", None) == "qwen3_5_moe_text":
        config.architectures = ["Qwen3_5MoeForCausalLM"]

    # Adapt to the case where MTP and the base model are placed together. The
    # Longcat Flash VL drafter does not run the vision tower, so select the
    # language-only NextN implementation directly instead of creating a VL
    # architecture alias.
    if is_draft_worker and config.architectures == ["LongcatFlashVLForConditionalGeneration"]:
        config.architectures = ["FLASHForCausalLMNextN"]
    elif (
        is_draft_worker
        and config.architectures
        and "NextN" not in config.architectures[0]
        and "Eagle" not in config.architectures[0]
        # DFlash draft 是独立架构 (DFlashDraftModel), 不追加 NextN 后缀。
        and "DFlash" not in config.architectures[0]
    ):
        config.architectures[0] += "NextN"

    # Intercept MTP of longcat flash, directly reuse deepseek, Only 1 nextn layer is supported
    if config.architectures in [
        ["LongcatCausalLMNextN"],
        ["LlamaForCausalLMNextN"],
        ["MeituanQwen3ForCausalLMNextN"],
        ["FLASHForCausalLMNextN"],
    ]:
        config.num_hidden_layers = 1

    if model_override_args:
        config.update(model_override_args)

    # ModelConfig reads the text sub-config for layer counts and attention
    # metadata. Keep both views of the flattened VL config consistent after
    # applying overrides.
    if (
        config.architectures == ["FLASHForCausalLMNextN"]
        and getattr(config, "model_type", None) in LongcatFlashVLConfig.MODEL_TYPES
        and text_config is not config
    ):
        text_config.architectures = list(config.architectures)
        text_config.num_hidden_layers = config.num_hidden_layers

    return config


@lru_cache_frozenset(maxsize=32)
def get_generation_config(
    model: str,
    trust_remote_code: bool,
    revision: Optional[str] = None,
    **kwargs,
):
    try:
        return GenerationConfig.from_pretrained(model, trust_remote_code=trust_remote_code, revision=revision, **kwargs)
    except OSError:
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
            "does not exist or is not currently imported." in str(e) or "requires you to execute the tokenizer file" in str(e)
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
            "Using a slow tokenizer. This might cause a significant slowdown. Consider using a fast tokenizer instead.",
            stacklevel=2,
        )

    attach_additional_stop_token_ids(tokenizer)
    return tokenizer


def get_tokenizer_from_processor(processor):
    """Extract the tokenizer from a processor object.

    Some models (e.g. Qwen3-VL) return the tokenizer itself when
    AutoProcessor.from_pretrained() is called, rather than a composite
    processor that wraps a tokenizer. Handle both cases gracefully.
    """
    if isinstance(processor, (PreTrainedTokenizer, PreTrainedTokenizerFast)):
        return processor
    return processor.tokenizer


def get_processor(
    tokenizer_name: str,
    *args,
    tokenizer_mode: str = "auto",
    trust_remote_code: bool = False,
    tokenizer_revision: Optional[str] = None,
    **kwargs,
):
    try:
        processor = AutoProcessor.from_pretrained(
            tokenizer_name,
            *args,
            trust_remote_code=trust_remote_code,
            tokenizer_revision=tokenizer_revision,
            **kwargs,
        )
    except (OSError, ValueError):
        # Model directory is missing custom files referenced by auto_map
        # (e.g. configuration_new.py). Fall back to tokenizer-only loading.
        # Note: callers may pass `revision=` instead of `tokenizer_revision=`,
        # so resolve the effective revision from both sources.
        from transformers import AutoTokenizer

        effective_revision = kwargs.get("revision", tokenizer_revision)
        processor = AutoTokenizer.from_pretrained(
            tokenizer_name,
            trust_remote_code=trust_remote_code,
            revision=effective_revision,
        )

    attach_additional_stop_token_ids(get_tokenizer_from_processor(processor))
    return processor


def attach_additional_stop_token_ids(tokenizer):
    # Special handling for stop token <|eom_id|> generated by llama 3 tool use.
    if "<|eom_id|>" in tokenizer.get_added_vocab():
        tokenizer.additional_stop_token_ids = set([tokenizer.get_added_vocab()["<|eom_id|>"]])
    else:
        tokenizer.additional_stop_token_ids = None
