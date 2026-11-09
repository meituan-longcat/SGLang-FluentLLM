"""
Multimodal processor registry and factory for fluentllm.

Design (aligned with upstream sglang):
  - PROCESSOR_MAPPING maps model class → processor class.
  - import_processors() auto-discovers all processors under a package by scanning
    every module with pkgutil and registering classes that subclass
    BaseMultimodalProcessor and declare a non-empty `models` list.
    **No manual registration is ever needed.** Adding a new processor file to
    python/sglang/srt/multimodal/processors/ is sufficient.
  - get_mm_processor() is the public factory, matching the same signature as
    the upstream version.

Usage:
    # Called once at server startup (lazy, triggered by first get_mm_processor call):
    from sglang.srt.multimodal.multimodal_processor import get_mm_processor
    processor = get_mm_processor(hf_config, server_args, hf_processor, transport_mode)

    # Adding a new model processor (no registry code required):
    #   1. Create python/sglang/srt/multimodal/processors/my_model.py
    #   2. Define class MyModelProcessor(BaseMultimodalProcessor):
    #          models = [MyModelForConditionalGeneration]
    #          ...
    #   Done – it is registered automatically on the next server startup.

References:
    - sglang/python/sglang/srt/managers/multimodal_processor.py (upstream)
"""

import importlib
import inspect
import logging
import pkgutil
from typing import Dict, Optional, Type

from sglang.srt.multimodal.processors.base_processor import BaseMultimodalProcessor
from sglang.srt.utils import get_colorful_logger

logger = get_colorful_logger(__name__)

# Registry: model class → processor class
PROCESSOR_MAPPING: Dict[type, Type[BaseMultimodalProcessor]] = {}

# Track whether auto-discovery has run
_processors_loaded = False


def import_processors(package_name: str) -> None:
    """
    Auto-discover and register all BaseMultimodalProcessor subclasses inside
    *package_name* (and any direct sub-modules – not recursive into sub-packages).

    For each discovered module, every class that:
      - is defined in that module (not re-exported from elsewhere),
      - is a non-abstract subclass of BaseMultimodalProcessor, and
      - has a non-empty `models` attribute,
    is registered in PROCESSOR_MAPPING for each entry of its `models` list.

    Import errors are caught and logged as warnings so that a processor that
    requires an optional dependency (e.g. a GPU-only library) does not break
    the entire registry.
    """
    package = importlib.import_module(package_name)
    for _finder, module_name, is_pkg in pkgutil.iter_modules(
        package.__path__, package_name + "."
    ):
        if is_pkg:
            continue  # only scan flat files, not sub-packages
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:
            logger.warning(
                f"Skipping processor module {module_name} due to import error: {exc}"
            )
            continue

        for _name, cls in inspect.getmembers(module, inspect.isclass):
            # Only classes defined in this module (avoid re-exports)
            if cls.__module__ != module.__name__:
                continue
            if not (issubclass(cls, BaseMultimodalProcessor) and cls is not BaseMultimodalProcessor):
                continue
            models = getattr(cls, "models", [])
            if not models:
                continue
            for model_cls in models:
                if model_cls not in PROCESSOR_MAPPING:
                    PROCESSOR_MAPPING[model_cls] = cls
                    model_name = model_cls if isinstance(model_cls, str) else getattr(model_cls, "__name__", repr(model_cls))
                    logger.debug(
                        f"Registered processor {cls.__name__} for {model_name}"
                    )

def _ensure_processors_loaded() -> None:
    """Lazily trigger auto-discovery the first time the registry is needed."""
    global _processors_loaded
    if _processors_loaded:
        return
    _processors_loaded = True
    import_processors("sglang.srt.multimodal.processors")


def get_mm_processor(
    hf_config,
    server_args,
    hf_processor,
    transport_mode=None,
    **kwargs,
) -> Optional[BaseMultimodalProcessor]:
    _ensure_processors_loaded()

    for model_cls, processor_cls in PROCESSOR_MAPPING.items():
        # model_cls may be a class or a string (string avoids import-time side-effects)
        arch_name = model_cls if isinstance(model_cls, str) else model_cls.__name__
        if arch_name not in hf_config.architectures:
            continue

        return processor_cls(
            hf_config, server_args, hf_processor, transport_mode, **kwargs
        )

    registered = [
        m if isinstance(m, str) else m.__name__ for m in PROCESSOR_MAPPING.keys()
    ]
    raise ValueError(
        f"No processor registered for architecture: {hf_config.architectures}.\n"
        f"Registered architectures: {registered}"
    )