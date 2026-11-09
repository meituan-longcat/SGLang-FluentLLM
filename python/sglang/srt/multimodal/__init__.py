"""
Multimodal processing module for fluentllm.

Provides:
- BaseMultimodalProcessor: base class for all model-specific processors
- MultimodalSpecialTokens: token string/id/regex management
- BaseMultiModalProcessorOutput: raw loaded media data container
- get_mm_processor(): factory to get the right processor for a model
- embed_mm_inputs(): fuse vision+text embeddings in model forward
- general_mm_embed_routine(): high-level embedding entry point
- hash_feature(): content hash for ViT Embedding Cache lookup
- MultimodalDataItem, MultimodalInputs, MultimodalProcessorOutput (in schedule_batch)
"""

from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor,
    BaseMultiModalProcessorOutput,
    MultimodalSpecialTokens,
)
from sglang.srt.multimodal.multimodal_processor import (
    get_mm_processor,
    PROCESSOR_MAPPING,
)
from sglang.srt.multimodal.mm_utils import (
    hash_feature,
    embed_mm_inputs,
    general_mm_embed_routine,
    get_new_expanded_mm_items,
    init_mm_embedding_cache,
)

__all__ = [
    "BaseMultimodalProcessor",
    "BaseMultiModalProcessorOutput",
    "MultimodalSpecialTokens",
    "get_mm_processor",
    "PROCESSOR_MAPPING",
    "hash_feature",
    "embed_mm_inputs",
    "general_mm_embed_routine",
    "get_new_expanded_mm_items",
    "init_mm_embedding_cache",
]
