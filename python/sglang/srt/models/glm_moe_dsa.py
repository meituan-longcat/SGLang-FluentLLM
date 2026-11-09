"""Inference-only GLM-5.2 (GlmMoeDsa) model.

GLM-5.2 is a DeepSeek-V3.2-style architecture: MLA attention with DSA
sparse attention (indexer on a subset of layers, see
config.indexer_types), sigmoid noaux_tc MoE routing with
e_score_correction_bias, and a single MTP layer stored at
model.layers.<num_hidden_layers> in the base checkpoint. The DeepSeek
implementations are reused as is; the per-layer indexer schedule is
handled by dsa_layer_has_indexer() in configs/model_config.py.
"""

from sglang.srt.models.deepseek_nextn import DeepseekV3ForCausalLMNextN
from sglang.srt.models.deepseek_v2 import DeepseekV2ForCausalLM


class GlmMoeDsaForCausalLM(DeepseekV2ForCausalLM):
    pass


class GlmMoeDsaForCausalLMNextN(DeepseekV3ForCausalLMNextN):
    """GLM-5.2 MTP draft model.

    The MTP layer is stored as model.layers.<num_hidden_layers> in the base
    checkpoint (same directory as the target model) and owns a full indexer,
    so the DeepSeek NextN implementation can be reused as is.
    """

    pass


EntryClass = [GlmMoeDsaForCausalLM, GlmMoeDsaForCausalLMNextN]
