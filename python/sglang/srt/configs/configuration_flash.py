import os

import torch
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

from sglang.srt.utils.common import divide

logger = logging.get_logger(__name__)

FLASH_PRETRAINED_CONFIG_ARCHIVE_MAP = {}


class FLASHConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`FLASHModel`]. It is used to instantiate an FLASH
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the FLASH.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 102400):
            Vocabulary size of the Deep model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`FLASHModel`]
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 11008):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer decoder.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer decoder.
        num_experts_per_tok (`int`, *optional*, defaults to None):
            Number of selected experts, None means dense model.
        norm_topk_prob (`bool`, *optional*, defaults to False):
            Whether to normalize the weights of the routed experts.
        num_key_value_heads (`int`, *optional*):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1 the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details checkout [this
            paper](https://arxiv.org/pdf/2305.13245.pdf). If it is not specified, will default to
            `num_attention_heads`.
        max_position_embeddings (`int`, *optional*, defaults to 2048):
            The maximum sequence length that this model might ever be used with.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        pad_token_id (`int`, *optional*):
            Padding token id.
        bos_token_id (`int`, *optional*, defaults to 1):
            Beginning of stream token id.
        eos_token_id (`int`, *optional*, defaults to 2):
            End of stream token id.
        pretraining_tp (`int`, *optional*, defaults to 1):
            Experimental feature. Tensor parallelism rank used during pretraining. Please refer to [this
            document](https://huggingface.co/docs/transformers/parallelism) to understand more about it. This value is
            necessary to ensure exact reproducibility of the pretraining results. Please refer to [this
            issue](https://github.com/pytorch/pytorch/issues/76232).
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to tie weight embeddings
        rope_theta (`float`, *optional*, defaults to 10000.0):
            The base period of the RoPE embeddings.
        rope_scaling (`Dict`, *optional*):
            Dictionary containing the scaling configuration for the RoPE embeddings. Currently supports two scaling
            strategies: linear and dynamic. Their scaling factor must be a float greater than 1. The expected format is
            `{"type": strategy name, "factor": scaling factor}`. When using this flag, don't update
            `max_position_embeddings` to the expected new maximum.
        attention_bias (`bool`, defaults to `False`, *optional*, defaults to `False`):
            Whether to use a bias in the query, key, value and output projection layers during self-attention.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.

    """

    model_type = "flash"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=131072,
        hidden_size=4096,
        intermediate_size=8192,
        num_layers=28,
        num_hidden_layers=None,
        num_attention_heads=96,
        num_key_value_heads=1,
        ep_size=1,
        kv_lora_rank=512,
        q_lora_rank=1536,
        qk_rope_head_dim=64,
        v_head_dim=128,
        qk_nope_head_dim=128,
        num_experts_per_tok=None,
        norm_topk_prob=False,
        max_position_embeddings=8192,
        initializer_range=0.02,
        rms_norm_eps=1e-05,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=100000,
        eos_token_id=100001,
        pretraining_tp=1,
        tie_word_embeddings=False,
        rope_theta=1000000.0,
        rope_scaling=None,
        attention_bias=False,
        attention_dropout=0.0,
        mla_scale_q_lora=False,
        mla_scale_kv_lora=False,
        torch_dtype="bfloat16",
        params_dtype="bfloat16",
        router_dtype="float32",
        router_bias=False,
        topk_method=None,
        routed_scaling_factor=None,
        zero_expert_num=0,
        zero_expert_type=None,
        nextn_use_scmoe=False,
        oe_vocab_size_ratio=None,
        oe_neighbor_num=None,
        oe_split_num=None,
        oe_ignore_tokens=None,
        ngram_vocab_size_ratio=None,
        emb_neighbor_num=None,
        emb_split_num=None,
        embP=None,
        indexer_rope_interleave: bool = True,
        linear_method=None,
        linear_hidden_size=None,
        linear_head_dim=None,
        linear_num_heads=None,
        linear_conv_size=None,
        fa_interval=4,
        use_mla=1,
        full_attention_layer_ids=None,
        **kwargs,
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            torch_dtype=torch_dtype,
            params_dtype=params_dtype,
            router_dtype=router_dtype,
            topk_method=topk_method,
            router_bias=router_bias,
            nextn_use_scmoe=nextn_use_scmoe,
            **kwargs,
        )
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers if num_hidden_layers is not None else num_layers
        self.num_attention_heads = num_attention_heads
        self.ep_size = ep_size
        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = q_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.qk_nope_head_dim = qk_nope_head_dim
        self.num_experts_per_tok = num_experts_per_tok
        self.norm_topk_prob = norm_topk_prob
        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.pretraining_tp = pretraining_tp
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.mla_scale_q_lora = mla_scale_q_lora
        self.mla_scale_kv_lora = mla_scale_kv_lora
        # Whether the indexer uses interleaved (GPT-J style) rotary position
        self.indexer_rope_interleave = indexer_rope_interleave
        self.zero_expert_num = zero_expert_num
        self.zero_expert_type = zero_expert_type
        self.routed_scaling_factor = routed_scaling_factor
        self.hidden_act = "silu"
        self.use_over_embedding = oe_vocab_size_ratio is not None
        self.over_embedding_vocab_size_ratio = oe_vocab_size_ratio
        self.over_embedding_base = vocab_size
        if self.use_over_embedding:
            self.over_embedding_m = int(self.over_embedding_base * oe_vocab_size_ratio)
            self.oe_neighbor_num = oe_neighbor_num
            self.oe_split_num = oe_split_num
            self.embp = embP
            self.oe_ignore_tokens = self.build_oe_ignore_tokens(oe_ignore_tokens)
        self.use_ngram_embedding = ngram_vocab_size_ratio is not None
        if self.use_ngram_embedding:
            self.use_over_embedding = True
            self.over_embedding_m = int(ngram_vocab_size_ratio * self.over_embedding_base)
            self.oe_neighbor_num = emb_neighbor_num
            self.oe_split_num = emb_split_num
            self.embp = None
            self.oe_ignore_tokens = self.build_oe_ignore_tokens(oe_ignore_tokens)
            self.eos_token_id = eos_token_id

        # Optional hybrid linear attention (KDA/FGB-KDA) settings.
        self.linear_method = linear_method
        self.linear_hidden_size = linear_hidden_size if linear_hidden_size is not None else hidden_size
        self.linear_head_dim = linear_head_dim if linear_head_dim is not None else hidden_size // num_attention_heads
        self.linear_num_heads = linear_num_heads if linear_num_heads is not None else num_attention_heads
        self.linear_conv_size = linear_conv_size if linear_conv_size is not None else 4
        self.fa_interval = fa_interval
        self.use_mla = use_mla
        self.kda_nope = kwargs.get("kda_nope", False)

        self.use_hybrid_linear_kda = str(self.linear_method).upper() in {"KDA", "FGBKDA"}
        total_attention_layers = self.num_hidden_layers * 2
        if full_attention_layer_ids is not None:
            self.full_attention_layer_ids = list(full_attention_layer_ids)
        elif self.use_hybrid_linear_kda and self.use_mla:
            interval = max(int(self.fa_interval), 1)
            self.full_attention_layer_ids = [i for i in range(total_attention_layers) if (i + 1) % interval == 0]
        else:
            self.full_attention_layer_ids = list(range(total_attention_layers))
        self.linear_layer_ids = [i for i in range(total_attention_layers) if i not in set(self.full_attention_layer_ids)]

    @staticmethod
    def build_oe_ignore_tokens(oe_ignore_tokens):
        """Parse and deduplicate the explicitly configured OE ignore tokens."""
        tokens = set()

        if oe_ignore_tokens is not None and oe_ignore_tokens != "":
            if isinstance(oe_ignore_tokens, str):
                for id_range in oe_ignore_tokens.split(","):
                    id_range = id_range.strip()
                    if not id_range:
                        continue
                    a, sep, b = id_range.partition(":")
                    if sep:
                        tokens.update(range(int(a), int(b)))
                    else:
                        tokens.add(int(a))
            else:
                tokens.update(int(t) for t in oe_ignore_tokens)

        return sorted(tokens)

    @property
    def linear_attn_config(self):
        return {
            "head_dim": self.linear_head_dim,
            "num_heads": self.linear_num_heads,
            "short_conv_kernel_size": self.linear_conv_size,
            "kda_layers": [i + 1 for i in self.linear_layer_ids],
            "full_attn_layers": [i + 1 for i in self.full_attention_layer_ids],
        }

    @property
    def mamba2_cache_params(self):
        from sglang.srt.layers.attention.dp_attention import get_attention_tp_size

        world_size = get_attention_tp_size()
        num_heads = self.linear_num_heads
        head_dim = self.linear_head_dim
        conv_kernel_size = self.linear_conv_size
        proj_size = num_heads * head_dim
        conv_state_shape = (divide(proj_size, world_size), conv_kernel_size - 1)
        temporal_state_shape = (divide(num_heads, world_size), head_dim, head_dim)
        conv_state_shape = conv_state_shape[1], conv_state_shape[0]
        conv_dtype = torch.bfloat16
        dtype_map = {"float32": torch.float32, "bfloat16": torch.bfloat16}
        ssm_dtype = dtype_map[os.environ.get("SGLANG_MAMBA_SSM_DTYPE", "bfloat16")]
        return (
            [conv_state_shape, conv_state_shape, conv_state_shape],
            temporal_state_shape,
            conv_dtype,
            ssm_dtype,
            self.linear_layer_ids,
        )

    @property
    def mamba_cache_per_req(self):
        conv_state_shapes, temporal_state_shape, conv_dtype, ssm_dtype, mamba_layers = self.mamba2_cache_params
        conv_items = sum(int(torch.tensor(shape).prod().item()) for shape in conv_state_shapes)
        temporal_items = int(torch.tensor(temporal_state_shape).prod().item())
        return (conv_items * conv_dtype.itemsize + temporal_items * ssm_dtype.itemsize) * len(mamba_layers)
