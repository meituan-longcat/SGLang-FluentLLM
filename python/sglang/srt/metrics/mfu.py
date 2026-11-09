from sglang.srt.utils import get_colorful_logger

logger = get_colorful_logger(__name__)


def _get_hf_attr(hf_config, names, default):
    """Read the first present attribute from hf_config; warn and return default otherwise."""
    if isinstance(names, str):
        names = (names,)
    for name in names:
        value = getattr(hf_config, name, None)
        if value is not None:
            return value
    logger.warning(f"MFU: hf_config missing {names!r}, fallback to default={default!r}")
    return default


class OverEmbedding:
    def __init__(
        self,
        embedding_dim,
        oe_neighbor_num,
        oe_split_num,
    ):
        self.embedding_dim = embedding_dim
        self.oe_neighbor_num = oe_neighbor_num
        self.oe_split_num = oe_split_num

    def get_flops(self, num_tokens):
        total_tokens = sum(num_tokens)
        bmm_flops = total_tokens * self.embedding_dim * self.embedding_dim * 2
        return bmm_flops


class MOE:
    def __init__(self, in_features, hidden_features, topk, num_experts, zero_expert_num=None, is_swiglu=True):
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.num_experts = num_experts
        self.topk = topk
        self.is_swiglu = is_swiglu
        self.zero_expert_num = zero_expert_num if zero_expert_num is not None else 0

    def get_flops(self, num_tokens):
        total_tokens = sum(num_tokens)
        expert_selection_flops = total_tokens * self.in_features * (self.num_experts + self.zero_expert_num) * 2
        per_token_per_expert_flops = self.in_features * self.hidden_features * 2 * 2
        if self.is_swiglu:
            per_token_per_expert_flops += self.hidden_features * self.in_features * 2
        else:
            per_token_per_expert_flops += self.hidden_features * 2 * self.in_features * 2
        expert_flops = self.topk * total_tokens * per_token_per_expert_flops
        return expert_selection_flops + expert_flops


class DenseFFN:
    def __init__(self, in_features, hidden_features, is_swiglu=True):
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.is_swiglu = is_swiglu

    def get_flops(self, num_tokens):
        total_tokens = sum(num_tokens)
        per_token_flops = self.in_features * self.hidden_features * 2 * 2
        if self.is_swiglu:
            per_token_flops += self.hidden_features * self.in_features * 2
        else:
            per_token_flops += self.hidden_features * 2 * self.in_features * 2
        return total_tokens * per_token_flops


class MultiQueryAttention:
    def __init__(self, q_head_num, q_head_dim, v_head_dim=None):
        self.q_head_num = q_head_num
        self.q_head_dim = q_head_dim
        self.v_head_dim = v_head_dim if v_head_dim is not None else q_head_dim

    def get_flops(self, num_tokens, seq_lens):
        # QK': each head computes [q_len, q_head_dim] x [q_head_dim, kv_len]
        attn_weights_flops = sum(q_len * kv_len * self.q_head_num * self.q_head_dim * 2 for q_len, kv_len in zip(num_tokens, seq_lens))

        # softmax(QK'/sqrt(d))V: each head computes [q_len, kv_len] x [kv_len, v_head_dim]
        attn_output_flops = sum(q_len * kv_len * self.q_head_num * self.v_head_dim * 2 for q_len, kv_len in zip(num_tokens, seq_lens))

        return attn_weights_flops + attn_output_flops


class DeepseekSparseAttention:
    def __init__(
        self,
        hidden_size,
        index_head_dim,
        index_n_heads,
        index_topk,
        kv_lora_rank,
        num_attention_heads,
        q_lora_rank,
        qk_nope_head_dim,
        qk_rope_head_dim,
    ):
        self.hidden_size = hidden_size
        self.index_head_dim = index_head_dim
        self.index_n_heads = index_n_heads
        self.index_topk = index_topk
        self.kv_lora_rank = kv_lora_rank
        self.num_attention_heads = num_attention_heads
        self.q_lora_rank = q_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim

        self.indexerMQA = MultiQueryAttention(q_head_num=self.index_n_heads, q_head_dim=self.index_head_dim)
        self.sparseMQA = MultiQueryAttention(
            q_head_num=self.num_attention_heads,
            q_head_dim=self.kv_lora_rank + self.qk_rope_head_dim,
            v_head_dim=self.kv_lora_rank,
        )

    def get_flops(self, num_tokens, seq_lens, calc_index_flops=True):
        # q_down_sample
        total_tokens = sum(num_tokens)
        q_down_sample_flops = total_tokens * self.hidden_size * self.q_lora_rank * 2

        # q_up_sample + q_rope_proj
        q_up_sample_flops = total_tokens * self.q_lora_rank * (self.num_attention_heads * self.qk_head_dim + self.qk_rope_head_dim) * 2

        # q_nope_absorb
        q_nope_absorb_flops = total_tokens * self.num_attention_heads * self.kv_lora_rank * (self.num_attention_heads * self.qk_head_dim) * 2

        # kv_down_sample
        kv_down_sample_flops = total_tokens * self.hidden_size * self.kv_lora_rank * 2

        # kv rope proj
        kv_rope_proj_flops = total_tokens * self.hidden_size * self.qk_rope_head_dim * 2

        # sparse multi query attention (matmul only)
        sparse_attn_flops = self.sparseMQA.get_flops(num_tokens, [self.index_topk] * len(num_tokens))

        # output projection
        output_absorb_flops = total_tokens * self.num_attention_heads * self.kv_lora_rank * self.qk_nope_head_dim * 2
        output_proj_flops = total_tokens * self.num_attention_heads * self.qk_nope_head_dim * self.hidden_size * 2

        total_flops = (
            q_down_sample_flops
            + q_up_sample_flops
            + q_nope_absorb_flops
            + kv_down_sample_flops
            + kv_rope_proj_flops
            + sparse_attn_flops
            + output_absorb_flops
            + output_proj_flops
        )

        if calc_index_flops:
            # indexer weights
            indexer_q_proj_flops = total_tokens * self.q_lora_rank * self.index_n_heads * self.index_head_dim * 2
            index_k_proj_flops = total_tokens * self.hidden_size * self.index_head_dim * 2
            index_w_proj_flops = total_tokens * self.hidden_size * self.index_n_heads * 2

            # indexer attention score (matmul only)
            indexer_attn_flops = self.indexerMQA.get_flops(num_tokens, seq_lens)

            total_flops = total_flops + indexer_q_proj_flops + index_k_proj_flops + index_w_proj_flops + indexer_attn_flops

        return total_flops


class LMHead:
    def __init__(self, hidden_size, vocab_size):
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

    def get_flops(self, num_tokens):
        total_tokens = sum(num_tokens)
        return total_tokens * self.hidden_size * self.vocab_size * 2


class MTP:
    def __init__(
        self,
        vocab_size,
        in_features,
        hidden_features,
        index_head_dim,
        index_n_heads,
        index_topk,
        kv_lora_rank,
        num_attention_heads,
        q_lora_rank,
        qk_nope_head_dim,
        qk_rope_head_dim,
    ):
        self.vocab_size = vocab_size
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.index_head_dim = index_head_dim
        self.index_n_heads = index_n_heads
        self.index_topk = index_topk
        self.kv_lora_rank = kv_lora_rank
        self.num_attention_heads = num_attention_heads
        self.q_lora_rank = q_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.attention_layer = DeepseekSparseAttention(
            hidden_size=in_features,
            index_head_dim=index_head_dim,
            index_n_heads=index_n_heads,
            index_topk=index_topk,
            kv_lora_rank=kv_lora_rank,
            num_attention_heads=num_attention_heads,
            q_lora_rank=q_lora_rank,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
        )
        self.dense_ffn = DenseFFN(
            in_features=in_features,
            hidden_features=hidden_features,
        )
        self.lm_head = LMHead(
            hidden_size=in_features,
            vocab_size=vocab_size,
        )

    def get_flops(self, num_tokens, seq_lens):
        # first concat to 2d, then mapping to d
        total_tokens = sum(num_tokens)
        eh_projection_flops = total_tokens * self.in_features * self.in_features * 2
        attention_flops = self.attention_layer.get_flops(num_tokens, seq_lens)
        ffn_flops = self.dense_ffn.get_flops(num_tokens)
        lm_head_flops = self.lm_head.get_flops(num_tokens)
        return eh_projection_flops + attention_flops + ffn_flops + lm_head_flops


class LongcatProModelConfig:
    # Defaults used when hf_config is missing a field
    DEFAULTS = {
        "hidden_size": 8192,
        "intermediate_size": 12288,
        "moe_intermediate_size": 2048,
        "vocab_size": 163840,
        "num_hidden_layers": 80,
        "num_attention_heads": 64,
        "q_lora_rank": 1536,
        "kv_lora_rank": 512,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "index_head_dim": 128,
        "index_n_heads": 32,
        "index_topk": 2048,
        "num_experts_per_tok": 12,
        "n_routed_experts": 768,
        "zero_expert_num": 128,
        "oe_neighbor_num": 5,
        "oe_split_num": 4,
        "num_nextn_predict_layers": 3,
    }

    def __init__(self, hf_config=None, num_layers=None):
        d = self.DEFAULTS

        if hf_config is None:
            logger.warning("MFU: LongcatProModelConfig built without hf_config; using all defaults")

            def getter(names, default):
                return default
        else:

            def getter(names, default):
                return _get_hf_attr(hf_config, names, default)

        hidden_size = getter("hidden_size", d["hidden_size"])
        intermediate_size = getter(("expert_ffn_hidden_size", "intermediate_size"), d["intermediate_size"])
        moe_intermediate_size = getter(
            ("moe_intermediate_size", "expert_ffn_hidden_size"),
            d["moe_intermediate_size"],
        )
        vocab_size = getter("vocab_size", d["vocab_size"])
        num_attention_heads = getter("num_attention_heads", d["num_attention_heads"])
        q_lora_rank = getter("q_lora_rank", d["q_lora_rank"])
        kv_lora_rank = getter("kv_lora_rank", d["kv_lora_rank"])
        qk_nope_head_dim = getter("qk_nope_head_dim", d["qk_nope_head_dim"])
        qk_rope_head_dim = getter("qk_rope_head_dim", d["qk_rope_head_dim"])
        index_head_dim = getter("index_head_dim", d["index_head_dim"])
        index_n_heads = getter("index_n_heads", d["index_n_heads"])
        index_topk = getter("index_topk", d["index_topk"])
        topk = getter(("num_experts_per_tok", "moe_topk"), d["num_experts_per_tok"])
        zero_expert_num = getter("zero_expert_num", d["zero_expert_num"])
        oe_neighbor_num = getter("oe_neighbor_num", d["oe_neighbor_num"])
        oe_split_num = getter("oe_split_num", d["oe_split_num"])
        num_mtp_layers = getter("num_nextn_predict_layers", d["num_nextn_predict_layers"])

        # num_experts may be a list (per-layer) or scalar on hf_config
        num_experts_attr = getattr(hf_config, "num_experts", None) if hf_config is not None else None
        if num_experts_attr is None:
            num_experts_attr = getattr(hf_config, "n_routed_experts", None) if hf_config is not None else None
        if isinstance(num_experts_attr, (list, tuple)):
            num_experts = next((x for x in num_experts_attr if x), d["n_routed_experts"])
        elif isinstance(num_experts_attr, int):
            num_experts = num_experts_attr
        else:
            logger.warning(f"MFU: hf_config missing num_experts/n_routed_experts, fallback to default={d['n_routed_experts']!r}")
            num_experts = d["n_routed_experts"]

        if num_layers is None:
            num_layers = getter("num_hidden_layers", d["num_hidden_layers"])

        self.oe = OverEmbedding(
            embedding_dim=hidden_size,
            oe_neighbor_num=oe_neighbor_num,
            oe_split_num=oe_split_num,
        )
        self.sparse_attention = DeepseekSparseAttention(
            hidden_size=hidden_size,
            index_head_dim=index_head_dim,
            index_n_heads=index_n_heads,
            index_topk=index_topk,
            kv_lora_rank=kv_lora_rank,
            num_attention_heads=num_attention_heads,
            q_lora_rank=q_lora_rank,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
        )
        self.ffn = DenseFFN(
            in_features=hidden_size,
            hidden_features=intermediate_size,
        )
        self.lm_head = LMHead(
            hidden_size=hidden_size,
            vocab_size=vocab_size,
        )
        self.moe = MOE(
            in_features=hidden_size,
            hidden_features=moe_intermediate_size,
            topk=topk,
            num_experts=num_experts,
            zero_expert_num=zero_expert_num,
        )
        self.mtp = MTP(
            vocab_size=vocab_size,
            in_features=hidden_size,
            hidden_features=intermediate_size,
            index_head_dim=index_head_dim,
            index_n_heads=index_n_heads,
            index_topk=index_topk,
            kv_lora_rank=kv_lora_rank,
            num_attention_heads=num_attention_heads,
            q_lora_rank=q_lora_rank,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
        )
        self.num_layers = num_layers
        self.num_mtp_layers = num_mtp_layers

    def get_flops(self, num_tokens, seq_lens):
        total_flops = 0
        total_flops += self.oe.get_flops(num_tokens)
        for _ in range(self.num_layers):
            total_flops += self.sparse_attention.get_flops(num_tokens, seq_lens)
            total_flops += self.sparse_attention.get_flops(num_tokens, seq_lens, calc_index_flops=False)
            total_flops += self.ffn.get_flops(num_tokens) * 2
            total_flops += self.moe.get_flops(num_tokens)
        total_flops += self.lm_head.get_flops(num_tokens)
        total_flops += self.mtp.get_flops(num_tokens, seq_lens) * self.num_mtp_layers
        return total_flops


class HardwareSpec:
    name: str
    flops: float
    _registry: dict[str, "HardwareSpec"] = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if hasattr(cls, "name"):
            HardwareSpec._registry[cls.name] = cls()


class H20_141G(HardwareSpec):  # noqa: N801
    name = "H20-141G"
    flops = 148 * 1e12


class NPU_910B2(HardwareSpec):  # noqa: N801
    name = "NPU-910B2"
    flops = 352 * 1e12


class HardWareMFU:
    def __init__(self, hardware, world_size):
        self.hardware = hardware
        self.world_size = world_size
        self.hardware_spec_table = HardwareSpec._registry

    def _get_hardware_total(self):
        return self.hardware_spec_table[self.hardware].flops * self.world_size

    def compute_mfu(self, achieved_flops, elapsed_seconds):
        if elapsed_seconds <= 0:
            return 0.0
        peak = self._get_hardware_total()
        return achieved_flops / elapsed_seconds / peak


class PrefillProfiler:
    def __init__(self, model_config, hardware_mfu=None):
        self.model_config = model_config
        self.hardware_mfu = hardware_mfu

    def profile(self, seq_lens, tokens, elapsed_seconds=None):
        total_flops = self.model_config.get_flops(tokens, seq_lens)
        print(f"Total FLOPs for prefill: {total_flops / 1e12:.2f} TFLOPs")
        result = {"total_flops": total_flops}
        if elapsed_seconds is not None:
            tflops_per_sec = total_flops / elapsed_seconds / 1e12
            print(f"Throughput: {tflops_per_sec:.2f} TFLOPs/s")
            result["tflops_per_sec"] = tflops_per_sec
            if self.hardware_mfu is not None:
                mfu = self.hardware_mfu.compute_mfu(total_flops, elapsed_seconds)
                print(f"MFU: {mfu * 100:.2f}%")
                result["mfu"] = mfu
        return result


MODEL_CONFIG_REGISTRY = {
    "longcat_pro": LongcatProModelConfig,
}


class MFURunner:
    def __init__(self, hardware, model, hf_config, world_size):
        if model not in MODEL_CONFIG_REGISTRY:
            raise ValueError(f"Unknown SGLANG_MFU_MODEL={model!r}, supported={list(MODEL_CONFIG_REGISTRY)}")
        if hardware not in HardwareSpec._registry:
            raise ValueError(f"Unknown SGLANG_MFU_HARDWARE={hardware!r}, supported={list(HardwareSpec._registry)}")
        self.hardware = hardware
        self.model = model
        self.world_size = world_size
        self.model_config = MODEL_CONFIG_REGISTRY[model](hf_config=hf_config)
        self.hardware_mfu = HardWareMFU(hardware=hardware, world_size=world_size)

    def measure(self, num_tokens, seq_lens, elapsed_seconds):
        if not num_tokens or elapsed_seconds is None or elapsed_seconds <= 0:
            return {}
        total_flops = self.model_config.get_flops(num_tokens, seq_lens)
        tflops_per_sec = total_flops / elapsed_seconds / 1e12
        mfu = self.hardware_mfu.compute_mfu(total_flops, elapsed_seconds)
        return {
            "total_flops": total_flops,
            "tflops_per_sec": tflops_per_sec,
            "mfu": mfu,
        }

    @classmethod
    def maybe_from_env(cls, hf_config, world_size):
        import os

        hardware = os.environ.get("SGLANG_MFU_HARDWARE")
        model = os.environ.get("SGLANG_MFU_MODEL")
        if not hardware or not model:
            return None
        return cls(
            hardware=hardware,
            model=model,
            hf_config=hf_config,
            world_size=world_size,
        )


if __name__ == "__main__":
    model_config = LongcatProModelConfig(hf_config=None)
    hardware_mfu = HardWareMFU(hardware="H20-141G", world_size=8)
    profiler = PrefillProfiler(model_config, hardware_mfu=hardware_mfu)
    profiler.profile(tokens=[5120, 5120], seq_lens=[5120, 5120], elapsed_seconds=10.0)

    npu_hardware_mfu = HardWareMFU(hardware="NPU-910B2", world_size=16)
    npu_profiler = PrefillProfiler(model_config, hardware_mfu=npu_hardware_mfu)
    npu_profiler.profile(tokens=[5120, 5120], seq_lens=[5120, 5120], elapsed_seconds=10.0)
