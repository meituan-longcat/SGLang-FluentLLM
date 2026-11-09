"""NPU 侧 DFlash draft 模型。复用 GPU ``dflash.py``, NPU override:
- ``set_embed_module_and_head``: 替换整个 embed_tokens 模块使 draft 复用 target 的 comm_group/shard。
- ``load_weights``: bf16 强转 + decoder_comm_manager no-op + o_proj hook/down_proj reduce 补 reduce。
"""

import logging

import torch

from sglang.srt.env import global_server_args_dict
from sglang.srt.layers.attention.dp_attention import get_attention_tp_group
from sglang.srt.layers.embedding.vocab_parallel_embedding import VocabParallelEmbedding
from sglang.srt.models.dflash import (
    DFlashAttention,
    DFlashDecoderLayer,
)
from sglang.srt.models.dflash import (
    DFlashDraftModel as _DFlashDraftModel,
)
from sglang.srt.utils import device_synchronize, get_device_module

__all__ = ["DFlashAttention", "DFlashDecoderLayer", "DFlashDraftModel", "EntryClass"]

logger = logging.getLogger(__name__)


class DFlashDraftModel(_DFlashDraftModel):
    """NPU 侧 DFlash draft 模型。"""

    def _use_replicated_lm_head(self) -> bool:
        # lm_head.weight 与 target 共享, vocab 分片判据须与 target 一致。
        return bool(
            global_server_args_dict["enable_dp_attention"]
            and not global_server_args_dict["npu_lmhead_tp_size"]
        )

    def set_embed_module_and_head(self, embed_tokens, head):
        # 替换整个模块 (非仅 weight): 复用 target 的 comm_group/shard_indices。
        self.embed_tokens = embed_tokens
        del self.lm_head.weight
        self.lm_head.weight = head
        get_device_module().empty_cache()
        device_synchronize()

    def set_embed_and_head(self, embed, head):
        # OE 传 VocabParallelEmbedding 模块 -> 替换模块; 非 OE 传 weight tensor -> 仅换 weight。
        if isinstance(embed, VocabParallelEmbedding):
            return self.set_embed_module_and_head(embed, head)
        del self.embed_tokens.weight
        del self.lm_head.weight
        self.embed_tokens.weight = embed
        self.lm_head.weight = head
        get_device_module().empty_cache()
        device_synchronize()

    def load_weights(self, weights):
        super().load_weights(weights)
        # NPU aclnn 算子要求 dtype 一致 (bf16), ckpt 可能 fp16, 全部强转。
        for _name, param in self.named_parameters():
            if param.dtype == torch.float16:
                param.data = param.data.to(torch.bfloat16)

        # decoder_comm_manager 覆盖为 no-op (NPU 下 attn_rsag 恒 None); 被跳过的 reduce 由下方补。
        from sglang.srt.distributed.decoder_comm_manager import DecoderCommMananger
        from sglang.srt.distributed.parallel_strategy import (
            AttnParallelStrategy,
            DenseParallelStategy,
        )
        from sglang.srt.env import global_server_args_dict

        for layer in self.layers:
            layer.decoder_comm_manager = DecoderCommMananger(
                layer_id=layer.layer_id,
                attn_parallel_strategy=AttnParallelStrategy.DATA_PARALLEL,
                dense_parallel_strategy=DenseParallelStategy.REPLICATED,
                moe_parallel_strategy=global_server_args_dict["moe_parallel_strategy"],
                is_moe_layer=False,
                num_layers=self.config.num_hidden_layers,
            )

        # 补 reduce: down_proj reduce_results=True; o_proj 用 hook 直接 all_reduce。
        attn_tp_group = get_attention_tp_group()

        def _all_reduce_o_proj(module, args, output):
            out, out_bias = output
            torch.distributed.all_reduce(out, group=attn_tp_group.device_group)
            return out, out_bias

        for layer in self.layers:
            o_proj = layer.self_attn.o_proj
            # load_weights 可能被多个 model_runner 各调一次, hook 重复注册会 all_reduce 多次。
            if attn_tp_group.world_size > 1 and not getattr(o_proj, "_dflash_ar_hooked", False):
                o_proj.register_forward_hook(_all_reduce_o_proj)
                o_proj._dflash_ar_hooked = True
            if hasattr(layer.mlp.down_proj, "reduce_results"):
                layer.mlp.down_proj.reduce_results = True


EntryClass = DFlashDraftModel
