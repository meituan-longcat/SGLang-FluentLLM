"""DFLASH 材化计算体 (图内路径)。

把 target 多层 aux hidden 投影成 draft 各层的 KV 写进 draft pool。与 eager 路径
``DFlashWorker._materialize_target_hidden_to_draft_kv`` 语义等价, 但被 target 的 ge 图
trace, 故不含 python 分支与 host 同步。
"""

from __future__ import annotations

import torch
from torch import nn


class DFlashMaterializer(nn.Module):
    """材化计算体, 挂在 target 的 logits_processor 上供 ge 图调用。

    **不注册为 target 的子模块** (调用方用 ``object.__setattr__`` 挂载): 否则 draft 的
    量化层会被 target 的 ``named_modules()`` 二次 ``process_weights_after_loading``,
    静默改坏权重。
    """

    def __init__(
        self,
        draft_model: nn.Module,
        draft_pool,
        max_rope_position: int,
        max_num_tokens: int,
    ) -> None:
        super().__init__()
        self.draft_model = draft_model
        self.draft_pool = draft_pool
        self.max_rope_position = int(max_rope_position)

        # rope 的 q 占位: 输出被丢弃且不被写脏 (forward_native 非原地), 故常量复用;
        # 各层 head 布局同源于一份 config, 全层共用一份。
        attn0 = draft_model.layers[0].self_attn
        self._dummy_q: torch.Tensor = torch.zeros(
            (max_num_tokens, attn0.num_kv_heads, attn0.head_dim),  # [N_max, Nkv, D]
            dtype=draft_pool.dtype,
            device=draft_pool.device,
        )

    @torch.no_grad()
    def forward(
        self,
        target_hidden: torch.Tensor,
        positions: torch.Tensor,
        cache_loc: torch.Tensor,
    ) -> None:
        """[N, K*H] -> 逐 draft 层的 (k, v) [N, Nkv, D] -> 写 draft pool。

        N = padded_bs * block_size, 与 cache_loc / positions 等长。
        """
        # 非原地: positions 是 target 的 forward_batch.positions, 原地会污染它。
        positions = positions.clamp(max=self.max_rope_position)
        context_hidden = self.draft_model.project_target_hidden(target_hidden)  # [N, H]
        dummy_q = self._dummy_q[: target_hidden.shape[0]]  # 切片长度是每档图的编译期常量

        for layer in self.draft_model.layers:
            attn = layer.self_attn
            k, v = attn.kv_proj_only(context_hidden)  # [N, Nkv*D] 各一份
            k = k.reshape(-1, attn.num_kv_heads, attn.head_dim).contiguous()
            v = v.reshape(-1, attn.num_kv_heads, attn.head_dim).contiguous()
            k = attn.apply_k_norm(k)
            _, k = attn.rotary_emb(positions, dummy_q, k)
            self.draft_pool.set_kv_buffer(attn.attn, cache_loc, k.contiguous(), v)
