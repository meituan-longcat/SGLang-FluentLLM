from __future__ import annotations

import torch
import logging
from sglang.srt.distributed import get_moe_tensor_parallel_world_size
from sglang.srt.layers.quantization.compressed_tensors.fused_moe import fused_marlin_moe

logger = logging.getLogger(__name__)


class WNA16MoeExecutor:
    def __init__(self, quant_config):
        config = quant_config.target_scheme_map["Linear"].get("weights")
        self.moe_tp_size = get_moe_tensor_parallel_world_size()
        self.num_bits = config.num_bits
        self.actorder = config.actorder
        self.is_k_full = not self.actorder and self.moe_tp_size > 1

    def forward(self, layer, hidden_states: torch.Tensor, topk_output, num_global_tokens=None, max_num_tokens_per_gpu=None):
        # TP version, temporarily skipping dispatch && combine
        final_hidden_states = fused_marlin_moe(
            hidden_states,
            layer.w13_weight_packed,
            layer.w2_weight_packed,
            layer.w13_weight_scale,
            layer.w2_weight_scale,
            topk_output.topk_weights,
            topk_output.topk_ids,
            g_idx1=layer.w13_weight_g_idx,
            g_idx2=layer.w2_weight_g_idx,
            sort_indices1=layer.w13_g_idx_sort_indices,
            sort_indices2=layer.w2_g_idx_sort_indices,
            num_bits=self.num_bits,
            is_k_full=self.is_k_full,
        )
        return final_hidden_states
