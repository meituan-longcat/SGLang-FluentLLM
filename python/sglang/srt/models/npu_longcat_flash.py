import re

import torch
import torch_npu
from typing import Any, List, Optional, Tuple, Union, Iterable

from transformers.configuration_utils import PretrainedConfig
from sglang.srt.configs import FLASHConfig
from sglang.srt.layers.npu_over_embedding import NpuOverEmbedding
from sglang.srt.distributed.model_tensor_tracer import get_load_number_layers
from sglang.srt.layers.attention.npu_attn.flash_attn import CacheConfig, DeepseekNSAWithMLA
from sglang.srt.layers.attention.npu_attn.deepseek_mla import DeepseekV2MLAAttention
from sglang.srt.layers.dense.npu_mlp import NPUParallelMLP
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.rotary_embedding import RotaryEmbeddingCosSinCache
from sglang.srt.layers.moe.npu_moe.layer import NpuEPMoE
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import ReplicatedLinear
from sglang.srt.layers.moe.npu_moe.router_topk import Router, LongCatTopK
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.utils import get_layer_id, PPMissingLayer
from sglang.srt.managers.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.managers.expert_location import ModelConfigForExpertLocation

from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.env import global_server_args_dict, ENV
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.npu_deepseek_v2 import DeepseekV2AttentionMLA
from sglang.srt.npu.utils import (
    npu_limit_core,
    npu_super_kernel,
    npu_stream_switch
)

from sglang.srt.utils import make_layers, get_colorful_logger, LazyValue, add_prefix
from sglang.srt.distributed import (
    get_ep_group,
    get_attn_tp_group,
    get_attn_tp_world_size,
    get_expert_model_parallel_world_size,
    get_mlp_tp_group_cross,
    get_mlp_tp_group, get_attn_ep_group, get_pp_group
)

logger = get_colorful_logger(__name__)

class LongcatMoe(torch.nn.Module):
    """
    Routes input to one of N MLP "experts"
    """

    def __init__(
        self,
        config,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.config = config
        self.num_experts = config.n_routed_experts
        self.zero_expert_num = getattr(self.config, "zero_expert_num", 0)
        self.zero_expert_type = getattr(self.config, "zero_expert_type", None)
        self.layer_id = layer_id
        self.topk = config.moe_topk
        self.quant_config = quant_config
        self.moe_ep_size = get_expert_model_parallel_world_size()
        self.num_physical_experts = self.num_experts + global_server_args_dict["ep_num_redundant_experts"]
        self.num_local_experts = self.num_physical_experts // self.moe_ep_size

        routed_scaling_factor = getattr(config, "routed_scaling_factor", 1.0)

        router_dtype = torch.float32  # warning: maybe use router_dtype in class FLASHConfig
        self.router = Router(config, layer_id, self.num_experts + self.zero_expert_num, router_dtype)
        self.topk = LongCatTopK(self.topk, self.router.e_score_correction_bias, routed_scaling_factor,
                                self.num_experts, router_dtype, layer_id)

        self.experts = NpuEPMoE(
            layer_id=layer_id,
            hidden_size=config.hidden_size,
            intermediate_size=getattr(config, "expert_ffn_hidden_size", config.ffn_hidden_size),
            hidden_act=config.hidden_act,
            num_experts=self.num_experts,
            zero_expert_num=self.zero_expert_num,
            zero_expert_type=self.zero_expert_type,
            reduce_results=True,
            quant_config=quant_config,
            moe_chunked_prefill_size=global_server_args_dict["npu_moe_chunked_prefill_size"],
        )

    def get_moe_routed_weights(self):
        import inspect
        def is_distrbiuted_expert_weights(weight_name, x) -> bool:
            return weight_name not in ["correction_bias"] \
                and "shared_experts" not in weight_name \
                and 'w13_smooth_scale' not in weight_name
        def assert_data_wrapper(weight_name, x):
            assert x.data.shape[0] == self.num_local_experts, \
                   f"shape of tensor {weight_name} is {x.data.shape} not match {self.num_local_experts}"
            return x.data
        return [
            assert_data_wrapper(name, x)
            for name, x in self.experts.named_parameters()
            if is_distrbiuted_expert_weights(name, x)
        ]

    def get_moe_device_local_routed_weights(self):
        def is_local_expert_weights(weight_name, x) -> bool:
            return weight_name == 'w13_smooth_scale_total'
        def assert_data_wrapper(weight_name, x):
            assert x.data.shape[0] == self.num_physical_experts, \
                   f"shape of tensor {weight_name} is {x.data.shape} not match {self.num_experts}"
            return x.data
        return [
            assert_data_wrapper(name, x)
            for name, x in self.experts.named_parameters()
            if is_local_expert_weights(name, x)
        ]

    def forward(self, hidden_states, forward_batch: ForwardBatch):
        router_logits = self.router(hidden_states)
        expert_topks, expert_weights, _ = self.topk(router_logits, hidden_states.dtype)
        expert_output = self.experts(hidden_states, expert_topks, expert_weights, forward_batch)
        return expert_output


class ParallelShortcutTransformerLayer(torch.nn.Module):
    def __init__(
        self,
        config: FLASHConfig,
        layer_id: int,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        moe_stream: Optional[torch.npu.Stream] = None,
    ) -> None:
        super().__init__()
        self.layer_id = layer_id

        self.attention_method = getattr(config, "attention_method", None)
        assert config.n_routed_experts is not None
        self.mlp = LongcatMoe(config, layer_id, quant_config)
        self.attn_gemma_ffn_postnorm = getattr(config, "attn_gemma_ffn_postnorm", False)
        self.moe_stream = moe_stream

        if self.attention_method == "MLA":
            self.self_attn = torch.nn.ModuleList(
                [
                    DeepseekV2MLAAttention(
                        config,
                        hidden_size=config.hidden_size,
                        num_heads=config.num_attention_heads,
                        qk_nope_head_dim=config.qk_nope_head_dim,
                        qk_rope_head_dim=config.qk_rope_head_dim,
                        v_head_dim=config.v_head_dim,
                        q_lora_rank=config.q_lora_rank,
                        kv_lora_rank=config.kv_lora_rank,
                        bias=getattr(config,"add_qkv_bias",False),
                        rope_theta=config.rope_theta,
                        rms_norm_eps=config.rms_norm_eps,
                        rope_scaling=getattr(config,"rope_scaling",None),
                        max_position_embeddings=config.max_position_embeddings,
                        cache_config=cache_config,
                        quant_config=None,
                        layer_id=self.layer_id,
                        reduce_results=True,
                    ),
                    DeepseekV2MLAAttention(
                        config,
                        hidden_size=config.hidden_size,
                        num_heads=config.num_attention_heads,
                        qk_nope_head_dim=config.qk_nope_head_dim,
                        qk_rope_head_dim=config.qk_rope_head_dim,
                        v_head_dim=config.v_head_dim,
                        q_lora_rank=config.q_lora_rank,
                        kv_lora_rank=config.kv_lora_rank,
                        bias=getattr(config,"add_qkv_bias",False),
                        rope_theta=config.rope_theta,
                        rms_norm_eps=config.rms_norm_eps,
                        rope_scaling=getattr(config,"rope_scaling",None),
                        max_position_embeddings=config.max_position_embeddings,
                        cache_config=cache_config,
                        quant_config=None,
                        layer_id=self.layer_id,
                        reduce_results=True,
                    ),
                ]
            )
        else:
            raise NotImplementedError

        if self.attn_gemma_ffn_postnorm:
            self.emb_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.input_layernorm = torch.nn.ModuleList(
            [
                RMSNorm(config.hidden_size, eps=config.rms_norm_eps),
                RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            ]
        )

        self.post_attention_layernorm = torch.nn.ModuleList(
            [
                RMSNorm(config.hidden_size, eps=config.rms_norm_eps),
                RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            ]
        )

        self.mlps = torch.nn.ModuleList(
            [
                NPUParallelMLP(
                    hidden_size=config.hidden_size,
                    intermediate_size=config.ffn_hidden_size,
                    hidden_act=config.hidden_act,
                    quant_config=quant_config,
                    bias=False,
                ),
                NPUParallelMLP(
                    hidden_size=config.hidden_size,
                    intermediate_size=config.ffn_hidden_size,
                    hidden_act=config.hidden_act,
                    quant_config=quant_config,
                    bias=False,
                ),
            ]
        )

    def forward(
            self,
            hidden_states: torch.Tensor,
            rotary_emb,
            positions: torch.Tensor,
            kv_cache: torch.Tensor,
            forward_batch: ForwardBatch,
    ):
        if forward_batch.all_decode_or_idle:
            if get_attn_tp_group().world_size == get_mlp_tp_group().world_size:
                return self.forward_decode_3b(rotary_emb, positions, hidden_states, kv_cache, forward_batch)
            else:
                return self.forward_decode(rotary_emb, positions, hidden_states, kv_cache, forward_batch)
        else:
            if get_ep_group().world_size == 16 and get_attn_tp_group().world_size == 16 and get_mlp_tp_group().world_size == 8:
                return self.forward_prefill_26B(rotary_emb, positions, hidden_states, kv_cache, forward_batch)
            elif get_mlp_tp_group().world_size == 1:
                return self.forward_prefill_dense_tp1(rotary_emb, positions, hidden_states, kv_cache, forward_batch)
            else:
                return self.forward_prefill(rotary_emb, positions, hidden_states, kv_cache, forward_batch)


    def forward_decode_3b(
            self,
            rotary_emb,
            positions: torch.Tensor,
            hidden_states: torch.Tensor,
            kv_cache: torch.Tensor,
            forward_batch: ForwardBatch):
        # pre mlp output for petch weight contrl in current layer
        pre_mlp = None
        if isinstance(hidden_states, tuple):
            hidden_states, pre_mlp = hidden_states
        aic, aiv = 24, 48 # TODO
        main_aic_limit = aic - ENV.npu_moe_core_num
        main_aiv_limit = aiv // aic * main_aic_limit

        moe_aic_limit = aic - ENV.npu_moe_core_num
        moe_aiv_limit = aiv // aic * moe_aic_limit
        ## 1: attn 0
        i = 0
        # use full core for first attn
        with npu_limit_core(main_aic_limit, main_aiv_limit, forward_batch.can_run_with_graph):
            with npu_super_kernel(f'main', 'stream-fusion=1', flag=forward_batch.can_run_with_graph):
                residual = hidden_states
                hidden_states = self.input_layernorm[i](hidden_states)
                if pre_mlp is not None:
                    self.self_attn[i].prefetch_half(pre_mlp)
                hidden_states, attn_out_before_comm = self.self_attn[i](
                    rotary_emb=rotary_emb,
                    positions=positions,
                    hidden_states=hidden_states,
                    kv_cache=kv_cache[i],
                    forward_batch=forward_batch,
                    return_ctrl=True,
                    atten_id=0,
                )
                hidden_states = hidden_states + residual
                ## 2: mlp 0
                residual = hidden_states
                hidden_states = self.post_attention_layernorm[i](hidden_states)
                sc_hidden_states = hidden_states
                # prefetch weight
                self.mlps[i].prefetch_gateup(attn_out_before_comm)
                prefetch_tensor = self.mlp.router.classifier.weight
                torch_npu.npu_prefetch(prefetch_tensor, attn_out_before_comm,
                                    prefetch_tensor.numel() * prefetch_tensor.element_size())

        with npu_stream_switch(forward_batch.can_run_with_graph, 'scmoe_steam'):
            with npu_limit_core(moe_aic_limit, moe_aiv_limit, forward_batch.can_run_with_graph):
                with npu_super_kernel(f'moe_{self.layer_id}', 'stream-fusion=1', flag=forward_batch.can_run_with_graph):
                    origin_len=sc_hidden_states.shape[0]
                    padding_len=(get_attn_tp_world_size()-sc_hidden_states.shape[0]%get_attn_tp_world_size())%get_attn_tp_world_size()
                    sc_hidden_states=torch.cat([sc_hidden_states, torch.zeros([padding_len, sc_hidden_states.shape[1]],
                                                                              dtype=sc_hidden_states.dtype,
                                                                              device=sc_hidden_states.device)], dim=0)

                    sp_len=sc_hidden_states.shape[0]//get_attn_tp_world_size()
                    sc_hidden_states=sc_hidden_states[
                        sp_len*get_attn_tp_group().rank_in_group:sp_len*(get_attn_tp_group().rank_in_group+1)]
                    shortcut_mlp_output=self.mlp(sc_hidden_states, forward_batch)
                    shortcut_mlp_output=get_attn_ep_group().all_gather(shortcut_mlp_output, dim=0)[:origin_len]

        with npu_limit_core(main_aic_limit, main_aiv_limit, forward_batch.can_run_with_graph):
            with npu_super_kernel(f'main', 'stream-fusion=1', flag=forward_batch.can_run_with_graph):
                # hidden_states = self.mlps[i].pre_comm(hidden_states)  # TODO: pass pad_size for eagle mode
                self.mlps[i].prefetch_down(hidden_states)
                # reduce_scatter for dp decode
                hidden_states, gate_out, _ = self.mlps[i](hidden_states, reduce_type='all_reduce',
                                                          return_ctrl=True)
                hidden_states = residual + hidden_states

                ## 3 attn 1
                i = 1
                residual = hidden_states
                hidden_states = self.input_layernorm[i](hidden_states)
                self.self_attn[i].prefetch_full(gate_out)
                hidden_states = self.self_attn[i](
                    rotary_emb=rotary_emb,
                    positions=positions,
                    hidden_states=hidden_states,
                    kv_cache=kv_cache[i],
                    forward_batch=forward_batch,
                    atten_id=1,
                )
                hidden_states = hidden_states + residual
                ## 4: mlp 1
                residual = hidden_states
                hidden_states = self.post_attention_layernorm[i](hidden_states)
                # hidden_states = self.mlps[i].pre_comm(hidden_states)  # TODO: pass pad_size for eagle mode
                hidden_states, _, down_out = self.mlps[i](hidden_states,
                                                          reduce_type='all_reduce',
                                                          return_ctrl=True)
                hidden_states = residual + hidden_states
                output = hidden_states + shortcut_mlp_output

        return output, down_out  # down_out for next layer prefetch weight dependency

    def forward_decode(
            self,
            rotary_emb,
            positions: torch.Tensor,
            hidden_states: torch.Tensor,
            kv_cache: torch.Tensor,
            forward_batch: ForwardBatch):
        # pre mlp output for petch weight contrl in current layer
        pre_mlp = None
        if isinstance(hidden_states, tuple):
            hidden_states, pre_mlp = hidden_states
        aic, aiv = 24, 48 # TODO
        ## 1: attn 0
        i = 0
        # use full core for first attn
        with npu_limit_core(aic, aiv, forward_batch.can_run_with_graph):
            with npu_super_kernel(f'attn0_{self.layer_id}', 'stream-fusion=1', flag=forward_batch.can_run_with_graph):
                residual = hidden_states
                hidden_states = self.input_layernorm[i](hidden_states)
                if pre_mlp is not None:
                    self.self_attn[i].prefetch_half(pre_mlp)
                hidden_states, attn_out_before_comm = self.self_attn[i](
                    rotary_emb=rotary_emb,
                    positions=positions,
                    hidden_states=hidden_states,
                    kv_cache=kv_cache[i],
                    forward_batch=forward_batch,
                    return_ctrl=True,
                    atten_id=0,
                )
                hidden_states = hidden_states + residual
                ## 2: mlp 0
                residual = hidden_states
                hidden_states = self.post_attention_layernorm[i](hidden_states)
        sc_hidden_states = hidden_states
        # prefetch weight
        self.mlps[i].prefetch_gateup(attn_out_before_comm)
        prefetch_tensor = self.mlp.router.classifier.weight
        torch_npu.npu_prefetch(prefetch_tensor, attn_out_before_comm,
                            prefetch_tensor.numel() * prefetch_tensor.element_size())
        aic_limit = aic - ENV.npu_moe_core_num
        aiv_limit = aiv // aic * aic_limit
        with npu_limit_core(aic_limit, aiv_limit, forward_batch.can_run_with_graph):
            with npu_super_kernel(f'attn1_{self.layer_id}', 'stream-fusion=1', flag=forward_batch.can_run_with_graph):
                # TODO 1: dense tp != attn tp
                hidden_states = self.mlps[i].pre_comm(hidden_states)  # TODO: pass pad_size for eagle mode
                self.mlps[i].prefetch_down(hidden_states)
                # reduce_scatter for dp decode
                hidden_states, gate_out, _ = self.mlps[i](hidden_states, reduce_type='reduce_scatter',
                                                          return_ctrl=True)
                hidden_states = residual + hidden_states

                ## 3 attn 1
                i = 1
                residual = hidden_states
                hidden_states = self.input_layernorm[i](hidden_states)
                self.self_attn[i].prefetch_full(gate_out)
                hidden_states = self.self_attn[i](
                    rotary_emb=rotary_emb,
                    positions=positions,
                    hidden_states=hidden_states,
                    kv_cache=kv_cache[i],
                    forward_batch=forward_batch,
                    atten_id=1,
                )
                hidden_states = hidden_states + residual
                ## 4: mlp 1
                residual = hidden_states
                hidden_states = self.post_attention_layernorm[i](hidden_states)
                hidden_states = self.mlps[i].pre_comm(hidden_states)  # TODO: pass pad_size for eagle mode
                hidden_states, _, down_out = self.mlps[i](hidden_states,
                                                          reduce_type='reduce_scatter',
                                                          return_ctrl=True)
                hidden_states = residual + hidden_states

        ## 5.scmoe
        aic_limit = ENV.npu_moe_core_num
        aiv_limit = aiv // aic * aic_limit
        with npu_stream_switch(forward_batch.can_run_with_graph, 'scmoe_steam'):
            with npu_limit_core(aic_limit, aiv_limit, forward_batch.can_run_with_graph):
                with npu_super_kernel(f'moe_{self.layer_id}', 'stream-fusion=1', flag=forward_batch.can_run_with_graph):
                    origin_len=sc_hidden_states.shape[0]
                    padding_len=(get_attn_tp_world_size()-sc_hidden_states.shape[0]%get_attn_tp_world_size())%get_attn_tp_world_size()
                    sc_hidden_states=torch.cat([sc_hidden_states, torch.zeros([padding_len, sc_hidden_states.shape[1]], dtype=sc_hidden_states.dtype, device=sc_hidden_states.device)], dim=0)

                    sp_len=sc_hidden_states.shape[0]//get_attn_tp_world_size()
                    sc_hidden_states=sc_hidden_states[
                        sp_len*get_attn_tp_group().rank_in_group:sp_len*(get_attn_tp_group().rank_in_group+1)]
                    shortcut_mlp_output=self.mlp(sc_hidden_states, forward_batch)
                    shortcut_mlp_output=get_attn_ep_group().all_gather(shortcut_mlp_output, dim=0)[:origin_len]

        ## 6. fuse scmoe + mlp
        # add will fused to rmsnorm of next layer, add superkernel name scope the same with that rmsnorm
        with npu_super_kernel(f'attn0_{self.layer_id + 1}', 'stream-fusion=1', flag=forward_batch.can_run_with_graph):
            output = hidden_states + shortcut_mlp_output

        return output, down_out  # down_out for next layer prefetch weight dependency

    def moe_mm1(self, expand_x, tokens_per_expert):
        # gmm1
        out_dtype = torch.int32 if expand_x.dtype == torch.int8 else torch.bfloat16
        return torch_npu.npu_grouped_matmul(
            [expand_x],
            [self.mlp.experts.w13_weight],
            bias=None,
            group_list=tokens_per_expert,
            split_item=3,      # 3 means combine result tokens belong to diff expert to a single tensor
            output_dtype=out_dtype,
            group_type=0,      # 0 means expand_x is grouped in m-axis (1 for k-axis is used in training backward)
            group_list_type=1  # 1 means group_list[i] is count for expert i (count mode), otherwise 0 is comsum mode
        )[0]

    def moe_act_mm2(self, hidden_states, tokens_per_expert, pertoken_scale=None):
        if hidden_states.dtype in [torch.int8, torch.int32]:
            moe_output, pertoken_scale = torch_npu.npu_dequant_swiglu_quant(
                hidden_states,
                weight_scale=self.mlp.experts.w13_weight_scale.to(torch.float),
                group_index=tokens_per_expert,
                activation_scale=pertoken_scale,
                activate_left=True,
                quant_mode=1,
                quant_scale=self.mlp.experts.w2_smooth_scale
            )
            output = torch_npu.npu_grouped_matmul(
                [moe_output],
                [self.mlp.experts.w2_weight],
                scale=[self.mlp.experts.w2_weight_scale],
                per_token_scale=[pertoken_scale.float()],
                bias=None,
                group_list=tokens_per_expert,
                split_item=3,
                output_dtype=torch.bfloat16,
                group_type=0,
                group_list_type=1
            )[0]
        else:
            moe_output = torch_npu.mlp_split_swiglu(
                hidden_states,
                tokens_per_expert.to(torch.int32),
                local_exp_start=0,
                local_exp_end=self.mlp.experts.num_local_experts
            )
            output = torch_npu.npu_grouped_matmul(
                [moe_output],
                [self.mlp.experts.w2_weight],
                bias=None,
                group_list=tokens_per_expert,
                split_item=3,
                output_dtype=torch.bfloat16,
                group_type=0,
                group_list_type=1
            )[0]
        return output

    def forward_prefill(
        self,
        rotary_emb,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        forward_batch: ForwardBatch):

        # a baseline implement for prefill, support dp/tp, no overlap
        global_sp_token_num=forward_batch.global_sp_num_tokens
        mlp_sp_token_num=get_mlp_tp_group().get_local_sp_token_num(global_sp_token_num)

        # main
        quant = self.mlp.experts.w13_weight.dtype == torch.int8

        i = 0
        residual = hidden_states
        if not forward_batch.forward_mode.is_idle():
            hidden_states = self.input_layernorm[i](hidden_states)
            hidden_states = self.self_attn[i](
                rotary_emb,
                positions,
                hidden_states,
                kv_cache[i],
                forward_batch,
                gather_qkv=True,
                reduce_type='reduce_scatter',
                atten_id=0
            )

            hidden_states, residual = self.post_attention_layernorm[i](hidden_states, residual)
        moe_hidden_states = hidden_states
        main_stream=torch.npu.current_stream()
        self.moe_stream.wait_stream(main_stream)

        if quant:
            hidden_states=self.mlps[i].pre_quant(hidden_states)
        hidden_states = self.mlps[i].pre_comm(hidden_states, output_split_sizes=mlp_sp_token_num)
        hidden_states=self.mlps[i](hidden_states, reduce_type='reduce_scatter', input_split_sizes=mlp_sp_token_num)

        i=1
        if not forward_batch.forward_mode.is_idle():
            hidden_states, residual=self.input_layernorm[i](hidden_states, residual)
            hidden_states=self.self_attn[i](
                rotary_emb,
                positions,
                hidden_states,
                kv_cache[i],
                forward_batch,
                gather_qkv=True,
                reduce_type='reduce_scatter',
                atten_id=1,
            )
            hidden_states, residual=self.post_attention_layernorm[i](hidden_states, residual)

        if quant:
            hidden_states=self.mlps[i].pre_quant(hidden_states)
        hidden_states=self.mlps[i].pre_comm(hidden_states, output_split_sizes=mlp_sp_token_num)
        hidden_states=self.mlps[i](hidden_states, reduce_type='reduce_scatter', input_split_sizes=mlp_sp_token_num)

        # Moe
        with torch.npu.stream(self.moe_stream):
            router_logits=self.mlp.router(moe_hidden_states)
            top_experts, expert_weights, _=self.mlp.topk(router_logits, moe_hidden_states.dtype)
            moe_output=self.mlp.experts(moe_hidden_states, top_experts, expert_weights, forward_batch)
        self.moe_stream.wait_stream(main_stream)
        main_stream.wait_stream(self.moe_stream)

        hidden_states=moe_output+hidden_states
        hidden_states=residual+hidden_states
        return hidden_states

    def forward_prefill_dense_tp1(
        self,
        rotary_emb,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        forward_batch: ForwardBatch):

        # a baseline implement for prefill, support dp/tp, no overlap
        global_sp_num_tokens=forward_batch.global_sp_num_tokens
        mlp_sp_token_num=get_mlp_tp_group().get_local_sp_token_num(global_sp_num_tokens)

        # main
        quant = self.mlp.experts.w13_weight.dtype == torch.int8
        moe_stream=self.moe_stream
        #
        # if get_attn_tp_group().rank_in_group==0:
        #     logger.info(f"{self.layer_id=} {hidden_states=}")

        i = 0
        residual = hidden_states
        if not forward_batch.forward_mode.is_idle():
            hidden_states = self.input_layernorm[i](hidden_states)
            hidden_states = self.self_attn[i](
                rotary_emb,
                positions,
                hidden_states,
                kv_cache[i],
                forward_batch,
                gather_qkv=True,
                reduce_type='reduce_scatter',
                atten_id=0
            )

            hidden_states, residual = self.post_attention_layernorm[i](hidden_states, residual)
        moe_hidden_states = hidden_states

        # Moe router
        router_logits=self.mlp.router(moe_hidden_states)
        top_experts, expert_weights, _=self.mlp.topk(router_logits, moe_hidden_states.dtype)

        if quant:
            hidden_states=self.mlps[i].pre_quant(hidden_states)
        hidden_states = self.mlps[i].pre_comm(hidden_states, output_split_sizes=mlp_sp_token_num)

        moe_stream.wait_stream(torch.npu.current_stream())
        hidden_states=self.mlps[i](hidden_states, reduce_type='reduce_scatter', input_split_sizes=mlp_sp_token_num)
        with torch.npu.stream(moe_stream):
            # Moe pre comm ag
            moe_global_hidden_states=self.mlp.experts.dispatcher_ag_rs.pre_comm(moe_hidden_states,
                                                                                sp_num_tokens=global_sp_num_tokens)
            (
                moe_global_top_experts,
                moe_global_expert_weights
            )=self.mlp.experts.dispatcher_ag_rs.pre_comm([top_experts, expert_weights],
                                                         sp_num_tokens=global_sp_num_tokens)
        torch.npu.current_stream().wait_stream(moe_stream)
        moe_stream.wait_stream(torch.npu.current_stream())

        i=1
        if not forward_batch.forward_mode.is_idle():
            hidden_states, residual=self.input_layernorm[i](hidden_states, residual)
            hidden_states=self.self_attn[i](
                rotary_emb,
                positions,
                hidden_states,
                kv_cache[i],
                forward_batch,
                gather_qkv=True,
                reduce_type='reduce_scatter',
                atten_id=1
            )
            hidden_states, residual=self.post_attention_layernorm[i](hidden_states, residual)

        if quant:
            hidden_states=self.mlps[i].pre_quant(hidden_states)
        hidden_states=self.mlps[i].pre_comm(hidden_states, output_split_sizes=mlp_sp_token_num)


        # Moe cac
        (
            expand_x, tokens_per_expert, dispatch_output
        )=self.mlp.experts.dispatcher_ag_rs.dispatch(moe_global_hidden_states, moe_global_top_experts,
                                                     skip_comm=True)
        moe_output=self.mlp.experts.quant_method.apply(self.mlp.experts, expand_x, tokens_per_expert)
        moe_output=self.mlp.experts.dispatcher_ag_rs.combine(
            moe_output, moe_global_top_experts, moe_global_expert_weights, dispatch_output, reduce_type='skip')
        moe_output_zero=self.mlp.experts.zero_expert(moe_hidden_states, top_experts, expert_weights)

        moe_stream.wait_stream(torch.npu.current_stream())
        hidden_states=self.mlps[i](hidden_states, reduce_type='reduce_scatter', input_split_sizes=mlp_sp_token_num)
        with torch.npu.stream(moe_stream):
            # Moe post comm(rs)
            moe_output = self.mlp.experts.dispatcher_ag_rs.post_comm(moe_output, 'reduce_scatter', sp_num_tokens=global_sp_num_tokens)
        torch.npu.current_stream().wait_stream(moe_stream)
        moe_stream.wait_stream(torch.npu.current_stream())
        moe_output = moe_output + moe_output_zero

        hidden_states=moe_output+hidden_states
        hidden_states=residual+hidden_states
        return hidden_states

    # for 26B, attn tp=16, dense tp8
    def forward_prefill_26B(
            self,
            rotary_emb,
            positions: torch.Tensor,
            hidden_states: torch.Tensor,
            kv_cache: torch.Tensor,
            forward_batch: ForwardBatch):
        quant = self.mlp.experts.w13_weight.dtype == torch.int8
        moe_stream = self.moe_stream

        global_sp_num_tokens = forward_batch.global_sp_num_tokens
        attn_sp_token_nums = get_attn_tp_group().get_local_sp_token_num(global_sp_num_tokens)
        mlp_sp_sum_tokens = get_mlp_tp_group().get_local_sp_token_num(global_sp_num_tokens)

        # 1. 主流：AG0_0 + attention0 + RS0_0
        i = 0
        residual = hidden_states
        hidden_states = self.input_layernorm[i](hidden_states)
        hidden_states = self.self_attn[i](
            rotary_emb,
            positions,
            hidden_states,
            kv_cache[i],
            forward_batch,
            gather_qkv=True,
            reduce_type='reduce_scatter',
            atten_id=0,
        )

        # 2. 主流: AG_0_1, Moe流: Scmoe Stage0(router0)
        hidden_states, residual = self.post_attention_layernorm[i](hidden_states, residual)
        moe_hidden_states = hidden_states
        # moe and mlp have diff smooth scale, dyname quant exec after allgather seperately
        if quant and not global_server_args_dict['npu_smooth_quant']:
            hidden_states = torch_npu.npu_dynamic_quant(hidden_states)

        ep_hidden_states = hidden_states
        moe_stream.wait_stream(torch.npu.current_stream())
        mlp_hidden_states = self.mlps[i].pre_comm(hidden_states, output_split_sizes=mlp_sp_sum_tokens)
        with torch.npu.stream(moe_stream):
            router_logits = self.mlp.router(moe_hidden_states)
            top_experts, expert_weights, _ = self.mlp.topk(router_logits, moe_hidden_states.dtype)
        torch.npu.current_stream().wait_stream(moe_stream)
        moe_stream.wait_stream(torch.npu.current_stream())

        # 3. 主流: MLP0, Moe流: Scmoe Stage0(router1->gather)
        moe_stream.wait_stream(torch.npu.current_stream())
        hidden_states = self.mlps[i](mlp_hidden_states, reduce_type='skip')
        with torch.npu.stream(moe_stream):
            moe_global_hidden_states = self.mlp.experts.dispatcher_ag_rs.pre_comm(ep_hidden_states, sp_num_tokens=global_sp_num_tokens)
            (
                global_top_experts,
                global_expert_weights
            ) = self.mlp.experts.dispatcher_ag_rs.pre_comm([top_experts, expert_weights], sp_num_tokens=global_sp_num_tokens)
        torch.npu.current_stream().wait_stream(moe_stream)
        moe_stream.wait_stream(torch.npu.current_stream())

        # 4. 主流: RS0_0 + AG1_0 + Attention1, Moe流: Scmoe Stage1(dispatch+gmm13)
        moe_stream.wait_stream(torch.npu.current_stream())
        hidden_states = self.mlps[i].post_comm(hidden_states, 'reduce_scatter', input_split_sizes=mlp_sp_sum_tokens)
        with (torch.npu.stream(moe_stream)):
            (
                expand_x, tokens_per_expert, dispatch_output
            ) = self.mlp.experts.dispatcher_ag_rs.dispatch(moe_global_hidden_states, global_top_experts, skip_comm=True)
            pertoken_scale = None
            if isinstance(expand_x, (list, tuple)):
                expand_x, pertoken_scale = expand_x
        torch.npu.current_stream().wait_stream(moe_stream)
        moe_stream.wait_stream(torch.npu.current_stream())

        i = 1
        hidden_states, residual = self.input_layernorm[i](hidden_states, residual)
        q = self.self_attn[i].q_a_proj(hidden_states)[0]
        latent_cache = self.self_attn[i].kv_a_proj_with_mqa(hidden_states)[0]
        moe_stream.wait_stream(torch.npu.current_stream())
        q = get_attn_tp_group().all_gather(q, dim=0, output_split_sizes=attn_sp_token_nums)
        latent_cache = get_attn_tp_group().all_gather(latent_cache, dim=0, output_split_sizes=attn_sp_token_nums)
        with torch.npu.stream(moe_stream):
            mm1_out = self.moe_mm1(expand_x, tokens_per_expert)
        torch.npu.current_stream().wait_stream(moe_stream)
        moe_stream.wait_stream(torch.npu.current_stream())

        hidden_states = self.self_attn[i](
            rotary_emb,
            positions,
            hidden_states,
            kv_cache[i],
            forward_batch,
            reduce_type='skip',
            extra_input=(q, latent_cache),
            atten_id=1,
        )

        # 5. 主流: RS1_0 + Scmoe, Moe流: Stage2(swiglu + gmm2)
        moe_stream.wait_stream(torch.npu.current_stream())
        hidden_states = self.self_attn[i].post_comm(hidden_states, 'reduce_scatter', input_split_sizes=attn_sp_token_nums)
        with torch.npu.stream(moe_stream):
            moe_output = self.moe_act_mm2(mm1_out, tokens_per_expert, pertoken_scale)
        torch.npu.current_stream().wait_stream(moe_stream)
        moe_stream.wait_stream(torch.npu.current_stream())

        # 6. 主流: AG1_1 + Scmoe, Moe流： Stage3_4(combine + zero expert)
        hidden_states, residual=self.post_attention_layernorm[i](hidden_states, residual)
        if quant:
            hidden_states = self.mlps[i].pre_quant(hidden_states)
        moe_stream.wait_stream(torch.npu.current_stream())
        mlp_hidden_states = self.mlps[i].pre_comm(hidden_states, output_split_sizes=mlp_sp_sum_tokens)
        with torch.npu.stream(moe_stream):
            moe_output = self.mlp.experts.dispatcher_ag_rs.combine(
                moe_output, global_top_experts, global_expert_weights, dispatch_output, reduce_type='skip')
            moe_output_zero = self.mlp.experts.zero_expert(moe_hidden_states, top_experts, expert_weights)
        torch.npu.current_stream().wait_stream(moe_stream)
        moe_stream.wait_stream(torch.npu.current_stream())

        # 7. 主流: MLP1
        with torch.npu.stream(moe_stream):
            moe_output = self.mlp.experts.dispatcher_ag_rs.post_comm(moe_output, 'reduce_scatter', sp_num_tokens=global_sp_num_tokens) + moe_output_zero
        hidden_states = self.mlps[i](mlp_hidden_states, reduce_type='skip')
        torch.npu.current_stream().wait_stream(moe_stream)
        moe_stream.wait_stream(torch.npu.current_stream())

        # 8. 主流: RS1_1
        hidden_states = self.mlps[i].post_comm(hidden_states, 'reduce_scatter', input_split_sizes=mlp_sp_sum_tokens)
        hidden_states = hidden_states + moe_output
        hidden_states = residual + hidden_states
        return hidden_states


class NPUFlashModel(torch.nn.Module):
    def __init__(
        self,
        config: Union[FLASHConfig, Any],
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.pp_group=get_pp_group()

        self.enable_over_embedding = getattr(config, "use_over_embedding", False)
        if self.enable_over_embedding:
            self.over_embedding=NpuOverEmbedding(
                num_embeddings=config.vocab_size,
                embedding_dim=config.hidden_size,
                over_embedding_m=config.over_embedding_m,
                over_embedding_k=config.oe_split_num,
                over_embedding_n=config.oe_neighbor_num,
                oe_ignore_tokens=config.oe_ignore_tokens,
            )
        else:
            self.embed_tokens=VocabParallelEmbedding(
                config.vocab_size, config.hidden_size, enable_tp=not global_server_args_dict["enable_dp_attention"])
        self.moe_stream=torch.npu.Stream()
        self.layers, self.start_layer, self.end_layer= make_layers(
            config.num_hidden_layers,
            lambda idx, prefix: ParallelShortcutTransformerLayer(
                config,
                idx,
                cache_config=cache_config,
                quant_config=quant_config,
                moe_stream=self.moe_stream,
            ),
            pp_rank=self.pp_group.rank_in_group,
            pp_size=self.pp_group.world_size,
            prefix=add_prefix("layers", prefix),
            return_index=3,
        )

        logger.info(f"{self.start_layer=} {self.end_layer=}")

        self.max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
        self.rope_scaling = getattr(config, "rope_scaling", None)
        if self.rope_scaling != None:
            from sglang.srt.layers.rotary_embedding import get_rope
            assert self.rope_scaling["rope_type"] == "deepseek_yarn"
            self.rotary_pos_emb = get_rope(config.qk_rope_head_dim,
                            rotary_dim=config.qk_rope_head_dim,
                            max_position=self.max_position_embeddings,
                            base=config.rope_theta,
                            rope_scaling=self.rope_scaling,
                            is_neox_style=True)
            self.rotary_pos_emb.prepare_for_npu()
        else:
            self.rotary_pos_emb = RotaryEmbeddingCosSinCache(
                config.qk_rope_head_dim, max_position_embeddings=self.max_position_embeddings, base=config.rope_theta,
                device=torch.npu.current_device()
            )

        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.layers_to_capture = []
        self.config = config

    @torch.inference_mode()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
    ) -> Union[torch.Tensor, PPProxyTensors]:
        if not forward_batch.all_decode_or_idle:
            enable_sp = True
            def get_sp_token_num(x, rank):
                base=x//get_attn_tp_group().world_size
                rem=x%get_attn_tp_group().world_size
                if rank < rem:
                    return base + 1
                else:
                    return base
            if forward_batch.global_num_tokens is None:
                forward_batch.global_num_tokens=[forward_batch.extend_num_tokens] * get_attn_tp_group().world_size
            forward_batch.global_sp_num_tokens=[get_sp_token_num(x, i % get_attn_tp_group().world_size) for i, x in enumerate(forward_batch.global_num_tokens)]
            # logger.info(f"{forward_batch.global_sp_num_tokens=} {forward_batch.global_num_tokens=}")
        else:
            enable_sp = False

        if self.pp_group.is_first_rank:
            if input_embeds is None:
                if self.enable_over_embedding:
                    hidden_states=self.over_embedding(input_ids, forward_batch, enable_sp=enable_sp)
                else:
                    hidden_states=self.embed_tokens(input_ids, global_sp_token_num=forward_batch.global_sp_num_tokens,
                                                    enable_sp=enable_sp)
            else:
                hidden_states = input_embeds
        else:
            pp_proxy_tensors=forward_batch.pp_proxy_tensors
            assert pp_proxy_tensors is not None
            hidden_states = pp_proxy_tensors["hidden_states"]

        cos, sin = self.rotary_pos_emb.get_cos_sin(hidden_states, self.max_position_embeddings)
        cos = cos[positions]
        sin = sin[positions]

        cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
        rotary_pos_emb = (cos, sin)
        aux_hidden_states=[]

        for i, layer in enumerate(self.layers):
            get_global_expert_distribution_recorder().set_current_layer(i)
            if not isinstance(layer, PPMissingLayer):
                if ENV.npu_enable_mla_split_kv_kr:
                    cur_kv_cache = [
                        forward_batch.token_to_kv_pool.get_kv_buffer(i * 2),
                        forward_batch.token_to_kv_pool.get_kv_buffer(i * 2 + 1)
                    ]
                else:
                    cur_kv_cache = [
                        forward_batch.token_to_kv_pool.get_key_buffer(i * 2),
                        forward_batch.token_to_kv_pool.get_key_buffer(i * 2 + 1)
                    ]
            else:
                cur_kv_cache = []
            hidden_states = layer(
                hidden_states,
                rotary_pos_emb,
                positions,
                cur_kv_cache,
                forward_batch,
            )
            get_global_expert_distribution_recorder().set_current_layer(None)
            # capture for eagle3
            if i + 1 in self.layers_to_capture:
                if isinstance(hidden_states, tuple):
                    hidden_states = hidden_states[0]
                aux_hidden_state = hidden_states.clone()
                if not forward_batch.all_decode_or_idle:
                    aux_hidden_state=get_attn_tp_group().gather_tensor(aux_hidden_state,
                                                                    global_sp_token_num=forward_batch.global_sp_num_tokens)
                aux_hidden_states.append(aux_hidden_state)

        if not self.pp_group.is_last_rank:
            return PPProxyTensors(
                {
                    "hidden_states": hidden_states,
                    "residual": torch.empty([], dtype=hidden_states.dtype, device=hidden_states.device),
                }
            )
        else:
            if isinstance(hidden_states, tuple):
                hidden_states=hidden_states[0]
            hidden_states = self.norm(hidden_states)

        if not forward_batch.all_decode_or_idle:
            hidden_states = get_attn_tp_group().gather_tensor(hidden_states, global_sp_token_num=forward_batch.global_sp_num_tokens)
        return hidden_states

        # for eagle3
        return hidden_states, aux_hidden_states

class NPUFlASHDenseDecoderLayer(torch.nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        is_nextn: bool = False,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
        self.enable_dp_attention = global_server_args_dict["enable_dp_attention"]
        self.is_longcat_flash = config.model_type == "flash"
        self.is_nextn = is_nextn
        self.self_attn = DeepseekV2MLAAttention(
            config,
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            qk_nope_head_dim=config.qk_nope_head_dim,
            qk_rope_head_dim=config.qk_rope_head_dim,
            v_head_dim=config.v_head_dim,
            q_lora_rank=(
                config.q_lora_rank if hasattr(config, "q_lora_rank") else None
            ),
            kv_lora_rank=config.kv_lora_rank,
            bias=False,
            rope_theta=config.rope_theta,
            rms_norm_eps=config.rms_norm_eps,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            cache_config=None,
            quant_config=None,
            layer_id=layer_id,
            reduce_results=True,
        )

        self.mlp = NPUParallelMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.ffn_hidden_size if hasattr(config, "ffn_hidden_size") else config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            bias=False,
        )

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.layer_id = layer_id

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
    ) -> torch.Tensor:
        cur_kv_cache=forward_batch.token_to_kv_pool.get_kv_buffer(self.layer_id)
        cur_kv_cache=list(cur_kv_cache) if isinstance(cur_kv_cache, (tuple, list)) else [cur_kv_cache]
        if forward_batch.all_decode_or_idle:
            return self.forward_decode(positions, hidden_states, cur_kv_cache, forward_batch, residual)
        else:
            return self.forward_prefill(positions, hidden_states, cur_kv_cache, forward_batch, residual)

    def forward_decode(self,
                        positions: torch.Tensor,
                        hidden_states: torch.Tensor,
                        kv_cache: torch.Tensor,
                        forward_batch: ForwardBatch,
                        residual: Optional[torch.Tensor],):
        if not forward_batch.forward_mode.is_idle():
            # residual is None in the first layer
            if residual is None:
                residual = hidden_states
                hidden_states = self.input_layernorm(hidden_states)
            else:
                hidden_states, residual = self.input_layernorm(hidden_states, residual)
            hidden_states=self.self_attn(
                rotary_emb=forward_batch.rotary_pos_emb,
                positions=positions,
                hidden_states=hidden_states,
                kv_cache=kv_cache,
                forward_batch=forward_batch
            )

            hidden_states, residual = self.post_attention_layernorm(
                hidden_states, residual
            )
        if get_attn_tp_group().world_size == get_mlp_tp_group().world_size:
            hidden_states=self.mlp(hidden_states, reduce_type='all_reduce')
        else:
            mlp_hidden_states=self.mlp.pre_comm(hidden_states)
            hidden_states=self.mlp(mlp_hidden_states, reduce_type='reduce_scatter')
        return hidden_states, residual

    def forward_prefill(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
    ):
        global_sp_token_nums=forward_batch.global_sp_num_tokens
        attn_sp_token_nums = get_attn_tp_group().get_local_sp_token_num(global_sp_token_nums)
        mlp_sp_token_nums = get_mlp_tp_group().get_local_sp_token_num(global_sp_token_nums)
        if residual is None:
            hidden_states=get_attn_tp_group().split_tensor(hidden_states, global_sp_token_nums)
            residual=hidden_states
            hidden_states=self.input_layernorm(hidden_states)
        else:
            hidden_states=get_attn_tp_group().split_tensor(hidden_states, global_sp_token_nums)
            residual=get_attn_tp_group().split_tensor(residual, global_sp_token_nums)
            hidden_states, residual=self.input_layernorm(hidden_states, residual)

        if not forward_batch.forward_mode.is_idle():
            hidden_states = self.self_attn(
                rotary_emb=forward_batch.rotary_pos_emb,
                positions=positions,
                hidden_states=hidden_states,
                kv_cache=kv_cache,
                forward_batch=forward_batch,
                gather_qkv=True,
                reduce_type='reduce_scatter'  # for prefill, decode force to rs
            )
            hidden_states, residual = self.post_attention_layernorm(
                hidden_states, residual
            )
        mlp_hidden_states=self.mlp.pre_comm(hidden_states, output_split_sizes=mlp_sp_token_nums)
        hidden_states=self.mlp(mlp_hidden_states, reduce_type='reduce_scatter', input_split_sizes=mlp_sp_token_nums)

        hidden_states=get_attn_tp_group().all_gather(hidden_states, dim=0, output_split_sizes=attn_sp_token_nums)
        residual=get_attn_tp_group().all_gather(residual, dim=0, output_split_sizes=attn_sp_token_nums)
        return hidden_states, residual


class FLASHForCausalLM(torch.nn.Module):
    def __init__(
        self,
        config: FLASHConfig,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        if get_load_number_layers() != 0:
            config.num_hidden_layers = get_load_number_layers()
        super().__init__()
        self.config = config
        self.quant_config = quant_config
        self.model = NPUFlashModel(config, quant_config=quant_config, prefix="model")
        self.attn_start_layer = self.model.start_layer * 2
        self.attn_end_layer = self.model.end_layer * 2
        self.fake_input_ids = torch.tensor([1], dtype = torch.int32).npu()
        self.fake_positions = torch.tensor([0], dtype = torch.int32).npu()
        self.enable_over_embedding = config.use_over_embedding
        if global_server_args_dict["enable_dp_attention"] and not global_server_args_dict["npu_lmhead_tp_size"]:
            self.lm_head = ReplicatedLinear(
                config.hidden_size,
                config.vocab_size,
                bias=False,
            )
            self.logits_processor = LogitsProcessor(config, skip_all_gather=True)
        else:
            self.lm_head = ParallelLMHead(
                config.vocab_size, config.hidden_size, quant_config=quant_config
            )
            self.logits_processor = LogitsProcessor(config)
        self._routed_experts_weights_of_layer = LazyValue(
            lambda: {
                layer_id: layer.mlp.get_moe_routed_weights()
                for layer_id, layer in enumerate(self.model.layers)
                if hasattr(layer.mlp, 'get_moe_routed_weights')
            }
        )
        self._device_local_routed_experts_weights_of_layer = LazyValue(
            lambda: {
                layer_id: layer.mlp.get_moe_device_local_routed_weights()
                for layer_id, layer in enumerate(self.model.layers)
                if hasattr(layer.mlp, 'get_moe_device_local_routed_weights')
            }
        )
        self.capture_aux_hidden_states = False
        self.pp_group = get_pp_group()

    @property
    def routed_experts_weights_of_layer(self):
        return self._routed_experts_weights_of_layer.value

    @property
    def device_local_routed_experts_weights_of_layer(self):
        return self._device_local_routed_experts_weights_of_layer.value

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch
    ) -> torch.Tensor:
        if forward_batch.forward_mode.is_idle():
            input_ids = self.fake_input_ids
            positions = self.fake_positions
        hidden_states = self.model(input_ids, positions, forward_batch)

        aux_hidden_states = None
        if self.capture_aux_hidden_states:
            hidden_states, aux_hidden_states = hidden_states

        if self.pp_group.is_last_rank:
            return self.logits_processor(
                input_ids, hidden_states, self.lm_head, forward_batch, aux_hidden_states
            )
        else:
            return hidden_states

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id, begin_size)
            ("gate_up_proj", "gate_proj", 0, None),
            ("gate_up_proj", "up_proj", 1, None),
        ]
        name_mapping = {
            "compress_attn": "attn.compress_attn",
            "compress_key": "compress_kv",     # kv_lora_rank
            "compress_value": "compress_k_pe", # q_pe
            "gate_fusion.gate_weight": "attn.gate_fusion.gate_weight.weight",
        }

        def no_tp_load(param, loaded_weight, *args):
            return default_weight_loader(param, loaded_weight)

        # Params for weights, fp8 weight scales, fp8 activation scales
        # (param_name, weight_name, expert_id, shard_id)
        expert_params_mapping = NpuEPMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.n_routed_experts if hasattr(self.config, "n_routed_experts") else self.config.num_experts[0],
        )

        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            layer_id = get_layer_id(name)
            if (
                layer_id is not None
                and hasattr(self.model, "start_layer")
                and (
                layer_id < self.model.start_layer
                or layer_id >= self.model.end_layer
            )
            ):
                continue
            if get_load_number_layers() != 0:
                match = re.search(r'\d+', name)
                if match:
                    number_layer = match.group()
                    if int(number_layer) >= get_load_number_layers():
                        continue
            if "rotary_emb.inv_freq" in name:
                continue
            for source_name, target_name in name_mapping.items():
                if source_name in name:
                    name = name.replace(source_name, target_name)
            for param_name, weight_name, shard_id, begin_size in stacked_params_mapping:
                if weight_name not in name:
                    continue
                if "mlp" in name and "mlps" not in name:
                    continue
                # [NSA]: to avoid stack 'gate_proj' in compress_attn or gate_fusion
                if "compress_attn" in name or "gate_fusion" in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if (name.endswith(".bias") or name.endswith("_bias")) and name not in params_dict:
                    continue
                # Skip mtp
                if ".mtp." in name:
                    continue
                if 'mlps' in name:
                    name = name.replace('weight_scale_inv', 'weight_scale')
                param = params_dict[name]
                weight_loader = getattr(
                    param, "weight_loader", no_tp_load
                )
                if begin_size is not None and name.endswith(".weight_scale_inv"):
                    begin_size = begin_size // self.config.quantization_config["weight_block_size"][0]
                if "fused_qkv_a_proj_with_mqa" in name:
                    weight_loader(param, loaded_weight, shard_id, begin_size)
                else:
                    weight_loader(param, loaded_weight, shard_id)
                break
            else:
                if "mlp.experts." in name:
                    for mapping in expert_params_mapping:
                        param_name, weight_name, local_expert_id, shard_id = mapping
                        if weight_name not in name:
                            continue
                        name = name.replace(weight_name, param_name)
                        # Skip mtp
                        if ".mtp." in name:
                            continue
                        if (
                            name.endswith(".bias") or name.endswith("_bias")
                        ) and name not in params_dict:
                            continue
                        if 'experts' in name:
                            name = name.replace('weight_scale_inv', 'weight_scale')
                        param = params_dict[name]
                        weight_loader = param.weight_loader
                        weight_loader(
                            param,
                            loaded_weight,
                            name,
                            shard_id=shard_id,
                            expert_id=local_expert_id,
                        )
                else:
                    # Skip loading extra bias for GPTQ models.
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    # Skip loading kv_scale from ckpts towards new design.
                    # Skip mtp
                    if name.endswith(".kv_scale") and name not in params_dict:
                        continue
                    # Skip mtp
                    if ".mtp." in name:
                        continue
                    if self.enable_over_embedding:
                        if ".embed_tokens." in name:
                            name = "model.over_embedding.word_embeder.weight"
                        if ".oe_embed_tokens" in name:
                            self.model.over_embedding.load_weight(None, name, loaded_weight)
                            continue
                        if ".oe_embed_proj" in name:
                            self.model.over_embedding.load_weight(None, name, loaded_weight)
                            continue
                        if ".ngram_embeddings" in name:
                            self.model.over_embedding.load_weight(None, name, loaded_weight)
                            continue
                    if 'weight_scale_inv' in name:
                        name = name.replace('weight_scale_inv', 'weight_scale')
                    if name is None:
                        continue
                    try:
                        param = params_dict[name]
                    except Exception as e:
                        print(f"name: {name}")
                        print(f"keys: {params_dict.keys()}")
                        raise e
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)

        self.post_load_weights()


    def post_load_weights(self):
        # weight transpose for absorb
        for layer_id in range(self.model.start_layer, self.model.end_layer):
            for i in range(2):
                self_attn:Union[DeepseekV2AttentionMLA, DeepseekNSAWithMLA] \
                    = self.model.layers[layer_id].self_attn[i]
                if hasattr(self.quant_config, "weight_block_size") and self_attn.kv_b_proj.weight.dtype in (
                        torch.float8_e4m3fn,
                        torch.float8_e4m3fnuz,
                ):
                    weight_block_size = self.quant_config.weight_block_size
                    if weight_block_size is not None:
                        raise NotImplementedError("NPU不支持FP8量化")
                else:
                    w = self_attn.kv_b_proj.weight

                w_kc, w_vc = w.unflatten(
                    0, (-1, self_attn.qk_nope_head_dim + self_attn.v_head_dim)
                ).split([self_attn.qk_nope_head_dim, self_attn.v_head_dim], dim=1)
                self_attn.w_kc = w_kc.transpose(1, 2).contiguous().transpose(1, 2)
                self_attn.w_vc = w_vc.contiguous().transpose(1, 2)
                if isinstance(self_attn, DeepseekNSAWithMLA):
                    self_attn.attn.w_vc = self_attn.w_vc


    @classmethod
    def get_model_config_for_expert_location(cls, config):
        return ModelConfigForExpertLocation(
            num_layers=config.num_hidden_layers,
            num_logical_experts=config.n_routed_experts,
            num_groups=None,
        )

    def get_embed_and_head(self):
        if not self.model.enable_over_embedding:
            return self.model.embed_tokens.weight, self.lm_head.weight
        else:
            return self.model.over_embedding, self.lm_head.weight

    def set_embed_and_head(self, embed, head):
        del self.model.embed_tokens.weight
        del self.lm_head.weight
        self.model.embed_tokens.weight = embed
        self.lm_head.weight = head
        torch.npu.empty_cache()
        torch.npu.synchronize()

    def set_eagle3_layers_to_capture(self, layer_ids: Optional[List[int]] = None):
        if layer_ids is None:
            self.capture_aux_hidden_states = True
            num_layers = self.config.num_hidden_layers
            self.model.layers_to_capture = [2, num_layers // 2, num_layers - 3]
        else:
            self.capture_aux_hidden_states = True
            self.model.layers_to_capture = [val + 1 for val in layer_ids]

    def fix_kvp_current_kv_cache(self, forward_batch: ForwardBatch):
        if global_server_args_dict["npu_kvp_accuracy_fix"]:
            return
        use_kvp = (global_server_args_dict["kvp_size"] > 1)
        if use_kvp:
            kvp_group = get_attn_tp_group()
            kvp_rank = kvp_group.rank_in_group
            use_dsa = hasattr(forward_batch.token_to_kv_pool, 'get_index_k_with_scale_buffer')
            if kvp_rank == 0:
                for i in range(self.config.num_hidden_layers):
                    assert ENV.npu_enable_mla_split_kv_kr
                    layer_cache = [
                        forward_batch.token_to_kv_pool.get_kv_buffer(i * 2),
                        forward_batch.token_to_kv_pool.get_kv_buffer(i * 2 + 1)
                    ]
                    if use_dsa:
                        index_k_cache = [
                            forward_batch.token_to_kv_pool.get_index_k_with_scale_buffer(i * 2),
                            forward_batch.token_to_kv_pool.get_index_k_with_scale_buffer(i * 2 + 1)
                        ]
                    for cur_kv_cache in layer_cache:
                        k = cur_kv_cache[0]
                        k_pe = cur_kv_cache[1]
                        kv_lora_rank = k.shape[-1]
                        qk_rope_head_dim = k_pe.shape[-1]
                        k = k.view(-1, kv_lora_rank)
                        k_pe = k_pe.view(-1, qk_rope_head_dim)
                        current_k = k[forward_batch.attn_metadata.slot_mapping]
                        current_k_pe = k_pe[forward_batch.attn_metadata.slot_mapping]
                        torch_npu.npu_scatter_nd_update_(k, forward_batch.attn_metadata.kvp_current_slots_mapping.reshape(-1, 1), current_k)
                        torch_npu.npu_scatter_nd_update_(k_pe, forward_batch.attn_metadata.kvp_current_slots_mapping.reshape(-1, 1), current_k_pe)
                    if use_dsa:
                        for cur_kv_cache in index_k_cache:
                            k = cur_kv_cache
                            index_head_dim = k.shape[-1]
                            k = k.view(-1, index_head_dim)
                            current_k = k[forward_batch.attn_metadata.slot_mapping]
                            torch_npu.npu_scatter_nd_update_(k, forward_batch.attn_metadata.kvp_current_slots_mapping.reshape(-1, 1), current_k)



EntryClass = FLASHForCausalLM
