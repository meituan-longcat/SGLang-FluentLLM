"""Two-stream scheduler for dense-TP1 W4A4/W8A8 Prefill.

The current stream owns model compute, while the layer's MoE stream owns EP
AllGather and ReduceScatter. Attention keeps ownership of its collectives and
exposes its pre-core boundary so MoE ReduceScatter can overlap Attention core.
"""

from typing import TYPE_CHECKING, Any

import torch
from sglang.srt.distributed import get_mlp_tp_group
from sglang.srt.managers.eplb.expert_distribution import (
    get_global_expert_distribution_recorder,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.utils import get_colorful_logger

if TYPE_CHECKING:
    from sglang.srt.models.npu_longcat_flash import (
        ParallelShortcutTransformerLayer,
    )


logger = get_colorful_logger(__name__)


def _submit_hidden_precomm(
    layer: "ParallelShortcutTransformerLayer",
    comm_stream: torch.npu.Stream,
    moe_pre_comm_input: torch.Tensor | list[torch.Tensor],
    global_sp_num_tokens,
):
    """Submit MoE hidden AllGather before Router/TopK."""
    with torch.npu.stream(comm_stream):
        comm_stream.wait_event(layer._w4a4_two_stream_fq_ready)
        return layer.mlp.experts.dispatcher_ag_rs.pre_comm(
            moe_pre_comm_input,
            sp_num_tokens=global_sp_num_tokens,
        )


def _submit_metadata_precomm(
    layer: "ParallelShortcutTransformerLayer",
    comm_stream: torch.npu.Stream,
    top_experts: torch.Tensor,
    expert_weights: torch.Tensor,
    global_sp_num_tokens,
):
    """Submit routing metadata after hidden on the same EP stream."""
    with torch.npu.stream(comm_stream):
        comm_stream.wait_event(layer._w4a4_two_stream_topk_ready)
        outputs = layer.mlp.experts.dispatcher_ag_rs.pre_comm(
            [top_experts, expert_weights],
            sp_num_tokens=global_sp_num_tokens,
        )
        layer._w4a4_two_stream_precomm_ready.record(stream=comm_stream)
        return outputs


def _run_local_moe(
    layer: "ParallelShortcutTransformerLayer",
    moe_hidden_states: torch.Tensor,
    moe_global_hidden_states,
    moe_global_top_experts: torch.Tensor,
    moe_global_expert_weights: torch.Tensor,
    top_experts: torch.Tensor,
    expert_weights: torch.Tensor,
):
    """Run Dispatch, expert compute, Combine and zero expert locally."""
    experts = layer.mlp.experts
    (
        expand_x,
        tokens_per_expert,
        dispatch_output,
    ) = experts.dispatcher_ag_rs.dispatch(
        moe_global_hidden_states,
        moe_global_top_experts,
        skip_comm=True,
    )
    get_global_expert_distribution_recorder().on_local_expert_counts(tokens_per_expert)
    if getattr(experts, "is_dynamic_w4a4", False):
        moe_output = experts.quant_method.apply(
            experts,
            expand_x,
            tokens_per_expert,
            gmm2_tuning_config=[128],
        )
    else:
        moe_output = experts.quant_method.apply(
            experts,
            expand_x,
            tokens_per_expert,
        )
    moe_output = experts.dispatcher_ag_rs.combine(
        moe_output,
        moe_global_top_experts,
        moe_global_expert_weights,
        dispatch_output,
        reduce_type="skip",
    )

    moe_output_zero = None
    if experts.has_identity_zero_expert:
        moe_output_zero = experts.zero_expert(
            moe_hidden_states,
            top_experts,
            expert_weights,
        )
    return moe_output, moe_output_zero


def _submit_moe_reduce_scatter(
    layer: "ParallelShortcutTransformerLayer",
    comm_stream: torch.npu.Stream,
    moe_output: torch.Tensor,
    global_sp_num_tokens,
):
    """Submit MoE ReduceScatter after Attention-1 preparation."""
    with torch.npu.stream(comm_stream):
        # This event is recorded on the compute stream after local MoE and all
        # Attention-1 preparation, so it covers both producer dependencies.
        comm_stream.wait_event(layer._w4a4_two_stream_attn_pre_core_ready)
        moe_output = layer.mlp.experts.dispatcher_ag_rs.post_comm(
            moe_output,
            "reduce_scatter",
            sp_num_tokens=global_sp_num_tokens,
        )
        layer._w4a4_two_stream_moe_rs_ready.record(stream=comm_stream)
    return moe_output


def forward_w4a4_prefill_two_stream(
    layer: "ParallelShortcutTransformerLayer",
    rotary_emb: Any,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    kv_cache: Any,
    forward_batch: ForwardBatch,
) -> torch.Tensor:
    """Run W4A4/W8A8 Prefill with serialized compute and dedicated EP comm."""
    experts = layer.mlp.experts
    is_dynamic_w4a4 = getattr(experts, "is_dynamic_w4a4", False)
    path_name = "W4A4" if is_dynamic_w4a4 else "W8A8"
    global_sp_num_tokens = forward_batch.global_sp_num_tokens
    mlp_sp_token_num = get_mlp_tp_group().get_local_sp_token_num(global_sp_num_tokens)

    compute_stream = torch.npu.current_stream()
    comm_stream = layer.moe_stream
    if comm_stream is None:
        raise RuntimeError(f"{path_name} Prefill two-stream requires a communication stream.")

    logger.info_once(f"{path_name} Prefill two-stream path is active: serialized compute and dedicated EP communication.")

    int8_quant = experts.w13_weight.dtype == torch.int8
    i = 0
    residual = hidden_states
    cli_prefix_producer_ready = None
    cli_prefix_state = None
    cli_prefix_ready = None
    if not forward_batch.forward_mode.is_idle():
        hidden_states = layer.input_layernorm[i](hidden_states)
        selected_next_pool = None
        if layer.self_attn[0].cli_factor == 2:
            next_cache = kv_cache[1]
            next_pool = next_cache[0] if isinstance(next_cache, (tuple, list)) else next_cache
            if next_pool.dtype == torch.int8:
                selected_next_pool = (next_pool.reshape(-1, next_pool.shape[-1]), layer.self_attn[1].attn_layer_id)
        hidden_states = layer.self_attn[i](
            rotary_emb,
            positions,
            hidden_states,
            kv_cache[i],
            forward_batch,
            gather_qkv=True,
            reduce_type="reduce_scatter",
            atten_id=0,
            prefill_selected_next_pool=selected_next_pool,
        )
        if layer.self_attn[1].can_prefetch_native_sfad_prefix(kv_cache[1], forward_batch, atten_id=1):
            # Record the earliest safe producer boundary. The prefetch is
            # host-submitted after Dense-0 has been queued, but may start as
            # soon as Attention-0 and its communicator traffic complete.
            cli_prefix_producer_ready = torch.npu.Event()
            cli_prefix_producer_ready.record(stream=compute_stream)
        hidden_states, residual = layer.post_attention_layernorm[i](
            hidden_states,
            residual,
        )

    moe_hidden_states = hidden_states

    if is_dynamic_w4a4:
        moe_pre_comm_input = list(
            experts.quant_method.quantize_moe_input(
                experts,
                moe_hidden_states,
            )
        )
    else:
        moe_pre_comm_input = moe_hidden_states
    layer._w4a4_two_stream_fq_ready.record(stream=compute_stream)
    moe_global_hidden_states = _submit_hidden_precomm(
        layer,
        comm_stream,
        moe_pre_comm_input,
        global_sp_num_tokens,
    )

    router_logits = layer.mlp.router(moe_hidden_states)
    top_experts, expert_weights = layer.mlp.topk.process(router_logits)
    layer._w4a4_two_stream_topk_ready.record(stream=compute_stream)

    # Queue Dense-0 before the metadata helper. The helper has measurable host
    # tiling/enqueue cost, so calling it first leaves an avoidable TopK->QMM
    # bubble even though the collective itself runs on the communication
    # stream. Dense-0 is communication-free under the strict dense-TP1 gate.
    if int8_quant:
        hidden_states = layer.mlps[i].pre_quant(hidden_states)
    hidden_states = layer.mlps[i].pre_comm(
        hidden_states,
        output_split_sizes=mlp_sp_token_num,
    )
    hidden_states = layer.mlps[i](
        hidden_states,
        reduce_type="reduce_scatter",
        input_split_sizes=mlp_sp_token_num,
    )

    (
        moe_global_top_experts,
        moe_global_expert_weights,
    ) = _submit_metadata_precomm(
        layer,
        comm_stream,
        top_experts,
        expert_weights,
        global_sp_num_tokens,
    )

    # Attention-1 reuses Attention-0 TopK when CLI is enabled. Its cached
    # prefix is independent of Dense/MoE, so dequantize and gather it now;
    # current extend KV is still produced by Attention-1 in BF16.
    if cli_prefix_producer_ready is not None:
        cli_prefix_state, cli_prefix_ready = layer.self_attn[1].prefetch_native_sfad_prefix(
            kv_cache[1],
            forward_batch,
            cli_prefix_producer_ready,
        )

    compute_stream.wait_event(layer._w4a4_two_stream_precomm_ready)
    moe_output, moe_output_zero = _run_local_moe(
        layer,
        moe_hidden_states,
        moe_global_hidden_states,
        moe_global_top_experts,
        moe_global_expert_weights,
        top_experts,
        expert_weights,
    )

    # Attention invokes this callback after its preparation collectives and
    # immediately before its core. MoE RS can then overlap Attention core and
    # the following Dense-1 compute.
    def launch_moe_rs_before_attention_core():
        nonlocal moe_output
        moe_output = _submit_moe_reduce_scatter(
            layer,
            comm_stream,
            moe_output,
            global_sp_num_tokens,
        )

    i = 1
    if not forward_batch.forward_mode.is_idle():
        hidden_states, residual = layer.input_layernorm[i](
            hidden_states,
            residual,
        )
        hidden_states = layer.self_attn[i](
            rotary_emb,
            positions,
            hidden_states,
            kv_cache[i],
            forward_batch,
            gather_qkv=True,
            reduce_type="reduce_scatter",
            atten_id=1,
            prefill_core_ready_event=(layer._w4a4_two_stream_attn_pre_core_ready),
            prefill_before_core_callback=launch_moe_rs_before_attention_core,
            prefill_sfad_prefix_state=cli_prefix_state,
            prefill_sfad_prefix_ready_event=cli_prefix_ready,
        )
        hidden_states, residual = layer.post_attention_layernorm[i](
            hidden_states,
            residual,
        )
    else:
        # Idle ranks still follow the same EP collective order. There is no
        # Attention callback, so establish the equivalent boundary explicitly.
        layer._w4a4_two_stream_attn_pre_core_ready.record(stream=compute_stream)
        launch_moe_rs_before_attention_core()

    if int8_quant:
        hidden_states = layer.mlps[i].pre_quant(hidden_states)
    hidden_states = layer.mlps[i].pre_comm(
        hidden_states,
        output_split_sizes=mlp_sp_token_num,
    )
    hidden_states = layer.mlps[i](
        hidden_states,
        reduce_type="reduce_scatter",
        input_split_sizes=mlp_sp_token_num,
    )

    compute_stream.wait_event(layer._w4a4_two_stream_moe_rs_ready)
    if moe_output_zero is not None:
        moe_output = moe_output + moe_output_zero

    hidden_states = moe_output + hidden_states
    hidden_states = residual + hidden_states
    return hidden_states
