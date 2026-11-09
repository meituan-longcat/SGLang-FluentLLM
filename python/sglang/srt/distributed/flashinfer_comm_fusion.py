import logging
from typing import Optional

import flashinfer.comm as comm
import torch

from sglang.srt.distributed import GroupCoordinator
from sglang.srt.env import global_server_args_dict
from sglang.srt.layers.dense.gemms.fp8.fp8_kernel import create_per_token_group_quant_fp8_output_scale

logger = logging.getLogger(__name__)

_workspace_manager = None


class FlashInferWorkspaceManager:
    def __init__(self):
        self.workspace_tensor = None
        self.ipc_handles = None
        self.world_size = None
        self.rank = None
        self.max_token_num = None
        self.max_token_num_for_vocab_gather = None
        self.hidden_dim = None
        self.use_fp32_lamport = None
        self.initialized = False
        self.vocab_gather_space_initialized = False
        self.local_vocab_size = None

    def init_workspace_for_vocab_gather(
        self,
        world_size: int,
        rank: int,
        local_vocab_size: int,
        group: GroupCoordinator,
        max_token_num_for_vocab_gather: int = 16,
    ):
        if (
            self.vocab_gather_space_initialized
            and self.world_size == world_size
            and self.max_token_num_for_vocab_gather == max_token_num_for_vocab_gather
            and self.local_vocab_size == local_vocab_size
        ):
            return
        self.cleanup_vocab_gather_buf()

        self.ipc_handles_for_gather_vocab, self.workspace_tensor_for_gather_vocab = comm.all_gather.create_ipc_workspace_for_allgather(
            rank, world_size, max_token_num_for_vocab_gather, local_vocab_size * world_size, False, group=group
        )
        self.vocab_gather_space_initialized = True
        self.world_size = world_size
        self.max_token_num_for_vocab_gather = max_token_num_for_vocab_gather
        self.local_vocab_size = local_vocab_size

    def initialize(
        self,
        world_size: int,
        rank: int,
        max_token_num: int,
        hidden_dim: int,
        group: GroupCoordinator,
        use_fp32_lamport: bool = False,
    ):
        """Initialize workspace"""
        if (
            self.initialized
            and self.world_size == world_size
            and self.max_token_num == max_token_num
            and self.hidden_dim == hidden_dim
            and self.use_fp32_lamport == use_fp32_lamport
        ):
            return

        self.cleanup()
        # allreduce_fusion, allgather_fusion, reducescatter_fusion all use the same workspace to create entry
        self.ipc_handles, self.workspace_tensor = comm.trtllm_create_ipc_workspace_for_all_reduce_fusion(
            rank,
            world_size,
            max_token_num,
            hidden_dim,
            group=group,
            use_fp32_lamport=use_fp32_lamport,
        )

        self.world_size = world_size
        self.rank = rank
        self.max_token_num = max_token_num
        self.hidden_dim = hidden_dim
        self.use_fp32_lamport = use_fp32_lamport
        self.initialized = True
        self.group = group

        logger.info(
            f"FlashInfer workspace initialized for rank {rank}, world_size {world_size}, max_token_num {max_token_num}, hidden_dim {hidden_dim} "
        )

    def cleanup(self):
        """Clean up workspace"""
        if self.initialized and self.ipc_handles is not None:
            try:
                comm.trtllm_destroy_ipc_workspace_for_all_reduce_fusion(self.ipc_handles, group=self.group.device_group)
            except Exception as e:
                logger.warning(f"Failed to cleanup FlashInfer workspace: {e}")
            finally:
                self.workspace_tensor = None
                self.ipc_handles = None
                self.initialized = False
                self.world_size = None
                self.rank = None
                self.max_token_num = None
                self.hidden_dim = None
                self.use_fp32_lamport = None

    def cleanup_vocab_gather_buf(self):
        if self.vocab_gather_space_initialized and self.ipc_handles_for_gather_vocab is not None:
            try:
                comm.destroy_ipc_workspace_for_allgather(self.ipc_handles_for_gather_vocab, group=self.group.device_group)
            except Exception as e:
                logger.warning(f"Failed to cleanup FlashInfer workspace: {e}")
            finally:
                self.vocab_gather_space_initialized = False
                self.world_size = None
                self.max_token_num_for_vocab_gather = None
                self.local_vocab_size = None


_workspace_manager = FlashInferWorkspaceManager()


#
# NOTE:
# Reduce-scatter now reuses `_workspace_manager` (allreduce-style IPC workspace).
# This avoids keeping a second, similarly-sized workspace alive.


def ensure_workspace_initialized(
    group: GroupCoordinator,
    max_token_num: int = 2048,
    hidden_dim: int = 4096,
    use_fp32_lamport: bool = False,
):
    world_size = group.world_size
    if world_size <= 1:
        return False

    rank = group.rank_in_group

    target_max_token_num = max_token_num
    target_hidden_dim = hidden_dim
    target_use_fp32_lamport = use_fp32_lamport
    if _workspace_manager.initialized and _workspace_manager.world_size == world_size:
        if _workspace_manager.max_token_num is not None:
            target_max_token_num = max(_workspace_manager.max_token_num, max_token_num)
        if _workspace_manager.hidden_dim is not None:
            target_hidden_dim = max(_workspace_manager.hidden_dim, hidden_dim)
        if _workspace_manager.use_fp32_lamport:
            target_use_fp32_lamport = True

    if (
        (not _workspace_manager.initialized)
        or (_workspace_manager.world_size != world_size)
        or (_workspace_manager.max_token_num != target_max_token_num)
        or (_workspace_manager.hidden_dim != target_hidden_dim)
        or (_workspace_manager.use_fp32_lamport != target_use_fp32_lamport)
    ):
        logger.info(
            "Re/initializing FlashInfer IPC workspace: "
            "world_size=%s rank=%s max_token_num=%s hidden_dim=%s use_fp32_lamport=%s "
            "(prev max_token_num=%s hidden_dim=%s use_fp32_lamport=%s)",
            world_size,
            rank,
            target_max_token_num,
            target_hidden_dim,
            target_use_fp32_lamport,
            _workspace_manager.max_token_num,
            _workspace_manager.hidden_dim,
            _workspace_manager.use_fp32_lamport,
        )
        _workspace_manager.initialize(
            world_size=world_size,
            rank=rank,
            max_token_num=target_max_token_num,
            hidden_dim=target_hidden_dim,
            use_fp32_lamport=target_use_fp32_lamport,
            group=group.device_group,
        )

    return _workspace_manager.initialized


def ensure_vocab_allgather_initialized(
    group: GroupCoordinator,
    max_token_num_for_vocab_gather: int = 16,
    local_vocab_size: int = 4096,
):
    world_size = group.world_size
    if world_size <= 1:
        return False
    rank = group.rank_in_group
    target_max_token_num = max_token_num_for_vocab_gather
    target_vocab_size = local_vocab_size
    if _workspace_manager.initialized and _workspace_manager.world_size == world_size:
        if _workspace_manager.max_token_num_for_vocab_gather is not None:
            target_max_token_num = max(_workspace_manager.max_token_num_for_vocab_gather, max_token_num_for_vocab_gather)
        if _workspace_manager.local_vocab_size is not None:
            target_vocab_size = max(_workspace_manager.local_vocab_size, local_vocab_size)

    if (
        (not _workspace_manager.vocab_gather_space_initialized)
        or (_workspace_manager.world_size != world_size)
        or (_workspace_manager.max_token_num != target_max_token_num)
        or (_workspace_manager.local_vocab_size != target_vocab_size)
    ):
        _workspace_manager.init_workspace_for_vocab_gather(
            world_size=world_size,
            rank=rank,
            group=group.device_group,
            max_token_num_for_vocab_gather=target_max_token_num,
            local_vocab_size=target_vocab_size,
        )

    return _workspace_manager.vocab_gather_space_initialized


def get_num_tokens_per_rank(world_size: int, total_tokens_in_group: int) -> list:
    token_list_in_group = []
    for rank in range(0, world_size):
        num_tokens_per_rank = total_tokens_in_group // world_size + (1 if (rank < total_tokens_in_group % world_size) else 0)
        token_list_in_group.append(num_tokens_per_rank)
    return token_list_in_group


def flashinfer_allgather_vocab(
    input_tensor: torch.Tensor,
    group: GroupCoordinator,
    max_tokens_num: int,
    local_vocab_size: int,
    max_sm_to_use: Optional[int] = None,
    launch_with_pdl: Optional[bool] = True,
    trigger_completion_at_end: Optional[bool] = False,
):
    assert input_tensor.dtype == torch.bfloat16, f"Only support bf16 for now, but get {input_tensor.dtype=} {input_tensor.shape=}"

    world_size = group.world_size
    assert world_size > 1, "Single GPU, no need for allreduce fusion"
    assert input_tensor.shape[0] <= max_tokens_num

    if not ensure_vocab_allgather_initialized(
        group=group,
        max_token_num_for_vocab_gather=max_tokens_num,
        local_vocab_size=local_vocab_size,
    ):
        raise RuntimeError("FlashInfer workspace not available")

    token_num, _ = input_tensor.shape
    vocab_size = local_vocab_size * world_size
    allgather_out = torch.empty((token_num, vocab_size), dtype=input_tensor.dtype, device=input_tensor.device)
    comm.all_gather.simple_all_gather(
        allgather_in=input_tensor,
        world_size=world_size,
        world_rank=group.rank_in_group,
        token_num=token_num,
        hidden_size=local_vocab_size,
        workspace_ptrs=_workspace_manager.workspace_tensor_for_gather_vocab,
        launch_with_pdl=launch_with_pdl,
        trigger_completion_at_end=trigger_completion_at_end,
        max_num_tokens=max_tokens_num,
        allgather_out=allgather_out,
        max_sm_to_use=max_sm_to_use,
    )
    return allgather_out


def flashinfer_allreduce_residual_rmsnorm(
    input_tensor: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    group: GroupCoordinator,
    eps: float = 1e-6,
    max_token_num: int = 2048,
    use_oneshot: Optional[bool] = None,
    trigger_completion_at_end: bool = False,
    fp32_acc: bool = False,
    block_quant_fp8: bool = False,
    has_partial_norm_out: bool = False,
    max_sm_to_use: Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Use FlashInfer's fused allreduce + residual + RMS norm operation

    Args:
        input_tensor: Input tensor that needs allreduce
        residual: Residual tensor
        weight: RMS norm weight
        eps: RMS norm epsilon
        max_token_num: Maximum token number
        use_oneshot: Whether to use oneshot mode
        trigger_completion_at_end: Whether to trigger completion at end
        fp32_acc: Whether to use fp32 precision

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: (norm_output, residual_output)
    """
    world_size = group.world_size
    assert world_size > 1, "Single GPU, no need for allreduce fusion"
    assert input_tensor.shape[0] <= max_token_num

    if not ensure_workspace_initialized(
        group=group,
        max_token_num=max_token_num,
        hidden_dim=input_tensor.shape[-1],
        use_fp32_lamport=(input_tensor.dtype == torch.float32),
    ):
        raise RuntimeError("FlashInfer workspace not available")

    token_num, hidden_dim = input_tensor.shape

    residual_out = torch.empty_like(residual)
    norm_out = torch.empty_like(input_tensor)

    partial_norm_out = None
    pattern_code = None
    if has_partial_norm_out:
        num_tokens_list = get_num_tokens_per_rank(world_size, input_tensor.shape[0])
        partial_num_tokens = num_tokens_list[group.device_group.rank()]
        partial_norm_out = torch.empty((partial_num_tokens, hidden_dim), dtype=input_tensor.dtype, device=input_tensor.device)
        pattern_code = (
            comm.AllReduceFusionPattern.kARResidualRMSNormPartialOutFP8BlockWiseQuant
            if block_quant_fp8
            else comm.AllReduceFusionPattern.kARResidualRMSNormPartialOut
        )
    else:
        pattern_code = (
            comm.AllReduceFusionPattern.kARResidualRMSNormFP8BlockWiseQuant if block_quant_fp8 else comm.AllReduceFusionPattern.kARResidualRMSNorm
        )

    if block_quant_fp8:
        quant_out = torch.empty(input_tensor.size(), dtype=torch.float8_e4m3fn, device=input_tensor.device)
        out_shape = (*quant_out.shape[:-1], quant_out.shape[-1])
        scale_out = create_per_token_group_quant_fp8_output_scale(
            x_shape=out_shape,
            device=quant_out.device,
            group_size=128,
            column_major_scales=True,
            scale_tma_aligned=True,
            scale_ue8m0=False,
        )
    else:
        quant_out = None
        scale_out = None

    comm.trtllm_allreduce_fusion(
        allreduce_in=input_tensor,
        world_size=world_size,
        world_rank=group.rank_in_group,
        token_num=token_num,
        hidden_dim=hidden_dim,
        workspace_ptrs=_workspace_manager.workspace_tensor,
        launch_with_pdl=not global_server_args_dict["disable_pdl"],
        use_oneshot=True,  # Residual is expected to be local-token (post RS) layout.
        trigger_completion_at_end=trigger_completion_at_end,
        fp32_acc=fp32_acc,
        pattern_code=(pattern_code),
        allreduce_out=None,
        residual_in=residual,
        residual_out=residual_out,
        norm_out=norm_out,
        quant_out=quant_out,
        scale_out=scale_out,
        rms_gamma=weight,
        rms_eps=eps,
        scale_factor=None,
        layout_code=None,
        max_sm_to_use=max_sm_to_use,
        partial_norm_out=partial_norm_out,
    )
    if block_quant_fp8:
        return quant_out, residual_out, scale_out, partial_norm_out
    else:
        return norm_out, residual_out, None, partial_norm_out


def flashinfer_reducescatter_residual_rmsnorm(
    input_tensor: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    group: GroupCoordinator,
    eps: float = 1e-6,
    max_token_num: int = 2048,
    use_oneshot: Optional[bool] = None,
    trigger_completion_at_end: bool = False,
    fp32_acc: bool = False,
    block_quant_fp8: bool = False,
    add_in: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    Use FlashInfer's fused reducescatter + residual + RMS norm operation

    Args:
        input_tensor: Input tensor that needs reducescatter
        residual: Residual tensor
        weight: RMS norm weight
        eps: RMS norm epsilon
        max_token_num: Maximum token number
        use_oneshot: Whether to use oneshot mode
        trigger_completion_at_end: Whether to trigger completion at end
        fp32_acc: Whether to use fp32 precision

    Returns:
        Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]: (norm_output, residual_output, scale_output)
    """
    world_size = group.world_size
    assert world_size > 1, "Single GPU, no need for reducescatter fusion"
    assert input_tensor.shape[0] <= max_token_num

    if not ensure_workspace_initialized(
        group=group,
        max_token_num=max_token_num,
        hidden_dim=input_tensor.shape[-1],
        use_fp32_lamport=(input_tensor.dtype == torch.float32),
    ):
        raise RuntimeError("FlashInfer reduce scatter workspace not available")

    token_num, hidden_dim = input_tensor.shape
    rank = group.rank_in_group

    tokens_per_rank = token_num // world_size
    remaining = token_num % world_size
    token_count = tokens_per_rank + (1 if rank < remaining else 0)

    residual_out = torch.empty((token_count, hidden_dim), dtype=residual.dtype, device=residual.device)
    norm_out = torch.empty((token_count, hidden_dim), dtype=input_tensor.dtype, device=input_tensor.device)
    if block_quant_fp8:
        if add_in is not None:
            pattern_code = comm.ReduceScatterFusionPattern.kRSAddResidualRMSNormFP8BlockWiseQuant
        else:
            pattern_code = comm.ReduceScatterFusionPattern.kRSResidualRMSNormFP8BlockWiseQuant
    else:
        if add_in is not None:
            pattern_code = comm.ReduceScatterFusionPattern.kRSAddResidualRMSNorm
        else:
            pattern_code = comm.ReduceScatterFusionPattern.kRSResidualRMSNorm

    if block_quant_fp8:
        quant_out = torch.empty((token_count, hidden_dim), dtype=torch.float8_e4m3fn, device=input_tensor.device)
        out_shape = (*quant_out.shape[:-1], quant_out.shape[-1])
        scale_out = create_per_token_group_quant_fp8_output_scale(
            x_shape=out_shape,
            device=quant_out.device,
            group_size=128,
            column_major_scales=True,
            scale_tma_aligned=True,
            scale_ue8m0=False,
        )
    else:
        quant_out = None
        scale_out = None
    comm.trtllm_reducescatter_fusion(
        reducescatter_in=input_tensor,
        world_size=world_size,
        world_rank=group.rank_in_group,
        token_num=token_num,
        hidden_dim=hidden_dim,
        workspace_ptrs=_workspace_manager.workspace_tensor,
        launch_with_pdl=not global_server_args_dict["disable_pdl"],
        trigger_completion_at_end=trigger_completion_at_end,
        num_token_current_rank=token_count,
        fp32_acc=fp32_acc,
        pattern_code=pattern_code,
        use_oneshot=use_oneshot,
        reducescatter_out=None,
        add_in=add_in,
        residual_in=residual,
        residual_out=residual_out,
        norm_out=norm_out,
        quant_out=quant_out,
        scale_out=scale_out,
        rms_gamma=weight,
        rms_eps=eps,
        scale_factor=None,
        layout_code=None,
    )
    if block_quant_fp8:
        return quant_out, residual_out, scale_out
    else:
        return norm_out, residual_out, None


def flashinfer_allgather_dual_rmsnorm(
    qkv: torch.Tensor,
    total_num_tokens: int,
    weight_q_a: torch.nn.Parameter,
    weight_kv_a: torch.nn.Parameter,
    group: GroupCoordinator,
    eps_q: float,
    eps_kv: float,
    max_token_num: int,
    block_quant_fp8: bool = False,
    trigger_completion_at_end: bool = False,
    fp32_acc: bool = False,
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Use FlashInfer's fused allgather + dual RMS norm + optional FP8 quantization operation

    Args:
        qkv: Input tensor to allgather, shape [num_token_current_rank, q_lora_rank + kv_lora_rank + qk_rope_head_dim]
        weight_q_a: RMS norm weight for Q
        weight_kv_a: RMS norm weight for KV
        eps_q: RMS norm epsilon for Q
        eps_kv: RMS norm epsilon for KV
        max_token_num: Maximum token number
        block_quant_fp8: Whether to perform FP8 block-wise quantization on the first norm output
        trigger_completion_at_end: Whether to trigger completion at end
        fp32_acc: Whether to use fp32 precision

    Returns:
        Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
            (allgather_out, quant_out or x_norm_out, y_norm_out, block_scale)
    """
    world_size = group.world_size
    assert world_size > 1, "Single GPU, no need for allgather fusion"

    num_token_current_rank = qkv.shape[0]
    hidden_dim = qkv.shape[1]

    if num_token_current_rank > max_token_num:
        raise RuntimeError(f"Token count {num_token_current_rank} exceeds max {max_token_num}")

    if not ensure_workspace_initialized(
        group=group,
        max_token_num=max_token_num,
        hidden_dim=hidden_dim,
        use_fp32_lamport=(qkv.dtype == torch.float32),
    ):
        raise RuntimeError("FlashInfer workspace not available")

    world_rank = group.rank_in_group

    q_lora_rank = weight_q_a.shape[0]
    kv_lora_rank = weight_kv_a.shape[0]
    qk_rope_head_dim = hidden_dim - q_lora_rank - kv_lora_rank

    num_token_all_group = total_num_tokens

    allgather_out = torch.empty((num_token_all_group, hidden_dim), dtype=qkv.dtype, device=qkv.device)

    x_norm_out = torch.empty((num_token_all_group, q_lora_rank), dtype=qkv.dtype, device=qkv.device)

    # y_norm_out output is on the slice of allgather_out
    y_norm_out = allgather_out[..., q_lora_rank : q_lora_rank + kv_lora_rank]

    if block_quant_fp8:
        block_size = 128
        quant_out = torch.empty((num_token_all_group, q_lora_rank), dtype=torch.float8_e4m3fn, device=qkv.device)
        out_shape = (*quant_out.shape[:-1], quant_out.shape[-1])
        scale_out = create_per_token_group_quant_fp8_output_scale(
            x_shape=out_shape,
            device=quant_out.device,
            group_size=block_size,
            column_major_scales=True,
            scale_tma_aligned=True,
            scale_ue8m0=False,
        )
    else:
        quant_out = None
        scale_out = None

    pattern_code = (
        comm.AllGatherFusionPattern.kAllGatherfusedRMSFP8BlockWiseQuant if block_quant_fp8 else comm.AllGatherFusionPattern.kAllGatherfusedRMS
    )

    comm.trtllm_allgather_fusion(
        allgather_in=qkv,
        world_size=world_size,
        world_rank=world_rank,
        hidden_dim=hidden_dim,
        workspace_ptrs=_workspace_manager.workspace_tensor,
        launch_with_pdl=not global_server_args_dict["disable_pdl"],
        trigger_completion_at_end=trigger_completion_at_end,
        num_token_current_rank=num_token_current_rank,
        allgather_out=allgather_out,
        num_token_all_group=num_token_all_group,
        pattern_code=pattern_code,
        use_oneshot=True,
        fp32_acc=fp32_acc,
        x_norm_out=x_norm_out,
        y_norm_out=y_norm_out,
        quant_out=quant_out,
        scale_out=scale_out,
        x_rms_gamma=weight_q_a,
        y_rms_gamma=weight_kv_a,
        x_rms_eps=eps_q,
        y_rms_eps=eps_kv,
        q_lora_rank=q_lora_rank,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
    )

    return allgather_out, quant_out if block_quant_fp8 else x_norm_out, y_norm_out, scale_out


def cleanup_flashinfer_workspace():
    global _workspace_manager
    if _workspace_manager is not None:
        _workspace_manager.cleanup()
