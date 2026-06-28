import torch
import torch_npu

from typing import List
from sglang.srt.utils import get_910b_num_gpus_per_node

from sglang.srt.distributed import get_all2all_ep_group, get_ep_group
from sglang.srt.env import ENV, global_server_args_dict


class TokensDispatcherAgRs:
    def __init__(
            self,
            layer,
            num_experts: int,
            zero_expert_num: int,
            ep_size: int,
            ep_rank: int,
            quant: bool=False) -> None:
        """
        Initialize the zero token dropping router.
        """
        self.quant_mode = 1 if quant else -1
        self.layer = layer
        self.num_experts = num_experts
        self.expert_idx_min = ep_rank * (num_experts // ep_size)
        self.expert_idx_max = self.expert_idx_min + (num_experts // ep_size)
        self.ep_group = get_ep_group()

    def pre_comm(self, hidden_states, sp_num_tokens=None):
        if not isinstance(hidden_states, (list, tuple)):
            return get_ep_group().all_gather(hidden_states, dim=0, output_split_sizes=sp_num_tokens)
        result = []
        for tensor in hidden_states:
            result.append(get_ep_group().all_gather(tensor, dim=0, output_split_sizes=sp_num_tokens))
        return result

    def post_comm(self, hidden_states, reduce_type='all_reduce', sp_num_tokens=None):
        if reduce_type == 'all_reduce':
            return self.ep_group.all_reduce(hidden_states)
        elif reduce_type == 'reduce_scatter':
            return self.ep_group.reduce_scatter(hidden_states, sp_num_tokens)
        else:
            raise ValueError(f'{reduce_type=} not in support type [all_reduce, reduce_scatter]')

    def dispatch(self, hidden_states, top_experts, skip_comm=False, sp_num_tokens=None):
        dynamic_quant = None
        if isinstance(hidden_states, (list, tuple)):
            hidden_states, dynamic_quant = hidden_states
        if not skip_comm:
            hidden_states = self.pre_comm(hidden_states, sp_num_tokens)
            if dynamic_quant is not None:
                dynamic_quant = self.pre_comm(dynamic_quant, sp_num_tokens)
        # if dynamic_quant is not None, quant has be done outside, set quant_mode -1 for no need quant here
        quant_mode = self.quant_mode if dynamic_quant is None else -1
        # if need quant here, set scale to smooth scale, else:
        # 1. set scale to None for no need quant, 2. or set scale to dynamic_quant for routing scale
        scale = self.layer.w13_smooth_scale if quant_mode == 1 else dynamic_quant
        expand_x, expanded_row_idx, expert_token_count, pertoken_scale = torch_npu.npu_moe_init_routing_v2(
            hidden_states,
            expert_idx=top_experts,
            scale=scale,
            active_num=top_experts.numel(),
            expert_num=self.num_experts,
            quant_mode=quant_mode,
            active_expert_range=[self.expert_idx_min, self.expert_idx_max],
            expert_tokens_num_type=1,
            expert_tokens_num_flag=True,
            row_idx_type=0,
            drop_pad_mode=0
        )
        if scale is not None or quant_mode != -1:
            expand_x = (expand_x, pertoken_scale)
        return expand_x, expert_token_count, expanded_row_idx

    def combine(self, expanded_permuted_rows, topk_ids, topk_weight, expanded_row_idx, reduce_type='reduce_scatter', sp_token_num=None):
        output_combine = torch_npu.npu_moe_finalize_routing(
            expanded_permuted_rows.unsqueeze(1),
            scales=topk_weight.to(expanded_permuted_rows.dtype),
            expanded_src_to_dst_row=expanded_row_idx,
            export_for_source_row=topk_ids,
            drop_pad_mode=3,
            skip1=None,
            skip2=None,
            bias=None
        )
        if reduce_type == 'skip':
            return output_combine
        return self.post_comm(output_combine, reduce_type, sp_token_num)


class TokensDispatcherPA2A:
    def __init__(
            self,
            layer,
            num_experts: int,
            zero_expert_num: int,
            ep_size: int,
            ep_rank: int,
            quant: bool=False) -> None:
        """
        Initialize the zero token dropping router.
        """
        self.quant_mode = 1 if quant else -1
        self.layer = layer
        self.num_experts = num_experts
        self.expert_idx_min = ep_rank * (num_experts // ep_size)
        self.expert_idx_max = self.expert_idx_min + (num_experts // ep_size)
        self.ep_group = get_ep_group()
        self.zero_expert_num = zero_expert_num

    def dispatch(self, hidden_states, top_experts):
        dynamic_quant = None
        if isinstance(hidden_states, (list, tuple)):
            hidden_states, dynamic_quant = hidden_states
        # if dynamic_quant is not None, quant has be done outside, set quant_mode -1 for no need quant here
        quant_mode = self.quant_mode if dynamic_quant is None else -1
        # if need quant here, set scale to smooth scale, else:
        # 1. set scale to None for no need quant, 2. or set scale to dynamic_quant for routing scale
        scale = self.layer.w13_smooth_scale_total if quant_mode == 1 else dynamic_quant
        has_quant = ((quant_mode != -1) or (dynamic_quant is not None))
        expanded_x, expanded_row_idx, tokens_per_expert, pertoken_scale = torch_npu.npu_moe_init_routing_v2(
            hidden_states,
            expert_idx=top_experts.to(torch.int32),
            scale=scale,
            active_num=top_experts.numel(),
            expert_num=self.num_experts,
            expert_tokens_num_type=1,
            expert_tokens_num_flag=True,
            quant_mode=quant_mode,
            active_expert_range=[0, self.num_experts],
            row_idx_type=0,
            drop_pad_mode=0
        )
        if top_experts.numel() == 0:
            tokens_per_expert.zero_()
        tokens_per_expert_group = tokens_per_expert.new_empty(self.num_experts)
        # (total_experts,)->(total_ranks*n_routed_experts_per_rank)
        torch.distributed.all_to_all_single(tokens_per_expert_group, tokens_per_expert, group=self.ep_group.device_group)
        # combine tensors, do reduceSum and D2H togather
        combine_tokens = torch.stack([tokens_per_expert_group, tokens_per_expert], dim=0)
        # view: EP, E // EP
        # sum: EP, 每个rank
        combine_tokens = combine_tokens.view(2, self.ep_group.world_size, -1).sum(-1)
        combine_tokens_cpu = combine_tokens.cpu().tolist()
        # alltoall input splits, the size is the total number
        # of tokens that the current rank routes to other ranks
        input_splits = combine_tokens_cpu[1]
        # alltoall output splits, the size is the number of tokens
        #  that each rank receives from other ranks
        output_splits = combine_tokens_cpu[0]
        expert_tokens = sum(input_splits)
        expanded_x = expanded_x[:expert_tokens]
        # alltoall output, the size is the total number of tokens
        # that each rank routes to other ranks
        local_tokens = sum(output_splits)

        gathered_tokens = expanded_x.new_empty([local_tokens, expanded_x.shape[1]])
        torch.distributed.all_to_all_single(gathered_tokens, expanded_x, output_splits, input_splits, group=self.ep_group.device_group)

        gathered_pertoken_scale = None
        if has_quant:
            pertoken_scale = pertoken_scale[:expert_tokens]
            gathered_pertoken_scale = pertoken_scale.new_empty(gathered_tokens.shape[0])
            torch.distributed.all_to_all_single(gathered_pertoken_scale, pertoken_scale,
                                                output_splits, input_splits, group=self.ep_group.device_group)

        # reroute
        (
            expand_x,
            gathered_pertoken_scale,
            permute_token_idx,
            expert_token_count
        ) = torch_npu.npu_moe_re_routing(gathered_tokens, tokens_per_expert_group.view(self.ep_group.world_size, -1),
                                         per_token_scales=gathered_pertoken_scale)
        if has_quant:
            expand_x = (expand_x, gathered_pertoken_scale)
        dispatch_output = (tokens_per_expert_group, expanded_row_idx, permute_token_idx, input_splits, output_splits)
        output = (expand_x, expert_token_count, dispatch_output)
        return output

    def combine(self, expert_output, topk_ids, topk_weight, dispatch_output):
        tokens_per_expert_group, expanded_row_idx, permute_token_idx, input_splits, output_splits = dispatch_output
        # todo: rerouting
        new_x = torch.index_select(expert_output, 0, permute_token_idx.float().argsort().int())
        # new_x = torch_npu.npu_moe_re_routing(
        #     expert_output,
        #     tokens_per_expert_group.view(self.ep_group.world_size, -1).T
        # )[0]
        gathered_tokens = new_x.new_empty(sum(input_splits), new_x.shape[1])
        torch.distributed.all_to_all_single(gathered_tokens, new_x, input_splits, output_splits, group=self.ep_group.device_group)
        if gathered_tokens.size(0) < expanded_row_idx.size(0):
            gathered_tokens = torch.cat(
                [gathered_tokens,
                 gathered_tokens.new_zeros(expanded_row_idx.size(0) - gathered_tokens.size(0), gathered_tokens.size(1))
                ], dim=0)
        out = torch_npu.npu_moe_finalize_routing(
            gathered_tokens.unsqueeze(1),
            scales=topk_weight.to(gathered_tokens.dtype),
            expanded_src_to_dst_row=expanded_row_idx,
            export_for_source_row=topk_ids,
            drop_pad_mode=3,
            skip1=None,
            skip2=None,
            bias=None
        )
        return out


class TokensDispatcherAll2All:
    def __init__(
            self,
            layer,
            num_experts: int,
            zero_expert_num: int,
            ep_size: int,
            ep_rank: int,
            tp_size: int=1,
            tp_rank: int=0,
            quant: bool=False) -> None:
        self.group_ep: str = ''
        if ENV.npu_enable_all2all_comm:
            self.group_ep = get_all2all_ep_group().device_group._get_backend(
                                    torch.device("npu")).get_hccl_comm_name(ep_rank)
        self.quant_mode = 2 if quant else 0
        self.layer = layer
        self.common_kwargs = dict(
            group_ep=self.group_ep,
            ep_world_size=ep_size,
            tp_world_size=tp_size,
            ep_rank_id=ep_rank,
            tp_rank_id=tp_rank,
            moe_expert_num=num_experts,
            copy_expert_num=zero_expert_num
        )

    def dispatch(self, hidden_states, top_experts, expert_weights):
        kwargs = dict(
            x=hidden_states,
            expert_ids=top_experts,
            scales=self.layer.w13_smooth_scale_total if self.quant_mode else None,
            quant_mode=self.quant_mode,
            global_bs=0,
            expert_scales=expert_weights,
        )
        if ENV.npu_enable_graph:
            kwargs['group_tp'] = self.group_ep
        if global_server_args_dict["npu_enable_a2_dispatch_combine_opt"] and get_910b_num_gpus_per_node() > 0:
            kwargs["comm_alg"] = "hierarchy"
        (
            expand_x,
            dynamic_quant,
            expand_idx,
            expert_token_count,
            ep_recv_counts,
            tp_recv_counts,
            expand_scales
        ) = torch_npu.npu_moe_distribute_dispatch_v2(**kwargs, **self.common_kwargs)
        dispatch_output = (expand_idx, ep_recv_counts, tp_recv_counts, expand_scales)
        if self.quant_mode == 2:
            expand_x = (expand_x, dynamic_quant)
        return expand_x, expert_token_count, dispatch_output

    def combine(self, expert_output, top_experts, expert_weights, hidden_states, dispatch_output):
        expand_idx, ep_recv_counts, tp_recv_counts, expand_scales = dispatch_output
        kwargs = dict(
            expand_x=expert_output,
            expert_ids=top_experts.to(torch.int32),
            assist_info_for_combine=expand_idx,
            ep_send_counts=ep_recv_counts,
            tp_send_counts=tp_recv_counts,
            expert_scales=expert_weights,
            expand_scales=expand_scales,
            global_bs=0,
            ori_x=hidden_states,
        )
        if ENV.npu_enable_graph:
            kwargs['group_tp'] = self.group_ep
        if global_server_args_dict["npu_enable_a2_dispatch_combine_opt"] and get_910b_num_gpus_per_node() > 0:
            kwargs["comm_alg"] = "hierarchy"
        output_combine_all2all = torch_npu.npu_moe_distribute_combine_v2(**kwargs, **self.common_kwargs)
        return output_combine_all2all
