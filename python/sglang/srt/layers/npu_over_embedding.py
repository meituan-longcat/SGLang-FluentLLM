from contextlib import nullcontext

import torch
import torch_npu
from torch import nn
from torch.nn import Parameter

from sglang.srt.layers.moe.npu_moe.token_dispatcher import TokensDispatcherAll2All
from sglang.srt.layers.quantization import QuantizeMethodBase
from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding, UnquantizedEmbeddingMethod
from sglang.srt.managers.schedule_batch import global_server_args_dict
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.distributed import get_tensor_model_parallel_rank, get_attn_tp_group, get_ep_group, \
    get_all2all_ep_group, get_tp_group, get_pp_group

from sglang.srt.layers.linear import ReplicatedLinear
from sglang.srt.utils import is_npu, logger

__is_npu__ = is_npu()

from sglang.srt.layers.over_embedding import FusedOverEmbedding

DEFAULT_VOCAB_PADDING_SIZE = 64

import torchair as tng

def npu_super_kernel(*args, flag=True, **kwargs):
    if global_server_args_dict['npu_enable_super_kernel'] and flag:
        return tng.scope.super_kernel(*args, **kwargs)
    else:
        return nullcontext()

class NpuOverEmbedding(torch.nn.Module):
    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 over_embedding_m: int,
                 over_embedding_k: int,
                 over_embedding_n: int,
                 oe_ignore_tokens):
        super().__init__()
        self.num_embeddings=num_embeddings
        self.embedding_dim=embedding_dim
        self.over_embedding_m=over_embedding_m
        self.over_embedding_k=over_embedding_k
        self.over_embedding_n=over_embedding_n
        self.oe_ignore_tokens=torch.tensor(oe_ignore_tokens)

        # 初始化普通词表 [vocab_size, hidden_dim]
        self.word_embeder=VocabParallelEmbedding(
            num_embeddings,
            embedding_dim,
            enable_tp=not global_server_args_dict["enable_dp_attention"],
        )
        device=global_server_args_dict["device"]

        self.n_grams=(over_embedding_n-1)*over_embedding_k

        # 初始化oe词表 [m0+m1+...+m11, self.oe_hidden_dim]
        self.oe_hidden_dim=embedding_dim//self.n_grams
        self.exclusive_oe_embeder_size_sums=torch.zeros([self.n_grams+1],
                                                        dtype=torch.int32,
                                                        device=device)
        for i in range(self.n_grams):
            m=int(over_embedding_m+i*2+1)
            self.exclusive_oe_embeder_size_sums[i+1]=self.exclusive_oe_embeder_size_sums[i]+m

        if get_pp_group().is_first_rank:
            self.oe_embeder = VocabParallelEmbedding(
                num_embeddings=self.exclusive_oe_embeder_size_sums[-1],
                embedding_dim=self.oe_hidden_dim,
                enable_tp=True,
                comm_group=get_tp_group(),
                padding_size=128,
                cpu_offload=global_server_args_dict["npu_enable_oe_cpu_offload"],
            )

        # 初始化oe projection [12, self.oe_hidden_dim, hidden_dim]
        self.oe_projection=nn.Parameter(
            torch.empty(self.n_grams, self.oe_hidden_dim, embedding_dim),
            requires_grad=False
        )

        # 初始化weight tensor，避免计算n-gram id的时候反复计算
        self.oe_mods=torch.zeros([self.over_embedding_n-1, self.over_embedding_k], dtype=torch.int32)
        self.oe_weights=torch.zeros([self.over_embedding_n-1, self.over_embedding_k, self.over_embedding_n],
                                    dtype=torch.int32)
        for n in range(2, self.over_embedding_n+1):
            for k in range(self.over_embedding_k):
                mod=self.over_embedding_m+2*((n-2)*self.over_embedding_k+k)+1
                self.oe_mods[n-2][k]=mod
                for delta in range(self.over_embedding_n):
                    self.oe_weights[n-2][k][delta]=pow(num_embeddings, delta, mod)
        self.scale = 1 + self.over_embedding_k * (self.over_embedding_n - 1)

        if global_server_args_dict["disaggregation_mode"]=="decode":
            self.dispatcher_all2all = TokensDispatcherAll2All(
                self,
                1,
                0,
                ep_size=get_all2all_ep_group().world_size,
                ep_rank=get_all2all_ep_group().rank_in_group,
                quant=False
            )

    def load_weight(self, param: Parameter, weight_name: str, loaded_weight: torch.Tensor):
        if '.embed_tokens.' in weight_name:
            # 普通词表，直接加载即可
            param.weight_loader(param, loaded_weight)
        elif '.oe_embed_tokens' in weight_name:
            if not get_pp_group().is_first_rank:
                return
            '''
            model.oe_embed_tokens0.weight
            oe词表，换算一下绝对行数，判断是否命中了当前TP的范围
            比如当前oe词表的绝对行数是[100,200] 当前TP只保存[75,150]
            那么只把绝对行数[100,150]的加载到当前的词表的[25,50]里面
            '''
            # 计算下当前是第几个oe词表
            index=int(weight_name.replace('model.oe_embed_tokens', '').replace('.weight', ''))
            # 当前词表在融合词表中的绝对行数
            oe_weight_start=self.exclusive_oe_embeder_size_sums[index]
            oe_weight_end=self.exclusive_oe_embeder_size_sums[index+1]
            assert oe_weight_end - oe_weight_start == loaded_weight.shape[
                0], f'Loaded weight size {loaded_weight.shape[0]} does not match expected size {oe_weight_end - oe_weight_start}.'
            # 当前TP会加载融合词表中的绝对行数
            tp_start=self.oe_embeder.shard_indices.org_vocab_start_index
            tp_end=self.oe_embeder.shard_indices.org_vocab_end_index
            # 需要加载的词表的绝对行数
            to_load_start=max(oe_weight_start, tp_start)
            to_load_end=min(oe_weight_end, tp_end)
            if to_load_start<to_load_end:
                # 计算在原始权重中的偏移量
                src_start=to_load_start-oe_weight_start
                src_end=to_load_end-oe_weight_start
                # 计算在当前TP权重中的偏移量
                dest_start=to_load_start-tp_start
                dest_end=to_load_end-tp_start
                self.oe_embeder.weight.data[dest_start:dest_end]=loaded_weight[src_start:src_end]
            else:
                return
        elif '.oe_embed_proj' in weight_name:
            '''
            model.oe_embed_proj0.weight
            oe投影矩阵，很小，完整加载即可
            '''
            index=int(weight_name.replace('model.oe_embed_proj', '').replace('.weight', ''))
            self.oe_projection[index].copy_(loaded_weight.data.t())

    def process_weights_after_loading(self, layer: torch.nn.Module):
        torch._dynamo.mark_static(self.exclusive_oe_embeder_size_sums)
        torch._dynamo.mark_static(self.oe_mods)
        torch._dynamo.mark_static(self.oe_weights)

        self.oe_projection.data = self.oe_projection.data / self.scale

    def forward(self,
                input_ids: torch.Tensor,
                forward_batch: ForwardBatch,
                enable_sp=False,
                is_draft=False):
        global_sp_token_num=forward_batch.global_sp_num_tokens
        if global_sp_token_num:
            oe_global_sp_token_num=[x*(self.over_embedding_n-1)*self.over_embedding_k for x in
                                    global_sp_token_num] if global_sp_token_num is not None else None
        else:
            oe_global_sp_token_num=None

        hidden_states = self.word_embeder(input_ids, global_sp_token_num=global_sp_token_num, enable_sp=enable_sp)
        if forward_batch.forward_mode.is_extend() or forward_batch.forward_mode.is_decode():
            if is_draft and forward_batch.forward_mode == ForwardMode.EXTEND:
                # After training side removes the logic that excludes the 0th token from n-gram id calculation during draft prefill, this logic can be removed
                forward_batch.oe_token_table[forward_batch.req_pool_indices, 0] = -forward_batch.oe_token_table[forward_batch.req_pool_indices, 0]

            exclusive_req_len_sums=torch.cumsum(forward_batch.oe_req_lens, dim=0, dtype=torch.int32)
            oe_n_gram_ids=torch_npu.compute_n_gram_ids(
                oe_weights=self.oe_weights,
                oe_mods=self.oe_mods,
                exclusive_oe_embeder_size_sums=self.exclusive_oe_embeder_size_sums,
                tokens=input_ids.to(torch.int32),
                exclusive_req_len_sums=exclusive_req_len_sums,
                oe_token_table=forward_batch.oe_token_table,
                row_indices=forward_batch.req_pool_indices,
                column_starts=forward_batch.oe_column_starts,
                batch_size=forward_batch.req_pool_indices.shape[0],
                oe_n=self.over_embedding_n,
                oe_k=self.over_embedding_k,
                max_context_len=forward_batch.oe_token_table.shape[1]
            )

            if is_draft and forward_batch.forward_mode == ForwardMode.EXTEND:
                forward_batch.oe_token_table[forward_batch.req_pool_indices, 0] = -forward_batch.oe_token_table[forward_batch.req_pool_indices, 0]
        else:
            oe_n_gram_ids=torch.empty([input_ids.shape[0], (self.over_embedding_n-1)*self.over_embedding_k], dtype=torch.int32, device=input_ids.device)

        if enable_sp:
            oe_n_gram_ids=get_attn_tp_group().split_tensor(oe_n_gram_ids.view(-1, (self.over_embedding_n-1)*self.over_embedding_k), global_sp_token_num)
            oe_n_gram_ids=oe_n_gram_ids.permute(1, 0).contiguous().view(-1)
            oe_n_gram_ids=self.oe_embeder.comm_group.gather_tensor(oe_n_gram_ids, global_sp_token_num=oe_global_sp_token_num).view(-1)
            oe_hidden_states=(self.oe_embeder(oe_n_gram_ids, global_sp_token_num=oe_global_sp_token_num, enable_sp=True, skip_pre_comm=True)
                                .view((self.over_embedding_n-1)*self.over_embedding_k, -1, self.oe_hidden_dim))
        else:
            oe_hidden_states=(self.oe_embeder(oe_n_gram_ids.permute(1, 0).contiguous().view(-1), global_sp_token_num=oe_global_sp_token_num)
                              .view((self.over_embedding_n-1)*self.over_embedding_k, -1, self.oe_hidden_dim))

        hidden_states=hidden_states/self.scale+torch.bmm(oe_hidden_states, self.oe_projection).sum(dim=0)
        return hidden_states
