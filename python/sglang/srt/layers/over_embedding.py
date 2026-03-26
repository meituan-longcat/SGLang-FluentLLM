from typing import Optional

import torch
from typing import List
from torch import nn
from torch.nn import Parameter

from eps.fast_oep import AllToAll

try:
    from flashinfer import compute_n_gram_ids_v2
except ImportError as e:
    import logging
    logger = logging.getLogger(__name__)
    logger.error(f"Failed to import compute_n_gram_ids_v2 from flashinfer: {e}")
    raise

from sglang.srt.layers.quantization import QuantizeMethodBase
from sglang.srt.distributed import (
    get_eps_communicator,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from sglang.srt.layers.vocab_parallel_embedding import UnquantizedEmbeddingMethod, VocabParallelEmbedding
from sglang.srt.env import global_server_args_dict
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.utils import get_colorful_logger

logger = get_colorful_logger(__name__)


class OEPEmbedding(torch.nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        rank: int,
        world_size: int,
        max_num_global_tokens: int,
        n_grams: int,
        params_dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.rank = rank
        self.world_size = world_size
        self.max_num_global_tokens = max_num_global_tokens * n_grams
        self.n_grams = n_grams
        if params_dtype is None:
            self.params_dtype = torch.get_default_dtype()
        else:
            self.params_dtype = params_dtype

        self.linear_method: QuantizeMethodBase = UnquantizedEmbeddingMethod()

        if num_embeddings % self.world_size != 0:
            self.num_embeddings = (num_embeddings + self.world_size - 1) // self.world_size * self.world_size
            logger.warning(f"{num_embeddings=} is not divisible by {self.world_size}, padded to {self.num_embeddings}")
        else:
            self.num_embeddings = num_embeddings
            logger.warning(f"{self.num_embeddings=}")

        self.num_embeddings_per_partition = self.num_embeddings // self.world_size

        self.begin = self.num_embeddings_per_partition * self.rank
        self.end = self.num_embeddings_per_partition * (self.rank + 1)

        self.linear_method.create_weights(
            self,
            self.embedding_dim,
            [self.num_embeddings_per_partition],
            params_dtype=self.params_dtype,
            weight_loader=None,
        )

        if self.world_size > 0:
            comm = get_eps_communicator()
            self.a2a = AllToAll(
                self.embedding_dim,
                self.num_embeddings_per_partition,
                self.max_num_global_tokens,
                comm.data_ptr()
            )

    def forward(self, input_, num_global_tokens):
        if self.world_size > 1:
            self.a2a.dispatch(input_)
            output = torch.empty(input_.shape[1],
                                 input_.shape[0],
                                 self.embedding_dim,
                                 dtype=self.params_dtype, device=input_.device)
            self.a2a.combine(
                output,
                num_global_tokens * self.n_grams,
                self.n_grams,
                True, # do_permute,
                self.weight
            )
        else:
            output = self.linear_method.embedding(self, input_.long())
        return output


class FusedOverEmbedding(torch.nn.Module):
    """
    Computation logic:
    Compute OE:
    - input: [seq_len]
    - Call compute_n_gram_ids, compute corresponding n_gram_ids [seq_len, n-1, k]
    -- Update sgl-kernel: pass an additional offset list, offset is used to add to n_gram_id for specific n and k cases, to correspond to the fused large vocabulary
    - Pass through oe_embeder (still using VocabParallelEmbedding class), get [num_oe, seq_len, hidden_dim/num_oe]
    - torch.bmm([num_oe, seq_len, hidden_dim/num_oe], [hidden_dim/num_oe, hidden_dim/num_oe])
    - Get [num_oe, seq_len, hidden_dim]
    Compute word embedding:
    - Pass through word_embeder, get [seq_len, hidden_dim]
    Merge and take mean:
    - cat get [num_oe + 1, seq_len, hidden_dim] embeddings = torch.cat([word_embedding.unsqueeze(0), oe_embeddings], dim=0)
    - torch.mean(embeddings, dim=0)
    """
    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 over_embedding_m: int,
                 over_embedding_k: int,
                 over_embedding_n: int,
                 oe_ignore_tokens: int,
                 oe_m_padding_size: int = 1,
                 num_embeddings_text: int = None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.over_embedding_m = over_embedding_m
        self.over_embedding_k = over_embedding_k
        self.over_embedding_n = over_embedding_n
        self.oe_m_padding_size =oe_m_padding_size
        self.oe_ignore_tokens = torch.tensor(oe_ignore_tokens)
        if num_embeddings_text is None:
            num_embeddings_text = num_embeddings

        # Initialize regular vocabulary [vocab_size, hidden_dim]
        self.word_embeder = VocabParallelEmbedding(
            num_embeddings,
            embedding_dim,
            enable_tp=not global_server_args_dict["enable_dp_attention"],
        )
        device = global_server_args_dict["device"]

        self.n_grams = (over_embedding_n - 1) * over_embedding_k

        # Initialize OE vocabulary [m0+m1+...+m11, self.oe_hidden_dim]
        self.oe_hidden_dim = embedding_dim // self.n_grams
        self.exclusive_oe_embeder_size_sums = torch.zeros([self.n_grams + 1],
                                                          dtype=torch.int32,
                                                          device=device)
        for i in range(self.n_grams):
            m = int(over_embedding_m + i * 2 + 1)
            padded_m = m + (self.oe_m_padding_size - m % self.oe_m_padding_size) % self.oe_m_padding_size
            self.exclusive_oe_embeder_size_sums[i + 1] = self.exclusive_oe_embeder_size_sums[i] + padded_m

        max_num_global_tokens = global_server_args_dict["chunked_prefill_size"] * get_tensor_model_parallel_world_size()
        self.oe_embeder = OEPEmbedding(
                num_embeddings=self.exclusive_oe_embeder_size_sums[-1].item(),
                embedding_dim=self.oe_hidden_dim,
                rank=get_tensor_model_parallel_rank(),
                world_size=get_tensor_model_parallel_world_size(),
                max_num_global_tokens=max_num_global_tokens,
                n_grams=self.n_grams
            )

        # Initialize OE projection [12, self.oe_hidden_dim, hidden_dim]
        self.oe_projection = nn.Parameter(
            torch.empty(self.n_grams, self.oe_hidden_dim, embedding_dim),
            requires_grad=False
        )

        # Initialize weight tensor, avoid repeated computation when calculating n-gram ids
        self.oe_mods = torch.zeros([self.over_embedding_n-1, self.over_embedding_k], dtype=torch.int32)
        self.oe_weights = torch.zeros([self.over_embedding_n-1, self.over_embedding_k, self.over_embedding_n], dtype=torch.int32)
        for n in range(2, self.over_embedding_n + 1):
            for k in range(self.over_embedding_k):
                mod = self.over_embedding_m + 2 * ((n - 2) * self.over_embedding_k + k) + 1
                self.oe_mods[n-2][k] = mod
                for delta in range(self.over_embedding_n):
                    self.oe_weights[n-2][k][delta] = pow(num_embeddings_text, delta, mod) # 在多模场景下，num_embeddings_text可能不等于num_embeddings
        self.oe_n_gram_ids = torch.zeros([global_server_args_dict['chunked_prefill_size'], self.n_grams], dtype=torch.int32, device=device)
        self.exclusive_req_len_sums = torch.zeros(global_server_args_dict['max_running_requests'] + 1, dtype=torch.int32, device=device)

    def load_weight(self, param: Parameter, weight_name: str, loaded_weight: torch.Tensor):
        if '.embed_tokens.' in weight_name:
            # Regular vocabulary, load directly
            param.weight_loader(param, loaded_weight)
        elif '.oe_embed_tokens' in weight_name or 'model.ngram_embeddings.embedders.' in weight_name:
            '''
            model.oe_embed_tokens0.weight
            OE vocabulary, calculate absolute row count to determine if it hits current TP range
            For example, if current OE vocabulary absolute row count is [100,200] and current TP only saves [75,150]
            Then only load absolute rows [100,150] into current vocabulary [25,50]
            '''
            # Calculate which OE vocabulary this is
            if '.oe_embed_tokens' in weight_name:
                index = int(weight_name.replace('model.oe_embed_tokens', '').replace('.weight', ''))
            else:
                index = int(weight_name.replace('model.ngram_embeddings.embedders.', '').replace('.weight', ''))
            # Absolute row count of current vocabulary in fused vocabulary
            oe_weight_start = self.exclusive_oe_embeder_size_sums[index]
            oe_weight_end = self.exclusive_oe_embeder_size_sums[index + 1]
            assert oe_weight_end - oe_weight_start == loaded_weight.shape[
                0], f'Loaded weight size {loaded_weight.shape[0]} does not match expected size {oe_weight_end - oe_weight_start}.'
            # Absolute rows that current TP will load from fused vocabulary
            begin = self.oe_embeder.begin
            end = self.oe_embeder.end
            # Absolute rows of vocabulary that need to be loaded
            to_load_start = max(oe_weight_start, begin)
            to_load_end = min(oe_weight_end, end)
            if to_load_start < to_load_end:
                # Calculate offset in original weights
                src_start = to_load_start - oe_weight_start
                src_end = to_load_end - oe_weight_start
                # Calculate offset in current TP weights
                dest_start = to_load_start - begin
                dest_end = to_load_end - begin
                self.oe_embeder.weight.data[dest_start:dest_end] = loaded_weight[src_start:src_end]
            else:
                return
        elif '.oe_embed_proj' in weight_name or 'model.ngram_embeddings.post_projs.' in weight_name:
            '''
            model.oe_embed_proj0.weight
            OE projection matrix, very small, load completely
            '''
            if '.oe_embed_proj' in weight_name:
                index = int(weight_name.replace('model.oe_embed_proj', '').replace('.weight', ''))
            else:
                index = int(weight_name.replace('model.ngram_embeddings.post_projs.', '').replace('.weight', ''))
            self.oe_projection[index].copy_(loaded_weight.data.t())

    def forward(self,
                input_ids: torch.Tensor,
                forward_batch: ForwardBatch,
                is_draft = False):
        if forward_batch.forward_mode.is_extend() or forward_batch.forward_mode.is_decode():
            if is_draft and forward_batch.forward_mode == ForwardMode.EXTEND:
                # After training side removes the logic that excludes the 0th token from n-gram id calculation during draft prefill, this logic can be removed
                forward_batch.oe_token_table[forward_batch.req_pool_indices, 0] = -forward_batch.oe_token_table[forward_batch.req_pool_indices, 0]
            torch.cumsum(forward_batch.oe_req_lens, dim=0, dtype=torch.int32, out=self.exclusive_req_len_sums[1:1+forward_batch.batch_size])
            # print(f"{forward_batch.oe_token_table[forward_batch.req_pool_indices,forward_batch.oe_column_starts-5:forward_batch.oe_column_starts+1]=}")
            compute_n_gram_ids_v2(
                oe_n=self.over_embedding_n,
                oe_k=self.over_embedding_k,
                oe_weights=self.oe_weights,
                oe_mods=self.oe_mods,
                tokens=input_ids.to(torch.int32),
                exclusive_oe_embeder_size_sums=self.exclusive_oe_embeder_size_sums,
                exclusive_req_len_sums=self.exclusive_req_len_sums[:forward_batch.batch_size + 1],
                oe_token_table=forward_batch.oe_token_table,
                row_indices=forward_batch.req_pool_indices,
                column_starts=forward_batch.oe_column_starts,
                n_gram_ids=self.oe_n_gram_ids[:forward_batch.batch_size]
            )
            if is_draft and forward_batch.forward_mode == ForwardMode.EXTEND:
                forward_batch.oe_token_table[forward_batch.req_pool_indices, 0] = -forward_batch.oe_token_table[forward_batch.req_pool_indices, 0]

        num_global_tokens, max_num_tokens_per_gpu = forward_batch.get_num_tokens(input_ids.shape[0])

        mean_hidden_states = self.compute_hidden_states(input_ids, self.oe_n_gram_ids[:len(input_ids)], num_global_tokens)
        return mean_hidden_states
    
    def compute_hidden_states(self, input_ids: torch.Tensor, oe_n_gram_ids: torch.Tensor, num_global_tokens: int):
        # [13, seq_len, hidden_dim]
        all_hidden_states = torch.empty([self.n_grams + 1, len(input_ids), self.embedding_dim], dtype=self.oe_projection.dtype, device=input_ids.device)
        all_hidden_states[0] = self.word_embeder(input_ids)
        # oe_hidden_states: [12, seq_len, hidden_dim / 12]
        oe_hidden_states = self.oe_embeder(oe_n_gram_ids, num_global_tokens)
        torch.bmm(oe_hidden_states, self.oe_projection, out=all_hidden_states[1:])
        # Add a word embedding path
        mean_hidden_states = all_hidden_states.mean(dim=0)
        # results = torch.where(oe_ignore_input_ids_flags.unsqueeze(1), all_hidden_states[0], mean_hidden_states)
        return mean_hidden_states
    
