from enum import Enum
from typing import Optional

import torch
from torch import nn
from torch.nn import Parameter

from sglang.srt.distributed import (
    get_attn_tp_group,
    get_eps_communicator,
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
)
from sglang.srt.env import global_server_args_dict
from sglang.srt.layers.embedding.vocab_parallel_embedding import UnquantizedEmbeddingMethod, VocabParallelEmbedding
from sglang.srt.layers.quantization import QuantizeMethodBase
from sglang.srt.layers.utils import CP_METADATA, cp_split_and_rebuild_data
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.utils import get_colorful_logger, is_npu

__is_npu__ = is_npu()
if not __is_npu__:
    from eps.fast_oep import AllToAll

    try:
        from flashinfer import compute_n_gram_ids_v2
    except ImportError as e:
        import logging

        logger = logging.getLogger(__name__)
        logger.error(f"Failed to import compute_n_gram_ids_v2 from flashinfer: {e}")
        raise

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
        # Delay AllToAll creation to avoid conflicts with torch_memory_saver
        self.a2a = None

    def _ensure_a2a_initialized(self):
        """Lazy initialization of AllToAll to avoid torch_memory_saver conflicts."""
        if self.world_size > 0 and self.a2a is None:
            comm = get_eps_communicator()
            self.a2a = AllToAll(self.embedding_dim, self.num_embeddings_per_partition, self.max_num_global_tokens, comm.data_ptr())

    def forward(self, input_, num_global_tokens, global_num_tokens):
        if self.world_size > 1:
            self._ensure_a2a_initialized()
            self.a2a.dispatch(input_)
            output = torch.empty(input_.shape[1], input_.shape[0], self.embedding_dim, dtype=self.params_dtype, device=input_.device)
            self.a2a.combine(
                output,
                num_global_tokens * self.n_grams,
                self.n_grams,
                True,  # do_permute,
                self.weight,
            )
        else:
            output = self.linear_method.embedding(self, input_.long())
            output = torch.transpose(output, 0, 1)
        return output


class TorchOEPEmbedding(VocabParallelEmbedding):
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
        super().__init__(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            params_dtype=params_dtype,
            padding_size=world_size,
        )
        # Alias for `shard_indices.org_vocab_{start,end}_index`, matching
        # `EpsOEPEmbedding.begin`/`.end` so `FusedOverEmbedding.load_weight`
        # can read the same attribute names regardless of OEP backend.
        self.begin = self.shard_indices.org_vocab_start_index
        self.end = self.shard_indices.org_vocab_end_index

    @classmethod
    def _get_sp_token_num(cls, dp_rank_token_num, attn_tp_rank):
        """Mirrors how SP splitting divides a dp-rank's tokens among its attn-tp ranks."""
        base = dp_rank_token_num // get_attn_tp_group().world_size
        rem = dp_rank_token_num % get_attn_tp_group().world_size
        if attn_tp_rank < rem:
            return base + 1
        else:
            return base

    def forward(self, input_, num_global_tokens, global_num_tokens):
        """Look up n-gram embeddings via VocabParallelEmbedding."""

        seq_len = input_.shape[0]
        n_grams = input_.shape[1]
        flat_input = input_.permute(1, 0).contiguous().view(-1)  # [n_grams * seq_len]

        # Per-rank flattened row count for `VocabParallelEmbedding.pre_comm`/
        # `post_comm` (see `ForwardBatch.global_num_tokens` for its layout),
        # scaled by `n_grams` to match `flat_input`.
        global_sp_token_num = (
            [TorchOEPEmbedding._get_sp_token_num(x, i % get_attn_tp_group().world_size) * n_grams for i, x in enumerate(global_num_tokens)]
            if global_num_tokens is not None
            else None
        )

        output = super().forward(flat_input, global_sp_token_num=global_sp_token_num)
        return output.view(n_grams, seq_len, self.embedding_dim)


class OEPBackend(Enum):
    eps = OEPEmbedding
    torch = TorchOEPEmbedding


class FusedOverEmbedding(torch.nn.Module):
    """
    Computation logic:
    Compute OE:
    - input: [seq_len]
    - Call compute_n_gram_ids, compute corresponding n_gram_ids [seq_len, n-1, k]
    -- Update sgl-kernel: pass an additional offset list, offset is used to add to n_gram_id
       for specific n and k cases, to correspond to the fused large vocabulary
    - Pass through oe_embeder (still using VocabParallelEmbedding class), get [num_oe, seq_len, hidden_dim/num_oe]
    - torch.bmm([num_oe, seq_len, hidden_dim/num_oe], [hidden_dim/num_oe, hidden_dim/num_oe])
    - Get [num_oe, seq_len, hidden_dim]
    Compute word embedding:
    - Pass through word_embeder, get [seq_len, hidden_dim]
    Merge and take mean:
    - cat get [num_oe + 1, seq_len, hidden_dim] embeddings = torch.cat([word_embedding.unsqueeze(0), oe_embeddings], dim=0)
    - torch.mean(embeddings, dim=0)
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        over_embedding_m: int,
        over_embedding_k: int,
        over_embedding_n: int,
        over_embedding_base: int,
        eos_token_id: int,
        oe_ignore_tokens: Optional[list[int]] = None,
        embp: int = None,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.over_embedding_m = over_embedding_m
        self.over_embedding_k = over_embedding_k
        self.over_embedding_n = over_embedding_n
        self.eos_token_id = eos_token_id

        # Initialize regular vocabulary [vocab_size, hidden_dim]
        self.word_embeder = VocabParallelEmbedding(
            num_embeddings,
            embedding_dim,
            enable_tp=(not global_server_args_dict["enable_dp_attention"] and not global_server_args_dict["enable_dsa_qcp"]),
        )
        device = global_server_args_dict["device"]
        if oe_ignore_tokens:
            ignore_mask = torch.zeros(num_embeddings, dtype=torch.bool, device=device)
            ignore_mask[oe_ignore_tokens] = True
            self.register_buffer("oe_ignore_mask", ignore_mask)
        else:
            self.oe_ignore_mask = None

        self.n_grams = (over_embedding_n - 1) * over_embedding_k

        # Initialize OE vocabulary [m0+m1+...+m11, self.oe_hidden_dim]
        self.oe_hidden_dim = embedding_dim // self.n_grams
        exclusive_oe_embeder_size_sums = torch.zeros([self.n_grams + 1], dtype=torch.int32, device=device)
        for i in range(self.n_grams):
            m = int(over_embedding_m + i * 2 + 1)
            if embp is not None:
                m = m + embp - m % embp
            exclusive_oe_embeder_size_sums[i + 1] = exclusive_oe_embeder_size_sums[i] + m
            self.register_buffer("exclusive_oe_embeder_size_sums", exclusive_oe_embeder_size_sums)
        max_num_global_tokens = global_server_args_dict["chunked_prefill_size"] * get_tensor_model_parallel_world_size()
        if get_pp_group().is_first_rank:
            if global_server_args_dict["enable_oe_cpu_offload"]:
                self.oe_embeder = VocabParallelEmbedding(
                    num_embeddings=self.exclusive_oe_embeder_size_sums[-1],
                    embedding_dim=self.oe_hidden_dim,
                    enable_tp=True,
                    comm_group=get_tp_group(),
                    padding_size=128,
                    cpu_offload=True,
                )
            else:
                oep_backend = OEPBackend[global_server_args_dict["oep_backend"]]
                logger.info(f"Using OEPEmbedding backend: {oep_backend.name}")
                self.oe_embeder = oep_backend.value(
                    num_embeddings=self.exclusive_oe_embeder_size_sums[-1].item(),
                    embedding_dim=self.oe_hidden_dim,
                    rank=get_tensor_model_parallel_rank(),
                    world_size=get_tensor_model_parallel_world_size(),
                    max_num_global_tokens=max_num_global_tokens,
                    n_grams=self.n_grams,
                )

        # Initialize OE projection [12, self.oe_hidden_dim, hidden_dim]
        self.oe_projection = nn.Parameter(torch.empty(self.n_grams, self.oe_hidden_dim, embedding_dim), requires_grad=False)

        # Initialize weight tensor, avoid repeated computation when calculating n-gram ids
        oe_mods = torch.zeros([self.over_embedding_n - 1, self.over_embedding_k], dtype=torch.int32, device=device)
        oe_weights = torch.zeros([self.over_embedding_n - 1, self.over_embedding_k, self.over_embedding_n], dtype=torch.int32, device=device)
        for n in range(2, self.over_embedding_n + 1):
            for k in range(self.over_embedding_k):
                mod = self.over_embedding_m + 2 * ((n - 2) * self.over_embedding_k + k) + 1
                oe_mods[n - 2][k] = mod
                for delta in range(self.over_embedding_n):
                    oe_weights[n - 2][k][delta] = pow(over_embedding_base, delta, mod)
        self.register_buffer("oe_mods", oe_mods)
        self.register_buffer("oe_weights", oe_weights)
        self.oe_n_gram_ids = torch.zeros([global_server_args_dict["chunked_prefill_size"], self.n_grams], dtype=torch.int32, device=device)
        self.exclusive_req_len_sums = torch.zeros(global_server_args_dict["max_running_requests"] + 1, dtype=torch.int32, device=device)

    def load_weight(self, param: Parameter, weight_name: str, loaded_weight: torch.Tensor):
        if ".embed_tokens." in weight_name:
            # Regular vocabulary, load directly
            param.weight_loader(param, loaded_weight)
        elif ".oe_embed_tokens" in weight_name or "model.ngram_embeddings.embedders." in weight_name:
            """
            model.oe_embed_tokens0.weight
            OE vocabulary, calculate absolute row count to determine if it hits current TP range
            For example, if current OE vocabulary absolute row count is [100,200] and current TP only saves [75,150]
            Then only load absolute rows [100,150] into current vocabulary [25,50]
            """
            # Calculate which OE vocabulary this is
            if ".oe_embed_tokens" in weight_name:
                index = int(weight_name.replace("model.oe_embed_tokens", "").replace(".weight", ""))
            else:
                index = int(weight_name.replace("model.ngram_embeddings.embedders.", "").replace(".weight", ""))

            if global_server_args_dict["enable_oe_cpu_offload"] and index % get_tp_group().world_size != get_tp_group().rank_in_group:
                return
            # Absolute row count of current vocabulary in fused vocabulary
            oe_weight_start = self.exclusive_oe_embeder_size_sums[index]
            oe_weight_end = self.exclusive_oe_embeder_size_sums[index + 1]
            assert oe_weight_end - oe_weight_start == loaded_weight.shape[0], (
                f"Loaded weight size {loaded_weight.shape[0]} does not match expected size {oe_weight_end - oe_weight_start}."
            )
            # Absolute rows that current TP will load from fused vocabulary
            if global_server_args_dict["enable_oe_cpu_offload"]:
                begin = self.oe_embeder.shard_indices.org_vocab_start_index
                end = self.oe_embeder.shard_indices.org_vocab_end_index
            else:
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
        elif ".oe_embed_proj" in weight_name or "model.ngram_embeddings.post_projs." in weight_name:
            """
            model.oe_embed_proj0.weight
            OE projection matrix, very small, load completely
            """
            if ".oe_embed_proj" in weight_name:
                index = int(weight_name.replace("model.oe_embed_proj", "").replace(".weight", ""))
            else:
                index = int(weight_name.replace("model.ngram_embeddings.post_projs.", "").replace(".weight", ""))
            self.oe_projection[index].copy_(loaded_weight.data.t())

    def share_weights(self, layer: torch.nn.Module):
        if hasattr(self, "oe_embeder") and global_server_args_dict["enable_oe_cpu_offload"]:
            self.oe_embeder.share_barrier()

    @torch.no_grad()
    def forward(self, input_ids: torch.Tensor, forward_batch: ForwardBatch, is_draft=False):
        if forward_batch.oe_token_table is None:
            raise RuntimeError(
                f"OE is enabled but oe_token_table is missing: forward_mode={forward_batch.forward_mode}, batch_size={forward_batch.batch_size}"
            )

        if forward_batch.forward_mode.is_extend() or forward_batch.forward_mode.is_decode():
            torch.cumsum(forward_batch.oe_req_lens, dim=0, dtype=torch.int32, out=self.exclusive_req_len_sums[1 : 1 + forward_batch.batch_size])
            compute_n_gram_ids_v2(
                oe_n=self.over_embedding_n,
                oe_k=self.over_embedding_k,
                oe_weights=self.oe_weights,
                oe_mods=self.oe_mods,
                tokens=input_ids.to(torch.int32),
                exclusive_oe_embeder_size_sums=self.exclusive_oe_embeder_size_sums,
                exclusive_req_len_sums=self.exclusive_req_len_sums[: forward_batch.batch_size + 1],
                oe_token_table=forward_batch.oe_token_table,
                row_indices=forward_batch.req_pool_indices,
                column_starts=forward_batch.oe_column_starts,
                n_gram_ids=self.oe_n_gram_ids,
                eos_token_id=self.eos_token_id,
            )

        num_global_tokens, max_num_tokens_per_gpu = forward_batch.get_num_tokens(input_ids.shape[0])
        # [13, seq_len, hidden_dim]
        effective_input_ids = input_ids
        if CP_METADATA:
            all_hidden_states = torch.empty(
                [self.n_grams + 1, sum(CP_METADATA.value.cur_split_list), self.embedding_dim], dtype=self.oe_projection.dtype, device=input_ids.device
            )
            cp_input_ids = cp_split_and_rebuild_data(input_ids, CP_METADATA.value.split_list, CP_METADATA.value.zigzag_index)
            effective_input_ids = cp_input_ids
            all_hidden_states[0] = self.word_embeder(cp_input_ids)
            oe_n_gram_ids = cp_split_and_rebuild_data(
                self.oe_n_gram_ids[: len(input_ids)], CP_METADATA.value.split_list, CP_METADATA.value.zigzag_index
            )
        else:
            all_hidden_states = torch.empty(
                [self.n_grams + 1, len(input_ids), self.embedding_dim], dtype=self.oe_projection.dtype, device=input_ids.device
            )
            all_hidden_states[0] = self.word_embeder(input_ids)
            oe_n_gram_ids = self.oe_n_gram_ids[: len(input_ids)]

        # oe_hidden_states: [12, seq_len, hidden_dim / 12]
        if global_server_args_dict["enable_oe_cpu_offload"]:
            oe_hidden_states = torch.nn.functional.embedding(oe_n_gram_ids.cpu(), self.oe_embeder.weight).to(device=input_ids.device)
            oe_hidden_states.transpose_(0, 1)
        else:
            oe_hidden_states = self.oe_embeder(oe_n_gram_ids, num_global_tokens, forward_batch.global_num_tokens)

        torch.bmm(oe_hidden_states, self.oe_projection, out=all_hidden_states[1:])
        # Add a word embedding path
        mean_hidden_states = all_hidden_states.mean(dim=0)
        if self.oe_ignore_mask is not None:
            oe_ignore_flags = self.oe_ignore_mask[effective_input_ids.clamp(0, self.oe_ignore_mask.shape[0] - 1)]
            mean_hidden_states = torch.where(oe_ignore_flags.unsqueeze(1), all_hidden_states[0], mean_hidden_states)
        return mean_hidden_states
