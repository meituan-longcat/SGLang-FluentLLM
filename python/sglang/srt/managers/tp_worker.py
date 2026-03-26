# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A tensor parallel worker."""
from __future__ import annotations
from typing import TYPE_CHECKING, Optional
import os

from sglang.srt.layers.dp_attention import get_attention_tp_size
from sglang.srt.utils import get_colorful_logger, is_npu
import threading
from typing import Optional, Tuple

import torch

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.hf_transformers_utils import get_processor, get_tokenizer
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.io_struct import (
    GetWeightsByNameReqInput,
    InitWeightsUpdateGroupReqInput,
    UpdateWeightFromDiskReqInput,
    UpdateWeightsFromDistributedReqInput,
    UpdateWeightsFromTensorReqInput,
)
from sglang.srt.managers.schedule_batch import ModelWorkerBatch
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import MultiprocessingSerializer, broadcast_pyobj, set_random_seed

if TYPE_CHECKING:
    from sglang.srt.managers.cache_controller import LayerDoneCounter

logger = get_colorful_logger(__name__)

_is_npu = is_npu()

_RANDOM_ROUTER = None
if _RANDOM_ROUTER is None:
    if os.getenv("RANDOM_ROUTER", "0") == "1":
        _RANDOM_ROUTER = True
    else:
        _RANDOM_ROUTER = False

class TpModelWorker:
    """A tensor parallel model worker."""

    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        attn_tp_rank: int,
        moe_ep_rank: int,
        global_rank: int,
        nccl_port: int,
        is_draft_worker: bool = False,
        req_to_token_pool= None,
        kv_allocator=None,
        oe_token_table=None
    ):
        # Parse args
        self.attn_tp_rank = attn_tp_rank

        # Init model and tokenizer
        self.model_config = ModelConfig(
            (
                server_args.model_path
                if not is_draft_worker
                else server_args.speculative_draft_model_path
            ),
            trust_remote_code=server_args.trust_remote_code,
            revision=server_args.revision,
            context_length=server_args.context_length,
            model_override_args=server_args.json_model_override_args,
            is_embedding=server_args.is_embedding,
            dtype=server_args.dtype,
            quantization=server_args.quantization,
            enable_tbo=server_args.enable_tbo if not is_draft_worker else False,
            enable_sbo=server_args.enable_sbo if not is_draft_worker else False,
            is_draft_worker=is_draft_worker,
            server_args=server_args,
        )
        self.use_over_embedding = self.model_config.use_over_embedding
        self.model_runner = ModelRunner(
            model_config=self.model_config,
            mem_fraction_static=server_args.mem_fraction_static,
            gpu_id=gpu_id,
            attn_tp_rank=attn_tp_rank,
            attn_tp_size=server_args.attn_tp_size,
            world_size=server_args.world_size,
            moe_ep_rank=moe_ep_rank,
            moe_ep_size=server_args.ep_size,
            global_rank=global_rank,
            nccl_port=nccl_port,
            server_args=server_args,
            is_draft_worker=is_draft_worker,
            req_to_token_pool=req_to_token_pool,
            kv_allocator=kv_allocator,
            oe_token_table=oe_token_table
        )
        if server_args.skip_tokenizer_init:
            self.tokenizer = self.processor = None
        else:
            if self.model_config.is_multimodal:
                self.processor = get_processor(
                    server_args.tokenizer_path,
                    tokenizer_mode=server_args.tokenizer_mode,
                    trust_remote_code=server_args.trust_remote_code,
                    revision=server_args.revision,
                )
                self.tokenizer = self.processor.tokenizer
            else:
                self.tokenizer = get_tokenizer(
                    server_args.tokenizer_path,
                    tokenizer_mode=server_args.tokenizer_mode,
                    trust_remote_code=server_args.trust_remote_code,
                    revision=server_args.revision,
                )
        self.device = self.model_runner.device

        # Profile number of tokens
        self.max_total_num_tokens = self.model_runner.max_total_num_tokens
        self.max_prefill_tokens = server_args.max_prefill_tokens
        self.max_running_requests = min(
            (
                self.max_total_num_tokens // 2
                if server_args.max_running_requests is None
                else server_args.max_running_requests
                // (server_args.dp_size if server_args.enable_dp_attention else 1)
            ),
            self.model_runner.req_to_token_pool.size,
        )
        self.max_req_len = min(
            self.model_config.context_len - 1,
            self.max_total_num_tokens - self.model_runner.page_size, # page 0 is for padding
        )
        self.max_req_input_len = self.max_req_len - 5
        assert (
            self.max_req_len > 0 and self.max_req_input_len > 0
        ), "Memory pool size is too small"

        # Sync random seed across TP workers
        if get_attention_tp_size() > 1:
            self.random_seed = broadcast_pyobj(
                [server_args.random_seed],
                self.model_runner.attention_tp_group.rank,
                self.model_runner.attention_tp_group.cpu_group if not _is_npu else self.model_runner.tp_group.cpu_group,
                src=self.model_runner.attention_tp_group.ranks[0],
            )[0]
        else:
            self.random_seed = server_args.random_seed
        set_random_seed(self.random_seed)
        if _RANDOM_ROUTER:
            set_random_seed(global_rank)
        else:
            set_random_seed(self.random_seed)

        # for hicache load
        self.hicache_layer_transfer_counter = None

    def get_worker_info(self):
        return (
            self.max_total_num_tokens,
            self.max_prefill_tokens,
            self.max_running_requests,
            self.max_req_len,
            self.max_req_input_len,
            self.random_seed,
            self.device,
            self.model_runner.req_to_token_pool.size,
            self.model_runner.req_to_token_pool.max_context_len,
            self.model_runner.token_to_kv_pool.size,
        )

    def get_pad_input_ids_func(self):
        return getattr(self.model_runner.model, "pad_input_ids", None)

    def get_tp_cpu_group(self):
        return self.model_runner.tp_group.cpu_group

    def get_tp_group(self):
        return self.model_runner.tp_group

    def get_attention_tp_cpu_group(self):
        return self.model_runner.attention_tp_group.cpu_group

    def get_attention_tp_group(self):
        return self.model_runner.attention_tp_group

    def get_memory_pool(self):
        return (
            self.model_runner.req_to_token_pool,
            self.model_runner.token_to_kv_pool,
            self.model_runner.kv_allocator,
        )

    def register_hicache_layer_transfer_counter(self, counter: LayerDoneCounter):
        self.hicache_layer_transfer_counter = counter

    def set_hicache_consumer(self, consumer_index: int):
        if self.hicache_layer_transfer_counter is not None:
            self.hicache_layer_transfer_counter.set_consumer(consumer_index)
    def forward_batch_generation(
        self,
        model_worker_batch: ModelWorkerBatch,
        launch_done: Optional[threading.Event] = None,
        skip_sample: bool = False,
    ) -> Tuple[LogitsProcessorOutput, Optional[torch.Tensor]]:
        self.set_hicache_consumer(model_worker_batch.hicache_consumer_index)
        forward_batch = ForwardBatch.init_new(model_worker_batch, self.model_runner)
        if self.use_over_embedding:
            if forward_batch.forward_mode == ForwardMode.EXTEND:
                forward_batch.oe_column_starts[:forward_batch.batch_size] = forward_batch.extend_prefix_lens
                forward_batch.oe_req_lens[:forward_batch.batch_size] = forward_batch.extend_seq_lens
            elif forward_batch.forward_mode == ForwardMode.DECODE:
                forward_batch.oe_column_starts[:forward_batch.batch_size] = forward_batch.seq_lens - 1
                forward_batch.oe_req_lens[:forward_batch.batch_size] = 1
        logits_output = self.model_runner.forward(forward_batch)
        if launch_done:
            launch_done.set()

        if skip_sample:
            next_token_ids = None
            next_token_multi_ids = None
        else:
            next_token_ids, output_tensor_dict = self.model_runner.sample(logits_output, forward_batch)
            if forward_batch.temp_multi_ids is not None:
                # Multi id decoding is completed by multimodal module itself
                assert next_token_ids.shape == (forward_batch.batch_size, )
                assert len(forward_batch.temp_multi_ids.shape) == 2 and forward_batch.temp_multi_ids.shape[0] == forward_batch.batch_size, f"\033[31m[{forward_batch=}]\033[0m"
                next_token_multi_ids = forward_batch.temp_multi_ids
            else:
                next_token_multi_ids = None
            if model_worker_batch.sampling_info.thinking_budgets is not None:  # check whether thinking budget is out of
                model_worker_batch.sampling_info.apply_thinking_budgets(model_worker_batch.seq_lens, next_token_ids)
            if model_worker_batch.disagg_set_aux_fn is not None:
                model_worker_batch.disagg_set_aux_fn(next_token_ids, logits_output)

        return logits_output, next_token_ids, next_token_multi_ids, output_tensor_dict

    def forward_batch_embedding(self, model_worker_batch: ModelWorkerBatch):
        forward_batch = ForwardBatch.init_new(model_worker_batch, self.model_runner)
        logits_output = self.model_runner.forward(forward_batch)
        embeddings = logits_output.embeddings
        return embeddings

    def update_weights_from_disk(self, recv_req: UpdateWeightFromDiskReqInput):
        success, message = self.model_runner.update_weights_from_disk(
            recv_req.model_path, recv_req.load_format
        )
        return success, message

    def init_weights_update_group(self, recv_req: InitWeightsUpdateGroupReqInput):
        success, message = self.model_runner.init_weights_update_group(
            recv_req.master_address,
            recv_req.master_port,
            recv_req.rank_offset,
            recv_req.world_size,
            recv_req.group_name,
            recv_req.backend,
        )
        return success, message

    def update_weights_from_distributed(
        self, recv_req: UpdateWeightsFromDistributedReqInput
    ):
        success, message = self.model_runner.update_weights_from_distributed(
            recv_req.name, recv_req.dtype, recv_req.shape
        )
        return success, message

    def update_weights_from_tensor(self, recv_req: UpdateWeightsFromTensorReqInput):
        success, message = self.model_runner.update_weights_from_tensor(
            named_tensors=MultiprocessingSerializer.deserialize(
                # TODO(2025-09-06 yanghao32): Support weight update for different parallel modes, currently only supports attn_tp_size==dense_dp_size case
                recv_req.serialized_named_tensors[self.attn_tp_rank]
            ),
            load_format=recv_req.load_format,
        )
        return success, message

    def get_weights_by_name(self, recv_req: GetWeightsByNameReqInput):
        parameter = self.model_runner.get_weights_by_name(
            recv_req.name, recv_req.truncate_size
        )
        return parameter
