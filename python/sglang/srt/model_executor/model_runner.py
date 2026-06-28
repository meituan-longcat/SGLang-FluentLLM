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
"""ModelRunner runs the forward passes of the models."""

import datetime
from typing import Optional

from sglang.srt.model_executor.attn_initializer import AttnInitializer
from sglang.srt.model_executor.weight_mixin import WeightMixin
from sglang.srt.distributed.parallel_state import get_world_group, get_pp_group
from sglang.srt.utils import get_colorful_logger, monkey_patch_p2p_access_check
import os
import time
from typing import List

import torch
import torch.distributed as dist

from sglang.srt.env import ENV
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.distributed import (
    get_tp_group,
    init_distributed_environment,
    initialize_model_parallel,
    set_custom_all_reduce
)
from sglang.srt.layers.dp_attention import (
    get_attention_tp_group,
    initialize_dp_attention,
    initialize_dp_dense,
    get_attention_dp_rank,
    initialize_dp_draft_model,
)
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.sampler import Sampler
from sglang.srt.env import global_server_args_dict, global_server_args_dict_update, ENV

from sglang.srt.model_executor.cuda_graph_runner import CudaGraphRunner
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.managers.expert_distribution import (
    get_global_expert_distribution_recorder,
)
from sglang.srt.model_executor.eplb_mixin import EPLBMixin
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.torch_memory_saver_adapter import TorchMemorySaverAdapter
from sglang.srt.utils import (
    enable_show_time_cost,
    get_available_gpu_memory,
    is_npu,
    set_cpu_offload_max_bytes,
)

__is_npu__ = is_npu()
if __is_npu__:
    from sglang.srt.layers.attention.npu_mla_backend import get_attn_meta_npu
    from sglang.srt.model_executor.npu_graph_runner import NpuGraphRunner
else:
    from sglang.srt.model_executor.prefill_cuda_graph_runner import PrefillCudaGraphRunner

logger = get_colorful_logger(__name__)
from sglang.srt.oe_utils import update_token_table

UNBALANCED_MODEL_LOADING_TIMEOUT_S = os.getenv("UNBALANCED_MODEL_LOADING_TIMEOUT_S", 300)


class ModelRunner(EPLBMixin, WeightMixin):
    """ModelRunner runs the forward passes of the models."""

    def __init__(
        self,
        model_config: ModelConfig,
        mem_fraction_static: float,
        gpu_id: int,
        attn_tp_rank: int,
        attn_tp_size: int,
        world_size: int,
        moe_ep_rank: int,
        moe_ep_size: int,
        global_rank: int,
        nccl_port: int,
        server_args: ServerArgs,
        is_draft_worker: bool = False,
        req_to_token_pool=None,
        kv_allocator=None,
        oe_token_table=None,
        enable_overlap: bool = False,
        draft_model_idx: Optional[int] = None
    ):
        # Parse args
        self.model_config = model_config
        self.mem_fraction_static = mem_fraction_static
        self.device = server_args.device
        self.gpu_id = gpu_id
        self.pp_rank = global_rank // server_args.tp_size
        self.pp_size = server_args.pp_size
        self.attn_tp_rank = attn_tp_rank
        self.attn_tp_size = attn_tp_size
        self.tp_rank = global_rank % server_args.tp_size
        self.tp_size = server_args.tp_size
        self.world_size = world_size
        self.moe_ep_rank = moe_ep_rank
        self.moe_ep_size = moe_ep_size
        self.global_rank = global_rank
        self.dist_port = nccl_port
        self.server_args = server_args
        self.is_draft_worker = is_draft_worker
        self.draft_model_idx = draft_model_idx
        self.is_generation = model_config.is_generation
        self.is_multimodal = model_config.is_multimodal
        self.should_log = global_rank == 0
        self.spec_algorithm = SpeculativeAlgorithm.from_string(server_args.speculative_algorithm)
        self.is_hybrid = False
        self.req_to_token_pool = req_to_token_pool
        self.kv_allocator = kv_allocator
        self.forward_pass_id = 0
        self.max_running_requests = server_args.max_running_requests
        self.oe_token_table = oe_token_table
        self.eagle3_layers_to_capture = server_args.eagle3_layers_to_capture

        self.device_graph_runner = None
        self.spec_num_steps = self.server_args.speculative_num_steps

        # Global vars
        if server_args.show_time_cost:
            enable_show_time_cost()
        if server_args.disable_outlines_disk_cache:
            from outlines.caching import disable_cache
            disable_cache()

        AttnInitializer.modify_args(self)
        global_server_args_dict_update(server_args)

        set_cpu_offload_max_bytes(int(server_args.cpu_offload_gb * 1024**3))
        # Get memory before model loading
        min_per_gpu_memory = self.init_torch_distributed()
        self.memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=self.server_args.enable_memory_saver
        )

        self.setup_eplb()

        self.sampler = Sampler()

        AttnInitializer.modify_args(self)
        global_server_args_dict_update(server_args)
        self.load_model(draft_model_idx=draft_model_idx)
        # Handle the case where some of models don't finish loading.
        try:
            dist.monitored_barrier(
                group=get_tp_group().cpu_group,
                timeout=datetime.timedelta(seconds=UNBALANCED_MODEL_LOADING_TIMEOUT_S),
                wait_all_ranks=True,
            )
        except RuntimeError:
            raise ValueError(
                f"TP rank {self.attn_tp_rank} could finish the model loading, but there are other ranks that didn't finish loading. It is likely due to unexpected failures (e.g., OOM) or a slow node."
            ) from None

        # Init memory pool and attention backends
        AttnInitializer.init_memory_pool(
            self,
            min_per_gpu_memory,
            server_args.max_running_requests,
            server_args.max_total_tokens,
            server_args.page_size
        )

        self.use_over_embedding = self.model_config.use_over_embedding
        if self.use_over_embedding and self.oe_token_table is None:
            self.oe_token_table = torch.empty(self.req_to_token_pool.size, self.model_config.context_len,
                                              dtype=torch.int32, device=server_args.device)
        AttnInitializer.init_attention_backend(self)
        if self.device == "cuda":
            self.init_cublas()
            self.init_cuda_graphs()
        else:
            self.prefill_cuda_graph_runner = None

        if self.spec_algorithm.is_eagle3() and not self.is_draft_worker:
            self.model.set_eagle3_layers_to_capture(self.eagle3_layers_to_capture)

    def init_torch_distributed(self):
        logger.info(f"Init torch distributed begin. Avail mem={get_available_gpu_memory(self.device, self.gpu_id):.4f} GB")

        torch.get_device_module(self.device).set_device(self.gpu_id)
        if self.device == "cuda":
            backend = "nccl"
        elif self.device == "npu":
            backend = "hccl"

        if not self.server_args.enable_p2p_check and self.device != "npu":
            monkey_patch_p2p_access_check()

        if self.server_args.dist_init_addr:
            dist_init_method = f"tcp://{self.server_args.dist_init_addr}"
        else:
            dist_init_method = f"tcp://127.0.0.1:{self.dist_port}"
        set_custom_all_reduce(not self.server_args.disable_custom_all_reduce)

        if not self.is_draft_worker:
            # Only initialize the distributed environment on the target model worker.
            init_distributed_environment(
                backend=backend,
                world_size=self.server_args.world_size,
                rank=self.global_rank,
                local_rank=self.gpu_id,
                distributed_init_method=dist_init_method,
                timeout=self.server_args.dist_timeout,
                num_nodes=self.server_args.nnodes
            )

            # Currently PP is not supported, for non-attention parts the communication group is world regardless of layout
            initialize_model_parallel(
                tensor_model_parallel_size=self.tp_size,
                expert_model_parallel_size=self.moe_ep_size,
                pipeline_model_parallel_size=self.pp_size,
                mlp_tensor_model_parallel_size = self.server_args.dense_tp_size,
                attn_tp_size=self.attn_tp_size,
            )

            # Establish communication groups related to attention, PP not considered for now
            max_num_tokens = self.server_args.chunked_prefill_size \
                if self.server_args.chunked_prefill_size > 0 \
                else self.server_args.max_prefill_tokens + self.server_args.context_length
            if self.server_args.enable_mla_l1_5_cache:
                max_num_tokens = max(self.server_args.mla_max_chunk_capacity, max_num_tokens)

            initialize_dp_attention(
                attn_tp_rank=self.attn_tp_rank,
                attn_tp_size=self.attn_tp_size,
                dp_size=self.server_args.dp_size,
                dp_rank=self.tp_rank // self.attn_tp_size,
                global_rank=self.global_rank,
                local_rank=self.global_rank % self.server_args.nprocs_per_node,
                hidden_size=self.model_config.hidden_size,
                max_num_tokens=max_num_tokens,
                force_deterministic_rsag=global_server_args_dict["force_deterministic_rsag"]
            )

            # Establish communication groups related to dense
            self.dense_tp_size=self.server_args.dense_tp_size
            self.dense_dp_size=self.world_size // self.dense_tp_size

            self.dense_tp_rank=self.global_rank % self.dense_tp_size
            self.dense_dp_rank=self.global_rank // self.dense_tp_size

            initialize_dp_dense(
                dense_tp_rank=self.dense_tp_rank,
                dense_tp_size=self.dense_tp_size,
                dense_dp_size=self.dense_dp_size,
                dense_dp_rank=self.dense_dp_rank,
                max_num_tokens=max_num_tokens,
                hidden_size=self.model_config.hidden_size,
                local_rank=self.global_rank % self.server_args.nprocs_per_node,
            )

            if __is_npu__:
                initialize_dp_draft_model()

        self.tp_group = get_tp_group()
        self.pp_group = get_pp_group()
        self.attention_tp_group = get_attention_tp_group()

        tp_rank = torch.distributed.get_rank(group=self.tp_group.device_group)
        attn_tp_rank = torch.distributed.get_rank(group=self.attention_tp_group.device_group)
        dp_rank = get_attention_dp_rank()

        logger.info(f"Init comm buff end. Avail mem={get_available_gpu_memory(self.device, self.gpu_id):.4f} GB")

        logger.info(
            f"Current Process distributed state : \n \
            global rank: {self.global_rank} tp_rank: {tp_rank} attn_tp_rank: {attn_tp_rank} dp_rank: {dp_rank}"
        )

        # If distributed, all_reduce available GPU memory across all GPUs
        min_per_gpu_memory = get_available_gpu_memory(
            self.device,
            self.gpu_id,
            distributed=self.tp_size > 1,
            cpu_group=get_world_group().cpu_group,
        )

        # Check memory for tensor parallelism
        if self.pp_size==1:
            if self.tp_size>1:
                local_gpu_memory=get_available_gpu_memory(self.device, self.gpu_id)
                if min_per_gpu_memory<local_gpu_memory*0.9:
                    raise ValueError(
                        "The memory capacity is unbalanced. Some GPUs may be occupied by other processes."
                    )
        else:
            local_gpu_memory=get_available_gpu_memory(self.device, self.gpu_id)
            logger.info(f"{local_gpu_memory=} {min_per_gpu_memory=}")

        return min_per_gpu_memory

    def init_cublas(self):
        """We need to run a small matmul to init cublas. Otherwise, it will raise some errors later."""
        dtype = torch.float16
        device = "cuda"
        a = torch.ones((16, 16), dtype=dtype, device=device)
        b = torch.ones((16, 16), dtype=dtype, device=device)
        c = a @ b
        return c

    def init_cuda_graphs(self):
        """Capture cuda graphs."""
        self.device_graph_runner = None
        self.prefill_cuda_graph_runner = None

        if not self.is_generation:
            # TODO: Currently, cuda graph only captures decode steps, which only exists for generation models
            return

        if self.server_args.disable_cuda_graph:
            return

        if getattr(self.model_config.hf_config, "use_nsa", False):
            self.model.init_cuda_graph_state_nsa(
                self.max_running_requests, self.max_total_num_tokens,
                self.model_config.context_len
            )

        if not self.spec_algorithm.is_none():
            # Outside
            return

        tic = time.time()
        if not self.server_args.disable_prefill_graph:
            before_capture_available_gpu_memory = get_available_gpu_memory(self.device, self.gpu_id)
            logger.info(
                "Capture Prefill cuda graph begin. This can take up to several minutes. "
                f"avail mem={before_capture_available_gpu_memory:.2f} GB in model runner!"
            )
            self.prefill_cuda_graph_runner = PrefillCudaGraphRunner(self)
            after_capture_available_gpu_memory = get_available_gpu_memory(self.device, self.gpu_id)
            logger.info(
                f"Capture Prefill cuda graph end. Time elapsed: {time.time() - tic:.2f} s. "
                f"avail mem={after_capture_available_gpu_memory:.2f} GB"
            )
            logger.info(
                f"{len(self.prefill_cuda_graph_runner.graphs)} graphs used "
                f"mem={(before_capture_available_gpu_memory - after_capture_available_gpu_memory):.2f} GB"
            )

        tic = time.time()
        before_capture_available_gpu_memory = get_available_gpu_memory(self.device, self.gpu_id)
        logger.info(
            "Capture cuda graph begin. This can take up to several minutes. "
            f"avail mem={before_capture_available_gpu_memory:.2f} GB in model runner!"
        )
        self.device_graph_runner = CudaGraphRunner(self)
        after_capture_available_gpu_memory = get_available_gpu_memory(self.device, self.gpu_id)
        logger.info(
            f"Capture cuda graph end. Time elapsed: {time.time() - tic:.2f} s. "
            f"avail mem={after_capture_available_gpu_memory:.2f} GB"
        )
        logger.info(
            f"{len(self.device_graph_runner.graphs)} graphs used "
            f"mem={(before_capture_available_gpu_memory - after_capture_available_gpu_memory):.2f} GB"
        )

    def init_npu_graphs(self):
        """Enable torch.compile and graph engine with Npu."""
        self.device_graph_runner = None

        if not self.is_generation:
            # TODO: Currently, npu graph only captures decode steps, which only exists for generation models
            return
        if self.is_draft_worker:
            logger.info(f"Draft worker skip init npu graphs.")
            return
        if not ENV.npu_enable_graph:
            return

        tic = time.perf_counter()
        before_mem = get_available_gpu_memory(self.device, self.gpu_id)
        logger.info(
            f"Compile graph with npu begin. This can take up to several minutes. avail mem={before_mem:.2f} GB"
        )
        self.device_graph_runner = NpuGraphRunner(self)
        after_mem = get_available_gpu_memory(self.device, self.gpu_id)
        logger.info(
            f"Compile graph end. Time elapsed: {time.perf_counter() - tic:.2f} s. "
            f"mem usage={(before_mem - after_mem):.2f} GB. avail mem={after_mem:.2f} GB."
        )

    def forward_decode(self, forward_batch: ForwardBatch):
        self.attn_backend.init_forward_metadata(forward_batch)

        if forward_batch.can_run_tbo:
            micro_batches = forward_batch.split_micro_batch()
            return self.model.forward(
                forward_batch.input_ids, forward_batch.positions, forward_batch, micro_batches=micro_batches
            )
        else:
            return self.model.forward(forward_batch.input_ids, forward_batch.positions, forward_batch)

    def forward_extend(self, forward_batch: ForwardBatch, skip_metadata_init: bool = False):
        if not skip_metadata_init:
            self.attn_backend.init_forward_metadata(forward_batch)

        kwargs = {}
        if forward_batch.input_embeds is not None:
            kwargs["input_embeds"] = forward_batch.input_embeds
        if not self.is_generation:
            kwargs["get_embedding"] = True
        if forward_batch.can_run_tbo:
            micro_batches = forward_batch.split_micro_batch()
            kwargs["micro_batches"] = micro_batches

        return self.model.forward(
            forward_batch.input_ids,
            forward_batch.positions,
            forward_batch,
            **kwargs,
        )

    def forward_idle(self, forward_batch: ForwardBatch):
        return self.model.forward(
            forward_batch.input_ids, forward_batch.positions, forward_batch
        )
    @torch.inference_mode()
    def forward(self, forward_batch: ForwardBatch) -> LogitsProcessorOutput:
        self.forward_pass_id += 1

        with get_global_expert_distribution_recorder().with_forward_pass(
            self.forward_pass_id,
            forward_batch,
        ):
            output = self._forward_raw(forward_batch)

        if self.eplb_manager is not None:
            self.eplb_manager.on_forward_pass_end()

        return output

    def _forward_raw(self, forward_batch: ForwardBatch) -> LogitsProcessorOutput:
        can_run_graph = self.device_graph_runner and self.device_graph_runner.can_run_graph(forward_batch)
        if __is_npu__:
            if can_run_graph:
                if self.server_args.speculative_num_draft_tokens == 1: # non mtp
                    compile_bs = self.device_graph_runner.compile_bs
                    if global_server_args_dict["npu_disable_all_gather"]:
                        padding_size = max(compile_bs)
                    else:
                        max_input_bs = max(forward_batch.global_num_tokens)
                        max_compile_bs = max(compile_bs)
                        assert max_input_bs <= max_compile_bs, f"max input batch ({max_input_bs}) should less equal than max compile bs({max_compile_bs}) in graph mode"
                        padding_size = min(bs for bs in compile_bs if bs >= max_input_bs)

                    forward_batch.padding_for_npu_graph(self, padding_size)
                    forward_batch.can_run_all2all = ENV.npu_enable_all2all_comm

                forward_batch.can_run_with_graph = True
                forward_batch.attn_metadata = get_attn_meta_npu(forward_batch, self)
                bs = getattr(forward_batch, "ori_bs", forward_batch.batch_size)
                with self.device_graph_runner.get_runner_context(
                    forward_batch
                ) as runner_fn:
                    if bs >= 1:
                        logger.debug(f"Forward with graph, bs {bs} padding to {forward_batch.batch_size}. ")
                    ret = runner_fn()
                    fix_kvp_current_kv_cache = getattr(self.device_graph_runner.model_runner.model, "fix_kvp_current_kv_cache", None)
                    if fix_kvp_current_kv_cache:
                        self.device_graph_runner.model_runner.model.fix_kvp_current_kv_cache(forward_batch)

                if self.server_args.speculative_num_draft_tokens == 1: # unpadding in non mtp
                    logits_output = LogitsProcessorOutput(
                        next_token_logits=ret.next_token_logits[:bs],
                        hidden_states=(
                            ret.hidden_states[:bs] if ret.hidden_states is not None else None
                        ),
                    )
                    forward_batch.batch_size = bs
                    return logits_output
                return ret
            else:
                if not forward_batch.forward_mode.is_idle():
                    forward_batch.attn_metadata = get_attn_meta_npu(forward_batch, self)
        elif forward_batch.forward_mode.is_cuda_graph() and can_run_graph:
            return self.device_graph_runner.replay(forward_batch)

        if (
            not __is_npu__
            and forward_batch.forward_mode == ForwardMode.EXTEND
            and self.prefill_cuda_graph_runner
            and self.prefill_cuda_graph_runner.can_run(forward_batch)
        ):
            return self.prefill_cuda_graph_runner.replay(forward_batch)

        if forward_batch.forward_mode.is_decode():
            return self.forward_decode(forward_batch)
        elif forward_batch.forward_mode.is_extend():
            return self.forward_extend(forward_batch)
        elif forward_batch.forward_mode.is_idle():
            return self.forward_idle(forward_batch)
        else:
            raise ValueError(f"Invalid forward mode: {forward_batch.forward_mode}")

    def _preprocess_logits(
        self, logits_output: LogitsProcessorOutput, sampling_info: SamplingBatchInfo
    ):
        # Apply logit bias
        if sampling_info.sampling_info_done:
            # Overlap mode: the function update_regex_vocab_mask was executed
            # in process_batch_result of the last batch.
            if sampling_info.grammars:
                sampling_info.sampling_info_done.wait()
        else:
            # Normal mode: Put CPU-heavy tasks here. They will be overlapped with the forward pass.
            sampling_info.update_regex_vocab_mask()
        sampling_info.apply_logits_bias(logits_output.next_token_logits)

    def update_output_logprobs(
        self,
        logits_output: LogitsProcessorOutput,
        sampling_info: SamplingBatchInfo,
        top_logprobs_nums: List[int],
        token_ids_logprobs: List[int],
        next_token_ids: torch.Tensor,
        *,
        num_tokens_per_req: List[int],
    ):
        """Update the logits_output's output logprob based on next_token_ids

        Args:
            logits_output: The logits output from the model forward
            sampling_info: Sampling info for logprob calculation
            top_logprobs_nums: Number of logprobs per request.
            next_token_ids: Next token ids.
            num_tokens_per_req: The number of tokens per request.

        Returns:
            A list of next_token_ids
        """
        self._preprocess_logits(logits_output, sampling_info)
        # We should repeat top_logprobs_nums to match num_tokens_per_req.
        top_logprobs_nums_repeat_interleaved = []
        token_ids_logprobs_repeat_interleaved = []
        for num, num_tokens in zip(top_logprobs_nums, num_tokens_per_req):
            top_logprobs_nums_repeat_interleaved.extend([num] * num_tokens)
        for token_ids, num_tokens in zip(token_ids_logprobs, num_tokens_per_req):
            token_ids_logprobs_repeat_interleaved.extend([token_ids] * num_tokens)
        self.sampler(
            logits_output,
            sampling_info,
            True,
            top_logprobs_nums_repeat_interleaved,
            token_ids_logprobs_repeat_interleaved,
            batch_next_token_ids=next_token_ids,
        )

    def sample(
        self,
        logits_output: LogitsProcessorOutput,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        """Sample and compute logprobs and update logits_output.

        Args:
            logits_output: The logits output from the model forward
            forward_batch: The forward batch that generates logits_output

        Returns:
            A list of next_token_ids
        """
        # For duplex models with multiple output streams.
        if isinstance(logits_output, tuple):
            return torch.stack(
                [self.sample(values, forward_batch) for values in logits_output],
                axis=-1,
            )

        self._preprocess_logits(logits_output, forward_batch.sampling_info)

        # Sample the next tokens
        next_token_ids = self.sampler(
            logits_output,
            forward_batch.sampling_info,
            forward_batch.return_logprob,
            forward_batch.top_logprobs_nums,
            forward_batch.token_ids_logprobs,
        )

        if self.use_over_embedding:
            # Update token_table start should be seq_len+1 here
            if not __is_npu__:
                forward_batch.oe_out_column_starts[:forward_batch.batch_size]=forward_batch.seq_lens
                forward_batch.oe_out_req_lens[:forward_batch.batch_size]=1
                update_token_table(oe_token_table=forward_batch.oe_token_table,
                                   tokens=next_token_ids,
                                   row_indices=forward_batch.req_pool_indices,
                                   column_starts=forward_batch.oe_out_column_starts,
                                   oe_req_lens=torch.ones_like(next_token_ids),
                                   )
            else:
                req_pool_indices=forward_batch.req_pool_indices[:forward_batch.batch_size]
                oe_out_column_starts=forward_batch.seq_lens[:forward_batch.batch_size]
                oe_out_req_lens=torch.ones((forward_batch.batch_size,), device=next_token_ids.device, dtype=torch.int32)
                update_token_table(oe_token_table=forward_batch.oe_token_table,
                                   tokens=next_token_ids,
                                   row_indices=req_pool_indices,
                                   column_starts=oe_out_column_starts,
                                   oe_req_lens=oe_out_req_lens,
                                   )
        return next_token_ids

    @property
    def model_is_mrope(self) -> bool:
        """Detect if the model has "mrope" rope_scaling type.
        mrope requires keep "rope_deltas" between prompt and decoding phases."""
        rope_scaling = getattr(self.model_config.hf_config, "rope_scaling", {})
        if rope_scaling is None:
            return False
        return rope_scaling.get("type", None) == "mrope"

    @property
    def is_hybrid_gdn(self):
        return self.model_config.hf_config.architectures[0] in [
            "Qwen3NextForCausalLM",
            "Qwen3NextForCausalLMNextN",
        ]

    @property
    def is_kimi_linear(self):
        return self.model_config.hf_config.architectures[0] in [
            "KimiLinearForCausalLM",
        ]

    @property
    def mambaish_config(self):
        return self.is_hybrid_gdn or self.is_kimi_linear
