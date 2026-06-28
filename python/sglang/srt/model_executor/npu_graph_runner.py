# Copyright 2025 SGLang Team
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
"""Run the model with npu graph engine and torch.compile."""

from __future__ import annotations

import bisect
import inspect
import os
import types
from contextlib import contextmanager
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Generator,
)

import torch
import tqdm

from sglang.srt.env import ENV
from sglang.srt.speculative.eagle_utils import EagleVerifyInput
from sglang.srt.distributed import get_tensor_model_parallel_rank
from sglang.srt.layers.attention.npu_mla_backend import get_attn_meta_npu
from sglang.srt.model_executor.cuda_graph_runner import DeviceRunnerBase
from sglang.srt.distributed import get_ep_group
from sglang.srt.env import global_server_args_dict

from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
)
from sglang.srt.utils import (
    get_available_gpu_memory,
    get_compiler_backend,
    rank0_log,
)

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner


class NpuGraphRunner(DeviceRunnerBase):
    """A NpuGraphRunner runs the forward pass of a model with npu graph engine and torch.compile."""

    def __init__(self, model_runner: ModelRunner):
        self.enable_cache = global_server_args_dict['npu_enable_graph_cache']
        self.graph_cache_dir = global_server_args_dict['npu_graph_cache_path']
        self.backend=get_compiler_backend()
        super().__init__(model_runner)

    def warm_up(self):
        if not self.enable_cache:
            self.model_runner.model.compile_forward = torch.compile(
                self.model_runner.model.forward,
                fullgraph=True,
                dynamic=False,
                backend=self.backend,
            )
        # self.model_runner.model.compile_forward = self.model_runner.model.forward
        self.run_fake()


    def build_forward_batch(self, bs):
        # construct warm up forward batch
        num_tokens = bs * self.model_runner.server_args.speculative_num_draft_tokens

        with torch.device(self.model_runner.device):
            input_ids = torch.zeros((num_tokens,), dtype=torch.int64)
            req_pool_indices = torch.zeros((bs,), dtype=torch.int64)
            seq_lens = torch.ones((bs,), dtype=torch.int32)
            positions = torch.zeros((num_tokens,), dtype=torch.int32)
            extend_seq_lens = None if self.model_runner.server_args.speculative_num_draft_tokens == 1 else \
                torch.full((bs,), 0, dtype=torch.int32)
            out_cache_loc = torch.zeros((num_tokens,), dtype=torch.int32)
            new_tokens_to_compute = torch.full((bs,), self.model_runner.server_args.speculative_num_draft_tokens, dtype=torch.int32)
            if self.model_runner.model_config.use_over_embedding:
                oe_column_starts=torch.zeros(bs, dtype=torch.int32, device="npu")
                oe_req_lens=torch.zeros(bs, dtype=torch.int32, device="npu")
                oe_out_column_starts=torch.zeros(bs, dtype=torch.int32, device="npu")
                oe_out_req_lens=torch.zeros(bs, dtype=torch.int32, device="npu")

        extend_logprob_start_lens_cpu = None if self.model_runner.server_args.speculative_num_draft_tokens == 1 else [0] * bs

        sync_group_size = get_ep_group().world_size
        global_num_tokens = [bs] * sync_group_size
        buffer_len = bs * sync_group_size
        gathered_buffer = torch.zeros(
            (buffer_len, self.model_runner.model_config.hidden_size),
            dtype=self.model_runner.dtype,
            device=self.model_runner.device,
        )

        spec_info = None if self.model_runner.server_args.speculative_num_draft_tokens == 1 else self.get_spec_info(bs)
        if self.capture_hidden_mode != CaptureHiddenMode.FULL:
            self.capture_hidden_mode = (
                spec_info.capture_hidden_mode if spec_info else CaptureHiddenMode.NULL
            )

        forward_batch = ForwardBatch(
            forward_mode=self.capture_forward_mode,
            batch_size=bs,
            input_ids=input_ids,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            req_to_token_pool=self.model_runner.req_to_token_pool,
            token_to_kv_pool=self.model_runner.token_to_kv_pool,
            attn_backend=self.model_runner.attn_backend,
            spec_num_steps = self.model_runner.server_args.speculative_num_steps,
            out_cache_loc=out_cache_loc,
            seq_lens_sum=seq_lens.sum(),
            return_logprob=False,
            positions=positions,
            global_num_tokens=global_num_tokens,
            gathered_buffer=gathered_buffer,
            mrope_positions=None,
            spec_algorithm=self.model_runner.spec_algorithm,
            spec_info=spec_info,
            capture_hidden_mode=self.capture_hidden_mode,
            all_decode_or_idle=True,
            extend_seq_lens=extend_seq_lens,
            extend_logprob_start_lens_cpu=extend_logprob_start_lens_cpu,
            new_tokens_to_compute=new_tokens_to_compute,
            new_tokens_total=0,
            oe_token_table=self.model_runner.oe_token_table if self.model_runner.model_config.use_over_embedding else None,
            oe_column_starts=oe_column_starts if self.model_runner.model_config.use_over_embedding else None,
            oe_req_lens=oe_req_lens if self.model_runner.model_config.use_over_embedding else None,
            oe_out_column_starts=oe_out_column_starts if self.model_runner.model_config.use_over_embedding else None,
            oe_out_req_lens=oe_out_req_lens if self.model_runner.model_config.use_over_embedding else None,
        )
        with torch.inference_mode(): # fix recompile
            attn_metadata = get_attn_meta_npu(forward_batch, self.model_runner, True)
            forward_batch.attn_metadata = attn_metadata
            if self.model_runner.server_args.speculative_num_draft_tokens == 1:
                forward_batch.input_ids = torch.zeros((num_tokens,), dtype=torch.int32, device="npu")
                forward_batch.positions = torch.zeros((num_tokens,), dtype=torch.int32, device="npu")
        forward_batch.can_run_all2all = forward_batch.all_decode_or_idle and ENV.npu_enable_all2all_comm
        return forward_batch

    def run_fake(self):
        compile_range = (
            tqdm.tqdm(list(reversed(self.compile_bs)))
            if get_tensor_model_parallel_rank() == 0
            else reversed(self.compile_bs)
        )

        def build_method(method_name):
            method_code = f"""
def {method_name}(self, input_ids, positions, forward_batch, **kwargs):
    return self.forward(input_ids, positions, forward_batch, **kwargs)
            """
            exec(method_code)
            return locals()[method_name]

        for bs in compile_range:
            if get_tensor_model_parallel_rank() == 0:
                avail_mem = get_available_gpu_memory(
                    self.model_runner.device,
                    self.model_runner.gpu_id,
                    empty_cache=False,
                )
                compile_range.set_description(
                    f"Capturing batch {bs} ({avail_mem=:.2f} GB)"
                )
            forward_batch = self.build_forward_batch(bs)

            # use cache
            if self.enable_cache:
                import torchair
                method_name = f'forward_bs_{bs}'
                compile_method_name = f'compile_forward_bs_{bs}'
                setattr(self.model_runner.model, method_name, types.MethodType(build_method(method_name), self.model_runner.model))
                setattr(self.model_runner.model, compile_method_name, torchair.inference.cache_compile( \
                    getattr(self.model_runner.model, method_name), dynamic=False, ge_cache=True, cache_dir=self.graph_cache_dir, backend=self.backend))

            # Run and capture
            @torch.inference_mode()
            def run_once():
                forward_batch.can_run_with_graph = True
                compile_forward = getattr(self.model_runner.model, compile_method_name) if self.enable_cache \
                    else self.model_runner.model.compile_forward
                logits_output = (
                    compile_forward(
                        forward_batch.input_ids,
                        forward_batch.positions,
                        forward_batch
                    )
                )
                return logits_output

            for _ in range(4):
               torch.npu.synchronize()
               self.model_runner.tp_group.barrier()
               run_once()
        return

    @contextmanager
    def get_runner_context(
        self, forward_batch: "ForwardBatch"
    ) -> Generator[Callable[[], Any], Any, None]:
        @torch.inference_mode()
        def runner_fn():
            compile_method_name = f'compile_forward_bs_{forward_batch.batch_size}'
            compile_forward = getattr(self.model_runner.model, compile_method_name) if self.enable_cache \
                else self.model_runner.model.compile_forward
            return compile_forward(
                forward_batch.input_ids,
                forward_batch.positions,
                forward_batch
            )

        forward_batch.attn_backend.init_forward_metadata(forward_batch)
        yield runner_fn

    def can_run_graph(self, forward_batch: "ForwardBatch") -> bool:
        return bool(
            forward_batch.all_decode_or_idle
            and self.model_runner.device_graph_runner
            and self.model_runner.device_graph_runner.enable_torch_compile
            and (forward_batch.batch_size <= max(self.model_runner.device_graph_runner.compile_bs))
        )


    def get_spec_info(self, bs: int):
        draft_token_num = self.model_runner.server_args.speculative_num_draft_tokens
        num_tokens = bs * draft_token_num
        spec_info = EagleVerifyInput(
            draft_token=torch.zeros(num_tokens, device="npu"),
            positions=torch.zeros(num_tokens, device="npu"),
            draft_token_num=draft_token_num,
            spec_steps = self.model_runner.server_args.speculative_num_steps,
            capture_hidden_mode=CaptureHiddenMode.FULL,
            is_all_greedy=True,
        )
        return spec_info
