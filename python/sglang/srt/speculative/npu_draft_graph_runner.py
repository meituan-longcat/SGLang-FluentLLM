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
"""Run the model with npu graph and torch.compile"""

from __future__ import annotations

import bisect
import inspect
import os
import threading
from typing import TYPE_CHECKING

import torch
import tqdm
from sglang.srt.env import ENV

from sglang.srt.distributed import get_tensor_model_parallel_rank
from sglang.srt.layers.attention.npu_mla_backend import get_attn_meta_npu
from sglang.srt.distributed import get_ep_group
from sglang.srt.npu.utils import npu_super_kernel
from sglang.srt.env import global_server_args_dict
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.speculative.eagle_utils import EagleDraftInput
from sglang.srt.utils import (
    get_available_gpu_memory,
    get_compiler_backend,
    rank0_log,
)

from sglang.srt.model_executor.cuda_graph_runner import get_batch_sizes_to_capture

if TYPE_CHECKING:
    from sglang.srt.speculative.eagle_worker import EAGLEWorker

class NpuDraftGraphRunner:
    def __init__(self, eagle_worker: EAGLEWorker, draft_model_idx=0):
        # Parse args
        if draft_model_idx == 0:
            self.model_runner = model_runner = eagle_worker.model_runner
        else:
            self.model_runner = model_runner = eagle_worker.model_runner_list[draft_model_idx]
        self.graphs = {}
        self.output_buffers = {}
        self.tp_size = self.model_runner.tp_size
        self.dp_size = model_runner.server_args.dp_size
        self.speculative_num_steps = model_runner.server_args.speculative_num_steps
        self.topk = model_runner.server_args.speculative_eagle_topk
        self.capture_bs, self.compile_bs = get_batch_sizes_to_capture(model_runner)
        rank0_log(f"EAGLEDraftExtendNpuGraphRunner: capture bs {self.capture_bs}, compile bs {self.compile_bs}")
        self.padded_static_len = -1
        self.capture()

    @torch.inference_mode()
    def replay(self, forward_batch: ForwardBatch):
        with npu_super_kernel('mtp1', 'stream-fusion=1', flag=forward_batch.can_run_with_graph):
            out = self.model_runner.model.compile_forward(
                forward_batch.input_ids,
                forward_batch.positions,
                forward_batch
            )
        fix_kvp_current_kv_cache = getattr(self.model_runner.model, "fix_kvp_current_kv_cache", None)
        if fix_kvp_current_kv_cache:
            self.model_runner.model.fix_kvp_current_kv_cache(forward_batch)
        return out

    def capture(self):
        self.model_runner.model.compile_forward = torch.compile(
            self.model_runner.model.forward,
            fullgraph=True,
            dynamic=False,
            backend=get_compiler_backend(),
        )
        compile_range = (
            tqdm.tqdm(list(reversed(self.compile_bs)))
            if get_tensor_model_parallel_rank() == 0
            else reversed(self.compile_bs)
        )
        for bs in compile_range:
            if get_tensor_model_parallel_rank() == 0:
                avail_mem = get_available_gpu_memory(
                    self.model_runner.device,
                    self.model_runner.gpu_id,
                    empty_cache=False,
                )
                compile_range.set_description(
                    f"Capturing MTP batch {bs} ({avail_mem=:.2f} GB)"
                )

            num_tokens = bs * self.model_runner.server_args.speculative_num_draft_tokens
            # for mtp1 warm up
            forward_batch_mtp_1 = self.prepare_forward_batch(bs, num_tokens, ForwardMode.DRAFT_EXTEND)
            with torch.inference_mode():
                forward_batch_mtp_1.attn_metadata = get_attn_meta_npu(forward_batch_mtp_1, self.model_runner, True)

            # for mtpn warm up
            if self.model_runner.server_args.speculative_num_draft_tokens > 2:
                forward_batch_mtp_n = self.prepare_forward_batch(bs, bs, ForwardMode.DECODE)
                with torch.inference_mode():
                    forward_batch_mtp_n.attn_metadata = get_attn_meta_npu(forward_batch_mtp_n, self.model_runner, True)

            @torch.inference_mode()
            def run_once(forward_batch):
                with npu_super_kernel('mtp1', 'stream-fusion=1', flag=forward_batch.can_run_with_graph):
                    ret = (
                        self.model_runner.model.compile_forward(
                            forward_batch.input_ids,
                            forward_batch.positions,
                            forward_batch
                        )
                    )
                    return ret

            for _ in range(4):
               torch.npu.synchronize()
               self.model_runner.tp_group.barrier()
               run_once(forward_batch_mtp_1)
               # for mtpn warm up
               if self.model_runner.server_args.speculative_num_draft_tokens > 2:
                   run_once(forward_batch_mtp_n)
        return

    def can_run(self, forward_batch: ForwardBatch):
        can =  bool(
            forward_batch.all_decode_or_idle
            and (forward_batch.forward_mode == ForwardMode.DRAFT_EXTEND or forward_batch.forward_mode.is_idle() or forward_batch.forward_mode.is_decode())
            and forward_batch.batch_size  <= max(self.compile_bs)
        )
        forward_batch.can_run_with_graph = can
        return can

    def prepare_forward_batch(self, bs: int, num_tokens: int, forward_mode) -> ForwardBatch:
        # Graph inputs
        with torch.device(self.model_runner.device):
            if forward_mode == ForwardMode.DECODE:
                with torch.inference_mode():
                    input_ids = torch.zeros((num_tokens,), dtype=torch.int64)
                    positions = torch.zeros((num_tokens,), dtype=torch.int32)
            else:
                input_ids = torch.zeros((num_tokens,), dtype=torch.int64)
                positions = torch.zeros((num_tokens,), dtype=torch.int32)

            req_pool_indices = torch.zeros((bs,), dtype=torch.int64)
            seq_lens = torch.full((bs,), 1, dtype=torch.int32)
            out_cache_loc = torch.zeros((num_tokens,), dtype=torch.int32)

            new_tokens_to_compute = torch.full((bs,), 2, device="npu", dtype=torch.int32)
            extend_seq_lens = torch.full((bs,), 0, device="npu", dtype=torch.int64)
            buffer_len = bs * get_ep_group().world_size
            gathered_buffer = torch.zeros(
                (buffer_len, self.model_runner.model_config.hidden_size),
                dtype=self.model_runner.dtype,
                device=self.model_runner.device,
            )

        extend_logprob_start_lens_cpu = [0] * bs

        spec_info = EagleDraftInput(
            topk_p=None,
            topk_index=None,
            hidden_states=None,
            capture_hidden_mode=CaptureHiddenMode.FULL
        )

        spec_info.accept_index = torch.ones(num_tokens, dtype=torch.int64, device=self.model_runner.device)
        with torch.inference_mode():
            eagle_hidden_size = self.model_runner.model_config.hidden_size
            if self.model_runner.spec_algorithm.is_eagle3():
                eagle_hidden_size = self.model_runner.model_config.hidden_size * 3
            spec_info.hidden_states = torch.zeros((num_tokens, eagle_hidden_size), dtype=torch.bfloat16, device=self.model_runner.device)

        if self.model_runner.model_config.use_over_embedding:
            self.oe_column_starts=torch.zeros(bs, dtype=torch.int32, device="npu")
            self.oe_req_lens=torch.zeros(bs, dtype=torch.int32, device="npu")
            self.oe_out_column_starts=torch.zeros(bs, dtype=torch.int32, device="npu")
            self.oe_out_req_lens=torch.zeros(bs, dtype=torch.int32, device="npu")

        global_num_tokens = [bs] * self.tp_size
        forward_batch = ForwardBatch(
            global_num_tokens = global_num_tokens,
            forward_mode=forward_mode,
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
            spec_algorithm=self.model_runner.spec_algorithm,
            spec_info=spec_info,
            capture_hidden_mode=CaptureHiddenMode.LAST,
            new_tokens_to_compute = new_tokens_to_compute,
            gathered_buffer=gathered_buffer,
            extend_seq_lens = extend_seq_lens,
            extend_logprob_start_lens_cpu = extend_logprob_start_lens_cpu,
            all_decode_or_idle=True,
            can_run_with_graph=True,
            can_run_all2all=ENV.npu_enable_all2all_comm,
            oe_token_table=self.model_runner.oe_token_table if self.model_runner.model_config.use_over_embedding else None,
            oe_column_starts=self.oe_column_starts  if self.model_runner.model_config.use_over_embedding else None,
            oe_req_lens=self.oe_req_lens  if self.model_runner.model_config.use_over_embedding else None,
            oe_out_column_starts=self.oe_out_column_starts  if self.model_runner.model_config.use_over_embedding else None,
            oe_out_req_lens=self.oe_out_req_lens if self.model_runner.model_config.use_over_embedding else None,
        )
        forward_batch.topk_indices = None
        return forward_batch
