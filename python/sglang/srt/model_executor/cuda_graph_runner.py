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
"""Run the model with cuda graph and torch.compile."""

from __future__ import annotations

import bisect
from contextlib import contextmanager
import gc
import os
from typing import TYPE_CHECKING, Callable

import torch
import tqdm

from sglang.srt.custom_op import CustomOp
from sglang.srt.distributed import get_tensor_model_parallel_rank
from sglang.srt.distributed.parallel_state import GroupCoordinator, graph_capture
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.utils import get_available_gpu_memory, get_bool_env_var, is_hip


is_hip_ = is_hip()

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner


@contextmanager
def freeze_gc(enable_cudagraph_gc: bool):
    """
    Optimize garbage collection during CUDA graph capture.
    Clean up, then freeze all remaining objects from being included
    in future collections if GC is disabled during capture.
    """
    gc.collect()
    should_freeze = not enable_cudagraph_gc
    if should_freeze:
        gc.freeze()
    try:
        yield
    finally:
        if should_freeze:
            gc.unfreeze()
            gc.collect()


def _to_torch(model: torch.nn.Module, reverse: bool, num_tokens: int):
    for sub in model._modules.values():
        if isinstance(sub, CustomOp):
            if reverse:
                sub.leave_torch_compile()
            else:
                sub.enter_torch_compile(num_tokens=num_tokens)
        if isinstance(sub, torch.nn.Module):
            _to_torch(sub, reverse, num_tokens)


@contextmanager
def patch_model(
    model: torch.nn.Module,
    enable_compile: bool,
    num_tokens: int,
    tp_group: GroupCoordinator,
):
    """Patch the model to make it compatible with with torch.compile"""
    backup_ca_comm = None

    try:
        if enable_compile:
            _to_torch(model, reverse=False, num_tokens=num_tokens)
            backup_ca_comm = tp_group.ca_comm
            # Use custom-allreduce here.
            # We found the custom allreduce is much faster than the built-in allreduce in torch,
            # even with ENABLE_INTRA_NODE_COMM=1.
            # tp_group.ca_comm = None
            yield torch.compile(
                torch.no_grad()(model.forward),
                mode=os.environ.get(
                    "SGLANG_TORCH_COMPILE_MODE", "max-autotune-no-cudagraphs"
                ),
                dynamic=is_hip_ and get_bool_env_var("SGLANG_TORCH_DYNAMIC_SHAPE"),
            )
        else:
            yield model.forward
    finally:
        if enable_compile:
            _to_torch(model, reverse=True, num_tokens=num_tokens)
            tp_group.ca_comm = backup_ca_comm


def set_torch_compile_config():
    import torch._inductor.config

    torch._inductor.config.coordinate_descent_tuning = True
    torch._inductor.config.triton.unique_kernel_names = True
    torch._inductor.config.fx_graph_cache = True  # Experimental feature to reduce compilation times, will be on by default in future

    # FIXME: tmp workaround
    torch._dynamo.config.accumulated_cache_size_limit = 1024
    if hasattr(torch._dynamo.config, "cache_size_limit"):
        torch._dynamo.config.cache_size_limit = 1024


def get_batch_sizes_to_capture(model_runner: ModelRunner):
    server_args = model_runner.server_args
    capture_bs = server_args.cuda_graph_bs

    if capture_bs is None:
        if server_args.speculative_algorithm is None:
            if server_args.disable_cuda_graph_padding:
                capture_bs = list(range(1, 33)) + [64, 96, 128, 160]
            else:
                capture_bs = [1, 2, 4] + [i * 8 for i in range(1, 21)]
        else:
            capture_bs = list(range(1, 33))

    if is_hip_:
        capture_bs += [i * 8 for i in range(21, 33)]

    if max(capture_bs) > model_runner.req_to_token_pool.size:
        # In some case (e.g., with a small GPU or --max-running-requests), the #max-running-requests
        # is very small. We add more values here to make sure we capture the maximum bs.
        capture_bs = list(
            sorted(
                set(
                    capture_bs
                    + [model_runner.req_to_token_pool.size - 1]
                    + [model_runner.req_to_token_pool.size]
                )
            )
        )

    capture_bs = [
        bs
        for bs in capture_bs
        if bs <= model_runner.req_to_token_pool.size
        and bs <= server_args.cuda_graph_max_bs
        and bs <= server_args.max_running_requests // server_args.dp_size
    ]
    compile_bs = (
        [bs for bs in capture_bs if bs <= server_args.torch_compile_max_bs]
        if server_args.enable_torch_compile
        else []
    )
    return capture_bs, compile_bs


# Reuse this memory pool across all cuda graph runners.
global_graph_memory_pool = None


def get_global_graph_memory_pool():
    return global_graph_memory_pool


def set_global_graph_memory_pool(val):
    global global_graph_memory_pool
    global_graph_memory_pool = val


class CudaGraphRunner:
    """A CudaGraphRunner runs the forward pass of a model with cuda graph and torch.compile."""

    def __init__(self, model_runner: ModelRunner):
        # Parse args
        self.model_runner = model_runner
        self.graphs = {}
        self.output_buffers = {}
        self.enable_torch_compile = model_runner.server_args.enable_torch_compile
        self.disable_padding = model_runner.server_args.disable_cuda_graph_padding
        self.enable_dp_attention = model_runner.server_args.enable_dp_attention
        self.world_size = model_runner.server_args.world_size
        self.dp_size = model_runner.server_args.dp_size

        # Batch sizes to capture
        self.capture_bs, self.compile_bs = get_batch_sizes_to_capture(model_runner)
        self.capture_forward_mode = ForwardMode.DECODE
        self.capture_hidden_mode = CaptureHiddenMode.NULL
        self.num_tokens_per_bs = 1
        if model_runner.spec_algorithm.is_eagle():
            if self.model_runner.is_draft_worker:
                raise RuntimeError("This should not happen")
            else:
                self.capture_forward_mode = ForwardMode.TARGET_VERIFY
                self.num_tokens_per_bs = (
                    self.model_runner.server_args.speculative_num_draft_tokens
                )

        # Attention backend
        self.max_bs = max(self.capture_bs)
        self.max_num_token = self.max_bs * self.num_tokens_per_bs
        self.model_runner.attn_backend.init_cuda_graph_state(self.max_num_token)
        self.seq_len_fill_value = (
            self.model_runner.attn_backend.get_cuda_graph_seq_len_fill_value()
        )
        self.seq_lens_cpu = torch.full(
            (self.max_bs,), self.seq_len_fill_value, dtype=torch.int32
        )
        hf_config = self.model_runner.model_config.hf_config
        self.use_over_embedding = getattr(hf_config, 'use_over_embedding', False)
        if self.use_over_embedding:
            self.over_embedding_n = hf_config.oe_neighbor_num
            self.over_embedding_k = hf_config.oe_split_num

        if self.enable_torch_compile:
            set_torch_compile_config()

        # Graph inputs
        with torch.device("cuda"):
            # Here for ds v3/r1, all 0 (bos) input_ids is unreasonable, will cause
            # nan in verify calculation process, conflicts with existing operators
            self.input_ids = torch.ones((self.max_num_token,), dtype=torch.int32)
            self.token_table = model_runner.oe_token_table if self.use_over_embedding else None
            self.oe_column_starts = torch.empty([self.max_bs], dtype=torch.int32) if self.use_over_embedding else None
            self.oe_req_lens = torch.empty([self.max_bs], dtype=torch.int32) if self.use_over_embedding else None
            self.oe_out_column_starts = torch.empty([self.max_bs],
                                                    dtype=torch.int32) if self.use_over_embedding else None
            self.oe_out_req_lens = torch.empty([self.max_bs], dtype=torch.int32) if self.use_over_embedding else None
            self.req_pool_indices = torch.zeros((self.max_bs,), dtype=torch.int64)
            self.seq_lens = torch.full(
                (self.max_bs,), self.seq_len_fill_value, dtype=torch.int32
            )
            self.out_cache_loc = torch.arange(0, self.max_num_token, dtype=torch.int64)
            # During capture, kv cache is actually written, limit to padding page
            self.out_cache_loc.clamp_(min=0, max=63)
            self.positions = torch.arange(0, self.max_num_token, dtype=torch.int64)
            self.mrope_positions = torch.zeros((3, self.max_bs), dtype=torch.int64)

            # Speculative_inference
            if model_runner.spec_algorithm.is_eagle():
                self.hidden_states = torch.zeros(
                    (self.max_num_token, self.model_runner.model_config.hidden_size),
                    dtype=self.model_runner.dtype,
                )

            if self.enable_dp_attention:
                self.gathered_buffer = torch.zeros(
                    (
                        self.max_num_token * self.world_size,
                        self.model_runner.model_config.hidden_size,
                    ),
                    dtype=self.model_runner.dtype,
                )

        # Capture
        try:
            with self.model_capture_mode():
                self.capture()
        except RuntimeError as e:
            raise Exception(
                f"Capture cuda graph failed: {e}\n"
                "Possible solutions:\n"
                "1. disable cuda graph by --disable-cuda-graph\n"
                "2. set --mem-fraction-static to a smaller value (e.g., 0.8 or 0.7)\n"
                "3. disable torch compile by not using --enable-torch-compile\n"
                "4. set --cuda-graph-max-bs to a smaller value (e.g., 32)\n"
                "Open an issue on GitHub https://github.com/sgl-project/sglang/issues/new/choose \n"
            )

    @contextmanager
    def model_capture_mode(self):
        if hasattr(self.model_runner.model, "capture_mode"):
            self.model_runner.model.capture_mode = True

        yield

        if hasattr(self.model_runner.model, "capture_mode"):
            self.model_runner.model.capture_mode = False

    def can_run(self, forward_batch: ForwardBatch):
        if self.enable_dp_attention:
            min_num_tokens, max_num_tokens = min(forward_batch.global_num_tokens), max(
                forward_batch.global_num_tokens
            )
            min_bsz = int(min_num_tokens / self.num_tokens_per_bs)
            max_bsz = int(max_num_tokens / self.num_tokens_per_bs)
            is_bs_supported = forward_batch.all_decode_or_idle and (
                (min_num_tokens == max_num_tokens and max_bsz in self.graphs)
                if self.disable_padding
                else max_bsz <= self.max_bs
            )
        else:
            is_bs_supported = (
                forward_batch.batch_size in self.graphs
                if self.disable_padding
                else forward_batch.batch_size <= self.max_bs
            )

        return is_bs_supported

    def capture(self):
        # Trigger CUDA graph capture for specific shapes.
        # Capture the large shapes first so that the smaller shapes
        # can reuse the memory pool allocated for the large shapes.
        with freeze_gc(
            self.model_runner.server_args.enable_cudagraph_gc
        ), graph_capture() as graph_capture_context:
            self.stream = graph_capture_context.stream
            capture_range = (
                tqdm.tqdm(self.capture_bs)
                if get_tensor_model_parallel_rank() == 0
                else self.capture_bs
            )
            for bs in capture_range:
                if get_tensor_model_parallel_rank() == 0:
                    avail_mem = get_available_gpu_memory(
                        self.model_runner.device,
                        self.model_runner.gpu_id,
                        empty_cache=False,
                    )
                    capture_range.set_description(
                        f"Capturing batches ({avail_mem=:.2f} GB)"
                    )

                with patch_model(
                    self.model_runner.model,
                    bs in self.compile_bs,
                    num_tokens=bs * self.num_tokens_per_bs,
                    tp_group=self.model_runner.tp_group,
                ) as forward:
                    (
                        graph,
                        output_buffers,
                    ) = self.capture_one_batch_size(bs, forward)
                    self.graphs[bs] = graph
                    self.output_buffers[bs] = output_buffers

    def capture_one_batch_size(self, bs: int, forward: Callable):
        graph = torch.cuda.CUDAGraph()
        stream = self.stream
        num_tokens = bs * self.num_tokens_per_bs

        # Graph inputs
        input_ids = self.input_ids[:num_tokens]
        req_pool_indices = self.req_pool_indices[:bs]
        seq_lens = self.seq_lens[:bs]
        out_cache_loc = self.out_cache_loc[:num_tokens]
        positions = self.positions[:num_tokens]
        mrope_positions = self.mrope_positions[:, :bs]

        if self.enable_dp_attention:
            if self.model_runner.spec_algorithm.is_eagle():
                global_num_tokens = [bs * self.num_tokens_per_bs] * self.world_size
            else:
                global_num_tokens = [bs] * self.world_size
            gathered_buffer = self.gathered_buffer[: bs * self.world_size * self.num_tokens_per_bs]
        else:
            global_num_tokens = None
            gathered_buffer = None

        spec_info = self.get_spec_info(num_tokens)
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
            seq_lens_cpu=None,
            req_to_token_pool=self.model_runner.req_to_token_pool,
            token_to_kv_pool=self.model_runner.token_to_kv_pool,
            attn_backend=self.model_runner.attn_backend,
            out_cache_loc=out_cache_loc,
            seq_lens_sum=seq_lens.sum(),
            oe_token_table=self.token_table,
            oe_column_starts=self.oe_column_starts[: bs] if self.use_over_embedding else None,
            oe_req_lens=self.oe_req_lens[: bs] if self.use_over_embedding else None,
            oe_out_column_starts=self.oe_out_column_starts[: bs] if self.use_over_embedding else None,
            oe_out_req_lens=self.oe_out_req_lens[: bs] if self.use_over_embedding else None,
            return_logprob=False,
            positions=positions,
            global_num_tokens=global_num_tokens,
            gathered_buffer=gathered_buffer,
            mrope_positions=mrope_positions,
            spec_algorithm=self.model_runner.spec_algorithm,
            spec_info=spec_info,
            capture_hidden_mode=self.capture_hidden_mode,
            all_decode_or_idle=True
        )
        self.model_runner.init_request_cache_capture(
            forward_batch,
            [self.num_tokens_per_bs] * bs,
            stream,
        )

        # Attention backend
        self.model_runner.attn_backend.init_forward_metadata_capture_cuda_graph(
            bs,
            num_tokens,
            req_pool_indices,
            seq_lens,
            forward_batch.forward_mode,
            forward_batch.spec_info,
        )

        # Micro batches for TBO
        micro_batches = forward_batch.split_micro_batch()

        # Run and capture
        def run_once():
            if micro_batches:
                logits_output = forward(input_ids, forward_batch.positions, forward_batch, micro_batches)
            else:
                logits_output = forward(input_ids, forward_batch.positions, forward_batch)
            return logits_output.next_token_logits, logits_output.hidden_states

        for _ in range(4):
            torch.cuda.synchronize()
            self.model_runner.tp_group.barrier()

            run_once()

        torch.cuda.synchronize()
        self.model_runner.tp_group.barrier()

        global global_graph_memory_pool
        with torch.cuda.graph(graph, pool=global_graph_memory_pool, stream=stream):
            out = run_once()

        torch.cuda.synchronize()
        self.model_runner.tp_group.barrier()

        self.model_runner.capture_sample_one_bs(logits=out[0], out_hidden=out[1], bs=bs)

        global_graph_memory_pool = graph.pool()
        return graph, out

    def recapture_if_needed(self, forward_batch: ForwardBatch):
        # If the capture_hidden_mode changes, we need to recapture the graph
        hidden_mode_from_spec_info = getattr(
            forward_batch.spec_info, "capture_hidden_mode", CaptureHiddenMode.NULL
        )
        if (
            forward_batch.capture_hidden_mode == CaptureHiddenMode.FULL
            and self.capture_hidden_mode != CaptureHiddenMode.FULL
        ):
            self.capture_hidden_mode = CaptureHiddenMode.FULL
            self.capture()
        elif (
            forward_batch.capture_hidden_mode != CaptureHiddenMode.FULL
            and self.capture_hidden_mode != hidden_mode_from_spec_info
        ):
            self.capture_hidden_mode = hidden_mode_from_spec_info
            self.capture()

    def replay(self, forward_batch: ForwardBatch):
        self.recapture_if_needed(forward_batch)

        raw_bs = forward_batch.batch_size
        raw_num_token = raw_bs * self.num_tokens_per_bs

        if forward_batch.forward_mode.is_idle() and self.model_runner.spec_algorithm.is_eagle():
            forward_batch.spec_info = self.get_spec_info(num_tokens=0)
            forward_batch.forward_mode = ForwardMode.TARGET_VERIFY

        if self.model_runner.spec_algorithm.is_eagle():
            assert forward_batch.forward_mode.is_target_verify()

        # Pad
        if self.enable_dp_attention:
            index = bisect.bisect_left(
                self.capture_bs, max(forward_batch.global_num_tokens) / self.num_tokens_per_bs
            )
        else:
            index = bisect.bisect_left(self.capture_bs, raw_bs)
        bs = self.capture_bs[index]
        if bs != raw_bs:
            self.seq_lens.fill_(1)
            self.out_cache_loc.zero_()

        # Common inputs
        self.input_ids[:raw_num_token].copy_(forward_batch.input_ids)
        self.req_pool_indices[:raw_bs].copy_(forward_batch.req_pool_indices)
        self.seq_lens[:raw_bs].copy_(forward_batch.seq_lens)
        self.out_cache_loc[:raw_num_token].copy_(forward_batch.out_cache_loc)
        self.positions[:raw_num_token].copy_(forward_batch.positions)
        if self.use_over_embedding:
            self.oe_column_starts[:raw_bs].copy_(forward_batch.oe_column_starts)
            self.oe_req_lens[:raw_bs].copy_(forward_batch.oe_req_lens)
        if forward_batch.seq_lens_cpu is not None:
            if bs != raw_bs:
                self.seq_lens_cpu.fill_(1)
            self.seq_lens_cpu[:raw_bs].copy_(forward_batch.seq_lens_cpu)

        if forward_batch.mrope_positions is not None:
            self.mrope_positions[:, :raw_bs].copy_(forward_batch.mrope_positions)
        if hasattr(forward_batch.spec_info, "hidden_states"):
            self.hidden_states[:raw_num_token] = forward_batch.spec_info.hidden_states

        self.model_runner.init_request_cache_replay(
            forward_batch,
            bs,
            raw_bs, 
            [seq_len - self.num_tokens_per_bs for seq_len in self.seq_lens_cpu[:raw_bs].tolist()] + [0] * (bs - raw_bs), 
            [self.num_tokens_per_bs] * raw_bs + [1] * (bs - raw_bs), 
            forward_batch.reqs + [None] * (bs - raw_bs),
        )

        # Attention backend
        self.model_runner.attn_backend.init_forward_metadata_replay_cuda_graph(
            bs,
            self.req_pool_indices,
            self.seq_lens,
            forward_batch.seq_lens_sum + (bs - raw_bs),
            forward_batch.forward_mode,
            forward_batch.spec_info,
            seq_lens_cpu=self.seq_lens_cpu,
        )

        self.graphs[bs].replay()

        next_token_logits, hidden_states = self.output_buffers[bs]

        logits_output = LogitsProcessorOutput(
            next_token_logits=next_token_logits[:raw_num_token],
            hidden_states=(
                hidden_states[:raw_num_token] if hidden_states is not None else None
            ),
        )
        return logits_output

    def get_spec_info(self, num_tokens: int):
        spec_info = None
        if self.model_runner.spec_algorithm.is_eagle():
            from sglang.srt.speculative.eagle_utils import EagleVerifyInput

            if self.model_runner.is_draft_worker:
                raise RuntimeError("This should not happen.")
            else:
                spec_info = EagleVerifyInput(
                    draft_token=None,
                    positions=None,
                    draft_token_num=self.model_runner.server_args.speculative_num_draft_tokens,
                    spec_steps=self.model_runner.server_args.speculative_num_steps,
                    capture_hidden_mode=CaptureHiddenMode.FULL,
                )

        return spec_info
