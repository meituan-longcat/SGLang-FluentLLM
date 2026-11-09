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

import os
import types
from collections.abc import Generator
from contextlib import contextmanager
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
)

import torch
import tqdm

from sglang.srt.distributed import get_ep_group, get_tensor_model_parallel_rank, get_world_group
from sglang.srt.env import ENV, global_server_args_dict
from sglang.srt.layers.attention.npu_mla_backend import get_attn_meta_npu
from sglang.srt.model_executor.cuda_graph_runner import DeviceRunnerBase
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
)
from sglang.srt.speculative.eagle_utils import EagleVerifyInput
from sglang.srt.utils import (
    get_available_gpu_memory,
    get_compiler_backend,
)

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner


class NpuGraphRunner(DeviceRunnerBase):
    """A NpuGraphRunner runs the forward pass of a model with npu graph engine and torch.compile."""

    def __init__(self, model_runner: ModelRunner):
        self.enable_cache = global_server_args_dict["npu_enable_graph_cache"]
        self.graph_cache_dir = global_server_args_dict["npu_graph_cache_path"]
        if ENV.npu_moe_mask_padding and self.graph_cache_dir:
            self.graph_cache_dir = os.path.join(self.graph_cache_dir, "moe-prefix-mask-v1")
        if global_server_args_dict.get("offload_cache", False):
            from sglang.srt.mem_cache.kv_offload.ge_writeback import (
                install_update_cache_writeback_fix,
            )

            install_update_cache_writeback_fix()
        self.backend = get_compiler_backend(model_runner.server_args)

        if hasattr(model_runner.model, "language_model"):
            model_runner.model = model_runner.model.language_model
        super().__init__(model_runner)

    def warm_up(self):
        if ENV.npu_disable_graph_mode:
            return
        self.run_fake()

    def build_forward_batch(self, bs):
        # construct warm up forward batch
        num_tokens = bs * self.model_runner.server_args.speculative_num_draft_tokens

        with torch.device(self.model_runner.device):
            input_ids = torch.zeros((num_tokens,), dtype=torch.int64)
            req_pool_indices = torch.zeros((bs,), dtype=torch.int64)
            seq_lens = torch.ones((bs,), dtype=torch.int32)
            positions = torch.zeros((num_tokens,), dtype=torch.int32)
            extend_seq_lens = None if self.model_runner.server_args.speculative_num_draft_tokens == 1 else torch.full((bs,), 0, dtype=torch.int32)
            out_cache_loc = torch.zeros((num_tokens,), dtype=torch.int32)
            # 材化的 cache_loc dummy: 全 0 落在保留槽 slot 0, 与 pad 行取值一致。
            # 判据须与 replay 侧同源 (materializer 是否挂载), 否则 guard 失配。
            _materializer = getattr(
                getattr(self.model_runner.model, "logits_processor", None),
                "dflash_materializer",
                None,
            )
            dflash_materialize_cache_loc = torch.zeros((num_tokens,), dtype=torch.int32) if _materializer is not None else None
            new_tokens_to_compute = torch.full((bs,), self.model_runner.server_args.speculative_num_draft_tokens, dtype=torch.int32)
            if self.model_runner.model_config.use_over_embedding:
                oe_column_starts = torch.zeros(bs, dtype=torch.int32)
                oe_req_lens = torch.zeros(bs, dtype=torch.int32)
                oe_out_column_starts = torch.zeros(bs, dtype=torch.int32)
                oe_out_req_lens = torch.zeros(bs, dtype=torch.int32)

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
            self.capture_hidden_mode = spec_info.capture_hidden_mode if spec_info else CaptureHiddenMode.NULL

        forward_batch = ForwardBatch(
            forward_mode=self.capture_forward_mode,
            batch_size=bs,
            input_ids=input_ids,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            req_to_token_pool=self.model_runner.req_to_token_pool,
            token_to_kv_pool=self.model_runner.token_to_kv_pool,
            attn_backend=self.model_runner.attn_backend,
            spec_num_steps=self.model_runner.server_args.speculative_num_steps,
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
            dflash_materialize_cache_loc=dflash_materialize_cache_loc,
        )
        forward_batch.prepare_moe_valid_token_mask(self.model_runner, num_tokens, num_tokens)
        with torch.inference_mode():
            # ref to python/sglang/srt/model_executor/model_runner.py - forward_batch.attn_metadata = get_attn_meta_npu(forward_batch, self)
            attn_metadata = get_attn_meta_npu(forward_batch, self.model_runner, True)
            forward_batch.attn_metadata = attn_metadata
        if self.model_runner.server_args.speculative_num_draft_tokens == 1:
            forward_batch.input_ids = torch.zeros((num_tokens,), dtype=torch.int32, device=self.model_runner.device)
        forward_batch.can_run_all2all = forward_batch.all_decode_or_idle and ENV.npu_enable_all2all_comm
        return forward_batch

    def run_fake(self):
        budget = getattr(self.model_runner, "offload_budget", None)
        calibrated = budget is not None and budget.graph_extra_bytes is not None
        capture_order = (list(budget.graph_capture_order) if calibrated
                         else list(reversed(self.compile_bs)))
        if calibrated and (len(capture_order) != len(self.compile_bs)
                           or set(capture_order) != set(self.compile_bs)):
            raise ValueError("Calibrated capture order does not match compiled graph set")
        compile_range = (tqdm.tqdm(capture_order)
                         if get_tensor_model_parallel_rank() == 0 else capture_order)
        first_graph_free = None

        def build_method(method_name):
            # TorchDynamo keys these graph entry points by their code-level
            # function names. Mutating __name__ on one closure is not enough
            # and can make inference_mode tracing fail on newer PyTorch.
            namespace = {}
            exec(  # noqa: S102
                f"def {method_name}(self, input_ids, positions, forward_batch, **kwargs):\n"
                "    return self.forward(input_ids, positions, forward_batch, **kwargs)\n",
                {},
                namespace,
            )
            return namespace[method_name]

        for bs in compile_range:
            if get_tensor_model_parallel_rank() == 0:
                avail_mem = get_available_gpu_memory(
                    self.model_runner.device,
                    self.model_runner.gpu_id,
                    empty_cache=False,
                )
                compile_range.set_description(f"Capturing batch {bs} ({avail_mem=:.2f} GB)")
            if self.model_runner.server_args.speculative_num_draft_tokens == 1:
                with torch.inference_mode():
                    forward_batch = self.build_forward_batch(bs)
            else:
                forward_batch = self.build_forward_batch(bs)

            method_name = f"forward_bs_{bs}"
            compile_method_name = f"compile_forward_bs_{bs}"
            setattr(self.model_runner.model, method_name, types.MethodType(build_method(method_name), self.model_runner.model))
            if self.enable_cache:
                import torchair

                setattr(
                    self.model_runner.model,
                    compile_method_name,
                    torchair.inference.cache_compile(
                        getattr(self.model_runner.model, method_name),
                        dynamic=False,
                        ge_cache=True,
                        cache_dir=self.graph_cache_dir,
                        backend=self.backend,
                    ),
                )
            else:
                setattr(
                    self.model_runner.model,
                    compile_method_name,
                    torch.compile(getattr(self.model_runner.model, method_name), fullgraph=True, dynamic=False, backend=self.backend),
                )

            # Run and capture
            @torch.inference_mode()
            def run_once(forward_batch=forward_batch, compile_method_name=compile_method_name):
                forward_batch.can_run_with_graph = True
                compile_forward = getattr(self.model_runner.model, compile_method_name)
                logits_output = compile_forward(forward_batch.input_ids, forward_batch.positions, forward_batch)
                return logits_output

            for i in range(4):
                torch.npu.synchronize()
                self.model_runner.tp_group.barrier()
                print(f"[rank {get_tensor_model_parallel_rank()}] run_once iter {i} for bs={bs}")
                run_once()
            if calibrated:
                torch.npu.synchronize()
                free_bytes = int(torch.npu.mem_get_info()[0])
                if first_graph_free is None:
                    first_graph_free = free_bytes
                local_extra = max(0, first_graph_free - free_bytes)
                worst_extra = torch.tensor(local_extra, dtype=torch.int64)
                torch.distributed.all_reduce(
                    worst_extra, op=torch.distributed.ReduceOp.MAX,
                    group=get_world_group().cpu_group,
                )
                measured = int(worst_extra.item())
                if get_tensor_model_parallel_rank() == 0:
                    print(f"Offload calibrated graph reserve: bs={bs} "
                          f"worst_extra_bytes={measured} "
                          f"reserved_bytes={budget.graph_extra_bytes}", flush=True)
                if measured > budget.graph_extra_bytes:
                    raise RuntimeError(
                        f"Main graph growth exceeded calibrated reserve: "
                        f"{measured} > {budget.graph_extra_bytes}; recalibrate this runtime"
                    )
        return

    @contextmanager
    def get_runner_context(self, forward_batch: ForwardBatch) -> Generator[Callable[[], Any], Any, None]:
        @torch.inference_mode()
        def runner_fn():
            # Per-request logprob metadata (return_logprob / top_logprobs_nums /
            # token_ids_logprobs) is read only in eager post-processing — the decode and
            # target-verify graphs never branch on it (logits_processor decode/verify path
            # ignores it; the sampler runs outside the graph). But torchair/dynamo still
            # emits ___check_obj_id guards on these attrs, so a logprob request flips the
            # guard and recompiles forward_bs_N on only the DP group that received it; its
            # collective HcomAllGather then waits for the other DP group that never
            # recompiled -> EI0006 socket timeout. Pin these to the warmup-captured
            # constants for the traced call only, then restore so the eager sampler
            # (model_runner.sample, which reads them off forward_batch) still emits logprobs.
            saved = (
                forward_batch.return_logprob,
                forward_batch.top_logprobs_nums,
                forward_batch.token_ids_logprobs,
            )
            forward_batch.return_logprob = False
            forward_batch.top_logprobs_nums = None
            forward_batch.token_ids_logprobs = None
            try:
                compile_method_name = f"compile_forward_bs_{forward_batch.batch_size}"
                if ENV.npu_disable_graph_mode:
                    forward_batch.can_run_with_graph = False
                    compile_forward = self.model_runner.model.forward
                else:
                    compile_forward = getattr(self.model_runner.model, compile_method_name)
                return compile_forward(forward_batch.input_ids, forward_batch.positions, forward_batch)
            finally:
                (
                    forward_batch.return_logprob,
                    forward_batch.top_logprobs_nums,
                    forward_batch.token_ids_logprobs,
                ) = saved

        forward_batch.attn_backend.init_forward_metadata(forward_batch)
        yield runner_fn

    def can_run_graph(self, forward_batch: ForwardBatch) -> bool:
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
            spec_steps=self.model_runner.server_args.speculative_num_steps,
            capture_hidden_mode=CaptureHiddenMode.FULL,
            is_all_greedy=True,
        )
        return spec_info
