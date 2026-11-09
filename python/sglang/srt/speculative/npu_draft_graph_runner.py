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

import types
from typing import TYPE_CHECKING

import torch
import tqdm
from sglang.srt.configs.model_config import is_deepseek_nsa
from sglang.srt.distributed import get_ep_group, get_tensor_model_parallel_rank
from sglang.srt.env import ENV, global_server_args_dict
from sglang.srt.layers.attention.npu_mla_backend import get_attn_meta_npu
from sglang.srt.model_executor.cuda_graph_runner import get_batch_sizes_to_capture
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.npu.utils import npu_super_kernel
from sglang.srt.speculative.eagle_utils import EagleDraftInput
from sglang.srt.utils import (
    get_available_gpu_memory,
    get_compiler_backend,
    rank0_log,
)

if TYPE_CHECKING:
    from sglang.srt.speculative.eagle_worker import EAGLEWorker


class NpuDraftGraphRunner:
    def __init__(self, eagle_worker: EAGLEWorker, draft_model_idx=0):
        # Parse args
        if draft_model_idx == 0:
            self.model_runner = model_runner = eagle_worker.model_runner
        else:
            self.model_runner = model_runner = eagle_worker.model_runner_list[draft_model_idx]
        self.draft_model_idx = draft_model_idx
        self.graphs = {}
        self.output_buffers = {}
        self.tp_size = self.model_runner.tp_size
        self.dp_size = model_runner.server_args.dp_size
        self.speculative_num_steps = model_runner.server_args.speculative_num_steps
        self.topk = model_runner.server_args.speculative_eagle_topk
        self.capture_bs, self.compile_bs = get_batch_sizes_to_capture(model_runner)
        rank0_log(f"EAGLEDraftExtendNpuGraphRunner: capture bs {self.capture_bs}, compile bs {self.compile_bs}")
        self.padded_static_len = -1
        self.enable_cache = global_server_args_dict["npu_enable_graph_cache"]
        # Old persisted MTP graphs may declare int32 lengths. Keep those bytes
        # separate from the int64 capture/replay ABI; main-model caches are unchanged.
        self.graph_cache_dir = f"{global_server_args_dict['npu_graph_cache_path']}/mtp-lengths-i64-v1"
        self.backend = get_compiler_backend(model_runner.server_args)
        if not ENV.npu_disable_graph_mode:
            self.capture()

    @torch.inference_mode()
    def replay(self, forward_batch: ForwardBatch):
        bs = forward_batch.batch_size
        if forward_batch.forward_mode == ForwardMode.DRAFT_EXTEND:
            tag = "draft_extend"
            compile_method_name = f"compile_forward_draft_extend_bs_{bs}_idx_{self.draft_model_idx}"
        elif forward_batch.forward_mode.is_dflash_extend():
            tag = "dflash_extend"
            compile_method_name = f"compile_forward_dflash_extend_bs_{bs}_idx_{self.draft_model_idx}"
        else:
            tag = "draft_decode"
            compile_method_name = f"compile_forward_draft_decode_bs_{bs}_idx_{self.draft_model_idx}"
        # Pin per-request logprob metadata to the warmup-captured constants for the traced
        # call only (the draft graph never reads it; the sampler reads it eager after).
        # Otherwise a logprob request flips the ___check_obj_id guard -> on-demand recompile
        # of only the DP group that got it -> collective deadlock (EI0006) / Unsupported.
        # Mirror of the fix in npu_graph_runner.get_runner_context.runner_fn.
        saved = (
            forward_batch.return_logprob,
            forward_batch.top_logprobs_nums,
            forward_batch.token_ids_logprobs,
            forward_batch.extend_seq_lens,
        )
        forward_batch.return_logprob = False
        forward_batch.top_logprobs_nums = None
        forward_batch.token_ids_logprobs = None
        try:
            # The persisted MTP graph captures extend_seq_lens as int64. Greedy
            # verification keeps accept_length int32 for the overlap buffers;
            # prepare_extend_after_decode has already added its bonus in-place.
            # Normalize only the graph-facing field, then restore its original
            # alias below. Eager and replay entries still marked IDLE, DECODE or
            # DFLASH keep their inputs. Originally idle DPs that padding already
            # converted to DRAFT_EXTEND must use the same int64 graph contract.
            if (
                not ENV.npu_disable_graph_mode
                and forward_batch.forward_mode == ForwardMode.DRAFT_EXTEND
            ):
                if forward_batch.extend_seq_lens.dtype not in (torch.int32, torch.int64):
                    raise TypeError("MTP extend_seq_lens must be int32 or int64")
                forward_batch.extend_seq_lens = forward_batch.extend_seq_lens.to(
                    dtype=torch.int64
                )
            if ENV.npu_disable_graph_mode:
                compile_forward = self.model_runner.model.forward
            else:
                if not hasattr(self.model_runner.model, compile_method_name):
                    raise RuntimeError(
                        f"[DFLASH] draft graph miss: bs={bs} not in compile_bs={list(self.compile_bs)}, "
                        f"missing {compile_method_name}. 修法: can_run 改为 bs in compile_bs, "
                        f"或把 draft bs pad 到最近的 compile_bs, 或用 --npu-compile-bs 补齐档位。"
                    )
                compile_forward = getattr(self.model_runner.model, compile_method_name)
            with npu_super_kernel(tag, "stream-fusion=1", flag=forward_batch.can_run_with_graph):
                out = compile_forward(forward_batch.input_ids, forward_batch.positions, forward_batch)
        finally:
            (
                forward_batch.return_logprob,
                forward_batch.top_logprobs_nums,
                forward_batch.token_ids_logprobs,
                forward_batch.extend_seq_lens,
            ) = saved
        # Draft replay bypasses ModelRunner._forward_raw, including its eager
        # KVP current-token write-back. Without it, rank 0's temporary current
        # KV never becomes persistent history and later steps read stale pages.
        # Keep this outside the compiled region, as on the target path; models
        # with in-graph write-back enabled make this hook a no-op.
        fix_kvp_current_kv_cache = getattr(
            self.model_runner.model, "fix_kvp_current_kv_cache", None
        )
        if fix_kvp_current_kv_cache is not None:
            inline_layers = forward_batch.attn_backend.inline_current_keyrope_layers(forward_batch)
            if inline_layers:
                fix_kvp_current_kv_cache(
                    forward_batch, skip_keyrope_layers=inline_layers, skip_index_layers=inline_layers
                )
            else:
                fix_kvp_current_kv_cache(forward_batch)
        return out

    def capture(self):
        def build_method(method_name):
            def _forward(self, input_ids, positions, forward_batch, **kwargs):
                return self.forward(input_ids, positions, forward_batch, **kwargs)

            # TorchAir matches compiled records by the original code object.
            # Give each mode/BS/model its own code, not just a different __name__.
            _forward.__code__ = _forward.__code__.replace(co_name=method_name)
            _forward.__name__ = method_name
            _forward.__qualname__ = method_name
            return _forward

        need_draft_decode = (
            self.model_runner.server_args.speculative_num_draft_tokens > 2
            and not self.model_runner.server_args.is_multi_head_eagle
            and not self.model_runner.spec_algorithm.is_dflash()
        )
        need_draft_extend = not self.model_runner.spec_algorithm.is_dflash()

        compile_range = tqdm.tqdm(list(reversed(self.compile_bs))) if get_tensor_model_parallel_rank() == 0 else reversed(self.compile_bs)
        for bs in compile_range:
            if get_tensor_model_parallel_rank() == 0:
                avail_mem = get_available_gpu_memory(
                    self.model_runner.device,
                    self.model_runner.gpu_id,
                    empty_cache=False,
                )
                compile_range.set_description(f"Capturing MTP batch {bs} ({avail_mem=:.2f} GB)")

            # Dynamically create a wrapper function and bind it as a stable instance
            # attribute on the model. This serves two purposes:
            # 1. types.MethodType binds the function to the model instance so that
            #    'self' is automatically passed, avoiding manual self-passing.
            # 2. setattr stores the bound method as an instance attribute, so that
            #    getattr returns the same object every time (stable id), preventing
            #    Dynamo from seeing obj_id changes and triggering recompilation.

            # draft extend
            if need_draft_extend:
                extend_method_name = f"forward_draft_extend_bs_{bs}_idx_{self.draft_model_idx}"
                compile_extend_method_name = f"compile_forward_draft_extend_bs_{bs}_idx_{self.draft_model_idx}"
                setattr(
                    self.model_runner.model,
                    extend_method_name,
                    types.MethodType(build_method(extend_method_name), self.model_runner.model),
                )
                if self.enable_cache:
                    import torchair

                    setattr(
                        self.model_runner.model,
                        compile_extend_method_name,
                        torchair.inference.cache_compile(
                            getattr(self.model_runner.model, extend_method_name),
                            dynamic=False,
                            ge_cache=True,
                            cache_dir=self.graph_cache_dir,
                            backend=self.backend,
                        ),
                    )
                else:
                    setattr(
                        self.model_runner.model,
                        compile_extend_method_name,
                        torch.compile(
                            getattr(self.model_runner.model, extend_method_name),
                            fullgraph=True,
                            dynamic=False,
                            backend=self.backend,
                        ),
                    )

            # draft decode (if needed)
            if need_draft_decode:
                decode_method_name = f"forward_draft_decode_bs_{bs}_idx_{self.draft_model_idx}"
                compile_decode_method_name = f"compile_forward_draft_decode_bs_{bs}_idx_{self.draft_model_idx}"
                setattr(
                    self.model_runner.model,
                    decode_method_name,
                    types.MethodType(build_method(decode_method_name), self.model_runner.model),
                )
                if self.enable_cache:
                    import torchair

                    setattr(
                        self.model_runner.model,
                        compile_decode_method_name,
                        torchair.inference.cache_compile(
                            getattr(self.model_runner.model, decode_method_name),
                            dynamic=False,
                            ge_cache=True,
                            cache_dir=self.graph_cache_dir,
                            backend=self.backend,
                        ),
                    )
                else:
                    setattr(
                        self.model_runner.model,
                        compile_decode_method_name,
                        torch.compile(
                            getattr(self.model_runner.model, decode_method_name),
                            fullgraph=True,
                            dynamic=False,
                            backend=self.backend,
                        ),
                    )

            num_tokens = bs * self.model_runner.server_args.speculative_num_draft_tokens
            if need_draft_extend:
                forward_batch_draft_extend = self.prepare_forward_batch(bs, num_tokens, ForwardMode.DRAFT_EXTEND)
                with torch.inference_mode():
                    forward_batch_draft_extend.attn_metadata = get_attn_meta_npu(forward_batch_draft_extend, self.model_runner, True)

            if need_draft_decode:
                with torch.inference_mode():
                    forward_batch_draft_decode = self.prepare_forward_batch(bs, bs, ForwardMode.DECODE)
                    forward_batch_draft_decode.attn_metadata = get_attn_meta_npu(forward_batch_draft_decode, self.model_runner, True)

            if need_draft_extend:
                for _ in range(4):
                    torch.npu.synchronize()
                    self.model_runner.tp_group.barrier()
                    self.run_once_draft_extend(forward_batch_draft_extend, compile_extend_method_name)

            if need_draft_decode:
                for _ in range(4):
                    torch.npu.synchronize()
                    self.model_runner.tp_group.barrier()
                    self.run_once_draft_decode(forward_batch_draft_decode, compile_decode_method_name)

            # DFLASH draft 只有 DFLASH_EXTEND 一种前向 (单步非因果出整块)。
            if self.model_runner.spec_algorithm.is_dflash():
                dflash_extend_method_name = f"forward_dflash_extend_bs_{bs}_idx_{self.draft_model_idx}"
                compile_dflash_extend_method_name = f"compile_forward_dflash_extend_bs_{bs}_idx_{self.draft_model_idx}"
                setattr(
                    self.model_runner.model,
                    dflash_extend_method_name,
                    types.MethodType(build_method(dflash_extend_method_name), self.model_runner.model),
                )
                if self.enable_cache:
                    import torchair

                    setattr(
                        self.model_runner.model,
                        compile_dflash_extend_method_name,
                        torchair.inference.cache_compile(
                            getattr(self.model_runner.model, dflash_extend_method_name),
                            dynamic=False,
                            ge_cache=True,
                            cache_dir=self.graph_cache_dir,
                            backend=self.backend,
                        ),
                    )
                else:
                    setattr(
                        self.model_runner.model,
                        compile_dflash_extend_method_name,
                        torch.compile(
                            getattr(self.model_runner.model, dflash_extend_method_name),
                            fullgraph=True,
                            dynamic=False,
                            backend=self.backend,
                        ),
                    )

                forward_batch_dflash_extend = self.prepare_forward_batch(bs, num_tokens, ForwardMode.DFLASH_EXTEND)
                # 不在 inference_mode 下建 metadata, 与 replay 侧 (no_grad) 的普通 tensor 一致。
                forward_batch_dflash_extend.attn_metadata = get_attn_meta_npu(forward_batch_dflash_extend, self.model_runner, True)
                for _ in range(4):
                    torch.npu.synchronize()
                    self.model_runner.tp_group.barrier()
                    self.run_once_dflash_extend(forward_batch_dflash_extend, compile_dflash_extend_method_name)
        return

    @torch.inference_mode()
    def run_once_draft_extend(self, forward_batch, compile_method_name):
        # draft_model_idx=0: topk_indices=None → NSA Indexer will be called
        # draft_model_idx>0: topk_indices is a tensor → NSA Indexer will be skipped (reuse idx=0's result)
        if is_deepseek_nsa(self.model_runner.model_config.hf_config):
            if self.draft_model_idx == 0:
                forward_batch.topk_indices = None
                forward_batch.topk_valid_chunks = None
            else:
                num_tokens = forward_batch.input_ids.shape[0]
                index_topk = self.model_runner.model_config.hf_config.index_topk
                forward_batch.topk_indices = torch.zeros(
                    (num_tokens, 1, index_topk),
                    dtype=torch.int32,
                    device=self.model_runner.device,
                )
                forward_batch.topk_valid_chunks = torch.full(
                    (num_tokens,),
                    (index_topk + 127) // 128,
                    dtype=torch.int32,
                    device=self.model_runner.device,
                )
        compile_forward = getattr(self.model_runner.model, compile_method_name)
        with npu_super_kernel("draft_extend", "stream-fusion=1", flag=forward_batch.can_run_with_graph):
            return compile_forward(
                forward_batch.input_ids,
                forward_batch.positions,
                forward_batch,
            )

    @torch.inference_mode()
    def run_once_dflash_extend(self, forward_batch, compile_method_name):
        compile_forward = getattr(self.model_runner.model, compile_method_name)
        with npu_super_kernel("dflash_extend", "stream-fusion=1", flag=forward_batch.can_run_with_graph):
            return compile_forward(
                forward_batch.input_ids,
                forward_batch.positions,
                forward_batch,
            )

    @torch.inference_mode()
    def run_once_draft_decode(self, forward_batch, compile_method_name):
        compile_forward = getattr(self.model_runner.model, compile_method_name)
        with npu_super_kernel("draft_decode", "stream-fusion=1", flag=forward_batch.can_run_with_graph):
            return compile_forward(
                forward_batch.input_ids,
                forward_batch.positions,
                forward_batch,
            )

    def can_run(self, forward_batch: ForwardBatch):
        can = bool(
            forward_batch.all_decode_or_idle
            and (
                forward_batch.forward_mode == ForwardMode.DRAFT_EXTEND
                or forward_batch.forward_mode.is_dflash_extend()
                or forward_batch.forward_mode.is_idle()
                or forward_batch.forward_mode.is_decode()
            )
            and forward_batch.batch_size <= max(self.compile_bs)
        )
        forward_batch.can_run_with_graph = can
        return can

    def prepare_forward_batch(self, bs: int, num_tokens: int, forward_mode) -> ForwardBatch:
        # Graph inputs
        with torch.device(self.model_runner.device):
            positions = torch.zeros((num_tokens,), dtype=torch.int32)

            req_pool_indices = torch.zeros((bs,), dtype=torch.int64)
            seq_lens = torch.full((bs,), 1, dtype=torch.int32)
            out_cache_loc = torch.zeros((num_tokens,), dtype=torch.int32)

            new_tokens_to_compute = torch.full((bs,), 2, device="npu", dtype=torch.int32)
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
            capture_hidden_mode=CaptureHiddenMode.FULL,
            draft_token_num=self.model_runner.server_args.speculative_num_draft_tokens,
        )

        spec_info.accept_index = torch.ones(num_tokens, dtype=torch.int64, device=self.model_runner.device)
        eagle_hidden_size = self.model_runner.model_config.hidden_size
        if self.model_runner.spec_algorithm.is_eagle3():
            eagle_hidden_size = self.model_runner.model_config.hidden_size * 3
        # ref to python/sglang/srt/speculative/eagle_worker.py - new_verified_id = self.preprocess_for_draft_after_decode
        # input_ids 类别须与 replay 侧一致 (dispatch key guard): DFLASH replay 是 no_grad
        # 环境 (普通 tensor), MTP replay 是 inference_mode (inference tensor)。
        if forward_mode == ForwardMode.DFLASH_EXTEND:
            input_ids = torch.zeros((num_tokens,), dtype=torch.int32, device=self.model_runner.device)
        with torch.inference_mode():
            if forward_mode != ForwardMode.DFLASH_EXTEND:
                input_ids = torch.zeros((num_tokens,), dtype=torch.int32, device=self.model_runner.device)
            # Replay normalizes this graph input to int64 while preserving the
            # verifier's original int32 tensor/alias outside the graph.
            extend_seq_lens = torch.full((bs,), 0, dtype=torch.int64, device=self.model_runner.device)
            spec_info.hidden_states = torch.zeros((num_tokens, eagle_hidden_size), dtype=torch.bfloat16, device=self.model_runner.device)

        # DFLASH_EXTEND 走 get_attn_meta_npu 普通 extend 分支需这些字段。
        if forward_mode == ForwardMode.DFLASH_EXTEND:
            block = self.model_runner.server_args.speculative_num_draft_tokens
            extend_seq_lens = torch.full((bs,), block, dtype=torch.int64, device=self.model_runner.device)
            extend_prefix_lens = torch.full((bs,), 1, dtype=torch.int32, device=self.model_runner.device)
            extend_prefix_lens_cpu = [1] * bs
            extend_seq_lens_cpu = [block] * bs
            extend_num_tokens = bs * block
            new_tokens_total = bs * block
            seq_lens = torch.full((bs,), 1 + block, dtype=torch.int32, device=self.model_runner.device)
        else:
            extend_prefix_lens = None
            extend_prefix_lens_cpu = None
            extend_seq_lens_cpu = None
            extend_num_tokens = None
            new_tokens_total = None

        if self.model_runner.model_config.use_over_embedding:
            self.oe_column_starts = torch.zeros(bs, dtype=torch.int32, device="npu")
            self.oe_req_lens = torch.zeros(bs, dtype=torch.int32, device="npu")
            self.oe_out_column_starts = torch.zeros(bs, dtype=torch.int32, device="npu")
            self.oe_out_req_lens = torch.zeros(bs, dtype=torch.int32, device="npu")

        if forward_mode == ForwardMode.DFLASH_EXTEND:
            global_num_tokens = [bs * self.model_runner.server_args.speculative_num_draft_tokens] * get_ep_group().world_size
        else:
            global_num_tokens = [bs] * self.tp_size

        # capture 须与 replay 一致, 否则 warmup 重编译: DFLASH_EXTEND replay 恒 NULL, multi-head 恒 FULL。
        if forward_mode == ForwardMode.DFLASH_EXTEND:
            capture_hidden_mode = CaptureHiddenMode.NULL
        elif self.model_runner.server_args.is_multi_head_eagle:
            capture_hidden_mode = CaptureHiddenMode.FULL
        else:
            capture_hidden_mode = CaptureHiddenMode.LAST

        forward_batch = ForwardBatch(
            global_num_tokens=global_num_tokens,
            forward_mode=forward_mode,
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
            spec_algorithm=self.model_runner.spec_algorithm,
            spec_info=spec_info,
            capture_hidden_mode=capture_hidden_mode,
            new_tokens_to_compute=new_tokens_to_compute,
            gathered_buffer=gathered_buffer,
            extend_seq_lens=extend_seq_lens,
            extend_prefix_lens=extend_prefix_lens,
            extend_prefix_lens_cpu=extend_prefix_lens_cpu,
            extend_seq_lens_cpu=extend_seq_lens_cpu,
            extend_num_tokens=extend_num_tokens,
            new_tokens_total=new_tokens_total,
            extend_logprob_start_lens_cpu=extend_logprob_start_lens_cpu,
            all_decode_or_idle=True,
            can_run_with_graph=True,
            can_run_all2all=ENV.npu_enable_all2all_comm,
            oe_token_table=self.model_runner.oe_token_table if self.model_runner.model_config.use_over_embedding else None,
            oe_column_starts=self.oe_column_starts if self.model_runner.model_config.use_over_embedding else None,
            oe_req_lens=self.oe_req_lens if self.model_runner.model_config.use_over_embedding else None,
            oe_out_column_starts=self.oe_out_column_starts if self.model_runner.model_config.use_over_embedding else None,
            oe_out_req_lens=self.oe_out_req_lens if self.model_runner.model_config.use_over_embedding else None,
        )
        forward_batch.topk_indices = None
        forward_batch.topk_valid_chunks = None
        return forward_batch
