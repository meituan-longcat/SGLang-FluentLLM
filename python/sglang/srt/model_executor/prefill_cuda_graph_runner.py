from __future__ import annotations

import bisect
import gc
from contextlib import contextmanager
from typing import TYPE_CHECKING

import torch
import tqdm

from sglang.srt.distributed import get_tensor_model_parallel_rank
from sglang.srt.distributed.parallel_state import graph_capture
from sglang.srt.layers.sampler.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.utils import get_available_gpu_memory
from sglang.srt.utils.oe_utils import OverEmbeddingInfo

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


def get_batch_sizes_to_capture(model_runner: ModelRunner):
    server_args = model_runner.server_args
    max_capture_bs = server_args.prefill_graph_max_bs
    assert max_capture_bs is not None
    capture_bs = list(range(1, min(max_capture_bs + 1, model_runner.req_to_token_pool.size)))
    return capture_bs


# Reuse this memory pool across all cuda graph runners.
global_graph_memory_pool = None


def get_global_graph_memory_pool():
    return global_graph_memory_pool


def set_global_graph_memory_pool(val):
    global global_graph_memory_pool
    global_graph_memory_pool = val


class PrefillCudaGraphRunner:
    """A CudaGraphRunner runs the forward pass of a model with cuda graph and torch.compile."""

    def __init__(self, model_runner: ModelRunner):
        # Parse args
        self.model_runner = model_runner
        self.graphs = {}
        self.output_buffers = {}
        self.disable_padding = model_runner.server_args.disable_cuda_graph_padding
        self.world_size = model_runner.server_args.world_size
        self.dp_size = model_runner.server_args.dp_size
        self.server_args = model_runner.server_args
        assert not self.server_args.enable_dp_attention, "Prefill Graph not support dp attention for now!"

        # Batch sizes to capture
        self.capture_bs = get_batch_sizes_to_capture(model_runner)
        self.capture_forward_mode = ForwardMode.EXTEND
        self.capture_hidden_mode = CaptureHiddenMode.NULL
        for bs in self.capture_bs:
            self.graphs[bs] = {}
        for bs in self.capture_bs:
            self.output_buffers[bs] = {}

        # Attention backend
        self.max_bs = max(self.capture_bs)
        self.max_num_token = self.server_args.prefill_graph_max_tokens

        num_tiles = (self.max_num_token + 16 - 1) // 16
        self.capture_num_tokens_range = [16 * (i + 1) for i in range(num_tiles)]
        self.model_runner.attn_backend.init_cuda_graph_state_prefill(self.max_num_token)
        self.seq_len_fill_value = self.model_runner.attn_backend.get_cuda_graph_seq_len_fill_value()
        self.seq_lens_cpu = torch.full((self.max_bs,), self.seq_len_fill_value, dtype=torch.int32)
        hf_config = self.model_runner.model_config.hf_config
        self.use_over_embedding = getattr(hf_config, "use_over_embedding", False)
        if self.use_over_embedding:
            self.over_embedding_n = hf_config.oe_neighbor_num
            self.over_embedding_k = hf_config.oe_split_num

        # Graph inputs
        with torch.device("cuda"):
            # Here for ds v3/r1, all 0 (bos) input_ids is unreasonable, will cause
            # nan in verify calculation process, conflicts with existing operators
            self.input_ids = torch.ones((self.max_num_token,), dtype=torch.int64)
            if self.use_over_embedding:
                raise AssertionError("TODO: use_over_embedding is not yet compatible with prefill cuda graph")
                over_embedding_input_ids = torch.ones((self.max_bs * self.over_embedding_n), dtype=torch.int32)
                oe_n_gram_ids = torch.zeros([self.max_bs, self.over_embedding_n - 1, self.over_embedding_k], dtype=torch.int32)
                oe_exclusive_req_len_sums = torch.zeros([self.max_bs + 1], dtype=torch.int32)
                oe_exclusive_oe_info_len_sums = torch.zeros([self.max_bs + 1], dtype=torch.int32)
                self.oe_info = OverEmbeddingInfo(
                    over_embedding_input_ids=over_embedding_input_ids,
                    oe_exclusive_req_len_sums=oe_exclusive_req_len_sums,
                    oe_exclusive_oe_info_len_sums=oe_exclusive_oe_info_len_sums,
                    oe_n_gram_ids=oe_n_gram_ids,
                )
            self.req_pool_indices = torch.zeros((self.max_bs,), dtype=torch.int32)
            self.seq_lens = torch.full((self.max_bs,), self.seq_len_fill_value, dtype=torch.int32)
            self.extend_seq_lens = torch.ones((self.max_bs,), dtype=torch.int32)
            self.extend_prefix_lens = torch.zeros((self.max_bs,), dtype=torch.int32)
            self.out_cache_loc = torch.arange(0, self.max_num_token, dtype=torch.int64)
            self.out_cache_loc.clamp_(min=0, max=63)
            self.positions = torch.arange(0, self.max_num_token, dtype=torch.int64)

        # Capture
        try:
            with self.model_capture_mode():
                self.capture()
        except RuntimeError as e:
            raise Exception(
                f"Capture Prefill cuda graph failed: {e}\n"
                "Possible solutions:\n"
                "1. disable cuda graph by --disable-cuda-graph\n"
                "2. set --mem-fraction-static to a smaller value (e.g., 0.8 or 0.7)\n"
                "3. disable torch compile by not using --enable-torch-compile\n"
                "4. set --cuda-graph-max-bs to a smaller value (e.g., 32)\n"
                "Open an issue on GitHub https://github.com/sgl-project/sglang/issues/new/choose \n"
            ) from e

    @contextmanager
    def model_capture_mode(self):
        if hasattr(self.model_runner.model, "capture_mode"):
            self.model_runner.model.capture_mode = True

        yield

        if hasattr(self.model_runner.model, "capture_mode"):
            self.model_runner.model.capture_mode = False

    def can_run(self, forward_batch: ForwardBatch):
        is_bs_supported = forward_batch.batch_size in self.graphs if self.disable_padding else forward_batch.batch_size <= self.max_bs
        is_num_tokens_supported = forward_batch.extend_num_tokens <= self.max_num_token
        return is_bs_supported and is_num_tokens_supported and forward_batch.forward_mode == ForwardMode.EXTEND

    def capture(self):
        # Trigger CUDA graph capture for specific shapes.
        # Capture the large shapes first so that the smaller shapes
        # can reuse the memory pool allocated for the large shapes.
        with freeze_gc(self.model_runner.server_args.enable_cudagraph_gc), graph_capture() as graph_capture_context:
            self.stream = graph_capture_context.stream
            capture_bs = sorted(self.capture_bs, reverse=True)
            capture_bs_range = tqdm.tqdm(capture_bs) if get_tensor_model_parallel_rank() == 0 else capture_bs
            capture_num_tokens_range = sorted(self.capture_num_tokens_range, reverse=True)
            for bs in capture_bs_range:
                if get_tensor_model_parallel_rank() == 0:
                    avail_mem = get_available_gpu_memory(
                        self.model_runner.device,
                        self.model_runner.gpu_id,
                        empty_cache=False,
                    )
                    capture_bs_range.set_description(f"Capturing batches ({avail_mem=:.2f} GB)")
                for num_tokens in capture_num_tokens_range:
                    (
                        graph,
                        output_buffers,
                    ) = self.capture_one_graph(bs, num_tokens)
                    self.graphs[bs][num_tokens] = graph
                    self.output_buffers[bs][num_tokens] = output_buffers

    def capture_one_graph(self, bs: int, num_tokens: int):
        graph = torch.cuda.CUDAGraph()
        stream = self.stream
        # Graph inputs
        input_ids = self.input_ids[:num_tokens]
        req_pool_indices = self.req_pool_indices[:bs]
        seq_lens = self.seq_lens[:bs]
        seq_lens[-1] = num_tokens - (bs - 1) * self.seq_len_fill_value
        extend_seq_lens = self.extend_seq_lens[:bs]
        extend_seq_lens[-1] = num_tokens - (bs - 1) * self.seq_len_fill_value
        out_cache_loc = self.out_cache_loc[:num_tokens]
        positions = self.positions[:num_tokens]
        extend_prefix_lens = self.extend_prefix_lens[:bs]

        global_num_tokens = None
        gathered_buffer = None

        forward_batch = ForwardBatch(
            forward_mode=self.capture_forward_mode,
            batch_size=bs,
            input_ids=input_ids,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            req_to_token_pool=self.model_runner.req_to_token_pool,
            token_to_kv_pool=self.model_runner.token_to_kv_pool,
            attn_backend=self.model_runner.attn_backend,
            out_cache_loc=out_cache_loc,
            seq_lens_sum=seq_lens.sum(),
            return_logprob=False,
            positions=positions,
            global_num_tokens=global_num_tokens,
            gathered_buffer=gathered_buffer,
            mrope_positions=None,
            spec_algorithm=None,
            spec_info=None,
            capture_hidden_mode=self.capture_hidden_mode,
            all_decode_or_idle=False,
            extend_prefix_lens=extend_prefix_lens,
            extend_seq_lens=extend_seq_lens,
            captureing_prefill_graph=True,
        )
        forward_batch.extend_seq_lens_cpu = extend_seq_lens.tolist()

        # Attention backend
        self.model_runner.attn_backend.init_forward_metadata_prefill_capture_cuda_graph(
            forward_batch,
            bs,
            num_tokens,
            seq_lens,
            extend_prefix_lens,
            forward_batch.req_pool_indices,
        )

        # Run and capture
        def run_once():
            logits_output = self.model_runner.forward_extend(forward_batch, skip_metadata_init=True)
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

        seq_lens[-1] = self.seq_len_fill_value
        extend_seq_lens[-1] = self.seq_len_fill_value
        torch.cuda.synchronize()
        self.model_runner.tp_group.barrier()

        global_graph_memory_pool = graph.pool()
        return graph, out

    def replay(self, forward_batch: ForwardBatch):
        raw_bs = forward_batch.batch_size
        raw_num_token = forward_batch.extend_num_tokens

        bs_index = bisect.bisect_left(self.capture_bs, raw_bs)
        bs = self.capture_bs[bs_index]
        num_token_index = bisect.bisect_left(self.capture_num_tokens_range, raw_num_token)
        num_tokens = self.capture_num_tokens_range[num_token_index]


        if bs != raw_bs or num_tokens != raw_num_token:
            self.seq_lens.fill_(1)
            self.out_cache_loc.zero_()
            self.extend_prefix_lens.zero_()
            self.extend_seq_lens.fill_(1)
            self.extend_prefix_lens.zero_()

        # Common inputs
        self.input_ids[:raw_num_token].copy_(forward_batch.input_ids)
        if self.use_over_embedding:
            self.oe_info.over_embedding_input_ids[: raw_bs * self.over_embedding_n].copy_(forward_batch.oe_info.over_embedding_input_ids)
            self.oe_info.oe_exclusive_req_len_sums[: raw_bs + 1].copy_(forward_batch.oe_info.oe_exclusive_req_len_sums)
            self.oe_info.oe_exclusive_oe_info_len_sums[: raw_bs + 1].copy_(forward_batch.oe_info.oe_exclusive_oe_info_len_sums)
            self.oe_info.oe_n_gram_ids[:raw_bs].copy_(forward_batch.oe_info.oe_n_gram_ids)
        self.req_pool_indices[:raw_bs].copy_(forward_batch.req_pool_indices)
        self.seq_lens[:raw_bs].copy_(forward_batch.seq_lens)
        self.extend_seq_lens[:raw_bs].copy_(forward_batch.extend_seq_lens)
        self.out_cache_loc[:raw_num_token].copy_(forward_batch.out_cache_loc)
        self.positions[:raw_num_token].copy_(forward_batch.positions)
        self.extend_prefix_lens[:raw_bs].copy_(forward_batch.extend_prefix_lens)
        if forward_batch.seq_lens_cpu is not None:
            if bs != raw_bs:
                self.seq_lens_cpu.fill_(1)
            self.seq_lens_cpu[:raw_bs].copy_(forward_batch.seq_lens_cpu)

        # Attention backend
        self.model_runner.attn_backend.init_forward_metadata_replay_prefill_cuda_graph(
            forward_batch, bs, num_tokens, self.seq_lens[:bs], self.req_pool_indices[:bs], self.extend_prefix_lens[:bs]
        )

        self.graphs[bs][num_tokens].replay()

        next_token_logits, hidden_states = self.output_buffers[bs][num_tokens]

        logits_output = LogitsProcessorOutput(
            next_token_logits=next_token_logits[:raw_num_token],
            hidden_states=(hidden_states[:raw_num_token] if hidden_states is not None else None),
        )
        return logits_output
