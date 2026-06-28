import bisect
from typing import TYPE_CHECKING, Optional

import torch
import tqdm
from contextlib import contextmanager

from sglang.srt.custom_op import CustomOp
from sglang.srt.layers.moe.fused_moe_native import fused_moe_forward_native
from sglang.srt.constrained.base_grammar_backend import create_grammar_backend
from sglang.srt.distributed import get_tensor_model_parallel_rank
from sglang.srt.distributed.parallel_state import graph_capture
from sglang.srt.hf_transformers_utils import get_tokenizer
from sglang.srt.model_executor.cuda_graph_runner import get_batch_sizes_to_capture
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo

if TYPE_CHECKING:
    from sglang.srt.speculative.eagle_worker import EAGLEWorker

from sglang.srt.utils import get_colorful_logger

logger = get_colorful_logger(__name__)

global_graph_memory_pool = None

def _to_torch(model: torch.nn.Module, reverse: bool, num_tokens: int, draft_token_num: int):
    for sub in model._modules.values():
        if isinstance(sub, CustomOp):
            if reverse:
                sub._forward_method = sub.forward_cuda
                setattr(sub, "is_torch_compile", False)
            else:
                # NOTE: Temporarily workaround MoE
                if "FusedMoE" in sub.__class__.__name__:
                    if num_tokens <= 1 * draft_token_num:
                        sub._forward_method = fused_moe_forward_native
                else:
                    sub._forward_method = sub.forward_native
                setattr(sub, "is_torch_compile", True)
        if isinstance(sub, torch.nn.Module):
            _to_torch(sub, reverse, num_tokens, draft_token_num)

@contextmanager
def patch_model(
    model: torch.nn.Module,
    enable_compile: bool,
    num_tokens: int,
    tp_group,
    draft_token_num : int
):
    """Patch the model to make it compatible with with torch.compile"""
    backup_ca_comm = None

    try:
        if enable_compile:
            _to_torch(model, reverse=False, num_tokens=num_tokens, draft_token_num=draft_token_num)
            backup_ca_comm = tp_group.ca_comm
            # Use custom-allreduce here.
            # We found the custom allreduce is much faster than the built-in allreduce in torch,
            # even with ENABLE_INTRA_NODE_COMM=1.
            # tp_group.ca_comm = None
            yield torch.compile(
                torch.no_grad()(model.forward),
                mode="max-autotune-no-cudagraphs",
                dynamic=False,
            )
        else:
            yield model.forward
    finally:
        if enable_compile:
            _to_torch(model, reverse=True, num_tokens=num_tokens, draft_token_num=draft_token_num)
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

class SpecDecodeCudaGraphRunner:
    def __init__(self, spec_decode_worker: "EAGLEWorker"):
        self.output_buffers = {}
        self.graphs = {}
        self.spec_decode_worker = spec_decode_worker
        self.enable_torch_compile = spec_decode_worker.model_runner.server_args.enable_torch_compile
        self.capture_bs, self.complile_bs = get_batch_sizes_to_capture(spec_decode_worker.model_runner)
        logger.info(f"{self.capture_bs=} {self.complile_bs=}")

        self.max_bs = max(self.capture_bs)
        self.draft_token_num = (
            spec_decode_worker.server_args.speculative_num_draft_tokens
        )
        self.max_num_token = self.max_bs * self.draft_token_num
        self.enable_dp_attention = (
            spec_decode_worker.model_runner.server_args.enable_dp_attention
        )
        self.world_size = spec_decode_worker.model_runner.server_args.world_size
        self.dp_size = spec_decode_worker.model_runner.server_args.dp_size
        self.capture_sample_graph = (
            spec_decode_worker.model_runner.server_args.capture_sample_graph
        )

        self.spec_decode_worker.target_worker.model_runner.attn_backend.init_cuda_graph_state(
            self.max_num_token
        )
        assert hasattr(self.spec_decode_worker, "model_runner_list")
        for runner in self.spec_decode_worker.model_runner_list:
            runner.attn_backend.init_cuda_graph_state(
                self.max_num_token
            )

        self.disable_padding = (
            self.spec_decode_worker.model_runner.server_args.disable_cuda_graph_padding
        )
        self.seq_lens_cpu = torch.full((self.max_bs,), 1, dtype=torch.int32).cpu()

        hf_config = spec_decode_worker.model_runner.model_config.hf_config
        if getattr(hf_config, 'use_over_embedding', False):
            self.over_embedding_n = hf_config.oe_neighbor_num
            self.over_embedding_k = hf_config.oe_split_num
        self.use_over_embedding = getattr(hf_config, 'use_over_embedding', False) or self.spec_decode_worker.target_worker.use_over_embedding

        if self.use_over_embedding:
            self.token_table = spec_decode_worker.oe_token_table

        if self.enable_torch_compile:
            set_torch_compile_config()

        with torch.device("cuda"):
            self.req_pool_indices = torch.zeros((self.max_bs,), dtype=torch.int64)
            self.draft_token = torch.ones(
                (self.max_bs * self.draft_token_num,), dtype=torch.int32
            )
            self.oe_column_starts = torch.empty([self.max_bs], dtype=torch.int32) if self.use_over_embedding else None
            self.oe_req_lens = torch.empty([self.max_bs], dtype=torch.int32) if self.use_over_embedding else None
            self.oe_out_column_starts = torch.empty([self.max_bs], dtype=torch.int32) if self.use_over_embedding else None
            self.oe_out_req_lens = torch.empty([self.max_bs], dtype=torch.int32) if self.use_over_embedding else None
            self.out_cache_loc_buffer = torch.arange(
                0, self.max_num_token, dtype=torch.int64
            )
            self.positions = torch.arange(0, self.max_num_token, dtype=torch.int64)
            self.mrope_positions = torch.zeros((3, self.max_bs), dtype=torch.int64)
            self.hidden_states = torch.zeros(
                (
                    self.max_num_token,
                    spec_decode_worker.model_runner.model_config.hidden_size,
                ),
                dtype=spec_decode_worker.model_runner.dtype,
            )
            self.seq_lens = torch.full((self.max_bs,), 1, dtype=torch.int32)
            self.new_tokens_to_compute = torch.full(
                (self.max_bs,),
                self.spec_decode_worker.speculative_num_steps - 1,
                device="cuda",
                dtype=torch.int32,
            )
            self.temperature_buffer = torch.zeros(
                (self.max_bs, 1), dtype=torch.float32
            )
            self.topk_buffer = torch.zeros((self.max_bs,), dtype=torch.int32)
            self.topp_buffer = torch.zeros((self.max_bs,), dtype=torch.float32)
            self.minp_buffer = torch.zeros((self.max_bs,), dtype=torch.float32)
            if self.enable_dp_attention:
                self.gathered_buffer = torch.zeros(
                    (
                        self.max_num_token * self.world_size,
                        self.spec_decode_worker.model_runner.model_config.hidden_size,
                    ),
                    dtype=self.spec_decode_worker.model_runner.dtype,
                )
            self.scaling_penalties = torch.ones(
                (self.max_bs,),
                dtype=torch.float32
            )
            self.cumulated_scaling_penalties = torch.ones(
                (self.max_bs, self.spec_decode_worker.target_worker.model_runner.model_config.vocab_size),
                dtype=torch.float32
            )

        self.grammar_backend = None
        if self.enable_grammar_backend():
            server_args = self.spec_decode_worker.model_runner.server_args
            self.tokenizer = get_tokenizer(
                server_args.tokenizer_path,
                tokenizer_mode=server_args.tokenizer_mode,
                trust_remote_code=server_args.trust_remote_code,
                revision=server_args.revision,
            )
            self.grammar_backend = create_grammar_backend(
                server_args,
                self.tokenizer,
                self.spec_decode_worker.model_runner.model_config.vocab_size,
            )
            dummpy_grammar = self.grammar_backend.init_value(("regex", r"[a-z]+"))
            self.vocab_masks = dummpy_grammar.allocate_vocab_mask(
                vocab_size=self.spec_decode_worker.model_runner.model_config.vocab_size,
                batch_size=self.max_bs * self.draft_token_num,
                device="cpu",
            )
            self.vocab_masks = self.vocab_masks.to("cuda")

        # Capture
        try:
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

    def can_run(self, forward_batch: ForwardBatch):
        if not (
            forward_batch.forward_mode.is_target_verify()
            or forward_batch.forward_mode.is_idle()
        ):
            return False
        if self.enable_dp_attention:
            min_num_tokens, max_num_tokens = (
                min(forward_batch.global_num_tokens),
                max(forward_batch.global_num_tokens),
            )
            max_bsz = int(max_num_tokens / self.draft_token_num)
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

    def replay(self, forward_batch: ForwardBatch, vocab_masks: Optional[torch.Tensor]):
        assert forward_batch.out_cache_loc is not None
        raw_bs = forward_batch.batch_size
        raw_num_token = raw_bs * self.draft_token_num

        # Pad
        if self.enable_dp_attention:
            index = bisect.bisect_left(
                self.capture_bs,
                max(forward_batch.global_num_tokens) // self.draft_token_num,
            )
        else:
            index = bisect.bisect_left(self.capture_bs, raw_bs)
        bs = self.capture_bs[index]
        if bs != raw_bs:
            self.seq_lens.fill_(1)
            self.out_cache_loc_buffer.zero_()
            self.positions.zero_()
        self.req_pool_indices.zero_()

        if forward_batch.forward_mode.is_idle():
            forward_batch.spec_info = self.init_verify_spec_info(bs=0)
            forward_batch.forward_mode = ForwardMode.TARGET_VERIFY

        assert forward_batch.forward_mode.is_target_verify()

        # Common inputs
        self.draft_token[:raw_num_token].copy_(forward_batch.spec_info.draft_token)
        if vocab_masks is not None:
            logger.debug(f"[SpecDecodeCudaGraphRUnner] copy vocab masks")
            self.vocab_masks[:raw_num_token].copy_(vocab_masks)
        elif self.grammar_backend:
            self.grammar_backend.reset_vocab_masks(self.vocab_masks[:raw_num_token])
        self.req_pool_indices[:raw_bs].copy_(forward_batch.req_pool_indices)
        self.seq_lens[:raw_bs].copy_(forward_batch.seq_lens)
        if forward_batch.seq_lens_cpu is not None:
            if bs != raw_bs:
                self.seq_lens_cpu.fill_(1)
            self.seq_lens_cpu[:raw_bs].copy_(forward_batch.seq_lens_cpu)
        self.out_cache_loc_buffer[:raw_num_token].copy_(forward_batch.out_cache_loc)
        self.positions[:raw_num_token].copy_(forward_batch.positions)
        if self.use_over_embedding:
            forward_batch.oe_column_starts[:raw_num_token].copy_(forward_batch.oe_column_starts[:raw_num_token])
            forward_batch.oe_req_lens[:raw_bs].copy_(forward_batch.oe_req_lens[:raw_bs])

        assert forward_batch.forward_mode.is_target_verify()

        sampling_info = forward_batch.sampling_info
        self.scaling_penalties[: raw_bs].copy_(sampling_info.repetition_penalties)
        self.cumulated_scaling_penalties[: raw_bs].copy_(forward_batch.spec_info.cumulated_scaling_penalties)

        if self.capture_sample_graph:
            self.temperature_buffer[:raw_bs].copy_(sampling_info.temperatures)
            self.topk_buffer[:raw_bs].copy_(sampling_info.top_ks)
            self.topp_buffer[:raw_bs].copy_(sampling_info.top_ps)
            self.minp_buffer[:raw_bs].copy_(sampling_info.min_ps)

        # Attention backend
        for runner in self.spec_decode_worker.model_runner_list:
            runner.attn_backend.init_forward_metadata_replay_cuda_graph(
                bs,
                self.req_pool_indices,
                self.seq_lens,
                forward_batch.seq_lens_sum
                + (bs - raw_bs),
                forward_batch.forward_mode,
                forward_batch.spec_info,
                seq_lens_cpu=self.seq_lens_cpu,
            )
        self.spec_decode_worker.target_worker.model_runner.attn_backend.init_forward_metadata_replay_cuda_graph(
            bs,
            self.req_pool_indices,
            self.seq_lens,
            forward_batch.seq_lens_sum + (bs - raw_bs),
            forward_batch.forward_mode,
            forward_batch.spec_info,
            seq_lens_cpu=self.seq_lens_cpu,
        )

        # Replay
        self.graphs[bs].replay()
        out = self.output_buffers[bs]
        if bs != raw_bs:
            forward_batch.batch_size = raw_bs
            forward_batch.positions = self.positions[:raw_num_token]
            forward_batch.seq_lens = self.seq_lens[:raw_bs]
            forward_batch.req_pool_indices = self.req_pool_indices[:raw_bs]
            out = self._postprocess_output_to_raw_bs(out, raw_bs)
        return out

    def init_verify_spec_info(self, bs: int):
        from sglang.srt.speculative.eagle_utils import EagleVerifyInput

        grammar = None
        if self.grammar_backend:
            grammar = self.grammar_backend.init_value(("regex", r".*"))

        spec_info = EagleVerifyInput(
            draft_token=self.draft_token[: bs * self.draft_token_num],
            positions=self.positions[: bs * self.draft_token_num],
            draft_token_num=self.spec_decode_worker.model_runner.server_args.speculative_num_draft_tokens,
            spec_steps=self.spec_decode_worker.model_runner.server_args.speculative_num_steps,
            capture_hidden_mode=CaptureHiddenMode.FULL,
            is_all_greedy=False if self.capture_sample_graph else True,
            grammar=grammar,
            cumulated_scaling_penalties=self.cumulated_scaling_penalties[:bs, :]
        )
        return spec_info

    def capture(self):
        with graph_capture() as graph_capture_context:
            self.stream = graph_capture_context.stream
            capture_range = (
                tqdm.tqdm(self.capture_bs)
                if get_tensor_model_parallel_rank() == 0
                else self.capture_bs
            )
            for bs in capture_range:
                if bs in self.complile_bs:
                    with patch_model(
                        self.spec_decode_worker.target_worker.model_runner.model,
                        True,
                        num_tokens=bs * self.draft_token_num,
                        tp_group=self.spec_decode_worker.target_worker.model_runner.tp_group,
                        draft_token_num=self.draft_token_num
                    ) as forward:
                        origin_forward_func = self.spec_decode_worker.target_worker.model_runner.model.forward
                        self.spec_decode_worker.target_worker.model_runner.model.forward = forward
                        (
                            graph,
                            output_buffers,
                        ) = self.capture_one_batch_size(bs)
                        self.spec_decode_worker.target_worker.model_runner.model.forward = origin_forward_func
                else:
                    (
                        graph,
                        output_buffers,
                    ) = self.capture_one_batch_size(bs)
                self.graphs[bs] = graph
                self.output_buffers[bs] = output_buffers

    def get_forward_batch(self, bs):
        input_ids = self.draft_token[: bs * self.draft_token_num]
        req_pool_indices = self.req_pool_indices[:bs]
        seq_lens = self.seq_lens[:bs]
        out_cache_loc = self.out_cache_loc_buffer[: bs * self.draft_token_num]
        positions = self.positions[: bs * self.draft_token_num]

        if self.enable_dp_attention:
            global_num_tokens = [bs * self.draft_token_num] * self.world_size
            gathered_buffer = self.gathered_buffer[
                : bs * self.world_size * self.draft_token_num
            ]
            global_batch_size = [bs] * self.world_size
        else:
            global_num_tokens = None
            gathered_buffer = None
            global_batch_size = None

        verify_spec_info = self.init_verify_spec_info(bs)
        forward_batch = ForwardBatch(
            forward_mode=ForwardMode.TARGET_VERIFY,
            batch_size=bs,
            input_ids=input_ids,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            req_to_token_pool=self.spec_decode_worker.req_to_token_pool,
            token_to_kv_pool=self.spec_decode_worker.target_worker.model_runner.token_to_kv_pool,
            attn_backend=self.spec_decode_worker.target_worker.model_runner.attn_backend,
            out_cache_loc=out_cache_loc,
            seq_lens_sum=seq_lens.sum(),
            oe_token_table=self.token_table if self.use_over_embedding else None,
            oe_column_starts=self.oe_column_starts[: bs] if self.use_over_embedding else None,
            oe_req_lens=self.oe_req_lens[: bs] if self.use_over_embedding else None,
            oe_out_column_starts=self.oe_out_column_starts[: bs] if self.use_over_embedding else None,
            oe_out_req_lens=self.oe_out_req_lens[: bs] if self.use_over_embedding else None,
            return_logprob=False,
            positions=positions,
            global_num_tokens=global_num_tokens,
            gathered_buffer=gathered_buffer,
            mrope_positions=None,
            spec_algorithm=self.spec_decode_worker.model_runner.spec_algorithm,
            spec_info=verify_spec_info,
            capture_hidden_mode=CaptureHiddenMode.FULL,
            all_decode_or_idle=True,
            draft_input_ids=None,
            new_tokens_to_compute=self.new_tokens_to_compute[:bs],
            global_batch_size=global_batch_size,
        )

        forward_batch.sampling_info = SamplingBatchInfo(
            temperatures=self.temperature_buffer[:bs],
            top_ks=self.topk_buffer[:bs],
            top_ps=self.topp_buffer[:bs],
            min_ps=self.minp_buffer[:bs],
            repetition_penalties=self.scaling_penalties[:bs],
            is_all_greedy=False if self.capture_sample_graph else True,
            need_min_p_sampling=False,
            vocab_size=self.spec_decode_worker.model_runner.model_config.vocab_size,
        )
        return forward_batch

    def capture_one_batch_size(self, bs: int):
        graph = torch.cuda.CUDAGraph()
        stream = self.stream

        forward_batch = self.get_forward_batch(bs)
        # Attention backend
        self.spec_decode_worker.target_worker.model_runner.attn_backend.init_forward_metadata_capture_cuda_graph(
            bs,
            bs * self.draft_token_num,
            forward_batch.req_pool_indices,
            forward_batch.seq_lens,
            forward_batch.forward_mode,
            forward_batch.spec_info,
        )
        for runner in self.spec_decode_worker.model_runner_list:
            runner.attn_backend.init_forward_metadata_capture_cuda_graph(
                bs,
                bs * self.draft_token_num,
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                forward_batch.forward_mode,
                forward_batch.spec_info,
            )

        # Run and capture
        def run_once():
            vocab_masks = None
            if self.enable_grammar_backend():
                vocab_masks = self.vocab_masks[: (bs * self.draft_token_num)]
            output = self.spec_decode_worker.forward_decode_spec(
                forward_batch, vocab_masks
            )
            return output

        for _ in range(4):
            torch.cuda.synchronize()
            self.spec_decode_worker.model_runner.tp_group.barrier()
            run_once()
            forward_batch = self.get_forward_batch(bs)

        torch.cuda.synchronize()
        self.spec_decode_worker.model_runner.tp_group.barrier()

        global global_graph_memory_pool
        with torch.cuda.graph(graph, pool=global_graph_memory_pool, stream=stream):
            out = run_once()

        torch.cuda.synchronize()
        self.spec_decode_worker.model_runner.tp_group.barrier()

        global_graph_memory_pool = graph.pool()
        return graph, out

    def _postprocess_output_to_raw_bs(self, out, raw_bs):
        (
            logits_output,
            output_ids,
            accept_length,
            new_verified_id,
            token_list,
        ) = out
        logits_output.next_token_logits = logits_output.next_token_logits[
            : raw_bs * self.draft_token_num
        ]
        logits_output.hidden_states = logits_output.hidden_states[
            : raw_bs * self.draft_token_num
        ]
        output_ids = output_ids[: raw_bs * self.draft_token_num]
        accept_length = accept_length[:raw_bs]
        new_verified_id = new_verified_id[:raw_bs]
        token_list = token_list[:raw_bs]
        return (
            logits_output,
            output_ids,
            accept_length,
            new_verified_id,
            token_list,
        )

    def enable_grammar_backend(self):
        server_args = self.spec_decode_worker.model_runner.server_args
        return server_args.grammar_backend is not None
