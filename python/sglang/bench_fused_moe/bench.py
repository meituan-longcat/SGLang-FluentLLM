from dataclasses import dataclass

import torch

from sglang.srt.layers.moe.config import DispatcherType
from sglang.srt.layers.moe.gemms.fp8.triton import TritonGemmWrapper
from sglang.srt.ops.topk import TopK, TopKOutputFormat


@dataclass
class QuantizationConfig:
    weight_block_size = [128, 128]


class MockedTPMOELayer:
    def __init__(self, num_experts, hidden_size, intermediate_size, tp_size, topk, correction_bias=None, zero_expert_num=0, quant_config=None):
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.tp_size = tp_size
        self.top_k = topk
        self.mock_weight_and_scale(num_experts, hidden_size, intermediate_size // tp_size)
        self.triton_executor = TritonGemmWrapper(topk, num_experts, self, quant_config).get_executor(DispatcherType.TP, "silu")
        self.topk = TopK(
            top_k=topk,
            layer_id=0,
            renormalize=False,
            use_grouped_topk=False,
            output_format=TopKOutputFormat.STANDARD,
            zero_expert_num=zero_expert_num,
            topk_indices_dtype=torch.int32,
            correction_bias=correction_bias,
            routed_scaling_factor=6.0,
        )

    def mock_weight_and_scale(self, num_experts, hidden_size, shard_intermediate_size):
        # E, N, K
        self.w13_weight = torch.randn(num_experts, shard_intermediate_size * 2, hidden_size, device="cuda").normal_(0, 0.01).to(torch.float8_e4m3fn)
        self.w2_weight = torch.randn(num_experts, hidden_size, shard_intermediate_size, device="cuda").normal_(0, 0.01).to(torch.float8_e4m3fn)

        # E, N
        self.w13_weight_scale_inv = torch.randn(
            (num_experts, shard_intermediate_size * 2 // 128, hidden_size // 128),
            dtype=torch.float32,
            device="cuda",
        ).normal_(0, 0.01)
        self.w2_weight_scale_inv = torch.randn(
            (num_experts, hidden_size // 128, shard_intermediate_size // 128), dtype=torch.float32, device="cuda"
        ).normal_(0, 0.01)
        self.w13_input_scale = None
        self.w2_input_scale = None

    def forward_triton(
        self,
        hidden_states: torch.Tensor,
        topk_output,
        num_global_tokens: int,
        max_num_tokens_per_gpu: int,
    ):
        return self.triton_executor.forward(self, hidden_states, topk_output, num_global_tokens, max_num_tokens_per_gpu)


def update_static_inputs(static_hidden_states, static_topk_output, new_hidden_states, new_topk_output):
    """Update static input tensors with new data for graph replay"""
    static_hidden_states.copy_(new_hidden_states)
    for field in static_topk_output._fields:
        static_tensor = getattr(static_topk_output, field)
        new_tensor = getattr(new_topk_output, field)
        static_tensor.copy_(new_tensor)


if __name__ == "__main__":
    # Initialize parameters and layer
    num_experts = 256
    hidden_size = 3072
    intermediate_size = 1024
    tp_size = 4
    topk = 12
    zero_expert_num = 128
    correction_bias = torch.randn((zero_expert_num + num_experts,), device="cuda")
    layer = MockedTPMOELayer(num_experts, hidden_size, intermediate_size, tp_size, topk, correction_bias, zero_expert_num, QuantizationConfig())
    # num_tokens = [1, 2, 4, 8, 16, 32, 64, 512, 1024, 2048, 4096, 8192, 16384]
    num_tokens = [4096]
    for num_token in num_tokens:
        # Create input data
        hidden_states = torch.randn(num_token, hidden_size, dtype=torch.bfloat16).cuda().normal_(0, 0.01)
        router_logits = torch.randn(num_token, (num_experts + zero_expert_num), dtype=torch.float32).cuda().normal_(0, 0.01)
        topk_output = layer.topk(hidden_states, router_logits)
        warmup = 10
        # ========== Capture CUDA Graphs ==========
        # Triton Graph
        graph_triton = torch.cuda.CUDAGraph()
        static_hidden_triton = hidden_states.clone()
        static_topk_triton = type(topk_output)(*[t.clone() for t in topk_output])
        torch.cuda.synchronize()
        for _ in range(warmup):
            layer.forward_triton(hidden_states.clone(), topk_output, None, None)

        torch.cuda.synchronize()
        with torch.cuda.graph(graph_triton):
            static_out_triton = layer.forward_triton(
                hidden_states=static_hidden_triton,
                topk_output=static_topk_triton,
                num_global_tokens=None,
                max_num_tokens_per_gpu=None,
            )

        # ========== Benchmark Performance ==========
        iters = 100
        results = {}
        torch.cuda.synchronize()
        for _ in range(warmup):
            layer.forward_triton(hidden_states.clone(), topk_output, None, None)
        # Benchmark Triton Graph
        update_static_inputs(static_hidden_triton, static_topk_triton, hidden_states, topk_output)
        for _ in range(warmup):
            graph_triton.replay()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start.record()
        """
        with profile(
            activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU],
            schedule=torch.profiler.schedule(wait=0, warmup=1, active=5),
            record_shapes=True,
            with_stack=True
        ) as prof:
        """
        for _ in range(iters):
            graph_triton.replay()
        end.record()
        torch.cuda.synchronize()
        results["triton_graph"] = start.elapsed_time(end) / iters
        """
        prof.export_chrome_trace("./trace_triton.json")
        """
        results["aok_graph"] = start.elapsed_time(end) / iters
        # ========== Print Results ==========
        print(f"Triton Fused Moe: {results['triton_graph']:.4f} ms  num_tokens: {num_token}")
