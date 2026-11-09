#!/usr/bin/env python3
"""Production-shaped W4A4 MoE precision/performance regression test.

This reproduces the FluentLLM decode path:

    FlatQuant -> DispatchV2(INT4) -> GMM1 -> SplitSwiGLU
              -> FlatQuant -> GMM2 -> CombineV2

The test runs only the production MoE stream.  The deployed INT32-packed
Dispatch kernel is restricted to A2 EP128 hierarchy, so this test intentionally
rejects smaller worlds instead of silently exercising another communication
path.

EP128 example (run the same torchrun rendezvous on all eight 16-NPU nodes):

    torchrun --nproc_per_node=16 --nnodes=8 --node_rank=$NODE_RANK \
      --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
      test_dispatch_w4a4_moe.py --execution graph

Use --save-gold DIR with the known-good package, then --check-gold DIR after
changing an operator package.  One compact file is written per rank.

The self-contained numerical reference routes BF16 tokens first and applies
FlatQuant after routing.  The candidate follows the online order and routes
already packed INT4 tokens plus their FP32 scales.  Comparing the two paths
does not assume that the candidate eager INT4 Dispatch implementation is
correct.
"""

import argparse
import json
import math
import os
import statistics
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch_npu

import flash_ops  # noqa: F401 - registers torch.ops.custom.flat_quant


NZ_FORMAT = 29
NZ_CUSTOM_DTYPE = 3
FLAT_QUANT_INT4_DTYPE = 16


def decompose_dim(size):
    """Match FluentLLM's Kronecker dimension decomposition."""
    left = math.isqrt(size)
    if left * left < size:
        left += 1
    while True:
        delta = left * left - size
        right = math.isqrt(delta)
        if right * right == delta:
            return left - right, left + right
        left += 1


def pack_fp32_scale(scale):
    """Pack FP32 GMM scales in the UINT64 layout used by FluentLLM."""
    return scale.float().contiguous().view(torch.int32).to(torch.int64).npu()


def make_int4_nz_weight(
    local_experts, k, n, seed, scale_value, structured=False
):
    """Create deterministic INT4 NZ weights without a large all-expert peak."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    reciprocal_scale = torch.ones(1, dtype=torch.float32, device="npu")
    packed_experts = []
    for expert_index in range(local_experts):
        if structured:
            # Preserve the production shape/NZ conversion while avoiding the
            # several-billion-element CPU RNG workload of an EP128 smoke run.
            # Adjacent experts use different non-zero INT4 values so expert,
            # payload, and scale routing mistakes remain observable.
            magnitude = (seed + expert_index) % 6 + 1
            integer_value = magnitude if magnitude <= 3 else 3 - magnitude
            value = torch.full(
                (k, n),
                float(integer_value),
                dtype=torch.float32,
                device="npu",
            )
            integer = None
        else:
            integer = torch.randint(
                -3, 4, (k, n), dtype=torch.int8, generator=generator
            )
            value = integer.float().npu()
        packed_experts.append(
            torch_npu.npu_quantize(
                value,
                reciprocal_scale,
                None,
                torch.quint4x2,
                -1,
                False,
            ).unsqueeze(0)
        )
        del integer, value
    packed = torch.cat(packed_experts, dim=0)
    del packed_experts
    weight = torch_npu.npu_format_cast(
        packed, NZ_FORMAT, customize_dtype=NZ_CUSTOM_DTYPE
    )
    scale = pack_fp32_scale(
        torch.full(
            (local_experts, 1, n), scale_value, dtype=torch.float32
        )
    )
    return weight, scale


def get_hccl_group_name(process_group, rank):
    backend = process_group._get_backend(torch.device("npu"))
    return backend.get_hccl_comm_name(rank)


class ProductionW4A4Moe(torch.nn.Module):
    """The operator sequence used by FluentLLM's INT4 MoE communication path."""

    def __init__(
        self,
        group_ep,
        rank,
        world_size,
        num_experts,
        copy_experts,
        comm_alg,
        graph_mode,
        w1,
        w1_scale,
        w2,
        w2_scale,
        w1_left,
        w1_right,
        w2_left,
        w2_right,
        clip_ratio,
        diagnostic_stage_sync,
    ):
        super().__init__()
        self.group_ep = group_ep
        self.rank = rank
        self.world_size = world_size
        self.num_experts = num_experts
        self.copy_experts = copy_experts
        self.comm_alg = comm_alg
        self.graph_mode = graph_mode
        self.clip_ratio = clip_ratio
        self.diagnostic_stage_sync = diagnostic_stage_sync
        self.register_buffer("w1", w1)
        self.register_buffer("w1_scale", w1_scale)
        self.register_buffer("w2", w2)
        self.register_buffer("w2_scale", w2_scale)
        self.register_buffer("w1_left", w1_left)
        self.register_buffer("w1_right", w1_right)
        self.register_buffer("w2_left", w2_left)
        self.register_buffer("w2_right", w2_right)

    def stage_done(self, name):
        if not self.diagnostic_stage_sync:
            return
        torch.npu.synchronize()
        print(f"[rank {self.rank}] stage complete: {name}", flush=True)

    def flat_quant(self, x, left, right, group_list=None):
        packed, token_scale = torch.ops.custom.flat_quant(
            x.view(x.shape[0], left.shape[0], right.shape[0]),
            left,
            right,
            group_list=group_list,
            clip_ratio=self.clip_ratio,
            dst_dtype=FLAT_QUANT_INT4_DTYPE,
            group_list_type=1,
        )
        return packed.flatten(1), token_scale

    def dispatch(self, packed_x, token_scale, expert_ids, expert_weights):
        return torch_npu.npu_moe_distribute_dispatch_v2(
            x=packed_x,
            expert_ids=expert_ids,
            scales=token_scale,
            quant_mode=0,
            global_bs=0,
            expert_scales=expert_weights,
            comm_alg=self.comm_alg,
            group_ep=self.group_ep,
            group_tp=self.group_ep if self.graph_mode else "",
            ep_world_size=self.world_size,
            tp_world_size=1,
            ep_rank_id=self.rank,
            tp_rank_id=0,
            moe_expert_num=self.num_experts,
            copy_expert_num=self.copy_experts,
        )

    @staticmethod
    def grouped_matmul(x, weight, scale, token_scale, group_list):
        return torch_npu.npu_grouped_matmul(
            [x],
            [weight],
            scale=[scale],
            per_token_scale=[token_scale],
            bias=None,
            group_list=group_list,
            split_item=3,
            output_dtype=torch.bfloat16,
            group_type=0,
            group_list_type=1,
        )[0]

    def combine(
        self,
        expert_output,
        expert_ids,
        expert_weights,
        original_x,
        assist,
        ep_counts,
        tp_counts,
        expand_scales,
    ):
        return torch_npu.npu_moe_distribute_combine_v2(
            expand_x=expert_output,
            expert_ids=expert_ids.to(torch.int32),
            assist_info_for_combine=assist,
            ep_send_counts=ep_counts,
            tp_send_counts=tp_counts,
            expert_scales=expert_weights,
            expand_scales=expand_scales,
            global_bs=0,
            ori_x=original_x,
            comm_alg=self.comm_alg,
            group_ep=self.group_ep,
            group_tp=self.group_ep if self.graph_mode else "",
            ep_world_size=self.world_size,
            tp_world_size=1,
            ep_rank_id=self.rank,
            tp_rank_id=0,
            moe_expert_num=self.num_experts,
            copy_expert_num=self.copy_experts,
        )

    def capture(self, hidden_states, expert_ids, expert_weights):
        pre_x, pre_scale = self.flat_quant(
            hidden_states, self.w1_left, self.w1_right
        )
        self.stage_done("flat_quant_pre_dispatch")
        dispatch_output = self.dispatch(
            pre_x, pre_scale, expert_ids, expert_weights
        )
        self.stage_done("dispatch")
        (
            routed_x,
            routed_scale,
            assist,
            expert_counts,
            ep_counts,
            tp_counts,
            expand_scales,
        ) = dispatch_output
        group_list = expert_counts.to(torch.int64)
        gmm1 = self.grouped_matmul(
            routed_x, self.w1, self.w1_scale, routed_scale, group_list
        )
        self.stage_done("gmm1")
        activation = torch_npu.mlp_split_swiglu(
            gmm1,
            expert_counts.to(torch.int32),
            local_exp_start=0,
            local_exp_end=self.w1.shape[0],
        )
        self.stage_done("split_swiglu")
        flat2_x, flat2_scale = self.flat_quant(
            activation, self.w2_left, self.w2_right, group_list
        )
        self.stage_done("flat_quant_post_activation")
        gmm2 = self.grouped_matmul(
            flat2_x, self.w2, self.w2_scale, flat2_scale, group_list
        )
        self.stage_done("gmm2")
        output = self.combine(
            gmm2,
            expert_ids,
            expert_weights,
            hidden_states,
            assist,
            ep_counts,
            tp_counts,
            expand_scales,
        )
        self.stage_done("combine")
        return {
            "pre_flat_quant_x": pre_x,
            "pre_flat_quant_scale": pre_scale,
            "dispatch_x": routed_x,
            "dispatch_scale": routed_scale,
            "expert_counts": expert_counts,
            "gmm1": gmm1,
            "swiglu": activation,
            "flat_quant_x": flat2_x,
            "flat_quant_scale": flat2_scale,
            "gmm2": gmm2,
            "combine": output,
            "dispatch_metadata": (assist, ep_counts, tp_counts, expand_scales),
        }

    def reference_capture(self, hidden_states, expert_ids, expert_weights):
        """Build an independent gold with BF16 routing before FlatQuant."""
        dispatch_output = self.dispatch(
            hidden_states, None, expert_ids, expert_weights
        )
        (
            routed_bf16,
            _,
            assist,
            expert_counts,
            ep_counts,
            tp_counts,
            expand_scales,
        ) = dispatch_output
        group_list = expert_counts.to(torch.int64)
        routed_x, routed_scale = self.flat_quant(
            routed_bf16, self.w1_left, self.w1_right, group_list
        )
        gmm1 = self.grouped_matmul(
            routed_x, self.w1, self.w1_scale, routed_scale, group_list
        )
        activation = torch_npu.mlp_split_swiglu(
            gmm1,
            expert_counts.to(torch.int32),
            local_exp_start=0,
            local_exp_end=self.w1.shape[0],
        )
        flat2_x, flat2_scale = self.flat_quant(
            activation, self.w2_left, self.w2_right, group_list
        )
        gmm2 = self.grouped_matmul(
            flat2_x, self.w2, self.w2_scale, flat2_scale, group_list
        )
        output = self.combine(
            gmm2,
            expert_ids,
            expert_weights,
            hidden_states,
            assist,
            ep_counts,
            tp_counts,
            expand_scales,
        )
        return {
            "dispatch_x": routed_x,
            "dispatch_scale": routed_scale,
            "expert_counts": expert_counts,
            "gmm1": gmm1,
            "swiglu": activation,
            "flat_quant_x": flat2_x,
            "flat_quant_scale": flat2_scale,
            "gmm2": gmm2,
            "combine": output,
            "dispatch_metadata": (assist, ep_counts, tp_counts, expand_scales),
        }

    def forward(self, hidden_states, expert_ids, expert_weights):
        pre_x, pre_scale = self.flat_quant(
            hidden_states, self.w1_left, self.w1_right
        )
        (
            routed_x,
            routed_scale,
            assist,
            expert_counts,
            ep_counts,
            tp_counts,
            expand_scales,
        ) = self.dispatch(pre_x, pre_scale, expert_ids, expert_weights)
        group_list = expert_counts.to(torch.int64)
        gmm1 = self.grouped_matmul(
            routed_x, self.w1, self.w1_scale, routed_scale, group_list
        )
        activation = torch_npu.mlp_split_swiglu(
            gmm1,
            expert_counts.to(torch.int32),
            local_exp_start=0,
            local_exp_end=self.w1.shape[0],
        )
        flat2_x, flat2_scale = self.flat_quant(
            activation, self.w2_left, self.w2_right, group_list
        )
        gmm2 = self.grouped_matmul(
            flat2_x, self.w2, self.w2_scale, flat2_scale, group_list
        )
        return self.combine(
            gmm2,
            expert_ids,
            expert_weights,
            hidden_states,
            assist,
            ep_counts,
            tp_counts,
            expand_scales,
        )


def make_inputs(args, rank, num_experts):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed + rank * 1009)
    hidden_states = torch.randn(
        (args.batch_size, args.hidden_size),
        generator=generator,
        dtype=torch.bfloat16,
    ).npu()
    route_scores = torch.rand(
        (args.batch_size, num_experts), generator=generator
    )
    expert_ids = torch.topk(
        route_scores, args.top_k, dim=1, largest=False
    ).indices.to(torch.int32).npu()
    expert_weights = torch.softmax(
        torch.randn(
            (args.batch_size, args.top_k), generator=generator
        ),
        dim=-1,
    ).float().npu()
    return hidden_states, expert_ids, expert_weights


def snapshot(capture):
    effective_m = int(capture["expert_counts"].sum().item())
    routed_names = {
        "dispatch_x",
        "dispatch_scale",
        "gmm1",
        "swiglu",
        "flat_quant_x",
        "flat_quant_scale",
        "gmm2",
    }
    result = {}
    for name, tensor in capture.items():
        if name == "dispatch_metadata":
            continue
        if name in routed_names:
            tensor = tensor[:effective_m]
        result[name] = tensor.detach().cpu()
    result["effective_m"] = effective_m
    return result


def compare_snapshots(actual, expected, rtol, atol):
    result = {}
    for name in expected:
        if name == "effective_m":
            result[name] = {
                "passed": actual.get(name) == expected[name],
                "actual": actual.get(name),
                "expected": expected[name],
                "max_abs_error": 0.0,
            }
            continue
        if name not in actual or actual[name].shape != expected[name].shape:
            result[name] = {
                "passed": False,
                "actual_shape": (
                    list(actual[name].shape) if name in actual else None
                ),
                "expected_shape": list(expected[name].shape),
                "max_abs_error": float("inf"),
            }
            continue
        lhs = actual[name]
        rhs = expected[name]
        if lhs.is_floating_point():
            lhs_fp32 = lhs.float()
            rhs_fp32 = rhs.float()
            delta = lhs_fp32 - rhs_fp32
            passed = bool(torch.isclose(lhs, rhs, rtol=rtol, atol=atol).all())
            max_error = (
                float(delta.abs().max())
                if lhs.numel()
                else 0.0
            )
            mean_error = float(delta.abs().mean()) if lhs.numel() else 0.0
            expected_l2 = float(torch.linalg.vector_norm(rhs_fp32))
            relative_l2 = (
                float(torch.linalg.vector_norm(delta)) / expected_l2
                if expected_l2
                else (0.0 if max_error == 0.0 else float("inf"))
            )
            max_expected = (
                float(rhs_fp32.abs().max()) if rhs.numel() else 0.0
            )
        else:
            passed = torch.equal(lhs, rhs)
            max_error = (
                float((lhs.to(torch.float64) - rhs.to(torch.float64)).abs().max())
                if lhs.numel()
                else 0.0
            )
            mean_error = max_error
            relative_l2 = 0.0 if max_error == 0.0 else float("inf")
            max_expected = (
                float(rhs.to(torch.float64).abs().max())
                if rhs.numel()
                else 0.0
            )
        result[name] = {
            "passed": passed,
            "max_abs_error": max_error,
            "mean_abs_error": mean_error,
            "relative_l2_error": relative_l2,
            "max_abs_expected": max_expected,
        }
    return result


def aggregate_comparison(local_result, device):
    names = sorted(local_result)
    flags = torch.tensor(
        [1 if local_result[name]["passed"] else 0 for name in names],
        dtype=torch.int32,
        device=device,
    )
    metric_names = (
        "max_abs_error",
        "mean_abs_error",
        "relative_l2_error",
        "max_abs_expected",
    )
    metrics = torch.tensor(
        [
            [local_result[name].get(metric, 0.0) for metric in metric_names]
            for name in names
        ],
        dtype=torch.float32,
        device=device,
    )
    dist.all_reduce(flags, op=dist.ReduceOp.MIN)
    dist.all_reduce(metrics, op=dist.ReduceOp.MAX)
    return {
        name: {
            "all_ranks_passed": bool(flags[index].item()),
            **{
                f"{metric}_all_ranks": float(metrics[index, metric_index].item())
                for metric_index, metric in enumerate(metric_names)
            },
        }
        for index, name in enumerate(names)
    }


def launch_on_stream(fn, operation_stream):
    """Run on one fixed stream, matching TorchAir graph stream affinity."""
    main_stream = torch.npu.current_stream()
    operation_stream.wait_stream(main_stream)
    with torch.npu.stream(operation_stream):
        result = fn()
    main_stream.wait_stream(operation_stream)
    torch.npu.synchronize()
    return result


def percentile(values, quantile):
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] * (upper - position) + values[upper] * (
        position - lower
    )


def summarize(values):
    values = sorted(values)
    return {
        "min_us": values[0],
        "mean_us": statistics.fmean(values),
        "p50_us": percentile(values, 0.50),
        "p90_us": percentile(values, 0.90),
        "p99_us": percentile(values, 0.99),
        "max_us": values[-1],
        "samples_us": values,
    }


def measure(fn, args, world_size, operation_stream):
    def launch_once(timed):
        start = torch.npu.Event(enable_timing=True) if timed else None
        end = torch.npu.Event(enable_timing=True) if timed else None
        with torch.npu.stream(operation_stream):
            if timed:
                start.record()
            result = fn()
            if timed:
                end.record()
        torch.npu.synchronize()
        if not timed:
            return None
        # Keep asynchronous results alive until the MoE stream has completed.
        _ = result
        return start.elapsed_time(end) * 1000.0

    for _ in range(args.warmup):
        launch_once(False)
    dist.barrier()
    local_samples = [launch_once(True) for _ in range(args.iterations)]
    sample_tensor = torch.tensor(
        local_samples, dtype=torch.float32, device="npu"
    )
    gathered = torch.empty(
        world_size * args.iterations, dtype=torch.float32, device="npu"
    )
    dist.all_gather_into_tensor(gathered, sample_tensor)
    rank_samples = gathered.view(world_size, args.iterations).cpu()
    global_samples = rank_samples.max(dim=0).values.tolist()
    return summarize(global_samples)


def compile_model(model):
    import torchair
    from torchair.configs.compiler_config import CompilerConfig

    torch._dynamo.reset()
    compiler_config = CompilerConfig()
    # Keep this in lock-step with FluentLLM get_compiler_backend().  In
    # particular, production MoE weights are frozen parameters/buffers; using
    # TorchAir defaults here can produce a graph with different constant
    # handling and is not a valid online-path comparison.
    compiler_config.experimental_config.frozen_parameter = True
    compiler_config.experimental_config.tiling_schedule_optimize = True
    compiler_config.experimental_config.topology_sorting_strategy = (
        "StableRDFS"
    )
    compiler_config.ge_config.optimization_switch = (
        "InplaceAddRmsNormFusionPass:off;"
        "AddRmsNormFusionGraphPass:off"
    )
    backend = torchair.get_npu_backend(compiler_config=compiler_config)
    return torch.compile(
        model, backend=backend, dynamic=False, fullgraph=True
    )


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--hidden-size", type=int, default=8192)
    parser.add_argument("--intermediate-size", type=int, default=2048)
    parser.add_argument("--local-experts", type=int, default=7)
    parser.add_argument("--copy-experts", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--comm-alg", default="hierarchy")
    parser.add_argument("--hccl-op-expansion-mode", type=int, default=0)
    parser.add_argument(
        "--hccl-buffer-size",
        type=int,
        default=0,
        help="per-process-group HCCL buffer size in MB (online decode uses 800)",
    )
    parser.add_argument("--clip-ratio", type=float, default=1.0)
    parser.add_argument("--weight-scale", type=float, default=0.001)
    parser.add_argument(
        "--structured-weights",
        action="store_true",
        help=(
            "use deterministic per-expert constant INT4 weights for a fast "
            "EP128 smoke run while retaining production tensor shapes"
        ),
    )
    parser.add_argument(
        "--diagnostic-stage-sync",
        action="store_true",
        help="synchronize and log after every operator in eager capture",
    )
    parser.add_argument("--execution", choices=("eager", "graph"), default="graph")
    parser.add_argument(
        "--graph-stage-diagnostic",
        action="store_true",
        help=(
            "compile the stage-capture forward and compare every graph "
            "intermediate with eager instead of returning only Combine"
        ),
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument(
        "--allow-non-production-world",
        action="store_true",
        help=(
            "allow a single-node diagnostic run; the production W4A4 "
            "acceptance configuration remains EP128"
        ),
    )
    parser.add_argument("--rtol", type=float, default=0.0)
    parser.add_argument("--atol", type=float, default=0.0)
    parser.add_argument("--save-gold", type=Path)
    parser.add_argument("--check-gold", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--skip-stage-benchmark",
        action="store_true",
        help="only benchmark the complete chain",
    )
    args = parser.parse_args()
    positive = {
        "batch-size": args.batch_size,
        "hidden-size": args.hidden_size,
        "intermediate-size": args.intermediate_size,
        "local-experts": args.local_experts,
        "top-k": args.top_k,
        "iterations": args.iterations,
    }
    for name, value in positive.items():
        if value <= 0:
            parser.error(f"{name} must be positive")
    if args.warmup < 0:
        parser.error("warmup must be non-negative")
    if args.copy_experts < 0:
        parser.error("copy-experts must be non-negative")
    if args.hccl_buffer_size < 0:
        parser.error("hccl-buffer-size must be non-negative")
    if args.save_gold and args.check_gold:
        parser.error("--save-gold and --check-gold are mutually exclusive")
    return args


def main():
    args = parse_args()
    required_env = ("RANK", "LOCAL_RANK", "WORLD_SIZE")
    missing = [name for name in required_env if name not in os.environ]
    if missing:
        raise RuntimeError(
            "launch with torchrun; missing environment variables: "
            + ", ".join(missing)
        )

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != 128 and not args.allow_non_production_world:
        raise RuntimeError(
            "the online INT32-packed Dispatch path requires A2 EP128 "
            f"hierarchy; got WORLD_SIZE={world_size}; use "
            "--allow-non-production-world only for a non-acceptance "
            "single-node diagnostic"
        )
    if args.comm_alg != "hierarchy":
        raise ValueError(
            "the online INT32-packed Dispatch path requires "
            "--comm-alg hierarchy"
        )
    num_experts = args.local_experts * world_size
    if args.top_k > num_experts:
        raise ValueError(
            f"top-k ({args.top_k}) exceeds total experts ({num_experts}); "
            "increase WORLD_SIZE/local-experts or lower --top-k"
        )

    torch_npu.npu.set_device(local_rank)
    torch_npu.npu.config.allow_internal_format = True
    dist.init_process_group("hccl")
    process_group = dist.group.WORLD
    if args.hccl_op_expansion_mode or args.hccl_buffer_size:
        options = torch_npu._C._distributed_c10d.ProcessGroupHCCL.Options()
        hccl_config = {}
        if args.hccl_op_expansion_mode:
            hccl_config["hccl_op_expansion_mode"] = (
                args.hccl_op_expansion_mode
            )
        if args.hccl_buffer_size:
            hccl_config["hccl_buffer_size"] = args.hccl_buffer_size
        options.hccl_config = hccl_config
        process_group = dist.new_group(
            list(range(world_size)), backend="hccl", pg_options=options
        )

    try:
        group_ep = get_hccl_group_name(process_group, rank)
        w1_l, w1_r = decompose_dim(args.hidden_size)
        w2_l, w2_r = decompose_dim(args.intermediate_size)
        if rank == 0:
            print(
                "[setup] building production-shaped W4A4 weights; "
                f"EP={world_size}, experts={num_experts}, "
                f"local_experts={args.local_experts}",
                flush=True,
            )
        w1, w1_scale = make_int4_nz_weight(
            args.local_experts,
            args.hidden_size,
            args.intermediate_size * 2,
            args.seed + rank * 17,
            args.weight_scale,
            args.structured_weights,
        )
        w2, w2_scale = make_int4_nz_weight(
            args.local_experts,
            args.intermediate_size,
            args.hidden_size,
            args.seed + rank * 17 + 1,
            args.weight_scale,
            args.structured_weights,
        )
        w1_left = torch.eye(w1_l, dtype=torch.bfloat16, device="npu")
        w1_right = torch.eye(w1_r, dtype=torch.bfloat16, device="npu")
        w2_left = torch.eye(w2_l, dtype=torch.bfloat16, device="npu")
        w2_right = torch.eye(w2_r, dtype=torch.bfloat16, device="npu")
        inputs = make_inputs(args, rank, num_experts)

        eager_model = ProductionW4A4Moe(
            group_ep,
            rank,
            world_size,
            num_experts,
            args.copy_experts,
            args.comm_alg,
            args.execution == "graph",
            w1,
            w1_scale,
            w2,
            w2_scale,
            w1_left,
            w1_right,
            w2_left,
            w2_right,
            args.clip_ratio,
            args.diagnostic_stage_sync,
        ).eval()
        if args.execution == "eager":
            selected_model = eager_model
        elif args.graph_stage_diagnostic:
            selected_model = compile_model(eager_model.capture)
        else:
            selected_model = compile_model(eager_model)
        moe_stream = torch.npu.Stream()

        # The candidate follows the online INT4 order.  The independent gold
        # routes BF16 first and quantizes the already-routed tokens.
        baseline_capture = launch_on_stream(
            lambda: eager_model.capture(*inputs), moe_stream
        )
        baseline = snapshot(baseline_capture)
        reference_capture = launch_on_stream(
            lambda: eager_model.reference_capture(*inputs), moe_stream
        )
        reference = snapshot(reference_capture)
        selected_output = launch_on_stream(
            lambda: selected_model(*inputs), moe_stream
        )
        if args.graph_stage_diagnostic:
            selected_snapshot = snapshot(selected_output)
            selected_reference = reference
        else:
            selected_snapshot = {"combine": selected_output.detach().cpu()}
            selected_reference = {"combine": reference["combine"]}
        precision = {
            "eager_int4_vs_bf16_route_gold": aggregate_comparison(
                compare_snapshots(baseline, reference, args.rtol, args.atol),
                inputs[0].device,
            ),
            f"{args.execution}_vs_bf16_route_gold": aggregate_comparison(
                compare_snapshots(
                    selected_snapshot, selected_reference, args.rtol, args.atol
                ),
                inputs[0].device,
            )
        }

        gold_path = None
        if args.save_gold:
            if rank == 0:
                args.save_gold.mkdir(parents=True, exist_ok=True)
            dist.barrier()
            gold_path = args.save_gold / f"rank_{rank}.pt"
            torch.save(reference, gold_path)
        elif args.check_gold:
            gold_path = args.check_gold / f"rank_{rank}.pt"
            gold = torch.load(gold_path, map_location="cpu", weights_only=True)
            precision["frozen_gold"] = aggregate_comparison(
                compare_snapshots(baseline, gold, args.rtol, args.atol),
                inputs[0].device,
            )

        metadata = baseline_capture["dispatch_metadata"]
        expert_counts = baseline_capture["expert_counts"]
        group_list = expert_counts.to(torch.int64)
        stage_functions = {
            "flat_quant_pre_dispatch": lambda: eager_model.flat_quant(
                inputs[0], eager_model.w1_left, eager_model.w1_right
            ),
            "dispatch": lambda: eager_model.dispatch(
                baseline_capture["pre_flat_quant_x"],
                baseline_capture["pre_flat_quant_scale"],
                inputs[1],
                inputs[2],
            ),
            "gmm1": lambda: eager_model.grouped_matmul(
                baseline_capture["dispatch_x"],
                eager_model.w1,
                eager_model.w1_scale,
                baseline_capture["dispatch_scale"],
                group_list,
            ),
            "split_swiglu": lambda: torch_npu.mlp_split_swiglu(
                baseline_capture["gmm1"],
                expert_counts.to(torch.int32),
                local_exp_start=0,
                local_exp_end=args.local_experts,
            ),
            "flat_quant": lambda: eager_model.flat_quant(
                baseline_capture["swiglu"],
                eager_model.w2_left,
                eager_model.w2_right,
                group_list,
            ),
            "gmm2": lambda: eager_model.grouped_matmul(
                baseline_capture["flat_quant_x"],
                eager_model.w2,
                eager_model.w2_scale,
                baseline_capture["flat_quant_scale"],
                group_list,
            ),
            "combine": lambda: eager_model.combine(
                baseline_capture["gmm2"],
                inputs[1],
                inputs[2],
                inputs[0],
                *metadata,
            ),
            "end_to_end": lambda: selected_model(*inputs),
        }
        if args.skip_stage_benchmark:
            stage_functions = {"end_to_end": stage_functions["end_to_end"]}

        performance = {}
        for stage_name, stage_fn in stage_functions.items():
            if rank == 0:
                print(f"[bench] moe_stream: {stage_name}", flush=True)
            performance[stage_name] = measure(
                stage_fn, args, world_size, moe_stream
            )

        all_precision_passed = all(
            item["all_ranks_passed"]
            for comparison in precision.values()
            for item in comparison.values()
        )
        result = {
            "passed": all_precision_passed,
            "configuration": {
                "execution": args.execution,
                "gold_path": "bf16_dispatch_then_flat_quant",
                "graph_stage_diagnostic": args.graph_stage_diagnostic,
                "production_ep128": world_size == 128,
                "ep_world_size": world_size,
                "batch_size_per_rank": args.batch_size,
                "global_batch_size": args.batch_size * world_size,
                "hidden_size": args.hidden_size,
                "intermediate_size": args.intermediate_size,
                "top_k": args.top_k,
                "num_experts": num_experts,
                "local_experts": args.local_experts,
                "structured_weights": args.structured_weights,
                "diagnostic_stage_sync": args.diagnostic_stage_sync,
                "copy_experts": args.copy_experts,
                "comm_alg": args.comm_alg,
                "hccl_op_expansion_mode": args.hccl_op_expansion_mode,
                "hccl_buffer_size": args.hccl_buffer_size,
                "w1_shape": [
                    args.local_experts,
                    args.hidden_size,
                    args.intermediate_size * 2,
                ],
                "w2_shape": [
                    args.local_experts,
                    args.intermediate_size,
                    args.hidden_size,
                ],
                "w1_kronecker": [w1_l, w1_r],
                "w2_kronecker": [w2_l, w2_r],
                "warmup": args.warmup,
                "iterations": args.iterations,
            },
            "precision": precision,
            "performance": performance,
            "gold_path_rank0": (
                str(args.save_gold / "rank_0.pt")
                if args.save_gold
                else str(args.check_gold / "rank_0.pt")
                if args.check_gold
                else None
            ),
        }
        if rank == 0:
            encoded = json.dumps(result, indent=2, sort_keys=True)
            print("W4A4_MOE_RESULT " + encoded, flush=True)
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(encoded + "\n", encoding="utf-8")
        dist.barrier()
        if not all_precision_passed:
            raise AssertionError("W4A4 MoE precision regression detected")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
