#!/usr/bin/env python3
"""Validate accuracy and performance of routed-MoE W4A4 GroupedMatmul.

Only GroupedMatmul is measured and validated. Quantize and FormatCast are used
once to construct INT4/NZ inputs outside the timed region; FlatQuant and other
MoE operators are not called. The ``tuning`` subcommand benchmarks direct eager
calls and checks every tuning choice against the untuned output. The
``production`` subcommand covers the two online GMM shapes with a static graph
capacity, TorchAir compilation, bit-exact eager or frozen-gold checks, and
host/device timing.

Examples:

    python test_w4a4_grouped_matmul.py tuning --k 2048 --n 8192
    python test_w4a4_grouped_matmul.py production --shape both
"""

import argparse
import gc
import json
import statistics
import time

import torch
import torch_npu


PRODUCTION_SHAPES = {
    "gmm1": (8192, 4096),
    "gmm2": (2048, 8192),
}
_TORCHAIR = None


def parse_ints(value):
    return [int(item) for item in value.split(",") if item]


def pack_fp32_scale(scale):
    return scale.float().contiguous().view(torch.int32).to(torch.int64).npu()


def make_int4(value, axis):
    unit = torch.ones(1, dtype=torch.float32, device="npu")
    return torch_npu.npu_quantize(
        value.float().npu(), unit, None, torch.quint4x2, axis, False
    )


def prepare_case(counts, k, n, transpose_weight, seed, capacity_m=None):
    torch.manual_seed(seed)
    expert_num = len(counts)
    effective_m = sum(counts)
    storage_m = effective_m if capacity_m is None else capacity_m
    if storage_m < effective_m:
        raise ValueError(
            f"capacity-m ({storage_m}) must be >= routed M ({effective_m})"
        )

    x_integer = torch.randint(-7, 8, (storage_m, k), dtype=torch.int8)
    x = make_int4(x_integer, -1)
    if transpose_weight:
        weight_integer = torch.randint(
            -7, 8, (expert_num, n, k), dtype=torch.int8
        )
        weight = make_int4(weight_integer, -2)
    else:
        weight_integer = torch.randint(
            -7, 8, (expert_num, k, n), dtype=torch.int8
        )
        weight = make_int4(weight_integer, -1)
    weight = torch_npu.npu_format_cast(weight, 29, customize_dtype=3)
    if transpose_weight:
        weight = weight.transpose(1, 2)

    scale = pack_fp32_scale(
        torch.ones((expert_num, 1, n), dtype=torch.float32)
    )
    per_token_scale = torch.ones(storage_m, dtype=torch.float32, device="npu")
    group_list = torch.tensor(counts, dtype=torch.int64, device="npu")
    torch.npu.synchronize()
    return x, weight, scale, per_token_scale, group_list


def grouped_matmul(case, tuning=None, transpose_in_graph=False):
    x, weight, scale, per_token_scale, group_list = case
    if transpose_in_graph:
        weight = weight.transpose(1, 2)
    return torch_npu.npu_grouped_matmul(
        [x],
        [weight],
        scale=[scale],
        per_token_scale=[per_token_scale],
        group_list=group_list,
        group_list_type=1,
        group_type=0,
        split_item=3,
        act_type=0,
        output_dtype=torch.bfloat16,
        tuning_config=None if tuning is None else [tuning],
    )[0]


def measure_eager(case, tuning, warmup, iterations, rounds):
    for _ in range(warmup):
        grouped_matmul(case, tuning)
    torch.npu.synchronize()

    samples = []
    for _ in range(rounds):
        start = time.perf_counter_ns()
        for _ in range(iterations):
            grouped_matmul(case, tuning)
        torch.npu.synchronize()
        samples.append((time.perf_counter_ns() - start) / iterations / 1_000.0)
    samples.sort()
    return {
        "host_median_us": statistics.median(samples),
        "host_min_us": samples[0],
        "host_max_us": samples[-1],
        "host_samples_us": samples,
    }


def run_tuning(args):
    case = prepare_case(
        args.counts,
        args.k,
        args.n,
        args.transpose_weight,
        args.seed,
        args.capacity_m,
    )
    effective_m = sum(args.counts)
    reference = grouped_matmul(case)
    torch.npu.synchronize()
    reference_cpu = reference[:effective_m].cpu()

    if args.check_output:
        gold = torch.load(args.check_output, map_location="cpu", weights_only=True)
        torch.testing.assert_close(reference_cpu, gold, rtol=0.0, atol=0.0)
    if args.save_output:
        torch.save(reference_cpu, args.save_output)

    results = []
    for tuning in args.tunings:
        normalized_tuning = None if tuning < 0 else tuning
        if normalized_tuning is not None and normalized_tuning > (
            args.capacity_m or effective_m
        ):
            continue
        output = grouped_matmul(case, normalized_tuning)
        torch.npu.synchronize()
        torch.testing.assert_close(
            output[:effective_m].cpu(), reference_cpu, rtol=0.0, atol=0.0
        )
        results.append(
            {
                "tuning": normalized_tuning,
                **measure_eager(
                    case,
                    normalized_tuning,
                    args.warmup,
                    args.iterations,
                    args.rounds,
                ),
            }
        )

    print(
        json.dumps(
            {
                "mode": "tuning",
                "accuracy": "bit-exact",
                "shape": {
                    "counts": args.counts,
                    "effective_m": effective_m,
                    "storage_m": args.capacity_m or effective_m,
                    "k": args.k,
                    "n": args.n,
                    "transpose_weight": args.transpose_weight,
                },
                "results": results,
            },
            indent=2,
        )
    )


class ProductionCapacityGmm(torch.nn.Module):
    def __init__(self, aicore_num, transpose_in_graph):
        super().__init__()
        self.aicore_num = aicore_num
        self.transpose_in_graph = transpose_in_graph

    def forward(self, x, weight, scale, per_token_scale, group_list):
        case = (x, weight, scale, per_token_scale, group_list)
        if self.aicore_num is None:
            return grouped_matmul(case, transpose_in_graph=self.transpose_in_graph)
        with _TORCHAIR.scope.limit_core_num(self.aicore_num, self.aicore_num * 2):
            return grouped_matmul(case, transpose_in_graph=self.transpose_in_graph)


def measure_graph(compiled, cases, warmup, iterations, rounds):
    for iteration in range(warmup):
        compiled(*cases[iteration % len(cases)])
    torch.npu.synchronize()

    host_samples = []
    device_samples = []
    for _ in range(rounds):
        start_event = torch.npu.Event(enable_timing=True)
        end_event = torch.npu.Event(enable_timing=True)
        start = time.perf_counter_ns()
        start_event.record()
        for iteration in range(iterations):
            compiled(*cases[iteration % len(cases)])
        end_event.record()
        torch.npu.synchronize()
        host_samples.append(
            (time.perf_counter_ns() - start) / iterations / 1_000.0
        )
        device_samples.append(
            start_event.elapsed_time(end_event) * 1_000.0 / iterations
        )
    host_samples.sort()
    device_samples.sort()
    return {
        "host_median_us": statistics.median(host_samples),
        "host_samples_us": host_samples,
        "device_median_us": statistics.median(device_samples),
        "device_samples_us": device_samples,
    }


def run_production_case(args, shape_name, torchair_module, compiler_config_type):
    k, n = PRODUCTION_SHAPES[shape_name]
    seed = args.seed if shape_name == "gmm1" else args.seed + 1
    effective_m = sum(args.counts)
    transpose_in_graph = args.weight_layout == "graph-transpose"
    case = prepare_case(
        args.counts,
        k,
        n,
        transpose_weight=transpose_in_graph,
        seed=seed,
        capacity_m=args.capacity_m,
    )
    if transpose_in_graph:
        x, transposed_weight, scale, per_token_scale, group_list = case
        case = (
            x,
            transposed_weight.transpose(1, 2),
            scale,
            per_token_scale,
            group_list,
        )

    eager = None
    if not args.graph_gold_only:
        eager = grouped_matmul(case, transpose_in_graph=transpose_in_graph)
        torch.npu.synchronize()

    graph_model = ProductionCapacityGmm(
        args.aicore_num,
        transpose_in_graph,
    ).npu().eval()
    with torch.no_grad():
        torch._dynamo.reset()
        backend = torchair_module.get_npu_backend(
            compiler_config=compiler_config_type()
        )
        compiled = torch.compile(
            graph_model,
            backend=backend,
            dynamic=False,
            fullgraph=True,
        )
        graph = compiled(*case)
        torch.npu.synchronize()

        benchmark_cases = [case]
        if args.weight_cycle > 1:
            x, weight, scale, per_token_scale, group_list = case
            for _ in range(args.weight_cycle - 1):
                benchmark_cases.append(
                    (
                        x.clone() if args.cycle_x else x,
                        weight.clone(),
                        scale,
                        per_token_scale,
                        group_list,
                    )
                )
            torch.npu.synchronize()
        latency = measure_graph(
            compiled,
            benchmark_cases,
            args.warmup,
            args.iterations,
            args.rounds,
        )

    valid_graph = graph[:effective_m].cpu()
    if eager is not None:
        torch.testing.assert_close(
            valid_graph,
            eager[:effective_m].cpu(),
            rtol=0.0,
            atol=0.0,
        )

    output_path = (
        f"{args.save_output_prefix}.{shape_name}.pt"
        if args.save_output_prefix
        else None
    )
    gold_path = (
        f"{args.check_output_prefix}.{shape_name}.pt"
        if args.check_output_prefix
        else None
    )
    if gold_path:
        gold = torch.load(gold_path, map_location="cpu", weights_only=True)
        torch.testing.assert_close(valid_graph, gold, rtol=0.0, atol=0.0)
    if output_path:
        torch.save(valid_graph, output_path)

    result = {
        "shape": shape_name,
        "counts": args.counts,
        "effective_m": effective_m,
        "capacity_m": args.capacity_m,
        "k": k,
        "n": n,
        "weight_layout": args.weight_layout,
        "aicore_num": args.aicore_num,
        "weight_cycle": args.weight_cycle,
        "cycle_x": args.cycle_x,
        "accuracy": "bit-exact",
        **latency,
    }

    del graph, eager, compiled, graph_model, case, benchmark_cases
    gc.collect()
    torch.npu.empty_cache()
    return result


def run_production(args):
    global _TORCHAIR

    import torchair
    from torchair.configs.compiler_config import CompilerConfig

    _TORCHAIR = torchair

    shape_names = (
        tuple(PRODUCTION_SHAPES) if args.shape == "both" else (args.shape,)
    )
    results = [
        run_production_case(
            args,
            shape_name,
            torchair,
            CompilerConfig,
        )
        for shape_name in shape_names
    ]
    print(
        json.dumps(
            {
                "mode": "production",
                "results": results,
            },
            indent=2,
        )
    )


def add_common_arguments(parser, default_counts):
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--counts",
        type=parse_ints,
        default=parse_ints(default_counts),
        help="comma-separated per-expert routed token counts",
    )
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=7)


def build_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    tuning = subparsers.add_parser(
        "tuning",
        help="benchmark eager tuning choices and check bit-exact output",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_common_arguments(tuning, "8,8,8,8,8,8,8")
    tuning.add_argument("--k", type=int, default=2048)
    tuning.add_argument("--n", type=int, default=8192)
    tuning.add_argument("--transpose-weight", action="store_true")
    tuning.add_argument("--capacity-m", type=int)
    tuning.add_argument(
        "--tunings",
        type=parse_ints,
        default=parse_ints("-1,1,4,8,16,32,64,128"),
    )
    tuning.add_argument("--save-output")
    tuning.add_argument("--check-output")

    production = subparsers.add_parser(
        "production",
        help="validate and benchmark the online static-capacity graph",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_common_arguments(production, "14,14,14,14,14,13,13")
    production.add_argument("--capacity-m", type=int, default=7168)
    production.add_argument(
        "--shape",
        choices=("gmm1", "gmm2", "both"),
        default="both",
    )
    production.add_argument("--aicore-num", type=int, default=12)
    production.add_argument("--weight-cycle", type=int, default=1)
    production.add_argument("--cycle-x", action="store_true")
    production.add_argument(
        "--weight-layout",
        choices=("production", "graph-transpose"),
        default="production",
    )
    production.add_argument("--graph-gold-only", action="store_true")
    production.add_argument("--save-output-prefix")
    production.add_argument("--check-output-prefix")
    return parser


def validate_arguments(parser, args):
    if not args.counts or any(count < 0 for count in args.counts):
        parser.error("counts must contain non-negative integers")
    if args.warmup < 0 or args.iterations <= 0 or args.rounds <= 0:
        parser.error("warmup must be non-negative; iterations and rounds must be positive")
    if args.mode == "tuning":
        if args.k <= 0 or args.n <= 0:
            parser.error("k and n must be positive")
        if args.capacity_m is not None and args.capacity_m < sum(args.counts):
            parser.error("capacity-m must be >= sum(counts)")
    else:
        if args.capacity_m < sum(args.counts):
            parser.error("capacity-m must be >= sum(counts)")
        if args.weight_cycle < 1:
            parser.error("weight-cycle must be positive")
        if args.aicore_num is not None and args.aicore_num <= 0:
            parser.error("aicore-num must be positive")
        if args.graph_gold_only and not args.check_output_prefix:
            parser.error("graph-gold-only requires --check-output-prefix")


def main():
    parser = build_parser()
    args = parser.parse_args()
    validate_arguments(parser, args)
    torch.npu.set_device(args.device)
    torch_npu.npu.config.allow_internal_format = True
    if args.mode == "tuning":
        run_tuning(args)
    else:
        run_production(args)


if __name__ == "__main__":
    main()
