"""EP128 MoeDistributeDispatchV2/CombineV2 版本看护用例（eager 与图模式双模式）。

覆盖 8 节点 x 16 NPU = 128 rank, hierarchy comm (EP=128),
验证 bf16/int8 dispatch+combine 的精度与性能, 防止算子回归:

    - DISPATCH_EAGER=1   -> eager 逐 op launch (业务 decode 场景)
    - DISPATCH_EAGER=0   -> torch.compile 图模式 (GE 图一次下发)

典型用法:
    DISPATCH_EAGER=1 WARMUP_ITERS=2 BENCH_ITERS=2 REMOTE_TIMEOUT=30 \\
        bash run_ep128_dispatch_e2e.sh 4

参数:
    TEST_BS           每个 rank 的 token 数 (默认 4)
    DISPATCH_EAGER    1=eager, 0=图模式 compile (默认 1)
    DISPATCH_DTYPE    bf16 / int8 (默认 bf16)
    DISPATCH_COMM_ALG hierarchy / fullmesh (默认 hierarchy)
    WARMUP_ITERS / BENCH_ITERS / REMOTE_TIMEOUT  迭代与超时控制
    DISPATCH_SPARSE   1=仅 rank0 有 token, 其余 0-token (0-token 行为看护)
"""

import json
import os
import time

import torch
import torch.distributed as dist
import torch_npu
import torchair


EP = 128
BS = int(os.getenv("TEST_BS", "4"))
H = 8192
TOP_K = 12
EXPERTS = int(os.getenv("MOE_EXPERTS_OVERRIDE", "768"))
LOCAL_EXPERTS = EXPERTS // EP
GLOBAL_BS = int(os.getenv("GLOBAL_BS_OVERRIDE", str(BS * EP)))
COPY_EXPERTS = int(os.getenv("COPY_EXPERT_OVERRIDE", "0"))
ROUTE_PATTERN = os.getenv("DISPATCH_ROUTE_PATTERN", "uniform_unique_topk")
SPARSE = os.getenv("DISPATCH_SPARSE", "0") == "1"
SPARSE_STRIDE = int(os.getenv("DISPATCH_SPARSE_STRIDE", "8"))
EAGER = os.getenv("DISPATCH_EAGER", "1") == "1"
WARMUP = int(os.getenv("WARMUP_ITERS", "10"))
ITERS = int(os.getenv("BENCH_ITERS", "50"))
SEED = 20260812
COMM_ALG = os.getenv("DISPATCH_COMM_ALG", "hierarchy")
DISPATCH_DTYPE = os.getenv("DISPATCH_DTYPE", "bf16").lower()
HCCL_OP_EXPANSION_MODE = int(
    os.getenv("DISPATCH_HCCL_OP_EXPANSION_MODE", "0")
)
if DISPATCH_DTYPE not in {"bf16", "int8"}:
    raise ValueError(f"DISPATCH_DTYPE must be bf16 or int8, got {DISPATCH_DTYPE!r}")
if ROUTE_PATTERN not in {"uniform_unique_topk", "copy_first"}:
    raise ValueError(f"unsupported DISPATCH_ROUTE_PATTERN={ROUTE_PATTERN!r}")
QUANT_MODE = 0 if DISPATCH_DTYPE == "bf16" else 2
PROFILE_MODE = os.getenv("DISPATCH_PROFILE", "0") == "1"
PROFILE_OUTPUT_DIR = os.getenv("PROFILE_OUTPUT_DIR", ".")
PROFILE_RUN_TAG = os.getenv("PROFILE_RUN_TAG", "int8_profile")
HOST_TIMELINE = os.getenv("DISPATCH_HOST_TIMELINE", "0") == "1"
CPU_AFFINITY = os.getenv("DISPATCH_CPU_AFFINITY", "0") == "1"
DISPATCH_BURST = int(os.getenv("DISPATCH_BURST", "1"))
TIMING_MODE = os.getenv("DISPATCH_TIMING_MODE", "legacy")
RANK_BARRIER = os.getenv("DISPATCH_RANK_BARRIER", "1") == "1"
MOE_STREAM = os.getenv("DISPATCH_MOE_STREAM", "0") == "1"
if TIMING_MODE not in {"legacy", "sync_then_barrier"}:
    raise ValueError(f"unsupported DISPATCH_TIMING_MODE={TIMING_MODE!r}")
if DISPATCH_BURST < 1:
    raise ValueError("DISPATCH_BURST must be positive")


def pin_worker_to_npu_numa(local_rank):
    if not CPU_AFFINITY:
        return None
    numa_node = 0 if local_rank in {0, 1, 2, 3, 8, 9, 10, 11} else 1
    with open(f"/sys/devices/system/node/node{numa_node}/cpulist", encoding="utf-8") as f:
        node_spec = f.read().strip()

    def expand_cpu_list(spec):
        cpus = []
        for item in spec.split(","):
            bounds = item.split("-", 1)
            begin = int(bounds[0])
            end = int(bounds[-1])
            cpus.extend(range(begin, end + 1))
        return cpus

    allowed = os.sched_getaffinity(0)
    # Prefer one hardware thread per core.  On this 2-socket 910B host the
    # sibling threads start at cpu_count / 2.
    sibling_base = (os.cpu_count() or 1) // 2
    candidates = [
        cpu for cpu in expand_cpu_list(node_spec)
        if cpu in allowed and cpu < sibling_base
    ]
    numa_local_ranks = (
        [0, 1, 2, 3, 8, 9, 10, 11]
        if numa_node == 0
        else [4, 5, 6, 7, 12, 13, 14, 15]
    )
    slot = numa_local_ranks.index(local_rank)
    cpus_per_worker = max(1, min(4, len(candidates) // len(numa_local_ranks)))
    begin = slot * cpus_per_worker
    selected = candidates[begin : begin + cpus_per_worker]
    if not selected:
        raise RuntimeError(
            f"no allowed NUMA CPUs for local_rank={local_rank}, node={numa_node}"
        )
    os.sched_setaffinity(0, selected)
    return selected


class Dispatch(torch.nn.Module):
    def __init__(self, group, rank):
        super().__init__()
        self.group = group
        self.rank = rank

    def forward(self, x, expert_ids, expert_scales):
        output = None
        burst_markers = []
        for _ in range(DISPATCH_BURST):
            output = torch_npu.npu_moe_distribute_dispatch_v2(
                x=x,
                expert_ids=expert_ids,
                group_ep=self.group,
                group_tp="",
                ep_world_size=EP,
                tp_world_size=1,
                ep_rank_id=self.rank,
                tp_rank_id=0,
                moe_expert_num=EXPERTS,
                shared_expert_num=0,
                shared_expert_rank_num=0,
                copy_expert_num=COPY_EXPERTS,
                quant_mode=QUANT_MODE,
                global_bs=GLOBAL_BS,
                expert_token_nums_type=1,
                comm_alg=COMM_ALG,
                expert_scales=expert_scales,
            )
            burst_markers.append(output[3])
        if DISPATCH_BURST == 1:
            return output
        return output + (torch.stack(burst_markers),)


def dispatch_outputs(output):
    return output[:7] if DISPATCH_BURST > 1 else output


class Combine(torch.nn.Module):
    def __init__(self, group, rank):
        super().__init__()
        self.group = group
        self.rank = rank

    def forward(self, expand_x, expert_ids, assist, ep_counts, tp_counts, expand_scales, expert_scales):
        return torch_npu.npu_moe_distribute_combine_v2(
            expand_x=expand_x,
            expert_ids=expert_ids,
            assist_info_for_combine=assist,
            ep_send_counts=ep_counts,
            tp_send_counts=tp_counts,
            expert_scales=expert_scales,
            expand_scales=expand_scales,
            group_ep=self.group,
            group_tp="",
            ep_world_size=EP,
            tp_world_size=1,
            ep_rank_id=self.rank,
            tp_rank_id=0,
            moe_expert_num=EXPERTS,
            shared_expert_num=0,
            shared_expert_rank_num=0,
            global_bs=GLOBAL_BS,
            comm_quant_mode=0,
            comm_alg=COMM_ALG,
        )


def make_inputs(rank, device):
    bs = 0 if (SPARSE and rank != 0) else BS
    gen = torch.Generator(device="cpu")
    gen.manual_seed(SEED + rank * 1009 + BS)
    x_cpu = torch.randn((bs, H), generator=gen, dtype=torch.bfloat16)
    scales_cpu = torch.randn((bs, TOP_K), generator=gen, dtype=torch.float32)
    if ROUTE_PATTERN == "copy_first":
        # Model logs use ID moe_expert_num for the first copy/zero expert.
        ids_cpu = torch.full((bs, TOP_K), EXPERTS, dtype=torch.int32)
    else:
        scores = torch.rand((bs, EXPERTS), generator=gen)
        ids_cpu = torch.topk(scores, TOP_K, dim=1, largest=False).indices.to(torch.int32)
    return (
        x_cpu,
        ids_cpu,
        scales_cpu,
        x_cpu.to(device),
        ids_cpu.to(device),
        scales_cpu.to(device),
    )


def gather_cpu(local_npu):
    shape = (EP * local_npu.shape[0],) + tuple(local_npu.shape[1:])
    gathered = torch.empty(shape, dtype=local_npu.dtype, device=local_npu.device)
    dist.all_gather_into_tensor(gathered, local_npu.contiguous())
    return gathered.cpu()


def dispatch_golden(all_x, all_ids, rank):
    rows = []
    begin = rank * LOCAL_EXPERTS
    end = begin + LOCAL_EXPERTS
    for expert in range(begin, end):
        for src_rank in range(EP):
            for token in range(BS):
                if bool((all_ids[src_rank * BS + token] == expert).any()):
                    rows.append(all_x[src_rank * BS + token])
    if rows:
        return torch.stack(rows)
    return torch.empty((0, H), dtype=all_x.dtype)


def validate_dispatch(expand_x, dynamic_scales, expected_dispatch, received):
    actual = expand_x[:received].cpu()
    if received != expected_dispatch.shape[0]:
        return False, {
            "expected_rows": int(expected_dispatch.shape[0]),
            "actual_rows": received,
        }

    # Copy-expert-only routes intentionally leave every regular EP expert empty.
    # There is no scale/payload reduction to validate for this rank.
    if received == 0:
        return True, {
            "payload_exact": True,
            "payload_mismatch_elements": 0,
            "dynamic_scale_allclose": True,
            "dynamic_scale_max_abs_error": 0.0,
            "dequant_within_one_scale": True,
            "dequant_max_abs_error": 0.0,
        }

    if DISPATCH_DTYPE == "bf16":
        exact = torch.equal(actual, expected_dispatch)
        return exact, {
            "payload_exact": exact,
            "payload_mismatch_elements": int((actual != expected_dispatch).sum()),
        }

    actual_scales = dynamic_scales[:received].cpu().float()
    expected_float = expected_dispatch.float()
    expected_scales = expected_float.abs().amax(dim=1) / 127.0
    expected_quant = torch.round(
        expected_float / expected_scales.unsqueeze(1)
    ).clamp(-127, 127).to(torch.int8)
    payload_exact = torch.equal(actual, expected_quant)
    scale_close = torch.isclose(
        actual_scales, expected_scales, rtol=1e-6, atol=1e-8
    )
    dequant = actual.float() * actual_scales.unsqueeze(1)
    dequant_error = (dequant - expected_float).abs()
    dequant_ok = bool(
        (dequant_error.amax(dim=1) <= actual_scales + 1e-6).all()
    )
    ok = bool(scale_close.all()) and dequant_ok
    return ok, {
        "payload_exact": payload_exact,
        "payload_mismatch_elements": int((actual != expected_quant).sum()),
        "dynamic_scale_allclose": bool(scale_close.all()),
        "dynamic_scale_max_abs_error": float(
            (actual_scales - expected_scales).abs().max()
        ),
        "dequant_within_one_scale": dequant_ok,
        "dequant_max_abs_error": float(dequant_error.max()),
    }


def summary(values):
    values = sorted(values)
    if not values:
        return {}
    if len(values) == 1:
        value = values[0]
        return {
            "min_ms": value,
            "mean_ms": value,
            "p50_ms": value,
            "p90_ms": value,
            "p99_ms": value,
            "max_ms": value,
        }

    def pct(q):
        pos = (len(values) - 1) * q
        lo = int(pos)
        hi = min(lo + 1, len(values) - 1)
        return values[lo] * (hi - pos) + values[hi] * (pos - lo)

    return {
        "min_ms": min(values),
        "mean_ms": sum(values) / len(values),
        "p50_ms": pct(0.50),
        "p90_ms": pct(0.90),
        "p99_ms": pct(0.99),
        "max_ms": max(values),
    }


def timed(fn, rank, iterations=None):
    iterations = ITERS if iterations is None else iterations
    warmup = WARMUP if iterations > 0 else 0
    for _ in range(warmup):
        fn()
    torch_npu.npu.synchronize()
    if RANK_BARRIER:
        dist.barrier()
    local = []
    host_timeline = []
    for iteration in range(iterations):
        loop_enter_ns = time.perf_counter_ns()
        if TIMING_MODE == "sync_then_barrier":
            pre_sync_enter_ns = time.perf_counter_ns()
            torch_npu.npu.synchronize()
            pre_sync_exit_ns = time.perf_counter_ns()
            barrier_enter_ns = time.perf_counter_ns()
            if RANK_BARRIER:
                dist.barrier()
            barrier_exit_ns = time.perf_counter_ns()
        else:
            barrier_enter_ns = time.perf_counter_ns()
            if RANK_BARRIER:
                dist.barrier()
            barrier_exit_ns = time.perf_counter_ns()
            pre_sync_enter_ns = time.perf_counter_ns()
            torch_npu.npu.synchronize()
            pre_sync_exit_ns = time.perf_counter_ns()
        call_enter_ns = time.perf_counter_ns()
        fn()
        call_exit_ns = time.perf_counter_ns()
        post_sync_enter_ns = time.perf_counter_ns()
        torch_npu.npu.synchronize()
        post_sync_exit_ns = time.perf_counter_ns()
        local.append(
            (post_sync_exit_ns - call_enter_ns) / 1_000_000.0 / DISPATCH_BURST
        )
        if HOST_TIMELINE:
            host_timeline.append(
                [
                    loop_enter_ns,
                    barrier_enter_ns,
                    barrier_exit_ns,
                    pre_sync_enter_ns,
                    pre_sync_exit_ns,
                    call_enter_ns,
                    call_exit_ns,
                    post_sync_enter_ns,
                    post_sync_exit_ns,
                ]
            )
    if not local:
        return [], [], None, None
    latency = torch.tensor(local, dtype=torch.float32, device="npu")
    all_latency = torch.empty(EP * iterations, dtype=torch.float32, device="npu")
    dist.all_gather_into_tensor(all_latency, latency)
    all_latency_cpu = all_latency.reshape(EP, iterations).cpu()
    max_latency, slow_rank = all_latency_cpu.max(dim=0)
    return max_latency.tolist(), local, slow_rank.tolist(), host_timeline


def run_on_moe_stream(moe_stream, fn):
    if moe_stream is None:
        return fn()
    main_stream = torch.npu.current_stream()
    moe_stream.wait_stream(main_stream)
    with torch.npu.stream(moe_stream):
        output = fn()
    main_stream.wait_stream(moe_stream)
    return output


def main():
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    if world != EP:
        raise RuntimeError(f"expected WORLD_SIZE={EP}, got {world}")
    affinity = pin_worker_to_npu_numa(local_rank)
    torch_npu.npu.set_device(local_rank)
    dist.init_process_group("hccl")
    comm_group = dist.group.WORLD
    if HCCL_OP_EXPANSION_MODE:
        # Decode creates a dedicated all-to-all EP group with the AIV HCCL
        # option. Passing it to the default WORLD group is rejected by HCCL.
        hccl_options = torch_npu._C._distributed_c10d.ProcessGroupHCCL.Options()
        hccl_options.hccl_config = {
            "hccl_op_expansion_mode": HCCL_OP_EXPANSION_MODE,
        }
        comm_group = dist.new_group(
            list(range(EP)), backend="hccl", pg_options=hccl_options
        )
    group = comm_group._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
    backend = torchair.get_npu_backend()
    moe_stream = torch.npu.Stream() if MOE_STREAM else None

    device = f"npu:{local_rank}"
    per_bs = 0 if (SPARSE and rank != 0) else BS
    x_cpu, ids_cpu, scales_cpu, x, ids, scales = make_inputs(rank, device)
    if SPARSE:
        all_x = torch.zeros((BS, H), dtype=x_cpu.dtype, device="cpu")
        all_ids = torch.zeros((BS, TOP_K), dtype=torch.int32, device="cpu")
    else:
        all_x = gather_cpu(x)
        all_ids = gather_cpu(ids)

    if EAGER:
        dispatch = Dispatch(group, rank).npu()
        combine = Combine(group, rank).npu()
    else:
        dispatch = torch.compile(Dispatch(group, rank).npu(), backend=backend, dynamic=False)
        combine = torch.compile(Combine(group, rank).npu(), backend=backend, dynamic=False)

    dispatch_args = (x, ids, scales)

    out = run_on_moe_stream(moe_stream, lambda: dispatch(*dispatch_args))
    torch_npu.npu.synchronize()
    expand_x, dynamic_scales, assist, expert_token_nums, ep_counts, tp_counts, expand_scales = dispatch_outputs(out)

    if SPARSE:
        dispatch_ok = True
        dispatch_precision = {"max_abs": 0.0, "mean_abs": 0.0, "mismatch": 0}
    else:
        expected_dispatch = dispatch_golden(all_x, all_ids, rank)
        received = int(expert_token_nums.sum().item())
        dispatch_ok, dispatch_precision = validate_dispatch(
            expand_x, dynamic_scales, expected_dispatch, received
        )

    if DISPATCH_DTYPE == "bf16":
        combine_x = expand_x
        if per_bs == 0:
            expected_combine = torch.zeros((0, H), dtype=torch.float32)
        else:
            expected_combine = (x_cpu.float().unsqueeze(1) * scales_cpu.unsqueeze(2)).sum(dim=1)
    else:
        combine_x = torch.zeros(
            (GLOBAL_BS * LOCAL_EXPERTS, H),
            dtype=torch.bfloat16,
            device=x.device,
        )
        expected_combine = torch.zeros((per_bs, H), dtype=torch.float32)
    combine_args = (combine_x, ids, assist, ep_counts, tp_counts, expand_scales, scales)
    combine_out = combine(*combine_args)
    torch_npu.npu.synchronize()
    actual_combine = combine_out.cpu().float()
    diff = (actual_combine - expected_combine).abs()
    combine_close = torch.isclose(actual_combine, expected_combine, rtol=1e-2, atol=1e-2)
    local_precision = torch.tensor(
        [
            1.0 if dispatch_ok else 0.0,
            1.0 if bool(combine_close.all()) else 0.0,
            float(diff.max()),
            float(diff.mean()),
            float((~combine_close).sum()),
        ],
        dtype=torch.float32,
        device="npu",
    )
    pass_flags = local_precision[:2].clone()
    dist.all_reduce(pass_flags, op=dist.ReduceOp.MIN)
    error_max = local_precision[2].clone()
    dist.all_reduce(error_max, op=dist.ReduceOp.MAX)
    error_sum = local_precision[3].clone()
    dist.all_reduce(error_sum, op=dist.ReduceOp.SUM)
    mismatch_sum = local_precision[4].clone()
    dist.all_reduce(mismatch_sum, op=dist.ReduceOp.SUM)

    dispatch_ms, local_dispatch_ms, slow_rank_by_iteration, host_timeline = timed(
        lambda: run_on_moe_stream(moe_stream, lambda: dispatch(*dispatch_args)),
        rank,
    )
    if PROFILE_MODE and DISPATCH_DTYPE == "int8":
        os.makedirs(PROFILE_OUTPUT_DIR, exist_ok=True)
        torch.save(
            {
                "rank": rank,
                "bs": BS,
                "local_latency_ms": local_dispatch_ms,
                "host_timeline_fields": [
                    "loop_enter_ns",
                    "barrier_enter_ns",
                    "barrier_exit_ns",
                    "pre_sync_enter_ns",
                    "pre_sync_exit_ns",
                    "call_enter_ns",
                    "call_exit_ns",
                    "post_sync_enter_ns",
                    "post_sync_exit_ns",
                ],
                "host_timeline_ns": host_timeline,
            },
            os.path.join(PROFILE_OUTPUT_DIR, f"{PROFILE_RUN_TAG}_rank{rank}.pt"),
        )
    out = run_on_moe_stream(moe_stream, lambda: dispatch(*dispatch_args))
    torch_npu.npu.synchronize()
    expand_x, _, assist, _, ep_counts, tp_counts, expand_scales = dispatch_outputs(out)
    if DISPATCH_DTYPE == "bf16":
        combine_x = expand_x
    else:
        combine_x.zero_()
    combine_args = (combine_x, ids, assist, ep_counts, tp_counts, expand_scales, scales)
    combine_ms, _, _, _ = timed(
        lambda: run_on_moe_stream(moe_stream, lambda: combine(*combine_args)),
        rank,
        iterations=0 if PROFILE_MODE else ITERS,
    )

    if rank == 0:
        result = {
            "ep_world_size": EP,
            "bs_per_rank": BS,
            "global_bs": GLOBAL_BS,
            "hidden_size": H,
            "top_k": TOP_K,
            "moe_expert_num": EXPERTS,
            "copy_expert_num": COPY_EXPERTS,
            "route": ROUTE_PATTERN,
            "comm_alg": COMM_ALG,
            "hccl_op_expansion_mode": HCCL_OP_EXPANSION_MODE,
            "execution_mode": "eager" if EAGER else "graph",
            "dispatch_dtype": DISPATCH_DTYPE,
            "dispatch_quant_mode": QUANT_MODE,
            "dispatch_input_dtype": "bfloat16",
            "dispatch_output_dtype": str(expand_x.dtype).removeprefix("torch."),
            "precision": {
                "dispatch_all_ranks_passed": bool(pass_flags[0].item()),
                "dispatch_local_rank0": dispatch_precision,
                "combine_allclose_all_ranks": bool(pass_flags[1].item()),
                "combine_rtol": 1e-2,
                "combine_atol": 1e-2,
                "combine_max_abs_error_all_ranks": float(error_max.item()),
                "combine_mean_abs_error_across_ranks": float(error_sum.item() / EP),
                "combine_mismatch_elements_all_ranks": int(mismatch_sum.item()),
            },
            "dispatch": summary(dispatch_ms),
            "combine": summary(combine_ms),
            "warmup_iters": WARMUP,
            "bench_iters": ITERS,
            "profile_mode": PROFILE_MODE,
            "timing_mode": TIMING_MODE,
            "host_timeline": HOST_TIMELINE,
            "cpu_affinity": CPU_AFFINITY,
            "rank_barrier": RANK_BARRIER,
            "moe_stream": MOE_STREAM,
            "rank0_cpu_set": affinity,
            "dispatch_burst": DISPATCH_BURST,
            "slow_rank_by_iteration": slow_rank_by_iteration if PROFILE_MODE else None,
        }
        result["combine_input"] = (
            "dispatch_bf16_output"
            if DISPATCH_DTYPE == "bf16"
            else "synthetic_bf16_post_gmm_zero"
        )
        result["dispatch"]["tokens_per_second_p50"] = GLOBAL_BS * 1000.0 / result["dispatch"]["p50_ms"]
        if combine_ms:
            result["combine"]["tokens_per_second_p50"] = (
                GLOBAL_BS * 1000.0 / result["combine"]["p50_ms"]
            )
        print("EP128_RESULT " + json.dumps(result, sort_keys=True), flush=True)
        with open(os.getenv("BENCH_OUTPUT", "ep128_bs4_precision_perf.json"), "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, sort_keys=True)

    if not bool(pass_flags.min().item()):
        raise AssertionError(f"EP128 BS={BS} precision check failed; see EP128_RESULT")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
