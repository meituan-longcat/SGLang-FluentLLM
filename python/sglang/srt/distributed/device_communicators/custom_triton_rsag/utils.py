import os
import re
from contextlib import nullcontext
from typing import Callable, List, Optional

import torch
import torch.distributed as dist


def benchmark_with_profiler(
    target_fn: Callable[[None], None],
    event_key_regex: str,
    warmup_iters: int = 200,
    benchmark_iters: int = 25,
    profile_ranks: Optional[List[int]] = None,
    flush_l2: bool = False,
) -> float:
    """
    Benchmark the target function with PyTorch profiler.

    Args:
        target_fn: The target function to benchmark.
        event_key_regex: The regex pattern to identify the profiler event
            associated with the target function.
        profile_ranks: The ranks to profile.
        warmup_iters: The number of warmup iterations.
        benchmark_iters: The number of benchmark iterations.
        flush_l2: Whether to flush the L2 cache before each invocation of the
            target function.

    Returns:
        The measured median latency in microseconds.
    """
    if "BENCHMARK_ITERS" in os.environ:
        benchmark_iters = int(os.environ["BENCHMARK_ITERS"])

    rank = dist.get_rank() if dist.is_initialized() else 0
    profile_ranks = profile_ranks or [0]

    if flush_l2:
        cache = torch.empty(int(256e6 // 4), dtype=torch.int, device="cuda")

    if rank in profile_ranks:
        try:
            from trace_handler import trace_handler
        except ImportError:
            trace_handler = None

        if "NO_TRACE" in os.environ:
            trace_handler = None

        prof = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            on_trace_ready=trace_handler,
        )
    else:
        prof = nullcontext()

    for _ in range(warmup_iters):
        target_fn()

    if dist.is_initialized():
        dist.barrier(device_ids=[torch.cuda.current_device()])
    torch.cuda.synchronize()

    with prof:
        torch.cuda._sleep(int(2e7))
        for i in range(benchmark_iters):
            if flush_l2:
                cache.zero_()
            target_fn()
        torch.cuda.synchronize()

    if rank not in profile_ranks:
        return 0

    latencies_us = []
    for event in prof.events():
        if re.match(event_key_regex, event.key):
            latencies_us.append(event.device_time)

    if len(latencies_us) == 0:
        return 0

    return torch.tensor(latencies_us).median().item()


def benchmark_with_event(
    target_fn: Callable[[None], None],
    warmup_iters: int = 200,
    benchmark_iters: int = 25,
    profile_ranks: Optional[List[int]] = None,
    flush_l2: bool = False,
    cuda_graph: bool = False,
) -> float:
    if cuda_graph:
        target_fn()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            target_fn()
        target_fn = lambda: g.replay()

    if "BENCHMARK_ITERS" in os.environ:
        benchmark_iters = int(os.environ["BENCHMARK_ITERS"])

    rank = dist.get_rank() if dist.is_initialized() else 0
    profile_ranks = profile_ranks or [0]

    if flush_l2:
        cache = torch.empty(int(256e6 // 4), dtype=torch.int, device="cuda")

    for _ in range(warmup_iters):
        target_fn()

    if dist.is_initialized():
        dist.barrier(device_ids=[torch.cuda.current_device()])
    torch.cuda.synchronize()

    begin_events = [torch.cuda.Event(enable_timing=True) for _ in range(benchmark_iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(benchmark_iters)]

    if rank in profile_ranks:
        try:
            from trace_handler import trace_handler
        except ImportError:
            trace_handler = None

        if "NO_TRACE" in os.environ:
            trace_handler = None

        prof = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            on_trace_ready=trace_handler,
        )
    else:
        prof = nullcontext()

    with prof:
        torch.cuda._sleep(int(2e7))
        for i in range(benchmark_iters):
            if flush_l2:
                cache.zero_()
            begin_events[i].record()
            target_fn()
            end_events[i].record()
        torch.cuda.synchronize()

    latencies = [b.elapsed_time(e) for b, e in zip(begin_events, end_events)]
    return torch.tensor(latencies).median().item() * 1000


triton_kernels = {}


def log_triton_kernel(kernel):
    import atexit
    import tempfile

    if dist.is_initialized() and dist.get_rank() != 0:
        return

    def on_exit():
        print("PTX files:")
        for kernel in triton_kernels:
            f = tempfile.NamedTemporaryFile(dir="/tmp", delete=False)
            f.write(kernel.asm["ptx"].encode("utf-8"))
            print(f"+- {kernel.name}: {f.name}")

    if len(triton_kernels) == 0:
        atexit.register(on_exit)

    if kernel not in triton_kernels:
        triton_kernels[kernel] = None
