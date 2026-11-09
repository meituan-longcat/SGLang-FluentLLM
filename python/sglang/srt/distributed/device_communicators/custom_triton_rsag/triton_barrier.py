import os

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
import triton
import triton.language as tl

from sglang.srt.distributed.device_communicators.custom_triton_rsag.triton_utils import get_flat_bid, get_flat_tid, sync_threads
from sglang.srt.distributed.device_communicators.custom_triton_rsag.utils import log_triton_kernel


@triton.jit
def send_signal(addrs, sem: tl.constexpr):
    if sem == "relaxed":
        tl.inline_asm_elementwise(
            """
            {
                .reg .u32   %tmp32_<1>;
                .reg .pred  %p<1>;

                send_signal:
                    atom.global.relaxed.sys.cas.b32 %tmp32_0, [$1], 0, 1;
                    setp.eq.u32 %p0, %tmp32_0, 0;
                    @!%p0 bra send_signal;
            }
            """,
            "=r, l",
            [addrs],
            dtype=tl.int32,
            is_pure=False,
            pack=1,
        )
    elif sem == "acq_rel":
        tl.inline_asm_elementwise(
            """
            {
                .reg .u32   %tmp32_<1>;
                .reg .pred  %p<1>;

                send_signal:
                    atom.global.release.sys.cas.b32 %tmp32_0, [$1], 0, 1;
                    setp.eq.u32 %p0, %tmp32_0, 0;
                    @!%p0 bra send_signal;
            }
            """,
            "=r, l",
            [addrs],
            dtype=tl.int32,
            is_pure=False,
            pack=1,
        )
    else:
        raise RuntimeError(f"Unrecognized sem: {sem}")


@triton.jit
def wait_signal(addrs, sem: tl.constexpr):
    if sem == "relaxed":
        tl.inline_asm_elementwise(
            """
            {
                .reg .u32   %tmp32_<1>;
                .reg .pred  %p<1>;

                wait_signal:
                    atom.global.sys.relaxed.cas.b32 %tmp32_0, [$1], 1, 0;
                    setp.eq.u32 %p0, %tmp32_0, 1;
                    @!%p0 bra wait_signal;
            }
            """,
            "=r, l",
            [addrs],
            dtype=tl.int32,
            is_pure=False,
            pack=1,
        )
    elif sem == "acq_rel":
        tl.inline_asm_elementwise(
            """
            {
                .reg .u32   %tmp32_<1>;
                .reg .pred  %p<1>;

                wait_signal:
                    atom.global.sys.acquire.cas.b32 %tmp32_0, [$1], 1, 0;
                    setp.eq.u32 %p0, %tmp32_0, 1;
                    @!%p0 bra wait_signal;
            }
            """,
            "=r, l",
            [addrs],
            dtype=tl.int32,
            is_pure=False,
            pack=1,
        )
    else:
        raise RuntimeError(f"Unrecognized sem: {sem}")


@triton.jit
def blockwise_barrier(
    signal_pad_ptrs,
    block_id,
    rank: tl.constexpr,
    world_size: tl.constexpr,
    sem: tl.constexpr,
):
    """
    Synchronizes blocks with matching block_id across participating devices.

    Note: the function itself is not a system level barrier/fence. It is a
    building block for expressing different synchronization patterns.

    Pattern 0: Ensures that all writes to symm_mem buffers from previous
    kernels across all devices are visible to the current kernel:

        blockwise_barrier(..., sem="relaxed")
        sync_threads()

    Pattern 1: Ensures that all writes to symm_mem buffers from the current
    block are visible to all remote blocks with matching blockIdx:

        sync_threads()
        blockwise_barrier(..., sem="acq_rel")
        sync_threads()

    Pattern 2: Ensures that symm_mem buffers read by the current kernel are safe
    for writing by subsequent kernels across all devices.

        sync_threads()
        blockwise_barrier(..., sem="relaxed")

    CUDA graph friendliness:

        This barrier operates through atomic operations on a zero-filled signal
        pad, which resets to a zero-filled state after each successful
        synchronization. This design eliminates the need for incrementing a
        flag from host.
    """
    if block_id is None:
        block_id = get_flat_bid()
    flat_tid = get_flat_tid()

    remote_ranks = tl.arange(0, world_size)
    signal_pad_ptrs = signal_pad_ptrs.to(tl.pointer_type(tl.uint64))
    remote_signal_pad_addrs = tl.load(signal_pad_ptrs + remote_ranks).to(tl.pointer_type(tl.uint32))
    send_addrs = remote_signal_pad_addrs + block_id * world_size + rank

    local_signal_pad_addr = tl.load(signal_pad_ptrs + rank).to(tl.pointer_type(tl.uint32))
    wait_addrs = local_signal_pad_addr + block_id * world_size + remote_ranks

    if flat_tid < world_size:
        send_signal(send_addrs, sem)
        wait_signal(wait_addrs, sem)


@triton.jit
def barrier_test_kernel(
    signal_pad_ptrs,
    rank: tl.constexpr,
    world_size: tl.constexpr,
):
    blockwise_barrier(signal_pad_ptrs, None, rank, world_size, "relaxed")
    sync_threads()


def barrier_test(t: torch.Tensor) -> None:
    symm_mem_hdl = symm_mem.rendezvous(t, group=dist.group.WORLD)

    kernel = barrier_test_kernel[(32, 1, 1)](
        symm_mem_hdl.signal_pad_ptrs_dev,
        rank=symm_mem_hdl.rank,
        world_size=symm_mem_hdl.world_size,
    )
    log_triton_kernel(kernel)

    signal_pad = symm_mem_hdl.get_signal_pad(symm_mem_hdl.rank)
    assert signal_pad.eq(0).all().item()


if __name__ == "__main__":
    """
    torchrun \
    --nnodes 1 --nproc-per-node 8 \
    --rdzv-backend c10d --rdzv-endpoint localhost:0 \
    --no_python python3 triton_barrier.py
    """
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    dist.init_process_group("nccl")

    t = symm_mem.empty(4096, device=device)
    barrier_test(t)
