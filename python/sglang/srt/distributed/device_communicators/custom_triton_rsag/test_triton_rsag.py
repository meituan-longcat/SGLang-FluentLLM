import os
import torch
import torch.distributed as dist
from sglang.srt.distributed.device_communicators.custom_triton_rsag.triton_rsag import TritonRSAG
import functools
from sglang.srt.distributed.device_communicators.custom_triton_rsag.utils import benchmark_with_profiler


def test_reduce_scatter(
    rank: int,
    group: dist.ProcessGroup,
    hidden_size: int,
    token_distribution_list: list,
    total_tokens_in_group: list,
    device: torch.device,
    verbose=False,
) -> None:
    out_row, out_col = 1, 16

    rsag = TritonRSAG(group, rank, 128 * 1024, hidden_size)
    if len(total_tokens_in_group) > 0:
        token_distribution_list.clear()
        for num in total_tokens_in_group:
            token_distribution = rsag.get_token_dist(num)
            token_distribution_list.append(token_distribution)

    for token_distribution in token_distribution_list:
        if rank == 0:
            print(f"[INFO] Bench token distribution {token_distribution}\n")
        hidden_states = torch.randn((sum(token_distribution), hidden_size), device=device, dtype=torch.bfloat16)
        if verbose:
            print(f"[Input]: rank {rank}, input shape: {hidden_states.shape}, input: {hidden_states[rank : rank + out_row, :out_col]}\n")

        """ torch reduce_scatter """
        output_dist = torch.empty((token_distribution[rank], hidden_size), device=device, dtype=torch.bfloat16)
        split_tensor = torch.split(hidden_states, token_distribution, dim=0)
        input_list = list(split_tensor)
        fn = functools.partial(dist.reduce_scatter, output_dist, input_list, dist.ReduceOp.SUM, group)
        base_latency = benchmark_with_profiler(fn, ".*reduce_scatter.*", warmup_iters=10, benchmark_iters=50)
        if rank == 0:
            if verbose:
                print(
                    f"[Output Torch]: rank {rank}, output shape: {output_dist.shape}, output: {output_dist[:out_row, :out_col]}, base latency: {base_latency}\n"
                )
            else:
                print(f"[INFO] base latency: {base_latency: .2f} us\n")


        """ triton reduce_scatter """
        fn = functools.partial(rsag.reduce_scatter, hidden_states, None, token_distribution)
        opt_latency = benchmark_with_profiler(fn, ".*reduce_scatter.*", warmup_iters=10, benchmark_iters=50)
        output_trit, offset = rsag.reduce_scatter(hidden_states, None, token_distribution)
        if rank == 0:
            if verbose:
                print(
                    f"[Output Triton]: rank {rank}, output shape: {output_trit.shape}, output: {output_trit[:out_row, :out_col]}, offset: {offset}, opt latency: {opt_latency}\n"
                )
            else:
                print(f"[INFO] opt latency: {opt_latency: .2f} us\n")

        result = torch.allclose(output_dist, output_trit, rtol=1e-1, atol=1e-1)
        if result:
            if rank == 0:
                print(f"[Success]\n")
        else:
            print(f"[Error]\n")


def test_all_gather(
    rank: int,
    group: dist.ProcessGroup,
    hidden_size: int,
    token_distribution_list: list,
    total_tokens_in_group: list,
    device: torch.device,
    verbose=False,
) -> None:
    out_row, out_col = 1, 8

    rsag = TritonRSAG(group, rank, 128 * 1024, hidden_size)
    if len(total_tokens_in_group) > 0:
        token_distribution_list.clear()
        for num in total_tokens_in_group:
            token_distribution = rsag.get_token_dist(num)
            token_distribution_list.append(token_distribution)

    for token_distribution in token_distribution_list:
        if rank == 0:
            print(f"[INFO] Bench token distribution {token_distribution}\n")
        hidden_states = torch.randn((token_distribution[rank], hidden_size), device=device, dtype=torch.bfloat16)
        output_list = [
            torch.empty((token_distribution[i], hidden_size), device=device, dtype=torch.bfloat16) for i in range(0, len(token_distribution))
        ]

        """ torch all_gather """
        fn = functools.partial(dist.all_gather, output_list, hidden_states, group)
        base_latency = benchmark_with_profiler(fn, ".*allgather.*", warmup_iters=10, benchmark_iters=50)
        output_dist = torch.concat(output_list, dim=0)
        if rank == 0:
            if verbose:
                print(
                    f"[Output Torch]: rank {rank}, output shape: {output_dist.shape}, output: {output_dist[:out_row, :out_col]}, base latency: {base_latency}\n"
                )
            else:
                print(f"[INFO] base latency: {base_latency:.2f} us\n")


        """ triton all_gather """
        fn = functools.partial(rsag.all_gather, hidden_states, None, token_distribution)
        opt_latency = benchmark_with_profiler(fn, ".*all_gather.*", warmup_iters=10, benchmark_iters=50)
        output_trit = rsag.all_gather(hidden_states, None, token_distribution)
        if rank == 0:
            if verbose:
                print(
                    f"[Output Triton]: rank {rank}, output shape: {output_trit.shape}, output: {output_trit[:out_row, :out_col]}, opt latency: {opt_latency}\n"
                )
            else:
                print(f"[INFO] opt latency: {opt_latency:.2f} us\n")

        result = torch.allclose(output_dist, output_trit, rtol=1e-6, atol=1e-6)
        if result:
            if rank == 0:
                print(f"[Success]\n")
        else:
            print(f"[Error]\n")


"""
EPS_HOME=path_to_eps LD_PRELOAD=/usr/lib64/libcuda.so PYTHONPATH=$EPS_HOME/python:$PYTHONPATH \
    torchrun --nnodes 1 --nproc_per_node 8 test_triton_rsag.py
"""
if __name__ == "__main__":
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    dist.init_process_group("nccl")

    global_rank = dist.get_rank()
    world_size = dist.get_world_size()

    group = dist.new_group(ranks=list(range(world_size)))

    torch.manual_seed(42 + global_rank)
    token_distribution_list = []
    for i in range(0, 11):
        num_token_per_rank_list = []
        for r in range(0, world_size):
            num_token_per_rank_list.append(2**i + r)
        token_distribution_list.append(num_token_per_rank_list)
    if world_size == 8:
        special_case = [1, 20, 3, 0, 5, 1, 0, 9]
        token_distribution_list.append(special_case)

    total_tokens_in_group = []
    for i in range(0, 1024):
        total_tokens_in_group.append(128 * 1024)

    hidden_size = 6144

    # eps init
    # ssdp.init_distributed_environment(world_size=dist.get_world_size(), rank=global_rank, local_rank=local_rank)
    # ssld.init_tp_dp_convertor(global_rank, 1024*8, dist.get_world_size(), hidden_size)
    dist.barrier()

    if global_rank == 0:
        print(f"[INFO] test_all_gather with {world_size} ranks\n")
    test_all_gather(global_rank, group, hidden_size, token_distribution_list, total_tokens_in_group, device, False)

    dist.destroy_process_group()
