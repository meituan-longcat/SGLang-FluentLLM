import argparse
import random
import time

import torch
import torch_npu

native_npu_gather_element = getattr(torch_npu, "npu_gather_element", None)
import flash_ops  # noqa: F401


TOPK = 2048
LOCAL = 2048
KVP_SIZE = 4


def reference(local: torch.Tensor, position: torch.Tensor, rank: int):
    local = local.cpu()
    position = position.cpu().to(torch.int64)
    tokens = position.shape[0]
    output = torch.full((tokens, 1, TOPK), -1, dtype=torch.int32)
    chunks = torch.empty(tokens, dtype=torch.int32)
    start = rank * LOCAL
    for token in range(tokens):
        mask = (position[token] >= start) & (position[token] < start + LOCAL)
        offset = position[token, mask] - start
        selected = local[token, 0, offset]
        selected = selected[selected >= 0]
        output[token, 0, : selected.numel()] = selected
        chunks[token] = (selected.numel() + 127) // 128
    return output, chunks


def run_case(local: torch.Tensor, position: torch.Tensor, rank: int):
    expected, expected_chunks = reference(local, position, rank)
    actual, actual_chunks = torch.ops.custom.npu_select_local_topk_indices(
        local.npu(), position.npu(), rank
    )
    torch.npu.synchronize()
    actual = actual.cpu()
    actual_chunks = actual_chunks.cpu()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual_chunks, expected_chunks, rtol=0, atol=0)


def exact_count_cases(dtype: torch.dtype):
    rank = 2
    counts = [0, 1, 127, 128, 129, 1024, 2048]
    local = torch.arange(10000, 10000 + len(counts) * LOCAL,
                         dtype=torch.int32).view(len(counts), 1, LOCAL)
    position = torch.empty((len(counts), TOPK), dtype=dtype)
    nonlocal_positions = torch.arange(TOPK, dtype=torch.int64) % LOCAL
    for token, count in enumerate(counts):
        local_positions = rank * LOCAL + torch.arange(count, dtype=torch.int64)
        other_positions = nonlocal_positions[: TOPK - count]
        row = torch.cat([local_positions, other_positions])
        # Stable compaction is observable: matches are deliberately interleaved.
        order = torch.randperm(TOPK, generator=torch.Generator().manual_seed(token))
        position[token] = row[order].to(dtype)
    run_case(local, position, rank)


def padded_local_case(dtype: torch.dtype):
    rank = 1
    local = torch.arange(LOCAL, dtype=torch.int32).view(1, 1, LOCAL)
    local[..., 19] = -1
    local[..., 777] = -1
    position = torch.arange(rank * LOCAL, (rank + 1) * LOCAL,
                            dtype=dtype).view(1, TOPK)
    run_case(local, position, rank)


def random_cases(dtype: torch.dtype, rounds: int):
    generator = torch.Generator().manual_seed(20260823)
    for _ in range(rounds):
        tokens = random.choice([1, 2, 4, 8, 16, 32, 64])
        rank = random.randrange(KVP_SIZE)
        local = torch.randint(
            0, 100000, (tokens, 1, LOCAL), dtype=torch.int32,
            generator=generator,
        )
        # Include padding inside the local row, not just as a suffix.
        padding = torch.rand(local.shape, generator=generator) < 0.03
        local[padding] = -1
        position = torch.randint(
            0, KVP_SIZE * LOCAL, (tokens, TOPK), dtype=torch.int64,
            generator=generator,
        ).to(dtype)
        run_case(local, position, rank)


def benchmark(tokens: int, rank: int):
    if native_npu_gather_element is None:
        raise RuntimeError(
            "This torch_npu build does not expose npu_gather_element; "
            "correctness tests remain supported."
        )
    local = torch.randint(0, 100000, (tokens, 1, LOCAL),
                          dtype=torch.int32, device="npu")
    position = torch.randint(0, KVP_SIZE * LOCAL, (tokens, TOPK),
                             dtype=torch.int32, device="npu")
    expanded = torch.full((tokens, KVP_SIZE * LOCAL), -1,
                          dtype=torch.int32, device="npu")
    expanded[:, rank * LOCAL:(rank + 1) * LOCAL] = local.squeeze(1)

    def fused():
        return torch.ops.custom.npu_select_local_topk_indices(
            local, position, rank
        )

    def original():
        gathered = native_npu_gather_element(expanded, 1, position)
        return torch.sort(gathered.float(), dim=-1, descending=True)[0].int()

    for _ in range(20):
        fused()
        original()
    torch.npu.synchronize()
    for name, function in (("fused", fused), ("original", original)):
        begin = time.perf_counter()
        for _ in range(200):
            function()
        torch.npu.synchronize()
        print(f"{name} T={tokens}: {(time.perf_counter() - begin) * 5000:.3f} us")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()
    random.seed(20260823)
    torch.npu.set_device(0)
    for dtype in (torch.int32, torch.int64):
        exact_count_cases(dtype)
        padded_local_case(dtype)
        random_cases(dtype, args.rounds)
        print(f"PASS dtype={dtype}")
    if args.benchmark:
        for tokens in (1, 4, 16, 64):
            benchmark(tokens, 2)


if __name__ == "__main__":
    main()
