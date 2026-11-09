#!/usr/bin/env python3
"""Continuously validate mixed sparse lengths for decode attention."""

import argparse
import gc
import math
import signal
import time
import traceback

import torch
import torch_npu
import flash_ops  # noqa: F401: registers the custom operator


HEADS = 64
LATENT = 512
ROPE = 64
PAGE_SIZE = 128
SPARSE_COUNT = 2048
S2_ACT = 64 * 1024
BLOCKS_PER_REQUEST = S2_ACT // PAGE_SIZE
TOTAL_BLOCKS = 640


def parse_args():
    parser = argparse.ArgumentParser(
        description="Continuous mixed-length sparse attention precision stress test")
    parser.add_argument("--operator", choices=("sfad", "sfa"), default="sfad",
                        help="decode新算子或原sparse flash attention算子")
    parser.add_argument("--batches", default="1,2,4,8,9,16")
    parser.add_argument("--tokens", default="4,8")
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--cycles", type=int, default=0,
                        help="0 means run until interrupted")
    parser.add_argument("--output-max-atol", type=float, default=1e-2)
    parser.add_argument("--output-mean-atol", type=float, default=1e-3)
    parser.add_argument("--lse-max-atol", type=float, default=5e-4)
    parser.add_argument("--stop-on-failure", action="store_true")
    return parser.parse_args()


def parse_int_list(value):
    return [int(item) for item in value.split(",") if item]


def make_inputs(batch_size, tokens_per_batch, seed):
    generator = torch.Generator().manual_seed(seed)
    total_q = batch_size * tokens_per_batch

    query = torch.rand(total_q, HEADS, LATENT,
                       dtype=torch.bfloat16, generator=generator)
    query_rope = torch.rand(total_q, HEADS, ROPE,
                            dtype=torch.bfloat16, generator=generator)
    key = torch.rand(TOTAL_BLOCKS, PAGE_SIZE, 1, LATENT,
                     dtype=torch.bfloat16, generator=generator)
    key_rope = torch.rand(TOTAL_BLOCKS, PAGE_SIZE, 1, ROPE,
                          dtype=torch.bfloat16, generator=generator)
    value = key

    block_table = torch.empty(batch_size, BLOCKS_PER_REQUEST,
                              dtype=torch.int32)
    for batch in range(batch_size):
        block_table[batch] = torch.randperm(
            TOTAL_BLOCKS, generator=generator, dtype=torch.int32
        )[:BLOCKS_PER_REQUEST]

    # Every case contains all three modes. Rotate their query positions across
    # seeds so prolonged testing also changes batch/token placement.
    valid_counts = torch.empty(total_q, dtype=torch.int32)
    sparse_indices = torch.full((total_q, 1, SPARSE_COUNT), -1,
                                dtype=torch.int32)
    mode_offset = seed % 3
    for token in range(total_q):
        mode = (token + mode_offset) % 3
        if mode == 0:
            valid = 0
        elif mode == 1:
            valid = int(torch.randint(
                0, SPARSE_COUNT + 1, (1,), generator=generator).item())
        else:
            valid = SPARSE_COUNT
        valid_counts[token] = valid
        if valid:
            sparse_indices[token, 0, :valid] = torch.randperm(
                S2_ACT, generator=generator, dtype=torch.int32
            )[:valid]

    seq_q = torch.arange(1, batch_size + 1, dtype=torch.int32)
    seq_q *= tokens_per_batch
    seq_kv = torch.full((batch_size,), S2_ACT, dtype=torch.int32)
    return {
        "query": query,
        "key": key,
        "value": value,
        "indices": sparse_indices,
        "query_rope": query_rope,
        "key_rope": key_rope,
        "block_table": block_table,
        "seq_q": seq_q,
        "seq_kv": seq_kv,
        "valid_counts": valid_counts,
    }


def run_operator(inputs, operator_name="sfad"):
    npu = {
        name: tensor.npu()
        for name, tensor in inputs.items()
        if name != "valid_counts"
    }

    def invoke():
        operator = (torch_npu.npu_sparse_flash_attention_decode
                    if operator_name == "sfad"
                    else torch_npu.npu_sparse_flash_attention)
        return operator(
            npu["query"], npu["key"], npu["value"], npu["indices"], 1.0,
            block_table=npu["block_table"],
            actual_seq_lengths_query=npu["seq_q"],
            actual_seq_lengths_kv=npu["seq_kv"],
            query_rope=npu["query_rope"], key_rope=npu["key_rope"],
            sparse_block_size=1, layout_query="TND", layout_kv="PA_BSND",
            sparse_mode=3, attention_mode=2, return_softmax_lse=True)

    first = invoke()
    torch_npu.npu.synchronize()
    start = torch_npu.npu.Event(enable_timing=True)
    end = torch_npu.npu.Event(enable_timing=True)
    start.record()
    second = invoke()
    end.record()
    torch_npu.npu.synchronize()

    out0, max0, sum0 = first
    out1, max1, sum1 = second
    repeat_out = (out0 - out1).abs().max().item()
    lse0 = max0.float() + torch.log(sum0.float())
    lse1 = max1.float() + torch.log(sum1.float())
    same_inf = torch.isinf(lse0) & torch.isinf(lse1) & (lse0 == lse1)
    repeat_lse = torch.where(
        same_inf, 0.0, (lse0 - lse1).abs()).max().item()
    if repeat_out != 0.0 or repeat_lse != 0.0:
        raise AssertionError(
            f"non-deterministic result: output={repeat_out}, lse={repeat_lse}")
    return out1.float().cpu(), lse1.cpu(), start.elapsed_time(end), npu


def cpu_reference(inputs, tokens_per_batch):
    query = inputs["query"]
    key = inputs["key"]
    value = inputs["value"]
    indices = inputs["indices"]
    query_rope = inputs["query_rope"]
    key_rope = inputs["key_rope"]
    block_table = inputs["block_table"]
    valid_counts = inputs["valid_counts"]
    total_q = query.shape[0]

    output = torch.zeros_like(query, dtype=torch.float32)
    lse = torch.full((total_q, HEADS), -torch.inf, dtype=torch.float32)
    for token in range(total_q):
        batch = token // tokens_per_batch
        token_in_batch = token % tokens_per_batch
        threshold = S2_ACT - tokens_per_batch + token_in_batch + 1
        valid = int(valid_counts[token])
        positions = indices[token, 0, :valid]
        positions = positions[(positions >= 0) &
                              (positions < threshold)].long()
        if positions.numel() == 0:
            continue
        physical_blocks = block_table[batch, positions // PAGE_SIZE].long()
        rows = positions % PAGE_SIZE
        selected_k = key[physical_blocks, rows, 0].float()
        selected_v = value[physical_blocks, rows, 0].float()
        selected_rope = key_rope[physical_blocks, rows, 0].float()
        logits = (query[token].float() @ selected_k.T +
                  query_rope[token].float() @ selected_rope.T)
        output[token] = torch.softmax(logits, dim=-1) @ selected_v
        lse[token] = torch.logsumexp(logits, dim=-1)
    return output, lse


def validate_case(batch_size, tokens_per_batch, seed, args):
    started = time.time()
    inputs = make_inputs(batch_size, tokens_per_batch, seed)
    actual_out, actual_lse, device_ms, npu_inputs = run_operator(
        inputs, getattr(args, "operator", "sfad"))
    reference_out, reference_lse = cpu_reference(inputs, tokens_per_batch)

    if actual_out.dtype != torch.float32:
        raise AssertionError(f"output dtype is {actual_out.dtype}, not float32")
    if torch.isnan(actual_out).any() or torch.isinf(actual_out).any():
        raise AssertionError("output contains NaN/Inf")

    out_diff = (actual_out - reference_out).abs()
    same_inf = (torch.isinf(actual_lse) & torch.isinf(reference_lse) &
                (actual_lse == reference_lse))
    lse_diff = torch.where(
        same_inf, 0.0, (actual_lse.reshape_as(reference_lse) -
                        reference_lse).abs())
    if torch.isnan(lse_diff).any() or torch.isinf(lse_diff).any():
        raise AssertionError("LSE comparison contains unexpected NaN/Inf")

    out_max = out_diff.max().item()
    out_mean = out_diff.mean().item()
    lse_max = lse_diff.max().item()
    if out_max > args.output_max_atol or out_mean > args.output_mean_atol:
        raise AssertionError(
            f"output diff max/mean={out_max}/{out_mean}, limits="
            f"{args.output_max_atol}/{args.output_mean_atol}")
    if lse_max > args.lse_max_atol:
        raise AssertionError(
            f"LSE diff max={lse_max}, limit={args.lse_max_atol}")

    zero_rows = inputs["valid_counts"] == 0
    if zero_rows.any():
        if torch.count_nonzero(actual_out[zero_rows]).item() != 0:
            raise AssertionError("zero-topk query produced non-zero output")
        if not torch.isneginf(
                actual_lse.reshape_as(reference_lse)[zero_rows]).all():
            raise AssertionError("zero-topk query did not produce -inf LSE")

    counts = inputs["valid_counts"]
    elapsed = time.time() - started
    del npu_inputs, actual_out, actual_lse, reference_out, reference_lse
    del inputs
    gc.collect()
    torch_npu.npu.empty_cache()
    return {
        "valid_min": int(counts.min()),
        "valid_mean": counts.float().mean().item(),
        "valid_max": int(counts.max()),
        "out_max": out_max,
        "out_mean": out_mean,
        "lse_max": lse_max,
        "device_ms": device_ms,
        "elapsed_s": elapsed,
    }


def main():
    args = parse_args()
    batches = parse_int_list(args.batches)
    tokens = parse_int_list(args.tokens)
    cases = [(batch, token) for batch in batches for token in tokens]
    torch.npu.set_compile_mode(jit_compile=False)
    torch.set_num_threads(max(1, min(32, torch.get_num_threads())))

    stop_requested = False

    def request_stop(_signum, _frame):
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    cycle = 0
    passed = 0
    failed = 0
    print(f"START operator={args.operator} cases={cases} "
          f"base_seed={args.seed}", flush=True)
    while not stop_requested and (args.cycles == 0 or cycle < args.cycles):
        cycle += 1
        print(f"CYCLE {cycle} START", flush=True)
        for case_index, (batch, token) in enumerate(cases):
            if stop_requested:
                break
            seed = args.seed + (cycle - 1) * len(cases) + case_index
            try:
                result = validate_case(batch, token, seed, args)
                passed += 1
                print(
                    f"PASS cycle={cycle} B={batch} T={token} seed={seed} "
                    f"valid={result['valid_min']}/"
                    f"{result['valid_mean']:.1f}/{result['valid_max']} "
                    f"out={result['out_max']:.6g}/"
                    f"{result['out_mean']:.6g} lse={result['lse_max']:.6g} "
                    f"npu={result['device_ms']:.3f}ms "
                    f"wall={result['elapsed_s']:.2f}s total={passed}/{failed}",
                    flush=True)
            except Exception:
                failed += 1
                print(f"FAIL cycle={cycle} B={batch} T={token} seed={seed} "
                      f"total={passed}/{failed}", flush=True)
                traceback.print_exc()
                if args.stop_on_failure:
                    raise
                gc.collect()
                torch_npu.npu.empty_cache()
                time.sleep(2)
        print(f"CYCLE {cycle} END passed={passed} failed={failed}", flush=True)
    print(f"STOP cycles={cycle} passed={passed} failed={failed}", flush=True)


if __name__ == "__main__":
    main()
