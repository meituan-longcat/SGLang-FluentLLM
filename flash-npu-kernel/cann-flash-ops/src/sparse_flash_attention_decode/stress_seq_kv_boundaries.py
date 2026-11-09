#!/usr/bin/env python3
"""Stress SFAD sequence/KV length and sparse-index boundary semantics."""

import argparse
import gc
import signal
import time
import traceback
from dataclasses import dataclass

import torch
import torch_npu
import flash_ops  # noqa: F401: registers the custom operator


HEADS = 64
LATENT = 512
ROPE = 64
PAGE = 128
SPARSE = 2048
PHYSICAL_BLOCKS = 160
COUNT_BOUNDARIES = (0, 1, 63, 64, 65, 127, 128, 129, 511, 512, 2047, 2048)


@dataclass(frozen=True)
class Case:
    name: str
    q_lens: tuple
    kv_lens: tuple
    table_width: int = 0
    dtype: torch.dtype = torch.bfloat16
    sparse_mode: int = 3
    scale: float = 1.0
    pre_tokens: int = 9223372036854775807
    next_tokens: int = 9223372036854775807
    attention_mode: int = 2


DIRECTED_CASES = (
    Case("minimum", (1,), (1,)),
    Case("tiny_causal", (4,), (4,)),
    Case("kv_63", (8,), (63,)),
    Case("kv_127", (8,), (127,)),
    Case("kv_128", (8,), (128,)),
    Case("kv_129", (8,), (129,)),
    Case("mixed_batches", (1, 2, 4, 8), (1, 63, 128, 129)),
    Case("non_aligned", (3, 5, 7, 9), (65, 127, 130, 257)),
    Case("large_kv", (4, 8), (2048, 65536)),
    Case("variable_q_same_kv", (4, 8), (65536, 65536)),
    Case("same_q_variable_kv", (8, 8), (63, 129)),
    Case("variable_q_variable_kv", (4, 8), (63, 129)),
    Case("count_boundaries", (12,), (65536,)),
    Case("batch_64", tuple([1] * 64), tuple([129] * 64)),
    Case("table_width_9999", (4,), (129,), 9999),
    Case("mode0_service", tuple([4] * 16), tuple([65536] * 16),
         table_width=8002, sparse_mode=0, scale=0.1578114),
    # Robustness cases. The intended causal threshold is clamped to zero.
    Case("q_longer_than_kv", (4, 8), (1, 3)),
    Case("zero_kv", (4,), (0,)),
    Case("float16", (3, 5), (127, 129), 0, torch.float16),
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Continuous SFAD seq/KV/index boundary precision stress")
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--cycles", type=int, default=0,
                        help="0 means run until interrupted")
    parser.add_argument("--cases", default="",
                        help="comma-separated directed case names")
    parser.add_argument("--stop-on-failure", action="store_true")
    parser.add_argument("--output-max-atol", type=float, default=1e-2)
    parser.add_argument("--output-mean-atol", type=float, default=2e-3)
    parser.add_argument("--lse-max-atol", type=float, default=5e-4)
    parser.add_argument(
        "--expect-rejected",
        choices=("block_table_10000", "batch_65", "sparse_2047",
                 "sparse_mode_2", "pre_tokens_0", "next_tokens_0",
                 "attention_mode_0"),
        help="run one invalid-shape case and require tiling rejection")
    return parser.parse_args()


def randperm_range(begin, end, count, generator):
    span = end - begin
    if count == 0:
        return torch.empty(0, dtype=torch.int32)
    if span >= count:
        return torch.randperm(span, generator=generator,
                              dtype=torch.int32)[:count] + begin
    return torch.randint(begin, end, (count,), generator=generator,
                         dtype=torch.int32)


def make_row(raw_count, mode, threshold, kv_len, address_limit, generator):
    if raw_count == 0:
        return torch.empty(0, dtype=torch.int32)
    threshold = max(0, threshold)
    valid_end = min(threshold, kv_len, address_limit)
    invalid_begin = min(max(threshold, 0), address_limit - 1)

    if mode == 0:
        return torch.empty(0, dtype=torch.int32)
    if mode == 1:
        # Exact causal/KV boundaries, then fill from the addressable invalid
        # range. Values are kept inside block-table shape to make OOB masking
        # observable without an unrelated GM fault.
        probes = [threshold - 1, threshold, threshold + 1,
                  kv_len - 1, kv_len, kv_len + 1,
                  0, address_limit - 1]
        values = [value for value in probes if 0 <= value < address_limit]
        while len(values) < raw_count:
            values.append(int(torch.randint(
                invalid_begin, address_limit, (1,), generator=generator)))
        return torch.tensor(values[:raw_count], dtype=torch.int32)
    if mode == 2 or valid_end == 0:
        return randperm_range(invalid_begin, address_limit, raw_count,
                              generator)
    if mode == 3:
        valid_count = raw_count // 2
        valid = randperm_range(0, valid_end, valid_count, generator)
        invalid = randperm_range(invalid_begin, address_limit,
                                 raw_count - valid_count, generator)
        values = torch.cat((valid, invalid))
        return values[torch.randperm(raw_count, generator=generator)]
    return randperm_range(0, valid_end, raw_count, generator)


def make_inputs(case, seed):
    generator = torch.Generator().manual_seed(seed)
    batch_size = len(case.q_lens)
    total_q = sum(case.q_lens)
    max_kv = max(case.kv_lens)
    required_blocks = (max_kv + PAGE - 1) // PAGE
    table_width = case.table_width or max(64, required_blocks + 32)
    if table_width < max(1, required_blocks):
        raise ValueError("block table is too narrow for the configured KV")
    address_limit = table_width * PAGE

    query = torch.rand(total_q, HEADS, LATENT, dtype=case.dtype,
                       generator=generator)
    query_rope = torch.rand(total_q, HEADS, ROPE, dtype=case.dtype,
                            generator=generator)
    key = torch.rand(PHYSICAL_BLOCKS, PAGE, 1, LATENT, dtype=case.dtype,
                     generator=generator)
    key_rope = torch.rand(PHYSICAL_BLOCKS, PAGE, 1, ROPE, dtype=case.dtype,
                          generator=generator)
    value = key

    logical = torch.arange(table_width, dtype=torch.int32)
    block_table = torch.stack([
        (logical + batch * 17 + seed) % PHYSICAL_BLOCKS
        for batch in range(batch_size)
    ])
    indices = torch.full((total_q, 1, SPARSE), -1, dtype=torch.int32)
    raw_counts = torch.empty(total_q, dtype=torch.int32)
    effective_counts = torch.empty(total_q, dtype=torch.int32)
    thresholds = torch.empty(total_q, dtype=torch.int32)
    has_duplicate_effective = False

    token = 0
    for batch, (q_len, kv_len) in enumerate(zip(case.q_lens, case.kv_lens)):
        for local_q in range(q_len):
            threshold = (kv_len if case.sparse_mode == 0 else
                         max(0, kv_len - q_len + local_q + 1))
            raw_count = COUNT_BOUNDARIES[(token + seed) % len(COUNT_BOUNDARIES)]
            mode = (token + seed // len(COUNT_BOUNDARIES)) % 5
            if mode == 0:
                raw_count = 0
            row = make_row(raw_count, mode, threshold, kv_len,
                           address_limit, generator)
            # Make mode 0 observably different from mode 3: these tail tokens
            # are valid in unmasked mode but causal-invalid for this query.
            if case.sparse_mode == 0 and row.numel():
                causal_threshold = max(0, kv_len - q_len + local_q + 1)
                mode0_only = torch.arange(
                    causal_threshold, kv_len, dtype=torch.int32)
                count = min(row.numel(), mode0_only.numel())
                row[:count] = mode0_only[:count]
            if row.numel():
                indices[token, 0, :row.numel()] = row
            raw_counts[token] = row.numel()
            effective_counts[token] = int(
                ((row >= 0) & (row < threshold)).sum())
            effective = row[(row >= 0) & (row < threshold)]
            has_duplicate_effective |= (
                effective.numel() != torch.unique(effective).numel())
            thresholds[token] = threshold
            token += 1

    seq_q = torch.tensor(case.q_lens, dtype=torch.int32).cumsum(
        0, dtype=torch.int32)
    seq_kv = torch.tensor(case.kv_lens, dtype=torch.int32)
    return {
        "query": query, "key": key, "value": value,
        "indices": indices, "query_rope": query_rope,
        "key_rope": key_rope, "block_table": block_table,
        "seq_q": seq_q, "seq_kv": seq_kv,
        "raw_counts": raw_counts, "effective_counts": effective_counts,
        "thresholds": thresholds,
        "has_duplicate_effective": has_duplicate_effective,
    }


def invoke(npu, case):
    return torch_npu.npu_sparse_flash_attention_decode(
        npu["query"], npu["key"], npu["value"], npu["indices"], case.scale,
        block_table=npu["block_table"],
        actual_seq_lengths_query=npu["seq_q"],
        actual_seq_lengths_kv=npu["seq_kv"],
        query_rope=npu["query_rope"], key_rope=npu["key_rope"],
        sparse_block_size=1, layout_query="TND", layout_kv="PA_BSND",
        sparse_mode=case.sparse_mode, pre_tokens=case.pre_tokens,
        next_tokens=case.next_tokens, attention_mode=case.attention_mode,
        return_softmax_lse=True)


def cpu_reference(inputs, case):
    output = torch.zeros_like(inputs["query"], dtype=torch.float32)
    lse = torch.full((inputs["query"].shape[0], HEADS), -torch.inf)
    q_begin = 0
    for batch, q_end in enumerate(inputs["seq_q"].tolist()):
        for token in range(q_begin, q_end):
            raw = int(inputs["raw_counts"][token])
            threshold = int(inputs["thresholds"][token])
            positions = inputs["indices"][token, 0, :raw]
            positions = positions[(positions >= 0) &
                                  (positions < threshold)].long()
            if positions.numel() == 0:
                continue
            blocks = inputs["block_table"][batch, positions // PAGE].long()
            rows = positions % PAGE
            selected_k = inputs["key"][blocks, rows, 0].float()
            selected_v = inputs["value"][blocks, rows, 0].float()
            selected_rope = inputs["key_rope"][blocks, rows, 0].float()
            logits = ((inputs["query"][token].float() @ selected_k.T +
                       inputs["query_rope"][token].float() @ selected_rope.T) *
                      case.scale)
            output[token] = torch.softmax(logits, dim=-1) @ selected_v
            lse[token] = torch.logsumexp(logits, dim=-1)
        q_begin = q_end
    return output, lse


def validate(case, seed, args):
    started = time.time()
    inputs = make_inputs(case, seed)
    npu = {
        name: tensor.npu() for name, tensor in inputs.items()
        if name not in ("raw_counts", "effective_counts", "thresholds",
                        "has_duplicate_effective")
    }
    first = invoke(npu, case)
    torch_npu.npu.synchronize()
    start = torch_npu.npu.Event(enable_timing=True)
    end = torch_npu.npu.Event(enable_timing=True)
    start.record()
    second = invoke(npu, case)
    end.record()
    torch_npu.npu.synchronize()

    out0, max0, sum0 = first
    out1, max1, sum1 = second
    lse0 = max0.float() + torch.log(sum0.float())
    lse1 = max1.float() + torch.log(sum1.float())
    out_repeat = (out0.float() - out1.float()).abs().max().item()
    same_inf = torch.isinf(lse0) & torch.isinf(lse1) & (lse0 == lse1)
    lse_repeat = torch.where(
        same_inf, 0.0, (lse0 - lse1).abs()).max().item()
    if out_repeat != 0.0 or lse_repeat != 0.0:
        out_rows = (out0.float() - out1.float()).abs().reshape(
            -1, HEADS * LATENT).amax(dim=1).cpu()
        lse_rows = torch.where(
            same_inf, 0.0, (lse0 - lse1).abs()).reshape(
                -1, HEADS).amax(dim=1).cpu()
        bad = (out_rows != 0) | torch.isnan(lse_rows) | \
              torch.isinf(lse_rows) | (lse_rows != 0)
        details = [
            (token, int(inputs["raw_counts"][token]),
             int(inputs["effective_counts"][token]),
             int(inputs["thresholds"][token]),
             float(out_rows[token]), float(lse_rows[token]))
            for token in torch.nonzero(bad).flatten().tolist()
        ]
        raise AssertionError(
            f"non-deterministic output/lse={out_repeat}/{lse_repeat}; "
            "rows=(token,raw,effective,threshold,out,lse) "
            f"{details}")

    actual_out = out1.float().cpu()
    actual_lse = lse1.reshape(-1, HEADS).cpu()
    reference_out, reference_lse = cpu_reference(inputs, case)
    out_diff = (actual_out - reference_out).abs()
    same_inf = (torch.isinf(actual_lse) & torch.isinf(reference_lse) &
                (actual_lse == reference_lse))
    lse_diff = torch.where(
        same_inf, 0.0, (actual_lse - reference_lse).abs())
    if not torch.isfinite(actual_out).all():
        raise AssertionError("output contains NaN/Inf")
    if torch.isnan(lse_diff).any() or torch.isinf(lse_diff).any():
        bad_rows = (torch.isnan(lse_diff) | torch.isinf(lse_diff)).any(dim=1)
        details = [
            (token, int(inputs["raw_counts"][token]),
             int(inputs["effective_counts"][token]),
             int(inputs["thresholds"][token]),
             int(torch.isfinite(actual_lse[token]).sum()),
             int(torch.isnan(actual_lse[token]).sum()),
             int(torch.isposinf(actual_lse[token]).sum()),
             int(torch.isneginf(actual_lse[token]).sum()))
            for token in torch.nonzero(bad_rows).flatten().tolist()
        ]
        raise AssertionError(
            "LSE contains unexpected NaN/Inf; "
            "rows=(token,raw,effective,threshold,finite,nan,+inf,-inf) "
            f"{details}")

    out_max = out_diff.max().item()
    out_mean = out_diff.mean().item()
    lse_max = lse_diff.max().item()
    output_limit = (3e-3 if case.dtype == torch.float16
                    else args.output_max_atol)
    # Repeated BF16 probabilities need not sum to exactly one.  Keep the
    # strict default for normal (unique TopK) rows, while retaining duplicate
    # TopK as an explicit robustness case with a bounded mean-error allowance.
    mean_limit = args.output_mean_atol
    if case.dtype == torch.bfloat16 and inputs["has_duplicate_effective"]:
        mean_limit = max(mean_limit, 4e-3)
    if out_max > output_limit or out_mean > mean_limit:
        raise AssertionError(
            f"output diff={out_max}/{out_mean}, limits="
            f"{output_limit}/{mean_limit}")
    if lse_max > args.lse_max_atol:
        raise AssertionError(
            f"LSE diff={lse_max}, limit={args.lse_max_atol}")

    empty = inputs["effective_counts"] == 0
    if empty.any():
        if torch.count_nonzero(actual_out[empty]).item() != 0:
            raise AssertionError("zero-effective-KV row produced nonzero output")
        if not torch.isneginf(actual_lse[empty]).all():
            raise AssertionError("zero-effective-KV row did not produce -inf LSE")

    result = {
        "raw_min": int(inputs["raw_counts"].min()),
        "raw_max": int(inputs["raw_counts"].max()),
        "effective_min": int(inputs["effective_counts"].min()),
        "effective_max": int(inputs["effective_counts"].max()),
        "out_max": out_max, "lse_max": lse_max,
        "npu_ms": start.elapsed_time(end),
        "wall_s": time.time() - started,
    }
    del npu, first, second, actual_out, actual_lse
    del reference_out, reference_lse, inputs
    gc.collect()
    torch_npu.npu.empty_cache()
    return result


def validate_rejected(name, seed):
    if name == "block_table_10000":
        case = Case(name, (1,), (1,), table_width=10000)
    elif name == "batch_65":
        case = Case(name, tuple([1] * 65), tuple([1] * 65))
    elif name == "sparse_mode_2":
        case = Case(name, (1,), (1,), sparse_mode=2)
    elif name == "pre_tokens_0":
        case = Case(name, (1,), (1,), pre_tokens=0)
    elif name == "next_tokens_0":
        case = Case(name, (1,), (1,), next_tokens=0)
    elif name == "attention_mode_0":
        case = Case(name, (1,), (1,), attention_mode=0)
    else:
        case = Case(name, (1,), (1,))
    inputs = make_inputs(case, seed)
    if name == "sparse_2047":
        inputs["indices"] = inputs["indices"][:, :, :2047].contiguous()
    npu = {
        key: value.npu() for key, value in inputs.items()
        if key not in ("raw_counts", "effective_counts", "thresholds",
                       "has_duplicate_effective")
    }
    try:
        invoke(npu, case)
        torch_npu.npu.synchronize()
    except Exception as error:
        print(f"PASS rejected={name} error={type(error).__name__}: {error}",
              flush=True)
        return
    raise AssertionError(f"invalid shape was accepted: {name}")


def main():
    args = parse_args()
    if args.expect_rejected:
        validate_rejected(args.expect_rejected, args.seed)
        return
    selected = set(filter(None, args.cases.split(",")))
    cases = tuple(case for case in DIRECTED_CASES
                  if not selected or case.name in selected)
    missing = selected - {case.name for case in cases}
    if missing:
        raise ValueError(f"unknown cases: {sorted(missing)}")
    torch.npu.set_compile_mode(jit_compile=False)
    torch.set_num_threads(max(1, min(32, torch.get_num_threads())))
    stopping = False

    def request_stop(_signum, _frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    cycle = passed = failed = 0
    print("START cases=" + ",".join(case.name for case in cases),
          flush=True)
    while not stopping and (args.cycles == 0 or cycle < args.cycles):
        cycle += 1
        print(f"CYCLE {cycle} START", flush=True)
        for index, case in enumerate(cases):
            if stopping:
                break
            seed = args.seed + (cycle - 1) * len(cases) + index
            try:
                result = validate(case, seed, args)
                passed += 1
                print(
                    f"PASS cycle={cycle} case={case.name} seed={seed} "
                    f"q={case.q_lens} kv={case.kv_lens} "
                    f"mode={case.sparse_mode} scale={case.scale} "
                    f"raw={result['raw_min']}/{result['raw_max']} "
                    f"effective={result['effective_min']}/"
                    f"{result['effective_max']} out={result['out_max']:.6g} "
                    f"lse={result['lse_max']:.6g} npu={result['npu_ms']:.3f}ms "
                    f"wall={result['wall_s']:.2f}s total={passed}/{failed}",
                    flush=True)
            except Exception:
                failed += 1
                print(f"FAIL cycle={cycle} case={case.name} seed={seed} "
                      f"q={case.q_lens} kv={case.kv_lens} "
                      f"total={passed}/{failed}", flush=True)
                traceback.print_exc()
                if args.stop_on_failure:
                    raise
                gc.collect()
                try:
                    torch_npu.npu.empty_cache()
                except Exception:
                    pass
        print(f"CYCLE {cycle} END passed={passed} failed={failed}", flush=True)
    print(f"STOP cycles={cycle} passed={passed} failed={failed}", flush=True)


if __name__ == "__main__":
    main()
