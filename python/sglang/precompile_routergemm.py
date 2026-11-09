"""
Simple RouterGEMM precompile entrypoint.

Only sweeps num_tokens and num_experts. For every shape, it launches
`dsv3_router_gemm` twice: once with BF16 weights and once with FP32 weights.

Example:
  python -m sglang.precompile_routergemm \
      --num-tokens-list 1,2,4,8,16,64 \
      --num-experts-list 256,384
"""

from __future__ import annotations

import argparse
import os
import re
import sys

import torch

HIDDEN_DIM = 8192


def parse_int_list(raw: str | None, name: str) -> list[int]:
    if raw is None or raw.strip() == "":
        return []

    values: list[int] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if ".." not in item:
            values.append(int(item))
            continue

        match = re.fullmatch(r"(\d+)\.\.(\d+)(?::(\d+))?", item)
        if match is None:
            raise ValueError(f"Invalid {name} range: {item}")
        start = int(match.group(1))
        end = int(match.group(2))
        step = int(match.group(3) or 1)
        values.extend(range(start, end + 1, step))

    values = list(dict.fromkeys(values))
    if any(value <= 0 for value in values):
        raise ValueError(f"{name} values must be positive: {values}")
    return values


def import_router_gemm():
    try:
        from flashinfer.dsv3_gemm import dsv3_router_gemm

        return dsv3_router_gemm
    except ModuleNotFoundError:
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        flashinfer_path = os.path.join(repo_root, "3rdparty", "flashinfer")
        if flashinfer_path not in sys.path:
            sys.path.insert(0, flashinfer_path)
        from flashinfer.dsv3_gemm import dsv3_router_gemm

        return dsv3_router_gemm


def launch_router_gemm(dsv3_router_gemm, num_tokens: int, num_experts: int, weight_dtype: torch.dtype) -> None:
    mat_a = torch.randn(
        (num_tokens, HIDDEN_DIM),
        dtype=torch.bfloat16,
        device="cuda",
    ).contiguous()
    mat_b = torch.randn(
        (num_experts, HIDDEN_DIM),
        dtype=weight_dtype,
        device="cuda",
    ).contiguous()

    out = dsv3_router_gemm(mat_a, mat_b, out_dtype=torch.float32)
    if out.shape != (num_tokens, num_experts):
        raise RuntimeError(f"unexpected output shape: {out.shape}")
    torch.cuda.synchronize()


def run(args: argparse.Namespace) -> None:
    num_tokens_list = parse_int_list(args.num_tokens_list, "--num-tokens-list")
    num_experts_list = parse_int_list(args.num_experts_list, "--num-experts-list")
    if not num_tokens_list or not num_experts_list:
        raise ValueError("--num-tokens-list and --num-experts-list are required")

    total = len(num_tokens_list) * len(num_experts_list) * 2
    print("Begin RouterGEMM precompile")
    print(f"  hidden_dim: {HIDDEN_DIM}")
    print(f"  total_shapes: {total}")

    if args.dry_run:
        for num_tokens in num_tokens_list:
            for num_experts in num_experts_list:
                print(f"[dry-run] num_tokens={num_tokens}, num_experts={num_experts}, mat_b=bf16")
                print(f"[dry-run] num_tokens={num_tokens}, num_experts={num_experts}, mat_b=fp32")
        return

    dsv3_router_gemm = import_router_gemm()

    i = 0
    for num_tokens in num_tokens_list:
        for num_experts in num_experts_list:
            for dtype_name, weight_dtype in (("bf16", torch.bfloat16), ("fp32", torch.float32)):
                i += 1
                print(
                    f"[{i}/{total}] launch num_tokens={num_tokens}, num_experts={num_experts}, mat_b={dtype_name}",
                    flush=True,
                )
                launch_router_gemm(dsv3_router_gemm, num_tokens, num_experts, weight_dtype)

    print("RouterGEMM precompile finished")


def add_cli_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--num-tokens-list", default="1,4,16,64,128,256,512,1024,1536,2048")
    parser.add_argument("--num-experts-list", default="256,768,896")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_cli_args(parser)
    run(parser.parse_args())


if __name__ == "__main__":
    main()

# python -m sglang.precompile_routergemm
