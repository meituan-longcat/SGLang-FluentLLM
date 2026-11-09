"""
Simple DeepGEMM JIT precompile entrypoint.

This script does not launch an SGLang server, load a model, or manage SGLang
environment variables. It directly allocates dummy CUDA tensors for requested
shapes and invokes DeepGEMM operators once. That operator launch is enough to
trigger DeepGEMM JIT compilation and fill DeepGEMM's own cache.

Examples:
  # Dense FP8 GEMM for all M values against one N/K pair.
  python -m sglang.precompile_kernels \
      --ops dense \
      --m-list 1,2,4,8,16,32,64,128 \
      --n-list 4096 \
      --k-list 7168

  # Grouped masked MoE GEMM for decode-like expected_m values.
  python -m sglang.precompile_kernels \
      --ops grouped-masked \
      --num-groups-list 256 \
      --max-m-list 128 \
      --expected-m-list 1,2,4,8,16,32,64 \
      --n-list 4096 \
      --k-list 7168

  # Explicit shapes can be mixed with list-generated ranges.
  python -m sglang.precompile_kernels \
      --ops dense,grouped-contiguous \
      --dense-shape 1,4096,7168 \
      --grouped-contiguous-shape 8,128,4096,7168
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import re
from collections.abc import Iterable, Sequence

import torch


@dataclasses.dataclass(frozen=True)
class DenseShape:
    m: int
    n: int
    k: int


@dataclasses.dataclass(frozen=True)
class GroupedContiguousShape:
    num_groups: int
    m_per_group: int
    n: int
    k: int


@dataclasses.dataclass(frozen=True)
class GroupedMaskedShape:
    num_groups: int
    max_m: int
    expected_m: int
    n: int
    k: int


def ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def align(x: int, y: int) -> int:
    return ceil_div(x, y) * y


def dedup_keep_order(items: Iterable):
    seen = set()
    output = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        output.append(item)
    return output


def parse_int_list(raw: str | None, name: str) -> list[int]:
    """Parse comma-separated ints and inclusive ranges like 1..128:8."""

    if raw is None or raw.strip() == "":
        return []

    values: list[int] = []
    for chunk in raw.split(","):
        token = chunk.strip()
        if not token:
            continue

        if ".." not in token:
            values.append(int(token))
            continue

        match = re.fullmatch(r"(-?\d+)\.\.(-?\d+)(?::(-?\d+))?", token)
        if match is None:
            raise ValueError(f"Invalid {name} range: {token!r}")
        start = int(match.group(1))
        end = int(match.group(2))
        step = int(match.group(3) or (1 if end >= start else -1))
        if step == 0:
            raise ValueError(f"Invalid {name} range with zero step: {token!r}")
        if start < end and step < 0:
            raise ValueError(f"Invalid ascending {name} range with negative step: {token!r}")
        if start > end and step > 0:
            raise ValueError(f"Invalid descending {name} range with positive step: {token!r}")

        stop = end + (1 if step > 0 else -1)
        values.extend(range(start, stop, step))

    values = dedup_keep_order(values)
    for value in values:
        if value <= 0:
            raise ValueError(f"{name} values must be positive, got {value}")
    return values


def parse_shape(raw: str, expected_len: int, name: str) -> tuple[int, ...]:
    parts = [part for part in re.split(r"[x,:]", raw.strip()) if part]
    if len(parts) != expected_len:
        raise ValueError(f"{name} expects {expected_len} integers, got {len(parts)} from {raw!r}")
    values = tuple(int(part) for part in parts)
    if any(value <= 0 for value in values):
        raise ValueError(f"{name} values must be positive, got {values}")
    return values


def parse_ops(raw: str) -> list[str]:
    valid_ops = {"dense", "grouped-contiguous", "grouped-masked"}
    ops = [op.strip() for op in raw.split(",") if op.strip()]
    unknown = sorted(set(ops) - valid_ops)
    if unknown:
        raise ValueError(f"Unknown op(s): {unknown}; valid ops: {sorted(valid_ops)}")
    return dedup_keep_order(ops)


def require_cuda(device_id: int) -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("DeepGEMM JIT precompile requires CUDA")
    if device_id < 0 or device_id >= torch.cuda.device_count():
        raise ValueError(f"Invalid CUDA device {device_id}; device_count={torch.cuda.device_count()}")
    torch.cuda.set_device(device_id)
    return torch.device(f"cuda:{device_id}")


def make_fp8(shape: Sequence[int], device: torch.device) -> torch.Tensor:
    return torch.zeros(tuple(shape), device=device, dtype=torch.float8_e4m3fn)


def make_fp32(shape: Sequence[int], device: torch.device) -> torch.Tensor:
    return torch.ones(tuple(shape), device=device, dtype=torch.float32)


def validate_nk(n: int, k: int, context: str) -> None:
    if n % 64 != 0:
        raise ValueError(f"{context}: N must be divisible by 64 for deep_gemm, got {n}")
    if k % 128 != 0:
        raise ValueError(f"{context}: K must be divisible by 128 for deep_gemm, got {k}")


def precompile_dense(shape: DenseShape, device: torch.device, use_pdl: bool) -> None:
    import deep_gemm

    validate_nk(shape.n, shape.k, f"dense {shape}")

    lhs = make_fp8((shape.m, shape.k), device)
    lhs_scales = make_fp32((shape.m, ceil_div(shape.k, 128)), device)
    rhs = make_fp8((shape.n, shape.k), device)
    rhs_scales = make_fp32((ceil_div(shape.n, 128), ceil_div(shape.k, 128)), device)
    out = torch.empty((shape.m, shape.n), device=device, dtype=torch.bfloat16)

    deep_gemm.gemm_fp8_fp8_bf16_nt((lhs, lhs_scales), (rhs, rhs_scales), out, use_pdl)
    torch.cuda.synchronize(device)


def precompile_grouped_contiguous(shape: GroupedContiguousShape, device: torch.device, use_pdl: bool) -> None:
    import deep_gemm

    validate_nk(shape.n, shape.k, f"grouped-contiguous {shape}")

    m_per_group = align(shape.m_per_group, 128)
    total_m = shape.num_groups * m_per_group
    lhs = make_fp8((total_m, shape.k), device)
    lhs_scales = make_fp32((total_m, ceil_div(shape.k, 128)), device)
    rhs = make_fp8((shape.num_groups, shape.n, shape.k), device)
    rhs_scales = make_fp32((shape.num_groups, ceil_div(shape.n, 128), ceil_div(shape.k, 128)), device)
    out = torch.empty((total_m, shape.n), device=device, dtype=torch.bfloat16)
    m_indices = torch.arange(shape.num_groups, device=device, dtype=torch.int32)
    m_indices = m_indices.repeat_interleave(m_per_group).contiguous()

    deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_contiguous((lhs, lhs_scales), (rhs, rhs_scales), out, m_indices, use_pdl)
    torch.cuda.synchronize(device)


def precompile_grouped_masked(shape: GroupedMaskedShape, device: torch.device, use_pdl: bool) -> None:
    import deep_gemm

    validate_nk(shape.n, shape.k, f"grouped-masked {shape}")

    max_m = align(shape.max_m, 128)
    if shape.expected_m > max_m:
        raise ValueError(f"expected_m must be <= aligned max_m, got {shape}")

    lhs = make_fp8((shape.num_groups, max_m, shape.k), device)
    lhs_scales = make_fp32((shape.num_groups, max_m, ceil_div(shape.k, 128)), device)
    rhs = make_fp8((shape.num_groups, shape.n, shape.k), device)
    rhs_scales = make_fp32((shape.num_groups, ceil_div(shape.n, 128), ceil_div(shape.k, 128)), device)
    out = torch.empty((shape.num_groups, max_m, shape.n), device=device, dtype=torch.bfloat16)
    masked_m = torch.full((shape.num_groups,), shape.expected_m, device=device, dtype=torch.int32)

    deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_masked((lhs, lhs_scales), (rhs, rhs_scales), out, masked_m, shape.expected_m, use_pdl)
    torch.cuda.synchronize(device)


def build_dense_shapes(args: argparse.Namespace) -> list[DenseShape]:
    shapes = [DenseShape(*parse_shape(raw, 3, "--dense-shape")) for raw in args.dense_shape]

    ms = parse_int_list(args.m_list, "--m-list")
    ns = parse_int_list(args.n_list, "--n-list")
    ks = parse_int_list(args.k_list, "--k-list")
    if ms and ns and ks:
        shapes.extend(DenseShape(m, n, k) for m, n, k in itertools.product(ms, ns, ks))
    elif any((ms, ns, ks)):
        raise ValueError("--m-list, --n-list, and --k-list must be provided together")

    return dedup_keep_order(shapes)


def build_grouped_contiguous_shapes(args: argparse.Namespace) -> list[GroupedContiguousShape]:
    shapes = [GroupedContiguousShape(*parse_shape(raw, 4, "--grouped-contiguous-shape")) for raw in args.grouped_contiguous_shape]

    groups = parse_int_list(args.num_groups_list, "--num-groups-list")
    m_per_groups = parse_int_list(args.m_per_group_list, "--m-per-group-list")
    ns = parse_int_list(args.n_list, "--n-list")
    ks = parse_int_list(args.k_list, "--k-list")
    if groups and m_per_groups and ns and ks:
        shapes.extend(
            GroupedContiguousShape(num_groups, m_per_group, n, k) for num_groups, m_per_group, n, k in itertools.product(groups, m_per_groups, ns, ks)
        )
    elif any((groups, m_per_groups)):
        raise ValueError("--num-groups-list, --m-per-group-list, --n-list, and --k-list must be provided together for grouped-contiguous")

    return dedup_keep_order(shapes)


def build_grouped_masked_shapes(args: argparse.Namespace) -> list[GroupedMaskedShape]:
    shapes = [GroupedMaskedShape(*parse_shape(raw, 5, "--grouped-masked-shape")) for raw in args.grouped_masked_shape]

    groups = parse_int_list(args.num_groups_list, "--num-groups-list")
    max_ms = parse_int_list(args.max_m_list, "--max-m-list")
    expected_ms = parse_int_list(args.expected_m_list, "--expected-m-list")
    ns = parse_int_list(args.n_list, "--n-list")
    ks = parse_int_list(args.k_list, "--k-list")
    if groups and expected_ms and ns and ks:
        if not max_ms:
            max_ms = [128]
        shapes.extend(
            GroupedMaskedShape(num_groups, max_m, expected_m, n, k)
            for num_groups, max_m, expected_m, n, k in itertools.product(groups, max_ms, expected_ms, ns, ks)
            if expected_m <= align(max_m, 128)
        )
    elif any((groups, max_ms, expected_ms)):
        raise ValueError(
            "--num-groups-list, --expected-m-list, --n-list, and --k-list must be provided together for grouped-masked; --max-m-list is optional"
        )

    return dedup_keep_order(shapes)


def run(args: argparse.Namespace) -> None:
    device = require_cuda(args.device)
    ops = parse_ops(args.ops)

    dense_shapes = build_dense_shapes(args) if "dense" in ops else []
    grouped_contiguous_shapes = build_grouped_contiguous_shapes(args) if "grouped-contiguous" in ops else []
    grouped_masked_shapes = build_grouped_masked_shapes(args) if "grouped-masked" in ops else []

    total = len(dense_shapes) + len(grouped_contiguous_shapes) + len(grouped_masked_shapes)
    if total == 0:
        raise ValueError("No shapes selected for direct DeepGEMM operator launch")

    print("Begin direct DeepGEMM operator precompile")
    print(f"  device: cuda:{args.device}")
    print(f"  use_pdl: {args.use_pdl}")
    print(f"  total_shapes: {total}")

    if args.dry_run:
        for shape in dense_shapes:
            print(f"[dry-run] dense {shape}")
        for shape in grouped_contiguous_shapes:
            aligned = dataclasses.replace(shape, m_per_group=align(shape.m_per_group, 128))
            print(f"[dry-run] grouped-contiguous {aligned}")
        for shape in grouped_masked_shapes:
            aligned = dataclasses.replace(shape, max_m=align(shape.max_m, 128))
            print(f"[dry-run] grouped-masked {aligned}")
        return

    i = 0
    for shape in dense_shapes:
        i += 1
        print(f"[{i}/{total}] launch dense {shape}", flush=True)
        precompile_dense(shape, device, args.use_pdl)

    for shape in grouped_contiguous_shapes:
        i += 1
        print(f"[{i}/{total}] launch grouped-contiguous {shape}", flush=True)
        precompile_grouped_contiguous(shape, device, args.use_pdl)

    for shape in grouped_masked_shapes:
        i += 1
        print(f"[{i}/{total}] launch grouped-masked {shape}", flush=True)
        precompile_grouped_masked(shape, device, args.use_pdl)

    print("DeepGEMM operator precompile finished")


def add_cli_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--ops",
        default="dense",
        help="Comma-separated ops: dense,grouped-contiguous,grouped-masked",
    )
    parser.add_argument("--device", type=int, default=0, help="CUDA device index")
    parser.add_argument("--dry-run", action="store_true", help="Print shapes without launching kernels")
    parser.add_argument(
        "--use-pdl",
        dest="use_pdl",
        action="store_true",
        default=True,
        help="Launch DeepGEMM with PDL enabled",
    )
    parser.add_argument(
        "--no-pdl",
        dest="use_pdl",
        action="store_false",
        help="Launch DeepGEMM with PDL disabled",
    )

    parser.add_argument(
        "--dense-shape",
        action="append",
        default=[],
        metavar="M,N,K",
        help="Explicit dense shape; repeatable. Separators ',', ':', and 'x' are accepted.",
    )
    parser.add_argument(
        "--grouped-contiguous-shape",
        action="append",
        default=[],
        metavar="GROUPS,M_PER_GROUP,N,K",
        help="Explicit grouped-contiguous shape; repeatable.",
    )
    parser.add_argument(
        "--grouped-masked-shape",
        action="append",
        default=[],
        metavar="GROUPS,MAX_M,EXPECTED_M,N,K",
        help="Explicit grouped-masked shape; repeatable.",
    )

    parser.add_argument("--m-list", default=None, help="Dense M values/ranges, e.g. 1,2,4 or 1..128:8")
    parser.add_argument("--n-list", default=None, help="N values/ranges shared by selected ops")
    parser.add_argument("--k-list", default=None, help="K values/ranges shared by selected ops")
    parser.add_argument("--num-groups-list", default=None, help="Grouped num_groups values/ranges")
    parser.add_argument("--m-per-group-list", default=None, help="Grouped-contiguous M per group values/ranges")
    parser.add_argument("--max-m-list", default=None, help="Grouped-masked max_m values/ranges")
    parser.add_argument("--expected-m-list", default=None, help="Grouped-masked expected_m values/ranges")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_cli_args(parser)
    run(parser.parse_args())


if __name__ == "__main__":
    main()

# python -m sglang.precompile_deepgemm   --ops dense,grouped-masked,grouped-contiguous
#   --m-list 1,16,128,512,1024,2048,3072,4096   --num-groups-list 96   --max-m-list 128,256
#   --expected-m-list 1,64,128,256   --n-list 2048,4096   --k-list 8192 --m-per-group-list 128,256
# --dry-run
