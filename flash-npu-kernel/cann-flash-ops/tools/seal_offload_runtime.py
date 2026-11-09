"""Bind a private OPP package to the selected offload sources and runtime files.

This records package consistency, not numerical acceptance of a rebuilt binary.
Run the rolling NPU oracle after rebuilding. No global OPP files are installed.
"""

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def seal(vendor):
    vendor = vendor.resolve(strict=True)
    source = Path(__file__).resolve().parents[1] / "src"
    tbe = Path("op_impl/ai_core/tbe")
    implementation = tbe / "flash_ops_impl"
    files = set()
    for op in ("gen_from_cache_indices", "update_cache", "gather_by_cache_indices", "update_cur_kv_to_selection",
               "gather_by_cache_indices_packed", "update_cache_packed", "update_cur_kv_to_selection_packed",
               "kv_cache_copy_rows_burst"):
        jit = implementation / "ascendc" / op
        required_names = [p.name for p in (source / op / "op_kernel").iterdir()
                          if p.is_file() and p.suffix in (".cpp", ".h")]
        for filename in required_names:
            packaged = vendor / jit / filename
            if not packaged.is_file() or sha256(packaged) != sha256(
                source / op / "op_kernel" / filename
            ):
                raise ValueError(f"Packaged JIT source does not match checkout: {jit / filename}")
        kernels = list((vendor / tbe / "kernel" / "ascend910b" / op).rglob("*.o"))
        if not kernels:
            raise ValueError(f"Missing ascend910b kernel objects for {op}")
        files.update(path.relative_to(vendor) for path in kernels)
        for path in (vendor / jit).rglob("*"):
            if path.is_file():
                # Some packages contain both flat and op_kernel/ JIT sources.
                if path.name in required_names and sha256(path) != sha256(
                    source / op / "op_kernel" / path.name
                ):
                    raise ValueError(f"Stale duplicate JIT source: {path.relative_to(vendor)}")
                files.add(path.relative_to(vendor))
        files.add(implementation / "dynamic" / (op + ".py"))
        files.add(tbe / "kernel/config/ascend910b" / (op + ".json"))
    for op in ("mla_prolog_v3", "fluent_mla_prolog_v3",
               "sparse_flash_attention_decode"):
        checkout_dir = source / op / "op_kernel"
        required_sources = sorted(
            path for path in checkout_dir.iterdir()
            if path.is_file() and path.suffix in (".h", ".cpp")
        )
        if checkout_dir / (op + ".cpp") not in required_sources:
            raise ValueError(f"Missing checkout kernel entry for {op}")
        # CMake installs both layouts. Enumerate the checkout, so missing whole
        # directories or individual canonical/policy/adapter headers cannot pass.
        for layout in (Path(), Path("op_kernel")):
            for checkout in required_sources:
                relative = implementation / "ascendc" / op / layout / checkout.name
                packaged = vendor / relative
                if not packaged.is_file() or sha256(packaged) != sha256(checkout):
                    raise ValueError(f"Packaged policy source mismatch: {relative}")
                files.add(relative)
        kernels = list((vendor / tbe / "kernel/ascend910b" / op).rglob("*.o"))
        if not kernels:
            raise ValueError(f"Missing policy kernel objects for {op}")
        files.update(path.relative_to(vendor) for path in kernels)
    files.update(
        {
            Path("op_api/lib/libcust_opapi.so"),
            Path("op_proto/lib/linux/x86_64/libcust_opsproto_rt2.0.so"),
            tbe / "op_tiling/liboptiling.so",
            tbe / "op_tiling/lib/linux/x86_64/libcust_opmaster_rt2.0.so",
            tbe / "op_master_device/lib/libcust_opmaster.so",
            tbe / "kernel/config/ascend910b/binary_info_config.json",
        }
    )
    missing = sorted(str(path) for path in files if not (vendor / path).is_file())
    if missing:
        raise ValueError(f"Incomplete runtime: {missing}")
    manifest = {
        "schema_version": "update_cache_alias_safe_v1",
        "gather_cross_query_reuse": True,
        "packed_kv_abi_version": 1,
        "packed_kv_row_bytes": 672,
        "layer_copy_burst_abi_version": 1,
        "layer_copy_burst_row_bytes": 672,
        "layer_copy_burst_blocks": 4,
        "layer_copy_burst_max_layers": 38,
        "layer_copy_burst_max_rows": 256,
        "layer_copy_burst_row_bytes_supported": [672, 1024, 128],
        "layer_copy_burst_bf16_max_rows": 128,
        "update_cache_wide_row_copy": True,
        "update_cache_skip_unchanged_alias_prefix": True,
        "files": {str(path): sha256(vendor / path) for path in sorted(files)},
    }
    marker = vendor / "update_cache_alias_safe.json"
    marker.write_text(json.dumps(manifest, indent=2) + "\n")
    return marker, len(files)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("vendor", type=Path, help="Private x86_64 OPP vendors/flash_ops directory")
    args = parser.parse_args()
    marker, count = seal(args.vendor)
    print(json.dumps({"marker": str(marker), "files": count}))


if __name__ == "__main__":
    main()
