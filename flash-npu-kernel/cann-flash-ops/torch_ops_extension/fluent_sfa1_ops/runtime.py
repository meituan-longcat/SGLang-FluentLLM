"""Validate private offload files and API ownership without loading OpProto.

Existing flash_ops operators retain the normal vendor/provider contract. GE
loads the installed vendor through its usual path; importing offload must not
register the whole ALL prototype or rewrite a global provider priority.
"""
import hashlib
import json
from pathlib import Path


_PRIVATE_OPERATORS = (
    "FluentMlaPrologV3",
    "GenFromCacheIndices", "GatherByCacheIndices", "UpdateCache",
    "UpdateCurKvToSelection", "GatherByCacheIndicesPacked",
    "UpdateCachePacked", "UpdateCurKvToSelectionPacked",
)


def validate_vendor(extension):
    api = Path(extension._resolve_provider("aclnnFluentMlaPrologV3")).resolve(strict=True)
    vendor = api.parents[2]
    marker = vendor / "update_cache_alias_safe.json"
    data = json.loads(marker.read_text())
    if data.get("schema_version") != "update_cache_alias_safe_v1":
        raise RuntimeError("Unsupported SFA1 vendor manifest")
    files = data.get("files", {})
    api_relative = "op_api/lib/libcust_opapi.so"
    proto_relative = "op_proto/lib/linux/x86_64/libcust_opsproto_rt2.0.so"
    if api_relative not in files or proto_relative not in files:
        raise RuntimeError("SFA1 manifest must bind its op-api and OpProto files")
    if not any(path.endswith(".o") for path in files):
        raise RuntimeError("SFA1 manifest does not bind compiled kernels")
    for name, expected in files.items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError("Invalid SFA1 vendor manifest path")
        path = vendor / relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"SFA1 vendor manifest mismatch: {name}")

    operators = {}
    for operator in _PRIVATE_OPERATORS:
        for suffix in ("", "GetWorkspaceSize"):
            symbol = "aclnn" + operator + suffix
            actual = Path(extension._resolve_provider(symbol)).resolve(strict=True)
            if actual != api:
                raise RuntimeError(f"Private offload provider differs from sealed runtime: {symbol}: {actual}")
            operators[symbol] = str(actual)
    return {
        "vendor": str(vendor), "marker": str(marker), "provider": str(api),
        "operators": operators,
        "op_proto": [{"path": str(vendor / proto_relative),
                      "sha256": files[proto_relative], "loaded_by_import": False}],
        "GE_device_provider": "PENDING_NORMAL_VENDOR_LOADING_VALIDATION",
    }
