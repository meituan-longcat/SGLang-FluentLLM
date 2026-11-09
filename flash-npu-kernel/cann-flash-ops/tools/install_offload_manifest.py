"""Install only this package's offload seal after checking both byte sets."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile


def install_manifest(source_vendor, installed_vendor):
    source_vendor = Path(source_vendor).resolve(strict=True)
    installed_vendor = Path(installed_vendor).resolve(strict=True)
    marker_name = "update_cache_alias_safe.json"
    contents = (source_vendor / marker_name).read_bytes()
    manifest = json.loads(contents)
    files = manifest.get("files", {})
    if (manifest.get("schema_version") != "update_cache_alias_safe_v1"
            or "op_api/lib/libcust_opapi.so" not in files
            or "op_proto/lib/linux/x86_64/libcust_opsproto_rt2.0.so" not in files
            or not any(name.endswith(".o") for name in files)):
        raise ValueError("Incomplete private offload seal")
    for name, expected in files.items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Invalid private offload seal path")
        for vendor in (source_vendor, installed_vendor):
            path = vendor / relative
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError(f"Offload package/installation mismatch: {path}")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=installed_vendor, prefix=".offload-seal-", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(contents)
        temporary.chmod(0o644)
        os.replace(temporary, installed_vendor / marker_name)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return {"status": "MATCHING_OFFLOAD_SEAL_INSTALLED", "vendor": str(installed_vendor),
            "files_verified": len(files), "other_vendors_or_priority_modified": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-vendor", type=Path, required=True)
    parser.add_argument("--installed-vendor", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(install_manifest(args.source_vendor, args.installed_vendor)))


if __name__ == "__main__":
    main()
