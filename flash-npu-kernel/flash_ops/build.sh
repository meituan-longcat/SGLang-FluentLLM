#!/bin/bash
set -e
cd "$(dirname "$0")"

NPU_ARCH=${NPU_ARCH:-dav-2201}
PACKAGE_DIR="flash_npu_kernel"

usage() {
    cat <<EOF
Usage:
  $0                        Full build: produce wheel only (no install, no tests).
  $0 --ops=<a>[,<b>,...]    Incrementally build libflash_<op>.so for one or
                            more ops (no install, no tests). Names are separated
                            by ','. Example: --ops=get_out_cache_loc,compute_n_gram_ids
  $0 -h | --help            Show this help.

Available ops:
$(ls csrc | grep -v CMakeLists.txt | sed 's/^/  /')
EOF
}

# Site-packages flash_npu_kernel/, or empty if not installed. cwd is stripped
# from sys.path so running from the project root doesn't resolve to the source
# tree.
resolve_install_dir() {
    python3 - <<'PY' 2>/dev/null || true
import sys, os
sys.path = [p for p in sys.path if p not in ('', os.getcwd())]
try:
    import flash_npu_kernel
    print(os.path.dirname(flash_npu_kernel.__file__))
except Exception:
    pass
PY
}

configure_build() {
    [[ -f build/CMakeCache.txt ]] && return
    echo "Configuring build/ (first run)..."
    local torch_dir torch_npu_path
    torch_dir=$(python3 -c 'import os,torch;print(os.path.join(torch.utils.cmake_prefix_path,"Torch"))')
    torch_npu_path=$(python3 -c 'import os,torch_npu;print(os.path.dirname(torch_npu.__file__))')
    cmake -S . -B build \
        -DCMAKE_BUILD_TYPE=Release \
        -DTorch_DIR="$torch_dir" \
        -DTORCH_NPU_PATH="$torch_npu_path" \
        -DNPU_ARCH="$NPU_ARCH"
}

# Parse --ops=a,b,c. Sets global OP_NAMES.
parse_ops_arg() {
    local ops_arg="${1#--ops=}"
    OP_NAMES=()
    local _parts _p
    IFS=',' read -r -a _parts <<< "$ops_arg"
    for _p in "${_parts[@]}"; do
        [[ -n "$_p" ]] && OP_NAMES+=("$_p")
    done
    if [[ ${#OP_NAMES[@]} -eq 0 ]]; then
        echo "ERROR: --ops= requires at least one operator name" >&2
        usage
        exit 1
    fi
    for _p in "${OP_NAMES[@]}"; do
        if [[ ! -d "csrc/${_p}/${NPU_ARCH}" ]]; then
            echo "ERROR: op not found at csrc/${_p}/${NPU_ARCH}" >&2
            exit 1
        fi
    done
}

# Build every op in OP_NAMES. Sets global BUILT_SOS (absolute paths).
build_ops() {
    configure_build

    local targets=() op
    for op in "${OP_NAMES[@]}"; do targets+=("flash_${op}"); done
    echo "Building: ${targets[*]}"
    cmake --build build --target "${targets[@]}" --parallel "$(nproc)"

    BUILT_SOS=()
    for op in "${OP_NAMES[@]}"; do
        local so="${PACKAGE_DIR}/libflash_${op}.so"
        if [[ ! -f "$so" ]]; then
            echo "ERROR: build succeeded but $so not found" >&2
            exit 1
        fi
        BUILT_SOS+=("$(readlink -f "$so")")
    done
}

# Only print the site-packages cp hint — and only when there's an installed
# copy distinct from the source tree with stale/missing .so files.
print_ops_summary() {
    local src_dir install_dir
    src_dir="$(readlink -f "${PACKAGE_DIR}")"
    install_dir="$(resolve_install_dir)"
    [[ -z "$install_dir" || "$install_dir" == "$src_dir" ]] && return

    local stale=() op src dst
    for op in "${OP_NAMES[@]}"; do
        src="${src_dir}/libflash_${op}.so"
        dst="${install_dir}/libflash_${op}.so"
        if [[ ! -f "$dst" ]] || ! cmp -s "$src" "$dst"; then
            stale+=("$src")
        fi
    done
    [[ ${#stale[@]} -eq 0 ]] && return

    echo
    echo "Installed flash_npu_kernel in site-packages is stale (${#stale[@]}/${#OP_NAMES[@]} .so)."
    echo "Run the following to sync:"
    echo "  cp ${stale[*]} \"${install_dir}/\""
}

build_wheel() {
    echo "Installing build dependencies..."
    pip install -r requirements.txt

    echo "Building the wheel..."
    python3 setup.py clean
    NPU_ARCH="$NPU_ARCH" python3 -m build --wheel --no-isolation

    local wheel
    wheel="$(ls -t dist/*.whl 2>/dev/null | head -n 1 || true)"
    if [[ -z "$wheel" ]]; then
        echo "ERROR: build completed but no wheel found under dist/" >&2
        exit 1
    fi
    wheel="$(readlink -f "$wheel")"

    echo
    echo "============================================================"
    echo "Full build done."
    echo "============================================================"
    echo "Wheel:"
    echo "  ${wheel}"
    echo
    echo "Install it with:"
    echo "  pip install \"${wheel}\" --force-reinstall --no-deps"
    echo
    echo "Run tests after install (optional):"
    echo "  pytest tests/ -v"
    echo "============================================================"
}

case "$1" in
    -h|--help)
        usage
        exit 0
        ;;
    --ops=*)
        if [[ $# -gt 1 ]]; then
            echo "ERROR: unexpected argument after --ops=: $2" >&2
            usage
            exit 1
        fi
        parse_ops_arg "$1"
        build_ops
        print_ops_summary
        ;;
    *)
        build_wheel
        ;;
esac
