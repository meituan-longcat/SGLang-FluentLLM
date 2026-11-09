#!/usr/bin/env bash
set -euo pipefail
script_dir=$(cd "$(dirname "$0")" && pwd)

# Preserve the generated installer, arguments and its normal vendor behavior.
bash "$script_dir/install_all_original.sh" "$@"

# Match the stock installer's destination precedence. Do not change its env.
requested=${INSTALL_PATH:-}
for argument in "$@"; do
    case "$argument" in
        --install-path=*) requested=${argument#*=}; requested=${requested%/} ;;
        --*) ;;
        *) break ;;
    esac
done
target=${requested:-${ASCEND_CUSTOM_OPP_PATH:-${ASCEND_OPP_PATH:-}}}
[[ "$target" == /* && "$target" != *:* ]] || { echo "Cannot identify installed OPP root" >&2; exit 2; }
python3 "$script_dir/offload_install/install_offload_manifest.py" \
    --source-vendor "$script_dir/packages/vendors/flash_ops" \
    --installed-vendor "$target/vendors/flash_ops"
