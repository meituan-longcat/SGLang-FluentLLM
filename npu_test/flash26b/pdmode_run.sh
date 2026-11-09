#!/usr/bin/env bash
# Decode hosts precede Prefill hosts in cluster_ip_list.
set -eo pipefail
prefill_nodes_num=$1
prefill_nodes_size=$2
decode_nodes_num=$3
decode_nodes_size=$4
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
read -r -a ips <<< "${cluster_ip_list//,/ }"
local_id=${Node_idx:?Set Node_idx.}
total_decode=$((decode_nodes_num * decode_nodes_size))
total_nodes=$((total_decode + prefill_nodes_num * prefill_nodes_size))
(( local_id >= 0 && local_id < total_nodes )) || { echo "Invalid Node_idx." >&2; exit 2; }
if (( local_id < total_decode )); then
    rank=$((local_id % decode_nodes_size))
    first=$((local_id / decode_nodes_size * decode_nodes_size))
    export MASTERIP="${ips[first]}"
    role_script=run_decode.sh
    node_count=${decode_nodes_size}
else
    offset=$((local_id - total_decode))
    rank=$((offset % prefill_nodes_size))
    first=$((total_decode + offset / prefill_nodes_size * prefill_nodes_size))
    export MASTERIP="${ips[first]}"
    role_script=run_prefill.sh
    node_count=${prefill_nodes_size}
fi
if [[ "${DRY_RUN:-0}" == 1 ]]; then
    bash "${SCRIPT_DIR}/${role_script}" "${rank}" "${node_count}"
    if (( local_id == total_nodes - 1 )); then
        bash "${SCRIPT_DIR}/run_minilb.sh" "$@"
    fi
    exit
fi
bash "${SCRIPT_DIR}/${role_script}" "${rank}" "${node_count}" &
worker_pid=$!
if (( local_id == total_nodes - 1 )); then
    bash "${SCRIPT_DIR}/run_minilb.sh" "$@" &
    router_pid=$!
fi
wait "${worker_pid}"
if [[ -n "${router_pid:-}" ]]; then wait "${router_pid}"; fi
