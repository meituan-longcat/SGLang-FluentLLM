#!/usr/bin/env bash
set -eo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/python:${PYTHONPATH:-}"
prefill_nodes_num=$1
prefill_nodes_size=$2
decode_nodes_num=$3
decode_nodes_size=$4
read -r -a ips <<< "${cluster_ip_list//,/ }"
prefill_urls=()
decode_urls=()
for ((i=0; i<decode_nodes_num; i++)); do
    decode_urls+=("http://${ips[i * decode_nodes_size]}:${SERVER_PORT:-6080}")
done
total_decode=$((decode_nodes_num * decode_nodes_size))
for ((i=0; i<prefill_nodes_num; i++)); do
    prefill_urls+=("http://${ips[total_decode + i * prefill_nodes_size]}:${SERVER_PORT:-6080}")
done
command=("${PYTHON_BIN:-python3}" -m sglang.srt.disaggregation.mini_lb \
    --host "${LB_HOST:-0.0.0.0}" --port "${LB_PORT:-8081}" \
    --prefill "${prefill_urls[@]}" --decode "${decode_urls[@]}")
if [[ "${DRY_RUN:-0}" == 1 ]]; then
    printf '%q ' "${command[@]}"
    printf '\n'
    exit 0
fi
exec "${command[@]}"
