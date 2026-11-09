#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${EP128_CONFIG:-${SCRIPT_DIR}/ep128_cluster.conf}"
source "${CONFIG_FILE}"

read -r -a BS_LIST <<< "${*:-1 2 4 8}"
BASE_PORT="${EP128_BASE_PORT:-41011}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
DISPATCH_DTYPE="${DISPATCH_DTYPE:-bf16}"
REMOTE_TIMEOUT="${REMOTE_TIMEOUT:-240}"
WARMUP_ITERS="${WARMUP_ITERS:-20}"
BENCH_ITERS="${BENCH_ITERS:-200}"
ASCEND_GLOBAL_LOG_LEVEL="${ASCEND_GLOBAL_LOG_LEVEL:-3}"
ASCEND_SLOG_PRINT_TO_STDOUT="${ASCEND_SLOG_PRINT_TO_STDOUT:-0}"
DISPATCH_PROFILE="${DISPATCH_PROFILE:-0}"
DISPATCH_EAGER="${DISPATCH_EAGER:-0}"
DISPATCH_SPARSE="${DISPATCH_SPARSE:-0}"
GLOBAL_BS_OVERRIDE="${GLOBAL_BS_OVERRIDE:-0}"
DISPATCH_TIMING_MODE="${DISPATCH_TIMING_MODE:-legacy}"
DISPATCH_HOST_TIMELINE="${DISPATCH_HOST_TIMELINE:-0}"
DISPATCH_CPU_AFFINITY="${DISPATCH_CPU_AFFINITY:-0}"
DISPATCH_BURST="${DISPATCH_BURST:-1}"
DISPATCH_COMM_ALG="${DISPATCH_COMM_ALG:-hierarchy}"
DISPATCH_HCCL_OP_EXPANSION_MODE="${DISPATCH_HCCL_OP_EXPANSION_MODE:-0}"
DISPATCH_RANK_BARRIER="${DISPATCH_RANK_BARRIER:-1}"
DISPATCH_MOE_STREAM="${DISPATCH_MOE_STREAM:-0}"
MOE_EXPERTS_OVERRIDE="${MOE_EXPERTS_OVERRIDE:-768}"
COPY_EXPERT_OVERRIDE="${COPY_EXPERT_OVERRIDE:-0}"
DISPATCH_ROUTE_PATTERN="${DISPATCH_ROUTE_PATTERN:-uniform_unique_topk}"

if (( ${#NODES[@]} != 8 )); then
    echo "EP128 requires exactly 8 nodes; got ${#NODES[@]}" >&2
    exit 1
fi
if [[ ! "${DISPATCH_DTYPE}" =~ ^(bf16|int8)$ ]]; then
    echo "DISPATCH_DTYPE must be bf16 or int8; got ${DISPATCH_DTYPE}" >&2
    exit 1
fi

remote_ssh() {
    SSHPASS="${SSH_PASSWORD}" sshpass -e ssh "${SSH_OPTIONS[@]}" "$@"
}

remote_scp() {
    SSHPASS="${SSH_PASSWORD}" sshpass -e scp "${SCP_OPTIONS[@]}" "$@"
}

cleanup_all() {
    local failed=0
    local jobs=()
    for target in "${NODES[@]}"; do
        remote_ssh "${target}" '
            pids=$(pgrep -f "[e]p128_dispatch_e2e.py" || true)
            test -z "$pids" || kill -TERM $pids 2>/dev/null || true
            sleep 1
            pids=$(pgrep -f "[e]p128_dispatch_e2e.py" || true)
            test -z "$pids" || kill -KILL $pids 2>/dev/null || true
        ' &
        jobs+=("$!")
    done
    for job in "${jobs[@]}"; do wait "${job}" || failed=1; done
    return "${failed}"
}

verify_clean() {
    local failed=0
    for rank in "${!NODES[@]}"; do
        local count
        count=$(remote_ssh "${NODES[$rank]}" \
            'pgrep -f "[e]p128_dispatch_e2e.py" | wc -l') || return 1
        if [[ "${count}" != "0" ]]; then
            echo "Node ${rank} still has ${count} related process(es)." >&2
            failed=1
        fi
    done
    return "${failed}"
}

trap 'cleanup_all || true' EXIT INT TERM

echo "[setup] Cleaning previous EP128 test processes..."
cleanup_all
verify_clean

echo "[setup] Synchronizing ep128_dispatch_e2e.py..."
for target in "${NODES[@]}"; do
    remote_ssh "${target}" "mkdir -p '${REMOTE_DIR}/logs' '${REMOTE_DIR}/profiles'"
    remote_scp "${SCRIPT_DIR}/ep128_dispatch_e2e.py" \
        "${target}:${REMOTE_DIR}/ep128_dispatch_e2e.py"
done

for index in "${!BS_LIST[@]}"; do
    bs="${BS_LIST[$index]}"
    if [[ ! "${bs}" =~ ^[1-9][0-9]*$ ]]; then
        echo "Invalid BS: ${bs}" >&2
        exit 1
    fi
    port=$((BASE_PORT + index * 10))
    result_name="${RUN_TAG}_${DISPATCH_DTYPE}_ep128_bs${bs}_precision_perf.json"
    profile_tag="${RUN_TAG}_${DISPATCH_DTYPE}_bs${bs}"

    echo "[BS=${bs}] Verifying clean state..."
    cleanup_all
    verify_clean
    if remote_ssh "${NODES[0]}" "ss -ltn | grep -q ':${port} '"; then
        echo "Port ${port} is already in use on rank 0." >&2
        exit 1
    fi

    launcher_pids=()
    echo "[BS=${bs}] Starting EP128 on port ${port}..."
    for rank in "${!NODES[@]}"; do
        target="${NODES[$rank]}"
        log="${REMOTE_DIR}/logs/${RUN_TAG}_bs${bs}_node${rank}.log"
        status="${log}.status"
        printf -v command \
            'cd %q; rm -f %q; set +u; source %q; set -u; export LD_LIBRARY_PATH=/usr/local/lib:/usr/local/Ascend/driver/lib64/driver:${LD_LIBRARY_PATH:-}; export ASCEND_OPS_TRANSFORMER_PATH=/usr/local/Ascend/latest/ops_transformer PYTORCH_NPU_ALLOC_CONF=expandable_segments:True,base_addr_aligned_kb:0 HCCL_DETERMINISTIC=True HCCL_IF_BASE_PORT=65100 HCCL_EXEC_TIMEOUT=120 HCCL_RDMA_TC=100 HCCL_RDMA_SL=3 HCCL_INTRA_PCIE_ENABLE=1 HCCL_INTRA_ROCE_ENABLE=0 HCCL_BUFFSIZE=2048 ASCEND_GLOBAL_LOG_LEVEL=%q ASCEND_SLOG_PRINT_TO_STDOUT=%q; set +e; TEST_BS=%q DISPATCH_DTYPE=%q WARMUP_ITERS=%q BENCH_ITERS=%q BENCH_OUTPUT=%q DISPATCH_PROFILE=%q PROFILE_OUTPUT_DIR=%q PROFILE_RUN_TAG=%q DISPATCH_TIMING_MODE=%q DISPATCH_HOST_TIMELINE=%q DISPATCH_CPU_AFFINITY=%q DISPATCH_BURST=%q DISPATCH_EAGER=%q DISPATCH_SPARSE=%q GLOBAL_BS_OVERRIDE=%q DISPATCH_COMM_ALG=%q DISPATCH_HCCL_OP_EXPANSION_MODE=%q DISPATCH_RANK_BARRIER=%q DISPATCH_MOE_STREAM=%q MOE_EXPERTS_OVERRIDE=%q COPY_EXPERT_OVERRIDE=%q DISPATCH_ROUTE_PATTERN=%q timeout --signal=TERM --kill-after=20s %qs torchrun --nproc_per_node=16 --nnodes=8 --node_rank=%q --master_addr=%q --master_port=%q %q >%q 2>&1; rc=$?; echo $rc >%q; exit $rc' \
            "${REMOTE_DIR}" "${status}" "${ASCEND_ENV}" "${ASCEND_GLOBAL_LOG_LEVEL}" "${ASCEND_SLOG_PRINT_TO_STDOUT}" "${bs}" \
            "${DISPATCH_DTYPE}" "${WARMUP_ITERS}" "${BENCH_ITERS}" "${REMOTE_DIR}/${result_name}" \
            "${DISPATCH_PROFILE}" "${REMOTE_DIR}/profiles" "${profile_tag}" \
            "${DISPATCH_TIMING_MODE}" \
            "${DISPATCH_HOST_TIMELINE}" \
            "${DISPATCH_CPU_AFFINITY}" \
            "${DISPATCH_BURST}" "${DISPATCH_EAGER}" "${DISPATCH_SPARSE}" "${GLOBAL_BS_OVERRIDE}" "${DISPATCH_COMM_ALG}" "${DISPATCH_HCCL_OP_EXPANSION_MODE}" \
            "${DISPATCH_RANK_BARRIER}" "${DISPATCH_MOE_STREAM}" \
            "${MOE_EXPERTS_OVERRIDE}" "${COPY_EXPERT_OVERRIDE}" "${DISPATCH_ROUTE_PATTERN}" \
            "${REMOTE_TIMEOUT}" "${rank}" "${MASTER_ADDR}" "${port}" \
            "${REMOTE_DIR}/ep128_dispatch_e2e.py" "${log}" "${status}"
        remote_ssh "${target}" "${command}" &
        launcher_pids+=("$!")
    done

    failed=0
    for pid in "${launcher_pids[@]}"; do wait "${pid}" || failed=1; done

    echo "[BS=${bs}] Cleaning all related processes..."
    cleanup_all
    verify_clean
    if (( failed != 0 )); then
        echo "BS=${bs} failed; check ${REMOTE_DIR}/logs/${RUN_TAG}_bs${bs}_node*.log" >&2
        exit 1
    fi

    remote_scp "${NODES[0]}:${REMOTE_DIR}/${result_name}" "${SCRIPT_DIR}/${result_name}"
    echo "[BS=${bs}] Complete: ${SCRIPT_DIR}/${result_name}"

    # 看护自检：校验精度结果, 输出 PASS/FAIL
    if python3 - "$SCRIPT_DIR/${result_name}" << 'PYEOF'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as f:
    d = json.load(f)
p = d["precision"]
ok = bool(p.get("dispatch_all_ranks_passed")) and bool(p.get("combine_allclose_all_ranks"))
print(f"[E2E-Check] BS={d.get('bs_per_rank')} alg={d.get('comm_alg')} "
      f"eager={d.get('execution_mode') or d.get('dispatch_eager')} -> "
      f"{'PASS' if ok else 'FAIL'}")
sys.exit(0 if ok else 1)
PYEOF
    then
        true
    else
        echo "[E2E-Check] precision check failed for BS=${bs}" >&2
        exit 1
    fi
done

trap - EXIT INT TERM
echo "All requested BS cases completed, and all related processes are clean."
