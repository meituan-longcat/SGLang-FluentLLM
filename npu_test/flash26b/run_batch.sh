#!/usr/bin/env bash
# Batch distribution and launch helper for run_w4a4kv8_mm_epd.sh.
set -eo pipefail
action="${1:-}"
case "${action}" in
    code|start|status) ;;
    *) echo "Usage: $0 code|start|status" >&2; exit 2 ;;
esac
: "${iplist:?Set iplist in run_w4a4kv8_mm_epd.sh.}"
: "${testpath:?Set testpath in run_w4a4kv8_mm_epd.sh.}"
: "${MODEL_PATH:?Set MODEL_PATH in run_w4a4kv8_mm_epd.sh.}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
codepath="$(cd "${SCRIPT_DIR}/../.." && pwd)"
log_tag="${time:-longcat25}"
[[ "${log_tag}" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo "Invalid log tag." >&2; exit 2; }
[[ "${testpath}" =~ ^/[a-zA-Z0-9_./-]+$ ]] || { echo "testpath must be an absolute path without spaces." >&2; exit 2; }
[[ "${testpath}" != *"/../"* && "${testpath}" != */.. ]] || { echo "Invalid testpath." >&2; exit 2; }
read -r -a nodes <<< "${iplist//$'\n'/ }"
expected=$((pnode * pnode_ep_size + dnode * dnode_ep_size))
[[ "${#nodes[@]}" -eq "${expected}" ]] || { echo "iplist must contain ${expected} hosts, Decode first." >&2; exit 2; }
for host in "${nodes[@]}"; do
    [[ "${host}" =~ ^[a-zA-Z0-9_][a-zA-Z0-9_.-]*$ ]] || { echo "Use hostnames or IPv4 addresses in iplist." >&2; exit 2; }
done
cluster_ip_list="$(IFS=,; printf '%s' "${nodes[*]}")"
runpath="${testpath%/}"
remote_repo="${runpath}/SGLang-FluentLLM"
ssh_options=(-o BatchMode=yes -o StrictHostKeyChecking=yes)
transport="ssh -o BatchMode=yes -o StrictHostKeyChecking=yes"
if [[ -n "${SSH_USER:-}" ]]; then
    [[ "${SSH_USER}" =~ ^[a-zA-Z0-9_][a-zA-Z0-9_.-]*$ ]] || {
        echo "Invalid SSH_USER." >&2; exit 2;
    }
fi
if [[ -n "${SSH_PORT:-}" ]]; then
    [[ "${SSH_PORT}" =~ ^[0-9]+$ ]] || { echo "Invalid SSH_PORT." >&2; exit 2; }
    port=$((10#${SSH_PORT}))
    (( port > 0 && port < 65536 )) || { echo "Invalid SSH_PORT." >&2; exit 2; }
    ssh_options+=(-p "${port}")
    transport+=" -p ${port}"
fi
run_command() {
    if [[ "${DRY_RUN:-0}" == 1 ]]; then
        printf '%q ' "$@"; printf '\n'
    else
        "$@"
    fi
}
for index in "${!nodes[@]}"; do
    target="${nodes[index]}"
    if [[ -n "${SSH_USER:-}" ]]; then target="${SSH_USER}@${target}"; fi
    case "${action}" in
        code)
            printf -v prepare 'mkdir -p %q' "${remote_repo}"
            run_command ssh "${ssh_options[@]}" "${target}" "${prepare}"
            run_command rsync -a -e "${transport}" \
                --exclude=.git --exclude=logs --exclude=__pycache__ \
                --exclude=abc.txt \
                "${codepath}/" "${target}:${remote_repo}/"
            ;;
        start)
            remote_script=""
            for name in MODEL_PATH served_model_name prefill_args decode_args engine_subfix_cmd \
                MAX_MODEL_LEN VIDEO_MAX_FRAMES VIDEO_MAX_PIXELS VIDEO_MIN_PIXELS \
                HCCL_CONNECT_TIMEOUT SGLANG_DISAGGREGATION_WAITING_TIMEOUT \
                SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT PYTHON_BIN; do
                printf -v assignment 'export %s=%q\n' "${name}" "${!name}"
                remote_script+="${assignment}"$'\n'
            done
            for name in SERVER_PORT DIST_PORT LB_HOST LB_PORT ASCEND_TOOLKIT_ENV ATB_ENV \
                HCCN_CONF_FILE; do
                if [[ -v "${name}" ]]; then
                    printf -v assignment 'export %s=%q\n' "${name}" "${!name}"
                    remote_script+="${assignment}"$'\n'
                fi
            done
            printf -v assignment 'export cluster_ip_list=%q Node_idx=%q\n' "${cluster_ip_list}" "${index}"
            remote_script+="${assignment}"$'\n'
            printf -v assignment 'cd %q\n' "${remote_repo}"
            remote_script+="${assignment}"$'\n'
            printf -v assignment 'export ASCEND_PROCESS_LOG_PATH=%q\n' "${runpath}/log_${log_tag}"
            remote_script+="${assignment}"$'\n'
            printf -v assignment 'mkdir -p %q\n' "${runpath}/log_${log_tag}"
            remote_script+="${assignment}"$'\n'
            printf -v assignment 'nohup bash npu_test/flash26b/pdmode_run.sh %q %q %q %q > %q 2>&1 < /dev/null &\n' \
                "${pnode}" "${pnode_ep_size}" "${dnode}" "${dnode_ep_size}" \
                "${runpath}/testlog_${log_tag}.log"
            remote_script+="${assignment}"
            printf -v remote_command 'bash -lc %q' "set -e"$'\n'"${remote_script}"
            run_command ssh "${ssh_options[@]}" "${target}" "${remote_command}"
            ;;
        status)
            printf -v remote_command 'tail -n 30 %q' "${runpath}/testlog_${log_tag}.log"
            run_command ssh "${ssh_options[@]}" "${target}" "${remote_command}"
            ;;
    esac
done
