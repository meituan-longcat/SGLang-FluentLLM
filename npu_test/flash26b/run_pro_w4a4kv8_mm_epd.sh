#!/usr/bin/env bash
# LongCat 2.5 W4A4 / KV INT8 multimodal EPD profile.
# P/D uses 192 NPUs; start the independent Encoder service separately.
set -eo pipefail

# 1. Action and deployment configuration.
action="${1:---help}"
case "${action}" in
    -h|--help)
        cat <<'EOF'
Usage: bash npu_test/flash26b/run_pro_w4a4kv8_mm_epd.sh ACTION [log_tag]
Actions: code | start | startp | startd | status | log
  code    Copy the checkout to each host's testpath/SGLang-FluentLLM directory.
  start   Start Decode, Prefill and MiniLB through pdmode_run.sh.
  startp  Start only Prefill hosts (including MiniLB on the last host).
  startd  Start only Decode hosts.
  status  Show the last 30 lines of each host's launch log.
  log     Collect launch logs under logdir/logs_<log_tag>.

Required environment:
  iplist      12 hosts, separated by spaces/newlines: 8 Decode, then 4 Prefill.
  MODEL_PATH  Absolute W4A4 P/D checkpoint directory on all hosts; no spaces or
              glob characters (required for start*, as used by the launch helpers).
  ENCODER_URLS Comma-separated HTTP(S) Encoder endpoints (required for start*).

Optional environment:
  testpath          Remote work directory (default: /data/longcat25).
  logdir            Local log directory (default: <repo>/logs/longcat25).
  time              Log tag (default: longcat25; second argument takes precedence).
  SSH_USER/SSH_PORT  SSH login (default: current user / port 22).
  SSH_PASSWORD_FILE Optional external sshpass password file; otherwise use SSH keys.
  DRY_RUN=1         Print commands without connecting to hosts or writing logs.

Example (replace the path and hosts):
  export MODEL_PATH=/data/meituan-longcat/LongCat-2.5-W4A4
  export ENCODER_URLS=http://encoder-node0:9101
  export iplist="decode-node0 decode-node1 decode-node2 decode-node3
decode-node4 decode-node5 decode-node6 decode-node7
prefill-node0 prefill-node1 prefill-node2 prefill-node3"
  DRY_RUN=1 bash npu_test/flash26b/run_pro_w4a4kv8_mm_epd.sh start
  bash npu_test/flash26b/run_pro_w4a4kv8_mm_epd.sh code
  bash npu_test/flash26b/run_pro_w4a4kv8_mm_epd.sh start

Start run_encoder_epd.sh on the Encoder node before starting P/D.
Prepare CANN/ATB, Python dependencies and matching NPU kernels on all hosts.
Keep testpath separate from the source checkout. Each P/D host uses 16 NPUs.
EOF
        exit 0
        ;;
    code|start|startp|startd|status|log) ;;
    *) echo "Unknown action: ${action}. Use --help." >&2; exit 2 ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
codepath="$(cd "${SCRIPT_DIR}/../.." && pwd)"
testpath="${testpath:-/data/longcat25}"
logdir="${logdir:-${codepath}/logs/longcat25}"
time="${2:-${time:-longcat25}}"
startshell=pdmode_run.sh
pnode=1
pnode_ep_size=4
dnode=1
dnode_ep_size=8
remote_repo="${testpath%/}/SGLang-FluentLLM"
: "${iplist:?Set iplist to 8 Decode hosts followed by 4 Prefill hosts.}"
[[ "${time}" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo "Invalid log tag." >&2; exit 2; }
[[ "${testpath}" =~ ^/[a-zA-Z0-9_./-]+$ && "${testpath}" != / ]] || {
    echo "testpath must be an absolute work directory without spaces." >&2; exit 2;
}
[[ "${testpath}" != *"/../"* && "${testpath}" != */.. ]] || {
    echo "testpath must not contain '..' path components." >&2; exit 2;
}
read -r -a hosts <<< "${iplist//$'\n'/ }"
[[ "${#hosts[@]}" -eq 12 ]] || { echo "Expected 12 hosts, Decode first." >&2; exit 2; }
for host in "${hosts[@]}"; do
    [[ "${host}" =~ ^[a-zA-Z0-9_][a-zA-Z0-9_.-]*$ ]] || {
        echo "Use SSH hostnames or IPv4 addresses in iplist." >&2; exit 2;
    }
done
case "${action}" in
    start|startp|startd)
        : "${MODEL_PATH:?Set MODEL_PATH to the W4A4 P/D checkpoint directory.}"
        [[ "${MODEL_PATH}" == /* && "${MODEL_PATH}" != *[[:space:]\*\?\[]* ]] || {
            echo "MODEL_PATH must be absolute, without spaces or glob characters." >&2; exit 2;
        }
        : "${ENCODER_URLS:?Set ENCODER_URLS to the independent Encoder HTTP endpoints.}"
        [[ "${ENCODER_URLS}" != *[[:space:]]* && "${ENCODER_URLS}" != ,* && "${ENCODER_URLS}" != *, && "${ENCODER_URLS}" != *,,* ]] || {
            echo "ENCODER_URLS must be comma-separated URLs without spaces or empty entries." >&2; exit 2;
        }
        IFS=, read -r -a encoder_urls <<< "${ENCODER_URLS}"
        for encoder_url in "${encoder_urls[@]}"; do
            [[ "${encoder_url}" == http://?* || "${encoder_url}" == https://?* ]] || {
                echo "Each ENCODER_URLS entry must start with http:// or https://." >&2; exit 2;
            }
        done
        ;;
esac

# 2. Keep the existing W4A4, KV INT8 and multimodal EPD parameters.
prefill_args="--attn-tp-size 8 --npu-lmhead-tp-size 8 --pp-size 8 --data-parallel-size 1 --dense-tp-size 1 --expert-parallel-size 8 \
        --npu-enable-int4-moe-comm --npu-smooth-quant --npu-dsa-kv-int8 --npu-dsa-packed-kv-row-bytes 672 \
        --chunked-prefill-size 16384 --max-prefill-tokens 16384 --mem-fraction-static 0.65 --enable-oe-cpu-offload --npu-o-proj-tp-size 1 \
        --index-k-dtype bf16 --npu-disable-kv-nz --npu-enable-sp-for-indexer --npu-disable-dsa-head-parallel --enable-mla-l1-5-cache \
        --pp-layer-nums 4 5 5 5 5 5 5 4 --max-running-requests 8 --grammar-backend xgrammar \
        --enable-hierarchical-cache --hicache-io-backend direct --hicache-size 60 --hicache-write-policy write_through \
        --enable-cache-report  --low-latency-max-num-tokens-per-gpu 512 --npu-enable-a2-dispatch-combine-opt --npu-enable-w4a4-prefill-two-stream"
decode_args="--attn-tp-size 8 --kvp-size 8 --npu-lmhead-tp-size 16 --data-parallel-size 16 --dense-tp-size 8 --expert-parallel-size 128 \
       --npu-enable-int4-moe-comm --npu-smooth-quant --npu-dsa-kv-int8 --npu-dsa-packed-kv-row-bytes 672 \
       --mem-fraction-static 0.76 --disaggregation-decode-extra-slots 4 --npu-o-proj-tp-size 8 --index-k-dtype bf16 --npu-disable-kv-nz --enable-mla-l1-5-cache \
       --max-running-requests 256 --disable-radix-cache --enable-eplb --ep-num-redundant-experts 128 --expert-distribution-recorder-mode stat --ep-dispatch-algorithm static_with_zero_expert \
       --expert-distribution-recorder-buffer-size 10000 --eplb-rebalance-num-iterations 10000 --grammar-backend xgrammar \
       --npu-compile-bs 1,8,16 --enable-cache-report  \
       --low-latency-max-num-tokens-per-gpu 512  --decode-log-interval 20 \
       --npu-enable-a2-dispatch-combine-opt --enable-speculative-sampling --speculative-sampling-top-k 32 --npu-split-k-router"
multimodal_args="--enable-multimodal --mm-attention-backend ascend_attn --mm-feature-transport default --language-only \
       --encoder-urls ${ENCODER_URLS} --encoder-output-format rvq_indices"
engine_subfix_cmd="--reasoning-parser longcat_xml --tool-call-parser longcat_xml --is-multi-head-eagle --draft-model-path-use-base \
    --speculative-algorithm NEXTN --speculative-num-draft-tokens 4 --speculative-num-steps 3 --speculative-eagle-topk 1 \
    --grammar-mask-timeout-ms 50 ${multimodal_args}"
served_model_name="${served_model_name:-LongCat-2.5}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-1048586}"
VIDEO_MAX_FRAMES="${VIDEO_MAX_FRAMES:-128}"
VIDEO_MAX_PIXELS="${VIDEO_MAX_PIXELS:-150528}"
VIDEO_MIN_PIXELS="${VIDEO_MIN_PIXELS:-100352}"
HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT:-600}"
SGLANG_DISAGGREGATION_WAITING_TIMEOUT="${SGLANG_DISAGGREGATION_WAITING_TIMEOUT:-600}"
SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT="${SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT:-600}"
cluster_ip_list="$(IFS=,; printf '%s' "${hosts[*]}")"

# 3. Dispatch directly to the shared P/D launch helpers.
ssh_command=(ssh -o StrictHostKeyChecking=yes)
if [[ -n "${SSH_PASSWORD_FILE:-}" ]]; then
    [[ -r "${SSH_PASSWORD_FILE}" ]] || { echo "Cannot read SSH_PASSWORD_FILE." >&2; exit 2; }
    ssh_command=(sshpass -f "${SSH_PASSWORD_FILE}" "${ssh_command[@]}")
else
    ssh_command+=(-o BatchMode=yes)
fi
if [[ -n "${SSH_USER:-}" ]]; then
    [[ "${SSH_USER}" =~ ^[a-zA-Z0-9_][a-zA-Z0-9_.-]*$ ]] || { echo "Invalid SSH_USER." >&2; exit 2; }
fi
if [[ -n "${SSH_PORT:-}" ]]; then
    [[ "${SSH_PORT}" =~ ^[0-9]+$ ]] || { echo "Invalid SSH_PORT." >&2; exit 2; }
    (( 10#${SSH_PORT} > 0 && 10#${SSH_PORT} < 65536 )) || { echo "Invalid SSH_PORT." >&2; exit 2; }
    ssh_command+=(-p "${SSH_PORT}")
fi
printf -v transport '%q ' "${ssh_command[@]}"
run_command() {
    if [[ "${DRY_RUN:-0}" == 1 ]]; then
        printf '%q ' "$@"; printf '\n'
    else
        "$@"
    fi
}

for index in "${!hosts[@]}"; do
    [[ "${action}" != startp || "${index}" -ge 8 ]] || continue
    [[ "${action}" != startd || "${index}" -lt 8 ]] || continue
    host="${hosts[index]}"
    target="${SSH_USER:+${SSH_USER}@}${host}"
    case "${action}" in
        code)
            printf -v remote_command 'mkdir -p %q' "${remote_repo}"
            run_command "${ssh_command[@]}" "${target}" "${remote_command}"
            run_command rsync -a -e "${transport}" --exclude=.git --exclude=logs \
                --exclude=__pycache__ --exclude=abc.txt "${codepath}/" "${target}:${remote_repo}/"
            ;;
        start|startp|startd)
            remote_script="set -e"$'\n'
            for name in MODEL_PATH served_model_name prefill_args decode_args engine_subfix_cmd \
                MAX_MODEL_LEN VIDEO_MAX_FRAMES VIDEO_MAX_PIXELS VIDEO_MIN_PIXELS \
                HCCL_CONNECT_TIMEOUT SGLANG_DISAGGREGATION_WAITING_TIMEOUT \
                SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT PYTHON_BIN; do
                printf -v assignment 'export %s=%q\n' "${name}" "${!name}"
                remote_script+="${assignment}"$'\n'
            done
            for name in SERVER_PORT DIST_PORT LB_HOST LB_PORT ASCEND_TOOLKIT_ENV ATB_ENV \
                HCCN_CONF_FILE; do
                if declare -p "${name}" &>/dev/null; then
                    printf -v assignment 'export %s=%q\n' "${name}" "${!name}"
                    remote_script+="${assignment}"$'\n'
                fi
            done
            printf -v assignment 'export cluster_ip_list=%q Node_idx=%q ASCEND_PROCESS_LOG_PATH=%q\n' \
                "${cluster_ip_list}" "${index}" "${testpath}/log_${time}"
            remote_script+="${assignment}"$'\n'
            printf -v assignment 'cd %q\nmkdir -p %q\n' "${remote_repo}" "${testpath}/log_${time}"
            remote_script+="${assignment}"$'\n'
            printf -v assignment 'nohup bash npu_test/flash26b/%q %q %q %q %q > %q 2>&1 < /dev/null &\n' \
                "${startshell}" "${pnode}" "${pnode_ep_size}" "${dnode}" "${dnode_ep_size}" \
                "${testpath}/testlog_${time}.log"
            remote_script+="${assignment}"
            printf -v remote_command 'bash -lc %q' "${remote_script}"
            run_command "${ssh_command[@]}" "${target}" "${remote_command}"
            ;;
        status)
            printf -v remote_command 'tail -n 30 %q' "${testpath}/testlog_${time}.log"
            run_command "${ssh_command[@]}" "${target}" "${remote_command}"
            ;;
        log)
            destination="${logdir}/logs_${time}/${host}"
            run_command mkdir -p "${destination}"
            run_command rsync -a -e "${transport}" "${target}:${testpath}/testlog_${time}.log" "${destination}/"
            ;;
    esac
done
