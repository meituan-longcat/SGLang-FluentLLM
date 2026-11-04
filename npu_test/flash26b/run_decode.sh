#!/usr/bin/env bash
set -eo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export PYTHON_BIN="${PYTHON_BIN:-python3}"
if [[ "${DRY_RUN:-0}" != 1 ]]; then
    source "${ASCEND_TOOLKIT_ENV:-/usr/local/Ascend/ascend-toolkit/set_env.sh}"
    source "${ATB_ENV:-/usr/local/Ascend/nnal/atb/set_env.sh}"
    if [[ ! -f /etc/hccn.conf && ! -f "${HCCN_CONF_FILE:-}" ]]; then
        echo "Set HCCN_CONF_FILE when /etc/hccn.conf is unavailable." >&2
        exit 1
    fi
fi
export PYTHONPATH="${REPO_ROOT}/python:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="/usr/local/lib:/usr/local/Ascend/driver/lib64/driver:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True,base_addr_aligned_kb:16}"
export TASK_QUEUE_ENABLE="${TASK_QUEUE_ENABLE:-2}"
export HCCL_INTRA_ROCE_ENABLE="${HCCL_INTRA_ROCE_ENABLE:-1}"
export HCCL_INTRA_PCIE_ENABLE="${HCCL_INTRA_PCIE_ENABLE:-0}"
export HCCL_RDMA_TC="${HCCL_RDMA_TC:-100}"
export HCCL_RDMA_SL="${HCCL_RDMA_SL:-3}"
export HCCL_RDMA_TIMEOUT="${HCCL_RDMA_TIMEOUT:-20}"
export HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT:-600}"
export HCCL_OP_RETRY_ENABLE="${HCCL_OP_RETRY_ENABLE:-L0:0,L1:0,L2:0}"
export NPU_ENABLE_WEIGHT_NZ="${NPU_ENABLE_WEIGHT_NZ:-0}"
export NPU_ENABLE_MC2="${NPU_ENABLE_MC2:-0}"
export SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK=1
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
export VIDEO_MAX_FRAMES="${VIDEO_MAX_FRAMES:-128}"
export VIDEO_MAX_PIXELS="${VIDEO_MAX_PIXELS:-150528}"
export VIDEO_MIN_PIXELS="${VIDEO_MIN_PIXELS:-100352}"
cd "${REPO_ROOT}"
: "${MODEL_PATH:?Set MODEL_PATH.}"
: "${MASTERIP:?Set MASTERIP.}"
if [[ "${DRY_RUN:-0}" != 1 && ! -d "${MODEL_PATH}" ]]; then
    echo "MODEL_PATH is not a directory." >&2; exit 1
fi
NODE_RANK=${NODE_RANK:-$1}
NODE_NUM=${NODE_NUM:-$2}
export NPROCS_PER_NODE=16
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}"
export SGLANG_DISAGGREGATION_WAITING_TIMEOUT="${SGLANG_DISAGGREGATION_WAITING_TIMEOUT:-600}"
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT="${SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT:-600}"
export ENABLE_TORCH_COMPILE_CACHE=0
export HCCL_DETERMINISTIC=False
export HCCL_IF_BASE_PORT="${HCCL_IF_BASE_PORT:-8282}"
export NPU_ENABLE_GRAPH=1
export NPU_SCMOE_MODE=2
export NPU_O_PROJ_TP_SIZE=8
export NPU_LMHEAD_TP_SIZE=16
export NPU_ENABLE_SFAD_RANK_MAJOR_QUERY=1
export NPU_DISTRIBUTE_ZERO=1
export NPU_ENABLE_GET_OUT_CACHE=1
export NPU_ENABLE_MOE_GATING_TOP_K=1
export NPU_MOE_CORE_NUM="${NPU_MOE_CORE_NUM:-12}"
export NPU_ENABLE_MLA_HW_PROLOG=1
export TOOL_CHOICE_BYPASS_CHECK=true
export RAYON_NUM_THREADS="${RAYON_NUM_THREADS:-8}"
read -r -a shared_args <<< "${engine_subfix_cmd//$'\n'/ }"
read -r -a role_args <<< "${decode_args//$'\n'/ }"
command=("${PYTHON_BIN}" -m sglang.launch_server "${shared_args[@]}" "${role_args[@]}" \
  --device npu --attention-backend npu_mla --model-path "${MODEL_PATH}" \
  --served-model-name "${served_model_name:-LongCat-2.5}" --trust-remote-code \
  --host 0.0.0.0 --port "${SERVER_PORT:-6080}" \
  --nnodes "${NODE_NUM}" --node-rank "${NODE_RANK}" \
  --dist-init-addr "${MASTERIP}:${DIST_PORT:-6090}" --nprocs-per-node 16 \
  --dense-parallel-strategy combine --moe-parallel-strategy ep \
  --disaggregation-mode decode --disaggregation-transfer-backend mooncake_async \
  --context-length "${MAX_MODEL_LEN:-1048586}" \
  --chunked-prefill-size 4096 --max-prefill-tokens 4096 \
  --enable-metrics --metrics-reporters prometheus --schedule-conservativeness 5 \
  --show-time-cost --disable-cuda-graph --page-size 128 \
  --npu-hccl-buffsize-a2a 800 --num-continuous-decode-steps 4 \
  --disable-custom-all-reduce --npu-enable-weight-nz --tokenizer-executor-num-processes 24)
if [[ "${DRY_RUN:-0}" == 1 ]]; then
    printf '%q ' "${command[@]}"
    printf '\n'
    exit 0
fi
exec "${command[@]}"
