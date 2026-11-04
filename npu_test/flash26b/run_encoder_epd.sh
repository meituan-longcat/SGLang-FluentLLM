#!/usr/bin/env bash
# Standalone NPU Encoder HTTP service.
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
: "${MODEL_PATH:?Set MODEL_PATH to the Encoder weight directory.}"
if [[ "${DRY_RUN:-0}" != 1 ]]; then
    [[ -d "${MODEL_PATH}" ]] || { echo "MODEL_PATH is not a directory." >&2; exit 1; }
    [[ -f "${REPO_ROOT}/python/sglang/srt/entrypoints/encoder_server.py" ]] || {
        echo "This source version does not include the EPD Encoder runtime." >&2; exit 1;
    }
fi
export ASCEND_RT_VISIBLE_DEVICES="${ENCODER_NPU_DEVICES:-0,1}"
export NPROCS_PER_NODE=2
export ENABLE_DEVICE_PREPROCESS="${ENABLE_DEVICE_PREPROCESS:-1}"
export HCCL_IF_BASE_PORT="${HCCL_IF_BASE_PORT:-8283}"
export HCCL_OP_EXPANSION_MODE="${HCCL_OP_EXPANSION_MODE:-AIV}"
command=("${PYTHON_BIN}" -m sglang.srt.entrypoints.encoder_server \
  --device npu --mm-attention-backend ascend_attn --model-path "${MODEL_PATH}" \
  --served-model-name "${served_model_name:-LongCat-2.5}" --trust-remote-code \
  --disaggregation-mode encode --encoder-output-format rvq_indices \
  --encoder-http-port "${ENCODER_HTTP_PORT:-9101}" --encoder-grpc-port -1 \
  --nnodes 1 --node-rank 0 --nprocs-per-node 2 --attn-tp-size 2 \
  --dist-init-addr "${ENCODER_DIST_INIT_ADDR:-127.0.0.1:3100}" \
  --data-parallel-size 1 --dense-tp-size 1 --expert-parallel-size 2 \
  --moe-parallel-strategy ep --dense-parallel-strategy combine --npu-lmhead-tp-size 2 \
  --mem-fraction-static 0 --disable-custom-all-reduce \
  --encoder-max-tokens-per-batch 8192 --encoder-max-items-per-batch 1 \
  --encoder-cache-size-mb 1024 --encoder-enable-tiered-cache \
  --encoder-cache-host-size-mb 8192 --encoder-register-timeout 300 \
  --encoder-encode-timeout 30 --log-level info --show-time-cost \
  --enable-metrics --metrics-reporters prometheus)
if [[ "${DRY_RUN:-0}" == 1 ]]; then
    printf '%q ' "${command[@]}"
    printf '\n'
    exit 0
fi
exec "${command[@]}"
