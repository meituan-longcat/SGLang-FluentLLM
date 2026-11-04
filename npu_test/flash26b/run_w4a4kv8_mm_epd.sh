#!/usr/bin/env bash
# LongCat 2.5 W4A4 / KV INT8 multimodal EPD profile.
set -eo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Configure the deployment hosts, weight path and Encoder endpoint.
export testpath=/data/longcat25
export time=longcat25
export iplist="
decode-node0
decode-node1
decode-node2
decode-node3
decode-node4
decode-node5
decode-node6
decode-node7
prefill-node0
prefill-node1
prefill-node2
prefill-node3
"
export MODEL_PATH=/data/meituan-longcat/LongCat-2.5-W4A4
export ENCODER_URLS=http://encoder-node0:9101
export pnode=1
export pnode_ep_size=4
export dnode=1
export dnode_ep_size=8
export served_model_name="${served_model_name:-LongCat-2.5}"
export PYTHON_BIN="${PYTHON_BIN:-python3}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-1048586}"
export VIDEO_MAX_FRAMES="${VIDEO_MAX_FRAMES:-128}"
export VIDEO_MAX_PIXELS="${VIDEO_MAX_PIXELS:-150528}"
export VIDEO_MIN_PIXELS="${VIDEO_MIN_PIXELS:-100352}"
export HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT:-600}"
export SGLANG_DISAGGREGATION_WAITING_TIMEOUT="${SGLANG_DISAGGREGATION_WAITING_TIMEOUT:-600}"
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT="${SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT:-600}"

#prefill
export prefill_args="--attn-tp-size 8 --npu-lmhead-tp-size 8 --pp-size 8 --data-parallel-size 1 --dense-tp-size 1 --expert-parallel-size 8 \
        --npu-enable-int4-moe-comm --npu-smooth-quant --npu-dsa-kv-int8 --npu-dsa-packed-kv-row-bytes 672 \
        --chunked-prefill-size 16384 --max-prefill-tokens 16384 --mem-fraction-static 0.65 --enable-oe-cpu-offload --npu-o-proj-tp-size 1 \
        --index-k-dtype bf16 --npu-disable-kv-nz --npu-enable-sp-for-indexer --npu-disable-dsa-head-parallel --enable-mla-l1-5-cache \
        --pp-layer-nums 4 5 5 5 5 5 5 4 --max-running-requests 8 --grammar-backend xgrammar \
        --enable-hierarchical-cache --hicache-io-backend direct --hicache-size 60 --hicache-write-policy write_through \
        --enable-cache-report  --low-latency-max-num-tokens-per-gpu 512 --npu-enable-a2-dispatch-combine-opt --npu-enable-w4a4-prefill-two-stream"

#decode
export decode_args="--attn-tp-size 8 --kvp-size 8 --npu-lmhead-tp-size 16 --data-parallel-size 16 --dense-tp-size 8 --expert-parallel-size 128 \
       --npu-enable-int4-moe-comm --npu-smooth-quant --npu-dsa-kv-int8 --npu-dsa-packed-kv-row-bytes 672 \
       --mem-fraction-static 0.76 --disaggregation-decode-extra-slots 4 --npu-o-proj-tp-size 8 --index-k-dtype bf16 --npu-disable-kv-nz --enable-mla-l1-5-cache \
       --max-running-requests 256 --disable-radix-cache --enable-eplb --ep-num-redundant-experts 128 --expert-distribution-recorder-mode stat --ep-dispatch-algorithm static_with_zero_expert \
       --expert-distribution-recorder-buffer-size 10000 --eplb-rebalance-num-iterations 10000 --grammar-backend xgrammar \
       --npu-compile-bs 1,8,16 --enable-cache-report  \
       --low-latency-max-num-tokens-per-gpu 512  --decode-log-interval 20 \
       --npu-enable-a2-dispatch-combine-opt --enable-speculative-sampling --speculative-sampling-top-k 32 --npu-split-k-router"

# 多模态 EPD：Encoder 单独部署，参数同时用于 Prefill 和 Decode。
multimodal_args="--enable-multimodal --mm-attention-backend ascend_attn --mm-feature-transport default --language-only \
       --encoder-urls ${ENCODER_URLS} --encoder-output-format rvq_indices"



export engine_subfix_cmd="--reasoning-parser longcat_xml --tool-call-parser longcat_xml --is-multi-head-eagle --draft-model-path-use-base \
    --speculative-algorithm NEXTN --speculative-num-draft-tokens 4 --speculative-num-steps 3 --speculative-eagle-topk 1 \
    --grammar-mask-timeout-ms 50 ${multimodal_args}"
bash "${SCRIPT_DIR}/run_batch.sh" "$@"
