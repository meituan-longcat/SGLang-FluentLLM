#!/bin/bash
# EPD Encoder Server 启动脚本（远端 pssh.sh 适配版）
# 通过 sglang.srt.entrypoints.encoder_server 启动纯 ViT 服务，
# 只加载 vision_tower（encoder + bridge + quantizer），不加载 LM。
# pssh.sh 已在远端 source 了 venv、设置了 PYTHONPATH，并 cd 到 REMOTE_WORKDIR。
#
# Encoder TP 模式：--attn-tp-size 2 --nprocs-per-node 2
#   main 进程 spawn 2 个 worker 子进程（各占 1 卡 GPU 0,1），
#   各自初始化 NCCL（通过 --dist-init-addr 建联），
#   rank 0 起 HTTP 服务 + step loop；rank 1 只跑 step loop。
#   ViT forward 的 RowParallelLinear 自动 all_reduce 合并各 rank 输出。

: "${EPS_HOME:?Set EPS_HOME to the EPS checkout directory before running this script.}"
export EPS_HOME
export PYTHONPATH=${CODE_REMOTE_DIR}/python:${CODE_REMOTE_DIR}:$PYTHONPATH
export PYTHONPATH=${EPS_HOME}/python/:$PYTHONPATH

export SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK=true

# 模型路径（与 LM 侧一致）,  mla
: "${MODEL_PATH:?Set MODEL_PATH to the model checkpoint directory before running this script.}"

# GPU image preprocess: resize/normalize/patchify on device, skipping the
# numpy→CPU→cuda round-trip of the default image_transform path.
# Debug knobs (optional): GPU_TRANSFORM_PROF=1 elevates per-stage timing to
# INFO; GPU_TRANSFORM_PROF_SYNC=1 serializes GPU phases for accurate kernel
# timings (debug only).
export ENABLE_GPU_PREPROCESS=1

echo "=== Starting EPD Encoder Server (TP=2, GPU preprocess) ==="
echo "MODEL_PATH=$MODEL_PATH"
echo "CODE_REMOTE_DIR=$CODE_REMOTE_DIR"
python3 -c "import sglang; print('sglang:', sglang.__file__)"

# Encoder: 2 卡 (GPU 0,1), TP=2
# --disaggregation-mode encode 会自动设置 encoder_only=True
# --dist-init-addr 127.0.0.1:3100: 单机内 2 卡 TP 建联（端口与 LM 侧 3000 及其衍生端口错开）
# --attn-tp-size 2: ViT 张量并行，RowParallelLinear 输出 all_reduce 合并
# --base-gpu-id 0 --gpu-id-step 1: rank 0 → GPU 0, rank 1 → GPU 1
CUDA_VISIBLE_DEVICES=0,1 python3 -m sglang.srt.entrypoints.encoder_server \
    --model-path "$MODEL_PATH" \
    --trust-remote-code \
    --disaggregation-mode encode \
    --nnodes 1 \
    --node-rank 0 \
    --nprocs-per-node 2 \
    --attn-tp-size 2 \
    --dp-size 1 \
    --port 31001 \
    --dist-init-addr 127.0.0.1:3100 \
    --base-gpu-id 0 \
    --gpu-id-step 1 \
    --encoder-output-format rvq_indices \
    --encoder-http-port 9101 \
    --encoder-grpc-port -1 \
    --encoder-max-tokens-per-batch 8192 \
    --encoder-max-items-per-batch 64 \
    --encoder-cache-size-mb 1024 \
    --encoder-enable-tiered-cache \
    --encoder-cache-host-size-mb 8192 \
    --encoder-register-timeout 120 \
    --encoder-encode-timeout 30 \
    --log-level info \
    --show-time-cost \