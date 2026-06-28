#!/bin/bash

# HiCache + PD 分离启动脚本
# 参考: debug.md 和 PD_Disaggregation_HiCache.md

set -e
export MC_GID_INDEX=-1
# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color
export SGLANG_CI_SMALL_KV_SIZE=16000
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# ============================================
# 0. 解析命令行参数
# ============================================
MODEL_NAME="${1:-qwen3}"
PREFETCH_POLICY="${2:-best_effort}"
MEM_LAYOUT="${3:-page_first}"
WRITE_POLICY="${4:-write_through}"
PREFETCH_THRESHOLD="${5:-1}"
IO_BACKEND="${6:-kernel}"
PREFETCH_TIMEOUT_BASE="${7:-1}"

case "$MODEL_NAME" in
    deepseek)
        MODEL_PATH="/mnt/hdfs/zw04mlnn01/checkpoint/llm_platform/model/DeepSeek-V2-Lite"
        log_info "使用模型: DeepSeek-V2-Lite"
        ;;
    qwen3)
        MODEL_PATH="/mnt/hdfs/zw04mlnn01/checkpoint/llm_platform/model/Qwen/Qwen3-4B-Instruct-2507/main"
        log_info "使用模型: Qwen3-4B-Instruct-2507"
        ;;
    *)
        log_error "未知的模型: $MODEL_NAME"
        echo "用法: $0 [model] [prefetch-policy] [mem-layout] [write-policy] [prefetch-threshold] [io-backend] [prefetch-timeout-base]"
        echo ""
        echo "参数说明:"
        echo "  model                 - deepseek 或 qwen3 (默认: qwen3)"
        echo "  prefetch-policy       - timeout 或其他策略 (默认: timeout)"
        echo "  mem-layout            - page_first 或其他布局 (默认: page_first)"
        echo "  write-policy          - write_through 或 write_back (默认: write_through)"
        echo "  prefetch-threshold    - HiCache prefetch threshold (可选)"
        echo "  io-backend            - kernel 或 direct (默认: kernel)"
        echo "  prefetch-timeout-base - HiCache prefetch timeout base (可选)"
        echo ""
        echo "示例:"
        echo "  $0 deepseek timeout page_first write_through"
        echo "  $0 qwen3 timeout page_first write_back 0.8 kernel 1000"
        echo "  $0 qwen3 timeout page_first write_back 0.8 direct 1000"
        exit 1
        ;;
esac

log_info "配置信息:"
log_info "  模型: $MODEL_NAME ($MODEL_PATH)"
log_info "  Prefetch Policy: $PREFETCH_POLICY"
log_info "  Memory Layout: $MEM_LAYOUT"
log_info "  Write Policy: $WRITE_POLICY"
[ -n "$PREFETCH_THRESHOLD" ] && log_info "  Prefetch Threshold: $PREFETCH_THRESHOLD"
[ -n "$IO_BACKEND" ] && log_info "  IO Backend: $IO_BACKEND"
[ -n "$PREFETCH_TIMEOUT_BASE" ] && log_info "  Prefetch Timeout Base: $PREFETCH_TIMEOUT_BASE"

# 清理旧进程
log_info "清理旧进程..."
pkill -f "sglang" || true
pkill -f "mini_lb" || true
pkill -f "mooncake_master" || true

sleep 20

# ============================================
# 健康检查函数
# ============================================
# 等待服务健康检查通过
# 参数: $1=服务名称, $2=URL, $3=最大等待秒数(默认120), $4=检查间隔秒数(默认5)
wait_for_service_health() {
    local service_name="$1"
    local url="$2"
    local max_wait="${3:-120}"
    local check_interval="${4:-5}"
    local elapsed=0

    log_info "等待 $service_name 健康检查通过 (URL: $url)..."

    while [ $elapsed -lt $max_wait ]; do
        if curl -s -f "$url" > /dev/null 2>&1; then
            log_info "✓ $service_name 健康检查通过"
            return 0
        fi
        sleep $check_interval
        elapsed=$((elapsed + check_interval))
        echo -n "."
    done

    echo ""
    log_error "✗ $service_name 健康检查超时 (等待 ${max_wait}s)"
    return 1
}

# ============================================
# 1. 启动 Mooncake Master
# ============================================
log_info "启动 Mooncake Master..."
nohup mooncake_master \
  -port 50051 \
  -max_threads 64 \
  -metrics_port 9004 \
  --enable_http_metadata_server=true \
  --http_metadata_server_host=0.0.0.0 \
  --http_metadata_server_port=8080 \
  --eviction_high_watermark_ratio=0.95 \
  > logs/mooncake_master.log 2>&1 &

MOONCAKE_PID=$!
log_info "Mooncake Master PID: $MOONCAKE_PID"

sleep 2
# ============================================
# 2. 启动 Load Balancer
# ============================================
log_info "启动 Load Balancer..."
nohup python3 -m sglang.srt.disaggregation.mini_lb \
  --host 0.0.0.0 \
  --port 8192 \
  > logs/lb.log 2>&1 &

LB_PID=$!
log_info "Load Balancer PID: $LB_PID"

sleep 2

# ============================================
# 构建 HiCache extra_config JSON
# ============================================
EXTRA_CONFIG_ARGS=()
if [ -n "$PREFETCH_THRESHOLD" ] || [ -n "$PREFETCH_TIMEOUT_BASE" ]; then
    CONFIG="{"
    if [ -n "$PREFETCH_THRESHOLD" ]; then
        CONFIG="${CONFIG}\"prefetch_threshold\": ${PREFETCH_THRESHOLD}"
    fi
    if [ -n "$PREFETCH_THRESHOLD" ] && [ -n "$PREFETCH_TIMEOUT_BASE" ]; then
        CONFIG="${CONFIG}, "
    fi
    if [ -n "$PREFETCH_TIMEOUT_BASE" ]; then
        CONFIG="${CONFIG}\"prefetch_timeout_base\": ${PREFETCH_TIMEOUT_BASE}"
    fi
    CONFIG="${CONFIG}}"
    EXTRA_CONFIG_ARGS=("--hicache-storage-backend-extra-config" "${CONFIG}")
fi

# ============================================
# 3. 启动 Prefill Worker
# ============================================
log_info "启动 Prefill Worker..."

export MOONCAKE_MASTER="127.0.0.1:50051"
export MOONCAKE_TE_META_DATA_SERVER="http://127.0.0.1:8080/metadata"
export MOONCAKE_PROTOCOL="rdma"
export MOONCAKE_DEVICE=""
export MOONCAKE_GLOBAL_SEGMENT_SIZE="64gb"

nohup python3 -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --max-running-requests 32 \
  --disaggregation-mode prefill \
  --base-gpu-id 0 \
  --port 8292 \
  --disable-cuda-graph \
  --enable-flashinfer-mla \
  --trust-remote-code \
  --moe-parallel-strategy ep \
  --dense-parallel-strategy rep \
  --nprocs-per-node 1 \
  --attn-tp-size 1 \
  --dp-size 1 \
  --random-seed 1234 \
  --context-length 8000 \
  --host 0.0.0.0 \
  --log-level debug \
  --disaggregation-transfer-backend mooncake_async \
  --metrics-reporters prometheus \
  --enable-metrics \
  --enable-hierarchical-cache \
  --hicache-storage-backend mooncake \
  --hicache-storage-prefetch-policy "$PREFETCH_POLICY" \
  --hicache-mem-layout "$MEM_LAYOUT" \
  --hicache-io-backend "$IO_BACKEND" \
  --hicache-write-policy "$WRITE_POLICY" \
  --pdlb-url http://0.0.0.0:8192 \
  "${EXTRA_CONFIG_ARGS[@]}" \
  > logs/pr.log 2>&1 &

PREFILL_PID=$!
log_info "Prefill Worker PID: $PREFILL_PID"

sleep 3

# ============================================
# 4. 启动 Decode Worker
# ============================================
log_info "启动 Decode Worker..."

nohup python3 -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --max-running-requests 32 \
  --disaggregation-mode decode \
  --base-gpu-id 1 \
  --port 8392 \
  --disable-cuda-graph \
  --enable-flashinfer-mla \
  --trust-remote-code \
  --moe-parallel-strategy ep \
  --dense-parallel-strategy rep \
  --nprocs-per-node 1 \
  --attn-tp-size 1 \
  --dp-size 1 \
  --random-seed 1234 \
  --host 0.0.0.0 \
  --log-level debug \
  --enable-hierarchical-cache \
  --context-length 8000 \
  --hicache-storage-backend mooncake \
  --metrics-reporters prometheus \
  --enable-metrics \
  --hicache-storage-prefetch-policy "$PREFETCH_POLICY" \
  --hicache-mem-layout "$MEM_LAYOUT" \
  --hicache-io-backend "$IO_BACKEND" \
  --hicache-write-policy write_through \
  --disaggregation-transfer-backend mooncake_async \
  --pdlb-url http://0.0.0.0:8192 \
  "${EXTRA_CONFIG_ARGS[@]}" \
  > logs/de.log 2>&1 &

DECODE_PID=$!
log_info "Decode Worker PID: $DECODE_PID"

# ============================================
# 5. 验证服务状态
# ============================================
log_info "等待服务启动..."
sleep 150
# 等待 Load Balancer 健康检查通过
wait_for_service_health "Load Balancer" "http://127.0.0.1:8192/health" 120 5

# 等待 Decode Worker 健康检查通过
wait_for_service_health "Decode Worker" "http://127.0.0.1:8392/health" 120 20
# 等待 Prefill Worker 健康检查通过
wait_for_service_health "Prefill Worker" "http://127.0.0.1:8292/health" 120 20

log_info "验证服务状态..."

# 检查 Mooncake Master
if ps -p $MOONCAKE_PID > /dev/null; then
    log_info "✓ Mooncake Master 运行中"
else
    log_error "✗ Mooncake Master 启动失败"
    tail -20 logs/mooncake_master.log
fi

# 检查 Load Balancer
if ps -p $LB_PID > /dev/null; then
    log_info "✓ Load Balancer 运行中"
else
    log_error "✗ Load Balancer 启动失败"
    tail -20 logs/lb.log
fi

# 检查 Prefill Worker
if ps -p $PREFILL_PID > /dev/null; then
    log_info "✓ Prefill Worker 运行中"
else
    log_error "✗ Prefill Worker 启动失败"
    tail -20 logs/pr.log
fi

# 检查 Decode Worker
if ps -p $DECODE_PID > /dev/null; then
    log_info "✓ Decode Worker 运行中"
else
    log_error "✗ Decode Worker 启动失败"
    tail -20 logs/de.log
fi

# ============================================
# 6. 显示服务信息
# ============================================
log_info "========================================="
log_info "HiCache + PD 分离服务已启动"
log_info "========================================="
log_info "Mooncake Master: 127.0.0.1:50051"
log_info "Metadata Server: http://127.0.0.1:8080/metadata"
log_info "Load Balancer: http://0.0.0.0:8192"
log_info "Prefill Worker: http://0.0.0.0:8292"
log_info "Decode Worker: http://0.0.0.0:8392"
log_info "========================================="
log_info "日志位置: logs/"
log_info "========================================="

# 保存 PID 到文件
cat > logs/pids.txt << EOF
MOONCAKE_PID=$MOONCAKE_PID
LB_PID=$LB_PID
PREFILL_PID=$PREFILL_PID
DECODE_PID=$DECODE_PID
EOF

log_info "PID 已保存到 pids.txt"
log_info "启动完成！"