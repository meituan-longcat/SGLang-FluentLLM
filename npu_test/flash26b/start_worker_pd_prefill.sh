#!/bin/bash
export MC_GID_INDEX=7
export MC_IB_DEVICES=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7

echo ${SERVING_DOCKER_LOGS_DIR}
exec >> /${SERVING_DOCKER_LOGS_DIR}/worker.log 2>&1

declare -px > "${SERVING_DOCKER_LOGS_DIR}/mlp_env.sh"

ps -axu | grep "sglang::" | awk '{ print $2 }' | xargs kill -9
pkill -9 sglang
ps -axu | grep "/opt/tritonserver/bin/tritonserver_real" | awk '{ print $2 }' | xargs kill -9
sleep 3

source /usr/local/Ascend/ascend-toolkit/set_env.sh

export MC_LOG_LEVEL=ERROR

export PYTHONPATH=/home/fluentllm/python:$PYTHONPATH
export LD_LIBRARY_PATH=/usr/local/lib:/usr/local/Ascend/driver/lib64/driver/:/usr/local/conda/lib:/opt/sglang/fluentllm/router/lib64:$LD_LIBRARY_PATH
export NPROCS_PER_NODE=16
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1

if ! [ -e "/etc/hccn.conf" ]; then
    custom_hccn_file=$HOME/hccn.conf
    bash $(dirname "$0")/gen_hccn_conf.sh ${custom_hccn_file}
    export HCCN_CONF_FILE=${custom_hccn_file}
fi

export OMP_NUM_THREADS=8
export HCCL_DETERMINISTIC=False
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True,base_addr_aligned_kb:16
num_910b=$(npu-smi info | grep 910B | wc -l)
if [[ $num_910b -gt 0 ]]; then
  echo 'CPU_AFFINITY_CONF 910B'
  export CPU_AFFINITY_CONF=npu0:2-6,npu1:7-11,npu2:12-16,npu3:17-21,npu4:50-54,npu5:55-59,npu6:60-64,npu7:65-69,npu8:22-27,npu9:28-32,npu10:33-37,npu11:38-42,npu12:70-74,npu13:75-79,npu14:80-84,npu15:85-89
else
  echo 'CPU_AFFINITY_CONF 910C'
  export CPU_AFFINITY_CONF=npu0:8-39,npu1:40-71,npu2:72-103,npu3:104-135,npu4:160-191,npu5:192-223,npu6:224-255,npu7:256-287,npu8:320-351,npu9:352-383,npu10:384-415,npu11:416-447,npu12:480-511,npu13:512-543,npu14:544-575,npu15:576-607
  if [ "$ARGS_NODE_NUM" -eq 1 ]; then
    export TORCH_HCCL_ZERO_COPY=1
 fi
fi
export TASK_QUEUE_ENABLE=2

export ENABLE_TORCH_COMPILE_CACHE=0
export HCCL_INTRA_ROCE_ENABLE=1
export HCCL_INTRA_PCIE_ENABLE=0
export HCCL_RDMA_TIMEOUT=20
export ASCEND_TRANSFER_TIMEOUT=31000
export ACL_STREAM_TIMEOUT=30000
export HCCL_RDMA_TC=100
export HCCL_RDMA_SL=3
export CPU_AFFINITY_CONF=1

export HCCL_IF_BASE_PORT=8282

export NPU_ENABLE_WEIGHT_NZ=0
export NPU_ENABLE_MC2=0
export HCCL_OP_RETRY_ENABLE="L0:0,L1:0,L2:0"

# tokenizer rayon线程池报错，限制最大线程数
export RAYON_NUM_THREADS=8

export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export ASCEND_PROCESS_LOG_PATH=$SERVING_DOCKER_LOGS_DIR
export NPU_ENABLE_ALL2ALL_COMM=0

# 优化prefill的3个ENV，要求attn_tp == dense_tp。目前在PD分离下需要开启，PD混合下得关闭
export NPU_PREFILL_AG_RS=1
export NPU_ENABLE_MOE_GATING_TOP_K=1
export NPU_LMHEAD_TP_SIZE=16

export GRPC_PORT="$ARGS_GRPC_PORT"
# 幻觉兜底
export TOOL_CHOICE_BYPASS_CHECK=true
# 不注册octo，设置个无效的AppKey RESET_SERVING_APPKEY
if [ -v RESET_SERVING_APPKEY ]; then
  export SERVING_APPKEY=${RESET_SERVING_APPKEY}
fi

if [ -v ENABLE_MTP ]; then
  echo "ENABLE_MTP"
fi

# 如果设置了OVERRIDE_MODEL_REPOSITORY，则覆盖ARGS_MODEL_REPOSITORY
if [ ! -z "${OVERRIDE_MODEL_REPOSITORY}" ]; then
  echo "Using OVERRIDE_MODEL_REPOSITORY: ${OVERRIDE_MODEL_REPOSITORY}"
  ARGS_MODEL_REPOSITORY=${OVERRIDE_MODEL_REPOSITORY}
fi

echo "start worker with args"
echo "ARGS_MODEL_REPOSITORY: ${ARGS_MODEL_REPOSITORY}"
echo "ARGS_MODEL_NAME: ${ARGS_MODEL_NAME}"
echo "ARGS_ALLOW_GRPC: ${ARGS_ALLOW_GRPC}"
echo "ARGS_ALLOW_WHALE_RPC: ${ARGS_ALLOW_WHALE_RPC}"
echo "ARGS_PORT: ${ARGS_PORT}"
echo "ARGS_MPORT: ${ARGS_MPORT}"
echo "ARGS_LOG_INFO: ${ARGS_LOG_INFO}"
echo "ARGS_WHALE_RPC_USE_ASYNC: ${ARGS_WHALE_RPC_USE_ASYNC}"
echo "ARGS_GRPC_PORT: ${ARGS_GRPC_PORT}"
echo "ARGS_ALLOW_METRICS: ${ARGS_ALLOW_METRICS}"
echo "ARGS_ALLOW_HTTP: ${ARGS_ALLOW_HTTP}"
echo "ARGS_THRIFT_TIMEOUT_MS: ${ARGS_THRIFT_TIMEOUT_MS}"
echo "SERVING_DOCKER_LOGS_DIR: ${SERVING_DOCKER_LOGS_DIR}"
echo "SERVING_APPKEY: ${SERVING_APPKEY}"

# 如果有PD_MASTER_HOST则打印
if [ ! -z "${PD_MASTER_HOST}" ]; then
  echo "PD_MASTER_HOST: ${PD_MASTER_HOST}"
  echo "NODE_RANK: ${ARGS_NODE_RANK}"
  echo "NODE_NUM: ${ARGS_NODE_NUM}"
fi

if [ ! -z "${PD_ROLE}" ]; then
  echo "PD_ROLE: ${PD_ROLE}"
fi

/opt/tritonserver/bin/tritonserver_real \
--model-repository=${ARGS_MODEL_REPOSITORY} \
--model-control-mode explicit \
--load-model=${ARGS_MODEL_NAME} \
--allow-grpc ${ARGS_ALLOW_GRPC} \
--allow-whale-rpc ${ARGS_ALLOW_WHALE_RPC} \
--whale-rpc-port ${ARGS_PORT} \
--mport ${ARGS_MPORT} \
--log-info ${ARGS_LOG_INFO} \
--whale-rpc-use-async ${ARGS_WHALE_RPC_USE_ASYNC} \
--grpc-port ${ARGS_GRPC_PORT} \
--allow-metrics ${ARGS_ALLOW_METRICS} \
--allow-http ${ARGS_ALLOW_HTTP} \
--log-verbos ${ARGS_LOG_VERBOS} \
--thrift-timeout-ms ${ARGS_THRIFT_TIMEOUT_MS} \
--log-dir ${SERVING_DOCKER_LOGS_DIR} \
--enable-model-monitor true \
--server-appkey ${SERVING_APPKEY} \
--unload-oldmodel-first true &

SGLANG_PID=$!
sleep 45
python3 $(dirname "$0")/bind_core.py ${ARGS_NODE_RANK} >> ${SERVING_DOCKER_LOGS_DIR}/bind_core.log 2>&1
wait $SGLANG_PID
