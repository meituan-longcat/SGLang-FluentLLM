source /usr/local/Ascend/ascend-toolkit/set_env.sh
WORKDIR=`pwd`

echo $engine_subfix_cmd

if [ -z "${MODEL_PATH}" ]; then
    MODEL_PATH=/workdir/npu_dev_test/models/flash_think_0901_iter7000_mtp_quant_allblock_int8
    echo "MODEL_PATH 不存在，设置为默认值: $MODEL_PATH"
else
    echo "MODEL_PATH 已存在，值为: $MODEL_PATH"
fi

ulimit -u unlimited


if [ ! -d "${MODEL_PATH}" ]; then
    echo "错误: 路径 ${MODEL_PATH} 不存在"
    exit 1
fi

export PYTHONPATH=$WORKDIR/python:$PYTHONPATH
export LD_LIBRARY_PATH=/usr/local/lib:/usr/local/Ascend/driver/lib64/driver/:$LD_LIBRARY_PATH
export SGLANG_DISAGGREGATION_WAITING_TIMEOUT=3000

export NPROCS_PER_NODE=16
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
# 幻觉兜底
export TOOL_CHOICE_BYPASS_CHECK=true
if ! [ -e "/etc/hccn.conf" ]; then
    custom_hccn_file=$(pwd)/hccn.conf
    bash $(dirname "$0")/gen_hccn_conf.sh ${custom_hccn_file}
    export HCCN_CONF_FILE=${custom_hccn_file}
fi

NODE_RANK=${NODE_RANK:-$1}
NODE_NUM=${NODE_NUM:-$2}
model_args=""
# 根据MAX_MODEL_LEN是否存在添加context-length
if [ -n "$MAX_MODEL_LEN" ]; then
    model_args="$model_args --context-length $MAX_MODEL_LEN"
fi
#rm -rf /log/run/plog/*
#rm -rf /log/debug/plog/*
rm -rf /root/ascend/log/debug/plog/*
rm -rf /root/ascend/log/run/plog/*
export OMP_NUM_THREADS=8
# export ASCEND_LAUNCH_BLOCKING=1
# export ASCEND_SLOG_PRINT_TO_STDOUT=1
# export ASCEND_GLOBAL_LOG_LEVEL=1
export HCCL_DETERMINISTIC=False
# export HCCL_DETERMINISTIC=True
# export LOAD_NUMBER_LAYERS=2

export TASK_QUEUE_ENABLE=2
export ENABLE_TORCH_COMPILE_CACHE=0
export HCCL_RDMA_TIMEOUT=20
export HCCL_INTRA_ROCE_ENABLE=1
export HCCL_INTRA_PCIE_ENABLE=0
export HCCL_RDMA_TC=100
export HCCL_RDMA_SL=3
export HCCL_OP_RETRY_ENABLE="L0:0,L1:0,L2:0"
num_910b=$(npu-smi info | grep 910B | wc -l)
if [[ $num_910b -gt 0 ]]; then
  echo 'CPU_AFFINITY_CONF 910B'
  export CPU_AFFINITY_CONF=npu0:2-6,npu1:7-11,npu2:12-16,npu3:17-21,npu4:50-54,npu5:55-59,npu6:60-64,npu7:65-69,npu8:22-27,npu9:28-32,npu10:33-37,npu11:38-42,npu12:70-74,npu13:75-79,npu14:80-84,npu15:85-89
  extra_args='--npu-enable-a2-dispatch-combine-opt'
else
  echo 'CPU_AFFINITY_CONF 910C'
  export CPU_AFFINITY_CONF=npu0:8-39,npu1:40-71,npu2:72-103,npu3:104-135,npu4:160-191,npu5:192-223,npu6:224-255,npu7:256-287,npu8:320-351,npu9:352-383,npu10:384-415,npu11:416-447,npu12:480-511,npu13:512-543,npu14:544-575,npu15:576-607
  extra_args='--npu-scheduler-comm'
#  --npu-enable-super-kernel
fi
#export HCCL_OP_EXPANSION_MODE=AIV

export HCCL_IF_BASE_PORT=8282

export NPU_ENABLE_WEIGHT_NZ=0
export NPU_ENABLE_MC2=0

# tokenizer rayon线程池报错，限制最大线程数
export RAYON_NUM_THREADS=8

pkill -9 sglang
ps -axu | grep "python3 -m sglang.launch_server" | awk '{ print $2 }' | xargs kill -9
ps -axu | grep "sglang::" | awk '{ print $2 }' | xargs kill -9

export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
# export NPU_ENABLE_ALL2ALL_COMM=0
export NPU_ENABLE_GRAPH=1
# export TORCH_LOGS="guards,recompiles"
#export TORCH_LOGS="+dynamo"
#export TORCHDYNAMO_VERBOSE=1
#export IGNORE_INFER_ERROR=1
export NPU_SCMOE_MODE=2
export NPU_O_PROJ_TP_SIZE=8
export NPU_LMHEAD_TP_SIZE=16
# profiling

#rm -rf /tmp/profile_dump
# export NPU_PROF_SAVE_DIR='/tmp/profile_dump'
# export NPU_PROF_WAIT_TIME=1000
# export NPU_PROF_WARMUP_TIME=0
# export NPU_PROF_ACTIVE_TIME=5
# sleep 120

## opti
export NPU_DISTRIBUTE_ZERO=1
export NPU_ENABLE_GET_OUT_CACHE=1
export NPU_ENABLE_MOE_GATING_TOP_K=1
export NPU_MOE_CORE_NUM=12  # default=12 for ep 32/64, 8 for ep128
export NPU_ENABLE_MLA_HW_PROLOG=1
# export NPU_EPLB=2
# export NPU_SUPER_KERNEL=1
# export BEST_EP_BATCH=12s

# py-spy record -o profile.svg --
python3  -m sglang.launch_server ${engine_subfix_cmd} ${decode_args} \
  --device npu \
  --attention-backend npu_mla \
  --model-path $MODEL_PATH \
  --served-model-name Qwen3 \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port 6080 \
  --nnodes ${NODE_NUM} \
  --node-rank ${NODE_RANK} \
  --dist-init-addr ${MASTERIP}:6090 \
  --nprocs-per-node 16 \
  --dense-parallel-strategy combine \
  --dense-tp-size 8 \
  --moe-parallel-strategy ep \
  --disaggregation-mode decode \
  --disaggregation-transfer-backend mooncake_async $model_args \
  --chunked-prefill-size 4096 \
  --max-prefill-tokens 4096 \
  --enable-metrics \
  --schedule-conservativeness 5 \
  --show-time-cost \
  --disable-cuda-graph \
  --page-size 128 \
  --npu-hccl-buffsize-a2a 800 \
  --num-continuous-decode-steps 4 \
  --disable-custom-all-reduce \
  --npu-enable-weight-nz \
  --tokenizer-executor-num-processes 24 \
  --app-key com.sankuai.hadoop.manabo \
  --tool-call-parser longcat \
  --npu-smooth-quant \
  --npu-disable-all-gather \
  $extra_args &

#  --dtype bfloat16 \
#  --npu-o-proj-tp-size 8 \

SGLANG_PID=$!
sleep 45
python3 $(dirname "$0")/bind_core.py ${NODE_RANK}
wait $SGLANG_PID

