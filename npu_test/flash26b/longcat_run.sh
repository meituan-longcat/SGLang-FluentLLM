#!/bin/bash

# 设置节点数量和每节点大小（设备数）
prefill_nodes_num=$1
prefill_nodes_size=$2  # 每个 prefill 节点的设备数 (ep_size)

decode_nodes_num=$3
decode_nodes_size=$4   # 每个 decode 节点的设备数 (ep_size)

echo "prefill: nodes=${prefill_nodes_num}, size_per_node=${prefill_nodes_size}"
echo "decode: nodes=${decode_nodes_num}, size_per_node=${decode_nodes_size}"

# 默认每个服务器16个NPU设备
npu_per_server=16

currpath=`pwd`
local_id=${Node_idx}

# 计算总设备数
total_prefill_devices=$(( prefill_nodes_num * prefill_nodes_size ))
total_decode_devices=$(( decode_nodes_num * decode_nodes_size ))


if [ ${local_id} -lt ${total_decode_devices} ]; then
    # ========== DECODE 节点 ==========
    # 计算该 rank 属于第几个 decode 节点
    decode_idx=$(( local_id / decode_nodes_size ))
    # 计算该 rank 在当前 decode 节点内的 rank
    decode_rank=$(( local_id % decode_nodes_size ))
    # 获取当前 decode 节点 rank0 的 IP (该节点的起始 local_id 对应的 IP)
    ips=($(echo $cluster_ip_list | tr ',' ' '))
    decode_rank0_local_id=$(( decode_idx * decode_nodes_size ))
    MASTERIP="${MASTERIP:-${ips[${decode_rank0_local_id}]}}"

    echo "Starting DECODE: local_id=${local_id}, decode_idx=${decode_idx}, decode_rank=${decode_rank}, MASTERIP=${MASTERIP}"

    export MASTERIP=${MASTERIP}
    bash npu_test/flash26b/run_decode.sh ${decode_rank} ${decode_nodes_size} &
else
    # ========== PREFILL 节点 ==========
    # 计算在 prefill 集群内的偏移
    prefill_local_id=$(( local_id - total_decode_devices ))
    # 计算该 rank 属于第几个 prefill 节点
    prefill_idx=$(( prefill_local_id / prefill_nodes_size ))
    # 计算该 rank 在当前 prefill 节点内的 rank
    prefill_rank=$(( prefill_local_id % prefill_nodes_size ))
    # 获取当前 prefill 节点 rank0 的 IP (该节点的起始 local_id 对应的 IP)
    ips=($(echo $cluster_ip_list | tr ',' ' '))
    prefill_rank0_local_id=${local_id}  # 当前 prefill 节点的起始 local_id 就是当前 local_id 减去 rank 偏移
    prefill_rank0_local_id=$(( total_decode_devices + prefill_idx * prefill_nodes_size ))
    MASTERIP="${MASTERIP:-${ips[${prefill_rank0_local_id}]}}"

    echo "Starting PREFILL: local_id=${local_id}, prefill_idx=${prefill_idx}, prefill_rank=${prefill_rank}, MASTERIP=${MASTERIP}"

    export MASTERIP=${MASTERIP}
    bash npu_test/flash26b/run_prefill.sh  ${prefill_rank} ${prefill_nodes_size} &
fi

last_node=$(( total_prefill_devices + total_decode_devices -1 ))

if [ ${local_id} -eq ${last_node} ]; then
    echo "Starting minilb..."
    bash npu_test/flash26b/run_minilb.sh ${prefill_nodes_num} ${prefill_nodes_size} ${decode_nodes_num} ${decode_nodes_size} &
fi

wait
