#!/bin/bash

source /usr/local/Ascend/ascend-toolkit/set_env.sh
WORKDIR=`pwd`

prefill_nodes_num=$1
prefill_nodes_size=$2
decode_nodes_num=$3
decode_nodes_size=$4

# 解析 IP 列表
ips=($(echo ${cluster_ip_list} | tr ',' ' '))

export PYTHONPATH=$WORKDIR/python:$PYTHONPATH
export LD_LIBRARY_PATH=/usr/local/lib:/usr/local/Ascend/driver/lib64/driver/:$LD_LIBRARY_PATH

url_fix="http://"

# 构建所有 decode 节点的 URL
# 每个 decode 节点的 rank0 local_id = idx * decode_nodes_size
decode_url=""
for (( i=0; i<decode_nodes_num; i++ )); do
    decode_rank0_local_id=$(( i * decode_nodes_size ))
    decode_ip=${ips[${decode_rank0_local_id}]}
    decode_url="${decode_url}${url_fix}${decode_ip}:6080 "
done

# 构建所有 prefill 节点的 URL
# 每个 prefill 节点的 rank0 local_id = decode_devices + idx * prefill_nodes_size
prefill_url=""
total_decode_devices=$(( decode_nodes_num * decode_nodes_size ))

for (( i=0; i<prefill_nodes_num; i++ )); do
    prefill_rank0_local_id=$(( total_decode_devices + i * prefill_nodes_size ))
    prefill_ip=${ips[${prefill_rank0_local_id}]}
    prefill_url="${prefill_url}${url_fix}${prefill_ip}:6080 "
done

echo "Starting minilb:"
echo "  decode_urls=${decode_url}"
echo "  prefill_urls=${prefill_url}"

python3 -m sglang.srt.disaggregation.mini_lb \
  --host 0.0.0.0 --port 8081 \
  --prefill ${prefill_url} \
  --decode ${decode_url}
