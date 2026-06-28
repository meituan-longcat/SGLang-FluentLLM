#!/bin/bash

# 获取所有NPU设备IP地址的脚本
# 定义输出文件，可按需修改
# output_file="/log/hccn.conf"
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PYTHONPATH=$WORKDIR/python:$PYTHONPATH
export LD_LIBRARY_PATH=/usr/local/lib:/usr/local/Ascend/driver/lib64/driver/:$LD_LIBRARY_PATH
output_file=$1

export PATH=/usr/local/Ascend/driver/tools/:$PATH
# 检查hccn_tool工具是否可用
if ! command -v hccn_tool &> /dev/null; then
    echo "错误: 未找到 hccn_tool 命令。请确保已安装昇腾驱动并正确设置环境变量。" >&2
    exit 1
fi

# 获取设备数量（通常为8卡，但可根据实际情况调整）
# 对于Atlas 800T A2/800I A2等服务器，设备ID通常为0-7 [6](@ref)
device_count=${NPROCS_PER_NODE}

# 创建或清空输出文件
> "$output_file"

echo "开始查询NPU设备IP信息..."
echo "采集时间: $(date)"
echo "=================================="

# 循环查询每个设备
for ((i=0; i<device_count; i++)); do
    echo "正在查询设备 $i 的IP信息..."
    
    # 使用hccn_tool查询设备IP地址 和子网掩码 [6,8](@ref)
    d_ip=$(hccn_tool -i $i -ip -g | grep "ipaddr" | awk -F: '{print $2}')
    echo address_${i}=${d_ip} >> "$output_file" 2>&1
    
    # 检查上一条命令执行是否成功
    if [ $? -eq 0 ]; then
        echo "设备 $i 查询成功"
    else
        echo "设备 $i 查询失败"
    fi
    
    echo "----------------------------------"
done

echo "所有设备查询完成！信息已保存到: $output_file"
