# PD分离前缀缓存测试文档
## 概述

本文档介绍了PD（Prefill-Decode）分离架构下的前缀缓存功能测试用例以及相关的开发修改。PD分离是一种将大语言模型推理过程分为prefill和decode两个阶段的服务架构，通过前缀缓存机制可以有效提升系统性能。

## 目录结构

```
benchmark/pd-disagg/
├── README.md                      # 本文档
├── test_pd_prefix_cache_detailed.py   # 详细前缀缓存测试用例
├── bench_pd_prefix_cache.py       # 基础前缀缓存基准测试(暂不使用)
```

## 核心概念

### PD分离架构
- **Prefill Server**: 处理输入序列的prefill阶段，负责计算KV缓存
- **Decode Server**: 处理生成阶段，复用已计算的KV缓存进行推理
- **Load Balancer**: 负责请求调度和负载均衡

### 前缀缓存机制
- **完全复用**: 多个请求具有完全相同的前缀
- **部分复用**: 请求间存在部分共享前缀
- **缓存驱逐**: 当缓存空间不足时的淘汰机制
- **缓存回缩**: 主动回收缓存空间的管理机制

## 测试用例详解

### 1. test_pd_prefix_cache_detailed.py

这是主要的前缀缓存详细测试文件，包含以下测试场景：

#### 1.1 完整前缀复用测试 (`test_full_prefix_reuse`)
**目的**: 验证相同前缀的多个请求能够完全复用缓存
**测试流程**:
1. 生成固定长度的基础前缀
2. 发送第一个请求建立缓存
3. 发送多个具有相同前缀但不同后缀的请求
4. 验证缓存复用率 > 90%

**关键指标**:
- 缓存匹配token数
- 缓存复用率
- 成功复用请求比例

#### 1.2 无前缀复用测试 (`test_no_prefix_reuse`)
**目的**: 验证完全不同的请求不会意外命中缓存
**测试流程**:
1. 生成多个完全不同的输入序列
2. 逐个发送请求并记录缓存匹配情况
3. 验证平均缓存匹配 < 30

**关键指标**:
- 平均缓存匹配数
- 最大缓存匹配数

#### 1.3 部分前缀复用测试 (`test_partial_prefix_reuse`)
**目的**: 验证不同长度的共享前缀能够部分复用
**测试流程**:
1. 建立基础前缀缓存
2. 发送包含50%、75%、90%基础前缀的请求
3. 验证部分复用效果 > 70%

**关键指标**:
- 不同比例的复用率
- 平均复用率

#### 1.4 混合场景测试 (`test_mixed_scenario`)
**目的**: 验证混合类型请求的并发处理能力
**测试场景**:
- 完全复用请求组
- 部分复用请求组  
- 无复用请求组
- 随机打乱发送顺序

**关键指标**:
- 各类型请求的平均缓存效率
- 并发处理稳定性

#### 1.5 并发压力测试 (`test_stress_concurrent_requests`)
**目的**: 验证高并发下缓存机制的稳定性
**测试参数**:
- 20个并发请求
- 包含不同复用类型的请求
- 多线程并发执行

**关键指标**:
- 成功率 > 90%
- 平均延迟 < 30秒
- QPS性能

#### 1.6 长前缀缓存测试 (`test_long_prefix_cache`)
**目的**: 验证大长度前缀的缓存效果
**测试长度**: [512, 1024, 2048, 4096]
**验证指标**: 各长度下的复用率 > 80%

#### 1.7 重复请求复用测试 (`test_duplicate_requests_reuse`)
**目的**: 验证多次发送相同请求的缓存命中率和性能稳定性
**测试参数**:
- 2轮测试
- 每轮5个相同请求
- 总计10个请求

**关键指标**:
- 缓存命中率 > 95%
- 缓存匹配变异系数 < 0.2
- 响应延迟变异系数 < 0.3

#### 1.8 环境变量触发Retract测试 (`test_retract_with_env_variable`)
**目的**: 通过`SGLANG_TEST_RETRACT`环境变量触发缓存回缩机制
**测试方法**:
1. 设置环境变量`SGLANG_TEST_RETRACT=true`
2. 发送少量请求触发回缩
3. 检查decode日志中的retract标记

#### 1.9 小KV缓存触发Evict测试 (`test_evict_with_small_kv`)
**目的**: 通过限制KV缓存空间触发驱逐机制
**测试方法**:
1. 设置`SGLANG_CI_SMALL_KV_SIZE=512`
2. 设置`SGLANG_NUM_RESERVED_DECODE_TOKENS=64`
3. 发送请求填满缓存
4. 检查evict标记

### 2. bench_pd_prefix_cache.py

这是基础的前缀缓存基准测试文件：

#### 2.1 DisaggregationSimulatedBase
基础测试类，提供：
- 服务启动和管理
- 环境配置
- 日志收集
- 进程清理

#### 2.2 TestDisaggregationSimulatedGSM8k
GSM8k数学问题评估测试，验证PD分离在真实工作负载下的准确性

#### 2.3 TestDisaggregationSimulatedRetract
Retract机制测试类，通过环境变量触发并验证回缩功能

#### 2.4 TestDisaggregationPrefixRetaction_系列
不同配置下的缓存驱逐测试：
- 小KV大小配置
- 保留token数量配置
- 缓存容量限制测试

## 开发修改说明

### 1. 架构修改

#### 1.1 服务分离
- **Prefill Server**: 专门处理prefill阶段，不涉及decode推理
- **Decode Server**: 专门处理decode阶段，复用prefill计算的KV缓存
- **Load Balancer**: 协调prefill和decode服务器之间的通信

#### 1.2 缓存机制优化
- **RadixCache**: 在decode端启用RadixCache支持前缀共享
- **内存管理**: 优化的内存分配和回收策略
- **传输优化**: 使用RDMA进行高效的KV缓存传输

### 2. 关键配置参数
目前代码启动在虚拟环境存在找不到环境的问题尚未解决，建议命令行启动
参考命令如下
``` bash
pkill -f "sglang" || true
pkill -f "mini_lb" || true

nohup python3 -m sglang.srt.disaggregation.mini_lb --host 0.0.0.0 --port 8192 >lb.log 2>&1 &
nohup python3 -m sglang.launch_server --model-path /mnt/hdfs/zw04mlnn01/checkpoint/llm_platform/model/DeepSeek-V2-Lite --disaggregation-mode prefill --base-gpu-id 0 --context-length 448 --low-latency-max-num-tokens-per-gpu 512 --chunked-prefill-size 4096 --port 8292 --disable-cuda-graph --enable-flashinfer-mla --trust-remote-code --moe-parallel-strategy ep --dense-parallel-strategy rep --nprocs-per-node 1 --attn-tp-size 1 --dp-size 1 --random-seed 1234 --host 0.0.0.0 --log-level debug --pdlb-url http://0.0.0.0:8192 >pr.log 2>&1 &
nohup python3 -m sglang.launch_server --model-path /mnt/hdfs/zw04mlnn01/checkpoint/llm_platform/model/DeepSeek-V2-Lite --disaggregation-mode decode --base-gpu-id 2 --context-length 448 --low-latency-max-num-tokens-per-gpu 512 --chunked-prefill-size 4096 --port 8392 --disable-cuda-graph --enable-flashinfer-mla --trust-remote-code --moe-parallel-strategy ep --dense-parallel-strategy rep --nprocs-per-node 1 --attn-tp-size 1 --dp-size 1 --random-seed 1234 --host 0.0.0.0 --log-level debug --pdlb-url http://0.0.0.0:8192 >de.log 2>&1 &

sleep 40
python3 benchmark/pd-disagg/test_pd_prefix_cache_detailed.py --test_case all

```

如果需要测试evict或者retract需要在服务拉起前设置环境变量
``` bash
export SGLANG_CI_SMALL_KV_SIZE=512
export SGLANG_NUM_RESERVED_DECODE_TOKENS=64
pkill -f "sglang" || true
pkill -f "mini_lb" || true

nohup python3 -m sglang.srt.disaggregation.mini_lb --host 0.0.0.0 --port 8192 >lb.log 2>&1 &
nohup python3 -m sglang.launch_server --model-path /mnt/hdfs/zw04mlnn01/checkpoint/llm_platform/model/DeepSeek-V2-Lite --disaggregation-mode prefill --base-gpu-id 0 --context-length 448 --low-latency-max-num-tokens-per-gpu 512 --chunked-prefill-size 4096 --port 8292 --disable-cuda-graph --enable-flashinfer-mla --trust-remote-code --moe-parallel-strategy ep --dense-parallel-strategy rep --nprocs-per-node 1 --attn-tp-size 1 --dp-size 1 --random-seed 1234 --host 0.0.0.0 --log-level debug --pdlb-url http://0.0.0.0:8192 >pr.log 2>&1 &
nohup python3 -m sglang.launch_server --model-path /mnt/hdfs/zw04mlnn01/checkpoint/llm_platform/model/DeepSeek-V2-Lite --disaggregation-mode decode --base-gpu-id 2 --context-length 448 --low-latency-max-num-tokens-per-gpu 512 --chunked-prefill-size 4096 --port 8392 --disable-cuda-graph --enable-flashinfer-mla --trust-remote-code --moe-parallel-strategy ep --dense-parallel-strategy rep --nprocs-per-node 1 --attn-tp-size 1 --dp-size 1 --random-seed 1234 --host 0.0.0.0 --log-level debug --pdlb-url http://0.0.0.0:8192 >de.log 2>&1 &

sleep 40
python3 benchmark/pd-disagg/test_pd_prefix_cache_detailed.py --test_case evict

unset SGLANG_TEST_RETRACT
unset SGLANG_CI_SMALL_KV_SIZE
unset SGLANG_NUM_RESERVED_DECODE_TOKENS
```

``` bash

export SGLANG_TEST_RETRACT=true
pkill -f "sglang" || true
pkill -f "mini_lb" || true

nohup python3 -m sglang.srt.disaggregation.mini_lb --host 0.0.0.0 --port 8192 >lb.log 2>&1 &
nohup python3 -m sglang.launch_server --model-path /mnt/hdfs/zw04mlnn01/checkpoint/llm_platform/model/DeepSeek-V2-Lite --disaggregation-mode prefill --base-gpu-id 0 --context-length 448 --low-latency-max-num-tokens-per-gpu 512 --chunked-prefill-size 4096 --port 8292 --disable-cuda-graph --enable-flashinfer-mla --trust-remote-code --moe-parallel-strategy ep --dense-parallel-strategy rep --nprocs-per-node 1 --attn-tp-size 1 --dp-size 1 --random-seed 1234 --host 0.0.0.0 --log-level debug --pdlb-url http://0.0.0.0:8192 >pr.log 2>&1 &
nohup python3 -m sglang.launch_server --model-path /mnt/hdfs/zw04mlnn01/checkpoint/llm_platform/model/DeepSeek-V2-Lite --disaggregation-mode decode --base-gpu-id 2 --context-length 448 --low-latency-max-num-tokens-per-gpu 512 --chunked-prefill-size 4096 --port 8392 --disable-cuda-graph --enable-flashinfer-mla --trust-remote-code --moe-parallel-strategy ep --dense-parallel-strategy rep --nprocs-per-node 1 --attn-tp-size 1 --dp-size 1 --random-seed 1234 --host 0.0.0.0 --log-level debug --pdlb-url http://0.0.0.0:8192 >de.log 2>&1 &

sleep 40
python3 benchmark/pd-disagg/test_pd_prefix_cache_detailed.py --test_case retract

unset SGLANG_TEST_RETRACT

```

#### 2.1 服务配置
```bash
# Prefill Server
--disaggregation-mode prefill
--base-gpu-id 0
--port 8292

# Decode Server  
--disaggregation-mode decode
--base-gpu-id 1
--port 8392
--disable-radix-cache=false

# Load Balancer
--host 0.0.0.0
--port 8192
```

#### 2.2 缓存配置
```bash
# 基础缓存配置
--page-size 64
--max-total-tokens 3210944
--context-length 163840

# 测试专用配置
SGLANG_CI_SMALL_KV_SIZE=512          # 小KV缓存空间
SGLANG_NUM_RESERVED_DECODE_TOKENS=64 # 保留token数量
SGLANG_TEST_RETRACT=true             # 触发retract机制
```

#### 2.3 性能优化配置
```bash
--enable-flashinfer-mla               # 启用MLA优化
--moe-parallel-strategy ep           # MoE专家并行
--dense-parallel-strategy rep         # 密集层复制并行
--disable-cuda-graph                 # 禁用CUDA图(兼容性)
```

### 3. 环境依赖

#### 3.1 基础环境
```bash
# Python环境
Python 3.9+
PyTorch 2.0+
CUDA 11.8+

# 依赖库
sglang
flashinfer
requests
numpy
```

#### 3.2 系统要求
```bash
# GPU要求
8卡GPU (至少2张，分别用于prefill和decode)
显存 >= 16GB per GPU

# 网络要求
RDMA支持 (用于高效传输)
足够的带宽支持
```

### 4. 日志分析

#### 4.1 重要日志标记
```bash
# Retract标记
[retract_decode] retracting

# Evict标记  
[evict]

# 缓存匹配信息
cached_tokens: xxx
decode_prefix_len: xxx
```

#### 4.2 性能指标
```bash
# 内存使用
avail mem=xxx GB
max_total_page_num=xxx

# 吞吐量
QPS: xxx requests/second
平均延迟: xxx seconds
P95延迟: xxx seconds
```

## 使用方法

### 1. 快速开始
```bash
# 运行所有测试
python test_pd_prefix_cache_detailed.py --test_case all

# 运行特定测试
python test_pd_prefix_cache_detailed.py --test_case full_reuse
python test_pd_prefix_cache_detailed.py --test_case partial_reuse
python test_pd_prefix_cache_detailed.py --test_case retract
python test_pd_prefix_cache_detailed.py --test_case evict

# 查看日志文件位置
python test_pd_prefix_cache_detailed.py --tail-logs
```

### 2. 手动启动服务
```bash
# 1. 启动Load Balancer
python3 -m sglang.srt.disaggregation.mini_lb --host 0.0.0.0 --port 8192

# 2. 启动Prefill Server
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m sglang.srt.launch_server \
    --model-path /mnt/hdfs/zw04mlnn01/checkpoint/llm_platform/model/DeepSeek-V2-Lite \
    --port 8292 \
    --disaggregation-mode prefill \
    --base-gpu-id 0 \
    --pdlb-url http://127.0.0.1:8192

# 3. 启动Decode Server  
CUDA_VISIBLE_DEVICES=4,5,6,7 python3 -m sglang.srt.launch_server \
    --model-path /mnt/hdfs/zw04mlnn01/checkpoint/llm_platform/model/DeepSeek-V2-Lite \
    --port 8392 \
    --disaggregation-mode decode \
    --base-gpu-id 4 \
    --pdlb-url http://127.0.0.1:8192
```

### 3. 测试验证
```bash
# 检查服务健康状态
curl http://127.0.0.1:8192/health
curl http://127.0.0.1:8292/health  
curl http://127.0.0.1:8392/health

# 发送测试请求
curl -X POST http://127.0.0.1:8192/generate \
    -H "Content-Type: application/json" \
    -d '{"input_ids": [1,2,3,4,5], "sampling_params": {"max_new_tokens": 10}}'
```

## 性能基准

### 1. 缓存效率目标
- **完全复用**: > 90% 缓存命中率
- **部分复用**: > 70% 平均复用率
- **混合场景**: > 60% 整体复用率
- **重复请求**: > 95% 缓存命中率

### 2. 性能目标
- **并发成功率**: > 90%
- **平均延迟**: < 30秒 (大请求)
- **响应稳定性**: 变异系数 < 0.3
- **QPS**: 根据硬件配置优化

### 3. 资源使用
- **内存利用率**: > 80%
- **GPU利用率**: > 70%
- **网络带宽**: 充分利用RDMA带宽

## 故障排查

### 1. 常见问题

#### 1.1 服务启动失败
```bash
# 检查端口占用
netstat -tulpn | grep 8192
netstat -tulpn | grep 8292
netstat -tulpn | grep 8392

# 检查GPU状态
nvidia-smi

# 检查环境变量
echo $CUDA_VISIBLE_DEVICES
echo $PYTHONPATH
```

#### 1.2 缓存效果不佳
```bash
# 检查缓存配置
grep -i cache *.log

# 检查RadixCache状态
grep "tree_cache" *.log

# 检查内存使用
grep "avail mem" *.log
```

#### 1.3 性能问题
```bash
# 检查RDMA状态
ibv_devinfo

# 检查网络连接
ping -c 3 <target_ip>

# 检查GPU利用率
nvidia-smi dmon -s u
```

### 2. 日志分析工具
```bash
# 实时查看日志
tail -f de.log | grep -E "(retract|evict|cached)"

# 统计缓存命中率
grep "cached_tokens" de.log | awk '{print $NF}' | sort | uniq -c

# 分析性能指标
grep "QPS\|延迟" *.log
```

## 开发指南

### 1. 添加新测试用例
1. 继承`DetailedPrefixCacheTest`类
2. 实现`@seperator`装饰的测试方法
3. 使用`send_request_and_get_resp`发送请求
4. 记录测试结果到`self.test_results`
5. 添加相应的断言验证

### 2. 修改测试参数
编辑测试用例中的配置参数：
```python
# 修改前缀长度
prefix_length = 256

# 修改请求数量  
num_requests = 20

# 修改缓存阈值
self.assertGreater(cache_efficiency, 0.8)
```
