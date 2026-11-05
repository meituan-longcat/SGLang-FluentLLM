# LongCat 2.5 W4A4模型在NPU上推理优化实践样例

## 概述
本样例介绍[LongCat 2.5开源代码](https://github.com/meituan-longcat/SGLang-FluentLLM.git) 在CANN平台上完成性能优化，支持在昇腾Atlas A2 Pod平台部署。

### 部署规模

基于Atlas A2 194 卡集群，P/D单机16卡，Encoder独立部署2卡，加载真实权重

| 基础模型 | 机器型号 | Encoder | Prefill | Decode |
| --- | --- | --- | --- | --- |
| LongCat 2.5 W4A4 | Atlas A2 194 卡 | 2卡 | 64卡 | 128卡 |

## 支持的产品型号
<term>Atlas A2 系列产品</term>

### 硬件要求
产品型号：Atlas A2 系列

操作系统：Linux X86_64

驱动版本：Ascend HDK 25.0.RC1.1 （[下载链接](https://support.huawei.com/enterprise/zh/ascend-computing/ascend-hdk-pid-252764743/software)）或配套cann9.0.0的驱动（见昇腾社区 [CANN版本兼容性文档](https://www.hiascend.com/document/detail/zh/canncommercial/latest/releasenote/release-notes.md))

Python版本：3.11

PyTorch版本：2.9

## 环境准备

### 手动准备环境

1. 安装CANN软件包。

   本样例的编译执行依赖CANN开发套件包（cann-toolkit）、CANN二进制算子包（910b-ops）及NNAL软件包中的ATB加速库，三个软件包需使用配套版本。

   请从[软件包下载地址](https://support.huawei.com/enterprise/zh/ascend-computing/ascend-hdk-pid-252764743/software)下载`Ascend-cann-toolkit_${version}_linux-${arch}.run`与`Ascend-cann-910b-ops_${version}_linux-${arch}.run`软件包，并安装配套的 `Ascend-cann-nnal_${version}_linux-${arch}.run`，安装方法参考[CANN安装文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/softwareinst/instg/instg_0001.html?Mode=PmIns&OS=Debian&Software=cannToolKit)进行安装。

    - `${version}`表示CANN包版本号，如9.0.0等。
    - `${arch}`表示CPU架构，如aarch64、x86_64。

2. 安装Ascend Extension for PyTorch（torch_npu/torchair）

   torch_npu 和 torchair 需使用与 PyTorch 2.9.0、CANN 9.0.0 配套的版本。

   op-plugins插件安装，参考 flash-npu-kernel/op-plugins/README.md
   torchair插件安装，参考 flash-npu-kernel/torchair/README.MD

3. 下载项目源码并安装依赖的python库。
    ```bash
    # 下载项目源码，使用支持LongCat 2.5 W4A4及EPD分离部署的NPU版本
    git clone  https://github.com/meituan-longcat/SGLang-FluentLLM.git
    cd SGLang-FluentLLM
    git checkout longcat-2.5-npu

    # 安装依赖的python库
    pip3 install -r ./npu_test/requirements.txt --no-deps
    ```

4. flash-npu-kernel算子更新

   以下命令从项目根目录执行。自定义算子目录结构如下：

   ```text
   flash-npu-kernel/
     compute_n_gram_ids/
     mlp_lightning_indexer/
     flash_ops/
     update_oe_token_table/
   ```

   设置环境变量：

   ```bash
   source "${ASCEND_TOOLKIT_ENV:-/usr/local/Ascend/ascend-toolkit/set_env.sh}"
   source "${ATB_ENV:-/usr/local/Ascend/nnal/atb/set_env.sh}"
   ```

   若 CANN 安装后的环境脚本为 `/usr/local/Ascend/cann/set_env.sh`，请先设置 `export ASCEND_TOOLKIT_ENV=/usr/local/Ascend/cann/set_env.sh`。ATB 使用其他安装路径时，通过 `ATB_ENV` 指定对应的 `set_env.sh`。

   编译、安装自定义算子，以 `compute_n_gram_ids` 为例：

   ```bash
   cd flash-npu-kernel/compute_n_gram_ids
   bash build.sh
   # 按实际生成的算子名称与架构替换包名
   bash build_out/{opName}_{arch}.run
   cd ../..
   ```

   - `${opName}` 表示自定义算子名称，如 `compute_n_gram_ids`、`update_oe_token_table` 等。
   - `${arch}` 表示 CPU 架构，如 `aarch64`、`x86_64`。

   按各目录说明依次编译、安装 `compute_n_gram_ids`、`mlp_lightning_indexer`、`flash_ops` 和 `update_oe_token_table`。每个算子安装完成后返回项目根目录。`flash_ops` 使用以下命令安装，wheel 文件名以实际构建结果为准：

   ```bash
   pip install --force-reinstall flash_npu_kernel-1.0.0-cp38-abi3-linux_x86_64.whl --no-deps
   ```

   安装 `torch_ops_extension`：

   ```bash
   cd flash-npu-kernel/torch_ops_extension
   bash build_and_install.sh
   cd ../..
   ```

5. transformer 仓编译安装

   参考 flash-npu-kernel/transformer/路径下的README.md

6. attention_update 仓编译安装

   参考 flash-npu-kernel/attention_update/路径下的README.md

7. lightning_indexer patch编译安装

   参考 flash-npu-kernel/lightning_indexer/路径下的README.md


8. 配置样例运行所需环境信息。

   修改 `npu_test/flash26b/run_w4a4kv8_mm_epd.sh` 中的如下字段：

   - `iplist`：配置各节点之间可解析、互通的 IP 地址或主机名，前 8 个为 Decode 节点，后 4 个为 Prefill 节点；
   - `MODEL_PATH`：P/D 权重所在路径，例如 `/data/meituan-longcat/LongCat-2.5-W4A4`；
   - `ENCODER_URLS`：Encoder HTTP 服务地址，例如 `http://encoder-node0:9101`，多个地址以逗号分隔；
   - `testpath`：节点上的代码和日志存放路径。

   批量分发节点需完成到 P/D 节点的免密 SSH 认证和主机指纹确认，分发节点与 P/D 节点均需安装 `rsync`。各节点需使用相同的运行环境和算子版本。

   NPU 节点需准备 `/etc/hccn.conf`；使用其他路径时，通过 `HCCN_CONF_FILE` 指定。根据集群网络配置 HCCL、Gloo 和 Mooncake 网卡参数，并保证 Encoder HTTP、编码结果回传及 P/D KV 传输通道互通。

   `testpath` 应与当前源码目录分离。部署新版本时使用独立的目标目录，避免目标目录中保留其他版本的文件。

## 权重准备

请根据所使用的模型类型自行下载原始权重到本地路径，例如`/data/meituan-longcat/LongCat-2.5-W4A4`。

LongCat-2.5-W4A4模型的权重下载地址为：[LongCat-2.5-W4A4权重](https://huggingface.co/meituan-longcat/LongCat-2.0-INT8)

下载方法可以参考：
```bash
#从huggingface下载权重
pip install -U huggingface_hub
mkdir -p /data/meituan-longcat
hf download meituan-longcat/LongCat-2.0 --local-dir /data/meituan-longcat/LongCat-2.5-W4A4
```

## 推理执行

1. 启动推理服务。

   本样例采用 EPD 分离部署。先在 Encoder 节点的项目根目录启动 HTTP 编码服务：

   ```shell
   export MODEL_PATH=/data/meituan-longcat/LongCat-2.5-Encoder
   export ENCODER_NPU_DEVICES=0,1
   bash npu_test/flash26b/run_encoder_epd.sh
   ```

   Encoder 默认使用 TP=2，HTTP 端口为 9101，输出格式为 `rvq_indices`。可通过 `ENCODER_HTTP_PORT` 修改服务端口。

   Encoder 启动后，在批量分发节点的项目根目录执行：

   ```shell
   bash npu_test/flash26b/run_w4a4kv8_mm_epd.sh code  # 分发代码到 P/D 节点
   bash npu_test/flash26b/run_w4a4kv8_mm_epd.sh start # 启动 Prefill、Decode 和 MiniLB
   ```

   P/D 启动参数包括 W4A4、KV INT8、NEXTN MTP 和多模态 `language_only` 模式。Prefill 使用 PP=8、Attention TP=8，Decode 使用 KVP=8、DP=16、EP=128；两侧通过 `ENCODER_URLS` 访问 Encoder。

   > 说明：代码已部署到相同路径时，可直接执行 `start`。`code` 将代码分发到各 P/D 节点的 `testpath/SGLang-FluentLLM/` 目录。代码分发使用 `rsync`；MiniLB 在 `iplist` 中最后一个 Prefill 节点启动，默认端口为 8081。

2. 启动结果检查。

   可通过以下命令查看 P/D 启动日志：

   ```shell
   bash npu_test/flash26b/run_w4a4kv8_mm_epd.sh status
   ```

   日志保存在各节点的 `testpath/testlog_${time}.log`。启动成功后，在 Decode 和 Prefill 的 rank 0 节点均能看到如下标志：

   ```shell
   INFO - The server is fired up and ready to roll!
   ```

   检查 Encoder 和 P/D 服务状态：

   ```shell
   curl --fail http://encoder-node0:9101/health
   curl --fail http://prefill-node0:6080/health
   curl --fail http://decode-node0:6080/health
   ```

   可通过样例单请求验证服务功能，其中 `service_ip` 为 MiniLB 所在节点的地址：

   ```shell
   export service_ip=prefill-node3 # 替换为 MiniLB 节点的实际地址
   curl -X POST http://${service_ip}:8081/v1/chat/completions \
   -H "Content-Type: application/json" \
   -d '{
     "model": "LongCat-2.5",
     "messages": [
       {"role": "user", "content": "你好，请介绍一下自己"}
     ],
     "max_tokens": 1024,
     "temperature": 0.7,
     "stream": false
   }'
   ```

   图像请求使用以下格式，将图像 URL 替换为服务可访问的地址：

   ```shell
   curl -X POST http://${service_ip}:8081/v1/chat/completions \
   -H "Content-Type: application/json" \
   -d '{
     "model": "LongCat-2.5",
     "messages": [
       {"role": "user", "content": [
         {"type": "text", "text": "请描述这张图片。"},
         {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}}
       ]}
     ],
     "max_tokens": 1024,
     "temperature": 0.7,
     "stream": false
   }'
   ```
