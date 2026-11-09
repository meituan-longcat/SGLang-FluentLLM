# LongCat 2.5 W4A4模型在NPU上推理优化实践样例

## 概述
LongCat 2.5 W4A4 是多模态模型，本样例采用 Encoder、Prefill、Decode（EPD）分离部署。

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

   根据所选运行环境准备配套的 `torch_npu` 和 `torchair` 发行包。本次 kernel 快照中的自定义算子 PyTorch 适配层位于 `flash-npu-kernel/cann-flash-ops/torch_ops_extension/`，在下面的算子安装步骤中单独构建。

3. 下载项目源码并安装依赖的python库。
    ```bash
    # 下载项目源码，使用支持LongCat 2.5 W4A4及EPD分离部署的NPU版本
    git clone  https://github.com/meituan-longcat/SGLang-FluentLLM.git
    cd SGLang-FluentLLM
    git checkout longcat-2.5-npu

    # 安装依赖的python库
    pip3 install -r ./npu_test/requirements.txt --no-deps
    ```

4. 编译并安装 flash-npu-kernel。

   完整构建说明见 [Kernel README](../flash-npu-kernel/README.md)。

   ```text
   flash-npu-kernel/
     cann-flash-ops/
       src/                  # CANN 算子实现
       build.sh              # 生成 CANN 自定义算子 .run 包
       torch_ops_extension/  # 生成 flash_ops wheel
     flash_ops/              # 生成 flash_npu_kernel wheel
     UPSTREAM_COMMIT
   ```

   以下命令均从项目根目录执行。先加载构建环境：

   ```bash
   source "${ASCEND_TOOLKIT_ENV:-/usr/local/Ascend/ascend-toolkit/set_env.sh}"
   source "${ATB_ENV:-/usr/local/Ascend/nnal/atb/set_env.sh}"
   : "${ASCEND_HOME_PATH:?请先加载 CANN 环境并设置 ASCEND_HOME_PATH}"
   python3 -m pip install setuptools wheel ninja
   ```

   若 CANN 环境脚本为 `/usr/local/Ascend/cann/set_env.sh`，先设置 `export ASCEND_TOOLKIT_ENV=/usr/local/Ascend/cann/set_env.sh`。ATB 使用其他安装路径时，通过 `ATB_ENV` 指定对应脚本。

   在 Atlas A2 节点构建并安装 CANN 自定义算子包：

   ```bash
   (
     cd flash-npu-kernel/cann-flash-ops
     bash build.sh -c ascend910b
     bash "output/CANN-flash-ops-linux.$(uname -m).run" \
       --quiet --install-path="${ASCEND_HOME_PATH}"
   )
   source "${ASCEND_HOME_PATH}/vendors/flash_ops/bin/set_env.bash"
   ```

   然后构建并安装两个 Python 算子包：

   ```bash
   (
     cd flash-npu-kernel/cann-flash-ops/torch_ops_extension
     bash build_and_install.sh
     python3 -m pip install --force-reinstall --no-deps dist/flash_ops-*.whl
   )
   (
     cd flash-npu-kernel/flash_ops
     bash build.sh
     python3 -m pip install --force-reinstall --no-deps dist/flash_npu_kernel-*.whl
   )
   ```

   `flash_ops` 与 `flash_npu_kernel` 是两个不同的包，需要分别安装。当前 `build_and_install.sh` 的安装命令被注释，只会构建 wheel；上面的显式 pip 命令用于完成安装。CANN 算子包和 wheel 应使用同一源码版本构建，并匹配各节点的芯片、CPU 架构及 Python/PyTorch 环境。

   检查安装结果：

   ```bash
   python3 -c "import torch, torch_npu, flash_ops, flash_npu_kernel; print('NPU available:', torch.npu.is_available())"
   ```

   各推理节点启动服务时都需加载 CANN、ATB 和 `${ASCEND_HOME_PATH}/vendors/flash_ops/bin/set_env.bash`；批量 SSH 启动时需在远端启动环境中配置这些变量。

5. 配置样例运行所需环境信息。

   运行 `npu_test/flash26b/run_pro_w4a4kv8_mm_epd.sh` 前，通过环境变量配置以下字段：

   - `iplist`：配置各节点之间可解析、互通的 IP 地址或主机名，前 8 个为 Decode 节点，后 4 个为 Prefill 节点；
   - `MODEL_PATH`：P/D 权重所在路径，例如 `/data/meituan-longcat/LongCat-2.5-W4A4`；
   - `ENCODER_URLS`：Encoder HTTP 服务地址，例如 `http://encoder-node0:9101`，多个地址以逗号分隔；
   - `testpath`：节点上的代码和日志存放路径，默认 `/data/longcat25`。

   在批量分发节点设置，例如：

   ```bash
   export MODEL_PATH=/data/meituan-longcat/LongCat-2.5-W4A4
   export ENCODER_URLS=http://encoder-node0:9101
   export iplist="decode-node0 decode-node1 decode-node2 decode-node3
   decode-node4 decode-node5 decode-node6 decode-node7
   prefill-node0 prefill-node1 prefill-node2 prefill-node3"
   export testpath=/data/longcat25
   export time=longcat25
   ```

   节点顺序固定为前 8 台 Decode、后 4 台 Prefill。`MODEL_PATH` 必须是无空格、无通配符的绝对路径；`ENCODER_URLS` 使用无空格的逗号分隔 HTTP(S) 地址。`start/startp/startd` 要求同时设置这两个变量；`code/status/log` 只需要节点列表。

   可通过 `SSH_USER`、`SSH_PORT` 指定登录账号和端口，默认使用当前用户、端口 22 和 SSH 密钥。需要密码认证时设置外部 `SSH_PASSWORD_FILE` 并安装 `sshpass`。

   批量分发节点需完成到 P/D 节点的免密 SSH 认证和主机指纹确认，分发节点与 P/D 节点均需安装 `rsync`。各节点需使用相同的运行环境和算子版本。

   NPU 节点需准备 `/etc/hccn.conf`；使用其他路径时，通过 `HCCN_CONF_FILE` 指定。根据集群网络配置 HCCL、Gloo 和 Mooncake 网卡参数，并保证 Encoder HTTP、编码结果回传及 P/D KV 传输通道互通。

   `testpath` 应与当前源码目录分离。部署新版本时使用独立的目标目录，避免目标目录中保留其他版本的文件。

## 权重准备

请根据所使用的模型类型自行下载原始权重到本地路径，例如`/data/meituan-longcat/LongCat-2.5-W4A4`。

LongCat-2.5-W4A4模型的权重下载地址为：[LongCat-2.5-W4A4权重](https://huggingface.co/meituan-longcat/LongCat-2.5-W4A4)

下载方法可以参考：
```bash
#从huggingface下载权重
pip install -U huggingface_hub
mkdir -p /data/meituan-longcat
hf download meituan-longcat/LongCat-2.5-W4A4 --local-dir /data/meituan-longcat/LongCat-2.5-W4A4
```

## 推理执行

1. 启动 Encoder。

   在 Encoder 节点的项目根目录启动 HTTP 编码服务，`MODEL_PATH` 指向配套的 Encoder 权重：

   ```shell
   export MODEL_PATH=/data/meituan-longcat/LongCat-2.5-Encoder
   export ENCODER_NPU_DEVICES=0,1
   bash npu_test/flash26b/run_encoder_epd.sh
   ```

   Encoder 默认使用 TP=2，HTTP 端口为 9101，输出格式为 `rvq_indices`。可通过 `ENCODER_HTTP_PORT` 修改服务端口。

2. 启动 Prefill、Decode 和 MiniLB。

   Encoder 就绪后，在批量分发节点设置前文的 `MODEL_PATH`（P/D 权重）、`ENCODER_URLS` 和 `iplist` 等环境变量，再从项目根目录执行：

   ```shell
   DRY_RUN=1 bash npu_test/flash26b/run_pro_w4a4kv8_mm_epd.sh start # 预览启动命令
   bash npu_test/flash26b/run_pro_w4a4kv8_mm_epd.sh code  # 分发代码到 P/D 节点
   bash npu_test/flash26b/run_pro_w4a4kv8_mm_epd.sh start # 启动 Prefill、Decode 和 MiniLB
   ```

   P/D 启动参数包括 W4A4、KV INT8、NEXTN MTP，以及 `--enable-multimodal --language-only`。其中 `--language-only` 表示 P/D 节点运行语言部分，视觉编码由独立 Encoder 完成；整个服务支持多模态推理。Prefill 使用 PP=8、Attention TP=8，Decode 使用 KVP=8、DP=16、EP=128；两侧通过 `ENCODER_URLS` 访问 Encoder。

   > 说明：代码已部署到相同路径时，可直接执行 `start`。`code` 将代码分发到各 P/D 节点的 `testpath/SGLang-FluentLLM/` 目录。代码分发使用 `rsync`；MiniLB 在 `iplist` 中最后一个 Prefill 节点启动，默认端口为 8081。

   脚本支持 `code/start/startp/startd/status/log`，可通过 `--help` 查看说明。`startp` 只启动 4 台 Prefill 及 MiniLB，`startd` 只启动 8 台 Decode；Encoder 仍需单独启动。

   第二个参数可指定日志标记，例如 `bash npu_test/flash26b/run_pro_w4a4kv8_mm_epd.sh start run1`。查看、收集日志时使用相同标记；`log` 默认收集到本地 `logs/longcat25/logs_${time}/`，可通过 `logdir` 覆盖。`DRY_RUN=1` 适用于全部操作，只打印命令。

3. 检查启动结果。

   可通过以下命令查看 P/D 启动日志：

   ```shell
   bash npu_test/flash26b/run_pro_w4a4kv8_mm_epd.sh status
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

4. 验证多模态推理。

   向 MiniLB 发送图像请求。`service_ip` 为 MiniLB 所在节点的地址；将示例图像 URL 替换为服务可访问的实际地址。默认服务模型名为 `LongCat-2.5`。

   ```shell
   export service_ip=prefill-node3 # 替换为 MiniLB 节点的实际地址
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
