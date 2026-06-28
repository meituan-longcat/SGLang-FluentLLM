# LongCat 2.0模型在NPU上推理优化实践样例

## 概述
本样例介绍[LongCat 2.0开源代码](https://github.com/meituan-longcat/SGLang-FluentLLM.git) 在CANN平台上完成性能优化，支持在昇腾Atlas A2 Pod平台部署。

### 部署规模

基于Atlas A2 192卡集群，单机16卡，加载真实权重

   | 基础模型     | 机器型号       | Prefill | Decode |
   |-------------|----------------|---------|--------|
   | LongCat 2.0 | Atlas A2 192卡 | 64卡    | 128卡  |

## 支持的产品型号
<term>Atlas A2 系列产品</term>

### 硬件要求
产品型号：Atlas A2 系列

操作系统：Linux X86_64

驱动版本：Ascend HDK 25.0.RC1.1 （[下载链接](https://support.huawei.com/enterprise/zh/ascend-computing/ascend-hdk-pid-252764743/software)）或配套cann8.5.0的驱动（见昇腾社区 [CANN版本兼容性文档](https://www.hiascend.com/document/detail/zh/canncommercial/850/releasenote/releasenote_0000.html))

Python版本：3.11

PyTorch版本：2.6

## 环境准备

### 手动准备环境

1. 安装CANN软件包。

   本样例的编译执行依赖CANN开发套件包（cann-toolkit）与CANN二进制算子包（910b-ops）。

   请从[软件包下载地址](https://support.huawei.com/enterprise/zh/ascend-computing/ascend-hdk-pid-252764743/software)下载`Ascend-cann-toolkit_${version}_linux-${arch}.run`与`Ascend-cann-910b-ops_${version}_linux-${arch}.run`软件包，并参考[CANN安装文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/softwareinst/instg/instg_0001.html?Mode=PmIns&OS=Debian&Software=cannToolKit)进行安装。

    - `${version}`表示CANN包版本号，如8.5.0等。
    - `${arch}`表示CPU架构，如aarch64、x86_64。

2. 安装Ascend Extension for PyTorch（torch_npu/torchair）

   op-plugins插件安装，参考 flash-npu-kernel/op-plugins/README.md
   torchair插件安装，参考 flash-npu-kernel/torchair/README.MD

3. 下载项目源码并安装依赖的python库。
    ```bash
    # 下载项目源码，以master分支为例
    git clone  https://github.com/meituan-longcat/SGLang-FluentLLM.git
    git checkout npu

    # 安装依赖的python库
    cd SGLang-FluentLLM
    pip3 install -r ./npu_test/requirements.txt --no-deps
    ```

4. flash-npu-kernel算子更新
   自定义算子目录结构如下：
        |—————— compute_n_gram_ids
        |—————— mlp_lightning_indexer
        |—————— flash_ops
        |—————— update_oe_token_table
   设置环境变量

        source /usr/local/Ascend/ascend-toolkit/set_env.sh
   编译算子

        cd flash-npu-kernel/compute_n_gram_ids
        bash build.sh
   安装编译生成的自定义算子包

        bash build_out/{opName}_{arch}.run
        - `${opName}`表示自定义算子名称，如compute_n_gram_ids,update_oe_token_table等。
        - `${arch}`表示CPU架构，如aarch64、x86_64。
   按照自定义算子目录结构，依次编译，安装自定义算子compute_n_gram_ids, mlp_lightning_indexer, flash_ops, update_oe_token_table。注：flash_ops 需要用pip install --force-reinstall flash_npu_kernel-1.0.0-cp38-abi3-linux_x86_64.whl --no-deps

   安装torch_ops_extension:
        cd flash-npu-kernel/torch_ops_extension
        bash build_and_install.sh


5. transformer 仓编译安装

   参考 flash-npu-kernel/transformer/路径下的README.md

6. attention_update 仓编译安装

   参考 flash-npu-kernel/attentio_update/路径下的README.md

7. lightning_indexer patch编译安装

   参考 flash-npu-kernel/lightning_indexer/路径下的README.md


8. 配置样例运行所需环境信息。
   修改`npu_test/flash26b/run_pro_dsa_pp.sh`中的如下字段:
   - `iplist`：配置所有节点的IP，按照rank id排序；
   - `MODE_PATH`: 权重所在路径，例如`/data/meituan-longcat/LongCat-2.0`。

## 权重准备

请根据所使用的模型类型自行下载原始权重到本地路径，例如`/data/meituan-longcat/LongCat-2.0`。

LongCat-2.0-Int8模型的权重下载地址为：[LongCat-2.0-Int8权重](https://huggingface.co/meituan-longcat/LongCat-2.0-INT8)

下载方法可以参考：
```bash
#从huggingface下载权重
pip install -U huggingface_hub
mkdir -p /data/meituan-longcat
hf download meituan-longcat/LongCat-2.0 --local-dir /data/meituan-longcat/LongCat-2.0
```

## 推理执行

1. 启动推理服务。

   ```shell
   ln -s SGLang-FluentLLM fluentllm
   cd fluentllm

   #修改npu_test/flash26b/run_pro_dsa_pp.sh脚本中如下参数
   #testpath和logdir为实际测试执行路径
   #iplist为实际测试节点ip
   #port为测试节点ssh登陆端口
   bash npu_test/flash26b/run_pro_dsa_pp.sh code #分发网络脚本到各个节点
   bash npu_test/flash26b/run_pro_dsa_pp.sh start #启动服务
   ```
   > 说明：如在测试执行节点上进行上述操作，则testpath不能与当前代码路径相同。

2. 启动结果检查。

   启动成功后，在decode和prefill节点的首台设备上，均能看到如下启动成功过标志

   ```shell
   INFO - The server is fired up and ready to roll!
   ``

   可通过样例单请求验证服务功能正常，其中service_ip为前述iplist配置中minilb所在设备ip
   ```shell
   curl -X POST http://${service_ip}:8081/v1/chat/completions  \
   -H "Content-Type: application/json" \
   -d '{
    "model": "",
    "messages": [
      {"role": "user", "content": "你好，请介绍一下自己"}
    ],
    "max_tokens": 1024,
    "temperature": 0.7,
    "stream": false
   }'
   ```
