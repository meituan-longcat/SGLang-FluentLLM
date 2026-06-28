# CANN Ops-Transformer 自定义算子编译指南
## 基本信息
    代码仓	https://gitcode.com/cann/ops-transformer
    基准分支	8.5.0
    基准 Commit	92533b8cdfc040b00a0012deae4ac4171f2e3f08
    目标 SoC	ascend910b
    修改概述
    基于 CANN 官方 ops-transformer 代码仓（分支 8.5.0，commit 92533b8）进行适配修改，主要包含以下内容：

## 准备基础代码，添加patch
1. 克隆 ops-transformer 代码仓.

    git clone https://gitcode.com/cann/ops-transformer
2. 切换到指定 commit(该 commit 属于 8.5.0 分支)

    cd ops-transformer
    git checkout origin/8.5.0
    git reset --hard 92533b8cdfc040b00a0012deae4ac4171f2e3f08

3. 添加补丁(按顺序，注意相对路径，必须从 transformer 目录下执行)

    patch -p1  < ../SGLang-FluentLLM/flash-npu-kernel/transformer/patches/0001-modify-lightning_indexer-sparse_flash_attention-moe_.patch

## 编译命令
    编译 lightning_indexer + sparse_flash_attention
    设置 Ascend 环境变量
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    bash build.sh --pkg --soc='ascend910b' --ops='lightning_indexer;sparse_flash_attention' --vendor_name='dsa_ops'
    编译 MC2 算子
    bash build.sh --pkg --ops="moe_distribute_dispatch;moe_distribute_combine;moe_distribute_dispatch_v2;moe_distribute_combine_v2"
        参数说明
        --pkg	生成安装包
        --soc	目标昇腾芯片型号，此处为 ascend910b
        --ops	需要编译的算子列表，分号分隔

## 环境要求
    CANN 软件包版本：与 8.5.0 分支匹配
    编译环境：昇腾 910B 开发环境或交叉编译环境
    依赖：ops-transformer 仓源码已同步至指定 commit

## 注意事项
    请确保基准 commit 92533b8cdfc040b00a0012deae4ac4171f2e3f08 已正确同步
