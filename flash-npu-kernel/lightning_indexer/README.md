# CANN Ops-Transformer 自定义算子编译指南
## 基本信息
    代码仓	https://gitcode.com/cann/ops-transformer
    基准分支	8.5.0
    基准 pr    5034
    目标 SoC	ascend910b
    修改概述
    基于 CANN 官方 ops-transformer 代码仓（分支 8.5.0，pr 5034）进行适配修改，主要包含以下内容：

## 准备基础代码，添加patch
1. 克隆 ops-transformer 代码仓.

    git clone https://gitcode.com/cann/ops-transformer

2. 切换到指定 pr

   cd ops-transformer
   git checkout origin/8.5.0
   git fetch origin
   git fetch https://gitcode.com/cann/ops-transformer.git +refs/merge-requests/5034/head:pr_5034
   git checkout pr_5034

## 编译命令
    编译 lightning_indexer
    设置 Ascend 环境变量
    source /usr/local/Ascend/ascend-toolkit/set_env.sh

    bash build.sh --pkg --soc='ascend910b' --ops='lightning_indexer' --vendor_name='lightning_indexer_patch'
        参数说明
        --pkg	生成安装包
        --soc	目标昇腾芯片型号，此处为 ascend910b
        --ops	需要编译的算子列表，分号分隔

## 环境要求
    CANN 软件包版本：与 8.5.0 分支匹配
    编译环境：昇腾 910B 开发环境或交叉编译环境
    依赖：ops-transformer 仓源码已同步至指定 pr

