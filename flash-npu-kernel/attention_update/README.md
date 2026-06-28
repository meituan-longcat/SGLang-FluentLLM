# CANN Ops-Transformer 自定义算子编译指南
## 基本信息
    代码仓	https://gitcode.com/cann/cann-recipes-infer.git
    基准分支	master
    基准pr	175
    目标 SoC	ascend910b
    修改概述
    基于 CANN 官方 cann-recipes-infer 代码仓进行适配修改，主要包含以下内容：

## 准备基础代码，添加patch
1. 克隆 cann-recipes-infer代码仓.

    git clone https://gitcode.com/cann/cann-recipes-infer.git

2. 切换到指定pr

   cd cann-recipes-infer
   git fetch origin
   git fetch https://gitcode.com/cann/cann-recipes-infer.git +refs/merge-requests/175/head:pr_175
   git checkout pr_175

## 编译命令
    编译 attention_update
    设置 Ascend 环境变量
    source /usr/local/Ascend/ascend-toolkit/set_env.sh

    cd ops/ascendc/
    bash build.sh -c "ascend910b" -n "attention_update" --disable-check-compatible
    参数说明
    -c	目标昇腾芯片型号，此处为 ascend910b
    -n	生成安装包
    --disable-check-compatible	不考虑兼容性

## 环境要求
    CANN 软件包版本：与 8.5.0 分支匹配
    编译环境：昇腾 910B 开发环境或交叉编译环境
    依赖：cann-recipes-infer

