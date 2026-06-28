# torchair 编译与 Patch 集成指南
    本文档基于 torchair 7.3.0 分支，结合在 昇腾 A2 环境中的实际验证，提供完整可执行的编译与安装流程。
## 基本信息
    工作目录：SGLang-FluentLLM/flash-npu-kernel/torchair
    torchair 仓库：https://gitcode.com/Ascend/torchair
    目标分支：7.3.0
    基准 Commit：72402379833c4f6f9e3968c32aaccefed5f79ffe

    需应用的补丁（位于 ../patches/）：
        0001-mlp_compute_n_gram_ids.patch
        0002-split-swiglu.patch
        0003-oe.patch
        0004-support-mlp_lightning_indexer-graph.patch
        0005-support-npu-attention-update-graph.patch

## 环境要求
    已安装华为 Ascend 驱动与 ascend-toolkit（推荐 7.0.0 及以上版本）
    已安装 gcc（7.3+）、cmake（3.15+）、make、git
    Python 3.8+，且 pip 可用
    确保当前用户对 /usr/local/Ascend 有读取权限，并能执行 source 脚本

## 详细操作步骤
1. 克隆 torchair 代码仓

    git clone https://gitcode.com/Ascend/torchair
2. 切换到指定 commit（该 commit 属于 7.3.0 分支）

    cd torchair
    git checkout 7.3.0
    git reset --hard 72402379833c4f6f9e3968c32aaccefed5f79ffe

    ⚠️ 若未来仓库强制推送，commit 可能变化；请以仓库实际可获得的 commit 为准。

3. 添加补丁（按顺序，注意相对路径，必须从 torchair 目录下执行）
    patch -p1 < ../patches/0001-mlp_compute_n_gram_ids.patch
    patch -p1 < ../patches/0002-split-swiglu.patch
    patch -p1 --force < ../patches/0003-oe.patch
    patch -p1 --force < ../patches/0004-support-mlp_lightning_indexer-graph.patch
    patch -p1 --force < ../patches/0005-support-npu-attention-update-graph.patch

4. 设置 Ascend 环境变量

    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    export LD_LIBRARY_PATH=/usr/local/lib:/usr/local/Ascend/driver/lib64/driver/:$LD_LIBRARY_PATH

5. 更新并初始化子模块

    git submodule update --init --recursive

6. 配置构建环境（configure）

    bash configure
    脚本会自动探测 Python 路径，若不需要自定义 Python 路径，直接按 Enter 跳过。
7. 编译 torchair

    mkdir build;cd build;cmake ..;make torchair -j8

8. 安装 torchair

    pip install dist/dist/torchair-0.1-py3-none-any.whl
    若需要升级或强制重装：
    pip install --force-reinstall dist/dist/torchair-0.1-py3-none-any.whl --no-deps
