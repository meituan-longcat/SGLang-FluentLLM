# op-plugin 编译安装与卸载指南
## 环境要求
    组件	版本要求
    Python	3.8 / 3.9 / 3.10 / 3.11
    PyTorch	v2.6.0-7.3.0 对应版本
    Ascend Toolkit	已安装并配置环境
## 编译安装
1. 下载代码
    git clone --branch 7.3.0 https://gitcode.com/ascend/op-plugin.git
    cd op-plugin
    git reset --hard 1dfd4eee147309ff787886bad9fa9d0fe0417ee9
2. 添加补丁（0001~0007）
    批量应用补丁
    git am --keep-cr ../SGLang-FluentLLM/flash-npu-kernel/op-plugins/patches/000*.patch
    若出现冲突：手动解决后执行 git add . → git am --continue
    放弃本次应用：git am --abort
3. 设置环境变量并编译
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    bash ci/build.sh --python=3.11 --pytorch=v2.6.0-7.3.0
    参数说明：--python 指定 Python 版本（3.8/3.9/3.10/3.11），--pytorch 指定 PyTorch 版本

4. 安装 torch_npu
    查看生成的 whl 包
    ls dist/*.whl

# 安装（根据实际包名替换）
    pip install --force-reinstall dist/torch_npu-{版本}-{python版本}-{架构}.whl
    示例：

    # ARM 架构 + Python 3.8
    pip install --force-reinstall dist/torch_npu-2.6.0.post13-cp38-cp38-manylinux_2_17_aarch64.manylinux2014_aarch64.whl --no-deps

    # x86_64 架构 + Python 3.11
    pip install --force-reinstall dist/torch_npu-2.6.0.post13-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl --no-deps
