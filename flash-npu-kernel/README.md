# flash-npu-kernel

基于华为昇腾 NPU 的自定义算子集合。源码快照对应 `76e1b67d582a30430ff695e4ec2b0c94ab6341c9`，来源记录见 [UPSTREAM_COMMIT](UPSTREAM_COMMIT)。

## 目录和构建产物

```text
flash-npu-kernel/
├── cann-flash-ops/
│   ├── src/                  # Ascend C 算子实现
│   ├── build.sh              # 构建 CANN 自定义算子包
│   └── torch_ops_extension/  # PyTorch 适配层与 torchair converter
├── flash_ops/                # 使用 fast kernel launch 的自定义算子
│   ├── csrc/
│   └── build.sh              # 构建 flash_npu_kernel wheel
└── UPSTREAM_COMMIT
```

| 源码目录 | 用途 | 构建产物 |
| --- | --- | --- |
| [cann-flash-ops](cann-flash-ops/README.md) | CANN 融合算子，包括 ComputeNGramIds、MLP Lightning Indexer、Sparse Flash Attention、KV Cache 操作等 | `output/CANN-flash-ops-linux.<arch>.run` |
| [cann-flash-ops/torch_ops_extension](cann-flash-ops/torch_ops_extension/README.md) | PyTorch 自定义算子注册和 torchair GE 图适配 | `dist/flash_ops-*.whl`，提供 `flash_ops` 和 `fluent_sfa1_ops` 包 |
| [flash_ops](flash_ops/README.md) | 使用 fast kernel launch 的算子，包括请求槽位分配、输出缓存位置计算和 Over-Embedding token table 更新等 | `dist/flash_npu_kernel-*.whl`，提供 `flash_npu_kernel` 包 |

`flash_ops` 和 `flash_npu_kernel` 是两个不同的 Python 包，推理环境中需要分别构建、安装。CANN 算子源码集中在 `cann-flash-ops/src/`，PyTorch 适配层位于其 `torch_ops_extension/` 子目录。

## 环境准备

在昇腾 Linux 构建环境中执行以下步骤。准备相互兼容的驱动、CANN Toolkit、PyTorch、`torch_npu` 和 `torchair`，以及 CMake、C++ 编译器和 CANN 配套的 Bisheng。运行 FluentLLM 时还需准备配套 ATB；本次 NPU 推理样例的版本要求见 [NPU 使用说明](../npu_test/ReadMe.md)。

`torch_npu`、`torchair` 等运行环境组件需单独安装；当前快照中的 `torch_ops_extension` 提供本项目的自定义算子适配层。

以下命令均从 SGLang-FluentLLM 项目根目录执行：

```bash
source "${ASCEND_TOOLKIT_ENV:-/usr/local/Ascend/ascend-toolkit/set_env.sh}"
source "${ATB_ENV:-/usr/local/Ascend/nnal/atb/set_env.sh}"
: "${ASCEND_HOME_PATH:?请先加载 CANN 环境并设置 ASCEND_HOME_PATH}"
python3 -m pip install setuptools wheel ninja
```

`ASCEND_HOME_PATH` 指向实际 CANN Toolkit 根目录。非默认安装位置通过 `ASCEND_TOOLKIT_ENV`、`ATB_ENV` 指定相应环境脚本。

## 构建和安装

### 1. CANN 自定义算子包

Atlas A2 使用 `ascend910b`：

```bash
(
  cd flash-npu-kernel/cann-flash-ops
  bash build.sh -c ascend910b
  bash "output/CANN-flash-ops-linux.$(uname -m).run" \
    --quiet --install-path="${ASCEND_HOME_PATH}"
)
source "${ASCEND_HOME_PATH}/vendors/flash_ops/bin/set_env.bash"
```

这里省略 `-n` 表示构建全部算子，单算子开发时可使用 `-n compute_n_gram_ids`。

安装目录为 `${ASCEND_HOME_PATH}/vendors/flash_ops/`。安装后加载其 `set_env.bash`，以设置 `ASCEND_CUSTOM_OPP_PATH` 和动态库搜索路径。

### 2. PyTorch 自定义算子适配层

```bash
(
  cd flash-npu-kernel/cann-flash-ops/torch_ops_extension
  bash build_and_install.sh
  python3 -m pip install --force-reinstall --no-deps dist/flash_ops-*.whl
)
```

**当前版本的 `build_and_install.sh` 只构建 wheel，脚本内的安装命令已注释，因此需要显式执行上面的 pip 安装命令。** wheel 中同时包含 `flash_ops` 和 `fluent_sfa1_ops`；需与同一源码版本构建的 CANN 算子包配套安装。

### 3. fast kernel launch 算子包

```bash
(
  cd flash-npu-kernel/flash_ops
  bash build.sh
  python3 -m pip install --force-reinstall --no-deps dist/flash_npu_kernel-*.whl
)
```

`build.sh` 安装本目录的构建依赖并生成 wheel，不自动安装 wheel。`NPU_ARCH` 可指定构建架构；详细构建及增量编译说明见 [flash_ops/README.md](flash_ops/README.md)。wheel 文件名由实际 Python 版本、CPU 架构和构建配置决定。

## 安装检查与部署

在项目根目录执行，检查两个适配包是否可导入，以及 NPU 是否可见：

```bash
python3 -c "import torch, torch_npu, flash_ops, flash_npu_kernel; print('NPU available:', torch.npu.is_available())"
```

每个推理节点都需安装相同版本的算子包和 wheel，并在启动服务的环境中加载 CANN、ATB 和 `${ASCEND_HOME_PATH}/vendors/flash_ops/bin/set_env.bash`。使用批量 SSH 启动时，这些配置需要在远端启动环境中生效。

后续按 [NPU 使用说明](../npu_test/ReadMe.md) 启动服务；算子测试入口及调用示例见各子目录文档。
