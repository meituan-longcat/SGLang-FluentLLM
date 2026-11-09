# torch_ops_extension

## 概述

`torch_ops_extension` 是 `cann-flash-ops` 项目的 PyTorch 自定义算子适配层，基于 PyTorch C++ Extension 机制，将昇腾 NPU 自定义融合算子封装为 Python 可调用的 `torch.ops.custom` 接口，同时支持 `torch_npu` 命名空间调用和 `torch.compile` (torchair) 场景下的 GE 图成图。

## 支持的算子

| 算子名称 | Python 调用接口 | 说明 |
|---------|----------------|------|
| ComputeNGramIds | `torch.ops.custom.npu_compute_n_gram_ids` | 计算 N-gram 嵌入 ID，用于推理阶段 Over-Embedding 加速 |
| GatherElementsV2 | `torch.ops.custom.npu_gather_element` | 按指定维度和索引从输入张量收集元素 |
| GatherSelectionKvCache | `torch.ops.custom.npu_gather_selection_kv_cache` | 根据 TopK 索引从全量 KV Cache 中收集选中块 |
| LightningIndexer | `torch.ops.custom.npu_lightning_indexer` | Lightning Attention 稀疏索引算子 |
| MlpLightningIndexer | `torch.ops.custom.mlp_lightning_indexer` | MLP Lightning Attention 稀疏索引算子 |
| MlpSplitSwiglu | `torch.ops.custom.mlp_split_swiglu` | MLP 分 expert SwiGLU 激活算子 |
| KvpAttentionMerge | `torch.ops.custom.npu_kvp_attention_merge` | 融合 KVP rank attention softmax、加权归约和 BF16 转换 |
| MoeExpertIdMap | `torch.ops.custom.npu_moe_expert_id_map` | 融合 expert ID clamp 与 INT32 映射表查询，用于 MoE Dispatch 前处理 |
| SparseFlashAttention | `torch.ops.custom.npu_sparse_flash_attention` | Lightning Attention 稀疏注意力算子 |

## 目录结构

```
torch_ops_extension/
├── build_and_install.sh          # 编译与安装脚本（一键执行）
├── setup.py                      # wheel 包编译配置文件
├── custom_ops/
│   ├── __init__.py               # 包初始化，加载 .so 并挂载算子到 torch_npu
│   ├── csrc/                     # C++ 算子适配层源码
│   │   ├── ops_def_registration.cpp          # 算子定义与 TORCH_LIBRARY 注册
│   │   ├── ops_common.h                      # 公共头文件（ACL 类型转换、NPU 调用宏等）
│   │   ├── ops_common.cpp                    # 公共实现
│   │   ├── npu_compute_n_gram_ids.cpp        # ComputeNGramIds 算子实现
│   │   ├── npu_gather_element.cpp             # GatherElementsV2 算子实现
│   │   ├── npu_gather_selection_kv_cache.cpp # GatherSelectionKvCache 算子实现
│   │   ├── npu_lightning_indexer.cpp         # LightningIndexer 算子实现
│   │   ├── mlp_lightning_indexer.cpp          # MlpLightningIndexer 算子实现
│   │   ├── mlp_split_swiglu.cpp              # MlpSplitSwiglu 算子实现
│   │   ├── npu_kvp_attention_merge.cpp        # KvpAttentionMerge 算子实现
│   │   ├── npu_moe_expert_id_map.cpp          # MoeExpertIdMap 算子实现
│   │   └── npu_sparse_flash_attention.cpp     # SparseFlashAttention 算子实现
│   └── converter/                # torchair GE 图 converter（用于 torch.compile 场景）
│       ├── __init__.py
│       ├── npu_compute_n_gram_ids.py
│       ├── npu_gather_element.py
│       ├── npu_gather_selection_kv_cache.py
│       ├── npu_lightning_indexer.py
│       ├── mlp_lightning_indexer.py
│       ├── mlp_split_swiglu.py
│       ├── npu_kvp_attention_merge.py
│       ├── npu_moe_expert_id_map.py
│       └── npu_sparse_flash_attention.py
├── build/                        # 编译中间产物（编译后生成）
├── dist/                         # wheel 包输出目录（编译后生成）
└── flash_ops.egg-info/           # egg 元信息（编译后生成）
```

## 环境依赖

### 硬件要求

- 昇腾 Atlas A2 或 Atlas A3 系列

### 软件依赖

| 依赖项 | 说明 |
|-------|------|
| CANN 开发套件 | 需已安装并配置，版本 >= 8.5.T4.0 |
| Python | >= 3.11 |
| PyTorch | 需已安装，与 torch_npu 版本匹配 |
| torch_npu | 需已安装，提供 NPU 适配层与 `NpuExtension` 编译支持 |
| torchair | 可选，用于 `torch.compile` 场景的 GE 图成图 |
| ninja | 可选，设置 `USE_NINJA=1` 时启用加速编译 |
| gcc/g++ | 支持 C++17 标准 |

## 环境准备

### 1. 设置 CANN 环境变量

编译前需确保 CANN 开发套件的环境变量已加载：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

若自定义安装路径，请替换为实际路径。

### 2. 确认 Python 依赖

确保已安装 `torch`、`torch_npu` 等依赖：

```bash
python3 -c "import torch; import torch_npu; print(torch.__version__)"
```

若需要 `torch.compile` 支持，还需确认 `torchair` 已安装：

```bash
python3 -c "import torchair; print('torchair OK')"
```

### 3. 确认自定义融合算子已安装

`torch_ops_extension` 依赖上层项目编译的自定义融合算子（`cann-flash-ops`），需先完成融合算子的编译安装：

```bash
# 在项目根目录执行
cd ../..
bash build.sh -c ascend910_93        # Atlas A3
# 或
bash build.sh -c ascend910b          # Atlas A2
# 或同时编译两种芯片
bash build.sh -c "ascend910b;ascend910_93"

# 安装融合算子
cd output
chmod +x CANN-flash-ops-linux.<arch>.run
./CANN-flash-ops-linux.<arch>.run --quiet --install-path=/usr/local/Ascend/ascend-toolkit/latest
source /usr/local/Ascend/ascend-toolkit/latest/vendors/customize/bin/set_env.bash
```

## 编译与安装

### 一键编译安装

```bash
cd torch_ops_extension
bash build_and_install.sh
```

该脚本执行以下流程：

1. **清理历史编译产物**：删除 `build/` 目录
2. **启用 ninja 加速编译**：设置 `USE_NINJA=1`
3. **设置并行编译任务数**：默认使用 `nproc` 获取 CPU 核心数，可通过 `MAX_JOBS` 环境变量覆盖
4. **编译 C++ Extension**：调用 `python3 setup.py build_ext` 编译 C++ 源码为 `.so` 动态库
5. **打包 wheel**：调用 `python3 setup.py bdist_wheel` 生成 wheel 包
6. **安装 wheel**：使用 `pip3 install` 安装生成的 wheel 包（`-I` 强制覆盖安装）

### 自定义并行编译任务数

```bash
export MAX_JOBS=16
bash build_and_install.sh
```

### 编译产物

编译成功后，在 `dist/` 目录生成 wheel 包：

```
dist/flash_ops-1.0-<python_version>-<python_version>-<arch>.whl
```

例如：`flash_ops-1.0-cp311-cp311-linux_x86_64.whl`

### 手动分步执行

如需分步控制编译和安装过程：

```bash
# 1. 设置环境变量
export USE_NINJA=1
export MAX_JOBS=$(nproc)

# 2. 清理历史编译产物
rm -rf build

# 3. 编译 C++ Extension
python3 setup.py build_ext

# 4. 打包 wheel
python3 setup.py bdist_wheel

# 5. 安装
pip3 install dist/*.whl -I
```

## 编译原理

### setup.py 配置说明

`setup.py` 基于 `torch.utils.cpp_extension` 和 `torch_npu.utils.cpp_extension.NpuExtension` 构建 C++ Extension：

- **源文件收集**：自动扫描 `custom_ops/csrc/` 下所有 `.cpp` 文件
- **头文件路径**：自动添加 `torch_npu` 安装路径下的 ACL 头文件
- **包映射**：将 `custom_ops` 目录映射为 `flash_ops` Python 包
- **子包**：包含 `flash_ops` 和 `flash_ops.converter` 两个子包
- **构建工具**：支持通过 `USE_NINJA` 环境变量选择 ninja 或 make 作为构建后端

### 算子注册机制

1. **C++ 侧**：`ops_def_registration.cpp` 通过 `TORCH_LIBRARY(custom, m)` 宏将所有算子注册到 `torch.ops.custom` 命名空间
2. **Python 侧**：`custom_ops/__init__.py` 在导入时加载 `custom_ops_lib.so`，并将 `torch.ops.custom` 下的算子自动挂载到 `torch_npu` 命名空间

## 使用方式

安装完成后，在 Python 中可通过以下方式使用：

### 方式一：torch.ops.custom 调用

```python
import torch
import torch_npu
import flash_ops

# 调用算子
result = torch.ops.custom.mlp_split_swiglu(x, group_list, local_exp_start=0, local_exp_end=-1)
```

### 方式二：torch_npu 调用

```python
import torch
import torch_npu
import flash_ops

# 通过 torch_npu 命名空间调用
result = torch_npu.mlp_split_swiglu(x, group_list, local_exp_start=0, local_exp_end=-1)
```

### 方式三：torch.compile 场景（torchair）

在 `torch.compile` 场景下，`converter/` 目录中的 converter 会自动将 `torch.ops.custom` 算子节点转换为 GE 图节点，无需额外操作。确保 `torchair` 已安装即可。

## 卸载

```bash
pip3 uninstall flash_ops
```

## 常见问题

### 编译报错：找不到 torch_npu 头文件

确认 `torch_npu` 已正确安装，且 `python3 -c "import torch_npu"` 无报错。`setup.py` 会自动从 `torch_npu` 安装路径获取 ACL 头文件。

### 编译报错：ninja 未找到

若未安装 ninja，可不设置 `USE_NINJA=1`，将使用默认的 make 构建后端。安装 ninja：

```bash
pip3 install ninja
```

### 运行时提示 "torch.ops.custom module is not found"

确认已正确安装 wheel 包，且 `import flash_ops` 在 `torch.ops.custom` 调用之前执行。`import flash_ops` 会触发 `.so` 加载和算子注册。

### 运行时算子调用失败

确认自定义融合算子 run 包已正确安装，且 `ASCEND_CUSTOM_OPP_PATH` 或 `ASCEND_OPP_PATH` 环境变量已设置。C++ 侧通过 `dlopen` 加载 `libcust_opapi.so` 获取底层 aclnn API 实现。

### 重新编译

直接再次执行 `bash build_and_install.sh` 即可，脚本会自动清理历史编译产物。
