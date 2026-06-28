# flash_npu_kernel

基于 AscendC 的 PyTorch 自定义算子库，使用 fast kernel launch（`<<<>>>` 直调）方式实现高性能 NPU 算子，编译为 Python wheel 包供直接安装使用。

**构建模型：一算子一 so。** 每个算子独立产出 `libflash_<op>.so`，支持单算子增量编译与热替换，无需重打 wheel。

## 项目结构

```
flash_npu_kernel/
├── CMakeLists.txt                  # 根 CMake 配置（只做 include / add_subdirectory）
├── setup.py                        # Python wheel 构建入口
├── requirements.txt                # 构建依赖
├── build.sh                        # 构建脚本：默认产出 wheel；--ops=a[,b,...] 单/多算子增量构建
├── cmake/                          # CMake 工具模块
│   ├── ascend.cmake                # 检测 Ascend 工具链路径，设置 Bisheng 编译器
│   ├── func.cmake                  # 提供 recursive_add_subdirectory / add_sources 宏
│   ├── python.cmake                # 检测 Python 解释器和开发库
│   ├── torch.cmake                 # 检测 PyTorch
│   └── torch_npu.cmake             # 检测 torch_npu
├── csrc/                           # C++ 源码目录（算子按名称组织）
│   ├── CMakeLists.txt              # 自动扫描子目录中的算子
│   ├── assign_req_to_token_pool/dav-2201/
│   │   ├── CMakeLists.txt
│   │   └── assign_req_to_token_pool.asc
│   ├── get_out_cache_loc/dav-2201/
│   │   ├── CMakeLists.txt
│   │   └── get_out_cache_loc.asc
│   ├── rearrange_accept_index/dav-2201/
│   │   ├── CMakeLists.txt
│   │   └── rearrange_accept_index.asc
│   └── update_oe_token_table/dav-2201/
│       ├── CMakeLists.txt
│       └── update_oe_token_table.asc
├── flash_npu_kernel/               # Python 包
│   ├── __init__.py                 # 扫描并 torch.ops.load_library 加载所有 libflash_*.so
│   └── libflash_*.so               # 各算子独立共享库（构建产物）
└── tests/                          # 测试目录（按算子名称组织）
    ├── assign_req_to_token_pool/test_assign_req_to_token_pool.py
    ├── get_out_cache_loc/test_get_out_cache_loc.py
    ├── rearrange_accept_index/test_rearrange_accept_index.py
    └── update_oe_token_table/test_update_oe_token_table.py
```

## 已包含算子

| 算子 | 调用方式 | 说明 |
|---|---|---|
| `npu_assign_req_to_token_pool` | `torch.ops.flash.npu_assign_req_to_token_pool` | In-place scatter to req_to_token_pool |
| `npu_get_out_cache_loc` | `torch.ops.flash.npu_get_out_cache_loc` | In-place gather from req_to_token |
| `npu_rearrange_accept_index` | `torch.ops.flash.npu_rearrange_accept_index` | In-place flatten of accept_index by accept_length |
| `npu_update_oe_token_table` | `torch.ops.flash.npu_update_oe_token_table` | Update OE token table |

## 添加新算子

以添加名为 `foo` 的算子为例，需要提供以下交付件：

### 1. 算子实现：`csrc/foo/dav-2201/foo.asc`

单文件包含算子的完整实现，由 4 个必要部分组成：

| 部分 | 宏 / 关键代码 | 作用 |
|------|--------------|------|
| **Schema 注册** | `TORCH_LIBRARY_FRAGMENT(EXTENSION_MODULE_NAME, m)` | 向 PyTorch 声明算子签名（名称、输入输出类型），使其可通过 `torch.ops.flash.foo` 调用 |
| **Meta 函数** | `TORCH_LIBRARY_IMPL(..., Meta, m)` | 推断输出 tensor 的 shape 和 dtype，不执行实际计算；支撑 `torch.compile` 和 AutoGrad |
| **Kernel 实现** | `__global__ __aicore__ void foo_kernel(...)` | AscendC 设备端代码，在 AI Core 上执行实际计算逻辑 |
| **NPU Dispatch** | `TORCH_LIBRARY_IMPL(..., PrivateUse1, m)` | Host 端调度：分配输出 tensor、计算 tiling 参数、按 dtype 分发调用 kernel |

### 2. 编译配置：`csrc/foo/dav-2201/CMakeLists.txt`

内容固定一行：

```cmake
add_sources()
```

### 3. 测试用例：`tests/foo/test_foo.py`

- **接口测试**：验证 `torch.ops.flash.foo` 已注册可用
- **精度测试**：参数化多种 shape 和 dtype，将 NPU 计算结果与 CPU PyTorch 参考实现对比

无需修改任何已有文件 —— 构建系统会自动扫描 csrc/ 下的新算子目录。

## 环境要求

- Python >= 3.8
- PyTorch
- torch_npu
- CANN 工具链（Ascend Toolkit，含 Bisheng 编译器）

## 编译

### 全量构建：一键脚本（仅产 wheel）

```bash
bash build.sh
```

该脚本依次执行：安装构建依赖 → 清理旧构建 → 编译 wheel 包。**不再自动安装，也不跑测试。** 结束时打印 wheel 的绝对路径，以及可直接拷贝的安装与测试命令：

```bash
pip install /path/to/dist/flash_npu_kernel-1.0.0-cp38-abi3-*.whl --force-reinstall --no-deps
pytest tests/ -v
```

产物为 Python Stable ABI（兼容 Python 3.8+），内含每个算子对应的 `libflash_<op>.so`。

### 单算子/多算子增量编译

在已完整构建过一次的前提下，仅重编指定算子的 `.so`，不走 wheel 打包与安装。支持同时编译多个算子，用 `,` 分隔：

```bash
bash build.sh --ops=get_out_cache_loc
bash build.sh --ops=get_out_cache_loc,rearrange_accept_index
```

执行流程：

1. 首次调用会自动 `cmake -S . -B build` 完成一次 configure；后续调用复用 `build/` 实现真正的增量编译。
2. 只编指定的 `flash_<op>` 目标，产物直接落到 `flash_npu_kernel/libflash_<op>.so`。
3. 若 site-packages 里安装的 wheel 版本与刚编译的内容不一致（逐字节比对），脚本会打印一条可直接执行的 `cp` 命令用于替换，例如：

   ```bash
   cp /path/to/flash_ops/flash_npu_kernel/libflash_get_out_cache_loc.so "/home/you/.local/lib/python3.12/site-packages/flash_npu_kernel/"
   ```

   两份一致（或尚未安装 wheel）时脚本不输出 `cp` 提示。Python 进程重启后 so 替换生效。

查看可用算子：

```bash
bash build.sh --help
```

### 全量手动步骤（等价）

```bash
pip install -r requirements.txt
python3 setup.py clean
NPU_ARCH=dav-2201 python3 -m build --wheel --no-isolation
pip install dist/*.whl --force-reinstall --no-deps
```

### 直接用 cmake 编译单个 so（不走脚本的场景）

```bash
cmake --build build --target flash_<op_name> -j
# 产物：flash_npu_kernel/libflash_<op_name>.so，再按上一节手动替换。
```

## 使用

```python
import torch
import torch_npu
import flash_npu_kernel

# Example: in-place flatten with accept_length mask
accept_index = torch.arange(0, 4 * 16, dtype=torch.int64).reshape(4, 16).npu()
accept_length = torch.tensor([3, 7, 5, 2], dtype=torch.int64).npu()
output = torch.zeros(int(accept_length.sum().item()), dtype=torch.int64).npu()
torch.ops.flash.npu_rearrange_accept_index(accept_index, accept_length, 4, output)
```

## 运行测试

```bash
pytest tests/ -v
```
