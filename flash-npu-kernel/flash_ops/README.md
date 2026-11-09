# flash_npu_kernel

基于 AscendC 的 PyTorch 自定义算子库，使用 fast kernel launch（`<<<>>>` 直调）方式实现高性能 NPU 算子，编译为 Python wheel 包供直接安装使用。

**构建模型：一算子一 so。** 每个算子独立产出 `libflash_<op>.so`，支持单算子增量编译与热替换，无需重打 wheel。

---

## 目录

- [项目结构](#项目结构)
- [已包含算子](#已包含算子)
- [环境要求](#环境要求)
- [编译](#编译)
- [安装](#安装)
- [使用](#使用)
- [测试验证](#测试验证)
- [卸载](#卸载)
- [新增算子开发说明](#新增算子开发说明)

---

## 项目结构

```
flash_ops/
├── CMakeLists.txt                  # 根 CMake 配置（项目定义、include/link 设置、添加 csrc 子目录）
├── setup.py                        # Python wheel 构建入口（自定义 cmake_build / clean / bdist_wheel）
├── requirements.txt                # 构建依赖（build, pyyaml, numpy, pytest）
├── build.sh                        # 构建脚本：默认产出 wheel；--ops=a[,b,...] 单/多算子增量构建
├── README.md                       # 本文档
├── cmake/                          # CMake 工具模块
│   ├── ascend.cmake                # 检测 Ascend 工具链路径，设置 Bisheng 编译器
│   ├── func.cmake                  # 提供 recursive_add_subdirectory / add_sources 宏
│   ├── python.cmake                # 检测 Python 解释器和开发库
│   ├── torch.cmake                 # 检测 PyTorch
│   └── torch_npu.cmake             # 检测 torch_npu
├── csrc/                           # C++ 源码目录（算子按名称组织，每个子目录一个算子）
│   ├── CMakeLists.txt              # 自动扫描子目录中的算子（recursive_add_subdirectory）
│   ├── assign_req_to_token_pool/   # 算子：assign_req_to_token_pool
│   │   ├── CMakeLists.txt          # 固定内容：add_sources()
│   │   ├── assign_req_to_token_pool.asc            # 主文件：Schema + Meta + Dispatch
│   │   ├── assign_req_to_token_pool_entry.h        # __global__ kernel 入口
│   │   ├── assign_req_to_token_pool_kernel.h       # AscendC kernel 类（设备端计算）
│   │   ├── assign_req_to_token_pool_tiling.h       # Host 端 tiling 参数计算
│   │   └── assign_req_to_token_pool_tilingdata.h   # TilingData POD 结构体
│   ├── get_out_cache_loc/          # 算子：get_out_cache_loc（同上 6 文件结构）
│   ├── rearrange_accept_index/     # 算子：rearrange_accept_index
│   └── update_oe_token_table/      # 算子：update_oe_token_table
├── flash_npu_kernel/               # Python 包
│   ├── __init__.py                 # 自动检测 NPU 架构并加载所有 libflash_*.so
│   └── libflash_*.so               # 各算子独立共享库（构建产物）
└── tests/                          # 测试目录（按算子名称组织）
    ├── assign_req_to_token_pool/test_assign_req_to_token_pool.py
    ├── get_out_cache_loc/test_get_out_cache_loc.py
    ├── rearrange_accept_index/test_rearrange_accept_index.py
    └── update_oe_token_table/test_update_oe_token_table.py
```

### 关键文件说明

| 文件 | 作用 |
|---|---|
| `CMakeLists.txt` | 根 CMake：定义项目名 `FlashNpuKernel`、版本 `1.0.0`、C++17、NPU_ARCH、EXTENSION_MODULE_NAME=`flash`，include/link 配置 |
| `cmake/func.cmake` | 核心构建宏：`recursive_add_subdirectory()` 自动扫描算子目录；`add_sources()` 为每个算子按架构生成独立 `.so` 目标 |
| `cmake/ascend.cmake` | 定位 Ascend Toolkit（优先 `$ASCEND_HOME_PATH`，其次默认路径），设置 Bisheng 编译器 |
| `setup.py` | Python 打包：自定义 `cmake_build` 命令自动检测 NPU_ARCH 并调用 cmake；`BinaryWheel` 生成平台特定 wheel |
| `build.sh` | 构建编排：全量 wheel 构建或增量单算子构建，自动检测 NPU_ARCH |
| `flash_npu_kernel/__init__.py` | 运行时加载器：检测 NPU 架构 → glob `libflash_*_<arch>.so` → `torch.ops.load_library()` |

---

## 已包含算子

| 算子 | 调用方式 | 支持 dtype | 说明 |
|---|---|---|---|
| `npu_assign_req_to_token_pool` | `torch.ops.flash.npu_assign_req_to_token_pool` | int32, int64 | In-place scatter to req_to_token_pool |
| `npu_get_out_cache_loc` | `torch.ops.flash.npu_get_out_cache_loc` | int32, int64 | In-place gather from req_to_token |
| `npu_rearrange_accept_index` | `torch.ops.flash.npu_rearrange_accept_index` | int32, int64 | In-place flatten of accept_index by accept_length |
| `npu_update_oe_token_table` | `torch.ops.flash.npu_update_oe_token_table` | int32 (tokens), int64 (row_indices) | Update OE token table |
| `npu_index_k_route_stats` | `torch.ops.flash.npu_index_k_route_stats` | int64 | Exact Index-K token/page histograms and invalid-input flag |

---

## 环境要求

### 硬件

- 华为 Ascend NPU（910B / 910_93 / 950 等）

### 软件

| 依赖 | 版本要求 | 说明 |
|---|---|---|
| Python | >= 3.8 | 构建和运行均需要 |
| PyTorch | 与 torch_npu 匹配版本 | 深度学习框架 |
| torch_npu | 与 PyTorch 匹配版本 | Ascend NPU 的 PyTorch 适配层 |
| CANN Toolkit | 含 Bisheng 编译器 | Ascend 编译工具链，提供 `bisheng` 编译器和 AscendC 运行时库 |
| cmake | >= 3.16 | 构建系统 |

### 环境变量

| 变量 | 必填 | 说明 |
|---|---|---|
| `ASCEND_HOME_PATH` | 推荐设置 | Ascend Toolkit 安装路径。未设置时自动查找 `/usr/local/Ascend/ascend-toolkit/latest`（root）或 `$HOME/Ascend/ascend-toolkit/latest`（非 root） |
| `NPU_ARCH` | 可选 | NPU 架构标识（如 `dav-2201`、`dav-3510`）。未设置时自动从硬件检测：Ascend910B/910_93 → `dav-2201`，Ascend950 → `dav-3510` |
| `FLASH_NPU_KERNEL_ARCH` | 可选 | 运行时 `.so` 加载时的架构标识（覆盖自动检测），格式为下划线形式（如 `dav_2201`） |

---

## 编译

### 全量构建：一键脚本（仅产 wheel）

```bash
bash build.sh
```

**执行流程：**

1. 安装构建依赖：`pip install -r requirements.txt`
2. 清理旧构建：`python3 setup.py clean`（删除 `build/`、`dist/`、`*.egg-info`、`*.pyc`）
3. 编译 wheel：`python3 -m build --wheel --no-isolation`
   - 内部自动调用 `cmake -S . -B build` 配置
   - 自动调用 `cmake --build build --parallel` 编译所有算子
   - 每个算子产出独立的 `libflash_<op>_<arch>.so` 到 `flash_npu_kernel/` 目录
   - 打包为平台特定 wheel（tag 格式：`cp3X-cp3X-linux_<arch>`）

**构建完成后输出：**

```
============================================================
Full build done.
============================================================
Wheel:
  /absolute/path/to/dist/flash_npu_kernel-1.0.0-cp310-cp310-linux_aarch64.whl

Install it with:
  pip install "/absolute/path/to/dist/flash_npu_kernel-1.0.0-cp310-cp310-linux_aarch64.whl" --force-reinstall --no-deps

Run tests after install (optional):
  pytest tests/ -v
============================================================
```

**产物：** `dist/flash_npu_kernel-<version>-cp3X-cp3X-linux_<arch>.whl`，内含每个算子对应的 `libflash_<op>_<arch>.so`。

### 单算子/多算子增量编译

在已完整构建过一次的前提下，仅重编指定算子的 `.so`，不走 wheel 打包与安装。

```bash
# 编译单个算子
bash build.sh --ops=get_out_cache_loc

# 编译多个算子（逗号分隔）
bash build.sh --ops=get_out_cache_loc,rearrange_accept_index

# 查看可用算子列表
bash build.sh --help
```

**执行流程：**

1. 首次调用自动执行 `cmake -S . -B build` 完成 configure；后续调用复用 `build/` 缓存，实现真正的增量编译
2. 只编指定的 `flash_<op>_<arch>` 目标，产物直接落到 `flash_npu_kernel/libflash_<op>_<arch>.so`
3. 若 site-packages 中已安装的版本与新编译的 `.so` 不一致（逐字节比对），脚本打印可直接执行的 `cp` 命令：

```bash
cp /path/to/flash_ops/flash_npu_kernel/libflash_get_out_cache_loc_dav_2201.so "/home/user/.local/lib/python3.10/site-packages/flash_npu_kernel/"
```

Python 进程重启后 `.so` 替换生效。

### 全量手动步骤（等价）

```bash
# 1. 安装构建依赖
pip install -r requirements.txt

# 2. 清理旧构建产物
python3 setup.py clean

# 3. 构建 wheel（内部自动执行 cmake configure + build）
python3 -m build --wheel --no-isolation

# 4. 安装
pip install dist/*.whl --force-reinstall --no-deps
```

也可以分步手动执行 cmake：

```bash
# 获取 PyTorch CMake 路径
Torch_DIR=$(python3 -c 'import os,torch;print(os.path.join(torch.utils.cmake_prefix_path,"Torch"))')
# 获取 torch_npu 路径
TORCH_NPU_PATH=$(python3 -c 'import os,torch_npu;print(os.path.dirname(torch_npu.__file__))')

# cmake 配置
cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DTorch_DIR="$Torch_DIR" \
    -DTORCH_NPU_PATH="$TORCH_NPU_PATH" \
    -DNPU_ARCH="dav-2201"

# cmake 编译全部算子
cmake --build build --parallel $(nproc)
```

### 直接用 cmake 编译单个 so（不走脚本的场景）

```bash
# 前提：已执行过 cmake -S . -B build 配置
cmake --build build --target flash_<op_name>_<arch_safe> -j

# 示例：编译 get_out_cache_loc（dav-2201 架构）
cmake --build build --target flash_get_out_cache_loc_dav_2201 -j

# 产物：flash_npu_kernel/libflash_get_out_cache_loc_dav_2201.so
# 需要手动替换到 site-packages 中（参见"安装"章节）
```

其中 `<arch_safe>` 是将架构名中的 `-` 替换为 `_`（如 `dav-2201` → `dav_2201`）。

### 多架构交叉编译

同时为多个 NPU 架构编译：

```bash
# 通过环境变量指定（逗号分隔）
NPU_ARCH=dav-2201,dav-3510 bash build.sh

# 或通过 cmake 指定（分号分隔）
cmake -S . -B build -DNPU_ARCH="dav-2201;dav-3510" ...
```

每个算子会为每个架构生成独立的 `.so`，如：
- `libflash_get_out_cache_loc_dav_2201.so`
- `libflash_get_out_cache_loc_dav_3510.so`

运行时 `__init__.py` 会根据当前硬件自动选择对应架构的 `.so` 加载。

---

## 安装

### wheel 安装（推荐）

```bash
pip install dist/flash_npu_kernel-1.0.0-cp3*-cp3*-linux_*.whl --force-reinstall --no-deps
```

**验证安装：**

```bash
# 检查包是否安装
pip show flash_npu_kernel

# 验证算子可加载
python3 -c "
import torch
import torch_npu
import flash_npu_kernel
ops = ['npu_assign_req_to_token_pool', 'npu_get_out_cache_loc', 'npu_rearrange_accept_index', 'npu_update_oe_token_table']
print('已注册算子:', [op for op in ops if hasattr(torch.ops.flash, op)])
"
```

预期输出：

```
已注册算子: ['npu_assign_req_to_token_pool', 'npu_get_out_cache_loc', 'npu_rearrange_accept_index', 'npu_update_oe_token_table']
```

### 手动 so 替换（热替换）

在增量编译后，可仅替换变更的 `.so` 文件，无需重新安装 wheel：

```bash
# 1. 查找已安装位置
python3 -c "import flash_npu_kernel; import os; print(os.path.dirname(flash_npu_kernel.__file__))"

# 2. 复制新编译的 .so 到安装目录
cp flash_npu_kernel/libflash_<op>_<arch>.so <安装目录>/

# 3. 重启 Python 进程使替换生效
```

---

## 使用

```python
import torch
import torch_npu
import flash_npu_kernel  # import 时自动加载所有 libflash_*.so

# 示例：rearrange_accept_index — in-place flatten with accept_length mask
accept_index = torch.arange(0, 4 * 16, dtype=torch.int64).reshape(4, 16).npu()
accept_length = torch.tensor([3, 7, 5, 2], dtype=torch.int64).npu()
output = torch.zeros(int(accept_length.sum().item()), dtype=torch.int64).npu()
torch.ops.flash.npu_rearrange_accept_index(accept_index, accept_length, 4, output)
print(output)

# 示例：get_out_cache_loc — in-place gather
bs = 4
pool_len = 64
req_to_token = torch.arange(1, bs * pool_len + 1, dtype=torch.int64).reshape(bs, pool_len).npu()
req_pool_indices = torch.tensor([0, 1, 2, 3], dtype=torch.int64).npu()
new_compute_lens = torch.tensor([10, 20, 15, 8], dtype=torch.int64).npu()
cache_lens = torch.tensor([0, 0, 0, 0], dtype=torch.int64).npu()
out_cache_loc = torch.zeros(53, dtype=torch.int64).npu()
torch.ops.flash.npu_get_out_cache_loc(req_to_token, req_pool_indices, new_compute_lens, cache_lens, out_cache_loc, bs)
print(out_cache_loc)
```

---

## 测试验证

### 运行全部测试

```bash
pytest tests/ -v
```

### 运行单个算子测试

```bash
# 测试单个算子
pytest tests/get_out_cache_loc/ -v

# 测试特定参数组合
pytest tests/get_out_cache_loc/test_get_out_cache_loc.py -v -k "bs1 and pool_len64 and int64"
```

### 测试内容说明

每个算子的测试文件包含以下测试项：

| 测试项 | 说明 |
|---|---|
| **接口测试** `test_<op>_interface_exist` 或 `test_npu_<op>_interface_exist` | 验证算子已通过 `torch.ops.flash` 注册可用 |
| **精度测试** `test_<op>` | 参数化多种 (batch_size, shape, dtype) 组合，将 NPU 计算结果与 CPU PyTorch 参考实现逐元素对比（`torch.equal`） |

**测试流程：**

1. 在 CPU 上构造随机输入数据
2. 用 CPU PyTorch 参考实现计算期望结果
3. 将输入拷贝到 NPU，调用 `torch.ops.flash.npu_<op>` 计算
4. `torch.npu.synchronize()` 同步后，将结果拷回 CPU
5. `torch.equal(actual, expected)` 严格比对

**跳过机制：** 若当前 NPU 架构下算子未注册（如 `.so` 未安装），整个测试模块自动跳过。

---

## 卸载

```bash
pip uninstall flash_npu_kernel -y
```

**验证卸载：**

```bash
pip show flash_npu_kernel  # 应输出 WARNING: Package(s) not found
```

**清理构建产物（可选）：**

```bash
# 清理 build/、dist/、*.egg-info、*.pyc
python3 setup.py clean

# 或手动清理
rm -rf build/ dist/ flash_npu_kernel.egg-info/
rm -f flash_npu_kernel/libflash_*.so
```

---

## 新增算子开发说明

### 交付件清单

新增一个算子需要提供以下交付件（以算子名 `foo` 为例）：

| # | 交付件 | 路径 | 说明 |
|---|---|---|---|
| 1 | 算子主文件 | `csrc/foo/foo.asc` | Schema 注册 + 输入校验 + Meta 函数 + NPU Dispatch |
| 2 | Kernel 入口 | `csrc/foo/foo_entry.h` | `__global__` 函数，连接 dispatch 与 kernel 类 |
| 3 | Kernel 实现 | `csrc/foo/foo_kernel.h` | AscendC 设备端计算类（Init + Process） |
| 4 | Tiling 计算 | `csrc/foo/foo_tiling.h` | Host 端 `calc_tiling_data()` 函数 |
| 5 | TilingData 结构体 | `csrc/foo/foo_tilingdata.h` | Host/Device 共享的 POD 结构体 |
| 6 | CMake 配置 | `csrc/foo/CMakeLists.txt` | 固定一行 `add_sources()` |
| 7 | 测试用例 | `tests/foo/test_foo.py` | 接口测试 + 精度测试 |

### 需要修改的文件

**无需修改任何已有文件。** 构建系统通过 `recursive_add_subdirectory()` 自动扫描 `csrc/` 下的新算子目录；`__init__.py` 运行时自动 glob 加载所有 `libflash_*.so`。

仅需在 `csrc/` 下新建算子目录和文件，在 `tests/` 下新建测试目录和文件。

### 命名规范

以算子名 `foo_bar`（snake_case）为例：

| 元素 | 命名规则 | 示例 |
|---|---|---|
| 算子目录 | `snake_case` | `csrc/foo_bar/` |
| 主文件 | `<op_name>.asc` | `foo_bar.asc` |
| C++ 命名空间 | `flash::<CamelCase>` | `flash::FooBar` |
| PyTorch 算子名 | `npu_<op_name>` | `torch.ops.flash.npu_foo_bar` |
| TilingData 结构体 | `<CamelCase>TilingData` | `FooBarTilingData` |
| Kernel 类名 | `<CamelCase>` | `FooBar` |
| Entry 函数 | `<op_name>` (模板函数) | `foo_bar<T>` |
| CMake 目标 | `flash_<op>_<arch_safe>` | `flash_foo_bar_dav_2201` |
| 共享库 | `libflash_<op>_<arch_safe>.so` | `libflash_foo_bar_dav_2201.so` |
| OpCommand API 名 | `<CamelCase>` | `"FooBar"` |
| 测试目录 | `tests/<op_name>/` | `tests/foo_bar/` |
| 测试文件 | `test_<op_name>.py` | `test_foo_bar.py` |

### 开发步骤（以算子 `foo` 为例）

#### 步骤 1：创建算子目录和文件

```bash
mkdir -p csrc/foo
mkdir -p tests/foo
```

#### 步骤 2：编写 TilingData 结构体 — `csrc/foo/foo_tilingdata.h`

定义 Host 和 Device 共享的 POD 结构体，包含 tiling 所需的所有参数：

```cpp
#ifndef FOO_TILINGDATA_H
#define FOO_TILINGDATA_H

#include <cstdint>

namespace flash {
namespace Foo {

struct FooTilingData {
    uint32_t usedCoreNum;
    uint32_t totalLength;
    // 根据算子需要添加其他 tiling 参数
};

} // namespace Foo
} // namespace flash

#endif // FOO_TILINGDATA_H
```

#### 步骤 3：编写 Tiling 计算 — `csrc/foo/foo_tiling.h`

Host 端函数，根据输入 shape 和硬件信息计算 tiling 参数：

```cpp
#ifndef FOO_TILING_H
#define FOO_TILING_H

#include <cstdint>
#include <torch/all.h>
#include "platform/platform_ascendc.h"
#include "foo_tilingdata.h"

namespace flash {
namespace Foo {

inline FooTilingData calc_tiling_data(int64_t totalLength)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    uint64_t ubSize;
    ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    int64_t coreNum = ascendcPlatform->GetCoreNumAiv();

    // 根据算子逻辑计算 tiling 参数
    uint32_t usedCoreNum = static_cast<uint32_t>(coreNum);

    return FooTilingData{
        usedCoreNum,
        static_cast<uint32_t>(totalLength),
    };
}

} // namespace Foo
} // namespace flash

#endif // FOO_TILING_H
```

#### 步骤 4：编写 Kernel 类 — `csrc/foo/foo_kernel.h`

AscendC 设备端计算类，包含 `Init()` 和 `Process()` 方法：

```cpp
#ifndef FOO_KERNEL_H
#define FOO_KERNEL_H

#include "kernel_operator.h"
#include "foo_tilingdata.h"

namespace flash {
namespace Foo {

using namespace AscendC;

template <typename T>
class Foo {
public:
    __aicore__ inline Foo() {}
    __aicore__ inline Foo(TPipe *pipe, const FooTilingData *tiling)
        : pipe_(pipe), tl_(tiling) {}

    __aicore__ inline void Init(GM_ADDR input, GM_ADDR output);
    __aicore__ inline void Process();

private:
    TPipe *pipe_ = nullptr;
    const FooTilingData *tl_;
    GlobalTensor<T> inputGm_;
    GlobalTensor<T> outputGm_;
};

template <typename T>
__aicore__ inline void Foo<T>::Init(GM_ADDR input, GM_ADDR output)
{
    inputGm_.SetGlobalBuffer((__gm__ T*)input);
    outputGm_.SetGlobalBuffer((__gm__ T*)output);
    // 初始化 pipe buffer 等
}

template <typename T>
__aicore__ inline void Foo<T>::Process()
{
    // 设备端计算逻辑
}

} // namespace Foo
} // namespace flash

#endif // FOO_KERNEL_H
```

#### 步骤 5：编写 Kernel 入口 — `csrc/foo/foo_entry.h`

`__global__` 函数，作为 host dispatch 与 kernel 类之间的桥梁：

```cpp
#ifndef FOO_ENTRY_H
#define FOO_ENTRY_H

#include "kernel_operator.h"
#include "foo_tilingdata.h"
#include "foo_kernel.h"

namespace flash {
namespace Foo {

template <typename T>
__global__ __aicore__ __vector__ void foo(
    GM_ADDR input, GM_ADDR output,
    FooTilingData tilingData)
{
    AscendC::TPipe pipe;
    Foo<T> op(&pipe, &tilingData);
    op.Init(input, output);
    op.Process();
}

} // namespace Foo
} // namespace flash

#endif // FOO_ENTRY_H
```

#### 步骤 6：编写算子主文件 — `csrc/foo/foo.asc`

包含 4 个必要部分：

```cpp
#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"

#include "foo_tiling.h"
#include "foo_entry.h"

namespace flash {
namespace Foo {

// ========== 第 1 部分：Schema 注册 ==========
// 向 PyTorch 声明算子签名（名称、输入输出类型）
// Tensor(a!) 表示 in-place 修改该 tensor 并返回
TORCH_LIBRARY_FRAGMENT(EXTENSION_MODULE_NAME, m)
{
    m.def("npu_foo(Tensor input, Tensor(a!) output, int total_length) -> Tensor(a!)");
}

// ========== 第 2 部分：输入校验 ==========
static void check_inputs(const torch::Tensor &input,
                         const torch::Tensor &output,
                         int64_t total_length)
{
    TORCH_CHECK(total_length > 0, "total_length must be positive, got ", total_length);
    TORCH_CHECK(input.dim() == 1, "input must be 1D, got dim=", input.dim());
    TORCH_CHECK(output.dim() == 1, "output must be 1D, got dim=", output.dim());

    auto dtype = input.scalar_type();
    TORCH_CHECK(dtype == torch::kInt64 || dtype == torch::kInt32,
        "Only int64 and int32 are supported, got ", dtype);
    TORCH_CHECK(output.scalar_type() == dtype,
        "output dtype must match input dtype.");
}

// ========== 第 3 部分：Meta 函数 ==========
// 仅做 shape/dtype 推断，不执行计算；支撑 torch.compile
// in-place 算子直接返回被修改的 output tensor
torch::Tensor foo_meta(const torch::Tensor & /*input*/,
                       torch::Tensor &output,
                       int64_t /*total_length*/)
{
    return output;
}

TORCH_LIBRARY_IMPL(EXTENSION_MODULE_NAME, Meta, m)
{
    m.impl("npu_foo", foo_meta);
}

// ========== 第 4 部分：NPU Dispatch ==========
// Host 端调度：校验输入 → 计算 tiling → 按 dtype 分发调用 kernel
torch::Tensor foo_npu(const torch::Tensor &input,
                       torch::Tensor &output,
                       int64_t total_length)
{
    check_inputs(input, output, total_length);

    const c10::OptionalDeviceGuard guard(input.device());
    auto stream = c10_npu::getCurrentNPUStream().stream(false);

    auto tl = calc_tiling_data(total_length);

    auto input_ptr = (GM_ADDR)input.data_ptr();
    auto output_ptr = (GM_ADDR)output.data_ptr();

    auto acl_call = [=]() -> int {
        AT_DISPATCH_SWITCH(
            input.scalar_type(), "foo_npu",
            AT_DISPATCH_CASE(torch::kInt64, [&] {
                foo<int64_t><<<tl.usedCoreNum, nullptr, stream>>>(
                    input_ptr, output_ptr, tl);
            })
            AT_DISPATCH_CASE(torch::kInt32, [&] {
                foo<int32_t><<<tl.usedCoreNum, nullptr, stream>>>(
                    input_ptr, output_ptr, tl);
            })
        );
        return 0;
    };
    at_npu::native::OpCommand::RunOpApi("Foo", acl_call);
    return output;
}

// 注册 NPU 实现（PrivateUse1 为 Ascend NPU 的 dispatch key）
TORCH_LIBRARY_IMPL(EXTENSION_MODULE_NAME, PrivateUse1, m)
{
    m.impl("npu_foo", foo_npu);
}

} // namespace Foo
} // namespace flash
```

#### 步骤 7：编写 CMake 配置 — `csrc/foo/CMakeLists.txt`

内容固定一行：

```cmake
add_sources()
```

`add_sources()` 宏会自动：
- 从目录名推导算子名（`OP_NAME = foo`）
- Glob `*.asc` 源文件
- 为每个 NPU 架构创建 `flash_foo_<arch_safe>` 目标
- 编译产出 `libflash_foo_<arch_safe>.so` 到 `flash_npu_kernel/` 目录

#### 步骤 8：编写测试用例 — `tests/foo/test_foo.py`

```python
#!/usr/bin/env python3
import torch
import torch_npu
import flash_npu_kernel
import pytest

if not hasattr(torch.ops.flash, "npu_foo"):
    pytest.skip(
        "flash.npu_foo not registered for current NPU_ARCH; skipping module",
        allow_module_level=True,
    )


def test_foo_interface_exist():
    assert hasattr(torch.ops.flash, "npu_foo"), \
        "The 'npu_foo' operator is not registered in 'torch.ops.flash'."


def _ref_foo(input_tensor, total_length):
    """CPU 参考实现（根据算子逻辑编写）"""
    output = torch.zeros_like(input_tensor)
    # ... 参考计算逻辑 ...
    return output


DTYPES = [torch.int64, torch.int32]
SIZES = [16, 64, 256, 1024, 4096]


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("size", SIZES)
def test_foo(size, dtype):
    input_tensor = torch.randint(0, 100, (size,), dtype=dtype)
    expected = _ref_foo(input_tensor.clone(), size)

    input_npu = input_tensor.clone().npu()
    output_npu = torch.zeros(size, dtype=dtype).npu()
    torch.ops.flash.npu_foo(input_npu, output_npu, size)
    torch.npu.synchronize()
    actual = output_npu.cpu()

    assert torch.equal(actual, expected), (
        f"Mismatch for size={size}, dtype={dtype}: "
        f"diff_count={(actual != expected).sum().item()}"
    )
```

#### 步骤 9：编译验证

```bash
# 增量编译新算子
bash build.sh --ops=foo

# 若已安装 wheel，按提示替换 .so
# cp flash_npu_kernel/libflash_foo_dav_2201.so "<site-packages>/flash_npu_kernel/"

# 运行测试
pytest tests/foo/ -v
```

#### 步骤 10：全量构建（可选）

```bash
# 重新构建 wheel（包含新算子）
bash build.sh

# 安装并测试
pip install dist/flash_npu_kernel-1.0.0-cp3*.whl --force-reinstall --no-deps
pytest tests/ -v
```

### `.asc` 文件 4 部分详解

每个算子的 `.asc` 主文件必须包含以下 4 个部分：

| 部分 | 宏 / 关键代码 | 作用 | 注意事项 |
|------|--------------|------|----------|
| **Schema 注册** | `TORCH_LIBRARY_FRAGMENT(EXTENSION_MODULE_NAME, m)` | 向 PyTorch 声明算子签名（名称、输入输出类型），使其可通过 `torch.ops.flash.npu_foo` 调用 | `EXTENSION_MODULE_NAME` 编译时宏，值为 `flash`；in-place 参数用 `Tensor(a!)` 标注 |
| **Meta 函数** | `TORCH_LIBRARY_IMPL(..., Meta, m)` | 推断输出 tensor 的 shape 和 dtype，不执行实际计算；支撑 `torch.compile` 和 AutoGrad | in-place 算子直接返回被修改的 output tensor；不做输入校验 |
| **Kernel 实现** | `#include "foo_entry.h"` | AscendC 设备端代码，在 AI Core 上执行实际计算逻辑 | 通过 `<<<coreNum, nullptr, stream>>>` 语法进行 fast kernel launch |
| **NPU Dispatch** | `TORCH_LIBRARY_IMPL(..., PrivateUse1, m)` | Host 端调度：校验输入、计算 tiling 参数、按 dtype 分发调用 kernel | `PrivateUse1` 为 Ascend NPU 的 dispatch key；通过 `OpCommand::RunOpApi` 启动 |

### 开发检查清单

- [ ] `csrc/foo/` 目录下包含 6 个文件：`.asc`、`_entry.h`、`_kernel.h`、`_tiling.h`、`_tilingdata.h`、`CMakeLists.txt`
- [ ] `.asc` 文件包含 4 个注册部分：Schema、Meta、Kernel include、NPU Dispatch
- [ ] C++ 命名空间为 `flash::<CamelCase>`
- [ ] PyTorch 算子名为 `npu_<snake_case>`
- [ ] `OpCommand::RunOpApi` 传入的名称与 C++ 命名空间中的类名一致（CamelCase）
- [ ] `AT_DISPATCH_SWITCH` 覆盖了所有支持的 dtype
- [ ] `calc_tiling_data()` 使用 `PlatformAscendCManager` 获取硬件信息
- [ ] `TilingData` 结构体为纯 POD（仅含 `uint32_t` 等基本类型）
- [ ] Entry 函数使用 `__global__ __aicore__ __vector__` 修饰
- [ ] Kernel 类包含 `Init()` 和 `Process()` 方法
- [ ] `tests/foo/test_foo.py` 包含接口测试和精度测试
- [ ] 测试覆盖多种 (shape, dtype) 组合
- [ ] `csrc/foo/CMakeLists.txt` 内容为 `add_sources()`
- [ ] 增量编译通过：`bash build.sh --ops=foo`
- [ ] 全部测试通过：`pytest tests/ -v`
