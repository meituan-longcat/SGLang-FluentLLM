## 概述

longcat GE 入图自定义算子库，当前项目中包含以下算子：

| 算子名称 | 说明 | 文档 |
|---------|------|------|
| ComputeNGramIds | 计算 N-gram 嵌入 ID，用于推理阶段 Over-Embedding 加速 | [文档](./docs/custom-npu_compute_n_gram_ids.md) |
| GatherSelectionKvCache | 根据 TopK 索引从全量 KV Cache 中收集选中块 | - |
| LightningIndexer | Lightning Attention 稀疏索引算子 | - |
| SparseFlashAttention | Lightning Attention 稀疏注意力算子 | - |
| MlpLightningIndexer | MLP 版 Lightning Indexer，支持 TND 与 PA_BSND 布局 | - |
| MlpCompressedLightningIndexer | MLP Lightning Indexer 的压缩表示变体（双路 query_pos/neg + key_max/min） | [文档](./docs/custom-npu_mlp_compressed_lightning_indexer.md) |
| MlpSparseLightningIndexer | MLP Lightning Indexer 的两级稀疏变体（前置 sparse_index 候选 block） | [文档](./docs/custom-npu_mlp_sparse_lightning_indexer.md) |

## 目录结构说明

融合算子代码目录结构如下：

  ```
  ├── cmake                                     # 项目工程编译目录
  │   ├── modules                               # CMake Find模块
  │   ├── scripts                               # 编译准备与版本校验脚本
  │   ├── config.cmake                          # 编译配置
  │   ├── func.cmake                            # 编译函数定义
  │   ├── intf.cmake                            # 接口定义
  │   └── intf_pub.cmake                        # 公共接口定义
  ├── docs                                      # 算子使用说明和资料
  ├── src                                       # 算子的源代码
  |   ├── compute_n_gram_ids                    # ComputeNGramIds算子实现
  |   ├── gather_selection_kv_cache             # GatherSelectionKvCache算子实现
  |   ├── lightning_indexer                     # LightningIndexer算子实现
  |   ├── sparse_flash_attention                # SparseFlashAttention算子实现
  |   ├── mlp_lightning_indexer                 # MlpLightningIndexer算子实现
  |   |   ├── op_host                           # 算子信息库、Tiling、InferShape相关实现目录
  |   |   ├── op_kernel                         # 算子Kernel目录
  |   ├── mlp_compressed_lightning_indexer      # MlpCompressedLightningIndexer算子实现
  |   |   ├── op_host                           # 算子信息库、Tiling、InferShape相关实现目录
  |   |   ├── op_kernel                         # 算子Kernel目录
  |   ├── mlp_sparse_lightning_indexer          # MlpSparseLightningIndexer算子实现
  |   |   ├── op_host                           # 算子信息库、Tiling、InferShape相关实现目录
  |   |   ├── op_kernel                         # 算子Kernel目录
  |   ├── utils                                 # 公共工具库
  |
  ├── torch_ops_extension                       # PyTorch自定义算子适配层目录
  |   ├── custom_ops
  |   │   ├── __init__.py                       # 包初始化，加载.so并挂载算子到torch_npu
  |   │   ├── csrc                              # 自定义算子适配层C++代码目录
  |   │   |   ├── ops_def_registration.cpp      #   算子定义与TORCH_LIBRARY注册（所有算子集中注册）
  |   │   |   ├── ops_common.h                  #   公共头文件（ACL类型转换、NPU调用宏等）
  |   │   |   ├── ops_common.cpp                #   公共实现
  |   │   |   └── npu_<op_name>.cpp             #   各算子C++适配层实现
  |   │   └── converter                         # 自定义算子包Python侧converter代码（torchair入图）
  |   │       ├── __init__.py
  |   │       └── npu_<op_name>.py              #   各算子GE图converter
  |   ├── setup.py                              # wheel包编译文件
  |   ├── build_and_install.sh                  # 自定义算子wheel包编译与安装脚本
  |
  ├── tests                                     # 算子的测试用例代码
  ├── build.sh                                  # 项目工程编译脚本
  ├── CMakeLists.txt                            # 项目工程编译配置文件
  ├── README.md
  ├── version.info                              # 项目版本信息
  ```

昇腾社区Ascend C自定义算子开发资料：[Ascend C自定义算子开发](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha002/devguide/opdevg/ascendcopdevg/atlas_ascendc_10_0001.html)


## 环境准备<a name="环境准备"></a>

### 硬件要求

- 昇腾 Atlas A2（ascend910b）或 Atlas A3（ascend910_93）系列

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
| bisheng | CANN 配套的昇腾编译器，安装 CANN 后自带 |


### 设置环境变量

  ```bash
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
  ```

若 CANN 安装在非默认路径，请将上述路径替换为实际安装路径下的 `set_env.sh`；也可通过设置 `ASCEND_HOME_PATH` 环境变量指向 CANN 套件根目录（`build.sh` 会自动识别 `ASCEND_HOME_PATH`）。可通过 `npu-smi info` 确认设备在网，通过 `python3 -c "import torch,torch_npu;print(torch.npu.is_available())"` 确认 NPU 可用。

---

## 编译

### 自定义融合算子编译

编译所有支持的芯片类型（Atlas A2 + Atlas A3）：

  ```bash
  cd ./flash-npu-kernel/cann-flash-ops
  bash build.sh -c "ascend910b;ascend910_93"
  ```

Atlas A3 编译：

  ```bash
  cd ./flash-npu-kernel/cann-flash-ops
  bash build.sh -c ascend910_93
  ```

Atlas A2 编译：

  ```bash
  cd ./flash-npu-kernel/cann-flash-ops
  bash build.sh -c ascend910b
  ```

仅编译指定算子（加速编译调试）：

  ```bash
  bash build.sh -c ascend910_93 -n "compute_n_gram_ids"
  ```

#### build.sh 参数说明

| 参数 | 说明 | 示例 |
|------|------|------|
| `-c` / `--compute-unit` | 指定芯片型号，多个用分号分隔，默认 `ascend910_93` | `-c "ascend910b;ascend910_93"` |
| `-n` / `--op-name` | 指定编译的算子名，多个用分号分隔，默认全部 | `-n "compute_n_gram_ids"` |
| `-p` / `--package-path` | 指定 CANN 开发套件安装路径，已设置 `ASCEND_HOME_PATH` 时可省略 | `-p /usr/local/Ascend/ascend-toolkit/latest` |
| `--verbose` | 显示详细编译信息 | `--verbose` |
| `--disable-check-compatible` | 跳过 CANN 版本兼容性校验 | `--disable-check-compatible` |
| `--ccache` | 指定 ccache 路径，设为 `false` 关闭 ccache | `--ccache /usr/bin/ccache` |
| `--tiling-key` | 设置算子 tiling key 列表 | `--tiling-key "1;2;3"` |
| `-h` / `--help` | 显示帮助信息 | `-h` |

#### 编译产物

编译成功后在 `output` 目录生成自定义算子包：`CANN-flash-ops-linux.<arch>.run`。其中，\<arch>表示操作系统架构（如 `x86_64`、`aarch64`）。

  ```
  Self-extractable archive "CANN-flash-ops-linux.<arch>.run" successfully created.
  ```

#### 编译常见问题

- **版本校验拦截**：执行 `bash build.sh --disable-check-compatible` 跳过拦截。
- **bisheng 未找到**：确认已执行 `source /usr/local/Ascend/ascend-toolkit/set_env.sh`。
- **同时编译多种芯片**：使用分号分隔，如 `bash build.sh -c "ascend910b;ascend910_93"`。
- **A2 环境编译**：Atlas A2 使用 `ascend910b`，命令为 `bash build.sh -c ascend910b`。
- **非标准镜像 CANN 包**：编译时加上 `--disable-check-compatible` 配置项。

### torch适配层whl包编译与安装

> **前置条件**：需先完成上述自定义融合算子的编译和安装（run 包），并 `source` 对应的 `set_env.bash`。

编译与安装命令如下：

  ```shell
  cd ./flash-npu-kernel/cann-flash-ops/torch_ops_extension
  bash build_and_install.sh
  ```

编译成功后在 `dist` 目录生成自定义 custom-ops 算子包：`flash_ops-1.0-<python_version>-<python_version>-<arch>.whl`。其中，\<python_version>表示 python 版本号，\<arch>表示操作系统架构。

---

## 安装

### 自定义融合算子安装

安装前，需确保所安装的自定义算子包与所安装 CANN 开发套件包 CPU 架构一致，安装命令如下：

  ```bash
  cd ./flash-npu-kernel/cann-flash-ops/output
  chmod +x CANN-flash-ops-linux.<arch>.run
  ./CANN-flash-ops-linux.<arch>.run --quiet --install-path=/usr/local/Ascend/ascend-toolkit/latest
  source /usr/local/Ascend/ascend-toolkit/latest/vendors/flash_ops/bin/set_env.bash
  ```

**说明：**

- `--install-path` 参数指定 CANN 开发套件包的根目录路径（非 opp 子目录），即包含 `opp/` 目录的上一层路径。
- 执行上述命令后，自定义融合算子对应的 run 包会安装到 CANN 软件包目录：`/usr/local/Ascend/ascend-toolkit/latest/vendors/flash_ops/`。
- 本项目算子厂商名（`VENDOR_NAME`）固定为 `flash_ops`（见 `CMakeLists.txt`），因此安装后 vendor 目录为 `vendors/flash_ops`；安装脚本会在该目录下自动生成 `bin/set_env.bash`，执行 `source` 后会设置 `ASCEND_CUSTOM_OPP_PATH` 与 `LD_LIBRARY_PATH`。
- 若此前安装过其它厂商名（如 `customize`/`custom`）的同名算子，建议先卸载旧 vendor（删除 `opp/vendors/<旧厂商名>` 目录并从 `opp/vendors/config.ini` 的 `load_priority` 中移除），避免重复注册导致冲突。

#### 安装后验证

  ```bash
  ls /usr/local/Ascend/ascend-toolkit/latest/vendors/flash_ops/
  echo $ASCEND_CUSTOM_OPP_PATH
  ```

### 算子torch适配包安装

通过上述 `build_and_install.sh` 一键编译安装，或手动安装 wheel 包：

  ```bash
  pip3 install dist/*.whl -I
  ```

#### 安装后验证

  ```bash
  pip3 show flash_ops
  python3 -c "import torch, torch_npu, flash_ops; print('flash_ops loaded successfully')"
  ```

---

## 测试验证

### 前置条件

测试用例依赖已安装的自定义融合算子（run 包）与 `torch_ops_extension` 算子包（`flash_ops`）。执行前请确认：

- 已执行 [自定义融合算子安装](#自定义融合算子安装) 并 `source` 对应的 `set_env.bash`；
- 已执行 [torch适配层whl包编译与安装](#torch适配层whl包编译与安装)，`pip3 show flash_ops` 可查到；
- `python3 -c "import torch, torch_npu, flash_ops"` 无报错，且 `torch.npu.is_available()` 为 `True`。

### 测试文件与算子对应关系

| 测试文件 | 被测算子 | 测试框架 |
|---------|---------|---------|
| test_npu_compute_n_gram_ids.py | ComputeNGramIds | pytest |
| test_npu_gather_selection_kv_cache.py | GatherSelectionKvCache | unittest |
| test_npu_lightning_indexer.py | LightningIndexer | unittest |
| test_mlp_lighting_indexer.py | MlpLightningIndexer（TND 场景） | unittest |
| test_npu_mlp_compressed_lightning_indexer.py | MlpCompressedLightningIndexer | 独立脚本 |
| test_npu_mlp_sparse_lightning_indexer.py | MlpSparseLightningIndexer | argparse 子命令 |

### 运行测试

  ```shell
  cd ./flash-npu-kernel/cann-flash-ops/tests
  python3 test_npu_compute_n_gram_ids.py
  python3 test_npu_gather_selection_kv_cache.py
  python3 test_npu_lightning_indexer.py
  python3 test_mlp_lighting_indexer.py
  python3 test_npu_mlp_compressed_lightning_indexer.py
  # MlpSparseLightningIndexer 支持三种子模式
  python3 test_npu_mlp_sparse_lightning_indexer.py               # 默认单 case（PA_BSND）
  python3 test_npu_mlp_sparse_lightning_indexer.py tnd           # 非 PA TND 单 case
  python3 test_npu_mlp_sparse_lightning_indexer.py gen --num-cases 20   # 泛化随机 case
  ```

**说明：**

- 所有用例均可通过 `python3 <文件名>` 直接运行。`test_npu_compute_n_gram_ids.py` 基于 pytest 参数化用例编写，其 `__main__` 入口通过 `pytest.main([__file__])` 驱动自身，因此 `python3` 与 `pytest test_npu_compute_n_gram_ids.py` 两种方式等价、均可运行。
- 其余用例基于 `torch_npu.testing.testcase.TestCase`（unittest 风格），既可用 `python3` 直接运行，也可用 `pytest` 统一运行。
- 若需查看详细执行情况可追加 `pytest -v`；若只跑单个用例可使用 `pytest <file>::<TestClass>::<test_method>`。

### 使用 pytest 统一运行全部用例

  ```shell
  cd ./flash-npu-kernel/cann-flash-ops/tests
  pytest -v
  ```

### 测试验证标准

- 所有用例执行完毕，无 `FAIL` 或 `ERROR` 输出。
- 精度测试用例通过 CPU 参考实现与 NPU 结果的逐元素比对（`np.array_equal`），确保计算正确性。
- torchair 入图测试（如 `test_compute_n_gram_ids_torchair`）验证 `torch.compile` 场景下算子的 GE 图成图能力。

---

## 卸载

### 卸载自定义融合算子（run 包）

删除安装目录下的 vendor 文件夹：

  ```bash
  rm -rf /usr/local/Ascend/ascend-toolkit/latest/vendors/flash_ops
  ```

若 CANN 安装在非默认路径，请替换为实际路径。卸载后 `ASCEND_CUSTOM_OPP_PATH` 中对应的路径自动失效。

若需清理 `config.ini` 中的加载优先级记录：

  ```bash
  # 编辑 config.ini，从 load_priority 行中移除 flash_ops
  vi /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/config.ini
  ```

### 卸载 torch_ops_extension 算子包（wheel 包）

  ```bash
  pip3 uninstall flash_ops -y
  ```

### 清理编译产物

  ```bash
  cd ./flash-npu-kernel/cann-flash-ops
  rm -rf build output

  cd torch_ops_extension
  rm -rf build dist flash_ops.egg-info
  ```

---

## 新增算子开发说明

### 开发流程概览

新增一个算子需要完成以下工作：

1. **算子 Kernel 开发**：基于 Ascend C 编写 NPU 侧 Kernel 实现
2. **算子 Host 侧开发**：编写算子定义、InferShape、Tiling 策略
3. **PyTorch 适配层开发**：编写 C++ 适配代码，注册到 `torch.ops.custom` 命名空间
4. **torchair Converter 开发**：编写 Python 侧 GE 图 converter（支持 `torch.compile` 场景）
5. **测试用例编写**：编写精度验证和入图测试
6. **文档编写**：编写算子使用说明

### 新增算子交付件清单

以新增算子 `MyOp`（目录名 `my_op`，Python 调用名 `npu_my_op`）为例，需要创建和修改以下文件：

#### 新建文件

| 序号 | 文件路径 | 说明 |
|------|---------|------|
| 1 | `src/my_op/CMakeLists.txt` | 算子编译配置，注册源文件到编译目标 |
| 2 | `src/my_op/op_host/my_op_def.cpp` | 算子定义：输入/输出/属性的类型、数据格式、支持的芯片型号 |
| 3 | `src/my_op/op_host/my_op_proto.cpp` | InferShape（输出 shape 推导）与 InferDataType（输出 dtype 推导） |
| 4 | `src/my_op/op_host/my_op_tiling.h` | Tiling 数据结构定义（`BEGIN_TILING_DATA_DEF` / `END_TILING_DATA_DEF`） |
| 5 | `src/my_op/op_host/my_op_tiling.cpp` | Tiling 策略实现：任务切分、核数分配、参数下发 |
| 6 | `src/my_op/op_kernel/my_op.cpp` | Kernel 入口函数（`extern "C" __global__ __aicore__`） |
| 7 | `src/my_op/op_kernel/my_op.h` | Kernel 类实现：Init / Process 方法、数据搬运与计算逻辑 |
| 8 | `torch_ops_extension/custom_ops/csrc/npu_my_op.cpp` | PyTorch C++ 适配层：参数转换、调用 `EXEC_NPU_CMD_V1` 宏 |
| 9 | `torch_ops_extension/custom_ops/converter/npu_my_op.py` | torchair GE 图 converter：`@register_fx_node_ge_converter` 装饰器 |
| 10 | `tests/test_npu_my_op.py` | 测试用例：精度验证 + torchair 入图测试 |
| 11 | `docs/custom-npu_my_op.md` | 算子使用说明文档（可选但推荐） |

#### 需要修改的现有文件

| 序号 | 文件路径 | 修改内容 |
|------|---------|---------|
| 1 | `torch_ops_extension/custom_ops/csrc/ops_def_registration.cpp` | 在 `TORCH_LIBRARY(custom, m)` 中添加算子签名定义 `m.def("npu_my_op(...)")` |
| 2 | `torch_ops_extension/custom_ops/__init__.py` | 在 `from .converter import ...` 行中添加 `npu_my_op` 的导入 |
| 3 | `README.md` | 更新算子列表、目录结构说明、测试文件对应关系表 |

> **说明**：`src/` 下的算子目录会被 CMake 自动扫描（通过 `cmake/func.cmake` 中的 `op_add_subdirectory` 函数 glob `src/*/CMakeLists.txt`），因此**无需修改顶层 `CMakeLists.txt`**，只需在 `src/` 下新建算子目录并包含 `CMakeLists.txt` 即可被编译系统自动发现。

### 各交付件编写要点

#### 1. 算子 CMakeLists.txt (`src/my_op/CMakeLists.txt`)

参考现有算子（如 `src/compute_n_gram_ids/CMakeLists.txt`），需完成：

- `add_ops_compile_options`：设置算子编译选项（如 `--cce-auto-sync=on`、`-Werror` 等）
- `target_sources(op_host_aclnn ...)`：注册 `*_def.cpp` 到 aclnn 目标
- `target_sources(optiling ...)`：注册 `*_tiling.cpp` 到 tiling 目标
- `target_sources(opsproto ...)`：注册 `*_proto.cpp` 到 proto 目标
- `target_include_directories(optiling ...)`：添加 tiling 头文件搜索路径

#### 2. 算子定义 (`*_def.cpp`)

- 继承 `OpDef`，定义输入（`Input`）、输出（`Output`）、属性（`Attr`）
- 每个输入/输出需指定 `ParamType`（`REQUIRED`/`OPTIONAL`）、`DataType`、`Format`
- 通过 `this->AICore().AddConfig("ascend910b")` 和 `AddConfig("ascend910_93")` 声明支持的芯片
- 最后通过 `OP_ADD(MyOp)` 注册

#### 3. InferShape (`*_proto.cpp`)

- 实现 `InferShape` 函数：根据输入 shape 和属性推导输出 shape
- 实现 `InferDataType` 函数：推导输出数据类型
- 通过 `IMPL_OP_INFERSHAPE(MyOp)` 宏注册

#### 4. Tiling (`*_tiling.h` + `*_tiling.cpp`)

- 在 `.h` 中通过 `BEGIN_TILING_DATA_DEF` / `TILING_DATA_FIELD_DEF` / `END_TILING_DATA_DEF` 宏定义 tiling 数据结构
- 通过 `REGISTER_TILING_DATA_CLASS(MyOp, MyOpTilingData)` 注册
- 在 `.cpp` 中实现 `DoTiling()` 方法：获取平台信息、计算任务切分、设置 `blockDim`、下发 tiling 数据
- 通过 `IMPL_OP_OPTILING(MyOp)` 宏注册 tiling 入口

#### 5. Kernel (`op_kernel/*.cpp` + `*.h`)

- 入口函数签名：`extern "C" __global__ __aicore__ void my_op(GM_ADDR input1, ..., GM_ADDR tiling)`
- 通过 `GET_TILING_DATA(tiling_data, tiling)` 获取 tiling 参数
- 在 `.h` 中实现算子类：`Init()` 完成 GlobalTensor / TBuf 初始化，`Process()` 执行计算逻辑
- 使用 `TILING_KEY_IS(1)` 判断 tiling key

#### 6. PyTorch C++ 适配层 (`csrc/npu_my_op.cpp`)

- 实现算子的 C++ 函数：准备输出 tensor、调用 `EXEC_NPU_CMD_V1(aclnnMyOp, ...)` 宏
- 实现 Meta 函数（用于 `torch.compile` trace 阶段的 shape 推导）
- 通过 `TORCH_LIBRARY_IMPL(custom, PrivateUse1, m)` 注册 NPU 实现
- 通过 `TORCH_LIBRARY_IMPL(custom, Meta, m)` 注册 Meta 实现

#### 7. 算子签名注册 (`ops_def_registration.cpp`)

在 `TORCH_LIBRARY(custom, m)` 块中添加算子签名：

  ```cpp
  m.def("npu_my_op(Tensor input1, Tensor input2, int attr1) -> Tensor");
  ```

签名格式遵循 PyTorch 的 [function schema](https://pytorch.org/docs/stable/notes/autograd.html#function-schema)。

#### 8. torchair Converter (`converter/npu_my_op.py`)

- 使用 `@register_fx_node_ge_converter(torch.ops.custom.npu_my_op.default)` 装饰器
- 函数参数与 C++ 侧签名一致
- 通过 `torchair.ge.custom_op("MyOp", inputs={...}, attrs={...}, outputs=[...])` 映射到 GE 图节点

#### 9. 包初始化 (`__init__.py`)

在 `from .converter import ...` 行中添加新算子的 converter 模块名：

  ```python
  from .converter import ..., npu_my_op
  ```

#### 10. 测试用例 (`tests/test_npu_my_op.py`)

- 精度测试：编写 CPU 参考实现，与 NPU 结果逐元素比对
- 接口存在性测试：`assert hasattr(torch.ops.custom, "npu_my_op")`
- torchair 入图测试：通过 `torch.compile` + torchair backend 执行并验证结果
- 支持 `python3 test_npu_my_op.py` 和 `pytest test_npu_my_op.py` 两种运行方式

### 新增算子开发 Checklist

- [ ] `src/my_op/CMakeLists.txt` 已创建
- [ ] `src/my_op/op_host/my_op_def.cpp` 已创建，定义了输入/输出/属性和芯片型号
- [ ] `src/my_op/op_host/my_op_proto.cpp` 已创建，实现了 InferShape 和 InferDataType
- [ ] `src/my_op/op_host/my_op_tiling.h` 已创建，定义了 tiling 数据结构
- [ ] `src/my_op/op_host/my_op_tiling.cpp` 已创建，实现了 tiling 策略
- [ ] `src/my_op/op_kernel/my_op.cpp` 已创建，实现了 kernel 入口
- [ ] `src/my_op/op_kernel/my_op.h` 已创建，实现了 kernel 类
- [ ] `torch_ops_extension/custom_ops/csrc/npu_my_op.cpp` 已创建，实现了 PyTorch 适配
- [ ] `torch_ops_extension/custom_ops/csrc/ops_def_registration.cpp` 已修改，添加了算子签名
- [ ] `torch_ops_extension/custom_ops/converter/npu_my_op.py` 已创建，实现了 GE 图 converter
- [ ] `torch_ops_extension/custom_ops/__init__.py` 已修改，添加了 converter 导入
- [ ] `tests/test_npu_my_op.py` 已创建，包含精度测试和入图测试
- [ ] `docs/custom-npu_my_op.md` 已创建（可选）
- [ ] `README.md` 已更新算子列表和测试文件对应关系
- [ ] 编译通过：`bash build.sh -c "ascend910b;ascend910_93"`
- [ ] torch_ops_extension 编译通过：`bash build_and_install.sh`
- [ ] 测试通过：`python3 tests/test_npu_my_op.py`

---

## 附录

### FAQ

- **如何同时编译支持多种芯片的算子**：使用分号分隔多个芯片型号，如`bash build.sh -c "ascend910b;ascend910_93"`可同时编译支持Atlas A2和Atlas A3的算子。
- **A2环境如何编译执行ascendc算子**：ascendc算子默认基于A3环境编译，包含Atlas A3 训练系列产品/Atlas A3 推理系列产品。Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件使用`ascend910b`，编译命令为`bash build.sh -c ascend910b`。
- **非标准镜像CANN包如何编译执行ascendc算子**：建议使用镜像版本，如果要使用不配套的cann包需要在算子编译时加上`--disable-check-compatible`配置项。
- **如何只编译单个算子以加速调试**：使用 `-n` 参数指定算子名，如 `bash build.sh -c ascend910_93 -n "compute_n_gram_ids"`。
- **编译报错找不到 bisheng**：确认已执行 `source /usr/local/Ascend/ascend-toolkit/set_env.sh` 加载 CANN 环境变量。
- **运行时提示 "torch.ops.custom module is not found"**：确认已正确安装 wheel 包，且 `import flash_ops` 在 `torch.ops.custom` 调用之前执行。
- **运行时算子调用失败**：确认自定义融合算子 run 包已正确安装，且 `ASCEND_CUSTOM_OPP_PATH` 环境变量已设置。
