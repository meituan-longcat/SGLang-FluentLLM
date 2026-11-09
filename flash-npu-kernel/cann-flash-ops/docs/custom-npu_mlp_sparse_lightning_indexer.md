# MlpSparseLightningIndexer 算子说明

## 功能描述

MlpSparseLightningIndexer 是 MLP Lightning Indexer 的"两级稀疏"变体：在原本对全长 KV 做 top-k 选择的基础上，先由前置管线给出候选 block 集合 `sparse_index`，本算子只在候选 block 内做 QK 匹配与 top-k 排序，从而将计算量从 O(kv_len) 降到 O(sparse_index_block_num × block_size)。

算子内部流程：
1. 按 `sparse_index[t]` 从 KV 中 gather 出候选 block（sparse_index 元素为 -1 时跳过）
2. 做 QK matmul → ReLU → weight 加权 → 沿 head 维 reduce，得到每个候选 token 的融合分数
3. 依据 `init_num` 与 `local_num` 分别把 sink window 与 local window 中的 token 强制赋高分（保证被选中）
4. 应用 causal mask（`sparse_mode=3`）
5. 硬件 sort 出 top-`sparse_count`，输出 sparse_indices 与可选的 sparse_values

配合前置的 block 选择器（例如 Lightning Indexer 一级筛选），可实现两级 KV 稀疏 attention。

## 算子接口

### 输入

| 参数名 | 数据类型 | 格式 | 是否必需 | 说明 |
|--------|---------|------|---------|------|
| query | bfloat16 / float16 | ND | 是 | query，BSND 或 TND 布局 |
| key | bfloat16 / float16 | ND | 是 | key，BSND / TND / PA_BSND 布局 |
| weights | float32 | ND | 是 | query 各 head 的融合权重 |
| cur_seq_lengths_query | int64 | ND | 否 | query 各 batch 的累积序列长度前缀和（TND 布局必需） |
| cur_seq_lengths_key | int64 | ND | 否 | key 各 batch 的累积序列长度前缀和（TND 布局必需） |
| block_table | int32 | ND | 否 | PA 模式下的物理 block 映射表，shape 为 [batch, max_block_num_per_batch] |
| sparse_index | int32 | ND | 否 | 每个 query token 的候选 block 编号，shape 为 [T1, sparse_block_num]；元素为 -1 表示无效 |

### 属性

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| layout_query | string | "BSND" | query 布局，可选 "BSND" / "TND" |
| layout_key | string | "PA_BSND" | key 布局，可选 "BSND" / "TND" / "PA_BSND" |
| sparse_count | int | 2048 | top-k 稀疏输出数量，**实测支持范围为 (0, 8192]**；超出范围会在 tiling 阶段报错（561002） |
| init_num | int | 0 | sink window 长度（起始固定窗口） |
| local_num | int | 0 | local window 长度（末尾滑窗） |
| sparse_mode | int | 3 | 稀疏 mask 模式，**仅支持** `0`（不加 mask）或 `3`（right-down causal） |
| pre_tokens | int | INT64_MAX | 保留字段 |
| next_tokens | int | INT64_MAX | 保留字段 |
| return_value | bool | false | 是否同时返回 sparse_values |
| sparse_index_block_size | int | 128 | sparse_index 中每个 block 覆盖的 token 数（非 PA 布局必需），**必须为 16 的倍数且属于 (0, 1024]** |

### 输出

| 参数名 | 数据类型 | 格式 | 说明 |
|--------|---------|------|------|
| sparse_indices | int32 | ND | 每个 query token 选中的 KV token 位置，shape 为 BSND 布局下 [B, S1, N2, sparse_count]，TND 布局下 [T, N2, sparse_count]；不足位置填 -1 |
| sparse_values | float32 | ND | 选中位置对应的融合分数，形状同 sparse_indices（return_value=false 时为空 tensor） |

## 约束说明

- `D`（head_dim）当前**固定为 128**。
- `Nkv`（key 头数）**固定为 1**。
- `Nq`（query 头数）为 query 的 `N` 维；为保证 run-to-run 确定性，**必须为 16 的倍数**。非 16 倍数 `Nq` 会在 tiling 阶段提前报错（561002）。
- `sparse_index_block_size` 当前**必须为 16 的倍数且属于 `(0, 1024]`**；非法值会在 tiling 阶段提前报错（561002）。
- `sparse_count` 当前**实测支持范围为 `(0, 8192]`**；超出范围会在 tiling 阶段提前报错（561002）。
- `sparse_mode` 当前**仅支持** `0`（不加 mask）或 `3`（right-down causal）。
- `BSND` 场景下，`cur_seq_lengths_query` / `cur_seq_lengths_key` 需传 **per-batch** 形态（dim0=B，例如单 batch 传 `[len]`），不能传前缀和 `[0, len]`。
- `sparse_index` 的 shape 必须是 **2-D** `[T1, block_topk]`；即使 query/key 采用 `BSND` 布局，也不能传 3-D batch 化的 `sparse_index`。
- 其余非法参数会在 tiling 阶段提前报错（561002）。

## 支持的芯片型号

- Atlas A2: ascend910b
- Atlas A3: ascend910_93

## 测试用例

- 详见 [test_npu_mlp_sparse_lightning_indexer.py](../tests/test_npu_mlp_sparse_lightning_indexer.py)
- 支持三种运行模式：
  - `python3 test_npu_mlp_sparse_lightning_indexer.py`：默认单 case，B=1, T1=2048, T2=32768, N=32, INIT=16, LOCAL=1024, SC=2048, BT=256，TND+PA_BSND 布局
  - `python3 test_npu_mlp_sparse_lightning_indexer.py tnd`：非 PA 布局单 case，B=1, T1=2, T2=65536，TND+TND 布局
  - `python3 test_npu_mlp_sparse_lightning_indexer.py gen --num-cases 20`：泛化 20+ 随机 case
- 比对逻辑：`compare_indices_with_value_tolerance` 按 rank 逐点比 index，不一致时看对应 value 相对差是否可接受

### 已知测试局限

CPU golden 使用 `WINDOW_VALUE=1111111` 作为 sink/local window 的强赋值，远超 bf16/fp16 表示范围（bf16 min ~1.18e-38，fp16 max = 65504）；NPU kernel 使用 `SECOND_MAX_LOGITS≈1.7e+38` 并在输出转 bf16 时会衰减为不同表示。加上 CPU 用 `torch.sort(stable=True)` 保序、NPU 用硬件 `Sort32`/`MrgSort` 排序，同分并列位置的 tie-breaking 顺序不同。这会导致 tolerance 逻辑在 sink/local 边界（例如 rank 16 = local window 起点）判为 unacceptable，但语义上 npu_out 与 cpu_out 在这些位置对应的 token 都在正确的高分平台上。
