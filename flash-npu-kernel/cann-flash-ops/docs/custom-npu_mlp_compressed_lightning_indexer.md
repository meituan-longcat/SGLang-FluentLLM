# MlpCompressedLightningIndexer 算子说明

## 功能描述

MlpCompressedLightningIndexer 是 MLP Lightning Indexer 的"压缩表示"变体，用于在超长上下文推理中做 top-k 稀疏 KV 选择的加速。

区别于标准 MlpLightningIndexer 单路 query + 单路 key 的输入，本算子引入两路对偶输入：
- `query_pos` / `query_neg`：query 侧的正负分量
- `key_max` / `key_min`：key 侧的上下界
两路输入的 QK 匹配分被 vector 侧融合成单一 logits，然后做 sink/local window 处理与 top-k 排序，最终产出 sparse_indices（top-k 选中的 KV 位置）和可选的 sparse_values（对应分数）。

配合前置的 KV 压缩管线，可在训练与推理阶段减少 attention 需要访问的 KV 数量。

## 算子接口

### 输入

| 参数名 | 数据类型 | 格式 | 是否必需 | 说明 |
|--------|---------|------|---------|------|
| query_pos | bfloat16 / float16 | ND | 是 | query 正分量，BSND 或 TND 布局 |
| query_neg | bfloat16 / float16 | ND | 是 | query 负分量，形状与 query_pos 一致 |
| key_max | bfloat16 / float16 | ND | 是 | 压缩后 key 的上界，BSND / TND / PA_BSND 布局 |
| key_min | bfloat16 / float16 | ND | 是 | 压缩后 key 的下界，形状与 key_max 一致 |
| weights | float32 | ND | 是 | query 各 head 的融合权重 |
| cur_seq_lengths_query | int64 | ND | 否 | query 各 batch 的累积序列长度前缀和（TND 布局必需） |
| cur_seq_lengths_key | int64 | ND | 否 | key 各 batch 的累积序列长度前缀和（TND 布局必需） |
| block_table | int32 | ND | 否 | PA（Page Attention）模式下的物理 block 映射表，shape 为 [batch, max_block_num_per_batch] |

### 属性

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| layout_query | string | "BSND" | query 布局，可选 "BSND" / "TND" |
| layout_key | string | "PA_BSND" | key 布局，可选 "BSND" / "TND" / "PA_BSND" |
| sparse_count | int | 2048 | top-k 稀疏输出数量，**实测支持范围为 (0, 8192]**；超出范围会在 tiling 阶段报错（561002） |
| block_len | int | 1 | KV 侧的 block 长度，**仅支持** {1, 2, 4, 8, 16, 32} |
| q_block_len | int | 1 | query 侧的 block 长度，**仅支持 1** |
| init_num | int | 0 | sink window 长度（起始固定窗口），需满足 `0 <= init_num <= 128` 且 `init_num % block_len == 0` |
| local_num | int | 0 | local window 长度（末尾滑窗），需满足 `0 <= local_num <= 2048` 且 `local_num % block_len == 0` |
| sparse_mode | int | 3 | 稀疏 mask 模式，**仅支持** `0`（不加 mask）或 `3`（right-down causal） |
| pre_tokens | int | INT64_MAX | 保留字段 |
| next_tokens | int | INT64_MAX | 保留字段 |
| return_value | bool | false | 是否同时返回 sparse_values（top-k 对应的 QK 分数） |

### 输出

| 参数名 | 数据类型 | 格式 | 说明 |
|--------|---------|------|------|
| sparse_indices | int32 | ND | 每个 query token 选中的 KV token 位置，shape 为 BSND 布局下 [B, S1, N2, sparse_count]，TND 布局下 [T, N2, sparse_count] |
| sparse_values | float32 | ND | 选中位置对应的融合分数，形状同 sparse_indices（return_value=false 时为空 tensor） |

## 约束说明

- `D`（head_dim）当前**固定为 128**。
- `Nkv`（key 头数）**固定为 1**。
- `Nq`（query 头数）为 query 的 `N` 维；为保证 run-to-run 确定性，**必须为偶数**。奇数 `Nq` 会在 tiling 阶段提前报错（561002）。
- 非 PA 场景下，`layout_query` 与 `layout_key` **必须相同**（即 `TND/TND` 或 `BSND/BSND`）。
- `BSND` 场景下，`cur_seq_lengths_query` / `cur_seq_lengths_key` 需传 **per-batch** 形态（dim0=B，例如单 batch 传 `[len]`），不能传前缀和 `[0, len]`。
- `sparse_count` 当前**实测支持范围为 `(0, 8192]`**；超出范围会在 tiling 阶段提前报错（561002），避免进入 kernel 后异常。
- `block_len`、`q_block_len`、`init_num`、`local_num` 的支持范围以属性表为准；非法值会在 tiling 阶段提前报错（561002）。

## 支持的芯片型号

- Atlas A2: ascend910b
- Atlas A3: ascend910_93

## 测试用例

- 详见 [test_npu_mlp_compressed_lightning_indexer.py](../tests/test_npu_mlp_compressed_lightning_indexer.py)
- 默认 case：`test_tnd_lightning_indexer_eager(b=1, t1=32768+888, head=32, block_len=8, init_num=8, local_num=72, q_block_len=1, sparse_count=256)`，layout_query="TND" / layout_key="TND"
- 使用 `print_symmetric_difference` 逐 token 比较 npu 与 cpu golden 的 indices 集合差异
