# ComputeNGramIds 算子说明

## 功能描述

ComputeNGramIds 算子用于计算 N-gram 的嵌入 ID。根据输入的 token 序列、权重表和 N-gram 参数，生成对应的 N-gram ID 张量，用于推理阶段的 Over-Embedding 加速。

## 算子接口

### 输入

| 参数名 | 数据类型 | 格式 | 说明 |
|--------|---------|------|------|
| oe_weights | int32 | ND | Over-embedding 权重 |
| oe_mods | int32 | ND | Over-embedding 模值 |
| exclusive_oe_embeder_size_sums | int32 | ND | 独占 OE embeder 大小前缀和 |
| tokens | int32 | ND | 输入 token 序列 |
| exclusive_req_len_sums | int32 | ND | 独占请求长度前缀和 |
| oe_token_table | int32 | ND | OE token 表 |
| row_indices | int64 | ND | 行索引 |
| column_starts | int32 | ND | 列起始位置 |

### 属性

| 参数名 | 类型 | 说明 |
|--------|------|------|
| batch_size | int | 批次大小 |
| oe_n | int | Over-embedding N 值 |
| oe_k | int | Over-embedding K 值 |
| max_context_len | int | 最大上下文长度 |
| eos_token_id | int | EOS token id，命中时（j>0）中断当前 n-gram 累加；传入负数（如 -1）可禁用该逻辑 |

### 输出

| 参数名 | 数据类型 | 格式 | 说明 |
|--------|---------|------|------|
| oe_n_gram_ids | int32 | ND | N-gram ID 张量，shape 为 [token_num, (oe_n - 1) * oe_k] |

## 支持的芯片型号

- Atlas A2: ascend910b
- Atlas A3: ascend910_93

## 测试用例

- 详见 [test_npu_compute_n_gram_ids.py](../tests/test_npu_compute_n_gram_ids.py)
