# MlpDflashInferAttn 接口说明

DFlash 推理 PageAttention 算子，Atlas A2/A3（ascend910b/ascend910_93）。
复用 serving 框架的 PageAttention KV cache 与 block_table，不做 KV gather/重建；
可见性用「逻辑区间枚举 + block_table 页翻译」实现，不物化任何 mask——
被跳过的 context 区间不产生 tile、不读 GM、不参与 QK/PV 计算。
等价于 FusedInferAttentionScore `sparse_mode=15` 的专用算子版（语义固定，无需传 sparse_mode）。

---

## 1、功能说明

DFlash 稀疏 attention 推理。输入 layout 固定 **TND**：query 为 `[T, N, D]`（T = 各 request
draft 长度之和），K/V 为 PageAttention 物理页缓存。每个 request 一个 draft block：
该 request 的 anchor 为 `context_end = kv_len - q_len`——前 `context_end` 个逻辑 token 是
已接受的 context，最后 `q_len` 个是当前 draft 块。

可见逻辑 KV 集合为两个区间的并集：

- **prefix sink 段** `[0, sink_end)`：序列头固定保留，`sink_end = next_tokens`
  （不超过 context_end）；`next_tokens = 0` 时该段为空。
- **尾带 window 段** `[tail_start, kv_len)`：`tail_start = context_end - pre_tokens`
  （不小于 0）；`pre_tokens = INT32_MAX` 时 `tail_start = 0`，即整段 context 可见。
  尾带右端到 `kv_len`，draft 块天然包含在内，且**双向可见**（每行 query 看整个 draft 块，
  与 causal SDPA mask 不同）。

两段相接或重叠时合并为 `[0, kv_len)`；分离时中间 gap 的 token 不可见。
物理页翻译：tile 的逻辑 token 起点 → `页号 = token / page_size` → `block_table[batch, 页号]`
得物理页，页内偏移取数，跨页自动续页。

三种典型配置：

| 场景 | pre_tokens | next_tokens | 可见集合 |
|---|---:|---:|---|
| FA + DFlash | INT32_MAX | 0 | 全 context + draft 块 |
| SWA + DFlash | W | 0 | 最近 W 个 context token + draft 块 |
| SWA + sink + DFlash | W | S（8 的倍数） | 序列头 S + 最近 W 个 + draft 块 |

## 2、参数说明

维度符号：

- B：batch，请求个数
- T：query 总 token 数 = 各请求 q_len 之和（TND 首维）
- N：query 头数（num_heads）
- Nkv：key/value 头数（num_key_value_heads），N 须被 Nkv 整除；0 = MHA（Nkv=N）
- q_len / kv_len：单请求的 draft 长度 / 逻辑 KV 长度（含 draft 块）
- D：headDim
- page_size：PageAttention 页大小（block_size）

| 入参/出参 | 属性类别 | 定义 | 支持类型 | 支持范围 |
|---|---|---|---|---|
| query | 必传入参 tensor | `[T, N, D]`（TND）。draft token 的 query | float16 / bfloat16 | D ≤ 256 |
| key | 必传入参 tensor | PageAttention K 缓存 `[numBlocks, page_size, Nkv*D]`（或 4 维 `[numBlocks, page_size, Nkv, D]`），页内 token 连续 | float16 / bfloat16 | 与 query 同 dtype |
| value | 必传入参 tensor | PageAttention V 缓存，布局同 key | float16 / bfloat16 | 同 key |
| block_table | 必传入参 tensor | `[B, maxNumPagesPerReq]`。逻辑页号 → 物理页号 | int32 | 页号 < numBlocks |
| actual_seq_lengths | 可选入参 NPU tensor | `[B]`。TND 前缀和形式的 draft 长度（第 b 项 = 前 b 个请求的 q_len 累加） | int64 | 逐 request q_len > 0 |
| actual_seq_lengths_kv | 可选入参 NPU tensor | `[B]`。逐 request 的逻辑 KV 长度（绝对值，非前缀和） | int64 | q_len ≤ kv_len ≤ maxNumPagesPerReq × page_size |
| num_heads | 必选 attr | query 头数 N | int | >0 |
| scale | 可选 attr | softmax 缩放系数 | float | 默认 1.0，常用 1/√D |
| pre_tokens | 可选 attr | 尾带窗口长度；INT32_MAX = 全 context | int | 默认 2147483647，≥0 |
| next_tokens | 可选 attr | prefix sink 长度；0 = 关闭；INT32_MAX = 不限（sink 覆盖整个 context，等价全量） | int | 默认 16；其余取值必须为 8 的正倍数 |
| input_layout | 可选 attr | 输入 layout | string | 默认 "TND"，仅支持 TND |
| num_key_value_heads | 可选 attr | kv 头数（GQA 分组） | int | 默认 0（=MHA），须整除 N |
| block_size | 可选 attr | PageAttention 页大小，须等于 K/V 缓存的页维度 | int | 默认 0；**当前仅支持 128** |
| softmax_lse_flag | 可选 attr | 是否输出 softmax_lse | bool | 默认 false |
| attention_out | 输出 tensor | `[T, N, D]`，与 query 同 shape。attention 输出 | float16 / bfloat16 | 与 query 同 dtype |
| softmax_lse | 输出 tensor | softmax 的 log-sum-exp（flag=false 时为空张量） | float32 | 固定 float32 |

> 注：PTA 签名里保留了 `sparse_mode` / `inner_precise` 两个形参（默认值即可），
> 不会传给算子——本算子语义固定为 DFlash paged、高精度 softmax。

## 3、约束说明

- **数据类型**：query / key / value / attention_out 一致，仅 float16 / bfloat16；block_table 为 int32；softmax_lse 固定 float32。
- **布局**：仅 TND；K/V 缓存必须页内 token 连续（物理偏移 = (物理页号 × page_size + 页内偏移) × Nkv×D）。
- **页大小**：当前仅支持 page_size = 128（一个 512-token 计算 tile 恰为 4 页，跨页取数已按此覆盖）。
- **长度关系**：每请求 `0 < q_len ≤ kv_len ≤ maxNumPagesPerReq × page_size`；`actual_seq_lengths` 传 TND 前缀和，`actual_seq_lengths_kv` 传逐请求绝对长度。
- **sink 参数**：`next_tokens` 为 0、INT32_MAX（不限，等价全量）或 8 的正倍数；`pre_tokens` ≥ 0（INT32_MAX = 全 context）。
- **不支持的特性**：atten_mask / pse_shift / learnable sink / 量化（antiquant）——输入里已没有这些口子。
- **GQA / MQA**：N 必须被 num_key_value_heads 整除；num_key_value_heads=0 表示 MHA。
- **可见性语义**：draft 块内双向可见（非 causal）；sink 段与尾带分离时中间 gap 精确不可见。
- **支持芯片**：Atlas A2/A3（ascend910b/ascend910_93）。

## 4、调用样例

```python
import math
import torch
import torch_npu

B, N, NKV, D, PAGE = 2, 8, 2, 128, 128
q_lens   = [7, 16]          # 每个请求的 draft 长度
kv_lens  = [4123, 8207]     # 每个请求的逻辑 KV 长度（含 draft）

# 物理页缓存与页表（页内 token 连续；物理页可乱序分配）
page_counts = [math.ceil(kl / PAGE) for kl in kv_lens]
num_blocks  = sum(page_counts) + 7
key_cache   = torch.randn(num_blocks, PAGE, NKV * D, dtype=torch.bfloat16).npu()
value_cache = torch.randn(num_blocks, PAGE, NKV * D, dtype=torch.bfloat16).npu()
block_table = torch.zeros(B, max(page_counts), dtype=torch.int32)
# ... 按分配结果填 block_table[b, :page_counts[b]] = 物理页号 ...

query = torch.randn(sum(q_lens), N, D, dtype=torch.bfloat16).npu()
actual_q = torch.tensor([q_lens[0], sum(q_lens)], dtype=torch.int64, device="npu")
actual_kv = torch.tensor(kv_lens, dtype=torch.int64, device="npu")

out, lse = torch_npu.mlp_dflash_infer_attn(
    query, key_cache, value_cache,
    actual_seq_lengths=actual_q,      # TND 前缀和
    actual_seq_lengths_kv=actual_kv,  # 逐请求绝对长度
    block_table=block_table.npu(),
    num_heads=N, scale=D ** -0.5,
    pre_tokens=4096,          # 尾带窗口 W；2147483647 = 全 context
    next_tokens=16,           # prefix sink 长度；0 = 关闭，正数为 8 的倍数
    input_layout="TND", num_key_value_heads=NKV,
    block_size=PAGE, softmax_lse_flag=False)
```

## 5、验证

测试入口：`test/test_mlp_dflash_infer_attn.py`（自 FIA `test_fia_dflash_paged.py` 改写）：

```bash
cd test
python3 test_mlp_dflash_infer_attn.py     # 或 pytest -v
```

覆盖：全量 / 纯 SWA / SWA+sink(8/16/24)、两段相接与分离、区间起点在页中间、
变长 batch（q_len 7/16 混合）、物理页随机打散；`DFLASH_TEST_KV_LEN` 环境变量可放大 cache 长度。
golden 为「按页表 gather 逻辑 KV + 可见下标取子集」的 fp32 参考（rtol=5e-3, atol=5e-2）。
