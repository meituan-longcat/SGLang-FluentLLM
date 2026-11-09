# CausalConv1d 算子接口

## 功能

逐通道 causal convolution，融合可选 bias、SiLU 和原地 history 更新。
不包含 KDA recurrence、Q/K 归一化或 gate。支持 ascend910b/ascend910_93。

## Torch 接口

```python
y = torch.ops.custom.npu_causal_conv1d(
    x, weight, conv_state,
    bias=None, query_start_loc=None, cache_indices=None, write_indices=None,
    initial_state_mode=None, num_accepted_tokens=None,
    activation_mode=1, pad_slot_id=-1, run_mode=0,
)
```

所有 tensor 位于同一 NPU，格式为 ND。`x`、`weight`、`conv_state`、`bias`
使用相同的 FP16 或 BF16 dtype。输出 `y` 与 `x` 同 shape、dtype；`conv_state`
按下述规则原地更新。`B` 为序列数，`T` 为 packed token 总数，`W` 为卷积宽度。

| 参数 | Shape / 类型 | 语义 |
| --- | --- | --- |
| `x` | `[T,C]` / `[B,C]` / `[B,N,C]` | packed Prefill / 单 token Decode / 定长序列；`C>0` 且为 16 的倍数 |
| `weight` | `[W,C]` | `2 <= W <= 4`，从旧 history 到当前 token 的卷积权重 |
| `conv_state` | `[P,L,C]`，stride `[S,C,1]` | `P>0`，`L>=W-1`，`S>=L*C`，首轴可非连续；允许非零 storage offset |
| `bias` | 可选 `[C]` | 卷积后加 bias |
| `query_start_loc` | 可选 int32/int64 `[B+1]` | packed Prefill 必填；从 0 开始、非递减、最后为 `T` 的序列边界 |
| `cache_indices` | 可选 int32/int64 `[B]` | 读页编号；省略时第 `i` 个序列使用第 `i` 页，须 `P>=B` |
| `write_indices` | 可选 int32/int64 `[B]` | 独立写页；省略时写回读页。Torch 要求 rank-2 `x`、同时传入同 shape/dtype 的 `cache_indices`；Decode 还要求 `L=W-1`、`B=x.size(0)` |
| `initial_state_mode` | 可选 bool/int32/int64 `[B]` | **仅 Prefill**：非零表示读取已有 history，零或省略表示从零 history 开始；可与 `write_indices` 同时使用 |
| `num_accepted_tokens` | 可选 int32/int64 `[B]` | 既有 speculative Update 参数，仅 `run_mode=1, W=4`；读取 history 的行偏移为 `clamp(value-1, 0, L-(W-1))`。定长 `[B,N,C]` 要求 `L>=W-1+N-1` |
| `activation_mode` | int，默认 1 | 0：不激活；1：SiLU |
| `pad_slot_id` | int，默认 -1 | 命中此值的读页或写页跳过整个序列 |
| `run_mode` | int，默认 0 | 0：Prefill；1：Decode/Update |

既有 speculative 路径也接受二维 `cache_indices`，只读取每行第一列；
本次独立 R/W 扩展只承诺上表的 rank-2 `x`、一维页索引组合，不扩展 speculative 协议。

## 模式与页生命周期

- Prefill：按序列处理输入；可选择 fresh（零 history）或 resume（读取 history）。
  普通 `L=W-1` 场景结束时，最后 `W-1` 个 token（不足时含初始 history）写入写页。
  空序列不写 state。定长输入 `[B,N,C]` 沿用既有 Prefill 路径。
- Decode/Update：读取已有 history 并更新；普通 Decode 为 `[B,C]`、每序列一个 token。
  `initial_state_mode` 在此模式明确拒绝，不能再当作写页参数传入。
- 两种模式共用 `[P,L,C]` cache layout。读页与写页可相同；活跃写页必须唯一，
  不支持某请求的写页覆盖同次调用另一请求的读页。调用者负责页生命周期及跨调用依赖。
- 读页或写页命中 `pad_slot_id`、小于 0 或不小于 `P` 时，kernel 跳过相应序列，
  **不写输出和 state**。即使 fresh Prefill 不读 history，也要提供有效页编号。

### padding 输出为何为零

零输出是本仓 **Torch 封装的既有约定**，不是 causal convolution 数学要求，
也不是本次 stride 支持新增的约定。`RunCausalConv1d` 在调用 ACLNN 前统一执行一次
`at::zeros_like(x)`，因此被跳过的 token 在 Torch 返回值中为零。kernel 不重复清零。
原生 NPUGraph 会捕获这次初始化，每次 replay 都执行，避免 padding 位置变化后残留旧输出。

直接调用底层 ACLNN/OPP 时，输出由调用者提供：被跳过的位置保持原内容，
若需要零值必须由调用者初始化；不能依赖 kernel 自动置零。

## ACLNN/OPP 接口变化与迁移

Torch schema 名称和参数列表不变；底层 `CausalConv1d` 输入顺序为
`x, weight, bias, convStates, queryStartLoc, cacheIndices, initialStateMode, numAcceptedTokens, writeIndices`，
属性顺序为 `activationMode, padSlotId, runMode, stateSlotStride`，输出为 `y`。
OPP 的 `activationMode` 默认 0（与 Torch 默认 1 不同），其余默认值分别为 -1、0、0。

| 接口层 | PR #20 / master `df169246` | 本版本 |
| --- | --- | --- |
| Torch `write_indices` | 已支持 Decode 独立读写；binding 借用底层 `initialStateMode` 传写页 | 映射到专门的底层 `writeIndices`，扩展到 Prefill |
| 底层 `initialStateMode` | Prefill 为初始 history mask，Decode 被复用为写页 | 只表示 Prefill history mask，Decode 不接受 |
| 底层 `writeIndices` | 无独立输入槽 | 新增可选输入，省略时仍写回读页 |
| 底层 `stateSlotStride` | 无，按连续 state 寻址 | 新增 int 属性，单位为元素；0 表示 `L*C` |

因此这不是“Decode 以前不能独立读写”，而是消除底层参数的双重含义，
让 Prefill 可同时传 history mask 和写页。底层调用者应将旧 Decode 写页参数移至
`writeIndices`，并将 `initialStateMode` 置空。

Torch 自动从 `conv_state.stride(0)` 传入 `stateSlotStride`。底层 strided 调用者必须
传真实步长，不能省略或传 0；页内必须连续，分配的 backing 必须覆盖全部页。
推荐首地址、行跨度、页跨度按 **256B 对齐**以减少性能损失；这不是当前接口的硬性校验条件。

**wheel 和 custom OPP 必须配套升级**：底层新增输入和属性改变了 ACLNN ABI，
不能混用旧 op-api 动态库。包能力标记为 `CAUSAL_CONV1D_STATE_SLOT_STRIDE=True`。

## 图模式与验证

非连续 state 支持 eager、原生 `torch.npu.NPUGraph`。当前 TorchAir GE 会连续化图输入，
converter 因此拒绝非连续 state，避免丢失原地更新；连续 state 的 GE replay 保留支持。

[测试用例](../tests/test_npu_causal_conv1d.py) 覆盖 Lite 两尺寸 × TP1/2/4/8
去重的五档通道数、FP16/BF16、Prefill/Decode 独立 R/W、非法读/写页、非零 offset、
整个 backing 的越界写检查，以及 padding 页变化后的原生图回放。

## 修订记录

- 2026-09-08：记录独立 `writeIndices`、首轴 stride、升级边界和单次输出初始化语义。
