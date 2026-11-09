# CausalConv1d：原地 state 与独立读写页

来自 `third_party/causal_conv1d.lock.json` 固定的公开实现；融合 depthwise causal-conv、
可选 bias/SiLU 和 history 更新，不包含 KDA recurrence、归一化或 gate。

完整参数、模式及版本迁移见 [算子接口文档](../../docs/custom-npu_causal_conv1d.md)。

## 输入合同

- `conv_state`: BF16/FP16 `[slots,L,C]`，stride `[S,C,1]`，`S >= L*C`；
  页内连续、首轴可非连续，允许非零 storage offset。其他页内 stride 明确拒绝。
- `cache_indices` 为读页，`write_indices` 为可选独立写页；省略写页时原页更新。
  Prefill/Decode 都支持 rank-2 输入的独立读写；索引是 `[batch]` 的 int32/int64。
- `run_mode=0` 为 packed Prefill，`query_start_loc` 给定序列边界；
  `initial_state_mode` 仅在 Prefill 表示是否读取 history，可与写页同时传入。
- `run_mode=1` 为 Decode/Update，读取已有 history，不接受 `initial_state_mode`。
- padding/无效读写页不修改 state，kernel 不写相应输出；Torch 封装统一初始化输出为零，
  kernel 不重复置零。直接调用 ACLNN 时，如需零输出须由调用者初始化。空序列不发布 state。
  活跃写页必须唯一；同一请求读写同页允许，跨请求写入另一请求的读页不受支持。

## 实现与安装边界

所有 GM state 页基址按 `slot * stateSlotStride` 计算；属性由 Torch binding 的
`conv_state.stride(0)` 推导，不要求调用者维护重复参数。OPP 使用 `IgnoreContiguous`；
仅依赖 `GetInputStride` 不充分，部分 ACLNN runtime 不提供 view metadata。
若 metadata 存在则同时校验它与属性一致。底层调用者省略属性时仅能依赖 dense 默认值，
对于 strided view 必须显式传真实步长。

Torch schema 仍为 `custom::npu_causal_conv1d`，底层 ACLNN/OPP 新增 `writeIndices`
输入与 `stateSlotStride` 属性。**wheel 与 custom OPP 必须由同一版本配套构建安装**，
不能混用旧 op-api 动态库。包标记 `CAUSAL_CONV1D_STATE_SLOT_STRIDE=True` 供消费者检查。

非连续 state 支持 eager 与原生 `torch.npu.NPUGraph`。当前 TorchAir GE 会在图输入端
连续化非连续 tensor，converter 因此明确拒绝非连续 state，避免原地 mutation 丢失或
按旧 stride 越界；连续 state 的 GE replay 支持保留。不修改/monkeypatch TorchAir 框架。

## 验证

`tests/test_npu_causal_conv1d.py` 覆盖 Lite 3B/Large × TP1/2/4/8 去重通道数
1536/3072/6144/12288/24576：dense Decode、strided Prefill/Decode、fresh/resume、
独立 R/W、padding、非零 storage offset、整个 backing sentinel 校验、非法页内 stride、
连续 GE changed-input/page replay 与非连续 GE fail-loud。TS 消费方另测真实 CacheArena
及原生 NPUGraph 回放；这两种图机制的结果不能混称。
