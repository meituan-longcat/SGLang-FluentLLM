# KvpAttentionMerge

`KvpAttentionMerge` fuses the post-processing used to merge attention results
from eight KVP ranks:

```text
Softmax(LSE) + Mul + ReduceSum(rank) + Cast(BF16)
                         -> KvpAttentionMerge
```

For each row `m`, the operator computes a stable FP32 softmax across ranks and
uses the result to combine the rank-local FP32 attention outputs:

```text
weight[r,m] = exp(lse[r,m] - max(lse[:,m])) /
              sum(exp(lse[:,m] - max(lse[:,m])))
output[m,:] = BF16(sum(weight[:,m] * attention_out[:,m,:]))
```

## Interface

| Tensor | Shape | Dtype | Description |
| --- | --- | --- | --- |
| `attention_out` | `[8, rows, 512]` | FP32 | Rank-local attention outputs |
| `lse` | `[8, rows]` | FP32 | Rank-local log-sum-exp values |
| `y` | `[rows, 512]` | BF16 | Merged attention output |

`rows` must be positive. Both inputs must be contiguous and on the same
device. The current implementation targets the production KVP world size and
hidden size; other rank counts and hidden sizes are rejected.

## Empty-rank contract

An empty rank must use a sufficiently negative LSE sentinel (`-2e38` or
negative infinity). Its softmax weight becomes zero, and the corresponding
attention row is ignored even if it contains NaN. If every rank is empty, the
operator returns an all-zero row. NaN LSE values are outside the input
contract.

## PyTorch interface

```python
output = torch.ops.custom.npu_kvp_attention_merge(attention_out, lse)
```

The operator supports eager execution and `torch.compile` through the
TorchAir converter in `torch_ops_extension`.

## Implementation

Each AIV core loads all LSE rows assigned to it into UB and calculates their
rank softmax in one vector batch. Attention data is then loaded one row at a
time, accumulated in FP32 and written directly as BF16. This avoids the FP32
`Mul` and `ReduceSum` intermediate tensors produced by the unfused chain.
