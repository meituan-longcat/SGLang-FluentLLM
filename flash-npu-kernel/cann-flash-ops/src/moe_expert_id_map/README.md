# MoeExpertIdMap

`MoeExpertIdMap` fuses the expert-id post-processing between
`MoeGatingTopK` and `MoeDistributeDispatchV2`:

```text
ClipByValue + Cast(INT32 -> INT64) + GatherElements
                         -> MoeExpertIdMap
```

For every element in `logical_ids`, the operator computes:

```text
physical_ids[i] = expert_map[clamp(logical_ids[i], 0, map_size - 1)]
```

Both inputs and the output use `INT32`. The final element of `expert_map` is
the zero-expert sentinel in the online `static_with_zero_expert` path, so the
caller must provide a map whose length is `num_experts + 1`.

## Interface

| Tensor | Shape | Dtype | Description |
| --- | --- | --- | --- |
| `logical_ids` | arbitrary contiguous ND | INT32 | Logical expert IDs from TopK |
| `expert_map` | `[map_size]`, `map_size > 0` | INT32 | Logical-to-physical lookup table |
| `physical_ids` | same as `logical_ids` | INT32 | Dispatch physical expert IDs |

The current online shape is `logical_ids=[8, 12]` and `expert_map=[769]`.
The map and all IDs are copied to UB once, then one AIV core performs the
bounded table lookup. Tiling depends only on tensor shapes; map values remain
runtime inputs and can change between graph replays.

## PyTorch interface

```python
physical_ids = torch.ops.custom.npu_moe_expert_id_map(
    logical_ids, expert_map
)
```

The C++ extension and torchair converter are included in
`torch_ops_extension`. The caller should verify that
`expert_map.shape[0] == num_experts + 1` before invoking the operator.

## Limitations

- `logical_ids`, `expert_map`, and `physical_ids` are INT32 only.
- Inputs must be contiguous and on the same device.
- The complete map and input/output ID buffers must fit in UB. This is
  intended for small routing tensors, not general large-table embedding.
