"""Native eager host-backed KVP row-copy path used by KVPCOPY3.

No installation, model-method replacement or device initialization on import.
HBM rows retain Index/Scatter. Indexer rows do too unless the caller identifies
layers whose current rows were already written by the Indexer. Source and
destination tensors keep the ordinary eager current-KV write order.
"""


def host_attention_ids(device_types, attention_count, use_dsa=False):
    # DSA appends one HBM Indexer flag per attention after the interleaved
    # attention KV/rope flags. That tail must not be interpreted as KV backing.
    expected = attention_count * (3 if use_dsa else 2)
    pairs = [device_types[2 * sub:2 * sub + 2] for sub in range(attention_count)]
    if (device_types is None or len(device_types) != expected
            or (use_dsa and device_types[attention_count * 2:] != ["npu"] * attention_count)
            or any(pair not in (["cpu", "cpu"], ["npu", "npu"]) for pair in pairs)):
        raise RuntimeError(
            "Unexpected split KV/rope plus Indexer backing layout: "
            f"{device_types} (expected {expected} entries for "
            f"{attention_count} attentions, use_dsa={use_dsa})"
        )
    return frozenset(sub for sub, pair in enumerate(pairs) if pair == ["cpu", "cpu"])


def apply_current_copy(pool, attention_count, source, slots, host_ids,
                       copy_rows, scatter, host_source=None, *, skip_indexer_layers=(),
                       skip_keyrope_layers=(), host_burst=None):
    """Copy current rows, retaining Indexer writes unless explicitly covered."""
    flat_slots = slots.reshape(-1)
    if host_source is None:
        host_source = source
    use_dsa = hasattr(pool, "get_index_k_with_scale_buffer")
    burst_done = (
        host_burst is not None
        and not any(sub in skip_keyrope_layers for sub in host_ids)
        and host_burst.copy(host_source, flat_slots)
    )
    for sub in range(attention_count):
        if sub not in skip_keyrope_layers and (sub not in host_ids or not burst_done):
            key, rope = pool.get_kv_buffer(sub)
            key = key.view(-1, key.shape[-1])
            rope = rope.view(-1, rope.shape[-1])
            if sub in host_ids:
                copy_rows(key, host_source, flat_slots)
                copy_rows(rope, host_source, flat_slots)
            else:
                scatter(key, slots, key[source])
                scatter(rope, slots, rope[source])
        if (use_dsa and sub not in skip_indexer_layers
                and (not hasattr(pool, "has_indexer_cache") or pool.has_indexer_cache(sub))):
            index = pool.get_index_k_with_scale_buffer(sub)
            index = index.view(-1, index.shape[-1])
            scatter(index, slots, index[source])
