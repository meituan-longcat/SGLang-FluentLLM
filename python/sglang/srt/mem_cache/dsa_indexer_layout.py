"""Logical Indexer layout shared by allocation, budgeting and transfers.

CLI reuses the first attention's selected indices; it does not share writable
IndexK storage. Unused logical layers keep independent one-page placeholders.
Draft pools retain their existing storage until their own consumers are audited.
"""


def normalize_indexer_mask(layer_count, mask=None):
    result = (True,) * layer_count if mask is None else tuple(mask)
    if len(result) != layer_count or any(type(active) is not bool for active in result):
        raise ValueError("Indexer mask requires one bool per logical attention layer")
    return result


def cli_indexer_mask(start_layer, end_layer, cli_factor=2, *, is_draft=False):
    if start_layer < 0 or end_layer < start_layer or cli_factor != 2:
        raise ValueError("Indexer placeholders require valid layer bounds and CLI factor 2")
    return tuple(is_draft or layer % cli_factor == 0 for layer in range(start_layer, end_layer))


def check_transfer_layout(source_offsets, destination_offsets, layer_id):
    if len(source_offsets) != len(destination_offsets):
        raise ValueError(
            f"P/D Indexer layout mismatch at attention layer {layer_id}: "
            f"source buffers={len(source_offsets)}, "
            f"destination buffers={len(destination_offsets)}. "
            "Use matching Indexer-placeholder settings on P and D."
        )


def check_stage_transfer_layout(source_offsets, destination_offsets, start_layer, layer_count, source_buffer_count, destination_buffer_count):
    """Freeze/check the entire P stage before its first layerwise DMA.

    Source rows are stage-local; destination rows use global attention ids.
    The final stage includes its unchanged draft rows after target attention.
    Checking only the submitted prefix could send even-layer data before an
    odd-layer placeholder mismatch or a missing draft tail is discovered.
    """
    if (
        type(start_layer) is not int
        or type(layer_count) is not int
        or start_layer < 0
        or layer_count < 0
        or len(source_offsets) != layer_count
        or start_layer + layer_count > len(destination_offsets)
    ):
        raise ValueError("P/D stage offsets do not cover the declared global layer range")
    checked = []
    for local_id, source in enumerate(source_offsets):
        global_id = start_layer + local_id
        destination = destination_offsets[global_id]
        check_transfer_layout(source, destination, global_id)
        for name, offsets, count in (("source", source, source_buffer_count), ("destination", destination, destination_buffer_count)):
            if any(type(index) is not int or not 0 <= index < count for index in offsets):
                raise ValueError(f"P/D {name} offset outside registered buffers at attention layer {global_id}")
        checked.append((tuple(source), tuple(destination)))
    return tuple(checked)


def allocate_indexer_pages(mask, page_count, page_size, head_dim, *, dtype, device):
    """Allocate compact real pages and independent zero placeholder pages.

    Kept outside the model so the actual allocation code can be checked on CPU.
    Logical order and the active layers' PA_BSND format remain unchanged.
    """
    import torch

    mask = normalize_indexer_mask(len(mask), mask)
    if page_count < 1 or page_size < 1 or head_dim < 1:
        raise ValueError("Indexer page dimensions must be positive")
    active = torch.zeros((sum(mask), page_count, page_size, 1, head_dim), dtype=dtype, device=device)
    inactive = torch.zeros((len(mask) - sum(mask), 1, page_size, 1, head_dim), dtype=dtype, device=device)
    views = []
    real_index = placeholder_index = 0
    for enabled in mask:
        if enabled:
            views.append(active[real_index])
            real_index += 1
        else:
            views.append(inactive[placeholder_index])
            placeholder_index += 1
    return active, inactive, views
