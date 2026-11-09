"""Selected-KV state keyed by attention id, with no torch dependency.

Only SFA1 (attention id 1) is offloaded. Its selected KV is gathered into the
HBM selection buffer while SFA0 of the same layer runs, and SFA1 attention
then reads that buffer. SFA0 keeps its KV on device and never publishes a
context, so has_selection_context(0) is False and SFA0 reads the device pool.
Keying by attention id lets each consumer ask for its own attention's context
instead of assuming the single live one belongs to it.
"""

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class SelectionContext:
    attention_id: int
    topk_indices: Any
    k_nope: Any
    k_rope: Any
    write_slot: Any
    cur_kv_write_slots: Any
    actual_seq_qlen: Any
    actual_kv_seq: Any
    update_cur_kv_done: Optional[Any] = None
    # The TopK the gather was planned from. topk_indices above is positional
    # over the gathered buffer, so it only means what this selection means;
    # a consumer holding a different TopK must not be given these operands.
    source_topk: Optional[Any] = None
    # Packed selection uses k_nope for complete byte rows and has no separate
    # RoPE tensor. Zero retains the split BF16 contract for existing callers.
    packed_row_bytes: int = 0
    # Scheduling hint produced with source_topk, frozen across CLI-pair reuse.
    source_valid_chunks: Optional[Any] = None
    # Padded logical token IDs supplied to Gen/Gather in packed mode. Reuse
    # this exact operand for LRU admission instead of rebuilding it later on
    # the update stream from source_topk. topk_indices above stays positional.
    cache_topk_indices: Optional[Any] = None


class SelectionContextStore:
    """One slot per attention id; an unpublished attention reads as absent."""

    def __init__(self):
        self._contexts = {}

    def publish(self, context: SelectionContext) -> None:
        self._contexts[context.attention_id] = context

    def get(self, attn_idx: int, required: bool = True) -> Optional[SelectionContext]:
        context = self._contexts.get(attn_idx)
        if required and context is None:
            raise RuntimeError(
                f"no selected KV context is live for attention id {attn_idx}"
            )
        return context

    def clear(self, attn_idx: Optional[int] = None) -> None:
        if attn_idx is None:
            self._contexts.clear()
        else:
            self._contexts.pop(attn_idx, None)
