from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from sglang.srt.utils import get_colorful_logger

if TYPE_CHECKING:
    from sglang.srt.mem_cache.radix_cache import TreeNode

logger = get_colorful_logger(__name__)

WriteTarget = tuple[Optional[str], Optional[tuple[int, int]]]
SINGLE_WRITE: list[WriteTarget] = [(None, None)]


class L3MixinWrite:
    def cross_pp_write_targets(self) -> list[WriteTarget]:
        kv_manager = getattr(self, "prefill_kv_manager", None)
        if kv_manager is None:
            return SINGLE_WRITE
        info_table = getattr(kv_manager, "prefill_parallel_info_table", None)
        if not info_table:
            return SINGLE_WRITE
        n = next(iter(info_table.values()))[3]
        if n <= 1:
            return SINGLE_WRITE

        layer_ranges = getattr(kv_manager, "prefill_pp_layer_ranges", None)
        if layer_ranges is None or any(i not in layer_ranges for i in range(n)):
            return SINGLE_WRITE

        draft_layer_num = self.token_to_kv_pool_host.layer_num - self.token_to_kv_pool.layer_num
        if draft_layer_num > 0:
            assert layer_ranges[n - 1][1] == self.token_to_kv_pool.layer_num, (
                f"cross-PP draft append assumes prefill total main layers "
                f"({layer_ranges[n - 1][1]}) == decode main layers "
                f"({self.token_to_kv_pool.layer_num}); got {layer_ranges=}"
            )

        targets = []
        for i in range(n):
            start, end = layer_ranges[i]
            if i == n - 1 and draft_layer_num > 0:
                end += draft_layer_num
            targets.append((f"pp{i}of{n}_", (start, end)))
        return targets

    def add_h2s_write(self, node: TreeNode):
        if self.l3_ctrl is None:
            return

        if self.l3_ctrl.is_backup_full():
            logger.info(
                f"[L3-Backup-Reject] inflight={self.l3_ctrl.h2s_write_inflight} "
                f">= hard={self.l3_ctrl.backup_inflight_hard} node={node.id} "
                f"pages={len(node.host_value)}"
            )
            return
        prefix_keys = self.get_node_prefix_keys(node)
        if prefix_keys is None and self.l3_ctrl.storage_backend_type == "mooncake":
            # Mooncake's backup-side existence query renews the complete
            # already-local prefix, regardless of the generic backend option.
            prefix_keys = node.get_prefix_hash_values(node.parent)
        # The asynchronous backup worker extends this list after every batch.
        # Detach it from TreeNode's cached prefix before transferring ownership.
        if prefix_keys is not None:
            prefix_keys = list(prefix_keys)
        logger.debug(f"start write storage {node.host_value=} {prefix_keys=}")

        for pp_prefix, layer_range in self.cross_pp_write_targets():
            node.protect_host()
            self.l3_ctrl.add_h2s_write(node, prefix_keys, pp_prefix=pp_prefix, layer_range=layer_range)
