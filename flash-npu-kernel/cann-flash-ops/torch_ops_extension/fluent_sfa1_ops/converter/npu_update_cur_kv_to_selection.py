"""Register the shared update_cur_kv_to_selection implementation."""
from flash_ops.offload.npu_update_cur_kv_to_selection import register

_library = register("fluent_sfa1")
