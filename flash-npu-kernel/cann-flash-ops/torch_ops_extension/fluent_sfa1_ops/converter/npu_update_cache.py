"""Register the shared update_cache implementation."""
from flash_ops.offload.npu_update_cache import register

_library = register("fluent_sfa1")
