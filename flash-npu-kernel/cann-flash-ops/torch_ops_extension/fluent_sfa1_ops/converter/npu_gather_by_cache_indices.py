"""Register the shared gather_by_cache_indices implementation."""
from flash_ops.offload.npu_gather_by_cache_indices import register

_library = register("fluent_sfa1")
