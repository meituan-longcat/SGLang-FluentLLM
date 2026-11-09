"""Register the shared gen_from_cache_indices implementation."""
from flash_ops.offload.npu_gen_from_cache_indices import register

_library = register("fluent_sfa1")
