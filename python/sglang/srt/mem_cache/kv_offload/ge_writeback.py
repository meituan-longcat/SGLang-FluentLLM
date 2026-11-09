"""Alias-safe UpdateCache writeback for TorchAir GE offload graphs.

Host offload writes the hot cache back through a kernel that stages the old
rows before overwriting them, so TorchAir's Assign-to-NetOutput pass may bind
the producer's output to its input storage. Other operators, including the
main and draft KV scatter paths, keep their existing optimization.

Operator package integrity is the build's job, not this module's; this path
only records which build it resolved.
"""


def require_alias_safe_writeback():
    """Reject a launch script still asking for the removed Assign variant.

    Scripts that set NPU_OFFLOAD_WRITEBACK_MODE=alias_safe keep working. One
    asking for assign must fail here rather than silently get alias-safe.
    """
    import os
    mode = os.environ.get("NPU_OFFLOAD_WRITEBACK_MODE", "alias_safe")
    if mode != "alias_safe":
        raise ValueError(
            "NPU_OFFLOAD_WRITEBACK_MODE only supports alias_safe; the Assign "
            "writeback variant has been removed"
        )


def _private_op_cache_path():
    """Point ASCEND_CACHE_PATH at a directory of ours, creating it if needed.

    CANN's default kernel cache is shared, so it can hand this worker kernels
    compiled from a previously installed flash_ops. An operator who already
    set the variable keeps their own location; otherwise derive one from the
    graph cache root, which qualify_graph_cache has already namespaced by the
    writeback variant, so it stays reusable across restarts.
    """
    import os
    import tempfile

    from sglang.srt.env import global_server_args_dict

    path = os.environ.get("ASCEND_CACHE_PATH")
    if not path:
        root = global_server_args_dict.get("npu_graph_cache_path")
        path = (os.path.join(root, "op-cache") if root
                else os.path.join(tempfile.gettempdir(), "sglang-npu-op-cache"))
        os.environ["ASCEND_CACHE_PATH"] = path
    os.makedirs(path, exist_ok=True)
    return path


def log_alias_safe_update_cache_runtime():
    """Record which operator build serves UpdateCache; the build seals its manifest.

    The operator package already verifies its own file set at build time, so
    re-hashing it here would only add a second source of truth. A private
    ASCEND_CACHE_PATH is still required: a shared kernel cache can hand this
    worker kernels compiled from a previously installed flash_ops.
    """
    import importlib
    import logging
    import os
    cache_path = _private_op_cache_path()
    module = importlib.import_module("fluent_sfa1_ops")
    logging.getLogger(__name__).info(
        "Alias-safe UpdateCache runtime: fluent_sfa1_ops=%s abi=%s custom_opp=%s cache=%s",
        getattr(module, "__file__", "<unknown>"), getattr(module, "ABI_VERSION", None),
        os.environ.get("ASCEND_CUSTOM_OPP_PATH", ""), cache_path,
    )


def install_update_cache_writeback_fix():
    require_alias_safe_writeback()
    log_alias_safe_update_cache_runtime()
