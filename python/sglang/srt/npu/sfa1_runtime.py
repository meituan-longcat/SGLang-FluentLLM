"""Model-local selection of the split-KV SFA1 ABI.

The default flash_ops wheel contains both custom (legacy) and fluent_sfa1
namespaces. SFA1 is the host-offload path's ABI: its 7/9-output Prolog
produces the current-KV tensors the offload writeback consumes, which the
legacy 5/7 ABI cannot. So offload decides the route, and eligibility decides
only whether offload could run on this worker's layout at all. The route is a
capability, not a destination: npu_mla_backend takes the SFA1 arm only where
the pool also carries a SparseCoordinator.

A worker with offload off keeps the legacy SFAD path - what mainline runs -
and never imports the optional module. Non-DSA and packed-C8 workers do not
import it either. Only the loaded operator namespace is cached globally;
target and draft model routes are separate immutable booleans, even when they
share one process.
"""

import importlib

_ops = None


def get_sfa1_ops():
    """Resolve the required 7/9-output ABI, rejecting broken installations."""
    global _ops
    if _ops is None:
        module = importlib.import_module("fluent_sfa1_ops")
        if module.ABI_VERSION != 1:
            raise RuntimeError("Unsupported fluent_sfa1 ABI; expected version 1")
        _ops = module.ops
    return _ops


def initialize_sfa1_runtime(*, eligible, offload_enabled, owns_offload_pool):
    """Return a model-local route before communication and graph compilation.

    Offload off means the legacy SFAD path, whatever is installed: having the
    operator package present is not a reason to change the attention
    implementation.

    Offload on requires SFA1, so a layout it cannot serve is an error rather
    than a silent fall back to the incompatible 5/7 ABI - but only where this
    worker actually holds an offload pool. attn_initializer builds the pool,
    and with it the SparseCoordinator, under a narrower predicate than
    --offload-cache: an MTP draft worker and a prefill node see the flag but
    own no pool, and the attention backend sends them to the mainline SFAD arm
    whatever this route says. Their model config is their own, so it can be
    ineligible while the target's is not, and refusing to start there would
    fail a worker that never needed SFA1.

    A missing or mismatched module stays fatal everywhere, pool or not: it
    means the wheel is wrong for the whole job, and the pool-owning workers
    would fail on it regardless.

    deepseek_mla resolves the same route when it builds a layer, with
    owns_offload_pool False because ModelRunner.__init__ already made the
    authoritative call. Both answers agree: this returns eligible and
    offload_enabled on every path that returns at all.
    """
    if not offload_enabled:
        return False
    if not eligible:
        if owns_offload_pool:
            raise ValueError("NPU KV offload requires the supported split-KV DSA SFA1 path")
        return False
    try:
        get_sfa1_ops()
    except ModuleNotFoundError as error:
        if error.name != "fluent_sfa1_ops":
            raise
        raise RuntimeError(
            "NPU KV offload requires fluent_sfa1_ops from the default unified "
            "flash_ops wheel; install the paired operator package"
        ) from error
    return True


def get_attention_ops(use_sfa1):
    if use_sfa1:
        return get_sfa1_ops()
    import torch

    return torch.ops.custom
