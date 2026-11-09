"""Host-KV defaults, optional row-copy acceleration and early initialization.

Applicability follows existing serving options. Missing optional copy support
keeps the original Index/Scatter implementation usable. Graph scheduling is
delegated to offload_schedule for alias-safe host offload.
"""
import hashlib
import json
import logging
import os
import re
from pathlib import Path

logger = logging.getLogger(__name__)
_copy_provider_checked = False
_copy_operator = None


def _value(args, name, default=None):
    return args.get(name, default) if isinstance(args, dict) else getattr(args, name, default)


def host_offload_enabled(args):
    return (
        _value(args, "device") == "npu"
        and _value(args, "disaggregation_mode") == "decode"
        and bool(_value(args, "offload_cache", False))
        and _value(args, "offload_cache_storage", "host") == "host"
    )


def parse_offload_sfad_bounded_holes(environ=None):
    """Strict startup-only opt-in; never read the environment during replay."""
    values = os.environ if environ is None else environ
    raw = values.get("NPU_OFFLOAD_SFAD_BOUNDED_HOLES", "0")
    if type(raw) is not str or raw not in ("0", "1"):
        raise ValueError("NPU_OFFLOAD_SFAD_BOUNDED_HOLES must be 0 or 1")
    return raw == "1"


def offload_sfad_bounded_holes_config(args, environ=None):
    requested = parse_offload_sfad_bounded_holes(environ)
    applicable = (host_offload_enabled(args)
                  and bool(_value(args, "npu_dsa_kv_int8", False))
                  and _value(args, "attention_backend") == "npu_mla")
    effective = requested and applicable
    frozen = _value(args, "_npu_offload_sfad_bounded_holes")
    if frozen is not None and frozen["effective"] != effective:
        raise RuntimeError("SFAD bounded holes changed after startup cache qualification; restart with matching environment")
    return dict(requested=requested, effective=effective, applicable=applicable)


# Host offload replays GE graphs only under the writeback variant that wrote
# them. Keep this string stable: changing it orphans every cached graph.
GRAPH_CACHE_VARIANT = "host-kv-copy-update-stream-dispatch-v1-alias_safe-ge-opcache-disabled-v1"
_PREVIOUS_GRAPH_CACHE_VARIANTS = (
    GRAPH_CACHE_VARIANT + "-keyrope-followup-v1",
    GRAPH_CACHE_VARIANT + "-keyrope-followup-v1-sfad-selected-v1",
    GRAPH_CACHE_VARIANT + "-keyrope-followup-v1-sfad-full-v1",
)
PACKED_GRAPH_CACHE_VARIANT = "host-kv-packed-int8-lru-v2-row672-postforward-copy-topk-reuse-update8-alias-safe"
PACKED_AFTER_ATTENTION_GRAPH_CACHE_VARIANT = PACKED_GRAPH_CACHE_VARIANT + "-update-after-attention-v1"
PACKED_AFTER_SFAD_GRAPH_CACHE_VARIANT = PACKED_GRAPH_CACHE_VARIANT + "-update-after-sfad-v1"
PACKED_LAST_QUERY_GRAPH_CACHE_VARIANTS = {
    timing: (PACKED_AFTER_SFAD_GRAPH_CACHE_VARIANT if timing == "after_sfad" else PACKED_GRAPH_CACHE_VARIANT)
    + "-last-query-" + timing + "-v1"
    for timing in ("before_attention", "after_attention", "after_sfad")
}
# V1 graphs still contain the in-graph Host publication and reconstruct the
# Update TopK. They cannot be replayed under the V2 post-forward contract.
_PREVIOUS_PACKED_GRAPH_CACHE_VARIANTS = (
    "host-kv-packed-int8-lru-v1-row672-alias-safe",
)


_INDEXER_CACHE_COMPONENT = re.compile(
    r"dsa-indexer-cli2-global-even-v1-(?:c8-row[1-9][0-9]*|split-bf16)"
    r"-p[1-9][0-9]*-pp[1-9][0-9]*-l[0-9a-f]{16}"
)
_BOUNDED_HOLES_CACHE_COMPONENT = "packed-sfad-bounded-holes-v1"


def _indexer_cache_component(*, packed, page_size, packed_row_bytes,
                             pp_size, pp_layer_nums):
    """Use target layout rules, never a draft runner's one-layer config."""
    if type(page_size) is not int or page_size <= 0:
        raise ValueError("Indexer cache identity requires a positive page size")
    if type(pp_size) is not int or pp_size <= 0:
        raise ValueError("Indexer cache identity requires a positive PP size")
    if packed and (type(packed_row_bytes) is not int or packed_row_bytes <= 0):
        raise ValueError("Indexer cache identity requires positive packed row bytes")
    partition = None if pp_layer_nums is None else tuple(pp_layer_nums)
    if pp_size > 1 and partition is None:
        raise ValueError("Indexer placeholders with PP require explicit pp_layer_nums")
    if partition is not None and (
        len(partition) != pp_size
        or any(type(n) is not int or n <= 0 for n in partition)
    ):
        raise ValueError("Indexer cache identity requires one positive layer count per PP stage")
    # The full PP partition is hashed, not merely its length/sum. ServerArgs
    # precedes model config, so encode the CLI2/global-even target mask rule
    # instead of guessing the target layer count from a later draft config.
    pp_layout = json.dumps(
        {"pp_size": pp_size, "pp_layer_nums": partition},
        sort_keys=True, separators=(",", ":"),
    )
    digest = hashlib.sha256(pp_layout.encode()).hexdigest()[:16]
    layout = f"c8-row{packed_row_bytes}" if packed else "split-bf16"
    return (f"dsa-indexer-cli2-global-even-v1-{layout}"
            f"-p{page_size}-pp{pp_size}-l{digest}")


def qualify_graph_cache(path, *, packed=False, update_timing="before_attention",
                        update_policy="all", offload_enabled=True,
                        indexer_placeholders=False, page_size=128,
                        packed_row_bytes=672, pp_size=1, pp_layer_nums=None,
                        sfad_bounded_holes=None):
    """One early root for num-token, target and draft graph caches.

    Existing callers default to the original offload identity. The optional
    target IndexK layout is a separate component, so every filename stays
    bounded. Recognize both managed components on dict reconstruction/toggle;
    never append a second offload identity below an IndexK-qualified path.
    """
    bounded_holes = (parse_offload_sfad_bounded_holes()
                     if sfad_bounded_holes is None else sfad_bounded_holes)
    if type(bounded_holes) is not bool:
        raise ValueError("SFAD bounded holes cache setting must be a bool")
    bounded_holes = bounded_holes and packed and offload_enabled
    if update_policy not in ("all", "last_query"):
        raise ValueError("Unknown offload cache update policy")
    if update_timing not in ("before_attention", "after_attention", "after_sfad"):
        raise ValueError("Unknown offload cache update timing")
    variant = PACKED_GRAPH_CACHE_VARIANT if packed else GRAPH_CACHE_VARIANT
    if packed and update_timing == "after_attention":
        variant = PACKED_AFTER_ATTENTION_GRAPH_CACHE_VARIANT
    if packed and update_timing == "after_sfad":
        variant = PACKED_AFTER_SFAD_GRAPH_CACHE_VARIANT
    if packed and update_policy == "last_query":
        variant = PACKED_LAST_QUERY_GRAPH_CACHE_VARIANTS[update_timing]
    if not packed and (update_policy != "all" or update_timing != "before_attention"):
        variant = f"{GRAPH_CACHE_VARIANT}-split-bf16-{update_policy}-{update_timing}"
    if not path:
        return path
    known_variants = (GRAPH_CACHE_VARIANT, PACKED_GRAPH_CACHE_VARIANT,
                      PACKED_AFTER_ATTENTION_GRAPH_CACHE_VARIANT,
                      PACKED_AFTER_SFAD_GRAPH_CACHE_VARIANT,
                      *PACKED_LAST_QUERY_GRAPH_CACHE_VARIANTS.values(),
                      *_PREVIOUS_GRAPH_CACHE_VARIANTS,
                      *_PREVIOUS_PACKED_GRAPH_CACHE_VARIANTS)
    base = Path(path)
    had_managed_identity = False
    packed_variants = (PACKED_GRAPH_CACHE_VARIANT, PACKED_AFTER_ATTENTION_GRAPH_CACHE_VARIANT,
                       PACKED_AFTER_SFAD_GRAPH_CACHE_VARIANT,
                       *PACKED_LAST_QUERY_GRAPH_CACHE_VARIANTS.values(),
                       *_PREVIOUS_PACKED_GRAPH_CACHE_VARIANTS)
    # The optional suffix belongs only to a managed packed graph identity.
    if (base.name == _BOUNDED_HOLES_CACHE_COMPONENT
            and base.parent.name in packed_variants):
        base = base.parent
        had_managed_identity = True
    while True:
        if (base.name in known_variants
                or base.name.startswith(GRAPH_CACHE_VARIANT + "-split-bf16-")
                or _INDEXER_CACHE_COMPONENT.fullmatch(base.name)):
            had_managed_identity = True
            base = base.parent
        else:
            break
    if not offload_enabled and not indexer_placeholders and not had_managed_identity:
        return path
    result = str(base)
    if indexer_placeholders:
        result = os.path.join(result, _indexer_cache_component(
            packed=packed, page_size=page_size, packed_row_bytes=packed_row_bytes,
            pp_size=pp_size, pp_layer_nums=pp_layer_nums,
        ))
    if offload_enabled:
        result = os.path.join(result, variant)
        if bounded_holes:
            result = os.path.join(result, _BOUNDED_HOLES_CACHE_COMPONENT)
    return result


def resolve_host_offload_defaults(args):
    """Fill omitted tuning values without changing the non-offload defaults."""
    enabled = host_offload_enabled(args)
    if args.offload_cache_workspace_mb is None:
        args.offload_cache_workspace_mb = 384 if enabled else 256


def _missing_copy_provider(reason):
    global _copy_provider_checked, _copy_operator
    _copy_operator = None
    _copy_provider_checked = True
    logger.warning(
        "Optional host-KV fused copy is unavailable (%s); using the original "
        "Index/Scatter current-row path. Existing offload remains enabled.", reason,
    )
    return None


def get_host_copy_operator():
    """Resolve optional interfaces once, without hiding a broken native package.

    Missing flash_ops, absent copy schema or unregistered native/Meta kernels
    select the legacy path. ABI import failures, missing dependencies inside
    flash_ops, and unexpected dispatcher errors still abort normal startup.
    This interface check does not establish native correctness/performance.
    """
    global _copy_provider_checked, _copy_operator
    if _copy_provider_checked:
        return _copy_operator
    import torch
    try:
        import flash_ops  # noqa: F401
    except ModuleNotFoundError as error:
        if error.name != "flash_ops":
            raise
        return _missing_copy_provider("flash_ops is not installed")

    namespace = torch.ops.custom
    try:
        operator = namespace.npu_kv_cache_copy_rows_
    except AttributeError:
        return _missing_copy_provider("custom::npu_kv_cache_copy_rows_ schema is absent")

    name = "custom::npu_kv_cache_copy_rows_"
    for key in ("PrivateUse1", "Meta"):
        if not torch._C._dispatch_has_kernel_for_dispatch_key(name, key):
            return _missing_copy_provider(f"Missing {name} {key}")
    _copy_operator = operator
    _copy_provider_checked = True
    return _copy_operator


def initialize_host_copy_provider(args, is_draft_worker=False):
    """Early interface check; model-specific backing checks remain at the caller."""
    if not is_draft_worker and host_offload_enabled(args):
        operator = get_host_copy_operator()
        if operator is not None:
            from sglang.srt.npu.offload_schedule import install
            install()
        return operator
    return None


class _HostLayerCopyBurst:
    """Frozen eager-only target capability; all cache storage stays pool-owned."""

    def __init__(self, buffers, operator):
        self.caches = tuple(t.view(-1, 672) for t in buffers)
        self.operator = operator
        self._last_indices = None

    def copy(self, source, destination):
        """Caller-stream FIFO only; False authorizes the complete legacy loop."""
        import torch

        if source.numel() == 0:
            return True
        if source.numel() > 256:
            # This policy is announced at startup. Copy ALL Host layers via
            # the pre-existing large-N path; never truncate a TensorList.
            return False
        # Metadata may otherwise be reused by the next forward. These private
        # copies are enqueued on the same caller stream before the one burst.
        private_source = source.clone().contiguous()
        private_destination = destination.clone().contiguous()
        stream = torch.npu.current_stream(self.caches[0].device)
        private_source.record_stream(stream)
        private_destination.record_stream(stream)
        self._last_indices = (private_source, private_destination)
        self.operator(self.caches, private_source, private_destination)
        return True


def initialize_layer_copy_burst(model_runner):
    """Opt in only after target-pool creation and before any graph capture.

    The package validator checks extension/OPP geometry, complete mutation
    schema/backends, sealed artifacts and actual aclnn provider resolution.
    OFF/P/draft/non-Host workers perform no burst and need no optional package.
    """
    model_runner.layer_copy_burst_runtime = None
    requested = os.environ.get("NPU_OFFLOAD_COPY_BURST", "0")
    if requested not in ("0", "1"):
        raise ValueError("NPU_OFFLOAD_COPY_BURST must be0 or1")
    if requested == "0" or model_runner.is_draft_worker:
        return None
    args = model_runner.server_args
    if not host_offload_enabled(args):
        return None
    import importlib

    import torch

    pool = model_runner.token_to_kv_pool
    config = getattr(model_runner.model_config, "hf_text_config", None)
    if not (
        _value(args, "npu_dsa_kv_int8", False)
        and _value(args, "npu_dsa_packed_kv_row_bytes") == 672
        and _value(args, "pp_size") == 1 and _value(args, "kvp_size") == 8
        and _value(args, "attention_backend") == "npu_mla"
        and _value(args, "npu_disable_kv_nz", False)
        and model_runner.dtype == torch.bfloat16
        and getattr(config, "cli_factor", 1) == 2
        and getattr(pool, "dsa_kv_int8_packed", False)
        and getattr(pool, "packed_host_offload", False)
        and getattr(pool, "page_size", None) == 128
        and getattr(pool, "kv_cache_dim", None) == 672
        and getattr(pool, "layer_num", None) == 76
        and getattr(pool, "start_layer", None) == 0
    ):
        raise ValueError("Layer-copy burst requires target packed Host BF16-activation CLI2/KVP8/page128/row672,PP1 Decode")
    getter = getattr(pool, "get_buffer_device_types", None)
    kinds = getter() if callable(getter) else None
    host_ids = tuple(i for i in range(76) if kinds is not None and kinds[i] == "cpu") if kinds is not None and len(kinds) >= 76 else ()
    if (kinds is None or len(kinds) != 152
            or host_ids != tuple(range(1, 76, 2))
            or any(kind != "npu" for kind in kinds[76:])):
        raise ValueError("Layer-copy burst requires all38 Host KV buffers and unchanged device IndexK registration")
    if len(pool.kv_buffer) != 76:
        raise ValueError("Layer-copy burst target KV tensor list is incomplete")
    buffers = tuple(pool.kv_buffer[i] for i in host_ids)
    shape = tuple(buffers[0].shape)
    if (len(shape) != 4 or shape[1:] != (128, 1, 672)
            or not 0 < shape[0] * 128 <= (1 << 31) - 1
            or buffers[0].device.type != "npu"
            or any(t.dtype != torch.int8 or tuple(t.shape) != shape
                   or not t.is_contiguous() or t.device != buffers[0].device
                   for t in buffers)):
        raise ValueError("Layer-copy burst Host caches must have equal contiguous C8 geometry")
    regions = sorted((t.data_ptr(), t.data_ptr() + t.nbytes) for t in buffers)
    if (any(begin <= 0 or begin % 32 for begin, end in regions)
            or any(a[1] > b[0] for a, b in zip(regions, regions[1:]))):
        raise ValueError("Layer-copy burst Host cache regions must be aligned and nonoverlapping")
    try:
        module = importlib.import_module("fluent_sfa1_ops")
    except ModuleNotFoundError as error:
        if error.name != "fluent_sfa1_ops":
            raise
        raise RuntimeError("NPU_OFFLOAD_COPY_BURST=1 requires the paired layer-burst operator package") from error
    validator = getattr(module, "validate_layer_copy_burst", None)
    if (getattr(module, "LAYER_COPY_BURST_ABI_VERSION", None) != 1
            or not callable(validator)):
        raise RuntimeError("Layer-copy burst extension ABI1/paired-provider validator is unavailable")
    receipt = validator()
    expected = dict(layer_copy_burst_abi_version=1, layer_copy_burst_row_bytes=672,
                    layer_copy_burst_blocks=4, layer_copy_burst_max_layers=38,
                    layer_copy_burst_max_rows=256)
    if (not isinstance(receipt, dict)
            or any(type(receipt.get(k)) is not int or receipt[k] != v for k, v in expected.items())
            or not receipt.get("marker") or not isinstance(receipt.get("provider"), dict)
            or set(receipt["provider"]) != {"aclnnKvCacheCopyRowsBurst", "aclnnKvCacheCopyRowsBurstGetWorkspaceSize"}):
        raise RuntimeError("Layer-copy burst validator returned an incompatible capability receipt")
    try:
        operator = module.ops.npu_kv_cache_copy_rows_burst_.default
    except AttributeError as error:
        raise RuntimeError("Layer-copy burst TensorList public operator is missing") from error
    schema = torch._C.parse_schema(
        "fluent_sfa1::npu_kv_cache_copy_rows_burst_(Tensor(a!)[] caches, Tensor src_indices, Tensor dst_indices) -> ()"
    )
    if str(operator._schema) != str(schema):
        raise RuntimeError("Layer-copy burst public TensorList/mutation schema changed")
    target = getattr(model_runner.model, "language_model", model_runner.model)
    if not callable(getattr(target, "fix_kvp_current_kv_cache", None)):
        raise ValueError("Layer-copy burst target does not expose the existing post-forward Host writer")
    capability = _HostLayerCopyBurst(buffers, operator)
    target._npu_offload_copy_burst = capability
    model_runner.layer_copy_burst_runtime = dict(
        **receipt, target_host_layers=38, enabled=True,
        fallback_policy="empty:no-op;N>256:complete legacy Host loop",
    )
    logger.info("NPU_LAYER_COPY_BURST %s", model_runner.layer_copy_burst_runtime)
    return capability
