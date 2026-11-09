"""Optional eager-only packed/BF16 layer burst and executable CPU model."""
import hashlib
import json
from pathlib import Path
import torch

SCHEMA = 'npu_kv_cache_copy_rows_burst_(Tensor(a!)[] caches, Tensor src_indices, Tensor dst_indices) -> ()'
GEOMETRY = dict(layer_copy_burst_abi_version=1, layer_copy_burst_row_bytes=672,
                layer_copy_burst_blocks=4, layer_copy_burst_max_layers=38, layer_copy_burst_max_rows=256,
                layer_copy_burst_row_bytes_supported=[672, 1024, 128], layer_copy_burst_bf16_max_rows=128)
_cpu_library = None


def copy_cpu(caches, src_indices, dst_indices):
    """Actual CPU dispatcher backend; reads bytes before any inplace writes."""
    if not 1 <= len(caches) <= 38:
        raise ValueError('Burst requires1..38 caches')
    first = caches[0]
    bf16 = first.dtype == torch.bfloat16
    if (first.ndim != 2 or not 1 <= first.shape[0] <= (1 << 31) - 1
            or not ((first.dtype == torch.int8 and first.shape[1] == 672)
                    or (bf16 and first.shape[1] in (512, 64)))):
        raise ValueError('Burst caches require INT8[R,672] or BF16[R,512/64]')
    if any(t.device.type != 'cpu' or t.device != first.device or t.dtype != first.dtype
           or t.shape != first.shape or not t.is_contiguous() for t in caches):
        raise ValueError('CPU burst requires equal contiguous CPU cache geometry/dtype')
    if (src_indices.ndim != 1 or dst_indices.ndim != 1 or src_indices.numel() != dst_indices.numel()
            or src_indices.numel() > (128 if bf16 else 256) or src_indices.dtype not in (torch.int32, torch.int64)
            or src_indices.dtype != dst_indices.dtype
            or any(t.device != first.device or not t.is_contiguous() for t in (src_indices, dst_indices))):
        raise ValueError('Burst requires shared contiguous CPU INT32/64 indices,N<=128 for BF16 or256 for INT8')
    ranges = [(t.data_ptr(), t.data_ptr() + t.numel() * t.element_size()) for t in caches]
    overlap = lambda a,b: a[0] < a[1] and b[0] < b[1] and a[0] < b[1] and b[0] < a[1]
    if any(a[0] % 32 or any(overlap(a,b) for b in ranges[:i]) for i,a in enumerate(ranges)):
        raise ValueError('Burst layers require aligned disjoint payload ranges')
    for tensor in (src_indices, dst_indices):
        region = (tensor.data_ptr(), tensor.data_ptr() + tensor.numel() * tensor.element_size())
        if (tensor.numel() and region[0] % 32) or any(overlap(region, cache) for cache in ranges):
            raise ValueError('Burst indices require aligned storage outside cache payload')
    source, destination = src_indices.tolist(), dst_indices.tolist()
    positions = [i for i,s in enumerate(source) if 0 <= s < first.shape[0]]
    ids = torch.tensor([source[i] for i in positions], dtype=torch.int64)
    for cache in caches:
        snapshot = cache.index_select(0, ids)
        for saved,i in enumerate(positions):
            if 0 <= destination[i] < first.shape[0]:
                cache[destination[i]].copy_(snapshot[saved])


def register_cpu(namespace='fluent_sfa1'):
    global _cpu_library
    library = torch.library.Library(namespace, 'IMPL', 'CPU')
    library.impl('npu_kv_cache_copy_rows_burst_', copy_cpu)
    if namespace == 'fluent_sfa1':
        _cpu_library = library
    return library


def validate_pair(extension, provider):
    if getattr(extension, 'LAYER_COPY_BURST_ABI_VERSION', 0) != 1:
        raise RuntimeError('Layer-copy burst extension ABI1 is not available')
    name = 'fluent_sfa1::npu_kv_cache_copy_rows_burst_'
    operator = torch.ops.fluent_sfa1.npu_kv_cache_copy_rows_burst_.default
    expected = torch._C.parse_schema('fluent_sfa1::' + SCHEMA)
    if str(operator._schema) != str(expected):
        raise RuntimeError('Layer-copy burst schema/mutation contract changed')
    for key in ('PrivateUse1', 'Meta', 'CPU'):
        if not torch._C._dispatch_has_kernel_for_dispatch_key(name, key):
            raise RuntimeError('Layer-copy burst missing dispatcher backend: ' + key)
    marker = Path(provider['marker']).resolve(strict=True)
    vendor = Path(provider['vendor']).resolve(strict=True)
    data = json.loads(marker.read_text())
    if any(data.get(key) != value for key,value in GEOMETRY.items()):
        raise RuntimeError('Layer-copy burst OPP capability/geometry does not match ABI1')
    api = Path(provider['provider']).resolve(strict=True)
    files = data.get('files', {})
    if files.get('op_api/lib/libcust_opapi.so') != hashlib.sha256(api.read_bytes()).hexdigest():
        raise RuntimeError('Layer-copy burst OPP manifest does not bind actual API provider')
    resolved = {}
    for suffix in ('', 'GetWorkspaceSize'):
        symbol = 'aclnnKvCacheCopyRowsBurst' + suffix
        actual = Path(extension._resolve_provider(symbol)).resolve(strict=True)
        if actual != api:
            raise RuntimeError('Layer-copy burst API resolves to another provider: ' + symbol)
        resolved[symbol] = str(actual)
    required = ['op_impl/ai_core/tbe/flash_ops_impl/dynamic/kv_cache_copy_rows_burst.py',
                'op_impl/ai_core/tbe/kernel/config/ascend910b/kv_cache_copy_rows_burst.json',
                'op_impl/ai_core/tbe/flash_ops_impl/ascendc/kv_cache_copy_rows_burst/kv_cache_copy_rows_burst.cpp']
    objects = [name for name in files if name.startswith('op_impl/ai_core/tbe/kernel/ascend910b/kv_cache_copy_rows_burst/') and name.endswith('.o')]
    if not objects or any(name not in files for name in required):
        raise RuntimeError('Layer-copy burst compiled/JIT/dynamic/config manifest is incomplete')
    for name in required + objects:
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts:
            raise RuntimeError('Layer-copy burst manifest path escapes vendor')
        path = vendor / relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != files[name]:
            raise RuntimeError('Layer-copy burst installed artifact hash differs: ' + name)
    return dict(**GEOMETRY, marker=str(marker), provider=resolved)
