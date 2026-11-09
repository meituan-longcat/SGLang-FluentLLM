"""Pair FLASH Update/Dispatch by FX module scope before adding GE controls.

This adapter is limited to alias-safe host offload, one Update and one Dispatch
per ParallelShortcutTransformerLayer. Missing or ambiguous scope fails closed.
No GE-node ordinal or graph-wide fixed layer count is used. GE itself rejects a
cyclic graph, so this does not re-derive the topology.
"""
import collections
import functools
import json
import logging
import os
import re
import time
from collections.abc import Mapping
from contextlib import contextmanager

logger = logging.getLogger(__name__)
_installed = False
UPDATE = 'UpdateCache'
UPDATE_TYPES = frozenset((UPDATE, 'UpdateCachePacked'))
DISPATCH = 'MoeDistributeDispatchV2'


def layer_scope(node):
    stack = node.meta.get('nn_module_stack', {})
    if not isinstance(stack, Mapping):
        raise RuntimeError('Missing FX nn_module_stack mapping')
    matches = {}
    for value in stack.values():
        if not isinstance(value, (tuple, list)) or len(value) < 2:
            continue
        path, cls = value[:2]
        if (getattr(cls, '__name__', None) != 'ParallelShortcutTransformerLayer'
                or not getattr(cls, '__module__', '').endswith('.npu_longcat_flash')):
            continue
        path = str(path)
        # Dynamo can spell module accesses via their literal _modules keys.
        # Normalize those keys without evaluating a path or accepting indices
        # inferred from GE names. Preserve the original path in the receipt.
        normalized = re.sub(r'''\._modules\[(["'])([A-Za-z_]\w*|\d+)\1\]''', r'.\2', path)
        match = re.search(r'(?:^|\.)layers(?:\.(\d+)|\[(\d+)\])(?:\.|$)', normalized)
        if match is None:
            match = re.search(r'(?:^|_)layers_(\d+)(?:_|$)', normalized)
        if match is None:
            raise RuntimeError(f'Unrecognized FLASH module layer path: {path}')
        index = next(group for group in match.groups() if group is not None)
        matches[int(index)] = path
    if len(matches) != 1:
        raise RuntimeError(f'Expected one semantic FLASH block scope; found {matches}')
    index, path = next(iter(matches.items()))
    return f'FLASH-block-{index}', path


def transform(graph):
    updates = [op for op in graph.op if op.type in UPDATE_TYPES]
    if not updates:
        return {'status': 'UNCHANGED_NO_UPDATE', 'edges': []}
    registry = getattr(graph, '_kv_offload_fx_scopes', {})
    groups = collections.defaultdict(dict)
    for op in graph.op:
        if op.type not in UPDATE_TYPES and op.type != DISPATCH:
            continue
        entry = registry.get(op.name)
        if not entry or entry.get('error') or entry['type'] != op.type:
            raise RuntimeError(f'No verified FX block scope for {op.type} {op.name}: {entry}')
        group = groups[entry['block']]
        role = UPDATE if op.type in UPDATE_TYPES else DISPATCH
        if role in group:
            raise RuntimeError(f'Multiple {op.type} in one FLASH block; unsupported offload layout')
        group[role] = op
    if any(set(group) != {UPDATE, DISPATCH} for group in groups.values()):
        raise RuntimeError('Update and Dispatch do not cover the same semantic FLASH blocks')
    edges = [{'block': block, 'update': group[UPDATE].name,
              'dispatch': group[DISPATCH].name,
              'edge': group[DISPATCH].name + ':-1',
              'update_scope': registry[group[UPDATE].name]['scope'],
              'dispatch_scope': registry[group[DISPATCH].name]['scope']}
             for block, group in sorted(groups.items())]
    if getattr(graph, '_kv_offload_dispatch_controls_applied', None) is not None:
        return {'status': 'ALREADY_ADDED_DISPATCH_UPDATE_CONTROLS', 'edges': edges,
                'semantic_mapping': 'FX nn_module_stack FLASH decoder block'}
    for row in edges:
        groups[row['block']][UPDATE].input.append(row['edge'])
    graph._kv_offload_dispatch_controls_applied = edges
    return {'status': 'ADDED_SEMANTIC_DISPATCH_UPDATE_CONTROLS', 'edges': edges,
            'semantic_mapping': 'FX nn_module_stack FLASH decoder block'}


def _converter_interfaces(converter):
    """Resolve the private entry points this adapter replaces."""
    try:
        cls = converter.GeConcreteGraph
        return cls, cls.converter_context, converter._mapping_assign_op_to_graph_output
    except AttributeError as error:
        location = getattr(converter, '__file__', '<unknown TorchAir module>')
        raise RuntimeError(
            f'Unsupported TorchAir offload scheduling interface in {location}: {error}'
        ) from error


def install():
    global _installed
    if _installed:
        return
    try:
        import torchair._ge_concrete_graph.fx2ge_converter as converter
    except ImportError as error:
        raise RuntimeError('TorchAir FX converter is required for alias-safe offload scheduling') from error
    cls, original_context, original_mapping = _converter_interfaces(converter)

    @contextmanager
    @functools.wraps(original_context)
    def context(owner, *, node):
        with original_context(owner, node=node):
            graph = owner.graph
            start = len(graph.op)
            yield
            new_ops = [op for op in graph.op[start:] if op.type in UPDATE_TYPES or op.type == DISPATCH]
            if not new_ops:
                return
            try:
                block, path = layer_scope(node)
                extra = {'block': block, 'scope': path}
            except RuntimeError as error:
                extra = {'error': str(error), 'meta_keys': sorted(map(str, node.meta))}
            registry = getattr(graph, '_kv_offload_fx_scopes', None)
            if registry is None:
                registry = graph._kv_offload_fx_scopes = {}
            for op in new_ops:
                entry = {'name': op.name, 'type': op.type, **extra}
                previous = registry.get(op.name)
                if not previous or previous.get('error'):
                    registry[op.name] = entry

    @functools.wraps(original_mapping)
    def mapping(graph):
        result = original_mapping(graph)
        receipt = transform(graph)
        if receipt['edges']:
            receipt.update(epoch=time.time(), pid=os.getpid())
            logger.info('NPU_DISPATCH_UPDATE_DEPENDENCY %s', json.dumps(receipt, sort_keys=True))
        return result

    cls.converter_context = context
    converter._mapping_assign_op_to_graph_output = mapping
    _installed = True
