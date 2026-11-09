"""CPU contracts for the default dual-namespace extension build.

Run directly with Python; these tests do not import torch or require an NPU.
They exercise setup.py and converter selection using dependency stubs.
"""

import __future__
import ast
import os
from pathlib import Path
import re
import runpy
import subprocess
import sys
import types
import tempfile
import unittest
from unittest.mock import patch


EXT = Path(__file__).resolve().parents[1] / 'torch_ops_extension'
OPS = EXT / 'custom_ops'
PRIVATE = EXT / 'fluent_sfa1_ops'


def extension_config(**environment):
    captured = {}
    modules = {}
    for name in ('torch', 'torch.utils', 'torch.utils.cpp_extension',
                 'torch_npu', 'torch_npu.utils', 'torch_npu.utils.cpp_extension', 'setuptools', 'setuptools.command', 'setuptools.command.build_py'):
        modules[name] = types.ModuleType(name)
    modules['torch_npu'].__file__ = '/stub/torch_npu/__init__.py'
    modules['setuptools.command.build_py'].build_py = object
    modules['setuptools'].setup = lambda **kwargs: captured.update(kwargs)
    modules['torch_npu.utils.cpp_extension'].NpuExtension = lambda **kwargs: types.SimpleNamespace(**kwargs)
    modules['torch.utils.cpp_extension'].BuildExtension = types.SimpleNamespace(with_options=lambda **kwargs: object)
    values = dict(os.environ, FLASH_OPS_MINIMAL='0', FLASH_OPS_SFA1_RUNTIME='0')
    values.update(environment)
    with patch.dict(sys.modules, modules), patch.dict(os.environ, values, clear=True):
        runpy.run_path(str(EXT / 'setup.py'), run_name='__profile_test__')
    return captured


def active_schemas(path, minimal=False):
    text = path.read_text()
    text = re.sub(r'/\*.*?\*/|//[^\n]*', '', text, flags=re.S)
    text = re.sub(r'^\s*#include[^\n]*$', '', text, flags=re.M)
    command = ['cpp', '-P', '-']
    if minimal:
        command.insert(1, '-DFLASH_OPS_MINIMAL')
    text = subprocess.check_output(command, input=text, text=True)
    schemas = {}
    for statement in re.findall(r'm\.def\((.*?)\);', text, re.S):
        schema = ''.join(ast.literal_eval(s) for s in re.findall(r'"(?:\\.|[^"\\])*"', statement))
        name = schema.split('(', 1)[0]
        if name in schemas:
            raise AssertionError(f'duplicate schema {name}')
        schemas[name] = schema
    return schemas


def converter_function(path, name, **dependencies):
    tree = ast.parse(path.read_text())
    if not any(isinstance(n, ast.FunctionDef) and n.name == name for n in tree.body):
        path = OPS / "offload" / path.name
        tree = ast.parse(path.read_text())
    function = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
    function.decorator_list = []
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    scope = dict(dependencies)
    exec(compile(module, str(path), 'exec', flags=__future__.annotations.compiler_flag), scope)
    return scope[name]


class BuildProfileTests(unittest.TestCase):
    def test_default_always_builds_both_namespaces(self):
        for environment in ({}, {"FLASH_OPS_SFA1_RUNTIME": "1"}, {"FLASH_OPS_MINIMAL": "1"}):
            config = extension_config(**environment)
            self.assertEqual([e.name for e in config["ext_modules"]],
                             ["flash_ops.custom_ops_lib", "fluent_sfa1_ops._C"])
            self.assertIn("fluent_sfa1_ops.converter", config["packages"])
            for extension in config["ext_modules"]:
                self.assertTrue(all(Path(path).is_file() for path in extension.sources))

    def test_asc_basic_api_consumer_gets_runtime_sdk_include(self):
        helper = runpy.run_path(str(EXT.parent / "tools/add_asc_std_include.py"))
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = "import os\ndef options(tikcpp_path):\n    options = []\n" + helper["ANCHOR"] + "    return options\n"
            gating = root / "moe_gating_top_k_expert_map.py"
            sfa = root / "fluent_sparse_flash_attention.py"
            gating.write_text(source)
            sfa.write_text(source)
            helper["patch"](root)
            sdk = root / "sdk"
            (sdk / "asc/include/utils/std").mkdir(parents=True)
            (sdk / "asc/include/utils/std/algorithm.h").touch()
            scope = {}
            exec(gating.read_text(), scope)
            self.assertIn("-I" + str(sdk / "asc/include"), scope["options"](str(sdk / "tikcpp")))
            self.assertEqual(sfa.read_text(), source)
            patched = gating.read_bytes()
            helper["patch"](root)
            self.assertEqual(gating.read_bytes(), patched)

    def test_four_prolog_contracts_and_cache_mutation(self):
        for root, ordinary, functional, keyword in (
            (OPS, 5, 7, "enable_rope=True"),
            (PRIVATE, 7, 9, "output_cur_k_nope_rope=False"),
        ):
            schemas = active_schemas(root / "csrc/ops_def_registration.cpp")
            for name, count in [("npu_mla_prolog_v3", ordinary), ("npu_mla_prolog_v3_functional", functional)]:
                self.assertEqual(schemas[name].split("->")[1].count("Tensor"), count)
                self.assertIn(keyword, schemas[name])
            self.assertIn("Tensor(a!) kv_cache", schemas["npu_mla_prolog_v3"])
            self.assertNotIn("Tensor(a!)", schemas["npu_mla_prolog_v3_functional"])

    def test_every_selected_implementation_has_a_schema(self):
        config = extension_config()
        for root, extension in zip((OPS, PRIVATE), config["ext_modules"]):
            schemas = active_schemas(root / "csrc/ops_def_registration.cpp")
            for path in extension.sources:
                if "TORCH_LIBRARY_FRAGMENT" in Path(path).read_text():
                    schemas.update(active_schemas(Path(path)))
            for path in extension.sources:
                for name in re.findall(r'm\.impl\("([^\"]+)"', Path(path).read_text()):
                    self.assertIn(name, schemas, path)

    def test_legacy_loader_preserves_native_aliases(self):
        tree = ast.parse((OPS / "__init__.py").read_text())
        node = next(n for n in tree.body if isinstance(n, ast.If) and
                    isinstance(n.test, ast.Compare) and isinstance(n.test.left, ast.Name) and
                    n.test.left.id == "custom_ops_module")
        native = {name: object() for name in (
            "compute_n_gram_ids", "npu_compute_n_gram_ids",
            "npu_moe_distribute_dispatch_v2", "npu_moe_distribute_combine_v2",
            "npu_lightning_indexer", "mlp_lightning_indexer",
        )}
        custom = types.SimpleNamespace(**{name: object() for name in native}, npu_mla_prolog_v3=object())
        pta = types.SimpleNamespace(**native)
        scope = dict(custom_ops_module=custom, torch_npu=pta)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), "loader", "exec"), scope)
        self.assertTrue(all(getattr(pta, k) is value for k, value in native.items()))
        self.assertIs(pta.npu_mla_prolog_v3, custom.npu_mla_prolog_v3)

    def test_update_completion_depends_on_real_outputs(self):
        fake_ge = types.SimpleNamespace(
            Const=lambda *a, **k: "zero",
            Identity=lambda value, dependencies: (value, tuple(dependencies)),
        )
        f = converter_function(PRIVATE / "converter/npu_update_cur_kv_to_selection.py",
            "convert_npu_update_cur_kv_to_selection_functional", ge=fake_ge,
            DataType=types.SimpleNamespace(DT_INT32=1),
            torchair=types.SimpleNamespace(ge=types.SimpleNamespace(
                custom_op=lambda *a, **k: tuple(k["outputs"]))))
        result = f(*range(6))
        self.assertEqual(result, (("zero", ("selection_k_nope",)),
                                  "selection_k_nope", "selection_k_rope"))
        ordinary = converter_function(PRIVATE / "converter/npu_update_cur_kv_to_selection.py",
            "convert_npu_update_cur_kv_to_selection", ge=fake_ge,
            DataType=types.SimpleNamespace(DT_INT32=1),
            torchair=types.SimpleNamespace(ge=types.SimpleNamespace(
                custom_op=lambda *a, **k: tuple(k["outputs"]))))
        self.assertEqual(ordinary(*range(6)), ("zero", ("selection_k_nope",)))

    def test_ge_converters_keep_distinct_operator_contracts(self):
        attrs = types.SimpleNamespace(**{name: lambda value: value for name in ('Bool', 'Int', 'Float', 'Str')})
        calls = []
        def custom_op(name, **kwargs):
            calls.append(dict(name=name, **kwargs))
            # GE returns one Tensor per declared output. The MLA converter
            # unpacks the nine GE outputs and exposes seven Torch outputs.
            return tuple(kwargs['outputs'])
        torchair = types.SimpleNamespace(ge=types.SimpleNamespace(custom_op=custom_op))
        mla = converter_function(PRIVATE / 'converter/npu_mla_prolog_v3.py',
                                 'convert_npu_npu_mla_prolog_v3', torchair=torchair, attr=attrs)
        result = mla(*range(11), output_cur_k_nope_rope=True)
        self.assertEqual(calls[-1]['name'], 'FluentMlaPrologV3')
        self.assertEqual(calls[-1]['outputs'],
                         ['query', 'query_rope', 'kv_cache', 'kr_cache', 'dequant_scale_q_nope',
                          'query_norm', 'dequant_scale_q_norm', 'cur_k_nope', 'cur_k_rope'])
        self.assertEqual(result, ('query', 'query_rope', 'dequant_scale_q_nope',
                                  'query_norm', 'dequant_scale_q_norm', 'cur_k_nope', 'cur_k_rope'))
        self.assertTrue(calls[-1]['attrs']['output_cur_k_nope_rope'])
        for root, op_name in ((OPS, 'SparseFlashAttention'),
                              (PRIVATE, 'FluentSparseFlashAttention')):
            sfa = converter_function(root / 'converter/npu_sparse_flash_attention.py',
                                     'convert_npu_sparse_flash_attention', torchair=torchair, attr=attrs)
            result = sfa(*range(5), return_softmax_lse=True)
            self.assertEqual(calls[-1]['name'], op_name)
            self.assertEqual(result, ('attention_out', 'softmax_max', 'softmax_sum'))
            self.assertTrue(calls[-1]['attrs']['return_softmax_lse'])


if __name__ == '__main__':
    unittest.main()
