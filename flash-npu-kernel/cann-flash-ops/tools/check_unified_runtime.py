"""Check installed dual ABI/provider on CPU; --with-ge adds an owned device context for IR checks."""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--order", choices=("legacy-first", "sfa1-first"), required=True)
    parser.add_argument("--with-ge", action="store_true", help="Explicit initialized GE check; requires an owned visible NPU, never used during CPU image builds")
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    report = {"status": "RUNNING", "order": args.order, "NPU_execution": "NOT_RUN",
              "GE_device_provider": "PENDING_DEVICE_VALIDATION", "checks": []}
    try:
        if not args.with_ge and os.environ.get("ASCEND_RT_VISIBLE_DEVICES") != "":
            raise RuntimeError("This CPU-only check requires hidden NPU devices")
        if args.with_ge and not os.environ.get("ASCEND_RT_VISIBLE_DEVICES"):
            raise RuntimeError("--with-ge requires an explicitly reserved visible NPU")
        import torch
        import torch_npu
        assert not torch_npu.npu.is_initialized()
        protected = ("compute_n_gram_ids", "npu_compute_n_gram_ids",
                     "npu_moe_distribute_dispatch_v2", "npu_moe_distribute_combine_v2",
                     "npu_lightning_indexer", "mlp_lightning_indexer")
        before = {name: getattr(torch_npu, name, None) for name in protected}
        for module in (("flash_ops", "fluent_sfa1_ops") if args.order == "legacy-first"
                       else ("fluent_sfa1_ops", "flash_ops")):
            importlib.import_module(module)
        import flash_ops
        import fluent_sfa1_ops
        from importlib import metadata
        distribution = metadata.distribution("flash_ops")
        assert distribution.version == "1.0", ("Wrong installed distribution", distribution.version)
        report["wheel_version"] = distribution.version
        # Match master: flash_ops publishes its custom wrappers. Importing the
        # private ABI must leave those bindings in place in either import order.
        exported = set(dir(torch.ops.custom))
        for name in protected:
            expected = getattr(torch.ops.custom, name) if name in exported else before[name]
            assert getattr(torch_npu, name, None) is expected, ("Master custom alias changed", name)
        report["master_custom_aliases"] = "PASS"
        for module in (flash_ops, fluent_sfa1_ops):
            assert Path(module.__file__).resolve().is_relative_to(args.wheel_root.resolve())
        extensions = (flash_ops.custom_ops_lib, fluent_sfa1_ops._C)
        report["extensions"] = [{"path": module.__file__, "sha256": hashlib.sha256(
            Path(module.__file__).read_bytes()).hexdigest()} for module in extensions]
        for namespace, counts, keyword in ((torch.ops.custom, (5, 7), "enable_rope"),
                                          (torch.ops.fluent_sfa1, (7, 9), "output_cur_k_nope_rope")):
            for suffix, count in zip(("", "_functional"), counts):
                op = getattr(namespace, "npu_mla_prolog_v3" + suffix).default
                schema = op._schema
                assert len(schema.returns) == count
                assert keyword in {arg.name for arg in schema.arguments}
                for name in ("kv_cache", "kr_cache"):
                    argument = next(arg for arg in schema.arguments if arg.name == name)
                    if suffix:
                        assert argument.alias_info is None
                    else:
                        assert argument.alias_info is not None and argument.alias_info.is_write
                report["checks"].append({"schema": str(schema), "mutation_contract": "PASS"})

        meta = lambda shape, dtype=torch.bfloat16: torch.empty(shape, dtype=dtype, device="meta")
        ngram = torch.ops.custom.npu_compute_n_gram_ids.default
        assert len(ngram._schema.returns) == 1
        for key in ("PrivateUse1", "Meta"):
            assert torch._C._dispatch_has_kernel_for_dispatch_key(
                "custom::npu_compute_n_gram_ids", key
            ), ("NGram implementation missing", key)
        ngram_result = ngram(
            meta((3,), torch.int64), meta((4,), torch.int64), meta((4,), torch.int64),
            meta((4,), torch.int32), meta((3,), torch.int64), meta((2, 16), torch.int32),
            meta((2,), torch.int64), meta((2,), torch.int32),
            batch_size=2, oe_n=3, oe_k=2, max_context_len=16, eos_token_id=2,
        )
        assert ngram_result.shape == (4, 4) and ngram_result.dtype == torch.int64
        report["checks"].append({"NGram_custom_schema": str(ngram._schema),
            "native_and_Meta_registered": True, "Meta_output": [4, 4],
            "legacy_alias_present": callable(before["compute_n_gram_ids"])})
        inputs = [meta((1, 1, 7168)), meta((7168, 1536)), meta((1536, 12288)),
                  meta((64, 128, 512)), meta((7168, 576)), meta((1536,)), meta((512,)),
                  meta((1, 1, 64)), meta((1, 1, 64)), meta((8, 128, 1, 512)),
                  meta((8, 128, 1, 64))]
        for namespace, counts, current in ((torch.ops.custom, (5, 7), False),
                                           (torch.ops.fluent_sfa1, (7, 9), True)):
            options = {"query_norm_flag": True}
            if current:
                options["output_cur_k_nope_rope"] = True
            ordinary = namespace.npu_mla_prolog_v3(*inputs, **options)
            functional = namespace.npu_mla_prolog_v3_functional(*inputs, **options)
            assert len(ordinary) == counts[0] and len(functional) == counts[1]
            expected_shapes = ((1, 1, 64, 512), (1, 1, 64, 64), (1,), (1, 1, 1536), (1,))
            expected_dtypes = (torch.bfloat16, torch.bfloat16, torch.float32, torch.bfloat16, torch.float32)
            for values in (ordinary, functional):
                assert tuple(tuple(t.shape) for t in values[:5]) == expected_shapes
                assert tuple(t.dtype for t in values[:5]) == expected_dtypes
            report["checks"].append({"Meta_output_signature": str(namespace),
                "ordinary": [{"shape": list(t.shape), "dtype": str(t.dtype)} for t in ordinary],
                "functional": [{"shape": list(t.shape), "dtype": str(t.dtype)} for t in functional]})
            assert functional[-2].shape == inputs[-2].shape and functional[-2] is not inputs[-2]
            assert functional[-1].shape == inputs[-1].shape and functional[-1] is not inputs[-1]
            if current:
                assert ordinary[-2].shape == (1, 1, 1, 512)
                assert ordinary[-1].shape == (1, 1, 1, 64)
                assert functional[5].shape == ordinary[5].shape
                assert functional[6].shape == ordinary[6].shape
                disabled = namespace.npu_mla_prolog_v3(*inputs, output_cur_k_nope_rope=False)
                assert disabled[-2].shape == disabled[-1].shape == (1,)
            transformed = torch.func.functionalize(
                lambda *values: namespace.npu_mla_prolog_v3(*values, **options))(*inputs)
            assert len(transformed) == counts[0]
            report["checks"].append({"meta_and_functionalization": str(namespace), "status": "PASS"})
        update_inputs = [meta((1, 1, 512)), meta((1, 1, 64)), meta((1, 2), torch.int32),
                         meta((16, 128, 1, 512)), meta((16, 128, 1, 64)), meta((1, 16), torch.int32)]
        for namespace in (torch.ops.fluent_sfa1,):
            op = namespace.npu_update_cur_kv_to_selection.default
            assert len(op._schema.returns) == 1
            for name in ("selection_k_nope", "selection_k_rope"):
                argument = next(a for a in op._schema.arguments if a.name == name)
                assert argument.alias_info is not None and argument.alias_info.is_write
            done = op(*update_inputs)
            assert done.shape == (1,) and done.dtype == torch.int32
            functional = namespace.npu_update_cur_kv_to_selection_functional(*update_inputs)
            assert len(functional) == 3
            assert functional[1].shape == update_inputs[3].shape and functional[1] is not update_inputs[3]
            assert functional[2].shape == update_inputs[4].shape and functional[2] is not update_inputs[4]
            transformed = torch.func.functionalize(lambda *values: op(*values))(*update_inputs)
            assert transformed.shape == (1,) and transformed.dtype == torch.int32
            report["checks"].append({"UpdateCur_Meta_functionalization": str(namespace),
                                      "ordinary_returns": 1, "functional_returns": 3, "status": "PASS"})
        cache = meta((4096, 512)); indices = meta((4,), torch.int64)
        assert torch.ops.custom.npu_kv_cache_copy_rows_(cache, indices, indices) is cache
        copied = torch.ops.custom.npu_kv_cache_copy_rows(cache, indices, indices)
        assert copied is not cache and copied.shape == cache.shape
        for name in ("npu_lightning_indexer", "mlp_lightning_indexer"):
            result = getattr(torch.ops.custom, name)(meta((4, 64, 128)), meta((100, 1, 128)),
                meta((4, 64)), layout_query="TND", layout_key="TND")
            assert len(result) == 2 and result[0].shape == (4, 1, 2048)

        provider = Path(fluent_sfa1_ops.PROVIDER["provider"]).resolve()
        for name, selected in fluent_sfa1_ops.PROVIDER["operators"].items():
            actual = Path(extensions[1]._resolve_provider(name)).resolve()
            assert actual == provider == Path(selected).resolve(), (name, actual, provider)
            report["checks"].append({"private_offload_resolver": name, "provider": str(actual)})
        report["provider"] = dict(fluent_sfa1_ops.PROVIDER, sha256=hashlib.sha256(provider.read_bytes()).hexdigest())

        if args.with_ge:
            # Reuse the previously audited initialization route. current_device
            # initializes a real context, so this is NEVER a CPU build action.
            torch_npu.npu.set_deterministic_level(2)
            torch_npu.npu.set_device(args.device)
            from torchair.core._backend import initialize_graph_engine
            initialize_graph_engine()
            report["NPU_execution"] = "CONTEXT_INITIALIZED_FOR_GE_IR_ONLY_NO_MODEL_OR_KERNEL_RUN"
            report["GE_device_provider"] = "GE_IR_REGISTRY_AND_GRAPH_CHECK_NOT_KERNEL_NUMERICS"
        # Import validates prototype bytes but does not preload them. Report
        # normal runtime mappings without requiring an import-side global load.
        proto_rows = []
        maps = Path("/proc/self/maps").read_text().splitlines()
        for proto in fluent_sfa1_ops.PROVIDER["op_proto"]:
            path = Path(proto["path"]).resolve()
            matching = [row for row in maps if row.split()[-1:] == [str(path)]]
            assert hashlib.sha256(path.read_bytes()).hexdigest() == proto["sha256"]
            proto_rows.append({"path": str(path), "sha256": proto["sha256"],
                               "mapped": bool(matching), "explicitly_preloaded_by_offload": False})
        report["op_proto_mappings"] = proto_rows
        from torchair.ge._ge_graph import GeGraph, DataType
        from torchair._ge_concrete_graph import ge_apis as ge
        from torchair._ge_concrete_graph.fx2ge_converter import _get_converter
        assert callable(_get_converter(ngram)), "NGram GE converter missing"
        for package, indices_expected in (("flash_ops", ((0, 1, 4, 5, 6), (0, 1, 4, 5, 6, 2, 3))),
                                           ("fluent_sfa1_ops", ((0, 1, 4, 5, 6, 7, 8), (0, 1, 4, 5, 6, 7, 8, 2, 3)))):
            module = importlib.import_module(package + ".converter.npu_mla_prolog_v3")
            for suffix, expected in zip(("", "_functional"), indices_expected):
                namespace = torch.ops.custom if package == "flash_ops" else torch.ops.fluent_sfa1
                fn = _get_converter(getattr(namespace, "npu_mla_prolog_v3" + suffix).default)
                assert callable(fn)
                if not args.with_ge:
                    report["checks"].append({"registered_GE_converter": package + suffix,
                        "callable_module": fn.__module__,
                        "status": "REGISTERED_INITIALIZED_GE_SESSION_PENDING"})
                    continue
                with GeGraph(name=package + suffix):
                    values = [ge.Data(index=i, dtype=DataType.DT_BF16, shape=list(t.shape), placement="NPU")
                              for i, t in enumerate(inputs)]
                    namespace = torch.ops.custom if package == "flash_ops" else torch.ops.fluent_sfa1
                    fn = _get_converter(getattr(namespace, "npu_mla_prolog_v3" + suffix).default)
                    options = {"query_norm_flag": True}
                    if package == "fluent_sfa1_ops":
                        options["output_cur_k_nope_rope"] = True
                    result = fn(*values, **options)
                    assert tuple(t.index for t in result) == expected
                    expected_type = "FluentMlaPrologV3" if package == "fluent_sfa1_ops" else "MlaPrologV3"
                    assert all(t.node.type == expected_type for t in result)
                    report["checks"].append({"GE_converter": package + suffix,
                                             "return_indices": list(expected), "status": "PASS_GRAPH_ONLY"})
        if args.with_ge:
            # GE converters are registered callables, not the decorator objects
            # stored in module globals in this pinned TorchAir version.
            for namespace_name, namespace in (("fluent_sfa1", torch.ops.fluent_sfa1),):
                with GeGraph(name=namespace_name + "_update_cur_contract"):
                    shapes = ((1, 1, 512), (1, 1, 64), (1, 2),
                              (16, 128, 1, 512), (16, 128, 1, 64), (1, 16))
                    values = [ge.Data(index=i, dtype=DataType.DT_INT32 if i in (2, 5) else DataType.DT_BF16,
                                      shape=list(shape), placement="NPU") for i, shape in enumerate(shapes)]
                    ordinary = _get_converter(namespace.npu_update_cur_kv_to_selection.default)(*values)
                    assert not isinstance(ordinary, (list, tuple)) and ordinary.node.type == "Identity"
                    assert any(value.startswith("UpdateCurKvToSelection") and value.endswith(":-1")
                               for value in ordinary.node.input)
                    functional = _get_converter(namespace.npu_update_cur_kv_to_selection_functional.default)(*values)
                    assert len(functional) == 3 and functional[0].node.type == "Identity"
                    assert [v.index for v in functional[1:]] == [0, 1]
                    producer = functional[1].controller
                    assert producer in functional[0].node.input
                    assert functional[1].node.name == functional[2].node.name
                    report["checks"].append({"GE_UpdateCur_contract": "PASS_GRAPH_ONLY", "namespace": str(namespace),
                                              "ordinary_returns": 1, "functional_returns": 3,
                                              "completion_dependency": producer})
        else:
            assert not torch_npu.npu.is_initialized()
            report["GE_device_provider"] = "PENDING_INITIALIZED_GE_SESSION_ON_OWNED_DEVICE"
        report["status"] = ("PASS_INSTALLED_DUAL_ABI_META_FUNCTIONALIZATION_GE_IR_GRAPH_DEVICE_CONTEXT_ONLY"
                            if args.with_ge else "PASS_INSTALLED_DUAL_ABI_META_FUNCTIONALIZATION_PROVIDER_CPU_GE_PENDING")
    except Exception as error:
        import traceback
        report.update(status="FAIL_OR_UNKNOWN", error=repr(error), traceback=traceback.format_exc())
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "checks": len(report["checks"]), "error": report.get("error")}))
    return 0 if report["status"].startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
