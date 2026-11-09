import pytest
import torch
import torch_npu

import flash_ops  # noqa: F401


def _attrs(k, out_flag=True):
    return {
        "k": k,
        "k_group": 1,
        "group_count": 1,
        "group_select_mode": 0,
        "renorm": 0,
        "norm_type": 0,
        "out_flag": out_flag,
        "routed_scaling_factor": 1.0,
        "eps": 1e-20,
    }


def _make_case(rows, experts, k, map_size, seed):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(rows, experts, generator=generator, dtype=torch.float32).npu()
    bias = (
        torch.randn(experts, generator=generator, dtype=torch.float32) * 0.01
    ).npu()
    expert_map_cpu = torch.randint(
        0, experts, (map_size,), generator=generator, dtype=torch.int32
    )
    expert_map_cpu[-1] = experts
    return x, bias, expert_map_cpu.npu(), expert_map_cpu, _attrs(k)


def _check(x, bias, expert_map, expert_map_cpu, attrs):
    reference = torch_npu.npu_moe_gating_top_k(x, bias=bias, **attrs)
    actual = torch.ops.custom.npu_moe_gating_top_k_expert_map(
        x, expert_map, bias, **attrs
    )
    torch.npu.synchronize()

    logical_ids = reference[1].cpu().clamp(0, expert_map_cpu.numel() - 1).long()
    assert torch.equal(actual[0].cpu(), reference[0].cpu())
    assert torch.equal(actual[1].cpu(), expert_map_cpu[logical_ids])
    if attrs["out_flag"]:
        assert torch.equal(actual[2].cpu(), reference[2].cpu())


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize(
    ("rows", "experts", "k", "map_size"),
    (
        (1, 32, 1, 1),
        (8, 896, 12, 769),
        (8, 128, 31, 129),
        (8, 128, 32, 129),
        (8, 128, 33, 129),
        (16, 1024, 16, 1025),
        (1, 2048, 2048, 2049),
    ),
)
def test_npu_moe_gating_top_k_expert_map_eager(rows, experts, k, map_size):
    _check(
        *_make_case(rows, experts, k, map_size, 20260904 + rows + experts + k)
    )


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_npu_moe_gating_top_k_expert_map_online_out_flag_false():
    x, bias, expert_map, expert_map_cpu, _ = _make_case(
        8, 896, 12, 769, 20260904
    )
    _check(
        x,
        bias,
        expert_map,
        expert_map_cpu,
        _attrs(12, out_flag=False),
    )


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_npu_moe_gating_top_k_expert_map_empty_rows_eager():
    x = torch.empty((0, 896), dtype=torch.float32, device="npu")
    bias = torch.empty((896,), dtype=torch.float32, device="npu")
    expert_map = torch.arange(769, dtype=torch.int32, device="npu")

    actual = torch.ops.custom.npu_moe_gating_top_k_expert_map(
        x, expert_map, bias, **_attrs(12, out_flag=False)
    )

    assert actual[0].shape == (0, 12)
    assert actual[0].dtype == torch.float32
    assert actual[1].shape == (0, 12)
    assert actual[1].dtype == torch.int32
    assert actual[2].shape == (0, 896)
    assert actual[2].dtype == torch.float32


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_npu_moe_gating_top_k_expert_map_empty_rows_torchair():
    import torchair

    class Model(torch.nn.Module):
        def forward(self, x, expert_map, bias):
            return torch.ops.custom.npu_moe_gating_top_k_expert_map(
                x, expert_map, bias, **_attrs(12, out_flag=False)
            )

    x = torch.empty((0, 896), dtype=torch.float32, device="npu")
    bias = torch.empty((896,), dtype=torch.float32, device="npu")
    expert_map = torch.arange(769, dtype=torch.int32, device="npu")
    backend = torchair.get_npu_backend(compiler_config=torchair.CompilerConfig())
    model = torch.compile(Model().npu(), backend=backend, fullgraph=True)

    actual = model(x, expert_map, bias)
    torch.npu.synchronize()
    assert [tuple(output.shape) for output in actual] == [
        (0, 12),
        (0, 12),
        (0, 896),
    ]


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_npu_moe_gating_top_k_expert_map_torchair_runtime_map():
    import torchair

    class Model(torch.nn.Module):
        def forward(self, x, expert_map, bias):
            return torch.ops.custom.npu_moe_gating_top_k_expert_map(
                x, expert_map, bias, **_attrs(12)
            )

    x, bias, expert_map, expert_map_cpu, attrs = _make_case(
        8, 896, 12, 769, 20260904
    )
    backend = torchair.get_npu_backend(compiler_config=torchair.CompilerConfig())
    model = torch.compile(Model().npu(), backend=backend, fullgraph=True)

    for current_map, current_map_cpu in (
        (expert_map, expert_map_cpu),
        (torch.flip(expert_map, (0,)).contiguous(), torch.flip(expert_map_cpu, (0,))),
    ):
        reference = torch_npu.npu_moe_gating_top_k(x, bias=bias, **attrs)
        actual = model(x, current_map, bias)
        torch.npu.synchronize()
        logical_ids = reference[1].cpu().clamp(0, current_map_cpu.numel() - 1).long()
        assert torch.equal(actual[0].cpu(), reference[0].cpu())
        assert torch.equal(actual[1].cpu(), current_map_cpu[logical_ids])
        assert torch.equal(actual[2].cpu(), reference[2].cpu())


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize(
    "kwargs",
    (
        {"k": 0},
        {"k": 13},
        {"k": 12, "group_count": 2},
        {"k": 12, "k_group": 2},
    ),
)
def test_npu_moe_gating_top_k_expert_map_rejects_unsupported(kwargs):
    x = torch.randn(8, 12, dtype=torch.float32, device="npu")
    expert_map = torch.arange(13, dtype=torch.int32, device="npu")
    with pytest.raises(RuntimeError):
        torch.ops.custom.npu_moe_gating_top_k_expert_map(
            x, expert_map, **kwargs
        )
