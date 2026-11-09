import pytest
import torch
import torch_npu

import flash_ops  # noqa: F401


def reference(logical_ids, expert_map):
    original_shape = logical_ids.shape
    indices = logical_ids.flatten().clamp(
        0, expert_map.numel() - 1
    ).to(torch.int64)
    return torch.gather(expert_map, 0, indices).view(original_shape)


def make_case(shape, map_size, seed):
    torch.manual_seed(seed)
    expert_map = torch.randperm(map_size, dtype=torch.int32, device="npu")
    logical_ids = torch.randint(
        -8, map_size + 9, shape, dtype=torch.int32, device="npu"
    )
    boundary_values = torch.tensor(
        [
            torch.iinfo(torch.int32).min,
            -1,
            0,
            map_size - 1,
            map_size,
            torch.iinfo(torch.int32).max,
        ],
        dtype=torch.int32,
        device="npu",
    )
    flat_ids = logical_ids.flatten()
    boundary_count = min(flat_ids.numel(), boundary_values.numel())
    if boundary_count > 0:
        flat_ids[:boundary_count] = boundary_values[:boundary_count]
    return logical_ids, expert_map


@pytest.fixture
def online_shape_case():
    return make_case((8, 12), 769, 20260829)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize(
    ("shape", "map_size"),
    (
        ((8, 12), 769),
        ((96,), 769),
        ((2, 4, 12), 769),
        ((4, 16), 1025),
        ((0, 12), 769),
        ((7,), 1),
    ),
)
def test_npu_moe_expert_id_map_eager(shape, map_size):
    logical_ids, expert_map = make_case(
        shape, map_size, 20260903 + len(shape) + map_size
    )
    expected = reference(logical_ids, expert_map)
    actual = torch.ops.custom.npu_moe_expert_id_map(logical_ids, expert_map)
    torch.npu.synchronize()
    assert actual.dtype == torch.int32
    assert actual.shape == logical_ids.shape
    assert torch.equal(actual.cpu(), expected.cpu())


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize(
    "invalid_case",
    (
        "logical_dtype",
        "map_dtype",
        "logical_noncontiguous",
        "map_noncontiguous",
        "empty_map",
        "map_rank",
        "different_device",
    ),
)
def test_npu_moe_expert_id_map_rejects_invalid_inputs(invalid_case):
    logical_ids = torch.arange(24, dtype=torch.int32, device="npu").view(4, 6)
    expert_map = torch.arange(32, dtype=torch.int32, device="npu")

    if invalid_case == "logical_dtype":
        logical_ids = logical_ids.to(torch.int64)
    elif invalid_case == "map_dtype":
        expert_map = expert_map.to(torch.int64)
    elif invalid_case == "logical_noncontiguous":
        logical_ids = logical_ids.transpose(0, 1)
    elif invalid_case == "map_noncontiguous":
        expert_map = expert_map[::2]
    elif invalid_case == "empty_map":
        expert_map = torch.empty(0, dtype=torch.int32, device="npu")
    elif invalid_case == "map_rank":
        expert_map = expert_map.view(4, 8)
    elif invalid_case == "different_device":
        expert_map = expert_map.cpu()

    with pytest.raises(RuntimeError):
        torch.ops.custom.npu_moe_expert_id_map(logical_ids, expert_map)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_npu_moe_expert_id_map_torchair_replay(online_shape_case):
    import torchair

    class Model(torch.nn.Module):
        def forward(self, logical_ids, expert_map):
            return torch.ops.custom.npu_moe_expert_id_map(
                logical_ids, expert_map
            )

    logical_ids, expert_map_a = online_shape_case
    expert_map_b = torch.flip(expert_map_a, dims=(0,)).contiguous()
    backend = torchair.get_npu_backend(
        compiler_config=torchair.CompilerConfig()
    )
    model = torch.compile(
        Model().npu(), backend=backend, fullgraph=True, dynamic=True
    )

    for expert_map in (expert_map_a, expert_map_b, expert_map_a):
        actual = model(logical_ids, expert_map)
        expected = reference(logical_ids, expert_map)
        torch.npu.synchronize()
        assert torch.equal(actual.cpu(), expected.cpu())


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_npu_moe_expert_id_map_torchair_shape_and_rank_variants():
    import torchair

    class Model(torch.nn.Module):
        def forward(self, logical_ids, expert_map):
            return torch.ops.custom.npu_moe_expert_id_map(
                logical_ids, expert_map
            )

    backend = torchair.get_npu_backend(
        compiler_config=torchair.CompilerConfig()
    )
    model = torch.compile(
        Model().npu(), backend=backend, fullgraph=True, dynamic=True
    )

    for index, (shape, map_size) in enumerate(
        (
            ((8, 12), 769),
            ((4, 16), 1025),
            ((96,), 769),
            ((2, 4, 12), 769),
            ((0, 12), 769),
        )
    ):
        logical_ids, expert_map = make_case(
            shape, map_size, 20261000 + index
        )
        actual = model(logical_ids, expert_map)
        expected = reference(logical_ids, expert_map)
        torch.npu.synchronize()
        assert actual.shape == logical_ids.shape
        assert torch.equal(actual.cpu(), expected.cpu())
