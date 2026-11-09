import pytest
import torch
import torch_npu

import flash_ops  # noqa: F401


@pytest.fixture
def gather_case():
    values = torch.tensor(
        [[10, 11, 12], [20, 21, 22]], dtype=torch.int32, device="npu"
    )
    indices = torch.tensor(
        [[2, 0], [1, 2]], dtype=torch.int64, device="npu"
    )
    expected = torch.tensor([[12, 10], [21, 22]], dtype=torch.int32)
    return values, indices, expected


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_npu_gather_element_eager(gather_case):
    values, indices, expected = gather_case
    actual = torch.ops.custom.npu_gather_element(values, 1, indices)
    torch.npu.synchronize()
    assert torch.equal(actual.cpu(), expected)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_npu_gather_element_eager_non_last_dim():
    values = torch.tensor(
        [[10, 11, 12], [20, 21, 22]], dtype=torch.int32, device="npu"
    )
    indices = torch.tensor([[1, 0, 1]], dtype=torch.int64, device="npu")
    expected = torch.tensor([[20, 11, 22]], dtype=torch.int32)

    actual = torch.ops.custom.npu_gather_element(values, 0, indices)
    torch.npu.synchronize()
    assert torch.equal(actual.cpu(), expected)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_npu_gather_element_torchair(gather_case):
    import torchair

    class Model(torch.nn.Module):
        def forward(self, values, indices):
            return torch.ops.custom.npu_gather_element(values, 1, indices)

    values, indices, expected = gather_case
    backend = torchair.get_npu_backend(compiler_config=torchair.CompilerConfig())
    model = torch.compile(Model().npu(), backend=backend, fullgraph=True)
    actual = model(values, indices)
    torch.npu.synchronize()
    assert torch.equal(actual.cpu(), expected)
