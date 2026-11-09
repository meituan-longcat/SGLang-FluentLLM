import torch
import torch_npu


def test_npu_format_cast_nd_nz_roundtrip():
    for dtype in (torch.float16, torch.bfloat16, torch.int8, torch.int32):
        expected = ((torch.arange(33 * 65).reshape(33, 65) % 101) - 50).to(dtype)
        value = expected.npu()
        nz = torch_npu.npu_format_cast(value, 29)
        restored = torch_npu.npu_format_cast(nz, 2)
        torch.npu.synchronize()
        assert torch_npu.get_npu_format(nz).value == 29
        assert torch.equal(restored.cpu(), expected)
