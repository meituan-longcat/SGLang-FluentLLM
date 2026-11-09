import pytest
import torch
import torch_npu

import flash_ops  # noqa: F401


class _FlatQuantModel(torch.nn.Module):
    def forward(self, x, p1, p2):
        return torch.ops.custom.flat_quant(x, p1, p2)


class _FlatQuantClearPaddingModel(torch.nn.Module):
    def forward(self, x, p1, p2, group_list):
        return torch.ops.custom.flat_quant(
            x,
            p1,
            p2,
            group_list=group_list,
            group_list_type=1,
            clear_padding=True,
        )


def _compile_flat_quant_graph():
    import torchair

    torch._dynamo.reset()
    backend = torchair.get_npu_backend(compiler_config=torchair.CompilerConfig())
    return torch.compile(
        _FlatQuantModel(), backend=backend, dynamic=False, fullgraph=True
    )


def _compile_flat_quant_clear_padding_graph():
    import torchair

    torch._dynamo.reset()
    backend = torchair.get_npu_backend(compiler_config=torchair.CompilerConfig())
    return torch.compile(
        _FlatQuantClearPaddingModel(),
        backend=backend,
        dynamic=False,
        fullgraph=True,
    )


def _reference(x, p1, p2, clip_ratio=1.0):
    transformed = p1.float() @ (x.float() @ p2.float())
    transformed = transformed.flatten(1)
    scale = transformed.abs().amax(dim=1) * clip_ratio / 7.0
    safe_scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    quantized = torch.round(transformed / safe_scale[:, None]).clamp(-8, 7)
    return quantized.to(torch.int32), safe_scale


def _unpack_int4(packed):
    packed = packed.cpu().to(torch.int64)
    values = [((packed >> shift) & 0xF) for shift in range(0, 32, 4)]
    unpacked = torch.stack(values, dim=-1)
    unpacked = torch.where(unpacked >= 8, unpacked - 16, unpacked)
    return unpacked.flatten(1)


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
def test_flat_quant_supports_zero_tokens(dtype):
    x = torch.empty((0, 16, 8), dtype=dtype, device="npu")
    p1 = torch.eye(16, dtype=dtype, device="npu")
    p2 = torch.eye(8, dtype=dtype, device="npu")

    output, scale = torch.ops.custom.flat_quant(x, p1, p2)
    torch.npu.synchronize()

    assert output.shape == (0, 16, 1)
    assert output.dtype == torch.int32
    assert scale.shape == (0,)
    assert scale.dtype == torch.float32


def test_flat_quant_graph_supports_zero_tokens():
    dtype = torch.float16
    x = torch.empty((0, 16, 8), dtype=dtype, device="npu")
    p1 = torch.eye(16, dtype=dtype, device="npu")
    p2 = torch.eye(8, dtype=dtype, device="npu")

    output, scale = _compile_flat_quant_graph()(x, p1, p2)
    torch.npu.synchronize()

    assert output.shape == (0, 16, 1)
    assert output.dtype == torch.int32
    assert scale.shape == (0,)
    assert scale.dtype == torch.float32


@pytest.mark.parametrize(
    ("group_list_type", "group_list_shape"),
    ((0, (0,)), (2, (0, 2))),
)
def test_flat_quant_supports_zero_tokens_with_group_list(
    group_list_type, group_list_shape
):
    x = torch.empty((0, 16, 8), dtype=torch.float16, device="npu")
    p1 = torch.eye(16, dtype=torch.float16, device="npu")
    p2 = torch.eye(8, dtype=torch.float16, device="npu")
    group_list = torch.empty(group_list_shape, dtype=torch.int64, device="npu")

    output, scale = torch.ops.custom.flat_quant(
        x,
        p1,
        p2,
        group_list=group_list,
        group_list_type=group_list_type,
    )
    torch.npu.synchronize()

    assert output.shape == (0, 16, 1)
    assert scale.shape == (0,)


def test_flat_quant_meta_validates_empty_shapes_and_attributes():
    x = torch.empty((0, 16, 8), dtype=torch.float16, device="meta")
    p1 = torch.empty((16, 16), dtype=torch.float16, device="meta")
    p2 = torch.empty((8, 8), dtype=torch.float16, device="meta")

    output, scale = torch.ops.custom.flat_quant(x, p1, p2)
    assert output.shape == (0, 16, 1)
    assert scale.shape == (0,)

    output, scale = torch.ops.custom.flat_quant(
        x, p1, p2, clear_padding=True
    )
    assert output.shape == (0, 16, 1)
    assert scale.shape == (0,)

    with pytest.raises(RuntimeError, match="clip_ratio"):
        torch.ops.custom.flat_quant(x, p1, p2, clip_ratio=0.0)
    with pytest.raises(RuntimeError, match="group_list_type"):
        torch.ops.custom.flat_quant(x, p1, p2, group_list_type=3)

    for shape in ((1, 0, 8), (1, 16, 0)):
        invalid_x = torch.empty(shape, dtype=torch.float16, device="meta")
        invalid_p1 = torch.empty((shape[1], shape[1]), dtype=torch.float16, device="meta")
        invalid_p2 = torch.empty((shape[2], shape[2]), dtype=torch.float16, device="meta")
        with pytest.raises(RuntimeError, match="dim1/dim2"):
            torch.ops.custom.flat_quant(invalid_x, invalid_p1, invalid_p2)


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
def test_flat_quant_matches_fp32_reference(dtype):
    torch.manual_seed(20260826)
    x = torch.randn(8, 16, 16, dtype=dtype)
    p1 = torch.randn(16, 16, dtype=dtype) / 4
    p2 = torch.randn(16, 16, dtype=dtype) / 4
    expected_output, expected_scale = _reference(x, p1, p2)

    output, scale = torch.ops.custom.flat_quant(x.npu(), p1.npu(), p2.npu())
    torch.npu.synchronize()

    torch.testing.assert_close(
        scale.cpu(), expected_scale, rtol=1e-2, atol=1e-2
    )
    delta = (_unpack_int4(output) - expected_output).abs()
    assert delta.max().item() <= 1
    tolerance = 0.03 if dtype == torch.bfloat16 else 0.01
    assert (delta != 0).float().mean().item() <= tolerance


def test_flat_quant_graph_matches_eager():
    torch.manual_seed(20260827)
    dtype = torch.float16
    x = torch.randn(8, 16, 16, dtype=dtype, device="npu")
    p1 = (torch.randn(16, 16, dtype=dtype) / 4).npu()
    p2 = (torch.randn(16, 16, dtype=dtype) / 4).npu()

    expected_output, expected_scale = torch.ops.custom.flat_quant(x, p1, p2)
    output, scale = _compile_flat_quant_graph()(x, p1, p2)
    torch.npu.synchronize()

    torch.testing.assert_close(output, expected_output, rtol=0.0, atol=0.0)
    torch.testing.assert_close(scale, expected_scale, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("valid_k", (0, 3))
def test_flat_quant_clear_padding_is_opt_in(valid_k):
    torch.manual_seed(20260905)
    dtype = torch.float16
    physical_k = 8
    x = torch.randn(physical_k, 16, 16, dtype=dtype, device="npu")
    p1 = torch.eye(16, dtype=dtype, device="npu")
    p2 = torch.eye(16, dtype=dtype, device="npu")
    group_list = torch.tensor([valid_k], dtype=torch.int64, device="npu")

    default_output, default_scale = torch.ops.custom.flat_quant(
        x,
        p1,
        p2,
        group_list=group_list,
        group_list_type=1,
    )
    cleared_output, cleared_scale = torch.ops.custom.flat_quant(
        x,
        p1,
        p2,
        group_list=group_list,
        group_list_type=1,
        clear_padding=True,
    )
    torch.npu.synchronize()

    assert torch.count_nonzero(cleared_output[valid_k:]).item() == 0
    assert torch.count_nonzero(cleared_scale[valid_k:]).item() == 0
    if valid_k > 0:
        torch.testing.assert_close(
            cleared_output[:valid_k],
            default_output[:valid_k],
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            cleared_scale[:valid_k],
            default_scale[:valid_k],
            rtol=0.0,
            atol=0.0,
        )


def test_flat_quant_graph_clear_padding_matches_eager():
    torch.manual_seed(20260905)
    dtype = torch.float16
    x = torch.randn(8, 16, 16, dtype=dtype, device="npu")
    p1 = torch.eye(16, dtype=dtype, device="npu")
    p2 = torch.eye(16, dtype=dtype, device="npu")
    group_list = torch.tensor([3], dtype=torch.int64, device="npu")

    expected_output, expected_scale = torch.ops.custom.flat_quant(
        x,
        p1,
        p2,
        group_list=group_list,
        group_list_type=1,
        clear_padding=True,
    )
    output, scale = _compile_flat_quant_clear_padding_graph()(
        x, p1, p2, group_list
    )
    torch.npu.synchronize()

    torch.testing.assert_close(output, expected_output, rtol=0.0, atol=0.0)
    torch.testing.assert_close(scale, expected_scale, rtol=0.0, atol=0.0)
