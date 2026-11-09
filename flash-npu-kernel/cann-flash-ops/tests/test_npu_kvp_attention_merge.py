import pytest
import torch
import torch_npu  # noqa: F401

import flash_ops  # noqa: F401


def reference(attention_out, lse):
    valid = lse > -1.0e30
    masked_lse = torch.where(valid, lse, torch.full_like(lse, -torch.inf))
    maximum = masked_lse.amax(dim=0, keepdim=True)
    all_empty = ~valid.any(dim=0, keepdim=True)
    maximum = torch.where(all_empty, torch.zeros_like(maximum), maximum)
    exponent = torch.where(valid, torch.exp(lse - maximum), torch.zeros_like(lse))
    denominator = exponent.sum(dim=0, keepdim=True)
    weights = torch.where(
        denominator > 0, exponent / denominator, torch.zeros_like(exponent)
    )
    safe_attention = torch.where(
        valid.unsqueeze(-1), attention_out, torch.zeros_like(attention_out)
    )
    return (safe_attention * weights.unsqueeze(-1)).sum(dim=0).to(torch.bfloat16)


def make_edge_case(case, rows=32):
    torch.manual_seed(20260830 + rows)
    attention_out = torch.randn(
        8, rows, 512, dtype=torch.float32, device="npu"
    )
    lse = torch.randn(8, rows, dtype=torch.float32, device="npu") * 20.0

    if case == "all_valid":
        pass
    elif case == "one_valid_rank":
        lse.fill_(-2.0e38)
        attention_out.fill_(torch.nan)
        lse[3] = torch.linspace(-100.0, 100.0, rows, device="npu")
        attention_out[3] = torch.randn_like(attention_out[3])
    elif case == "partial_empty_ranks":
        lse[1::2] = -2.0e38
        attention_out[1::2] = torch.nan
    elif case == "all_empty":
        lse.fill_(-2.0e38)
        attention_out.fill_(torch.nan)
    elif case == "extreme_finite_lse":
        values = torch.tensor(
            [-1.0e4, -1.0e3, -100.0, -1.0, 1.0, 100.0, 1.0e3, 1.0e4],
            dtype=torch.float32, device="npu"
        )
        lse.copy_(values[:, None].expand_as(lse))
    else:
        raise AssertionError(f"unknown edge case: {case}")
    return attention_out, lse


@pytest.mark.parametrize("rows", (32, 256, 512))
@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_npu_kvp_attention_merge_matches_reference(rows):
    torch.manual_seed(20260830 + rows)
    attention_out = torch.randn(
        8, rows, 512, dtype=torch.float32, device="npu"
    )
    # Exercise stable-softmax behavior with a substantially wider range than
    # the online LSE values normally use.
    lse = (
        torch.randn(8, rows, dtype=torch.float32, device="npu") * 20.0
    )

    expected = reference(attention_out, lse)
    actual = torch.ops.custom.npu_kvp_attention_merge(attention_out, lse)
    torch.npu.synchronize()

    assert actual.dtype == torch.bfloat16
    assert actual.shape == (rows, 512)
    torch.testing.assert_close(
        actual.float().cpu(), expected.float().cpu(), rtol=4e-3, atol=8e-3
    )


@pytest.mark.parametrize(
    "case",
    (
        "all_valid",
        "one_valid_rank",
        "partial_empty_ranks",
        "all_empty",
        "extreme_finite_lse",
    ),
)
@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_npu_kvp_attention_merge_empty_rank_semantics(case):
    attention_out, lse = make_edge_case(case)
    expected = reference(attention_out, lse)
    actual = torch.ops.custom.npu_kvp_attention_merge(attention_out, lse)
    torch.npu.synchronize()

    assert not torch.isnan(actual).any().item()
    if case == "all_empty":
        assert torch.count_nonzero(actual).item() == 0
    torch.testing.assert_close(
        actual.float().cpu(), expected.float().cpu(), rtol=4e-3, atol=8e-3
    )


@pytest.mark.parametrize(
    "invalid_case",
    (
        "attention_rank",
        "lse_rank",
        "attention_dtype",
        "lse_dtype",
        "attention_noncontiguous",
        "lse_noncontiguous",
        "rank_count",
        "row_count",
        "empty_rows",
        "hidden_size",
        "different_device",
    ),
)
@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_npu_kvp_attention_merge_rejects_invalid_inputs(invalid_case):
    attention_out = torch.randn(
        8, 32, 512, dtype=torch.float32, device="npu"
    )
    lse = torch.randn(8, 32, dtype=torch.float32, device="npu")

    if invalid_case == "attention_rank":
        attention_out = attention_out.view(8, 32, 16, 32)
    elif invalid_case == "lse_rank":
        lse = lse.unsqueeze(-1)
    elif invalid_case == "attention_dtype":
        attention_out = attention_out.to(torch.bfloat16)
    elif invalid_case == "lse_dtype":
        lse = lse.to(torch.bfloat16)
    elif invalid_case == "attention_noncontiguous":
        attention_out = attention_out.transpose(1, 2)
    elif invalid_case == "lse_noncontiguous":
        lse = lse.transpose(0, 1)
    elif invalid_case == "rank_count":
        attention_out = attention_out[:4]
        lse = lse[:4]
    elif invalid_case == "row_count":
        lse = lse[:, :16]
    elif invalid_case == "empty_rows":
        attention_out = attention_out[:, :0]
        lse = lse[:, :0]
    elif invalid_case == "hidden_size":
        attention_out = attention_out[:, :, :256]
    elif invalid_case == "different_device":
        lse = lse.cpu()

    with pytest.raises(RuntimeError):
        torch.ops.custom.npu_kvp_attention_merge(attention_out, lse)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_npu_kvp_attention_merge_torchair_replay_is_stable():
    import torchair

    class Model(torch.nn.Module):
        def forward(self, attention_out, lse):
            return torch.ops.custom.npu_kvp_attention_merge(
                attention_out, lse
            )

    torch.manual_seed(20260830)
    attention_out = torch.randn(
        8, 512, 512, dtype=torch.float32, device="npu"
    )
    lse_a = torch.randn(8, 512, dtype=torch.float32, device="npu")
    lse_b = torch.full_like(lse_a, -2.0e38)
    lse_b[2] = torch.randn(512, dtype=torch.float32, device="npu")
    attention_b = torch.zeros_like(attention_out)
    attention_b[2] = attention_out[2]
    backend = torchair.get_npu_backend(
        compiler_config=torchair.CompilerConfig()
    )
    model = torch.compile(Model().npu(), backend=backend, fullgraph=True)

    for current_attention, lse in (
        (attention_out, lse_a),
        (attention_b, lse_b),
        (attention_out, lse_a),
    ):
        expected = reference(current_attention, lse)
        actual = model(current_attention, lse)
        torch.npu.synchronize()
        torch.testing.assert_close(
            actual.float().cpu(), expected.float().cpu(),
            rtol=4e-3, atol=8e-3
        )
