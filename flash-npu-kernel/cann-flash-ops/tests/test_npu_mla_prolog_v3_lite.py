"""Lite and FluentLLM model-shape checks for MlaPrologV3."""

import torch
import torch_npu

import flash_ops  # noqa: F401  # registers torch.ops.custom


HCQ = 1536
HCKV = 512
D = 128
DR = 64
EPS = 1.0e-5
LITE_MODELS = (("lite-3b", 3072, 32), ("lite-large", 4096, 64))
FLASH_MODELS = (("flash-3b", 3072, 32), ("flash-pro", 8192, 64))


def _random_bf16(shape, scale=0.01):
    return (torch.randn(shape, dtype=torch.float32) * scale).to(torch.bfloat16).npu()


def _rms_norm(value, width):
    value = value[..., :width].float()
    return (value * torch.rsqrt(value.square().mean(-1, keepdim=True) + EPS)).to(
        torch.bfloat16
    )


def _make_tensors(he, heads, tokens):
    tensors = {
        "x": _random_bf16((tokens, he)),
        "wdq_nd": _random_bf16((he, HCQ)),
        "wuq_nd": _random_bf16((HCQ, heads * (D + DR))),
        "wuk": _random_bf16((heads, D, HCKV)),
        "wdkv_nd": _random_bf16((he, HCKV + DR)),
        "gcq": torch.ones(HCQ, dtype=torch.bfloat16, device="npu"),
        "gckv": torch.ones(HCKV, dtype=torch.bfloat16, device="npu"),
        "sin": _random_bf16((tokens, DR)),
        "cos": _random_bf16((tokens, DR)),
    }
    for name in ("wdq", "wuq", "wdkv"):
        tensors[name] = torch_npu.npu_format_cast(tensors[f"{name}_nd"], 29)
    return tensors


def _run(
    heads, enable_rope, combined_cache, tensors, cache_index, cache_mode="PA_BSND"
):
    tokens = tensors["x"].shape[0]
    cache_width = HCKV + DR if combined_cache else HCKV
    if cache_mode == "PA_BSND":
        kv_cache = torch.zeros(
            (2, 128, 1, cache_width), dtype=torch.bfloat16
        ).npu()
        kr_cache = torch.zeros((2, 128, 1, DR), dtype=torch.bfloat16).npu()
    elif cache_mode == "TND":
        kv_cache = torch.zeros((tokens, 1, cache_width), dtype=torch.bfloat16).npu()
        kr_cache = torch.zeros((tokens, 1, DR), dtype=torch.bfloat16).npu()
    else:
        raise ValueError(f"unsupported test cache mode: {cache_mode}")
    kwargs = {
        "cache_mode": cache_mode,
        "ckvkr_repo_mode": int(combined_cache),
    }
    if cache_mode == "PA_BSND":
        kwargs["cache_index"] = cache_index
    if enable_rope is not None:
        kwargs["enable_rope"] = enable_rope
    outputs = torch.ops.custom.npu_mla_prolog_v3(
        tensors["x"],
        tensors["wdq"],
        tensors["wuq"],
        tensors["wuk"],
        tensors["wdkv"],
        tensors["gcq"],
        tensors["gckv"],
        tensors["sin"],
        tensors["cos"],
        kv_cache,
        kr_cache,
        **kwargs,
    )
    assert outputs[0].shape == (tokens, heads, HCKV)
    assert outputs[1].shape == (tokens, heads, DR)
    return outputs, kv_cache, kr_cache


def _assert_lite_nope_packed(he, heads, tokens):
    tensors = _make_tensors(he, heads, tokens)
    cache_index = torch.arange(127, 127 + tokens, dtype=torch.int64, device="npu")
    outputs, packed_cache, kr_cache = _run(
        heads, False, True, tensors, cache_index
    )

    cq = _rms_norm(torch.matmul(tensors["x"], tensors["wdq_nd"]), HCQ)
    qc_qr = torch.matmul(cq, tensors["wuq_nd"]).view(tokens, heads, D + DR)
    expected_qr = qc_qr[..., D:]
    expected_q = torch.einsum("bnd,ndc->bnc", qc_qr[..., :D], tensors["wuk"])
    ckv_kr = torch.matmul(tensors["x"], tensors["wdkv_nd"])
    cache_rows = torch.cat((_rms_norm(ckv_kr, HCKV), ckv_kr[..., HCKV:]), -1)
    expected_cache = torch.zeros_like(packed_cache).view(-1, HCKV + DR)
    expected_cache.index_copy_(0, cache_index, cache_rows)

    torch.testing.assert_close(outputs[0], expected_q, rtol=1e-2, atol=1e-3)
    torch.testing.assert_close(outputs[1], expected_qr, rtol=1e-2, atol=1e-3)
    torch.testing.assert_close(
        packed_cache.view(-1, HCKV + DR), expected_cache, rtol=1e-2, atol=1e-3
    )
    torch.testing.assert_close(kr_cache, torch.zeros_like(kr_cache), rtol=0, atol=0)

    if heads in (4, 8):
        tensors["sin"] = _random_bf16((tokens, DR), scale=3.0)
        tensors["cos"] = _random_bf16((tokens, DR), scale=3.0)
        outputs_again, cache_again, _ = _run(
            heads, False, True, tensors, cache_index
        )
        torch.testing.assert_close(outputs_again[1], outputs[1], rtol=0, atol=0)
        torch.testing.assert_close(cache_again, packed_cache, rtol=0, atol=0)


@torch.inference_mode()
def test_lite_nope_packed_cache_tp_matrix():
    torch.manual_seed(7303)
    torch.npu.config.allow_internal_format = True
    torch.npu.set_device(0)

    for _, he, global_heads in LITE_MODELS:
        for tp, tokens in zip((1, 2, 4, 8), (1, 2, 8, 32)):
            _assert_lite_nope_packed(he, global_heads // tp, tokens)


@torch.inference_mode()
def test_flash_models_rope_split_cache_compatibility():
    torch.manual_seed(7304)
    torch.npu.config.allow_internal_format = True
    torch.npu.set_device(0)

    for _, he, global_heads in FLASH_MODELS:
        heads = global_heads // 8
        tensors = _make_tensors(he, heads, 1)
        cache_index = torch.tensor([127], dtype=torch.int64, device="npu")
        explicit, kv_explicit, kr_explicit = _run(
            heads, True, False, tensors, cache_index
        )
        default, kv_default, kr_default = _run(
            heads, None, False, tensors, cache_index
        )
        for actual, expected in zip(default[:2], explicit[:2]):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(kv_default, kv_explicit, rtol=0, atol=0)
        torch.testing.assert_close(kr_default, kr_explicit, rtol=0, atol=0)


@torch.inference_mode()
def test_lite_packed_cache_requires_pa_bsnd():
    torch.manual_seed(7305)
    torch.npu.config.allow_internal_format = True
    torch.npu.set_device(0)

    tensors = _make_tensors(3072, 4, 1)
    cache_index = torch.tensor([0], dtype=torch.int64, device="npu")
    try:
        _run(4, False, True, tensors, cache_index, cache_mode="TND")
    except RuntimeError:
        return
    raise AssertionError("TND must reject non-quantized combined CKV/KR cache")
