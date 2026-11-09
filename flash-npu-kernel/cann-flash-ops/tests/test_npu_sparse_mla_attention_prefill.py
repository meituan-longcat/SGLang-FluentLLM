import math
import os
import threading

import pytest
import torch


torch_npu = pytest.importorskip("torch_npu")
try:
    import custom_ops  # noqa: F401
except ImportError:
    pytest.importorskip("flash_ops")


HEADS = 64
LATENT = 512
WIDTH = 576
TOPK = 2048
PAGE = 128


def test_prefill_output_with_concurrent_unique():
    """Check Prefill output while another host thread submits unique.

    Both threads deliberately share a stream to reproduce the observed
    intermittent numerical corruption. This catches output regressions; it
    does not by itself distinguish workspace lifetime from kernel ordering
    defects, or reproduce the serving AICPU exception.
    """
    iterations = int(os.getenv("SFAD_CONCURRENT_TEST_ITERATIONS", "10000"))
    assert iterations > 0
    device_index = torch.npu.current_device()
    device = f"npu:{device_index}"
    stream = torch.npu.Stream(device=device_index)
    expected_value = 0.03125
    pages = torch.full((16, PAGE, WIDTH), expected_value,
                       dtype=torch.bfloat16, device=device)
    page_table = torch.arange(15, -1, -1, dtype=torch.int32, device=device).view(1, 16)
    inputs = []
    for count in (1, 8, 16, 127, 128, 129, 256, 512, 1350):
        inputs.append((
            torch.zeros((count, HEADS, WIDTH), dtype=torch.bfloat16, device=device),
            torch.arange(TOPK, dtype=torch.int32, device=device).repeat(count, 1),
            torch.full((count,), TOPK, dtype=torch.int32, device=device),
            torch.zeros(count, dtype=torch.int32, device=device),
        ))
    slots = torch.arange(8192 * PAGE, dtype=torch.int32, device=device)
    torch.npu.synchronize()
    stop = threading.Event()
    started = threading.Event()
    worker_errors = []
    calls = [0]

    def allocate_on_shared_stream():
        try:
            torch.npu.set_device(device_index)
            with torch.npu.stream(stream):
                started.set()
                while not stop.is_set():
                    count = (128, 1024, 5376, 65536, 1048576)[calls[0] % 5]
                    actual = torch.unique_consecutive(slots[:count] >> 7)
                    assert actual.numel() == (count + PAGE - 1) // PAGE
                    if calls[0] % 50 == 0:
                        expected = torch.arange((count + PAGE - 1) // PAGE, dtype=torch.int32)
                        torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
                    calls[0] += 1
        except BaseException as error:
            worker_errors.append(error)
            stop.set()
            started.set()

    worker = threading.Thread(target=allocate_on_shared_stream, daemon=True)
    worker.start()
    try:
        assert started.wait(timeout=30), "Allocation worker did not start"
        with torch.npu.stream(stream):
            for iteration in range(iterations):
                if worker_errors:
                    raise worker_errors[0]
                query, rows, counts, sequence = inputs[iteration % len(inputs)]
                actual = torch.ops.custom.npu_sparse_mla_attention_prefill(
                    query, pages, rows, counts, page_table, sequence,
                    1.0 / math.sqrt(WIDTH), True,
                )
                if iteration % 10 == 0 or iteration == iterations - 1:
                    # Uniform values have the same attention result for every
                    # query, independent of the selected page ordering.
                    observed = actual.cpu()
                    bad = observed != expected_value
                    assert not bad.any(), (
                        f"iteration={iteration}, query_tokens={query.size(0)}, "
                        f"wrong_elements={bad.sum().item()}, "
                        f"min={observed.min().item()}, max={observed.max().item()}"
                    )
        torch.npu.synchronize()
    finally:
        stop.set()
        worker.join(timeout=30)
    assert not worker.is_alive(), "Allocation worker did not stop"
    assert not worker_errors, repr(worker_errors)
    assert calls[0] > 0, "No concurrent allocations were exercised"


def _reference(query, pages, rows, valid_count, scale):
    flat = pages.reshape(-1, WIDTH).float()
    outputs = []
    for task in range(query.shape[0]):
        valid = int(valid_count[task].item())
        if valid == 0:
            outputs.append(
                torch.zeros(
                    (HEADS, LATENT), device=query.device, dtype=torch.bfloat16
                )
            )
            continue
        selected = flat.index_select(0, rows[task, :valid].long())
        score = torch.matmul(query[task].float(), selected.transpose(0, 1))
        prob = torch.softmax(score * scale, dim=-1).to(torch.bfloat16)
        # Match SFAD MM2: BF16 probability x BF16 latent, FP32 accumulate,
        # followed by one BF16 semantic-boundary rounding.
        outputs.append(torch.matmul(prob.float(), selected[:, :LATENT]).to(torch.bfloat16))
    return torch.stack(outputs)


@pytest.mark.parametrize("query_tokens", [1, 16])
@pytest.mark.parametrize("valid", [0, 1, 127, 128, 129, 2048])
@pytest.mark.parametrize("head_major_output", [False, True])
def test_sparse_mla_attention_prefill_logical_rows(
    query_tokens, valid, head_major_output
):
    torch.manual_seed(20260906 + query_tokens + valid)
    device = "npu"
    pages = torch.randn(
        (16, PAGE, WIDTH), device=device, dtype=torch.bfloat16
    ) * 0.05
    query = torch.randn(
        (query_tokens, HEADS, WIDTH), device=device, dtype=torch.bfloat16
    ) * 0.05
    rows = torch.full(
        (query_tokens, TOPK), -1, device=device, dtype=torch.int32
    )
    if valid:
        row = torch.randperm(TOPK, device=device, dtype=torch.int64)[:valid]
        rows[:, :valid] = row.to(torch.int32)
    counts = torch.full(
        (query_tokens,), valid, device=device, dtype=torch.int32
    )
    page_table = torch.arange(
        15, -1, -1, device=device, dtype=torch.int32
    ).view(1, -1)
    query_seq = torch.zeros(query_tokens, device=device, dtype=torch.int32)
    scale = 1.0 / math.sqrt(WIDTH)

    actual = torch.ops.custom.npu_sparse_mla_attention_prefill(
        query,
        pages,
        rows,
        counts,
        page_table,
        query_seq,
        scale,
        head_major_output,
    )
    physical_rows = rows.clone()
    if valid:
        logical = rows[:, :valid].long()
        physical_rows[:, :valid] = (
            page_table[0, logical // PAGE] * PAGE
            + (logical & (PAGE - 1))
        ).to(torch.int32)
    expected = _reference(query, pages, physical_rows, counts, scale)
    if head_major_output:
        expected = expected.transpose(0, 1).contiguous()

    torch.npu.synchronize()
    assert actual.dtype == torch.bfloat16
    expected_shape = (
        (HEADS, query_tokens, LATENT)
        if head_major_output
        else (query_tokens, HEADS, LATENT)
    )
    assert actual.shape == expected_shape
    torch.testing.assert_close(
        actual.float(), expected.float(), rtol=2e-2, atol=2e-2
    )


def test_sparse_mla_attention_prefill_production_query_shape():
    query_tokens = 2048
    pages = torch.zeros((1, PAGE, WIDTH), device="npu", dtype=torch.bfloat16)
    query = torch.zeros(
        (query_tokens, HEADS, WIDTH), device="npu", dtype=torch.bfloat16
    )
    rows = torch.full(
        (query_tokens, TOPK), -1, device="npu", dtype=torch.int32
    )
    counts = torch.zeros(query_tokens, device="npu", dtype=torch.int32)
    page_table = torch.zeros((1, 1), device="npu", dtype=torch.int32)
    query_seq = torch.zeros(query_tokens, device="npu", dtype=torch.int32)

    actual = torch.ops.custom.npu_sparse_mla_attention_prefill(
        query,
        pages,
        rows,
        counts,
        page_table,
        query_seq,
        1.0 / math.sqrt(WIDTH),
        True,
    )
    torch.npu.synchronize()

    assert actual.shape == (HEADS, query_tokens, LATENT)
    assert torch.count_nonzero(actual).item() == 0
