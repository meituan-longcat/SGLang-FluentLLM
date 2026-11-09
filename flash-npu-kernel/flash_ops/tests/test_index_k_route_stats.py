"""Exact integer route statistics, including DMA tails and absent groups."""

import pytest
import torch

pytest.importorskip("torch_npu")
import flash_npu_kernel  # noqa: F401, E402


@pytest.mark.parametrize("groups_count", [1, 4, 64, 256])
@pytest.mark.parametrize(
    "n",
    [0, 1, 3, 31, 32, 33, 127, 128, 129, 511, 512, 513, 1023, 1024, 1728, 4097, 32769],
)
@pytest.mark.parametrize("skewed", [False, True])
def test_statistics(groups_count, n, skewed):
    rng = torch.Generator().manual_seed(20261027)
    groups = torch.randint(groups_count, (n,), generator=rng, dtype=torch.int64)
    if skewed:
        groups.fill_(groups_count - 1)
    lengths = torch.randint(1, 129, (n,), generator=rng, dtype=torch.int64)
    pages = torch.ones(n, dtype=torch.int64)
    if n and skewed:
        pages[-1] = 0
    expected = torch.zeros(2 * groups_count + 1, dtype=torch.int64)
    expected[:groups_count].scatter_add_(0, groups, lengths)
    expected[groups_count:-1] = torch.bincount(groups, minlength=groups_count)
    expected[-1] = (pages < 1).any()
    actual = torch.ops.flash.npu_index_k_route_stats(
        groups.npu(), lengths.npu(), pages.npu(), groups_count, 1
    )
    assert torch.equal(actual.cpu(), expected)


def test_large_integer_totals():
    groups = torch.tensor([0, 0, 63, 63], dtype=torch.int64, device="npu")
    lengths = torch.tensor([2**30 + 1, 2**30 + 3, 2**24 + 1, 2**24 + 7], device="npu")
    pages = torch.full_like(groups, 2**40)
    result = torch.ops.flash.npu_index_k_route_stats(
        groups, lengths, pages, 64, 2**32
    ).cpu()
    assert result[0] == 2**31 + 4
    assert result[63] == 2**25 + 8
    assert result[64] == result[127] == 2
    assert result[-1] == 0


def test_bad_group_is_safe():
    groups = torch.tensor([-1, 64], dtype=torch.int64, device="npu")
    ones = torch.ones_like(groups)
    result = torch.ops.flash.npu_index_k_route_stats(groups, ones, ones, 64, 1).cpu()
    assert result[:-1].count_nonzero() == 0
    assert result[-1] == 1


def test_invalid_values_and_zero_length():
    groups = torch.tensor([0, 1, 0, 1], dtype=torch.int64, device="npu")
    lengths = torch.tensor([-1, 0, 4, 3], dtype=torch.int64, device="npu")
    pages = torch.tensor([1, 1, 0, 2], dtype=torch.int64, device="npu")
    result = torch.ops.flash.npu_index_k_route_stats(groups, lengths, pages, 2, 1)
    # Negative lengths are excluded. Zero-length pages still count, and an
    # unallocated page sets the flag without changing valid group statistics.
    assert torch.equal(result.cpu(), torch.tensor([4, 3, 1, 2, 1]))


@pytest.mark.parametrize(
    "invalid",
    [
        "dtype",
        "noncontiguous",
        "size",
        "too_many_groups",
        "no_groups",
        "device",
        "reserved",
    ],
)
def test_reject_invalid_metadata(invalid):
    x = torch.ones(8, dtype=torch.int64, device="npu")
    inputs = {
        "dtype": (x.float(), x, x, 64, 1),
        "noncontiguous": (x[::2], x[:4], x[:4], 64, 1),
        "size": (x, x[:4], x, 64, 1),
        "too_many_groups": (x, x, x, 257, 1),
        "no_groups": (x, x, x, 0, 1),
        "device": (x, x, x.cpu(), 64, 1),
        "reserved": (x, x, x, 64, -1),
    }
    with pytest.raises((RuntimeError, NotImplementedError)):
        torch.ops.flash.npu_index_k_route_stats(*inputs[invalid])


def test_meta_shape():
    x = torch.ones(8, dtype=torch.int64, device="meta")
    result = torch.ops.flash.npu_index_k_route_stats(x, x, x, 64, 1)
    assert result.shape == (129,)
    assert result.dtype == torch.int64
    assert result.device.type == "meta"


def test_queued_calls_on_two_streams():
    inputs = []
    for n in range(1, 257):
        groups = torch.arange(n, dtype=torch.int64) % 64
        lengths = torch.arange(n, dtype=torch.int64) + 1
        expected = torch.zeros(129, dtype=torch.int64)
        expected[:64].scatter_add_(0, groups, lengths)
        expected[64:-1] = torch.bincount(groups, minlength=64)
        inputs.append(
            (groups.npu(), lengths.npu(), torch.ones_like(groups).npu(), expected)
        )
    streams = [torch.npu.Stream(), torch.npu.Stream()]
    for stream in streams:
        stream.wait_stream(torch.npu.current_stream())
    outputs = []
    # All H2D copies precede this loop; no synchronous copy serializes calls.
    for i, (groups, lengths, pages, expected) in enumerate(inputs):
        with torch.npu.stream(streams[i % 2]):
            outputs.append(
                (
                    torch.ops.flash.npu_index_k_route_stats(
                        groups, lengths, pages, 64, 1
                    ),
                    expected,
                )
            )
    for stream in streams:
        stream.synchronize()
    for actual, expected in outputs:
        assert torch.equal(actual.cpu(), expected)
