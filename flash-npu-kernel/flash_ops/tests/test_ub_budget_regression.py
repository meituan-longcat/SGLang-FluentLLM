"""Device regressions for the two copy operators' aligned UB tiling.

Run only on idle devices with the rebuilt libraries. Boundary lengths below
straddle the old/new BS1 tiles for the measured 196352-byte 910B UB budget.
"""

import pytest
import torch

pytest.importorskip("torch_npu")
import flash_npu_kernel  # noqa: F401, E402


def run_and_check(call, output, expected, mode):
    call()
    assert torch.equal(output.cpu(), expected)
    if mode == "graph":
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            call()
        for _ in range(10):
            output.fill_(-991)
            graph.replay()
        assert torch.equal(output.cpu(), expected)


@pytest.mark.parametrize("bs", [1, 2, 3, 4, 8, 16, 49])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("mode", ["eager", "graph"])
@pytest.mark.parametrize("pattern", ["small", "tile_boundary"])
def test_copy_outputs(bs, dtype, mode, pattern):
    if pattern == "small":
        # Decode, draft/verify and empty copies, including the bonus token.
        choices = [0, 1, 2, 4]
    else:
        choices = (
            [24535, 24536, 24537, 24543, 24544, 24545]
            if dtype == torch.int32
            else [12267, 12268, 12269, 12271, 12272, 12273]
        )
    # Rotate the lengths so BS1 also exercises every boundary, not only the
    # first value. All cases use an independent CPU slice reference.
    for shift in range(len(choices)):
        lengths = torch.tensor(
            [choices[(i + shift) % len(choices)] for i in range(bs)], dtype=dtype
        )
        pool_len = max(choices) + 4
        table = torch.arange(bs * pool_len, dtype=dtype).reshape(bs, pool_len)
        rows = torch.arange(bs - 1, -1, -1, dtype=dtype)
        cache = torch.arange(bs, dtype=dtype) % 3
        total = int(lengths.sum())
        # Guard cells make accidental over-writes visible, including zero copy.
        expected_get = torch.full((total + 8,), -991, dtype=dtype)
        expected_get[:total] = torch.cat(
            [
                table[row, int(cache[row]) : int(cache[row]) + length]
                for row, length in zip(rows.tolist(), lengths.tolist())
            ]
        )
        expected_rearrange = torch.full_like(expected_get, -991)
        expected_rearrange[:total] = torch.cat(
            [table[row, :length] for row, length in enumerate(lengths.tolist())]
        )
        table_npu, rows_npu, lengths_npu, cache_npu = [
            value.npu() for value in (table, rows, lengths, cache)
        ]
        out_get = torch.full_like(expected_get, -991).npu()
        out_rearrange = torch.full_like(expected_rearrange, -991).npu()

        def get():
            torch.ops.flash.npu_get_out_cache_loc(
                table_npu, rows_npu, lengths_npu, cache_npu, out_get, bs
            )

        def rearrange():
            torch.ops.flash.npu_rearrange_accept_index(
                table_npu, lengths_npu, bs, out_rearrange
            )

        run_and_check(get, out_get, expected_get, mode)
        run_and_check(rearrange, out_rearrange, expected_rearrange, mode)
