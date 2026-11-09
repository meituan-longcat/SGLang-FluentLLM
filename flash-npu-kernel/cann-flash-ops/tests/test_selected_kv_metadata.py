"""Metadata against independent physical-address and compact-index references.

Run after importing the complete matching flash_ops package; no model needed.
"""

import torch
import torch_npu  # noqa: F401 - registers device/operator implementations

# flash_ops links against libraries loaded by torch/torch_npu.
import flash_ops  # noqa: F401  # isort: skip


def check_address_capacity():
    # Largest supported local address, on the last owner. Addresses are
    # token rows, not bytes; exceeding the supported pool must fail early.
    local_rows = 6 * 524288
    world = 8
    owner = world - 1
    physical_page = ((local_rows // 128 - 1 - 1) * world + owner + 1) * 128
    topk = torch.full((1, 2048), -1, dtype=torch.int32, device="npu")
    topk[0, :2] = torch.tensor([0, 127], dtype=torch.int32, device="npu")
    args = (
        topk,
        torch.tensor([[physical_page]], dtype=torch.int32, device="npu"),
        torch.zeros(1, dtype=torch.int32, device="npu"),
        torch.tensor([128], dtype=torch.int32, device="npu"),
        torch.zeros(1, dtype=torch.int32, device="npu"),
        world,
        local_rows,
        1,
    )
    workspace = torch.ops.custom.npu_sparse_kv_build_index_start(*args)
    slots, indices = torch.ops.custom.npu_sparse_kv_build_index_finish(*args, workspace)
    assert slots[:2].cpu().tolist() == [local_rows - 128, local_rows - 1]
    assert indices[0, :2].cpu().tolist() == [0, 1]
    assert int(workspace[3][:, 0].sum().item()) == 2
    try:
        torch.ops.custom.npu_sparse_kv_build_index_start(*args[:6], local_rows + 128, 1)
    except RuntimeError as error:
        assert "metadata workspace capacity exceeded" in str(error)
    else:
        raise AssertionError("Oversized local pool must fail before allocation")
    del workspace, slots, indices


def main():
    torch.npu.set_device(0)
    torch.set_num_threads(2)
    check_address_capacity()
    # Check every possible page quotient, including exact multiples of six.
    p = torch.arange(24576, device="npu", dtype=torch.int32)
    assert torch.equal((p.float() * (1.0 / 6)).floor().int(), p // 6)
    for W, R, rows, Q, B, L, fragmented, empty in [
        (8, 1, 3145728, 17, 2, 4096, False, False),
        (8, 128, 2172672, 2048, 8, 65536, False, False),
        (8, 1, 2172672, 2048, 8, 566912, True, False),
        (1, 1, 16384, 33, 2, 4096, False, False),
        (2, 1, 8192, 33, 2, 4096, True, False),
        (8, 1, 2172672, 2048, 8, 65536, False, True),
    ]:
        torch.manual_seed(1010)
        P = (L + 2048 + 127) // 128
        available = (rows // 128 - R) * W
        pg = (torch.randperm(available)[: B * P] if fragmented else torch.arange(B * P)) + R
        pb = (pg.int().view(B, P) * 128).npu()
        seq = (torch.arange(Q) * B // Q).int().npu()
        prefix = torch.full((B,), L, dtype=torch.int32, device="npu")
        offset = torch.arange(B, dtype=torch.int32, device="npu") * 2048
        topk = torch.randint(L, (Q, 2048), dtype=torch.int32, device="npu")
        topk[:, 0] = -1
        topk[:, 1] = L
        topk[:, 2] = L - 1
        if empty:
            topk.fill_(-1)
        args = (topk, pb, seq, prefix, offset, W, rows, R)
        ws = torch.ops.custom.npu_sparse_kv_build_index_start(*args)
        for position in (0, 3, 4, 6):
            malformed = list(ws)
            malformed[position] = malformed[position].reshape(-1)[:0]
            try:
                torch.ops.custom.npu_sparse_kv_build_index_finish(*args, malformed)
            except RuntimeError as error:
                assert "invalid build_index workspace" in str(error)
            else:
                raise AssertionError("Malformed workspace must fail before launch")
        local, remap = torch.ops.custom.npu_sparse_kv_build_index_finish(*args, ws)
        cnt = ws[3]
        n = int(cnt[:, 0].sum().item())
        counts = cnt[:, 0].view(W, 6).sum(1).cpu()
        owner = torch.repeat_interleave(torch.arange(W), counts).npu()
        local = local[:n].long()
        # Reconstruct global physical addresses independently from owner-local
        # requests and counts, then check every original TopK occurrence.
        unique = (((local // 128 - R) * W + owner + R) * 128 + local % 128).int()
        mask = (topk >= 0) & (topk < L)
        pos = topk.clamp_min(0)
        expected = pb[seq.long()[:, None], pos.long() // 128] + pos % 128
        assert torch.equal(unique[remap[mask].long()], expected[mask]), ("lookup", W, R, fragmented, empty)
        assert torch.equal(unique.sort().values, expected[mask].unique(sorted=True)), ("unique", W, R)
        assert (remap[topk < 0] == -1).all()
        ext = topk >= L
        assert torch.equal(remap[ext], (topk - L + offset[seq.long(), None] + n)[ext])
        print("STRIPED_PLAN_EXACT", W, R, rows, Q, B, L, fragmented, empty, n, flush=True)
    print("STRIPED_PLAN_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
