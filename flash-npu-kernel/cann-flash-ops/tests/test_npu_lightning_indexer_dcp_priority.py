"""NPU regression for forced-score priority after cross-rank TopK.

Run with ASCEND_CUSTOM_OPP_PATH pointing at the tested vendor package and
LI_VENDOR_OPAPI pointing at its op_api/lib/libcust_opapi.so. The existing
ACLNN harness loads that exact library, avoiding a stale PyTorch binding.
"""

import os
import unittest

import torch
import torch_npu

from test_npu_lightning_indexer_topk48_init_local import AclnnLightningIndexer, DEFAULT_VENDOR_OPAPI


class TestDcpPriority(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.op = AclnnLightningIndexer(os.environ.get("LI_VENDOR_OPAPI", DEFAULT_VENDOR_OPAPI))

    def run_indexer(self, kv_lens, q_lens, init, local, *, mode, layout, topk, return_value=True, score_one=False):
        total_q = sum(q_lens)
        query = torch.full((total_q, 64, 128), float(score_one), dtype=torch.bfloat16, device="npu")
        weights = torch.ones((total_q, 64), dtype=torch.bfloat16, device="npu")
        block_table = None
        if layout == "TND":
            key = torch.ones((sum(kv_lens), 1, 128), dtype=torch.bfloat16, device="npu")
            actual_k = torch.tensor(kv_lens, dtype=torch.int32).cumsum(0).to(torch.int32)
        else:
            blocks = (max(kv_lens) + 127) // 128
            key = torch.ones((len(kv_lens) * blocks, 128, 1, 128), dtype=torch.bfloat16, device="npu")
            block_table = torch.arange(len(kv_lens) * blocks, dtype=torch.int32, device="npu").reshape(len(kv_lens), blocks)
            actual_k = torch.tensor(kv_lens, dtype=torch.int32)
        actual_q = torch.tensor(q_lens, dtype=torch.int32).cumsum(0).to(torch.int32)
        idx, val = self.op(query, key, weights, actual_q.npu(), actual_k.npu(), block_table,
                           torch.tensor(init, dtype=torch.int32, device="npu"),
                           torch.tensor(local, dtype=torch.int32, device="npu"),
                           "TND", layout, topk, mode, return_value)
        return idx.cpu().squeeze(1), val.cpu().float().squeeze(1) if return_value else None

    def test_exact_forced_sets_and_clipping(self):
        # Short/full/overlapping windows, two requests, moving causal rows,
        # aligned and unaligned tails, and S2 tile boundaries.
        for layout in ("TND", "PA_BSND"):
            for mode in (0, 3):
                for kv_lens, q_lens, init, local, topk in (
                    ([7, 11], [4, 3], [32, 2], [32, 2, 7, 0, 9, 8, 7], 16),
                    ([9], [4], [4], [5, 5, 5, 5], 16),
                    ([521], [4], [513], [4, 3, 2, 1], 544),
                    ([1031], [4], [3], [13, 12, 11, 10], 32),
                    ([4097], [4], [3], [13, 12, 11, 10], 32),
                ):
                    with self.subTest(layout=layout, mode=mode, kv_lens=kv_lens, init=init):
                        indices, values = self.run_indexer(kv_lens, q_lens, init, local, mode=mode, layout=layout, topk=topk)
                        row = 0
                        for b, (kv_len, q_len) in enumerate(zip(kv_lens, q_lens)):
                            for j in range(q_len):
                                valid = kv_len - q_len + j + 1 if mode == 3 else kv_len
                                expected = set(range(min(init[b], valid))) | set(range(max(0, valid - local[row]), valid))
                                forced = set(indices[row][values[row] > 1e30].tolist())
                                self.assertEqual(forced, expected)
                                chosen = indices[row][indices[row] >= 0]
                                self.assertEqual(chosen.numel(), min(valid, topk))
                                self.assertTrue(bool((chosen < valid).all()))
                                self.assertEqual(chosen.unique().numel(), chosen.numel())
                                self.assertTrue(bool(torch.isneginf(values[row][indices[row] < 0]).all()))
                                row += 1

    def test_global_merge_keeps_fully_forced_short_rank(self):
        for layout in ("TND", "PA_BSND"):
            short_idx, short_values = self.run_indexer([4], [1], [2], [2], mode=0, layout=layout, topk=16)
            long_idx, long_values = self.run_indexer([64], [1], [0], [0], mode=0, layout=layout, topk=16, score_one=True)
            all_values = torch.cat((short_values, long_values), dim=1)
            global_ids = torch.cat((short_idx, long_idx + 4), dim=1)
            selected = global_ids.gather(1, all_values.topk(16, dim=1).indices)
            self.assertTrue(set(range(4)).issubset(set(selected[0].tolist())))

    def test_id_only_full_row_remains_valid(self):
        for layout in ("TND", "PA_BSND"):
            indices, _ = self.run_indexer([7], [1], [16], [32], mode=0, layout=layout, topk=16, return_value=False)
            self.assertEqual(set(indices[0][indices[0] >= 0].tolist()), set(range(7)))


if __name__ == "__main__":
    unittest.main()
