import unittest

import torch
import torch_npu


def _route(x, expert_idx, *, scale=None, count_flag=True, count_type=1):
    return torch_npu.npu_moe_init_routing_v2(
        x,
        expert_idx,
        scale=scale,
        active_num=expert_idx.numel(),
        expert_capacity=-1,
        expert_num=4,
        drop_pad_mode=0,
        expert_tokens_num_type=count_type,
        expert_tokens_num_flag=count_flag,
        quant_mode=-1,
        active_expert_range=[],
        row_idx_type=1,
    )


@unittest.skipUnless(torch.npu.is_available(), "NPU device not found")
class TestMoeInitRoutingV3(unittest.TestCase):
    def test_float32_unquantized_routing(self):
        x = torch.arange(32, dtype=torch.float32).reshape(4, 8).npu()
        expert_idx = torch.tensor(
            [[1, 0], [2, 3], [1, 3], [0, 2]],
            dtype=torch.int32,
            device="npu",
        )
        scale = torch.tensor(
            [10, 20, 30, 40], dtype=torch.float32, device="npu"
        )

        outputs = _route(x, expert_idx, scale=scale)
        torch.npu.synchronize()
        expanded_x, expanded_row_idx, count, expanded_scale = outputs

        expected_x = torch.tensor(
            [
                [0, 1, 2, 3, 4, 5, 6, 7],
                [24, 25, 26, 27, 28, 29, 30, 31],
                [0, 1, 2, 3, 4, 5, 6, 7],
                [16, 17, 18, 19, 20, 21, 22, 23],
                [8, 9, 10, 11, 12, 13, 14, 15],
                [24, 25, 26, 27, 28, 29, 30, 31],
                [8, 9, 10, 11, 12, 13, 14, 15],
                [16, 17, 18, 19, 20, 21, 22, 23],
            ],
            dtype=torch.float32,
        )
        self.assertTrue(torch.equal(expanded_x.cpu(), expected_x))
        self.assertTrue(
            torch.equal(
                expanded_row_idx.cpu(), torch.tensor([1, 6, 0, 4, 2, 7, 3, 5])
            )
        )
        self.assertTrue(
            torch.equal(count.cpu(), torch.tensor([2, 2, 2, 2], dtype=torch.int64))
        )
        self.assertTrue(
            torch.equal(
                expanded_scale.cpu(),
                torch.tensor(
                    [10, 40, 10, 30, 20, 40, 20, 30], dtype=torch.float32
                ),
            )
        )

    def test_int32_unquantized_routing(self):
        expected_x = torch.tensor(
            [
                [0, 1, 2, 3],
                [12, 13, 14, 15],
                [0, 1, 2, 3],
                [8, 9, 10, 11],
                [4, 5, 6, 7],
                [12, 13, 14, 15],
                [4, 5, 6, 7],
                [8, 9, 10, 11],
            ],
            dtype=torch.int32,
        )
        for count_flag in (False, True):
            with self.subTest(count_flag=count_flag):
                x = torch.arange(16, dtype=torch.int32).reshape(4, 4).npu()
                expert_idx = torch.tensor(
                    [[1, 0], [2, 3], [1, 3], [0, 2]],
                    dtype=torch.int32,
                    device="npu",
                )
                outputs = _route(
                    x,
                    expert_idx,
                    count_flag=count_flag,
                    count_type=0,
                )
                torch.npu.synchronize()
                expanded_x, expanded_row_idx, count, _ = outputs

                self.assertTrue(torch.equal(expanded_x.cpu(), expected_x))
                self.assertTrue(
                    torch.equal(
                        expanded_row_idx.cpu(),
                        torch.tensor([1, 6, 0, 4, 2, 7, 3, 5]),
                    )
                )
                if count_flag:
                    self.assertTrue(
                        torch.equal(
                            count.cpu(), torch.tensor([2, 4, 6, 8], dtype=torch.int64)
                        )
                    )


if __name__ == "__main__":
    unittest.main()
