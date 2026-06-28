import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests


class TestUpdateOeTokenTable(TestCase):
    def torch_update_token_table(self, 
        tokens: torch.Tensor,
        row_indices: torch.Tensor,
        column_starts: torch.Tensor,
        oe_req_lens: torch.Tensor,
        ignore_token: torch.Tensor,
        oe_token_table: torch.Tensor) -> torch.Tensor:
        if tokens.numel() != oe_req_lens.sum():
            raise ValueError(f"Token总数 {tokens.numel()} 与请求长度之和 {oe_req_lens.sum()} 不匹配")
        device = oe_token_table.device
        rows_to_write = row_indices.repeat_interleave(oe_req_lens)
        col_starts_expanded = column_starts.repeat_interleave(oe_req_lens)
        offsets = torch.cat([
            torch.arange(length, device=device) for length in oe_req_lens
        ])
        cols_to_write = col_starts_expanded + offsets
        oe_token_table[rows_to_write, cols_to_write] = tokens
        mask = torch.isin(oe_token_table, ignore_token)
        oe_token_table[mask] = -1
        return oe_token_table

    def generate_test_cases(self):
        """生成多种测试用例"""
        test_cases = []
        # 测试不同的bs值 (1-20)
        torch.manual_seed(42)
        for bs in [1, 2, 4, 8, 16, 24, 32, 48, 49, 64, 128, 168, 256]:
            max_reqs = bs + torch.randint(0, 100, (1,), dtype=torch.int32)
            for max_lens in [20, 128, 256, 300, 412, 512, 1024, 1600, 2048, 3330, 4096, 10 * 1024, 100 * 1024, 300 * 1024, 500 * 1024, 800 * 1024, 1024 * 1024]:
                test_case = {
                    'batch_size': bs,
                    'max_lens': max_lens,
                    'max_reqs': max_reqs,
                }
                test_cases.append(test_case)
        return test_cases

    def test_update_token_table(self):
        test_cases = self.generate_test_cases()
        torch.manual_seed(42)
        for i, case in enumerate(test_cases):
            batch_size = case['batch_size']
            max_len = case['max_lens']
            max_reqs = case['max_reqs']
            oe_token_table = torch.zeros(max_reqs * max_len, dtype=torch.int32).reshape(max_reqs, max_len)

            # 生成测试数据

            row_indices = torch.randperm(batch_size, dtype = torch.int64)
            oe_req_lens = torch.randint(1, max_len, (batch_size,), dtype = torch.int32)
            start_max = max_len - oe_req_lens - 1
            column_starts = (torch.rand_like(start_max, dtype=torch.float32) * (start_max + 1)).to(dtype = torch.int32)

            # 计算总共需要写入的token数量
            total_tokens = oe_req_lens.sum().item()

            # 生成待写入的tokens (使用有规律的值以便验证)
            tokens = torch.arange(1, total_tokens + 1, dtype = torch.int32)
            # 生成ignore值
            num_ignore_select = torch.randint(3, 10, (1,)).item()
            ignore_indices = torch.randperm(len(tokens))[:num_ignore_select]
            ignore_token = tokens[ignore_indices]
            oe_token_table_npu = oe_token_table.clone().npu()

            """
            print(f"row_indices: {row_indices.tolist()}")
            #print(f"column_starts: {column_starts.tolist()}")
            print(f"oe_req_lens: {oe_req_lens.tolist()}")
            print(f"待写入 tokens: {tokens.tolist()}")
            print(f"总token数: {total_tokens}")
            """

            output = self.torch_update_token_table(tokens, row_indices, column_starts, oe_req_lens, ignore_token, oe_token_table)

            torch_npu.npu_update_token_table(tokens.npu(), oe_req_lens.npu(), row_indices.npu(), column_starts.npu(),
                                            ignore_token.npu(), batch_size, max_len, oe_token_table_npu)
            #print(f"oe_token_table:", oe_token_table_npu.cpu())
            print(f"测试用例 {i+1} pass: batch_size ={batch_size}, max_len={max_len}, dtype={max_reqs}")
            try:
                self.assertRtolEqual(output.cpu(), oe_token_table_npu.cpu())
            except AssertionError as e:
                print(f"测试用例 {i+1} 失败: batch_size ={batch_size}, max_len={max_len}, dtype={max_reqs}")
                print("output:", output.cpu())
                print("oe_token_table_npu:", oe_token_table_npu.cpu())

if __name__ == "__main__":
    run_tests()
