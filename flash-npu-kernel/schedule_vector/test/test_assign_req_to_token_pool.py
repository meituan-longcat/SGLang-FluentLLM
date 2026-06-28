import torch
import torch_npu
import numpy as np
from torch_npu.testing.testcase import TestCase, run_tests
torch.npu.config.allow_internal_format = False
torch.npu.set_compile_mode(jit_compile=False)

class TestAssignReqToTokenPool(TestCase):
    def assign_req_to_token_pool(self, bs, extend_lens, out_cache_loc, req_to_token, alloced_lens, req_pool_indices):
        pt_offsets = torch.zeros_like(extend_lens)
        torch.cumsum(extend_lens[:-1], dim=0, out=pt_offsets[1:])
        current_allocs = alloced_lens[req_pool_indices]
        row_indices = torch.repeat_interleave(req_pool_indices, extend_lens)
        col_offsets = torch.arange(extend_lens.sum(), dtype=extend_lens.dtype, device=extend_lens.device) - \
                    torch.repeat_interleave(pt_offsets, extend_lens)
        col_indices = torch.repeat_interleave(current_allocs, extend_lens) + col_offsets
        req_to_token.index_put_((row_indices, col_indices), out_cache_loc[:extend_lens.sum()])
        return req_to_token

    def generate_test_cases(self):
        """生成多种测试用例"""
        test_cases = []
        
        # 测试不同的bs值 (1-168)
        for bs in range(1, 168):
            for alloc_size in [16, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 24567, 32768, 65535, 131070]:
                # 测试不同的数据类型
                for dtype in [torch.int64, torch.int32]:
                    test_case = {
                        'bs': bs,
                        'alloc_size': alloc_size,
                        'dtype': dtype
                    }
                    test_cases.append(test_case)
        return test_cases

    def test_assign_req_to_token_pool(self):
        test_cases = self.generate_test_cases()
        for i, case in enumerate(test_cases):
            bs = case['bs']
            alloc_size = case['alloc_size']
            dtype = case['dtype']
            print(f"\n 用例: {i+1}/{len(test_cases)} 参数: bs={bs:2d}, pool_len={alloc_size:5d}, dtype={dtype}", end=" ")

            np.random.seed(42)
            torch.manual_seed(42)
            pool_len = alloc_size * 3
            pool_size = np.random.randint(bs, bs * 2) # 确保pool_size至少为bs
            req_to_token = torch.zeros((pool_size, pool_len), dtype=dtype)
            extend_lens = torch.randint(1, 1 * alloc_size, (bs,), dtype=dtype)
            out_cache_loc = torch.arange(1, extend_lens.sum() + 1, dtype=dtype)
            req_pool_indices = torch.randperm(bs, dtype=dtype)
            alloced_lens = torch.randint(1 * alloc_size, 2 * alloc_size, (bs,), dtype=dtype)
            req_to_token_npu = req_to_token.clone().npu()

            output = self.assign_req_to_token_pool(
                bs, extend_lens, out_cache_loc, req_to_token, alloced_lens, req_pool_indices
            )
            torch_npu.npu_assign_req_to_token_pool(req_pool_indices.npu(), extend_lens.npu(),
                   alloced_lens.npu(), out_cache_loc.npu(), req_to_token_npu, bs)
            torch.npu.synchronize()

            try:
                self.assertRtolEqual(output.cpu(), req_to_token_npu.cpu())
            except AssertionError as e:
                print(f"测试用例 {i+1} 失败: bs={bs}, alloc_size={alloc_size}, dtype={dtype}")
                print("output:", output.cpu())
                print("req_to_token:", req_to_token_npu.cpu())

if __name__ == "__main__":
    run_tests()
