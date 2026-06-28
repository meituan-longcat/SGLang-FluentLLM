import torch
import torch_npu
import numpy as np
from torch_npu.testing.testcase import TestCase, run_tests

torch.npu.config.allow_internal_format = False
torch.npu.set_compile_mode(jit_compile=False)

class TestAssignReqToTokenPool(TestCase):

    def get_out_cache_loc_pytorch(self,
        bs,
        out_cache_loc,
        req_pool_indices,
        new_compute_lens,
        cache_lens,
        req_to_token
    ):
        # 1、构造累加和
        cumsum_offsets = torch.zeros_like(new_compute_lens)
        torch.cumsum(new_compute_lens[:-1], dim=0, out=cumsum_offsets[1:])
        cache_starts = cache_lens[req_pool_indices]
        # 2、 构造行索引
        row_indices = torch.repeat_interleave(req_pool_indices, new_compute_lens)
        # 3、 构造列索引
        total_new_compute = torch.sum(new_compute_lens)
        col_indices = (torch.repeat_interleave(cache_starts, new_compute_lens) +
                    torch.arange(total_new_compute, dtype=new_compute_lens.dtype, device=new_compute_lens.device) -
                    torch.repeat_interleave(cumsum_offsets, new_compute_lens))
        # 4、 按照索引搬出数据
        out_cache_loc[:col_indices.size(0)] = req_to_token[row_indices, col_indices]
        return out_cache_loc

    def generate_test_cases(self):
        """生成多种测试用例"""
        np.random.seed(42)  # 固定随机种子以便复现
        test_cases = []
        for bs in range(1, 192):
            for alloc_size in [16, 64, 128, 256, 1024, 2048, 4096, 6400, 8192, 16384, 32768, 65535, 131070]:
                for dtype in [torch.int64, torch.int32]:
                    test_cases.append((bs, alloc_size, dtype))
        return test_cases

    def _run_single_test_case(self, bs, pool_len, dtype):
        """运行单个测试用例的辅助函数"""
        max_cache_len = pool_len // 4
        req_to_token = torch.arange(1, bs * pool_len+1, dtype=dtype).reshape(bs, pool_len)
        torch.manual_seed(42)
        req_pool_indices = torch.randperm(bs, dtype=dtype)
        cache_lens = torch.randint(0, max_cache_len, (bs,), dtype=dtype)
        new_compute_lens = torch.randint(1, pool_len - max_cache_len, (bs,), dtype=dtype)

        total_tokens = new_compute_lens.sum().item()
        out_cache_loc = torch.zeros(total_tokens, dtype=dtype)
        out_cache_loc_npu = out_cache_loc.clone().npu()

        # Run PyTorch implementation
        self.get_out_cache_loc_pytorch(bs, out_cache_loc, req_pool_indices,
                                  new_compute_lens, cache_lens, req_to_token)

        torch_npu.npu_get_out_cache_loc(req_to_token.npu(), req_pool_indices.npu(),
                new_compute_lens.npu(), cache_lens.npu(), out_cache_loc_npu, bs)
        torch.npu.synchronize()
        
        try:
            self.assertRtolEqual(out_cache_loc.cpu(), out_cache_loc_npu.cpu())
        except AssertionError as e:
            print(f"测试用例 失败: bs={bs}, pool_len={pool_len}")
            print("output:", out_cache_loc.cpu())
            print("req_to_token:", out_cache_loc_npu.cpu())

    def test_get_out_cache_loc_random_combinations(self):
        """随机组合测试"""
        test_cases = self.generate_test_cases()
        success_count = 0
        for bs, pool_len, dtype in test_cases:
            print(f"\n 用例:{success_count} 参数: bs={bs:2d}, pool_len={pool_len:5d}, dtype={dtype}", end=" ")
            self._run_single_test_case(bs=bs, pool_len=pool_len, dtype=dtype)
            success_count += 1

        print(f"\n测试结果: {success_count} 通过")

if __name__ == "__main__":
    run_tests()
