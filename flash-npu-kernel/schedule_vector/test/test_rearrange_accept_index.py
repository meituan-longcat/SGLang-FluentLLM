import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests

torch.npu.config.allow_internal_format = False
torch.npu.set_compile_mode(jit_compile=False)

class TestRearrangeAcceptIndex(TestCase):
    def rearrange_accept_index(self, accept_index, accept_length, bs, output_torch):
        # 1、先创建一个和accept_index.shape()相同的range_tensor
        range_tensor = torch.arange(accept_index.size(1), device=accept_index.device).expand(bs, -1)
        expanded_length = torch.unsqueeze(accept_length, dim=1)
        # 2、根据expanded_length 计算出mask
        mask = range_tensor < expanded_length
        # 3、根据mask，取到accept_index中的有效数据
        valid_elements = accept_index[mask]   # 耗时操作
        # 4、 根据有效数据的shape，搬运有效值到输出
        output_torch[:valid_elements.size(0)] = valid_elements
        return output_torch

    def generate_test_case(self):
        test_cases = []
        for bs in range(1, 180):
            for pool_size in [16, 64, 128, 256, 1024, 2048, 4096, 1600, 6400, 8192, 16384, 32768, 65535, 131070]:
                for dtype in [torch.int64, torch.int32]:
                    test_case = {
                        'bs': bs,
                        'pool_size': pool_size,
                        'dtype': dtype
                    }
                    test_cases.append(test_case)
        return test_cases
    
    def test_rearrange_accept_index(self):
        test_cases = self.generate_test_case()
        for i, case in enumerate(test_cases):
            bs = case['bs']
            pool_size = case['pool_size']
            dtype = case['dtype']
            print(f"\n 用例: {i+1}/{len(test_cases)} 参数: bs={bs:2d}, pool_len={pool_size:5d}, dtype={dtype}", end=" ")

            torch.manual_seed(42)
            accept_index = torch.arange(0, bs * pool_size, dtype=dtype, device='cpu').reshape(bs, pool_size)
            accept_length = torch.randint(1, pool_size, (bs,), dtype=dtype)
            output_cpu = torch.zeros((torch.sum(accept_length)), dtype=dtype, device='cpu')
            output_npu = torch.zeros((torch.sum(accept_length)), dtype=dtype, device='npu')
            output_cpu = self.rearrange_accept_index(accept_index, accept_length, bs, output_cpu)
            torch_npu.npu_rearrange_accept_index(accept_index.npu(), accept_length.npu(), bs, output_npu)
            torch.npu.synchronize()

            try:
                self.assertRtolEqual(output_cpu.cpu(), output_npu.cpu())
            except AssertionError as e:
                print(f"测试用例 {i+1} 失败: bs={bs}, pool_size={pool_size}, dtype={dtype}")
                print("output:", output_cpu.cpu())
                print("req_to_token:", output_npu.cpu())

if __name__ == "__main__":
    run_tests()