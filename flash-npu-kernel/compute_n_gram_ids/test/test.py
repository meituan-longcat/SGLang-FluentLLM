import torch
import torch_npu
import torchair

# def test_eager(x, y):
#     return torch.ops.npu.mlp_add_custom(x, y)
#
# config = torchair.CompilerConfig()
# config.mode = "reduce-overhead"        # 表示aclgraph模式
# @torch.compile(backend=torchair.get_npu_backend(compiler_config=config))
# def test_torchair_reduce_overhead(x, y):
#     return torch.ops.npu.mlp_add_custom(x, y)
#
# config = torchair.CompilerConfig()
# config.mode = "max-autotune"          # 表示Ascend IR模式
# @torch.compile(backend=torchair.get_npu_backend(compiler_config=config))
# def test_torchair_max_autotune(x, y):
#     return torch.ops.npu.mlp_add_custom(x, y)
#
# x = torch.rand([16], dtype=torch.float16).npu()
# y = torch.rand([16], dtype=torch.float16).npu()
#
# # print(test_eager(x, y))
# # torch.npu.synchronize()
# # print("Eager ok")
# # print(test_torchair_reduce_overhead(x, y))
# # torch.npu.synchronize()
# # print("TorchAir-reduce-overhead ok")
# print(test_torchair_max_autotune(x, y))
# torch.npu.synchronize()
# print("TorchAir-max-autotune ok")
# print(x+y)

