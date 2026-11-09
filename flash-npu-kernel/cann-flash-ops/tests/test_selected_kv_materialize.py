"""NPU exactness for batched Selected-KV materialization.

Run with the matching complete flash_ops package and OPP installed.
The CPU reference independently decodes packed KV and verifies padding.
"""

import torch
import torch_npu  # noqa: F401 - registers device/operator implementations

# flash_ops links against libraries loaded by torch/torch_npu.
import flash_ops  # noqa: F401  # isort: skip


def main():
    torch.npu.set_device(0)
    torch.manual_seed(20261010)
    torch.set_num_threads(2)
    fn = torch.ops.custom.npu_sparse_kv_materialize
    for width in [656, 672]:
        for n, e in [
            (0, 1),
            (0, 16384),
            (1, 0),
            (15, 17),
            (16, 128),
            (17, 15),
            (47, 1),
            (48, 127),
            (49, 128),
            (127, 129),
            (128, 17),
            (129, 0),
            (767, 1),
            (768, 0),
            (769, 17),
            (19282, 16384),
            (48926, 16384),
            (90824, 2048),
        ]:
            packed = torch.randint(-128, 128, (n, 1, width), dtype=torch.int8)
            rope = torch.randn(n, 1, 64).bfloat16()
            scales = torch.rand(n, 1, 4) * 0.02 + 0.0001
            packed[:, :, 512:640] = rope.view(torch.int8)
            packed[:, :, 640:656] = scales.view(torch.int8)
            ext = torch.randn(e, 576).bfloat16()
            want = torch.zeros(((n + e + 127) // 128) * 128, 576, dtype=torch.bfloat16)
            want[:n, :512] = (packed[:, :, :512].float().reshape(n, 4, 128) * scales.reshape(n, 4, 1)).reshape(n, 512).bfloat16()
            want[:n, 512:] = rope.reshape(n, 64)
            want[n : n + e] = ext
            p, x = packed.npu(), ext.npu()
            for _ in range(2):
                actual = fn(p, x)
                torch.npu.synchronize()
                assert torch.equal(actual.cpu().reshape(-1, 576), want), (
                    width,
                    n,
                    e,
                    (actual.cpu().reshape(-1, 576).float() - want.float()).abs().max().item(),
                )
            print("MATERIALIZE_EXACT", width, n, e, flush=True)
            del p, x, actual
    print("MATERIALIZE_PASS", flush=True)


if __name__ == "__main__":
    main()
