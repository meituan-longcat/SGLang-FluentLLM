import numpy as np
import torch
import torch_npu

import flash_ops  # noqa: F401 - registers torch.ops.custom


def make_packed_kv(num_blocks: int, block_size: int) -> torch.Tensor:
    # C8 combine layout: 512 int8 KV bytes, 64 bf16 rope values, then four
    # float32 per-128-element scales.
    packed = np.zeros((num_blocks, block_size, 1, 656), dtype=np.int8)
    scale_bytes = np.frombuffer(np.ones(4, dtype=np.float32).tobytes(), dtype=np.int8)
    packed[..., 640:656] = scale_bytes
    return torch.from_numpy(packed).npu()


def main() -> None:
    torch.npu.set_device(0)
    token_count = 32
    query_heads = 64
    sparse_count = 16

    query = torch.randn(token_count, query_heads, 576, dtype=torch.bfloat16).npu()
    packed_kv = make_packed_kv(1, 128)
    topk = torch.arange(sparse_count, dtype=torch.int32).view(1, 1, -1)
    topk = topk.expand(token_count, 1, -1).contiguous().npu()
    block_table = torch.zeros((1, 1), dtype=torch.int32).npu()
    actual_q = torch.tensor([token_count], dtype=torch.int32).npu()
    actual_kv = torch.tensor([token_count], dtype=torch.int32).npu()

    kwargs = dict(
        block_table=block_table,
        actual_seq_lengths_query=actual_q,
        actual_seq_lengths_kv=actual_kv,
        scale_value=1.0,
        key_quant_mode=2,
        value_quant_mode=2,
        sparse_block_size=1,
        layout_query="TND",
        layout_kv="PA_BSND",
        sparse_mode=3,
        attention_mode=2,
        quant_scale_repo_mode=1,
        tile_size=128,
        rope_head_dim=64,
        return_softmax_lse=True,
    )
    old_o, old_max, old_sum = torch.ops.custom.npu_kv_quant_sparse_flash_attention_lse(
        query, packed_kv, packed_kv, topk, **kwargs
    )
    new_o, new_max, new_sum = torch.ops.custom.npu_kv_quant_sparse_flash_attention_lse(
        query, packed_kv, packed_kv, topk, softmax_lse_tnd=True, **kwargs
    )
    torch.npu.synchronize()

    expected_max = old_max.permute(1, 0, 2).reshape(token_count, query_heads, 1)
    expected_sum = old_sum.permute(1, 0, 2).reshape(token_count, query_heads, 1)
    torch.testing.assert_close(new_o.cpu(), old_o.cpu(), rtol=0, atol=0)
    torch.testing.assert_close(new_max.cpu(), expected_max.cpu(), rtol=0, atol=0)
    torch.testing.assert_close(new_sum.cpu(), expected_sum.cpu(), rtol=0, atol=0)
    print("PASS", tuple(new_o.shape), tuple(new_max.shape), tuple(new_sum.shape))


if __name__ == "__main__":
    main()
