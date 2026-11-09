from typing import List, Tuple
import torch
import triton


def prepare_block_fp8_matmul_inputs(
    A: torch.Tensor,
    B: torch.Tensor,
    As: torch.Tensor,
    Bs: torch.Tensor,
    block_size: List[int],
    output_dtype: torch.dtype = torch.float16,
) -> Tuple[int, int, int]:
    assert len(block_size) == 2
    block_n, block_k = block_size[0], block_size[1]

    assert A.shape[-1] == B.shape[-1]
    assert A.shape[:-1] == As.shape[:-1]
    assert A.is_contiguous()

    if As.dtype == torch.float:
        assert triton.cdiv(A.shape[-1], block_k) == As.shape[-1]
    elif As.dtype == torch.int:
        assert triton.cdiv(triton.cdiv(A.shape[-1], block_k), 4) == As.shape[-1], f"{A.shape=} {As.shape=} {block_size=}"
    else:
        raise NotImplementedError

    M = A.numel() // A.shape[-1]

    assert B.ndim == 2
    assert B.is_contiguous()
    assert Bs.ndim == 2
    N, K = B.shape

    if Bs.dtype == torch.float:
        assert triton.cdiv(N, block_n) == Bs.shape[0]
        assert triton.cdiv(K, block_k) == Bs.shape[1]
    elif Bs.dtype == torch.int:
        assert N == Bs.shape[0], f"{B.shape=} {Bs.shape=} {block_size=}"
        assert triton.cdiv(triton.cdiv(K, block_k), 4) == Bs.shape[1], f"{B.shape=} {Bs.shape=} {block_size=}"
    else:
        raise NotImplementedError

    C_shape = A.shape[:-1] + (N,)
    C = A.new_empty(C_shape, dtype=output_dtype)

    return M, N, K, C
