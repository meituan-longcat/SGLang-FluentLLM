from typing import Callable

from sglang.srt.layers.dense.gemms.fp8.flashinfer import ENABLE_FLASHINFER_GEMM
from sglang.srt.utils import is_sm89_supported, is_sm90_supported


def dispatch_w8a8_block_fp8_linear() -> Callable:
    if is_sm89_supported():
        from sglang.srt.layers.dense.gemms.fp8.cutlass import (
            cutlass_w8a8_block_fp8_linear_with_fallback,
        )

        return cutlass_w8a8_block_fp8_linear_with_fallback
    elif is_sm90_supported():
        from sglang.srt.layers.dense.gemms.fp8.deep_geem import (
            deepgemm_w8a8_block_fp8_linear_with_fallback,
        )

        return deepgemm_w8a8_block_fp8_linear_with_fallback
    elif ENABLE_FLASHINFER_GEMM:
        from sglang.srt.layers.dense.gemms.fp8.flashinfer import (
            flashinfer_gemm_w8a8_block_fp8_linear,
        )

        return flashinfer_gemm_w8a8_block_fp8_linear
    else:
        raise RuntimeError("No supported block fp8 gemm")
