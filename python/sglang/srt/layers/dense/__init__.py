
from sglang.srt.utils import is_npu
__is_npu__ = is_npu()
if not __is_npu__:
    from sglang.srt.layers.dense.layouts.fp8 import Fp8LinearMethod

