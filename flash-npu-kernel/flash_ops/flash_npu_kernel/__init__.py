"""flash_npu_kernel"""
__version__ = "1.0.0"

import glob
import os

import torch

_pkg_dir = os.path.dirname(__file__)
_so_list = sorted(glob.glob(os.path.join(_pkg_dir, "libflash_*.so")))
if not _so_list:
    raise ImportError(
        f"No flash op libraries found under {_pkg_dir}. "
        "Please make sure `flash_npu_kernel` is properly installed."
    )
for _so in _so_list:
    torch.ops.load_library(_so)
