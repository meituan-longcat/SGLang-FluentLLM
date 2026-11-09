"""flash_npu_kernel"""
__version__ = "1.0.0"

import glob
import os

import torch

_pkg_dir = os.path.dirname(__file__)

_SOC_TO_ARCH = {
    "Ascend910B": "dav_2201",
    "Ascend910_93": "dav_2201",
    "Ascend950": "dav_3510",
}


def _detect_npu_arch():
    arch = os.environ.get("FLASH_NPU_KERNEL_ARCH")
    if arch:
        return arch.replace("-", "_")
    try:
        if torch.npu.is_available():
            device_name = torch.npu.get_device_name(0)
            for soc_prefix, arch_name in _SOC_TO_ARCH.items():
                if device_name.startswith(soc_prefix):
                    return arch_name
    except Exception:
        pass
    return None


_npu_arch = _detect_npu_arch()

if _npu_arch:
    _so_list = sorted(glob.glob(os.path.join(_pkg_dir, f"libflash_*_{_npu_arch}.so")))

if not _npu_arch or not _so_list:
    _so_list = sorted(glob.glob(os.path.join(_pkg_dir, "libflash_*.so")))

if not _so_list:
    raise ImportError(
        f"No flash op libraries found under {_pkg_dir}. "
        "Please make sure `flash_npu_kernel` is properly installed."
    )

for _so in _so_list:
    torch.ops.load_library(_so)
