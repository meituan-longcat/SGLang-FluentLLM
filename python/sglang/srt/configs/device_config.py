from typing import Optional

import torch

from sglang.srt.utils import get_colorful_logger

logger = get_colorful_logger(__name__)


class DeviceConfig:
    device: Optional[torch.device]

    def __init__(self, device: str = "cuda") -> None:
        if device in ["cuda", "npu"]:
            self.device_type = device
        else:
            raise RuntimeError(f"Not supported device type: {device}")
        self.device = torch.device(self.device_type)
