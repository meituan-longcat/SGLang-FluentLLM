from enum import Enum


class FusedWeightScaleSupported(Enum):
    TENSOR = "tensor"
    CHANNEL = "channel"
    GROUP = "group"
    BLOCK = "block"
