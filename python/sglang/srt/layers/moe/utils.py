from __future__ import annotations

import logging
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


class MoeA2ABackend(Enum):
    NONE = "none"
    DEEPEP = "deepep"

    @classmethod
    def _missing_(cls, value):
        if value is None:
            return cls.NONE
        for member in cls:
            if value == member.value:
                return member
        raise ValueError(f"No {cls.__name__} member for value {value}")

    def is_none(self):
        return self == MoeA2ABackend.NONE

    def is_deepep(self):
        return self == MoeA2ABackend.DEEPEP


class MoeRunnerBackend(Enum):
    AUTO = "auto"
    TRITON = "triton"
    TRITON_KERNEL = "triton_kernel"
    FLASHINFER_TRTLLM = "flashinfer_trtllm"
    FLASHINFER_CUTLASS = "flashinfer_cutlass"
    FLASHINFER_MXFP4 = "flashinfer_mxfp4"
    FLASHINFER_CUTEDSL = "flashinfer_cutedsl"

    def is_auto(self):
        return self == MoeRunnerBackend.AUTO

    def is_triton(self):
        return self == MoeRunnerBackend.TRITON

    def is_triton_kernel(self):
        return self == MoeRunnerBackend.TRITON_KERNEL

    def is_flashinfer_trtllm(self):
        return self == MoeRunnerBackend.FLASHINFER_TRTLLM

    def is_flashinfer_cutlass(self):
        return self == MoeRunnerBackend.FLASHINFER_CUTLASS

    def is_flashinfer_cutedsl(self):
        return self == MoeRunnerBackend.FLASHINFER_CUTEDSL

    def is_flashinfer_mxfp4(self):
        return self == MoeRunnerBackend.FLASHINFER_MXFP4


class DeepEPMode(Enum):
    NORMAL = "normal"
    LOW_LATENCY = "low_latency"
    AUTO = "auto"

    def enable_normal(self) -> bool:
        return self in [DeepEPMode.NORMAL, DeepEPMode.AUTO]

    def enable_low_latency(self) -> bool:
        return self in [DeepEPMode.LOW_LATENCY, DeepEPMode.AUTO]

    def resolve(self, is_extend_in_batch: bool) -> DeepEPMode:
        if self != DeepEPMode.AUTO:
            return self

        if is_extend_in_batch:
            return DeepEPMode.NORMAL
        else:
            return DeepEPMode.LOW_LATENCY

    def is_normal(self) -> bool:
        return self == DeepEPMode.NORMAL

    def is_low_latency(self) -> bool:
        return self == DeepEPMode.LOW_LATENCY

    def is_auto(self) -> bool:
        return self == DeepEPMode.AUTO


MOE_A2A_BACKEND: MoeA2ABackend | None = None
MOE_RUNNER_BACKEND: MoeRunnerBackend | None = None
DEEPEP_MODE: DeepEPMode | None = None
IS_TBO_ENABLED: bool | None = None
IS_SBO_ENABLED: bool | None = None
TBO_TOKEN_DISTRIBUTION_THRESHOLD: float | None = None
DEEPEP_CONFIG: str | None = None
DISABLE_FLASHINFER_CUTLASS_MOE_FP4_ALLGATHER: bool | None = None


def initialize_moe_config(server_args: ServerArgs):
    global MOE_A2A_BACKEND
    global MOE_RUNNER_BACKEND
    global DEEPEP_MODE
    global DEEPEP_CONFIG
    global IS_TBO_ENABLED
    global IS_SBO_ENABLED
    global TBO_TOKEN_DISTRIBUTION_THRESHOLD
    global DISABLE_FLASHINFER_CUTLASS_MOE_FP4_ALLGATHER

    MOE_A2A_BACKEND = MoeA2ABackend(server_args.moe_a2a_backend)
    MOE_RUNNER_BACKEND = MoeRunnerBackend(server_args.moe_runner_backend)
    DEEPEP_MODE = DeepEPMode(server_args.deepep_mode)
    DEEPEP_CONFIG = server_args.deepep_config or ""
    IS_TBO_ENABLED = server_args.enable_two_batch_overlap
    IS_SBO_ENABLED = server_args.enable_single_batch_overlap
    TBO_TOKEN_DISTRIBUTION_THRESHOLD = server_args.tbo_token_distribution_threshold
    DISABLE_FLASHINFER_CUTLASS_MOE_FP4_ALLGATHER = server_args.disable_flashinfer_cutlass_moe_fp4_allgather


def get_moe_a2a_backend() -> MoeA2ABackend:
    global MOE_A2A_BACKEND
    if MOE_A2A_BACKEND is None:
        logger.warning("MOE_A2A_BACKEND is not initialized, using default backend")
        MOE_A2A_BACKEND = MoeA2ABackend.NONE
    return MOE_A2A_BACKEND


def get_moe_runner_backend() -> MoeRunnerBackend:
    global MOE_RUNNER_BACKEND
    if MOE_RUNNER_BACKEND is None:
        logger.warning("MOE_RUNNER_BACKEND is not initialized, using triton backend")
        MOE_RUNNER_BACKEND = MoeRunnerBackend.AUTO
    return MOE_RUNNER_BACKEND


def get_deepep_mode() -> DeepEPMode:
    global DEEPEP_MODE
    if DEEPEP_MODE is None:
        logger.warning("DEEPEP_MODE is not initialized, using auto mode")
        DEEPEP_MODE = DeepEPMode.AUTO
    return DEEPEP_MODE


def get_deepep_config() -> str:
    global DEEPEP_CONFIG
    if DEEPEP_CONFIG is None:
        logger.warning("DEEPEP_CONFIG is not initialized, using default config")
        DEEPEP_CONFIG = ""
    return DEEPEP_CONFIG


def is_tbo_enabled() -> bool:
    global IS_TBO_ENABLED
    if IS_TBO_ENABLED is None:
        IS_TBO_ENABLED = False
    return IS_TBO_ENABLED


def is_sbo_enabled() -> bool:
    global IS_SBO_ENABLED
    if IS_SBO_ENABLED is None:
        IS_SBO_ENABLED = False
    return IS_SBO_ENABLED


def get_tbo_token_distribution_threshold() -> float:
    global TBO_TOKEN_DISTRIBUTION_THRESHOLD
    if TBO_TOKEN_DISTRIBUTION_THRESHOLD is None:
        logger.warning("TBO_TOKEN_DISTRIBUTION_THRESHOLD is not initialized, using 0.48")
        TBO_TOKEN_DISTRIBUTION_THRESHOLD = 0.48
    return TBO_TOKEN_DISTRIBUTION_THRESHOLD
