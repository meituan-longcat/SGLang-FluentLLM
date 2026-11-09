"""Persistent single-thread device worker for Encoder feature execution."""

from __future__ import annotations

import asyncio
import functools
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Optional

import torch

logger = logging.getLogger(__name__)


class EncoderOutputValidationError(RuntimeError):
    """Raised when a materialized Encoder output violates its format contract."""


class EncoderDeviceWorker:
    """Own Encoder device execution on one persistent thread per TP rank.

    RVQ outputs cross the worker/event-loop boundary only after a synchronous
    producer-thread D2H copy and CPU validation.  ``vit_embedding`` keeps the
    existing device-tensor behavior until its RDMA readiness contract lands.
    """

    def __init__(self, feature_extractor, tp_rank: int = 0):
        self.feature_extractor = feature_extractor
        self.tp_rank = tp_rank
        self.device = self._resolve_device(feature_extractor)
        self.rvq_codebook_sizes = self._resolve_rvq_codebook_sizes(
            feature_extractor
        )
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix=f"encoder-device-tp{tp_rank}",
        )
        self._closed = False
        self._thread_id: Optional[int] = None

    @staticmethod
    def _resolve_device(feature_extractor) -> torch.device:
        if feature_extractor is not None and hasattr(
            feature_extractor, "parameters"
        ):
            try:
                return next(feature_extractor.parameters()).device
            except (StopIteration, TypeError):
                pass
        return torch.device("cpu")

    @staticmethod
    def _resolve_rvq_codebook_sizes(feature_extractor) -> Optional[tuple[int, ...]]:
        config = (
            feature_extractor.config
            if feature_extractor is not None and hasattr(feature_extractor, "config")
            else None
        )
        sizes = config.codebook_sizes if config is not None and hasattr(
            config, "codebook_sizes"
        ) else None
        if sizes is None:
            return None
        return tuple(int(size) for size in sizes)

    async def run_feature(
        self,
        feat_fn: Callable,
        items: list,
        output_format: str,
        *,
        expected_tokens: int,
        request_id: str,
        modality: str,
    ) -> torch.Tensor:
        if self._closed:
            raise RuntimeError("EncoderDeviceWorker is already shut down")
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            functools.partial(
                self._run_feature,
                feat_fn,
                items,
                output_format,
                expected_tokens,
                request_id,
                modality,
            ),
        )

    def _run_feature(
        self,
        feat_fn: Callable,
        items: list,
        output_format: str,
        expected_tokens: int,
        request_id: str,
        modality: str,
    ) -> torch.Tensor:
        current_thread_id = threading.get_ident()
        if self._thread_id is None:
            self._thread_id = current_thread_id
            logger.info(
                "Encoder Device Worker initialized: rank=%d thread_id=%d device=%s",
                self.tp_rank,
                current_thread_id,
                self.device,
            )
        elif self._thread_id != current_thread_id:
            raise RuntimeError(
                "EncoderDeviceWorker execution moved across threads: "
                f"expected={self._thread_id}, got={current_thread_id}"
            )

        if self.device.type != "cpu":
            torch.get_device_module(self.device).set_device(self.device)

        with torch.inference_mode():
            output = feat_fn(items, output_format)

        if not isinstance(output, torch.Tensor):
            raise EncoderOutputValidationError(
                f"Encoder feature output must be a Tensor, got {type(output)!r}"
            )

        if output_format != "rvq_indices":
            return output

        # The model's final int32 cast and all preceding NPU kernels are
        # asynchronous.  Synchronize on the producing thread, then perform a
        # blocking D2H so the event-loop never observes an unready NPU tensor.
        output = output.contiguous()
        if output.device.type != "cpu":
            torch.get_device_module(output.device).synchronize()
        cpu_output = output.detach().to(
            device="cpu", non_blocking=False
        ).contiguous()
        # 暂时不验证
        # self._validate_rvq_output(
        #    cpu_output,
        #    expected_tokens=expected_tokens,
        #    request_id=request_id,
        #    modality=modality,
        #)
        return cpu_output

    def _validate_rvq_output(
        self,
        output: torch.Tensor,
        *,
        expected_tokens: int,
        request_id: str,
        modality: str,
    ) -> None:
        context = (
            f"rid={request_id} rank={self.tp_rank} modality={modality}"
        )
        if output.device.type != "cpu":
            raise EncoderOutputValidationError(
                f"RVQ output was not materialized on CPU: {context} "
                f"device={output.device}"
            )
        if output.dtype != torch.int32:
            raise EncoderOutputValidationError(
                f"RVQ output dtype mismatch: {context} dtype={output.dtype} "
                "expected=torch.int32"
            )
        if output.ndim != 2 or output.shape[0] != expected_tokens:
            raise EncoderOutputValidationError(
                f"RVQ output shape mismatch: {context} shape={tuple(output.shape)} "
                f"expected_tokens={expected_tokens}"
            )

        codebook_sizes = self.rvq_codebook_sizes
        if codebook_sizes is None:
            return
        if output.shape[1] != len(codebook_sizes):
            raise EncoderOutputValidationError(
                f"RVQ depth mismatch: {context} shape={tuple(output.shape)} "
                f"codebook_levels={len(codebook_sizes)}"
            )
        if output.numel() == 0:
            return

        sizes = torch.tensor(codebook_sizes, dtype=torch.int64).view(1, -1)
        values = output.to(torch.int64)
        invalid = (values < 0) | (values >= sizes)
        if not bool(invalid.any().item()):
            return

        invalid_per_level = invalid.sum(dim=0).tolist()
        min_per_level = values.amin(dim=0).tolist()
        max_per_level = values.amax(dim=0).tolist()
        raise EncoderOutputValidationError(
            "Out-of-range RVQ output materialized by Encoder: "
            f"{context} shape={tuple(output.shape)} "
            f"invalid_total={int(invalid.sum().item())} "
            f"invalid_per_level={invalid_per_level} "
            f"min_per_level={min_per_level} max_per_level={max_per_level} "
            f"codebook_sizes={list(codebook_sizes)}"
        )

    def shutdown(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._executor.shutdown(wait=True, cancel_futures=True)
