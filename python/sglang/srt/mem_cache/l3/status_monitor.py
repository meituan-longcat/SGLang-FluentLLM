from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional


def parse_l3_status_config(
    extra_config: dict, storage_backend_type: Optional[str]
) -> tuple[int, float]:
    """Consume status-monitor config only for the Mooncake backend."""
    if storage_backend_type != "mooncake":
        return 1, 30.0

    failure_threshold = extra_config.pop("l3_failure_threshold", 1)
    recovery_interval_sec = extra_config.pop("l3_recovery_interval_sec", 30.0)
    if not isinstance(failure_threshold, int) or failure_threshold < 1:
        raise ValueError("l3_failure_threshold must be a positive int")
    if not isinstance(recovery_interval_sec, (int, float)) or recovery_interval_sec < 0:
        raise ValueError("l3_recovery_interval_sec must be a non-negative number")
    return failure_threshold, float(recovery_interval_sec)


@dataclass(frozen=True)
class L3OperationToken:
    epoch: int
    is_recovery_check: bool = False


class L3StatusMonitor:
    """Track whether Mooncake L3 is healthy enough to receive operations."""

    def __init__(
        self,
        failure_threshold: int,
        recovery_interval_sec: float,
        clock: Callable[[], float] = time.monotonic,
    ):
        if failure_threshold < 1:
            raise ValueError("failure_threshold must be at least 1")
        if recovery_interval_sec < 0:
            raise ValueError("recovery_interval_sec must be non-negative")

        self._failure_threshold = failure_threshold
        self._recovery_interval_sec = recovery_interval_sec
        self._clock = clock
        self._lock = threading.Lock()
        self._consecutive_failures = 0
        self._unavailable_until = 0.0
        self._epoch = 0
        self._recovery_check_inflight = False

    def begin_operation(self) -> Optional[L3OperationToken]:
        with self._lock:
            now = self._clock()
            if self._unavailable_until == 0.0:
                return L3OperationToken(self._epoch)
            if now < self._unavailable_until or self._recovery_check_inflight:
                return None

            self._recovery_check_inflight = True
            return L3OperationToken(self._epoch, is_recovery_check=True)

    def report_failure(self, token: L3OperationToken) -> bool:
        """Record a failure and return whether L3 became unavailable."""
        with self._lock:
            if token.epoch != self._epoch:
                return False

            self._consecutive_failures += 1
            if (
                not token.is_recovery_check
                and self._consecutive_failures < self._failure_threshold
            ):
                return False

            self._epoch += 1
            self._unavailable_until = self._clock() + self._recovery_interval_sec
            self._recovery_check_inflight = False
            return True

    def report_success(self, token: L3OperationToken) -> bool:
        """Record success and return whether a recovery check restored L3."""
        with self._lock:
            if token.epoch != self._epoch:
                return False

            if token.is_recovery_check:
                self._epoch += 1
                self._unavailable_until = 0.0
                self._recovery_check_inflight = False
                self._consecutive_failures = 0
                return True

            if self._unavailable_until == 0.0:
                self._consecutive_failures = 0
            return False

    def cancel_check(self, token: L3OperationToken) -> None:
        """Release a no-op recovery check without marking L3 healthy."""
        with self._lock:
            if (
                token.epoch == self._epoch
                and token.is_recovery_check
                and self._recovery_check_inflight
            ):
                self._recovery_check_inflight = False
