"""Utility helpers for the Encoder Server package.

Provides :class:`Timer`, a multi-phase monotonic timing recorder used
throughout :mod:`encoder.scheduler` to instrument the critical path.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterator


class Timer:
    """Accumulates elapsed time for multiple named phases of one operation.

    A single instance tracks **all** phases of one logical operation (e.g.
    one ViT batch, one ``get_mm_meta`` call).  This avoids passing individual
    timing values through function signatures — instead the whole ``Timer``
    is forwarded and each callee records its own phase in place.

    Phases are recorded via the :meth:`phase` context manager; externally
    computed values (e.g. ``queue_wait`` derived from ``enqueue_time``) can
    be stored via :meth:`set`.  The creation timestamp is captured so that a
    "total" spanning the whole operation can be stamped with
    :meth:`mark_total`.

    Usage::

        timer = Timer("batch")
        with timer.phase("cache_recheck"):
            ...
        with timer.phase("vit_forward"):
            ...
        timer.set("queue_wait_max", 5.2)
        timer.mark_total()
        logger.info("done %s", timer.format())

    The overhead per phase is a single ``time.monotonic()`` call on enter and
    exit, so it is safe to use on hot paths.
    """

    __slots__ = ("label", "_phases", "_t_start")

    def __init__(self, label: str = ""):
        self.label = label
        self._phases: dict[str, float] = {}
        self._t_start: float = time.monotonic()

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        """Record a named phase's elapsed time (ms) on exit."""
        t0 = time.monotonic()
        try:
            yield
        finally:
            self._phases[name] = (time.monotonic() - t0) * 1000.0

    def set(self, name: str, elapsed_ms: float) -> None:
        """Store an externally-computed timing value (ms)."""
        self._phases[name] = elapsed_ms

    def mark_total(self, name: str = "total") -> None:
        """Stamp the elapsed time since this ``Timer`` was created."""
        self._phases[name] = (time.monotonic() - self._t_start) * 1000.0

    def elapsed_ms(self, name: str) -> float:
        """Return a phase's elapsed time (ms), 0.0 if not recorded."""
        return self._phases.get(name, 0.0)

    def has_phase(self, name: str) -> bool:
        """Return whether *name* was recorded.

        This distinguishes a phase that did not run (for example download and
        decode for pre-extracted video frames) from a phase whose measured
        duration rounded to zero.
        """
        return name in self._phases

    def format(self, *names: str) -> str:
        """Render phases as ``"name=X.Xms name=Y.Yms"``.

        If *names* is given, only those phases are included (missing ones
        are skipped); otherwise all recorded phases are rendered in
        insertion order.
        """
        if names:
            return " ".join(
                f"{k}={self._phases[k]:.1f}ms"
                for k in names
                if k in self._phases
            )
        return " ".join(
            f"{k}={v:.1f}ms" for k, v in self._phases.items()
        )
