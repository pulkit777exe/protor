"""Auto-scaling concurrency controller for scraper and crawler."""

from __future__ import annotations

import time

from .config import (
    SCALING_COOLDOWN,
    SCALING_DOWN_THRESHOLD,
    SCALING_MAX_CONCURRENCY,
    SCALING_MIN_CONCURRENCY,
    SCALING_UP_THRESHOLD,
    SCALING_WINDOW,
)


class AutoScaler:
    """Dynamically adjusts concurrency based on recent success rates."""

    def __init__(
        self,
        initial: int,
        min_c: int = SCALING_MIN_CONCURRENCY,
        max_c: int = SCALING_MAX_CONCURRENCY,
        up_threshold: float = SCALING_UP_THRESHOLD,
        down_threshold: float = SCALING_DOWN_THRESHOLD,
        window: int = SCALING_WINDOW,
        cooldown: float = SCALING_COOLDOWN,
    ) -> None:
        self.concurrency = initial
        self._min = min_c
        self._max = max_c
        self._up = up_threshold
        self._down = down_threshold
        self._window = window
        self._cooldown = cooldown
        self._results: list[bool] = []
        self._last_scale = time.monotonic()

    def record(self, success: bool) -> None:
        self._results.append(success)
        if len(self._results) > self._window:
            self._results = self._results[-self._window :]

    def maybe_scale(self) -> int:
        now = time.monotonic()
        if now - self._last_scale < self._cooldown:
            return self.concurrency
        if len(self._results) < self._window:
            return self.concurrency

        rate = sum(self._results) / len(self._results)
        old = self.concurrency

        if rate >= self._up and self.concurrency < self._max:
            self.concurrency = min(self.concurrency + 2, self._max)
        elif rate <= self._down and self.concurrency > self._min:
            self.concurrency = max(self.concurrency - 1, self._min)

        if self.concurrency != old:
            self._last_scale = now

        return self.concurrency
