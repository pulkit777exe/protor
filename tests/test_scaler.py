"""Tests for protor.scaler module."""

import time

from protor.scaler import AutoScaler


class TestAutoScaler:
    def test_initial_concurrency(self):
        scaler = AutoScaler(initial=5)
        assert scaler.concurrency == 5

    def test_custom_bounds(self):
        scaler = AutoScaler(initial=4, min_c=2, max_c=10)
        assert scaler.concurrency == 4
        assert scaler._min == 2
        assert scaler._max == 10

    def test_record_adds_result(self):
        scaler = AutoScaler(initial=4, window=5)
        scaler.record(True)
        scaler.record(False)
        assert len(scaler._results) == 2

    def test_record_trims_window(self):
        scaler = AutoScaler(initial=4, window=3)
        for _ in range(5):
            scaler.record(True)
        assert len(scaler._results) == 3

    def test_maybe_scale_cooldown(self):
        scaler = AutoScaler(initial=4, cooldown=10.0)
        scaler._results = [True] * 10
        # Should not scale during cooldown
        result = scaler.maybe_scale()
        assert result == 4

    def test_maybe_scale_insufficient_data(self):
        scaler = AutoScaler(initial=4, window=10, cooldown=0)
        scaler._last_scale = time.monotonic() - 100
        # Not enough results yet
        result = scaler.maybe_scale()
        assert result == 4

    def test_maybe_scale_up(self):
        scaler = AutoScaler(initial=4, window=3, cooldown=0, up_threshold=0.8, max_c=10)
        scaler._last_scale = time.monotonic() - 100
        # 100% success rate should scale up
        for _ in range(3):
            scaler.record(True)
        result = scaler.maybe_scale()
        assert result == 6  # +2

    def test_maybe_scale_down(self):
        scaler = AutoScaler(initial=6, window=3, cooldown=0, down_threshold=0.3, min_c=2)
        scaler._last_scale = time.monotonic() - 100
        # 0% success rate should scale down
        for _ in range(3):
            scaler.record(False)
        result = scaler.maybe_scale()
        assert result == 5  # -1

    def test_maybe_scale_stays_at_max(self):
        scaler = AutoScaler(initial=10, window=3, cooldown=0, max_c=10, up_threshold=0.5)
        scaler._last_scale = time.monotonic() - 100
        for _ in range(3):
            scaler.record(True)
        result = scaler.maybe_scale()
        assert result == 10  # Can't go higher

    def test_maybe_scale_stays_at_min(self):
        scaler = AutoScaler(initial=2, window=3, cooldown=0, min_c=2, down_threshold=0.5)
        scaler._last_scale = time.monotonic() - 100
        for _ in range(3):
            scaler.record(False)
        result = scaler.maybe_scale()
        assert result == 2  # Can't go lower

    def test_maybe_scale_no_change_at_boundary(self):
        scaler = AutoScaler(initial=4, window=3, cooldown=0, up_threshold=0.9, down_threshold=0.1)
        scaler._last_scale = time.monotonic() - 100
        # 50% success - no change
        scaler.record(True)
        scaler.record(False)
        scaler.record(True)
        result = scaler.maybe_scale()
        assert result == 4

    def test_scale_resets_cooldown(self):
        scaler = AutoScaler(initial=4, window=3, cooldown=5.0, up_threshold=0.8)
        scaler._last_scale = time.monotonic() - 100
        for _ in range(3):
            scaler.record(True)
        scaler.maybe_scale()
        # Second call should be in cooldown
        result = scaler.maybe_scale()
        assert result == scaler.concurrency
