"""Unit tests for morphohand.rl.watchdog (trainer-side collapse guard)."""
from __future__ import annotations

import torch
import pytest

from morphohand.rl.watchdog import (
    TrainingCollapseError,
    attach_collapse_watchdog,
    mean_ep_metric,
)

KEY = "Metrics/lift_height/object_height"


class FakeLogger:
    def __init__(self):
        self.ep_extras = []
        self.log_calls = []

    def log(self, **kwargs):
        self.log_calls.append(kwargs)
        self.ep_extras.clear()  # rsl_rl's Logger.log clears the buffer


class FakeRunner:
    def __init__(self):
        self.logger = FakeLogger()


def test_mean_ep_metric_matches_logger_aggregation():
    extras = [
        {KEY: torch.tensor([0.10, 0.20])},
        {"other": torch.tensor([9.9])},          # missing key -> skipped
        {KEY: torch.tensor(0.30)},               # 0-dim tensor
        {KEY: 0.40},                             # raw float
    ]
    assert mean_ep_metric(extras, KEY) == pytest.approx((0.1 + 0.2 + 0.3 + 0.4) / 4)
    assert mean_ep_metric([], KEY) is None
    assert mean_ep_metric([{"other": 1.0}], KEY) is None


def _step(runner, it, value):
    runner.logger.ep_extras.append({KEY: torch.tensor([value])})
    runner.logger.log(it=it)


def test_no_fire_before_guard_iter_or_above_threshold(tmp_path):
    runner = FakeRunner()
    sentinel = tmp_path / "t.log.COLLAPSED"
    attach_collapse_watchdog(runner, collapse_z=0.045, guard_from_iter=40, sentinel=sentinel)
    _step(runner, it=10, value=0.001)   # collapsed value but before guard iter
    _step(runner, it=50, value=0.090)   # healthy hold
    assert len(runner.logger.log_calls) == 2
    assert not sentinel.exists()


def test_fires_and_writes_sentinel_after_guard(tmp_path):
    runner = FakeRunner()
    sentinel = tmp_path / "t.log.COLLAPSED"
    attach_collapse_watchdog(runner, collapse_z=0.045, guard_from_iter=40, sentinel=sentinel)
    with pytest.raises(TrainingCollapseError, match="at iter 41"):
        _step(runner, it=41, value=0.010)
    assert sentinel.exists() and "0.0100" in sentinel.read_text()
    # the underlying log() still ran (metrics for the fatal iter are recorded)
    assert len(runner.logger.log_calls) == 1


def test_missing_metric_never_fires(tmp_path):
    runner = FakeRunner()
    attach_collapse_watchdog(runner, collapse_z=0.045, guard_from_iter=0, sentinel=None)
    runner.logger.log(it=100)  # no ep_extras at all -> no value -> no fire
    assert len(runner.logger.log_calls) == 1
