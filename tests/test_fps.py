"""Unit tests for src.fps.FpsCounter: pure logic driven by a fake clock, no real time.sleep."""

import pytest

from src.fps import FpsCounter


def _fake_clock(times):
    values = iter(times)
    return lambda: next(values)


def test_tick_returns_none_before_interval_elapses():
    # init consumes t=0.0; ticks at 0.1s and 0.2s are both under the 1.0s interval
    clock = _fake_clock([0.0, 0.1, 0.2])
    counter = FpsCounter(report_interval_seconds=1.0, clock=clock)

    assert counter.tick() is None
    assert counter.tick() is None


def test_tick_reports_fps_once_interval_elapses():
    # init consumes t=0.0; two frames land at 0.5s and 1.0s -> 2 frames / 1.0s = 2.0 fps
    clock = _fake_clock([0.0, 0.5, 1.0])
    counter = FpsCounter(report_interval_seconds=1.0, clock=clock)

    assert counter.tick() is None
    fps = counter.tick()

    assert fps == pytest.approx(2.0)


def test_window_resets_after_reporting():
    clock = _fake_clock([0.0, 1.0, 2.0])
    counter = FpsCounter(report_interval_seconds=1.0, clock=clock)

    first = counter.tick()  # elapsed 1.0s, 1 frame -> 1.0 fps; window resets to t=1.0
    second = counter.tick()  # elapsed 1.0s (2.0 - 1.0), 1 frame -> 1.0 fps

    assert first == pytest.approx(1.0)
    assert second == pytest.approx(1.0)


def test_invalid_interval_raises():
    with pytest.raises(ValueError):
        FpsCounter(report_interval_seconds=0)
