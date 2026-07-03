"""Unit tests for src.timing: pure pacing + smoothing math, fake clock, no sleeps."""

import pytest

from src.timing import (
    MIN_WAIT_MS,
    StageTimer,
    StageTimings,
    ema,
    frame_budget_seconds,
    pace_wait_ms,
    remaining_sleep_seconds,
)


def _fake_clock(times):
    values = iter(times)
    return lambda: next(values)


# --------------------------------------------------------------------------- #
# StageTimer
# --------------------------------------------------------------------------- #

def test_stage_timer_measures_elapsed_ms_with_fake_clock():
    clock = _fake_clock([100.0, 100.5])  # 0.5s -> 500ms
    with StageTimer(clock) as timer:
        pass
    assert timer.ms == pytest.approx(500.0)


def test_stage_timer_ms_is_zero_before_use():
    assert StageTimer().ms == 0.0


# --------------------------------------------------------------------------- #
# frame_budget_seconds
# --------------------------------------------------------------------------- #

def test_frame_budget_seconds_is_inverse_of_fps():
    assert frame_budget_seconds(30) == pytest.approx(1.0 / 30.0)


def test_frame_budget_seconds_rejects_non_positive():
    with pytest.raises(ValueError):
        frame_budget_seconds(0)


# --------------------------------------------------------------------------- #
# pace_wait_ms
# --------------------------------------------------------------------------- #

def test_pace_wait_ms_returns_remaining_budget():
    # 30fps -> 33.3ms budget; 10ms of work -> ~23ms remaining.
    assert pace_wait_ms(30, 0.010) == 23


def test_pace_wait_ms_full_budget_when_no_work():
    assert pace_wait_ms(30, 0.0) == 33


def test_pace_wait_ms_clamps_to_min_when_over_budget():
    # 50ms of work already blew the 33ms budget -> still wait the minimum.
    assert pace_wait_ms(30, 0.050) == MIN_WAIT_MS


def test_pace_wait_ms_respects_custom_min():
    assert pace_wait_ms(30, 1.0, min_ms=5) == 5


# --------------------------------------------------------------------------- #
# remaining_sleep_seconds
# --------------------------------------------------------------------------- #

def test_remaining_sleep_seconds_is_budget_minus_elapsed():
    # 30fps -> 33.3ms budget; 10ms elapsed -> ~23.3ms of sleep.
    assert remaining_sleep_seconds(30, 0.010) == pytest.approx(1.0 / 30.0 - 0.010)


def test_remaining_sleep_seconds_never_negative_when_over_budget():
    assert remaining_sleep_seconds(30, 0.100) == 0.0


# --------------------------------------------------------------------------- #
# ema
# --------------------------------------------------------------------------- #

def test_ema_moves_toward_sample():
    assert ema(10.0, 20.0, alpha=0.5) == pytest.approx(15.0)


def test_ema_alpha_one_takes_sample():
    assert ema(10.0, 20.0, alpha=1.0) == pytest.approx(20.0)


def test_ema_alpha_zero_keeps_previous():
    assert ema(10.0, 20.0, alpha=0.0) == pytest.approx(10.0)


# --------------------------------------------------------------------------- #
# StageTimings
# --------------------------------------------------------------------------- #

def test_stage_timings_defaults_are_zero():
    t = StageTimings()
    assert (t.capture_ms, t.track_ms, t.composite_ms, t.sink_ms, t.loop_ms) == (
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    )


def test_stage_timings_is_frozen():
    t = StageTimings()
    with pytest.raises(Exception):
        t.loop_ms = 5.0  # frozen dataclass -> FrozenInstanceError
