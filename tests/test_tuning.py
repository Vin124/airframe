"""Unit tests for src.tuning: the pinch clamp and the tuning/confidence snapshots.

Pure math + dataclasses, no threads/UI. Covers ``clamp_tuning`` edge cases
(inverted, equal, out-of-range) that guard the hysteresis invariant exit > enter.
"""

import pytest

from src.tuning import (
    CONFIDENCE_MAX,
    CONFIDENCE_MIN,
    DEFAULT_MEDIAPIPE_CONFIDENCE,
    DEFAULT_PINCH_ENTER,
    DEFAULT_PINCH_EXIT,
    MIN_HYSTERESIS_GAP,
    PINCH_THRESHOLD_MAX,
    PINCH_THRESHOLD_MIN,
    GestureTuning,
    TrackerConfidence,
    clamp_tuning,
)


def test_valid_pair_passes_through_unchanged():
    tuning = clamp_tuning(DEFAULT_PINCH_ENTER, DEFAULT_PINCH_EXIT)
    assert tuning == GestureTuning(DEFAULT_PINCH_ENTER, DEFAULT_PINCH_EXIT)


def test_inverted_pair_lifts_exit_above_enter():
    tuning = clamp_tuning(0.08, 0.06)  # exit below enter
    assert tuning.enter == pytest.approx(0.08)
    assert tuning.exit == pytest.approx(0.08 + MIN_HYSTERESIS_GAP)
    assert tuning.exit > tuning.enter


def test_equal_pair_gets_minimum_gap():
    tuning = clamp_tuning(0.05, 0.05)
    assert tuning.enter == pytest.approx(0.05)
    assert tuning.exit == pytest.approx(0.05 + MIN_HYSTERESIS_GAP)


def test_out_of_range_high_pulled_into_range_keeping_gap():
    tuning = clamp_tuning(0.2, 0.3)  # both above the max
    assert tuning.exit == pytest.approx(PINCH_THRESHOLD_MAX)
    assert tuning.enter == pytest.approx(PINCH_THRESHOLD_MAX - MIN_HYSTERESIS_GAP)
    assert PINCH_THRESHOLD_MIN <= tuning.enter <= PINCH_THRESHOLD_MAX


def test_out_of_range_low_clamped_up_to_minimum():
    tuning = clamp_tuning(0.001, 0.001)  # both below the min
    assert tuning.enter == pytest.approx(PINCH_THRESHOLD_MIN)
    assert tuning.exit == pytest.approx(PINCH_THRESHOLD_MIN + MIN_HYSTERESIS_GAP)


def test_enter_at_max_pulls_pair_down_to_preserve_gap():
    tuning = clamp_tuning(PINCH_THRESHOLD_MAX, PINCH_THRESHOLD_MAX)
    assert tuning.exit == pytest.approx(PINCH_THRESHOLD_MAX)
    assert tuning.enter == pytest.approx(PINCH_THRESHOLD_MAX - MIN_HYSTERESIS_GAP)
    assert tuning.exit > tuning.enter


def test_clamp_always_keeps_both_in_range_and_ordered():
    for enter, exit in [(-1.0, 5.0), (0.0, 0.0), (0.15, 0.02), (0.03, 0.9)]:
        tuning = clamp_tuning(enter, exit)
        assert PINCH_THRESHOLD_MIN <= tuning.enter <= PINCH_THRESHOLD_MAX
        assert PINCH_THRESHOLD_MIN <= tuning.exit <= PINCH_THRESHOLD_MAX
        assert tuning.exit >= tuning.enter + MIN_HYSTERESIS_GAP - 1e-9


def test_clamp_coerces_int_like_input_to_float():
    tuning = clamp_tuning(0, 1)  # ints
    assert isinstance(tuning.enter, float)
    assert isinstance(tuning.exit, float)


def test_gesture_tuning_is_frozen():
    tuning = GestureTuning(0.05, 0.07)
    with pytest.raises(Exception):
        tuning.enter = 0.1  # frozen dataclass


def test_tracker_confidence_defaults_to_mediapipe_default():
    conf = TrackerConfidence()
    assert conf.detection == DEFAULT_MEDIAPIPE_CONFIDENCE
    assert conf.presence == DEFAULT_MEDIAPIPE_CONFIDENCE
    assert conf.tracking == DEFAULT_MEDIAPIPE_CONFIDENCE
    assert CONFIDENCE_MIN <= conf.detection <= CONFIDENCE_MAX
