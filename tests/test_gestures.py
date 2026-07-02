"""Unit tests for src.gestures: pinch/grab math and hysteresis, no webcam needed."""

import pytest

from src.gestures import (
    GestureDetector,
    GestureState,
    PINCH_ENTER_THRESHOLD,
    PINCH_EXIT_THRESHOLD,
    detect_grab,
    pinch_distance_and_midpoint,
)

WRIST = 0
THUMB_TIP = 4
INDEX_TIP = 8
MIDDLE_TIP = 12
RING_TIP = 16
PINKY_TIP = 20


def _open_hand() -> list:
    """21 landmarks for a fully open hand: fingertips far from the wrist.

    Wrist at the origin; each fingertip is placed well beyond its PIP/MCP
    joint so nothing is curled and thumb/index are far apart (no pinch).
    """
    hand = [(0.0, 0.0, 0.0)] * 21
    hand[WRIST] = (0.5, 0.9, 0.0)
    hand[THUMB_TIP] = (0.3, 0.5, 0.0)
    hand[INDEX_TIP] = (0.7, 0.1, 0.0)

    # Index/middle/ring/pinky: tip far beyond (i.e. farther from wrist than)
    # their PIP joint, i.e. extended, not curled.
    for pip_idx, tip_idx in ((6, 8), (10, 12), (14, 16), (18, 20)):
        hand[pip_idx] = (0.5, 0.4, 0.0)
        hand[tip_idx] = (0.5, 0.1, 0.0)

    return hand


def _pinching_hand() -> list:
    """Open hand but with thumb tip and index tip touching."""
    hand = _open_hand()
    hand[THUMB_TIP] = (0.50, 0.10, 0.0)
    hand[INDEX_TIP] = (0.50, 0.11, 0.0)
    return hand


def _curled_hand() -> list:
    """21 landmarks for a fist: fingertips folded back past their PIP joints."""
    hand = [(0.0, 0.0, 0.0)] * 21
    hand[WRIST] = (0.5, 0.9, 0.0)
    hand[THUMB_TIP] = (0.5, 0.7, 0.0)

    # Tip closer to the wrist than the PIP joint is -- curled.
    for pip_idx, tip_idx in ((6, 8), (10, 12), (14, 16), (18, 20)):
        hand[pip_idx] = (0.5, 0.5, 0.0)   # farther from wrist (0.9)
        hand[tip_idx] = (0.5, 0.7, 0.0)   # closer to wrist

    return hand


def test_pinch_distance_and_midpoint_math():
    hand = [(0.0, 0.0, 0.0)] * 21
    hand[THUMB_TIP] = (0.0, 0.0, 0.0)
    hand[INDEX_TIP] = (0.3, 0.4, 0.0)  # 3-4-5 triangle -> distance 0.5

    distance, midpoint = pinch_distance_and_midpoint(hand)

    assert distance == 0.5
    assert midpoint == (0.15, 0.2)


def test_pinch_distance_ignores_z():
    hand = [(0.0, 0.0, 0.0)] * 21
    hand[THUMB_TIP] = (0.0, 0.0, 0.0)
    hand[INDEX_TIP] = (0.0, 0.0, 100.0)  # huge z difference, x/y identical

    distance, _midpoint = pinch_distance_and_midpoint(hand)

    assert distance == 0.0


def test_detect_grab_true_for_curled_hand():
    assert detect_grab(_curled_hand()) is True


def test_detect_grab_false_for_open_hand():
    assert detect_grab(_open_hand()) is False


def test_detector_no_hand_returns_open_non_pinching_state():
    detector = GestureDetector()
    state = detector.update(None)

    assert isinstance(state, GestureState)
    assert state.is_pinching is False
    assert state.is_grab is False


def test_detector_crossing_enter_threshold_starts_pinch():
    hand = [(0.0, 0.0, 0.0)] * 21
    hand[WRIST] = (0.5, 0.9, 0.0)
    hand[THUMB_TIP] = (0.5, 0.5, 0.0)
    hand[INDEX_TIP] = (0.5, 0.5 + (PINCH_ENTER_THRESHOLD - 0.01), 0.0)

    detector = GestureDetector()
    state = detector.update(hand)

    assert state.is_pinching is True


def test_detector_staying_between_enter_and_exit_keeps_pinching():
    detector = GestureDetector()

    close_hand = [(0.0, 0.0, 0.0)] * 21
    close_hand[WRIST] = (0.5, 0.9, 0.0)
    close_hand[THUMB_TIP] = (0.5, 0.5, 0.0)
    close_hand[INDEX_TIP] = (0.5, 0.5 + (PINCH_ENTER_THRESHOLD - 0.01), 0.0)
    first_state = detector.update(close_hand)
    assert first_state.is_pinching is True

    # Distance now between ENTER and EXIT: wouldn't newly trigger a pinch,
    # but hysteresis should hold the already-pinching state.
    mid_distance = (PINCH_ENTER_THRESHOLD + PINCH_EXIT_THRESHOLD) / 2.0
    drifted_hand = [(0.0, 0.0, 0.0)] * 21
    drifted_hand[WRIST] = (0.5, 0.9, 0.0)
    drifted_hand[THUMB_TIP] = (0.5, 0.5, 0.0)
    drifted_hand[INDEX_TIP] = (0.5, 0.5 + mid_distance, 0.0)
    second_state = detector.update(drifted_hand)

    assert second_state.is_pinching is True


def test_detector_crossing_exit_threshold_stops_pinching():
    detector = GestureDetector()

    close_hand = [(0.0, 0.0, 0.0)] * 21
    close_hand[WRIST] = (0.5, 0.9, 0.0)
    close_hand[THUMB_TIP] = (0.5, 0.5, 0.0)
    close_hand[INDEX_TIP] = (0.5, 0.5 + (PINCH_ENTER_THRESHOLD - 0.01), 0.0)
    first_state = detector.update(close_hand)
    assert first_state.is_pinching is True

    far_hand = [(0.0, 0.0, 0.0)] * 21
    far_hand[WRIST] = (0.5, 0.9, 0.0)
    far_hand[THUMB_TIP] = (0.5, 0.5, 0.0)
    far_hand[INDEX_TIP] = (0.5, 0.5 + (PINCH_EXIT_THRESHOLD + 0.01), 0.0)
    second_state = detector.update(far_hand)

    assert second_state.is_pinching is False


def test_detector_open_hand_never_pinches():
    detector = GestureDetector()
    state = detector.update(_open_hand())
    assert state.is_pinching is False


def test_detector_pinch_point_is_midpoint_of_thumb_and_index():
    hand = [(0.0, 0.0, 0.0)] * 21
    hand[WRIST] = (0.5, 0.9, 0.0)
    hand[THUMB_TIP] = (0.2, 0.2, 0.0)
    hand[INDEX_TIP] = (0.4, 0.2, 0.0)

    detector = GestureDetector()
    state = detector.update(hand)

    assert state.pinch_point == pytest.approx((0.3, 0.2))


def test_detector_resets_pinch_state_after_hand_lost():
    detector = GestureDetector()

    close_hand = [(0.0, 0.0, 0.0)] * 21
    close_hand[WRIST] = (0.5, 0.9, 0.0)
    close_hand[THUMB_TIP] = (0.5, 0.5, 0.0)
    close_hand[INDEX_TIP] = (0.5, 0.5 + (PINCH_ENTER_THRESHOLD - 0.01), 0.0)
    assert detector.update(close_hand).is_pinching is True

    assert detector.update(None).is_pinching is False

    # A distance that would have kept a pre-existing pinch alive (between
    # enter and exit) must NOT resume pinching after a reset -- it has to
    # cross the tighter enter threshold again.
    mid_distance = (PINCH_ENTER_THRESHOLD + PINCH_EXIT_THRESHOLD) / 2.0
    drifted_hand = [(0.0, 0.0, 0.0)] * 21
    drifted_hand[WRIST] = (0.5, 0.9, 0.0)
    drifted_hand[THUMB_TIP] = (0.5, 0.5, 0.0)
    drifted_hand[INDEX_TIP] = (0.5, 0.5 + mid_distance, 0.0)

    assert detector.update(drifted_hand).is_pinching is False
