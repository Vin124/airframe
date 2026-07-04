"""Unit tests for src.gestures: pinch/grab math and hysteresis, no webcam needed."""

import pytest

from src.gestures import (
    GestureDetector,
    GestureState,
    PINCH_ENTER_THRESHOLD,
    PINCH_EXIT_THRESHOLD,
    detect_grab,
    pinch_ratio_and_midpoint,
)


WRIST = 0
THUMB_TIP = 4
INDEX_TIP = 8
MIDDLE_MCP = 9
MIDDLE_TIP = 12
RING_TIP = 16
PINKY_TIP = 20

# PIP joints (per src.gestures topology).
INDEX_PIP = 6
MIDDLE_PIP = 10
RING_PIP = 14
PINKY_PIP = 18



def _open_hand() -> list:
    """21 landmarks for a fully open hand: fingertips far from the wrist.

    Wrist at the bottom; each fingertip is placed well beyond its PIP/MCP
    joint so nothing is curled and thumb/index are far apart (no pinch). The
    middle-finger MCP sits exactly 1.0 from the wrist, so the pinch RATIO
    (thumb-index distance / palm length) equals the raw thumb-index distance
    and test geometry stays easy to reason about.
    """
    hand = [(0.0, 0.0, 0.0)] * 21
    hand[WRIST] = (0.5, 0.9, 0.0)
    hand[MIDDLE_MCP] = (0.5, -0.1, 0.0)  # palm length exactly 1.0
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


def test_pinch_ratio_and_midpoint_math():
    hand = [(0.0, 0.0, 0.0)] * 21
    hand[WRIST] = (0.0, 0.0, 0.0)
    hand[MIDDLE_MCP] = (0.0, 1.0, 0.0)  # palm length 1.0 -> ratio == distance
    hand[THUMB_TIP] = (0.0, 0.0, 0.0)
    hand[INDEX_TIP] = (0.3, 0.4, 0.0)  # 3-4-5 triangle -> distance 0.5

    ratio, midpoint = pinch_ratio_and_midpoint(hand)

    assert ratio == 0.5
    assert midpoint == (0.15, 0.2)


def test_pinch_ratio_normalized_by_palm_length():
    # Same finger gap over half the palm length -> double the ratio.
    hand = [(0.0, 0.0, 0.0)] * 21
    hand[WRIST] = (0.0, 0.0, 0.0)
    hand[MIDDLE_MCP] = (0.0, 0.5, 0.0)  # palm length 0.5
    hand[THUMB_TIP] = (0.0, 0.0, 0.0)
    hand[INDEX_TIP] = (0.3, 0.4, 0.0)  # distance 0.5

    ratio, _midpoint = pinch_ratio_and_midpoint(hand)

    assert ratio == pytest.approx(1.0)


def test_pinch_ratio_is_scale_invariant():
    # Shrinking the whole hand (a hand farther from the camera) must not
    # change the ratio -- this is the property the palm normalization buys.
    hand = _open_hand()
    shrunk = [(x * 0.25, y * 0.25, z) for (x, y, z) in hand]

    ratio, _ = pinch_ratio_and_midpoint(hand)
    shrunk_ratio, _ = pinch_ratio_and_midpoint(shrunk)

    assert shrunk_ratio == pytest.approx(ratio)


def test_pinch_ratio_ignores_z():
    hand = [(0.0, 0.0, 0.0)] * 21
    hand[WRIST] = (0.0, 0.0, 0.0)
    hand[MIDDLE_MCP] = (0.0, 1.0, 0.0)
    hand[THUMB_TIP] = (0.0, 0.0, 0.0)
    hand[INDEX_TIP] = (0.0, 0.0, 100.0)  # huge z difference, x/y identical

    ratio, _midpoint = pinch_ratio_and_midpoint(hand)

    assert ratio == 0.0


def test_pinch_ratio_degenerate_palm_does_not_explode():
    # All landmarks collapsed (tracking glitch): the epsilon floor keeps the
    # ratio finite instead of dividing by zero.
    hand = [(0.5, 0.5, 0.0)] * 21

    ratio, _midpoint = pinch_ratio_and_midpoint(hand)

    assert ratio == 0.0


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
    detector = GestureDetector()
    state = detector.update(_hand_with_gap(PINCH_ENTER_THRESHOLD - 0.01))

    assert state.is_pinching is True


def test_detector_staying_between_enter_and_exit_keeps_pinching():
    detector = GestureDetector()

    first_state = detector.update(_hand_with_gap(PINCH_ENTER_THRESHOLD - 0.01))
    assert first_state.is_pinching is True

    # Ratio now between ENTER and EXIT: wouldn't newly trigger a pinch,
    # but hysteresis should hold the already-pinching state.
    mid_ratio = (PINCH_ENTER_THRESHOLD + PINCH_EXIT_THRESHOLD) / 2.0
    second_state = detector.update(_hand_with_gap(mid_ratio))

    assert second_state.is_pinching is True


def test_detector_crossing_exit_threshold_stops_pinching():
    detector = GestureDetector()

    first_state = detector.update(_hand_with_gap(PINCH_ENTER_THRESHOLD - 0.01))
    assert first_state.is_pinching is True

    second_state = detector.update(_hand_with_gap(PINCH_EXIT_THRESHOLD + 0.01))

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


def _hand_with_gap(gap: float) -> list:
    """A hand whose pinch RATIO equals ``gap``: unit palm, thumb-index = gap."""
    hand = [(0.0, 0.0, 0.0)] * 21
    hand[WRIST] = (0.5, 0.9, 0.0)
    hand[MIDDLE_MCP] = (0.5, -0.1, 0.0)  # palm length exactly 1.0
    hand[THUMB_TIP] = (0.5, 0.5, 0.0)
    hand[INDEX_TIP] = (0.5, 0.5 + gap, 0.0)
    return hand


def test_detector_honors_injected_enter_threshold():
    # Custom tight thresholds: a ratio of 0.02 is under a 0.03 enter -> pinch.
    detector = GestureDetector(enter_threshold=0.03, exit_threshold=0.05)
    assert detector.update(_hand_with_gap(0.02)).is_pinching is True  # < enter
    # A fresh detector: a 0.04 ratio is above the 0.03 enter -> no pinch, even
    # though it's well below the module default enter -- proves the injected
    # threshold, not the module constant, is used on the ENTER path.
    fresh = GestureDetector(enter_threshold=0.03, exit_threshold=0.05)
    assert fresh.update(_hand_with_gap(0.04)).is_pinching is False


def test_detector_honors_injected_exit_threshold_preserving_hysteresis():
    detector = GestureDetector(enter_threshold=0.03, exit_threshold=0.05)
    # Enter the pinch (gap under the injected enter).
    assert detector.update(_hand_with_gap(0.02)).is_pinching is True
    # Drift to 0.04: between injected enter (0.03) and exit (0.05) -> hysteresis
    # holds the pinch.
    assert detector.update(_hand_with_gap(0.04)).is_pinching is True
    # Cross the injected exit (0.05) -> release.
    assert detector.update(_hand_with_gap(0.06)).is_pinching is False


def test_detector_set_thresholds_swaps_bounds_live():
    detector = GestureDetector(enter_threshold=0.03, exit_threshold=0.05)
    # A 0.04 gap does not enter under the tight 0.03 enter.
    assert detector.update(_hand_with_gap(0.04)).is_pinching is False
    # Loosen the enter threshold; now the same gap enters.
    detector.set_thresholds(0.06, 0.08)
    assert detector.update(_hand_with_gap(0.04)).is_pinching is True


def test_detector_defaults_match_module_thresholds():
    detector = GestureDetector()
    assert detector._enter_threshold == PINCH_ENTER_THRESHOLD
    assert detector._exit_threshold == PINCH_EXIT_THRESHOLD


def test_detector_resets_pinch_state_after_hand_lost():
    detector = GestureDetector()

    assert detector.update(_hand_with_gap(PINCH_ENTER_THRESHOLD - 0.01)).is_pinching is True

    assert detector.update(None).is_pinching is False

    # A ratio that would have kept a pre-existing pinch alive (between
    # enter and exit) must NOT resume pinching after a reset -- it has to
    # cross the tighter enter threshold again.
    mid_ratio = (PINCH_ENTER_THRESHOLD + PINCH_EXIT_THRESHOLD) / 2.0

    assert detector.update(_hand_with_gap(mid_ratio)).is_pinching is False
