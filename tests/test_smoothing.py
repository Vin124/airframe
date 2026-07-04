"""Unit tests for src.smoothing: the One-Euro landmark smoother. Pure math."""

from src.smoothing import LandmarkSmoother, OneEuroFilter


def _hand(x, y, z=0.0, n=21):
    return [(x, y, z)] * n


# --------------------------------------------------------------------------- #
# OneEuroFilter (scalar)
# --------------------------------------------------------------------------- #

def test_first_sample_passes_through_unchanged():
    f = OneEuroFilter()
    assert f.filter(0.5, 0.0) == 0.5


def test_constant_signal_stays_constant():
    f = OneEuroFilter()
    f.filter(0.5, 0.0)
    for i in range(1, 10):
        assert abs(f.filter(0.5, i / 30.0) - 0.5) < 1e-9


def test_step_input_is_smoothed_toward_target_not_instant():
    f = OneEuroFilter()
    f.filter(0.0, 0.0)
    out = f.filter(1.0, 1 / 30.0)  # a sudden jump
    assert 0.0 < out < 1.0  # eased toward the target, not snapped


def test_repeated_timestamp_does_not_divide_by_zero():
    f = OneEuroFilter()
    f.filter(0.0, 0.0)
    # dt == 0 must be handled (no ZeroDivisionError), returns a finite value.
    val = f.filter(0.5, 0.0)
    assert val == val  # not NaN


def test_reset_forgets_history():
    f = OneEuroFilter()
    f.filter(0.0, 0.0)
    f.filter(0.5, 0.1)
    f.reset()
    assert f.filter(0.9, 0.2) == 0.9  # first sample after reset passes through


# --------------------------------------------------------------------------- #
# LandmarkSmoother (whole hands)
# --------------------------------------------------------------------------- #

def test_smooth_preserves_shape():
    s = LandmarkSmoother(num_hands=2)
    out = s.smooth([_hand(0.5, 0.5)], t=0.0)
    assert len(out) == 1
    assert len(out[0]) == 21
    assert len(out[0][0]) == 3


def test_smooth_first_frame_is_identity():
    s = LandmarkSmoother(num_hands=1)
    hand = _hand(0.3, 0.7)
    out = s.smooth([hand], t=0.0)
    assert out[0][0] == (0.3, 0.7, 0.0)


def test_smooth_reduces_jitter_on_a_still_hand():
    s = LandmarkSmoother(num_hands=1)
    s.smooth([_hand(0.5, 0.5)], t=0.0)
    # Feed a jittery-but-centered signal; the smoothed output should sit closer
    # to the mean (0.5) than the raw jitter extremes.
    s.smooth([_hand(0.51, 0.49)], t=1 / 30.0)
    out = s.smooth([_hand(0.49, 0.51)], t=2 / 30.0)
    assert abs(out[0][0][0] - 0.5) < 0.02


def test_missing_hand_resets_that_slot():
    s = LandmarkSmoother(num_hands=1)
    s.smooth([_hand(0.2, 0.2)], t=0.0)
    s.smooth([], t=0.1)  # hand gone -> slot reset
    # A reappearing hand starts fresh (first sample passes through, no smear).
    out = s.smooth([_hand(0.8, 0.8)], t=0.2)
    assert out[0][0] == (0.8, 0.8, 0.0)


def test_two_hand_slots_are_independent():
    s = LandmarkSmoother(num_hands=2)
    out = s.smooth([_hand(0.2, 0.2), _hand(0.8, 0.8)], t=0.0)
    assert out[0][0] == (0.2, 0.2, 0.0)
    assert out[1][0] == (0.8, 0.8, 0.0)
