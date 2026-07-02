"""Unit tests for src.overlay: transform math, hit-test, compositing, and the
grab/resize interaction state machines. All pure/headless -- no webcam, no
window, synthetic numpy frames only.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from src.overlay import (
    CENTER_MAX,
    CENTER_MIN,
    MAX_SCALE,
    MIN_SCALE,
    GrabState,
    InteractionState,
    Overlay,
    OverlayError,
    ResizeState,
    Transform,
    scaled_size,
    step,
    top_left_px,
    update_grab,
    update_resize,
)

FRAME_SIZE = (100, 100)  # (width, height)


def _gesture(is_pinching, pinch_point=(0.0, 0.0)):
    """A minimal stand-in for GestureState (only the fields step/grab read)."""
    return SimpleNamespace(is_pinching=is_pinching, pinch_point=pinch_point)


def _solid_overlay(size=(10, 10), bgr=(200, 160, 80), alpha=255):
    """A ``size`` (w,h) BGRA overlay filled with one color/alpha."""
    w, h = size
    img = np.zeros((h, w, 4), dtype=np.uint8)
    img[:, :, 0] = bgr[0]
    img[:, :, 1] = bgr[1]
    img[:, :, 2] = bgr[2]
    img[:, :, 3] = alpha
    return Overlay(img)


def _blank_frame(color=(0, 0, 0)):
    frame = np.zeros((FRAME_SIZE[1], FRAME_SIZE[0], 3), dtype=np.uint8)
    frame[:, :] = color
    return frame


# --------------------------------------------------------------------------- #
# Transform math + clamping
# --------------------------------------------------------------------------- #

def test_default_transform_is_centered_unit_scale():
    t = Transform()
    assert t.center_x == 0.5
    assert t.center_y == 0.5
    assert t.scale == 1.0


def test_moved_to_clamps_center_into_frame():
    t = Transform().moved_to(2.0, -1.0)
    assert t.center_x == CENTER_MAX
    assert t.center_y == CENTER_MIN


def test_moved_to_preserves_scale():
    t = Transform(scale=2.0).moved_to(0.3, 0.7)
    assert t.scale == 2.0
    assert (t.center_x, t.center_y) == (0.3, 0.7)


def test_scaled_by_clamps_to_max():
    t = Transform(scale=1.0).scaled_by(1000.0)
    assert t.scale == MAX_SCALE


def test_scaled_by_clamps_to_min():
    t = Transform(scale=1.0).scaled_by(0.00001)
    assert t.scale == MIN_SCALE


def test_scaled_by_preserves_center():
    t = Transform(center_x=0.2, center_y=0.8, scale=1.0).scaled_by(2.0)
    assert (t.center_x, t.center_y) == (0.2, 0.8)
    assert t.scale == 2.0


def test_with_scale_is_absolute_and_clamped():
    assert Transform(scale=1.0).with_scale(3.0).scale == 3.0
    assert Transform(scale=1.0).with_scale(-5.0).scale == MIN_SCALE


def test_scaled_size_multiplies_and_floors_to_one():
    assert scaled_size((10, 20), 2.0) == (20, 40)
    # A tiny scale must never produce a zero dimension.
    assert scaled_size((10, 10), 0.0001) == (1, 1)


def test_top_left_px_centers_the_image():
    # 10x10 centered on a 100x100 frame -> top-left at (45, 45).
    assert top_left_px(Transform(), (10, 10), FRAME_SIZE) == (45, 45)


# --------------------------------------------------------------------------- #
# contains() hit-test
# --------------------------------------------------------------------------- #

def test_contains_center_point_is_inside():
    overlay = _solid_overlay()
    assert overlay.contains(Transform(), (0.5, 0.5), FRAME_SIZE) is True


def test_contains_top_left_edge_is_inclusive():
    overlay = _solid_overlay()  # rect px [45,55) x [45,55)
    assert overlay.contains(Transform(), (0.45, 0.45), FRAME_SIZE) is True


def test_contains_bottom_right_edge_is_exclusive():
    overlay = _solid_overlay()
    assert overlay.contains(Transform(), (0.55, 0.55), FRAME_SIZE) is False


def test_contains_point_just_outside_left_is_false():
    overlay = _solid_overlay()
    assert overlay.contains(Transform(), (0.44, 0.5), FRAME_SIZE) is False


def test_contains_far_point_is_false():
    overlay = _solid_overlay()
    assert overlay.contains(Transform(), (0.9, 0.9), FRAME_SIZE) is False


# --------------------------------------------------------------------------- #
# Loading / alpha handling
# --------------------------------------------------------------------------- #

def test_load_missing_file_raises_overlay_error():
    with pytest.raises(OverlayError):
        Overlay.load("this_file_does_not_exist_12345.png")


def test_constructor_adds_opaque_alpha_when_missing():
    overlay = Overlay(np.zeros((5, 7, 3), dtype=np.uint8))
    assert overlay.image.shape == (5, 7, 4)
    assert np.all(overlay.image[:, :, 3] == 255)


def test_load_real_asset_has_alpha():
    overlay = Overlay.load("assets/test.png")
    assert overlay.image.shape[2] == 4
    assert overlay.base_size == (300, 300)


# --------------------------------------------------------------------------- #
# Compositing
# --------------------------------------------------------------------------- #

def test_composite_opaque_inside_paints_and_leaves_outside_untouched():
    overlay = _solid_overlay(bgr=(10, 20, 30), alpha=255)
    frame = _blank_frame((0, 0, 0))
    overlay.composite(frame, Transform())  # rect px [45,55)

    assert tuple(int(v) for v in frame[50, 50]) == (10, 20, 30)  # inside
    assert tuple(int(v) for v in frame[0, 0]) == (0, 0, 0)       # outside untouched


def test_composite_alpha_blends_known_pixel():
    alpha = 128
    overlay = _solid_overlay(bgr=(200, 160, 80), alpha=alpha)
    frame = _blank_frame((0, 0, 0))
    overlay.composite(frame, Transform())

    ratio = alpha / 255.0
    expected = tuple(int(ratio * c) for c in (200, 160, 80))
    got = tuple(int(v) for v in frame[50, 50])
    assert all(abs(g - e) <= 1 for g, e in zip(got, expected))


@pytest.mark.parametrize(
    "center, probe",
    [
        ((0.0, 0.5), (50, 2)),    # clipped at left edge, right half visible
        ((1.0, 0.5), (50, 97)),   # clipped at right edge
        ((0.5, 0.0), (2, 50)),    # clipped at top edge
        ((0.5, 1.0), (97, 50)),   # clipped at bottom edge
    ],
)
def test_composite_partially_offscreen_clips_without_crashing(center, probe):
    overlay = _solid_overlay(bgr=(11, 22, 33), alpha=255)
    frame = _blank_frame((0, 0, 0))
    overlay.composite(frame, Transform(center_x=center[0], center_y=center[1]))
    row, col = probe
    assert tuple(int(v) for v in frame[row, col]) == (11, 22, 33)


def test_composite_fully_offscreen_is_a_noop():
    # Tiny frame + a 1px scaled image parked past the last column -> no overlap.
    overlay = _solid_overlay(size=(10, 10), bgr=(255, 255, 255), alpha=255)
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    original = frame.copy()
    overlay.composite(frame, Transform(center_x=1.0, center_y=1.0, scale=MIN_SCALE))
    assert np.array_equal(frame, original)


# --------------------------------------------------------------------------- #
# Grab state machine (via step, single hand; secondary never pinches)
# --------------------------------------------------------------------------- #

def _idle():
    return InteractionState.initial()


def test_pinch_start_on_image_grabs():
    overlay = _solid_overlay()
    state = step(_idle(), overlay, [_gesture(True, (0.5, 0.5)), _gesture(False)], FRAME_SIZE)
    assert state.grab.is_grabbed is True


def test_pinch_start_off_image_does_not_grab_even_when_dragged_over():
    overlay = _solid_overlay()
    # Pinch starts off the image.
    s1 = step(_idle(), overlay, [_gesture(True, (0.9, 0.9)), _gesture(False)], FRAME_SIZE)
    assert s1.grab.is_grabbed is False
    # Still pinching, now dragged over the image: no rising edge -> no grab.
    s2 = step(s1, overlay, [_gesture(True, (0.5, 0.5)), _gesture(False)], FRAME_SIZE)
    assert s2.grab.is_grabbed is False


def test_move_follows_pinch_keeping_grab_offset():
    overlay = _solid_overlay()
    # Grab at an off-center point inside the image (rect normalized [0.45,0.55)).
    s1 = step(_idle(), overlay, [_gesture(True, (0.47, 0.52)), _gesture(False)], FRAME_SIZE)
    assert s1.grab.is_grabbed is True
    # Offset = center(0.5,0.5) - grab point(0.47,0.52) = (0.03, -0.02).
    s2 = step(s1, overlay, [_gesture(True, (0.60, 0.60)), _gesture(False)], FRAME_SIZE)
    assert s2.transform.center_x == pytest.approx(0.63)
    assert s2.transform.center_y == pytest.approx(0.58)


def test_unpinch_releases():
    overlay = _solid_overlay()
    s1 = step(_idle(), overlay, [_gesture(True, (0.5, 0.5)), _gesture(False)], FRAME_SIZE)
    s2 = step(s1, overlay, [_gesture(False, (0.5, 0.5)), _gesture(False)], FRAME_SIZE)
    assert s2.grab.is_grabbed is False


def test_hand_loss_releases_without_moving_to_origin():
    overlay = _solid_overlay()
    s1 = step(_idle(), overlay, [_gesture(True, (0.5, 0.5)), _gesture(False)], FRAME_SIZE)
    # Hand loss is reported by the detector as not-pinching with a (0,0) point.
    s2 = step(s1, overlay, [_gesture(False, (0.0, 0.0)), _gesture(False)], FRAME_SIZE)
    assert s2.grab.is_grabbed is False
    # Must NOT have snapped the image center to the origin.
    assert (s2.transform.center_x, s2.transform.center_y) == (0.5, 0.5)


def test_update_grab_pinch_start_off_image_consumes_edge():
    overlay = _solid_overlay()
    grab, transform = update_grab(
        GrabState(), Transform(), overlay, True, (0.9, 0.9), FRAME_SIZE
    )
    assert grab.is_grabbed is False
    assert grab.was_pinching is True  # edge consumed -> no later grab-on-drag-over


# --------------------------------------------------------------------------- #
# Two-hand resize
# --------------------------------------------------------------------------- #

def test_resize_scales_by_distance_ratio():
    overlay = _solid_overlay()
    # First two-hand frame captures the baseline (distance 0.2, scale 1.0).
    s1 = step(_idle(), overlay, [_gesture(True, (0.4, 0.5)), _gesture(True, (0.6, 0.5))], FRAME_SIZE)
    assert s1.resize.is_resizing is True
    assert s1.transform.scale == pytest.approx(1.0)
    # Hands spread to distance 0.4 -> factor 2.0.
    s2 = step(s1, overlay, [_gesture(True, (0.3, 0.5)), _gesture(True, (0.7, 0.5))], FRAME_SIZE)
    assert s2.transform.scale == pytest.approx(2.0)
    # Hands close to distance 0.1 -> factor 0.5 (relative to the baseline 0.2).
    s3 = step(s2, overlay, [_gesture(True, (0.45, 0.5)), _gesture(True, (0.55, 0.5))], FRAME_SIZE)
    assert s3.transform.scale == pytest.approx(0.5)


def test_resize_releases_and_recaptures_fresh_baseline():
    overlay = _solid_overlay()
    s1 = step(_idle(), overlay, [_gesture(True, (0.4, 0.5)), _gesture(True, (0.6, 0.5))], FRAME_SIZE)
    s2 = step(s1, overlay, [_gesture(True, (0.3, 0.5)), _gesture(True, (0.7, 0.5))], FRAME_SIZE)
    assert s2.transform.scale == pytest.approx(2.0)
    # One hand releases -> resize ends, scale holds.
    s3 = step(s2, overlay, [_gesture(True, (0.3, 0.5)), _gesture(False)], FRAME_SIZE)
    assert s3.resize.is_resizing is False
    assert s3.transform.scale == pytest.approx(2.0)
    # Re-pinch captures a NEW baseline (distance 0.2, scale 2.0)...
    s4 = step(s3, overlay, [_gesture(True, (0.4, 0.5)), _gesture(True, (0.6, 0.5))], FRAME_SIZE)
    assert s4.transform.scale == pytest.approx(2.0)
    # ...then doubling the distance to 0.4 doubles from that new baseline -> 4.0.
    s5 = step(s4, overlay, [_gesture(True, (0.3, 0.5)), _gesture(True, (0.7, 0.5))], FRAME_SIZE)
    assert s5.transform.scale == pytest.approx(4.0)


def test_resize_guards_against_degenerate_zero_distance():
    overlay = _solid_overlay()
    # Both pinch at the same point: baseline distance ~0.
    s1 = step(_idle(), overlay, [_gesture(True, (0.5, 0.5)), _gesture(True, (0.5, 0.5))], FRAME_SIZE)
    assert s1.resize.is_resizing is True
    # Next frame with a real distance must not divide by zero / must not crash.
    s2 = step(s1, overlay, [_gesture(True, (0.4, 0.5)), _gesture(True, (0.6, 0.5))], FRAME_SIZE)
    assert s2.transform.scale == pytest.approx(1.0)  # unchanged, guard held


def test_resize_takes_priority_over_grab():
    overlay = _solid_overlay()
    # Grab first with the primary hand.
    s1 = step(_idle(), overlay, [_gesture(True, (0.5, 0.5)), _gesture(False)], FRAME_SIZE)
    assert s1.grab.is_grabbed is True
    # Second hand starts pinching too -> resize mode, grab idled.
    s2 = step(s1, overlay, [_gesture(True, (0.4, 0.5)), _gesture(True, (0.6, 0.5))], FRAME_SIZE)
    assert s2.resize.is_resizing is True
    assert s2.grab.is_grabbed is False


def test_update_resize_not_both_pinching_resets():
    resize, transform = update_resize(
        ResizeState(is_resizing=True, start_distance=0.2, start_scale=1.0),
        Transform(scale=2.0),
        False,
        (0.0, 0.0),
        (0.0, 0.0),
    )
    assert resize.is_resizing is False
    assert transform.scale == 2.0  # unchanged when not resizing
