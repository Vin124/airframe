"""Unit tests for src.drawing: the pure air-drawing model, smoothing, and
rasterization. Headless -- numpy/cv2 only, no camera/MediaPipe/display."""

import numpy as np
import pytest

from src.drawing import (
    DEFAULT_STROKE_WIDTH,
    MAX_STROKE_WIDTH,
    MIN_STROKE_WIDTH,
    STROKE_PALETTE,
    DrawState,
    bbox_to_transform,
    clamped_width,
    draw_cursor,
    draw_live_state,
    draw_stroke_overlay,
    freeze_draw_state,
    next_color_index,
    rasterize_stroke,
    rasterize_strokes,
    smooth_points,
)
from src.scene import DEFAULT_SCALE, top_left_px


# --------------------------------------------------------------------------- #
# DrawState model (immutable + functional pen transitions)
# --------------------------------------------------------------------------- #

def test_draw_state_defaults_empty_and_off():
    state = DrawState()
    assert state.mode is False
    assert state.is_empty is True
    assert state.all_strokes == ()


def test_pen_move_appends_without_mutating():
    original = DrawState()
    moved = original.pen_move((0.1, 0.2))
    assert original.current == ()  # unchanged
    assert moved.current == ((0.1, 0.2),)


def test_pen_move_coerces_to_float():
    state = DrawState().pen_move((1, 0))
    assert state.current == ((1.0, 0.0),)
    assert isinstance(state.current[0][0], float)


def test_pen_up_banks_a_real_stroke():
    state = DrawState().pen_move((0.1, 0.1)).pen_move((0.2, 0.2)).pen_up()
    assert state.segments == (((0.1, 0.1), (0.2, 0.2)),)
    assert state.current == ()  # ready for the next stroke


def test_pen_up_discards_a_tap():
    # A single-point "tap" (< MIN_STROKE_POINTS) is not banked.
    state = DrawState().pen_move((0.1, 0.1)).pen_up()
    assert state.segments == ()
    assert state.current == ()


def test_all_strokes_includes_current():
    state = DrawState().pen_move((0.1, 0.1)).pen_move((0.2, 0.2)).pen_up().pen_move((0.5, 0.5))
    assert len(state.all_strokes) == 2  # one banked segment + the in-progress one


def test_cleared_drops_everything():
    state = DrawState().pen_move((0.1, 0.1)).pen_move((0.2, 0.2)).pen_up().pen_move((0.5, 0.5))
    cleared = state.cleared()
    assert cleared.is_empty is True


def test_with_mode_toggles_without_mutating():
    original = DrawState()
    assert original.with_mode(True).mode is True
    assert original.mode is False


def test_color_and_width_transitions():
    state = DrawState()
    assert state.with_next_color().color_index == 1
    assert state.with_width_delta(-1000).width == MIN_STROKE_WIDTH
    assert state.with_width_delta(1000).width == MAX_STROKE_WIDTH
    assert state.color == STROKE_PALETTE[0]


# --------------------------------------------------------------------------- #
# smooth_points (centered moving average)
# --------------------------------------------------------------------------- #

def test_smooth_points_known_values_window_three():
    points = ((0.0, 0.0), (0.2, 0.0), (0.4, 0.0), (0.6, 0.0))
    out = smooth_points(points, window=3)
    # Endpoints pinned; interior points are the centered radius-1 mean.
    assert out[0] == pytest.approx((0.0, 0.0))
    assert out[1] == pytest.approx((0.2, 0.0))
    assert out[2] == pytest.approx((0.4, 0.0))
    assert out[3] == pytest.approx((0.6, 0.0))


def test_smooth_points_preserves_length():
    points = tuple((i / 10.0, 0.0) for i in range(7))
    assert len(smooth_points(points, window=5)) == len(points)


def test_smooth_points_pins_endpoints():
    points = ((0.0, 0.0), (0.5, 0.4), (0.3, 0.9), (0.8, 0.7), (1.0, 1.0))
    out = smooth_points(points, window=5)
    assert out[0] == points[0]
    assert out[-1] == points[-1]


def test_smooth_points_two_point_stroke_unchanged():
    # Regression: an un-pinned moving average collapsed a minimal 2-point
    # stroke to its midpoint, freezing a dot instead of a short line.
    points = ((0.2, 0.3), (0.8, 0.9))
    assert smooth_points(points) == points


def test_smooth_points_too_few_points_unchanged():
    assert smooth_points(((0.1, 0.2),)) == ((0.1, 0.2),)
    assert smooth_points(()) == ()


def test_smooth_points_window_one_is_identity():
    points = ((0.0, 0.0), (1.0, 1.0))
    assert smooth_points(points, window=1) == points


# --------------------------------------------------------------------------- #
# rasterize_stroke (bbox, non-empty pixels, padding, color)
# --------------------------------------------------------------------------- #

def test_rasterize_bbox_includes_padding_and_half_width():
    points = ((0.25, 0.25), (0.75, 0.75))
    color = (0, 0, 255)  # red (BGR)
    bgra, bbox = rasterize_stroke(points, (100, 100), color, width=6, padding=8)
    # margin = padding(8) + ceil(width/2)=3 -> 11 on each side.
    assert bbox == (14, 14, 72, 72)
    assert bgra.shape == (72, 72, 4)


def test_rasterize_produces_non_empty_opaque_pixels_in_stroke_color():
    color = (0, 0, 255)
    bgra, _bbox = rasterize_stroke(
        ((0.25, 0.25), (0.75, 0.75)), (100, 100), color, width=6, padding=8
    )
    alpha = bgra[:, :, 3]
    assert alpha.max() > 0  # the line was actually drawn
    # Every drawn pixel carries the stroke color (AA edges included).
    drawn = alpha > 0
    assert np.all(bgra[drawn, 0] == color[0])
    assert np.all(bgra[drawn, 1] == color[1])
    assert np.all(bgra[drawn, 2] == color[2])


def test_rasterize_padding_border_is_transparent():
    bgra, _bbox = rasterize_stroke(
        ((0.25, 0.25), (0.75, 0.75)), (100, 100), (0, 0, 255), width=6, padding=8
    )
    alpha = bgra[:, :, 3]
    # The outermost rows/cols sit inside the transparent padding margin.
    assert alpha[0, :].sum() == 0
    assert alpha[-1, :].sum() == 0
    assert alpha[:, 0].sum() == 0
    assert alpha[:, -1].sum() == 0


def test_rasterize_empty_points_is_1x1_transparent():
    bgra, bbox = rasterize_stroke((), (100, 100), (0, 0, 255), width=6)
    assert bgra.shape == (1, 1, 4)
    assert bbox == (0, 0, 1, 1)
    assert bgra[:, :, 3].sum() == 0


def test_rasterize_single_point_draws_a_dot():
    bgra, bbox = rasterize_stroke(((0.5, 0.5),), (100, 100), (255, 0, 0), width=8, padding=4)
    assert bgra[:, :, 3].max() > 0  # a dot is drawn, not empty


# --------------------------------------------------------------------------- #
# bbox_to_transform (normalized bbox -> layer placement)
# --------------------------------------------------------------------------- #

def test_bbox_to_transform_centers_on_bbox_center():
    transform = bbox_to_transform((10, 20, 40, 60), (100, 100))
    assert transform.center_x == pytest.approx(0.30)  # (10 + 20) / 100
    assert transform.center_y == pytest.approx(0.50)  # (20 + 30) / 100
    assert transform.scale == DEFAULT_SCALE


def test_bbox_to_transform_round_trips_through_compositor_top_left():
    # At scale 1.0 the composited image's top-left must land back at (x0, y0),
    # i.e. exactly where the stroke was rasterized -> "appears where drawn".
    bbox = (14, 14, 72, 72)
    frame_size = (100, 100)
    transform = bbox_to_transform(bbox, frame_size)
    x0, y0, w, h = bbox
    assert top_left_px(transform, (w, h), frame_size) == (x0, y0)


# --------------------------------------------------------------------------- #
# Palette + width helpers (the 'k' and '[' / ']' keys)
# --------------------------------------------------------------------------- #

def test_next_color_index_wraps_around():
    assert next_color_index(0) == 1
    assert next_color_index(len(STROKE_PALETTE) - 1) == 0


def test_clamped_width_bounds():
    assert clamped_width(MIN_STROKE_WIDTH - 5) == MIN_STROKE_WIDTH
    assert clamped_width(MAX_STROKE_WIDTH + 5) == MAX_STROKE_WIDTH
    assert clamped_width(DEFAULT_STROKE_WIDTH) == DEFAULT_STROKE_WIDTH


# --------------------------------------------------------------------------- #
# Overlay drawing helpers (smoke: mutate the frame, no raise)
# --------------------------------------------------------------------------- #

def test_draw_stroke_overlay_marks_pixels():
    frame = np.zeros((50, 50, 3), dtype=np.uint8)
    draw_stroke_overlay(frame, ((0.2, 0.5), (0.8, 0.5)), (0, 255, 0), width=3)
    assert frame.sum() > 0


def test_draw_stroke_overlay_empty_is_noop():
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    draw_stroke_overlay(frame, (), (0, 255, 0), width=3)
    assert frame.sum() == 0


def test_draw_cursor_marks_pixels():
    frame = np.zeros((50, 50, 3), dtype=np.uint8)
    draw_cursor(frame, (0.5, 0.5), (255, 255, 255))
    assert frame.sum() > 0


# --------------------------------------------------------------------------- #
# Multi-stroke: rasterize_strokes + freeze_draw_state + draw_live_state
# --------------------------------------------------------------------------- #

def test_rasterize_strokes_combined_bbox_covers_all_strokes():
    strokes = (((0.1, 0.1), (0.2, 0.2)), ((0.7, 0.7), (0.8, 0.8)))
    result = rasterize_strokes(strokes, (100, 100), (0, 0, 255), width=4, padding=2)
    assert result is not None
    bgra, (x0, y0, w, h) = result
    # The bbox must span from near the first stroke's min to the last's max.
    assert x0 < 10 and y0 < 10
    assert x0 + w > 80 and y0 + h > 80
    assert bgra[:, :, 3].max() > 0  # both strokes drew ink


def test_rasterize_strokes_empty_returns_none():
    assert rasterize_strokes((), (100, 100), (0, 0, 255), width=4) is None
    assert rasterize_strokes(((),), (100, 100), (0, 0, 255), width=4) is None


def test_freeze_draw_state_two_strokes_into_one_layer():
    state = (
        DrawState()
        .pen_move((0.2, 0.2)).pen_move((0.3, 0.3)).pen_up()
        .pen_move((0.6, 0.6)).pen_move((0.7, 0.7))
    )
    frozen = freeze_draw_state(state, (100, 100))
    assert frozen is not None
    bgra, transform = frozen
    assert bgra.shape[2] == 4  # BGRA
    assert transform.scale == DEFAULT_SCALE
    # Center sits between the two strokes' combined extent.
    assert 0.3 < transform.center_x < 0.7


def test_freeze_draw_state_nothing_drawn_is_none():
    assert freeze_draw_state(DrawState(), (100, 100)) is None
    # A lone tap (below MIN_STROKE_POINTS) also yields nothing to freeze.
    assert freeze_draw_state(DrawState().pen_move((0.5, 0.5)), (100, 100)) is None


def test_freeze_draw_state_round_trips_placement():
    # The frozen layer lands where it was drawn (same guarantee as bbox_to_transform).
    state = DrawState().pen_move((0.4, 0.4)).pen_move((0.6, 0.6))
    frozen = freeze_draw_state(state, (100, 100))
    assert frozen is not None
    bgra, transform = frozen
    h, w = bgra.shape[:2]
    x0, y0 = top_left_px(transform, (w, h), (100, 100))
    # The drawn points (40..60 px) sit inside the placed layer's rect.
    assert x0 <= 40 and y0 <= 40
    assert x0 + w >= 60 and y0 + h >= 60


def test_draw_live_state_marks_frame_and_empty_is_noop():
    frame = np.zeros((50, 50, 3), dtype=np.uint8)
    draw_live_state(frame, DrawState())  # nothing drawn
    assert frame.sum() == 0
    active = DrawState().pen_move((0.2, 0.5)).pen_move((0.8, 0.5))
    draw_live_state(frame, active)
    assert frame.sum() > 0  # ink + cursor
