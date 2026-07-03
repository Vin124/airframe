"""Unit tests for src.drawing: the pure air-drawing model, smoothing, and
rasterization. Headless -- numpy/cv2 only, no camera/MediaPipe/display."""

import numpy as np
import pytest

from src.drawing import (
    DEFAULT_STROKE_WIDTH,
    MAX_STROKE_WIDTH,
    MIN_STROKE_WIDTH,
    STROKE_PALETTE,
    Stroke,
    bbox_to_transform,
    clamped_width,
    draw_cursor,
    draw_stroke_overlay,
    next_color_index,
    rasterize_stroke,
    smooth_points,
)
from src.scene import DEFAULT_SCALE, top_left_px


# --------------------------------------------------------------------------- #
# Stroke model (immutable + functional append)
# --------------------------------------------------------------------------- #

def test_stroke_defaults_empty():
    stroke = Stroke()
    assert stroke.is_empty is True
    assert len(stroke) == 0


def test_stroke_append_returns_new_stroke_without_mutating():
    original = Stroke()
    appended = original.append((0.1, 0.2))
    assert original.points == ()  # unchanged
    assert appended.points == ((0.1, 0.2),)
    assert len(appended) == 1


def test_stroke_append_chains_points_in_order():
    stroke = Stroke().append((0.1, 0.2)).append((0.3, 0.4))
    assert stroke.points == ((0.1, 0.2), (0.3, 0.4))
    assert len(stroke) == 2


def test_stroke_append_coerces_to_float():
    stroke = Stroke().append((1, 0))  # ints in
    assert stroke.points == ((1.0, 0.0),)
    assert isinstance(stroke.points[0][0], float)


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
