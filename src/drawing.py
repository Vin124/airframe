"""Air-drawing: the stroke model, smoothing, rasterization, and live overlay.

Pure and dependency-light (numpy + cv2 only -- no camera, MediaPipe, or
threads), so every helper here is unit-testable headlessly. The MAIN loop
(``PipelineRuntime``) owns the accumulating ``Stroke`` -- it is the single
writer of scene/interaction -- and calls into this module to smooth, render, and
finally rasterize a released stroke into a BGRA image that becomes a normal
scene ``Layer``. Points stay normalized 0-1 everywhere; pixel conversion happens
only at draw/rasterize time, per the project convention.

The one non-``numpy``/``cv2`` import is ``scene.Transform`` (itself a pure,
dependency-light frozen dataclass): ``bbox_to_transform`` maps a rasterized
stroke's pixel bounding box to the layer transform that places it exactly where
it was drawn, and returning a ``Transform`` keeps that placement math in one
testable spot instead of leaking center/scale tuples up to the caller.
"""

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from src.scene import DEFAULT_SCALE, Size, Transform

Point2D = Tuple[float, float]

# Palette of BGR stroke colors cycled by the 'k' key -- bright, well-separated
# hues that read clearly over a webcam feed. Index 0 is the launch default.
STROKE_PALETTE: Tuple[Tuple[int, int, int], ...] = (
    (0, 0, 255),      # red
    (0, 255, 0),      # green
    (255, 0, 0),      # blue
    (0, 255, 255),    # yellow
    (255, 0, 255),    # magenta
    (255, 255, 255),  # white
)

# Stroke line width (px) bounds + the '[' / ']' step. DEFAULT is the launch value.
MIN_STROKE_WIDTH = 2
MAX_STROKE_WIDTH = 40
DEFAULT_STROKE_WIDTH = 6
STROKE_WIDTH_STEP = 2

# Transparent margin (px) added around the stroke's extent when rasterizing, so a
# thick antialiased line isn't clipped at the layer image's edges. The line's own
# half-width is added on top of this (see ``rasterize_stroke``).
RASTER_PADDING_PX = 8

# A stroke needs at least this many points before it can be smoothed / rasterized
# / frozen into a layer (a single tap isn't a stroke).
MIN_STROKE_POINTS = 2

# Centered moving-average window (odd) used to smooth the jittery fingertip path.
SMOOTHING_WINDOW = 5

# Radius of the fingertip cursor dot drawn while a stroke is in progress.
CURSOR_RADIUS_PX = 6

# Air-drawing runtime keys (Phase 11, image mode only): 'c' discards the
# in-progress stroke, 'k' cycles the stroke color through STROKE_PALETTE, and
# '['/']' step the stroke width down/up (clamped). Chosen to avoid the keys
# already bound in main.py (q/s/r/+/-/v/a/w/m/x/f).
CLEAR_STROKE_KEY = ord("c")
CYCLE_COLOR_KEY = ord("k")
STROKE_WIDTH_DOWN_KEY = ord("[")
STROKE_WIDTH_UP_KEY = ord("]")


@dataclass(frozen=True)
class Stroke:
    """An immutable air-drawing stroke: an ordered tuple of normalized points.

    Functional updates only (``append`` returns a new ``Stroke``), matching the
    project's immutability rule so the main loop never mutates shared state.
    """

    points: Tuple[Point2D, ...] = ()

    def append(self, point: Point2D) -> "Stroke":
        """Return a new stroke with ``point`` added to the end."""
        return Stroke(points=self.points + ((float(point[0]), float(point[1])),))

    @property
    def is_empty(self) -> bool:
        """True when the stroke has no points yet."""
        return len(self.points) == 0

    def __len__(self) -> int:
        return len(self.points)


def smooth_points(
    points: Tuple[Point2D, ...], window: int = SMOOTHING_WINDOW
) -> Tuple[Point2D, ...]:
    """Centered moving-average smoothing over a normalized point sequence.

    Interior points become the mean of the up-to-``window`` input points
    centered on them (clamped near the ends); the FIRST and LAST points are
    pinned unchanged so the stroke starts and ends exactly at the fingertip
    (an un-pinned average pulls the endpoints inward -- a 2-point stroke would
    collapse to its midpoint). Fewer than three points (or a window <= 1) are
    returned unchanged.
    """
    count = len(points)
    if count < 3 or window <= 1:
        return tuple(points)
    radius = window // 2
    smoothed: List[Point2D] = [points[0]]
    for i in range(1, count - 1):
        lo = max(0, i - radius)
        hi = min(count, i + radius + 1)
        span = points[lo:hi]
        mean_x = sum(p[0] for p in span) / len(span)
        mean_y = sum(p[1] for p in span) / len(span)
        smoothed.append((mean_x, mean_y))
    smoothed.append(points[-1])
    return tuple(smoothed)


def next_color_index(index: int) -> int:
    """Return the next palette index, wrapping around (for the 'k' key)."""
    return (index + 1) % len(STROKE_PALETTE)


def clamped_width(width: int) -> int:
    """Clamp a stroke width into ``[MIN_STROKE_WIDTH, MAX_STROKE_WIDTH]``."""
    return max(MIN_STROKE_WIDTH, min(MAX_STROKE_WIDTH, int(width)))


def _to_pixels(points: Tuple[Point2D, ...], frame_size: Size) -> List[Point2D]:
    """Map normalized points into frame-pixel float coordinates."""
    frame_w, frame_h = frame_size
    return [(x * frame_w, y * frame_h) for x, y in points]


def rasterize_stroke(
    points: Tuple[Point2D, ...],
    frame_size: Size,
    color: Tuple[int, int, int],
    width: int,
    padding: int = RASTER_PADDING_PX,
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """Render ``points`` to a BGRA layer image cropped to its bounding box.

    Returns ``(bgra, bbox)`` where ``bgra`` is an ``HxWx4`` array (transparent
    except along the stroke) and ``bbox`` is ``(x0, y0, w, h)`` in FRAME pixel
    coordinates -- the rectangle the image occupies on the frame, which
    ``bbox_to_transform`` turns into the layer's placement. The box extends a
    ``margin`` (``padding`` plus the line's half-width) beyond the extreme point
    centers on every side so a thick line is never clipped; ``x0``/``y0`` may be
    negative when the stroke hugs the top-left edge (the compositor clips).

    Alpha follows the drawn (antialiased) line so the frozen layer blends cleanly
    over the scene. An empty point list yields a 1x1 transparent image.
    """
    if not points:
        return np.zeros((1, 1, 4), dtype=np.uint8), (0, 0, 1, 1)

    pixels = _to_pixels(points, frame_size)
    xs = [px for px, _ in pixels]
    ys = [py for _, py in pixels]
    margin = padding + (width + 1) // 2
    x0 = int(math.floor(min(xs))) - margin
    y0 = int(math.floor(min(ys))) - margin
    x1 = int(math.ceil(max(xs))) + margin
    y1 = int(math.ceil(max(ys))) + margin
    img_w = max(1, x1 - x0)
    img_h = max(1, y1 - y0)

    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    local = np.array(
        [[int(round(px - x0)), int(round(py - y0))] for px, py in pixels],
        dtype=np.int32,
    )
    if len(local) == 1:
        cv2.circle(mask, tuple(local[0]), max(1, width // 2), 255, -1, cv2.LINE_AA)
    else:
        cv2.polylines(mask, [local], False, 255, int(width), cv2.LINE_AA)

    bgra = np.zeros((img_h, img_w, 4), dtype=np.uint8)
    drawn = mask > 0
    for channel in range(3):
        bgra[drawn, channel] = color[channel]
    bgra[:, :, 3] = mask
    return bgra, (x0, y0, img_w, img_h)


def bbox_to_transform(
    bbox: Tuple[int, int, int, int], frame_size: Size
) -> Transform:
    """Map a rasterized stroke's pixel ``bbox`` to its placement ``Transform``.

    The rasterized image is the layer's native size, so the layer is placed at
    ``DEFAULT_SCALE`` (1.0) with its center at the bounding box's center. At
    scale 1.0 the compositor's ``top_left_px`` puts the image's top-left back at
    ``(x0, y0)``, i.e. exactly where the stroke was drawn. ``Transform`` clamps
    the center into the frame, so a stroke drawn partly off-frame is nudged just
    enough to keep part of it visible -- consistent with every other layer.
    """
    x0, y0, w, h = bbox
    frame_w, frame_h = frame_size
    center_x = (x0 + w / 2.0) / frame_w
    center_y = (y0 + h / 2.0) / frame_h
    return Transform(center_x, center_y, DEFAULT_SCALE)


def freeze_stroke(
    points: Tuple[Point2D, ...],
    frame_size: Size,
    color: Tuple[int, int, int],
    width: int,
) -> Optional[Tuple[np.ndarray, Transform]]:
    """Turn a released stroke into its frozen-layer ingredients, or ``None``.

    The pure half of freezing: smooth the path, rasterize it to a BGRA image at
    its bounding box, and map that box to the placement ``Transform`` that puts
    the layer exactly where the stroke was drawn. Returns ``None`` for a stroke
    with fewer than ``MIN_STROKE_POINTS`` points (a tap isn't a drawing). The
    caller wraps the image in a source and adds the scene layer.
    """
    if len(points) < MIN_STROKE_POINTS:
        return None
    smoothed = smooth_points(points)
    bgra, bbox = rasterize_stroke(smoothed, frame_size, color, width)
    return bgra, bbox_to_transform(bbox, frame_size)


def draw_live_stroke(
    frame: np.ndarray,
    points: Tuple[Point2D, ...],
    color: Tuple[int, int, int],
    width: int,
) -> None:
    """Draw the in-progress stroke (smoothed) + its fingertip cursor in place."""
    smoothed = smooth_points(points)
    draw_stroke_overlay(frame, smoothed, color, width)
    if smoothed:
        draw_cursor(frame, smoothed[-1], color)


def draw_stroke_overlay(
    frame: np.ndarray,
    points: Tuple[Point2D, ...],
    color: Tuple[int, int, int],
    width: int,
) -> None:
    """Draw an in-progress stroke's polyline onto ``frame`` in place.

    Normalized ``points`` are converted to pixels here. A single point renders
    as a dot; two or more render as an antialiased polyline. A no-op for an
    empty point list.
    """
    if not points:
        return
    frame_h, frame_w = frame.shape[:2]
    local = np.array(
        [[int(round(x * frame_w)), int(round(y * frame_h))] for x, y in points],
        dtype=np.int32,
    )
    if len(local) == 1:
        cv2.circle(frame, tuple(local[0]), max(1, width // 2), color, -1, cv2.LINE_AA)
    else:
        cv2.polylines(frame, [local], False, color, int(width), cv2.LINE_AA)


def draw_cursor(
    frame: np.ndarray,
    point: Point2D,
    color: Tuple[int, int, int],
    radius: int = CURSOR_RADIUS_PX,
) -> None:
    """Draw a small filled fingertip-cursor dot at ``point`` (normalized), in place."""
    frame_h, frame_w = frame.shape[:2]
    center = (int(round(point[0] * frame_w)), int(round(point[1] * frame_h)))
    cv2.circle(frame, center, radius, color, -1, cv2.LINE_AA)
